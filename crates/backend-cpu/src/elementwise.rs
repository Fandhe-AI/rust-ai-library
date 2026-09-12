//! elementwise カーネル（二項演算 `add`・`mul`、活性化 `relu`・`exp`・`tanh`。
//! TASK-1.6b・#22）。
//!
//! 演算セットは PoC-v2-2／PoC-v2-5 実測済みセット（`docs/public-api-design.md`
//! §3.2・§4.2 `BackendOps`）に合わせる。`BackendOps` トレイト・`DeviceBuffer`
//! への接続は TASK-1.9（#43）のスコープであり、本モジュールはそこから後で
//! 呼び出せる関数ベースのカーネル API のみを提供する（トレイト定義はしない）。
//! `autodiff` の演算入口（#16〜#18）からも同様に呼ばれる想定。
//!
//! # 2 層構成
//!
//! 1. **スライスカーネル層**（`*_slice` 関数）: `tensor-core` に依存しない
//!    低レベル層。両オペランドが contiguous であることを呼び出し元が
//!    保証した上で呼ぶ契約とする（PoC-v2-5 `cpu_ref.rs` の参照実装
//!    〈`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/cpu_ref.rs`〉
//!    を productize したもの）。
//! 2. **Tensor 入口層**（`add`/`mul`/`relu`/`exp`/`tanh`）: `Tensor<f32>` を
//!    受け取り `elementwise_out_shape`（NumPy 互換ブロードキャスト。
//!    `tensor-core::broadcast_shape` 委譲）で出力 shape を確定し、
//!    contiguous な場合はスライスカーネルへ直行（fast path）、
//!    非 contiguous（ブロードキャスト view・transpose 後 view 等）な場合は
//!    strided 反復で読む（general path）。
//!
//! # 並列化（rayon）
//!
//! 二項演算・活性化はいずれも要素ごとに独立な map 演算であり、演算順序が
//! 結果に影響しない（浮動小数点の結合則非成立は要素間で加算・乗算を跨がない
//! elementwise 演算には現れないため無関係）。このため rayon による並列化は
//! 数値へ影響しない。小サイズ入力での rayon オーバーヘッド（タスク分割・
//! 同期コスト）を避けるため、要素数が [`PARALLEL_THRESHOLD`] 未満の場合は
//! 逐次実行にフォールバックする。閾値はベンチ根拠が出るまでの保守的な
//! 固定値であり、チューニングは #24（TASK-1.6d）のスコープとする。
//!
//! # 数値契約（REQ-2・`.claude/rules/coding-rust.md`）
//!
//! - `add`/`mul`/`relu` は加減乗算・比較のみで libm を経由しない（PoC-v2-5 で
//!   backend 間 bit 一致を実測済みの区分）。
//! - `exp`/`tanh` は `f32::exp`/`f32::tanh`（libm）を使用する。GPU との数値
//!   突合は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の
//!   対象だが、これは TASK-2.2 のスコープであり本モジュールの単体テストは
//!   CPU 単体の厳密一致（`f32::exp`/`f32::tanh` の直接計算値との一致）まで
//!   とする。
//! - FMA 契約（`f32::mul_add`）は積和演算（GEMM。#21）の契約であり、
//!   elementwise 演算には積和（1 回の乗算結果を別の加算に畳み込む処理）が
//!   現れないため適用外とする。
//! - `relu` は `f32::max`（`x.max(0.0)`）を用いるため、Rust の `f32::max`
//!   仕様どおり NaN 入力を無視し `relu(NaN) == 0.0` を返す（NumPy の
//!   `maximum(x, 0)`・PyTorch の `relu` は NaN を伝播するため異なる）。本
//!   モジュールは CPU 参照実装候補であり、GPU バックエンド（CUDA・Metal）が
//!   NaN 伝播する実装を採る場合は数値一致判定（TASK-2.2）で不一致となりうる
//!   ため、GPU 側実装時に本挙動との整合を確認すること。
//!
//! # 境界検査（REQ-8）
//!
//! 添字アクセスは Rust の境界検査付きインデックス（スライスの `[]`・
//! `Tensor::get`）のみを用い、`get_unchecked` 等の unchecked アクセスや
//! `unsafe` は使わない（本モジュールに `unsafe` は不要）。strided 反復の
//! offset 計算は `Tensor::get` 内部で `offset + Σ idx[i] * strides[i]` を
//! `isize` で行う契約（`tensor.rs`）に委ねる。現行 `tensor-core` は負
//! stride を生成しないため、本モジュールは非負 stride のみを前提とする。

use fandhe_ai_tensor_core::{Element, ShapeError, Tensor, elementwise_out_shape};
use rayon::prelude::*;

/// rayon 並列化へ切り替える要素数の下限。
///
/// 小サイズ入力ではタスク分割・スレッド同期のオーバーヘッドが逐次実行の
/// 利得を上回るため、この値未満は逐次実行にフォールバックする
/// （モジュール doc コメント「並列化」参照。チューニングは #24 のスコープ）。
///
/// `pub(crate)`: TASK-12.1c（#163）の `fused_elementwise` モジュールが
/// 同じ閾値・同じ理由（per-op 演算と同じ要素独立 map 演算のため）で
/// 逐次／並列を切り替える際に再利用する（マジックナンバーの重複を避ける）。
pub(crate) const PARALLEL_THRESHOLD: usize = 1 << 15;

// --- スライスカーネル層 ---
//
// 全関数共通の契約: `a`（・`b`）と `out` は同じ長さであることを呼び出し元
// （Tensor 入口層の fast path）が事前に保証する。この層は `tensor-core` に
// 依存せず contiguous な `&[f32]` のみを扱う（TASK-1.9 で `BackendOps` から
// 直接再利用できるようにするための分離）。
//
// 長さ不一致は `assert_eq!`（release ビルドでも有効）で検査する。これらは
// pub 関数であり TASK-1.9（#43）で `BackendOps` から直接再利用される想定
// のため、契約違反時に `zip` が黙って短い方へ切り詰めて誤った結果を返す
// （release ビルドで `debug_assert_eq!` は消える）事態を避ける。

/// 要素ごとの加算（`out[i] = a[i] + b[i]`）。加減算のみで libm 非経由。
pub fn add_slice(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "add_slice: length mismatch (a vs b)");
    assert_eq!(a.len(), out.len(), "add_slice: length mismatch (a vs out)");
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|((o, &x), &y)| *o = x + y);
    } else {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
    }
}

/// 要素ごとの乗算（`out[i] = a[i] * b[i]`）。乗算のみで libm 非経由。
pub fn mul_slice(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "mul_slice: length mismatch (a vs b)");
    assert_eq!(a.len(), out.len(), "mul_slice: length mismatch (a vs out)");
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|((o, &x), &y)| *o = x * y);
    } else {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
    }
}

/// 要素ごとの ReLU（`out[i] = a[i].max(0.0)`）。比較のみで libm 非経由。
pub fn relu_slice(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len(), "relu_slice: length mismatch");
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = x.max(0.0));
    } else {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = x.max(0.0);
        }
    }
}

/// 要素ごとの `exp`（`f32::exp`。libm 経由。モジュール doc コメント
/// 「数値契約」参照）。
pub fn exp_slice(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len(), "exp_slice: length mismatch");
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = x.exp());
    } else {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = x.exp();
        }
    }
}

/// 要素ごとの `tanh`（`f32::tanh`。libm 経由。モジュール doc コメント
/// 「数値契約」参照）。
pub fn tanh_slice(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len(), "tanh_slice: length mismatch");
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = x.tanh());
    } else {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = x.tanh();
        }
    }
}

/// 条件テンソルによる要素選択（`torch.where` 相当。イシュー #1637）。
/// 真偽判定は `c != 0.0`（`BackendOps::where_cond` doc の契約と同一）。
/// `cond`／`a`／`b` は同長であること（`Tensor` 入口 [`where_cond`]
/// が shape 一致を検査済み）。
pub fn where_slice(cond: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(
        cond.len(),
        a.len(),
        "where_slice: length mismatch (cond vs a)"
    );
    assert_eq!(
        cond.len(),
        b.len(),
        "where_slice: length mismatch (cond vs b)"
    );
    assert_eq!(
        cond.len(),
        out.len(),
        "where_slice: length mismatch (cond vs out)"
    );
    if cond.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(cond.par_iter())
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|(((o, &c), &av), &bv)| *o = if c != 0.0 { av } else { bv });
    } else {
        for (((o, &c), &av), &bv) in out.iter_mut().zip(cond).zip(a).zip(b) {
            *o = if c != 0.0 { av } else { bv };
        }
    }
}

/// マスク位置を定数で置換する（`torch.masked_fill` 相当。イシュー
/// #1637）。真偽判定は `m != 0.0`（`BackendOps::masked_fill` doc の
/// 契約と同一）。`x`／`mask` は同長であること。
pub fn masked_fill_slice(x: &[f32], mask: &[f32], value: f32, out: &mut [f32]) {
    assert_eq!(
        x.len(),
        mask.len(),
        "masked_fill_slice: length mismatch (x vs mask)"
    );
    assert_eq!(
        x.len(),
        out.len(),
        "masked_fill_slice: length mismatch (x vs out)"
    );
    if x.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(x.par_iter())
            .zip(mask.par_iter())
            .for_each(|((o, &xv), &mv)| *o = if mv != 0.0 { value } else { xv });
    } else {
        for ((o, &xv), &mv) in out.iter_mut().zip(x).zip(mask) {
            *o = if mv != 0.0 { value } else { xv };
        }
    }
}

// --- ベンチ専用: 逐次／並列強制版（`#[cfg(test)]` 限定） ---
//
// `add_slice`／`mul_slice`／`exp_slice` は `PARALLEL_THRESHOLD` による
// 自動判定を内蔵するため、同一要素数で逐次経路と並列経路を比較できない
// （codex-review 指摘・イシュー #1027）。下記モジュールは判定を経由せず
// 経路を固定するベンチ専用版と、その同一サイズ比較スイープ（`#[ignore]`）を
// 提供する。
//
// `#[cfg(test)]` の private モジュールに限定する理由（PR #1066 codex-review
// P1 再指摘対応）: 当初の `#[doc(hidden)] pub` 方式は rustdoc 表示を隠す
// だけで可視性・semver 上の公開性は変わらず、release ビルドで
// `debug_assert_eq!` が消えた状態の関数（長さ不一致を `zip` が黙って短い方へ
// 切り詰める、本番 `*_slice` の release でも有効な `assert_eq!` 契約と異なる
// 面）を外部クレートが呼べてしまう。`#[cfg(test)]` 化により通常ビルド・
// 公開面から完全に消え、本番 panic 経路も外部到達可能な異契約面も残らない。

#[cfg(test)]
mod bench_internal {
    //! `PARALLEL_THRESHOLD` 判定を除いた逐次／並列強制版カーネルと、その
    //! 同一サイズ直列 vs 並列比較スイープ（イシュー #1027）。強制版の本体
    //! ロジックはスライスカーネル層と同一・分岐のみ除去（`gemm_blis`
    //! 〈直列専用入口〉／`gemm_blis_parallel`〈並列専用入口〉が別関数で
    //! あるのと同じ発想）で、本番の `*_slice` 関数の閾値判定・数値契約には
    //! 影響しない。自動判定版（公開 API の `add_slice` 等）のスイープは
    //! `tests/elementwise_threshold_perf.rs`（integration test）を参照。
    //!
    //! 長さ検査は `debug_assert_eq!` に留める（`#[cfg(test)]` 限定のため
    //! 本番経路には存在せず、テストビルドでは debug assertion が有効）。
    //!
    //! スイープは `#[ignore]` のため通常の `cargo test` では実行されず、
    //! テスト全体の実行時間へ影響しない。実行例:
    //! ```text
    //! cargo test -p fandhe-ai-backend-cpu --release -- --ignored elementwise_serial_vs_parallel_sweep
    //! ```

    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run};
    use rayon::prelude::*;
    use std::hint::black_box;

    /// `add_slice` の本体ロジックから `PARALLEL_THRESHOLD` 判定を除いた
    /// 逐次強制版（ベンチ専用。モジュール doc コメント参照）。
    fn add_slice_force_serial(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len(), "add_slice_force_serial: a vs b");
        debug_assert_eq!(a.len(), out.len(), "add_slice_force_serial: a vs out");
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
    }

    /// `add_slice` の本体ロジックから `PARALLEL_THRESHOLD` 判定を除いた
    /// 並列強制版（ベンチ専用。モジュール doc コメント参照）。
    fn add_slice_force_parallel(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len(), "add_slice_force_parallel: a vs b");
        debug_assert_eq!(a.len(), out.len(), "add_slice_force_parallel: a vs out");
        out.par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|((o, &x), &y)| *o = x + y);
    }

    /// `mul_slice` の逐次強制版（ベンチ専用。モジュール doc コメント参照）。
    fn mul_slice_force_serial(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len(), "mul_slice_force_serial: a vs b");
        debug_assert_eq!(a.len(), out.len(), "mul_slice_force_serial: a vs out");
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
    }

    /// `mul_slice` の並列強制版（ベンチ専用。モジュール doc コメント参照）。
    fn mul_slice_force_parallel(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len(), "mul_slice_force_parallel: a vs b");
        debug_assert_eq!(a.len(), out.len(), "mul_slice_force_parallel: a vs out");
        out.par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|((o, &x), &y)| *o = x * y);
    }

    /// `exp_slice` の逐次強制版（ベンチ専用。モジュール doc コメント参照）。
    fn exp_slice_force_serial(a: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), out.len(), "exp_slice_force_serial: a vs out");
        for (o, &x) in out.iter_mut().zip(a) {
            *o = x.exp();
        }
    }

    /// `exp_slice` の並列強制版（ベンチ専用。モジュール doc コメント参照）。
    fn exp_slice_force_parallel(a: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), out.len(), "exp_slice_force_parallel: a vs out");
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = x.exp());
    }

    fn random_vec(seed: u64, len: usize) -> Vec<f32> {
        Xorshift64Star::new(seed).fill_vec(len)
    }

    /// 同一要素数で逐次強制版・並列強制版を測り、中央値比（並列 / 逐次。
    /// 1.0 未満＝並列が有利）を報告する。サイズを固定するため、要素数増加の
    /// 影響を排除して並列化オーバーヘッドそのものを捉える。
    ///
    /// `run` はクロージャ呼び出し自体を `black_box` するのみで、クロージャ
    /// 内部の入出力までは保護しない（`bench_harness::protocol::run`
    /// ドキュメンテーションコメント参照）。release/LTO で入力・出力バッファが
    /// 定数畳み込み・dead code 除去の対象にならないよう、計測対象の入出力を
    /// 明示的に `black_box` で保護する（codex-review 指摘・イシュー #1027）。
    fn measure_size_serial_vs_parallel(label: &str, n: usize) {
        let a = random_vec(7000 + n as u64, n);
        let b = random_vec(8000 + n as u64, n);
        let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

        macro_rules! measure_pair {
            ($name:literal, $serial_fn:expr, $parallel_fn:expr, $out_len:expr) => {{
                let mut out_s = vec![0.0f32; $out_len];
                let serial = run(&config, || {
                    $serial_fn(black_box(&a), black_box(&b), black_box(&mut out_s));
                })
                .expect(concat!($name, " 逐次強制版の計測に失敗"));
                black_box(&out_s);

                let mut out_p = vec![0.0f32; $out_len];
                let parallel = run(&config, || {
                    $parallel_fn(black_box(&a), black_box(&b), black_box(&mut out_p));
                })
                .expect(concat!($name, " 並列強制版の計測に失敗"));
                black_box(&out_p);

                (serial.median_secs, parallel.median_secs)
            }};
        }

        let (add_serial_secs, add_parallel_secs) = measure_pair!(
            "add_slice",
            add_slice_force_serial,
            add_slice_force_parallel,
            n
        );
        let (mul_serial_secs, mul_parallel_secs) = measure_pair!(
            "mul_slice",
            mul_slice_force_serial,
            mul_slice_force_parallel,
            n
        );

        // `exp_slice_force_{serial,parallel}` は単項（`b` を使わない）のため
        // 上記マクロの二項シグネチャに合わせられず個別に計測する。
        let mut out_exp_s = vec![0.0f32; n];
        let exp_serial = run(&config, || {
            exp_slice_force_serial(black_box(&a), black_box(&mut out_exp_s));
        })
        .expect("exp_slice 逐次強制版の計測に失敗");
        black_box(&out_exp_s);

        let mut out_exp_p = vec![0.0f32; n];
        let exp_parallel = run(&config, || {
            exp_slice_force_parallel(black_box(&a), black_box(&mut out_exp_p));
        })
        .expect("exp_slice 並列強制版の計測に失敗");
        black_box(&out_exp_p);

        println!(
            "label={label} n={n} \
             add_serial_ns={as_ns:.1} add_parallel_ns={ap_ns:.1} add_parallel_ratio={ar:.3} \
             mul_serial_ns={ms_ns:.1} mul_parallel_ns={mp_ns:.1} mul_parallel_ratio={mr:.3} \
             exp_serial_ns={es_ns:.1} exp_parallel_ns={ep_ns:.1} exp_parallel_ratio={er:.3}",
            as_ns = add_serial_secs * 1e9,
            ap_ns = add_parallel_secs * 1e9,
            ar = add_parallel_secs / add_serial_secs,
            ms_ns = mul_serial_secs * 1e9,
            mp_ns = mul_parallel_secs * 1e9,
            mr = mul_parallel_secs / mul_serial_secs,
            es_ns = exp_serial.median_secs * 1e9,
            ep_ns = exp_parallel.median_secs * 1e9,
            er = exp_parallel.median_secs / exp_serial.median_secs,
        );
    }

    /// [`measure_size_serial_vs_parallel`] による同一サイズ直列 vs 並列
    /// 比較のスイープ（codex-review 指摘・イシュー #1027）。
    /// `PARALLEL_THRESHOLD` を挟む要素数 2^12〜2^18 で、要素数増加の影響を
    /// 排除した並列化オーバーヘッドそのものの交差点を実測する。
    #[test]
    #[ignore = "実測ハーネス（--release 推奨）。通常 CI では実行しない"]
    fn elementwise_serial_vs_parallel_sweep() {
        for exp2 in 12..=18u32 {
            let n = 1usize << exp2;
            measure_size_serial_vs_parallel(&format!("2^{exp2}"), n);
        }
    }
}

// --- Tensor 入口層 ---

/// 行優先の多次元 index を 1 進める（最終軸から繰り上げ）。
///
/// general path（非 contiguous な strided 反復）の内部ヘルパー。`shape` の
/// 範囲内でのみ繰り上がるため、`Tensor::get` に渡す index は常に軸範囲内
/// になる（`tensor.rs` の `contiguous()` と同じ走査パターン）。
fn increment_index(index: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

/// [`binary_elementwise`] general path 用の 1 オペランド読み出し抽象
/// （イシュー #1583）。`autodiff::grad::MaskReadOperand`（#1577）と同型の
/// 設計で、`Tensor::get`（rank・範囲検査つきの要素ごとアクセス）を経由せず
/// 借用スライスへ直接インデックスする。`binary_elementwise` は呼び出し前に
/// 両オペランドを `broadcast_to(&out_shape)` 済みのため、ここでの `View`
/// stride は broadcast 拡張軸の stride 0 も含みうる。
enum ElementwiseReadOperand<'a> {
    Contig(&'a [f32]),
    View {
        span: &'a [f32],
        strides: Vec<usize>,
    },
}

impl<'a> ElementwiseReadOperand<'a> {
    /// `as_slice()`（真に contiguous）を最優先、次に `as_view_slice()`
    /// （非負 stride の view。broadcast 拡張軸の stride 0 を含む）を試す。
    /// いずれも失敗する場合（負 stride 等。現行公開 API の
    /// `transpose`/`narrow`/`broadcast_to` はいずれも負 stride を生成
    /// しないため到達しないが、将来拡張への fail-safe）は `None` を返し、
    /// 呼び出し元 [`binary_elementwise`] は `Tensor::get` 経由の既存経路
    /// へ経路全体を丸ごとフォールバックする（部分的に混在させない。
    /// `.claude/rules/security.md` A08 が禁じる判定迂回ではない）。
    fn classify(t: &'a Tensor<f32>) -> Option<Self> {
        if let Some(s) = t.as_slice() {
            return Some(Self::Contig(s));
        }
        let span = t.as_view_slice()?;
        let strides: Vec<usize> = t
            .strides()
            .iter()
            .map(|&s| usize::try_from(s).ok())
            .collect::<Option<_>>()?;
        Some(Self::View { span, strides })
    }

    /// `Contig` は `flat`（行優先の平坦 index）、`View` は `idx` と
    /// strides の内積で計算したオフセットで読む。`as_view_slice()` が
    /// 保証する `span`（`Tensor::as_view_slice` doc の
    /// `span = 1 + Σ (shape_i − 1)·stride_i`）により、`classify` 呼び出し
    /// 元が渡す `out_shape` 範囲内の `idx` では常に `Some` を返す
    /// （`MaskReadOperand::read` と同じ安全性の根拠）。
    #[inline]
    fn read(&self, idx: &[usize], flat: usize) -> Option<f32> {
        match self {
            Self::Contig(s) => s.get(flat).copied(),
            Self::View { span, strides } => {
                let mut off = 0usize;
                for (&i, &st) in idx.iter().zip(strides.iter()) {
                    off = off.checked_add(i.checked_mul(st)?)?;
                }
                span.get(off).copied()
            }
        }
    }
}

/// 二項 elementwise 演算の Tensor 入口共通処理。
///
/// `elementwise_out_shape`（`tensor-core::broadcast_shape` 委譲）で出力
/// shape を確定し、`Tensor::broadcast_to` で両オペランドを共通 shape の
/// view（拡張軸は stride 0 の zero-copy read）に揃える。両 view が
/// contiguous な場合（ブロードキャスト・view を伴わない同一 shape 入力）は
/// `slice_kernel` へ直行する。そうでない場合、両オペランドとも
/// [`ElementwiseReadOperand::classify`] が成功すれば stride 読み（`Tensor::
/// get` の rank・範囲検査を経由しない借用ベース読み。イシュー #1583。
/// `autodiff::grad::MaskReadOperand`〈#1577〉と同型）へ、失敗する場合の
/// み `Tensor::get` による strided 反復（変更前の経路）へフォールバックする
/// （`contiguous()` による事前実体化はメモリ倍増を招くため行わない。
/// 計画の設計方針）。
fn binary_elementwise(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    slice_kernel: fn(&[f32], &[f32], &mut [f32]),
    scalar_kernel: fn(f32, f32) -> f32,
) -> Result<Tensor<f32>, ShapeError> {
    let out_shape = elementwise_out_shape(a.shape(), b.shape())?;
    let ba = a.broadcast_to(&out_shape)?;
    let bb = b.broadcast_to(&out_shape)?;

    if let (Some(sa), Some(sb)) = (ba.as_slice(), bb.as_slice()) {
        // fast path: 両 view が contiguous（ブロードキャスト拡張軸を
        // 含まない・非 contiguous view でない）。
        let mut out = vec![0.0f32; sa.len()];
        slice_kernel(sa, sb, &mut out);
        return Tensor::new(out, &out_shape);
    }

    let numel = out_shape.iter().product::<usize>();
    let mut index = vec![0usize; out_shape.len()];

    if let (Some(a_op), Some(b_op)) = (
        ElementwiseReadOperand::classify(&ba),
        ElementwiseReadOperand::classify(&bb),
    ) {
        // stride 読み経路: ブロードキャスト拡張軸・非 contiguous view を
        // 含むが、両オペランドとも非負 stride（イシュー #1583）。
        let mut out = Vec::with_capacity(numel);
        for flat in 0..numel {
            let x = a_op.read(&index, flat);
            let y = b_op.read(&index, flat);
            debug_assert!(
                x.is_some() && y.is_some(),
                "binary_elementwise: as_view_slice の span 保証が破れ、境界外アクセスを検知した (index {index:?})"
            );
            out.push(scalar_kernel(
                x.unwrap_or_else(Element::zero),
                y.unwrap_or_else(Element::zero),
            ));
            increment_index(&mut index, &out_shape);
        }
        return Tensor::new(out, &out_shape);
    }

    // フォールバック経路: `Tensor::get` による strided 反復（変更前の
    // 経路と同一。`ElementwiseReadOperand::classify` が失敗した場合
    // （現行公開 API では到達しない）のみここに来る）。
    let mut out = Vec::with_capacity(numel);
    for _ in 0..numel {
        let x = ba.get(&index);
        let y = bb.get(&index);
        debug_assert!(
            x.is_some() && y.is_some(),
            "binary_elementwise: shape 走査ロジックのバグにより index {index:?} が範囲外になった"
        );
        out.push(scalar_kernel(
            x.unwrap_or_else(Element::zero),
            y.unwrap_or_else(Element::zero),
        ));
        increment_index(&mut index, &out_shape);
    }
    Tensor::new(out, &out_shape)
}

/// 単項 elementwise 演算（活性化）の Tensor 入口共通処理。shape は不変。
fn unary_elementwise(
    a: &Tensor<f32>,
    slice_kernel: fn(&[f32], &mut [f32]),
    scalar_kernel: fn(f32) -> f32,
) -> Result<Tensor<f32>, ShapeError> {
    if let Some(sa) = a.as_slice() {
        let mut out = vec![0.0f32; sa.len()];
        slice_kernel(sa, &mut out);
        return Tensor::new(out, a.shape());
    }

    let shape = a.shape();
    let numel = a.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        let x = a.get(&index);
        debug_assert!(
            x.is_some(),
            "unary_elementwise: shape 走査ロジックのバグにより index {index:?} が範囲外になった"
        );
        out.push(scalar_kernel(x.unwrap_or_else(Element::zero)));
        increment_index(&mut index, shape);
    }
    Tensor::new(out, shape)
}

/// 二項加算（ブロードキャスト対応）。`docs/public-api-design.md` §3.2/§4.2
/// `BackendOps::add` に対応する CPU 参照実装。
pub fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    binary_elementwise(a, b, add_slice, |x, y| x + y)
}

/// 二項乗算（ブロードキャスト対応）。`BackendOps::mul` に対応する
/// CPU 参照実装。
pub fn mul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    binary_elementwise(a, b, mul_slice, |x, y| x * y)
}

/// ReLU 活性化。`BackendOps::relu` に対応する CPU 参照実装。
pub fn relu(a: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    unary_elementwise(a, relu_slice, |x| x.max(0.0))
}

/// 指数関数活性化（`f32::exp`）。`BackendOps::exp` に対応する
/// CPU 参照実装。
pub fn exp(a: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    unary_elementwise(a, exp_slice, f32::exp)
}

/// 双曲線正接活性化（`f32::tanh`）。`BackendOps::tanh` に対応する
/// CPU 参照実装。
pub fn tanh(a: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    unary_elementwise(a, tanh_slice, f32::tanh)
}

/// 条件テンソルによる要素選択（`BackendOps::where_cond` に対応する
/// CPU 参照実装。イシュー #1637）。呼び出し元（`Var::where_cond`）が
/// `cond`／`a`／`b` を同一 `out_shape` へ broadcast 済みで渡す契約
/// だが、実装側でも shape 一致を再検査する（`.claude/rules/
/// security.md` A08。既存の `binary_elementwise` の strided 読み経路
/// の 3 項化は本イシューでは行わず、非 contiguous 入力は
/// `contiguous()` で実体化してから読む）。
pub fn where_cond(
    cond: &Tensor<f32>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, ShapeError> {
    let out_shape = a.shape().to_vec();
    if cond.shape() != out_shape.as_slice() {
        return Err(ShapeError::ShapeMismatch {
            lhs: cond.shape().to_vec(),
            rhs: out_shape,
        });
    }
    if b.shape() != out_shape.as_slice() {
        return Err(ShapeError::ShapeMismatch {
            lhs: b.shape().to_vec(),
            rhs: out_shape,
        });
    }

    if let (Some(cs), Some(as_), Some(bs)) = (cond.as_slice(), a.as_slice(), b.as_slice()) {
        let mut out = vec![0.0f32; cs.len()];
        where_slice(cs, as_, bs, &mut out);
        return Tensor::new(out, &out_shape);
    }

    let cc = cond.contiguous();
    let ac = a.contiguous();
    let bcv = b.contiguous();
    let cs = cc.as_slice().unwrap_or(&[]);
    let as_ = ac.as_slice().unwrap_or(&[]);
    let bs = bcv.as_slice().unwrap_or(&[]);
    debug_assert_eq!(
        cs.len(),
        out_shape.iter().product::<usize>(),
        "where_cond: contiguous() 直後は必ず as_slice を返すはず（契約違反）"
    );
    let mut out = vec![0.0f32; cs.len()];
    where_slice(cs, as_, bs, &mut out);
    Tensor::new(out, &out_shape)
}

/// マスク位置を定数で置換する（`BackendOps::masked_fill` に対応する
/// CPU 参照実装。イシュー #1637）。`mask` は `x` と同一 shape へ
/// broadcast 済み（呼び出し元契約）だが、実装側でも再検査する。
pub fn masked_fill(
    x: &Tensor<f32>,
    mask: &Tensor<f32>,
    value: f32,
) -> Result<Tensor<f32>, ShapeError> {
    let out_shape = x.shape().to_vec();
    if mask.shape() != out_shape.as_slice() {
        return Err(ShapeError::ShapeMismatch {
            lhs: mask.shape().to_vec(),
            rhs: out_shape,
        });
    }

    if let (Some(xs), Some(ms)) = (x.as_slice(), mask.as_slice()) {
        let mut out = vec![0.0f32; xs.len()];
        masked_fill_slice(xs, ms, value, &mut out);
        return Tensor::new(out, &out_shape);
    }

    let xc = x.contiguous();
    let mc = mask.contiguous();
    let xs = xc.as_slice().unwrap_or(&[]);
    let ms = mc.as_slice().unwrap_or(&[]);
    debug_assert_eq!(
        xs.len(),
        out_shape.iter().product::<usize>(),
        "masked_fill: contiguous() 直後は必ず as_slice を返すはず（契約違反）"
    );
    let mut out = vec![0.0f32; xs.len()];
    masked_fill_slice(xs, ms, value, &mut out);
    Tensor::new(out, &out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- スライスカーネル層 ---

    #[test]
    fn add_slice_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let mut out = [0.0; 3];
        add_slice(&a, &b, &mut out);
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn mul_slice_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let mut out = [0.0; 3];
        mul_slice(&a, &b, &mut out);
        assert_eq!(out, [10.0, 40.0, 90.0]);
    }

    #[test]
    fn relu_slice_basic() {
        let a = [-1.0, 0.0, 2.5, -0.001];
        let mut out = [0.0; 4];
        relu_slice(&a, &mut out);
        assert_eq!(out, [0.0, 0.0, 2.5, 0.0]);
    }

    #[test]
    fn relu_slice_nan_input_returns_zero() {
        // モジュール doc コメント「数値契約」に明記した挙動: `f32::max` 仕様
        // により NaN は無視され `relu(NaN) == 0.0` を返す（NumPy/PyTorch の
        // NaN 伝播とは異なる）。GPU バックエンド実装時に整合確認する対象。
        let a = [f32::NAN, -1.0, 1.0];
        let mut out = [0.0; 3];
        relu_slice(&a, &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn exp_slice_matches_std() {
        let a = [0.0, 1.0, -1.0, 2.5];
        let mut out = [0.0; 4];
        exp_slice(&a, &mut out);
        for (o, &x) in out.iter().zip(&a) {
            assert_eq!(*o, x.exp());
        }
    }

    #[test]
    fn tanh_slice_matches_std() {
        let a = [0.0, 1.0, -1.0, 2.5];
        let mut out = [0.0; 4];
        tanh_slice(&a, &mut out);
        for (o, &x) in out.iter().zip(&a) {
            assert_eq!(*o, x.tanh());
        }
    }

    #[test]
    fn slice_kernels_above_parallel_threshold_match_sequential() {
        // rayon 並列経路と逐次経路で数値が一致することを確認する
        // （モジュール doc コメント「並列化」: 演算順序に依存しない map 演算）。
        let n = PARALLEL_THRESHOLD + 17;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 5.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002).collect();

        let mut out_par = vec![0.0f32; n];
        add_slice(&a, &b, &mut out_par);
        let mut out_seq = vec![0.0f32; n];
        for ((o, &x), &y) in out_seq.iter_mut().zip(&a).zip(&b) {
            *o = x + y;
        }
        assert_eq!(out_par, out_seq);

        let mut mul_par = vec![0.0f32; n];
        mul_slice(&a, &b, &mut mul_par);
        let mut mul_seq = vec![0.0f32; n];
        for ((o, &x), &y) in mul_seq.iter_mut().zip(&a).zip(&b) {
            *o = x * y;
        }
        assert_eq!(mul_par, mul_seq);

        let mut relu_par = vec![0.0f32; n];
        relu_slice(&a, &mut relu_par);
        let mut relu_seq = vec![0.0f32; n];
        for (o, &x) in relu_seq.iter_mut().zip(&a) {
            *o = x.max(0.0);
        }
        assert_eq!(relu_par, relu_seq);

        let mut exp_par = vec![0.0f32; n];
        exp_slice(&a, &mut exp_par);
        let mut exp_seq = vec![0.0f32; n];
        for (o, &x) in exp_seq.iter_mut().zip(&a) {
            *o = x.exp();
        }
        assert_eq!(exp_par, exp_seq);

        let mut tanh_par = vec![0.0f32; n];
        tanh_slice(&a, &mut tanh_par);
        let mut tanh_seq = vec![0.0f32; n];
        for (o, &x) in tanh_seq.iter_mut().zip(&a) {
            *o = x.tanh();
        }
        assert_eq!(tanh_par, tanh_seq);
    }

    // --- Tensor 入口層: 数値一致 ---

    #[test]
    fn add_same_shape() {
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let out = add(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(out.get(&[1, 1]).unwrap(), 44.0);
    }

    #[test]
    fn mul_same_shape() {
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let out = mul(&a, &b).unwrap();
        assert_eq!(out.get(&[0, 0]).unwrap(), 10.0);
        assert_eq!(out.get(&[1, 1]).unwrap(), 160.0);
    }

    #[test]
    fn relu_tensor_matches_expected() {
        let a = Tensor::<f32>::new(vec![-1.0, 0.0, 2.5, -0.001], &[4]).unwrap();
        let out = relu(&a).unwrap();
        let expected = [0.0, 0.0, 2.5, 0.0];
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(out.get(&[i]).unwrap(), v);
        }
    }

    #[test]
    fn exp_tensor_matches_std() {
        let a = Tensor::<f32>::new(vec![0.0, 1.0, -1.0], &[3]).unwrap();
        let out = exp(&a).unwrap();
        for i in 0..3 {
            assert_eq!(out.get(&[i]).unwrap(), a.get(&[i]).unwrap().exp());
        }
    }

    #[test]
    fn tanh_tensor_matches_std() {
        let a = Tensor::<f32>::new(vec![0.0, 1.0, -1.0], &[3]).unwrap();
        let out = tanh(&a).unwrap();
        for i in 0..3 {
            assert_eq!(out.get(&[i]).unwrap(), a.get(&[i]).unwrap().tanh());
        }
    }

    // --- Tensor 入口層: ブロードキャスト ---

    #[test]
    fn add_broadcast_row_and_column() {
        // 受け入れ条件対象の代表例: [3,1] + [1,4] -> [3,4]。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]).unwrap();
        let out = add(&a, &b).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                let expected = (i as f32 + 1.0) + (j as f32 + 1.0) * 10.0;
                assert_eq!(out.get(&[i, j]).unwrap(), expected);
            }
        }
    }

    #[test]
    fn add_bias_row_vector_over_matrix() {
        // [2,3] + [3]（行ベクトル bias）。PoC-v2-1 の代表的な組合せ。
        let a = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let bias = Tensor::<f32>::new(vec![100.0, 200.0, 300.0], &[3]).unwrap();
        let out = add(&a, &bias).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(
                    out.get(&[i, j]).unwrap(),
                    a.get(&[i, j]).unwrap() + bias.get(&[j]).unwrap()
                );
            }
        }
    }

    #[test]
    fn mul_scalar_like_shape() {
        // rank 0（スカラー相当）とのブロードキャスト。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let s = Tensor::<f32>::new(vec![2.0], &[]).unwrap();
        let out = mul(&a, &s).unwrap();
        assert_eq!(out.shape(), &[3]);
        for i in 0..3 {
            assert_eq!(out.get(&[i]).unwrap(), a.get(&[i]).unwrap() * 2.0);
        }
    }

    #[test]
    fn add_broadcast_incompatible_returns_error() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[4]).unwrap();
        let err = add(&a, &b).unwrap_err();
        assert!(matches!(err, ShapeError::BroadcastIncompatible { .. }));
    }

    // --- Tensor 入口層: 非 contiguous view 入力 ---

    #[test]
    fn add_with_transposed_view_matches_contiguous() {
        let a = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let a_t = a.transpose(0, 1).unwrap(); // shape [3, 2], 非 contiguous
        let b = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 2]).unwrap();

        let out_view = add(&a_t, &b).unwrap();
        let out_contig = add(&a_t.contiguous(), &b).unwrap();

        assert_eq!(out_view.shape(), &[3, 2]);
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(
                    out_view.get(&[i, j]).unwrap(),
                    out_contig.get(&[i, j]).unwrap()
                );
            }
        }
    }

    #[test]
    fn relu_with_narrowed_view_matches_contiguous() {
        let a = Tensor::<f32>::new(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[8]).unwrap();
        let n = a.narrow(0, 2, 4).unwrap(); // [-1.0, 0.0, 1.0, 2.0]、非 contiguous ではないが offset 付き view

        let out_view = relu(&n).unwrap();
        let out_contig = relu(&n.contiguous()).unwrap();
        for i in 0..4 {
            assert_eq!(out_view.get(&[i]).unwrap(), out_contig.get(&[i]).unwrap());
        }
    }

    // --- スライスカーネル層: 閾値境界（#25） ---

    /// `PARALLEL_THRESHOLD` ちょうど・±1 の 3 点で、5 つの `*_slice` カーネル
    /// すべての逐次/並列切替点が数値一致することを確認する。既存の
    /// `slice_kernels_above_parallel_threshold_match_sequential` は
    /// 閾値+17 の 1 点（並列側）のみを確認しており、閾値ちょうど・
    /// 直下（逐次側）の切替境界は未検証だった（#25 棚卸しで特定した
    /// ギャップ）。
    #[test]
    fn slice_kernels_match_sequential_at_threshold_boundary() {
        for n in [
            PARALLEL_THRESHOLD - 1,
            PARALLEL_THRESHOLD,
            PARALLEL_THRESHOLD + 1,
        ] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 5.0).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002).collect();

            let mut out_par = vec![0.0f32; n];
            add_slice(&a, &b, &mut out_par);
            let mut out_seq = vec![0.0f32; n];
            for ((o, &x), &y) in out_seq.iter_mut().zip(&a).zip(&b) {
                *o = x + y;
            }
            assert_eq!(out_par, out_seq, "add_slice mismatch at n={n}");

            let mut mul_par = vec![0.0f32; n];
            mul_slice(&a, &b, &mut mul_par);
            let mut mul_seq = vec![0.0f32; n];
            for ((o, &x), &y) in mul_seq.iter_mut().zip(&a).zip(&b) {
                *o = x * y;
            }
            assert_eq!(mul_par, mul_seq, "mul_slice mismatch at n={n}");

            let mut relu_par = vec![0.0f32; n];
            relu_slice(&a, &mut relu_par);
            let mut relu_seq = vec![0.0f32; n];
            for (o, &x) in relu_seq.iter_mut().zip(&a) {
                *o = x.max(0.0);
            }
            assert_eq!(relu_par, relu_seq, "relu_slice mismatch at n={n}");

            let mut exp_par = vec![0.0f32; n];
            exp_slice(&a, &mut exp_par);
            let mut exp_seq = vec![0.0f32; n];
            for (o, &x) in exp_seq.iter_mut().zip(&a) {
                *o = x.exp();
            }
            assert_eq!(exp_par, exp_seq, "exp_slice mismatch at n={n}");

            let mut tanh_par = vec![0.0f32; n];
            tanh_slice(&a, &mut tanh_par);
            let mut tanh_seq = vec![0.0f32; n];
            for (o, &x) in tanh_seq.iter_mut().zip(&a) {
                *o = x.tanh();
            }
            assert_eq!(tanh_par, tanh_seq, "tanh_slice mismatch at n={n}");
        }
    }

    // --- スライスカーネル層: 長さ不一致 assert 契約（#25） ---
    //
    // `*_slice` は TASK-1.9（#43）で `BackendOps` から直接再利用される想定の
    // pub 関数であり、長さ不一致を `assert_eq!`（release でも有効）で拒否する
    // 契約を持つ（モジュール doc コメント「スライスカーネル層」参照）。
    // 以下のテストは通常の `cargo test`（debug assertion 有効ビルド）で
    // 「長さ不一致がパニックで拒否されること」を固定するもので、
    // `assert_eq!` から `debug_assert_eq!` への後退を検出できるのは
    // debug assertion を無効化したビルド（`cargo test --release` 等。
    // 本リポの CI ゲートには含まれない）で実行した場合に限る。

    #[test]
    #[should_panic(expected = "add_slice: length mismatch (a vs b)")]
    fn add_slice_rejects_a_b_length_mismatch() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];
        let mut out = [0.0; 3];
        add_slice(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "add_slice: length mismatch (a vs out)")]
    fn add_slice_rejects_a_out_length_mismatch() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let mut out = [0.0; 2];
        add_slice(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "mul_slice: length mismatch (a vs b)")]
    fn mul_slice_rejects_length_mismatch() {
        let a = [1.0, 2.0];
        let b = [1.0];
        let mut out = [0.0; 2];
        mul_slice(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "relu_slice: length mismatch")]
    fn relu_slice_rejects_length_mismatch() {
        let a = [1.0, 2.0];
        let mut out = [0.0; 1];
        relu_slice(&a, &mut out);
    }

    #[test]
    #[should_panic(expected = "exp_slice: length mismatch")]
    fn exp_slice_rejects_length_mismatch() {
        let a = [1.0, 2.0];
        let mut out = [0.0; 1];
        exp_slice(&a, &mut out);
    }

    #[test]
    #[should_panic(expected = "tanh_slice: length mismatch")]
    fn tanh_slice_rejects_length_mismatch() {
        let a = [1.0, 2.0];
        let mut out = [0.0; 1];
        tanh_slice(&a, &mut out);
    }

    // --- Tensor 入口層: 境界形状 ---

    #[test]
    fn add_empty_tensor_returns_empty() {
        let a = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        let out = add(&a, &b).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
        assert!(out.is_empty());
    }

    #[test]
    fn relu_empty_tensor_returns_empty() {
        let a = Tensor::<f32>::zeros(&[0]).unwrap();
        let out = relu(&a).unwrap();
        assert!(out.is_empty());
    }

    /// 空テンソルの網羅（#25 棚卸しで特定したギャップ）: 既存カバレッジは
    /// `add`/`relu` のみで、`mul`（二項）・`exp`/`tanh`（単項 libm 経由）の
    /// 空入力は未検証だった。shape のバリエーション（`[0]`・`[0,3]`・
    /// `[3,0]`）も併せて確認する。
    #[test]
    fn mul_exp_tanh_empty_tensor_returns_empty() {
        for shape in [&[0usize][..], &[0, 3][..], &[3, 0][..]] {
            let a = Tensor::<f32>::zeros(shape).unwrap();
            let b = Tensor::<f32>::zeros(shape).unwrap();

            let mul_out = mul(&a, &b).unwrap();
            assert_eq!(mul_out.shape(), shape);
            assert!(mul_out.is_empty());

            let exp_out = exp(&a).unwrap();
            assert_eq!(exp_out.shape(), shape);
            assert!(exp_out.is_empty());

            let tanh_out = tanh(&a).unwrap();
            assert_eq!(tanh_out.shape(), shape);
            assert!(tanh_out.is_empty());
        }
    }

    /// 片側ゼロサイズ shape を含む broadcast: `[0,3] + [3]` -> `[0,3]`。
    /// `elementwise_out_shape`（`tensor-core::broadcast_shape` 委譲）は
    /// 次元 0 と次元 1 の broadcast を許容し（`(a, 1) => a` の分岐で
    /// `a=0` も一致扱い）、出力は非ゼロ側でなく `0` 側に揃う。実装の受理
    /// 仕様どおりの結果を固定化する（#25 棚卸しで特定したギャップ）。
    #[test]
    fn add_broadcast_with_zero_size_shape() {
        let a = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        let b = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let out = add(&a, &b).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
        assert!(out.is_empty());
    }

    /// 非正方 rank3 での broadcast（#25 棚卸しで特定したギャップ）:
    /// `[2,3,4] * [1,3,1]`。各次元が異なる値を持つ形状で、拡張軸
    /// （axis 0・axis 2）が正しく stride 0 として読まれることを確認する。
    #[test]
    fn mul_broadcast_rank3_non_square_shapes() {
        let a = Tensor::<f32>::new((0..24).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0], &[1, 3, 1]).unwrap();
        let out = mul(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let expected = a.get(&[i, j, k]).unwrap() * b.get(&[0, j, 0]).unwrap();
                    assert_eq!(out.get(&[i, j, k]).unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn add_single_element() {
        let a = Tensor::<f32>::new(vec![3.0], &[1]).unwrap();
        let b = Tensor::<f32>::new(vec![4.0], &[1]).unwrap();
        let out = add(&a, &b).unwrap();
        assert_eq!(out.get(&[0]).unwrap(), 7.0);
    }

    #[test]
    fn add_around_parallel_threshold_matches_scalar_expected() {
        // rayon 閾値前後のサイズで Tensor 入口層の数値一致を確認する
        // （閾値そのもののチューニングは #24 のスコープ）。
        for n in [
            PARALLEL_THRESHOLD - 1,
            PARALLEL_THRESHOLD,
            PARALLEL_THRESHOLD + 1,
        ] {
            let av: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let bv: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
            let a = Tensor::<f32>::new(av.clone(), &[n]).unwrap();
            let b = Tensor::<f32>::new(bv.clone(), &[n]).unwrap();
            let out = add(&a, &b).unwrap();
            for i in 0..n {
                assert_eq!(out.get(&[i]).unwrap(), av[i] + bv[i]);
            }
        }
    }

    // --- where_cond／masked_fill（イシュー #1637） ---

    #[test]
    fn where_slice_selects_by_condition() {
        let cond = vec![1.0, 0.0, 1.0, 0.0];
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        where_slice(&cond, &a, &b, &mut out);
        assert_eq!(out, vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn where_slice_length_mismatch_panics() {
        let cond = vec![1.0, 0.0];
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let mut out = vec![0.0; 2];
        where_slice(&cond, &a, &b, &mut out);
    }

    #[test]
    fn where_slice_around_parallel_threshold_matches_scalar_expected() {
        for n in [
            PARALLEL_THRESHOLD - 1,
            PARALLEL_THRESHOLD,
            PARALLEL_THRESHOLD + 1,
        ] {
            let cond: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            let mut out = vec![0.0; n];
            where_slice(&cond, &a, &b, &mut out);
            for i in 0..n {
                let expected = if cond[i] != 0.0 { a[i] } else { b[i] };
                assert_eq!(out[i], expected);
            }
        }
    }

    #[test]
    fn masked_fill_slice_replaces_masked_positions() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0; 4];
        masked_fill_slice(&x, &mask, -1.0, &mut out);
        assert_eq!(out, vec![-1.0, 2.0, -1.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn masked_fill_slice_length_mismatch_panics() {
        let x = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0];
        let mut out = vec![0.0; 3];
        masked_fill_slice(&x, &mask, 0.0, &mut out);
    }

    #[test]
    fn masked_fill_slice_around_parallel_threshold_matches_scalar_expected() {
        for n in [
            PARALLEL_THRESHOLD - 1,
            PARALLEL_THRESHOLD,
            PARALLEL_THRESHOLD + 1,
        ] {
            let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mask: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
            let mut out = vec![0.0; n];
            masked_fill_slice(&x, &mask, -9.0, &mut out);
            for i in 0..n {
                let expected = if mask[i] != 0.0 { -9.0 } else { x[i] };
                assert_eq!(out[i], expected);
            }
        }
    }

    #[test]
    fn where_cond_tensor_entry_matches_slice() {
        let cond = Tensor::<f32>::new(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let out = where_cond(&cond, &a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        for i in 0..4 {
            let expected = if [1.0, 0.0, 1.0, 0.0][i] != 0.0 {
                [1.0, 2.0, 3.0, 4.0][i]
            } else {
                [10.0, 20.0, 30.0, 40.0][i]
            };
            assert_eq!(out.get(&[i / 2, i % 2]).unwrap(), expected);
        }
    }

    #[test]
    fn where_cond_shape_mismatch_returns_error() {
        let cond = Tensor::<f32>::new(vec![1.0, 0.0], &[2]).unwrap();
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let err = where_cond(&cond, &a, &b).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn where_cond_with_non_contiguous_view_input_matches_contiguous() {
        let a_base = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let a_t = a_base.transpose(0, 1).unwrap(); // [3, 2], non-contiguous
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]).unwrap();
        let cond = Tensor::<f32>::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[3, 2]).unwrap();
        let out = where_cond(&cond, &a_t, &b).unwrap();
        let a_t_contig = a_t.contiguous();
        let out_contig = where_cond(&cond, &a_t_contig, &b).unwrap();
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(out.get(&[i, j]).unwrap(), out_contig.get(&[i, j]).unwrap());
            }
        }
    }

    #[test]
    fn masked_fill_tensor_entry_matches_slice() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mask = Tensor::<f32>::new(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
        let out = masked_fill(&x, &mask, -1.0).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        for i in 0..4 {
            let expected = if [1.0, 0.0, 1.0, 0.0][i] != 0.0 {
                -1.0
            } else {
                [1.0, 2.0, 3.0, 4.0][i]
            };
            assert_eq!(out.get(&[i / 2, i % 2]).unwrap(), expected);
        }
    }

    #[test]
    fn masked_fill_shape_mismatch_returns_error() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let mask = Tensor::<f32>::new(vec![1.0, 0.0], &[2]).unwrap();
        let err = masked_fill(&x, &mask, 0.0).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn masked_fill_with_non_contiguous_view_input_matches_contiguous() {
        let x_base = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let x_t = x_base.transpose(0, 1).unwrap(); // [3, 2], non-contiguous
        let mask = Tensor::<f32>::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[3, 2]).unwrap();
        let out = masked_fill(&x_t, &mask, -9.0).unwrap();
        let x_t_contig = x_t.contiguous();
        let out_contig = masked_fill(&x_t_contig, &mask, -9.0).unwrap();
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(out.get(&[i, j]).unwrap(), out_contig.get(&[i, j]).unwrap());
            }
        }
    }
}
