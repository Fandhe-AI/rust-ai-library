//! naive CPU の forward 値計算（クレート非公開・暫定参照実装）。
//!
//! `Var`（`var.rs`）の各演算メソッドが `tensor-core::Tensor<f32>` の
//! 値を実際に計算するために呼ぶ。`backend-cpu`（TASK-1.6・#20 以降）が
//! まだ未完のため、TASK-1.9（バックエンド抽象層への接続）で backend
//! 経由の実行に置き換えるまでの暫定実装である（PoC-v2-2 の
//! `docs/spec/03-poc/poc-v2-2-autodiff/` 構成に合わせ、テープ機構と
//! 値計算を分離しておくことで差し替えの影響範囲をこのファイルに限定
//! する）。
//!
//! **FMA 契約**: `matmul` の内積蓄積は `f32::mul_add` を用いる
//! （`.claude/rules/coding-rust.md`「CPU 参照実装は `f32::mul_add` を
//! 用い、GPU 側の既定 FMA 契約と揃える」。PoC-v2-5 の K=4096 ストレス
//! ケースで実測確認済みの丸め方針）。
//!
//! shape の事前検査（`matmul_out_shape`/`broadcast_shape`/
//! `require_same_shape`/`reduce_out_shape`）は呼び出し元（`var.rs`）が
//! 済ませてから本モジュールを呼ぶ契約とする。本モジュールの関数は
//! shape が既に整合していることを前提とし、`ShapeError` を返さない
//! （`tensor-core::Tensor` 側 API のエラーも本番経路の
//! `unwrap()`/`expect()` は使わず `debug_assert!` 経由のフォールバックで
//! 吸収する。`.claude/rules/coding-rust.md`）。

use std::borrow::Cow;

use fandhe_ai_tensor_core::{GruBackwardOutput, GruPointwiseOutput, LstmPointwiseOutput, Tensor};

use crate::layout;
use crate::var::Reduction;

/// 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm のホスト
/// 参照実装（イシュー #1621）。行数が大きいため子モジュールへ分ける
/// （モジュール冒頭コメント参照）。
pub(crate) mod linalg;

std::thread_local! {
    /// `matmul`（下記）が転置 view（`grad.rs::transpose2d` が作る
    /// zero-copy view）を `layout::classify_2d` で分類できず、
    /// `dense_vec`（`contiguous()` 経由のホスト側転置コピー）へ
    /// フォールバックした回数（イシュー #1046。`backend-metal::ops::
    /// RESIDENT_HOST_REPACK_COUNT` と同型の可観測点）。
    ///
    /// イシュー #1211 以降、本番の matmul VJP（`grad.rs::matmul_vjp`・
    /// `Op::LinearResident` の `d_weight`）は `eval::matmul` ではなく
    /// `BackendOps::gemm` を経由するため、本カウンタが観測するのは
    /// `NaiveOps`／`TestOps`（compat・テストの forward 参照実装経路）
    /// 経由の呼び出しに限られる（`grad.rs` の `matmul_vjp_does_not_
    /// repack_transposed_operands` テストは `test_ops()` 越しにこの
    /// compat 経路のゼロコピーを検証している。本番 `CpuBackendOps::
    /// gemm` 経路の転置再パックは、片側転置〈NT/TN〉かつ dense な転置
    /// 格納を判定できる場合、CPU（イシュー #1213）・CUDA（イシュー
    /// #1214）・Metal（イシュー #1215）とも解消済み
    /// （`backend-cpu::ops::GEMM_HOST_REPACK_COUNT`／`backend-cuda::
    /// ops::GEMM_HOST_REPACK_COUNT`／`backend-metal::ops::
    /// GEMM_HOST_REPACK_COUNT`〈いずれも本カウンタとは別の crate 内部
    /// カウンタ〉で可観測。Metal は既存 NN 経路とは別カーネルを通る
    /// ため数値契約が bit 一致ではなく REQ-2 複合判定である点が
    /// CPU／CUDA と異なる）。`docs/matmul-vjp-zero-copy-decision.md`
    /// §4・§4.2・§4.3・§4.4 追補）。
    pub(crate) static MATMUL_HOST_REPACK_COUNT: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

/// テンソルを行優先連続バッファへ実体化し `Vec<f32>` として取り出す。
///
/// `contiguous()` は非 contiguous な入力（transpose・stride 0
/// ブロードキャスト view 等）を実体化するため、その結果に対する
/// `as_slice()` は理論上必ず `Some` を返す。それでも本番経路で
/// `unwrap()`/`expect()` は使わない方針（`.claude/rules/coding-rust.md`）
/// のため、`None` 経路は多次元インデックス走査によるコピーへ
/// フォールバックする（到達すれば `contiguous()`/`is_contiguous()` の
/// 契約違反であり、`debug_assert!` で検知可能にする）。
///
/// `pub(crate)`: `grad.rs`（TASK-1.5b・#17）が各演算の VJP 計算・
/// 数値微分突合テストで forward と同じ稠密化ロジックを再利用する
/// （数式の実体を 2 か所に別実装しない方針。PoC-v2-2 準拠）。
pub(crate) fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let contiguous = tensor.contiguous();
    if let Some(slice) = contiguous.as_slice() {
        return slice.to_vec();
    }
    debug_assert!(
        false,
        "dense_vec: contiguous() 後の as_slice() が None を返した（契約違反）"
    );
    let shape = contiguous.shape().to_vec();
    let numel = contiguous.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        out.push(contiguous.get(&index).unwrap_or(0.0));
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

/// `dense_vec` の読み取り専用・コピー回避版（イシュー #1026・
/// `perf(backend-cpu): 学習ループのホスト側コピー・再構築を除去する`）。
///
/// `Sgd::step`／`AdamW::step`（`crates/autodiff/src/optim/sgd.rs`・
/// `crates/autodiff/src/nn/optim/adamw.rs`）は各 step で `param`／`grad`／
/// momentum バッファを走査するだけで書き換えない（更新後の値は別の
/// 新規 `Vec` へ積んで `Tensor::new` で構築し直す）。この読み取り専用の
/// 用途では `dense_vec` の `slice.to_vec()`（ヒープ確保 + 全要素コピー）
/// は不要であり、既に contiguous な入力（`Linear::weight`/`bias`・
/// `Gradients` 出力はいずれも密なバッファ）に対しては `tensor.as_slice()`
/// が直接借用スライスを返す（`contiguous()` を経由しない）ため、それを
/// そのまま返せば呼び出し元の走査は成立する。
///
/// 戻り値を `Cow<[f32]>` にしているのは、非 contiguous な入力
/// （transpose 済み view 等）では `contiguous()` が新しい `Tensor` を
/// 実体化する必要があり、その結果は本関数のローカル変数になるため
/// スライスを呼び出し元へ借用として返せない（ダングリング参照になり
/// コンパイルエラーになる）ためである。この場合のみ `dense_vec`
/// （所有権を持つ `Vec` を返す既存の稠密化ロジック。二重実装しない）
/// へフォールバックし `Cow::Owned` として返す。
///
/// `pub(crate)`: `dense_vec` と同じ可視性（optimizer モジュールから
/// 呼ばれるための最小限の公開範囲）。
pub(crate) fn dense_vec_ref(tensor: &Tensor<f32>) -> Cow<'_, [f32]> {
    match tensor.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(dense_vec(tensor)),
    }
}

/// `g: [m, n]`（rank-2）の**行方向の和**（列ごとに `sum_{row=0}^{m-1}
/// g[row, col]`）を計算し、長さ `n` の `Vec<f32>` を返す（イシュー
/// #1566）。
///
/// **数値方式（2026-09-12 ユーザー承認 A・PR #1659 codex-review P1
/// 是正）**: `.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64`
/// アキュムレータで統一する」規約（イシュー #1102・PR #1120）に従い、
/// 各列は `f64` アキュムレータへ `dense_vec` で稠密化した行を `f64` へ
/// 昇格して蓄積し、最後に 1 回だけ `f32` へ downcast する（単純な `f32`
/// 逐次 `+=` は `[1e8, 1.0, -1e8]` のような相殺パターンで寄与が丸め
/// 落ちして消える。回帰テスト `reduce_bias_grad_rows_tests::
/// preserves_cancelling_contribution_via_f64_accumulator` 参照）。
/// この変更により `grad::reduce_to_shape(g, &[n])`（同じ縮約を行う
/// 汎用パス。`f32` 逐次和のまま**変更しない**——weight 勾配・`reduce_
/// to_shape` 自体の bit 同一契約は本イシューのスコープ外）との bit
/// 完全一致は失われる（意図的な乖離。両者の使い分けは呼び出し元 doc
/// 「resident 経路のみ」を参照）。
///
/// **`m == 1` の特殊扱い**（PR #1659 codex-review P2 是正。f64 化後も
/// 不変）: `m == 1` は加算を経由せず入力を直接コピーするため、`f64`
/// 昇格・downcast のラウンドトリップでも符号付きゼロ（`-0.0`）を保持
/// する（`f32` → `f64` → `f32` は値を変えない可逆変換）。
///
/// `pub(crate)`: `grad.rs`（`Op::LinearResident` の非 resident bias
/// フォールバック。呼び出しは変更しない——既存の `reduce_to_shape` 経路
/// を維持し、本関数は resident 経路のみで使う）・`optim::device_store`
/// （weight tying 発生時の bias 勾配 tie 累積。`ResidentResolver::
/// fill_resident_weight_grad` doc「bias tie」参照）から呼ばれる。
///
/// `g.shape()` が `[m, n]`（rank-2）でない呼び出しは契約違反
/// （`debug_assert!` で検知。本番経路は空 `Vec` を返す安全側フォール
/// バックとし panic しない。`.claude/rules/coding-rust.md`「本番経路で
/// `unwrap()`/`expect()` を使わない」）。
pub(crate) fn reduce_bias_grad_rows(g: &Tensor<f32>) -> Vec<f32> {
    let shape = g.shape();
    if shape.len() != 2 {
        debug_assert!(
            false,
            "reduce_bias_grad_rows: g は rank-2 のはず（契約違反）"
        );
        return Vec::new();
    }
    let (m, n) = (shape[0], shape[1]);
    let data = dense_vec(g);
    if m == 1 {
        // `reduce_to_shape` は `m == 1` の軸を縮約しないため、直接
        // コピーして `-0.0` 等の符号付きゼロを保持する（上記 doc 参照）。
        return data[0..n].to_vec();
    }
    let mut acc = vec![0f64; n];
    for row in 0..m {
        for (col, a) in acc.iter_mut().enumerate() {
            *a += f64::from(data[row * n + col]);
        }
    }
    acc.into_iter().map(|v| v as f32).collect()
}

/// shape とデータ長の一致を型で保証する非 panic 構築（TASK-12.1d・
/// #164。`docs/fusion-graph-design.md` §2.5「eval.rs 非 panic 化の設計
/// 方針」）。`Tensor::from_shape_fill`（`tensor-core` 側の総コンスト
/// ラクタ。`pub` + `#[doc(hidden)]`）は `shape` から `numel` を導出し
/// `fill` で埋める。呼び出し元（本モジュール内）はすべて事前に shape
/// 検査済みの出力を組み立てるため、実運用では `data.len()` と `shape`
/// は必ず一致し要素数積のオーバーフローも起こらない
/// （`debug_assert_eq!` で契約違反を検知可能にする。不一致時は
/// `get(i).copied().unwrap_or(0.0)` により欠落分を `0.0` で安全側に
/// 埋める）。
///
/// **`from_shape_fill` は shape の要素数積を `checked_numel` で検査する
/// `Result` を返す（PR #403 codex-review P1 是正。`tensor.rs` の該当
/// コメント参照）**: `materialize_non_fallible`〈`tape.rs`〉が要求する
/// 「構造的に失敗しない」契約（`docs/fusion-graph-design.md` §3.5.3
/// (iii)）を保つため、本関数自体は引き続き必ず値を返す非 panic 関数の
/// ままとする——`Err`（理論上到達しない契約違反）は `debug_assert!` で
/// 検知しつつ [`fandhe_ai_tensor_core::Tensor::scalar`]（真に infallible）による
/// 安全側フォールバックへ吸収する。
pub(crate) fn build_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    debug_assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "build_tensor: shape 検査済みのはずのデータ長が一致しない（契約違反）"
    );
    Tensor::from_shape_fill(shape, |i| data.get(i).copied().unwrap_or(0.0)).unwrap_or_else(|_| {
        debug_assert!(
            false,
            "build_tensor: shape の要素数積がオーバーフローした（契約違反）"
        );
        Tensor::scalar(0.0)
    })
}

/// クラス添字テンソル（`Tensor<i32>`）の稠密化。`dense_vec`（上記）の
/// `i32` 版で、`cross_entropy_loss`（下記）・`Var::cross_entropy_loss`
/// （`var.rs`。targets 添字の範囲検査）が読み出し専用で使う。
pub(crate) fn dense_vec_i32(tensor: &Tensor<i32>) -> Vec<i32> {
    let contiguous = tensor.contiguous();
    if let Some(slice) = contiguous.as_slice() {
        return slice.to_vec();
    }
    debug_assert!(
        false,
        "dense_vec_i32: contiguous() 後の as_slice() が None を返した（契約違反）"
    );
    let shape = contiguous.shape().to_vec();
    let numel = contiguous.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        out.push(contiguous.get(&index).unwrap_or(0));
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

/// `matmul`（下記）のオペランド 1 個を、ホスト側転置コピーなしで
/// 読み出せる形へ変換する（イシュー #1046）。
///
/// `layout::classify_2d`（`crate::layout`。`backend-metal::layout` と
/// 同一規則の双子モジュール。PR #1077 で `tensor-core` からクレート内
/// 非公開モジュールへ差し戻した。詳細は `crate::layout` のクレート
/// ドキュメント参照）が行優先 contiguous・転置 view（`grad.rs::transpose2d` が作る
/// `strides == [1, ld]` の zero-copy view）のいずれかに分類できる場合、
/// `Tensor::as_view_slice`（借用）をそのまま返し `MATMUL_HOST_REPACK_COUNT`
/// を増やさない。分類できない形状（stride 0 のブロードキャスト等）
/// のみ、従来どおり `dense_vec`（`contiguous()` 経由のホスト側コピー）
/// へフォールバックしカウンタを増やす。
fn matmul_operand(tensor: &Tensor<f32>) -> (Cow<'_, [f32]>, layout::MatrixLayout) {
    if let Some(matrix_layout) = layout::classify_2d(tensor.shape(), tensor.strides())
        && let Some(slice) = tensor.as_view_slice()
    {
        return (Cow::Borrowed(slice), matrix_layout);
    }
    MATMUL_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    (
        Cow::Owned(dense_vec(tensor)),
        layout::MatrixLayout {
            rows,
            cols,
            ld: cols,
            transposed: false,
        },
    )
}

/// 2 次元 `matmul`（`lhs: [m,k]` × `rhs: [k,n]` → `[m,n]`）。
/// shape 検査（`matmul_out_shape`）は呼び出し元が済ませている前提。
///
/// イシュー #1046: `matmul_vjp`（`grad.rs`）が `transpose2d`（zero-copy
/// view）で作った転置オペランドをそのまま渡しても、`matmul_operand` が
/// `layout::MatrixLayout` の添字式（`transposed` フラグで行優先／列優先
/// を切替）で読み出すためホスト側転置コピーが発生しない。行優先
/// contiguous 入力（従来からの主経路）では `ld == cols` となり、
/// 添字式は変更前の `lhs_data[i * k + p]`／`rhs_data[p * n + j]` と
/// 完全に一致する（k ループの反復順・`mul_add` 呼び出しも不変のため
/// 既存の bit 完全一致テストを崩さない）。
pub(crate) fn matmul(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];
    let (lhs_data, lhs_layout) = matmul_operand(lhs);
    let (rhs_data, rhs_layout) = matmul_operand(rhs);
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let a = if lhs_layout.transposed {
                    lhs_data[p * lhs_layout.ld + i]
                } else {
                    lhs_data[i * lhs_layout.ld + p]
                };
                let b = if rhs_layout.transposed {
                    rhs_data[j * rhs_layout.ld + p]
                } else {
                    rhs_data[p * rhs_layout.ld + j]
                };
                // FMA 契約統一（コメント冒頭参照）: 積和を `mul_add` で行う。
                acc = a.mul_add(b, acc);
            }
            out[i * n + j] = acc;
        }
    }
    build_tensor(out, &[m, n])
}

/// ブロードキャスト付き要素ごとの二項演算（`add`/`mul` 共通実装）。
/// shape 検査（`broadcast_shape`）は呼び出し元が済ませている前提。
/// `tensor-core::Tensor::broadcast_with` で両者を共通 shape の view へ
/// 揃えたうえで要素ごとに `op` を適用する。
fn broadcast_binary(
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    op: impl Fn(f32, f32) -> f32,
) -> Tensor<f32> {
    let (blhs, brhs) = match lhs.broadcast_with(rhs) {
        Ok(pair) => pair,
        Err(_) => {
            debug_assert!(
                false,
                "broadcast_binary: 呼び出し元の broadcast_shape 検査済み前提が崩れた"
            );
            return lhs.clone();
        }
    };
    let shape = blhs.shape().to_vec();
    let lhs_data = dense_vec(&blhs);
    let rhs_data = dense_vec(&brhs);
    let out: Vec<f32> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| op(a, b))
        .collect();
    build_tensor(out, &shape)
}

/// bias broadcast を含む要素ごとの加算（`docs/public-api-design.md` §3.2）。
pub(crate) fn add(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    broadcast_binary(lhs, rhs, |a, b| a + b)
}

/// ブロードキャスト付き要素ごとの乗算。
pub(crate) fn mul(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    broadcast_binary(lhs, rhs, |a, b| a * b)
}

/// shape 不変の要素ごとの単項演算（`relu`/`exp`/`tanh` 共通実装）。
fn unary(input: &Tensor<f32>, op: impl Fn(f32) -> f32) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let data = dense_vec(input);
    let out: Vec<f32> = data.into_iter().map(op).collect();
    build_tensor(out, &shape)
}

/// NaN 伝播する 2 項最大値（IEEE 754 `maximum` セマンティクス相当）。
///
/// `f32::max` は非 `NaN` 側のオペランドを返すため、上流で発生した
/// `NaN` が `relu`/`max` reduction を通過すると forward 値から消え、
/// テープに記録される数値のデバッグやバックエンド間数値一致検証
/// （`.claude/rules/coding-rust.md`「相対誤差 1e-3 未満 または絶対誤差
/// 1e-5 未満」）に影響しうる（Cursor Bugbot 指摘。PR #221）。
/// いずれかが `NaN` なら `NaN` を返し、伝播を保つ。
fn nan_propagating_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

pub(crate) fn relu(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, |v| nan_propagating_max(v, 0.0))
}

pub(crate) fn exp(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, f32::exp)
}

pub(crate) fn tanh(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, f32::tanh)
}

/// 数値安定形のシグモイド。`x >= 0` は `1/(1+exp(-x))`、`x < 0` は
/// `exp(x)/(1+exp(x))` を使い分け、大きな負値入力での `exp` オーバー
/// フロー（`exp(-x)` が `+inf` に発散する経路）を回避する
/// （TASK-9.1b・#92。`nn::activation::Sigmoid` の forward 実体）。
/// `NaN` 入力はいずれの分岐も `NaN` を伝播する（`is_sign_negative` は
/// `NaN` に対して符号ビットで分岐するが、後続の演算が `NaN` を保つため
/// 結果は変わらない）。
fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn sigmoid(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, sigmoid_scalar)
}

/// `dim` に沿った reduction（`sum`/`max` 共通の走査ロジック）。
/// `input` は行優先連続データとして走査し、`axis` を
/// 「外側（outer）× 走査軸（axis_len）× 内側（inner）」の 3 段に分解
/// することで任意軸の縮約を単一ループ構造で表現する
/// （`dim: None` の全軸縮約は呼び出し元がスカラー特別扱いする）。
///
/// **TASK-12.1d（#164）**: `Var::sum`/`Var::max`（`var.rs`）の実行は
/// `eval.rs` 直接呼び出しから `self.tape.ops().sum`/`max`（`BackendOps`
/// 経由）へ置き換えたため、本関数（および `sum`/`max`。下記）は
/// `Var::sum`/`max` の本番経路では呼ばれなくなった。ただし
/// **codex-review 第 19〜21 波・PR #403 の P1 是正（2026-08-08 追記）**
/// で `default_ops::NaiveOps`（`Tape::default()`／
/// `compat::Sequential::predict` 無引数版が使う compat 用
/// `BackendOps` 実装）がこの `sum`/`max` に委譲するようになったため、
/// `#[cfg(test)]` は外し本番ビルドにも含める。統合テストの数値微分
/// 突合（`test_support.rs`・`grad.rs` の VJP テスト）も引き続き同じ
/// 実装を使う（数式の実体を二重管理しない）。
fn reduce_axis(
    input: &Tensor<f32>,
    axis: usize,
    init: f32,
    op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let shape = input.shape();
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense_vec(input);
    let mut out = vec![init; outer * inner];
    for o in 0..outer {
        for a in 0..axis_len {
            for i in 0..inner {
                let src = (o * axis_len + a) * inner + i;
                let dst = o * inner + i;
                out[dst] = op(out[dst], data[src]);
            }
        }
    }
    out
}

/// `sum(dim)`。`dim: None` は全要素の総和をスカラー（shape `[]`）で返す。
/// `Var::sum` の本番経路（`BackendOps::sum`）からは呼ばれないが、
/// `default_ops::NaiveOps::sum`（compat 経路）とテスト（数値微分突合）が
/// 使う（上記 `reduce_axis` コメント参照。TASK-12.1d・#164）。
pub(crate) fn sum(input: &Tensor<f32>, dim: Option<usize>, out_shape: &[usize]) -> Tensor<f32> {
    match dim {
        None => {
            let total: f32 = dense_vec(input).into_iter().sum();
            build_tensor(vec![total], out_shape)
        }
        Some(axis) => build_tensor(reduce_axis(input, axis, 0.0, |a, b| a + b), out_shape),
    }
}

/// `max(dim)`。`dim: None` は全要素中の最大値をスカラー（shape `[]`）で
/// 返す。空テンソル（`numel() == 0`）は呼び出し元の `reduce_out_shape`
/// 検査を通過しうるが、そのケースでは `f32::NEG_INFINITY` を返す
/// （`fold` の初期値のまま。NumPy の `max` は空配列でエラーにするのが
/// 慣習だが、本イシューでは shape 検査のみをスコープとし数値的な特殊
/// ケースの扱いは #19（回帰テスト・数値突合）で確定する）。
/// `Var::max` の本番経路（`BackendOps::max`）からは呼ばれないが、
/// `default_ops::NaiveOps::max`（compat 経路）とテスト（数値微分突合）が
/// 使う（`reduce_axis` コメント参照。TASK-12.1d・#164）。
pub(crate) fn max(input: &Tensor<f32>, dim: Option<usize>, out_shape: &[usize]) -> Tensor<f32> {
    match dim {
        None => {
            let m = dense_vec(input)
                .into_iter()
                .fold(f32::NEG_INFINITY, nan_propagating_max);
            build_tensor(vec![m], out_shape)
        }
        Some(axis) => build_tensor(
            reduce_axis(input, axis, f32::NEG_INFINITY, nan_propagating_max),
            out_shape,
        ),
    }
}

/// 二乗誤差の縮約（スカラー出力）。shape 一致検査
/// （`require_same_shape`）は呼び出し元が済ませている前提。`reduction`
/// で mean（全要素平均）/sum（全要素総和）を切り替える（#190。
/// `Var::mse_loss_with`（`var.rs`）から呼ばれる）。`numel == 0` は
/// mean・sum とも 0.0 を返す（mean 側はゼロ除算回避、sum 側は空和が
/// 数学的に 0 のため元々の定義と一致）。
pub(crate) fn mse_loss(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    reduction: crate::var::Reduction,
) -> Tensor<f32> {
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let numel = pred_data.len();
    let sum_sq: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            diff * diff
        })
        .sum();
    let out = match reduction {
        crate::var::Reduction::Mean => {
            if numel == 0 {
                0.0
            } else {
                sum_sq / numel as f32
            }
        }
        crate::var::Reduction::Sum => sum_sq,
    };
    build_tensor(vec![out], &[])
}

/// RMSNorm（`x · rsqrt(mean(x²) + eps) · w`。`w` が `None` の場合は乗算を
/// スキップ）の行内統計（`mean`・`rstd`）を `f64` で計算する
/// （イシュー #1596）。`x_row` は 1 行分（長さ `hidden`）。
///
/// **縮約精度契約**（`.claude/rules/coding-rust.md`「正規化統計の二乗和
/// は要素を先に `f64` へ昇格してから二乗する」）: 二乗和は要素を
/// `f64` へ昇格してから `f64::mul_add` で二乗・蓄積し、`rstd` へ代入
/// する 1 回だけ `f32` へ downcast する（`backend-cpu::rmsnorm::
/// rmsnorm_row_scalar` と同じ縮約方式のホスト参照実装ミラー）。
/// `hidden == 0` は呼び出し元（[`rmsnorm_rows`]）が空出力として
/// 早期処理する契約のため、本関数は `hidden >= 1` を前提とする
/// （`inv_n` は呼び出し元が `1/hidden` を渡す）。
pub(crate) fn row_rms_stats(x_row: &[f32], eps: f32, inv_n: f64) -> f32 {
    let mut acc = 0.0f64;
    for &v in x_row {
        let v = v as f64;
        acc = v.mul_add(v, acc);
    }
    (1.0f64 / acc.mul_add(inv_n, eps as f64).sqrt()) as f32
}

/// LayerNorm（`(x − mean(x)) · rsqrt(var(x) + eps) · w + b`。分散は
/// biased ÷N）の行内統計（`mean`・`rstd`）を `f64` で計算する
/// （イシュー #1596）。`Var::layer_norm` のホスト参照実装
/// （`Unsupported` フォールバック）・`grad::vjp` の `Op::LayerNorm`
/// 逆伝播（統計再計算）の双方から呼ばれる。
///
/// 平均・分散とも [`warp_reduce_f64`] により GPU（CUDA／Metal）の
/// warp／simdgroup butterfly 縮約と同一の加算順序で蓄積する（単純な
/// 先頭からの逐次和ではない。PR #1671 codex-review P1 是正:
/// 相殺を含む入力で加算順序により結果が乖離するため。
/// `backend-cpu::layer_norm::warp_reduce_f64` doc comment 参照）。
/// [`row_rms_stats`] とは異なり二乗和ではなく「二パス分散」
/// （`Σ(x−μ)²`。`E[x²]−μ²` は使わない。実装計画 §3-3）のため専用
/// 関数とする。`hidden >= 1` を前提とする。
///
/// **`mean`／`var` は `sum`／`sq_acc` を `hidden` で直接除算して求める
/// （事前丸めした逆数 `1/hidden` との積ではない。codex-review 指摘:
/// `x=[1e30f32;49]` のような一様行で `sum * (1/hidden)` は 2 回の
/// 丸め〈逆数の丸め・乗算の丸め〉が複合し、本来 0 であるべき偏差
/// `x−mean` が巨大な非ゼロ値になり出力を歪める。IEEE 754 の
/// 除算は単一の正しく丸められた演算のため、`sum` が `hidden` 個の
/// 同一値の和である場合に丸め誤差を持ち込まない）。
///
/// **`rstd` も `f64` のまま返す**（`row_rms_stats` は `f32` downcast
/// 済みだが、LayerNorm は呼び出し元が `(x − mean)` を `f64` のまま
/// 減算する必要があり、早期に `mean`／`rstd` いずれかを `f32` へ丸める
/// と偏差計算の精度が損なわれる。codex-review 指摘: `x` の値域が
/// `f32` 仮数精度限界〈`2^24` 付近〉に達する入力で `mean` の
/// 早期丸めが出力を大きく歪める）。呼び出し元は `x̂` を書き出す
/// 直前の 1 回だけ `f32` へ downcast する。
pub(crate) fn row_ln_stats(x_row: &[f32], eps: f32, hidden: usize) -> (f64, f64) {
    let n = hidden as f64;
    let sum = warp_reduce_f64(hidden, |idx, acc| acc + x_row[idx] as f64);
    let mean = sum / n;
    let sq_acc = warp_reduce_f64(hidden, |idx, acc| {
        let d = x_row[idx] as f64 - mean;
        d.mul_add(d, acc)
    });
    let var = sq_acc / n;
    let rstd = 1.0f64 / (var + eps as f64).sqrt();
    (mean, rstd)
}

/// GPU（CUDA `kernels_layer_norm.rs`::`__shfl_xor_sync`／Metal
/// `layer_norm.metal`::`simd_shuffle_xor`）の warp／simdgroup 縮約と
/// **同一の演算順序**を再現する（`backend-cpu::layer_norm::
/// warp_reduce_f64` のホスト参照実装ミラー。PR #1671 codex-review P1
/// 指摘・イシュー #1596 是正）。詳細な背景（なぜ単純な逐次和ではなく
/// この順序へ揃えるか）は `backend-cpu::layer_norm` モジュール doc
/// comment・同名関数の doc comment を正本とし、ここでは二重管理しない。
///
/// [`row_ln_stats`] のみが使用する（[`row_rms_stats`] は二乗和のみで
/// 相殺が生じないため対象外。`docs/norm-ops-design.md`）。
fn warp_reduce_f64(hidden: usize, mut contribute: impl FnMut(usize, f64) -> f64) -> f64 {
    const LANES: usize = 32;
    let mut lanes = [0.0f64; LANES];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut idx = lane;
        while idx < hidden {
            *slot = contribute(idx, *slot);
            idx += LANES;
        }
    }
    let mut offset = 16usize;
    while offset > 0 {
        let snapshot = lanes;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            *slot = snapshot[lane] + snapshot[lane ^ offset];
        }
        offset >>= 1;
    }
    lanes[0]
}

/// RMSNorm のホスト参照実装（`BackendOps::rmsnorm` が
/// `Err(BackendError::Unsupported(_))` を返したときのみ `Var::rms_norm`
/// がフォールバックする。イシュー #1596。`docs/norm-ops-design.md`）。
///
/// `x` は `[rows, hidden]` の行優先 1 次元化済みテンソル、`w` を渡す
/// 場合は長さ `hidden` を要求する（呼び出し元 `var.rs::Var::rms_norm`
/// が shape 検査済み）。`rows == 0` または `hidden == 0` は空出力を
/// 返す（`backend-cpu::rmsnorm::run_rmsnorm_f32_raw` と同じ早期
/// return 契約）。
pub(crate) fn rmsnorm_rows(
    x: &Tensor<f32>,
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    if rows == 0 || hidden == 0 {
        return build_tensor(Vec::new(), &shape);
    }
    let data = dense_vec(x);
    let inv_n = 1.0f64 / hidden as f64;
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row = &data[r * hidden..(r + 1) * hidden];
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        let rstd = row_rms_stats(row, eps, inv_n);
        match w {
            Some(w) => {
                for ((o, &v), &wv) in out_row.iter_mut().zip(row.iter()).zip(w.iter()) {
                    *o = v * rstd * wv;
                }
            }
            None => {
                for (o, &v) in out_row.iter_mut().zip(row.iter()) {
                    *o = v * rstd;
                }
            }
        }
    }
    build_tensor(out, &shape)
}

/// LayerNorm のホスト参照実装（`BackendOps::layer_norm` が
/// `Err(BackendError::Unsupported(_))` を返したときのみ
/// `Var::layer_norm` がフォールバックする。イシュー #1596。
/// `docs/norm-ops-design.md`）。[`rmsnorm_rows`] と同じ shape 契約・
/// 早期 return 契約を持つ。`bias` は `w` と独立に `None` を取りうる
/// （`elementwise_affine=false` の `LayerNorm::without_affine` から
/// 呼ばれる場合等）。
pub(crate) fn layer_norm_rows(
    x: &Tensor<f32>,
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    if rows == 0 || hidden == 0 {
        return build_tensor(Vec::new(), &shape);
    }
    let data = dense_vec(x);
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row = &data[r * hidden..(r + 1) * hidden];
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        let (mean, rstd) = row_ln_stats(row, eps, hidden);
        // `mean`／`rstd` を `f64` のまま偏差計算まで保持し、`x̂` を書き出す
        // 直前の 1 回だけ `f32` へ downcast する（codex-review 指摘。
        // [`row_ln_stats`] doc 参照）。affine は CUDA カーネルの既定 FMA
        // contraction と揃えるため `w`／`b` がともに指定された場合のみ
        // `f32::mul_add` で明示的に融合する（`.claude/rules/coding-rust.md`
        // の FMA 契約統一）。`w`／`b` が `None` の演算は従来どおりスキップ
        // する（`-0.0` 等の符号付きゼロを不要な `+0.0` 加算で変えない）。
        match (w, b) {
            (Some(w), Some(b)) => {
                for (i, &v) in row.iter().enumerate() {
                    let xhat = ((v as f64 - mean) * rstd) as f32;
                    out_row[i] = xhat.mul_add(w[i], b[i]);
                }
            }
            (Some(w), None) => {
                for (i, &v) in row.iter().enumerate() {
                    let xhat = ((v as f64 - mean) * rstd) as f32;
                    out_row[i] = xhat * w[i];
                }
            }
            (None, Some(b)) => {
                for (i, &v) in row.iter().enumerate() {
                    let xhat = ((v as f64 - mean) * rstd) as f32;
                    out_row[i] = xhat + b[i];
                }
            }
            (None, None) => {
                for (i, &v) in row.iter().enumerate() {
                    out_row[i] = ((v as f64 - mean) * rstd) as f32;
                }
            }
        }
    }
    build_tensor(out, &shape)
}

/// `axis` に沿った数値安定形 softmax（シフト → `exp` → 正規化）。
/// `cross_entropy_loss`（forward。下記）の log-sum-exp 計算と
/// `grad.rs::cross_entropy_loss_vjp`（`softmax(x) − onehot(t)`）が同じ
/// 「シフトして exp・正規化する」実体を共有する（数式の実体を
/// forward/backward で二重実装しない方針。`grad.rs` 冒頭 doc）。
/// `pub(crate)`: `grad.rs` が VJP 計算で再利用する。
pub(crate) fn softmax_along(input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    // 要素数ゼロ（shape のいずれかの次元が 0）のとき、`shape[..axis]`／
    // `shape[axis+1..]` の部分積は数学的には無関係な次元（例:
    // `usize::MAX`）を含みうり、`checked_numel`（`Tensor::new` 側）が
    // 通した shape でも部分積単体では usize オーバーフローしうる
    // （全体積は途中の 0 で吸収されるが部分積はそれを経由しない）。
    // 本番経路 panic 禁止規約（`.claude/rules/coding-rust.md`）に従い、
    // outer/axis_len/inner を計算する前に空出力へ早期 return する。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense_vec(input);
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut m = f32::NEG_INFINITY;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                m = nan_propagating_max(m, data[idx]);
            }
            let mut sum_exp = 0f32;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let e = (data[idx] - m).exp();
                out[idx] = e;
                sum_exp += e;
            }
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                out[idx] /= sum_exp;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `axis` に沿った数値安定形 log_softmax（`x − m − ln(Σexp(x−m))`）。
/// `softmax_along`（直上）と同じ「シフト → exp → 縮約」走査構造を共有
/// するが、`ln(softmax_along(...))` へ委譲しない（`BackendOps::
/// log_softmax` doc「`ln(softmax(x))` にしない理由」参照: softmax が
/// アンダーフローで `0.0` になった要素の `ln(0.0) = -inf` を経由すると
/// 数値精度を落とすため、解析形で直接計算する）。`pub(crate)`:
/// `grad.rs` が VJP で・`var.rs` がホストフォールバックで再利用する。
pub(crate) fn log_softmax_along(input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    // `softmax_along` 直上と同じ早期 return（部分積オーバーフロー回避）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense_vec(input);
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut m = f32::NEG_INFINITY;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                m = nan_propagating_max(m, data[idx]);
            }
            let mut sum_exp = 0f32;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                sum_exp += (data[idx] - m).exp();
            }
            // `m + ln(sum_exp)` を先に加算してから `data[idx]` から引くと、
            // `m` が大きい（かつ `data[idx]` と近い）場合に `m` 自身の丸め
            // 精度で `ln(sum_exp)` の寄与が失われる（例: 全要素 1e8 のとき
            // `m + ln(sum_exp)` は `1e8` に丸まり `ln(2)` 分が消え、
            // `log_softmax` が `0.0`〈期待値 `-ln(2)`〉になる）。
            // `data[idx] - m` は Sterbenz の補題により丸め誤差なしで計算
            // できるため、先にこちらを計算してから `ln(sum_exp)` を引く
            // 順序（`(x - m) - ln(sum_exp)`）で丸め落ちを避ける。
            let ln_sum_exp = sum_exp.ln();
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                out[idx] = (data[idx] - m) - ln_sum_exp;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `inputs` を `dim` 軸で連結するホスト参照実装（`torch.cat` 相当。
/// イシュー #1598）。`BackendOps::concat` が `Unsupported` を返した
/// ときのみ `grad::concat_with_fallback` から呼ばれる（`softmax_along`
/// と同じ「バックエンド実装 → フォールバック」の二段構成。判定迂回
/// 経路を作らない）。
///
/// `out_shape` は呼び出し元（`Var::cat`／`grad::concat_with_fallback`）
/// が [`fandhe_ai_tensor_core::concat_out_shape`] で検査・確定済みの
/// 出力 shape をそのまま渡す（本関数は shape 再検査を行わない前提）。
/// `inputs` は strided view（`dense_vec_ref` で稠密化してから読む）で
/// よい——`Op::Narrow` の VJP（`grad::concat_with_fallback` 経由）が
/// zero-pad テンソルを渡す際、そのテンソル自体は contiguous のため
/// 実害はないが、`Var::cat` の入力が transpose 直後の view でも
/// 正しく動く契約とする。
///
/// レイアウト分解: `outer = prod(out_shape[..dim])`・
/// `inner = prod(out_shape[dim+1..])`・出力の線形添字は
/// `(o * total + off_i + s) * inner + i`
/// （`o`: outer 添字・`s`: 入力 i 内の dim 添字・`i`: inner 添字・
/// `off_i`: 入力 i より前の dim 累積長・`total = out_shape[dim]`）。
pub(crate) fn concat(inputs: &[&Tensor<f32>], dim: usize, out_shape: &[usize]) -> Tensor<f32> {
    // 要素数ゼロ（`out_shape` のいずれかの次元が 0）のとき、
    // `out_shape[..dim]`／`out_shape[dim+1..]` の部分積は数学的には
    // 無関係な次元（例: `usize::MAX`）を含みうり、`concat_out_shape`
    // が通した shape でも部分積単体では usize オーバーフローしうる
    // （全体積は途中の 0 で吸収されるが部分積はそれを経由しない）。
    // `softmax_along`（本ファイル上部）と同じ理由・同じ対処で、
    // 本番経路 panic 禁止規約（`.claude/rules/coding-rust.md`）に従い
    // outer/inner/out_numel を計算する前に空出力へ早期 return する。
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), out_shape);
    }
    let outer: usize = out_shape[..dim].iter().product();
    let inner: usize = out_shape[dim + 1..].iter().product();
    let total = out_shape[dim];
    let out_numel: usize = out_shape.iter().product();
    let mut out = vec![0f32; out_numel];
    if outer == 0 || inner == 0 || total == 0 {
        return build_tensor(out, out_shape);
    }
    let mut off = 0usize;
    for input in inputs {
        let seg = input.shape()[dim];
        if seg == 0 {
            continue;
        }
        let data = dense_vec_ref(input);
        for o in 0..outer {
            for s in 0..seg {
                let src_row_start = (o * seg + s) * inner;
                let dst_row_start = (o * total + off + s) * inner;
                out[dst_row_start..dst_row_start + inner]
                    .copy_from_slice(&data[src_row_start..src_row_start + inner]);
            }
        }
        off += seg;
    }
    build_tensor(out, out_shape)
}

/// 条件テンソルによる要素選択のホスト参照実装（`torch.where` 相当。
/// イシュー #1637）。`BackendOps::where_cond` が `Unsupported` を
/// 返したときのみ `Var::where_cond` から呼ばれる（`concat` と同じ
/// 「バックエンド実装 → フォールバック」二段構成）。
///
/// `cond`／`a`／`b` はいずれも呼び出し元（`Var::where_cond`）が
/// `out_shape` へ broadcast 済み（`cond` は f32 マスクへ変換済み）で
/// あることを前提とし、本関数は shape 再検査を行わない。真偽判定は
/// [`fandhe_ai_tensor_core::BackendOps::where_cond`] と同じ
/// `c != 0.0` 契約。`dense_vec_ref` で稠密化してから読むため、
/// strided view（broadcast view 等）でも正しく動く。
pub(crate) fn where_cond(
    cond: &Tensor<f32>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out_shape: &[usize],
) -> Tensor<f32> {
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), out_shape);
    }
    let cond_data = dense_vec_ref(cond);
    let a_data = dense_vec_ref(a);
    let b_data = dense_vec_ref(b);
    let out: Vec<f32> = cond_data
        .iter()
        .zip(a_data.iter())
        .zip(b_data.iter())
        .map(|((&c, &av), &bv)| if c != 0.0 { av } else { bv })
        .collect();
    build_tensor(out, out_shape)
}

/// マスク位置を定数で置換するホスト参照実装（`torch.masked_fill`
/// 相当。イシュー #1637）。`BackendOps::masked_fill` が
/// `Unsupported` を返したときのみ `Var::masked_fill` から呼ばれる。
///
/// `x`／`mask` は呼び出し元が同一 shape（`mask` は broadcast 済み
/// f32 マスク）であることを保証済み。`mask != 0.0` の位置を `value`
/// に置換し、それ以外は `x` の値をそのまま返す。
pub(crate) fn masked_fill(x: &Tensor<f32>, mask: &Tensor<f32>, value: f32) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let x_data = dense_vec_ref(x);
    let mask_data = dense_vec_ref(mask);
    let out: Vec<f32> = x_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&xv, &mv)| if mv != 0.0 { value } else { xv })
        .collect();
    build_tensor(out, &shape)
}

/// CrossEntropy 損失（log-sum-exp 安定化。クラス次元 `class_dim` 指定。
/// #191・親イシュー #189）。shape 検査（`class_dim` 範囲・targets
/// shape 一致・targets 添字範囲）は呼び出し元（`var.rs::
/// Var::cross_entropy_loss`）が済ませている前提。
///
/// `class_dim` を除いた添字の組（サンプル）ごとに
/// `loss = log_sum_exp(logits) − logits[target]`
/// （`= −log_softmax(logits)[target]`。オーバーフロー回避のシフト量
/// `m = max_c logits[c]` を経由するため大振幅入力でも有限値を保つ）を
/// 計算し、`reduction` で集約する。
///
/// 空バッチ（サンプル数 `N == 0`）は `mse_loss`（上記）の先例に合わせ
/// 0.0 を返す（PyTorch は `NaN`。差異は許容: #191 実装計画 §3.3）。
pub(crate) fn cross_entropy_loss(
    logits: &Tensor<f32>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
) -> Tensor<f32> {
    let shape = logits.shape().to_vec();
    let outer: usize = shape[..class_dim].iter().product();
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..].iter().product();
    let data = dense_vec(logits);
    let target_data = dense_vec_i32(targets);
    let n = outer * inner;

    let mut total = 0f32;
    for o in 0..outer {
        for i in 0..inner {
            let mut m = f32::NEG_INFINITY;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                m = nan_propagating_max(m, data[idx]);
            }
            let mut sum_exp = 0f32;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                sum_exp += (data[idx] - m).exp();
            }
            let lse = m + sum_exp.ln();
            let t = target_data[o * inner + i];
            // 呼び出し元（`var.rs::Var::cross_entropy_loss`）が
            // `0 <= t < axis_len` を検査済みの前提。範囲外は契約違反で
            // あり `unwrap()`/`expect()` を使わず `debug_assert!` で
            // 検知しつつ安全側（loss 寄与 0）へフォールバックする
            // （`.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
            let target_logit = if t >= 0 && (t as usize) < axis_len {
                data[(o * axis_len + t as usize) * inner + i]
            } else {
                debug_assert!(false, "cross_entropy_loss: target 添字が範囲外（契約違反）");
                lse
            };
            total += lse - target_logit;
        }
    }

    let loss = match reduction {
        Reduction::Mean if n > 0 => total / n as f32,
        Reduction::Mean => 0.0,
        Reduction::Sum => total,
    };
    build_tensor(vec![loss], &[])
}

// =====================================================================
// RNN／LSTM／GRU セル演算のホスト参照実装（イシュー #1647・設計
// `docs/autodiff-rnn-cell-tape-design.md` 決定 1・1b・1c・5・12）。
//
// `BackendOps::{lstm_pointwise,lstm_hidden_backward,lstm_cell_backward,
// gru_pointwise,gru_backward}`（`tensor-core::backend_ops`）が
// [`fandhe_ai_tensor_core::BackendError::Unsupported`] を返した場合の
// フォールバック（`var.rs::Var::{lstm_cell,gru_cell}` から呼ばれる。A08:
// `Unsupported` 以外のエラーは伝播し暗黙にはここへ来ない）。数式の正は
// 本モジュールであり、CPU／CUDA／Metal の各カーネル実装は同じ数式を
// バックエンド固有の並列化・FMA 契約で再実装する（`.claude/rules/
// coding-rust.md`）。
//
// ゲート配置（決定 5・PyTorch 準拠）: LSTM は列ブロック順 `i,f,g,o`
// （`pre: [B, 4H]`）、GRU は `r,z,n`（`pre_i`／`pre_h`: `[B, 3H]`）。
// =====================================================================

/// `pre: [B, G*H]` から `(b, gate, col)` の要素を読む（行優先連続データ
/// 前提。呼び出し元が `dense_vec` 済みのスライスを渡す）。
fn gate_elem(data: &[f32], gates: usize, hidden: usize, b: usize, gate: usize, j: usize) -> f32 {
    data[b * (gates * hidden) + gate * hidden + j]
}

/// LSTM セルの pointwise 段（決定 1・1b）参照実装。`pre: [B, 4H]`
/// （列ブロック順 `i,f,g,o`）・`c_prev: [B, H]` から `gates`（活性化後
/// `i,f,g,o`。`[B, 4H]`）・`c`（新セル状態。`[B, H]`）・`h`（新隠れ状態。
/// `[B, H]`）を計算する。`H` は `c_prev` の列数から導出する。
///
/// `c = f·c_prev + i·g`（`f32::mul_add` で FMA 契約統一）、
/// `h = o·tanh(c)`。
pub(crate) fn lstm_pointwise(pre: &Tensor<f32>, c_prev: &Tensor<f32>) -> LstmPointwiseOutput {
    let b_dim = c_prev.shape().first().copied().unwrap_or(0);
    let hidden = c_prev.shape().get(1).copied().unwrap_or(0);
    let pre_data = dense_vec(pre);
    let c_prev_data = dense_vec(c_prev);

    let mut gates = vec![0f32; b_dim * 4 * hidden];
    let mut c_out = vec![0f32; b_dim * hidden];
    let mut h_out = vec![0f32; b_dim * hidden];

    for b in 0..b_dim {
        for j in 0..hidden {
            let i_pre = gate_elem(&pre_data, 4, hidden, b, 0, j);
            let f_pre = gate_elem(&pre_data, 4, hidden, b, 1, j);
            let g_pre = gate_elem(&pre_data, 4, hidden, b, 2, j);
            let o_pre = gate_elem(&pre_data, 4, hidden, b, 3, j);

            let i_val = sigmoid_scalar(i_pre);
            let f_val = sigmoid_scalar(f_pre);
            let g_val = g_pre.tanh();
            let o_val = sigmoid_scalar(o_pre);

            let base = b * 4 * hidden;
            gates[base + j] = i_val;
            gates[base + hidden + j] = f_val;
            gates[base + 2 * hidden + j] = g_val;
            gates[base + 3 * hidden + j] = o_val;

            let c_prev_val = c_prev_data[b * hidden + j];
            let c_val = f_val.mul_add(c_prev_val, i_val * g_val);
            let h_val = o_val * c_val.tanh();
            c_out[b * hidden + j] = c_val;
            h_out[b * hidden + j] = h_val;
        }
    }

    LstmPointwiseOutput {
        gates: build_tensor(gates, &[b_dim, 4 * hidden]),
        c: build_tensor(c_out, &[b_dim, hidden]),
        h: build_tensor(h_out, &[b_dim, hidden]),
    }
}

/// [`Op::LstmHidden`] の VJP 補助（決定 1b・1b 追記）参照実装。
/// `d_pre_o = dh·tanh(c)·o·(1−o)`、`dc = dh·o·(1−tanh(c)²)`。
pub(crate) fn lstm_hidden_backward(
    c: &Tensor<f32>,
    gate_o: &Tensor<f32>,
    dh: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = c.shape().to_vec();
    let c_data = dense_vec(c);
    let o_data = dense_vec(gate_o);
    let dh_data = dense_vec(dh);

    let mut d_pre_o = vec![0f32; c_data.len()];
    let mut dc = vec![0f32; c_data.len()];
    for idx in 0..c_data.len() {
        let tanh_c = c_data[idx].tanh();
        let o_val = o_data[idx];
        let dh_val = dh_data[idx];
        d_pre_o[idx] = dh_val * tanh_c * o_val * (1.0 - o_val);
        dc[idx] = dh_val * o_val * (1.0 - tanh_c * tanh_c);
    }
    (build_tensor(d_pre_o, &shape), build_tensor(dc, &shape))
}

/// [`Op::LstmCell`] の VJP 補助（決定 1b）参照実装。`gates_ifg: [B, 3H]`
/// （活性化後の `i,f,g`）・`c_prev: [B, H]`・`dc: [B, H]` から
/// `d_pre_ifg: [B, 3H]`・`dc_prev: [B, H]` を計算する。
///
/// `d_pre_i = dc·g·i·(1−i)`、`d_pre_f = dc·c_prev·f·(1−f)`、
/// `d_pre_g = dc·i·(1−g²)`、`dc_prev = dc·f`。
pub(crate) fn lstm_cell_backward(
    gates_ifg: &Tensor<f32>,
    c_prev: &Tensor<f32>,
    dc: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    let b_dim = c_prev.shape().first().copied().unwrap_or(0);
    let hidden = c_prev.shape().get(1).copied().unwrap_or(0);
    let gates_data = dense_vec(gates_ifg);
    let c_prev_data = dense_vec(c_prev);
    let dc_data = dense_vec(dc);

    let mut d_pre_ifg = vec![0f32; b_dim * 3 * hidden];
    let mut dc_prev = vec![0f32; b_dim * hidden];
    for b in 0..b_dim {
        for j in 0..hidden {
            let i_val = gate_elem(&gates_data, 3, hidden, b, 0, j);
            let f_val = gate_elem(&gates_data, 3, hidden, b, 1, j);
            let g_val = gate_elem(&gates_data, 3, hidden, b, 2, j);
            let c_prev_val = c_prev_data[b * hidden + j];
            let dc_val = dc_data[b * hidden + j];

            let base = b * 3 * hidden;
            d_pre_ifg[base + j] = dc_val * g_val * i_val * (1.0 - i_val);
            d_pre_ifg[base + hidden + j] = dc_val * c_prev_val * f_val * (1.0 - f_val);
            d_pre_ifg[base + 2 * hidden + j] = dc_val * i_val * (1.0 - g_val * g_val);
            dc_prev[b * hidden + j] = dc_val * f_val;
        }
    }

    (
        build_tensor(d_pre_ifg, &[b_dim, 3 * hidden]),
        build_tensor(dc_prev, &[b_dim, hidden]),
    )
}

/// GRU セルの pointwise 段（決定 1c・5。`reset_after=True` 規約）参照
/// 実装。`pre_i`／`pre_h: [B, 3H]`（列ブロック順 `r,z,n`）・
/// `h_prev: [B, H]` から `gates`（活性化後 `r,z,n`。`[B, 3H]`）・`q`
/// （再帰側アフィン値 `pre_h` の n 列ブロック。`[B, H]`）・`h`（新隠れ
/// 状態。`[B, H]`）を計算する。
///
/// `r = σ(pre_i_r + pre_h_r)`、`z = σ(pre_i_z + pre_h_z)`、
/// `q = pre_h_n`、`n = tanh(r·q + pre_i_n)`、
/// `h = z·h_prev + (1−z)·n`。
pub(crate) fn gru_pointwise(
    pre_i: &Tensor<f32>,
    pre_h: &Tensor<f32>,
    h_prev: &Tensor<f32>,
) -> GruPointwiseOutput {
    let b_dim = h_prev.shape().first().copied().unwrap_or(0);
    let hidden = h_prev.shape().get(1).copied().unwrap_or(0);
    let pre_i_data = dense_vec(pre_i);
    let pre_h_data = dense_vec(pre_h);
    let h_prev_data = dense_vec(h_prev);

    let mut gates = vec![0f32; b_dim * 3 * hidden];
    let mut q_out = vec![0f32; b_dim * hidden];
    let mut h_out = vec![0f32; b_dim * hidden];

    for b in 0..b_dim {
        for j in 0..hidden {
            let r_pre = gate_elem(&pre_i_data, 3, hidden, b, 0, j)
                + gate_elem(&pre_h_data, 3, hidden, b, 0, j);
            let z_pre = gate_elem(&pre_i_data, 3, hidden, b, 1, j)
                + gate_elem(&pre_h_data, 3, hidden, b, 1, j);
            let q_val = gate_elem(&pre_h_data, 3, hidden, b, 2, j);
            let pre_i_n = gate_elem(&pre_i_data, 3, hidden, b, 2, j);

            let r_val = sigmoid_scalar(r_pre);
            let z_val = sigmoid_scalar(z_pre);
            let n_val = r_val.mul_add(q_val, pre_i_n).tanh();

            let base = b * 3 * hidden;
            gates[base + j] = r_val;
            gates[base + hidden + j] = z_val;
            gates[base + 2 * hidden + j] = n_val;
            q_out[b * hidden + j] = q_val;

            let h_prev_val = h_prev_data[b * hidden + j];
            h_out[b * hidden + j] = z_val.mul_add(h_prev_val, (1.0 - z_val) * n_val);
        }
    }

    GruPointwiseOutput {
        gates: build_tensor(gates, &[b_dim, 3 * hidden]),
        q: build_tensor(q_out, &[b_dim, hidden]),
        h: build_tensor(h_out, &[b_dim, hidden]),
    }
}

/// [`Op::GruCell`] の VJP 補助参照実装。`gates_rzn: [B, 3H]`（活性化後の
/// `r,z,n`）・`q: [B, H]`（決定 1c）・`h_prev: [B, H]`・`dh: [B, H]` から
/// `d_pre_i: [B, 3H]`・`d_pre_h: [B, 3H]`・`dh_prev_direct: [B, H]` を
/// 計算する。
///
/// `dn = dh·(1−z)`、`dz = dh·(h_prev−n)`、`dh_prev_direct = dh·z`、
/// `d_pre_n = dn·(1−n²)`、`dr = d_pre_n·q`、`d_pre_r = dr·r·(1−r)`、
/// `d_pre_z = dz·z·(1−z)`。`d_pre_i = [d_pre_r, d_pre_z, d_pre_n]`、
/// `d_pre_h = [d_pre_r, d_pre_z, d_pre_n·r]`（`n` の `q` に対する偏微分が
/// `r` であるため、`n` 列ブロックのみ追加で `r` を乗じる）。
pub(crate) fn gru_backward(
    gates_rzn: &Tensor<f32>,
    q: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    dh: &Tensor<f32>,
) -> GruBackwardOutput {
    let b_dim = h_prev.shape().first().copied().unwrap_or(0);
    let hidden = h_prev.shape().get(1).copied().unwrap_or(0);
    let gates_data = dense_vec(gates_rzn);
    let q_data = dense_vec(q);
    let h_prev_data = dense_vec(h_prev);
    let dh_data = dense_vec(dh);

    let mut d_pre_i = vec![0f32; b_dim * 3 * hidden];
    let mut d_pre_h = vec![0f32; b_dim * 3 * hidden];
    let mut dh_prev_direct = vec![0f32; b_dim * hidden];

    for b in 0..b_dim {
        for j in 0..hidden {
            let r_val = gate_elem(&gates_data, 3, hidden, b, 0, j);
            let z_val = gate_elem(&gates_data, 3, hidden, b, 1, j);
            let n_val = gate_elem(&gates_data, 3, hidden, b, 2, j);
            let q_val = q_data[b * hidden + j];
            let h_prev_val = h_prev_data[b * hidden + j];
            let dh_val = dh_data[b * hidden + j];

            let dn = dh_val * (1.0 - z_val);
            let dz = dh_val * (h_prev_val - n_val);
            let d_pre_n = dn * (1.0 - n_val * n_val);
            let dr = d_pre_n * q_val;
            let d_pre_r = dr * r_val * (1.0 - r_val);
            let d_pre_z = dz * z_val * (1.0 - z_val);

            let base = b * 3 * hidden;
            d_pre_i[base + j] = d_pre_r;
            d_pre_i[base + hidden + j] = d_pre_z;
            d_pre_i[base + 2 * hidden + j] = d_pre_n;

            d_pre_h[base + j] = d_pre_r;
            d_pre_h[base + hidden + j] = d_pre_z;
            d_pre_h[base + 2 * hidden + j] = d_pre_n * r_val;

            dh_prev_direct[b * hidden + j] = dh_val * z_val;
        }
    }

    (
        build_tensor(d_pre_i, &[b_dim, 3 * hidden]),
        build_tensor(d_pre_h, &[b_dim, 3 * hidden]),
        build_tensor(dh_prev_direct, &[b_dim, hidden]),
    )
}

#[cfg(test)]
mod dense_vec_ref_tests {
    use super::*;

    // イシュー #1026「学習ループのホスト側コピー・再構築を除去する」の
    // 機械的な回帰検証（advisor 助言: `dense_vec_ref` は `MemoryOps`
    // 境界を持たないため `AllocationTracker` ではコピー回数を数えられ
    // ない。ここでは「返した `Cow` が呼び出し元の `Tensor` のバッファを
    // 直接指している（ポインタ一致）」ことを確認することで、
    // `slice.to_vec()` によるヒープコピーが発生していないことを機械的に
    // 検証する）。

    #[test]
    fn contiguous_input_borrows_without_copy() {
        let tensor = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let borrowed = dense_vec_ref(&tensor);

        assert!(
            matches!(borrowed, std::borrow::Cow::Borrowed(_)),
            "contiguous な入力は Cow::Borrowed（コピーなし）を返す契約"
        );
        // ポインタ一致で「元の `Tensor` のバッファをそのまま指している」
        // ことを確認する（`to_vec()` していれば別のヒープ確保になり
        // ポインタが一致しない）。
        let original_ptr = tensor
            .as_slice()
            .expect("test fixture: contiguous")
            .as_ptr();
        assert_eq!(borrowed.as_ptr(), original_ptr);
        assert_eq!(&*borrowed, &[1.0, 2.0, 3.0, 4.0][..]);
    }

    #[test]
    fn non_contiguous_input_falls_back_to_owned_dense_vec() {
        // transpose 済み view は非 contiguous になるため `as_slice()` が
        // `None` を返す（`tensor.rs` doc）。`dense_vec_ref` は `dense_vec`
        // へフォールバックし、値は一致するが所有権を持つ `Cow::Owned` を
        // 返す契約。
        let tensor = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let transposed = tensor
            .transpose(0, 1)
            .expect("test fixture: 2 次元 tensor の transpose(0, 1) は常に成功する");
        assert!(
            transposed.as_slice().is_none(),
            "test fixture: transpose 後は非 contiguous であることが前提"
        );

        let owned = dense_vec_ref(&transposed);
        assert!(
            matches!(owned, std::borrow::Cow::Owned(_)),
            "非 contiguous な入力は Cow::Owned（dense_vec フォールバック）を返す契約"
        );
        assert_eq!(&*owned, &dense_vec(&transposed)[..]);
    }
}

#[cfg(test)]
mod norm_rows_tests {
    //! [`rmsnorm_rows`]／[`layer_norm_rows`]（イシュー #1596）のホスト
    //! 参照実装単体テスト。`backend-cpu` の実機カーネルとの parity は
    //! `crates/backend-cpu/tests/{rmsnorm,layer_norm}_parity.rs` が担う
    //! （本モジュールは `eval.rs` 自体の正しさのみを検証する）。

    use super::*;

    #[test]
    fn rmsnorm_rows_basic_no_weight() {
        // hidden=4, x=[1,2,3,4] -> mean(x^2)=(1+4+9+16)/4=7.5
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let out = rmsnorm_rows(&x, None, 0.0, 1, 4);
        let rstd = 1.0f32 / 7.5f32.sqrt();
        for (o, v) in dense_vec(&out).iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!((o - v * rstd).abs() < 1e-5);
        }
    }

    #[test]
    fn rmsnorm_rows_applies_weight() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = [2.0f32, 1.0, 0.5, 1.0];
        let out = dense_vec(&rmsnorm_rows(&x, Some(&w), 0.0, 1, 4));
        let rstd = 1.0f32 / 7.5f32.sqrt();
        let expected = [1.0 * rstd * 2.0, 2.0 * rstd, 3.0 * rstd * 0.5, 4.0 * rstd];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5);
        }
    }

    #[test]
    fn rmsnorm_rows_empty_rows_or_hidden_is_empty_output() {
        let x0 = Tensor::new(Vec::<f32>::new(), &[0, 4]).unwrap();
        assert_eq!(
            dense_vec(&rmsnorm_rows(&x0, None, 1e-5, 0, 4)),
            Vec::<f32>::new()
        );
        let x1 = Tensor::new(Vec::<f32>::new(), &[3, 0]).unwrap();
        assert_eq!(
            dense_vec(&rmsnorm_rows(&x1, None, 1e-5, 3, 0)),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn rmsnorm_rows_nan_propagates() {
        let x = Tensor::new(vec![f32::NAN, 1.0, 1.0, 1.0], &[1, 4]).unwrap();
        let out = dense_vec(&rmsnorm_rows(&x, None, 1e-5, 1, 4));
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn layer_norm_rows_matches_manual_computation() {
        // x = [1, 2, 3, 4] -> mean=2.5, var=Sigma(x-2.5)^2/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let out = dense_vec(&layer_norm_rows(&x, None, None, 0.0, 1, 4));
        let rstd = 1.0f32 / 1.25f32.sqrt();
        let expected = [-1.5 * rstd, -0.5 * rstd, 0.5 * rstd, 1.5 * rstd];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    /// PR #1671 codex-review P1 指摘（イシュー #1596）の反例を
    /// ホスト参照実装側で再現する回帰テスト。詳細は
    /// `backend-cpu::layer_norm::tests::
    /// run_layer_norm_f32_matches_gpu_butterfly_order_on_cancelling_row`
    /// の doc comment を参照（両実装は同じ `warp_reduce_f64` 順序を
    /// 使うため同じ期待値になる）。
    #[test]
    fn layer_norm_rows_matches_gpu_butterfly_order_on_cancelling_row() {
        let x = Tensor::new(vec![1e30f32, 1.0, -1e30, 0.0], &[1, 4]).unwrap();
        let w = [1.0f32, 1.0, 1.0, 1e30];
        let out = dense_vec(&layer_norm_rows(&x, Some(&w), None, 1e-5, 1, 4));

        let expected_out3 = -std::f64::consts::SQRT_2 / 4.0;
        assert!(
            (out[3] as f64 - expected_out3).abs() < 1e-3,
            "out[3]={} expected~={expected_out3}",
            out[3]
        );
        assert_ne!(out[3], 0.0, "単純な逐次和への後退の可能性がある");
    }

    #[test]
    fn layer_norm_rows_applies_weight_and_bias() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = [2.0f32, 1.0, 1.0, 0.5];
        let b = [1.0f32, 0.0, -1.0, 2.0];
        let out = dense_vec(&layer_norm_rows(&x, Some(&w), Some(&b), 0.0, 1, 4));
        let rstd = 1.0f32 / 1.25f32.sqrt();
        let xhat = [-1.5 * rstd, -0.5 * rstd, 0.5 * rstd, 1.5 * rstd];
        let expected = [
            xhat[0] * 2.0 + 1.0,
            xhat[1] * 1.0 + 0.0,
            xhat[2] * 1.0 - 1.0,
            xhat[3] * 0.5 + 2.0,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn layer_norm_rows_empty_rows_or_hidden_is_empty_output() {
        let x0 = Tensor::new(Vec::<f32>::new(), &[0, 4]).unwrap();
        assert_eq!(
            dense_vec(&layer_norm_rows(&x0, None, None, 1e-5, 0, 4)),
            Vec::<f32>::new()
        );
        let x1 = Tensor::new(Vec::<f32>::new(), &[3, 0]).unwrap();
        assert_eq!(
            dense_vec(&layer_norm_rows(&x1, None, None, 1e-5, 3, 0)),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn layer_norm_rows_nan_propagates() {
        let x = Tensor::new(vec![f32::NAN, 1.0, 1.0, 1.0], &[1, 4]).unwrap();
        let out = dense_vec(&layer_norm_rows(&x, None, None, 1e-5, 1, 4));
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn layer_norm_rows_extreme_scale_does_not_overflow_stats() {
        // f64 promotion before squaring avoids overflow at this scale
        // (coding-rust.md normalization-stat accumulator contract).
        let x = Tensor::new(vec![2e20f32, -2e20, 2e20, -2e20], &[1, 4]).unwrap();
        let out = dense_vec(&layer_norm_rows(&x, None, None, 1e-5, 1, 4));
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }
}

#[cfg(test)]
mod reduce_bias_grad_rows_tests {
    use super::*;

    // イシュー #1566・PR #1659 codex-review P1 是正（2026-09-12 ユーザー
    // 承認 A）: `reduce_bias_grad_rows` は `f64` アキュムレータで列ごと
    // に蓄積するため、単純な `f32` 逐次 `+=` なら桁落ちで消える寄与
    // （`1e8 + 1.0 + (-1e8)` の `1.0`）が保持されることを確認する
    // （`.claude/rules/coding-rust.md` の勾配長軸縮約 f64 方針）。
    #[test]
    fn preserves_cancelling_contribution_via_f64_accumulator() {
        // 列 0: 1e8 + 1.0 + (-1e8) は f32 逐次和だと桁落ちで 1.0 の
        // 寄与が失われ 0.0 になる（以前の実装の回帰記録は git 履歴
        // 参照）が、f64 アキュムレータでは 1.0 が正しく残る。
        let g = Tensor::<f32>::new(vec![1.0e8, 10.0, 1.0, 20.0, -1.0e8, 30.0], &[3, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);

        // 参考: 素朴な f32 逐次和では 1.0 の寄与が失われることの確認
        // （contrast のための計算。got との比較には使わない）。
        let mut naive_f32_col0 = 0.0f32;
        naive_f32_col0 += 1.0e8;
        naive_f32_col0 += 1.0;
        naive_f32_col0 += -1.0e8;
        assert_eq!(
            naive_f32_col0, 0.0,
            "対照: f32 逐次和では桁落ちにより 1.0 の寄与が失われる"
        );

        assert_eq!(got.len(), 2);
        assert_eq!(
            got[0], 1.0,
            "f64 アキュムレータでは 1e8 + 1.0 + (-1e8) の 1.0 が保持されるはず"
        );
        assert_eq!(got[1], 60.0);
    }

    #[test]
    fn preserves_negative_zero_and_nan_and_inf() {
        let g = Tensor::<f32>::new(
            vec![-0.0, f32::NAN, f32::INFINITY, 1.0, -0.0, f32::NEG_INFINITY],
            &[3, 2],
        )
        .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);
        assert_eq!(got.len(), 2);
        // row0=[-0.0, NaN]・row1=[+inf, 1.0]・row2=[-0.0, -inf]
        // （data は row-major: [row0col0, row0col1, row1col0, ...]）。
        // col0: -0.0 + (+inf) + -0.0 = +inf
        assert!(got[0].is_infinite() && got[0] > 0.0);
        // col1: NaN + 1.0 + -inf = NaN（NaN の伝播）
        assert!(got[1].is_nan());
    }

    #[test]
    fn single_row_returns_row_unchanged() {
        let g = Tensor::<f32>::new(vec![1.5, -2.5, 3.5], &[1, 3])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);
        assert_eq!(got, vec![1.5f32, -2.5, 3.5]);
    }

    // PR #1659 codex-review P2 是正の回帰テスト（`reduce_bias_grad_rows`
    // doc「`m == 1` の特殊扱い」）: `m == 1` の単純な `+=` 版
    // （`0.0f32 + (-0.0f32) == +0.0f32`）だと符号付きゼロが失われる
    // ことを直接検知する（`is_sign_negative` で `+0.0`/`-0.0` を区別）。
    #[test]
    fn single_row_preserves_negative_zero_sign() {
        let g = Tensor::<f32>::new(vec![-0.0f32, 0.0f32], &[1, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);
        assert_eq!(got.len(), 2);
        assert!(
            got[0].is_sign_negative(),
            "m == 1 では -0.0 の符号を保持するはず（reduce_to_shape との bit 完全一致契約）"
        );
        assert!(!got[1].is_sign_negative());
    }
}

#[cfg(test)]
mod log_softmax_along_precision_tests {
    use super::*;

    // codex-review 指摘（PR #1664）の回帰検証: `m + ln(sum_exp)` を
    // 先に加算してから `x` から引く実装では、`m` が大きい共通オフセット
    // を持つ入力で丸め落ちが発生し、`log_softmax([1e8, 1e8])` が
    // 期待値 `[-ln(2), -ln(2)]` ではなく `[0.0, 0.0]` になっていた
    // （`m + ln(2)` が `f32` の丸め精度で `m` そのものに丸まるため）。
    // `(x - m) - ln(sum_exp)` の順で計算することで `x - m` を Sterbenz
    // の補題により誤差なく求め、丸め落ちを避ける。
    #[test]
    fn large_common_offset_does_not_round_away_ln_sum_exp() {
        let input = Tensor::<f32>::new(vec![1e8, 1e8], &[1, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let out = log_softmax_along(&input, 1);
        let expected = -(2.0f32).ln();
        for c in 0..2 {
            let v = out.get(&[0, c]).unwrap();
            assert!(
                (v - expected).abs() < 1e-4,
                "log_softmax([1e8,1e8])[{c}] = {v}（期待値 {expected} 近傍）"
            );
        }
    }
}

#[cfg(test)]
mod softmax_empty_tensor_overflow_tests {
    use super::*;

    // codex-review 指摘（PR #1664）の回帰検証: `Tensor::new(vec![],
    // &[0, 0, usize::MAX, 2])` は `checked_numel` が要素数積を `0`
    // （先頭の `0` が後続の積を吸収する）と評価するため構築できるが、
    // `log_softmax_along(input, 1)` の `inner = shape[2..].iter()
    // .product()`（`= usize::MAX * 2`）はこの吸収を経由しない部分積
    // のため、overflow チェック有効時に本番経路の外で panic していた。
    // `softmax_along`／`log_softmax_along` 冒頭の早期 return
    // （`shape` がいずれかの次元 `0` を含めば空出力を返す）で、
    // 部分積を計算する前に安全側へ倒れることを確認する。
    #[test]
    fn log_softmax_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let input = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let out = log_softmax_along(&input, 1);
        assert_eq!(out.shape(), &shape);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn softmax_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let input = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let out = softmax_along(&input, 1);
        assert_eq!(out.shape(), &shape);
        assert_eq!(out.numel(), 0);
    }
}

#[cfg(test)]
mod concat_empty_out_shape_overflow_tests {
    use super::*;

    // codex-review 指摘（PR #1680）の回帰検証: `concat_out_shape` が
    // 受理しうる有効な空 `out_shape`（先頭が `0` で後続次元の部分積が
    // overflow するケース）に対し、`concat` が `outer`／`inner` を
    // ゼロ軸チェックより先に `.iter().product()` で計算していたため、
    // overflow チェック有効時に本番経路の外（debug ビルド）で panic
    // していた（`softmax_along` と同型の bug。上記
    // `softmax_empty_tensor_overflow_tests` 参照）。`out_shape` に `0`
    // を含む場合は部分積を計算する前に空出力へ早期 return することを
    // 確認する（`dim=0` のとき `inner = out_shape[1..].iter().product()`
    // `= usize::MAX * 2` が旧実装で overflow していた）。
    #[test]
    fn concat_empty_out_shape_with_overflow_prone_inner_does_not_panic() {
        let out_shape = [0usize, usize::MAX, 2];
        let out = concat(&[], 0, &out_shape);
        assert_eq!(out.shape(), &out_shape);
        assert_eq!(out.numel(), 0);
    }
}
