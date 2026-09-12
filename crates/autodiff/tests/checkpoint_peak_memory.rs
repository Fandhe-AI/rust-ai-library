//! 受け入れ条件「activation checkpointing 有効時、forward+backward の
//! 純増分ピークメモリが非 checkpoint より小さい」（イシュー #1624・
//! `docs/autodiff-checkpoint-design.md` §6 (c)）を
//! `bench_harness::alloc_tracker::TrackingAllocator`（`#[global_allocator]`
//! フック）で機械的に実測する統合テスト。`view_zero_alloc.rs` と同じ
//! `harness = false` プロセス分離方式（`crates/autodiff/Cargo.toml`
//! `[[test]] name = "checkpoint_peak_memory"`）を採用する理由も同一
//! （`TrackingAllocator` はプロセス全体で共有される静的カウンタを持ち、
//! libtest 既定の並列ディスパッチ下では他テストの確保・解放が計測区間へ
//! 混入する）。
//!
//! ## 実測方針
//!
//! `K` 個の checkpoint 区間を連鎖させる（各区間: `matmul → sigmoid →
//! matmul → sigmoid`。1 区間あたり 3 個の中間ノード〈1 回目の
//! matmul・1 回目の sigmoid・2 回目の matmul〉と 1 個の区間出力
//! ノード〈2 回目の sigmoid〉を持つ）。
//!
//! - **checkpoint なし**: forward が進むにつれ全区間の中間ノード
//!   （`3 * K` 個）と区間出力（`K` 個）がすべて `TapeNode::value` に
//!   残り続ける（`Tape::reset()`／`drop` まで解放されない）。
//! - **checkpoint あり**: 各区間の中間ノード（3 個）は区間終了直後に
//!   解放される（`Tape::checkpoint` → `release_checkpoint_region`）。
//!   区間出力（次区間の入力になるため解放不可）は変わらず `K` 個
//!   残るが、中間ノード分（`3 * K` 個）のピークは避けられる。
//!
//! よって `peak_checkpoint < peak_plain` を機械的に検証する。
//! **`Gradients` が全ノード分の `Vec<Option<Tensor<f32>>>` を保持する
//! ため、削減できるのは中間ノードの forward 値のみ（勾配・区間出力の
//! forward 値は checkpoint 有無に関わらず残る）**——期待削減幅は
//! おおよそ「中間ノード数 `3 * K` 個分の活性化サイズ」であり、全 K
//! 区間分の活性化がまるごと消えるわけではない旨をログにも明記する
//! （`docs/autodiff-checkpoint-design.md` §8 のスコープ外事項「`Gradients`
//! からの非葉勾配の早期解放」参照）。

mod common;

use bench_harness::alloc_tracker::TrackingAllocator;
use bench_harness::alloc_tracker::measure;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// 正方行列の一辺。`N * N * size_of::<f32>()` = 256 * 256 * 4 =
/// 256 KiB を 1 個の活性化サイズとする（中間ノード・区間出力とも
/// 同 shape で連鎖するため、全ノードがこのサイズ）。ノード管理の
/// オーバーヘッド（`Vec` 成長等。高々数 KiB）に対して活性化サイズが
/// 十分大きくなるよう選んだ（`view_zero_alloc.rs` の `N=2048` と同じ
/// 考え方だが、本テストは `K` 個の行列を保持するため小さめに取る）。
const N: usize = 256;

/// 連鎖させる checkpoint 区間数。区間ごとに中間ノード 3 個・区間出力
/// 1 個を生む（本ファイル冒頭コメント参照）。`K` が大きいほど
/// 「中間ノード解放」の効果（`3 * K` 個分）が「区間出力の必須保持」
/// （`K` 個分）に対して相対的に大きくなり、判定の余裕（マージン）が
/// 増す。
const K: usize = 6;

fn activation_bytes() -> u64 {
    (N * N * std::mem::size_of::<f32>()) as u64
}

fn make_matrix(seed: u64) -> Tensor<f32> {
    // 決定的な擬似乱数（線形合同法。`.claude/rules/coding-rust.md`
    // 「学習系回帰テストには決定的シード設定ユーティリティを使う」の
    // 精神に沿い、外部乱数クレートを追加せず自前で十分小さい値を生成
    // する）。値域を [-0.05, 0.05) に絞り、matmul/sigmoid の連鎖を
    // `K=6` 回重ねても発散しない（sigmoid が出力を (0,1) へ押し込める
    // ため実際には発散しないが、中間 matmul の値も程よい範囲に保つ）。
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let data: Vec<f32> = (0..N * N)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (state >> 40) as u32; // 上位 24bit 相当を使う
            ((bits as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect();
    Tensor::new(data, &[N, N])
        .expect("checkpoint_peak_memory: test fixture: shape とデータ長は事前に一致させている")
}

/// 1 区間分の計算本体（`matmul → sigmoid → matmul → sigmoid`）。
/// `Tape::checkpoint`／非 checkpoint のどちらからも同一関数を呼ぶ
/// ことで、区間の境界以外の挙動を完全に揃える。
fn region_compute<'t>(h: &Var<'t>, w_a: &Var<'t>, w_b: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let a = h.matmul(w_a)?; // 中間 1
    let b = a.sigmoid(); // 中間 2
    let c = b.matmul(w_b)?; // 中間 3
    Ok(c.sigmoid()) // 区間出力
}

/// `K` 個の区間を連鎖させ、`use_checkpoint` に応じて `Tape::checkpoint`
/// で包む／包まないを切り替えて forward → backward を実行する。戻り値
/// は使わず（`measure` の計測対象を forward+backward 全体にするため
/// 呼び出し元で `black_box` する）、テープ自体を返して呼び出し元が
/// 破棄タイミングを制御できるようにする。
fn run_chain(tape: &Tape, use_checkpoint: bool) {
    let leaf = tape.var(&make_matrix(0));
    let mut h = leaf;
    let weights_a: Vec<_> = (0..K)
        .map(|i| tape.var(&make_matrix(2 * i as u64 + 1)))
        .collect();
    let weights_b: Vec<_> = (0..K)
        .map(|i| tape.var(&make_matrix(2 * i as u64 + 2)))
        .collect();
    for i in 0..K {
        let w_a = &weights_a[i];
        let w_b = &weights_b[i];
        h = if use_checkpoint {
            tape.checkpoint(|| region_compute(&h, w_a, w_b))
                .expect("checkpoint 区間の forward は常に成功する構成")
        } else {
            region_compute(&h, w_a, w_b).expect("forward は常に成功する構成")
        };
    }
    let loss = h.sum(None).expect("sum(None) は常に成功する（全軸縮約）");
    let grads = tape.backward(&loss).expect("backward は常に成功する構成");
    std::hint::black_box(&grads);
}

fn check_checkpoint_reduces_peak_memory() {
    let (_plain, peak_plain) = measure(|| {
        let tape = Tape::new_with_ops(common::naive_ops());
        run_chain(&tape, false);
        tape
    });
    let (_ckpt, peak_ckpt) = measure(|| {
        let tape = Tape::new_with_ops(common::naive_ops());
        run_chain(&tape, true);
        tape
    });

    let peak_plain = peak_plain
        .expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");
    let peak_ckpt = peak_ckpt
        .expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");

    // 期待削減幅（本ファイル冒頭コメント参照）: 中間ノード `3 * K` 個分の
    // 活性化サイズ。`Gradients` の全ノード分 `Vec<Option<Tensor<f32>>>`
    // 保持（勾配自体は checkpoint 有無で変わらない）により、削減幅は
    // これより小さくなりうるため、判定閾値は緩め（半分以上の削減）に
    // 取る。
    let expected_reduction = activation_bytes() * (3 * K) as u64;
    println!(
        "checkpoint_peak_memory: peak_plain={peak_plain} bytes, peak_ckpt={peak_ckpt} bytes, \
         activation_bytes={} bytes, expected_reduction(理論値)={expected_reduction} bytes, \
         K={K}",
        activation_bytes()
    );
    assert!(
        peak_ckpt < peak_plain,
        "checkpoint ありのピーク（{peak_ckpt} バイト）が checkpoint なしのピーク（{peak_plain} \
         バイト）を下回らなかった——activation checkpointing による解放が機能していない疑い"
    );

    let actual_reduction = peak_plain - peak_ckpt;
    assert!(
        actual_reduction > expected_reduction / 2,
        "checkpoint による削減幅（{actual_reduction} バイト）が理論値の半分（{} バイト）に \
         満たない——解放が一部の区間でしか機能していない、または `Gradients` 側の保持が \
         想定より大きい疑い",
        expected_reduction / 2
    );
}

/// review 指摘（イシュー #1624 Review。`docs/autodiff-checkpoint-design.md`
/// §3.1 点 4 の契約検証）: checkpoint 区間の `lo`（区間内で最初に push
/// されたノード）が loss への勾配経路上にない（使い捨ての中間値）場合
/// でも、backward の再解放（`Tape::release_checkpoints_ending_at`）が
/// 正しく発火することを実測する。
///
/// `region_compute_with_dead_end` は各区間に `h.matmul(&w_dead)`
/// （出力を使わない使い捨て演算）を追加する。`dead_lead=true` では
/// これが区間の `lo` に対応するノードとなり、`backward_impl` の逆走査
/// でこのノードの `grads[id]` は決して `Some` にならない（どこからも
/// upstream 勾配を受け取らない）。修正前は
/// `let Some(upstream) = ... else { continue };` が
/// `release_checkpoints_ending_at(id)` の呼び出し（ループ末尾）を
/// 素通りするため、この区間は backward 側の再解放が一度も走らず、
/// 再計算でキャッシュされた中間値が `Tape::reset()` まで生き残る
/// （ピークメモリ削減がこの区間分だけ機能しない）。
///
/// 判定方法: 「使い捨て演算が区間の先頭（`lo`）に来る」構成と
/// 「使い捨て演算を区間の非 `lo` 位置に置く」構成で同じ区間数・同じ
/// 演算数の chain を組み、両者のピークメモリを比較する。`lo` が
/// 勾配経路上にあるかどうかで解放の可否が変わらない（修正後は常に
/// 解放される）ことを確認する——両者の差が小さいことをもって、`lo`
/// が非到達ノードであっても再解放が機能していると判定する。
fn dead_end_matmul<'t>(h: &Var<'t>, w_dead: &Var<'t>) -> Var<'t> {
    // 出力を一切使わない使い捨て演算。テープへ実際にノードを push する
    // ため（`h.matmul` は `Op::MatMul` を記録する）コンパイラ最適化で
    // 消えることはないが、`black_box` で意図を明示する。
    std::hint::black_box(h.matmul(w_dead).expect("forward は常に成功する構成"))
}

/// `dead_lead` が `true` の場合、区間内で最初に push されるノードを
/// 使い捨て演算（loss への勾配経路上にない）にする。`false` の場合は
/// 使い捨て演算を区間の非 `lo` 位置に置き、`lo` は通常どおり勾配経路上
/// のノードになる。
fn region_compute_with_dead_end<'t>(
    h: &Var<'t>,
    w_a: &Var<'t>,
    w_b: &Var<'t>,
    w_dead: &Var<'t>,
    dead_lead: bool,
) -> Result<Var<'t>, AutodiffError> {
    if dead_lead {
        let _dead = dead_end_matmul(h, w_dead);
        let a = h.matmul(w_a)?;
        let b = a.sigmoid();
        let c = b.matmul(w_b)?;
        Ok(c.sigmoid())
    } else {
        let a = h.matmul(w_a)?;
        let b = a.sigmoid();
        let _dead = dead_end_matmul(&b, w_dead);
        let c = b.matmul(w_b)?;
        Ok(c.sigmoid())
    }
}

fn run_chain_with_dead_end(tape: &Tape, dead_lead: bool) {
    let leaf = tape.var(&make_matrix(0));
    let mut h = leaf;
    let weights_a: Vec<_> = (0..K)
        .map(|i| tape.var(&make_matrix(3 * i as u64 + 1)))
        .collect();
    let weights_b: Vec<_> = (0..K)
        .map(|i| tape.var(&make_matrix(3 * i as u64 + 2)))
        .collect();
    let weights_dead: Vec<_> = (0..K)
        .map(|i| tape.var(&make_matrix(3 * i as u64 + 3)))
        .collect();
    for i in 0..K {
        let w_a = &weights_a[i];
        let w_b = &weights_b[i];
        let w_dead = &weights_dead[i];
        h = tape
            .checkpoint(|| region_compute_with_dead_end(&h, w_a, w_b, w_dead, dead_lead))
            .expect("checkpoint 区間の forward は常に成功する構成");
    }
    let loss = h.sum(None).expect("sum(None) は常に成功する（全軸縮約）");
    let grads = tape.backward(&loss).expect("backward は常に成功する構成");
    std::hint::black_box(&grads);
}

fn check_checkpoint_releases_when_lo_is_dead_end() {
    let (_lead, peak_lead) = measure(|| {
        let tape = Tape::new_with_ops(common::naive_ops());
        run_chain_with_dead_end(&tape, true);
        tape
    });
    let (_tail, peak_tail) = measure(|| {
        let tape = Tape::new_with_ops(common::naive_ops());
        run_chain_with_dead_end(&tape, false);
        tape
    });

    let peak_lead = peak_lead
        .expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");
    let peak_tail = peak_tail
        .expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");

    println!(
        "checkpoint_peak_memory(dead-end lo): peak_lead={peak_lead} bytes, \
         peak_tail={peak_tail} bytes, K={K}"
    );

    // 使い捨て演算が区間の `lo`（先頭）に来る場合と非 `lo` 位置に来る
    // 場合とで、ピークメモリが大きく変わらないことを検証する。修正前
    // （review 指摘の bug）では `dead_lead=true` 構成の全区間で
    // backward 側の再解放が発火せず、中間ノードの再計算値が
    // `Tape::reset()` まで残ってピークが大きく増える。許容差は活性化
    // 1 個分（`activation_bytes()`）の半分未満とし、ノード管理
    // オーバーヘッド程度の揺らぎは許容しつつ「区間丸ごと未解放」の
    // ような大きな退行は検出する。
    let diff = peak_lead.abs_diff(peak_tail);
    let tolerance = activation_bytes() / 2;
    assert!(
        diff < tolerance,
        "checkpoint 区間の `lo` が勾配経路上にない場合とある場合とでピークメモリの差が \
         大きすぎる（diff={diff} バイト、許容={tolerance} バイト、peak_lead={peak_lead}、\
         peak_tail={peak_tail}）——`lo` が非到達ノードのとき backward 側の再解放が \
         機能していない疑い（イシュー #1624 review 指摘の再発）"
    );
}

fn main() {
    check_checkpoint_reduces_peak_memory();
    check_checkpoint_releases_when_lo_is_dead_end();
    println!("checkpoint_peak_memory: all checks passed");
}
