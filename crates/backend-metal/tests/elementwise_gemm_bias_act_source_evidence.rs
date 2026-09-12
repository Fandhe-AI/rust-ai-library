//! イシュー #605: elementwise 5 カーネル・GEMM epilogue 融合カーネル
//! （`gemm_tiled_bias_act`）の文字列証跡テスト。
//! `tests/rmsnorm_softmax_source_evidence.rs`（#604）と同方針:
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI（GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証を兼ねる。`contains` は文字列の
//! 存在のみを証明し、epilogue が境界ガードの内側にネストしていることまでは
//! 検証できない（構造的なネストの正しさはコードレビュー・
//! `tests/gemm_bias_act_parity.rs`〈Metal 実機・`#[ignore]`〉の数値一致
//! テストで別途担保する）。

/// `crates/backend-metal/src/shaders/elementwise.metal` のソース全文。
const ELEMENTWISE_METAL_SOURCE: &str = include_str!("../src/shaders/elementwise.metal");

/// `crates/backend-metal/src/shaders/gemm.metal` のソース全文。
const GEMM_METAL_SOURCE: &str = include_str!("../src/shaders/gemm.metal");

/// elementwise 7 カーネル（`ew_add_f32`／`ew_mul_f32`／`ew_relu_f32`／
/// `ew_exp_f32`／`ew_tanh_f32`／`ew_where_f32`／`ew_masked_fill_f32`。
/// 後 2 者はイシュー #1637 で追加）がいずれも実在することをロックする。
#[test]
fn elementwise_metal_defines_all_seven_kernels() {
    for name in [
        "kernel void ew_add_f32(",
        "kernel void ew_mul_f32(",
        "kernel void ew_relu_f32(",
        "kernel void ew_exp_f32(",
        "kernel void ew_tanh_f32(",
        "kernel void ew_where_f32(",
        "kernel void ew_masked_fill_f32(",
    ] {
        assert!(
            ELEMENTWISE_METAL_SOURCE.contains(name),
            "elementwise.metal に `{name}` が見つかりません"
        );
    }
}

/// REQ-8: elementwise 7 カーネルすべてが `if (idx < numel)` の手動境界
/// チェックを維持していることをロックする（1 スレッド = 1 要素の 1 次元
/// グリッドで `div_ceil` により切り上げ生成するため、末尾ブロックで
/// `idx` が `numel` を超えるスレッドが必ず発生する。
/// `crate::elementwise::ew_dispatch_sizes` 参照）。
#[test]
fn elementwise_metal_kernels_keep_manual_bounds_check() {
    // ファイル冒頭コメント中の言及分（1 件）を含みうるため `>= 7`（7 カーネル
    // 分。イシュー #1637 で 5→7 へ更新）で検証する
    // （`elementwise_metal_defines_all_seven_kernels` が 7 カーネルの
    // 実在を別途ロックしているため、コメントとの取り違えでの見逃しは
    // 起きない）。
    let occurrences = ELEMENTWISE_METAL_SOURCE.matches("if (idx < numel)").count();
    assert!(
        occurrences >= 7,
        "elementwise.metal の 7 カーネルすべてに `if (idx < numel)` の手動境界チェックが\
         必要です（REQ-8）。実際の出現数: {occurrences}"
    );
}

/// `exp`／`tanh` がリポジトリの precise math 方針（`metal::precise::*`
/// 明示使用）に従っていることをロックする（`elementwise.metal` 冒頭
/// コメント「意味論の正」参照）。
#[test]
fn elementwise_metal_uses_precise_math_for_exp_and_tanh() {
    assert!(ELEMENTWISE_METAL_SOURCE.contains("metal::precise::exp("));
    assert!(ELEMENTWISE_METAL_SOURCE.contains("metal::precise::tanh("));
}

/// GEMM epilogue 融合カーネル（`gemm_tiled_bias_act`）が実在することを
/// ロックする（イシュー #605。CUDA 側 `TILED_BIAS_ACT_F32` の対応版）。
#[test]
fn gemm_metal_defines_tiled_bias_act_kernel() {
    assert!(
        GEMM_METAL_SOURCE.contains("kernel void gemm_tiled_bias_act("),
        "gemm.metal に `gemm_tiled_bias_act` が見つかりません"
    );
}

/// REQ-8: `gemm_tiled_bias_act` の C 書き込みガード（`row < m && col < n`）
/// が実在し、epilogue（`has_bias`／`act` 分岐）がそのガードの内側の
/// ブロック内で `bias[col]` を参照していることをロックする（`contains` に
/// よる存在検査。ネストの正しさの担保範囲は本ファイル冒頭コメント参照）。
#[test]
fn gemm_tiled_bias_act_keeps_manual_bounds_check_and_bias_inside_guard() {
    assert!(
        GEMM_METAL_SOURCE.contains("if (row < m && col < n) {"),
        "gemm_tiled_bias_act の C 書き込みガード（REQ-8）が見つかりません"
    );
    assert!(
        GEMM_METAL_SOURCE.contains("if (has_bias != 0) {\n            v += bias[col];"),
        "gemm_tiled_bias_act の bias epilogue（`bias[col]` 参照）が見つかりません"
    );
}

/// `gemm_tiled_bias_act` の activation epilogue（ReLU: `act == 1`）が
/// 実在することをロックする（CPU 参照実装・CUDA 側と同じ比較演算子の向き
/// `v > 0.0f ? v : 0.0f`）。
#[test]
fn gemm_tiled_bias_act_applies_relu_with_cpu_matching_comparison_direction() {
    assert!(
        GEMM_METAL_SOURCE.contains("v = v > 0.0f ? v : 0.0f;"),
        "gemm_tiled_bias_act の ReLU epilogue が見つからないか比較演算子の向きが\
         CPU 参照実装と異なります"
    );
}
