// elementwise（二項 `add`／`mul`、単項 `relu`／`exp`／`tanh`）の MSL カーネル
// ソース（イシュー #605。CUDA 側 `backend-cuda::kernels_elementwise`〈#599〉
// の Metal 対応版）。
//
// `crate::elementwise` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/gemm.metal`・`shaders/rmsnorm.metal` と同一設定）で実行時
// コンパイルする。1 スレッド = 1 要素の 1 次元グリッドで、5 カーネルとも
// `if (idx < numel)` の手動境界チェックを維持する（REQ-8。
// `.claude/rules/coding-rust.md`「カーネル実装の境界検査」）。
//
// # 意味論の正
//
// `backend-cpu::elementwise`（`crates/backend-cpu/src/elementwise.rs`）が
// 意味論の正である。`add`／`mul`／`relu` は加減乗算・比較のみで
// libm を経由せず、`relu` は CPU 参照実装と同じ比較演算子の向き
// （`x > 0.0f ? x : 0.0f`）で NaN 入力時に NaN を伝播せず 0.0 を返す
// （CUDA 側 `kernels_elementwise.rs` と同じ論拠）。`exp`／`tanh` は
// リポジトリの precise math 方針（`docs/backend-matrix.md` §数値一致
// (b)）に従い `metal::precise::exp`／`metal::precise::tanh` を明示使用する
// （`MTLMathFloatingPointFunctions::Precise` を指定しても標準ライブラリ
// 関数がすべて precise 経路を通るとは限らないため、GPU 精度契約の遵守を
// コンパイルオプションだけに委ねず呼び出し側でも明示する）。GPU 側の
// 丸めが CPU の `f32::exp`／`f32::tanh`（libm）と厳密一致する保証はない
// ため、backend 間の数値突合は統一複合判定「相対誤差 1e-3 未満 または
// 絶対誤差 1e-5 未満」（`.claude/rules/coding-rust.md`）で検証する。
//
// # ブロードキャスト
//
// 本カーネルは同一長の 1 次元化済みバッファのみを扱う（ブロードキャスト
// 非対応）。ブロードキャスト対応は呼び出し元（`ops.rs`）が
// `Tensor::broadcast_with` → `contiguous()` で同一 shape へ実体化してから
// 本カーネルへ渡す契約とする（CUDA 側 `kernels_elementwise.rs` と同じ
// 「2 層構成」における役割分担）。

#include <metal_stdlib>
using namespace metal;

kernel void ew_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        out[idx] = a[idx] + b[idx];
    }
}

kernel void ew_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        out[idx] = a[idx] * b[idx];
    }
}

// ReLU（`max(x, 0)`）。CPU 参照実装（`backend-cpu::elementwise::relu`）と
// 同じ比較演算子の向き（`x > 0.0f`）を用いるため、NaN 入力時は Rust の
// `f32::max` と同様 NaN を無視し `relu(NaN) == 0.0` を返す（本ファイル
// 冒頭コメント「意味論の正」参照）。
kernel void ew_relu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        float x = a[idx];
        out[idx] = x > 0.0f ? x : 0.0f;
    }
}

// `exp(x)`（`metal::precise::exp`。上記コメント「意味論の正」参照）。
kernel void ew_exp_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        out[idx] = metal::precise::exp(a[idx]);
    }
}

// `tanh(x)`（`metal::precise::tanh`。上記コメント「意味論の正」参照）。
kernel void ew_tanh_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        out[idx] = metal::precise::tanh(a[idx]);
    }
}

// 条件テンソルによる要素選択（`torch.where` 相当。イシュー #1637）。
// 真偽判定は `c != 0.0f`（`backend-cpu::elementwise::where_slice`・CUDA
// 側 `ew_where_f32` と同一契約。比較のみで libm 非経由）。3 入力とも
// 同長（ブロードキャスト非対応。本ファイル冒頭「ブロードキャスト」
// 節と同じ役割分担）。
kernel void ew_where_f32(
    device const float* cond [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        out[idx] = (cond[idx] != 0.0f) ? a[idx] : b[idx];
    }
}

// マスク位置を定数で置換する（`torch.masked_fill` 相当。イシュー
// #1637）。真偽判定は `m != 0.0f`（`ew_where_f32` と同一契約）。
kernel void ew_masked_fill_f32(
    device const float* x [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& value [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        out[idx] = (mask[idx] != 0.0f) ? value : x[idx];
    }
}
