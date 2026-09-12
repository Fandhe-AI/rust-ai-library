//! elementwise（二項 `add`／`mul`、単項 `relu`／`exp`／`tanh`）の CUDA C
//! カーネルソース（NVRTC 実行時コンパイル用の静的文字列。イシュー #599）。
//!
//! `elementwise.rs`（呼び出し元）は本モジュールの定数を
//! `nvrtc::compile_ptx` に渡し `CudaFunction` を得る。`kernels.rs`
//! （GEMM カーネル）と同じ理由でソースを `nvcc` 事前コンパイルせず文字列の
//! まま埋め込む（ビルド時に nvcc/CUDA ヘッダを一切要求しない。「CUDA
//! toolkit 非搭載環境でも `cargo build --workspace` が成立する」契約を
//! 維持する。`.claude/rules/deps-policy.md`）。
//!
//! # 意味論の正
//!
//! `backend-cpu::elementwise`（`crates/backend-cpu/src/elementwise.rs`）が
//! 意味論の正である。全カーネルとも加減乗算・比較のみで libm を経由せず
//! （`add`／`mul`／`relu`）、`relu` は `f32::max` 相当（`x > 0.0f ? x :
//! 0.0f`）で NaN 入力時に CPU 版と同じ扱い（NaN を伝播せず 0.0 を返す）に
//! なるよう比較演算子の向きを揃えている。`exp`／`tanh` は CUDA 組み込みの
//! `expf`／`tanhf`（単精度版。`double` 版 `exp`／`tanh` への暗黙昇格を避ける）
//! を用いる。GPU 側の丸めが CPU の `f32::exp`／`f32::tanh`（libm）と厳密
//! 一致する保証はないため、backend 間の数値突合は統一複合判定「相対誤差
//! 1e-3 未満 または 絶対誤差 1e-5 未満」（`.claude/rules/coding-rust.md`）で
//! 検証する（`tests/gemm_bias_act_parity.rs` の
//! `elementwise_matches_cpu_across_ops`（`#[ignore]`）で add／mul／relu／
//! exp／tanh の 5 演算を検証する）。
//!
//! # ブロードキャスト
//!
//! 本カーネルは同一長の 1 次元化済みバッファのみを扱う（ブロードキャスト
//! 非対応）。ブロードキャスト対応は呼び出し元（`ops.rs`）が
//! `Tensor::broadcast_with` → `contiguous()` で同一 shape へ実体化して
//! から本カーネルへ渡す契約とする（`backend-cpu::elementwise` の「2 層
//! 構成」における「スライスカーネル層」と同じ役割分担）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 全カーネルは `if (idx < numel)` の手動境界チェックを維持する。1
//! スレッド = 1 要素でグリッドを `div_ceil` により切り上げ生成するため、
//! 末尾ブロックでは `idx` が `numel` を超えるスレッドが必ず発生する
//! （`kernels.rs` の naive/tiled カーネルと同じ理由。
//! `.claude/rules/coding-rust.md` の REQ-8 規約）。`numel` はホスト側
//! （`elementwise.rs::validate_elementwise_dims`）で `i32::MAX` に収まる
//! ことを起動前に検証しており（`gemm.rs::validate_gemm_dims` の i32 積
//! ガードと同じ考え方）、カーネル内の `int` 算術（`idx`／`numel`）が
//! オーバーフローしないことをホスト側検証と合わせて保証する。

/// 1 スレッドブロックあたりのスレッド数（1 次元）。
///
/// GEMM の `TILE`（2 次元・共有メモリタイル境界）とは無関係の独立した
/// パラメータ（elementwise カーネルは共有メモリを使わないため、ブロック
/// サイズはオキュパンシ最適化のみが関心事）。PoC 実測なしの保守的な固定値
/// （256 = warp サイズ 32 の倍数でよく使われる値）とし、チューニングは
/// 別イシューのスコープとする（out-of-scope-tracking.md 対象）。
pub const EW_BLOCK_DIM: u32 = 256;

/// 二項加算 `out[i] = a[i] + b[i]`（f32）。
pub const EW_ADD_F32: &str = r#"
extern "C" __global__ void ew_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] + b[idx];
    }
}
"#;

/// 二項乗算 `out[i] = a[i] * b[i]`（f32）。
pub const EW_MUL_F32: &str = r#"
extern "C" __global__ void ew_mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] * b[idx];
    }
}
"#;

/// ReLU（`max(x, 0)`）。CPU 参照実装（`backend-cpu::elementwise::relu`）と
/// 同じ比較演算子の向き（`x > 0.0f`）を用いるため、NaN 入力時は Rust の
/// `f32::max` と同様 NaN を無視し `relu(NaN) == 0.0` を返す（本ファイル
/// 冒頭ドキュメントコメント「意味論の正」参照）。
pub const EW_RELU_F32: &str = r#"
extern "C" __global__ void ew_relu_f32(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = a[idx];
        out[idx] = x > 0.0f ? x : 0.0f;
    }
}
"#;

/// `exp(x)`（単精度 `expf`）。
pub const EW_EXP_F32: &str = r#"
extern "C" __global__ void ew_exp_f32(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = expf(a[idx]);
    }
}
"#;

/// `tanh(x)`（単精度 `tanhf`）。
pub const EW_TANH_F32: &str = r#"
extern "C" __global__ void ew_tanh_f32(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = tanhf(a[idx]);
    }
}
"#;

/// 条件テンソルによる要素選択（`torch.where` 相当。イシュー #1637）。
/// 真偽判定は `c != 0.0f`（`backend-cpu::elementwise::where_slice` と
/// 同一契約。比較のみで libm 非経由）。3 入力とも同長（ブロードキャスト
/// 非対応。呼び出し元が事前に同一 shape へ実体化する契約は本モジュール
/// 冒頭の「ブロードキャスト」節と同じ）。
pub const EW_WHERE_F32: &str = r#"
extern "C" __global__ void ew_where_f32(
    const float* __restrict__ cond,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = (cond[idx] != 0.0f) ? a[idx] : b[idx];
    }
}
"#;

/// マスク位置を定数で置換する（`torch.masked_fill` 相当。イシュー
/// #1637）。真偽判定は `m != 0.0f`（`where_slice` と同一契約）。
pub const EW_MASKED_FILL_F32: &str = r#"
extern "C" __global__ void ew_masked_fill_f32(
    const float* __restrict__ x,
    const float* __restrict__ mask,
    float value,
    float* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = (mask[idx] != 0.0f) ? value : x[idx];
    }
}
"#;
