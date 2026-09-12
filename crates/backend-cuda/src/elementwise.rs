//! elementwise（`add`／`mul`／`relu`／`exp`／`tanh`）の起動 API（NVRTC
//! コンパイル・保持・実行。イシュー #599）。
//!
//! `gemm.rs::CudaGemm` と同じ構成方針を踏襲する: `CudaElementwise::new` が
//! `CudaDevice` から 5 カーネルを一括 NVRTC コンパイルして保持し、以降は
//! `run_*_f32` へホスト側スライスを渡すだけで GPU 実行できる（H2D 転送 →
//! 起動 → 同期 → D2H 転送を内部で完結させる）。カーネルソース自体は
//! `kernels_elementwise.rs`（NVRTC 文字列埋め込み）に閉じ込め、本モジュール
//! はコンパイル結果（`CudaFunction`）の保持とメモリ転送・起動手続きのみを
//! 扱う（`gemm.rs` 冒頭コメントと同じ責務分離）。
//!
//! `ops.rs::CudaBackendOps` から `BackendOps::add`／`mul`／`relu`／`exp`／
//! `tanh` の実装として呼ばれる。ブロードキャスト対応（NumPy 互換）は
//! `ops.rs` 側が `Tensor::broadcast_with` → `contiguous()` で同一 shape の
//! 密なバッファへ実体化してから本モジュールへ渡す契約（本モジュール自体は
//! 同一長バッファの 1:1 演算のみを扱う。`kernels_elementwise.rs` 冒頭
//! コメント「ブロードキャスト」参照）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::{BinaryElementwiseOp, UnaryElementwiseOp};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_elementwise::{self, EW_BLOCK_DIM};
use crate::memory::{CudaArg, CudaArgMut};
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// elementwise 演算 1 回あたりのブロック次元（1 次元、`EW_BLOCK_DIM` 幅）。
const EW_BLOCK: (u32, u32, u32) = (EW_BLOCK_DIM, 1, 1);

/// `a_len`／`b_len` が一致し、かつ `i32::MAX` に収まることを検証する
/// （二項演算向け）。
///
/// カーネル引数 `int numel` は C の 32bit 符号付き整数のため、GEMM の
/// `gemm.rs::validate_gemm_dims` と同じ理由で起動前に上限を検査する
/// （OWASP A03。`.claude/rules/security.md`）。`pub(crate)`:
/// 実機非依存の単体テスト（本ファイル末尾 `#[cfg(test)]`）から直接呼べる
/// よう公開範囲をクレート内に限定する。
pub(crate) fn validate_elementwise_binary_dims(
    a_len: usize,
    b_len: usize,
) -> Result<(), CudaError> {
    if a_len != b_len {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("elementwise length mismatch: a_len={a_len}, b_len={b_len}"),
        });
    }
    validate_elementwise_len(a_len)
}

/// `cond_len`／`a_len`／`b_len` が全て一致し、かつ `i32::MAX` に収まる
/// ことを検証する（3 入力演算〈`where_cond`〉向け。イシュー #1637）。
/// `validate_elementwise_binary_dims` の 3 項版。
pub(crate) fn validate_elementwise_ternary_dims(
    cond_len: usize,
    a_len: usize,
    b_len: usize,
) -> Result<(), CudaError> {
    if cond_len != a_len || cond_len != b_len {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "elementwise length mismatch: cond_len={cond_len}, a_len={a_len}, b_len={b_len}"
            ),
        });
    }
    validate_elementwise_len(cond_len)
}

/// 単項演算向け: 長さが `i32::MAX` に収まることのみを検証する。
pub(crate) fn validate_elementwise_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "elementwise numel must fit in i32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

/// `numel` に対し `EW_BLOCK` を `div_ceil` で包含するグリッド次元を構築する
/// （`gemm.rs::launch_config` と同じ「末尾ブロックの余剰スレッドはカーネル
/// 内境界チェックに委ねる」契約。REQ-8）。
fn elementwise_launch_config(numel: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (numel.div_ceil(EW_BLOCK.0), 1, 1),
        block_dim: EW_BLOCK,
        shared_mem_bytes: 0,
    }
}

/// elementwise 5 カーネル（`add`／`mul`／`relu`／`exp`／`tanh`。いずれも
/// f32）のコンパイル済みハンドルを保持する。
///
/// `stream` は [`CudaDevice`] から `Arc` クローンで受け取る（`gemm.rs` の
/// 共有契約どおり）。`new` 時に 5 カーネルを一括コンパイルするのは
/// `nvrtc::compile_ptx` の呼び出し契約（`gemm.rs::CudaGemm` のドキュメント
/// コメント参照）を守るためであり、`run_*` 呼び出しのたびに再コンパイル
/// しない。
pub struct CudaElementwise {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（イシュー #1349・codex-review P0
    /// 指摘・PR #1390 是正）。`Self::with_driver_call` が
    /// `context_cache::with_driver_call` を呼ぶ際のキーとして使う
    /// （`gemm.rs::CudaGemm::ordinal` と同じ役割）。
    ordinal: usize,
    /// 出力バッファのサイズクラス別プール（イシュー #1020・REQ-14）。
    /// `gemm.rs::CudaGemm::allocator` と同一の設計（`crate::pool` 冒頭
    /// コメント参照）。`context_cache::cached_allocator` 経由で
    /// `CudaGemm` と同じ `(ordinal, 既定 stream)` 単位プールを共有する。
    allocator: Arc<CudaAllocator>,
    add_f32: CudaFunction,
    mul_f32: CudaFunction,
    relu_f32: CudaFunction,
    exp_f32: CudaFunction,
    tanh_f32: CudaFunction,
    /// `torch.where` 相当（イシュー #1637）。`add_f32` 等と同じ NVRTC
    /// コンパイル済みハンドルを `new` 時に一括保持する。
    where_f32: CudaFunction,
    /// `torch.masked_fill` 相当（イシュー #1637）。
    masked_fill_f32: CudaFunction,
}

impl CudaElementwise {
    /// `device` 上で elementwise 5 カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する。
    ///
    /// 手順: `kernels_elementwise::{EW_ADD_F32,EW_MUL_F32,EW_RELU_F32,
    /// EW_EXP_F32,EW_TANH_F32}` を `device.arch()` 向けに
    /// `nvrtc::compile_ptx` でコンパイル → `device.context().load_module()`
    /// → `load_function(...)`（`gemm.rs::CudaGemm::new` の naive/tiled
    /// 4 カーネルと同一手順）。コンパイル失敗（NVRTC 不在・構文エラー等）は
    /// `CudaError` として早期 return する（naive/tiled の 4 カーネルと同じ
    /// く `#include` を使わず全 compute capability で成立するため、
    /// WMMA(TF32) 系カーネルのような `Option` フィールド化・失敗の退避は
    /// 不要と判断した）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let add_ptx = compile_ptx(kernels_elementwise::EW_ADD_F32, arch)?;
        let mul_ptx = compile_ptx(kernels_elementwise::EW_MUL_F32, arch)?;
        let relu_ptx = compile_ptx(kernels_elementwise::EW_RELU_F32, arch)?;
        let exp_ptx = compile_ptx(kernels_elementwise::EW_EXP_F32, arch)?;
        let tanh_ptx = compile_ptx(kernels_elementwise::EW_TANH_F32, arch)?;
        let where_ptx = compile_ptx(kernels_elementwise::EW_WHERE_F32, arch)?;
        let masked_fill_ptx = compile_ptx(kernels_elementwise::EW_MASKED_FILL_F32, arch)?;

        let add_f32 = device
            .context()
            .load_module(add_ptx)?
            .load_function("ew_add_f32")?;
        let mul_f32 = device
            .context()
            .load_module(mul_ptx)?
            .load_function("ew_mul_f32")?;
        let relu_f32 = device
            .context()
            .load_module(relu_ptx)?
            .load_function("ew_relu_f32")?;
        let exp_f32 = device
            .context()
            .load_module(exp_ptx)?
            .load_function("ew_exp_f32")?;
        let tanh_f32 = device
            .context()
            .load_module(tanh_ptx)?
            .load_function("ew_tanh_f32")?;
        let where_f32 = device
            .context()
            .load_module(where_ptx)?
            .load_function("ew_where_f32")?;
        let masked_fill_f32 = device
            .context()
            .load_module(masked_fill_ptx)?
            .load_function("ew_masked_fill_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            add_f32,
            mul_f32,
            relu_f32,
            exp_f32,
            tanh_f32,
            where_f32,
            masked_fill_f32,
        })
    }

    /// `CudaElementwise` の driver 呼び出し（H2D 転送・カーネル起動・
    /// D2H readback）を CUDA Graph capture 排他へ参加させる共通ヘルパー
    /// （`gemm.rs::CudaGemm::with_driver_call` と同じ設計。codex-review
    /// P0 指摘対応・PR #1390 是正）。
    ///
    /// `run_add_f32`／`run_relu_f32` 等の公開低レベル API は
    /// `CudaBackendOps`（`ops.rs`）を経由せず crate 外から直接呼び出せる
    /// ため、従来はこれらの呼び出しが `context_cache::begin_driver_call`
    /// の capture 排他検査を一切通らず、共有ストリームへ直接転送・確保・
    /// カーネル起動を発行していた（別スレッドが SGD capture 中でも
    /// 拒否されず、drain・拒否を迂回して一時バッファ等の無関係な操作が
    /// graph に混入しうる欠陥。codex-review 指摘）。本ヘルパーを
    /// `run_binary`／`run_unary`（全公開エントリの共通実装）の本体先頭で
    /// 呼ぶことで、直接構築・`ops.rs` 経由いずれの呼び出し経路でも同じ
    /// 排他区間を通るようにする。
    ///
    /// `ops.rs` 経由の呼び出しは外側の `with_driver_call` と二重に排他
    /// 区間へ入るが、`begin_driver_call` は同一スレッドからの再入を
    /// 拒否しない設計（`context_cache::begin_driver_call` doc コメント
    /// 参照）のため、二重の排他は安全側の重複であり挙動を変えない。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// 二項演算共通の起動手続き（H2D → 起動 → 同期 → D2H）。
    ///
    /// `a.len() == 0`（呼び出し元の shape が空要素）の場合は `gemm.rs`
    /// の `m == 0 || n == 0` 早期 return と同じ理由（0 バイトデバイス確保を
    /// 一部 CUDA driver が拒否しうる）でカーネル起動自体を回避し、空の
    /// 結果を返す。
    fn run_binary(&self, func: &CudaFunction, a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        validate_elementwise_binary_dims(a.len(), b.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        // codex-review P0 指摘対応（PR #1390 是正）: `Self::with_driver_call`
        // で本体（H2D・確保・起動・readback）全体を capture 排他へ参加
        // させる（`gemm_wmma.rs::CudaWmmaGemm::launch_f16_kernel` と同じ
        // 「1 回の呼び出しにまとめて包む」方式）。
        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let b_dev = self.stream.clone_htod(b)?;
            // イシュー #1020: 全カーネル（`ew_add_f32` 等）が
            // `if (idx < numel)` ガード内で `out[idx]` を必ず埋める
            // （`kernels_elementwise.rs` 参照）ため `alloc_uninit_f32` を使う。
            let mut out_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = elementwise_launch_config(numel as u32);
            let numel_i = numel as i32;

            // SAFETY: カーネル引数（a_dev/b_dev/out_dev・numel_i）は上記で
            // 検証済みの numel と 1:1 対応するデバイスバッファ長・値であり、
            // カーネル内の手動境界チェック（`if (idx < numel)`。
            // `kernels_elementwise.rs` 参照、REQ-8）と合わせて OOB 読み書きが
            // 起きない根拠とする。グリッド次元は `div_ceil` で numel を包含
            // するよう構築しており（`elementwise_launch_config`）、末尾ブロック
            // の余剰スレッドはカーネル内境界チェックで弾かれる。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&a_dev)
                    .arg(&b_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            // 同期点は readback ヘルパーへ集約（#1013）。プール割当ハンドル
            // （`PooledCudaHandle`。イシュー #1020）は `DevicePtr` を直接実装しない
            // ため、論理長ビュー（`as_view()`）を渡す。
            crate::memory::readback(&self.stream, &out_dev.as_view())
        })
    }

    /// 3 項演算共通の起動手続き（`where_cond` 専用。イシュー #1637）。
    /// [`Self::run_binary`] と同一構造で入力が 1 本増えただけ。
    fn run_ternary(
        &self,
        func: &CudaFunction,
        cond: &[f32],
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, CudaError> {
        validate_elementwise_ternary_dims(cond.len(), a.len(), b.len())?;
        let numel = cond.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let cond_dev = self.stream.clone_htod(cond)?;
            let a_dev = self.stream.clone_htod(a)?;
            let b_dev = self.stream.clone_htod(b)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = elementwise_launch_config(numel as u32);
            let numel_i = numel as i32;

            // SAFETY: run_binary と同一の根拠（デバイスバッファ長は
            // numel と 1:1 対応・カーネル内 `if (idx < numel)` 境界検査。
            // REQ-8）。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&cond_dev)
                    .arg(&a_dev)
                    .arg(&b_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            crate::memory::readback(&self.stream, &out_dev.as_view())
        })
    }

    /// 二項＋スカラー演算共通の起動手続き（`masked_fill` 専用。イシュー
    /// #1637）。`value`（`f32` スカラー）はカーネル引数として `x`／
    /// `mask` の後・`out` の前に渡す（`kernels_elementwise::
    /// EW_MASKED_FILL_F32` の引数順と一致させる）。
    fn run_binary_scalar(
        &self,
        func: &CudaFunction,
        x: &[f32],
        mask: &[f32],
        value: f32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_elementwise_binary_dims(x.len(), mask.len())?;
        let numel = x.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let x_dev = self.stream.clone_htod(x)?;
            let mask_dev = self.stream.clone_htod(mask)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = elementwise_launch_config(numel as u32);
            let numel_i = numel as i32;

            // SAFETY: run_binary と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&x_dev)
                    .arg(&mask_dev)
                    .arg(&value)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            crate::memory::readback(&self.stream, &out_dev.as_view())
        })
    }

    /// 条件テンソルによる要素選択（`torch.where` 相当。イシュー #1637）。
    /// `cond`／`a`／`b` は同一長であること。
    pub fn run_where_f32(&self, cond: &[f32], a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_ternary(&self.where_f32, cond, a, b)
    }

    /// マスク位置を定数で置換する（`torch.masked_fill` 相当。イシュー
    /// #1637）。`x`／`mask` は同一長であること。
    pub fn run_masked_fill_f32(
        &self,
        x: &[f32],
        mask: &[f32],
        value: f32,
    ) -> Result<Vec<f32>, CudaError> {
        self.run_binary_scalar(&self.masked_fill_f32, x, mask, value)
    }

    /// 単項演算共通の起動手続き。[`Self::run_binary`] と同一構造。
    fn run_unary(&self, func: &CudaFunction, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        validate_elementwise_len(a.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        // codex-review P0 指摘対応（PR #1390 是正）: `run_binary` と同じ
        // 理由で `Self::with_driver_call` へ参加させる。
        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = elementwise_launch_config(numel as u32);
            let numel_i = numel as i32;

            // SAFETY: run_binary と同一の根拠（上記コメント参照）。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&a_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            // 同期点は readback ヘルパーへ集約（#1013）。プール割当ハンドル
            // （`PooledCudaHandle`。イシュー #1020）は `DevicePtr` を直接実装しない
            // ため、論理長ビュー（`as_view()`）を渡す。
            crate::memory::readback(&self.stream, &out_dev.as_view())
        })
    }

    /// `out[i] = a[i] + b[i]`（f32・同一長）。
    pub fn run_add_f32(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_binary(&self.add_f32, a, b)
    }

    /// `out[i] = a[i] * b[i]`（f32・同一長）。
    pub fn run_mul_f32(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_binary(&self.mul_f32, a, b)
    }

    /// `out[i] = max(a[i], 0)`（f32）。
    pub fn run_relu_f32(&self, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_unary(&self.relu_f32, a)
    }

    /// `out[i] = exp(a[i])`（f32、単精度 `expf`）。
    pub fn run_exp_f32(&self, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_unary(&self.exp_f32, a)
    }

    /// `out[i] = tanh(a[i])`（f32、単精度 `tanhf`）。
    pub fn run_tanh_f32(&self, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_unary(&self.tanh_f32, a)
    }

    /// `op` に対応するコンパイル済みカーネルを返す（`launch_binary_resident`
    /// の内部選択専用。ホスト版 `run_add_f32`／`run_mul_f32` と同一
    /// カーネルを再利用するため bit 同一契約が成立する）。
    ///
    /// `BinaryElementwiseOp` は `#[non_exhaustive]`。未知 variant を
    /// `_ =>` で `add_f32` 等へフォールバックすると、将来 variant が
    /// 追加された際に「別の演算を代わりに計算して黙って成功する」
    /// fail-open になる（advisor 指摘。イシュー #1584）。`gemm_bias_act`
    /// の `Activation` 未知 variant 拒否と同方針で `Err` を返す。
    fn function_for_binary(&self, op: BinaryElementwiseOp) -> Result<&CudaFunction, CudaError> {
        match op {
            BinaryElementwiseOp::Add => Ok(&self.add_f32),
            BinaryElementwiseOp::Mul => Ok(&self.mul_f32),
            _ => Err(CudaError::UnsupportedElementwiseOp {
                detail: format!("unknown BinaryElementwiseOp variant: {op:?}"),
            }),
        }
    }

    /// [`Self::function_for_binary`] の単項版。
    fn function_for_unary(&self, op: UnaryElementwiseOp) -> Result<&CudaFunction, CudaError> {
        match op {
            UnaryElementwiseOp::Relu => Ok(&self.relu_f32),
            UnaryElementwiseOp::Exp => Ok(&self.exp_f32),
            UnaryElementwiseOp::Tanh => Ok(&self.tanh_f32),
            _ => Err(CudaError::UnsupportedElementwiseOp {
                detail: format!("unknown UnaryElementwiseOp variant: {op:?}"),
            }),
        }
    }

    /// `a op b` を [`crate::memory::DeviceBuffer`] 常駐のまま計算する
    /// （イシュー #1584。`tensor-core::BackendOps::
    /// binary_elementwise_device` の CUDA 実装が呼ぶ）。[`Self::
    /// run_binary`] と異なり H2D／D2H・`readback`（同期）を一切行わない:
    /// `a`／`b`／`out` はいずれも呼び出し元（`ops.rs`）がデバイス常駐
    /// バッファから取り出した [`CudaArg`]／[`CudaArgMut`] で、カーネル
    /// 起動をストリームへ積むだけに留める（同期点は呼び出し元の
    /// `download` へ集約する契約。`docs/backend-cuda-async-execution-
    /// design.md`）。`a.len() == b.len() == out.len() == numel` は
    /// 呼び出し元が検証済みの前提とする（本関数はカーネル引数の `int`
    /// 上限のみ再検証する）。
    pub(crate) fn launch_binary_resident(
        &self,
        op: BinaryElementwiseOp,
        a: &CudaArg<'_>,
        b: &CudaArg<'_>,
        out: &mut CudaArgMut<'_>,
        numel: usize,
    ) -> Result<(), CudaError> {
        validate_elementwise_binary_dims(a.len(), b.len())?;
        validate_elementwise_len(out.len())?;
        // `a.len()==b.len()==out.len()==numel` を明示検証する（advisor
        // 指摘: 是正前は `a.len()==b.len()` と `out.len()` の i32 上限
        // しか見ておらず、SAFETY コメントが主張する「`numel` 要素と
        // 1:1 対応」を実際には担保していなかった。ここで fail-closed に
        // 拒否することで SAFETY コメントの前提を実装が満たすようにする）。
        if a.len() != numel || out.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "elementwise buffer length must equal numel: a_len={}, b_len={}, \
                     out_len={}, numel={numel}",
                    a.len(),
                    b.len(),
                    out.len()
                ),
            });
        }
        if numel == 0 {
            return Ok(());
        }

        let func = self.function_for_binary(op)?;
        let cfg = elementwise_launch_config(numel as u32);
        let numel_i = numel as i32;

        // SAFETY: `a`／`b`／`out` は呼び出し元（`ops.rs::
        // binary_elementwise_device`）が `numel` 要素と 1:1 対応する
        // ことを検証済みのデバイスバッファ・ビューであり（上記の
        // `a.len()`／`out.len()` 検証で本関数自身も再検証している）、
        // カーネル内の手動境界チェック（`if (idx < numel)`。
        // `kernels_elementwise.rs` 参照、REQ-8）と合わせて OOB 読み書き
        // が起きない根拠とする。
        // `CudaArg`／`CudaArgMut` は配置（`Device`／`Managed`）ごとに
        // 異なる cudarc 型へ委譲するのみで、カーネル本体・起動 config
        // は配置に依らず完全に共有する（`gemm.rs::
        // launch_tiled_bias_act_f32_resident` と同じ設計）。
        unsafe {
            let mut builder = self.stream.launch_builder(func);
            a.push(&mut builder);
            b.push(&mut builder);
            out.push(&mut builder);
            builder.arg(&numel_i).launch(cfg)?;
        }
        Ok(())
    }

    /// [`Self::launch_binary_resident`] の単項版。
    pub(crate) fn launch_unary_resident(
        &self,
        op: UnaryElementwiseOp,
        a: &CudaArg<'_>,
        out: &mut CudaArgMut<'_>,
        numel: usize,
    ) -> Result<(), CudaError> {
        validate_elementwise_len(a.len())?;
        validate_elementwise_len(out.len())?;
        // `a.len()==out.len()==numel` を明示検証する（`launch_binary_
        // resident` と同じ是正。advisor 指摘）。
        if a.len() != numel || out.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "elementwise buffer length must equal numel: a_len={}, out_len={}, numel={numel}",
                    a.len(),
                    out.len()
                ),
            });
        }
        if numel == 0 {
            return Ok(());
        }

        let func = self.function_for_unary(op)?;
        let cfg = elementwise_launch_config(numel as u32);
        let numel_i = numel as i32;

        // SAFETY: `launch_binary_resident` と同一の根拠（`a.len()`／
        // `out.len()` を `numel` と一致検証済み）。
        unsafe {
            let mut builder = self.stream.launch_builder(func);
            a.push(&mut builder);
            out.push(&mut builder);
            builder.arg(&numel_i).launch(cfg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_elementwise_binary_dims_accepts_matching_lengths() {
        assert!(validate_elementwise_binary_dims(4, 4).is_ok());
    }

    #[test]
    fn validate_elementwise_binary_dims_rejects_length_mismatch() {
        let err = validate_elementwise_binary_dims(4, 5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_len_rejects_exceeding_i32_max() {
        let err = validate_elementwise_len(i32::MAX as usize + 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_len_accepts_i32_max() {
        assert!(validate_elementwise_len(i32::MAX as usize).is_ok());
    }

    #[test]
    fn validate_elementwise_ternary_dims_accepts_matching_lengths() {
        assert!(validate_elementwise_ternary_dims(4, 4, 4).is_ok());
    }

    #[test]
    fn validate_elementwise_ternary_dims_rejects_cond_mismatch() {
        let err = validate_elementwise_ternary_dims(4, 4, 5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_ternary_dims_rejects_a_mismatch() {
        let err = validate_elementwise_ternary_dims(4, 5, 4).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }
}
