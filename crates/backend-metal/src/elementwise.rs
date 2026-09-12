//! elementwise（`add`／`mul`／`relu`／`exp`／`tanh`）の起動 API（イシュー
//! #605。CUDA 側 `backend-cuda::elementwise`〈#599〉の Metal 対応版）。
//!
//! [`MetalElementwise::new`] が `shaders/elementwise.metal` を実行時
//! コンパイルして 5 パイプラインを保持し、[`MetalElementwise::run_add_f32`]
//! 等へホスト側スライスを渡すだけでバッファ確保・ディスパッチ・readback を
//! 内部で完結できる（`crate::gemm::MetalGemm`・`crate::rmsnorm::MetalRmsNorm`
//! と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps` から `BackendOps::add`／`mul`／`relu`／`exp`／
//! `tanh` の実装として呼ばれる。ブロードキャスト対応（NumPy 互換）は
//! `ops.rs` 側が `Tensor::broadcast_with` → `contiguous()` で同一 shape の
//! 密なバッファへ実体化してから本モジュールへ渡す契約（本モジュール自体は
//! 同一長バッファの 1:1 演算のみを扱う。`shaders/elementwise.metal` 冒頭
//! コメント「ブロードキャスト」参照）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::{BinaryElementwiseOp, UnaryElementwiseOp};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/elementwise.metal` のソース（5 カーネルを含む）。
const ELEMENTWISE_MSL_SRC: &str = include_str!("shaders/elementwise.metal");

/// 1 スレッドグループあたりのスレッド数（1 次元）。`crate::gemm` の
/// `THREADGROUP_SIDE`（16×16・2 次元）とは無関係の独立したパラメータ
/// （elementwise カーネルは threadgroup 共有メモリを使わないため、幅は
/// オキュパンシ最適化のみが関心事）。PoC 実測なしの保守的な固定値
/// （CUDA 側 `kernels_elementwise::EW_BLOCK_DIM` と同じ 256）とし、
/// チューニングは別イシューのスコープとする（out-of-scope-tracking.md 対象）。
const EW_THREADGROUP_WIDTH: usize = 256;

/// `a_len`／`b_len` が一致することを検証する（二項演算向け）。
///
/// `pub(crate)`: 実機非依存の単体テスト（本ファイル末尾 `#[cfg(test)]`）
/// から直接呼べるよう公開範囲をクレート内に限定する（CUDA 側
/// `elementwise.rs::validate_elementwise_binary_dims` と同じ設計）。
pub(crate) fn validate_elementwise_binary_dims(
    a_len: usize,
    b_len: usize,
) -> Result<(), MetalError> {
    if a_len != b_len {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!("elementwise length mismatch: a_len={a_len}, b_len={b_len}"),
        });
    }
    validate_elementwise_len(a_len)
}

/// `cond_len`／`a_len`／`b_len` が全て一致することを検証する（3 入力
/// 演算〈`where_cond`〉向け。イシュー #1637）。
/// [`validate_elementwise_binary_dims`] の 3 項版。
pub(crate) fn validate_elementwise_ternary_dims(
    cond_len: usize,
    a_len: usize,
    b_len: usize,
) -> Result<(), MetalError> {
    if cond_len != a_len || cond_len != b_len {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "elementwise length mismatch: cond_len={cond_len}, a_len={a_len}, b_len={b_len}"
            ),
        });
    }
    validate_elementwise_len(cond_len)
}

/// 単項演算向け: 長さが `u32::MAX` に収まることのみを検証する。
///
/// カーネル引数 `constant uint& numel`（`shaders/elementwise.metal`）は
/// 32bit のため、`numel as u32` キャストが検証なしだと `numel >
/// u32::MAX` で切り詰まり、出力バッファの一部のみ計算されゼロ埋めの
/// まま正常応答してしまう（数値契約違反。codex-review 指摘・CUDA 側
/// `elementwise.rs::validate_elementwise_len` と同じ理由。OWASP A03。
/// `.claude/rules/security.md`）。
pub(crate) fn validate_elementwise_len(len: usize) -> Result<(), MetalError> {
    if len > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "elementwise numel must fit in u32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

/// elementwise 5 カーネル（`add`／`mul`／`relu`／`exp`／`tanh`。いずれも
/// f32）のコンパイル済みパイプラインを保持するハンドル。
pub struct MetalElementwise {
    add_f32: objc2::rc::Retained<MtlPipeline>,
    mul_f32: objc2::rc::Retained<MtlPipeline>,
    relu_f32: objc2::rc::Retained<MtlPipeline>,
    exp_f32: objc2::rc::Retained<MtlPipeline>,
    tanh_f32: objc2::rc::Retained<MtlPipeline>,
    /// `torch.where` 相当（イシュー #1637）。
    where_f32: objc2::rc::Retained<MtlPipeline>,
    /// `torch.masked_fill` 相当（イシュー #1637）。
    masked_fill_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalElementwise {
    /// `ctx` のデバイス上で elementwise 5 カーネルを実行時コンパイルし
    /// パイプラインを構築する。
    ///
    /// `pipeline::compile_options()`（`MathMode::Safe` +
    /// `MathFloatingPointFunctions::Precise`）を使う点は
    /// `crate::pipeline::compile_gemm_library`・`crate::rmsnorm::MetalRmsNorm::new`
    /// と同一であり、両カーネル間で丸め・関数ディスパッチ先の精度契約が
    /// 揃うことを保証する。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(ELEMENTWISE_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let add_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_add_f32")?;
        let mul_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_mul_f32")?;
        let relu_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_relu_f32")?;
        let exp_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_exp_f32")?;
        let tanh_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_tanh_f32")?;
        let where_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_where_f32")?;
        let masked_fill_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "ew_masked_fill_f32")?;

        Ok(Self {
            add_f32,
            mul_f32,
            relu_f32,
            exp_f32,
            tanh_f32,
            where_f32,
            masked_fill_f32,
        })
    }

    /// 二項演算共通の起動手続き（バッファ確保 → ディスパッチ → readback）。
    ///
    /// `a.len() == 0`（呼び出し元の shape が空要素）の場合はカーネル起動
    /// 自体を回避し空の結果を返す（`crate::gemm::MetalGemm::dispatch_variant`
    /// の `m == 0 || n == 0` 早期 return と同じ理由。0 バイトバッファ確保は
    /// `crate::buffer::MetalBuffer` が `ZeroLengthAllocation` として拒否する
    /// ため、その手前で回避する）。
    fn run_binary(
        &self,
        ctx: &MetalContext,
        pipeline: &MtlPipeline,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        validate_elementwise_binary_dims(a.len(), b.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let b_buf = MetalBuffer::new_with_data(ctx, b)?;
        // イシュー #1021: encode_binary_dispatch／encode_unary_dispatch は
        // numel 全要素を書き切る出力専用バッファのため alloc_uninit_pooled
        // を使う（設計文書 §6「A02」）。
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_binary_dispatch(encoder, pipeline, &a_buf, &b_buf, &out_buf, numel as u32);
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// 3 項演算共通の起動手続き（`where_cond` 専用。イシュー #1637）。
    /// [`Self::run_binary`] と同一構造で入力が 1 本増えただけ。
    fn run_ternary(
        &self,
        ctx: &MetalContext,
        pipeline: &MtlPipeline,
        cond: &[f32],
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        validate_elementwise_ternary_dims(cond.len(), a.len(), b.len())?;
        let numel = cond.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let cond_buf = MetalBuffer::new_with_data(ctx, cond)?;
        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let b_buf = MetalBuffer::new_with_data(ctx, b)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_ternary_dispatch(
                encoder,
                pipeline,
                &cond_buf,
                &a_buf,
                &b_buf,
                &out_buf,
                numel as u32,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// 二項＋スカラー演算共通の起動手続き（`masked_fill` 専用。イシュー
    /// #1637）。[`Self::run_binary`] と同一構造だがスカラー `value` を
    /// 追加で `setBytes_length_atIndex` する。
    fn run_binary_scalar(
        &self,
        ctx: &MetalContext,
        pipeline: &MtlPipeline,
        x: &[f32],
        mask: &[f32],
        value: f32,
    ) -> Result<Vec<f32>, MetalError> {
        validate_elementwise_binary_dims(x.len(), mask.len())?;
        let numel = x.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let mask_buf = MetalBuffer::new_with_data(ctx, mask)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_binary_scalar_dispatch(
                encoder,
                pipeline,
                &x_buf,
                &mask_buf,
                &out_buf,
                value,
                numel as u32,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// 条件テンソルによる要素選択（`torch.where` 相当。イシュー #1637）。
    pub fn run_where_f32(
        &self,
        ctx: &MetalContext,
        cond: &[f32],
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        self.run_ternary(ctx, &self.where_f32, cond, a, b)
    }

    /// マスク位置を定数で置換する（`torch.masked_fill` 相当。イシュー
    /// #1637）。
    pub fn run_masked_fill_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        mask: &[f32],
        value: f32,
    ) -> Result<Vec<f32>, MetalError> {
        self.run_binary_scalar(ctx, &self.masked_fill_f32, x, mask, value)
    }

    /// 単項演算共通の起動手続き。[`Self::run_binary`] と同一構造。
    fn run_unary(
        &self,
        ctx: &MetalContext,
        pipeline: &MtlPipeline,
        a: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        validate_elementwise_len(a.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        // イシュー #1021: encode_binary_dispatch／encode_unary_dispatch は
        // numel 全要素を書き切る出力専用バッファのため alloc_uninit_pooled
        // を使う（設計文書 §6「A02」）。
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_unary_dispatch(encoder, pipeline, &a_buf, &out_buf, numel as u32);
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// `out[i] = a[i] + b[i]`（f32・同一長）。
    pub fn run_add_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        self.run_binary(ctx, &self.add_f32, a, b)
    }

    /// `out[i] = a[i] * b[i]`（f32・同一長）。
    pub fn run_mul_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        self.run_binary(ctx, &self.mul_f32, a, b)
    }

    /// `out[i] = max(a[i], 0)`（f32）。
    pub fn run_relu_f32(&self, ctx: &MetalContext, a: &[f32]) -> Result<Vec<f32>, MetalError> {
        self.run_unary(ctx, &self.relu_f32, a)
    }

    /// `out[i] = exp(a[i])`（f32、`metal::precise::exp`）。
    pub fn run_exp_f32(&self, ctx: &MetalContext, a: &[f32]) -> Result<Vec<f32>, MetalError> {
        self.run_unary(ctx, &self.exp_f32, a)
    }

    /// `out[i] = tanh(a[i])`（f32、`metal::precise::tanh`）。
    pub fn run_tanh_f32(&self, ctx: &MetalContext, a: &[f32]) -> Result<Vec<f32>, MetalError> {
        self.run_unary(ctx, &self.tanh_f32, a)
    }

    /// `op` に対応するコンパイル済みパイプラインを返す
    /// （[`Self::dispatch_binary_resident`] 専用の内部選択。ホスト版
    /// `run_add_f32`／`run_mul_f32` と同一カーネルを再利用するため bit
    /// 同一契約が成立する）。
    ///
    /// `BinaryElementwiseOp` は `#[non_exhaustive]`。未知 variant を
    /// `_ =>` で `add_f32` へフォールバックすると、将来 variant が
    /// 追加された際に「別の演算を代わりに計算して黙って成功する」
    /// fail-open になる（advisor 指摘。イシュー #1584。CUDA 側
    /// `elementwise.rs::function_for_binary` と同じ是正）。
    fn pipeline_for_binary(&self, op: BinaryElementwiseOp) -> Result<&MtlPipeline, MetalError> {
        match op {
            BinaryElementwiseOp::Add => Ok(&self.add_f32),
            BinaryElementwiseOp::Mul => Ok(&self.mul_f32),
            _ => Err(MetalError::UnsupportedElementwiseOp {
                detail: format!("unknown BinaryElementwiseOp variant: {op:?}"),
            }),
        }
    }

    /// [`Self::pipeline_for_binary`] の単項版。
    fn pipeline_for_unary(&self, op: UnaryElementwiseOp) -> Result<&MtlPipeline, MetalError> {
        match op {
            UnaryElementwiseOp::Relu => Ok(&self.relu_f32),
            UnaryElementwiseOp::Exp => Ok(&self.exp_f32),
            UnaryElementwiseOp::Tanh => Ok(&self.tanh_f32),
            _ => Err(MetalError::UnsupportedElementwiseOp {
                detail: format!("unknown UnaryElementwiseOp variant: {op:?}"),
            }),
        }
    }

    /// `a op b` を [`MetalBuffer`] 常駐のまま計算する（イシュー #1584。
    /// `tensor-core::BackendOps::binary_elementwise_device` の Metal
    /// 実装が呼ぶ）。CUDA 版（`backend-cuda::elementwise::
    /// launch_binary_resident`）と異なり、本メソッドは
    /// [`MetalContext::dispatch_sync`]（encode → `waitUntilCompleted`
    /// の同期版）を使う: Metal の encode-only（非同期・待たない）経路
    /// は失敗検出のために `*_tracked` 版・`failure_token` への登録が
    /// 必要になる設計上の要件があり（`linear_forward_device` が使う
    /// `encode_strided_bias_act_prepared` 系と同じ制約）、本イシューの
    /// スコープでは踏み込まない。そのため本 API は呼び出しごとに 1 回
    /// 同期する（H2D／D2H 相当の転送は発生しないが、CUDA 版の「同期点を
    /// 呼び出し元の `download` へ集約する」契約とは異なる）。
    pub(crate) fn dispatch_binary_resident(
        &self,
        ctx: &MetalContext,
        op: BinaryElementwiseOp,
        a: &MetalBuffer,
        b: &MetalBuffer,
        out: &MetalBuffer,
        numel: usize,
    ) -> Result<(), MetalError> {
        validate_elementwise_len(numel)?;
        // `a`／`b`／`out` の長さが `numel` と 1:1 対応することを明示検証
        // する（advisor 指摘: CUDA 版 `launch_binary_resident` と同じ
        // 是正。是正前は `numel` の u32 上限のみ検証しており、呼び出し元
        // が渡す `MetalBuffer` の実長との整合は未検証だった）。
        if a.len() != numel || b.len() != numel || out.len() != numel {
            return Err(MetalError::InvalidElementwiseShape {
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
        let pipeline = self.pipeline_for_binary(op)?;
        ctx.dispatch_sync(|encoder| {
            encode_binary_dispatch(encoder, pipeline, a, b, out, numel as u32);
        })
    }

    /// [`Self::dispatch_binary_resident`] の単項版。
    pub(crate) fn dispatch_unary_resident(
        &self,
        ctx: &MetalContext,
        op: UnaryElementwiseOp,
        a: &MetalBuffer,
        out: &MetalBuffer,
        numel: usize,
    ) -> Result<(), MetalError> {
        validate_elementwise_len(numel)?;
        // `dispatch_binary_resident` と同じ是正（advisor 指摘）。
        if a.len() != numel || out.len() != numel {
            return Err(MetalError::InvalidElementwiseShape {
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
        let pipeline = self.pipeline_for_unary(op)?;
        ctx.dispatch_sync(|encoder| {
            encode_unary_dispatch(encoder, pipeline, a, out, numel as u32);
        })
    }
}

/// `numel` に対する grid/threadgroup サイズを構築する（`div_ceil` による
/// 末尾ブロックの余剰スレッドはカーネル内境界チェックに委ねる契約。REQ-8。
/// `crate::gemm` の `THREADGROUP_SIDE`／grid 計算と同じ考え方の 1 次元版）。
fn ew_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: EW_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(EW_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// 二項カーネル共通のエンコード（バッファ結線 index 0〜2・`numel`
/// index 3・ディスパッチ）。[`MetalElementwise::run_binary`] が
/// [`MetalContext::dispatch_sync`] のクロージャから呼ぶ。
fn encode_binary_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`crate::gemm::encode_dispatch` の同種コメント参照）。`a_buf`／
    // `b_buf`／`out_buf` は呼び出し元 `ctx.dispatch_sync` が完了するまで
    // 生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 2);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。`numel` はローカル変数でありポインタは
    // 本呼び出し中生存し、長さは `size_of::<u32>()` と一致する
    // （`shaders/elementwise.metal` の `constant uint& numel` 宣言と型を
    // 揃える）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
    }

    let (threadgroups, threads_per_tg) = ew_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// 単項カーネル共通のエンコード（バッファ結線 index 0〜1・`numel`
/// index 2・ディスパッチ）。[`encode_binary_dispatch`] と同一構造だが
/// バッファ引数が 1 つ少ないため index が 1 つずつ前へずれる
/// （`shaders/elementwise.metal` の `ew_relu_f32`／`ew_exp_f32`／
/// `ew_tanh_f32` のバッファ宣言と一致させる）。
fn encode_unary_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_binary_dispatch` と同一の根拠（該当コメント参照）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: `encode_binary_dispatch` と同一の根拠（該当コメント参照）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
    }

    let (threadgroups, threads_per_tg) = ew_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// 3 項カーネル共通のエンコード（`where_cond` 専用。イシュー #1637）。
/// [`encode_binary_dispatch`] と同一構造で `cond`／`a`／`b` の 3 入力
/// バッファを index 0〜2、`out` を index 3、`numel` を index 4 へ結線
/// する（`shaders/elementwise.metal::ew_where_f32` のバッファ宣言と
/// 一致させる）。
fn encode_ternary_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    cond_buf: &MetalBuffer,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_binary_dispatch` と同一の根拠（該当コメント参照）。
    // `cond_buf`／`a_buf`／`b_buf`／`out_buf` は呼び出し元
    // `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(cond_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 3);
    }

    // SAFETY: `encode_binary_dispatch` と同一の根拠。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    let (threadgroups, threads_per_tg) = ew_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// 二項＋スカラー演算共通のエンコード（`masked_fill` 専用。イシュー
/// #1637）。`x`／`mask` を index 0〜1、`out` を index 2、`value`
/// （スカラー f32）を index 3、`numel` を index 4 へ結線する
/// （`shaders/elementwise.metal::ew_masked_fill_f32` のバッファ宣言と
/// 一致させる）。
fn encode_binary_scalar_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    mask_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    value: f32,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_binary_dispatch` と同一の根拠。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(mask_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 2);
    }

    // SAFETY: `value`／`numel` はローカル変数でありポインタは本呼び出し
    // 中生存し、長さはそれぞれ `size_of::<f32>()`／`size_of::<u32>()`
    // と一致する（`encode_binary_dispatch` と同一の根拠。`shaders/
    // elementwise.metal` の `constant float& value`／`constant uint&
    // numel` 宣言と型を揃える）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&value).cast(),
            std::mem::size_of::<f32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    let (threadgroups, threads_per_tg) = ew_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
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
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_len_accepts_small_len() {
        assert!(validate_elementwise_len(4).is_ok());
    }

    #[test]
    fn validate_elementwise_len_rejects_len_exceeding_u32() {
        let err = validate_elementwise_len(u32::MAX as usize + 1).unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_binary_dims_rejects_len_exceeding_u32_even_when_matching() {
        let len = u32::MAX as usize + 1;
        let err = validate_elementwise_binary_dims(len, len).unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn ew_dispatch_sizes_covers_all_elements_with_div_ceil() {
        let (threadgroups, threads_per_tg) = ew_dispatch_sizes(EW_THREADGROUP_WIDTH as u32 + 1);
        assert_eq!(threads_per_tg.width, EW_THREADGROUP_WIDTH);
        assert_eq!(threadgroups.width, 2);
    }

    #[test]
    fn validate_elementwise_ternary_dims_accepts_matching_lengths() {
        assert!(validate_elementwise_ternary_dims(4, 4, 4).is_ok());
    }

    #[test]
    fn validate_elementwise_ternary_dims_rejects_cond_mismatch() {
        let err = validate_elementwise_ternary_dims(4, 4, 5).unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_ternary_dims_rejects_a_mismatch() {
        let err = validate_elementwise_ternary_dims(4, 5, 4).unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }
}
