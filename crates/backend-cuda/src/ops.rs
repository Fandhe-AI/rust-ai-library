//! CUDA バックエンドの `BackendOps` 実装（TASK-1.9c・#46。イシュー #599 で
//! elementwise 5 演算・`gemm_bias_act` 実融合化を追加）。
//!
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の CUDA 実装。GEMM は
//! `gemm::CudaGemm::run_tiled_f32` へ委譲する（既存カーネル・許容誤差・
//! 境界検査には触れない）。`run_tiled_f32` 自体は内部で cp.async 3 stage
//! パイプラインカーネルへ形状条件付きに分岐しうる（整列形状のみ。
//! イシュー #1137・`gemm.rs::CudaGemm::select_tiled_f32_kernel`）ため、
//! 本ファイルのコードはこの分岐を意識せず既定 `run_tiled_f32` を呼ぶだけで
//! よい。elementwise（`add`／`mul`／`relu`／`exp`／
//! `tanh`）は `elementwise::CudaElementwise` へ委譲する（イシュー #599）。
//! 汎用 reduction（`sum`／`max`。全軸・単一軸）は `reduce::CudaReduce`
//! （`kernels_reduce.rs`。f64 アキュムレータ契約〈sum〉・厳密選択
//! `fmaxf`〈max〉）へ委譲する（イシュー #1584・親イシュー #1571。旧
//! `#599` スコープ外記述を解消）。DeviceBuffer 常駐版 elementwise
//! （`binary_elementwise_device`／`unary_elementwise_device`。イシュー
//! #1584）は `elementwise::CudaElementwise` の常駐起動 API
//! （`launch_binary_resident`／`launch_unary_resident`。H2D/D2H なし）へ
//! 委譲する。イシュー #592 で `run_fused` を
//! オーバーライドし、canonical RMSNorm 融合プラン（`x * rsqrt(sum(x^2))`）
//! 検出時のみ融合カーネル（[`crate::rmsnorm::CudaRmsNorm`]）へルーティング
//! する（`sum`／`max` 単独 API とは独立した経路）。
//!
//! `device.rs` の「動的ロード panic 回避ゲート」方針をそのまま踏襲する:
//! `CudaDevice::new` は driver 不在を `Err(CudaError::DriverUnavailable)`
//! で返す non-panicking な入口であり、本実装はこれを経由してから
//! `BackendError::CudaUnavailable` へ変換する（panic しない。
//! `.claude/rules/coding-rust.md`）。

use std::cell::Cell;
use std::sync::Arc;

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    Activation, BackendOps, BinaryElementwiseOp, DType, DispatchFailureCell, FusionPlan,
    GruBackwardOutput, GruPointwiseOutput, LstmPointwiseOutput, MatrixNormOrd, MseReduction,
    QrFactors, SegmentKey, SegmentResource, SegmentRun, ShapeError, SvdFactors, Tensor,
    UnaryElementwiseOp, reduce_out_shape, require_same_shape, row_norm_layout, row_softmax_layout,
};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::elementwise::CudaElementwise;
use crate::error::CudaError;
use crate::memory::{CudaBufferHandle, CudaMemory, CudaStorage, map_cuda_error};
use crate::rmsnorm::match_rmsnorm_plan;
use crate::softmax::match_softmax_plan;

thread_local! {
    /// `gemm_fp32_strict_impl`／`gemm_resident_lhs` の呼び出しのうち、
    /// 片側オペランドが dense な転置 view（[`dense_transposed_view`] が
    /// `Some` を返す形状）と判定できず `Tensor::contiguous()` の再パック
    /// コピー（またはそれに相当する `MemoryOps::upload` 内部の
    /// `contiguous()`）へフォールバックした回数（イシュー #1214）。
    /// `backend-cpu::ops::GEMM_HOST_REPACK_COUNT`（#1213）と同型の
    /// 可観測点で、`#[cfg(test)]` クレート内テストから「NT/TN 判定が
    /// 効いてフォールバックを通っていないこと」を検証するために使う
    /// （`pub(crate)`。クレート境界外の統合テストからは参照できないため、
    /// 外部テストファイルは数値一致のみを検証する契約とする）。
    pub(crate) static GEMM_HOST_REPACK_COUNT: Cell<u64> = const { Cell::new(0) };
}

/// `t` が「dense な転置格納」（`Tensor::transpose_2d()` を経た zero-copy
/// view のうち、元テンソルが行優先連続だったもの）であれば、その
/// storage をそのまま借用したフラットスライスを返す（イシュー #1214。
/// `backend-cpu::ops::dense_transposed_view`（#1213）と判定条件・契約が
/// 完全に同一の複製。`tensor-core`〈crates.io 公開クレート〉の公開 API
/// 拡張を避けるため backend-cuda 内に private 複製する。将来的な
/// `tensor-core` への昇格候補は PR 本文の out-of-scope に記す）。
///
/// 判定条件は `rank() == 2 && strides() == [1, shape()[0]]`。これは
/// `Tensor::transpose_2d`（`transpose(0,1)` の薄い委譲）が行優先連続
/// テンソルへ適用された結果と同値であり、返るスライスは「転置元の
/// テンソル」を行優先で並べたバイト列そのものになる（呼び出し元
/// `gemm_fp32_strict_impl`／`gemm_resident_lhs` が
/// `gemm::CudaGemm::run_tiled_f32_nt`／`run_tiled_f32_tn`／
/// `launch_tiled_f32_resident_nt` の `bt`／`at` 引数としてそのまま渡す
/// 前提）。
///
/// `narrow` 後の転置（一般 stride）・stride 0 の broadcast・rank ≠ 2 は
/// `None`（従来どおり `contiguous()` へフォールバックさせる。一般 stride
/// 化は本イシューのスコープ外。`docs/matmul-vjp-zero-copy-decision.md`
/// §3.2）。`rows == 0 || cols == 0` も呼び出し元の分岐を単純に保つため
/// `None` とし、`contiguous()`（`is_contiguous()` が空テンソルで常に
/// `true` を返す契約）に委ねる。
fn dense_transposed_view(t: &Tensor<f32>) -> Option<&[f32]> {
    if t.rank() != 2 {
        return None;
    }
    let shape = t.shape();
    let strides = t.strides();
    let (rows, cols) = (shape[0], shape[1]);
    if rows == 0 || cols == 0 {
        return None;
    }
    if strides.len() != 2 || strides[0] != 1 || strides[1] != rows as isize {
        return None;
    }
    let view = t.as_view_slice()?;
    if view.len() != rows.checked_mul(cols)? {
        return None;
    }
    Some(view)
}

/// RNN／LSTM／GRU セル演算（イシュー #1647）の入口検査: `shape` が
/// rank-2 であることを検証する（`backend-cpu::ops::require_rank2` と
/// 同型。平坦化後の要素数一致だけでは異形状の取り違えを検出できない
/// ため、`lstm_pointwise`／`lstm_hidden_backward`／`lstm_cell_backward`
/// ／`gru_pointwise`／`gru_backward` の各エントリで使う。codex-review
/// P2 指摘）。
fn require_rank2_cell(shape: &[usize]) -> Result<(), BackendError> {
    if shape.len() != 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    Ok(())
}

/// RNN／LSTM／GRU 系エントリが形状比較の前に必要とする `gates * hidden`
/// （ゲート幅）を `checked_mul` で検証する（`backend-cpu::ops::
/// checked_gate_width` と同型）。
///
/// 本番経路 panic 禁止（AGENTS.md）: `4 * hidden`／`3 * hidden` を
/// 未検証のまま `require_same_shape` の期待値へ埋め込むと、`hidden`
/// が `usize::MAX` 近傍（要素数 0 の空テンソルは `shape[1]` を自由に
/// 取れる）のとき乗算が overflow して期待幅が小さい値へ周回し、
/// 本来 shape mismatch で拒否すべき不正な入力を誤って受理してしまう
/// （受理後は `rnn_cell` 側カーネルが `hidden` を使った添字アクセス
/// で範囲外参照する）。イシュー #1647 codex-review P1 指摘。
fn checked_gate_width(gates: usize, hidden: usize) -> Result<usize, BackendError> {
    gates.checked_mul(hidden).ok_or(BackendError::ShapeMismatch(
        ShapeError::ElementCountMismatch {
            expected: usize::MAX,
            actual: 0,
        },
    ))
}

/// CUDA バックエンドの `BackendOps` 実装。`ordinal` は `Device::Cuda(_)`
/// の一致判定に使う `cudarc` のデバイス番号
/// （`CudaContext::new(ordinal)` に対応。`fandhe_ai_tensor_core::device::Device`
/// の doc コメント参照）。
///
/// イシュー #929: `CudaDevice`／`CudaGemm`／`CudaElementwise`／
/// `CudaRmsNorm`／`CudaSoftmax` は各メソッド呼び出し時に都度構築せず、
/// `crate::context_cache`（`ordinal` キーのプロセス内キャッシュ）経由で
/// 取得する。同一プロセス内の 2 回目以降の呼び出しは `CudaContext` 生成・
/// NVRTC コンパイルを再実行しない（`context_cache` モジュール冒頭コメント
/// 参照。実測根拠: `scripts/bench/framework-compare/results/
/// summary.md:177`）。エラー（driver 不在等）はキャッシュされず毎回
/// 再試行される（fail-fast 契約は不変）ため、`Self::device_handle` の
/// 戻り値型が `Result<..., BackendError>` である点・エラー伝播の意味論
/// 自体は変更しない。
#[derive(Debug, Clone, Copy)]
pub struct CudaBackendOps {
    ordinal: usize,
}

impl CudaBackendOps {
    /// GEMM 本体（f32）の FP32 厳密経路（`run_tiled_f32`）のみを実行する
    /// 内部ヘルパー。`crate::precision::gemm_precision()` の状態に
    /// 関わらず常に FP32 厳密で計算する（`Tf32`／`Tf32x3` いずれの
    /// opt-in モードも一切見ない）。
    ///
    /// `gemm`（公開経路。opt-in 時は `Tf32`／`Tf32x3` へ分岐しうる）・
    /// `gemm_bias_act`
    /// の `ComposedFallback`（非融合合成経路）・
    /// `BackendOps::gemm_fp32_strict`（`dyn BackendOps` 経由の学習経路
    /// 向け入口。イシュー #1211 codex-review 指摘・PR #1223）の 3 者から
    /// 呼ばれる。いずれも `self.gemm(a, b)` を直接呼ぶと `gemm` 側の TF32
    /// 分岐へ意図せず波及し、`crate::precision` モジュール冒頭コメントの
    /// 契約（「適用範囲は `CudaBackendOps::gemm`（素の f32 GEMM）のみ」）
    /// に反するため、必ずこのヘルパーを経由する（`gemm_bias_act` は
    /// codex-review 指摘・PR #1091 で同様の理由により導入済み）。
    ///
    /// イシュー #1214: 片側オペランドが dense な転置 view（
    /// [`dense_transposed_view`] が `Some` を返す形状）と判定できる場合、
    /// `Tensor::contiguous()` の再パックコピーを経由せず
    /// `gemm::CudaGemm::run_tiled_f32_nt`／`run_tiled_f32_tn`（GPU 側 smem
    /// 転置カーネルで転置してから既存 NN GEMM カーネルへ渡す）へ直接
    /// 渡す（CPU 版 #1213 と同型の判定・分岐）。両方転置（TT）・判定
    /// 不能（一般 stride・broadcast 等）・転置カーネル自体が使用不能
    /// （`new` 時のコンパイル失敗）な環境は従来どおり両オペランドを
    /// `contiguous()` で実体化し `run_tiled_f32` を呼ぶ（フォールバック
    /// でオペランドを再パックした回数は [`GEMM_HOST_REPACK_COUNT`] へ
    /// 計上する。`docs/matmul-vjp-zero-copy-decision.md` §4.3）。
    fn gemm_fp32_strict_impl(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        let gemm = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_gemm(&device)
            },
        )?;

        // 転置カーネル自体が使用不能（`CudaGemm::new` 時のコンパイル
        // 失敗）な環境では NT/TN 判定結果に関わらず常に従来経路へ
        // フォールバックする（`transpose_smem_f32_available` は driver
        // 呼び出しを伴わない静的照会のため `with_driver_call` の外で
        // 判定してよい。`tiled_f32_kernel_for` 等の可用性照会 API と
        // 同じ扱い）。
        let out = if gemm.transpose_smem_f32_available() {
            match (dense_transposed_view(a), dense_transposed_view(b)) {
                // TN: a は転置格納（at: 論理形状 [k,m] 行優先）、b は通常。
                // `crate::transpose::transpose_rows_fit_grid_y_limit(k)` は
                // `run_tiled_f32_tn` 内の `transpose_to_pooled(at_dev, k, m)`
                // が構築する grid.y（`k.div_ceil(TRANSPOSE_TILE)`）が CUDA の
                // グリッド次元上限を超えないことの事前検査（イシュー #1214
                // codex-review 指摘）。超過する場合は NN 経路（この制約を
                // 受けない）へフォールバックする。
                (Some(at), None) if crate::transpose::transpose_rows_fit_grid_y_limit(k) => {
                    if !b.is_contiguous() {
                        GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
                    }
                    let b_owned = b.contiguous();
                    let b_slice = b_owned.as_slice().ok_or_else(|| {
                        BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into())
                    })?;
                    self.with_driver_call(
                        &[],
                        |e| BackendError::KernelLaunchFailed(e.to_string()),
                        || gemm.run_tiled_f32_tn(at, b_slice, m, n, k),
                    )?
                }
                // NT: b は転置格納（bt: 論理形状 [n,k] 行優先）、a は通常。
                // 事前検査は上記 TN 分岐と同型で、対象は
                // `run_tiled_f32_nt` 内の `transpose_to_pooled(bt_dev, n, k)`
                // が構築する grid.y（`n.div_ceil(TRANSPOSE_TILE)`）。
                (None, Some(bt)) if crate::transpose::transpose_rows_fit_grid_y_limit(n) => {
                    if !a.is_contiguous() {
                        GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
                    }
                    let a_owned = a.contiguous();
                    let a_slice = a_owned.as_slice().ok_or_else(|| {
                        BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into())
                    })?;
                    self.with_driver_call(
                        &[],
                        |e| BackendError::KernelLaunchFailed(e.to_string()),
                        || gemm.run_tiled_f32_nt(a_slice, bt, m, n, k),
                    )?
                }
                _ => self.gemm_fp32_strict_fallback(&gemm, a, b, m, n, k)?,
            }
        } else {
            self.gemm_fp32_strict_fallback(&gemm, a, b, m, n, k)?
        };
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::gemm_fp32_strict_impl`] の従来経路（両オペランドを
    /// `contiguous()` で実体化してから `run_tiled_f32` を呼ぶ）。TT
    /// （両方転置）・判定不能形状・転置カーネル使用不能環境の 3 通り
    /// から共通で呼ばれる（イシュー #1214）。
    fn gemm_fp32_strict_fallback(
        &self,
        gemm: &crate::gemm::CudaGemm,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, BackendError> {
        if !a.is_contiguous() {
            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
        }
        if !b.is_contiguous() {
            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
        }
        // `run_tiled_f32` は contiguous な `&[f32]` を要求する（CPU 実装と
        // 同じ契約。`ops.rs`（backend-cpu）参照）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;
        self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || gemm.run_tiled_f32(a_slice, b_slice, m, n, k),
        )
    }

    /// [`BackendOps::gemm_fp32_strict_into`]／[`BackendOps::
    /// gemm_fp32_strict_into_tracked`] 共通の検証・ディスパッチ本体
    /// （イシュー #1559。`backend-metal::ops::
    /// MetalBackendOps::gemm_fp32_strict_into_impl` と同型の二重化回避
    /// パターンだが、CUDA には Metal のような共有コマンドバッファ
    /// バッチング問題がないため `token` 引数は持たない——
    /// [`BackendOps::gemm_fp32_strict_into_tracked`] のトレイト側 doc
    /// 「CUDA は既定のままでよい」参照。実際に `_tracked` 側の
    /// オーバーライドは本メソッドへ委譲するだけの薄い明示委譲であり、
    /// `context_cache` のポイズン検査（`begin_driver_call`／
    /// `observe_cuda_result`）は本メソッド自身が呼び出しごとに同期的に
    /// 完結させる）。
    ///
    /// NT/TN 以外（NN・TT・分類不能形状・退化形状・転置カーネル使用
    /// 不能環境）は `Unsupported` を返さず、ホスト経路
    /// `gemm_fp32_strict`（trait 契約上 `Self::gemm_fp32_strict_impl` と
    /// bit 同一）の戻り値を `CudaMemory::upload_into` で `out` へ
    /// 書き込むフォールバックにする（`backend-metal` の同名フォール
    /// バックと同じ理由——`DeviceParamStore::fill_resident_weight_grad`
    /// は最初の `Unsupported` でストア全体の `resident_grad_capability`
    /// を `Some(false)` にキャッシュするため、形状単位で `Unsupported`
    /// を返すと同一 backward 内で先に成功済みの resident slot が読めなく
    /// なる〈`MissingGradient`〉）。
    fn gemm_fp32_strict_into_impl(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
    ) -> Result<(), BackendError> {
        // codex-review 指摘（PR #1569）対応: capture 中の拒否は
        // NT/TN 経路の `with_sync_point_call`（後続）だけでなく、本関数
        // 全体の共通入口でも行う。NN/TT・分類不能・退化形状はフォール
        // バック（`gemm_fp32_strict` → `run_tiled_f32` 系）へ進み、
        // そちらは内部で `Self::with_driver_call`（`gemm.rs` 側。
        // capture 中でも同一スレッドなら通す設計）経由の同期的な
        // D2H readback（`clone_dtoh` 相当）を伴う——このホスト
        // ブロッキング読み出しは capture 対象にできない（グラフに記録
        // できるのは driver 呼び出しの列であり、ホスト側の同期完了待ちは
        // 記録できない）ため、そもそも `with_sync_point_call` を経由
        // しない分岐が存在する。加えて NT/TN 経路自体も、`cached_gemm`
        // 取得（後続の `with_driver_call`）や NT/TN 判定用の
        // `dense_transposed_view`／`contiguous()` 呼び出しより後で
        // ようやく `with_sync_point_call` に到達するため、それより前の
        // 区間（判定自体は driver 非依存だが `cached_gemm` の
        // `with_driver_call` は driver に触れる）で capture 状態を
        // 分岐ごとに個別判定するのではなく、最初の driver 呼び出しより
        // 前の本関数入口 1 箇所で一律拒否する（`docs/
        // backend-cuda-async-execution-design.md` §15「同期点は driver
        // 操作前に拒否する」契約）。`context_cache::
        // is_capturing_on_current_thread` は driver に触れない純粋な
        // レジストリ照会（`begin_sync_point_call` 内部と同じ判定）。
        if context_cache::is_capturing_on_current_thread(self.ordinal) {
            return Err(BackendError::Unsupported(
                "cuda graph capture: gemm_fp32_strict_into is a host synchronization point and \
                 cannot be captured"
                    .into(),
            ));
        }

        if out.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let _ = out_shape; // shape 検証のみに使用（`gemm_fp32_strict_into` の CPU/Metal 実装と同型）

        // REQ-8「カーネル側の手動境界チェックを省略しない」・OWASP A03:
        // `out_offset + m*n` を `checked_mul`/`checked_add` で検査し、
        // `out.numel()` を超える書き込みを driver 呼び出し（NT/TN 判定・
        // カーネル起動）より前に拒否する。
        let mn = m.checked_mul(n).ok_or_else(|| {
            BackendError::InvalidArgument("gemm_fp32_strict_into: m * n overflowed usize".into())
        })?;
        let end = out_offset.checked_add(mn).ok_or_else(|| {
            BackendError::InvalidArgument(
                "gemm_fp32_strict_into: out_offset + m * n overflowed usize".into(),
            )
        })?;
        if end > out.numel() {
            return Err(BackendError::InvalidArgument(format!(
                "gemm_fp32_strict_into: write range [{out_offset}, {end}) exceeds out buffer \
                 length {}",
                out.numel()
            )));
        }

        // 退化形状ガード（`m == 0 || n == 0 || k == 0`）は分類（NT/TN
        // 判定）を行わず直接フォールバックへ進む。理由: `dense_
        // transposed_view(t)` は `t` 自身の shape が 0 次元を含む場合
        // `None` を返す実装のため、`m == 0` のとき
        // `dense_transposed_view(a)` が `None` になる一方で
        // `dense_transposed_view(b)`（`b` は shape `[k, n]` で `k`／`n`
        // が非ゼロなら非退化）が `Some` を返しうる——`(None, Some(bt))`
        // という NT パターンに誤って一致しうる（`n == 0` も対称的に
        // TN パターンへの誤一致を起こしうる）。`k == 0` 自体は
        // `launch_tiled_f32_nt_into`／`_tn_into` 側に `zero_fill`
        // （`CudaArgMut::zero_fill`）による正しい処理があるが、`m == 0`
        // ／`n == 0` にはこの誤分類を吸収する手段がないため、3 条件を
        // まとめて一律フォールバックへ回し個別のゼロ埋め分岐を持たない
        // ようにする（フォールバックの `gemm_fp32_strict` 自身が
        // `run_tiled_f32_nt`／`_tn`／`run_tiled_f32` いずれの経路でも
        // `m==0||n==0`／`k==0` を正しく処理する）。
        if m != 0 && n != 0 && k != 0 {
            let gemm = self.with_driver_call(
                &[],
                |e| BackendError::CudaUnavailable(e.to_string()),
                || {
                    let device = self.device_handle_raw()?;
                    context_cache::cached_gemm(&device)
                },
            )?;

            if gemm.transpose_smem_f32_available() {
                let (m32, n32, k32) = (m as u32, n as u32, k as u32);
                match (dense_transposed_view(a), dense_transposed_view(b)) {
                    (Some(at), None) if crate::transpose::transpose_rows_fit_grid_y_limit(k32) => {
                        if !b.is_contiguous() {
                            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
                        }
                        let b_owned = b.contiguous();
                        let b_slice = b_owned.as_slice().ok_or_else(|| {
                            BackendError::KernelLaunchFailed(
                                "gemm_fp32_strict_into: rhs not contiguous".into(),
                            )
                        })?;
                        let out_gen = out.generation();
                        let out_handle = out
                            .downcast_handle_mut::<CudaBufferHandle>()
                            .ok_or(BackendError::DeviceMismatch)?;
                        let storage = out_handle.storage.as_mut().ok_or_else(|| {
                            BackendError::DeviceAllocationFailed(
                                "gemm_fp32_strict_into: out buffer has numel > 0 but no device \
                                 allocation"
                                    .into(),
                            )
                        })?;
                        return self.with_sync_point_call(
                            &[out_gen],
                            "gemm_fp32_strict_into",
                            |e| BackendError::KernelLaunchFailed(e.to_string()),
                            || {
                                let mut c_arg = storage.view_mut(out_offset..end);
                                gemm.launch_tiled_f32_tn_into(
                                    at, b_slice, &mut c_arg, m32, n32, k32,
                                )
                            },
                        );
                    }
                    (None, Some(bt)) if crate::transpose::transpose_rows_fit_grid_y_limit(n32) => {
                        if !a.is_contiguous() {
                            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
                        }
                        let a_owned = a.contiguous();
                        let a_slice = a_owned.as_slice().ok_or_else(|| {
                            BackendError::KernelLaunchFailed(
                                "gemm_fp32_strict_into: lhs not contiguous".into(),
                            )
                        })?;
                        let out_gen = out.generation();
                        let out_handle = out
                            .downcast_handle_mut::<CudaBufferHandle>()
                            .ok_or(BackendError::DeviceMismatch)?;
                        let storage = out_handle.storage.as_mut().ok_or_else(|| {
                            BackendError::DeviceAllocationFailed(
                                "gemm_fp32_strict_into: out buffer has numel > 0 but no device \
                                 allocation"
                                    .into(),
                            )
                        })?;
                        return self.with_sync_point_call(
                            &[out_gen],
                            "gemm_fp32_strict_into",
                            |e| BackendError::KernelLaunchFailed(e.to_string()),
                            || {
                                let mut c_arg = storage.view_mut(out_offset..end);
                                gemm.launch_tiled_f32_nt_into(
                                    a_slice, bt, &mut c_arg, m32, n32, k32,
                                )
                            },
                        );
                    }
                    _ => {}
                }
            }
        }

        // フォールバック（NN・TT・分類不能形状・退化形状・転置カーネル
        // 使用不能環境）: `Unsupported` を返さず常に正しい結果を書き込む
        // （`gemm_fp32_strict`〈内部で自身の poison/世代検査・NT/TN 判定・
        // repack 計上を完結する〉→ `CudaMemory::upload_into`〈H2D。
        // `gemm_resident_lhs` のフォールバックと同型で新規 `DeviceBuffer`
        // を escape させない〉）。
        let result = self.gemm_fp32_strict(a, b)?;
        let device = self.with_driver_call(
            &[out.generation()],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        let mem = CudaMemory::new(&device);
        mem.upload_into(&result, out, out_offset)
    }

    /// 指定した `ordinal` に対応する `CudaBackendOps` を構築する。
    /// 構築自体は driver 初期化を行わないため常に成功する（実際の
    /// driver 呼び出しは各メソッドが `Self::device_handle`（`context_cache`
    /// 経由）を呼んだ時点）。
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    /// `context_cache::cached_device` を経由してデバイスハンドルを取得
    /// する（イシュー #929。プロセス内キャッシュのヒット時は
    /// `CudaContext::new` を再実行しない）。driver 不在・初期化失敗は
    /// `BackendError::CudaUnavailable` へ変換する（panic 回避ゲートは
    /// `CudaDevice::new` 内部で完結する。`device.rs` 参照）。
    ///
    /// **poison 検査を経由しない**（codex-review P0 指摘・PR #1064 追補・
    /// `ops.rs:147` 相当: `device_handle()` はキャッシュミス時に
    /// `CudaDevice::new` を呼び実際に driver を操作するが、これを
    /// `with_driver_call`（`begin_driver_call` によるポイズン検査を含む）
    /// より前に呼ぶと、poison 済み ordinal でも拒否前に driver 操作が
    /// 走ってしまい、その失敗も観測されない）。そのため本メソッドは
    /// [`Self::memory_ops`]／[`Self::device_memory_pool_stats`] という
    /// 「driver へは触れず `Option` で fail-safe に縮退する」経路専用に
    /// 限定して使い、driver を実際に操作する演算（`gemm`／`elementwise`
    /// 等）は [`Self::device_handle_raw`] を `with_driver_call` の
    /// クロージャ内部から呼ぶ（poison 検査の後）。
    fn device_handle(&self) -> Result<Arc<CudaDevice>, BackendError> {
        self.device_handle_raw()
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))
    }

    /// `resources` から [`SegmentResource`] 列を導出する（イシュー
    /// #1349）。`captured_segment_key`（capture 対象キーの新規発行）・
    /// `run_captured_sgd_step_segment`（replay 直前の再検証。codex-review P0
    /// 指摘対応）の双方が同じロジックを使う必要がある——**2 箇所で
    /// アドレス導出ロジックが乖離すると、片方だけを見て再検証が
    /// 「常に一致する」だけの無意味な処理へ形骸化しうる**ため、本関数へ
    /// 一本化する。`numel == 0`（空バッファ）は `addr == 0` とする契約
    /// （`SegmentResource` doc コメント参照）。
    fn segment_resources_for(
        resources: &[&fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>],
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<Vec<SegmentResource>, BackendError> {
        let mut segment_resources = Vec::with_capacity(resources.len());
        for buf in resources {
            let numel = buf.numel();
            let addr = if numel == 0 {
                0u64
            } else {
                let handle = buf
                    .downcast_handle::<CudaBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                let storage = handle.storage.as_ref().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "segment_resources_for: buffer has numel > 0 but no device allocation"
                            .to_string(),
                    )
                })?;
                // `CudaStorage::Device`／`Managed`（イシュー #1352）の
                // いずれも `cudarc::driver::DevicePtr` を実装するため、
                // graph capture 対象の segment key（アドレス比較による
                // 差し替え検出）は配置に依らず同じ経路で導出できる
                // （`crate::memory::CudaStorage::as_arg` と同じ分岐方針）。
                match storage {
                    CudaStorage::Device(slice) => {
                        let (ptr, _sync) = cudarc::driver::DevicePtr::device_ptr(slice, stream);
                        ptr
                    }
                    CudaStorage::Managed(unified) => {
                        let (ptr, _sync) = cudarc::driver::DevicePtr::device_ptr(unified, stream);
                        ptr
                    }
                }
            };
            segment_resources.push(SegmentResource { addr, numel });
        }
        Ok(segment_resources)
    }

    /// [`Self::device_handle`] の `CudaError` 版。`with_driver_call` の
    /// クロージャ内部（＝ `begin_driver_call` によるポイズン検査の後）から
    /// 呼ぶことで、`CudaDevice::new`（キャッシュミス時の driver 初期化）
    /// 自体も poison 検査・sticky エラー観測の対象に含める
    /// （codex-review P0 指摘・PR #1064 追補）。
    fn device_handle_raw(&self) -> Result<Arc<CudaDevice>, CudaError> {
        context_cache::cached_device(self.ordinal)
    }

    /// `BackendOps` の各公開メソッドが唯一の driver 呼び出し境界として
    /// 使う共通ヘルパー（イシュー #1013 設計文書 §9 item 7・9。PR #1064
    /// の Phase C 結線。`memory.rs::CudaMemory::with_driver_call` と同じ
    /// 設計）。
    ///
    /// `context_cache::begin_driver_call` を演算入口で 1 回だけ呼び
    /// （`resource_generations` には当該演算が読み書きするデバイス常駐
    /// バッファ〈`DeviceBuffer`／`DeviceBufferView`〉の
    /// [`fandhe_ai_tensor_core::buffer::DeviceBuffer::generation`] を渡す。
    /// ホスト `Tensor` のみを読み書きする演算〈`gemm`／`add`／`relu` 等〉
    /// には検査対象の既存デバイス常駐バッファがないため空スライスでよく、
    /// これは検査を省略する fail-open ではなく「1 回の呼び出し内で
    /// 完結し、跨ぐ世代が存在しない」ことに対応する）、`f` の内部で
    /// `gemm.rs`／`elementwise.rs`／`softmax.rs`／`rmsnorm.rs`／`sgd.rs`
    /// が行う 1 回以上の driver 呼び出しの結果（`?` で直結しているため
    /// 呼び出し元まで伝播する `CudaError` は常に最初に失敗した 1 回を
    /// 表す）を `observe_cuda_result` で観測し、sticky エラーなら
    /// ordinal を poison する。
    ///
    /// **cold-cache 構築も同じ境界に含める**（Cursor Bugbot 指摘・
    /// PR #1064 追補）: `context_cache::cached_gemm`／`cached_elementwise`／
    /// `cached_rmsnorm`／`cached_softmax`／`cached_sgd` はキャッシュミス時
    /// （初回呼び出し、または将来 `invalidate` が新世代のコンテキストを
    /// 再構築した直後）に NVRTC コンパイル・モジュールロードという実際の
    /// driver 呼び出しを行う。この構築呼び出しを `with_driver_call` の
    /// 外側（`device_handle()` 直後）で素通しに実行すると、構築中に
    /// sticky エラーが発生しても観測されず ordinal が poison されない
    /// まま fail-open になる（構築失敗自体はキャッシュされず毎回
    /// 再試行されるため、poison されない限りこの経路は永久に「観測なしで
    /// 消費される」窓になる）。各呼び出し元は `cached_*` 取得自体も
    /// 本ヘルパーで包む（`f` に `context_cache::cached_gemm(&device)` 等を
    /// 渡す）ことでこの窓を閉じる。`resource_generations` は続く実行部と
    /// 同じ値を渡し（`begin_driver_call` は世代不一致以外の目的では
    /// 副作用を持たないため二重に渡しても安全）、構築用のトークンと
    /// 実行用のトークンは別個に取得・解放する。
    fn with_driver_call<T>(
        &self,
        resource_generations: &[u64],
        map: impl FnOnce(CudaError) -> BackendError,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, BackendError> {
        let token = context_cache::begin_driver_call(self.ordinal, resource_generations)?;
        context_cache::observe_cuda_result(self.ordinal, &token, f()).map_err(map)
    }

    /// [`Self::with_driver_call`] と同じだが、CUDA Graph capture 中
    /// （イシュー #1349・`docs/backend-cuda-graph-step-capture-design.md`
    /// §4.2）は driver に触れる前に拒否する（`context_cache::
    /// begin_sync_point_call`。`memory.rs::CudaMemory::
    /// with_sync_point_call` と同型で、`CudaBackendOps` 側にも同じ排他が
    /// 要る呼び出し向けに複製する）。イシュー #1559 の
    /// `gemm_fp32_strict_into` NT/TN 経路（関数内で明示 `stream.
    /// synchronize()` を行うホストブロック型の同期点。`docs/
    /// backend-cuda-async-execution-design.md` §15）が使う。`what` は
    /// 診断メッセージ用の呼び出し名。
    fn with_sync_point_call<T>(
        &self,
        resource_generations: &[u64],
        what: &'static str,
        map: impl FnOnce(CudaError) -> BackendError,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, BackendError> {
        let token = context_cache::begin_sync_point_call(self.ordinal, resource_generations, what)?;
        context_cache::observe_cuda_result(self.ordinal, &token, f()).map_err(map)
    }

    /// 二項 elementwise 共通のディスパッチ（`add`／`mul`）。
    ///
    /// `Tensor::broadcast_with`（NumPy 互換ブロードキャスト。CPU
    /// `elementwise::binary_elementwise` と同じ意味論）で共通 shape の
    /// view を得たのち `contiguous()` で密なバッファへ実体化してから
    /// `CudaElementwise`（同一長バッファのみを扱う。`elementwise.rs` 冒頭
    /// コメント「ブロードキャスト」参照）へ渡す。`run` は
    /// `CudaElementwise::run_add_f32`／`run_mul_f32` のいずれかを呼ぶ
    /// クロージャとして呼び出し側から注入される。
    fn elementwise_binary(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        run: impl FnOnce(&CudaElementwise, &[f32], &[f32]) -> Result<Vec<f32>, CudaError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let (a_bc, b_bc) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let out_shape = a_bc.shape().to_vec();

        let a_owned = a_bc.contiguous();
        let b_owned = b_bc.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: lhs not contiguous".into())
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: rhs not contiguous".into())
        })?;

        let ew = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_elementwise(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&ew, a_slice, b_slice),
        )?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 3 項 elementwise 共通のディスパッチ（`where_cond`。イシュー
    /// #1637）。[`Self::elementwise_binary`] と異なり broadcast は
    /// 行わない（`BackendOps::where_cond` doc の契約どおり、`cond`／
    /// `a`／`b` は呼び出し元〈`Var::where_cond`〉が同一 `out_shape` へ
    /// 実体化済みで渡す前提。ここでは形状一致を再検査するのみ・
    /// `.claude/rules/security.md` A08）。
    fn elementwise_ternary(
        &self,
        cond: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        run: impl FnOnce(&CudaElementwise, &[f32], &[f32], &[f32]) -> Result<Vec<f32>, CudaError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = a.shape().to_vec();
        if cond.shape() != out_shape.as_slice() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: cond.shape().to_vec(),
                rhs: out_shape,
            }));
        }
        if b.shape() != out_shape.as_slice() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: b.shape().to_vec(),
                rhs: out_shape,
            }));
        }

        let cond_owned = cond.contiguous();
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let cond_slice = cond_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: cond not contiguous".into())
        })?;
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: lhs not contiguous".into())
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: rhs not contiguous".into())
        })?;

        let ew = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_elementwise(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&ew, cond_slice, a_slice, b_slice),
        )?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 二項＋スカラー elementwise 共通のディスパッチ（`masked_fill`。
    /// イシュー #1637）。broadcast は行わない（`BackendOps::
    /// masked_fill` doc の契約どおり `mask` は `x` と同一 shape）。
    fn elementwise_binary_scalar(
        &self,
        x: &Tensor<f32>,
        mask: &Tensor<f32>,
        value: f32,
        run: impl FnOnce(&CudaElementwise, &[f32], &[f32], f32) -> Result<Vec<f32>, CudaError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = x.shape().to_vec();
        if mask.shape() != out_shape.as_slice() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: mask.shape().to_vec(),
                rhs: out_shape,
            }));
        }

        let x_owned = x.contiguous();
        let mask_owned = mask.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: lhs not contiguous".into())
        })?;
        let mask_slice = mask_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: mask not contiguous".into())
        })?;

        let ew = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_elementwise(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&ew, x_slice, mask_slice, value),
        )?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 単項 elementwise 共通のディスパッチ（`relu`／`exp`／`tanh`）。
    /// ブロードキャストが発生しない点を除き [`Self::elementwise_binary`]
    /// と同一構造。
    fn elementwise_unary(
        &self,
        a: &Tensor<f32>,
        run: impl FnOnce(&CudaElementwise, &[f32]) -> Result<Vec<f32>, CudaError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = a.shape().to_vec();
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: input not contiguous".into())
        })?;

        let ew = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_elementwise(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&ew, a_slice),
        )?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::sum`]／[`Self::max`] 共通のディスパッチ（イシュー #1584）。
    /// `reduce_out_shape` で `dim` を検査・出力 shape を導出した後、
    /// `a.contiguous()` を `reduce::CudaReduce` の該当エントリ
    /// （`kind` で分岐）へ渡す。`dim = Some(axis)` は `reduce::
    /// reduce_axis_layout` で `(outer, axis_len, inner)` を導出する
    /// （`ops.rs` 側は shape の正しさのみ検査し、`i32::MAX` 上限等の
    /// カーネル起動前検証は `reduce.rs` 側に委ねる二重責務分離）。
    fn reduce_dispatch(
        &self,
        a: &Tensor<f32>,
        dim: Option<usize>,
        kind: ReduceKind,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = reduce_out_shape(a.shape(), dim).map_err(BackendError::ShapeMismatch)?;
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("reduce: input not contiguous".into())
        })?;

        let reduce = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_reduce(&device)
            },
        )?;

        let data = match dim {
            None => {
                let value = self.with_driver_call(&[], map_reduce_error, || match kind {
                    ReduceKind::Sum => reduce.run_sum_all_f32(a_slice),
                    ReduceKind::Max => reduce.run_max_all_f32(a_slice),
                })?;
                vec![value]
            }
            Some(axis) => {
                let (outer, axis_len, inner) =
                    crate::reduce::reduce_axis_layout(a_owned.shape(), axis)
                        .map_err(map_reduce_error)?;
                self.with_driver_call(&[], map_reduce_error, || match kind {
                    ReduceKind::Sum => reduce.run_sum_axis_f32(a_slice, outer, axis_len, inner),
                    ReduceKind::Max => reduce.run_max_axis_f32(a_slice, outer, axis_len, inner),
                })?
            }
        };
        Tensor::new(data, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::run_fused`] の RMSNorm 一致経路（イシュー #592）。
    /// `match_rmsnorm_plan` が一致した後の dtype／leaf 数／leaf shape の
    /// 起動前 fail-closed 検証と、`CudaRmsNorm::run_rmsnorm_f32_raw`
    /// （`inv_n = 1.0`・`eps = 0.0`・`w = None`）への委譲を行う。
    fn run_fused_rmsnorm(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        // `match_rmsnorm_plan` は op 列・leaf 数・`row_fusion()` の形状
        // のみを照合し、`FusionPlan::from_ops` が受理しうる任意の
        // `dtype`（`FusionPlan` の DTO は現状 `DType` を素通しする。
        // `plan.rs` §2.1 参照）を検査しない。カーネル起動前に
        // `plan.dtype() == DType::F32` を明示検証しないと、例えば
        // `DType::F64` のプランでも f32 CUDA カーネルとして実行されて
        // しまう（`backend-cpu::fused_elementwise::run_fused_elementwise`
        // が実施する同種の fail-closed 検証との不整合。codex-review
        // 指摘・PR #706 レビュー）。
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: unsupported dtype {:?} (canonical RMSNorm fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: canonical RMSNorm プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        // leaf の shape が `plan.output_shape()` と一致することも明示
        // 検証する。`match_rmsnorm_plan` は要素数（`row_fusion().row_len()`）
        // のみを照合するため、要素数が一致しつつ shape（次元分割）が
        // 異なる leaf（例: `[8]` に対する `[2, 4]`）を渡しても
        // `run_rmsnorm_f32_raw` の長さ検証だけでは検出できない。canonical
        // プランは `axis: None`（全軸縮約）で `x` と出力の shape が恒等
        // （elementwise 型の最終 Mul）である契約のため、ここで shape 恒等
        // を fail-closed に強制する（`backend-cpu::fused_elementwise` の
        // leaf shape 検証と同じ契約）。
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: rmsnorm input not contiguous".into())
        })?;

        let rmsnorm = self.with_driver_call(&[], map_fused_kernel_init_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_rmsnorm(&device)
        })?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rmsnorm.run_rmsnorm_f32_raw(x_slice, None, 0.0, 1.0, 1, hidden),
        )?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::run_fused`] の softmax 一致経路（イシュー #594）。
    /// `run_fused_rmsnorm` と同じ起動前 fail-closed 検証パターン（dtype
    /// F32 限定・leaf 1 個・leaf shape 恒等）を踏襲し、
    /// `CudaSoftmax::run_softmax_f32_raw` を `scale = log2(e)` で呼ぶ。
    fn run_fused_softmax(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: unsupported dtype {:?} (canonical softmax fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: canonical softmax プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        // `run_fused_rmsnorm` と同じ理由（要素数一致だけでは shape の
        // 取り違えを検出できない）で leaf shape の恒等性を明示検証する。
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: softmax input not contiguous".into())
        })?;

        let softmax = self.with_driver_call(&[], map_fused_kernel_init_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_softmax(&device)
        })?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || softmax.run_softmax_f32_raw(x_slice, std::f32::consts::LOG2_E, rows, cols),
        )?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// [`CudaBackendOps::gemm_bias_act`] が融合カーネル
/// （`gemm::CudaGemm::run_tiled_bias_act_f32`）と
/// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト実装
/// （非融合 `gemm`→`add`→`relu` 3 段合成）のどちらを経由するかを表す。
///
/// `backend-cpu::ops::CpuBackendOps::gemm_bias_act` の分岐条件
/// （`bias` が `None`、または `bias.shape()` が厳密に `[n]`
/// の場合にのみ融合カーネルへ進む）と同一の意味論を CUDA 側にも適用する
/// （バックエンド間で `gemm_bias_act` の経路依存の挙動差を作らない。
/// イシュー #203 Review 指摘と同じ理由）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmBiasActRoute {
    /// 融合カーネル（epilogue 内で bias 加算・activation を適用）へ進む。
    Fused,
    /// デフォルト実装（`gemm`→`add`→act の非融合合成）へフォールバックする。
    ComposedFallback,
}

/// [`crate::rmsnorm::CudaRmsNorm::new`]／[`crate::softmax::CudaSoftmax::new`] の初期化失敗を
/// `BackendError` へ変換する（純関数。実機なしで単体テスト可能）。
///
/// `CudaError::DriverUnavailable`／`NvrtcUnavailable` のみを環境不在
/// （`BackendError::CudaUnavailable`。CUDA/NVRTC 非搭載環境での早期
/// フォールバックを想定した variant）として扱う。それ以外
/// （NVRTC コンパイルエラー・関数ロード失敗・デバイス属性負値検出の
/// `InvalidKernelDescriptor` 等）を一律 `CudaUnavailable` に丸めると、
/// CUDA/NVRTC が利用可能な環境でもカーネル実装側の回帰が「環境不在」に
/// 化けて握りつぶされる（`tests/rmsnorm_parity.rs` の env-adaptive
/// スモークテストは `CudaUnavailable` を無条件に成功扱いするため。
/// codex-review 指摘・PR #706 レビュー）。よって環境不在の既知 variant
/// 以外は `BackendError::KernelLaunchFailed` として実装回帰を検出できる
/// ようにする（`memory.rs::map_cuda_error` と同じ variant 分岐方針。
/// `#[non_exhaustive]` の `CudaError` に対する将来 variant 追加への
/// フォールバックとして `KernelLaunchFailed` を wildcard の受け皿とする
/// 点も揃える）。
///
/// イシュー #594: 判定ロジックは RMSNorm 固有ではなく `CudaError` の
/// variant 分岐のみに依るため、`run_fused` の softmax ルーティング
/// （[`crate::softmax::CudaSoftmax::new`] の初期化失敗変換）でもそのまま共用する（実装
/// 計画 §3.4「初期化エラー変換は共通化」。旧名 `map_rmsnorm_init_error`
/// から RMSNorm 専用でない名前へ改名した）。
fn map_fused_kernel_init_error(err: CudaError) -> BackendError {
    match err {
        CudaError::DriverUnavailable { detail } => BackendError::CudaUnavailable(detail),
        CudaError::NvrtcUnavailable { detail } => BackendError::CudaUnavailable(detail),
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// [`GemmBiasActRoute`] の選択ロジック（純関数。実機なしで単体テスト可能。
/// 本ファイル末尾 `#[cfg(test)]` 参照）。
///
/// `bias_shape` は呼び出し元の `bias.map(|t| t.shape())`、`n` は
/// `B: [k, n]` の列数。`bias_shape` が `None`（bias 指定なし）または
/// 厳密に `[n]`（行方向複製）の場合にのみ [`GemmBiasActRoute::Fused`] を
/// 返す。`pub(crate)`: `CudaBackendOps::gemm_bias_act` から呼ばれる。
pub(crate) fn gemm_bias_act_route(bias_shape: Option<&[usize]>, n: usize) -> GemmBiasActRoute {
    match bias_shape {
        None => GemmBiasActRoute::Fused,
        Some(shape) if shape == [n] => GemmBiasActRoute::Fused,
        Some(_) => GemmBiasActRoute::ComposedFallback,
    }
}

/// `ordinal` に対応する `&'static CudaMemory` をプロセス内キャッシュから
/// 取得する（イシュー #935）。
///
/// `BackendOps::memory_ops(&self) -> Option<&dyn MemoryOps>` は戻り値の
/// 参照を `&self`（`CudaBackendOps`。`ordinal: usize` のみを持つ軽量な
/// `Copy` 値で、呼び出しのたびに新規構築されうる）の寿命に束縛できる型で
/// 返す必要がある一方、`AllocationTracker` の計測系列（`docs/
/// device-resident-update-design.md` §3.3d「計測系列単一化」）を維持する
/// には `CudaMemory` 自体をプロセス全体で 1 個だけ共有しなければならない。
/// `context_cache`（`Arc<T>` を返す）はこの用途に使えない（`Arc` の中身は
/// `&self` の寿命へ縮小できるが、`Arc` 自体をどこかに所有し続ける主体が
/// 必要で、`CudaBackendOps` 自身はフィールド追加不可の `Copy` 値のため
/// 保持先がない）ため、本関数は `Box::leak` で `'static` 参照へ格上げして
/// 保持する。`ordinal`（物理 GPU 台数で有界）をキーとする点は
/// `context_cache` と同じ「エントリはプロセスの生存期間中 evict されない」
/// 設計（`context_cache.rs` モジュール冒頭コメント「所有モデル・生存
/// 期間」）に倣った意図的なリークであり、通常のメモリリークとは区別する。
fn static_cuda_memory(
    ordinal: usize,
    device: &CudaDevice,
) -> Result<&'static CudaMemory, BackendError> {
    use std::collections::HashMap;
    use std::sync::Mutex;

    static CACHE: std::sync::OnceLock<Mutex<HashMap<usize, &'static CudaMemory>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().map_err(|_| {
        BackendError::DeviceUnavailable("static_cuda_memory: cache mutex poisoned".to_string())
    })?;
    if let Some(mem) = guard.get(&ordinal) {
        return Ok(mem);
    }
    let mem: &'static CudaMemory = Box::leak(Box::new(CudaMemory::new(device)));
    guard.insert(ordinal, mem);
    Ok(mem)
}

/// [`CudaBackendOps::reduce_dispatch`] が `sum`／`max` のどちらを実行
/// するかを選ぶ内部専用の選択子（イシュー #1584）。`tensor-core` 公開
/// API の一部ではなく `ops.rs` 内でのみ使う（`BackendOps::sum`／`max`
/// は演算ごとに別メソッドのため、この enum 自体は crate 外へ公開しない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceKind {
    Sum,
    Max,
}

/// `reduce::CudaReduce`／`reduce::reduce_axis_layout` が返す `CudaError`
/// を `BackendError` へ写像する（イシュー #1584）。`CudaError::
/// EmptyReduction` は `backend-cpu::reduction::ReduceError::
/// EmptyReduction` と同一の `BackendError::KernelLaunchFailed` 文言
/// （`"empty reduction for op \"{op}\""`）へ、`InvalidReduceShape`
/// （起動前 `i32::MAX` 上限・`checked_mul` オーバーフロー検査の失敗。
/// `ops.rs` 側の shape 検証〈`reduce_out_shape`〉を通過した入力からは
/// 実質到達しない防御的経路）は `ShapeError::ElementCountOverflow` へ、
/// それ以外は既存 [`map_cuda_error`] へ委譲する。
fn map_reduce_error(err: CudaError) -> BackendError {
    match err {
        CudaError::EmptyReduction { op } => {
            BackendError::KernelLaunchFailed(format!("empty reduction for op \"{op}\""))
        }
        CudaError::InvalidReduceShape { .. } => {
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        }
        other => map_cuda_error(other),
    }
}

impl BackendOps for CudaBackendOps {
    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }

    /// `context_cache::cached_device` で得たデバイス上に `ordinal` キーの
    /// プロセス内シングルトン `CudaMemory` を構築・共有する
    /// （`static_cuda_memory`。イシュー #935）。driver 不在等で
    /// `device_handle()` が失敗した場合は `None` を返す（`memory_ops`
    /// のデフォルト契約と同じ fail-safe。`tensor-core::backend_ops`
    /// 参照）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        let device = self.device_handle().ok()?;
        static_cuda_memory(self.ordinal, &device)
            .ok()
            .map(|m| m as &dyn MemoryOps)
    }

    /// SGD の 1 パラメータ分の更新を in-place で実行する（イシュー #935・
    /// `docs/device-resident-update-design.md` §3.2・§5.2）。
    /// `context_cache::cached_sgd`（`ordinal` キーのプロセス内 NVRTC
    /// コンパイル済みカーネルキャッシュ）を経由するため、学習ループの
    /// 2 回目以降のステップは再コンパイルを支払わない。
    fn sgd_step_device(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
    ) -> Result<(), BackendError> {
        if param.device() != Device::Cuda(self.ordinal)
            || grad.device() != Device::Cuda(self.ordinal)
        {
            return Err(BackendError::DeviceMismatch);
        }
        if param.shape() != grad.shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: param.shape().to_vec(),
                rhs: grad.shape().to_vec(),
            }));
        }
        let use_momentum = config.momentum != 0.0;
        if let Some(v) = &velocity {
            // デバイス不一致とテンソル shape 不一致を同一の
            // `ShapeMismatch` に丸めていた（Review 指摘）。
            // `BackendOps::sgd_step_device` の契約（`param`/`grad` と同じ
            // く、デバイス不一致は `DeviceMismatch` を返す）に velocity
            // も揃えるため、判定を分離する。
            if v.device() != Device::Cuda(self.ordinal) {
                return Err(BackendError::DeviceMismatch);
            }
            if v.shape() != param.shape() {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: param.shape().to_vec(),
                    rhs: v.shape().to_vec(),
                }));
            }
        }
        if use_momentum && velocity.is_none() {
            return Err(BackendError::Unsupported(
                "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
            ));
        }

        // イシュー #1013 設計文書 §9 item 7: `param`／`grad`／`velocity` は
        // 学習ループを跨いで生存するデバイス常駐バッファ（`docs/
        // device-resident-update-design.md` §3.2）であり、`invalidate` に
        // よる回復（poison → 新世代）を跨いで使い回されうる唯一の経路
        // （`gemm`／`elementwise` 等はホスト `Tensor` を都度アップロードし
        // 直すため世代を跨がない）。ハンドルを可変借用する前に、この
        // 時点の世代を収集しておく（`downcast_handle_mut` 後は `param`／
        // `velocity` を再度 `&` で読めないため）。
        let resource_generations: Vec<u64> = std::iter::once(param.generation())
            .chain(std::iter::once(grad.generation()))
            .chain(velocity.as_deref().map(|v| v.generation()))
            .collect();

        let sgd = self.with_driver_call(&resource_generations, map_cuda_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_sgd(&device)
        })?;

        let grad_handle = grad
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        // `download`（`memory.rs::CudaMemory::download`）と同じ「空バッファ
        // は `storage: None`」契約のため、numel == 0 はカーネル起動前に
        // 早期 return する（`CudaSgd::run` 側の `numel == 0` early-return
        // では `grad_slice` を取り出す前に `param_slice` を要求してしまう
        // ため、ここで先に判定する）。
        let numel = param.numel();
        if numel == 0 {
            return Ok(());
        }
        let Some(grad_storage) = grad_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "sgd_step_device: grad buffer has numel > 0 but no device allocation".into(),
            ));
        };
        // 配置非依存の読み取り専用引数へ変換する（イシュー #1352。
        // `crate::memory::CudaArg` ドキュメンテーションコメント参照。
        // `launch_builder` の前に名前付きローカルとして宣言する）。
        let grad_arg = grad_storage.as_arg();

        let velocity_arg = match velocity {
            Some(v) => {
                let handle = v
                    .downcast_handle_mut::<CudaBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                let storage = handle.storage.as_mut().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "sgd_step_device: velocity buffer has numel > 0 but no device allocation"
                            .into(),
                    )
                })?;
                Some(storage.as_arg_mut())
            }
            None => None,
        };

        let param_handle = param
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(param_storage) = param_handle.storage.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "sgd_step_device: param buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let param_arg = param_storage.as_arg_mut();

        let kernel_params = crate::sgd::SgdKernelParams {
            lr: config.lr,
            momentum: config.momentum,
            dampening: config.dampening,
            weight_decay: config.weight_decay,
            nesterov: config.nesterov,
            is_first_step: config.is_first_step,
        };
        self.with_driver_call(&resource_generations, map_cuda_error, || {
            sgd.run(param_arg, grad_arg, velocity_arg, &kernel_params)
        })
    }

    /// 学習 step の update 区間（[`Self::sgd_step_device_tracked`]）を
    /// CUDA Graph で capture・再利用できるかを判定する（イシュー #1349・
    /// `docs/backend-cuda-graph-step-capture-design.md` §4.4）。
    ///
    /// opt-in（`crate::graph::step_graph_enabled()`）OFF・現在のデバイスが
    /// capture 可能なストリーム（`CudaDevice::is_capturable_stream`）で
    /// 初期化されていない場合は `Ok(None)` を返し、呼び出し元は現行の
    /// 直接実行経路へフォールバックする——ただし後者（opt-in ON だが
    /// 対象デバイスが legacy stream のまま）は「opt-in を最初のデバイス
    /// 初期化より後に有効化した」設定順序の誤りを示すため、
    /// [`BackendError::Unsupported`] を返し fail-closed に顕在化させる
    /// （design doc §4.7。opt-in OFF 時の `Ok(None)` と区別する）。
    ///
    /// **driver 呼び出し境界（codex-review P0 指摘対応）**: 旧稿は
    /// `Self::device_handle`（poison 検査を経由しない。`device_handle`
    /// doc コメント参照）を driver 呼び出し境界より前に呼んでいたため、
    /// poison・Retiring・別スレッド capture 中のいずれでも拒否される前に
    /// `CudaDevice::new`（キャッシュミス時）や `segment_resources_for`
    /// （`DevicePtr::device_ptr` によるイベント同期を伴いうる）が driver
    /// を操作しうる fail-closed 契約違反があった。本実装は `resources`
    /// の世代収集（host-only・driver 非接触）→
    /// `context_cache::begin_driver_call`（poison／世代検査）→
    /// `observe_cuda_result` で包んだ `Self::device_handle_raw` →
    /// `segment_resources_for` の順に固定する。
    fn captured_segment_key(
        &self,
        resources: &[&fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>],
        config_key: u64,
    ) -> Result<Option<SegmentKey>, BackendError> {
        if !crate::graph::step_graph_enabled() {
            return Ok(None);
        }
        // codex-review P0 指摘対応（PR #1390 再修正）: `sgd_step_device`
        // （通常経路）と同じ `Device::Cuda(self.ordinal)` 一致検査を
        // driver に触れる前（host-only）に行う。この検査を欠くと、GPU 0
        // の `CudaBackendOps` へ GPU 1 のバッファを渡して `SegmentKey` を
        // 導出できてしまい、`run_captured_sgd_step_segment` 側の一致
        // 検査（`key.resources` との比較）は「同じ誤った組」を渡せば
        // 通過してしまうため防御にならない（`run_captured_sgd_step_segment`
        // doc コメント「replay 直前の再検証」参照）。
        if resources
            .iter()
            .any(|b| b.device() != Device::Cuda(self.ordinal))
        {
            return Err(BackendError::DeviceMismatch);
        }
        // host-only（driver 非接触）: 世代は `DeviceBuffer` 自身が保持する
        // ため、driver に触れずに集められる。
        let generations: Vec<u64> = resources.iter().map(|b| b.generation()).collect();
        let token = context_cache::begin_driver_call(self.ordinal, &generations)?;
        let device =
            context_cache::observe_cuda_result(self.ordinal, &token, self.device_handle_raw())
                .map_err(|e| BackendError::CudaUnavailable(e.to_string()))?;
        if !device.is_capturable_stream() {
            return Err(BackendError::Unsupported(
                "captured_segment_key: CUDA Graph step capture is enabled but this device was \
                 initialized with a non-capturable legacy stream. This happens either because \
                 the opt-in was set after the first CUDA device initialization, or because the \
                 `internal-diagnostics` build feature is enabled (that feature forces \
                 StreamKind::Legacy unconditionally so that `CudaDevice::context()`/`stream()` \
                 can be exposed as `pub` without breaking the single-stream invariant that \
                 `disable_event_tracking()` relies on; see `device.rs::CudaDevice::new` cfg \
                 branch comment). Enable the opt-in before the first CUDA device initialization \
                 and build without `internal-diagnostics` to use CUDA Graph step capture."
                    .to_string(),
            ));
        }
        let stream = device.stream();
        let segment_resources = Self::segment_resources_for(resources, stream)?;
        Ok(Some(SegmentKey {
            generation: token.generation(),
            config_key,
            resources: segment_resources,
        }))
    }

    /// [`Self::captured_segment_key`] が返した `key` に対応する SGD 更新
    /// 区間を capture・再生する（イシュー #1349）。実体は
    /// `crate::graph::run_captured_sgd_step_segment`（thread-local
    /// graph キャッシュ・stream capture の手順は同モジュールのドキュ
    /// メント参照）。
    ///
    /// **任意クロージャの撤廃（codex-review P0 指摘対応）**: 旧稿の
    /// `resources: &mut [&mut DeviceBuffer<f32>]` + 任意クロージャ
    /// `body` という public 安全 API は、`body` が `resources` に
    /// 含まれない外部バッファをクロージャキャプチャ経由で触れる抜け道を
    /// 持っていた（トレイト doc コメント参照）。本メソッドは区間が
    /// 触れる全リソース（`param`／`grad`／`velocity`）を直接引数として
    /// 受け取り、クロージャは受け取らない。
    ///
    /// **driver 呼び出し境界（codex-review P0 指摘対応）**: `captured_
    /// segment_key` と同じ理由で、`param`／`grad`／`velocity` の世代
    /// 収集（host-only）→ `begin_driver_call` → `observe_cuda_result` で
    /// 包んだ `device_handle_raw` → アドレス再検証、の順に固定する。
    ///
    /// **replay 直前の再検証（codex-review P0 指摘対応。`docs/backend-
    /// cuda-graph-step-capture-design.md` §4.4 追記）**: `SegmentKey`
    /// 自身はバッファの所有権・借用を保持しない値型のため、`param`／
    /// `grad`／`velocity`（この呼び出しの間ライフタイムが保証される
    /// 借用）から現在のアドレス集合を再計算し、`key.resources` と完全
    /// 一致することを確認してから初めて `crate::graph::
    /// run_captured_sgd_step_segment` へ委譲する。不一致（呼び出し元が
    /// `key` を取得した [`Self::captured_segment_key`] 呼び出しと異なる
    /// バッファを渡した契約違反、または `key` 自体が別ドメインの値）
    /// なら、graph 機構の driver 呼び出し（capture・replay）に進まず
    /// [`BackendError::InvalidArgument`] で拒否する（fail-closed。
    /// 解放済み・別バッファへ再利用済みのアドレスを参照する古い graph を
    /// 安全確認なしに再生する事態を防ぐ）。
    fn run_captured_sgd_step_segment(
        &self,
        key: SegmentKey,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
        token: &fandhe_ai_tensor_core::DispatchFailureCell,
    ) -> Result<SegmentRun, BackendError> {
        // codex-review P0 指摘対応（PR #1390 再修正）: `sgd_step_device`
        // （通常経路）にある `Device::Cuda(self.ordinal)` 一致検査を
        // driver に触れる前（host-only）に行う。この検査を欠くと、同一
        // スレッドで GPU 0 の graph をキャッシュした `key` と、GPU 1 の
        // `param`／`grad`／`velocity`（`Device::Cuda(1)`）を GPU 1 の
        // `CudaBackendOps`（`self.ordinal == 1`）へ渡した際、世代番号が
        // たまたま一致すれば下記の「replay 直前の再検証」（アドレス
        // 一致検査）も通過してしまい、検査・エラー観測が GPU 1 に対して
        // 行われる一方でキャッシュ済み graph は GPU 0 上で再生される
        // ——GPU 0 の capture 排他・poison 機構を迂回する（design doc
        // §4.4）。
        if param.device() != Device::Cuda(self.ordinal)
            || grad.device() != Device::Cuda(self.ordinal)
        {
            return Err(BackendError::DeviceMismatch);
        }
        if let Some(v) = velocity.as_deref()
            && v.device() != Device::Cuda(self.ordinal)
        {
            return Err(BackendError::DeviceMismatch);
        }
        let mut generations = vec![param.generation(), grad.generation()];
        if let Some(v) = velocity.as_deref() {
            generations.push(v.generation());
        }
        let call_token = context_cache::begin_driver_call(self.ordinal, &generations)?;
        let device =
            context_cache::observe_cuda_result(self.ordinal, &call_token, self.device_handle_raw())
                .map_err(|e| BackendError::CudaUnavailable(e.to_string()))?;
        let stream = device.stream();

        let shared_view: Vec<&fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>> = {
            let mut v: Vec<&fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>> = vec![&*param, grad];
            if let Some(vv) = velocity.as_deref() {
                v.push(vv);
            }
            v
        };
        let current_resources = Self::segment_resources_for(&shared_view, stream)?;
        drop(shared_view);
        if current_resources != key.resources {
            return Err(BackendError::InvalidArgument(
                "run_captured_sgd_step_segment: the live `param`/`grad`/`velocity` passed to \
                 this call do not match the resources recorded in `key` (the buffers used to \
                 obtain `key` via `captured_segment_key` must be the exact same borrows, in \
                 the same order, passed here); refusing to replay or capture against a \
                 possibly stale/freed buffer address"
                    .to_string(),
            ));
        }
        // このメソッド自身のトークンはここで役目を終える
        // （poison／世代検査・アドレス再検証は完了した）。以降の実際の
        // capture／replay は `crate::graph::run_captured_sgd_step_segment`
        // が `context_cache::begin_capture_session`（呼び出しスレッドが
        // トークンを 1 つも保持していないことを前提とする in_flight
        // ドレイン契約。同関数 doc コメント参照）から独自に開始する
        // ため、ここでトークンを保持し続けると自己デッドロックしうる。
        drop(call_token);

        crate::graph::run_captured_sgd_step_segment(
            self.ordinal,
            stream,
            key,
            self,
            param,
            grad,
            velocity,
            config,
            token,
        )
    }

    /// GEMM 本体（f32）。既定は FP32 厳密経路（`run_tiled_f32`）で、
    /// 本イシュー導入前と bit-exact に不変（`crate::precision` モジュール
    /// 冒頭コメントの契約）。`crate::precision::gemm_precision()` が
    /// [`crate::precision::CudaGemmPrecision::Tf32`] の場合は WMMA TF32
    /// Tensor Core 単発経路（[`crate::gemm::CudaGemm::run_wmma_tf32`]。
    /// イシュー #1042）、[`crate::precision::CudaGemmPrecision::Tf32x3`]
    /// の場合は 3×TF32 split-single 経路
    /// （[`crate::gemm_mma_tf32x3::CudaMmaTf32x3Gemm::run_tf32x3`]。
    /// イシュー #1355）へそれぞれ分岐する。opt-in 時にモード固有の
    /// カーネルが使用不能（cc<8.0・NVRTC コンパイル失敗・整列制約
    /// 不成立等）な場合は型付きエラーをそのまま `BackendError` へ変換
    /// して伝播し、FP32 への黙示フォールバックはしない（fail-closed。
    /// 明示 opt-in の計測条件を静かに崩さない方針。`crate::precision`
    /// 参照）。
    ///
    /// **注意**: `gemm_bias_act` の `ComposedFallback` および
    /// `gemm_fp32_strict`（学習経路向け入口）からはこのメソッドを
    /// 呼ばない（`gemm_fp32_strict_impl` を使う）。本メソッドは
    /// 精度モードの適用対象である「素の公開 GEMM 入口」専用。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let mode = crate::precision::gemm_precision();
        if mode == crate::precision::CudaGemmPrecision::Fp32Strict {
            return self.gemm_fp32_strict_impl(a, b);
        }

        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        let out = match mode {
            crate::precision::CudaGemmPrecision::Tf32 => {
                let gemm = self.with_driver_call(
                    &[],
                    |e| BackendError::CudaUnavailable(e.to_string()),
                    || {
                        let device = self.device_handle_raw()?;
                        context_cache::cached_gemm(&device)
                    },
                )?;
                let out = self.with_driver_call(
                    &[],
                    |e| BackendError::KernelLaunchFailed(e.to_string()),
                    || gemm.run_wmma_tf32(a_slice, b_slice, m, n, k),
                )?;
                crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));
                out
            }
            crate::precision::CudaGemmPrecision::Tf32x3 => {
                // デバイスハンドル取得（driver 不在等）は他モードと同じ
                // `CudaUnavailable` へ写像し、続くカーネル構築
                // （`cached_mma_tf32x3`。cc<8.0・NVRTC コンパイル失敗等）
                // のみを 3×TF32 固有の fail-closed メッセージ
                // （`KernelLaunchFailed`）へ写像する（`Tf32` 分岐は両者を
                // 区別せず一括で `CudaUnavailable` にしているが、driver
                // そのものが不在の場合に「3xTF32 gemm unavailable」と
                // 誤解を招くメッセージを返さないよう、本分岐では意図的に
                // 分離する）。
                let device = self.with_driver_call(
                    &[],
                    |e| BackendError::CudaUnavailable(e.to_string()),
                    || self.device_handle_raw(),
                )?;
                let gemm = self.with_driver_call(
                    &[],
                    |e| {
                        BackendError::KernelLaunchFailed(format!(
                            "3xTF32 gemm unavailable (fail-closed): {e}"
                        ))
                    },
                    || context_cache::cached_mma_tf32x3(&device),
                )?;
                let out = self.with_driver_call(
                    &[],
                    |e| {
                        BackendError::KernelLaunchFailed(format!(
                            "3xTF32 gemm unavailable (fail-closed): {e}"
                        ))
                    },
                    || gemm.run_tf32x3(a_slice, b_slice, m, n, k),
                )?;
                crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));
                out
            }
            // `Fp32Strict` はこの `match` に到達する前に早期 return 済み
            // （関数冒頭）。ここでの唯一の残り経路は理論上到達しないが、
            // `#[non_exhaustive]` の `CudaGemmPrecision`（`precision.rs`
            // 参照。将来モード追加に備える）は同一クレート内でも将来の
            // 変更で列挙し忘れうるため、ワイルドカードで fail-closed に
            // エラーを返す（未知モードへ黙示フォールバックしない）。
            _ => {
                return Err(BackendError::KernelLaunchFailed(format!(
                    "gemm: unsupported CudaGemmPrecision mode {mode:?} (fail-closed; \
                     no implicit fallback to FP32)"
                )));
            }
        };
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_fp32_strict`] のオーバー
    /// ライド。`gemm`（精度モード分岐を持つ公開経路）を経由せず、
    /// `gemm_fp32_strict_impl`（`crate::precision::gemm_precision()` を
    /// 一切見ない FP32 厳密経路）へ直結する。`autodiff::grad` の VJP
    /// （`matmul_vjp`・`Op::LinearResident` の `d_weight`）が `dyn
    /// BackendOps` 経由で呼ぶ入口で、`Tf32`／`Tf32x3` いずれの opt-in
    /// モードが有効な間も backward を暗黙に精度変更しない契約を保証する
    /// （`crate::precision` モジュール冒頭コメントの「学習経路は本
    /// モジュールのスコープ外」契約。codex-review 指摘・イシュー #1211・
    /// PR #1223。3 モード化はイシュー #1355）。
    fn gemm_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.gemm_fp32_strict_impl(a, b)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_fp32_strict_into`] の
    /// CUDA オーバーライド（イシュー #1559・親 #1557〜#1558）。
    /// `DeviceParamStore::fill_resident_weight_grad`（`Op::LinearResident`
    /// の d_weight）が、結果をホストへ戻さず（D2H）呼び出し元の
    /// `DeviceBuffer<f32>` へ直接書き込む（`gemm_fp32_strict_into_impl`
    /// の doc コメント参照。NT/TN 経路は #1214 で追加済みの GPU 側 smem
    /// 転置カーネルを再利用し、旧経路〈`gemm_fp32_strict` の D2H →
    /// `DeviceParamStore::step` 内 `upload_into` の H2D〉の往復を解消
    /// する）。実体は `gemm_fp32_strict_into_impl`。
    ///
    /// **`resident_grad_capability` との関係**: `backend-metal::ops::
    /// MetalBackendOps::gemm_fp32_strict_into` の doc と同じ理由により、
    /// 本メソッドは NN・TT・分類不能形状・退化形状を含め
    /// `DeviceMismatch`／`InvalidArgument` 以外では失敗しないため、
    /// `resident_grad_capability` は CUDA でも常に `Some(true)`
    /// （または致命的なデバイスエラー）へ確定する（**例外**:
    /// CUDA Graph capture 中〈本メソッドが `#[cfg(test)]` 外の呼び出しで
    /// 現状到達しない区間。イシュー #1349 のスコープは update 区間限定
    /// で backward〈本メソッドの呼び出し元〉は capture 対象外）は
    /// `context_cache::is_capturing_on_current_thread` により
    /// `BackendError::Unsupported` を返す——本メソッドは NT/TN 経路の
    /// ホスト同期〈`stream.synchronize()`〉に加え、NN・TT・分類不能・
    /// 退化形状のフォールバック〈`gemm_fp32_strict` → `run_tiled_f32`
    /// 系〉も内部で D2H readback（ホスト同期）を伴うため、いずれの
    /// 分岐も capture 不能。`gemm_fp32_strict_into_impl` doc コメント
    /// 参照）。
    fn gemm_fp32_strict_into(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
    ) -> Result<(), BackendError> {
        self.gemm_fp32_strict_into_impl(a, b, out, out_offset)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_fp32_strict_into_tracked`]
    /// の CUDA オーバーライド（イシュー #1559）。
    ///
    /// **`token` を無視して既定実装と機能的に同一な明示委譲を置く理由**:
    /// トレイト側 doc は「Metal のみ…オーバーライドし…CUDA を名指し
    /// していない＝CPU と同じ扱いが想定されている」と記す。CUDA は
    /// Metal のような「複数スレッドが共有 `MetalContext`／コマンド
    /// バッファをバッチングし、他スレッドの `synchronize()` が自スレッド
    /// の dispatch 登録前にバッチを drain してしまう」問題を持たない
    /// （`context_cache` の poison／世代検査は `begin_driver_call`／
    /// `observe_cuda_result` により各呼び出しごとに同期的に完結し、かつ
    /// `gemm_fp32_strict_into_impl` の NT/TN 経路は設計判断 A
    /// 〈`gemm.rs::CudaGemm::launch_tiled_f32_nt_into` ドキュメンテー
    /// ションコメント参照〉により関数内で `stream.synchronize()` を行う
    /// ため、失敗はこの呼び出しの戻り値へ即座に伝播する）。本メソッドは
    /// デフォルト実装（`self.gemm_fp32_strict_into(a, b, out,
    /// out_offset)` へ委譲するだけ）と機能的に同一の明示オーバーライド
    /// であり、トレイト側 doc の「CUDA は既定 `Unsupported` のまま」と
    /// いう更新漏れの記述（本イシューで doc 側も更新済み）を実体面でも
    /// 解消する。
    fn gemm_fp32_strict_into_tracked(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
        _token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.gemm_fp32_strict_into(a, b, out, out_offset)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、GEMM epilogue に bias 加算・
    /// activation を融合したカーネル
    /// （[`crate::gemm::CudaGemm::run_tiled_bias_act_f32`]）へ差し替える
    /// （イシュー #599・TASK-12.1f）。`backend-cpu::ops::CpuBackendOps` の
    /// オーバーライドと同型の分岐（`gemm_bias_act_route` 参照）を採り、
    /// `bias` が `None` またはブロードキャストの厳密一致形状 `[n]`
    /// の場合にのみ融合カーネルを使う。それ以外（`[1]`・`[1, n]` 等の
    /// ブロードキャスト可能だが `[n]` ちょうどでない shape）はデフォルト
    /// 実装と同じ 3 段合成（`self.gemm_fp32_strict_impl` → `self.add` →
    /// `self.relu`）へフォールバックする。`self.gemm`（TF32 opt-in 分岐
    /// を持つ公開経路）ではなく `gemm_fp32_strict_impl` を使うのは、
    /// `gemm_bias_act` が `crate::precision` モジュール冒頭コメントの
    /// 契約どおり本イシュー（#1042）のスコープ外のまま常に FP32 で
    /// 動作することを保証するため（codex-review 指摘。PR #1091）。
    /// 両バックエンドは本イシュー時点で `add`／`relu`
    /// が実装済みのため CPU と異なり `Unsupported` を透過しない
    /// （モジュール冒頭コメント参照）。
    ///
    /// フォールバック時も CPU 実装と同じ順序契約（GEMM 本体を実行する前に
    /// `fandhe_ai_tensor_core::broadcast_shape` でブロードキャスト可否のみ先に検証。
    /// REQ-8・OWASP A03）を保つ。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        let bias_shape = bias.map(|t| t.shape());
        match gemm_bias_act_route(bias_shape, n as usize) {
            GemmBiasActRoute::ComposedFallback => {
                if let Some(bias) = bias {
                    // GEMM 本体を実行する前にブロードキャスト可否を検証
                    // する（CPU 実装 `CpuBackendOps::gemm_bias_act` と同じ
                    // 「カーネル本体アクセス前に検証」の順序契約）。
                    fandhe_ai_tensor_core::broadcast_shape(&out_shape, bias.shape())
                        .map_err(BackendError::ShapeMismatch)?;
                }
                let mut out = self.gemm_fp32_strict_impl(a, b)?;
                if let Some(bias) = bias {
                    out = self.add(&out, bias)?;
                }
                out = match act {
                    Activation::None => out,
                    Activation::Relu => self.relu(&out)?,
                    // `Activation` は `#[non_exhaustive]`。CPU 実装と同じ
                    // 方針で未知 variant を黙って恒等関数として扱わず
                    // 明示的に拒否する。
                    _ => {
                        return Err(BackendError::Unsupported(format!(
                            "gemm_bias_act: unsupported activation {act:?} in non-fused fallback path"
                        )));
                    }
                };
                Ok(out)
            }
            GemmBiasActRoute::Fused => {
                let act_relu = match act {
                    Activation::None => false,
                    Activation::Relu => true,
                    _ => {
                        return Err(BackendError::Unsupported(format!(
                            "gemm_bias_act: unsupported activation {act:?} in fused epilogue path"
                        )));
                    }
                };

                let a_owned = a.contiguous();
                let b_owned = b.contiguous();
                let a_slice = a_owned.as_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed("gemm_bias_act: lhs not contiguous".into())
                })?;
                let b_slice = b_owned.as_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed("gemm_bias_act: rhs not contiguous".into())
                })?;

                let bias_owned;
                let bias_slice = match bias {
                    Some(bias) => {
                        bias_owned = bias.contiguous();
                        Some(bias_owned.as_slice().ok_or_else(|| {
                            BackendError::KernelLaunchFailed(
                                "gemm_bias_act: bias not contiguous".into(),
                            )
                        })?)
                    }
                    None => None,
                };

                let gemm = self.with_driver_call(
                    &[],
                    |e| BackendError::CudaUnavailable(e.to_string()),
                    || {
                        let device = self.device_handle_raw()?;
                        context_cache::cached_gemm(&device)
                    },
                )?;
                let out = self.with_driver_call(
                    &[],
                    |e| BackendError::KernelLaunchFailed(e.to_string()),
                    || gemm.run_tiled_bias_act_f32(a_slice, b_slice, bias_slice, act_relu, m, n, k),
                )?;
                Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
            }
        }
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」）。`a`（活性化値）のみをホストから
    /// アップロードし、`w`／`bias` は [`crate::gemm::CudaGemm::
    /// launch_tiled_bias_act_f32_resident`]（`CudaView` 部分ビュー起動。
    /// #1023 のパラメータ横断連結バッファ化後、`w`／`bias` は連結
    /// バッファ内のオフセット範囲としてしか表現できないため、`w`／
    /// `bias` を `DeviceBufferView`（`offset`／`shape` 付き）で受け取り、
    /// `CudaSlice::slice(offset..offset+numel)` で `CudaView` を構築して
    /// カーネルへ渡す）へそのまま渡すことで、これらの download を
    /// 発生させない（`sgd_step_device` と同じ「転送コストを最小化する」
    /// 方針）。
    fn gemm_resident_rhs(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        let a_shape = a.shape();
        if a_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a_shape.len(),
            }));
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let w_shape = w.shape();
        if w_shape.len() != 2 || w_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.to_vec(),
                rhs: w_shape.to_vec(),
            }));
        }
        let n = w_shape[1];
        if let Some(b) = bias {
            if b.device() != Device::Cuda(self.ordinal) {
                return Err(BackendError::DeviceMismatch);
            }
            if b.shape() != [n] {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![n],
                }));
            }
        }
        if k == 0 {
            // `fandhe_ai_autodiff::nn::linear::Linear::new` が
            // `in_features == 0` を構築時に拒否するため、この分岐は
            // `Sequential` 経由の forward では到達しない
            // （`tensor-core::backend_ops::BackendOps::gemm_resident_rhs`
            // doc 参照）。CPU 参照実装のようにホスト側 epilogue のみで
            // 済ませるには resident `bias` を download する必要があり、
            // それは本メソッドが排除する対象の D2H そのものになるため、
            // フォールバックを設けず型付きエラーで拒否する。
            return Err(BackendError::InvalidArgument(
                "gemm_resident_rhs: k == 0 is unreachable via Linear::new (in_features == 0 is \
                 rejected at construction); a host epilogue fallback would require downloading \
                 the resident bias, defeating the zero-D2H contract this method exists for"
                    .to_string(),
            ));
        }
        if m == 0 || n == 0 {
            // 早期 return でも poison 状態は fail-closed に検査する
            // （Cursor Bugbot 指摘・PR #1064 追補: 空入力の早期 return は
            // driver へ触れないため、begin_driver_call の poison 検査を明示的に
            // 経由しないと poison 済み ordinal でも「空 step」相当が黙って
            // 成功してしまう）。世代も通常経路（`resident_generations`。
            // 本関数下部参照）と同じ `w`／`bias` の generation を渡す
            // （codex-review P1 指摘・PR #1064 追補: 空スライスのままだと
            // `invalidate` 後の旧世代 `w`／`bias` ビューがこの分岐だけ
            // `StaleDeviceGeneration` を経由せず成功してしまい、「旧世代は
            // 全て拒否する」という公開エラー契約を経路依存に破る）。
            let empty_shape_generations = [
                Some(w.buffer().generation()),
                bias.map(|b| b.buffer().generation()),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            context_cache::begin_driver_call(self.ordinal, &empty_shape_generations)?;
            return Tensor::new(Vec::new(), &[m, n]).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_full) = w_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_view = w_full.view(w.offset()..w.offset() + w.numel());
        let bias_handle = bias
            .map(|b| {
                b.buffer()
                    .downcast_handle::<CudaBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)
                    .map(|h| (h, b.offset(), b.numel()))
            })
            .transpose()?;
        let bias_view = match &bias_handle {
            Some((h, offset, numel)) => {
                let Some(full) = h.storage.as_ref() else {
                    return Err(BackendError::DeviceAllocationFailed(
                        "gemm_resident_rhs: bias buffer has numel > 0 but no device allocation"
                            .into(),
                    ));
                };
                Some(full.view(*offset..*offset + *numel))
            }
            None => None,
        };

        // `w`（・`bias`）はデバイス常駐のまま渡す唯一の入力（`a` はこの
        // 呼び出し内で毎回アップロードし直すため世代を跨がない。イシュー
        // #1013 設計文書 §9 item 7）。
        let resident_generations = [
            Some(w.buffer().generation()),
            bias.map(|b| b.buffer().generation()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        // `device_handle_raw`（キャッシュミス時の `CudaDevice::new`）自体も
        // poison 検査・観測の対象に含める（codex-review P0 指摘・PR #1064
        // 追補・`ops.rs:147` 相当: `device_handle()` を `with_driver_call`
        // より前に呼ぶと、poison 済み ordinal でも拒否前に driver 初期化が
        // 走ってしまう）。
        let device = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        let mem = CudaMemory::new(&device);
        let a_dev_buf = mem.upload(a)?;
        let a_handle = a_dev_buf
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_storage) = a_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: a buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let a_arg = a_storage.as_arg();

        let mut c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_storage) = c_handle.storage.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let mut c_arg = c_storage.as_arg_mut();

        // キャッシュ構築（`cached_gemm`）自体の driver 呼び出し（NVRTC
        // コンパイル・モジュールロード）も同じ観測対象とする（Cursor
        // Bugbot 指摘・PR #1064 追補: cold-cache 構築中の sticky エラーが
        // 観測なしで消費され fail-open になっていた）。
        let gemm = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || context_cache::cached_gemm(&device),
        )?;
        self.with_driver_call(
            &resident_generations,
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || {
                gemm.launch_tiled_bias_act_f32_resident(
                    &a_arg,
                    &w_view,
                    bias_view.as_ref(),
                    false,
                    &mut c_arg,
                    m as u32,
                    n as u32,
                    k as u32,
                )
            },
        )?;

        mem.download(&c_dev_buf)
    }

    /// `a`（デバイス常駐）・`w`（デバイス常駐）・`bias`（デバイス常駐・
    /// 任意）から `y = act(a @ w + bias)` を、入力・出力いずれも
    /// ホストへ実体化せずに計算する（イシュー #1216・`docs/inference-
    /// forward-fixed-cost-design.md` §3.2「段階 B」）。[`Self::
    /// gemm_resident_rhs_act`]（`a` を毎回 upload・結果を毎回 download）
    /// と同じ融合カーネル（[`crate::gemm::CudaGemm::
    /// launch_tiled_bias_act_f32_resident`]）を使うが、`a` も呼び出し元が
    /// 既にデバイスへ置いた [`fandhe_ai_tensor_core::buffer::DeviceBuffer`] として受け取り、結果も
    /// `DeviceBuffer` のまま返す点が異なる（`gemm_resident_rhs*` 系は
    /// 「`w`／`bias` のみ常駐」・本メソッドは「`a`／`w`／`bias`／戻り値の
    /// 全てが常駐」）。多層 MLP 推論チェーン
    /// （`fandhe_ai_autodiff::optim::device_store` の呼び出し元）が本
    /// メソッドを連鎖させることで、層間の D2H→H2D を発生させず最終
    /// 出力の 1 回の `download` へ同期点を集約できる（trait 定義側の
    /// doc comment・`tensor-core::backend_ops::BackendOps::
    /// linear_forward_device` 参照）。
    ///
    /// **世代検査の対象に `a` を含む**: `gemm_resident_rhs*` は `a` を
    /// 呼び出し内で毎回 upload するため世代検査の対象外だったが（同
    /// ファイル `gemm_resident_rhs` doc 参照）、本メソッドの `a` は
    /// 呼び出しを跨いで生存するデバイス常駐バッファであるため、`w`・
    /// `bias` と同じく `resident_generations` へ含める。
    ///
    /// **出力バッファの確保元**: 呼び出し元へ escape する戻り値のため
    /// `CudaMemory::new(&device)`（呼び出し内で死ぬ一時 tracker）ではなく
    /// `static_cuda_memory`（`memory_ops()` と同一インスタンス）の
    /// `alloc_zeroed` を使う（REQ-14 の単一計測系列。`docs/device-
    /// resident-update-design.md` §3.3d）。CPU 実装が `shared_cpu_memory()`
    /// を使うのと同じ判断。
    fn linear_forward_device(
        &self,
        a: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Cuda(self.ordinal) || w.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        let a_shape = a.shape();
        if a_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a_shape.len(),
            }));
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let w_shape = w.shape();
        if w_shape.len() != 2 || w_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.to_vec(),
                rhs: w_shape.to_vec(),
            }));
        }
        let n = w_shape[1];
        if let Some(b) = bias {
            if b.device() != Device::Cuda(self.ordinal) {
                return Err(BackendError::DeviceMismatch);
            }
            if b.shape() != [n] {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![n],
                }));
            }
        }
        let act_relu = match act {
            Activation::None => false,
            Activation::Relu => true,
            // `Activation` は `#[non_exhaustive]`。CPU 実装・
            // `gemm_bias_act` と同じ方針で未知 variant を黙って恒等関数
            // として扱わず明示的に拒否する。
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "linear_forward_device: unsupported activation {act:?}"
                )));
            }
        };
        if k == 0 {
            // `gemm_resident_rhs` と同じ理由（`Linear::new` が
            // `in_features == 0` を構築時に拒否するため到達不能）で
            // フォールバックを設けず型付きエラーで拒否する。
            return Err(BackendError::InvalidArgument(
                "linear_forward_device: k == 0 is unreachable via Linear::new (in_features == 0 \
                 is rejected at construction); a host epilogue fallback would require \
                 downloading the resident inputs, defeating the zero-D2H contract this method \
                 exists for"
                    .to_string(),
            ));
        }

        // `a` はこのメソッドでは呼び出しを跨いで生存するデバイス常駐
        // バッファのため、`w`・`bias` と同じく世代検査の対象に含める
        // （`gemm_resident_rhs` の doc・本メソッド doc 参照）。
        let resident_generations = [
            Some(a.generation()),
            Some(w.buffer().generation()),
            bias.map(|b| b.buffer().generation()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        if m == 0 || n == 0 {
            // 早期 return でも poison 状態・世代は fail-closed に検査する
            // （`gemm_resident_rhs` の同種分岐と同じ理由。PR #1064 追補）。
            context_cache::begin_driver_call(self.ordinal, &resident_generations)?;
            let device = self.with_driver_call(
                &resident_generations,
                |e| BackendError::CudaUnavailable(e.to_string()),
                || self.device_handle_raw(),
            )?;
            let mem = static_cuda_memory(self.ordinal, &device)?;
            return mem.alloc_zeroed(&[m, n]);
        }

        let a_handle = a
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_full) = a_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "linear_forward_device: a buffer has numel > 0 but no device allocation".into(),
            ));
        };
        if a_full.len() != a.numel() {
            // `DeviceBuffer::new` 経由で構築される限り到達しないはずだが、
            // shape とハンドル実体のずれを本番経路で `unwrap`/`expect` に
            // 頼らず検出する（CPU 実装 `linear_forward_device` と同種の
            // 防御。REQ-8・OWASP A03）。
            return Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch {
                    expected: a.numel(),
                    actual: a_full.len(),
                },
            ));
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_full) = w_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "linear_forward_device: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_view = w_full.view(w.offset()..w.offset() + w.numel());
        let bias_handle = bias
            .map(|b| {
                b.buffer()
                    .downcast_handle::<CudaBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)
                    .map(|h| (h, b.offset(), b.numel()))
            })
            .transpose()?;
        let bias_view = match &bias_handle {
            Some((h, offset, numel)) => {
                let Some(full) = h.storage.as_ref() else {
                    return Err(BackendError::DeviceAllocationFailed(
                        "linear_forward_device: bias buffer has numel > 0 but no device \
                         allocation"
                            .into(),
                    ));
                };
                Some(full.view(*offset..*offset + *numel))
            }
            None => None,
        };

        // `device_handle_raw`（キャッシュミス時の `CudaDevice::new`）自体も
        // poison 検査・観測の対象に含める（`gemm_resident_rhs` と同じ
        // 理由。PR #1064 追補）。
        let device = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        // 出力は呼び出し元へ escape するため `static_cuda_memory`（`memory_ops()`
        // と同一インスタンス）で確保する（本メソッド doc「出力バッファの
        // 確保元」参照。`gemm_resident_rhs` の一時 `CudaMemory::new` とは
        // 異なる）。
        let mem = static_cuda_memory(self.ordinal, &device)?;
        let mut c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_storage) = c_handle.storage.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "linear_forward_device: output buffer has numel > 0 but no device allocation"
                    .into(),
            ));
        };
        let mut c_arg = c_storage.as_arg_mut();
        let a_arg = a_full.as_arg();

        // キャッシュ構築（`cached_gemm`）自体の driver 呼び出しも同じ
        // 観測対象とする（`gemm_resident_rhs` と同じ理由。PR #1064
        // 追補）。
        let gemm = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || context_cache::cached_gemm(&device),
        )?;
        self.with_driver_call(
            &resident_generations,
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || {
                gemm.launch_tiled_bias_act_f32_resident(
                    &a_arg,
                    &w_view,
                    bias_view.as_ref(),
                    act_relu,
                    &mut c_arg,
                    m as u32,
                    n as u32,
                    k as u32,
                )
            },
        )?;

        // `download` しない: 同期点は呼び出し元の `download`（`readback`
        // の `synchronize`）へ集約される（`docs/backend-cuda-async-
        // execution-design.md` の契約どおり。同一ストリーム FIFO により
        // 次層カーネルは前層の出力完了後に実行される）。
        Ok(c_dev_buf)
    }

    /// `a op b`（`op` は [`BinaryElementwiseOp`]）を `a`／`b`／戻り値
    /// いずれも [`DeviceBuffer`] 常駐のまま計算する（イシュー #1584）。
    /// `linear_forward_device` と同じ手順（世代検査 →
    /// `numel == 0` 早期 return → ハンドル取り出し → `static_cuda_memory`
    /// で出力確保 → `elementwise::CudaElementwise::launch_binary_
    /// resident` → `download` せず返す）を踏襲する。`a`／`b` は shape
    /// 完全一致限定（ブロードキャスト非対応）で、不一致は起動前に
    /// `ShapeMismatch` で拒否する。
    fn binary_elementwise_device(
        &self,
        op: BinaryElementwiseOp,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Cuda(self.ordinal) || b.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        if a.shape() != b.shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a.shape().to_vec(),
                rhs: b.shape().to_vec(),
            }));
        }
        let shape = a.shape().to_vec();
        let numel = a.numel();

        let resident_generations = [a.generation(), b.generation()];

        if numel == 0 {
            // 早期 return でも poison 状態・世代は fail-closed に検査する
            // （`linear_forward_device` の `m == 0 || n == 0` 早期
            // return と同じ理由。PR #1064 追補）。
            context_cache::begin_driver_call(self.ordinal, &resident_generations)?;
            let device = self.with_driver_call(
                &resident_generations,
                |e| BackendError::CudaUnavailable(e.to_string()),
                || self.device_handle_raw(),
            )?;
            let mem = static_cuda_memory(self.ordinal, &device)?;
            return mem.alloc_zeroed(&shape);
        }

        let a_handle = a
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_full) = a_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "binary_elementwise_device: a buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let b_handle = b
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_full) = b_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "binary_elementwise_device: b buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let device = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        // 出力は呼び出し元へ escape するため `static_cuda_memory`
        // （`linear_forward_device` の「出力バッファの確保元」doc と
        // 同じ判断。REQ-14 の単一計測系列）。
        let mem = static_cuda_memory(self.ordinal, &device)?;
        let mut out_dev_buf = mem.alloc_zeroed(&shape)?;
        let out_handle = out_dev_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(out_storage) = out_handle.storage.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "binary_elementwise_device: output buffer has numel > 0 but no device \
                 allocation"
                    .into(),
            ));
        };
        let mut out_arg = out_storage.as_arg_mut();
        let a_arg = a_full.as_arg();
        let b_arg = b_full.as_arg();

        let ew = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || context_cache::cached_elementwise(&device),
        )?;
        self.with_driver_call(
            &resident_generations,
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || ew.launch_binary_resident(op, &a_arg, &b_arg, &mut out_arg, numel),
        )?;

        // `download` しない: 同期点は呼び出し元の `download` へ集約
        // （`linear_forward_device` と同じ契約）。
        Ok(out_dev_buf)
    }

    /// [`Self::binary_elementwise_device`] の単項版（イシュー #1584）。
    fn unary_elementwise_device(
        &self,
        op: UnaryElementwiseOp,
        a: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        let shape = a.shape().to_vec();
        let numel = a.numel();

        let resident_generations = [a.generation()];

        if numel == 0 {
            context_cache::begin_driver_call(self.ordinal, &resident_generations)?;
            let device = self.with_driver_call(
                &resident_generations,
                |e| BackendError::CudaUnavailable(e.to_string()),
                || self.device_handle_raw(),
            )?;
            let mem = static_cuda_memory(self.ordinal, &device)?;
            return mem.alloc_zeroed(&shape);
        }

        let a_handle = a
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_full) = a_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "unary_elementwise_device: a buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let device = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        let mem = static_cuda_memory(self.ordinal, &device)?;
        let mut out_dev_buf = mem.alloc_zeroed(&shape)?;
        let out_handle = out_dev_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(out_storage) = out_handle.storage.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "unary_elementwise_device: output buffer has numel > 0 but no device allocation"
                    .into(),
            ));
        };
        let mut out_arg = out_storage.as_arg_mut();
        let a_arg = a_full.as_arg();

        let ew = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || context_cache::cached_elementwise(&device),
        )?;
        self.with_driver_call(
            &resident_generations,
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || ew.launch_unary_resident(op, &a_arg, &mut out_arg, numel),
        )?;

        Ok(out_dev_buf)
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。`Op::LinearResident` の VJP が `d_input^T = w @ g^T`
    /// を計算するために使う。[`Self::gemm_resident_rhs`] と同じく `w` は
    /// download せず `CudaGemm::launch_tiled_f32_resident`
    /// （`CudaView` 部分ビュー起動。バックエンド crate 内部専用 API のため
    /// intra-doc link ではなくコードスパン表記とする）へそのまま渡す。
    fn gemm_resident_lhs(
        &self,
        w: DeviceBufferView<'_>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        let w_shape = w.shape();
        if w_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: w_shape.len(),
            }));
        }
        let (p, q) = (w_shape[0], w_shape[1]);
        let b_shape = b.shape();
        if b_shape.len() != 2 || b_shape[0] != q {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: w_shape.to_vec(),
                rhs: b_shape.to_vec(),
            }));
        }
        let r = b_shape[1];
        if p == 0 || r == 0 {
            // 早期 return でも poison 状態は fail-closed に検査する
            // （Cursor Bugbot 指摘・PR #1064 追補: 空入力の早期 return は
            // driver へ触れないため、begin_driver_call の poison 検査を明示的に
            // 経由しないと poison 済み ordinal でも「空 step」相当が黙って
            // 成功してしまう）。世代も通常経路と同じ `w` の generation を
            // 渡す（codex-review P1 指摘・PR #1064 追補: 空スライスの
            // ままだと `invalidate` 後の旧世代 `w` ビューがこの分岐だけ
            // `StaleDeviceGeneration` を経由せず成功してしまう）。
            context_cache::begin_driver_call(self.ordinal, &[w.buffer().generation()])?;
            return Tensor::new(Vec::new(), &[p, r]).map_err(BackendError::ShapeMismatch);
        }
        if q == 0 {
            // `w` の縮約次元（`out_features`。`Linear::new` は
            // `out_features == 0` を許容する）が 0 の場合、GEMM の数学的
            // 定義どおり結果は全 0（`gemm`／`run_tiled_f32` の `k == 0`
            // 契約と同じ）。GPU 起動を回避してホスト側で直接構築する。
            // 早期 return でも poison 状態は fail-closed に検査する
            // （Cursor Bugbot 指摘・PR #1064 追補: 空入力の早期 return は
            // driver へ触れないため、begin_driver_call の poison 検査を
            // 明示的に経由しないと poison 済み ordinal でも「空 step」
            // 相当が黙って成功してしまう）。世代も通常経路と同じ `w` の
            // generation を渡す（codex-review P1 指摘・PR #1064 追補）。
            context_cache::begin_driver_call(self.ordinal, &[w.buffer().generation()])?;
            return Tensor::from_shape_fill(&[p, r], |_| 0.0).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_full) = w_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_view = w_full.view(w.offset()..w.offset() + w.numel());

        // `w` のみがデバイス常駐入力（`b` はこの呼び出し内で毎回
        // アップロードし直すため世代を跨がない。イシュー #1013 設計文書
        // §9 item 7）。`device_handle_raw`（キャッシュミス時の
        // `CudaDevice::new`）自体も poison 検査・観測の対象に含める
        // （codex-review P0 指摘・PR #1064 追補・`ops.rs:147` 相当）。
        let device = self.with_driver_call(
            &[w.buffer().generation()],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        let mem = CudaMemory::new(&device);

        // キャッシュ構築（`cached_gemm`）自体の driver 呼び出し（NVRTC
        // コンパイル・モジュールロード）も同じ観測対象とする（Cursor
        // Bugbot 指摘・PR #1064 追補: cold-cache 構築中の sticky エラーが
        // 観測なしで消費され fail-open になっていた）。NT 判定
        // （イシュー #1214）に `transpose_smem_f32_available` が要るため
        // `b` のアップロードより前に取得する。
        let gemm = self.with_driver_call(
            &[w.buffer().generation()],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || context_cache::cached_gemm(&device),
        )?;

        // イシュー #1214: `b`（`gᵀ` に相当）が dense な転置 view と判定
        // できる場合、`MemoryOps::upload`（内部で `contiguous()` 実体化
        // する）を経由せず転置元 storage を直接 H2D 転送し
        // `launch_tiled_f32_resident_nt` へ渡す（`gemm_fp32_strict_impl`
        // と同型の判定・fail-soft 方針。転置カーネル使用不能環境は
        // 従来経路へフォールバックする）。`transpose_rows_fit_grid_y_limit(r)`
        // は `launch_tiled_f32_resident_nt` 内の `transpose_to_pooled(
        // bt_dev, r, q)` が構築する grid.y（`r.div_ceil(TRANSPOSE_TILE)`）
        // の事前検査（イシュー #1214 codex-review 指摘。`gemm_fp32_strict_impl`
        // の NT 分岐と同型）。超過する場合は従来経路へフォールバックする。
        if gemm.transpose_smem_f32_available()
            && crate::transpose::transpose_rows_fit_grid_y_limit(r as u32)
            && let Some(bt) = dense_transposed_view(b)
        {
            let out = self.with_driver_call(
                &[w.buffer().generation()],
                |e| BackendError::KernelLaunchFailed(e.to_string()),
                || {
                    // イシュー #1585: `gemm.upload_h2d_new` を経由し
                    // H2D pinned staging（opt-in・既定 OFF）へ参加させる
                    // （`gemm.rs::CudaGemm::upload_h2d_new` ドキュメンテー
                    // ションコメント参照。フラグ OFF 時は `device.stream()
                    // .clone_htod(bt)` と経路・出力とも bit 同一）。
                    let bt_dev = gemm.upload_h2d_new(bt)?;
                    let mut c_dev = device.stream().alloc_zeros::<f32>(p * r)?;
                    // `launch_tiled_f32_resident_nt` が返す転置中間バッファ
                    // （`PooledCudaHandle`）は `readback` 完了まで保持する
                    // 契約（同メソッドのドキュメンテーションコメント
                    // 参照。advisor 指摘: 早期 drop はプール返却・実解放が
                    // GEMM カーネル完了前に走りうる）。`_b_std` として
                    // 束縛し、`readback` 呼び出しが終わるまで生存させる。
                    let _b_std = gemm.launch_tiled_f32_resident_nt(
                        &w_view,
                        w.offset(),
                        &bt_dev,
                        &mut c_dev,
                        p as u32,
                        r as u32,
                        q as u32,
                    )?;
                    crate::memory::readback(device.stream(), &c_dev)
                },
            )?;
            return Tensor::new(out, &[p, r]).map_err(BackendError::ShapeMismatch);
        }

        // フォールバック（TT 相当・判定不能形状・転置カーネル使用不能
        // 環境）: 従来どおり `MemoryOps::upload` を経由する。`b` が
        // 非 contiguous なら `upload` 内部の `contiguous()` が再パック
        // コピーを行うため計上する。
        if !b.is_contiguous() {
            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
        }
        let b_dev_buf = mem.upload(b)?;
        let b_handle = b_dev_buf
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_storage) = b_handle.storage.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: b buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let b_arg = b_storage.as_arg();

        let mut c_dev_buf = mem.alloc_zeroed(&[p, r])?;
        let c_handle = c_dev_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_storage) = c_handle.storage.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let mut c_arg = c_storage.as_arg_mut();

        self.with_driver_call(
            &[w.buffer().generation()],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || {
                gemm.launch_tiled_f32_resident(
                    &w_view,
                    w.offset(),
                    &b_arg,
                    &mut c_arg,
                    p as u32,
                    r as u32,
                    q as u32,
                )
            },
        )?;

        mem.download(&c_dev_buf)
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, a_s, b_s| ew.run_add_f32(a_s, b_s))
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, a_s, b_s| ew.run_mul_f32(a_s, b_s))
    }

    /// `BackendOps::where_cond` の CUDA 実装（イシュー #1637）。
    /// `elementwise::CudaElementwise::run_where_f32` へ委譲する。
    fn where_cond(
        &self,
        cond: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_ternary(cond, a, b, |ew, cond_s, a_s, b_s| {
            ew.run_where_f32(cond_s, a_s, b_s)
        })
    }

    /// `BackendOps::masked_fill` の CUDA 実装（イシュー #1637）。
    fn masked_fill(
        &self,
        x: &Tensor<f32>,
        mask: &Tensor<f32>,
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary_scalar(x, mask, value, |ew, x_s, mask_s, v| {
            ew.run_masked_fill_f32(x_s, mask_s, v)
        })
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, a_s| ew.run_relu_f32(a_s))
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, a_s| ew.run_exp_f32(a_s))
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, a_s| ew.run_tanh_f32(a_s))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::run_fused`] のデフォルト実装
    /// （`Unsupported` fail-safe）を、canonical RMSNorm 融合プラン
    /// （`x * rsqrt(sum(x^2))`。mean 化・eps・weight を含まない厳密形状）
    /// 検出時に [`crate::rmsnorm::CudaRmsNorm`]（イシュー #592）へ、
    /// canonical softmax 融合プラン（`exp(x - max(x)) / sum(exp(x -
    /// max(x)))`。最終軸または全軸縮約の厳密形状）検出時に
    /// [`crate::softmax::CudaSoftmax`]（イシュー #594）へルーティング
    /// する。
    ///
    /// プラン一致判定は `match_rmsnorm_plan`／`match_softmax_plan`
    /// （いずれも純関数。プランの op 列・leaf 数・`row_fusion()` の形状を
    /// 厳密照合する）に委ねる。RMSNorm 判定を先に試し、一致しなければ
    /// softmax 判定を試す（op 列長〈6 vs 8〉が異なるため両方に一致する
    /// プランは存在しない）。どちらにも一致しないプラン
    /// （elementwise-only・中間軸 softmax 等）は本オーバーライドの対象外
    /// としてデフォルト実装（`Unsupported`）へ委ね、呼び出し元
    /// （`fandhe_ai_autodiff::Tape` の実体化経路）の per-op フォールバックへ倒す
    /// （`backend-cpu::fused_elementwise::run_fused_elementwise` の
    /// allowlist 拒否方針と同じ fail-closed。`.claude/rules/security.md`
    /// A08「判定の迂回経路を作らない」）。
    ///
    /// RMSNorm 一致時: プランの意味論 `x * rsqrt(sum(x^2))` に厳密一致
    /// させるため `crate::rmsnorm::CudaRmsNorm::run_rmsnorm_f32_raw`
    /// （`inv_n` を明示できる内部エントリ）を `inv_n = 1.0`・`eps = 0.0`・
    /// `w = None`（`has_weight = 0`）で直接呼ぶ（`mean` 化・`eps` 加算・
    /// `weight` 乗算を勝手に補わない。標準 RMSNorm 用の公開 API
    /// [`crate::rmsnorm::CudaRmsNorm::run_rmsnorm_f32`] は `inv_n =
    /// 1/hidden` を内部導出してしまうため canonical プランには使えない。
    /// `rmsnorm.rs` ドキュメンテーションコメント参照）。`rows = 1` は
    /// canonical プランが `axis: None`（全軸縮約）のみを受理する
    /// （`match_rmsnorm_plan` 参照）ため、行方向融合ではなく単一行として
    /// 扱う。
    ///
    /// softmax 一致時: プランの意味論 `exp(x - max(x)) / sum(...)` に
    /// 厳密一致させるため `crate::softmax::CudaSoftmax::run_softmax_f32_raw`
    /// を `scale = log2(e)` で直接呼ぶ（プランの `Exp` は自然指数だが
    /// カーネルは `exp2(x*log2(e))` を計算する恒等式を用いる。数値的な
    /// 一致判定は per-op 経路と丸めが異なるため REQ-2 複合判定に依る。
    /// `softmax.rs` モジュール冒頭コメント「意味論注記」参照）。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        if let Some(hidden) = match_rmsnorm_plan(plan) {
            return self.run_fused_rmsnorm(plan, leaves, hidden);
        }
        if let Some((rows, cols)) = match_softmax_plan(plan) {
            return self.run_fused_softmax(plan, leaves, rows, cols);
        }
        Err(BackendError::Unsupported(
            "CudaBackendOps::run_fused: プランが canonical RMSNorm 形状（x * \
             rsqrt(sum(x^2))）・canonical softmax 形状（exp(x-max(x))/sum(...)）の \
             いずれにも一致しないため融合カーネルへルーティングできない \
             （#592／#594 スコープ。呼び出し元の per-op フォールバックに委ねる）"
                .into(),
        ))
    }

    /// 全軸・単一軸 `sum`（`reduce::CudaReduce::run_sum_all_f32`／
    /// `run_sum_axis_f32` への委譲。イシュー #1584・親イシュー #1571）。
    /// `reduce_out_shape` で `dim` の範囲検査・出力 shape を導出した後、
    /// `a.contiguous()` で密なバッファへ実体化してから渡す（`elementwise_
    /// binary`／`elementwise_unary` と同じ「非 contiguous はここで解消
    /// する」方針）。空縮約の意味論は `reduce.rs` 冒頭コメント参照
    /// （`backend-cpu::reduction::sum` と同一）。
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.reduce_dispatch(a, dim, ReduceKind::Sum)
    }

    /// [`Self::sum`] と同じ委譲構造（`run_max_all_f32`／
    /// `run_max_axis_f32`）。`fmaxf` による厳密選択（丸めなし。`reduce.rs`
    /// 冒頭コメント「max は厳密選択」参照）で、空縮約は
    /// `BackendError::KernelLaunchFailed("empty reduction for op \"max\"")`
    /// （`backend-cpu::reduction::max` と同一文言。`map_reduce_error`
    /// 参照）を返す。
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.reduce_dispatch(a, dim, ReduceKind::Max)
    }

    /// 線形代数（イシュー #1621・`docs/autodiff-linalg-design.md`）は
    /// GPU カーネル未実装（設計文書「スコープ外」節: GPU カーネル実装は
    /// 本イシューのスコープ外・別イシューへ引き継ぐ）。既定
    /// `Unsupported` を明示オーバーライドし、`device_handle()`（driver
    /// 初期化）を経由しない——`Self::sum`／`Self::max` と同じ「未実装を
    /// 明示する」方針（実機の有無に関わらず常に `Unsupported`）。
    fn linalg_inv(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::linalg_inv: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_solve(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::linalg_solve: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_det(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::linalg_det: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_cholesky(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::linalg_cholesky: 線形代数カーネル未実装（イシュー #1621 \
             スコープ外）"
                .into(),
        ))
    }

    fn linalg_qr(&self, _a: &Tensor<f32>) -> Result<QrFactors, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::linalg_qr: 線形代数カーネル未実装（イシュー #1621 スコープ外）".into(),
        ))
    }

    fn linalg_svd(&self, _a: &Tensor<f32>) -> Result<SvdFactors, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::linalg_svd: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_matrix_norm(
        &self,
        _a: &Tensor<f32>,
        _ord: MatrixNormOrd,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::linalg_matrix_norm: 線形代数カーネル未実装（イシュー #1621 \
             スコープ外）"
                .into(),
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss`] の CUDA 実装
    /// （イシュー #1045）。`Self::sum`／`Self::max`（汎用 reduction）とは
    /// 独立した専用融合カーネル（`crate::mse::CudaMse`）へのディスパッチ
    /// であり、`Self::sum` の未実装状態とは無関係にここで実装する
    /// （`backend_ops.rs::BackendOps::mse_loss` doc の設計判断参照:
    /// `Op::MseLoss` は解析形の専用ノードであり融合 IR〈`run_fused`〉を
    /// 経由しない）。
    ///
    /// `reduction` に応じた `factor`（`Mean` は `1.0/n`、`Sum` は `1.0`）は
    /// ここで計算してカーネルへ渡す（`CudaMse::run_mse_loss_f32` は
    /// reduction 種別を知らない。`kernels_mse.rs` 冒頭コメント参照）。
    /// 未知 `MseReduction` variant は `backend-cpu::ops::CpuBackendOps::
    /// mse_loss` と同じく `Unsupported` として拒否する。
    fn mse_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss: target not contiguous".into())
        })?;
        let numel = pred_slice.len();
        let factor = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    1.0
                } else {
                    1.0 / numel as f32
                }
            }
            MseReduction::Sum => 1.0,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "mse_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };

        let mse = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_mse(&device)
            },
        )?;
        let value = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || mse.run_mse_loss_f32(pred_slice, target_slice, factor),
        )?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss_backward`] の CUDA
    /// 実装（イシュー #1045）。`dTarget = −dPred` は呼び出し元
    /// （`fandhe_ai_autodiff::grad::vjp`）がホスト側で符号反転して得る
    /// 契約のため、本メソッドは `dPred` のみを計算して返す
    /// （`backend_ops.rs::BackendOps::mse_loss_backward` doc 参照）。
    fn mse_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss_backward: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss_backward: target not contiguous".into())
        })?;

        let mse = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_mse(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || mse.run_mse_backward_f32(pred_slice, target_slice, scale),
        )?;
        Tensor::new(out, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::rmsnorm`] の CUDA 実装
    /// （イシュー #1596）。既存の [`Self::run_fused`] 経由（`match_
    /// rmsnorm_plan` の canonical プラン一致限定・`mean` 化なし・`eps`
    /// なし・`weight` なし）とは別の独立エントリで、[`row_norm_layout`]
    /// で `(rows, hidden)` を導出してから
    /// `crate::rmsnorm::CudaRmsNorm::run_rmsnorm_f32`（`mean` 化・`eps`・
    /// 任意 `weight` を含む標準 RMSNorm）を直接呼ぶ
    /// （`run_fused_rmsnorm` と同じ `context_cache::cached_rmsnorm`
    /// キャッシュを再利用する）。
    fn rmsnorm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("rmsnorm: input not contiguous".into())
        })?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice = match &w_owned {
            Some(w) => Some(w.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("rmsnorm: weight not contiguous".into())
            })?),
            None => None,
        };

        let rmsnorm = self.with_driver_call(&[], map_fused_kernel_init_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_rmsnorm(&device)
        })?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rmsnorm.run_rmsnorm_f32(x_slice, w_slice, eps, rows, hidden),
        )?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::layer_norm`] の CUDA 実装
    /// （イシュー #1596）。[`Self::rmsnorm`] と同じ `row_norm_layout`
    /// 導出だが、`run_fused`（canonical 融合プラン一致経路）への
    /// LayerNorm 一致経路は追加しない——LayerNorm は本エントリ経由でのみ
    /// 到達する（`docs/norm-ops-design.md`）。新設カーネル
    /// `crate::layer_norm::CudaLayerNorm::run_layer_norm_f32`・
    /// `context_cache::cached_layer_norm` を使う。
    fn layer_norm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("layer_norm: input not contiguous".into())
        })?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice = match &w_owned {
            Some(w) => Some(w.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("layer_norm: weight not contiguous".into())
            })?),
            None => None,
        };
        let b_owned = bias.map(|b| b.contiguous());
        let b_slice = match &b_owned {
            Some(b) => Some(b.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("layer_norm: bias not contiguous".into())
            })?),
            None => None,
        };

        let layer_norm = self.with_driver_call(&[], map_fused_kernel_init_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_layer_norm(&device)
        })?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || layer_norm.run_layer_norm_f32(x_slice, w_slice, b_slice, eps, rows, hidden),
        )?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_pointwise`] の CUDA
    /// 実装（イシュー #1647）。`hidden` は `c_prev` の列数から導出する。
    fn lstm_pointwise(
        &self,
        pre: &Tensor<f32>,
        c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        require_rank2_cell(c_prev.shape())?;
        let hidden = c_prev.shape()[1];
        let b_dim = c_prev.shape()[0];
        // `pre` の rank・shape も検証する（平坦化後の要素数一致だけ
        // では異形状を誤って受理しうる。イシュー #1647 codex-review
        // P2 指摘）。
        let gate_width = checked_gate_width(4, hidden)?;
        require_same_shape(pre.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        let pre_owned = pre.contiguous();
        let c_prev_owned = c_prev.contiguous();
        let pre_slice = pre_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_pointwise: pre not contiguous".into())
        })?;
        let c_prev_slice = c_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_pointwise: c_prev not contiguous".into())
        })?;

        let rnn = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_rnn_cell(&device)
            },
        )?;
        let (gates, c, h) = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rnn.run_lstm_pointwise_f32(pre_slice, c_prev_slice, hidden),
        )?;
        Ok(LstmPointwiseOutput {
            gates: Tensor::new(gates, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            c: Tensor::new(c, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
            h: Tensor::new(h, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_hidden_backward`] の
    /// CUDA 実装（イシュー #1647）。
    fn lstm_hidden_backward(
        &self,
        c: &Tensor<f32>,
        gate_o: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        require_rank2_cell(c.shape())?;
        require_same_shape(gate_o.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        let shape = c.shape().to_vec();
        let c_owned = c.contiguous();
        let gate_o_owned = gate_o.contiguous();
        let dh_owned = dh.contiguous();
        let c_slice = c_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_hidden_backward: c not contiguous".into())
        })?;
        let gate_o_slice = gate_o_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_hidden_backward: gate_o not contiguous".into())
        })?;
        let dh_slice = dh_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_hidden_backward: dh not contiguous".into())
        })?;

        let rnn = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_rnn_cell(&device)
            },
        )?;
        let (d_pre_o, dc) = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rnn.run_lstm_hidden_backward_f32(c_slice, gate_o_slice, dh_slice),
        )?;
        Ok((
            Tensor::new(d_pre_o, &shape).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dc, &shape).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_cell_backward`] の CUDA
    /// 実装（イシュー #1647）。`hidden` は `c_prev` の列数から導出する。
    fn lstm_cell_backward(
        &self,
        gates_ifg: &Tensor<f32>,
        c_prev: &Tensor<f32>,
        dc: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        require_rank2_cell(c_prev.shape())?;
        let hidden = c_prev.shape()[1];
        let b_dim = c_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_ifg.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dc.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        let gates_owned = gates_ifg.contiguous();
        let c_prev_owned = c_prev.contiguous();
        let dc_owned = dc.contiguous();
        let gates_slice = gates_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_cell_backward: gates_ifg not contiguous".into())
        })?;
        let c_prev_slice = c_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_cell_backward: c_prev not contiguous".into())
        })?;
        let dc_slice = dc_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_cell_backward: dc not contiguous".into())
        })?;

        let rnn = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_rnn_cell(&device)
            },
        )?;
        let (d_pre_ifg, dc_prev) = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rnn.run_lstm_cell_backward_f32(gates_slice, c_prev_slice, dc_slice, hidden),
        )?;
        Ok((
            Tensor::new(d_pre_ifg, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dc_prev, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gru_pointwise`] の CUDA 実装
    /// （イシュー #1647）。`hidden` は `h_prev` の列数から導出する。
    fn gru_pointwise(
        &self,
        pre_i: &Tensor<f32>,
        pre_h: &Tensor<f32>,
        h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        require_rank2_cell(h_prev.shape())?;
        let hidden = h_prev.shape()[1];
        let b_dim = h_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(pre_i.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(pre_h.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        let pre_i_owned = pre_i.contiguous();
        let pre_h_owned = pre_h.contiguous();
        let h_prev_owned = h_prev.contiguous();
        let pre_i_slice = pre_i_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_pointwise: pre_i not contiguous".into())
        })?;
        let pre_h_slice = pre_h_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_pointwise: pre_h not contiguous".into())
        })?;
        let h_prev_slice = h_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_pointwise: h_prev not contiguous".into())
        })?;

        let rnn = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_rnn_cell(&device)
            },
        )?;
        let (gates, q, h) = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rnn.run_gru_pointwise_f32(pre_i_slice, pre_h_slice, h_prev_slice, hidden),
        )?;
        Ok(GruPointwiseOutput {
            gates: Tensor::new(gates, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            q: Tensor::new(q, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
            h: Tensor::new(h, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gru_backward`] の CUDA 実装
    /// （イシュー #1647）。`hidden` は `h_prev` の列数から導出する。
    fn gru_backward(
        &self,
        gates_rzn: &Tensor<f32>,
        q: &Tensor<f32>,
        h_prev: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<GruBackwardOutput, BackendError> {
        require_rank2_cell(h_prev.shape())?;
        let hidden = h_prev.shape()[1];
        let b_dim = h_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_rzn.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(q.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        let gates_owned = gates_rzn.contiguous();
        let q_owned = q.contiguous();
        let h_prev_owned = h_prev.contiguous();
        let dh_owned = dh.contiguous();
        let gates_slice = gates_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: gates_rzn not contiguous".into())
        })?;
        let q_slice = q_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: q not contiguous".into())
        })?;
        let h_prev_slice = h_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: h_prev not contiguous".into())
        })?;
        let dh_slice = dh_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: dh not contiguous".into())
        })?;

        let rnn = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_rnn_cell(&device)
            },
        )?;
        let (d_pre_i, d_pre_h, dh_prev_direct) = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rnn.run_gru_backward_f32(gates_slice, q_slice, h_prev_slice, dh_slice, hidden),
        )?;
        Ok((
            Tensor::new(d_pre_i, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(d_pre_h, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dh_prev_direct, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::softmax`] の CUDA 実装
    /// （イシュー #1594）。[`row_softmax_layout`] が非最終軸を `Ok(None)`
    /// として区別する契約に従い、その場合はデフォルトの
    /// `Unsupported`（`Var::softmax` がホスト参照実装へフォールバック
    /// する合図）と同じ挙動を返す。最終軸の場合は `run_fused_softmax`
    /// （`run_fused` の softmax 一致経路）と同じ
    /// `context_cache::cached_softmax` キャッシュ・`CudaSoftmax::
    /// run_softmax_f32` を直接呼ぶ（融合プランを経由しない独立入口）。
    fn softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        let Some((rows, cols)) =
            row_softmax_layout(x.shape(), dim).map_err(BackendError::ShapeMismatch)?
        else {
            return Err(BackendError::Unsupported(
                "softmax: CUDA 行カーネルは最終軸限定（非最終軸はホスト参照実装へ委ねる）".into(),
            ));
        };

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("softmax: input not contiguous".into())
        })?;

        let softmax = self.with_driver_call(&[], map_fused_kernel_init_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_softmax(&device)
        })?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || softmax.run_softmax_f32(x_slice, rows, cols),
        )?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::release_cached_device_memory`] の CUDA 実装
    /// （イシュー #1020・REQ-14）。`gemm.rs`／`elementwise.rs`／`softmax.rs`
    /// が `context_cache::cached_allocator` 経由で共有する
    /// `(ordinal, 既定 stream)` 単位のサイズクラス別プールを即座に解放する。
    ///
    /// エラー文字列にはフェーズ識別子（`crate::pool::ReleasePhase`。
    /// "pre-free sync"／"handle release"／"post-free sync"／"driver trim"）
    /// を含める（新しい `BackendError` variant は追加しない設計判断。
    /// `docs/backend-cuda-pool-allocator-decision.md` 参照）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        // イシュー #1349: プール解放は同期（`ctx.synchronize()`）を伴う
        // 明示的な同期点であり、CUDA Graph capture 中に呼ぶと capture
        // の前提（同一ストリーム上の driver 呼び出しのみで完結する
        // こと）を破る。driver に触れる前に拒否する
        // （`context_cache::begin_sync_point_call` と同じ判定）。
        if context_cache::is_capturing_on_current_thread(self.ordinal) {
            return Err(BackendError::Unsupported(
                "cuda graph capture: release_cached_device_memory is a host synchronization \
                 point and cannot be captured"
                    .to_string(),
            ));
        }
        // `device_handle_raw`（キャッシュミス時の `CudaDevice::new`）自体も
        // poison 検査・観測の対象に含める（codex-review P0 指摘・PR #1064
        // 追補・`ops.rs:147` 相当）。
        let device = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;

        // `cached_allocator`／`release_cached`（pre/post-free の
        // `stream.synchronize()`・driver トリム）自体も poison 検査・観測
        // の対象に含める（codex-review P0 指摘・PR #1064 追補: これらは
        // 先行する非同期カーネルの sticky エラーを最初に観測しうる同期点
        // であり、`ReleaseCacheError` へ変換されるだけで poison 化されない
        // と、次の演算が Active のまま通ってしまう fail-open 経路になる）。
        // `release_cached` の戻り値型 `ReleaseCacheError` は `CudaError`
        // そのものではないため `with_driver_call` の一律インターフェース
        // には載せず、`begin_driver_call`／`observe_cuda_error_ref` を
        // 直接呼んで分類・poison 化のみを行い、`ReleaseCacheError` が
        // 運ぶフェーズ識別子はそのまま `BackendError` のメッセージへ残す
        // （`pool.rs::ReleaseCacheError` ドキュメンテーションコメント
        // 参照）。
        let token = context_cache::begin_driver_call(self.ordinal, &[])?;
        let allocator = match context_cache::observe_cuda_result(
            self.ordinal,
            &token,
            context_cache::cached_allocator(&device),
        ) {
            Ok(allocator) => allocator,
            Err(e) => return Err(BackendError::CudaUnavailable(e.to_string())),
        };
        match allocator.release_cached() {
            Ok(_freed_bytes) => Ok(()),
            Err(e) => {
                context_cache::observe_cuda_error_ref(self.ordinal, &token, &e.detail);
                Err(BackendError::DeviceAllocationFailed(format!(
                    "release_cached_device_memory: {e}"
                )))
            }
        }
    }

    /// [`fandhe_ai_tensor_core::BackendOps::device_memory_pool_stats`] の CUDA 実装。
    /// driver 不在等で `device_handle()` が失敗した場合は `None` を返す
    /// （`memory_ops` と同じ fail-safe 契約。統計取得の失敗で呼び出し元の
    /// エラー処理を複雑化させない）。
    fn device_memory_pool_stats(&self) -> Option<fandhe_ai_tensor_core::PoolStats> {
        let device = self.device_handle().ok()?;
        let allocator = context_cache::cached_allocator(&device).ok()?;
        Some(allocator.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`map_fused_kernel_init_error`]: `DriverUnavailable`／
    /// `NvrtcUnavailable` は環境不在として `BackendError::CudaUnavailable`
    /// へ変換される（env-adaptive スモークテストの早期 return 判定と
    /// 揃う）。RMSNorm／softmax 両経路で共用する（イシュー #594）。
    #[test]
    fn map_fused_kernel_init_error_treats_known_unavailable_variants_as_cuda_unavailable() {
        assert!(matches!(
            map_fused_kernel_init_error(CudaError::DriverUnavailable {
                detail: "no libcuda".into()
            }),
            BackendError::CudaUnavailable(msg) if msg.contains("no libcuda")
        ));
        assert!(matches!(
            map_fused_kernel_init_error(CudaError::NvrtcUnavailable {
                detail: "no libnvrtc".into()
            }),
            BackendError::CudaUnavailable(msg) if msg.contains("no libnvrtc")
        ));
    }

    /// [`map_fused_kernel_init_error`]: 環境不在以外の失敗（NVRTC
    /// コンパイルエラー・デバイス属性負値検出等）は
    /// `BackendError::KernelLaunchFailed` として実装回帰を検出できる状態を
    /// 保つ（`CudaUnavailable` に丸めて env-adaptive テストの早期 return
    /// に握りつぶされるのを防ぐ。codex-review 指摘・PR #706 レビュー）。
    #[test]
    fn map_fused_kernel_init_error_treats_other_variants_as_kernel_launch_failed() {
        let err = map_fused_kernel_init_error(CudaError::InvalidKernelDescriptor {
            detail: "negative SM count".into(),
        });
        assert!(matches!(
            err,
            BackendError::KernelLaunchFailed(msg) if msg.contains("negative SM count")
        ));
    }

    #[test]
    fn gemm_bias_act_route_selects_fused_when_bias_is_none() {
        assert_eq!(gemm_bias_act_route(None, 8), GemmBiasActRoute::Fused);
    }

    #[test]
    fn gemm_bias_act_route_selects_fused_when_bias_shape_matches_n_exactly() {
        assert_eq!(gemm_bias_act_route(Some(&[8]), 8), GemmBiasActRoute::Fused);
    }

    #[test]
    fn gemm_bias_act_route_falls_back_when_bias_shape_is_broadcastable_but_not_n() {
        // `[1]` は `[n]` へブロードキャスト可能だが厳密一致ではないため
        // フォールバック（CPU 実装と同じ分岐条件）。
        assert_eq!(
            gemm_bias_act_route(Some(&[1]), 8),
            GemmBiasActRoute::ComposedFallback
        );
        // `[1, n]` も同様（2 次元形状は `[n]` と厳密一致しない）。
        assert_eq!(
            gemm_bias_act_route(Some(&[1, 8]), 8),
            GemmBiasActRoute::ComposedFallback
        );
    }

    #[test]
    fn gemm_bias_act_route_falls_back_when_bias_len_mismatches_n() {
        assert_eq!(
            gemm_bias_act_route(Some(&[4]), 8),
            GemmBiasActRoute::ComposedFallback
        );
    }

    /// 環境適応（CUDA 非搭載環境でも実行可能。実機なら本体まで検証）:
    /// `gemm_bias_act`（`bias.shape() == [n]`）が実際に融合カーネル
    /// （`gemm::CudaGemm::run_tiled_bias_act_f32`）へ到達し、
    /// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト
    /// 実装（非融合 3 段合成）を経由していないことを、
    /// [`crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`] の増加で検証する
    /// （実装計画 3.3 節「フォールバックを経由しないことのテスト機構」）。
    /// CUDA 非搭載環境では `BackendError::CudaUnavailable` を確認して
    /// 早期 return する（`tests/backend_ops_real_device.rs` と同じ
    /// 分岐パターン）。
    ///
    /// カウンタはスレッドローカル（`gemm.rs::BIAS_ACT_FUSED_LAUNCH_COUNT`
    /// のドキュメンテーションコメント参照。codex-review 指摘・PR #688）
    /// のため、`cargo test` の既定並列実行下で他スレッドの別テストが
    /// 同じ融合カーネルを起動しても `before`/`after` の差分には混入しない
    /// （直列化・プロセス全体 Mutex は不要）。
    #[test]
    fn gemm_bias_act_fused_path_increments_launch_counter_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");
        let bias = Tensor::new(vec![1.0, 1.0], &[2]).expect("valid tensor");

        let before = crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());
        match cuda.gemm_bias_act(&a, &b, Some(&bias), Activation::Relu) {
            Ok(_) => {
                let after = crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());
                assert!(
                    after > before,
                    "融合カーネルの起動カウンタが増加していない（デフォルト非融合合成へ \
                     フォールバックした疑い）: before={before}, after={after}"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for gemm_bias_act: {other}"),
        }
    }

    /// フラグはプロセスグローバル（`crate::precision`）のため、他の
    /// テストとの競合を避けて直列化・原状復帰する RAII ガード
    /// （`precision.rs::tests::FlagGuard` と同型。イシュー #1042）。
    ///
    /// **enum 保存/復元（イシュー #1355）**: `original` を `bool` ではなく
    /// [`crate::precision::CudaGemmPrecision`] で保存する。`bool` のまま
    /// では `Tf32x3` 状態が `set_tf32_gemm_enabled` の互換ラッパー経由で
    /// `Fp32Strict`／`Tf32` の 2 値へ lossy に丸められて復元され、テスト
    /// 間で `Tf32x3` 状態が意図せず消える干渉を招く（実装計画 §4 ステップ
    /// 1）。
    struct Tf32FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: crate::precision::CudaGemmPrecision,
    }

    impl Tf32FlagGuard {
        fn acquire() -> Self {
            // `precision.rs::tests::FlagGuard` と単一ロックを共有する
            // （codex-review P2・Cursor Bugbot Medium 指摘。別々の
            // `static LOCK` を持つと直列化が効かず `GEMM_PRECISION`
            // を巡るレースが起こりうる。PR #1091）。
            let lock = crate::precision::test_support::tf32_flag_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = crate::precision::gemm_precision();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for Tf32FlagGuard {
        fn drop(&mut self) {
            crate::precision::set_gemm_precision(self.original);
        }
    }

    /// 環境適応（CUDA 非搭載環境でも実行可能。実機なら本体まで検証）:
    /// `crate::precision::tf32_gemm_enabled()` が既定 `false`（OFF）の
    /// 場合、`gemm` が TF32 経路（[`crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT`]）
    /// へ一切到達しないことを検証する（イシュー #1042 実装計画 §2.1
    /// 「既定は OFF（FP32 厳密）」契約。`gemm_bias_act_fused_path_
    /// increments_launch_counter_env_adaptive` と同じ分岐パターン）。
    #[test]
    fn gemm_stays_on_fp32_path_when_tf32_optin_flag_is_disabled_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        crate::precision::set_tf32_gemm_enabled(false);

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

        let before = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        let _ = cuda.gemm(&a, &b);
        let after = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "既定 OFF のはずが TF32 opt-in 経路のカウンタが増加した（フラグ OFF 時の \
             bit-exact 不変契約違反の疑い）: before={before}, after={after}"
        );
    }

    /// `gemm_fp32_strict`（`BackendOps` トレイト経由。`autodiff::grad` の
    /// VJP が使う入口。イシュー #1211 codex-review 指摘・PR #1223）は、
    /// TF32 opt-in フラグを **有効化した状態でも** TF32 経路
    /// （[`crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT`]）へ一切到達しない
    /// ことを検証する。`crate::precision` モジュール冒頭コメントの
    /// 「学習経路は本イシューのスコープ外のまま FP32」契約の直接検証で、
    /// backward が opt-in フラグへ暗黙追従しないことを担保する。
    #[test]
    fn gemm_fp32_strict_ignores_tf32_optin_flag_even_when_enabled_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        crate::precision::set_tf32_gemm_enabled(true);

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

        let before = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        let _ = cuda.gemm_fp32_strict(&a, &b);
        let after = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "TF32 opt-in フラグが有効でも gemm_fp32_strict は FP32 厳密経路のまま \
             であるべきだが、TF32 opt-in 経路のカウンタが増加した（学習経路の \
             FP32 契約違反の疑い）: before={before}, after={after}"
        );
    }

    /// opt-in（`true`）時、CUDA 実機が利用可能で TF32 カーネルが使用可能な
    /// 環境では `gemm` が [`crate::gemm::CudaGemm::run_wmma_tf32`] 経路へ
    /// 実際にルーティングされる（[`crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT`]
    /// の増加で検証）ことを確認する。CUDA 非搭載環境・TF32 カーネル使用
    /// 不能環境（`CudaError::WmmaUnavailable` 由来の
    /// `BackendError::KernelLaunchFailed`）ではエラーの型のみ確認して
    /// 早期 return する（fail-closed 契約: FP32 への黙示フォールバックを
    /// しないことの裏返しとして、エラーはそのまま伝播される）。
    #[test]
    fn gemm_routes_to_tf32_path_when_optin_flag_is_enabled_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        crate::precision::set_tf32_gemm_enabled(true);

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

        let before = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        match cuda.gemm(&a, &b) {
            Ok(_) => {
                let after = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
                assert!(
                    after > before,
                    "opt-in 時に TF32 経路の起動カウンタが増加していない（既定 \
                     FP32 経路へ黙示フォールバックした疑い）: before={before}, after={after}"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(BackendError::KernelLaunchFailed(msg)) => {
                // TF32 カーネル使用不能環境（cc<8.0 等）の fail-closed 伝播。
                // FP32 への黙示フォールバックはしない契約（`crate::precision`
                // モジュール冒頭コメント参照）。
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for tf32 opt-in gemm: {other}"),
        }
    }

    /// 環境適応: `crate::precision::gemm_precision()` が
    /// [`crate::precision::CudaGemmPrecision::Tf32x3`] のとき、`gemm` は
    /// 単発 TF32 経路（[`crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT`]）
    /// ではなく 3×TF32 経路
    /// （[`crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT`]）へのみ到達する
    /// ことを検証する（イシュー #1355）。CUDA 非搭載環境・カーネル使用
    /// 不能環境では `BackendError::CudaUnavailable`／
    /// `KernelLaunchFailed`（fail-closed 伝播）の型のみ確認する。
    ///
    /// 入力形状は 4×4（`n`・`k` とも 4 の倍数）を使う。3×TF32 経路は
    /// `gemm_mma_tf32.rs::validate_mma_tf32_alignment` により `n % 4 == 0
    /// && k % 4 == 0`（`cp.async` 16 バイト整列制約）を要求するため、
    /// 2×2 のような非対応形状では対応 GPU 上でも必ず
    /// `BackendError::KernelLaunchFailed`（形状拒否）で早期リターンし、
    /// 起動カウンタ増加を伴う本来の経路検証（成功時分岐）に到達しない
    /// （codex-review 指摘・PR #1400）。非対応形状の拒否自体は別途
    /// `gemm_mma_tf32.rs::tests::validate_mma_tf32_alignment_rejects_*`
    /// が検証する。
    #[test]
    fn gemm_routes_to_tf32x3_path_when_precision_is_tf32x3_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        crate::precision::set_gemm_precision(crate::precision::CudaGemmPrecision::Tf32x3);

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new((1..=16).map(|v| v as f32).collect(), &[4, 4]).expect("valid tensor");
        let b = Tensor::new((1..=16).map(|v| v as f32).collect(), &[4, 4]).expect("valid tensor");

        let before_tf32 = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        let before_tf32x3 = crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        match cuda.gemm(&a, &b) {
            Ok(_) => {
                let after_tf32 = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
                let after_tf32x3 = crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
                assert_eq!(
                    before_tf32, after_tf32,
                    "Tf32x3 opt-in 時に単発 TF32 経路のカウンタが増加した（誤配線の疑い）"
                );
                assert!(
                    after_tf32x3 > before_tf32x3,
                    "Tf32x3 opt-in 時に 3×TF32 経路の起動カウンタが増加していない: \
                     before={before_tf32x3}, after={after_tf32x3}"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(BackendError::KernelLaunchFailed(msg)) => {
                // 3×TF32 カーネル使用不能環境（cc<8.0 等）の fail-closed
                // 伝播。FP32 への黙示フォールバックはしない契約
                // （`crate::precision` モジュール冒頭コメント参照）。
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for tf32x3 opt-in gemm: {other}"),
        }
    }

    /// `Fp32Strict`／`Tf32` のいずれでも `gemm` が 3×TF32 経路
    /// （[`crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT`]）へ到達しない
    /// ことを検証する（イシュー #1355。3 モード分岐の相互排他性）。
    #[test]
    fn gemm_does_not_route_to_tf32x3_path_for_other_precision_modes_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");
        let cuda = CudaBackendOps::new(0);

        for mode in [
            crate::precision::CudaGemmPrecision::Fp32Strict,
            crate::precision::CudaGemmPrecision::Tf32,
        ] {
            crate::precision::set_gemm_precision(mode);
            let before = crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
            let _ = cuda.gemm(&a, &b);
            let after = crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
            assert_eq!(
                before, after,
                "mode={mode:?} で 3×TF32 経路のカウンタが増加した（誤配線の疑い）"
            );
        }
    }

    /// `gemm_fp32_strict`（VJP が使う入口）は `Tf32x3` opt-in 時にも
    /// 3×TF32 経路へ到達しない（`gemm_fp32_strict_ignores_tf32_optin_
    /// flag_even_when_enabled_env_adaptive` の Tf32x3 版。学習経路の
    /// FP32 契約は精度モードに関わらず一貫して守られる）。
    #[test]
    fn gemm_fp32_strict_ignores_tf32x3_precision_mode_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        crate::precision::set_gemm_precision(crate::precision::CudaGemmPrecision::Tf32x3);

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

        let before = crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        let _ = cuda.gemm_fp32_strict(&a, &b);
        let after = crate::gemm::TF32X3_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "Tf32x3 opt-in 時でも gemm_fp32_strict は FP32 厳密経路のままであるべきだが、\
             3×TF32 経路のカウンタが増加した: before={before}, after={after}"
        );
    }

    /// `run_fused` の canonical RMSNorm プラン検出（`rmsnorm.rs::
    /// match_rmsnorm_plan`）の型と同型の 6 op 列を組み立てる（`hidden`
    /// のみ差し替え）。`rmsnorm.rs::tests::build_canonical_rmsnorm_plan`
    /// と同じ op 列（`plan.rs::
    /// from_segment_builds_rmsnorm_plan_with_row_fusion_metadata` 参照）。
    fn build_canonical_rmsnorm_plan(
        hidden: usize,
        dtype: fandhe_ai_tensor_core::DType,
    ) -> FusionPlan {
        let ops = vec![
            fandhe_ai_tensor_core::FusedOpKind::Input { leaf_index: 0 },
            fandhe_ai_tensor_core::FusedOpKind::Mul { lhs: 0, rhs: 0 },
            fandhe_ai_tensor_core::FusedOpKind::Sum {
                input: 1,
                axis: None,
            },
            fandhe_ai_tensor_core::FusedOpKind::Rsqrt { input: 2 },
            fandhe_ai_tensor_core::FusedOpKind::Broadcast {
                input: 3,
                axis: None,
            },
            fandhe_ai_tensor_core::FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        FusionPlan::from_ops(ops, vec![hidden], dtype, 1).unwrap()
    }

    /// `run_fused` はカーネル起動（デバイスアクセス）前に `plan.dtype()
    /// == DType::F32` を検証するため、非 F32 プランは
    /// `BackendError::Unsupported` を返す（CUDA 非搭載環境でも決定的に
    /// 実行可能。`match_rmsnorm_plan` が一致した後の検証であることを
    /// 確認するため canonical op 列をそのまま使う。codex-review 指摘・
    /// PR #706 レビュー「融合プランの dtype と leaf shape を起動前に
    /// 検証する」）。
    #[test]
    fn run_fused_rejects_non_f32_dtype_before_device_access() {
        let plan = build_canonical_rmsnorm_plan(8, fandhe_ai_tensor_core::DType::F16);
        let x = Tensor::new(vec![1.0f32; 8], &[8]).expect("valid tensor");
        let cuda = CudaBackendOps::new(0);

        let err = cuda.run_fused(&plan, &[&x]).unwrap_err();
        assert!(
            matches!(err, BackendError::Unsupported(_)),
            "expected Unsupported for non-F32 dtype, got {err:?}"
        );
    }

    /// `run_fused` は leaf の shape が `plan.output_shape()` と厳密一致
    /// することも起動前に検証する。要素数が `row_len` と一致するだけの
    /// 異なる shape（`[8]` に対する `[2, 4]`）は
    /// `BackendError::ShapeMismatch` で拒否する（codex-review 指摘・
    /// PR #706 レビュー同上）。
    #[test]
    fn run_fused_rejects_leaf_shape_mismatch_before_device_access() {
        let plan = build_canonical_rmsnorm_plan(8, fandhe_ai_tensor_core::DType::F32);
        // 要素数（8）は `row_len` と一致するが shape が異なる。
        let x = Tensor::new(vec![1.0f32; 8], &[2, 4]).expect("valid tensor");
        let cuda = CudaBackendOps::new(0);

        let err = cuda.run_fused(&plan, &[&x]).unwrap_err();
        assert!(
            matches!(err, BackendError::ShapeMismatch(_)),
            "expected ShapeMismatch for leaf shape != output_shape, got {err:?}"
        );
    }

    // ---------------------------------------------------------------
    // Cursor Bugbot 指摘（PR #1064 追補）の回帰テスト。
    //
    // `context_cache` のプロセスワイド static レジストリはテスト間で
    // 共有されるため、他所（`context_cache.rs::poison_state_tests` は
    // 10000 番台、実機依存テストは ordinal 0/1）と衝突しない専用 ordinal
    // を払い出す（`context_cache.rs::poison_state_tests::unique_ordinal`
    // と同方針）。
    // ---------------------------------------------------------------

    fn unique_test_ordinal() -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(30_000);
        NEXT.fetch_add(1, Ordering::SeqCst)
    }

    fn sticky_driver_error() -> cudarc::driver::result::DriverError {
        cudarc::driver::result::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
        )
    }

    /// [`CudaBackendOps::with_driver_call`] の回帰テスト（Cursor Bugbot
    /// Medium 指摘・`ops.rs:100` 相当）: cold-cache 構築（`cached_gemm`
    /// 等）を表す最初のクロージャが sticky な driver エラーを返した場合、
    /// 修正前は構築呼び出しが `with_driver_call` の外側で素通しに実行され
    /// 観測されないため ordinal が poison されず、以降の呼び出しも
    /// fail-open のまま成功し続けた。修正後は構築呼び出し自体も
    /// `with_driver_call` で包むため、直後の（実行フェーズ相当の）
    /// 呼び出しが `DeviceContextPoisoned` で拒否されることを確認する。
    #[test]
    fn with_driver_call_poisons_ordinal_when_construction_closure_returns_sticky_error() {
        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);

        // 注意（CI 実機構成差で 1 度落ちた教訓）: `map` クロージャで
        // `CudaError::Driver(e)` を `e.to_string()`／`{e:?}` で整形すると、
        // `cudarc::driver::result::DriverError` の `Debug` 実装が
        // `cuGetErrorString`（driver API 経由。`culib()` の遅延ロードを
        // 要求する）を呼ぶ（cudarc-0.19.8 `src/driver/result.rs`）。
        // CUDA toolkit 非搭載環境（本テストの前提。CI `build-no-cuda-
        // toolkit` ジョブ）ではこのロードが `panic_no_lib_found` で
        // panic するため、poison 化ロジック自体とは無関係にテストが
        // 落ちる。本テストは poison 検査の副作用のみを検証すればよく
        // 実際の driver エラー詳細文字列は不要なため、`map` では
        // `CudaError` を整形せず固定メッセージにする。
        let construction_result: Result<(), BackendError> = cuda.with_driver_call(
            &[],
            |_e| BackendError::CudaUnavailable("simulated sticky driver error (test)".to_string()),
            || Err(CudaError::Driver(sticky_driver_error())),
        );
        assert!(
            construction_result.is_err(),
            "構築失敗はそのまま Err として伝播するはず"
        );

        let run_result: Result<(), BackendError> = cuda.with_driver_call(
            &[],
            |_e| {
                BackendError::KernelLaunchFailed(
                    "unexpected: should be rejected before this                 map is reached"
                        .to_string(),
                )
            },
            || Ok(()),
        );
        assert!(
            matches!(run_result, Err(BackendError::DeviceContextPoisoned(_))),
            "構築呼び出しで観測された sticky エラーにより ordinal は poison され、             以降の呼び出しは fail-closed に拒否されるはず: {run_result:?}"
        );
    }

    /// テスト専用の最小 `BufferHandle`（`tensor-core::backend_ops::
    /// EmptyHandle` と同型。データの実体は持たず、`DeviceBuffer` を
    /// 構築するためだけの空ハンドル）。
    #[derive(Debug)]
    struct EmptyHandle;

    impl fandhe_ai_tensor_core::buffer::BufferHandle for EmptyHandle {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    /// [`CudaBackendOps::gemm_resident_rhs`] の回帰テスト（Cursor Bugbot
    /// Low 指摘・`ops.rs:468` 相当の一般化）: `n == 0`（空入力）の早期
    /// return 分岐は、修正前は `device_handle()`／driver 呼び出しの手前で
    /// 無条件に `Ok` を返していたため、poison 済み ordinal でも「空
    /// 出力」が黙って成功していた。修正後は早期 return の直前で
    /// `begin_driver_call` の poison 検査を通すため、この分岐は driver
    /// 呼び出し（＝実機）を要求せずに poison 状態のみで再現・検証できる
    /// （`device_handle()` より手前で拒否されるため CUDA 非搭載環境でも
    /// 実行可能）。
    #[test]
    fn gemm_resident_rhs_rejects_on_poisoned_ordinal_even_via_trivial_empty_shape_early_return() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();

        // context_cache の poison 状態機械を直接操作して poison 化する
        // （`context_cache::poison_state_tests` と同じ手法）。
        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(sticky_driver_error())),
        );
        drop(token);

        let cuda = CudaBackendOps::new(ordinal);
        // `a` は `[m, k] = [1, 1]`（k != 0 のため k==0 分岐は通らない）。
        let a = Tensor::new(vec![1.0f32], &[1, 1]).expect("valid tensor");
        // `w` は `[k, n] = [1, 0]`（n == 0 のため対象の早期 return 分岐へ
        // 到達する）。ビュー構築は `buffer.numel() >= offset + numel`
        // のみを検査し、`numel == 0` のビューは任意のバッキングバッファに
        // 対して構築できる。
        let w_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let cuda_result = cuda.gemm_resident_rhs(&a, w, None);
        assert!(
            matches!(cuda_result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では n == 0 の早期 return 分岐も fail-closed に              拒否されるはず: {cuda_result:?}"
        );
    }

    /// [`CudaBackendOps::gemm_resident_rhs`] の回帰テスト（codex-review
    /// P1 指摘・`ops.rs:785` 相当・PR #1064 追補）: `m == 0 || n == 0`
    /// の早期 return 分岐は、修正前は `begin_driver_call` を空スライス
    /// （generation 検査なし）で呼んでいたため、`invalidate` 後の旧世代
    /// `w` ビューでもこの分岐だけ `StaleDeviceGeneration` を経由せず
    /// 成功してしまい、「旧世代のバッファは全て拒否する」という公開
    /// エラー契約を経路依存に破っていた。修正後は通常経路と同じ
    /// `w.buffer().generation()` を渡すため、旧世代スタンプ済みの `w`
    /// では空 shape でも `StaleDeviceGeneration` を返す（実機不要。
    /// `current_generation` は新規 ordinal で既定 `0` のため、`w` を
    /// 意図的にそれと異なる世代でスタンプするだけで再現できる）。
    #[test]
    fn gemm_resident_rhs_rejects_stale_generation_even_via_trivial_empty_shape_early_return() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        assert_eq!(
            context_cache::current_generation(ordinal),
            0,
            "新規 ordinal の現行世代は既定 0 のはず"
        );

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0f32], &[1, 1]).expect("valid tensor");
        // `w` を現行世代（0）とは異なる世代（1）でスタンプする
        // （`invalidate` による回復後に取り残された旧世代バッファを
        // 模す）。`m == 0 || n == 0` の早期 return 分岐（`w` は
        // `[k, n] = [1, 0]` で n == 0）へ到達させる。
        let w_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1],
            Box::new(EmptyHandle),
            1,
        );
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let result = cuda.gemm_resident_rhs(&a, w, None);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 w ビューは空 shape の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// [`CudaBackendOps::gemm_resident_lhs`] の同種回帰テスト（`p == 0 ||
    /// r == 0` 分岐）。
    #[test]
    fn gemm_resident_lhs_rejects_stale_generation_even_via_trivial_empty_shape_early_return() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        // `b` は `[q, r] = [1, 0]`（r == 0 のため `p == 0 || r == 0` 分岐へ
        // 到達する）。`w` は `[p, q] = [1, 1]` で世代 1 にスタンプする。
        let b = Tensor::new(Vec::new(), &[1, 0]).expect("valid tensor");
        let w_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1],
            Box::new(EmptyHandle),
            1,
        );
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 1]).expect("view construction succeeds");

        let result = cuda.gemm_resident_lhs(w, &b);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 w ビューは空 shape の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// [`CudaBackendOps::gemm_resident_lhs`] の `q == 0` 分岐の同種回帰
    /// テスト。
    #[test]
    fn gemm_resident_lhs_rejects_stale_generation_even_via_trivial_zero_contraction_dim_early_return()
     {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        // `w` は `[p, q] = [1, 0]`（q == 0）・`b` は `[q, r] = [0, 1]`。
        let b = Tensor::new(Vec::new(), &[0, 1]).expect("valid tensor");
        let w_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1],
            Box::new(EmptyHandle),
            1,
        );
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let result = cuda.gemm_resident_lhs(w, &b);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 w ビューは q == 0 の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// [`CudaBackendOps::gemm`] の回帰テスト（codex-review P0 指摘・
    /// `ops.rs:147` 相当。PR #1064 追補）: 修正前は `device_handle()`
    /// （キャッシュミス時に `CudaDevice::new` を呼び実際に driver を
    /// 操作する）が `with_driver_call`（`begin_driver_call` による poison
    /// 検査を含む）より前に呼ばれており、poison 済み ordinal でも拒否
    /// される前に driver 初期化が試みられ、その失敗も観測されなかった。
    /// 修正後は `device_handle_raw()` を `with_driver_call` のクロージャ
    /// 内部（poison 検査の後）へ移したため、poison 済み ordinal では
    /// `device_handle_raw()` 自体が一切呼ばれず、`begin_driver_call` の
    /// 拒否がそのまま返る。これは CUDA 非搭載環境でも検証できる: もし
    /// 修正が入っていなければ、この環境では `device_handle_raw()` が
    /// `BackendError::CudaUnavailable` 相当（`CudaError::DriverUnavailable`
    /// 等）を先に返してしまい、`DeviceContextPoisoned` へは到達しない
    /// （＝ poison 状態が観測できないまま別のエラーにすり替わる）。
    #[test]
    fn gemm_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        let ordinal = unique_test_ordinal();

        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(sticky_driver_error())),
        );
        drop(token);

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0, 2.0], &[2, 1]).expect("valid tensor");

        let result = cuda.gemm(&a, &b);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では device_handle_raw() が試行される前に              begin_driver_call が拒否するはず（CUDA 非搭載環境でも              CudaUnavailable にすり替わらないことを確認する）: {result:?}"
        );
    }

    /// イシュー #1014（設計文書 §8 T3b: 呼び出し側の拒否経路）: 上記
    /// `gemm_rejects_on_poisoned_ordinal_before_device_handle_is_attempted`・
    /// `gemm_resident_rhs_rejects_on_poisoned_ordinal_even_via_trivial_empty_shape_early_return`
    /// が既にカバーする `gemm`／`gemm_resident_rhs` を除く、残る公開演算
    /// エントリ（`add`〈`mul` は同じ `elementwise_binary` 経路を通るため
    /// 併せて代表する。`relu`〈`exp`／`tanh` は同じ `elementwise_unary`
    /// 経路を通るため併せて代表する〉は下記
    /// `relu_rejects_on_poisoned_ordinal_before_device_handle_is_attempted`
    /// で個別に検証する（codex-review 指摘・PR #1067。`elementwise_binary`
    /// と `elementwise_unary` はディスパッチ関数自体が分かれているため
    /// `add` 側の検証だけでは `elementwise_unary` 経路を通らない）〉・
    /// `sgd_step_device`・`gemm_resident_lhs`・
    /// `release_cached_device_memory`）が、poison 済み ordinal では実処理
    /// （driver 呼び出し）へ一切入らず `BackendError::DeviceContextPoisoned`
    /// を即座に返すことを確認する（GPU 不要・CI 常時実行）。
    ///
    /// 実 CUDA fault 注入は行わず（`.claude/rules/coding-rust.md`
    /// カーネル境界検査の規約に反するため）、`context_cache` の poison
    /// 状態機械を直接セットする方式を採る（設計文書 §8 T3b が明示的に
    /// 許容する代替経路）。`MemoryOps::upload`／`download` は同一の
    /// `with_driver_call` ゲート（`memory.rs` の `CudaMemory::
    /// with_driver_call`）を通るが、そちらは `context_cache::
    /// poison_state_tests`（GPU 非依存モック）で既に等価な検証がある
    /// ため、ordinal 0/1（実機テストが使う共有 ordinal）を汚染してまで
    /// ここで重複検証しない（イシュー #1014 実装計画 §3 方針 2 の判断）。
    fn poison_ordinal(ordinal: usize) {
        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(sticky_driver_error())),
        );
        drop(token);
    }

    #[test]
    fn add_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");

        let result = cuda.add(&a, &b);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では add（elementwise_binary 経由。mul も同一経路）は             device_handle_raw() が試行される前に拒否されるはず: {result:?}"
        );
    }

    #[test]
    fn relu_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        // codex-review 指摘（PR #1067）: 上記 `add_rejects_on_poisoned_
        // ordinal_before_device_handle_is_attempted` は `elementwise_binary`
        // 経路のみを通り、`relu`／`exp`／`tanh` が使う `elementwise_unary`
        // 経路は未検証だった。`elementwise_binary`／`elementwise_unary` は
        // ともに `with_driver_call` を経由する同一のゲート構造だが、
        // ディスパッチ関数自体が分かれているため実際に両方を通しておく。
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0, -2.0], &[1, 2]).expect("valid tensor");

        let result = cuda.relu(&a);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では relu（elementwise_unary 経由。exp／tanh も             同一経路）は device_handle_raw() が試行される前に拒否されるはず:             {result:?}"
        );
    }

    #[test]
    fn mse_loss_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        // イシュー #1045: `mse_loss`／`mse_loss_backward` は `elementwise_
        // binary`／`elementwise_unary` とは別のディスパッチ関数
        // （`context_cache::cached_mse` 経由）のため、`with_driver_call`
        // ゲートが正しく結線されていることを個別に確認する
        // （`relu_rejects_on_poisoned_ordinal...` のコメントと同じ理由）。
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let pred = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");
        let target = Tensor::new(vec![0.0, 0.0], &[1, 2]).expect("valid tensor");

        let forward = cuda.mse_loss(&pred, &target, MseReduction::Mean);
        assert!(
            matches!(forward, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では mse_loss は device_handle_raw() が試行される前に \
             拒否されるはず: {forward:?}"
        );

        let backward = cuda.mse_loss_backward(&pred, &target, 1.0);
        assert!(
            matches!(backward, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では mse_loss_backward は device_handle_raw() が試行される \
             前に拒否されるはず: {backward:?}"
        );
    }

    #[test]
    fn sgd_step_device_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        use fandhe_ai_tensor_core::SgdStepConfig;
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let mut param =
            DeviceBuffer::<f32>::new(Device::Cuda(ordinal), vec![4], Box::new(EmptyHandle));
        let grad = DeviceBuffer::<f32>::new(Device::Cuda(ordinal), vec![4], Box::new(EmptyHandle));
        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };

        // momentum == 0.0 のため velocity は不要（`use_momentum` 分岐を
        // 通らない。`ops.rs::sgd_step_device` 参照）。poison 検査
        // （`cached_sgd` 構築の `with_driver_call`）は device/shape の
        // 事前検証の後・実際のバッファ downcast より前に走る。
        let result = cuda.sgd_step_device(&mut param, &grad, None, &config);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では sgd_step_device は cached_sgd 構築より前に             拒否されるはず: {result:?}"
        );
    }

    #[test]
    fn gemm_resident_lhs_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        // `w` は `[p, q] = [1, 0]`（q == 0）の早期 return 分岐（`begin_driver_call`
        // による poison 検査を経由する）へ到達させる。`p`／`r` が非ゼロの
        // 一般経路は poison 検査より先に `w.buffer().downcast_handle::
        // <CudaBufferHandle>()` を呼ぶため、テスト専用の `EmptyHandle`
        // （`CudaBufferHandle` ではない）を渡すと `DeviceContextPoisoned`
        // ではなく `DeviceMismatch` にすり替わってしまう（`gemm_resident_lhs`
        // の実装順序どおり）。早期 return 分岐を使うことでこの問題を回避
        // する（`gemm_resident_lhs_rejects_stale_generation_even_via_trivial_
        // zero_contraction_dim_early_return` と同じ手法）。
        let w_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");
        let b = Tensor::new(Vec::new(), &[0, 1]).expect("valid tensor");

        let result = cuda.gemm_resident_lhs(w, &b);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では gemm_resident_lhs は q == 0 の早期 return 分岐でも             拒否されるはず: {result:?}"
        );
    }

    #[test]
    fn release_cached_device_memory_rejects_on_poisoned_ordinal_before_device_handle_is_attempted()
    {
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let result = cuda.release_cached_device_memory();
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では release_cached_device_memory は             device_handle_raw() が試行される前に拒否されるはず: {result:?}"
        );
    }
    // ---------------------------------------------------------------
    // `CudaBackendOps::linear_forward_device`（イシュー #1216）の CI 実行
    // 可能な回帰テスト（実機不要）。`gemm_resident_rhs` の同種テスト
    // （poison／stale generation／DeviceMismatch）と同じ手法・同じ専用
    // ordinal 払い出し方針。
    // ---------------------------------------------------------------

    /// poison 済み ordinal では `m == 0 || n == 0` の早期 return 分岐も
    /// fail-closed に拒否されるはず（`gemm_resident_rhs` の同種テストと
    /// 同じ理由。`a` の世代検査対象追加〈本メソッド固有〉があっても
    /// poison 検査自体は変わらないことを確認する）。
    #[test]
    fn linear_forward_device_rejects_on_poisoned_ordinal_even_via_trivial_empty_shape_early_return()
    {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();

        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(sticky_driver_error())),
        );
        drop(token);

        let cuda = CudaBackendOps::new(ordinal);
        // `a` は `[m, k] = [1, 1]`（k != 0 のため k==0 分岐は通らない）。
        let a_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1, 1], Box::new(EmptyHandle));
        // `w` は `[k, n] = [1, 0]`（n == 0 のため対象の早期 return 分岐へ
        // 到達する）。
        let w_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let result = cuda.linear_forward_device(&a_buffer, w, None, Activation::None);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では n == 0 の早期 return 分岐も fail-closed に              拒否されるはず: {result:?}"
        );
    }

    /// 旧世代の `w` ビューは空 shape の早期 return でも
    /// `StaleDeviceGeneration` で拒否されるはず（`gemm_resident_rhs` の
    /// 同種テストと同じ理由）。
    #[test]
    fn linear_forward_device_rejects_stale_generation_of_w_even_via_trivial_empty_shape_early_return()
     {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        assert_eq!(
            context_cache::current_generation(ordinal),
            0,
            "新規 ordinal の現行世代は既定 0 のはず"
        );

        let cuda = CudaBackendOps::new(ordinal);
        let a_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1, 1], Box::new(EmptyHandle));
        // `w` を現行世代（0）とは異なる世代（1）でスタンプする。
        let w_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1],
            Box::new(EmptyHandle),
            1,
        );
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let result = cuda.linear_forward_device(&a_buffer, w, None, Activation::None);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 w ビューは空 shape の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// `a` 自体が旧世代（呼び出しを跨いで生存する常駐バッファという本
    /// メソッド固有の性質）でも、空 shape の早期 return で
    /// `StaleDeviceGeneration` により拒否されるはず（`gemm_resident_rhs`
    /// は `a` を世代検査対象にしないため存在しない、本メソッド固有の
    /// 回帰テスト。doc comment「世代検査の対象に `a` を含む」参照）。
    #[test]
    fn linear_forward_device_rejects_stale_generation_of_a_even_via_trivial_empty_shape_early_return()
     {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        assert_eq!(
            context_cache::current_generation(ordinal),
            0,
            "新規 ordinal の現行世代は既定 0 のはず"
        );

        let cuda = CudaBackendOps::new(ordinal);
        // `a` を現行世代（0）とは異なる世代（1）でスタンプする。
        let a_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1, 1],
            Box::new(EmptyHandle),
            1,
        );
        // `w` は `[k, n] = [1, 0]`（n == 0 のため早期 return 分岐へ到達）。
        let w_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let result = cuda.linear_forward_device(&a_buffer, w, None, Activation::None);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 a バッファは空 shape の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// CPU バッファ（`Device::Cpu`）を `a` に渡すと driver へ触れる前に
    /// `DeviceMismatch` で拒否されるはず（`build-no-cuda-toolkit` ジョブ
    /// でも実行可能。実機不要）。
    #[test]
    fn linear_forward_device_rejects_device_mismatch_on_a() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        // `a` は CPU デバイスのバッファ（driver へ触れる前に拒否される
        // ことを確認するのが目的のため、CudaBufferHandle である必要は
        // ない）。
        let a_buffer = DeviceBuffer::new(Device::Cpu, vec![1, 1], Box::new(EmptyHandle));
        let w_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 1]).expect("view construction succeeds");

        let result = cuda.linear_forward_device(&a_buffer, w, None, Activation::None);
        assert!(
            matches!(result, Err(BackendError::DeviceMismatch)),
            "CPU デバイスの a は DeviceMismatch で拒否されるはず: {result:?}"
        );
    }

    /// `w` が別 ordinal（別 GPU）のバッファの場合も `DeviceMismatch` で
    /// 拒否されるはず（実機不要）。
    #[test]
    fn linear_forward_device_rejects_device_mismatch_on_w() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let other_ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        let a_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1, 1], Box::new(EmptyHandle));
        let w_buffer =
            DeviceBuffer::new(Device::Cuda(other_ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 1]).expect("view construction succeeds");

        let result = cuda.linear_forward_device(&a_buffer, w, None, Activation::None);
        assert!(
            matches!(result, Err(BackendError::DeviceMismatch)),
            "別 ordinal の w は DeviceMismatch で拒否されるはず: {result:?}"
        );
    }

    // ---------------------------------------------------------------
    // `CudaBackendOps::gemm_fp32_strict_into`／`_tracked`（イシュー
    // #1559）の CI 実行可能な回帰テスト（実機不要）。境界検査
    // （`DeviceMismatch`／`InvalidArgument`）はいずれも driver 呼び出し
    // より前の host-only チェックのため、`EmptyHandle`（`CudaBufferHandle`
    // ではないダミーハンドル）を渡した `out` でも検証できる
    // （`gemm_resident_rhs` 等の同種テストと同じ手法）。
    // ---------------------------------------------------------------

    /// `out.device()` が `self.ordinal` と一致しない場合、shape・offset
    /// 検証やカーネル起動を一切行わず `DeviceMismatch` を返すはず
    /// （driver に一切触れない host-only チェック。`gemm_fp32_strict_into_impl`
    /// 実装順序の先頭。`EmptyHandle` を渡しても downcast まで到達しない
    /// ため安全に検証できる）。
    #[test]
    fn gemm_fp32_strict_into_rejects_wrong_device_out() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let other_ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);

        let a = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let mut out =
            DeviceBuffer::new(Device::Cuda(other_ordinal), vec![4], Box::new(EmptyHandle));

        let result = cuda.gemm_fp32_strict_into(&a, &b, &mut out, 0);
        assert!(
            matches!(result, Err(BackendError::DeviceMismatch)),
            "別 ordinal の out は DeviceMismatch で拒否されるはず: {result:?}"
        );
    }

    /// `out_offset + m*n` が `out.numel()` を超える場合、driver に触れず
    /// `InvalidArgument` を返すはず（REQ-8・OWASP A03。カーネル起動より
    /// 前の境界検査）。
    #[test]
    fn gemm_fp32_strict_into_rejects_out_of_range_offset() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);

        let a = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        // m*n == 4 だが `out` は 3 要素分しか確保しないため
        // `out_offset(0) + 4 > 3` で拒否されるはず。
        let mut out = DeviceBuffer::new(Device::Cuda(ordinal), vec![3], Box::new(EmptyHandle));

        let result = cuda.gemm_fp32_strict_into(&a, &b, &mut out, 0);
        assert!(
            matches!(result, Err(BackendError::InvalidArgument(_))),
            "out_offset + m*n が out.numel() を超える場合は InvalidArgument で             拒否されるはず: {result:?}"
        );
    }

    /// `out_offset` が `usize::MAX` 等で `checked_add` オーバーフローする
    /// 場合も、driver に触れず `InvalidArgument` を返すはず。
    #[test]
    fn gemm_fp32_strict_into_rejects_offset_overflow() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);

        let a = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let mut out = DeviceBuffer::new(Device::Cuda(ordinal), vec![4], Box::new(EmptyHandle));

        let result = cuda.gemm_fp32_strict_into(&a, &b, &mut out, usize::MAX);
        assert!(
            matches!(result, Err(BackendError::InvalidArgument(_))),
            "out_offset + m*n が usize でオーバーフローする場合は InvalidArgument で             拒否されるはず: {result:?}"
        );
    }

    /// poison 済み ordinal では、退化形状（`m == 0`）によりフォールバック
    /// （`self.gemm_fp32_strict(a, b)`）へ直行する経路でも
    /// `DeviceContextPoisoned` で拒否されるはず（`gemm_resident_lhs_
    /// rejects_on_poisoned_ordinal_before_device_handle_is_attempted` と
    /// 同じ手法。`gemm_fp32_strict` 自身の poison 検査を経由することを
    /// 示す）。`out` は `EmptyHandle` のまま（フォールバックの
    /// `gemm_fp32_strict` 呼び出しで拒否されるため downcast まで到達
    /// しない）。
    #[test]
    fn gemm_fp32_strict_into_rejects_on_poisoned_ordinal_via_degenerate_shape_fallback() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        // m == 0 の退化形状。`a`：[0, 2]、`b`：[2, 2] → out shape [0, 2]。
        let a = Tensor::new(Vec::new(), &[0, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let mut out = DeviceBuffer::new(Device::Cuda(ordinal), vec![0], Box::new(EmptyHandle));

        let result = cuda.gemm_fp32_strict_into(&a, &b, &mut out, 0);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では gemm_fp32_strict_into は退化形状フォールバック             経由でも拒否されるはず: {result:?}"
        );
    }

    /// [`CudaBackendOps::gemm_fp32_strict_into_tracked`] は `token` を
    /// 無視して [`CudaBackendOps::gemm_fp32_strict_into`] と同一の結果を
    /// 返すはず（`sgd_step_device_tracked_default_delegates_to_
    /// sgd_step_device`〈tensor-core〉と同型の委譲検証。エラーが決定的な
    /// poison 済み ordinal・退化形状の組み合わせで両呼び出しを比較する）。
    #[test]
    fn gemm_fp32_strict_into_tracked_delegates_to_gemm_fp32_strict_into() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(Vec::new(), &[0, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");

        let mut out_direct =
            DeviceBuffer::new(Device::Cuda(ordinal), vec![0], Box::new(EmptyHandle));
        let result_direct = cuda.gemm_fp32_strict_into(&a, &b, &mut out_direct, 0);

        let mut out_tracked =
            DeviceBuffer::new(Device::Cuda(ordinal), vec![0], Box::new(EmptyHandle));
        let token = DispatchFailureCell::new();
        let result_tracked =
            cuda.gemm_fp32_strict_into_tracked(&a, &b, &mut out_tracked, 0, &token);

        assert!(
            matches!(result_direct, Err(BackendError::DeviceContextPoisoned(_))),
            "前提: gemm_fp32_strict_into 自体が poison で拒否されること: {result_direct:?}"
        );
        assert_eq!(
            format!("{result_direct:?}"),
            format!("{result_tracked:?}"),
            "gemm_fp32_strict_into_tracked は token を無視して gemm_fp32_strict_into と             同一の結果を返すはず"
        );
    }

    /// [`CudaBackendOps::with_sync_point_call`]（イシュー #1559。
    /// `gemm_fp32_strict_into` NT/TN 経路が使う capture 対応の同期点
    /// ガード）が、CUDA Graph capture 中は driver に一切触れず
    /// `Unsupported` で拒否することを確認する（`context_cache::
    /// begin_sync_point_call_rejects_before_touching_driver_while_capturing`
    /// と同じ検証を `CudaBackendOps` 側の薄いラッパー経由で行う。GPU 不要
    /// ——`begin_capture_session` 自体は純粋な状態機械操作で driver を
    /// 呼ばない）。
    #[test]
    fn with_sync_point_call_rejects_before_touching_driver_while_capturing() {
        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        let _guard = context_cache::begin_capture_session(ordinal).expect(
            "begin_capture_session は driver に触れない純粋な状態機械操作のため \
             GPU 非依存で成功するはず",
        );

        let result: Result<(), BackendError> = cuda.with_sync_point_call(
            &[],
            "gemm_fp32_strict_into",
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || Ok(()),
        );
        assert!(
            matches!(&result, Err(BackendError::Unsupported(msg)) if msg.contains("gemm_fp32_strict_into")),
            "capture 中の with_sync_point_call は Unsupported で拒否されるはず: {result:?}"
        );
    }

    /// codex-review 指摘（PR #1569）の再発防止テスト: `gemm_fp32_strict_
    /// into` の NN 分岐（`dense_transposed_view` がどちらも `None` を
    /// 返す通常 shape）は `with_sync_point_call`（NT/TN 経路限定）を
    /// 経由せずフォールバック（`gemm_fp32_strict` → `run_tiled_f32` 系。
    /// 内部でホスト同期の D2H readback を伴う）へ進むため、修正前は
    /// capture 中でも `cached_gemm` 取得（`with_driver_call`。同一
    /// スレッド capture 中は通過する設計）を経て実際に driver へ触れ
    /// うる欠陥があった。`gemm_fp32_strict_into_impl` 冒頭の共通入口
    /// 検査（`context_cache::is_capturing_on_current_thread`）が
    /// **どの driver 呼び出しよりも前**に `Unsupported` で拒否する
    /// ことを、GPU 不要（`begin_capture_session` は純粋な状態機械
    /// 操作。driver ハンドルを一切取得しない未初期化 ordinal で検証する
    /// ことで「driver に到達する前に拒否された」ことを間接的に確認する）
    /// で検証する。
    #[test]
    fn gemm_fp32_strict_into_rejects_nn_fallback_branch_before_touching_driver_while_capturing() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        let _guard = context_cache::begin_capture_session(ordinal)
            .expect("begin_capture_session は driver 非依存の状態機械操作のため成功するはず");

        // NN（両オペランドとも `dense_transposed_view` が `None` を返す
        // 通常 shape）。転置カーネル可用性照会（`transpose_smem_f32_
        // available`）にすら到達せず、本関数入口で拒否されるはず。
        let a = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("valid tensor");
        let mut out = DeviceBuffer::new(Device::Cuda(ordinal), vec![0], Box::new(EmptyHandle));

        let result = cuda.gemm_fp32_strict_into(&a, &b, &mut out, 0);
        assert!(
            matches!(&result, Err(BackendError::Unsupported(msg)) if msg.contains("gemm_fp32_strict_into")),
            "capture 中の gemm_fp32_strict_into（NN フォールバック分岐）は \
             with_sync_point_call を経由せずとも本関数入口で Unsupported を \
             返すはず: {result:?}"
        );
    }
}

/// [`dense_transposed_view`]（イシュー #1214）の純ロジック検証（GPU
/// 不要。`backend-cpu::ops::repack_count_tests` と同じ判定条件・入力を
/// 使い、CPU 版と同一の判定結果になることを確認する）。
#[cfg(test)]
mod dense_transposed_view_tests {
    use super::*;

    #[test]
    fn dense_transposed_view_returns_some_for_transpose_2d_of_contiguous_tensor() {
        let w = Tensor::new(vec![1.0f32; 5 * 3], &[5, 3]).expect("valid tensor");
        let w_t = w.transpose_2d().expect("transpose succeeds");
        assert!(
            dense_transposed_view(&w_t).is_some(),
            "行優先連続テンソルの transpose_2d() は dense 転置 view と判定されるはず"
        );
    }

    #[test]
    fn dense_transposed_view_returns_none_for_narrow_then_transpose() {
        // 列方向の narrow は行ストライド（= 元の列数）が narrow 後の
        // 列数より大きくなるため、一般 stride（`ld != rows`）になる
        // （`backend-cpu::ops::repack_count_tests` と同じ理由）。
        let w0 = Tensor::new(vec![1.0f32; 5 * 7], &[5, 7]).expect("valid tensor");
        let w_narrowed = w0.narrow(1, 1, 3).expect("narrow succeeds");
        let w_t = w_narrowed.transpose_2d().expect("transpose succeeds");
        assert!(
            dense_transposed_view(&w_t).is_none(),
            "narrow 後の転置（一般 stride）は None を返すはず"
        );
    }

    #[test]
    fn dense_transposed_view_returns_none_for_rank_mismatch() {
        let t = Tensor::new(vec![1.0f32; 6], &[6]).expect("valid tensor");
        assert!(
            dense_transposed_view(&t).is_none(),
            "rank != 2 は None を返すはず"
        );
    }

    #[test]
    fn dense_transposed_view_returns_none_for_zero_dim_shape() {
        let w = Tensor::new(Vec::new(), &[0, 3]).expect("valid tensor");
        let w_t = w.transpose_2d().expect("transpose succeeds");
        assert!(
            dense_transposed_view(&w_t).is_none(),
            "rows == 0 || cols == 0 は None を返すはず（呼び出し元分岐の単純化）"
        );
    }
}

/// [`GEMM_HOST_REPACK_COUNT`]／[`crate::gemm::GEMM_TRANSPOSED_ENTRY_
/// LAUNCH_COUNT`]（クレート境界外の統合テストから見えない `pub(crate)`
/// カウンタ）が「dense 転置 view（NT/TN）では GPU 側転置カーネルへ到達し
/// `contiguous()` フォールバックを通らない・TT／一般 stride ではフォール
/// バックを通る」ことを検証するクレート内テスト（イシュー #1214。
/// `backend-cpu::ops::repack_count_tests`〈#1213〉と同型）。
///
/// `gemm_fp32_strict_impl` は NT/TN 判定より前に `with_driver_call` で
/// `cached_gemm` を取得するため（実装計画 §4「`gemm` 取得と実行を 1 つの
/// `with_driver_call` クロージャ内に収める」・PR #1064 の fail-open 窓
/// 再開防止と同じ理由でこの順序自体は変えない）、CUDA 非搭載環境では
/// カウンタ計上より前に `BackendError::CudaUnavailable` で早期 return
/// する。そのため本テスト群は環境適応（`gemm_bias_act_fused_path_
/// increments_launch_counter_env_adaptive` と同じ Ok/Err(CudaUnavailable)
/// 分岐パターン）とする。数値一致自体は統合テスト
/// `tests/gemm_transposed_parity.rs` が担当し、本テストはルーティングの
/// 健全性のみを確認する。
#[cfg(test)]
mod repack_count_tests {
    use super::*;

    fn reset_counters() {
        GEMM_HOST_REPACK_COUNT.with(|c| c.set(0));
        crate::gemm::GEMM_TRANSPOSED_ENTRY_LAUNCH_COUNT.with(|c| c.set(0));
    }

    #[test]
    fn gemm_fp32_strict_dense_transposed_view_routes_to_transpose_entry_env_adaptive() {
        reset_counters();
        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0f32; 4 * 3], &[4, 3]).expect("valid tensor");
        let w = Tensor::new(vec![1.0f32; 5 * 3], &[5, 3]).expect("valid tensor");
        let w_t = w.transpose_2d().expect("transpose succeeds");

        let repack_before = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        let entry_before = crate::gemm::GEMM_TRANSPOSED_ENTRY_LAUNCH_COUNT.with(|c| c.get());
        match cuda.gemm_fp32_strict_impl(&a, &w_t) {
            Ok(_) => {
                assert_eq!(
                    GEMM_HOST_REPACK_COUNT.with(|c| c.get()),
                    repack_before,
                    "dense な転置 view（NT）は contiguous() フォールバックを通らないはず"
                );
                assert!(
                    crate::gemm::GEMM_TRANSPOSED_ENTRY_LAUNCH_COUNT.with(|c| c.get())
                        > entry_before,
                    "GPU 側転置カーネル（transpose_smem_f32）へ到達していない疑い"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for gemm_fp32_strict_impl: {other}"),
        }
    }

    #[test]
    fn gemm_fp32_strict_narrow_then_transpose_increments_repack_counter_env_adaptive() {
        reset_counters();
        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0f32; 4 * 3], &[4, 3]).expect("valid tensor");
        let w0 = Tensor::new(vec![1.0f32; 5 * 7], &[5, 7]).expect("valid tensor");
        let w_narrowed = w0.narrow(1, 1, 3).expect("narrow succeeds");
        let w_t = w_narrowed.transpose_2d().expect("transpose succeeds");

        let before = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        match cuda.gemm_fp32_strict_impl(&a, &w_t) {
            Ok(_) => {
                assert_eq!(
                    GEMM_HOST_REPACK_COUNT.with(|c| c.get()),
                    before + 1,
                    "narrow 後の転置（一般 stride）は contiguous() フォールバックを通るはず"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for gemm_fp32_strict_impl: {other}"),
        }
    }

    #[test]
    fn gemm_fp32_strict_both_transposed_increments_repack_counter_twice_env_adaptive() {
        reset_counters();
        let cuda = CudaBackendOps::new(0);
        let orig_a = Tensor::new(vec![1.0f32; 3 * 4], &[3, 4]).expect("valid tensor");
        let a_t = orig_a.transpose_2d().expect("transpose succeeds");
        let orig_b = Tensor::new(vec![1.0f32; 5 * 3], &[5, 3]).expect("valid tensor");
        let b_t = orig_b.transpose_2d().expect("transpose succeeds");

        let before = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        match cuda.gemm_fp32_strict_impl(&a_t, &b_t) {
            Ok(_) => {
                assert_eq!(
                    GEMM_HOST_REPACK_COUNT.with(|c| c.get()),
                    before + 2,
                    "両方転置（TT）は両オペランドとも contiguous() フォールバックを通るはず"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for gemm_fp32_strict_impl: {other}"),
        }
    }
}
