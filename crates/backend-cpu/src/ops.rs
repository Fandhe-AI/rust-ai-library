//! CPU バックエンドの `BackendOps` 実装（TASK-1.9c・#46）。
//!
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の CPU 実装。既存カーネル
//! （`gemm_blis::gemm_blis_parallel`・`elementwise::{add,mul,relu,exp,tanh}`・
//! `reduction::{sum,max}`）への薄い委譲に徹し、カーネル本体・許容誤差・
//! 境界検査には一切触れない（`.claude/rules/delegation-impl.md` の
//! 実装フロー標準どおり、本ファイルはディスパッチ層のみを追加する）。
//! CPU は常に利用可能なため（`device::CpuDeviceProvider` と同じ位置付け）
//! 全 8 演算とも `Unsupported` を返す経路は持たない（TASK-1.9c の受け入れ
//! 条件「3 バックエンドが呼び分けられる」の参照実装として、CPU は常に
//! 実カーネルを実行できることを保証する）。

use std::cell::Cell;
use std::sync::OnceLock;

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    Activation, BackendOps, BinaryElementwiseOp, ChecksumReadout, DType, FusionPlan, GemmChecksum,
    GruBackwardOutput, GruPointwiseOutput, LstmPointwiseOutput, MatrixNormOrd, MseReduction,
    QrFactors, SgdStepConfig, ShapeError, SvdFactors, Tensor, UnaryElementwiseOp,
    require_same_shape, row_norm_layout, row_softmax_layout,
};

use crate::gemm_blis::{
    gemm_blis_bias_act_parallel, gemm_blis_parallel, gemm_blis_parallel_nt, gemm_blis_parallel_tn,
};
use crate::layer_norm;
use crate::linalg::{self, LinalgError};
use crate::memory::{CpuBufferHandle, CpuMemory};
use crate::rmsnorm::{self, match_rmsnorm_plan};
use crate::softmax::{self, match_softmax_plan};
use crate::{elementwise, fused_elementwise, mse, reduction, rnn_cell};

/// `CpuBackendOps` が `MemoryOps` を実装するための、プロセスワイドに共有
/// する単一 `CpuMemory`（イシュー #935・`docs/device-resident-update-design.md`
/// §3.3d「`AllocationTracker` の計測系列単一化」）。
///
/// `CpuBackendOps` は `#[derive(Debug, Default, Clone, Copy)] pub struct
/// CpuBackendOps;`（unit struct）であり、`fandhe-ai-backend-cpu` として
/// crates.io へ公開済みのためフィールド追加は破壊的変更になりうる
/// （実装計画 §3.2「CPU は `CpuBackendOps` が unit struct で公開済みの
/// ためフィールド追加不可」）。そのためプロセスワイド `static` で
/// `AllocationTracker` の計測系列を共有する（`CpuMemory::new()` を毎回
/// 呼ぶと `Arc<AllocationTracker>` が呼び出しごとに新規生成され、
/// `sgd_step_device` の一連の `alloc_zeroed`／`upload`／`download` 呼び出し
/// 間でピーク計測が繋がらなくなるため）。
fn shared_cpu_memory() -> &'static CpuMemory {
    static SHARED: OnceLock<CpuMemory> = OnceLock::new();
    SHARED.get_or_init(CpuMemory::new)
}

/// CPU バックエンドの `BackendOps` 実装。状態を持たないゼロサイズ型
/// （CPU カーネルはホストメモリのみを扱い、CUDA `CudaDevice`／Metal
/// `MetalContext` のようなデバイスハンドルを必要としないため）。
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackendOps;

impl CpuBackendOps {
    /// 新規 `CpuBackendOps` を構築する。
    pub fn new() -> Self {
        Self
    }
}

/// `Tensor::contiguous()` 実体化後もなお `as_slice()` が `None` を返す
/// （契約上到達しないはずだが、`Tensor` 実装のバグに対する fail-safe と
/// して型付きエラーで受ける）場合の変換ヘルパー。shape 不一致ではなく
/// 実行時の契約違反であるため `BackendError::KernelLaunchFailed` を返す
/// （命名を実際のエラー種別に合わせ `gemm_shape_mismatch` から改名。
/// Review 指摘対応）。
fn gemm_contiguity_fail_safe(msg: impl std::fmt::Display) -> BackendError {
    BackendError::KernelLaunchFailed(msg.to_string())
}

thread_local! {
    /// `gemm`／`gemm_resident_lhs` の呼び出しのうち、片側オペランドが
    /// dense な転置 view（[`dense_transposed_view`] が `Some` を返す
    /// 形状）と判定できず `Tensor::contiguous()` の再パックコピーへ
    /// フォールバックした回数（イシュー #1213）。`backend-metal::ops::
    /// RESIDENT_HOST_REPACK_COUNT` と同型の可観測点で、`#[cfg(test)]`
    /// クレート内テストから「NT/TN 判定が効いてフォールバックを通って
    /// いないこと」を検証するために使う（`pub(crate)`。クレート境界外の
    /// 統合テストからは参照できないため、外部テストファイルは数値一致
    /// のみを検証する契約とする。`RESIDENT_HOST_REPACK_COUNT` ドキュメント
    /// コメントと同じ設計判断）。
    pub(crate) static GEMM_HOST_REPACK_COUNT: Cell<u64> = const { Cell::new(0) };
}

/// `t` が「dense な転置格納」（`Tensor::transpose_2d()` を経た zero-copy
/// view のうち、元テンソルが行優先連続だったもの）であれば、その
/// storage をそのまま借用したフラットスライスを返す（イシュー #1213）。
///
/// 判定条件は `rank() == 2 && strides() == [1, shape()[0]]`。これは
/// `Tensor::transpose_2d`（`transpose(0,1)` の薄い委譲）が行優先連続
/// テンソルへ適用された結果と同値であり、返るスライスは「転置元の
/// テンソル」を行優先で並べたバイト列そのものになる（呼び出し元が
/// `ATPackTile`／`BTPackTile`〈`crate::gemm_blis::pack`〉の `k_total`／
/// `m_total`／`n_total` を正しく解釈する前提。`gemm`／`gemm_resident_lhs`
/// のみが呼ぶ）。
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

/// `CpuBackendOps::gemm` の出力バッファをゼロ確保するしきい値
/// （要素数。イシュー #1299）。`gemm_reuse_phase_diag_tests` の
/// フェーズ分解実測（`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
/// §15）で、出力確保 `alloc_c` が `iter_total` に対し実測上意味を持つのは
/// DGX Spark GB10 の N=2048（4,194,304 要素 = 16 MiB）のみで、N=1024
/// （1,048,576 要素）以下は無視できる水準（24.3 µs 未満）だった。この
/// しきい値未満は従来どおり [`zeroed_output_with_threshold`] 内で
/// `vec![0.0f32; len]`（calloc 相当の逐次ゼロ確保）を使い、既存経路と
/// 完全同一のまま変更しない。
///
/// **本番既定は `usize::MAX`（並列分岐を常に無効化）。**
///
/// イシュー #1299 の M4 Max スモーク実測（`docs/perf/logs/
/// cpu-matmul-fixed-cost-1299/`）では、他 worktree の並走ビルドで
/// load average 9〜11 という高負荷下の計測により N=2048 の `ops_gemm`
/// が中央値約 29% 後退したため、当初は `usize::MAX`（無効化）を本番
/// 既定としていた。イシュー #1301 が DGX Spark GB10（Grace CPU）・
/// Apple M4 Max の両実機で 5 回独立プロセス起動・on/off 比較を実施し
/// （`docs/perf/logs/cpu-matmul-fixed-cost-1301/`）、事前宣言した判定
/// 規則（`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20.1）の
/// うち規則 4（candle 比の非後退）が実測後の緩和なしでは 6 セル中 3
/// セルで不成立だったため、いったんは緩和後の基準で `2 << 20` へ
/// 有効化していたが、PR #1448 の codex-review 指摘（計測後に緩和した
/// 基準だけで本番採用を確定しない）を受けて **`usize::MAX` へ差し戻した**。
/// §20.3 の実測系列自体は改善方向の参考値として維持しつつ、規則 4 の
/// 改定版（§20.1a。同 doc の事前登録版として以後固定）を用いた
/// **独立の再計測**（イシュー #1481・同 doc §20.7）が両実機で完了し、
/// §20.1 規則 1〜3・5＋§20.1a 改定版規則 4 を計測後の緩和なしで機械
/// 適用した結果 **verdict=REJECT** と確定した（規則 2: DGX N=2048 の
/// `alloc_c` が on/off で 2.1199 倍に増加〈削減ではない〉／規則 3:
/// 対照セル 8 中 3 が 1.05 超過／規則 4: M4 Max N=512 が 0.9060 <
/// 0.9524）。したがって `usize::MAX`（無効化）を**確定既定**とする
/// （イシュー #1482）。再検討は同一の事前登録規則を機械適用する将来の
/// 再計測（正式系列の新ピン更新時等）に限る（同 doc §20.6・§20.7・
/// `docs/cpu-matmul-fixed-cost-design.md` §10 参照）。
pub(crate) const GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS: usize = usize::MAX;

/// [`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS`] 以上の並列ゼロ書き込みにおける
/// rayon チャンク粒度（要素数。イシュー #1299）。`with_min_len` へ渡し
/// 過分割（タスク生成オーバーヘッドがゼロ書き込み自体を上回る）を防ぐ。
/// 256 KiB 分（`1 << 16` 要素 × 4 byte）はページ粒度（4〜16 KiB）より
/// 十分大きく、`gemm_blis` 側の並列粒度（`gemm_blis/mod.rs` の行パネル
/// 分割）と同じオーダーの経験則値。
pub(crate) const GEMM_OUTPUT_PARALLEL_ZERO_MIN_CHUNK_ELEMS: usize = 1 << 16;

/// `CpuBackendOps::gemm` の出力バッファ確保（イシュー #1299。設計は
/// `docs/cpu-matmul-fixed-cost-design.md` §3.C 案 1a）。
/// [`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS`] しきい値で分岐する:
///
/// - 未満: `vec![0.0f32; len]`（従来どおり。calloc 相当の逐次ゼロ確保）
/// - 以上: `Vec::with_capacity(len)`（非ゼロ確保）へ rayon で並列に
///   `0.0f32` を書き込む（`RepeatN` を `with_min_len` で粗くチャンク
///   分割し `collect_into_vec` で埋める）。ゼロ書き込みを複数スレッドへ
///   分散することで、単一スレッドが `len` 全体を触る（calloc の遅延
///   ページフォールト解決も含め）経路より `alloc_c` 区間の実測時間を
///   縮める狙い（§3.C の仮説。機構検証〈page fault 計数〉は #1301 へ
///   引き継ぎ）
///
/// どちらの分岐も返す `Vec<f32>` は「長さ `len`・全要素 `0.0f32`」で
/// 意味的に同一（`zeroed_output_tests` で全要素 `to_bits() == 0` を
/// 固定）。`gemm_blis_parallel` 系カーネルは出力へ蓄積するだけ
/// （読み出さない）契約のため、ゼロ書き込みの順序・並列度は後続の
/// `C = A @ B` の bit 完全一致に影響しない
/// （`tests/gemm_output_alloc_bit_exact.rs` で回帰確認）。`unsafe` は
/// 使わない（`set_len`／`alloc_zeroed` 直呼び等は設計上不採用。
/// `docs/cpu-matmul-fixed-cost-design.md` §3.C・ユーザー承認事項）。
pub(crate) fn zeroed_output_with_threshold(len: usize, min_elems: usize) -> Vec<f32> {
    if len < min_elems {
        return vec![0.0f32; len];
    }
    use rayon::iter::{IndexedParallelIterator, repeat_n};
    let mut out = Vec::with_capacity(len);
    repeat_n(0.0f32, len)
        .with_min_len(GEMM_OUTPUT_PARALLEL_ZERO_MIN_CHUNK_ELEMS)
        .collect_into_vec(&mut out);
    out
}

/// [`zeroed_output_with_threshold`] を本番既定しきい値
/// （[`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS`]）で呼ぶ薄いラッパー
/// （`CpuBackendOps::gemm` の唯一の呼び出し口。イシュー #1299）。
pub(crate) fn zeroed_output(len: usize) -> Vec<f32> {
    zeroed_output_with_threshold(len, GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS)
}

/// `gemm`／`gemm_fp32_strict_into` 共通の転置判定付きディスパッチ
/// （イシュー #1213 の NT/TN 入口を #1212 の staging 直接書き込み経路
/// でも共有する。codex-review P2・PR #1224）。`out` は呼び出し元が
/// ゼロ初期化済みの `m*n` スライス（累積カーネル契約。`gemm` doc 参照）。
/// `a`／`b` の片側が `dense_transposed_view` で判定できる転置格納なら
/// `contiguous()` の再パックコピーを経由せず [`gemm_blis_parallel_tn`]／
/// [`gemm_blis_parallel_nt`] へ渡し、それ以外は従来の
/// [`gemm_blis_parallel`] へフォールバックする（再パック回数は
/// `GEMM_HOST_REPACK_COUNT` へ計上）。`ctx` はエラーメッセージ先頭の
/// 呼び出し元名。
fn gemm_into_slice(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ctx: &str,
) -> Result<(), BackendError> {
    match (dense_transposed_view(a), dense_transposed_view(b)) {
        (Some(at), None) => {
            // TN: a は転置格納（at: 論理形状 [k,m] 行優先）、b は通常。
            if !b.is_contiguous() {
                GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
            }
            let b_owned = b.contiguous();
            let b_slice = b_owned.as_slice().ok_or_else(|| {
                gemm_contiguity_fail_safe(format!("{ctx}: rhs not contiguous after contiguous()"))
            })?;
            gemm_blis_parallel_tn(at, b_slice, out, m, n, k)
                .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        }
        (None, Some(bt)) => {
            // NT: b は転置格納（bt: 論理形状 [n,k] 行優先）、a は通常。
            if !a.is_contiguous() {
                GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
            }
            let a_owned = a.contiguous();
            let a_slice = a_owned.as_slice().ok_or_else(|| {
                gemm_contiguity_fail_safe(format!("{ctx}: lhs not contiguous after contiguous()"))
            })?;
            gemm_blis_parallel_nt(a_slice, bt, out, m, n, k)
                .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        }
        _ => {
            // TT（両方転置）・判定不能（一般 stride・broadcast 等）:
            // 従来どおり両オペランドを contiguous() で実体化する
            // （`Tensor::as_slice` は非 contiguous では `None` を返す
            // 契約。`crates/tensor-core/src/tensor.rs` 参照）。
            if !a.is_contiguous() {
                GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
            }
            if !b.is_contiguous() {
                GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
            }
            let a_owned = a.contiguous();
            let b_owned = b.contiguous();
            let a_slice = a_owned.as_slice().ok_or_else(|| {
                gemm_contiguity_fail_safe(format!("{ctx}: lhs not contiguous after contiguous()"))
            })?;
            let b_slice = b_owned.as_slice().ok_or_else(|| {
                gemm_contiguity_fail_safe(format!("{ctx}: rhs not contiguous after contiguous()"))
            })?;
            gemm_blis_parallel(a_slice, b_slice, out, m, n, k)
                .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        }
    }
    Ok(())
}

/// [`MemoryOps`] の CPU 実装（イシュー #935）。`shared_cpu_memory()`
/// （プロセスワイド共有 `CpuMemory`）へ委譲する薄いラッパー。
impl MemoryOps for CpuBackendOps {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        shared_cpu_memory().alloc_zeroed(shape)
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        shared_cpu_memory().upload(tensor)
    }

    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        shared_cpu_memory().download(buffer)
    }

    /// [`MemoryOps::upload_into`] の CPU 実装（イシュー #1212）。CPU の
    /// 「デバイス」はホストメモリそのものであるため、実データは
    /// `contiguous()` した `tensor` を `dst` の `CpuBufferHandle::data`
    /// 該当範囲へ `copy_from_slice` するだけで完結する（FFI・実転送
    /// なし。`upload` の CPU 実装と同じ位置付け）。
    fn upload_into(
        &self,
        tensor: &Tensor<f32>,
        dst: &mut DeviceBuffer<f32>,
        dst_offset: usize,
    ) -> Result<(), BackendError> {
        upload_into_cpu_buffer(tensor, dst, dst_offset)
    }

    /// [`MemoryOps::with_host_view`] の CPU 実装（イシュー #1335
    /// codex-review P2 指摘）。`shared_cpu_memory()` へ委譲しないまま
    /// 既定実装（`download` 経由のコピー）へフォールバックすると、
    /// `crate::memory::CpuMemory::with_host_view`（イシュー #1335）が
    /// 実装したコピーなし借用の効果が `CpuBackendOps` 経由の呼び出し
    /// （`facade`／`autodiff` からの実到達経路）では失われるため、他の
    /// メソッドと同じ委譲パターンで明示的に転送する。
    fn with_host_view(
        &self,
        buffer: &DeviceBuffer<f32>,
        f: &mut dyn FnMut(&[f32]),
    ) -> Result<(), BackendError> {
        shared_cpu_memory().with_host_view(buffer, f)
    }
}

/// [`MemoryOps::upload_into`] の CPU 実装本体。`CpuBackendOps`・
/// `crate::memory::CpuMemory` の両実装（本ファイルと `memory.rs`）が
/// 同一ロジックを共有する（`impl MemoryOps for CpuBackendOps` doc
/// 「ホットパス」参照。契約は完全に同一のため 1 箇所にまとめる）。
///
/// 範囲検査（REQ-8「カーネル側の手動境界チェックを省略しない」・
/// OWASP A03）: `dst_offset + tensor.numel()` を `checked_add` で検査し、
/// `dst.numel()` を超える場合は書き込み前に `InvalidArgument` で拒否する。
pub(crate) fn upload_into_cpu_buffer(
    tensor: &Tensor<f32>,
    dst: &mut DeviceBuffer<f32>,
    dst_offset: usize,
) -> Result<(), BackendError> {
    if dst.device() != Device::Cpu {
        return Err(BackendError::DeviceMismatch);
    }
    let contiguous = tensor.contiguous();
    let src = contiguous.as_slice().ok_or_else(|| {
        gemm_contiguity_fail_safe("upload_into: tensor not contiguous after contiguous()")
    })?;
    let numel = src.len();
    let end = dst_offset.checked_add(numel).ok_or_else(|| {
        BackendError::InvalidArgument(
            "upload_into: dst_offset + tensor.numel() overflowed usize".to_string(),
        )
    })?;
    if end > dst.numel() {
        return Err(BackendError::InvalidArgument(format!(
            "upload_into: write range [{dst_offset}, {end}) exceeds dst buffer length {}",
            dst.numel()
        )));
    }
    let handle = dst
        .downcast_handle_mut::<CpuBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)?;
    handle.data[dst_offset..end].copy_from_slice(src);
    Ok(())
}

/// `BackendOps::gemm_resident_rhs`／`gemm_resident_rhs_act` の共有本体
/// （イシュー #1044）。両メソッドとも shape 検証・`DeviceBufferView`
/// のゼロコピー読み出し（`downcast_handle` 直読み + オフセット範囲
/// スライス）は同一で、epilogue の `act` のみが異なるため 1 箇所に
/// まとめる（`gemm` 本体を 2 か所に複製しない）。
fn gemm_resident_rhs_impl(
    a: &Tensor<f32>,
    w: DeviceBufferView<'_>,
    bias: Option<DeviceBufferView<'_>>,
    act: Activation,
) -> Result<Tensor<f32>, BackendError> {
    if w.device() != Device::Cpu {
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
    let w_handle = w
        .buffer()
        .downcast_handle::<CpuBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)?;
    let w_slice = &w_handle.data[w.offset()..w.offset() + w.numel()];

    let bias_handle = match bias {
        Some(b) => {
            if b.device() != Device::Cpu {
                return Err(BackendError::DeviceMismatch);
            }
            if b.shape() != [n] {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![n],
                }));
            }
            let handle = b
                .buffer()
                .downcast_handle::<CpuBufferHandle>()
                .ok_or(BackendError::DeviceMismatch)?;
            Some(&handle.data[b.offset()..b.offset() + b.numel()])
        }
        None => None,
    };

    let a_owned = a.contiguous();
    let a_slice = a_owned.as_slice().ok_or_else(|| {
        gemm_contiguity_fail_safe("gemm_resident_rhs: lhs not contiguous after contiguous()")
    })?;

    let mut out = vec![0.0f32; m * n];
    gemm_blis_bias_act_parallel(a_slice, w_slice, &mut out, m, n, k, bias_handle, act)
        .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
    Tensor::new(out, &[m, n]).map_err(BackendError::ShapeMismatch)
}

impl BackendOps for CpuBackendOps {
    fn device(&self) -> Device {
        Device::Cpu
    }

    /// `CpuBackendOps` 自身が [`MemoryOps`] を実装する（上記 `impl
    /// MemoryOps for CpuBackendOps`）ため、`self` をそのまま返す
    /// （イシュー #935）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        Some(self)
    }

    /// SGD の 1 パラメータ分の更新を in-place で実行する（イシュー #935・
    /// `docs/device-resident-update-design.md` §3.2・§5.2）。CPU は
    /// 「デバイス」がホストメモリそのものであるため、`downcast_handle_mut`
    /// で取り出した `Vec<f32>` を直接書き換えるだけで完結する（転送コスト
    /// ゼロ）。
    ///
    /// 更新式の項順序は `fandhe_ai_autodiff::optim::sgd::Sgd::step`（ホスト
    /// 参照実装）と同一（weight_decay → momentum〈`is_first_step` で
    /// `b ← g` 分岐〉→ nesterov → 減算）。丸えは `.claude/rules/
    /// coding-rust.md` の FMA 契約統一方針に従い `f32::mul_add` を使う
    /// （GEMM 系 CPU 参照実装と同じく、CUDA `fmaf`／Metal
    /// `fma`〈`shaders/sgd.metal`〉と丸めを揃えるため。`Sgd::step` 自身は
    /// PyTorch 参照 fixture との parity を優先し `mul_add` を使わない別の
    /// 契約を持つ〈`sgd.rs` 該当コメント参照〉が、本メソッドは 3
    /// バックエンド間一致が目的のため対象が異なる）。
    fn sgd_step_device(
        &self,
        param: &mut DeviceBuffer<f32>,
        grad: &DeviceBuffer<f32>,
        velocity: Option<&mut DeviceBuffer<f32>>,
        config: &SgdStepConfig,
    ) -> Result<(), BackendError> {
        if param.device() != Device::Cpu || grad.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        if param.shape() != grad.shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: param.shape().to_vec(),
                rhs: grad.shape().to_vec(),
            }));
        }
        let grad_handle = grad
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;

        let use_momentum = config.momentum != 0.0;
        let mut velocity_handle = match velocity {
            Some(v) => {
                if v.device() != Device::Cpu {
                    return Err(BackendError::DeviceMismatch);
                }
                if v.shape() != param.shape() {
                    return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                        lhs: param.shape().to_vec(),
                        rhs: v.shape().to_vec(),
                    }));
                }
                Some(
                    v.downcast_handle_mut::<CpuBufferHandle>()
                        .ok_or(BackendError::DeviceMismatch)?,
                )
            }
            None => {
                if use_momentum {
                    return Err(BackendError::Unsupported(
                        "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
                    ));
                }
                None
            }
        };
        let param_handle = param
            .downcast_handle_mut::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;

        for j in 0..param_handle.data.len() {
            let p = param_handle.data[j];
            let mut g = grad_handle.data[j];
            if config.weight_decay != 0.0 {
                g = config.weight_decay.mul_add(p, g);
            }
            if use_momentum {
                // 直前の分岐（`velocity.is_none() && use_momentum` の
                // 早期 return）により、ここへ到達する時点で
                // `velocity_handle` は必ず `Some` である。
                let Some(velocity_handle) = velocity_handle.as_deref_mut() else {
                    return Err(BackendError::Unsupported(
                        "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
                    ));
                };
                let prev = velocity_handle.data[j];
                let b = if config.is_first_step {
                    g
                } else {
                    config.momentum.mul_add(prev, (1.0 - config.dampening) * g)
                };
                velocity_handle.data[j] = b;
                g = if config.nesterov {
                    config.momentum.mul_add(b, g)
                } else {
                    b
                };
            }
            param_handle.data[j] = p - config.lr * g;
        }
        Ok(())
    }

    /// VJP 専用 NT/TN 2 パターン入口（イシュー #1213）: `matmul_vjp` の
    /// d_input（`g @ Wᵀ`）・d_weight（`Aᵀ @ g`）が渡す片側転置オペランド
    /// （`transpose2d` の zero-copy view）を `dense_transposed_view` で
    /// 判定できる場合、`Tensor::contiguous()` の再パックコピーを経由せず
    /// [`gemm_blis_parallel_nt`]／[`gemm_blis_parallel_tn`]（BLIS packing
    /// が転置格納から直接吸収する）へ渡す。両方転置（TT）・一般 stride
    /// （`narrow` 後の転置等）は判定失敗として従来の `contiguous()` 経路
    /// （[`gemm_blis_parallel`]）へフォールバックする（`docs/matmul-vjp-
    /// zero-copy-decision.md` §3.2。一般 stride 化は本イシューのスコープ
    /// 外）。フォールバックでオペランドを再パックした回数は
    /// `GEMM_HOST_REPACK_COUNT` へ計上する（可観測点。`backend-metal::
    /// ops::upload_operand_for_resident_gemm` と同型の設計）。
    ///
    /// 出力バッファの確保は `zeroed_output`（イシュー #1299）へ委譲する。
    /// `m*n` がしきい値以上なら rayon 並列ゼロ書き込みへ切り替わるが、
    /// カーネル呼び出し契約・累積セマンティクスは不変（`gemm` 自体は
    /// 変更なし）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let mut out = zeroed_output(m * n);
        gemm_into_slice(a, b, &mut out, m, n, k, "gemm")?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_checksum`] の CPU 実装
    /// （イシュー #1339）。CPU はデバイス常駐バッファを持たず `gemm` 自体
    /// が転送コストを発生させないため、本実装は「GPU バックエンドと同じ
    /// API 面を満たす」意味論的対称の位置づけに留まる（実測上の
    /// 読み戻し削減効果は CUDA／Metal 側が主眼）。
    ///
    /// `C` は [`Self::gemm`]（`gemm_into_slice`）と bit 同一（同一
    /// カーネル呼び出し）。checksum は `C` を先頭から `f64` へ昇格して
    /// 逐次和で求める（`out.iter().map(|&x| x as f64).sum()`。固定順序で
    /// 決定的）。この順序は framework-compare の既存ハーネス側 checksum
    /// 実装（`bench-common` の `checksum_f64`）と同一であり、off/on の
    /// 複合判定で完全一致することを期待する（`docs/perf/device-
    /// checksum-readback-ab.md` §5）。
    fn gemm_checksum(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        readout: ChecksumReadout,
    ) -> Result<GemmChecksum, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let mut out = vec![0.0f32; m * n];
        gemm_into_slice(a, b, &mut out, m, n, k, "gemm_checksum")?;
        let checksum: f64 = out.iter().map(|&x| x as f64).sum();
        let output = match readout {
            ChecksumReadout::ChecksumOnly => None,
            ChecksumReadout::WithOutput => {
                Some(Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)?)
            }
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "gemm_checksum: unsupported ChecksumReadout variant {readout:?}"
                )));
            }
        };
        Ok(GemmChecksum { checksum, output })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_fp32_strict_into`] の CPU
    /// 実装（イシュー #1212）。`gemm` と同じ [`gemm_blis_parallel`] を
    /// 使うため数値は `gemm`/`gemm_fp32_strict` と bit 同一だが、結果を
    /// 新規 `Tensor` として返さず `out` の `out_offset` から直接書き込む
    /// （`DeviceParamStore` の grad staging バッファへ、d_weight の D2H を
    /// 経由せず書き込むための入口。TF32 の概念を持たない CPU は元々
    /// `gemm_fp32_strict` と `gemm` が同一実装のため区別を要さない）。
    ///
    /// **累積ではなく上書き契約**: [`gemm_blis_parallel`] は C
    /// （`out[out_offset..out_offset+m*n]`）へ **FMA で累積**する
    /// カーネル（`gemm_blis::mod` の `dispatch_region` doc「累積計算」）
    /// であり、呼び出し元が確保した `Vec` を毎回ゼロ初期化してから渡す
    /// ことで実質的な代入契約を保っている（`gemm` 参照）。`out` は
    /// `DeviceParamStore` が使い回す**永続バッファ**（前ステップの残留値
    /// を保持しうる）ため、`gemm` と異なりここで明示的に対象範囲を
    /// `fill(0.0)` してから同じカーネルへ渡す（トレイト契約「上書き」を
    /// 満たすための CPU 側の対処。CUDA/Metal のカーネルは C を代入で
    /// 書くため対応不要。`docs/device-resident-update-design.md` 追補
    /// 参照）。
    fn gemm_fp32_strict_into(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
    ) -> Result<(), BackendError> {
        if out.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        // REQ-8「カーネル側の手動境界チェックを省略しない」・OWASP A03:
        // `out_offset + m*n` を `checked_mul`/`checked_add` で検査し、
        // `out.numel()` を超える書き込みを事前に拒否する（カーネル起動
        // 前・`out` への可変借用取得前）。
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

        let handle = out
            .downcast_handle_mut::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let dst = &mut handle.data[out_offset..end];
        // 上書き契約（doc 参照）: 永続バッファの残留値を消してから
        // 累積カーネルへ渡す。転置オペランド（`LinearResident` の
        // d_weight が渡す `x_t = transpose2d(x)`）は `gemm` と同じ
        // NT/TN 判定を共有し、`contiguous()` の転置コピーを経由しない
        // （codex-review P2・PR #1224）。
        dst.fill(0.0);
        gemm_into_slice(a, b, dst, m, n, k, "gemm_fp32_strict_into")?;
        let _ = out_shape; // shape 検証のみに使用（`matmul_out_shape` の失敗検出）
        Ok(())
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、CPU カーネル内で epilogue を融合
    /// する [`gemm_blis_bias_act_parallel`] へ差し替える（TASK-12.1f・
    /// #203）。CUDA は同型のオーバーライド（`backend-cuda::ops::
    /// CudaBackendOps::gemm_bias_act`）をイシュー #599 で追加済み。Metal
    /// はこのオーバーライドを持たずデフォルト実装（非融合合成）を使う
    /// （elementwise 未実装により `bias`／`act` 指定時は `Unsupported` を
    /// 透過的に返す。モジュールドキュメント冒頭・`fandhe_ai_tensor_core::
    /// backend_ops` のコメント参照）。
    ///
    /// 融合カーネル（[`gemm_blis_bias_act_parallel`]）は bias の行方向
    /// 複製（shape が厳密に `[n]`）のみ対応する。`bias.shape() == [1]` の
    /// ようなブロードキャスト可能だが `[n]` ちょうどでない shape は、
    /// デフォルト実装と同じ `gemm` → `add`（NumPy 互換ブロードキャスト。
    /// `crate::elementwise::add` 経由）→ act の非融合パスへフォールバック
    /// する。こうしないと `BackendOps::gemm_bias_act` の同一メソッドが
    /// CPU では拒否し CUDA／Metal のデフォルト実装では成功するという
    /// バックエンド依存の挙動差が生じる（Issue #203 Review 指摘）。
    /// 非融合パスへ落ちる場合も `gemm_blis_bias_act_parallel` の
    /// `BiasLenMismatch` 検証（カーネル本体アクセス前に検証。REQ-8・
    /// OWASP A03）と同じ順序契約を保つため、`self.gemm` を実行する前に
    /// `fandhe_ai_tensor_core::broadcast_shape` でブロードキャスト可否のみ先に
    /// 検証する（m×n×k の GEMM 本体を実行してから失敗が判明する、という
    /// 順序にしない）。エラーは `broadcast_shape` のものをそのまま返す
    /// （誤った `ShapeError` variant を独自に組み立てて診断精度を
    /// 落とさない）。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        if let Some(bias) = bias
            && bias.shape() != [n]
        {
            // 融合カーネルの対応範囲外（行方向複製の厳密一致ではない
            // shape）。デフォルト実装と同じ 3 段合成へフォールバックする。
            // GEMM 本体を実行する前にブロードキャスト可否を検証する
            // （REQ-8・OWASP A03。`gemm_blis` の `BiasLenMismatch` と
            // 同じ「カーネル本体アクセス前に検証」の順序契約）。
            fandhe_ai_tensor_core::broadcast_shape(&out_shape, bias.shape())
                .map_err(BackendError::ShapeMismatch)?;
            let mut out = self.gemm(a, b)?;
            out = self.add(&out, bias)?;
            out = match act {
                Activation::None => out,
                Activation::Relu => self.relu(&out)?,
                // `Activation` は `#[non_exhaustive]`（`tensor-core` 側で
                // 将来 variant 追加を見込む）。融合カーネル側
                // （`gemm_blis_bias_act_parallel` 内 `apply_epilogue`）も
                // `_ =>` で未知 variant を静かに無視せず拒否する方針
                // （同ファイル該当コメント参照）と合わせ、ここでも黙って
                // 恒等関数として扱わず明示的に拒否する。
                _ => {
                    return Err(BackendError::Unsupported(format!(
                        "gemm_bias_act: unsupported activation {act:?} in non-fused fallback path"
                    )));
                }
            };
            return Ok(out);
        }

        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm_bias_act: lhs not contiguous after contiguous()")
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm_bias_act: rhs not contiguous after contiguous()")
        })?;

        // ここに到達するのは bias が `None`、または shape が厳密に `[n]`
        // の場合のみ（上の早期リターンで他ケースは処理済み）。
        let bias_owned;
        let bias_slice = match bias {
            Some(bias) => {
                bias_owned = bias.contiguous();
                Some(bias_owned.as_slice().ok_or_else(|| {
                    gemm_contiguity_fail_safe(
                        "gemm_bias_act: bias not contiguous after contiguous()",
                    )
                })?)
            }
            None => None,
        };

        let mut out = vec![0.0f32; m * n];
        gemm_blis_bias_act_parallel(a_slice, b_slice, &mut out, m, n, k, bias_slice, act)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」）。CPU は「デバイス」がホストメモリ
    /// そのもの（`CpuBufferHandle.data: Vec<f32>`）であるため、
    /// `downcast_handle` で連結バッファを直接読み、[`DeviceBufferView`]
    /// の `offset()..offset()+numel()` 範囲をスライスするだけでゼロ
    /// コピーに `gemm_blis_bias_act_parallel` へ渡せる（`sgd_step_device`
    /// と同じ「転送コストゼロ」契約）。`bias` は `[n]`（`w` の列数）への
    /// 厳密一致のみ対応する（`gemm_bias_act` の融合カーネル契約と同じ）。
    /// カーネル本体（`gemm_blis_bias_act_parallel`）へ触れる前に shape を
    /// 検証する（REQ-8・OWASP A03）。範囲自体の検査は
    /// `DeviceBufferView::new` が構築時に済ませているため、ここでは
    /// スライスの長さ（`w.numel()`）が shape 由来の期待値と一致する前提で
    /// 直接インデックスする。
    fn gemm_resident_rhs(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        gemm_resident_rhs_impl(a, w, bias, Activation::None)
    }

    /// [`Self::gemm_resident_rhs`] の activation 融合版（イシュー #1044・
    /// `docs/kernel-fusion.md` §2.2「学習経路への結線」）。`Linear` 層に
    /// 続く `ReLU` を別カーネル起動にせず、bias 加算と同じ epilogue へ
    /// 折り込む（呼び出し元は `fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::linear_forward_with_activation`）。カーネル本体
    /// （`gemm_blis_bias_act_parallel`）は元々 `act` 引数を受け取れる
    /// ため、本メソッドは shape 検証・ゼロコピー読み出しを共有する
    /// `gemm_resident_rhs_impl` へ `act` をそのまま渡すだけ（`gemm_bias_act`
    /// と同じ「常駐版は `Activation::None` 固定、activation 版は `act`
    /// を透過する」非破壊拡張パターン）。
    fn gemm_resident_rhs_act(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        gemm_resident_rhs_impl(a, w, bias, act)
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。`Op::LinearResident` の VJP（`fandhe_ai_autodiff::
    /// grad`）が `d_input^T = w @ g^T` を計算するために使う。[`Self::
    /// gemm_resident_rhs`] と同じくゼロコピー（`downcast_handle` 直読み
    /// + オフセット範囲スライス）。
    ///
    /// `b`（`g^T` に相当。呼び出し元は `Op::LinearResident` d_input）が
    /// `dense_transposed_view` で判定できる dense な転置格納なら
    /// [`gemm_blis_parallel_nt`] へ渡し `contiguous()` の再パックコピー
    /// を経由しない（イシュー #1213）。判定できない場合は従来どおり
    /// `contiguous()` へフォールバックし `GEMM_HOST_REPACK_COUNT` を
    /// 計上する。`w` はデバイス常駐バッファ（`DeviceBufferView`）で
    /// `Tensor` view の転置意味論を持たないため判定対象外（従来どおり
    /// 直読みのみ）。
    fn gemm_resident_lhs(
        &self,
        w: DeviceBufferView<'_>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Cpu {
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
        let w_handle = w
            .buffer()
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let w_slice = &w_handle.data[w.offset()..w.offset() + w.numel()];

        let mut out = vec![0.0f32; p * r];
        if let Some(bt) = dense_transposed_view(b) {
            gemm_blis_parallel_nt(w_slice, bt, &mut out, p, r, q)
                .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        } else {
            if !b.is_contiguous() {
                GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
            }
            let b_owned = b.contiguous();
            let b_slice = b_owned.as_slice().ok_or_else(|| {
                gemm_contiguity_fail_safe(
                    "gemm_resident_lhs: rhs not contiguous after contiguous()",
                )
            })?;
            gemm_blis_parallel(w_slice, b_slice, &mut out, p, r, q)
                .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        }
        Tensor::new(out, &[p, r]).map_err(BackendError::ShapeMismatch)
    }

    /// `a`（デバイス常駐）・`w`（デバイス常駐）・`bias`（デバイス常駐・
    /// 任意）から `y = act(a @ w + bias)` を、入出力ともホストへ実体化
    /// せずに計算する（イシュー #1028・`docs/inference-forward-fixed-
    /// cost-design.md` §3.2）。CPU は「デバイス」がホストメモリその
    /// ものであるため `downcast_handle` の直読みだけでゼロコピーに
    /// なる（`gemm_resident_rhs`／`sgd_step_device` と同じモデル）。
    ///
    /// **bit-exactness 契約**（`docs/inference-forward-fixed-cost-
    /// design.md` §3.3 (b)）: 旧経路（`Sequential::predict` 等が
    /// `tape.ops()` 経由で呼ぶ非融合 `gemm` → `add`（bias 行方向複製）
    /// → `relu` の 3 段合成）と**同一の累積順序**を保つため、本メソッドは
    /// `gemm_bias_act`／`gemm_resident_rhs` が使う融合カーネル
    /// （[`gemm_blis_bias_act_parallel`]。bias／act をカーネル内
    /// epilogue で適用するため tiling 次第で加算順序が変わりうる）を
    /// 使わず、`gemm`（[`gemm_blis_parallel`]）→ bias 行方向複製加算
    /// （単一の `a + b` はグルーピングに依らず IEEE 754 で一意に定まる
    /// ため、ループ構造が異なっても `elementwise::add` と bit-exact）→
    /// `relu`（`max(x, 0.0)`。`elementwise::relu_slice` と同一定義）の
    /// 3 段を明示的に合成する。将来 CPU 側に融合カーネル版を追加する
    /// 場合は、本 doc の bit-exactness 契約ごと見直すこと。
    fn linear_forward_device(
        &self,
        a: &DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Cpu || w.device() != Device::Cpu {
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

        let a_handle = a
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if a_handle.data.len() != a.numel() {
            // `DeviceBuffer::new` 経由で構築される限り到達しないはずだが、
            // shape とハンドル実体のずれを本番経路で `unwrap`/`expect` に
            // 頼らず検出する（REQ-8・OWASP A03 と同種の防御）。
            return Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch {
                    expected: a.numel(),
                    actual: a_handle.data.len(),
                },
            ));
        }
        let a_slice = &a_handle.data[..];

        let w_handle = w
            .buffer()
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let w_slice = &w_handle.data[w.offset()..w.offset() + w.numel()];

        let bias_handle_slice = match bias {
            Some(b) => {
                if b.device() != Device::Cpu {
                    return Err(BackendError::DeviceMismatch);
                }
                if b.shape() != [n] {
                    return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                        lhs: b.shape().to_vec(),
                        rhs: vec![n],
                    }));
                }
                let handle = b
                    .buffer()
                    .downcast_handle::<CpuBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                Some(&handle.data[b.offset()..b.offset() + b.numel()])
            }
            None => None,
        };

        // 1 段目: 非融合 `gemm`（`self.gemm` と同一カーネル）。
        // `m * n` はホスト入力由来の shape 積であり、アロケーション前に
        // オーバーフロー検査を行う（REQ-8・OWASP A03。`reduction.rs` の
        // `ElementCountOverflow` 使用箇所と同一方針）。
        let out_len = m.checked_mul(n).ok_or(BackendError::ShapeMismatch(
            ShapeError::ElementCountOverflow,
        ))?;
        let mut out = vec![0.0f32; out_len];
        gemm_blis_parallel(a_slice, w_slice, &mut out, m, n, k)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;

        // 2 段目: bias の行方向複製加算（`elementwise::add` の broadcast
        // と同じ結果を単一ループで生成。単一の浮動小数点加算はグルー
        // ピングに依らず一意に定まるため bit-exact）。
        if let Some(bias_slice) = bias_handle_slice {
            for i in 0..m {
                let row = &mut out[i * n..(i + 1) * n];
                for (c, b) in row.iter_mut().zip(bias_slice.iter()) {
                    *c += b;
                }
            }
        }

        // 3 段目: activation（`elementwise::relu_slice` と同一定義）。
        match act {
            Activation::None => {}
            Activation::Relu => {
                for v in out.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "linear_forward_device: unsupported activation {act:?}"
                )));
            }
        }

        shared_cpu_memory().wrap_vec(out, vec![m, n])
    }

    /// `a op b`（`op` は [`BinaryElementwiseOp`]）を [`DeviceBuffer`]
    /// 常駐のまま計算する（イシュー #1584）。CPU は「デバイス」が
    /// ホストメモリそのものであるため、H2D／D2H に相当する転送は元々
    /// 発生しない（`linear_forward_device` と同じ位置付け）。`a`・`b`
    /// は shape 完全一致限定（ブロードキャスト非対応。`tensor-core::
    /// BackendOps::binary_elementwise_device` の契約）で、対応する
    /// ホスト版（`Self::add`／`Self::mul`）と同一のスライス関数
    /// （`elementwise::add_slice`／`mul_slice`）を使うため bit 同一。
    fn binary_elementwise_device(
        &self,
        op: BinaryElementwiseOp,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Cpu || b.device() != Device::Cpu {
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

        let a_handle = a
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let b_handle = b
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if a_handle.data.len() != numel || b_handle.data.len() != numel {
            // `linear_forward_device` の同種防御と同じ理由
            // （`DeviceBuffer::new` 経由で構築される限り到達しないはず
            // だが、shape とハンドル実体のずれを本番経路で `unwrap`/
            // `expect` に頼らず検出する。REQ-8・OWASP A03）。
            return Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch {
                    expected: numel,
                    actual: a_handle.data.len().max(b_handle.data.len()),
                },
            ));
        }

        let mut out = vec![0.0f32; numel];
        match op {
            BinaryElementwiseOp::Add => {
                elementwise::add_slice(&a_handle.data, &b_handle.data, &mut out)
            }
            BinaryElementwiseOp::Mul => {
                elementwise::mul_slice(&a_handle.data, &b_handle.data, &mut out)
            }
            // `BinaryElementwiseOp` は `#[non_exhaustive]`。未知 variant
            // は `linear_forward_device` の未知 `Activation` 拒否と同じ
            // 方針で明示的に拒否する（黙って恒等的な値を返さない）。
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "binary_elementwise_device: unsupported op {op:?}"
                )));
            }
        }

        shared_cpu_memory().wrap_vec(out, shape)
    }

    /// [`Self::binary_elementwise_device`] の単項版（イシュー #1584）。
    fn unary_elementwise_device(
        &self,
        op: UnaryElementwiseOp,
        a: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let shape = a.shape().to_vec();
        let numel = a.numel();

        let a_handle = a
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if a_handle.data.len() != numel {
            return Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch {
                    expected: numel,
                    actual: a_handle.data.len(),
                },
            ));
        }

        let mut out = vec![0.0f32; numel];
        match op {
            UnaryElementwiseOp::Relu => elementwise::relu_slice(&a_handle.data, &mut out),
            UnaryElementwiseOp::Exp => elementwise::exp_slice(&a_handle.data, &mut out),
            UnaryElementwiseOp::Tanh => elementwise::tanh_slice(&a_handle.data, &mut out),
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "unary_elementwise_device: unsupported op {op:?}"
                )));
            }
        }

        shared_cpu_memory().wrap_vec(out, shape)
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::add(a, b).map_err(BackendError::ShapeMismatch)
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::mul(a, b).map_err(BackendError::ShapeMismatch)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::relu(a).map_err(BackendError::ShapeMismatch)
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::exp(a).map_err(BackendError::ShapeMismatch)
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::tanh(a).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::where_cond` の CPU 実装（イシュー #1637）。
    /// `elementwise::where_cond`（`PARALLEL_THRESHOLD` による rayon
    /// 自動並列化）へ委譲する（`add`／`mul` と同じ位置づけ）。
    fn where_cond(
        &self,
        cond: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        elementwise::where_cond(cond, a, b).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::masked_fill` の CPU 実装（イシュー #1637）。
    fn masked_fill(
        &self,
        x: &Tensor<f32>,
        mask: &Tensor<f32>,
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        elementwise::masked_fill(x, mask, value).map_err(BackendError::ShapeMismatch)
    }

    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        reduction::sum(a, dim).map_err(reduce_error_to_backend_error)
    }

    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        reduction::max(a, dim).map_err(reduce_error_to_backend_error)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss`] の CPU 実装
    /// （イシュー #1045）。shape 検証・contiguous 化・`mse::
    /// mse_sum_sq_f32` への委譲・`reduction` に応じた最終変換（`Mean`/
    /// `Sum`）・スカラー `Tensor` への詰め直しを行う（`ops.rs` の既存
    /// 方針。モジュール冒頭コメント参照）。
    ///
    /// `reduction` 分岐をここで解決する理由: `MseReduction` は
    /// `#[non_exhaustive]` であり、`f32` を返す `mse::mse_sum_sq_f32`
    /// 側では未知 variant に対する「安全な既定値」が存在しない
    /// （`mse.rs` モジュール doc 参照）。`BackendError` を返せる本メソッド
    /// でのみ、未知 variant を `Unsupported` として型付きに拒否できる。
    fn mse_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_c = pred.contiguous();
        let target_c = target.contiguous();
        // `contiguous()` の戻り値は常に `as_slice()` が `Some` を返す
        // （`Tensor::contiguous` の契約。`tensor.rs`）ため、`unwrap_or`
        // で空スライスへ後退することはない（shape 一致検証済みで両者
        // 同じ要素数のため、以降のスライス長も一致する）。
        let pred_slice = pred_c.as_slice().unwrap_or(&[]);
        let target_slice = target_c.as_slice().unwrap_or(&[]);
        let numel = pred_slice.len();
        let sum_sq = mse::mse_sum_sq_f32(pred_slice, target_slice)?;
        let value = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    0.0
                } else {
                    sum_sq / numel as f32
                }
            }
            MseReduction::Sum => sum_sq,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "mse_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss_backward`] の CPU
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
        let pred_c = pred.contiguous();
        let target_c = target.contiguous();
        let pred_slice = pred_c.as_slice().unwrap_or(&[]);
        let target_slice = target_c.as_slice().unwrap_or(&[]);
        let mut dpred = vec![0.0f32; pred_slice.len()];
        mse::mse_loss_backward_f32(pred_slice, target_slice, scale, &mut dpred)?;
        Tensor::new(dpred, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::rmsnorm`] の CPU 実装
    /// （イシュー #1596）。既存の [`Self::run_fused`] 経由（`match_
    /// rmsnorm_plan` の canonical プラン一致限定・`mean` 化なし・`eps`
    /// なし・`weight` なし）とは別の独立エントリで、`row_norm_layout`
    /// で `(rows, hidden)` を導出してから [`rmsnorm::run_rmsnorm_f32`]
    /// （`mean` 化・`eps`・任意 `weight` を含む標準 RMSNorm）を直接
    /// 呼ぶ。
    fn rmsnorm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("rmsnorm: input not contiguous"))?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice = match &w_owned {
            Some(w) => Some(
                w.as_slice()
                    .ok_or_else(|| gemm_contiguity_fail_safe("rmsnorm: weight not contiguous"))?,
            ),
            None => None,
        };
        let out = rmsnorm::run_rmsnorm_f32(x_slice, w_slice, eps, rows, hidden)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::layer_norm`] の CPU 実装
    /// （イシュー #1596）。[`Self::rmsnorm`] と同じ `row_norm_layout`
    /// 導出だが、`run_fused`（canonical 融合プラン一致経路）への
    /// LayerNorm 一致経路は追加しない——LayerNorm は本エントリ経由でのみ
    /// 到達する（`docs/norm-ops-design.md`）。新設カーネル
    /// [`layer_norm::run_layer_norm_f32`] を直接呼ぶ。
    fn layer_norm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("layer_norm: input not contiguous"))?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice =
            match &w_owned {
                Some(w) => Some(w.as_slice().ok_or_else(|| {
                    gemm_contiguity_fail_safe("layer_norm: weight not contiguous")
                })?),
                None => None,
            };
        let b_owned = bias.map(|b| b.contiguous());
        let b_slice = match &b_owned {
            Some(b) => Some(
                b.as_slice()
                    .ok_or_else(|| gemm_contiguity_fail_safe("layer_norm: bias not contiguous"))?,
            ),
            None => None,
        };
        let out = layer_norm::run_layer_norm_f32(x_slice, w_slice, b_slice, eps, rows, hidden)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_pointwise`] の CPU 実装
    /// （イシュー #1647）。`hidden` は `c_prev` の列数から導出する。
    fn lstm_pointwise(
        &self,
        pre: &Tensor<f32>,
        c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        require_rank2(c_prev.shape())?;
        let hidden = c_prev.shape()[1];
        let b_dim = c_prev.shape()[0];
        // `pre` の rank・shape も検証する（平坦化後の要素数一致だけ
        // では `pre=[4,2]` を `c_prev=[2,1]` に対する `[2,4]` と誤って
        // 受理してしまう。イシュー #1647 codex-review P2 指摘）。
        let gate_width = checked_gate_width(4, hidden)?;
        require_same_shape(pre.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        let pre_c = pre.contiguous();
        let c_prev_c = c_prev.contiguous();
        let pre_slice = pre_c.as_slice().unwrap_or(&[]);
        let c_prev_slice = c_prev_c.as_slice().unwrap_or(&[]);
        let (gates, c, h) = rnn_cell::lstm_pointwise(pre_slice, c_prev_slice, hidden)?;
        Ok(LstmPointwiseOutput {
            gates: Tensor::new(gates, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            c: Tensor::new(c, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
            h: Tensor::new(h, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_hidden_backward`] の
    /// CPU 実装（イシュー #1647）。
    fn lstm_hidden_backward(
        &self,
        c: &Tensor<f32>,
        gate_o: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        require_rank2(c.shape())?;
        require_same_shape(gate_o.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        let shape = c.shape().to_vec();
        let c_c = c.contiguous();
        let gate_o_c = gate_o.contiguous();
        let dh_c = dh.contiguous();
        let (d_pre_o, dc) = rnn_cell::lstm_hidden_backward(
            c_c.as_slice().unwrap_or(&[]),
            gate_o_c.as_slice().unwrap_or(&[]),
            dh_c.as_slice().unwrap_or(&[]),
        )?;
        Ok((
            Tensor::new(d_pre_o, &shape).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dc, &shape).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_cell_backward`] の CPU
    /// 実装（イシュー #1647）。`hidden` は `c_prev` の列数から導出する。
    fn lstm_cell_backward(
        &self,
        gates_ifg: &Tensor<f32>,
        c_prev: &Tensor<f32>,
        dc: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        require_rank2(c_prev.shape())?;
        let hidden = c_prev.shape()[1];
        let b_dim = c_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_ifg.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dc.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        let gates_c = gates_ifg.contiguous();
        let c_prev_c = c_prev.contiguous();
        let dc_c = dc.contiguous();
        let (d_pre_ifg, dc_prev) = rnn_cell::lstm_cell_backward(
            gates_c.as_slice().unwrap_or(&[]),
            c_prev_c.as_slice().unwrap_or(&[]),
            dc_c.as_slice().unwrap_or(&[]),
            hidden,
        )?;
        Ok((
            Tensor::new(d_pre_ifg, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dc_prev, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gru_pointwise`] の CPU 実装
    /// （イシュー #1647）。`hidden` は `h_prev` の列数から導出する。
    fn gru_pointwise(
        &self,
        pre_i: &Tensor<f32>,
        pre_h: &Tensor<f32>,
        h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        require_rank2(h_prev.shape())?;
        let hidden = h_prev.shape()[1];
        let b_dim = h_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(pre_i.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(pre_h.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        let pre_i_c = pre_i.contiguous();
        let pre_h_c = pre_h.contiguous();
        let h_prev_c = h_prev.contiguous();
        let (gates, q, h) = rnn_cell::gru_pointwise(
            pre_i_c.as_slice().unwrap_or(&[]),
            pre_h_c.as_slice().unwrap_or(&[]),
            h_prev_c.as_slice().unwrap_or(&[]),
            hidden,
        )?;
        Ok(GruPointwiseOutput {
            gates: Tensor::new(gates, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            q: Tensor::new(q, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
            h: Tensor::new(h, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gru_backward`] の CPU 実装
    /// （イシュー #1647）。`hidden` は `h_prev` の列数から導出する。
    fn gru_backward(
        &self,
        gates_rzn: &Tensor<f32>,
        q: &Tensor<f32>,
        h_prev: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<GruBackwardOutput, BackendError> {
        require_rank2(h_prev.shape())?;
        let hidden = h_prev.shape()[1];
        let b_dim = h_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_rzn.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(q.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        let gates_c = gates_rzn.contiguous();
        let q_c = q.contiguous();
        let h_prev_c = h_prev.contiguous();
        let dh_c = dh.contiguous();
        let (d_pre_i, d_pre_h, dh_prev_direct) = rnn_cell::gru_backward(
            gates_c.as_slice().unwrap_or(&[]),
            q_c.as_slice().unwrap_or(&[]),
            h_prev_c.as_slice().unwrap_or(&[]),
            dh_c.as_slice().unwrap_or(&[]),
            hidden,
        )?;
        Ok((
            Tensor::new(d_pre_i, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(d_pre_h, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dh_prev_direct, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::softmax`] の CPU 実装
    /// （イシュー #1594）。[`row_softmax_layout`] が非最終軸を `Ok(None)`
    /// で区別する契約に従い、その場合はデフォルトの `Unsupported`
    /// （`Var::softmax` がホスト参照実装 `eval::softmax_along` へ
    /// フォールバックする合図）と同じ挙動を返す。最終軸の場合は
    /// [`softmax::run_softmax_f32`]（`run_fused` の softmax 一致経路
    /// `run_fused_softmax` が使うものと同一カーネル）を直接呼ぶ。
    fn softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        let Some((rows, cols)) =
            row_softmax_layout(x.shape(), dim).map_err(BackendError::ShapeMismatch)?
        else {
            return Err(BackendError::Unsupported(
                "softmax: CPU 行カーネルは最終軸限定（非最終軸はホスト参照実装へ委ねる）".into(),
            ));
        };
        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("softmax: input not contiguous"))?;
        let out = softmax::run_softmax_f32(x_slice, rows, cols)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::log_softmax`] の CPU 実装
    /// （イシュー #1594）。[`Self::softmax`] と同じ最終軸限定契約。
    /// `x − m − ln(Σexp(x − m))` の解析形（[`softmax::
    /// run_log_softmax_f32`]）で計算する（`ln(softmax(x))` にしない
    /// 理由は `BackendOps::log_softmax` doc 参照）。
    fn log_softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        let Some((rows, cols)) =
            row_softmax_layout(x.shape(), dim).map_err(BackendError::ShapeMismatch)?
        else {
            return Err(BackendError::Unsupported(
                "log_softmax: CPU 行カーネルは最終軸限定（非最終軸はホスト参照実装へ委ねる）"
                    .into(),
            ));
        };
        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("log_softmax: input not contiguous"))?;
        let out = softmax::run_log_softmax_f32(x_slice, rows, cols)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::run_fused`] のデフォルト実装（`Unsupported`
    /// fail-safe）を、CPU 単一パス融合カーネル
    /// [`fused_elementwise::run_fused_elementwise`] へ差し替える。
    ///
    /// # 結線ギャップの経緯（#167 実装時に発見）
    /// TASK-12.1 系列は融合 IR（#163・PR #400）と CPU カーネル本体
    /// （#164・PR #403）の双方をマージ済みだったが、#400 は「CPU 融合実行
    /// への結線は #164 のスコープ」、#403 は「`run_fused` オーバーライドの
    /// 提供元は backend-cpu 側（#163 のスコープ）」としており、双方が
    /// 相手に委ねた結果、本オーバーライドが一度も追加されず
    /// `CpuBackendOps` は常にデフォルト `Unsupported` を返して per-op
    /// フォールバックに倒れていた（融合カーネルが実行系上で一度も起動
    /// しない状態）。TASK-12.2a（本イシュー #167）の受け入れ条件は
    /// 「融合効果の実測記録」であり、この状態では融合条件と非融合条件が
    /// 区別不能で実測が構造的に不可能なため、本イシューの前提ステップと
    /// してここで結線する（新規設計を含まない・両先行イシューの設計文書
    /// が明示的に予定していた結線であることが安全側判断の根拠）。
    ///
    /// # 3 分岐ルーティング（イシュー #607 で拡張）
    /// `match_rmsnorm_plan`／`match_softmax_plan`（いずれも純関数。プランの
    /// op 列・leaf 数・`row_fusion()` の形状を厳密照合する。
    /// `backend-cuda::rmsnorm`／`backend-cuda::softmax` と同一契約の CPU
    /// 側ミラー）で canonical RMSNorm／softmax 融合プランを検出した場合は
    /// それぞれ `rmsnorm::run_rmsnorm_f32_raw`／[`softmax::run_softmax_f32`]
    /// へルーティングする。どちらにも一致しない場合（elementwise-only・
    /// 中間軸 softmax 等）は従来どおり [`fused_elementwise::
    /// run_fused_elementwise`] の allowlist 検査へ委ねる（既存 elementwise
    /// 融合経路の挙動は不変）。RMSNorm 判定を先に試す理由は CUDA 側と同じ
    /// （op 列長〈6 vs 8〉が異なるため両方に一致するプランは存在しない）。
    ///
    /// # 呼び出し元
    /// `fandhe_ai_autodiff::tape` の遅延評価 2 層（`materialize_fallible`／
    /// `materialize_non_fallible`）から `BackendOps::run_fused` 経由で
    /// 呼ばれる。CUDA／Metal は独自の融合カーネル実装（#592/#594/#604）を
    /// 持つため本オーバーライドとは独立。
    ///
    /// # 数値契約
    /// `run_fused_elementwise` は per-op 逐次合成と同一スカラー演算
    /// （`f32::mul_add` を用いない単純四則・超越関数）で構成されるため
    /// 数値は不変。REQ-2 複合判定（相対誤差 1e-3 未満 または絶対誤差
    /// 1e-5 未満）での一致は
    /// `crates/backend-cpu/tests/fused_elementwise_parity.rs` で検証済み。
    /// RMSNorm／softmax 経路の数値一致は `tests/rmsnorm_parity.rs`・
    /// `tests/softmax_parity.rs` で検証する。
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
        fused_elementwise::run_fused_elementwise(plan, leaves)
    }

    /// 線形代数（イシュー #1621・`docs/autodiff-linalg-design.md`）。
    /// `linalg` モジュール（本クレート `src/linalg.rs`。`autodiff::eval::
    /// linalg` と同一アルゴリズム・同一符号規約の意図的複製）への薄い
    /// 委譲。`LinalgError`（特異・非正定値・非収束）はいずれも
    /// `BackendError::InvalidArgument` へ変換する（`linalg_error_to_
    /// backend_error` 参照）。
    fn linalg_inv(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        require_square_2d(a.shape(), "linalg_inv")?;
        linalg::inv(a).map_err(linalg_error_to_backend_error)
    }

    fn linalg_solve(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let n = require_square_2d(a.shape(), "linalg_solve")?;
        let b_shape = b.shape();
        if b_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: b_shape.len(),
            }));
        }
        if b_shape[0] != n {
            return Err(BackendError::InvalidArgument(format!(
                "linalg_solve: a の行数 {n} と b の行数 {} が一致しない",
                b_shape[0]
            )));
        }
        linalg::solve(a, b).map_err(linalg_error_to_backend_error)
    }

    fn linalg_det(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        require_square_2d(a.shape(), "linalg_det")?;
        linalg::det(a).map_err(linalg_error_to_backend_error)
    }

    fn linalg_cholesky(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        require_square_2d(a.shape(), "linalg_cholesky")?;
        linalg::cholesky(a).map_err(linalg_error_to_backend_error)
    }

    fn linalg_qr(&self, a: &Tensor<f32>) -> Result<QrFactors, BackendError> {
        require_rank2(a.shape())?;
        let (q, r) = linalg::qr(a).map_err(linalg_error_to_backend_error)?;
        Ok(QrFactors { q, r })
    }

    fn linalg_svd(&self, a: &Tensor<f32>) -> Result<SvdFactors, BackendError> {
        require_rank2(a.shape())?;
        let (u, s, vh) = linalg::svd(a).map_err(linalg_error_to_backend_error)?;
        Ok(SvdFactors { u, s, vh })
    }

    fn linalg_matrix_norm(
        &self,
        a: &Tensor<f32>,
        ord: MatrixNormOrd,
    ) -> Result<Tensor<f32>, BackendError> {
        require_rank2(a.shape())?;
        linalg::matrix_norm(a, ord).map_err(linalg_error_to_backend_error)
    }
}

/// `linalg_*` の公開エントリ（`BackendOps` トレイトメソッド。呼び出し元は
/// `Var::inv` 等の shape 検査済み経路とは限らない——`CpuBackendOps` は
/// `BackendOps` トレイトオブジェクトとして直接呼び出しうる公開 API の
/// ため、rank-1 テンソル等の不正形状が `linalg.rs` 内部の `debug_assert`
/// （呼び出し元検査済みの内部契約）まで素通りして panic するのを防ぐ
/// 境界検査を担う（codex-review 指摘。`.claude/rules/security.md`
/// 「本番経路の panic 禁止」）。`fandhe_ai_autodiff::var::require_square`
/// と同じ判定規律: rank ≠ 2 は `ShapeMismatch(RankMismatch)`（構造的
/// 形状エラー）、非正方は `InvalidArgument`（`ShapeMismatch` の
/// `lhs`/`rhs` は「2 つの shape の不一致」を表す variant のため、単一
/// shape の正方性検査には意味的に合わない）で通知する。成功時は
/// 行数（= 列数）を返す。`op_name` はエラーメッセージに埋め込む呼び出し
/// 元の演算名。
fn require_square_2d(shape: &[usize], op_name: &str) -> Result<usize, BackendError> {
    let n = require_rank2(shape)?;
    if shape[0] != shape[1] {
        return Err(BackendError::InvalidArgument(format!(
            "{op_name}: 正方行列（[n,n]）が必要（形状 {shape:?}）"
        )));
    }
    Ok(n)
}

/// rank-2（`[m,n]`）検査のみ（正方性は要求しない。`qr`／`svd`／
/// `matrix_norm` 用）。上記 [`require_square_2d`] のドキュメント参照。
fn require_rank2(shape: &[usize]) -> Result<usize, BackendError> {
    if shape.len() != 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    Ok(shape[0])
}

/// RNN／LSTM／GRU 系エントリ（`lstm_pointwise`／`lstm_cell_backward`／
/// `gru_pointwise`／`gru_backward`）が形状比較の前に必要とする
/// `gates * hidden`（ゲート幅）を `checked_mul` で検証する。
///
/// 本番経路 panic 禁止（AGENTS.md）: `4 * hidden`／`3 * hidden` を
/// 未検証のまま `require_same_shape` の期待値へ埋め込むと、`hidden`
/// が `usize::MAX` 近傍（例: 要素数 0 の空テンソルなら
/// `c_prev.shape() = [0, 1usize << 62]` のように shape[1] を自由に
/// 取れる）のとき乗算が overflow して期待幅が小さい値へ周回し、
/// 本来 shape mismatch で拒否すべき不正な `pre`／`gates_ifg`
/// を誤って受理してしまう（受理後は `rnn_cell` 側カーネルが
/// `hidden` を使った添字アクセスで範囲外参照する）。イシュー #1647
/// codex-review P1 指摘。呼び出し元は本関数の戻り値をそのまま
/// `require_same_shape` の期待 shape へ使う。
fn checked_gate_width(gates: usize, hidden: usize) -> Result<usize, BackendError> {
    gates.checked_mul(hidden).ok_or(BackendError::ShapeMismatch(
        ShapeError::ElementCountMismatch {
            expected: usize::MAX,
            actual: 0,
        },
    ))
}

impl CpuBackendOps {
    /// [`BackendOps::run_fused`] の RMSNorm 一致経路（イシュー #607）。
    /// `match_rmsnorm_plan` が一致した後の dtype／leaf 数／leaf shape の
    /// 起動前 fail-closed 検証（`backend-cuda::ops::CudaBackendOps::
    /// run_fused_rmsnorm` と同じ検査順序）と、
    /// [`rmsnorm::run_rmsnorm_f32_raw`]（`inv_n = 1.0`・`eps = 0.0`・
    /// `w = None`）への委譲を行う。
    fn run_fused_rmsnorm(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: unsupported dtype {:?} (canonical RMSNorm fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: canonical RMSNorm プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("run_fused: rmsnorm input not contiguous"))?;

        let out = rmsnorm::run_rmsnorm_f32_raw(x_slice, None, 0.0, 1.0, 1, hidden)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::run_fused`] の softmax 一致経路（イシュー #607）。
    /// `run_fused_rmsnorm` と同じ起動前 fail-closed 検証パターンを踏襲し、
    /// [`softmax::run_softmax_f32`] を直接呼ぶ。
    fn run_fused_softmax(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: unsupported dtype {:?} (canonical softmax fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: canonical softmax プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("run_fused: softmax input not contiguous"))?;

        let out = softmax::run_softmax_f32(x_slice, rows, cols)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// [`LinalgError`] を `BackendError::InvalidArgument` へ写像する
/// （`reduce_error_to_backend_error` と同じ「専用 variant を設けず既存
/// `InvalidArgument` に寄せる」方針。`BackendOps::linalg_*` doc の
/// 契約「特異／非正定値／非収束は `InvalidArgument`」に対応する）。
fn linalg_error_to_backend_error(err: LinalgError) -> BackendError {
    BackendError::InvalidArgument(err.to_string())
}

/// `reduction::ReduceError`（`Shape`／`EmptyReduction` の 2 variant）を
/// `BackendError` へ写像する。`EmptyReduction` は shape 由来ではない
/// 実行時失敗のため `KernelLaunchFailed` に寄せる（`BackendError` に
/// reduction 専用 variant は設けない。§4.4 の 5 variant + TASK-1.9a/1.9c
/// 拡張の範囲に収める）。
fn reduce_error_to_backend_error(err: reduction::ReduceError) -> BackendError {
    match err {
        reduction::ReduceError::Shape(shape_err) => BackendError::ShapeMismatch(shape_err),
        reduction::ReduceError::EmptyReduction { op } => {
            BackendError::KernelLaunchFailed(format!("empty reduction for op \"{op}\""))
        }
    }
}

/// [`GEMM_HOST_REPACK_COUNT`]（クレート境界外の統合テストから見えない
/// `pub(crate)` カウンタ）が「dense 転置 view では増加しない・一般
/// stride／TT では増加する」ことを検証するクレート内テスト（イシュー
/// #1213）。数値一致自体は統合テスト `tests/gemm_transposed_parity.rs`
/// が担当し、本テストはフォールバック経路の健全性（NT/TN 判定が実際に
/// 効いていること）のみを確認する。
#[cfg(test)]
mod repack_count_tests {
    use super::*;

    fn reset_counter() {
        GEMM_HOST_REPACK_COUNT.with(|c| c.set(0));
    }

    fn counter() -> u64 {
        GEMM_HOST_REPACK_COUNT.with(|c| c.get())
    }

    #[test]
    fn gemm_dense_transposed_view_does_not_increment_repack_counter() {
        reset_counter();
        let ops = CpuBackendOps::new();
        let g = Tensor::new(vec![1.0f32; 4 * 3], &[4, 3]).unwrap();
        let w = Tensor::new(vec![1.0f32; 5 * 3], &[5, 3]).unwrap();
        let w_t = w.transpose_2d().unwrap();
        let before = counter();
        ops.gemm(&g, &w_t).unwrap();
        assert_eq!(
            counter(),
            before,
            "dense な転置 view（NT）は contiguous() フォールバックを通らないはず"
        );
    }

    /// `gemm_fp32_strict_into`（#1212 の staging 直接書き込み）でも
    /// `LinearResident` の d_weight が渡す転置 lhs（TN）を再パックせず
    /// `gemm` と bit 同一の結果を書くこと（codex-review P2・PR #1224）。
    #[test]
    fn gemm_fp32_strict_into_transposed_lhs_shares_tn_entry() {
        use fandhe_ai_tensor_core::MemoryOps;
        reset_counter();
        let ops = CpuBackendOps::new();
        let mem = crate::memory::CpuMemory::new();
        let (m, k, n) = (4usize, 6usize, 5usize);
        let x = Tensor::new((0..k * m).map(|i| i as f32 * 0.5 - 3.0).collect(), &[k, m]).unwrap();
        let g = Tensor::new((0..k * n).map(|i| (i % 7) as f32 - 2.0).collect(), &[k, n]).unwrap();
        let x_t = x.transpose_2d().unwrap();
        let expected = ops.gemm(&x_t, &g).unwrap();
        let seed = Tensor::new(vec![0.0f32; 2 + m * n], &[2 + m * n]).unwrap();
        let mut staging = mem.upload(&seed).unwrap();
        let before = counter();
        ops.gemm_fp32_strict_into(&x_t, &g, &mut staging, 2)
            .unwrap();
        assert_eq!(
            counter(),
            before,
            "dense な転置 lhs（TN）は再パックを通らないはず"
        );
        let host = mem.download(&staging).unwrap().contiguous();
        assert_eq!(&host.as_slice().unwrap()[2..], expected.as_slice().unwrap());
    }

    #[test]
    fn gemm_narrow_then_transpose_increments_repack_counter() {
        reset_counter();
        let ops = CpuBackendOps::new();
        let g = Tensor::new(vec![1.0f32; 4 * 3], &[4, 3]).unwrap();
        // narrow は先頭次元（行）ではなく末尾次元（列）に対して行う必要
        // がある: 行方向の narrow は offset のみでストライドは
        // row_major_strides のまま変わらず、transpose 後も dense 転置
        // 判定に合致してしまう（実際に正しく NT 経路を通せるため誤りでは
        // ないが、本テストが検証したい「一般 stride で判定に落ちる」
        // ケースにならない）。列方向の narrow は行ストライド（= 元の
        // 列数）が narrow 後の列数より大きくなるため、真に一般 stride
        // （`ld != rows`）になる。
        let w0 = Tensor::new(vec![1.0f32; 5 * 7], &[5, 7]).unwrap();
        let w_narrowed = w0.narrow(1, 1, 3).unwrap();
        let w_t = w_narrowed.transpose_2d().unwrap();
        let before = counter();
        ops.gemm(&g, &w_t).unwrap();
        assert_eq!(
            counter(),
            before + 1,
            "narrow 後の転置（一般 stride）は contiguous() フォールバックを通るはず"
        );
    }

    #[test]
    fn gemm_both_transposed_increments_repack_counter_twice() {
        reset_counter();
        let ops = CpuBackendOps::new();
        let orig_a = Tensor::new(vec![1.0f32; 3 * 4], &[3, 4]).unwrap();
        let a_t = orig_a.transpose_2d().unwrap();
        let orig_b = Tensor::new(vec![1.0f32; 5 * 3], &[5, 3]).unwrap();
        let b_t = orig_b.transpose_2d().unwrap();
        let before = counter();
        ops.gemm(&a_t, &b_t).unwrap();
        assert_eq!(
            counter(),
            before + 2,
            "両方転置（TT）は両オペランドとも contiguous() フォールバックを通るはず"
        );
    }

    /// 本番 NN（転置なし）経路では `GEMM_HOST_REPACK_COUNT` が増えない
    /// ことを固定する回帰（イシュー #1299・設計 §3.A「本番 NN 経路で
    /// contiguous コピーは発生していない」の確認。`docs/cpu-matmul-
    /// fixed-cost-design.md` §3.A は変更なしと結論づけたため、この
    /// 結論をテストで固定し将来の意図しない後退を検知する）。
    #[test]
    fn gemm_nn_does_not_increment_repack_counter() {
        reset_counter();
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32; 4 * 3], &[4, 3]).unwrap();
        let b = Tensor::new(vec![1.0f32; 3 * 5], &[3, 5]).unwrap();
        let before = counter();
        ops.gemm(&a, &b).unwrap();
        assert_eq!(
            counter(),
            before,
            "本番 NN 経路（両オペランドとも転置なし・contiguous）は \
             contiguous() フォールバックを通らないはず"
        );
    }
}

/// [`zeroed_output`]／[`zeroed_output_with_threshold`]（イシュー #1299）
/// の意味論を固定するクレート内テスト。実際の bit 完全一致回帰は
/// `tests/gemm_output_alloc_bit_exact.rs`（統合テスト）が担当し、本
/// モジュールはヘルパー自体の契約（長さ・全要素ゼロ・両分岐・境界値）
/// のみを検証する。
#[cfg(test)]
mod zeroed_output_tests {
    use super::*;

    fn assert_all_zero_bits(v: &[f32], len: usize) {
        assert_eq!(v.len(), len);
        assert!(
            v.iter().all(|x| x.to_bits() == 0),
            "全要素が bit 表現で 0.0f32（符号なしゼロ）であるはず"
        );
    }

    #[test]
    fn below_threshold_uses_sequential_path() {
        let len = 1024;
        let out = zeroed_output_with_threshold(len, 2048);
        assert_all_zero_bits(&out, len);
    }

    #[test]
    fn at_threshold_uses_parallel_path() {
        // len == min_elems ちょうど: 「以上」分岐（並列経路）に入る境界。
        let min_elems = 4096;
        let out = zeroed_output_with_threshold(min_elems, min_elems);
        assert_all_zero_bits(&out, min_elems);
    }

    #[test]
    fn above_threshold_uses_parallel_path() {
        let min_elems = 4096;
        let len = min_elems + 1;
        let out = zeroed_output_with_threshold(len, min_elems);
        assert_all_zero_bits(&out, len);
    }

    #[test]
    fn zero_length_both_branches() {
        assert_all_zero_bits(&zeroed_output_with_threshold(0, 0), 0);
        assert_all_zero_bits(&zeroed_output_with_threshold(0, 1), 0);
    }

    #[test]
    fn default_wrapper_matches_production_threshold() {
        // 本番既定しきい値は `usize::MAX`（並列分岐は常に無効。#1299 の
        // M4 Max スモークが N=2048 で後退を確認したため。PR #1448 の
        // codex-review 指摘を受け、#1301 の実測後に緩和した基準のみを
        // 根拠とする有効化を差し戻した。独立の再計測（`docs/perf/
        // cpu-gemm-candle-gate-remeasurement.md` §20.7・イシュー #1481）
        // で verdict=REJECT が確定し、#1482 で確定既定とした）。
        // したがって `zeroed_output` は現実的なサイズでは常に「未満」
        // 分岐（逐次経路）へ入る。並列分岐自体は
        // `above_threshold_uses_parallel_path`／`at_threshold_uses_parallel_path`
        // が明示的な小さいしきい値を渡して別途カバーする。
        let len = 1usize << 24; // 16M 要素（64 MiB）でも usize::MAX 未満。
        assert_all_zero_bits(&zeroed_output(len), len);
    }

    /// 本番既定しきい値が無効化状態（`usize::MAX`）であることを固定する
    /// 回帰（イシュー #1299・#1301・PR #1448 codex-review 対応で差し戻し。
    /// `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20.7 の独立
    /// 再計測（イシュー #1481）で verdict=REJECT と確定し、#1482 で
    /// 確定既定とした）。
    #[test]
    fn default_threshold_is_disabled_confirmed_by_independent_remeasurement() {
        assert_eq!(
            GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS,
            usize::MAX,
            "#1481 の独立再計測（§20.7）は事前登録した規則 1〜5 を \
             計測後の緩和なしで機械適用した結果 verdict=REJECT と確定した \
             （規則 2: DGX N=2048 alloc_c が 2.1199 倍に増加／規則 3: \
             対照セル 8 中 3 超過／規則 4: M4 Max N=512 が 0.9060 < \
             0.9524）ため本番既定は並列分岐を無効化した状態であるはず。 \
             同一の事前登録規則を機械適用する将来の再計測で ADOPT へ \
             転じた場合のみ本テストの期待値を更新すること"
        );
    }

    /// 並列ゼロ書き込み分岐（本番既定では `usize::MAX` により到達不能。
    /// 明示的にしきい値 0 を渡して強制する）を実際の
    /// GEMM カーネルへ通した結果が、逐次ゼロ確保（`vec![0.0f32; ..]`）
    /// 経由の結果と bit 完全一致することを確認する（イシュー #1299・
    /// codex-review 相当の指摘: 統合テスト
    /// `tests/gemm_output_alloc_bit_exact.rs` は小さい形状が中心で
    /// 並列分岐を実走できないため、クレート内テストで並列分岐自体の
    /// 正しさを別途固定する）。
    #[test]
    fn parallel_branch_output_matches_sequential_branch_through_kernel() {
        let (m, k, n) = (37usize, 65usize, 33usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.25 - 1.0).collect();

        // 並列分岐を強制（min_elems=0 は「常に以上」を意味する）。
        let mut parallel_out = zeroed_output_with_threshold(m * n, 0);
        gemm_blis_parallel(&a, &b, &mut parallel_out, m, n, k)
            .expect("gemm_blis_parallel must succeed (parallel branch)");

        // 逐次分岐を強制（min_elems=usize::MAX は「常に未満」を意味する）。
        let mut sequential_out = zeroed_output_with_threshold(m * n, usize::MAX);
        gemm_blis_parallel(&a, &b, &mut sequential_out, m, n, k)
            .expect("gemm_blis_parallel must succeed (sequential branch)");

        let parallel_bits: Vec<u32> = parallel_out.iter().map(|x| x.to_bits()).collect();
        let sequential_bits: Vec<u32> = sequential_out.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            parallel_bits, sequential_bits,
            "並列ゼロ書き込み分岐を通した GEMM 結果は逐次ゼロ確保分岐と \
             bit 完全一致するはず（カーネル入口で C が全要素ゼロである \
             限りゼロ書き込みの並列度は出力に影響しない契約。設計 §4）"
        );
    }
}

/// `CpuBackendOps::with_host_view`（イシュー #1335 codex-review P2 是正）
/// が `shared_cpu_memory()` へ実際に転送され、既定実装（`download` 経由
/// のコピー）ではなくコピーなし借用の実装（`CpuMemory::with_host_view`）
/// へ到達することを検証する。
#[cfg(test)]
mod with_host_view_forwarding_tests {
    use super::*;

    #[test]
    fn cpu_backend_ops_with_host_view_matches_download_bit_exact() {
        let ops = CpuBackendOps::new();
        let data = vec![1.0f32, -2.5, 3.25, f32::MIN_POSITIVE, f32::MAX];
        let tensor = Tensor::<f32>::new(data.clone(), &[5]).unwrap();
        let buf = ops.upload(&tensor).unwrap();

        let mut observed = Vec::new();
        ops.with_host_view(&buf, &mut |slice| observed = slice.to_vec())
            .unwrap();

        let downloaded = ops.download(&buf).unwrap();
        assert_eq!(
            observed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            downloaded
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "CpuBackendOps::with_host_view は download() と bit 同一のはず"
        );
    }
}

/// [`CpuBackendOps::gemm_checksum`]（イシュー #1339）の回帰テスト。
#[cfg(test)]
mod gemm_checksum_tests {
    use super::*;

    #[test]
    fn gemm_checksum_only_matches_host_f64_sum_of_gemm_and_has_no_output() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let reference = ops.gemm(&a, &b).unwrap();
        let expected: f64 = reference
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| x as f64)
            .sum();

        let result = ops
            .gemm_checksum(&a, &b, ChecksumReadout::ChecksumOnly)
            .unwrap();

        assert_eq!(result.checksum, expected);
        assert!(result.output.is_none());
    }

    #[test]
    fn gemm_checksum_with_output_is_bit_identical_to_gemm() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, -2.5, 3.25, 0.5, -1.0, 2.0], &[2, 3]).unwrap();
        let b = Tensor::new(vec![0.5f32, 1.5, -1.0, 2.0, 3.0, -2.0], &[3, 2]).unwrap();

        let reference = ops.gemm(&a, &b).unwrap();
        let result = ops
            .gemm_checksum(&a, &b, ChecksumReadout::WithOutput)
            .unwrap();

        let output = result.output.expect("WithOutput は output を返すはず");
        assert_eq!(
            output
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            reference
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "gemm_checksum(WithOutput).output は gemm と bit 同一のはず"
        );

        let expected: f64 = reference
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| x as f64)
            .sum();
        assert_eq!(result.checksum, expected);
    }
}

/// `BackendOps::linalg_*`（`CpuBackendOps` の公開エントリ）の shape
/// 検査（codex-review 指摘。P1 #1 の修正回帰）。`Var::inv` 等の
/// shape 検査済み経路を経由しない直接呼び出しでも、rank-1 テンソル等の
/// 不正形状が `linalg.rs` 内部の `debug_assert`（呼び出し元検査済みの
/// 内部契約）まで素通りして panic しないことを確認する。
#[cfg(test)]
mod linalg_shape_validation_tests {
    use super::*;

    #[test]
    fn linalg_inv_rank1_is_shape_mismatch_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let result = ops.linalg_inv(&a);
        assert!(matches!(result, Err(BackendError::ShapeMismatch(_))));
    }

    #[test]
    fn linalg_inv_non_square_is_invalid_argument_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = ops.linalg_inv(&a);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn linalg_solve_row_count_mismatch_is_invalid_argument_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        // b の行数（3）が a の行数（2）と一致しない。
        let b = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3, 1]).unwrap();
        let result = ops.linalg_solve(&a, &b);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn linalg_solve_b_rank_mismatch_is_shape_mismatch_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        let result = ops.linalg_solve(&a, &b);
        assert!(matches!(result, Err(BackendError::ShapeMismatch(_))));
    }

    #[test]
    fn linalg_det_rank1_is_shape_mismatch_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(matches!(
            ops.linalg_det(&a),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn linalg_cholesky_rank1_is_shape_mismatch_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(matches!(
            ops.linalg_cholesky(&a),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn linalg_qr_rank1_is_shape_mismatch_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(matches!(
            ops.linalg_qr(&a),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn linalg_svd_rank1_is_shape_mismatch_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(matches!(
            ops.linalg_svd(&a),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn linalg_matrix_norm_rank1_is_shape_mismatch_not_panic() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(matches!(
            ops.linalg_matrix_norm(&a, MatrixNormOrd::Fro),
            Err(BackendError::ShapeMismatch(_))
        ));
    }
}

/// [`CpuBackendOps::binary_elementwise_device`]／
/// [`CpuBackendOps::unary_elementwise_device`]（イシュー #1584）の回帰
/// テスト。ホスト版（`Self::add`／`mul`／`relu`／`exp`／`tanh`）と bit
/// 同一であること・空 shape・shape 不一致・device 不一致の fail-closed
/// を確認する。
#[cfg(test)]
mod device_resident_elementwise_tests {
    use super::*;

    fn bits(t: &Tensor<f32>) -> Vec<u32> {
        t.as_slice()
            .expect("test tensor must be contiguous")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn bits_buf(ops: &CpuBackendOps, buf: &DeviceBuffer<f32>) -> Vec<u32> {
        let t = ops.download(buf).unwrap();
        bits(&t)
    }

    #[test]
    fn binary_elementwise_device_add_matches_host_add_bit_exact() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, -2.5, 3.25, f32::MIN_POSITIVE], &[4]).unwrap();
        let b = Tensor::new(vec![0.5f32, 1.5, -3.25, 2.0], &[4]).unwrap();

        let a_buf = ops.upload(&a).unwrap();
        let b_buf = ops.upload(&b).unwrap();
        let out_buf = ops
            .binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf)
            .unwrap();

        let host_out = ops.add(&a, &b).unwrap();
        assert_eq!(bits_buf(&ops, &out_buf), bits(&host_out));
    }

    #[test]
    fn binary_elementwise_device_mul_matches_host_mul_bit_exact() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, -2.5, 3.25, 2.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![0.5f32, 1.5, -3.25, 4.0], &[2, 2]).unwrap();

        let a_buf = ops.upload(&a).unwrap();
        let b_buf = ops.upload(&b).unwrap();
        let out_buf = ops
            .binary_elementwise_device(BinaryElementwiseOp::Mul, &a_buf, &b_buf)
            .unwrap();

        let host_out = ops.mul(&a, &b).unwrap();
        assert_eq!(bits_buf(&ops, &out_buf), bits(&host_out));
    }

    #[test]
    fn unary_elementwise_device_relu_matches_host_relu_bit_exact() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![-1.0f32, 0.0, 2.5, -3.25, f32::NAN], &[5]).unwrap();

        let a_buf = ops.upload(&a).unwrap();
        let out_buf = ops
            .unary_elementwise_device(UnaryElementwiseOp::Relu, &a_buf)
            .unwrap();

        let host_out = ops.relu(&a).unwrap();
        assert_eq!(bits_buf(&ops, &out_buf), bits(&host_out));
    }

    #[test]
    fn unary_elementwise_device_exp_matches_host_exp_bit_exact() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![-1.0f32, 0.0, 2.5, -3.25], &[4]).unwrap();

        let a_buf = ops.upload(&a).unwrap();
        let out_buf = ops
            .unary_elementwise_device(UnaryElementwiseOp::Exp, &a_buf)
            .unwrap();

        let host_out = ops.exp(&a).unwrap();
        assert_eq!(bits_buf(&ops, &out_buf), bits(&host_out));
    }

    #[test]
    fn unary_elementwise_device_tanh_matches_host_tanh_bit_exact() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![-1.0f32, 0.0, 2.5, -3.25], &[4]).unwrap();

        let a_buf = ops.upload(&a).unwrap();
        let out_buf = ops
            .unary_elementwise_device(UnaryElementwiseOp::Tanh, &a_buf)
            .unwrap();

        let host_out = ops.tanh(&a).unwrap();
        assert_eq!(bits_buf(&ops, &out_buf), bits(&host_out));
    }

    #[test]
    fn binary_elementwise_device_handles_empty_numel() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();
        let b = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();
        let a_buf = ops.upload(&a).unwrap();
        let b_buf = ops.upload(&b).unwrap();

        let out_buf = ops
            .binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf)
            .unwrap();
        assert_eq!(out_buf.shape(), &[0]);
        assert_eq!(out_buf.numel(), 0);
    }

    #[test]
    fn binary_elementwise_device_rejects_shape_mismatch() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        let b = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let a_buf = ops.upload(&a).unwrap();
        let b_buf = ops.upload(&b).unwrap();

        let result = ops.binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf);
        assert!(matches!(result, Err(BackendError::ShapeMismatch(_))));
    }

    #[test]
    fn binary_elementwise_device_rejects_device_mismatch() {
        // `Device::Cuda(0)` を名乗る CPU 由来ではないバッファを渡すと
        // `downcast_handle` 前の device 検査で拒否される（`CudaBufferHandle`
        // を構築せずとも `DeviceBuffer::device()` の値だけで判定される
        // ため、CPU クレート内で到達可能な検査）。
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        let a_buf = ops.upload(&a).unwrap();
        let mismatched = DeviceBuffer::new(Device::Cuda(0), vec![2], Box::new(NotCpuHandle));

        let result = ops.binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &mismatched);
        assert!(matches!(result, Err(BackendError::DeviceMismatch)));
    }

    /// [`binary_elementwise_device_rejects_device_mismatch`] 専用のダミー
    /// ハンドル（`CpuBufferHandle` 以外の任意型であればよい。値自体は
    /// 使われない）。
    #[derive(Debug)]
    struct NotCpuHandle;

    impl fandhe_ai_tensor_core::buffer::BufferHandle for NotCpuHandle {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }
}
