//! カーネルディスパッチ機構（TASK-1.9c・#46）。
//!
//! 単一の計算記述（[`BackendOps`] を受け取る関数）から CPU／CUDA／Metal
//! いずれのバックエンドのカーネルへも呼び分けられるようにする入口。
//! `device`（TASK-1.9a・#44）と同じ依存逆転構成を踏襲する: trait 定義を
//! 3 バックエンドクレートが依存できる本クレートに置き、各バックエンド
//! クレート（`backend-cpu`／`backend-cuda`／`backend-metal`）側で実装する
//! （`tensor-core` → `backend-*` の逆依存は作らない）。
//!
//! シグネチャは `docs/public-api-design.md` §4.2 の `BackendOps` trait案を
//! 正本としつつ、以下の点で拡張・簡略化している（同文書「TASK-1.9 実装
//! イシューで本文書との突合を行うこと」に対応。突合結果は同文書にも
//! 注記する）:
//!
//! - **`DeviceBuffer`／`upload`／`download` を含めない**。§4.2 が示す
//!   デバイス常駐バッファ型・転送 API は TASK-1.9b（#45）の担当であり、
//!   本イシュー時点で `tensor-core`・3 バックエンドクレートいずれにも
//!   存在しない（実装開始時に `git fetch origin main` で確認済み）。
//!   本イシューの受け入れ条件は「同一コードで 3 バックエンドのカーネルが
//!   呼び分けられる」（機構的な呼び分け）であり、既存カーネル入口
//!   （CPU `gemm_blis_parallel`・CUDA `CudaGemm::run_tiled_f32`・Metal
//!   `MetalGemm::dispatch_auto`）がいずれもホスト常駐 `&[f32]` を受け取り
//!   内部で H2D／D2H 転送を完結させる契約であるため、`DeviceBuffer` なしで
//!   本受け入れ条件を満たせる。§4.2 の `DeviceBuffer` 版シグネチャへの
//!   移行（`upload`／`download` の追加）は #45 のマージ後、`BackendOps` の
//!   非破壊拡張（デフォルトメソッド追加等）として TASK-1.9d（#47）以降で
//!   検討する
//! - 各メソッドはホスト常駐 [`Tensor<f32>`](crate::Tensor) を受け取り
//!   [`Tensor<f32>`](crate::Tensor) を返す（§4.2 の `DeviceBuffer<f32>` を
//!   `Tensor<f32>` に読み替えた形）。CPU 実装は転送コストが発生しないため
//!   このままで問題なく、CUDA／Metal 実装は各メソッド内で
//!   `Tensor::as_slice` → カーネル呼び出し（内部で H2D／D2H）→
//!   `Tensor::new` で完結させる
//! - 未実装カーネル（CUDA／Metal の elementwise・reduction。TASK-1.9c 時点
//!   では両バックエンドとも GEMM カーネルのみ実装済み）は
//!   [`crate::device::BackendError::Unsupported`]（本イシューで追加した
//!   非破壊拡張 variant）を返す fail-safe 実装とする。GPU 側
//!   elementwise・reduction カーネルの実装自体は本イシューのスコープ外
//!   （out-of-scope-tracking.md 対象。引き継ぎ先はユーザー承認を得て別
//!   Issue で追跡する）
//!
//! ディスパッチ規則（形状・HW 判定による経路選択）は TASK-11.2b（#68）の
//! 担当でありスコープ外（`docs/dispatch-rules-design.md`。TASK-11.2a・
//! #67）。既定デバイス選択ロジック（CUDA 既定有効化の構成決定含む）も
//! ユーザー承認必須のためスコープ外（`device` モジュールと同方針）。
//! 3 バックエンド横断の統合テストは TASK-1.9d（#47）が本格的に担当し、
//! 本イシューは受け入れ条件検証に必要な最小限のテストに留める。

use crate::Tensor;
use crate::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use crate::device::{BackendError, Device};
use crate::dispatch_failure::DispatchFailureCell;
use crate::fusion::FusionPlan;
use crate::pool_core::PoolStats;

/// [`BackendOps::sgd_step_device`] の 1 ステップ分のハイパーパラメータ
/// （イシュー #935・`docs/device-resident-update-design.md` §3.1）。
///
/// `fandhe_ai_autodiff::optim::sgd::SgdConfig`（ホスト参照実装。`lr`／
/// `momentum`／`dampening`／`weight_decay`／`nesterov` の 5 フィールド）と
/// 同じ意味論のフィールドに `is_first_step` を加えたもの。`autodiff`
/// クレートは `tensor-core` へ依存する側（`tensor-core` → `autodiff` の
/// 逆依存は作らない）であるため、`SgdConfig` をここへ再エクスポートせず
/// 独立した型として定義する（`fandhe_ai_autodiff::optim::device_store::
/// DeviceParamStore::step` が `SgdConfig` から本型へ変換して渡す）。
///
/// `is_first_step`: PyTorch `torch.optim.SGD` の momentum 初期化規則
/// （`docs/spec` 由来。`fandhe_ai_autodiff::optim::sgd` モジュールコメント
/// 「Algorithm」節）は「初回 step は `b ← g`、2 回目以降は
/// `b ← μ·b + (1−τ)·g`」であり、この分岐はパラメータの値そのものではなく
/// 呼び出し元（`DeviceParamStore`）が保持するステップカウンタに依存する。
/// `SgdConfig` 自体は構築後不変（`fandhe_ai_autodiff::optim::sgd::SgdConfig`
///参照）だが、`is_first_step` はステップごとに変化するため `SgdConfig` の
/// フィールドではなく本型（呼び出しごとに構築する値）のフィールドとする。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdStepConfig {
    /// 学習率。
    pub lr: f32,
    /// momentum 係数 `μ`。`0.0` は momentum 無効（`velocity` 引数は
    /// 無視してよい）。
    pub momentum: f32,
    /// dampening `τ`。
    pub dampening: f32,
    /// weight decay `λ`（L2 正則化。`torch.optim.SGD` と同じく `p` に
    /// 係数を乗じて勾配へ加算する）。
    pub weight_decay: f32,
    /// nesterov momentum を使うか。
    pub nesterov: bool,
    /// このパラメータ列にとって最初の `step()` 呼び出しか（momentum
    /// バッファの初期化分岐。上記フィールドドキュメント参照）。
    pub is_first_step: bool,
}

/// GEMM epilogue で適用する activation 種別（TASK-12.1f・#203）。
///
/// [`BackendOps::gemm_bias_act`] の第 4 引数として渡す。CUTLASS 系実測
/// （epilogue 融合で平均 1.38〜1.45 倍。イシュー #203）が動機の
/// Linear+bias+ReLU 相当パターンを表現できれば TASK-12.1f の受け入れ
/// 条件を満たせるため、まず `Relu` のみを持つ。`#[non_exhaustive]` は
/// 公開 API 非破壊（ガードレール条件・`.claude/rules/security.md`）を
/// 保ちながら将来 `Gelu`／`Sigmoid` 等を追加できるようにするため
/// （呼び出し側の網羅的 match を破壊しない。`GemmError`・`ParityError`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// activation なし（bias 加算のみ、または恒等関数）。
    None,
    /// `max(x, 0)`。`BackendOps::relu` と同一の定義を epilogue 内で適用する。
    Relu,
}

/// [`BackendOps::binary_elementwise_device`] が適用する 2 項 elementwise
/// 演算の種別（イシュー #1584。`BackendOps::add`／`mul` と同一の演算を
/// [`DeviceBuffer`] 常駐のまま実行するための選択子）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／`MseReduction`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryElementwiseOp {
    /// `a + b`（`BackendOps::add` と同一の定義）。
    Add,
    /// `a * b`（`BackendOps::mul` と同一の定義）。
    Mul,
}

/// [`BackendOps::unary_elementwise_device`] が適用する単項 elementwise
/// 演算の種別（イシュー #1584。`BackendOps::relu`／`exp`／`tanh` と同一の
/// 演算を [`DeviceBuffer`] 常駐のまま実行するための選択子）。
///
/// `#[non_exhaustive]`: `BinaryElementwiseOp` と同方針。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryElementwiseOp {
    /// `max(x, 0)`（`BackendOps::relu` と同一の定義）。
    Relu,
    /// `exp(x)`（`BackendOps::exp` と同一の定義）。
    Exp,
    /// `tanh(x)`（`BackendOps::tanh` と同一の定義）。
    Tanh,
}

/// [`BackendOps::mse_loss`]／[`BackendOps::mse_loss_backward`] の縮約種別
/// （イシュー #1045・親イシュー #1043「カーネル融合・autodiff 実行モデル
/// の強化」）。
///
/// `fandhe_ai_autodiff::var::Reduction`（`Mean`／`Sum`）と同一の意味論を
/// 持つが、`tensor-core` → `autodiff` の逆依存は作れない（本ファイル
/// 冒頭コメント・`SgdStepConfig` と同じ整理）ため独立した型として定義
/// する。`autodiff` 側で `impl From<Reduction> for MseReduction` を用意し
/// 変換する（`var.rs` 参照）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation` と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MseReduction {
    /// 全要素平均（`Σ(pred−target)² / n`）。
    Mean,
    /// 全要素総和（`Σ(pred−target)²`）。
    Sum,
}

/// [`BackendOps::captured_segment_key`]／[`BackendOps::run_captured_sgd_step_segment`]
/// が扱う 1 個のデバイスバッファの識別子（イシュー #1349・親 #1348・
/// ルート #1341 → #1269）。
///
/// CUDA Graph capture は「同じアドレス・同じ要素数のバッファへ、同じ
/// カーネル引数で launch する」ことを再利用の前提とする（`docs/
/// backend-cuda-graph-step-capture-design.md` §4.4）。`addr` はバックエンド
/// 実装（`backend-cuda::ops::CudaBackendOps`）がバッファのハンドルから
/// 取り出す値で、`tensor-core` 自体はその由来（`cudarc::driver::
/// DevicePtr::device_ptr` 等）を知らない・関与しない（バックエンド非依存
/// の型として定義するため）。`numel == 0`（空バッファ）は `addr == 0` で
/// 表す契約とする（呼び出し元がゼロ要素バッファを capture 対象に含めた
/// 場合の識別に使う。実際に capture するかどうかの判断＝空 graph の回避
/// は呼び出し元〈`run_captured_sgd_step_segment` 実装〉の責務）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentResource {
    /// バックエンド固有のバッファ識別子（CUDA では device pointer）。
    pub addr: u64,
    /// バッファの要素数。
    pub numel: usize,
}

/// capture 済み CUDA Graph の再利用可否を判定するキー（イシュー #1349）。
///
/// `generation` はバックエンドの poison 状態機械の世代
/// （`backend-cuda::context_cache::current_generation` 等）と一致させる
/// ことで、`invalidate` による回復（poison → 新世代）を跨いだ古い graph
/// を再利用しない（世代不一致は「別のデバイスコンテキストのグラフ」を
/// 意味し、キャッシュ側で evict する）。`config_key` は当該区間の
/// カーネル起動パラメータ（学習率等のハイパーパラメータ・`is_first_step`
/// 等の分岐フラグ）を呼び出し元が `u64` へ畳み込んだ値で、設定変更を
/// 検出して再 capture を促す。`resources` は区間が触れる全バッファの
/// [`SegmentResource`]（`Vec` の順序も含めて比較する。同じ集合でも順序が
/// 異なれば別キー＝別 graph として扱う。これは呼び出し元が毎回同じ順序で
/// 構築する契約であるため実害はなく、`Hash`/`Eq` の実装を単純に保つ
/// ための割り切り）。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentKey {
    /// バックエンドの poison 状態機械の世代（世代不一致の古い graph を
    /// 再利用しないためのキー要素）。
    pub generation: u64,
    /// 当該区間のカーネル起動パラメータを畳み込んだ値（設定変更の検出）。
    pub config_key: u64,
    /// 当該区間が触れる全バッファの識別子（順序を含めて比較する）。
    pub resources: Vec<SegmentResource>,
}

/// [`BackendOps::run_captured_sgd_step_segment`] が実際に capture したか、
/// 既存の graph を再生（replay）しただけかを呼び出し元へ伝える（イシュー
/// #1349。呼び出し元の launch 回数計測・テストでの制御フロー検証に使う）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentRun {
    /// 新規に stream capture → instantiate → 初回 launch した。
    Captured,
    /// 既存のキャッシュ済み graph を launch（再生）した。
    Replayed,
}

/// [`BackendOps::gemm_checksum`] の読み戻しモード（イシュー #1339）。
///
/// framework-compare の gemm 計測窓では、GEMM 出力 `C = A@B` を毎反復
/// ホストへ D2H した上で「縮退検出用の全要素和（checksum）」をホスト側
/// `f64` 逐次和で求め直しており、`docs/perf/cuda-gemm-reuse-phase-
/// breakdown.md`・`metal-gemm-reuse-phase-breakdown.md` の実測でこの
/// `host_copy`＋`checksum` の 2 段がハーネス計測窓の 66〜75% を占める
/// ことが確定した（イシュー #1338 承認）。本 enum は「毎反復は checksum
/// のみ 8 バイト読み戻す」（`ChecksumOnly`）か「加えて `C` 自体もホストへ
/// download する」（`WithOutput`。末尾反復の parity 検証用）かを選ぶ。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumReadout {
    /// checksum（8 バイト）のみ読み戻す。`output` は `None`。
    ChecksumOnly,
    /// checksum に加えて `C` 全体もホストへ download する。
    WithOutput,
}

/// [`BackendOps::gemm_checksum`] の戻り値。`checksum` は `C = A@B`
/// （論理領域 `m×n`）の全要素和を **`f64`（Metal は Neumaier 補償和で
/// `f64` 相当）アキュムレータ・固定順序**で求めた値（決定的。同一入力・
/// 同一カーネル選択であれば bit 決定的に同じ値を返す契約。CPU／CUDA／
/// Metal の各実装 doc 参照）。`output` は
/// [`ChecksumReadout::WithOutput`] のときのみ `Some`（[`BackendOps::gemm`]
/// と bit 同一の `Tensor<f32>`）。
#[derive(Debug, Clone)]
pub struct GemmChecksum {
    /// `C` の全要素和（f64 アキュムレータ・固定順序で決定的）。
    pub checksum: f64,
    /// [`ChecksumReadout::WithOutput`] のときのみ `Some`。
    pub output: Option<Tensor<f32>>,
}

/// [`BackendOps::linalg_qr`] の戻り値（イシュー #1621。`docs/
/// autodiff-linalg-design.md`）。`torch.linalg.qr(mode="reduced")` と
/// 同じ reduced QR（`A: [m,n]` → `q: [m,k]`・`r: [k,n]`、`k = min(m,n)`）。
/// `r` の対角は非負に正規化する（`fandhe_ai_autodiff::eval::linalg`
/// と本クレートの実装が同一符号規約を採る契約。設計文書 §3.5「符号・
/// ゲージ規約」）。フィールドは `pub`（`MseReduction`／`Activation` の
/// ような `#[non_exhaustive]` enum ではなく、バックエンド実装が値を
/// 直接構築する struct のため。`GemmChecksum` と同方針）。
#[derive(Debug, Clone)]
pub struct QrFactors {
    /// `[m, k]`（`k = min(m, n)`）。列直交（`QᵀQ ≈ I_k`）。
    pub q: Tensor<f32>,
    /// `[k, n]`（`k = min(m, n)`）。上三角・対角非負。
    pub r: Tensor<f32>,
}

/// [`BackendOps::linalg_svd`] の戻り値（イシュー #1621）。
/// `torch.linalg.svd(A, full_matrices=False)` と同じ reduced SVD
/// （`A: [m,n]` → `u: [m,k]`・`s: [k]`・`vh: [k,n]`、`k = min(m,n)`）。
/// `s` は降順（同値は安定ソート）に正規化する契約（設計文書 §3.5）。
#[derive(Debug, Clone)]
pub struct SvdFactors {
    /// `[m, k]`（`k = min(m, n)`）。列直交。
    pub u: Tensor<f32>,
    /// `[k]`。特異値（降順・非負）。
    pub s: Tensor<f32>,
    /// `[k, n]`（`k = min(m, n)`）。行直交（`V^T`）。
    pub vh: Tensor<f32>,
}

/// [`BackendOps::linalg_matrix_norm`] が計算する行列ノルムの種類
/// （イシュー #1621。`torch.linalg.matrix_norm` の `ord` 引数のうち
/// facade が対応する 5 種）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／`MseReduction`
/// と同方針）。将来 `ord=p`（任意次数）・`dim` 指定版を追加しうる
/// （設計文書「スコープ外」節）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNormOrd {
    /// Frobenius ノルム（`√Σ a_ij²`）。
    Fro,
    /// 最大絶対列和（`max_j Σ_i |a_ij|`）。
    One,
    /// 最大絶対行和（`max_i Σ_j |a_ij|`）。
    Inf,
    /// 核ノルム（特異値の総和）。
    Nuc,
    /// スペクトルノルム（最大特異値）。
    Spectral,
}

/// [`BackendOps::gru_backward`] の戻り値型エイリアス（イシュー #1647）。
/// `(d_pre_i, d_pre_h, dh_prev_direct)`（順に `[B, 3H]`・`[B, 3H]`・
/// `[B, H]`）。`clippy::type_complexity` 回避のための命名（doc は
/// `gru_backward` 側に集約する）。
pub type GruBackwardOutput = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// 各バックエンド（CPU／CUDA／Metal）が実装するカーネル入口
/// （`docs/public-api-design.md` §4.2。差分はモジュール冒頭コメント参照）。
///
/// object-safe に設計している（`&dyn BackendOps` として扱える。
/// [`ops_for`] が複数バックエンドを横断して選択する際に使用する）。
/// v1 は PoC-v2-5 実測 API（`MetalOps`）のスコープに合わせて `f32` 固定
/// とする（f16 経路のジェネリック化は §4.2 6-8 のとおり保留）。
///
/// 公開 API はすべて safe。`unsafe` は各バックエンド実装内部の FFI 境界
/// （`cudarc`・`objc2` 系呼び出し）に閉じ込める
/// （`.claude/rules/coding-rust.md`）。
pub trait BackendOps {
    /// このインスタンスが対応する [`Device`]（呼び出し元がログ・
    /// エラーメッセージで識別するために使う）。
    fn device(&self) -> Device;

    /// このバックエンドの [`MemoryOps`]（確保・アップロード・ダウンロード）
    /// 実装への参照（イシュー #935・`docs/device-resident-update-design.md`
    /// §3.1）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `None`（`MemoryOps` を持たない）。`BackendOps` を
    /// `MemoryOps` の supertrait にする案（`buffer.rs` モジュール冒頭
    /// コメント旧稿）は crates.io 公開済み trait への破壊的変更となる
    /// ため不採用と確定した（設計文書 §3.1）。本デフォルトメソッド追加は
    /// `gemm_bias_act`／`run_fused` と同じ非破壊拡張パターン（`BackendOps`
    /// を実装する外部クレートは何もしなくても既存実装のままコンパイル
    /// が通る）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::new` が
    /// `tape.ops().memory_ops()` を呼び、`None` の場合は
    /// [`BackendError::Unsupported`] としてデバイス常駐パラメータ更新を
    /// 拒否する（fail-closed。「`memory_ops()` を呼ぶフォールバック合成は
    /// 設けない」という設計文書 §3.2 改訂の確定事項に従い、`Some` を返す
    /// バックエンドのみがこの経路をサポートする）。CPU／CUDA／Metal の 3
    /// バックエンドはいずれも本デフォルトを `Some(self)` へオーバーライド
    /// する（各バックエンドクレートの `ops.rs` 参照）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        None
    }

    /// SGD の 1 パラメータ分の更新をデバイス上で in-place に実行する
    /// （イシュー #935・`docs/device-resident-update-design.md` §3.2）。
    ///
    /// `param`／`grad`／`velocity`（momentum 有効時のみ）はいずれも
    /// このバックエンド自身が確保した [`DeviceBuffer<f32>`]
    /// （[`MemoryOps::alloc_zeroed`]／[`MemoryOps::upload`] の戻り値）を
    /// 要求する契約。呼び出し元（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::step`）は毎ステップ `grad` のみをアップロードし
    /// `param`／`velocity` は前ステップから使い回すことで、param の
    /// ホスト再アップロードを排除する（本イシューの受け入れ条件）。
    ///
    /// **呼び出し元は全パラメータを連結した単一バッファで 1 回だけ呼ぶ**
    /// （イシュー #1023「パラメータ横断の単一連結バッファ化」）。
    /// `param`／`grad`／`velocity` はいずれもパラメータ数だけ個別に渡す
    /// のではなく、`DeviceParamStore` が全パラメータを 1 本の shape
    /// `[total_numel]` バッファへ連結して常駐させ、`step()` ごとに本
    /// メソッド（`sgd_step_device_tracked` 経由）を 1 回だけ起動する。
    /// 本メソッド自体は要素単位で shape 非依存に定義されているため、
    /// この呼び出し規約変更はシグネチャ・カーネル実装（CPU／CUDA／
    /// Metal のいずれも）に一切変更を要求しない。
    ///
    /// 更新式は `fandhe_ai_autodiff::optim::sgd`（`Sgd::step` ホスト参照
    /// 実装）と同一の項順序（weight_decay → momentum〈`is_first_step` で
    /// `b ← g` 分岐〉→ nesterov → 減算）を 3 バックエンドで揃える契約
    /// （設計文書 §5.2）。カーネル境界検査は省略しない（REQ-8・
    /// `.claude/rules/coding-rust.md`）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は常に [`BackendError::Unsupported`] を返す fail-closed
    /// （`memory_ops()` を呼ぶフォールバック合成は設けない。設計文書
    /// §3.2 改訂）。CPU／CUDA／Metal はこのデフォルトを実カーネルで
    /// オーバーライドする。
    ///
    /// # エラー
    /// - `param`／`grad`／`velocity` のいずれかがこのバックエンドの
    ///   ハンドル型へダウンキャストできない・デバイスが一致しない →
    ///   [`BackendError::DeviceMismatch`]
    /// - shape が一致しない → [`BackendError::ShapeMismatch`]
    /// - `config.momentum != 0.0` なのに `velocity` が `None` →
    ///   [`BackendError::Unsupported`]
    fn sgd_step_device(
        &self,
        _param: &mut DeviceBuffer<f32>,
        _grad: &DeviceBuffer<f32>,
        _velocity: Option<&mut DeviceBuffer<f32>>,
        _config: &SgdStepConfig,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            "sgd_step_device: default fail-safe (no in-place SGD kernel available)".into(),
        ))
    }

    /// [`BackendOps::sgd_step_device`] と同型だが、Metal のコマンド
    /// バッファ共有（イシュー #1017・`docs/backend-metal-command-
    /// batching-design.md`）向けに共有失敗トークン
    /// [`DispatchFailureCell`] を追加引数として受け取る非破壊拡張
    /// （`gemm_bias_act`／`run_fused` と同じ「デフォルトメソッド追加」
    /// パターン。`BackendOps` の SemVer 非破壊拡張）。
    ///
    /// # デフォルト実装
    /// 既定は `token` を無視して [`BackendOps::sgd_step_device`] へ
    /// そのまま委譲する。CPU は dispatch ごとに同期実行するため実行時
    /// エラーが呼び出し元に即座に返り、遅延失敗トークンを必要としない
    /// （このデフォルトのままでよい）。CUDA はイシュー #1013
    /// （`docs/backend-cuda-async-execution-design.md` §5）でカーネル
    /// 起動直後の都度 `synchronize()` を除去し非同期実行契約へ移行した
    /// が、本 `token`（`DispatchFailureCell`）は使わずオーバーライドも
    /// しない（このデフォルトのまま）。`backend-cuda::context_cache` は
    /// ordinal 単位の poison 状態機械（`begin_driver_call`／
    /// `observe_driver_result`／`observe_cuda_result`／`is_poisoned`。
    /// 単一ストリームの FIFO 順序保証を前提に sticky エラー観測時点で
    /// ordinal を poison する設計）を備え、PR #1064（イシュー #1013 の
    /// codex-review P0 指摘への対応）で `backend-cuda::ops`／
    /// `backend-cuda::memory` の `BackendOps`／`MemoryOps` 実装境界
    /// （`with_driver_call` ヘルパー）へ結線済みである
    /// （`docs/backend-cuda-async-execution-design.md` §12）。
    /// これにより、`sgd_step_device` 自身のカーネル起動が sticky な
    /// 実行時エラーを引き起こした場合、その ordinal 上で最初に
    /// `observe_cuda_result` が観測した時点（同一ステップの起動自体・
    /// 別の同一ステップ内 driver 呼び出し・または別テンソル演算の
    /// いずれか）で poison 化され、以降の `sgd_step_device` 呼び出しは
    /// `begin_driver_call` の拒否により `Err` を返す。`DeviceParamStore::
    /// step` は `sgd_step_device_tracked` が返す `Err` を常に
    /// `poisoned.store(true, ..)` へ変換する（`device_store.rs`
    /// `step` 実装参照）ため、この `Err` は必ず `StorePoisoned` への
    /// 自己遷移につながる。ただし検出は「次に同一 ordinal 上で
    /// driver 呼び出しが起きた時点」に限られ、poison からの**回復**
    /// （`context_cache::invalidate_with` の呼び出し）は #1062 へ
    /// 引き継いだままである。Metal のみ
    /// `backend-metal::ops::MetalBackendOps` がオーバーライドし、
    /// `MetalContext::encode` と**同一ロック区間で** `token` をバッチへ
    /// 登録する（encode と登録の間に別スレッドの `synchronize` が
    /// 割り込む競合を防ぐ。設計文書 §3.7 (2)）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::step`
    /// が呼び出し元となり、自身が保持する `failure_token` を渡す
    /// （4 つの状態機械エントリ全てが `token.is_set()` を検査して
    /// 自己 poison する。`device_store.rs` モジュール冒頭コメント参照）。
    fn sgd_step_device_tracked(
        &self,
        param: &mut DeviceBuffer<f32>,
        grad: &DeviceBuffer<f32>,
        velocity: Option<&mut DeviceBuffer<f32>>,
        config: &SgdStepConfig,
        _token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.sgd_step_device(param, grad, velocity, config)
    }

    /// 学習 step の一区間（イシュー #1349 では
    /// [`Self::sgd_step_device_tracked`] の update 区間のみ）を CUDA Graph
    /// で capture・再利用できるかを判定し、可能なら [`SegmentKey`] を返す
    /// （opt-in・既定 OFF。`docs/backend-cuda-graph-step-capture-design.md`
    /// §4.4）。
    ///
    /// `resources` に渡す各 [`DeviceBuffer<f32>`] は当該区間が読み書きする
    /// 全バッファ（例: `param`／`grad_staging`／`velocity`）を呼び出し元
    /// が**毎回同じ順序**で並べる契約（[`SegmentKey::resources`] の順序
    /// 込み比較）。`config_key` は当該区間のカーネル起動パラメータ
    /// （学習率等）を呼び出し元が `u64` へ畳み込んだ値。
    ///
    /// 呼び出し元（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::step`）は本メソッドが `Ok(Some(key))` を返した
    /// ときのみ [`Self::run_captured_sgd_step_segment`] を呼ぶ（`Ok(None)`
    /// は「このバックエンド・現在の設定では capture 非対応」を意味し、
    /// 呼び出し元は区間を直接実行する現行経路へフォールバックする）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は常に `Ok(None)`（graph 機構を持たないバックエンド・opt-in
    /// OFF の既定状態）。CUDA opt-in ON 時のみ
    /// `backend-cuda::ops::CudaBackendOps` がオーバーライドする。CPU・
    /// Metal はこのデフォルトのまま（graph 機構自体を持たない）。
    fn captured_segment_key(
        &self,
        _resources: &[&DeviceBuffer<f32>],
        _config_key: u64,
    ) -> Result<Option<SegmentKey>, BackendError> {
        Ok(None)
    }

    /// [`Self::captured_segment_key`] が返した `key` に対応する SGD 更新
    /// 区間（[`Self::sgd_step_device_tracked`] 相当）を capture（初回）
    /// または再生（2 回目以降）する（イシュー #1349）。
    ///
    /// **codex-review P0 指摘対応（任意クロージャの public 安全 API 化を
    /// 撤回）**: 旧稿は「`resources: &mut [&mut DeviceBuffer<f32>]` と
    /// 任意クロージャ `body: &mut dyn FnMut(&mut [&mut DeviceBuffer<f32>])`
    /// を受け取り、実装が `resources` のアドレスのみを再検証してから
    /// `body` を呼ぶ」形だった。この形には、`body` が **Rust クロージャの
    /// 環境キャプチャ経由で `resources` に含まれない外部の
    /// `DeviceBuffer<f32>` を直接触れる**（`resources` 引数を無視して
    /// クロージャがキャプチャした変数へ書き込む）という抜け道があり、
    /// 実装側の「`resources` のアドレス一致」再検証はその外部バッファを
    /// 一切カバーしない。CUDA Graph は capture 時点で触れた全アドレスを
    /// 焼き込むため、そのバッファが後で drop・再利用されると replay が
    /// 解放済み／別用途のアドレスを参照するメモリ安全性違反になりうる
    /// （`body` の型が `dyn FnMut` である限り、この抜け道を型システムで
    /// 塞ぐ手段がない）。本メソッドはこの型を公開 API から排除し、
    /// 「区間が触れる全リソースをメソッド自身の引数として直接受け取り、
    /// 区間本体（SGD 更新）もこのメソッドの実装が固定的に行う」——
    /// 任意クロージャを一切受け取らない操作記述に置き換える。これにより
    /// capture 中に触れうる `DeviceBuffer<f32>` は `param`／`grad`／
    /// `velocity` の 3 引数に限定され、実装のアドレス再検証がこの区間が
    /// 触れる全リソースを漏れなくカバーすることを型で保証する
    /// （`docs/backend-cuda-graph-step-capture-design.md` §4.4 追記）。
    ///
    /// `param`／`grad`／`velocity` は [`Self::captured_segment_key`] を
    /// 得たときと**同一の借用**で渡す契約: `SegmentKey` 自身はバッファの
    /// 所有権・借用を保持しない値型（`addr`／`numel` のみを畳み込んだ
    /// 識別子）であるため、呼び出し元が対応するバッファを drop した後に
    /// 本メソッドを呼ぶと、解放済み（または別バッファへ再利用済み）の
    /// アドレスを参照する古い graph を安全確認なしに再生してしまいうる
    /// （メモリ安全性違反）。実装は **replay 直前に** `param`／`grad`／
    /// `velocity` から導出した現在のアドレス集合が `key.resources`
    /// （capture 時点で刻印済み）と一致することを再検証し、不一致
    /// （呼び出し元の借用が `key` 発行時と異なる＝契約違反）なら
    /// [`BackendError::InvalidArgument`] で replay も新規 capture も
    /// 行わず拒否する（fail-closed）。
    ///
    /// `velocity` は [`Self::captured_segment_key`] の `resources` に
    /// velocity を含めた呼び出しと対でなければならない（`config.momentum
    /// != 0.0` かつ velocity 引数が異なる本メソッド・`captured_segment_key`
    /// を混在させない）。
    ///
    /// 初回（キャッシュミス）は stream capture を開始し、実装内部で
    /// [`Self::sgd_step_device_tracked`] 相当の更新を 1 回実行してから
    /// capture を終了・instantiate し、得られた graph をキーへ紐づけて
    /// キャッシュしたのち初回 launch する（[`SegmentRun::Captured`]）。
    /// 2 回目以降（キャッシュヒット）は更新を再実行せず、キャッシュ済み
    /// graph をそのまま launch する（[`SegmentRun::Replayed`]）。
    ///
    /// 区間本体が `Err` を返した場合、capture 自体は安全に終了させた
    /// うえでその `Err` を呼び出し元へ返す（capture 失敗の graph は
    /// キャッシュに残さない）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は常に [`BackendError::Unsupported`] を返す fail-closed。
    /// 呼び出し元は [`Self::captured_segment_key`] が `Some` を返した
    /// ときのみ本メソッドを呼ぶ契約であり、デフォルト実装のまま
    /// `Some` を返すバックエンドは存在しない（既定 `captured_segment_key`
    /// が常に `Ok(None)` を返すため、通常この `Err` へは到達しない。
    /// 到達した場合は呼び出し元・バックエンド実装間の契約違反であり、
    /// fail-closed に拒否する）。
    #[allow(clippy::too_many_arguments)]
    fn run_captured_sgd_step_segment(
        &self,
        _key: SegmentKey,
        _param: &mut DeviceBuffer<f32>,
        _grad: &DeviceBuffer<f32>,
        _velocity: Option<&mut DeviceBuffer<f32>>,
        _config: &SgdStepConfig,
        _token: &DispatchFailureCell,
    ) -> Result<SegmentRun, BackendError> {
        Err(BackendError::Unsupported(
            "run_captured_sgd_step_segment: default fail-safe (no CUDA Graph capture \
             mechanism available for this backend)"
                .into(),
        ))
    }

    /// 行列積 `C = A @ B` を計算する（`A: [m, k]`・`B: [k, n]` の 2 次元
    /// テンソルのみ受け付ける。shape 不整合は
    /// [`BackendError::ShapeMismatch`]）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    /// `gemm` と同じ行列積だが、**`crate::precision`（`backend-cuda`）の
    /// TF32 opt-in フラグ（`set_cuda_tf32_gemm_enabled`）の状態に関わらず
    /// 常に FP32 厳密で計算する**ことを契約するエントリ。
    ///
    /// `docs/cuda-tf32-optin-api-decision.md`・`backend-cuda::precision`
    /// モジュール冒頭コメントの契約「適用範囲は `CudaBackendOps::gemm`
    /// （素の公開 GEMM 入口）のみ。学習経路は本イシューのスコープ外の
    /// まま FP32 で動作する」を、`autodiff::grad`（VJP。イシュー #1211）
    /// のように **バックエンド非依存 `dyn BackendOps` 経由**で GEMM を
    /// 呼ぶ学習経路が満たすための入口。`ops.gemm(..)` を直接呼ぶと、
    /// CUDA では opt-in フラグが有効な間バックプロパゲーションが暗黙に
    /// TF32 化してしまう（codex-review 指摘。PR #1223）。
    ///
    /// 既定実装は `self.gemm(a, b)` に委譲する（TF32 の概念を持たない
    /// CPU・Metal はこれで契約を満たす）。TF32 opt-in を持つ
    /// `backend-cuda::CudaBackendOps` のみ、フラグを一切参照しない FP32
    /// 厳密経路（`run_tiled_f32`）へオーバーライドする。
    fn gemm_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.gemm(a, b)
    }

    /// [`Self::gemm_fp32_strict`] と同じ行列積 `C = A @ B` を計算するが、
    /// 結果をホストへ戻さず**呼び出し元が渡す既存の [`DeviceBuffer<f32>`]
    /// の指定オフセットへ直接書き込む**（イシュー #1212・`docs/
    /// device-resident-update-design.md` 追補）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore` が
    /// `Op::LinearResident` の d_weight（`crate::grad::vjp` が
    /// `gemm_fp32_strict` で計算し、GPU バックエンドでは戻り値の
    /// `Tensor<f32>` 構築自体が D2H を伴っていた）を、自身が保持する
    /// grad staging バッファへデバイス常駐のまま直接書き込むための入口。
    /// D2H（本メソッドの戻り値）に続く `DeviceParamStore::step` 側の
    /// H2D（`MemoryOps::upload`）を 1 パラメータぶん丸ごと排除する。
    ///
    /// # 契約
    ///
    /// - `out[out_offset .. out_offset + m*n]` を `A @ B` の結果で**上書き**
    ///   する（累積ではない）。`out_offset + m*n` は呼び出し元・実装側の
    ///   両方で `checked_mul`/`checked_add` により `out.numel()` 以内で
    ///   あることを検査し、範囲外は [`BackendError::InvalidArgument`]
    ///   （REQ-8「シェーダ・カーネル側の手動境界チェックを省略しない」・
    ///   OWASP A03）。
    /// - 数値は同 shape の [`Self::gemm_fp32_strict`] と **bit 同一**
    ///   （同一カーネル選択・CUDA の TF32 opt-in フラグには追従しない）。
    /// - `out.device()` は `self.device()` と一致すること
    ///   （[`BackendError::DeviceMismatch`]）。
    /// - `a`／`b` は 2 次元のみ（[`BackendError::ShapeMismatch`]）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張パターン。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`grad::vjp`
    /// の `Op::LinearResident` 分岐は `Unsupported` のときのみ既存の
    /// ホスト経路（`gemm_fp32_strict` を呼び戻り値をそのまま勾配として
    /// 使う）へフォールバックする（判定迂回を作らない。`.claude/rules/
    /// security.md` A08）。`backend-cpu::CpuBackendOps`（#1212）と
    /// `backend-metal::MetalBackendOps`（#1555。NT/TN 限定・encode-only）
    /// `backend-cuda::CudaBackendOps`（#1559。NT/TN 限定・GPU 側 smem
    /// 転置カーネル再利用）がオーバーライドする（3 バックエンドすべてが
    /// オーバーライド済み。詳細は `docs/perf/
    /// train-resident-grad-device-update.md`）。
    fn gemm_fp32_strict_into(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
        _out: &mut DeviceBuffer<f32>,
        _out_offset: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            "gemm_fp32_strict_into: default fail-safe (no in-place device GEMM kernel \
             available for this backend)"
                .into(),
        ))
    }

    /// [`Self::gemm_fp32_strict_into`] と同型だが、[`Self::
    /// sgd_step_device_tracked`] と同じく Metal のコマンドバッファ共有
    /// （イシュー #1017・`docs/backend-metal-command-batching-design.md`）
    /// 向けに共有失敗トークン [`DispatchFailureCell`] を追加引数として
    /// 受け取る非破壊拡張（`sgd_step_device`／`sgd_step_device_tracked`
    /// と同じ「デフォルトメソッド追加」パターン。`BackendOps` の SemVer
    /// 非破壊拡張）。
    ///
    /// # デフォルト実装
    /// 既定は `token` を無視して [`Self::gemm_fp32_strict_into`] へ
    /// そのまま委譲する。CPU は都度同期実行のため実行時エラーが
    /// 呼び出し元へ即座に返り、遅延失敗トークンを必要としない（この
    /// デフォルトのままでよい）。CUDA（`backend-cuda::ops::
    /// CudaBackendOps`。#1559）も同じ理由（`context_cache` の poison／
    /// 世代検査が各呼び出しごとに同期的に完結し、NT/TN 経路自体が
    /// 内部で `stream.synchronize()` する）でこのデフォルトのままで
    /// 良いが、トレイト doc の更新漏れを避けるため機能的に同一の明示
    /// オーバーライドを置いている（`ops.rs::CudaBackendOps::
    /// gemm_fp32_strict_into_tracked` のドキュメンテーションコメント
    /// 参照）。
    ///
    /// Metal のみ `backend-metal::ops::MetalBackendOps` がオーバーライド
    /// し、encode-only（待たない）で直接書き込む NT/TN 経路
    /// （`gemm_fp32_strict_into` doc「NT/TN 経路のみ encode-only にできる
    /// 理由」参照）で `MetalContext::encode` と**同一ロック区間で**
    /// `token` をバッチへ登録する（`sgd_step_device_tracked` doc と同じ
    /// 「encode と登録の間に別スレッドの `synchronize` が割り込む競合を
    /// 防ぐ」設計。codex-review 指摘・PR #1556: この登録がないと、
    /// 共有 `MetalContext` を使う別スレッドが先に `synchronize()` して
    /// GPU エラーを回収した場合、当該バッチは `committed` 列から drain
    /// 済みになり、呼び出し元（`DeviceParamStore`）自身の後続
    /// `download`／`upload_into` がエラーを observe できないまま成功
    /// してしまう——未完成または前回の勾配を正常値として読み出し・更新
    /// に使ってしまう fail-closed 違反を防ぐ）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::
    /// fill_resident_weight_grad` が呼び出し元となり、自身が保持する
    /// `failure_token` を渡す（`sgd_step_device_tracked` doc「4 つの
    /// 状態機械エントリ」と同様、`step()` 冒頭の `failure_token.is_set()`
    /// 検査が自己 poison する）。
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

    /// [`Self::gemm_fp32_strict_into_tracked`] と同じ `C = A @ B` を
    /// 計算するが、加えて `b`（`Op::LinearResident` の VJP では
    /// `d_weight = x_t @ g` の `g` そのもの）の**行方向の和**（bias 勾配。
    /// `autodiff::grad::reduce_to_shape` の rank-2→rank-1 特殊ケースと
    /// 同型の縮約〈shape の対応は同一だが蓄積方式は下記「# 引数」参照〉）
    /// を計算できる場合は `out` の別範囲へ同時に書き込むための非破壊
    /// 拡張（イシュー #1566・`docs/backend-metal-command-batching-
    /// design.md` §10）。
    ///
    /// `docs/perf/train-resident-grad-device-update.md`（#1212）で
    /// `d_weight` を resident staging へ直接書き込む経路が確立した後も、
    /// bias 勾配（`Op::LinearResident.bias`）は依然ホスト側
    /// `reduce_to_shape`（f32 逐次和）で計算し `MemoryOps::upload_into`
    /// で書き戻していた。この `upload_into` が防御的に呼ぶ
    /// `MetalContext::synchronize()` が `command_batching_bench` に残る
    /// 最後の同期点だった（`docs/backend-metal-command-batching-design.md`
    /// §10「案 A′」）。本メソッドは、既にアップロード済みの `g`
    /// （d_weight 計算に使う `b` 引数）を再利用して bias 勾配も同一
    /// ディスパッチ内で encode-only に計算することで、この同期点を
    /// 削減する経路を提供する。
    ///
    /// # 引数
    ///
    /// `bias` が `Some((bias_offset, n))` の場合、`out[bias_offset ..
    /// bias_offset + n]` へ `b` の行方向和（`b: [m, n]` の各列 `j` に
    /// ついて `sum_{i=0}^{m-1} b[i, j]`。走査順は行 `0..m` 昇順）を書き
    /// 込む。蓄積方式は `.claude/rules/coding-rust.md` の勾配長軸縮約
    /// `f64` アキュムレータ方針（2026-09-12 ユーザー承認 A）に従い、
    /// 実装はホスト `f64` アキュムレータ（`acc: f64 = 0.0` から行 `0..m`
    /// を昇順に加算し最後に 1 回 `as f32`）、または `double` 非対応の
    /// Metal では同じ演算列を IEEE 754 binary64 加算の 64bit 整数ソフト
    /// ウェアエミュレーションで再現するカーネル（`fandhe_ai_backend_
    /// metal::shaders::gemm_bias_grad_reduce_f32`。逐語モデルは
    /// `fandhe_ai_backend_metal::soft_f64`）を用いる。いずれも
    /// `fandhe_ai_backend_metal::layout::reduce_bias_grad_rows_host` と
    /// **bit 完全一致**する（NaN のみ payload がハードウェア依存のため
    /// クラス一致。`docs/backend-metal-command-batching-design.md`
    /// §10.14）。
    /// `bias_offset + n` は [`Self::gemm_fp32_strict_into`]
    /// の `out_offset + m*n` と同じ検査規約（`checked_add`・範囲外は
    /// [`BackendError::InvalidArgument`]。REQ-8・OWASP A03）を適用する。
    ///
    /// # 戻り値
    ///
    /// `Ok(bias_filled)`: weight（`out[out_offset..]`）は常に書き込まれる
    /// （成功時）。`bias_filled` は `bias` が `Some` のときに実際に
    /// `out[bias_offset..]` へも書き込めたかを示す——`bias` が `None` の
    /// ときは常に `false`。バックエンドが weight のみ対応し bias 縮約を
    /// 実装しない場合も `bias` を `Some` のまま `Ok(false)` を返してよい
    /// （呼び出し元はホスト `reduce_to_shape` へフォールバックする。
    /// weight/bias の対応可否は独立という契約——`autodiff::tape::
    /// ResidentResolver::fill_resident_weight_grad` doc 参照）。
    /// `Err`（`Unsupported` 含む）の場合は weight・bias いずれも `out` へ
    /// 書き込まれていないことを呼び出し元は仮定してよい（`gemm_fp32_
    /// strict_into` と同じ全体成功/失敗契約）。
    ///
    /// # デフォルト実装
    ///
    /// `bias` を無視して [`Self::gemm_fp32_strict_into_tracked`]
    /// （weight のみ）へ委譲し `Ok(false)` を返す（`CpuBackendOps`
    /// （#1212）・`CudaBackendOps`（既定 `Unsupported` のまま）は本メソッド
    /// を一切オーバーライドしない＝挙動変更ゼロ）。`backend-metal::ops::
    /// MetalBackendOps`（#1566）のみオーバーライドし、NT/TN
    /// （encode-only）経路では同一 `ctx.encode` 呼び出し内で bias 縮約
    /// も追加ディスパッチし、それ以外（NN/TT・分類不能形状）は
    /// ホスト経路フォールバック（`gemm_fp32_strict` → `upload_into` に
    /// 続けて bias もホスト計算 → `upload_into`）で `bias` の有無に
    /// 関わらず常に成功する（`gemm_fp32_strict_into_impl` の
    /// `resident_grad_capability` 汚染防止契約を維持する）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp32_strict_into_with_bias_reduce_tracked(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
        bias: Option<(usize, usize)>,
        token: &DispatchFailureCell,
    ) -> Result<bool, BackendError> {
        let _ = bias;
        self.gemm_fp32_strict_into_tracked(a, b, out, out_offset, token)?;
        Ok(false)
    }

    // elementwise（`docs/public-api-design.md` §4.2 と同じ 5 演算）
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    // reduction（`docs/public-api-design.md` §4.2 と同じ 2 演算）
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;

    /// 平均二乗誤差 `reduction(Σ(pred−target)²)` の forward を 1 個の
    /// 融合カーネルで計算する（イシュー #1045・親イシュー #1043）。
    ///
    /// `pred`／`target` は同一 shape（呼び出し元が [`crate::ops_shape::
    /// require_same_shape`] で検証済み）。戻り値は shape `[]`（スカラー）。
    /// `numel == 0` は `Mean`／`Sum` とも `0.0`（`fandhe_ai_autodiff::eval::
    /// mse_loss` の既存契約と同じ。mean 側はゼロ除算回避、sum 側は空和が
    /// 数学的に 0 のため元々の定義と一致）。
    ///
    /// # デフォルト実装
    ///
    /// 本メソッドは `gemm_bias_act`・`sgd_step_device` と同じ非破壊拡張
    /// （デフォルトメソッド追加。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする。
    /// `fandhe_ai_autodiff::var::Var::mse_loss_with` は `Unsupported` の
    /// ときのみ従来のホスト参照実装（`eval::mse_loss`）へフォールバック
    /// し、それ以外のエラーは伝播する（判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。CPU／CUDA／Metal の各実装は
    /// このデフォルトをカーネル内融合実装でオーバーライドする
    /// （`backend-cpu::mse`・`backend-cuda::mse`・`backend-metal::mse`
    /// 参照）。
    fn mse_loss(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "mse_loss: default fail-safe (no fused MSE forward kernel available)".into(),
        ))
    }

    /// 平均二乗誤差の backward（`dPred = scale·(pred−target)`）を 1 個の
    /// 融合カーネルで計算する（イシュー #1045）。
    ///
    /// `scale` は呼び出し元（`fandhe_ai_autodiff::grad::vjp` の
    /// `Op::MseLoss` 分岐）が上流勾配 `g`（スカラー）と `reduction` から
    /// 事前計算して渡す（`Mean` は `g·2/n`、`Sum` は `g·2`）。カーネル側は
    /// 縮約種別を意識せずこの `scale` を適用するだけでよい。
    ///
    /// `dTarget = −dPred` は常に成り立つ（`d/dtarget (pred−target)² =
    /// −2(pred−target)`）ため、本メソッドは `dPred` の 1 テンソルのみを
    /// 返す契約とする（`dTarget` を別テンソルとしてカーネルに計算・
    /// 転送させるのは無駄な allocation・D2H を増やすだけであり、融合
    /// カーネルで転送量を削減するという本イシューの目的と矛盾する）。
    /// 呼び出し元がホスト側で `dPred` を符号反転するだけで `dTarget` を
    /// 得る（`grad.rs` 参照）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`Var::
    /// mse_loss_with` の呼び出し元（`grad::vjp`）は `Unsupported` の
    /// ときのみ既存のホスト参照実装（`mse_loss_vjp`）へフォールバックする。
    fn mse_loss_backward(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "mse_loss_backward: default fail-safe (no fused MSE backward kernel available)".into(),
        ))
    }

    /// 行方向 softmax（`exp(x - max(x)) / sum(exp(x - max(x)))`）の
    /// 独立エントリ（イシュー #1594）。既存の [`Self::run_fused`] 経由
    /// （`match_softmax_plan` の canonical プラン一致限定）とは別に、
    /// `fandhe_ai_autodiff::var::Var::softmax` が直接呼べる入口を提供する。
    ///
    /// **契約**: `dim` が `x` の最終軸のときのみ計算を試みてよい
    /// （[`crate::ops_shape::row_softmax_layout`] が非最終軸を `Ok(None)`
    /// として区別する契約に対応。行カーネルは最終軸専用のため、
    /// 非最終軸は本メソッドをオーバーライドしない実装でも
    /// [`BackendError::Unsupported`] を返す既定のままでよい）。戻り値の
    /// shape は入力 `x` と恒等（softmax は shape 不変）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`Var::
    /// softmax` は `Unsupported` のときのみホスト参照実装
    /// （`eval::softmax_along`）へフォールバックする（それ以外のエラーは
    /// 伝播する。判定迂回経路を作らない。`.claude/rules/security.md`
    /// A08）。CPU／CUDA／Metal はいずれもこのデフォルトを既存の融合
    /// softmax カーネル（`run_fused` の softmax 一致経路が使うものと同一
    /// のカーネル実体）でオーバーライドする。
    fn softmax(&self, _x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "softmax: default fail-safe (no fused softmax kernel available)".into(),
        ))
    }

    /// 行方向 log_softmax（`x − m − ln(Σ exp(x − m))`。`m` は行 max）の
    /// 独立エントリ（イシュー #1594）。[`Self::softmax`] と同じ最終軸
    /// 限定契約・非破壊拡張・フォールバック規律に従う
    /// （`Var::log_softmax` は `Unsupported` のときのみ `eval::
    /// log_softmax_along` へフォールバックする）。
    ///
    /// **`ln(softmax(x))` にしない理由**: softmax の出力がアンダー
    /// フローで `0.0` になった要素で `ln(0.0) = -inf` を経由し数値精度を
    /// 落とすため、`x − m − ln(Σexp(x−m))` の解析形で計算する（PyTorch
    /// `F.log_softmax` と同じ安定化方針）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::softmax`] と同じ非破壊拡張・fail-safe。本イシュー時点で
    /// GPU 側（CUDA／Metal）に log_softmax 専用カーネルは存在しないため
    /// 両バックエンドともこの既定のまま（ホストフォールバックに委ねる）
    /// で、CPU のみ融合カーネル（`backend-cpu::softmax::
    /// run_log_softmax_f32`）でオーバーライドする。
    fn log_softmax(&self, _x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "log_softmax: default fail-safe (no fused log_softmax kernel available)".into(),
        ))
    }

    /// `inputs` を `dim` 軸で連結する（`torch.cat` 相当。イシュー
    /// #1598）。入力は strided view（`contiguous()` を経ずに渡されうる）
    /// でよく、出力は必ず contiguous・**bit 完全一致のコピー**（丸め
    /// なし。REQ-2 の複合判定より強い bit 同一を parity テストで要求
    /// する）。`dim` 以外の軸の shape 一致は呼び出し元
    /// （[`crate::ops_shape::concat_out_shape`]）で検査済みだが、実装側
    /// でも `inputs` の shape を再検査し、不一致は
    /// [`BackendError::ShapeMismatch`] を返すこと（fail-closed。
    /// 判定迂回経路を作らない。`.claude/rules/security.md` A08）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::softmax`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::cat`（`grad.rs::
    /// concat_with_fallback` 経由）は `Unsupported` のときのみホスト
    /// 参照実装（`eval::concat`）へフォールバックする（それ以外の
    /// エラーは伝播する）。
    fn concat(&self, _inputs: &[&Tensor<f32>], _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "concat: default fail-safe (no fused concat kernel available)".into(),
        ))
    }

    /// 条件テンソルによる要素選択（`torch.where` 相当。イシュー
    /// #1637）。`cond` は `a`／`b` と同じ shape へ broadcast 済みの
    /// **f32 マスク**として渡される（`Tensor<bool>` はデバイス転送
    /// 契約〈`MemoryOps` は f32 専用〉の対象外のため、呼び出し元
    /// `fandhe_ai_autodiff::var::Var::where_cond` が bool→f32 変換を
    /// 1 回だけ行い `out_shape` ちょうどの contiguous テンソルへ
    /// 実体化する）。真偽の判定契約は**3 バックエンド共通で
    /// `c != 0.0`**（CPU `Rust c != 0.0`・CUDA `c != 0.0f`・Metal MSL
    /// `c != 0.0f`。NaN マスクは呼び出し元で発生し得ないため考慮不要）。
    /// `cond`／`a`／`b` は全て同一 shape（`out_shape`）であること。
    /// 出力 shape は `a`（＝`b`＝`cond`）と恒等。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::concat`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::where_cond` は
    /// `Unsupported` のときのみホスト参照実装（`eval::where_cond`）へ
    /// フォールバックする（それ以外のエラーは伝播する。判定迂回経路を
    /// 作らない。`.claude/rules/security.md` A08）。実装側でも
    /// `cond`／`a`／`b` の shape を再検査し、不一致は
    /// [`BackendError::ShapeMismatch`] を返すこと（fail-closed）。
    fn where_cond(
        &self,
        _cond: &Tensor<f32>,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "where_cond: default fail-safe (no fused where kernel available)".into(),
        ))
    }

    /// マスク位置を定数 `value` で置換する（`torch.masked_fill` 相当。
    /// イシュー #1637）。`mask` は `x` と同じ shape へ broadcast 済みの
    /// f32 マスク（[`Self::where_cond`] と同じ `c != 0.0` 判定契約・
    /// bool→f32 変換の位置づけ）。`mask` の要素が真（`!= 0.0`）の位置を
    /// `value` に置換し、それ以外は `x` の値をそのまま返す。出力 shape
    /// は `x` と恒等。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::where_cond`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::masked_fill` は
    /// `Unsupported` のときのみホスト参照実装（`eval::masked_fill`）へ
    /// フォールバックする。実装側でも `x`／`mask` の shape 一致を
    /// 再検査し、不一致は [`BackendError::ShapeMismatch`] を返すこと
    /// （fail-closed）。
    fn masked_fill(
        &self,
        _x: &Tensor<f32>,
        _mask: &Tensor<f32>,
        _value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "masked_fill: default fail-safe (no fused masked_fill kernel available)".into(),
        ))
    }

    /// GEMM の epilogue（bias 加算・activation）を融合した
    /// `act(A @ B + bias)` を計算する（TASK-12.1f・#203）。
    ///
    /// `bias` は `[n]`（`B` の列数）の 1 次元テンソルで、`A @ B: [m, n]` の
    /// 各行へブロードキャスト加算される（`None` の場合は bias 加算を
    /// 省略する）。`act` は bias 加算後に適用する
    /// （[`Activation::None`] なら恒等関数）。
    ///
    /// # デフォルト実装（非融合合成）
    ///
    /// 本メソッドは **デフォルトメソッド**として追加している（`BackendOps`
    /// の非破壊拡張。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）。デフォルト実装は `gemm` →
    /// （`bias` があれば）`add`（行方向ブロードキャスト。
    /// `docs/public-api-design.md` §4.2 のブロードキャスト規約に従い
    /// `[n]` を `[1, n]` として `[m, n]` へ揃える）→ `act` に応じた
    /// activation メソッド呼び出しの 3 段合成である。CPU バックエンドは
    /// [`crate`] を利用する `backend-cpu::ops::CpuBackendOps` がこの
    /// デフォルトを **カーネル内融合実装でオーバーライド**し、中間
    /// `Tensor` 2 個の割当・GEMM 結果の再読み出しパスを削減する
    /// （CUTLASS 系実測で epilogue 融合が平均 1.38〜1.45 倍。動機は
    /// イシュー #203）。CUDA はイシュー #599 で
    /// `backend-cuda::ops::CudaBackendOps::gemm_bias_act` が本デフォルトを
    /// **カーネル内融合実装でオーバーライド**した（CPU と同じ「bias が
    /// `None` または `[n]` 厳密一致なら融合、それ以外は非融合合成へ
    /// フォールバック」という分岐条件。`backend-cuda::ops::
    /// gemm_bias_act_route` 参照）。Metal はイシュー #605 で
    /// `backend-metal::ops::MetalBackendOps::gemm_bias_act` が本デフォルトを
    /// **カーネル内融合実装でオーバーライド**した（CPU／CUDA と同じ「bias
    /// が `None` または `[n]` 厳密一致なら融合、それ以外は非融合合成へ
    /// フォールバック」という分岐条件。`backend-metal::ops::
    /// gemm_bias_act_route` 参照）。CPU／CUDA／Metal の 3 バックエンドが
    /// すべて融合カーネルでオーバーライド済みとなった。
    ///
    /// `bias` の shape が `[n]` の場合（CPU バックエンドでは融合カーネルの
    /// 対応範囲）はそのまま計算する。`[n]` でない場合は `add` の NumPy
    /// 互換ブロードキャスト判定へ委譲し、`out: [m, n]` へブロードキャスト
    /// **不能**な場合にのみ [`BackendError::ShapeMismatch`] を返す
    /// （`[1]`・`[1, n]`・`[m, n]` 等ブロードキャスト可能な shape は
    /// 成功する。CPU／CUDA／Metal で同一の意味論。#203 Review 指摘）。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let mut out = self.gemm(a, b)?;
        if let Some(bias) = bias {
            out = self.add(&out, bias)?;
        }
        out = match act {
            Activation::None => out,
            Activation::Relu => self.relu(&out)?,
        };
        Ok(out)
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」・`docs/device-resident-update-design.md`
    /// §3.3e）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::linear_forward`
    /// が学習ループの forward で使う。`a`（ホスト常駐）は毎ステップ変化
    /// する活性化値、`w`（デバイス常駐）は学習対象パラメータであり、
    /// `sgd_step_device` と同じく **本メソッドが `w`／`bias` を
    /// ホストへ download しない**ことが受け入れ条件の中核（本イシューが
    /// 排除する対象は「forward のたびにパラメータをホストへ落とす」
    /// D2H であり、`a`・戻り値の D2H は含まない。`docs/device-resident-
    /// update-design.md` §1.2 の解釈）。
    ///
    /// `w`／`bias` は [`DeviceBufferView`]（イシュー #1023「パラメータ
    /// 横断の単一連結バッファ化」後、`DeviceParamStore` が全パラメータを
    /// 1 本の連結 `DeviceBuffer<f32>` として保持するため、個々の
    /// パラメータは連結バッファ内の要素オフセット範囲としてしか
    /// 表現できない。「R3: 要素オフセット付き常駐ビュー」設計。
    /// `docs/device-resident-update-design.md` 追補参照）で渡す。実装は
    /// `view.offset()..view.offset() + view.numel()` の範囲のみを
    /// `view.shape()` の重みとして扱う契約（この範囲チェック自体は
    /// [`DeviceBufferView::new`] が構築時に行うため、本メソッドの実装は
    /// 追加のオフセット境界検査を要しないが、カーネル側の手動境界検査
    /// 〈REQ-8〉は従来どおり省略しない）。
    ///
    /// `bias` は `Some` の場合 `[n]`（`w` の列数）への行方向複製のみ
    /// 対応する（[`BackendOps::gemm_bias_act`] の融合カーネルと同じ厳密
    /// 一致契約。ブロードキャスト全般は非対応）。`k`（`a` の列数 = `w`
    /// の行数）が 0 の呼び出しは `sgd_step_device` と同様に呼び出し元
    /// （`fandhe_ai_autodiff::nn::linear::Linear::new` が `in_features == 0`
    /// を構築時に拒否する）の契約により実運用では到達しない。
    ///
    /// # デフォルト実装
    ///
    /// 本メソッドは `sgd_step_device`／`gemm_bias_act` と同じ非破壊拡張
    /// （デフォルトメソッド追加。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする（デバイス
    /// 常駐オペランドを扱えないバックエンドが誤って黙示のホスト
    /// フォールバック〈`w` を download してから `gemm_bias_act` へ委譲する
    /// 等〉を行い、D2H 排除という受け入れ条件を静かに破ることを防ぐため。
    /// `download` してよいなら本メソッドを呼ぶ意味がない）。CPU／CUDA／
    /// Metal の各実装はこのデフォルトをカーネル呼び出しでオーバーライド
    /// する（`backend-cpu::ops::CpuBackendOps`・`backend-cuda::ops::
    /// CudaBackendOps`・`backend-metal::ops::MetalBackendOps` 参照）。
    fn gemm_resident_rhs(
        &self,
        _a: &Tensor<f32>,
        _w: DeviceBufferView<'_>,
        _bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_resident_rhs: default fail-safe (no resident-operand GEMM kernel available)"
                .into(),
        ))
    }

    /// [`Self::gemm_resident_rhs`] の activation 融合版（イシュー #1044・
    /// `docs/kernel-fusion.md` §2.2）。学習 forward の `Linear` 層に続く
    /// `ReLU` 層をこの epilogue へ折り込み、層 1 個あたりのカーネル
    /// 起動数を 2（gemm+bias／relu）から 1（gemm+bias+act）へ減らす。
    /// 呼び出し元は `fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::linear_forward_with_activation`（次層が
    /// `ReLU` の場合のみ `Activation::Relu` を渡し、それ以外は
    /// `Activation::None` を渡す。bias のみの融合は既存
    /// `gemm_resident_rhs` と同じ）。
    ///
    /// `bias` は `Some` の場合 `[n]`（`w` の列数）への行方向複製のみ
    /// 対応する（[`Self::gemm_resident_rhs`] と同じ厳密一致契約）。
    ///
    /// # デフォルト実装（非破壊拡張）
    ///
    /// `gemm_bias_act`（`gemm` → `add` → `act` の 3 段合成）と同型の
    /// フェイルセーフ合成: [`Self::gemm_resident_rhs`]（bias 融合のみ・
    /// `act` なし）を呼んだ後、`act == Relu` なら結果へ
    /// `self.relu`（ホスト常駐 `Tensor` に対する elementwise。実体化済み
    /// のためこの合成は追加のデバイス常駐制約を破らない）を適用する。
    /// `gemm_resident_rhs` 自体が `Unsupported` を返すバックエンド
    /// （本メソッドをオーバーライドしていないバックエンド）では、この
    /// デフォルトも同じ `Unsupported` を透過的に伝播する。CPU
    /// バックエンド（`backend-cpu::ops::CpuBackendOps`）はこのデフォルトを
    /// カーネル内融合実装（`gemm_blis_bias_act_parallel` へ `act` を
    /// 直接渡す）でオーバーライドする。CUDA／Metal は本イシューの
    /// スコープ外（実機検証環境が必要。`launch_tiled_bias_act_f32_
    /// resident`／`dispatch_strided_bias_act_prepared` は既に `act_relu`
    /// を受け取れるため、後続イシューでの結線は型検査のみで済む）。
    fn gemm_resident_rhs_act(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out = self.gemm_resident_rhs(a, w, bias)?;
        match act {
            Activation::None => Ok(out),
            Activation::Relu => self.relu(&out),
        }
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。
    ///
    /// `DeviceParamStore` の resident backward（`Op::LinearResident` の
    /// VJP。`fandhe_ai_autodiff::grad`）が `d_input^T = w @ g^T` を計算する
    /// ために使う（`w: [k, n]`・`b: [n, m]` → `c: [k, m]`。呼び出し元が
    /// `c` を転置して `d_input: [m, k]` を得る）。[`Self::gemm_resident_rhs`]
    /// と対になる「常駐オペランドが左辺」の形（`w` が左、`b` がホスト
    /// 常駐の右辺）。`w` が [`DeviceBufferView`] を取る理由は
    /// [`Self::gemm_resident_rhs`] と同じ。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gemm_resident_rhs`] と同じ理由・同じ fail-safe 方針
    /// （[`BackendError::Unsupported`]）のデフォルトメソッド。
    fn gemm_resident_lhs(
        &self,
        _w: DeviceBufferView<'_>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_resident_lhs: default fail-safe (no resident-operand GEMM kernel available)"
                .into(),
        ))
    }

    /// `a`（デバイス常駐）・`w`（デバイス常駐）・`bias`（デバイス常駐・
    /// 任意）から `y = act(a @ w + bias)` を、入力・出力いずれもホストへ
    /// 実体化せずに計算する（イシュー #1028・`docs/inference-forward-
    /// fixed-cost-design.md` §3.2）。
    ///
    /// [`Self::gemm_resident_rhs`] は `a`・戻り値がホスト常駐 `Tensor<f32>`
    /// であり、多層 MLP の推論チェーンでは層ごとに D2H（戻り値）→ H2D
    /// （次層の `a`）が発生する（`docs/backend-cuda-async-execution-
    /// design.md` §2.3 が指摘する「ホスト `Tensor` を返す `BackendOps` API
    /// は戻り値の D2H が構造的な同期点」の具体例）。本メソッドは `a`・
    /// 戻り値をいずれも [`DeviceBuffer`] のまま扱うことで、推論チェーンの
    /// 同期点を最終出力の 1 回（呼び出し元が明示的に `download` する
    /// 箇所）へ集約できるようにする。
    ///
    /// `act` は bias 加算後の elementwise 適用（[`Activation::Relu`] は
    /// `max(0, x)` で数値的に恒等な後段適用。[`Self::gemm_bias_act`] と
    /// 同じ契約）。`bias` は `Some` の場合 `[n]`（`w` の列数）への行方向
    /// 複製のみ対応する（[`Self::gemm_resident_rhs`] と同じ厳密一致契約。
    /// ブロードキャスト全般は非対応）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gemm_resident_rhs`]・[`Self::gemm_resident_lhs`] と同じ
    /// 非破壊拡張（デフォルトメソッド追加）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする（デバイス
    /// 常駐の入出力を扱えないバックエンドが誤って黙示のホスト
    /// フォールバック〈`a` を download して `gemm_bias_act` へ委譲し
    /// 結果を再 upload する等〉を行い、「入出力とも D2H/H2D しない」
    /// という受け入れ条件を静かに破ることを防ぐため。呼び出し元
    /// （`fandhe_ai_autodiff::optim::device_store` の推論ヘルパー）は
    /// `Unsupported` を検出した場合、層構成全体を [`Self::gemm_bias_act`]
    /// ベースの per-op 経路へフォールバックする契約とする）。CPU 実装
    /// （`backend-cpu::ops::CpuBackendOps`）はこのデフォルトをカーネル
    /// 呼び出しでオーバーライドする。CUDA（`backend-cuda::ops::
    /// CudaBackendOps`）・Metal（`backend-metal::ops::MetalBackendOps`）
    /// は #1216 で実装済み（融合カーネル `launch_tiled_bias_act_f32_
    /// resident`／`encode_strided_bias_act_prepared`〈`dispatch_
    /// strided_bias_act_prepared` の encode-only 版。`ctx.synchronize()`
    /// を呼ばずコマンドバッファへ積むのみで待たない〉を再利用し、
    /// `a`／戻り値もデバイス常駐のまま扱う。実機実測は `docs/perf/
    /// linear-forward-device-gpu.md`）。
    fn linear_forward_device(
        &self,
        _a: &DeviceBuffer<f32>,
        _w: DeviceBufferView<'_>,
        _bias: Option<DeviceBufferView<'_>>,
        _act: Activation,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linear_forward_device: default fail-safe (no device-resident chained forward \
             kernel available)"
                .into(),
        ))
    }

    /// `a op b`（`op` は [`BinaryElementwiseOp`]）を `a`／`b`／戻り値
    /// いずれも [`DeviceBuffer`] 常駐のまま計算する（イシュー #1584）。
    /// `linear_forward_device` と同じ動機（`docs/inference-forward-
    /// fixed-cost-design.md` §2.3 の「ホスト `Tensor` を返す `BackendOps`
    /// API は戻り値の D2H が構造的な同期点」）で、H2D／D2H・同期を伴わず
    /// ストリーム／コマンドバッファへ積むだけの経路を提供し、呼び出し元
    /// が複数の elementwise 演算を連鎖させたうえで最後に 1 回だけ
    /// `download` できるようにする。
    ///
    /// `a`・`b` は shape 完全一致限定（ブロードキャスト非対応。不一致は
    /// `BackendError::ShapeMismatch` で fail-closed）。数値は対応する
    /// ホスト版（`BackendOps::add`／`mul`）と同一カーネルにより bit 同一
    /// となる契約。
    ///
    /// # 同期契約はバックエンドごとに異なる（イシュー #1675 codex-review
    /// 指摘）
    ///
    /// 上記「ストリーム／コマンドバッファへ積むだけで待たない」は
    /// `backend-cuda`（ストリーム順序実行。`docs/backend-cuda-async-
    /// execution-design.md`）の同期契約であり、**全バックエンド共通の
    /// トレイト契約ではない**。`backend-metal` の実装（`elementwise::
    /// MetalElementwise::dispatch_binary_resident`）は `MetalContext::
    /// dispatch_sync` 経由のため、呼び出しごとに 1 回 `waitUntilCompleted`
    /// する（`docs/backend-metal-command-batching-design.md`。呼び出し元
    /// が複数演算を連鎖させても同期点は 1 回に集約されない）。「同期点を
    /// 呼び出し元の `download` へ集約する」設計を前提にする呼び出し側は
    /// バックエンドごとの実装 doc（各 `fn binary_elementwise_device`／
    /// `unary_elementwise_device` オーバーライド）を確認すること。
    ///
    /// # デフォルト実装
    ///
    /// `linear_forward_device` と同じ非破壊拡張パターン（`BackendOps`
    /// トレイトへのデフォルトメソッド追加。公開 API 非破壊はガードレール
    /// 条件・`.claude/rules/security.md`）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする（デバイス
    /// 常駐の入出力を扱えないバックエンドが黙示のホストフォールバック
    /// 〈`a`／`b` を download して `add`／`mul` へ委譲し結果を再 upload
    /// する等〉を行い、「入出力とも D2H/H2D しない」という受け入れ条件を
    /// 静かに破ることを防ぐため）。`backend-cpu`・`backend-cuda`・
    /// `backend-metal` の各実装はこのデフォルトをオーバーライドする。
    fn binary_elementwise_device(
        &self,
        _op: BinaryElementwiseOp,
        _a: &DeviceBuffer<f32>,
        _b: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "binary_elementwise_device: default fail-safe (no device-resident elementwise \
             kernel available)"
                .into(),
        ))
    }

    /// `op(a)`（`op` は [`UnaryElementwiseOp`]）を `a`・戻り値いずれも
    /// [`DeviceBuffer`] 常駐のまま計算する（イシュー #1584）。
    /// [`Self::binary_elementwise_device`] の単項版で契約・デフォルト
    /// 実装の設計方針は同一（数値は対応するホスト版〈`BackendOps::
    /// relu`／`exp`／`tanh`〉と同一カーネルにより bit 同一）。
    ///
    /// # デフォルト実装
    /// [`Self::binary_elementwise_device`] と同じ非破壊拡張パターン。
    /// 既定は [`BackendError::Unsupported`]。
    fn unary_elementwise_device(
        &self,
        _op: UnaryElementwiseOp,
        _a: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "unary_elementwise_device: default fail-safe (no device-resident elementwise \
             kernel available)"
                .into(),
        ))
    }

    /// 融合グラフ（#162 が検出した elementwise 連鎖・#163 が生成する
    /// カーネル）を 1 回のカーネル呼び出しで実行する（TASK-12.1d・#164）。
    ///
    /// `gemm_bias_act` と同型の非破壊拡張（デフォルトメソッド追加）。
    /// デフォルト実装は `BackendError::Unsupported` を返す fail-safe
    /// （既存 elementwise・reduction 未実装カーネルと同じ設計）であり、
    /// `fandhe_ai_autodiff::Tape` の実体化経路（`materialize_fallible`／
    /// `materialize_non_fallible`。`crates/autodiff/src/tape.rs`）は
    /// `Unsupported` を検出した場合に `leaves` を使わず `self`（同じ
    /// `ops`）の per-op メソッド（`add`／`mul`／`relu`／`exp`／`tanh`）へ
    /// 逐次フォールバックする契約（`docs/fusion-graph-design.md` §3.4・
    /// §3.5.2・§3.5.3）。CPU 融合実行の提供元は `backend-cpu` 側の
    /// `run_fused` オーバーライド（#163 のスコープ。本イシュー〈#164〉
    /// 時点では #163 が未マージのため、CPU 側も本デフォルト実装のまま
    /// フォールバックする）。CUDA／Metal は融合カーネル生成が未実装の間
    /// このデフォルトへフォールバックする。
    fn run_fused(
        &self,
        _plan: &FusionPlan,
        _leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "run_fused: default fail-safe (no fusion kernel available)".into(),
        ))
    }

    /// REQ-14 の明示解放 API（イシュー #1018 ツリー・#1019 設計・#1020
    /// CUDA 実装・#1021 Metal 実装）。このバックエンドのデバイスメモリ
    /// プールがアイドル保持しているバッファを全て解放する。CUDA で
    /// `has_async_alloc()` が真の環境では、自作プール層の解放に加え
    /// driver 側 memory pool のトリム（`cuMemPoolTrimTo(0)` 相当）・
    /// 2 回の対象 stream 同期を内部で行う（`docs/device-memory-pool-
    /// design.md` §3.6 (2) の 4 フェーズ）。Metal は driver トリムを
    /// 持たないためフェーズが少ない（同 doc §3.6 (2)「バックエンド別の
    /// 該当フェーズ」表参照）。
    ///
    /// `crate::pool::PooledMemory::release_all_pooled`（`MemoryOps`
    /// デコレータ側の解放 API）とは別経路であり、本メソッドはホット
    /// パス確保（`backend_ops::BackendOps` 経由の GEMM／elementwise／
    /// softmax カーネル）が使う `SizeClassPool`（`pool_core.rs`）を
    /// 対象とする。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `Ok(())`（プールを持たないバックエンドは解放対象なし。
    /// fail-open ではなく「対象が存在しないため自明に成功」という
    /// 意味）。CPU バックエンドは常にこのデフォルトのまま（本イシューの
    /// 対象外。#1026）。CUDA（`backend-cuda::CudaBackendOps`）／Metal
    /// は同 doc §3.6 (2) の契約で実カーネルへオーバーライドする
    /// （`crates/backend-cuda/src/ops.rs`・`crates/backend-metal/src/
    /// ops.rs`）。
    ///
    /// # エラー
    /// `Err` は同 doc §3.6 (2)「バックエンド別の該当フェーズ」表が定める
    /// フェーズのいずれかの失敗を表す。実際に到達しうる `Err` の種別・
    /// 個数はバックエンドごとに異なるため（例: Metal はフェーズ (ii) が
    /// 失敗しない設計のため実質的にフェーズ (i) 失敗の 1 種類のみへ
    /// 到達しうる。CPU は本メソッドを常にデフォルト実装のまま使うため
    /// 到達しない）、本 doc comment では数を明記しない（正本は同 doc
    /// §3.6 (2) の表）。黙殺・panic は禁止する（fail-closed。
    /// `.claude/rules/coding-rust.md`）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// デバイスメモリプールの統計スナップショット（診断用。イシュー
    /// #1020・#1021）。[`PoolStats`]（POD。内部ハンドル表現を一切
    /// 含まない）のみを返す。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `None`（プールを持たないバックエンド。`backend-cpu`）。
    /// `backend-cuda`・`backend-metal` は `Some(stats)` を返す
    /// オーバーライドを持つ。
    fn device_memory_pool_stats(&self) -> Option<PoolStats> {
        None
    }

    /// `C = A @ B` を [`Self::gemm`] と**同一カーネル・同一選択ロジック**
    /// で計算し（`output` を返す場合は `gemm` と bit 同一）、`C` の論理
    /// 領域 `m×n` 全要素和（checksum）を `f64` アキュムレータ（Metal は
    /// Neumaier 補償和で `f64` 相当）・固定順序で決定的に求める（イシュー
    /// #1339・親イシュー #1338）。
    ///
    /// `readout` が [`ChecksumReadout::ChecksumOnly`] のとき、GPU
    /// バックエンド（CUDA／Metal）は `C` をホストへ download せず
    /// checksum（8 バイト）のみ読み戻す契約とする（framework-compare の
    /// 毎反復計測窓から `host_copy` を排除する目的。`docs/perf/
    /// device-checksum-readback-ab.md` 参照）。[`ChecksumReadout::
    /// WithOutput`] のときは続けて `C` も download し
    /// [`GemmChecksum::output`] へ格納する。
    ///
    /// # デフォルト実装
    ///
    /// 本メソッドは `gemm_bias_act`・`linear_forward_device` と同じ
    /// 非破壊拡張パターン（`BackendOps` トレイトへのデフォルトメソッド
    /// 追加。公開 API 非破壊はガードレール条件・`.claude/rules/
    /// security.md`）であり、既定は [`BackendError::Unsupported`] を
    /// 返す fail-safe とする。`fandhe_ai_autodiff::var::Var::
    /// matmul_checksum` は `Unsupported` を透過し呼び出し元（framework-
    /// compare のハーネス）が判定する契約とする（判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。`backend-cpu`・`backend-cuda`・
    /// `backend-metal` の各実装はこのデフォルトをオーバーライドする。
    fn gemm_checksum(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
        _readout: ChecksumReadout,
    ) -> Result<GemmChecksum, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_checksum: default fail-safe (no device-side reduction kernel available)".into(),
        ))
    }

    /// 行方向 RMSNorm（`x · rsqrt(mean(x²) + eps) · w`。`w` が `None` の
    /// 場合は乗算をスキップ。イシュー #1596）の独立エントリ。既存の
    /// [`Self::run_fused`] 経由（`match_rmsnorm_plan` の canonical プラン
    /// 一致限定・`mean` 化なし・`eps` なし・`weight` なし）とは別に、
    /// `fandhe_ai_autodiff::var::Var::rms_norm` が直接呼べる入口を
    /// 提供する（`docs/compat-feature-gap.md` §2.7）。
    ///
    /// **契約**: 正規化軸は常に最終軸（[`crate::ops_shape::
    /// row_norm_layout`] が `(rows, hidden)` を導出する）。`weight` を
    /// 渡す場合は shape `[hidden]` を要求する（呼び出し元
    /// `Var::rms_norm` が事前検査する）。戻り値の shape は入力 `x` と
    /// 恒等（正規化は形状を変えない）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`Var::
    /// rms_norm` は `Unsupported` のときのみホスト参照実装
    /// （`eval::rmsnorm_rows`）へフォールバックする（それ以外のエラーは
    /// 伝播する。判定迂回経路を作らない。`.claude/rules/security.md`
    /// A08）。CPU／CUDA／Metal はいずれもこのデフォルトを既存の
    /// RMSNorm 行カーネル（`run_fused` の RMSNorm 一致経路が使うものと
    /// 同一のカーネル実体）でオーバーライドする。
    fn rmsnorm(
        &self,
        _x: &Tensor<f32>,
        _weight: Option<&Tensor<f32>>,
        _eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "rmsnorm: default fail-safe (no fused RMSNorm kernel available)".into(),
        ))
    }

    /// 行方向 LayerNorm（`(x − mean(x)) · rsqrt(var(x) + eps) · w + b`。
    /// `w`／`b` はそれぞれ `None` の場合は対応する演算をスキップ。
    /// 分散は biased（÷N）。イシュー #1596）の独立エントリ。[`Self::
    /// rmsnorm`] と同じ最終軸限定契約・非破壊拡張・フォールバック規律
    /// に従う（`Var::layer_norm` は `Unsupported` のときのみ
    /// `eval::layer_norm_rows` へフォールバックする）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::rmsnorm`] と同じ非破壊拡張・fail-safe。CPU／CUDA／Metal
    /// はいずれも本イシューで新設する専用カーネルでこのデフォルトを
    /// オーバーライドする（既存の融合 `run_fused` canonical プランには
    /// LayerNorm 一致経路を追加しない。LayerNorm は本エントリ経由でのみ
    /// 到達する）。
    fn layer_norm(
        &self,
        _x: &Tensor<f32>,
        _weight: Option<&Tensor<f32>>,
        _bias: Option<&Tensor<f32>>,
        _eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "layer_norm: default fail-safe (no fused LayerNorm kernel available)".into(),
        ))
    }

    /// LSTM セルの pointwise 段（イシュー #1647・設計 `docs/autodiff-
    /// rnn-cell-tape-design.md` 決定 1・決定 1b・決定 5）。融合 GEMM
    /// `pre = x·W_ih + b_ih + h_prev·W_hh + b_hh`（`[B,4H]`。列ブロック
    /// 順 `i,f,g,o`）を受け取り、4 ゲートの活性化・セル状態更新・隠れ
    /// 状態を 1 呼び出しで計算する。
    ///
    /// `pre` は `[B, 4H]`（`H` は `c_prev` の列数から導出）、`c_prev` は
    /// `[B, H]`。戻り値 `gates` は活性化後の `i,f,g,o`（`[B, 4H]`。
    /// `Op::LstmCell`／`Op::LstmHidden` の VJP が backward で読む
    /// payload そのもの）、`c` は新セル状態 `[B, H]`、`h` は新隠れ状態
    /// `[B, H]`。
    ///
    /// `c = f·c_prev + i·g`（FMA 契約統一・`.claude/rules/coding-rust.md`）、
    /// `h = o·tanh(c)`。
    ///
    /// `fandhe_ai_autodiff::var::Var::lstm_cell` から呼ばれ、
    /// [`BackendError::Unsupported`] のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::lstm_pointwise`）へフォールバックする
    /// （A08。判定迂回経路を作らない）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gemm_bias_act`] と同じ非破壊拡張パターン。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe。CPU／CUDA／Metal
    /// の各実装がこのデフォルトをオーバーライドする。
    fn lstm_pointwise(
        &self,
        _pre: &Tensor<f32>,
        _c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        Err(BackendError::Unsupported(
            "lstm_pointwise: default fail-safe (no fused LSTM pointwise kernel available)".into(),
        ))
    }

    /// `Op::LstmHidden` の VJP 補助（決定 1b・決定 1b 追記）。`c`
    /// （現在のセル状態）・`gate_o`（forward 記録済みの o ゲート値）・
    /// `dh`（上流勾配）から、o ゲートの pre-activation 勾配
    /// `d_pre_o = dh·tanh(c)·o·(1−o)` と、`cell`（`Op::LstmCell`）へ
    /// 伝播するセル状態勾配 `dc = dh·o·(1−tanh(c)²)` を計算する。
    ///
    /// `c`／`gate_o`／`dh` はいずれも `[B, H]`。戻り値 `(d_pre_o, dc)`
    /// も `[B, H]`。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn lstm_hidden_backward(
        &self,
        _c: &Tensor<f32>,
        _gate_o: &Tensor<f32>,
        _dh: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        Err(BackendError::Unsupported(
            "lstm_hidden_backward: default fail-safe (no fused LSTM hidden backward kernel available)"
                .into(),
        ))
    }

    /// `Op::LstmCell` の VJP 補助（決定 1b）。forward 記録済みの
    /// `gates_ifg`（`i,f,g` の活性化後値。`[B, 3H]`）・`c_prev`
    /// （`[B, H]`）・上流のセル状態勾配 `dc`（`[B, H]`。`Op::LstmHidden`
    /// からの寄与と次 step の `dc_prev` の fan-in 合算済み）から、
    /// `i,f,g` 3 ゲートの pre-activation 勾配 `d_pre_ifg`（`[B, 3H]`）と
    /// 前セル状態への勾配 `dc_prev = dc·f`（`[B, H]`）を計算する。
    ///
    /// `d_pre_i = dc·g·i·(1−i)`、`d_pre_f = dc·c_prev·f·(1−f)`、
    /// `d_pre_g = dc·i·(1−g²)`。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn lstm_cell_backward(
        &self,
        _gates_ifg: &Tensor<f32>,
        _c_prev: &Tensor<f32>,
        _dc: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        Err(BackendError::Unsupported(
            "lstm_cell_backward: default fail-safe (no fused LSTM cell backward kernel available)"
                .into(),
        ))
    }

    /// GRU セルの pointwise 段（決定 1c・決定 5。`reset_after=True`
    /// 規約）。`pre_i = x·W_ih + b_ih`・`pre_h = h_prev·W_hh + b_hh`
    /// （いずれも `[B, 3H]`。列ブロック順 `r,z,n`）と前隠れ状態
    /// `h_prev`（`[B, H]`）を受け取り、`r,z` ゲート・`n`（新候補）・
    /// 新隠れ状態を計算する。
    ///
    /// `r = σ(pre_i_r + pre_h_r)`、`z = σ(pre_i_z + pre_h_z)`、
    /// `q = pre_h_n`（decision 1c: GEMM 再計算を避けるため payload に
    /// 保持する再帰側アフィン値）、`n = tanh(r·q + pre_i_n)`、
    /// `h = z·h_prev + (1−z)·n`。
    ///
    /// 戻り値 `gates` は活性化後の `r,z,n`（`[B, 3H]`）、`q` は再帰側
    /// アフィン値（`[B, H]`。決定 1c）、`h` は新隠れ状態（`[B, H]`）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn gru_pointwise(
        &self,
        _pre_i: &Tensor<f32>,
        _pre_h: &Tensor<f32>,
        _h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        Err(BackendError::Unsupported(
            "gru_pointwise: default fail-safe (no fused GRU pointwise kernel available)".into(),
        ))
    }

    /// `Op::GruCell` の VJP 補助。forward 記録済みの `gates_rzn`
    /// （`[B, 3H]`）・`q`（決定 1c の再帰側アフィン値。`[B, H]`）・
    /// `h_prev`（`[B, H]`）・上流勾配 `dh`（`[B, H]`）から、`W_ih` 側
    /// pre-activation 勾配 `d_pre_i`（`[B, 3H]`）・`W_hh` 側
    /// pre-activation 勾配 `d_pre_h`（`[B, 3H]`）・`h_prev` への直接
    /// 勾配 `dh_prev_direct = dh·z`（`[B, H]`）を計算する。
    ///
    /// `dn = dh·(1−z)`、`dz = dh·(h_prev−n)`、`d_pre_n = dn·(1−n²)`、
    /// `dr = d_pre_n·q`、`d_pre_r = dr·r·(1−r)`、
    /// `d_pre_z = dz·z·(1−z)`。`d_pre_i = [d_pre_r, d_pre_z, d_pre_n]`、
    /// `d_pre_h = [d_pre_r, d_pre_z, d_pre_n·r]`（`n` の `q` に対する
    /// 偏微分が `r` であるため、`W_hh` 側の n 列ブロックのみ追加で `r`
    /// を乗じる）。呼び出し元（`grad::vjp`）は `d_pre_h` を用いて
    /// `dh_prev = dh_prev_direct + d_pre_h·W_hhᵀ` を合成する。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn gru_backward(
        &self,
        _gates_rzn: &Tensor<f32>,
        _q: &Tensor<f32>,
        _h_prev: &Tensor<f32>,
        _dh: &Tensor<f32>,
    ) -> Result<GruBackwardOutput, BackendError> {
        Err(BackendError::Unsupported(
            "gru_backward: default fail-safe (no fused GRU backward kernel available)".into(),
        ))
    }

    /// `A^{-1}`（`A: [n,n]`）。イシュー #1621・`docs/autodiff-linalg-design.md`。
    ///
    /// # デフォルト実装
    /// `gemm_checksum` と同じ非破壊拡張パターン。既定は
    /// [`BackendError::Unsupported`]（GPU バックエンドは本イシュー時点で
    /// 未実装。`fandhe_ai_autodiff::var::Var::inv` がこの `Unsupported`
    /// のみをホスト参照実装 `eval::linalg::inv` へフォールバックし、
    /// それ以外のエラー〈特異行列の `InvalidArgument` 等〉は伝播する）。
    /// `A` が特異（ピボットが厳密 0）の場合は `BackendError::
    /// InvalidArgument` を返す契約とする。
    fn linalg_inv(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_inv: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// `A X = B` を解く（`A: [n,n]`・`B: [n,k]` → `X: [n,k]`）。
    /// イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// `A` が特異の場合は [`BackendError::InvalidArgument`]。
    fn linalg_solve(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_solve: default fail-safe (no device-side linear-algebra kernel available)"
                .into(),
        ))
    }

    /// `det(A)`（`A: [n,n]` → スカラー `[]`）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// 特異行列は `0.0` を返す（`torch.linalg.det` と同じくエラーに
    /// しない。設計文書 §3.5「エラー分類」）。
    fn linalg_det(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_det: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// Cholesky 分解（`A: [n,n]`〈対称正定値。下三角のみ読む〉→
    /// `L: [n,n]`〈下三角、`A = L Lᵀ`〉）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// 非正定値（対角が非有限または非正）の場合は
    /// [`BackendError::InvalidArgument`]。
    fn linalg_cholesky(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_cholesky: default fail-safe (no device-side linear-algebra kernel available)"
                .into(),
        ))
    }

    /// reduced QR 分解（`A: [m,n]` → [`QrFactors`]）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    fn linalg_qr(&self, _a: &Tensor<f32>) -> Result<QrFactors, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_qr: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// reduced SVD（`A: [m,n]` → [`SvdFactors`]）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// 反復が収束しない場合は [`BackendError::InvalidArgument`]。
    fn linalg_svd(&self, _a: &Tensor<f32>) -> Result<SvdFactors, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_svd: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// 行列ノルム（`A: [m,n]`・`ord` → スカラー `[]`）。イシュー #1621。
    /// `ord` が [`MatrixNormOrd::Nuc`]／[`MatrixNormOrd::Spectral`] の
    /// 場合、実装内部で特異値分解を用いてよい（`Var` 側で `svd` ノードを
    /// 別途合成しない。設計文書 §3.2）。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    fn linalg_matrix_norm(
        &self,
        _a: &Tensor<f32>,
        _ord: MatrixNormOrd,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_matrix_norm: default fail-safe (no device-side linear-algebra kernel \
             available)"
                .into(),
        ))
    }
}

/// [`BackendOps::lstm_pointwise`] の戻り値（イシュー #1647）。
///
/// `gates` は活性化後の `i,f,g,o`（`[B, 4H]`。`Op::LstmCell`／
/// `Op::LstmHidden` の VJP が backward で参照する payload そのもの）、
/// `c` は新セル状態、`h` は新隠れ状態（いずれも `[B, H]`）。他クレート
/// （`backend-cpu`／`backend-cuda`／`backend-metal`）が構築するため
/// `#[non_exhaustive]` は付けない（フィールド追加は破壊的変更として
/// 扱う）。
#[derive(Debug, Clone)]
pub struct LstmPointwiseOutput {
    pub gates: Tensor<f32>,
    pub c: Tensor<f32>,
    pub h: Tensor<f32>,
}

/// [`BackendOps::gru_pointwise`] の戻り値（イシュー #1647）。
///
/// `gates` は活性化後の `r,z,n`（`[B, 3H]`）、`q` は決定 1c の再帰側
/// アフィン値（`[B, H]`。GEMM 再計算なしで `∂n/∂r` を復元するための
/// payload）、`h` は新隠れ状態（`[B, H]`）。
#[derive(Debug, Clone)]
pub struct GruPointwiseOutput {
    pub gates: Tensor<f32>,
    pub q: Tensor<f32>,
    pub h: Tensor<f32>,
}

/// 複数の `&dyn BackendOps` を横断して `device` に一致する実装を選択する。
///
/// `device::select_from`（TASK-1.9a）と同型の注入式ディスパッチ:
/// `tensor-core` は `backend-cpu`／`backend-cuda`／`backend-metal` を直接
/// 参照できないため、呼び出し側（結線を担う上位クレート・テスト）が
/// `ops` を注入する。本関数こそが受け入れ条件「同一コードで 3 バック
/// エンドのカーネルが呼び分けられる」の直接の実装であり、`device` の
/// variant にのみ基づいて対応実装を返す（形状・HW ヒューリスティクスは
/// 一切持ち込まない。TASK-11.2b・#68 のスコープ）。
///
/// 対応する実装が `ops` に含まれない場合は
/// [`BackendError::DeviceUnavailable`] を返す（`device::select_from` と
/// 同じエラー variant・同じ意味論。「対応 provider／ops 未登録」を表す）。
pub fn ops_for<'a>(
    ops: &[&'a dyn BackendOps],
    device: Device,
) -> Result<&'a dyn BackendOps, BackendError> {
    ops.iter()
        .find(|candidate| candidate.device() == device)
        .copied()
        .ok_or_else(|| {
            BackendError::DeviceUnavailable(format!(
                "no BackendOps registered for device {device:?}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::BufferHandle;
    use crate::error::ShapeError;
    use std::any::Any;

    /// テスト専用のモック `BackendOps`。実バックエンドに依存せず
    /// `ops_for` の選択ロジックを検証するために `tensor-core` 内で定義
    /// する（実バックエンドの検証は各バックエンドクレートの結合テスト
    /// で行う。`device` モジュールの `MockProvider` と同じ位置付け）。
    struct MockOps(Device);

    impl BackendOps for MockOps {
        fn device(&self) -> Device {
            self.0
        }

        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: gemm".into()))
        }

        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: add".into()))
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: mul".into()))
        }

        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: relu".into()))
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: max".into()))
        }
    }

    /// `gemm_bias_act` のデフォルト実装（非融合合成）を数値検証するための
    /// naive 計算モック。`MockOps`（常に `Unsupported`）と異なり `gemm`／
    /// `add`／`relu` を実際に計算する（行方向ブロードキャストのみ対応する
    /// 簡易 `add`。テスト用途のため `Tensor::get`／strided view には
    /// 対応しない）。
    struct ComputingMockOps;

    impl BackendOps for ComputingMockOps {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let n = b.shape()[1];
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc = a_data[i * k + p].mul_add(b_data[p * n + j], acc);
                    }
                    out[i * n + j] = acc;
                }
            }
            Tensor::new(out, &[m, n]).map_err(BackendError::ShapeMismatch)
        }

        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            // テストで使う形状のみ対応: `a: [m, n]`・`b: [n]`（行方向
            // ブロードキャスト）または同一 shape。
            let a_shape = a.shape().to_vec();
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let out = if b.shape() == a.shape() {
                a_data
                    .iter()
                    .zip(b_data)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<_>>()
            } else if b.shape().len() == 1 && a_shape.len() == 2 && b.shape()[0] == a_shape[1] {
                let n = a_shape[1];
                a_data
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| x + b_data[idx % n])
                    .collect::<Vec<_>>()
            } else {
                return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                    expected: a_shape.len(),
                    actual: b.shape().len(),
                }));
            };
            Tensor::new(out, &a_shape).map_err(BackendError::ShapeMismatch)
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: mul".into()))
        }

        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let data = a.as_slice().expect("test: a must be contiguous");
            let out = data.iter().map(|x| x.max(0.0)).collect::<Vec<_>>();
            Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: max".into()))
        }
    }

    /// object-safe であることの型検査を兼ねる（`Box<dyn BackendOps>` が
    /// 構築できることをコンパイル時に確認する）。
    fn assert_object_safe(_ops: &dyn BackendOps) {}

    #[test]
    fn gemm_bias_act_default_matches_manual_composition() {
        let ops = ComputingMockOps;
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let bias = Tensor::new(vec![-100.0, 1.0], &[2]).unwrap();

        // A@B = [[19, 22], [43, 50]] → + bias [-100, 1] → [[-81, 23], [-57, 51]]
        // → relu → [[0, 23], [0, 51]]
        let out = ops
            .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
            .expect("gemm_bias_act should succeed");
        assert_eq!(out.as_slice().unwrap(), &[0.0, 23.0, 0.0, 51.0]);
    }

    #[test]
    fn gemm_bias_act_default_no_bias_no_act_matches_gemm() {
        let ops = ComputingMockOps;
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let plain_gemm = ops.gemm(&a, &b).unwrap();
        let fused = ops
            .gemm_bias_act(&a, &b, None, Activation::None)
            .expect("gemm_bias_act should succeed");
        assert_eq!(
            plain_gemm.as_slice().unwrap(),
            fused.as_slice().unwrap(),
            "bias=None・act=None は gemm と同一結果のはず"
        );
    }

    #[test]
    fn gemm_bias_act_default_propagates_unsupported_from_composed_ops() {
        // `MockOps` は `gemm` 自体が `Unsupported` を返すため、
        // デフォルト実装が最初のステップのエラーをそのまま伝播することを
        // 検証する（GPU バックエンドが GEMM 自体未実装の場合の fail-safe。
        // elementwise 未実装〈`add`/`relu` が `Unsupported`〉の伝播は
        // `backend-cuda`/`backend-metal` の結合テスト側で検証する）。
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = ops.gemm_bias_act(&a, &b, None, Activation::Relu);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    #[test]
    fn ops_for_dispatches_to_matching_device() {
        let cpu = MockOps(Device::Cpu);
        let cuda = MockOps(Device::Cuda(0));
        let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

        let selected = ops_for(&ops, Device::Cuda(0)).expect("cuda ops registered");
        assert_eq!(selected.device(), Device::Cuda(0));
        assert_object_safe(selected);

        let selected = ops_for(&ops, Device::Cpu).expect("cpu ops registered");
        assert_eq!(selected.device(), Device::Cpu);
    }

    #[test]
    fn ops_for_missing_device_returns_device_unavailable() {
        let cpu = MockOps(Device::Cpu);
        let ops: Vec<&dyn BackendOps> = vec![&cpu];

        // `ops_for` の `Ok` 側は `&dyn BackendOps` を含み `Debug` を実装
        // しないため `expect_err` は使わず、`is_err`／`matches!` で
        // `Err` 経路のみ検査する。
        let result = ops_for(&ops, Device::Cuda(0));
        assert!(result.is_err());
        assert!(matches!(result, Err(BackendError::DeviceUnavailable(_))));
    }

    #[test]
    fn unsupported_error_carries_shape_error_independently() {
        // `BackendError::Unsupported` が既存 variant（`ShapeMismatch` 等）と
        // 独立して構築・表示できることを確認する（非破壊追加の検証）。
        let err = BackendError::Unsupported("elementwise add on cuda".into());
        assert!(err.to_string().contains("elementwise add on cuda"));

        let shape_err = BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: 1,
        });
        assert!(!shape_err.to_string().is_empty());
    }

    #[test]
    fn run_fused_default_returns_unsupported() {
        // `run_fused`（TASK-12.1d・#164）のデフォルト実装は `Unsupported`
        // を返す fail-safe（`gemm_bias_act` 等の既存 elementwise・
        // reduction 未実装カーネルと同型の設計。backend_ops.rs 冒頭コメ
        // ント参照）。`MockOps` はこのデフォルトを override しない。
        let ops = MockOps(Device::Cpu);
        // `from_ops`（`fusion::plan`。TASK-12.1c・#163）は「`Input` エント
        // リのみで elementwise ノードが 1 個も無い」プランを
        // `FusionPlanError::NoElementwiseNode` として拒否する契約
        // （融合する意味が無いため。`plan.rs` ドキュメント参照）ため、本
        // テストは最小の elementwise ノード（`Relu`）を 1 個含む有効な
        // プランを使う。
        let plan = crate::fusion::FusionPlan::from_ops(
            vec![
                crate::fusion::FusedOpKind::Input { leaf_index: 0 },
                crate::fusion::FusedOpKind::Relu { input: 0 },
            ],
            vec![4],
            crate::dispatch::DType::F32,
            1,
        )
        .expect("from_ops should succeed for a minimal single-op plan");
        let leaf = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let leaves: Vec<&Tensor<f32>> = vec![&leaf];
        let result = ops.run_fused(&plan, &leaves);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// テスト専用の最小 `BufferHandle`（イシュー #1017・
    /// `sgd_step_device_tracked_default_delegates_to_sgd_step_device`
    /// が `DeviceBuffer<f32>` を構築するためだけに使う。データの実体は
    /// 持たず downcast のためだけの空ハンドル）。
    #[derive(Debug)]
    struct EmptyHandle;

    impl BufferHandle for EmptyHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn empty_device_buffer(device: Device) -> DeviceBuffer<f32> {
        DeviceBuffer::new(device, vec![1], Box::new(EmptyHandle))
    }

    /// [`BackendOps::sgd_step_device_tracked`] のデフォルト実装が
    /// `token` を無視して [`BackendOps::sgd_step_device`] へそのまま
    /// 委譲することを確認する（イシュー #1017 の非破壊拡張ガード。
    /// `MockOps` はいずれのメソッドもオーバーライドしていないため、
    /// 両者が同一の `Unsupported` メッセージを返すことで委譲を検証する）。
    #[test]
    fn sgd_step_device_tracked_default_delegates_to_sgd_step_device() {
        let ops = MockOps(Device::Cpu);
        let mut param = empty_device_buffer(Device::Cpu);
        let grad = empty_device_buffer(Device::Cpu);
        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };
        let token = DispatchFailureCell::new();

        let direct = ops.sgd_step_device(&mut param, &grad, None, &config);
        let tracked = ops.sgd_step_device_tracked(&mut param, &grad, None, &config, &token);

        match (direct, tracked) {
            (Err(BackendError::Unsupported(a)), Err(BackendError::Unsupported(b))) => {
                assert_eq!(a, b);
            }
            other => panic!("expected both to return the same Unsupported error: {other:?}"),
        }
        // デフォルト委譲は token に一切触れない。
        assert!(!token.is_set());
    }

    /// [`BackendOps::linear_forward_device`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1028）。デバイス常駐の入出力を扱えないバックエンド（`MockOps`）
    /// が黙示のホストフォールバックへ落ちず、明示的に拒否することが
    /// 受け入れ条件の中核（`docs/inference-forward-fixed-cost-design.md`
    /// §3.2 のフォールバック契約）。
    #[test]
    fn linear_forward_device_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = empty_device_buffer(Device::Cpu);
        let w_buf = empty_device_buffer(Device::Cpu);
        let w_view = DeviceBufferView::new(&w_buf, 0, &[1]).unwrap();

        let result = ops.linear_forward_device(&a, w_view, None, Activation::None);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::binary_elementwise_device`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1584。`linear_forward_device_default_is_unsupported` と同型の
    /// ガード）。
    #[test]
    fn binary_elementwise_device_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = empty_device_buffer(Device::Cpu);
        let b = empty_device_buffer(Device::Cpu);

        let result = ops.binary_elementwise_device(BinaryElementwiseOp::Add, &a, &b);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::unary_elementwise_device`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1584。`binary_elementwise_device_default_is_unsupported` と同型）。
    #[test]
    fn unary_elementwise_device_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = empty_device_buffer(Device::Cpu);

        let result = ops.unary_elementwise_device(UnaryElementwiseOp::Relu, &a);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::mse_loss`]／[`BackendOps::mse_loss_backward`] の
    /// 既定実装が両方とも fail-safe（[`BackendError::Unsupported`]）を
    /// 返すことを確認する（イシュー #1045。`run_fused_default_returns_
    /// unsupported`・`linear_forward_device_default_is_unsupported` と
    /// 同型のガード）。`MockOps` はいずれのメソッドもオーバーライドして
    /// いないため、融合カーネル未実装のバックエンドが黙示のホスト
    /// フォールバックへ落ちず明示的に拒否することが受け入れ条件の中核。
    #[test]
    fn mse_loss_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let pred = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let target = Tensor::new(vec![0.0, 0.0], &[2]).unwrap();

        let forward = ops.mse_loss(&pred, &target, MseReduction::Mean);
        let backward = ops.mse_loss_backward(&pred, &target, 1.0);

        assert!(matches!(forward, Err(BackendError::Unsupported(_))));
        assert!(matches!(backward, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::softmax`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する
    /// （イシュー #1594。`mse_loss_default_is_unsupported` と同型）。
    #[test]
    fn softmax_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = ops.softmax(&x, 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::log_softmax`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1594）。
    #[test]
    fn log_softmax_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = ops.log_softmax(&x, 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::concat`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1598）。
    #[test]
    fn concat_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0], &[2]).unwrap();

        let result = ops.concat(&[&a, &b], 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::where_cond`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1637）。
    #[test]
    fn where_cond_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let cond = Tensor::new(vec![1.0, 0.0], &[2]).unwrap();
        let a = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0], &[2]).unwrap();

        let result = ops.where_cond(&cond, &a, &b);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::masked_fill`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1637）。
    #[test]
    fn masked_fill_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let mask = Tensor::new(vec![1.0, 0.0], &[2]).unwrap();

        let result = ops.masked_fill(&x, &mask, -1.0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::captured_segment_key`]／[`BackendOps::
    /// run_captured_sgd_step_segment`] の既定実装が非破壊拡張の
    /// fail-safe 契約（前者は `Ok(None)`・後者は `Err(Unsupported)`）を
    /// 満たすことを確認する（イシュー #1349）。`MockOps` は CUDA Graph
    /// 機構を持たないため、graph 非対応バックエンド・opt-in OFF の既定
    /// 状態を模す。
    #[test]
    fn captured_segment_key_default_is_none() {
        let ops = MockOps(Device::Cpu);
        let buf = empty_device_buffer(Device::Cpu);
        let key = ops.captured_segment_key(&[&buf], 0);
        assert!(matches!(key, Ok(None)));
    }

    /// [`BackendOps::run_captured_sgd_step_segment`] の既定実装は区間
    /// 本体（SGD 更新）を一切実行せずに `Unsupported` を返す（呼び出し元
    /// の契約「`captured_segment_key` が `Some` を返したときのみ呼ぶ」が
    /// 守られていれば到達しない経路だが、契約違反時も二重実行等の副作用
    /// を起こさないことを固定する）。`param` が変化しないことで「本体
    /// 未実行」を確認する（codex-review P0 指摘対応で `body` クロージャ
    /// を廃したため、旧テストの `call_count` の代わりに副作用の不在を
    /// 直接観測する）。
    #[test]
    fn run_captured_sgd_step_segment_default_is_unsupported_and_has_no_effect() {
        let ops = MockOps(Device::Cpu);
        let key = SegmentKey {
            generation: 0,
            config_key: 0,
            resources: vec![SegmentResource { addr: 0, numel: 0 }],
        };
        let mut param = empty_device_buffer(Device::Cpu);
        let grad = empty_device_buffer(Device::Cpu);
        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };
        let token = DispatchFailureCell::new();
        let result =
            ops.run_captured_sgd_step_segment(key, &mut param, &grad, None, &config, &token);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// `&dyn BackendOps` 経由でも新規デフォルトメソッドを呼べる
    /// （object-safety が壊れていない）ことを確認する（`run_captured_
    /// sgd_step_segment` が object-safe な形で trait に追加できている
    /// ことの回帰ガード）。
    #[test]
    fn captured_segment_methods_are_object_safe() {
        let ops = MockOps(Device::Cpu);
        let dyn_ops: &dyn BackendOps = &ops;
        let buf = empty_device_buffer(Device::Cpu);
        assert!(matches!(dyn_ops.captured_segment_key(&[&buf], 0), Ok(None)));
    }
    /// [`BackendOps::gemm_checksum`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1339。`mse_loss_default_is_unsupported` と同型のガード）。
    #[test]
    fn gemm_checksum_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0], &[1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], &[2, 1]).unwrap();

        let result = ops.gemm_checksum(&a, &b, ChecksumReadout::ChecksumOnly);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::rmsnorm`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1596。`mse_loss_default_is_unsupported` と同型）。
    #[test]
    fn rmsnorm_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

        let result = ops.rmsnorm(&x, None, 1e-6);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::layer_norm`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1596）。
    #[test]
    fn layer_norm_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

        let result = ops.layer_norm(&x, None, None, 1e-5);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// RNN／LSTM／GRU セル演算（イシュー #1647）の 5 メソッドすべてが
    /// 既定実装で `BackendError::Unsupported` を返すことを確認する
    /// （`gemm_checksum_default_is_unsupported` と同型のガード。
    /// `fandhe_ai_autodiff::var::{rnn_cell,lstm_cell,gru_cell}` はこの
    /// 契約に依存してホスト参照実装〈`eval.rs`〉へフォールバックする）。
    #[test]
    fn rnn_cell_ops_default_are_unsupported() {
        let ops = MockOps(Device::Cpu);
        let b = 2usize;
        let hidden = 3usize;
        let pre4h = Tensor::new(vec![0.0f32; b * 4 * hidden], &[b, 4 * hidden]).unwrap();
        let pre3h = Tensor::new(vec![0.0f32; b * 3 * hidden], &[b, 3 * hidden]).unwrap();
        let bh = Tensor::new(vec![0.0f32; b * hidden], &[b, hidden]).unwrap();

        assert!(matches!(
            ops.lstm_pointwise(&pre4h, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.lstm_hidden_backward(&bh, &bh, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.lstm_cell_backward(&pre3h, &bh, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.gru_pointwise(&pre3h, &pre3h, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.gru_backward(&pre3h, &bh, &bh, &bh),
            Err(BackendError::Unsupported(_))
        ));
    }

    /// [`BackendOps::linalg_*`]（7 メソッド）の既定実装がいずれも
    /// fail-safe（[`BackendError::Unsupported`]）を返すことを確認する
    /// （イシュー #1621。`gemm_checksum_default_is_unsupported` と同型の
    /// 回帰ガード）。
    #[test]
    fn linalg_defaults_are_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], &[2, 1]).unwrap();

        assert!(matches!(
            ops.linalg_inv(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_solve(&a, &b),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_det(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_cholesky(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_qr(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_svd(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_matrix_norm(&a, MatrixNormOrd::Fro),
            Err(BackendError::Unsupported(_))
        ));
    }
}
