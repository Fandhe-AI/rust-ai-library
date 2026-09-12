//! CPU バックエンド（参照実装）。
//!
//! `tensor-core` の演算グラフノードを CPU カーネルへ変換して実行する。バックエンド切替は
//! feature フラグなしの cfg ベースを基本とし（PoC-v2-5 実証構成。REQ-2）、本バックエンドは
//! 無条件で有効化される数値一致の参照点となる。並列化は `backend-cpu` 固有の許容依存である
//! `rayon` を用いる（PoC-v2-1 で naive/blocked 比 約 6〜8.5 倍改善を実測。
//! `.claude/rules/deps-policy.md`）。
//!
//! `backend-cuda` / `backend-metal` との数値一致は統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」で検証する。丸め方針（FMA 契約）は GPU 側の既定 FMA 契約と揃えるため
//! `f32::mul_add` を用いる（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。
//! `.claude/rules/coding-rust.md`）。カーネルの手動境界検査は最適化を理由に省略しない（REQ-8）。
//!
//! TASK-1.6a（#21）で GEMM カーネル（[`gemm`] モジュール。naive / blocked / rayon 並列の
//! 3 段構成）を追加した。TASK-1.6b（#22）で elementwise カーネル（二項演算 `add`・`mul`、
//! 活性化 `relu`・`exp`・`tanh`）を追加した。TASK-1.6c（#23）で `reduction`（`sum`・`max`・
//! `mean`・軸指定 reduction）を追加した。TASK-1.6d（#24）で `rayon` 並列の粒度・
//! ブロックサイズ（[`gemm::BlockSizes`]）を実測チューニングし、PoC-v2-1 比の性能改善比
//! （naive/blocked 比 約 6〜8.5 倍）が本環境でも再現することを確認した
//! （計測記録: `docs/perf/cpu-gemm-rayon-tuning.md`）。TASK-1.6f（#184）で [`mod@gemm_blis`]
//! モジュール（BLIS/GotoBLAS2 5-loop model・`std::arch` intrinsics マイクロカーネル・
//! A/B packing）を追加した。TASK-1.6g（#185）で `gemm_blis`／`gemm_blis_parallel` の
//! マイクロカーネル選択をコンパイル時 cfg のみから実行時 CPU 機能検出（NEON／AVX2／
//! AVX-512。`gemm_blis::microkernel` の `Isa::detect`）による dispatch へ拡張した。
//! `gemm` モジュールの関数（naive/blocked/parallel/parallel_tuned）は #24 の段階比較の
//! 参照点として変更しない（公開 API 非破壊。`gemm_blis` は独立した新規追加でシグネチャも
//! #185 で変更しない）。`BackendOps` トレイトからの結線は TASK-1.9（#43）で行う
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.6）。
//!
//! TASK-1.9a（#44）で `device` モジュール（[`device::CpuDeviceProvider`]）を追加した。
//! `fandhe_ai_tensor_core::device::DeviceProvider` の CPU 実装であり、CUDA／Metal 実装
//! （`backend-cuda::device::CudaDeviceProvider`／`backend-metal::device::MetalDeviceProvider`）
//! と同一 trait で列挙・選択できることを `tests/device_provider_integration.rs` で検証する。
//!
//! TASK-2.2a（#53）で [`parity`] モジュール（REQ-2 統一複合判定ユーティリティ・
//! FMA 契約参照 matmul）を追加した。#54（CPU-CUDA ペア）・#55（CPU-Metal ペア）は
//! 本モジュールの `parity::compare`／`parity::assert_parity`／
//! `parity::matmul_reference_fma` を共通利用し、ペアごとに判定ロジックを
//! 重複実装しない想定である（`docs/spec/05-tasks.md` TASK-2.2）。
//!
//! TASK-1.9b（#45）で [`memory`] モジュール（[`memory::CpuMemory`]）を追加した。
//! `fandhe_ai_tensor_core::buffer::MemoryOps` の CPU 実装であり、`upload`/`download`/
//! `alloc_zeroed` は FFI を伴わず `Vec<f32>` の複製のみで完結する
//! （`backend-cuda::CudaMemory`／`backend-metal::MetalMemory` の数値一致の
//! 参照点。`.claude/rules/coding-rust.md` の「CPU 参照実装」方針）。
//! TASK-14.1a（#174）で `fandhe_ai_tensor_core::memory_stats::MemoryStats` を実装し、
//! 確保済みバイト数のピーク値を取得できるようにした（`CpuMemory` は
//! `Arc<AllocationTracker>` を共有する非 `Copy` 型に変更。CUDA/Metal への
//! 同フック組み込みは #175）。
//!
//! **破壊的変更（`backend-cpu` 0.2.0。PR #359 codex-review 指摘 P1 を受けて
//! ここに移行手引きを明記）**: 従来 `CpuMemory` はフィールドを持たない
//! unit struct（`Copy`）だったため `let mem = CpuMemory;` のような直接構築が
//! 可能だったが、本変更でトラッカー（`Arc<AllocationTracker>`）を保持する
//! ようになり `Copy` を外した。ワークスペース内に unit struct 構築や
//! `Copy` 依存箇所が無いことは確認済みだが、外部呼び出し側は次のとおり
//! 移行する:
//! - `CpuMemory` → [`CpuMemory::new()`]（または `CpuMemory::default()`。
//!   いずれも新規の計測系列を持つトラッカーを生成する）
//! - `Copy` 依存（暗黙コピーでの使い回し）→ `Clone`（`clone()` は
//!   同一計測系列〈トラッカー〉の共有を意味し、暗黙コピーとは意味が異なる
//!   点に注意。ピークを集約したい場合は明示的に `clone()` する）
//!
//! TASK-1.9c（#46）で `ops` モジュール（[`ops::CpuBackendOps`]）を追加した。
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の CPU 実装であり、既存カーネル
//! （[`gemm_blis::gemm_blis_parallel`]・`elementwise` の `add`/`mul`/`relu`/
//! `exp`/`tanh`・[`reduction`] の `sum`/`max`）への薄い委譲に徹する。CUDA／
//! Metal 実装（`backend-cuda::ops::CudaBackendOps`／
//! `backend-metal::ops::MetalBackendOps`）と同一 trait でカーネルディスパッチ
//! できることを `tests/backend_ops_dispatch.rs` で検証する。
//!
//! TASK-12.1f（#203）で [`gemm_blis::gemm_blis_bias_act_parallel`]（GEMM epilogue
//! 〈bias 加算・activation〉のカーネル内融合）を追加し、[`ops::CpuBackendOps`] の
//! `gemm_bias_act`（`fandhe_ai_tensor_core::BackendOps` のデフォルトメソッド。非融合合成）を
//! オーバーライドして接続した。非融合実行（`gemm` → `add` → `relu` の 3 パス・中間
//! `Tensor` 2 個割当）に対する性能改善は `docs/perf/cpu-gemm-epilogue-fusion.md` に
//! 実測記録している（CUTLASS 系実測の動機は平均 1.38〜1.45 倍。本環境実測は 1.46〜
//! 2.56 倍）。融合版と非融合合成の bit 完全一致は `tests/gemm_epilogue_parity.rs` で
//! 検証する。CUDA／Metal は GPU カーネル内 epilogue 融合をスコープ外とし、
//! `gemm_bias_act` のデフォルト実装（elementwise 未実装のため `Unsupported`）に留める。
//!
//! TASK-12.1c（#163）で [`fused_elementwise`] モジュール
//! （[`fused_elementwise::run_fused_elementwise`]）を追加した。
//! `fandhe_ai_tensor_core::fusion`（TASK-12.1a〜c・#161〜#163）が検出・生成した
//! elementwise 連鎖（`fandhe_ai_tensor_core::FusionPlan`）を、per-op カーネル
//! （`elementwise`）の逐次合成ではなく単一パスのレジスタ内評価で実行
//! する CPU 参照実装である（PoC-9 `ElemwiseFuse` 方式。詳細は
//! `fused_elementwise` モジュール冒頭コメント）。`fandhe_ai_tensor_core::BackendOps::
//! run_fused`（trait への追加・[`ops::CpuBackendOps`] での override 実装）
//! への結線は #164 のスコープであり、#163 時点では関数ベースのカーネル
//! API として独立に提供する（[`gemm`]／`elementwise` と同じ「trait
//! 定義なし・関数ベース」構成）。数値契約は per-op カーネルと完全に
//! 揃え、融合の有無で許容誤差・演算定義を変えない（`tests/
//! fused_elementwise_parity.rs` で融合 vs 非融合の数値一致を検証する。
//! 受け入れ条件）。

//! イシュー #607 で [`rmsnorm`]・[`softmax`] モジュール（融合 RMSNorm／
//! softmax 順伝播の NEON + `rayon` 参照実装。`backend-cuda::rmsnorm`
//! （#592）・`backend-cuda::softmax`（#594）・`backend-metal`（#604）と
//! 同じ意味論・プラン一致契約）を追加し、`ops::CpuBackendOps::run_fused`
//! を「RMSNorm 一致 → softmax 一致 → 既存 elementwise 融合」の 3 分岐へ
//! 拡張した。exp 実装方式は標準 `f32::exp` を採用（[`softmax`] モジュール
//! 冒頭コメント参照。tolerance 緩和は行わない）。
//!
//! イシュー #1363（親 #1362・祖 #1361）で `thread_limit` モジュール
//! （macOS `hw.perflevel0.logicalcpu`／Linux sysfs `cpu_capacity` による
//! 大コア数判定・判定不能時フォールバック。診断用の [`ThreadLimitReport`]／
//! [`thread_limit_report()`] のみ公開）を追加し、`gemm_blis` 並列 GEMM の
//! 既定並列度を大コア数へ限定する制御を単一 const ゲート
//! （`thread_limit::BIG_CORE_LIMIT_ENABLED`）付きで結線した。性能上の
//! 採否判断は #1364（両実機 framework-compare 前後比較）へ引き継ぐ
//! （`docs/perf/cpu-gemm-default-thread-limit.md`）。
//!
//! イシュー #1313 で `gemm_blis` の並列化戦略を静的行パネル分割
//! （`par_chunks_mut`）から (mc, nc) 2D job 動的分配（`GemmDriverVariant::
//! TwoDDynamic`。イシュー #1311・#1312）へ、単一 const ゲート
//! （`gemm_blis::TWO_D_DYNAMIC_PRODUCTION_ENABLED`）付きで結線した。
//! Apple M4 Max 専有ゲート付き再計測で採用ゲート（対 `RowPanel` 比。DGX
//! Spark GB10・Apple M4 Max 両実機）を満たしたことを確認し（`docs/perf/
//! cpu-gemm-2d-dynamic-partition-ab.md`「#1313 追記」節）、framework-compare
//! gemm cpu の before/after（両実機・全 12 セル非後退）で ADOPT を確定した
//! （`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20）。

mod device;
mod elementwise;
pub mod fused_elementwise;
pub mod gemm;
pub mod gemm_blis;
// イシュー #1576: GB10（DGX Spark GB10）小形状 GEMM の大コア OS
// affinity 自機判定（`crate::thread_limit`〈スレッド数制限のみ・
// #1364 REJECT 確定〉とは独立の別系統機構。既定 OFF）。
// `crate::gemm_blis::gemm_blis_parallel_with_transpose`／
// `gemm_blis_bias_act_parallel` から呼ばれる。
mod gb10_affinity;
// イシュー #1290: `gemm --mode reuse --phases`（framework-compare。
// #1182）の `matmul` 区間内訳（alloc_c／kernel／tensor_wrap／host_copy／
// checksum）を実測分解する CPU 側診断テスト。`crate::gemm_blis::
// gemm_blis_parallel`（`pub`）・`crate::ops::CpuBackendOps`（`pub(crate)`
// フィールドは使わないが `crate::ops` 内部の型として到達）へ結線する
// ため、CUDA `gemm_reuse_phase_diag_tests`（#1182）・Metal 同名ファイル
// （#1189）と同じ理由でクレートルートの兄弟モジュールとして配置する。
#[cfg(test)]
mod gemm_reuse_phase_diag_tests;
// イシュー #1319: `docs/cpu-gemm-prefetch-decision.md`（#489・#751）の
// 「原則不要（HW ストリームプリフェッチャー任せ）」格下げ判断を覆す
// 帯域律速根拠の有無を実測する診断テスト（候補 2 `vld1q_f32_x3` 経路
// prefetch）。`crate::gemm_blis::microkernel::NeonKernel`（`pub`）へ
// 到達するためクレートルートの兄弟モジュールとして配置する
// （`gemm_reuse_phase_diag_tests` と同じ配置理由）。`unsafe asm!` は
// 本ファイル・本番経路のいずれにも追加しない（着手にはユーザー承認が
// 必要）。`NeonKernel` は aarch64 限定型（`microkernel.rs` の
// `#[cfg(target_arch = "aarch64")]`）のため、本モジュールも同条件で
// 限定する（x86_64 では `cargo check --target x86_64-unknown-linux-gnu`
// がクレート全体をスキップせず正常終了する）。
#[cfg(all(test, target_arch = "aarch64"))]
mod gemm_prefetch_bandwidth_diag_tests;
pub mod layer_norm;
pub mod linalg;
pub mod memory;
mod mse;
mod ops;
pub mod parity;
pub mod reduction;
pub mod rmsnorm;
mod rnn_cell;
mod small_shape_thread_cap;
// イシュー #1587: Arm SME（Scalable Matrix Extension）の実行時検出
// （fail-closed。macOS sysctl／Linux /proc/cpuinfo・rdsvl による SVL 確認）。
// `gemm_blis::microkernel::SmeKernel::try_new` から呼ばれる。診断専用の
// `SmeReport`／`sme_report()` のみ公開し facade へは昇格しない
// （`thread_limit::ThreadLimitReport` と同型の位置づけ）。
mod sme_detect;
pub mod softmax;
mod thread_limit;

pub use device::CpuDeviceProvider;
pub use elementwise::{
    add, add_slice, exp, exp_slice, mul, mul_slice, relu, relu_slice, tanh, tanh_slice,
};
pub use fused_elementwise::run_fused_elementwise;
pub use gb10_affinity::{Gb10AffinityReport, gb10_affinity_report};
pub use gemm::{
    BlockSizes, GemmError, gemm_blocked, gemm_naive, gemm_parallel, gemm_parallel_tuned,
};
pub use gemm_blis::{
    gemm_blis, gemm_blis_bias_act_parallel, gemm_blis_parallel, gemm_blis_parallel_nt,
    gemm_blis_parallel_tn,
};
pub use layer_norm::{LayerNormError, run_layer_norm_f32};
pub use linalg::LinalgError;
pub use memory::CpuMemory;
pub use ops::CpuBackendOps;
pub use parity::{
    ABSOLUTE_RESCUE_THRESHOLD, CompareReport, ParityError, RELATIVE_TOLERANCE, assert_parity,
    compare, matmul_reference_fma,
};
pub use rmsnorm::{RmsNormError, run_rmsnorm_f32};
pub use small_shape_thread_cap::{SmallShapeCapReport, small_shape_cap_report};
pub use sme_detect::{SmeReport, sme_report};
pub use softmax::{SoftmaxError, run_log_softmax_f32, run_softmax_f32};
pub use thread_limit::{ThreadLimitReport, thread_limit_report};
