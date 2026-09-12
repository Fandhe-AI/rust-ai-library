//! BLIS/GotoBLAS2 5-loop model（jc→pc→ic→jr→ir）による GEMM（TASK-1.6f・#184）。
//!
//! [`crate::gemm`] の `gemm_naive`／`gemm_blocked`／`gemm_parallel`（TASK-1.6a）
//! は自動ベクトル化頼みのスカラーループで、PoC-v2-1 実測では対 PyTorch CPU 比
//! 5.3%（M=N=K=2048/4096 最小値）に留まり REQ-8 の CPU 最適化後下限（20%）に
//! 届かない。本モジュールは `std::arch` intrinsics（NEON／AVX2+FMA／AVX-512F）
//! による マイクロカーネル・A/B packing（`pack`）・キャッシュ階層ブロッキング
//! （MC/KC/NC）を実装し、性能向上を狙う。
//!
//! ## 実行時 ISA ディスパッチ（#185・TASK-1.6g）
//!
//! TASK-1.6f まではマイクロカーネルの ISA 選択がコンパイル時 cfg のみ
//! だったため、x86_64 の既定ビルド（`RUSTFLAGS` なし）では実行 CPU が
//! AVX2/AVX-512 を持っていてもスカラーへ落ちていた。本モジュールの
//! 公開入口（[`gemm_blis`]／[`gemm_blis_parallel`]）は `dispatch_region`
//! で 1 回だけ ISA トークンの検出・選択を行い、`gemm_blis_region` を
//! モノモーフィック化されたジェネリック関数として呼ぶ（トークン型による
//! 健全な dispatch の設計は [`microkernel`] モジュールドキュメント参照）。
//!
//! ## 公開 API 非破壊（既存 3 関数は変更しない）
//!
//! [`crate::gemm::gemm_naive`]／`gemm_blocked`／`gemm_parallel` は #24
//! （TASK-1.6d・PoC-v2-1 比 3 段階性能確認）の参照点として変更しない
//! （公開 API 非破壊はガードレール条件・`.claude/rules/security.md`）。
//! [`gemm_blis`]／[`gemm_blis_parallel`] のシグネチャも #185 で変更しない
//! （dispatch はこれら関数の内部実装としてのみ追加）。
//!
//! ## bit 完全一致契約（REQ-2）
//!
//! [`microkernel`] の各カーネルは C 要素ごとの累積を p 昇順の FMA 連鎖で
//! 行い、レーン間縮約（split-k 等）を一切行わない設計とすることで、
//! `gemm_naive` と bit 完全一致が成立する（`tests/gemm_blis_parity.rs`）。
//! 累積順序を変える最適化（split-k・ゼロ初期化してからの後加算方式）は
//! 本契約を壊すため、将来追加する場合は数値一致テストの契約変更として
//! ユーザー承認事項である。実行時 ISA ディスパッチ導入後もどの ISA が
//! 選ばれても結果は bit 完全一致するため、既存 parity テストはそのまま
//! 実行時 dispatch 経路の検証を兼ねる。
//!
//! ## 境界検査（REQ-8）
//!
//! 公開入口は `crate::gemm::validate_dims`（`checked_mul` によるオーバー
//! フロー検査・スライス長検査）を再利用する。packing・端タイルの C
//! 書き戻しは安全な slice 操作で行い、intrinsics のロード／ストアは
//! マイクロカーネル関数入口の `assert!` で長さを検査した直後の最小
//! `unsafe` ブロックに限定する（`microkernel::neon`／`microkernel::avx2`／
//! `microkernel::avx512` 参照）。dispatch 導入を理由とした境界検査の
//! 省略は行わない。

// `cache_params`／`partition` は #753 の本番未結線スコープ（下記
// [`gemm_blis_parallel_2d_with_blocks`] ドキュメント参照）のため、
// 呼び出し元がテスト専用パラメータ化入口・実機 A/B ハーネス
// （いずれも `#[cfg(test)]`）に限られる。`gemm_blis_with_kernel_and_blocks`・
// `gemm_blis_parallel_with_blocks`（#564）・`gemm_blis_shared_b_region`・
// `dispatch_shared_b`（#750）と同じ「本番未結線の間はモジュール自体を
// `#[cfg(test)]` にする」既存パターンを踏襲し、`cargo build`（`cfg(test)`
// 無効時）での dead_code 検出を構造的に避ける（`#[allow(dead_code)]` に
// よる黙らせは行わない。`.claude/rules/coding-rust.md`）。sysctl FFI
// （`cache_params::sysctl_ffi`。macOS 限定）の型・借用検査は `cargo test`
// （`rust-ci` の test ジョブ）が `cfg(test)` を有効化した状態でコンパイル
// する際に行われる（同ジョブは Linux ホストのため `cfg(target_os =
// "macos")` 自体は無効化されコンパイル対象に含まれない。macOS 実機での
// 検証手順は `docs/perf/cpu-gemm-runtime-cache-detect.md` 参照）。
#[cfg(test)]
mod cache_params;
pub mod microkernel;
mod pack;
// イシュー #1313 で本番結線（`gemm_blis_parallel_with_transpose`／
// `gemm_blis_bias_act_parallel` からの [`dispatch_two_d_dynamic`] 呼び出し）
// を追加したため、`partition::job_grid` が本番経路から到達可能になり
// モジュール自体の `#[cfg(test)]` は外した。モジュール内のテスト専用
// ヘルパー（`split_evenly`／`row_ranges_for_workers`。#753 の
// [`gemm_blis_parallel_2d_with_blocks`] 経由のみ使用）は個別に
// `#[cfg(test)]` を付与している（`partition.rs` 冒頭コメント参照）。
mod partition;

use std::ops::Range;
// `IcDynamic`（イシュー #1366）専用: 行パネルの動的配布に使う
// `AtomicUsize` カウンタ・排他アクセス用 `Mutex`。本番未結線のため
// `#[cfg(test)]` ゲート（本ファイル冒頭「cache_params／partition」節と
// 同じ理由。`.claude/rules/coding-rust.md` の dead_code 黙らせ回避方針）。
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::{Mutex, PoisonError};

use fandhe_ai_tensor_core::Activation;

use crate::gemm::{BlockSizes, GemmError, validate_dims};
#[cfg(not(target_arch = "x86_64"))]
use microkernel::Isa;
use microkernel::Microkernel;
// `mod tests`（`use super::*;` で全取り込み）内のテストが ISA 分岐を問わず
// `ScalarKernel` を直接参照するため、cfg 条件に `test` を加える
// （aarch64 の非テストビルドではスカラーカーネル型を経路上使わないが、
// テストバイナリのコンパイルには必要。#185 レビュー指摘）。
#[cfg(any(not(target_arch = "aarch64"), test))]
use microkernel::ScalarKernel;
use pack::{
    APackTile, ATPackTile, BPackTile, BTPackTile, pack_a, pack_a_from_transposed, pack_b,
    pack_b_from_transposed,
};
use rayon::prelude::*;

/// 端タイル専用の C スタックバッファ最大要素数（`MR * NR` の全 ISA 中の
/// 最大値。AVX-512 の 8×32=256 が最大。ジェネリック const 式は stable
/// Rust では使えないため固定長で確保し、各カーネルモジュールの
/// `const _: () = assert!(MR * NR <= 256);` でこの上限を守ることを
/// コンパイル時に検査する）。
///
/// #557 により、完全タイル（`mr_eff == MR && nr_eff == NR`）は C の実
/// バッファへ [`Microkernel::run_with_ldc`] の `ldc` 契約経由で直接ロード/ストア
/// するため、本バッファは境界検査（REQ-8）が必要な**端タイル専用**に
/// 用途が絞られた（参照実装 matrixmultiply の「完全タイルは直接、端
/// タイルのみマスク付きバッファ」という設計に倣う。以前は全タイルが
/// 本バッファ経由でタイルあたり 2×MR×NR 要素のコピー往復を余分に
/// 払っていた）。
const MAX_TILE: usize = 256;

/// キャッシュブロッキングの行方向ブロックサイズ（A のパネル高さ）。
///
/// [`crate::gemm`] の既存定数（PoC-v2-1 実測環境で選定）を起点値として
/// 踏襲する。マイクロカーネルの MR/NR は ISA ごとに異なる（[`microkernel`]
/// 参照）ため、本モジュール独自の定数として持つ（`gemm` モジュールの
/// MC/KC/NC とは独立にチューニング可能にする）。
///
/// **対象実機・再選定状況（#564・#749・#753 へ引き継ぎ）**: 本 3 定数の
/// チューニング対象実機は **Apple M4 Max**（firestorm 系。
/// `docs/perf/gemm-optimization-baseline.md` §3・イシュー #481 で確定）。
/// BLIS/OpenBLAS の aarch64 向け参照実装値（firestorm: MC=480/KC=4096/
/// NC=9600 等）近傍を含む実機スイープは 2026-08-19 に M4 Max で実施済みで、
/// 詳細実測値・採否根拠は `docs/perf/cpu-gemm-blocking-sweep.md` §7 に
/// 記録する（#749）。MC・KC は単独拡大が全サイズで劣化したため PoC-v2-1
/// 値のまま維持する。
///
/// **NC の n 依存拡大（NC=9600・n>=4096 で約 9.9% 改善）は本 PR 時点で
/// 未有効化（PR #766・codex-review 再指摘）**: 実測を行った M4 Max
/// 個体の正確な `hw.model` 識別子が実測セッション終了時点で記録されて
/// おらず本 PR 時点では復元不能（`docs/perf/cpu-gemm-blocking-sweep.md`
/// の「実測値の捏造・placeholder 値での完了扱いは行わない」方針に従い、
/// `Mac16,` prefix 等の広い一致条件へ後退させて有効化することもしない）。
/// 実機固有値を検証していないターゲット・機種への適用は
/// `.claude/rules/coding-rust.md` の方針に反するため、識別子が判明する
/// までは常時 `default_blocks()`（固定 NC=512）を返す（#753〈sysctl
/// ベース MC/KC/NC 動的算出〉で機種識別を含めて再検討する）。
const MC: usize = 128;
/// 縮約次元（K）のブロックサイズ。B パネル（KC×NC×4B）が L1/L2 に収まる値。
///
/// **KC=128〜512 の細粒度再スイープ（イシュー #1315）**: #749 は KC=4096 への大幅拡大
/// （単独）のみを M4 Max 単独で検証していたが、#1315 は現行値近傍のグリッド
/// （KC ∈ {128, 192, 256, 384, 512}）を Apple M4 Max・DGX Spark GB10（Grace CPU）の
/// 両実機で 5 回独立プロセス中央値スイープした。いずれの実機・KC 値でも N=1024・2048
/// の両方で現行 KC=256 を上回る候補は確認できず（`docs/perf/cpu-gemm-candle-cpu-retune.md`
/// §8.1 実測表）、KC=256 を維持する（REJECT・不採用確定）。
const KC: usize = 256;
/// 列方向ブロックサイズ（B のパネル幅）。本 PR 時点の唯一の適用値
/// （上記 NC 拡大の未有効化理由を参照）。
const NC: usize = 512;

/// [`gemm_blis`]／`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`
/// （本番 3 公開関数）が使う既定ブロックサイズ（上記 `MC`/`KC`/`NC`
/// 定数と同一値）。[`crate::gemm::BlockSizes`] 型を再利用してパラメータ化
/// した理由は [`dispatch_region`] のドキュメント参照（#564・§3.1
/// `crate::gemm::gemm_blocked` 向け `BlockSizes` 導入〈#24〉と同じ前例
/// 踏襲）。値自体は `gemm` モジュールの既定値と独立にチューニング可能な
/// ままにするため、`BlockSizes::poc_v2_1_default()` を直接使わずここで
/// 本モジュール専用の定数から構築する。
const fn default_blocks() -> BlockSizes {
    BlockSizes {
        mc: MC,
        kc: KC,
        nc: NC,
    }
}

/// `gemm_blis_parallel`／`gemm_blis_bias_act_parallel` の rayon 行パネル
/// 並列化を直列（[`dispatch_region`] 直呼び）へフォールバックさせる
/// 実効ワークロード積の下限（イシュー #811・#1027）。
///
/// **本番未結線（`#[cfg(test)]` 限定・codex-review P1 指摘への対応）**:
/// [`should_serialize`] は本番公開入口（[`gemm_blis_parallel`]／
/// [`gemm_blis_bias_act_parallel`]）からは参照しない。理由は境界実測が
/// ローカル QEMU x86_64 に限られ、REQ-8 の正式対象実機 Apple M4 Max
/// での再スイープが未実施のため（PR #830 codex-review P1 指摘で
/// 一度 `#[cfg(test)]` 限定化した後、#1027 で本番結線を試みたが同種の
/// 指摘を再度受けたことを踏まえ、`dispatch_shared_b`〈#750〉・
/// `gemm_blis_parallel_with_blocks`〈#564〉と同じ「実機ゲート未通過の
/// うちは攻めた値を本番結線しない」方針〈PR #758 前例〉に統一する）。
/// `m*n*k` 単純積が細長形状〈例 m=512,n=1,k=512〉を誤って直列側へ倒す
/// 〈Cursor Bugbot 指摘〉問題自体は下記 `NR_CLAMP` 導入で解消済みだが、
/// M4 Max 実機での境界再スイープが残る限りは本番結線しない。閾値
/// 589,824 自体は保守的な値として変更しない
/// （`docs/perf/cpu-gemm-small-shape-serial-fallback.md` 追補参照）。
/// `m == 1`（gemv）専用経路 [`gemm_row_vector`] は本判断と独立
/// （マイクロカーネルが複数行分の packing を前提とする構造に対し
/// `m == 1` が常に無駄になるというアーキテクチャ非依存の性質のため）。
///
/// **`gemm` crate（OSS 比較対象）との設計差**: `gemm` crate は
/// (jc,pc) ブロック単位の `total_work = m * n_chunk * k_chunk` を
/// 閾値と比較し、5-loop の途中でも直列へ切り替えうる（ローカル cargo
/// cache の `gemm-common-0.19.0/src/gemm.rs:109,515-522` で実体確認済み。
/// `docs/oss-comparison-harness-decision.md`）。自作実装は並列化が
/// **呼び出しレベル**（rayon `par_chunks_mut` による行パネル分割）に
/// 1 箇所しかないため、ブロック単位判定をそのまま移植せず、呼び出し
/// 直後の 1 回判定という等価な意味論で最も単純な形を採る。
///
/// **値の根拠**: `gemm` crate の既定閾値と同値の `48 * 48 * 256 =
/// 589_824` を採用する。自作実装での実測（ローカル QEMU x86_64 12 コア
/// AVX2+FMA。`docs/perf/cpu-gemm-small-shape-serial-fallback.md`）でも
/// 正方形状 64³=262,144 では並列が直列より最大 12 倍遅く、128³=
/// 2,097,152 では並列が直列より約 1.8〜1.9 倍速いという明確な交差が
/// この閾値の間（262,144 と 2,097,152 の間）に収まっており、
/// 保守的な採用として妥当と判断した（同 doc §結論）。
///
/// **bit 完全一致契約への影響なし**: 直列フォールバックは
/// [`dispatch_region`] を呼ぶ経路自体を変えるだけで、C 要素ごとの
/// p 昇順 FMA 連鎖（モジュール doc「bit 完全一致契約」）には触れない
/// ため `gemm_naive`／並列実行時と bit 完全一致のまま不変
/// （`tests/gemm_blis_parity.rs` の境界形状ケースで検証）。
///
/// **性能ヒューリスティクスであり非該当事項**: 本閾値はガードレール
/// 閾値・テスト許容誤差のいずれでもない（数値結果は不変のまま）ため、
/// `.claude/rules/delegation-impl.md` の「実装 Agent にガードレール
/// 閾値・テスト許容誤差を緩和させない」禁止事項には該当しない。
#[cfg(test)]
pub(crate) const GEMM_THREADING_THRESHOLD: usize = 48 * 48 * 256;

/// [`should_serialize`] が `n` の実効値としてクランプする下限（#1027）。
///
/// B/C パネルの packing はマイクロカーネル NR 幅（NEON 12/8・AVX2/
/// AVX-512 も複数列。[`microkernel`] 各実装の `NR` 定数参照）までゼロ
/// パディングして処理するため、`n` が小さい細長形状（例 m=512,n=1,k=512）
/// でも実際に消費される仕事量は `n` そのものではなく NR 幅程度が下限になる。
/// 全 ISA の NR 定数の中で最小値（NEON/AVX2 の 8）をこの下限として採用し、
/// ISA ごとに異なる実際の NR 値より安全側（並列判定が並列寄りになる方向）
/// に倒す保守的な近似とする（`should_serialize` の判定は「実行される
/// 実効仕事量」の proxy であり FLOP 数そのものではない）。
#[cfg(test)]
const NR_CLAMP: usize = 8;

/// `gemm_blis_parallel`／`gemm_blis_bias_act_parallel` が rayon 行パネル
/// 並列化を直列（[`dispatch_region`] 直呼び）へフォールバックさせるか
/// 判定する（イシュー #811・#1027）。`true` なら直列、`false` なら並列。
///
/// **Cursor Bugbot 指摘（PR #830）の解消**: 並列化は呼び出しレベルの
/// 行パネル分割（rayon `par_chunks_mut`。並列度は実質 `m` に依存）の
/// ため、単純な `m*n*k` 積は `m` が大きく `n`／`k` が小さい細長形状
/// （例 m=512,n=1,k=512。`m*n*k=262,144<GEMM_THREADING_THRESHOLD` だが
/// 行パネル分割は十分機能し並列が有利。`docs/perf/
/// cpu-gemm-small-shape-serial-fallback.md` 実測 1 の当該行:
/// parallel/serial 2.233x）を誤って直列側へ倒す。`n` を [`NR_CLAMP`]
/// で下限クランプしてから積を取ることで、この非対称性を吸収する
/// （m=512,n=1,k=512 は `512 * 8 * 512 = 2,097,152 >=
/// GEMM_THREADING_THRESHOLD` となり並列判定に是正される）。
///
/// **既存実測との整合**: `docs/perf/cpu-gemm-small-shape-serial-fallback.md`
/// 実測 1〜3 の全形状（正方 16/32/64/128/256・gevv m=n=256〜2048,k=1/2）は
/// `n >= NR_CLAMP` のため本クランプの影響を受けず、従来の `m*n*k` 単純積
/// と同じ判定結果になる（机上突合は `#1027` 実装計画・下記単体テスト
/// `should_serialize_matches_measured_table` で固定する）。
///
/// **オーバーフロー時の安全側フォールバック**: `saturating_mul` を使う
/// ため、三重積オーバーフロー時は `usize::MAX` へ飽和し常に `false`
/// （並列側）を返す。
///
/// `#[cfg(test)]` 限定（本番未結線。上記 [`GEMM_THREADING_THRESHOLD`]
/// ドキュメント「本番未結線」参照）。実測表との突合テスト
/// （`should_serialize_matches_measured_table` 等）からのみ呼ばれる。
#[cfg(test)]
pub(crate) fn should_serialize(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(n.max(NR_CLAMP)).saturating_mul(k) < GEMM_THREADING_THRESHOLD
}

/// gemv 相当（`m == 1`。単一行 × B 全体）の内積計算を BLIS packing・
/// マイクロカーネルタイル経由ではなく直接計算する専用経路（イシュー
/// #811・§3.2）。
///
/// **採用根拠**: 実測（`docs/perf/cpu-gemm-small-shape-serial-fallback.md`
/// §gemv_m1）で `m == 1` 形状は [`gemm_blis`]（直列 BLIS 経路）が
/// `gemm_naive` 比最大 8.3 倍遅い（MR 単位の A パネル packing が
/// 単一行を MR 行ぶんにゼロパディングする無駄が支配的。NEON では
/// MR=8・x86_64 AVX2/AVX-512 でも同様に複数行分のタイル前提のため）。
/// `m == 1` は rayon 行パネル分割でも並列化の恩恵がない（分割対象が
/// 1 行のみのため `par_chunks_mut` のチャンク数は常に 1 になり、
/// 並列化オーバーヘッドだけが乗る。実測 `parallel_vs_serial` が
/// 常におよそ 1.0 前後で推移することからも確認できる）。
///
/// **bit 完全一致契約**: [`crate::gemm::gemm_naive`] の `i == 0` 反復と
/// 完全に同じ p 昇順 `f32::mul_add` 連鎖で計算するため、`gemm_naive`
/// および BLIS 経路（`tests/gemm_blis_parity.rs` の既存網羅検証により
/// `gemm_naive` と bit 完全一致することが確認済み）の両方と自明に
/// bit 完全一致する（新規カーネル・新規累積順序を導入しないため）。
///
/// # 契約
///
/// 呼び出し元が `m == 1` を確定した上で呼ぶ。`a` は `k` 要素、
/// `b` は `k * n` 要素、`c` は `n` 要素であることを呼び出し元
/// （[`validate_dims`] 通過済みの公開入口）が保証する前提とし、
/// 本関数自体は追加の長さ検査を行わない（構造的にスライス添字が
/// 境界外に出ないことは `a.iter().enumerate()` と `b[p*n..p*n+n]` の
/// 組み合わせで保証される。`p < k` かつ `b.len() == k*n` のため
/// `p*n+n <= k*n == b.len()`）。
fn gemm_row_vector(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for (p, &a_p) in a.iter().enumerate() {
        let b_row = &b[p * n..p * n + n];
        for (c_j, &b_j) in c.iter_mut().zip(b_row.iter()) {
            // FMA 契約統一（REQ-2）: `gemm_naive` と同一の
            // `f32::mul_add` 連鎖（`crate::gemm::gemm_naive` 参照）。
            *c_j = a_p.mul_add(b_j, *c_j);
        }
    }
}

/// [`gemm_row_vector`] の NT パターン版（#1213。`m == 1` かつ B が転置
/// 格納 `bt`〈論理形状 `[n, k]` の行優先〉の場合の専用経路）。
///
/// **bit 完全一致契約**: `gemm_row_vector` は列 `j` ごとに独立な
/// アキュムレータへ p（K 方向）昇順の `f32::mul_add` 連鎖を適用する
/// （どの列も他列の計算に影響しないため、ループの外側/内側を p/j
/// どちらに取っても各列の FMA 連鎖の順序自体は不変）。本関数は
/// ループ順を j 外側・p 内側へ入れ替えるのみで、列 `j` の連鎖内容
/// （`c[j] = a[0]*bt[j,0] + a[1]*bt[j,1] + ... `を p 昇順 `mul_add`
/// で評価する点）は [`gemm_row_vector`] と完全に同一である
/// （`bt[j,p] == b[p,j]` は同じ数学的要素を指すため、読み出し元の
/// メモリレイアウトが異なるだけで計算内容は変わらない）。よって
/// [`gemm_row_vector`] と bit 完全一致する（`gemm_blis/mod.rs` 内の
/// `gemm_row_vector_nt_matches_gemm_row_vector_bit_exact` で検証）。
///
/// # 契約
///
/// 呼び出し元が `m == 1` を確定した上で呼ぶ。`a` は `k` 要素、`bt` は
/// `n * k` 要素、`c` は `n` 要素であることを呼び出し元（[`validate_dims`]
/// 通過済みの公開入口）が保証する前提とし、本関数自体は追加の長さ検査を
/// 行わない（[`gemm_row_vector`] と同じ設計判断）。
fn gemm_row_vector_nt(a: &[f32], bt: &[f32], c: &mut [f32], k: usize) {
    for (j, c_j) in c.iter_mut().enumerate() {
        let bt_row = &bt[j * k..j * k + k];
        let mut acc = *c_j;
        for (&a_p, &bt_p) in a.iter().zip(bt_row.iter()) {
            acc = a_p.mul_add(bt_p, acc);
        }
        *c_j = acc;
    }
}

/// GEMM オペランドの転置パターン（VJP 専用 NT/TN 2 パターン限定入口。
/// #1213）。
///
/// - `Nn`: 通常（A・B とも論理形状どおりの行優先連続）
/// - `Nt`: A は通常、B は転置格納（`bt`。論理形状 `[n, k]` の行優先。
///   元の B `[k, n]` を転置した view の実体）
/// - `Tn`: A は転置格納（`at`。論理形状 `[k, m]` の行優先。元の A
///   `[m, k]` を転置した view の実体）、B は通常
///
/// 両方転置（TT）は本イシューのスコープ外のため variant を持たない
/// （呼び出し元 `backend-cpu::ops` が `contiguous()` で吸収してから
/// `Nn` を渡す。`docs/matmul-vjp-zero-copy-decision.md` §3.2）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GemmTranspose {
    Nn,
    Nt,
    Tn,
}

/// panel packing バッファ（A/B 各 1 本）を gemm 呼び出し単位で 1 回だけ
/// 確保し、5-loop 内の全 jc×pc×ic ブロックで使い回すための保持構造体
/// （#556。matrixmultiply 等参照実装の「gemm 呼び出しあたり 1 回確保＋
/// オフセット分割」方針に倣う）。
///
/// [`dispatch_region`] がカーネル型（`K::MR`／`K::NR`）確定直後に 1 回
/// 構築し、[`gemm_blis_region`] へ可変参照で渡す。直列経路（`gemm_blis`）
/// は gemm 呼び出しあたり 1 組、並列経路（`gemm_blis_parallel`／
/// `gemm_blis_bias_act_parallel`）は rayon の行パネルタスクごとに
/// `dispatch_region` が呼ばれるため**タスクごとに 1 組を所有**する
/// （`Vec` の所有権がタスクローカルに閉じるため、事前一括確保＋
/// `par_chunks_mut` とのオフセット分割を要さずコンパイル時にデータ競合
/// が排除される。B packing のスレッド間重複計算〈同じ B 列ブロックを
/// 複数タスクが個別に pack し直す〉は本変更のスコープ外＝既存挙動のまま。
/// 将来の並列分割再構成候補として PR 本文に記載する）。
struct PanelBuffers {
    /// B パネル用バッファ（全 jc×pc ブロック中で最大の nc_len×kc_len 組が
    /// 必要とする要素数で確保。ループ内は `nr_blocks*kc_len*nr` 要素の
    /// 先頭サブスライスのみ使う）。
    b_panel: Vec<f32>,
    /// A パネル用バッファ（全 jc×pc×ic ブロック中で最大の mc_len×kc_len
    /// 組が必要とする要素数で確保。ループ内は `mr_blocks*kc_len*mr` 要素の
    /// 先頭サブスライスのみ使う）。
    a_panel: Vec<f32>,
}

impl PanelBuffers {
    /// `n`（C の列数）・`k_dim`（縮約次元）・`mc_total`（この呼び出しが
    /// 担当する行数。並列時はパネル 1 つぶん）から、[`gemm_blis_region`]
    /// の全反復を通じて必要になる最大バッファ長を 1 回で計算し確保する。
    ///
    /// 各反復の必要量は `min(blocks.nc, n).div_ceil(nr)*min(blocks.kc,k_dim)*nr`
    /// （B）／`min(blocks.mc, mc_total).div_ceil(mr)*min(blocks.kc,k_dim)*mr`
    /// （A）が常に上界になる（先頭ブロックが nc_len/mc_len/kc_len 最大で、
    /// 末尾ブロックはこれらが縮むのみ。§4.1 計画）。`blocks`（[`BlockSizes`]）
    /// はパラメータ化（#564）により実行時値になったが、乗算オーバーフローの
    /// 懸念はない: 呼び出し元（[`gemm_blis_with_kernel_and_blocks`]／
    /// [`gemm_blis_parallel_with_blocks`] 等のパラメータ化入口）が
    /// [`validate_dims`] を先に通しており、`m*k`／`k*n`／`m*n` が `usize`
    /// で非オーバーフローと確定済みの `n`／`k_dim`／`mc_total` に対して
    /// `blocks.{nc,kc,mc}` は常に `.min()` でクランプしてから乗算される
    /// （`blocks` 側にどれだけ大きな値〈firestorm 参照値 KC=4096/NC=9600
    /// 等〉を渡しても、実際の乗算対象は非オーバーフロー確定済みの dim
    /// 由来値に収まる）。本番 3 公開関数は [`default_blocks`]（コンパイル
    /// 時定数）のみを渡すためこの経路も従来通り安全。`n == 0`／
    /// `k_dim == 0`／`mc_total == 0` では長さ 0 のバッファになり、5-loop
    /// 自体が回らないためサブスライス取得も発生せず問題ない。
    fn new<K: Microkernel>(n: usize, k_dim: usize, mc_total: usize, blocks: BlockSizes) -> Self {
        let (b_len, a_len) = panel_capacity(n, k_dim, mc_total, K::MR, K::NR, blocks);
        PanelBuffers {
            b_panel: vec![0.0f32; b_len],
            a_panel: vec![0.0f32; a_len],
        }
    }
}

/// [`PanelBuffers::new`] の容量計算本体（B 長・A 長の順で返す）。`mr`／`nr`
/// を型パラメータでなく引数に取ることで、単体テストが
/// [`Microkernel`] 実装を経由せず MR/NR の任意の組（全 ISA カーネル定数
/// 相当）に対して総当たり検証できるようにしている（#556 テスト計画
/// §5-3）。`blocks`（[`BlockSizes`]）は #564 でパラメータ化した MC/KC/NC。
fn panel_capacity(
    n: usize,
    k_dim: usize,
    mc_total: usize,
    mr: usize,
    nr: usize,
    blocks: BlockSizes,
) -> (usize, usize) {
    let kc_len_max = blocks.kc.min(k_dim);
    let nc_len_max = blocks.nc.min(n);
    let mc_len_max = blocks.mc.min(mc_total);
    let nr_blocks_max = nc_len_max.div_ceil(nr);
    let mr_blocks_max = mc_len_max.div_ceil(mr);
    (
        nr_blocks_max * kc_len_max * nr,
        mr_blocks_max * kc_len_max * mr,
    )
}

/// 単一スレッドの BLIS 5-loop GEMM（jc→pc→ic→jr→ir）。
///
/// `gemm_blis_parallel` はこの関数の内部ロジック（`gemm_blis_region`）を
/// 行パネルごとに並列呼び出しすることで並列化する（`gemm_blocked`／
/// `gemm_parallel` と同じ構成。`crate::gemm` 参照）。
pub fn gemm_blis(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    // gemv 相当（m == 1）は BLIS packing のパディング無駄が大きいため
    // 専用経路へ迂回する（[`gemm_row_vector`] ドキュメント参照）。
    if m == 1 {
        gemm_row_vector(a, b, c, n);
        return Ok(());
    }
    dispatch_region(a, b, c, n, k, 0..m, default_blocks(), GemmTranspose::Nn)
}

/// `gemm_blis` を `rayon` で行パネル並列化した版。
///
/// C を行方向にパネル分割し、各パネルを独立スレッドで `gemm_blis_region`
/// に渡す（`crate::gemm::gemm_parallel` と同じ並列化戦略。C の書き込み
/// 範囲がパネルごとに排他的なためデータ競合なし）。
///
/// 本体は `gemm_blis_parallel_with_transpose`（`Nn` 固定）への委譲
/// （#1213。VJP 専用 NT/TN 2 パターン入口 [`gemm_blis_parallel_nt`]／
/// [`gemm_blis_parallel_tn`] と実装を共有するための切り出し）。
pub fn gemm_blis_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    gemm_blis_parallel_with_transpose(a, b, c, m, n, k, GemmTranspose::Nn)
}

/// [`gemm_blis_parallel`]／[`gemm_blis_parallel_nt`]／[`gemm_blis_parallel_tn`]
/// （#1213）共通の実装本体。`transpose` で A・B オペランドどちらが転置
/// 格納（`GemmTranspose` ドキュメント参照）かを指定する。
///
/// `Nn` 時の挙動は #1213 以前の `gemm_blis_parallel` と完全に同一
/// （ロジックは移設のみで変更していない）。`Nt`／`Tn` は `dispatch_region`
/// 以下（[`gemm_blis_region`]／[`gemm_blis_ic_loop`]）が転置格納から
/// 直接 packing する経路へ分岐する（`pack.rs` モジュールドキュメント
/// 「転置格納からの直接 packing」節参照）。
pub(crate) fn gemm_blis_parallel_with_transpose(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    transpose: GemmTranspose,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;

    // n == 0 は shape として合法だが par_chunks_mut(panel_rows * n) は
    // チャンクサイズ 0 でパニックする（`crate::gemm::gemm_parallel` と
    // 同じ理由・同じ対処。3 実装間の parity 契約を保つため no-op で返す）。
    if n == 0 {
        return Ok(());
    }

    // gemv 相当（m == 1）は rayon 分割の恩恵がなく（`par_chunks_mut` の
    // チャンク数が常に 1 になる）BLIS packing のパディング無駄だけが
    // 乗るため専用経路へ迂回する（[`gemm_row_vector`] ドキュメント参照）。
    // `Tn`（`a` 引数は転置格納 `at`。論理形状 `[k,1]` 行優先＝通常の
    // `[1,k]` 行ベクトルとメモリ上完全に同一）は `gemm_row_vector` を
    // そのまま呼べる（#1213）。`Nt`（`b` 引数は転置格納 `bt`）は専用の
    // [`gemm_row_vector_nt`] を呼ぶ。
    if m == 1 {
        match transpose {
            GemmTranspose::Nn | GemmTranspose::Tn => gemm_row_vector(a, b, c, n),
            GemmTranspose::Nt => gemm_row_vector_nt(a, b, c, k),
        }
        return Ok(());
    }

    // ワークロード閾値直列フォールバック（イシュー #811・#1027）は
    // 本番結線しない（[`GEMM_THREADING_THRESHOLD`] ドキュメント
    // 「本番未結線」参照。REQ-8 正式対象実機 Apple M4 Max での境界
    // 再スイープ未実施のため。`should_serialize` は `#[cfg(test)]`
    // 限定で実装・実測突合テストのみ保持する）。

    // 呼び出しあたり 1 回だけ確定し、行パネル・2D 動的分配のどちらの
    // 経路でも共有するブロックサイズ（既定ブロックサイズは n に依存
    // しないためループ外で確定させても意味は変わらないが、
    // `dispatch_region`／`dispatch_two_d_dynamic` 呼び出しごとの再計算を
    // 避ける従来方針を踏襲する）。
    let blocks = default_blocks();

    // B パネル共有経路（`dispatch_shared_b`。案 B・イシュー #750）は
    // 本番既定へは**採用しない**（`docs/cpu-gemm-b-packing-sharing.md`・
    // `docs/cpu-gemm-b-packing-sharing-decision.md` 参照）。受け入れ条件 2
    // （Apple M4 Max 実機実測での M=N=K=2048/4096 非劣化確認）を満たす
    // 前提条件が未充足のため、実装・bit 完全一致検証・単体/統合テスト
    // （`dispatch_shared_b`／`gemm_blis_shared_b_region`／
    // `gemm_blis_ic_loop`・`#[cfg(test)]` の
    // [`gemm_blis_parallel_with_blocks`] 経由）は残しつつ、本番公開入口
    // からの呼び出しは行わない（PR #758〈#740 mma_f16 swizzle〉の
    // 「実装は入れるが実機ゲート未通過のうちは本番結線しない」判断と
    // 同型。実機実測とユーザー承認を経た別 PR でのみ既定切替を検討する）。
    // 従来どおり行パネルごとに [`dispatch_region`] を独立呼び出しする
    // （B packing はタスクごとに個別実行＝共有化前の挙動）。
    //
    // イシュー #1041 の pc 外側候補（`dispatch_shared_b_pc_outer`。B 共有化に
    // 加え A packing の重複〈jc 反復ごとの再 pack〉も解消する狙い）も、GB10
    // （#1140）・Apple M4 Max（#1141）双方の実機 5 回中央値実測で現行既定
    // `RowPanel` を大きく下回り（M4 Max: 1024 で約 33〜35%・2048/4096 で約
    // 22〜24% 低いスループット。GB10: 1024/2048 のみ実測で約 45〜54% 低い。
    // GB10 4096 は候補側未計測）、#1144 で本番結線しないと確定した。詳細・
    // 数値は `docs/perf/cpu-gemm-candle-cpu-retune.md` §8 を参照。
    //
    // ic 限定動的配布 `IcDynamic`（[`GemmDriverVariant::IcDynamic`]。
    // イシュー #1366）は `SharedBPcOuter` の静的行パネル等分割による
    // 負荷不均衡への対処候補として `#[cfg(test)]` 限定で追加済み。
    // 結線可否は #1367 の両実機実測後に判断する（本番未結線）。
    //
    // `Nt`（`b` 引数が転置格納 `bt`）は行範囲に依存せず全パネルへ同じ
    // `bt` を渡す（B packing は行パネル分割と無関係）。`Tn`（`a` 引数が
    // 転置格納 `at`）も同様に全パネルへ同じ `at` を渡し、行オフセットは
    // `gemm_blis_region` 内部（絶対行位置 `row_start + ic + ir`）で解決
    // する（`at` は m 方向でなく k 方向が先頭軸のため `a[row_start*k..]`
    // の単純スライスができない。#1213）。
    //
    // 本番結線（イシュー #1313）: [`TWO_D_DYNAMIC_PRODUCTION_ENABLED`]
    // （単一 const ゲート）が `true` の間は (mc, nc) 2D job 動的分配
    // （[`dispatch_two_d_dynamic`]）へ切り替える。`false` へ差し戻された
    // 場合は #1313 以前と同一の静的行パネル分割（`par_chunks_mut`）へ
    // 戻る（`RowPanel` 参照実装と bit 完全一致・`num_threads`／
    // `panel_rows` の算出も従来どおり `else` 節内でのみ行う）。
    // GB10 小形状 GEMM 大コア affinity ルーティング（イシュー #1576）:
    // `m*n*k` が小さい形状のみ GEMM 専用の大コア pin 済み rayon
    // ThreadPool 上で以下を実行する（既定 OFF・判定不能・大形状は
    // `f()` を直接呼ぶだけで下記ロジックは一切変更されない。
    // `crate::gb10_affinity` モジュール doc「適用範囲」節参照）。
    // M4 Max 側の仕事量ベース並列度上限（イシュー #1575・
    // [`crate::small_shape_thread_cap`]）とは自機判定で排他する別機構
    // （両モジュールの doc 相互参照を参照）。GB10 affinity プールが
    // 適用される場合、内側の `run_capped` は `rayon::current_num_threads()`
    // を専用プールのスレッド数として観測するため、両機構は無改変で
    // 積み重ねて動作する。
    crate::gb10_affinity::with_gb10_affinity_if_applicable(m, n, k, || {
        if TWO_D_DYNAMIC_PRODUCTION_ENABLED {
            // 小形状 GEMM の仕事量ベース rayon 並列度上限（イシュー #1575・
            // [`crate::small_shape_thread_cap`]）。M4 Max 自機判定・仕事量
            // 閾値未満の場合のみ専用の小さいプールで実行し、それ以外は
            // 現行プール（GB10 affinity プール適用時はそのプール・
            // 非適用時はグローバルプール）で `dispatch_two_d_dynamic` を
            // そのまま呼ぶ（`crate::thread_limit`〈コア種別ベース。#1363〉
            // とは独立の別機構）。
            crate::small_shape_thread_cap::run_capped(m, n, k, || {
                dispatch_two_d_dynamic(
                    a,
                    b,
                    c,
                    n,
                    k,
                    0..m,
                    blocks,
                    transpose,
                    TWO_D_JOBS_PER_WORKER,
                )
            })
        } else {
            // 行パネル分割数を rayon の既定スレッド数ではなく実効スレッド数
            // （`crate::thread_limit::effective_num_threads`）から算出する
            // （イシュー #1363。macOS `hw.perflevel0.logicalcpu`／Linux sysfs
            // `cpu_capacity` による大コア数判定で `RAYON_NUM_THREADS` 未指定時の
            // 既定並列度を大コア数へ限定し、判定不能時は従来どおり
            // `rayon::current_num_threads()` へフォールバックする。異種コア
            // 構成での非単調性仮説の検証が目的で、性能上の採否は #1364 が
            // 判断する。詳細は `docs/perf/cpu-gemm-default-thread-limit.md`）。
            // GB10 affinity プール内で実行される場合も同じ関数呼び出しで
            // `rayon::current_num_threads()` が専用プールのスレッド数を
            // 正しく返すため、本ロジックは無改変で動作する（#1576）。
            let num_threads =
                crate::thread_limit::effective_num_threads(rayon::current_num_threads());
            let panel_rows = m.div_ceil(num_threads).max(1);
            c.par_chunks_mut(panel_rows * n)
                .enumerate()
                .try_for_each(|(panel_idx, c_chunk)| {
                    let row_start = panel_idx * panel_rows;
                    let row_end = (row_start + c_chunk.len() / n).min(m);
                    dispatch_region(a, b, c_chunk, n, k, row_start..row_end, blocks, transpose)
                })
        }
    })
}

/// `gemm_blis_parallel_with_transpose` を `Nt`（B オペランドが転置
/// 格納）で呼ぶ VJP 専用入口（#1213）。`bt` は論理形状 `[n, k]` の行優先
/// （元の B `[k, n]` を転置した view の実体。`GemmTranspose` ドキュメント
/// 参照）。`matmul_vjp` の d_input（`g @ Wᵀ`）・`Op::LinearResident` の
/// d_input（`W @ gᵀ`。呼び出し元は `gemm_resident_lhs` 経由）が該当する。
///
/// 一般 stride（`narrow` 後の転置等）には対応しない。呼び出し元
/// （`backend-cpu::ops`）が `Tensor::strides() == [1, shape[0]]` の dense
/// な転置 view であることを判定してから呼ぶ契約とする
/// （`docs/matmul-vjp-zero-copy-decision.md` §3.2）。
pub fn gemm_blis_parallel_nt(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    gemm_blis_parallel_with_transpose(a, bt, c, m, n, k, GemmTranspose::Nt)
}

/// `gemm_blis_parallel_with_transpose` を `Tn`（A オペランドが転置
/// 格納）で呼ぶ VJP 専用入口（#1213）。`at` は論理形状 `[k, m]` の行優先
/// （元の A `[m, k]` を転置した view の実体。`GemmTranspose` ドキュメント
/// 参照）。`matmul_vjp` の d_weight（`Aᵀ @ g`）・`Op::LinearResident` の
/// d_weight（`xᵀ @ g`）が該当する。
///
/// [`gemm_blis_parallel_nt`] と同じ契約（一般 stride 非対応・呼び出し元
/// が dense 転置 view を判定済み）。
pub fn gemm_blis_parallel_tn(
    at: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    gemm_blis_parallel_with_transpose(at, b, c, m, n, k, GemmTranspose::Tn)
}

/// `gemm_blis_parallel` に GEMM epilogue（bias 加算・activation）を融合した版
/// （TASK-12.1f・#203）。
///
/// 非融合実行（`gemm_blis_parallel` → `add`〈bias〉→ `relu` の 3 パス・
/// 中間 `Vec<f32>` 2 個割当）に対し、C を行パネル並列で計算した直後（各
/// パネルがまだキャッシュ熱いうち）に同じ `rayon` タスク内で epilogue を
/// 適用することで、C の再読み出しパス・中間バッファ割当を削減する
/// （CUTLASS 系実測で epilogue 融合が平均 1.38〜1.45 倍。動機はイシュー
/// #203。`docs/perf/cpu-gemm-epilogue-fusion.md` に本環境での実測を記録）。
///
/// `bias` は `Some(&[f32])` の場合 `n`（`B` の列数）と同じ長さが必須で、
/// 各行へ加算される（`docs/public-api-design.md` §4.2 のブロードキャスト
/// 規約と同じ「`[n]` を行方向へ複製」の意味論。`fandhe_ai_tensor_core::BackendOps::
/// gemm_bias_act` のデフォルト実装〈`add` の broadcast〉と等価）。長さ
/// 不一致は [`GemmError::BiasLenMismatch`]（カーネル本体アクセス前に
/// 検証。REQ-8・OWASP A03）。
///
/// # bit 完全一致契約
///
/// epilogue（bias 加算・activation）は要素ごとに独立な演算で、パネル間の
/// 演算順序（並列実行のタスク分割）に依存しない。したがって本関数の
/// 結果は「`gemm_blis_parallel` で C を計算した後、全体へ 1 回だけ
/// bias 加算・activation を適用した結果」と **bit 完全一致**する
/// （`tests/gemm_epilogue_parity.rs` で検証）。GEMM 本体の FMA 契約
/// （`f32::mul_add`）・累積順序は `gemm_blis_parallel` から変更しない。
///
/// `#[allow(clippy::too_many_arguments)]`: `gemm_blis_parallel`（`a`／`b`／
/// `c`／`m`／`n`／`k` の 6 引数）に epilogue パラメータ（`bias`／`act`）を
/// 追加した結果 8 引数になる。GEMM カーネル公開入口の既存慣例
/// （`crates/backend-cpu/src/gemm.rs` の同 attribute 使用箇所と同方針）に
/// 従い、構造体化はせず素朴な引数列のまま許容する。
#[allow(clippy::too_many_arguments)]
pub fn gemm_blis_bias_act_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
    act: Activation,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    if let Some(bias) = bias
        && bias.len() != n
    {
        return Err(GemmError::BiasLenMismatch {
            expected: n,
            actual: bias.len(),
        });
    }
    // `Activation` は未知 variant（`#[non_exhaustive]`）を早期に拒否する。
    // `apply_epilogue` 内で検証すると、`try_for_each` が行パネル並列の
    // 途中パネルまで epilogue を適用した後にエラー終了しうる（`c` が
    // 部分適用の不定状態で返る）。呼び出し前にここで検証し、GEMM 本体・
    // epilogue のいずれにも触れない状態でのみエラーを返す。
    if !matches!(act, Activation::None | Activation::Relu) {
        return Err(GemmError::UnsupportedActivation);
    }

    // n == 0 は shape として合法（`gemm_blis_parallel` と同じ理由で
    // no-op）。epilogue も対象要素が無いため何もすることがない。
    if n == 0 {
        return Ok(());
    }

    // gemv 相当（m == 1）は `gemm_blis_parallel` と同じ理由（rayon 分割の
    // 恩恵なし・BLIS packing パディング無駄）で専用経路へ迂回する。
    // epilogue（bit 完全一致契約）は要素独立のため `gemm_row_vector` の
    // 結果（`c` 全 n 要素＝行パネル 1 つ分）へそのまま適用してよい。
    if m == 1 {
        gemm_row_vector(a, b, c, n);
        return apply_epilogue(c, n, bias, act);
    }

    // ワークロード閾値直列フォールバック（イシュー #811・#1027）は
    // `gemm_blis_parallel` と同じ理由（[`GEMM_THREADING_THRESHOLD`]
    // ドキュメント「本番未結線」参照）で本番結線しない。

    // 呼び出しあたり 1 回だけ確定し、行パネル・2D 動的分配のどちらの
    // 経路でも共有するブロックサイズ（`gemm_blis_parallel_with_transpose`
    // と同じ理由）。
    let blocks = default_blocks();

    // GEMM 本体の分岐は `gemm_blis_parallel_with_transpose` と同一の
    // 単一 const ゲート（[`TWO_D_DYNAMIC_PRODUCTION_ENABLED`]。イシュー
    // #1313）を共有する。`Nn` 固定のため `gemm_blis_bias_act_parallel`
    // 自体は `transpose` 引数を持たない。
    //
    // epilogue（bias 加算・activation）は要素ごとに独立な演算で
    // job／パネルの分割順序に依存しないため（本関数冒頭のドキュメン
    // テーションコメント「bit 完全一致契約」参照）、GEMM 本体が
    // どちらの経路でも完了後に `c` 全体へ 1 回だけ適用する（`TwoDDynamic`
    // 経路では各要素がちょうど 1 job に属するため多重適用が構造的に
    // 起きない。設計 `docs/cpu-gemm-2d-dynamic-partition-design.md`
    // §12「job ごとに K 全域完了後 1 回」の「または join 後に全体へ 1 回」
    // 案を採用）。
    // GB10 小形状 GEMM 大コア affinity ルーティング（イシュー #1576。
    // `gemm_blis_parallel_with_transpose` と同じ理由・同じヘルパ。
    // `crate::gb10_affinity` モジュール doc「適用範囲」節参照）。M4 Max
    // 側の仕事量ベース並列度上限（イシュー #1575・
    // [`crate::small_shape_thread_cap`]）とは自機判定で排他する別機構。
    crate::gb10_affinity::with_gb10_affinity_if_applicable(m, n, k, || {
        if TWO_D_DYNAMIC_PRODUCTION_ENABLED {
            // 小形状 GEMM の仕事量ベース rayon 並列度上限（イシュー #1575。
            // 上記 `gemm_blis_parallel_with_transpose` と同じ理由・同じ
            // 機構を共有する。epilogue はプール切替と無関係に GEMM 本体
            // 完了後へ適用する）。
            crate::small_shape_thread_cap::run_capped(m, n, k, || {
                dispatch_two_d_dynamic(
                    a,
                    b,
                    c,
                    n,
                    k,
                    0..m,
                    blocks,
                    GemmTranspose::Nn,
                    TWO_D_JOBS_PER_WORKER,
                )
            })?;
            apply_epilogue(c, n, bias, act)
        } else {
            // 行パネル分割数の実効スレッド数への差し替えは
            // `gemm_blis_parallel_with_transpose` と同じ理由（イシュー
            // #1363。同関数の実装コメント参照）。GB10 affinity プール内
            // で実行される場合の挙動も同関数と同じ（#1576）。
            let num_threads =
                crate::thread_limit::effective_num_threads(rayon::current_num_threads());
            let panel_rows = m.div_ceil(num_threads).max(1);
            // GEMM 本体は `gemm_blis_parallel_with_transpose` と同じ理由で
            // B パネル共有経路（`dispatch_shared_b`）を採用せず、従来どおり
            // 行パネルごとに `dispatch_region` を独立呼び出しする
            // （#1313 以前と同一の分岐。`par_chunks_mut(panel_rows * n)` で
            // epilogue も行パネル並列に適用することで、T=1
            // （`panel_rows == m` で単一チャンク）では従来と同一の 1 パスに
            // なる）。
            c.par_chunks_mut(panel_rows * n)
                .enumerate()
                .try_for_each(|(panel_idx, c_chunk)| {
                    let row_start = panel_idx * panel_rows;
                    let row_end = (row_start + c_chunk.len() / n).min(m);
                    dispatch_region(
                        a,
                        b,
                        c_chunk,
                        n,
                        k,
                        row_start..row_end,
                        blocks,
                        GemmTranspose::Nn,
                    )?;
                    apply_epilogue(c_chunk, n, bias, act)
                })
        }
    })
}

/// [`gemm_blis_parallel`] の**テスト専用**ワークロード閾値直列
/// フォールバック適用版（イシュー #811・#1027・[`GEMM_THREADING_THRESHOLD`]
/// ドキュメント「本番未結線」参照）。
///
/// M4 Max 実機ゲート未通過のうちは本番公開入口（[`gemm_blis_parallel`]）
/// から [`should_serialize`] を参照しないため、境界形状での
/// フォールバック挙動（bit 完全一致・非劣化）を検証し続けるための
/// テスト専用エントリを別途設ける（`dispatch_shared_b`〈#750〉と同じ
/// 「本番未結線でも検証は維持する」既存パターン。#1027 codex-review
/// P1 指摘: 削除した本関数を復元し、候補ディスパッチ〈[`should_serialize`]
/// による分岐〉を実際に通す回帰経路を維持する）。`m == 1` は
/// [`gemm_blis`]（内部で [`gemm_row_vector`] 専用経路へ迂回する）へも
/// [`gemm_blis_parallel`]（同じく `gemm_row_vector` へ迂回）へも同じ
/// 結果を返すため、[`should_serialize`] が `true` なら [`gemm_blis`]
/// （直列）、`false` なら [`gemm_blis_parallel`]（並列。実機ゲート通過後
/// の本番結線候補と同一コード経路）へ振り分けるだけで、本番結線時の
/// inline 判定（`if should_serialize(m, n, k) { ... } else { ... }`）と
/// 等価になる。
#[cfg(test)]
fn gemm_blis_parallel_thresholded(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    if should_serialize(m, n, k) {
        gemm_blis(a, b, c, m, n, k)
    } else {
        gemm_blis_parallel(a, b, c, m, n, k)
    }
}

/// [`gemm_blis_bias_act_parallel`] の**テスト専用**ワークロード閾値
/// 直列フォールバック適用版（[`gemm_blis_parallel_thresholded`] と同じ
/// 理由・同じ閾値。イシュー #811・#1027）。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn gemm_blis_bias_act_parallel_thresholded(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
    act: Activation,
) -> Result<(), GemmError> {
    // 本番側 `gemm_blis_bias_act_parallel` と同じ順序で検証してから
    // n == 0 の早期 return へ入る（`validate_dims`／bias 長／activation
    // 検証を経ずに Ok(()) を返すと、不正な n == 0 入力が本来のエラーを
    // 素通りしてしまう。PR #830 Cursor Bugbot 指摘）。
    validate_dims(a, b, c, m, n, k)?;
    if let Some(bias) = bias
        && bias.len() != n
    {
        return Err(GemmError::BiasLenMismatch {
            expected: n,
            actual: bias.len(),
        });
    }
    if !matches!(act, Activation::None | Activation::Relu) {
        return Err(GemmError::UnsupportedActivation);
    }

    // n == 0 は `gemm_blis_bias_act_parallel` と同じ理由（本関数冒頭の
    // ドキュメンテーションコメント参照）で shape として合法な no-op。
    // `should_serialize` は n == 0 のとき `n.max(NR_CLAMP)` が
    // `NR_CLAMP` へクランプされるため常に `true`（直列側）と判定し
    // 下記の直列分岐へ入るが、この早期 return が無いと
    // `apply_epilogue(c, 0, Some(bias), act)` が `chunks_mut(0)` で
    // パニックする（本番側 `gemm_blis_bias_act_parallel` は分岐前に
    // n == 0 を早期 return しており契約が一致していなかった。イシュー
    // #811 レビュー指摘）。
    if n == 0 {
        return Ok(());
    }
    if should_serialize(m, n, k) {
        gemm_blis(a, b, c, m, n, k)?;
        apply_epilogue(c, n, bias, act)
    } else {
        gemm_blis_bias_act_parallel(a, b, c, m, n, k, bias, act)
    }
}

/// [`gemm_blis_bias_act_parallel`] の epilogue 適用部（bias 行ブロード
/// キャスト加算 → activation）。`c_panel` は行パネル 1 つ分（`rows * n`
/// 要素、`n` 単位の行区切り）を対象とし、境界検査は行・列とも `n`
/// 由来のスライス長で構造的に保証する（明示 `assert`／`unsafe` を要しない。
/// REQ-8）。
///
/// `Activation` は `#[non_exhaustive]`（`fandhe_ai_tensor_core::backend_ops`）のため
/// `_ =>` で未知 variant を静かに無視せず、[`GemmError::UnsupportedActivation`]
/// を返す（未対応 activation を無視して不正な結果を返す fail-open を避ける。
/// `tensor-core` と同一ワークスペースで管理されるため通常到達しないが、
/// variant 追加時に本関数の更新漏れがあれば早期に検出できる）。
fn apply_epilogue(
    c_panel: &mut [f32],
    n: usize,
    bias: Option<&[f32]>,
    act: Activation,
) -> Result<(), GemmError> {
    if let Some(bias) = bias {
        for row in c_panel.chunks_mut(n) {
            for (x, b) in row.iter_mut().zip(bias.iter()) {
                *x += *b;
            }
        }
    }
    match act {
        Activation::None => {}
        Activation::Relu => {
            for x in c_panel.iter_mut() {
                *x = x.max(0.0);
            }
        }
        _ => return Err(GemmError::UnsupportedActivation),
    }
    Ok(())
}

/// 検出済みトークンを優先順位（Avx512 > Avx2 > Scalar）で直接 `try_new`
/// し、最初に構築できたトークンで [`gemm_blis_region`] を呼ぶ（実行時
/// dispatch の唯一の入口。`gemm_blis`／`gemm_blis_parallel` の両方から
/// 呼ばれる）。[`microkernel::Isa::detect`]（[`microkernel::select_isa`]
/// と同じ優先順位ロジック）を経由せず `try_new` の成否だけで分岐する
/// ことで、本番経路に `unwrap`／`expect`／`unreachable!` を一切置かない
/// （`.claude/rules/coding-rust.md` の「本番経路で unwrap()/expect() を
/// 使わない」規約。`Isa::detect` 自体は単体テスト・将来のテレメトリ用の
/// introspection API として残す）。環境変数等による dispatch 上書きは
/// 設けない（OWASP A03・`.claude/rules/security.md`）。
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn dispatch_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
    transpose: GemmTranspose,
) -> Result<(), GemmError> {
    // AVX-512 経路は `avx512_stable` cfg（`backend-cpu` クレートルートの
    // `build.rs` が AVX-512F intrinsics のコンパイル可否を実測して発行。
    // [`microkernel::avx512`] モジュールドキュメント参照）が立っている
    // 場合のみ試す。立っていない rustc では
    // `Avx512Kernel` 自体がコンパイル対象外のため、AVX2 から直接試行する
    // （実行 CPU が AVX-512F を持っていてもコンパイラの stable 化状況に
    // 応じて AVX2 へフォールバックする。数値一致は ISA 間 bit 完全一致
    // 契約のため結果には影響しない）。
    let mc_total = rows.end - rows.start;
    #[cfg(avx512_stable)]
    if let Some(kernel) = microkernel::Avx512Kernel::try_new() {
        let mut bufs = PanelBuffers::new::<microkernel::Avx512Kernel>(n, k, mc_total, blocks);
        return gemm_blis_region(kernel, a, b, c, n, k, rows, &mut bufs, blocks, transpose);
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        let mut bufs = PanelBuffers::new::<microkernel::Avx2Kernel>(n, k, mc_total, blocks);
        gemm_blis_region(kernel, a, b, c, n, k, rows, &mut bufs, blocks, transpose)
    } else {
        let mut bufs = PanelBuffers::new::<ScalarKernel>(n, k, mc_total, blocks);
        gemm_blis_region(
            ScalarKernel,
            a,
            b,
            c,
            n,
            k,
            rows,
            &mut bufs,
            blocks,
            transpose,
        )
    }
}

/// aarch64 版 [`dispatch_region`]。NEON は baseline ISA のため
/// [`microkernel::Isa::detect`] は常に `Isa::Neon` を返す（実行時検出不要。
/// [`microkernel`] モジュールドキュメント参照）。
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn dispatch_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
    transpose: GemmTranspose,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    let mc_total = rows.end - rows.start;
    let mut bufs = PanelBuffers::new::<microkernel::NeonKernel>(n, k, mc_total, blocks);
    gemm_blis_region(
        microkernel::NeonKernel,
        a,
        b,
        c,
        n,
        k,
        rows,
        &mut bufs,
        blocks,
        transpose,
    )
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_region`]。実行時検出対象の
/// ISA を持たないため常に [`ScalarKernel`] を使う。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[allow(clippy::too_many_arguments)]
fn dispatch_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
    transpose: GemmTranspose,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    let mc_total = rows.end - rows.start;
    let mut bufs = PanelBuffers::new::<ScalarKernel>(n, k, mc_total, blocks);
    gemm_blis_region(
        ScalarKernel,
        a,
        b,
        c,
        n,
        k,
        rows,
        &mut bufs,
        blocks,
        transpose,
    )
}

/// `gemm_blis`／`gemm_blis_parallel` 共通の 5-loop 本体。`c` はパネル
/// 先頭が `row_start` 行目に対応するスライス（`crate::gemm::gemm_blocked_region`
/// と同じ相対オフセット規約）。引数検証は呼び出し元の公開入口で完了済み
/// の前提。
///
/// ループ順は jc（列ブロック）→ pc（K ブロック）→ ic（行ブロック）→
/// jr（NR 単位の列パネル）→ ir（MR 単位の行パネル）。B パネルは pc/jc
/// ブロックごとに 1 回だけ packing して ic ループ全体で再利用し、A パネルは
/// ic ブロックごとに 1 回だけ packing して jr ループ全体で再利用する
/// （BLIS/GotoBLAS2 の packing 再利用による bandwidth 削減。PoC-v2-1
/// README「設計判断」節と同じ狙い）。
///
/// `K::MR`／`K::NR`（[`Microkernel`] トレイトの定数）でタイル形状を決め、
/// `kernel.run_with_ldc(...)` で累積計算を呼ぶ（#185 でジェネリック化。ISA ごとに
/// 呼び出し元でモノモーフィックに特殊化されるため、この関数自体に
/// `unsafe` は現れない）。C タイルは [`MAX_TILE`]（全 ISA 中の MR*NR 最大値）
/// 固定長スタックバッファを確保し、`K::MR * K::NR` ぶんだけスライスして
/// 使う（ジェネリック const 式は stable Rust で使えないための対処。各
/// カーネルモジュールの `const _: () = assert!(MR * NR <= MAX_TILE);` が
/// この前提をコンパイル時に検査する）。
///
/// `bufs`（[`PanelBuffers`]）は呼び出し元（[`dispatch_region`]）が
/// カーネル型確定直後に 1 回確保した A/B panel バッファで、jc×pc×ic の
/// 全反復を通じて使い回す（#556。以前は各反復で `vec![...]` を都度確保
/// していた。M=N=K=4096・MC=128/KC=256/NC=512 換算で B は 128 回・A は
/// 4,096 回のヒープ確保がこの 1 回確保へ削減される。実測は
/// `docs/perf/cpu-gemm-packing-buffer-reuse.md`）。各反復は `bufs` の
/// 先頭から必要長ぶんのサブスライスを再借用するのみで、`pack_a`／
/// `pack_b` が呼び出しのたびに有効レーンを全上書き（端タイルは
/// `dst.fill(0.0)` してから書く。`pack.rs` 参照）するため前反復の残留値
/// には依存せず、bit 完全一致契約（REQ-2）・FMA 契約・累積順序は一切
/// 変更しない。
///
/// `#[allow(clippy::too_many_arguments)]`: `bufs`／`blocks` 追加により
/// 9 引数になる。本ファイル内の既存慣例（[`gemm_blis_bias_act_parallel`] の
/// 同 attribute 使用箇所と同方針）に従い、構造体化はせず素朴な引数列の
/// まま許容する。
///
/// `blocks`（[`BlockSizes`]）は #564 で MC/KC/NC をパラメータ化したもの。
/// K 方向の加算順序・C タイルのロード／書き戻し構造は `blocks` の値に
/// 依らず不変のため、任意の `blocks` で [`crate::gemm::gemm_naive`] との
/// bit 完全一致契約（REQ-2）が成立する（`gemm_blis_with_kernel_and_blocks`
/// のパリティテストで直接検証）。
/// ## エラー伝播（#691 レビュー P1 再指摘への対応）
///
/// 内部の `kernel.run_with_ldc(...)` 呼び出しは構築上（組み込みカーネルの
/// `MR`／`NR` はコンパイル時定数で 1 以上、完全タイル・端タイルいずれも
/// 必要長ちょうどのスライス／バッファを渡す）常に境界検査を満たすため
/// `TileBoundsError` を返さないはずだが、以前は `unwrap_or_else` +
/// `unreachable!` で `Result` を panic へ変換していた（AGENTS.md「本番
/// 経路の panic 禁止」・`.claude/rules/coding-rust.md`「本番経路で
/// unwrap()/expect() を使わない」への抵触）。本関数は `Result<(), GemmError>`
/// を返し、`?`（[`GemmError::MicrokernelTileBounds`] への `From` 変換）で
/// 呼び出し元まで型付きエラーとして伝播させる（[`GemmError`] の `#[non_exhaustive]`
/// により呼び出し元の網羅的 match は破壊しない）。
#[allow(clippy::too_many_arguments)]
fn gemm_blis_region<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k_dim: usize,
    rows: Range<usize>,
    bufs: &mut PanelBuffers,
    blocks: BlockSizes,
    transpose: GemmTranspose,
) -> Result<(), GemmError> {
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;

    // `Nn`／`Nt`: `a` 引数は通常どおり行優先 `[m,k]`。従来どおり行方向
    // オフセットぶんスライスし、以降 [`gemm_blis_ic_loop`] は「オフセット
    // 後の a」を 0 始まりの相対行位置で読む。
    // `Tn`: `a` 引数は実際には転置格納 `at`（論理形状 `[k_dim, m_total]`
    // 行優先。`m_total` は行方向の飛び幅ではなく列幅であるため
    // `a[row_start*k_dim..]` の単純スライスができない）。フルスライスの
    // まま渡し、絶対行位置（`row_offset + ic + ir`）で `pack_a_from_transposed`
    // を呼べるよう `row_offset`／`m_total` を [`gemm_blis_ic_loop`] へ渡す
    // （#1213）。
    let (a_for_loop, row_offset, m_total): (&[f32], usize, usize) = match transpose {
        GemmTranspose::Tn => (a, row_start, a.len() / k_dim.max(1)),
        GemmTranspose::Nn | GemmTranspose::Nt => (&a[row_start * k_dim..], 0, 0),
    };

    for jc in (0..n).step_by(blocks.nc) {
        let nc_len = blocks.nc.min(n - jc);
        for pc in (0..k_dim).step_by(blocks.kc) {
            let kc_len = blocks.kc.min(k_dim - pc);

            // B パネル packing: nc_len を NR 単位のブロックに分割し、各
            // ブロックを kc_len*NR 要素の連続領域として 1 本のバッファに
            // 詰める（ic ループ全体で使い回すため pc/jc ブロックごとに
            // 1 回のみ実行）。pack_b が panel サブスライスへ直接書き込む
            // ため中間 Vec 確保・copy_from_slice は発生しない（#554:
            // BLIS/matrixmultiply の呼び出し側確保バッファへ直接書き込む
            // packing 方式に合わせ二段コピーを廃止）。バッファ自体は
            // `bufs.b_panel`（呼び出し元 `dispatch_region` が 1 回確保
            // 済み）の先頭サブスライスを再借用する（#556。ループ内での
            // `vec![...]` 確保をゼロにする）。直列経路（本関数）は単一
            // タスクのみが呼ぶため、B packing はここでは並列化しない
            // （並列化版は [`gemm_blis_shared_b_region`] 側。#750）。
            //
            // `Nt` では `b` 引数が転置格納 `bt`（論理形状 `[n, k_dim]`
            // 行優先）のため [`pack_b_from_transposed`] へ分岐する
            // （#1213。`pack.rs` モジュールドキュメント参照）。
            let nr_blocks = nc_len.div_ceil(nr);
            let b_panel = &mut bufs.b_panel[..nr_blocks * kc_len * nr];
            for jr_block in 0..nr_blocks {
                let jr = jr_block * nr;
                let nr_eff = nr.min(nc_len - jr);
                let dst = &mut b_panel[jr_block * kc_len * nr..(jr_block + 1) * kc_len * nr];
                match transpose {
                    GemmTranspose::Nt => pack_b_from_transposed(
                        dst,
                        b,
                        BTPackTile {
                            k_total: k_dim,
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start: jc + jr,
                            nr,
                            nr_eff,
                        },
                    ),
                    GemmTranspose::Nn | GemmTranspose::Tn => pack_b(
                        dst,
                        b,
                        BPackTile {
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start: jc + jr,
                            nr,
                            nr_eff,
                        },
                    ),
                }
            }

            // ic（行パネル）以下のループ本体は [`gemm_blis_ic_loop`] へ
            // 切り出し済み（#750）。並列経路（[`gemm_blis_shared_b_region`]）
            // が「共有 B パネルをタスクごとの行範囲へ適用する」ために同じ
            // 関数を再利用するための共通化であり、本関数（直列経路）では
            // `bufs.a_panel` 全体・`mc_total`（このパネル全域）をそのまま
            // 渡すだけで従来どおりの計算になる。
            let ctx = IcLoopContext {
                b_panel,
                n,
                k_dim,
                pc,
                kc_len,
                jc,
                nr_blocks,
                nc_len,
                blocks,
            };
            gemm_blis_ic_loop(
                kernel,
                a_for_loop,
                c,
                mc_total,
                &mut bufs.a_panel,
                &ctx,
                transpose,
                row_offset,
                m_total,
            )?;
        }
    }
    Ok(())
}

/// [`gemm_blis_ic_loop`] へ渡す (jc,pc) ブロック文脈パラメータ（引数数を
/// 抑えるための束ね。[`pack::APackTile`]／[`pack::BPackTile`] と同じ設計
/// 判断。#750）。全フィールド `Copy`（`b_panel` は共有参照）のため
/// `#[derive(Clone, Copy)]` で値渡しできる。
#[derive(Clone, Copy)]
struct IcLoopContext<'b> {
    /// 呼び出し元が pc/jc ブロックごとに 1 回だけ pack 済みの B パネル
    /// （`nr_blocks * kc_len * nr` 要素）。[`gemm_blis_region`]（直列）
    /// では `bufs.b_panel` のサブスライス、[`gemm_blis_shared_b_region`]
    /// （並列・#750）ではタスク間で共有する 1 本のバッファのサブスライス。
    b_panel: &'b [f32],
    /// C の行ストライド（`ldc`）としてのみ使う（B の列幅としては使わない。
    /// `gemm_blis_ic_loop` 内の完全タイル直接アドレッシング
    /// `row0 = (ic+ir)*n + col_base` を参照）。この契約により、`n`（C の
    /// 実際の行幅）とは異なる値（job-local な連続バッファの行ストライド）
    /// を渡しても `gemm_blis_ic_loop` を無改変で再利用できる（2D 動的
    /// 分配 `TwoDDynamic` の job-local C staging〈S′〉方式。イシュー
    /// #1311・`run_two_d_job` 参照）。
    n: usize,
    k_dim: usize,
    pc: usize,
    kc_len: usize,
    jc: usize,
    nr_blocks: usize,
    nc_len: usize,
    blocks: BlockSizes,
}

/// `gemm_blis_region`（直列）・`gemm_blis_shared_b_region`（並列・共有 B。
/// #750）共通の ic（行パネル）以下のループ本体（ic→jr→ir）。
///
/// 呼び出し元が pc/jc ブロックごとに 1 回だけ pack 済みの `ctx.b_panel`
/// を読み取り専用で受け取り、`a`（このタスクが担当する行範囲の先頭が
/// 行 0 に対応するよう既にオフセット済みのスライス）・`c`（同じく行 0
/// 対応でオフセット済み）に対して `mc_total` 行ぶんの A packing・カーネル
/// 呼び出しを行う。A packing 用バッファ `a_panel` は呼び出し元が所有する
/// もの（直列経路は `bufs.a_panel`、並列経路はタスクローカルな
/// `Vec<f32>`）を可変参照で受け取り、`gemm_blis_region` から移設した
/// ロジック自体は一切変更していない（ic/jr/ir の反復順・`pack_a` の
/// 書き込み内容・C タイルのロード/書き戻し構造がそのままのため、
/// bit 完全一致契約〈REQ-2〉・FMA 契約・累積順序を保つ。#750 実装計画
/// §4.1「誰がいつ pack したか」だけが変わるという設計根拠の直接反映）。
#[allow(clippy::too_many_arguments)]
fn gemm_blis_ic_loop<K: Microkernel>(
    kernel: K,
    a: &[f32],
    c: &mut [f32],
    mc_total: usize,
    a_panel: &mut [f32],
    ctx: &IcLoopContext,
    transpose: GemmTranspose,
    row_offset: usize,
    m_total: usize,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let IcLoopContext {
        b_panel,
        n,
        k_dim,
        pc,
        kc_len,
        jc,
        nr_blocks,
        nc_len,
        blocks,
    } = *ctx;

    let mut ic = 0;
    while ic < mc_total {
        let mc_len = blocks.mc.min(mc_total - ic);

        // A パネル packing: mc_len を MR 単位のブロックに分割（jr ループ
        // 全体で使い回すため ic ブロックごとに 1 回のみ）。pack_a が
        // panel サブスライスへ直接書き込むため中間 Vec 確保・
        // copy_from_slice は発生しない（#554。B packing と同じ理由）。
        //
        // `Tn` では `a` 引数が転置格納 `at`（論理形状 `[k_dim, m_total]`
        // 行優先。`gemm_blis_region` がフルスライスのまま渡している）
        // のため [`pack_a_from_transposed`] へ分岐し、行位置は呼び出し元
        // 由来の絶対オフセット `row_offset + ic + ir` を渡す（#1213。
        // `pack.rs` モジュールドキュメント参照）。
        let mr_blocks = mc_len.div_ceil(mr);
        let a_panel = &mut a_panel[..mr_blocks * kc_len * mr];
        for ir_block in 0..mr_blocks {
            let ir = ir_block * mr;
            let mr_eff = mr.min(mc_len - ir);
            let dst = &mut a_panel[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr];
            match transpose {
                GemmTranspose::Tn => pack_a_from_transposed(
                    dst,
                    a,
                    ATPackTile {
                        k_total: k_dim,
                        m_total,
                        row_start: row_offset + ic + ir,
                        mr,
                        mr_eff,
                        kc_start: pc,
                        kc_len,
                    },
                ),
                GemmTranspose::Nn | GemmTranspose::Nt => pack_a(
                    dst,
                    a,
                    APackTile {
                        k_total: k_dim,
                        row_start: ic + ir,
                        mr,
                        mr_eff,
                        kc_start: pc,
                        kc_len,
                    },
                ),
            }
        }

        for jr_block in 0..nr_blocks {
            let jr = jr_block * nr;
            let nr_eff = nr.min(nc_len - jr);
            let bp_slice = &b_panel[jr_block * kc_len * nr..(jr_block + 1) * kc_len * nr];

            for ir_block in 0..mr_blocks {
                let ir = ir_block * mr;
                let mr_eff = mr.min(mc_len - ir);
                let ap_slice = &a_panel[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr];
                let col_base = jc + jr;

                if mr_eff == mr && nr_eff == nr {
                    // 完全タイル（#557）: C の実バッファへ
                    // `Microkernel::run_with_ldc` の `ldc` 契約経由で直接
                    // ロード/ストアし、コピーイン/コピーアウトの往復を
                    // 省く。`row0` はこのタイル原点（行 ic+ir・列
                    // col_base）の C 上のオフセットで、サブスライス長
                    // `(mr-1)*n + nr` は行 mr-1・列 nr-1 までを覆う
                    // （`ldc = n`）。完全タイルゆえ `col_base + nr <= n`
                    // が成立し `ldc(=n) >= nr` も自動的に満たされる
                    // （[`microkernel::Microkernel::run_with_ldc`] の
                    // `ldc` 契約参照）。スライス取得自体が範囲外なら
                    // panic する安全操作であり、カーネル入口の `ldc`／
                    // 長さ検査と合わせ REQ-8 の境界検査を二重に満たす。
                    // `run_with_ldc` は外部の `Microkernel` 実装からも
                    // 到達しうる公開入口のため `Result` を返す契約
                    // （#691 レビュー P1 対応）で、本呼び出しは private
                    // な本関数内部から組み込みカーネル（`ScalarKernel`／
                    // `NeonKernel`／`Avx2Kernel`／`Avx512Kernel`。いずれも
                    // `MR`／`NR` はモジュール定数でコンパイル時に 1
                    // 以上）へ、完全タイルゆえ自動的に満たされる
                    // `ldc(=n) >= nr` と、上記スライス長
                    // `(mr-1)*n+nr`（= 必要長そのもの）で呼ぶため、
                    // 境界検査は構築上常に成功するはずだが、
                    // `unreachable!` による panic 変換（#691 レビュー
                    // P1 再指摘）を避け、`?` で
                    // `GemmError::MicrokernelTileBounds` として呼び出し
                    // 元まで型付きエラーで伝播させる（実際に `Err` に
                    // なることは想定していない fail-safe だが、本番経路
                    // の panic 禁止規約を優先する）。
                    let row0 = (ic + ir) * n + col_base;
                    let c_direct = &mut c[row0..row0 + (mr - 1) * n + nr];
                    kernel.run_with_ldc(ap_slice, bp_slice, c_direct, n, kc_len)?;
                } else {
                    // 端タイル: 従来どおり `MAX_TILE` スタックバッファへ
                    // コピーインし、有効部（mr_eff×nr_eff）のみコピー
                    // バックする（padding レーン mr_eff..mr, nr_eff..nr
                    // はゼロのままでよい。書き戻し時に不使用）。
                    // ir_block×jr_block のたびのヒープ確保を避けるため
                    // 固定長スタック配列を使う（Review 指摘: M=N=K=2048
                    // では ir/jr ループの反復数が数十万に達し `Vec`
                    // 確保が無視できないオーバーヘッドになるため）。
                    let mut c_tile_buf = [0.0f32; MAX_TILE];
                    let c_tile = &mut c_tile_buf[..mr * nr];
                    for i in 0..mr_eff {
                        let src =
                            &c[(ic + ir + i) * n + col_base..(ic + ir + i) * n + col_base + nr_eff];
                        c_tile[i * nr..i * nr + nr_eff].copy_from_slice(src);
                    }

                    // `ldc = nr` は組み込みカーネルの `NR` 定数そのもの
                    // であり `c_tile` は `mr*nr` ちょうどの長さで確保
                    // しているため、境界検査は構築上常に成功するはず
                    // （上記完全タイル分岐と同じ根拠）。同様に `?` で
                    // 型付きエラーとして伝播させ、`unreachable!` による
                    // panic 変換を避ける（#691 レビュー P1 再指摘）。
                    kernel.run_with_ldc(ap_slice, bp_slice, c_tile, nr, kc_len)?;

                    for i in 0..mr_eff {
                        let dst = &mut c
                            [(ic + ir + i) * n + col_base..(ic + ir + i) * n + col_base + nr_eff];
                        dst.copy_from_slice(&c_tile[i * nr..i * nr + nr_eff]);
                    }
                }
            }
        }

        ic += blocks.mc;
    }
    Ok(())
}

/// `#[cfg(test)]` の [`gemm_blis_parallel_with_blocks`] の複数タスク経路
/// （実タスク数 >= 2）が呼ぶ、B パネルをタスク間で 1 本だけ共有する
/// 5-loop 本体（イシュー #750・設計 doc
/// `docs/cpu-gemm-b-packing-sharing-decision.md` 案 B）。本番公開入口
/// （[`gemm_blis_parallel`]／[`gemm_blis_bias_act_parallel`]）からは
/// 呼ばれない（下記「本番未結線」節参照）。
///
/// jc/pc ループは直列（[`gemm_blis_region`] と同じ昇順）のまま、各
/// (jc,pc) ブロックで B を 1 本だけ pack して `&[f32]` として全タスクへ
/// 共有し、ic（行パネル）だけをタスク間で並列化する（[`gemm_blis_ic_loop`]
/// を「可変 pack → 不変 `&[f32]` 共有読み」の借用分割で呼ぶ。データ競合は
/// コンパイル時に排除される。`unsafe` は使わない）。C 各要素の FMA 連鎖
/// （p 昇順）・`pack_a`／`pack_b` の書き込み内容は [`gemm_blis_region`]
/// と一切変わらないため、`gemm_naive` との bit 完全一致契約（REQ-2）を
/// 保つ（「誰がいつ pack したか」だけが変わる）。
///
/// B packing 自体も `nr` ブロック単位で `par_chunks_mut` により並列化する
/// （各チャンクは書き込み先が排他かつ内容が他チャンクに依存しないため
/// 実行順序に関わらず結果は直列版と同一。#750）。
///
/// A パネル用バッファはタスクごとに 1 本ずつ（`Vec` の所有権がタスク
/// ローカルに閉じるため事前一括確保＋オフセット分割を要さずコンパイル時
/// にデータ競合が排除される。[`PanelBuffers`] ドキュメントコメントが
/// 「タスクごとに 1 組所有する」と述べていた従来設計をそのまま踏襲する）
/// gemm 呼び出しの (jc,pc) ループ全体で使い回すため、ループ外で 1 回だけ
/// 確保する（#556 の確保削減方針をタスク数ぶんに拡張）。
///
/// **本番未結線（#750・codex-review P1 是正）**: 本関数・
/// [`dispatch_shared_b`] は本番公開入口（[`gemm_blis_parallel`]／
/// [`gemm_blis_bias_act_parallel`]）からは呼ばれない
/// （`docs/perf/cpu-gemm-b-packing-sharing.md` 参照。受け入れ条件 2
/// ＝ Apple M4 Max 実測非劣化確認を満たすまでの採用ゲート）。
/// `#[cfg(test)]`（`#[cfg(test)]` の [`gemm_blis_parallel_with_blocks`]
/// 経由）で bit 完全一致検証のみ行う。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn gemm_blis_shared_b_region<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k_dim: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;
    let a = &a[row_start * k_dim..];

    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let panel_rows = mc_total.div_ceil(num_threads).max(1);
    let num_tasks = mc_total.div_ceil(panel_rows);

    // 共有 B バッファ: 全 jc ブロック中の最大 nc_len に対応する 1 本
    // （`panel_capacity` の B 長算出は `mc_total` に依存しない）。
    let (b_cap, _) = panel_capacity(n, k_dim, mc_total, mr, nr, blocks);
    let mut b_panel_buf = vec![0.0f32; b_cap];

    // タスクローカル A バッファ: 各タスクが担当しうる最大行数
    // （`panel_rows`。最終タスクのみこれより少ない行数を担当しうるが、
    // `panel_capacity` は `mc_total` に対して単調非減少なため
    // `panel_rows` を渡した容量が全タスクの上界になる）に対応する容量で
    // 1 回だけ確保し、(jc,pc) 反復間で使い回す。
    let (_, a_cap) = panel_capacity(n, k_dim, panel_rows, mr, nr, blocks);
    let mut a_bufs: Vec<Vec<f32>> = (0..num_tasks).map(|_| vec![0.0f32; a_cap]).collect();

    for jc in (0..n).step_by(blocks.nc) {
        let nc_len = blocks.nc.min(n - jc);
        for pc in (0..k_dim).step_by(blocks.kc) {
            let kc_len = blocks.kc.min(k_dim - pc);
            let nr_blocks = nc_len.div_ceil(nr);
            let b_panel = &mut b_panel_buf[..nr_blocks * kc_len * nr];

            // B packing の nr ブロック単位並列化（#750）。各チャンクは
            // 排他的な書き込み先で内容も他チャンクに依存しないため、
            // 実行順序に関わらず結果は直列版（[`gemm_blis_region`]）と
            // 同一（bit 完全一致契約に影響しない）。
            b_panel
                .par_chunks_mut(kc_len * nr)
                .enumerate()
                .for_each(|(jr_block, dst)| {
                    let jr = jr_block * nr;
                    let nr_eff = nr.min(nc_len - jr);
                    pack_b(
                        dst,
                        b,
                        BPackTile {
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start: jc + jr,
                            nr,
                            nr_eff,
                        },
                    );
                });

            let b_panel_ref: &[f32] = b_panel;
            let ctx = IcLoopContext {
                b_panel: b_panel_ref,
                n,
                k_dim,
                pc,
                kc_len,
                jc,
                nr_blocks,
                nc_len,
                blocks,
            };

            // ic（行パネル）のタスク間並列化。`c`（この呼び出しが担当する
            // 行範囲全体）を `panel_rows` 行ずつに分割し、各タスクが
            // `b_panel_ref`（この (jc,pc) ブロックで 1 回だけ pack 済み・
            // 全タスク共有の読み取り専用スライス）を参照しつつ、自分の
            // タスクローカル A バッファへ packing して計算する
            // （データ競合はコンパイル時の借用分割で排除。`unsafe` 不要）。
            c.par_chunks_mut(panel_rows * n)
                .enumerate()
                .zip(a_bufs.par_iter_mut())
                .try_for_each(|((task_idx, c_chunk), a_buf)| {
                    let task_mc = c_chunk.len() / n;
                    if task_mc == 0 {
                        return Ok(());
                    }
                    let task_row_start = task_idx * panel_rows;
                    let a_task = &a[task_row_start * k_dim..];
                    gemm_blis_ic_loop(
                        kernel,
                        a_task,
                        c_chunk,
                        task_mc,
                        a_buf,
                        &ctx,
                        GemmTranspose::Nn,
                        0,
                        0,
                    )
                })?;
        }
    }
    Ok(())
}

/// 検出済みトークンを優先順位（Avx512 > Avx2 > Scalar）で直接 `try_new`
/// し、最初に構築できたトークンで [`gemm_blis_shared_b_region`] を呼ぶ
/// （[`dispatch_region`] の共有 B 版・実タスク数 >= 2 の並列経路専用の
/// dispatch 入口。イシュー #750）。ロジックは `dispatch_region` と同一で、
/// 呼ぶ先の関数のみが異なる。本番未結線（[`gemm_blis_shared_b_region`]
/// ドキュメンテーションコメント参照）のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "x86_64")]
fn dispatch_shared_b(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    #[cfg(avx512_stable)]
    if let Some(kernel) = microkernel::Avx512Kernel::try_new() {
        return gemm_blis_shared_b_region(kernel, a, b, c, n, k, rows, blocks);
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        gemm_blis_shared_b_region(kernel, a, b, c, n, k, rows, blocks)
    } else {
        gemm_blis_shared_b_region(ScalarKernel, a, b, c, n, k, rows, blocks)
    }
}

/// aarch64 版 [`dispatch_shared_b`]（#750）。[`dispatch_region`] の
/// aarch64 版と同じ理由で NEON 固定（実行時検出不要）。本番未結線
/// （[`gemm_blis_shared_b_region`] ドキュメンテーションコメント参照）
/// のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
fn dispatch_shared_b(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    gemm_blis_shared_b_region(microkernel::NeonKernel, a, b, c, n, k, rows, blocks)
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_shared_b`]（#750）。
/// [`dispatch_region`] の同 arch 版と同じ理由で [`ScalarKernel`] 固定。
/// 本番未結線（[`gemm_blis_shared_b_region`] ドキュメンテーションコメント
/// 参照）のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dispatch_shared_b(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    gemm_blis_shared_b_region(ScalarKernel, a, b, c, n, k, rows, blocks)
}

/// テスト専用: 実行環境の実際の ISA 検出結果に依らず、指定したカーネル
/// トークンを強制して [`gemm_blis`] 相当の計算を行う（受け入れ条件
/// 「非対応環境でスカラーフォールバックが動作する」を環境非依存で検証
/// するためのヘルパー。`#[cfg(test)]` 到達可能な `pub(crate)` として公開し、
/// 統合テスト側からは使わない〈lib 単体テストの `mod tests` から使う〉）。
#[cfg(test)]
pub(crate) fn gemm_blis_with_kernel<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    let mut bufs = PanelBuffers::new::<K>(n, k, m, default_blocks());
    gemm_blis_region(
        kernel,
        a,
        b,
        c,
        n,
        k,
        0..m,
        &mut bufs,
        default_blocks(),
        GemmTranspose::Nn,
    )
}

/// [`gemm_blis_with_kernel_and_blocks`]／[`gemm_blis_parallel_with_blocks`]
/// （実行時に任意の [`BlockSizes`] を受け付けるパラメータ化入口。#564
/// スイープ基盤）が [`PanelBuffers::new`] へ到達する前に検証する:
/// `mc`／`kc`／`nc` のいずれかが 0 だと `gemm_blis_region` 内の
/// `step_by(0)`（`crate::gemm::gemm_blocked_region` の同種バグ〈Cursor
/// Bugbot #231〉と同じ既知の危険）でパニックするため、`GemmError::
/// ZeroBlockSize`（`crate::gemm` 側で既に定義済みのエラー variant。
/// gemm/gemm_blis 間で共有する `GemmError` 型のため新規 variant 追加は
/// 不要）を再利用して早期拒否する（OWASP A03・`.claude/rules/security.md`）。
///
/// panel 容量計算（[`panel_capacity`]）の乗算オーバーフローについては、
/// 本関数の引数として渡る `n`／`k_dim`／`mc_total` が呼び出し元
/// `validate_dims` を先に通過済み（`m*k`／`k*n`／`m*n` が `usize` に収まると
/// 検証済み）であることと、`panel_capacity` が常に `blocks.{mc,kc,nc}
/// .min(dim)` でクランプしてから乗算する構造（[`panel_capacity`] 参照）
/// により、`blocks` 側にどれだけ大きな値（firestorm 参照値
/// KC=4096/NC=9600 等）を渡しても実際の乗算対象は非オーバーフロー確定
/// 済みの dim 由来値に収まるため、追加のオーバーフロー検査は実装上
/// 到達不能と判断し設けない（0 値検査のみで fail-closed 契約を満たす）。
#[cfg(test)]
fn validate_block_sizes(blocks: BlockSizes) -> Result<(), GemmError> {
    if blocks.mc == 0 || blocks.kc == 0 || blocks.nc == 0 {
        return Err(GemmError::ZeroBlockSize {
            mc: blocks.mc,
            kc: blocks.kc,
            nc: blocks.nc,
        });
    }
    Ok(())
}

/// テスト・スイープ専用: [`gemm_blis_with_kernel`] の任意 `BlockSizes` 版
/// （#564）。実運用経路（[`gemm_blis`] 等）は [`default_blocks`]
/// （コンパイル時定数）のみを渡すためこの入口は通過しない。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_blis_with_kernel_and_blocks<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;
    let mut bufs = PanelBuffers::new::<K>(n, k, m, blocks);
    gemm_blis_region(
        kernel,
        a,
        b,
        c,
        n,
        k,
        0..m,
        &mut bufs,
        blocks,
        GemmTranspose::Nn,
    )
}

/// テスト専用: [`gemm_blis_shared_b_region`]（B パネル共有経路。#750）の
/// bit 完全一致を任意 `BlockSizes`（#564）で検証するための入口。
/// `gemm_blis_parallel`（本番公開入口）は共有経路を採用しない
/// （[`gemm_blis_shared_b_region`] ドキュメンテーションコメント参照）ため、
/// 本関数は `gemm_blis_parallel` 本体とは分岐が異なる（実タスク数 1 なら
/// `dispatch_region`・2 以上なら `dispatch_shared_b` を常に経由し、
/// 共有経路のテストカバレッジを維持する）。`blocks` のみ呼び出し元から
/// 注入できる。
#[cfg(test)]
pub(crate) fn gemm_blis_parallel_with_blocks(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    // n == 0 の no-op 対処は `gemm_blis_parallel` と同じ理由（本ファイル
    // 冒頭のコメント参照）。
    if n == 0 {
        return Ok(());
    }

    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let panel_rows = m.div_ceil(num_threads).max(1);

    if m <= panel_rows {
        return dispatch_region(a, b, c, n, k, 0..m, blocks, GemmTranspose::Nn);
    }
    dispatch_shared_b(a, b, c, n, k, 0..m, blocks)
}

/// A/B 計測（`GemmDriverVariant::RowPanel`。イシュー #1041）専用: 本番
/// 公開入口 [`gemm_blis_parallel`] とロジックを完全一致させつつ `blocks`
/// のみ注入できるようにした入口。
///
/// [`gemm_blis_parallel_with_blocks`]（テスト専用の共有 B 経路カバレッジ
/// 維持関数）は実タスク数 2 以上で常に [`dispatch_shared_b`] へ分岐する
/// ため、`RowPanel`（本番既定と同一であることが前提の A/B 計測基準線）
/// がそちらを誤って計測してしまう指摘（PR #1075 codex-review・Cursor
/// Bugbot。ともに同一箇所を独立検出）を受けて追加した。[`gemm_blis_parallel`]
/// 本体（タスク数に関わらず常に [`dispatch_region`] を行パネルごとに
/// 独立呼び出し）と分岐を完全一致させ、`blocks` の注入だけを追加する。
#[cfg(test)]
pub(crate) fn gemm_blis_parallel_row_panel_with_blocks(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    // n == 0／m == 1 の no-op・専用経路対処は呼び出し元
    // `gemm_blis_parallel_variant` が既に行っている（本関数は
    // `RowPanel` 用の内部ディスパッチのみを担う）ため、ここでは
    // `gemm_blis_parallel` 本体の rayon 行パネル分割のみを再現する。
    if n == 0 {
        return Ok(());
    }

    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let panel_rows = m.div_ceil(num_threads).max(1);

    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .try_for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            dispatch_region(
                a,
                b,
                c_chunk,
                n,
                k,
                row_start..row_end,
                blocks,
                GemmTranspose::Nn,
            )
        })
}

/// [`gemm_blis_parallel_row_panel_with_blocks`] の任意マイクロカーネル版
/// （イシュー #1317・`GemmDriverVariant::RowPanelBLaneqVec` 用）。
/// `dispatch_region`（本番駆動経路。実行時 ISA 検出で `NeonKernel` 等を
/// 選ぶ）を経由せず、呼び出し元が指定した `K: Microkernel` へ直接
/// `gemm_blis_region` を呼ぶ点のみが差分（行パネル分割・rayon 並列化の
/// ロジックは完全に同一）。これにより `RowPanel`（本番既定）との A/B
/// 比較の差分をマイクロカーネル自体（laneq のベクトル転置化）だけに
/// 限定できる（計画 §3.4）。`#[cfg(test)]` の A/B 計測専用入口であり
/// `dispatch_region`／本番公開入口は変更しない。aarch64 限定（唯一の
/// 呼び出し元 [`gemm_blis_parallel_variant`] の `RowPanelBLaneqVec`
/// アームが aarch64 限定のため、他アーキでは未使用関数になり
/// `-D warnings` の `dead_code` lint に抵触する）。
#[cfg(all(test, target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn gemm_blis_parallel_row_panel_with_kernel<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    if n == 0 {
        return Ok(());
    }

    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let panel_rows = m.div_ceil(num_threads).max(1);

    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .try_for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            let mc_total = row_end - row_start;
            let mut bufs = PanelBuffers::new::<K>(n, k, mc_total, blocks);
            gemm_blis_region(
                kernel,
                a,
                b,
                c_chunk,
                n,
                k,
                row_start..row_end,
                &mut bufs,
                blocks,
                GemmTranspose::Nn,
            )
        })
}

/// #753: MC タイル境界に整列した行範囲分配（[`partition::row_ranges_for_workers`]）
/// を使う `gemm_blis_parallel` の 2 次元タイルジョブ分配版。
///
/// 従来の `panel_rows = m.div_ceil(num_threads)` による静的パネル分割
/// （[`gemm_blis_parallel`]）は、MC タイル**数**が `num_threads` で
/// 割り切れない形状では端数タイルを特定 worker へ偏らせる（#753 実装
/// 計画 §3.2）。本関数は行バンド数を [`partition::split_evenly`]
/// （gemm crate `gemm.rs` の n_jobs 分配方式を参照した均等割り）で
/// 均等化してから行範囲へ変換することで、その偏りを ±1 タイルへ抑える。
///
/// 各 worker が受け取る行範囲は依然として `[0, m)` を隙間なく分割した
/// disjoint な連続区間であるため、`c.split_at_mut` の連鎖のみで
/// `unsafe` なしに実現できる。タイル単位で非連続に分配する完全な 2 次元
/// ジョブ分配（gemm crate 本来の方式）は生ポインタによる `unsafe`
/// ラッパーを要するため、PR #766（「常に不活性な sysctl FFI」が P0/P1
/// 指摘で撤去された経緯）を踏まえ #753 では採用しない（
/// [`partition`] モジュールドキュメント「unsafe を使わない設計判断」・
/// `docs/perf/cpu-gemm-runtime-cache-detect.md` 参照）。
///
/// 本番未結線（[`gemm_blis_parallel`]／[`gemm_blis_bias_act_parallel`]
/// からは呼ばれない。受け入れ条件 2＝実機 5 回中央値での非劣化確認が
/// 本 PR のスコープ外のため。#750・#758 と同型の判断）。テスト専用
/// パラメータ化入口として `blocks` を直接受け取る（[`gemm_blis_parallel_with_blocks`]
/// と同じ設計）。
#[cfg(test)]
pub(crate) fn gemm_blis_parallel_2d_with_blocks(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    // n == 0 の no-op 対処は `gemm_blis_parallel` と同じ理由（本ファイル
    // 冒頭のコメント参照）。
    if n == 0 {
        return Ok(());
    }

    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let row_ranges = partition::row_ranges_for_workers(m, blocks.mc, num_threads);

    // 安全な disjoint 分割: `row_ranges` は `[0, m)` を隙間なく連続分割
    // したものなので（[`partition::row_ranges_for_workers`] の被覆完全性
    // 契約。`partition::tests::row_ranges_for_workers_covers_m_contiguously_and_disjointly`
    // で検証済み）、`c` を先頭から順に `split_at_mut` で切り出せば各
    // worker の担当範囲が構築上重複しない（コンパイル時の借用検査で保証。
    // `unsafe` 不要）。
    let mut remaining: &mut [f32] = c;
    let mut chunks: Vec<(usize, &mut [f32])> = Vec::with_capacity(row_ranges.len());
    for r in &row_ranges {
        let len = (r.end - r.start) * n;
        let (head, tail) = remaining.split_at_mut(len);
        chunks.push((r.start, head));
        remaining = tail;
    }

    chunks.into_par_iter().try_for_each(|(row_start, c_chunk)| {
        let row_end = row_start + c_chunk.len() / n;
        dispatch_region(
            a,
            b,
            c_chunk,
            n,
            k,
            row_start..row_end,
            blocks,
            GemmTranspose::Nn,
        )
    })
}

/// [`gemm_blis_shared_b_pc_outer_region`] 用: 1 タスクが 1 pc ブロックで
/// 保持する A パネル容量（`task_mc` を `blocks.mc` にクランプしない点が
/// [`panel_capacity`] と異なる。同関数のドキュメント参照）。`mr_blocks *
/// kc_len_max * mr` の乗算は `checked_mul` でオーバーフローを検出し
/// `GemmError::DimProductOverflow` へ変換する（OWASP A03・
/// `.claude/rules/security.md`）。`task_mc`（タスク行数）は呼び出し元の
/// `mc_total`（`validate_dims` 済みの `m` 以下）由来のため実用上到達
/// しない想定だが、`unwrap` を使わず型付きエラーで fail-closed にする。
#[cfg(test)]
fn task_a_capacity(task_mc: usize, kc_len_max: usize, mr: usize) -> Result<usize, GemmError> {
    let mr_blocks = task_mc.div_ceil(mr.max(1));
    mr_blocks
        .checked_mul(kc_len_max)
        .and_then(|v| v.checked_mul(mr))
        .ok_or(GemmError::DimProductOverflow)
}

/// [`gemm_blis_ic_dynamic_region`] 専用: 動的配布する行パネル 1 枚あたりの
/// 行数（イシュー #1366）。
///
/// `SharedBPcOuter`（[`gemm_blis_shared_b_pc_outer_region`]）は
/// `mc_total.div_ceil(num_workers)` をそのままパネル行数にするため、
/// `blocks.mc` を跨ぐ大きなパネルになりうる（各パネルは 1 タスクへ
/// 静的に固定される前提のため問題にならない）。本 variant は行パネルを
/// `AtomicUsize` カウンタで動的配布するため、パネル数がワーカー数
/// 以上になるよう `blocks.mc` の上限も同時に満たす必要がある（さもないと
/// N=1024 のような中形状でパネル数 < ワーカー数となり、後発のワーカーが
/// 仕事を持てず動的配布の意味がない。issue #1366 実装計画 §3.2）。
///
/// 算出方針: `mc_total` をワーカー数で均等割りした行数を [`Microkernel::MR`]
/// の倍数へ切り上げ（カーネルタイル境界に揃える。切り上げても実際の
/// 端タイルは `mr_eff` で処理されるため境界検査・bit 一致契約には影響
/// しない）、`blocks.mc` を超えないようクランプし、最後に 1 未満になら
/// ないよう下限を敷く（`num_workers` や `mc_total` が極端な値でも
/// 0 除算・無限ループを生まない。呼び出し元は戻り値を `c.chunks_mut
/// (panel_rows * n)`（チャンクサイズ 0 はパニック）・`div_ceil` の除数
/// として使うため 0 は許容できない）。
#[cfg(test)]
fn ic_dynamic_panel_rows(mc_total: usize, mc: usize, mr: usize, num_workers: usize) -> usize {
    let num_workers = num_workers.max(1);
    let mr = mr.max(1);
    let per_worker = mc_total.div_ceil(num_workers).max(1);
    let aligned = per_worker.div_ceil(mr).saturating_mul(mr);
    mc.min(aligned).max(1)
}

/// [`gemm_blis_ic_dynamic_region`] 専用: pc ブロックごとに列全幅 `n` を
/// 1 回だけ pack する共有 B バッファの容量（イシュー #1366）。
///
/// `SharedBPcOuter` の [`panel_capacity`] は `blocks.nc` で列を分割した
/// 1 個の nc ブロックぶんの容量を返すが、本 variant は jc（列ブロック）
/// ループを持たず pc ごとに列全幅を 1 回で pack するため、`nc_len_max`
/// の代わりに `n` そのものを使う（[`GemmDriverVariant::IcDynamic`] の
/// ドキュメント「pc ごとの B パネル共有 pack」参照）。乗算は
/// [`task_a_capacity`] と同型の `checked_mul` 連鎖でオーバーフローを
/// 検出し `GemmError::DimProductOverflow` へ変換する（OWASP A03）。
#[cfg(test)]
fn ic_dynamic_b_capacity(n: usize, kc_len_max: usize, nr: usize) -> Result<usize, GemmError> {
    let nr_blocks = n.div_ceil(nr.max(1));
    nr_blocks
        .checked_mul(kc_len_max)
        .and_then(|v| v.checked_mul(nr))
        .ok_or(GemmError::DimProductOverflow)
}

/// [`gemm_blis_shared_b_pc_outer_region`] 専用: `a_panel` にタスク担当
/// 行範囲全体が pc ブロックぶん 1 回で packing 済みであることを前提に、
/// jr→ir のみを回して `c_chunk`（タスク行 0 起点でオフセット済み）へ
/// 書き戻す（[`gemm_blis_ic_loop`] から A packing と `ic`（`blocks.mc`
/// 単位のブロッキング）を除いた版）。
///
/// タイル書き戻しロジック（完全タイル直接 / 端タイル copy-in-copy-out）
/// は [`gemm_blis_ic_loop`] と一字一句同一にすることで bit 完全一致
/// 契約（REQ-2）・FMA 契約を保つ（コード重複は「ic ブロッキングの
/// 有無」というループ構造の違いを吸収するための意図的な選択。共通化
/// すると `ic` 変数の有無で分岐が増え可読性が下がるため、#750 の
/// `IcLoopContext` と同型の「文脈を引数で渡す」設計のみ踏襲する）。
#[cfg(test)]
fn gemm_blis_jr_ir_loop<K: Microkernel>(
    kernel: K,
    c_chunk: &mut [f32],
    task_mc: usize,
    a_panel: &[f32],
    ctx: &IcLoopContext,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let IcLoopContext {
        b_panel,
        n,
        kc_len,
        jc,
        nr_blocks,
        nc_len,
        ..
    } = *ctx;

    let mr_blocks = task_mc.div_ceil(mr);

    for jr_block in 0..nr_blocks {
        let jr = jr_block * nr;
        let nr_eff = nr.min(nc_len - jr);
        let bp_slice = &b_panel[jr_block * kc_len * nr..(jr_block + 1) * kc_len * nr];

        for ir_block in 0..mr_blocks {
            let ir = ir_block * mr;
            let mr_eff = mr.min(task_mc - ir);
            let ap_slice = &a_panel[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr];
            let col_base = jc + jr;

            if mr_eff == mr && nr_eff == nr {
                // 完全タイル直接ロード/ストア（`gemm_blis_ic_loop` の
                // 同分岐と同一根拠。#557）。
                let row0 = ir * n + col_base;
                let c_direct = &mut c_chunk[row0..row0 + (mr - 1) * n + nr];
                kernel.run_with_ldc(ap_slice, bp_slice, c_direct, n, kc_len)?;
            } else {
                // 端タイル: `MAX_TILE` スタックバッファ経由（同上）。
                let mut c_tile_buf = [0.0f32; MAX_TILE];
                let c_tile = &mut c_tile_buf[..mr * nr];
                for i in 0..mr_eff {
                    let src = &c_chunk[(ir + i) * n + col_base..(ir + i) * n + col_base + nr_eff];
                    c_tile[i * nr..i * nr + nr_eff].copy_from_slice(src);
                }
                kernel.run_with_ldc(ap_slice, bp_slice, c_tile, nr, kc_len)?;
                for i in 0..mr_eff {
                    let dst =
                        &mut c_chunk[(ir + i) * n + col_base..(ir + i) * n + col_base + nr_eff];
                    dst.copy_from_slice(&c_tile[i * nr..i * nr + nr_eff]);
                }
            }
        }
    }
    Ok(())
}

/// pc 外側ループ・A 1 回 pack 版の並列 5-loop 本体（イシュー #1041）。
///
/// [`gemm_blis_shared_b_region`]（#750）は jc→pc→ic の順で、B を
/// (jc,pc) ブロックごとに 1 回だけ pack して全タスクへ共有するが、A は
/// 各タスクが (jc,pc) の組ごとに packing し直す（jc の反復回数ぶん同じ
/// 行範囲を重複 pack する）。gemm crate（faer 実体）との直接比較
/// （`docs/perf/oss-gemm-comparison-baseline.md` §7.2・イシュー #1041
/// 診断）では N=1024/2048 で 0.84〜0.91 倍と劣位であり、
/// `docs/cpu-gemm-b-packing-sharing-decision.md` の重複コストモデル
/// により、A packing 重複（メモリ帯域）が中形状ほど演算量 2N³ に対して
/// 相対的に重いと診断された（`docs/perf/cpu-gemm-candle-cpu-retune.md`
/// §2 参照）。
///
/// 本関数は pc→jc の順に入れ替え、各 pc ブロックで各タスクが自分の
/// 担当行範囲を **1 回だけ** pack してから、その pc に属する全 jc
/// ブロックへ使い回す（B は [`gemm_blis_shared_b_region`] と同じく
/// (jc,pc) ごとに 1 回 pack して全タスクへ共有する。A・B とも
/// 「誰がいつ pack するか」だけが変わる）。
///
/// ## bit 完全一致契約（REQ-2）を保つ根拠
///
/// `gemm_naive` との bit 完全一致は「C の各要素が pc（縮約次元の
/// ブロック）昇順に `f32::mul_add` で蓄積される」ことにのみ依存する
/// （本ファイル冒頭ドキュメント参照）。C の各要素は (m,n) 座標で一意に
/// 決まる 1 つの (タスク行範囲, jc ブロック) の組にのみ属し、ある pc
/// 値の中で jc・タスク行範囲の反復順を入れ替えても、その要素が
/// 「どの pc の時に触れられるか」の集合と大小関係は変化しない（pc は
/// 本関数でも [`gemm_blis_shared_b_region`] と同じく外側で昇順に回る
/// ため）。jc・タスク行範囲の入れ替えは互いに素な C 要素集合を担当
/// するだけで、同一要素の蓄積順序には影響しない。
///
/// ## A パネル容量の拡張
///
/// [`panel_capacity`] は「ic ループが `blocks.mc` 単位で A パネルを
/// 使い回す」前提で `mc_len_max = blocks.mc.min(mc_total)` にクランプ
/// した容量を返すが、本関数はタスクの担当行範囲全体（`task_mc` 行）を
/// pc ブロックごとに 1 回で pack し、その pc の全 jc 反復で使い回す
/// ため `blocks.mc` によるクランプを行わず `task_mc` 行ぶんの容量が
/// 要る（[`task_a_capacity`] 参照）。
///
/// **実機ゲート結果: 非採用（#1140／#1141／#1144）**。GB10・Apple M4 Max
/// 双方の実機 5 回中央値実測で本番既定 `RowPanel` を大きく下回り、採用
/// ゲート（`docs/perf/cpu-gemm-candle-cpu-retune.md` §1・§8 の受け入れ
/// 条件）を満たさなかった。本番結線はしない。次候補（B 側 laneq の
/// ベクトル転置化・prefetch・KC 再スイープ）の A/B 計測資産として
/// `#[cfg(test)]` のまま維持する（[`GemmDriverVariant`] 経由で
/// bit 完全一致検証・A/B 一括計測ハーネスから到達する）。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn gemm_blis_shared_b_pc_outer_region<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k_dim: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;
    let a = &a[row_start * k_dim..];

    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let panel_rows = mc_total.div_ceil(num_threads).max(1);
    let num_tasks = mc_total.div_ceil(panel_rows);

    // 共有 B バッファ: `gemm_blis_shared_b_region` と同じ算出方法。
    let (b_cap, _) = panel_capacity(n, k_dim, mc_total, mr, nr, blocks);
    let mut b_panel_buf = vec![0.0f32; b_cap];

    // タスクごとの A バッファ: `blocks.mc` にクランプせず、担当行範囲
    // 全体（最終タスク以外は `panel_rows` 行）を 1 pc ブロックぶん保持
    // できる容量で確保する（上記ドキュメント参照）。
    let kc_len_max = blocks.kc.min(k_dim);
    let a_cap = task_a_capacity(panel_rows, kc_len_max, mr)?;
    let mut a_bufs: Vec<Vec<f32>> = (0..num_tasks).map(|_| vec![0.0f32; a_cap]).collect();

    for pc in (0..k_dim).step_by(blocks.kc) {
        let kc_len = blocks.kc.min(k_dim - pc);

        // A packing をタスク単位で 1 回（この pc ブロックの全 jc 反復で
        // 使い回す）。各タスクの書き込み先は互いに素な `a_bufs[task]`
        // のため `par_iter_mut` でデータ競合なく並列化できる
        // （`unsafe` 不要）。
        a_bufs
            .par_iter_mut()
            .enumerate()
            .for_each(|(task_idx, a_buf)| {
                let task_row_start = task_idx * panel_rows;
                if task_row_start >= mc_total {
                    return;
                }
                let task_mc = panel_rows.min(mc_total - task_row_start);
                let a_task = &a[task_row_start * k_dim..];
                let mr_blocks = task_mc.div_ceil(mr);
                for ir_block in 0..mr_blocks {
                    let ir = ir_block * mr;
                    let mr_eff = mr.min(task_mc - ir);
                    pack_a(
                        &mut a_buf[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr],
                        a_task,
                        APackTile {
                            k_total: k_dim,
                            row_start: ir,
                            mr,
                            mr_eff,
                            kc_start: pc,
                            kc_len,
                        },
                    );
                }
            });

        for jc in (0..n).step_by(blocks.nc) {
            let nc_len = blocks.nc.min(n - jc);
            let nr_blocks = nc_len.div_ceil(nr);
            let b_panel = &mut b_panel_buf[..nr_blocks * kc_len * nr];

            // B packing の nr ブロック単位並列化（`gemm_blis_shared_b_region`
            // と同じ理由・同じ実装。#750）。
            b_panel
                .par_chunks_mut(kc_len * nr)
                .enumerate()
                .for_each(|(jr_block, dst)| {
                    let jr = jr_block * nr;
                    let nr_eff = nr.min(nc_len - jr);
                    pack_b(
                        dst,
                        b,
                        BPackTile {
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start: jc + jr,
                            nr,
                            nr_eff,
                        },
                    );
                });

            let b_panel_ref: &[f32] = b_panel;
            let ctx = IcLoopContext {
                b_panel: b_panel_ref,
                n,
                k_dim,
                pc,
                kc_len,
                jc,
                nr_blocks,
                nc_len,
                blocks,
            };

            // ic 相当（タスク行範囲）の並列化。A は既にこの pc ブロック
            // ぶん packing 済み（`a_bufs[task_idx]`）のため、ここでは
            // 読み取り専用で参照するだけで pack をやり直さない
            // （[`gemm_blis_jr_ir_loop`] 参照。データ競合はコンパイル時の
            // 借用分割で排除。`unsafe` 不要）。
            c.par_chunks_mut(panel_rows * n)
                .enumerate()
                .zip(a_bufs.par_iter())
                .try_for_each(|((task_idx, c_chunk), a_buf)| {
                    let task_mc = c_chunk.len() / n;
                    if task_mc == 0 {
                        return Ok(());
                    }
                    let _ = task_idx;
                    gemm_blis_jr_ir_loop(kernel, c_chunk, task_mc, a_buf, &ctx)
                })?;
        }
    }
    Ok(())
}

/// [`dispatch_shared_b`] の pc 外側版（イシュー #1041）。ロジックは
/// `dispatch_shared_b` と同一で、呼ぶ先
/// （[`gemm_blis_shared_b_pc_outer_region`]）のみが異なる。本番未結線
/// のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "x86_64")]
fn dispatch_shared_b_pc_outer(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    #[cfg(avx512_stable)]
    if let Some(kernel) = microkernel::Avx512Kernel::try_new() {
        return gemm_blis_shared_b_pc_outer_region(kernel, a, b, c, n, k, rows, blocks);
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        gemm_blis_shared_b_pc_outer_region(kernel, a, b, c, n, k, rows, blocks)
    } else {
        gemm_blis_shared_b_pc_outer_region(ScalarKernel, a, b, c, n, k, rows, blocks)
    }
}

/// aarch64 版 [`dispatch_shared_b_pc_outer`]（#1041）。[`dispatch_shared_b`]
/// の aarch64 版と同じ理由で NEON 固定。本番未結線のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
fn dispatch_shared_b_pc_outer(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    gemm_blis_shared_b_pc_outer_region(microkernel::NeonKernel, a, b, c, n, k, rows, blocks)
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_shared_b_pc_outer`]（#1041）。
/// [`dispatch_shared_b`] の同 arch 版と同じ理由で [`ScalarKernel`] 固定。
/// 本番未結線のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dispatch_shared_b_pc_outer(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    gemm_blis_shared_b_pc_outer_region(ScalarKernel, a, b, c, n, k, rows, blocks)
}

/// pc（K ブロック）最外・ic（行パネル）を `AtomicUsize` カウンタで動的
/// 配布する並列 5-loop 本体（イシュー #1366）。
///
/// [`gemm_blis_shared_b_pc_outer_region`]（#1041）は行パネルを
/// `mc_total.div_ceil(num_workers)` で **静的に** 1 タスク 1 パネルへ
/// 固定するため、MC タイル数がワーカー数で割り切れない形状や異種コア
/// 環境では負荷不均衡が生じうる（issue #1366 実装計画 §1）。本関数は
/// 行パネルの配布方式を静的から `AtomicUsize` の `fetch_add` カウンタに
/// よる動的配布へ変更するだけでなく、**B の列ブロッキングも変更して
/// いる**: [`gemm_blis_shared_b_pc_outer_region`] は B を (pc,jc) ごと
/// に `blocks.nc` 幅で pack し jc ループで列を順に処理するのに対し、
/// 本関数は jc ループを持たず `blocks.nc` を使わずに pc ごとへ列全幅
/// `n` を 1 回で pack する（[`ic_dynamic_b_capacity`] 参照）。この違い
/// により B バッファのメモリ使用量（`nc` 幅 → 列全幅 `n`。形状によって
/// は増大する）・キャッシュ局所性（列全幅を一度に触れるか `nc` 単位で
/// 分割して触れるか）・pc ごとの同期点の数（jc ブロック数ぶん → 1 回）
/// が変わる。行パネルを `AtomicUsize` の `fetch_add` カウンタで動的
/// 配布し、先に終わった worker が次のパネルを取れるようにする点は
/// 変更していない。
///
/// ## 動的配布の機構（`unsafe` を追加しない設計）
///
/// C を [`ic_dynamic_panel_rows`] 行ずつの `panel_rows` パネルへ分割し、
/// 各パネルの `&mut [f32]` を `Mutex<Option<&mut [f32]>>` スロットへ
/// 1 個ずつ格納する。worker は `AtomicUsize::fetch_add(1, Relaxed)` で
/// 自分が担当するパネル index を確定してから、対応するスロットを
/// `lock().take()` して `&mut` を取り出す。index はパネル数を上限に
/// 単調増加し、かつ各 index は高々 1 worker しか claim しないため、
/// 同じスロットへ 2 つの worker が同時にロックを取り合うことは構築上
/// 発生しない（1 パネル 1 回のロックで常に無競合）。`&mut` パネル間の
/// 排他性はコンパイル時の借用検査（`Mutex<Option<&mut [f32]>>` に格納
/// した時点で各要素の借用が互いに素であることが保証される）で担保
/// されており、`unsafe` は不要（issue #1366 のスコープ「`unsafe` を
/// 新規導入しない」）。取り出し後は poison 有無に関わらず即座に
/// `drop(guard)` してロックを解放する（他 worker のブロッキングを
/// 最小化する。各 index は高々 1 回しか claim されないため以後この
/// スロットへのアクセスは発生しない）。
///
/// ## bit 完全一致契約（REQ-2）を保つ根拠
///
/// [`gemm_blis_shared_b_pc_outer_region`] ドキュメントと同じ論法が
/// そのまま成り立つ: pc は本関数でも外側で昇順に回り、C の各要素は
/// (pc, 行パネル) の組で見て一意な 1 つの行パネルにのみ属する。
/// 行パネルをどの worker が・どの順序で claim するかは、互いに素な
/// C 要素集合の処理順序を並び替えるだけで、同一要素の pc 昇順・
/// カーネル内 p 昇順の蓄積順序には影響しない。
///
/// カウンタが `num_panels` を超えて進んだ worker は対応するスロットが
/// 存在しないため直ちにループを終える（`idx >= num_panels` 判定）。
/// 構築上 `slots[idx]` が既に `None`（他 worker が同じ idx を claim
/// 済み）になることはないが、`unreachable!` によるパニック変換を避け
/// `continue` で次の index へ進む fail-safe を入れている
/// （`.claude/rules/coding-rust.md` の panic 禁止方針）。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn gemm_blis_ic_dynamic_region<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k_dim: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;
    let a = &a[row_start * k_dim..];

    let num_workers = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let panel_rows = ic_dynamic_panel_rows(mc_total, blocks.mc, mr, num_workers);
    let num_panels = mc_total.div_ceil(panel_rows);

    // 共有 B バッファ: pc ごとに列全幅 n を 1 回だけ pack する
    // （`ic_dynamic_b_capacity` 参照。jc ループを持たないため
    // `blocks.nc` は使わない）。
    let kc_len_max = blocks.kc.min(k_dim);
    let b_cap = ic_dynamic_b_capacity(n, kc_len_max, nr)?;
    let mut b_panel_buf = vec![0.0f32; b_cap];

    // worker ごとの A バッファ: pc ループ外で 1 回確保し、pc ごとに
    // claim したパネルぶんを 1 回だけ pack して使い回す
    // （`gemm_blis_shared_b_pc_outer_region` の A バッファ確保方針と
    // 同型。`panel_rows` は動的配布パネルの最大行数のため、これを
    // 容量算出に使えば claim したどのパネルも収まる）。
    let a_cap = task_a_capacity(panel_rows, kc_len_max, mr)?;
    let mut a_bufs: Vec<Vec<f32>> = (0..num_workers).map(|_| vec![0.0f32; a_cap]).collect();

    for pc in (0..k_dim).step_by(blocks.kc) {
        let kc_len = blocks.kc.min(k_dim - pc);
        let nr_blocks = n.div_ceil(nr);
        let b_panel = &mut b_panel_buf[..nr_blocks * kc_len * nr];

        // B packing の nr ブロック単位並列化
        // （`gemm_blis_shared_b_pc_outer_region` と同じ理由・実装。
        // jc を固定 0・列全幅 n の 1 ブロックとして扱うため
        // `col_start` はそのまま `jr`）。
        b_panel
            .par_chunks_mut(kc_len * nr)
            .enumerate()
            .for_each(|(jr_block, dst)| {
                let jr = jr_block * nr;
                let nr_eff = nr.min(n - jr);
                pack_b(
                    dst,
                    b,
                    BPackTile {
                        n_total: n,
                        kc_start: pc,
                        kc_len,
                        col_start: jr,
                        nr,
                        nr_eff,
                    },
                );
            });

        let b_panel_ref: &[f32] = b_panel;
        let ctx = IcLoopContext {
            b_panel: b_panel_ref,
            n,
            k_dim,
            pc,
            kc_len,
            jc: 0,
            nr_blocks,
            nc_len: n,
            blocks,
        };

        // 行パネルを Mutex スロットへ 1 個ずつ格納し、AtomicUsize
        // カウンタで動的配布する（上記ドキュメント参照）。`slots` の
        // 借用は本 pc 反復のスコープ内で完結する（次の pc 反復で
        // `c.chunks_mut` を再度呼ぶ前に `slots` がドロップされる）。
        let slots: Vec<Mutex<Option<&mut [f32]>>> = c
            .chunks_mut(panel_rows * n)
            .map(|panel| Mutex::new(Some(panel)))
            .collect();
        let counter = AtomicUsize::new(0);

        a_bufs.par_iter_mut().try_for_each(|a_buf| {
            loop {
                let idx = counter.fetch_add(1, Ordering::Relaxed);
                if idx >= num_panels {
                    return Ok::<(), GemmError>(());
                }
                let mut guard = slots[idx].lock().unwrap_or_else(PoisonError::into_inner);
                let Some(c_panel) = guard.take() else {
                    // 構築上到達しない（各 index は高々 1 worker しか
                    // claim しないため既に `None` になることはない）が、
                    // panic せず次の index へ進む fail-safe。
                    continue;
                };
                drop(guard);

                let task_mc = c_panel.len() / n;
                if task_mc == 0 {
                    continue;
                }
                let row_start_local = idx * panel_rows;
                let a_task = &a[row_start_local * k_dim..];
                let mr_blocks = task_mc.div_ceil(mr);
                for ir_block in 0..mr_blocks {
                    let ir = ir_block * mr;
                    let mr_eff = mr.min(task_mc - ir);
                    pack_a(
                        &mut a_buf[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr],
                        a_task,
                        APackTile {
                            k_total: k_dim,
                            row_start: ir,
                            mr,
                            mr_eff,
                            kc_start: pc,
                            kc_len,
                        },
                    );
                }

                gemm_blis_jr_ir_loop(kernel, c_panel, task_mc, a_buf, &ctx)?;
            }
        })?;
    }
    Ok(())
}

/// [`dispatch_shared_b_pc_outer`] の動的配布版（イシュー #1366）。
/// ロジックは `dispatch_shared_b_pc_outer` と同一で、呼ぶ先
/// （[`gemm_blis_ic_dynamic_region`]）のみが異なる。本番未結線のため
/// `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "x86_64")]
fn dispatch_ic_dynamic(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    #[cfg(avx512_stable)]
    if let Some(kernel) = microkernel::Avx512Kernel::try_new() {
        return gemm_blis_ic_dynamic_region(kernel, a, b, c, n, k, rows, blocks);
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        gemm_blis_ic_dynamic_region(kernel, a, b, c, n, k, rows, blocks)
    } else {
        gemm_blis_ic_dynamic_region(ScalarKernel, a, b, c, n, k, rows, blocks)
    }
}

/// aarch64 版 [`dispatch_ic_dynamic`]（#1366）。[`dispatch_shared_b_pc_outer`]
/// の aarch64 版と同じ理由で NEON 固定。本番未結線のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
fn dispatch_ic_dynamic(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    gemm_blis_ic_dynamic_region(microkernel::NeonKernel, a, b, c, n, k, rows, blocks)
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_ic_dynamic`]（#1366）。
/// [`dispatch_shared_b_pc_outer`] の同 arch 版と同じ理由で
/// [`ScalarKernel`] 固定。本番未結線のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dispatch_ic_dynamic(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    gemm_blis_ic_dynamic_region(ScalarKernel, a, b, c, n, k, rows, blocks)
}

/// [`gemm_blis_two_d_dynamic_region`] 専用: (行帯, 列帯) job 1 個が担当する
/// C の行セグメント集合（イシュー #1311・設計 §4.2 主案 S′〈job-local C
/// staging〉・§6）。`c_rows[i]` は `rows.start + i` 行目のうち `cols`
/// 範囲に対応する `&'c mut [f32]`（長さ `cols.end - cols.start`）で、
/// [`split_c_into_jobs`] が `c.chunks_mut(n)`（行分割）→ 列帯境界での
/// `split_at_mut` 連鎖により job 間で重複しないことをコンパイル時借用
/// 検査で保証して構築する（`unsafe` 非導入。設計 §4.2）。
///
/// 本番結線（#1313）により `#[cfg(test)]` を外した（旧 `#[cfg(test)]` は
/// #1311 導入時点の本番未結線を反映していた）。
struct TwoDJob<'c> {
    rows: Range<usize>,
    cols: Range<usize>,
    c_rows: Vec<&'c mut [f32]>,
}

/// [`gemm_blis_two_d_dynamic_region`] 専用: C を `grid`（[`partition::JobGrid`]）
/// の (行帯 × 列帯) へ分割し、job ごとに独立な `&mut [f32]` 集合を返す
/// （イシュー #1311・設計 §4.2）。
///
/// 手順: (1) `c.chunks_mut(n)` で行スライスへ分割（各行は他行と disjoint。
/// コンパイル時借用検査で保証される）。(2) 各行を列帯境界（`partition::bands(n,
/// grid.nc_job)`）で `split_at_mut` 連鎖し、列帯ごとの行セグメントへ分割
/// する。(3) 行帯（`partition::bands(m, grid.mc_job)`）ごとに行を束ねて
/// job を組み立てる。
///
/// job 配列は **column-band-major**（同一列帯の行帯を連続）に並べる
/// （設計 §6。性能上の推奨であり、job 分割の正しさ〈§3〉には影響しない。
/// 隣接 job を同時処理する worker 群が同じ B 列を共有キャッシュ上で
/// 参照しやすくするための順序）。
///
/// 本番結線（#1313）により `#[cfg(test)]` を外した。
fn split_c_into_jobs<'c>(
    c: &'c mut [f32],
    n: usize,
    grid: &partition::JobGrid,
) -> Vec<TwoDJob<'c>> {
    if n == 0 || grid.row_bands == 0 || grid.col_bands == 0 {
        return Vec::new();
    }
    let m = c.len() / n;
    let row_bands = partition::bands(m, grid.mc_job);
    let col_bands = partition::bands(n, grid.nc_job);
    if row_bands.is_empty() || col_bands.is_empty() {
        return Vec::new();
    }

    // 行ごとに列帯境界で split_at_mut 連鎖し、列帯ごとの行セグメント列
    // （`per_col_band[col_band_idx]` は絶対行順に並ぶ `Vec<&'c mut [f32]>`）
    // へ集約する。
    let mut per_col_band: Vec<Vec<&'c mut [f32]>> = (0..col_bands.len())
        .map(|_| Vec::with_capacity(m))
        .collect();
    for row in c.chunks_mut(n) {
        let mut remaining: &mut [f32] = row;
        for (col_band_idx, band) in col_bands.iter().enumerate() {
            let len = band.end - band.start;
            let (head, tail) = remaining.split_at_mut(len);
            per_col_band[col_band_idx].push(head);
            remaining = tail;
        }
    }

    // column-band-major で job を組み立てる（設計 §6）。行帯は
    // `per_col_band[col_band_idx]` の先頭から連続して並んでいる
    // （`c.chunks_mut(n)` が絶対行順を保つため）ことを利用し、
    // 行帯の長さぶんずつ前から `drain` して切り出す。
    let mut jobs = Vec::with_capacity(row_bands.len() * col_bands.len());
    for (col_band_idx, col_band) in col_bands.iter().enumerate() {
        let rows_for_band = &mut per_col_band[col_band_idx];
        for row_band in &row_bands {
            let band_len = row_band.end - row_band.start;
            let c_rows: Vec<&'c mut [f32]> = rows_for_band.drain(0..band_len).collect();
            jobs.push(TwoDJob {
                rows: row_band.clone(),
                cols: col_band.clone(),
                c_rows,
            });
        }
    }
    jobs
}

/// [`gemm_blis_two_d_dynamic_region`] 専用: 1 job（(行帯, 列帯) タイル）を
/// 単一 worker が K 全域を通して処理する本体（イシュー #1311・設計 §2.2・
/// §4.2 主案 S′）。
///
/// **job-local C staging（S′）**: job は自分の `mc_len × nc_len_job` の
/// C 部分ブロックを job-local な連続バッファ `c_local`（行ストライド
/// `ldc = nc_len_job`）へ 1 回だけ copy-in し、K 全域（jc→pc→ic→jr→ir）
/// を**既存・無改変の [`gemm_blis_ic_loop`]** で処理してから 1 回だけ
/// copy-out する。[`gemm_blis_ic_loop`] は `IcLoopContext.n` を**C の
/// 行ストライドとしてのみ**使うため（B の幅としては使わない設計。同構造体
/// ドキュメント参照）、`ctx.n = nc_len_job`・`ctx.jc` は job 内相対位置
/// として素通しでき、カーネル本体・FMA 契約（REQ-2）を一切変更せずに
/// job 単位の並列実行を実現できる（設計 §4.2「bit 完全一致（設計 §3
/// 条件 1〜8）の充足」）。
///
/// B は呼び出し元から共有せず job ごとに個別 pack する（設計 §2.2
/// 「packing: job 内 private」）。`region_row_offset` は呼び出し元
/// （[`gemm_blis_two_d_dynamic_region`]）が受け取った `rows: Range<usize>`
/// の `start`（通常 0）で、`Tn`（転置格納 A）の絶対行位置解決にのみ使う
/// （`gemm_blis_region` の `row_offset`／`m_total` 引き回しと同型。#1213）。
///
/// 本番結線（#1313）により `#[cfg(test)]` を外した。
#[allow(clippy::too_many_arguments)]
fn run_two_d_job<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    job: &mut TwoDJob,
    n: usize,
    k_dim: usize,
    blocks: BlockSizes,
    transpose: GemmTranspose,
    region_row_offset: usize,
) -> Result<(), GemmError> {
    let nr = K::NR;
    let mc_len = job.rows.end - job.rows.start;
    let nc_len_job = job.cols.end - job.cols.start;
    if mc_len == 0 || nc_len_job == 0 {
        return Ok(());
    }

    let c_local_len = mc_len
        .checked_mul(nc_len_job)
        .ok_or(GemmError::DimProductOverflow)?;
    let mut c_local = vec![0.0f32; c_local_len];
    for (i, row) in job.c_rows.iter().enumerate() {
        c_local[i * nc_len_job..(i + 1) * nc_len_job].copy_from_slice(row);
    }

    // `gemm_blis_region`（`mod.rs:1064` 付近）と同じ Nn/Nt/Tn 分岐
    // （#1213）。`Tn` は絶対行位置解決のため `region_row_offset +
    // job.rows.start` を渡す（job.rows は region 内の相対範囲のため）。
    let (a_for_loop, row_offset, m_total): (&[f32], usize, usize) = match transpose {
        GemmTranspose::Tn => (
            a,
            region_row_offset + job.rows.start,
            a.len() / k_dim.max(1),
        ),
        GemmTranspose::Nn | GemmTranspose::Nt => (&a[job.rows.start * k_dim..], 0, 0),
    };

    let mut bufs = PanelBuffers::new::<K>(nc_len_job, k_dim, mc_len, blocks);

    for jc in (0..nc_len_job).step_by(blocks.nc) {
        let nc_len = blocks.nc.min(nc_len_job - jc);
        for pc in (0..k_dim).step_by(blocks.kc) {
            let kc_len = blocks.kc.min(k_dim - pc);
            let nr_blocks = nc_len.div_ceil(nr);
            let b_panel = &mut bufs.b_panel[..nr_blocks * kc_len * nr];
            for jr_block in 0..nr_blocks {
                let jr = jr_block * nr;
                let nr_eff = nr.min(nc_len - jr);
                let dst = &mut b_panel[jr_block * kc_len * nr..(jr_block + 1) * kc_len * nr];
                // B は絶対列位置（`job.cols.start + jc + jr`）で pack する
                // （job 内の相対 jc に対し、実際の B 参照は元の n 幅の
                // 行列へアクセスするため）。
                let col_start = job.cols.start + jc + jr;
                match transpose {
                    GemmTranspose::Nt => pack_b_from_transposed(
                        dst,
                        b,
                        BTPackTile {
                            k_total: k_dim,
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start,
                            nr,
                            nr_eff,
                        },
                    ),
                    GemmTranspose::Nn | GemmTranspose::Tn => pack_b(
                        dst,
                        b,
                        BPackTile {
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start,
                            nr,
                            nr_eff,
                        },
                    ),
                }
            }

            // `ctx.n = nc_len_job`（job 幅。C 行ストライドとしてのみ
            // 使われる。`IcLoopContext` ドキュメント参照）・`ctx.jc` は
            // job 内相対位置。`gemm_blis_ic_loop` は無改変で再利用する。
            let ctx = IcLoopContext {
                b_panel,
                n: nc_len_job,
                k_dim,
                pc,
                kc_len,
                jc,
                nr_blocks,
                nc_len,
                blocks,
            };
            gemm_blis_ic_loop(
                kernel,
                a_for_loop,
                &mut c_local,
                mc_len,
                &mut bufs.a_panel,
                &ctx,
                transpose,
                row_offset,
                m_total,
            )?;
        }
    }

    for (i, row) in job.c_rows.iter_mut().enumerate() {
        row.copy_from_slice(&c_local[i * nc_len_job..(i + 1) * nc_len_job]);
    }

    Ok(())
}

/// (mc, nc) 2D 動的分配の 5-loop 本体（イシュー #1311・設計 §2.2・§6）。
///
/// `partition::job_grid` で worker 数より多い job 数（`jobs_per_worker *
/// num_threads` 目標）を算出し、[`split_c_into_jobs`] で C を job ごとの
/// 独立な `&mut` 集合へ分割、rayon の適応分割（work stealing）で分配する
/// （`jobs.into_par_iter().with_min_len(1)`。カスタム同期・`Mutex` は
/// 使わない。設計 §6「主案」）。各 job は自分の C 部分ブロックに対し
/// **K 全域を単一 worker が同期なしで処理**する（pc ごとのバリアなし・
/// split-K 禁止。`IcDynamic`〈#1366〉の「pc ごとの同期点＋列全幅 B pack」
/// という構造〈GB10 での後退の推定要因。#1367〉を繰り返さない設計）。
///
/// **本番結線（イシュー #1313）**: 両実機 A/B・採否判定は #1312（DGX 実測。
/// 判定 undetermined）・#1313 Phase 0（Apple M4 Max 専有ゲート通過後の
/// 再計測。jpw=2・jpw=4 とも Tier 1 条件充足）を経て、
/// [`TWO_D_DYNAMIC_PRODUCTION_ENABLED`] 単一 const ゲート経由で本関数・
/// [`dispatch_two_d_dynamic`] が本番公開入口
/// （[`gemm_blis_parallel_with_transpose`]／[`gemm_blis_bias_act_parallel`]）
/// から呼ばれるようになった。実測記録は
/// `docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`・
/// `docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/` を参照。
#[allow(clippy::too_many_arguments)]
fn gemm_blis_two_d_dynamic_region<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k_dim: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
    transpose: GemmTranspose,
    jobs_per_worker: usize,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;
    if mc_total == 0 || n == 0 {
        return Ok(());
    }

    // `Tn` は a をフルスライスのまま渡し、絶対行位置は
    // `run_two_d_job` 側で `region_row_offset + job.rows.start` として
    // 解決する（`gemm_blis_region` と同じ分岐。#1213）。
    let a_for_region: &[f32] = match transpose {
        GemmTranspose::Tn => a,
        GemmTranspose::Nn | GemmTranspose::Nt => &a[row_start * k_dim..],
    };
    let c_region = &mut c[row_start * n..(row_start + mc_total) * n];

    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let grid = partition::job_grid(mc_total, n, mr, nr, &blocks, num_threads, jobs_per_worker)?;
    let mut jobs = split_c_into_jobs(c_region, n, &grid);

    jobs.par_iter_mut().try_for_each(|job| {
        run_two_d_job(
            kernel,
            a_for_region,
            b,
            job,
            n,
            k_dim,
            blocks,
            transpose,
            row_start,
        )
    })
}

/// [`dispatch_ic_dynamic`] の 2D 動的分配版（イシュー #1311）。
/// x86_64: AVX-512（stable cfg 時）→ AVX2 → スカラーの優先順位で ISA
/// トークンを 1 回だけ確定する（既存 dispatch 系と同一方針）。本番結線
/// 済み（#1313）。
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn dispatch_two_d_dynamic(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
    transpose: GemmTranspose,
    jobs_per_worker: usize,
) -> Result<(), GemmError> {
    #[cfg(avx512_stable)]
    if let Some(kernel) = microkernel::Avx512Kernel::try_new() {
        return gemm_blis_two_d_dynamic_region(
            kernel,
            a,
            b,
            c,
            n,
            k,
            rows,
            blocks,
            transpose,
            jobs_per_worker,
        );
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        gemm_blis_two_d_dynamic_region(
            kernel,
            a,
            b,
            c,
            n,
            k,
            rows,
            blocks,
            transpose,
            jobs_per_worker,
        )
    } else {
        gemm_blis_two_d_dynamic_region(
            ScalarKernel,
            a,
            b,
            c,
            n,
            k,
            rows,
            blocks,
            transpose,
            jobs_per_worker,
        )
    }
}

/// aarch64 版 [`dispatch_two_d_dynamic`]（#1311）。既定は NEON 固定
/// （他 dispatch 系〈[`dispatch_ic_dynamic`] 等〉と同じ理由。本番結線済み
/// （#1313）だが、[`SME_PRODUCTION_ENABLED`] が `true` かつ形状が
/// [`sme_shape_eligible`] を満たし、かつ実行 CPU が SME・非拡張 FP32
/// 外積に対応する（[`microkernel::SmeKernel::try_new`]）場合のみ SME
/// マイクロカーネルへ切り替える（イシュー #1587）。NEON は aarch64 の
/// baseline ISA のため実行時検出不要（本関数 doc 冒頭参照）だが、SME は
/// `Avx2Kernel`／`Avx512Kernel` と同型の「検出済みトークンのみ構築可能」
/// パターンで安全性を担保する。
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn dispatch_two_d_dynamic(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
    transpose: GemmTranspose,
    jobs_per_worker: usize,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    if SME_PRODUCTION_ENABLED
        && sme_shape_eligible(rows.end - rows.start, n, k)
        && let Some(kernel) = microkernel::SmeKernel::try_new()
    {
        return gemm_blis_two_d_dynamic_region(
            kernel,
            a,
            b,
            c,
            n,
            k,
            rows,
            blocks,
            transpose,
            jobs_per_worker,
        );
    }
    gemm_blis_two_d_dynamic_region(
        microkernel::NeonKernel,
        a,
        b,
        c,
        n,
        k,
        rows,
        blocks,
        transpose,
        jobs_per_worker,
    )
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_two_d_dynamic`]（#1311）。
/// [`ScalarKernel`] 固定。本番結線済み（#1313）。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[allow(clippy::too_many_arguments)]
fn dispatch_two_d_dynamic(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
    transpose: GemmTranspose,
    jobs_per_worker: usize,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    gemm_blis_two_d_dynamic_region(
        ScalarKernel,
        a,
        b,
        c,
        n,
        k,
        rows,
        blocks,
        transpose,
        jobs_per_worker,
    )
}

/// [`GemmDriverVariant::TwoDDynamic`] の既定 `jobs_per_worker`（設計
/// §5.2「`jobs_per_worker` は const（既定 2。§5.3 の表で `RowPanel` の
/// pack 総量を全形状で下回る側）」）。#1312 が `{2, 4}`（必要なら 8）を
/// [`gemm_blis_parallel_two_d_dynamic_with_params`] でスイープし、#1313
/// Phase 0（Apple M4 Max 専有ゲート通過後の再計測）でも jpw=2・jpw=4 とも
/// Tier 1 条件を満たしたため、既定値 2（#1311 導入時点の値）を変更せず
/// 本番結線した（`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`
/// 「#1313 追記」節）。本番経路（[`gemm_blis_parallel_with_transpose`]／
/// [`gemm_blis_bias_act_parallel`]）もこの値を使う。
const TWO_D_JOBS_PER_WORKER: usize = 2;

/// [`GemmDriverVariant::TwoDDynamic`]（(mc, nc) 2D job 動的分配）を本番
/// 公開入口（[`gemm_blis_parallel_with_transpose`]／
/// [`gemm_blis_bias_act_parallel`]）へ結線するかどうかの単一 const ゲート
/// （イシュー #1313）。
///
/// `thread_limit::BIG_CORE_LIMIT_ENABLED`（#1363/#1364）と同型の設計:
/// ADOPT 時は `true` のまま維持し、後日の実測（例: framework-compare
/// 実践規模での後退発見）で REJECT が確定した場合は本 const のみ
/// `false` へ差し戻す（コード・テストは削除せず保持する）。
///
/// 採否根拠: #1312（DGX Spark GB10 実機実測。専有ゲート通過・
/// `jobs_per_worker ∈ {2, 4}` とも全形状で `RowPanel` を 1.09〜1.80 倍
/// 上回る）・#1313 Phase 0（Apple M4 Max 専有ゲート通過後の再計測。
/// jpw=2: 1024/2048/4096 で 1.23／1.31／1.03 倍、jpw=4: 1.23／1.31／
/// 1.09 倍、いずれも Tier 1 条件（N=1024/2048 で比 1.00 以上・N=4096 で
/// 0.95 以上・勝ち run 3/5 以上）を満たす）・#1313 framework-compare gemm cpu
/// before/after（両実機・reuse 非後退）。詳細は
/// `docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`・
/// `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20 を参照。
const TWO_D_DYNAMIC_PRODUCTION_ENABLED: bool = true;

/// aarch64 SME（`fmopa`。イシュー #1587）マイクロカーネルを
/// [`dispatch_two_d_dynamic`] の aarch64 版へ形状条件付きで結線するか
/// どうかの単一 const ゲート（[`TWO_D_DYNAMIC_PRODUCTION_ENABLED`]・
/// `thread_limit::BIG_CORE_LIMIT_ENABLED` と同型のロールバック機構）。
///
/// `false` の間は実行 CPU が SME に対応していても常に
/// [`microkernel::NeonKernel`] が選ばれる（#1313 以前と bit 完全一致）。
/// 実測（R1〜R4。issue #1587 コメントの事前登録規則）が完了し ADOPT が
/// 確定するまでは `false` を維持する。採否記録は
/// `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` を参照。
#[cfg(target_arch = "aarch64")]
const SME_PRODUCTION_ENABLED: bool = false;

/// [`SME_PRODUCTION_ENABLED`] が `true` のときに SME 経路を候補とする
/// 形状しきい値（イシュー #1587 事前登録規則 R4）。`sme_shape_eligible`
/// の純関数として実装し単体テスト可能にする。しきい値の値自体は
/// マイクロ A/B（R4）で確定するまでの仮値であり、`docs/perf/
/// cpu-gemm-sme-fmopa-microkernel.md` の実測完了後に確定値へ更新する。
#[cfg(target_arch = "aarch64")]
const SME_MIN_M: usize = 256;
#[cfg(target_arch = "aarch64")]
const SME_MIN_N: usize = 256;
#[cfg(target_arch = "aarch64")]
const SME_MIN_K: usize = 64;

/// `(m_total, n, k)` が SME 経路の形状しきい値（[`SME_MIN_M`]／
/// [`SME_MIN_N`]／[`SME_MIN_K`]）を満たすかどうかを判定する純関数
/// （イシュー #1587）。`m_total` は呼び出し元の region 全体の行数
/// （job 分割前）を渡す契約（`dispatch_two_d_dynamic` の `rows.end -
/// rows.start` 相当）。
#[cfg(target_arch = "aarch64")]
fn sme_shape_eligible(m_total: usize, n: usize, k: usize) -> bool {
    m_total >= SME_MIN_M && n >= SME_MIN_N && k >= SME_MIN_K
}

/// テスト・A/B 計測専用: [`GemmDriverVariant::TwoDDynamic`] の
/// `jobs_per_worker`／`transpose` を注入できるパラメータ化入口
/// （イシュー #1311・設計 §5.2「`#[cfg(test)]` のパラメータ化入口」）。
/// `validate_dims`・`validate_block_sizes`・`n == 0`／`m == 1`（gemv 専用
/// 経路。`gemm_blis_parallel_with_transpose` の分岐と同一）を経てから
/// [`dispatch_two_d_dynamic`] を呼ぶ。`jobs_per_worker == 0` は `1` へ
/// クランプする（`partition::job_grid` の `num_threads == 1` 特例と同じ
/// fail-closed 方針。0 を渡すと目標 job 数が 0 になり空の job 集合に
/// なってしまうのを防ぐ）。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_blis_parallel_two_d_dynamic_with_params(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
    jobs_per_worker: usize,
    transpose: GemmTranspose,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    if n == 0 {
        return Ok(());
    }
    if m == 1 {
        match transpose {
            GemmTranspose::Nn | GemmTranspose::Tn => gemm_row_vector(a, b, c, n),
            GemmTranspose::Nt => gemm_row_vector_nt(a, b, c, k),
        }
        return Ok(());
    }

    dispatch_two_d_dynamic(
        a,
        b,
        c,
        n,
        k,
        0..m,
        blocks,
        transpose,
        jobs_per_worker.max(1),
    )
}

/// A/B 一括計測ハーネス（イシュー #1041）向け: 並列 5-loop ドライバの
/// 候補を 1 つの入口で選べるようにする列挙。`#[cfg(test)]` 限定。
/// GB10（#1140）・Apple M4 Max（#1141）双方の実機実測の結果、`SharedB`・
/// `SharedBPcOuter` は本番既定 `RowPanel` を上回れず非採用が確定した
/// （#1144。[`gemm_blis_shared_b_pc_outer_region`] ドキュメント・
/// `docs/perf/cpu-gemm-candle-cpu-retune.md` §8 参照）。次候補の A/B
/// 計測資産として本列挙・ハーネスは維持する（本番未結線のまま）。
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmDriverVariant {
    /// 本番公開入口（[`gemm_blis_parallel`]）と分岐を完全一致させた
    /// 行パネル分割（タスク数に関わらず常に `dispatch_region`。
    /// [`gemm_blis_parallel_with_blocks`] は実タスク数 2 以上で
    /// `dispatch_shared_b` へ分岐するため使わない。
    /// [`gemm_blis_parallel_row_panel_with_blocks`] 参照）。
    RowPanel,
    /// B パネル共有・A は (jc,pc) ごとに再 pack（#750。[`dispatch_shared_b`]
    /// を実タスク数に関わらず強制する版）。
    SharedB,
    /// B パネル共有・A はタスクごとに pc ブロックあたり 1 回だけ pack
    /// （本 PR・イシュー #1041・[`dispatch_shared_b_pc_outer`]）。
    SharedBPcOuter,
    /// B パネル共有（pc ごとに列全幅を 1 回 pack）・行パネルを
    /// `AtomicUsize` カウンタで動的配布（イシュー #1366・
    /// [`dispatch_ic_dynamic`]）。`SharedBPcOuter` の静的等分割による
    /// 負荷不均衡（issue #1366 実装計画 §1）への対処候補。両実機実測
    /// （#1367）の結果 REJECT（不採用）確定。DGX Spark GB10 で N=1024/2048
    /// の対 `RowPanel` 比が 0.63／0.85 と大きく後退し採用ゲートを満たさない
    /// （`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §14・
    /// `docs/perf/cpu-gemm-ic-dynamic-variant.md` §6）。本番未結線のまま
    /// `#[cfg(test)]` 限定を維持する。
    IcDynamic,
    /// B 側 laneq ベクトル転置版マイクロカーネル（[`microkernel::NeonBLaneqVecKernel`]。
    /// イシュー #1317）を [`RowPanel`](Self::RowPanel) と同一の行パネル
    /// 分割・並列化ロジックへ差し込んだ候補（[`gemm_blis_parallel_row_panel_with_kernel`]
    /// 経由）。差分をマイクロカーネル自体（C タイル転置のベクトル化）に
    /// 限定した A/B 計測基準線であり、`RowPanel` との bit 完全一致
    /// （有限値入力）が理論契約として成り立つ（`neon` モジュール冒頭
    /// #1317 節）。採否・実機実測は #1318 が引き継ぐ。aarch64 限定
    /// （`NeonBLaneqVecKernel` 自体が aarch64 限定トークンのため）。
    #[cfg(target_arch = "aarch64")]
    RowPanelBLaneqVec,
    /// (mc, nc) 2D タイル job の動的分配（rayon work stealing。イシュー
    /// #1311・設計 `docs/cpu-gemm-2d-dynamic-partition-design.md`）。
    /// [`partition::job_grid`] が worker 数より多い job 数を算出し、各
    /// job が自分の C 部分ブロックに対し K 全域を単一 worker が同期
    /// なしで処理する（`IcDynamic` の「pc ごとの同期点＋列全幅 B pack」
    /// という構造を繰り返さない設計。設計 §2.2）。C 列分割は job-local
    /// C staging（[`run_two_d_job`] の copy-in/out）方式（設計 §4.2
    /// 主案 S′）を採り `unsafe` を追加しない。両実機 A/B・採否判定は
    /// #1312、本番結線は #1313 が引き継ぐ。
    TwoDDynamic,
    /// (mc, nc) 2D 動的分配を SME（`fmopa`。イシュー #1587）マイクロ
    /// カーネル強制で計測する A/B 専用候補。実行 CPU が SME・非拡張
    /// FP32 外積に対応しない環境では [`gemm_blis_parallel_variant`] が
    /// panic する（`#[cfg(test)]` 限定の A/B 計測専用入口のため `Result`
    /// 化はせず早期に原因を明示する。`all_gemm_driver_variants` には
    /// 含めない・呼び出し元が `microkernel::SmeKernel::try_new().is_some()`
    /// を確認してから使う契約）。aarch64 限定（`SmeKernel` 自体が
    /// aarch64 限定トークンのため）。
    #[cfg(target_arch = "aarch64")]
    TwoDDynamicSme,
}

/// [`GemmDriverVariant`] で指定した候補を強制実行する A/B 計測専用入口
/// （イシュー #1041）。[`gemm_blis_parallel_with_blocks`] と同じ検証
/// （[`validate_dims`]・[`validate_block_sizes`]・`n == 0`／`m == 1` の
/// 専用経路）を経てから候補を分岐する。本番公開入口
/// （[`gemm_blis_parallel`]／[`gemm_blis_bias_act_parallel`]）は変更
/// しない（本関数はテスト・計測専用の新規入口）。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_blis_parallel_variant(
    variant: GemmDriverVariant,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    if n == 0 {
        return Ok(());
    }
    if m == 1 {
        gemm_row_vector(a, b, c, n);
        return Ok(());
    }

    match variant {
        GemmDriverVariant::RowPanel => {
            gemm_blis_parallel_row_panel_with_blocks(a, b, c, m, n, k, blocks)
        }
        GemmDriverVariant::SharedB => dispatch_shared_b(a, b, c, n, k, 0..m, blocks),
        GemmDriverVariant::SharedBPcOuter => {
            dispatch_shared_b_pc_outer(a, b, c, n, k, 0..m, blocks)
        }
        GemmDriverVariant::IcDynamic => dispatch_ic_dynamic(a, b, c, n, k, 0..m, blocks),
        #[cfg(target_arch = "aarch64")]
        GemmDriverVariant::RowPanelBLaneqVec => gemm_blis_parallel_row_panel_with_kernel(
            microkernel::NeonBLaneqVecKernel,
            a,
            b,
            c,
            m,
            n,
            k,
            blocks,
        ),
        GemmDriverVariant::TwoDDynamic => dispatch_two_d_dynamic(
            a,
            b,
            c,
            n,
            k,
            0..m,
            blocks,
            GemmTranspose::Nn,
            TWO_D_JOBS_PER_WORKER,
        ),
        #[cfg(target_arch = "aarch64")]
        GemmDriverVariant::TwoDDynamicSme => {
            // 本 variant のドキュメント参照: 呼び出し元が
            // `microkernel::SmeKernel::try_new().is_some()` を確認済み
            // であることを契約とする（`#[cfg(test)]` 限定の A/B 計測
            // 専用入口のため `expect` で早期に原因を明示する）。
            let kernel = microkernel::SmeKernel::try_new()
                .expect("TwoDDynamicSme variant requires SME-capable CPU");
            gemm_blis_two_d_dynamic_region(
                kernel,
                a,
                b,
                c,
                n,
                k,
                0..m,
                blocks,
                GemmTranspose::Nn,
                TWO_D_JOBS_PER_WORKER,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`GemmDriverVariant`] の全候補一覧（A/B 一括計測ハーネス・全候補
    /// bit 完全一致回帰の 3 箇所で共用。イシュー #1317）。
    /// `RowPanelBLaneqVec`（[`microkernel::NeonBLaneqVecKernel`] 経由）は
    /// aarch64 限定トークンに依存するため aarch64 版のみ追加する
    /// （`x86_64` で `let mut v = vec![…]; #[cfg(aarch64)] v.push(…)` と
    /// すると `unused_mut` lint が `-D warnings` で落ちる罠を避けるため、
    /// 2 定義に分ける方式を採る。計画 §3.4）。
    #[cfg(target_arch = "aarch64")]
    fn all_gemm_driver_variants() -> Vec<GemmDriverVariant> {
        vec![
            GemmDriverVariant::RowPanel,
            GemmDriverVariant::SharedB,
            GemmDriverVariant::SharedBPcOuter,
            GemmDriverVariant::IcDynamic,
            GemmDriverVariant::RowPanelBLaneqVec,
            GemmDriverVariant::TwoDDynamic,
        ]
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn all_gemm_driver_variants() -> Vec<GemmDriverVariant> {
        vec![
            GemmDriverVariant::RowPanel,
            GemmDriverVariant::SharedB,
            GemmDriverVariant::SharedBPcOuter,
            GemmDriverVariant::IcDynamic,
            GemmDriverVariant::TwoDDynamic,
        ]
    }

    #[test]
    fn gemm_blis_matches_hand_computed_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_blis_rejects_a_len_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        let err = gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap_err();
        assert!(matches!(
            err,
            GemmError::ALenMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    // MC/KC/NC 境界を跨ぐ多ブロック形状・並列版との bit 完全一致は
    // `bench_harness`（乱数生成）を要するため、lib 単体テスト
    // （本 `mod tests`）ではなく統合テスト `tests/gemm_blis_parity.rs`
    // 側に集約する。理由: `bench_harness` は `serde_json` を推移依存に
    // 持ち、lib 単体テストバイナリへ持ち込むと `reduction.rs` 側の
    // 無関係な `assert_eq!(&[usize], &[])` が `usize: PartialEq<_>` の
    // 複数実装（`core` と `serde_json::Value` 向け）で型推論あいまいに
    // なり `E0282/E0283` を起こす（同一バイナリにリンクされる依存の
    // trait 実装がクレート全体で可視になるため。実装時に実測確認済み）。
    // 統合テスト（`tests/`）は個別バイナリのためこの問題が生じない。

    #[test]
    fn gemm_blis_parallel_handles_zero_n_as_noop() {
        let a = vec![1.0f32; 4];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        assert!(gemm_blis_parallel(&a, &b, &mut c, 2, 0, 2).is_ok());
    }

    #[test]
    fn gemm_blis_handles_zero_dims() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        assert!(gemm_blis(&a, &b, &mut c, 0, 0, 0).is_ok());
    }

    // --- ワークロード閾値直列フォールバック（イシュー #811・#1027。
    //     `#[cfg(test)]` 限定の判定ロジック検証） ---
    //
    // `should_serialize` は `#[cfg(test)]` 限定であり本番公開入口
    // （`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）からは
    // 未参照（`GEMM_THREADING_THRESHOLD` ドキュメントコメント「本番
    // 未結線」参照）。以下はテスト専用の判定ロジック自体が実測表・
    // 境界形状で `gemm_naive`／非融合経路と bit 完全一致することを
    // 検証する（本番経路への影響はない）。乱数生成は `mod tests`
    // 既存の `xorshift32_vec`（`bench_harness` 非依存の理由は同関数の
    // ドキュメントコメント参照）を再利用する。

    /// `should_serialize` が実測表（`docs/perf/
    /// cpu-gemm-small-shape-serial-fallback.md` 実測 1〜3）の全形状で
    /// 期待される経路（直列／並列）と一致することを机上突合で固定する
    /// （#1027 実装計画 §2 の突合表）。`n >= NR_CLAMP` の形状はクランプの
    /// 影響を受けないため `m*n*k` 単純積と同じ判定になり、Bugbot 反例
    /// 形状（m=512,n=1,k=512。n=1 < NR_CLAMP）のみクランプにより並列側へ
    /// 是正される。
    #[test]
    fn should_serialize_matches_measured_table() {
        let cases: &[(&str, usize, usize, usize, bool)] = &[
            // 正方（実測 1〜3）: 16/32/64 は直列が優位、128/256 は並列が優位。
            ("square_16", 16, 16, 16, true),
            ("square_32", 32, 32, 32, true),
            ("square_64", 64, 64, 64, true),
            ("square_128", 128, 128, 128, false),
            ("square_256", 256, 256, 256, false),
            // 細長（gemv 相当）: m=512,n=1,k=512 は Bugbot 反例（単純
            // m*n*k=262,144 は閾値未満だが並列が 2.233x 優位）。
            // NR_CLAMP により n を 8 とみなすと 512*8*512=2,097,152
            // で閾値を超えるため並列側へ是正される。
            ("tall_skinny_m512_n1_k512", 512, 1, 512, false),
            // gevv（k<=2）: m=n=256〜512 は直列寄り、m=n=1024〜2048 は
            // 並列がやや優位（実測 1）。n は 256 以上のためクランプ非該当。
            ("gevv_256_k1", 256, 256, 1, true),
            ("gevv_256_k2", 256, 256, 2, true),
            ("gevv_512_k1", 512, 512, 1, true),
            ("gevv_512_k2", 512, 512, 2, true),
            ("gevv_1024_k1", 1024, 1024, 1, false),
            ("gevv_1024_k2", 1024, 1024, 2, false),
            ("gevv_2048_k1", 2048, 2048, 1, false),
            ("gevv_2048_k2", 2048, 2048, 2, false),
        ];
        for &(label, m, n, k, expect_serial) in cases {
            assert_eq!(
                should_serialize(m, n, k),
                expect_serial,
                "should_serialize({m}, {n}, {k})（{label}）の判定が実測表と不一致"
            );
        }
    }

    /// オーバーフロー時は `saturating_mul` により `usize::MAX` へ飽和し
    /// 常に並列側（安全側）へ倒れることを確認する。
    #[test]
    fn should_serialize_overflow_saturates_to_parallel() {
        assert!(!should_serialize(usize::MAX, usize::MAX, usize::MAX));
    }

    /// `GEMM_THREADING_THRESHOLD` 直下（[`should_serialize`] が直列判定
    /// する形状）で、テスト専用候補ディスパッチ
    /// [`gemm_blis_parallel_thresholded`]（実機ゲート通過後の本番結線
    /// 候補と同一の分岐ロジック）が実際に直列側（[`gemm_blis`]）を選択
    /// したうえで `gemm_naive` と bit 完全一致することを確認する（#1027
    /// codex-review P1 指摘: 削除された分岐検証の回帰経路を復元）。
    #[test]
    fn gemm_blis_parallel_threshold_boundary_below_matches_naive() {
        // 64*64*64 = 262,144 < GEMM_THREADING_THRESHOLD(589,824)。
        let (m, n, k) = (64, 64, 64);
        assert!(should_serialize(m, n, k));
        let a = xorshift32_vec(0x3333_3333, m * k);
        let b = xorshift32_vec(0x4444_4444, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_thresholded = vec![0.0f32; m * n];
        gemm_blis_parallel_thresholded(&a, &b, &mut c_thresholded, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_thresholded,
            "閾値直下（直列フォールバック経路）は gemm_naive と bit 完全一致するはず"
        );
    }

    /// `GEMM_THREADING_THRESHOLD` 直上（[`should_serialize`] が並列判定
    /// する形状）で、テスト専用候補ディスパッチ
    /// [`gemm_blis_parallel_thresholded`] が実際に並列側
    /// （[`gemm_blis_parallel`]）を選択したうえで `gemm_naive` と bit
    /// 完全一致することを確認する（#1027 codex-review P1 指摘対応）。
    #[test]
    fn gemm_blis_parallel_threshold_boundary_above_matches_naive() {
        // 128*128*128 = 2,097,152 > GEMM_THREADING_THRESHOLD(589,824)。
        let (m, n, k) = (128, 128, 128);
        assert!(!should_serialize(m, n, k));
        let a = xorshift32_vec(0x5555_5555, m * k);
        let b = xorshift32_vec(0x6666_6666, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_thresholded = vec![0.0f32; m * n];
        gemm_blis_parallel_thresholded(&a, &b, &mut c_thresholded, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_thresholded,
            "閾値直上（並列経路）は gemm_naive と bit 完全一致するはず"
        );
    }

    /// 閾値直下の直列フォールバック経路（epilogue 融合版のテスト専用
    /// 候補ディスパッチ [`gemm_blis_bias_act_parallel_thresholded`]）が、
    /// 分岐後も bias 加算・activation を含めて `gemm_blis_parallel` →
    /// 個別 bias/act 適用と bit 完全一致することを確認する
    /// （`gemm_blis_bias_act_parallel` 冒頭ドキュメント「bit 完全一致
    /// 契約」・#1027 codex-review P1 指摘: 分岐後の epilogue 適用検証を
    /// 復元）。
    #[test]
    fn gemm_blis_bias_act_parallel_threshold_boundary_below_matches_unfused() {
        let (m, n, k) = (32, 48, 40); // 32*48*40 = 61,440 < 589,824
        assert!(should_serialize(m, n, k));
        let a = xorshift32_vec(0x7777_7777, m * k);
        let b = xorshift32_vec(0x8888_8888, k * n);
        let bias = xorshift32_vec(0x9999_9999, n);

        let mut c_fused = vec![0.0f32; m * n];
        gemm_blis_bias_act_parallel_thresholded(
            &a,
            &b,
            &mut c_fused,
            m,
            n,
            k,
            Some(&bias),
            Activation::Relu,
        )
        .unwrap();

        let mut c_unfused = vec![0.0f32; m * n];
        gemm_blis_parallel(&a, &b, &mut c_unfused, m, n, k).unwrap();
        apply_epilogue(&mut c_unfused, n, Some(&bias), Activation::Relu).unwrap();

        assert_eq!(
            c_fused, c_unfused,
            "閾値直下の epilogue 融合経路（直列分岐後）は非融合の gemm_blis_parallel + apply_epilogue と bit 完全一致するはず"
        );
    }

    /// `n == 0` は `should_serialize` が常に `true`（直列 `gemm_blis` +
    /// `apply_epilogue` 分岐）になるが、`gemm_blis_bias_act_parallel`
    /// （本番入口）は分岐より前に `n == 0` を早期 return するため
    /// `apply_epilogue` の `chunks_mut(0)` パニックは発生しない（PR #830
    /// Cursor Bugbot 指摘の回帰）。`bias` が `Some` の境界形状で no-op に
    /// なることを確認する。
    #[test]
    fn gemm_blis_bias_act_parallel_handles_zero_n_as_noop() {
        let a = vec![1.0f32; 4];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        let bias: [f32; 0] = [];
        assert!(
            gemm_blis_bias_act_parallel(&a, &b, &mut c, 2, 0, 2, Some(&bias), Activation::Relu,)
                .is_ok()
        );
    }

    /// `n == 0` の早期 return が検証（`validate_dims`／bias 長／
    /// activation）より前に実行されて不正入力を素通りしないことを確認する
    /// （PR #830 Cursor Bugbot 指摘）。`a` の長さが `m * k` と不一致な
    /// n == 0 形状は `GemmError::ALenMismatch` を返すはず。
    #[test]
    fn gemm_blis_bias_act_parallel_validates_before_zero_n_early_return() {
        let a = vec![1.0f32; 3]; // m * k == 4 を期待するが 3 要素しかない不正形状
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        let bias: [f32; 0] = [];
        let result =
            gemm_blis_bias_act_parallel(&a, &b, &mut c, 2, 0, 2, Some(&bias), Activation::Relu);
        assert!(
            matches!(
                result,
                Err(GemmError::ALenMismatch {
                    expected: 4,
                    actual: 3
                })
            ),
            "n == 0 でも validate_dims の検証は先に実行されるはず: {result:?}"
        );
    }

    /// Cursor Bugbot 指摘（PR #830）の tall-skinny 回帰形状（`m` が大きく
    /// `n`/`k` が小さいため単純 `m*n*k` は閾値未満だが行パネル分割は十分
    /// 機能する）で、テスト専用候補ディスパッチ
    /// [`gemm_blis_parallel_thresholded`] が `NR_CLAMP` 適用後の判定に
    /// 従って実際に並列側（[`gemm_blis_parallel`]）を選択したうえで
    /// `gemm_naive` と bit 完全一致することを確認する（`NR_CLAMP` により
    /// `should_serialize` は `false`（並列側）と判定するはず。
    /// `should_serialize_matches_measured_table` でも同形状を検証済み。
    /// #1027 codex-review P1 指摘: 候補ディスパッチを実際に通す回帰経路
    /// を復元）。
    #[test]
    fn gemm_blis_parallel_tall_skinny_matches_naive() {
        // m=512,n=1,k=512: 単純 m*n*k = 262,144 < GEMM_THREADING_THRESHOLD(589,824)
        // だが NR_CLAMP 適用後は 512*8*512 = 2,097,152 で閾値超過。
        let (m, n, k) = (512, 1, 512);
        assert!(!should_serialize(m, n, k));
        let a = xorshift32_vec(0x1111_2222, m * k);
        let b = xorshift32_vec(0x3333_4444, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_thresholded = vec![0.0f32; m * n];
        gemm_blis_parallel_thresholded(&a, &b, &mut c_thresholded, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_thresholded,
            "tall-skinny 形状でもテスト専用候補ディスパッチは gemm_naive と bit 完全一致するはず"
        );
    }

    // --- gemv 相当（m == 1）専用経路（イシュー #811） ---

    /// `gemm_blis`（直列公開入口）の m==1 専用経路が `gemm_naive` と
    /// bit 完全一致することを確認する。
    #[test]
    fn gemm_blis_row_vector_matches_naive() {
        let (m, n, k) = (1, 777, 513);
        let a = xorshift32_vec(0xaaaa_aaaa, m * k);
        let b = xorshift32_vec(0xbbbb_bbbb, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_blis = vec![0.0f32; m * n];
        gemm_blis(&a, &b, &mut c_blis, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_blis,
            "m==1 の gemm_blis 専用経路は gemm_naive と bit 完全一致するはず"
        );
    }

    /// `gemm_blis_parallel` の m==1 専用経路が `gemm_naive` と bit 完全
    /// 一致することを確認する（`m*n*k` が閾値を超える大きい n・k でも
    /// m==1 判定が閾値判定より先に効くことを確認するため大きめの形状を
    /// 選ぶ）。
    #[test]
    fn gemm_blis_parallel_row_vector_matches_naive() {
        let (m, n, k) = (1, 4096, 4096);
        assert!(m * n * k > GEMM_THREADING_THRESHOLD);
        let a = xorshift32_vec(0xcccc_cccc, m * k);
        let b = xorshift32_vec(0xdddd_dddd, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_parallel = vec![0.0f32; m * n];
        gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_parallel,
            "m==1 の gemm_blis_parallel 専用経路は gemm_naive と bit 完全一致するはず"
        );
    }

    /// `gemm_blis_bias_act_parallel` の m==1 専用経路が bias/act を
    /// 含めて非融合経路と bit 完全一致することを確認する。
    #[test]
    fn gemm_blis_bias_act_parallel_row_vector_matches_unfused() {
        let (m, n, k) = (1, 300, 250);
        let a = xorshift32_vec(0xeeee_eeee, m * k);
        let b = xorshift32_vec(0xffff_0001, k * n);
        let bias = xorshift32_vec(0xffff_0002, n);

        let mut c_fused = vec![0.0f32; m * n];
        gemm_blis_bias_act_parallel(&a, &b, &mut c_fused, m, n, k, Some(&bias), Activation::Relu)
            .unwrap();

        let mut c_unfused = vec![0.0f32; m * n];
        gemm_blis_parallel(&a, &b, &mut c_unfused, m, n, k).unwrap();
        apply_epilogue(&mut c_unfused, n, Some(&bias), Activation::Relu).unwrap();

        assert_eq!(
            c_fused, c_unfused,
            "m==1 の epilogue 融合経路は非融合経路と bit 完全一致するはず"
        );
    }

    /// xorshift32 による疑似乱数ベクトル生成（テスト専用。`bench_harness`
    /// を持ち込まない理由は [`microkernel::avx2`] の同名関数のドキュメント
    /// コメント参照。lib 単体テストバイナリへ `serde_json` 推移依存を
    /// 持ち込むと型推論あいまいで E0282/E0283 を起こすため）。
    fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f64 / u32::MAX as f64) as f32
            })
            .collect()
    }

    /// 受け入れ条件「非対応環境でスカラーフォールバックが動作する」を
    /// 実行環境の実際の ISA 検出結果に依らず検証する（[`gemm_blis_with_kernel`]
    /// で [`ScalarKernel`] を強制し、[`crate::gemm::gemm_naive`] と bit
    /// 完全一致することを確認）。MC/KC/NC 境界を跨ぐ形状を選ぶ。
    #[test]
    fn gemm_blis_scalar_kernel_forced_matches_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0x1111_1111, m * k);
        let b = xorshift32_vec(0x2222_2222, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_scalar = vec![0.0f32; m * n];
        gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_scalar, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_scalar,
            "ScalarKernel 強制経路は gemm_naive と bit 完全一致するはず"
        );
    }

    /// 実行環境で検出された ISA を使う公開入口 [`gemm_blis`] の結果と、
    /// [`ScalarKernel`] 強制経路の結果が bit 完全一致することを確認する
    /// （ISA 間 bit 一致契約〈REQ-2〉の実行時 dispatch 版検証。どの ISA が
    /// 選ばれても [`gemm_blis`] は `ScalarKernel` 強制経路と同じ結果を返す
    /// はず）。
    #[test]
    fn gemm_blis_detected_isa_matches_scalar_forced_bit_exact() {
        let (m, n, k) = (129, 130, 131);
        let a = xorshift32_vec(0x3333_3333, m * k);
        let b = xorshift32_vec(0x4444_4444, k * n);

        let mut c_detected = vec![0.0f32; m * n];
        gemm_blis(&a, &b, &mut c_detected, m, n, k).unwrap();

        let mut c_scalar = vec![0.0f32; m * n];
        gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_scalar, m, n, k).unwrap();

        assert_eq!(
            c_detected, c_scalar,
            "実行時検出された ISA 経路と ScalarKernel 強制経路は bit 完全一致するはず"
        );
    }

    /// #557: 全タイルが完全タイル（直接経路のみを通る）形状で
    /// `gemm_naive` と bit 完全一致することを確認する（[`ScalarKernel`]
    /// 強制。MR=NR=4 の scalar タイル形状に対し m・n がともに倍数の形状
    /// を選ぶことで、端タイル分岐（コピー経路）を一切通さずに直接経路
    /// のみを検証する）。
    #[test]
    fn gemm_blis_scalar_kernel_all_full_tiles_matches_naive_bit_exact() {
        // ScalarKernel は MR=4・NR=4（scalar.rs 参照）。m・n をともに 4 の
        // 倍数にし、かつ MC/KC/NC 境界（128/256/512）を跨ぐ形状を選び、
        // 全 ic/jc/jr/ir 反復で mr_eff == MR && nr_eff == NR が成立する
        // ようにする。
        let (m, n, k) = (256, 512, 300);
        assert_eq!(m % ScalarKernel::MR, 0);
        assert_eq!(n % ScalarKernel::NR, 0);

        let a = xorshift32_vec(0x5555_5555, m * k);
        let b = xorshift32_vec(0x6666_6666, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_direct = vec![0.0f32; m * n];
        gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_direct, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_direct,
            "全タイル完全（C 直接経路のみ）でも gemm_naive と bit 完全一致するはず"
        );
    }

    /// [`validate_block_sizes`] が `mc`／`kc`／`nc` の 0 値を早期拒否する
    /// ことを検証する（`step_by(0)` パニック防止。`crate::gemm::
    /// GemmError::ZeroBlockSize` 同種バグ〈Cursor Bugbot #231〉の
    /// gemm_blis 版再発防止。#564）。3 フィールドそれぞれを 0 にしたケースを
    /// 個別に検査する。
    #[test]
    fn gemm_blis_with_kernel_and_blocks_rejects_zero_block_size() {
        let a = vec![1.0f32; 4];
        let b = vec![1.0f32; 4];
        let mut c = vec![0.0f32; 4];

        for blocks in [
            BlockSizes {
                mc: 0,
                kc: 4,
                nc: 4,
            },
            BlockSizes {
                mc: 4,
                kc: 0,
                nc: 4,
            },
            BlockSizes {
                mc: 4,
                kc: 4,
                nc: 0,
            },
        ] {
            let err =
                gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c, 2, 2, 2, blocks)
                    .unwrap_err();
            assert!(
                matches!(err, GemmError::ZeroBlockSize { .. }),
                "blocks={blocks:?} は ZeroBlockSize で拒否されるはず: {err:?}"
            );
        }
    }

    /// 参照実装値近傍を含む非既定 `BlockSizes`（#564 §3.4 候補グリッドの
    /// 縮小版・境界を跨ぐ／下回る／firestorm 近傍の組）でも
    /// [`crate::gemm::gemm_naive`] と bit 完全一致することを検証する
    /// （§3.2「bit 完全一致契約が維持される根拠」の直接検証。x86_64 でも
    /// 実行可能。ScalarKernel 強制で ISA 差を排除する）。
    #[test]
    fn gemm_blis_non_default_block_sizes_match_naive_bit_exact() {
        let (m, n, k) = (37, 53, 71);
        let a = xorshift32_vec(0x9999_9999, m * k);
        let b = xorshift32_vec(0xaaaa_aaaa, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        // 現行値・境界跨ぎの小さい奇数系・firestorm 参照値近傍（MC=480/
        // KC=4096/NC=9600 は m,n,k=37/53/71 に対して常に 1 ブロックに
        // クランプされるが、境界跨ぎと同じ「clamp が正しく効く」経路を
        // 検証する意味を持つ）を横断する。
        for blocks in [
            default_blocks(),
            BlockSizes {
                mc: 8,
                kc: 4,
                nc: 12,
            },
            BlockSizes {
                mc: 16,
                kc: 17,
                nc: 19,
            },
            BlockSizes {
                mc: 480,
                kc: 4096,
                nc: 9600,
            },
            BlockSizes {
                mc: 256,
                kc: 1024,
                nc: 4096,
            },
        ] {
            let mut c_blocked = vec![0.0f32; m * n];
            gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c_blocked, m, n, k, blocks)
                .unwrap();

            assert_eq!(
                c_naive, c_blocked,
                "blocks={blocks:?} は gemm_naive と bit 完全一致するはず"
            );
        }
    }

    /// [`gemm_blis_parallel_with_blocks`]（実運用の行パネル並列経路
    /// `gemm_blis_parallel` の任意 `BlockSizes` 版。#564 スイープ基盤）が、
    /// 非既定 `blocks`（境界跨ぎ・firestorm 参照値近傍）でも
    /// [`crate::gemm::gemm_naive`] と bit 完全一致することを検証する
    /// （epilogue 融合と同じ理由〈要素ごとの演算はタスク分割順序に依存
    /// しない〉で、並列パネル分割数に依らず結果が一致するはずという
    /// §3.2 の主張を並列経路でも直接確認する。x86_64 でも実行可能）。
    #[test]
    fn gemm_blis_parallel_non_default_block_sizes_match_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0xdddd_dddd, m * k);
        let b = xorshift32_vec(0xeeee_eeee, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        for blocks in [
            default_blocks(),
            BlockSizes {
                mc: 16,
                kc: 17,
                nc: 19,
            },
            BlockSizes {
                mc: 480,
                kc: 4096,
                nc: 9600,
            },
        ] {
            let mut c_parallel = vec![0.0f32; m * n];
            gemm_blis_parallel_with_blocks(&a, &b, &mut c_parallel, m, n, k, blocks).unwrap();

            assert_eq!(
                c_naive, c_parallel,
                "blocks={blocks:?} の並列経路は gemm_naive と bit 完全一致するはず"
            );
        }
    }

    /// #753: 2 次元タイルジョブ分配版
    /// [`gemm_blis_parallel_2d_with_blocks`] が `gemm_naive` と bit 完全
    /// 一致することを、MC タイル数がスレッド数で割り切れない形状
    /// （`blocks.mc=64` に対し `m=523` → タイル数 9・スレッド数
    /// [1,2,3,5,16] のいずれとも非整除）× 複数スレッド数で検証する。
    /// `gemm_blis_parallel_with_blocks` と異なる行範囲計算
    /// （[`partition::row_ranges_for_workers`]）を経由しても、C 各要素の
    /// FMA 連鎖・累積順序は `dispatch_region`／`gemm_blis_region` を
    /// そのまま再利用するため変化しない（bit 完全一致契約・REQ-2）。
    #[test]
    fn gemm_blis_parallel_2d_matches_naive_bit_exact_across_thread_pools() {
        let (m, n, k) = (523, 600, 700);
        let blocks = BlockSizes {
            mc: 64,
            kc: 256,
            nc: 512,
        };
        let a = xorshift32_vec(0x5e5e_5e5e, m * k);
        let b = xorshift32_vec(0x6f6f_6f6f, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        for num_threads in [1usize, 2, 3, 5, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap_or_else(|e| panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}"));

            let mut c_2d = vec![0.0f32; m * n];
            pool.install(|| {
                gemm_blis_parallel_2d_with_blocks(&a, &b, &mut c_2d, m, n, k, blocks).unwrap()
            });

            assert_eq!(
                c_naive, c_2d,
                "gemm_blis_parallel_2d_with_blocks（num_threads={num_threads}）が \
                 gemm_naive と bit 一致しない"
            );
        }
    }

    /// #753: MC/KC/NC の実行時キャッシュ検出（[`cache_params::detected_blocks`]）
    /// で算出したブロックサイズを [`gemm_blis_parallel_with_blocks`] へ
    /// 渡しても `gemm_naive` と bit 完全一致することを検証する（実行環境
    /// 依存の値であっても、GEMM 本体の FMA 契約・累積順序は `blocks` の
    /// 値に依らず不変という [`gemm_blis_region`] の契約〈本ファイル冒頭
    /// ドキュメント「bit 完全一致契約」〉が MC/KC/NC の動的算出後も
    /// 成立することの回帰テスト）。`ScalarKernel` の `MR`／`NR` を渡し、
    /// 実行 ISA に依らず全環境で同じ形状の `blocks` になるようにする。
    #[test]
    fn gemm_blis_parallel_detected_blocks_match_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0x7a7a_7a7a, m * k);
        let b = xorshift32_vec(0x8b8b_8b8b, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let blocks = cache_params::detected_blocks(ScalarKernel::MR, ScalarKernel::NR);

        let mut c_parallel = vec![0.0f32; m * n];
        gemm_blis_parallel_with_blocks(&a, &b, &mut c_parallel, m, n, k, blocks).unwrap();

        assert_eq!(
            c_naive, c_parallel,
            "detected_blocks() 由来の blocks={blocks:?} は gemm_naive と bit 完全一致するはず"
        );
    }

    /// #753 レビュー指摘: 上記
    /// `gemm_blis_parallel_detected_blocks_match_naive_bit_exact` は Linux CI
    /// 上では `cache_params::read_cache_sizes()` が常に `None` を返すため
    /// `detected_blocks()` が実質 `default_blocks()` へフォールバックし、
    /// 動的算出（`compute_blocks`）経由の `BlockSizes` を一度も
    /// `gemm_blis_parallel_with_blocks` へ通していなかった（Linux では
    /// `sysctl` 自体が存在せず実測は macOS 実機限定のため、CI 上で動的
    /// 検出を再現するには [`cache_params::compute_blocks`] へ具体値を
    /// 直接渡す必要がある）。Apple M4 Max P コア相当の代表値
    /// （L1D=192KiB・L2=16MiB。#481 §3）を渡して得た `BlockSizes`
    /// （`default_blocks()` とは異なる算出値になる）を GEMM 本体へ通し、
    /// `gemm_naive` と bit 完全一致することを検証する。
    ///
    /// #753 レビュー指摘（Bugbot／codex-review）: 本テストは
    /// `gemm_blis_parallel_with_blocks` を直接呼ぶため、aarch64 実行時は
    /// `dispatch_region`（aarch64 版）が常に `NeonKernel`（MR=8・NR=12。
    /// #559）を選ぶ。`compute_blocks` へ渡す `mr`／`nr` を実行時に選ばれる
    /// カーネルと一致させないと「M4 Max 相当の代表値」の意図から外れる
    /// ため、`ScalarKernel::MR`／`NR`（4×4）ではなく実際に aarch64 上で
    /// 走る `NeonKernel` の値を使う（他 arch でも `cargo test` は本テストを
    /// コンパイル対象に含むため、`NeonKernel` 型自体が存在しない非 aarch64
    /// では `ScalarKernel` の値へフォールバックする。bit 完全一致の
    /// 検証意図には mr/nr の実値そのものは影響しない＝非 aarch64 での
    /// フォールバックは検証結果を歪めない）。
    #[test]
    fn gemm_blis_parallel_compute_blocks_m4_max_like_values_match_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0x9c9c_9c9c, m * k);
        let b = xorshift32_vec(0xadad_adad, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        #[cfg(target_arch = "aarch64")]
        let (mr, nr) = (microkernel::NeonKernel::MR, microkernel::NeonKernel::NR);
        #[cfg(not(target_arch = "aarch64"))]
        let (mr, nr) = (ScalarKernel::MR, ScalarKernel::NR);

        let blocks = cache_params::compute_blocks(192 * 1024, 16 * 1024 * 1024, mr, nr)
            .expect("M4 Max 相当の正当な値は Some を返すはず");
        assert_ne!(
            blocks,
            default_blocks(),
            "本テストは動的算出経路（default_blocks と異なる値）を検証する意図のため、\
             両者が一致すると検証意図が失われる"
        );

        let mut c_parallel = vec![0.0f32; m * n];
        gemm_blis_parallel_with_blocks(&a, &b, &mut c_parallel, m, n, k, blocks).unwrap();

        assert_eq!(
            c_naive, c_parallel,
            "compute_blocks() 由来の blocks={blocks:?} は gemm_naive と bit 完全一致するはず"
        );
    }

    /// イシュー #750 の受け入れ条件 3（スレッド数 1 では従来と同一経路・
    /// 同一性能）を直接検証する: `num_threads(1)` の rayon プール内で
    /// `gemm_blis_parallel` を呼ぶと `m <= panel_rows`（`panel_rows == m`）
    /// が常に成立し `dispatch_shared_b`（B パネル共有経路）を一切経由し
    /// ない設計になっている（`gemm_blis_parallel` 実装コメント参照）。
    /// 本テストはその結果として `gemm_naive` と bit 完全一致することを
    /// MC/KC/NC 境界を跨ぐ形状で確認する。
    #[test]
    fn gemm_blis_parallel_single_thread_pool_matches_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0x1a1a_1a1a, m * k);
        let b = xorshift32_vec(0x2b2b_2b2b, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap_or_else(|e| panic!("1 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_parallel = vec![0.0f32; m * n];
        pool.install(|| gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap());

        assert_eq!(
            c_naive, c_parallel,
            "num_threads=1 の gemm_blis_parallel は gemm_naive と bit 完全一致するはず（#750 受け入れ条件 3）"
        );
    }

    /// イシュー #750: B パネル共有経路（[`gemm_blis_shared_b_region`]）が
    /// 複数 (jc,pc) ブロック（＝複数回の同期点）を跨いでも
    /// [`gemm_blis_region`]（直列経路）と bit 完全一致することを、小さい
    /// `BlockSizes`（mc=16・kc=17・nc=19。既定値より大幅に小さく多数の
    /// (jc,pc) 反復を強制する）で検証する。固定 4 スレッドプールで
    /// `m > panel_rows`（実タスク数 >= 2）を確定させ、共有 B 経路を確実に
    /// 通す。
    #[test]
    fn gemm_blis_shared_b_region_multi_sync_point_matches_serial_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let blocks = BlockSizes {
            mc: 16,
            kc: 17,
            nc: 19,
        };
        let a = xorshift32_vec(0x3c3c_3c3c, m * k);
        let b = xorshift32_vec(0x4d4d_4d4d, k * n);

        let mut c_serial = vec![0.0f32; m * n];
        gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c_serial, m, n, k, blocks)
            .unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap_or_else(|e| panic!("4 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_shared_b = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_with_blocks(&a, &b, &mut c_shared_b, m, n, k, blocks).unwrap()
        });

        assert_eq!(
            c_serial, c_shared_b,
            "多 (jc,pc) 同期点を跨ぐ B パネル共有経路は直列経路と bit 完全一致するはず（#750）"
        );
    }

    /// イシュー #750: 実タスク数 Q が rayon の稼働スレッド数 T を下回る
    /// 形状（m が小さく `m.div_ceil(num_threads) < num_threads` となる
    /// ケース）でも `gemm_naive` と bit 完全一致することを確認する
    /// （`gemm_blis_shared_b_region` の `num_tasks = mc_total.div_ceil(
    /// panel_rows)` が実際のタスク数を正しく導出し、`a_bufs`／
    /// `c.par_chunks_mut` の長さがずれないことの回帰）。
    ///
    /// 本番公開入口 `gemm_blis_parallel` は B パネル共有経路
    /// （`dispatch_shared_b`）を採用しない（本ファイル冒頭
    /// `gemm_blis_parallel` 実装コメント参照。#750・codex-review P1 是正）
    /// ため、`gemm_blis_shared_b_region` の Q<T 回帰を実際に検証するには
    /// `#[cfg(test)]` 限定のテスト専用入口 [`gemm_blis_parallel_with_blocks`]
    /// を経由する必要がある（Cursor Bugbot 指摘・commit f27f233 是正: 本
    /// テストが `gemm_blis_parallel` を直接呼ぶと共有経路を一切経由せず
    /// 回帰検証が失効する）。
    #[test]
    fn gemm_blis_parallel_matches_naive_bit_exact_when_tasks_fewer_than_threads() {
        let (m, n, k) = (10, 130, 40);
        let a = xorshift32_vec(0x5e5e_5e5e, m * k);
        let b = xorshift32_vec(0x6f6f_6f6f, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap_or_else(|e| panic!("16 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_parallel = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_with_blocks(&a, &b, &mut c_parallel, m, n, k, default_blocks())
                .unwrap()
        });

        assert_eq!(
            c_naive, c_parallel,
            "m={m} 行を num_threads=16 で分割する実タスク数 Q < T のケースは \
             gemm_naive と bit 完全一致するはず（#750）"
        );
    }

    /// [`panel_capacity`]（[`PanelBuffers::new`] の容量計算本体）が、
    /// `gemm_blis_region` の全 jc×pc×ic ブロック反復における実際の必要量
    /// （`nr_blocks*kc_len*nr`／`mr_blocks*kc_len*mr`）の上界になっている
    /// ことを、複数形状 × 全 ISA カーネル定数相当の (mr, nr) 総当たりで
    /// 検証する（#556。§4.1 の「先頭ブロックが最大」という設計根拠の
    /// 直接検証。ブロック境界を跨ぐ形状〈MC/KC/NC 未満・ちょうど・超過〉
    /// を含める）。
    #[test]
    fn panel_capacity_upper_bounds_all_block_iterations() {
        // scalar 4x4・neon 8x12（既定）・neon 12x8（A/B 対抗変種。#559）・
        // avx2 6x16・avx512 8x32（`microkernel/*.rs`）。
        const KERNEL_DIMS: [(usize, usize); 5] = [(4, 4), (8, 12), (12, 8), (6, 16), (8, 32)];
        // MC/KC/NC 境界（128/256/512）を跨ぐ・下回る・ちょうどの形状。
        const SHAPES: [(usize, usize, usize); 6] = [
            (1, 1, 1),
            (7, 7, 7),
            (128, 256, 128),
            (129, 257, 129),
            (600, 700, 300),
            (512, 256, 128),
        ];
        // 検証対象の BlockSizes 候補（レビュー指摘 #564: 既定値のみでは
        // `gemm_blis_with_kernel_and_blocks`／`gemm_blis_parallel_with_blocks`
        // が新規に開放した非既定 blocks 経路の upper bound 不変条件を
        // 検証できない）。既定値・MC/KC/NC 境界を跨ぐ小さめの値・firestorm
        // 参照値近傍（実機スイープ候補として想定される大きめの値）を含める。
        // この不変条件が破れた場合の失敗モードは `gemm_blis_region` 内での
        // スライスインデックスパニックであり、境界を跨ぐ値でこそ検知できる。
        let blocks_candidates: [BlockSizes; 3] = [
            default_blocks(),
            BlockSizes {
                mc: 16,
                kc: 17,
                nc: 19,
            },
            BlockSizes {
                mc: 480,
                kc: 4096,
                nc: 9600,
            },
        ];

        for &blocks in &blocks_candidates {
            for &(mr, nr) in &KERNEL_DIMS {
                for &(n, k_dim, mc_total) in &SHAPES {
                    let (b_cap, a_cap) = panel_capacity(n, k_dim, mc_total, mr, nr, blocks);

                    for jc in (0..n).step_by(blocks.nc) {
                        let nc_len = blocks.nc.min(n - jc);
                        for pc in (0..k_dim).step_by(blocks.kc) {
                            let kc_len = blocks.kc.min(k_dim - pc);
                            let nr_blocks = nc_len.div_ceil(nr);
                            let b_needed = nr_blocks * kc_len * nr;
                            assert!(
                                b_needed <= b_cap,
                                "B 容量不足: blocks={blocks:?},mr={mr},nr={nr},n={n},k={k_dim},\
                                 mc_total={mc_total},jc={jc},pc={pc}: needed={b_needed} > cap={b_cap}"
                            );

                            let mut ic = 0;
                            while ic < mc_total {
                                let mc_len = blocks.mc.min(mc_total - ic);
                                let mr_blocks = mc_len.div_ceil(mr);
                                let a_needed = mr_blocks * kc_len * mr;
                                assert!(
                                    a_needed <= a_cap,
                                    "A 容量不足: blocks={blocks:?},mr={mr},nr={nr},n={n},k={k_dim},\
                                     mc_total={mc_total},jc={jc},pc={pc},ic={ic}: \
                                     needed={a_needed} > cap={a_cap}"
                                );
                                ic += blocks.mc;
                            }
                        }
                    }
                }
            }
        }
    }

    /// `default_blocks()` の値がすべて 0 でないこと（0 だと
    /// `panel_capacity` の `div_ceil`／`step_by` がパニックしうる）を
    /// 確認する（実装計画 §6 OWASP A03 の静的保証をテストで裏付ける）。
    #[test]
    fn default_blocks_never_returns_zero_sized_block() {
        let blocks = default_blocks();
        assert!(blocks.mc > 0 && blocks.kc > 0 && blocks.nc > 0);
    }

    /// n が大きい場合（4096 前後。#749 で NC 拡大分岐の対象だった範囲）
    /// でも本番経路（[`gemm_blis`]／[`gemm_blis_parallel`]）が
    /// `gemm_naive` と bit 完全一致することを、閾値の直前・直後・NR=12
    /// （NEON 8×12 既定カーネルの NR）非整数倍の端タイルが生じる n で
    /// 検証する（m／k は小さく保ち実行時間を抑える）。
    #[test]
    fn gemm_blis_and_parallel_large_n_match_naive_bit_exact() {
        for &(m, k) in &[(5usize, 7usize), (131, 259)] {
            for n in [4095usize, 4096, 4097, 4100] {
                let a = xorshift32_vec(0x1234_5678, m * k);
                let b = xorshift32_vec(0x9abc_def0, k * n);

                let mut c_naive = vec![0.0f32; m * n];
                crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

                let mut c_blis = vec![0.0f32; m * n];
                gemm_blis(&a, &b, &mut c_blis, m, n, k).unwrap();
                assert_eq!(
                    c_naive, c_blis,
                    "gemm_blis（m={m},n={n},k={k}）は gemm_naive と bit 完全一致するはず"
                );

                let mut c_parallel = vec![0.0f32; m * n];
                gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();
                assert_eq!(
                    c_naive, c_parallel,
                    "gemm_blis_parallel（m={m},n={n},k={k}）は gemm_naive と bit 完全一致するはず"
                );
            }
        }
    }

    // --- NEON MR=8×NR=12 拡張の aarch64 限定検証（イシュー #559）---
    //
    // x86_64 開発環境では実行不能なため `cfg(target_arch = "aarch64")` で
    // ゲートする（`cargo check --target aarch64-unknown-linux-gnu` の
    // クロス型検査対象にはなるが、x86_64 通常 CI では実行されない。
    // 実機での bit 一致・A/B 実測は `docs/perf/cpu-gemm-neon-mr8-nr12.md`
    // 参照）。

    /// 既定 8×12 カーネル・12×8 A/B 対抗変種いずれも [`ScalarKernel`]
    /// 強制経路と bit 完全一致することを確認する（受け入れ条件 3: parity
    /// テストが green であること）。MC/KC/NC 境界を跨ぐ形状を選ぶ。
    ///
    /// k の一覧はイシュー #561（NEON k=4 アンロール）の主ループ／端数
    /// 分離（k_main = k - k%4）の剰余網羅用に拡張した: 元の k=700 は
    /// KC=256 ブロック分割で各領域の kc_len が 256/256/188（いずれも
    /// 4 の倍数）となり k%4 の剰余が常に 0 で端数ループを一切通らない
    /// （#561 で新設した端数ループの検証漏れになる）。k=701〜703 を
    /// 追加し、各々 KC 分割後の最終領域 kc_len が 189/190/191（k%4 が
    /// 1/2/3）になることで剰余 1/2/3 を通す。MC/NC 境界跨ぎは元の
    /// k=700・(m,n)=(200,600) ケースで既に検証済みのため、追加した
    /// k=701〜703 は端数分岐の検証のみが目的（重複検証を避け aarch64
    /// 実機セッションでの実行コストを抑えるため）小さい (m,n) を使う。
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_8x12_and_12x8_match_scalar_forced_bit_exact() {
        use microkernel::{Neon12x8Kernel, NeonBLaneqKernel, NeonBLaneqVecKernel, NeonKernel};

        for (i, &(m, n, k)) in [
            (200usize, 600usize, 700usize),
            (16, 20, 701),
            (16, 20, 702),
            (16, 20, 703),
        ]
        .iter()
        .enumerate()
        {
            let seed_a = 0x5555_5555u32 + i as u32;
            let seed_b = 0x6666_6666u32 + i as u32;
            let a = xorshift32_vec(seed_a, m * k);
            let b = xorshift32_vec(seed_b, k * n);

            let mut c_scalar = vec![0.0f32; m * n];
            gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_scalar, m, n, k).unwrap();

            let mut c_neon_8x12 = vec![0.0f32; m * n];
            gemm_blis_with_kernel(NeonKernel, &a, &b, &mut c_neon_8x12, m, n, k).unwrap();
            assert_eq!(
                c_scalar, c_neon_8x12,
                "NeonKernel（既定 8×12・k={k}）は ScalarKernel 強制経路と bit 完全一致するはず"
            );

            let mut c_neon_12x8 = vec![0.0f32; m * n];
            gemm_blis_with_kernel(Neon12x8Kernel, &a, &b, &mut c_neon_12x8, m, n, k).unwrap();
            assert_eq!(
                c_scalar, c_neon_12x8,
                "Neon12x8Kernel（A/B 対抗変種・k={k}）は ScalarKernel 強制経路と bit 完全一致するはず"
            );

            // イシュー #748: B 側レーン参照 FMA 変種（列優先 acc）も
            // ScalarKernel 強制経路と bit 完全一致するはず（k%4 の剰余
            // 網羅は上記グリッドを共用）。
            let mut c_neon_b_laneq = vec![0.0f32; m * n];
            gemm_blis_with_kernel(NeonBLaneqKernel, &a, &b, &mut c_neon_b_laneq, m, n, k).unwrap();
            assert_eq!(
                c_scalar, c_neon_b_laneq,
                "NeonBLaneqKernel（B レーン参照変種・k={k}）は ScalarKernel 強制経路と bit 完全一致するはず"
            );

            // イシュー #1317: B 側 laneq ベクトル転置版（[`compute_b_laneq`]
            // の C タイル転置をベクトル化した候補）も ScalarKernel 強制
            // 経路と bit 完全一致するはず（k%4 の剰余網羅は上記グリッドを
            // 共用）。
            let mut c_neon_b_laneq_vec = vec![0.0f32; m * n];
            gemm_blis_with_kernel(
                NeonBLaneqVecKernel,
                &a,
                &b,
                &mut c_neon_b_laneq_vec,
                m,
                n,
                k,
            )
            .unwrap();
            assert_eq!(
                c_scalar, c_neon_b_laneq_vec,
                "NeonBLaneqVecKernel（B laneq ベクトル転置版・k={k}）は ScalarKernel 強制経路と \
                 bit 完全一致するはず"
            );
        }
    }

    /// [`NeonKernel`]（既定 8×12）と [`Neon12x8Kernel`]（firestorm 型 A/B
    /// 対抗変種）のスループットを同一形状で計測し中央値を報告する
    /// （`.claude/rules/coding-rust.md` の 5 回計測中央値規約）。
    /// 採用可否の判定は本テストの出力を見た人間／後続セッションが行う
    /// ため、本テスト自体は勝敗を assert しない（イシュー #559 §2.3）。
    /// `#[ignore]` 分離（実機実行専用。`.claude/rules/coding-rust.md`
    /// 実機分離方針）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "aarch64 実機での A/B 性能計測専用（--release 実行推奨）"]
    fn neon_8x12_vs_12x8_ab_median_throughput() {
        use microkernel::{Neon12x8Kernel, NeonKernel};
        use std::time::Instant;

        fn run_once<K: Microkernel>(
            kernel: K,
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k_dim: usize,
        ) -> f64 {
            let mut c = vec![0.0f32; m * n];
            let start = Instant::now();
            gemm_blis_with_kernel(kernel, a, b, &mut c, m, n, k_dim).unwrap();
            start.elapsed().as_secs_f64()
        }

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        for dim in [512usize, 1024, 2048] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0x7777_7777, m * k);
            let b = xorshift32_vec(0x8888_8888, k * n);

            // キャッシュ・TLB を双方カーネルで温めてから計測に入る
            // （ウォームアップなしだと先に計測する側が cold cache を
            // 引き、後段が温まった状態を引き継ぐため系統的に偏る）。
            run_once(NeonKernel, &a, &b, m, n, k);
            run_once(Neon12x8Kernel, &a, &b, m, n, k);

            // 各反復で計測順序を交互化し、cursor[bot] 指摘（PR #693・
            // レビュースレッド PRRT_kwDOTuUCJc6ZnFza）が挙げた
            // 「NeonKernel を常に先に計測するため後段の
            // Neon12x8Kernel がキャッシュ/TLB の温まった状態を
            // 引き継ぎ 12x8 有利に系統的偏りうる」問題を解消する。
            let mut samples_8x12 = Vec::with_capacity(5);
            let mut samples_12x8 = Vec::with_capacity(5);
            for i in 0..5 {
                if i % 2 == 0 {
                    samples_8x12.push(run_once(NeonKernel, &a, &b, m, n, k));
                    samples_12x8.push(run_once(Neon12x8Kernel, &a, &b, m, n, k));
                } else {
                    samples_12x8.push(run_once(Neon12x8Kernel, &a, &b, m, n, k));
                    samples_8x12.push(run_once(NeonKernel, &a, &b, m, n, k));
                }
            }

            let median_8x12 = median(samples_8x12);
            let median_12x8 = median(samples_12x8);

            println!(
                "dim={dim}: NeonKernel(8x12) median={median_8x12:.6}s, \
                 Neon12x8Kernel(12x8) median={median_12x8:.6}s"
            );
        }
    }

    /// [`microkernel::NeonKernel`]（既定・A レーン参照・行優先 acc）と
    /// [`microkernel::NeonBLaneqKernel`]（B レーン参照・列優先 acc。
    /// イシュー #748）のスループットを同一形状で計測し中央値を報告する
    /// （`.claude/rules/coding-rust.md` の 5 回計測中央値規約・上記
    /// `neon_8x12_vs_12x8_ab_median_throughput` と同型のウォームアップ・
    /// 交互実行パターン）。C タイル転置コストを新規に抱えるため、
    /// 採用可否（既定ディスパッチへの接続）の判定は本テスト出力を見た
    /// 人間／後続セッションが行う（本テスト自体は勝敗を assert しない。
    /// #748 実装計画 §2 の fail-closed 方針）。`#[ignore]` 分離（実機実行
    /// 専用。`.claude/rules/coding-rust.md` 実機分離方針）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "aarch64 実機での A/B 性能計測専用（--release 実行推奨）"]
    fn neon_8x12_vs_b_laneq_ab_median_throughput() {
        use microkernel::{NeonBLaneqKernel, NeonKernel};
        use std::time::Instant;

        fn run_once<K: Microkernel>(
            kernel: K,
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k_dim: usize,
        ) -> f64 {
            let mut c = vec![0.0f32; m * n];
            let start = Instant::now();
            gemm_blis_with_kernel(kernel, a, b, &mut c, m, n, k_dim).unwrap();
            start.elapsed().as_secs_f64()
        }

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        for dim in [512usize, 1024, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0x9999_9999, m * k);
            let b = xorshift32_vec(0xAAAA_AAAA, k * n);

            // ウォームアップ（キャッシュ・TLB を双方カーネルで温める。
            // `neon_8x12_vs_12x8_ab_median_throughput` と同じ理由）。
            run_once(NeonKernel, &a, &b, m, n, k);
            run_once(NeonBLaneqKernel, &a, &b, m, n, k);

            // 計測順序を交互化し系統的偏りを避ける（同上）。
            let mut samples_a_lane = Vec::with_capacity(5);
            let mut samples_b_lane = Vec::with_capacity(5);
            for i in 0..5 {
                if i % 2 == 0 {
                    samples_a_lane.push(run_once(NeonKernel, &a, &b, m, n, k));
                    samples_b_lane.push(run_once(NeonBLaneqKernel, &a, &b, m, n, k));
                } else {
                    samples_b_lane.push(run_once(NeonBLaneqKernel, &a, &b, m, n, k));
                    samples_a_lane.push(run_once(NeonKernel, &a, &b, m, n, k));
                }
            }

            let median_a_lane = median(samples_a_lane);
            let median_b_lane = median(samples_b_lane);

            println!(
                "dim={dim}: NeonKernel(A-lane) median={median_a_lane:.6}s, \
                 NeonBLaneqKernel(B-lane) median={median_b_lane:.6}s"
            );
        }
    }

    /// MC/KC/NC 実機スイープ（イシュー #564）: 参照実装値（BLIS
    /// firestorm: MC=480/KC=4096/NC=9600・OpenBLAS: NC 相当 4096）近傍を
    /// 含む候補グリッド × REQ-8 判定形状で [`gemm_blis_parallel_with_blocks`]
    /// （実運用経路 `gemm_blis_parallel` 相当）の中央値スループットを
    /// 計測・報告する（`.claude/rules/coding-rust.md` の 5 回計測中央値
    /// 規約。計測順は [`neon_8x12_vs_12x8_ab_median_throughput`] と同じ
    /// 理由でインターリーブし cache/TLB の系統的偏りを避ける）。
    ///
    /// 対象実機は **Apple M4 Max**（firestorm 系。#481 §3 確定・
    /// `docs/perf/gemm-optimization-baseline.md`）。候補グリッド・選定
    /// 判断基準・実測結果の記録先は `docs/perf/cpu-gemm-blocking-sweep.md`
    /// を参照。採用可否の判定は本テストの出力を見た人間／後続セッションが
    /// 行うため、本テスト自体は勝敗を assert しない（#559 §2.3 と同方針）。
    ///
    /// `#[ignore]` 分離（実機実行専用。`cargo test -p fandhe-ai-backend-cpu --release
    /// -- --ignored mc_kc_nc_blocking_sweep_median_throughput` で M4 Max
    /// 上から実行する）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "aarch64 実機（Apple M4 Max）での MC/KC/NC スイープ計測専用（--release 実行推奨。#564）"]
    fn mc_kc_nc_blocking_sweep_median_throughput() {
        use std::time::Instant;

        // §3.4 候補グリッド（現行値・軸別分離・firestorm 参照値そのまま・
        // 中間点）。MC は NEON 既定マイクロカーネル MR=8（#559）の倍数。
        const CANDIDATES: [(&str, BlockSizes); 8] = [
            (
                "現行値",
                BlockSizes {
                    mc: 128,
                    kc: 256,
                    nc: 512,
                },
            ),
            (
                "NC拡大(OpenBLAS相当)",
                BlockSizes {
                    mc: 128,
                    kc: 256,
                    nc: 4096,
                },
            ),
            (
                "NC拡大(firestorm)",
                BlockSizes {
                    mc: 128,
                    kc: 256,
                    nc: 9600,
                },
            ),
            (
                "KC拡大(firestorm)",
                BlockSizes {
                    mc: 128,
                    kc: 4096,
                    nc: 512,
                },
            ),
            (
                "MC拡大(firestorm)",
                BlockSizes {
                    mc: 480,
                    kc: 256,
                    nc: 512,
                },
            ),
            (
                "firestorm全軸",
                BlockSizes {
                    mc: 480,
                    kc: 4096,
                    nc: 9600,
                },
            ),
            (
                "中間点",
                BlockSizes {
                    mc: 256,
                    kc: 1024,
                    nc: 4096,
                },
            ),
            (
                "firestormMC/KC+OpenBLAS-NC",
                BlockSizes {
                    mc: 480,
                    kc: 4096,
                    nc: 4096,
                },
            ),
        ];

        fn run_once(
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k_dim: usize,
            blocks: BlockSizes,
        ) -> f64 {
            let mut c = vec![0.0f32; m * n];
            let start = Instant::now();
            gemm_blis_parallel_with_blocks(a, b, &mut c, m, n, k_dim, blocks).unwrap();
            start.elapsed().as_secs_f64()
        }

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        // REQ-8 判定形状（M=N=K=2048/4096）+ 参考 1024（計画 §3.4）。
        for dim in [1024usize, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xbbbb_bbbb, m * k);
            let b = xorshift32_vec(0xcccc_cccc, k * n);

            // 全候補を 1 巡ウォームアップしてから計測に入る（cache/TLB を
            // 均等に温める。単純な「先頭候補が cold cache を引く」偏りを
            // 避ける）。
            for &(_, blocks) in &CANDIDATES {
                run_once(&a, &b, m, n, k, blocks);
            }

            // 各反復で候補の計測順序をローテーションし、特定候補が常に
            // 先頭/末尾になることによる系統的偏りを避ける
            // （[`neon_8x12_vs_12x8_ab_median_throughput`] の A/B
            // インターリーブと同じ狙いを候補数 N へ一般化）。
            let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(5); CANDIDATES.len()];
            for rep in 0..5 {
                for offset in 0..CANDIDATES.len() {
                    let idx = (offset + rep) % CANDIDATES.len();
                    let (_, blocks) = CANDIDATES[idx];
                    samples[idx].push(run_once(&a, &b, m, n, k, blocks));
                }
            }

            println!("=== dim={dim} ===");
            for (i, &(label, blocks)) in CANDIDATES.iter().enumerate() {
                let med = median(samples[i].clone());
                println!(
                    "  {label} (mc={},kc={},nc={}): median={med:.6}s",
                    blocks.mc, blocks.kc, blocks.nc
                );
            }
        }
    }

    /// #753 実装計画 §3.3: `default_blocks()`（固定値）・
    /// `cache_params::detected_blocks()`（実行時キャッシュ検出）・
    /// `gemm_blis_parallel_2d_with_blocks`（2 次元タイルジョブ分配）の
    /// 3 経路を dim ∈ {512, 1024, 2048, 4096} で 5 回計測の中央値により
    /// A/B 計測する（`.claude/rules/coding-rust.md` の 5 回計測中央値
    /// 規約・[`mc_kc_nc_blocking_sweep_median_throughput`] と同じ計測順
    /// インターリーブ方針）。
    ///
    /// 受け入れ条件 2（実機 5 回中央値での非劣化・gemm crate との差の
    /// 縮小または逆転）の判定自体は本テストの範囲外（本テストは計測結果を
    /// 標準出力へ記録するのみで勝敗を assert しない。`.claude/rules/coding-rust.md`
    /// の「実機依存テストは `#[ignore]` で分離」・#564 の
    /// `mc_kc_nc_blocking_sweep_median_throughput` と同方針）。
    ///
    /// `#[ignore]`（実機実行専用。`cargo test -p fandhe-ai-backend-cpu --release --
    /// --ignored runtime_cache_detect_and_2d_partition_ab_median_throughput`
    /// で個別実行する想定。非 macOS・非実機環境でも `--release` なしで
    /// フォールバック経路のスモークとして完走することは
    /// `cache_params::tests::detected_blocks_returns_valid_block_sizes_on_any_platform`
    /// が別途保証する）。
    #[test]
    #[ignore = "実機（対象は Apple M4 Max。#753）での A/B 計測専用。--release 推奨"]
    fn runtime_cache_detect_and_2d_partition_ab_median_throughput() {
        use std::time::Instant;

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        // #753 レビュー指摘（Bugbot／codex-review）: 本ハーネスは実機
        // （Apple M4 Max）実行時の性能ゲート判断材料のため、
        // `dispatch_region`（aarch64 版）が実際に選ぶ `NeonKernel`
        // （MR=8・NR=12。#559）の値で `detected_blocks()` を評価する
        // 必要がある。従来 `ScalarKernel` の MR/NR（4×4）を渡していたが、
        // これは実行される `run_detected`／`run_2d`（いずれも
        // `gemm_blis_parallel_*_with_blocks` 経由で aarch64 上は常に
        // `NeonKernel` を実行する）のカーネル形状と乖離しており、
        // KC = (L1/2)/(4*(mr+nr)) の算出結果が実カーネルに対して最適化
        // されない値になっていた（本番 3 公開関数の既定 `default_blocks()`
        // は MR/NR に依存しない固定定数〈本ファイル冒頭ドキュメント参照〉
        // のため無関係）。他 arch でも `cargo test`（`--ignored` を渡さない
        // 通常実行時含む）が本関数をコンパイル対象に含むため、
        // `NeonKernel` 型自体が存在しない非 aarch64 では `ScalarKernel` の
        // 値へフォールバックする（本ハーネス自体が Apple M4 Max 実機専用
        // のため非 aarch64 での実行は想定外＝フォールバック値は計測結果に
        // 影響しない）。
        #[cfg(target_arch = "aarch64")]
        let (mr, nr) = (microkernel::NeonKernel::MR, microkernel::NeonKernel::NR);
        #[cfg(not(target_arch = "aarch64"))]
        let (mr, nr) = (ScalarKernel::MR, ScalarKernel::NR);
        let detected = cache_params::detected_blocks(mr, nr);
        println!(
            "detected_blocks: mc={} kc={} nc={} (default: mc={} kc={} nc={})",
            detected.mc, detected.kc, detected.nc, MC, KC, NC
        );

        for dim in [512usize, 1024, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0x9c9c_9c9c, m * k);
            let b = xorshift32_vec(0xadad_adad, k * n);

            fn run_default(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> f64 {
                let mut c = vec![0.0f32; m * n];
                let start = Instant::now();
                gemm_blis_parallel(a, b, &mut c, m, n, k).unwrap();
                start.elapsed().as_secs_f64()
            }
            // #753 レビュー指摘（codex-review P2）: `gemm_blis_parallel_with_blocks`
            // は実タスク数が 2 以上のとき常に `dispatch_shared_b`（B パネル
            // 共有経路。#750・本番未採用）を経由するため、これを A/B 計測に
            // 使うと「キャッシュ検出由来の MC/KC/NC 変更」に「B packing
            // 戦略の変更」が混入し、受け入れ条件 2（実機 5 回中央値での
            // 非劣化確認）の判断材料として解釈できなくなる。本関数は
            // `gemm_blis_parallel`（本番公開入口）と同じ分配（`panel_rows`
            // 静的パネル分割）・同じ dispatch（`dispatch_region`。B パネル
            // 共有なし）を用い、`blocks` だけを差し替えることで、比較対象を
            // 「キャッシュ検出の効果」のみに限定する。
            fn run_detected(
                a: &[f32],
                b: &[f32],
                m: usize,
                n: usize,
                k: usize,
                blocks: BlockSizes,
            ) -> f64 {
                let mut c = vec![0.0f32; m * n];
                let start = Instant::now();
                let num_threads =
                    crate::thread_limit::effective_num_threads(rayon::current_num_threads());
                let panel_rows = m.div_ceil(num_threads).max(1);
                c.par_chunks_mut(panel_rows * n)
                    .enumerate()
                    .try_for_each(|(panel_idx, c_chunk)| {
                        let row_start = panel_idx * panel_rows;
                        let row_end = (row_start + c_chunk.len() / n).min(m);
                        dispatch_region(
                            a,
                            b,
                            c_chunk,
                            n,
                            k,
                            row_start..row_end,
                            blocks,
                            GemmTranspose::Nn,
                        )
                    })
                    .unwrap();
                start.elapsed().as_secs_f64()
            }
            fn run_2d(
                a: &[f32],
                b: &[f32],
                m: usize,
                n: usize,
                k: usize,
                blocks: BlockSizes,
            ) -> f64 {
                let mut c = vec![0.0f32; m * n];
                let start = Instant::now();
                gemm_blis_parallel_2d_with_blocks(a, b, &mut c, m, n, k, blocks).unwrap();
                start.elapsed().as_secs_f64()
            }

            // 全経路を 1 巡ウォームアップ（cache/TLB を均等に温める。
            // `mc_kc_nc_blocking_sweep_median_throughput` と同じ狙い）。
            run_default(&a, &b, m, n, k);
            run_detected(&a, &b, m, n, k, detected);
            run_2d(&a, &b, m, n, k, default_blocks());

            let mut samples_default = Vec::with_capacity(5);
            let mut samples_detected = Vec::with_capacity(5);
            let mut samples_2d = Vec::with_capacity(5);
            for i in 0..5 {
                // 計測順序を交互化し系統的偏りを避ける（同上）。
                match i % 3 {
                    0 => {
                        samples_default.push(run_default(&a, &b, m, n, k));
                        samples_detected.push(run_detected(&a, &b, m, n, k, detected));
                        samples_2d.push(run_2d(&a, &b, m, n, k, default_blocks()));
                    }
                    1 => {
                        samples_detected.push(run_detected(&a, &b, m, n, k, detected));
                        samples_2d.push(run_2d(&a, &b, m, n, k, default_blocks()));
                        samples_default.push(run_default(&a, &b, m, n, k));
                    }
                    _ => {
                        samples_2d.push(run_2d(&a, &b, m, n, k, default_blocks()));
                        samples_default.push(run_default(&a, &b, m, n, k));
                        samples_detected.push(run_detected(&a, &b, m, n, k, detected));
                    }
                }
            }

            println!(
                "dim={dim}: default median={:.6}s / detected median={:.6}s / \
                 2d-partition median={:.6}s",
                median(samples_default),
                median(samples_detected),
                median(samples_2d),
            );
        }
    }

    // --- pc 外側ループ・A 1 回 pack 候補（イシュー #1041） ---

    /// [`gemm_blis_shared_b_pc_outer_region`] が多数の (pc,jc) 同期点を
    /// 跨いでも直列経路（[`gemm_blis_with_kernel_and_blocks`]）と bit
    /// 完全一致することを、小さい `BlockSizes`（既定値より大幅に小さく
    /// 多数の pc/jc 反復を強制する）で検証する（#750 の
    /// `gemm_blis_shared_b_region_multi_sync_point_matches_serial_bit_exact`
    /// と同型のテスト構成）。固定 4 スレッドプールで `m > panel_rows`
    /// （実タスク数 >= 2）を確定させ、pc 外側の共有 A・共有 B 経路を
    /// 確実に通す。
    #[test]
    fn gemm_blis_shared_b_pc_outer_multi_sync_point_matches_serial_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let blocks = BlockSizes {
            mc: 16,
            kc: 17,
            nc: 19,
        };
        let a = xorshift32_vec(0x7a7a_7a7a, m * k);
        let b = xorshift32_vec(0x8b8b_8b8b, k * n);

        let mut c_serial = vec![0.0f32; m * n];
        gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c_serial, m, n, k, blocks)
            .unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap_or_else(|e| panic!("4 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_pc_outer = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_variant(
                GemmDriverVariant::SharedBPcOuter,
                &a,
                &b,
                &mut c_pc_outer,
                m,
                n,
                k,
                blocks,
            )
            .unwrap()
        });

        assert_eq!(
            c_serial, c_pc_outer,
            "多 (pc,jc) 同期点を跨ぐ pc 外側・A 1 回 pack 経路は直列経路と \
             bit 完全一致するはず（#1041）"
        );
    }

    /// イシュー #1041: 実タスク数 Q が rayon の稼働スレッド数 T を下回る
    /// 形状でも [`gemm_blis_shared_b_pc_outer_region`] が `gemm_naive` と
    /// bit 完全一致することを確認する（#750 の
    /// `gemm_blis_parallel_matches_naive_bit_exact_when_tasks_fewer_than_threads`
    /// と同型。`a_bufs`／`c.par_chunks_mut` の長さがずれないことの回帰）。
    #[test]
    fn gemm_blis_shared_b_pc_outer_matches_naive_bit_exact_when_tasks_fewer_than_threads() {
        let (m, n, k) = (10, 130, 40);
        let a = xorshift32_vec(0x9c9c_9c9c, m * k);
        let b = xorshift32_vec(0xadad_adad, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap_or_else(|e| panic!("16 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_pc_outer = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_variant(
                GemmDriverVariant::SharedBPcOuter,
                &a,
                &b,
                &mut c_pc_outer,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap()
        });

        assert_eq!(
            c_naive, c_pc_outer,
            "m={m} 行を num_threads=16 で分割する実タスク数 Q < T のケースは \
             gemm_naive と bit 完全一致するはず（#1041）"
        );
    }

    /// イシュー #1041（#1366 で `IcDynamic` を追加）: [`GemmDriverVariant`]
    /// の全候補（`RowPanel`・`SharedB`・`SharedBPcOuter`・`IcDynamic`）が
    /// MC/KC/NC 境界を跨ぐ複数形状・複数スレッド数の組で `gemm_naive` と
    /// bit 完全一致することを網羅的に検証する（A/B 一括計測ハーネスが
    /// 候補間で数値的に等価な結果を比較することの前提を担保する）。
    #[test]
    fn gemm_blis_parallel_variant_all_candidates_match_naive_bit_exact() {
        let shapes: &[(usize, usize, usize)] = &[
            (1, 64, 64),     // m == 1（gemv 専用経路）
            (5, 7, 3),       // 極小・端タイルのみ
            (64, 64, 64),    // MC 未満（診断表の N=1024 相当パターン）
            (128, 128, 96),  // MC ちょうど（診断表の N=2048 相当パターン）
            (200, 600, 700), // MC/KC/NC 境界を跨ぐ非正方
        ];
        let thread_counts = [1usize, 2, 3, 16];

        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0xbebe_bebe ^ (m as u32), m * k);
            let b = xorshift32_vec(0xcfcf_cfcf ^ (n as u32), k * n);

            let mut c_naive = vec![0.0f32; m * n];
            crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

            for &num_threads in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}")
                    });

                for variant in all_gemm_driver_variants() {
                    let mut c = vec![0.0f32; m * n];
                    pool.install(|| {
                        gemm_blis_parallel_variant(
                            variant,
                            &a,
                            &b,
                            &mut c,
                            m,
                            n,
                            k,
                            default_blocks(),
                        )
                        .unwrap()
                    });
                    assert_eq!(
                        c_naive, c,
                        "variant={variant:?} shape=({m},{n},{k}) num_threads={num_threads} は \
                         gemm_naive と bit 完全一致するはず（#1041・#1366）"
                    );
                }
            }
        }
    }

    /// イシュー #1587: `GemmDriverVariant::TwoDDynamicSme`（`fmopa`
    /// マイクロカーネル強制）が本番入口（`gemm_blis_two_d_dynamic_region`
    /// 経由）で `gemm_naive` と bit 完全一致することを検証する。
    /// `all_gemm_driver_variants` には含めない（SME 非対応環境で
    /// `.expect` panic するため）ため、`SmeKernel::try_new()` で実行時
    /// スキップする独立テストとする（`microkernel::sme::tests` の単体
    /// テストが個別カーネル呼び出しを検証するのに対し、本テストは
    /// [`SME_MIN_M`]／[`SME_MIN_N`]／[`SME_MIN_K`] のしきい値を跨ぐ
    /// 形状・端タイル・複数スレッド数を通じた 5-loop ドライバ全体の
    /// bit 完全一致を検証する）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。 \
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored \
                gemm_blis_parallel_variant_sme_matches_naive_bit_exact_when_available \
                --nocapture）"]
    fn gemm_blis_parallel_variant_sme_matches_naive_bit_exact_when_available() {
        if microkernel::SmeKernel::try_new().is_none() {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        }
        let shapes: &[(usize, usize, usize)] = &[
            (SME_MIN_M, SME_MIN_N, SME_MIN_K),             // しきい値ちょうど
            (SME_MIN_M + 5, SME_MIN_N + 3, SME_MIN_K + 7), // 端タイルあり
            (SME_MIN_M * 2, SME_MIN_N * 2, SME_MIN_K * 3), // MC/KC/NC 境界を跨ぐ
        ];
        let thread_counts = [1usize, 2, 4];

        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0xa5a5_a5a5 ^ (m as u32), m * k);
            let b = xorshift32_vec(0x5a5a_5a5a ^ (n as u32), k * n);

            let mut c_naive = vec![0.0f32; m * n];
            crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

            for &num_threads in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}")
                    });

                let mut c = vec![0.0f32; m * n];
                pool.install(|| {
                    gemm_blis_parallel_variant(
                        GemmDriverVariant::TwoDDynamicSme,
                        &a,
                        &b,
                        &mut c,
                        m,
                        n,
                        k,
                        default_blocks(),
                    )
                    .unwrap()
                });
                assert_eq!(
                    c_naive, c,
                    "TwoDDynamicSme shape=({m},{n},{k}) num_threads={num_threads} は \
                     gemm_naive と bit 完全一致するはず（#1587）"
                );
            }
        }
    }

    /// イシュー #1587: [`sme_shape_eligible`] の境界値挙動（純関数の
    /// 単体テスト。実行環境の SME 対応有無に依らず実行できる）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sme_shape_eligible_boundary_values() {
        assert!(sme_shape_eligible(SME_MIN_M, SME_MIN_N, SME_MIN_K));
        assert!(!sme_shape_eligible(SME_MIN_M - 1, SME_MIN_N, SME_MIN_K));
        assert!(!sme_shape_eligible(SME_MIN_M, SME_MIN_N - 1, SME_MIN_K));
        assert!(!sme_shape_eligible(SME_MIN_M, SME_MIN_N, SME_MIN_K - 1));
        assert!(sme_shape_eligible(
            SME_MIN_M * 4,
            SME_MIN_N * 4,
            SME_MIN_K * 4
        ));
    }

    /// イシュー #1587: [`SME_PRODUCTION_ENABLED`] が `false` の間は
    /// 本番入口（`dispatch_two_d_dynamic` 経由）が実行 CPU の SME 対応
    /// 有無に関わらず常に [`microkernel::NeonKernel`] を選ぶ（#1313 以前
    /// と bit 完全一致）ことを、大きめの形状（しきい値を超える）で確認
    /// する回帰テスト。`SME_PRODUCTION_ENABLED` が誤って `true` へ変更
    /// された場合、この形状は本テストではなく
    /// `gemm_blis_parallel_variant_sme_matches_naive_bit_exact_when_available`
    /// 側の bit 一致契約でカバーされるため、本テストは
    /// `SME_PRODUCTION_ENABLED` の現在値が `false` であることそのものを
    /// 検証する（値のドリフト検出）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn sme_production_enabled_is_false_pending_measurement() {
        assert!(
            !SME_PRODUCTION_ENABLED,
            "R1〜R4（イシュー #1587 事前登録規則）の実測完了・ADOPT 確定まで \
             SME_PRODUCTION_ENABLED は false を維持する契約"
        );
    }

    /// イシュー #1366: `IcDynamic` が小さい **kc** で pc 同期点を複数
    /// 強制する形状（`gemm_blis_shared_b_pc_outer_multi_sync_point_matches_serial_bit_exact`
    /// と同型の意図）でも直列実装（`ScalarKernel` 経由の
    /// `gemm_blis_with_kernel_and_blocks`）と bit 完全一致することを
    /// 検証する。`mc=16` により [`ic_dynamic_panel_rows`] が 16 へ
    /// クランプされ、パネル数（200.div_ceil(16) = 13）がスレッド数 4 を
    /// 上回るため動的配布（`AtomicUsize::fetch_add` の複数回呼び出し・
    /// worker の複数パネル claim）を確実に通す。`nc` は本 variant が
    /// jc ループを持たないため使われないことも合わせて確認する
    /// （`ic_dynamic_b_capacity` ドキュメント参照）。
    #[test]
    fn gemm_blis_ic_dynamic_multi_pc_sync_point_matches_serial_bit_exact() {
        let (m, n, k) = (200usize, 600usize, 700usize);
        let blocks = BlockSizes {
            mc: 16,
            kc: 17,
            nc: 19,
        };
        let a = xorshift32_vec(0x1111_2222, m * k);
        let b = xorshift32_vec(0x3333_4444, k * n);

        let mut c_serial = vec![0.0f32; m * n];
        gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c_serial, m, n, k, blocks)
            .unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap_or_else(|e| panic!("4 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_dynamic = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_variant(
                GemmDriverVariant::IcDynamic,
                &a,
                &b,
                &mut c_dynamic,
                m,
                n,
                k,
                blocks,
            )
            .unwrap()
        });

        assert_eq!(
            c_serial, c_dynamic,
            "小さい kc で pc 同期点を複数強制する形状は直列実装と bit 完全一致するはず（#1366）"
        );
    }

    /// イシュー #1366: 実タスク数（パネル数）がワーカー数を大きく下回る
    /// ケース（[`ic_dynamic_panel_rows`] が MR の倍数へ整列した最小値へ
    /// クランプされる）で、空振りする worker（`AtomicUsize::fetch_add` の
    /// 結果が `num_panels` を即座に超える worker）が正しく `Ok(())` で
    /// 終了し、他 worker の処理結果が `gemm_naive` と bit 完全一致する
    /// ことを検証する（`gemm_blis_shared_b_pc_outer_matches_naive_bit_exact_when_tasks_fewer_than_threads`
    /// と同型の回帰）。
    #[test]
    fn gemm_blis_ic_dynamic_matches_naive_bit_exact_when_tasks_fewer_than_threads() {
        let (m, n, k) = (10, 130, 40);
        let a = xorshift32_vec(0x5555_6666, m * k);
        let b = xorshift32_vec(0x7777_8888, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap_or_else(|e| panic!("16 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_dynamic = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_variant(
                GemmDriverVariant::IcDynamic,
                &a,
                &b,
                &mut c_dynamic,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap()
        });

        assert_eq!(
            c_naive, c_dynamic,
            "m={m} 行を num_threads=16 で分割する実タスク数 Q < T のケースは \
             gemm_naive と bit 完全一致するはず（#1366）"
        );
    }

    /// イシュー #1366: `IcDynamic` が `RowPanel` と bit 完全一致すること
    /// を、MC/KC/NC 境界を跨ぐ複数形状（`m` がパネルで割り切れない・
    /// `n % NR != 0`・`k % KC != 0`・`k == 0` の no-op ケースを含む）×
    /// 複数スレッド数で直接検証する（issue 表題「RowPanel との bit 完全
    /// 一致を回帰テストで確認する」の直接検証）。各形状で C の初期値を
    /// 非ゼロ乱数にし、累積の意味論（既存 C 値への `f32::mul_add` 累積）
    /// が両 variant で揃うことも確認する。
    #[test]
    fn gemm_blis_ic_dynamic_matches_row_panel_bit_exact_across_shapes_and_threads() {
        let shapes: &[(usize, usize, usize)] = &[
            (5, 7, 3),
            (64, 64, 64),
            (128, 128, 96),
            (129, 130, 257),
            (1000, 96, 300),
            (523, 600, 700),
            (2, 3, 0),
            (512, 512, 512),
        ];
        let thread_counts = [1usize, 2, 3, 16];

        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0xaaaa_1111 ^ (m as u32), m * k);
            let b = xorshift32_vec(0xbbbb_2222 ^ (n as u32), k * n);
            let c_init = xorshift32_vec(0xcccc_3333 ^ (k as u32), m * n);

            for &num_threads in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}")
                    });

                let mut c_row_panel = c_init.clone();
                pool.install(|| {
                    gemm_blis_parallel_variant(
                        GemmDriverVariant::RowPanel,
                        &a,
                        &b,
                        &mut c_row_panel,
                        m,
                        n,
                        k,
                        default_blocks(),
                    )
                    .unwrap()
                });

                let mut c_ic_dynamic = c_init.clone();
                pool.install(|| {
                    gemm_blis_parallel_variant(
                        GemmDriverVariant::IcDynamic,
                        &a,
                        &b,
                        &mut c_ic_dynamic,
                        m,
                        n,
                        k,
                        default_blocks(),
                    )
                    .unwrap()
                });

                assert_eq!(
                    c_row_panel, c_ic_dynamic,
                    "shape=({m},{n},{k}) num_threads={num_threads} は IcDynamic と RowPanel が \
                     bit 完全一致するはず（#1366）"
                );
            }
        }
    }

    /// イシュー #1366: 大形状（1024/2048/4096 正方）でも `IcDynamic` が
    /// `RowPanel` と bit 完全一致することを実機で確認する（release
    /// ビルド・プール既定スレッド数）。デバッグビルドでの計算量が大きい
    /// ため通常 CI では実行しない（#1367 の実機実測に先立ち 1 回実行する
    /// ことを想定）。
    #[test]
    #[ignore = "実機（M4 Max / GB10）での大形状 bit 一致確認専用（#1366。cargo test \
                -p fandhe-ai-backend-cpu --release -- --ignored \
                gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large --nocapture）"]
    fn gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large() {
        for &dim in &[1024usize, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xdddd_4444 ^ (dim as u32), m * k);
            let b = xorshift32_vec(0xeeee_5555 ^ (dim as u32), k * n);

            let mut c_row_panel = vec![0.0f32; m * n];
            gemm_blis_parallel_variant(
                GemmDriverVariant::RowPanel,
                &a,
                &b,
                &mut c_row_panel,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap();

            let mut c_ic_dynamic = vec![0.0f32; m * n];
            gemm_blis_parallel_variant(
                GemmDriverVariant::IcDynamic,
                &a,
                &b,
                &mut c_ic_dynamic,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap();

            assert_eq!(
                c_row_panel, c_ic_dynamic,
                "dim={dim} は IcDynamic と RowPanel が bit 完全一致するはず（#1366）"
            );
        }
    }

    /// イシュー #1311: `TwoDDynamic`（(mc, nc) 2D job 動的分配）が
    /// `RowPanel`（本番既定）と bit 完全一致することを、C 初期値
    /// 非ゼロ乱数・端あり形状（`m`/`n` が MR/NR 非倍数・`m<mr`・`n<nr`・
    /// `k==0`・`k<kc`・非正方・512³）× スレッド数 1/2/3/16 ×
    /// `jobs_per_worker` {1,2,4,8} で直接検証する（設計 §10・
    /// `gemm_blis_ic_dynamic_matches_row_panel_bit_exact_across_shapes_and_threads`
    /// と同型。debug ビルドの所要時間を抑えるため `RowPanel` 参照値は
    /// (形状, スレッド数) ごとに 1 回だけ計算し `jobs_per_worker`
    /// スイープで再利用する）。
    #[test]
    fn gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_across_shapes_and_threads() {
        let shapes: &[(usize, usize, usize)] = &[
            (5, 7, 3),
            (64, 64, 64),
            (128, 128, 96),
            (129, 130, 257),
            (1000, 96, 300),
            (523, 600, 700),
            (2, 3, 0),
            (7, 5, 64),
            (64, 5, 64),
            (300, 300, 100),
            (512, 512, 512),
        ];
        let thread_counts = [1usize, 2, 3, 16];
        let jobs_per_worker_values = [1usize, 2, 4, 8];

        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0xaaaa_1111 ^ (m as u32), m * k);
            let b = xorshift32_vec(0xbbbb_2222 ^ (n as u32), k * n);
            let c_init = xorshift32_vec(0xcccc_3333 ^ (k as u32), m * n);

            for &num_threads in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}")
                    });

                let mut c_row_panel = c_init.clone();
                pool.install(|| {
                    gemm_blis_parallel_variant(
                        GemmDriverVariant::RowPanel,
                        &a,
                        &b,
                        &mut c_row_panel,
                        m,
                        n,
                        k,
                        default_blocks(),
                    )
                    .unwrap()
                });

                for &jobs_per_worker in &jobs_per_worker_values {
                    let mut c_two_d = c_init.clone();
                    pool.install(|| {
                        gemm_blis_parallel_two_d_dynamic_with_params(
                            &a,
                            &b,
                            &mut c_two_d,
                            m,
                            n,
                            k,
                            default_blocks(),
                            jobs_per_worker,
                            GemmTranspose::Nn,
                        )
                        .unwrap()
                    });

                    assert_eq!(
                        c_row_panel, c_two_d,
                        "shape=({m},{n},{k}) num_threads={num_threads} \
                         jobs_per_worker={jobs_per_worker} は TwoDDynamic と RowPanel が \
                         bit 完全一致するはず（#1311）"
                    );
                }
            }
        }
    }

    /// イシュー #1311: 小 `kc`／`mc`／`nc`（`BlockSizes { mc: 16, kc: 8,
    /// nc: 24 }`）で job 内 jc/pc/ic ループが複数回通ることを固定し、
    /// `RowPanel`（同一小ブロックサイズ）・直列 [`gemm_blis`] と bit
    /// 完全一致することを検証する（設計 §10）。
    #[test]
    fn gemm_blis_two_d_dynamic_multi_pc_matches_serial_bit_exact() {
        let small_blocks = BlockSizes {
            mc: 16,
            kc: 8,
            nc: 24,
        };
        let shapes: &[(usize, usize, usize)] = &[(64, 72, 40), (100, 50, 33), (33, 100, 17)];
        let thread_counts = [1usize, 4, 8];

        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0x1111_aaaa ^ (m as u32), m * k);
            let b = xorshift32_vec(0x2222_bbbb ^ (n as u32), k * n);

            let mut c_serial = vec![0.0f32; m * n];
            gemm_blis_with_kernel_and_blocks(
                ScalarKernel,
                &a,
                &b,
                &mut c_serial,
                m,
                n,
                k,
                small_blocks,
            )
            .unwrap();

            for &num_threads in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}")
                    });

                let mut c_row_panel = vec![0.0f32; m * n];
                pool.install(|| {
                    gemm_blis_parallel_variant(
                        GemmDriverVariant::RowPanel,
                        &a,
                        &b,
                        &mut c_row_panel,
                        m,
                        n,
                        k,
                        small_blocks,
                    )
                    .unwrap()
                });
                assert_eq!(
                    c_serial, c_row_panel,
                    "shape=({m},{n},{k}) T={num_threads}: RowPanel は直列 gemm_blis と \
                     bit 完全一致するはず（前提確認）"
                );

                let mut c_two_d = vec![0.0f32; m * n];
                pool.install(|| {
                    gemm_blis_parallel_two_d_dynamic_with_params(
                        &a,
                        &b,
                        &mut c_two_d,
                        m,
                        n,
                        k,
                        small_blocks,
                        2,
                        GemmTranspose::Nn,
                    )
                    .unwrap()
                });

                assert_eq!(
                    c_serial, c_two_d,
                    "shape=({m},{n},{k}) T={num_threads}: TwoDDynamic は小ブロックサイズ \
                     （複数 pc/jc/ic 反復）でも直列 gemm_blis と bit 完全一致するはず（#1311）"
                );
            }
        }
    }

    /// イシュー #1311: `TwoDDynamic` が同一入力・同一プール（スレッド数
    /// 3・16）で 2 回実行しても bit 同一であることを検証する（job の
    /// 分配順序・rayon work stealing のスケジューリングに依存しない
    /// ことの直接確認。設計 §10）。
    #[test]
    fn gemm_blis_two_d_dynamic_is_deterministic_across_runs() {
        let (m, n, k) = (523usize, 611usize, 400usize);
        let a = xorshift32_vec(0x3333_cccc, m * k);
        let b = xorshift32_vec(0x4444_dddd, k * n);
        let c_init = xorshift32_vec(0x5555_eeee, m * n);

        for &num_threads in &[3usize, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap_or_else(|e| panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}"));

            let mut c_run1 = c_init.clone();
            pool.install(|| {
                gemm_blis_parallel_two_d_dynamic_with_params(
                    &a,
                    &b,
                    &mut c_run1,
                    m,
                    n,
                    k,
                    default_blocks(),
                    2,
                    GemmTranspose::Nn,
                )
                .unwrap()
            });

            let mut c_run2 = c_init.clone();
            pool.install(|| {
                gemm_blis_parallel_two_d_dynamic_with_params(
                    &a,
                    &b,
                    &mut c_run2,
                    m,
                    n,
                    k,
                    default_blocks(),
                    2,
                    GemmTranspose::Nn,
                )
                .unwrap()
            });

            assert_eq!(
                c_run1, c_run2,
                "num_threads={num_threads}: TwoDDynamic は同一入力の 2 回実行で \
                 bit 同一のはず（#1311）"
            );
        }
    }

    /// イシュー #1311: `TwoDDynamic` を `Nt`／`Tn` で実行した結果が、
    /// 本番 `RowPanel` 経路（[`gemm_blis_parallel_nt`]／
    /// [`gemm_blis_parallel_tn`]）と bit 完全一致することを検証する
    /// （#1313 結線対象の事前保証。設計 §10「転置（Nt/Tn）の bit 一致」）。
    #[test]
    fn gemm_blis_two_d_dynamic_transposed_matches_row_panel_bit_exact() {
        let shapes: &[(usize, usize, usize)] = &[(64, 64, 64), (129, 130, 97), (512, 512, 512)];
        let thread_counts = [1usize, 4, 16];

        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0x6666_1111 ^ (m as u32), m * k);
            let b = xorshift32_vec(0x7777_2222 ^ (n as u32), k * n);
            // bt: 論理形状 [n, k] 行優先（元の B [k, n] を転置した実体）。
            let bt = xorshift32_vec(0x8888_3333 ^ (k as u32), n * k);
            // at: 論理形状 [k, m] 行優先（元の A [m, k] を転置した実体）。
            let at = xorshift32_vec(0x9999_4444 ^ (m as u32), k * m);

            for &num_threads in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}")
                    });

                // Nt
                let mut c_prod_nt = vec![0.0f32; m * n];
                pool.install(|| gemm_blis_parallel_nt(&a, &bt, &mut c_prod_nt, m, n, k).unwrap());
                let mut c_two_d_nt = vec![0.0f32; m * n];
                pool.install(|| {
                    gemm_blis_parallel_two_d_dynamic_with_params(
                        &a,
                        &bt,
                        &mut c_two_d_nt,
                        m,
                        n,
                        k,
                        default_blocks(),
                        2,
                        GemmTranspose::Nt,
                    )
                    .unwrap()
                });
                assert_eq!(
                    c_prod_nt, c_two_d_nt,
                    "shape=({m},{n},{k}) T={num_threads} Nt: TwoDDynamic は \
                     gemm_blis_parallel_nt と bit 完全一致するはず（#1311）"
                );

                // Tn
                let mut c_prod_tn = vec![0.0f32; m * n];
                pool.install(|| gemm_blis_parallel_tn(&at, &b, &mut c_prod_tn, m, n, k).unwrap());
                let mut c_two_d_tn = vec![0.0f32; m * n];
                pool.install(|| {
                    gemm_blis_parallel_two_d_dynamic_with_params(
                        &at,
                        &b,
                        &mut c_two_d_tn,
                        m,
                        n,
                        k,
                        default_blocks(),
                        2,
                        GemmTranspose::Tn,
                    )
                    .unwrap()
                });
                assert_eq!(
                    c_prod_tn, c_two_d_tn,
                    "shape=({m},{n},{k}) T={num_threads} Tn: TwoDDynamic は \
                     gemm_blis_parallel_tn と bit 完全一致するはず（#1311）"
                );
            }
        }
    }

    /// イシュー #1311: 大形状（1024/2048/4096 正方）でも `TwoDDynamic` が
    /// `RowPanel` と bit 完全一致することを実機で確認する（release
    /// ビルド・プール既定スレッド数。`gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large`
    /// と同型）。デバッグビルドでの計算量が大きいため通常 CI では実行
    /// しない。
    #[test]
    #[ignore = "実機（M4 Max / GB10）での大形状 bit 一致確認専用（#1311。cargo test \
                -p fandhe-ai-backend-cpu --release -- --ignored \
                gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large --nocapture）"]
    fn gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large() {
        for &dim in &[1024usize, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xdddd_6666 ^ (dim as u32), m * k);
            let b = xorshift32_vec(0xeeee_7777 ^ (dim as u32), k * n);

            let mut c_row_panel = vec![0.0f32; m * n];
            gemm_blis_parallel_variant(
                GemmDriverVariant::RowPanel,
                &a,
                &b,
                &mut c_row_panel,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap();

            let mut c_two_d = vec![0.0f32; m * n];
            gemm_blis_parallel_two_d_dynamic_with_params(
                &a,
                &b,
                &mut c_two_d,
                m,
                n,
                k,
                default_blocks(),
                TWO_D_JOBS_PER_WORKER,
                GemmTranspose::Nn,
            )
            .unwrap();

            assert_eq!(
                c_row_panel, c_two_d,
                "dim={dim} は TwoDDynamic と RowPanel が bit 完全一致するはず（#1311）"
            );
        }
    }

    /// [`split_c_into_jobs`] が生成する各 job の `c_rows` が、C 全体を
    /// 過不足なく被覆し互いに素であることを、番兵値の書き込みで検証する
    /// （設計 §10「`split_c_into_jobs_rows_are_disjoint_and_cover_c`」）。
    #[test]
    fn split_c_into_jobs_rows_are_disjoint_and_cover_c() {
        let blocks = BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };
        let (m, n) = (257usize, 193usize);
        let grid = partition::job_grid(m, n, 8, 12, &blocks, 8, 2).unwrap();

        let mut c = vec![0.0f32; m * n];
        let jobs = split_c_into_jobs(&mut c, n, &grid);

        // job 数は row_bands * col_bands のはず。
        assert_eq!(jobs.len(), grid.row_bands * grid.col_bands);

        // 各 job について、番兵値（job インデックス+1）を書き込む。
        for (job_idx, job) in jobs.into_iter().enumerate() {
            let sentinel = (job_idx + 1) as f32;
            let mut job = job;
            for row in job.c_rows.iter_mut() {
                for v in row.iter_mut() {
                    *v = sentinel;
                }
            }
        }

        // C 全要素が「ちょうど 1 回」上書きされ、0.0（未上書き）が残って
        // いないことを確認する（被覆完全・互いに素の直接検証）。
        assert!(
            c.iter().all(|&v| v != 0.0),
            "split_c_into_jobs の job は C を過不足なく被覆するはず"
        );
    }

    /// イシュー #1317: `RowPanelBLaneqVec`（B 側 laneq ベクトル転置版
    /// マイクロカーネル）が `RowPanel`（本番既定）と bit 完全一致する
    /// ことを、MC/KC/NC 境界を跨ぐ複数形状（端タイル・`n % NR != 0`・
    /// `k % KC != 0`・`k == 0` no-op を含む）× 複数スレッド数で直接検証
    /// する（[`gemm_blis_ic_dynamic_matches_row_panel_bit_exact_across_shapes_and_threads`]
    /// と同一パターン）。両 variant の唯一の差分がマイクロカーネル（C タイル
    /// 転置方式）であることを、`gemm_blis_parallel_row_panel_with_kernel`
    /// が `RowPanel` と行パネル分割・並列化ロジックを共有することで保証
    /// する（§3.4）。端タイル（`ldc=NR` スタックバッファ経路）・完全タイル
    /// （`ldc=n` 直接経路）の両方を通す形状を含む。aarch64 限定
    /// （`RowPanelBLaneqVec` 自体が aarch64 限定 variant のため）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn gemm_blis_row_panel_b_laneq_vec_matches_row_panel_bit_exact_across_shapes_and_threads() {
        let shapes: &[(usize, usize, usize)] = &[
            (5, 7, 3),
            (64, 64, 64),
            (128, 128, 96),
            (129, 130, 257),
            (1000, 96, 300),
            (523, 600, 700),
            (2, 3, 0),
            (512, 512, 512),
        ];
        let thread_counts = [1usize, 2, 3, 16];

        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0xb1a2_1111 ^ (m as u32), m * k);
            let b = xorshift32_vec(0xb1a2_2222 ^ (n as u32), k * n);
            let c_init = xorshift32_vec(0xb1a2_3333 ^ (k as u32), m * n);

            for &num_threads in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}")
                    });

                let mut c_row_panel = c_init.clone();
                pool.install(|| {
                    gemm_blis_parallel_variant(
                        GemmDriverVariant::RowPanel,
                        &a,
                        &b,
                        &mut c_row_panel,
                        m,
                        n,
                        k,
                        default_blocks(),
                    )
                    .unwrap()
                });

                let mut c_b_laneq_vec = c_init.clone();
                pool.install(|| {
                    gemm_blis_parallel_variant(
                        GemmDriverVariant::RowPanelBLaneqVec,
                        &a,
                        &b,
                        &mut c_b_laneq_vec,
                        m,
                        n,
                        k,
                        default_blocks(),
                    )
                    .unwrap()
                });

                assert_eq!(
                    c_row_panel, c_b_laneq_vec,
                    "shape=({m},{n},{k}) num_threads={num_threads} は \
                     RowPanelBLaneqVec と RowPanel が bit 完全一致するはず（#1317）"
                );
            }
        }
    }

    /// イシュー #1317: 大形状（1024/2048/4096 正方）でも
    /// `RowPanelBLaneqVec` が `RowPanel` と bit 完全一致することを実機で
    /// 確認する（release ビルド・プール既定スレッド数。#1318 の実機実測
    /// に先立つ事前確認用。[`gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large`]
    /// と同一パターン）。デバッグビルドでの計算量が大きいため通常 CI
    /// では実行しない。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "実機（M4 Max / GB10）での大形状 bit 一致確認専用（#1317。cargo test \
                -p fandhe-ai-backend-cpu --release -- --ignored \
                gemm_blis_row_panel_b_laneq_vec_matches_row_panel_bit_exact_large --nocapture）"]
    fn gemm_blis_row_panel_b_laneq_vec_matches_row_panel_bit_exact_large() {
        for &dim in &[1024usize, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xb1a2_4444 ^ (dim as u32), m * k);
            let b = xorshift32_vec(0xb1a2_5555 ^ (dim as u32), k * n);

            let mut c_row_panel = vec![0.0f32; m * n];
            gemm_blis_parallel_variant(
                GemmDriverVariant::RowPanel,
                &a,
                &b,
                &mut c_row_panel,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap();

            let mut c_b_laneq_vec = vec![0.0f32; m * n];
            gemm_blis_parallel_variant(
                GemmDriverVariant::RowPanelBLaneqVec,
                &a,
                &b,
                &mut c_b_laneq_vec,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap();

            assert_eq!(
                c_row_panel, c_b_laneq_vec,
                "dim={dim} は RowPanelBLaneqVec と RowPanel が bit 完全一致するはず（#1317）"
            );
        }
    }

    /// [`ic_dynamic_panel_rows`] の純関数契約を検証する（イシュー #1366）。
    #[test]
    fn ic_dynamic_panel_rows_bounds_and_alignment() {
        // 戻り値は常に 1 以上（0 除算・無限ループを防ぐ下限。mc_total=0 は
        // 実際には呼び出し元で num_panels=0 になり本関数の戻り値自体は
        // 使われないが、下限契約は常に成立する）。
        assert!(ic_dynamic_panel_rows(0, 128, 8, 4) >= 1);
        assert_eq!(ic_dynamic_panel_rows(1, 128, 8, 16), 8);

        // 戻り値は常に mc 以下（`blocks.mc` を超えるパネルを作らない）。
        let rows = ic_dynamic_panel_rows(10_000, 128, 8, 2);
        assert!(rows <= 128, "rows={rows} は mc=128 以下のはず");

        // mc 未満のときは mr の倍数へ整列される。
        let rows = ic_dynamic_panel_rows(100, 128, 8, 4);
        assert_eq!(rows % 8, 0, "rows={rows} は mr=8 の倍数のはず");

        // mc_total が num_workers*mc 以上なら mc（`SharedBPcOuter` の
        // `panel_capacity` と同じ「MC でクランプ済み」帯へ揃う）。
        assert_eq!(ic_dynamic_panel_rows(10_000, 128, 8, 4), 128);

        // num_workers=1 は 1 パネルで全行を担当するため mc（既に整列済み
        // でない限り mc 自体へクランプ）へ張り付く。
        assert_eq!(ic_dynamic_panel_rows(300, 64, 8, 1), 64);
    }

    /// [`ic_dynamic_b_capacity`] の純関数契約を検証する（イシュー #1366）。
    #[test]
    fn ic_dynamic_b_capacity_matches_formula_and_detects_overflow() {
        // 代表値: n=100, nr=8 → nr_blocks=13, kc_len_max=17 → 13*17*8=1768。
        // `GemmError` は `#[non_exhaustive]`・`PartialEq` 非実装のため
        // `assert_eq!` ではなく `.unwrap()`／`matches!` で検証する
        // （`task_a_capacity` の既存単体テストと同じ理由。同種の
        // オーバーフロー検査を持つ他テストが存在しないためここが初出）。
        assert_eq!(ic_dynamic_b_capacity(100, 17, 8).unwrap(), 1768);

        // n が nr で割り切れる場合。
        assert_eq!(ic_dynamic_b_capacity(64, 32, 8).unwrap(), 64 * 32);

        // usize::MAX 近傍でのオーバーフロー検出（`task_a_capacity` と同型）。
        assert!(
            matches!(
                ic_dynamic_b_capacity(usize::MAX, usize::MAX, 8),
                Err(GemmError::DimProductOverflow)
            ),
            "usize::MAX 近傍は DimProductOverflow を返すはず"
        );
    }

    /// A/B 一括計測ハーネス（イシュー #1041。実機セッションで実行する
    /// 想定。値は本 PR の採用根拠にしない。x86_64 ローカルでの smoke
    /// 実行は sanity 目的のみ）。`#[ignore]` のため通常 CI では実行
    /// されない。5 回独立実行の中央値は呼び出し側運用（既存ベンチ
    /// テストと同方針。`docs/perf/cpu-gemm-candle-cpu-retune.md` 参照）。
    ///
    /// 候補間の計測順序はサンプル単位でローテーションする（PR #1075
    /// codex-review 指摘）。固定順で候補ごとに一括計測すると、5 回
    /// 独立実行してもプロセス内の測定順序自体は毎回同じになるため、
    /// キャッシュ状態・周波数ブースト・サーマルスロットリングの影響が
    /// 常に同じ候補（例: 常に 1 番目に測る RowPanel）へ偏る順序バイアスが
    /// 生じ、実機採用ゲートで候補間の性能差を誤判定しうる。本関数は
    /// ウォームアップ・本計測いずれもサンプル単位で全候補を round-robin
    /// 実行し、走査開始位置を反復ごとに 1 つずつずらす（`rotate`）ことで、
    /// どの候補も「常に先頭」「常に末尾」に固定されないようにする。
    /// A/B 一括計測ハーネス（[`gemm_blis_variant_ab_1024_2048`]・
    /// [`gemm_blis_variant_ab_4096`]）共有のヘルパー群。イシュー #1041 で
    /// `gemm_blis_variant_ab_1024_2048` のローカル関数として導入したものを、
    /// 4096 計測（イシュー #1141。`docs/perf/cpu-gemm-candle-cpu-retune.md`
    /// §5 手順 5・#1144 の 4096 非劣化ゲートの分子を供給する）と重複させない
    /// ためテストモジュール直下（`#[cfg(test)]` 限定）へ引き上げた。挙動は
    /// 変更しない（値は本番採否の入力であり、本番経路〈`gemm_blis_parallel`・
    /// `gemm_blis_bias_act_parallel`〉は変更しない）。
    fn median(mut xs: Vec<f64>) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    /// `candidates`（`(GemmDriverVariant, BlockSizes)` の組）の全候補を
    /// 1 プロセス内で計測するが、ウォームアップ・本計測ともサンプル単位で
    /// round-robin し、走査開始位置を反復ごとにローテーションする。これに
    /// より「候補 X は常に他候補より先に（＝周波数ブースト前や熱の低い
    /// 状態で）測られる」といった順序バイアスを均す（PR #1075 codex-review
    /// 指摘）。戻り値は `candidates` と同じ順序の中央値 GFLOP/s。
    ///
    /// イシュー #1315（KC 再スイープ）で [`run_variants_interleaved`]
    /// （`variant` のみを変え `blocks` は常に [`default_blocks`] 固定）から
    /// 一般化した。`blocks` も候補ごとに変えられるようにすることで、KC の
    /// ような `BlockSizes` フィールド単位のスイープを同一ハーネスで計測
    /// できる（`GemmDriverVariant` に KC 専用 variant を追加しない設計判断は
    /// 計画 §1.4-2 を参照。KC は driver の分岐ロジックではなく
    /// `gemm_blis_parallel_variant` へ渡す `blocks` 引数のパラメータに過ぎ
    /// ないため）。
    fn run_candidates_interleaved(
        candidates: &[(GemmDriverVariant, BlockSizes)],
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Vec<f64> {
        use std::time::Instant;

        let mut outputs: Vec<Vec<f32>> = candidates.iter().map(|_| vec![0.0f32; m * n]).collect();

        // ウォームアップも round-robin＋反復ごとの開始位置ローテーションで
        // 実行し、ウォームアップ順自体が本計測のキャッシュ・熱状態に
        // 与える偏りを避ける。
        for w in 0..3 {
            for offset in 0..candidates.len() {
                let idx = (offset + w) % candidates.len();
                let (variant, blocks) = candidates[idx];
                gemm_blis_parallel_variant(variant, a, b, &mut outputs[idx], m, n, k, blocks)
                    .unwrap();
            }
        }

        let mut samples: Vec<Vec<f64>> = candidates
            .iter()
            .map(|_| Vec::with_capacity(iters))
            .collect();
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        for it in 0..iters {
            // 反復ごとに走査開始位置を 1 つずつずらす。1 回の反復内では
            // 全候補を測るため合計サンプル数は候補間で完全に揃ったまま、
            // 「常に同じ候補が先に測られる」偏りだけを取り除く。
            for offset in 0..candidates.len() {
                let idx = (offset + it) % candidates.len();
                let (variant, blocks) = candidates[idx];
                let start = Instant::now();
                gemm_blis_parallel_variant(variant, a, b, &mut outputs[idx], m, n, k, blocks)
                    .unwrap();
                let elapsed = start.elapsed().as_secs_f64();
                samples[idx].push(flops / elapsed / 1e9);
            }
        }

        samples.into_iter().map(median).collect()
    }

    /// [`run_candidates_interleaved`] を `variant` のみ変えて既定
    /// `BlockSizes`（[`default_blocks`]）で計測する互換ラッパー（#1041
    /// 導入時点の元シグネチャを維持する。既存呼び出し元
    /// [`gemm_blis_variant_ab_1024_2048`]／[`gemm_blis_variant_ab_4096`]
    /// の出力形式・値を不変に保つ）。
    fn run_variants_interleaved(
        variants: &[GemmDriverVariant],
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Vec<f64> {
        let candidates: Vec<(GemmDriverVariant, BlockSizes)> =
            variants.iter().map(|&v| (v, default_blocks())).collect();
        run_candidates_interleaved(&candidates, a, b, m, n, k, iters)
    }

    #[test]
    #[ignore = "実機（M4 Max / GB10）5 回独立実行の A/B 計測専用。ローカル smoke 目的のみ \
                （#1041。cargo test -p fandhe-ai-backend-cpu --release -- --ignored \
                gemm_blis_variant_ab_1024_2048 --nocapture）"]
    fn gemm_blis_variant_ab_1024_2048() {
        let variants = all_gemm_driver_variants();

        for &dim in &[1024usize, 2048] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xdede_dede, m * k);
            let b = xorshift32_vec(0xefef_efef, k * n);

            let gflops = run_variants_interleaved(&variants, &a, &b, m, n, k, 20);
            for (variant, gflops) in variants.iter().zip(gflops) {
                println!("variant={variant:?} size={dim} median_gflops={gflops:.3}");
            }
        }
    }

    /// N=4096 版の A/B 計測（イシュー #1141）。既存
    /// `gemm_blis_variant_ab_1024_2048` は #1041 導入時点で 1024/2048 のみを
    /// 対象としていたが、`docs/perf/cpu-gemm-candle-cpu-retune.md` §5 手順 5・
    /// 後続 #1144 の採用ゲートは「1024/2048 で gemm crate 以上 **かつ 4096 で
    /// 非劣化**」を条件とするため、4096 の候補別値を独立に計測できるよう本
    /// テストを追加した。出力形式（`variant=… size=… median_gflops=…`）は
    /// 既存テストと揃え、後続の記入表集計スクリプトを共用できるようにする。
    #[test]
    #[ignore = "実機（M4 Max / GB10）5 回独立実行の A/B 計測専用（4096 非劣化ゲート用。\
                #1141。cargo test -p fandhe-ai-backend-cpu --release -- --ignored \
                gemm_blis_variant_ab_4096 --nocapture）"]
    fn gemm_blis_variant_ab_4096() {
        let variants = all_gemm_driver_variants();

        let (m, n, k) = (4096usize, 4096usize, 4096usize);
        let a = xorshift32_vec(0xdede_dede, m * k);
        let b = xorshift32_vec(0xefef_efef, k * n);

        let gflops = run_variants_interleaved(&variants, &a, &b, m, n, k, 20);
        for (variant, gflops) in variants.iter().zip(gflops) {
            println!("variant={variant:?} size={n} median_gflops={gflops:.3}");
        }
    }

    // --- KC 再スイープ（イシュー #1315。候補 3・`docs/perf/cpu-gemm-candle-cpu-retune.md`
    //     §8「候補 3 KC 再スイープ」の実測資産） ---

    /// KC 再スイープの対象グリッド。現行本番既定 `KC`（256）を含む
    /// 128〜512 の 5 点（計画 §3 で計測前に確定した固定グリッド。実測後の
    /// 追加・削除は行わない）。
    const KC_SWEEP_GRID: [usize; 5] = [128, 192, 256, 384, 512];

    /// KC 再スイープの候補列（`RowPanel` 固定・`MC`/`NC` は
    /// [`default_blocks`] を継承し `kc` のみ [`KC_SWEEP_GRID`] で差し替え）
    /// を [`run_candidates_interleaved`] へ渡せる形で構築する。KC は
    /// [`GemmDriverVariant`] の分岐対象ではなく `BlockSizes` のフィールドの
    /// ため、driver variant を追加せず `blocks` 側で表現する（計画 §1.4-2）。
    fn kc_sweep_candidates() -> Vec<(GemmDriverVariant, BlockSizes)> {
        let base = default_blocks();
        KC_SWEEP_GRID
            .iter()
            .map(|&kc| {
                (
                    GemmDriverVariant::RowPanel,
                    BlockSizes {
                        mc: base.mc,
                        kc,
                        nc: base.nc,
                    },
                )
            })
            .collect()
    }

    #[test]
    #[ignore = "実機（M4 Max / GB10）5 回独立実行の A/B 計測専用（KC 再スイープ・イシュー \
                #1315。cargo test -p fandhe-ai-backend-cpu --release -- --ignored \
                gemm_blis_kc_sweep_ab_1024_2048 --nocapture）"]
    fn gemm_blis_kc_sweep_ab_1024_2048() {
        let candidates = kc_sweep_candidates();

        for &dim in &[1024usize, 2048] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xdede_dede, m * k);
            let b = xorshift32_vec(0xefef_efef, k * n);

            let gflops = run_candidates_interleaved(&candidates, &a, &b, m, n, k, 20);
            for ((variant, blocks), gflops) in candidates.iter().zip(gflops) {
                println!(
                    "variant={variant:?} kc={} size={dim} median_gflops={gflops:.3}",
                    blocks.kc
                );
            }
        }
    }

    /// N=4096 版の KC 再スイープ（イシュー #1315。#1141 の
    /// `gemm_blis_variant_ab_4096` と同じ理由〈4096 非劣化ゲートの分子を
    /// 独立に計測する必要〉で 1024/2048 版と分離する）。
    #[test]
    #[ignore = "実機（M4 Max / GB10）5 回独立実行の A/B 計測専用（KC 再スイープ・4096 非劣化 \
                ゲート用。イシュー #1315。cargo test -p fandhe-ai-backend-cpu --release -- \
                --ignored gemm_blis_kc_sweep_ab_4096 --nocapture）"]
    fn gemm_blis_kc_sweep_ab_4096() {
        let candidates = kc_sweep_candidates();

        let (m, n, k) = (4096usize, 4096usize, 4096usize);
        let a = xorshift32_vec(0xdede_dede, m * k);
        let b = xorshift32_vec(0xefef_efef, k * n);

        let gflops = run_candidates_interleaved(&candidates, &a, &b, m, n, k, 20);
        for ((variant, blocks), gflops) in candidates.iter().zip(gflops) {
            println!(
                "variant={variant:?} kc={} size={n} median_gflops={gflops:.3}",
                blocks.kc
            );
        }
    }

    /// [`KC_SWEEP_GRID`] の全 KC 値で `RowPanel` が [`crate::gemm::gemm_naive`]
    /// と bit 完全一致することを検証する（REQ-2 の bit 一致契約。
    /// `docs/perf/cpu-gemm-blocking-sweep.md` §3.2「C タイルは pc（K
    /// ブロック）をまたいで現在値をロードして FMA 連鎖を継続するため、
    /// 累積順序は KC の値に依らず常に p 昇順」により KC の変更は縮約順序を
    /// 変えない、という主張を KC 再スイープの候補グリッドで直接確認する）。
    /// `k` は境界を跨ぐ値（`KC_SWEEP_GRID` のどの値でも割り切れない・末尾
    /// `kc_len` が `k % 4 ∈ {1,2,3}` を含むよう choose）にして端タイル
    /// 処理の bit 一致もあわせて検証する。x86_64 でも実行可能（CI 対象）。
    #[test]
    fn gemm_blis_row_panel_kc_grid_matches_naive_bit_exact() {
        let (m, n, k) = (37, 53, 1099);
        let a = xorshift32_vec(0x1111_2222, m * k);
        let b = xorshift32_vec(0x3333_4444, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        for &kc in &KC_SWEEP_GRID {
            let blocks = BlockSizes {
                mc: default_blocks().mc,
                kc,
                nc: default_blocks().nc,
            };
            let mut c_blocked = vec![0.0f32; m * n];
            gemm_blis_parallel_variant(
                GemmDriverVariant::RowPanel,
                &a,
                &b,
                &mut c_blocked,
                m,
                n,
                k,
                blocks,
            )
            .unwrap();

            assert_eq!(
                c_naive, c_blocked,
                "kc={kc} は gemm_naive と bit 完全一致するはず（#1315）"
            );
        }
    }

    /// イシュー #1315: KC 再スイープ候補グリッド（[`KC_SWEEP_GRID`]）が、
    /// 大形状（1024/2048/4096 正方）でも本番既定 KC（256・[`default_blocks`]）
    /// の `RowPanel` 出力と bit 完全一致することを実機で確認する（release
    /// ビルド。#1366 の `gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large`
    /// と同型）。KC=256 自体はグリッドに含まれるため自明に一致するが、
    /// グリッド全点を大形状でも横断することで小形状テストでは踏まない
    /// メモリレイアウト・並列分割経路を確認する。
    #[test]
    #[ignore = "実機（M4 Max / GB10）での大形状 bit 一致確認専用（KC 再スイープ・イシュー \
                #1315。cargo test -p fandhe-ai-backend-cpu --release -- --ignored \
                gemm_blis_row_panel_kc_grid_matches_default_kc_bit_exact_large --nocapture）"]
    fn gemm_blis_row_panel_kc_grid_matches_default_kc_bit_exact_large() {
        for &dim in &[1024usize, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xaaaa_1111 ^ (dim as u32), m * k);
            let b = xorshift32_vec(0xbbbb_2222 ^ (dim as u32), k * n);

            let mut c_default_kc = vec![0.0f32; m * n];
            gemm_blis_parallel_variant(
                GemmDriverVariant::RowPanel,
                &a,
                &b,
                &mut c_default_kc,
                m,
                n,
                k,
                default_blocks(),
            )
            .unwrap();

            for &kc in &KC_SWEEP_GRID {
                let blocks = BlockSizes {
                    mc: default_blocks().mc,
                    kc,
                    nc: default_blocks().nc,
                };
                let mut c_kc = vec![0.0f32; m * n];
                gemm_blis_parallel_variant(
                    GemmDriverVariant::RowPanel,
                    &a,
                    &b,
                    &mut c_kc,
                    m,
                    n,
                    k,
                    blocks,
                )
                .unwrap();

                assert_eq!(
                    c_default_kc, c_kc,
                    "dim={dim} kc={kc} は既定 KC=256 と bit 完全一致するはず（#1315）"
                );
            }
        }
    }

    // --- VJP 専用 NT/TN 2 パターン入口（#1213） ---

    /// [`gemm_row_vector_nt`] が [`gemm_row_vector`] と bit 完全一致する
    /// ことを検証する（`bt[j,p] == b[p,j]` を素朴な転置コピーで確認する）。
    #[test]
    fn gemm_row_vector_nt_matches_gemm_row_vector_bit_exact() {
        let (k, n) = (513usize, 777usize);
        let a = xorshift32_vec(0x1234_5678, k);
        let b = xorshift32_vec(0x9abc_def0, k * n);
        // bt[j*k+p] = b[p*n+j]（b の転置コピー。論理形状 [n,k] 行優先）。
        let mut bt = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                bt[j * k + p] = b[p * n + j];
            }
        }

        let mut c_ref = vec![0.0f32; n];
        gemm_row_vector(&a, &b, &mut c_ref, n);
        let mut c_nt = vec![0.0f32; n];
        gemm_row_vector_nt(&a, &bt, &mut c_nt, k);

        assert_eq!(
            c_ref, c_nt,
            "gemm_row_vector_nt は gemm_row_vector と bit 完全一致するはず"
        );
    }

    /// 転置コピーヘルパー（テスト専用）。`src` を `[rows,cols]` 行優先と
    /// みなし、`[cols,rows]` 行優先へ転置コピーする。
    fn transpose_copy_row_major(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = src[r * cols + c];
            }
        }
        out
    }

    /// [`gemm_blis_parallel_nt`]（`b` が転置格納 `bt`）が、`bt` を
    /// `contiguous()` 相当（素朴な転置コピー）してから
    /// [`gemm_blis_parallel`] へ渡した結果と bit 完全一致することを
    /// MC/KC/NC 境界を跨ぐ複数形状で検証する（#1213 の bit 完全一致
    /// 契約の直接検証）。
    #[test]
    fn gemm_blis_parallel_nt_matches_gemm_blis_parallel_of_contiguous_copy() {
        // (m, n, k): MC=128/KC=256/NC=512（default_blocks 実測前提。#564）
        // の境界を跨ぐ組み合わせを含む。
        let shapes = [
            (2usize, 3usize, 4usize),
            (7, 11, 13),
            (64, 64, 64),
            (127, 129, 131),
            (200, 300, 400),
        ];
        for (m, n, k) in shapes {
            let a = xorshift32_vec(0x1111_0000 ^ (m as u32), m * k);
            // bt: 論理形状 [n,k] 行優先（元の b [k,n] の転置）。
            let bt = xorshift32_vec(0x2222_0000 ^ (n as u32), n * k);
            let b = transpose_copy_row_major(&bt, n, k); // bt を転置し戻した b（コピー）。

            let mut c_ref = vec![0.0f32; m * n];
            gemm_blis_parallel(&a, &b, &mut c_ref, m, n, k).unwrap();

            let mut c_nt = vec![0.0f32; m * n];
            gemm_blis_parallel_nt(&a, &bt, &mut c_nt, m, n, k).unwrap();

            assert_eq!(
                c_ref, c_nt,
                "gemm_blis_parallel_nt は shape ({m},{n},{k}) で gemm_blis_parallel(contiguous) と bit 完全一致するはず"
            );
        }
    }

    /// [`gemm_blis_parallel_tn`]（`a` が転置格納 `at`）版。
    #[test]
    fn gemm_blis_parallel_tn_matches_gemm_blis_parallel_of_contiguous_copy() {
        let shapes = [
            (2usize, 3usize, 4usize),
            (7, 11, 13),
            (64, 64, 64),
            (127, 129, 131),
            (200, 300, 400),
        ];
        for (m, n, k) in shapes {
            // at: 論理形状 [k,m] 行優先（元の a [m,k] の転置）。
            let at = xorshift32_vec(0x3333_0000 ^ (k as u32), k * m);
            let a = transpose_copy_row_major(&at, k, m); // at を転置し戻した a（コピー）。
            let b = xorshift32_vec(0x4444_0000 ^ (n as u32), k * n);

            let mut c_ref = vec![0.0f32; m * n];
            gemm_blis_parallel(&a, &b, &mut c_ref, m, n, k).unwrap();

            let mut c_tn = vec![0.0f32; m * n];
            gemm_blis_parallel_tn(&at, &b, &mut c_tn, m, n, k).unwrap();

            assert_eq!(
                c_ref, c_tn,
                "gemm_blis_parallel_tn は shape ({m},{n},{k}) で gemm_blis_parallel(contiguous) と bit 完全一致するはず"
            );
        }
    }

    /// `m == 1`（gemv 相当）での NT/TN 経路が非 m==1 経路と同じ結果に
    /// なることを確認する（`gemm_blis_parallel_with_transpose` の m==1
    /// 早期分岐が正しく [`gemm_row_vector`]／[`gemm_row_vector_nt`] を
    /// 呼び分けることの回帰検証）。
    #[test]
    fn gemm_blis_parallel_nt_tn_row_vector_m1_matches_reference() {
        let (n, k) = (300usize, 250usize);

        // NT: m=1
        let a = xorshift32_vec(0x5555_0001, k);
        let bt = xorshift32_vec(0x5555_0002, n * k);
        let b = transpose_copy_row_major(&bt, n, k);
        let mut c_ref = vec![0.0f32; n];
        gemm_blis_parallel(&a, &b, &mut c_ref, 1, n, k).unwrap();
        let mut c_nt = vec![0.0f32; n];
        gemm_blis_parallel_nt(&a, &bt, &mut c_nt, 1, n, k).unwrap();
        assert_eq!(c_ref, c_nt);

        // TN: m=1（at は [k,1] 行優先＝a と同一メモリ内容）
        let at = xorshift32_vec(0x6666_0001, k);
        let b2 = xorshift32_vec(0x6666_0002, k * n);
        let mut c_ref2 = vec![0.0f32; n];
        gemm_blis_parallel(&at, &b2, &mut c_ref2, 1, n, k).unwrap();
        let mut c_tn = vec![0.0f32; n];
        gemm_blis_parallel_tn(&at, &b2, &mut c_tn, 1, n, k).unwrap();
        assert_eq!(c_ref2, c_tn);
    }

    /// NT/TN 入口の長さ不一致が `validate_dims` により早期拒否される
    /// ことを確認する（slice アクセス前の境界検査。REQ-8）。
    #[test]
    fn gemm_blis_parallel_nt_tn_reject_length_mismatch() {
        let (m, n, k) = (4usize, 4usize, 4usize);
        let a = vec![0.0f32; m * k];
        let bt_short = vec![0.0f32; n * k - 1];
        let mut c = vec![0.0f32; m * n];
        assert!(matches!(
            gemm_blis_parallel_nt(&a, &bt_short, &mut c, m, n, k),
            Err(GemmError::BLenMismatch { .. })
        ));

        let at_short = vec![0.0f32; k * m - 1];
        let b = vec![0.0f32; k * n];
        assert!(matches!(
            gemm_blis_parallel_tn(&at_short, &b, &mut c, m, n, k),
            Err(GemmError::ALenMismatch { .. })
        ));
    }
    // --- `TwoDDynamic` vs `RowPanel` 両実機 A/B（イシュー #1312。
    //     `docs/cpu-gemm-2d-dynamic-partition-design.md` §9・§11 の
    //     採用ゲート判定用実測資産） ---

    /// [`run_candidates_interleaved`]（`GemmDriverVariant`＋`BlockSizes` の
    /// 組でしか候補を表現できない）を、`TwoDDynamic` の `jobs_per_worker`
    /// スイープ（イシュー #1312 計画 §4.4）まで表現できるよう一般化した
    /// 候補列挙。`RowPanel` は [`gemm_blis_parallel_variant`]（本番分岐と
    /// 完全一致）を、`TwoDDynamic` は [`gemm_blis_parallel_two_d_dynamic_with_params`]
    /// （`jobs_per_worker` を注入できるテスト専用入口）をそれぞれ経由する。
    /// `#[cfg(test)]` 限定・本番経路（`gemm_blis_parallel`・
    /// `gemm_blis_bias_act_parallel`）は不変。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum AbCandidate {
        /// 本番既定分岐（[`GemmDriverVariant::RowPanel`]）をそのまま計測する。
        RowPanel,
        /// (mc, nc) 2D job 動的分配（イシュー #1311）を指定
        /// `jobs_per_worker` で計測する。
        TwoDDynamic { jobs_per_worker: usize },
    }

    impl AbCandidate {
        /// ログ出力・集計スクリプト（`docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/aggregate.py`）
        /// 側のキーとして使う候補名。
        fn label(self) -> &'static str {
            match self {
                AbCandidate::RowPanel => "RowPanel",
                AbCandidate::TwoDDynamic { .. } => "TwoDDynamic",
            }
        }

        /// ログ出力用の `jobs_per_worker`（`RowPanel` は分配方式自体を
        /// 使わないため `0` を出力する。集計スクリプト側は
        /// `(variant, jobs_per_worker)` の組をキーとして扱う）。
        fn jobs_per_worker_for_log(self) -> usize {
            match self {
                AbCandidate::RowPanel => 0,
                AbCandidate::TwoDDynamic { jobs_per_worker } => jobs_per_worker,
            }
        }

        fn run(
            self,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<(), GemmError> {
            match self {
                AbCandidate::RowPanel => gemm_blis_parallel_variant(
                    GemmDriverVariant::RowPanel,
                    a,
                    b,
                    c,
                    m,
                    n,
                    k,
                    default_blocks(),
                ),
                AbCandidate::TwoDDynamic { jobs_per_worker } => {
                    gemm_blis_parallel_two_d_dynamic_with_params(
                        a,
                        b,
                        c,
                        m,
                        n,
                        k,
                        default_blocks(),
                        jobs_per_worker,
                        GemmTranspose::Nn,
                    )
                }
            }
        }
    }

    /// [`run_candidates_interleaved`] と同じ round-robin＋反復ごとの開始
    /// 位置ローテーション方式（PR #1075 codex-review 指摘・順序バイアス
    /// 対策）を [`AbCandidate`] へ適用したもの。戻り値は `candidates` と
    /// 同じ順序の中央値 GFLOP/s。
    fn run_ab_candidates_interleaved(
        candidates: &[AbCandidate],
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Vec<f64> {
        use std::time::Instant;

        let mut outputs: Vec<Vec<f32>> = candidates.iter().map(|_| vec![0.0f32; m * n]).collect();

        for w in 0..3 {
            for offset in 0..candidates.len() {
                let idx = (offset + w) % candidates.len();
                candidates[idx]
                    .run(a, b, &mut outputs[idx], m, n, k)
                    .unwrap();
            }
        }

        let mut samples: Vec<Vec<f64>> = candidates
            .iter()
            .map(|_| Vec::with_capacity(iters))
            .collect();
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        for it in 0..iters {
            for offset in 0..candidates.len() {
                let idx = (offset + it) % candidates.len();
                let start = Instant::now();
                candidates[idx]
                    .run(a, b, &mut outputs[idx], m, n, k)
                    .unwrap();
                let elapsed = start.elapsed().as_secs_f64();
                samples[idx].push(flops / elapsed / 1e9);
            }
        }

        samples.into_iter().map(median).collect()
    }

    /// 本 A/B（#1312）の固定候補集合: `RowPanel`・`TwoDDynamic(jpw=2)`・
    /// `TwoDDynamic(jpw=4)` の 3 つのみ（計画 §3.1(c)。他 variant を含めない
    /// ことで計測時間・ノイズを抑える）。
    fn two_d_dynamic_ab_candidates() -> Vec<AbCandidate> {
        vec![
            AbCandidate::RowPanel,
            AbCandidate::TwoDDynamic { jobs_per_worker: 2 },
            AbCandidate::TwoDDynamic { jobs_per_worker: 4 },
        ]
    }

    /// 実行中の `RAYON_NUM_THREADS`（未設定なら既定値）を出力するための
    /// 実効スレッド数。集計側がログファイル名からスレッド数を推定せず
    /// 出力行自体から読み取れるようにする（計画 §3.1(c)）。
    fn ab_effective_num_threads() -> usize {
        crate::thread_limit::effective_num_threads(rayon::current_num_threads())
    }

    /// N=1024/2048 の `TwoDDynamic` vs `RowPanel` A/B 計測（イシュー
    /// #1312）。両実機（DGX Spark GB10・Apple M4 Max）で
    /// `RAYON_NUM_THREADS` を変えながら 5 回独立プロセス実行し、出力を
    /// `docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/aggregate.py` で集計する。
    /// 出力形式は固定: `variant={RowPanel|TwoDDynamic} jobs_per_worker={n}
    /// num_threads={t} size={dim} median_gflops={v:.3}`。
    #[test]
    #[ignore = "実機（M4 Max / GB10）5 回独立実行の A/B 計測専用（#1312。 \
                cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
                gemm_blis_two_d_dynamic_ab_1024_2048 --nocapture）"]
    fn gemm_blis_two_d_dynamic_ab_1024_2048() {
        let candidates = two_d_dynamic_ab_candidates();
        let num_threads = ab_effective_num_threads();

        for &dim in &[1024usize, 2048] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xdede_dede, m * k);
            let b = xorshift32_vec(0xefef_efef, k * n);

            let gflops = run_ab_candidates_interleaved(&candidates, &a, &b, m, n, k, 20);
            for (candidate, gflops) in candidates.iter().zip(gflops) {
                println!(
                    "variant={} jobs_per_worker={} num_threads={num_threads} size={dim} \
                     median_gflops={gflops:.3}",
                    candidate.label(),
                    candidate.jobs_per_worker_for_log(),
                );
            }
        }
    }

    /// N=4096 版（イシュー #1312。`gemm_blis_variant_ab_4096` と同型に
    /// 分離する理由は同コメント参照: 1024/2048 とは別のゲート条件
    /// 〈4096 で非劣化〉を独立に計測できるようにするため）。
    #[test]
    #[ignore = "実機（M4 Max / GB10）5 回独立実行の A/B 計測専用（#1312。 \
                cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
                gemm_blis_two_d_dynamic_ab_4096 --nocapture）"]
    fn gemm_blis_two_d_dynamic_ab_4096() {
        let candidates = two_d_dynamic_ab_candidates();
        let num_threads = ab_effective_num_threads();

        let (m, n, k) = (4096usize, 4096usize, 4096usize);
        let a = xorshift32_vec(0xdede_dede, m * k);
        let b = xorshift32_vec(0xefef_efef, k * n);

        let gflops = run_ab_candidates_interleaved(&candidates, &a, &b, m, n, k, 20);
        for (candidate, gflops) in candidates.iter().zip(gflops) {
            println!(
                "variant={} jobs_per_worker={} num_threads={num_threads} size={n} \
                 median_gflops={gflops:.3}",
                candidate.label(),
                candidate.jobs_per_worker_for_log(),
            );
        }
    }

    // --- SME（`fmopa`）vs NEON A/B（イシュー #1587。事前登録規則
    //     R4「しきい値の決め方」・R1〜R2「本番性能・非後退」の生データ
    //     取得用。5 プロセス起動中央値の正式プロトコルは
    //     `docs/perf/logs/cpu-gemm-sme-fmopa-1587/` のオーケストレーション
    //     スクリプトへ委ねる） ---

    /// SME マイクロカーネル（`GemmDriverVariant::TwoDDynamicSme`）と本番
    /// NEON 経路（`GemmDriverVariant::TwoDDynamic`）を interleave 方式
    /// （[`run_ab_candidates_interleaved`] と同じ round-robin＋開始位置
    /// ローテーション）で比較する。SME 非対応環境（GB10 等）では実行時
    /// スキップする。出力形式は既存 A/B と揃え（`variant=… size=…
    /// median_gflops=…`）、集計スクリプトを共用できるようにする。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "実機（M4 Max。SME 対応環境限定）での A/B 計測専用（#1587。 \
                cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
                sme_vs_neon_ab_shape_sweep --nocapture）"]
    fn sme_vs_neon_ab_shape_sweep() {
        if microkernel::SmeKernel::try_new().is_none() {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        }
        // R4 しきい値スイープの対象格子（issue #1587 事前登録規則）に加え、
        // R1 の framework-compare 対象形状に近い正方形状も含める。
        let shapes: &[(usize, usize, usize)] = &[
            (64, 64, 32),
            (128, 128, 64),
            (256, 256, 64),
            (256, 256, 128),
            (512, 512, 256),
            (1024, 1024, 512),
            (2048, 2048, 1024),
        ];
        for &(m, n, k) in shapes {
            let a = xorshift32_vec(0x1234_abcd ^ (m as u32), m * k);
            let b = xorshift32_vec(0x5678_ef01 ^ (n as u32), k * n);

            let candidates: Vec<(GemmDriverVariant, BlockSizes)> = vec![
                (GemmDriverVariant::TwoDDynamic, default_blocks()),
                (GemmDriverVariant::TwoDDynamicSme, default_blocks()),
            ];
            let gflops = run_candidates_interleaved(&candidates, &a, &b, m, n, k, 10);
            let labels = ["NEON(TwoDDynamic)", "SME(TwoDDynamicSme)"];
            for (label, gflops) in labels.iter().zip(gflops) {
                println!("variant={label} size=({m},{n},{k}) median_gflops={gflops:.3}");
            }
        }
    }
}
