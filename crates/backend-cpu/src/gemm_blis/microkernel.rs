//! マイクロカーネル契約と ISA ごとの実装への配線。
//!
//! [`super`] の 5-loop ドライバから呼ばれる。契約は共通:
//! packed A（`ap`、`MR * kc_len` 要素、p-major）・packed B（`bp`、
//! `kc_len * NR` 要素、p-major）から MR×NR の C タイル（`c_tile`、
//! row-major・ld=NR）へ `kc_len` ぶんの寄与を p 昇順の `mul_add`（または
//! 対応する SIMD FMA）で加算する。C タイルの現在値ロード・書き戻しは
//! 呼び出し元（ドライバ）の責務であり、本モジュール以下の関数は
//! `c_tile` の中身のみを扱う。
//!
//! ## ISA 選択（#185・TASK-1.6g で実行時ディスパッチへ移行）
//!
//! TASK-1.6f（#184）まではコンパイル時 cfg のみで ISA を固定していたが、
//! それでは x86_64 の既定ビルド（`RUSTFLAGS` 指定なし）で実行 CPU が
//! AVX2/AVX-512 を持っていてもスカラーへ落ちてしまい REQ-8 の CPU 性能
//! 下限（対 PyTorch CPU 比 20%）達成を妨げる。本モジュールは faer の
//! `pulp` 方式（検出済みトークン型による dispatch）を依存追加なしで
//! 自前実装し、[`Isa::detect`] による実行時 CPU 機能検出でマイクロ
//! カーネルを選択する:
//!
//! - aarch64: NEON は常時有効の baseline ISA のため無条件で `neon` を選ぶ
//!   （実行時検出は不要。[`Isa::detect`] は常に `Isa::Neon` を返す）
//! - x86_64: [`Isa::detect`] が `is_x86_feature_detected!("avx512f")` →
//!   `"avx2"` かつ `"fma"` → 非対応の順に検出し、対応する `Avx512Kernel` /
//!   `Avx2Kernel` / [`ScalarKernel`] トークンを選ぶ
//! - その他 arch: 常に [`ScalarKernel`]
//!
//! ### 健全性契約（トークン型による dispatch）
//!
//! `Avx2Kernel`／`Avx512Kernel` は生成経路を検出済みの場合に限定する
//! （`try_new` が検出成功時のみ `Some` を返す非公開コンストラクタ）。
//! トークンのインスタンスが存在すること自体が「実行 CPU が当該 ISA を
//! サポートする」証明となり、[`Microkernel::run`] 内部の
//! `unsafe { kernel_unchecked(...) }` 呼び出しの SAFETY 根拠になる
//! （Safety 契約の履行責務をコンストラクタに集約することで、`run` 呼び
//! 出し側は `unsafe` を意識せず安全に dispatch できる）。
//!
//! 環境変数等による dispatch 上書き機構は設けない（外部入力が `unsafe`
//! カーネル選択を制御できると SIGILL・未定義動作の攻撃面になるため。
//! OWASP A03・`.claude/rules/security.md`）。
//!
//! ### 公開 API 非破壊
//!
//! TASK-1.6f で公開していたコンパイル時 cfg 選択の `pub use
//! {neon,avx2,scalar}::{MR, NR, kernel}` はそのまま残す（既存呼び出し元
//! 互換のため）。ただし [`super::gemm_blis`]／[`super::gemm_blis_parallel`]
//! の駆動経路は本モジュールの実行時ディスパッチ（[`Isa::detect`] 経由）
//! へ切り替わっており、この `pub use` 経路は駆動経路から外れている点に
//! 注意する。
//!
//! `avx2`／`avx512` モジュールは `cfg(target_arch = "x86_64")` のみで
//! コンパイルし `target_feature` ではゲートしない（レビュー指摘: モジュール
//! 単位でゲートすると既定ビルドで本体が一切コンパイルされず、テスト
//! 限定の実行時検出ガード付き直接検証が不可能になるため）。
//!
//! ただし `avx512` モジュールのみ、上記に加えて `avx512_stable` cfg
//! （`backend-cpu` クレートルートの `build.rs` が、AVX-512F intrinsics と
//! `#[target_feature(enable = "avx512f")]` を実際にコンパイルする probe を
//! 実行して発行。バージョン番号の決め打ちではなく実測判定である理由は
//! `build.rs` のコメント参照。PR #337 の CI 実測: self-hosted runner の
//! rustc 1.88.0 では `_mm512_*` intrinsics が `stdarch_x86_avx512` unstable
//! ライブラリ機能のため E0658 でビルド不能）でもゲートする。AVX2 はこの
//! 制約を受けない（AVX2 intrinsics は長らく stable）。

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

// イシュー #1587: Arm SME（Scalable Matrix Extension）`fmopa` マイクロ
// カーネル。NEON と同じく `cfg(target_arch = "aarch64")` 限定だが、NEON
// と異なり実行時検出（[`SmeKernel::try_new`]）を経由しなければ安全に
// 呼べない（`avx2`／`avx512` と同型の「検出済みトークンのみ構築可能」
// パターン。本モジュール doc 冒頭参照）。
#[cfg(target_arch = "aarch64")]
pub mod sme;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", avx512_stable))]
pub mod avx512;

use std::fmt;
use std::sync::OnceLock;

/// `check_c_tile_bounds` が検出する `ldc`／`c` 長の契約違反を表す型付き
/// エラー（#691 レビュー P1 再指摘への対応）。
///
/// 従来は `assert!`／`panic!` で検出結果を通知していたが、
/// [`Microkernel::run_with_ldc`] 既定実装・各 ISA の `kernel_with_ldc`／
/// `kernel_unchecked_with_ldc` 入口は外部の `Microkernel` 実装からも到達
/// 可能な**公開**入口であり、呼び出し元が誤った `ldc`／`c` を渡した場合に
/// panic が本番ライブラリ経路から漏れていた（AGENTS.md「本番経路の panic
/// 禁止」・`.claude/rules/coding-rust.md`「本番経路で unwrap()/expect() を
/// 使わない」）。本型により検出結果を `Result::Err` として公開入口まで
/// 伝播させ、呼び出し元がハンドリングできるようにする。検査ロジック
/// 自体（REQ-8 境界検査規約に基づく呼び出し元契約違反の早期検出）は
/// 変更しない。
///
/// `#[non_exhaustive]` を付与する（#691 レビュー P0 再指摘
/// `PRRT_kwDOTuUCJc6ZrQZE` 対応で [`NonPositiveNr`](Self::NonPositiveNr)
/// を追加する際、`match` を非網羅的に強制することで既存の外部利用者の
/// 網羅的 `match` を静かに壊さない。`GemmError`〈`mod.rs`〉と同じ方針）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TileBoundsError {
    /// `ldc` が `NR` 未満。
    LdcTooSmall {
        /// 呼び出し元が渡した `ldc`。
        ldc: usize,
        /// マイクロカーネルタイルの列数（[`Microkernel::NR`]）。
        nr: usize,
    },
    /// `MR` が 0、または `(MR-1)*ldc` がオーバーフローする（[`Microkernel::MR`]
    /// の「1 以上」契約違反を含む）。
    NonPositiveMrOrOverflow {
        /// マイクロカーネルタイルの行数（[`Microkernel::MR`]）。
        mr: usize,
        /// 呼び出し元が渡した `ldc`。
        ldc: usize,
    },
    /// `c` の長さが必要長（`(MR-1)*ldc+NR`）未満。
    CBufferTooSmall {
        /// `(MR-1)*ldc+NR` で求まる必要長。
        required: usize,
        /// 呼び出し元が渡した `c` の実際の長さ。
        actual: usize,
    },
    /// `NR` が 0（[`Microkernel::NR`] の「1 以上」契約違反。#691 レビュー
    /// P0 再指摘 `PRRT_kwDOTuUCJc6ZrQZG`／`PRRT_kwDOTuUCJc6ZrQZE`）。
    ///
    /// 修正前は `ldc < nr` 判定が `nr == 0` のとき恒偽（`usize` は負に
    /// ならない）になり、後続の `required = (mr-1)*ldc+nr` も `nr=0` を
    /// そのまま許容してしまうため、`MR=1, NR=0, ldc=0, c=[]` が検査を
    /// 素通りして `Self::run` へ到達しうる状態だった（REQ-8 境界検査規約
    /// 違反）。本バリアントで `NR == 0` を明示的に拒否する。
    NonPositiveNr {
        /// 呼び出し元が渡した `nr`（常に 0）。
        nr: usize,
    },
    /// `ap`／`bp`（packed A／B パネル）の長さが `MR * kc_len`／`kc_len * NR`
    /// と一致しない、またはその積が `usize` でオーバーフローする（#691
    /// レビュー P0 再指摘 `PRRT_kwDOTuUCJc6ZrXKs`）。
    ///
    /// `check_panel_lengths` のドキュメント参照。素朴な `usize` 乗算
    /// （`MR * kc_len` 等）は release ビルド（`overflow-checks` 無効）で
    /// オーバーフロー時にラップし、意図しない短い `ap`／`bp` を境界検査の
    /// 素通りさせて後続の `unsafe` SIMD ロードを未定義動作へ導きうる
    /// （NEON `kernel_with_ldc` で具体的に指摘された経路）ため、
    /// `checked_mul` で判定しオーバーフロー自体も不一致として扱う。
    PanelLengthMismatch {
        /// 検証対象のパネル種別（`"ap"` または `"bp"`）。
        panel: &'static str,
        /// 呼び出し元が渡した実際の長さ。
        actual: usize,
    },
}

impl fmt::Display for TileBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LdcTooSmall { ldc, nr } => {
                write!(f, "ldc must be at least NR (ldc={ldc}, NR={nr})")
            }
            Self::NonPositiveMrOrOverflow { mr, ldc } => write!(
                f,
                "MR must be positive, or ldc*MR overflow (MR={mr}, ldc={ldc})"
            ),
            Self::CBufferTooSmall { required, actual } => write!(
                f,
                "C tile buffer too small for MR*ldc access pattern (required={required}, actual={actual})"
            ),
            Self::NonPositiveNr { nr } => write!(f, "NR must be positive (NR={nr})"),
            Self::PanelLengthMismatch { panel, actual } => write!(
                f,
                "packed {panel} panel length mismatch (or overflow computing the expected length): actual={actual}"
            ),
        }
    }
}

impl std::error::Error for TileBoundsError {}

/// `actual == factor_a * factor_b` を `checked_mul` で判定する（#691
/// レビュー P0 再指摘 `PRRT_kwDOTuUCJc6ZrXKs` への対応）。
///
/// 各 ISA の `kernel`／`kernel_unchecked`／`kernel_with_ldc`／
/// `kernel_unchecked_with_ldc` は入口で `ap.len() == MR * kc_len`・
/// `bp.len() == kc_len * NR` を検証しているが、`kc_len` は呼び出し元
/// （5-loop ドライバ）が渡す値で理論上任意の `usize` を取りうる。素朴な
/// `MR * kc_len` の乗算は release ビルド（`overflow-checks` 無効）では
/// オーバーフロー時に `usize::MAX` からラップし、極端に大きい `kc_len`
/// と短い `ap`／`bp` の組み合わせでも偶然一致し検査を素通りしうる。その
/// 状態で `compute` 内の NEON／AVX2／AVX-512 `unsafe` ロード（スライスの
/// 範囲チェックを経ない生ポインタ読み出し）へ進むと未定義動作になる
/// （NEON `kernel_with_ldc` で具体的に指摘された経路。同型の未検査乗算は
/// 全 ISA の入口に共通するため、本関数へ集約し `checked_mul` で
/// オーバーフローそのものを「不一致」として扱う）。
pub(crate) fn panel_len_matches(actual: usize, factor_a: usize, factor_b: usize) -> bool {
    factor_a.checked_mul(factor_b) == Some(actual)
}

/// `ap`／`bp` の長さ契約（[`panel_len_matches`]）を検査し、違反時は
/// [`TileBoundsError::PanelLengthMismatch`] を返す。
///
/// 各 ISA の `kernel_with_ldc`／`kernel_unchecked_with_ldc`（既に `Result`
/// を返す公開入口）が [`check_c_tile_bounds`] の直前に呼ぶ。従来これらの
/// 入口は `assert_eq!` で ap/bp 長不一致を検出していたが、`Result` を
/// 返せる入口である以上ここも panic ではなく型付きエラーへ揃えるのが
/// 一貫する（#691 レビュー P0 再指摘への対応）。一方、従来どおり `()` を
/// 返す必須シグネチャの後方互換ラッパー（各 ISA の `kernel`／
/// `kernel_unchecked`／`kernel_12x8`）は `Result` を返せないため、
/// [`panel_len_matches`] を直接使う `assert!` のまま維持する（`assert!`
/// は `debug_assert!` と異なり release ビルドでも有効であり、
/// オーバーフロー起因の検査素通りは起きない）。
pub(crate) fn check_panel_lengths(
    mr: usize,
    nr: usize,
    kc_len: usize,
    ap_len: usize,
    bp_len: usize,
) -> Result<(), TileBoundsError> {
    if !panel_len_matches(ap_len, mr, kc_len) {
        return Err(TileBoundsError::PanelLengthMismatch {
            panel: "ap",
            actual: ap_len,
        });
    }
    if !panel_len_matches(bp_len, kc_len, nr) {
        return Err(TileBoundsError::PanelLengthMismatch {
            panel: "bp",
            actual: bp_len,
        });
    }
    Ok(())
}

/// `ldc`／`c` 長の契約検査（scalar・neon・avx2・avx512 の各 ISA 実装間で
/// 重複していたロジックを集約）。
///
/// `mr == 0`・`ldc < NR`・`(MR-1)*ldc+NR` のオーバーフロー・`c` の長さ
/// 不足のいずれかであれば [`TileBoundsError`] を返す（呼び出し元契約違反
/// 〈呼び出し元バグ〉を早期検出する REQ-8 境界検査規約に基づく検証で
/// あり、実行時外部入力の検証ではない。各 ISA の `kernel_with_ldc`／
/// `kernel_unchecked_with_ldc` 入口から呼ばれる）。
///
/// ## `mr > 0` の明示検査（#691 レビュー再指摘への対応）
///
/// [`Microkernel`] trait は `MR > 0` を型・ドキュメントいずれでも契約と
/// して課していなかった（本対応で trait 側にも明記。[`Microkernel::MR`]
/// 参照）。外部実装が `MR = 0` の状態で（`ldc != NR` の）
/// [`Microkernel::run_with_ldc`] 既定実装から本関数を呼ぶと、`mr - 1` の
/// 減算が本来の境界検査より先に発生してしまう（debug ビルドでは減算
/// オーバーフローで panic、release ビルド〈overflow-checks 無効〉では
/// `usize::MAX` へラップし後続の `checked_mul`/`checked_add` がほぼ全ての
/// `ldc`/`nr` 組み合わせで `None` または長さ不足の `Err` に倒れるため
/// 実害は無いが、意図した「MR must be positive」という診断ではなく
/// 偶発的なオーバーフロー起因の結果になってしまう）。`mr.checked_sub(1)`
/// を計算チェーンへ含め、`mr == 0` を明示的に検出して意図の分かる
/// [`TileBoundsError::NonPositiveMrOrOverflow`] を返す。
///
/// ## `nr > 0` の明示検査（#691 レビュー P0 再指摘 `PRRT_kwDOTuUCJc6ZrQZE`
/// ／`PRRT_kwDOTuUCJc6ZrQZG` への対応）
///
/// `nr == 0` の場合、`ldc < nr` は `usize` が負値を取れないため恒偽になり
/// 素通りしてしまう（例: `mr=1, nr=0, ldc=0, c_len=0` は従来 `required=0`
/// ・`c_len>=required` で `Ok(())` になっていた）。[`Microkernel::NR`] の
/// 「1 以上」契約違反を後続の算術に委ねず、`nr == 0` をここで明示的に
/// 検出して [`TileBoundsError::NonPositiveNr`] を返す。
pub(crate) fn check_c_tile_bounds(
    mr: usize,
    nr: usize,
    ldc: usize,
    c_len: usize,
) -> Result<(), TileBoundsError> {
    if nr == 0 {
        return Err(TileBoundsError::NonPositiveNr { nr });
    }
    if ldc < nr {
        return Err(TileBoundsError::LdcTooSmall { ldc, nr });
    }
    match mr
        .checked_sub(1)
        .and_then(|mr_minus_1| mr_minus_1.checked_mul(ldc))
        .and_then(|v| v.checked_add(nr))
    {
        Some(required) if c_len >= required => Ok(()),
        Some(required) => Err(TileBoundsError::CBufferTooSmall {
            required,
            actual: c_len,
        }),
        None => Err(TileBoundsError::NonPositiveMrOrOverflow { mr, ldc }),
    }
}

// デフォルト経路の選択（コンパイル時 cfg。公開 API 非破壊のため残すが、
// [`gemm_blis`]／[`gemm_blis_parallel`] の駆動経路は本モジュールの
// 実行時ディスパッチへ切り替わっている。モジュールドキュメント参照）。

#[cfg(target_arch = "aarch64")]
pub use neon::{MR, NR, kernel};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub use avx2::{MR, NR, kernel};

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
pub use scalar::{MR, NR, kernel};

/// 実行時 ISA ディスパッチが選択可能なマイクロカーネルの共通契約。
///
/// `super::gemm_blis_region` はこの trait をジェネリック境界に取り、
/// `K::MR`／`K::NR` でタイル形状を、[`Microkernel::run`] で累積計算を
/// 行う。各実装（[`ScalarKernel`]／`NeonKernel`／`Avx2Kernel`／
/// `Avx512Kernel`）は `Copy + Sync` な ZST（サイズ 0 の型）であり、
/// rayon のクロージャへそのまま値渡しできる。
pub trait Microkernel: Copy + Sync {
    /// マイクロカーネルタイルの行数。**契約: 1 以上でなければならない**
    /// （#691 レビュー指摘。`MR == 0` は `check_c_tile_bounds` の
    /// `mr.checked_sub(1)` 経由で明示的に
    /// [`TileBoundsError::NonPositiveMrOrOverflow`] を返すが、`0` を許容
    /// する設計ではない。実装がこの契約を破ると
    /// [`Microkernel::run_with_ldc`] 既定実装・各 ISA の
    /// `kernel`／`kernel_unchecked` は正しい結果を返さない）。
    const MR: usize;
    /// マイクロカーネルタイルの列数。**契約: 1 以上でなければならない**
    /// （[`Self::MR`] 同様。`NR == 0` は `check_c_tile_bounds` が
    /// `nr == 0` を明示的に検出して
    /// [`TileBoundsError::NonPositiveNr`] を返す〈#691 レビュー P0 再指摘
    /// `PRRT_kwDOTuUCJc6ZrQZE`〉。以前は `ldc < nr` 判定が `nr == 0` の
    /// とき `usize` の性質上恒偽になり後続の算術検査もろとも素通りしていた
    /// が、現在は本ガードにより「0 を返してはならない」契約が明示検査で
    /// 担保される）。
    const NR: usize;

    /// `ap`（packed A）・`bp`（packed B）から `c_tile`（MR×NR、row-major・
    /// ld=NR）へ `kc_len` ぶんの寄与を加算する。安全な呼び出し専用の入口
    /// であり、内部の `unsafe`（intrinsics 呼び出し）はトークンの生成
    /// 経路（検出済みの場合のみ構築可能）によって健全性が保証される。
    ///
    /// ## 公開 API 非破壊（#691 レビュー指摘への再対応）
    ///
    /// 当初 #691 対応として本メソッドへデフォルト実装（[`Self::run_with_ldc`]
    /// への委譲）を与え `run_with_ldc` を必須メソッドとしたが、これは
    /// 「従来 `run` のみを実装するクレート外部の `Microkernel` 実装」が
    /// 新設の必須メソッド `run_with_ldc` 未実装により `E0046` でコンパイル
    /// 不能になる別の破壊的変更を生んでいた（codex-review・Cursor Bugbot
    /// 双方の再指摘）。そのため本メソッドを**従来どおり必須メソッド
    /// （デフォルト実装なし）として維持**し、`ldc` 拡張は
    /// [`Self::run_with_ldc`] 側にのみデフォルト実装を持たせる非対称な形へ
    /// 修正した。既存の外部実装（`run` のみをオーバーライド）は無変更で
    /// コンパイル可能になる（`run_with_ldc` のデフォルト実装が `run` へ
    /// フォールバックする。[`Self::run_with_ldc`] のドキュメント参照）。
    /// 組み込み実装（[`ScalarKernel`] 等）は `run` の実装を
    /// `run_with_ldc(..., Self::NR, ...)` への委譲として与えつつ、
    /// `run_with_ldc` 自体は ISA ごとの直接 C 経路（#557）を実装する。
    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize);

    /// `ldc` 契約版（#557: 完全タイルの C 直接ロード/ストア）。`c` は要素
    /// `c[i * ldc + j]`（`i in 0..MR`・`j in 0..NR`）のみを読み書きする
    /// 対象とし、それ以外のインデックスへは触れない。この契約により、
    /// 呼び出し元（`super::gemm_blis_region`）は 2 通りの呼び出し方が
    /// できる:
    ///
    /// - 完全タイル（`mr_eff == MR && nr_eff == NR`）: C の実バッファから
    ///   タイル原点起点のサブスライスを直接渡し `ldc = n`（C の列数）と
    ///   する。コピーイン/コピーアウトが不要になる（#557 の主目的）
    /// - 端タイル: 従来どおり `MAX_TILE` スタックバッファの先頭
    ///   `MR*NR` 要素を渡し `ldc = NR` とする（現行の密パッキング契約は
    ///   `ldc = NR` の特殊ケースとして包含される）
    ///
    /// ## デフォルト実装（#691 再指摘への対応。公開 API 非破壊）
    ///
    /// 本メソッドを必須にすると [`Self::run`] のみを実装する既存の外部
    /// `Microkernel` 実装を破壊するため、デフォルト実装を設けて
    /// オーバーライド不要にする:
    ///
    /// - `ldc == Self::NR`（密パッキング契約）: 追加コピーなしで
    ///   [`Self::run`] へそのまま委譲する
    /// - `ldc != Self::NR`（#557 の直接 C 経路が使われるケース）:
    ///   [`Self::run`] は `ldc = NR` の密パッキングしか扱えないため、
    ///   ヒープ確保した `MR*NR` 要素のスクラッチタイルへ `c` の現在値を
    ///   `ldc` ストライドでギャザーし、[`Self::run`] を密パッキング契約で
    ///   呼んだ後、結果を `ldc` ストライドで `c` へスキャッタし直す
    ///   （正しさ優先のフォールバック。#557 が狙うコピー往復削減の効果は
    ///   本フォールバック経路には及ばないが、組み込みカーネル
    ///   （[`ScalarKernel`]／`NeonKernel`／`Avx2Kernel`／
    ///   `Avx512Kernel`）は全て本メソッドを直接オーバーライドしており
    ///   本番の駆動経路（`super::gemm_blis_region`）はこのフォールバック
    ///   を通らない）
    ///
    /// ## 型付きエラー化（#691 レビュー P1 再指摘への対応）
    ///
    /// `ldc`／`c` 長の契約検査（`check_c_tile_bounds`）は本メソッド・
    /// 各 ISA の `kernel_with_ldc`／`kernel_unchecked_with_ldc` から到達
    /// 可能な**公開**入口であり、外部の `Microkernel` 実装が誤った
    /// `ldc`／`c` を渡した場合に panic が本番経路から漏れていた
    /// （AGENTS.md「本番経路の panic 禁止」）。契約違反の早期検出という
    /// 目的（REQ-8 境界検査規約）自体は変えず、検出結果を
    /// `Result::Err(TileBoundsError)` として返す。[`Self::run`] は
    /// `ldc == Self::NR`（密パッキング契約）でのみ呼ばれ境界検査を経ない
    /// ため無変更（既存の必須メソッドのシグネチャは非破壊のまま）。
    ///
    /// ## `ap`／`bp` 長検査の追加（#691 レビュー再指摘 cursor
    /// `PRRT_kwDOTuUCJc6Zr8oU` への対応）
    ///
    /// 上記の型付きエラー化は `c`／`ldc` の検査のみを対象としており、
    /// `ap`／`bp` の長さ検査は含んでいなかった。本メソッドは `Result` を
    /// 返す公開入口である以上、各 ISA の `kernel_with_ldc`／
    /// `kernel_unchecked_with_ldc` と同様に `check_panel_lengths`（`ap`／
    /// `bp` 長不一致・オーバーフローを `TileBoundsError::
    /// PanelLengthMismatch` として拒否）も `check_c_tile_bounds` と併せて
    /// 実行する。
    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        check_panel_lengths(Self::MR, Self::NR, kc_len, ap.len(), bp.len())?;
        // #691 レビュー P1 再指摘（密パッキング経路が境界検査を迂回する）
        // への対応: `ldc == Self::NR` の高速経路も含め、[`Self::run`] へ
        // 委譲する前に必ず [`check_c_tile_bounds`] を通す。以前は
        // `ldc == Self::NR` の場合に検査より先に `run` へ委譲していたため、
        // 短い `c`（[`Self::MR`]／[`Self::NR`] の契約を満たさない `c.len()`）
        // や `Self::MR == 0` を渡す外部 `Microkernel` 実装が
        // `Result::Err(TileBoundsError)` を得られず、`run` 内部のスライス
        // 添字アクセスで検証不能な panic を起こしうる状態だった（REQ-8
        // 境界検査契約・AGENTS.md「本番経路の panic 禁止」）。
        check_c_tile_bounds(Self::MR, Self::NR, ldc, c.len())?;
        if ldc == Self::NR {
            // #691 レビュー P1 指摘（PRRT_kwDOTuUCJc6Zr-hh）への対応:
            // `check_c_tile_bounds` は密パッキング（`ldc == Self::NR`）でも
            // `c.len() >= Self::MR * Self::NR` しか要求しないため、`c` は
            // 検証済みタイル範囲より長いことがある。`Self::run` の契約は
            // 「ちょうど `MR * NR` の密な `c_tile`」であり、外部の
            // `Microkernel` 実装が組み込みカーネル同様に長さの完全一致を
            // 検査していると、余剰領域を含む `c` をそのまま渡した場合に
            // panic しうる（本番経路 panic 禁止。AGENTS.md）。検証済みの
            // 先頭 `MR * NR` 要素だけを切り出して渡す。
            self.run(ap, bp, &mut c[..Self::MR * Self::NR], kc_len);
            return Ok(());
        }
        let mut tile = vec![0.0f32; Self::MR * Self::NR];
        for i in 0..Self::MR {
            tile[i * Self::NR..(i + 1) * Self::NR].copy_from_slice(&c[i * ldc..i * ldc + Self::NR]);
        }
        self.run(ap, bp, &mut tile, kc_len);
        for i in 0..Self::MR {
            c[i * ldc..i * ldc + Self::NR].copy_from_slice(&tile[i * Self::NR..(i + 1) * Self::NR]);
        }
        Ok(())
    }
}

/// 全 arch 共通のスカラーフォールバックトークン。検出不要のため
/// `ScalarKernel` は常に構築可能（`Default` 相当の unit struct）。
#[derive(Clone, Copy)]
pub struct ScalarKernel;

impl Microkernel for ScalarKernel {
    const MR: usize = scalar::MR;
    const NR: usize = scalar::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // `run` は #691 レビュー再指摘（公開 API 非破壊）により従来どおり
        // 返り値を持たない必須メソッドとして維持しているため、境界検査
        // 違反を型付きエラーとして呼び出し元へ返せない。[`Self::run_with_ldc`]
        // （`Result` を返す `check_c_tile_bounds` 検査経由）へは委譲せず、
        // `ldc = Self::NR` 契約の [`scalar::kernel`] へ直接委譲する（#691
        // レビュー P1 再指摘 `PRRT_kwDOTuUCJc6ZrQZG` への対応: `Result` を
        // `panic!` へ変換する経路を持たない。`scalar::kernel` 自体は
        // #557 以前から通常の Rust スライス添字アクセスで完結しており、
        // 契約違反時は言語組み込みの範囲外添字 panic のまま — これは
        // `panic!("{e}")` によって新規に生んだ経路ではなく #557 以前と
        // 観測可能な挙動が同一）。
        scalar::kernel(ap, bp, c_tile, kc_len);
    }

    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        scalar::kernel_with_ldc(ap, bp, c, ldc, kc_len)
    }
}

/// aarch64 NEON トークン。NEON は baseline ISA のため実行時検出不要
/// （[`neon`] モジュールドキュメント参照）。
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct NeonKernel;

#[cfg(target_arch = "aarch64")]
impl Microkernel for NeonKernel {
    const MR: usize = neon::MR;
    const NR: usize = neon::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // [`ScalarKernel::run`] のドキュメント参照（`Result` を `panic!`
        // へ変換する経路を持たず [`neon::kernel`] へ直接委譲する）。
        neon::kernel(ap, bp, c_tile, kc_len);
    }

    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        neon::kernel_with_ldc(ap, bp, c, ldc, kc_len)
    }
}

/// aarch64 NEON 12×8（firestorm 型）A/B 対抗トークン。[`neon::kernel_12x8`]
/// と同じく NEON は baseline ISA のため実行時検出不要。`super::dispatch_region`
/// の駆動経路には接続せず、`gemm_blis::mod` の `#[cfg(test)]` A/B 計測
/// テスト（`super::gemm_blis_with_kernel` 経由）専用のトークン（#559）。
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct Neon12x8Kernel;

#[cfg(target_arch = "aarch64")]
impl Microkernel for Neon12x8Kernel {
    const MR: usize = neon::MR_12X8;
    const NR: usize = neon::NR_12X8;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        neon::kernel_12x8(ap, bp, c_tile, kc_len);
    }

    /// `run_with_ldc` の override（Cursor Bugbot 指摘・review 4947832636・
    /// thread PRRT_kwDOTuUCJc6Zq5PH への対応）。
    ///
    /// このトークンはオーバーライドせず [`Microkernel::run_with_ldc`] の
    /// デフォルト実装（`ldc != NR` でヒープ確保 `Vec` によるギャザー/
    /// スキャッタ）に頼っていたが、`super::gemm_blis_region` の完全タイル
    /// 経路（#557）は `ldc = n`（C の実列数）で常に `run_with_ldc` を呼ぶ
    /// ため、[`NeonKernel`]（8×12 側）が `run_with_ldc` を直接オーバー
    /// ライドして直接ロード/ストアで応じるのに対し、`Neon12x8Kernel`
    /// （12×8 側）のみ full-tile 呼び出しのたびに `MR*NR`（=96 要素）タイル
    /// を毎回ヒープ確保するという非対称が生じていた。本トークンは
    /// `super::dispatch_region` の駆動経路には接続せず `gemm_blis::mod` の
    /// `#[cfg(test)]` A/B 計測テスト（8×12 vs 12×8 のスループット比較。
    /// #559）専用のため、正当性への影響はなかったが、比較対象の
    /// `NeonKernel` 側だけコピー往復が省かれヒープ確保も無い一方
    /// `Neon12x8Kernel` 側は毎呼び出しヒープ確保が乗るため、A/B 比較の
    /// 公平性が損なわれていた。
    ///
    /// [`neon::kernel_12x8`] 自体は `ldc` 一般化（[`neon::kernel_with_ldc`]
    /// 相当の strided ロード/ストア）を持たないため、ここではスタック
    /// 固定長バッファ（ヒープ確保なし）へのギャザー/スキャッタで
    /// デフォルト実装と同じ正当性を保ちつつ `Vec` 確保のみを除去する
    /// （A/B 計測の対称性回復が目的であり、`NeonKernel` と同水準の
    /// strided 直接アクセスへ揃えるほどの追加実装コストは、本番駆動
    /// 経路に接続しないテスト専用トークンには見合わないと判断した）。
    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        // #691 レビュー P1 再指摘（[`Microkernel::run_with_ldc`] デフォルト
        // 実装の同種修正参照）: `ldc == Self::NR` の高速経路も含め、`run`
        // へ委譲する前に必ず境界検査を通す。`ap`／`bp` 長検査
        // （[`check_panel_lengths`]）は #691 レビュー再指摘 cursor
        // `PRRT_kwDOTuUCJc6Zr8oU` への対応で追加した（デフォルト実装の
        // 同種修正参照）。
        check_panel_lengths(Self::MR, Self::NR, kc_len, ap.len(), bp.len())?;
        check_c_tile_bounds(Self::MR, Self::NR, ldc, c.len())?;
        if ldc == Self::NR {
            // #691 レビュー P1 指摘（PRRT_kwDOTuUCJc6Zr-hh）と同種の修正
            // （[`Microkernel::run_with_ldc`] デフォルト実装コメント参照）:
            // `check_c_tile_bounds` は `c.len() >= Self::MR * Self::NR` しか
            // 要求せず、`c` は検証済みタイル範囲より長いことがある。`run`
            // の契約はちょうど `MR * NR` の密な `c_tile` のため、検証済み
            // の先頭 `MR * NR` 要素だけを切り出して渡す。
            self.run(ap, bp, &mut c[..Self::MR * Self::NR], kc_len);
            return Ok(());
        }
        // ヒープ確保（`Vec`）を避けるため MR_12X8*NR_12X8（=96）固定長の
        // スタック配列を使う（`super::MAX_TILE`〈256〉以内。デフォルト
        // 実装との唯一の差分はここのみで、ギャザー/スキャッタのロジック
        // 自体は同一）。
        let mut tile = [0.0f32; neon::MR_12X8 * neon::NR_12X8];
        for i in 0..Self::MR {
            tile[i * Self::NR..(i + 1) * Self::NR].copy_from_slice(&c[i * ldc..i * ldc + Self::NR]);
        }
        neon::kernel_12x8(ap, bp, &mut tile, kc_len);
        for i in 0..Self::MR {
            c[i * ldc..i * ldc + Self::NR].copy_from_slice(&tile[i * Self::NR..(i + 1) * Self::NR]);
        }
        Ok(())
    }
}

/// aarch64 NEON B 側レーン参照 FMA 変種トークン（イシュー #748）。
/// [`neon::kernel_b_laneq_with_ldc`] へ委譲する。MR/NR は [`NeonKernel`]
/// （既定 8×12）と同一だが、`vfmaq_laneq_f32` のレーン参照オペランドを
/// B 側に割り当てアキュムレータを列優先で保持する点が異なる
/// （[`neon`] モジュール冒頭 #748 節参照）。[`Neon12x8Kernel`] で判明した
/// A/B 公平性問題（ヒープ確保の非対称。同トークンのドキュメント参照）を
/// 避けるため、当初から strided 直接ロード/ストア経路
/// （[`neon::kernel_b_laneq_with_ldc`]）を `run_with_ldc` で直接呼び、
/// デフォルト実装（ヒープ確保するギャザー/スキャッタ）に頼らない。
/// `super::dispatch_region` の既定駆動経路には接続せず、`gemm_blis::mod`
/// の `#[cfg(test)]` A/B 計測テスト専用（#748 実装計画。実機での bit
/// 一致・非劣化確認後に既定接続を判断する fail-closed 方針）。
///
/// ## `#[cfg(test)]` 限定（PR #765 codex-review P1 対応）
///
/// 上記のとおり本トークンは `gemm_blis::mod` の `#[cfg(test)]` テストからのみ
/// 使われ、本番駆動経路（`Isa::detect` 経由の実行時 dispatch）には一切
/// 接続されない。にもかかわらず型・`impl`（および委譲先の
/// [`neon::kernel_b_laneq`]）が本番ビルドへコンパイルされていると、
/// `run` が `assert!`/`assert_eq!` で `panic!` する経路を本番コードが
/// 抱える形になり「本番経路の panic 禁止」規約
/// （`.claude/rules/coding-rust.md`・AGENTS.md）の対象として誤認・誤用
/// されうる。テスト専用の性質をコンパイル単位でも保証するため
/// `#[cfg(test)]` を追加する。本番で当該変種を使う場合は
/// [`neon::kernel_b_laneq_with_ldc`]（`Result` を返す境界検査版）を
/// 直接呼ぶ設計へ変更すること（`assert!` 版へ戻さない）。
#[cfg(all(target_arch = "aarch64", test))]
#[derive(Clone, Copy)]
pub struct NeonBLaneqKernel;

#[cfg(all(target_arch = "aarch64", test))]
impl Microkernel for NeonBLaneqKernel {
    const MR: usize = neon::MR;
    const NR: usize = neon::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // [`ScalarKernel::run`] のドキュメント参照（`Result` を `panic!`
        // へ変換する経路を持たず、`assert!` 検査版の
        // [`neon::kernel_b_laneq`] へ直接委譲する）。両者とも
        // `#[cfg(test)]` 限定（本コメント冒頭参照）。
        neon::kernel_b_laneq(ap, bp, c_tile, kc_len);
    }

    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        neon::kernel_b_laneq_with_ldc(ap, bp, c, ldc, kc_len)
    }
}

/// aarch64 NEON B 側 laneq ベクトル転置版トークン（イシュー #1317）。
/// [`NeonBLaneqKernel`]（#748・スカラー gather/scatter 転置）の C タイル
/// 転置を `neon::transpose_4x4` によるベクトル化転置へ置き換えた候補で、
/// [`neon::kernel_b_laneq_vec_with_ldc`] へ委譲する。MR/NR・累積契約は
/// [`NeonBLaneqKernel`] と同一（[`neon`] モジュール冒頭 #1317 節参照）。
/// [`NeonBLaneqKernel`] と同じ理由（[`Neon12x8Kernel`] のヒープ確保
/// 非対称問題を避ける）で strided 直接ロード/ストア経路を `run_with_ldc`
/// で直接呼び、デフォルト実装のヒープ確保ギャザー/スキャッタに頼らない。
/// `super::dispatch_region` の既定駆動経路には接続せず、`gemm_blis::mod`
/// の `#[cfg(test)]` A/B 計測テスト専用（採否・実機実測は #1318 へ
/// 引き継ぐ）。
///
/// ## `#[cfg(test)]` 限定
///
/// [`NeonBLaneqKernel`] と同じ理由（本番ビルドの到達可能経路を持たない
/// トークンに対応する `assert!` 検査版委譲先を本番へコンパイルさせない
/// ため）で `#[cfg(test)]` を付ける。
#[cfg(all(target_arch = "aarch64", test))]
#[derive(Clone, Copy)]
pub struct NeonBLaneqVecKernel;

#[cfg(all(target_arch = "aarch64", test))]
impl Microkernel for NeonBLaneqVecKernel {
    const MR: usize = neon::MR;
    const NR: usize = neon::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        neon::kernel_b_laneq_vec(ap, bp, c_tile, kc_len);
    }

    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        neon::kernel_b_laneq_vec_with_ldc(ap, bp, c, ldc, kc_len)
    }
}

/// aarch64 SME（Scalable Matrix Extension）`fmopa` トークン（イシュー
/// #1587）。`SmeKernel::try_new` 経由でのみ構築でき、これが「**構築した
/// スレッド**の実行 CPU が SME・非拡張 FP32 外積（`SME_F32F32`）に対応し
/// SVL=512 bit である」ことを保証する（`Avx2Kernel`〈x86_64 限定のため
/// コードスパン表記〉と同型の「検出済みトークンのみ構築可能」パターン）。
///
/// ## SVL はスレッドごとに異なりうる（codex-review P0 指摘
/// `PRRT_kwDOTuUCJc6h0P7Y` 対応）
///
/// Arm SME の SVL は Linux では `prctl(PR_SME_SET_VL)` でスレッドごとに
/// 変更可能であり、`Avx2Kernel`／`Avx512Kernel` の CPUID ベース検出
/// （プロセス内で不変）とは異なりプロセス全体で不変とは限らない。本
/// トークンは `Copy` のため構築したスレッドとは別のスレッド（Rayon
/// worker 等）へそのまま渡されうるが、[`Microkernel::run`]／
/// [`Microkernel::run_with_ldc`] は **実際に `fmopa` を発行するスレッド
/// 自身で [`crate::sme_detect::sme_report`] を呼び直し、SVL がその
/// スレッド上でも要求値と一致することを確認してから**
/// `unsafe { sme::kernel_unchecked… }` を呼ぶ。一致しない場合は
/// `panic!` で停止する（本カーネルは MR=16×NR=16 固定でパックされた
/// 入力を前提とし、MR=8×NR=12 の [`NeonKernel`] 等へ実行時に安全に
/// 差し替えることはできない〈パック済みバッファの形状が食い違う〉ため、
/// 差し替えフォールバックではなく `unsafe` 呼び出し自体を行わない
/// fail-closed を採る。`.claude/rules/security.md` の fail-closed 方針）。
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct SmeKernel {
    /// 外部からの直接構築を禁止する非公開フィールド（`Avx2Kernel`〈x86_64
    /// 限定のためコードスパン表記〉と同じ封止パターン）。
    _private: (),
}

#[cfg(target_arch = "aarch64")]
impl SmeKernel {
    /// 実行 CPU が SME・非拡張 FP32 外積（`SME_F32F32`）に対応し
    /// SVL=512 bit（64 バイト）の場合のみ `Some` を返す
    /// （[`crate::sme_detect::sme_report`] が fail-closed に判定する。
    /// モジュール doc「検出との関係」節参照）。**この判定は呼び出した
    /// スレッド上でのみ有効**（構造体 doc「SVL はスレッドごとに異なり
    /// うる」節参照）であり、`run`／`run_with_ldc` は実行スレッド自身で
    /// 再確認する。
    pub(crate) fn try_new() -> Option<Self> {
        if crate::sme_detect::sme_report().kernel_enabled {
            Some(Self { _private: () })
        } else {
            None
        }
    }

    /// `run`／`run_with_ldc` が `unsafe` 呼び出し直前に共通で行う
    /// **実行スレッド自身の** SVL 再確認（構造体 doc 参照）。
    /// `sme_detect::sme_report()` の SVL 読み取り部分はキャッシュせず
    /// 毎回 `rdsvl`（メモリアクセスを伴わない読み取り専用の 1 命令）を
    /// 発行するため、呼び出しのたびに再検証しても計測に有意な影響を
    /// 与えない（`sme_detect` モジュール doc 参照）。
    ///
    /// ## panic ではなく bool を返す（codex-review P1 再指摘
    /// `PRRT_kwDOTuUCJc6h0ZMD` への対応）
    ///
    /// 以前は本メソッドが `assert!` で判定していたため、`SmeKernel` を
    /// 構築したスレッドと実際に `fmopa` を発行するスレッド（Rayon
    /// worker 等）の SVL が異なる場合、正常な形状の入力でも panic して
    /// いた。`run_with_ldc` は `Result` を返す入口である一方 `run` は
    /// トレイトの必須メソッド（`#691` レビュー再指摘により非破壊のため
    /// 非 `Result`。[`Microkernel::run`] doc 参照）で `Result` 化でき
    /// ないため、両者で共通に扱えるよう本メソッド自体は `bool` を返す
    /// 判定のみに留め、呼び出し元（`run`／`run_with_ldc`）が非対応時に
    /// [`sme::scalar_fallback_kernel`]／[`sme::scalar_fallback_with_ldc`]
    /// （`compute` と同一の演算列を安全な Rust で再現し、有限値入力で
    /// bit 完全一致するフォールバック。`sme` モジュール該当関数 doc
    /// 参照）へ切り替える。`.claude/rules/security.md`／AGENTS.md
    /// 「本番経路の panic 禁止」への抵触を解消しつつ、実行スレッドでの
    /// SVL 再確認自体は維持する。
    fn current_thread_capable(&self) -> bool {
        crate::sme_detect::sme_report().kernel_enabled
    }
}

#[cfg(target_arch = "aarch64")]
impl Microkernel for SmeKernel {
    const MR: usize = sme::MR;
    const NR: usize = sme::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // [`ScalarKernel::run`] のドキュメント参照（`Result` を `panic!`
        // へ変換する経路を持たず、非対応時は
        // [`sme::scalar_fallback_kernel`] へ委譲する。`current_thread_capable`
        // doc 参照）。
        if self.current_thread_capable() {
            // SAFETY: `current_thread_capable` が **この呼び出しを実行
            // しているスレッド自身**で SME 対応・SVL=64 バイトを確認済み
            // （`SmeKernel` 構造体 doc「SVL はスレッドごとに異なりうる」
            // 節。`Self` が try_new() 経由でのみ構築可能という事実だけ
            // では、構築スレッドと実行スレッドが異なりうる Rayon worker
            // 分配のもとでは不十分なため、ここで実行スレッド自身の確認
            // を必須とする）。`sme::kernel_unchecked` の `# Safety` 契約
            // を満たす。
            unsafe { sme::kernel_unchecked(ap, bp, c_tile, kc_len) }
        } else {
            sme::scalar_fallback_kernel(ap, bp, c_tile, kc_len);
        }
    }

    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        if self.current_thread_capable() {
            // SAFETY: `run` と同じ理由で `current_thread_capable` を
            // 実行スレッド自身で必ず確認してから呼ぶ
            // （`sme::kernel_unchecked_with_ldc` の `# Safety` 契約を
            // 満たす）。
            unsafe { sme::kernel_unchecked_with_ldc(ap, bp, c, ldc, kc_len) }
        } else {
            sme::scalar_fallback_with_ldc(ap, bp, c, ldc, kc_len)
        }
    }
}

/// x86_64 AVX2+FMA トークン。`Avx2Kernel::try_new` 経由でのみ構築でき、
/// これが実行 CPU の AVX2+FMA 対応を保証する（[`Microkernel::run`] 内部の
/// `unsafe { avx2::kernel_unchecked(...) }` の SAFETY 根拠）。
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Avx2Kernel {
    /// 外部からの直接構築を禁止する非公開フィールド（[`Avx2Kernel::try_new`]
    /// 経由の検出済み構築のみを許可するための封止）。
    _private: (),
}

#[cfg(target_arch = "x86_64")]
impl Avx2Kernel {
    /// 実行 CPU が AVX2+FMA をサポートする場合のみ `Some` を返す。この
    /// 判定こそが [`Microkernel::run`] 内 `unsafe` 呼び出しの安全根拠。
    pub(crate) fn try_new() -> Option<Self> {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Some(Self { _private: () })
        } else {
            None
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Microkernel for Avx2Kernel {
    const MR: usize = avx2::MR;
    const NR: usize = avx2::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // [`ScalarKernel::run`] のドキュメント参照（`Result` を `panic!`
        // へ変換する経路を持たず [`avx2::kernel_unchecked`] へ直接委譲する）。
        //
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx2") && ("fma") を確認済み
        // （avx2::kernel_unchecked の `# Safety` 契約を満たす）。
        unsafe { avx2::kernel_unchecked(ap, bp, c_tile, kc_len) }
    }

    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx2") && ("fma") を確認済み
        // （avx2::kernel_unchecked_with_ldc の `# Safety` 契約を満たす）。
        unsafe { avx2::kernel_unchecked_with_ldc(ap, bp, c, ldc, kc_len) }
    }
}

/// x86_64 AVX-512F トークン。`Avx512Kernel::try_new` 経由でのみ構築でき、
/// これが実行 CPU の AVX-512F 対応を保証する。`avx512_stable` cfg
/// （[`avx512`] モジュールドキュメント参照）が立っている rustc（AVX-512F
/// intrinsics が stable 化済みと `build.rs` の probe が確認できた場合）
/// でのみコンパイル対象となる。
#[cfg(all(target_arch = "x86_64", avx512_stable))]
#[derive(Clone, Copy)]
pub struct Avx512Kernel {
    /// [`Avx2Kernel`] と同じ封止パターン。
    _private: (),
}

#[cfg(all(target_arch = "x86_64", avx512_stable))]
impl Avx512Kernel {
    /// 実行 CPU が AVX-512F をサポートする場合のみ `Some` を返す。
    pub(crate) fn try_new() -> Option<Self> {
        if is_x86_feature_detected!("avx512f") {
            Some(Self { _private: () })
        } else {
            None
        }
    }
}

#[cfg(all(target_arch = "x86_64", avx512_stable))]
impl Microkernel for Avx512Kernel {
    const MR: usize = avx512::MR;
    const NR: usize = avx512::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // [`ScalarKernel::run`] のドキュメント参照（`Result` を `panic!`
        // へ変換する経路を持たず [`avx512::kernel_unchecked`] へ直接
        // 委譲する）。
        //
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx512f") を確認済み
        // （avx512::kernel_unchecked の `# Safety` 契約を満たす）。
        unsafe { avx512::kernel_unchecked(ap, bp, c_tile, kc_len) }
    }

    fn run_with_ldc(
        &self,
        ap: &[f32],
        bp: &[f32],
        c: &mut [f32],
        ldc: usize,
        kc_len: usize,
    ) -> Result<(), TileBoundsError> {
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx512f") を確認済み
        // （avx512::kernel_unchecked_with_ldc の `# Safety` 契約を満たす）。
        unsafe { avx512::kernel_unchecked_with_ldc(ap, bp, c, ldc, kc_len) }
    }
}

/// 実行時に選択された ISA を表す列挙型。[`super::gemm_blis`]／
/// [`super::gemm_blis_parallel`] の公開入口が [`Isa::detect`] の結果で
/// 1 回だけ match し、モノモーフィック化された `gemm_blis_region::<K>`
/// へ分岐する（`K` は対応する ISA トークン型）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    Neon,
    Avx2,
    Avx512,
}

/// 検出結果（bool）から選択する ISA を決める純関数。優先順位は
/// Avx512 > Avx2 > Scalar（x86_64）。aarch64 は常に Neon、その他 arch は
/// 常に Scalar（呼び出し元の cfg 分岐が担保する。[`select_isa`] 自体は
/// arch に依存しない純ロジックとして単体テスト可能にする）。
///
/// x86_64 の [`Isa::detect_uncached`] からのみ本番経路で呼ばれる。他 arch
/// では駆動経路から外れるため `cfg(any(target_arch = "x86_64", test))` で
/// dead_code 警告を避けつつ、単体テストは arch に依らず実行できるように
/// 残す（`cargo check --target aarch64-unknown-linux-gnu` のクロス検証で
/// 未使用警告が出ないようにするための cfg）。
#[cfg(any(target_arch = "x86_64", test))]
fn select_isa(has_avx2_fma: bool, has_avx512f: bool) -> Isa {
    if has_avx512f {
        Isa::Avx512
    } else if has_avx2_fma {
        Isa::Avx2
    } else {
        Isa::Scalar
    }
}

impl Isa {
    /// プロセス内で 1 回だけ実行 CPU の機能検出を行い、以降は結果を
    /// キャッシュする（`is_x86_feature_detected!` 自体も std 内部で
    /// キャッシュされるが、dispatch 判定を 1 箇所に固定する意図で
    /// `OnceLock` を用いる）。
    pub fn detect() -> Isa {
        static ISA: OnceLock<Isa> = OnceLock::new();
        *ISA.get_or_init(Self::detect_uncached)
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_uncached() -> Isa {
        Isa::Neon
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_uncached() -> Isa {
        let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        // `avx512_stable` cfg が立っていない rustc では `Avx512Kernel` 自体が
        // コンパイル対象外となり `dispatch_region`（`super` モジュール）は
        // AVX-512F 実行 CPU 上でも AVX2/scalar へフォールバックする
        // （`build.rs` のコメント・本モジュール冒頭のドキュメント参照）。
        // ここで実行 CPU の avx512f 対応を無条件採用すると、この
        // introspection API（`Isa::detect`）が実際の dispatch 結果と
        // 食い違う（Bugbot 指摘: PR #337 review 4886262265・comment
        // 3738511491）。`avx512_stable` 未設定時は has_avx512f を常に
        // false とし、実際の dispatch 経路と一致させる。
        #[cfg(avx512_stable)]
        let has_avx512f = is_x86_feature_detected!("avx512f");
        #[cfg(not(avx512_stable))]
        let has_avx512f = false;
        select_isa(has_avx2_fma, has_avx512f)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn detect_uncached() -> Isa {
        Isa::Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `run` のみをオーバーライドする「クレート外部実装」を模したトークン
    /// （#691 再指摘への回帰テスト）。`Microkernel::run_with_ldc` は
    /// デフォルト実装のみに頼り、MR=2・NR=2 の単純な mul_add 累積を
    /// 密パッキング（`ldc == NR`）契約でのみ行う。
    #[derive(Clone, Copy)]
    struct LegacyRunOnlyKernel;

    impl Microkernel for LegacyRunOnlyKernel {
        const MR: usize = 2;
        const NR: usize = 2;

        fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
            for i in 0..Self::MR {
                for j in 0..Self::NR {
                    let mut acc = c_tile[i * Self::NR + j];
                    for p in 0..kc_len {
                        acc = ap[p * Self::MR + i].mul_add(bp[p * Self::NR + j], acc);
                    }
                    c_tile[i * Self::NR + j] = acc;
                }
            }
        }
    }

    /// `run` のみを実装するレガシー実装は、密パッキング契約（`ldc == NR`）
    /// では追加コピーなしで `run_with_ldc` のデフォルト実装から呼び出せる
    /// （#691 再指摘: 公開 API 非破壊の核心シナリオ）。
    #[test]
    fn run_with_ldc_default_impl_delegates_to_run_when_ldc_equals_nr() {
        let k = LegacyRunOnlyKernel;
        let ap = [1.0f32, 2.0, 3.0, 4.0]; // kc_len=2, MR=2 (p-major)
        let bp = [5.0f32, 6.0, 7.0, 8.0]; // kc_len=2, NR=2 (p-major)
        let mut via_run = [0.0f32; 4];
        k.run(&ap, &bp, &mut via_run, 2);

        let mut via_default_ldc = [0.0f32; 4];
        k.run_with_ldc(&ap, &bp, &mut via_default_ldc, 2, 2)
            .unwrap();

        assert_eq!(
            via_run, via_default_ldc,
            "ldc == NR では run へ委譲するはず"
        );
    }

    /// `c_tile.len() == Self::MR * Self::NR`（ちょうど密なタイル長）を
    /// 検査する「クレート外部実装」を模したトークン（#691 レビュー P1
    /// 指摘 `PRRT_kwDOTuUCJc6Zr-hh` への回帰テスト専用）。`run` の従来
    /// 契約はちょうど `MR * NR` の密な `c_tile` であるため、この検査自体
    /// は契約に忠実な実装として正当であり、`run_with_ldc` のデフォルト
    /// 実装側が余剰領域を含む `c` を切り詰めずに渡すと panic する。
    #[derive(Clone, Copy)]
    struct ExactLenAssertingKernel;

    impl Microkernel for ExactLenAssertingKernel {
        const MR: usize = 2;
        const NR: usize = 2;

        fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
            assert_eq!(
                c_tile.len(),
                Self::MR * Self::NR,
                "run は MR*NR ぴったりの密な c_tile のみを受け取る契約"
            );
            for i in 0..Self::MR {
                for j in 0..Self::NR {
                    let mut acc = c_tile[i * Self::NR + j];
                    for p in 0..kc_len {
                        acc = ap[p * Self::MR + i].mul_add(bp[p * Self::NR + j], acc);
                    }
                    c_tile[i * Self::NR + j] = acc;
                }
            }
        }
    }

    /// 密パッキング（`ldc == NR`）でも `check_c_tile_bounds` は
    /// `c.len() >= MR * NR` しか要求しないため、`c` がタイル長より長い
    /// 場合に `run_with_ldc` のデフォルト実装が余剰領域を切り詰めずに
    /// `run` へ渡すと、長さ完全一致を検査する外部実装で panic しうる
    /// （#691 レビュー P1 指摘 `PRRT_kwDOTuUCJc6Zr-hh`）。検証済みの先頭
    /// `MR * NR` 要素だけが渡ることを確認する。
    #[test]
    fn run_with_ldc_default_impl_slices_tile_exactly_for_dense_packing_with_excess_len() {
        let k = ExactLenAssertingKernel;
        let ap = [1.0f32, 2.0, 3.0, 4.0];
        let bp = [5.0f32, 6.0, 7.0, 8.0];

        // MR*NR(=4) より長い c（末尾に余剰領域を含む）を ldc == NR で渡す。
        let mut c = [0.0f32, 0.0, 0.0, 0.0, 9.0, 9.0];
        k.run_with_ldc(&ap, &bp, &mut c, 2, 2).unwrap();

        let mut expected_tile = [0.0f32; 4];
        k.run(&ap, &bp, &mut expected_tile, 2);
        assert_eq!(
            &c[..4],
            &expected_tile[..],
            "タイル範囲の計算結果は run と一致するはず"
        );
        assert_eq!(
            &c[4..],
            &[9.0, 9.0],
            "検証済みタイル範囲外の余剰領域は変更されないはず"
        );
    }

    /// `run` のみを実装するレガシー実装でも、`ldc != NR`（#557 の直接 C
    /// 経路が使うストライド）で呼ばれた場合はギャザー/スキャッタの
    /// フォールバックにより正しい結果を返す（正しさ優先。性能は
    /// 組み込みカーネルほど出ないがコンパイル不能にはならない）。
    #[test]
    fn run_with_ldc_default_impl_gather_scatter_fallback_matches_dense_result() {
        let k = LegacyRunOnlyKernel;
        let ap = [1.0f32, 2.0, 3.0, 4.0];
        let bp = [5.0f32, 6.0, 7.0, 8.0];

        // 密パッキング（ldc = NR = 2）で得られる期待値。
        let mut dense = [0.0f32; 4];
        k.run_with_ldc(&ap, &bp, &mut dense, 2, 2).unwrap();

        // ldc = 3 の広い C バッファ（行間に 1 要素のギャップ）へ同じ演算を行う。
        // 初期値は 0 埋めなので dense と同じ結果になるはず。
        let mut strided = [0.0f32; 6];
        k.run_with_ldc(&ap, &bp, &mut strided, 3, 2).unwrap();

        assert_eq!(strided[0], dense[0]);
        assert_eq!(strided[1], dense[1]);
        assert_eq!(strided[3], dense[2]);
        assert_eq!(strided[4], dense[3]);
        // ギャップ列（各行末尾）には触れない契約であるはず。
        assert_eq!(strided[2], 0.0);
        assert_eq!(strided[5], 0.0);
    }

    /// [`panel_len_matches`] は通常の一致・不一致を `usize` 乗算どおりに
    /// 判定する（オーバーフロー無関係の基本ケース）。
    #[test]
    fn panel_len_matches_basic_cases() {
        assert!(panel_len_matches(12, 3, 4));
        assert!(!panel_len_matches(11, 3, 4));
    }

    /// [`panel_len_matches`] は `factor_a * factor_b` がオーバーフローする
    /// 組み合わせを必ず不一致として扱う（#691 レビュー P0 再指摘
    /// `PRRT_kwDOTuUCJc6ZrXKs` への回帰テスト）。指摘の具体例（NEON の
    /// `MR=8`・`kc_len = 1 << (usize::BITS - 2)`）に基づき、素朴な `usize`
    /// 乗算では `8 * 2^62` が `usize::MAX` を超えラップし `0` になって
    /// 空の `ap`（`actual=0`）と一致してしまうケースを直接検証する。
    #[test]
    fn panel_len_matches_rejects_overflowing_factors() {
        let huge_kc_len = 1usize << (usize::BITS - 2);
        assert!(!panel_len_matches(0, 8, huge_kc_len));
        assert!(!panel_len_matches(usize::MAX, huge_kc_len, huge_kc_len));
    }

    /// [`check_panel_lengths`] は ap/bp 長不一致・オーバーフローいずれも
    /// panic ではなく `Result::Err(TileBoundsError::PanelLengthMismatch)`
    /// として返す（#691 レビュー P0 再指摘 `PRRT_kwDOTuUCJc6ZrXKs` への
    /// 対応: NEON `kernel_with_ldc` 等の公開 `*_with_ldc` 入口はこの関数
    /// 経由で ap/bp 長を検証する）。
    #[test]
    fn check_panel_lengths_rejects_ap_and_bp_mismatch() {
        assert_eq!(
            check_panel_lengths(2, 2, 2, 3, 4),
            Err(TileBoundsError::PanelLengthMismatch {
                panel: "ap",
                actual: 3
            })
        );
        assert_eq!(
            check_panel_lengths(2, 2, 2, 4, 3),
            Err(TileBoundsError::PanelLengthMismatch {
                panel: "bp",
                actual: 3
            })
        );
        assert_eq!(check_panel_lengths(2, 2, 2, 4, 4), Ok(()));
    }

    /// `mr == 0`（[`Microkernel::MR`] の契約違反）を [`check_c_tile_bounds`]
    /// が意図の分かる型付きエラーで検出することの回帰テスト（PR #691
    /// レビュー指摘 `PRRT_kwDOTuUCJc6Zq7vw`）。修正前は `mr - 1` の減算
    /// オーバーフロー由来の panic（debug ビルド）だったが、
    /// `mr.checked_sub(1)` により意図した
    /// `TileBoundsError::NonPositiveMrOrOverflow` へ変わったことを確認
    /// する（さらに #691 P1 再指摘対応で panic ではなく `Result::Err` に
    /// なった）。
    #[test]
    fn check_c_tile_bounds_rejects_mr_zero_with_explicit_message() {
        assert_eq!(
            check_c_tile_bounds(0, 2, 2, 4),
            Err(TileBoundsError::NonPositiveMrOrOverflow { mr: 0, ldc: 2 })
        );
    }

    /// `nr == 0`（[`Microkernel::NR`] の契約違反）を [`check_c_tile_bounds`]
    /// が明示的に拒否することの回帰テスト（PR #691 レビュー P0 再指摘
    /// `PRRT_kwDOTuUCJc6ZrQZE`／`PRRT_kwDOTuUCJc6ZrQZG`）。修正前は
    /// `mr=1, nr=0, ldc=0, c_len=0` が `ldc < nr` 判定（`usize` の恒偽）を
    /// 素通りし `Ok(())` を返していた（境界検査なしで `Self::run` へ到達
    /// しうる REQ-8 違反）。
    #[test]
    fn check_c_tile_bounds_rejects_nr_zero_with_explicit_message() {
        assert_eq!(
            check_c_tile_bounds(1, 0, 0, 0),
            Err(TileBoundsError::NonPositiveNr { nr: 0 })
        );
    }

    /// [`check_c_tile_bounds`] が検出する境界違反は、[`Microkernel::run_with_ldc`]
    /// の公開入口（外部の `Microkernel` 実装からも到達可能）まで panic せず
    /// `Result::Err` として伝播することの回帰テスト（#691 レビュー P1
    /// 再指摘 `PRRT_kwDOTuUCJc6Zq_7u` への対応）。
    #[test]
    fn run_with_ldc_returns_err_instead_of_panicking_on_ldc_too_small() {
        let k = LegacyRunOnlyKernel;
        let ap = [1.0f32, 2.0, 3.0, 4.0];
        let bp = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        // ldc = 1 < NR(=2) は境界検査違反だが、panic せず Err を返す。
        let err = k.run_with_ldc(&ap, &bp, &mut c, 1, 2).unwrap_err();
        assert_eq!(err, TileBoundsError::LdcTooSmall { ldc: 1, nr: 2 });
    }

    /// [`check_c_tile_bounds`] の `c` バッファ不足検出が `Result::Err`
    /// として返ることの回帰テスト（#691 レビュー P1 再指摘への対応）。
    #[test]
    fn check_c_tile_bounds_rejects_c_buffer_too_small() {
        assert_eq!(
            check_c_tile_bounds(2, 2, 3, 4),
            Err(TileBoundsError::CBufferTooSmall {
                required: 5,
                actual: 4
            })
        );
    }

    #[test]
    fn select_isa_prefers_avx512_over_avx2() {
        assert_eq!(select_isa(true, true), Isa::Avx512);
    }

    #[test]
    fn select_isa_prefers_avx2_over_scalar() {
        assert_eq!(select_isa(true, false), Isa::Avx2);
    }

    #[test]
    fn select_isa_falls_back_to_scalar() {
        assert_eq!(select_isa(false, false), Isa::Scalar);
    }

    #[test]
    fn select_isa_avx512_alone_selects_avx512() {
        assert_eq!(select_isa(false, true), Isa::Avx512);
    }

    /// [`Isa::detect`] は実行環境に依らず必ず何らかの ISA を返す
    /// （panic しないこと自体が契約。実測値は環境依存のため固定しない）。
    #[test]
    fn isa_detect_returns_consistent_result() {
        let first = Isa::detect();
        let second = Isa::detect();
        assert_eq!(
            first, second,
            "OnceLock キャッシュにより結果は不変であるはず"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernel_try_new_matches_feature_detection() {
        let expected = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        assert_eq!(Avx2Kernel::try_new().is_some(), expected);
    }

    /// [`SmeKernel::try_new`] は [`crate::sme_detect::sme_report`] の
    /// `kernel_enabled` 判定と一致する（イシュー #1587。`avx2` 版と
    /// 同型の回帰テスト。SME 対応・非対応いずれの実行環境でも
    /// 意味のある表明になる）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sme_kernel_try_new_matches_feature_detection() {
        let expected = crate::sme_detect::sme_report().kernel_enabled;
        assert_eq!(SmeKernel::try_new().is_some(), expected);
    }

    #[cfg(all(target_arch = "x86_64", avx512_stable))]
    #[test]
    fn avx512_kernel_try_new_matches_feature_detection() {
        let expected = is_x86_feature_detected!("avx512f");
        assert_eq!(Avx512Kernel::try_new().is_some(), expected);
    }

    /// `avx512_stable` cfg 未設定の rustc では `Avx512Kernel` 自体が
    /// コンパイル対象外となり `super::dispatch_region` は実行 CPU の
    /// avx512f 対応に関わらず AVX2/scalar のみを試す。この条件下で
    /// `Isa::detect` が `Isa::Avx512` を返すと、実際の dispatch と
    /// introspection API（`Isa::detect`）が食い違う（PR #337 review
    /// 4886262265・comment 3738511491 の Bugbot 指摘）。本テストは
    /// その食い違いへの回帰を検知する。
    #[cfg(all(target_arch = "x86_64", not(avx512_stable)))]
    #[test]
    fn isa_detect_never_reports_avx512_when_not_stable() {
        assert_ne!(Isa::detect(), Isa::Avx512);
    }
}
