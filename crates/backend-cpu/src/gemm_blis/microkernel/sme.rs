//! aarch64 SME（Scalable Matrix Extension）マイクロカーネル
//! （MR=16×NR=16、`fmopa` 非拡張 FP32 外積・ZA0 単一タイル。イシュー #1587）。
//!
//! **モジュールは `cfg(target_arch = "aarch64")` のみでコンパイルする**
//! （`super::avx2`〈x86_64 限定のためコードスパン表記〉と同じ理由: モジュール単位で追加条件を課すと、
//! テスト限定の実行時検出ガード付き直接検証が行えなくなるため）。実際に
//! SME 命令を発行する `compute`（非公開関数のためコードスパン表記）は
//! 「呼び出し元が実行 CPU の SME 対応（`super::SmeKernel::try_new`。
//! `pub(crate)` のためコードスパン表記）経由の実行時検出）を保証する」契約の
//! `unsafe fn` とし、コンパイル時 `target_feature` によるゲートは行わない
//! （SME はコンパイラ intrinsics ではなく生アセンブリで発行するため、
//! `#[target_feature(enable = "sme")]` は不要かつ rustc stable では
//! 認識されない）。
//!
//! ## 設計判断: MR=16×NR=16・ZA0 単一タイル
//!
//! SVL=512 bit（f32 ベクトルレジスタ 1 本 = 16 要素）環境では ZA0〜ZA3 の
//! 4 タイルを使う MR=32×NR=32 案（2×2 外積ブロック）も可能だが、本実装は
//! `super::super::MAX_TILE`（非公開定数のためコードスパン表記。256 要素。
//! 全 ISA 共通の端タイル用スタック
//! バッファ長）を変更せず既存 ISA（GB10 の NEON 経路等）へ副作用を
//! 与えないことを優先し、ZA0 単一タイル（MR*NR=16*16=256=MAX_TILE で
//! ちょうど収まる）の MR=16×NR=16 を採用する（計画リスク §10「フォール
//! バック案」）。将来 32×32（4 タイル）を追加する場合は `MAX_TILE` 拡大の
//! 副作用（GB10 NEON 端タイル計測）を別途実施すること。
//!
//! ## bit 完全一致契約（REQ-2・`.claude/rules/coding-rust.md` FMA 契約統一）
//!
//! `fmopa za0.s, p0/m, p0/m, zn.s, zm.s` は Arm ARM（DDI0616）により
//! 「非拡張 FP32 外積: 各要素 `za[i][j] = fma(zn[i], zm[j], za[i][j])`
//! （単一丸めの fused multiply-add）」と定義される。`compute`（非公開
//! 関数のためコードスパン表記）は
//! ループ前に **C の現在値を ZA0 へプリロード**し、`p` を昇順に走査して
//! `kc_len` 回 `fmopa` を発行した後に ZA0 を C へストアし直す（zero-init
//! して最後に加算する方式は丸めが 1 回増え不一致になるため採らない）。
//! これにより各 `c[i][j]` は「初期値 = 呼び出し時点の `c[i][j]`、p 昇順に
//! 1 回ずつ `fma(a[p][i], b[p][j], acc)`」という演算列になり、
//! [`super::neon`] の `vfmaq_laneq_f32(acc, b, a, lane)` = `acc + b*a[lane]`
//! （単一 FMA・p 昇順・レーン間縮約なし）と**乗算の可換性（IEEE-754-2008
//! §5.4.1。有限値に限る）を除いて演算列が完全に同一**になる。有限値
//! 入力での bit 完全一致は `tests/gemm_blis_parity.rs`（本番入口経由）・
//! 本モジュール下部の単体テスト（scalar 参照・NEON 参照との直接比較）で
//! 検証する。
//!
//! NaN を含む入力については、両オペランドが異なる payload の NaN のとき
//! IEEE-754-2008 §6.2 の NaN 選択規則が `fma(a,b,c)` と
//! `mul_add`／`vfmaq_laneq_f32` の実装間で一致する保証がない（`neon`
//! モジュール `compute_b_laneq` §748 節と同じ理由）ため、NaN 混入入力に
//! 対する bit 一致は主張しない（panic なし・NaN 位置一致のみを診断的に
//! 確認する）。
//!
//! ## 検出との関係
//!
//! 本モジュールの関数はいずれも「実行 CPU が SME・非拡張 FP32 外積
//! （`SME_F32F32`）に対応し、かつ SVL=64 バイト（512 bit）である」ことを
//! 呼び出し元契約とする（`super::SmeKernel::try_new`（`pub(crate)`
//! のためコードスパン表記）。
//! `crate::sme_detect::sme_report()` が fail-closed に判定する）。

use std::arch::asm;

/// マイクロカーネルタイルの行数（ZA0 の行数。SVL=512 bit 前提）。
pub const MR: usize = 16;
/// マイクロカーネルタイルの列数（ZA0 の列数）。
pub const NR: usize = 16;

// [`super::super::gemm_blis_region`] の C タイルスタックバッファは
// `MAX_TILE`（256 要素）固定長で確保するため、コンパイル時に検査する
// （MR*NR=256 でちょうど一致。他 ISA と同型の契約）。
const _: () = assert!(MR * NR <= 256);
const _: () = assert!(MR == 16 && NR == 16);

/// [`kernel_unchecked_with_ldc`]／[`kernel_unchecked`] 共通の演算本体。
///
/// # Safety
///
/// 呼び出し元は次を保証しなければならない:
/// - 実行 CPU が SME・非拡張 FP32 外積（`SME_F32F32`）に対応し、SVL が
///   64 バイト（512 bit。`crate::sme_detect` の `REQUIRED_SVL_BYTES` と
///   同じ値）であること（`smstart` 後の
///   `fmopa`／`mova`／`ld1w`／`st1w` がこの前提でのみ健全）。
/// - `ap.len() == MR * kc_len`・`bp.len() == kc_len * NR`（p-major packing。
///   `pack_a`／`pack_b` の `dst[p*mr+i]`／`dst[p*nr+j]` 契約）。
/// - `c.len() >= (MR - 1) * ldc + NR`（`ldc >= NR`）。
unsafe fn compute(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    // 呼び出し元契約（本関数の `# Safety` 節）により、以下のロード／
    // ストアはいずれもこの範囲内のオフセットに限定される:
    // - bp: p*NR の最大は p=kc_len-1 でも (kc_len-1)*NR+NR = bp.len() を
    //   超えない。
    // - ap: p*MR の最大は p=kc_len-1 でも (kc_len-1)*MR+MR = ap.len() を
    //   超えない。
    // - c（プリロード・ストア両ループとも i in 0..MR）: 最大オフセットは
    //   (MR-1)*ldc+NR <= c.len()（`ldc >= NR` は呼び出し元契約）。
    let c_ptr = c.as_mut_ptr();
    let a_ptr = ap.as_ptr();
    let b_ptr = bp.as_ptr();
    let ldc_bytes = ldc * size_of::<f32>();
    let kc = kc_len;

    // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）により実行 CPU は
    // SME・非拡張 FP32 外径（`SME_F32F32`）に対応し SVL=64 バイト。
    //
    // レジスタ・状態契約（`docs/cpu-gemm-sme-fmopa-microkernel.md` §3 の
    // asm 契約節を参照。1 箇所に局所化）:
    // - `.arch_extension sme` はアセンブラへ SME 命令の使用を許可する
    //   ディレクティブ（コンパイル時のみに影響）。
    // - `smstart`/`smstop` を 1 つの asm! ブロック内で対にして閉じ、
    //   ブロック内で SME 以外の SIMD 命令を混在させない。
    // - `smstart` は Z0-Z31/P0-P15（および FFR）の内容を不定化するため
    //   `out("v0") _ ... out("v31") _`・`out("p0") _ ... out("p15") _` を
    //   全列挙する（FFR は本ルーチンが `ldff1`/`ldnf1` 系の投機ロードを
    //   使わないため未使用・未依存。値の破棄自体は smstart/smstop の
    //   ハードウェア契約でありコンパイラへ追加の宣言余地はない）。
    // - `mova` のスライスインデックスレジスタは w12-w15 限定のため
    //   `out("w12") _` を明示する。
    // - `subs`/`cmp`/`b.lt`/`b.ne`/`cbz` でフラグを書き換えるため
    //   `preserves_flags` は付けない。
    // - `ld1w`/`st1w`（通常のメモリアクセス）を使うため `nomem`/`pure`
    //   は付けない。スタックを使わないため `options(nostack)`。
    // - ポインタ演算（`ldc_bytes`）は asm 外で `usize` 乗算により確定し、
    //   asm へは検査済みバイトストライドのみ渡す（呼び出し元契約が
    //   保証する範囲内でのみ加算するため、桁あふれの検査は本関数の
    //   `# Safety` 契約〈呼び出し元が渡す `ap`/`bp`/`c` の長さ検査〉に
    //   委ねる。呼び出し元 `kernel_with_ldc`/`kernel_unchecked_with_ldc`
    //   はいずれも `super::check_panel_lengths`/`check_c_tile_bounds`
    //   〈`checked_mul`/`checked_add` 使用〉を先に通す。REQ-8）。
    // - C プリロード（`cpre`）・ストア（`cpost`）は同一の初期ポインタ値
    //   （`c_ptr`）から独立に 2 つの汎用レジスタへ展開し、各ループ内で
    //   `ldc_bytes` ずつ進める（プリロードループがポインタを消費しても
    //   ストアループ用の元ポインタが失われないようにするため）。
    unsafe {
        asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.s",
            // C プリロード: 16 行を za0h.s[0..16] へロードする
            // （行 i は c_ptr + i*ldc_bytes から NR=16 要素）。
            "mov w12, #0",
            "10:",
            "ld1w {{z0.s}}, p0/z, [{cpre}]",
            "mova za0h.s[w12, #0], p0/m, z0.s",
            "add {cpre}, {cpre}, {ldc_bytes}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lt 10b",
            // k ループ: kc_len == 0 ならスキップ（za0 は C の値のまま）。
            // `kc` は `usize`（64 bit）のフルレジスタ（`{kc}`＝X レジスタ）で
            // 扱う。`cbz`/`subs` の 32 bit（`{kc:w}`＝W レジスタ）版は
            // `kc_len` を暗黙的に下位 32 bit へ切り詰めるため、
            // `kc_len > u32::MAX` の呼び出し（`BlockSizes::kc` が極端に
            // 大きい場合。理論上は `usize` の契約上あり得る）で無音に
            // 誤った反復回数になりうる（advisor レビュー指摘。到達可能な
            // `kc_len` は `blocks.kc` で事実上有界だが、SAFETY 契約を
            // レジスタ幅の暗黙の仮定に依存させない）。
            "cbz {kc}, 12f",
            "11:",
            "ld1w {{z1.s}}, p0/z, [{a}]",
            "ld1w {{z2.s}}, p0/z, [{b}]",
            "fmopa za0.s, p0/m, p0/m, z1.s, z2.s",
            "add {a}, {a}, #64",
            "add {b}, {b}, #64",
            "subs {kc}, {kc}, #1",
            "b.ne 11b",
            "12:",
            // ストア: za0h.s[0..16] を C の 16 行へ書き戻す。
            "mov w12, #0",
            "13:",
            "mova z3.s, p0/m, za0h.s[w12, #0]",
            "st1w {{z3.s}}, p0, [{cpost}]",
            "add {cpost}, {cpost}, {ldc_bytes}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lt 13b",
            "smstop",
            cpre = inout(reg) c_ptr => _,
            cpost = inout(reg) c_ptr => _,
            a = inout(reg) a_ptr => _,
            b = inout(reg) b_ptr => _,
            kc = inout(reg) kc => _,
            ldc_bytes = in(reg) ldc_bytes,
            out("w12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
            out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
            out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _,
            out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _,
            out("p15") _,
            options(nostack),
        );
    }
}

/// [`super::TileBoundsError`] 検査つきの `ldc` 契約版（`compute`〈非公開
/// 関数のためコードスパン表記〉へ委譲）。[`super::neon::kernel_with_ldc`]
/// と同型の入口だが、SME は実行時検出済みトークン（[`super::SmeKernel`]）
/// 経由でのみ安全に呼べるため `unsafe fn` とする（`super::avx2::kernel_unchecked_with_ldc`。
/// x86_64 限定のためコードスパン表記）と同型）。
///
/// # Safety
///
/// 呼び出し元は実行 CPU が SME・非拡張 FP32 外積（`SME_F32F32`）に対応し
/// SVL=`crate::sme_detect::REQUIRED_SVL_BYTES`（非公開定数のため
/// コードスパン表記。64 バイト）であることを保証しなければならない
/// （`super::SmeKernel::try_new`〈`pub(crate)` のためコードスパン表記〉
/// 経由の実行時検出済み呼び出しがこれを
/// 満たす）。
pub unsafe fn kernel_unchecked_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    super::check_panel_lengths(MR, NR, kc_len, ap.len(), bp.len())?;
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;
    // SAFETY: [`compute`] のドキュメント参照（直前の検査により長さ前提を
    // 満たし、SME 対応は本関数の呼び出し元契約として引き継ぐ）。
    unsafe { compute(ap, bp, c, ldc, kc_len) };
    Ok(())
}

/// 従来シグネチャ後方互換ラッパー（`ldc = NR` 固定・密パッキング契約。
/// `super::avx2::kernel_unchecked`（x86_64 限定のためコードスパン表記）
/// と同型）。
///
/// # Safety
///
/// [`kernel_unchecked_with_ldc`] と同一。
pub unsafe fn kernel_unchecked(ap: &[f32], bp: &[f32], c: &mut [f32], kc_len: usize) {
    assert!(
        super::panel_len_matches(ap.len(), MR, kc_len),
        "packed A panel length mismatch (or MR*kc_len overflow): ap.len()={}, MR={MR}, kc_len={kc_len}",
        ap.len()
    );
    assert!(
        super::panel_len_matches(bp.len(), kc_len, NR),
        "packed B panel length mismatch (or kc_len*NR overflow): bp.len()={}, kc_len={kc_len}, NR={NR}",
        bp.len()
    );
    assert_eq!(c.len(), MR * NR, "C tile length mismatch");
    // SAFETY: [`compute`] のドキュメント参照（呼び出し元契約を引き継ぐ）。
    unsafe { compute(ap, bp, c, NR, kc_len) };
}

/// `unsafe { compute(...) }`（`fmopa` アセンブリ）を発行できない場合の
/// 安全な Rust フォールバック（イシュー #1587 codex-review P1 再指摘
/// `PRRT_kwDOTuUCJc6h0ZMD` への対応）。
///
/// `SmeKernel` は「構築したスレッド」の SME 対応・SVL を保証するのみで、
/// `Copy` により別スレッド（Rayon worker）へ渡された場合はそのスレッド
/// 自身の SVL が異なる（Linux では `prctl(PR_SME_SET_VL)` によりスレッド
/// ごとに変更可能）ことがある（[`super::SmeKernel`] doc 参照）。以前は
/// 実行スレッド自身の再確認に失敗した場合 `panic!` していたが、これは
/// `cfg(test)` 外の本番ライブラリコードであり
/// `.claude/rules/security.md`／AGENTS.md「本番経路の panic 禁止」に
/// 抵触する。本関数は `compute` の代わりに実行される安全なフォール
/// バックとして、`compute` と**同一の演算列**（p 昇順・要素ごとに 1 回
/// の `f32::mul_add` を適用し、レーン間の並べ替え・再結合を行わない）を
/// 再現する。これにより `fmopa`（Arm ARM DDI0616 の非拡張 FP32 外積:
/// `za[i][j] = fma(zn[i], zm[j], za[i][j])`）と本関数の結果は有限値
/// 入力で bit 完全一致する（モジュール冒頭「bit 完全一致契約」節参照。
/// `compute` が実行される経路と全く同一の呼び出し元契約〈`ap`／`bp`
/// の長さ・`ldc` 境界〉を要求するため、境界検査は呼び出し元
/// （[`kernel_unchecked_with_ldc`] 相当）にまかせず本関数自身でも
/// 行う）。
fn scalar_fallback(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    for p in 0..kc_len {
        for i in 0..MR {
            let a_val = ap[p * MR + i];
            for j in 0..NR {
                let idx = i * ldc + j;
                c[idx] = a_val.mul_add(bp[p * NR + j], c[idx]);
            }
        }
    }
}

/// [`scalar_fallback`] の `ldc` 契約版・境界検査つき入口（
/// [`kernel_unchecked_with_ldc`] と同型の検査を行い、`compute` の代わりに
/// [`scalar_fallback`] を呼ぶ。`unsafe` を含まないため `unsafe fn` では
/// ない）。
pub(crate) fn scalar_fallback_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    super::check_panel_lengths(MR, NR, kc_len, ap.len(), bp.len())?;
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;
    scalar_fallback(ap, bp, c, ldc, kc_len);
    Ok(())
}

/// [`scalar_fallback_with_ldc`] の従来シグネチャ版（`ldc = NR` 固定・
/// 密パッキング契約。[`kernel_unchecked`] と同型の長さ検査を行う）。
pub(crate) fn scalar_fallback_kernel(ap: &[f32], bp: &[f32], c: &mut [f32], kc_len: usize) {
    assert!(
        super::panel_len_matches(ap.len(), MR, kc_len),
        "packed A panel length mismatch (or MR*kc_len overflow): ap.len()={}, MR={MR}, kc_len={kc_len}",
        ap.len()
    );
    assert!(
        super::panel_len_matches(bp.len(), kc_len, NR),
        "packed B panel length mismatch (or kc_len*NR overflow): bp.len()={}, kc_len={kc_len}, NR={NR}",
        bp.len()
    );
    assert_eq!(c.len(), MR * NR, "C tile length mismatch");
    scalar_fallback(ap, bp, c, NR, kc_len);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm_blis::microkernel::{Microkernel, SmeKernel};

    /// 有限値・非正規化数入力を含むスカラー参照（p 昇順 `f32::mul_add`
    /// 連鎖）との bit 完全一致を検証する下請け関数。以下の各テストは
    /// SME 実機（例: Apple M4）依存のため `#[ignore]` で分離する
    /// （`.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]`
    /// で分離」。codex-review P1 指摘 `PRRT_kwDOTuUCJc6h0P7c` 対応）。
    /// `#[ignore]` 実行時点でも `SmeKernel::try_new()` が `None`（実行
    /// 環境が SME 非対応）を返す場合は実行時スキップする（非対応実機
    /// での `--ignored` 実行を green に保つための二重の安全策）。
    fn scalar_reference(
        ap: &[f32],
        bp: &[f32],
        c_init: &[f32],
        ldc: usize,
        kc_len: usize,
    ) -> Vec<f32> {
        let mut c = c_init.to_vec();
        for i in 0..MR {
            for j in 0..NR {
                let mut acc = c[i * ldc + j];
                for p in 0..kc_len {
                    acc = ap[p * MR + i].mul_add(bp[p * NR + j], acc);
                }
                c[i * ldc + j] = acc;
            }
        }
        c
    }

    fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_matches_scalar_reference_finite_values() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        for &kc_len in &[0usize, 1, 3, 4, 17, 255, 256] {
            let ap = xorshift32_vec(0x1234_5678 ^ kc_len as u32, MR * kc_len);
            let bp = xorshift32_vec(0x9abc_def0 ^ kc_len as u32, kc_len * NR);
            let c_init = xorshift32_vec(0x1111_2222 ^ kc_len as u32, MR * NR);

            let mut c_sme = c_init.clone();
            kernel.run(&ap, &bp, &mut c_sme, kc_len);

            let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
            assert_eq!(
                c_sme, expected,
                "kc_len={kc_len} で scalar 参照と bit 完全一致するはず"
            );
        }
    }

    /// 端タイル相当（`ldc > NR`）での bit 完全一致（[`kernel_unchecked_with_ldc`]
    /// 経由）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_with_ldc_matches_scalar_reference_strided() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 37;
        let ldc = NR + 5;
        let ap = xorshift32_vec(0xaaaa_bbbb, MR * kc_len);
        let bp = xorshift32_vec(0xcccc_dddd, kc_len * NR);
        // `ldc` ストライドの C バッファ（行間ギャップあり）。
        let mut c = xorshift32_vec(0xeeee_ffff, (MR - 1) * ldc + ldc);
        let c_init = c.clone();

        kernel.run_with_ldc(&ap, &bp, &mut c, ldc, kc_len).unwrap();

        // `scalar_reference` は任意の `ldc` を尊重し `c[i*ldc+j]`
        // （`j in 0..NR`）のみを書き換える（ギャップ列 `j>=NR` は
        // 触れない）ため、ストライド付きバッファ全体を 1 回で計算できる。
        let expected = scalar_reference(&ap, &bp, &c_init, ldc, kc_len);
        assert_eq!(
            c, expected,
            "ldc={ldc} のストライド付きバッファ全体が scalar 参照と \
             bit 完全一致するはず（ギャップ領域が変更されないことを含む）"
        );
    }

    /// run-to-run bit 同一性（同一入力を 2 回実行して比較。R3(a)）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_is_deterministic_across_runs() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 129;
        let ap = xorshift32_vec(0x2468_1357, MR * kc_len);
        let bp = xorshift32_vec(0x1357_2468, kc_len * NR);
        let c_init = xorshift32_vec(0x0f0f_0f0f, MR * NR);

        let mut c1 = c_init.clone();
        kernel.run(&ap, &bp, &mut c1, kc_len);
        let mut c2 = c_init.clone();
        kernel.run(&ap, &bp, &mut c2, kc_len);

        assert_eq!(c1, c2, "同一入力の 2 回実行は bit 同一であるはず");
    }

    /// 非正規化数（約 1e-40）入力での scalar 参照との bit 完全一致
    /// （R3(b)。ストリーミングモードでの FPCR/FZ 差の検出を狙う）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_matches_scalar_reference_denormal_values() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 8;
        let denormal = 1e-40f32;
        assert!(
            denormal != 0.0 && denormal.abs() < f32::MIN_POSITIVE,
            "テスト前提: 1e-40 は非正規化数であるはず"
        );
        let ap = vec![denormal; MR * kc_len];
        let bp = vec![denormal; kc_len * NR];
        let c_init = vec![denormal; MR * NR];

        let mut c_sme = c_init.clone();
        kernel.run(&ap, &bp, &mut c_sme, kc_len);

        let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
        assert_eq!(
            c_sme, expected,
            "非正規化数入力でも scalar 参照と bit 完全一致するはず"
        );
    }

    /// 非正規化数「結果」の FTZ（flush-to-zero）差異を検出する
    /// （advisor レビュー指摘: 上記
    /// `sme_kernel_matches_scalar_reference_denormal_values` は入力が
    /// 非正規化数のケースのみを検証しており、`a=b=c=1e-40` では
    /// `fma(1e-40,1e-40,1e-40)` の積項が 0 へアンダーフローするため
    /// 「非正規化数の乗算結果がストリーミングモードでフラッシュされない
    /// こと」自体は検証できていなかった）。本テストは正規化数どうしの
    /// 積が非正規化数（`1e-20 * 1e-20 = 1e-40`）になる入力を使い、SME
    /// 側が FTZ でこの結果を 0 へ潰さず scalar 参照と bit 完全一致する
    /// ことを確認する（R3(b) が本来意図した FZ 判別）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_matches_scalar_reference_denormal_result_from_normal_operands() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 1;
        let normal_small = 1e-20f32;
        assert!(
            normal_small.is_normal(),
            "テスト前提: 1e-20 は正規化数であるはず"
        );
        let denormal_product = normal_small * normal_small;
        assert!(
            denormal_product != 0.0 && denormal_product.abs() < f32::MIN_POSITIVE,
            "テスト前提: 1e-20 * 1e-20 は非正規化数（アンダーフロー結果）であるはず"
        );

        let ap = vec![normal_small; MR * kc_len];
        let bp = vec![normal_small; kc_len * NR];
        let c_init = vec![0.0f32; MR * NR];

        let mut c_sme = c_init.clone();
        kernel.run(&ap, &bp, &mut c_sme, kc_len);

        let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
        // scalar_reference（`f32::mul_add`）自身が非正規化数の結果を
        // 生成することを前提の一部として確認する（テスト自体の健全性）。
        assert!(
            expected
                .iter()
                .all(|&v| v != 0.0 && v.abs() < f32::MIN_POSITIVE),
            "scalar 参照側も非正規化数を生成するはず（テスト前提）"
        );
        assert_eq!(
            c_sme, expected,
            "正規化数どうしの積がアンダーフローする非正規化数の結果を、\
             SME 側が FTZ で 0 へ潰さず scalar 参照と bit 完全一致するはず"
        );
    }

    /// NaN 混入入力で panic しないことのみを確認する（bit 一致は
    /// 主張しない。モジュール冒頭ドキュメント参照）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_nan_input_does_not_panic() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 4;
        let mut ap = xorshift32_vec(0x1a1a_1a1a, MR * kc_len);
        ap[0] = f32::NAN;
        let bp = xorshift32_vec(0x2b2b_2b2b, kc_len * NR);
        let mut c = xorshift32_vec(0x3c3c_3c3c, MR * NR);
        kernel.run(&ap, &bp, &mut c, kc_len);
        // panic しないことのみを確認する。
    }
}
