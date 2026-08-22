//! `nav.toml` の読み込み・検証から HTML ページ・テーマ CSS の書き出しまでの
//! ビルドパイプライン本体。
//!
//! # 呼び出し文脈
//!
//! `main.rs`（CLI）から [`build_site`] が呼ばれる。本モジュールは
//! `crate::nav`（[`crate::nav::parse_nav`] / [`crate::nav::validate_sources`]）・
//! `crate::markdown`（[`crate::markdown::markdown_to_nodes`]）・
//! `crate::layout`（[`crate::layout::docs_page`]）・`crate::theme`
//! （[`crate::theme::SITE_CSS`]）に依存し、`<root>/site/nav.toml` の読み込みから
//! `<out>` 配下への実 HTML・CSS 書き出しまでを結線する（実装計画 §2.6・#870）。

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::html;
use crate::layout;
use crate::markdown;
use crate::nav::{self, Nav, NavError};

/// `nav.toml` の相対配置（`repo_root` からの相対パス）。
const NAV_TOML_RELATIVE_PATH: &str = "site/nav.toml";

/// 生成物 HTML の先頭に前置する doctype 宣言（`layout::docs_page` は `<html>` の
/// 中身のみを返すため、書き出し時にここで前置する。実装計画 §2.4 注記）。
const DOCTYPE: &str = "<!DOCTYPE html>\n";

/// `assets/site.css` の出力先（`out` からの相対パス）。
const SITE_CSS_RELATIVE_PATH: &str = "assets/site.css";

/// `page.source`（Markdown 原稿）の入力サイズ上限。[`nav::MAX_INPUT_BYTES`] と
/// 同値を採用する（DoS 抑止の読み込み前実効化。実装計画 §2.6 手順 2）。
const MAX_SOURCE_BYTES: u64 = nav::MAX_INPUT_BYTES as u64;

/// [`build_site`] の失敗理由。[`NavError`] と I/O エラーを包む型付き enum。
/// `Display` に機微情報（入力全文・絶対パス・環境変数）を載せない方針は
/// [`NavError`] 側の契約をそのまま引き継ぐ（`page.source` 等の相対パス文字列は
/// 診断に必要な最小限として含める）。
#[derive(Debug)]
pub enum BuildError {
    /// `<root>/site/nav.toml` の読み込みに失敗した（不在・権限不足等）。
    ReadNavToml(std::io::Error),
    /// [`nav::parse_nav`] / [`nav::validate_sources`] のいずれかが失敗した。
    Nav(NavError),
    /// 出力ディレクトリ（`--out`）の作成に失敗した。
    CreateOutDir(std::io::Error),
    /// `page.source` が [`MAX_SOURCE_BYTES`] を超過した（読み込み前にサイズ確認で検出）。
    SourceTooLarge(String),
    /// `page.source` の読み込みに失敗した（`validate_sources` 通過後の再読込エラー。
    /// TOCTOU・権限変更等の稀な経路）。
    ReadSource {
        source: String,
        error: std::io::Error,
    },
    /// ページ HTML の書き出し（親ディレクトリ作成含む）に失敗した。
    WritePage { path: String, error: std::io::Error },
    /// `assets/site.css` の書き出しに失敗した。
    WriteAsset(std::io::Error),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::ReadNavToml(err) => {
                write!(f, "failed to read {NAV_TOML_RELATIVE_PATH}: {err}")
            }
            BuildError::Nav(err) => write!(f, "{err}"),
            BuildError::CreateOutDir(err) => {
                write!(f, "failed to create output directory: {err}")
            }
            BuildError::SourceTooLarge(source) => {
                write!(
                    f,
                    "page.source `{source}` exceeds the {MAX_SOURCE_BYTES} byte size limit"
                )
            }
            BuildError::ReadSource { source, error } => {
                write!(f, "failed to read page.source `{source}`: {error}")
            }
            BuildError::WritePage { path, error } => {
                write!(f, "failed to write page output `{path}`: {error}")
            }
            BuildError::WriteAsset(err) => {
                write!(f, "failed to write {SITE_CSS_RELATIVE_PATH}: {err}")
            }
        }
    }
}

impl std::error::Error for BuildError {}

impl From<NavError> for BuildError {
    fn from(err: NavError) -> Self {
        BuildError::Nav(err)
    }
}

/// [`build_site`] 成功時のレポート。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildReport {
    /// `nav.toml` 内で検証済みのページ総数（全セクション合算）。
    pub pages: usize,
    /// 出力先ディレクトリ（`--out` の値をそのまま保持する）。
    pub out_dir: PathBuf,
    /// 実際に書き出した生成物の一覧（`out_dir` からの相対パス。各ページの
    /// `index.html` + `assets/site.css`）。
    pub written: Vec<PathBuf>,
}

/// `page.path`（`/` 始まり・`/` 終わりが [`nav::parse_nav`] で保証済み）から
/// `<out>` 配下の出力ファイルパスを組み立てる。
///
/// `page.path` は常に `/` 始まりであり、`Path::join` は絶対パス相当の
/// コンポーネントを渡すと受け側（`out`）を丸ごと破棄してしまう
/// （`nav.rs` の `looks_like_windows_drive_path` と同種の `Path::join`
/// セマンティクスの罠。レビュー指摘）。よって結合前に必ず先頭の `/` を取り除く。
fn page_output_path(out: &Path, page_path: &str) -> PathBuf {
    let relative = page_path.trim_start_matches('/');
    if relative.is_empty() {
        out.join("index.html")
    } else {
        out.join(relative).join("index.html")
    }
}

/// `path` の親ディレクトリを作成してからファイルへ書き出す。
fn write_file_creating_parent(path: &Path, contents: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)
}

/// `<root>/site/nav.toml` を読み込み・パース・検証し、各ページを Markdown→HTML
/// 変換したうえで `out` 配下へ実ファイルとして書き出すビルドパイプライン。
///
/// 手順:
/// 1. `<root>/site/nav.toml` を読む（不在・読み込み失敗はエラー）。
///    [`nav::MAX_INPUT_BYTES`] 超過は、DoS 抑止の実効性を保つため
///    `fs::metadata` でファイルサイズを見てから `fs::read_to_string` する
///    （超過ファイル全体をメモリに読み切ってから [`nav::parse_nav`] 内で
///    拒否する経路だと、この唯一の FS アクセス経路では「読み込み前」の
///    抑止が効かない。レビュー指摘）
/// 2. [`nav::parse_nav`] でスキーマ検証 → [`nav::validate_sources`] で
///    `page.source` の実ファイル存在検証
/// 3. 各 `page.source` を（読み込み前サイズ検査つきで）読み、
///    [`markdown::markdown_to_nodes`] → [`layout::docs_page`] → [`html::render`]
///    の順で HTML 文字列へ変換する。**全ページの変換をメモリ上で終えてから**
///    書き出しを開始する（I/O エラー発生時に部分生成物を減らす安全側の順序。
///    完全な fail-closed 原子性は #872 linkcheck の責務）
/// 4. `out` ディレクトリを作成し、各ページを `<out>{page.path}index.html` へ、
///    テーマ CSS を `<out>/assets/site.css` へ書き出す
/// 5. 書き出し件数を含む [`BuildReport`] を返す
///
/// # Errors
///
/// `nav.toml` の読み込み失敗・スキーマ／source 検証失敗・`page.source` の
/// サイズ超過／読み込み失敗・出力ディレクトリ作成失敗・書き出し失敗のいずれかで
/// [`BuildError`] を返す。
pub fn build_site(root: &Path, out: &Path) -> Result<BuildReport, BuildError> {
    let nav_toml_path = root.join(NAV_TOML_RELATIVE_PATH);

    // DoS 抑止（`nav::MAX_INPUT_BYTES`）をこの唯一の FS アクセス経路で実効化する
    // ため、ファイル全体を `fs::read_to_string` で読み切る前に `fs::metadata` で
    // サイズを確認する。`parse_nav` 内の `input.len()` 検査（`nav.rs` 側）は
    // 既にメモリに載った文字列に対する検査であり、この読み込み前チェックとは
    // 独立に維持する（`parse_nav` 単体テストの FS 非依存性を壊さないため）。
    let metadata = fs::metadata(&nav_toml_path).map_err(BuildError::ReadNavToml)?;
    if metadata.len() > nav::MAX_INPUT_BYTES as u64 {
        return Err(BuildError::Nav(NavError::TooLarge));
    }

    let input = fs::read_to_string(&nav_toml_path).map_err(BuildError::ReadNavToml)?;

    let parsed: Nav = nav::parse_nav(&input)?;
    nav::validate_sources(&parsed, root)?;

    // 手順 3: 全ページを先にメモリ上でレンダリングし切ってから書き出す。
    let mut rendered_pages: Vec<(PathBuf, String)> = Vec::new();
    for section in &parsed.sections {
        for page in &section.pages {
            let source_path = root.join(&page.source);
            let source_metadata =
                fs::metadata(&source_path).map_err(|error| BuildError::ReadSource {
                    source: page.source.clone(),
                    error,
                })?;
            if source_metadata.len() > MAX_SOURCE_BYTES {
                return Err(BuildError::SourceTooLarge(page.source.clone()));
            }
            let markdown_input =
                fs::read_to_string(&source_path).map_err(|error| BuildError::ReadSource {
                    source: page.source.clone(),
                    error,
                })?;

            let body = markdown::markdown_to_nodes(&markdown_input);
            let page_node = layout::docs_page(&parsed, &page.title, &page.path, body);
            let html_out = format!("{DOCTYPE}{}", html::render(&page_node));

            let out_path = page_output_path(out, &page.path);
            rendered_pages.push((out_path, html_out));
        }
    }

    // 手順 4: 出力ディレクトリ作成 → ページ書き出し → テーマ CSS 書き出し。
    fs::create_dir_all(out).map_err(BuildError::CreateOutDir)?;

    let mut written = Vec::with_capacity(rendered_pages.len() + 1);
    for (out_path, html_out) in &rendered_pages {
        write_file_creating_parent(out_path, html_out).map_err(|error| BuildError::WritePage {
            path: out_path.display().to_string(),
            error,
        })?;
        if let Ok(relative) = out_path.strip_prefix(out) {
            written.push(relative.to_path_buf());
        }
    }

    let css_path = out.join(SITE_CSS_RELATIVE_PATH);
    write_file_creating_parent(&css_path, crate::theme::SITE_CSS)
        .map_err(BuildError::WriteAsset)?;
    written.push(PathBuf::from(SITE_CSS_RELATIVE_PATH));

    let pages = parsed.sections.iter().map(|s| s.pages.len()).sum();
    Ok(BuildReport {
        pages,
        out_dir: out.to_path_buf(),
        written,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト専用の一時ディレクトリ。`Drop` でベストエフォート削除する。
    /// 外部クレート（`tempfile` 等）を追加せず `std::env::temp_dir()` +
    /// プロセス固有サフィックスで代用する（REQ-1 v2: 外部依存ゼロを維持する）。
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "rust-ai-library-docs-site-build-test-{tag}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temp dir for build.rs test");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn build_site_succeeds_for_valid_fixture_root() {
        let root = TempDir::new("build-valid");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/intro/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/intro.md"), b"# Intro").unwrap();

        let out = TempDir::new("build-valid-out");
        let report = build_site(&root.0, &out.0).expect("build should succeed");
        assert_eq!(report.pages, 1);
        assert!(out.0.is_dir());
        assert!(out.0.join("intro/index.html").is_file());
        assert!(out.0.join("assets/site.css").is_file());
        let html = fs::read_to_string(out.0.join("intro/index.html")).unwrap();
        assert!(html.starts_with("<!DOCTYPE html>\n"));
        assert!(html.contains("<h1>Intro</h1>"));
        assert!(html.contains("assets/site.css"));
    }

    #[test]
    fn build_site_writes_root_page_path_directly_under_out_without_escaping() {
        // レビュー指摘の回帰テスト: `page.path` は必ず `/` 始まりのため、
        // `Path::join` の絶対パスセマンティクスに任せると `out` を丸ごと破棄して
        // ファイルシステムルートへ書き出してしまう（`page_output_path` が
        // 先頭 `/` を除去してから結合することの検証）。
        let root = TempDir::new("build-root-path");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Top"
source = "site/index.md"
path = "/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/index.md"), b"# Top").unwrap();

        let out = TempDir::new("build-root-path-out");
        let report = build_site(&root.0, &out.0).expect("build should succeed");
        let written_path = out.0.join("index.html");
        assert!(written_path.is_file());
        assert!(written_path.starts_with(&out.0));
        assert!(report.written.contains(&PathBuf::from("index.html")));
    }

    #[test]
    fn build_site_reports_all_written_page_files_and_theme_css() {
        let root = TempDir::new("build-report-written");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/guide/intro/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/intro.md"), b"# Intro").unwrap();

        let out = TempDir::new("build-report-written-out");
        let report = build_site(&root.0, &out.0).expect("build should succeed");
        assert!(
            report
                .written
                .contains(&PathBuf::from("guide/intro/index.html"))
        );
        assert!(report.written.contains(&PathBuf::from("assets/site.css")));
    }

    #[test]
    fn build_site_rejects_oversized_page_source_without_reading_it_fully() {
        let root = TempDir::new("build-oversized-source");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/intro/"
"#,
        )
        .unwrap();
        let oversized = "a".repeat((MAX_SOURCE_BYTES + 1) as usize);
        fs::write(root.0.join("site/intro.md"), oversized).unwrap();

        let out = TempDir::new("build-oversized-source-out");
        match build_site(&root.0, &out.0) {
            Err(BuildError::SourceTooLarge(source)) => assert_eq!(source, "site/intro.md"),
            other => panic!("expected SourceTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn build_site_fails_when_nav_toml_missing() {
        let root = TempDir::new("build-missing-nav");
        let out = TempDir::new("build-missing-nav-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::ReadNavToml(_))
        ));
    }

    #[test]
    fn build_site_fails_when_source_missing() {
        let root = TempDir::new("build-missing-source");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/does-not-exist.md"
path = "/intro/"
"#,
        )
        .unwrap();

        let out = TempDir::new("build-missing-source-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::Nav(NavError::MissingSource(_)))
        ));
    }

    #[test]
    fn build_site_fails_on_invalid_nav_schema() {
        let root = TempDir::new("build-invalid-schema");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(root.0.join("site/nav.toml"), "not valid toml subset\n").unwrap();

        let out = TempDir::new("build-invalid-schema-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::Nav(NavError::Parse { .. }))
        ));
    }

    #[test]
    fn build_site_rejects_oversized_nav_toml_without_reading_it_fully() {
        // `fs::metadata` によるサイズ確認が `fs::read_to_string` の前段で効いて
        // いることの回帰テスト（レビュー指摘: DoS 抑止の「読み込み前」実効化）。
        let root = TempDir::new("build-oversized-nav");
        fs::create_dir_all(root.0.join("site")).unwrap();
        let mut oversized = String::from("[site]\ntitle = \"");
        oversized.push_str(&"a".repeat(nav::MAX_INPUT_BYTES + 1));
        oversized.push_str("\"\nbase_path = \"\"\n");
        fs::write(root.0.join("site/nav.toml"), oversized).unwrap();

        let out = TempDir::new("build-oversized-nav-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::Nav(NavError::TooLarge))
        ));
    }
}
