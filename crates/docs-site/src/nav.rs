//! `site/nav.toml`（docs サイトのナビゲーション構成マニフェスト）を検証する
//! fail-closed TOML サブセットパーサー。
//!
//! # 呼び出し文脈
//!
//! `main.rs`（CLI）から `build::build_site`（`crate::build`）経由で呼ばれ、
//! [`parse_nav`] → [`validate_sources`] の順に適用される。得られた [`Nav`] は
//! 兄弟イシュー #870 の Markdown→HTML 変換・layout モジュールが読み取る入力になる
//! （本イシュー #869 では HTML 生成を行わない。実装計画 §1 参照）。
//!
//! # 対応する TOML サブセット
//!
//! `nav.toml` は以下の構文のみを許可するサブセットとして扱う（それ以外はすべて
//! [`NavError::Parse`] で明示的に失敗する。fail-closed。未対応構文を黙って無視しない）。
//!
//! - `#` から始まる行コメント、および文字列値の終端後に続く `# ...`
//! - `[site]` テーブル（`title` / `base_path` の 2 キー）
//! - `[[section]]` array-of-tables（`title` の 1 キー必須 + `index_path` の
//!   1 キー任意。`index_path` はイシュー #870 で追加した拡張キーで、layout
//!   モジュールがヘッダのセクションメニューのリンク先として使う。参照実装
//!   （fandhe-backend `crates/docs-site/src/nav.rs`）の同キーに倣うが、
//!   `page.path` 重複検査には含めない: 実 `site/nav.toml` は各セクションの
//!   `index_path` がそのセクション内のいずれかの `page.path` と意図的に
//!   一致する値を取るため（例: `index_path = "/guides/"` と
//!   `page.path = "/guides/"` の共存）、含めると誤って `DuplicatePath` を誘発する
//! - `[[section.page]]` array-of-tables（直前の `[[section]]` に属する。
//!   `title` / `source` / `path` の 3 キー）
//! - `key = "value"`（ダブルクォート文字列のみ。エスケープは `\"` `\\` `\n` `\t` の
//!   4 種類のみ対応）
//!
//! 整数・真偽値・inline table・複数行文字列・配列などは非対応であり、
//! 出現した場合はエラーにする。
//!
//! # 参照実装との差分（理由）
//!
//! `fandhe-backend/crates/docs-site/src/nav.rs` は `fandhe_frontend_core::Node` を
//! 使ったサイドバー・ヘッダーナビの HTML 生成まで担うが、本リポジトリは
//! `fandhe-frontend` 系クレートへ依存できない（deps-policy.md の許容 9 区分外・
//! ユーザー承認必須のため追加しない）。よって本モジュールはデータモデルと検証のみを
//! 担う FS 非依存の純粋パーサーとし、HTML 生成は #870 の自作 layout モジュールへ
//! 委ねる。

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

/// `nav.toml` 入力の上限サイズ。DoS 抑止のため超過は即エラーにする。
/// 行単位・非再帰パースのためネスト深度に起因する問題は生じない。
pub const MAX_INPUT_BYTES: usize = 1024 * 1024;

/// `nav.toml` 全体をパースした結果のモデル。フィールドはすべて検証済み
/// （必須キー充足・`page.path` / `site.base_path` 形式・`page.path` 重複なし）。
/// `page.source` の実ファイル存在は [`validate_sources`] が別途担う
/// （パーサ本体を FS 非依存に保ち単体テストしやすくするため）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nav {
    /// サイト全体設定。
    pub site: Site,
    /// 宣言順を保持したセクション列（1 件以上）。
    pub sections: Vec<Section>,
}

/// `[site]` テーブル。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    /// サイトタイトル。
    pub title: String,
    /// GitHub Pages プロジェクトサイト等でルート以外にホストする場合のベースパス。
    /// `""` または `/` 始まり・`/` 終わりでない文字列。
    pub base_path: String,
}

/// `[[section]]` 1 件分。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// サイドバーの見出しとして表示するセクションタイトル。
    pub title: String,
    /// ヘッダのセクションメニューのリンク先（任意）。指定する場合は `page.path` と
    /// 同じ形式規則（`/` 始まり・`/` 終わり・セグメントは英数字・`-`・`_`）を満たす
    /// 必要がある（[`validate_page_path`]）。ページ実在との突合は行わない
    /// （#872 linkcheck の責務。イシュー #870 実装計画 §2.3）。
    pub index_path: Option<String>,
    /// 宣言順を保持したページ列（1 件以上。空セクションはパース時点でエラー）。
    pub pages: Vec<Page>,
}

/// `[[section.page]]` 1 件分。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// サイドバー・前後ナビのリンクテキスト。
    pub title: String,
    /// Markdown ソースファイルの `repo_root` からの相対パス
    /// （[`validate_sources`] が実在確認する）。
    pub source: String,
    /// 出力 URL パス。`/` 始まり・`/` 終わり必須。
    pub path: String,
}

/// [`parse_nav`] / [`validate_sources`] の失敗理由。
///
/// `Display` 実装は行番号と理由・値の断片のみを含み、入力全文・絶対パス・
/// 環境変数は含めない（`.claude/rules/security.md` の機微情報露出防止方針）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavError {
    /// 入力サイズが [`MAX_INPUT_BYTES`] を超えた。
    TooLarge,
    /// 構文エラー（未知のテーブル・未知のキー・非対応の値型・重複キー等）。
    Parse {
        /// 1 始まりの行番号。ファイル全体に関するエラーは `0`。
        line: usize,
        /// エラー理由（入力値の断片は含めても入力全文は含めない）。
        message: String,
    },
    /// 複数セクションにまたがり `page.path` が重複している。
    DuplicatePath(String),
    /// `page.source` が `repo_root` 配下のファイルとして実在しない。
    MissingSource(String),
    /// `page.source` が相対パスの安全条件（絶対パス禁止・`..` 禁止・`\` 禁止・
    /// 非空）を満たさない（パストラバーサル対策）。
    UnsafeSource(String),
    /// `page.path` が `/` 始まり・`/` 終わり、またはセグメントのホワイトリスト
    /// （英数字・`-`・`_`）を満たさない（`..` によるパストラバーサルを含む）。
    InvalidPagePath(String),
    /// `section.index_path` が指定されているが `page.path` と同じ形式規則を
    /// 満たさない。
    InvalidIndexPath(String),
    /// `site.base_path` が `""` または `/` 始まり・`/` 終わりでない、の形式を満たさない。
    InvalidBasePath(String),
    /// 必須キーが欠落している。
    MissingKey {
        /// 欠落箇所（`"site"` / `"section"` / `"section.page"`）。
        context: String,
        /// 欠落したキー名。
        key: String,
    },
    /// セクションにページが 1 件も宣言されていない。
    EmptySection(String),
    /// `[[section]]` が 1 件も宣言されていない。
    NoSections,
}

impl fmt::Display for NavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NavError::TooLarge => {
                write!(f, "nav.toml exceeds the {MAX_INPUT_BYTES} byte size limit")
            }
            NavError::Parse { line, message } => write!(f, "nav.toml:{line}: {message}"),
            NavError::DuplicatePath(path) => write!(f, "duplicate page.path `{path}`"),
            NavError::MissingSource(source) => {
                write!(f, "page.source `{source}` does not exist under repo_root")
            }
            NavError::UnsafeSource(source) => {
                write!(f, "page.source `{source}` is not a safe relative path")
            }
            NavError::InvalidPagePath(path) => write!(
                f,
                "page.path `{path}` must start and end with `/` with segments limited to alphanumerics, `-`, `_`"
            ),
            NavError::InvalidIndexPath(path) => write!(
                f,
                "section.index_path `{path}` must start and end with `/` with segments limited to alphanumerics, `-`, `_`"
            ),
            NavError::InvalidBasePath(base_path) => write!(
                f,
                "site.base_path `{base_path}` must be \"\" or start with `/` and not end with `/`"
            ),
            NavError::MissingKey { context, key } => {
                write!(f, "missing required key `{key}` in [{context}]")
            }
            NavError::EmptySection(title) => write!(f, "section `{title}` has no pages"),
            NavError::NoSections => {
                write!(f, "nav.toml must declare at least one [[section]]")
            }
        }
    }
}

impl std::error::Error for NavError {}

/// パース中に組み立て途上のセクション。必須キーの充足は全行走査後にまとめて
/// 検証する（欠落順序に依存しない一貫したエラーにするため）。
struct SectionBuilder {
    title: Option<String>,
    index_path: Option<String>,
    pages: Vec<PageBuilder>,
}

struct PageBuilder {
    title: Option<String>,
    source: Option<String>,
    path: Option<String>,
}

/// 現在どのテーブルの直下を走査しているかを表す。`[[section.page]]` は
/// 直前に開始された `[[section]]`（`sections` の末尾）に属する。
enum Ctx {
    None,
    Site,
    Section(usize),
    Page(usize, usize),
}

fn parse_err(line: usize, message: impl Into<String>) -> NavError {
    NavError::Parse {
        line,
        message: message.into(),
    }
}

/// テーブルヘッダ・値の後続部分が「空、または `#` 始まりのコメント」であることを
/// 検証する。それ以外の残存文字列はサブセット外構文として拒否する。
fn check_trailing(rest: &str, line: usize) -> Result<(), NavError> {
    let rest = rest.trim_start();
    if rest.is_empty() || rest.starts_with('#') {
        Ok(())
    } else {
        Err(parse_err(
            line,
            format!("unexpected trailing content `{rest}`"),
        ))
    }
}

/// `value_part`（`=` の右側、先頭空白は trim 済み）からダブルクォート文字列 1 個を
/// 読み取る。エスケープは `\"` `\\` `\n` `\t` のみ対応。
/// 戻り値は `(パース済み文字列, 閉じクォート以降の残り文字列)`。
fn parse_quoted_string(value_part: &str, line: usize) -> Result<(String, &str), NavError> {
    let mut chars = value_part.char_indices();
    match chars.next() {
        Some((_, '"')) => {}
        _ => {
            return Err(parse_err(
                line,
                "expected a double-quoted string value (this parser accepts no other TOML value type)",
            ));
        }
    }

    let mut out = String::new();
    loop {
        match chars.next() {
            None => return Err(parse_err(line, "unterminated string literal")),
            Some((idx, '"')) => {
                let remainder = &value_part[idx + '"'.len_utf8()..];
                return Ok((out, remainder));
            }
            Some((_, '\\')) => match chars.next() {
                Some((_, '"')) => out.push('"'),
                Some((_, '\\')) => out.push('\\'),
                Some((_, 'n')) => out.push('\n'),
                Some((_, 't')) => out.push('\t'),
                Some((_, other)) => {
                    return Err(parse_err(
                        line,
                        format!("unsupported escape sequence `\\{other}`"),
                    ));
                }
                None => return Err(parse_err(line, "unterminated escape sequence")),
            },
            Some((_, c)) => out.push(c),
        }
    }
}

fn set_once(
    slot: &mut Option<String>,
    value: String,
    line: usize,
    name: &str,
) -> Result<(), NavError> {
    if slot.is_some() {
        return Err(parse_err(line, format!("duplicate key `{name}`")));
    }
    *slot = Some(value);
    Ok(())
}

/// `site.base_path` を検証する。`""` または `/` 始まり・`/` 終わりでない形式に加え、
/// `page.path` と同じセグメントホワイトリスト（[`is_safe_path_segment`]）を課す。
/// 形式（先頭・末尾 `/`）だけの検証では `base_path = "/../../etc"` のような
/// パストラバーサル用セグメント（`..`）を通してしまうため（`page.path` 側の
/// [`validate_page_path`] と同方針。#870 が `base_path` を出力レイアウトの
/// プレフィックスとして使う前提のため、こちらも fail-closed に揃える）。
fn validate_base_path(base_path: &str) -> Result<(), NavError> {
    if base_path.is_empty() {
        return Ok(());
    }
    if !base_path.starts_with('/') || base_path.ends_with('/') {
        return Err(NavError::InvalidBasePath(base_path.to_string()));
    }
    let inner = &base_path[1..];
    if inner.split('/').all(is_safe_path_segment) {
        Ok(())
    } else {
        Err(NavError::InvalidBasePath(base_path.to_string()))
    }
}

/// `segment` が出力パス片として安全（英数字・`-`・`_` のみ、非空）かを検証する。
/// `#870` が `page.path` を `--out` 配下の実ファイルパスとして使う前提のため、
/// `/` 始まり・`/` 終わりの形式検証だけでは `path = "/../../etc/"` のような
/// パストラバーサル用セグメント（`..`）を通してしまう。ここで fail-closed に
/// 拒否し、書き出し側（#870）へ危険な `page.path` を渡さない多層防御とする
/// （`page.source` 側の [`validate_source_shape`] と同方針）。
fn is_safe_path_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// `page.path` / `section.index_path` に共通の形式規則を検査する（`/` 始まり・
/// `/` 終わり・内部セグメントは英数字・`-`・`_` のみ）。呼び出し元がそれぞれ
/// 異なる [`NavError`] バリアントへ包む。
fn path_format_is_valid(path: &str) -> bool {
    if !path.starts_with('/') || !path.ends_with('/') {
        return false;
    }
    if path.len() == 1 {
        // "/"（サイトトップ）はセグメントなしで許可する。開始・終了の '/' が
        // 同一バイトを指すため、下の `path[1..path.len() - 1]` スライス
        // （1..0）は範囲が逆転してパニックする。長さ 1 の場合はスライス計算に
        // 入る前に早期リターンする（参照実装 fandhe-backend #473 の教訓）。
        return true;
    }
    let inner = &path[1..path.len() - 1];
    if inner.is_empty() {
        // "//" 等、"/" 以外で内部セグメントが空になるケース。サイトトップは
        // 上の `path.len() == 1` 分岐（`path == "/"`）だけで既に許可済みのため、
        // ここに到達するのは "//" のような縮退表記のみであり、`"/"` とは
        // 別文字列として `DuplicatePath` 検査（文字列一致）をすり抜けて共存しうる
        // （#870 が `page.path` を `--out` 配下の実ファイル書き出しパスとして使う
        // ため、"/" と "//" が別ページとして共存すると出力先の衝突・上書きに
        // つながる）。よってここは許可せず拒否する。
        return false;
    }
    inner.split('/').all(is_safe_path_segment)
}

fn validate_page_path(path: &str) -> Result<(), NavError> {
    if path_format_is_valid(path) {
        Ok(())
    } else {
        Err(NavError::InvalidPagePath(path.to_string()))
    }
}

/// `section.index_path` の形式検証。ページ実在との突合は行わない（#872 の責務。
/// モジュール冒頭コメント参照）。
fn validate_index_path(path: &str) -> Result<(), NavError> {
    if path_format_is_valid(path) {
        Ok(())
    } else {
        Err(NavError::InvalidIndexPath(path.to_string()))
    }
}

/// `source` の先頭が Windows ドライブレター絶対パス（`C:/...`・`C:\...`）の
/// 形式かを判定する。`\` は [`validate_source_shape`]側で別途拒否済みだが、
/// `C:/...` は `/` 始まりではないため単純な `starts_with('/')` 検査を
/// すり抜ける。Cursor Bugbot 指摘（PR #890）: これを許すと後段の
/// `repo_root.join(source)` が `Path::join` のセマンティクスにより絶対パス
/// 側（`source`）で `repo_root` を丸ごと置き換えてしまい、`validate_sources`
/// の `repo_root` 配下チェックを迂回して任意ファイルを指せる。
fn looks_like_windows_drive_path(source: &str) -> bool {
    let mut chars = source.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.next() == Some(':')
        && matches!(chars.next(), Some('/') | Some('\\') | None)
}

/// `source` が相対パスの安全条件（非空・絶対パス禁止・`..` セグメント禁止・
/// `\` 禁止・Windows ドライブレター絶対パス禁止）を満たすかを構文レベルで
/// 検証する（パストラバーサル対策の早期検出。実ファイル存在確認・シンボリック
/// リンク経由の `repo_root` 脱出防止は [`validate_sources`] が別途行う）。
fn validate_source_shape(source: &str) -> Result<(), NavError> {
    let looks_safe = !source.is_empty()
        && !source.starts_with('/')
        && !source.contains('\\')
        && !looks_like_windows_drive_path(source)
        && source.split('/').all(|segment| segment != "..");
    if looks_safe {
        Ok(())
    } else {
        Err(NavError::UnsafeSource(source.to_string()))
    }
}

/// `nav.toml` の内容（文字列）をパースし、スキーマ・`page.path` /
/// `site.base_path` の形式・`page.path` の重複検証までを行う純関数。
/// ファイルシステムには一切アクセスしない（`page.source` の実在確認は
/// [`validate_sources`] を別途呼ぶこと）。
///
/// # Errors
///
/// 対応外の TOML 構文・必須キー欠落・空セクション・セクション 0 件・
/// `page.path` 重複・`page.path` / `site.base_path` の形式違反・`page.source` の
/// 構文上の危険性（絶対パス・`..`・`\`）のいずれかがあれば [`NavError`] を返す。
pub fn parse_nav(input: &str) -> Result<Nav, NavError> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(NavError::TooLarge);
    }

    let mut ctx = Ctx::None;
    let mut site_title: Option<String> = None;
    let mut site_base_path: Option<String> = None;
    let mut sections: Vec<SectionBuilder> = Vec::new();

    for (line_no0, raw_line) in input.lines().enumerate() {
        let line = line_no0 + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("[[") {
            let end = rest
                .find("]]")
                .ok_or_else(|| parse_err(line, "expected closing `]]`"))?;
            let header = rest[..end].trim();
            check_trailing(&rest[end + 2..], line)?;
            match header {
                "section" => {
                    sections.push(SectionBuilder {
                        title: None,
                        index_path: None,
                        pages: Vec::new(),
                    });
                    ctx = Ctx::Section(sections.len() - 1);
                }
                "section.page" => {
                    let sidx = sections.len().checked_sub(1).ok_or_else(|| {
                        parse_err(line, "[[section.page]] appeared before any [[section]]")
                    })?;
                    sections[sidx].pages.push(PageBuilder {
                        title: None,
                        source: None,
                        path: None,
                    });
                    let pidx = sections[sidx].pages.len() - 1;
                    ctx = Ctx::Page(sidx, pidx);
                }
                other => return Err(parse_err(line, format!("unknown table `[[{other}]]`"))),
            }
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix('[') {
            let end = rest
                .find(']')
                .ok_or_else(|| parse_err(line, "expected closing `]`"))?;
            let header = rest[..end].trim();
            check_trailing(&rest[end + 1..], line)?;
            match header {
                "site" => ctx = Ctx::Site,
                other => return Err(parse_err(line, format!("unknown table `[{other}]`"))),
            }
            continue;
        }

        let eq = trimmed
            .find('=')
            .ok_or_else(|| parse_err(line, "expected `key = \"value\"`"))?;
        let key = trimmed[..eq].trim();
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(parse_err(line, format!("invalid key `{key}`")));
        }
        let value_part = trimmed[eq + 1..].trim_start();
        let (value, remainder) = parse_quoted_string(value_part, line)?;
        check_trailing(remainder, line)?;

        match ctx {
            Ctx::None => return Err(parse_err(line, "key-value pair outside of any table")),
            Ctx::Site => match key {
                "title" => set_once(&mut site_title, value, line, "site.title")?,
                "base_path" => set_once(&mut site_base_path, value, line, "site.base_path")?,
                other => return Err(parse_err(line, format!("unknown key `{other}` in [site]"))),
            },
            Ctx::Section(sidx) => match key {
                "title" => set_once(&mut sections[sidx].title, value, line, "section.title")?,
                "index_path" => set_once(
                    &mut sections[sidx].index_path,
                    value,
                    line,
                    "section.index_path",
                )?,
                other => {
                    return Err(parse_err(
                        line,
                        format!("unknown key `{other}` in [[section]]"),
                    ));
                }
            },
            Ctx::Page(sidx, pidx) => {
                let page = &mut sections[sidx].pages[pidx];
                match key {
                    "title" => set_once(&mut page.title, value, line, "page.title")?,
                    "source" => set_once(&mut page.source, value, line, "page.source")?,
                    "path" => set_once(&mut page.path, value, line, "page.path")?,
                    other => {
                        return Err(parse_err(
                            line,
                            format!("unknown key `{other}` in [[section.page]]"),
                        ));
                    }
                }
            }
        }
    }

    let site = Site {
        title: site_title.ok_or_else(|| NavError::MissingKey {
            context: "site".to_string(),
            key: "title".to_string(),
        })?,
        base_path: site_base_path.ok_or_else(|| NavError::MissingKey {
            context: "site".to_string(),
            key: "base_path".to_string(),
        })?,
    };
    validate_base_path(&site.base_path)?;

    if sections.is_empty() {
        return Err(NavError::NoSections);
    }

    let mut seen_paths: BTreeSet<String> = BTreeSet::new();
    let mut out_sections = Vec::with_capacity(sections.len());
    for section in sections {
        let title = section.title.ok_or_else(|| NavError::MissingKey {
            context: "section".to_string(),
            key: "title".to_string(),
        })?;
        if let Some(index_path) = &section.index_path {
            validate_index_path(index_path)?;
        }
        if section.pages.is_empty() {
            return Err(NavError::EmptySection(title));
        }
        let mut out_pages = Vec::with_capacity(section.pages.len());
        for page in section.pages {
            let title = page.title.ok_or_else(|| NavError::MissingKey {
                context: "section.page".to_string(),
                key: "title".to_string(),
            })?;
            let source = page.source.ok_or_else(|| NavError::MissingKey {
                context: "section.page".to_string(),
                key: "source".to_string(),
            })?;
            let path = page.path.ok_or_else(|| NavError::MissingKey {
                context: "section.page".to_string(),
                key: "path".to_string(),
            })?;
            validate_page_path(&path)?;
            validate_source_shape(&source)?;
            if !seen_paths.insert(path.clone()) {
                return Err(NavError::DuplicatePath(path));
            }
            out_pages.push(Page {
                title,
                source,
                path,
            });
        }
        out_sections.push(Section {
            title,
            index_path: section.index_path,
            pages: out_pages,
        });
    }

    Ok(Nav {
        site,
        sections: out_sections,
    })
}

/// 各 `page.source` が `repo_root` 配下の実ファイルとして存在することを検証する。
/// [`parse_nav`] から FS アクセスを分離し、単体テストをファイルシステムに
/// 依存させないための独立関数（実装計画 §4）。
///
/// codex-review P0 指摘（PR #890）: `repo_root.join(&page.source).is_file()`
/// のみの検証は、パス中の中間コンポーネントがシンボリックリンクで
/// `repo_root` 外（例: `/etc/passwd`）を指す場合でも、解決結果が通常ファイル
/// であれば成功してしまう。`validate_source_shape` の構文検査（`..` 禁止等）は
/// リンク先までは追跡できないため、ここで実ファイルシステム上の解決結果を
/// `canonicalize` し、`repo_root` の正規化パス配下であることを `starts_with`
/// で fail-closed に確認する（シンボリックリンク経由の脱出を拒否する回帰
/// テストは `tests` モジュールの `validate_sources_rejects_symlink_escape` を
/// 参照）。
///
/// # Errors
///
/// - `repo_root` 自体が `canonicalize` できない（存在しない・権限不足等）場合は
///   [`NavError::MissingSource`]（`repo_root` の表示文字列を含む）を返す。
/// - いずれかの `page.source` が `repo_root` 配下のファイルとして存在しない
///   （または `canonicalize` に失敗する）場合、最初に見つかった不在ファイルに
///   ついて [`NavError::MissingSource`] を返す。
/// - `page.source` が実在するが、シンボリックリンクの解決の結果
///   `repo_root` の外を指す場合は [`NavError::UnsafeSource`] を返す。
pub fn validate_sources(nav: &Nav, repo_root: &Path) -> Result<(), NavError> {
    let canonical_root = repo_root
        .canonicalize()
        .map_err(|_| NavError::MissingSource(repo_root.display().to_string()))?;
    for section in &nav.sections {
        for page in &section.pages {
            let full_path = repo_root.join(&page.source);
            let canonical_path = full_path
                .canonicalize()
                .map_err(|_| NavError::MissingSource(page.source.clone()))?;
            if !canonical_path.is_file() {
                return Err(NavError::MissingSource(page.source.clone()));
            }
            if !canonical_path.starts_with(&canonical_root) {
                return Err(NavError::UnsafeSource(page.source.clone()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[site]
title = "rust-ai-library docs"
base_path = "/rust-ai-library"

[[section]]
title = "Guide"

[[section.page]]
title = "Introduction"
source = "docs/guide/intro.md"
path = "/guide/intro/"

[[section.page]]
title = "Getting Started"
source = "docs/guide/getting-started.md"
path = "/guide/getting-started/"

[[section]]
title = "Reference"

[[section.page]]
title = "API"
source = "docs/reference/api.md"
path = "/reference/api/"
"#;

    // ---- 正常系 ----

    #[test]
    fn parses_site_sections_and_pages_in_declaration_order() {
        let nav = parse_nav(SAMPLE).expect("valid nav.toml should parse");
        assert_eq!(nav.site.title, "rust-ai-library docs");
        assert_eq!(nav.site.base_path, "/rust-ai-library");
        assert_eq!(nav.sections.len(), 2);
        assert_eq!(nav.sections[0].title, "Guide");
        assert_eq!(nav.sections[0].pages.len(), 2);
        assert_eq!(nav.sections[0].pages[0].title, "Introduction");
        assert_eq!(nav.sections[0].pages[0].source, "docs/guide/intro.md");
        assert_eq!(nav.sections[0].pages[0].path, "/guide/intro/");
        assert_eq!(nav.sections[0].pages[1].title, "Getting Started");
        assert_eq!(nav.sections[1].title, "Reference");
        assert_eq!(nav.sections[1].pages.len(), 1);
        assert_eq!(nav.sections[1].pages[0].path, "/reference/api/");
    }

    #[test]
    fn supports_full_line_and_trailing_comments() {
        let input = r#"
# full line comment
[site]
title = "Docs" # trailing comment
base_path = ""

[[section]] # comment after header
title = "Guide"

[[section.page]]
title = "Intro"
source = "intro.md"
path = "/intro/"
"#;
        let nav = parse_nav(input).expect("comments should be tolerated");
        assert_eq!(nav.site.title, "Docs");
        assert_eq!(nav.site.base_path, "");
    }

    #[test]
    fn supports_basic_string_escapes() {
        let input = r#"
[site]
title = "Line1\nLine2 \"quoted\" \\backslash\\"
base_path = ""

[[section]]
title = "S"

[[section.page]]
title = "P"
source = "p.md"
path = "/p/"
"#;
        let nav = parse_nav(input).expect("escapes should be supported");
        assert_eq!(nav.site.title, "Line1\nLine2 \"quoted\" \\backslash\\");
    }

    // ---- 異常系 ----

    #[test]
    fn rejects_duplicate_path_across_sections() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/dup/"

[[section]]
title = "B"

[[section.page]]
title = "P2"
source = "p2.md"
path = "/dup/"
"#;
        match parse_nav(input) {
            Err(NavError::DuplicatePath(path)) => assert_eq!(path, "/dup/"),
            other => panic!("expected DuplicatePath, got {other:?}"),
        }
    }

    #[test]
    fn parses_optional_section_index_path() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guides"
index_path = "/guides/"

[[section.page]]
title = "Guides"
source = "guides.md"
path = "/guides/"
"#;
        let nav = parse_nav(input).expect("index_path should be accepted");
        assert_eq!(nav.sections[0].index_path.as_deref(), Some("/guides/"));
    }

    #[test]
    fn section_index_path_is_optional() {
        let nav = parse_nav(SAMPLE).expect("valid nav.toml should parse");
        assert_eq!(nav.sections[0].index_path, None);
    }

    #[test]
    fn index_path_duplicating_a_page_path_does_not_trigger_duplicate_path() {
        // 実 `site/nav.toml` は `index_path` がそのセクションの `page.path` と
        // 意図的に一致する（例: `/guides/`）。dedup 対象は `page.path` のみで
        // あるべき回帰テスト。
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guides"
index_path = "/guides/"

[[section.page]]
title = "Guides"
source = "guides.md"
path = "/guides/"
"#;
        assert!(parse_nav(input).is_ok());
    }

    #[test]
    fn rejects_invalid_index_path_format() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guides"
index_path = "guides"

[[section.page]]
title = "Guides"
source = "guides.md"
path = "/guides/"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidIndexPath(_))
        ));
    }

    #[test]
    fn rejects_unknown_key_in_site_table() {
        let input = r#"
[site]
title = "Docs"
base_path = ""
unknown = "x"

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1/"
"#;
        assert!(matches!(parse_nav(input), Err(NavError::Parse { .. })));
    }

    #[test]
    fn rejects_unknown_table() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[unknown_table]]
title = "A"
"#;
        assert!(matches!(parse_nav(input), Err(NavError::Parse { .. })));
    }

    #[test]
    fn rejects_missing_required_site_key() {
        let input = r#"
[site]
title = "Docs"

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1/"
"#;
        match parse_nav(input) {
            Err(NavError::MissingKey { context, key }) => {
                assert_eq!(context, "site");
                assert_eq!(key, "base_path");
            }
            other => panic!("expected MissingKey, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_required_page_key() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
path = "/p1/"
"#;
        match parse_nav(input) {
            Err(NavError::MissingKey { context, key }) => {
                assert_eq!(context, "section.page");
                assert_eq!(key, "source");
            }
            other => panic!("expected MissingKey, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_section() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Empty"
"#;
        match parse_nav(input) {
            Err(NavError::EmptySection(title)) => assert_eq!(title, "Empty"),
            other => panic!("expected EmptySection, got {other:?}"),
        }
    }

    #[test]
    fn rejects_no_sections() {
        let input = r#"
[site]
title = "Docs"
base_path = ""
"#;
        assert!(matches!(parse_nav(input), Err(NavError::NoSections)));
    }

    #[test]
    fn rejects_section_page_before_any_section() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section.page]]
title = "Orphan"
source = "orphan.md"
path = "/orphan/"
"#;
        assert!(matches!(parse_nav(input), Err(NavError::Parse { .. })));
    }

    #[test]
    fn rejects_unsupported_value_types() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1/"
weight = 1
"#;
        assert!(matches!(parse_nav(input), Err(NavError::Parse { .. })));
    }

    #[test]
    fn rejects_unsupported_escape_sequence() {
        let input = r#"
[site]
title = "bad \x escape"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1/"
"#;
        assert!(matches!(parse_nav(input), Err(NavError::Parse { .. })));
    }

    #[test]
    fn rejects_unterminated_string() {
        let input = "[site]\ntitle = \"unterminated\nbase_path = \"\"\n";
        assert!(matches!(parse_nav(input), Err(NavError::Parse { .. })));
    }

    #[test]
    fn rejects_input_larger_than_size_limit() {
        let mut input = String::from("[site]\ntitle = \"");
        input.push_str(&"a".repeat(MAX_INPUT_BYTES + 1));
        input.push_str("\"\nbase_path = \"\"\n");
        assert!(matches!(parse_nav(&input), Err(NavError::TooLarge)));
    }

    #[test]
    fn rejects_invalid_base_path() {
        let input = r#"
[site]
title = "Docs"
base_path = "no-leading-slash"

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1/"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidBasePath(_))
        ));
    }

    #[test]
    fn rejects_base_path_with_parent_traversal_segment() {
        // `page.path` 側の `is_safe_path_segment` ホワイトリストとの非対称
        // （レビュー指摘）を解消する回帰テスト。
        let input = r#"
[site]
title = "Docs"
base_path = "/../../etc"

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1/"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidBasePath(_))
        ));
    }

    #[test]
    fn rejects_page_path_without_leading_slash() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "p1/"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidPagePath(_))
        ));
    }

    #[test]
    fn rejects_page_path_without_trailing_slash() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidPagePath(_))
        ));
    }

    #[test]
    fn rejects_page_path_with_parent_traversal_segment() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/../../etc/"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidPagePath(_))
        ));
    }

    #[test]
    fn rejects_double_slash_page_path_degenerate_case() {
        // "/" と "//" が別々の妥当な path として共存すると、#870 の実ファイル
        // 書き出しで衝突・上書きにつながりうる（レビュー指摘）。
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "//"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidPagePath(_))
        ));
    }

    #[test]
    fn rejects_double_slash_and_root_path_coexisting_across_sections() {
        // "//" 単体の拒否だけでなく、"/" と "//" が別セクションに分かれて
        // 存在するケースでも重複検査をすり抜けず拒否されることを確認する。
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "Top"
source = "index.md"
path = "/"

[[section]]
title = "B"

[[section.page]]
title = "Dup"
source = "dup.md"
path = "//"
"#;
        assert!(matches!(
            parse_nav(input),
            Err(NavError::InvalidPagePath(_))
        ));
    }

    #[test]
    fn accepts_site_root_page_path() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "Top"
source = "index.md"
path = "/"
"#;
        let nav = parse_nav(input).expect("path = \"/\" should be accepted as the site root");
        assert_eq!(nav.sections[0].pages[0].path, "/");
    }

    #[test]
    fn rejects_parent_traversal_in_source() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "../secret.md"
path = "/p1/"
"#;
        match parse_nav(input) {
            Err(NavError::UnsafeSource(source)) => assert_eq!(source, "../secret.md"),
            other => panic!("expected UnsafeSource, got {other:?}"),
        }
    }

    #[test]
    fn rejects_absolute_path_source() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "/etc/passwd"
path = "/p1/"
"#;
        assert!(matches!(parse_nav(input), Err(NavError::UnsafeSource(_))));
    }

    #[test]
    fn rejects_backslash_in_source() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "a\\b.md"
path = "/p1/"
"#;
        assert!(matches!(parse_nav(input), Err(NavError::UnsafeSource(_))));
    }

    // ---- validate_sources（FS 依存） ----

    /// テスト専用の一時ディレクトリ。`Drop` でベストエフォート削除する。
    /// 外部クレート（`tempfile` 等）を追加せず `std::env::temp_dir()` +
    /// プロセス固有サフィックスで代用する（REQ-1 v2: 外部依存ゼロを維持する）。
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "rust-ai-library-docs-site-nav-test-{tag}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create temp dir for nav.rs test");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn validate_sources_reports_missing_source_file() {
        let temp = TempDir::new("missing-source");
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "does-not-exist.md"
path = "/p1/"
"#;
        let nav = parse_nav(input).expect("structurally valid nav.toml should parse");
        match validate_sources(&nav, &temp.0) {
            Err(NavError::MissingSource(source)) => assert_eq!(source, "does-not-exist.md"),
            other => panic!("expected MissingSource, got {other:?}"),
        }
    }

    #[test]
    fn validate_sources_accepts_existing_files() {
        let temp = TempDir::new("existing-source");
        std::fs::write(temp.0.join("p1.md"), b"# hello").expect("write fixture source file");
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "p1.md"
path = "/p1/"
"#;
        let nav = parse_nav(input).expect("valid nav.toml should parse");
        assert!(validate_sources(&nav, &temp.0).is_ok());
    }

    /// codex-review P0 指摘（PR #890）の回帰テスト: `page.source` が
    /// シンボリックリンクを経由して `repo_root` 外の実ファイルを指す場合、
    /// リンク先が通常ファイルであっても [`validate_sources`] が
    /// `UnsafeSource` で fail-closed に拒否することを確認する
    /// （symlink を使わない環境〈Windows 等〉では対象外のため unix 限定）。
    #[cfg(unix)]
    #[test]
    fn validate_sources_rejects_symlink_escape() {
        let root = TempDir::new("symlink-escape-root");
        let outside = TempDir::new("symlink-escape-outside");
        let secret = outside.0.join("secret.md");
        std::fs::write(&secret, b"outside repo_root").expect("write fixture outside repo_root");
        std::os::unix::fs::symlink(&secret, root.0.join("linked.md"))
            .expect("create symlink escaping repo_root for test fixture");

        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "linked.md"
path = "/p1/"
"#;
        let nav = parse_nav(input).expect("structurally valid nav.toml should parse");
        match validate_sources(&nav, &root.0) {
            Err(NavError::UnsafeSource(source)) => assert_eq!(source, "linked.md"),
            other => panic!("expected UnsafeSource for symlink escape, got {other:?}"),
        }
    }

    /// Cursor Bugbot 指摘（PR #890）の回帰テスト: `validate_source_shape` が
    /// `/` 始まりのみを絶対パス扱いしているため、Windows ドライブレター
    /// 絶対パス（例: `C:/secret.txt`）を相対パスと誤認して通してしまうと、
    /// 後段の `repo_root.join(source)` が `Path::join` のセマンティクスにより
    /// `repo_root` を丸ごと破棄して絶対パス側を採用してしまう
    /// （`std::path::Path::join` のドキュメント参照）。構文検証時点で
    /// fail-closed に拒否することを確認する。
    #[test]
    fn parse_nav_rejects_windows_drive_path_source() {
        let input = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "A"

[[section.page]]
title = "P1"
source = "C:/secret.txt"
path = "/p1/"
"#;
        assert!(matches!(parse_nav(input), Err(NavError::UnsafeSource(_))));
    }
}
