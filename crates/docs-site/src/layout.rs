//! ページ骨格（`<html>` 全体）の組み立て: ヘッダ・サイドバー・本文。
//!
//! # 呼び出し文脈
//!
//! `build.rs` が [`nav::parse_nav`](crate::nav::parse_nav) 済みの [`Nav`] と
//! `markdown::markdown_to_nodes` の変換結果（本文ノード列）を [`docs_page`] へ渡し、
//! 得られた [`crate::html::Node`] を `html::render` で文字列化してから
//! `<!DOCTYPE html>` を前置してファイルへ書き出す（実装計画 §2.4・§2.6）。
//!
//! # スコープ境界（イシュー #870 時点）
//!
//! 3 カラム TOC・全文検索 UI・テーマトグルボタン・FOUC 抑止スクリプトは
//! 兄弟イシュー #871 のスコープであり実装しない。本モジュールはそれらを
//! 後付けできる骨格（ヘッダのアクション領域・CSS 側の `data-theme` フック
//! （`theme.rs`/`assets/site.css`）まで）に留める。

use crate::html::Node;
use crate::nav::Nav;

/// rust-ai-library の GitHub リポジトリ URL。ヘッダの外部リンク先として使う。
const GITHUB_REPO_URL: &str = "https://github.com/Fandhe-AI/rust-ai-library";

/// `base_path` と `relative`（`/` 始まりでも始まりでなくてもよい）を結合し、
/// GitHub Pages プロジェクトサイト等の `base_path` プレフィックスを考慮した
/// href を組み立てる単一実装点（実装計画 §2.4「asset_href」）。
///
/// - `base_path` は `""` または `/` 始まり・`/` 終わりでない文字列
///   （[`crate::nav::Nav`] の `site.base_path` はパース時点でこの形式を保証済み）
/// - `relative` の先頭 `/` は正規化のため一旦除去してから結合する（二重スラッシュ
///   防止）。`relative` が `/` 単体（サイトトップ）の場合は末尾スラッシュを保つ
pub fn asset_href(base_path: &str, relative: &str) -> String {
    let trimmed = relative.trim_start_matches('/');
    format!("{base_path}/{trimmed}")
}

/// ヘッダ: サイトタイトル（ルートへのリンク）・`index_path` を持つセクションの
/// メニューリンク・GitHub リポジトリリンク。
fn header(nav: &Nav) -> Node {
    let base_path = nav.site.base_path.as_str();

    let mut menu_items: Vec<Node> = nav
        .sections
        .iter()
        .filter_map(|section| {
            section.index_path.as_ref().map(|index_path| {
                Node::element(
                    "li",
                    vec![],
                    vec![Node::element(
                        "a",
                        vec![("href".to_string(), asset_href(base_path, index_path))],
                        vec![Node::text(section.title.clone())],
                    )],
                )
            })
        })
        .collect();

    menu_items.push(Node::element(
        "li",
        vec![],
        vec![Node::element(
            "a",
            vec![
                ("href".to_string(), GITHUB_REPO_URL.to_string()),
                ("target".to_string(), "_blank".to_string()),
                // tabnabbing 対策（.claude/rules/security.md A05）。
                ("rel".to_string(), "noopener noreferrer".to_string()),
            ],
            vec![Node::text("GitHub")],
        )],
    ));

    Node::element(
        "header",
        vec![("class".to_string(), "site-header".to_string())],
        vec![
            Node::element(
                "div",
                vec![("class".to_string(), "site-title".to_string())],
                vec![Node::element(
                    "a",
                    vec![("href".to_string(), asset_href(base_path, "/"))],
                    vec![Node::text(nav.site.title.clone())],
                )],
            ),
            Node::element("nav", vec![], vec![Node::element("ul", vec![], menu_items)]),
        ],
    )
}

/// サイドバー: nav.toml の宣言順どおりセクション見出し + ページリンク列。
/// 現在ページ（`current_path`。生の `page.path` と比較する。`base_path` を
/// 含まない）には `aria-current="page"` を付与する。
fn sidebar(nav: &Nav, current_path: &str) -> Node {
    let base_path = nav.site.base_path.as_str();
    let mut children = Vec::new();

    for section in &nav.sections {
        children.push(Node::element(
            "h2",
            vec![],
            vec![Node::text(section.title.clone())],
        ));

        let page_items: Vec<Node> = section
            .pages
            .iter()
            .map(|page| {
                let mut attrs = vec![("href".to_string(), asset_href(base_path, &page.path))];
                if page.path == current_path {
                    attrs.push(("aria-current".to_string(), "page".to_string()));
                }
                Node::element(
                    "li",
                    vec![],
                    vec![Node::element(
                        "a",
                        attrs,
                        vec![Node::text(page.title.clone())],
                    )],
                )
            })
            .collect();
        children.push(Node::element("ul", vec![], page_items));
    }

    Node::element(
        "aside",
        vec![("class".to_string(), "site-sidebar".to_string())],
        vec![Node::element("nav", vec![], children)],
    )
}

/// ページ全体（`<html>`）を組み立てる。`body` は本文（`markdown::markdown_to_nodes`
/// の戻り値）。`current_path` はサイドバーの `aria-current` 判定に使う生の
/// `page.path`（`base_path` を含まない）。
///
/// `<!DOCTYPE html>` はここでは前置しない（`build.rs` が書き出し時に前置する。
/// モジュール冒頭コメント参照）。
pub fn docs_page(nav: &Nav, page_title: &str, current_path: &str, body: Vec<Node>) -> Node {
    let base_path = nav.site.base_path.as_str();
    let css_href = asset_href(base_path, "assets/site.css");

    let head = Node::element(
        "head",
        vec![],
        vec![
            Node::element(
                "meta",
                vec![("charset".to_string(), "utf-8".to_string())],
                vec![],
            ),
            Node::element(
                "meta",
                vec![
                    ("name".to_string(), "viewport".to_string()),
                    (
                        "content".to_string(),
                        "width=device-width, initial-scale=1".to_string(),
                    ),
                ],
                vec![],
            ),
            Node::element(
                "title",
                vec![],
                vec![Node::text(format!("{page_title} | {}", nav.site.title))],
            ),
            Node::element(
                "link",
                vec![
                    ("rel".to_string(), "stylesheet".to_string()),
                    ("href".to_string(), css_href),
                ],
                vec![],
            ),
        ],
    );

    let body_node = Node::element(
        "body",
        vec![],
        vec![
            header(nav),
            Node::element(
                "div",
                vec![("class".to_string(), "site-body".to_string())],
                vec![
                    sidebar(nav, current_path),
                    Node::element(
                        "main",
                        vec![("class".to_string(), "site-main".to_string())],
                        vec![Node::element("article", vec![], body)],
                    ),
                ],
            ),
        ],
    );

    Node::element(
        "html",
        vec![("lang".to_string(), "ja".to_string())],
        vec![head, body_node],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::html::render;
    use crate::nav::parse_nav;

    const SAMPLE_NAV: &str = r#"
[site]
title = "rust-ai-library"
base_path = "/rust-ai-library"

[[section]]
title = "Guides"
index_path = "/guides/"

[[section.page]]
title = "Guides"
source = "guides.md"
path = "/guides/"

[[section.page]]
title = "Backends"
source = "backends.md"
path = "/guides/backends/"

[[section]]
title = "API"

[[section.page]]
title = "API"
source = "api.md"
path = "/api/"
"#;

    #[test]
    fn asset_href_joins_base_path_and_relative() {
        assert_eq!(
            asset_href("/rust-ai-library", "assets/site.css"),
            "/rust-ai-library/assets/site.css"
        );
        assert_eq!(asset_href("", "assets/site.css"), "/assets/site.css");
        assert_eq!(
            asset_href("/rust-ai-library", "/guides/"),
            "/rust-ai-library/guides/"
        );
        assert_eq!(asset_href("", "/"), "/");
        assert_eq!(asset_href("/rust-ai-library", "/"), "/rust-ai-library/");
    }

    #[test]
    fn sidebar_preserves_nav_section_and_page_order() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&sidebar(&nav, "/guides/backends/"));
        let guides_pos = html.find("Guides").expect("Guides section present");
        let api_pos = html.find(">API<").expect("API section present");
        assert!(
            guides_pos < api_pos,
            "sections must appear in declaration order"
        );
        assert!(html.contains("href=\"/rust-ai-library/guides/\""));
        assert!(html.contains("href=\"/rust-ai-library/guides/backends/\""));
        assert!(html.contains("href=\"/rust-ai-library/api/\""));
    }

    #[test]
    fn sidebar_marks_current_page_with_aria_current() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&sidebar(&nav, "/guides/backends/"));
        assert!(html.contains(
            "<a href=\"/rust-ai-library/guides/backends/\" aria-current=\"page\">Backends</a>"
        ));
        // 現在ページでないリンクには aria-current を付与しない。
        assert!(!html.contains("<a href=\"/rust-ai-library/guides/\" aria-current=\"page\">"));
    }

    #[test]
    fn header_includes_index_path_menu_and_github_link_with_safe_rel() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&header(&nav));
        assert!(html.contains("href=\"/rust-ai-library/guides/\">Guides</a>"));
        // index_path を持たないセクション（API）はヘッダメニューに現れない。
        assert!(!html.contains(">API</a>"));
        assert!(html.contains(&format!("href=\"{GITHUB_REPO_URL}\"")));
        assert!(html.contains("target=\"_blank\""));
        assert!(html.contains("rel=\"noopener noreferrer\""));
    }

    #[test]
    fn docs_page_wraps_body_and_links_stylesheet() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&docs_page(
            &nav,
            "Guides",
            "/guides/",
            vec![Node::element("h1", vec![], vec![Node::text("Guides")])],
        ));
        assert!(html.starts_with("<html lang=\"ja\">"));
        assert!(html.contains("<title>Guides | rust-ai-library</title>"));
        assert!(html.contains("href=\"/rust-ai-library/assets/site.css\""));
        assert!(html.contains("<article><h1>Guides</h1></article>"));
    }
}
