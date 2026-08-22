//! 最小 HTML ノード層。`markdown`・`layout` の両モジュールが生成物を組み立てる際の
//! 唯一の HTML 文字列化経路を担う。
//!
//! # 呼び出し文脈
//!
//! `markdown::markdown_to_nodes` が Markdown 原稿を [`Node`] 木へ変換し、
//! `layout::docs_page` がそれをページ骨格（ヘッダ・サイドバー・本文）に組み込む。
//! 最終的な HTML 文字列化は必ず [`render`] / [`render_all`] を経由し、`build.rs` は
//! それらの戻り値をそのままファイルへ書き出す（`format!` によるタグ組み立ての
//! 迂回経路をこのモジュール外に作らないことで XSS 対策を一元化する。実装計画
//! §2.1・イシュー #870）。
//!
//! # 安全性契約（イシュー #870 の設計判断）
//!
//! - [`Node`] は `Element` と `Text` の 2 バリアントのみで構成し、**生 HTML を
//!   注入できるバリアント（`RawHtml` 相当）を持たない**。これにより
//!   「エスケープをスキップする経路」自体が型として存在しない
//!   （参照実装 fandhe-backend の `fandhe_frontend_core::Node` より安全側に単純化。
//!   実装計画 §2.1）
//! - テキスト・属性値は [`render`] 系関数の内部でのみ HTML エスケープする。
//!   呼び出し側（`markdown`・`layout`）はエスケープ済みでない生文字列を
//!   [`Node::Text`] / 属性値としてそのまま渡してよい

/// HTML ノード木。`Element` は開始・終了タグと属性・子ノードを持ち、`Text` は
/// レンダリング時に必ずエスケープされる素のテキストを保持する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// 要素ノード（`tag` は小文字の HTML タグ名を想定）。
    Element {
        tag: String,
        attrs: Vec<(String, String)>,
        children: Vec<Node>,
    },
    /// テキストノード。`render` 時に `&` `<` `>` をエスケープする。
    Text(String),
}

impl Node {
    /// 要素ノードを組み立てる便利コンストラクタ。
    pub fn element(
        tag: impl Into<String>,
        attrs: Vec<(String, String)>,
        children: Vec<Node>,
    ) -> Node {
        Node::Element {
            tag: tag.into(),
            attrs,
            children,
        }
    }

    /// テキストノードを組み立てる便利コンストラクタ。
    pub fn text(value: impl Into<String>) -> Node {
        Node::Text(value.into())
    }
}

/// void 要素（終了タグを持たない要素）のホワイトリスト。この一覧に一致するタグは
/// 常に自己終端（`<tag ... />`）でレンダリングし、子ノードがあっても出力しない
/// （呼び出し側が void 要素へ子ノードを渡すのは呼び出し側のバグであり、ここで
/// 黙って握りつぶす。子を渡さない契約は呼び出し側〈layout.rs〉が守る）。
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

/// テキストコンテンツの HTML エスケープ（`&` `<` `>` の 3 種類）。
fn escape_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
    }
    out
}

/// 属性値の HTML エスケープ（`&` `<` `>` `"` `'` の 5 種類。属性値はダブルクォートで
/// 囲むため `"` を、`'` も念のため無害化する）。
fn escape_attr(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// 単一ノードを HTML 文字列へレンダリングする。
pub fn render(node: &Node) -> String {
    let mut out = String::new();
    render_into(node, &mut out);
    out
}

/// ノード列を連結して HTML 文字列へレンダリングする（`<article>` 本文等、
/// 兄弟ノード列をまとめて扱う呼び出し元向け）。
pub fn render_all(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        render_into(node, &mut out);
    }
    out
}

fn render_into(node: &Node, out: &mut String) {
    match node {
        Node::Text(text) => out.push_str(&escape_text(text)),
        Node::Element {
            tag,
            attrs,
            children,
        } => {
            out.push('<');
            out.push_str(tag);
            for (key, value) in attrs {
                out.push(' ');
                out.push_str(key);
                out.push_str("=\"");
                out.push_str(&escape_attr(value));
                out.push('"');
            }
            if VOID_ELEMENTS.contains(&tag.as_str()) {
                out.push_str(" />");
                return;
            }
            out.push('>');
            for child in children {
                render_into(child, out);
            }
            out.push_str("</");
            out.push_str(tag);
            out.push('>');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_text_content() {
        let node = Node::text("<script>alert('x')&\"y\"</script>");
        assert_eq!(
            render(&node),
            "&lt;script&gt;alert('x')&amp;\"y\"&lt;/script&gt;"
        );
    }

    #[test]
    fn escapes_attribute_values() {
        let node = Node::element(
            "a",
            vec![("href".to_string(), "\"><script>x</script>".to_string())],
            vec![Node::text("link")],
        );
        assert_eq!(
            render(&node),
            "<a href=\"&quot;&gt;&lt;script&gt;x&lt;/script&gt;\">link</a>"
        );
    }

    #[test]
    fn renders_nested_elements() {
        let node = Node::element(
            "ul",
            vec![],
            vec![
                Node::element("li", vec![], vec![Node::text("a")]),
                Node::element("li", vec![], vec![Node::text("b")]),
            ],
        );
        assert_eq!(render(&node), "<ul><li>a</li><li>b</li></ul>");
    }

    #[test]
    fn renders_void_elements_self_closing_without_children() {
        let node = Node::element(
            "link",
            vec![
                ("rel".to_string(), "stylesheet".to_string()),
                ("href".to_string(), "/assets/site.css".to_string()),
            ],
            vec![],
        );
        assert_eq!(
            render(&node),
            "<link rel=\"stylesheet\" href=\"/assets/site.css\" />"
        );
    }

    #[test]
    fn render_all_concatenates_sibling_nodes() {
        let nodes = vec![
            Node::element("h1", vec![], vec![Node::text("Title")]),
            Node::element("p", vec![], vec![Node::text("Body")]),
        ];
        assert_eq!(render_all(&nodes), "<h1>Title</h1><p>Body</p>");
    }
}
