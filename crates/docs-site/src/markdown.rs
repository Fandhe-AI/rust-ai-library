//! 自作 Markdown → [`crate::html::Node`] 変換。外部 Markdown クレートは使わない
//! （deps-policy.md の許容 9 区分外。イシュー #870 実装計画 §2.2）。
//!
//! # 呼び出し文脈
//!
//! `build.rs` が各 `page.source`（`site/*.md`）の内容を読み込み、
//! [`markdown_to_nodes`] へ渡して `<article>` 本文の子ノード列を得る。得られた
//! [`crate::html::Node`] 列は `layout::docs_page` がページ骨格へ組み込み、
//! 最終的な文字列化は `html::render` 系関数のみが担う（本モジュールは
//! `Node::Text` にエスケープ前の生文字列を渡すだけでよい。`html.rs` モジュール
//! コメント参照）。
//!
//! # 対応する記法（実装計画 §2.2）
//!
//! - ブロック: ATX 見出し（h1〜h6）・段落・箇条書き/番号リスト（ネスト対応）・
//!   フェンスコードブロック・引用（blockquote）・GFM テーブル
//! - インライン: インラインコード・強調（`*em*`/`**strong**`/`***em+strong***`）・
//!   リンク `[text](url)`
//!
//! # 安全性・DoS 対策（レビュー指摘反映済みの参照実装 fandhe-backend の設計を踏襲）
//!
//! - [`MAX_DEPTH`][]: 引用・リスト・インライン再帰のスタックオーバーフロー対策。
//!   超過時は残り行をエスケープ済み段落テキストへフォールバックする（パニックしない）
//! - [`MAX_INLINE_SCAN_WINDOW`][]: 閉じマーカー（インラインコード・強調）探索の
//!   計算量 O(n²) 化を防ぐ走査幅上限。**文字数**（バイト数ではない）で数える。
//!   本リポジトリの実 `site/` 原稿は日本語が主体で `&input[a..b]` のようなバイト
//!   スライスは非文字境界でパニックしうるため、インライン走査は必ず `Vec<char>`
//!   上のインデックスで行い、生文字列への `&str[..]` バイトスライスを行わない
//! - リンク URL は allow-list（http / https / 相対のみ）。正規化（`\t\n\r` 除去 +
//!   先頭制御文字トリム）した後にスキームを判定するため `java\tscript:` のような
//!   偽装を素通りさせない。不合格はリンク化せずテキストのみ出力する（fail-closed）
//! - Markdown 中の生 HTML タグは構文として解釈しない。通常テキストとして
//!   [`crate::html::Node::Text`] に積まれ、`html::render` が機械的にエスケープする
//! - 未知・非対応の構文は段落フォールバックとし、本モジュールはパニックしない
//!   全域関数（`unwrap()` 不使用）として実装する

use crate::html::Node;

/// 引用・リスト・インライン再帰の深さ上限。超過時はそれ以上再帰せず、残りを
/// エスケープ済みテキストとして扱う（スタックオーバーフロー対策）。
const MAX_DEPTH: usize = 16;

/// インライン走査（インラインコード・強調の閉じマーカー探索）の走査幅上限
/// （文字数）。閉じマーカーが見つからないまま入力末尾まで探索し続ける
/// O(n²) の計算量 DoS を防ぐ。
const MAX_INLINE_SCAN_WINDOW: usize = 2000;

/// Markdown 原稿全文を [`crate::html::Node`] 列（`<article>` の子として並べる想定）
/// へ変換する。パニックしない全域関数。
pub fn markdown_to_nodes(input: &str) -> Vec<Node> {
    let lines: Vec<&str> = input.lines().collect();
    parse_blocks(&lines, 0)
}

// ---------------------------------------------------------------------
// ブロックパーサー
// ---------------------------------------------------------------------

fn parse_blocks(lines: &[&str], depth: usize) -> Vec<Node> {
    if depth > MAX_DEPTH {
        // 深さ上限超過: それ以上構造化せず、残り行をまとめて 1 段落のテキストに
        // フォールバックする（未対応構文と同じ安全側の扱い。パニックしない）。
        let joined = lines.join(" ");
        if joined.trim().is_empty() {
            return Vec::new();
        }
        return vec![Node::element("p", vec![], vec![Node::text(joined)])];
    }

    let mut nodes = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        if let Some((level, text)) = atx_heading(line) {
            let tag = format!("h{level}");
            nodes.push(Node::element(tag, vec![], parse_inline(text, depth + 1)));
            i += 1;
            continue;
        }

        if let Some(lang) = fence_open(line) {
            let mut code_lines: Vec<&str> = Vec::new();
            i += 1;
            while i < lines.len() && !is_fence_close(lines[i]) {
                code_lines.push(lines[i]);
                i += 1;
            }
            if i < lines.len() {
                i += 1; // 閉じフェンス行を消費する
            }
            let code_text = code_lines.join("\n");
            let code_attrs = match lang {
                Some(lang) => vec![("class".to_string(), format!("language-{lang}"))],
                None => vec![],
            };
            let code_node = Node::element("code", code_attrs, vec![Node::text(code_text)]);
            nodes.push(Node::element("pre", vec![], vec![code_node]));
            continue;
        }

        if line.trim_start().starts_with('>') {
            let mut quote_lines: Vec<&str> = Vec::new();
            while i < lines.len() && lines[i].trim_start().starts_with('>') {
                let stripped = lines[i].trim_start();
                let stripped = stripped.strip_prefix('>').unwrap_or(stripped);
                let stripped = stripped.strip_prefix(' ').unwrap_or(stripped);
                quote_lines.push(stripped);
                i += 1;
            }
            let inner = parse_blocks(&quote_lines, depth + 1);
            nodes.push(Node::element("blockquote", vec![], inner));
            continue;
        }

        if list_item_prefix(line).is_some() {
            let (list_node, consumed) = parse_list(&lines[i..], depth);
            nodes.push(list_node);
            i += consumed.max(1); // 消費 0 件による無限ループを防ぐ安全弁
            continue;
        }

        if line.contains('|') && i + 1 < lines.len() && is_table_separator(lines[i + 1]) {
            let (table_node, consumed) = parse_table(&lines[i..]);
            nodes.push(table_node);
            i += consumed.max(1);
            continue;
        }

        // 段落: 他ブロック構文の開始行に遭遇するか空行に達するまで連続行を集約する
        // （Markdown のソフト改行は半角スペースへ畳み込む）。
        let mut para_lines: Vec<&str> = Vec::new();
        while i < lines.len() {
            let l = lines[i];
            if l.trim().is_empty() {
                break;
            }
            if atx_heading(l).is_some()
                || fence_open(l).is_some()
                || l.trim_start().starts_with('>')
                || list_item_prefix(l).is_some()
            {
                break;
            }
            // GFM テーブル開始行（次行がセパレータ行）でも段落集約を打ち切る。
            // ここを見落とすと、空行を挟まず段落直後にテーブルが続く原稿
            // （著者が意図的に書く自然な構成）でテーブル全体がリテラルな
            // パイプ文字列として段落へ飲み込まれてしまう
            // （レビュー指摘・イシュー #870。回帰テスト参照）。
            if l.contains('|') && i + 1 < lines.len() && is_table_separator(lines[i + 1]) {
                break;
            }
            para_lines.push(l.trim());
            i += 1;
        }
        let text = para_lines.join(" ");
        nodes.push(Node::element("p", vec![], parse_inline(&text, depth + 1)));
    }
    nodes
}

/// 行頭の半角スペース数（インデント幅）を数える。
fn leading_spaces(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

/// ATX 見出し（`#`〜`######` + 半角スペース + テキスト）を判定する。
/// `##テキスト`（スペースなし）は見出しとして扱わない（CommonMark 準拠）。
fn atx_heading(line: &str) -> Option<(u8, &str)> {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &trimmed[hashes..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    Some((hashes as u8, rest.trim()))
}

/// フェンスコードブロックの開始行かを判定する。開始行が言語トークンを持つ場合は
/// `Some(Some(lang))`、持たない場合は `Some(None)` を返す。言語トークンは
/// 英数字・`_`・`+`・`.`・`-` のホワイトリストのみを `class="language-*"` として
/// 採用し、それ以外の文字を含む場合はトークンを採用しない（クラス名インジェクション
/// 対策・実装計画 §2.2）。
fn fence_open(line: &str) -> Option<Option<String>> {
    let trimmed = line.trim();
    let after = trimmed.strip_prefix("```")?;
    if after.is_empty() {
        return Some(None);
    }
    let is_safe_lang_token = after
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '.' | '-'));
    if is_safe_lang_token {
        Some(Some(after.to_string()))
    } else {
        Some(None)
    }
}

fn is_fence_close(line: &str) -> bool {
    line.trim() == "```"
}

/// 行が箇条書き/番号リストの項目マーカーで始まるかを判定し、
/// `(ordered, インデント後のマーカー込みプレフィックス長)` を返す。
fn list_item_prefix(line: &str) -> Option<(bool, usize)> {
    let indent = leading_spaces(line);
    let rest = line.get(indent..)?;
    if rest.starts_with("- ") || rest.starts_with("* ") {
        return Some((false, 2));
    }
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let after_digits = rest.get(digits.len()..)?;
    if after_digits.starts_with(". ") {
        return Some((true, digits.len() + 2));
    }
    None
}

/// `lines[0]` から始まる同一インデント・同一種別（順序/非順序）の連続リスト項目を
/// 消費し、`(<ul>/<ol> ノード, 消費行数)` を返す。ネストしたリストは項目内容を
/// 再帰的に [`parse_blocks`] へ渡すことで対応する（インデントで判定）。
fn parse_list(lines: &[&str], depth: usize) -> (Node, usize) {
    let Some((ordered0, _)) = list_item_prefix(lines[0]) else {
        // 呼び出し元は list_item_prefix が Some であることを確認済みだが、
        // 全域関数として不変条件が崩れても panic しない安全側のフォールバック。
        return (Node::element("p", vec![], vec![Node::text(lines[0])]), 1);
    };
    let base_indent = leading_spaces(lines[0]);
    let tag = if ordered0 { "ol" } else { "ul" };
    let mut items = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            break;
        }
        let indent = leading_spaces(line);
        if indent != base_indent {
            break;
        }
        let Some((ordered, prefix_len)) = list_item_prefix(line) else {
            break;
        };
        if ordered != ordered0 {
            break; // 種別が変われば別リストとして扱い、ここでは打ち切る
        }

        let first_rest = line.get(base_indent + prefix_len..).unwrap_or("");
        let mut item_lines: Vec<String> = vec![first_rest.to_string()];
        i += 1;
        // ネスト行（インデントが深い継続行）を同一項目として吸収する。
        while i < lines.len() {
            let l = lines[i];
            if l.trim().is_empty() {
                break;
            }
            let l_indent = leading_spaces(l);
            if l_indent <= base_indent {
                break;
            }
            let strip = (base_indent + prefix_len).min(l_indent);
            item_lines.push(l.get(strip..).unwrap_or("").to_string());
            i += 1;
        }

        let item_line_refs: Vec<&str> = item_lines.iter().map(String::as_str).collect();
        let inner = parse_blocks(&item_line_refs, depth + 1);
        // 項目本文の先頭段落は <li><p>...</p></li> ではなく <li>...</li> へ平坦化
        // する（一般的な Markdown レンダラの慣習に合わせる）。ネストしたリスト等
        // 後続ブロックがある場合はそのまま兄弟ノードとして残す。
        let li_children = match inner.first() {
            Some(Node::Element { tag, children, .. }) if tag == "p" => {
                let mut flat = children.clone();
                flat.extend(inner.iter().skip(1).cloned());
                flat
            }
            _ => inner,
        };
        items.push(Node::element("li", vec![], li_children));
    }

    (Node::element(tag, vec![], items), i)
}

/// GFM 風テーブル区切り行（`---`・`:---:` 等をパイプ区切りしたもの）かを判定する。
/// 各セルは `-` を最低 1 つ含み、`-`・`:` のみで構成されることを要求する。
fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let inner = trimmed.trim_matches('|');
    if inner.is_empty() {
        return false;
    }
    inner.split('|').all(|cell| {
        let cell = cell.trim();
        !cell.is_empty() && cell.contains('-') && cell.chars().all(|c| c == '-' || c == ':')
    })
}

/// `line` を `|` 区切りテーブル行としてセル列へ分割する（先頭・末尾の `|` は除去）。
fn split_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = trimmed.trim_matches('|');
    inner.split('|').map(|c| c.trim().to_string()).collect()
}

/// `lines[0]` をヘッダ行、`lines[1]` を区切り行とするテーブルを消費し、
/// `(<table> ノード, 消費行数)` を返す。データ行は `|` を含む間続けて読む。
fn parse_table(lines: &[&str]) -> (Node, usize) {
    let header_cells = split_table_row(lines[0]);
    let mut i = 2; // ヘッダ行 + 区切り行

    let mut body_rows: Vec<Vec<String>> = Vec::new();
    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() || !line.contains('|') {
            break;
        }
        body_rows.push(split_table_row(line));
        i += 1;
    }

    let thead_row = Node::element(
        "tr",
        vec![],
        header_cells
            .iter()
            .map(|cell| Node::element("th", vec![], parse_inline(cell, 1)))
            .collect(),
    );
    let thead = Node::element("thead", vec![], vec![thead_row]);

    let tbody_rows: Vec<Node> = body_rows
        .iter()
        .map(|row| {
            Node::element(
                "tr",
                vec![],
                row.iter()
                    .map(|cell| Node::element("td", vec![], parse_inline(cell, 1)))
                    .collect(),
            )
        })
        .collect();
    let tbody = Node::element("tbody", vec![], tbody_rows);

    (Node::element("table", vec![], vec![thead, tbody]), i)
}

// ---------------------------------------------------------------------
// インラインパーサー
// ---------------------------------------------------------------------

/// インラインテキストを [`crate::html::Node`] 列へ変換する。走査は必ず `Vec<char>`
/// （文字インデックス）で行い、バイトインデックスの `&str[..]` スライスは行わない
/// （日本語主体の原稿でも非文字境界パニックを起こさないため。モジュール冒頭
/// コメント参照）。
fn parse_inline(text: &str, depth: usize) -> Vec<Node> {
    if depth > MAX_DEPTH {
        return vec![Node::text(text)];
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        match c {
            '`' => {
                if let Some(end) = find_plain_close(&chars, i + 1, '`') {
                    flush_text(&mut buf, &mut out);
                    let code_text: String = chars[i + 1..end].iter().collect();
                    out.push(Node::element("code", vec![], vec![Node::text(code_text)]));
                    i = end + 1;
                } else {
                    buf.push(c);
                    i += 1;
                }
            }
            '*' => {
                if let Some((node, next_i)) = try_emphasis(&chars, i, depth) {
                    flush_text(&mut buf, &mut out);
                    out.push(node);
                    i = next_i;
                } else {
                    buf.push(c);
                    i += 1;
                }
            }
            '[' => {
                if let Some((node, next_i)) = try_link(&chars, i, depth) {
                    flush_text(&mut buf, &mut out);
                    out.push(node);
                    i = next_i;
                } else {
                    buf.push(c);
                    i += 1;
                }
            }
            other => {
                buf.push(other);
                i += 1;
            }
        }
    }
    flush_text(&mut buf, &mut out);
    out
}

fn flush_text(buf: &mut String, out: &mut Vec<Node>) {
    if !buf.is_empty() {
        out.push(Node::text(std::mem::take(buf)));
    }
}

/// `chars[from..]` の中から `target` 文字が最初に現れる位置を、[`MAX_INLINE_SCAN_WINDOW`]
/// 文字分だけ探索する（コードスパンの閉じバッククォート探索用。強調のように
/// コードスパンをスキップする必要はない）。
fn find_plain_close(chars: &[char], from: usize, target: char) -> Option<usize> {
    let limit = (from + MAX_INLINE_SCAN_WINDOW).min(chars.len());
    (from..limit).find(|&idx| chars[idx] == target)
}

/// リンク URL 部分 `(...)` の閉じ括弧位置を、URL 内に現れうるネストした `(` `)`
/// （例: `[x](javascript:alert(1))`）の深さを数えながら探索する。単純な
/// 最初の `)` 一致（[`find_plain_close`]）だと URL 内の `)` で早期に打ち切られ、
/// 外側の閉じ括弧が地の文として漏れ出す（レビューで判明した実装バグ）。
fn find_url_close_paren(chars: &[char], from: usize) -> Option<usize> {
    let limit = (from + MAX_INLINE_SCAN_WINDOW).min(chars.len());
    let mut depth: u32 = 0;
    for (idx, c) in chars.iter().enumerate().take(limit).skip(from) {
        match c {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return Some(idx);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// `chars[from..]` の中からインラインコードスパン（`` `...` ``）をスキップしつつ
/// `run`（例: `"**"`）と完全一致する区間の開始位置を探索する。強調マーカーの
/// 誤検出（コードスパン内の `*` を強調境界と誤認する）を防ぐ（実装計画 §2.2
/// 「skip_backtick_span」）。
fn find_marker_close(chars: &[char], from: usize, run: &[char]) -> Option<usize> {
    let limit = (from + MAX_INLINE_SCAN_WINDOW).min(chars.len());
    let mut idx = from;
    while idx < limit {
        if chars[idx] == '`'
            && let Some(end) = find_plain_close(chars, idx + 1, '`')
        {
            idx = end + 1;
            continue;
        }
        if chars[idx..].starts_with(run) {
            return Some(idx);
        }
        idx += 1;
    }
    None
}

/// `*` の連続run（1〜3 文字: em / strong / em+strong）を検出し、対応する
/// 強調ノードを組み立てる。閉じマーカーは長い run から順に試す
/// （`***text***` を `*<em>*text*</em>*` のように誤分割しない）。
fn try_emphasis(chars: &[char], i: usize, depth: usize) -> Option<(Node, usize)> {
    let run_len = {
        let mut n = 0;
        while i + n < chars.len() && chars[i + n] == '*' && n < 3 {
            n += 1;
        }
        n
    };
    if run_len == 0 {
        return None;
    }
    // 開き `*` の直後が空白/行末だと強調として成立しない（左フランキング規則の簡易版）。
    if chars.get(i + run_len).is_none_or(|c| c.is_whitespace()) {
        return None;
    }

    for len in (1..=run_len).rev() {
        let marker: Vec<char> = vec!['*'; len];
        let content_start = i + len;
        if let Some(close) = find_marker_close(chars, content_start, &marker) {
            if close == content_start {
                continue; // 空の強調（`**`）は強調として扱わない
            }
            let inner_text: String = chars[content_start..close].iter().collect();
            let inner_nodes = parse_inline(&inner_text, depth + 1);
            let node = match len {
                3 => Node::element(
                    "strong",
                    vec![],
                    vec![Node::element("em", vec![], inner_nodes)],
                ),
                2 => Node::element("strong", vec![], inner_nodes),
                _ => Node::element("em", vec![], inner_nodes),
            };
            return Some((node, close + len));
        }
    }
    None
}

/// `[text](url)` 形式のリンクを検出する。`url` が [`is_allowed_url`] を満たさない
/// 場合はリンク化せず、`text` 部分のみをプレーンテキストとして返す（fail-closed。
/// リンクとして成立しない `[`（閉じ `]` や `(url)` が見つからない）場合は `None`
/// を返して呼び出し元に通常文字として扱わせる）。
fn try_link(chars: &[char], i: usize, depth: usize) -> Option<(Node, usize)> {
    let close_bracket = find_plain_close(chars, i + 1, ']')?;
    if chars.get(close_bracket + 1) != Some(&'(') {
        return None;
    }
    let url_start = close_bracket + 2;
    let close_paren = find_url_close_paren(chars, url_start)?;

    let text: String = chars[i + 1..close_bracket].iter().collect();
    let url: String = chars[url_start..close_paren].iter().collect();
    let next_i = close_paren + 1;

    let inner_nodes = parse_inline(&text, depth + 1);
    if is_allowed_url(&url) {
        Some((
            Node::element("a", vec![("href".to_string(), url)], inner_nodes),
            next_i,
        ))
    } else {
        // 不合格 URL はリンク化せずテキストのみ出力する（fail-closed。実装計画 §2.2）。
        Some((Node::element("span", vec![], inner_nodes), next_i))
    }
}

/// URL の `\t` `\n` `\r` を除去し、先頭の制御文字・半角スペースをトリムして
/// 正規化する。`java\tscript:` のようなタブ挿入によるスキーム偽装を無害化して
/// から [`is_allowed_url`] のスキーム判定にかけるための前処理。半角スペース
/// （U+0020）は `char::is_control` が Cc カテゴリしか見ないため対象外になり、
/// ` //evil.com` のような先頭スペース付きプロトコル相対 URL が `//` 始まり
/// 判定を回避して `is_allowed_url` の相対パス許可分岐へ落ちてしまう
/// （Review 指摘・issue #870: WHATWG URL 仕様のトリム対象と同様に空白も除く）。
fn normalize_url(raw: &str) -> String {
    let filtered: String = raw
        .chars()
        .filter(|c| !matches!(c, '\t' | '\n' | '\r'))
        .collect();
    filtered
        .trim_start_matches(|c: char| c.is_control() || c == ' ')
        .to_string()
}

/// リンク URL の allow-list 判定。http / https（大小文字不問）・`/` 始まりの
/// 絶対相対パス・スキームを持たない相対パスのみを許可する。`javascript:`・
/// `data:`・`mailto:` 等のスキームは拒否する（実装計画 §2.2「URL は allow-list」）。
fn is_allowed_url(raw: &str) -> bool {
    let normalized = normalize_url(raw);
    if normalized.is_empty() {
        return false;
    }
    // `\` を含む URL は http(s) 絶対 URL・サイト内相対パスのいずれにも正当な
    // 用途がなく、WHATWG URL 仕様ではブラウザが special scheme の相対解決で
    // `\` を `/` と等価に扱うため `\\/evil.com` 等が `//evil.com` のプロトコル
    // 相対 URL 偽装として機能しうる。`/` 始まり判定より先に無条件で拒否する
    // （`nav.rs::validate_source_shape` が `source` の `\` を拒否するのと同方針。
    // Review 指摘・issue #870）。
    if normalized.contains('\\') {
        return false;
    }
    let lower = normalized.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return true;
    }
    // `//` 始まりはプロトコル相対 URL（ページの現在スキームを継承した外部リンク）
    // であり、ドキュメンテーションコメントが定める許可対象（http/https・`/` 始まりの
    // 絶対相対パス・スキームなし相対パス）のいずれにも該当しないため、`/` 始まり
    // 判定より先に明示的に拒否する（Review 指摘: crates/docs-site issue #870）。
    if normalized.starts_with("//") {
        return false;
    }
    if normalized.starts_with('/') {
        return true;
    }
    // スキームの有無は「最初の `/` より前に `:` があるか」で判定する。
    // スキームがなければ相対パスとして許可する。
    let colon_pos = normalized.find(':');
    let slash_pos = normalized.find('/');
    match (colon_pos, slash_pos) {
        (Some(c), Some(s)) if c < s => false,
        (Some(_), None) => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::html::render_all;

    fn html_of(markdown: &str) -> String {
        render_all(&markdown_to_nodes(markdown))
    }

    // ---- 見出し ----

    #[test]
    fn renders_headings_h1_to_h6() {
        for level in 1..=6u8 {
            let hashes = "#".repeat(level as usize);
            let out = html_of(&format!("{hashes} Title"));
            assert_eq!(out, format!("<h{level}>Title</h{level}>"));
        }
    }

    #[test]
    fn rejects_hash_without_space_as_heading() {
        // CommonMark: `#タイトル`（スペースなし）は見出しにならず段落になる。
        assert_eq!(html_of("#タイトル"), "<p>#タイトル</p>");
    }

    // ---- 段落 ----

    #[test]
    fn joins_soft_line_breaks_into_single_paragraph() {
        assert_eq!(html_of("Line1\nLine2"), "<p>Line1 Line2</p>");
    }

    #[test]
    fn separates_paragraphs_by_blank_line() {
        assert_eq!(html_of("Para1\n\nPara2"), "<p>Para1</p><p>Para2</p>");
    }

    // ---- リスト ----

    #[test]
    fn renders_unordered_list() {
        assert_eq!(html_of("- a\n- b"), "<ul><li>a</li><li>b</li></ul>");
    }

    #[test]
    fn renders_ordered_list() {
        assert_eq!(html_of("1. a\n2. b"), "<ol><li>a</li><li>b</li></ol>");
    }

    #[test]
    fn renders_nested_unordered_list() {
        let out = html_of("- a\n  - nested\n- b");
        assert_eq!(out, "<ul><li>a<ul><li>nested</li></ul></li><li>b</li></ul>");
    }

    // ---- フェンスコードブロック ----

    #[test]
    fn renders_fenced_code_block_with_language_class() {
        let out = html_of("```rust\nfn main() {}\n```");
        assert_eq!(
            out,
            "<pre><code class=\"language-rust\">fn main() {}</code></pre>"
        );
    }

    #[test]
    fn escapes_script_tag_content_inside_fenced_code_block() {
        let out = html_of("```html\n<script>alert(1)</script>\n```");
        assert_eq!(
            out,
            "<pre><code class=\"language-html\">&lt;script&gt;alert(1)&lt;/script&gt;</code></pre>"
        );
    }

    #[test]
    fn renders_fenced_code_block_without_closing_fence_at_eof() {
        let out = html_of("```rust\nfn main() {}");
        assert_eq!(
            out,
            "<pre><code class=\"language-rust\">fn main() {}</code></pre>"
        );
    }

    #[test]
    fn omits_class_for_unsafe_language_token() {
        let out = html_of("```rust\";alert(1)//\nx\n```");
        assert_eq!(out, "<pre><code>x</code></pre>");
    }

    // ---- 引用 ----

    #[test]
    fn renders_blockquote() {
        assert_eq!(
            html_of("> quoted line"),
            "<blockquote><p>quoted line</p></blockquote>"
        );
    }

    #[test]
    fn renders_nested_blockquote() {
        let out = html_of("> outer\n> > inner");
        assert_eq!(
            out,
            "<blockquote><p>outer</p><blockquote><p>inner</p></blockquote></blockquote>"
        );
    }

    // ---- テーブル ----

    #[test]
    fn renders_gfm_table() {
        let out = html_of("| A | B |\n|---|---|\n| 1 | 2 |");
        assert_eq!(
            out,
            "<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
        );
    }

    #[test]
    fn renders_table_immediately_after_paragraph_without_blank_line() {
        // レビュー指摘（イシュー #870）の回帰テスト: 段落集約ループが
        // GFM テーブル開始行（次行がセパレータ行）で打ち切られない場合、
        // 空行を挟まず段落直後にテーブルが続く原稿でテーブル全体が
        // リテラルなパイプ文字列として段落に飲み込まれてしまう。
        let out = html_of("text\n| A | B |\n|---|---|\n| 1 | 2 |");
        assert_eq!(
            out,
            "<p>text</p><table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
        );
    }

    #[test]
    fn does_not_misdetect_non_separator_second_line_as_table() {
        // 2 行目が区切り行の形式（`-`/`:` のみ）を満たさないため通常の段落になる。
        let out = html_of("| A | B |\nnot a separator");
        assert_eq!(out, "<p>| A | B | not a separator</p>");
    }

    // ---- インライン ----

    #[test]
    fn renders_inline_code() {
        assert_eq!(
            html_of("use `cargo test`"),
            "<p>use <code>cargo test</code></p>"
        );
    }

    #[test]
    fn renders_emphasis_variants() {
        assert_eq!(html_of("*em*"), "<p><em>em</em></p>");
        assert_eq!(html_of("**strong**"), "<p><strong>strong</strong></p>");
        assert_eq!(
            html_of("***both***"),
            "<p><strong><em>both</em></strong></p>"
        );
    }

    #[test]
    fn renders_link_with_allowed_relative_url() {
        assert_eq!(
            html_of("[intro](/guide/intro/)"),
            "<p><a href=\"/guide/intro/\">intro</a></p>"
        );
    }

    #[test]
    fn renders_link_with_allowed_https_url() {
        assert_eq!(
            html_of("[repo](https://github.com/Fandhe-AI/rust-ai-library)"),
            "<p><a href=\"https://github.com/Fandhe-AI/rust-ai-library\">repo</a></p>"
        );
    }

    #[test]
    fn drops_link_for_disallowed_javascript_scheme() {
        assert_eq!(html_of("[x](javascript:alert(1))"), "<p><span>x</span></p>");
    }

    #[test]
    fn drops_link_for_tab_disguised_javascript_scheme() {
        assert_eq!(
            html_of("[x](java\tscript:alert(1))"),
            "<p><span>x</span></p>"
        );
    }

    #[test]
    fn drops_link_for_disallowed_data_scheme() {
        assert_eq!(html_of("[x](data:text/html,evil)"), "<p><span>x</span></p>");
    }

    #[test]
    fn drops_link_for_disallowed_mailto_scheme() {
        assert_eq!(html_of("[x](mailto:a@b.com)"), "<p><span>x</span></p>");
    }

    #[test]
    fn drops_link_for_protocol_relative_url() {
        // `//evil.com/path` はページの現在スキームを継承した外部サイトへの
        // リンクになる（プロトコル相対 URL）。`/` 始まり判定に飲み込まれて
        // 誤って許可されないことを回帰検査する（Review 指摘: issue #870）。
        assert_eq!(html_of("[x](//evil.com/path)"), "<p><span>x</span></p>");
    }

    #[test]
    fn drops_link_for_backslash_disguised_protocol_relative_url() {
        // WHATWG URL 仕様ではブラウザが special scheme の相対解決で `\` を `/`
        // と等価に扱うため、`\` 混じりの `//evil.com` 変種も外部誘導になりうる
        // （Review 指摘・issue #870）。`\` を含む URL を無条件拒否することで
        // バリエーションをまとめて遮断できることを回帰検査する。
        for url in ["\\/evil.com/x", "/\\evil.com/x", "\\\\evil.com/x"] {
            assert_eq!(
                html_of(&format!("[x]({url})")),
                "<p><span>x</span></p>",
                "url={url} が拒否されなかった"
            );
        }
    }

    #[test]
    fn drops_link_for_space_disguised_protocol_relative_url() {
        // 半角スペース（U+0020）は `char::is_control`（Cc カテゴリ）の対象外の
        // ため、先頭スペース付き ` //evil.com` が `normalize_url` のトリムを
        // すり抜けて `//` 始まり判定を回避していた（Review 指摘・issue #870）。
        assert_eq!(html_of("[x]( //evil.com/path)"), "<p><span>x</span></p>");
    }

    #[test]
    fn drops_link_for_space_and_backslash_disguised_protocol_relative_url() {
        // 先頭スペース・`\` 偽装の組み合わせも同時に閉じることを回帰検査する
        // （Review 指摘・issue #870）。
        assert_eq!(html_of("[x]( \\/evil.com/path)"), "<p><span>x</span></p>");
    }

    // ---- 生 HTML はテキストとしてエスケープされる ----

    #[test]
    fn escapes_raw_html_in_paragraph_text() {
        assert_eq!(
            html_of("<b>not bold</b>"),
            "<p>&lt;b&gt;not bold&lt;/b&gt;</p>"
        );
    }

    // ---- 日本語（マルチバイト文字境界の安全性回帰） ----

    #[test]
    fn handles_japanese_text_with_emphasis_and_inline_code_without_panicking() {
        let input = "日本語の説明文で `compat::Sequential` を使い、*重要* な点を **強調** する。";
        let out = html_of(input);
        assert!(out.contains("<code>compat::Sequential</code>"));
        assert!(out.contains("<em>重要</em>"));
        assert!(out.contains("<strong>強調</strong>"));
    }

    #[test]
    fn handles_long_japanese_paragraph_spanning_inline_scan_window_without_panicking() {
        // MAX_INLINE_SCAN_WINDOW（2000 文字）境界をまたぐ日本語段落でも
        // 非文字境界パニックを起こさないことを確認する回帰テスト。
        let filler: String = "あ".repeat(2100);
        let input = format!("{filler}`code`{filler}");
        let out = html_of(&input);
        assert!(out.starts_with("<p>"));
        assert!(out.ends_with("</p>"));
    }

    #[test]
    fn depth_limit_falls_back_to_paragraph_without_panicking() {
        let nested = "> ".repeat(MAX_DEPTH + 5) + "deeply nested";
        // パニックしないことのみを確認する（フォールバック内容の厳密一致は問わない）。
        let _ = html_of(&nested);
    }
}
