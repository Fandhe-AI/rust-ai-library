//! ビルド時に埋め込むテーマ CSS 定数。
//!
//! # 呼び出し文脈
//!
//! `build.rs` が [`SITE_CSS`] を `<out>/assets/site.css` へそのまま書き出す。
//! CSS 実体は `crates/docs-site/assets/site.css`（リポジトリルートの `site/`
//! ツリーとは分離した本クレート専用アセット。実装計画 §2.5）に置き、
//! `include_str!` でバイナリへ埋め込む（外部依存追加なし。deps-policy.md）。
//!
//! ライト/ダークテーマの切替 JS（`localStorage` 永続化・トグルボタン）は
//! 兄弟イシュー #871 のスコープであり、本モジュールは CSS 側のフック
//! （`prefers-color-scheme` メディアクエリ + `[data-theme]` 属性セレクタ）のみを
//! 用意する（実装計画 §2.5）。

/// `<out>/assets/site.css` へ書き出すテーマ CSS 全文。
pub const SITE_CSS: &str = include_str!("../assets/site.css");
