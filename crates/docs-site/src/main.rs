//! `docs-site` CLI: `site/nav.toml` を検証し、出力ディレクトリを作成する薄いラッパー。
//!
//! # 呼び出し例
//!
//! ```text
//! cargo run -p docs-site -- --root . --out target/site-dist
//! ```
//!
//! `--out <dir>` は必須（欠落・値欠落は fail-closed で非 0 終了する。既定値への
//! 黙示フォールバックはしない）。`--root <dir>` は任意（既定 `.`）で、E2E テストが
//! `tests/fixtures/` 配下のフィクスチャルートを渡すために存在する（本イシューでは
//! リポジトリルートに `site/nav.toml` を作らない。実装計画 §1・#873 のスコープ）。
//!
//! 引数パースは外部クレートを使わず手実装する（deps-policy.md の許容 9 区分に
//! CLI 引数パーサーは含まれず、本イシューはユーザー承認済みの新規依存追加を行わない
//! 方針のため）。ビルド本体は [`docs_site::build::build_site`] に委譲する。

use std::path::PathBuf;
use std::process::ExitCode;

use docs_site::build::build_site;

/// パース済み CLI 引数。
struct Args {
    root: PathBuf,
    out: PathBuf,
}

/// CLI 引数パースの失敗理由。usage メッセージ生成に使う。
#[derive(Debug)]
enum ArgsError {
    MissingOut,
    MissingValue { flag: String },
    UnknownArg { arg: String },
}

impl ArgsError {
    fn message(&self) -> String {
        match self {
            ArgsError::MissingOut => "missing required argument `--out <dir>`".to_string(),
            ArgsError::MissingValue { flag } => format!("missing value for `{flag}`"),
            ArgsError::UnknownArg { arg } => format!("unknown argument `{arg}`"),
        }
    }
}

const USAGE: &str = "usage: docs-site --out <dir> [--root <dir>]";

/// `std::env::args()` 由来の引数列（先頭のバイナリ名は除く）をパースする。
/// `--out` 必須・`--root` 任意（既定 `.`）・未知引数はすべて fail-closed で拒否する
/// （既定値への黙示フォールバックをしない。実装計画 §4「main.rs（CLI 骨格）」）。
fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, ArgsError> {
    let mut root: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => {
                let value = iter.next().ok_or(ArgsError::MissingValue {
                    flag: "--out".to_string(),
                })?;
                out = Some(PathBuf::from(value));
            }
            "--root" => {
                let value = iter.next().ok_or(ArgsError::MissingValue {
                    flag: "--root".to_string(),
                })?;
                root = Some(PathBuf::from(value));
            }
            other => {
                return Err(ArgsError::UnknownArg {
                    arg: other.to_string(),
                });
            }
        }
    }

    let out = out.ok_or(ArgsError::MissingOut)?;
    let root = root.unwrap_or_else(|| PathBuf::from("."));
    Ok(Args { root, out })
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("docs-site: {}", err.message());
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match build_site(&args.root, &args.out) {
        Ok(report) => {
            println!(
                "docs-site: validated {} page(s); wrote {} file(s) to {}",
                report.pages,
                report.written.len(),
                report.out_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("docs-site: build failed: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_required_out_and_defaults_root() {
        let parsed = parse_args(args(&["--out", "dist"])).expect("valid args should parse");
        assert_eq!(parsed.out, PathBuf::from("dist"));
        assert_eq!(parsed.root, PathBuf::from("."));
    }

    #[test]
    fn parses_explicit_root_and_out() {
        let parsed =
            parse_args(args(&["--root", "fixtures/valid", "--out", "dist"])).expect("should parse");
        assert_eq!(parsed.root, PathBuf::from("fixtures/valid"));
        assert_eq!(parsed.out, PathBuf::from("dist"));
    }

    #[test]
    fn rejects_missing_out() {
        assert!(matches!(
            parse_args(args(&["--root", "fixtures/valid"])),
            Err(ArgsError::MissingOut)
        ));
    }

    #[test]
    fn rejects_out_without_value() {
        assert!(matches!(
            parse_args(args(&["--out"])),
            Err(ArgsError::MissingValue { .. })
        ));
    }

    #[test]
    fn rejects_unknown_argument() {
        assert!(matches!(
            parse_args(args(&["--out", "dist", "--bogus"])),
            Err(ArgsError::UnknownArg { .. })
        ));
    }
}
