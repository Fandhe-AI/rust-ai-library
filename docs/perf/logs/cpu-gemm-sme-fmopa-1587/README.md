# SME `fmopa` マイクロカーネル A/B（イシュー #1587）実測ログ置き場

正式記録は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` を参照。

## 本 PR 時点の状態

- R3（数値契約: run-to-run bit 同一・SME vs scalar 参照の有限値／非正規化数
  bit 一致・NaN 混入で panic なし・本番入口 bit 一致）は Apple M4 Max
  実機で完全実施・全 PASS 済み（`docs/perf/cpu-gemm-sme-fmopa-microkernel.md`
  §4）。ログは `cargo test` の標準出力そのもの（決定的・非タイミング系の
  ため生ログの保存は不要と判断し本ディレクトリには含めない）。
- R1（framework-compare 非後退）・R2（checksum）・R4（しきい値の正式
  5-run 独立プロセス起動スイープ）は**未実施**（事前登録コメントで
  「本セッションの制約」として明記済み）。参考（非正式）計測は
  `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.2 に記載済み。
- 本ディレクトリには正式実測が完了した時点で以下を格納する:
  - `orchestrate_m4max.sh` / `orchestrate_dgx.sh`（5 プロセス独立起動・
    `uptime` 前後記録・`--dry-run` 対応）
  - `aggregate.py`（python3 標準ライブラリのみ・`--self-test` 付き）
  - 生ログ（`sme_vs_neon_ab_run{1..5}.log` 等）・`env_info.txt`
    （内部ホスト名はマスクする）
  - framework-compare gemm/train/infer cpu の before/after JSONL・
    `compare_*.md`

## 事前登録判定規則（issue #1587 コメントの転記）

正式版・一次ソースは GitHub issue #1587 の実装着手前コメント
（`https://github.com/Fandhe-AI/fandhe-ai/issues/1587#issuecomment-5648848287`）。
本 README は要旨のみを転記し、規則自体はコメント側を正とする
（事後の緩和はコメント側を編集せず新規コメントで記録する規約）。
