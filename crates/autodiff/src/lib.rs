//! 動的テープ式の自動微分エンジン。
//!
//! `tensor-core` が定義するテンソル型・演算グラフの上に、順伝播で実行した演算を
//! 動的テープへ記録し逆伝播で勾配を計算する（REQ-1 v2）。互換 API 層
//! （`compat::array`／`compat::Sequential` 等。REQ-9）はこのテープ機構を
//! 薄くラップして呼び出す。TASK-9.2a（#95）で `compat` モジュールへの隔離を
//! 確定した: 互換 API 層固有のロジック（numpy/Keras 慣習の入出力変換）は
//! `compat` モジュールに閉じ込め、コア（`tape`/`var`/`nn`）側へは互換慣習を
//! 一切漏らさない（依存方向は `compat` → `nn`/`var`/`tape` の一方向で、逆
//! 方向の `use` はない）という設計方針は継続する。
//!
//! **TASK-9.4（イシュー #411）で `compat` の唯一のサポート対象実装は
//! `fandhe_ai::compat` へ移設した**（10 クレート化・`facade` 新設〈TASK-9.3・
//! #410〉を受け、compat 公開面を composition root と同じ `facade`
//! クレートへ一本化するサポート境界の明文化。`docs/compat-api-scope.md`
//! 「サポート境界」節・`docs/spec/04-requirements.md:209-210` の
//! 2026-08-08 追記参照）。**移行期間中は本クレートの `compat` モジュール
//! （`pub mod compat`・非推奨シム）に旧実装を複製して残し、既存の
//! `fandhe_ai_autodiff::compat::{array, Sequential, SequentialVars}` 利用コードの
//! ソース互換性を保つ**（codex-review PR #424 P1 是正。詳細は
//! `crates/autodiff/src/compat/mod.rs` モジュール doc 参照）。本
//! クレート（`autodiff`）はこの移設後も compat 層が依拠する `Tape`/`Var`/
//! `nn`（`Module`・`Linear`・`activation` 等）を `pub` API として提供し
//! 続ける内部クレートである（利用者が `autodiff` を直接使うことは
//! サポート対象外。同ドキュメント参照）。
//!
//! TASK-1.5a（#16）でテープ構造（`Tape`/`TapeId`/`NodeId`）・
//! forward 演算群（`Var::matmul`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/
//! `max`/`mse_loss`）の値計算とノード記録を実装した（spec 根拠:
//! `docs/spec/05-tasks.md` TASK-1.5、`docs/public-api-design.md` §3）。
//! `Op`（`tape.rs`・非公開）が各演算の入力 `NodeId` を保持する構造にする
//! ことで、後続タスクが発生順に記録されたノード列を逆走査できる下地とする。
//!
//! TASK-1.5b（#17）で各演算の勾配関数（VJP: vector-Jacobian product）と
//! `Op` 単位のディスパッチ入口 `vjp()`（`grad.rs`・非公開）を実装した。
//! 数値微分との突合テスト（受け入れ条件）は `grad.rs` 内のユニット
//! テストに含む。
//!
//! TASK-1.5c（本イシュー・#18）で勾配伝播 API（`Tape::backward`・
//! `Gradients`。`backward.rs`）を実装した。テープを発生順とは逆順に
//! 走査して `grad::vjp()` を呼び、複数経路から同一ノードへ流入する
//! 勾配を合算する（PoC-v2-2 の `accumulate()` 相当）。合成関数
//! end-to-end 勾配の受け入れ条件検証は `tests/backward.rs` に含む。
//!
//! TASK-1.5d（#19）で PoC-v2-2 数値突合の回帰テストを追加した
//! （`tests/poc_v2_2_parity.rs`）。PoC-v2-2 の確定ケース（2 層 MLP grad
//! check・50 step SGD 学習の決定性）を `Tape`/`Var`/`Tape::backward`/
//! `Gradients` 経由で再現し、PoC evidence（`docs/spec/03-poc/
//! poc-v2-2-autodiff/evidence/`）の判定結果と整合することを固定する。
//! これにより #16〜#19（TASK-1.5 全体）が完了する。
//!
//! TASK-9.1b（#92）で活性化関数プリミティブ `Var::sigmoid`
//! （`var.rs`・`Op::Sigmoid`・VJP は `grad.rs`）と、互換 API 層
//! （REQ-9）が積む薄いレイヤー実装群の入口 `nn`（`nn::activation`。
//! ReLU/Sigmoid/Tanh）を追加した。共通 `Module` trait の定義は
//! TASK-9.2（#94/#95・`compat::Sequential`）に委ねる。
//!
//! forward の値計算は `backend-cpu`（TASK-1.6・#20 以降。並行実装中で
//! 未完）が完成するまでの暫定参照実装（`eval.rs`、非公開）で行い、
//! TASK-1.9（バックエンド抽象層への接続）で backend 経由の実行に
//! 差し替える（PoC-v2-2 と同じ構成）。`grad.rs`・`backward.rs` も同じ
//! `eval.rs` のヘルパーを再利用するため、差し替えの影響範囲は
//! forward/backward 双方でこの 1 ファイルに閉じる。
//!
//! TASK-9.1a（#91）で `nn` モジュール（`Linear` 等、自作 NN モジュール）
//! を追加した。`nn` は `Tape`/`Var` に直接依存する自作コア側の部品で
//! あり、互換 API 層（`compat::array`/`compat::Sequential`。REQ-9・
//! TASK-9.2）とは区別される（`nn/mod.rs` の境界説明を参照）。上記の
//! 「互換レイヤ固有のロジックを持ち込まない」方針は `compat` 層本体を
//! 指しており、`nn` モジュールには適用されない。
//!
//! #190（親 #189「損失関数（MSE・CrossEntropy）の実装」）で
//! `Var::mse_loss_with`/[`Reduction`]（mean/sum 縮約）と
//! `nn::loss::MseLoss`（薄いラッパー）を追加した。既存 `Var::mse_loss`
//! は `mse_loss_with(target, Reduction::Mean)` への委譲に変更したが、
//! シグネチャ・既定の意味（mean）は維持する（公開 API 非破壊）。
//!
//! #191（親イシュー #189）で CrossEntropy 損失（log-sum-exp 安定化・
//! クラス次元指定）を追加した。`Var::cross_entropy_loss`（`var.rs`・
//! `Op::CrossEntropyLoss`）は log-softmax → NLL を個別オペ合成せず
//! `MseLoss` と同じ 1 個の融合オペとして実装し、VJP（`grad.rs`）は
//! 解析形 `softmax(x) − onehot(t)` で閉じる。`nn::loss::
//! CrossEntropyLoss` はその薄いラッパー。`Reduction`（`Mean`/`Sum`）は
//! #190 が `var.rs` に定義したものをそのまま再利用する（`nn::loss` 側に
//! 重複定義は置かない）。
//!
//! #193（親 #192「optimizer（SGD・AdamW）・gradient clipping の実装」）
//! で optimizer の第 1 分割 `optim::Sgd`/`optim::SgdConfig`（momentum・
//! dampening・weight decay・nesterov 対応。PyTorch `torch.optim.SGD`
//! 準拠）を追加した。`nn`（`Tape`/`Var` に直接依存する層プリミティブ）
//! とは別モジュールとし（`optim/mod.rs` 参照）、既存 `nn`/`lib.rs` 冒頭の
//! 記述は変更しない。AdamW（#194）・gradient clipping／LR スケジューラ
//! （#195）は `optim` 配下への後続分割。
//!
//! TASK-9.2a（#95）で互換 API 層（当時は本クレート内の `compat`
//! モジュール）を追加した。共通 `nn::Module` trait（`nn/module.rs`）を
//! 確定し、`compat::array`（numpy `np.array` 慣習のテンソル生成）・
//! `compat::Sequential`（Keras `Sequential` 慣習のレイヤー積み上げ
//! ビルダー）を実装した。`Sequential` は `nn::Module` 経由で `Linear`・
//! 活性化関数（ReLU・Sigmoid・Tanh）を均一に扱う（対象範囲は
//! `docs/compat-api-scope.md` 準拠。学習〈勾配取得・パラメータ更新〉は
//! 当時対象外）。**TASK-9.4（#411）で唯一のサポート対象実装を
//! `fandhe_ai::compat` へ移設し、本クレートの `compat` モジュールは移行期間中
//! のソース互換シムとして実装を複製して残した**（本ファイル冒頭の
//! クレート doc・`compat/mod.rs` 参照。`nn::Module`・`Linear`・
//! `activation` 等、`compat` が依拠する `nn` 側の部品は本クレートに残る）。

//! #1047（親 #1043「カーネル融合・autodiff 実行モデルの強化」）で
//! view 系ノード `Var::reshape`/`Var::transpose`（`tape::Op::Reshape`/
//! `Op::Transpose`）を追加した。`push_eager`/`push_lazy` に続く第 3 の
//! 登録経路 `Tape::push_view` はホスト値を一切保持せず、backward 時
//! （または後続の実体化要求時）に入力ノードから `tape::resolve_view`
//! で再導出する（burn-autodiff の `MemoryBound { retro_forward }` 相当。
//! `tensor-core::Tensor::reshape`/`transpose` の zero-copy 性質（`Arc`
//! 共有）を利用するため中間バッファを一切確保しない。設計判断・メモリ
//! 実測は `docs/autodiff-view-recompute-decision.md` を参照）。
//! elementwise 5 演算の融合連鎖には参加しない融合境界ノードであり、
//! これは `docs/kernel-fusion.md` が既に確定させた「transpose を挟む
//! 連鎖は融合しない」方針と整合する。

//! イシュー #1624 で activation checkpointing（`torch.utils.checkpoint`
//! 相当）を追加した。`Tape::checkpoint`（閉包版）／`Var::checkpoint_from`
//! （低儀式版。facade からはこちらのみ既存の `Var` 再エクスポート経由で
//! 到達可能）が区間内の再計算可能ノード（`Op::MatMul`／`Sigmoid`／
//! `Sum`／`Max`。`Op::is_checkpoint_eligible()`）の forward 値を解放し、
//! 上記 view 系ノードの `resolve_view` 機構を一般化した `tape::
//! recompute_value`／`recompute_fallible`／`recompute_infallible` が
//! backward 時に再導出する。設計判断・実装記録は `docs/
//! autodiff-checkpoint-design.md` を参照。
//! #1597 で同じ `push_view`／`resolve_view` 骨格を任意軸並べ替え・
//! ブロードキャストへ一般化した `Var::permute`（`tape::Op::Permute`。
//! zero-copy）・`Var::broadcast_to`／`expand`（`tape::Op::BroadcastTo`。
//! forward は stride 0 view で zero-copy だが VJP は `Op::Add`/`Op::Mul`
//! の暗黙ブロードキャストと同じ `reduce_to_shape` 縮約を使うため勾配
//! バッファを確保する）を追加した。`Var::squeeze`／`unsqueeze`／
//! `flatten` は新規 `Op` を持たず `Var::reshape` へ委譲する（案 A
//! 制約を継承。`docs/autodiff-view-recompute-decision.md` §5 が予告
//! した拡張）。

mod backward;
pub mod compat;
mod default_ops;
mod error;
mod eval;
mod grad;
mod layout;
pub mod nn;
pub mod optim;
mod tape;
#[cfg(test)]
mod test_support;
mod var;

pub use backward::Gradients;
pub use error::AutodiffError;
pub use tape::{NodeId, Tape, TapeId};
pub use var::{GateParams, QrVars, Reduction, SvdVars, Var, VarHostView};
