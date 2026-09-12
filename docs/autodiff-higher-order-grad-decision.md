# 高階微分（grad of grad）の設計判断（イシュー #1622）

- 対応イシュー: #1622（親 #1573〈Tier 2〉→ ルート #1570）
- 位置づけ: 本文書は**設計判断のみ**を記録する。`crates/` 配下・`docs/spec/`（正本 submodule）へのコード変更は行わない。tolerance／baseline（数値一致の許容誤差・非後退ベースライン）の変更も対象外
- 基準コミット: `origin/main` `581d5208`（#1675「elementwise の DeviceBuffer 常駐版と sum／max reduce を実装する」時点）。行番号・事実はすべて本コミットで再確認した

## 1. 背景

- 機能ギャップ表 `docs/compat-feature-gap.md` §2.11「高階微分（`grad of grad`）」行（`docs/compat-feature-gap.md:310`）は「なし（テープは 1 階のみを前提とした構造と推定）・必要物: Op 自体を微分可能にする再設計（VJP の VJP）・難度 XL」で、設計から要検討のまま残っていた
- spec 側は 2026-09-12 の REQ-9 改定（`docs/spec/04-requirements.md:232`）で高階微分を **Tier 2**（長尾。PyTorch／TensorFlow 機能網羅の第 2 段階）に明記済み。実装リポ側の `docs/compat-api-scope.md` §1.3 Tier 2 表は当該行を「高階微分 | #1622（設計記録）」（`docs/compat-api-scope.md:242`）としており、本イシューが設計記録の担当
- 対比対象: PyTorch `torch.autograd.grad(create_graph=True)`（VJP をテープ上の演算として記録し、その計算グラフをさらに `backward()` できる）、TensorFlow `GradientTape` のネスト（外側テープが内側テープの `gradient()` 呼び出しを記録する）。いずれも Hessian-vector product・二階の損失正則化（gradient penalty）・メタ学習（MAML 等）の実装に使われる
- 目的（受入基準の構造化）:
  1. VJP の VJP・テープ再設計の要否を検討し、設計内容と判断根拠を doc として記録する（実装は含めない）
  2. 承認事項がある場合は doc 本文に明記し、実装着手の前提として残す（§10）
  3. tolerance・baseline は変更しない

## 2. 現状のコード事実（`origin/main` `581d5208`）

| 事実 | 出典 |
|---|---|
| `grad::vjp(op, out_value, upstream, nodes, ops, resident, tape_id, tape_epoch) -> Result<Vec<(NodeId, Tensor<f32>)>, AutodiffError>` は入力・出力とも**生 `Tensor<f32>`**（`Var` ではない）で計算する。`ops: &dyn BackendOps`（forward と同じカーネル。#1674 で elementwise VJP を `BackendOps` 経由化）と `eval::*`（ホスト参照実装。三角ソルブ等）を直接呼ぶのみで `Var`／`Tape::push_*` を一切使わない。**二階の計算グラフはテープ上に一切生成されない** | `crates/autodiff/src/grad.rs:184-198` |
| `Tape::backward_impl` は逆走査ループ全体を単一の不変借用 `let nodes = self.nodes.borrow();` で完結させる設計（`RefCell` 二重可変借用 panic を経路ごと排除する意図的な設計。doc コメントに明記）。同一テープへ VJP 演算を `push_*`（`self.nodes.borrow_mut()`）で記録しようとすると、この不変借用と衝突し panic する | `crates/autodiff/src/backward.rs:132-145` |
| `Gradients { tape_id, epoch, grads, resident_fingerprint }` は `Vec<Option<Tensor<f32>>>` を保持する値型で `Tape` を借用しない。`Tape::reset()` は葉プレフィックスのみ `truncate` で残し `epoch` を 1 進める（`Tape::reset`）。`Gradients::get` は `tape_id`／`epoch` 不一致で `TapeMismatch` を返す fail-closed 設計 | `crates/autodiff/src/backward.rs:40-48`・`crates/autodiff/src/tape.rs:796-802,620-626` |
| `Tape` は `Send` が静的アサーションで固定されている（`assert_send::<Tape>()`） | `crates/autodiff/tests/fusion_backend_integration.rs:388-393` |
| `Op`（`pub(crate)` のクローズド enum）のうち `QrQ{r}`／`QrR{q}`／`SvdU{s,vh}`／`SvdS`／`SvdVh`／`CrossEntropyLoss{targets: Tensor<i32>}` は**非追跡ペイロード**（`Tensor<f32>`／`Tensor<i32>` を直接持つ）を持つ。`LinearAct`／`LinearResident`／`ResidentLeaf`（reuse 経路）の d_weight は `ResidentResolver::fill_resident_weight_grad` がデバイス常駐 staging へ直接書き込み `Gradients` に一切現れない | `crates/autodiff/src/tape.rs:140,167,189,221,286-304`・`crates/autodiff/src/grad.rs:367-596`（`Op::LinearResident` の match arm）・`crates/autodiff/src/tape.rs:449` |
| VJP 内部で使う演算のうち**現行 `Var` に公開 API として存在しないもの**: 減算・スカラー倍相当（`tanh_grad_factor`〈1−y²〉・`sigmoid_grad_factor`〈y(1−y)〉・`mse_loss_scale`）、マスク積（`elementwise_mul_mask`）、`reduce_to_shape`／`unreduce_broadcast`（broadcast の逆・keepdim 復元）、`max_vjp`（argmax 散布）、`softmax_vjp_along`・`log_softmax_vjp_along`（行ごとの内積と broadcast）、`cross_entropy_loss_vjp`（one-hot／gather 相当）、線形代数 VJP（`inv`／`det`／`cholesky` の三角ソルブ等）。`Var::sub`／`Var::neg` は未実装（#1593） | `crates/autodiff/src/grad.rs:833,978,1212,1222,1239,1285,1330,1391,1500,1515`・`crates/autodiff/src/var.rs`（`sub`／`neg` 不在をソース走査で確認） |
| 遅延評価（`push_lazy`・`MAX_FUSED_CHAIN_LEN`）・`push_view`（reshape／transpose の再計算方式。`docs/autodiff-view-recompute-decision.md`）は forward の elementwise 連鎖のみを対象とする。`docs/fusion-graph-design.md` §3.3「backward（VJP）は融合対象外」（`docs/fusion-graph-design.md:767,774,789`）が明示契約 | `docs/fusion-graph-design.md:767-789` |
| 公開面: `facade::Tape(pub(crate) fandhe_ai_autodiff::Tape)` の newtype・`Gradients`／`Var` を素で再エクスポート。`crates/facade/tests/api_surface.rs` が `pub use` での `Tape`／`BackendOps`／`new_with_ops` 再エクスポート禁止・`pub fn` が `BackendOps` を直接引数に取ることの禁止を機械検査する | `crates/facade/src/lib.rs:117,154`・`crates/facade/tests/api_surface.rs:66-100` |
| `AutodiffError`（`#[non_exhaustive]`）の variant は `Shape`／`Backward(String)`／`TapeMismatch`／`InvalidArgument(String)`／`Backend`。「（現時点で）未対応の演算」を表す専用 variant はなく、既存実装は `InvalidArgument` を「shape 検査より前に弾く構築時エラー」用途で使っている | `crates/autodiff/src/error.rs:21-65` |
| REQ-12「利用者向け融合制御 API を提供しない」の受け入れ基準は `facade` を唯一の公開面としバックエンド結線を composition root に集約することで充足する設計（`create_graph` 相当のフラグを追加する場合、これが融合制御 API に該当しないことの整理が必要） | `docs/spec/04-requirements.md:277-280` |
| PoC-v2-2（`docs/spec/03-poc/poc-v2-2-autodiff/README.md`）・`docs/public-api-design.md` §3 は 1 階の動的テープのみを確定しており高階微分への言及がない（grep 済み） | `docs/spec/03-poc/poc-v2-2-autodiff/README.md`・`docs/public-api-design.md` |

**追記（イシュー #1624・activation checkpointing）**: 上表「`Tape::backward_impl` は逆走査ループ全体を単一の不変借用 `let nodes = self.nodes.borrow();` で完結させる」という事実は #1624 で変わった——checkpoint 区間の再解放（`Tape::release_checkpoints_ending_at`）が逆走査の途中で `self.nodes.borrow_mut()` を必要とするため、`backward_impl` は「反復ごとに `Ref` を取得し、その反復内（VJP 呼び出しまで）で drop する」方式へ再構成された（`crates/autodiff/src/backward.rs` の `for id in (0..n).rev() { let contributions = { let nodes = self.nodes.borrow(); ... }; ...; self.release_checkpoints_ending_at(id); }`）。ただし本節の結論（A-1「同一テープ方式」の困難）は不変: 依然として「ある反復の処理中（`Ref` が生存している間）に同一テープへ `push_*`〈`borrow_mut()`〉で VJP 演算を記録する」ことはできない（`Ref` の生存期間が反復単位に狭まっただけで、反復内で push しようとすれば同様に衝突する）。`docs/autodiff-checkpoint-design.md` §3.1 点 4 参照。

## 3. 契約整理（設計が守るべき既存契約）

1. 1 階の `Tape::backward` の結果は**bit 同一で不変**（既存 backward／parity／bit 一致テスト群を後退させない）。REQ-2 統一複合判定・FMA 契約・tolerance／baseline はいずれも不変
2. `docs/fusion-graph-design.md` §3.3「backward（VJP）は融合対象外」契約（現状は「VJP がテープに一切乗らない」というさらに強い状態であり、これを緩めて VJP をテープへ乗せる案は本契約の解釈拡張を伴う）
3. `Tape: Send`（`crates/autodiff/tests/fusion_backend_integration.rs:388`）は維持する
4. `TapeId`／`epoch` の世代契約（`Gradients::get` の fail-closed 検査）・`Tape::reset` の葉プレフィックス契約は維持する
5. REQ-12「利用者向け融合制御 API を提供しない」・facade 公開面の機械検査（`api_surface.rs`）は維持する
6. reuse（デバイス常駐）経路・`Op::LinearResident`／`ResidentLeaf`・CUDA Graph capture（#1349）は高階微分の対象外とする（staging 書き込みは `Gradients` に現れず、二階の入力として使えない）

## 4. 設計案の比較

| 案 | 概要 | テープ再設計 | `RefCell` 借用の回避 | 前提 issue | §3 契約への影響 | 数値契約 | facade 公開面 | 実装難度 |
|---|---|---|---|---|---|---|---|---|
| **A-1: 同一テープへの `create_graph`** | `backward` 中に VJP 演算を同一 `Tape` へ `push_*` で記録する | 要（backward を「不変借用のまま逆走査」から「都度 borrow を解放する二相化」へ再設計） | 借用衝突を構造的に解消する必要（走査と記録を分離するか、`RefCell` を別のインデックス方式に置換） | #1593／#1597／#1599／#1601（全 VJP 演算を `Var` 演算として再表現するため） | 契約 2（VJP を融合対象に含めるか要再定義）・契約 4（backward 中に生成したノードの世代扱い）に抵触。承認事項 | 変更なし（backward の実行順序自体は不変） | `Var` を返す新 API が必要 | 高（backward の内部設計を破壊的に変更） |
| **A-2: 子テープ（`create_graph` を新規 `Tape` へ記録）** | `backward` が既存 `Tape` とは別の新規 `Tape` を構築し、VJP を `Var` 演算としてそこへ記録して返す | 不要（既存 `backward_impl` のロック方式は温存し、外側からは別インスタンスとして扱う） | 解消（別インスタンスの `RefCell` のため衝突しない） | 同上（VJP 演算の `Var` 化） | 契約 2・3（子 `Tape` の `ops`〈`Box<dyn BackendOps + Send>`〉の所有権をどこから得るか）は要検討だが契約自体の書き換えは不要 | 変更なし | `Tape::backward_with_graph(&self, loss) -> (Gradients, Tape)` 相当の新 API が必要（承認事項） | 高（全 VJP の `Var` 再表現＋子テープの `ops` 供給経路の設計） |
| **B: ネストテープ（TF `GradientTape` 方式）** | 外側テープが内側 `backward` の各 VJP 呼び出しを `Op::Backward{...}` 相当の 1 ノードとして記録し、その VJP（＝二階 VJP）を Op ごとに手書きする | 要（新 Op variant・二階 VJP の手書き実装が Op の数だけ必要） | 解消（内側 backward は独立して完結） | 手書き二階 VJP の網羅性検討が別途必要（前提 issue は A と同一集合＋線形代数・非追跡ペイロード Op の二階 VJP 追加） | 契約 6（非追跡ペイロード Op の二階微分は未定義のまま据え置きやすい）との整合は取りやすいが、Op 追加のたびに二階 VJP を保守する負担が恒久的に生じる | 変更なし | 新 API が必要 | 非常に高（Op ごとに二階 VJP を手書き・保守コスト大） |
| **C: forward-over-reverse（JVP／双対数）による HVP 限定提供** | Hessian-vector product のみを目的とし、`vjp(x)` の JVP（前方モード）を計算する。テープを介さない | 不要 | 該当なし（テープを使わない） | `Op` ごとの JVP 追加（`BackendOps` trait 拡張に相当。#1623〈custom Op〉の設計と関係） | 契約 2〜6 への抵触なし（VJP／テープの外側で完結） | 変更なし | HVP 専用の新 API（`facade` へ載せる場合は §7 の手続きが必要） | 中（HVP 用途に限定すれば Op 種別は絞れるが、`BackendOps` trait 拡張は承認事項） |
| **D: 有限差分による HVP の暫定提供** | `(grad(x + εv) − grad(x)) / ε` を数値的に計算する | 不要 | 該当なし | なし | 契約への抵触なし | 変更なし（tolerance は新規に定めない） | 載せない（診断用途限定） | 低（既存 `grad` の呼び出しを組み合わせるだけ） |
| **E: 現時点では非対応と明文化し再開条件を定義する** | 高階微分は未対応のまま、型付きエラー（`InvalidArgument` 相当）または `compat-api-scope.md` への明記で意図的に留保する | 不要 | 該当なし | なし（前提 issue の完了を再開条件とする） | 契約への抵触なし | 変更なし | 変更なし | 極小（ドキュメント整理のみ） |

## 5. 推奨（段階的・前提ゲート付き）

自動運転での作業のため、以下は「推奨」であり**採用の確定はユーザー承認に委ねる**（§10）。

- **段階 0（本イシューで完了）**: 案 E を採用し、高階微分が現時点で未対応であること・その理由（§2・§3 の事実）・再開条件を本 doc に明文化する。前提 issue（#1593 sub／neg・#1597 expand／unsqueeze・#1599 where／gather／scatter・#1601 keepdim／複数軸・#1612 no_grad／detach／retain_graph）はいずれも本執筆時点で OPEN であり、これらの完了を高階微分の実装再開条件とする
- **段階 1（前提充足後・別イシュー）**: 主案として **案 A-2（子テープ方式の `create_graph`）**、代替として **案 C（HVP 限定 JVP）** を比較実装 issue の起票候補とする。対象 Op は elementwise・matmul・sum／max・softmax・MSE／CE に限定し、線形代数 Op（非追跡ペイロード）・reuse 経路・`LinearAct` は初期スコープ外とする
- **判断根拠**:
  - 同一テープ方式（案 A-1）は `backward_impl` の不変借用前提（契約維持のための panic 回避設計。`backward.rs:132-145`）を壊すため、backward の実行モデル自体の再設計を要する。子テープ方式（案 A-2）は既存 `backward_impl` を温存したまま「二階の勾配グラフをどこに置くか」という論点だけを切り出せるため、既存契約への影響が最小
  - ネストテープ方式（案 B）は Op 追加のたびに二階 VJP を手書きで保守する恒久コストを生む。本リポは `Op` を頻繁に拡張している最中（線形代数・RNN 系イシュー等）であり、保守負担の増大が大きい
  - 案 C（JVP／HVP 限定）はテープ再設計が不要で契約への影響が最小だが、提供できる機能が HVP に限られ「grad of grad」の一般形（任意階数の `create_graph`）には届かない。段階 1 の代替案として位置づける
  - 案 D（有限差分）は実装難度が最小だが数値精度が ε 依存であり REQ-2 の統一複合判定の対象にできない。診断・デバッグ用途の暫定手段としてのみ有効

## 6. 数値一致・既存テストとの整合

- 1 階 backward の不変性テスト（既存 `crates/autodiff/tests/backward.rs`・`tape_recording.rs` 等）の非後退を、段階 1 実装 issue の受入条件として引き継ぐ
- 二階検証テスト案（段階 1 で具体化）: 小形状（2×2〜4×4 程度）での解析解（手計算または `f64` ホスト参照実装）との突合、案 D（有限差分）との相互検証を組み合わせる
- **新規 tolerance は本イシューでは定めない**。段階 1 で二階微分専用の許容誤差・baseline が必要になった場合は、既存の tolerance／baseline 変更と同様にユーザー承認を要する（§10）

## 7. 公開 API・spec 整合

- REQ-9 は 2026-09-12 改定で高階微分を Tier 2（長尾）に明記済み（`docs/spec/04-requirements.md:232`）。実装リポ側 `docs/compat-api-scope.md` §1.3 は当該行を「高階微分 | #1622（設計記録）」としており、本 doc がその設計記録に該当する
- facade 公開面（`api_surface.rs`）は `Tape`／`BackendOps` の再エクスポート・`BackendOps` を直接引数に取る `pub fn` を禁止する。段階 1 で `Var` を返す高階 API（案 A-2 の `Tape::backward_with_graph` 相当）を facade へ載せる場合、`docs/compat-api-scope.md` §5「範囲拡張の手続き」（正本 REQ-9 改定または本リポのユーザー承認）を経る必要がある
- REQ-12「利用者向け融合制御 API を提供しない」と `create_graph` 相当のフラグの関係: `create_graph` は「二階の勾配グラフを構築するか否か」の選択であり、カーネル融合の制御（§3.3 の融合境界）とは別軸である。段階 1 の設計時に、このフラグが REQ-12 の禁止対象（融合制御 API）に該当しないことを明示的に整理する必要がある（`docs/spec/04-requirements.md:280` の「autodiff の ops 受け取り構築子はサポート外の内部 API」と同型の整理を要する）

## 8. 実装イシューへの引き継ぎ（起票草案。起票自体は行わない）

段階 1 の起票候補（`.claude/rules/out-of-scope-tracking.md` に従い、ユーザー承認後に起票する）:

- **対象**: 案 A-2（子テープ方式の `create_graph`）を主案として、elementwise・matmul・sum／max・softmax・MSE／CE の VJP を `Var` 演算として再表現する
- **前提 issue**: #1593（sub／neg）・#1597（expand／unsqueeze）・#1599（where／gather／scatter）・#1601（keepdim／複数軸）・#1612（no_grad／detach／retain_graph）の完了
- **スコープ外**（初期実装）: 線形代数 Op（`Qr*`／`Svd*` 等の非追跡ペイロード）・reuse／resident 経路（`LinearResident`／`ResidentLeaf`）・CUDA Graph capture・`LinearAct`
- **受入基準候補**: 1 階 backward の非後退（既存テスト群）、小形状での二階微分の解析解一致（新規テスト）、facade 公開面変更を伴う場合は §7 の手続き完了

## 9. スコープ外

- reuse／resident 経路（`LinearResident`／`ResidentLeaf`）・CUDA Graph capture（#1349）・線形代数 Op の高階微分
- 混合精度（#1625）・activation checkpointing（#1624。再計算との併用は本 doc では言及のみに留める）
- custom autograd Function（#1623。`Op` enum 拡張の論点は重なるが、本 doc では参照に留め決定しない）

## 10. 承認事項（実装着手の前提）

1. `docs/fusion-graph-design.md` §3.3「VJP は融合対象外」契約の解釈拡張（案 A／B を採る場合、VJP 自体をテープへ記録する行為がこの契約とどう整合するかの再定義）
2. `Op` enum／`BackendOps` trait の拡張（JVP 追加〈案 C〉・二階 VJP 追加〈案 A／B〉）
3. facade 公開面への高階 API 追加（`docs/compat-api-scope.md` §5 手続き）
4. 二階微分の数値判定方式（新規 tolerance／baseline を伴う場合）
5. 段階 1 実装 issue の起票

## 11. 出典

- Issue #1622・#1573・#1570
- `docs/compat-feature-gap.md:310`（§2.11）
- `docs/compat-api-scope.md:242`（§1.3）
- `docs/spec/04-requirements.md:232`（REQ-9 2026-09-12 追記）・`docs/spec/04-requirements.md:277-280`（REQ-12）
- `docs/fusion-graph-design.md:767-789`（§3.3）
- `docs/autodiff-view-recompute-decision.md`
- `docs/autodiff-nograd-leaf-dinput-skip-decision.md`
- `docs/device-resident-update-design.md` §3.3b
- `docs/public-api-design.md` §3
- `docs/spec/03-poc/poc-v2-2-autodiff/README.md`
- `crates/autodiff/src/grad.rs:184-198,833,978,1212,1222,1239,1285,1330,1391,1500,1515`
- `crates/autodiff/src/backward.rs:40-48,132-145`
- `crates/autodiff/src/tape.rs:140,167,189,221,286-304,449,620-626,796-802`
- `crates/autodiff/src/error.rs:21-65`
- `crates/autodiff/tests/fusion_backend_integration.rs:388-393`
- `crates/facade/src/lib.rs:117,154`
- `crates/facade/tests/api_surface.rs:66-100`

内部ホスト名・秘密情報は含めない。
