# カーネル融合の適用範囲・限界（TASK-12.2b）

> 役割・参照元: 本文書は REQ-12（`docs/spec/04-requirements.md:246`）の
> v2 読み替え後タスク分解 TASK-12.2（`docs/spec/05-tasks.md:377`。
> 「TASK-12.1 の実装に対し、elementwise 連鎖・fan-out・transpose 混在
> ワークロードでの融合効果を実測する。matmul・softmax を含む複合
> ワークロードでは融合の効果を前提とした性能目標を設定しないこと
> （REQ-8 の複合ワークロード系目標との整合）を明記する」）の文書化部分
> （TASK-12.2b・本イシュー #168）の成果物である。親イシューは #166
> （TASK-12.2）、実測部分の兄弟イシューは #167（TASK-12.2a）。
>
> - **設計正本**: `docs/fusion-graph-design.md`（TASK-12.1a・#161。融合
>   IR・実体化境界・`BackendOps` 契約接続の確定設計）
> - **実測正本**: `docs/perf/cpu-elementwise-fusion-effect.md`
>   （TASK-12.2a・#167。elementwise 連鎖・fan-out・transpose 混在の実測）、
>   および `docs/perf/cpu-gemm-epilogue-fusion.md`（TASK-12.1f・#203。
>   GEMM epilogue 融合の実測）
> - **本文書の更新時点の状態**: TASK-12.2a（#167）で elementwise 連鎖
>   融合の実測を実施済み（実測正本: `docs/perf/
>   cpu-elementwise-fusion-effect.md`）。#167 の実装時点で
>   `CpuBackendOps::run_fused` が融合カーネルへ未結線（TASK-12.1 系列
>   〈#163・#164〉が結線を相手のスコープと委ね合っていたギャップ）で
>   あることが判明したため、同イシューの前提ステップとして結線を実施
>   したうえで実測している（詳細は同記録 §0）。
>   その後 **Phase G（#582。融合・正規化・量子化の複合ワークロード
>   軸）**の G-2（#586）・G-3（#588）で `FusionOp` へ `Sum`／`Max`／
>   `Rsqrt`／`Sub`／`Div` を追加し、行方向 reduction + broadcast
>   （RMSNorm／softmax 型）セグメントをプラン表現・`FusionPlanError`
>   による fail-closed 検証まで実装した（`crates/tensor-core/src/
>   fusion/{graph,plan}.rs`）。ただしバックエンド側の `run_fused` 実行
>   はこの拡張に追随しておらず、CPU 実装（`crates/backend-cpu/src/
>   fused_elementwise.rs:129`）は reduction（`Sum`／`Max`）・`Rsqrt` を
>   含むプランを明示的に未対応として拒否する。バックエンドカーネル実装は
>   G-6 以降（CUDA #592・#594、Metal #604、CPU #607）で進行中であり、
>   本文書の更新時点では未完了である。本イシュー（#591）はこの Phase G
>   による IR 拡張を受けた §1・§3 の記述改定要否を判断した結果を反映
>   したものであり、判断の詳細は §7 に記録する。

## 1. 判断サマリ

- (a) 融合の適用範囲は **elementwise 演算連鎖（4〜6 段程度。`add`／
  `mul`／`relu`／`exp`／`tanh` の混在を含む）** と **GEMM epilogue
  （bias・activation）** の 2 系統に限る（`docs/fusion-graph-design.md`
  §1・§2.1、TASK-12.1f・#203）。
- (b) **reduction エピローグ（`.sum()`／`.max()`）・matmul／softmax を
  挟む複合連鎖は初期スコープ外**とする（TASK-12.2b 策定時点＝
  `docs/fusion-graph-design.md` §3.2 (a)(b)、PoC-9 実測が根拠）。
  **Phase G（#582。#591 で反映）による更新**: G-2（#586）・G-3
  （#588）で `FusionOp` に `Sum`／`Max`／`Rsqrt` を追加し、行方向
  reduction + broadcast（RMSNorm／softmax 型）セグメントを IR・プラン
  検証（`FusionPlanError` による fail-closed 拒否）レベルで表現可能に
  した。これにより「初期スコープ外」なのは matmul を挟む連鎖のみで
  あり、reduction エピローグ・softmax 型の行方向縮約は IR 表現上は
  スコープ内に拡張された。ただしバックエンド実行（`run_fused`）は
  この IR 拡張に未追随（CPU は reduction／`Rsqrt` を含むプランを明示
  的に未対応拒否。§3 表 1・3 行目参照）であり、**利用者が観測できる
  性能挙動としては本文書執筆時点でなお変化していない**（詳細は §7）。
- (c) **matmul・softmax を含む複合ワークロード（attention 系連鎖等）
  では融合効果を前提とした性能目標を設定しない**（REQ-12 受け入れ
  基準 `docs/spec/04-requirements.md:255`。REQ-8 整合。§5 参照）。
  **Phase G 後も現行維持**（§4・§7 参照。Phase G は複合 WL の実効性能を
  改善しうる実装群だが、REQ-8 下限（`docs/performance-targets.md`）の
  前提を変更するものではない）。
- (d) **利用者向け融合制御 API は提供しない**（REQ-12 受け入れ基準
  `docs/spec/04-requirements.md:252`）。融合は `facade` クレートの
  composition root（`Device` → 具体 `BackendOps` の結線）を経由した
  既定バックエンド経由でのみ到達し、任意 `BackendOps` を注入できる
  公開 API は設けない（`docs/fusion-graph-design.md` §1「既定バック
  エンド供給は `facade` クレートの composition root が担う」・
  `docs/spec/05-tasks.md:316` TASK-9.3）。
- (e) transpose を挟む連鎖も初期スコープでは**融合しない（非融合
  フォールバックへ倒す）**（`docs/fusion-graph-design.md` §1・§2.3。
  §3 参照）。

## 2. 適用範囲

### 2.1 elementwise 連鎖融合

- **対象**: `add`／`mul`（二項）・`relu`／`exp`／`tanh`（単項）の
  elementwise 5 演算からなる 4〜6 段程度の連鎖。fan-out（1 つの中間
  テンソルを複数ノードから消費する分岐）も対象に含む（`docs/
  fusion-graph-design.md` §2.4。PoC-9 の `ew_fanout` 実測が根拠）。
- **成立条件**（`docs/fusion-graph-design.md` §2〜§3 の実装仕様を要約。
  実装ファイルは `crates/tensor-core/src/fusion/{graph,detect,mod}.rs`）:
  - 連鎖が単一の連結成分（`FusionOp` の DAG）として elementwise のみで
    閉じていること。`gemm`／`sum`／`max` は融合境界ノードであり連鎖を
    分断する（§3.2 (a)(b)）。
  - 各ノードの `NodeMeta.contiguous == true`（transpose／broadcast view
    を含まない）こと（§2.3・§2.1 の非融合フォールバック判定）。
  - 連鎖長が 4〜6 段の上限に収まること（§3.2 (d)。`Tape::push_lazy`
    （`crates/autodiff/src/tape.rs`）が push 時点で `lazy_chain_size`
    （新規ノードから `build_lazy_plan` が実際に収容する未実体化
    interior ノード数の上界。fan-in を伴う枝合流も総和で捉える。
    codex-review PR #406 の P1 是正で最大値ベースから再設計）を計算し、
    `tensor_core::MAX_FUSED_CHAIN_LEN`（= 6）到達時点でその場の演算を
    実体化してから連鎖を再開する**実装済み**（#404）。詳細は §3 表 5
    行目を参照）。
  - `dtype` は `DType::F32` 固定（`BackendOps` の現行スコープに合わせる。
    §2.1・§2.3）。f16 融合は §4 の未決事項。
- **利用者の記述形との対応**: 独立した複数回の公開 `Var` 呼び出し
  （`a.add(&b)?.relu().exp().tanh()` のように演算をまたぐ記述）が融合対象
  になりうる（`docs/fusion-graph-design.md` §1「演算跨ぎの遅延・二項
  elementwise 演算の遅延化」）。融合の成否は `Tape::new(ops)`（§1）へ渡
  された `ops` の具体的な実装によらず同一方針であり、利用者が明示的に
  融合を切り替える経路は存在しない（判断サマリ (d)）。
- **実測（CPU・TASK-12.2a・#167。出典: `docs/perf/
  cpu-elementwise-fusion-effect.md`「実測結果」節）**: `ew4`（4 段連鎖）・
  `ew6`（6 段連鎖）・`ew_fanout`（fan-out）のいずれも非融合（per-op
  フォールバック）比 **1.52〜1.57 倍**の改善（5 回計測の中央値。全パターン
  で融合が非融合を上回る）。transpose 混在（`ew_transpose`）はほぼ互角
  （中央値 0.981x、レンジ 0.865〜1.581x。§3 表 4 行目参照）。PoC-9（v1・
  Metal 実機）の定性的傾向（連鎖・fan-out で融合が有意に効く）と一致する
  （絶対値は実行系の違いにより異なるため直接比較しない）。数値一致は
  REQ-2 統一複合判定で確認済み。

### 2.2 GEMM epilogue 融合

- **対象**: `matmul` の直後に続く bias 加算・activation（Linear+bias+
  ReLU 相当）。`BackendOps::gemm_bias_act`（`CpuBackendOps` 実装）が
  行パネル並列の GEMM 完了直後・同一 `rayon` タスク内で bias 加算・
  activation を適用し、中間 `Tensor` 割当を出力 1 個に抑える（`docs/
  perf/cpu-gemm-epilogue-fusion.md`）。
- **実測（CPU・TASK-12.1f・#203。出典: `docs/perf/
  cpu-gemm-epilogue-fusion.md`「実測結果」節）**: Linear+bias+ReLU 相当
  形状・正方形状の計 5 形状で、非融合（`BackendOps::gemm_bias_act` の
  デフォルト実装＝ `gemm` → `add` → `relu` の逐次呼び出し）比 **1.46〜
  2.56 倍**の改善（5 回中央値。全形状で融合が非融合を上回る）。数値一致
  は bit 完全一致（epilogue が要素ごとに独立な演算で演算順序に依存しない
  ため。同文書「数値一致」節）。
- **GPU（CUDA NVRTC・Metal MSL）**: イシュー #599 で CUDA 側に elementwise
  5 演算（`add`／`mul`／`relu`／`exp`／`tanh`）と `gemm_bias_act` の
  epilogue 融合カーネル（`kernels::TILED_BIAS_ACT_F32`・
  `CudaGemm::run_tiled_bias_act_f32`）を実装した。`bias` が `None`、
  または `B: [k, n]` の列数 `n` に厳密一致する `[n]` 形状の場合に融合
  カーネルへ進み（`ops::gemm_bias_act_route`）、`gemm`→`add`→`relu` の
  非融合合成と同一の tiled アキュムレーション順序を経由するため bit
  完全一致する（CPU `gemm_blis_bias_act_parallel` と同じ「epilogue を
  カーネル内で完結させる」設計思想）。`[1]`・`[1, n]` 等ブロードキャスト
  可能だが `[n]` ちょうどでない shape はデフォルト実装（非融合合成）へ
  フォールバックする（CUDA は本イシューで `add`／`relu` を実装済みの
  ため CPU と異なり `Unsupported` を経由しない）。実機での実測（融合 vs
  非融合の 5 回計測中央値）は `docs/perf/cuda-gemm-epilogue-fusion.md`
  を参照（未実施の場合はその旨が明記される）。**イシュー #1137 追記**:
  非融合合成側の `gemm`（`CudaGemm::run_tiled_f32`）は #1137 で cp.async
  3 stage パイプラインカーネルへ形状条件付きに分岐しうるようになったが、
  GB10 実機実測（`tests/cpu_cuda_tiled_pipeline_parity.rs::
  tiled_pipeline_matches_tiled_f32_classic_bit_exact`）でパイプライン版と
  classic 版（`TILED_F32`。融合カーネル `TILED_BIAS_ACT_F32` と同一の
  タイリング・アキュムレーション順序を持つ側）がビット完全一致すること
  を確認済みのため、本節の bit 完全一致契約（`tests/gemm_bias_act_
  parity.rs` の `assert_eq!` 検査）は #1137 後も成立する
  （`docs/perf/cuda-gemm-tiled-pipeline.md`「#1137 本番結線判断」節参照）。
  Metal はイシュー #605 で
  同様の elementwise 5 演算と `gemm_bias_act` epilogue 融合カーネル
  （`shaders/gemm.metal::gemm_tiled_bias_act`・
  `MetalGemm::run_tiled_bias_act_f32`）を実装した。経路選択の分岐条件
  （`ops::gemm_bias_act_route`）・bit 完全一致の論拠は CUDA 側と同一
  （いずれも `gemm_tiled`／`gemm_tiled_bias_act` の同一タイリング順序を
  経由するため）。既存 `gemm_naive`／`gemm_tiled`／`gemm_simdgroup`／
  `gemm_simdgroup_tiled`／`gemm_simdgroup_f16` カーネルには一切触れて
  おらず GEMM 単体は構造的に非後退。実機での実測（融合 vs 非融合の 5 回
  計測中央値・複合 WL 適用前後）は `docs/perf/metal-gemm-epilogue-fusion.md`
  を参照（本リポの開発環境が Linux のため実機検証未完・再現コマンドの
  記録に留まる）。

#### 2.2.1 学習経路への結線（イシュー #1044）

`gemm_bias_act`（上記）は §2.2 実装当初は GEMM epilogue の融合カーネルと
して用意されていたが、学習 forward（`fandhe_ai_facade::compat::
sequential::Sequential::forward`／`SequentialVars::forward`）からは
どこからも呼ばれておらず、`Linear` → `ReLU` は「`matmul`（`Op::MatMul`）
→ `add`（`Op::Add`。bias の broadcast leaf `[n]` は `run_fused` の
`ShapeMismatch` 対象外形状のため per-op フォールバック）→ `relu`
（別ノードの `Op::Relu`）」という 3 カーネル起動のままだった。イシュー
#1044 で以下のとおり結線した。

- **autodiff 側**: `fandhe_ai_autodiff::nn::linear::LinearVars::
  forward_with_activation(input, act)`（新設）が `Var::linear_act`
  （`var.rs`。`matmul`／`add` と同じ「①クロステープ検査 → ②shape 検査
  〈`matmul_out_shape`＋`broadcast_shape`〉→ ③forward 値計算 →
  ④ノード記録」の順で処理）経由で `tape.ops().gemm_bias_act` を直接
  呼び、`Op::LinearAct { input, weight, bias, act }`（新設・非
  elementwise・常時実体化）として 1 ノードで記録する。`Op::
  LinearResident`（デバイス常駐 weight。#1022）にも `act: Activation`
  フィールドを追加し、`DeviceParamStore::linear_forward_with_activation`
  が `BackendOps::gemm_resident_rhs_act`（新設。`tensor-core` の
  デフォルト実装は `gemm_resident_rhs` → `act == Relu` なら `self.relu`
  の非破壊合成。CPU（`backend-cpu::ops::CpuBackendOps`）はこれを
  `gemm_blis_bias_act_parallel` へ `act` を直接渡すカーネル内融合で
  オーバーライド済み）へ委譲する。
- **backward 側**: `Op::LinearAct`／`Op::LinearResident`（`act ==
  Relu`）の VJP（`grad.rs`）は、前活性化を再計算せず forward 記録値
  `out_value`（bias 加算・activation 適用後の値）から
  `out_value > 0` でマスクを復元する。既存 `Op::Relu` の VJP は
  `x > 0`（`x` は前活性化）でマスクするが、`relu(x) = max(x, 0)` の
  性質上 `x > 0 ⇔ relu(x) > 0` かつ非正 `x`・NaN 入力のいずれも
  両基準で同じ「マスク不成立（勾配 0）」に揃うため、両者は同一の
  劣勾配規約を表す（`grad.rs::vjp` の `Op::LinearResident`／
  `Op::LinearAct` 分岐コメント参照）。
- **層対検出は `Sequential` 側**: `fandhe_ai_autodiff::nn::module::
  Module::as_relu()`（新設フック。既定 `false`・`Relu` のみ `true`。
  `as_linear` と同じ明示列挙方式）を使い、`Sequential::forward`・
  `SequentialVars::forward`・`forward_from_flat_leaves`（resident 版）
  が `Linear` 層に出会うたびに「次層が `ReLU` か」を先読みする。
  **融合対象は `Linear` → `ReLU` の組に限定する**（レビュー指摘
  #1079・PRRT_kwDOTuUCJc6dgIt-。策定当初は「次層が `ReLU` でなくても
  `Activation::None` で bias のみ融合する」設計だったが、CPU の
  `gemm_bias_act`／`gemm_resident_rhs_act` は `bias.shape() == [n]`
  （通常経路）で融合カーネル `gemm_blis_bias_act_parallel` を直接呼ぶ
  ため、`docs/inference-forward-fixed-cost-design.md` §3.1 が
  `forward_host` について明記する bit-exactness 契約（「融合 epilogue
  はカーネル内 tiling 次第で加算順序が変わりうるため旧経路〈`gemm` →
  `add`〉と同じ演算列を使う」）を学習 forward 側で無条件に破ることに
  なっていた）。`ReLU` が続く場合のみ `Activation::Relu` を渡して
  `ReLU` 層自体のノード追加をスキップし、`ReLU` が続かない単体
  `Linear`・`Linear` → `Sigmoid`／`Tanh` は従来どおり非融合合成
  （`LinearVars::forward`。`matmul` → `add`）へ委譲する。ただし
  `forward_from_flat_leaves`（resident 版）が呼ぶ
  `DeviceParamStore::linear_forward_with_activation` は `Activation::
  None` でも `gemm_resident_rhs_act` → 内部で `gemm_resident_rhs` を
  呼ぶだけの非破壊拡張（CPU は `gemm_resident_rhs_impl` を同一引数で
  共有、`tensor-core` デフォルトも `gemm_resident_rhs` を呼んだ後
  `act == None` ならそのまま返す）であり `DeviceParamStore::
  linear_forward`（非 act 版）と数値的に同一のため、この経路は
  `Activation::None` のままでよい（数値経路の分岐は不要）。
  `Sigmoid`／`Tanh` は `Activation` に対応する variant を持たないため
  従来どおり別ノード（融合対象外）。
- **効果（層あたりカーネル起動数。CPU・host 常駐 forward）**:
  `matmul`＋`add`＋`relu` の 3 起動 → `gemm_bias_act` の 1 起動へ
  削減（デバイス常駐 forward は `gemm_resident_rhs`＋`relu` の 2 起動 →
  `gemm_resident_rhs_act` の 1 起動）。機械検証は
  `crates/facade/src/compat/sequential.rs` の
  `sequential_forward_fuses_linear_relu_into_single_gemm_bias_act_launch`
  （`CountingOps` による `BackendOps` 呼び出し回数の直接カウント）。
  勾配の bit 一致検証は同ファイルの
  `sequential_vars_forward_with_activation_grad_matches_manual_composition`
  と `crates/autodiff/src/optim/device_store.rs` の
  `linear_forward_with_activation_relu_grad_matches_manual_relu_composition`。
  CPU 実測は `docs/perf/train-linear-epilogue-fusion.md` を参照。
- **スコープ外（#1044 時点）**: CUDA／Metal は `gemm_resident_rhs_act`
  をオーバーライドせず、`tensor-core` のデフォルト（`gemm_resident_rhs`
  ＋ `relu` の 2 起動）にフォールバックする（値は正しいが 1 起動には
  ならない）。両バックエンドの常駐 GEMM カーネル入口
  （CUDA `launch_tiled_bias_act_f32_resident`・Metal
  `dispatch_strided_bias_act_prepared`）はいずれも `act`／`act_relu`
  を既に受け取れるため、後続イシューでの結線は型検査（`make
  build-cross`／`check-cross-metal-tests`）のみで済む見込み（実機
  必須の poison／generation 検査を含む既存コードへの変更は本イシュー
  では見送った）。backward 側の GEMM（`grad.rs::matmul_vjp` が
  `eval::matmul` をホスト直接計算している点）の `BackendOps` 経由化・
  転置ゼロコピーはイシュー #1046 の領域。`Module::forward_host`
  （推論の tape 不要経路。#1028 の bit-exact 契約）への
  `gemm_bias_act` 適用はイシュー #1218 で CPU 固定経路
  （`fandhe_ai_facade::compat::sequential::Sequential::predict`。
  `Linear::forward_host_with_activation`〈`nn/linear.rs`〉を新設し
  `Linear`→`ReLU` 限定で結線。`Module::forward_host` trait メソッド
  自体は汎用バックエンド向けの非融合のまま不変）を対応した。CPU
  融合カーネルが epilogue を GEMM 完了後に適用し非融合合成と bit
  完全一致することは `crates/backend-cpu/tests/gemm_epilogue_parity.rs`
  で確認済みのため、`forward_host` の bit-exactness 契約（trait レベル）
  とは矛盾しない別経路として追加した。CUDA／Metal の
  `forward_host` への同種結線は融合オーバーライドの bit 一致が
  未保証のため対象外のまま。実測は `docs/perf/
  cpu-infer-predict-profile.md`（#1218）を参照。

## 3. 限界

融合が働かない、または連鎖が分断されるパターンを列挙する。いずれも
`docs/fusion-graph-design.md` §1・§3.2 の実体化境界の設計根拠であり、
v1 PoC-9（Metal 実機、Burn/CubeCL 前提）の実測知見に基づく判断である
（下記「v1 PoC-9 実測の位置づけ」参照）。

| # | パターン | 挙動 | 根拠 |
|---|---------|------|------|
| 1 | reduction エピローグ（`.sum()`／`.max()`） | **IR・プラン検証レベル（Phase G・G-2 #586 以降）**: セグメント軸が一致する限り `Sum`／`Max` を融合セグメントへ組み込める（軸不一致は `FusionPlanError::MismatchedReductionAxis` で fail-closed 拒否）。**バックエンド実行レベル**: CPU の `run_fused_elementwise`（`crates/backend-cpu/src/fused_elementwise.rs:129`）は reduction を含むプランを明示的に未対応として拒否し、per-op フォールバックへ倒す。CUDA／Metal も本文書執筆時点で reduction 融合カーネル未実装（G-6 以降 #592・#594・#604・#607 で進行中）。したがって利用者が観測できる挙動は当初と同じ「連鎖部分のみ融合、reduction 自体は融合対象外」のまま（Phase G で解消予定）。**例外（イシュー #1045）**: MSE 損失（`Var::mse_loss_with` の forward・backward）は `run_fused`（汎用 IR）経由ではなく、専用 `BackendOps::mse_loss`／`mse_loss_backward`（`Op::MseLoss` の解析形専用ノード）として CPU／CUDA／Metal の 3 バックエンドとも融合カーネルで実装済み（`backend-cpu::mse`・`backend-cuda::mse`・`backend-metal::mse`）。`Sum`／`Max` 単独 API・汎用 reduction 融合（本行が指す `run_fused` 経由の対象）は引き続き未実装のまま。**追記（イシュー #1584）**: CUDA `BackendOps::sum`／`max`（`run_fused` を経由しない単独 API）は `reduce::CudaReduce`（`kernels_reduce.rs`）により実装済みとなった。Metal の単独 API は本文書執筆時点でも未実装。`run_fused` 経由の汎用 reduction 融合（本行の対象）自体は CUDA／Metal とも変更なく未実装のまま | IR: `docs/fusion-graph-design.md` §3.2 (a) 改定・`crates/tensor-core/src/fusion/graph.rs`（`FusionOp::Sum`／`Max`）。バックエンド未実装: `crates/backend-cpu/src/fused_elementwise.rs:129`。TASK-12.2b 策定時点の根拠: PoC-9 `ew_reduce`（連鎖部分は `ElemwiseFuse` に融合されるが `.sum()` は別カーネル群）。MSE 専用融合: イシュー #1045・`crates/tensor-core/src/backend_ops.rs`（`BackendOps::mse_loss`／`mse_loss_backward`） |
| 2 | matmul を挟む連鎖 | `gemm` ノードで融合セグメントが分断される（GEMM 本体は §2.2 の epilogue 融合対象だが、matmul をまたいだ elementwise 連鎖全体の融合ではない） | `docs/fusion-graph-design.md` §3.2 (b)。PoC-9 `ew_matmul_ew`（matmul 前後は個別カーネルのまま） |
| 3 | softmax 単体・attention 系連鎖 | **IR レベル（Phase G・G-3 #588 以降）**: softmax の構成要素（`x - max(x)`・`exp(..)`・`.. / sum(..)`）を表現する `Sub`／`Div`（G-3）・`Rsqrt`（G-2、RMSNorm 用）が `FusionOp` enum に追加され、行方向 reduction + broadcast セグメントとしてプラン表現・検証（`FusionPlanError`）まで実装済み。「`FusionOp` enum に含まれない」という当初の記述はもはや正確ではない。**バックエンド実行レベル**: CPU／CUDA／Metal のいずれも softmax・attention 系の融合カーネルは未実装（表 1 行目と同じギャップ。G-6 以降で進行中）であり、matmul（attention の QK^T・AV）を挟む連鎖は §3.2 (b)・表 2 行目のとおり分断されるため attention 全体の融合は引き続きスコープ外。利用者が観測できる挙動は当初と同じ「融合の影響をほぼ受けない」のまま（Phase G で解消予定） | IR: `docs/fusion-graph-design.md` §3.2 (b) 関連・`crates/tensor-core/src/fusion/graph.rs`（`FusionOp::Sub`／`Div`／`Rsqrt`）。REQ-12 受け入れ基準（`docs/spec/04-requirements.md:255`）。TASK-12.2b 策定時点の根拠: PoC-9 `softmax`・`attention_chain`（融合の影響をほぼ受けない、または部分的） |
| 4 | transpose 混在連鎖 | `NodeMeta.contiguous == false` を検出した時点で融合セグメントを打ち切り、非融合フォールバックへ倒す（初期スコープでは transpose を融合対象に含めない）。**実測（#167、5 回計測の中央値）**: 融合条件（`run_fused` を試みて `Unsupported` で失敗 → per-op フォールバック再実行の 2 段経路）と非融合条件（最初から per-op 経路のみ）の速度比中央値は 0.981x（ほぼ互角）。5 回のレンジは 0.865〜1.581x と QEMU 仮想 CPU のノイズで大きくばらつくため、フォールバック試行コストは実測上ノイズに埋もれる程度であり明確な悪化とは言えない（当初予測の「速度比 ≈ 1.0」を裏付ける結果。計測条件・5 回分の内訳は `docs/perf/cpu-elementwise-fusion-effect.md` §5 を参照）。**`Var::reshape`/`Var::transpose`（イシュー #1047）は上記と同じ理由で融合境界とする**: view 系ノードは `push_view`（中間バッファを持たない再計算方式。`tape::resolve_view` 参照）で登録され、elementwise 5 演算の融合連鎖（`build_lazy_plan` 等 3 走査器）には参加しない「葉」として扱われる。本行が既に確定させた非融合方針と整合するため、view 系ノード追加は融合適用範囲の後退ではない（`docs/autodiff-view-recompute-decision.md`）。**`Var::permute`／`Var::broadcast_to`（イシュー #1597）も同じ `push_view` 骨格で登録される同種の view 系ノードのため同じく融合境界**（`Var::squeeze`／`unsqueeze`／`flatten` は新規 `Op` を持たず `Var::reshape` へ委譲するため本行の対象に新規 `Op` 追加はない）。**`Var::narrow`（`split`／`split_with_sizes`／`chunk` の実体。イシュー #1598）も同じ `push_view` 骨格の view 系ノードのため同じく融合境界。`Var::cat`（同イシュー）が記録する `Op::Concat` はコピーを伴う非 elementwise ノードのため `push_eager` で登録し、`Op::LinearAct`／`Op::MatMul` 等と同じく融合対象外（`is_lazy_elementwise` の elementwise 5 演算に含めない）**。**`Var::where_cond`／`masked_fill`（イシュー #1637）が記録する `Op::Where`／`Op::MaskedFill` も `Op::Concat` と同じく `push_eager` の非 elementwise ノードで融合対象外** | `docs/fusion-graph-design.md` §1「transpose を挟む連鎖は融合しない」・§2.3。実測: `docs/perf/cpu-elementwise-fusion-effect.md` §5。view 系ノードの融合境界化: `docs/autodiff-view-recompute-decision.md`（#1047） |
| 5 | 連鎖長が 4〜6 段の上限を超える | 上限到達時点でその場の演算を実体化してから連鎖を再開する（実装済み・#404。codex-review PR #406 の P1 是正で fan-in 反例を修正）。`Tape::push_lazy`（`crates/autodiff/src/tape.rs`）が新規ノードの `lazy_chain_size`（未実体化入力の実効サイズの**総和** + 1。fan-in で合流する枝ごとのノード数を合算して捉える）を計算し、`tensor_core::MAX_FUSED_CHAIN_LEN`（= 6。`crates/tensor-core/src/fusion/detect.rs` を単一真実源とし遅延評価経路と結線済み）到達時に呼び出し元（`Var::add`/`mul`/`relu`/`exp`/`tanh`）がその場実体化する。以後このノードは実体化済み扱いとなり連鎖のカウントが自然にリセットされるため、`build_lazy_plan` が構築する `interior` の distinct ノード数は常に `lazy_chain_size`（したがって上限）未満で有界（`node_index` の線形探索コストも定数上限で抑えられる） | `docs/fusion-graph-design.md` §3.2 (d)・§3.5.4。実装は #404 |
| 6 | backward（VJP）の勾配式計算 | `grad.rs::vjp` が計算する勾配式そのもの（例: `tanh` の VJP `grad * (1 - y * y)`）は融合 IR に記録されず、常に具体 `Tensor` として直接計算される。融合されるのは forward が記録した elementwise 遅延グラフを `Tape::backward` が読み出す箇所のみ | `docs/fusion-graph-design.md` §3.3「backward（VJP）は融合対象外」 |

### v1 PoC-9 実測の位置づけ

`docs/spec/03-poc/poc-9-kernel-fusion/README.md` は Burn/CubeCL（`burn-
wgpu` の `fusion` feature）を前提とした v1 実測であり、**v2 自作融合
機構（本文書が扱う `tensor-core::fusion` モジュール）の保証値ではない**。
elementwise 連鎖・fan-out・transpose 混在パターンでの適用範囲判断
（表中 1〜5）は v1 実測が示した構造的な傾向（PoC-9「実施内容」節）を
参考に v2 の初期スコープを決定した根拠として引用しているが、v2 自作
機構での性能改善比・分断挙動は #167（TASK-12.2a）の実測を正とする。
v1 実測の代表数値（Metal 実機、`code/src/bin/elementwise_bench.rs`。
出典: 同 README「実行時間の計測」節）:

| パターン | 速度比（融合あり/なし） | 判定 |
|---|---|---|
| `ew4`（elementwise 4 段連鎖） | 2.25 倍 | 融合が有意に効く |
| `ew6`（elementwise 6 段連鎖） | 2.61 倍 | 融合が有意に効く |
| `ew_fanout`（fan-out を含む連鎖） | 3.19 倍 | 融合が有意に効く |
| `ew_reshape`（transpose 混在。v1 は融合有効時にメタデータ変換として取り込む） | 13.89 倍 | 融合が有意に効く（v1 実装依存。§3 表 4 参照） |
| `ew_matmul_ew`（matmul を挟む連鎖） | 0.82 倍（融合なしがわずかに高速） | 有意差なし |
| `softmax` 単体 | 0.94 倍（融合なしがわずかに高速） | 有意差なし |
| `attention_chain` | 0.98 倍（ほぼ差なし） | 有意差なし |

`ew_reshape` の 13.89 倍差は v1 の融合エンジン内部実装（ストライド付き
ビューを融合セグメント内で扱う機構）に依存した挙動であり、v2 自作融合
IR（`docs/fusion-graph-design.md` §2）は現時点でストライド付きビューを
表現・伝播する仕組みを持たないため、**v2 初期スコープはこの性能水準を
達成しない**という受け入れコストを明示的に記録する（`docs/
fusion-graph-design.md` §1・§6.2「transpose 混在連鎖のメタデータ融合」）。

## 4. 性能目標との関係（REQ-8 整合）

- REQ-12 受け入れ基準は「matmul・softmax を含む複合ワークロード
  （attention 系連鎖等）では融合の効果を前提とした性能目標を設定しない
  こと（PoC-9 で有意差なしと確認済み。REQ-8 の複合ワークロード系目標と
  整合させること）」と定める（`docs/spec/04-requirements.md:255`）。
- `docs/performance-targets.md` はこの整合を実装済みである: Transformer
  複合ワークロード（attention/softmax/LayerNorm を含む複合演算）の行
  （`docs/performance-targets.md:30`）は初期リリース・最適化後のいずれも
  **「下限を設定しない」**（融合の有無に依存させない）としており、
  QEMU 仮想 CPU での非実機参考値（約 6.1%）を性能目標の根拠に用いない
  ことを明記する。
- 融合を性能下限の前提とするのは §2 の elementwise 連鎖・GEMM epilogue
  という限定範囲のみであり、複合ワークロードの性能下限は
  `docs/performance-targets.md` を正とする（本文書はその根拠の一部を
  提供するのみで、下限値そのものを重複管理しない）。
- **Phase G（#582）との関係**: トラッキングイシュー #479 は Phase G
  （融合・正規化・量子化）を「GEMM 単体（REQ-8 の 5 行）への寄与は
  0% であり、REQ-8 の別行『Transformer 複合ワークロード』を対象とする
  別軸の Phase」と定義する。Phase G の G-4・G-12・G-14 等は実機
  ベースラインの確定を目的とし、`docs/performance-targets.md:30` の
  Transformer 複合ワークロード行を「下限を設定しない」から「目標を
  前提化する」方向へ変更するものではない。すなわち Phase G は §1 (b)
  の IR 表現力を拡張したが、§1 (c)・本節が定める「複合 WL では融合を
  性能目標の前提にしない」という方針自体は不変である（REQ-8 下限の
  変更は F-5・#577・人間承認事項に限定され、本文書の改定範囲外）。

## 5. 実測結果の参照

| 実測対象 | 記録 | 状態 |
|---|---|---|
| GEMM epilogue 融合（CPU、TASK-12.1f・#203） | `docs/perf/cpu-gemm-epilogue-fusion.md` | 実測済み（1.46〜2.56 倍、5 形状） |
| elementwise 連鎖・fan-out・transpose 混在の融合効果（v2、TASK-12.2a・#167） | `docs/perf/cpu-elementwise-fusion-effect.md` | 実測済み（5 回計測の中央値。連鎖・fan-out: 1.52〜1.57 倍改善。transpose 混在: ほぼ互角〈中央値 0.981x、レンジ 0.865〜1.581x〉） |
| v1 参考値（Burn/CubeCL 前提、Metal 実機） | `docs/spec/03-poc/poc-9-kernel-fusion/README.md` | 実測済み（v1 保証値であり v2 の保証値ではない。§3 参照） |

## 6. 将来拡張・スコープ外

- **reduction を含めた手動完全融合**: REQ-12 受け入れ基準は「性能クリ
  ティカルな箇所では、CubeCL カスタムカーネルによる手動融合（reduction
  を含めた完全融合）を組込み演算として提供する選択肢を将来検討課題と
  すること」と記載する（`docs/spec/04-requirements.md:254`。v1 CubeCL
  前提の文言だが、自作カーネルでの reduction epilogue 融合という論点
  自体は引き継ぐ）。本文書は §3 表 1 で reduction を実体化境界として
  扱う（初期スコープ外）に留め、将来対応は `docs/fusion-graph-
  design.md` §6.2「reduction エピローグの手動融合」への記録に留める。
- **backward（VJP）融合**: §3 表 6 のとおり初期スコープ外・未確定。
  VJP 計算式専用の融合グラフ構築・`FusedOpKind` への演算表現拡張
  （`sub` 等）を要する（`docs/fusion-graph-design.md` §6.2「backward
  （VJP）融合」）。
- **GPU バックエンドへの融合展開**: CUDA は §2.2 のとおりイシュー #599 で
  elementwise 5 演算・`gemm_bias_act` epilogue 融合を実装済み。Metal は
  引き続き GEMM カーネルのみ実装済みで elementwise 未実装のため、GPU
  側の融合はデフォルト実装（非融合合成）へのフォールバックに留まる
  （§2.2）。Metal 側の融合展開は実機（Metal 実機）での検証を要するため、
  ユーザー承認を得たうえで別イシューとして追跡する
  （`.claude/rules/out-of-scope-tracking.md`）。CUDA 側の `run_fused`
  （融合 IR 実行。§2.1）自体は本イシューのスコープ外のまま
  （`tensor_core::backend_ops::BackendOps::run_fused` デフォルト実装
  ＝`Unsupported` を継続使用）。
- **f16 対応**: `BackendOps`・`NodeMeta.dtype` とも現状 f32 固定であり、
  f16 融合カーネルの型設計は未着手（`docs/fusion-graph-design.md`
  §6.2「f16 対応」）。
- **transpose 混在連鎖のメタデータ融合**: §3「v1 PoC-9 実測の位置づけ」
  のとおり、ストライド付きビューを融合 IR 内で表現・伝播する設計
  ができれば transpose を融合対象へ含められる可能性がある（`docs/
  fusion-graph-design.md` §6.2「transpose 混在連鎖のメタデータ融合」）。
- **REQ-12 自体の v2 書き直し**: REQ-12 の受け入れ基準文言は `burn-
  wgpu` の `fusion` feature・`CUBECL_DEBUG_LOG` を前提としたまま（v2
  全面改定未実施）である。本文書は TASK-12.1〜12.2 の読み替え（自作
  elementwise 融合機構）に基づいて記述しているが、REQ-12 自体の文言
  更新は正本リポジトリ（`Fandhe-AI/fandhe-ai-spec`）側の課題で
  あり、本文書のスコープ外である（`docs/spec/` は編集しない）。

- **RNN／LSTM／GRU セル演算（イシュー #1647）**: `Op::RnnCell`／
  `Op::LstmCell`／`Op::LstmHidden`／`Op::GruCell`（設計 `docs/
  autodiff-rnn-cell-tape-design.md`）は `push_eager`（非 elementwise。
  `Op::is_lazy_elementwise` の `matches!` に未列挙のため常に `false`）
  で登録され、`reshape`／`transpose`（view 系。§3 表参照）と同じく
  融合対象外。GEMM 部分は既存の融合カーネル（`gemm_bias_act`）を
  再利用するが、ゲート pointwise 演算（`lstm_pointwise` 等）は独立の
  非融合カーネルとして実行する。

## 7. Phase G を受けた方針改定要否の判断（#591）

Phase G（#582）の G-2（#586）・G-3（#588）が融合 IR のスコープを
拡張したことを受け、本文書の方針記述の改定要否を次の 3 分岐で判断
した。

- (i) 本リポ `docs/kernel-fusion.md` の改定のみで足りる
- (ii) REQ-12 の改定提案として正本リポジトリ側へ回す（`docs/spec/`
  は編集せず、提案文を作成して回付する）
- (iii) 現行のまま維持する

**採用: (i)**。根拠:

- §1 判断サマリ (c)・§4 が定める「matmul・softmax を含む複合 WL では
  融合効果を前提とした**性能目標**を設定しない」（REQ-12 受け入れ
  基準 `docs/spec/04-requirements.md:255` 由来）は、Phase G 完了後も
  引き続き有効である。`docs/performance-targets.md:30` の Transformer
  複合ワークロード行は初期リリース・最適化後のいずれも「下限を
  設定しない」のまま変更されておらず、トラッキングイシュー #479 も
  Phase G を「GEMM 単体 REQ-8 比への寄与 0%・REQ-8 別行を対象とする
  別軸の Phase」と定義する（G-4・G-12・G-14 はベースライン確定で
  あって目標の前提化ではない）。REQ-8 下限の変更は F-5（#577・人間
  承認）に限定される
- したがって REQ-12 の受け入れ基準（性能目標の建て付け）自体は
  改定不要であり、(ii) は不要と判断する。乖離していたのは本リポ
  文書側の**実装状態の記述**（IR スコープ・限界表。§1 (b)・§3 表
  1・3 行目）のみであり、これは本リポの docs 改定（本コミット）で
  完結する
- REQ-12 の文言が v1（`burn-wgpu` `fusion` feature）前提のままである
  という既知の spec 側課題は §6「REQ-12 自体の v2 書き直し」に従来
  どおり残置し、本判断で新たに拡大しない（F-7・#580 への統合が必要な
  新規論点は生じていない）
- (iii) は §3 表 3 行目の「`FusionOp` enum に含まれない」が Phase G
  後は事実と反する記述になっていたため不採用とした

**確定した改定内容**: §1 判断サマリ (b) を「初期スコープ（TASK-12.2b
策定時点）」である旨を明示したうえで Phase G による IR 拡張を注記、
(c) は現行維持を明記、§3 限界表 1・3 行目を実装状態（IR 表現は
拡張済み・バックエンド実行は未追随）へ更新、§4 に Phase G の別軸
位置づけを追記した。`docs/performance-targets.md`・`crates/`・CI
設定・`docs/spec/` への変更は行っていない（docs-only 変更）。
