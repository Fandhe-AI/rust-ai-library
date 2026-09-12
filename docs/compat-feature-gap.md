> 調査日 2026-09-12・対象 HEAD `097bff19`（#1556 Metal resident grad staging
> 含む）。出典: 低レイヤー診断 artifact
> （https://claude.ai/code/artifact/4e107064-a190-4861-897d-3dce44d05428）
> §2「ギャップ根拠」節。イシューツリー起票（`p1-docs`）に伴い本ドキュメントへ
> 取り込む。内容は取り込み元から変更していない（下記「#1621 追記」節を除く）。
>
> **#1621 追記（線形代数）**: 本調査時点で §2.6「線形代数」に inv／solve／
> det／qr／cholesky／svd／matrix_norm の行は存在しなかった（`matmul`／`bmm`／
> `einsum`／`transpose` の 4 行のみ）。イシュー #1621 でこれらを実装した
> （`Var::inv`／`solve`／`det`／`cholesky`／`qr`／`svd`／`matrix_norm`。
> `docs/autodiff-linalg-design.md` 参照）。§0「Var の演算メソッドは 15 個の
> み」・§1.5 の一覧・§1.7 の `BackendOps` 演算 API 一覧・§2.6 の表へ、
> 実装結果を反映する追記を各該当箇所に加える（取り込み元の記述自体は残し、
> 追記であることを明示する）。

# fandhe-ai 公開面（facade）と PyTorch/TensorFlow の機能ギャップ表

調査日: 2026-09-12。対象は `crates/facade/src/`（`fandhe_ai` crate）から到達可能な公開 API のみ。
実装コードを読んで確定した事実のみを「あり」とする（ドキュメント上の意図・将来計画は「なし」扱い）。

## 0. 総括

fandhe-ai の公開面は現時点で **MLP（全結合＋3 活性化＋MSE/CrossEntropy 損失＋SGD/AdamW）専用**の
薄いラッパーである。テンソルの汎用演算（indexing・slice・cat・stack・任意 elementwise・
縮約の複数軸同時指定・`keepdim`・除算・べき乗等）は `Var`（autodiff 公開型）レベルにすら存在せず
（単一軸の `dim: Option<usize>` 指定〈`sum`/`max`〉自体はあるが複数軸同時指定・`keepdim` はない）、
`Tensor<T>`（tensor-core 型）レベルでも shape 変形（transpose/permute/reshape/broadcast/narrow）
止まりで、CNN・RNN・Attention を組むための演算プリミティブは構造的に欠落している。

- Var の演算メソッドは **15 個のみ**（`crates/autodiff/src/var.rs`）:
  `matmul`・`matmul_checksum`・`add`・`mul`・`sum`・`max`・`mse_loss`・`mse_loss_with`・
  `cross_entropy_loss`・`relu`・`exp`・`tanh`・`sigmoid`・`reshape`・`transpose`。
  減算（`sub`）・除算（`div`）・べき乗（`pow`）・平方根（`sqrt`）・対数（`log`）・
  比較演算・`softmax`（Var メソッドとしては存在しない）は **一切ない**。
  **#1621 追記**: 上記 15 個に加え、線形代数 7 個（`inv`・`solve`・`det`・
  `cholesky`・`qr`・`svd`・`matrix_norm`）を実装した（§1.5・§1.7・§2.6 参照）。
  減算・除算・べき乗等の欠落は本追記の対象外のまま不変。
- `nn::Module` 実装は `Linear`・`Relu`・`Sigmoid`・`Tanh` の **4 種のみ**
  （`crates/autodiff/src/nn/module.rs:105,171,193,212`）。Conv・BatchNorm・LayerNorm・
  RMSNorm・Dropout・Embedding・Attention・RNN/LSTM/GRU・Pooling は **一切ない**。
- 損失は `MseLoss`・`CrossEntropyLoss` の 2 種のみ（`crates/autodiff/src/nn/loss.rs`）。
- optimizer は `Sgd`（momentum・dampening・weight_decay・nesterov 対応）・`AdamW` の 2 種
  （`crates/facade/src/optim.rs:78-81`）。Adam（無 weight-decay 版）・RMSprop・Adagrad・
  LAMB は **ない**。
- scheduler は `ConstantLr`・`StepLr` の 2 種のみ（`crates/autodiff/src/nn/optim/lr_scheduler.rs`）。
  Cosine annealing・ExponentialLR・ReduceLROnPlateau・OneCycle は **ない**。
- dtype は `f32` 演算専用。`Tensor<T>` は `Element` trait 経由で `f32`/`f64`/`i32`/`f16`/`i64`/`bool`
  をジェネリックに保持できる（`crates/tensor-core/src/element.rs`）が、
  **算術カーネル dispatch（`BackendOps`）は `f32` 固定**（同ファイル冒頭コメント「`i64`/`bool` は
  算術・backend dispatch 対象には含めない」）。`facade` の `Var`/`Tensor<f32>` は事実上 f32 のみ。
- device は CPU／CUDA（単一 ordinal）／Metal の 3 種（`crates/tensor-core/src/device.rs`）。
  多 GPU 学習（DDP・model/tensor parallel）は **ない**。
- データローディング（`DataLoader`/`Dataset`）・`state_dict` 相当の汎用シリアライズ・
  ONNX export・量子化・JIT/`compile()` は facade に **ない**（ONNX の読み込みは
  `onnx-interop` にあるが非公開クレート）。
- メモリ管理 API（`release_cached_memory`・`memory_pool_stats`）・デバイス常駐パラメータ更新
  （`DeviceParamStore`）・CUDA Graph capture opt-in・TF32/split-K opt-in 等、
  **性能インフラ層の公開 API は PyTorch/TF より充実**している箇所もある（下記 1 節参照）。

---

## 1. fandhe-ai 公開面の実装済み機能一覧

### 1.1 composition root（`crates/facade/src/lib.rs`）

| 機能 | 場所 |
|------|------|
| `tape()`（既定 CPU バックエンドで `Tape` 構築） | `crates/facade/src/lib.rs:271-275` |
| `tape_for(Device)`（CPU/CUDA/Metal を明示選択） | `crates/facade/src/lib.rs:283-286` |
| `Device`・`BackendError`・`PoolStats`・`Tensor` 再エクスポート | `crates/facade/src/lib.rs:122` |
| `release_cached_memory(Device)`（REQ-14 明示解放） | `crates/facade/src/lib.rs:334-336` |
| `memory_pool_stats(Device)`（プール統計スナップショット） | `crates/facade/src/lib.rs:345-347` |
| `set_cuda_tf32_gemm_enabled`/`cuda_tf32_gemm_enabled`（CUDA TF32 opt-in） | `crates/facade/src/lib.rs:364-372` |
| `set_cuda_gemm_precision`/`cuda_gemm_precision`（`Fp32Strict`/`Tf32`/`Tf32x3`） | `crates/facade/src/lib.rs:471-479` |
| `set_cuda_graph_step_enabled`/`cuda_graph_step_enabled`（CUDA Graph step capture opt-in） | `crates/facade/src/lib.rs:400-408` |
| `cuda_graph_step_mode`/`cuda_graph_step_stats`（診断） | `crates/facade/src/lib.rs:426-437` |
| `set_cuda_managed_memory_enabled`/`cuda_managed_memory_enabled`（managed memory opt-in） | `crates/facade/src/lib.rs:518-526` |
| `set_metal_split_k_gemm_enabled`/`metal_split_k_gemm_enabled`（Metal split-K opt-out。macOS 限定） | `crates/facade/src/lib.rs:556-565` |
| `Tape::reset`/`leaf_count`/`leaf`（tape 再利用。学習ループ最適化） | `crates/facade/src/lib.rs:169-184` |
| `Tape::step_device_param_store`/`backward_device_param_store`/`sync_device_param_store_to_host`/`resident_grads_to_host`/`param_grads_to_host`（デバイス常駐更新） | `crates/facade/src/lib.rs:194-262` |

### 1.2 `compat::array`（numpy `np.array` 慣習）

`Tensor<f32>` を 1-D（`Vec<f32>`/`&[f32]`/`[f32; N]`）・2-D（`Vec<Vec<f32>>`/`[[f32; N]; M]`）から生成。
jagged 2-D 入力は事前検証で拒否。`crates/facade/src/compat/array.rs:98-101`。

### 1.3 `compat::Sequential`（Keras `Sequential` 慣習）

| 機能 | 場所 |
|------|------|
| `add_linear`/`add_relu`/`add_sigmoid`/`add_tanh` | `crates/facade/src/compat/sequential.rs:116-144` |
| `forward`（学習用。外部 `Tape` 上、Linear→ReLU 融合結線あり） | 同 152-194 |
| `predict`（推論。tape 不要経路→フォールバックで tape 経路） | 同 221-304 |
| `bind`/`SequentialVars`（学習可能パラメータのテープ登録） | 同 318-331, 648-785 |
| `trainable_parameters`/`apply_parameters`（shape 保存・2-pass アトミック更新） | 同 339-462 |
| `init_device_param_store`/`forward_resident`/`predict_resident`（デバイス常駐パラメータ学習・推論） | 同 477-640 |

### 1.4 `optim`（`crates/facade/src/optim.rs`）

| 機能 | 場所 |
|------|------|
| `Sgd`/`SgdConfig`（momentum・dampening・weight_decay・nesterov） | `crates/autodiff/src/optim/sgd.rs:32-224` |
| `AdamW`/`AdamWConfig` | `crates/autodiff/src/nn/optim/adamw.rs:23-160` |
| `clip_grad_norm`/`global_grad_norm`/`ClipGradResult` | `crates/autodiff/src/nn/optim/clip.rs` |
| `ConstantLr`/`StepLr`/`LrScheduler` | `crates/autodiff/src/nn/optim/lr_scheduler.rs` |

### 1.5 `Var` の演算メソッド一覧（`crates/autodiff/src/var.rs`）

| メソッド | 行 | 備考 |
|---------|-----|------|
| `value`/`to_tensor`/`host_view` | 98,107,135 | 借用ビュー読み出しは #1335 で追加 |
| `matmul` | 195 | GEMM。CPU BLIS／CUDA／Metal 各カーネルへ dispatch |
| `matmul_checksum` | 233 | デバイス側 f64 checksum 縮約（#1339） |
| `add`/`mul` | 339,362 | elementwise binary（融合対象） |
| `sum`/`max` | 382,402 | 縮約。`dim: Option<usize>` のみ（`amax` 均等分配なし。tie は最初の要素へ先勝ち） |
| `mse_loss`/`mse_loss_with` | 425,447 | mean/sum 縮約。単一融合カーネルあり |
| `cross_entropy_loss` | 508 | log-softmax 安定化込みの融合オペ |
| `relu`/`exp`/`tanh`/`sigmoid` | 556,568,580,600 | elementwise unary |
| `reshape`/`transpose` | 623,672 | view 系（#1080 で再計算方式・中間バッファなし） |
| `inv`/`solve`/`det`/`cholesky` | - | **#1621 追記**。線形代数（rank-2 限定）。CPU 実装先行・GPU は `Unsupported` フォールバック |
| `qr`/`svd` | - | **#1621 追記**。多出力（`QrVars`/`SvdVars`。テープは 1 ノード 1 出力のため出力ごとに別ノード） |
| `matrix_norm` | - | **#1621 追記**。`MatrixNormOrd`（`Fro`/`One`/`Inf`/`Nuc`/`Spectral`）指定 |

`Op` enum（`crates/autodiff/src/tape.rs:87-`）はこの Var メソッド集合と 1:1 対応する
（`Leaf`・`MatMul`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`Sum`・`Max`・`MseLoss`・
`CrossEntropyLoss`・`ResidentLeaf`・`LinearResident`・`LinearAct`・`Reshape`・`Transpose`・
`Inv`・`Solve`・`Det`・`Cholesky`・`QrQ`・`QrR`・`SvdU`・`SvdS`・`SvdVh`・`MatrixNorm`
〈**#1621 追記**〉）。

### 1.6 `Tensor<T>`（`crates/tensor-core/src/tensor.rs`）の shape 操作

`new`/`from_slice`/`zeros`/`ones`/`full`/`from_shape_fill`/`scalar`（生成）、
`shape`/`strides`/`offset`/`rank`/`numel`/`is_empty`/`get`/`as_slice`/`host_slice`/`as_view_slice`（アクセス）、
`transpose`/`transpose_2d`/`permute`/`narrow`/`is_contiguous`/`reshape`/`contiguous`/`broadcast_to`/`broadcast_with`（shape 変形）。
**indexing（花形インデックス）・`cat`/`stack`・`split`・`squeeze`/`unsqueeze`・`gather`/`scatter`・`expand`（`broadcast_to` はあるが `expand` API 名はない）は `Tensor<T>` レベルにも存在しない**。

### 1.7 `BackendOps` trait（`crates/tensor-core/src/backend_ops.rs`）が定義する演算 API 全量

`gemm`・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・`max`・`mse_loss`/`mse_loss_backward`・
`gemm_bias_act`（Linear+活性化 epilogue 融合）・`gemm_resident_rhs`/`_act`・`gemm_resident_lhs`・
`linear_forward_device`・`run_fused`（elementwise 融合実行）・`sgd_step_device`/`_tracked`・
`captured_segment_key`/`run_captured_sgd_step_segment`（CUDA Graph）・`gemm_checksum`・
`release_cached_device_memory`/`device_memory_pool_stats`・`linalg_inv`/`_solve`/`_det`/
`_cholesky`/`_qr`/`_svd`/`_matrix_norm`（**#1621 追記**。CPU 実装済み・CUDA／Metal は既定
`Unsupported` を明示オーバーライド）。**`sub`/`div`/`pow`/`sqrt`/`log`/
`sigmoid`（`BackendOps` に独立メソッドなし。`Op::Sigmoid` は `eval::sigmoid`〈`crates/autodiff/src/eval.rs:355-356`。数値安定形のホスト scalar 参照実装〉で計算し、GPU バックエンド選択時も `BackendOps` を経由しない）/`softmax`/`layer_norm`/
`conv`/`batch_norm`/`embedding`/`gather`/`scatter` はいずれも `BackendOps` に存在しない**。

なお `rmsnorm`・`softmax` の**行レベルカーネル実装自体**は CPU/CUDA/Metal 各バックエンドの
内部モジュール（`crates/backend-cpu/src/rmsnorm.rs`・`softmax.rs`、
`crates/backend-cuda/src/kernels_rmsnorm.rs`・`kernels_softmax.rs`、
`crates/backend-metal/src/rmsnorm.rs`・`softmax.rs`）に**存在する**が、`BackendOps` trait の
公開メソッドとして立っておらず、`Var`/`Tape`（autodiff）・`facade` のいずれからも到達できない
（ベンチ・parity テスト専用の内部実装。`crates/bench-harness/src/transformer_workload.rs` 等が
直接呼ぶのみ）。

### 1.8 dtype・device

- dtype: `Element` trait（`f32`/`f64`/`i32`/`half::f16`/`i64`/`bool`）は `Tensor<T>` の生成 API のみ対応
  （`crates/tensor-core/src/element.rs:26-84`）。算術は `f32` 固定。`half::f16` は GPU カーネル内部の
  中間表現としては使われる（CUDA/Metal の Tensor Core 経路）が、facade の公開型としては現れない。
- device: `Device::Cpu`/`Device::Cuda(ordinal)`/`Device::Metal`（`cfg(target_os = "macos")`）
  （`crates/tensor-core/src/device.rs`）。`to()` に相当する明示転送 API・複数デバイス間の
  自動フォールバック・デバイス列挙（`Device::available()`）は `docs/public-api-design.md` §4.1 で
  未決事項として明記され未実装（`crates/facade/src/lib.rs:75-78`）。

### 1.9 リポ内非公開（`onnx-interop`。crates.io 非公開・facade から到達不可）

ONNX opset の一部演算がホスト参照実装として存在する（`crates/onnx-interop/src/onnx/interp.rs:824-845`）:
`Gemm`・`Relu`・`Sigmoid`・`Shape`・`Gather`・`Unsqueeze`・`Concat`・`Slice`・`Add`・`Mul`・`Div`・`Mod`・
`Sqrt`・`Constant`・`Cast`・`Reshape`・`Squeeze`・`Transpose`・`MatMul`・`Softmax`・`Erf`・
`LayerNormalization`。**これらは autograd（`Tape`/`Var`）に接続されておらず推論専用のグラフ解釈器**
であり、`fandhe_ai`（facade）からは到達しない。`docs/compat-api-scope.md` の対象範囲外。

---

## 2. ギャップ表（大分類ごと）

凡例: 状態 = あり／部分／なし／リポ内非公開。難度 S=数時間〜1日, M=数日, L=1〜2週, XL=それ以上（複数バックエンド×新カーネル×VJP×parity 一式）。

### 2.1 テンソル生成

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `torch.tensor`/`from_numpy` | `tf.constant` | あり（`compat::array`・`Tensor::new`） | - | - |
| `torch.zeros`/`ones`/`full` | `tf.zeros`/`ones`/`fill` | あり（`Tensor::zeros`/`ones`/`full`。ただし `Var`/`compat` からは未再エクスポート＝`Tensor` 経由のみ） | compat 側の薄いラッパー追加 | S |
| `torch.arange`/`linspace` | `tf.range`/`linspace` | なし | `Tensor` 生成関数 1 個追加 | S |
| `torch.randn`/`rand`（乱数テンソル） | `tf.random.normal` 等 | なし（`Linear::new` 内部の重み初期化にシードベース乱数はあるが公開 API なし） | 汎用乱数テンソル生成 API（RNG 契約含む） | S〜M |
| `torch.eye` | `tf.eye` | なし | 生成関数 1 個 | S |

### 2.2 index/slice/gather/scatter

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| 基本スライス `x[a:b]` | `tf.slice`/`x[a:b]` | なし（`Tensor::narrow` は単一 dim 開始/長さのみ。`Var` レベルでは全くなし） | `Var::narrow`（VJP: 逆方向は zero-pad scatter）＋3 バックエンドカーネル or ホスト実装＋parity | M |
| 花形インデックス `x[idx]` | `tf.gather`（`gather_nd`） | なし | 新 Op（`Gather`）＋forward/backward＋3 バックエンド | L |
| `scatter`/`scatter_add`/`index_put_` | `tf.tensor_scatter_nd_*` | なし | 新 Op（`Scatter`）＋VJP＋3 バックエンド | L |
| `masked_select`/`where` | `tf.where` | なし | 新 Op（条件付き選択）＋VJP | M |
| `torch.topk`/`sort` | `tf.math.top_k`/`sort` | なし | 縮約系の拡張・非連続勾配経路の設計 | L |

### 2.3 形状操作

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `reshape`/`view` | `tf.reshape` | あり（`Var::reshape`。`Tensor::reshape`） | - | - |
| `permute`/`transpose` | `tf.transpose` | あり（`Var::transpose` は 2 軸限定 swap のみ・`Tensor::permute` は任意順だが `Var` に未接続） | `Var::permute`（VJP は逆置換）＋既存 `Tensor::permute` への接続 | S〜M |
| `cat`/`stack` | `tf.concat`/`tf.stack` | なし | 新 Op（`Concat`）＋backward（勾配を各入力へ narrow で分配）＋3 バックエンド | M〜L |
| `split`/`chunk` | `tf.split` | なし | 新 Op（`Split`）＋backward（勾配を `Concat`）＋3 バックエンド | M |
| `expand`/`broadcast_to` | `tf.broadcast_to` | 部分（`Tensor::broadcast_to`/`broadcast_with` はあるが明示 `expand`/`Var::broadcast_to` API はない。**`Var::add`/`mul` は暗黙ブロードキャストに対応済み（確定）**: `BackendOps::add`/`mul` の shape 検査 `elementwise_out_shape`〈`crates/tensor-core/src/ops_shape.rs:82-86`〉が `broadcast_shape` へ委譲しており NumPy 慣習のブロードキャストを行う） | 明示 `Var::broadcast_to`/`expand` API の追加（VJP の縮約方向勾配はブロードキャスト対応 add/mul で既に検証済みのロジックを流用可能） | S〜M |
| `squeeze`/`unsqueeze` | `tf.squeeze`/`expand_dims` | なし（`Tensor` レベルにも直接の API はない。`reshape` で代用可） | 薄いラッパー | S |

### 2.4 要素演算

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `+`/`add` | 同左 | あり（`Var::add`） | - | - |
| `*`/`mul` | 同左 | あり（`Var::mul`） | - | - |
| `-`/`sub` | 同左 | **なし** | 新 Op（`Sub`）＋VJP（片方 `-upstream`）＋`BackendOps::sub` を CPU/CUDA/Metal へ追加＋parity | M |
| `/`/`div` | 同左 | **なし** | 新 Op（`Div`）＋VJP（商の微分）＋3 バックエンドカーネル＋parity | M |
| 比較演算（`>`,`==` 等） | 同左 | なし | bool 出力の新 Op 群（非連続勾配のため VJP はゼロ扱い） | M |
| `sin`/`cos`/`tan` | 同左 | なし | elementwise unary Op×3＋VJP＋3 バックエンド | M（各） |
| `pow`/`sqrt` | 同左 | なし | elementwise unary/binary Op＋VJP＋3 バックエンド | M |
| `log`/`log2`/`log10` | 同左 | なし（`cross_entropy_loss` 内部にのみ log-sum-exp あり。汎用 log は非公開） | elementwise unary Op＋VJP（`1/x`）＋3 バックエンド | M |
| `sigmoid` | `tf.sigmoid` | あり（`Var::sigmoid`。`eval::sigmoid`〈ホスト scalar 参照実装〉で計算し `BackendOps` を経由しない。VJP は `out_value` 再利用方式で既存） | 高速化するならバックエンド専用カーネル追加＋`BackendOps::sigmoid` 新設 | S〜M（現状で機能は十分・性能改善のみ） |
| `gelu`/`silu`(swish) | 同左 | **なし**（`docs/compat-api-scope.md` が GELU を明示的スコープ外と記載） | elementwise unary Op＋VJP＋3 バックエンド。Transformer 必須 | M |
| `softmax`/`log_softmax` | 同左 | **なし（Var メソッドとしては無い）**。`cross_entropy_loss` 内部に log-softmax の融合実装があるのみ（`grad.rs`）。行カーネル自体は CPU/CUDA/Metal に既存（`softmax.rs` 系）だが `BackendOps` 未接続 | `BackendOps::softmax` を 3 バックエンドの既存行カーネルへ接続＋新 Op＋VJP（Jacobian-vector 積） | M（カーネルは既にあるため配線中心） |

### 2.5 縮約

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `sum(dim)` | `tf.reduce_sum` | あり（`Var::sum`。単一 dim または全体のみ、複数軸指定不可） | 複数軸・`keepdim` 対応への拡張 | S〜M |
| `mean(dim)` | `tf.reduce_mean` | なし（`mse_loss` 内部にのみ mean 縮約あり。汎用 `Var::mean` はない） | `sum` を分母で割るラッパー、または専用 Op | S |
| `max(dim)`/`min(dim)` | `tf.reduce_max`/`min` | 部分（`Var::max` あり・`min` なし・`argmax`/`argmin` なし・`amax`/`amin` の均等分配は明示的スコープ外） | `min`/`argmax`/`argmin` の新規 Op | M |
| `var`/`std` | `tf.math.reduce_variance`/`reduce_std` | なし（rmsnorm の内部計算にのみ二乗和はある） | 縮約 Op（二乗和ベース。f64/scale-ssq アキュムレータ契約に従う） | M |
| `norm`（L1/L2） | `tf.norm` | 部分（`clip_grad_norm`/`global_grad_norm` に L2 ノルム計算はあるが `Var` 演算としては非公開） | `Var::norm` の新設 | S〜M |

### 2.6 線形代数

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `matmul`（2D） | `tf.matmul` | あり（`Var::matmul`。CPU BLIS／CUDA Tensor Core／Metal simdgroup 全対応・TF32/split-K opt-in 込み） | - | - |
| `bmm`（バッチ行列積） | `tf.linalg.matmul`（バッチ次元対応） | **なし（確定）**。`crates/tensor-core/src/ops_shape.rs:43-` `matmul_out_shape` が `lhs.len() != 2`/`rhs.len() != 2` を `ShapeError::RankMismatch` で拒否し、`Var::matmul`（`var.rs:195-208`）はこれを経由するため rank 2 のみ受理する | バッチ次元対応の GEMM 拡張（3 バックエンドのループ or バッチ化カーネル） | L |
| `einsum` | `tf.einsum` | なし | 汎用縮約記法の解釈器＋既存 GEMM/縮約への分解実装 | XL |
| `transpose`（線形代数用） | 同左 | あり（2.3 節参照） | - | - |
| `torch.linalg.inv` | `tf.linalg.inv` | **あり（#1621）**。`Var::inv`（rank-2 正方限定。CPU 実装・GPU は `Unsupported`） | - | - |
| `torch.linalg.solve` | `tf.linalg.solve` | **あり（#1621）**。`Var::solve` | - | - |
| `torch.linalg.det` | `tf.linalg.det` | **あり（#1621）**。`Var::det`（特異行列は `0.0`。エラーにしない） | - | - |
| `torch.linalg.cholesky` | `tf.linalg.cholesky` | **あり（#1621）**。`Var::cholesky`（下三角のみ・`upper=True` 相当は対象外） | - | - |
| `torch.linalg.qr` | `tf.linalg.qr` | **あり（#1621）**。`Var::qr`（reduced QR のみ・`m<n` backward は対象外） | - | - |
| `torch.linalg.svd` | `tf.linalg.svd` | **あり（#1621）**。`Var::svd`（reduced SVD のみ・相異なる特異値前提の backward） | - | - |
| `torch.linalg.matrix_norm` | `tf.norm` | **あり（#1621）**。`Var::matrix_norm`（`MatrixNormOrd`: Fro/One/Inf/Nuc/Spectral） | - | - |
| `torch.linalg.eigh`/`lstsq`/`pinv`/`matrix_rank`/`slogdet` | 相当 API | なし（#1621 スコープ外） | 各分解アルゴリズムの追加実装 | M〜L |

### 2.7 NN 層

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `nn.Linear` | `layers.Dense` | あり（`Linear`。bias 有無・epilogue 融合済み） | - | - |
| `nn.Conv1d`/`Conv2d` | `layers.Conv1D`/`Conv2D` | **なし** | im2col か直接畳み込みカーネル（CPU/CUDA/Metal）＋VJP（d_input は転置畳み込み・d_weight は相関）＋parity。GEMM 基盤を再利用可能だが新カーネル必須 | XL |
| `nn.BatchNorm2d` | `layers.BatchNormalization` | **なし**（rmsnorm はあるが batchnorm は統計対象軸・running stats が異なる） | 新 Op（バッチ統計・running mean/var の状態保持）＋3 バックエンド | L |
| `nn.LayerNorm` | `layers.LayerNormalization` | **なし**（`BackendOps` に layer_norm メソッドなし。onnx-interop にはホスト実装あり・非公開） | `BackendOps::layer_norm` 新設＋VJP＋3 バックエンド（rmsnorm の実装パターンを流用可能） | M〜L |
| RMSNorm | （TF に相当レイヤーなし。カスタム実装が一般的） | リポ内非公開（`backend-{cpu,cuda,metal}::rmsnorm` に行カーネルあり・`BackendOps`/`Var` 未接続） | `BackendOps::rmsnorm` 新設・`Var`/`nn::RmsNorm` 配線 | M（カーネルは既存） |
| `nn.Dropout` | `layers.Dropout` | **なし** | RNG 契約設計＋マスク適用 Op（train/eval モード分岐）＋VJP | M |
| `nn.Embedding` | `layers.Embedding` | **なし** | gather 系 Op が前提（2.2 節）＋embedding テーブル管理 | L |
| `nn.MultiheadAttention` | `layers.MultiHeadAttention` | **なし** | softmax・batched matmul・(optional) causal mask・reshape/transpose の組合せ実装。前提演算が軒並み未実装 | XL |
| RNN/LSTM/GRU | `layers.SimpleRNN`/`LSTM`/`GRU` | 内部クレート `fandhe_ai_autodiff::nn::rnn`（`RnnCell`/`LstmCell`/`GruCell`・`Rnn`/`Lstm`/`Gru`）に実装済み（3 バックエンド〈CPU・CUDA・Metal〉数値一致。Metal は実機実測完了・CUDA は本エージェント実行環境に実機なしのため未実測明記）。**facade（`fandhe_ai`）未公開**（`docs/compat-api-scope.md` §5 の範囲拡張手続きのうちユーザー承認が未取得のため。決定 10）。`forward_seq`（tape 経路）の出力は `Var::stack`〈#1598〉未実装のため `[T,B,H]` ではなく `Vec<Var>`（per-step）。設計: `docs/autodiff-rnn-cell-tape-design.md`（#1646）・実装記録: 同文書 §8（#1647） | XL（設計・内部実装は完了。facade 公開のみ残作業） |
| Pooling（Max/AvgPool） | `layers.MaxPooling2D` 等 | **なし** | Conv 同様の空間走査カーネル＋VJP（max は argmax 経路の逆伝播） | L |

### 2.8 損失

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `MSELoss` | `losses.MeanSquaredError` | あり（`Var::mse_loss`/`mse_loss_with`。mean/sum 縮約・融合カーネル） | - | - |
| `CrossEntropyLoss` | `losses.SparseCategoricalCrossentropy` | あり（`Var::cross_entropy_loss`。log-softmax 融合） | - | - |
| `BCELoss`/`BCEWithLogitsLoss` | `losses.BinaryCrossentropy` | なし | 新融合 Op（MSE/CrossEntropy と同じ設計パターン） | M |
| `NLLLoss` | - | なし（`cross_entropy_loss` が事実上兼ねる設計） | 既存 CrossEntropy から log 済み入力を受け付ける版を分離するかは要判断 | S〜M |
| `HuberLoss`/`SmoothL1Loss` | `losses.Huber` | なし | 新融合 Op（区分的関数の VJP） | M |

### 2.9 optimizer

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `SGD`（momentum/nesterov/weight_decay） | `optimizers.SGD` | あり（`optim::Sgd`。デバイス常駐版 `DeviceParamStore::step` も対応） | - | - |
| `AdamW` | `optimizers.AdamW` | あり（`optim::AdamW`） | - | - |
| `Adam`（coupled L2 weight decay） | `optimizers.Adam` | 部分（decoupled 版 `AdamW` はあり、`weight_decay=0` のときは `Adam` と完全に一致する。`weight_decay>0` の coupled L2 版〈PyTorch `Adam(weight_decay>0)`〉はなし） | `AdamW` の decay 適用箇所（勾配へ加算 vs パラメータへ直接減算）を分岐する薄い派生 | S |
| `RMSprop` | `optimizers.RMSprop` | なし | 新 optimizer 型（値型・純関数。`Sgd`/`AdamW` と同型） | S〜M |
| `Adagrad` | `optimizers.Adagrad` | なし | 同上 | S〜M |
| LAMB | - | なし | 同上（layer-wise trust ratio の追加） | M |

### 2.10 scheduler

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `StepLR` | `LearningRateSchedule`（`ExponentialDecay` 等） | あり（`StepLr`） | - | - |
| 定数 LR | 同左 | あり（`ConstantLr`） | - | - |
| `CosineAnnealingLR` | `CosineDecay` | なし | 新 `LrScheduler` 実装（純関数） | S |
| `ExponentialLR` | `ExponentialDecay` | なし | 同上 | S |
| `ReduceLROnPlateau` | `ReduceLROnPlateau`（callback） | なし | 状態保持（履歴・patience）を持つ scheduler 型 | M |
| `OneCycleLR` | - | なし | 同上 | M |

### 2.11 autograd

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| 動的テープ・`backward()` | `tf.GradientTape` | あり（`Tape::backward`。動的テープ方式。`Tape::reset` で再利用可能） | - | - |
| `no_grad()`/`torch.inference_mode()` | `tf.stop_gradient` | なし（テープに載せない選択肢は「別 `Tape` を使わない」設計自体にない。`tape_free` 推論経路〈`Sequential::predict`〉はあるが汎用 `no_grad` コンテキストではない） | 葉ノード登録をスキップする API、または `Var` を「追跡なし」でラップする型 | M |
| `detach()` | `tf.stop_gradient` | なし | 既存 `Var` から新規葉ノードへ変換する Op | S〜M |
| `retain_graph=True` | - | 該当なし（`Tape::backward` はグラフノード〈`nodes`〉を破棄せず、呼び出しごとに独立した `Gradients` を新規生成するのみで、グラフ保持は既定動作。`retain_graph` フラグ自体が不要な設計。ただし PyTorch の `.grad` 蓄積〈複数回 `backward()` の勾配加算〉に相当する契約は無く、同一 loss に対する複数回 `backward()` の勾配蓄積セマンティクスは未検証） | 設計判断が必要（複数回 backward の勾配蓄積契約） | M |
| 高階微分（`grad of grad`） | `tf.GradientTape` のネスト | なし（テープは 1 階のみを前提とした構造と推定） | Op 自体を微分可能にする再設計（VJP の VJP）。設計: `docs/autodiff-higher-order-grad-decision.md`（#1622） | XL |
| custom `autograd.Function` | `tf.custom_gradient` | なし（`Op` enum は crate 非公開の固定 variant 集合。ユーザー定義 Op を挿す口がない） | 拡張可能な Op プラグイン機構の設計（現行のクローズドな `Op` enum 設計を変更）。設計: `docs/autodiff-custom-function-decision.md`（#1623） | XL |
| `torch.utils.checkpoint`（activation checkpointing） | `tf.recompute_grad` | 部分（view 系ノード〈reshape/transpose〉は #1080 で再計算方式化済みだが、任意サブグラフの再計算チェックポイントではない） | 汎用チェックポイント機構の設計 | L |

### 2.12 dtype

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `float32` | `float32` | あり（唯一の演算 dtype） | - | - |
| `float64` | `float64` | 部分（`Tensor<f64>` は生成できるが `BackendOps`/`Var` の算術対象外） | `Element` 抽象を活かした dtype 別 dispatch の設計・全カーネルの多重化 | XL |
| `float16`/`bfloat16` | 同左 | 部分（GPU カーネル内部の中間表現・Tensor Core 経路にのみ存在。公開 dtype ではない） | 公開 `Tensor<f16>` 演算経路・VJP のスケーリング契約設計（mixed precision） | XL |
| `int32`/`int64`/`bool` | 同左 | 部分（`Tensor<T>` 生成のみ。CrossEntropy の `targets: Tensor<i32>` のように限定的に内部使用） | 汎用整数演算・型変換 API | L |
| `.to(dtype)`（型変換） | `tf.cast` | なし | dtype 変換 Op（勾配は型により打ち切り／恒等など個別設計） | M |
| AMP（自動混合精度） | `tf.keras.mixed_precision` | なし（`optim.rs` doc に「損失スケーリング（AMP）は現時点で未実装」と明記） | 損失スケーリング・unscale ステップの追加（`optim.rs` の適用順序契約に定義済みの拡張点） | L |

### 2.13 device

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| 単一 GPU 選択（`device='cuda:0'`） | `tf.device` | あり（`Device::Cuda(ordinal)`） | - | - |
| `.to(device)`（テンソル転送） | `tf.identity` with device | なし（`tape_for` でバックエンドごと `Tape` を切替える設計。テンソル単体を明示転送する API はない） | `Tensor`/`Var` のデバイス間コピー API | M |
| 複数 GPU・`DataParallel`/`DDP` | `tf.distribute.MirroredStrategy` | なし | 勾配 all-reduce・パラメータ複製の設計（ネットワーク層から必要） | XL |
| デバイス自動列挙（`torch.cuda.device_count()`） | `tf.config.list_physical_devices` | なし（`docs/public-api-design.md` §4.1 未決事項として明記） | `Device::available()` 相当の列挙 API | S〜M |

### 2.14 データ

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `Dataset`/`DataLoader`（バッチ化・シャッフル） | `tf.data.Dataset` | なし | イテレータ・シャッフル・バッチ化の薄い層（既存 `Tensor` の上に構築可能） | M |

### 2.15 保存・相互運用

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `state_dict()`/`load_state_dict()` | `model.save_weights` | 部分（`Sequential::trainable_parameters`/`apply_parameters` で手動シリアライズは組める。汎用 `state_dict` 相当の名前付き辞書 API はない） | 層名→テンソルのマップ API | S〜M |
| safetensors 読み書き | - | リポ内非公開（`onnx-interop::st_load`/`st_save`。facade 未接続） | facade からの再エクスポート、または `Sequential` の save/load ラッパー | M |
| ONNX export | `tf2onnx` 等 | なし（`onnx-interop` は import 方向のみ・かつ非公開） | export 側の実装＋facade 公開判断 | XL |
| ONNX import | `torch.onnx`（逆方向） | リポ内非公開（`onnx-interop::onnx::interp`。autograd 未接続の推論専用グラフ解釈器） | facade への公開判断＋（学習させるなら）`Tape` への変換層 | L（公開のみなら）〜XL（学習可能化） |

### 2.16 推論・その他

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `torch.compile`（グラフ最適化 JIT） | `tf.function`（AutoGraph・XLA） | 部分（`run_fused`＝elementwise カーネル融合・CUDA Graph capture opt-in はあるが、汎用グラフ JIT コンパイラではない） | 既存インフラの延長線上で拡張可能（新規 JIT は不要） | - |
| 量子化（int8 等） | `tf.lite` 量子化 | なし | 量子化 dtype・演算対応（2.12 節の dtype 拡張が前提） | XL |
| 乱数シード固定（`manual_seed`） | `tf.random.set_seed` | 部分（`Linear::new(.., seed: u64)` など個別 API にシード引数はあるが、グローバル RNG 状態を握る `manual_seed` 相当はない） | グローバル RNG 契約の設計（Dropout 等 今後追加する確率的演算との整合が前提） | M |

---

## 3. 「MLP → CNN → Transformer」に必要な最小集合（Tier 1）と長尾（Tier 2）

### Tier 1（MLP はほぼ揃っている。CNN・Transformer に必須の欠落）

| 機能 | 現状 | compat-api-scope.md の位置づけ |
|------|------|-------------------------------|
| **MLP** | ほぼ揃っている（Linear・ReLU/Sigmoid/Tanh・MSE/CrossEntropy・SGD/AdamW） | 対象範囲内（1 節） |
| `Var::sub`/`div`/`pow`/`sqrt` | なし | 記載なし（対象範囲・対象外いずれにも明記なし＝5 節の範囲拡張手続きが必要な未定義事項） |
| `softmax`（Var/nn 単体） | なし（CrossEntropy 内部にのみ融合実装） | **明示的に対象外**（`docs/compat-api-scope.md:194` 「CrossEntropy と密結合のため対象外」）。単体 softmax の公開には spec 側（対象範囲）の改定が必要 |
| GELU/SiLU | なし | **明示的に対象外**（同 :195「必要になった時点で後続イシューに切り出す」） |
| LayerNorm | なし（`BackendOps` 未接続） | 記載なし（Keras の「全レイヤー種別の網羅を対象外」に該当し得るため spec 側確認が必要） |
| Conv1d/Conv2d | なし | **明示的に対象外**（同 :192「Conv 系」） |
| MultiheadAttention | なし（softmax・batched matmul・masking いずれも未実装） | 記載なし（Keras レイヤー網羅の対象外に準ずると推定） |
| cat/stack/split（Transformer の head 分割・結合に必須） | なし | 記載なし |
| gather（Embedding の前提） | なし | 記載なし |
| Embedding | なし | **明示的に対象外**（同 :192「Embedding 等」） |
| Dropout | なし | 記載なし（Keras レイヤー網羅の対象外に準ずると推定） |
| bmm（batched matmul。Attention の QK^T に必須） | **なし（確定。`matmul_out_shape` が rank 2 以外を拒否）** | 記載なし |

**結論**: MLP は Tier 1 として成立している。CNN（Conv・Pooling・BatchNorm）・Transformer
（softmax・LayerNorm・Embedding・MultiheadAttention・cat/split・bmm）に進むには、
上表の欠落項目のうち **softmax・GELU・Conv・Embedding の 4 項目は `docs/compat-api-scope.md`
2 節が明示的にスコープ外と定めており、対象範囲を広げるには 5 節の手続き（正本 spec 側の
REQ-9 受け入れ基準改定、またはユーザー承認を得たうえでの本文書更新）を経る必要がある**。
LayerNorm・MultiheadAttention・Dropout・cat/stack・bmm・gather は現行の compat-api-scope.md に
明記がなく、範囲判断自体を要する。

### Tier 2（長尾。当面の MLP/CNN/Transformer 到達には不要）

RNN/LSTM/GRU・Pooling 各種・BatchNorm・einsum・高階微分・custom autograd Function・
AMP・量子化・DDP・多 GPU・DataLoader/Dataset・ONNX export・state_dict 汎用シリアライズ・
花形インデックス・scatter 系・比較演算・三角関数・var/std・cosine/exponential/plateau
scheduler・RMSprop/Adagrad/LAMB。

---

## 4. 既存 Op 追加のパターン（実例トレース）

**題材**: `BackendOps::mse_loss`/`mse_loss_backward` の融合実装追加（イシュー #1045、コミット
`1e13b773`「perf(autodiff): MSE loss の reduction を単一カーネルへ融合する (#1078)」）。
既存 `Op::MseLoss` 自体はホスト参照実装（`eval::mse_loss`）としてそれ以前から存在しており、
本コミットは「ホスト参照実装 → 3 バックエンド融合カーネル」への昇格パターンを示す好例
（新規 Op の追加も基本的に同じファイル群に触れる）。

| 層 | 触ったファイル | 内容 |
|----|----------------|------|
| tensor-core（trait 定義） | `crates/tensor-core/src/backend_ops.rs`（+107 行） | `BackendOps::mse_loss`/`mse_loss_backward` のデフォルト実装（`Unsupported`）と `MseReduction` enum を追加 |
| tensor-core（再エクスポート） | `crates/tensor-core/src/lib.rs` | 新規型の公開 |
| autodiff（Var/Op/VJP） | `crates/autodiff/src/var.rs`（+60）、`crates/autodiff/src/grad.rs`（+58） | `Var::mse_loss_with` が `BackendOps::mse_loss` を優先し `Unsupported` のみ `eval::mse_loss` へフォールバック。VJP も同様に `mse_loss_backward` 優先＋フォールバック |
| autodiff（テスト） | `crates/autodiff/tests/mse_loss_fusion.rs`（新規 385 行） | 融合 forward/backward の数値・フォールバック契約テスト |
| backend-cpu | `crates/backend-cpu/src/mse.rs`（新規 188 行）・`ops.rs`（+72）・`lib.rs`（+1） | CPU カーネル実装・`BackendOps` 実装への配線 |
| backend-cpu（parity） | `crates/backend-cpu/tests/mse_parity.rs`（新規 139 行） | 数値一致テスト |
| backend-cuda | `crates/backend-cuda/src/kernels_mse.rs`（新規 254 行）・`mse.rs`（新規 229 行）・`ops.rs`（+131）・`context_cache.rs`（+13）・`lib.rs`（+3） | NVRTC カーネル・コンテキストキャッシュ・`BackendOps` 配線 |
| backend-cuda（parity） | `crates/backend-cuda/tests/mse_parity.rs`（新規 145 行） | 数値一致テスト（`#[ignore]` 実機依存） |
| backend-metal | `crates/backend-metal/src/mse.rs`（新規 326 行）・`shaders/mse.metal`（新規 163 行）・`ops.rs`（+82）・`context_cache.rs`（+9）・`lib.rs`（+4） | MSL カーネル・`BackendOps` 配線 |
| backend-metal（parity・証跡） | `crates/backend-metal/tests/mse_parity.rs`（新規 111 行）・`mse_source_evidence.rs`（新規 104 行） | 数値一致テスト・融合カーネルが実際に呼ばれることのソース走査証跡 |
| ドキュメント | `docs/kernel-fusion.md`（+2） | 汎用 reduction 融合の限界注記を更新 |

**合計 23 ファイル・約 2569 行追加**（`1e13b773` の diffstat）。新規 elementwise Op
（例: `sub`/`div`/`gelu`）を素朴に追加する場合はこれよりやや小さく（VJP が単純・
struct variant 化不要）、Conv や Attention のような新規演算カテゴリを追加する場合は
これより大きくなる（新規 shader/kernel 設計・タイル戦略・reuse/fresh 両モード対応が必要）。

**再利用可能なテンプレート（この 1 コミットから読み取れる型）**:
1. `tensor-core::BackendOps` に新メソッド（デフォルト実装 `Unsupported`）を追加
2. `autodiff::Var` に新メソッド、`autodiff::grad::vjp` に対応 VJP 分岐を追加
   （`BackendOps` 優先・`Unsupported` のみホスト実装へフォールバックする二段構え）
3. `backend-cpu`/`backend-cuda`/`backend-metal` それぞれで実装・`ops.rs` へ配線
4. 各バックエンドに parity テスト（数値一致複合判定）・Metal は「実際にそのカーネルが
   呼ばれる」ことを保証する source-evidence テストを追加
5. `docs/` の関連設計ドキュメントを更新

## 追補（2026-09-12・イシュー #1594）

§2.4 の「`softmax`/`log_softmax`」行（上記表）は取り込み元スナップショット
（調査日 2026-09-12 時点）のギャップ記述のため変更していないが、同イシューで
このギャップは解消済みである: `tensor-core::BackendOps::softmax`／
`log_softmax`（デフォルト `Unsupported`）を新設し、`autodiff::tape::Op::
Softmax`／`LogSoftmax`＋VJP・`autodiff::var::Var::softmax`／`log_softmax`・
`nn::activation::Softmax`／`LogSoftmax`（`Module` 実装）を追加した（上記
「再利用可能なテンプレート」と同型のテンプレートで実装）。3 バックエンドの
既存行カーネル（CPU/CUDA/Metal の `softmax.rs`。§2.4 が「既存」と記す
カーネル）へ接続済み。**GPU（CUDA／Metal）の `log_softmax` は行カーネルを
新設せず、`BackendOps::log_softmax` の既定 `Unsupported` のままホスト参照
実装（`eval::log_softmax_along`）へフォールバックする**（CPU のみ融合
カーネル `backend-cpu::softmax::run_log_softmax_f32` で本番オーバーライド
する）。非最終軸 softmax／log_softmax も同様にホストフォールバック。
`cross_entropy_loss` 内部の log-softmax（`eval::softmax_along`。1 個の融合
オペとして解析形で forward/backward を閉じる既存実装）は本イシューで
変更しない（別実装のまま独立に存在する）。GPU `log_softmax` カーネル・
非最終軸 GPU 対応は out-of-scope として記録し、起票はユーザー承認後に限る
（`.claude/rules/out-of-scope-tracking.md`）。

### 追補（イシュー #1596）

§2.7 の表（`nn.LayerNorm`／RMSNorm の行）は本ドキュメント作成時点（対象 HEAD
`097bff19`）のスナップショットとして不変のまま残す。イシュー #1596 で以下を実装し、
上記ギャップを解消した（設計・実機実測状況の詳細は `docs/norm-ops-design.md` を正とする）:

- `fandhe_ai_tensor_core::BackendOps::rmsnorm`／`layer_norm` を新設（既定 `Unsupported`）。
  `rmsnorm` は既存の RMSNorm 行カーネル（`backend-{cpu,cuda,metal}::rmsnorm`）を
  `run_fused`（canonical 融合プラン限定経路）とは別の独立エントリとして接続した。
  `layer_norm` は 3 バックエンドとも新設カーネルで実装した
- `fandhe_ai_autodiff::Var::rms_norm`／`layer_norm`・`nn::RmsNorm`／`LayerNorm`
  （`RmsNormVars`／`LayerNormVars` 込み）を追加し、`Module` trait を実装した
- VJP（`grad.rs::rmsnorm_vjp_rows`／`layer_norm_vjp_rows`）をホスト側に実装し、
  数値微分・解析的性質（`Σ_row dx = 0` 等）で検証した
- facade（`crates/facade/src/`）への新規 `pub use`／`pub fn` 追加は**行っていない**。
  既存の `Var` 再エクスポート経由でユーザーへ到達する（#1594 softmax と同型の方針）
- 対象外: GPU backward カーネルの結線（VJP はホスト側実装のまま）・CUDA 既存
  RMSNorm backward カーネル（`rmsnorm_bwd_*`）への接続・多次元 `normalized_shape`・
  `Sequential::add_rms_norm`／`add_layer_norm`（#1618 のスコープ）・CPU 側 NEON
  ベクトル化・CUDA 実機実測（本エージェント実行環境に CUDA 実機への到達手段がない
  ため未実施のまま記入欄を残す）

## 追補（イシュー #1597）

§2.3「形状操作」表（上記）は取り込み元スナップショットのギャップ記述の
ため変更していないが、`permute`／`expand`（`broadcast_to`）／
`squeeze`／`unsqueeze` の各行が指す欠落は本イシューで解消済みである。

- `Var::permute`（新規 `tape::Op::Permute { input, perm }`。`Tensor::permute`
  への zero-copy 接続・VJP は逆置換 `upstream.permute(&inverse_perm)`）
- `Var::broadcast_to`（新規 `tape::Op::BroadcastTo { input }`。
  `Tensor::broadcast_to` への stride 0 view 接続・VJP は `Op::Add`/`Op::Mul`
  の暗黙ブロードキャストと同じ `reduce_to_shape` 縮約を再利用）
- `Var::expand`（`Var::broadcast_to` への薄い委譲。PyTorch 名の別名。
  負値〈-1 で軸維持〉は `shape: &[usize]` の型上表現できないため非対応）
- `Var::squeeze`／`Var::unsqueeze` は新規 `Op` を持たず、いずれも
  `Var::reshape`（既存 `Op::Reshape`）へ委譲する薄いラッパー（§4 の
  「薄いラッパー」区分どおり）。`squeeze(Some(d))` は `shape[d] != 1` の
  場合 PyTorch 準拠で no-op（numpy／TF はエラーにするが、適合する
  `ShapeError` variant が存在せず crates.io 公開クレート `tensor-core` の
  公開 enum への variant 追加は semver 可視の変更になるため見送った）
- `Var::flatten`（PyTorch `torch.flatten(start_dim, end_dim)` 相当。同じく
  `Var::reshape` への委譲。§2.3 表には行がないが同じ Tier 1 issue の
  スコープとして実装した）

いずれも `Var::reshape`／`transpose` と同じ「非 contiguous な入力への
適用は `ShapeError::NonContiguousReshape`」制約（案 A）を継承する（
`Var::contiguous()`〈明示コピー Op〉は本イシューでは追加せず、
out-of-scope として記録した）。`cat`／`stack`／`split`（§2.3 の残り 2 行）
は #1598 へ引き継ぎ。5 演算自体はホスト `Tensor` の stride 再解釈のみで
`BackendOps` を経由しないため 3 バックエンドへの個別実装は不要——
下流の融合経路（`add`）・GEMM カーネル（`matmul`）が view を正しく
消費することを `crates/facade/tests/shape_ops_backend_parity.rs` で
検証した（CPU 属性なし・Metal／CUDA `#[ignore]`）。

## 追補（イシュー #1598）

`cat`／`stack`／`split`（§2.3）・`narrow`（`docs/spec/04-requirements.md`
「index 系」・#1599 の対象だった行）を実装済み化した:

- `Var::cat(vars, dim)` — 新 Op `Op::Concat`（コピーを伴う `push_eager`
  ノード。`BackendOps::concat`〈既定 `Unsupported`〉→ `eval::concat`
  フォールバック）。3 バックエンド（CPU／CUDA／Metal）の専用カーネルは
  本イシューでは実装せず（既定実装のホストフォールバックのみ）、
  性能最適化は別イシューへ引き継ぐ（out-of-scope）。
- `Var::stack(vars, dim)` — 各要素を `unsqueeze(dim)` してから `cat`
  （PyTorch の定義そのもの）。
- `Var::narrow(dim, start, len)` — 新 Op `Op::Narrow`（zero-copy view
  ノード。`push_view`／`resolve_view` 経由。`Tensor::narrow` の再導出）。
  `docs/compat-api-scope.md` §1.2「index 系」行の narrow はこれで解消
  済み（残る where／gather／scatter は #1599 が引き続き対象）。
- `Var::split(split_size, dim)`／`split_with_sizes(sizes, dim)`／
  `chunk(chunks, dim)` — いずれも `Var::narrow` への委譲（PyTorch
  意味論。`split` の VJP は「Split の VJP は Concat」の原則で
  `Op::Narrow` の VJP が zero-pad `Concat` により入力 shape へ戻す）。

facade への到達経路は既存の `pub use fandhe_ai_autodiff::Var` 再エクス
ポートのみで、新規 `pub use`／`pub fn` は追加していない（`docs/
compat-api-scope.md` §5 の手続きは Tier 1 列挙済み機能につき再適用
不要と判断）。

## 追補（イシュー #1637）

§2.2「`masked_select`/`where`」行（スナップショット時点の記述は不変の
まま）を実装済み化した:

- `Var::where_cond(cond: &Tensor<bool>, a: &Var, b: &Var)` — 新 Op
  `Op::Where { cond: Tensor<f32>, a: NodeId, b: NodeId }`（コピーを伴う
  `push_eager` ノード。`BackendOps::where_cond`〈既定 `Unsupported`〉→
  `eval::where_cond` フォールバック）。`cond`（`&Tensor<bool>`）は
  `Var::where_cond` が `out_shape`（`a`／`b` の broadcast 後 shape）へ
  broadcast してから 1 回だけ f32 マスク（`{0.0, 1.0}`）へ変換し Op が
  保持する（`MemoryOps` の f32 専用契約に合わせるため）。真偽判定は
  3 バックエンド共通で `c != 0.0`。
- `Var::masked_fill(&self, mask: &Tensor<bool>, value: f32)` — 新 Op
  `Op::MaskedFill { input: NodeId, mask: Tensor<f32> }`（`value` は
  forward が焼き込んだ `TapeNode::value` に含まれるため Op へ二重保持
  しない）。`BackendOps::masked_fill`〈既定 `Unsupported`〉→ `eval::
  masked_fill` フォールバック。
- **CPU／CUDA／Metal の 3 バックエンドとも専用カーネルを実装**
  （`backend-cpu::elementwise::{where_slice, masked_fill_slice}`・
  CUDA `kernels_elementwise.rs::{EW_WHERE_F32, EW_MASKED_FILL_F32}`・
  Metal `shaders/elementwise.metal::{ew_where_f32, ew_masked_fill_f32}`。
  `#1598` の cat／narrow 系とは異なりホストフォールバックのみに留めて
  いない）。選択演算は丸めを伴わないため 3 バックエンドとも bit 同一
  になることを parity テスト（`crates/backend-cuda/tests/
  where_masked_fill_parity.rs`・`crates/backend-metal/tests/
  where_masked_fill_parity.rs`。Metal は M4 Max 実機実測完了・CUDA は
  本エージェント実行環境に実機なしのため未実測明記）で確認した。
- **VJP はホスト実装**（`Op::Relu` と同型。`grad::elementwise_mul_mask`
  を再利用）。デバイス常駐 VJP（`binary_elementwise_device` 相当）は
  本イシューのスコープ外。
- facade への到達経路は既存の `pub use fandhe_ai_autodiff::Var` 再エク
  スポートのみで、新規 `pub use`／`pub fn` は追加していない（`docs/
  compat-api-scope.md` §1.2「index 系」行を参照。§5 の範囲拡張手続きは
  Tier 1 列挙済み機能につき再適用不要と判断）。
- gather／scatter／scatter_add／index_select（`docs/compat-api-scope.md`
  §1.2「index 系」行の残対象）は #1638 へ引き継ぐ。
