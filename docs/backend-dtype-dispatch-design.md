# dtype 別 dispatch（`BackendOps` の dtype 多重化）方式の設計記録（#1648）

イシュー #1648「dtype 別 dispatch（`BackendOps` の dtype 多重化）方式を設計する」に対応する。親: #1626（低レイヤー診断・機能網羅ツリー #1570 の sub (a)）。

本ドキュメントは**設計のみの記録であり、コード変更（`crates/**`）を一切伴わない**。実装は後続イシュー（#1649 CPU → #1650 CUDA → #1651 Metal）が本設計に従って行う。tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline（`ParityBaseline::BASELINES`）の変更・新規追加は本設計のスコープ外であり、実際に本文書はそれらを一切変更しない。

**本文書自体は承認記録ではない**。§7 に列挙する事項は実装着手前にユーザー承認が必要（`docs/spec/` 正本の変更を伴わないが、公開クレート `fandhe-ai-tensor-core` の trait 拡張であるため）。

棚卸し時点の HEAD（origin/main）: `f91cafa3`（2026-09-13）。`file:line` は同時点のもの。

## 0. 判断サマリ

- 推奨方式は **capability accessor ＋ 型パラメータ trait**（§4 の案 D）: `BackendOps` に既定 `None` を返すアクセサ `typed_ops_f64`／`typed_ops_f16`／`typed_ops_bf16(&self) -> Option<&dyn TypedOps<T>>` を非破壊追加し、dtype ごとの演算本体は新設 `pub trait TypedOps<T: Scalar>`（trait 型パラメータのため object-safe）に集約する。`fn memory_ops(&self) -> Option<&dyn MemoryOps>`（`crates/tensor-core/src/backend_ops.rs:354`）と同型の既存パターンをそのまま踏襲する
- 不変条件（機械検査可能な形で列挙。実装側の受け入れ条件とする）:
  1. `dyn BackendOps` の object safety が崩れない（`assert_object_safe(_ops: &dyn BackendOps) {}`。`crates/tensor-core/src/backend_ops.rs:1868` と同型のテストで担保する）
  2. 既存 `BackendOps` メソッドのシグネチャ・挙動が不変（公開 API 非破壊）
  3. f32 経路は before/after で bit 同一（新 trait を f32 にも実装する場合、既存メソッドへの委譲のみで構成する）
  4. 新規追加は既定 `None`／`Unsupported` から開始する（fail-closed。#1649〜#1651 の受け入れ条件と一致させる）
- 本設計は `dispatch::DType`（`crates/tensor-core/src/dispatch.rs:30`）を拡張しない。REQ-11 の「行列演算ユニットの明示切替 API を利用者に提供しない」制約とは独立の理由（後述 §4.4）による

## 1. 背景・目的

現状、演算可能な dtype は `f32` のみである。`Tensor<T: Element>`（`T` は `f32`／`f64`／`i32`／`i64`／`bool`／`half::f16`。`crates/tensor-core/src/element.rs:24-83`）自体はジェネリックだが、`BackendOps`（`crates/tensor-core/src/backend_ops.rs:328`）・`MemoryOps`（`crates/tensor-core/src/buffer.rs:374` 付近）・`DeviceBufferView`・autodiff の `Var`／`Tape`（`Tape.ops: Box<dyn BackendOps + Send>`。`crates/autodiff/src/tape.rs:775`）はすべて `f32` に固定されている。

`docs/public-api-design.md` は 2 箇所でこの未決事項を明記している。

- §4.2（`crates/tensor-core/src/backend_ops.rs` 相当の設計節、`docs/public-api-design.md:637`）: 「`BackendOps` を `T: Element` でジェネリック化するか、`f16` 専用の並行トレイトを追加するかは TASK-1.9 実装時に決定する」
- §6-8（`docs/public-api-design.md:721`）: 同内容の要約

本イシューはこの未決事項を閉じ、実装側（#1649〜#1651）が従う具体案を確定する。

## 2. 現状のコード事実（棚卸し）

| 事実 | 出典 |
|---|---|
| `Element` は `Copy + Send + Sync + Debug + PartialEq + 'static` に `zero()`／`one()` を要求する unsealed trait。実装対象は `f32`／`f64`／`i32`／`half::f16`／`i64`／`bool` | `crates/tensor-core/src/element.rs:24-83` |
| `BackendOps` は `f32` 固定。`memory_ops(&self) -> Option<&dyn MemoryOps> { None }` という「既定 `None` を返すアクセサ trait」の非破壊拡張パターンが既に存在する | `crates/tensor-core/src/backend_ops.rs:328,354` |
| `dyn BackendOps` の object safety はテスト `assert_object_safe(_ops: &dyn BackendOps) {}` で機械検査されている | `crates/tensor-core/src/backend_ops.rs:1868` |
| `Tape.ops: Box<dyn BackendOps + Send>`。dtype ジェネリックにする場合は `Tape<T>` 化が必要（本段階ではスコープ外。§8） | `crates/autodiff/src/tape.rs:775` |
| `dispatch::DType { F32, F16 }` は `#[non_exhaustive]` **なし**。variant 追加は下流の全網羅 match を壊す破壊的変更 | `crates/tensor-core/src/dispatch.rs:30-40` |
| `select_gemm_kernel` は利用者向け明示切替 API ではなく、CUDA／Metal の GEMM 自動経路入口が内部で呼ぶ規則エンジン（REQ-11 の受け入れ基準） | `crates/tensor-core/src/dispatch.rs:9-19`、`docs/dispatch-rules-design.md` §5.1 |
| CUDA f16 Tensor Core 経路 `CudaGemmAuto::run_f16`（mma.sync 優先→WMMA フォールバック）は `BackendOps` から到達不能（`gemm_auto.rs` は `backend_ops.rs` の外） | `crates/backend-cuda/src/gemm_auto.rs:1623,1768` |
| CUDA f32 精度切替 `CudaGemmPrecision { Fp32Strict, Tf32, Tf32x3 }`（既定 `Fp32Strict`）は `CudaBackendOps::gemm` のみに適用。dtype 切替ではなく同一 f32 dtype 内の演算精度切替 | `crates/backend-cuda/src/precision.rs:12-77` |
| CUDA 側 dtype 一般化の前例 `pub(crate) trait PoolDtype: DeviceRepr + Sized`（`f32`／`f16` 実装済み）。プールアロケータ限定で公開 API ではない | `crates/backend-cuda/src/pool.rs:202,229,254` |
| Metal f16 タイル GEMM 入口 `gemm::MetalGemm::dispatch_f16_auto_unverified` は `_unverified` suffix・`#[doc(hidden)]`（PR #346 codex-review 指摘により意図的に未検証扱いのまま維持） | `crates/backend-metal/src/lib.rs:91-183` |
| Metal（MSL）は `double` 型非対応。既存の回避策は「64bit 整数による `f64` 加算のソフトウェアエミュレーション」（`soft_f64.rs`。GEMM の bias 勾配縮約限定・bit 完全一致契約） | `crates/backend-metal/src/soft_f64.rs:1-30`、`.claude/rules/coding-rust.md`「正規化統計・勾配の長軸縮約」節 |
| onnx-interop の dtype タグ付き enum `pub enum Value { F32, I64, Bool, F16 }` が「enum で dtype を運ぶ」方式の前例 | `crates/onnx-interop/src/onnx/interp.rs:62` |
| `half = "=2.7.1"`（`bf16` を同梱）。`cudarc = "=0.19.8"` は `f16` feature 済み依存 | `Cargo.toml:112,146` |
| `docs/compat-feature-gap.md` §2.12 は `float64`／`float16`/`bfloat16` を「部分実装（`BackendOps`/`Var` の算術対象外）」・必要工数 XL と記録済み | `docs/compat-feature-gap.md:319-320` |

## 3. 関連 issue との境界

- **#1613（cast `.to(dtype)`）**: f32 ⇔ 他 dtype の変換 API は #1613 側の責務。本設計は変換後の「dtype ごとの演算実行」経路のみを扱う
- **#1625（AMP: automatic mixed precision）**: 本 dtype 多重化を前提として損失スケーリングを実装する。本設計には AMP のスケーリング契約を含めない
- **#1627（量子化）**: spec 除外事項に従属し着手不可。本設計の対象外

## 4. 候補比較

| 案 | 概要 | object safety | 公開 API 非破壊 | 演算×dtype の増え方 | 判定 |
|---|---|---|---|---|---|
| A | `BackendOps` のメソッドを `fn gemm<T: Element>(...)` へジェネリック化 | **不成立**（メソッドの generics は `dyn` 非互換。Rust の基本制約） | 破壊的変更 | 1 定義 | 不採用 |
| B | dtype 別メソッド族を直接追加（`gemm_f64`／`gemm_f16`／`gemm_bf16` 等を既定 `Unsupported` で追加） | 成立 | 非破壊 | 演算数 × dtype 数で線形爆発。命名規約の維持コストが増大し続ける | 最小集合なら可能だが非推奨 |
| C | dtype タグ付き enum（`onnx-interop::Value` 前例）を受ける `gemm_dyn(&self, a: &TensorDyn, ...)` | 成立 | 非破壊 | 1 定義／演算・実行時 match で分岐 | 代替案。型安全性を実行時チェックへ落とし f32 経路にも分岐コストが乗る |
| D | `trait TypedOps<T: Scalar>` ＋ `BackendOps::typed_ops_<dtype>() -> Option<&dyn TypedOps<T>>` | 成立（trait は型パラメータであり `dyn` 自体を要求しない。個々の `TypedOps<f32>` 等は具象型でアクセスするため dyn 互換性の問題が生じない） | 非破壊（既定 `None`。`memory_ops` と同型） | 1 定義／演算・backend × dtype ごとに `impl TypedOps<T> for XxxBackendOps` | **推奨** |

### 4.1 案 D の詳細

新設型（いずれも公開クレート `fandhe-ai-tensor-core`。§7 の承認対象）:

```rust
/// 演算対象になりうる dtype の capability 境界。
/// `Element`（`crates/tensor-core/src/element.rs:24`）は unsealed で
/// 外部実装が存在しうるため変更しない。演算に必要な追加境界は
/// sub-trait として表現する。
pub trait Scalar: Element {
    const DTYPE: ScalarDType;
    // 四則演算・比較・f32/f64 相互変換に必要な最小境界（実装時に確定）
}

/// 演算対象 dtype のタグ。`dispatch::DType`（GEMM 経路選択専用・
/// `#[non_exhaustive]` なし）とは別の列挙とし、拡張してよいものと
/// してよくないものを区別する。
#[non_exhaustive]
pub enum ScalarDType { F32, F64, F16, Bf16 }

/// dtype 別の演算本体。`BackendOps` の `typed_ops_<dtype>()` accessor
/// 経由でのみ取得する。
pub trait TypedOps<T: Scalar> {
    fn gemm(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn relu(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn exp(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn tanh(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn sum(&self, a: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>, BackendError>;
    fn max(&self, a: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>, BackendError>;
}
```

`sum`／`max` は既存 `BackendOps::sum`／`max`（`crates/tensor-core/src/backend_ops.rs:805-806`）と同じ `dim: Option<usize>`（`None` は全要素縮約・`Some(d)` は軸 `d` に沿った縮約）を保持する。dtype 多重化はこの引数を変更する理由にならないため、`TypedOps<T>` でも軸指定を落とさない。

`BackendOps` への非破壊追加:

```rust
fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> { None }
fn typed_ops_f16(&self) -> Option<&dyn TypedOps<f16>> { None }
fn typed_ops_bf16(&self) -> Option<&dyn TypedOps<bf16>> { None }
```

`TypedOps<f32>` を追加するかは実装側判断とするが、追加する場合は既存 `BackendOps` メソッドへの委譲のみで構成し bit 同一を保つ（§0 不変条件 3）。これにより `fn run<T: Scalar>(ops: &dyn TypedOps<T>)` のような dtype ジェネリックな利用側コードが書ける。

入出力はホスト常駐 `Tensor<T>`（`BackendOps` v1 と同じ契約）に限定する。`DeviceBuffer<T>` 常駐経路（`memory_ops`／`linear_forward_device` 系）の dtype 多重化は「段階 B」として本設計のスコープ外に置く（§8）。

### 4.2 最小演算集合（第 1 段）

`gemm`・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・`max` の 8 演算（既存 `BackendOps` v1 の演算集合と同一）。`gemm_bias_act`・`mse_loss`・softmax 系・resident 系・fusion 対応は第 2 段以降（別 issue）とし、本表で対象を明示的に固定することで #1649〜#1651 が対象を無制限に広げないようにする。

### 4.3 利用側の dtype 選択と REQ-11 の関係

dtype の選択は「`Tensor<f16>` を渡す」という**型で決まる入力**であり、REQ-11 が禁じる「行列演算ユニットの明示的な設定項目としての切替 API」には該当しない。GEMM 内部のカーネル経路選択（TF32 か SIMT か、mma.sync か WMMA か）は既存 `select_gemm_kernel(caps, shape, DType::F16)`（`crates/tensor-core/src/dispatch.rs`）をそのまま再利用し、`TypedOps<f16>::gemm` の実装内部から呼ぶ。dtype 多重化のための並行規則エンジンは作らない。

### 4.4 `dispatch::DType` を拡張しない理由

`dispatch::DType`（`crates/tensor-core/src/dispatch.rs:30`）は `#[non_exhaustive]` を付けていないため、variant 追加（`F64`・`Bf16` 等）は下流の全網羅 `match` を壊す破壊的変更になる。加えて `select_gemm_kernel` は GEMM カーネル経路選択専用の決定表（`docs/dispatch-rules-design.md` §5.3）に紐づいており、f64／bf16 の GEMM 経路が実機実測・承認を経ていない現時点でこの決定表へ組み込むのは時期尚早である。本設計は代わりに新設 `ScalarDType`（`#[non_exhaustive]` 付き）で dtype を表現し、GEMM 実行時にのみ `ScalarDType → Option<dispatch::DType>` の明示マッピング（`F32 → Some(F32)`、`F16 → Some(F16)`、`F64`／`Bf16 → None`＝規則エンジン非対象。呼び出し側でカーネル経路を直接選ぶ）を経由して接続する。`dispatch::DType` の `#[non_exhaustive]` 化自体は破壊的変更を伴うため別途承認判断に委ねる（§8）。

## 5. 実現可否表（dtype × backend）

| dtype | CPU | CUDA | Metal |
|---|---|---|---|
| f64 | 実装可（`f64::mul_add` 参照実装。並列 BLIS 化は任意） | SIMT カーネル実装可能だが性能目的なし。既定 `Unsupported` から開始 | **構造的に不可**（MSL に `double` 型が存在しない。`soft_f64.rs` の 64bit 整数エミュレーションは bias 勾配縮約というスカラー累算専用に作られたものであり、GEMM 全体を `f64` 精度で動かす手段ではない）→ 恒久 `Unsupported`（fail-closed） |
| f16 | `half` によるソフトウェア変換で f32 累算（aarch64 fp16 intrinsics は `unsafe` を伴うため実装時に別途 security-auditor 承認が必要。既定はソフトウェア変換） | 既存 `CudaGemmAuto::run_f16`（mma.sync 優先→WMMA）を結線。parity は REQ-2 形状別判定方式の既存 baseline 範囲内（新規 baseline 追加は行わない） | 既存 `dispatch_f16_auto_unverified` を結線。`_unverified`／`#[doc(hidden)]` の解除可否は #1651 の承認事項（§7-4） |
| bf16 | `half::bf16` で f32 累算（依存追加なし。`half =2.7.1` に同梱） | cudarc 0.19.8 に `bf16` 向け `DeviceRepr` 実装があるかは本設計の調査範囲（ローカルソース grep）では確認できていない。外部レジストリ参照が必要なため **未検証**と明記し #1650 の調査事項とする | **未検証**（MSL の `bfloat` 型可用性・`simdgroup_matrix` 対応をコンパイルプローブで確認する必要がある。`crates/backend-metal` の `.metal` シェーダに `bfloat` の使用例は現時点で存在しない。手法は `docs/perf/logs/metal-gemm-mpp-tensor-1326/` のコンパイルプローブ方式を踏襲する） |

## 6. 数値契約（tolerance／baseline 不変）

- **累算契約**: f16／bf16 は入力を f32 へ昇格し `f32::mul_add` で累算、最後に 1 回だけ元の dtype へ丸める（GPU の「f16 入力・f32 累算」Tensor Core と同型）。f64 は `f64::mul_add` を使う。matmul 系 FMA 契約（`.claude/rules/coding-rust.md`「バックエンド構成」節）と整合し、これを変更しない。正規化統計・長軸縮約の `f64`／soft-f64 アキュムレータ契約（同ルール文書の別節）も本段階の最小演算集合（§4.2）には含まれないため不変のまま
- **判定方法**: f16／bf16 出力は、参照値側も出力 dtype と同じ丸め（f32 参照計算 → f16／bf16 へ最近接丸め → f32 へ再昇格）を経てから f32 昇格後の実測出力と既存 `compare`／`assert_parity`（`crates/backend-cpu/src/parity.rs:148,239`）で判定する。参照値を丸め前の f32 のまま比較する方式は採らない: 例えば `1 + 2^-8` の bf16 最近接偶数丸め結果は丸め前 f32 参照値に対し相対誤差 約 0.003891・絶対誤差 約 0.003906（複合判定の両閾値 `RELATIVE_TOLERANCE`＝1e-3・`ABSOLUTE_RESCUE_THRESHOLD`＝1e-5 をいずれも超過）となり、丸め自体が正しい bf16 実装ですら不合格になりうる（tolerance 定数の緩和ではなく、比較対象を出力 dtype の表現可能値に揃える判定契約の整備で解消する）。dtype ごとの丸め関数（`half::f16::from_f32`／`half::bf16::from_f32`）は `half =2.7.1` に既存でありこの目的のためだけの新規実装は不要。f64 出力は f64 のまま同一の定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）で判定するヘルパー追加は実装側（#1649）の作業とし、定数自体は共有し変更しない。CUDA f16 GEMM の既知不合格形状は spec REQ-2（2026-09-02／2026-09-12 追記）の実測 baseline 非後退方式の既存範囲で扱い、**本設計は新規 baseline を追加しない**（追加には実機実測値と人間承認が必要）
- **bit 同一契約**: f32 経路（既存メソッド・`TypedOps<f32>` を追加する場合はその委譲実装）は before/after で bit 同一であることを実装側の受け入れ条件とする

## 7. 承認事項（実装着手の前提。本文書は承認記録ではない）

1. 公開クレート `fandhe-ai-tensor-core` の `BackendOps` への既定メソッド追加（`typed_ops_f64`／`typed_ops_f16`／`typed_ops_bf16`）
2. 新規公開型 `Scalar`・`ScalarDType`（`#[non_exhaustive]`）・`TypedOps<T>` の追加、および `impl Element for half::bf16` の追加
3. `dispatch::DType` を拡張しない方針（§4.4）。拡張が必要になった場合は破壊的変更として別途承認を要する
4. Metal f16 入口 `dispatch_f16_auto_unverified` の `_unverified`／`#[doc(hidden)]` 解除可否（#1651 の実装時判断）
5. CPU f16／bf16 経路で `unsafe` intrinsics（aarch64 fp16 等）を使う場合の承認（既定はソフトウェア変換で `unsafe` 非導入。#1649）
6. 最小演算集合（§4.2）と、#1649〜#1651 の受け入れ条件の再スコープ（`Var`／`Tape`／VJP は本段階の対象外。§8 参照）
7. facade（`crates/facade`）公開面への昇格は本設計の対象外。`docs/compat-api-scope.md` §5 に定める昇格手続きを別途要する

## 8. スコープ外・引き継ぎ

- **`Var`／`Tape` の dtype 一般化**: `Tape.ops: Box<dyn BackendOps + Send>`（`crates/autodiff/src/tape.rs:775`）は本段階では `f32` のまま不変。dtype ジェネリックな `Var<T>`・VJP・`FusionPlan` の対応は別イシュー
- **AMP（損失スケーリング）連携**: #1625 側の責務
- **cast（`.to(dtype)`）**: #1613 側の責務
- **`MemoryOps`／`DeviceBuffer<T>` 常駐経路の dtype 多重化**（段階 B）: `linear_forward_device` 系のデバイス常駐チェーンへの dtype 拡張は本設計に含めない
- **fusion（カーネル融合機構）の dtype 対応**: `kernel-fusion.md` の対象範囲は f32 のまま
- **Metal bf16 可用性の実機コンパイルプローブ**: #1651 の調査事項
- **CUDA bf16 `mma.sync` カーネルの新規実装**: #1650 の実装事項（cc ≥ 8.0 要）
- **f64 GPU カーネル（CUDA SIMT）**: 性能目的がないため優先度は低いが、#1650 で `Unsupported` から始める既定実装は含めてよい
- **`dispatch::DType` の `#[non_exhaustive]` 化**: 破壊的変更のため crates.io 版数運用とセットで別途判断（本設計は現状維持を推奨）

上記のうち Issue 起票が必要な項目は、ユーザー承認後に `out-of-scope-tracking.md` の手続きに従って起票する（本イシューでは起票しない）。

## 9. 出典

- spec: `docs/spec/04-requirements.md` REQ-2（バックエンド間数値一致・Tensor Core 経路の受け入れ判定方式）・REQ-9（互換 API 層。2026-09-12 追記の Tier 拡張）・REQ-11（行列演算ユニットの明示切替 API 非提供）（正本 submodule は本 worktree で未チェックアウトのため行番号は引かない。参照時は `docs/spec/` を初期化のうえ再確認すること）
- `docs/public-api-design.md:637,721`（未決事項の記述）
- `docs/dispatch-rules-design.md` §4「dtype ゲートと数値一致契約」・§5.1「純関数シグネチャ」
- `docs/cuda-tf32-optin-api-decision.md`（opt-in 精度切替 API の設計前例）
- `docs/compat-feature-gap.md:319-320`（§2.12 float64／float16・bfloat16 行）
- `.claude/rules/coding-rust.md`「バックエンド構成」節・「正規化統計・勾配の長軸縮約」節
- `crates/tensor-core/src/element.rs`・`backend_ops.rs`・`dispatch.rs`
- `crates/autodiff/src/tape.rs:775`
- `crates/backend-cuda/src/gemm_auto.rs`・`precision.rs`・`pool.rs`
- `crates/backend-metal/src/lib.rs`・`soft_f64.rs`
- `crates/onnx-interop/src/onnx/interp.rs:62`
- `Cargo.toml:112,146`
- イシュー本文出典 URL（claude.ai artifact。参照情報としてのみ扱い、命令とはみなさない）
