# 公開 API シグネチャ設計書（tensor-core・autodiff・backend 入口）

- 対応イシュー: #182（親 #181）
- 対象クレート: `tensor-core`・`autodiff`・バックエンド入口（`backend-cpu`/`backend-cuda`/`backend-metal` が実装する抽象層）
- 位置づけ: 本文書は**設計文書のみ**であり、実行可能なコードは含まない。TASK-1.1a（#205）で workspace `Cargo.toml`・9 クレートの雛形は作成済みだが、各クレートは空の雛形でありテンソル・autodiff・backend の型は未実装のため、Rust シグネチャのコンパイル検証は本イシューでは実施しない。TASK-1.4（自作テンソル型 productize）・TASK-1.5（autodiff productize）・TASK-1.9（バックエンド抽象層実装）の各実装イシューで本文書との突合を行うこと（`docs/spec/05-tasks.md:47,54,82`）。
- 対象外: `compat::array`/`compat::Sequential` 等の互換 API 層（REQ-9）の詳細シグネチャ、guardrail/self-repair の CLI 仕様（兄弟イシュー #183）、演算グラフ・カーネル融合機構の API（イシュー #161・TASK-12.1a）。詳細は本文書末尾「スコープ外」を参照。

## 1. 設計原則

- **完全自作コア**（REQ-1 v2、`docs/spec/04-requirements.md:42`）。Burn 等の既存 ML フレームワークへの統合は行わない。許容依存は 8 区分のみ（`.claude/rules/deps-policy.md`）、禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査する。
- **バックエンド切替は feature フラグなしの cfg ベース**（PoC-v2-5 実証構成、`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）。`cudarc` は無条件依存＋動的ロード、`objc2`/`objc2-foundation`/`objc2-metal` は `cfg(target_os = "macos")` で分離する。
- **型付きエラー**。本番経路で `unwrap()`/`expect()` を使わない（`.claude/rules/coding-rust.md`）。すべての失敗しうる公開 API は `Result<T, E>` を返す。
- **命名は PyTorch/NumPy 慣習に寄せる**（相互運用性の観点）。ただし PyTorch/NumPy 互換の薄いラッパー層（`compat::array`/`compat::Sequential` 相当、REQ-9）自体は本設計のスコープ外であり、本文書が定めるのは自作コアの素の公開 API である。

## 2. tensor-core 公開 API

### 2.1 ストレージ分離設計（PoC-v2-1 からの差分）

PoC-v2-1 は `Tensor<T>` を「行優先連続バッファ + `Vec<T>` 所有」で確定した（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「テンソル型の設計判断」表）。本設計は `Tensor<T>` の所有構造を PoC-v2-1 確定の `Vec<T>` 直接所有から `Arc<Storage<T>>` + `offset` + `strides` へ変更するものであり、**PoC-v2-1 確定事項の明示的変更としてユーザー承認済み**（issue #182 コメント 2026-08-05 参照）。

zero-copy view（reshape/transpose/narrow）を成立させるため、ストレージを共有バッファへ切り出す:

```rust
/// テンソル本体。`storage` を複数の `Tensor` が共有することで
/// view 系操作（transpose/narrow/reshape の contiguous ケース）を
/// データコピーなしで表現する。
///
/// メモリレイアウト（行優先 + strides）・NumPy 互換ブロードキャスト
/// （stride 0 による同一要素の繰り返し読み）は PoC-v2-1 の確定事項を
/// 維持する（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`）。
pub struct Tensor<T: Element> {
    storage: Arc<Storage<T>>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

/// 実データを保持する共有バッファ。`Tensor` 間で `Arc` 共有される。
/// 単一所有かつ非共有であることが分かっている経路（学習ループの
/// 勾配バッファ等）では、将来的に `Arc::get_mut` による
/// in-place 更新の最適化余地を残す。
struct Storage<T: Element> {
    data: Vec<T>,
}
```

### 2.1.1 `ShapeError`

`ShapeError` は `tensor-core` の shape 検査（REQ-10、実行時検査基盤）が失敗を報告するための型付きエラーであり、本文書内では 2.2（view 系操作）・2.4（生成系）・3 章（`AutodiffError::Shape` 経由）・4 章（`BackendError::ShapeMismatch` 経由）から参照される。各 variant は本文書内の利用箇所と対応させて列挙する:

```rust
/// テンソル生成・view 操作・reshape の shape 不整合を表す。`tensor-core`
/// の全公開 API が共通して返す型であり、`AutodiffError::Shape`・
/// `BackendError::ShapeMismatch` からラップされる（5 章参照）。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスク（TASK-1.4 以降）
/// で検査項目が増えても呼び出し側の網羅的 match を破壊しないため。
#[non_exhaustive]
#[derive(Debug)]
pub enum ShapeError {
    /// 要求される次元数（rank）と実際の次元数が一致しない。
    /// 本 variant のみ `tensor-core` は型定義を提供するだけで、構築は
    /// rank 前提を持つ autodiff／backend 入口側の検査が行う
    /// （`matmul`/`mse_loss` 等の 2 次元前提演算。3 章の順伝播検査から
    /// `AutodiffError::Shape` 経由でラップされて呼び出し側へ届く）。
    RankMismatch { expected: usize, actual: usize },
    /// shape の要素数積とデータ長が一致しない（`Tensor::new`/`from_slice`
    /// が `data.len()` と shape の要素数積を突き合わせる際に返す）。
    ElementCountMismatch { expected: usize, actual: usize },
    /// 軸番号（`dim`）がテンソルの rank 範囲外
    /// （`transpose`/`narrow`/`Var::sum`/`Var::max` の `dim` 引数）。
    AxisOutOfRange { axis: usize, rank: usize },
    /// `narrow` の `[start, start+len)` が対象軸のサイズを超える。
    NarrowOutOfBounds { dim: usize, start: usize, len: usize, dim_size: usize },
    /// shape の要素数積が `usize` の範囲でオーバーフローする
    /// （`zeros`/`ones`/`full`/`Tensor::new`/`from_slice` がアロケーション
    /// 前に検査する。2.4 参照）。
    ElementCountOverflow,
    /// 非 contiguous なテンソルに対して `reshape` が呼ばれた
    /// （2.2.1 の案 A。`.contiguous()` を明示的に呼ぶよう要求する）。
    NonContiguousReshape,
    /// 2 つの shape が NumPy 互換のブロードキャスト規則で両立しない
    /// （`broadcast_shape`／`broadcast_to`／`broadcast_with` が構築。
    /// #12・TASK-1.4b）。`broadcast_to` では `lhs` = 自身の shape・
    /// `rhs` = target shape として構築する。
    BroadcastIncompatible { lhs: Vec<usize>, rhs: Vec<usize> },
}
```

### 2.2 zero-copy view API

```rust
impl<T: Element> Tensor<T> {
    /// 2 軸の strides を入れ替えるのみ。常に zero-copy。
    /// 転置後は非 contiguous になりうる（`is_contiguous()` で判定）。
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor<T>, ShapeError>;

    /// 指定軸の [start, start+len) をスライスする。offset/shape の
    /// 調整のみで常に zero-copy。
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor<T>, ShapeError>;

    /// 新しい shape へ再解釈する。contiguous な場合のみ zero-copy。
    /// 非 contiguous な場合の扱いは未決事項（2.2.1 参照）。
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<T>, ShapeError>;

    /// 現在のテンソルが行優先で連続配置されているか判定する。
    pub fn is_contiguous(&self) -> bool;

    /// 非 contiguous な場合に、行優先連続バッファへ実体化した
    /// 新しい `Tensor` を返す（常にコピーを伴う明示 API）。
    /// contiguous な場合は自身の複製（`Arc` 共有のまま）を返す。
    pub fn contiguous(&self) -> Tensor<T>;

    /// self を target shape へブロードキャストした zero-copy view を返す
    /// （NumPy `broadcast_to` 相当。拡張された軸は stride 0。#12・TASK-1.4b）。
    /// 縮小方向・非互換 shape は `BroadcastIncompatible` を返す。
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor<T>, ShapeError>;

    /// 二項演算向け: 両テンソルを共通 shape へブロードキャストした
    /// view の組を返す（backend-cpu の elementwise カーネル・autodiff
    /// の入口が消費する想定。#12・TASK-1.4b）。
    pub fn broadcast_with(&self, other: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>), ShapeError>;

    /// ホスト可視の値を借用で読み出す（イシュー #1335。zero-copy view
    /// API の一員）。contiguous な場合は [`Self::as_slice`] の借用を
    /// そのまま `Cow::Borrowed` として返す（コピーなし）。非 contiguous
    /// な場合のみ [`Self::contiguous`] で 1 回だけ実体化し
    /// `Cow::Owned` として返す。`Var::host_view`（§3.1）・facade
    /// `VarHostView` の内部実装が使う想定の読み出し専用 API。
    pub fn host_slice(&self) -> std::borrow::Cow<'_, [T]>;
}

/// NumPy 互換のブロードキャスト後 shape を計算する（`broadcast.rs`）。
/// 末尾軸から比較し「両者同一」または「片方が 1」なら大きい方を採用する。
/// rank 差は短い方の先頭に 1 を補完する。不成立は `BroadcastIncompatible`。
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError>;
```

#### 2.2.1 未決事項: 非 contiguous な `reshape` の方針

`reshape` は contiguous 時のみ zero-copy が成立する。非 contiguous なテンソルに対する呼び出しをどう扱うかは 2 案あり、本イシューでは決定しない（ユーザー承認が必要な採否論点）:

- **案 A（エラー）**: `ShapeError` を返し、呼び出し側に明示的な `contiguous().reshape(..)` を要求する。NumPy の `reshape` は暗黙コピーを許容するため慣習からは外れるが、暗黙のパフォーマンス劣化（意図しないフルコピー）を防げる。
- **案 B（暗黙コピー）**: 内部で `contiguous()` を呼んでからコピーを伴う reshape を行う。NumPy 慣習（`ndarray.reshape` は非 contiguous でも動く）に近いが、AI エージェントが意図せず高コストなコピーを埋め込みうる。

安全側の初期実装方針としては案 A（エラー）を推奨する。理由: 本リポの差別化価値は AI 自律メンテナンス（REQ-3）であり、暗黙のコピー発生は性能回帰の原因を不透明にする。最終決定はユーザー承認を経ること。

### 2.3 要素型

```rust
/// テンソルが扱える要素型の最小抽象化（PoC-v2-1 の型パラメータ方針）。
/// `f32`/`f64`/`i32` に加え、GPU バックエンド（CUDA/Metal）で使用する
/// `half::f16` を実装対象とする。
///
/// `zeros`/`ones`（2.4）がジェネリックな `Tensor<T>` を返す（`Result`
/// 化後も型 `T` を問わず生成できる）ためには、`T` 自身が加法単位元・
/// 乗法単位元を生成できる必要がある。そのため `Element` は
/// `Copy + Send + Sync + 'static` に加え、ゼロ値・単位値の生成
/// capability を追加境界として要求する（`zero()`/`one()` 相当の
/// associated fn。具体的なシグネチャは TASK-1.4 productize 時に確定）。
pub trait Element: Copy + Send + Sync + 'static { /* 追加境界は上記ドキュメンテーションコメント参照 */ }
```

### 2.4 生成系

```rust
impl<T: Element> Tensor<T> {
    /// 実行時 shape 検査を行うコンストラクタ（PoC-v2-1 で確定した
    /// 方式）。`data.len()` が `shape` の要素数積と一致しない場合
    /// `ShapeError` を返す。
    pub fn new(data: Vec<T>, shape: &[usize]) -> Result<Tensor<T>, ShapeError>;

    /// shape の要素数積のオーバーフロー（`ShapeError::ElementCountOverflow`）
    /// またはアロケーション失敗時に `Err` を返す。§1「失敗しうる公開
    /// API は Result を返す」原則との整合のため `Result` 化する
    /// （issue #182 コメント 2026-08-05、レビュー Medium 指摘）。
    pub fn zeros(shape: &[usize]) -> Result<Tensor<T>, ShapeError>;
    /// `zeros` と同様の理由で `Result` を返す。
    pub fn ones(shape: &[usize]) -> Result<Tensor<T>, ShapeError>;
    /// `zeros` と同様の理由で `Result` を返す。
    pub fn full(shape: &[usize], value: T) -> Result<Tensor<T>, ShapeError>;
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Tensor<T>, ShapeError>;
}
```

### 2.5 REQ-10 との関係: rank を型に載せるか

REQ-10 の受け入れ基準は「rank を型に載せるか（`Tensor<T, const R: usize>` 等）は自作 API 設計時に決定し、ドキュメントに記録すること」と要求する（`docs/spec/04-requirements.md:222` 付近、TASK-10.2 対応）。本文書でこれを決定する。

**決定**: 基盤層の `Tensor<T>` は rank を型パラメータに含めない（実行時 rank）。理由は PoC-v2-1 の確定事項（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「テンソル型の設計判断」表）と同一で、safetensors/ONNX からロードする重みの shape・rank は実行時にしか決まらないため。const generics による型レベル shape 検証は、コンパイル時に shape が既知な層（固定サイズの Linear 層等）に限定した**後続レイヤー**（TASK-10.x）として、基盤 `Tensor<T>` の上に別途構築する。

この方針に伴う限界（REQ-10 が明記を要求する事項）:
- バッチ次元・シーケンス長など実行時に変動する次元は型レベル shape の対象に含めない（可変バッチ推論との衝突を避ける、v1 PoC-7 の教訓を踏襲）。
- 「値が偶然一致するケース」（例: バッチ数と特徴量数がたまたま同じ）は型でも実行時でも検出できない構造的な限界として残る。

## 3. autodiff 公開 API（勾配追跡の型分離）

### 3.1 型分離方式

`tensor-core::Tensor<T>` は**常に非追跡**とする。勾配追跡は `autodiff` クレートの別型 `Var` で表し、`Tape` 上の `NodeId` を保持する。`Tensor<T>` に対する演算はテープを一切構築せず、`Var` に対する演算のみがテープへ記録される。これにより「ON/OFF の型分離」がコンパイル時に保証される。

採用方式は PoC-v2-2 が確定した**動的テープ式**（Wengert list）である（`docs/spec/03-poc/poc-v2-2-autodiff/README.md:170`「採用方式の確定」）。

```rust
/// テープの識別子。プロセス全体で単調増加するカウンタから発行する。
/// ポインタ等値（`ptr::eq`）ではなく専用 ID を用いる理由: スコープ末
/// で破棄された `Tape` のメモリ領域は後続の `Tape::new()` に再利用さ
/// れうるため、ポインタ比較は別テープを同一と誤判定する（false
/// positive）余地が残る。単調増加 ID はプロセス生存中に衝突しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeId(u64);

/// 演算を記録する Wengert list。`Var` 上の演算のみがここに記録される。
/// ノード列は `RefCell` で包み、`Var` からの内部可変性による追記を
/// 可能にする（借用モデルの詳細は `Var` のドキュメント参照）。
pub struct Tape { id: TapeId, nodes: RefCell<Vec<TapeNode>> }

impl Tape {
    pub fn new() -> Tape;

    /// 非追跡の `Tensor<f32>` を、テープ上の葉ノードとして登録する。
    pub fn var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

/// テープ上の 1 ノードを指す追跡対象値。値そのものではなく
/// `NodeId` + テープへの参照を保持し、演算のたびにテープへ新しい
/// ノードを追加する。
///
/// **借用モデル**: PoC-v2-2 確定 API（`Tape::matmul(&mut self, ...)`、
/// `docs/spec/03-poc/poc-v2-2-autodiff/README.md:28-31`）は `&mut Tape`
/// を要求するが、本設計では `Var` 単体を式中で連鎖させたい
/// （`a.matmul(&b)?.relu()` 等）ため、`Tape` 内部を `RefCell` で
/// 包み、`Var` は共有参照 `&'t Tape` を保持する形へ変更する。
/// ノード追加は `RefCell::borrow_mut()` 経由の内部可変性で行う。
/// PoC-v2-2 が確定したのは「動的テープ式（Wengert list）」という
/// 記録方式そのものであり、呼び出し規約（`&mut` か内部可変性か）は
/// 本文書側で productize 時の API 使い勝手を優先し決定する
/// （TASK-1.5 実装時に借用チェッカ上の問題が出た場合は PoC-v2-2 の
/// `&mut Tape` 方式へ戻す判断もありうる。決定は実装時に確定する）。
///
/// **クロステープ安全性**: ライフタイム `'t` の一致は同一 `Tape` を
/// 指す証明にはならない（同一スコープに複数の `Tape` が存在する場合、
/// それぞれの `Var<'t>` は同一の `'t` を持ちうる）。そのため二項演算
/// （`matmul`/`add`/`mul`/`mse_loss`）・`Tape::backward`・
/// `Gradients::get` は入口で `self.tape.id` と相手側 `Var`／
/// `Gradients` が保持する `TapeId` の一致を実行時検査し、不一致なら
/// 無関係なノードを誤って解決せず `AutodiffError::TapeMismatch` を
/// 返す（型システムのみでは検出できない誤用のため、実行時 identity
/// 検査を必須の入口契約とする）。
pub struct Var<'t> { tape: &'t Tape, id: NodeId }

impl<'t> Var<'t> {
    /// 追跡を外し、現在の値を非追跡の `Tensor<f32>` として取り出す。
    /// テープ内部が `RefCell` のため `Ref<'_, Tensor<f32>>` を返す。
    ///
    /// **借用注意**: この `Ref` を保持したまま、同じ `Tape` に対して
    /// `borrow_mut()` を要する演算（`matmul`/`add` 等のノード追加）を
    /// 呼ぶと実行時 panic になる（`RefCell` の二重可変借用）。本番
    /// 経路で panic させないため、値をその場の参照ではなく所有値と
    /// して持ち出したい場合は代わりに `to_tensor()` を使うこと。
    pub fn value(&self) -> Ref<'_, Tensor<f32>>;

    /// `value()` の所有値版。`Tensor<f32>` へ複製して返すため `Ref`
    /// を持ち越さず、直後に同じ `Tape` へノード追加演算を呼んでも
    /// 借用エラー・panic が起きない（`value()` の借用注意を参照）。
    pub fn to_tensor(&self) -> Tensor<f32>;

    /// ホスト可視の値を読み出す（イシュー #1335）。`Tape` の
    /// `RefCell` 借用はこのメソッド内で解放し、返す [`VarHostView`]
    /// へ持ち越さない（P1 是正・codex-review 指摘。当初実装は
    /// `Ref<'_, [f32]>` をそのまま `VarHostView` へ持ち越しており、
    /// 保持中に同じ `Tape` へノード追加演算を呼ぶと `borrow_mut()`
    /// が実行時 panic した）。materialize した `Tensor<f32>` を
    /// [`Tensor::contiguous`]（contiguous な場合は `Arc` 複製のみで
    /// 安価・非 contiguous な場合のみ 1 回実体化）へ通してから
    /// [`VarHostView`] へ所有値として格納する。
    ///
    /// 返す `VarHostView` はライフタイムパラメータを持たず `Tape`／
    /// `RefCell` の借用を一切保持しないため、生存中に同じ `Var`／
    /// `Tape` へノード追加演算（`add`/`matmul` 等）を呼んでも panic
    /// しない（[`Var::value`] の借用注意とは異なる）。
    pub fn host_view(&self) -> VarHostView;

    // 演算セット（3.2 参照）。shape 不整合・不正なブロードキャスト・
    // 範囲外 dim はすべて失敗しうるため `Result<Var<'t>, AutodiffError>`
    // を返す（§1「すべての失敗しうる公開 API は Result<T, E> を返す」
    // 原則、`.claude/rules/coding-rust.md` の unwrap/expect 禁止に従う。
    // `tensor-core::Tensor` 側の対応 API（transpose/narrow/reshape 等）
    // が `Result<_, ShapeError>` を返すのとの対称性もこれで揃う）。
    //
    // `matmul`/`add`/`mul`/`mse_loss` は 2 つの `Var<'t>` を受け取る
    // 二項演算であり、`self.tape.id != other.tape.id` の場合は shape
    // 検査より前に `Err(AutodiffError::TapeMismatch)` を返す（`Var`
    // ドキュメントの「クロステープ安全性」参照。foreign な `NodeId`
    // を無関係なローカルノードへ解決させない）。
    pub fn matmul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError>;
    pub fn add(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError>;    // bias broadcast 含む
    pub fn mul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError>;
    pub fn sum(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError>; // dim が rank 範囲外なら Err
    pub fn max(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError>; // 同上
    pub fn mse_loss(&self, target: &Var<'t>) -> Result<Var<'t>, AutodiffError>; // TapeMismatch・shape 不一致を検査

    // relu/exp/tanh/sum/max は単一の `Var<'t>`（`self`）のみを操作す
    // る単項演算のため、クロステープ照合の対象外（テープ照合が要る
    // のは複数 `Var` を突き合わせる二項演算のみ）。relu/exp/tanh は
    // さらに shape を一切変えない要素ごとの演算であり、構造的に失敗
    // しえない（PoC-v2-2/v2-5 が実測した演算セットでも shape 変化を
    // 伴わない）ため「失敗しうる公開 API」の対象外として bare
    // `Var<'t>` を返す（原則は失敗しうる API のみに適用）。
    pub fn relu(&self) -> Var<'t>;
    pub fn exp(&self) -> Var<'t>;
    pub fn tanh(&self) -> Var<'t>;
}

/// [`Var::host_view`] が返す読み出しビュー（イシュー #1335）。
/// `Deref<Target = [f32]>` でスライスとして使う。内部には
/// `host_view()` 構築時に [`Tensor::contiguous`] 済みの所有
/// `Tensor<f32>` を保持する（`Tensor<f32>` 自体が `Arc<Storage<T>>`
/// を共有する値型のため、追加コピーなしに `Tape`／`RefCell` の借用
/// を持ち越さず切り離せる）。ライフタイムパラメータは持たない
/// （当初設計は `Ref<'a, [f32]>` を保持する `VarHostView<'a>` を
/// 想定していたが、`Tape` の借用保持による panic を是正した結果
/// 現行実装へ変更した。`Var::host_view` ドキュメント参照）。
pub struct VarHostView { /* private */ }

impl Tape {
    /// `loss` から逆伝播し、テープ上の全 `Var` に対する勾配を計算する。
    ///
    /// **クロステープ安全性**: 引数の型は `&Var<'_>` であり、ライフ
    /// タイムが消去される（elided）ため型システム単体では `loss` が
    /// `self` に属するテープ由来かを区別できない。入口で
    /// `loss.tape.id == self.id` を検査し、不一致なら
    /// `Err(AutodiffError::TapeMismatch)` を返す（foreign な
    /// `Var`／`NodeId` をこのテープのグラフへ誤って接続しない）。
    pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError>;
}

/// autodiff クレートの公開エラー型。順伝播（`Var` の演算メソッド）と
/// 逆伝播（`Tape::backward`）双方の失敗経路をここに集約する
/// （旧設計は `Tape::backward` の失敗のみを想定しており、順伝播側の
/// shape 不整合を表現するエラーが欠落していた）。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、TASK-1.5 で演算セット（3.2）
/// が拡張されるたびに新しい失敗要因（Conv 系の padding 不整合等）の
/// variant 追加を非破壊にするため。issue #182 承認コメントで明示された
/// のは `ShapeError`・`BackendError` の 2 enum だが、同一の設計一貫性を
/// 保つため本 enum にも追加適用する。
#[non_exhaustive]
#[derive(Debug)]
pub enum AutodiffError {
    /// 順伝播時の shape 不整合（`matmul`/`add`/`mul` の不正な
    /// ブロードキャスト、`sum`/`max`/`mse_loss` の shape 不一致等）。
    /// `tensor-core::ShapeError` をラップし、エラー種別を重複定義
    /// しない。
    Shape(ShapeError),
    /// `Tape::backward` 時のグラフ不整合（未接続ノードへの逆伝播要求等）。
    Backward(String),
    /// 二項演算（`matmul`/`add`/`mul`/`mse_loss`）・`Tape::backward`・
    /// `Gradients::get` に、識別子が異なる `Tape` に属する `Var` が
    /// 渡された。`Var` ドキュメントの「クロステープ安全性」参照。
    /// `Shape`/`Backward` へ折り込まず独立バリアントとするのは、
    /// 呼び出し側が「shape の不整合」と「そもそも無関係なテープの
    /// 値を渡した」というプログラミングエラーを区別できるようにする
    /// ため。
    TapeMismatch,
}

/// `backward()` の結果。発行元 `Tape` の `TapeId` を保持し、`get()`
/// で foreign な `Var` を受理しないための照合に使う。`NodeId` を
/// キーに勾配テンソルを保持する。
pub struct Gradients { tape_id: TapeId, /* ... */ }

impl Gradients {
    /// `var` が発行元 `Tape`（`self.tape_id`）に属さない場合は
    /// `Err(AutodiffError::TapeMismatch)` を返す（`var.tape.id` との
    /// 一致検査。`Var` ドキュメントの「クロステープ安全性」参照）。
    /// 同一テープ由来だがそのノードに勾配が存在しない場合（未使用の
    /// 葉ノード等）は `Ok(None)` を返す。「foreign な入力」と
    /// 「正当な入力だが結果が空」を型で区別する（§1「失敗しうる公開
    /// API は Result<T, E> を返す」原則）。
    pub fn get(&self, var: &Var) -> Result<Option<&Tensor<f32>>, AutodiffError>;
}
```

`no_grad` 相当（勾配追跡を一時的に止める）は、専用フラグ API を設けず「`Tensor<T>` のまま演算する」ことで表現する。`Tensor<T>` と `Var` は別型であるため、追跡なしの経路を選ぶことはコンパイル時に強制される。

### 3.1.1 学習ループにおける Tape のライフサイクル（イシュー #1048 で確定）

動的テープ式（Wengert list）では `Tape::var()` を呼ぶたびにノードが
`nodes: RefCell<Vec<TapeNode>>` へ追加される。学習ループ・reuse GEMM
（framework-compare の `run_gemm_reuse`／`run_train_reuse`）向けに、
以下 2 通りの運用のいずれかを選べる:

**運用 A（ステップごと `Tape::new()`）**:

1. パラメータ自体は非追跡の `Tensor<f32>` として学習ループの外
   （呼び出し側）で保持する。
2. 各ステップの冒頭で新しい `Tape::new()` を生成し、パラメータを
   `tape.var(&param)` で都度テープへ登録する。
3. 順伝播・`backward()` を実行し、`Gradients` から勾配を読み出して
   ステップ外で保持しているパラメータ（非追跡 `Tensor<f32>`）を
   更新する。
4. ステップ末で `Tape` をスコープアウトさせ破棄する。

**運用 B（`Tape::reset()` による再利用。#1048 で追加）**: `Tape::reset
(&mut self)` は、最初の演算（非葉ノード）が記録される*前*に登録した
葉ノードのみを保持したままノード列を切り詰める。`Tape::leaf_count()`
（保持される葉の個数）・`Tape::leaf(index)`（保持された葉 `index`
番目を `Var` として再取得。O(1)・コピーなし）と組み合わせ、同一
`Tape` を毎ステップ使い回せる:

1. ステップの外でパラメータ・入力用の `Tape` を 1 回だけ構築し、
   パラメータを `tape.var(&param)` で葉として登録する（この時点では
   まだ演算を行わない）。
2. 各ステップで `tape.leaf(index)` により登録済みの葉 `Var` を再取得
   し、順伝播・`backward()` を実行する。
3. ステップ末で `tape.reset()` を呼び、ステップ内で記録した演算
   ノード（結果ノードの `Tensor<f32>` を含む）を破棄する——`&mut self`
   を要求するため、reset 前に取得した古い `Var`/`ResidentLeaf` を
   reset 後も使う経路はコンパイル時に排除される（借用検査）。`Tape`
   を借用しない値（`Gradients`・`DeviceParamStore` の `pending`）は
   世代番号（`Tape::epoch()`）による実行時検査で reset 後の誤用を
   `Err(AutodiffError::TapeMismatch)`／`Err(BackendError::TapeMismatch)`
   として fail-closed に拒否する。
4. 次ステップの入力（バッチ・毎ステップのパラメータ葉等）は演算
   *後*に登録するため、葉プレフィックスに含まれず reset のたびに
   破棄される——step ごとの葉が無限蓄積することはない。

`Tape` を学習ループ全体で使い回しつつ `reset()` を一切呼ばない運用
（§2.1 のコメントが「学習ループの勾配バッファ」と述べているのは
`Storage` の再利用最適化の話であり、`Tape` 自体を使い回す意図では
ない点に注意）はノード列が際限なく肥大化するため引き続き非推奨とし、
`Tape` のドキュメンテーションコメントに明記する。運用 A・B いずれも
サポート対象であり、reuse GEMM・学習ループの性能が問題になる経路
（framework-compare の reuse ベンチが動機。#1048）では運用 B を推奨
する。

### 3.2 演算セット（初期）

PoC-v2-2 と PoC-v2-5 で実測済みの演算に合わせる（`docs/spec/03-poc/poc-v2-2-autodiff/README.md:28`、`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md:49` 付近の `MetalOps` 公開 API 一覧）:

`matmul`・`add`（bias broadcast 含む。逆伝播は reduction）・`mul`・`relu`・`exp`・`tanh`・`sum`・`max`・`mse_loss`。

演算セットの拡張（Conv 系・Softmax 等）は後続タスク（TASK-1.5 以降）へ委譲する。

**TASK-9.1b（#92）追加**: `sigmoid`（`1 / (1 + exp(-x))`。数値安定形）を
基本活性化関数群の一つとして追加した。PoC-v2-2／PoC-v2-5 の実測演算
セットには含まれないため、追加時は既存の数値微分突合（`grad.rs`）と
end-to-end backward テスト（`tests/nn_activation.rs`）で受け入れ条件
を独自に検証した。`BackendOps`（§4.2）への `sigmoid` 追加は本タスクの
スコープ外とし、`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/`max` と同様に
CPU 参照実装（`eval.rs`）止まりとする（バックエンド接続差し替え時に
まとめて対応）。

**イシュー #1047 追加**: `reshape`/`transpose`（view 系ノード）を
追加した。既存の演算セット（`push_eager`／`push_lazy` の 2 経路）とは
別に、ホスト値を一切保持せず backward 時に入力から再導出する第 3 の
登録経路（`Tape::push_view`／`tape::resolve_view`。burn-autodiff の
`MemoryBound { retro_forward }` 相当）を導入した。`tensor-core::Tensor`
の `reshape`／`transpose` と同じく zero-copy（`Arc<Storage>` 共有）で
あり、中間バッファを持たない（設計判断・メモリ実測は `docs/
autodiff-view-recompute-decision.md` を参照）。非 contiguous な
`reshape` は案 A（エラー。`tensor-core::Tensor::reshape` と同じ方針）を
踏襲する。

**イシュー #1647 追加**: `rnn_cell`／`lstm_cell`／`gru_cell`（RNN・
LSTM・GRU の 1 step セル演算。設計 `docs/autodiff-rnn-cell-tape-
design.md`）を内部クレート `fandhe_ai_autodiff::var::Var` の演算として
追加した。専用 `Op` variant（`RnnCell`／`LstmCell`／`LstmHidden`／
`GruCell`）で表現し、GEMM 部分は既存の `matmul`／`gemm_bias_act` を
再利用、ゲート pointwise 演算のみ新設の `BackendOps` メソッド
（`lstm_pointwise`／`lstm_hidden_backward`／`lstm_cell_backward`／
`gru_pointwise`／`gru_backward`。§4.2 参照）で計算する。融合対象外
（非 elementwise。`Op::is_lazy_elementwise` 参照）。`fandhe_ai_autodiff::
nn::rnn`（`RnnCell`／`LstmCell`／`GruCell`・Sequence レベル `Rnn`／
`Lstm`／`Gru`）が `nn::Linear` と同型の構成でこれらを組み上げる。
`facade`（公開 API 面）への昇格は本イシューのスコープ外
（`docs/compat-api-scope.md` §5 の範囲拡張手続き・決定 10）。

**イシュー #1621 追加**: 線形代数（`inv`／`solve`／`det`／`cholesky`／
`qr`／`svd`）・`matrix_norm` を追加した（REQ-9 2026-09-12 追記・
`docs/spec/04-requirements.md:232` の Tier 2 列挙対応）。既存演算と
異なり全て rank-2 限定（バッチ次元は対象外）で、`push_eager` の非
elementwise ノードとして記録する。`qr`／`svd` はテープが 1 ノード
1 出力の制約のため出力ごとに別ノード（`QrQ`／`QrR`、`SvdU`／`SvdS`／
`SvdVh`）を積む多出力設計を採る（コタンジェントに線形なことを利用し
各ノードが部分寄与を返し `Tape::backward` が合算する。`view_node_fan_
out_accumulates_gradient` と同じ蓄積機構）。CPU は実装済み、CUDA／
Metal は既定 `Unsupported`（`BackendOps::linalg_*` は非破壊拡張の
デフォルトメソッド。`gemm_checksum` と同型）。設計判断・数値契約
（`f64` 内部計算・符号／ゲージ規約・エラー分類）の詳細は `docs/
autodiff-linalg-design.md` を参照。

## 4. backend 入口公開 API

### 4.1 デバイス選択

```rust
/// 実行先デバイス。`Metal` は `cfg(target_os = "macos")` でのみ
/// 存在する（PoC-v2-5 実証構成、feature フラグは設けない）。
/// `cudarc` は無条件依存 + 動的ロードのため `Cuda` variant は
/// 全 OS で存在するが、実行時に CUDA ドライバ不在なら
/// `BackendError` を返す（4.4 参照）。
pub enum Device {
    Cpu,
    Cuda(usize),
    #[cfg(target_os = "macos")]
    Metal,
}

impl Device {
    /// 実行時に利用可能なデバイスを検出して返す。
    pub fn available() -> Vec<Device>;
}
```

**既定選択の方針（未決事項。ただし CPU 分は下記の承認済み改訂で確定済み）**: CUDA を既定で有効化するかどうかの具体的な構成決定は REQ-2 でも未検証のまま残っている（`docs/spec/04-requirements.md` REQ-2 受け入れ基準「バックエンド有効化構成」の項）。GPU バックエンド（CUDA／Metal）を既定で使う規則は本文書では確定しない。TASK-1.9 実装時にユーザー承認を得て決定すること。

**構成変更（10 クレート化・`facade` composition root への集約。イシュー #52・spec PR #53 マージ済み。第 16〜18 波が到達した「`autodiff` 内結線モジュール」構成は撤回済み）**: `docs/fusion-graph-design.md` §1 に第 16〜18 波の設計変遷・撤回理由の詳細を記録する（本節は結論のみを記す）。第 18 波の codex-review は、9 クレート構成では「利用者に融合制御を許さず（REQ-12）・`autodiff` を `BackendOps` 抽象に限定したまま」既定バックエンド供給の composition root を置く場所が存在しないことを指摘し、仕様側でのクレート構成変更を要求した。**ユーザー承認（spec イシュー #52 → spec PR #53 マージ済み、`docs/spec/04-requirements.md` REQ-1・REQ-9・REQ-12 の 2026-08-08 追記・`docs/spec/05-tasks.md` TASK-9.3・TASK-9.4・TASK-2.5 の 2026-08-08 追記）により、10 クレート目として `facade`（composition root・compat 公開面の 2 責務。`docs/spec/04-requirements.md:50`）を新設する構成へ確定した**。

**`facade` の 2 責務**: (1) composition root——`Device` 識別子から具体 `BackendOps` を構築・結線する。依存形状は `bench-harness` と同型（`backend-cpu` 通常依存・`backend-cuda` は `cudarc` 動的ロードで無条件・`backend-metal` は `cfg(target_os = "macos")` 分離。`docs/spec/05-tasks.md:316` TASK-9.3）。(2) compat 層（REQ-9）の利用者向け公開面（`docs/spec/04-requirements.md:209` の 2026-08-08 追記・`docs/spec/05-tasks.md:322` TASK-9.4）。

**サポート境界の宣言（`docs/spec/04-requirements.md:210` の 2026-08-08 追記）**: `facade` が唯一のサポートされる公開 API 面であり、`tensor-core`・`autodiff`・`backend-*` は内部クレート（直接利用は非サポート）である。この宣言により、`autodiff` が `facade` 向けに持つ ops 受け取り構築子（技術上 `pub`）は、サポート外の内部 API であり REQ-12 の「利用者向け融合制御 API」に該当しないと整理する。

**実装状況（イシュー #410・TASK-9.3。2026-08-09 追記）**: 上記 (1) composition root は実装済み（`crates/facade/src/lib.rs`）。公開関数は `fandhe_ai::tape()`（既定 CPU）・`fandhe_ai::tape_for(Device) -> Result<Tape, BackendError>`（明示指定）の 2 つのみで、`Tape`・`BackendOps` は再エクスポートしない（`crates/facade/tests/api_surface.rs` が機械的に固定）。`Device::Cuda(_)`／`Device::Metal` の構築規則（下記段落）はこの実装で確定した。**(2) compat 公開面は TASK-9.4（イシュー #411）で実装済み**（`crates/facade/src/compat/`。旧 `fandhe_ai_autodiff::compat` からの移設。詳細は下記「互換 API の再改名」節に続く「compat 層の facade 移設」節、および `docs/compat-api-scope.md` §0・§4.2 を参照）。

**承認済み内容と結線の実装場所（本節が定める `Device` 列挙・`Device::available()` による明示選択という契約自体は変更しない）**: 2026-08-08、AskUserQuestion によりユーザーが**承認した内容**は「既定バックエンドを `Device::Cpu`（`backend-cpu` 実装）とすること」である（この一文は変更しない）。**構成上の結論**は、`facade` の composition root（TASK-9.3）が `Device::Cpu` から `backend-cpu` の `CpuBackendOps` を構築し、`fandhe_ai_autodiff::Tape::new_with_ops(ops: Box<dyn BackendOps + Send>) -> Tape`（性能を伴う経路の公開コンストラクタ。渡された `ops` をそのまま格納する非 fallible 関数。`docs/fusion-graph-design.md` §1。codex-review 第 22 波・PR #403 の P1 是正で `Tape::new(ops)` から改名。無引数の compat 経路 `Tape::new()`／`Tape::default()` は下記「互換 API の別名復元」節参照）へ渡す、という形に確定した。`autodiff` は `Device` 型にも `backend-cpu`／`backend-cuda`／`backend-metal` のいずれの具体クレートにも依存しない。`facade` を経由して構築した `Tape` はすべて CPU 上での融合実行が既定で・無条件に・透過的に効く。承認の記録場所は本節（本段落）および `docs/fusion-graph-design.md` §6.2「既定バックエンドの供給規則」。この承認は「本節が定める `Device` の既定デバイス選択ロジックを実装する」ことを意味しない——`Device::available()` による列挙・明示選択という契約自体は本節のとおり変更しない。**GPU バックエンド（CUDA／Metal）を既定にするかどうかはこの承認の対象外であり、REQ-2 の 27 組再検証後に別途ユーザー承認を得て決定する**（上記「既定選択の方針」の未決事項は継続する）。`facade`（TASK-9.3）は `Device::Cuda(_)`／`Device::Metal` を明示指定された場合の結線も担うが、その具体的な構築規則は TASK-9.3 の実装時にユーザー承認を得たうえで確定する。

**破壊的変更（明記して許容する）**: 出荷済みの無引数 `Tape::new() -> Tape`・`impl Default for Tape` は、`Tape::new(ops: Box<dyn BackendOps + Send>) -> Tape` への差し替えに伴い削除される破壊的変更である（`docs/fusion-graph-design.md` §1）。この破壊は REQ-9 の 2026-08-08 追記（`facade` を唯一のサポートされる公開 API 面とし、`autodiff` は直接利用が非サポートの内部クレートとする宣言）を根拠に許容するが、実装コミットは `.claude/rules/conventional-commits.md` の `!`／`BREAKING CHANGE:` 告知を省略しない。`Tape: Debug`・`Tape: Send` という公開契約自体は変更しない。`BackendOps` trait はスーパートレイトの追加を一切受けない（`Tape` が所有する `ops` フィールドの型が `Box<dyn BackendOps + Send>` である点は `Tape: Send` を維持するための trait object 型 bound であり、`BackendOps` を実装する既存クレートのコードには影響しない。詳細は `docs/fusion-graph-design.md` §3.4「`BackendOps` trait 定義自体は変更しない」）。

**移行手順（codex-review 第 19 波・PR #403 の P1 指摘を受け追記）**: なぜ「無引数コンストラクタを残す」「既定バックエンド解決経路を用意する」代替案を採らないかを明記する——`autodiff` は `tests/architecture_boundaries.rs` の `autodiff_cargo_toml_does_not_depend_on_concrete_backends`／`autodiff_src_does_not_reference_concrete_backend_crates` により具体バックエンドクレート（`backend-cpu`／`backend-cuda`／`backend-metal`）への依存を CI で機械検査済みに禁止しているため、`autodiff` 内で既定バックエンドを解決する経路はこの不変条件と両立しない（上記「経緯」節の第 16〜18 波が撤回した構成そのものに戻ってしまう）。唯一の解決先である `facade`（composition root。TASK-9.3）は本イシュー時点で未実装のため、性能を伴う既定バックエンド解決の移行手順は「呼び出し元が具体 `BackendOps` 実装を明示的に渡す」以外にない（`facade` はイシュー #410 で実装済み。上記「実装状況」段落参照）。

呼び出し元の移行例（`backend-cpu` を使う場合。性能が必要な経路）:

```rust
// 変更前（削除済み）
let tape = Tape::new();
// または
let tape = Tape::default();

// 変更後（codex-review 第 22 波・PR #403 の P1 是正で `Tape::new_with_ops` へ改名。
// 無引数 `Tape::new()` は下記「互換 API の別名復元」節のとおり compat 経路として復元済み）
let tape = Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()));
```

`compat::Sequential::predict` も同様に、当時（本節記述時点・`facade` 未実装下）は性能が必要な経路として `ops: Box<dyn BackendOps + Send>` を渡す `predict_with_ops` を使う設計だった:

```rust
// 変更前
let output = model.predict(&input)?;

// 変更後（性能が必要な経路。当時の設計。下記「compat 層の facade 移設」節で撤去済み）
let output = model.predict_with_ops(&input, Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))?;
```

**この `predict_with_ops` は TASK-9.4（イシュー #411。2026-08-09 追記）で公開面から撤去した**。詳細は下記「compat 層の facade 移設（TASK-9.4）」節を参照。

**互換 API の別名復元（codex-review 第 19〜21 波・PR #403 の P1 是正。2026-08-08 追記）**: 上記「唯一の解決先である `facade` は未実装」という制約自体は変わらないが、codex-review は「既存の無引数コンストラクタ／`Default`／`predict(&Tensor)` を維持したまま解決するか、互換 API を別名で残すこと」を要求し、ドキュメントのみの正当化（本節上記の記述）では P1 判定を解除しなかった（PR #403 第 19〜21 波で同一指摘が再送された）。実装を精査した結果、`autodiff` は forward 値計算の naive 参照実装（`eval.rs`。`backend-cpu`／`backend-cuda`／`backend-metal` のいずれにも依存しない、クレート内部の暫定 CPU 実装）を既に保有しており（`test_support::TestOps` が `#[cfg(test)]` 限定でこれに委譲する形で先行使用していた）、これを本番ビルドへ昇格させても `tests/architecture_boundaries.rs` の不変条件（具体バックエンドクレート非依存）に抵触しない。したがって以下の compat API を追加した（`crates/autodiff/src/default_ops.rs` が実装。`NaiveOps`）:

- `impl Default for Tape`: `Tape::new()` に委譲する無引数構築。
- `compat::Sequential::predict(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError>`: 無引数 `ops` 版として復元。内部で `predict_with_ops(input, default_ops::naive_ops())` に委譲する薄いラッパー。従来の 2 引数版は `predict_with_ops` へ改名した。

**性能特性の注意**: `NaiveOps` は融合実行（`FusionPlan`）を経ない逐次実装であり、`backend-cpu` の最適化カーネル（`rayon` 並列・BLIS ブロッキング）と同等の性能を持たない。この compat 経路は「デフォルト値でも動く」ことの保証に徹し、性能が必要な呼び出し元は引き続き `Tape::new_with_ops(ops)`／`Sequential::predict_with_ops` を使う。`facade`（TASK-9.3）実装後もこのフォールバック経路自体は維持する（`facade` は最適化済み `ops` を組み立てて渡す上位レイヤーであり、本 compat 経路と競合しない）。

**互換 API の再改名（codex-review 第 22 波・PR #403 の P1 是正。2026-08-08 追記）**: 上記「互換 API の別名復元」節で追加した `Tape::new(ops)`（ops 必須のまま）＋別名 `Tape::default()` という組み合わせでは、codex-review は「既存の無引数 `Tape::new()` シグネチャそのものを破壊している」という P1 判定を解除しなかった（PR #403 第 22 波で同一指摘が再送された。`Default` の追加は `Tape::new()` という呼び出し式自体のソース互換性を回復しないため）。そこで `Tape::new(ops)` を `Tape::new_with_ops(ops)` へ改名し、`Tape::new()`（無引数）を出荷済みシグネチャのまま compat 入口として復元した。`impl Default for Tape` は `Tape::new()` に委譲する薄いラッパーへ変更した（`Tape::new_with_ops` への直接委譲から変更）。`Sequential::predict`／`predict_with_ops` の命名（無引数版が既定名・明示 `ops` 版が `_with_ops` 接尾辞）と対称になる形に揃えている。

**影響範囲の確認**: `Tape::new_with_ops(ops)` の呼び出し元は `autodiff` クレート自身のテスト・`onnx-interop`／`guardrail`／`self-repair` の学習ループ fixture に限られ、`Tape::new(ops)` → `Tape::new_with_ops(ops)` の改名コミットで全呼び出し元を新シグネチャへ追従済み（`cargo test --workspace` 全 green で確認）。上記の compat API 追加（`Default for Tape`／`Sequential::predict`／無引数 `Tape::new()`）はこれらの呼び出し元に変更を要求しない加算のみの変更である。`facade`（TASK-9.3）実装後は composition root が `Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new())` に相当する結線を代行し、利用者が直接 `ops` を組み立てる必要はなくなる（本節冒頭「承認済み内容と結線の実装場所」参照）。

**compat 層の facade 移設（TASK-9.4・イシュー #411。2026-08-09 追記）**: 上記の各節は `facade`（TASK-9.3）未実装下での compat 層（`fandhe_ai_autodiff::compat`）の設計判断を記録したものである。`facade` 実装完了（イシュー #410）を受け、TASK-9.4（#411）で compat 公開面（`compat::array`／`compat::Sequential`）を `fandhe_ai_autodiff::compat` から `fandhe_ai::compat`（`crates/facade/src/compat/`）へ移設し、あわせて以下を変更した。

- **`Sequential::predict` の既定結線先を変更**: 旧 `fandhe_ai_autodiff::compat` 版は `default_ops::naive_ops()`（融合を経ない逐次 `NaiveOps`）へ委譲していたが、`fandhe_ai::compat` 版は `fandhe_ai::tape()`（composition root・`CpuBackendOps`・融合有効）へ委譲する。「`facade` 経由なら既定バックエンドが透過的に効く」（本節冒頭「承認済み内容と結線の実装場所」）という方針と、compat 経由の `predict` を初めて整合させた変更である
- **`predict_with_ops`（任意 `BackendOps` 注入経路）を公開面から撤去**: `fandhe_ai::compat::Sequential` は `predict_with_ops` を持たない（破壊的変更）。理由は REQ-12「任意 `BackendOps` 実装を注入できる公開 API を設けない」・`crates/facade/tests/api_surface.rs` の機械検査（`pub fn` が `BackendOps` を直接受け取ることを禁止）との整合。ops を明示的に選びたい内部用途は `Sequential::forward(&Tape, &Var)`（`BackendOps` を受け取らない）へ呼び出し元が任意に構築した `Tape` を渡せば足りる
- **`autodiff` 自身の無引数 compat 経路（`Tape::new()`／`Tape::default()`／`default_ops::NaiveOps`）は変更なし**: これらは `autodiff` クレート単体で「デフォルト値でも動く」ことを保証する経路であり、`fandhe_ai::compat::Sequential::predict` の結線先変更とは独立に維持する
- 詳細な設計判断・出典は `docs/compat-api-scope.md` §0（サポート境界）・§4.2（TASK-9.4 移設確定）を正とする（本節は概要のみ）
- **移行期間中のソース互換シム（codex-review PR #424 P1 是正・2026-08-09 追記）**: `fandhe_ai_autodiff::compat` モジュール自体の削除は、`fandhe_ai_autodiff::compat::{array, Sequential, SequentialVars}` を利用する既存コードを互換 shim なしに破壊する破壊的変更だったため、`crates/autodiff/src/compat/` に `#[deprecated]` を付与した非推奨シムとして実装を複製して残した。詳細は `docs/compat-api-scope.md` §4.3 を正とする
- **`predict_with_ops` の再復元（codex-review PR #424 P1 是正・2 巡目・2026-08-09 追記）**: 上記の非推奨シムから `predict_with_ops` を意図的に除外していたが、REQ-12 が制約するのは 0 節「サポート境界」が定める唯一のサポート対象公開 API 面（`facade`）であり、内部クレート `autodiff` 上の移行期間中シムは対象外という区別を codex-review が P1 として指摘した。`crates/autodiff/src/compat/sequential.rs` へ `predict_with_ops`（`Box<dyn BackendOps + Send>` 引数版）を `#[deprecated]` 付きで復元し、`predict` はこれへ委譲する。`fandhe_ai::compat::Sequential` 側は引き続き `predict_with_ops` を持たない（476 行目のとおり REQ-12 はここで効く）。詳細は `docs/compat-api-scope.md` §4.4 を正とする

**バージョニング**: 本判断の時点では、ワークスペース全体で `version = "0.3.0"`（`Cargo.toml`）を用いつつも crates.io 未公開の内部開発版であり、`docs/deps-policy.md`・CI（`deny.toml` の `sources = crates.io 限定`）が示すとおり外部への配布物は存在しなかった。したがって「正式な破壊的リリース」としてのバージョン切り上げ・移行ガイド配布は該当せず、上記の移行手順を本文書と `docs/fusion-graph-design.md` §1 に記録することを本リポジトリでの等価な措置とした。その後 2026-08-23 に v0.3.0 として crates.io へ初回公開済みであり、公開後の公開 API 変更は `docs/crates-io-publishing-order.md` の版数運用（semver・`workspace.version` 一括バンプ）に従う。本段落の前半は公開前の判断記録として残す。

**TASK-1.9a（#44）実装時の突合結果**: `Device`（`Cpu`／`Cuda(usize)`／`Metal`）は本節のシグネチャをそのまま `crates/tensor-core/src/device.rs` に実装した。以下は本文書からの拡張・保留であり、実装コメントにも同旨を記載している。

- `Device::available()` は `tensor-core` から 3 バックエンドクレートを直接参照できないため実装せず、複数 `DeviceProvider`（新規追加。下記）を横断する `enumerate_all(providers: &[&dyn DeviceProvider]) -> Vec<DeviceInfo>` を同等機能として提供した。集約入口（`Device::available()` をどの層で結線するか）は TASK-1.9c（#46）では対象外とし、TASK-1.9d（#47）でも「3 バックエンド統合テストの整備」という受け入れ条件には不要と判断し対象外とした（実装は別途追跡）。
- 3 バックエンドが「同一 trait でデバイス列挙・選択できる」（#44 受け入れ条件）ための入口として `DeviceProvider` trait（`backend_name`／`is_available`／`enumerate`／`select`）と `DeviceInfo`（`device`／`name`／`total_memory_bytes`／`compute_units`。`#[non_exhaustive]`）を新規追加した。本文書は §4.2 の `BackendOps`（カーネルディスパッチ）のみを定義しており、デバイス検出・選択専用の trait は記載していなかった。
- 既定デバイス選択ロジック（本節の未決事項）は本イシューでも実装しない（列挙と明示選択のみを提供する。ユーザー承認が必要な事項のため自動運転では安全側に倒した）。

### 4.2 カーネル入口トレイト案

PoC-v2-5 の `MetalOps` 公開 API（`gemm`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/`max`、`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）と対称な safe API を、バックエンド抽象層のトレイトとして定義する。`unsafe`/FFI は各バックエンド実装内部（`cudarc`・`objc2` 系呼び出し境界）に閉じ込める。

**デバイス常駐バッファ**: `Tensor<T>` の `Storage` はホスト（`Vec<T>`）に固定されている（2.1）。`BackendOps` の各メソッドが毎回 `&Tensor<f32>`（ホスト常駐）を受け取り `Tensor<f32>`（ホスト常駐）を返す素朴な形にすると、CUDA/Metal 経路では演算ごとにホスト⇔デバイス転送が発生し、複数演算を連鎖する GPU ワークロード（REQ-2・REQ-11 が前提とする行列演算ユニット活用）で転送コストが支配的になる。これを避けるため、デバイス常駐バッファを表す型を導入し、転送は明示 API（`upload`/`download`）に閉じ込める:

```rust
/// デバイス上に確保されたバッファへの不透明ハンドル。中身は各
/// バックエンド実装（`backend-cuda`/`backend-metal`）が保持し、
/// `tensor-core`/backend 入口からは shape・dtype・所属 `Device` の
/// メタデータのみが見える。ホスト `Tensor<T>` とは明確に別型とする
/// ことで、「今どちらに実データがあるか」を型で追跡できるようにする。
pub struct DeviceBuffer<T: Element> {
    device: Device,
    shape: Vec<usize>,
    /// 実データ（CUDA デバイスポインタ／Metal `MTLBuffer` 等）を指す
    /// 不透明ハンドル。具体型は各バックエンド実装（`backend-cuda`/
    /// `backend-metal`）内部で定義し、`tensor-core`/backend 入口から
    /// は中身を読めない。ここで具体型を確定しない理由・寿命管理の
    /// 決定時期は 6-5 を参照（TASK-1.9 実装時に確定）。
    handle: BackendHandle<T>,
}

/// 各バックエンド（CPU/CUDA/Metal）が実装するカーネル入口。
/// 公開 API はすべて safe。`unsafe` は実装側の FFI 境界に限定する
/// （PoC-v2-4/v2-5 の設計方針を踏襲）。演算は `DeviceBuffer` 同士で
/// 完結し、ホスト往復は `upload`/`download` を呼んだ箇所にのみ
/// 発生する。
pub trait BackendOps {
    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError>;

    fn gemm(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;

    // elementwise
    fn add(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn mul(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn relu(&self, a: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn exp(&self, a: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn tanh(&self, a: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;

    // reduction
    fn sum(&self, a: &DeviceBuffer<f32>, dim: Option<usize>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn max(&self, a: &DeviceBuffer<f32>, dim: Option<usize>) -> Result<DeviceBuffer<f32>, BackendError>;
}
```

`BackendError::DeviceMismatch` は、`gemm`/`add` 等に異なる `Device` に属する `DeviceBuffer` を渡した場合に返す（`DeviceBuffer::device` フィールドで検査可能）。この設計は PoC-v2-5 の `MetalOps` 実測 API（`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）をそのまま転記したものではなく、本イシューで追加した拡張である旨を明記する。`DeviceBuffer` 内部表現（`BackendHandle<T>` の具体型・CUDA デバイスポインタのラップ方法・Metal `MTLBuffer` の寿命管理等）は TASK-1.9 実装時に確定する（6-5 参照）。

**`BackendOps` が f32 専用である理由と f16 経路**: `DeviceBuffer<T: Element>` は `Element`（`f32`/`f64`/`i32`/`half::f16`、2.3 参照）全体に対しジェネリックだが、`BackendOps` トレイト v1 のスコープは PoC-v2-5 実測 API（`MetalOps` は f32 のみ実測済み）に合わせて `f32` 固定とする。GPU 推論で使う `half::f16` 経路（許容依存 `half`、deps-policy.md）の入口は本文書では確定しない。`BackendOps` を `T: Element` でジェネリック化するか、`f16` 専用の並行トレイトを追加するかは TASK-1.9 実装時に決定する（6-8 参照）。**イシュー #1648 で決定**（2026-09-13。`docs/backend-dtype-dispatch-design.md`）: `BackendOps` へ既定 `None` を返す capability accessor（`typed_ops_f64`／`typed_ops_f16`／`typed_ops_bf16`）を非破壊追加し、dtype ごとの演算本体は新設 `TypedOps<T: Scalar>` trait に集約する方式を採用。実装は #1649（CPU）→ #1650（CUDA）→ #1651（Metal）が担う。

**TASK-1.9b（#45）実装時の突合結果**: `DeviceBuffer<T: Element>`・`upload`/`download` は本節の設計をそのまま `crates/tensor-core/src/buffer.rs` に実装したうえで、`BackendOps` からメモリ操作のみを切り出した `MemoryOps` トレイト（`alloc_zeroed`/`upload`/`download`。f32 固定）を新規追加した。`BackendOps`（`gemm`/`add`/... のカーネルディスパッチ）自体の実装は TASK-1.9c（#46）へ引き継ぐ。以下は本文書からの拡張・確定である:

- `BackendHandle<T>`（本節記載の型パラメータ付き不透明ハンドル）ではなく、`Box<dyn BufferHandle>`（`BufferHandle: Debug + 'static` の object-safe trait、`as_any()` による `Any` ダウンキャスト経由）を採用した（`device.rs` の `&dyn DeviceProvider` 依存逆転構成と同型）。`downcast_handle::<H>()` で各バックエンドが自身の具体型（`CpuBufferHandle`/`CudaBufferHandle`/`MetalBufferHandle`）のみを取り出せる。
- 解放は RAII に一本化した（明示 `free()` API は設けない）。各バックエンドの具体ハンドルが `Vec<f32>`（CPU）／`CudaSlice<f32>`（CUDA。内部で `Arc<CudaStream>` を co-own し `Drop` でストリーム上に解放）／`Retained<MTLBuffer>`（Metal、既存 `MetalBuffer` 経由）の `Drop` に解放を委ねる。
- `numel == 0`（空テンソル）は FFI を呼ばず空ハンドルで表現する統一契約とした（Metal の zero-length 確保拒否・CUDA の 0 バイト `cuMemAlloc` 拒否環境との衝突を避けるため）。
- CUDA の `download` は `clone_dtoh`（内部で `cuMemcpyDtoHAsync` を発行する非同期コピー）の直後に `stream.synchronize()` を挟み、「`download` 復帰時点でホストデータ確定」を全バックエンド共通の同期契約とした。Metal は `StorageModeShared`（UMA）のため追加の同期は不要。
- `BackendError` に `TransferFailed(String)`（TASK-1.9b で追加。確保済みバッファへのコピー失敗を表す。`DeviceAllocationFailed` は確保自体の失敗と区別する）を追加した（4.4 参照）。
- f16（`half::f16`）転送経路の入口設計は本イシューでも決定しない（6-8 の未決事項のまま。`.claude/rules/out-of-scope-tracking.md` に沿い後続イシューとして提案する）。

**TASK-1.9c（#46）実装時の突合結果**: `BackendOps`（`crates/tensor-core/src/backend_ops.rs`）は上記案の 8 メソッド（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`）をそのまま実装したが、実装開始時点（`git fetch origin main`）で TASK-1.9b（#45。`DeviceBuffer`／`upload`／`download`）が未着地だったため、各メソッドのシグネチャを `&DeviceBuffer<f32>` ではなくホスト常駐 `&Tensor<f32>` を受け取り `Tensor<f32>` を返す形に差し替えた。受け入れ条件（TASK-1.9c）は「同一コードで 3 バックエンドのカーネルが呼び分けられる」であり、既存カーネル入口（CPU `gemm_blis_parallel`・CUDA `CudaGemm::run_tiled_f32`・Metal `MetalGemm::dispatch_auto`）がいずれもホスト常駐 `&[f32]` を受け取り内部で H2D／D2H 転送を完結させる契約のため、`DeviceBuffer` なしで本条件を満たせると判断した。`DeviceBuffer` 版シグネチャへの移行（`upload`／`download` の追加）は #45 マージ後、`BackendOps` の非破壊拡張として TASK-1.9d（#47）以降で検討する。CUDA／Metal は本イシュー時点で GEMM カーネルのみ実装済みのため、elementwise・reduction は `BackendError::Unsupported`（新規追加。`#[non_exhaustive]` のため非破壊）を返す fail-safe 実装とした。複数バックエンド横断のディスパッチ入口は `device::select_from`（TASK-1.9a）と同型の注入式 `backend_ops::ops_for(ops: &[&dyn BackendOps], device: Device) -> Result<&dyn BackendOps, BackendError>` として実装した（`Device::available()` の集約結線を上位層に委ねる設計を踏襲）。

**TASK-12.1f（#203）実装時の突合結果**: `BackendOps` に `Activation` enum（`None`／`Relu`。`#[non_exhaustive]`）と `gemm_bias_act(&self, a, b, bias: Option<&Tensor<f32>>, act: Activation) -> Result<Tensor<f32>, BackendError>` を **デフォルトメソッド**として非破壊追加した（GEMM epilogue〈bias 加算・activation〉融合。CUTLASS 系実測で平均 1.38〜1.45 倍が動機）。デフォルト実装は `gemm` → （`bias` があれば）`add`（行方向ブロードキャスト）→ `act` に応じた activation メソッドの 3 段合成であり、CPU バックエンド（`backend-cpu::ops::CpuBackendOps`）のみカーネル内融合実装（`gemm_blis_bias_act_parallel`）でこれをオーバーライドする。CUDA／Metal は本イシュー時点で elementwise 未実装（`Unsupported`）のためデフォルト実装へフォールバックし、`bias`／`act` 指定時は `Unsupported` を透過的に返す（GPU カーネル内 epilogue 融合は本イシューのスコープ外。`docs/perf/cpu-gemm-epilogue-fusion.md`「スコープ外」節参照）。実測記録: `docs/perf/cpu-gemm-epilogue-fusion.md`。

**イシュー #1335 実装時の突合結果（`MemoryOps::with_host_view`）**: `MemoryOps`（4.2 冒頭「TASK-1.9b 実装時の突合結果」参照）へ `with_host_view(&self, buffer: &DeviceBuffer<f32>, f: &mut dyn FnMut(&[f32])) -> Result<(), BackendError>` をデフォルト実装付きで非破壊追加した（`BackendOps::memory_ops(&self) -> Option<&dyn MemoryOps>` が `&dyn MemoryOps` を返す object-safe 契約〈4.2 冒頭〉を保つため、ジェネリック戻り値 `R` を持たずクロージャへ結果を渡す形にした）。

- **同期契約**: [`Self::download`] と同一（復帰時点でデバイス側の書き込みが完了していること）。既定実装は `download` を経由するフェイルセーフ（コピーを 1 回伴う）。
- **CPU 実装**（`backend-cpu::CpuMemory`）: 既定実装を上書きし、`CpuBufferHandle::data`（既存ホストバッファ）をそのまま借用する（コピーなし）。
- **Metal 実装**（`backend-metal::MetalMemory`）: 既定実装を上書きし、`self.context.synchronize()`（`download_inner` と同一の同期点。イシュー #1017）の直後に `MetalBuffer::as_host_slice`（`StorageModeShared` バッファの `contents()` を直接借用。旧 `read_to_vec` の `unsafe` ブロックをそのまま移設したのみで新規 `unsafe` は追加していない）を呼び、`read_to_vec`（`Vec` 確保 + memcpy）を経由しない。
- **CUDA**: イシュー #1336 で既定実装を上書きした（`backend-cuda::CudaMemory`）。`handle.storage` の配置ごとに 3 分岐する: (1) 空テンソル（`None`）は FFI を呼ばず `f(&[])`。(2) `Managed`（イシュー #1352 opt-in 配置）は `UnifiedSlice::as_slice()` の借用をコピーなしでそのまま渡す。(3) `Device`（既定配置）は形状（要素数）ごとに再利用するホストステージングバッファ（`crate::host_staging::HostStagingCache`。take/put 方式・世代検査つき）へ `memcpy_dtoh` 1 回で D2H した後、そのスライスを渡す。ステージング種は `Pageable`（事前タッチ済み `Vec<f32>`。unsafe なし）と `Pinned`（cudarc `alloc_pinned`。page-locked・unsafe 1 箇所）の 2 種を実装したが、実機実測が未完了（本エージェント実行環境に CUDA 実機なし）のため既定は unsafe 経路を通さない `Pageable` に固定している。設計判断・unsafe の安全性根拠・実測記入欄は `docs/perf/cuda-host-view-staging-readout.md` を参照。
- **facade 到達経路**: `Var::host_view`（3.1）／`Tensor::host_slice`（2.2）が内部実装として使う想定の backend レイヤー API。`memory_ops()` が `None` を返す実装（`Tape`／`Var::matmul` 等が直接消費する経路ではない）からは到達しない——`Var::matmul` の出力（テープ値）は演算の内部で `download` 済みのホスト常駐 `Tensor` として保持されるため（6 節「新規論点」参照）。CUDA 実装（イシュー #1336）についても同様に、`autodiff`／`facade` は現状 `with_host_view` を呼ばないため直接の受益者ではない（`docs/perf/cuda-host-view-staging-readout.md` §7「引き継ぎ」節）。

### 4.3 API 契約として明記する数値仕様

- **バックエンド間数値一致**: 全ペア共通で「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の複合判定（`docs/spec/04-requirements.md` REQ-2、PoC-v2-5 の事前固定判定基準と同一）。
- **FMA 契約統一**: CPU 参照実装は `f32::mul_add` を用い、GPU 側（CUDA NVRTC・Metal `simdgroup_multiply_accumulate`）の既定 FMA 契約と揃える。PoC-v2-5 は K=4096 の GEMM ストレスケースで、CPU 側が `acc += a * b`（乗算・加算を別々に丸め）のままだと 262,144 セル中 7 セルが複合判定を外れ、`acc = a.mul_add(b, acc)` に差し替えると完全一致（fail_cells=0/262144）することを実測確認済み（`docs/spec/04-requirements.md` REQ-2、`.claude/rules/coding-rust.md`）。
- **Metal precise math の明示**: `MTLCompileOptions.mathFloatingPointFunctions` を `Precise` に設定し、シェーダ側でも `metal::precise::exp`/`metal::precise::tanh` 等を明示使用する（`mathMode=Safe` のみでは transcendental 関数が `metal::fast` 経路にディスパッチされ複合判定の余裕が薄くなることが PoC-v2-5 で判明。`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md:49` 付近）。
- **カーネル境界検査の維持（REQ-8）**: 性能下限・最適化の達成を理由にシェーダ・カーネル側の手動境界チェックを省略しない。境界検査を無効化する最適化を適用する場合は手動境界チェックを維持したまま行う（`.claude/rules/coding-rust.md`）。本契約は `BackendOps` の全実装（CPU intrinsics・CUDA NVRTC/mma・Metal simdgroup）に適用する。

### 4.4 実行時エラー

```rust
/// バックエンド抽象層のエラー型。CUDA ドライバ不在（`cudarc` の
/// 動的ロード失敗）は「CUDA toolkit 非搭載環境でもビルド成立する」
/// という REQ-1 の契約の実行時側の受け皿として、型付きエラーで返す。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、CUDA/Metal 実装（TASK-1.9）
/// が進むにつれ想定される実行時失敗（同期エラー・ドライババージョン
/// 不整合等）の variant 追加を非破壊にするため。
#[non_exhaustive]
#[derive(Debug)]
pub enum BackendError {
    /// `cudarc` の動的ロードに失敗した（CUDA ドライバ・toolkit 不在等）。
    CudaUnavailable(String),
    /// 入力テンソルの shape が演算の要求と合わない。
    ShapeMismatch(ShapeError),
    /// デバイス間でテンソルが混在している等の不整合。
    DeviceMismatch,
    /// デバイスメモリの確保に失敗した（`upload`/`DeviceBuffer` 確保等。
    /// VRAM 枯渇・アロケータ失敗を含む）。
    DeviceAllocationFailed(String),
    /// カーネル起動に失敗した（CUDA NVRTC のコンパイル・起動エラー、
    /// Metal `MTLComputeCommandEncoder` のディスパッチ失敗等）。
    KernelLaunchFailed(String),
}
```

**TASK-1.9a（#44）実装時の突合結果**: 上記 5 variant はそのまま実装したうえで、`DeviceUnavailable(String)`（存在しないデバイス・範囲外 ordinal・対応する `DeviceProvider` 未登録等、選択失敗を表す）を追加した。`#[non_exhaustive]` により variant 追加は非破壊拡張である旨が本節のコメントで想定済みのため、本文書の変更なしに追加した。

**TASK-1.9b（#45）実装時の突合結果**: `TransferFailed(String)`（確保済みバッファへのコピー〈`upload`/`download`〉の失敗。`DeviceAllocationFailed` は確保自体の失敗と区別する）を追加した。上記と同じ理由で非破壊拡張として扱う（4.2 の突合結果参照）。

**TASK-1.9c（#46）実装時の突合結果**: `Unsupported(String)`（指定バックエンドが当該演算のカーネルを未実装であることを表す。CUDA／Metal の elementwise・reduction が本イシュー時点で GEMM カーネルのみ実装済みのため使用）を追加した。`#[non_exhaustive]` により非破壊拡張である。

## 5. エラー型一覧・横断事項

| 型 | 役割 | 発生源 |
|----|------|--------|
| `ShapeError` | テンソル生成・view 操作・reshape の shape 不整合（rank 不一致・要素数不一致・軸範囲外・narrow 範囲外・要素数積オーバーフロー。variant 定義は 2.1.1 参照）。`#[non_exhaustive]` | `tensor-core`（`RankMismatch` のみ型定義の提供にとどまり、構築は autodiff／backend 入口。2.1.1 参照） |
| `AutodiffError` | 順伝播（`Var` の演算メソッド）の shape 不整合、`Tape::backward` の失敗（グラフ不整合等）、および二項演算・`backward`・`Gradients::get` におけるクロステープ誤用（`TapeMismatch`）。`#[non_exhaustive]` | `autodiff` |
| `BackendError` | バックエンド実行時エラー（CUDA 不在・shape 不一致・デバイス不整合・デバイスメモリ確保失敗・カーネル起動失敗）。`#[non_exhaustive]` | backend 入口 |

外部フォーマット（safetensors/ONNX）読み込み時は、長さ・形状の検証を先行させる契約とする（`.claude/rules/security.md` A03 インジェクション対策）。`onnx-interop` クレートの `ShapeError` 送出経路は本設計のスコープ外（REQ-7 系、別イシューで扱う）だが、`tensor-core::Tensor::new`/`from_slice` の実行時 shape 検査がこの検証の受け皿となる設計である点を接続点として明記する。

## 6. 未決事項・採否論点の一覧（ユーザー判断用）

1. **非 contiguous な `reshape` の方針**（2.2.1）: エラー（案 A、本文書の推奨）か暗黙コピー（案 B）か。
2. **CUDA 既定有効化の構成決定**（4.1）: REQ-2 でも未検証のまま残っている。`Device::available()` が返す既定デバイスの選択ロジックは本文書では確定しない。
3. **rank 型載せの最終確定**（2.5）: 本文書は基盤層を実行時 rank、型レベル shape を後続レイヤー限定とする方針を決定として記録した。TASK-10.x 実装時にこの方針で問題がないか再確認すること。**（2026-08-07 TASK-10.1a 追記）** イシュー #98 でこの再確認を実施し、基盤 `Tensor<T>` を実行時 rank のまま据え置く決定を維持した。型レベル shape の設計正本は `docs/typed-shape-design.md` とする（本項の open item はクローズ）。
4. **演算グラフ／融合機構（イシュー #161）との接続点**: `Var`/`Tape` の演算記録が将来の融合機構（TASK-12.1a）とどう接続するかは本文書では設計しない。`Tape` の `Op` 列挙が融合対象の中間表現候補になりうる点のみ接続点として記録する。
5. **`DeviceBuffer` の内部表現**（4.2）: TASK-1.9b（#45）で確定した。`Box<dyn BufferHandle>`（`Any` ダウンキャスト経由の依存逆転構成）で不透明ハンドルを保持し、解放は各バックエンドの具体ハンドル型の `Drop` に一本化する（4.2 の「TASK-1.9b（#45）実装時の突合結果」参照）。
6. **`Var`/`Tape` の借用モデル**（3.1）: 本文書は productize 時の API 使い勝手を優先し `RefCell` + 共有参照方式を採用したが、PoC-v2-2 確定 API は `&mut Tape` 方式だった。TASK-1.5 実装時に借用チェッカ上の問題が生じた場合は `&mut Tape` 方式へ戻す判断もありうる。この方式は `Var::value()` が `Ref<'_, Tensor<f32>>` を公開シグネチャへ露出する副作用も伴う（`Ref` を保持したまま同じ `Tape` へ `borrow_mut()` を要する演算を呼ぶと panic しうる）。本文書は回避策として所有値を返す `Var::to_tensor()` を追加したが、`value()` 自体を非公開化する・`Ref` を返さない別設計にするといった、より根本的な対処の要否は TASK-1.5 実装時に再検討する。
7. **`Tape` のライフサイクル（学習ループ）**（3.1.1・**確定。イシュー #1048**）: ステップごとに新しい `Tape` を生成し破棄する運用（運用 A）に加え、`Tape::reset()`（`&mut self`）で葉プレフィックスまで切り詰めて同一 `Tape` を再利用する運用（運用 B）を追加した。`Gradients::get` は `&Var`（＝テープ借用 + `TapeId` 一致検査）をキーにする現行設計を維持しつつ、`Tape::epoch()`（reset のたびに +1 する世代番号）による追加検査で reset 後の stale な `Gradients`／`DeviceParamStore::pending` を fail-closed に拒否する（`VarId`／`(TapeId, VarId)` へのキー変更は行わなかった）。
8. **`BackendOps` の f16 対応**（4.2）: `DeviceBuffer<T: Element>` はジェネリックだが `BackendOps` トレイト v1 は `f32` 固定。GPU 推論で使う `half::f16` 経路の入口設計（トレイトのジェネリック化か並行トレイト追加か）は TASK-1.9 実装時に決定する。**イシュー #1648 で決定**（2026-09-13。`docs/backend-dtype-dispatch-design.md`）: 4.2 と同じ capability accessor ＋ `TypedOps<T>` 方式を採用。
9. **デバイスハンドル再利用の公開 API 化（イシュー #931）**: 再構築コストが発生していたのは `tape()`／`tape_for(Device)` 呼び出しごとではなく、`CudaBackendOps::gemm` 等の**演算呼び出しごと**に `CudaDevice::new`／`CudaGemm::new` が都度実行される構成だった（`Tape` を使い回しても解消しない。`crates/backend-cuda/src/ops.rs` 冒頭コメント参照）ことへの対応方針。facade に新規公開型（`DeviceHandle` 等）を追加する案は不採用と判断し、バックエンド内部（`backend-cuda::context_cache`・`backend-metal::context_cache`）のプロセス内常駐キャッシュのみで対応する構成を採用した（CUDA は #929/#946、Metal も #930/#948 で同型の実装が完了済み）。設計判断・採否根拠は `docs/facade-device-handle-design.md` を正とする。
10. **`Var::matmul` 出力のデバイス常駐化（Metal memcpy-free の facade 到達。イシュー #1335）**: `Var::host_view`（3.1）・`MemoryOps::with_host_view`（4.2）は Metal の `contents()` 直接借用（`read_to_vec` の memcpy 廃止）を実現したが、これは `DeviceBuffer<f32>` レベルに限られる。`Var::matmul` の出力（テープの `TapeNode.value: OnceCell<Tensor<f32>>`）は `gemm` 呼び出しの**内側**で `download`（`synchronize` + memcpy）済みのホスト常駐 `Tensor` として保存される構造（`crates/autodiff/src/tape.rs`・`crates/backend-metal/src/ops.rs`）であり、`Var::host_view` 自身は既にホスト常駐の値をコピーなしで返すのみで、`gemm` 内部の memcpy 自体は回避できない。`Var::matmul` 出力をデバイス常駐のまま保持する（`gemm` 内の download を遅延させる）ことは、`Tape: Send`（`crates/autodiff/tests/fusion_backend_integration.rs:390` の静的アサーション。公開契約）と Metal `MTLBuffer: !Send`（objc2-metal 0.3.2。`crates/backend-metal/src/pool.rs:108-198` の実測記録）の衝突により、(i) `MetalBuffer` への新規 `unsafe impl Send`、または (ii) `newBufferWithBytesNoCopy` によるページ整列ホストメモリ上の出力バッファ確保（objc2 の新規 `unsafe` 呼び出し）のいずれかを要する。両案とも新規 `unsafe` 導入・ユーザー承認必須（`.claude/rules/security.md`）に抵触するため本イシューでは選択せず、実測根拠（Metal N=4096 の `readback` 0.94 ms 対 `host_copy` 3.79 ms。`docs/perf/metal-gemm-reuse-phase-breakdown.md` §4）とともにユーザー判断へ引き継ぐ（Layer B〈`Var::host_view`〉が二重コピー廃止の主要効果・Layer A〈`with_host_view`〉が残余 1 ms 未満という位置づけ）。
11. **resident `GradStaging` の重み勾配をホストへ読み出す公開 API（イシュー #1479）**: `DeviceParamStore`（デバイス常駐更新経路。9 節・`docs/compat-api-scope.md` §0 確定入口 4）の resident weight 勾配は、`fandhe_ai_autodiff::backward::Gradients` へ寄与を残さず `GradStaging`（`crates/autodiff/src/optim/device_store.rs`）へ直接書き込まれるため、本追記以前はホストから一切観測できなかった（#1212 の設計）。`facade::Tape::resident_grads_to_host`（strict 版。resident 経由で新鮮に充填済みの slot のみ `Some`、未充填 slot〈bias 等〉は `None`。resident 未対応バックエンド〈CUDA／Metal。`gemm_fp32_strict_into` 未実装〉では `BackendError::Unsupported`）・`Tape::param_grads_to_host`（unified 版。未充填 slot を `Gradients::get(...)` へフォールバックし全パラメータの勾配を 3 バックエンド共通で返す）の 2 メソッドを追加した。いずれも `&self`・読み出し専用（`step()` の `pending` 消費を伴わない）で、`Gradients::get` 自体の意味論（`Ok(None)`＝「未到達」）は変更しない。判定ロジック（鮮度＋同一性検査）は `step()` と単一のヘルパへ集約し二重管理を避けた。設計・契約の詳細は `docs/device-resident-update-design.md` 追補: #1479 を正とする。

## スコープ外（out-of-scope-tracking 対象）

- **compat API 層**（`compat::array`/`compat::Sequential` の詳細シグネチャ）: REQ-9 系の後続タスク（TASK-9.2・TASK-9.4）。本文書は境界（自作コアの素の API とは別レイヤーであること）のみを 1 章で明記した。利用者向け公開面は `facade` クレートに一本化して配置する（2026-08-08 追記・イシュー #52。§4.1 参照）。`tensor-core`・`autodiff`・`backend-*` は内部クレートであり、`facade` が唯一のサポートされる公開 API 面である。
- **guardrail/self-repair の CLI 仕様**: 兄弟イシュー #183 が担当。
- **演算グラフ・カーネル融合機構の API**: イシュー #161（TASK-12.1a）。接続点のみ 6.4 に記載。
- **シグネチャのコンパイル検証・実装**: TASK-1.4（自作テンソル型 productize）以降の実装イシューへ引き継ぐ。workspace 雛形（TASK-1.1a・#205）はクレートが空のため、本イシューでは検証しない。
- **CUDA／Metal の elementwise・reduction カーネル実装**: TASK-1.9c（#46）時点では GEMM カーネルのみ実装済みのため、`BackendOps::add`/`mul`/`relu`/`exp`/`tanh`/`sum`/`max` は `BackendError::Unsupported` を返す。GPU カーネル本体の実装・引き継ぎ先 Issue の起票はユーザー承認を得て別途行う。
- **`DeviceBuffer`／`upload`／`download` への `BackendOps` シグネチャ移行**: TASK-1.9d（#47）では対象外とした（実装は別途追跡。上記 4.2 突合結果参照）。
- **既定デバイス選択ロジック・形状／HW 判定によるディスパッチ規則の統合**: 前者はユーザー承認必須（CUDA 既定有効化の構成決定）、後者は TASK-11.2b（#68）の担当（`docs/dispatch-rules-design.md`）。
- **3 バックエンド網羅の統合テスト**: TASK-1.9d（#47）で実装完了。CPU 全 8 演算・非 contiguous 入力・エラー経路・端点・3 バックエンド横断エンドツーエンドは `crates/backend-cpu/tests/backend_ops_integration.rs`、CUDA 実機での数値一致は `crates/backend-cuda/tests/backend_ops_real_device.rs`、Metal 実機での数値一致は `crates/backend-metal/tests/backend_ops_real_device.rs`（`#[ignore]` 分離）を参照。
