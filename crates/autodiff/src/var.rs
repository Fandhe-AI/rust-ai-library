//! テープ上の 1 ノードを指す追跡対象値 `Var` と、その forward 演算群。
//!
//! `fandhe_ai_tensor_core::Tensor<f32>` に対する演算は一切テープを構築しない
//! （非追跡）。`Var` に対する演算のみが `Tape::push`（`tape.rs`）を
//! 経由してテープへ記録される。この「型分離」により、勾配追跡の
//! ON/OFF がコンパイル時に保証される（`docs/public-api-design.md`
//! §3.1「型分離方式」）。
//!
//! 各演算メソッドは
//! 「①クロステープ検査 → ②shape 検査 → ③forward 値計算（`eval.rs`）
//! → ④ノード記録（`Tape::push`）」の順で処理する。値計算の借用
//! （`Ref`）はスコープを閉じてから `push`（`borrow_mut`）を呼ぶ
//! （`RefCell` の二重可変借用 panic を避けるための実装規律。
//! `.claude/rules/coding-rust.md` の本番経路 panic 禁止方針）。

use std::cell::Ref;

use fandhe_ai_tensor_core::{
    Activation, BackendError, BackendOps, ChecksumReadout, GemmChecksum, GruPointwiseOutput,
    LstmPointwiseOutput, MatrixNormOrd, MseReduction, ShapeError, Tensor, broadcast_shape,
    concat_out_shape, matmul_out_shape, reduce_out_shape, require_same_shape, row_norm_layout,
};

use crate::error::AutodiffError;
use crate::eval;
use crate::grad::concat_with_fallback;
use crate::tape::{NodeId, Op, Tape, materialize_fallible, materialize_non_fallible};

/// `Var::mse_loss_with` の縮約種別（#190・TASK-9.1c 相当。親イシュー
/// #189「損失関数（MSE・CrossEntropy）の実装」）。PyTorch
/// `nn.MSELoss(reduction=...)` の `mean`/`sum` に対応する。
///
/// `#[non_exhaustive]` とする理由: 将来 `none`（要素ごと損失。PyTorch
/// `reduction='none'` 相当）を追加しうるが、本イシューでは #190 実装
/// 計画のスコープ外（out-of-scope-tracking.md 準拠でユーザー承認後に
/// 別途追加）としたため、追加時に呼び出し側の非網羅的 `match` を破壊
/// しないようにする。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Reduction {
    /// 全要素平均（`Σ(pred−target)² / n`）。`Var::mse_loss` の既定
    /// （PyTorch `nn.MSELoss` の既定 `reduction='mean'` と一致）。
    Mean,
    /// 全要素総和（`Σ(pred−target)²`）。
    Sum,
}

/// `crate::var::Reduction` → `fandhe_ai_tensor_core::MseReduction` の変換
/// （イシュー #1045）。`tensor-core` → `autodiff` の逆依存は作れないため
/// `MseReduction` は `Reduction` の再エクスポートではなく独立した型
/// （`backend_ops.rs::MseReduction` doc 参照）であり、`Var::mse_loss_with`
/// が `BackendOps::mse_loss`／`mse_loss_backward` を呼ぶ直前にここで変換
/// する。両者は `Mean`/`Sum` の 2 variant のみで意味論も同一のため
/// 単純な 1 対 1 写像。
impl From<Reduction> for MseReduction {
    fn from(value: Reduction) -> Self {
        match value {
            Reduction::Mean => MseReduction::Mean,
            Reduction::Sum => MseReduction::Sum,
        }
    }
}

/// テープ上の 1 ノードを指す追跡対象値。値そのものではなく `NodeId` +
/// テープへの共有参照を保持し、演算のたびにテープへ新しいノードを
/// 追加する（`docs/public-api-design.md` §3.1）。
///
/// **クロステープ安全性**: ライフタイム `'t` の一致は同一 `Tape` を
/// 指す証明にはならない（同一スコープに複数の `Tape` が存在する場合、
/// それぞれの `Var<'t>` は同一の `'t` を持ちうる）。そのため二項演算
/// （`matmul`/`add`/`mul`/`mse_loss`）は入口で `self.tape.id` と相手側
/// `Var` が保持する `TapeId` の一致を実行時検査し、不一致なら
/// `AutodiffError::TapeMismatch` を返す。
#[derive(Debug, Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    id: NodeId,
}

impl<'t> Var<'t> {
    /// `Tape::var()` からのみ呼ばれる内部コンストラクタ。
    pub(crate) fn from_raw(tape: &'t Tape, id: NodeId) -> Var<'t> {
        Var { tape, id }
    }

    /// 追跡を外し、現在の値を非追跡の `Tensor<f32>` の借用として取り出す。
    ///
    /// **TASK-12.1d（#164）**: 対象ノードが未実体化（elementwise の遅延
    /// グラフの一部）であれば `materialize_non_fallible`（層 2。融合を
    /// 試み、失敗すれば `ops` の per-op メソッド → `eval.rs` の順に必ず
    /// 値を返す）経由で実体化する。`matmul`/`sum`/`max`・`Tape::backward`
    /// が使う層 1（`crate::tape::materialize_fallible`）とは異なる
    /// エラー処理契約を持つ（`docs/fusion-graph-design.md` §3.5.3）。
    ///
    /// **借用注意**: この `Ref` を保持したまま、同じ `Tape` に対して
    /// `borrow_mut()` を要する演算（`matmul`/`add` 等のノード追加）を
    /// 呼ぶと `RefCell` の二重可変借用で実行時 panic になる。値をその場
    /// の参照ではなく所有値として持ち出したい場合は `to_tensor()` を
    /// 使うこと（`docs/public-api-design.md` §3.1）。
    pub fn value(&self) -> Ref<'_, Tensor<f32>> {
        Ref::map(self.tape.nodes.borrow(), |nodes| {
            materialize_non_fallible(nodes, self.tape.ops(), self.id)
        })
    }

    /// `value()` の所有値版。`Tensor<f32>` へ複製して返すため `Ref` を
    /// 持ち越さず、直後に同じ `Tape` へノード追加演算を呼んでも借用
    /// エラー・panic が起きない。
    pub fn to_tensor(&self) -> Tensor<f32> {
        let nodes = self.tape.nodes.borrow();
        materialize_non_fallible(&nodes, self.tape.ops(), self.id).clone()
    }

    /// ホスト可視の値を借用で読み出す（イシュー #1335・`docs/public-api-
    /// design.md` §3.1「`VarHostView`」）。
    ///
    /// **P1 是正（codex-review 指摘・イシュー #1335）**: 当初実装は
    /// `Tape` の `RefCell` 借用（`Ref<'_, [f32]>`）をそのまま
    /// `VarHostView` へ持ち越しており、`host_view()` を保持したまま
    /// 同じ `Tape` へノード追加演算（`add`/`matmul` 等の `push_lazy`）を
    /// 呼ぶと `borrow_mut()` が実行時 panic した（本番経路 panic 禁止・
    /// `.claude/rules/coding-rust.md` 違反）。`value()` の既存制約と
    /// 同型ではあるが、新規公開 API がその制約をそのまま持ち込む理由には
    /// ならないため、`RefCell` 借用を関数内に閉じ込める形へ是正した。
    ///
    /// `Tensor<f32>` は内部 `storage: Arc<Storage<T>>` を `Arc` 共有する
    /// 値型（`tensor.rs` モジュール冒頭コメント「`Clone` は `Arc` の
    /// ポインタ複製のみで安価」参照）であるため、`materialize_non_fallible`
    /// が返す `&Tensor<f32>`（`Tape` の `RefCell` 借用が生存中のみ有効）
    /// を `Tensor::contiguous()`（contiguous な場合は内部で `self.clone()`
    /// する安価な `Arc` 複製、非 contiguous な場合のみ 1 回実体化）へ
    /// 通してから所有値として持ち出せば、いずれの分岐でもデータの
    /// 追加コピーなしに `RefCell` 借用（`nodes`）をこの関数のスコープ内
    /// で確実に解放できる。返す [`VarHostView`] は `Tape` の借用を一切
    /// 保持しないため、生存中に同じ `Var`／`Tape` への他の演算を呼んでも
    /// panic しない。
    pub fn host_view(&self) -> VarHostView {
        let tensor = {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), self.id).contiguous()
        };
        VarHostView { tensor }
    }

    /// 実体化なしに読める構造的な出力 shape（`TapeNode.shape`。
    /// TASK-12.1d・#164）。演算入口の shape 検査は本メソッドを使い、
    /// `value()`/`materialize_fallible` を呼ばない（`docs/
    /// fusion-graph-design.md` §3.5.1「shape 検証と実行を分離する」）。
    fn shape(&self) -> Vec<usize> {
        self.tape.nodes.borrow()[self.id.0].shape.clone()
    }

    /// 演算入口で必ず shape 検査より前に呼ぶクロステープ検査
    /// （`docs/public-api-design.md` §3.1「クロステープ安全性」）。
    fn check_same_tape(&self, other: &Var<'t>) -> Result<(), AutodiffError> {
        if self.tape.id != other.tape.id {
            return Err(AutodiffError::TapeMismatch);
        }
        Ok(())
    }

    /// この `Var` が属する `Tape` の識別子。`backward.rs`（TASK-1.5c・
    /// #18）は別モジュールのため `tape` フィールド（private）へ直接
    /// 触れられず、`Tape::backward`/`Gradients::get` のクロステープ検査
    /// （`check_same_tape` と同じ「入口で必ず shape・NodeId 解決より前に
    /// 検査する」契約）にこのアクセサを使う。
    pub(crate) fn tape_id(&self) -> crate::tape::TapeId {
        self.tape.id
    }

    /// この `Var` が指すテープ内ノードの識別子。`backward.rs` が
    /// `Gradients` から当該ノードの勾配を引くための添字として使う
    /// （`tape_id()` と同じくクレート内限定公開）。
    pub(crate) fn node_id(&self) -> NodeId {
        self.id
    }

    /// この `Var` が属する `Tape` の現在の世代番号（#1048）。`Gradients::get`
    /// が「この `Var` は reset 後の別世代のものか」を fail-closed に検査
    /// するための比較対象（`tape::Tape::epoch` doc 参照）。`Var<'t>` 自体は
    /// `&'t Tape` を静的に借用するため reset 後に stale な `Var` を作る
    /// ことはコンパイル時に排除されるが、`Gradients`（`Tape` を借用しない
    /// 値）は reset をまたいで生存しうるため、こちらは実行時検査が要る。
    pub(crate) fn tape_epoch(&self) -> u64 {
        self.tape.epoch()
    }

    /// 2 次元 `matmul`（`docs/public-api-design.md` §3.2）。
    ///
    /// **TASK-12.1d（#164）**: 非 elementwise のため常に実体化済みで
    /// 返る（`push_eager`）。実行は `eval.rs` 直接呼び出しから
    /// `self.tape.ops().gemm`（`BackendOps` 経由）へ置き換えた
    /// （TASK-1.9「backend 経由実行への置き換え」・設計書 §3.5.2）。
    /// 入力が elementwise の遅延グラフであった場合は
    /// `materialize_fallible`（層 1）で自身の実行の一部として実体化
    /// する。
    pub fn matmul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        matmul_out_shape(&lhs_shape, &rhs_shape)?;
        let (lhs_val, rhs_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), other.id)?.clone();
            (lhs_val, rhs_val)
        };
        let value = self.tape.ops().gemm(&lhs_val, &rhs_val)?;
        let id = self.tape.push_eager(Op::MatMul(self.id, other.id), value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// `C = self @ other` を計算しつつ、`C` の全要素和（checksum）を
    /// バックエンド側の `f64` reduction（`BackendOps::gemm_checksum`）で
    /// 求める（イシュー #1339）。framework-compare の gemm 計測窓が毎
    /// 反復行っていた「D2H → ホスト `f64` 逐次和」の 2 段を、GPU
    /// バックエンドでは「checksum（8 バイト）のみ読み戻す」経路へ置き
    /// 換えるための入口（`docs/perf/device-checksum-readback-ab.md`）。
    ///
    /// **[`Var::matmul`] との相違（tape への非記録）**: 本メソッドは
    /// `self.tape.push_eager` を呼ばず、戻り値の `GemmChecksum` は
    /// tape ノードを持たない生の計算結果として返す（backward の対象外。
    /// ベンチハーネスの計測専用入口という位置づけであり、学習経路
    /// （`Var::matmul` チェーン）とは独立している）。`readout` が
    /// [`ChecksumReadout::ChecksumOnly`] のときバックエンド実装は `C`
    /// をホストへ download しない契約（`BackendOps::gemm_checksum` doc
    /// 参照）。
    ///
    /// 検証手順は `matmul` と同じ「①クロステープ検査 → ②shape 検査 →
    /// ③入力実体化」までを行い、④のみ `ops().gemm` ではなく
    /// `ops().gemm_checksum` を呼ぶ。既定実装（CUDA／Metal は本イシュー
    /// 時点で未オーバーライド）が返す [`BackendError::Unsupported`] は
    /// そのまま呼び出し元（framework-compare ハーネス）へ伝播する
    /// （判定迂回経路を作らない。`.claude/rules/security.md` A08）。
    pub fn matmul_checksum(
        &self,
        other: &Var<'t>,
        readout: ChecksumReadout,
    ) -> Result<GemmChecksum, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        matmul_out_shape(&lhs_shape, &rhs_shape)?;
        let (lhs_val, rhs_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), other.id)?.clone();
            (lhs_val, rhs_val)
        };
        Ok(self.tape.ops().gemm_checksum(&lhs_val, &rhs_val, readout)?)
    }

    /// `y = act(self.matmul(weight) (+ bias))` を 1 ノード
    /// （[`Op::LinearAct`]）として記録する（イシュー #1044・`docs/
    /// kernel-fusion.md` §2.2「学習経路への結線」）。
    /// `fandhe_ai_autodiff::nn::linear::LinearVars::forward_with_activation`
    /// が唯一の呼び出し元（`Var::matmul` と同じ「①クロステープ検査 →
    /// ②shape 検査 → ③forward 値計算 → ④ノード記録」の順で処理する
    /// 非 elementwise・常時実体化の演算）。
    ///
    /// `bias` の shape 検証は `broadcast_shape`（`out_shape` へブロード
    /// キャスト可能かの NumPy 互換判定）のみを行い、`[n]`（`weight` の
    /// 列数）と厳密一致しない bias（`[1, n]` 等）も含めてそのまま
    /// `BackendOps::gemm_bias_act` へ委譲する。**非融合合成へのフォール
    /// バックは本メソッド・呼び出し元（`LinearVars::
    /// forward_with_activation`）のどちらの責務でもなく、
    /// `BackendOps::gemm_bias_act` 自身の契約**（`tensor-core::
    /// backend_ops` の doc 参照。CPU／CUDA／Metal の融合カーネル実装は
    /// bias が `[n]` 厳密一致でない場合 `matmul` → `add`（NumPy 互換
    /// ブロードキャスト）→ activation の非融合合成へ内部的に
    /// フォールバックし、デフォルト実装も同じ合成のため、いずれの
    /// バックエンドでも `[n]` 以外の broadcast 可能な bias が
    /// `ShapeMismatch` になることはない）。本メソッドが呼び出し前に
    /// `broadcast_shape` で検証するのは「`gemm_bias_act` に委譲する前に
    /// ブロードキャスト不能な shape を早期に拒否する」ためであり、
    /// フォールバック経路の選択自体は行わない。
    pub(crate) fn linear_act(
        &self,
        weight: &Var<'t>,
        bias: Option<&Var<'t>>,
        act: Activation,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(weight)?;
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }
        let lhs_shape = self.shape();
        let rhs_shape = weight.shape();
        let out_shape = matmul_out_shape(&lhs_shape, &rhs_shape)?;
        if let Some(b) = bias {
            broadcast_shape(&out_shape, &b.shape())?;
        }
        let (lhs_val, rhs_val, bias_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), weight.id)?.clone();
            let bias_val = match bias {
                Some(b) => Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone()),
                None => None,
            };
            (lhs_val, rhs_val, bias_val)
        };
        let value = self
            .tape
            .ops()
            .gemm_bias_act(&lhs_val, &rhs_val, bias_val.as_ref(), act)?;
        let id = self.tape.push_eager(
            Op::LinearAct {
                input: self.id,
                weight: weight.id,
                bias: bias.map(|b| b.id),
                act,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// bias broadcast を含む要素ごとの加算（`docs/public-api-design.md`
    /// §3.2）。
    ///
    /// **TASK-12.1d（#164）**: elementwise 5 演算の 1 つ。shape 検証
    /// （①クロステープ検査・②shape 検査）のみ即時実行し、値計算
    /// （③）は実体化境界まで遅延させる（`push_lazy`。`Ok` を返すことは
    /// 「shape が妥当でノードが記録された」ことのみを意味し「加算が
    /// 計算済み」であることを意味しない。設計書 §3.5.1）。
    ///
    /// **連鎖長上限（#404・設計書 §3.5.4）**: `push_lazy` を呼ぶ**前**に
    /// `Tape::pre_materialize_for_binary_merge` で fan-in 事前実体化を
    /// 行う（2 本の未実体化枝を合流させた結果が単独で上限を超えるなら
    /// 大きい方の枝を先に実体化する。codex-review PR #406 の P1 是正。
    /// push 後の自己実体化だけでは fan-in を防げないため必須）。続けて
    /// `push_lazy` が返す `at_limit` が `true`（新規ノードの
    /// `lazy_chain_size` が `MAX_FUSED_CHAIN_LEN` に到達）の場合、層 1
    /// （`materialize_fallible`）でその場実体化する。**いずれの実体化
    /// も**発生した場合、`Ok` の意味は「shape が妥当でノードが記録され
    /// **かつバックエンド実行が成功した**」へ拡張される（同じ層 1 契約
    /// を持つ `matmul`/`sum`/`max` と同型の `Ok` 意味）。実体化失敗は
    /// （事前実体化・push 後の自己実体化のいずれも）`?` でそのまま伝播
    /// する。
    pub fn add(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        // fan-in 事前実体化（#404・codex-review PR #406 の P1 是正）:
        // 2 本の未実体化枝を合流させる前に、合流後サイズが上限を超える
        // なら大きい方の枝を先に実体化する（`Tape::
        // pre_materialize_for_binary_merge` のドキュメント参照）。
        self.tape
            .pre_materialize_for_binary_merge(self.id, other.id)?;
        let (id, at_limit) = self.tape.push_lazy(Op::Add(self.id, other.id), out_shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), id)?;
        }
        Ok(Var::from_raw(self.tape, id))
    }

    /// ブロードキャスト付き要素ごとの乗算。elementwise 5 演算の 1 つ
    /// （`add` と同じ遅延契約・fan-in 事前実体化契約・連鎖長上限での
    /// 自己実体化契約。`Ok` の意味の拡張も `add` と同型。
    /// TASK-12.1d・#164・#404・codex-review PR #406）。
    pub fn mul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        // fan-in 事前実体化（`add` と同じ契約。#404・codex-review PR #406
        // の P1 是正）。
        self.tape
            .pre_materialize_for_binary_merge(self.id, other.id)?;
        let (id, at_limit) = self.tape.push_lazy(Op::Mul(self.id, other.id), out_shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), id)?;
        }
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約和。`dim: None` は全軸縮約（スカラー）。
    /// 非 elementwise のため常に実体化済みで返る（`matmul` と同じ
    /// TASK-12.1d の置き換え方針。実行は `self.tape.ops().sum` 経由）。
    pub fn sum(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = self.tape.ops().sum(&input_val, dim)?;
        let id = self.tape.push_eager(
            Op::Sum {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約最大値。`dim: None` は全軸縮約（スカラー）。
    /// `sum` と同じ置き換え方針（`self.tape.ops().max` 経由）。
    pub fn max(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = self.tape.ops().max(&input_val, dim)?;
        let id = self.tape.push_eager(
            Op::Max {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 平均二乗誤差（`self` = 予測値、`target` = 正解値。全要素平均・
    /// PyTorch `nn.MSELoss` の既定 `reduction='mean'` 相当）。
    /// `mse_loss_with(target, Reduction::Mean)` への委譲（#190）。
    /// 既存呼び出し元（`nn::activation` 系テスト・`tests/backward.rs`
    /// 等）のシグネチャ・意味を変えないため本メソッドは維持する。
    pub fn mse_loss(&self, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.mse_loss_with(target, Reduction::Mean)
    }

    /// 平均二乗誤差（`self` = 予測値、`target` = 正解値）。`reduction`
    /// で mean/sum の縮約種別を選べる（#190。親イシュー #189「損失関数
    /// （MSE・CrossEntropy）の実装」）。`nn::loss::MseLoss`（`nn/loss.rs`）
    /// はこのメソッドを呼ぶだけの薄いラッパー（REQ-9）。
    ///
    /// **TASK-12.1d（#164）→ イシュー #1045 で更新**: 入力を層 1
    /// （`materialize_fallible`）で実体化したうえで、`self.tape.ops()`
    /// の `BackendOps::mse_loss`（CPU／CUDA／Metal の融合カーネル。
    /// `docs/kernel-fusion.md`）を試みる。`Err(BackendError::
    /// Unsupported(_))` のときのみ従来のホスト参照実装 `eval::mse_loss`
    /// へフォールバックし、それ以外のエラー（融合カーネルが実行時に
    /// 失敗した場合等）は伝播する（判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08。`materialize_fallible` の
    /// `run_fused` フォールバック規律と同じ方針。`tape.rs:905` 参照）。
    /// `require_same_shape` が既に shape 一致を検査済みのため、
    /// バックエンド実装が返す `ShapeMismatch` は「バックエンド実装の
    /// 契約違反」を意味し、こちらも `Unsupported` 同様フォールバック
    /// せず伝播する（想定内の分岐で握り潰さない）。
    pub fn mse_loss_with(
        &self,
        target: &Var<'t>,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(target)?;
        let lhs_shape = self.shape();
        let rhs_shape = target.shape();
        require_same_shape(&lhs_shape, &rhs_shape)?;
        let (pred_val, target_val) = {
            let nodes = self.tape.nodes.borrow();
            let pred_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let target_val = materialize_fallible(&nodes, self.tape.ops(), target.id)?.clone();
            (pred_val, target_val)
        };
        let value = match self
            .tape
            .ops()
            .mse_loss(&pred_val, &target_val, reduction.into())
        {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // mse_loss` doc「戻り値は shape `[]`」）を検証する
                // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                if !v.shape().is_empty() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: Vec::new(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::mse_loss(&pred_val, &target_val, reduction),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::MseLoss {
                pred: self.id,
                target: target.id,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 行方向 RMSNorm（`x · rsqrt(mean(x²) + eps) · w`。`w` が `None`
    /// の場合は乗算をスキップ。イシュー #1596）。正規化軸は常に
    /// 最終軸（[`row_norm_layout`] が `(rows, hidden)` を導出する）。
    ///
    /// 検査順序（`mse_loss_with` と同じ演算メソッド規律。`weight` を
    /// 渡す場合のみクロステープ検査を追加）: ①`weight` があれば
    /// `check_same_tape` → ②`eps` が有限かつ非負であることを検査
    /// （違反は `AutodiffError::InvalidArgument`。`docs/norm-ops-design.md`
    /// 「`eps` 検査」節） → ③`row_norm_layout` で `self` の shape から
    /// `hidden` を導出し、`weight` の shape が `[hidden]` と厳密一致する
    /// ことを検査（`ShapeError::ShapeMismatch`） → ④実体化（層 1）
    /// → ⑤`self.tape.ops().rmsnorm` を試み `Unsupported` のときのみ
    /// `eval::rmsnorm_rows` へフォールバック（それ以外のエラーは伝播。
    /// 判定迂回経路を作らない。`.claude/rules/security.md` A08）
    /// → ⑥バックエンド契約検証（戻り値 shape が入力と恒等）
    /// → ⑦ノード記録。
    pub fn rms_norm(&self, weight: Option<&Var<'t>>, eps: f32) -> Result<Var<'t>, AutodiffError> {
        if let Some(w) = weight {
            self.check_same_tape(w)?;
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::rms_norm: eps must be finite and non-negative, got {eps}"
            )));
        }
        let x_shape = self.shape();
        let (_, hidden) = row_norm_layout(&x_shape)?;
        if let Some(w) = weight {
            require_same_shape(&w.shape(), &[hidden])?;
        }
        let (x_val, w_val) = {
            let nodes = self.tape.nodes.borrow();
            let x_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(&nodes, self.tape.ops(), w.id)?.clone()),
                None => None,
            };
            (x_val, w_val)
        };
        let value = match self.tape.ops().rmsnorm(&x_val, w_val.as_ref(), eps) {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // rmsnorm` doc「戻り値の shape は入力 `x` と恒等」）を
                // 検証する（実装バグの黙認防止。`.claude/rules/
                // security.md` A08）。
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => {
                let (rows, hidden) = row_norm_layout(&x_shape)?;
                // `as_slice()` は非 contiguous な入力（`weight` に転置
                // view 等が渡された場合）で `None` を返しうるため、
                // `dense_vec`（`contiguous()` 経由の稠密化。`eval.rs`）
                // を使う——`as_slice()` を直接使うと非 contiguous な
                // `weight` を誤って「重みなし」（乗算スキップ）として
                // 扱ってしまう（判定迂回経路。`.claude/rules/
                // security.md` A08）。
                let w_dense = w_val.as_ref().map(eval::dense_vec);
                eval::rmsnorm_rows(&x_val, w_dense.as_deref(), eps, rows, hidden)
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::RmsNorm {
                input: self.id,
                weight: weight.map(|w| w.id),
                eps,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 行方向 LayerNorm（`(x − mean(x)) · rsqrt(var(x) + eps) · w + b`。
    /// `w`／`b` はそれぞれ `None` の場合は対応する演算をスキップ。
    /// 分散は biased（÷N）。イシュー #1596）。[`Self::rms_norm`] と
    /// 同じ最終軸限定契約・検査順序（`bias` も `weight` と同じ
    /// クロステープ検査・shape `[hidden]` 検査を受ける）。
    pub fn layer_norm(
        &self,
        weight: Option<&Var<'t>>,
        bias: Option<&Var<'t>>,
        eps: f32,
    ) -> Result<Var<'t>, AutodiffError> {
        if let Some(w) = weight {
            self.check_same_tape(w)?;
        }
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::layer_norm: eps must be finite and non-negative, got {eps}"
            )));
        }
        let x_shape = self.shape();
        let (_, hidden) = row_norm_layout(&x_shape)?;
        if let Some(w) = weight {
            require_same_shape(&w.shape(), &[hidden])?;
        }
        if let Some(b) = bias {
            require_same_shape(&b.shape(), &[hidden])?;
        }
        let (x_val, w_val, b_val) = {
            let nodes = self.tape.nodes.borrow();
            let x_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(&nodes, self.tape.ops(), w.id)?.clone()),
                None => None,
            };
            let b_val = match bias {
                Some(b) => Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone()),
                None => None,
            };
            (x_val, w_val, b_val)
        };
        let value = match self
            .tape
            .ops()
            .layer_norm(&x_val, w_val.as_ref(), b_val.as_ref(), eps)
        {
            Ok(v) => {
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => {
                let (rows, hidden) = row_norm_layout(&x_shape)?;
                // `rms_norm` と同じ理由（直上コメント）で `dense_vec`
                // を使う（`as_slice()` は非 contiguous を無音で
                // 「なし」化してしまう）。
                let w_dense = w_val.as_ref().map(eval::dense_vec);
                let b_dense = b_val.as_ref().map(eval::dense_vec);
                eval::layer_norm_rows(
                    &x_val,
                    w_dense.as_deref(),
                    b_dense.as_deref(),
                    eps,
                    rows,
                    hidden,
                )
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::LayerNorm {
                input: self.id,
                weight: weight.map(|w| w.id),
                bias: bias.map(|b| b.id),
                eps,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 行方向 softmax（`exp(x − max(x)) / Σexp(x − max(x))`。イシュー
    /// #1594）。`dim` は [`reduce_out_shape`] で範囲検査する（既存の
    /// `sum`/`max` と同じ軸検査ヘルパーを再利用。softmax は shape 不変
    /// のため戻り値 shape 自体には使わないが、`AxisOutOfRange` の検査
    /// 目的のみで呼ぶ）。
    ///
    /// `self.tape.ops().softmax`（`BackendOps::softmax`。CPU／CUDA／
    /// Metal の行カーネル。最終軸専用）を試み、`Err(BackendError::
    /// Unsupported(_))` のときのみホスト参照実装 `eval::softmax_along`
    /// （最終軸に限らず任意軸へ対応）へフォールバックする（それ以外の
    /// エラーは伝播する。判定迂回経路を作らない。`.claude/rules/
    /// security.md` A08。`mse_loss_with` と同じ規律）。
    pub fn softmax(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, Some(dim))?;
        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().softmax(&x_val, dim) {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // softmax` doc「戻り値 shape は入力と恒等」）を検証する
                // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::softmax_along(&x_val, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::Softmax {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 行方向 log_softmax（`x − m − ln(Σexp(x − m))`。イシュー #1594）。
    /// [`Self::softmax`] と同じ `dim` 検査・フォールバック規律
    /// （`BackendOps::log_softmax` → `Unsupported` のときのみ `eval::
    /// log_softmax_along` へフォールバック）。
    pub fn log_softmax(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, Some(dim))?;
        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().log_softmax(&x_val, dim) {
            Ok(v) => {
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::log_softmax_along(&x_val, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::LogSoftmax {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191・
    /// 親イシュー #189）。`self` = logits（追跡対象）、`targets` = 正解
    /// クラス添字（非追跡・`Tensor<i32>`。勾配は定義されないため
    /// `Var` にしない。`tape::Op::CrossEntropyLoss` doc 参照）。
    ///
    /// 検査順序（本メソッド冒頭 doc の演算メソッド規律に、targets
    /// 範囲検査〈REQ-8 趣旨の境界外アクセス防止・A03 対策〉を追加）:
    /// ①`class_dim` 範囲・targets shape 一致（`reduce_out_shape` を
    /// 再利用。`class_dim >= rank` は `ShapeError::AxisOutOfRange`）
    /// → ②targets 全添字が `0 <= t < C`（違反は
    /// `AutodiffError::InvalidArgument`）→ ③実体化（層 1）→ ④forward
    /// 値計算（`eval::cross_entropy_loss`。`mse_loss_with` と同じく
    /// `BackendOps` に対応メソッドがないため融合対象外）→ ⑤ノード記録。
    pub fn cross_entropy_loss(
        &self,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        let logits_shape = self.shape();
        let expected_targets_shape = reduce_out_shape(&logits_shape, Some(class_dim))?;
        require_same_shape(targets.shape(), &expected_targets_shape)?;

        // `reduce_out_shape` が成功した時点で `class_dim < logits_shape.len()`
        // が保証されるため、この添字アクセスは安全（`.claude/rules/
        // coding-rust.md` REQ-8「境界検査を省略しない」の趣旨に沿い、
        // 検査済みの添字のみでアクセスする）。
        let num_classes = logits_shape[class_dim];
        for t in eval::dense_vec_i32(targets) {
            if t < 0 || (t as usize) >= num_classes {
                return Err(AutodiffError::InvalidArgument(format!(
                    "cross_entropy_loss: target 添字 {t} が範囲 [0, {num_classes}) を外れている"
                )));
            }
        }

        let logits_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = eval::cross_entropy_loss(&logits_val, targets, class_dim, reduction);
        let id = self.tape.push_eager(
            Op::CrossEntropyLoss {
                logits: self.id,
                targets: targets.clone(),
                class_dim,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// ReLU。shape を変えない要素ごとの演算のため構造的に失敗しえない
    /// （`docs/public-api-design.md` §3.2）。elementwise 5 演算の 1 つ
    /// （`add`/`mul` と同じ遅延契約。TASK-12.1d・#164）。
    ///
    /// **連鎖長上限（#404・設計書 §3.5.4）**: 非 fallible な単項演算
    /// のため、上限到達時は層 2（`materialize_non_fallible`）でその場
    /// 実体化する（`add`/`mul` の層 1 とは異なり、必ず値が入り
    /// panic／`Err` を返さない）。
    pub fn relu(&self) -> Var<'t> {
        let shape = self.shape();
        let (id, at_limit) = self.tape.push_lazy(Op::Relu(self.id), shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), id);
        }
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの指数関数。elementwise 5 演算の 1 つ（`relu` と同じ
    /// 遅延契約・連鎖長上限での自己実体化契約。#404）。
    pub fn exp(&self) -> Var<'t> {
        let shape = self.shape();
        let (id, at_limit) = self.tape.push_lazy(Op::Exp(self.id), shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), id);
        }
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの双曲線正接。elementwise 5 演算の 1 つ（`relu` と同じ
    /// 遅延契約・連鎖長上限での自己実体化契約。#404）。
    pub fn tanh(&self) -> Var<'t> {
        let shape = self.shape();
        let (id, at_limit) = self.tape.push_lazy(Op::Tanh(self.id), shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), id);
        }
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとのシグモイド（`1 / (1 + exp(-x))`）。`relu`/`exp`/`tanh`
    /// と同じく shape 不変の単項演算のため構造的に失敗しえない
    /// （TASK-9.1b・#92。`nn::activation::Sigmoid` の薄いラッパーが
    /// このメソッドを呼ぶ）。forward は `eval::sigmoid`（数値安定形）
    /// を使う。
    ///
    /// **TASK-12.1d（#164）**: `BackendOps` に対応メソッドがないため
    /// 融合対象外とし、常に実体化済みで返る（`push_eager`）。入力読み
    /// 出しは非 fallible な本メソッド自身の契約に合わせ `value()`
    /// （層 2）経由とする（設計書 §3.5.1）。
    pub fn sigmoid(&self) -> Var<'t> {
        let value = eval::sigmoid(&self.value());
        let id = self.tape.push_eager(Op::Sigmoid(self.id), value);
        Var::from_raw(self.tape, id)
    }

    /// activation checkpointing（イシュー #1624・`docs/
    /// autodiff-checkpoint-design.md`）: `self` を区間の出力、`inputs`
    /// を区間の外側入力として、`inputs` より後・`self` 以前に push
    /// された再計算可能ノード（`Op::is_checkpoint_eligible()`）を解放
    /// する（`Tape::register_checkpoint` 経由）。`self` 自身は解放
    /// されない（呼び出し元が戻り値として保持し続けるため）。
    ///
    /// [`Tape::checkpoint`]（閉包版）の低儀式な代替入口——facade は
    /// `Tape` newtype への新規 `pub fn` 追加を承認事項として保留して
    /// いる一方、`Var` は素で再エクスポート済みのため、本メソッドが
    /// facade 経由でも到達可能な唯一の checkpoint 入口となる
    /// （`docs/compat-api-scope.md` §1.3）。
    ///
    /// `inputs` が空の場合、区間はテープ先頭（node id 0）から `self`
    /// までとする。`inputs` のいずれかが別 `Tape` に属する場合は
    /// `Err(TapeMismatch)`（クロステープ検査。`check_same_tape` doc
    /// 参照）。`self` の node id が `inputs` のどれよりも小さい（区間が
    /// 空）場合は no-op で `self` をそのまま返す。
    pub fn checkpoint_from(&self, inputs: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError> {
        for &input in inputs {
            self.check_same_tape(input)?;
        }
        let lo = inputs.iter().map(|v| v.node_id().0 + 1).max().unwrap_or(0);
        let output_id = self.id.0;
        if output_id < lo {
            return Ok(Var::from_raw(self.tape, self.id));
        }
        self.tape.register_checkpoint(lo, output_id)?;
        Ok(Var::from_raw(self.tape, self.id))
    }

    /// 新しい shape へ再解釈する view 系ノード（イシュー #1047・親
    /// #1043）。`Tensor::reshape`（`tensor-core`）と同じく contiguous な
    /// 入力に限り zero-copy（案 A・エラー方式。`docs/spec/
    /// public-api-design.md` §2.2.1 は未決事項としていたが、本イシューは
    /// 自動運転モードのため安全側の案 A を踏襲する。案 B〈暗黙コピー〉
    /// への変更はユーザー承認事項として `tensor-core::Tensor::reshape`
    /// のドキュメント参照）。
    ///
    /// 検査順序（演算メソッド規律「①クロステープ検査 → ②shape 検査 →
    /// ③forward 値計算 → ④ノード記録」を view 系向けに具体化）:
    /// ①要素数一致（クロステープ検査は単項演算のため不要）→ ②層 1
    /// （`materialize_fallible`）で入力を実体化 → ③実体化値の
    /// `is_contiguous()` を検査（非 contiguous なら
    /// `ShapeError::NonContiguousReshape` を返し、暗黙コピーでバッファ
    /// 確保 0 の契約を破らない）→ ④`Tape::push_view` でホスト値を持たない
    /// ノードとして記録する（`tape::Op::Reshape` doc「forward のたびに
    /// バッファ確保しない」の中核）。
    pub fn reshape(&self, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let in_numel: usize = in_shape.iter().product();
        // `checked_numel` 相当のオーバーフロー検査（`tensor-core` は
        // この関数を非公開にしているため、`autodiff` 側で `checked_mul`
        // を用いて自前実装する。REQ-8 趣旨の境界検査 A03 対策）。
        let out_numel = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        let out_numel = match out_numel {
            Some(n) => n,
            None => {
                return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
            }
        };
        if out_numel != in_numel {
            return Err(AutodiffError::Shape(ShapeError::ElementCountMismatch {
                expected: out_numel,
                actual: in_numel,
            }));
        }

        // 入力を層 1 で実体化する（`Tape::push_view` の呼び出し契約:
        // `input` は push 前に必ず実体化済みであること）。`Ref` を保持
        // したまま `push_view`（`borrow_mut`）を呼ぶと `RefCell` の
        // 二重可変借用 panic になるため、`is_contiguous()` 検査まで
        // 完了してからスコープを閉じる。
        {
            let nodes = self.tape.nodes.borrow();
            let input_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?;
            if !input_val.is_contiguous() {
                return Err(AutodiffError::Shape(ShapeError::NonContiguousReshape));
            }
        }
        let id = self
            .tape
            .push_view(Op::Reshape { input: self.id }, shape.to_vec());
        Ok(Var::from_raw(self.tape, id))
    }

    /// 2 軸の転置（view 系ノード。イシュー #1047・親 #1043）。
    /// `Tensor::transpose`（`tensor-core`）と同じく常に zero-copy
    /// （strides の入れ替えのみ）。`dim0 == dim1` は恒等 view として
    /// 許容する（`Tensor::transpose` 自体が同じ挙動）。
    ///
    /// 検査順序: ①`dim0`／`dim1` が rank 範囲内 → ②`Tape::push_view` で
    /// ホスト値を持たないノードとして記録する（`reshape` と異なり
    /// transpose は非 contiguous 化しても失敗しない演算のため実体化前
    /// 検査は軸範囲のみで足りる。ただし `Tape::push_view` の呼び出し
    /// 契約〈`input` 事前実体化〉を満たすため、軸検査の後に層 1で入力を
    /// 実体化してから記録する）。
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim0 >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim0,
                rank,
            }));
        }
        if dim1 >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim1,
                rank,
            }));
        }
        let mut out_shape = in_shape;
        out_shape.swap(dim0, dim1);

        // `Tape::push_view` の呼び出し契約（`input` は push 前に実体化
        // 済み）を満たす。`Ref` を保持したまま `push_view` を呼ばない
        // よう、実体化はスコープ内で完結させる。
        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self.tape.push_view(
            Op::Transpose {
                input: self.id,
                dim0,
                dim1,
            },
            out_shape,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 任意軸並べ替え（view 系ノード。イシュー #1597）。`transpose` の
    /// 2 軸限定を補う N 階一般対応版（`Tensor::permute`／ONNX
    /// `Transpose` と同じ規約: `perm[k]` は出力軸 `k` が指す入力軸）。
    /// 常に zero-copy（strides の並べ替えのみ）。
    ///
    /// 検査順序: ①`perm` の長さが rank と一致（不一致は
    /// `ShapeError::RankMismatch`）→ ②各軸が範囲内（`AxisOutOfRange`）
    /// かつ重複なし（`DuplicateAxis`）→ ③層 1 で入力を実体化してから
    /// `Tape::push_view`（`transpose` と同じ理由で、実体化前検査は
    /// shape 情報のみで足りるため軸検査の後に行う）。
    ///
    /// `perm` が 2 軸のみを入れ替える順列（例 `[1, 0]`）の場合、出力
    /// strides は `Var::transpose(0, 1)` と同一になるため、CPU／Metal
    /// の NT/TN 転置入口（#1213／#1215）の高速経路は本メソッド経由でも
    /// 従来どおり到達する（strides 判定のため）。
    pub fn permute(&self, perm: &[usize]) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if perm.len() != rank {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: rank,
                actual: perm.len(),
            }));
        }
        let mut seen = vec![false; rank];
        for &axis in perm {
            if axis >= rank {
                return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                    axis,
                    rank,
                }));
            }
            if seen[axis] {
                return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis }));
            }
            seen[axis] = true;
        }
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();

        // `Tape::push_view` の呼び出し契約（`input` は push 前に実体化
        // 済み）を満たす（`transpose` と同じ規律）。
        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self.tape.push_view(
            Op::Permute {
                input: self.id,
                perm: perm.to_vec(),
            },
            out_shape,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// RNN（tanh 版）セル 1 step（イシュー #1647・設計 `docs/autodiff-
    /// rnn-cell-tape-design.md` 決定 1・4・5）。
    /// `h_t = tanh(x·W_ih + b_ih + h_{t-1}·W_hh + b_hh)`。
    ///
    /// `self` が `x_t: [B, D]`、`h_prev: [B, H]`、`p.w_ih: [D, H]`、
    /// `p.w_hh: [H, H]`、`p.b_ih`／`p.b_hh` はいずれも `[H]`（両方
    /// `Some` か両方 `None`。片方のみは [`AutodiffError::InvalidArgument`]）。
    /// `D == 0`／`H == 0` は zero-K ガードとして拒否する。
    pub fn rnn_cell(
        &self,
        h_prev: &Var<'t>,
        p: GateParams<'_, 't>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(h_prev)?;
        self.check_same_tape(p.w_ih)?;
        self.check_same_tape(p.w_hh)?;
        if let Some(b) = p.b_ih {
            self.check_same_tape(b)?;
        }
        if let Some(b) = p.b_hh {
            self.check_same_tape(b)?;
        }
        check_bias_pair(p.b_ih, p.b_hh)?;

        let x_shape = self.shape();
        let h_shape = h_prev.shape();
        let w_ih_shape = p.w_ih.shape();
        let w_hh_shape = p.w_hh.shape();
        require_rank2_all(&[&x_shape, &h_shape, &w_ih_shape, &w_hh_shape], "rnn_cell")?;
        let hidden = h_shape[1];
        require_positive_dims(x_shape[1], hidden, "rnn_cell")?;

        let out_ih = matmul_out_shape(&x_shape, &w_ih_shape)?;
        let out_hh = matmul_out_shape(&h_shape, &w_hh_shape)?;
        require_same_shape(&out_ih, &out_hh)?;
        if out_ih[1] != hidden {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: out_ih,
                rhs: vec![h_shape[0], hidden],
            }));
        }
        if let Some(b) = p.b_ih {
            require_same_shape(&b.shape(), &[hidden])?;
        }
        if let Some(b) = p.b_hh {
            require_same_shape(&b.shape(), &[hidden])?;
        }

        let (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let x_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let h_prev_val = materialize_fallible(&nodes, ops, h_prev.id)?.clone();
            let w_ih_val = materialize_fallible(&nodes, ops, p.w_ih.id)?.clone();
            let w_hh_val = materialize_fallible(&nodes, ops, p.w_hh.id)?.clone();
            let b_ih_val = optional_materialize(&nodes, ops, p.b_ih)?;
            let b_hh_val = optional_materialize(&nodes, ops, p.b_hh)?;
            (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val)
        };

        let value = rnn_cell_forward_value(
            self.tape.ops(),
            &x_val,
            &h_prev_val,
            &CellWeights {
                w_ih: &w_ih_val,
                w_hh: &w_hh_val,
                b_ih: b_ih_val.as_ref(),
                b_hh: b_hh_val.as_ref(),
            },
        )?;

        let id = self.tape.push_eager(
            Op::RnnCell {
                x: self.id,
                h_prev: h_prev.id,
                w_ih: p.w_ih.id,
                w_hh: p.w_hh.id,
                b_ih: p.b_ih.map(|b| b.id),
                b_hh: p.b_hh.map(|b| b.id),
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `shape` へブロードキャストする view 系ノード（イシュー #1597）。
    /// `Tensor::broadcast_to`（NumPy `broadcast_to` 相当）と同じ規約:
    /// 拡張軸（元の軸長 1 が `shape` 側で 1 より大きい値に広がる軸）は
    /// stride 0 の view になり zero-copy。同一 shape への broadcast は
    /// 恒等 view として許容する。
    ///
    /// 検査順序: ①要素数オーバーフロー検査（`reshape` と同じ自前
    /// `checked_mul` 実装。`tensor-core` は `checked_numel` を非公開に
    /// しているため）→ ②`shape.len() < rank` または各軸が
    /// `src == dst || src == 1` を満たさない場合は
    /// `ShapeError::BroadcastIncompatible` → ③層 1 で入力を実体化して
    /// から `Tape::push_view`。
    ///
    /// VJP（`grad.rs`）は `Op::Add`／`Op::Mul` の暗黙ブロードキャストと
    /// 同じ `reduce_to_shape` 縮約を使うため、勾配バッファの確保を
    /// 伴う（`reshape`／`transpose`／`permute` の VJP は zero-copy だが
    /// 本メソッドの VJP は異なる。`tape::Op::BroadcastTo` doc 参照）。
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        // `checked_numel` 相当のオーバーフロー検査（`reshape` と同じ
        // 自前実装。REQ-8 趣旨の境界検査 A03 対策）。
        let out_numel = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        if out_numel.is_none() {
            return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
        }
        if shape.len() < in_shape.len() {
            return Err(AutodiffError::Shape(ShapeError::BroadcastIncompatible {
                lhs: in_shape,
                rhs: shape.to_vec(),
            }));
        }
        let offset_axes = shape.len() - in_shape.len();
        for (&src, &dst) in in_shape.iter().zip(&shape[offset_axes..]) {
            if src != dst && src != 1 {
                return Err(AutodiffError::Shape(ShapeError::BroadcastIncompatible {
                    lhs: in_shape,
                    rhs: shape.to_vec(),
                }));
            }
        }

        // `Tape::push_view` の呼び出し契約（`input` は push 前に実体化
        // 済み）を満たす。
        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self
            .tape
            .push_view(Op::BroadcastTo { input: self.id }, shape.to_vec());
        Ok(Var::from_raw(self.tape, id))
    }

    /// [`Var::broadcast_to`] の PyTorch 名別名（`Tensor.expand`。イシュー
    /// #1597）。負値（-1 で当該軸を維持する PyTorch の記法）は
    /// `shape: &[usize]` の型上表現できないため非対応——呼び出し側は
    /// 維持したい軸の実サイズを明示的に渡すこと。
    pub fn expand(&self, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
        self.broadcast_to(shape)
    }

    /// 長さ 1 の軸を除去する view 系ノード（`Var::reshape` への委譲。
    /// イシュー #1597）。
    ///
    /// - `dim: None` — 長さ 1 の軸をすべて除去する（NumPy／PyTorch
    ///   `squeeze()` と同じ）。
    /// - `dim: Some(d)` — 軸 `d` が rank 範囲外なら
    ///   `ShapeError::AxisOutOfRange`。`d` が範囲内だが `shape[d] != 1`
    ///   の場合は **PyTorch 準拠の no-op**（shape を変えず `reshape` を
    ///   記録する。numpy／TensorFlow はここをエラーにするが、適合する
    ///   `ShapeError` variant が存在せず、crates.io 公開クレート
    ///   `tensor-core` の公開 enum への variant 追加は semver 可視の
    ///   変更になるため、本イシューでは PyTorch 方式を採用する）。
    ///
    /// 非 contiguous な入力（例: `permute`／`transpose`／`broadcast_to`
    /// の直後）に対する制約は `reshape` と同じ（`ShapeError::
    /// NonContiguousReshape`）。
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        let out_shape: Vec<usize> = match dim {
            None => in_shape.into_iter().filter(|&d| d != 1).collect(),
            Some(d) => {
                if d >= rank {
                    return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                        axis: d,
                        rank,
                    }));
                }
                if in_shape[d] != 1 {
                    // PyTorch 準拠 no-op（上記 doc 参照）。
                    in_shape
                } else {
                    let mut out = in_shape;
                    out.remove(d);
                    out
                }
            }
        };
        self.reshape(&out_shape)
    }

    /// 長さ 1 の軸を挿入する view 系ノード（`Var::reshape` への委譲。
    /// イシュー #1597）。`dim` は挿入後の rank（`rank + 1`）に対する
    /// 軸位置として扱うため有効範囲は `0..=rank`（PyTorch
    /// `unsqueeze` と同じ: 末尾への挿入 `dim == rank` を許容する）。
    /// 範囲外は `ShapeError::AxisOutOfRange { axis: dim, rank: rank + 1
    /// }`。
    ///
    /// 非 contiguous な入力に対する制約は `reshape` と同じ。
    pub fn unsqueeze(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim > rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank: rank + 1,
            }));
        }
        let mut out_shape = in_shape;
        out_shape.insert(dim, 1);
        self.reshape(&out_shape)
    }

    /// `[start_dim, end_dim]`（両端含む）の連続する軸を 1 軸へ潰す
    /// view 系ノード（`Var::reshape` への委譲。イシュー #1597）。
    /// PyTorch `torch.flatten(start_dim, end_dim)` と同じ規約。
    ///
    /// `end_dim >= rank` または `start_dim > end_dim` は
    /// `ShapeError::AxisOutOfRange`（後者は `axis: start_dim` として
    /// 報告する）。rank 0（スカラー）は `(start_dim, end_dim) ==
    /// (0, 0)` のみ許容し `[1]` を返す（PyTorch と同じ）。潰す軸区間
    /// の部分積自体は `usize::MAX` を含むゼロ長軸混在形状で
    /// オーバーフローしうるため `checked_mul` で検査し、オーバー
    /// フロー時は `ShapeError::ElementCountOverflow` を返す（総要素数
    /// が `reshape` 側で検査済みでも部分積は別途検査が要る）。
    ///
    /// 非 contiguous な入力に対する制約は `reshape` と同じ。
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if rank == 0 {
            if start_dim == 0 && end_dim == 0 {
                return self.reshape(&[1]);
            }
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: end_dim,
                rank,
            }));
        }
        if end_dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: end_dim,
                rank,
            }));
        }
        if start_dim > end_dim {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: start_dim,
                rank,
            }));
        }
        // 潰す軸区間の部分積は `checked_mul` で計算する（`reshape`／
        // `broadcast_to` と同じ自前実装。ゼロ長軸を含む形状〈例:
        // shape=[0, usize::MAX, 2]〉でも debug panic・release ラップを
        // 起こさないための境界検査。REQ-8 趣旨の境界検査 A03 対策）。
        let flattened = match in_shape[start_dim..=end_dim]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        {
            Some(n) => n,
            None => {
                return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
            }
        };
        let mut out_shape: Vec<usize> = in_shape[..start_dim].to_vec();
        out_shape.push(flattened);
        out_shape.extend_from_slice(&in_shape[end_dim + 1..]);
        self.reshape(&out_shape)
    }

    /// 複数の `Var` を `dim` 軸で連結する（`torch.cat` 相当。イシュー
    /// #1598）。関連関数（`&self` を取らない）——`Var` は `Copy` の
    /// ため `&[Var<'t>]` で受ける。
    ///
    /// 検査順序: ①空リストは `vars[0]` に触れる前に
    /// `AutodiffError::InvalidArgument` → ②先頭要素基準の
    /// `check_same_tape`（`AutodiffError::TapeMismatch`）→ ③shape 検査
    /// （[`fandhe_ai_tensor_core::concat_out_shape`]。rank 不一致・
    /// `dim` 範囲外・`dim` 以外の軸不一致・要素数オーバーフローを
    /// 個別 variant で報告）→ ④全入力を層 1 で実体化 → ⑤
    /// `Tape::push_eager` で記録する（`Op::Concat` doc 参照）。
    ///
    /// forward 値は `grad::concat_with_fallback`（`ops.concat` →
    /// `Unsupported` のときのみ `eval::concat`）で計算する。VJP
    /// （`grad.rs`）は各入力へ `upstream.narrow` を zero-copy に分配
    /// する。1 要素リストも通常どおり `Op::Concat` ノードを記録する
    /// （恒等コピー）。
    pub fn cat(vars: &[Var<'t>], dim: usize) -> Result<Var<'t>, AutodiffError> {
        let first = match vars.first() {
            Some(v) => v,
            None => {
                return Err(AutodiffError::InvalidArgument(
                    "Var::cat: vars must not be empty".into(),
                ));
            }
        };
        for v in &vars[1..] {
            first.check_same_tape(v)?;
        }
        let shapes: Vec<Vec<usize>> = vars.iter().map(|v| v.shape()).collect();
        let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
        let out_shape = concat_out_shape(&shape_refs, dim).map_err(AutodiffError::Shape)?;

        // `Tape::push_eager` に渡す forward 値を計算するため、全入力を
        // 層 1 で実体化してから所有値として持ち出す（`nodes` の
        // `RefCell` 借用を閉じてから `push_eager`〈`borrow_mut`〉を
        // 呼ぶ規律。モジュール冒頭コメント参照）。
        let materialized: Vec<Tensor<f32>> = {
            let nodes = first.tape.nodes.borrow();
            let ops = first.tape.ops();
            let mut out = Vec::with_capacity(vars.len());
            for v in vars {
                out.push(materialize_fallible(&nodes, ops, v.id)?.clone());
            }
            out
        };
        let refs: Vec<&Tensor<f32>> = materialized.iter().collect();
        let value = concat_with_fallback(first.tape.ops(), &refs, dim, &out_shape)?;

        let id = first.tape.push_eager(
            Op::Concat {
                inputs: vars.iter().map(|v| v.id).collect(),
                dim,
            },
            value,
        );
        Ok(Var::from_raw(first.tape, id))
    }

    /// 複数の `Var` を新規軸 `dim` で積み上げる（`torch.stack` 相当。
    /// イシュー #1598）。各要素を `unsqueeze(dim)` してから
    /// [`Self::cat`] する（PyTorch の定義そのもの）。関連関数。
    ///
    /// 検査順序: ①空リスト → `InvalidArgument`（`vars[0]` に触れる
    /// 前）→ ②先頭要素基準の `check_same_tape` → ③`dim <= rank`
    /// （`ShapeError::AxisOutOfRange { rank: rank + 1 }`）→ ④全要素の
    /// shape 完全一致（`ShapeError::ShapeMismatch`）→ ここまで全て
    /// 通過してから `unsqueeze` ノードを push する（失敗した `stack`
    /// がテープに中間ノードを残さないよう、`unsqueeze` を呼ぶ前に
    /// 全要素の contiguity も検査する）。
    ///
    /// `unsqueeze` は `reshape` へ委譲するため、**非 contiguous な
    /// 要素**（`permute`／`transpose` 直後）は
    /// `ShapeError::NonContiguousReshape` を返す（暗黙 `contiguous()`
    /// はしない。`Var::reshape` doc「案 A」と同じ規律）。
    pub fn stack(vars: &[Var<'t>], dim: usize) -> Result<Var<'t>, AutodiffError> {
        let first = match vars.first() {
            Some(v) => v,
            None => {
                return Err(AutodiffError::InvalidArgument(
                    "Var::stack: vars must not be empty".into(),
                ));
            }
        };
        for v in &vars[1..] {
            first.check_same_tape(v)?;
        }
        let in_shape = first.shape();
        let rank = in_shape.len();
        if dim > rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank: rank + 1,
            }));
        }
        for v in &vars[1..] {
            let s = v.shape();
            if s != in_shape {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: in_shape.clone(),
                    rhs: s,
                }));
            }
        }
        for v in vars {
            let nodes = v.tape.nodes.borrow();
            let val = materialize_fallible(&nodes, v.tape.ops(), v.id)?;
            if !val.is_contiguous() {
                return Err(AutodiffError::Shape(ShapeError::NonContiguousReshape));
            }
        }
        let unsqueezed: Vec<Var<'t>> = vars
            .iter()
            .map(|v| v.unsqueeze(dim))
            .collect::<Result<_, _>>()?;
        Self::cat(&unsqueezed, dim)
    }

    /// `[start, start+len)` を切り出す zero-copy view（`torch.narrow`
    /// 相当。イシュー #1598・#1599「narrow」行の解消）。
    ///
    /// 検査順序: ①`dim < rank`（`ShapeError::AxisOutOfRange`）→
    /// ②`start + len <= shape[dim]`（`checked_add`。
    /// `ShapeError::NarrowOutOfBounds`）→ ③層 1 で入力を実体化してから
    /// `Tape::push_view`（`transpose`／`permute` と同じ規律）。
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        let dim_size = in_shape[dim];
        let in_bounds = start.checked_add(len).is_some_and(|end| end <= dim_size);
        if !in_bounds {
            return Err(AutodiffError::Shape(ShapeError::NarrowOutOfBounds {
                dim,
                start,
                len,
                dim_size,
            }));
        }
        let mut out_shape = in_shape;
        out_shape[dim] = len;

        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self.tape.push_view(
            Op::Narrow {
                input: self.id,
                dim,
                start,
                len,
            },
            out_shape,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 指定した各長さ（`sizes`）で `dim` 軸を分割する（`torch.split`
    /// の list 形式相当。イシュー #1598）。各出力は [`Self::narrow`]
    /// （zero-copy view）。
    ///
    /// `sizes.iter().sum()`（`checked_add`）が `shape[dim]` と一致しな
    /// い場合 `ShapeError::ShapeMismatch { lhs: [shape[dim]], rhs:
    /// [sum] }` を返す。
    pub fn split_with_sizes(
        &self,
        sizes: &[usize],
        dim: usize,
    ) -> Result<Vec<Var<'t>>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        let dim_size = in_shape[dim];
        let mut sum: usize = 0;
        for &s in sizes {
            sum = sum
                .checked_add(s)
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        }
        if sum != dim_size {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![dim_size],
                rhs: vec![sum],
            }));
        }
        let mut out = Vec::with_capacity(sizes.len());
        let mut start = 0usize;
        for &len in sizes {
            out.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(out)
    }

    /// 先頭から `split_size` 刻みで `dim` 軸を分割する（`torch.split`
    /// の int 形式相当。末尾は端数。イシュー #1598）。
    ///
    /// `split_size == 0` は `shape[dim]` の値に依らず一律
    /// `AutodiffError::InvalidArgument`（PyTorch は `shape[dim] == 0`
    /// のとき許容するが、0 除算相当の分岐を持たない単純な契約を
    /// 優先する）。`shape[dim] == 0` かつ `split_size > 0` は PyTorch
    /// と同じく `[narrow(dim, 0, 0)]` の 1 要素を返す。
    pub fn split(&self, split_size: usize, dim: usize) -> Result<Vec<Var<'t>>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        if split_size == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Var::split: split_size must be nonzero".into(),
            ));
        }
        let dim_size = in_shape[dim];
        let mut sizes = Vec::new();
        if dim_size == 0 {
            sizes.push(0);
        } else {
            let mut remaining = dim_size;
            while remaining > 0 {
                let take = remaining.min(split_size);
                sizes.push(take);
                remaining -= take;
            }
        }
        self.split_with_sizes(&sizes, dim)
    }

    /// `dim` 軸を `chunks` 個以下に分割する（`torch.chunk` 相当。
    /// イシュー #1598）。`chunk_size = ceil(shape[dim] / chunks)`
    /// （`div_ceil`）を [`Self::split`] へ委譲する。
    ///
    /// `chunks == 0` は `AutodiffError::InvalidArgument`。
    /// `shape[dim] == 0` は `chunk_size` が 0 になり `split` の
    /// `split_size == 0` 規則と衝突するため、`split` へ委譲せず
    /// PyTorch と同じく `chunks` 個の空 `narrow(dim, 0, 0)` を返す
    /// （[`Self::split_with_sizes`] へ `[0; chunks]` を渡す）。
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Var<'t>>, AutodiffError> {
        if chunks == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Var::chunk: chunks must be nonzero".into(),
            ));
        }
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        let dim_size = in_shape[dim];
        if dim_size == 0 {
            return self.split_with_sizes(&vec![0; chunks], dim);
        }
        let chunk_size = dim_size.div_ceil(chunks);
        self.split(chunk_size, dim)
    }
    /// LSTM セル 1 step（イシュー #1647・設計 `docs/autodiff-rnn-cell-
    /// tape-design.md` 決定 1・1b・1c・4・5・12）。ゲート順は `i,f,g,o`
    /// （`p.w_ih`／`p.w_hh` は `[D, 4H]`／`[H, 4H]`、bias は `[4H]`）。
    /// 2 ノード（`Op::LstmCell` → `Op::LstmHidden`）を **この順で**
    /// push する（決定 1b の push 順序契約。backward の逆走査が
    /// `LstmHidden` を先に処理し `nodes[cell.0].op` を参照するため）。
    ///
    /// 戻り値は `(h_t, c_t)`。
    pub fn lstm_cell(
        &self,
        h_prev: &Var<'t>,
        c_prev: &Var<'t>,
        p: GateParams<'_, 't>,
    ) -> Result<(Var<'t>, Var<'t>), AutodiffError> {
        self.check_same_tape(h_prev)?;
        self.check_same_tape(c_prev)?;
        self.check_same_tape(p.w_ih)?;
        self.check_same_tape(p.w_hh)?;
        if let Some(b) = p.b_ih {
            self.check_same_tape(b)?;
        }
        if let Some(b) = p.b_hh {
            self.check_same_tape(b)?;
        }
        check_bias_pair(p.b_ih, p.b_hh)?;

        let x_shape = self.shape();
        let h_shape = h_prev.shape();
        let c_shape = c_prev.shape();
        let w_ih_shape = p.w_ih.shape();
        let w_hh_shape = p.w_hh.shape();
        require_rank2_all(
            &[&x_shape, &h_shape, &c_shape, &w_ih_shape, &w_hh_shape],
            "lstm_cell",
        )?;
        require_same_shape(&h_shape, &c_shape)?;
        let hidden = h_shape[1];
        require_positive_dims(x_shape[1], hidden, "lstm_cell")?;

        let out_ih = matmul_out_shape(&x_shape, &w_ih_shape)?;
        let out_hh = matmul_out_shape(&h_shape, &w_hh_shape)?;
        require_same_shape(&out_ih, &out_hh)?;
        let gate_width_4h = checked_gate_width(4, hidden)?;
        if out_ih[1] != gate_width_4h {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: out_ih,
                rhs: vec![h_shape[0], gate_width_4h],
            }));
        }
        if let Some(b) = p.b_ih {
            require_same_shape(&b.shape(), &[gate_width_4h])?;
        }
        if let Some(b) = p.b_hh {
            require_same_shape(&b.shape(), &[gate_width_4h])?;
        }

        let (x_val, h_prev_val, c_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let x_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let h_prev_val = materialize_fallible(&nodes, ops, h_prev.id)?.clone();
            let c_prev_val = materialize_fallible(&nodes, ops, c_prev.id)?.clone();
            let w_ih_val = materialize_fallible(&nodes, ops, p.w_ih.id)?.clone();
            let w_hh_val = materialize_fallible(&nodes, ops, p.w_hh.id)?.clone();
            let b_ih_val = optional_materialize(&nodes, ops, p.b_ih)?;
            let b_hh_val = optional_materialize(&nodes, ops, p.b_hh)?;
            (
                x_val, h_prev_val, c_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val,
            )
        };

        let out = lstm_cell_forward_values(
            self.tape.ops(),
            &x_val,
            &h_prev_val,
            &c_prev_val,
            &CellWeights {
                w_ih: &w_ih_val,
                w_hh: &w_hh_val,
                b_ih: b_ih_val.as_ref(),
                b_hh: b_hh_val.as_ref(),
            },
        )?;

        // 決定 1b: `gates` (`[B, 4H]`。列ブロック順 `i,f,g,o`) から
        // `LstmCell` payload（`i,f,g`。`[B, 3H]`）と `LstmHidden`
        // payload（`o`。`[B, H]`）を切り出す。`narrow` は zero-copy
        // view のため `contiguous()` で実体化してから非追跡 payload
        // として保持する（`Op` payload はホスト常駐の独立 `Tensor`
        // でなければならない。view のまま埋め込むと backward 時に
        // `resolve_view` が想定しない経路になる）。
        // `gate_width_4h` が overflow せず検証済み（上記）のため、
        // その内訳である `3 * hidden` も overflow しない
        // （`3 * hidden < 4 * hidden <= usize::MAX`）。
        let gate_width_3h = checked_gate_width(3, hidden)?;
        let gates_ifg = out
            .gates
            .narrow(1, 0, gate_width_3h)
            .map(|t| t.contiguous())?;
        let gate_o = out
            .gates
            .narrow(1, gate_width_3h, hidden)
            .map(|t| t.contiguous())?;

        let cell_id = self.tape.push_eager(
            Op::LstmCell {
                x: self.id,
                h_prev: h_prev.id,
                c_prev: c_prev.id,
                w_ih: p.w_ih.id,
                w_hh: p.w_hh.id,
                b_ih: p.b_ih.map(|b| b.id),
                b_hh: p.b_hh.map(|b| b.id),
                gates_ifg,
            },
            out.c,
        );
        let hidden_id = self.tape.push_eager(
            Op::LstmHidden {
                cell: cell_id,
                gate_o,
            },
            out.h,
        );
        Ok((
            Var::from_raw(self.tape, hidden_id),
            Var::from_raw(self.tape, cell_id),
        ))
    }

    /// GRU セル 1 step（イシュー #1647・設計 `docs/autodiff-rnn-cell-
    /// tape-design.md` 決定 1c・4・5・12。`reset_after=True` 規約）。
    /// ゲート順は `r,z,n`（`p.w_ih`／`p.w_hh` は `[D, 3H]`／`[H, 3H]`、
    /// bias は `[3H]`）。1 ノード（`Op::GruCell`）で表現する（GRU は
    /// LSTM と異なり単一出力 `h_t` のため 2 ノード分割は不要）。
    pub fn gru_cell(
        &self,
        h_prev: &Var<'t>,
        p: GateParams<'_, 't>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(h_prev)?;
        self.check_same_tape(p.w_ih)?;
        self.check_same_tape(p.w_hh)?;
        if let Some(b) = p.b_ih {
            self.check_same_tape(b)?;
        }
        if let Some(b) = p.b_hh {
            self.check_same_tape(b)?;
        }
        check_bias_pair(p.b_ih, p.b_hh)?;

        let x_shape = self.shape();
        let h_shape = h_prev.shape();
        let w_ih_shape = p.w_ih.shape();
        let w_hh_shape = p.w_hh.shape();
        require_rank2_all(&[&x_shape, &h_shape, &w_ih_shape, &w_hh_shape], "gru_cell")?;
        let hidden = h_shape[1];
        require_positive_dims(x_shape[1], hidden, "gru_cell")?;

        let out_ih = matmul_out_shape(&x_shape, &w_ih_shape)?;
        let out_hh = matmul_out_shape(&h_shape, &w_hh_shape)?;
        require_same_shape(&out_ih, &out_hh)?;
        let gate_width_3h = checked_gate_width(3, hidden)?;
        if out_ih[1] != gate_width_3h {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: out_ih,
                rhs: vec![h_shape[0], gate_width_3h],
            }));
        }
        if let Some(b) = p.b_ih {
            require_same_shape(&b.shape(), &[gate_width_3h])?;
        }
        if let Some(b) = p.b_hh {
            require_same_shape(&b.shape(), &[gate_width_3h])?;
        }

        let (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let x_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let h_prev_val = materialize_fallible(&nodes, ops, h_prev.id)?.clone();
            let w_ih_val = materialize_fallible(&nodes, ops, p.w_ih.id)?.clone();
            let w_hh_val = materialize_fallible(&nodes, ops, p.w_hh.id)?.clone();
            let b_ih_val = optional_materialize(&nodes, ops, p.b_ih)?;
            let b_hh_val = optional_materialize(&nodes, ops, p.b_hh)?;
            (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val)
        };

        let out = gru_cell_forward_values(
            self.tape.ops(),
            &x_val,
            &h_prev_val,
            &CellWeights {
                w_ih: &w_ih_val,
                w_hh: &w_hh_val,
                b_ih: b_ih_val.as_ref(),
                b_hh: b_hh_val.as_ref(),
            },
        )?;

        let id = self.tape.push_eager(
            Op::GruCell {
                x: self.id,
                h_prev: h_prev.id,
                w_ih: p.w_ih.id,
                w_hh: p.w_hh.id,
                b_ih: p.b_ih.map(|b| b.id),
                b_hh: p.b_hh.map(|b| b.id),
                gates_rzn: out.gates,
                q: out.q,
            },
            out.h,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `A^{-1}`（`A: [n,n]`）。イシュー #1621・親イシュー #1573
    /// 「Tier 2: 線形代数」・`docs/spec/04-requirements.md` REQ-9
    /// 2026-09-12 追記・`docs/autodiff-linalg-design.md`。
    ///
    /// 検査順序（`mse_loss_with` と同じ規律）: ①shape 検査（rank-2・
    /// 正方）→ ②入力実体化（層 1）→ ③`self.tape.ops().linalg_inv` を
    /// 試み `Err(BackendError::Unsupported(_))` のときのみ
    /// `eval::linalg::inv`（ホスト参照実装）へフォールバック（それ以外の
    /// エラー〈特異行列の `InvalidArgument` 等〉は伝播する。判定迂回
    /// 経路を作らない。`.claude/rules/security.md` A08）→ ④ノード記録。
    pub fn inv(&self) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        require_square(&shape, "Var::inv")?;
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_inv(&a_val) {
            Ok(v) => {
                verify_shape(v.shape(), &shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::inv(&a_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(Op::Inv { input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// `A X = B` を解く（`self: [n,n]`・`b: [n,k]` → `[n,k]`）。
    /// イシュー #1621。`inv` と同じ二段フォールバック規律。
    pub fn solve(&self, b: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(b)?;
        let a_shape = self.shape();
        let n = require_square(&a_shape, "Var::solve")?;
        let b_shape = b.shape();
        if b_shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: b_shape.len(),
            }));
        }
        if b_shape[0] != n {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::solve: a の行数 {n} と b の行数 {} が一致しない",
                b_shape[0]
            )));
        }
        let (a_val, b_val) = {
            let nodes = self.tape.nodes.borrow();
            let a_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let b_val = materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone();
            (a_val, b_val)
        };
        let expected_shape = vec![n, b_shape[1]];
        let value = match self.tape.ops().linalg_solve(&a_val, &b_val) {
            Ok(v) => {
                verify_shape(v.shape(), &expected_shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::solve(&a_val, &b_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(
            Op::Solve {
                a: self.id,
                b: b.id,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `det(A)`（`A: [n,n]` → スカラー `[]`）。イシュー #1621。特異行列は
    /// forward で `0.0`（エラーにしない。`eval::linalg::det` doc・
    /// `torch.linalg.det` と同じ挙動）。
    pub fn det(&self) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        require_square(&shape, "Var::det")?;
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_det(&a_val) {
            Ok(v) => {
                verify_shape(v.shape(), &[])?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::det(&a_val),
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(Op::Det { input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// Cholesky 分解（`A: [n,n]`〈対称正定値。下三角のみ読む〉→
    /// `L: [n,n]`〈下三角、`A = L Lᵀ`〉）。イシュー #1621。非正定値は
    /// `AutodiffError::InvalidArgument(_)`（CPU 本番経路・フォールバック
    /// とも `unify_backend_error` で同一 variant に統一済み。
    /// codex-review 指摘の是正: 以前は逆方向〈`Backend(InvalidArgument)`〉
    /// へ統一していたため、本番経路の数値エラーがドキュメント記載の
    /// variant と一致しなかった）。
    pub fn cholesky(&self) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        require_square(&shape, "Var::cholesky")?;
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_cholesky(&a_val) {
            Ok(v) => {
                verify_shape(v.shape(), &shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::cholesky(&a_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(Op::Cholesky { input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// reduced QR 分解（`A: [m,n]` → [`QrVars`]。`k = min(m,n)`）。
    /// イシュー #1621。
    ///
    /// **多出力の扱い**（`docs/autodiff-linalg-design.md` §3.3）: テープは
    /// 1 ノード 1 出力のため `Q`／`R` を別ノード（`Op::QrQ`／
    /// `Op::QrR`）として積む。各ノードは兄弟ノードの forward 値を
    /// payload として保持し、VJP（`grad.rs`）はコタンジェントに線形な
    /// ことを利用して各出力ノードの部分寄与を返す
    /// （`Tape::backward` が入力ノードへ合算する）。
    pub fn qr(&self) -> Result<QrVars<'t>, AutodiffError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
        let (m, n) = (shape[0], shape[1]);
        let k = m.min(n);
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let (q_val, r_val) = match self.tape.ops().linalg_qr(&a_val) {
            Ok(factors) => {
                verify_shape(factors.q.shape(), &[m, k])?;
                verify_shape(factors.r.shape(), &[k, n])?;
                (factors.q, factors.r)
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::qr(&a_val),
            Err(other) => return Err(unify_backend_error(other)),
        };
        let q_id = self.tape.push_eager(
            Op::QrQ {
                input: self.id,
                r: r_val.clone(),
            },
            q_val.clone(),
        );
        let r_id = self.tape.push_eager(
            Op::QrR {
                input: self.id,
                q: q_val,
            },
            r_val,
        );
        Ok(QrVars {
            q: Var::from_raw(self.tape, q_id),
            r: Var::from_raw(self.tape, r_id),
        })
    }

    /// reduced SVD（`A: [m,n]` → [`SvdVars`]。`k = min(m,n)`）。
    /// イシュー #1621。`qr` と同じ多出力設計（`Op::SvdU`／
    /// `Op::SvdS`／`Op::SvdVh`）。反復が収束しない場合は
    /// `AutodiffError::InvalidArgument(_)`（CPU 本番経路・フォールバック
    /// とも `unify_backend_error` で同一 variant に統一済み。
    /// codex-review 指摘の是正）。
    pub fn svd(&self) -> Result<SvdVars<'t>, AutodiffError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
        let (m, n) = (shape[0], shape[1]);
        let k = m.min(n);
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let (u_val, s_val, vh_val) = match self.tape.ops().linalg_svd(&a_val) {
            Ok(factors) => {
                verify_shape(factors.u.shape(), &[m, k])?;
                verify_shape(factors.s.shape(), &[k])?;
                verify_shape(factors.vh.shape(), &[k, n])?;
                (factors.u, factors.s, factors.vh)
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::svd(&a_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let u_id = self.tape.push_eager(
            Op::SvdU {
                input: self.id,
                s: s_val.clone(),
                vh: vh_val.clone(),
            },
            u_val.clone(),
        );
        let s_id = self.tape.push_eager(
            Op::SvdS {
                input: self.id,
                u: u_val.clone(),
                vh: vh_val.clone(),
            },
            s_val.clone(),
        );
        let vh_id = self.tape.push_eager(
            Op::SvdVh {
                input: self.id,
                u: u_val,
                s: s_val,
            },
            vh_val,
        );
        Ok(SvdVars {
            u: Var::from_raw(self.tape, u_id),
            s: Var::from_raw(self.tape, s_id),
            vh: Var::from_raw(self.tape, vh_id),
        })
    }

    /// 行列ノルム（`A: [m,n]`・`ord` → スカラー `[]`）。イシュー #1621。
    /// `ord` が [`MatrixNormOrd::Nuc`]／[`MatrixNormOrd::Spectral`] の
    /// 場合、フォールバック実装（`eval::linalg::matrix_norm`）内部で
    /// 特異値分解を用いる（`Var` 側で `svd` ノードを合成しない設計。
    /// `docs/autodiff-linalg-design.md` §3.2）。
    pub fn matrix_norm(&self, ord: MatrixNormOrd) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_matrix_norm(&a_val, ord) {
            Ok(v) => {
                verify_shape(v.shape(), &[])?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::matrix_norm(&a_val, ord)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(
            Op::MatrixNorm {
                input: self.id,
                ord,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }
}

/// [`Var::rnn_cell`]／[`Var::lstm_cell`]／[`Var::gru_cell`] へ渡す
/// ゲートパラメータのまとめ（イシュー #1647・設計 `docs/autodiff-rnn-
/// cell-tape-design.md` 決定 5・10）。PyTorch のパラメータ順
/// （`weight_ih, weight_hh, bias_ih, bias_hh`）を踏襲する。
pub struct GateParams<'a, 't> {
    pub w_ih: &'a Var<'t>,
    pub w_hh: &'a Var<'t>,
    pub b_ih: Option<&'a Var<'t>>,
    pub b_hh: Option<&'a Var<'t>>,
}

/// `b_ih`／`b_hh` が両方 `Some` か両方 `None` であることを検査する
/// （決定 4 のセル API 契約。片方のみは bias 加算の意味論が定義され
/// ないため拒否する）。
fn check_bias_pair(b_ih: Option<&Var<'_>>, b_hh: Option<&Var<'_>>) -> Result<(), AutodiffError> {
    if b_ih.is_some() != b_hh.is_some() {
        return Err(AutodiffError::InvalidArgument(
            "gate cell: b_ih and b_hh must be both Some or both None".to_string(),
        ));
    }
    Ok(())
}

/// 全入力が rank-2 であることを検査する（セル API 共通の shape 検査
/// 冒頭）。
fn require_rank2_all(shapes: &[&[usize]], op_name: &str) -> Result<(), AutodiffError> {
    for shape in shapes {
        if shape.len() != 2 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{op_name}: all operands must be rank-2 (got shape {shape:?})"
            )));
        }
    }
    Ok(())
}

/// `input_size`（`D`）・`hidden_size`（`H`）双方が 0 でないことを検査
/// する（zero-K ガード。決定 4）。
fn require_positive_dims(d: usize, hidden: usize, op_name: &str) -> Result<(), AutodiffError> {
    if d == 0 || hidden == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: input_size (D={d}) and hidden_size (H={hidden}) must both be > 0"
        )));
    }
    Ok(())
}

/// `gates * hidden`（ゲート幅）を `checked_mul` で検証する
/// （`nn::rnn::checked_gate_width` と同型）。
///
/// 本番経路 panic 禁止（AGENTS.md）: `lstm_cell`／`gru_cell` は
/// `h_prev.shape()[1]` から `hidden` を導出するが、`h_prev` が要素数
/// 0 の空 `Var`（例: `h_shape = [0, 1usize << 62]`）であれば
/// `require_positive_dims` の `hidden == 0` 検査を通過したまま
/// `hidden` が `usize::MAX` 近傍になりうる。`4 * hidden`／`3 * hidden`
/// を未検証のまま比較・`narrow` 幅へ使うと overflow により期待幅が
/// 周回し、不正な形状を誤って受理してしまう（イシュー #1647
/// codex-review P1 指摘）。
fn checked_gate_width(gates: usize, hidden: usize) -> Result<usize, AutodiffError> {
    gates.checked_mul(hidden).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "gates (={gates}) * hidden (={hidden}) overflowed usize"
        ))
    })
}

/// `Option<&Var>` を `Option<Tensor<f32>>` へ実体化する共通ヘルパー
/// （`nodes` の借用スコープ内で使う）。
fn optional_materialize(
    nodes: &[crate::tape::TapeNode],
    ops: &dyn BackendOps,
    var: Option<&Var<'_>>,
) -> Result<Option<Tensor<f32>>, AutodiffError> {
    match var {
        Some(v) => Ok(Some(materialize_fallible(nodes, ops, v.id)?.clone())),
        None => Ok(None),
    }
}

// =====================================================================
// セル forward の値計算（イシュー #1647）。`Var::{rnn_cell,lstm_cell,
// gru_cell}`（tape 経路）と `nn::rnn`（`forward_host`。tape 不要経路）
// の両方から呼ばれる共有ロジックであり、同じ関数を通すことで両経路の
// forward が bit-exact に一致することを構造的に保証する
// （`docs/autodiff-rnn-cell-tape-design.md` 決定 9）。
// =====================================================================

/// ゲート演算（RNN／LSTM／GRU セル）の重み・bias をまとめた引数束
/// （`clippy::too_many_arguments` 回避。イシュー #1647）。
/// [`rnn_cell_forward_value`]／[`lstm_cell_forward_values`]／
/// [`gru_cell_forward_values`] が共通で受け取る。
pub(crate) struct CellWeights<'a> {
    pub w_ih: &'a Tensor<f32>,
    pub w_hh: &'a Tensor<f32>,
    pub b_ih: Option<&'a Tensor<f32>>,
    pub b_hh: Option<&'a Tensor<f32>>,
}

/// RNN（tanh 版）セルの forward 値計算。`BackendOps::gemm_bias_act` を
/// 2 回（`x·W_ih+b_ih`・`h_prev·W_hh+b_hh`）呼び、`add` → `tanh` で
/// 閉じる（決定 1「RNN は新規カーネル不要」）。
pub(crate) fn rnn_cell_forward_value(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    w: &CellWeights<'_>,
) -> Result<Tensor<f32>, AutodiffError> {
    let pre_ih = ops.gemm_bias_act(x, w.w_ih, w.b_ih, Activation::None)?;
    let pre_hh = ops.gemm_bias_act(h_prev, w.w_hh, w.b_hh, Activation::None)?;
    let pre = ops.add(&pre_ih, &pre_hh)?;
    Ok(ops.tanh(&pre)?)
}

/// LSTM セルの forward 値計算。`pre = x·W_ih+b_ih + h_prev·W_hh+b_hh`
/// （`[B, 4H]`）を計算したのち `BackendOps::lstm_pointwise` へ渡す
/// （`Unsupported` のときのみ `eval::lstm_pointwise` へフォールバック。
/// A08: 判定迂回経路を作らない）。
pub(crate) fn lstm_cell_forward_values(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    c_prev: &Tensor<f32>,
    w: &CellWeights<'_>,
) -> Result<LstmPointwiseOutput, AutodiffError> {
    let pre_ih = ops.gemm_bias_act(x, w.w_ih, w.b_ih, Activation::None)?;
    let pre_hh = ops.gemm_bias_act(h_prev, w.w_hh, w.b_hh, Activation::None)?;
    let pre = ops.add(&pre_ih, &pre_hh)?;
    match ops.lstm_pointwise(&pre, c_prev) {
        Ok(v) => Ok(v),
        Err(BackendError::Unsupported(_)) => Ok(eval::lstm_pointwise(&pre, c_prev)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// GRU セルの forward 値計算。`pre_i = x·W_ih+b_ih`・
/// `pre_h = h_prev·W_hh+b_hh`（いずれも `[B, 3H]`。独立した 2 本の
/// GEMM のため 1 本に足し込まない点が LSTM と異なる）を計算したのち
/// `BackendOps::gru_pointwise` へ渡す（`Unsupported` のときのみ
/// `eval::gru_pointwise` へフォールバック）。
pub(crate) fn gru_cell_forward_values(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    w: &CellWeights<'_>,
) -> Result<GruPointwiseOutput, AutodiffError> {
    let pre_i = ops.gemm_bias_act(x, w.w_ih, w.b_ih, Activation::None)?;
    let pre_h = ops.gemm_bias_act(h_prev, w.w_hh, w.b_hh, Activation::None)?;
    match ops.gru_pointwise(&pre_i, &pre_h, h_prev) {
        Ok(v) => Ok(v),
        Err(BackendError::Unsupported(_)) => Ok(eval::gru_pointwise(&pre_i, &pre_h, h_prev)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// rank-2・正方であることを検査し、辺長 `n` を返す（線形代数演算
/// 共通のヘルパー。イシュー #1621）。非正方は `ShapeError` の既存
/// variant で意味的に表現できないため `AutodiffError::InvalidArgument`
/// とする（`cross_entropy_loss` の target 範囲検査と同方針）。
fn require_square(shape: &[usize], op_name: &str) -> Result<usize, AutodiffError> {
    if shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    if shape[0] != shape[1] {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: 正方行列（[n,n]）が必要（形状 {shape:?}）"
        )));
    }
    Ok(shape[0])
}

/// バックエンド実装（`BackendOps::linalg_*`）の戻り値 shape が契約
/// （doc comment）どおりであることを検証する（`mse_loss_with` の
/// 「バックエンド実装の契約を検証する」規律と同型。実装バグの黙認
/// 防止。`.claude/rules/security.md` A08）。
fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// CPU 本番経路（`BackendOps::linalg_*` が `Unsupported` 以外の `Err` を
/// 返した場合）の `BackendError` を、公開ドキュメント（`Var::inv` 等の
/// doc comment・`docs/autodiff-linalg-design.md` §3.5「エラー分類」）が
/// 約束する `AutodiffError` variant へ写像する。`BackendError::
/// InvalidArgument(_)`（特異行列・非正定値・非収束のいずれか）は
/// `eval::linalg`（ホスト参照実装。`Unsupported` 時のフォールバック
/// 経路）が返す `AutodiffError::InvalidArgument(_)` と同一 variant に
/// 揃える——以前は逆方向（フォールバック側を `Backend(InvalidArgument)`
/// へ包む）へ統一していたため、`fandhe_ai::tape()`（CPU 本番経路）を
/// 使う呼び出し元が公開ドキュメントどおり `AutodiffError::
/// InvalidArgument` を照合しても本番経路の数値エラーを捕捉できない
/// 不整合があった（codex-review 指摘。再試行はしない・エラー内容は
/// そのまま伝播するのみ）。`InvalidArgument` 以外の `BackendError`
/// （`Unsupported`〈本関数の呼び出し元では既に分岐済み〉・
/// `ShapeMismatch` 等）は意味を変えず `AutodiffError::Backend(_)` の
/// まま伝播する。
fn unify_backend_error(err: BackendError) -> AutodiffError {
    match err {
        BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
        other => AutodiffError::Backend(other),
    }
}

/// [`Var::qr`] の戻り値。`Q`（`[m,k]`）・`R`（`[k,n]`）は別テープノード
/// （多出力設計。`Op::QrQ`／`Op::QrR` doc 参照）。
#[derive(Debug, Clone, Copy)]
pub struct QrVars<'t> {
    /// `[m, k]`（`k = min(m, n)`）。列直交。
    pub q: Var<'t>,
    /// `[k, n]`（`k = min(m, n)`）。上三角・対角非負。
    pub r: Var<'t>,
}

/// [`Var::svd`] の戻り値。`U`（`[m,k]`）・`S`（`[k]`）・`Vh`（`[k,n]`）は
/// 別テープノード（多出力設計。`Op::SvdU`／`Op::SvdS`／`Op::SvdVh` doc
/// 参照）。
#[derive(Debug, Clone, Copy)]
pub struct SvdVars<'t> {
    /// `[m, k]`（`k = min(m, n)`）。列直交。
    pub u: Var<'t>,
    /// `[k]`。特異値（降順・非負）。
    pub s: Var<'t>,
    /// `[k, n]`（`k = min(m, n)`）。行直交（`V^T`）。
    pub vh: Var<'t>,
}

/// [`Var::host_view`] が返す借用ビュー（イシュー #1335。P1 是正で
/// `Tape` の借用を保持しない設計へ変更——`Var::host_view` ドキュメント
/// コメント参照）。
///
/// `Deref<Target = [f32]>` でスライスとして使う。内部に保持する
/// `Tensor<f32>` は `host_view()` 構築時に既に
/// [`fandhe_ai_tensor_core::Tensor::contiguous`] 済み（[`Tensor::as_slice`]
/// が必ず `Some` を返す状態）であり、`Deref::deref` はその場で
/// borrow-checker 上の `&self` の借用として `&[f32]` を返すだけで
/// 追加コピーを伴わない。
///
/// **寿命契約**: `VarHostView` はライフタイムパラメータを持たず
/// `Tape`／`RefCell` の借用を一切保持しないため、生存中に同じ `Var`／
/// `Tape` へノード追加演算（`add`/`matmul` 等）を呼んでも panic しない
/// （[`Var::value`] の借用注意とは異なる）。
#[derive(Debug)]
pub struct VarHostView {
    tensor: Tensor<f32>,
}

impl std::ops::Deref for VarHostView {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        // `host_view()` が `contiguous()` 済みの `Tensor` のみを格納する
        // ため `as_slice()` は必ず `Some`。防御的に `unwrap_or(&[])`。
        self.tensor.as_slice().unwrap_or(&[])
    }
}

#[cfg(test)]
mod linear_act_tests {
    use super::*;
    use crate::tape::Tape;

    /// codex-review 指摘（PR #1079・discussion_r3889050931）の実測検証:
    /// `linear_act` は bias が `[n]`（`weight` の列数）と厳密一致しない
    /// broadcast 可能な shape（ここでは `[1, n]`）でも `ShapeMismatch` を
    /// 返さず、`matmul` → `add`（NumPy 互換ブロードキャスト）→ `relu` の
    /// 非融合合成と bit 一致する結果を返すことを確認する。フォール
    /// バックは `linear_act`／呼び出し元ではなく `BackendOps::
    /// gemm_bias_act` 自身の契約（`tensor-core::backend_ops` の doc・
    /// 各バックエンドの `ComposedFallback` 分岐）で行われる（本メソッド
    /// の doc コメント参照）。`Linear`（`nn::linear`）は `from_parameters`
    /// で bias を `[out_features]` 厳密一致にしか構築できないため、この
    /// broadcast bias 経路は `Linear` 経由では到達できない
    /// （`pub(crate)` の `linear_act` を直接呼ぶ本テストでのみ検証可能）。
    #[test]
    fn linear_act_accepts_broadcastable_bias_not_strictly_matching_out_features() {
        let tape = Tape::new();
        // input: [2, 2]、weight: [2, 3] → out: [2, 3]。
        let input = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let weight = tape.var(&Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[2, 3]).unwrap());
        // bias: `[3]`（out_features 厳密一致）ではなく `[1, 3]`
        // （broadcast 可能だが厳密一致ではない shape）。
        let bias = tape.var(&Tensor::new(vec![10.0, -5.0, 0.0], &[1, 3]).unwrap());

        let fused = input
            .linear_act(&weight, Some(&bias), Activation::Relu)
            .expect("broadcast bias は ShapeMismatch にならず成功するはず");

        let composed = input
            .matmul(&weight)
            .and_then(|y| y.add(&bias))
            .map(|y| y.relu())
            .expect("非融合合成（matmul→add→relu）も同じ broadcast bias で成功するはず");

        assert_eq!(
            fused.value().as_slice().unwrap(),
            composed.value().as_slice().unwrap(),
            "broadcast bias 経路は融合・非融合合成で bit 一致するはず"
        );
    }
}

#[cfg(test)]
mod host_view_tests {
    use super::*;
    use crate::tape::Tape;

    /// P1 是正の回帰テスト（イシュー #1335 codex-review 指摘）: `host_view()`
    /// が返す `VarHostView` を保持したまま同じ `Tape` へノード追加演算
    /// （`add`）を呼んでも `RefCell` の二重可変借用 panic が起きないこと
    /// を確認する。是正前は `Ref<'a, [f32]>` をそのまま保持していたため
    /// `x.add(&x)` の `push_lazy` 内 `borrow_mut()` が panic していた
    /// （指摘の再現手順そのもの）。
    #[test]
    fn host_view_does_not_panic_when_tape_op_follows() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());

        let view = x.host_view();
        // `view` 生存中に同じ `Tape` へノード追加演算を呼ぶ（是正前は panic）。
        let result = x.add(&x).expect("同一 shape の加算は成功するはず");
        drop(view);

        assert_eq!(
            result.to_tensor().as_slice().unwrap(),
            &[2.0, 4.0, 6.0, 8.0],
            "host_view() 生存中の add は通常どおりの結果を返すはず"
        );
    }

    /// 非 contiguous（`transpose` 後）な `Var` でも `host_view()` が
    /// `Tape` の借用を持ち越さないことを確認する（`contiguous()` 分岐の
    /// 回帰カバレッジ）。
    #[test]
    fn host_view_on_transposed_var_does_not_panic_when_tape_op_follows() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap());
        let xt = x.transpose(0, 1).expect("transpose は 2 次元で成功する");

        let view = xt.host_view();
        let result = x
            .add(&x)
            .expect("transpose 済み view の生存中でも add は成功するはず");
        drop(view);

        assert_eq!(
            result.to_tensor().as_slice().unwrap(),
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        );
    }
}
