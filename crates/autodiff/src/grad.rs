//! 演算ごとの勾配関数（VJP: vector-Jacobian product）と `Op` 単位の
//! ディスパッチ入口 `vjp()`。
//!
//! TASK-1.5a（#16）が記録したテープ構造（`tape::Op`/`TapeNode`）に対し、
//! 「出力側勾配（upstream）→ 各入力 `NodeId` への勾配」の変換層を提供
//! する（spec 根拠: `docs/spec/05-tasks.md` TASK-1.5、
//! `docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/tape.rs` の
//! backward 実装）。`Tape::backward`（`backward.rs`・TASK-1.5c・#18）は
//! ノード列を発生順とは逆順に走査しながら本モジュールの `vjp()` を
//! 呼び、返り値（入力 `NodeId` ごとの勾配寄与）を蓄積する。**勾配の
//! 蓄積そのものは本モジュールの責務ではない**（`backward.rs` 側で
//! 複数の出力先から同一入力ノードへ流入する勾配を合算する）。
//!
//! 値計算は `eval.rs`（クレート非公開の暫定 CPU 参照実装）のヘルパー
//! を再利用し、forward と勾配計算で数式の実体を 2 か所に別実装しない
//! （PoC-v2-2 の方針を踏襲）。ただし `MatMul`／`LinearAct`／
//! `LinearResident` の GEMM 系 VJP（`matmul_vjp`・`Op::LinearResident`
//! の `d_weight`）はイシュー #1211 で `eval::matmul`（scalar 参照実装）
//! から `BackendOps::gemm_fp32_strict`（forward と同じ CPU BLIS／CUDA／
//! Metal カーネルを使うが、CUDA の TF32 opt-in フラグ
//! （`set_cuda_tf32_gemm_enabled`）の状態に関わらず常に FP32 厳密で
//! 計算する入口。`ops.gemm` をそのまま使うと backward が opt-in フラグ
//! に暗黙追従してしまい、`docs/cuda-tf32-optin-api-decision.md`・
//! `backend-cuda::precision` モジュール冒頭コメントの「学習経路は
//! スコープ外のまま FP32」契約に反するため区別する。codex-review
//! 指摘・PR #1223）へ切り替え済み。backward の支配的コスト（`docs/perf/
//! train-step-phase-breakdown.md` §11・§15）を forward と同じ既定 FMA
//! 契約・並列実装で計算するための変更で、`eval::matmul` は
//! `NaiveOps`／`TestOps`（compat・テスト経路）に限り引き続き使われる
//! （`docs/perf/train-backward-gemm-wiring.md`）。

use fandhe_ai_tensor_core::{
    Activation, BackendError, BackendOps, ShapeError, Tensor, row_norm_layout,
};

use crate::error::AutodiffError;
use crate::eval::{self, build_tensor, dense_vec};
use crate::tape::{
    NodeId, Op, ResidentBiasTarget, ResidentResolver, TapeId, TapeNode, materialize_fallible,
};
use crate::var::Reduction;

/// elementwise VJP（`Op::Mul`／`Op::Exp`／`Op::Tanh`／`Op::Sigmoid` の
/// 乗算、`backward.rs::accumulate` の fan-out 勾配合算）を
/// `BackendOps`（forward と同じ CPU 並列／CUDA／Metal カーネル）経由で
/// 計算するか、ホスト逐次参照実装（`eval::mul`／`eval::add`）のまま
/// にするかを切り替えるゲート（イシュー #1583）。#1211 が GEMM 系 VJP
/// （`matmul_vjp`）へ適用した「backward を forward と同じ実装で計算
/// する」方針の elementwise 版。
///
/// いずれも単一 IEEE 演算（乗算／加算 1 回）のみで縮約を含まないため、
/// `ops.mul`／`ops.add` と `eval::mul`／`eval::add` は run-to-run・
/// バックエンド間を問わず bit 同一（`.claude/rules/coding-rust.md`
/// 「バックエンド間数値一致は複合判定」が対象とする縮約系演算には
/// 該当しない）。**対象外**（本ゲートの影響を受けない・ホスト経路の
/// まま不変）: `Op::Relu`／`LinearAct`／`LinearResident` のマスク演算
/// （[`elementwise_mul_mask`]。#1577 の stride 対応 host 経路。
/// `BackendOps` にマスク演算面がなく追加は公開 trait 拡張のため別途
/// ユーザー承認事項）、`Op::Add` の broadcast 縮約（[`reduce_bias_grad`]
/// ／[`reduce_to_shape`]。f64 アキュムレータ統一・Metal 側
/// `BackendOps::sum` 未実装のため対象外）。
///
/// 実測・出荷判断の経緯は `docs/perf/elementwise-vjp-backend-ops.md`
/// を参照（事前登録規則はイシュー #1583 のコメントに固定済み）。
pub(crate) const ELEMENTWISE_VJP_VIA_BACKEND_OPS: bool = false;

/// [`ELEMENTWISE_VJP_VIA_BACKEND_OPS`] に従い `g ⊙ rhs`（elementwise
/// 積。broadcast 前提だが本関数の呼び出し元はいずれも同 shape で渡す）
/// を `ops.mul` または `eval::mul` で計算する。
///
/// フォールバックは [`BackendError::Unsupported`] の場合のみ
/// `eval::mul` へ切り替える（バックエンドがそもそも当該演算を持たない
/// 場合の救済。`matmul_vjp` と異なり無条件フォールバックを許すのは、
/// elementwise 積が単一 IEEE 演算で forward／backward・バックエンド間
/// を問わず bit 同一であり、フォールバックしても「backward だけ別の
/// 数値経路になる」ことがないため）。他のエラー（デバイス割当失敗等）
/// は `AutodiffError::Backend` として fail-closed に伝播する
/// （`.claude/rules/security.md` A08）。戻り値の shape が
/// `broadcast_shape(g, rhs)` と一致しない場合も fail-closed で
/// エラーにする（バックエンド実装のバグを静かに呑み込まない）。
fn vjp_elementwise_mul(
    ops: &dyn BackendOps,
    g: &Tensor<f32>,
    rhs: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    vjp_elementwise_mul_via(ops, g, rhs, ELEMENTWISE_VJP_VIA_BACKEND_OPS)
}

/// [`vjp_elementwise_mul`] の実体。ゲート値を引数として受け取ることで
/// ビルド時定数 [`ELEMENTWISE_VJP_VIA_BACKEND_OPS`] の値に関わらず
/// 両分岐を単体テストできるようにする（イシュー #1583）。
fn vjp_elementwise_mul_via(
    ops: &dyn BackendOps,
    g: &Tensor<f32>,
    rhs: &Tensor<f32>,
    via_backend_ops: bool,
) -> Result<Tensor<f32>, AutodiffError> {
    if !via_backend_ops {
        return Ok(eval::mul(g, rhs));
    }
    match ops.mul(g, rhs) {
        Ok(out) => {
            let expected = fandhe_ai_tensor_core::broadcast_shape(g.shape(), rhs.shape())
                .map_err(|err| AutodiffError::Backend(BackendError::ShapeMismatch(err)))?;
            if out.shape() != expected.as_slice() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: out.shape().to_vec(),
                        rhs: expected,
                    },
                )));
            }
            Ok(out)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::mul(g, rhs)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`ELEMENTWISE_VJP_VIA_BACKEND_OPS`] に従い `a + b`（同 shape 前提の
/// elementwise 和。`backward.rs::accumulate` の fan-out 勾配合算専用）
/// を `ops.add` または `eval::add` で計算する。エラー処理・フォール
/// バック方針は [`vjp_elementwise_mul`] と同一。`pub(crate)`:
/// `backward.rs::accumulate` から呼ばれる。
pub(crate) fn vjp_elementwise_add(
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    vjp_elementwise_add_via(ops, a, b, ELEMENTWISE_VJP_VIA_BACKEND_OPS)
}

/// [`vjp_elementwise_add`] の実体。[`vjp_elementwise_mul_via`] と同じ
/// 理由でゲート値を引数化する（イシュー #1583）。
fn vjp_elementwise_add_via(
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    via_backend_ops: bool,
) -> Result<Tensor<f32>, AutodiffError> {
    if !via_backend_ops {
        return Ok(eval::add(a, b));
    }
    match ops.add(a, b) {
        Ok(out) => {
            let expected = fandhe_ai_tensor_core::broadcast_shape(a.shape(), b.shape())
                .map_err(|err| AutodiffError::Backend(BackendError::ShapeMismatch(err)))?;
            if out.shape() != expected.as_slice() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: out.shape().to_vec(),
                        rhs: expected,
                    },
                )));
            }
            Ok(out)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::add(a, b)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// ノード 1 個分の VJP。`upstream`（出力側勾配）と記録済みノード列
/// `nodes` から、各入力 `NodeId` への勾配寄与を返す。`out_value` は
/// 当該ノードの forward 記録値で、`Exp`/`Tanh`/`Sigmoid`/`Max` が
/// 再計算を避けて再利用する（`Sigmoid` は TASK-9.1b・#92 で追加）。
/// `Op::Leaf` は入力を持たないため空 `Vec` を返す。
///
/// **TASK-12.1d（#164）**: `Add`／`Sum` の入力 shape は `TapeNode.shape`
/// （実体化なしに算出済み。`tape.rs`）から直接読み、実体化を要求しない
/// （`docs/fusion-graph-design.md` §3.5.1）。`MatMul`／`Mul`／`Relu`／
/// `Max`／`MseLoss`／`CrossEntropyLoss` は入力の実際の値を要するため、
/// forward 記録済みの未実体化ノードを [`materialize_fallible`]（層 1。
/// `run_fused` の失敗のうち `Unsupported` 以外は `?` で伝播する）経由で
/// 読む（`Var::value`〈層 2〉は呼ばない。§3.5.2）。
///
/// `resident`（イシュー #1022）: `Op::LinearResident` の VJP が
/// `weight`／`bias` のデバイス常駐バッファを取得するための
/// [`ResidentResolver`]。素の [`crate::tape::Tape::backward`] からは
/// `None` が渡り、`DeviceParamStore::backward`（`optim::device_store`）
/// 経由の呼び出し（`Tape::backward_with_resident`）でのみ `Some` になる
/// （`tape::Op::LinearResident` doc「素の `Tape::backward`（resolver
/// なし）では型付きエラー」参照）。`Op::ResidentLeaf` 自身は `Op::Leaf`
/// と同じく入力を持たないため `resident` を参照しない。
#[allow(clippy::too_many_arguments)]
pub(crate) fn vjp(
    op: &Op,
    out_value: &Tensor<f32>,
    upstream: &Tensor<f32>,
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    resident: Option<&dyn ResidentResolver>,
    // イシュー #1212 codex-review P0 追加是正: `Op::LinearResident` の
    // VJP が `ResidentResolver::fill_resident_weight_grad` へ「どの
    // テープ・どの世代を今まさに微分しているか」を伝えるため
    // （`tape::ResidentResolver::fill_resident_weight_grad` doc 参照）。
    // `backward_impl`（`backward.rs`）から差分対象 `Tape` 自身の
    // `id`／`epoch()` をそのまま渡す。
    tape_id: TapeId,
    tape_epoch: u64,
) -> Result<Vec<(NodeId, Tensor<f32>)>, AutodiffError> {
    // `Op` は `CrossEntropyLoss` の `targets: Tensor<i32>` payload
    // ゆえに `Copy` を持たない（`tape.rs::Op` doc 参照）。旧
    // `match *op`（`Copy` 前提の値コピー）を `op.clone()` に置き換え、
    // それ以外の分岐は変更しない。
    let contributions = match op.clone() {
        Op::Leaf => Vec::new(),
        Op::MatMul(a, b) => {
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let (da, db) = matmul_vjp(ops, a_val, b_val, upstream)?;
            vec![(a, da), (b, db)]
        }
        Op::Add(a, b) => {
            let a_shape = &nodes[a.0].shape;
            let b_shape = &nodes[b.0].shape;
            // イシュー #1566・PR #1659→#1665→#1666 取り込み後の追加
            // ユーザー承認（2026-09-12）: `Op::Add` の broadcast 縮約
            // のうち bias パターン（`upstream: [m, n]` → `[n]`／
            // `[1, n]` の行方向縮約。`reduce_bias_grad` の shape 構造
            // 判定と同一条件）に限り `reduce_bias_grad`（f64 相当の
            // アキュムレータ。`Op::LinearAct`／`Op::LinearResident` の
            // bias フォールバックと共通）へ委譲する。`LinearVars::
            // forward`（`nn/linear.rs`。`matmul → add` の非融合合成。
            // `nn::Linear` の既定 forward 経路）の bias 勾配はこの
            // `Op::Add` の VJP を経由するため、これまで `LinearAct`／
            // `LinearResident`（同一の bias 縮約が f64 相当）と
            // 数値方式が食い違っていた（`[1e8, 1.0, -1e8]` で結果が
            // 変わる）。条件を満たさない broadcast 形状（bias パターン
            // 以外の一般的な `Op::Add` 縮約）は `reduce_bias_grad` が
            // 内部で `reduce_to_shape`（`f32` 逐次和・任意 rank・任意軸
            // 対応）へそのまま委譲するため挙動を変えない（`reduce_bias_
            // grad` doc 参照）。
            let da = reduce_bias_grad(upstream, a_shape);
            let db = reduce_bias_grad(upstream, b_shape);
            vec![(a, da), (b, db)]
        }
        Op::Mul(a, b) => {
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let da = reduce_to_shape(&vjp_elementwise_mul(ops, upstream, b_val)?, a_val.shape());
            let db = reduce_to_shape(&vjp_elementwise_mul(ops, upstream, a_val)?, b_val.shape());
            vec![(a, da), (b, db)]
        }
        Op::Relu(a) => {
            // 劣勾配は x = 0 で 0 とする（PoC-v2-2 準拠）。NaN 入力は
            // マスク不成立（`v > 0.0` が false）となり勾配 0 を返す。
            // `upstream`（reuse backward の下流層からは非連続転置 view
            // でありうる）・`a_val` とも `elementwise_mul_mask` が
            // stride 対応で読む（イシュー #1577）。
            let a_val = materialize_fallible(nodes, ops, a)?;
            let da = elementwise_mul_mask(upstream, a_val, |v| v > 0.0);
            vec![(a, da)]
        }
        Op::Exp(a) => {
            // d/dx exp(x) = exp(x)。forward 記録値 `out_value` を
            // 再利用し `exp` を再計算しない。
            let da = vjp_elementwise_mul(ops, upstream, out_value)?;
            vec![(a, da)]
        }
        Op::Tanh(a) => {
            // d/dx tanh(x) = 1 - tanh(x)^2。同じく `out_value` を再利用。
            let factor = tanh_grad_factor(out_value);
            let da = vjp_elementwise_mul(ops, upstream, &factor)?;
            vec![(a, da)]
        }
        Op::Sigmoid(a) => {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))。
            // `Exp`/`Tanh` と同じく forward 記録値 `out_value`
            // （= sigmoid(x)）を再利用し再計算しない（TASK-9.1b・#92）。
            let factor = sigmoid_grad_factor(out_value);
            let da = vjp_elementwise_mul(ops, upstream, &factor)?;
            vec![(a, da)]
        }
        Op::Softmax { input, dim } => {
            // d/dx softmax(x) = y ⊙ (g − Σ_dim(g ⊙ y))（`y` = forward
            // 記録値 `out_value` = softmax(x)。`Exp`/`Sigmoid` と同じ
            // 「再計算しない」方針）。軸方向の縮約は f64 アキュムレータ
            // （要素積は f32 で確定してから f64 へ昇格。
            // `.claude/rules/coding-rust.md`「勾配の長軸縮約は f64
            // アキュムレータで統一する」）。
            let da = softmax_vjp_along(out_value, upstream, dim);
            vec![(input, da)]
        }
        Op::LogSoftmax { input, dim } => {
            // d/dx log_softmax(x) = g − exp(y) ⊙ Σ_dim(g)（`y` = forward
            // 記録値 `out_value` = log_softmax(x)）。軸方向の縮約
            // （`Σ_dim(g)`）は f64 アキュムレータ。
            let da = log_softmax_vjp_along(out_value, upstream, dim);
            vec![(input, da)]
        }
        Op::Sum { input, dim } => {
            let input_shape = &nodes[input.0].shape;
            let da = unreduce_broadcast(upstream, input_shape, dim);
            vec![(input, da)]
        }
        Op::Max { input, dim } => {
            let input_val = materialize_fallible(nodes, ops, input)?;
            let da = max_vjp(input_val, dim, out_value, upstream);
            vec![(input, da)]
        }
        Op::MseLoss {
            pred,
            target,
            reduction,
        } => {
            let pred_val = materialize_fallible(nodes, ops, pred)?;
            let target_val = materialize_fallible(nodes, ops, target)?;
            let n = pred_val.numel();
            let (dpred, dtarget) = if n == 0 {
                // `mse_loss_vjp` と同じゼロ除算回避（`scale` 計算前に
                // 早期 return。融合カーネル呼び出しを回避することで
                // `n == 0` を渡すバックエンド実装契約を単純に保つ）。
                let zeros = build_tensor(vec![0f32; 0], pred_val.shape());
                (zeros.clone(), zeros)
            } else {
                let g_value = dense_vec(upstream).first().copied().unwrap_or(0.0);
                let scale = mse_loss_scale(g_value, n, reduction);
                match ops.mse_loss_backward(pred_val, target_val, scale) {
                    Ok(dpred) => {
                        if dpred.shape() != pred_val.shape() {
                            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                                    lhs: dpred.shape().to_vec(),
                                    rhs: pred_val.shape().to_vec(),
                                },
                            )));
                        }
                        // `dTarget = −dPred`（`backend_ops.rs::BackendOps::
                        // mse_loss_backward` doc 参照）。カーネル側は
                        // `dPred` のみを計算する契約のため、符号反転は
                        // ホスト側の単純な逐次 map（新規 GPU カーネル
                        // 起動・D2H を増やさない）で行う。
                        let dtarget_data: Vec<f32> =
                            dense_vec(&dpred).iter().map(|&v| -v).collect();
                        let dtarget = build_tensor(dtarget_data, dpred.shape());
                        (dpred, dtarget)
                    }
                    Err(BackendError::Unsupported(_)) => {
                        mse_loss_vjp(pred_val, target_val, upstream, reduction)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                }
            };
            vec![(pred, dpred), (target, dtarget)]
        }
        Op::CrossEntropyLoss {
            logits,
            targets,
            class_dim,
            reduction,
        } => {
            let logits_val = materialize_fallible(nodes, ops, logits)?;
            let dlogits =
                cross_entropy_loss_vjp(logits_val, &targets, class_dim, reduction, upstream);
            // `targets` は非追跡（`Var`/`NodeId` を持たない）ため勾配
            // 寄与を返すのは `logits` の 1 系統のみ（`tape::Op::
            // CrossEntropyLoss` doc 参照）。
            vec![(logits, dlogits)]
        }
        // デバイス常駐パラメータの葉（イシュー #1022）。`Op::Leaf` と同じく
        // 入力を持たないため寄与なし（`tape::Op::ResidentLeaf` doc 参照）。
        Op::ResidentLeaf { .. } => Vec::new(),
        // デバイス常駐 weight（・bias）で forward した Linear 相当ノード
        // （イシュー #1022）。`resident`（`ResidentResolver`）経由でしか
        // `weight` の `DeviceBuffer<f32>` を取得できないため、`None` の
        // 場合は型付きエラーで拒否する（`tape::Op::LinearResident` doc
        // 「素の `Tape::backward`（resolver なし）では型付きエラー」）。
        Op::LinearResident {
            input,
            weight,
            bias,
            act,
        } => {
            let Some(resident) = resident else {
                return Err(AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LinearResident requires DeviceParamStore::backward (a plain \
                     Tape::backward cannot resolve the resident weight buffer)"
                        .to_string(),
                ));
            };
            // イシュー #1022 P1 是正（codex-review 指摘）: `weight`／
            // `bias` の `NodeId` は `DeviceParamStore::
            // register_resident_params`／`snapshot_resident_params` が発行した
            // `ResidentLeaf` から来るが、`ResidentLeaf` 自体はライフタイム
            // 引数のみで `Tape` の同一性を保証しない（`optim::device_store::
            // ResidentLeaf::tape_id` 検証は `linear_forward` 側の別途対応。
            // `optim/device_store.rs` モジュール冒頭参照）。ここでは
            // 縦深防御として `nodes[weight.0]` の直接添字アクセス（別
            // テープの葉が混入した場合に範囲外添字 panic・無関係ノード
            // 誤読の余地があった）を `nodes.get(...)` へ置き換え、
            // fail-closed に拒否する（`.claude/rules/security.md` A08）。
            let weight_node = nodes.get(weight.0).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LinearResident.weight node_id is out of range for this tape \
                     (contract violation: leaf registered on a different Tape?)"
                        .to_string(),
                )
            })?;
            let (store_id, slot) = match &weight_node.op {
                Op::ResidentLeaf { store_id, slot } => (*store_id, *slot),
                _ => {
                    return Err(AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LinearResident.weight does not point to an \
                         Op::ResidentLeaf node (contract violation)"
                            .to_string(),
                    ));
                }
            };
            let w_dev = resident.resident_buffer(store_id, slot)?;
            let x_val = materialize_fallible(nodes, ops, input)?;

            // epilogue activation のマスク段（イシュー #1044）。`act ==
            // Relu` の場合、フォワードで融合した ReLU の劣勾配
            // （`Op::Relu` の VJP と同じ `out_value > 0` 規約。`out_value`
            // は bias 加算後・activation 適用後の forward 記録値なので
            // ここから直接マスクを復元でき、前活性化の再計算・追加ノード
            // を必要としない）を先に適用し、以降は「非融合の
            // `Op::LinearResident`（`act: None`）の VJP と同じ勾配 `g`」
            // として扱う。`upstream` はこの層が最終出力層でない限り
            // 下流の `Op::LinearResident` d_input が返す非連続転置 view
            // でありうるが、`elementwise_mul_mask` が stride 対応で
            // 読むためコピーは発生しない（イシュー #1577）。
            let masked_upstream;
            let g: &Tensor<f32> = match act {
                Activation::None => upstream,
                Activation::Relu => {
                    masked_upstream = elementwise_mul_mask(upstream, out_value, |v| v > 0.0);
                    &masked_upstream
                }
                // `Activation` は `#[non_exhaustive]`（`tensor-core::
                // backend_ops`）のため、autodiff クレート外から見た未知の
                // 将来 variant に対しては、誤った勾配（マスクなし）を
                // 静かに返すのではなく fail-closed で拒否する
                // （`.claude/rules/security.md` A08）。
                _ => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "grad::vjp: Op::LinearResident has an unsupported Activation variant \
                         ({act:?}); the VJP mask is only defined for None/Relu"
                    )));
                }
            };

            // d_weight = x^T @ g（既存 `matmul_vjp` の `dB` と同一式。
            // `x`・`g` はいずれもホスト常駐）。イシュー #1211:
            // `ops.gemm_fp32_strict`（forward と同じ CPU BLIS／CUDA／
            // Metal カーネルを経由するが CUDA の TF32 opt-in フラグには
            // 追従しない入口。冒頭コメント参照）を経由するため、
            // `x_t`（`transpose2d` の zero-copy view）はバックエンド側の
            // `gemm`（`CpuBackendOps::gemm` は `gemm_fp32_strict` の既定
            // 実装がそのまま委譲する）実装依存で扱いが変わる。CPU は
            // イシュー #1213 で dense な転置 view（`strides() == [1,
            // shape()[0]]`）を判定できる限り `contiguous()` の再パック
            // コピーを経由せず BLIS packing 側で直接吸収する専用入口
            // （TN パターン。CPU 実装クレートの `gemm_blis_parallel_tn`）
            // へ渡す（`narrow` 後の転置・TT は一般 stride 非対応のため
            // 従来どおり `contiguous()` フォールバック）。CUDA（イシュー
            // #1214）も同型の判定で GPU 側 smem 転置カーネル → 既存 NN
            // GEMM カーネルへ渡す専用入口（`CudaGemm::run_tiled_f32_tn`）
            // を持つ。Metal（イシュー #1215）は片側転置（NT/TN）を
            // `layout::classify_2d` で分類できる場合に限り、`contiguous()`
            // を経由せず classic strided カーネル入口
            // （`gemm::MetalGemm::dispatch_strided_bias_act_prepared`）へ
            // 分岐する（既存 NN 経路 `dispatch_auto` とは別カーネルの
            // ため、数値契約は bit 一致ではなく REQ-2 統一複合判定。
            // `docs/matmul-vjp-zero-copy-decision.md` §4.4）。TT・分類
            // 不能形状は Metal でも従来どおり `contiguous()` を経由する。
            let x_t = transpose2d(x_val);

            // イシュー #1212: d_weight をホストへ戻さずデバイス常駐の
            // まま `resolver`（`DeviceParamStore`）の grad staging へ
            // 直接書き込めるか試みる（`ResidentResolver::
            // fill_resident_weight_grad`。既定 `Ok(false)`）。成功した
            // 場合、`weight` の勾配は `contributions` に含めない
            // （`Gradients::get()` からは「未到達」と区別できなくなる
            // が、公開 API から `Op::ResidentLeaf` の `Var` を得る経路は
            // 元々存在しないため実害はない。`tape::Op::ResidentLeaf`
            // doc・`optim::device_store::ResidentLeaf` doc 参照。
            // `DeviceParamStore::step` は自身の grad staging を直接
            // 参照するため `Gradients` 経由の読み出しを必要としない）。
            // `Unsupported`（バックエンドが `gemm_fp32_strict_into`／
            // `MemoryOps` を実装しない。現時点で CUDA／Metal はここに
            // 該当する）の場合のみ、従来どおりホスト経路
            // （`ops.gemm_fp32_strict`）へフォールバックする（判定迂回
            // を作らない。`.claude/rules/security.md` A08）。
            // イシュー #1212 codex-review P0 追加是正: `weight`
            // （`Op::LinearResident.weight`）は今まさに差分している
            // `weight_node` の `NodeId` そのもの。`tape_id`／
            // `tape_epoch` と併せて resident 書き込みの由来として
            // 実装側（`DeviceParamStore`）へ渡す（`ResidentResolver::
            // fill_resident_weight_grad` doc 参照）。
            //
            // イシュー #1563: この `fill_resident_weight_grad`
            // （encode-only。Metal では GPU コマンドバッファへ積むだけ
            // で同期しない）を、下の `gemm_resident_lhs`（d_input。
            // Metal では同期点を持つ）より **前** に呼ぶ。d_weight と
            // d_input は独立な計算（`x^T @ g` と `W @ g^T`）であり
            // どちらを先に encode しても出力は bit 同一（構造的に
            // 保証される順序無依存性）。この順序により、同じ層の
            // d_weight の GPU コマンドが d_input の同期点へ「合流」し、
            // 層ごとに開きっぱなしだったコマンドバッファが d_input の
            // 同期 1 回で一緒に flush・wait される（Metal の同期境界
            // 回収。`docs/backend-metal-command-batching-design.md`
            // §7.4）。CPU／CUDA は本経路に同期境界を持たないため本質
            // 的な影響はない（CUDA の `gemm_fp32_strict_into` NT/TN は
            // 内部 `stream.synchronize()` を持つが性能中立）。
            //
            // イシュー #1566: bias の `Op::ResidentLeaf` 解決を
            // `fill_resident_weight_grad` 呼び出しより前に行う（bias も
            // 同時に resident staging へ書き込めるか試みるため。
            // `docs/backend-metal-command-batching-design.md` §10
            // 「案 A′」）。bias が `Some` でも `Op::ResidentLeaf` でない
            // ／`store_id` が weight と異なる場合は `bias_target` を
            // `None` のままにし、bias は常にホスト `reduce_bias_grad`
            // フォールバックへ回す（`fill_resident_weight_grad` は
            // weight のみを試み `bias_filled: false` を返す）。
            // `nodes.get(...)` は `weight` と同じ理由（範囲外添字 panic
            // 防止・fail-closed）で経由する。
            let bias_node = match bias {
                Some(bias_id) => Some(nodes.get(bias_id.0).ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LinearResident.bias node_id is out of range for this \
                         tape (contract violation: leaf registered on a different Tape?)"
                            .to_string(),
                    )
                })?),
                None => None,
            };
            let bias_target = match (bias, bias_node) {
                (Some(bias_id), Some(node)) => match &node.op {
                    Op::ResidentLeaf {
                        store_id: bias_store_id,
                        slot: bias_slot,
                    } if *bias_store_id == store_id => Some(ResidentBiasTarget {
                        slot: *bias_slot,
                        node_id: bias_id,
                        shape: node.shape.clone(),
                    }),
                    // 別 store の葉、または `Op::ResidentLeaf` 以外
                    // （理論上到達しないはず——`DeviceParamStore::
                    // linear_forward` は bias も `ResidentLeaf` としてのみ
                    // 受け付ける——だが fail-closed に「resident 化を
                    // 試みない」側へ倒す。誤った勾配を書き込むより安全）。
                    _ => None,
                },
                _ => None,
            };

            let outcome = resident.fill_resident_weight_grad(
                ops,
                store_id,
                slot,
                tape_id,
                tape_epoch,
                weight,
                &x_t,
                g,
                bias_target,
            )?;

            // d_input^T = W @ g^T（`W: [k,n]`・`g: [m,n]` → `g^T: [n,m]`
            // → `tmp: [k,m]`）。`W` はデバイス常駐のまま
            // `ops.gemm_resident_lhs` へ渡し、ホストへ download しない
            // （本イシューの受け入れ条件の中核）。イシュー #1563: 本
            // 呼び出しの同期点は、直前に encode-only で積んだ同じ層の
            // d_weight のコマンドバッファも合流させて完了させる合流点
            // になる（`crates/backend-metal/src/ops.rs::
            // gemm_resident_lhs` doc 参照）。
            let g_t = transpose2d(g);
            let tmp = ops
                .gemm_resident_lhs(w_dev, &g_t)
                .map_err(AutodiffError::Backend)?;
            let d_input = transpose2d(&tmp);

            let mut contributions = vec![(input, d_input)];
            if !outcome.weight_filled {
                let d_weight = ops
                    .gemm_fp32_strict(&x_t, g)
                    .map_err(AutodiffError::Backend)?;
                contributions.push((weight, d_weight));
            }
            if let (Some(bias_id), Some(bias_node)) = (bias, bias_node)
                && !outcome.bias_filled
            {
                // bias の勾配は `Op::Add` の VJP と同じ縮約の基本形
                // （行方向ブロードキャストの逆演算）だが、`reduce_bias_
                // grad`（f64 アキュムレータ経由。上記 doc 参照）へ委譲
                // する——resident 経由で書き込めた場合（`outcome.
                // bias_filled`）はここへ来ない（weight と対称の
                // 「resident 成功時は無駄な計算をスキップする」最適化。
                // `fill_resident_weight_grad` doc 参照）が、resident
                // 非対応バックエンドのフォールバックがここに来るため、
                // resident 経路（f64 逐次和）と数値方式を揃える
                // 必要がある（イシュー #1566・PR #1659 codex-review P1）。
                let d_bias = reduce_bias_grad(g, &bias_node.shape);
                contributions.push((bias_id, d_bias));
            }
            contributions
        }
        Op::LinearAct {
            input,
            weight,
            bias,
            act,
        } => {
            let w_val = materialize_fallible(nodes, ops, weight)?;
            let x_val = materialize_fallible(nodes, ops, input)?;

            // epilogue activation のマスク段（`Op::LinearResident` と同じ
            // `out_value > 0` 規約。イシュー #1044）。`upstream` が
            // 非連続転置 view の場合の扱いは `Op::LinearResident` 分岐
            // の同型コメント（イシュー #1577）を参照。
            let masked_upstream;
            let g: &Tensor<f32> = match act {
                Activation::None => upstream,
                Activation::Relu => {
                    masked_upstream = elementwise_mul_mask(upstream, out_value, |v| v > 0.0);
                    &masked_upstream
                }
                // `Activation` は `#[non_exhaustive]`（`tensor-core::
                // backend_ops`）のため、autodiff クレート外から見た未知の
                // 将来 variant に対しては、誤った勾配（マスクなし）を
                // 静かに返すのではなく fail-closed で拒否する
                // （`.claude/rules/security.md` A08）。
                _ => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "grad::vjp: Op::LinearAct has an unsupported Activation variant \
                         ({act:?}); the VJP mask is only defined for None/Relu"
                    )));
                }
            };

            let (d_input, d_weight) = matmul_vjp(ops, x_val, w_val, g)?;
            let mut contributions = vec![(input, d_input), (weight, d_weight)];
            if let Some(bias_id) = bias {
                let bias_shape = &nodes[bias_id.0].shape;
                // `Op::LinearResident` の resident フォールバック（上記
                // `reduce_bias_grad` doc 参照）と数値方式を揃える
                // （fresh〈本 Op〉/reuse 間の一致。イシュー #1566・PR
                // #1659 codex-review P1）。
                let d_bias = reduce_bias_grad(g, bias_shape);
                contributions.push((bias_id, d_bias));
            }
            contributions
        }
        // view ノード（イシュー #1047・親 #1043「カーネル融合・autodiff
        // 実行モデルの強化」）。`Reshape`/`Transpose` は逆写像も同じ演算
        // 族（reshape は「元の shape へ戻す」・transpose は対合）で
        // 表現でき、いずれも zero-copy（`Tensor::reshape`/`transpose`
        // が `storage: Arc<Storage>` を共有するのみ）。中間バッファを
        // 持たないという本イシューの受け入れ条件は、forward（`tape.rs::
        // resolve_view`）だけでなく backward（本 VJP）でも成立する。
        Op::Reshape { input } => {
            let input_shape = &nodes[input.0].shape;
            // `upstream` は out_shape（このノード自身の shape）を持つ。
            // 非 contiguous（例: 上流に `transpose` が挟まる）な場合は
            // zero-copy な `reshape` が `ShapeError::NonContiguousReshape`
            // を返しうるため、その場合に限り `contiguous()`（明示コピー）
            // を経由してから戻す（勾配バッファ側の話であり、view ノード
            // 自身が確保を持つわけではない。zero-copy を優先する順序を
            // 明記する）。
            let da = match upstream.reshape(input_shape) {
                Ok(t) => t,
                Err(_) => upstream.contiguous().reshape(input_shape).unwrap_or_else(|_| {
                    debug_assert!(
                        false,
                        "grad::vjp: Op::Reshape の逆伝播で reshape が失敗した（forward 側の契約違反）"
                    );
                    upstream.clone()
                }),
            };
            vec![(input, da)]
        }
        Op::Transpose { input, dim0, dim1 } => {
            // transpose は対合（同じ軸で 2 回適用すると恒等）のため、
            // 逆伝播も同じ `dim0`/`dim1` で `upstream` を transpose する
            // だけで閉じる（zero-copy。`tape::Op::Transpose` doc 参照）。
            let da = upstream.transpose(dim0, dim1).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::Transpose の逆伝播で transpose が失敗した（forward 側の契約違反）"
                );
                upstream.clone()
            });
            vec![(input, da)]
        }
        Op::RmsNorm { input, weight, eps } => {
            // forward（`Var::rms_norm`）記録値 `out_value` からは `weight`
            // に 0 要素があると `x`（正規化前入力）を逆算できないため、
            // `input`（と `weight` があれば `weight`）を実体化し直して
            // 行内統計（`rstd`）を再計算する（`eval::row_rms_stats` と
            // 同じ縮約精度契約。`.claude/rules/coding-rust.md`）。
            let x_val = materialize_fallible(nodes, ops, input)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(nodes, ops, w)?.clone()),
                None => None,
            };
            let x_shape = x_val.shape().to_vec();
            let (rows, hidden) = row_norm_layout(&x_shape).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::RmsNorm の row_norm_layout が forward 側の契約に反して失敗した"
                );
                (0, 0)
            });
            let x_slice = dense_vec(&x_val);
            let w_slice = w_val.as_ref().map(dense_vec);
            let dy_slice = dense_vec(upstream);
            let (dx, dw) =
                rmsnorm_vjp_rows(&x_slice, w_slice.as_deref(), eps, rows, hidden, &dy_slice);
            let mut contributions = vec![(input, build_tensor(dx, &x_shape))];
            if let (Some(w), Some(dw)) = (weight, dw) {
                contributions.push((w, build_tensor(dw, &[hidden])));
            }
            contributions
        }
        Op::LayerNorm {
            input,
            weight,
            bias,
            eps,
        } => {
            // `Op::RmsNorm` と同じ理由で `input`／`weight` を実体化し直し
            // `mean`／`rstd` を再計算する（`eval::row_ln_stats`）。
            let x_val = materialize_fallible(nodes, ops, input)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(nodes, ops, w)?.clone()),
                None => None,
            };
            let x_shape = x_val.shape().to_vec();
            let (rows, hidden) = row_norm_layout(&x_shape).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::LayerNorm の row_norm_layout が forward 側の契約に反して失敗した"
                );
                (0, 0)
            });
            let x_slice = dense_vec(&x_val);
            let w_slice = w_val.as_ref().map(dense_vec);
            let dy_slice = dense_vec(upstream);
            let (dx, dw, db) = layer_norm_vjp_rows(
                &x_slice,
                w_slice.as_deref(),
                bias.is_some(),
                eps,
                rows,
                hidden,
                &dy_slice,
            );
            let mut contributions = vec![(input, build_tensor(dx, &x_shape))];
            if let (Some(w), Some(dw)) = (weight, dw) {
                contributions.push((w, build_tensor(dw, &[hidden])));
            }
            if let (Some(b), Some(db)) = (bias, db) {
                contributions.push((b, build_tensor(db, &[hidden])));
            }
            contributions
        }
        // RNN（tanh 版）セル 1 step（イシュー #1647・設計 `docs/autodiff-
        // rnn-cell-tape-design.md` 決定 1・5）。`out_value` は forward
        // 記録済みの `h_t`（= `tanh(pre)`）。`Op::Tanh` と同じ
        // `tanh_grad_factor` を再利用したのち、`gate_affine_vjp` の
        // 単方向版 `affine_vjp` を `x`/`w_ih` 側・`h_prev`/`w_hh` 側の
        // 2 回に分けて呼ぶ（RNN は列ブロック分割を持たないため
        // `col_start = 0`・`total_cols = H`）。
        Op::RnnCell {
            x,
            h_prev,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        } => {
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let factor = tanh_grad_factor(out_value);
            let d_pre = eval::mul(upstream, &factor);
            let (dx, dw_ih, db_ih) = affine_vjp(ops, x_val, w_ih_val, &d_pre, 0, total_cols)?;
            let (dh_prev, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre, 0, total_cols)?;
            let mut contributions = vec![(x, dx), (h_prev, dh_prev), (w_ih, dw_ih), (w_hh, dw_hh)];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // LSTM セルの `c_t` ノード（決定 1b）。`upstream` は
        // `backward_impl` の fan-in 蓄積により、`Op::LstmHidden` からの
        // `dc_from_h` 寄与と（多 step の場合）次 step の `Op::LstmCell`
        // からの `dc_prev` 寄与が既に合算された `dc` である。
        Op::LstmCell {
            x,
            h_prev,
            c_prev,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            gates_ifg,
        } => {
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let c_prev_val = materialize_fallible(nodes, ops, c_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let (d_pre_ifg, dc_prev) =
                match ops.lstm_cell_backward(&gates_ifg, c_prev_val, upstream) {
                    Ok(v) => v,
                    Err(BackendError::Unsupported(_)) => {
                        eval::lstm_cell_backward(&gates_ifg, c_prev_val, upstream)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                };
            let (dx, dw_ih, db_ih) = affine_vjp(ops, x_val, w_ih_val, &d_pre_ifg, 0, total_cols)?;
            let (dh_prev, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre_ifg, 0, total_cols)?;
            let mut contributions = vec![
                (x, dx),
                (h_prev, dh_prev),
                (c_prev, dc_prev),
                (w_ih, dw_ih),
                (w_hh, dw_hh),
            ];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // LSTM セルの `h_t` ノード（決定 1b・決定 1b 追記）。`cell` が
        // 指す先が必ず `Op::LstmCell` である push 順序契約（`tape::
        // Op::LstmCell` doc）を利用し、`nodes[cell.0].op` から
        // `x`/`h_prev`/`w_ih`/`w_hh`/`b_ih`/`b_hh` の `NodeId` を読み出す
        // （決定 1b 追記。`cell` 以外を指すことは想定しないため `_` 分岐
        // では型付きエラーで fail-closed に拒否し、パニックしない）。
        Op::LstmHidden { cell, gate_o } => {
            let c_val = materialize_fallible(nodes, ops, cell)?;
            let (d_pre_o, dc) = match ops.lstm_hidden_backward(c_val, &gate_o, upstream) {
                Ok(v) => v,
                Err(BackendError::Unsupported(_)) => {
                    eval::lstm_hidden_backward(c_val, &gate_o, upstream)
                }
                Err(other) => return Err(AutodiffError::Backend(other)),
            };
            let cell_node = nodes.get(cell.0).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LstmHidden.cell node_id is out of range for this tape \
                     (contract violation)"
                        .to_string(),
                )
            })?;
            let (x, h_prev, w_ih, w_hh, b_ih, b_hh) = match &cell_node.op {
                Op::LstmCell {
                    x,
                    h_prev,
                    w_ih,
                    w_hh,
                    b_ih,
                    b_hh,
                    ..
                } => (*x, *h_prev, *w_ih, *w_hh, *b_ih, *b_hh),
                _ => {
                    return Err(AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LstmHidden.cell does not point to an Op::LstmCell node \
                         (contract violation: push order invariant broken)"
                            .to_string(),
                    ));
                }
            };
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let hidden = d_pre_o.shape().get(1).copied().unwrap_or(0);
            let col_start = total_cols.saturating_sub(hidden);
            let (dx, dw_ih, db_ih) =
                affine_vjp(ops, x_val, w_ih_val, &d_pre_o, col_start, total_cols)?;
            let (dh_prev, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre_o, col_start, total_cols)?;
            let mut contributions = vec![
                (cell, dc),
                (x, dx),
                (h_prev, dh_prev),
                (w_ih, dw_ih),
                (w_hh, dw_hh),
            ];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // GRU セル 1 step（決定 1c・5。`reset_after=True` 規約）。`pre_i`
        // 側（`x`/`w_ih`）と `pre_h` 側（`h_prev`/`w_hh`）は独立した
        // GEMM のため、`d_pre_i`/`d_pre_h` をそれぞれ `affine_vjp` へ
        // 個別に渡す（RNN／LSTM の「1 個の d_pre を共有」とは異なる）。
        // `h_prev` への寄与は `dh_prev_direct`（`z` 経由の直接項）と
        // `d_pre_h` の affine 逆伝播の 2 系統あり、`backward.rs::
        // accumulate` が同一 `NodeId` への複数寄与を合算する契約
        // （本 `Vec` 内に 2 エントリを push するだけでよい）。
        Op::GruCell {
            x,
            h_prev,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            gates_rzn,
            q,
        } => {
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let (d_pre_i, d_pre_h, dh_prev_direct) =
                match ops.gru_backward(&gates_rzn, &q, h_prev_val, upstream) {
                    Ok(v) => v,
                    Err(BackendError::Unsupported(_)) => {
                        eval::gru_backward(&gates_rzn, &q, h_prev_val, upstream)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                };
            let (dx, dw_ih, db_ih) = affine_vjp(ops, x_val, w_ih_val, &d_pre_i, 0, total_cols)?;
            let (dh_prev_affine, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre_h, 0, total_cols)?;
            let mut contributions = vec![
                (x, dx),
                (w_ih, dw_ih),
                (w_hh, dw_hh),
                (h_prev, dh_prev_direct),
                (h_prev, dh_prev_affine),
            ];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // 線形代数（イシュー #1621・`docs/autodiff-linalg-design.md`
        // §3.4）。三角解法・特異値スケーリングを要する VJP 本体は
        // `eval::linalg`（`f64` 内部計算。数式の実体を二重管理しない）
        // に集約し、ここでは各 Op の入出力（forward 記録値・upstream・
        // 兄弟ノードの forward 値）を渡すだけに徹する。
        Op::Inv { input } => {
            // `out_value`（forward が返す `f32` 記録値の `A^{-1}`）は
            // 再利用せず、`input` から改めて `f64` で計算する
            // （`eval::linalg::inv_vjp` doc 参照。codex-review 指摘）。
            let a_val = materialize_fallible(nodes, ops, input)?;
            let da = eval::linalg::inv_vjp(a_val, upstream)?;
            vec![(input, da)]
        }
        Op::Solve { a, b } => {
            // `out_value`（forward の解 `X` の `f32` 記録値）は再利用
            // せず、`a`／`b` から改めて `f64` で計算する
            // （`eval::linalg::solve_vjp` doc 参照。codex-review 指摘）。
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let (da, db) = eval::linalg::solve_vjp(a_val, b_val, upstream)?;
            vec![(a, da), (b, db)]
        }
        Op::Det { input } => {
            let a_val = materialize_fallible(nodes, ops, input)?;
            let g_scalar = dense_vec(upstream).first().copied().unwrap_or(0.0);
            let da = eval::linalg::det_vjp(a_val, g_scalar)?;
            vec![(input, da)]
        }
        Op::Cholesky { input } => {
            let da = eval::linalg::cholesky_vjp(out_value, upstream)?;
            vec![(input, da)]
        }
        // reduced QR の多出力ノード（イシュー #1621・`tape::Op::QrQ`
        // doc「多出力の扱い」）。`QrQ` は `dQ = upstream`・`dR = 0`、
        // `QrR` はその逆として部分寄与を計算し、`Tape::backward` が
        // `input` ノードへ合算する。
        Op::QrQ { input, r } => {
            let dr_zero = build_tensor(vec![0.0; r.numel()], r.shape());
            let da = eval::linalg::qr_vjp(out_value, &r, upstream, &dr_zero)?;
            vec![(input, da)]
        }
        Op::QrR { input, q } => {
            let dq_zero = build_tensor(vec![0.0; q.numel()], q.shape());
            let da = eval::linalg::qr_vjp(&q, out_value, &dq_zero, upstream)?;
            vec![(input, da)]
        }
        // reduced SVD の多出力ノード（`tape::Op::SvdU` doc）。3 ノード
        // それぞれが自身のコタンジェントのみ非ゼロとして部分寄与を返す。
        Op::SvdU { input, s, vh } => {
            let da = eval::linalg::svd_vjp(out_value, &s, &vh, Some(upstream), None, None)?;
            vec![(input, da)]
        }
        Op::SvdS { input, u, vh } => {
            let da = eval::linalg::svd_vjp(&u, out_value, &vh, None, Some(upstream), None)?;
            vec![(input, da)]
        }
        Op::SvdVh { input, u, s } => {
            let da = eval::linalg::svd_vjp(&u, &s, out_value, None, None, Some(upstream))?;
            vec![(input, da)]
        }
        Op::MatrixNorm { input, ord } => {
            let a_val = materialize_fallible(nodes, ops, input)?;
            let g_scalar = dense_vec(upstream).first().copied().unwrap_or(0.0);
            let da = eval::linalg::matrix_norm_vjp(a_val, ord, g_scalar)?;
            vec![(input, da)]
        }
        // `Var::permute` が記録する view ノード（イシュー #1597）。
        // 逆写像は逆置換（`inverse_permutation`）で `upstream` を
        // permute するだけで閉じる（zero-copy。`tape::Op::Permute`
        // doc 参照）。
        Op::Permute { input, perm } => {
            let inv = inverse_permutation(&perm);
            let da = upstream.permute(&inv).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::Permute の逆伝播で permute が失敗した（forward 側の契約違反）"
                );
                upstream.clone()
            });
            vec![(input, da)]
        }
        // `Var::broadcast_to`（`Var::expand` はこれへ委譲）が記録する
        // view ノード（イシュー #1597）。`upstream` は out_shape
        // （ブロードキャスト後）を持つため、`Op::Add`/`Op::Mul` の
        // 暗黙ブロードキャストと**同じ数値契約**で入力 shape へ縮約
        // する（`tape::Op::BroadcastTo` doc 参照）。`reduce_bias_grad`
        // は `[1, n]` 行方向縮約等の特定パターンに限り f64 アキュムレ
        // ータ経路（`eval::reduce_bias_grad_rows`）へ委譲し、それ以外
        // は `reduce_to_shape`（f32 逐次和）へフォールバックする関数
        // であり、`Op::Add` の暗黙 broadcast 縮約（232〜233 行目）と
        // 同一の関数を呼ぶことで明示 broadcast・暗黙 broadcast 間の
        // 数値方式の食い違い（codex-review P1 是正）を防ぐ。
        Op::BroadcastTo { input } => {
            let input_shape = &nodes[input.0].shape;
            let da = reduce_bias_grad(upstream, input_shape);
            vec![(input, da)]
        }
        // `Var::cat` が記録するノード（イシュー #1598）。VJP は各入力
        // へ `upstream.narrow(dim, off_i, len_i)`（zero-copy view）を
        // 分配する（「Concat の VJP は Split（Narrow）」）。同一
        // `NodeId` の重複（`cat(&[x, x])`）は呼び出し元（`backward.rs::
        // accumulate`）が合算するため、ここでは単純に列挙する。
        Op::Concat { inputs, dim } => {
            let mut off = 0usize;
            let mut contributions = Vec::with_capacity(inputs.len());
            for input in inputs {
                let len = nodes[input.0].shape[dim];
                let da = upstream.narrow(dim, off, len).unwrap_or_else(|_| {
                    debug_assert!(
                        false,
                        "grad::vjp: Op::Concat の逆伝播で narrow が失敗した（forward 側の契約違反）"
                    );
                    upstream.clone()
                });
                contributions.push((input, da));
                off += len;
            }
            contributions
        }
        // `Var::narrow`（`split`／`split_with_sizes`／`chunk` の実体）が
        // 記録するノード（イシュー #1598）。「Split の VJP は Concat」
        // の原則どおり、選択されなかった前後の区間を zero-pad した
        // テンソルと `upstream` を `dim` で連結し入力 shape へ戻す
        // （`concat_with_fallback` を `Var::cat` の forward と共用。
        // 空区間はスキップして 1 要素連結〈恒等コピー〉に落とす）。
        Op::Narrow {
            input,
            dim,
            start,
            len,
        } => {
            let input_shape = nodes[input.0].shape.clone();
            let before_len = start;
            let after_len = input_shape[dim] - start - len;
            let mut before_shape = input_shape.clone();
            before_shape[dim] = before_len;
            let mut after_shape = input_shape.clone();
            after_shape[dim] = after_len;
            let before =
                eval::build_tensor(vec![0f32; before_shape.iter().product()], &before_shape);
            let after = eval::build_tensor(vec![0f32; after_shape.iter().product()], &after_shape);
            let mut pieces: Vec<&Tensor<f32>> = Vec::with_capacity(3);
            if before_len > 0 {
                pieces.push(&before);
            }
            pieces.push(upstream);
            if after_len > 0 {
                pieces.push(&after);
            }
            let da = concat_with_fallback(ops, &pieces, dim, &input_shape)?;
            vec![(input, da)]
        }
        // `Var::where_cond` が記録するノード（イシュー #1637）。
        // `cond` は forward 時点で実体化済みの f32 マスク
        // （`out_shape` ちょうど）を Op が保持する。本体は
        // [`where_vjp`]（単体テストから直接呼べるよう分離。
        // `matmul_vjp` と同じ切り出し方針）。
        Op::Where { cond, a, b } => {
            let a_shape = nodes[a.0].shape.clone();
            let b_shape = nodes[b.0].shape.clone();
            let (da, db) = where_vjp(&cond, upstream, &a_shape, &b_shape);
            vec![(a, da), (b, db)]
        }
        // `Var::masked_fill` が記録するノード（イシュー #1637）。
        // 本体は [`masked_fill_vjp`]。
        Op::MaskedFill { input, mask } => {
            let d_input = masked_fill_vjp(&mask, upstream);
            vec![(input, d_input)]
        }
    };
    Ok(contributions)
}

/// [`Op::Where`] の VJP 本体（イシュー #1637）。`cond` は `out_shape`
/// ちょうど（forward 時点で broadcast 済み）の f32 マスク。`Op::Mul`
/// と同じ「まず out_shape で計算してから `reduce_to_shape` で入力
/// shape へ縮約する」契約に従う: `da = reduce(mask_keep(g, cond, c !=
/// 0.0), a_shape)`・`db = reduce(mask_keep(g, cond, c == 0.0),
/// b_shape)`。
fn where_vjp(
    cond: &Tensor<f32>,
    upstream: &Tensor<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
) -> (Tensor<f32>, Tensor<f32>) {
    let da = elementwise_mul_mask(upstream, cond, |c| c != 0.0);
    let db = elementwise_mul_mask(upstream, cond, |c| c == 0.0);
    (reduce_to_shape(&da, a_shape), reduce_to_shape(&db, b_shape))
}

/// [`Op::MaskedFill`] の VJP 本体（イシュー #1637）。fill 位置
/// （`mask != 0.0`）の勾配は 0。`mask` は `input` と同 shape のため
/// broadcast 縮約は不要（`Op::Relu` の VJP と同型）。
fn masked_fill_vjp(mask: &Tensor<f32>, upstream: &Tensor<f32>) -> Tensor<f32> {
    elementwise_mul_mask(upstream, mask, |m| m == 0.0)
}

/// [`Op::Concat`] の forward（`Var::cat`）と [`Op::Narrow`] の VJP
/// （直上）が共用する連結ヘルパー（イシュー #1598）。`ops.concat` →
/// `Unsupported` のときのみ `eval::concat` へフォールバックする
/// （`Op::Softmax` の「バックエンド実装 → フォールバック」二段構成と
/// 同型。判定迂回経路を作らない。`.claude/rules/security.md` A08）。
///
/// バックエンド実装（`Unsupported` 以外）が返した出力 shape を
/// `out_shape` と照合し、不一致は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を返す
/// （実装バグの黙認防止）。
pub(crate) fn concat_with_fallback(
    ops: &dyn BackendOps,
    inputs: &[&Tensor<f32>],
    dim: usize,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.concat(inputs, dim) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::concat(inputs, dim, out_shape)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Where`] の forward（`Var::where_cond`）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1637）。
/// [`concat_with_fallback`] と同型: `ops.where_cond` →
/// `Unsupported` のときのみ `eval::where_cond` へフォールバックし、
/// それ以外のエラーは伝播する（判定迂回経路を作らない）。バックエンド
/// 実装が返した出力 shape を `out_shape` と照合し、不一致は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を返す。
pub(crate) fn where_cond_with_fallback(
    ops: &dyn BackendOps,
    cond: &Tensor<f32>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.where_cond(cond, a, b) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::where_cond(cond, a, b, out_shape)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::MaskedFill`] の forward（`Var::masked_fill`）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1637）。
/// [`where_cond_with_fallback`] と同型。出力 shape は `x` と恒等。
pub(crate) fn masked_fill_with_fallback(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    mask: &Tensor<f32>,
    value: f32,
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.masked_fill(x, mask, value) {
        Ok(v) => {
            if v.shape() != x.shape() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: x.shape().to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::masked_fill(x, mask, value)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// `Op::RmsNorm` の VJP 本体（イシュー #1596）:
/// `x̂ = x·r`（`r` = `rstd`）・`dx̂ = dy·w`（`w` なしは `dy`）・
/// `dx = r·(dx̂ − x̂·mean(dx̂·x̂))`・`dw = Σ_rows dy·x̂`。
///
/// 行内（`hidden` 軸）の `mean(dx̂·x̂)` は `softmax_vjp_along`
/// （`crates/autodiff/tests` 側の先例。要素積を `f32` で確定してから
/// `f64` へ昇格して蓄積し、`rstd` との最終乗算・`dy` からの減算は
/// `f64` のまま保持して 1 回だけ `f32` へ downcast する）と同じ overflow
/// 回避方針を踏襲する。`dw` の行方向（`rows` 軸）蓄積は
/// `.claude/rules/coding-rust.md`「勾配の長軸縮約の要素積は `f32` で
/// 確定してから `f64` へ昇格して蓄積する」契約に厳密に従う（コメント
/// が挙げる代表例そのもの）。`rows == 0 || hidden == 0` は
/// [`eval::rmsnorm_rows`] と同じ早期 return で空／ゼロ出力を返す。
fn rmsnorm_vjp_rows(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
    if rows == 0 || hidden == 0 {
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        return (dx, dw);
    }
    let inv_n = 1.0f64 / hidden as f64;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let rstd = eval::row_rms_stats(row, eps, inv_n);
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dy_row[i] * w[i],
                None => dy_row[i],
            }
        };
        let mut dot_acc = 0.0f64;
        for (i, &xv) in row.iter().enumerate() {
            let xhat = xv * rstd;
            let term = dxhat_at(i) * xhat;
            dot_acc += term as f64;
        }
        let mean_dot = dot_acc * inv_n;
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, (&xv, dxv)) in row.iter().zip(dx_row.iter_mut()).enumerate() {
            let xhat = xv * rstd;
            let dxhat = dxhat_at(i);
            let d = (rstd as f64) * (dxhat as f64 - (xhat as f64) * mean_dot);
            *dxv = d as f32;
        }
        if let Some(dw_acc) = dw_acc.as_mut() {
            for (i, (&xv, &dyv)) in row.iter().zip(dy_row.iter()).enumerate() {
                let xhat = xv * rstd;
                let term = dyv * xhat;
                dw_acc[i] += term as f64;
            }
        }
    }
    let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    (dx, dw)
}

/// `Op::LayerNorm` の VJP 本体（イシュー #1596）:
/// `x̂ = (x−μ)·r`・`dx̂ = dy·w`・`dx = r·(dx̂ − mean(dx̂) − x̂·mean(dx̂·x̂))`・
/// `dw = Σ_rows dy·x̂`・`db = Σ_rows dy`。[`rmsnorm_vjp_rows`] と同じ
/// f64 縮約方針（行内 `mean(dx̂)`／`mean(dx̂·x̂)` は要素を `f32` で確定
/// してから `f64` 蓄積・`dw`／`db` の行方向蓄積も同型）。`has_bias` は
/// forward で `bias` が `Some` だったか（`weight` の有無とは独立）を
/// 表し、`db` を計算するかどうかを決める。
fn layer_norm_vjp_rows(
    x: &[f32],
    w: Option<&[f32]>,
    has_bias: bool,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
    let mut db_acc: Option<Vec<f64>> = if has_bias {
        Some(vec![0.0f64; hidden])
    } else {
        None
    };
    if rows == 0 || hidden == 0 {
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        let db = db_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        return (dx, dw, db);
    }
    let n = hidden as f64;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let (mean, rstd) = eval::row_ln_stats(row, eps, hidden);
        // `mean`／`rstd` を `f64` のまま偏差計算に使い、`x̂` を確定する
        // 直前の 1 回だけ `f32` へ downcast する（forward `eval::
        // layer_norm_rows` と同じ理由。codex-review 指摘: `mean` の
        // 早期丸めは forward・backward 双方の `x̂` を歪める）。
        let xhat_at = |i: usize| -> f32 { ((row[i] as f64 - mean) * rstd) as f32 };
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dy_row[i] * w[i],
                None => dy_row[i],
            }
        };
        let mut sum_dxhat = 0.0f64;
        let mut dot_acc = 0.0f64;
        for i in 0..row.len() {
            let xhat = xhat_at(i);
            let dxhat = dxhat_at(i);
            sum_dxhat += dxhat as f64;
            let term = dxhat * xhat;
            dot_acc += term as f64;
        }
        // `mean` と同じ理由（`row_ln_stats` doc 参照）で、事前丸めした
        // 逆数との積ではなく `hidden` による直接除算で求める。
        let mean_dxhat = sum_dxhat / n;
        let mean_dot = dot_acc / n;
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, dxv) in dx_row.iter_mut().enumerate() {
            let xhat = xhat_at(i);
            let dxhat = dxhat_at(i);
            let d = rstd * (dxhat as f64 - mean_dxhat - (xhat as f64) * mean_dot);
            *dxv = d as f32;
        }
        if let Some(dw_acc) = dw_acc.as_mut() {
            for (i, &dyv) in dy_row.iter().enumerate() {
                let xhat = xhat_at(i);
                let term = dyv * xhat;
                dw_acc[i] += term as f64;
            }
        }
        if let Some(db_acc) = db_acc.as_mut() {
            for (acc, &dyv) in db_acc.iter_mut().zip(dy_row.iter()) {
                *acc += dyv as f64;
            }
        }
    }
    let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    let db = db_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    (dx, dw, db)
}

/// [`Op::Permute`] の VJP（`vjp` 内）が使う逆置換の算出（イシュー
/// #1597）。`perm[k] = p` は「出力軸 `k` が入力軸 `p` を指す」ことを
/// 表すため、逆写像 `inv` は `inv[p] = k` を満たす（`perm ∘ inv ==
/// identity`）。`Var::permute`（`var.rs`）が push 前に `perm` を
/// `0..rank` の順列として検査済み（長さ一致・範囲内・重複なし）のため、
/// 本関数は常に `perm` と同じ長さの妥当な順列を返す（infallible）。
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (k, &p) in perm.iter().enumerate() {
        inv[p] = k;
    }
    inv
}

/// ゲート演算（RNN／LSTM／GRU セル）の GEMM 部分の VJP 共通ヘルパー
/// （イシュー #1647・設計 `docs/autodiff-rnn-cell-tape-design.md` 決定
/// 5 の列ブロック配置に対応）。`d_pre_blk`（あるゲートブロックの
/// pre-activation 勾配。`[B, w]`）と、その GEMM の片側オペランド
/// （`input_val: [B, D]`・`weight_val: [D, total_cols]`）から、
/// `d_input = d_pre_blk · (weight[:, blk])ᵀ`（`[B, D]`）・
/// `d_weight`（`weight` と同じ `[D, total_cols]`。ブロック外はゼロ埋め）・
/// `d_bias`（`[total_cols]`。同じくブロック外はゼロ埋め）を計算する。
///
/// 全幅（`col_start == 0 && d_pre_blk` の列数 `== total_cols`）の場合は
/// `narrow`／embed を経由せず `weight_val`／結果をそのまま使う（RNN・
/// LSTM の `LstmCell` 分岐が該当。GRU・`LstmHidden` は常に部分幅）。
///
/// [`matmul_vjp`] と同じ `BackendOps::gemm_fp32_strict`（`eval::matmul`
/// へのフォールバックなし。A08）を経由する。
///
/// 戻り値 `(d_input, d_weight, d_bias)`。`clippy::type_complexity` 回避
/// のため [`AffineVjpOutput`] という名前を与える。
type AffineVjpOutput = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

fn affine_vjp(
    ops: &dyn BackendOps,
    input_val: &Tensor<f32>,
    weight_val: &Tensor<f32>,
    d_pre_blk: &Tensor<f32>,
    col_start: usize,
    total_cols: usize,
) -> Result<AffineVjpOutput, AutodiffError> {
    let block_width = d_pre_blk.shape().get(1).copied().unwrap_or(0);
    let in_dim = weight_val.shape().first().copied().unwrap_or(0);
    let full_width = col_start == 0 && block_width == total_cols;

    let weight_blk_owned;
    let weight_blk: &Tensor<f32> = if full_width {
        weight_val
    } else {
        weight_blk_owned = weight_val
            .narrow(1, col_start, block_width)
            .map(|t| t.contiguous())
            .unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "affine_vjp: narrow(1, col_start, block_width) が失敗した（呼び出し元の \
                     列ブロック整合違反）"
                );
                weight_val.contiguous()
            });
        &weight_blk_owned
    };
    let weight_blk_t = transpose2d(weight_blk);
    let d_input = ops
        .gemm_fp32_strict(d_pre_blk, &weight_blk_t)
        .map_err(AutodiffError::Backend)?;

    let input_t = transpose2d(input_val);
    let d_weight_blk = ops
        .gemm_fp32_strict(&input_t, d_pre_blk)
        .map_err(AutodiffError::Backend)?;
    let d_weight_full = if full_width {
        d_weight_blk
    } else {
        embed_columns_2d(&d_weight_blk, in_dim, total_cols, col_start)
    };

    // 勾配の長軸縮約は f64（`.claude/rules/coding-rust.md`）。
    // `d_pre_blk: [rows, block_width]` → `[block_width]` は行方向
    // （軸 0）縮約であり `reduce_bias_grad` の row-axis 判定を満たす
    // ため f64 アキュムレータ経路（`eval::reduce_bias_grad_rows`）へ
    // 委譲する（RNN／LSTM／GRU 共通 bias 勾配。イシュー #1647
    // codex-review P1 是正: 旧 `reduce_to_shape` は f32 逐次和のため
    // 大きく相殺する上流勾配で桁落ちする）。
    let d_bias_blk = reduce_bias_grad(d_pre_blk, &[block_width]);
    let d_bias_full = if full_width {
        d_bias_blk
    } else {
        embed_columns_1d(&d_bias_blk, total_cols, col_start)
    };

    Ok((d_input, d_weight_full, d_bias_full))
}

/// `partial: [rows, block_width]` を `[rows, total_cols]` の零行列の
/// `[col_start, col_start+block_width)` 列範囲へ埋め込む（[`affine_vjp`]
/// の重み勾配の列ブロック配置を復元するためのホスト側 scatter）。
fn embed_columns_2d(
    partial: &Tensor<f32>,
    rows: usize,
    total_cols: usize,
    col_start: usize,
) -> Tensor<f32> {
    let block_width = partial.shape().get(1).copied().unwrap_or(0);
    let partial_data = dense_vec(partial);
    let mut out = vec![0f32; rows * total_cols];
    for r in 0..rows {
        for c in 0..block_width {
            out[r * total_cols + col_start + c] = partial_data[r * block_width + c];
        }
    }
    build_tensor(out, &[rows, total_cols])
}

/// `partial: [block_width]` を `[total_cols]` の零ベクトルの
/// `[col_start, col_start+block_width)` 範囲へ埋め込む（[`affine_vjp`]
/// の bias 勾配の列ブロック配置を復元するためのホスト側 scatter）。
fn embed_columns_1d(partial: &Tensor<f32>, total_cols: usize, col_start: usize) -> Tensor<f32> {
    let block_width = partial.shape().first().copied().unwrap_or(0);
    let partial_data = dense_vec(partial);
    let mut out = vec![0f32; total_cols];
    out[col_start..col_start + block_width].copy_from_slice(&partial_data);
    build_tensor(out, &[total_cols])
}

/// 2 次元 `matmul` の転置。shape 検査は forward（`Var::matmul` →
/// `matmul_out_shape`）が済ませた 2 次元前提であり、`transpose(0, 1)`
/// は構造的に失敗しえない。それでも本番経路で `unwrap()`/`expect()`
/// を使わない方針（`.claude/rules/coding-rust.md`）のため、失敗時は
/// `debug_assert!` で契約違反を検知しつつ入力をそのまま返す
/// （到達すれば forward 側の shape 検査ロジックにバグがある）。
fn transpose2d(tensor: &Tensor<f32>) -> Tensor<f32> {
    match tensor.transpose(0, 1) {
        Ok(t) => t,
        Err(_) => {
            debug_assert!(
                false,
                "transpose2d: matmul VJP の rank-2 前提が崩れた（forward 側の契約違反）"
            );
            tensor.clone()
        }
    }
}

/// `MatMul(A, B)` の VJP: `dA = g @ Bᵀ`、`dB = Aᵀ @ g`
/// （`A: [m,k]`・`B: [k,n]`・`g: [m,n]`）。イシュー #1211: forward と
/// 同じ `BackendOps::gemm_fp32_strict`（CPU は BLIS 並列 GEMM・CUDA/Metal
/// はデバイス GEMM。`gemm` と同じカーネルを使うが、CUDA の TF32 opt-in
/// フラグ〈`set_cuda_tf32_gemm_enabled`〉には追従せず常に FP32 厳密で
/// 計算する。backward は `docs/cuda-tf32-optin-api-decision.md`・
/// `backend-cuda::precision` モジュール冒頭コメントの契約でスコープ外の
/// まま FP32 のため区別する。codex-review 指摘・PR #1223）を経由する
/// ため、backward の支配的コスト（`docs/perf/
/// train-step-phase-breakdown.md` §11・§15）がバックエンド既定の並列・
/// デバイス実装の恩恵を受ける。FMA 契約は各バックエンドの `gemm` 既定
/// 契約に従う（`coding-rust.md` の FMA 契約統一方針。forward の
/// `Var::matmul` と同一カーネルを通るため経路間で分岐しない）。
///
/// `transpose2d`（下記。`Tensor::transpose` の zero-copy stride view）
/// で作った転置オペランドはそのまま `ops.gemm_fp32_strict` へ渡す。
/// CPU（イシュー #1213）は片側転置（NT/TN）かつ dense な転置格納
/// （`strides() == [1, shape()[0]]`）と判定できる場合に限り
/// `contiguous()` の再パックコピーを経由せず BLIS packing 側で直接
/// 吸収する専用入口（CPU 実装クレートの `gemm_blis_parallel_nt`／
/// `gemm_blis_parallel_tn`）へ分岐する。両方転置（TT）・一般 stride
/// （`narrow` 後の転置等）は CPU でも従来どおり `contiguous()` を経由
/// する（一般 stride 化はスコープ外。`docs/matmul-vjp-zero-copy-
/// decision.md` §3.2・§4.2・§4.3 追補）。CUDA（イシュー #1214）も同型の
/// NT/TN 専用入口（GPU 側 smem 転置カーネル → 既存 NN GEMM カーネル。
/// `CudaGemm::run_tiled_f32_nt`／`run_tiled_f32_tn`）を持つ。Metal
/// （イシュー #1215）も同型の NT/TN 判定で classic strided カーネル
/// 入口（`gemm::MetalGemm::dispatch_strided_bias_act_prepared`）へ分岐
/// する専用経路を持つが、既存 NN 経路（`dispatch_auto`）とは別カーネル
/// のため数値契約は bit 一致ではなく REQ-2 統一複合判定
/// （`docs/matmul-vjp-zero-copy-decision.md` §4.4）。
///
/// エラーは fail-closed で `AutodiffError::Backend` として伝播し、
/// `eval::matmul` への暗黙フォールバックは設けない（forward と backward
/// で数値経路が分岐する判定迂回を作らないため。`.claude/rules/
/// security.md` A08）。
fn matmul_vjp(
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
    let b_t = transpose2d(b);
    let a_t = transpose2d(a);
    let da = ops
        .gemm_fp32_strict(g, &b_t)
        .map_err(AutodiffError::Backend)?;
    let db = ops
        .gemm_fp32_strict(&a_t, g)
        .map_err(AutodiffError::Backend)?;
    Ok((da, db))
}

/// ブロードキャストの逆演算。`add`/`mul` の VJP が返す勾配は forward
/// 出力の shape（ブロードキャスト後）を持つため、元の入力 shape
/// （`target_shape`）へ縮約する必要がある。NumPy 風ブロードキャスト
/// は「先頭に新設された軸」と「入力側が size 1 だった軸」を複製する
/// ため、その逆演算は同じ軸集合を合計で潰せばよい（PoC-v2-2 の
/// `sum_rows_data` を任意 shape・任意軸へ一般化した実装）。
fn reduce_to_shape(g: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    let g_shape = g.shape().to_vec();
    if g_shape == target_shape {
        return g.clone();
    }
    debug_assert!(
        g_shape.len() >= target_shape.len(),
        "reduce_to_shape: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = g_shape.len() - target_shape.len();
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let mut data = dense_vec(g);
    let mut cur_shape = g_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let outer: usize = cur_shape[..axis].iter().product();
            let axis_len = cur_shape[axis];
            let inner: usize = cur_shape[axis + 1..].iter().product();
            let mut reduced = vec![0f32; outer * inner];
            for o in 0..outer {
                for a in 0..axis_len {
                    for i in 0..inner {
                        let src = (o * axis_len + a) * inner + i;
                        reduced[o * inner + i] += data[src];
                    }
                }
            }
            data = reduced;
            cur_shape[axis] = 1;
        }
    }
    build_tensor(data, target_shape)
}

/// bias 勾配専用の縮約ディスパッチ（イシュー #1566・PR #1659 codex-review
/// P1 是正・2026-09-12 ユーザー承認 A の横展開）。
///
/// `Op::LinearResident` は resident 経由の成功時（`outcome.bias_filled`）
/// `eval::reduce_bias_grad_rows`（`f64` アキュムレータ。ホスト経路）・
/// GPU カーネル `gemm_bias_grad_reduce_f32`（binary64 加算の 64bit 整数
/// エミュレーション。ホストと bit 一致）のいずれか
/// で bias を計算する（`docs/backend-metal-command-batching-design.md`
/// §10.8）。`outcome.bias_filled == false`（CPU／CUDA 等 resident 非対応
/// バックエンド、または weight tying で bias 自身が非 resident 扱いに
/// なった場合のフォールバック）だけが `g` を単純な `f32` 逐次和
/// （`reduce_to_shape`）で縮約していたため、**同一の `Op::LinearResident`
/// が実行環境（バックエンド／resident 対応可否）によって異なる縮約方式
/// を使う**という不整合があった（`[1e8, 1.0, -1e8]` のような相殺入力で
/// Metal resident 経路と CPU／CUDA フォールバックが食い違う。codex-review
/// 指摘）。
///
/// `Op::LinearAct`（`Op::LinearResident` の resident 化を伴わない同型の
/// bias 縮約）にも同じ理由で適用し、両 Op 間の縮約方式を揃える
/// （fresh〈`LinearAct`〉と reuse〈`LinearResident` フォールバック〉が
/// 同じ形状パターンで異なる数値を返さないようにする）。
///
/// **`Op::Add` への横展開（2026-09-12 ユーザー承認・PR #1659→#1665→
/// #1666 取り込み後の追加是正）**: `nn::Linear` の既定 forward 経路
/// （`LinearVars::forward`。`matmul → add` の非融合合成。`nn/linear.rs`
/// doc「bias 加算は `Var::add` の broadcast に委ねる」参照）は `Op::Add`
/// の VJP を経由するため、`Op::Add(a, b)` 側でも本関数へ委譲する
/// （`da`／`db` 双方。`Op::Add` は可換なので bias がどちらの引数に来ても
/// 対称に扱える）。これにより、同じ `Linear` 層が `LinearVars::forward`
/// （fresh・非融合）と `forward_with_activation`（`Op::LinearAct`。
/// epilogue 融合）・`DeviceParamStore::linear_forward_with_activation`
/// （`Op::LinearResident`。reuse）のいずれで forward されても bias
/// 勾配の数値方式が揃う。`Op::Add` は bias 以外の一般的な broadcast
/// （bias パターンに一致しない任意 shape の加算）にも使われる汎用 Op
/// のため、下記の shape 構造判定を満たさない呼び出しは本関数の内部で
/// 既存の `reduce_to_shape` へそのまま委譲され挙動を変えない（bias
/// パターンに限定した横展開であり、汎用 `Op::Add`・`reduce_to_shape`
/// 本体自体は不変）。
///
/// 適用条件は `g` が rank-2 `[m, n]` かつ `target_shape` が「軸 0
/// （行／batch 軸）方向の縮約」を表す形状（末尾次元が `n` と一致し、
/// それより前の全次元が `1`。典型例: `[n]`・`[1, n]`。`nn::Linear` の
/// bias `[out_features]` を含む）の場合に限る。`eval::reduce_bias_grad_
/// rows` は「rank-2 入力を行 `0..m` で縮約し列ごとの和を返す」という
/// 固定の契約（`gemm_bias_grad_reduce_f32` の `MatrixLayout` と同型）
/// のため、**軸 1（列）方向を縮約する broadcast 形状（例: `g: [2, 2]`
/// に対する `target_shape: [2, 1]`。各行の bias が列方向へ複製される
/// パターン）には適用できない**——列ごとの和という異なる縮約軸の結果を
/// 返してしまい、値そのものが誤りになる（PR #1659 codex-review P2
/// 是正。回帰テスト `reduce_bias_grad_does_not_misapply_row_reduction_
/// to_column_broadcast_bias` 参照）。この判定は総要素数の一致だけでは
/// 検出できない（`[2, 1]` も総要素数 `2` で `g` の列数 `2` と一致して
/// しまうため、旧実装は shape 構造を見ずに誤って f64 経路へ分岐して
/// いた）。`nn::Linear`〈`from_parameters` が bias を `[out_features]`
/// 厳密一致にしか構築しない〉経由では軸 1 縮約の broadcast bias は
/// 到達しない（`pub(crate) fn linear_act` を直接呼ぶ経路・`Op::Add`
/// 経由で `Var::add` に非 bias 形状の broadcast を直接渡す経路限定。
/// `var.rs` doc「`linear_act` は `[n]` と厳密一致しない broadcast
/// 可能な bias も受理する」参照）。`Op::Add` への横展開後もこの shape
/// 構造判定自体は不変であり、`[2, 1]` のような軸 1 縮約は `Op::Add`
/// 経由でも同様に誤適用を回避する。適用条件を満たさない場合は既存の
/// `reduce_to_shape`（`f32` 逐次和・任意 rank・任意軸対応）のまま
/// 維持し挙動を変えない（安全側）。
fn reduce_bias_grad(g: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    let g_shape = g.shape();
    let is_row_axis_reduction = g_shape.len() == 2
        && target_shape.last() == Some(&g_shape[1])
        && target_shape[..target_shape.len().saturating_sub(1)]
            .iter()
            .all(|&d| d == 1);
    if is_row_axis_reduction {
        let data = eval::reduce_bias_grad_rows(g);
        return build_tensor(data, target_shape);
    }
    reduce_to_shape(g, target_shape)
}

/// 同 shape の 2 テンソルに対する要素ごとの条件付き選択
/// （`g` をそのまま通すか 0 にするかを `mask_src` の値で決める）。
/// `Relu` の VJP（`g ⊙ 1[x > 0]`）専用の最小実装。
///
/// **イシュー #1577**: reuse 学習（`DeviceParamStore` 経由・
/// `Op::LinearResident`）の backward では、下流層の VJP が返す
/// `d_input = transpose2d(&tmp)`（本ファイル内 `Op::LinearResident`
/// 分岐）が stride `[1, m]` のゼロコピー転置 view であり、単一寄与
/// なら `backward.rs::accumulate` がコピーせずそのまま上流層の
/// `upstream` になる。旧実装は `dense_vec`（`eval::dense_vec` →
/// `Tensor::contiguous()`）が非連続入力を要素ごと `get(&index)`
/// （rank 検査・軸ごとの範囲検査を伴う）で走査するため、連続入力比で
/// 大幅に劣化していた（実測は `docs/perf/
/// lowlayer-diagnosis-2026-09-12.md` §4・`docs/perf/
/// train-reuse-relu-mask-stride.md`）。
///
/// 本実装は `g`・`mask_src`（`Op::LinearAct`／`Op::LinearResident` では
/// `out_value` が `materialize_fallible` 経由で view になりうる）の
/// 双方を独立に [`Tensor::as_view_slice`]（全 strides が非負な限り
/// `contiguous()` を経由せず storage を借用で読む。`as_slice` が成功
/// するケース〈真に contiguous〉も同じ formula で正しく読める＝分岐を
/// 増やさず包含する）で読み、`get()` の rank・範囲検査コストを避けて
/// 出力を 1 パスで構築する。`as_view_slice` が `None`（負 stride 等。
/// 現行公開 API の `transpose`/`narrow`/`broadcast_to` はいずれも負
/// stride を生成しないため到達しないが将来拡張への fail-safe）の
/// 場合や shape 不一致・オフセット計算のオーバーフロー等、想定外の
/// 状態を検知した場合は、静かに 0 で埋めたり判定を迂回したりせず、
/// `dense_vec` を使う既存の走査へ**経路全体を丸ごと**フォールバック
/// する（数値的に同一のコピー経路であり、`.claude/rules/security.md`
/// A08 が禁じる判定迂回ではない）。出力は要素ごとの選択（算術なし）
/// のため走査順に依存せず bit 同一（run-to-run・変更前後とも）を
/// 維持する。
fn elementwise_mul_mask(
    g: &Tensor<f32>,
    mask_src: &Tensor<f32>,
    keep: impl Fn(f32) -> bool,
) -> Tensor<f32> {
    let shape = g.shape().to_vec();
    if let Some(out) = try_elementwise_mul_mask_strided(g, mask_src, &shape, &keep) {
        return build_tensor(out, &shape);
    }
    let g_data = dense_vec(g);
    let mask_data = dense_vec(mask_src);
    let out: Vec<f32> = g_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
        .collect();
    build_tensor(out, &shape)
}

/// [`elementwise_mul_mask`] の stride 対応主経路。読み出しに失敗しうる
/// 要因（shape 不一致・オフセット計算オーバーフロー）を検出した場合は
/// `None` を返し、呼び出し元が `dense_vec` 経路へ丸ごとフォールバック
/// する（部分的に誤った値を返さない）。
fn try_elementwise_mul_mask_strided(
    g: &Tensor<f32>,
    mask_src: &Tensor<f32>,
    shape: &[usize],
    keep: &impl Fn(f32) -> bool,
) -> Option<Vec<f32>> {
    if mask_src.shape() != shape {
        // 既存実装（`dense_vec` の zip）は shape 不一致時に短い方へ
        // 暗黙に切り詰めていた。多次元 index による読み出しはこの
        // 前提を要求するため、不一致時は無条件でフォールバックし
        // 既存の暗黙切り詰め挙動をそのまま保つ。
        return None;
    }
    let numel: usize = shape.iter().product();
    let g_op = MaskReadOperand::classify(g);
    let mask_op = MaskReadOperand::classify(mask_src);

    // fresh 経路（`Op::Relu`／`Op::LinearAct` の `upstream` が
    // `matmul_vjp` の連続な GEMM 出力である通常ケース）を含む、
    // 両オペランドとも連続な最頻ケースの高速経路。`read(idx, flat)`
    // 経由の enum ディスパッチ・オフセット計算を経由せず、借用スライス
    // 2 本の `zip`／`map`／`collect` に落とすことでコンパイラの自動
    // ベクトル化を妨げない（`MaskReadOperand::read` 経由の一般化した
    // 経路は非連続 view 専用に限定する）。
    if let (MaskReadOperand::Contig(g_s), MaskReadOperand::Contig(m_s)) = (&g_op, &mask_op) {
        return Some(
            g_s.iter()
                .zip(m_s.iter())
                .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
                .collect(),
        );
    }

    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        // reuse backward の実際のホットパス（下流層 d_input が
        // `transpose2d` のゼロコピー view・上流層 `out_value` は連続な
        // forward 記録値、またはその逆）を狙い撃ちした専用経路。片方が
        // `Contig`（行優先の連続スライスを直接インデックス）・片方が
        // `View`（行ごとの基準オフセット `i * s0` を 1 回だけ計算し、
        // 列方向は `+ j * s1` の加算のみ）に限定して読み出すことで、
        // 一般化した `MaskReadOperand::read` 経由（列ごとに strides を
        // ゼロから内積するオーバーヘッド）より高速化する（実測は
        // `docs/perf/train-reuse-relu-mask-stride.md` §5）。
        if let (
            MaskReadOperand::Contig(g_s),
            MaskReadOperand::View {
                span: m_span,
                strides: m_strides,
            },
        ) = (&g_op, &mask_op)
        {
            let (ms0, ms1) = (m_strides[0], m_strides[1]);
            let mut out = Vec::with_capacity(numel);
            for i in 0..rows {
                let row_start = i * cols;
                let g_row = g_s.get(row_start..row_start + cols)?;
                let m_base = i * ms0;
                for (j, &gv) in g_row.iter().enumerate() {
                    let mv = *m_span.get(m_base + j * ms1)?;
                    out.push(if keep(mv) { gv } else { 0.0 });
                }
            }
            return Some(out);
        }
        if let (
            MaskReadOperand::View {
                span: g_span,
                strides: g_strides,
            },
            MaskReadOperand::Contig(m_s),
        ) = (&g_op, &mask_op)
        {
            let (gs0, gs1) = (g_strides[0], g_strides[1]);
            let mut out = Vec::with_capacity(numel);
            for i in 0..rows {
                let row_start = i * cols;
                let m_row = m_s.get(row_start..row_start + cols)?;
                let g_base = i * gs0;
                for (j, &mv) in m_row.iter().enumerate() {
                    let gv = *g_span.get(g_base + j * gs1)?;
                    out.push(if keep(mv) { gv } else { 0.0 });
                }
            }
            return Some(out);
        }
    }

    let mut out = Vec::with_capacity(numel);

    if shape.len() == 2 {
        // 上記 2 分岐（片方 `Contig`・片方 `View`）に該当しない rank-2
        // （両方 `View`／`Owned` を含むケース）向けの一般経路。固定長
        // 2 要素の index 配列のみでスタック上で完結し、一般 N-d 経路の
        // `Vec<usize>` 繰り上げより軽い。
        let (rows, cols) = (shape[0], shape[1]);
        for i in 0..rows {
            for j in 0..cols {
                let idx = [i, j];
                let flat = i * cols + j;
                let gv = g_op.read(&idx, flat)?;
                let mv = mask_op.read(&idx, flat)?;
                out.push(if keep(mv) { gv } else { 0.0 });
            }
        }
        return Some(out);
    }

    // 一般 N-d: index ベクタを行優先（最終軸が最速）で繰り上げる。
    let mut idx = vec![0usize; shape.len()];
    for flat in 0..numel {
        let gv = g_op.read(&idx, flat)?;
        let mv = mask_op.read(&idx, flat)?;
        out.push(if keep(mv) { gv } else { 0.0 });
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    Some(out)
}

/// [`elementwise_mul_mask`] が読む 1 オペランド分の抽象。
///
/// `as_slice()`（真に contiguous）が成功すれば `Contig` として最優先で
/// 扱う（`try_elementwise_mul_mask_strided` の全 contig 高速経路・
/// rank-2／一般 N-d 経路いずれからも `flat` 添字で直接読める）。次に
/// `as_view_slice()`（`transpose`/`narrow`/`broadcast_to` の非負
/// stride view を含む）が成功すれば `View` として **`usize` へ変換
/// 済みの** strides 付きで借用を保持する。いずれも失敗した場合のみ
/// `dense_vec`（コピー）を保持する `Owned` へフォールバックする。
///
/// `View` のオフセット計算（[`Self::read`]）は要素ごとに `checked_mul`/
/// `checked_add`/`isize`↔`usize` 変換を経由せず、プレーンな `usize`
/// 乗算・加算のみを行う。安全性の根拠: `as_view_slice()` が `Some` を
/// 返した時点で全 strides が非負であることが確定しており（`classify`
/// で 1 回だけ `usize` へ変換）、かつ同メソッドは
/// `span = 1 + Σ (shape_i − 1)·stride_i` を `checked_add`/`checked_mul`
/// で検証済みである。したがって shape 範囲内の任意の `idx` に対し
/// `Σ idx_i·stride_i < span == span.len()` が保証され、本メソッド内で
/// 改めて overflow を心配する必要はない（対象テンソルの要素数は
/// 学習用途の実用範囲で `usize::MAX` に遠く及ばない）。境界外
/// アクセスの検出自体は最終的な `span.get(off)` の 1 回の `Option`
/// 判定に集約し、そこで `None` になった場合のみ
/// `try_elementwise_mul_mask_strided` 全体が `dense_vec` 経路へ
/// フォールバックする（`.claude/rules/coding-rust.md` の `unwrap`／
/// `expect` 非使用方針を保ちつつ、要素ごとの checked 演算チェーンに
/// よる速度低下〈初版実装で実測。`docs/perf/
/// train-reuse-relu-mask-stride.md` §5 参照〉を避ける）。
enum MaskReadOperand<'a> {
    Contig(&'a [f32]),
    View {
        span: &'a [f32],
        strides: Vec<usize>,
    },
    Owned(Vec<f32>),
}

impl<'a> MaskReadOperand<'a> {
    fn classify(t: &'a Tensor<f32>) -> Self {
        if let Some(s) = t.as_slice() {
            return MaskReadOperand::Contig(s);
        }
        if let Some(span) = t.as_view_slice() {
            // `as_view_slice()` が `Some` を返した時点で strides は
            // 全て非負が保証されるため、この `usize::try_from` は
            // 通常失敗しない。万一の不整合（`Tensor` 側の契約違反）
            // に備え、フォールバック先である `Owned` へ迂回する。
            let strides: Option<Vec<usize>> = t
                .strides()
                .iter()
                .map(|&s| usize::try_from(s).ok())
                .collect();
            if let Some(strides) = strides {
                return MaskReadOperand::View { span, strides };
            }
        }
        MaskReadOperand::Owned(dense_vec(t))
    }

    /// `Contig`／`Owned` の場合は `flat`（行優先の平坦 index。
    /// `as_slice()`／`dense_vec` の走査順と一致）で、`View` の場合は
    /// `idx`（strides との内積でオフセットを計算。プレーン `usize`
    /// 演算のみ・型定義側 doc 参照）で読む。境界外アクセスを検知
    /// した場合（`View` の `span.get` が `None` を返す場合。通常到達
    /// しない防御的経路）は `None` を返し、呼び出し元の
    /// `try_elementwise_mul_mask_strided` 全体を `dense_vec` 経路へ
    /// フォールバックさせる。両オペランドとも `Contig` の最頻ケースは
    /// この汎用経路を経由せず、呼び出し元の専用高速経路で処理する
    /// （enum ディスパッチのオーバーヘッドを避けるため）。
    #[inline]
    fn read(&self, idx: &[usize], flat: usize) -> Option<f32> {
        match self {
            MaskReadOperand::Contig(s) => s.get(flat).copied(),
            MaskReadOperand::View { span, strides } => {
                let mut off = 0usize;
                for (&i, &s) in idx.iter().zip(strides.iter()) {
                    off += i * s;
                }
                span.get(off).copied()
            }
            MaskReadOperand::Owned(v) => v.get(flat).copied(),
        }
    }
}

/// `Tanh` の VJP 係数 `1 - tanh(x)^2` を forward 記録値 `out_value`
/// （= `tanh(x)`）から計算する（再計算を避ける）。
fn tanh_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| 1.0 - v * v).collect();
    build_tensor(out, &shape)
}

/// `Sigmoid` の VJP 係数 `sigmoid(x) * (1 - sigmoid(x))` を forward
/// 記録値 `out_value`（= `sigmoid(x)`）から計算する（TASK-9.1b・#92。
/// `tanh_grad_factor` と同型の out_value 再利用パターン）。
fn sigmoid_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| v * (1.0 - v)).collect();
    build_tensor(out, &shape)
}

/// `Op::Softmax` の VJP 本体: `dx = y ⊙ (g − Σ_dim(g ⊙ y))`。
/// `eval::softmax_along`／`log_softmax_along` と同じ「外側（outer）×
/// 走査軸（axis_len）× 内側（inner）」の 3 段走査（`dim` は forward
/// 側で範囲検査済みの前提）。`Σ_dim(g ⊙ y)` の要素積は `f32` で確定
/// してから `f64` へ昇格して蓄積し（`.claude/rules/coding-rust.md`
/// 「勾配の長軸縮約の要素積は f32 で確定してから f64 へ昇格」）、
/// `g` からの減算・`y` との最終乗算も（`log_softmax_vjp_along` と同じ
/// 理由で）`f64` のまま保持し、最終書き出しで 1 回だけ `f32` へ
/// downcast する（縮約値を先に `f32` へ戻すと、有限の `f32` 入力でも
/// 減算・乗算の結果が overflow しうるため）。
fn softmax_vjp_along(out_value: &Tensor<f32>, upstream: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    // `log_softmax_vjp_along` 直下と同じ早期 return（部分積オーバー
    // フロー回避。`eval::softmax_along` 冒頭のコメント参照）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let y = dense_vec(out_value);
    let g = dense_vec(upstream);
    let mut out = vec![0f32; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut dot_acc: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let term = g[idx] * y[idx];
                dot_acc += term as f64;
            }
            // `dot_acc`（f64）を乗算前に `f32` へ downcast すると、
            // `y * (g - dot)` が有限の `f32` 入力でも overflow しうる
            // （例: `y=[0.25,0.75]`・上流勾配 `g=[3e38,-3e38]` で正しい
            // 入力勾配 `[~1.125e38, ...]` が `[inf, ...]` になる）。
            // `log_softmax_vjp_along` と同じ f64 アキュムレータ契約
            // （`.claude/rules/coding-rust.md`）に従い、`g` からの減算・
            // `y` との最終乗算まで f64 で保持し、最終書き出しでのみ
            // `f32` へ downcast する。
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let d = (y[idx] as f64) * (g[idx] as f64 - dot_acc);
                out[idx] = d as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `Op::LogSoftmax` の VJP 本体: `dx = g − exp(y) ⊙ Σ_dim(g)`。
/// `softmax_vjp_along` と同じ 3 段走査・f64 縮約方針（本関数の縮約は
/// 要素積ではなく単純和のため、`g` の各要素をそのまま `f64` へ昇格して
/// 蓄積する）。`Σ_dim(g)` との乗算（`exp(y) ⊙ Σ_dim(g)`）・`g` からの
/// 減算も f64 のまま行い、最終書き出しで 1 回だけ `f32` へ downcast
/// する（縮約値を先に `f32` へ戻すと、有限の `f32` 入力でも乗算結果が
/// overflow しうるため）。
fn log_softmax_vjp_along(
    out_value: &Tensor<f32>,
    upstream: &Tensor<f32>,
    axis: usize,
) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    // `softmax_vjp_along` 直上と同じ早期 return（部分積オーバーフロー
    // 回避。`eval::softmax_along` 冒頭のコメント参照）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let y = dense_vec(out_value);
    let g = dense_vec(upstream);
    let mut out = vec![0f32; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut sum_acc: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                sum_acc += g[idx] as f64;
            }
            // `sum_acc`（f64）を乗算前に `f32` へ downcast すると、
            // `exp(y) * sum_g` が有限の `f32` 入力でも overflow しうる
            // （例: `y=[0,0]`・上流勾配 `g=[2e38,2e38]` で正しい入力勾配
            // `[0,0]` が `[-inf,-inf]` になる）。`.claude/rules/
            // coding-rust.md` の f64 アキュムレータ契約に従い、
            // `exp(y)` との乗算・`g` からの減算まで f64 で保持し、
            // 最終書き出しでのみ `f32` へ downcast する。
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let d = g[idx] as f64 - (y[idx].exp() as f64) * sum_acc;
                out[idx] = d as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `Sum` の VJP: 出力側勾配 `g` を入力 shape へブロードキャストして
/// 複製する（`sum` の逆演算は複製、`reduce_out_shape` が縮約軸を
/// 除去済み〈keepdim なし〉のため、`dim: Some(axis)` はいったん
/// size-1 軸を挿入してから `broadcast_to` する）。
fn unreduce_broadcast(g: &Tensor<f32>, input_shape: &[usize], dim: Option<usize>) -> Tensor<f32> {
    match dim {
        None => {
            let value = dense_vec(g).first().copied().unwrap_or(0.0);
            match Tensor::full(input_shape, value) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: dim=None の full() 構築が失敗した（契約違反）"
                    );
                    g.clone()
                }
            }
        }
        Some(axis) => {
            let mut inserted_shape = g.shape().to_vec();
            inserted_shape.insert(axis, 1);
            let reshaped = match g.contiguous().reshape(&inserted_shape) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: reduce_out_shape 逆算の reshape が失敗した（契約違反）"
                    );
                    return g.clone();
                }
            };
            match reshaped.broadcast_to(input_shape) {
                Ok(t) => t.contiguous(),
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: 挿入軸からの broadcast_to が失敗した（契約違反）"
                    );
                    reshaped
                }
            }
        }
    }
}

/// `Max` の VJP: 出力側勾配 `g` を、縮約軸に沿った最大値の位置のみへ
/// 伝播する。**同値タイは「最初に現れる最大要素 1 箇所のみ」へ伝播
/// する**（PyTorch `amax` の均等分配とは異なる、決定的な選択。
/// PoC-v2-2 のビット一致決定性方針・`train_repro` と整合させるための
/// 設計判断）。`out_value` は forward 記録済みの縮約後最大値で、走査中
/// に現れる要素と exact 一致するかで argmax 位置を判定する（同一デー
/// タ・同一 reduction 経路のため bit 一致する）。
///
/// Issue #224（先勝ち挙動の再確認。compat 層〈REQ-9〉実装時に要再確認
/// としていた事項）の結論: **本挙動を維持する（変更なし）**。
/// compat 層（TASK-9.2a・#95 で実装。TASK-9.4・#411 で `fandhe_ai::compat`
/// へ移設済み）の公開面は `array()`／`Sequential`（Linear・ReLU・
/// Sigmoid・Tanh）に限定され（`docs/compat-api-scope.md` §1〜2）、
/// `max`/`amax` 相当 API が存在しないため PyTorch 互換を要求する利用者
/// 向け経路が現時点でない。均等分配へ変更すると勾配値そのものが変わり
/// 上記の決定性方針と衝突するため、先勝ちを維持する。再検討条件:
/// `fandhe_ai::compat`（REQ-9 追記・#52）の公開面に `amax` 相当の縮約 API を
/// 追加する段階になった場合にのみ PyTorch 互換の要否を改めて判断する
/// （`docs/compat-api-scope.md` にも記録）。
fn max_vjp(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_value: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    let in_shape = input.shape().to_vec();
    let in_data = dense_vec(input);
    let g_data = dense_vec(g);
    let out_data = dense_vec(out_value);
    let mut grad = vec![0f32; in_data.len()];
    match dim {
        None => {
            if let (Some(target), Some(gv)) = (out_data.first(), g_data.first())
                && let Some(idx) = in_data.iter().position(|&v| v == *target)
            {
                grad[idx] = *gv;
            }
        }
        Some(axis) => {
            let outer: usize = in_shape[..axis].iter().product();
            let axis_len = in_shape[axis];
            let inner: usize = in_shape[axis + 1..].iter().product();
            for o in 0..outer {
                for i in 0..inner {
                    let out_idx = o * inner + i;
                    // `out_data`/`g_data` の要素数は `reduce_out_shape`
                    // の契約上 `outer * inner` と一致するはずだが、
                    // 本ファイルの他ヘルパー（`transpose2d`／
                    // `unreduce_broadcast` 等）と同様、契約違反時に
                    // release ビルドで境界外アクセス panic させず
                    // `debug_assert!` で検知しつつ安全側（当該要素の
                    // 勾配は 0 のまま）へフォールバックする
                    // （coding-rust.md「本番経路で unwrap/expect を
                    // 使わない」方針の趣旨に揃える）。
                    let (Some(&target), Some(&g_val)) =
                        (out_data.get(out_idx), g_data.get(out_idx))
                    else {
                        debug_assert!(
                            false,
                            "max_vjp: out_value/g の要素数が reduce_out_shape の想定と不一致（契約違反）"
                        );
                        continue;
                    };
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        match in_data.get(src) {
                            Some(&v) if v == target => {
                                grad[src] = g_val;
                                break;
                            }
                            Some(_) => {}
                            None => {
                                debug_assert!(
                                    false,
                                    "max_vjp: input の要素数が in_shape と不一致（契約違反）"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    build_tensor(grad, &in_shape)
}

/// `MseLoss{pred, target, reduction}` の VJP: `dPred = g · 2(pred −
/// target) / n`（mean）／`g · 2(pred − target)`（sum）、
/// `dTarget = −dPred`（`g` はスカラー上流勾配）。#190 で sum 縮約を
/// 追加（`reduction` 分岐は forward の `eval::mse_loss` と対称）。
/// `n == 0` は mean・sum ともゼロ除算を避け zeros を返す。PoC-v2-2 は
/// `target` 側の勾配計算をスキップしていたが、本実装は数学的に完全な
/// VJP を返す（`target` 側を使うか捨てるかは #18 の勾配蓄積側の責務で
/// あり、ここでは両方提供する）。
fn mse_loss_vjp(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    g: &Tensor<f32>,
    reduction: Reduction,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = pred.shape().to_vec();
    let n = pred.numel();
    if n == 0 {
        let zeros = build_tensor(vec![0f32; 0], &shape);
        return (zeros.clone(), zeros);
    }
    let g_value = dense_vec(g).first().copied().unwrap_or(0.0);
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let scale = mse_loss_scale(g_value, n, reduction);
    let dpred_data: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| scale * (p - t))
        .collect();
    let dtarget_data: Vec<f32> = dpred_data.iter().map(|&v| -v).collect();
    let dpred = build_tensor(dpred_data, &shape);
    let dtarget = build_tensor(dtarget_data, &shape);
    (dpred, dtarget)
}

/// `mse_loss_vjp`（ホスト参照実装）と融合カーネル経路（`vjp()` の
/// `Op::MseLoss` 分岐。イシュー #1045）の双方が使う `scale` 算出の
/// 共有ロジック: `dPred = scale·(pred−target)`（`Mean` は `g·2/n`、
/// `Sum` は `g·2`）。`BackendOps::mse_loss_backward` の呼び出し元が
/// このスケールを事前計算して渡す契約（`backend_ops.rs` doc 参照）
/// であり、フォールバック（`mse_loss_vjp`）と融合カーネル経路とで
/// 同一の数式を 2 か所に別実装しないための切り出し。
fn mse_loss_scale(g_value: f32, n: usize, reduction: Reduction) -> f32 {
    match reduction {
        Reduction::Mean => g_value * 2.0 / n as f32,
        Reduction::Sum => g_value * 2.0,
    }
}

/// `CrossEntropyLoss(logits, targets)` の VJP:
/// `d loss / d logits[..., c, ...] = (softmax(logits)[..., c, ...] − 1{c == t}) × g`
/// （`g` はサンプルごとのスカラー係数。`Mean` は `g = upstream / N`、
/// `Sum` は `g = upstream`。`N` はサンプル数 `= targets.numel()`）。
/// `eval::softmax_along` を再利用し、forward（`eval::cross_entropy_loss`
/// の log-sum-exp）と数式の実体を分離しない（`grad.rs` 冒頭 doc）。
/// `targets` は非追跡のため戻り値は `logits` 側の勾配のみ（呼び出し元
/// `vjp()` の `CrossEntropyLoss` 分岐参照）。
fn cross_entropy_loss_vjp(
    logits: &Tensor<f32>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
    upstream: &Tensor<f32>,
) -> Tensor<f32> {
    let shape = logits.shape().to_vec();
    let outer: usize = shape[..class_dim].iter().product();
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..].iter().product();
    let n = outer * inner;

    let softmax = eval::softmax_along(logits, class_dim);
    let mut grad = dense_vec(&softmax);
    let target_data = eval::dense_vec_i32(targets);

    let g_value = dense_vec(upstream).first().copied().unwrap_or(0.0);
    let scale = match reduction {
        Reduction::Mean if n > 0 => g_value / n as f32,
        Reduction::Mean => 0.0,
        Reduction::Sum => g_value,
    };

    for o in 0..outer {
        for i in 0..inner {
            let t = target_data[o * inner + i];
            // forward（`Var::cross_entropy_loss`）が事前検査済みの前提
            // （`0 <= t < axis_len`）。範囲外は契約違反であり
            // `debug_assert!` で検知しつつ onehot 減算をスキップする
            // 安全側フォールバック（`eval::cross_entropy_loss` と同型の
            // 契約違反対応）。
            if t >= 0 && (t as usize) < axis_len {
                let idx = (o * axis_len + t as usize) * inner + i;
                grad[idx] -= 1.0;
            } else {
                debug_assert!(
                    false,
                    "cross_entropy_loss_vjp: target 添字が範囲外（契約違反）"
                );
            }
        }
    }
    let scaled: Vec<f32> = grad.iter().map(|&v| v * scale).collect();
    build_tensor(scaled, &shape)
}

#[cfg(test)]
mod tests {
    //! 受け入れ条件「各演算の解析勾配が数値微分と一致する」の直接検証。
    //!
    //! 各演算について、固定の重みテンソル `s`（forward 出力と同じ
    //! shape）によるスカラー射影 `L(x) = Σ (op(x) ⊙ s)` を定義し、
    //! 解析側（`vjp` 経由）と数値側（中央差分。f64 で集計）を突合する。
    //! `Tape`/`Var` を経由せず `eval.rs` の値計算を直接叩くため、
    //! `Tape::backward`（#18）が未実装でも検証できる。
    //!
    //! **判定基準**: 要素ごとに「相対誤差
    //! `|ad − num| / max(|ad|, |num|, τ)` が 1e-2 以下」または
    //! 「絶対誤差が 1e-3 以下」（`h = 1e-3`・`τ = 1e-4`）。PoC-v2-2 の
    //! 1e-4 は f64 前提（`torch.autograd.gradcheck` と同じ理由で f64
    //! 必須と PoC 自身が明記）だが、本実装は f32 のため中央差分の
    //! 丸め誤差床 `≈ ε_f32 · |L| / h ≈ 1e-4` を踏まえた本イシュー
    //! 新規の grad-check 専用閾値とする（バックエンド間数値一致判定
    //! 〈相対 1e-3 / 絶対 1e-5〉とは別系統）。
    //!
    //! **承認記録（#223・承認済み）**: `CLAUDE.md` Conventions・
    //! `.claude/rules/delegation-impl.md` の「テスト許容誤差の変更は
    //! ユーザー承認必須」規定に基づく本閾値（新規 grad-check 専用
    //! `REL_TOL`/`ABS_TOL`）の承認は完了している。
    //! - 承認者: ユーザー／承認日: 2026-08-09
    //! - 承認記録: <https://github.com/Fandhe-AI/fandhe-ai/issues/223#issuecomment-5230026874>
    //! - 判断材料: 全 grad-check テストの実測誤差マージン採取で、
    //!   最も僅差のケースでも絶対誤差側に約 3.4 倍の余裕
    //!   （実測 `diff ≈ 2.9e-4` に対し `ABS_TOL = 1e-3`）を確認
    //!
    //! 値（`REL_TOL`/`ABS_TOL`/`TAU`/`H`）の変更が必要になった場合は
    //! 改めてユーザー承認が必須であり、#223 系譜の新規 Issue で追跡する。
    //!
    //! **キンク・タイ回避**: ReLU は `|x| >= 10h` の固定入力のみ、
    //! Max は同値タイのない固定入力のみを使う（固定値のため再生成
    //! ガードは不要。PoC-v2-2 `grad_check.rs` と同方針）。

    use super::*;

    const H: f64 = 1e-3;
    const TAU: f32 = 1e-4;
    const REL_TOL: f32 = 1e-2;
    const ABS_TOL: f32 = 1e-3;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape)
            .expect("test fixture: shape とデータ長は事前に一致させている")
    }

    fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
        let a = dense_vec(analytic);
        let n = dense_vec(numeric);
        assert_eq!(
            a.len(),
            n.len(),
            "{label}: analytic/numeric の要素数が一致しない"
        );
        for (i, (&av, &nv)) in a.iter().zip(n.iter()).enumerate() {
            let diff = (av - nv).abs();
            let rel = diff / av.abs().max(nv.abs()).max(TAU);
            assert!(
                rel <= REL_TOL || diff <= ABS_TOL,
                "{label}[{i}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
            );
        }
    }

    /// `L(x) = Σ (forward(x) ⊙ s)` の f64 集計によるスカラー射影値。
    fn scalar_dot(a: &Tensor<f32>, s: &Tensor<f32>) -> f64 {
        dense_vec(a)
            .iter()
            .zip(dense_vec(s).iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    }

    /// 単一入力 `x` に対する `L` の中央差分勾配（要素ごと・f64 集計）。
    fn numeric_grad_unary(
        x: &Tensor<f32>,
        s: &Tensor<f32>,
        forward: impl Fn(&Tensor<f32>) -> Tensor<f32>,
    ) -> Tensor<f32> {
        let shape = x.shape().to_vec();
        let mut data = dense_vec(x);
        let mut grad = vec![0f32; data.len()];
        for i in 0..data.len() {
            let orig = data[i] as f64;
            data[i] = (orig + H) as f32;
            let lp = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = (orig - H) as f32;
            let lm = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = orig as f32;
            grad[i] = ((lp - lm) / (2.0 * H)) as f32;
        }
        build_tensor(grad, &shape)
    }

    // --- MatMul ---

    #[test]
    fn matmul_grad_matches_numeric() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let g = s.clone();
        let (da, db) = matmul_vjp(&test_ops(), &a, &b, &g).unwrap();

        let num_da = numeric_grad_unary(&a, &s, |x| eval::matmul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::matmul(&a, x));

        assert_grad_close("matmul dA", &da, &num_da);
        assert_grad_close("matmul dB", &db, &num_db);
    }

    /// イシュー #1046 受け入れ条件 (a) の機械検証: `matmul_vjp` は
    /// `transpose2d`（zero-copy view）で作った転置オペランドを
    /// `eval::matmul` へ渡すが、`eval::matmul_operand` が
    /// `layout::classify_2d` で分類して直接読み出すため、ホスト側
    /// 転置コピー（`eval::MATMUL_HOST_REPACK_COUNT`）が発生しない
    /// ことを確認する。
    #[test]
    fn matmul_vjp_does_not_repack_transposed_operands() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let g = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let before = eval::MATMUL_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = matmul_vjp(&test_ops(), &a, &b, &g).unwrap();
        let after = eval::MATMUL_HOST_REPACK_COUNT.with(|c| c.get());

        assert_eq!(
            before, after,
            "matmul_vjp: test_ops()（TestOps → eval::matmul 委譲。#1211 で \
             本番経路は BackendOps::gemm_fp32_strict 経由になったが、この \
             compat 経路の \
             ゼロコピー保証は変わらない）が転置オペランド（transpose2d の \
             zero-copy view）でホスト側転置コピーへフォールバックした \
             （MATMUL_HOST_REPACK_COUNT が増加した）"
        );
    }

    // --- Add（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn add_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(same) dA", &da, &num_da);
        assert_grad_close("add(same) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(bias) dA", &da, &num_da);
        assert_grad_close("add(bias) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(scalar) dA", &da, &num_da);
        assert_grad_close("add(scalar) dB", &db, &num_db);
    }

    // --- Mul（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn mul_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(same) dA", &da, &num_da);
        assert_grad_close("mul(same) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(bias) dA", &da, &num_da);
        assert_grad_close("mul(bias) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(scalar) dA", &da, &num_da);
        assert_grad_close("mul(scalar) dB", &db, &num_db);
    }

    // --- Relu（正負混在。|x| >= 10h でキンク回避） ---

    #[test]
    fn relu_grad_matches_numeric() {
        let a = t(&[2.0, -3.0, 0.5, -0.02, 1.5, -1.5], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.0], &[2, 3]);

        let g = s.clone();
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        let num_da = numeric_grad_unary(&a, &s, eval::relu);

        assert_grad_close("relu dA", &da, &num_da);
    }

    #[test]
    fn relu_subgradient_at_zero_is_zero() {
        // x = 0 における劣勾配は 0 とする（PoC-v2-2 準拠。中央差分は
        // キンクで数値的に不安定なため、ここは解析式の直接検証のみ）。
        let a = t(&[0.0], &[1]);
        let g = t(&[3.0], &[1]);
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        assert_eq!(dense_vec(&da), vec![0.0]);
    }

    // --- イシュー #1577: elementwise_mul_mask の stride 対応 ---
    //
    // 新実装（`try_elementwise_mul_mask_strided` を経由する
    // `elementwise_mul_mask`）と、旧実装をそのまま残した参照実装
    // （`dense_vec` を zip するだけの経路）の出力を `to_bits()` で
    // 完全一致比較する。数値的に同一の値を出す契約（bit 同一）を
    // 直接検証する。

    /// `dense_vec` 経由の参照実装（旧 `elementwise_mul_mask` そのもの）。
    /// 新実装との bit 同一性を突き合わせる基準として使う。
    fn elementwise_mul_mask_reference(
        g: &Tensor<f32>,
        mask_src: &Tensor<f32>,
        keep: impl Fn(f32) -> bool,
    ) -> Tensor<f32> {
        let shape = g.shape().to_vec();
        let g_data = dense_vec(g);
        let mask_data = dense_vec(mask_src);
        let out: Vec<f32> = g_data
            .iter()
            .zip(mask_data.iter())
            .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
            .collect();
        build_tensor(out, &shape)
    }

    fn assert_bits_eq(label: &str, actual: &Tensor<f32>, expected: &Tensor<f32>) {
        let a = dense_vec(actual);
        let e = dense_vec(expected);
        assert_eq!(a.len(), e.len(), "{label}: 要素数不一致");
        for (i, (&av, &ev)) in a.iter().zip(e.iter()).enumerate() {
            assert_eq!(
                av.to_bits(),
                ev.to_bits(),
                "{label}[{i}]: actual={av:?}（bits={:#x}） expected={ev:?}（bits={:#x}）",
                av.to_bits(),
                ev.to_bits()
            );
        }
    }

    #[test]
    fn mask_stride_transpose_view_matches_reference() {
        // reuse backward が生む `d_input = transpose2d(&tmp)` を再現
        // （`tmp: [k, m]` 連続 → 転置後 `[m, k]`・strides `[1, k]`）。
        let tmp = t(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]);
        let g = transpose2d(&tmp); // shape [3, 2]、strides [1, 3]
        let mask_src = t(&[1.0, -1.0, 0.0, 2.0, -2.0, 0.5], &[3, 2]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("transpose view (g 側)", &actual, &expected);
    }

    #[test]
    fn mask_stride_mask_src_side_non_contiguous() {
        // `mask_src` 側だけが非連続（`out_value` が view の場合の
        // 想定。`g` は連続）。
        let g = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let mask_tmp = t(&[1.0, -1.0, 0.0, 2.0, -2.0, 0.5], &[2, 3]);
        let mask_src = transpose2d(&mask_tmp); // shape [3, 2]

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("transpose view (mask_src 側)", &actual, &expected);
    }

    #[test]
    fn mask_stride_narrow_offset_view() {
        // `narrow` 後の view（offset != 0・かつ真に非連続）。列方向
        // （dim 1）の `narrow` は、行方向（dim 0）の `narrow` と異なり
        // 元の行幅（stride 4）が残ったまま shape が縮む（`[3,4]` の
        // 列 1..3 を切り出すと shape `[3,2]`・strides `[4,1]` となり、
        // 新 shape の標準行優先 stride `[2,1]` とは一致しない）ため
        // `as_slice()` が `None` を返す（`Tensor::is_contiguous` 契約）。
        // 行方向の `narrow` は新 shape でも標準行優先 stride のまま
        // 残り `as_slice()` が成功してしまう（`Contig` 分類）ため、
        // 本テストの意図（`View` 分類・rank-2 `Contig`×`View` 専用
        // 経路のオフセット付きケース）を検証するには列方向でなければ
        // ならない（advisor 指摘。行方向版は誤って `Contig`×`Contig`
        // 高速経路しか検証していなかった）。
        let base = t(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let g = base
            .narrow(1, 1, 2)
            .expect("narrow: 事前に範囲内であることを確認済み");
        assert!(
            g.as_slice().is_none(),
            "narrow(dim=1) は非連続 view のはず（本テストが検証したい前提。\
release ビルドの `cargo test --release` でも前提崩れを検知できるよう \
`debug_assert!` ではなく `assert!` を使う）"
        );
        let mask_src = t(&[-1.0, 1.0, 0.0, -2.0, 3.0, -3.0], &[3, 2]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("narrow(dim=1) view", &actual, &expected);
    }

    #[test]
    fn mask_stride_both_operands_transposed_rank2() {
        // `g`・`mask_src` の両方が非連続 view（`View`×`View`）の
        // rank-2 ケース。rank-2 専用経路のうち「片方 `Contig`・片方
        // `View`」の 2 分岐（advisor 指摘で追加）のどちらにも該当
        // しないため、`try_elementwise_mul_mask_strided` 内の
        // 一般化した `read(idx, flat)` 経由の rank-2 経路（`MaskReadOperand::
        // read` の `View` アーム）を確実に踏む。
        let tmp_g = t(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]);
        let g = transpose2d(&tmp_g); // shape [3, 2]、非連続 view
        let tmp_mask = t(&[1.0, -1.0, 0.0, 2.0, -2.0, 0.5], &[2, 3]);
        let mask_src = transpose2d(&tmp_mask); // shape [3, 2]、非連続 view
        assert!(
            g.as_slice().is_none() && mask_src.as_slice().is_none(),
            "両オペランドとも非連続 view のはず（本テストが検証したい前提。\
release ビルドでも検知できるよう `assert!` を使う）"
        );

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("View×View rank-2", &actual, &expected);
    }

    #[test]
    fn mask_stride_broadcast_zero_stride_view() {
        // `broadcast_to` が生む stride 0 の軸を含む view。
        let row = t(&[1.0, -1.0, 2.0], &[1, 3]);
        let g = row
            .broadcast_to(&[2, 3])
            .expect("broadcast_to: shape 互換性は事前に確認済み");
        let mask_src = t(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0], &[2, 3]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("broadcast (stride 0) view", &actual, &expected);
    }

    #[test]
    fn mask_stride_rank1_and_rank3() {
        // rank-1（transpose2d 適用対象外だが narrow で非連続を作る）。
        let base1 = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let g1 = base1
            .narrow(0, 1, 3)
            .expect("narrow: 事前に範囲内であることを確認済み");
        let mask1 = t(&[-1.0, 1.0, -1.0], &[3]);
        let actual1 = elementwise_mul_mask(&g1, &mask1, |v| v > 0.0);
        let expected1 = elementwise_mul_mask_reference(&g1, &mask1, |v| v > 0.0);
        assert_bits_eq("rank-1 narrow", &actual1, &expected1);

        // rank-3: 2x2x3 を transpose(0, 2) で非連続にする。
        let base3 = t(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 2, 3],
        );
        let g3 = base3
            .transpose(0, 2)
            .expect("transpose: rank-3 は 0,2 とも範囲内");
        let mask3 = t(
            &[
                1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 3.0, -3.0, 0.25,
            ],
            &[3, 2, 2],
        );
        let actual3 = elementwise_mul_mask(&g3, &mask3, |v| v > 0.0);
        let expected3 = elementwise_mul_mask_reference(&g3, &mask3, |v| v > 0.0);
        assert_bits_eq("rank-3 transpose", &actual3, &expected3);
    }

    #[test]
    fn mask_stride_empty_tensor() {
        let g = t(&[], &[0, 3]);
        let mask_src = t(&[], &[0, 3]);
        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        assert_eq!(dense_vec(&actual), Vec::<f32>::new());
    }

    #[test]
    fn mask_stride_nan_and_signed_zero_and_subnormal() {
        // NaN（マスク不成立で 0 を返す規約）・-0.0・subnormal を含む
        // 転置 view で bit 同一性を確認する。
        let tmp = t(
            &[f32::NAN, -0.0, f32::MIN_POSITIVE / 2.0, 1.0, -1.0, 0.0],
            &[2, 3],
        );
        let g = transpose2d(&tmp); // shape [3, 2]
        let mask_src = t(&[1.0, f32::NAN, -0.0, 1.0, 0.0, -1.0], &[3, 2]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("NaN / -0.0 / subnormal", &actual, &expected);
    }

    /// マイクロベンチ（`#[ignore]`。手動実行専用。
    /// `docs/perf/lowlayer-diagnosis-2026-09-12.md` §4 の
    /// `diag_elementwise_mask_bench` と同構成。64×256 の連続入力 と
    /// `[256,64]→transpose2d` の非連続転置 view を 1000 回反復した
    /// 中央値を、新実装（`elementwise_mul_mask`）・旧参照実装
    /// （`elementwise_mul_mask_reference`。`dense_vec` zip 経路）の
    /// 双方・連続／非連続の計 4 系列で比較する。stderr 出力のみで
    /// assert は行わない（実測記録は `docs/perf/
    /// train-reuse-relu-mask-stride.md`）。
    /// 実行例:
    /// `cargo test -p fandhe-ai-autodiff --release -- --ignored
    /// --nocapture mask_stride_microbench`
    #[test]
    #[ignore = "手動実行専用のマイクロベンチ（stderr 出力のみ）"]
    fn mask_stride_microbench() {
        use std::time::Instant;

        const ROWS: usize = 64;
        const COLS: usize = 256;
        const ITERS: usize = 1000;

        let contiguous_data: Vec<f32> = (0..ROWS * COLS).map(|i| ((i % 7) as f32) - 3.0).collect();
        let contiguous = t(&contiguous_data, &[ROWS, COLS]);
        let mask_contig = t(&contiguous_data, &[ROWS, COLS]);

        let transposed_src_data: Vec<f32> =
            (0..COLS * ROWS).map(|i| ((i % 7) as f32) - 3.0).collect();
        let transposed_src = t(&transposed_src_data, &[COLS, ROWS]);
        let non_contig = transpose2d(&transposed_src); // shape [ROWS, COLS]
        let mask_non_contig = t(&contiguous_data, &[ROWS, COLS]);

        let mut contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask(&contiguous, &mask_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            contig_times.push(start.elapsed());
        }
        let mut non_contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask(&non_contig, &mask_non_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            non_contig_times.push(start.elapsed());
        }
        // 旧実装（`dense_vec` zip 経路）との対照。新実装の連続経路が
        // 旧実装の連続経路を大きく下回っていないか（退行していないか）
        // を直接確認するための参考値。
        let mut reference_contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask_reference(&contiguous, &mask_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            reference_contig_times.push(start.elapsed());
        }
        let mut reference_non_contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask_reference(&non_contig, &mask_non_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            reference_non_contig_times.push(start.elapsed());
        }

        contig_times.sort();
        non_contig_times.sort();
        reference_contig_times.sort();
        reference_non_contig_times.sort();
        eprintln!(
            "mask_stride_microbench: contiguous median={:?} non_contiguous(transpose view) median={:?} reference_contiguous(dense_vec zip) median={:?} reference_non_contiguous(dense_vec zip) median={:?}",
            contig_times[ITERS / 2],
            non_contig_times[ITERS / 2],
            reference_contig_times[ITERS / 2],
            reference_non_contig_times[ITERS / 2]
        );
    }

    // --- Exp ---

    #[test]
    fn exp_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::exp(&a);
        let g = s.clone();
        let da = eval::mul(&g, &out_value);
        let num_da = numeric_grad_unary(&a, &s, eval::exp);

        assert_grad_close("exp dA", &da, &num_da);
    }

    // --- Tanh ---

    #[test]
    fn tanh_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::tanh(&a);
        let g = s.clone();
        let factor = tanh_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::tanh);

        assert_grad_close("tanh dA", &da, &num_da);
    }

    // --- Sigmoid（飽和域を含む） ---

    #[test]
    fn sigmoid_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid dA", &da, &num_da);
    }

    #[test]
    fn sigmoid_grad_saturated_region_matches_numeric() {
        // |x| が大きい飽和域（勾配 ≈ 0）でも中央差分と一致することを
        // 確認する（`eval::sigmoid` の数値安定形が飽和域で NaN/Inf を
        // 出さないことの間接検証も兼ねる）。
        let a = t(&[8.0, -8.0, 15.0, -15.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid(saturated) dA", &da, &num_da);
    }

    // --- Sum（dim: None / Some(0) / Some(1)） ---

    #[test]
    fn sum_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), None);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, None, &[]));

        assert_grad_close("sum(None) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(0));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(0), &[3]));

        assert_grad_close("sum(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(1));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(1), &[2]));

        assert_grad_close("sum(dim=1) dA", &da, &num_da);
    }

    // --- Max（dim: None / Some(0) / Some(1)。同値タイなし） ---

    #[test]
    fn max_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let out_value = eval::max(&a, None, &[]);
        let g = s.clone();
        let da = max_vjp(&a, None, &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, None, &[]));

        assert_grad_close("max(None) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let out_value = eval::max(&a, Some(0), &[3]);
        let g = s.clone();
        let da = max_vjp(&a, Some(0), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(0), &[3]));

        assert_grad_close("max(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let out_value = eval::max(&a, Some(1), &[2]);
        let g = s.clone();
        let da = max_vjp(&a, Some(1), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(1), &[2]));

        assert_grad_close("max(dim=1) dA", &da, &num_da);
    }

    // --- Max（同値タイ。#224: 先勝ち決定的挙動の回帰固定） ---
    //
    // タイ発生時は最大値位置が複数あり数値微分（中央差分）が定義でき
    // ないため、上記の同値タイなしケースとは異なり厳密値アサーション
    // （数値微分比較なし）で「最初に現れる最大要素 1 箇所のみに勾配が
    // 伝播し、他はゼロになる」先勝ち挙動そのものを固定する。

    #[test]
    fn max_grad_dim_none_tie_first_wins() {
        // 最大値 5.0 がインデックス 1・3 の 2 箇所に現れるタイケース。
        let a = t(&[1.0, 5.0, 3.0, 5.0], &[4]);
        let g = t(&[2.0], &[]);

        let out_value = eval::max(&a, None, &[]);
        let da = max_vjp(&a, None, &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![0.0, 2.0, 0.0, 0.0],
            "max(None) タイ時は最初に現れる最大要素（idx=1）のみへ伝播するはず"
        );
        // 勾配総量が上流勾配 g と一致すること（先勝ちでも保存量は保たれる）。
        assert_eq!(grad.iter().sum::<f32>(), 2.0);
    }

    #[test]
    fn max_grad_dim_axis_tie_first_wins() {
        // shape [2, 3]。行 0 は列 0・2 が 5.0 でタイ、行 1 はタイなし。
        let a = t(&[5.0, 1.0, 5.0, 1.0, -2.0, 4.0], &[2, 3]);
        let g = t(&[3.0, 7.0], &[2]);

        let out_value = eval::max(&a, Some(1), &[2]);
        let da = max_vjp(&a, Some(1), &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![3.0, 0.0, 0.0, 0.0, 0.0, 7.0],
            "max(dim=1) タイ行（行 0）は軸方向で最初の最大要素（列 0）のみへ伝播するはず"
        );
        // 各 (outer) スライスごとに勾配総量が上流勾配 g[outer] と一致すること。
        assert_eq!(grad[0..3].iter().sum::<f32>(), 3.0);
        assert_eq!(grad[3..6].iter().sum::<f32>(), 7.0);
    }

    // --- MseLoss（pred/target 両勾配） ---

    #[test]
    fn mse_loss_grad_mean_matches_numeric() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Mean));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Mean));

        assert_grad_close("mse(mean) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(mean) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_sum_matches_numeric() {
        // sum 縮約（#190）。scale が `2/n` ではなく `2` になる分岐を
        // mean と同じ数値微分ハーネスで検証する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Sum));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Sum));

        assert_grad_close("mse(sum) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(sum) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_n_zero_is_zero() {
        // numel() == 0 はゼロ除算を避け zeros を返す（ガード条件の
        // 直接検証。中央差分は空テンソルに対して定義できないため
        // 解析式のみで確認する）。mean/sum いずれも同じ早期 return
        // 経路（`n == 0` 分岐）を通るため mean のみ代表して検証する。
        let pred = build_tensor(Vec::new(), &[0]);
        let target = build_tensor(Vec::new(), &[0]);
        let g = t(&[1.0], &[]);
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert!(dense_vec(&dpred).is_empty());
        assert!(dense_vec(&dtarget).is_empty());
    }

    // --- CrossEntropyLoss（#191。PyTorch 参照値との突合は
    //     `tests/nn_cross_entropy.rs`、ここでは既存の
    //     `numeric_grad_unary`/`assert_grad_close`〈中央差分〉基盤に
    //     揃えた eval レベルの grad check を行う） ---

    #[test]
    fn cross_entropy_loss_grad_matches_numeric() {
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        // forward 出力は既に scalar shape [] のため、`s` も scalar
        // （`mse_loss_grad_matches_numeric` と同じ「射影 s がスカラー」
        // パターン）。
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let dlogits = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Mean, &g);
        let num_dlogits = numeric_grad_unary(&logits, &s, |x| {
            eval::cross_entropy_loss(x, &targets, 1, Reduction::Mean)
        });

        assert_grad_close("cross_entropy_loss(mean) dLogits", &dlogits, &num_dlogits);
    }

    #[test]
    fn cross_entropy_loss_grad_sum_matches_numeric() {
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let dlogits = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Sum, &g);
        let num_dlogits = numeric_grad_unary(&logits, &s, |x| {
            eval::cross_entropy_loss(x, &targets, 1, Reduction::Sum)
        });

        assert_grad_close("cross_entropy_loss(sum) dLogits", &dlogits, &num_dlogits);
    }

    // --- reduce_to_shape（中間軸縮約。ランク同一で先頭・末尾以外の
    //     軸を broadcast 元へ潰す経路。add/mul の bias broadcast テスト
    //     は末尾軸・スカラーテストは rank 0 のみで、この経路は未カバー
    //     だった） ---

    #[test]
    fn reduce_to_shape_middle_axis_matches_numeric() {
        // g: [2,4,3] を target_shape [2,1,3]（中間軸 dim=1 が size 1）
        // へ縮約する。add(a, b) の b 側勾配として同じ経路を通す
        // （a: [2,4,3]、b: [2,1,3] からの broadcast）。
        let a = t(
            &[
                1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 1.5, -0.5, 2.5, -1.5, 0.5, -2.5, 3.0, -1.0, 0.5,
                -0.5, 1.0, -1.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0,
            ],
            &[2, 4, 3],
        );
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[2, 1, 3]);
        let s = t(
            &[
                1.0, -1.0, 2.0, 0.5, 1.0, -0.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0, 0.2, -0.2, 0.4,
                0.1, 0.2, -0.1, 0.4, -0.4, 0.1, -0.1, 0.3, -0.2,
            ],
            &[2, 4, 3],
        );

        let db = reduce_to_shape(&s, b.shape());
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("reduce_to_shape(middle axis) dB", &db, &num_db);
    }

    // --- reduce_bias_grad（イシュー #1566・PR #1659 codex-review P2 是正） ---

    /// `reduce_bias_grad` が「軸 0（行）方向の bias 縮約」用の f64 経路
    /// （`eval::reduce_bias_grad_rows`）を、**軸 1（列）方向の
    /// broadcast bias**（`target_shape` の末尾次元が `g` の列数と
    /// 一致しない形状。例: `g: [2, 2]` に対する `target_shape: [2, 1]`）
    /// へ誤って適用しないことを確認する回帰テスト（codex-review 指摘。
    /// `[2, 1]` は総要素数が `2` で `g` の列数 `2` と偶然一致するため、
    /// 総要素数のみで判定する実装だと誤って行縮約の f64 経路へ分岐し
    /// てしまっていた）。
    #[test]
    fn reduce_bias_grad_does_not_misapply_row_reduction_to_column_broadcast_bias() {
        // g = [[1, 2], [3, 4]]（行優先）。target_shape = [2, 1] は
        // 各行の bias 値が 2 列へ複製される broadcast（軸 1 縮約）。
        // 正しい勾配は行ごとの和: row0 = 1+2=3・row1 = 3+4=7。
        // 行縮約（列ごとの和 col0=1+3=4・col1=2+4=6）を誤って適用すると
        // 全く異なる値になる。
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let got = reduce_bias_grad(&g, &[2, 1]);
        let expected = reduce_to_shape(&g, &[2, 1]);

        assert_eq!(got.shape(), &[2, 1]);
        assert_eq!(
            got.contiguous().as_slice().unwrap(),
            expected.contiguous().as_slice().unwrap(),
            "reduce_bias_grad は軸 1 縮約（[2, 1]）には reduce_to_shape をそのまま使う              はず（f64 行縮約経路を誤適用してはいけない）"
        );
        assert_eq!(
            got.contiguous().as_slice().unwrap(),
            &[3.0f32, 7.0],
            "軸 1 縮約の正しい値（行ごとの和）と一致するはず"
        );
    }

    /// 対照: 軸 0（行）方向の標準的な bias 縮約（`target_shape` の末尾
    /// 次元が `g` の列数と一致し、それより前の次元がすべて `1`）は
    /// 引き続き `eval::reduce_bias_grad_rows`（f64 経路）へ委譲される
    /// ことを、`[n]`・`[1, n]` の両形状で確認する（値は
    /// `reduce_to_shape`〈こちらは `f32` 逐次和〉と一致する範囲——
    /// 相殺による桁落ちがない入力なので両経路の値自体は一致するが、
    /// 経路選択の正しさを shape 網羅で確認する意図）。
    #[test]
    fn reduce_bias_grad_applies_row_reduction_for_rank1_and_leading_one_targets() {
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let got_rank1 = reduce_bias_grad(&g, &[2]);
        assert_eq!(got_rank1.shape(), &[2]);
        assert_eq!(
            got_rank1.contiguous().as_slice().unwrap(),
            &[4.0f32, 6.0],
            "target_shape=[2] は列ごとの和（col0=1+3=4・col1=2+4=6）のはず"
        );

        let got_leading_one = reduce_bias_grad(&g, &[1, 2]);
        assert_eq!(got_leading_one.shape(), &[1, 2]);
        assert_eq!(
            got_leading_one.contiguous().as_slice().unwrap(),
            &[4.0f32, 6.0],
            "target_shape=[1, 2] も同じ列ごとの和になるはず（先頭次元 1 個の reshape）"
        );
    }

    // --- vjp() ディスパッチの疎通確認（#18 との継ぎ目契約） ---
    //
    // Low 指摘: MatMul のみが vjp() 経由で疎通確認されており、他の
    // 8 演算（Add/Mul/Relu/Exp/Tanh/Sum/Max/MseLoss）は内部ヘルパーを
    // 直接呼ぶ形でしか検証されていなかった。各 match アームの配線
    // （`nodes[a.0]`/`nodes[b.0]` の対応順序）を通しで検証するため、
    // 全 9 演算（Leaf を除く）を vjp() 経由でテストする。Sigmoid
    // （TASK-9.1b・#92）追加により対象は 10 演算に拡大。
    // CrossEntropyLoss（#191）追加により対象は 11 演算に拡大
    // （Add/Mul/Relu/Exp/Tanh/Sigmoid/Sum/Max/MseLoss/MatMul/
    // CrossEntropyLoss）。

    fn leaf_node(value: Tensor<f32>) -> TapeNode {
        // `TapeNode`（TASK-12.1d・#164）は `shape` を独立フィールドとして
        // 持ち、`value` は `OnceCell` になった。テスト用の葉ノードは
        // 常に実体化済み（`OnceCell::from`）として構築する。
        let shape = value.shape().to_vec();
        TapeNode {
            op: Op::Leaf,
            shape,
            value: std::cell::OnceCell::from(value),
            lazy_chain_size: 0,
        }
    }

    /// `vjp()` の第 5 引数（`ops: &dyn BackendOps`）用テストフィクスチャ。
    /// 本モジュールのテストはすべて `leaf_node` で葉ノード（常に実体化
    /// 済み）のみを組み立てるため `materialize_fallible` は早期リターン
    /// し、`ops` の実体は使われない（`crate::test_support::TestOps` を
    /// 形式的に渡すのみ）。
    fn test_ops() -> crate::test_support::TestOps {
        crate::test_support::TestOps
    }

    #[test]
    fn vjp_dispatch_matmul_returns_both_inputs() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let out_value = eval::matmul(&a, &b);
        let (expected_da, expected_db) =
            matmul_vjp(&test_ops(), &a, &b, &t(&[1.0, -0.5, 0.3, 2.0], &[2, 2])).unwrap();
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::MatMul(NodeId(0), NodeId(1));
        let g = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_da));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_db));
    }

    #[test]
    fn vjp_dispatch_add_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::add(&a, &b);
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::Add(NodeId(0), NodeId(1));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&g));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&g));
    }

    #[test]
    fn vjp_dispatch_mul_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::mul(&a, &b);
        let nodes = vec![leaf_node(a.clone()), leaf_node(b.clone())];
        let op = Op::Mul(NodeId(0), NodeId(1));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&eval::mul(&g, &b)));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&eval::mul(&g, &a)));
    }

    #[test]
    fn vjp_dispatch_relu_returns_single_input() {
        let a = t(&[2.0, -3.0, 0.5, -0.02], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::relu(&a);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Relu(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&elementwise_mul_mask(&g, &a, |v| v > 0.0))
        );
    }

    #[test]
    fn vjp_dispatch_exp_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::exp(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Exp(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&eval::mul(&g, &out_value))
        );
    }

    #[test]
    fn vjp_dispatch_tanh_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::tanh(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Tanh(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &tanh_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sigmoid_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::sigmoid(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sigmoid(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &sigmoid_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sum_returns_single_input() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::sum(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sum {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = unreduce_broadcast(&g, &[2, 3], Some(0));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_max_returns_single_input() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::max(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Max {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = max_vjp(&a, Some(0), &out_value, &g);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_mse_loss_mean_returns_both_inputs_in_order() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Mean);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Mean,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_mse_loss_sum_returns_both_inputs_in_order() {
        // sum 縮約（#190）でも `Op::MseLoss` ディスパッチが reduction を
        // 正しく `mse_loss_vjp` へ引き渡すことを確認する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Sum);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Sum,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_cross_entropy_loss_returns_single_input() {
        // `targets` は非追跡（`NodeId` を持たない Op payload）のため、
        // `MseLoss`（pred/target 2 系統）とは異なり寄与は `logits` の
        // 1 系統のみ（`grads.len() == 1`）であることが配線検証の要点
        // （`tape::Op::CrossEntropyLoss` doc 参照）。
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let g = t(&[3.0], &[]);
        let out_value = eval::cross_entropy_loss(&logits, &targets, 1, Reduction::Mean);
        let nodes = vec![leaf_node(logits.clone())];
        let op = Op::CrossEntropyLoss {
            logits: NodeId(0),
            targets: targets.clone(),
            class_dim: 1,
            reduction: Reduction::Mean,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Mean, &g);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    // --- RmsNorm / LayerNorm（イシュー #1596） ---

    #[test]
    fn rmsnorm_grad_matches_numeric_no_weight() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let s_slice = dense_vec(&s);
        let (da, dw) = rmsnorm_vjp_rows(&x_slice, None, eps, rows, hidden, &s_slice);
        assert!(dw.is_none());
        let da = build_tensor(da, &[2, 3]);

        let num_da =
            numeric_grad_unary(&x, &s, |xt| eval::rmsnorm_rows(xt, None, eps, rows, hidden));
        assert_grad_close("rmsnorm dx (no weight)", &da, &num_da);
    }

    #[test]
    fn rmsnorm_grad_matches_numeric_with_weight() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let w = t(&[2.0, -1.0, 0.5], &[3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let w_slice = dense_vec(&w);
        let s_slice = dense_vec(&s);
        let (da, dw) = rmsnorm_vjp_rows(&x_slice, Some(&w_slice), eps, rows, hidden, &s_slice);
        let da = build_tensor(da, &[2, 3]);
        let dw = build_tensor(dw.expect("weight present"), &[3]);

        let num_da = numeric_grad_unary(&x, &s, |xt| {
            eval::rmsnorm_rows(xt, Some(&w_slice), eps, rows, hidden)
        });
        assert_grad_close("rmsnorm dx (weighted)", &da, &num_da);

        let num_dw = numeric_grad_unary(&w, &s, |wt| {
            eval::rmsnorm_rows(&x, Some(&dense_vec(wt)), eps, rows, hidden)
        });
        assert_grad_close("rmsnorm dw", &dw, &num_dw);
    }

    #[test]
    fn layer_norm_grad_matches_numeric_no_affine() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let s_slice = dense_vec(&s);
        let (da, dw, db) = layer_norm_vjp_rows(&x_slice, None, false, eps, rows, hidden, &s_slice);
        assert!(dw.is_none());
        assert!(db.is_none());
        let da = build_tensor(da, &[2, 3]);

        let num_da = numeric_grad_unary(&x, &s, |xt| {
            eval::layer_norm_rows(xt, None, None, eps, rows, hidden)
        });
        assert_grad_close("layer_norm dx (no affine)", &da, &num_da);
    }

    #[test]
    fn layer_norm_grad_matches_numeric_with_weight_and_bias() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let w = t(&[2.0, -1.0, 0.5], &[3]);
        let b = t(&[0.1, -0.2, 0.3], &[3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let w_slice = dense_vec(&w);
        let s_slice = dense_vec(&s);
        let (da, dw, db) =
            layer_norm_vjp_rows(&x_slice, Some(&w_slice), true, eps, rows, hidden, &s_slice);
        let da = build_tensor(da, &[2, 3]);
        let dw = build_tensor(dw.expect("weight present"), &[3]);
        let db = build_tensor(db.expect("bias present"), &[3]);

        let num_da = numeric_grad_unary(&x, &s, |xt| {
            eval::layer_norm_rows(xt, Some(&w_slice), Some(&dense_vec(&b)), eps, rows, hidden)
        });
        assert_grad_close("layer_norm dx (affine)", &da, &num_da);

        let num_dw = numeric_grad_unary(&w, &s, |wt| {
            eval::layer_norm_rows(
                &x,
                Some(&dense_vec(wt)),
                Some(&dense_vec(&b)),
                eps,
                rows,
                hidden,
            )
        });
        assert_grad_close("layer_norm dw", &dw, &num_dw);

        let num_db = numeric_grad_unary(&b, &s, |bt| {
            eval::layer_norm_rows(&x, Some(&w_slice), Some(&dense_vec(bt)), eps, rows, hidden)
        });
        assert_grad_close("layer_norm db", &db, &num_db);
    }

    #[test]
    fn vjp_dispatch_rms_norm_returns_input_and_weight() {
        let x = t(&[1.0, 2.0, -1.0, 0.5], &[1, 4]);
        let w = t(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[1, 4]);
        let eps = 1e-5f32;
        let out_value = eval::rmsnorm_rows(&x, Some(&dense_vec(&w)), eps, 1, 4);
        let nodes = vec![leaf_node(x.clone()), leaf_node(w.clone())];
        let op = Op::RmsNorm {
            input: NodeId(0),
            weight: Some(NodeId(1)),
            eps,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
    }

    #[test]
    fn vjp_dispatch_layer_norm_returns_input_weight_bias() {
        let x = t(&[1.0, 2.0, -1.0, 0.5], &[1, 4]);
        let w = t(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let b = t(&[0.0, 0.0, 0.0, 0.0], &[4]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[1, 4]);
        let eps = 1e-5f32;
        let out_value =
            eval::layer_norm_rows(&x, Some(&dense_vec(&w)), Some(&dense_vec(&b)), eps, 1, 4);
        let nodes = vec![
            leaf_node(x.clone()),
            leaf_node(w.clone()),
            leaf_node(b.clone()),
        ];
        let op = Op::LayerNorm {
            input: NodeId(0),
            weight: Some(NodeId(1)),
            bias: Some(NodeId(2)),
            eps,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(grads[2].0, NodeId(2));
    }

    // --- Softmax / LogSoftmax（イシュー #1594） ---
    //
    // softmax の行和は常に 1（一様重みでは L(x) = Σ softmax(x) が
    // 定数となり勾配が恒等的に 0 になってしまい検証が空になる）ため、
    // 射影重み `s` はすべて非一様にする。

    #[test]
    fn softmax_grad_matches_numeric_dim1() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::softmax_along(&x, 1);
        let da = softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::softmax_along(x, 1));

        assert_grad_close("softmax dim1", &da, &num_da);
    }

    #[test]
    fn softmax_grad_matches_numeric_dim0() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::softmax_along(&x, 0);
        let da = softmax_vjp_along(&out_value, &s, 0);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::softmax_along(x, 0));

        assert_grad_close("softmax dim0", &da, &num_da);
    }

    #[test]
    fn softmax_grad_matches_numeric_3d_middle_axis() {
        let x = t(
            &[
                1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.3, -0.2, 1.0, -1.0, 2.0, 0.1,
            ],
            &[2, 3, 2],
        );
        let s = t(
            &[
                1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 0.4, -0.8, 1.2, -0.3, 0.6, -1.5,
            ],
            &[2, 3, 2],
        );

        let out_value = eval::softmax_along(&x, 1);
        let da = softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::softmax_along(x, 1));

        assert_grad_close("softmax 3d middle axis", &da, &num_da);
    }

    #[test]
    fn log_softmax_grad_matches_numeric_dim1() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::log_softmax_along(&x, 1);
        let da = log_softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::log_softmax_along(x, 1));

        assert_grad_close("log_softmax dim1", &da, &num_da);
    }

    #[test]
    fn log_softmax_grad_matches_numeric_dim0() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::log_softmax_along(&x, 0);
        let da = log_softmax_vjp_along(&out_value, &s, 0);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::log_softmax_along(x, 0));

        assert_grad_close("log_softmax dim0", &da, &num_da);
    }

    #[test]
    fn log_softmax_grad_matches_numeric_3d_middle_axis() {
        let x = t(
            &[
                1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.3, -0.2, 1.0, -1.0, 2.0, 0.1,
            ],
            &[2, 3, 2],
        );
        let s = t(
            &[
                1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 0.4, -0.8, 1.2, -0.3, 0.6, -1.5,
            ],
            &[2, 3, 2],
        );

        let out_value = eval::log_softmax_along(&x, 1);
        let da = log_softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::log_softmax_along(x, 1));

        assert_grad_close("log_softmax 3d middle axis", &da, &num_da);
    }

    #[test]
    fn vjp_dispatch_softmax_returns_single_input() {
        let a = t(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::softmax_along(&a, 1);
        let nodes = vec![leaf_node(a)];
        let op = Op::Softmax {
            input: NodeId(0),
            dim: 1,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = softmax_vjp_along(&out_value, &g, 1);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_log_softmax_returns_single_input() {
        let a = t(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::log_softmax_along(&a, 1);
        let nodes = vec![leaf_node(a)];
        let op = Op::LogSoftmax {
            input: NodeId(0),
            dim: 1,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = log_softmax_vjp_along(&out_value, &g, 1);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    // codex-review 指摘（PR #1664）の回帰検証: `sum_acc`（f64）を
    // `exp(y)` との乗算前に `f32` へ downcast する実装では、有限の
    // `f32` 上流勾配でも overflow しうる（`logits=[0,0]` すなわち
    // `y=log_softmax([0,0])=[-ln(2),-ln(2)]`・上流勾配 `g=[2e38,2e38]`
    // で、正しい入力勾配 `[0,0]`〈`Σ_dim(g)=4e38` に対し `exp(y)=0.5`
    // なので `g - exp(y)*Σg = 2e38 - 0.5*4e38 = 0` のはずが、`sum_g`
    // を `f32` へ戻してから `exp(y) as f32 * sum_g` を計算すると
    // `0.5 * 4e38 = 2e38` は有限だが、`f32::MAX ≈ 3.4e38` に近い値の
    // 掛け算・加減算が丸め誤差で `-inf` を生む経路がある）。
    // `exp(y)` との乗算・`g` からの減算まで f64 で保持することで
    // overflow を避ける。
    #[test]
    fn log_softmax_vjp_along_large_upstream_grad_does_not_overflow() {
        let logits = t(&[0.0, 0.0], &[1, 2]);
        let y = eval::log_softmax_along(&logits, 1);
        let g = t(&[2e38, 2e38], &[1, 2]);
        let dx = log_softmax_vjp_along(&y, &g, 1);
        for (c, v) in dense_vec(&dx).iter().enumerate() {
            assert!(
                v.is_finite(),
                "dx[{c}] = {v} は有限であるべき（overflow 回帰）"
            );
            assert!(v.abs() < 1.0, "dx[{c}] = {v}（期待値は 0 近傍）");
        }
    }

    // codex-review 指摘（PR #1664）の回帰検証: 縮約後の `dot`（`Σ_dim
    // (g ⊙ y)`）を `y[idx] * (g[idx] - dot)` の減算まで `f32` で行う
    // 実装では、有限で表現可能な入力勾配が overflow して `inf`/`-inf`
    // になる。`y=[0.25,0.75]`・上流勾配 `g=[3e38,-3e38]` では
    // `dot=-1.5e38` に対し `g[0]-dot=4.5e38` が `f32::MAX`（約 3.4e38）
    // を超えて `f32` では overflow するが、正しい入力勾配
    // `y[0]*(g[0]-dot)=0.25*4.5e38=1.125e38` は有限。`log_softmax_vjp_
    // along` と同じく `g` からの減算・`y` との最終乗算まで `f64` で
    // 保持することで overflow を避ける。
    #[test]
    fn softmax_vjp_along_large_upstream_grad_does_not_overflow() {
        let y = t(&[0.25, 0.75], &[1, 2]);
        let g = t(&[3e38, -3e38], &[1, 2]);
        let dx = softmax_vjp_along(&y, &g, 1);
        let dx = dense_vec(&dx);
        for (c, v) in dx.iter().enumerate() {
            assert!(
                v.is_finite(),
                "dx[{c}] = {v} は有限であるべき（overflow 回帰）"
            );
        }
        assert!(
            (dx[0] - 1.125e38).abs() < 1e33,
            "dx[0] = {}（期待値 1.125e38 近傍）",
            dx[0]
        );
        assert!(
            (dx[1] + 1.125e38).abs() < 1e33,
            "dx[1] = {}（期待値 -1.125e38 近傍）",
            dx[1]
        );
    }

    // codex-review 指摘（PR #1664）の回帰検証: `eval::softmax_along`
    // 冒頭コメント参照。`shape[axis+1..]` 等の部分積は `checked_numel`
    // が通した shape（要素数積は `0`）でも overflow しうるため、
    // `softmax_vjp_along`／`log_softmax_vjp_along` も同じ早期 return
    // で部分積計算前に安全側へ倒れることを確認する。
    #[test]
    fn softmax_vjp_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let y = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let g = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let dx = softmax_vjp_along(&y, &g, 1);
        assert_eq!(dx.shape(), &shape);
        assert_eq!(dx.numel(), 0);
    }

    #[test]
    fn log_softmax_vjp_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let y = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let g = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let dx = log_softmax_vjp_along(&y, &g, 1);
        assert_eq!(dx.shape(), &shape);
        assert_eq!(dx.numel(), 0);
    }

    // --- イシュー #1583: elementwise VJP の BackendOps 経由化 ---
    //
    // `vjp_elementwise_mul`／`vjp_elementwise_add` はビルド時定数
    // `ELEMENTWISE_VJP_VIA_BACKEND_OPS` の値でゲートされるため、
    // `_via` バリアントへ両方の値を明示的に渡して両分岐を検証する
    // （`ELEMENTWISE_VJP_VIA_BACKEND_OPS` 自体の現在値に関わらず
    // テストが両分岐をカバーする）。

    /// `ops.mul`／`ops.add` を任意のエラーで応答させ、フォールバック・
    /// エラー伝播の分岐をテストするためだけの `BackendOps` モック。
    /// `mul`/`add` 以外は到達しないため `unreachable!` で明示的に失敗
    /// させる（静かな 0 埋め等の判定迂回を作らない。security.md A08）。
    struct MockOps {
        mul_result: Option<Result<Tensor<f32>, BackendError>>,
        add_result: Option<Result<Tensor<f32>, BackendError>>,
    }

    impl BackendOps for MockOps {
        fn device(&self) -> fandhe_ai_tensor_core::Device {
            fandhe_ai_tensor_core::Device::Cpu
        }
        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::gemm はイシュー #1583 テストでは使わない")
        }
        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            match &self.add_result {
                Some(Ok(t)) => Ok(t.clone()),
                Some(Err(e)) => Err(clone_backend_error(e)),
                None => Ok(crate::eval::add(a, b)),
            }
        }
        fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            match &self.mul_result {
                Some(Ok(t)) => Ok(t.clone()),
                Some(Err(e)) => Err(clone_backend_error(e)),
                None => Ok(crate::eval::mul(a, b)),
            }
        }
        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::relu はイシュー #1583 テストでは使わない")
        }
        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::exp はイシュー #1583 テストでは使わない")
        }
        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::tanh はイシュー #1583 テストでは使わない")
        }
        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::sum はイシュー #1583 テストでは使わない")
        }
        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::max はイシュー #1583 テストでは使わない")
        }
    }

    /// `BackendError` は `Clone` を持たないため、テスト用に必要な
    /// variant のみ手動で複製する（`Unsupported`／`ShapeMismatch`）。
    fn clone_backend_error(e: &BackendError) -> BackendError {
        match e {
            BackendError::Unsupported(msg) => BackendError::Unsupported(msg.clone()),
            BackendError::ShapeMismatch(err) => BackendError::ShapeMismatch(err.clone()),
            other => panic!("clone_backend_error: 未対応の variant {other:?}"),
        }
    }

    #[test]
    fn vjp_elementwise_mul_via_false_uses_eval_reference() {
        let g = t(&[1.0, 2.0, 3.0, -4.0], &[2, 2]);
        let rhs = t(&[5.0, -6.0, 0.5, 2.0], &[2, 2]);
        let got = vjp_elementwise_mul_via(&test_ops(), &g, &rhs, false).unwrap();
        let expected = eval::mul(&g, &rhs);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// ゲート `true`・`TestOps`（`ops.mul` が `eval::mul` へ委譲する
    /// 参照実装）経由でも `eval::mul` 直呼びと bit 完全一致する
    /// （NaN／-0.0 を含む。単一 IEEE 演算のため）。
    #[test]
    fn vjp_elementwise_mul_via_true_matches_eval_bit_exact() {
        let g = t(&[f32::NAN, -0.0, 1.0, 2.0], &[2, 2]);
        let rhs = t(&[3.0, 4.0, -0.0, f32::NAN], &[2, 2]);
        let got = vjp_elementwise_mul_via(&test_ops(), &g, &rhs, true).unwrap();
        let expected = eval::mul(&g, &rhs);
        for (a, b) in dense_vec(&got).iter().zip(dense_vec(&expected).iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// ブロードキャストを伴う `g ⊙ rhs` も `eval::mul` と一致する
    /// （`Op::Mul` の呼び出しパターン: `upstream` は forward 出力
    /// shape、`rhs` は入力側 shape で異なりうる）。
    #[test]
    fn vjp_elementwise_mul_via_true_handles_broadcast() {
        let g = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let rhs = t(&[10.0, 20.0, 30.0], &[3]);
        let got = vjp_elementwise_mul_via(&test_ops(), &g, &rhs, true).unwrap();
        let expected = eval::mul(&g, &rhs);
        assert_eq!(got.shape(), expected.shape());
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// `ops.mul` が `BackendError::Unsupported` を返した場合のみ
    /// `eval::mul` へフォールバックすることを確認する（vjp_elementwise_
    /// mul_via doc 参照）。
    #[test]
    fn vjp_elementwise_mul_via_true_falls_back_to_eval_on_unsupported() {
        let g = t(&[1.0, 2.0], &[2]);
        let rhs = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: Some(Err(BackendError::Unsupported("test".to_string()))),
            add_result: None,
        };
        let got = vjp_elementwise_mul_via(&mock, &g, &rhs, true).unwrap();
        let expected = eval::mul(&g, &rhs);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// `Unsupported` 以外のエラー（例: デバイス割当失敗を模した
    /// `ShapeMismatch`）は暗黙にフォールバックせず
    /// `AutodiffError::Backend` として伝播する（fail-closed。
    /// security.md A08）。
    #[test]
    fn vjp_elementwise_mul_via_true_propagates_non_unsupported_error() {
        let g = t(&[1.0, 2.0], &[2]);
        let rhs = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: Some(Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: vec![2],
                    rhs: vec![3],
                },
            ))),
            add_result: None,
        };
        let err = vjp_elementwise_mul_via(&mock, &g, &rhs, true).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    /// バックエンド実装が誤った shape のテンソルを返した場合、
    /// 静かに受け入れず fail-closed でエラーにする
    /// （`vjp_elementwise_mul_via` doc 参照）。
    #[test]
    fn vjp_elementwise_mul_via_true_rejects_wrong_output_shape() {
        let g = t(&[1.0, 2.0], &[2]);
        let rhs = t(&[3.0, 4.0], &[2]);
        let wrong_shape = t(&[1.0, 2.0, 3.0], &[3]);
        let mock = MockOps {
            mul_result: Some(Ok(wrong_shape)),
            add_result: None,
        };
        let err = vjp_elementwise_mul_via(&mock, &g, &rhs, true).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn vjp_elementwise_add_via_false_uses_eval_reference() {
        let a = t(&[1.0, 2.0, 3.0, -4.0], &[2, 2]);
        let b = t(&[5.0, -6.0, 0.5, 2.0], &[2, 2]);
        let got = vjp_elementwise_add_via(&test_ops(), &a, &b, false).unwrap();
        let expected = eval::add(&a, &b);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// `backward.rs::accumulate` の fan-out 合算（同 shape の 2 項和）を
    /// 模した bit 完全一致確認（NaN／-0.0 込み）。
    #[test]
    fn vjp_elementwise_add_via_true_matches_eval_bit_exact() {
        let a = t(&[f32::NAN, -0.0, 1.0, 2.0], &[2, 2]);
        let b = t(&[3.0, 4.0, -0.0, f32::NAN], &[2, 2]);
        let got = vjp_elementwise_add_via(&test_ops(), &a, &b, true).unwrap();
        let expected = eval::add(&a, &b);
        for (x, y) in dense_vec(&got).iter().zip(dense_vec(&expected).iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn vjp_elementwise_add_via_true_falls_back_to_eval_on_unsupported() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: None,
            add_result: Some(Err(BackendError::Unsupported("test".to_string()))),
        };
        let got = vjp_elementwise_add_via(&mock, &a, &b, true).unwrap();
        let expected = eval::add(&a, &b);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    #[test]
    fn vjp_elementwise_add_via_true_propagates_non_unsupported_error() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: None,
            add_result: Some(Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: vec![2],
                    rhs: vec![3],
                },
            ))),
        };
        let err = vjp_elementwise_add_via(&mock, &a, &b, true).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    /// `ELEMENTWISE_VJP_VIA_BACKEND_OPS` の現在の出荷値（ドリフト検出。
    /// `docs/perf/elementwise-vjp-backend-ops.md` の verdict と一致する
    /// ことを確認する。値を変える場合は同 doc の verdict 更新とセットで
    /// 変更すること）。
    #[test]
    fn elementwise_vjp_via_backend_ops_gate_matches_documented_default() {
        // `ELEMENTWISE_VJP_VIA_BACKEND_OPS` はビルド時定数のため、素の
        // `assert!` へ渡すと clippy::assertions_on_constants
        // （`-D warnings` 下でエラー）に抵触する。`const { }` ブロックへ
        // 包んでコンパイル時評価であることを明示し、ドリフト検出の意図
        // （ゲート既定値と doc の verdict の一致を機械的に固定する）を
        // 保ったまま clippy を通す。
        const { assert!(!ELEMENTWISE_VJP_VIA_BACKEND_OPS) };
    }

    // `Op::Mul`／`Op::Exp`／`Op::Tanh`／`Op::Sigmoid` の既存 grad-check
    // テスト群（本ファイル冒頭。数値微分との突合）はゲート値に関わらず
    // 現行ビルド設定（`ELEMENTWISE_VJP_VIA_BACKEND_OPS`）の下で実行され、
    // 既に green であることを既存テスト実行で確認済み（`vjp` 経由の
    // 統合経路のカバレッジは既存テストが担う。本節は `vjp_elementwise_
    // *_via` 単体のカバレッジを補う）。
    // --- Permute / BroadcastTo（イシュー #1597） ---

    #[test]
    fn inverse_permutation_roundtrip() {
        for perm in [
            vec![0usize, 1, 2],
            vec![2, 0, 1],
            vec![1, 0],
            vec![0],
            vec![3, 1, 0, 2],
        ] {
            let inv = inverse_permutation(&perm);
            for (k, &p) in perm.iter().enumerate() {
                assert_eq!(inv[p], k, "perm={perm:?} inv={inv:?} で往復しない");
            }
        }
    }

    #[test]
    fn permute_grad_matches_numeric() {
        let x = t(
            &[
                1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 2.5, -1.25,
            ],
            &[2, 3, 2],
        );
        let perm = [2usize, 0, 1];
        let out_value = x.permute(&perm).unwrap();
        let s = t(
            &[
                1.0, -0.5, 0.3, 2.0, -1.0, 0.5, -0.2, 1.2, 0.7, -0.3, 1.1, -0.9,
            ],
            out_value.shape(),
        );

        let g = s.clone();
        let inv = inverse_permutation(&perm);
        let da = g.permute(&inv).unwrap();

        let num_da = numeric_grad_unary(&x, &s, |v| v.permute(&perm).unwrap());
        assert_grad_close("permute dx", &da, &num_da);
    }

    #[test]
    fn broadcast_to_grad_matches_numeric_new_leading_axis() {
        // (a) 先頭軸新設: [3] → [2,3]
        let x = t(&[1.0, -2.0, 0.5], &[3]);
        let out_shape = [2usize, 3];
        let out_value = x.broadcast_to(&out_shape).unwrap();
        let s = t(&[1.0, -0.5, 0.3, 2.0, -1.0, 0.5], &out_shape);

        let da = reduce_to_shape(&s, x.shape());
        let num_da = numeric_grad_unary(&x, &s, |v| v.broadcast_to(&out_shape).unwrap());
        assert_grad_close("broadcast_to (new leading axis) dx", &da, &num_da);

        // out_value は forward zero-copy の確認（値そのものの検証は
        // shape 一致で足りる。broadcast_to の値契約自体は tensor-core
        // 側で検証済み）。
        assert_eq!(out_value.shape(), &out_shape);
    }

    #[test]
    fn broadcast_to_grad_matches_numeric_size_one_axis() {
        // (b) size-1 軸拡張: [2,1] → [2,3]
        let x = t(&[1.0, -2.0], &[2, 1]);
        let out_shape = [2usize, 3];
        let s = t(&[1.0, -0.5, 0.3, 2.0, -1.0, 0.5], &out_shape);

        let da = reduce_to_shape(&s, x.shape());
        let num_da = numeric_grad_unary(&x, &s, |v| v.broadcast_to(&out_shape).unwrap());
        assert_grad_close("broadcast_to (size-1 axis) dx", &da, &num_da);
    }

    #[test]
    fn broadcast_to_grad_matches_numeric_scalar() {
        // (c) rank 0 スカラー → [2,2]
        let x = t(&[2.0], &[]);
        let out_shape = [2usize, 2];
        let s = t(&[1.0, -0.5, 0.3, 2.0], &out_shape);

        let da = reduce_to_shape(&s, x.shape());
        let num_da = numeric_grad_unary(&x, &s, |v| v.broadcast_to(&out_shape).unwrap());
        assert_grad_close("broadcast_to (scalar) dx", &da, &num_da);
    }

    #[test]
    fn vjp_dispatch_permute_returns_single_input() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let perm = vec![1usize, 0];
        let out_value = a.permute(&perm).unwrap();
        let g = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.5], out_value.shape());
        let nodes = vec![leaf_node(a)];
        let op = Op::Permute {
            input: NodeId(0),
            perm: perm.clone(),
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = g.permute(&inverse_permutation(&perm)).unwrap();
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_broadcast_to_returns_single_input() {
        let a = t(&[1.0, -2.0, 0.5], &[3]);
        let out_shape = [2usize, 3];
        let out_value = a.broadcast_to(&out_shape).unwrap();
        let g = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.5], &out_shape);
        let nodes = vec![leaf_node(a)];
        let op = Op::BroadcastTo { input: NodeId(0) };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = reduce_to_shape(&g, &[3]);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    /// codex-review P2 指摘の是正（PRRT_kwDOTuUCJc6hxBOl。設計 `docs/
    /// autodiff-rnn-cell-tape-design.md` 決定 11(h)）: `Op::GruCell` の
    /// `q`（決定 1c: `pre_h` の n 列ブロック。GEMM 再計算を避けるため
    /// backward で `∂n/∂r` の復元に直接使う payload）を意図的に
    /// 破損させると、backward の結果が変化することを確認する構造
    /// テスト。`eval::gru_backward` の式（`dr = d_pre_n * q_val` →
    /// `d_pre_r`。本ファイル冒頭の doc 参照）により、`q` は r ゲート
    /// 列ブロックの勾配にのみ影響するため、`col_start=0` の全幅 embed
    /// を経由する `w_ih`（`affine_vjp` の `d_weight` 戻り値。r 列
    /// ブロックを含む全幅）が `q` の値に応じて変化するはずである。
    /// 変化しなければ `q` が実際には使われていない（GEMM 再計算に
    /// フォールバックしている、または死んでいる）ことを意味する。
    #[test]
    fn vjp_gru_cell_backward_is_sensitive_to_stored_q_payload() {
        // D=2, hidden=1, B=1（`total_cols = gates(=3) * hidden = 3`）。
        let x = t(&[1.0, -0.5], &[1, 2]);
        let h_prev = t(&[0.3], &[1, 1]);
        let w_ih = t(&[0.1, 0.2, -0.1, 0.05, 0.3, -0.2], &[2, 3]);
        let w_hh = t(&[0.2, -0.1, 0.05], &[1, 3]);
        // gates_rzn（活性化後の r,z,n。値域は sigmoid/tanh 範囲内）。
        let gates_rzn = t(&[0.6, 0.4, 0.2], &[1, 3]);
        let dh = t(&[1.0], &[1, 1]);
        // Op::GruCell 分岐は `out_value` を参照しない（本ファイル上部の
        // `Op::GruCell` 分岐実装参照）ためプレースホルダで足りる。
        let out_value = t(&[0.0], &[1, 1]);

        let nodes = vec![
            leaf_node(x),
            leaf_node(h_prev),
            leaf_node(w_ih),
            leaf_node(w_hh),
        ];

        let q_correct = t(&[0.5], &[1, 1]);
        let q_corrupted = t(&[9.0], &[1, 1]);

        let op_correct = Op::GruCell {
            x: NodeId(0),
            h_prev: NodeId(1),
            w_ih: NodeId(2),
            w_hh: NodeId(3),
            b_ih: None,
            b_hh: None,
            gates_rzn: gates_rzn.clone(),
            q: q_correct,
        };
        let op_corrupted = Op::GruCell {
            x: NodeId(0),
            h_prev: NodeId(1),
            w_ih: NodeId(2),
            w_hh: NodeId(3),
            b_ih: None,
            b_hh: None,
            gates_rzn,
            q: q_corrupted,
        };

        let grads_correct = vjp(
            &op_correct,
            &out_value,
            &dh,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();
        let grads_corrupted = vjp(
            &op_corrupted,
            &out_value,
            &dh,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        let dw_ih_correct = grads_correct
            .iter()
            .find(|(id, _)| *id == NodeId(2))
            .map(|(_, g)| dense_vec(g))
            .expect("w_ih への寄与が存在するはず");
        let dw_ih_corrupted = grads_corrupted
            .iter()
            .find(|(id, _)| *id == NodeId(2))
            .map(|(_, g)| dense_vec(g))
            .expect("w_ih への寄与が存在するはず");

        assert_ne!(
            dw_ih_correct, dw_ih_corrupted,
            "q を破損させても w_ih 勾配が変化しない: q payload が backward で実際に \
             使われていない（GEMM 再計算・死んだ payload 等の）疑いがある"
        );
    }

    /// codex-review P2 指摘の是正（PRRT_kwDOTuUCJc6hxBOl。設計 `docs/
    /// autodiff-rnn-cell-tape-design.md` 決定 11(j)）: `Op::LstmHidden`
    /// の VJP が `cell`（`NodeId`）経由で `nodes[cell.0].op` を参照し、
    /// そこに保持された `Op::LstmCell.w_ih`／`w_hh` の**実データ**を
    /// 読んでいることを確認する構造テスト。`cell` が指す先の
    /// `Op::LstmCell` ノードの `w_ih`／`w_hh` leaf データだけを差し替え
    /// (`x`／`h_prev`／`gates_ifg`／`gate_o` 等は完全に同一のまま)、
    /// `affine_vjp` が返す `dx`（`w_ih_val` に依存）・`dh_prev`
    /// （`w_hh_val` に依存）が変化することを確認する。変化しなければ
    /// `cell` 参照が実際には読まれていない（固定値・別経路へのフォール
    /// バック等）ことを意味する。
    #[test]
    fn vjp_lstm_hidden_reads_referenced_cell_node_weight_data() {
        let x = t(&[1.0, -0.5], &[1, 2]);
        let h_prev = t(&[0.3, -0.2], &[1, 2]);
        let c_prev = t(&[0.1, 0.4], &[1, 2]);
        let w_ih_a = t(
            &[
                0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, -0.05, 0.2, -0.3, 0.1, 0.25, 0.05, -0.1,
                0.2, -0.15,
            ],
            &[2, 8],
        );
        let w_hh_a = t(
            &[
                0.2, -0.1, 0.05, 0.1, -0.2, 0.3, 0.1, 0.05, -0.1, 0.2, 0.15, -0.05, 0.1, 0.2,
                -0.05, 0.15,
            ],
            &[2, 8],
        );
        // w_ih_b／w_hh_b は w_ih_a／w_hh_a と全要素 +1.0 だけ異なる
        // （shape 同一・データのみ破損させた「別の」重み）。
        let w_ih_b = t(
            &dense_vec(&w_ih_a)
                .iter()
                .map(|v| v + 1.0)
                .collect::<Vec<_>>(),
            &[2, 8],
        );
        let w_hh_b = t(
            &dense_vec(&w_hh_a)
                .iter()
                .map(|v| v + 1.0)
                .collect::<Vec<_>>(),
            &[2, 8],
        );
        let gates_ifg = t(&[0.6, 0.4, 0.3, 0.7, -0.2, 0.5], &[1, 6]);
        let gate_o = t(&[0.55, 0.45], &[1, 2]);
        let c_t = t(&[0.2, -0.1], &[1, 2]);
        let h_t = t(&[0.1, 0.05], &[1, 2]);
        let dh = t(&[1.0, -1.0], &[1, 2]);

        let build_nodes = |w_ih: Tensor<f32>, w_hh: Tensor<f32>| {
            let cell_op = Op::LstmCell {
                x: NodeId(0),
                h_prev: NodeId(1),
                c_prev: NodeId(2),
                w_ih: NodeId(3),
                w_hh: NodeId(4),
                b_ih: None,
                b_hh: None,
                gates_ifg: gates_ifg.clone(),
            };
            // `Op::LstmCell` ノードは常に実体化済み（`tape.rs::Op::
            // LstmCell` doc の push_eager 契約）のため、テスト用にも
            // `OnceCell::from` で事前に値を設定する。
            let cell_node = TapeNode {
                op: cell_op,
                shape: c_t.shape().to_vec(),
                value: std::cell::OnceCell::from(c_t.clone()),
                lazy_chain_size: 0,
            };
            vec![
                leaf_node(x.clone()),
                leaf_node(h_prev.clone()),
                leaf_node(c_prev.clone()),
                leaf_node(w_ih),
                leaf_node(w_hh),
                cell_node,
            ]
        };

        let nodes_a = build_nodes(w_ih_a, w_hh_a);
        let nodes_b = build_nodes(w_ih_b, w_hh_b);
        let op_hidden = Op::LstmHidden {
            cell: NodeId(5),
            gate_o,
        };

        let grads_a = vjp(
            &op_hidden,
            &h_t,
            &dh,
            &nodes_a,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();
        let grads_b = vjp(
            &op_hidden,
            &h_t,
            &dh,
            &nodes_b,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        let dx_a = grads_a
            .iter()
            .find(|(id, _)| *id == NodeId(0))
            .map(|(_, g)| dense_vec(g))
            .expect("x への寄与が存在するはず");
        let dx_b = grads_b
            .iter()
            .find(|(id, _)| *id == NodeId(0))
            .map(|(_, g)| dense_vec(g))
            .expect("x への寄与が存在するはず");
        let dh_prev_a = grads_a
            .iter()
            .find(|(id, _)| *id == NodeId(1))
            .map(|(_, g)| dense_vec(g))
            .expect("h_prev への寄与が存在するはず");
        let dh_prev_b = grads_b
            .iter()
            .find(|(id, _)| *id == NodeId(1))
            .map(|(_, g)| dense_vec(g))
            .expect("h_prev への寄与が存在するはず");

        assert_ne!(
            dx_a, dx_b,
            "cell 参照先の w_ih データを差し替えても dx が変化しない: LstmHidden の \
             VJP が cell 経由の w_ih を実際に読んでいない疑いがある"
        );
        assert_ne!(
            dh_prev_a, dh_prev_b,
            "cell 参照先の w_hh データを差し替えても dh_prev が変化しない: LstmHidden \
             の VJP が cell 経由の w_hh を実際に読んでいない疑いがある"
        );
    }

    // --- Where／MaskedFill（イシュー #1637） ---

    /// `where_vjp` の解析勾配が数値微分と一致することを確認する
    /// （同 shape・分岐反転なしの固定 `cond`）。`cond` 自体は摂動対象
    /// 外（定数マスク）のため `numeric_grad_unary` をそのまま使える。
    #[test]
    fn where_grad_matches_numeric_same_shape() {
        let cond = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let g = s.clone();
        let (da, db) = where_vjp(&cond, &g, &[2, 2], &[2, 2]);

        let num_da = numeric_grad_unary(&a, &s, |x| eval::where_cond(&cond, x, &b, &[2, 2]));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::where_cond(&cond, &a, x, &[2, 2]));

        assert_grad_close("where dA", &da, &num_da);
        assert_grad_close("where dB", &db, &num_db);
    }

    /// broadcast（`a: [2,2]`, `b: [2]`）で `db` が行方向へ縮約される
    /// ことを確認する（`Op::Mul` の broadcast VJP と同じ縮約契約）。
    #[test]
    fn where_grad_broadcast_reduces_to_input_shape() {
        let cond = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let (da, db) = where_vjp(&cond, &g, &[2, 2], &[2]);

        assert_eq!(da.shape(), &[2, 2]);
        assert_eq!(db.shape(), &[2]);
        // da: cond!=0 の位置のみ g を通す。
        assert_eq!(dense_vec(&da), vec![1.0, 0.0, 0.0, 4.0]);
        // db: cond==0 の位置のみ g を通し、行方向（broadcast 元軸）で
        // 合算する。cond=[[1,0],[0,1]]・g=[[1,2],[3,4]] より
        // masked=[[0,2],[3,0]]・列ごとの和=[0+3, 2+0]=[3, 2]。
        assert_eq!(dense_vec(&db), vec![3.0, 2.0]);
    }

    /// 同一 `Var` を `a`／`b` 両方に指定した場合（`where(c, x, x)`）、
    /// `accumulate` が合算する前提のもと、`da + db == g`（全域で
    /// upstream をそのまま通す）ことを確認する。
    #[test]
    fn where_grad_same_var_both_sides_sums_to_upstream() {
        let cond = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let (da, db) = where_vjp(&cond, &g, &[2, 2], &[2, 2]);
        let sum: Vec<f32> = dense_vec(&da)
            .iter()
            .zip(dense_vec(&db).iter())
            .map(|(&a, &b)| a + b)
            .collect();
        assert_eq!(sum, dense_vec(&g));
    }

    /// NaN が非選択側に留まる（選択側の値・勾配へ伝播しない）ことを
    /// 確認する。`cond` が `1.0` の位置では `b` 側に NaN があっても
    /// forward 出力・`da` は NaN の影響を受けない。
    #[test]
    fn where_forward_and_grad_isolate_nan_to_unselected_side() {
        let cond = t(&[1.0, 0.0], &[2]);
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[f32::NAN, 20.0], &[2]);
        let out_shape = [2usize];

        let value = eval::where_cond(&cond, &a, &b, &out_shape);
        assert_eq!(dense_vec(&value), vec![1.0, 20.0]);

        let g = t(&[1.0, 1.0], &[2]);
        let (da, db) = where_vjp(&cond, &g, &out_shape, &out_shape);
        assert_eq!(dense_vec(&da), vec![1.0, 0.0]);
        // db[0] は `cond[0] != 0.0` により 0 になるはず（NaN の位置は
        // 選択されていないため upstream を通さない）。
        assert_eq!(dense_vec(&db)[0], 0.0);
        assert_eq!(dense_vec(&db)[1], 1.0);
    }

    /// `masked_fill_vjp` の解析勾配が数値微分と一致することを確認する
    /// （fill 位置の勾配は 0、それ以外は upstream をそのまま通す）。
    #[test]
    fn masked_fill_grad_matches_numeric() {
        let mask = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);
        let value = -9.0f32;

        let g = s.clone();
        let dx = masked_fill_vjp(&mask, &g);

        let num_dx = numeric_grad_unary(&x, &s, |t| eval::masked_fill(t, &mask, value));

        assert_grad_close("masked_fill dX", &dx, &num_dx);
    }

    /// fill 位置の勾配が厳密に 0 であることを直接確認する。
    #[test]
    fn masked_fill_grad_zero_at_filled_positions() {
        let mask = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let dx = masked_fill_vjp(&mask, &g);

        assert_eq!(dense_vec(&dx), vec![0.0, 2.0, 0.0, 4.0]);
    }
}
