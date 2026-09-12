//! activation checkpointing（イシュー #1624・`docs/
//! autodiff-checkpoint-design.md`）の統合テスト。
//!
//! `Tape::checkpoint`（閉包版）／`Var::checkpoint_from`（低儀式版）が
//! 登録・解放する区間について、以下を検証する:
//! - checkpoint 有無で `Tape::backward` の勾配が **bit 同一**になる
//!   （再計算が forward と同じ `ops`／`eval` 呼び出しを再現するため）。
//! - 数値微分（中央差分）との突合。
//! - 複数区間・入れ子区間・`reset()` 後の再利用・エラー系（クロス
//!   テープ・閉包失敗時の非解放・空区間の no-op）。
//! - 非適格 Op（`qr`）を含む区間でもエラーにならず勾配が変わらない。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

const H: f64 = 1e-3;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// `mlp_grads` の戻り値（dx／dw1／dw2 の 3 テンソル）用の型エイリアス
/// （clippy::type_complexity 対応）。
type MlpGrads = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

/// `matmul → relu → matmul → sigmoid → mse_loss` の MLP 連鎖。
/// `use_checkpoint` で中間 2 段（`h1 = relu(x@w1)`・`h2 =
/// sigmoid(h1@w2)`）を `Tape::checkpoint` の区間に収める。
fn mlp_loss(
    tape: &Tape,
    x: &Tensor<f32>,
    w1: &Tensor<f32>,
    w2: &Tensor<f32>,
    target: &Tensor<f32>,
    use_checkpoint: bool,
) -> Result<f32, AutodiffError> {
    let xv = tape.var(x);
    let w1v = tape.var(w1);
    let w2v = tape.var(w2);
    let tv = tape.var(target);

    let h2 = if use_checkpoint {
        tape.checkpoint(|| {
            let h1 = xv.matmul(&w1v)?.relu();
            Ok(h1.matmul(&w2v)?.sigmoid())
        })?
    } else {
        let h1 = xv.matmul(&w1v)?.relu();
        h1.matmul(&w2v)?.sigmoid()
    };
    let loss = h2.mse_loss(&tv)?;
    Ok(scalar(&loss.to_tensor()))
}

fn mlp_grads(
    x: &Tensor<f32>,
    w1: &Tensor<f32>,
    w2: &Tensor<f32>,
    target: &Tensor<f32>,
    use_checkpoint: bool,
) -> Result<MlpGrads, AutodiffError> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let w1v = tape.var(w1);
    let w2v = tape.var(w2);
    let tv = tape.var(target);

    let h2 = if use_checkpoint {
        tape.checkpoint(|| {
            let h1 = xv.matmul(&w1v)?.relu();
            Ok(h1.matmul(&w2v)?.sigmoid())
        })?
    } else {
        let h1 = xv.matmul(&w1v)?.relu();
        h1.matmul(&w2v)?.sigmoid()
    };
    let loss = h2.mse_loss(&tv)?;
    let grads = tape.backward(&loss)?;
    let dx = grads
        .get(&xv)?
        .cloned()
        .unwrap_or_else(|| t(vec![0.0; dense(x).len()], x.shape()));
    let dw1 = grads
        .get(&w1v)?
        .cloned()
        .unwrap_or_else(|| t(vec![0.0; dense(w1).len()], w1.shape()));
    let dw2 = grads
        .get(&w2v)?
        .cloned()
        .unwrap_or_else(|| t(vec![0.0; dense(w2).len()], w2.shape()));
    Ok((dx, dw1, dw2))
}

fn fixture() -> (Tensor<f32>, Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    (
        t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]),
        t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]),
        t(vec![0.3, -0.4, 0.7, -0.1], &[2, 2]),
        t(vec![0.1, 0.0, 0.9, 0.2], &[2, 2]),
    )
}

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>) {
    let da = dense(a);
    let db = dense(b);
    assert_eq!(da.len(), db.len(), "shape mismatch in bit-identity check");
    for (i, (x, y)) in da.iter().zip(db.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "element {i} differs: {x} (bits {:x}) vs {y} (bits {:x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

// --- (1) checkpoint 有無で勾配が bit 同一 ---

#[test]
fn checkpoint_grads_are_bit_identical_to_no_checkpoint() {
    let (x, w1, w2, target) = fixture();
    let (dx0, dw10, dw20) = mlp_grads(&x, &w1, &w2, &target, false).unwrap();
    let (dx1, dw11, dw21) = mlp_grads(&x, &w1, &w2, &target, true).unwrap();
    assert_bit_identical(&dx0, &dx1);
    assert_bit_identical(&dw10, &dw11);
    assert_bit_identical(&dw20, &dw21);
}

// --- (2) 数値微分との突合（checkpoint あり側） ---

#[test]
fn checkpoint_grad_matches_numeric_grad() {
    let (x, w1, w2, target) = fixture();
    let (dx, _dw1, _dw2) = mlp_grads(&x, &w1, &w2, &target, true).unwrap();

    let shape = x.shape().to_vec();
    let numel: usize = shape.iter().product();
    let base = dense(&x);
    let mut numeric = vec![0f32; numel];
    for i in 0..numel {
        let mut plus = base.clone();
        plus[i] = (plus[i] as f64 + H) as f32;
        let lp = mlp_loss(
            &Tape::new_with_ops(common::naive_ops()),
            &t(plus, &shape),
            &w1,
            &w2,
            &target,
            true,
        )
        .unwrap() as f64;
        let mut minus = base.clone();
        minus[i] = (minus[i] as f64 - H) as f32;
        let lm = mlp_loss(
            &Tape::new_with_ops(common::naive_ops()),
            &t(minus, &shape),
            &w1,
            &w2,
            &target,
            true,
        )
        .unwrap() as f64;
        numeric[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    let analytic = dense(&dx);
    for (i, (&a, &n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let diff = (a as f64 - n as f64).abs();
        let rel = diff / (n.abs() as f64).max(1.0);
        assert!(
            rel < 1e-2 || diff < 1e-3,
            "element {i}: analytic={a} numeric={n} diff={diff} rel={rel}"
        );
    }
}

// --- (3) 複数区間（3 区間連鎖） ---

#[test]
fn multiple_sequential_checkpoint_regions_match_no_checkpoint() {
    let tape_plain = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]);
    let w1 = t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]);
    let w2 = t(vec![0.3, -0.4, 0.7, -0.1], &[2, 2]);
    let w3 = t(vec![0.2, 0.1, -0.3, 0.4], &[2, 2]);

    let xv = tape_plain.var(&x);
    let w1v = tape_plain.var(&w1);
    let w2v = tape_plain.var(&w2);
    let w3v = tape_plain.var(&w3);
    let h1 = xv.matmul(&w1v).unwrap().relu();
    let h2 = h1.matmul(&w2v).unwrap().relu();
    let h3 = h2.matmul(&w3v).unwrap().sigmoid();
    let loss_plain = h3.sum(None).unwrap();
    let grads_plain = tape_plain.backward(&loss_plain).unwrap();
    let dx_plain = grads_plain.get(&xv).unwrap().cloned().unwrap();

    let tape_ckpt = Tape::new_with_ops(common::naive_ops());
    let xv = tape_ckpt.var(&x);
    let w1v = tape_ckpt.var(&w1);
    let w2v = tape_ckpt.var(&w2);
    let w3v = tape_ckpt.var(&w3);
    let h1 = tape_ckpt
        .checkpoint(|| Ok(xv.matmul(&w1v)?.relu()))
        .unwrap();
    let h2 = tape_ckpt
        .checkpoint(|| Ok(h1.matmul(&w2v)?.relu()))
        .unwrap();
    let h3 = tape_ckpt
        .checkpoint(|| Ok(h2.matmul(&w3v)?.sigmoid()))
        .unwrap();
    let loss_ckpt = h3.sum(None).unwrap();
    let grads_ckpt = tape_ckpt.backward(&loss_ckpt).unwrap();
    let dx_ckpt = grads_ckpt.get(&xv).unwrap().cloned().unwrap();

    assert_bit_identical(&dx_plain, &dx_ckpt);
}

// --- (4) 入れ子区間 ---

#[test]
fn nested_checkpoint_regions_match_no_checkpoint() {
    let (x, w1, w2, target) = fixture();
    let w3 = t(vec![0.2, 0.1, -0.3, 0.4], &[2, 2]);

    let tape_plain = Tape::new_with_ops(common::naive_ops());
    let xv = tape_plain.var(&x);
    let w1v = tape_plain.var(&w1);
    let w2v = tape_plain.var(&w2);
    let w3v = tape_plain.var(&w3);
    let tv = tape_plain.var(&target);
    let h1 = xv.matmul(&w1v).unwrap().relu();
    let h2 = h1.matmul(&w2v).unwrap().sigmoid();
    let h3 = h2.matmul(&w3v).unwrap().sigmoid();
    let loss_plain = h3.mse_loss(&tv).unwrap();
    let grads_plain = tape_plain.backward(&loss_plain).unwrap();
    let dx_plain = grads_plain.get(&xv).unwrap().cloned().unwrap();

    let tape_ckpt = Tape::new_with_ops(common::naive_ops());
    let xv = tape_ckpt.var(&x);
    let w1v = tape_ckpt.var(&w1);
    let w2v = tape_ckpt.var(&w2);
    let w3v = tape_ckpt.var(&w3);
    let tv = tape_ckpt.var(&target);
    // 外側区間の中で内側区間を 1 回呼ぶ（入れ子）。
    let h3 = tape_ckpt
        .checkpoint(|| {
            let h1 = xv.matmul(&w1v)?.relu();
            let h2 = tape_ckpt.checkpoint(|| Ok(h1.matmul(&w2v)?.sigmoid()))?;
            Ok(h2.matmul(&w3v)?.sigmoid())
        })
        .unwrap();
    let loss_ckpt = h3.mse_loss(&tv).unwrap();
    let grads_ckpt = tape_ckpt.backward(&loss_ckpt).unwrap();
    let dx_ckpt = grads_ckpt.get(&xv).unwrap().cloned().unwrap();

    assert_bit_identical(&dx_plain, &dx_ckpt);
}

// --- (5) backward() を 2 回呼んでも同一結果 ---

#[test]
fn backward_called_twice_after_checkpoint_yields_same_gradient() {
    let (x, w1, w2, target) = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let w1v = tape.var(&w1);
    let w2v = tape.var(&w2);
    let tv = tape.var(&target);
    let h2 = tape
        .checkpoint(|| {
            let h1 = xv.matmul(&w1v)?.relu();
            Ok(h1.matmul(&w2v)?.sigmoid())
        })
        .unwrap();
    let loss = h2.mse_loss(&tv).unwrap();
    let grads1 = tape.backward(&loss).unwrap();
    let grads2 = tape.backward(&loss).unwrap();
    let dx1 = grads1.get(&xv).unwrap().cloned().unwrap();
    let dx2 = grads2.get(&xv).unwrap().cloned().unwrap();
    assert_bit_identical(&dx1, &dx2);
}

// --- (6) 閉包が Err を返した場合、何も解放されずテープは使い続けられる ---

#[test]
fn failing_closure_does_not_release_and_tape_remains_usable() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let other_tape = Tape::new_with_ops(common::naive_ops());
    let y_other = other_tape.var(&t(vec![1.0, 2.0], &[2]));

    // クロステープの `matmul` はエラーを返す（shape 検査より前に
    // `check_same_tape` で弾かれる）。
    let result = tape.checkpoint(|| x.matmul(&y_other));
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));

    // テープはまだ使える（矛盾した状態で壊れていない）。
    let z = x.sum(None).unwrap();
    let grads = tape.backward(&z).unwrap();
    assert!(grads.get(&x).unwrap().is_some());
}

// --- (7) 他テープの Var を返すと TapeMismatch ---

#[test]
fn checkpoint_returning_var_from_another_tape_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let _x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let other_tape = Tape::new_with_ops(common::naive_ops());
    let y_other = other_tape.var(&t(vec![1.0, 2.0], &[2]));

    let result = tape.checkpoint(|| y_other.sum(None));
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

// --- (8) 区間が空（既存 Var を返す）は no-op ---

#[test]
fn empty_region_returning_existing_var_is_noop() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    // 閉包が新規ノードを push せず既存の x をそのまま返す。
    let out = tape.checkpoint(|| Ok(x)).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&x).unwrap().is_some());
}

// --- (9) reset 後にレジストリが消え新 epoch でも問題なく動く ---

#[test]
fn checkpoint_registry_is_cleared_after_reset() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    {
        let x = tape.var(&t(vec![1.0, 2.0], &[1, 2]));
        let w = tape.var(&t(vec![0.5, -0.5], &[2]));
        let y = tape.checkpoint(|| x.matmul(&w.reshape(&[2, 1])?)).unwrap();
        let loss = y.sum(None).unwrap();
        let _ = tape.backward(&loss).unwrap();
    }
    tape.reset();
    // reset 後も新しい演算・checkpoint が問題なく機能する。
    let x2 = tape.var(&t(vec![3.0, 4.0], &[1, 2]));
    let w2 = tape.var(&t(vec![1.0, 1.0], &[2]));
    let y2 = tape
        .checkpoint(|| x2.matmul(&w2.reshape(&[2, 1])?))
        .unwrap();
    let loss2 = y2.sum(None).unwrap();
    let grads2 = tape.backward(&loss2).unwrap();
    assert!(grads2.get(&x2).unwrap().is_some());
}

// --- (10) MAX_FUSED_CHAIN_LEN 到達を含む融合連鎖でも勾配 bit 同一 ---

/// (10) 用の計算本体。`Var<'t> -> Result<Var<'t>, _>` を自由関数化する
/// ことで、テスト内クロージャの `Var<'_>` 楽観推論が生む独立ライフタイム
/// 変数間の不一致（`'1` と `'2` が同一だと単一化できない）を避ける。
fn long_chain_compute<'t>(
    v: &fandhe_ai_autodiff::Var<'t>,
) -> Result<fandhe_ai_autodiff::Var<'t>, AutodiffError> {
    let a = v.relu();
    let b = a.exp();
    let c = b.tanh();
    let d = c.add(&a)?;
    let e = d.mul(&b)?;
    Ok(e)
}

#[test]
fn checkpoint_region_with_long_elementwise_chain_matches_no_checkpoint() {
    let x = t(vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.7, -0.1], &[2, 4]);

    let run = |use_checkpoint: bool| -> Tensor<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let out = if use_checkpoint {
            tape.checkpoint(|| long_chain_compute(&xv)).unwrap()
        } else {
            long_chain_compute(&xv).unwrap()
        };
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        grads.get(&xv).unwrap().cloned().unwrap()
    };

    let dx_plain = run(false);
    let dx_ckpt = run(true);
    assert_bit_identical(&dx_plain, &dx_ckpt);
}

// --- (11) 逃げ出した中間 Var に value()（層 2）を呼んでも再計算値が
//          元値と bit 同一 ---

#[test]
fn escaped_intermediate_var_value_matches_original_after_release() {
    let (x, w1, w2, _target) = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let w1v = tape.var(&w1);
    let w2v = tape.var(&w2);

    let mut escaped: Option<fandhe_ai_autodiff::Var<'_>> = None;
    let h2 = tape
        .checkpoint(|| {
            let h1 = xv.matmul(&w1v)?.relu();
            escaped = Some(h1);
            Ok(h1.matmul(&w2v)?.sigmoid())
        })
        .unwrap();
    let h1 = escaped.unwrap();
    // `h1` は `Op::Relu`（`Op::is_checkpoint_eligible()` が `false`
    // を返す非対象 Op）であり checkpoint による解放対象ではない。
    // 未実体化なのは checkpoint とは独立の理由（elementwise の遅延
    // グラフ末端。`push_lazy`）であり、`value()`（層 2）が
    // `materialize_infallible` 経由で通常どおり実体化するだけである
    // （review 指摘: #1624 のコメント記述の訂正）。
    let recomputed = h1.to_tensor();

    // 独立に同じ forward を計算し、期待値と突き合わせる。
    let expected_tape = Tape::new_with_ops(common::naive_ops());
    let expected = expected_tape
        .var(&x)
        .matmul(&expected_tape.var(&w1))
        .unwrap()
        .relu()
        .to_tensor();

    assert_bit_identical(&recomputed, &expected);

    // ついでに h2 側の勾配も一応取得できることを確認する。
    let loss = h2.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&xv).unwrap().is_some());
}

// --- (12) transpose/reshape を含む区間（view 解放 → 基底再計算） ---

/// (12) 用の計算本体（自由関数化の理由は (10) と同じ）。
fn view_ops_compute<'t>(
    xv: &fandhe_ai_autodiff::Var<'t>,
    wv: &fandhe_ai_autodiff::Var<'t>,
) -> Result<fandhe_ai_autodiff::Var<'t>, AutodiffError> {
    // `transpose` の view は非連続なため `reshape` へ直接連鎖できない
    // （`Tensor::reshape` の contiguous 前提。`ShapeError::
    // NonContiguousReshape`）。二重 transpose（対合性で元 shape へ戻る
    // view チェーン）に留め、view の再帰再計算経路（`Op::Transpose` の
    // 入力側再帰）のみを exercise する。
    let xt = xv.transpose(0, 1)?; // [3, 2]
    let wt = wv.transpose(0, 1)?.transpose(0, 1)?; // [2, 3] へ戻る二重 view
    let y = xt.matmul(&wt)?; // [3, 2] @ [2, 3] -> [3, 3]
    Ok(y.sigmoid())
}

#[test]
fn checkpoint_region_with_view_ops_matches_no_checkpoint() {
    let x = t(vec![1.0, -0.5, 0.3, 2.0, -1.0, 0.4], &[2, 3]);
    let w = t(vec![0.2, -0.3, 0.5, 0.1, -0.4, 0.6], &[2, 3]);

    let run = |use_checkpoint: bool| -> Tensor<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let wv = tape.var(&w);
        let out = if use_checkpoint {
            tape.checkpoint(|| view_ops_compute(&xv, &wv)).unwrap()
        } else {
            view_ops_compute(&xv, &wv).unwrap()
        };
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        grads.get(&xv).unwrap().cloned().unwrap()
    };

    let dx_plain = run(false);
    let dx_ckpt = run(true);
    assert_bit_identical(&dx_plain, &dx_ckpt);
}

// --- (13) Var::checkpoint_from と Tape::checkpoint の結果一致 ---

#[test]
fn checkpoint_from_matches_closure_based_checkpoint() {
    let (x, w1, w2, target) = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv = tape_a.var(&x);
    let w1v = tape_a.var(&w1);
    let w2v = tape_a.var(&w2);
    let tv = tape_a.var(&target);
    let h2 = tape_a
        .checkpoint(|| {
            let h1 = xv.matmul(&w1v)?.relu();
            Ok(h1.matmul(&w2v)?.sigmoid())
        })
        .unwrap();
    let loss_a = h2.mse_loss(&tv).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dx_a = grads_a.get(&xv).unwrap().cloned().unwrap();

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv = tape_b.var(&x);
    let w1v = tape_b.var(&w1);
    let w2v = tape_b.var(&w2);
    let tv = tape_b.var(&target);
    let h1 = xv.matmul(&w1v).unwrap().relu();
    let h2 = h1.matmul(&w2v).unwrap().sigmoid();
    let h2 = h2.checkpoint_from(&[&xv, &w1v, &w2v]).unwrap();
    let loss_b = h2.mse_loss(&tv).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dx_b = grads_b.get(&xv).unwrap().cloned().unwrap();

    assert_bit_identical(&dx_a, &dx_b);
}

// --- (14) 非適格 Op（qr）を含む区間でもエラーにならず勾配が変わらない ---

/// (14) 用の計算本体（自由関数化の理由は (10) と同じ）。
fn ineligible_op_compute<'t>(
    av: &fandhe_ai_autodiff::Var<'t>,
    wv: &fandhe_ai_autodiff::Var<'t>,
) -> Result<fandhe_ai_autodiff::Var<'t>, AutodiffError> {
    let qr = av.qr()?;
    Ok(qr.q.matmul(wv)?.sigmoid())
}

#[test]
fn checkpoint_region_containing_ineligible_op_does_not_error() {
    // 正方行列を渡し、qr の Q を後続演算に使う（`Op::QrQ`/`Op::QrR` は
    // `is_checkpoint_eligible() == false`。§8「非適格 Op」参照）。
    let a = t(vec![1.0, 0.5, 0.2, 1.0], &[2, 2]);
    let w = t(vec![0.3, -0.2, 0.4, 0.1], &[2, 2]);

    let run = |use_checkpoint: bool| -> Tensor<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let av = tape.var(&a);
        let wv = tape.var(&w);
        let out = if use_checkpoint {
            tape.checkpoint(|| ineligible_op_compute(&av, &wv)).unwrap()
        } else {
            ineligible_op_compute(&av, &wv).unwrap()
        };
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        grads.get(&av).unwrap().cloned().unwrap()
    };

    let dx_plain = run(false);
    let dx_ckpt = run(true);
    assert_bit_identical(&dx_plain, &dx_ckpt);
}
