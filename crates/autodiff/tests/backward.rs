//! 受け入れ条件「合成関数の end-to-end 勾配が期待値と一致する」を直接
//! 検証する統合テスト（TASK-1.5c・イシュー #18）。
//!
//! - 手計算できる小さな合成関数（`mul` の自己参照・`add`）で解析解と
//!   厳密一致を確認する（`mul` は勾配蓄積〈複数経路からの合算〉の検証を
//!   兼ねる）。
//! - MLP 1 層相当（`matmul → add(bias) → relu → mse_loss`）の合成関数で
//!   `Tape::backward` の解析勾配と中央差分（数値微分）を突合する。判定
//!   閾値は `grad.rs` の grad-check テスト（#17）が用いた値をそのまま
//!   再利用し、新しい許容誤差は導入しない（`H=1e-3`・相対 1e-2 または
//!   絶対 1e-3・`τ=1e-4`。承認追跡は Issue #223）。
//! - `Tape::backward`/`Gradients::get` の API 契約（クロステープ検査・
//!   未到達ノード・境界外アクセス・非スカラー loss の暗黙総和射影）を
//!   検証する。
//!
//! `Tape`/`Var` を経由する end-to-end 経路のみを対象とし、PoC-v2-2 の
//! 実測ケース網羅・回帰テスト化は #19（TASK-1.5d）のスコープのため
//! 含めない。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. mul の自己参照（勾配蓄積の検証）: loss = sum(x * x) → dx = 2x ---

#[test]
fn mul_self_reference_accumulates_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));

    // 同一 `Var` を `mul` の両引数に渡す。x のノードへは
    // `Mul(x, x)` の VJP から dA・dB 両方の寄与が流入するため、
    // `backward()` の蓄積（合算）ロジックを直接検証する。
    let y = x.mul(&x).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");

    // d/dx sum(x*x) = 2x
    assert_eq!(dx.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(dx.get(&[0, 1]).unwrap(), -4.0);
    assert_eq!(dx.get(&[1, 0]).unwrap(), 6.0);
    assert_eq!(dx.get(&[1, 1]).unwrap(), 1.0);
}

// --- 2. add: loss = sum(a + b) → da = db = ones ---

#[test]
fn add_grad_is_ones_for_both_operands() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
    let b = tape.var(&t(vec![0.5, 1.5, -1.0], &[3]));

    let y = a.add(&b).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    let db = grads.get(&b).unwrap().expect("b は loss に到達する");

    for i in 0..3 {
        assert_eq!(da.get(&[i]).unwrap(), 1.0);
        assert_eq!(db.get(&[i]).unwrap(), 1.0);
    }
}

// --- 3. MLP 1 層相当の合成関数: 数値微分との end-to-end 突合 ---
//
// `loss = mse_loss(relu(x.matmul(w) + b), target)`。ReLU 入力は
// `|value| >= 10h`（`h = 1e-3`）の固定値でキンクを回避する（#17 と同方針）。

const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let shape = analytic.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for flat in 0..numel {
        let av = analytic.get(&index).unwrap_or(0.0);
        let nv = numeric.get(&index).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{flat:?} idx={index:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// `Tape`/`Var` を新規構築して forward を再評価し、スカラー loss 値
/// （f32）を返す。中央差分の各サンプル点でテープを 1 回使い捨てる
/// （`tape.rs` が前提とする学習ループ運用と同じパターン）。
fn forward_loss(x: &Tensor<f32>, w: &Tensor<f32>, b: &Tensor<f32>, target: &Tensor<f32>) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let bv = tape.var(b);
    let tv = tape.var(target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    scalar(&loss.to_tensor())
}

/// 指定テンソルの各要素を中央差分で摂動し、`forward_loss` に対する
/// 数値勾配を計算する。f64 で集計し丸め誤差を抑える
/// （`grad.rs` の `numeric_grad_unary` と同方針）。
fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

struct MlpFixture {
    x: Tensor<f32>,
    w: Tensor<f32>,
    b: Tensor<f32>,
    target: Tensor<f32>,
}

fn mlp_fixture() -> MlpFixture {
    // pre-relu = x @ w + b の各要素の絶対値は 0.1 以上
    // （[-0.15, -1.3, 3.25, -0.1]）。h=1e-3 の摂動で符号が変わらず
    // ReLU のキンクを踏まない。
    MlpFixture {
        x: t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]),
        w: t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]),
        b: t(vec![0.1, -0.2], &[2]),
        target: t(vec![0.1, 0.0, 3.0, 0.0], &[2, 2]),
    }
}

#[test]
fn mlp_grad_w_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.b, &f.target));
    assert_grad_close("mlp dW", dw, &num_dw);
}

#[test]
fn mlp_grad_b_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");

    let num_db = numeric_grad(&f.b, |b| forward_loss(&f.x, &f.w, &b, &f.target));
    assert_grad_close("mlp dB", db, &num_db);
}

#[test]
fn mlp_grad_x_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&f.x, |x| forward_loss(&x, &f.w, &f.b, &f.target));
    assert_grad_close("mlp dX", dx, &num_dx);
}

// --- 4. API 契約テスト ---

#[test]
fn unreachable_leaf_returns_ok_none() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    // loss に一切関与しない葉ノード。
    let unused = tape.var(&t(vec![9.0], &[1]));
    let loss = x.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&unused).unwrap().is_none());
    assert!(grads.get(&x).unwrap().is_some());
}

#[test]
fn backward_with_foreign_tape_var_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let loss_a = tape_a.var(&t(vec![1.0], &[1])).sum(None).unwrap();
    // `tape_b` から `backward` を呼びつつ、`tape_a` の loss を渡す。
    let x_b = tape_b.var(&t(vec![2.0], &[1]));
    let _ = x_b.sum(None).unwrap(); // tape_b にも何かノードを積んでおく

    let result = tape_b.backward(&loss_a);
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

#[test]
fn gradients_get_with_foreign_tape_var_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let loss_a = x_a.sum(None).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();

    let x_b = tape_b.var(&t(vec![3.0], &[1]));
    let result = grads_a.get(&x_b);
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

#[test]
fn get_for_var_added_after_backward_returns_ok_none() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = x.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    // backward 完了後に同一テープへ新規ノードを追加する
    // （`grads.grads.len()` を超える `NodeId` になる）。
    let after = tape.var(&t(vec![9.0], &[1]));
    assert!(grads.get(&after).unwrap().is_none());
}

#[test]
fn non_scalar_loss_seed_is_implicit_sum_projection() {
    // 非スカラー loss（shape [2]）は「暗黙の総和射影」
    // （`sum(loss).backward()` 相当）として扱われ、シードは全要素 1。
    // ここでは loss = x（恒等）とし、seed = ones と直接一致することを
    // 確認する（各要素が独立にそのまま出力へ伝わるため）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, -1.0], &[2]));
    let grads = tape.backward(&x).unwrap();
    let dx = grads.get(&x).unwrap().expect("x 自身が loss である");

    assert_eq!(dx.get(&[0]).unwrap(), 1.0);
    assert_eq!(dx.get(&[1]).unwrap(), 1.0);
}

// --- view 系ノード（reshape / transpose）の backward（イシュー #1047・
// 親 #1043「カーネル融合・autodiff 実行モデルの強化」） ---

/// 14. `reshape` 単体の backward: loss = sum(reshape(x, [4]))。
///     sum は形状に依存しないため dx は全要素 1（x の元 shape [2,2]）。
#[test]
fn reshape_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let r = x.reshape(&[4]).unwrap();
    let loss = r.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dx.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

/// 15. `transpose` 単体の backward: loss = sum(transpose(x, 0, 1))。
///     sum は形状・順序に依存しないため dx は全要素 1（x の元 shape）。
#[test]
fn transpose_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let tr = x.transpose(0, 1).unwrap();
    let loss = tr.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(dx.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

/// 16. `reshape → matmul`（view を fallible 演算〈matmul〉へ渡す経路）。
///     `w` を単位行列にして matmul を恒等化し、reshape 単体の VJP
///     （zero-copy `reshape` の逆写像）が正しく合成されることを
///     解析解と厳密一致で検証する。
#[test]
fn reshape_then_matmul_backward_matches_analytical() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // x: shape [4] → reshape → [2,2]
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let w = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])); // 単位行列
    let r = x.reshape(&[2, 2]).unwrap();
    let y = r.matmul(&w).unwrap(); // y == r（w が単位行列のため）
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    // d(sum(r @ I))/dr = ones[2,2] → reshape の逆写像で dx = ones[4]
    assert_eq!(dx.shape(), &[4]);
    for i in 0..4 {
        assert_eq!(dx.get(&[i]).unwrap(), 1.0);
    }
}

/// 17. `relu → transpose → add`（view が融合境界になる経路）。融合
///     （`push_lazy`）と view（`push_view`）が混在する連鎖で、非融合
///     参照実装と同じ勾配になることを解析解と突合する。
///
///     `y = relu(x)` [2,3] → `t = transpose(y, 0, 1)` [3,2] →
///     `z = t + bias`（bias: [2]、列方向 broadcast）→ `loss = sum(z)`。
///     `dz = ones[3,2]` → `dbias = reduce_to_shape(dz, [2])`（各列 3 要素
///     の総和 = 3） → `dt = ones[3,2]` → `dy = transpose(dt, 0, 1) =
///     ones[2,3]` → `dx = dy * (x > 0)`（ReLU 劣勾配）。
#[test]
fn relu_transpose_add_fusion_boundary_matches_reference() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 2.0, -3.0, 4.0, 5.0, -6.0], &[2, 3]));
    let bias = tape.var(&t(vec![10.0, 20.0], &[2]));

    let y = x.relu();
    let tr = y.transpose(0, 1).unwrap();
    let z = tr.add(&bias).unwrap();
    let loss = z.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    let dbias = grads.get(&bias).unwrap().expect("bias は loss に到達する");

    // ReLU 劣勾配: x > 0 の位置のみ 1、それ以外 0。
    let expected_dx = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    for (idx, &expected) in expected_dx.iter().enumerate() {
        let (i, j) = (idx / 3, idx % 3);
        assert_eq!(
            dx.get(&[i, j]).unwrap(),
            expected,
            "dx[{i},{j}] 不一致（ReLU 劣勾配）"
        );
    }
    // bias は [2] へ 3 要素ずつ縮約されるため、各要素は 3.0。
    assert_eq!(dbias.get(&[0]).unwrap(), 3.0);
    assert_eq!(dbias.get(&[1]).unwrap(), 3.0);
}

/// 18. view ノード自身の fan-out（同一 `reshape` 結果を `mul` の両
///     オペランドとして 2 回消費し、勾配が合算されることを検証する）。
///     `loss = sum(r * r)`（`r = reshape(x, [2,2])`）→ `dr = 2r` →
///     `dx = reshape(2r, [4]) = 2x`。
#[test]
fn view_node_fan_out_accumulates_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[4]));
    let r = x.reshape(&[2, 2]).unwrap();
    let y = r.mul(&r).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[4]);
    assert_eq!(dx.get(&[0]).unwrap(), 2.0);
    assert_eq!(dx.get(&[1]).unwrap(), -4.0);
    assert_eq!(dx.get(&[2]).unwrap(), 6.0);
    assert_eq!(dx.get(&[3]).unwrap(), 1.0);
}

// --- permute / broadcast_to / expand / squeeze / unsqueeze / flatten
// （イシュー #1597） ---

/// `Tensor<f32>` を行優先で読み出す（`broadcast_to` の stride 0 view は
/// `as_slice()` が `None` を返すため、`Var::to_tensor()` の値比較に
/// `get`／`contiguous()` 経由の本ヘルパーを使う。`tape_recording.rs`
/// の同名ヘルパーと同型）。
fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

/// 19. `permute` 単体の backward: loss = sum(permute(x, perm))。sum は
///     形状・順序に依存しないため dx は全要素 1（x の元 shape）。
#[test]
fn permute_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let p = x.permute(&[1, 0]).unwrap();
    let loss = p.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(dx.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

/// 20. `broadcast_to` 単体の backward: loss = sum(broadcast_to(x, s))。
///     dx は各出力要素が入力のどの要素に対応するかの複製回数（解析値）
///     になる（`x: [1,3] → [2,3]` は 2 回複製されるため dx = [2,2,2]）。
#[test]
fn broadcast_to_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let b = x.broadcast_to(&[2, 3]).unwrap();
    let loss = b.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[1, 3]);
    for j in 0..3 {
        assert_eq!(dx.get(&[0, j]).unwrap(), 2.0);
    }
}

/// 21. `squeeze → unsqueeze → flatten` の連鎖（すべて `reshape` への
///     委譲）の backward。sum の入力なので勾配は全要素 1（元 shape）。
#[test]
fn squeeze_unsqueeze_flatten_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]));

    let sq = x.squeeze(Some(1)).unwrap(); // [2,3]
    let u = sq.unsqueeze(0).unwrap(); // [1,2,3]
    let f = u.flatten(1, 2).unwrap(); // [1,6]
    let loss = f.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 1, 3]);
    for i in 0..2 {
        for k in 0..3 {
            assert_eq!(dx.get(&[i, 0, k]).unwrap(), 1.0);
        }
    }
}

/// 22. bit 同一 parity: `broadcast_to` を明示してから `add` した結果
///     （forward・`dx`／`dy`）が、`add` の暗黙ブロードキャストのみで
///     計算した結果と bit 同一であることを検証する（`Op::BroadcastTo`
///     の VJP と `Op::Add` の暗黙ブロードキャスト VJP が同じ
///     `reduce_bias_grad` を使うため。イシュー #1597 の parity 要件・
///     codex-review P1 是正で `reduce_to_shape` から切替済み）。
#[test]
fn broadcast_to_then_add_matches_implicit_broadcast_add_bit_exact() {
    let x_data = vec![1.0f32, -2.0, 3.0];
    let y_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];

    // 経路 A: broadcast_to を明示してから add。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&t(x_data.clone(), &[3]));
    let y_a = tape_a.var(&t(y_data.clone(), &[2, 3]));
    let bx_a = x_a.broadcast_to(&[2, 3]).unwrap();
    let z_a = bx_a.add(&y_a).unwrap();
    let loss_a = z_a.sum(None).unwrap();
    let forward_a = dense_vec(&z_a.to_tensor());
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dx_a = dense_vec(grads_a.get(&x_a).unwrap().unwrap());
    let dy_a = dense_vec(grads_a.get(&y_a).unwrap().unwrap());

    // 経路 B: add の暗黙ブロードキャストのみ。
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_b = tape_b.var(&t(x_data, &[3]));
    let y_b = tape_b.var(&t(y_data, &[2, 3]));
    let z_b = x_b.add(&y_b).unwrap();
    let loss_b = z_b.sum(None).unwrap();
    let forward_b = dense_vec(&z_b.to_tensor());
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dx_b = dense_vec(grads_b.get(&x_b).unwrap().unwrap());
    let dy_b = dense_vec(grads_b.get(&y_b).unwrap().unwrap());

    assert_eq!(forward_a, forward_b, "forward 値が bit 同一でない");
    assert_eq!(dx_a, dx_b, "dx が bit 同一でない");
    assert_eq!(dy_a, dy_b, "dy が bit 同一でない");
}

/// 22b. `Op::BroadcastTo` の VJP が `Op::Add` の暗黙ブロードキャスト
///      縮約（`reduce_bias_grad`。行方向縮約パターンは `f64`
///      アキュムレータ〈`eval::reduce_bias_grad_rows`〉経由）と同じ
///      数値契約であることを、相殺を含む上流勾配（行順
///      `[1e8, 1, -1e8]`）で検証する（codex-review P1 是正の回帰:
///      旧実装は `reduce_to_shape`〈`f32` 逐次和〉のみを使い、
///      `1e8 + 1` が `f32` 丸めで `1e8` へ吸収されたあと `-1e8` すると
///      `0.0` になってしまっていたが、`f64` 経由では `1.0` が正しい
///      解析値）。`loss = sum(broadcast_to(x, [3,1]) * c)` は
///      `dx = sum(c)`（`c` は broadcast_to の上流勾配そのものに一致
///      させるための定数）となり、`Op::Add` の行方向縮約と同一の
///      shape 構造（`g: [3,1]`・`target: [1]`）を `Op::BroadcastTo`
///      単体の VJP 経路で踏む。
#[test]
fn broadcast_to_backward_row_reduction_uses_f64_accumulator_on_cancelling_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0], &[1]));
    let c = tape.var(&t(vec![1.0e8, 1.0, -1.0e8], &[3, 1]));
    let bx = x.broadcast_to(&[3, 1]).unwrap();
    let z = bx.mul(&c).unwrap();
    let loss = z.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[1]);
    assert_eq!(
        dx.get(&[0]).unwrap(),
        1.0,
        "f64 アキュムレータ経由の解析値（1e8 + 1 - 1e8 = 1.0）と一致しない \
         （f32 逐次和のままだと 1e8 + 1 が丸めで 1e8 に吸収され 0.0 になる）"
    );
}

/// 23. bit 同一 parity: `x.permute(&[1,0])?.matmul(&w)` と
///     `x.transpose(0,1)?.matmul(&w)` の forward・`dx` が bit 同一で
///     あることを検証する（2 軸 swap の `perm` は `transpose` と同一
///     strides を生成するため。イシュー #1597 の parity 要件）。
#[test]
fn permute_then_matmul_matches_transpose_then_matmul_bit_exact() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let w_data = vec![1.0f32, -1.0, 0.5, 2.0]; // [2,2]

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&t(x_data.clone(), &[2, 3]));
    let w_a = tape_a.var(&t(w_data.clone(), &[2, 2]));
    let p_a = x_a.permute(&[1, 0]).unwrap(); // [3,2]
    let y_a = p_a.matmul(&w_a).unwrap();
    let loss_a = y_a.sum(None).unwrap();
    let forward_a = dense_vec(&y_a.to_tensor());
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dx_a = dense_vec(grads_a.get(&x_a).unwrap().unwrap());

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_b = tape_b.var(&t(x_data, &[2, 3]));
    let w_b = tape_b.var(&t(w_data, &[2, 2]));
    let tr_b = x_b.transpose(0, 1).unwrap(); // [3,2]
    let y_b = tr_b.matmul(&w_b).unwrap();
    let loss_b = y_b.sum(None).unwrap();
    let forward_b = dense_vec(&y_b.to_tensor());
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dx_b = dense_vec(grads_b.get(&x_b).unwrap().unwrap());

    assert_eq!(forward_a, forward_b, "forward 値が bit 同一でない");
    assert_eq!(dx_a, dx_b, "dx が bit 同一でない");
}

/// 24. 異常系: `permute`（perm 長不一致・範囲外・重複軸）・
///     `broadcast_to`（縮小方向・非互換 shape）・`unsqueeze`（rank+1
///     超過）・`flatten`（`start_dim > end_dim`）が
///     `AutodiffError::Shape(..)` を返すことを検証する（`var.rs` 側の
///     検査順序の end-to-end 確認）。
#[test]
fn shape_op_error_paths_return_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    assert!(matches!(
        x.permute(&[0]).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch { .. })
    ));
    assert!(matches!(
        x.permute(&[0, 0]).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::DuplicateAxis { .. })
    ));
    assert!(matches!(
        x.broadcast_to(&[3]).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::BroadcastIncompatible { .. })
    ));
    assert!(matches!(
        x.unsqueeze(3).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
    assert!(matches!(
        x.flatten(1, 0).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

// --- cat / stack / narrow / split / chunk（イシュー #1598） ---

/// 24. `cat` の backward: `loss = sum(cat([x, y], dim=1) * c)` は
///     dim=1 で連結された各入力へ `c` の対応区間がそのまま流れる
///     （解析値）。
#[test]
fn cat_backward_distributes_upstream_to_each_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![5.0, 6.0], &[2, 1]));
    let c = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let cat = fandhe_ai_autodiff::Var::cat(&[x, y], 1).unwrap();
    assert_eq!(cat.to_tensor().shape(), &[2, 3]);
    let z = cat.mul(&c).unwrap();
    let loss = z.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    let dy = grads.get(&y).unwrap().expect("y は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0, 2.0, 4.0, 5.0]);
    assert_eq!(dense_vec(dy), vec![3.0, 6.0]);
}

/// 25. `cat(&[x, x])` の fan-out 合算: 同一 `Var` を 2 回連結した場合、
///     backward が両方の寄与を合算することを確認する（`Op::Concat`
///     doc「同一 NodeId の重複は `accumulate` が合算する」）。
#[test]
fn cat_self_reference_accumulates_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[1, 2]));
    let cat = fandhe_ai_autodiff::Var::cat(&[x, x], 0).unwrap();
    let loss = cat.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![2.0, 2.0]);
}

/// 26. `stack` の backward: `loss = sum(stack([x, y], dim=0))` は
///     sum が形状・順序に依存しないため dx/dy は全要素 1。
#[test]
fn stack_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![3.0, 4.0], &[2]));
    let s = fandhe_ai_autodiff::Var::stack(&[x, y], 0).unwrap();
    assert_eq!(s.to_tensor().shape(), &[2, 2]);
    let loss = s.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    let dy = grads.get(&y).unwrap().expect("y は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0, 1.0]);
    assert_eq!(dense_vec(dy), vec![1.0, 1.0]);
}

/// 27. `split`／`chunk` の backward: 各出力を異なる係数で重み付けした
///     loss の解析勾配を検証する（各出力区間へ対応係数がそのまま
///     流れる）。
#[test]
fn split_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let parts = x.split(2, 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_tensor().shape(), &[2]);
    assert_eq!(parts[2].to_tensor().shape(), &[1]);

    // 各パートに係数 1, 2, 3 を掛けてから合算する。
    let coeffs = [1.0f32, 2.0, 3.0];
    let mut terms = Vec::with_capacity(parts.len());
    for (part, &coeff) in parts.iter().zip(coeffs.iter()) {
        let scaled = part.sum(None).unwrap();
        let c = tape.var(&t(vec![coeff], &[]));
        terms.push(scaled.mul(&c).unwrap());
    }
    let mut loss = terms[0];
    for term in &terms[1..] {
        loss = loss.add(term).unwrap();
    }

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    // part0=[1,2]*1, part1=[3,4]*2, part2=[5]*3
    assert_eq!(dense_vec(dx), vec![1.0, 1.0, 2.0, 2.0, 3.0]);
}

/// 28. `chunk` の backward（`split` と同型の検証）。
#[test]
fn chunk_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let parts = x.chunk(3, 0).unwrap();
    // ceil(5/3) = 2 -> [2, 2, 1]
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_tensor().shape(), &[2]);
    assert_eq!(parts[1].to_tensor().shape(), &[2]);
    assert_eq!(parts[2].to_tensor().shape(), &[1]);

    let loss = parts[0]
        .sum(None)
        .unwrap()
        .add(&parts[1].sum(None).unwrap())
        .unwrap()
        .add(&parts[2].sum(None).unwrap())
        .unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0, 1.0, 1.0, 1.0, 1.0]);
}

/// 29. `narrow` 単体の backward: 未選択領域の勾配は 0。
#[test]
fn narrow_backward_zeros_unselected_region() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let n = x.narrow(0, 1, 2).unwrap();
    assert_eq!(n.to_tensor().shape(), &[2]);
    let loss = n.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![0.0, 1.0, 1.0, 0.0, 0.0]);
}

/// 30. `split` → `cat` の往復: forward が bit 同一・backward が全要素
///     1（sum の入力）であることを検証する。
#[test]
fn split_then_cat_roundtrip_matches_original_bit_exact_and_backward_is_ones() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]));
    let parts = x.split(2, 0).unwrap();
    let rejoined = fandhe_ai_autodiff::Var::cat(&parts, 0).unwrap();
    assert_eq!(dense_vec(&rejoined.to_tensor()), dense_vec(&x.to_tensor()));

    let loss = rejoined.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0; 6]);
}

/// 32. `cat` 経由で loss に到達したパラメータの勾配（`dim=1` の
///     連結なので非 contiguous な narrow view）を `optim::Sgd::step`
///     に渡して 1 step 更新できることを確認する（view 勾配の消費側
///     契約の回帰）。
#[test]
fn cat_gradient_view_can_be_consumed_by_sgd_step() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![5.0, 6.0], &[2, 1]));
    let cat = fandhe_ai_autodiff::Var::cat(&[w, y], 1).unwrap();
    let loss = cat.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dw = grads
        .get(&w)
        .unwrap()
        .expect("w は loss に到達する")
        .clone();

    let mut sgd =
        fandhe_ai_autodiff::optim::Sgd::new(fandhe_ai_autodiff::optim::SgdConfig::new(0.1))
            .unwrap();
    let w_val = w.to_tensor();
    let updated = sgd
        .step(&[&w_val], &[&dw])
        .expect("view 勾配でも step が成功する");
    assert_eq!(updated[0].shape(), &[2, 2]);
}

/// 33. edge ケース: `chunk` の `shape[dim] == 0`（`chunks` 個の空
///     narrow）・`split` の `shape[dim] == 0`（1 個の空 narrow）・
///     全区間 0 長の `cat`。
#[test]
fn zero_length_edge_cases_for_split_chunk_and_cat() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let empty = tape.var(&t(Vec::new(), &[0]));

    let chunks = empty.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    for c in &chunks {
        assert_eq!(c.to_tensor().shape(), &[0]);
    }

    let splits = empty.split(4, 0).unwrap();
    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].to_tensor().shape(), &[0]);

    let cat = fandhe_ai_autodiff::Var::cat(&[empty, empty], 0).unwrap();
    assert_eq!(cat.to_tensor().shape(), &[0]);
}

// --- cat / stack / narrow / split / chunk のエラー経路 ---

#[test]
fn cat_empty_list_is_invalid_argument() {
    let err = fandhe_ai_autodiff::Var::cat(&[], 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn stack_empty_list_is_invalid_argument() {
    let err = fandhe_ai_autodiff::Var::stack(&[], 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn cat_cross_tape_is_rejected() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x = tape_a.var(&t(vec![1.0], &[1]));
    let y = tape_b.var(&t(vec![2.0], &[1]));
    let err = fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn cat_rank_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let err = fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn cat_axis_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![1.0, 2.0, 3.0], &[3, 1]));
    let err = fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn stack_axis_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let err = fandhe_ai_autodiff::Var::stack(&[x], 2).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange {
            axis: 2,
            rank: 2
        })
    ));
}

#[test]
fn stack_shape_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = fandhe_ai_autodiff::Var::stack(&[x, y], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn stack_non_contiguous_input_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let tr = x.transpose(0, 1).unwrap();
    let err = fandhe_ai_autodiff::Var::stack(&[tr], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NonContiguousReshape)
    ));
}

#[test]
fn narrow_out_of_bounds_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.narrow(0, 2, 2).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NarrowOutOfBounds { .. })
    ));
}

#[test]
fn split_zero_size_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.split(0, 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn chunk_zero_chunks_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.chunk(0, 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn split_with_sizes_sum_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.split_with_sizes(&[1, 1], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}
// --- Where／MaskedFill（イシュー #1637） ---

fn tb(data: Vec<bool>, shape: &[usize]) -> Tensor<bool> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// ①forward 値（解析）: `where_cond` が `cond` の真偽で `a`／`b` の
/// 要素を選択することを直接確認する。
#[test]
fn where_forward_selects_by_condition() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let b = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
    let cond = tb(vec![true, false, true, false], &[4]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![1.0, 20.0, 3.0, 40.0]);
}

/// ②`where_cond` の backward を中央差分と突合する（`cond` は摂動対象
/// 外の定数マスク）。
#[test]
fn where_backward_matches_numeric() {
    let cond = tb(vec![true, false, true, false], &[4]);
    let a0 = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let b0 = t(vec![10.0, 20.0, 30.0, 40.0], &[4]);

    let forward = |a: &Tensor<f32>, b: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let av = tape.var(a);
        let bv = tape.var(b);
        let out = fandhe_ai_autodiff::Var::where_cond(&cond, &av, &bv).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a0);
    let bv = tape.var(&b0);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &av, &bv).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().expect("a は loss に到達する");
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");

    let num_da = numeric_grad(&a0, |a| forward(&a, &b0));
    let num_db = numeric_grad(&b0, |b| forward(&a0, &b));
    assert_grad_close("where dA", da, &num_da);
    assert_grad_close("where dB", db, &num_db);
}

/// ③broadcast（`x:[2,3]`, `y:[3]`, `cond:[2,3]`）で `dy` が行方向へ
/// 縮約されることを確認する。
#[test]
fn where_backward_broadcast_reduces_dy_to_input_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let y = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let cond = tb(vec![true, false, true, false, true, false], &[2, 3]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &x, &y).unwrap();
    assert_eq!(out.to_tensor().shape(), &[2, 3]);
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dy = grads.get(&y).unwrap().expect("y は loss に到達する");
    assert_eq!(dy.shape(), &[3]);
    // cond=[[T,F,T],[F,T,F]] → y が選ばれる位置は (0,1)・(1,0)・(1,2)。
    // 各列で合算: col0 = row1(1) = 1.0・col1 = row0(1) = 1.0・
    // col2 = row1(1) = 1.0（upstream は sum の勾配で全要素 1）。
    assert_eq!(dense_vec(dy), vec![1.0, 1.0, 1.0]);
}

/// ④同一 `Var` を `a`／`b` 両方に指定した場合（`where(c, x, x)`）、
/// `accumulate` が合算し `dx = g`（全要素 upstream をそのまま通す）
/// ことを確認する。
#[test]
fn where_backward_same_var_both_sides_accumulates_to_upstream() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let cond = tb(vec![true, false, true, false], &[4]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &x, &x).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), dense_vec(&x.to_tensor()));
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0; 4]);
}

/// ⑤NaN が非選択側に留まることを確認する（forward 出力に NaN が
/// 現れない）。
#[test]
fn where_forward_isolates_nan_to_unselected_side() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, f32::NAN], &[2]));
    let b = tape.var(&t(vec![f32::NAN, 20.0], &[2]));
    let cond = tb(vec![true, false], &[2]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap();
    let v = dense_vec(&out.to_tensor());
    assert!(v[0].is_finite() && v[0] == 1.0);
    assert!(v[1].is_finite() && v[1] == 20.0);
}

/// ⑥エラー経路: `cond` が `a`／`b` の broadcast 後 shape へ
/// broadcast 不能なら `AutodiffError::Shape` を返す。
#[test]
fn where_cond_non_broadcastable_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let cond = tb(vec![true, false, true], &[3]);
    let err = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

/// ⑥エラー経路: 異なるテープの `Var` を `where_cond` に渡すと
/// `AutodiffError::TapeMismatch` を返す。
#[test]
fn where_cond_cross_tape_is_rejected() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape_b.var(&t(vec![3.0, 4.0], &[2]));
    let cond = tb(vec![true, false], &[2]);
    let err = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

/// `masked_fill` の forward 値を確認する。
#[test]
fn masked_fill_forward_replaces_masked_positions() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let mask = tb(vec![true, false, true, false], &[4]);
    let out = x.masked_fill(&mask, -1.0).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![-1.0, 2.0, -1.0, 4.0]);
}

/// `masked_fill` の backward を中央差分と突合する（fill 位置の勾配は
/// 0）。
#[test]
fn masked_fill_backward_matches_numeric() {
    let mask = tb(vec![true, false, true, false], &[4]);
    let x0 = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let out = xv.masked_fill(&mask, -9.0).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let out = xv.masked_fill(&mask, -9.0).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("masked_fill dX", dx, &num_dx);
    // fill 位置の勾配は厳密に 0。
    assert_eq!(dense_vec(dx)[0], 0.0);
    assert_eq!(dense_vec(dx)[2], 0.0);
}

/// `masked_fill` のエラー経路: `mask` が `self` の shape へ
/// broadcast 不能なら `AutodiffError::Shape` を返す。
#[test]
fn masked_fill_non_broadcastable_mask_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let mask = tb(vec![true, false, true], &[3]);
    let err = x.masked_fill(&mask, 0.0).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}
