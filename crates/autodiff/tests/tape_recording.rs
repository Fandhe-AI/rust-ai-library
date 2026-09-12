//! 受け入れ条件「forward 実行時にテープへ演算が記録される」を直接検証
//! する統合テスト（TASK-1.5a・イシュー #16）。
//!
//! `Tape::len()` でノード数の増加・発生順記録を、`Var::to_tensor()` で
//! 各演算の forward 値が naive 計算の期待値と一致することを確認する。
//! shape 不整合・クロステープ検査の異常系、`RefCell` 借用モデルの
//! 回帰も併せて検証する（実機依存なし。CI 実行可能）。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 1. forward 実行の連鎖（matmul → add → relu → mse_loss）で、テープ
///    ノード数が発生順に増加することを検証する（受け入れ条件の直接検証）。
#[test]
fn forward_execution_records_nodes_in_order() {
    let tape = Tape::new_with_ops(common::naive_ops());
    assert!(tape.is_empty());

    // x: [2,2], w: [2,2], b: [2] (bias broadcast), target: [2,2]
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let w = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let b = tape.var(&t(vec![10.0, 20.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]));
    // 4 leaf ノードが登録された時点でノード数は 4。
    assert_eq!(tape.len(), 4);

    let y = x.matmul(&w).unwrap(); // node 5
    assert_eq!(tape.len(), 5);
    let y = y.add(&b).unwrap(); // node 6 (bias broadcast)
    assert_eq!(tape.len(), 6);
    let y = y.relu(); // node 7
    assert_eq!(tape.len(), 7);
    let loss = y.mse_loss(&target).unwrap(); // node 8
    assert_eq!(tape.len(), 8);

    // matmul(x, identity) == x, +bias, relu (全て正値のため不変)
    let expected_y = [11.0, 22.0, 13.0, 24.0];
    let sq_sum: f32 = expected_y.iter().map(|v| v * v).sum();
    let expected_loss = sq_sum / 4.0;
    assert_eq!(loss.to_tensor().get(&[]).unwrap(), expected_loss);
}

/// 2. 各 forward 演算（matmul/add/mul/relu/exp/tanh/sum/max）の出力値が
///    期待値と一致することを確認する。
#[test]
fn forward_values_match_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    let mm = a.matmul(&b).unwrap();
    // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_eq!(
        mm.to_tensor().get(&[0, 0]).unwrap(),
        19.0,
        "matmul(0,0) 不一致"
    );
    assert_eq!(mm.to_tensor().get(&[1, 1]).unwrap(), 50.0);

    let add = a.add(&b).unwrap();
    assert_eq!(add.to_tensor().get(&[0, 0]).unwrap(), 6.0);

    let mul = a.mul(&b).unwrap();
    assert_eq!(mul.to_tensor().get(&[0, 0]).unwrap(), 5.0);
    assert_eq!(mul.to_tensor().get(&[1, 1]).unwrap(), 32.0);

    let neg = tape.var(&t(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]));
    let relu = neg.relu();
    assert_eq!(relu.to_tensor().get(&[0, 0]).unwrap(), 0.0);
    assert_eq!(relu.to_tensor().get(&[0, 1]).unwrap(), 2.0);

    let zero = tape.var(&t(vec![0.0], &[1]));
    let exp = zero.exp();
    assert_eq!(exp.to_tensor().get(&[0]).unwrap(), 1.0);

    let tanh0 = zero.tanh();
    assert_eq!(tanh0.to_tensor().get(&[0]).unwrap(), 0.0);

    let sigmoid0 = zero.sigmoid();
    assert_eq!(sigmoid0.to_tensor().get(&[0]).unwrap(), 0.5);

    let s = a.sum(None).unwrap();
    assert_eq!(s.to_tensor().get(&[]).unwrap(), 10.0);
    let s_axis0 = a.sum(Some(0)).unwrap();
    assert_eq!(s_axis0.to_tensor().shape(), &[2]);
    assert_eq!(s_axis0.to_tensor().get(&[0]).unwrap(), 4.0); // 1+3
    assert_eq!(s_axis0.to_tensor().get(&[1]).unwrap(), 6.0); // 2+4

    let m = a.max(None).unwrap();
    assert_eq!(m.to_tensor().get(&[]).unwrap(), 4.0);
}

/// 3. shape 不整合（matmul の内側次元不一致・mse_loss の shape 不一致・
///    sum の dim 範囲外）が `AutodiffError::Shape(..)` を返すことを検証する。
#[test]
fn shape_mismatches_return_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());

    let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let bad_rhs = tape.var(&t(vec![1.0, 2.0], &[2, 1]));
    let err = a.matmul(&bad_rhs).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));

    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = pred.mse_loss(&target).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));

    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let err = x.sum(Some(5)).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

/// 4. 別 `Tape` 由来の `Var` を二項演算に渡すと `TapeMismatch` を返す
///    ことを検証する（クロステープ安全性。`docs/public-api-design.md` §3.1）。
#[test]
fn cross_tape_operations_return_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());

    let a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape_b.var(&t(vec![1.0, 2.0], &[2]));

    let err = a.add(&b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));

    let err = a.mul(&b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));

    let err = a.mse_loss(&b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));

    let ma = tape_a.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let mb = tape_b.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let err = ma.matmul(&mb).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

/// 5. `to_tensor()` 呼び出し直後にノード追加演算を呼んでも panic しない
///    ことを検証する（`RefCell` 借用モデルの回帰テスト）。
#[test]
fn to_tensor_does_not_hold_borrow_across_node_addition() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let owned = a.to_tensor();
    assert_eq!(owned.get(&[0]).unwrap(), 1.0);
    // to_tensor() は Ref を持ち越さないため、直後のノード追加演算
    // （relu が RefCell::borrow_mut を呼ぶ）が panic しないことを確認する。
    let r = a.relu();
    assert_eq!(r.to_tensor().get(&[0]).unwrap(), 1.0);

    // value() 経由でも、一時 Ref をその式文の終わりで解放していれば
    // 後続のノード追加演算が panic しないことを確認する。
    let val_sum: f32 = a.value().get(&[0]).unwrap() + a.value().get(&[1]).unwrap();
    assert_eq!(val_sum, 3.0);
    let r2 = a.exp();
    assert!(r2.to_tensor().get(&[0]).unwrap() > 0.0);
}

/// 6b. `Var::sigmoid`（TASK-9.1b・#92）がテープへノードを 1 個追記する
///     ことを検証する（受け入れ条件「forward 実行時にテープへ演算が
///     記録される」の Sigmoid 個別確認）。
#[test]
fn sigmoid_records_single_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -1.0], &[2]));
    let before = tape.len();
    let _ = x.sigmoid();
    assert_eq!(tape.len(), before + 1);
}

/// 6. ブロードキャスト付き `add`（bias 加算 `[N,M] + [M]`）の forward 値と
///    記録を検証する。
#[test]
fn broadcast_add_bias_over_matrix() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let bias = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let before = tape.len();
    let y = x.add(&bias).unwrap();
    assert_eq!(tape.len(), before + 1);
    let out = y.to_tensor();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.get(&[0, 0]).unwrap(), 11.0);
    assert_eq!(out.get(&[0, 1]).unwrap(), 22.0);
    assert_eq!(out.get(&[0, 2]).unwrap(), 33.0);
    assert_eq!(out.get(&[1, 0]).unwrap(), 14.0);
    assert_eq!(out.get(&[1, 1]).unwrap(), 25.0);
    assert_eq!(out.get(&[1, 2]).unwrap(), 36.0);
}

/// 7. `Tape::default()`（codex-review 第 19〜21 波・PR #403 の P1 是正で
///    追加した compat 経路）が `Tape::new_with_ops(default_ops::naive_ops())`
///    と同じ挙動（forward 演算が記録・実行できる）を持つこと。
///
///    **TASK-9.4（#411）での位置づけ**: このテストはもともと
///    `crates/autodiff/tests/compat_sequential.rs`（`fandhe_ai_autodiff::compat`
///    経由の統合テスト）にあったが、対象は `compat::Sequential`/
///    `compat::array` ではなく `Tape::default()` 単体の挙動であるため、
///    compat 層が `fandhe_ai::compat` へ移設された後も本ファイル（`autodiff`
///    の公開 API のみを使う `Tape` 系テスト）に残置する。入力生成は
///    `compat::array`（移設先: `fandhe_ai::compat::array`）ではなく
///    `fandhe_ai_tensor_core::Tensor::new`（`autodiff` の直接の依存先）を直接使う
///    よう置き換えた。
#[test]
fn tape_default_records_and_evaluates_ops() {
    let tape = Tape::default();
    let a = tape.var(&t(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]));
    let c = a.add(&b).unwrap();

    assert_eq!(dense_vec(&c.to_tensor()), vec![6.0, 8.0, 10.0, 12.0]);

    // `sum`（非 elementwise・`BackendOps::sum` 経由の即時実行経路。
    // `default_ops::NaiveOps::sum` の `#[cfg(test)]` 解除を検証する）。
    let total = c.sum(None).unwrap();
    assert_eq!(dense_vec(&total.to_tensor()), vec![36.0]);
}

/// テスト専用の連続化ヘルパー（`fandhe_ai_tensor_core::Tensor` の `pub` API のみを
/// 使用。`crates/facade/src/compat/array.rs` のテストヘルパーと同型）。
fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

// --- view 系ノード（reshape / transpose。イシュー #1047・親 #1043） ---

/// 8. `Var::reshape`/`Var::transpose` がテープへノードを 1 個ずつ追記
///    し、`to_tensor()` の値が期待どおりであることを検証する（受け入れ
///    条件「forward 実行時にテープへ演算が記録される」の view 系個別
///    確認）。
#[test]
fn reshape_and_transpose_record_single_node_each() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let before = tape.len();

    let r = x.reshape(&[3, 2]).unwrap();
    assert_eq!(tape.len(), before + 1);
    assert_eq!(r.to_tensor().shape(), &[3, 2]);
    assert_eq!(
        dense_vec(&r.to_tensor()),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let tr = x.transpose(0, 1).unwrap();
    assert_eq!(tape.len(), before + 2);
    assert_eq!(tr.to_tensor().shape(), &[3, 2]);
    // [[1,2,3],[4,5,6]]^T == [[1,4],[2,5],[3,6]]
    assert_eq!(tr.to_tensor().get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(tr.to_tensor().get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(tr.to_tensor().get(&[2, 0]).unwrap(), 3.0);
    assert_eq!(tr.to_tensor().get(&[2, 1]).unwrap(), 6.0);
}

/// 9. `reshape`/`transpose` が zero-copy（既存バッファの `Arc` 共有）で
///    あることを実測する（イシュー #1047 の中核契約「中間バッファを
///    持たない」の直接検証。`as_view_slice().as_ptr()` の一致でバッファ
///    共有を確認する）。
#[test]
fn reshape_and_transpose_share_underlying_buffer() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let x_val = x.to_tensor();
    let x_ptr = x_val
        .as_view_slice()
        .expect("contiguous な葉テンソルは必ず as_view_slice を返す")
        .as_ptr();

    let r = x.reshape(&[4]).unwrap();
    let r_val = r.to_tensor();
    let r_ptr = r_val
        .as_view_slice()
        .expect("reshape は zero-copy のため as_view_slice を返す")
        .as_ptr();
    assert_eq!(r_ptr, x_ptr, "reshape は入力と storage を共有するはず");

    let tr = x.transpose(0, 1).unwrap();
    let tr_val = tr.to_tensor();
    // transpose 後は非 contiguous なため `as_slice()` は使えないが、
    // `as_view_slice()` は非 contiguous でも同一 storage を指す限り
    // 同じ開始アドレスを返す（`tensor-core::Tensor::as_view_slice` の
    // 実装が `offset` を起点にするため）。
    let tr_ptr = tr_val
        .as_view_slice()
        .expect("transpose は zero-copy のため as_view_slice を返す")
        .as_ptr();
    assert_eq!(tr_ptr, x_ptr, "transpose は入力と storage を共有するはず");
}

/// 10. view の view（`transpose` → `reshape` は非 contiguous のため
///     エラー、`reshape` → `transpose` は成功）を検証する
///     （`resolve_view` の再帰対応・`Tensor::reshape` の案 A〈非
///     contiguous はエラー〉との整合）。
#[test]
fn view_of_view_reshape_after_transpose_is_non_contiguous_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let tr = x.transpose(0, 1).unwrap(); // shape [3,2]・非 contiguous
    let err = tr.reshape(&[6]).unwrap_err();
    assert!(
        matches!(
            err,
            AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NonContiguousReshape)
        ),
        "非 contiguous な transpose 結果への reshape は NonContiguousReshape のはず: {err:?}"
    );
}

/// 11. `reshape` → `transpose`（先に contiguous 化する view の合成）は
///     成功し、値も期待どおりであることを検証する。
#[test]
fn view_of_view_transpose_after_reshape_succeeds() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let r = x.reshape(&[3, 2]).unwrap(); // [[1,2],[3,4],[5,6]]
    let tr = r.transpose(0, 1).unwrap(); // [[1,3,5],[2,4,6]]
    assert_eq!(tr.to_tensor().shape(), &[2, 3]);
    assert_eq!(
        dense_vec(&tr.to_tensor()),
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

/// 12. `reshape` の異常系（要素数不一致・非 contiguous 入力）が
///     `AutodiffError::Shape(..)` を返すことを検証する。
#[test]
fn reshape_shape_mismatches_return_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));

    let err = x.reshape(&[3]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ElementCountMismatch { .. })
    ));
}

/// 13. `transpose` の異常系（軸範囲外）が `AutodiffError::Shape(..)` を
///     返すことを検証する。
#[test]
fn transpose_axis_out_of_range_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));

    let err = x.transpose(0, 5).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

// --- permute / broadcast_to / expand / squeeze / unsqueeze / flatten
// （イシュー #1597） ---

/// 14. `Var::permute`/`Var::broadcast_to`/`Var::squeeze`/
///     `Var::unsqueeze`/`Var::flatten` がテープへノードを 1 個ずつ
///     追記することを検証する（受け入れ条件「forward 実行時にテープへ
///     演算が記録される」の個別確認。`squeeze`/`unsqueeze`/`flatten` は
///     `reshape` へ委譲するため記録されるノード種別は `Op::Reshape`）。
#[test]
fn shape_ops_record_single_node_each() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let before = tape.len();

    let p = x.permute(&[1, 0]).unwrap();
    assert_eq!(tape.len(), before + 1);
    assert_eq!(p.to_tensor().shape(), &[3, 2]);

    let b = x.broadcast_to(&[2, 2, 3]).unwrap();
    assert_eq!(tape.len(), before + 2);
    assert_eq!(b.to_tensor().shape(), &[2, 2, 3]);

    let sq = x.reshape(&[1, 6]).unwrap().squeeze(Some(0)).unwrap();
    assert_eq!(sq.to_tensor().shape(), &[6]);

    let u = x.unsqueeze(0).unwrap();
    assert_eq!(u.to_tensor().shape(), &[1, 2, 3]);

    let f = x.flatten(0, 1).unwrap();
    assert_eq!(f.to_tensor().shape(), &[6]);
}

/// 15. `Var::permute` が zero-copy（既存バッファの `Arc` 共有）で
///     あることを実測する（`transpose` 同種の検証。イシュー #1597）。
#[test]
fn permute_shares_underlying_buffer() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let x_val = x.to_tensor();
    let x_ptr = x_val
        .as_view_slice()
        .expect("contiguous な葉テンソルは必ず as_view_slice を返す")
        .as_ptr();

    let p = x.permute(&[1, 0]).unwrap();
    let p_val = p.to_tensor();
    let p_ptr = p_val
        .as_view_slice()
        .expect("permute は zero-copy のため as_view_slice を返す")
        .as_ptr();
    assert_eq!(p_ptr, x_ptr, "permute は入力と storage を共有するはず");
}

/// 16. `Var::permute` の forward 値が `Var::transpose(0, 1)` と一致する
///     こと（2 軸 swap の一般化として正しいこと）を検証する。
#[test]
fn permute_two_axis_swap_matches_transpose() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let p = x.permute(&[1, 0]).unwrap();
    let tr = x.transpose(0, 1).unwrap();
    assert_eq!(p.to_tensor().shape(), tr.to_tensor().shape());
    assert_eq!(dense_vec(&p.to_tensor()), dense_vec(&tr.to_tensor()));
}

/// 17. `permute` の異常系（perm 長不一致・範囲外・重複軸）が
///     `AutodiffError::Shape(..)` を返すことを検証する。
#[test]
fn permute_invalid_perm_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let err = x.permute(&[0]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch { .. })
    ));

    let err = x.permute(&[0, 5]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));

    let err = x.permute(&[0, 0]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::DuplicateAxis { .. })
    ));
}

/// 18. `broadcast_to`/`expand` の異常系（縮小方向・非互換 shape）が
///     `AutodiffError::Shape(BroadcastIncompatible)` を返すことを検証
///     する。`expand` は `broadcast_to` への薄い委譲のため同じ結果に
///     なることも併せて確認する。
#[test]
fn broadcast_to_and_expand_incompatible_shape_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));

    let err = x.broadcast_to(&[2]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::BroadcastIncompatible { .. })
    ));

    let err = x.expand(&[2]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::BroadcastIncompatible { .. })
    ));
}

/// 19. `expand` が `broadcast_to` と同じ forward 値を生むことを検証
///     する（PyTorch 名の別名であることの直接確認）。
#[test]
fn expand_matches_broadcast_to() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));

    let b = x.broadcast_to(&[2, 3]).unwrap();
    let e = x.expand(&[2, 3]).unwrap();
    assert_eq!(dense_vec(&b.to_tensor()), dense_vec(&e.to_tensor()));
}

/// 20. `squeeze(Some(d))` の PyTorch 準拠 no-op 挙動（`shape[d] != 1` の
///     場合は shape を変えずに成功する）を検証する（`Var::squeeze` doc
///     の numpy／TensorFlow との差の直接確認）。
#[test]
fn squeeze_specific_axis_not_size_one_is_noop() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let sq = x.squeeze(Some(0)).unwrap();
    assert_eq!(sq.to_tensor().shape(), &[2, 3]);
    assert_eq!(dense_vec(&sq.to_tensor()), dense_vec(&x.to_tensor()));
}

/// 21. `squeeze(None)` が長さ 1 の軸をすべて除去することを検証する。
#[test]
fn squeeze_none_removes_all_size_one_axes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3]));

    let sq = x.squeeze(None).unwrap();
    assert_eq!(sq.to_tensor().shape(), &[2, 3]);
}

/// 22. `squeeze(Some(d))` の軸範囲外エラーを検証する。
#[test]
fn squeeze_axis_out_of_range_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));

    let err = x.squeeze(Some(5)).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

/// 23. `unsqueeze` が末尾挿入（`dim == rank`）を許容し、範囲外
///     （`dim > rank`）はエラーになることを検証する。
#[test]
fn unsqueeze_allows_trailing_insertion_and_rejects_out_of_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let u = x.unsqueeze(2).unwrap();
    assert_eq!(u.to_tensor().shape(), &[2, 3, 1]);

    let err = x.unsqueeze(3).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

/// 24. `flatten` の異常系（`end_dim` 範囲外・`start_dim > end_dim`）を
///     検証する。
#[test]
fn flatten_invalid_range_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let err = x.flatten(0, 5).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));

    let err = x.flatten(1, 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

/// 24b. `flatten` が潰す軸区間の部分積オーバーフローを `checked_mul`
///      で検査し `ShapeError::ElementCountOverflow` を返すことを検証
///      する（codex-review P1 是正の回帰: ゼロ長軸を含む形状
///      `[0, usize::MAX, 2]` は総要素数自体は `0` で `Tensor::new` を
///      通過するが、`flatten(1, 2)` が潰す区間 `[usize::MAX, 2]` の
///      部分積は `checked_mul` なしでは debug panic・release ラップを
///      起こす）。
#[test]
fn flatten_partial_product_overflow_returns_element_count_overflow() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // 総要素数は 0（先頭軸が 0）のため空データで構築できる。
    let x = tape.var(&t(vec![], &[0, usize::MAX, 2]));

    let err = x.flatten(1, 2).unwrap_err();
    assert!(
        matches!(
            err,
            AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ElementCountOverflow)
        ),
        "オーバーフローを検出できていない: {err:?}"
    );
}

/// 25. `permute → flatten`（非 contiguous 化した後の `reshape` 委譲）が
///     `reshape` と同じく `NonContiguousReshape` を返すことを検証する
///     （案 A 制約の継承。`Var::flatten` doc 参照）。
#[test]
fn flatten_after_permute_is_non_contiguous_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let p = x.permute(&[1, 0]).unwrap(); // shape [3,2]・非 contiguous
    let err = p.flatten(0, 1).unwrap_err();
    assert!(
        matches!(
            err,
            AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NonContiguousReshape)
        ),
        "非 contiguous な permute 結果への flatten は NonContiguousReshape のはず: {err:?}"
    );
}

/// 26. `cat`／`stack` が 1 ノードとして記録されることを検証する
///     （イシュー #1598。`cat` はコピーを伴う `push_eager`・`stack` は
///     内部で `unsqueeze`〈`reshape` 委譲〉ノードを要素数分 push して
///     から `cat` するため、要素数 n の `stack` は n+1 ノード増える）。
#[test]
fn cat_and_stack_record_expected_node_counts() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[1, 2]));
    let y = tape.var(&t(vec![3.0, 4.0], &[1, 2]));
    let before = tape.len();

    let cat = fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap();
    assert_eq!(tape.len(), before + 1, "cat は 1 ノードのみ追加するはず");
    assert_eq!(cat.to_tensor().shape(), &[2, 2]);

    let before2 = tape.len();
    let stacked = fandhe_ai_autodiff::Var::stack(&[x, y], 0).unwrap();
    // 2 要素の unsqueeze（各 1 ノード）+ cat（1 ノード）= 3 ノード。
    assert_eq!(tape.len(), before2 + 3);
    assert_eq!(stacked.to_tensor().shape(), &[2, 1, 2]);
}

/// 27. `narrow`（および `split`／`chunk` の実体）がホスト値を持たない
///     view ノードとして記録され、入力と storage を共有する（zero-copy）
///     ことを検証する（`reshape`／`transpose` と同型の契約。イシュー
///     #1598）。
#[test]
fn narrow_records_view_node_and_shares_underlying_buffer() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let x_val = x.to_tensor();
    let x_ptr = x_val
        .as_view_slice()
        .expect("contiguous な葉テンソルは必ず as_view_slice を返す")
        .as_ptr();
    let before = tape.len();

    let n = x.narrow(0, 1, 2).unwrap();
    assert_eq!(tape.len(), before + 1, "narrow は 1 ノードのみ追加するはず");
    let n_val = n.to_tensor();
    assert_eq!(n_val.shape(), &[2]);
    assert_eq!(dense_vec(&n_val), vec![2.0, 3.0]);
    let n_ptr = n_val
        .as_view_slice()
        .expect("narrow は zero-copy のため as_view_slice を返す")
        .as_ptr();
    // narrow(0, 1, 2) は先頭から 1 要素分オフセットした view のため、
    // ポインタが `x_ptr` そのものと一致するわけではなく、同一 storage
    // 上で 1 要素分（4 バイト）だけ進んだ位置を指すはず（zero-copy の
    // 実測: 新規アロケーションであればこのオフセット関係は成立しない）。
    let expected_ptr = x_ptr.wrapping_add(1);
    assert_eq!(
        n_ptr, expected_ptr,
        "narrow は入力と storage を共有し、start 分だけオフセットされるはず"
    );
}

/// 28. `split`／`chunk` が各出力とも `narrow` へ委譲した view ノードで
///     あることを、出力個数とそれぞれの shape で検証する。
#[test]
fn split_and_chunk_record_narrow_view_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let before = tape.len();

    let splits = x.split(2, 0).unwrap();
    assert_eq!(
        tape.len(),
        before + 3,
        "split は出力個数分のノードを追加するはず"
    );
    assert_eq!(splits.len(), 3);
    assert_eq!(splits[0].to_tensor().shape(), &[2]);
    assert_eq!(splits[1].to_tensor().shape(), &[2]);
    assert_eq!(splits[2].to_tensor().shape(), &[1]);

    let before2 = tape.len();
    let chunks = x.chunk(2, 0).unwrap();
    assert_eq!(tape.len(), before2 + 2);
    assert_eq!(chunks.len(), 2);
    // ceil(5/2) = 3 -> [3, 2]
    assert_eq!(chunks[0].to_tensor().shape(), &[3]);
    assert_eq!(chunks[1].to_tensor().shape(), &[2]);
}

/// 29. `Var::cat`／`Var::stack`／`Var::narrow`／`Var::split`／
///     `Var::split_with_sizes`／`Var::chunk` のエラー経路を網羅する
///     （イシュー #1598）。
#[test]
fn cat_stack_narrow_split_chunk_error_paths() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));

    // cat: 空リスト。
    assert!(matches!(
        fandhe_ai_autodiff::Var::cat(&[], 0).unwrap_err(),
        AutodiffError::InvalidArgument(_)
    ));

    // cat: rank 不一致。
    assert!(matches!(
        fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch { .. })
    ));

    // cat: dim 範囲外。
    assert!(matches!(
        fandhe_ai_autodiff::Var::cat(&[x], 5).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));

    // stack: dim 範囲外（rank+1 まで許容）。
    assert!(matches!(
        fandhe_ai_autodiff::Var::stack(&[x], 3).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange {
            axis: 3,
            rank: 3
        })
    ));

    // narrow: dim 範囲外。
    assert!(matches!(
        x.narrow(5, 0, 1).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));

    // narrow: 範囲外（start+len 超過）。
    assert!(matches!(
        x.narrow(0, 1, 5).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NarrowOutOfBounds { .. })
    ));

    // split: split_size == 0。
    assert!(matches!(
        x.split(0, 0).unwrap_err(),
        AutodiffError::InvalidArgument(_)
    ));

    // split_with_sizes: 合計不一致。
    assert!(matches!(
        x.split_with_sizes(&[1], 0).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));

    // chunk: chunks == 0。
    assert!(matches!(
        x.chunk(0, 0).unwrap_err(),
        AutodiffError::InvalidArgument(_)
    ));
}
/// 30. `where_cond`／`masked_fill`（イシュー #1637）が 1 ノードのみ
///     追加する `push_eager`（実体化済み）ノードとして記録される
///     ことを検証する（`Op::Concat`／`Op::Softmax` と同型。view ノード
///     ではない）。
#[test]
fn where_and_masked_fill_record_single_eager_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let b = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
    let cond = Tensor::new(vec![true, false, true, false], &[4])
        .expect("test fixture: shape とデータ長は事前に一致させている");

    let before = tape.len();
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap();
    assert_eq!(
        tape.len(),
        before + 1,
        "where_cond は 1 ノードのみ追加するはず"
    );
    assert_eq!(out.to_tensor().shape(), &[4]);

    let before2 = tape.len();
    let mask = Tensor::new(vec![true, false, true, false], &[4])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let filled = a.masked_fill(&mask, -1.0).unwrap();
    assert_eq!(
        tape.len(),
        before2 + 1,
        "masked_fill は 1 ノードのみ追加するはず"
    );
    assert_eq!(filled.to_tensor().shape(), &[4]);
}
