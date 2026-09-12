//! activation checkpointing（イシュー #1624・`docs/
//! autodiff-checkpoint-design.md`）の GPU バックエンド実機検証。
//!
//! `Var::checkpoint_from`（`crates/autodiff/src/var.rs`。facade へは
//! 既存の `Var` 再エクスポート経由でのみ到達し、facade `Tape` newtype
//! への新規 `pub fn` は追加しない。承認事項として設計 doc に記載済み）が
//! `crates/autodiff/tests/checkpoint.rs` で確認済みの「checkpoint 有無で
//! 勾配が bit 同一」という契約を、CPU 参照実装ではなく実バックエンド
//! （Metal の `BackendOps::gemm`／`eval::sigmoid`）に対しても成立させる
//! ことを実機で検証する。`recompute_fallible`（`tape.rs`）は forward
//! 計算時と同じ `ops.gemm`／`ops.sum`／`ops.max`／`eval::sigmoid` 呼び出し
//! を再現するため、理論上はどのバックエンドでも bit 同一になるはず
//! （`docs/autodiff-checkpoint-design.md` §3.4「forward と bit 同一で
//! ある理由」）——本テストはこの理論を Metal 実機で直接確認する。
//!
//! **実機ゲーティング**: `cfg(target_os = "macos")` は非 macOS の CI
//! での対象除外にしかならず実機の有無までは保証しないため
//! （`.claude/rules/ci.md`「実機依存」節）、`#[ignore]` で通常 CI から
//! 分離する（`device_param_store_metal_mixed_shape_grad.rs` と同型）。
//!
//! **CUDA は本エージェント実行環境に実機到達手段がないため未実測**
//! （本ファイルには含めず、設計 doc の「実装記録」節に明記する。
//! `docs/autodiff-checkpoint-design.md` §8 スコープ外事項参照）。
//!
//! 実行コマンド（Apple Silicon 実機）:
//!
//! ```sh
//! cargo test -p fandhe-ai --test checkpoint_backend_bit_identity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use fandhe_ai::{AutodiffError, Device, Tensor, Var};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    let c = t.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn assert_bit_identical(actual: &Tensor<f32>, expected: &Tensor<f32>) {
    let a = dense(actual);
    let e = dense(expected);
    assert_eq!(a.len(), e.len(), "shape mismatch in bit-identity check");
    for (i, (x, y)) in a.iter().zip(e.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "element {i} differs on Metal: {x} (bits {:x}) vs {y} (bits {:x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

/// `matmul → relu → matmul → sigmoid` の 2 層チェーン。checkpoint
/// 区間は `h1`（1 層目 relu 出力）から `h2`（2 層目 sigmoid 出力）まで
/// （`Var::checkpoint_from(&[&x, &w1, &w2])`）。
fn two_layer_grad(
    device: Device,
    x: &Tensor<f32>,
    w1: &Tensor<f32>,
    w2: &Tensor<f32>,
    use_checkpoint: bool,
) -> Result<Tensor<f32>, AutodiffError> {
    let tape = fandhe_ai::tape_for(device).expect("Metal 実機が利用可能なはず");
    let xv = tape.var(x);
    let w1v = tape.var(w1);
    let w2v = tape.var(w2);

    let h1 = xv.matmul(&w1v)?.relu();
    let h2 = h1.matmul(&w2v)?.sigmoid();
    let h2: Var<'_> = if use_checkpoint {
        h2.checkpoint_from(&[&xv, &w1v, &w2v])?
    } else {
        h2
    };
    // `Var::sum`／`max` は Metal 未実装（`MetalBackendOps::sum`／`max`
    // が `Unsupported` を返す。層 1 `materialize_fallible` はこれを
    // フォールバックせず伝播する契約のため checkpoint 有無を問わず
    // 失敗する）。全バックエンド対応の `mse_loss`（ゼロ target）を
    // スカラー化に使う（`crate::var::Var::sum` doc・`backend-metal::
    // ops.rs::sum` doc 参照）。
    let zeros = tape.var(&tensor(vec![0.0; 4], &[2, 2]));
    let loss = h2.mse_loss(&zeros)?;
    let grads = tape.backward(&loss)?;
    Ok(grads
        .get(&xv)?
        .cloned()
        .expect("x は loss に到達するため勾配が存在するはず"))
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn metal_checkpoint_grad_matches_no_checkpoint_bit_exact() {
    let x = tensor(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]);
    let w1 = tensor(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]);
    let w2 = tensor(vec![0.3, -0.4, 0.7, -0.1], &[2, 2]);

    let dx_plain = two_layer_grad(Device::Metal, &x, &w1, &w2, false)
        .expect("checkpoint なしの backward は成功するはず");
    let dx_ckpt = two_layer_grad(Device::Metal, &x, &w1, &w2, true)
        .expect("checkpoint ありの backward は成功するはず");

    assert_bit_identical(&dx_ckpt, &dx_plain);
}
