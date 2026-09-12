//! イシュー #1637: `BackendOps::where_cond`／`masked_fill`（`torch.where`／
//! `torch.masked_fill` 相当）の CPU-Metal 数値一致検証（CUDA 側
//! `backend-cuda::tests::where_masked_fill_parity` の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`gemm_bias_act_parity.rs`〈#605〉と同方針。`#![cfg(target_os =
//! "macos")]` により Linux CI ではコンパイル対象外になり、`#[ignore]`
//! により通常の `cargo test` からも除外される）。
//!
//! 判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。選択演算は丸めを伴わない
//! ため 3 バックエンドとも bit 同一になる見込みであり、`assert_parity`
//! （REQ-2 複合判定）に加え bit 同一（より強い検証）も併記する。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test where_masked_fill_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `Xorshift64Star::fill_vec`（`[-1, 1)` の一様乱数）を再利用し、0.5 未満
/// を真とみなす f32 マスク（`c != 0.0` 契約に合わせた 0.0/1.0 値）を
/// 決定的シードで作る（CUDA 側テストと同一関数）。
fn cond_mask(seed: u64, numel: usize) -> Vec<f32> {
    Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| if v < 0.5 { 1.0 } else { 0.0 })
        .collect()
}

fn assert_where_parity(seed_cond: u64, seed_a: u64, seed_b: u64, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let cond = Tensor::new(cond_mask(seed_cond, numel), shape).expect("valid tensor");
    let a = Tensor::new(Xorshift64Star::new(seed_a).fill_vec(numel), shape).expect("valid tensor");
    let b = Tensor::new(Xorshift64Star::new(seed_b).fill_vec(numel), shape).expect("valid tensor");

    let cpu_result = cpu
        .where_cond(&cond, &a, &b)
        .expect("cpu where_cond always succeeds");
    let metal_result = metal
        .where_cond(&cond, &a, &b)
        .expect("metal where_cond must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("where_cond cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "where_cond: 選択演算は丸めを伴わないため bit 同一のはず（shape={shape:?}）"
    );
}

fn assert_masked_fill_parity(seed_x: u64, seed_mask: u64, shape: &[usize], value: f32) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let x = Tensor::new(Xorshift64Star::new(seed_x).fill_vec(numel), shape).expect("valid tensor");
    let mask = Tensor::new(cond_mask(seed_mask, numel), shape).expect("valid tensor");

    let cpu_result = cpu
        .masked_fill(&x, &mask, value)
        .expect("cpu masked_fill always succeeds");
    let metal_result = metal
        .masked_fill(&x, &mask, value)
        .expect("metal masked_fill must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("masked_fill cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "masked_fill: 選択演算は丸めを伴わないため bit 同一のはず（shape={shape:?}）"
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`PARALLEL_THRESHOLD`
/// （CPU 側 rayon 閾値）境界前後・broadcast なしの各形状を検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn where_masked_fill_matches_cpu_across_shapes() {
    let shapes: &[&[usize]] = &[
        &[1],
        &[4],
        &[3, 5],
        &[2, 3, 4],
        &[1 << 15], // PARALLEL_THRESHOLD ちょうど
        &[(1 << 15) - 1],
        &[(1 << 15) + 1],
    ];
    let mut seed = 9000u64;
    for &shape in shapes {
        seed += 3;
        assert_where_parity(seed, seed + 1, seed + 2, shape);
        assert_masked_fill_parity(seed, seed + 1, shape, -1.5);
    }
}

/// shape 不一致は `BackendError::ShapeMismatch` を返す（実装側の
/// 再検査。`.claude/rules/security.md` A08）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn where_cond_shape_mismatch_is_rejected() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let bad_cond = Tensor::new(vec![1.0, 0.0], &[2]).expect("valid tensor");
    let err = metal
        .where_cond(&bad_cond, &a, &b)
        .expect_err("shape mismatch must be rejected");
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// NaN／±inf 通過（選択されなかった側の値が伝播しないことを確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn where_forward_isolates_nan_and_inf_to_unselected_side() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let cond = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], &[4]).expect("valid tensor");
    let a = Tensor::new(vec![1.0, f32::NAN, f32::INFINITY, 4.0], &[4]).expect("valid tensor");
    let b = Tensor::new(vec![f32::NAN, 20.0, 30.0, f32::NEG_INFINITY], &[4]).expect("valid tensor");
    let cpu_result = cpu
        .where_cond(&cond, &a, &b)
        .expect("cpu where_cond always succeeds");
    let metal_result = metal
        .where_cond(&cond, &a, &b)
        .expect("metal where_cond must succeed");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    for (i, (&c, &g)) in cpu_slice.iter().zip(metal_slice.iter()).enumerate() {
        assert_eq!(
            c.to_bits(),
            g.to_bits(),
            "where_cond NaN/inf 通過が cpu/metal で一致しない（index {i}）"
        );
    }
}
