//! イシュー #1637: `BackendOps::where_cond`／`masked_fill`（`torch.where`／
//! `torch.masked_fill` 相当）の CPU-CUDA 数値一致検証。
//!
//! `gemm_bias_act_parity.rs`（#599）と同じ構成方針を踏襲する: 環境適応
//! スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ検証）
//! と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。
//! 判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。選択演算は丸めを伴わない
//! ため 3 バックエンドとも bit 同一になる見込みであり、`assert_parity`
//! （REQ-2 複合判定）に加え bit 同一（より強い検証）も併記する。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test where_masked_fill_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

mod common;

/// `Xorshift64Star::fill_vec`（`[-1, 1)` の一様乱数）を再利用し、0.5 未満
/// を真とみなす f32 マスク（`c != 0.0` 契約に合わせた 0.0/1.0 値。真偽の
/// 分布が偏っていても正しさの検証には影響しない）を決定的シードで作る。
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
    let cuda = CudaBackendOps::new(0);

    let cond = Tensor::new(cond_mask(seed_cond, numel), shape).expect("valid tensor");
    let a = Tensor::new(Xorshift64Star::new(seed_a).fill_vec(numel), shape).expect("valid tensor");
    let b = Tensor::new(Xorshift64Star::new(seed_b).fill_vec(numel), shape).expect("valid tensor");

    let cpu_result = cpu
        .where_cond(&cond, &a, &b)
        .expect("cpu where_cond always succeeds");
    let cuda_result = cuda
        .where_cond(&cond, &a, &b)
        .expect("cuda where_cond must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("where_cond cpu-cuda parity shape={shape:?}"),
        cuda_slice,
        cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "where_cond: 選択演算は丸めを伴わないため bit 同一のはず（shape={shape:?}）"
    );
}

fn assert_masked_fill_parity(seed_x: u64, seed_mask: u64, shape: &[usize], value: f32) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let x = Tensor::new(Xorshift64Star::new(seed_x).fill_vec(numel), shape).expect("valid tensor");
    let mask = Tensor::new(cond_mask(seed_mask, numel), shape).expect("valid tensor");

    let cpu_result = cpu
        .masked_fill(&x, &mask, value)
        .expect("cpu masked_fill always succeeds");
    let cuda_result = cuda
        .masked_fill(&x, &mask, value)
        .expect("cuda masked_fill must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("masked_fill cpu-cuda parity shape={shape:?}"),
        cuda_slice,
        cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "masked_fill: 選択演算は丸めを伴わないため bit 同一のはず（shape={shape:?}）"
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`gemm_bias_act_parity.rs::gemm_bias_act_parity_smoke_env_adaptive`
/// と同じ分岐パターン）。実機なら形状網羅ケースまで実行する。
#[test]
fn where_masked_fill_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cond = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]).expect("valid tensor");
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).expect("valid tensor");

    match cuda.where_cond(&cond, &a, &b) {
        Ok(_) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_where_parity(701, 702, 703, &[4]);
            assert_where_parity(704, 705, 706, &[3, 5]);
            assert_masked_fill_parity(707, 708, &[4], -9.0);
            assert_masked_fill_parity(709, 710, &[3, 5], 0.0);

            // shape 不一致は `BackendError::ShapeMismatch` を返す（実装
            // 側の再検査。`.claude/rules/security.md` A08）。
            let bad_cond = Tensor::new(vec![1.0, 0.0], &[2]).expect("valid tensor");
            let err = cuda
                .where_cond(&bad_cond, &a, &b)
                .expect_err("shape mismatch must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::where_cond: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`PARALLEL_THRESHOLD`
/// （CPU 側 rayon 閾値。イシュー #1637 は CUDA 側に自動並列判定を持た
/// ないが CPU 参照実装との突合形状として網羅する）境界前後・broadcast
/// なしの各形状・NaN／±inf 通過を検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn where_masked_fill_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let shapes: &[&[usize]] = &[
        &[1],
        &[4],
        &[3, 5],
        &[2, 3, 4],
        &[1 << 15], // PARALLEL_THRESHOLD ちょうど
        &[(1 << 15) - 1],
        &[(1 << 15) + 1],
    ];
    let mut seed = 8000u64;
    for &shape in shapes {
        seed += 3;
        assert_where_parity(seed, seed + 1, seed + 2, shape);
        assert_masked_fill_parity(seed, seed + 1, shape, -1.5);
    }

    // NaN／±inf 通過（選択されなかった側の値が伝播しないことを確認）。
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let cond = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], &[4]).expect("valid tensor");
    let a = Tensor::new(vec![1.0, f32::NAN, f32::INFINITY, 4.0], &[4]).expect("valid tensor");
    let b = Tensor::new(vec![f32::NAN, 20.0, 30.0, f32::NEG_INFINITY], &[4]).expect("valid tensor");
    let cpu_result = cpu
        .where_cond(&cond, &a, &b)
        .expect("cpu where_cond always succeeds");
    let cuda_result = cuda
        .where_cond(&cond, &a, &b)
        .expect("cuda where_cond must succeed");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    for (i, (&c, &g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
        assert_eq!(
            c.to_bits(),
            g.to_bits(),
            "where_cond NaN/inf 通過が cpu/cuda で一致しない（index {i}）"
        );
    }
}
