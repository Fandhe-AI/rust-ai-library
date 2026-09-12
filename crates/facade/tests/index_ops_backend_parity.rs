//! `where_cond`／`masked_fill`（イシュー #1637）・`narrow`（#1598／
//! #1680 で実装済みだが 3 バックエンド parity テストが `crates/facade/
//! tests/` に存在しなかった分の受入補完）の facade 到達経路（既存
//! `Var` 再エクスポート経由。新規 `pub use`／`pub fn` は追加していない）
//! の受け入れ条件対応テスト（`shape_ops_backend_parity.rs`・
//! `softmax_backend_parity.rs` と同型）。
//!
//! `where_cond`／`masked_fill` 自体は `BackendOps` を経由する非融合
//! ノード（`Op::Where`／`Op::MaskedFill`。`push_eager`）のため、CPU／
//! CUDA／Metal の 3 バックエンドとも直接カーネルの parity を検証する。
//! `narrow` はホスト view（`Tensor::narrow` の stride 再解釈。
//! `BackendOps` を経由しない）のため、下流演算（`matmul`／`add`）が
//! view を正しく消費することを検証する（`narrow` 自体の 3 バックエンド
//! parity 受入条件はこれで担保する。実装計画 §3 参照）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward／
//!   backward を REQ-2 統一複合判定で突き合わせる。選択演算
//!   （`where_cond`／`masked_fill`）は丸めを伴わないため bit 同一も
//!   併記する。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する（`narrow → matmul` は GEMM カーネルが異なるため
//!   `assert_parity` のみ・`where_cond`／`masked_fill` は bit 同一も
//!   併記する）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`shape_ops_backend_parity.rs`
/// の `VarSource` と同じ理由・同じ構成）。
trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

/// `Xorshift64Star::fill_vec`（`[-1, 1)` の一様乱数）から 0.5 未満を真と
/// みなす bool テンソルを作る（`Var::where_cond`／`masked_fill` の
/// `cond`／`mask` 引数用。`backend-cuda::tests::where_masked_fill_parity`
/// と同じ生成方針）。
fn leaf_bool(seed: u64, shape: &[usize]) -> Tensor<bool> {
    let numel: usize = shape.iter().product();
    let data: Vec<bool> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v < 0.5)
        .collect();
    Tensor::new(data, shape).expect("leaf_bool: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- where_cond（属性なし: CPU vs NaiveOps） ---

/// `where_cond` forward の CPU（`BackendOps::where_cond`）と NaiveOps
/// （ホスト `eval::where_cond` 参照実装）の parity。選択演算は丸めを
/// 伴わないため bit 同一も確認する。
#[test]
fn cpu_where_cond_forward_matches_naive_reference() {
    let shape = [2usize, 3];
    let cond = leaf_bool(1, &shape);

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(2, &shape));
    let b_cpu = cpu_tape.make_var(&leaf(3, &shape));
    let out_cpu = Var::where_cond(&cond, &a_cpu, &b_cpu)
        .expect("where_cond: 同 shape のため常に成功する")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(2, &shape));
    let b_naive = naive_tape.make_var(&leaf(3, &shape));
    let out_naive = Var::where_cond(&cond, &a_naive, &b_naive)
        .expect("where_cond: 同 shape のため常に成功する")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::where_cond）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "where_cond: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}

/// `where_cond` backward（`da`／`db`）の CPU と NaiveOps の parity。
#[test]
fn cpu_where_cond_backward_matches_naive_reference() {
    let shape = [2usize, 3];
    let cond = leaf_bool(1, &shape);

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(2, &shape));
    let b_cpu = cpu_tape.make_var(&leaf(3, &shape));
    let out_cpu = Var::where_cond(&cond, &a_cpu, &b_cpu).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let da_cpu = grads_cpu.get(&a_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(2, &shape));
    let b_naive = naive_tape.make_var(&leaf(3, &shape));
    let out_naive = Var::where_cond(&cond, &a_naive, &b_naive).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let da_naive = grads_naive.get(&a_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_naive).unwrap().expect("到達する");

    assert_parity(
        "where_cond backward（da）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(da_cpu),
        &contiguous_slice(da_naive),
    );
    assert_parity(
        "where_cond backward（db）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(db_cpu),
        &contiguous_slice(db_naive),
    );
}

// --- masked_fill（属性なし: CPU vs NaiveOps） ---

/// `masked_fill` forward・backward の CPU と NaiveOps の parity。
#[test]
fn cpu_masked_fill_matches_naive_reference() {
    let shape = [2usize, 3];
    let mask = leaf_bool(4, &shape);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(5, &shape));
    let out_cpu = x_cpu.masked_fill(&mask, -3.5).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let forward_cpu = out_cpu.to_tensor();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(5, &shape));
    let out_naive = x_naive.masked_fill(&mask, -3.5).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let forward_naive = out_naive.to_tensor();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    let cpu_fwd_slice = contiguous_slice(&forward_cpu);
    let naive_fwd_slice = contiguous_slice(&forward_naive);
    assert_parity(
        "masked_fill forward: CpuBackendOps vs NaiveOps",
        &cpu_fwd_slice,
        &naive_fwd_slice,
    );
    assert_eq!(
        cpu_fwd_slice, naive_fwd_slice,
        "masked_fill forward: 選択演算は丸めを伴わないため bit 同一のはず"
    );
    assert_parity(
        "masked_fill backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
}

// --- narrow → matmul／narrow → add（属性なし: CPU vs NaiveOps） ---

/// `narrow → matmul` forward・backward の CPU と NaiveOps の parity
/// （`narrow` はホスト view のため `BackendOps` を経由しないが、下流の
/// GEMM 経路〈`Var::matmul`〉が正しく view を消費することを確認する）。
#[test]
fn cpu_narrow_matmul_matches_naive_reference() {
    let x_shape = [4usize, 3];
    let w_shape = [3usize, 2];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(6, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(7, &w_shape));
    let y_cpu = x_cpu
        .narrow(0, 1, 2)
        .expect("narrow(0,1,2) は [4,3] に対し常に成功する")
        .matmul(&w_cpu)
        .expect("matmul: [2,3] x [3,2] は形状適合");
    let loss_cpu = y_cpu.sum(None).unwrap();
    let forward_cpu = y_cpu.to_tensor();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(6, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(7, &w_shape));
    let y_naive = x_naive.narrow(0, 1, 2).unwrap().matmul(&w_naive).unwrap();
    let loss_naive = y_naive.sum(None).unwrap();
    let forward_naive = y_naive.to_tensor();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");

    assert_parity(
        "narrow→matmul forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&forward_cpu),
        &contiguous_slice(&forward_naive),
    );
    assert_parity(
        "narrow→matmul backward（dW）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
}

/// `narrow → add` forward・backward の CPU と NaiveOps の parity。
#[test]
fn cpu_narrow_add_matches_naive_reference() {
    let x_shape = [4usize];
    let y_shape = [2usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(8, &x_shape));
    let y_cpu = cpu_tape.make_var(&leaf(9, &y_shape));
    let z_cpu = x_cpu.narrow(0, 1, 2).unwrap().add(&y_cpu).unwrap();
    let loss_cpu = z_cpu.sum(None).unwrap();
    let forward_cpu = z_cpu.to_tensor();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(8, &x_shape));
    let y_naive = naive_tape.make_var(&leaf(9, &y_shape));
    let z_naive = x_naive.narrow(0, 1, 2).unwrap().add(&y_naive).unwrap();
    let loss_naive = z_naive.sum(None).unwrap();
    let forward_naive = z_naive.to_tensor();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_parity(
        "narrow→add forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&forward_cpu),
        &contiguous_slice(&forward_naive),
    );
    assert_parity(
        "narrow→add backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn where_cond_forward_on(device: Device, cond: &Tensor<bool>) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.make_var(&leaf(2, &[2, 3]));
    let b = tape.make_var(&leaf(3, &[2, 3]));
    Var::where_cond(cond, &a, &b)
        .expect("where_cond: 同 shape のため常に成功する")
        .to_tensor()
}

fn masked_fill_forward_on(device: Device, mask: &Tensor<bool>) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(5, &[2, 3]));
    x.masked_fill(mask, -3.5)
        .expect("masked_fill: 同 shape のため常に成功する")
        .to_tensor()
}

fn narrow_matmul_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(6, &[4, 3]));
    let w = tape.make_var(&leaf(7, &[3, 2]));
    x.narrow(0, 1, 2)
        .expect("narrow(0,1,2) は [4,3] に対し常に成功する")
        .matmul(&w)
        .expect("matmul: [2,3] x [3,2] は形状適合")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`shape_ops_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_where_cond_forward_matches_cpu() {
    let cond = leaf_bool(1, &[2, 3]);
    let metal_out = where_cond_forward_on(Device::Metal, &cond);
    let cpu_out = where_cond_forward_on(Device::Cpu, &cond);

    let metal_slice = contiguous_slice(&metal_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "where_cond forward: Metal tape_for vs CPU tape_for",
        &metal_slice,
        &cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "where_cond: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_masked_fill_forward_matches_cpu() {
    let mask = leaf_bool(4, &[2, 3]);
    let metal_out = masked_fill_forward_on(Device::Metal, &mask);
    let cpu_out = masked_fill_forward_on(Device::Cpu, &mask);

    let metal_slice = contiguous_slice(&metal_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "masked_fill forward: Metal tape_for vs CPU tape_for",
        &metal_slice,
        &cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "masked_fill: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_narrow_matmul_forward_matches_cpu() {
    let metal_out = narrow_matmul_forward_on(Device::Metal);
    let cpu_out = narrow_matmul_forward_on(Device::Cpu);

    assert_parity(
        "narrow→matmul forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_where_cond_forward_matches_cpu() {
    let cond = leaf_bool(1, &[2, 3]);
    let cuda_out = where_cond_forward_on(Device::Cuda(0), &cond);
    let cpu_out = where_cond_forward_on(Device::Cpu, &cond);

    let cuda_slice = contiguous_slice(&cuda_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "where_cond forward: CUDA tape_for vs CPU tape_for",
        &cuda_slice,
        &cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "where_cond: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_masked_fill_forward_matches_cpu() {
    let mask = leaf_bool(4, &[2, 3]);
    let cuda_out = masked_fill_forward_on(Device::Cuda(0), &mask);
    let cpu_out = masked_fill_forward_on(Device::Cpu, &mask);

    let cuda_slice = contiguous_slice(&cuda_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "masked_fill forward: CUDA tape_for vs CPU tape_for",
        &cuda_slice,
        &cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "masked_fill: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_narrow_matmul_forward_matches_cpu() {
    let cuda_out = narrow_matmul_forward_on(Device::Cuda(0));
    let cpu_out = narrow_matmul_forward_on(Device::Cpu);

    assert_parity(
        "narrow→matmul forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}
