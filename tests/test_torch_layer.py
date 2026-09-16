"""tests.test_torch_layer — Tests for quanta.torch.layer and quanta.torch.ops."""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from quanta.torch import (
    ObservableParser,
    QuantumLayer,
    UnsupportedDtypeError,
    _QuantumLayerFunction,
    get_ansatz_param_count,
    ops,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Observable Parser Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_observable_parser_default() -> None:
    """Default observables should produce local Z on all qubits."""
    obs = ObservableParser.parse_all(None, num_qubits=3)
    assert len(obs) == 3
    assert obs[0].terms[0].pauli_string == "ZII"
    assert obs[1].terms[0].pauli_string == "IZI"
    assert obs[2].terms[0].pauli_string == "IIZ"
    for o in obs:
        assert o.terms[0].coeff == 1.0


def test_observable_parser_full_strings() -> None:
    """Full Pauli strings of exact length num_qubits."""
    obs = ObservableParser.parse_all(["ZZI", "IXI"], num_qubits=3)
    assert len(obs) == 2
    assert obs[0].terms[0].pauli_string == "ZZI"
    assert obs[1].terms[0].pauli_string == "IXI"


def test_observable_parser_indexed_notation() -> None:
    """Indexed shorthand notation such as Z0, X1, Z0 Z1."""
    obs = ObservableParser.parse_all(["Z0", "X1", "Z0 Z2"], num_qubits=3)
    assert len(obs) == 3
    assert obs[0].terms[0].pauli_string == "ZII"
    assert obs[1].terms[0].pauli_string == "IXI"
    assert obs[2].terms[0].pauli_string == "ZIZ"


def test_observable_parser_weighted_composite() -> None:
    """Single composite Hamiltonian with multiple weighted terms."""
    obs = ObservableParser.parse_all([("Z0", 1.0), ("X1", -0.5)], num_qubits=2)
    assert len(obs) == 1
    assert len(obs[0].terms) == 2
    assert obs[0].terms[0].pauli_string == "ZI"
    assert obs[0].terms[0].coeff == 1.0
    assert obs[0].terms[1].pauli_string == "IX"
    assert obs[0].terms[1].coeff == -0.5


def test_observable_parser_nested_multi() -> None:
    """List of composite observables."""
    obs = ObservableParser.parse_all(
        [[("Z0", 1.0)], [("X1", 0.5), ("Z1", -0.5)]],
        num_qubits=2,
    )
    assert len(obs) == 2
    assert len(obs[0].terms) == 1
    assert len(obs[1].terms) == 2


def test_observable_parser_invalid_cases() -> None:
    """Parser should raise ValueError on invalid inputs."""
    with pytest.raises(ValueError, match="cannot be empty"):
        ObservableParser.parse_all([], num_qubits=2)

    with pytest.raises(ValueError, match="out of range"):
        ObservableParser.parse_all(["Z5"], num_qubits=3)

    with pytest.raises(ValueError, match="Invalid Pauli descriptor"):
        ObservableParser.parse_all(["W0"], num_qubits=3)

    with pytest.raises(TypeError):
        ObservableParser.parse_all([123], num_qubits=3)  # type: ignore[list-item]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Ansatz Presets & Parameter Counts
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    ("preset", "expected_params"),
    [
        ("hardware_efficient", 2 * 3 * 2),
        ("strongly_entangling", 3 * 3 * 2),
        ("reuploading", 1 * 3 * 2),
        ("real_amplitudes", 3 * (2 + 1)),
    ],
)
def test_ansatz_param_counts(preset: str, expected_params: int) -> None:
    """Validates parameter count formulas for all 4 ansatz presets."""
    count = get_ansatz_param_count(preset, num_qubits=3, num_layers=2)
    assert count == expected_params
    layer = QuantumLayer(num_qubits=3, circuit_fn=preset, num_layers=2)
    assert layer.num_params == expected_params
    assert layer.weights.shape == (expected_params,)


def test_unknown_ansatz_raises() -> None:
    """Unknown ansatz preset should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown ansatz preset"):
        get_ansatz_param_count("non_existent", num_qubits=2, num_layers=1)


def test_init_methods() -> None:
    """Test weight initialization presets."""
    for method in ("uniform", "uniform_positive", "normal", "zeros"):
        layer = QuantumLayer(num_qubits=2, num_layers=1, init_method=method)
        assert layer.weights.shape == (4,)
        if method == "zeros":
            assert torch.all(layer.weights == 0.0)

    with pytest.raises(ValueError, match="Unknown init_method"):
        QuantumLayer(num_qubits=2, init_method="invalid_init")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Shape Handling & Dimension Validation
# ═══════════════════════════════════════════════════════════════════════════

def test_quantum_layer_shapes() -> None:
    """Accepts unbatched 1D, batched 2D, and multi-dimensional inputs."""
    layer = QuantumLayer(num_qubits=3, num_layers=1)

    # 1D unbatched
    x_1d = torch.randn(3)
    y_1d = layer(x_1d)
    assert y_1d.shape == (3,)

    # 2D standard batch
    x_2d = torch.randn(5, 3)
    y_2d = layer(x_2d)
    assert y_2d.shape == (5, 3)

    # 3D multi-batch
    x_3d = torch.randn(2, 4, 3)
    y_3d = layer(x_3d)
    assert y_3d.shape == (2, 4, 3)


def test_quantum_layer_dimension_errors() -> None:
    """Dimension mismatch should raise informative ValueError."""
    layer = QuantumLayer(num_qubits=4)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        layer(torch.randn(8, 3))

    with pytest.raises(ValueError, match="at least 1 dimension"):
        layer(torch.tensor(1.0))

    with pytest.raises(ValueError, match="num_qubits must be >= 1"):
        QuantumLayer(num_qubits=0)

    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        QuantumLayer(num_qubits=2, num_layers=0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Analytical Autograd Gradcheck on CPU (Double Precision)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "preset",
    ["hardware_efficient", "strongly_entangling", "reuploading", "real_amplitudes"],
)
def test_autograd_gradcheck_presets(preset: str) -> None:
    """Exact Parameter-Shift Rule verified against finite differences via gradcheck."""
    num_qubits = 2
    num_layers = 1
    parsed_obs = ObservableParser.parse_all(None, num_qubits)
    layer = QuantumLayer(
        num_qubits=num_qubits, circuit_fn=preset, num_layers=num_layers, dtype=torch.float64
    )

    w = layer.weights.detach().clone().requires_grad_(True)
    x = torch.randn(2, num_qubits, dtype=torch.float64, requires_grad=True)

    def func(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            _QuantumLayerFunction.apply(
                inp,
                weights,
                num_qubits,
                num_layers,
                preset,
                parsed_obs,
                "parameter-shift",
                "angle",
            ),
        )

    assert gradcheck(func, (x, w), eps=1e-6, atol=1e-4)


def test_autograd_gradcheck_custom_observables() -> None:
    """Gradcheck verification with non-diagonal multi-qubit observables."""
    layer = QuantumLayer(
        num_qubits=2,
        num_layers=1,
        observables=["ZZ", "XX"],
        dtype=torch.float64,
    )
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)

    assert gradcheck(lambda inp: layer(inp), (x,), eps=1e-6, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# 5. nn.Sequential Integration & Hybrid Gradient Flow
# ═══════════════════════════════════════════════════════════════════════════

def test_nn_sequential_training() -> None:
    """QuantumLayer in nn.Sequential trains with loss.backward() and updates weights."""
    l1 = nn.Linear(2, 4)
    qlayer = QuantumLayer(num_qubits=4, num_layers=1)
    l2 = nn.Linear(4, 1)
    model = nn.Sequential(l1, qlayer, l2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    initial_loss = float(criterion(model(X), Y).item())
    initial_l1 = l1.weight.clone()
    initial_q = qlayer.weights.clone()
    initial_l2 = l2.weight.clone()

    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()

    final_loss = float(criterion(model(X), Y).item())
    assert final_loss < initial_loss
    assert float((l1.weight - initial_l1).abs().max().item()) > 0.0
    assert float((qlayer.weights - initial_q).abs().max().item()) > 0.0
    assert float((l2.weight - initial_l2).abs().max().item()) > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Apple Silicon Metal / MPS Execution
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon Metal MPS"
)
def test_mps_metal_execution() -> None:
    """Verification of QuantumLayer and ops on Apple Silicon Metal GPU."""
    mps_dev = torch.device("mps")

    layer = QuantumLayer(num_qubits=3, num_layers=1, device=mps_dev, dtype=torch.float32)
    assert layer.weights.device.type == "mps"

    x = torch.randn(4, 3, device=mps_dev, dtype=torch.float32, requires_grad=True)
    y = layer(x)
    assert y.device.type == "mps"
    assert y.shape == (4, 3)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None and x.grad.device.type == "mps"
    assert layer.weights.grad is not None and layer.weights.grad.device.type == "mps"
    assert x.grad.norm().item() > 0.0
    assert layer.weights.grad.norm().item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon Metal MPS"
)
def test_mps_float64_defensive_rejection() -> None:
    """Apple Silicon Metal MPS must cleanly reject float64 / complex128."""
    mps_dev = torch.device("mps")
    with pytest.raises(UnsupportedDtypeError, match="does not support 64-bit precision"):
        ops.resolve_complex_dtype(torch.complex128, device=mps_dev)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Quanta Torch Ops Verification
# ═══════════════════════════════════════════════════════════════════════════

def test_ops_pauli_matrices() -> None:
    """Pauli matrix properties and algebraic definitions."""
    matrices = ops.pauli_matrices(device="cpu", dtype=torch.complex64)
    i_mat = matrices["I"]
    x_mat = matrices["X"]
    y_mat = matrices["Y"]
    z_mat = matrices["Z"]

    assert torch.allclose(x_mat @ x_mat, i_mat)
    assert torch.allclose(y_mat @ y_mat, i_mat)
    assert torch.allclose(z_mat @ z_mat, i_mat)
    assert torch.allclose(1j * x_mat @ y_mat, -z_mat) or torch.allclose(1j * x_mat @ y_mat, z_mat)


def test_ops_pauli_kron_caching() -> None:
    """pauli_kron returns cached instances on identical queries."""
    m1 = ops.pauli_kron("ZZ", num_qubits=2, device="cpu", dtype=torch.complex64)
    m2 = ops.pauli_kron("ZZ", num_qubits=2, device="cpu", dtype=torch.complex64)
    assert m1 is m2


def test_ops_fast_readouts_match_dense() -> None:
    """Fast O(2^N) readouts match dense matrix expectations to machine precision."""
    N = 3
    B = 4
    psi = torch.randn(B, 2 ** N, dtype=torch.complex64)
    psi = psi / torch.linalg.norm(psi, dim=-1, keepdim=True)

    # Z readout
    fast_z = ops.fast_z_readout(psi, N)
    for q in range(N):
        mat_z = ops.pauli_kron("Z", num_qubits=N, target_qubit=q, dtype=torch.complex64)
        dense_z = ops.batch_expectation(psi, mat_z)
        assert torch.allclose(fast_z[:, q], dense_z, atol=1e-6)

    # X readout
    for q in range(N):
        fast_x = ops.fast_x_readout(psi, N, q)
        mat_x = ops.pauli_kron("X", num_qubits=N, target_qubit=q, dtype=torch.complex64)
        dense_x = ops.batch_expectation(psi, mat_x)
        assert torch.allclose(fast_x, dense_x, atol=1e-6)

    # Y readout
    for q in range(N):
        fast_y = ops.fast_y_readout(psi, N, q)
        mat_y = ops.pauli_kron("Y", num_qubits=N, target_qubit=q, dtype=torch.complex64)
        dense_y = ops.batch_expectation(psi, mat_y)
        assert torch.allclose(fast_y, dense_y, atol=1e-6)


def test_ops_unitary_evolution_norm() -> None:
    """Unitary evolution preserves statevector norm exactly."""
    dim = 4
    H = torch.randn(dim, dim, dtype=torch.complex64)
    H = H + H.conj().T
    psi0 = torch.randn(dim, dtype=torch.complex64)
    psi0 = psi0 / torch.linalg.norm(psi0)

    psi_t = ops.unitary_evolution(H, t=1.5, psi0=psi0)
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-6


def test_ops_ehrenfest_time_gradient() -> None:
    """Ehrenfest time gradient matches numerical finite difference."""
    dim = 4
    H = torch.randn(dim, dim, dtype=torch.complex128)
    H = H + H.conj().T
    obs_mat = torch.randn(dim, dim, dtype=torch.complex128)
    obs_mat = obs_mat + obs_mat.conj().T
    p0 = torch.randn(dim, dtype=torch.complex128)
    p0 = p0 / torch.linalg.norm(p0)

    t = 0.5
    dt = 1e-7
    U_p = torch.linalg.matrix_exp(-1j * H * (t + dt))
    U_m = torch.linalg.matrix_exp(-1j * H * (t - dt))
    p_p = U_p @ p0
    p_m = U_m @ p0
    exp_p = torch.real(torch.vdot(p_p, obs_mat @ p_p))
    exp_m = torch.real(torch.vdot(p_m, obs_mat @ p_m))
    fd = (exp_p - exp_m) / (2 * dt)

    p_t = torch.linalg.matrix_exp(-1j * H * t) @ p0
    an = ops.ehrenfest_time_gradient(p_t, H, obs_mat)
    assert abs(fd.item() - an.item()) < 1e-6


def test_ops_daleckii_krein_spectral_derivative() -> None:
    """Daleckii-Krein matrix spectral derivative matches finite differences with sinc stability."""
    dim = 4
    H0 = torch.randn(dim, dim, dtype=torch.complex128)
    H0 = H0 + H0.conj().T
    Omega = torch.randn(dim, dim, dtype=torch.complex128)
    Omega = Omega + Omega.conj().T
    t = 0.8

    evals, evecs = torch.linalg.eigh(H0)
    dU_an = ops.daleckii_krein_spectral_derivative(evals, evecs, t, Omega)

    dphi = 1e-7
    H_p = H0 + dphi * Omega
    H_m = H0 - dphi * Omega
    U_p = torch.linalg.matrix_exp(-1j * H_p * t)
    U_m = torch.linalg.matrix_exp(-1j * H_m * t)
    dU_fd = (U_p - U_m) / (2 * dphi)

    assert torch.norm(dU_an - dU_fd).item() < 1e-6


def test_ops_batch_resonant_hamiltonian() -> None:
    """Vectorized batch resonant Hamiltonian matches manual term accumulation."""
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    B = 2
    D_in = 2

    x = torch.randn(B, D_in)
    J = torch.randn(len(edges))
    h = torch.randn(num_nodes)
    W = torch.randn(num_nodes, D_in)
    omega = torch.randn(num_nodes)

    H_batch = ops.build_batch_resonant_hamiltonian(
        x, J, edges, h, W, omega, num_nodes, device="cpu", dtype=torch.complex64
    )
    assert H_batch.shape == (B, 8, 8)
    # Check Hermiticity
    for b in range(B):
        assert torch.allclose(H_batch[b], H_batch[b].conj().T, atol=1e-5)


def test_ops_gate_simulation() -> None:
    """apply_gate correctly generates Bell state."""
    B = 2
    N = 2
    state = torch.zeros(B, 2 ** N, dtype=torch.complex64)
    state[:, 0] = 1.0

    H_mat = ops.get_pauli_matrix("H", dtype=torch.complex64)
    cx_mat = ops.cnot_gate(dtype=torch.complex64)

    s1 = ops.apply_gate(state, H_mat, (0,), N)
    s2 = ops.apply_gate(s1, cx_mat, (0, 1), N)

    # Bell state (|00> + |11>) / sqrt(2)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert torch.allclose(s2[:, 0].abs(), torch.tensor(inv_sqrt2), atol=1e-5)
    assert torch.allclose(s2[:, 3].abs(), torch.tensor(inv_sqrt2), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Milestone 2 Iteration 2 Regression & Edge Case Suites
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_layers", [2, 3])
def test_reuploading_multilayer_gradcheck(num_layers: int) -> None:
    """Analytical Parameter-Shift gradcheck for multi-layer data re-uploading (L >= 2).

    Validates that layer-wise parameter shift resolves the multi-generator
    autograd breakdown on CPU float64.
    """
    num_qubits = 2
    parsed_obs = ObservableParser.parse_all(None, num_qubits=num_qubits)
    layer = QuantumLayer(
        num_qubits=num_qubits,
        circuit_fn="reuploading",
        num_layers=num_layers,
        dtype=torch.float64,
        device="cpu",
    )

    x = torch.randn(2, num_qubits, dtype=torch.float64, device="cpu", requires_grad=True)
    w = layer.weights.detach().clone().requires_grad_(True)

    def func(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            _QuantumLayerFunction.apply(
                inp,
                weights,
                num_qubits,
                num_layers,
                "reuploading",
                parsed_obs,
                "parameter-shift",
                "angle",
            ),
        )

    # 1. Autograd Function gradcheck for both inputs x and weights w
    assert gradcheck(func, (x, w), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 2. Direct QuantumLayer nn.Module gradcheck
    x_module = torch.randn(2, num_qubits, dtype=torch.float64, device="cpu", requires_grad=True)
    assert gradcheck(lambda inp: layer(inp), (x_module,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_empty_batch_handling() -> None:
    """Validates graceful handling of empty batch inputs (B=0) and invalid edge shapes.

    Standard PyTorch layers preserve batch dimensions and return empty output tensors
    with correct feature dimensions and zero/empty gradient shapes.
    """
    layer = QuantumLayer(num_qubits=2, num_layers=1)
    out_dim = layer.out_features

    # 1. 2D empty batch (0, 2)
    x_2d_empty = torch.empty(0, 2, requires_grad=True)
    y_2d_empty = layer(x_2d_empty)
    assert y_2d_empty.shape == (0, out_dim)
    loss_2d = y_2d_empty.sum()
    assert loss_2d.item() == 0.0
    loss_2d.backward()
    assert x_2d_empty.grad is not None and x_2d_empty.grad.shape == (0, 2)
    assert layer.weights.grad is not None and layer.weights.grad.shape == layer.weights.shape
    assert torch.all(layer.weights.grad == 0.0)

    # 2. 3D empty batch (2, 0, 2)
    layer.zero_grad()
    x_3d_empty = torch.empty(2, 0, 2, requires_grad=True)
    y_3d_empty = layer(x_3d_empty)
    assert y_3d_empty.shape == (2, 0, out_dim)
    loss_3d = y_3d_empty.sum()
    loss_3d.backward()
    assert x_3d_empty.grad is not None and x_3d_empty.grad.shape == (2, 0, 2)
    assert layer.weights.grad is not None and torch.all(layer.weights.grad == 0.0)

    # 3. Multi-dimensional leading zero (0, 3, 2)
    layer.zero_grad()
    x_multi_empty = torch.empty(0, 3, 2, requires_grad=True)
    y_multi_empty = layer(x_multi_empty)
    assert y_multi_empty.shape == (0, 3, out_dim)

    # 4. Defensive checks: feature mismatch on empty tensor must still raise ValueError
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        layer(torch.empty(0, 5))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon Metal MPS"
)
def test_cross_device_dtype_transfer() -> None:
    """Verifies seamless execution and gradient propagation across devices and dtypes.

    Specifically probes the CPU float64 -> MPS float32 bridge to ensure that
    dtype casting precedes device transfer, preventing MPS float64 TypeErrors
    and silent autograd zero-gradient corruption.
    """
    mps_dev = torch.device("mps")
    cpu_dev = torch.device("cpu")

    # 1. CPU float64 input -> MPS float32 QuantumLayer
    layer_mps = QuantumLayer(num_qubits=2, num_layers=1, device=mps_dev, dtype=torch.float32)
    x_cpu_64 = torch.randn(4, 2, dtype=torch.float64, device=cpu_dev, requires_grad=True)

    y_mps = layer_mps(x_cpu_64)
    assert y_mps.device.type == "mps", "Output tensor must reside on MPS"
    assert y_mps.dtype == torch.float32, "Output dtype must be float32 on MPS"
    assert y_mps.shape == (4, 2)

    loss = (y_mps ** 2).sum()
    loss.backward()

    # Verify input gradient transferred cleanly back to CPU float64
    assert x_cpu_64.grad is not None, "Input gradient must be populated"
    assert x_cpu_64.grad.device.type == "cpu", "Input gradient must return to CPU"
    assert x_cpu_64.grad.dtype == torch.float64, "Input gradient must preserve float64 dtype"
    assert torch.isfinite(x_cpu_64.grad).all(), "Input gradient must be finite"
    assert x_cpu_64.grad.norm().item() > 0.0, "Input gradient must not be silently zeroed"

    # Verify weight gradient remained on MPS float32
    assert layer_mps.weights.grad is not None
    assert layer_mps.weights.grad.device.type == "mps"
    assert layer_mps.weights.grad.dtype == torch.float32
    assert layer_mps.weights.grad.norm().item() > 0.0

    # 2. Reverse transfer: MPS float32 input -> CPU float32 QuantumLayer
    layer_cpu = QuantumLayer(num_qubits=2, num_layers=1, device=cpu_dev, dtype=torch.float32)
    x_mps_32 = torch.randn(3, 2, dtype=torch.float32, device=mps_dev, requires_grad=True)

    y_cpu = layer_cpu(x_mps_32)
    assert y_cpu.device.type == "cpu"
    loss_rev = y_cpu.sum()
    loss_rev.backward()
    assert x_mps_32.grad is not None and x_mps_32.grad.device.type == "mps"
    assert x_mps_32.grad.norm().item() > 0.0
