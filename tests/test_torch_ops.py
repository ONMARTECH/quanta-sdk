"""tests.test_torch_ops — Comprehensive unit tests for quanta.torch.ops.

Verifies full algebraic correctness, physical invariants, boundary edge cases,
and device/dtype transfers for all routines in quanta.torch.ops:
1. Device and Dtype Resolvers (resolve_device, resolve_complex_dtype, get_real_dtype,
   get_complex_dtype).
2. Statevector Conversions & Initializers (to_torch_state, to_numpy_state,
   create_initial_state).
3. Pauli Algebra & Multi-Qubit Kronecker Engine (get_pauli_matrix, pauli_matrices,
   pauli_kron, batch_pauli_kron).
4. Expectation Evaluation Engine (batch_expectation, batch_multi_expectation,
   hamiltonian_expectation).
5. Fast Readout Engines (fast_z_readout, fast_x_readout, fast_y_readout,
   simultaneous_readout).
6. Continuous Resonant Layer Operators (ResonantInteractionBasis,
   build_batch_resonant_hamiltonian, unitary_evolution, ehrenfest_time_gradient,
   daleckii_krein_spectral_derivative).
7. Quantum Gate Operations (apply_gate, rotation_x, rotation_y, rotation_z,
   cnot_gate, cz_gate).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from quanta.torch import (
    UnsupportedDtypeError,
    ops,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Device and Dtype Resolvers
# ═══════════════════════════════════════════════════════════════════════════

def test_resolve_device_explicit() -> None:
    """Explicit device specifications must return exact torch.device instances."""
    cpu_dev = ops.resolve_device("cpu")
    assert cpu_dev == torch.device("cpu")

    dev_obj = torch.device("cpu")
    assert ops.resolve_device(dev_obj) == dev_obj

    # Default device resolution
    resolved = ops.resolve_device(None)
    assert isinstance(resolved, torch.device)
    if torch.backends.mps.is_available():
        assert resolved == torch.device("mps")
    else:
        assert resolved == torch.device("cpu")


def test_resolve_device_fallback_when_mps_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MPS is not available, resolve_device(None) must cleanly fall back to CPU."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert ops.resolve_device(None) == torch.device("cpu")


def test_resolve_complex_dtype_resolutions() -> None:
    """Validates real and complex dtype mappings across CPU."""
    # None defaults to complex64
    assert ops.resolve_complex_dtype(None, "cpu") == torch.complex64

    # 32-bit real/complex map to complex64
    assert ops.resolve_complex_dtype(torch.float32, "cpu") == torch.complex64
    assert ops.resolve_complex_dtype(torch.complex64, "cpu") == torch.complex64

    # 64-bit real/complex map to complex128
    assert ops.resolve_complex_dtype(torch.float64, "cpu") == torch.complex128
    assert ops.resolve_complex_dtype(torch.complex128, "cpu") == torch.complex128

    # Unrecognized or custom complex dtype passes through
    assert ops.resolve_complex_dtype(torch.complex32, "cpu") == torch.complex32


def test_resolve_complex_dtype_mps_64bit_error() -> None:
    """Apple Silicon MPS must reject 64-bit precision with UnsupportedDtypeError."""
    msg = "Apple Silicon MPS does not support 64-bit precision"
    with pytest.raises(UnsupportedDtypeError, match=msg):
        ops.resolve_complex_dtype(torch.complex128, "mps")

    with pytest.raises(UnsupportedDtypeError, match=msg):
        ops.resolve_complex_dtype(torch.float64, "mps")


def test_get_real_and_complex_dtype_inverses() -> None:
    """Tests bijective mapping between complex and real scalar dtypes and error handling."""
    assert ops.get_real_dtype(torch.complex64) == torch.float32
    assert ops.get_real_dtype(torch.complex128) == torch.float64
    with pytest.raises(ValueError, match="Unsupported complex dtype"):
        ops.get_real_dtype(torch.float32)

    assert ops.get_complex_dtype(torch.float32) == torch.complex64
    assert ops.get_complex_dtype(torch.float64) == torch.complex128
    with pytest.raises(ValueError, match="Unsupported real dtype"):
        ops.get_complex_dtype(torch.complex64)


# ═══════════════════════════════════════════════════════════════════════════
# 2. State Conversions & Initial State Generators
# ═══════════════════════════════════════════════════════════════════════════

def test_to_torch_state_from_numpy() -> None:
    """Converts NumPy array into normalized complex PyTorch statevector."""
    arr = np.array([3.0, 4.0], dtype=np.complex64)
    state = ops.to_torch_state(arr, device="cpu", dtype=torch.complex64, normalize=True)
    assert isinstance(state, torch.Tensor)
    assert state.dtype == torch.complex64
    assert state.device == torch.device("cpu")
    # Norm of [3, 4] is 5; normalized state has components 0.6 and 0.8
    expected = torch.tensor([0.6 + 0.0j, 0.8 + 0.0j], dtype=torch.complex64)
    assert torch.allclose(state, expected, atol=1e-6)

    # normalize=False
    raw_state = ops.to_torch_state(
        arr, device="cpu", dtype=torch.complex64, normalize=False
    )
    expected_raw = torch.tensor([3.0 + 0.0j, 4.0 + 0.0j], dtype=torch.complex64)
    assert torch.allclose(raw_state, expected_raw, atol=1e-6)


def test_to_torch_state_from_tensor_and_invalid_type() -> None:
    """Handles PyTorch tensors and raises TypeError for unsupported input objects."""
    t = torch.tensor([1.0, -1.0], dtype=torch.float32)
    state = ops.to_torch_state(t, device="cpu", dtype=torch.complex128)
    assert state.dtype == torch.complex128
    norm_val = torch.linalg.norm(state)
    assert torch.isclose(norm_val, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

    # Near-zero norm clamping verification
    zero_t = torch.zeros(4, dtype=torch.complex64)
    zero_state = ops.to_torch_state(zero_t, device="cpu")
    assert not torch.isnan(zero_state).any()

    with pytest.raises(TypeError, match="Expected np.ndarray or torch.Tensor"):
        ops.to_torch_state([1.0, 0.0])  # type: ignore[arg-type]


def test_to_numpy_state_normal_and_conjugated() -> None:
    """to_numpy_state must properly detach, transfer to CPU, and resolve conjugate views."""
    t = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64)
    arr = ops.to_numpy_state(t)
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64))

    # Conjugated view resolution
    t_conj = t.conj()
    assert t_conj.is_conj()
    arr_conj = ops.to_numpy_state(t_conj)
    assert isinstance(arr_conj, np.ndarray)
    assert np.allclose(arr_conj, np.array([1.0 - 2.0j, 3.0 + 4.0j], dtype=np.complex64))


def test_create_initial_state_modes_and_broadcasting() -> None:
    """create_initial_state constructs zero, plus, bell/ghz states and batch expansion."""
    # Mode: 'zero' or '0'
    s_zero = ops.create_initial_state("zero", num_qubits=2, device="cpu")
    assert s_zero.shape == (4,)
    assert s_zero[0] == 1.0 + 0.0j
    assert torch.sum(torch.abs(s_zero[1:])).item() == 0.0

    s_zero_alt = ops.create_initial_state("0", num_qubits=1, device="cpu")
    assert s_zero_alt[0] == 1.0

    # Mode: 'plus' or '+'
    s_plus = ops.create_initial_state("plus", num_qubits=2, device="cpu")
    assert s_plus.shape == (4,)
    assert torch.allclose(s_plus, torch.ones(4, dtype=torch.complex64) * 0.5)

    s_plus_alt = ops.create_initial_state("+", num_qubits=1, device="cpu")
    assert torch.allclose(
        s_plus_alt, torch.ones(2, dtype=torch.complex64) / math.sqrt(2.0)
    )

    # Mode: 'ghz' or 'bell' -> (|00> + |11>) / sqrt(2)
    s_bell = ops.create_initial_state("bell", num_qubits=2, device="cpu")
    assert s_bell.shape == (4,)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert torch.isclose(s_bell[0], torch.tensor(inv_sqrt2, dtype=torch.complex64))
    assert torch.isclose(s_bell[3], torch.tensor(inv_sqrt2, dtype=torch.complex64))
    assert s_bell[1] == 0.0
    assert s_bell[2] == 0.0

    s_ghz = ops.create_initial_state("ghz", num_qubits=3, device="cpu")
    assert s_ghz.shape == (8,)
    assert torch.isclose(s_ghz[0], torch.tensor(inv_sqrt2, dtype=torch.complex64))
    assert torch.isclose(s_ghz[7], torch.tensor(inv_sqrt2, dtype=torch.complex64))

    # Custom tensor with batch expansion
    custom_arr = np.array([1.0, 0.0], dtype=np.complex64)
    batched = ops.create_initial_state(
        custom_arr, num_qubits=1, batch_size=5, device="cpu"
    )
    assert batched.shape == (5, 2)
    for b in range(5):
        assert batched[b, 0] == 1.0

    # Unknown mode raises ValueError
    with pytest.raises(ValueError, match="Unknown initial state mode"):
        ops.create_initial_state("invalid_mode", num_qubits=2)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pauli Algebra & Multi-Qubit Kronecker Products
# ═══════════════════════════════════════════════════════════════════════════

def test_get_pauli_matrix_all_operators_and_algebra() -> None:
    """Verifies all single-qubit Pauli matrices, ladder operators, and commutation rules."""
    matrices = ops.pauli_matrices("cpu", dtype=torch.complex64)
    eye_op = matrices["I"]
    x_op = matrices["X"]
    y_op = matrices["Y"]
    z_op = matrices["Z"]
    plus_op = matrices["+"]
    minus_op = matrices["-"]
    h_op = matrices["H"]

    # Involutory unitaries: X^2 = Y^2 = Z^2 = H^2 = I
    assert torch.allclose(x_op @ x_op, eye_op)
    assert torch.allclose(y_op @ y_op, eye_op)
    assert torch.allclose(z_op @ z_op, eye_op)
    assert torch.allclose(h_op @ h_op, eye_op)

    # Pauli commutation relations: [X, Y] = 2i Z, etc.
    assert torch.allclose(x_op @ y_op - y_op @ x_op, 2.0j * z_op)
    assert torch.allclose(y_op @ z_op - z_op @ y_op, 2.0j * x_op)
    assert torch.allclose(z_op @ x_op - x_op @ z_op, 2.0j * y_op)

    # Ladder operators: sigma_+ = |0><1|, sigma_- = |1><0|
    assert torch.allclose(plus_op + minus_op, x_op)
    assert torch.allclose(plus_op - minus_op, 1.0j * y_op)

    # Aliases
    assert torch.allclose(ops.get_pauli_matrix("plus", "cpu", torch.complex64), plus_op)
    assert torch.allclose(ops.get_pauli_matrix("minus", "cpu", torch.complex64), minus_op)

    # Cache hit returns same object
    mat1 = ops.get_pauli_matrix("X", "cpu", torch.complex64)
    mat2 = ops.get_pauli_matrix("x", "cpu", torch.complex64)
    assert mat1 is mat2

    with pytest.raises(ValueError, match="Unknown Pauli operator name"):
        ops.get_pauli_matrix("W")


def test_pauli_kron_edge_cases_and_padding() -> None:
    """pauli_kron must validate target_qubit ranges and correctly pad or error on lengths."""
    # target_qubit without num_qubits raises ValueError
    msg = "num_qubits must be specified when target_qubit is provided"
    with pytest.raises(ValueError, match=msg):
        ops.pauli_kron("X", target_qubit=1)

    # target_qubit out of range
    with pytest.raises(ValueError, match="out of range"):
        ops.pauli_kron("X", num_qubits=3, target_qubit=-1)
    with pytest.raises(ValueError, match="out of range"):
        ops.pauli_kron("X", num_qubits=3, target_qubit=3)

    # Auto-padding when len(pauli_str) < num_qubits: 'Z' padded with 'I' for num_qubits=3 -> 'ZII'
    z_padded = ops.pauli_kron("Z", num_qubits=3, device="cpu")
    z_explicit = ops.pauli_kron("ZII", num_qubits=3, device="cpu")
    assert torch.allclose(z_padded, z_explicit)

    # Exceeding num_qubits raises ValueError
    with pytest.raises(ValueError, match="exceeds num_qubits"):
        ops.pauli_kron("ZZZZ", num_qubits=2)

    # Target qubit embedding matches explicit string
    x1_embedded = ops.pauli_kron("X", num_qubits=3, target_qubit=1, device="cpu")
    x1_explicit = ops.pauli_kron("IXI", num_qubits=3, device="cpu")
    assert torch.allclose(x1_embedded, x1_explicit)

    # Cached retrieval returns identical tensor
    cached_1 = ops.pauli_kron("ZZ", num_qubits=2, device="cpu")
    cached_2 = ops.pauli_kron("ZZ", num_qubits=2, device="cpu")
    assert cached_1 is cached_2


def test_batch_pauli_kron_construction() -> None:
    """batch_pauli_kron must stack multiple operators into shape (M, 2^N, 2^N)."""
    operators = ["XX", "YY", "ZZ", "XI"]
    stacked = ops.batch_pauli_kron(operators, num_qubits=2, device="cpu")
    assert stacked.shape == (4, 4, 4)

    for i, name in enumerate(operators):
        expected = ops.pauli_kron(name, num_qubits=2, device="cpu")
        assert torch.allclose(stacked[i], expected)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Expectation Evaluation Engine
# ═══════════════════════════════════════════════════════════════════════════

def test_batch_expectation_dimensions_and_device_transfer() -> None:
    """batch_expectation must handle 1D, 2D, and multi-dimensional statevectors."""
    z = ops.pauli_kron("Z", num_qubits=1, device="cpu")

    # 1D statevector: |0> has <Z> = 1.0, |1> has <Z> = -1.0
    psi_0 = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64)
    exp_0 = ops.batch_expectation(psi_0, z)
    assert exp_0.dim() == 0  # Squeezed to scalar
    assert torch.isclose(exp_0, torch.tensor(1.0))

    psi_1 = torch.tensor([0.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
    exp_1 = ops.batch_expectation(psi_1, z)
    assert torch.isclose(exp_1, torch.tensor(-1.0))

    # 2D batch: (B=2, D=2)
    psi_batch = torch.stack([psi_0, psi_1], dim=0)
    exp_batch = ops.batch_expectation(psi_batch, z)
    assert exp_batch.shape == (2,)
    assert torch.allclose(exp_batch, torch.tensor([1.0, -1.0]))

    # 3D tensor: shape (2, 3, 2)
    psi_3d = psi_batch.unsqueeze(1).expand(2, 3, 2).contiguous()
    exp_3d = ops.batch_expectation(psi_3d, z)
    assert exp_3d.shape == (2, 3)
    assert torch.allclose(exp_3d[0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(exp_3d[1], torch.tensor([-1.0, -1.0, -1.0]))

    # Cross-device observable transfer
    if torch.backends.mps.is_available():
        psi_mps = psi_batch.to(device="mps")
        z_cpu = z.to(device="cpu")
        exp_mps = ops.batch_expectation(psi_mps, z_cpu)
        assert exp_mps.device.type == "mps"
        assert torch.allclose(exp_mps.cpu(), torch.tensor([1.0, -1.0]))


def test_batch_multi_expectation_evaluations() -> None:
    """batch_multi_expectation computes fused expectations for M observables."""
    num_qubits = 2
    paulis = ["ZZ", "XX", "YY"]
    num_m = len(paulis)
    obs_stack = ops.batch_pauli_kron(paulis, num_qubits=num_qubits, device="cpu")

    # 1. 1D statevector: |00>
    # <00|ZZ|00> = 1.0, <00|XX|00> = 0.0, <00|YY|00> = 0.0
    psi_00 = torch.zeros(4, dtype=torch.complex64)
    psi_00[0] = 1.0
    res_1d = ops.batch_multi_expectation(psi_00, obs_stack)
    assert res_1d.shape == (num_m,)
    assert torch.allclose(res_1d, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)

    # 2. 2D batched statevector: |00> and Bell state (|00> + |11>)/sqrt(2)
    # For Bell state: <ZZ> = 1.0, <XX> = 1.0, <YY> = -1.0
    psi_bell = torch.zeros(4, dtype=torch.complex64)
    psi_bell[0] = 1.0 / math.sqrt(2.0)
    psi_bell[3] = 1.0 / math.sqrt(2.0)

    psi_batch = torch.stack([psi_00, psi_bell], dim=0)
    res_batch = ops.batch_multi_expectation(psi_batch, obs_stack)
    assert res_batch.shape == (2, num_m)
    assert torch.allclose(res_batch[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(res_batch[1], torch.tensor([1.0, 1.0, -1.0]), atol=1e-6)

    # Cross-device observables transfer
    if torch.backends.mps.is_available():
        psi_mps = psi_batch.to(device="mps")
        obs_cpu = obs_stack.to(device="cpu")
        res_mps = ops.batch_multi_expectation(psi_mps, obs_cpu)
        assert res_mps.device.type == "mps"
        assert torch.allclose(res_mps.cpu(), res_batch, atol=1e-5)


def test_hamiltonian_expectation_weighted_paulis() -> None:
    """hamiltonian_expectation evaluates sum_k c_k <P_k> matching matrix multiplication."""
    num_qubits = 2
    terms = [("ZZ", 1.5), ("XX", -0.5), ("IZ", 0.8)]

    # Build explicit matrix H_mat = 1.5 ZZ - 0.5 XX + 0.8 IZ
    h_mat = (
        1.5 * ops.pauli_kron("ZZ", num_qubits=2, device="cpu")
        - 0.5 * ops.pauli_kron("XX", num_qubits=2, device="cpu")
        + 0.8 * ops.pauli_kron("IZ", num_qubits=2, device="cpu")
    )

    # Test state 1: 1D Bell state
    psi_1d = ops.create_initial_state("bell", num_qubits=2, device="cpu")
    exp_val = ops.hamiltonian_expectation(psi_1d, terms, num_qubits=num_qubits)
    assert exp_val.dim() == 0

    # Direct <psi| H |psi>
    expected_1d = torch.real(torch.sum(psi_1d.conj() * (h_mat @ psi_1d)))
    assert torch.isclose(exp_val, expected_1d, atol=1e-6)

    # Test state 2: 2D batch of random states
    torch.manual_seed(42)
    raw = torch.randn(4, 4, dtype=torch.complex64)
    norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
    psi_batch = raw / norms

    exp_batch = ops.hamiltonian_expectation(psi_batch, terms, num_qubits=num_qubits)
    assert exp_batch.shape == (4,)

    for b in range(4):
        exp_direct = torch.real(torch.sum(psi_batch[b].conj() * (h_mat @ psi_batch[b])))
        assert torch.isclose(exp_batch[b], exp_direct, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Fast Readout Engines
# ═══════════════════════════════════════════════════════════════════════════

def test_fast_z_readout_1d_and_target_selection() -> None:
    """fast_z_readout calculates local Z expectations for 1D states and target subsets."""
    # State |01>: qubit 0 is 0 (exp = +1), qubit 1 is 1 (exp = -1)
    psi_1d = torch.zeros(4, dtype=torch.complex64)
    psi_1d[1] = 1.0  # |01> in big-endian

    # All qubits readout
    res_all = ops.fast_z_readout(psi_1d, num_qubits=2)
    assert res_all.shape == (2,)
    assert torch.allclose(res_all, torch.tensor([1.0, -1.0]))

    # Target subset [1]
    res_sub = ops.fast_z_readout(psi_1d, num_qubits=2, target_qubits=[1])
    assert res_sub.shape == (1,)
    assert torch.isclose(res_sub[0], torch.tensor(-1.0))


def test_fast_x_readout_single_qubit_and_multiqubit() -> None:
    """fast_x_readout covers both 1-qubit tensor slicing and multi-qubit reduction."""
    # 1. Single qubit (N=1)
    plus_1q = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
    x_plus = ops.fast_x_readout(plus_1q, num_qubits=1, target_qubit=0)
    assert torch.isclose(x_plus, torch.tensor(1.0, dtype=torch.float32), atol=1e-6)

    minus_1q = torch.tensor([1.0, -1.0], dtype=torch.complex64) / math.sqrt(2.0)
    x_minus = ops.fast_x_readout(minus_1q, num_qubits=1, target_qubit=0)
    assert torch.isclose(x_minus, torch.tensor(-1.0, dtype=torch.float32), atol=1e-6)

    # 2. Multi-qubit (N=3) batched
    # State |+ 0 0>: X on qubit 0 is 1.0, X on qubit 1 is 0.0, X on qubit 2 is 0.0
    plus_3q = torch.zeros(8, dtype=torch.complex64)
    plus_3q[0] = 1.0 / math.sqrt(2.0)  # |000>
    plus_3q[4] = 1.0 / math.sqrt(2.0)  # |100>
    x0 = ops.fast_x_readout(plus_3q, num_qubits=3, target_qubit=0)
    assert torch.isclose(x0, torch.tensor(1.0), atol=1e-6)
    x1 = ops.fast_x_readout(plus_3q, num_qubits=3, target_qubit=1)
    assert torch.isclose(x1, torch.tensor(0.0), atol=1e-6)


def test_fast_y_readout_single_qubit_and_multiqubit() -> None:
    """fast_y_readout covers 1-qubit tensor slicing and multi-qubit imaginary reduction."""
    # 1. Single qubit (N=1): |+i> = (|0> + i|1>)/sqrt(2) has <Y> = 1.0
    plus_i = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64) / math.sqrt(2.0)
    y_plus_i = ops.fast_y_readout(plus_i, num_qubits=1, target_qubit=0)
    assert torch.isclose(y_plus_i, torch.tensor(1.0, dtype=torch.float32), atol=1e-6)

    # |-i> = (|0> - i|1>)/sqrt(2) has <Y> = -1.0
    minus_i = torch.tensor([1.0 + 0.0j, 0.0 - 1.0j], dtype=torch.complex64) / math.sqrt(2.0)
    y_minus_i = ops.fast_y_readout(minus_i, num_qubits=1, target_qubit=0)
    assert torch.isclose(y_minus_i, torch.tensor(-1.0, dtype=torch.float32), atol=1e-6)

    # 2. Multi-qubit (N=2)
    # State |0> (x) |+i> has Y on qubit 1 equal to 1.0
    psi_2q = torch.tensor([1.0, 1.0j, 0.0, 0.0], dtype=torch.complex64) / math.sqrt(2.0)
    y0 = ops.fast_y_readout(psi_2q, num_qubits=2, target_qubit=0)
    assert torch.isclose(y0, torch.tensor(0.0), atol=1e-6)
    y1 = ops.fast_y_readout(psi_2q, num_qubits=2, target_qubit=1)
    assert torch.isclose(y1, torch.tensor(1.0), atol=1e-6)


def test_simultaneous_readout_1d_and_observable_types() -> None:
    """simultaneous_readout evaluates concurrent Z, X, Y expectations across all nodes."""
    psi_1d = ops.create_initial_state("zero", num_qubits=2, device="cpu")
    # For |00>: Z0=1, Z1=1, X0=0, X1=0, Y0=0, Y1=0
    out_all = ops.simultaneous_readout(psi_1d, num_nodes=2, observable_types=("Z", "X", "Y"))
    assert out_all.shape == (6,)
    expected = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(out_all, expected, atol=1e-6)

    # Unsupported observable raises ValueError
    with pytest.raises(ValueError, match="Unsupported observable type for fast readout"):
        ops.simultaneous_readout(psi_1d, num_nodes=2, observable_types=("Z", "W"))


# ═══════════════════════════════════════════════════════════════════════════
# 6. Continuous Dynamics & Resonant Operators
# ═══════════════════════════════════════════════════════════════════════════

def test_resonant_interaction_basis_and_hamiltonian_builder() -> None:
    """Verifies ResonantInteractionBasis and build_batch_resonant_hamiltonian."""
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    basis = ops.ResonantInteractionBasis(num_nodes=num_nodes, edges=edges, device="cpu")
    assert len(basis.xy_matrices) == 2
    assert len(basis.z_matrices) == 3
    assert len(basis.x_matrices) == 3
    assert basis.longitudinal_signs.shape == (3, 8)

    # Ensure all interaction basis operators are Hermitian
    for xy in basis.xy_matrices:
        assert torch.allclose(xy, xy.conj().T, atol=1e-6)

    # Test build_batch_resonant_hamiltonian with basis=None (auto-construction)
    b_size = 2
    x = torch.randn(b_size, 3)
    coupling_j = torch.tensor([0.5, -0.3])
    bias_h = torch.tensor([0.1, 0.2, -0.1])
    weight_w = torch.randn(3, 3)
    field_omega = torch.tensor([0.4, 0.0, -0.4])

    h_batch = ops.build_batch_resonant_hamiltonian(
        x=x,
        J=coupling_j,
        edges=edges,
        h=bias_h,
        W=weight_w,
        omega=field_omega,
        num_nodes=num_nodes,
        basis=None,
        device="cpu",
    )
    assert h_batch.shape == (b_size, 8, 8)
    for b in range(b_size):
        # Hamiltonian must be strictly Hermitian
        assert torch.allclose(h_batch[b], h_batch[b].conj().T, atol=1e-6)


def test_unitary_evolution_tensor_times_and_dimensions() -> None:
    """unitary_evolution covers scalar, 3D times, 1D/2D states, and dimension checks."""
    dim = 4
    # Simple diagonal Hamiltonian H = diag(1, -1, 2, -2)
    h_mat = torch.diag(torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.complex64))
    psi0 = torch.ones(dim, dtype=torch.complex64) / 2.0

    # 1. Scalar tensor time: t.dim() == 0 (line 747)
    t_0d = torch.tensor(0.5)
    psi_t0 = ops.unitary_evolution(h_mat, t_0d, psi0)
    assert torch.isclose(torch.linalg.norm(psi_t0), torch.tensor(1.0), atol=1e-6)

    # 2. 3D tensor time: shape (B, 1, 1) (line 751)
    b_size = 2
    h_batch = h_mat.unsqueeze(0).repeat(b_size, 1, 1)
    t_3d = torch.tensor([[[0.5]], [[1.0]]])
    psi_t_3d = ops.unitary_evolution(h_batch, t_3d, psi0)
    assert psi_t_3d.shape == (b_size, dim)
    assert torch.isclose(torch.linalg.norm(psi_t_3d[0]), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(torch.linalg.norm(psi_t_3d[1]), torch.tensor(1.0), atol=1e-6)

    # 3. 2D psi0 with 3D h_batch (lines 769-770)
    psi0_batch = psi0.unsqueeze(0).repeat(b_size, 1)
    psi_t_batched = ops.unitary_evolution(h_batch, 0.5, psi0_batch)
    assert psi_t_batched.shape == (b_size, dim)

    # 4. Cross-device transfer (line 759)
    if torch.backends.mps.is_available():
        h_mps = h_mat.to("mps")
        psi0_cpu = psi0.to("cpu")
        psi_mps = ops.unitary_evolution(h_mps, 0.5, psi0_cpu)
        assert psi_mps.device.type == "mps"

    # 5. Unsupported psi0 dimension raises ValueError (line 772)
    with pytest.raises(ValueError, match="Unsupported psi0 dimension"):
        ops.unitary_evolution(h_mat, 0.5, torch.zeros((2, 2, 2), dtype=torch.complex64))


def test_ehrenfest_time_gradient_branches() -> None:
    """ehrenfest_time_gradient must handle 1D states, 3D Hamiltonians, and device transfers."""
    h_mat = ops.pauli_kron("X", num_qubits=1, device="cpu")
    obs_mat = ops.pauli_kron("Z", num_qubits=1, device="cpu")

    # State |0>
    psi_0 = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64)

    # 1. 1D statevector input (squeeze)
    grad_1d = ops.ehrenfest_time_gradient(psi_0, h_mat, obs_mat)
    assert grad_1d.dim() == 0
    # ZX = -iY = [[0, -1], [1, 0]], <0|ZX|0> = 0, so grad_1d = 0.
    assert torch.isclose(grad_1d, torch.tensor(0.0), atol=1e-6)

    # For state |+y> = (|0> + i|1>)/sqrt(2), <+y|Y|+y> = 1.0.
    # Since ZX = iY, <+y|ZX|+y> = i, and 2 Im[i] = +2.0.
    psi_y = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64) / math.sqrt(2.0)
    grad_y = ops.ehrenfest_time_gradient(psi_y, h_mat, obs_mat)
    assert torch.isclose(grad_y, torch.tensor(2.0), atol=1e-6)

    # 2. 3D Hamiltonian input: shape (B, D, D) (line 799)
    b_size = 3
    h_3d = h_mat.unsqueeze(0).repeat(b_size, 1, 1)
    psi_batch = psi_y.unsqueeze(0).repeat(b_size, 1)
    grad_3d = ops.ehrenfest_time_gradient(psi_batch, h_3d, obs_mat)
    assert grad_3d.shape == (b_size,)
    assert torch.allclose(grad_3d, torch.tensor([2.0, 2.0, 2.0]), atol=1e-6)

    # 3. Cross-device transfers (lines 791, 793)
    if torch.backends.mps.is_available():
        psi_mps = psi_batch.to("mps")
        h_cpu = h_3d.to("cpu")
        obs_cpu = obs_mat.to("cpu")
        grad_mps = ops.ehrenfest_time_gradient(psi_mps, h_cpu, obs_cpu)
        assert grad_mps.device.type == "mps"
        assert torch.allclose(grad_mps.cpu(), grad_3d, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Quantum Gate Operations
# ═══════════════════════════════════════════════════════════════════════════

def test_rotation_x_matrix_properties() -> None:
    """rotation_x must satisfy RX(theta) = cos(theta/2) I - i sin(theta/2) X."""
    # At theta = 0: RX(0) = I
    rx_0 = ops.rotation_x(torch.tensor(0.0), device="cpu")
    assert torch.allclose(rx_0, torch.eye(2, dtype=torch.complex64), atol=1e-6)

    # At theta = pi: RX(pi) = -i X = [[0, -i], [-i, 0]]
    rx_pi = ops.rotation_x(torch.tensor(math.pi), device="cpu")
    expected_pi = torch.tensor([[0.0, -1.0j], [-1.0j, 0.0]], dtype=torch.complex64)
    assert torch.allclose(rx_pi, expected_pi, atol=1e-6)

    # Unitary: RX^dagger RX = I
    theta = torch.tensor(1.234)
    rx = ops.rotation_x(theta, device="cpu")
    assert torch.allclose(rx.conj().T @ rx, torch.eye(2, dtype=torch.complex64), atol=1e-6)

    # Eigenvalues: exp(-i theta/2) and exp(i theta/2)
    evals = torch.linalg.eigvals(rx)
    expected_eval_1 = torch.exp(-0.5j * theta)
    expected_eval_2 = torch.exp(0.5j * theta)
    eval_match = (
        torch.isclose(evals[0], expected_eval_1, atol=1e-5)
        or torch.isclose(evals[0], expected_eval_2, atol=1e-5)
    )
    assert eval_match


def test_rotation_y_matrix_properties() -> None:
    """rotation_y must satisfy RY(theta) = cos(theta/2) I - i sin(theta/2) Y."""
    # At theta = 0: RY(0) = I
    ry_0 = ops.rotation_y(torch.tensor(0.0), device="cpu")
    assert torch.allclose(ry_0, torch.eye(2, dtype=torch.complex64), atol=1e-6)

    # At theta = pi: RY(pi) = -i Y = [[0, -1], [1, 0]]
    ry_pi = ops.rotation_y(torch.tensor(math.pi), device="cpu")
    expected_pi = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.complex64)
    assert torch.allclose(ry_pi, expected_pi, atol=1e-6)

    # Action on |0>: RY(pi/2)|0> = [1/sqrt(2), 1/sqrt(2)] = |+>
    ry_half_pi = ops.rotation_y(torch.tensor(math.pi / 2.0), device="cpu")
    ket_0 = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    plus_state = ry_half_pi @ ket_0
    expected_plus = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
    assert torch.allclose(plus_state, expected_plus, atol=1e-6)

    # Unitary: RY^dagger RY = I
    theta = torch.tensor(2.468)
    ry = ops.rotation_y(theta, device="cpu")
    assert torch.allclose(ry.conj().T @ ry, torch.eye(2, dtype=torch.complex64), atol=1e-6)


def test_rotation_z_matrix_properties() -> None:
    """rotation_z must satisfy RZ(theta) = diag(exp(-i theta/2), exp(i theta/2))."""
    # At theta = 0: RZ(0) = I
    rz_0 = ops.rotation_z(torch.tensor(0.0), device="cpu")
    assert torch.allclose(rz_0, torch.eye(2, dtype=torch.complex64), atol=1e-6)

    # At theta = pi: RZ(pi) = diag(-i, i)
    rz_pi = ops.rotation_z(torch.tensor(math.pi), device="cpu")
    expected_pi = torch.diag(torch.tensor([-1.0j, 1.0j], dtype=torch.complex64))
    assert torch.allclose(rz_pi, expected_pi, atol=1e-6)

    # Action on |+>: RZ(pi) |+> = -i |->
    ket_plus = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
    ket_minus = torch.tensor([1.0, -1.0], dtype=torch.complex64) / math.sqrt(2.0)
    assert torch.allclose(rz_pi @ ket_plus, -1.0j * ket_minus, atol=1e-6)

    # Unitary: RZ^dagger RZ = I
    theta = torch.tensor(0.876)
    rz = ops.rotation_z(theta, device="cpu")
    assert torch.allclose(rz.conj().T @ rz, torch.eye(2, dtype=torch.complex64), atol=1e-6)


def test_cz_gate_matrix_and_action() -> None:
    """cz_gate constructs the exact 2-qubit Controlled-Z unitary matrix diag(1, 1, 1, -1)."""
    cz = ops.cz_gate(device="cpu")
    assert cz.shape == (4, 4)

    expected = torch.diag(torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.complex64))
    assert torch.allclose(cz, expected, atol=1e-6)

    # Unitary & Hermitian
    assert torch.allclose(cz.conj().T @ cz, torch.eye(4, dtype=torch.complex64), atol=1e-6)
    assert torch.allclose(cz, cz.conj().T, atol=1e-6)

    # Computational basis actions: |00>->|00>, |01>->|01>, |10>->|10>, |11>-> -|11>
    e00 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    e01 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.complex64)
    e10 = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.complex64)
    e11 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.complex64)

    assert torch.allclose(cz @ e00, e00)
    assert torch.allclose(cz @ e01, e01)
    assert torch.allclose(cz @ e10, e10)
    assert torch.allclose(cz @ e11, -e11)

    # Relation to CNOT: CZ = (I (x) H) @ CNOT @ (I (x) H)
    h_gate = ops.get_pauli_matrix("H", "cpu", torch.complex64)
    i_gate = ops.get_pauli_matrix("I", "cpu", torch.complex64)
    ih = torch.kron(i_gate, h_gate)
    cnot = ops.cnot_gate(device="cpu")
    cz_from_cnot = ih @ cnot @ ih
    assert torch.allclose(cz, cz_from_cnot, atol=1e-6)


def test_apply_gate_1d_and_multi_qubit_permutations() -> None:
    """apply_gate transforms 1D and batched statevectors across arbitrary qubit orderings."""
    # 1. 1D statevector (covers lines 856, 885-886)
    psi_1d = ops.create_initial_state("zero", num_qubits=2, device="cpu")  # |00>
    h_gate = ops.get_pauli_matrix("H", "cpu", torch.complex64)

    # Apply H to qubit 0: |00> -> (|00> + |10>) / sqrt(2)
    out_1d = ops.apply_gate(psi_1d, h_gate, qubits=(0,), num_qubits=2)
    assert out_1d.dim() == 1
    assert out_1d.shape == (4,)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert torch.isclose(out_1d[0], torch.tensor(inv_sqrt2, dtype=torch.complex64))
    assert torch.isclose(out_1d[2], torch.tensor(inv_sqrt2, dtype=torch.complex64))
    assert out_1d[1] == 0.0
    assert out_1d[3] == 0.0

    # 2. 2-qubit CZ gate on |11> -> -|11>
    psi_11 = torch.zeros(4, dtype=torch.complex64)
    psi_11[3] = 1.0
    cz = ops.cz_gate(device="cpu")
    out_cz = ops.apply_gate(psi_11, cz, qubits=(0, 1), num_qubits=2)
    assert torch.allclose(out_cz, -psi_11, atol=1e-6)

    # Symmetric application: qubits=(1, 0)
    out_cz_rev = ops.apply_gate(psi_11, cz, qubits=(1, 0), num_qubits=2)
    assert torch.allclose(out_cz_rev, -psi_11, atol=1e-6)

    # 3. Cross-device gate auto-transfer
    if torch.backends.mps.is_available():
        psi_mps = psi_1d.to("mps")
        h_cpu = h_gate.to("cpu")
        out_mps = ops.apply_gate(psi_mps, h_cpu, qubits=(0,), num_qubits=2)
        assert out_mps.device.type == "mps"
        assert torch.allclose(out_mps.cpu(), out_1d, atol=1e-5)
