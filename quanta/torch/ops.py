"""quanta.torch.ops — PyTorch Native Quantum Operations & Pauli Tensor Engine.

Provides high-performance, autograd-differentiable quantum tensor operations,
Pauli algebra, batch expectation evaluation, Hamiltonian builders, and device transfers
for QuantumLayer and ContinuousResonantLayer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from quanta.core.types import QuantaError

__all__ = [
    # Exception
    "UnsupportedDtypeError",
    # Device & Dtype Resolvers
    "resolve_device",
    "resolve_complex_dtype",
    "get_real_dtype",
    "get_complex_dtype",
    "to_torch_state",
    "to_numpy_state",
    "create_initial_state",
    # Pauli Algebra
    "get_pauli_matrix",
    "pauli_matrices",
    "pauli_kron",
    "batch_pauli_kron",
    # Expectation Values
    "batch_expectation",
    "batch_multi_expectation",
    "fast_z_readout",
    "fast_x_readout",
    "fast_y_readout",
    "simultaneous_readout",
    "hamiltonian_expectation",
    # Continuous Resonant Layer Operators
    "build_batch_resonant_hamiltonian",
    "ResonantInteractionBasis",
    "unitary_evolution",
    "ehrenfest_time_gradient",
    "daleckii_krein_spectral_derivative",
    # Gate Application
    "apply_gate",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "cnot_gate",
    "cz_gate",
]


class UnsupportedDtypeError(QuantaError, TypeError):
    """Raised when an unsupported dtype is requested for a specific device."""


# ── Global Caches ──
_PAULI_KRON_CACHE: dict[tuple[str, int, str, torch.dtype], torch.Tensor] = {}
_PAULI_MAT_CACHE: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Device and Dtype Resolvers
# ═══════════════════════════════════════════════════════════════════════════

def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolves target execution device. Defaults to Apple Silicon MPS if available, otherwise CPU.

    Args:
        device: Device string or torch.device, or None.

    Returns:
        Resolved torch.device.
    """
    if device is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def resolve_complex_dtype(
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.dtype:
    """Resolves complex dtype with defensive validation for Apple Silicon MPS.

    Args:
        dtype: Requested complex or floating-point dtype.
        device: Target execution device.

    Returns:
        Appropriate complex torch.dtype (torch.complex64 or torch.complex128).

    Raises:
        UnsupportedDtypeError: If 64-bit precision is requested on Apple Silicon MPS.
    """
    dev = resolve_device(device)
    if dtype is None:
        return torch.complex64
    if dev.type == "mps" and dtype in (torch.complex128, torch.float64):
        raise UnsupportedDtypeError(
            f"Apple Silicon MPS does not support 64-bit precision ({dtype}). "
            f"Use torch.complex64 on MPS or switch execution to device='cpu'."
        )
    if dtype in (torch.float32, torch.complex64):
        return torch.complex64
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return dtype


def get_real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    """Maps a complex dtype to its corresponding real scalar floating-point dtype.

    Args:
        complex_dtype: Complex dtype (torch.complex64 or torch.complex128).

    Returns:
        torch.float32 or torch.float64.
    """
    if complex_dtype == torch.complex64:
        return torch.float32
    if complex_dtype == torch.complex128:
        return torch.float64
    raise ValueError(f"Unsupported complex dtype: {complex_dtype}")


def get_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Maps a real scalar floating-point dtype to its matching complex dtype.

    Args:
        real_dtype: Real dtype (torch.float32 or torch.float64).

    Returns:
        torch.complex64 or torch.complex128.
    """
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise ValueError(f"Unsupported real dtype: {real_dtype}")


def to_torch_state(
    state: np.ndarray[Any, Any] | torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Converts a NumPy array or PyTorch tensor into a normalized complex PyTorch statevector.

    Args:
        state: Input array or tensor.
        device: Target execution device.
        dtype: Target complex dtype.
        normalize: Whether to normalize the statevector to unit norm.

    Returns:
        Contiguous complex PyTorch tensor on target device.
    """
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)

    if isinstance(state, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(state)).to(device=dev, dtype=c_dtype)
    elif isinstance(state, torch.Tensor):
        tensor = state.to(device=dev, dtype=c_dtype)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(state)}")

    if normalize:
        norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
        tensor = tensor / torch.clamp(norm, min=1e-12)

    return tensor.contiguous()


def to_numpy_state(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    """Converts a PyTorch complex statevector into a contiguous NumPy array on CPU.

    Args:
        tensor: PyTorch statevector.

    Returns:
        Contiguous complex NumPy array.
    """
    detached = tensor.detach().cpu()
    if detached.is_conj():
        detached = detached.resolve_conj()
    return detached.numpy()  # type: ignore[no-any-return]


def create_initial_state(
    initial_state: str | torch.Tensor | np.ndarray[Any, Any] = "zero",
    num_qubits: int = 1,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Generates a standard or custom initial statevector of shape (2^N,) or (B, 2^N).

    Args:
        initial_state: Initial state mode ('zero', 'plus', 'ghz', 'bell') or tensor/array.
        num_qubits: Number of qubits in the register.
        batch_size: Optional batch size dimension to broadcast.
        device: Target execution device.
        dtype: Target complex dtype.

    Returns:
        Complex statevector tensor.
    """
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    dim = 2 ** num_qubits

    if isinstance(initial_state, str):
        mode = initial_state.lower()
        if mode in ("zero", "0"):
            state = torch.zeros(dim, dtype=c_dtype, device=dev)
            state[0] = 1.0
        elif mode in ("plus", "+"):
            state = torch.ones(dim, dtype=c_dtype, device=dev) / math.sqrt(dim)
        elif mode in ("ghz", "bell"):
            state = torch.zeros(dim, dtype=c_dtype, device=dev)
            state[0] = 1.0 / math.sqrt(2.0)
            state[-1] = 1.0 / math.sqrt(2.0)
        else:
            raise ValueError(f"Unknown initial state mode: {initial_state!r}")
    else:
        state = to_torch_state(initial_state, device=dev, dtype=c_dtype, normalize=True)

    if batch_size is not None and batch_size > 0 and state.dim() == 1:
        state = state.unsqueeze(0).expand(batch_size, -1).contiguous()

    return state


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pauli Algebra & Multi-Qubit Kronecker Products
# ═══════════════════════════════════════════════════════════════════════════

def get_pauli_matrix(
    name: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Returns a 2x2 complex PyTorch tensor for a single Pauli or ladder operator.

    Supported names: 'I', 'X', 'Y', 'Z', '+', '-', 'H'.

    Args:
        name: Operator name.
        device: Target execution device.
        dtype: Target complex dtype.

    Returns:
        Cached 2x2 complex tensor.
    """
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    key = name.upper()
    cache_key = (key, str(dev), c_dtype)

    cached = _PAULI_MAT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if key == "I":
        mat = torch.eye(2, dtype=c_dtype, device=dev)
    elif key == "X":
        mat = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=c_dtype, device=dev)
    elif key == "Y":
        mat = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=c_dtype, device=dev)
    elif key == "Z":
        mat = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=c_dtype, device=dev)
    elif key in ("+", "PLUS"):
        mat = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=c_dtype, device=dev)
    elif key in ("-", "MINUS"):
        mat = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=c_dtype, device=dev)
    elif key == "H":
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        mat = torch.tensor(
            [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]],
            dtype=c_dtype,
            device=dev,
        )
    else:
        raise ValueError(f"Unknown Pauli operator name: {name!r}")

    res = mat.contiguous()
    _PAULI_MAT_CACHE[cache_key] = res
    return res


def pauli_matrices(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Returns a dictionary containing all single-qubit Pauli and ladder matrices."""
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    return {
        name: get_pauli_matrix(name, dev, c_dtype)
        for name in ("I", "X", "Y", "Z", "+", "-", "H")
    }


def pauli_kron(
    pauli_str: str,
    num_qubits: int | None = None,
    target_qubit: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes the multi-qubit Kronecker tensor product for a Pauli string with LRU caching.

    Args:
        pauli_str: Pauli descriptor (e.g. 'ZZ', 'IXY', 'Z').
        num_qubits: Total number of qubits in the system.
        target_qubit: If specified, embeds a single Pauli operator at target_qubit with I padding.
        device: Target execution device.
        dtype: Target complex dtype.

    Returns:
        Hermitian matrix of shape (2^N, 2^N).
    """
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)

    if target_qubit is not None:
        if num_qubits is None:
            raise ValueError("num_qubits must be specified when target_qubit is provided")
        if target_qubit < 0 or target_qubit >= num_qubits:
            raise ValueError(f"target_qubit {target_qubit} out of range [0, {num_qubits - 1}]")
        chars = ["I"] * num_qubits
        chars[target_qubit] = pauli_str.upper()
        canonical_str = "".join(chars)
    else:
        canonical_str = pauli_str.upper()
        if num_qubits is not None:
            if len(canonical_str) < num_qubits:
                canonical_str = canonical_str.ljust(num_qubits, "I")
            elif len(canonical_str) > num_qubits:
                raise ValueError(f"pauli_str '{pauli_str}' exceeds num_qubits={num_qubits}")

    cache_key = (canonical_str, len(canonical_str), str(dev), c_dtype)
    cached = _PAULI_KRON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    paulis = pauli_matrices(dev, c_dtype)
    res = paulis[canonical_str[0]]
    for ch in canonical_str[1:]:
        res = torch.kron(res, paulis[ch])

    res = res.contiguous()
    _PAULI_KRON_CACHE[cache_key] = res
    return res


def batch_pauli_kron(
    pauli_strs: Sequence[str],
    num_qubits: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Constructs and stacks multiple multi-qubit Pauli operators into shape (M, 2^N, 2^N)."""
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    matrices = [pauli_kron(s, num_qubits=num_qubits, device=dev, dtype=c_dtype) for s in pauli_strs]
    return torch.stack(matrices, dim=0).contiguous()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Expectation Value Evaluation Engine
# ═══════════════════════════════════════════════════════════════════════════

def batch_expectation(
    statevectors: torch.Tensor,
    observable: torch.Tensor,
) -> torch.Tensor:
    """Computes <psi|O|psi> for statevectors of shape (B, 2^N) or (2^N,) and observable (2^N, 2^N).

    Args:
        statevectors: Statevector tensor of shape (2^N,), (B, 2^N), or (*batch, 2^N).
        observable: Hermitian matrix of shape (2^N, 2^N).

    Returns:
        Real expectation values matching the batch dimensions of statevectors.
    """
    is_1d = statevectors.dim() == 1
    orig_shape = statevectors.shape
    dim = observable.shape[0]

    if is_1d:
        statevectors = statevectors.unsqueeze(0)
    elif statevectors.dim() > 2:
        statevectors = statevectors.reshape(-1, dim)

    if observable.device != statevectors.device:
        observable = observable.to(device=statevectors.device)

    # (B, D) @ (D, D) -> (B, D)
    O_psi = torch.matmul(statevectors, observable.T)
    exp_vals = torch.real(torch.sum(statevectors.conj() * O_psi, dim=-1))

    if is_1d:
        return exp_vals.squeeze(0)
    if len(orig_shape) > 2:
        return exp_vals.reshape(orig_shape[:-1])
    return exp_vals


def batch_multi_expectation(
    statevectors: torch.Tensor,
    observables: torch.Tensor,
) -> torch.Tensor:
    """Fuses expectation evaluations for M observables (M, 2^N, 2^N) and batch states (B, 2^N).

    Uses Apple Silicon Metal GEMM-accelerated batched matrix multiplication.

    Args:
        statevectors: Complex statevectors of shape (B, 2^N) or (2^N,).
        observables: Stack of Hermitian matrices of shape (M, 2^N, 2^N).

    Returns:
        Real tensor of shape (B, M) or (M,).
    """
    is_1d = statevectors.dim() == 1
    if is_1d:
        statevectors = statevectors.unsqueeze(0)

    if observables.device != statevectors.device:
        observables = observables.to(device=statevectors.device)

    # (M, D, D) @ (D, B) -> (M, D, B)
    O_psi = torch.matmul(observables, statevectors.T)
    # (M, D, B) -> (B, M, D)
    O_psi = O_psi.permute(2, 0, 1)
    # Sum over state dimension D: (B, 1, D) * (B, M, D) -> (B, M)
    res = torch.real(torch.sum(statevectors.unsqueeze(1).conj() * O_psi, dim=-1))

    return res.squeeze(0) if is_1d else res


def fast_z_readout(
    statevectors: torch.Tensor,
    num_qubits: int,
    target_qubits: Sequence[int] | None = None,
) -> torch.Tensor:
    """Computes expectations of local Z_j observables in O(2^N) time using basis probabilities.

    Zero matrix allocations, zero Kronecker expansions.

    Args:
        statevectors: Complex statevectors of shape (B, 2^N) or (2^N,).
        num_qubits: Total qubits N.
        target_qubits: Qubit indices to read out (defaults to all qubits [0..N-1]).

    Returns:
        Real tensor of shape (B, M) or (M,) where M = len(target_qubits).
    """
    is_1d = statevectors.dim() == 1
    if is_1d:
        statevectors = statevectors.unsqueeze(0)

    B, D = statevectors.shape
    device = statevectors.device
    r_dtype = get_real_dtype(statevectors.dtype)

    if target_qubits is None:
        target_qubits = list(range(num_qubits))

    indices = torch.arange(D, device=device)
    bits = torch.stack([(indices >> (num_qubits - 1 - q)) & 1 for q in target_qubits], dim=0)
    signs = (1.0 - 2.0 * bits.to(r_dtype)).T  # (D, M)

    probs = torch.real(statevectors * statevectors.conj())  # (B, D)
    res = torch.matmul(probs, signs)  # (B, M)

    return res.squeeze(0) if is_1d else res


def fast_x_readout(
    statevectors: torch.Tensor,
    num_qubits: int,
    target_qubit: int,
) -> torch.Tensor:
    """Computes single-qubit X_j expectation in O(2^N) time via orthogonal tensor slicing.

    Args:
        statevectors: Complex statevectors of shape (B, 2^N) or (2^N,).
        num_qubits: Total qubits N.
        target_qubit: Qubit index to measure.

    Returns:
        Real expectation values of shape (B,) or scalar.
    """
    is_1d = statevectors.dim() == 1
    if is_1d:
        statevectors = statevectors.unsqueeze(0)

    B = statevectors.shape[0]
    state_tensor = statevectors.reshape([B] + [2] * num_qubits)
    slice0 = state_tensor.select(target_qubit + 1, 0)
    slice1 = state_tensor.select(target_qubit + 1, 1)
    prod = slice0.conj() * slice1

    if num_qubits > 1:
        sum_dims = tuple(range(1, num_qubits))
        res = 2.0 * torch.real(torch.sum(prod, dim=sum_dims))
    else:
        res = 2.0 * torch.real(prod)

    return res.squeeze(0) if is_1d else res


def fast_y_readout(
    statevectors: torch.Tensor,
    num_qubits: int,
    target_qubit: int,
) -> torch.Tensor:
    """Computes single-qubit Y_j expectation in O(2^N) time via orthogonal tensor slicing.

    Args:
        statevectors: Complex statevectors of shape (B, 2^N) or (2^N,).
        num_qubits: Total qubits N.
        target_qubit: Qubit index to measure.

    Returns:
        Real expectation values of shape (B,) or scalar.
    """
    is_1d = statevectors.dim() == 1
    if is_1d:
        statevectors = statevectors.unsqueeze(0)

    B = statevectors.shape[0]
    state_tensor = statevectors.reshape([B] + [2] * num_qubits)
    slice0 = state_tensor.select(target_qubit + 1, 0)
    slice1 = state_tensor.select(target_qubit + 1, 1)
    prod = slice0.conj() * slice1

    if num_qubits > 1:
        sum_dims = tuple(range(1, num_qubits))
        res = 2.0 * torch.imag(torch.sum(prod, dim=sum_dims))
    else:
        res = 2.0 * torch.imag(prod)

    return res.squeeze(0) if is_1d else res


def simultaneous_readout(
    statevectors: torch.Tensor,
    num_nodes: int,
    observable_types: tuple[str, ...] = ("Z", "X"),
) -> torch.Tensor:
    """Performs concurrent multi-observable readout across all nodes in O(2^N) time.

    Args:
        statevectors: Statevector tensor of shape (B, 2^N) or (2^N,).
        num_nodes: Number of qubits/nodes N.
        observable_types: Tuple of observable types ('Z', 'X', 'Y').

    Returns:
        Tensor of shape (B, num_nodes * len(observable_types)) or
        (num_nodes * len(observable_types),).
    """
    is_1d = statevectors.dim() == 1
    if is_1d:
        statevectors = statevectors.unsqueeze(0)

    outputs: list[torch.Tensor] = []
    for obs_type in observable_types:
        obs = obs_type.upper()
        if obs == "Z":
            outputs.append(fast_z_readout(statevectors, num_nodes))
        elif obs == "X":
            x_vals = [
                fast_x_readout(statevectors, num_nodes, j).unsqueeze(-1)
                for j in range(num_nodes)
            ]
            outputs.append(torch.cat(x_vals, dim=-1))
        elif obs == "Y":
            y_vals = [
                fast_y_readout(statevectors, num_nodes, j).unsqueeze(-1)
                for j in range(num_nodes)
            ]
            outputs.append(torch.cat(y_vals, dim=-1))
        else:
            raise ValueError(f"Unsupported observable type for fast readout: {obs_type!r}")

    res = torch.cat(outputs, dim=-1)
    return res.squeeze(0) if is_1d else res


def hamiltonian_expectation(
    statevectors: torch.Tensor,
    terms: Sequence[tuple[str, float]],
    num_qubits: int,
) -> torch.Tensor:
    """Evaluates the expectation of a linear combination of Pauli strings: sum c_k <O_k>.

    Args:
        statevectors: Complex statevectors of shape (B, 2^N) or (2^N,).
        terms: Sequence of (pauli_string, coeff) pairs.
        num_qubits: Total number of qubits.

    Returns:
        Expectation values of shape (B,) or scalar.
    """
    is_1d = statevectors.dim() == 1
    if is_1d:
        statevectors = statevectors.unsqueeze(0)

    dev = statevectors.device
    r_dtype = get_real_dtype(statevectors.dtype)
    total = torch.zeros(statevectors.shape[0], dtype=r_dtype, device=dev)

    for pauli_str, coeff in terms:
        obs_mat = pauli_kron(pauli_str, num_qubits=num_qubits, device=dev, dtype=statevectors.dtype)
        total += coeff * batch_expectation(statevectors, obs_mat)

    return total.squeeze(0) if is_1d else total


# ═══════════════════════════════════════════════════════════════════════════
# 4. Continuous Resonant Layer Operators
# ═══════════════════════════════════════════════════════════════════════════

class ResonantInteractionBasis:
    """Precomputed device-resident interaction basis operators for graph Hamiltonians."""

    def __init__(
        self,
        num_nodes: int,
        edges: list[tuple[int, int]],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.edges = edges
        self.device = resolve_device(device)
        self.dtype = resolve_complex_dtype(dtype, self.device)
        self.dim = 2 ** num_nodes

        # Precompute XY interaction matrices: sigma_j^x sigma_k^x + sigma_j^y sigma_k^y
        self.xy_matrices: list[torch.Tensor] = []
        for j, k in edges:
            xj = pauli_kron(
                "X", num_qubits=num_nodes, target_qubit=j, device=self.device, dtype=self.dtype
            )
            xk = pauli_kron(
                "X", num_qubits=num_nodes, target_qubit=k, device=self.device, dtype=self.dtype
            )
            yj = pauli_kron(
                "Y", num_qubits=num_nodes, target_qubit=j, device=self.device, dtype=self.dtype
            )
            yk = pauli_kron(
                "Y", num_qubits=num_nodes, target_qubit=k, device=self.device, dtype=self.dtype
            )
            self.xy_matrices.append(xj @ xk + yj @ yk)

        # Precompute local Z_j and X_j matrices
        self.z_matrices: list[torch.Tensor] = [
            pauli_kron(
                "Z", num_qubits=num_nodes, target_qubit=j, device=self.device, dtype=self.dtype
            )
            for j in range(num_nodes)
        ]
        self.x_matrices: list[torch.Tensor] = [
            pauli_kron(
                "X", num_qubits=num_nodes, target_qubit=j, device=self.device, dtype=self.dtype
            )
            for j in range(num_nodes)
        ]

        # Precompute longitudinal signs for diagonal modulation
        r_dtype = get_real_dtype(self.dtype)
        indices = torch.arange(self.dim, device=self.device)
        bits = torch.stack([(indices >> (num_nodes - 1 - j)) & 1 for j in range(num_nodes)], dim=0)
        self.longitudinal_signs = 1.0 - 2.0 * bits.to(r_dtype)  # (N, dim)


def build_batch_resonant_hamiltonian(
    x: torch.Tensor,
    J: torch.Tensor,
    edges: list[tuple[int, int]],
    h: torch.Tensor,
    W: torch.Tensor,
    omega: torch.Tensor,
    num_nodes: int,
    basis: ResonantInteractionBasis | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Builds batched continuous network Hamiltonians H(x, theta) of shape (B, 2^N, 2^N).

    Uses vectorized diagonal injection to eliminate redundant Kronecker allocations.
    """
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    B = x.shape[0]
    dim = 2 ** num_nodes

    if basis is None:
        basis = ResonantInteractionBasis(num_nodes, edges, dev, c_dtype)

    # 1. Static component
    H_static = torch.zeros((dim, dim), dtype=c_dtype, device=dev)
    for j_val, xy_mat in zip(J, basis.xy_matrices, strict=False):
        H_static += j_val * xy_mat
    for j in range(num_nodes):
        H_static += h[j] * basis.z_matrices[j]
        H_static += omega[j] * basis.x_matrices[j]

    # 2. Vectorized diagonal input modulation
    # v_batch: (B, N)
    v_batch = torch.matmul(x, W.T)
    # diag_batch: (B, dim)
    diag_batch = torch.matmul(v_batch, basis.longitudinal_signs)

    # Broadcast and add diagonal
    H_batch = H_static.unsqueeze(0).repeat(B, 1, 1)
    diag_indices = torch.arange(dim, device=dev)
    H_batch[:, diag_indices, diag_indices] += diag_batch.to(c_dtype)

    return H_batch


def unitary_evolution(
    H: torch.Tensor,
    t: torch.Tensor | float,
    psi0: torch.Tensor,
) -> torch.Tensor:
    """Performs unitary matrix exponential state evolution: |psi(t)> = exp(-i H t) |psi0>.

    Uses complex128 precision for continuous spectral evolution to eliminate numerical
    norm drift and guarantee machine-precision unitarity.
    """
    orig_dtype = psi0.dtype
    target_device = H.device
    if psi0.device != target_device:
        psi0 = psi0.to(device=target_device)

    # Resolve calculation precision (complex128 on CPU/CUDA, complex64 on MPS)
    if target_device.type != "mps":
        c_calc_dtype = torch.complex128
        r_calc_dtype = torch.float64
    else:
        c_calc_dtype = torch.complex64
        r_calc_dtype = torch.float32

    # Scale/format t
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            t_scaled: float | torch.Tensor = t.item()
        elif t.dim() == 1:
            t_scaled = t.to(dtype=r_calc_dtype).unsqueeze(-1).unsqueeze(-1)
        else:
            t_scaled = t.to(dtype=r_calc_dtype)
    else:
        t_scaled = float(t)

    H_calc = H.to(dtype=c_calc_dtype)

    # Check if H is Hermitian for exact spectral evolution: H = V diag(lambda) V^H
    is_hermitian = False
    if (H.dim() == 2 and H.shape[0] == H.shape[1]) or (
        H.dim() == 3 and H.shape[1] == H.shape[2]
    ):
        is_hermitian = bool(torch.allclose(H_calc, H_calc.mH, atol=1e-5))

    if is_hermitian:
        H_herm = (H_calc + H_calc.mH) / 2.0
        evals, evecs = torch.linalg.eigh(H_herm)
        if isinstance(t_scaled, torch.Tensor):
            t_flat = t_scaled
            while t_flat.dim() > 2:
                t_flat = t_flat.squeeze(-1)
            if t_flat.dim() == 0:
                phases = torch.exp(-1.0j * evals * t_flat.item())
            elif t_flat.dim() == 1 and evals.dim() == 2:
                phases = torch.exp(-1.0j * evals * t_flat.unsqueeze(-1))
            else:
                phases = torch.exp(-1.0j * evals * t_flat)
        else:
            phases = torch.exp(-1.0j * evals * t_scaled)
        U = evecs @ torch.diag_embed(phases) @ evecs.mH
    else:
        U = torch.linalg.matrix_exp(-1.0j * H_calc * t_scaled)

    psi0_calc = psi0.to(dtype=c_calc_dtype)

    if psi0.dim() == 1:
        if H.dim() == 3:
            B = H.shape[0]
            psi_exp = psi0_calc.unsqueeze(0).expand(B, -1).unsqueeze(-1)
            psi_t = torch.matmul(U, psi_exp).squeeze(-1)
        else:
            psi_t = torch.matmul(U, psi0_calc)
    elif psi0.dim() == 2:
        psi_exp = psi0_calc.unsqueeze(-1)
        psi_t = torch.matmul(U, psi_exp).squeeze(-1)
    else:
        raise ValueError(f"Unsupported psi0 dimension: {psi0.dim()}")

    return psi_t.to(dtype=orig_dtype).contiguous()


def ehrenfest_time_gradient(
    psi_t: torch.Tensor,
    H: torch.Tensor,
    observable: torch.Tensor,
) -> torch.Tensor:
    """Computes exact time derivative d<O>/dt via the Ehrenfest theorem in O(2^N) time.

    Formula: +2 * Im[ <psi(t)| O H |psi(t)> ].
    """
    is_1d = psi_t.dim() == 1
    if is_1d:
        psi_t = psi_t.unsqueeze(0)

    if H.device != psi_t.device:
        H = H.to(device=psi_t.device)
    if observable.device != psi_t.device:
        observable = observable.to(device=psi_t.device)

    if H.dim() == 2:
        # H @ psi.T -> (D, B) -> (B, D)
        H_psi = torch.matmul(psi_t, H.T)
    else:
        H_psi = torch.matmul(H, psi_t.unsqueeze(-1)).squeeze(-1)

    O_H_psi = torch.matmul(H_psi, observable.T)
    z = torch.sum(psi_t.conj() * O_H_psi, dim=-1)
    grad_t = 2.0 * torch.imag(z)

    return grad_t.squeeze(0) if is_1d else grad_t


def daleckii_krein_spectral_derivative(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    t: float | torch.Tensor,
    omega_matrix: torch.Tensor,
) -> torch.Tensor:
    """Computes Daleckii-Krein matrix spectral Fréchet derivative with stable sinc parameterization.

    Returns d(exp(-i H t)) / d phi = V [ (V^dagger Omega V) * M(t) ] V^dagger.
    """
    orig_dtype = omega_matrix.dtype
    dev = omega_matrix.device
    if dev.type != "mps":
        c_calc_dtype = torch.complex128
        r_calc_dtype = torch.float64
    else:
        c_calc_dtype = torch.complex64
        r_calc_dtype = torch.float32

    evals = eigenvalues.to(dtype=r_calc_dtype)
    evecs = eigenvectors.to(dtype=c_calc_dtype)
    omega = omega_matrix.to(dtype=c_calc_dtype)
    t_val = t.to(dtype=r_calc_dtype) if isinstance(t, torch.Tensor) else float(t)

    diff = evals.unsqueeze(-1) - evals.unsqueeze(-2)
    mean = (evals.unsqueeze(-1) + evals.unsqueeze(-2)) / 2.0

    arg = (diff * t_val) / (2.0 * math.pi)
    sinc_val = torch.sinc(arg).to(dtype=c_calc_dtype)
    phase = torch.exp(-1.0j * mean * t_val).to(dtype=c_calc_dtype)
    M = -1.0j * t_val * phase * sinc_val

    omega_tilde = evecs.mH @ omega @ evecs
    dU_tilde = omega_tilde * M
    dU = evecs @ dU_tilde @ evecs.mH

    return dU.to(dtype=orig_dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Batched PyTorch Native Quantum Gate Simulation
# ═══════════════════════════════════════════════════════════════════════════

def apply_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubits: tuple[int, ...],
    num_qubits: int,
) -> torch.Tensor:
    """Applies a k-qubit unitary gate to batched statevectors via tensor contraction.

    Args:
        state: Statevectors of shape (B, 2^N) or (2^N,).
        gate: Unitary matrix of shape (2^k, 2^k).
        qubits: Target qubit indices.
        num_qubits: Total number of qubits in the register.

    Returns:
        Transformed statevectors of matching shape.
    """
    is_1d = state.dim() == 1
    if is_1d:
        state = state.unsqueeze(0)

    if gate.device != state.device:
        gate = gate.to(device=state.device)

    B = state.shape[0]
    k = len(qubits)
    state_tensor = state.reshape([B] + [2] * num_qubits)
    gate_tensor = gate.reshape([2] * (2 * k))

    gate_axes = list(range(k, 2 * k))
    state_axes = [q + 1 for q in qubits]

    res = torch.tensordot(gate_tensor, state_tensor, dims=(gate_axes, state_axes))

    # Permute axes back: batch at 0, qubits at 1..N
    perm = [0] * (num_qubits + 1)
    perm[0] = k
    curr_uncontracted = k + 1
    for q in range(num_qubits):
        state_ax = q + 1
        if q in qubits:
            perm[state_ax] = qubits.index(q)
        else:
            perm[state_ax] = curr_uncontracted
            curr_uncontracted += 1

    res_flat: torch.Tensor = res.permute(perm).contiguous().reshape(B, -1)
    if is_1d:
        out: torch.Tensor = res_flat.squeeze(0)
        return out
    return res_flat


def rotation_x(
    theta: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Constructs parameterized single-qubit RX rotation: cos(theta/2) I - i sin(theta/2) X."""
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    I_mat = get_pauli_matrix("I", dev, c_dtype)
    X_mat = get_pauli_matrix("X", dev, c_dtype)
    return c * I_mat - 1.0j * s * X_mat


def rotation_y(
    theta: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Constructs parameterized single-qubit RY rotation: cos(theta/2) I - i sin(theta/2) Y."""
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    I_mat = get_pauli_matrix("I", dev, c_dtype)
    Y_mat = get_pauli_matrix("Y", dev, c_dtype)
    return c * I_mat - 1.0j * s * Y_mat


def rotation_z(
    theta: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Constructs parameterized single-qubit RZ rotation: cos(theta/2) I - i sin(theta/2) Z."""
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    I_mat = get_pauli_matrix("I", dev, c_dtype)
    Z_mat = get_pauli_matrix("Z", dev, c_dtype)
    return c * I_mat - 1.0j * s * Z_mat


def cnot_gate(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Constructs standard 2-qubit CNOT (CX) unitary matrix."""
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    cx = torch.zeros((4, 4), dtype=c_dtype, device=dev)
    cx[0, 0] = 1.0
    cx[1, 1] = 1.0
    cx[2, 3] = 1.0
    cx[3, 2] = 1.0
    return cx


def cz_gate(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Constructs standard 2-qubit Controlled-Z (CZ) unitary matrix."""
    dev = resolve_device(device)
    c_dtype = resolve_complex_dtype(dtype, dev)
    cz = torch.zeros((4, 4), dtype=c_dtype, device=dev)
    cz[0, 0] = 1.0
    cz[1, 1] = 1.0
    cz[2, 2] = 1.0
    cz[3, 3] = -1.0
    return cz
