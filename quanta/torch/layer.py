"""quanta.torch.layer — Variational Quantum Circuit Layer for PyTorch.

Provides QuantumLayer(nn.Module), an autograd-differentiable quantum circuit
layer supporting arbitrary variational ansatzes, flexible Pauli observable readouts,
and exact analytical gradients via the Parameter-Shift Rule.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from quanta.torch import ops

__all__ = [
    "QuantumLayer",
    "ParsedObservable",
    "PauliTerm",
    "ObservableParser",
    "_QuantumLayerFunction",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Observable Data Structures & Parser
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PauliTerm:
    """A single Pauli tensor product with a scalar coefficient.

    Attributes:
        pauli_string: String of length num_qubits, e.g. "ZII", "IXY".
        coeff: Real coefficient.
    """

    pauli_string: str
    coeff: float = 1.0


@dataclass(frozen=True)
class ParsedObservable:
    """A quantum observable composed of one or more Pauli terms.

    Attributes:
        terms: Tuple of Pauli terms summing to the observable H = Σ c_i P_i.
        name: Human-readable display label.
    """

    terms: tuple[PauliTerm, ...]
    name: str


class ObservableParser:
    """Compiles diverse user observable inputs into canonical ParsedObservables."""

    VALID_PAULIS = frozenset({"I", "X", "Y", "Z"})

    @classmethod
    def parse_all(
        cls,
        observables: Sequence[Any] | None,
        num_qubits: int,
    ) -> list[ParsedObservable]:
        """Parse observable specifications into canonical form.

        Args:
            observables: None, list of strings, list of tuples, or nested lists.
            num_qubits: Number of qubits in the circuit.

        Returns:
            List of ParsedObservable instances.
        """
        if observables is None:
            # Default: Single-qubit Z on all qubits: [Z0, Z1, ..., Z_{N-1}]
            return [
                ParsedObservable(
                    terms=(
                        PauliTerm(
                            pauli_string="I" * q + "Z" + "I" * (num_qubits - q - 1),
                            coeff=1.0,
                        ),
                    ),
                    name=f"Z{q}",
                )
                for q in range(num_qubits)
            ]

        if not observables:
            raise ValueError("Observables list cannot be empty.")

        # Check if user passed a single composite Hamiltonian: list of (str, float) tuples
        first = observables[0]
        if (
            isinstance(first, tuple)
            and len(first) == 2
            and isinstance(first[0], str)
            and isinstance(first[1], (int, float))
        ):
            # Single composite observable
            composite_terms = tuple(
                cls._parse_term(term_str, float(coeff), num_qubits)
                for term_str, coeff in observables
            )
            return [ParsedObservable(terms=composite_terms, name="H_0")]

        parsed_list: list[ParsedObservable] = []
        for idx, obs in enumerate(observables):
            if isinstance(obs, str):
                term = cls._parse_term(obs, 1.0, num_qubits)
                parsed_list.append(ParsedObservable(terms=(term,), name=obs))
            elif (
                isinstance(obs, tuple)
                and len(obs) == 2
                and isinstance(obs[0], str)
                and isinstance(obs[1], (int, float))
            ):
                term = cls._parse_term(obs[0], float(obs[1]), num_qubits)
                parsed_list.append(
                    ParsedObservable(terms=(term,), name=f"{obs[0]}*{obs[1]}")
                )
            elif isinstance(obs, (list, tuple)):
                # Composite multi-term observable
                if not obs:
                    raise ValueError("Composite observable cannot be empty.")
                terms = tuple(
                    cls._parse_term(t_str, float(c), num_qubits)
                    for t_str, c in obs
                )
                parsed_list.append(
                    ParsedObservable(terms=terms, name=f"Obs_{idx}")
                )
            else:
                raise TypeError(
                    f"Unsupported observable type at index {idx}: {type(obs)}. "
                    f"Expected str, tuple[str, float], or list of tuples."
                )

        return parsed_list

    @classmethod
    def _parse_term(cls, raw: str, coeff: float, num_qubits: int) -> PauliTerm:
        """Parses a Pauli string term (full string or indexed shorthand)."""
        raw = raw.strip().upper()
        if not raw:
            raise ValueError("Observable descriptor string cannot be empty or whitespace.")

        # Case 1: Full canonical Pauli string of exact length num_qubits
        if len(raw) == num_qubits and all(ch in cls.VALID_PAULIS for ch in raw):
            return PauliTerm(pauli_string=raw, coeff=coeff)

        # Case 2: Indexed notation, e.g. "Z0", "X1", "Z0 Z1", "X0 Y1"
        chars = list("I" * num_qubits)
        tokens = raw.split()
        pattern = re.compile(r"^([XYZI])(\d+)$")

        for token in tokens:
            match = pattern.match(token)
            if not match:
                raise ValueError(
                    f"Invalid Pauli descriptor '{token}'. Must be full string "
                    f"of length {num_qubits} (e.g. 'ZIZ') or indexed (e.g. 'Z0', 'X1')."
                )
            pauli_ch, q_str = match.groups()
            qubit_idx = int(q_str)
            if qubit_idx < 0 or qubit_idx >= num_qubits:
                raise ValueError(
                    f"Qubit index {qubit_idx} out of range [0, {num_qubits - 1}] "
                    f"in observable descriptor '{token}'."
                )
            chars[qubit_idx] = pauli_ch

        return PauliTerm(pauli_string="".join(chars), coeff=coeff)


# ═══════════════════════════════════════════════════════════════════════════
#  Ansatz Specifications
# ═══════════════════════════════════════════════════════════════════════════

def get_ansatz_param_count(ansatz: str, num_qubits: int, num_layers: int) -> int:
    """Compute total parameter count for built-in ansatz presets.

    Args:
        ansatz: Name of ansatz preset.
        num_qubits: Number of qubits.
        num_layers: Number of variational layers.

    Returns:
        Integer parameter count.
    """
    preset = ansatz.lower()
    if preset == "hardware_efficient":
        return 2 * num_qubits * num_layers
    elif preset == "strongly_entangling":
        return 3 * num_qubits * num_layers
    elif preset == "reuploading":
        return 1 * num_qubits * num_layers
    elif preset == "real_amplitudes":
        return num_qubits * (num_layers + 1)
    else:
        raise ValueError(
            f"Unknown ansatz preset '{ansatz}'. Supported presets: "
            f"'hardware_efficient', 'strongly_entangling', 'reuploading', 'real_amplitudes'."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Autograd Function (_QuantumLayerFunction)
# ═══════════════════════════════════════════════════════════════════════════

class _QuantumLayerFunction(torch.autograd.Function):
    """Custom autograd Function executing batch quantum circuits with analytical Parameter-Shift."""

    @staticmethod
    def _simulate_batch(
        x: torch.Tensor,
        weights: torch.Tensor,
        num_qubits: int,
        num_layers: int,
        circuit_fn: Callable[..., Any] | str,
        parsed_observables: list[ParsedObservable],
        encoding: str = "angle",
    ) -> torch.Tensor:
        """Executes forward batch circuit simulation and returns expectation tensor (B, D_out)."""
        is_layerwise = (x.ndim == 3)
        B = x.shape[1] if is_layerwise else x.shape[0]
        N = num_qubits
        device = x.device
        complex_dtype = ops.resolve_complex_dtype(ops.get_complex_dtype(x.dtype), device)
        real_dtype = ops.get_real_dtype(complex_dtype)

        # Short-circuit empty batch (B=0)
        if B == 0:
            return torch.empty(
                (0, len(parsed_observables)),
                dtype=real_dtype,
                device=device,
            )

        # 1. Custom callable path
        if callable(circuit_fn):
            res = circuit_fn(x, weights, num_qubits, num_layers, parsed_observables)
            if isinstance(res, torch.Tensor):
                if res.shape[-1] == len(parsed_observables) and res.ndim == 2:
                    return res.to(dtype=real_dtype, device=device)
                # If callable returns a state tensor, continue to observable readout
                state = res.to(dtype=complex_dtype, device=device)
            else:
                raise TypeError(f"Custom circuit_fn returned unexpected type: {type(res)}")
        else:
            # Initialize batch ground state |0...0>
            state = torch.zeros((B,) + (2,) * N, dtype=complex_dtype, device=device)
            state[(slice(None),) + (0,) * N] = 1.0

            ansatz_name = circuit_fn.lower()

            # 2. Classical Data Encoding (Angle encoding RY(pi * x_k))
            if ansatz_name != "reuploading" and encoding == "angle":
                for q in range(min(x.shape[1], N)):
                    angles = math.pi * x[:, q]
                    c = torch.cos(angles / 2.0).to(complex_dtype)
                    s = torch.sin(angles / 2.0).to(complex_dtype)
                    g_gate = torch.zeros((B, 2, 2), dtype=complex_dtype, device=device)
                    g_gate[:, 0, 0] = c
                    g_gate[:, 0, 1] = -s
                    g_gate[:, 1, 0] = s
                    g_gate[:, 1, 1] = c

                    s_perm = torch.moveaxis(state, q + 1, 1)
                    s_shape = s_perm.shape
                    out_bmm = torch.bmm(g_gate, s_perm.reshape(B, 2, -1))
                    state = torch.moveaxis(out_bmm.reshape(s_shape), 1, q + 1)

            # 3. Variational Circuit Layers
            cx_mat = ops.cnot_gate(device=device, dtype=complex_dtype).reshape(2, 2, 2, 2)

            if ansatz_name == "hardware_efficient":
                idx = 0
                for _ in range(num_layers):
                    for q in range(N):
                        # RY
                        ty = weights[idx]
                        idx += 1
                        cy = torch.cos(ty / 2.0).to(complex_dtype)
                        sy = torch.sin(ty / 2.0).to(complex_dtype)
                        ry = torch.tensor([[cy, -sy], [sy, cy]], dtype=complex_dtype, device=device)
                        state = torch.moveaxis(
                            torch.tensordot(ry, state, dims=([1], [q + 1])), 0, q + 1
                        )

                        # RZ
                        tz = weights[idx]
                        idx += 1
                        phase_pos = torch.exp(-1j * (tz / 2.0).to(complex_dtype))
                        phase_neg = torch.exp(1j * (tz / 2.0).to(complex_dtype))
                        rz = torch.diag(torch.stack([phase_pos, phase_neg]))
                        state = torch.moveaxis(
                            torch.tensordot(rz, state, dims=([1], [q + 1])), 0, q + 1
                        )

                    if N > 1:
                        for q in range(N - 1):
                            state = torch.moveaxis(
                                torch.tensordot(cx_mat, state, dims=([2, 3], [q + 1, q + 2])),
                                [0, 1],
                                [q + 1, q + 2],
                            )

            elif ansatz_name == "strongly_entangling":
                idx = 0
                for l_idx in range(num_layers):
                    for q in range(N):
                        # RX
                        tx = weights[idx]
                        idx += 1
                        cx_rot = torch.cos(tx / 2.0).to(complex_dtype)
                        sx_rot = torch.sin(tx / 2.0).to(complex_dtype)
                        rx = torch.tensor(
                            [[cx_rot, -1j * sx_rot], [-1j * sx_rot, cx_rot]],
                            dtype=complex_dtype,
                            device=device,
                        )
                        state = torch.moveaxis(
                            torch.tensordot(rx, state, dims=([1], [q + 1])), 0, q + 1
                        )

                        # RY
                        ty = weights[idx]
                        idx += 1
                        cy = torch.cos(ty / 2.0).to(complex_dtype)
                        sy = torch.sin(ty / 2.0).to(complex_dtype)
                        ry = torch.tensor([[cy, -sy], [sy, cy]], dtype=complex_dtype, device=device)
                        state = torch.moveaxis(
                            torch.tensordot(ry, state, dims=([1], [q + 1])), 0, q + 1
                        )

                        # RZ
                        tz = weights[idx]
                        idx += 1
                        phase_pos = torch.exp(-1j * (tz / 2.0).to(complex_dtype))
                        phase_neg = torch.exp(1j * (tz / 2.0).to(complex_dtype))
                        rz = torch.diag(torch.stack([phase_pos, phase_neg]))
                        state = torch.moveaxis(
                            torch.tensordot(rz, state, dims=([1], [q + 1])), 0, q + 1
                        )

                    if N > 1:
                        shift = l_idx % max(1, N - 1)
                        for q in range(N):
                            target = (q + shift + 1) % N
                            state = torch.moveaxis(
                                torch.tensordot(cx_mat, state, dims=([2, 3], [q + 1, target + 1])),
                                [0, 1],
                                [q + 1, target + 1],
                            )

            elif ansatz_name == "reuploading":
                idx = 0
                for l_idx in range(num_layers):
                    # Data re-uploading
                    xl = x[l_idx] if is_layerwise else x
                    for q in range(min(xl.shape[1], N)):
                        angles = math.pi * xl[:, q]
                        c = torch.cos(angles / 2.0).to(complex_dtype)
                        s = torch.sin(angles / 2.0).to(complex_dtype)
                        g_gate = torch.zeros((B, 2, 2), dtype=complex_dtype, device=device)
                        g_gate[:, 0, 0] = c
                        g_gate[:, 0, 1] = -s
                        g_gate[:, 1, 0] = s
                        g_gate[:, 1, 1] = c

                        s_perm = torch.moveaxis(state, q + 1, 1)
                        s_shape = s_perm.shape
                        out_bmm = torch.bmm(g_gate, s_perm.reshape(B, 2, -1))
                        state = torch.moveaxis(out_bmm.reshape(s_shape), 1, q + 1)

                    # Variational RY
                    for q in range(N):
                        ty = weights[idx]
                        idx += 1
                        cy = torch.cos(ty / 2.0).to(complex_dtype)
                        sy = torch.sin(ty / 2.0).to(complex_dtype)
                        ry = torch.tensor([[cy, -sy], [sy, cy]], dtype=complex_dtype, device=device)
                        state = torch.moveaxis(
                            torch.tensordot(ry, state, dims=([1], [q + 1])), 0, q + 1
                        )

                    if N > 1:
                        for q in range(N - 1):
                            state = torch.moveaxis(
                                torch.tensordot(cx_mat, state, dims=([2, 3], [q + 1, q + 2])),
                                [0, 1],
                                [q + 1, q + 2],
                            )

            elif ansatz_name == "real_amplitudes":
                idx = 0
                for _ in range(num_layers):
                    for q in range(N):
                        ty = weights[idx]
                        idx += 1
                        cy = torch.cos(ty / 2.0).to(complex_dtype)
                        sy = torch.sin(ty / 2.0).to(complex_dtype)
                        ry = torch.tensor([[cy, -sy], [sy, cy]], dtype=complex_dtype, device=device)
                        state = torch.moveaxis(
                            torch.tensordot(ry, state, dims=([1], [q + 1])), 0, q + 1
                        )

                    if N > 1:
                        for q in range(N - 1):
                            state = torch.moveaxis(
                                torch.tensordot(cx_mat, state, dims=([2, 3], [q + 1, q + 2])),
                                [0, 1],
                                [q + 1, q + 2],
                            )

                # Final layer of RY
                for q in range(N):
                    ty = weights[idx]
                    idx += 1
                    cy = torch.cos(ty / 2.0).to(complex_dtype)
                    sy = torch.sin(ty / 2.0).to(complex_dtype)
                    ry = torch.tensor([[cy, -sy], [sy, cy]], dtype=complex_dtype, device=device)
                    state = torch.moveaxis(
                        torch.tensordot(ry, state, dims=([1], [q + 1])), 0, q + 1
                    )
            else:
                raise ValueError(f"Unsupported circuit ansatz preset: '{circuit_fn}'")

        # 4. Readout of Parsed Observables
        dim = 2 ** N
        flat_state = state.reshape(B, dim)
        probs = torch.real(flat_state * flat_state.conj())
        indices = torch.arange(dim, device=device)

        exp_list: list[torch.Tensor] = []
        for parsed_obs in parsed_observables:
            total_exp = torch.zeros(B, dtype=real_dtype, device=device)
            for term in parsed_obs.terms:
                coeff = term.coeff
                p_str = term.pauli_string
                # Check for fast diagonal path (only I and Z)
                if all(ch in ("I", "Z") for ch in p_str):
                    signs = torch.ones(dim, dtype=real_dtype, device=device)
                    for q, ch in enumerate(p_str):
                        if ch == "Z":
                            shift = N - 1 - q
                            bit = (indices >> shift) & 1
                            signs = signs * (1.0 - 2.0 * bit.to(real_dtype))
                    term_exp = torch.matmul(probs, signs)
                else:
                    mat = ops.pauli_kron(
                        p_str, num_qubits=N, device=device, dtype=complex_dtype
                    )
                    term_exp = ops.batch_expectation(flat_state, mat)
                total_exp += coeff * term_exp
            exp_list.append(total_exp)

        return torch.stack(exp_list, dim=-1)

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        weights: torch.Tensor,
        num_qubits: int,
        num_layers: int,
        circuit_fn: Callable[..., Any] | str,
        parsed_observables: list[ParsedObservable],
        diff_method: str = "parameter-shift",
        encoding: str = "angle",
    ) -> torch.Tensor:
        """Forward pass executing batched circuit simulation and expectation readout."""
        ctx.save_for_backward(x, weights)
        ctx.num_qubits = num_qubits
        ctx.num_layers = num_layers
        ctx.circuit_fn = circuit_fn
        ctx.parsed_observables = parsed_observables
        ctx.diff_method = diff_method
        ctx.encoding = encoding

        return _QuantumLayerFunction._simulate_batch(
            x,
            weights,
            num_qubits,
            num_layers,
            circuit_fn,
            parsed_observables,
            encoding,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Backward pass executing analytical Parameter-Shift Rule or finite differences."""
        x, weights = ctx.saved_tensors
        num_qubits: int = ctx.num_qubits
        num_layers: int = ctx.num_layers
        circuit_fn = ctx.circuit_fn
        parsed_observables: list[ParsedObservable] = ctx.parsed_observables
        diff_method: str = ctx.diff_method
        encoding: str = ctx.encoding

        B, D_in = x.shape
        P = weights.shape[0]

        grad_x: torch.Tensor | None = None
        grad_weights: torch.Tensor | None = None

        # Short-circuit empty batch gradients (B=0)
        if B == 0:
            if ctx.needs_input_grad[0]:
                grad_x = torch.zeros_like(x)
            if ctx.needs_input_grad[1]:
                grad_weights = torch.zeros_like(weights)
            return grad_x, grad_weights, None, None, None, None, None, None

        # 1. Gradient with respect to Input Features X
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(x)
            if diff_method == "parameter-shift":
                shift_x = 0.5  # shift = pi / (2 * pi) = 0.5 for RY(pi * x)
                scale_x = math.pi / 2.0
                ansatz_name = circuit_fn.lower() if isinstance(circuit_fn, str) else ""

                if ansatz_name == "reuploading" and num_layers >= 2:
                    # Vectorized layer-wise parameter shift: multivariable chain rule
                    # d<O>/dx_k = \sum_{l=0}^{L-1} (pi / 2) [ <O>(x_{k,l} + 0.5)
                    #                                        - <O>(x_{k,l} - 0.5) ]
                    for k in range(min(D_in, num_qubits)):
                        x_rep_plus = x.unsqueeze(0).unsqueeze(0).repeat(
                            num_layers, num_layers, 1, 1
                        )
                        x_rep_minus = x.unsqueeze(0).unsqueeze(0).repeat(
                            num_layers, num_layers, 1, 1
                        )
                        for layer_idx in range(num_layers):
                            x_rep_plus[layer_idx, layer_idx, :, k] += shift_x
                            x_rep_minus[layer_idx, layer_idx, :, k] -= shift_x

                        x_plus_vec = x_rep_plus.reshape(num_layers, num_layers * B, D_in)
                        x_minus_vec = x_rep_minus.reshape(num_layers, num_layers * B, D_in)

                        y_plus_vec = _QuantumLayerFunction._simulate_batch(
                            x_plus_vec,
                            weights,
                            num_qubits,
                            num_layers,
                            circuit_fn,
                            parsed_observables,
                            encoding,
                        )
                        y_minus_vec = _QuantumLayerFunction._simulate_batch(
                            x_minus_vec,
                            weights,
                            num_qubits,
                            num_layers,
                            circuit_fn,
                            parsed_observables,
                            encoding,
                        )

                        diff_layers = (
                            (y_plus_vec - y_minus_vec)
                            .reshape(num_layers, B, -1)
                            .sum(dim=0)
                        )
                        jacobian_slice_k = scale_x * diff_layers
                        grad_x[:, k] = (grad_output * jacobian_slice_k).sum(dim=-1)
                else:
                    for k in range(min(D_in, num_qubits)):
                        x_plus = x.clone()
                        x_plus[:, k] += shift_x
                        y_plus = _QuantumLayerFunction._simulate_batch(
                            x_plus,
                            weights,
                            num_qubits,
                            num_layers,
                            circuit_fn,
                            parsed_observables,
                            encoding,
                        )

                        x_minus = x.clone()
                        x_minus[:, k] -= shift_x
                        y_minus = _QuantumLayerFunction._simulate_batch(
                            x_minus,
                            weights,
                            num_qubits,
                            num_layers,
                            circuit_fn,
                            parsed_observables,
                            encoding,
                        )

                        jacobian_slice_k = scale_x * (y_plus - y_minus)
                        grad_x[:, k] = (grad_output * jacobian_slice_k).sum(dim=-1)
            else:
                # Finite difference fallback
                eps = 1e-4
                for k in range(min(D_in, num_qubits)):
                    x_plus = x.clone()
                    x_plus[:, k] += eps
                    y_plus = _QuantumLayerFunction._simulate_batch(
                        x_plus,
                        weights,
                        num_qubits,
                        num_layers,
                        circuit_fn,
                        parsed_observables,
                        encoding,
                    )

                    x_minus = x.clone()
                    x_minus[:, k] -= eps
                    y_minus = _QuantumLayerFunction._simulate_batch(
                        x_minus,
                        weights,
                        num_qubits,
                        num_layers,
                        circuit_fn,
                        parsed_observables,
                        encoding,
                    )

                    jacobian_slice_k = (y_plus - y_minus) / (2.0 * eps)
                    grad_x[:, k] = (grad_output * jacobian_slice_k).sum(dim=-1)

        # 2. Gradient with respect to Variational Parameters \theta
        if ctx.needs_input_grad[1]:
            grad_weights = torch.zeros_like(weights)
            if diff_method == "parameter-shift":
                shift_theta = math.pi / 2.0
                scale_theta = 0.5

                for j in range(P):
                    w_plus = weights.clone()
                    w_plus[j] += shift_theta
                    y_plus = _QuantumLayerFunction._simulate_batch(
                        x,
                        w_plus,
                        num_qubits,
                        num_layers,
                        circuit_fn,
                        parsed_observables,
                        encoding,
                    )

                    w_minus = weights.clone()
                    w_minus[j] -= shift_theta
                    y_minus = _QuantumLayerFunction._simulate_batch(
                        x,
                        w_minus,
                        num_qubits,
                        num_layers,
                        circuit_fn,
                        parsed_observables,
                        encoding,
                    )

                    jacobian_slice_j = scale_theta * (y_plus - y_minus)
                    grad_weights[j] = (grad_output * jacobian_slice_j).sum()
            else:
                # Finite difference fallback
                eps = 1e-4
                for j in range(P):
                    w_plus = weights.clone()
                    w_plus[j] += eps
                    y_plus = _QuantumLayerFunction._simulate_batch(
                        x,
                        w_plus,
                        num_qubits,
                        num_layers,
                        circuit_fn,
                        parsed_observables,
                        encoding,
                    )

                    w_minus = weights.clone()
                    w_minus[j] -= eps
                    y_minus = _QuantumLayerFunction._simulate_batch(
                        x,
                        w_minus,
                        num_qubits,
                        num_layers,
                        circuit_fn,
                        parsed_observables,
                        encoding,
                    )

                    jacobian_slice_j = (y_plus - y_minus) / (2.0 * eps)
                    grad_weights[j] = (grad_output * jacobian_slice_j).sum()

        return grad_x, grad_weights, None, None, None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════
#  QuantumLayer nn.Module
# ═══════════════════════════════════════════════════════════════════════════

class QuantumLayer(nn.Module):
    """Variational Quantum Circuit Layer for PyTorch.

    Accepts classical input tensors and evaluates expectation values of specified
    quantum observables across parameterized quantum states. Differentiable via
    the analytical Parameter-Shift Rule for both variational parameters and inputs.

    Args:
        num_qubits: Number of simulated qubits (>= 1).
        circuit_fn: Ansatz preset name ('hardware_efficient', 'strongly_entangling',
            'reuploading', 'real_amplitudes'), or a custom Callable.
        num_layers: Depth of the variational circuit (>= 1).
        observables: Observables to measure. Defaults to [Z0, Z1, ..., Z_{N-1}].
        diff_method: Differentiation method ('parameter-shift' or 'finite-diff').
        device: Torch device for execution and parameters.
        dtype: Floating-point precision (torch.float32 or torch.float64).
        init_method: Weight initialization scheme ('uniform', 'normal', 'zeros').
        encoding: Feature encoding method ('angle').
        num_params: Optional explicit parameter count override for custom callables.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from quanta.torch import QuantumLayer
        >>>
        >>> layer = QuantumLayer(num_qubits=4, num_layers=2)
        >>> x = torch.randn(8, 4)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([8, 4])
        >>>
        >>> # Sequential model integration
        >>> model = nn.Sequential(
        ...     nn.Linear(2, 4),
        ...     QuantumLayer(num_qubits=4, num_layers=1),
        ...     nn.Linear(4, 1),
        ... )
    """

    def __init__(
        self,
        num_qubits: int,
        circuit_fn: Callable[..., Any] | Any | str = "hardware_efficient",
        num_layers: int = 1,
        observables: Sequence[Any] | None = None,
        diff_method: str = "parameter-shift",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        init_method: str = "uniform",
        encoding: str = "angle",
        num_params: int | None = None,
    ) -> None:
        super().__init__()

        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if diff_method not in {"parameter-shift", "finite-diff"}:
            raise ValueError(
                f"Unsupported diff_method '{diff_method}'. "
                "Supported: 'parameter-shift', 'finite-diff'"
            )

        self.num_qubits: int = num_qubits
        self.num_layers: int = num_layers
        self.circuit_fn: Callable[..., Any] | str = circuit_fn
        self.diff_method: str = diff_method
        self.encoding: str = encoding
        self.init_method: str = init_method

        # Dimensions
        self.in_features: int = num_qubits
        self._parsed_observables: list[ParsedObservable] = ObservableParser.parse_all(
            observables, num_qubits
        )
        self.out_features: int = len(self._parsed_observables)

        # Calculate parameter count
        if num_params is not None:
            self.num_params: int = num_params
        elif isinstance(circuit_fn, str):
            self.num_params = get_ansatz_param_count(circuit_fn, num_qubits, num_layers)
        elif hasattr(circuit_fn, "param_count"):
            self.num_params = int(circuit_fn.param_count(num_qubits, num_layers))
        elif hasattr(circuit_fn, "num_params"):
            self.num_params = int(circuit_fn.num_params)
        else:
            raise ValueError(
                "Could not infer parameter count for custom circuit_fn. "
                "Please specify num_params explicitly."
            )

        target_dev = ops.resolve_device(device) if device is not None else None
        self.weights = nn.Parameter(
            torch.empty(self.num_params, dtype=dtype, device=target_dev)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize variational parameters."""
        with torch.no_grad():
            if self.init_method == "uniform":
                nn.init.uniform_(self.weights, a=-math.pi, b=math.pi)
            elif self.init_method == "uniform_positive":
                nn.init.uniform_(self.weights, a=0.0, b=2.0 * math.pi)
            elif self.init_method == "normal":
                nn.init.normal_(self.weights, mean=0.0, std=0.1)
            elif self.init_method == "zeros":
                nn.init.zeros_(self.weights)
            else:
                raise ValueError(f"Unknown init_method: '{self.init_method}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward simulation and returns observable expectations.

        Args:
            x: Input feature tensor of shape (B, in_features), (in_features,),
                or (*batch, in_features).

        Returns:
            Expectation tensor of shape (B, out_features), (out_features,),
            or (*batch, out_features).
        """
        if x.ndim == 0:
            raise ValueError("Input tensor must have at least 1 dimension, got 0D scalar.")

        last_dim = x.shape[-1]
        if last_dim != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: expected in_features={self.in_features} "
                f"(matching {self.num_qubits} qubits), but got x.shape[-1]={last_dim}."
            )

        orig_shape = x.shape
        if x.ndim == 1:
            x_2d = x.unsqueeze(0)
        elif x.ndim > 2:
            x_2d = x.reshape(-1, last_dim)
        else:
            x_2d = x

        # Ensure device and dtype match weights (convert dtype first, then device)
        if x_2d.dtype != self.weights.dtype:
            x_2d = x_2d.to(dtype=self.weights.dtype)
        if x_2d.device != self.weights.device:
            x_2d = x_2d.to(device=self.weights.device)

        out_tensor = cast(
            torch.Tensor,
            _QuantumLayerFunction.apply(
                x_2d,
                self.weights,
                self.num_qubits,
                self.num_layers,
                self.circuit_fn,
                self._parsed_observables,
                self.diff_method,
                self.encoding,
            ),
        )

        if x.ndim == 1:
            return out_tensor.squeeze(0)
        elif x.ndim > 2:
            return out_tensor.reshape(*orig_shape[:-1], self.out_features)
        return out_tensor

    def extra_repr(self) -> str:
        circuit_name = (
            self.circuit_fn
            if isinstance(self.circuit_fn, str)
            else getattr(self.circuit_fn, "__name__", str(self.circuit_fn))
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_qubits={self.num_qubits}, num_layers={self.num_layers}, "
            f"ansatz='{circuit_name}', diff_method='{self.diff_method}', "
            f"num_params={self.num_params}"
        )
