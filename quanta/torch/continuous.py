"""quanta.torch.continuous — Continuous Resonant Quantum Neural Network Layer.

Pillar 2 of Quanta SDK (Milestone 3):
Brain-inspired continuous-time quantum resonance, non-local quantum coherence,
and simultaneous non-sequential state evolution grounded in foundational physics
(Einstein-Podolsky-Rosen non-locality, continuous-time quantum walks, many-body spin networks).

Evolves quantum statevectors continuously under a parameterized network Hamiltonian:
    H(x, θ) = H_XY(J) + H_Z(x, h, W) + H_X(ω)
    |ψ(t)⟩ = exp(-i H(x, θ) t) |ψ0⟩
with simultaneous multi-observable expectation readout across all nodes in O(2^N) time.
Features exact analytical autograd gradients via Ehrenfest's theorem (for duration t)
and Daleckii-Krein matrix spectral Fréchet derivatives (for J, h, W, ω, x).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from quanta.torch import ops

__all__ = [
    "ContinuousResonantLayer",
    "_ContinuousResonantFunction",
    "GraphTopologyParser",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Graph Topology Engine
# ═══════════════════════════════════════════════════════════════════════════

class GraphTopologyParser:
    """Parses diverse user graph inputs into canonical sorted edge lists.

    Supports presets ('complete', 'ring', 'line', 'star', 'none'),
    adjacency matrices, and explicit edge sequences with strict validation.
    """

    @classmethod
    def parse(
        cls,
        graph: torch.Tensor | Sequence[tuple[int, int]] | str,
        num_nodes: int,
    ) -> list[tuple[int, int]]:
        """Parses and validates graph topologies into canonical sorted edge lists.

        Args:
            graph: Graph topology preset name, adjacency matrix, or edge sequence.
            num_nodes: Total number of nodes/qubits (>= 1).

        Returns:
            Canonical list of sorted (u, v) edge tuples with 0 <= u < v < num_nodes.

        Raises:
            ValueError: On invalid node count, out-of-bounds index, self-loop,
                or non-symmetric adjacency matrix.
            TypeError: On unsupported input type.
        """
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")

        # 1. String Presets
        if isinstance(graph, str):
            preset = graph.lower().strip()
            if preset in ("complete", "all_to_all", "clique"):
                return [(j, k) for j in range(num_nodes) for k in range(j + 1, num_nodes)]
            if preset in ("ring", "cycle"):
                if num_nodes <= 1:
                    return []
                if num_nodes == 2:
                    return [(0, 1)]
                edges = [(j, j + 1) for j in range(num_nodes - 1)] + [(0, num_nodes - 1)]
                return sorted(edges)
            if preset in ("line", "chain", "path"):
                return [(j, j + 1) for j in range(num_nodes - 1)]
            if preset == "star":
                return [(0, j) for j in range(1, num_nodes)]
            if preset in ("none", "empty", "disconnected"):
                return []
            raise ValueError(
                f"Unknown coupling_graph preset: '{graph}'. Supported presets: "
                f"'complete', 'ring', 'line', 'star', 'none'."
            )

        # 2. Adjacency Matrix Tensor
        if isinstance(graph, torch.Tensor):
            if graph.ndim != 2 or graph.shape[0] != num_nodes or graph.shape[1] != num_nodes:
                raise ValueError(
                    f"Adjacency matrix must have shape ({num_nodes}, {num_nodes}), "
                    f"got {tuple(graph.shape)}."
                )
            if not torch.allclose(graph, graph.T):
                raise ValueError("Adjacency matrix must be symmetric (A == A.T).")

            edges_set: set[tuple[int, int]] = set()
            for j in range(num_nodes):
                for k in range(j + 1, num_nodes):
                    if graph[j, k] != 0 or graph[k, j] != 0:
                        edges_set.add((j, k))
            return sorted(list(edges_set))

        # 3. Explicit Sequence of Edge Tuples
        if isinstance(graph, (list, tuple)):
            raw_edges: set[tuple[int, int]] = set()
            for idx, edge in enumerate(graph):
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    raise ValueError(
                        f"Invalid edge at index {idx}: expected 2-tuple (u, v), got {edge}."
                    )
                u, v = int(edge[0]), int(edge[1])
                if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                    raise ValueError(
                        f"Edge ({u}, {v}) at index {idx} out of node range [0, {num_nodes - 1}]."
                    )
                if u == v:
                    raise ValueError(
                        f"Self-loop edge ({u}, {v}) at index {idx} is not allowed."
                    )
                raw_edges.add((min(u, v), max(u, v)))
            return sorted(list(raw_edges))

        raise TypeError(
            f"Unsupported coupling_graph type: {type(graph)}. "
            f"Expected str, list of (int, int) tuples, or torch.Tensor."
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Custom Autograd Engine: _ContinuousResonantFunction
# ═══════════════════════════════════════════════════════════════════════════

class _ContinuousResonantFunction(torch.autograd.Function):
    """Custom PyTorch Autograd Function for Continuous Quantum Resonance.

    Forward pass:
        Evolves quantum statevectors via exact spectral decomposition of the
        many-body Hamiltonian H(x, θ) = H_XY(J) + H_Z(x, h, W) + H_X(ω) over duration t.
        Simultaneously measures multi-observable expectations in O(2^N) time.

    Backward pass:
        Evaluates exact analytical vector-Jacobian products (VJPs):
        1. Exact Ehrenfest time derivative:
           d<O>/dt = +2 * Im[ <psi(t)| O H |psi(t)> ] in O(B * 2^N) flops.
        2. Daleckii-Krein matrix spectral Fréchet derivative with numerically
           stable sinc kernel M_ab(t) = -i t exp(-i λ_bar t) sinc(Δ t / 2π).
        3. Decoupled Hermitian Gradient Density Matrix S_b = G_b + G_b^H yielding
           instantaneous parameter projections for J, h, W, ω, and input x.
        4. Vector-only observable contraction |w_b⟩ = Σ_m Y_bar_{b,m} O_m |ψ_b(t)⟩
           without dense observable matrix allocation (O(B * 2^N) memory).
    """

    @staticmethod
    def _simulate_forward(
        x: torch.Tensor,
        J: torch.Tensor,
        h: torch.Tensor,
        W: torch.Tensor,
        omega: torch.Tensor,
        t: torch.Tensor,
        num_nodes: int,
        edges: list[tuple[int, int]],
        observable_types: tuple[str, ...],
        initial_state: str | torch.Tensor,
        basis: ops.ResonantInteractionBasis,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        c_dtype = ops.resolve_complex_dtype(dtype, device)
        B = x.shape[0]
        dim = 2 ** num_nodes

        # 1. Assemble batch Hamiltonian: shape (B, dim, dim)
        H_offdiag = torch.zeros((dim, dim), dtype=c_dtype, device=device)
        if len(edges) > 0:
            for j_val, xy_mat in zip(J, basis.xy_matrices, strict=False):
                H_offdiag = H_offdiag + j_val * xy_mat
        for j in range(num_nodes):
            H_offdiag = H_offdiag + omega[j] * basis.x_matrices[j]

        # Vectorized longitudinal potential: v_batch = x @ W.T + h (B, N)
        v_batch = torch.matmul(x, W.T) + h
        # Diagonal modulation: diag_batch = v_batch @ longitudinal_signs (B, dim)
        diag_batch = torch.matmul(v_batch, basis.longitudinal_signs)

        H = H_offdiag.unsqueeze(0).repeat(B, 1, 1)
        diag_indices = torch.arange(dim, device=device)
        H[:, diag_indices, diag_indices] = H[:, diag_indices, diag_indices] + diag_batch.to(c_dtype)

        # 2. Spectral decomposition: H = V Lambda V^dagger
        lambdas, V = torch.linalg.eigh(H)  # (B, dim), (B, dim, dim)

        # 3. Transform initial state to eigenbasis: xi_0 = V^dagger |psi_0>
        if isinstance(initial_state, str):
            psi0 = ops.create_initial_state(
                initial_state, num_qubits=num_nodes, device=device, dtype=c_dtype
            )
        else:
            psi0 = initial_state.to(device=device, dtype=c_dtype)

        if psi0.dim() == 1:
            xi0 = torch.matmul(
                V.conj().transpose(-2, -1), psi0.unsqueeze(0).expand(B, -1).unsqueeze(-1)
            ).squeeze(-1)
        else:
            xi0 = torch.matmul(V.conj().transpose(-2, -1), psi0.unsqueeze(-1)).squeeze(-1)

        # 4. State evolution in eigenbasis: |psi(t)> = V (exp(-i Lambda t) * xi_0)
        t_val = t.squeeze()
        exp_diag = torch.exp(-1.0j * lambdas * t_val)
        psi_t_eigen = exp_diag * xi0
        psi_t = torch.matmul(V, psi_t_eigen.unsqueeze(-1)).squeeze(-1)  # (B, dim)

        # 5. Simultaneous multi-observable readout
        out = ops.simultaneous_readout(psi_t, num_nodes, observable_types)

        return out, lambdas, V, xi0, psi_t, psi_t_eigen

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        J: torch.Tensor,
        h: torch.Tensor,
        W: torch.Tensor,
        omega: torch.Tensor,
        t: torch.Tensor,
        num_nodes: int,
        edges: list[tuple[int, int]],
        observable_types: tuple[str, ...],
        initial_state: str | torch.Tensor,
        basis: ops.ResonantInteractionBasis | None = None,
        diff_method: str = "exact",
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        c_dtype = ops.resolve_complex_dtype(dtype, device)
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)

        B = x.shape[0]
        out_features = num_nodes * len(observable_types)

        # Handle empty batch B=0 immediately
        if B == 0:
            out_empty = torch.empty((0, out_features), dtype=x.dtype, device=device)
            ctx.save_for_backward(x, J, h, W, omega, t)
            ctx.is_empty = True
            ctx.is_1d = is_1d
            return out_empty

        if basis is None:
            basis = ops.ResonantInteractionBasis(
                num_nodes, edges, device=device, dtype=c_dtype
            )

        out, lambdas, V, xi0, psi_t, psi_t_eigen = _ContinuousResonantFunction._simulate_forward(
            x, J, h, W, omega, t, num_nodes, edges, observable_types, initial_state, basis
        )

        ctx.save_for_backward(x, J, h, W, omega, t, lambdas, V, xi0, psi_t, psi_t_eigen)
        ctx.num_nodes = num_nodes
        ctx.edges = edges
        ctx.observable_types = observable_types
        ctx.initial_state = initial_state
        ctx.basis = basis
        ctx.is_1d = is_1d
        ctx.diff_method = diff_method
        ctx.is_empty = False

        return out.squeeze(0) if is_1d else out

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        is_1d: bool = ctx.is_1d
        if is_1d and grad_output.dim() == 1:
            grad_output = grad_output.unsqueeze(0)

        grad_x: torch.Tensor | None = None
        grad_J: torch.Tensor | None = None
        grad_h: torch.Tensor | None = None
        grad_W: torch.Tensor | None = None
        grad_omega: torch.Tensor | None = None
        grad_t: torch.Tensor | None = None

        # Handle empty batch B=0
        if getattr(ctx, "is_empty", False):
            x, J, h, W, omega, t = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                grad_x = torch.zeros_like(x)
            if ctx.needs_input_grad[1]:
                grad_J = torch.zeros_like(J)
            if ctx.needs_input_grad[2]:
                grad_h = torch.zeros_like(h)
            if ctx.needs_input_grad[3]:
                grad_W = torch.zeros_like(W)
            if ctx.needs_input_grad[4]:
                grad_omega = torch.zeros_like(omega)
            if ctx.needs_input_grad[5]:
                grad_t = torch.zeros_like(t)
            return (
                grad_x,
                grad_J,
                grad_h,
                grad_W,
                grad_omega,
                grad_t,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        (
            x,
            J,
            h,
            W,
            omega,
            t,
            lambdas,
            V,
            xi0,
            psi_t,
            psi_t_eigen,
        ) = ctx.saved_tensors
        num_nodes: int = ctx.num_nodes
        edges: list[tuple[int, int]] = ctx.edges
        obs_types: tuple[str, ...] = ctx.observable_types
        basis: ops.ResonantInteractionBasis = ctx.basis
        diff_method: str = ctx.diff_method

        device = x.device
        dtype = x.dtype
        c_dtype = ops.resolve_complex_dtype(dtype, device)
        dim = 2 ** num_nodes

        # ── Optional Finite-Difference Differentiation Strategy ──
        if diff_method == "finite-diff":
            eps = 1e-4
            grad_x_fd = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
            grad_J_fd = torch.zeros_like(J) if ctx.needs_input_grad[1] else None
            grad_h_fd = torch.zeros_like(h) if ctx.needs_input_grad[2] else None
            grad_W_fd = torch.zeros_like(W) if ctx.needs_input_grad[3] else None
            grad_omega_fd = torch.zeros_like(omega) if ctx.needs_input_grad[4] else None
            grad_t_fd = torch.zeros_like(t) if ctx.needs_input_grad[5] else None

            def eval_f(
                x_in: torch.Tensor,
                J_in: torch.Tensor,
                h_in: torch.Tensor,
                W_in: torch.Tensor,
                omega_in: torch.Tensor,
                t_in: torch.Tensor,
            ) -> torch.Tensor:
                out_eval, _, _, _, _, _ = _ContinuousResonantFunction._simulate_forward(
                    x_in, J_in, h_in, W_in, omega_in, t_in,
                    num_nodes, edges, obs_types, ctx.initial_state, basis
                )
                return out_eval

            if ctx.needs_input_grad[5]:
                t_plus = t + eps
                t_minus = t - eps
                out_p = eval_f(x, J, h, W, omega, t_plus)
                out_m = eval_f(x, J, h, W, omega, t_minus)
                jac = (out_p - out_m) / (2.0 * eps)
                grad_t_fd = (grad_output * jac).sum().reshape(t.shape)

            if ctx.needs_input_grad[0] and grad_x_fd is not None:
                for b in range(x.shape[0]):
                    for d in range(x.shape[1]):
                        x_p = x.clone()
                        x_m = x.clone()
                        x_p[b, d] += eps
                        x_m[b, d] -= eps
                        out_p = eval_f(x_p, J, h, W, omega, t)
                        out_m = eval_f(x_m, J, h, W, omega, t)
                        jac = (out_p - out_m) / (2.0 * eps)
                        grad_x_fd[b, d] = (grad_output * jac).sum()
                if is_1d:
                    grad_x_fd = grad_x_fd.squeeze(0)

            if ctx.needs_input_grad[1] and grad_J_fd is not None:
                for e_idx in range(len(edges)):
                    J_p = J.clone()
                    J_m = J.clone()
                    J_p[e_idx] += eps
                    J_m[e_idx] -= eps
                    out_p = eval_f(x, J_p, h, W, omega, t)
                    out_m = eval_f(x, J_m, h, W, omega, t)
                    jac = (out_p - out_m) / (2.0 * eps)
                    grad_J_fd[e_idx] = (grad_output * jac).sum()

            if ctx.needs_input_grad[2] and grad_h_fd is not None:
                for j in range(num_nodes):
                    h_p = h.clone()
                    h_m = h.clone()
                    h_p[j] += eps
                    h_m[j] -= eps
                    out_p = eval_f(x, J, h_p, W, omega, t)
                    out_m = eval_f(x, J, h_m, W, omega, t)
                    jac = (out_p - out_m) / (2.0 * eps)
                    grad_h_fd[j] = (grad_output * jac).sum()

            if ctx.needs_input_grad[3] and grad_W_fd is not None:
                for j in range(num_nodes):
                    for d in range(W.shape[1]):
                        W_p = W.clone()
                        W_m = W.clone()
                        W_p[j, d] += eps
                        W_m[j, d] -= eps
                        out_p = eval_f(x, J, h, W_p, omega, t)
                        out_m = eval_f(x, J, h, W_m, omega, t)
                        jac = (out_p - out_m) / (2.0 * eps)
                        grad_W_fd[j, d] = (grad_output * jac).sum()

            if ctx.needs_input_grad[4] and grad_omega_fd is not None:
                for j in range(num_nodes):
                    omega_p = omega.clone()
                    omega_m = omega.clone()
                    omega_p[j] += eps
                    omega_m[j] -= eps
                    out_p = eval_f(x, J, h, W, omega_p, t)
                    out_m = eval_f(x, J, h, W, omega_m, t)
                    jac = (out_p - out_m) / (2.0 * eps)
                    grad_omega_fd[j] = (grad_output * jac).sum()

            return (
                grad_x_fd,
                grad_J_fd,
                grad_h_fd,
                grad_W_fd,
                grad_omega_fd,
                grad_t_fd,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # ── Exact Analytical Ehrenfest & Daleckii-Krein Gradients ──

        # 1. Vector-only observable contraction: |w_b> = sum_m grad_output[b,m] O_m |psi_b(t)>
        w = torch.zeros_like(psi_t)
        indices = torch.arange(dim, device=device)
        curr_col = 0
        for obs in obs_types:
            obs_upper = obs.upper()
            if obs_upper == "Z":
                grad_z = grad_output[:, curr_col : curr_col + num_nodes]
                curr_col += num_nodes
                z_coeff = torch.matmul(grad_z, basis.longitudinal_signs)
                w = w + z_coeff.to(c_dtype) * psi_t
            elif obs_upper == "X":
                grad_x_obs = grad_output[:, curr_col : curr_col + num_nodes]
                curr_col += num_nodes
                for j in range(num_nodes):
                    perm_j = indices ^ (1 << (num_nodes - 1 - j))
                    w = w + grad_x_obs[:, j].unsqueeze(-1).to(c_dtype) * psi_t[:, perm_j]
            elif obs_upper == "Y":
                grad_y_obs = grad_output[:, curr_col : curr_col + num_nodes]
                curr_col += num_nodes
                for j in range(num_nodes):
                    perm_j = indices ^ (1 << (num_nodes - 1 - j))
                    y_factor = -1.0j * basis.longitudinal_signs[j]
                    y_term = (
                        grad_y_obs[:, j].unsqueeze(-1).to(c_dtype) * y_factor
                    ) * psi_t[:, perm_j]
                    w = w + y_term
            else:
                raise ValueError(f"Unsupported observable type: {obs!r}")

        # 2. Exact Ehrenfest time derivative: +2 Im[ <w | H | psi(t)> ]
        if ctx.needs_input_grad[5]:
            H_psi_t = torch.matmul(V, (lambdas * psi_t_eigen).unsqueeze(-1)).squeeze(-1)
            grad_t = 2.0 * torch.imag(torch.sum(w.conj() * H_psi_t, dim=-1)).sum().reshape(t.shape)

        # 3. Daleckii-Krein matrix spectral Fréchet derivatives
        any_param_grad = (
            ctx.needs_input_grad[0]
            or ctx.needs_input_grad[1]
            or ctx.needs_input_grad[2]
            or ctx.needs_input_grad[3]
            or ctx.needs_input_grad[4]
        )

        if any_param_grad:
            t_val = t.squeeze()
            diff = lambdas.unsqueeze(-1) - lambdas.unsqueeze(-2)
            mean = (lambdas.unsqueeze(-1) + lambdas.unsqueeze(-2)) / 2.0
            arg = (diff * t_val) / (2.0 * math.pi)
            M = -1.0j * t_val * torch.exp(-1.0j * mean * t_val) * torch.sinc(arg)

            eta = torch.matmul(V.conj().transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)
            P = torch.einsum("ba,bc,bac->bac", eta.conj(), xi0, M)
            G = torch.matmul(V, torch.matmul(P.transpose(-2, -1), V.conj().transpose(-2, -1)))
            S = G + G.conj().transpose(-2, -1)

            # Longitudinal fields: nu_{b, j} = h_j + sum_d W_{jd} x_{b, d}
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
                diag_S = torch.diagonal(S, dim1=-2, dim2=-1).real
                g_nu = torch.matmul(diag_S, basis.longitudinal_signs.T)  # (B, N)

                if ctx.needs_input_grad[2]:
                    grad_h = g_nu.sum(dim=0).reshape(h.shape)
                if ctx.needs_input_grad[3]:
                    grad_W = torch.matmul(g_nu.T, x).reshape(W.shape)
                if ctx.needs_input_grad[0]:
                    grad_x_mat = torch.matmul(g_nu, W)
                    grad_x = grad_x_mat.squeeze(0) if is_1d else grad_x_mat.reshape(x.shape)

            # Coupling strengths J along graph edges
            if ctx.needs_input_grad[1]:
                grad_J = torch.zeros_like(J)
                if len(edges) > 0:
                    for e_idx, xy_mat in enumerate(basis.xy_matrices):
                        grad_J[e_idx] = torch.real(torch.sum(S * xy_mat.T))

            # Transverse tunneling drives omega
            if ctx.needs_input_grad[4]:
                grad_omega = torch.zeros_like(omega)
                for j in range(num_nodes):
                    perm_j = indices ^ (1 << (num_nodes - 1 - j))
                    grad_omega[j] = torch.real(torch.sum(S[:, indices, perm_j]))

        return (
            grad_x,
            grad_J,
            grad_h,
            grad_W,
            grad_omega,
            grad_t,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. ContinuousResonantLayer nn.Module
# ═══════════════════════════════════════════════════════════════════════════

class ContinuousResonantLayer(nn.Module):
    """Continuous-Time Resonant Quantum Neural Network Layer.

    Evolves quantum statevectors under an interacting many-body graph Hamiltonian:
        H(x, θ) = H_XY(J) + H_Z(x, h, W) + H_X(ω)
        |ψ(t)⟩ = exp(-i H(x, θ) t) |ψ0⟩
    with simultaneous multi-observable expectation readout across all nodes.

    Args:
        num_nodes: Number of qubits/nodes in the network (>= 1).
        in_features: Dimension of input feature vector x (>= 1).
        coupling_graph: Interaction graph topology:
            - Preset string: 'complete', 'ring', 'line', 'star', 'none'.
            - Adjacency matrix: torch.Tensor of shape (num_nodes, num_nodes).
            - Custom edge list: Sequence of (u, v) tuples.
        observable_types: Tuple of observables to measure across all nodes.
            Supported: ('Z', 'X'), ('Z',), ('Z', 'X', 'Y').
            Output feature dimension is num_nodes * len(observable_types).
        initial_state: Reference initial quantum state:
            - Preset mode: 'zero' (|0...0>), 'plus' (|+...+>), 'ghz'.
            - Custom tensor: Normalized statevector of shape (2^num_nodes,).
        learnable_time: Whether evolution duration t is a trainable parameter.
            If True, registered as nn.Parameter; if False, registered as buffer.
        initial_time: Initial evolution duration t0 (> 0.0). Defaults to 1.0.
        device: Target execution device ('cpu', 'mps', or torch.device).
            Defaults to MPS if available, otherwise CPU.
        dtype: Real scalar floating-point precision (torch.float32 or torch.float64).
            Note: Apple Silicon MPS only supports torch.float32.
        init_method: Parameter initialization strategy ('default', 'uniform', 'normal', 'zeros').
        diff_method: Differentiation strategy ('exact', 'spectral', 'autograd', 'finite-diff').

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from quanta.torch import ContinuousResonantLayer
        >>>
        >>> layer = ContinuousResonantLayer(num_nodes=4, in_features=3, coupling_graph="ring")
        >>> x = torch.randn(8, 3)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([8, 8])
        >>>
        >>> # Seamless Sequential integration
        >>> model = nn.Sequential(
        ...     nn.Linear(2, 4),
        ...     ContinuousResonantLayer(num_nodes=4, in_features=4, coupling_graph="complete"),
        ...     nn.Linear(8, 1),
        ... )
    """

    initial_state: torch.Tensor
    t: torch.Tensor

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        coupling_graph: torch.Tensor | Sequence[tuple[int, int]] | str = "complete",
        observable_types: tuple[str, ...] = ("Z", "X"),
        initial_state: str | torch.Tensor = "zero",
        learnable_time: bool = True,
        initial_time: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        init_method: str = "default",
        diff_method: str = "exact",
    ) -> None:
        super().__init__()

        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if initial_time <= 0.0:
            raise ValueError(f"initial_time must be > 0.0, got {initial_time}")

        target_dev = ops.resolve_device(device) if device is not None else None
        dev = target_dev if target_dev is not None else torch.device("cpu")
        if (
            target_dev is not None
            and target_dev.type == "mps"
            and dtype in (torch.float64, torch.complex128)
        ):
            raise ops.UnsupportedDtypeError(
                f"Apple Silicon MPS does not support 64-bit precision ({dtype}). "
                f"Use torch.float32 on MPS or switch execution to device='cpu'."
            )

        self.num_nodes: int = num_nodes
        self.in_features: int = in_features
        self.coupling_graph_spec = coupling_graph
        self.edges: list[tuple[int, int]] = GraphTopologyParser.parse(coupling_graph, num_nodes)
        self.num_edges: int = len(self.edges)

        canonical_obs = tuple(obs.upper() for obs in observable_types)
        for obs in canonical_obs:
            if obs not in ("Z", "X", "Y"):
                raise ValueError(
                    f"Unsupported observable type '{obs}'. Supported types: 'Z', 'X', 'Y'."
                )
        self.observable_types: tuple[str, ...] = canonical_obs
        self.out_features: int = num_nodes * len(self.observable_types)

        self.learnable_time: bool = learnable_time
        self.initial_time: float = float(initial_time)
        self.init_method: str = init_method
        self.diff_method: str = diff_method
        self._cached_basis: ops.ResonantInteractionBasis | None = None

        self.complex_dtype = ops.resolve_complex_dtype(dtype, dev)

        # Variational parameters
        self.W = nn.Parameter(torch.empty(num_nodes, in_features, dtype=dtype, device=target_dev))
        self.h = nn.Parameter(torch.empty(num_nodes, dtype=dtype, device=target_dev))
        self.omega = nn.Parameter(torch.empty(num_nodes, dtype=dtype, device=target_dev))
        self.J = nn.Parameter(torch.empty(self.num_edges, dtype=dtype, device=target_dev))

        if self.learnable_time:
            self.t = nn.Parameter(
                torch.tensor([self.initial_time], dtype=dtype, device=target_dev)
            )
        else:
            self.register_buffer(
                "t", torch.tensor([self.initial_time], dtype=dtype, device=target_dev)
            )

        # Initial reference quantum state
        self.initial_state_spec = initial_state
        if isinstance(initial_state, torch.Tensor) and initial_state.numel() != 2 ** num_nodes:
            raise ValueError(
                f"Custom initial_state must have length {2 ** num_nodes}, "
                f"got {initial_state.numel()}."
            )

        init_mode = "plus" if initial_state == "superposition" else initial_state
        resolved_dev = target_dev if target_dev is not None else torch.device("cpu")
        psi0 = ops.create_initial_state(
            init_mode, num_qubits=num_nodes, device=resolved_dev, dtype=self.complex_dtype
        )
        self.register_buffer("initial_state", psi0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes variational parameters based on physical resonance criteria."""
        with torch.no_grad():
            if self.init_method == "default":
                stdv = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 1.0
                nn.init.uniform_(self.W, -stdv, stdv)
                nn.init.uniform_(self.h, -0.1, 0.1)
                nn.init.uniform_(self.omega, 0.5, 1.5)
                if self.num_edges > 0:
                    nn.init.uniform_(self.J, -0.5, 0.5)
                if self.learnable_time and hasattr(self, "t") and isinstance(self.t, nn.Parameter):
                    self.t.data.fill_(self.initial_time)
            elif self.init_method == "uniform":
                nn.init.uniform_(self.W, -0.1, 0.1)
                nn.init.uniform_(self.h, -0.1, 0.1)
                nn.init.uniform_(self.omega, 0.5, 1.5)
                if self.num_edges > 0:
                    nn.init.uniform_(self.J, -0.1, 0.1)
                if self.learnable_time and hasattr(self, "t") and isinstance(self.t, nn.Parameter):
                    self.t.data.fill_(self.initial_time)
            elif self.init_method == "normal":
                nn.init.normal_(self.W, mean=0.0, std=0.1)
                nn.init.normal_(self.h, mean=0.0, std=0.1)
                nn.init.normal_(self.omega, mean=1.0, std=0.1)
                if self.num_edges > 0:
                    nn.init.normal_(self.J, mean=0.0, std=0.2)
                if self.learnable_time and hasattr(self, "t") and isinstance(self.t, nn.Parameter):
                    self.t.data.fill_(self.initial_time)
            elif self.init_method == "zeros":
                nn.init.zeros_(self.W)
                nn.init.zeros_(self.h)
                nn.init.zeros_(self.omega)
                if self.num_edges > 0:
                    nn.init.zeros_(self.J)
                if self.learnable_time and hasattr(self, "t") and isinstance(self.t, nn.Parameter):
                    self.t.data.fill_(self.initial_time)
            else:
                raise ValueError(
                    f"Unknown init_method: '{self.init_method}'. "
                    f"Supported: 'default', 'uniform', 'normal', 'zeros'."
                )

    def _get_basis(
        self, device: torch.device, complex_dtype: torch.dtype
    ) -> ops.ResonantInteractionBasis:
        """Retrieves or lazily constructs device-resident interaction basis operators."""
        if (
            self._cached_basis is None
            or self._cached_basis.device != device
            or self._cached_basis.dtype != complex_dtype
        ):
            self._cached_basis = ops.ResonantInteractionBasis(
                num_nodes=self.num_nodes,
                edges=self.edges,
                device=device,
                dtype=complex_dtype,
            )
        return self._cached_basis

    def _apply(self, fn: Any, recurse: bool = True) -> ContinuousResonantLayer:
        init_st = self._buffers.pop("initial_state", None)
        super()._apply(fn, recurse=recurse)
        self._cached_basis = None
        target_dev = self.W.device
        target_c_dtype = ops.resolve_complex_dtype(self.W.dtype, target_dev)
        self.complex_dtype = target_c_dtype
        if init_st is not None:
            if isinstance(self.initial_state_spec, str):
                init_mode = (
                    "plus"
                    if self.initial_state_spec == "superposition"
                    else self.initial_state_spec
                )
                self.register_buffer(
                    "initial_state",
                    ops.create_initial_state(
                        init_mode,
                        num_qubits=self.num_nodes,
                        device=target_dev,
                        dtype=target_c_dtype,
                    ),
                )
            else:
                self.register_buffer(
                    "initial_state",
                    init_st.to(device=target_dev, dtype=target_c_dtype),
                )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes continuous Hamiltonian evolution and returns expectation readouts.

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
                f"Feature dimension mismatch: expected in_features={self.in_features}, "
                f"but got x.shape[-1]={last_dim}."
            )

        orig_shape = x.shape
        if x.ndim == 1:
            x_2d = x.unsqueeze(0)
        elif x.ndim > 2:
            x_2d = x.reshape(-1, last_dim)
        else:
            x_2d = x

        # Ensure dtype matches first, then device (critical for MPS 64-bit safety)
        if x_2d.dtype != self.W.dtype:
            x_2d = x_2d.to(dtype=self.W.dtype)
        if x_2d.device != self.W.device:
            x_2d = x_2d.to(device=self.W.device)

        basis = self._get_basis(self.W.device, self.complex_dtype)

        # Check for empty batch
        if x_2d.shape[0] == 0:
            if self.diff_method == "autograd":
                out_tensor = torch.empty(
                    (0, self.out_features), dtype=self.W.dtype, device=self.W.device
                )
            else:
                out_tensor = cast(
                    torch.Tensor,
                    _ContinuousResonantFunction.apply(
                        x_2d,
                        self.J,
                        self.h,
                        self.W,
                        self.omega,
                        self.t,
                        self.num_nodes,
                        self.edges,
                        self.observable_types,
                        self.initial_state,
                        basis,
                        self.diff_method,
                    ),
                )
        elif self.diff_method == "autograd":
            # Native PyTorch autograd through matrix_exp
            H_batch = ops.build_batch_resonant_hamiltonian(
                x=x_2d,
                J=self.J,
                edges=self.edges,
                h=self.h,
                W=self.W,
                omega=self.omega,
                num_nodes=self.num_nodes,
                basis=basis,
                device=self.W.device,
                dtype=self.complex_dtype,
            )
            psi_t = ops.unitary_evolution(H_batch, self.t, self.initial_state)
            out_tensor = ops.simultaneous_readout(psi_t, self.num_nodes, self.observable_types)
        else:
            out_tensor = cast(
                torch.Tensor,
                _ContinuousResonantFunction.apply(
                    x_2d,
                    self.J,
                    self.h,
                    self.W,
                    self.omega,
                    self.t,
                    self.num_nodes,
                    self.edges,
                    self.observable_types,
                    self.initial_state,
                    basis,
                    self.diff_method,
                ),
            )

        if x.ndim == 1:
            return out_tensor.squeeze(0)
        if x.ndim > 2:
            return out_tensor.reshape(*orig_shape[:-1], self.out_features)
        return out_tensor

    def extra_repr(self) -> str:
        graph_desc = (
            self.coupling_graph_spec
            if isinstance(self.coupling_graph_spec, str)
            else f"{self.num_edges} edges"
        )
        return (
            f"num_nodes={self.num_nodes}, in_features={self.in_features}, "
            f"out_features={self.out_features}, graph='{graph_desc}', "
            f"observables={self.observable_types}, learnable_time={self.learnable_time}, "
            f"initial_time={self.initial_time}, diff_method='{self.diff_method}'"
        )
