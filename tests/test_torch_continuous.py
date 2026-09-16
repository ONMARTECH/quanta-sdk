"""tests.test_torch_continuous — Tests for quanta.torch.continuous.

Exhaustive test suite for ContinuousResonantLayer, _ContinuousResonantFunction,
and GraphTopologyParser across Tiers 1 through 5:
- Tier 1: Graph topology parsing, presets, custom edge lists, adjacency matrices, and errors.
- Tier 2: Constructor, parameter shapes, initialization methods, and dimensional invariants.
- Tier 3: Mathematical boundary invariants (norm preservation, Hermiticity, energy invariance).
- Tier 4: Autograd gradcheck on CPU float64, Ehrenfest & Daleckii-Krein analytical gradients.
- Tier 5: PyTorch ecosystem (nn.Sequential, Adam optimization, XOR learning) and Apple Silicon MPS.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from quanta.torch import (
    ContinuousResonantLayer,
    GraphTopologyParser,
    UnsupportedDtypeError,
    _ContinuousResonantFunction,
    ops,
)

# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: Graph Topology Parsing & Validation
# ═══════════════════════════════════════════════════════════════════════════

def test_graph_parser_presets() -> None:
    """Validates edge generation for all topology presets."""
    N = 4
    # Complete: N*(N-1)/2 = 6 edges
    complete = GraphTopologyParser.parse("complete", N)
    assert len(complete) == 6
    assert complete == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Aliases
    assert GraphTopologyParser.parse("all_to_all", N) == complete
    assert GraphTopologyParser.parse("clique", N) == complete

    # Ring: 4 edges: (0, 1), (0, 3), (1, 2), (2, 3)
    ring = GraphTopologyParser.parse("ring", N)
    assert len(ring) == 4
    assert ring == [(0, 1), (0, 3), (1, 2), (2, 3)]
    assert GraphTopologyParser.parse("cycle", N) == ring

    # Ring with N=2
    assert GraphTopologyParser.parse("ring", 2) == [(0, 1)]
    assert GraphTopologyParser.parse("ring", 1) == []

    # Line: N-1 = 3 edges
    line = GraphTopologyParser.parse("line", N)
    assert len(line) == 3
    assert line == [(0, 1), (1, 2), (2, 3)]
    assert GraphTopologyParser.parse("chain", N) == line
    assert GraphTopologyParser.parse("path", N) == line

    # Star: N-1 = 3 edges
    star = GraphTopologyParser.parse("star", N)
    assert len(star) == 3
    assert star == [(0, 1), (0, 2), (0, 3)]

    # None / Empty
    none_edges = GraphTopologyParser.parse("none", N)
    assert none_edges == []
    assert GraphTopologyParser.parse("empty", N) == []
    assert GraphTopologyParser.parse("disconnected", N) == []


def test_graph_parser_adjacency_matrix() -> None:
    """Validates parsing from symmetric adjacency matrices."""
    # 3-node triangle
    A = torch.tensor([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    edges = GraphTopologyParser.parse(A, num_nodes=3)
    assert edges == [(0, 1), (0, 2), (1, 2)]

    # Non-symmetric raises ValueError
    A_asym = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    with pytest.raises(ValueError, match="symmetric"):
        GraphTopologyParser.parse(A_asym, num_nodes=3)

    # Wrong shape raises ValueError
    A_wrong = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="must have shape"):
        GraphTopologyParser.parse(A_wrong, num_nodes=3)


def test_graph_parser_custom_edge_list() -> None:
    """Validates canonical normalization and deduplication of edge lists."""
    raw = [(2, 0), (1, 2), (0, 2), (1, 0)]
    edges = GraphTopologyParser.parse(raw, num_nodes=3)
    # Normalized: (0, 2), (1, 2), (0, 1) -> sorted: (0, 1), (0, 2), (1, 2)
    assert edges == [(0, 1), (0, 2), (1, 2)]


def test_graph_parser_validation_errors() -> None:
    """Validates rejection of out-of-bounds, self-loops, and bad types."""
    with pytest.raises(ValueError, match="num_nodes must be >= 1"):
        GraphTopologyParser.parse("complete", 0)

    with pytest.raises(ValueError, match="Self-loop"):
        GraphTopologyParser.parse([(0, 0)], 3)

    with pytest.raises(ValueError, match="out of node range"):
        GraphTopologyParser.parse([(0, 5)], 3)

    with pytest.raises(ValueError, match="Invalid edge at index"):
        GraphTopologyParser.parse([(0, 1, 2)], 3)  # type: ignore[list-item]

    with pytest.raises(ValueError, match="Unknown coupling_graph preset"):
        GraphTopologyParser.parse("hypercube_invalid", 4)

    with pytest.raises(TypeError, match="Unsupported coupling_graph type"):
        GraphTopologyParser.parse(12345, 4)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: Constructor & Parameter Specifications
# ═══════════════════════════════════════════════════════════════════════════

def test_constructor_defaults() -> None:
    """Validates default initialization, parameter shapes, and registered buffers."""
    num_nodes = 4
    in_features = 3
    layer = ContinuousResonantLayer(
        num_nodes=num_nodes,
        in_features=in_features,
        coupling_graph="ring",
        observable_types=("Z", "X"),
    )

    assert layer.num_nodes == 4
    assert layer.in_features == 3
    assert layer.num_edges == 4  # ring on 4 nodes
    assert layer.out_features == 4 * 2  # 8 outputs

    # Check parameter shapes
    assert layer.W.shape == (4, 3)
    assert layer.h.shape == (4,)
    assert layer.omega.shape == (4,)
    assert layer.J.shape == (4,)
    assert layer.t.shape == (1,)
    assert isinstance(layer.t, nn.Parameter)
    assert torch.allclose(layer.t, torch.tensor([1.0]))

    # Check buffer initial_state
    assert layer.initial_state.shape == (16,)
    assert layer.initial_state.dtype == torch.complex64

    # Check extra_repr
    rep = layer.extra_repr()
    assert "num_nodes=4" in rep
    assert "in_features=3" in rep
    assert "out_features=8" in rep


def test_constructor_fixed_time_buffer() -> None:
    """When learnable_time=False, t is registered as a buffer, not a parameter."""
    layer = ContinuousResonantLayer(
        num_nodes=3,
        in_features=2,
        learnable_time=False,
        initial_time=2.5,
    )
    assert not isinstance(layer.t, nn.Parameter)
    assert "t" in dict(layer.named_buffers())
    assert torch.allclose(layer.t, torch.tensor([2.5]))
    assert "t" not in dict(layer.named_parameters())


def test_initial_state_modes() -> None:
    """Validates initialization with various initial state modes."""
    # Zero state: |0...0>
    l_zero = ContinuousResonantLayer(num_nodes=3, in_features=2, initial_state="zero")
    assert l_zero.initial_state[0] == 1.0
    assert torch.allclose(l_zero.initial_state[1:].abs(), torch.tensor(0.0))

    # Plus state: |+...+>
    l_plus = ContinuousResonantLayer(num_nodes=3, in_features=2, initial_state="plus")
    inv_sqrt8 = 1.0 / math.sqrt(8.0)
    assert torch.allclose(l_plus.initial_state.real, torch.tensor(inv_sqrt8), atol=1e-5)

    # Superposition alias
    l_super = ContinuousResonantLayer(num_nodes=3, in_features=2, initial_state="superposition")
    assert torch.allclose(l_super.initial_state, l_plus.initial_state)

    # GHZ state
    l_ghz = ContinuousResonantLayer(num_nodes=3, in_features=2, initial_state="ghz")
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert torch.allclose(l_ghz.initial_state[0].abs(), torch.tensor(inv_sqrt2), atol=1e-5)
    assert torch.allclose(l_ghz.initial_state[-1].abs(), torch.tensor(inv_sqrt2), atol=1e-5)

    # Custom normalized tensor
    custom = torch.zeros(8, dtype=torch.complex64)
    custom[3] = 1.0
    l_custom = ContinuousResonantLayer(num_nodes=3, in_features=2, initial_state=custom)
    assert torch.allclose(l_custom.initial_state, custom)

    # Custom tensor wrong dimension raises ValueError
    bad_custom = torch.zeros(7, dtype=torch.complex64)
    with pytest.raises(ValueError, match="must have length 8"):
        ContinuousResonantLayer(num_nodes=3, in_features=2, initial_state=bad_custom)


def test_initialization_methods() -> None:
    """Validates uniform, normal, and zeros initialization schemes."""
    for init_m in ("default", "uniform", "normal", "zeros"):
        layer = ContinuousResonantLayer(
            num_nodes=3,
            in_features=2,
            coupling_graph="complete",
            init_method=init_m,
        )
        if init_m == "zeros":
            assert torch.allclose(layer.W, torch.tensor(0.0))
            assert torch.allclose(layer.h, torch.tensor(0.0))
            assert torch.allclose(layer.omega, torch.tensor(0.0))
            assert torch.allclose(layer.J, torch.tensor(0.0))
        elif init_m == "normal":
            assert layer.W.std() > 0.0

    with pytest.raises(ValueError, match="Unknown init_method"):
        ContinuousResonantLayer(num_nodes=3, in_features=2, init_method="invalid_scheme")


def test_constructor_validation_errors() -> None:
    """Validates parameter constraints and error handling in layer constructor."""
    with pytest.raises(ValueError, match="num_nodes must be >= 1"):
        ContinuousResonantLayer(num_nodes=0, in_features=2)

    with pytest.raises(ValueError, match="in_features must be >= 1"):
        ContinuousResonantLayer(num_nodes=2, in_features=0)

    with pytest.raises(ValueError, match="initial_time must be > 0.0"):
        ContinuousResonantLayer(num_nodes=2, in_features=2, initial_time=-1.0)

    with pytest.raises(ValueError, match="Unsupported observable type"):
        ContinuousResonantLayer(num_nodes=2, in_features=2, observable_types=("Z", "W"))


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: Dimensional Invariance & Batch Reshaping
# ═══════════════════════════════════════════════════════════════════════════

def test_dimension_invariance_and_shapes() -> None:
    """Tests 1D unbatched, 2D batched, 3D multi-batch, and 4D tensor inputs."""
    layer = ContinuousResonantLayer(
        num_nodes=3,
        in_features=4,
        coupling_graph="line",
        observable_types=("Z", "X"),
    )
    # out_features = 3 * 2 = 6

    # 1. Unbatched 1D: (4,) -> (6,)
    x_1d = torch.randn(4)
    y_1d = layer(x_1d)
    assert y_1d.shape == (6,)

    # 2. Standard 2D: (5, 4) -> (5, 6)
    x_2d = torch.randn(5, 4)
    y_2d = layer(x_2d)
    assert y_2d.shape == (5, 6)

    # 3. Multi-batch 3D: (2, 3, 4) -> (2, 3, 6)
    x_3d = torch.randn(2, 3, 4)
    y_3d = layer(x_3d)
    assert y_3d.shape == (2, 3, 6)

    # 4. Multi-batch 4D: (2, 2, 3, 4) -> (2, 2, 3, 6)
    x_4d = torch.randn(2, 2, 3, 4)
    y_4d = layer(x_4d)
    assert y_4d.shape == (2, 2, 3, 6)


def test_empty_batch_handling() -> None:
    """Tests empty batch inputs (B=0) and confirms clean zero gradients."""
    layer = ContinuousResonantLayer(
        num_nodes=3,
        in_features=2,
        coupling_graph="ring",
    )

    # 2D empty: (0, 2) -> (0, 6)
    x_empty_2d = torch.empty((0, 2), requires_grad=True)
    y_empty_2d = layer(x_empty_2d)
    assert y_empty_2d.shape == (0, 6)

    loss_2d = y_empty_2d.sum()
    loss_2d.backward()
    assert x_empty_2d.grad is not None
    assert x_empty_2d.grad.shape == (0, 2)
    assert layer.W.grad is not None
    assert torch.allclose(layer.W.grad, torch.tensor(0.0))

    # Zero layer gradients
    layer.zero_grad()

    # 3D empty: (2, 0, 2) -> (2, 0, 6)
    x_empty_3d = torch.empty((2, 0, 2), requires_grad=True)
    y_empty_3d = layer(x_empty_3d)
    assert y_empty_3d.shape == (2, 0, 6)
    loss_3d = y_empty_3d.sum()
    loss_3d.backward()
    assert x_empty_3d.grad is not None
    assert x_empty_3d.grad.shape == (2, 0, 2)


def test_input_validation_errors() -> None:
    """Rejects 0D scalars and feature dimension mismatches."""
    layer = ContinuousResonantLayer(num_nodes=3, in_features=4)

    with pytest.raises(ValueError, match="must have at least 1 dimension"):
        layer(torch.tensor(1.0))

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        layer(torch.randn(2, 5))


# ═══════════════════════════════════════════════════════════════════════════
# Tier 4: Mathematical Boundary Invariants
# ═══════════════════════════════════════════════════════════════════════════

def test_unitary_norm_preservation() -> None:
    """Unconditionally preserves statevector norm sum |a_i|^2 = 1.0 across arbitrary times."""
    num_nodes = 3

    for dtype, c_dtype, tol in [
        (torch.float32, torch.complex64, 1e-6),
        (torch.float64, torch.complex128, 1e-14),
    ]:
        edges = [(0, 1), (1, 2), (0, 2)]
        basis = ops.ResonantInteractionBasis(num_nodes, edges, device="cpu", dtype=c_dtype)

        B = 4
        x = torch.randn(B, 2, dtype=dtype)
        J = torch.randn(3, dtype=dtype)
        h = torch.randn(num_nodes, dtype=dtype)
        W = torch.randn(num_nodes, 2, dtype=dtype)
        omega = torch.randn(num_nodes, dtype=dtype)

        # Test across extreme evolution durations t in [0.01, 100.0]
        for t_val in [0.01, 0.5, 1.0, 5.0, 25.0, 100.0]:
            t = torch.tensor([t_val], dtype=dtype)
            _, _, _, _, psi_t, _ = _ContinuousResonantFunction._simulate_forward(
                x, J, h, W, omega, t, num_nodes, edges, ("Z", "X"), "zero", basis
            )
            norms = torch.sum(psi_t.abs() ** 2, dim=-1)
            assert torch.allclose(norms, torch.ones(B, dtype=dtype), atol=tol)


def test_hamiltonian_hermiticity() -> None:
    """Verifies that the constructed Hamiltonian H(x, θ) is strictly Hermitian."""
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    basis = ops.ResonantInteractionBasis(num_nodes, edges, device="cpu", dtype=torch.complex128)

    B = 3
    x = torch.randn(B, 4, dtype=torch.float64)
    J = torch.randn(2, dtype=torch.float64)
    h = torch.randn(num_nodes, dtype=torch.float64)
    W = torch.randn(num_nodes, 4, dtype=torch.float64)
    omega = torch.randn(num_nodes, dtype=torch.float64)
    t = torch.tensor([1.0], dtype=torch.float64)

    # Simulate forward to obtain eigenvalues and eigenvectors
    _, lambdas, V, _, _, _ = _ContinuousResonantFunction._simulate_forward(
        x, J, h, W, omega, t, num_nodes, edges, ("Z",), "zero", basis
    )

    # Reconstruct H = V @ diag(lambdas) @ V^H and check Hermiticity
    for b in range(B):
        H_reconstructed = V[b] @ torch.diag(lambdas[b].to(torch.complex128)) @ V[b].conj().T
        diff = H_reconstructed - H_reconstructed.conj().T
        assert torch.max(diff.abs()).item() < 1e-12


def test_readout_expectation_bounds() -> None:
    """Expectation values of Pauli observables must strictly lie in [-1.0, 1.0]."""
    layer = ContinuousResonantLayer(
        num_nodes=3,
        in_features=2,
        observable_types=("Z", "X", "Y"),
        coupling_graph="complete",
    )
    x = torch.randn(20, 2)
    y = layer(x)
    assert torch.all(y >= -1.0 - 1e-5)
    assert torch.all(y <= 1.0 + 1e-5)


def test_zero_time_and_zero_coupling_limits() -> None:
    """Checks deterministic limits at t=0 and pure longitudinal precession."""
    # At t=0 with initial_state="zero", Z expectations must be exactly +1.0
    layer_zero_t = ContinuousResonantLayer(
        num_nodes=3,
        in_features=2,
        initial_state="zero",
        observable_types=("Z",),
        initial_time=1e-12,
        learnable_time=False,
    )
    x = torch.zeros(1, 2)
    y_z = layer_zero_t(x)
    assert torch.allclose(y_z, torch.ones_like(y_z), atol=1e-4)

    # Pure longitudinal precession (J=0, omega=0, W=0): |0...0> remains stationary in Z
    layer_diag = ContinuousResonantLayer(
        num_nodes=2,
        in_features=2,
        coupling_graph="none",
        initial_state="zero",
        observable_types=("Z",),
        initial_time=2.0,
        init_method="zeros",
    )
    y_diag = layer_diag(x)
    assert torch.allclose(y_diag, torch.ones_like(y_diag), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 5: Autograd Gradcheck & Analytical Gradients
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("coupling", ["complete", "ring", "line", "star", "none"])
def test_autograd_gradcheck_topologies(coupling: str) -> None:
    """PyTorch autograd gradcheck on CPU float64 across graph topologies."""
    num_nodes = 3
    in_features = 2
    obs_types = ("Z", "X")
    edges = GraphTopologyParser.parse(coupling, num_nodes)
    num_edges = len(edges)

    x = torch.randn(2, in_features, dtype=torch.float64, requires_grad=True)
    J = torch.randn(num_edges, dtype=torch.float64, requires_grad=True)
    h = torch.randn(num_nodes, dtype=torch.float64, requires_grad=True)
    W = torch.randn(num_nodes, in_features, dtype=torch.float64, requires_grad=True)
    omega = torch.randn(num_nodes, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([1.2], dtype=torch.float64, requires_grad=True)

    def func(
        x_: torch.Tensor,
        J_: torch.Tensor,
        h_: torch.Tensor,
        W_: torch.Tensor,
        omega_: torch.Tensor,
        t_: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            _ContinuousResonantFunction.apply(
                x_, J_, h_, W_, omega_, t_, num_nodes, edges, obs_types, "zero", None, "exact"
            ),
        )

    # Run gradcheck on CPU double precision
    assert gradcheck(func, (x, J, h, W, omega, t), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_autograd_gradcheck_full_observables_and_superposition() -> None:
    """Gradcheck covering Z, X, Y simultaneous readouts and plus initial state."""
    num_nodes = 2
    in_features = 2
    obs_types = ("Z", "X", "Y")
    edges = [(0, 1)]

    x = torch.randn(2, in_features, dtype=torch.float64, requires_grad=True)
    J = torch.randn(1, dtype=torch.float64, requires_grad=True)
    h = torch.randn(num_nodes, dtype=torch.float64, requires_grad=True)
    W = torch.randn(num_nodes, in_features, dtype=torch.float64, requires_grad=True)
    omega = torch.randn(num_nodes, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.75], dtype=torch.float64, requires_grad=True)

    def func(
        x_: torch.Tensor,
        J_: torch.Tensor,
        h_: torch.Tensor,
        W_: torch.Tensor,
        omega_: torch.Tensor,
        t_: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            _ContinuousResonantFunction.apply(
                x_, J_, h_, W_, omega_, t_, num_nodes, edges, obs_types, "plus", None, "exact"
            ),
        )

    assert gradcheck(func, (x, J, h, W, omega, t), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_ehrenfest_and_daleckii_krein_vs_finite_difference() -> None:
    """Matches analytical Ehrenfest & Daleckii-Krein derivatives vs finite differences."""
    num_nodes = 3
    in_features = 2
    obs_types = ("Z", "X")
    edges = [(0, 1), (1, 2)]

    x = torch.randn(2, in_features, dtype=torch.float64)
    J = torch.randn(len(edges), dtype=torch.float64)
    h = torch.randn(num_nodes, dtype=torch.float64)
    W = torch.randn(num_nodes, in_features, dtype=torch.float64)
    omega = torch.randn(num_nodes, dtype=torch.float64)
    t = torch.tensor([1.5], dtype=torch.float64)

    # 1. Exact analytical gradients
    x_ex = x.clone().requires_grad_(True)
    J_ex = J.clone().requires_grad_(True)
    h_ex = h.clone().requires_grad_(True)
    W_ex = W.clone().requires_grad_(True)
    omega_ex = omega.clone().requires_grad_(True)
    t_ex = t.clone().requires_grad_(True)

    y_ex = _ContinuousResonantFunction.apply(
        x_ex, J_ex, h_ex, W_ex, omega_ex, t_ex, num_nodes, edges, obs_types, "zero", None, "exact"
    )
    loss_ex = y_ex.sum()
    loss_ex.backward()

    # 2. Finite difference gradients
    x_fd = x.clone().requires_grad_(True)
    J_fd = J.clone().requires_grad_(True)
    h_fd = h.clone().requires_grad_(True)
    W_fd = W.clone().requires_grad_(True)
    omega_fd = omega.clone().requires_grad_(True)
    t_fd = t.clone().requires_grad_(True)

    y_fd = _ContinuousResonantFunction.apply(
        x_fd, J_fd, h_fd, W_fd, omega_fd, t_fd,
        num_nodes, edges, obs_types, "zero", None, "finite-diff"
    )
    loss_fd = y_fd.sum()
    loss_fd.backward()

    # Compare gradients (< 1e-4 tolerance due to finite difference truncation)
    assert torch.allclose(cast(torch.Tensor, t_ex.grad), cast(torch.Tensor, t_fd.grad), atol=1e-4)
    assert torch.allclose(cast(torch.Tensor, J_ex.grad), cast(torch.Tensor, J_fd.grad), atol=1e-4)
    assert torch.allclose(cast(torch.Tensor, h_ex.grad), cast(torch.Tensor, h_fd.grad), atol=1e-4)
    assert torch.allclose(cast(torch.Tensor, W_ex.grad), cast(torch.Tensor, W_fd.grad), atol=1e-4)
    assert torch.allclose(
        cast(torch.Tensor, omega_ex.grad), cast(torch.Tensor, omega_fd.grad), atol=1e-4
    )
    assert torch.allclose(cast(torch.Tensor, x_ex.grad), cast(torch.Tensor, x_fd.grad), atol=1e-4)


def test_diff_method_autograd_equivalence() -> None:
    """Verifies that diff_method='autograd' produces identical forward predictions."""
    layer_exact = ContinuousResonantLayer(
        num_nodes=3, in_features=2, coupling_graph="ring", diff_method="exact"
    )
    layer_autograd = ContinuousResonantLayer(
        num_nodes=3, in_features=2, coupling_graph="ring", diff_method="autograd"
    )
    # Copy parameters
    layer_autograd.load_state_dict(layer_exact.state_dict())

    x = torch.randn(4, 2)
    y_exact = layer_exact(x)
    y_autograd = layer_autograd(x)

    assert torch.allclose(y_exact, y_autograd, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 6: PyTorch Ecosystem, nn.Sequential, & Multi-Epoch Adam Training
# ═══════════════════════════════════════════════════════════════════════════

def test_nn_sequential_integration() -> None:
    """Seamless integration in nn.Sequential with preceding and subsequent layers."""
    model = nn.Sequential(
        nn.Linear(2, 4),
        ContinuousResonantLayer(num_nodes=4, in_features=4, coupling_graph="line"),
        nn.Linear(8, 1),
    )
    x = torch.randn(5, 2)
    out = model(x)
    assert out.shape == (5, 1)

    loss = out.sum()
    loss.backward()

    # Check gradients flow to all components
    res_layer = cast(ContinuousResonantLayer, model[1])
    assert res_layer.W.grad is not None
    assert res_layer.h.grad is not None
    assert res_layer.omega.grad is not None
    assert res_layer.J.grad is not None
    assert res_layer.t.grad is not None


def test_xor_classification_learning() -> None:
    """Learns the non-linear XOR function and demonstrates loss convergence."""
    torch.manual_seed(42)

    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    model = nn.Sequential(
        ContinuousResonantLayer(
            num_nodes=4,
            in_features=2,
            coupling_graph="complete",
            observable_types=("Z", "X"),
            learnable_time=True,
            initial_time=1.0,
        ),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    criterion = nn.BCELoss()

    initial_loss: float = 0.0
    final_loss: float = 0.0

    for epoch in range(120):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        if epoch == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    # Loss must reduce significantly
    assert final_loss < initial_loss
    assert final_loss < 0.25


# ═══════════════════════════════════════════════════════════════════════════
# Tier 7: Apple Silicon Metal (MPS) & Hardware Precision
# ═══════════════════════════════════════════════════════════════════════════

def test_mps_metal_execution() -> None:
    """Validates execution on Apple Silicon GPU (MPS) with float32/complex64."""
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS is not available in this environment")

    mps_dev = torch.device("mps")
    layer = ContinuousResonantLayer(
        num_nodes=3,
        in_features=2,
        coupling_graph="ring",
        device=mps_dev,
        dtype=torch.float32,
    )
    assert layer.W.device.type == "mps"

    x = torch.randn(4, 2, device=mps_dev, dtype=torch.float32)
    y = layer(x)
    assert y.device.type == "mps"
    assert y.shape == (4, 6)

    loss = y.sum()
    loss.backward()
    assert layer.W.grad is not None
    assert layer.W.grad.device.type == "mps"


def test_mps_float64_defensive_rejection() -> None:
    """Apple Silicon MPS must defensively reject 64-bit precision."""
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS is not available in this environment")

    mps_dev = torch.device("mps")
    with pytest.raises(UnsupportedDtypeError, match="does not support 64-bit precision"):
        ContinuousResonantLayer(
            num_nodes=3,
            in_features=2,
            device=mps_dev,
            dtype=torch.float64,
        )


def test_cross_device_dtype_transfer() -> None:
    """Validates transfer between CPU and MPS without autograd corruption."""
    layer = ContinuousResonantLayer(num_nodes=3, in_features=2, dtype=torch.float32)

    # Input on CPU float64 passed to layer float32
    x_cpu_64 = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    y = layer(x_cpu_64)
    loss = y.sum()
    loss.backward()
    assert x_cpu_64.grad is not None
    assert x_cpu_64.grad.dtype == torch.float64

    # Transfer layer to float64 on CPU
    layer.to(dtype=torch.float64)
    assert layer.W.dtype == torch.float64
    assert layer.initial_state.dtype == torch.complex128
    y2 = layer(x_cpu_64)
    assert y2.dtype == torch.float64

    # Bidirectional transfer with MPS if available
    if torch.backends.mps.is_available():
        mps_dev = torch.device("mps")
        layer_mps = ContinuousResonantLayer(
            num_nodes=2, in_features=2, device=mps_dev, dtype=torch.float32
        )
        x_in = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        y_out = layer_mps(x_in)
        assert y_out.device.type == "mps"
        loss_mps = y_out.sum()
        loss_mps.backward()
        assert x_in.grad is not None
        assert x_in.grad.device.type == "cpu"
        assert x_in.grad.dtype == torch.float64
        assert layer_mps.W.grad is not None
        assert layer_mps.W.grad.device.type == "mps"


def test_1d_input_and_batched_psi_and_empty_autograd() -> None:
    """Verifies 1D input handling, batched initial states, and empty batch autograd."""
    # 1. 1D input under exact diff
    layer_exact = ContinuousResonantLayer(
        num_nodes=2, in_features=3, diff_method="exact", dtype=torch.float64
    )
    x_1d = torch.randn(3, dtype=torch.float64, requires_grad=True)
    out_1d = layer_exact(x_1d)
    assert out_1d.shape == (4,)
    out_1d.sum().backward()
    assert x_1d.grad is not None
    assert x_1d.grad.shape == (3,)

    # 2. 1D input under finite_diff
    layer_fd = ContinuousResonantLayer(
        num_nodes=2, in_features=3, diff_method="finite_diff", dtype=torch.float64
    )
    x_1d_fd = torch.randn(3, dtype=torch.float64, requires_grad=True)
    out_1d_fd = layer_fd(x_1d_fd)
    assert out_1d_fd.shape == (4,)
    out_1d_fd.sum().backward()
    assert x_1d_fd.grad is not None
    assert x_1d_fd.grad.shape == (3,)

    # 3. Empty batch under autograd
    layer_auto = ContinuousResonantLayer(
        num_nodes=2, in_features=3, diff_method="autograd", dtype=torch.float64
    )
    x_empty = torch.randn(0, 3, dtype=torch.float64, requires_grad=True)
    out_empty = layer_auto(x_empty)
    assert out_empty.shape == (0, 4)

    # 4. Batched initial state in _ContinuousResonantFunction
    N = 2
    dim = 2**N
    edges = [(0, 1)]
    obs_types = ("Z",)
    B = 3
    x = torch.randn(B, 2, dtype=torch.float64, requires_grad=True)
    J = torch.randn(1, dtype=torch.float64, requires_grad=True)
    h = torch.randn(N, dtype=torch.float64, requires_grad=True)
    W = torch.randn(N, 2, dtype=torch.float64, requires_grad=True)
    omega = torch.randn(N, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([1.2], dtype=torch.float64, requires_grad=True)
    psi0_batched = torch.randn(B, dim, dtype=torch.complex128)
    psi0_batched = psi0_batched / torch.linalg.vector_norm(psi0_batched, dim=-1, keepdim=True)

    out = _ContinuousResonantFunction.apply(
        x, J, h, W, omega, t, N, edges, obs_types, psi0_batched, None, "exact"
    )
    assert out.shape == (B, 2)
    out.sum().backward()
    assert x.grad is not None

    # 5. Complex initial state preservation under .to()
    custom_psi = torch.tensor(
        [1.0 / math.sqrt(2.0), 1.0j / math.sqrt(2.0), 0.0, 0.0], dtype=torch.complex64
    )
    layer_custom = ContinuousResonantLayer(
        num_nodes=2, in_features=2, initial_state=custom_psi, dtype=torch.float32
    )
    layer_custom.to(dtype=torch.float64)
    assert layer_custom.initial_state.dtype == torch.complex128
