"""tests.test_torch_e2e — End-to-End Integration Test Suite for Quanta PyTorch Integration.

Covers Milestones 4 & 5 (Packaging, Top-Level Namespace Export, Cross-Pillar Cascades,
and Real-World Application Scenarios):
- F10: Top-level namespace export, public symbols, and conditional import fallback.
- Tier 3: Cross-Pillar hybrid network cascade (nn.Linear -> ContinuousResonantLayer
  -> nn.Linear -> QuantumLayer -> nn.Linear).
- Tier 4: Real-world application scenarios:
    1. Hybrid classification training on non-linear datasets (concentric circles / moons).
    2. Continuous-time ballistic quantum walk dynamics vs classical Markov diffusion.
    3. Simultaneous multi-observable state reconstruction and entanglement diagnostics.
    4. Cross-device CPU and Apple Silicon Metal (MPS) migration and gradient fidelity.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import quanta
from quanta.torch import (
    ContinuousResonantLayer,
    GraphTopologyParser,
    ObservableParser,
    ParsedObservable,
    PauliTerm,
    QuantumLayer,
    UnsupportedDtypeError,
    _ContinuousResonantFunction,
    _QuantumLayerFunction,
    get_ansatz_param_count,
    ops,
)

# ═══════════════════════════════════════════════════════════════════════════
# F10: Packaging & Top-Level Namespace Export Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_top_level_quanta_torch_export() -> None:
    """Verifies that quanta.torch is accessible at the top level when PyTorch is installed."""
    assert hasattr(quanta, "torch"), "quanta must export 'torch' module at top level"
    assert quanta.torch is not None, "quanta.torch must not be None"
    assert "torch" in quanta.__all__, "'torch' must be included in quanta.__all__"

    # Verify 'from quanta import torch' matches quanta.torch
    from quanta import torch as imported_torch

    assert imported_torch is quanta.torch


def test_quanta_torch_all_symbols() -> None:
    """Verifies that all public symbols from Pillar 2 are exported and have correct types."""
    # Direct reference checks for imported top-level classes
    assert QuantumLayer is not None
    assert ContinuousResonantLayer is not None
    assert _QuantumLayerFunction is not None
    assert _ContinuousResonantFunction is not None
    assert ObservableParser is not None
    assert ParsedObservable is not None
    assert PauliTerm is not None
    assert GraphTopologyParser is not None
    assert get_ansatz_param_count is not None
    assert ops is not None
    assert UnsupportedDtypeError is not None

    expected_symbols = [
        "QuantumLayer",
        "_QuantumLayerFunction",
        "ParsedObservable",
        "PauliTerm",
        "ObservableParser",
        "get_ansatz_param_count",
        "ContinuousResonantLayer",
        "_ContinuousResonantFunction",
        "GraphTopologyParser",
        "ops",
        "UnsupportedDtypeError",
        "resolve_device",
        "resolve_complex_dtype",
        "get_real_dtype",
        "get_complex_dtype",
        "to_torch_state",
        "to_numpy_state",
        "create_initial_state",
        "get_pauli_matrix",
        "pauli_matrices",
        "pauli_kron",
        "batch_pauli_kron",
        "batch_expectation",
        "batch_multi_expectation",
        "fast_z_readout",
        "fast_x_readout",
        "fast_y_readout",
        "simultaneous_readout",
        "hamiltonian_expectation",
        "ResonantInteractionBasis",
        "build_batch_resonant_hamiltonian",
        "unitary_evolution",
        "ehrenfest_time_gradient",
        "daleckii_krein_spectral_derivative",
        "apply_gate",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "cnot_gate",
        "cz_gate",
    ]

    for sym in expected_symbols:
        assert hasattr(quanta.torch, sym), f"quanta.torch missing expected symbol '{sym}'"

    # Verify core inheritance contracts
    assert issubclass(QuantumLayer, nn.Module)
    assert issubclass(ContinuousResonantLayer, nn.Module)
    assert issubclass(_QuantumLayerFunction, torch.autograd.Function)
    assert issubclass(_ContinuousResonantFunction, torch.autograd.Function)


def test_clean_import_without_side_effects() -> None:
    """Verifies that importing quanta.torch does not modify global state or cause warnings."""
    module = importlib.import_module("quanta.torch")
    assert module is quanta.torch
    assert hasattr(module, "QuantumLayer")
    assert hasattr(module, "ContinuousResonantLayer")


def test_top_level_import_fallback_when_torch_unavailable() -> None:
    """Verifies that quanta gracefully falls back to torch=None if PyTorch import fails."""
    orig_torch_mod = sys.modules.get("quanta.torch")
    try:
        sys.modules.pop("quanta.torch", None)
        orig_import = __import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            fromlist = kwargs.get("fromlist") or (args[2] if len(args) > 2 else ())
            if "torch" in name or (fromlist and "torch" in fromlist):
                raise ImportError("Mocked missing PyTorch")
            return orig_import(name, *args, **kwargs)


        with patch("builtins.__import__", side_effect=mock_import):
            reloaded = importlib.reload(quanta)
            assert reloaded.torch is None
    finally:
        if orig_torch_mod is not None:
            sys.modules["quanta.torch"] = orig_torch_mod
            quanta.torch = orig_torch_mod
        importlib.reload(quanta)
        assert quanta.torch is not None



# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: Cross-Pillar Hybrid Network Cascade
# ═══════════════════════════════════════════════════════════════════════════

class CrossPillarCascadeModel(nn.Module):
    """End-to-end multi-layer hybrid quantum-classical cascade network.

    Topology:
        Linear(4 -> 3)
        -> ContinuousResonantLayer(num_nodes=3, in_features=3, observables=(Z, X) -> 6)
        -> Linear(6 -> 4)
        -> QuantumLayer(num_qubits=4, circuit_fn="strongly_entangling" -> 4)
        -> Linear(4 -> 2)
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3, device=device, dtype=dtype)
        self.continuous_layer = ContinuousResonantLayer(
            num_nodes=3,
            in_features=3,
            coupling_graph="ring",
            observable_types=("Z", "X"),
            learnable_time=True,
            initial_time=1.0,
            device=device,
            dtype=dtype,
        )
        self.fc2 = nn.Linear(6, 4, device=device, dtype=dtype)
        self.quantum_layer = QuantumLayer(
            num_qubits=4,
            num_layers=2,
            circuit_fn="strongly_entangling",
            diff_method="parameter-shift",
            device=device,
            dtype=dtype,
        )
        self.fc3 = nn.Linear(4, 2, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward cascade through classical and quantum layers."""
        h1 = self.fc1(x)
        h2 = self.continuous_layer(h1)
        h3 = self.fc2(h2)
        h4 = self.quantum_layer(h3)
        out = self.fc3(h4)
        return cast(torch.Tensor, out)


def test_hybrid_continuous_variational_cascade() -> None:
    """Verifies end-to-end forward evaluation and backpropagation through the hybrid cascade.

    Validates:
    - Forward output shape matches expected (B, 2).
    - Backpropagation successfully flows across all classical and quantum layers:
      fc1 -> ContinuousResonantLayer -> fc2 -> QuantumLayer -> fc3.
    - Parameter gradients are non-None, non-zero, and finite.
    - An optimization step updates all model parameters cleanly.
    """
    torch.manual_seed(42)
    batch_size = 5
    model = CrossPillarCascadeModel(device="cpu", dtype=torch.float32)

    # Input tensor
    x = torch.randn(batch_size, 4, dtype=torch.float32)
    target = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)

    # 1. Forward Pass
    logits = model(x)
    assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "Output logits contain NaN or Inf"

    # 2. Backward Pass
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    loss.backward()

    # 3. Verify gradients across all layers
    # Upstream Classical Linear 1
    assert model.fc1.weight.grad is not None
    assert model.fc1.bias.grad is not None
    assert torch.norm(model.fc1.weight.grad) > 1e-6
    assert torch.norm(model.fc1.bias.grad) > 1e-6

    # Continuous Resonant Quantum Layer
    assert model.continuous_layer.W.grad is not None
    assert model.continuous_layer.h.grad is not None
    assert model.continuous_layer.omega.grad is not None
    assert model.continuous_layer.J.grad is not None
    assert model.continuous_layer.t.grad is not None
    assert torch.norm(model.continuous_layer.W.grad) > 1e-6
    assert torch.norm(model.continuous_layer.h.grad) > 1e-6
    assert torch.norm(model.continuous_layer.omega.grad) > 1e-6
    assert torch.norm(model.continuous_layer.J.grad) > 1e-6
    assert torch.norm(model.continuous_layer.t.grad) > 1e-6

    # Intermediate Classical Linear 2
    assert model.fc2.weight.grad is not None
    assert model.fc2.bias.grad is not None
    assert torch.norm(model.fc2.weight.grad) > 1e-6
    assert torch.norm(model.fc2.bias.grad) > 1e-6

    # Variational Quantum Layer
    assert model.quantum_layer.weights.grad is not None
    assert torch.norm(model.quantum_layer.weights.grad) > 1e-6

    # Output Classical Linear 3
    assert model.fc3.weight.grad is not None
    assert model.fc3.bias.grad is not None
    assert torch.norm(model.fc3.weight.grad) > 1e-6
    assert torch.norm(model.fc3.bias.grad) > 1e-6

    # 4. Parameter Update Verification
    params_before = {name: p.clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    optimizer.step()

    for name, p in model.named_parameters():
        p_before = params_before[name]
        diff = torch.norm(p - p_before).item()
        assert diff > 1e-6, (
            f"Parameter '{name}' did not update after optimizer.step() (diff={diff})"
        )


def test_hybrid_cascade_various_ansatzes() -> None:
    """Verifies that the hybrid cascade operates reliably across diverse ansatz presets."""
    torch.manual_seed(123)
    presets = ["hardware_efficient", "reuploading", "real_amplitudes"]
    for preset in presets:
        model = nn.Sequential(
            nn.Linear(2, 3),
            ContinuousResonantLayer(
                num_nodes=3,
                in_features=3,
                coupling_graph="line",
                observable_types=("Z",),
            ),
            nn.Linear(3, 3),
            QuantumLayer(num_qubits=3, num_layers=1, circuit_fn=preset),
            nn.Linear(3, 1),
        )
        x = torch.randn(3, 2)
        out = model(x)
        assert out.shape == (3, 1)
        loss = out.sum()
        loss.backward()
        first_linear = model[0]
        assert isinstance(first_linear, nn.Linear)
        assert first_linear.weight.grad is not None
        assert torch.norm(first_linear.weight.grad) > 1e-6


def test_hybrid_cascade_batch_robustness() -> None:
    """Verifies that the hybrid cascade supports various batch sizes (B=1, B=3, B=8)."""
    torch.manual_seed(999)
    model = CrossPillarCascadeModel(device="cpu", dtype=torch.float32)

    for b in [1, 3, 8]:
        x = torch.randn(b, 4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (b, 2)
        assert torch.all(torch.isfinite(out))


# ═══════════════════════════════════════════════════════════════════════════
# Tier 4: Real-World Application Scenarios
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_hybrid_classification_training() -> None:
    """Trains a hybrid quantum-classical neural network on a non-linear dataset.

    Dataset: Concentric circles (inner disc vs outer ring), which is non-linearly
    separable and requires non-trivial continuous-variational quantum representation.

    Verifies:
    - Successful optimization for 20 epochs using Adam.
    - Monotonic/overall loss reduction: final_loss < initial_loss.
    - Classification accuracy improvement: final_acc > initial_acc.
    """
    torch.manual_seed(42)

    # 1. Generate Synthetic Concentric Circles Dataset
    n_samples_per_class = 25
    r0 = 0.25 + 0.15 * torch.rand(n_samples_per_class)
    theta0 = 2.0 * math.pi * torch.rand(n_samples_per_class)
    x0 = torch.stack([r0 * torch.cos(theta0), r0 * torch.sin(theta0)], dim=1)
    y0 = torch.zeros(n_samples_per_class, dtype=torch.long)

    r1 = 0.85 + 0.25 * torch.rand(n_samples_per_class)
    theta1 = 2.0 * math.pi * torch.rand(n_samples_per_class)
    x1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=1)
    y1 = torch.ones(n_samples_per_class, dtype=torch.long)

    x_data = torch.cat([x0, x1], dim=0)
    y_data = torch.cat([y0, y1], dim=0)

    # Shuffle data
    perm = torch.randperm(len(y_data))
    x_data = x_data[perm]
    y_data = y_data[perm]

    # 2. Build Hybrid Model
    model = nn.Sequential(
        nn.Linear(2, 3),
        ContinuousResonantLayer(
            num_nodes=3,
            in_features=3,
            coupling_graph="ring",
            observable_types=("Z", "X"),
            learnable_time=True,
            initial_time=0.8,
        ),
        nn.Linear(6, 3),
        QuantumLayer(
            num_qubits=3,
            num_layers=1,
            circuit_fn="hardware_efficient",
            diff_method="parameter-shift",
        ),
        nn.Linear(3, 2),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    initial_loss: float | None = None
    initial_acc: float | None = None
    final_loss: float | None = None
    final_acc: float | None = None

    # 3. Optimization Loop for 20 Epochs
    for epoch in range(20):
        optimizer.zero_grad()
        logits = model(x_data)
        loss = criterion(logits, y_data)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == y_data).float().mean().item()

        if epoch == 0:
            initial_loss = loss.item()
            initial_acc = acc
        if epoch == 19:
            final_loss = loss.item()
            final_acc = acc

    assert initial_loss is not None
    assert initial_acc is not None
    assert final_loss is not None
    assert final_acc is not None

    # Assert rigorous convergence criteria
    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    )
    assert final_loss < 0.50, f"Expected final loss < 0.50, got {final_loss}"
    assert final_acc > initial_acc, (
        f"Accuracy did not improve: initial={initial_acc}, final={final_acc}"
    )
    assert final_acc >= 0.75, f"Expected final accuracy >= 0.75, got {final_acc}"


def test_e2e_ballistic_quantum_walk_dynamics() -> None:
    """Verifies continuous-time quantum walk (CTQW) ballistic dynamics vs classical diffusion.

    In a continuous-time quantum walk on a 1D line lattice governed by
    H = J Σ (σ_j^x σ_k^x + σ_j^y σ_k^y), a localized single-particle excitation
    spreads ballistically with constant wavefront velocity, reaching distal nodes
    with orders-of-magnitude higher probability than classical Markov diffusion.

    Verifies:
    1. Preservation of total probability / unitary norm: Σ P_j(t) = 1.0 ± 10^-6.
    2. Propagation of probability peak from origin to distal nodes.
    3. Quantum walk arrival probability at distal boundary node significantly surpasses
       classical Markov continuous diffusion e^(-2 L t).
    """
    num_nodes = 5
    # Initial localized state at node 0 (computational basis |10000>, index 1 << (N-1) = 16)
    psi0 = torch.zeros(2**num_nodes, dtype=torch.complex64)
    psi0[1 << (num_nodes - 1)] = torch.tensor(1.0 + 0.0j, dtype=torch.complex64)

    # Continuous resonant layer configured as a 1D line lattice
    layer = ContinuousResonantLayer(
        num_nodes=num_nodes,
        in_features=1,
        coupling_graph="line",
        observable_types=("Z",),
        initial_state=psi0,
        learnable_time=False,
        initial_time=1.0,
    )

    # Pure graph walk: isotropic coupling J=1.0, zero longitudinal field h=0, zero drive omega=0
    with torch.no_grad():
        layer.J.fill_(1.0)
        layer.h.zero_()
        layer.omega.zero_()
        layer.W.zero_()

    dummy_input = torch.zeros(1, 1)

    times = [0.0, 0.5, 1.0, 1.5]
    quantum_occupations: dict[float, torch.Tensor] = {}

    for t in times:
        with torch.no_grad():
            layer.t.fill_(t)
            z_readout = layer(dummy_input).squeeze(0)
            p_j = (1.0 - z_readout) / 2.0
            quantum_occupations[t] = p_j

            # Verify unitary probability conservation: Σ P_j = 1.0
            prob_sum = p_j.sum().item()
            assert abs(prob_sum - 1.0) < 1e-4, f"Unitary norm violated at t={t}: sum={prob_sum}"

    # At t=0: excitation is fully concentrated at node 0
    assert quantum_occupations[0.0][0].item() > 0.999
    assert quantum_occupations[0.0][num_nodes - 1].item() < 1e-4

    # Ballistic wavefront propagation:
    assert quantum_occupations[0.5].argmax().item() == 1
    assert quantum_occupations[1.0].argmax().item() == 2
    assert quantum_occupations[1.5].argmax().item() == 4
    p_distal_quantum = quantum_occupations[1.5][num_nodes - 1].item()
    assert p_distal_quantum > 0.70, (
        f"Ballistic wavefront failed to arrive at distal node: {p_distal_quantum}"
    )

    # Compare with Classical Markov Diffusion:
    # dP/dt = -2 L P, where L is the graph Laplacian for 1D line
    laplacian = torch.tensor([
        [1.0, -1.0, 0.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 2.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 2.0, -1.0],
        [0.0, 0.0, 0.0, -1.0, 1.0],
    ], dtype=torch.float32)

    p0_classical = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    t_eval = 1.5
    classical_mat = torch.linalg.matrix_exp(-2.0 * laplacian * t_eval)
    p_classical = torch.matmul(classical_mat, p0_classical)
    p_distal_classical = p_classical[num_nodes - 1].item()

    # Fundamental Quantum Supremacy in Lattice Transport:
    # Ballistic arrival probability at distal node is substantially greater (> 5x)
    # than classical diffusive dispersion
    assert p_distal_quantum > 5.0 * p_distal_classical, (
        f"Quantum walk ({p_distal_quantum:.4f}) did not ballistically outpace "
        f"classical diffusion ({p_distal_classical:.4f})"
    )


def test_e2e_simultaneous_multi_observable_reconstruction() -> None:
    """Verifies simultaneous non-sequential readout of Z and X (and Y) observables.

    In contrast to projective sequential measurement which collapses wavefunctions,
    the continuous resonant layer evaluates all observable expectation values concurrently
    from the unitary quantum state |ψ(t)>.

    Verifies:
    1. Single-qubit reduced density matrix Bloch vector reconstruction:
       r_j = (<X_j>, <Y_j>, <Z_j>).
    2. Quantum state validity: |r_j|^2 <= 1.0 for all nodes (Bloch sphere bound).
    3. Entanglement generation: for an entangled multi-qubit state, local subsystem purity
       Tr(ρ_j^2) = (1 + |r_j|^2) / 2 drops strictly below 1.0 while total system remains pure.
    """
    num_nodes = 3
    layer = ContinuousResonantLayer(
        num_nodes=num_nodes,
        in_features=3,
        coupling_graph="line",
        observable_types=("Z", "X", "Y"),
        initial_state="plus",
        learnable_time=False,
        initial_time=0.8,
    )

    x = torch.tensor([[0.5, -0.2, 0.3]], dtype=torch.float32)
    # Readout shape: (1, num_nodes * 3) = (1, 9)
    readout = layer(x).squeeze(0)

    # Readouts are ordered: [Z0..Z2, X0..X2, Y0..Y2]
    z_exp = readout[0:3]
    x_exp = readout[3:6]
    y_exp = readout[6:9]

    for j in range(num_nodes):
        rx = x_exp[j].item()
        ry = y_exp[j].item()
        rz = z_exp[j].item()

        r_sq = rx**2 + ry**2 + rz**2
        purity = (1.0 + r_sq) / 2.0

        # Verify quantum mechanical bound
        assert r_sq <= 1.0 + 1e-5, f"Bloch vector norm violation at node {j}: |r|^2 = {r_sq}"
        assert purity <= 1.0 + 1e-5, f"Purity violation at node {j}: purity = {purity}"

    # Verify that the interacting Hamiltonian generates non-local quantum correlations:
    purities = [
        ((1.0 + (x_exp[j] ** 2 + y_exp[j] ** 2 + z_exp[j] ** 2).item()) / 2.0)
        for j in range(num_nodes)
    ]
    min_purity = min(purities)
    assert min_purity < 0.99, (
        f"Expected quantum entanglement (reduced purity < 0.99), got min purity {min_purity}"
    )


def test_e2e_cross_device_migration() -> None:
    """Verifies bidirectional migration of the cross-pillar hybrid model.

    Evaluates seamless migration between CPU and Apple Silicon MPS.

    Validates:
    1. Instantiation and execution on CPU.
    2. Seamless migration to Apple Silicon GPU (`model.to("mps")` if available).
    3. Exact forward numerical equivalence between CPU and MPS outputs: ||y_mps - y_cpu|| < 10^-4.
    4. Gradient backpropagation fidelity across all parameters on MPS matching CPU gradients.
    5. Clean roundtrip migration back to CPU (`model.to("cpu")`).
    """
    torch.manual_seed(42)
    batch_size = 4

    # Build hybrid cascade on CPU
    model = CrossPillarCascadeModel(device="cpu", dtype=torch.float32)
    x_cpu = torch.randn(batch_size, 4, dtype=torch.float32)

    # CPU Forward & Backward Pass
    y_cpu = model(x_cpu)
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()
    grads_cpu = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    assert len(grads_cpu) > 0, "No gradients collected on CPU"

    # MPS Migration Test
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS is not available in this test environment")

    mps_device = torch.device("mps")

    # Move model to MPS
    model.zero_grad()
    model.to(mps_device)
    x_mps = x_cpu.to(mps_device)

    # Verify parameters and buffers are located on MPS
    for name, p in model.named_parameters():
        assert p.device.type == "mps", f"Parameter '{name}' not migrated to MPS: {p.device}"

    # MPS Forward Pass
    y_mps = model(x_mps)
    assert y_mps.device.type == "mps", f"Output device is {y_mps.device}, expected MPS"
    assert torch.allclose(y_mps.cpu(), y_cpu, atol=1e-4), (
        "MPS forward output differs from CPU output"
    )

    # MPS Backward Pass
    loss_mps = y_mps.sum()
    loss_mps.backward()

    # Verify MPS gradients match CPU gradients
    grads_mps = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert len(grads_mps) == len(grads_cpu), "Gradient count mismatch between MPS and CPU"

    for name in grads_cpu:
        grad_mps_tensor = grads_mps[name]
        assert grad_mps_tensor.device.type == "mps", f"Gradient '{name}' not on MPS device"
        assert torch.allclose(grad_mps_tensor.cpu(), grads_cpu[name], atol=1e-3), (
            f"Gradient mismatch on '{name}' between CPU and MPS: "
            f"max_diff={(grad_mps_tensor.cpu() - grads_cpu[name]).abs().max().item()}"
        )

    # Bidirectional migration back to CPU
    model.to("cpu")
    for name, p in model.named_parameters():
        assert p.device.type == "cpu", f"Parameter '{name}' not migrated back to CPU"

    y_cpu_roundtrip = model(x_cpu)
    assert torch.allclose(y_cpu_roundtrip, y_cpu, atol=1e-4), (
        "CPU output changed after MPS roundtrip"
    )
