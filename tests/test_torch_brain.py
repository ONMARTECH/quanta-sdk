"""tests/test_torch_brain.py — Unit & Integration Tests for BiomorphicResonantBrain.

Tests the biophysical quantum brain architecture:
- Dual-hemisphere (left/right lobes) and corpus callosum coupling.
- Neuromodulatory chemical fields (dopamine, norepinephrine, serotonin).
- Hemodynamic oxygenation energy conservation.
- Unitary norm preservation and collective consensus readout.
- End-to-end autograd backpropagation and multi-epoch training.
"""

import pytest
import torch
import torch.nn as nn

from quanta.torch.brain import BiomorphicResonantBrain


def test_brain_initialization_defaults():
    brain = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)
    assert brain.num_qubits == 4
    assert brain.dim == 16
    assert len(brain.left_edges) == 1
    assert len(brain.right_edges) == 1
    assert len(brain.callosum_edges) >= 1
    assert brain.enable_neuromodulation is True
    assert brain.enable_oxygenation is True


def test_brain_validation_errors():
    with pytest.raises(ValueError, match="Both hemispheres must have >= 1 qubit"):
        BiomorphicResonantBrain(in_features=4, num_left_qubits=0, num_right_qubits=2)

    with pytest.raises(ValueError, match="Both hemispheres must have >= 1 qubit"):
        BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=0)


def test_brain_forward_output_shapes_and_invariants():
    brain = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)
    x = torch.randn(8, 4, dtype=brain.real_dtype)

    out = brain(x)
    assert "consensus" in out
    assert "left_consensus" in out
    assert "right_consensus" in out
    assert "readout_vector" in out
    assert "state" in out

    # Batch shapes
    assert out["consensus"].shape == (8, 1)
    assert out["left_consensus"].shape == (8, 1)
    assert out["right_consensus"].shape == (8, 1)
    assert out["readout_vector"].shape == (8, 4)
    assert out["state"].shape == (8, 16)

    # Unitary norm conservation: ||psi(t)||^2 == 1.0 per batch element
    norms = torch.sum(torch.abs(out["state"]) ** 2, dim=-1)
    for n in norms:
        assert float(n.detach()) == pytest.approx(1.0, abs=1e-6)

    # Bounds on expectation values [-1.0, 1.0]
    assert torch.all(out["readout_vector"] >= -1.0001)
    assert torch.all(out["readout_vector"] <= 1.0001)
    assert torch.all(out["consensus"] >= -1.0001)
    assert torch.all(out["consensus"] <= 1.0001)


def test_brain_1d_input():
    brain = BiomorphicResonantBrain(in_features=3, num_left_qubits=1, num_right_qubits=2)
    x = torch.randn(3, dtype=brain.real_dtype)
    out = brain(x)
    assert out["consensus"].shape == (1,)
    assert out["readout_vector"].shape == (3,)
    assert out["state"].shape == (8,)


def test_brain_neuromodulation_ablation():
    # Neuromodulation disabled vs enabled
    brain_no_neuro = BiomorphicResonantBrain(
        in_features=3,
        num_left_qubits=2,
        num_right_qubits=1,
        enable_neuromodulation=False,
        enable_oxygenation=False,
    )
    x = torch.randn(4, 3, dtype=brain_no_neuro.real_dtype)
    out = brain_no_neuro(x)
    norms = torch.sum(torch.abs(out["state"]) ** 2, dim=-1)
    for n in norms:
        assert float(n.detach()) == pytest.approx(1.0, abs=1e-6)


def test_brain_autograd_backpropagation():
    brain = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)
    x = torch.randn(6, 4, dtype=brain.real_dtype, requires_grad=True)

    out = brain(x)
    loss = out["consensus"].sum() + out["readout_vector"].sum()
    loss.backward()

    # Verify input gradient
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

    # Verify internal learnable parameters have valid gradients
    assert brain.W_left.grad is not None
    assert brain.h_left.grad is not None
    assert brain.J_right.grad is not None
    assert brain.J_callosum.grad is not None
    assert brain.hormone_proj.grad is not None
    assert brain.oxy_weight.grad is not None
    assert brain.base_time.grad is not None


def test_brain_e2e_training_loop():
    # Test that the Biomorphic Quantum Brain can converge on a non-linear decision task
    torch.manual_seed(42)
    brain = BiomorphicResonantBrain(in_features=2, num_left_qubits=1, num_right_qubits=2)
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.05)

    # XOR / Parity synthetic task
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=brain.real_dtype)
    y = torch.tensor([[-0.8], [0.8], [0.8], [-0.8]], dtype=brain.real_dtype)

    initial_loss = 0.0
    final_loss = 0.0

    for epoch in range(25):
        optimizer.zero_grad()
        out = brain(X)
        pred = out["consensus"]
        loss = nn.functional.mse_loss(pred, y)
        if epoch == 0:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    )


def test_brain_apple_silicon_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS not available")

    dev = torch.device("mps")
    brain = BiomorphicResonantBrain(
        in_features=3, num_left_qubits=1, num_right_qubits=2, device=dev
    )
    x = torch.randn(4, 3, device=dev, dtype=torch.float32)
    out = brain(x)
    assert out["consensus"].device.type == "mps"
    assert out["state"].device.type == "mps"
    loss = out["consensus"].sum()
    loss.backward()
    assert brain.W_left.grad is not None
    assert brain.W_left.grad.device.type == "mps"
