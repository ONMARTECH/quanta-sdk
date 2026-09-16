"""tests/test_torch_brain.py — Unit & Integration Tests for BiomorphicResonantBrain.

Tests the biophysical quantum brain architecture:
- Dual-hemisphere (left/right lobes) and corpus callosum coupling.
- Neuromodulatory chemical fields (dopamine, norepinephrine, serotonin).
- Hemodynamic oxygenation energy conservation.
- Unitary norm preservation and collective consensus readout.
- End-to-end autograd backpropagation and multi-epoch training.
"""

import math

import pytest
import torch
import torch.nn as nn

from quanta.torch import ops
from quanta.torch.brain import (
    BiomorphicResonantBrain,
    QuantumREMSleep,
    QuantumZenoAttention,
)


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


# ═══════════════════════════════════════════════════════════════════════════
# QuantumREMSleep Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_rem_sleep_initialization_and_repr():
    brain = BiomorphicResonantBrain(in_features=3, num_left_qubits=1, num_right_qubits=2)
    rem = QuantumREMSleep(
        brain, sleep_cycles=12, learning_rate=0.02, orthogonalization_weight=1.5
    )
    assert rem.brain_module is brain
    assert rem.sleep_cycles == 12
    assert rem.learning_rate == 0.02
    assert rem.orthogonalization_weight == 1.5
    assert len(rem.memory_states) == 0
    repr_str = repr(rem)
    assert "QuantumREMSleep" in repr_str
    assert "sleep_cycles=12" in repr_str
    assert "stored_states=0" in repr_str


def test_rem_sleep_register_and_clear_memory():
    brain = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)
    rem = QuantumREMSleep(brain)

    # 1D statevector
    state_1d = torch.randn(16, dtype=brain.complex_dtype)
    rem.register_memory_state(state_1d)
    assert len(rem.memory_states) == 1
    assert torch.isclose(
        torch.linalg.norm(rem.memory_states[0]),
        torch.tensor(1.0, dtype=brain.real_dtype),
        atol=1e-5,
    )

    # 2D batch of statevectors
    state_2d = torch.randn(3, 16, dtype=brain.complex_dtype)
    rem.register_memory_state(state_2d)
    assert len(rem.memory_states) == 4

    # Invalid tensor dimensions
    with pytest.raises(ValueError, match="Expected 1D or 2D state tensor"):
        rem.register_memory_state(torch.randn(2, 2, 16))

    # Clear memory
    rem.clear_memory_states()
    assert len(rem.memory_states) == 0


def test_rem_sleep_validation_on_empty_or_single_state():
    brain = BiomorphicResonantBrain(in_features=3, num_left_qubits=1, num_right_qubits=2)
    rem = QuantumREMSleep(brain)

    with pytest.raises(ValueError, match="At least two memory states are required"):
        rem.sleep()

    rem.register_memory_state(torch.randn(8, dtype=brain.complex_dtype))
    with pytest.raises(ValueError, match="At least two memory states are required"):
        rem.sleep()


def test_rem_sleep_forward_routing():
    brain = BiomorphicResonantBrain(in_features=3, num_left_qubits=1, num_right_qubits=2)
    rem = QuantumREMSleep(brain)

    x = torch.randn(5, 3, dtype=brain.real_dtype)
    out_brain = brain(x)
    out_rem = rem(x)

    assert torch.allclose(out_brain["consensus"], out_rem["consensus"])
    assert torch.allclose(out_brain["readout_vector"], out_rem["readout_vector"])
    assert torch.allclose(out_brain["state"], out_rem["state"])

    # Test 1D input routing
    x_1d = torch.randn(3, dtype=brain.real_dtype)
    out_1d = rem(x_1d)
    assert out_1d["consensus"].shape == (1,)


def test_rem_sleep_annealing_overlap_reduction():
    torch.manual_seed(42)
    brain = BiomorphicResonantBrain(in_features=2, num_left_qubits=1, num_right_qubits=1)
    rem = QuantumREMSleep(brain, sleep_cycles=15, learning_rate=0.08)

    # Create two highly overlapping memory statevectors
    dim = brain.dim  # 4
    psi_a = torch.zeros(dim, dtype=brain.complex_dtype)
    psi_a[0] = 1.0

    psi_b = torch.zeros(dim, dtype=brain.complex_dtype)
    psi_b[0] = 0.95
    psi_b[1] = math.sqrt(1.0 - 0.95**2)

    rem.register_memory_state(psi_a)
    rem.register_memory_state(psi_b)
    assert len(rem.memory_states) == 2

    res = rem.sleep(cycles=15)
    assert "initial_overlap" in res
    assert "final_overlap" in res
    assert "cycles" in res
    assert "loss_history" in res
    assert "retention_estimate" in res

    assert res["cycles"] == 15
    assert len(res["loss_history"]) == 15
    # Annealing must decrease overlap
    assert res["final_overlap"] < res["initial_overlap"]
    assert res["retention_estimate"] >= 0.50
    # Memory states were updated with consolidated representations
    assert len(rem.memory_states) == 2


def test_rem_sleep_explicit_memory_states_param():
    brain = BiomorphicResonantBrain(in_features=2, num_left_qubits=1, num_right_qubits=1)
    rem = QuantumREMSleep(brain)

    dim = brain.dim
    s1 = torch.randn(dim, dtype=brain.complex_dtype)
    s2 = torch.randn(dim, dtype=brain.complex_dtype)

    res = rem.sleep(memory_states=[s1, s2], cycles=5)
    assert res["cycles"] == 5
    assert len(rem.memory_states) == 0  # self.memory_states untouched when passed explicitly


def test_rem_sleep_autograd_training_after_sleep():
    # Verify that sleep annealing leaves brain fully trainable via autograd
    brain = BiomorphicResonantBrain(in_features=2, num_left_qubits=1, num_right_qubits=2)
    rem = QuantumREMSleep(brain, sleep_cycles=3, learning_rate=0.01)

    s1 = torch.randn(brain.dim, dtype=brain.complex_dtype)
    s2 = torch.randn(brain.dim, dtype=brain.complex_dtype)
    rem.sleep(memory_states=[s1, s2], cycles=3)

    # Subsequent waking task backpropagation
    x = torch.randn(4, 2, dtype=brain.real_dtype, requires_grad=True)
    out = rem(x)
    loss = out["consensus"].sum()
    loss.backward()

    assert x.grad is not None
    assert brain.W_left.grad is not None
    assert brain.J_right.grad is not None
    assert brain.J_callosum.grad is not None


def test_rem_sleep_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS not available")

    dev = torch.device("mps")
    brain = BiomorphicResonantBrain(
        in_features=2, num_left_qubits=1, num_right_qubits=1, device=dev
    )
    rem = QuantumREMSleep(brain, sleep_cycles=5, device=dev)

    s1 = torch.randn(brain.dim, dtype=torch.complex64, device=dev)
    s2 = torch.randn(brain.dim, dtype=torch.complex64, device=dev)
    rem.register_memory_state(s1)
    rem.register_memory_state(s2)

    res = rem.sleep(cycles=5)
    assert res["final_overlap"] <= res["initial_overlap"]

    x = torch.randn(2, 2, device=dev, dtype=torch.float32)
    out = rem(x)
    assert out["consensus"].device.type == "mps"


# ═══════════════════════════════════════════════════════════════════════════
# QuantumZenoAttention Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_zeno_attention_initialization_and_repr():
    attn = QuantumZenoAttention(
        dim=8, observation_frequency=12.0, dopamine_coupling=0.75, num_heads=2
    )
    assert attn.dim == 8
    assert attn.num_heads == 2
    assert attn.head_dim == 4
    assert float(attn.obs_freq.detach()) == pytest.approx(12.0)
    assert float(attn.dopamine_coupling.detach()) == pytest.approx(0.75)
    repr_str = repr(attn)
    assert "QuantumZenoAttention" in repr_str
    assert "dim=8" in repr_str
    assert "num_heads=2" in repr_str


def test_zeno_attention_validation_errors():
    with pytest.raises(ValueError, match="dim must be positive"):
        QuantumZenoAttention(dim=0)

    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        QuantumZenoAttention(dim=7, num_heads=2)


def test_zeno_attention_forward_shapes():
    attn = QuantumZenoAttention(dim=6, num_heads=2)

    # 1D tensor [dim]
    x_1d = torch.randn(6, dtype=attn.real_dtype)
    out_1d = attn(x_1d)
    assert out_1d.shape == (6,)

    # 2D tensor [B, dim]
    x_2d = torch.randn(4, 6, dtype=attn.real_dtype)
    out_2d = attn(x_2d)
    assert out_2d.shape == (4, 6)

    # 3D tensor [B, L, dim]
    x_3d = torch.randn(4, 5, 6, dtype=attn.real_dtype)
    out_3d = attn(x_3d)
    assert out_3d.shape == (4, 5, 6)


def test_zeno_attention_pinning_vs_dopamine_tunneling():
    # Test Theorem 3: low dopamine pins hypothesis (QZE), high dopamine tunnels (Anti-Zeno)
    attn = QuantumZenoAttention(dim=4, observation_frequency=25.0, dopamine_coupling=1.5)
    x = torch.randn(2, 4, dtype=attn.real_dtype)

    # Case 1: Low / Zero Dopamine -> Zeno Attentional Pinning
    diag_low = attn(x, dopamine=0.0, return_diagnostics=True)
    p_zeno_low = float(diag_low["P_zeno"].mean().item())
    assert p_zeno_low > 0.85, f"Expected high Zeno pinning, got {p_zeno_low}"
    diff_pinned = torch.norm(diag_low["output"] - diag_low["h_pinned"])
    diff_tunnel = torch.norm(diag_low["output"] - diag_low["h_tunnel"])
    assert diff_pinned < diff_tunnel

    # Case 2: High Dopamine Surge -> Anti-Zeno Tunneling Exploration
    diag_high = attn(x, dopamine=5.0, return_diagnostics=True)
    p_zeno_high = float(diag_high["P_zeno"].mean().item())
    assert p_zeno_high < 0.20, f"Expected low Zeno pinning under dopamine surge, got {p_zeno_high}"
    nu_eff_low = float(diag_low["nu_eff"].mean().item())
    nu_eff_high = float(diag_high["nu_eff"].mean().item())
    assert nu_eff_high < nu_eff_low

    # Case 3: Verify monotonic ordering: P_zeno(surge) < P_zeno(baseline)
    assert p_zeno_high < p_zeno_low


def test_zeno_attention_diagnostics_dictionary():
    attn = QuantumZenoAttention(dim=4, observation_frequency=10.0)
    x = torch.randn(3, 4, dtype=attn.real_dtype)

    diag = attn(x, return_diagnostics=True)
    assert "output" in diag
    assert "P_zeno" in diag
    assert "nu_eff" in diag
    assert "phi" in diag
    assert "dopamine" in diag
    assert "h_pinned" in diag
    assert "h_tunnel" in diag
    assert "h_explore" in diag

    assert diag["output"].shape == (3, 4)
    assert diag["P_zeno"].shape == (3, 1)
    assert diag["nu_eff"].shape == (3, 1)
    assert diag["phi"].shape == (3, 1)

    # Verify mathematical invariant: phi = pi / nu_eff
    expected_phi = math.pi / diag["nu_eff"]
    assert torch.allclose(diag["phi"], expected_phi)

    # Verify mathematical invariant: P_zeno = exp(-1 / nu_eff)
    expected_p = torch.exp(-1.0 / diag["nu_eff"])
    assert torch.allclose(diag["P_zeno"], expected_p)


def test_zeno_attention_autograd_differentiability():
    torch.manual_seed(42)
    attn = QuantumZenoAttention(dim=4, observation_frequency=8.0, dopamine_coupling=0.6)

    x = torch.randn(3, 4, dtype=attn.real_dtype, requires_grad=True)
    dopamine = torch.tensor([[0.5], [1.2], [0.1]], dtype=attn.real_dtype, requires_grad=True)

    out = attn(x, dopamine=dopamine)
    loss = out.sum()
    loss.backward()

    # Inputs and dopamine gradients
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert dopamine.grad is not None
    assert torch.all(torch.isfinite(dopamine.grad))

    # Module parameter gradients
    assert attn.obs_freq.grad is not None
    assert attn.dopamine_coupling.grad is not None
    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None
    assert attn.kickback_proj.weight.grad is not None


def test_zeno_attention_fixed_frequency():
    attn = QuantumZenoAttention(dim=4, observation_frequency=15.0, learnable_frequency=False)
    assert not isinstance(attn.obs_freq, nn.Parameter)
    assert "obs_freq" in dict(attn.named_buffers())
    x = torch.randn(2, 4, dtype=attn.real_dtype)
    out = attn(x)
    assert out.shape == (2, 4)


def test_zeno_attention_external_dopamine_formats():
    attn = QuantumZenoAttention(dim=4)
    x = torch.randn(2, 3, 4, dtype=attn.real_dtype)

    # Float scalar
    out_scalar = attn(x, dopamine=0.4)
    assert out_scalar.shape == (2, 3, 4)

    # 0D tensor
    out_0d = attn(x, dopamine=torch.tensor(0.4, dtype=attn.real_dtype))
    assert out_0d.shape == (2, 3, 4)

    # 1D tensor
    out_1d = attn(x, dopamine=torch.tensor([0.2, 0.8], dtype=attn.real_dtype))
    assert out_1d.shape == (2, 3, 4)


def test_zeno_attention_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS not available")

    dev = torch.device("mps")
    attn = QuantumZenoAttention(dim=4, device=dev)
    x = torch.randn(2, 4, device=dev, dtype=torch.float32, requires_grad=True)
    out = attn(x)
    assert out.device.type == "mps"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.device.type == "mps"


def test_integrated_brain_sleep_and_zeno_pipeline():
    # Holistic integration:
    # QuantumZenoAttention -> BiomorphicResonantBrain wrapped in QuantumREMSleep
    torch.manual_seed(42)
    attention = QuantumZenoAttention(dim=4, observation_frequency=12.0)
    brain = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)
    sleep_layer = QuantumREMSleep(brain, sleep_cycles=5)

    x = torch.randn(3, 4, dtype=brain.real_dtype)
    attended_x = attention(x)
    assert attended_x.shape == (3, 4)

    out = sleep_layer(attended_x)
    assert out["consensus"].shape == (3, 1)
    assert out["readout_vector"].shape == (3, 4)
    assert out["state"].shape == (3, 16)

    # Register output states and run sleep cycle
    sleep_layer.register_memory_state(out["state"])
    assert len(sleep_layer.memory_states) == 3
    sleep_result = sleep_layer.sleep(cycles=5)
    assert sleep_result["final_overlap"] <= sleep_result["initial_overlap"]


def test_brain_repr_and_edge_cases():
    brain = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)
    assert "BiomorphicResonantBrain" in repr(brain)

    # 4D tensor raises ValueError in QuantumZenoAttention
    attn = QuantumZenoAttention(dim=4)
    with pytest.raises(ValueError, match="Expected 1D, 2D, or 3D tensor"):
        attn(torch.randn(2, 2, 2, 4))

    # Unsupported dopamine type raises TypeError
    with pytest.raises(TypeError, match="Unsupported dopamine type"):
        attn(torch.randn(2, 4), dopamine="invalid_dopamine")  # type: ignore[arg-type]

    # 3D dopamine tensor
    out_3d_dopamine = attn(torch.randn(2, 3, 4), dopamine=torch.rand(2, 3, 1))
    assert out_3d_dopamine.shape == (2, 3, 4)

    # 1D dopamine tensor with different length
    out_1d_mismatch = attn(torch.randn(2, 3, 4), dopamine=torch.tensor([0.3]))
    assert out_1d_mismatch.shape == (2, 3, 4)

    # Dtype auto-conversion when float32 input passed to float64 module
    attn_f64 = QuantumZenoAttention(dim=4, dtype=torch.float64, device="cpu")
    out_conv = attn_f64(torch.randn(2, 4, dtype=torch.float32))
    assert out_conv.dtype == torch.float64


def test_mps_unsupported_64bit_dtype_errors():
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS not available")

    # MPS does not support float64
    with pytest.raises(ops.UnsupportedDtypeError):
        BiomorphicResonantBrain(
            in_features=2, num_left_qubits=1, num_right_qubits=1, device="mps", dtype=torch.float64
        )

    brain = BiomorphicResonantBrain(
        in_features=2, num_left_qubits=1, num_right_qubits=1, device="mps", dtype=torch.float32
    )
    with pytest.raises(ops.UnsupportedDtypeError):
        QuantumREMSleep(brain, device="mps", dtype=torch.float64)

    with pytest.raises(ops.UnsupportedDtypeError):
        QuantumZenoAttention(dim=4, device="mps", dtype=torch.float64)

