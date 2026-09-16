"""tests/test_brain_autograd_stress.py — Empirical Autograd & Numerical Stress Test Suite.

Adversarial testing harness developed by Challenger 1 (Autograd & Numerical Challenger):
1. Autograd Backpropagation: Test gradient computation with respect to inputs, internal
   parameters, and dopamine signals under extreme input magnitudes and learning rates.
2. Degenerate Eigenvalue Limits: Test the Daleckii-Krein matrix spectral Fréchet derivative
   sinc parameterization under degenerate or near-degenerate Hamiltonian spectra (Delta_ab -> 0).
3. QuantumZenoAttention Boundaries: Stress-test extreme observation frequencies (nu_obs -> 0, 100),
   extreme dopamine surges (d = 0.0, 1.0, 50.0), and multi-head configurations.
4. Device compatibility: Verify operation and autograd on CPU and Apple Silicon MPS.
"""

from __future__ import annotations

import cmath
import math

import pytest
import torch

from quanta.torch import ops
from quanta.torch.brain import (
    BiomorphicResonantBrain,
    QuantumREMSleep,
    QuantumZenoAttention,
)
from quanta.torch.continuous import ContinuousResonantLayer

# ═══════════════════════════════════════════════════════════════════════════
# 1. Autograd Backpropagation & Extreme Numerical Inputs
# ═══════════════════════════════════════════════════════════════════════════


class TestAutogradBackpropagation:
    """Stress-tests autograd backward passes under extreme inputs and optimization conditions."""

    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e2, 1e4, 1e6])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_zeno_attention_extreme_inputs(self, scale: float, dtype: torch.dtype) -> None:
        """Tests that extreme input magnitudes do not produce NaNs or Infs in outputs/gradients."""
        attn = QuantumZenoAttention(dim=8, dtype=dtype)
        x = (torch.randn(3, 4, 8, dtype=dtype) * scale).requires_grad_()

        out = attn(x)
        assert isinstance(out, torch.Tensor)
        assert torch.isfinite(out).all(), f"Output contains non-finite values at scale {scale}"

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), f"x.grad contains non-finite values at scale {scale}"
        assert attn.q_proj.weight.grad is not None and torch.isfinite(attn.q_proj.weight.grad).all()
        assert attn.k_proj.weight.grad is not None and torch.isfinite(attn.k_proj.weight.grad).all()
        assert attn.v_proj.weight.grad is not None and torch.isfinite(attn.v_proj.weight.grad).all()
        assert (
            attn.kickback_proj.weight.grad is not None
            and torch.isfinite(attn.kickback_proj.weight.grad).all()
        )

    @pytest.mark.parametrize("d_val", [0.0, 0.01, 1.0, 10.0, 50.0, 100.0])
    def test_zeno_attention_dopamine_gradient_flow(self, d_val: float) -> None:
        """Verifies smooth autograd backpropagation directly through the dopamine tensor."""
        attn = QuantumZenoAttention(dim=6, observation_frequency=15.0, dopamine_coupling=0.8)
        x = torch.randn(2, 3, 6, dtype=attn.real_dtype, requires_grad=True)
        dopamine = torch.full((2, 3, 1), d_val, dtype=attn.real_dtype, requires_grad=True)

        out = attn(x, dopamine=dopamine)
        assert isinstance(out, torch.Tensor)
        assert torch.isfinite(out).all()

        loss = out.norm()
        loss.backward()

        assert dopamine.grad is not None
        assert torch.isfinite(dopamine.grad).all(), f"dopamine.grad not finite at d={d_val}"
        assert x.grad is not None and torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("lr", [1e-4, 1e-2, 1.0, 10.0])
    def test_rem_sleep_learning_rate_stability(self, lr: float) -> None:
        """Verifies closed-system REM sleep annealing under diverse learning rates."""
        brain = BiomorphicResonantBrain(in_features=3, num_left_qubits=1, num_right_qubits=2)
        rem = QuantumREMSleep(brain, sleep_cycles=4, learning_rate=lr)

        s1 = torch.randn(brain.dim, dtype=brain.complex_dtype)
        s2 = torch.randn(brain.dim, dtype=brain.complex_dtype)

        res = rem.sleep(memory_states=[s1, s2], cycles=4)
        has_nan = any(math.isnan(loss_val) for loss_val in res["loss_history"])
        has_inf = any(math.isinf(loss_val) for loss_val in res["loss_history"])
        assert not has_nan, f"NaN in loss history at lr={lr}"
        assert not has_inf, f"Inf in loss history at lr={lr}"
        assert 0.0 <= res["retention_estimate"] <= 1.0

    def test_rem_sleep_identical_states_adversarial(self) -> None:
        """Adversarially feeds identical states to test zero-distance collapse handling."""
        brain = BiomorphicResonantBrain(in_features=2, num_left_qubits=1, num_right_qubits=1)
        rem = QuantumREMSleep(brain, sleep_cycles=6, learning_rate=0.05)

        s = torch.randn(brain.dim, dtype=brain.complex_dtype)
        s = s / torch.linalg.norm(s)

        res = rem.sleep(memory_states=[s, s.clone()], cycles=6)
        assert res["final_overlap"] <= res["initial_overlap"] + 1e-6
        assert not any(math.isnan(loss_val) for loss_val in res["loss_history"])

    def test_rem_sleep_orthogonal_states_stability(self) -> None:
        """Verifies that already-orthogonal states produce stable zero/minimal loss without NaNs."""
        brain = BiomorphicResonantBrain(in_features=2, num_left_qubits=1, num_right_qubits=1)
        rem = QuantumREMSleep(brain, sleep_cycles=4, learning_rate=0.01)

        dim = brain.dim
        s1 = torch.zeros(dim, dtype=brain.complex_dtype)
        s1[0] = 1.0
        s2 = torch.zeros(dim, dtype=brain.complex_dtype)
        s2[1] = 1.0

        res = rem.sleep(memory_states=[s1, s2], cycles=4)
        assert not any(math.isnan(loss_val) for loss_val in res["loss_history"])
        assert res["retention_estimate"] >= 0.80


# ═══════════════════════════════════════════════════════════════════════════
# 2. Degenerate Eigenvalue Limits: Daleckii-Krein Spectral Fréchet Derivative
# ═══════════════════════════════════════════════════════════════════════════


class TestDegenerateEigenvalueLimits:
    """Verifies Daleckii-Krein matrix spectral Fréchet derivative under degenerate spectra."""

    def test_daleckii_krein_exact_degeneracy_analytical(self) -> None:
        """Tests Daleckii-Krein derivative when all eigenvalues are degenerate (H = c * I).

        Analytically, for H = c * I:
            d(exp(-i H t)) / d phi = -i * t * exp(-i c t) * Omega
        """
        dim = 4
        c = 3.14159
        t = 1.5
        evals_deg = torch.full((dim,), c, dtype=torch.float64)
        evecs_deg = torch.eye(dim, dtype=torch.complex128)

        # Arbitrary Hermitian drive perturbation Omega
        omega_mat = torch.randn(dim, dim, dtype=torch.complex128)
        omega_mat = omega_mat + omega_mat.conj().T

        dU_dk = ops.daleckii_krein_spectral_derivative(evals_deg, evecs_deg, t, omega_mat)
        dU_analytical = -1.0j * t * cmath.exp(-1.0j * c * t) * omega_mat

        diff = torch.norm(dU_dk - dU_analytical)
        assert diff.item() < 1e-12, f"Exact degeneracy mismatch: error = {diff.item():.2e}"

    @pytest.mark.parametrize("delta", [1.0, 1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 0.0])
    def test_daleckii_krein_smooth_limit_delta_to_zero(self, delta: float) -> None:
        """Tests that as spectral gap Delta_ab -> 0, derivative remains smooth and non-singular."""
        dim = 4
        c = 2.0
        t = 0.8
        evals = torch.tensor([c, c + delta, c + 2 * delta, c - delta], dtype=torch.float64)
        evecs = torch.eye(dim, dtype=torch.complex128)

        omega_mat = torch.randn(dim, dim, dtype=torch.complex128)
        omega_mat = omega_mat + omega_mat.conj().T

        dU = ops.daleckii_krein_spectral_derivative(evals, evecs, t, omega_mat)

        assert torch.isfinite(dU).all(), f"Non-finite dU at delta={delta}"
        assert not torch.isnan(dU).any()
        assert not torch.isinf(dU).any()

    def test_daleckii_krein_numerical_finite_difference(self) -> None:
        """Validates Daleckii-Krein derivative against numerical matrix exp finite-difference."""
        dim = 4
        t = 1.2
        torch.manual_seed(42)

        H = torch.randn(dim, dim, dtype=torch.complex128)
        H = (H + H.conj().T) / 2.0
        evals, evecs = torch.linalg.eigh(H)

        Omega = torch.randn(dim, dim, dtype=torch.complex128)
        Omega = (Omega + Omega.conj().T) / 2.0

        dU_dk = ops.daleckii_krein_spectral_derivative(evals, evecs, t, Omega)

        eps = 1e-5
        U_plus = torch.linalg.matrix_exp(-1.0j * (H + eps * Omega) * t)
        U_minus = torch.linalg.matrix_exp(-1.0j * (H - eps * Omega) * t)
        dU_fd = (U_plus - U_minus) / (2.0 * eps)

        rel_err = (torch.norm(dU_dk - dU_fd) / torch.norm(dU_dk)).item()
        assert rel_err < 1e-6, (
            f"Finite-difference validation failed: relative error = {rel_err:.2e}"
        )

    def test_continuous_layer_autograd_under_zero_hamiltonian(self) -> None:
        """Stress-tests ContinuousResonantLayer autograd when H = 0 (maximal degeneracy)."""
        layer = ContinuousResonantLayer(
            num_nodes=3,
            in_features=2,
            coupling_graph=[(0, 1), (1, 2)],
            observable_types=("Z", "X"),
            dtype=torch.float64,
        )

        with torch.no_grad():
            layer.J.zero_()
            layer.h.zero_()
            layer.W.zero_()
            layer.omega.zero_()

        x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        out = layer(x)
        assert torch.isfinite(out).all()

        loss = out.sum()
        loss.backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert layer.J.grad is not None and torch.isfinite(layer.J.grad).all()
        assert layer.h.grad is not None and torch.isfinite(layer.h.grad).all()
        assert layer.W.grad is not None and torch.isfinite(layer.W.grad).all()
        assert layer.omega.grad is not None and torch.isfinite(layer.omega.grad).all()
        assert layer.t.grad is not None and torch.isfinite(layer.t.grad).all()

    def test_biomorphic_brain_eigh_degeneracy_contrast(self) -> None:
        """Empirical contrast: PyTorch native eigh backward produces NaNs under degeneracy H=0.

        This demonstrates why Daleckii-Krein matrix spectral Fréchet derivative with sinc
        parameterization (as verified in ContinuousResonantLayer) is strictly necessary
        to guarantee non-singular gradient flow under degenerate spectra.
        """
        brain = BiomorphicResonantBrain(in_features=2, num_left_qubits=2, num_right_qubits=2)
        with torch.no_grad():
            brain.W_left.zero_()
            brain.h_left.zero_()
            brain.J_right.zero_()
            brain.J_callosum.zero_()
            brain.hormone_bias.zero_()
            brain.hormone_proj.zero_()

        x = torch.randn(2, 2, dtype=brain.real_dtype, requires_grad=True)
        out = brain(x)
        loss = out["consensus"].sum()
        loss.backward()

        assert brain.W_left.grad is not None
        assert brain.h_left.grad is not None
        # PyTorch native eigh backward produces non-finite gradients under exact degeneracy
        has_nan = (
            torch.isnan(brain.W_left.grad).any().item()
            or torch.isnan(brain.h_left.grad).any().item()
        )
        assert has_nan, "Expected native eigh autograd to produce NaNs under exact H=0 degeneracy"


# ═══════════════════════════════════════════════════════════════════════════
# 3. QuantumZenoAttention Boundary Stress Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantumZenoAttentionBoundaries:
    """Stress-tests boundary conditions for QuantumZenoAttention."""

    @pytest.mark.parametrize("freq", [1e-4, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0])
    def test_extreme_observation_frequencies(self, freq: float) -> None:
        """Tests wide observation frequency spectrum for numerical stability."""
        attn = QuantumZenoAttention(dim=8, observation_frequency=freq)
        x = torch.randn(3, 4, 8, dtype=attn.real_dtype, requires_grad=True)

        diag = attn(x, return_diagnostics=True)
        assert isinstance(diag, dict)
        assert torch.isfinite(diag["output"]).all()
        assert torch.isfinite(diag["P_zeno"]).all()
        assert torch.isfinite(diag["nu_eff"]).all()
        assert torch.isfinite(diag["phi"]).all()

        loss = diag["output"].sum()
        loss.backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert attn.obs_freq.grad is not None and torch.isfinite(attn.obs_freq.grad).all()

    @pytest.mark.parametrize("d_val", [0.0, 1.0, 50.0])
    def test_dopamine_surge_boundaries_and_ordering(self, d_val: float) -> None:
        """Stress-tests dopamine surge boundary conditions: d=0 (pinning) vs d=50 (tunneling)."""
        attn = QuantumZenoAttention(dim=8, observation_frequency=20.0, dopamine_coupling=1.0)
        x = torch.randn(2, 8, dtype=attn.real_dtype)

        diag = attn(x, dopamine=d_val, return_diagnostics=True)
        assert isinstance(diag, dict)
        p_zeno = float(diag["P_zeno"].mean().item())
        nu_eff = float(diag["nu_eff"].mean().item())

        if d_val == 0.0:
            assert p_zeno > 0.85, f"Expected strong Zeno pinning at d=0, got P_zeno={p_zeno}"
            assert nu_eff > 15.0
        elif d_val == 50.0:
            assert p_zeno < 0.01, (
                f"Expected complete Anti-Zeno collapse at d=50, got P_zeno={p_zeno}"
            )
            assert nu_eff <= 0.25

    @pytest.mark.parametrize(
        "dim,num_heads",
        [(8, 1), (8, 2), (8, 4), (16, 2), (16, 8), (32, 4), (64, 8)],
    )
    def test_multi_head_attention_configurations(self, dim: int, num_heads: int) -> None:
        """Verifies multi-head attention partitioning, score scaling, and output dimensionality."""
        attn = QuantumZenoAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(2, 5, dim, dtype=attn.real_dtype, requires_grad=True)

        out = attn(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 5, dim)
        assert torch.isfinite(out).all()

        loss = out.sum()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_zeno_attention_float32_softplus_underflow_boundary(self) -> None:
        """Verifies numerical stability clamp: when obs_freq parameter becomes <= -50.0 in float32,
        the lower bound clamp (min=1e-5) prevents 1 / nu_eff^2 gradient explosion / NaNs.
        In float64, precision extends further, demonstrating the numerical boundary limits.
        """
        # Under normal operations and positive frequencies, everything is stable
        attn_stable = QuantumZenoAttention(dim=4, observation_frequency=1.0, dtype=torch.float32)
        x = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        out = attn_stable(x)
        assert isinstance(out, torch.Tensor)
        out.sum().backward()
        assert attn_stable.obs_freq.grad is not None
        assert torch.isfinite(attn_stable.obs_freq.grad).all()

        # At extreme negative parameter value obs_freq = -50.0 in float32:
        attn_extreme = QuantumZenoAttention(dim=4, observation_frequency=-50.0, dtype=torch.float32)
        x_ext = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        out_ext = attn_extreme(x_ext)
        assert isinstance(out_ext, torch.Tensor)
        out_ext.sum().backward()
        # Gradient is clamped and remains strictly finite without NaN
        assert attn_extreme.obs_freq.grad is not None
        assert x_ext.grad is not None
        assert torch.isfinite(attn_extreme.obs_freq.grad).all(), (
            "Expected float32 gradient to be finite with clamp"
        )
        assert torch.isfinite(x_ext.grad).all(), (
            "Expected float32 input gradient to be finite with clamp"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Device Compatibility: CPU & Apple Silicon MPS
# ═══════════════════════════════════════════════════════════════════════════


class TestDeviceCompatibility:
    """Tests CPU and Apple Silicon Metal Performance Shaders (MPS) execution and autograd."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_cpu_full_suite(self, dtype: torch.dtype) -> None:
        """Verifies all modules execute cleanly on CPU across float32 and float64 precisions."""
        # 1. Attention
        attn = QuantumZenoAttention(dim=4, device="cpu", dtype=dtype)
        x_attn = torch.randn(2, 4, device="cpu", dtype=dtype, requires_grad=True)
        out_attn = attn(x_attn)
        assert isinstance(out_attn, torch.Tensor)
        out_attn.sum().backward()
        assert x_attn.grad is not None and torch.isfinite(x_attn.grad).all()

        # 2. Brain
        brain = BiomorphicResonantBrain(
            in_features=2,
            num_left_qubits=1,
            num_right_qubits=1,
            device="cpu",
            dtype=dtype,
        )
        x_brain = torch.randn(2, 2, device="cpu", dtype=dtype, requires_grad=True)
        out_brain = brain(x_brain)
        out_brain["consensus"].sum().backward()
        assert x_brain.grad is not None and torch.isfinite(x_brain.grad).all()

        # 3. REM Sleep
        rem = QuantumREMSleep(brain, sleep_cycles=2, device="cpu", dtype=dtype)
        s1 = torch.randn(brain.dim, dtype=brain.complex_dtype)
        s2 = torch.randn(brain.dim, dtype=brain.complex_dtype)
        res = rem.sleep(memory_states=[s1, s2], cycles=2)
        assert res["final_overlap"] <= res["initial_overlap"] + 1e-6

    def test_mps_device_autograd(self) -> None:
        """Verifies Apple Silicon MPS execution, float32, and backward gradient computation."""
        if not torch.backends.mps.is_available():
            pytest.skip("Apple Silicon MPS not available")

        dev = torch.device("mps")

        # 1. QuantumZenoAttention on MPS
        attn = QuantumZenoAttention(dim=8, num_heads=2, device=dev, dtype=torch.float32)
        x = torch.randn(3, 4, 8, device=dev, dtype=torch.float32, requires_grad=True)
        dop = torch.full((3, 4, 1), 2.0, device=dev, dtype=torch.float32, requires_grad=True)
        out = attn(x, dopamine=dop)
        assert isinstance(out, torch.Tensor)
        assert out.device.type == "mps"
        out.sum().backward()
        assert x.grad is not None and x.grad.device.type == "mps"
        assert torch.isfinite(x.grad).all()
        assert dop.grad is not None and dop.grad.device.type == "mps"
        assert torch.isfinite(dop.grad).all()

        # 2. BiomorphicResonantBrain on MPS
        brain = BiomorphicResonantBrain(
            in_features=4,
            num_left_qubits=2,
            num_right_qubits=2,
            device=dev,
            dtype=torch.float32,
        )
        x_b = torch.randn(2, 4, device=dev, dtype=torch.float32, requires_grad=True)
        out_b = brain(x_b)
        assert out_b["consensus"].device.type == "mps"
        out_b["consensus"].sum().backward()
        assert x_b.grad is not None and x_b.grad.device.type == "mps"
        assert torch.isfinite(x_b.grad).all()

        # 3. QuantumREMSleep on MPS
        rem = QuantumREMSleep(brain, sleep_cycles=3, device=dev, dtype=torch.float32)
        s1 = torch.randn(16, dtype=torch.complex64, device=dev)
        s2 = torch.randn(16, dtype=torch.complex64, device=dev)
        res_sleep = rem.sleep(memory_states=[s1, s2], cycles=3)
        assert res_sleep["cycles"] == 3
        assert not any(math.isnan(loss_val) for loss_val in res_sleep["loss_history"])
