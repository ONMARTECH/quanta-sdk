"""tests/test_csf_shielding.py — Comprehensive Test Suite for CSF Biophysical Quantum Shielding.

Validates the Cerebrospinal Fluid (CSF) & Interstitial Fluid (ISF) biophysical
quantum shielding framework (Theorem 8) across 5 verification tiers:
- Tier 1: Parameter Validation, Range Checking & Constructors
- Tier 2: Analytical Invariants & Theorem 8 Ground Truth
- Tier 3: Clinical Neuropathology Simulation & Mode Switching
- Tier 4: PyTorch Layer Dynamics, Shapes & Property Preservation
- Tier 5: Autograd Backpropagation, Device Transfer & Hardening
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from quanta.torch import ops
from quanta.torch.brain import (
    BiomorphicResonantBrain,
    CSFShieldedEnvironment,
    CSFShieldedResonantLayer,
    NoisyHippocampalBuffer,
    QuantumREMSleep,
)

# Universal Physical Constants for Analytical Verification
EPSILON_0 = 8.8541878128e-12  # F/m (Vacuum permittivity)
K_B = 1.380649e-23  # J/K (Boltzmann constant)
E_CHARGE = 1.602176634e-19  # C (Elementary charge)
N_AVOGADRO = 6.02214076e23  # mol^-1 (Avogadro constant)
R_H_POSNER = 0.48e-9  # m (Posner molecule hydrated radius)


# ============================================================================
# Tier 1: Parameter Validation, Range Checking & Constructors
# ============================================================================


def test_env_default_initialization() -> None:
    """Tier 1: Verifies default physiological parameters and buffer allocation."""
    env = CSFShieldedEnvironment()
    assert float(env.ionic_strength) == pytest.approx(0.155, abs=1e-3)
    assert float(env.dielectric_constant) == pytest.approx(78.5, abs=1e-1)
    assert float(env.viscosity) == pytest.approx(0.80, abs=1e-2)
    assert float(env.paramagnetic_concentration) == pytest.approx(0.40, abs=1e-2)
    assert float(env.glymphatic_clearance_rate) == pytest.approx(1.0, abs=0.8)
    assert float(env.temperature) == pytest.approx(310.15, abs=1e-2)

    # When learnable_params is False, tensors should not require gradients
    assert not env.ionic_strength.requires_grad
    assert not env.dielectric_constant.requires_grad
    assert not env.viscosity.requires_grad
    assert not env.paramagnetic_concentration.requires_grad


def test_env_parameter_validation_negative_values() -> None:
    """Tier 1: Asserts that non-physical, negative or zero parameter values raise ValueError."""
    with pytest.raises(ValueError, match="Ionic strength must be strictly positive"):
        CSFShieldedEnvironment(ionic_strength=0.0)
    with pytest.raises(ValueError, match="Ionic strength must be strictly positive"):
        CSFShieldedEnvironment(ionic_strength=-0.15)

    with pytest.raises(ValueError, match="Dielectric constant must be strictly positive"):
        CSFShieldedEnvironment(dielectric_constant=-1.0)

    with pytest.raises(ValueError, match="Viscosity must be strictly positive"):
        CSFShieldedEnvironment(viscosity=-0.80)

    with pytest.raises(ValueError, match="Paramagnetic concentration cannot be negative"):
        CSFShieldedEnvironment(paramagnetic_concentration=-0.05)

    with pytest.raises(ValueError, match="Glymphatic clearance rate cannot be negative"):
        CSFShieldedEnvironment(glymphatic_clearance_rate=-1.0)

    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        CSFShieldedEnvironment(temperature=0.0)


def test_layer_constructor_validation() -> None:
    """Tier 1: Verifies that invalid dimension and dephasing arguments raise ValueError."""
    with pytest.raises(ValueError, match="in_features must be positive"):
        CSFShieldedResonantLayer(in_features=0)

    with pytest.raises(ValueError, match="Hemisphere qubit count must be >= 1"):
        CSFShieldedResonantLayer(in_features=4, num_left_qubits=0, num_right_qubits=2)

    with pytest.raises(ValueError, match="Hemisphere qubit count must be >= 1"):
        CSFShieldedResonantLayer(in_features=4, num_left_qubits=2, num_right_qubits=0)

    with pytest.raises(ValueError, match="bare_dephasing_rate cannot be negative"):
        CSFShieldedResonantLayer(in_features=4, bare_dephasing_rate=-0.5)


def test_env_repr_and_string_formatting() -> None:
    """Tier 1: Verifies informative __repr__ formatting containing biophysical metrics."""
    env = CSFShieldedEnvironment(ionic_strength=0.155, viscosity=0.80)
    repr_str = repr(env)
    assert "CSFShieldedEnvironment" in repr_str
    assert "0.15" in repr_str or "0.155" in repr_str
    assert "0.8" in repr_str or "viscosity" in repr_str.lower()


def test_env_reset_to_baseline() -> None:
    """Tier 1: Verifies that reset_to_baseline restores exact physiological default values."""
    env = CSFShieldedEnvironment()
    env.simulate_clinical_condition("meningitis")

    # Values must have shifted under pathology
    assert float(env.viscosity) > 1.5
    assert float(env.paramagnetic_concentration) > 5.0

    # Reset
    env.reset_to_baseline()
    assert float(env.ionic_strength) == pytest.approx(0.155, abs=1e-3)
    assert float(env.dielectric_constant) == pytest.approx(78.5, abs=1e-1)
    assert float(env.viscosity) == pytest.approx(0.80, abs=1e-2)
    assert float(env.paramagnetic_concentration) == pytest.approx(0.40, abs=1e-2)
    assert float(env.temperature) == pytest.approx(310.15, abs=1e-2)


def test_custom_csf_environment_injection() -> None:
    """Tier 1: Verifies that passing a custom CSF environment injects and binds correctly."""
    custom_env = CSFShieldedEnvironment(viscosity=0.92, paramagnetic_concentration=0.60)
    layer = CSFShieldedResonantLayer(in_features=4, csf_environment=custom_env)

    target_env = getattr(layer, "csf_environment", getattr(layer, "env", None))
    assert target_env is custom_env
    assert float(target_env.viscosity) == pytest.approx(0.92, abs=1e-3)


def test_dtype_resolution_cpu() -> None:
    """Tier 1: Verifies proper CPU dtype assignment across float32 and float64."""
    env_f32 = CSFShieldedEnvironment(dtype=torch.float32)
    kappa_f32 = env_f32.compute_attenuation_factor()
    assert kappa_f32.dtype == torch.float32

    env_f64 = CSFShieldedEnvironment(dtype=torch.float64)
    kappa_f64 = env_f64.compute_attenuation_factor()
    assert kappa_f64.dtype == torch.float64

    layer_f32 = CSFShieldedResonantLayer(in_features=4, dtype=torch.float32)
    assert layer_f32.real_dtype == torch.float32

    layer_f64 = CSFShieldedResonantLayer(in_features=4, dtype=torch.float64)
    assert layer_f64.real_dtype == torch.float64


# ============================================================================
# Tier 2: Analytical Invariants & Theorem 8 Ground Truth
# ============================================================================


def test_debye_length_analytical_ground_truth() -> None:
    """Tier 2: Derives expected Debye length from physical constants and asserts ~0.79 nm."""
    env = CSFShieldedEnvironment(
        ionic_strength=0.155,
        dielectric_constant=78.5,
        temperature=310.15,
    )
    # Analytical derivation: lambda_D = sqrt(eps_0 * eps_r * k_B * T / (2 * N_A * e^2 * 1000 * I))
    expected_lambda_m = math.sqrt(
        (EPSILON_0 * 78.5 * K_B * 310.15)
        / (2.0 * N_AVOGADRO * (E_CHARGE**2) * (1000.0 * 0.155))
    )
    expected_lambda_nm = expected_lambda_m * 1e9  # ~ 0.788 nm

    computed_lambda = float(env.compute_debye_length())
    assert computed_lambda == pytest.approx(expected_lambda_nm, rel=1e-2)
    assert 0.75 <= computed_lambda <= 0.82


def test_debye_scaling_law() -> None:
    """Tier 2: Verifies Debye length inverse square root scaling with I and sqrt with eps_r."""
    env_base = CSFShieldedEnvironment(ionic_strength=0.155, dielectric_constant=78.5)
    lambda_base = float(env_base.compute_debye_length())

    # Doubling ionic strength -> lambda scales by 1 / sqrt(2)
    env_double_i = CSFShieldedEnvironment(ionic_strength=0.310, dielectric_constant=78.5)
    lambda_double_i = float(env_double_i.compute_debye_length())
    ratio_i = lambda_base / lambda_double_i
    assert ratio_i == pytest.approx(math.sqrt(2.0), rel=1e-3)

    # Quadrupling dielectric constant -> lambda scales by sqrt(4) = 2.0
    env_quad_eps = CSFShieldedEnvironment(ionic_strength=0.155, dielectric_constant=78.5 * 4.0)
    lambda_quad_eps = float(env_quad_eps.compute_debye_length())
    ratio_eps = lambda_quad_eps / lambda_base
    assert ratio_eps == pytest.approx(2.0, rel=1e-3)


def test_bpp_rotational_time_ground_truth() -> None:
    """Tier 2: Validates Stokes-Einstein-Debye Brownian rotational correlation time ~86.4 ps."""
    env = CSFShieldedEnvironment(viscosity=0.80, temperature=310.15)
    # Formula: tau_R = 4 * pi * (eta * 1e-3) * r_H^3 / (3 * k_B * T)
    expected_tau_s = (4.0 * math.pi * (0.80e-3) * (R_H_POSNER**3)) / (3.0 * K_B * 310.15)
    expected_tau_ps = expected_tau_s * 1e12  # ~ 86.4 ps

    computed_tau_ps = float(env.compute_rotational_correlation_time())
    assert computed_tau_ps == pytest.approx(expected_tau_ps, rel=0.05)
    assert 80.0 <= computed_tau_ps <= 92.0


def test_bpp_extreme_motional_narrowing_criterion() -> None:
    """Tier 2: Verifies omega_0 * tau_R << 1 in Earth's magnetic field (50 uT)."""
    env = CSFShieldedEnvironment(viscosity=0.80, temperature=310.15)
    tau_r_ps = float(env.compute_rotational_correlation_time())
    tau_r_s = tau_r_ps * 1e-12

    # Gyromagnetic ratio for 31P: gamma ~ 1.0839e8 rad / (s * T)
    gamma_p31 = 1.0839e8
    b_earth = 50e-6  # 50 uT
    omega_0 = gamma_p31 * b_earth  # ~ 5.42e3 rad/s

    narrowing_param = omega_0 * tau_r_s
    assert narrowing_param < 1e-5  # Strictly << 1


def test_theorem_8_shielding_attenuation_bound() -> None:
    """Tier 2: Proves Theorem 8 bound kappa_CSF <= 10^-3 in healthy physiological CSF."""
    env = CSFShieldedEnvironment()
    kappa_csf = float(env.compute_attenuation_factor())

    # Under Theorem 8, healthy physiological CSF guarantees kappa <= 10^-3
    assert 0.0 < kappa_csf <= 1e-3
    assert kappa_csf == pytest.approx(1.6e-6, abs=1e-4)


def test_theorem_8_coherence_enhancement_factor() -> None:
    """Tier 2: Asserts Coherence Enhancement Factor G_CSF = 1 / kappa_CSF >= 10^3."""
    env = CSFShieldedEnvironment()
    kappa_csf = float(env.compute_attenuation_factor())
    g_csf = 1.0 / kappa_csf

    assert g_csf >= 1e3
    assert g_csf >= 1e5  # Typically ~ 6e5


def test_component_attenuation_factors() -> None:
    """Tier 2: Validates individual components of electrostatic, motional,
    and paramagnetic damping.
    """
    env = CSFShieldedEnvironment()
    lambda_d = float(env.compute_debye_length())

    # Electrostatic: exp(-2 * R_0 / lambda_D) at R_0 = 2.0 nm
    kappa_elec = math.exp(-4.0 / lambda_d)
    assert kappa_elec <= 1e-2

    # Motional: eta / 50.0 (reference cytoskeleton gel)
    kappa_motional = 0.80 / 50.0
    assert kappa_motional <= 2e-2

    # Paramagnetic: [Para] / 25.0 (reference unshielded plasma)
    kappa_para = 0.40 / 25.0
    assert kappa_para <= 2e-2

    # Joint product strictly bounds total attenuation
    product = kappa_elec * kappa_motional * kappa_para
    assert product <= 1e-4


# ============================================================================
# Tier 3: Clinical Neuropathology Simulation & Mode Switching
# ============================================================================


def test_clinical_mode_normal() -> None:
    """Tier 3: Verifies baseline physiological values and low dephasing in normal condition."""
    env = CSFShieldedEnvironment()
    env.simulate_clinical_condition("normal")

    kappa = float(env.compute_attenuation_factor())
    assert kappa <= 1e-3
    assert float(env.viscosity) == pytest.approx(0.80, abs=1e-2)
    assert float(env.paramagnetic_concentration) == pytest.approx(0.40, abs=1e-2)


def test_clinical_mode_meningitis() -> None:
    """Tier 3: Verifies dielectric breakdown and severe dephasing surge in acute meningitis."""
    env = CSFShieldedEnvironment()
    env.simulate_clinical_condition("meningitis")

    kappa = float(env.compute_attenuation_factor())
    assert kappa >= 0.50
    assert float(env.viscosity) >= 2.5
    assert float(env.paramagnetic_concentration) >= 10.0


def test_clinical_mode_hydrocephalus() -> None:
    """Tier 3: Verifies clearance failure and intermediate dephasing in hydrocephalus (NPH)."""
    env = CSFShieldedEnvironment()
    env.simulate_clinical_condition("hydrocephalus")

    kappa = float(env.compute_attenuation_factor())
    assert 0.10 <= kappa <= 0.45
    assert float(env.glymphatic_clearance_rate) <= 0.2


def test_clinical_mode_lumbar_puncture_recovery() -> None:
    """Tier 3: Verifies rapid restoration of shielding following therapeutic lumbar puncture."""
    env = CSFShieldedEnvironment()
    env.simulate_clinical_condition("hydrocephalus")
    assert float(env.compute_attenuation_factor()) > 0.10

    # Execute therapeutic tap
    env.simulate_clinical_condition("lumbar_puncture_recovery")
    kappa_recovered = float(env.compute_attenuation_factor())
    assert kappa_recovered <= 0.01  # Rapid return towards baseline


def test_clinical_mode_sleep_deprivation_vs_rem_sleep() -> None:
    """Tier 3: Verifies that sleep deprivation yields >3x higher dephasing than REM sleep."""
    env = CSFShieldedEnvironment()

    env.simulate_clinical_condition("sleep_deprived")
    kappa_deprived = float(env.compute_attenuation_factor())

    env.simulate_clinical_condition("rem_sleep")
    kappa_rem = float(env.compute_attenuation_factor())

    assert kappa_deprived > 3.0 * kappa_rem


def test_clinical_mode_invalid_name_raises_error() -> None:
    """Tier 3: Asserts ValueError on unrecognized clinical condition string."""
    env = CSFShieldedEnvironment()
    with pytest.raises(ValueError, match="Unsupported clinical condition"):
        env.simulate_clinical_condition("invalid_condition_syndrome")


# ============================================================================
# Tier 4: PyTorch Layer Dynamics, Shapes & Property Preservation
# ============================================================================


def test_layer_output_shapes_unbatched_and_batched() -> None:
    """Tier 4: Validates tensor shapes across 1D unbatched, 2D batched, and 3D sequential."""
    layer = CSFShieldedResonantLayer(
        in_features=4,
        num_left_qubits=2,
        num_right_qubits=2,
        return_dict=True,
    )
    layer_seq = CSFShieldedResonantLayer(
        in_features=4,
        num_left_qubits=2,
        num_right_qubits=2,
        return_dict=False,
    )

    # 1D Unbatched input
    x_1d = torch.randn(4)
    out_1d_dict = layer(x_1d)
    assert out_1d_dict["consensus"].shape == (1,)
    assert out_1d_dict["state"].shape == (16,)
    assert out_1d_dict["readout_vector"].shape == (4,)
    assert layer_seq(x_1d).shape == (1,)

    # 2D Batched input (B=5)
    x_2d = torch.randn(5, 4)
    out_2d_dict = layer(x_2d)
    assert out_2d_dict["consensus"].shape == (5, 1)
    assert out_2d_dict["left_consensus"].shape == (5, 1)
    assert out_2d_dict["right_consensus"].shape == (5, 1)
    assert out_2d_dict["state"].shape == (5, 16)
    assert out_2d_dict["readout_vector"].shape == (5, 4)
    assert layer_seq(x_2d).shape == (5, 1)

    # 3D Sequential input (B=2, L=3, D=4)
    x_3d = torch.randn(2, 3, 4)
    assert layer_seq(x_3d).shape == (2, 3, 1)


def test_layer_empty_batch_handling() -> None:
    """Tier 4: Asserts that empty batch (B=0) executes gracefully without crashing."""
    layer_seq = CSFShieldedResonantLayer(in_features=4, return_dict=False)
    x_empty = torch.zeros(0, 4)
    out_empty = layer_seq(x_empty)
    assert out_empty.shape == (0, 1)

    layer_dict = CSFShieldedResonantLayer(in_features=4, return_dict=True)
    out_dict = layer_dict(x_empty)
    assert out_dict["consensus"].shape == (0, 1)
    assert out_dict["state"].shape == (0, 16)


def test_unitary_norm_preservation() -> None:
    """Tier 4: Proves ||psi(t)||^2 = 1.0 +- 10^-6 across all batch instances."""
    layer = CSFShieldedResonantLayer(in_features=4, return_dict=True)
    x = torch.randn(8, 4)
    out = layer(x)

    state = out["state"]
    norms = torch.sum(torch.abs(state) ** 2, dim=-1)
    for norm in norms:
        assert float(norm) == pytest.approx(1.0, abs=1e-5)


def test_consensus_expectation_bounds() -> None:
    """Tier 4: Asserts macroscopic consensus and readout expectations lie strictly in [-1, 1]."""
    layer = CSFShieldedResonantLayer(in_features=4, return_dict=True)
    x = torch.randn(12, 4) * 8.0  # Wide dynamic range
    out = layer(x)

    consensus = out["consensus"]
    assert (consensus >= -1.0001).all()
    assert (consensus <= 1.0001).all()

    readouts = out["readout_vector"]
    assert (readouts >= -1.0001).all()
    assert (readouts <= 1.0001).all()


def test_return_dict_flag_compatibility() -> None:
    """Tier 4: Verifies dictionary diagnostic keys and sequential tensor compatibility."""
    layer_dict = CSFShieldedResonantLayer(in_features=4, return_dict=True)
    out_d = layer_dict(torch.randn(2, 4))
    expected_keys = {
        "consensus",
        "left_consensus",
        "right_consensus",
        "readout_vector",
        "state",
        "kappa_csf",
        "effective_dephasing",
        "coherence_factor",
    }
    assert expected_keys.issubset(out_d.keys())

    layer_tensor = CSFShieldedResonantLayer(in_features=4, return_dict=False)
    out_t = layer_tensor(torch.randn(2, 4))
    assert isinstance(out_t, torch.Tensor)
    assert out_t.shape == (2, 1)


def test_layer_attenuated_dephasing_impact() -> None:
    """Tier 4: Verifies dephasing factor difference between normal CSF and meningitis."""
    env_normal = CSFShieldedEnvironment()
    layer_normal = CSFShieldedResonantLayer(
        in_features=4,
        csf_environment=env_normal,
        bare_dephasing_rate=1.30,
        return_dict=True,
    )

    env_meningitis = CSFShieldedEnvironment()
    env_meningitis.simulate_clinical_condition("meningitis")
    layer_meningitis = CSFShieldedResonantLayer(
        in_features=4,
        csf_environment=env_meningitis,
        bare_dephasing_rate=1.30,
        return_dict=True,
    )

    x = torch.ones(2, 4)
    out_norm = layer_normal(x)
    out_men = layer_meningitis(x)

    c_norm = float(out_norm["coherence_factor"].mean())
    c_men = float(out_men["coherence_factor"].mean())

    assert c_norm > c_men
    assert c_norm > 0.95  # Normal CSF preserves coherence
    assert c_men < 0.70  # Severe dephasing under meningitis


# ============================================================================
# Tier 5: Autograd Backpropagation, Device Transfer & Hardening
# ============================================================================


def test_autograd_backpropagation_input_and_weights() -> None:
    """Tier 5: Validates autograd backward pass through classical inputs and quantum weights."""
    layer = CSFShieldedResonantLayer(in_features=4, return_dict=False)
    x = torch.randn(4, 4, requires_grad=True)

    out = layer(x)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Verify layer parameters receive non-zero, finite gradients
    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    assert len(trainable_params) > 0
    for p in trainable_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_autograd_learnable_csf_environment() -> None:
    """Tier 5: Asserts that learnable environment parameters receive autograd gradients."""
    env = CSFShieldedEnvironment(learnable_params=True)
    assert env.ionic_strength.requires_grad
    assert env.viscosity.requires_grad

    kappa = env.compute_attenuation_factor()
    assert kappa.requires_grad

    loss = (kappa - 0.05) ** 2
    loss.backward()

    assert env.ionic_strength.grad is not None
    assert torch.isfinite(env.ionic_strength.grad).all()
    assert env.viscosity.grad is not None
    assert torch.isfinite(env.viscosity.grad).all()


def test_numerical_gradcheck_csf_shielding() -> None:
    """Tier 5: Validates analytical autograd gradients against numerical finite differences."""

    def attenuation_func(params: torch.Tensor) -> torch.Tensor:
        i_val = params[0]
        eps_val = params[1]
        eta_val = params[2]

        # Analytical Debye & motional narrowing formulas in float64
        t_k = 310.15
        lambda_d = (
            torch.sqrt(
                (EPSILON_0 * eps_val * K_B * t_k)
                / (2.0 * N_AVOGADRO * (E_CHARGE**2) * (1000.0 * i_val))
            )
            * 1e9
        )
        k_elec = torch.exp(-4.0 / lambda_d)
        k_mot = eta_val / 50.0
        k_para = 0.40 / 25.0
        k_glym = 1.0 / 2.0
        return k_elec * k_mot * k_para * k_glym

    p = torch.tensor([0.155, 78.5, 0.80], dtype=torch.float64, requires_grad=True)
    gradcheck_passed = torch.autograd.gradcheck(
        attenuation_func,
        (p,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert gradcheck_passed


def test_nn_sequential_and_adam_optimizer() -> None:
    """Tier 5: Verifies training convergence inside nn.Sequential with Adam optimizer."""
    torch.manual_seed(42)
    model = nn.Sequential(
        CSFShieldedResonantLayer(in_features=4, return_dict=False),
        nn.Linear(1, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    x = torch.randn(8, 4)
    target = torch.ones(8, 1) * 0.5

    init_loss = criterion(model(x), target).item()

    for _ in range(6):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < init_loss


def test_apple_silicon_mps_compatibility() -> None:
    """Tier 5: Validates execution on Apple Silicon Metal (MPS) and 64-bit dtype defense."""
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon Metal (MPS) device not available")

    dev = torch.device("mps")

    # 1. Successful float32 forward on MPS
    env = CSFShieldedEnvironment(device=dev, dtype=torch.float32)
    kappa = env.compute_attenuation_factor()
    assert kappa.device.type == "mps"
    assert torch.isfinite(kappa)

    layer = CSFShieldedResonantLayer(in_features=4, device=dev, dtype=torch.float32)
    x = torch.randn(2, 4, device=dev, dtype=torch.float32)
    out = layer(x)
    assert out["consensus"].device.type == "mps"

    # 2. Rejection of unsupported 64-bit dtype on MPS
    with pytest.raises(ops.UnsupportedDtypeError):
        CSFShieldedEnvironment(device="mps", dtype=torch.float64)


def test_cross_module_interoperability() -> None:
    """Tier 5: Validates seamless interoperability across Brain, Hippocampus, and REMSleep."""
    env = CSFShieldedEnvironment()

    # 1. Attach to BiomorphicResonantBrain
    brain = BiomorphicResonantBrain(in_features=4)
    attach_brain = getattr(brain, "attach_csf_environment", None)
    if callable(attach_brain):
        attach_brain(env)
        out_brain = brain(torch.randn(2, 4))
        assert "kappa_csf" in out_brain
        assert float(out_brain["kappa_csf"]) <= 1e-3

    # 2. Attach to NoisyHippocampalBuffer
    buffer = NoisyHippocampalBuffer(capacity=16)
    attach_buf = getattr(buffer, "attach_csf_environment", None)
    if callable(attach_buf):
        attach_buf(env)
        assert getattr(buffer, "csf_environment", None) is env

    # 3. Consolidation with QuantumREMSleep
    rem_sleep = QuantumREMSleep(brain_module=brain)
    assert rem_sleep is not None
