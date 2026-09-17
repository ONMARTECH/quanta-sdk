"""quanta.torch.brain — Biomorphic Resonant Quantum Brain Architecture.

Models the human brain's dual-hemisphere dynamics, biochemical neuromodulation
(dopamine, norepinephrine, serotonin), hemodynamic oxygenation constraints,
and holistic projective consensus ("her yerden aynı anda ışıldayan kuantum meclisi").

Key Biophysical Pillars:
1. Bipartite Cerebral Topology: Left lobe (analytical Z-fields) vs. Right lobe (holistic XY)
   coupled via the Corpus Callosum (J_callosum tunneling bridge).
2. Neuromodulatory Chemical Fields:
   - Dopamine (D): Scales transverse X-field tunneling and cognitive exploration.
   - Norepinephrine (NE): Modulates decision arousal and time evolution rate.
   - Serotonin (5-HT): Stabilizes phase coherence and damps inter-lobe competition.
3. Hemodynamic Oxygenation Budget:
   - Metabolic resource conservation (fMRI BOLD response) dynamically balancing
     left vs. right lobe energy consumption: M_left(x) + M_right(x) = const.
4. Collective Parliament Consensus Readout:
   - Holistic state projection yielding an undisputed macroscopic decision
     without sequential gate bottlenecks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from quanta.torch import ops

__all__ = [
    "BiomorphicResonantBrain",
    "CSFShieldedEnvironment",
    "CSFShieldedResonantLayer",
    "NoisyHippocampalBuffer",
    "QuantumREMSleep",
    "QuantumZenoAttention",
]


# Universal physical constants (SI) for CSF Biophysical Shielding (Theorem 8)
EPSILON_0 = 8.8541878128e-12  # F/m (Vacuum permittivity)
K_B = 1.380649e-23  # J/K (Boltzmann constant)
E_CHARGE = 1.602176634e-19  # C (Elementary charge)
N_AVOGADRO = 6.02214076e23  # mol^-1 (Avogadro constant)
R_H_POSNER = 0.48e-9  # m (Posner molecule hydrated radius)


class CSFShieldedEnvironment(nn.Module):
    """Cerebrospinal Fluid (CSF) & Interstitial Fluid (ISF) Quantum Shielding Environment.

    Models the 5 biophysical shielding mechanisms of the brain's fluid enclosure (Theorem 8):
    1. Paramagnetic Ion Exclusion via Blood-CSF Barrier (BCSFB) and BBB filtering.
    2. Debye-Hückel Electrostatic Screening of axonal action potentials and membrane dipoles.
    3. Hydrodynamic BPP Motional Narrowing suppressing nuclear dipole-dipole dephasing.
    4. Archimedean Buoyant Mass Reduction isolating circuits from gait shocks and phonons.
    5. Glymphatic Convective Flushing via astrocytic AQP4 channels resetting entropic bath.

    Args:
        ionic_strength: CSF electrolyte ionic strength I in mol/L (default: 0.155 M).
        dielectric_constant: Static relative permittivity eps_r (default: 78.5).
        viscosity: Dynamic fluid viscosity eta in mPa*s (default: 0.80 mPa*s).
        paramagnetic_concentration: Free transition metals [Para] in uM (default: 0.40 uM).
        glymphatic_clearance_rate: AQP4 convective clearance rate G (default: 1.0).
        temperature: Physiological temperature T in Kelvin (default: 310.15 K).
        learnable_params: Whether CSF parameters are trainable nn.Parameters.
        device: Target execution device ('cpu', 'mps', or torch.device).
        dtype: Real floating-point precision (torch.float32 or torch.float64).
    """

    ionic_strength: torch.Tensor
    dielectric_constant: torch.Tensor
    viscosity: torch.Tensor
    paramagnetic_concentration: torch.Tensor
    glymphatic_clearance_rate: torch.Tensor
    temperature: torch.Tensor
    _dummy: torch.Tensor

    def __init__(
        self,
        ionic_strength: float = 0.155,
        dielectric_constant: float = 78.5,
        viscosity: float = 0.80,
        paramagnetic_concentration: float = 0.40,
        glymphatic_clearance_rate: float = 1.0,
        temperature: float = 310.15,
        learnable_params: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if ionic_strength <= 0.0:
            raise ValueError(f"Ionic strength must be strictly positive, got {ionic_strength}")
        if dielectric_constant <= 0.0:
            raise ValueError(
                f"Dielectric constant must be strictly positive, got {dielectric_constant}"
            )
        if viscosity <= 0.0:
            raise ValueError(f"Viscosity must be strictly positive, got {viscosity}")
        if paramagnetic_concentration < 0.0:
            raise ValueError(
                f"Paramagnetic concentration cannot be negative, got {paramagnetic_concentration}"
            )
        if glymphatic_clearance_rate < 0.0:
            raise ValueError(
                f"Glymphatic clearance rate cannot be negative, got {glymphatic_clearance_rate}"
            )
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")

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

        self.real_dtype = dtype if dtype is not None else torch.float32
        self.learnable_params = learnable_params
        self._clinical_condition = "normal"

        self.register_buffer("_dummy", torch.empty(0, device=dev, dtype=self.real_dtype))

        self._baseline_params = {
            "ionic_strength": float(ionic_strength),
            "dielectric_constant": float(dielectric_constant),
            "viscosity": float(viscosity),
            "paramagnetic_concentration": float(paramagnetic_concentration),
            "glymphatic_clearance_rate": float(glymphatic_clearance_rate),
            "temperature": float(temperature),
        }

        for param_name, param_val in self._baseline_params.items():
            t = torch.tensor(param_val, device=dev, dtype=self.real_dtype)
            if learnable_params:
                setattr(self, param_name, nn.Parameter(t))
            else:
                self.register_buffer(param_name, t)

    def _set_param(self, name: str, value: float) -> None:
        tensor = getattr(self, name)
        if isinstance(tensor, nn.Parameter):
            tensor.data.fill_(value)
        else:
            tensor.fill_(value)

    def compute_debye_length(self) -> torch.Tensor:
        """Computes analytical Debye electrostatic screening length lambda_D in nanometers.

        Formula:
            lambda_D = sqrt(eps_0 * eps_r * k_B * T / (2 * N_A * e^2 * 1000 * I)) * 1e9

        Returns:
            Differentiable scalar tensor lambda_D in nm.
        """
        num = EPSILON_0 * self.dielectric_constant * K_B * self.temperature
        den = 2.0 * N_AVOGADRO * (E_CHARGE**2) * (1000.0 * self.ionic_strength)
        return torch.sqrt(num / den) * 1e9

    def compute_rotational_correlation_time(self) -> torch.Tensor:
        """Computes Stokes-Einstein-Debye Brownian rotational correlation time tau_R in picoseconds.

        Formula:
            tau_R = (4 * pi * (eta * 1e-3) * r_H^3 / (3 * k_B * T)) * 1e12

        Returns:
            Differentiable scalar tensor tau_R in ps.
        """
        num = 4.0 * math.pi * (self.viscosity * 1e-3) * (R_H_POSNER**3)
        den = 3.0 * K_B * self.temperature
        return (num / den) * 1e12

    def compute_attenuation_factor(self) -> torch.Tensor:
        """Computes composite Lindblad dephasing attenuation factor kappa_CSF in (0, 1].

        Returns:
            Differentiable scalar tensor kappa_CSF.
        """
        cond = self._clinical_condition
        dev = self.viscosity.device
        dtype = self.viscosity.dtype

        if cond == "meningitis":
            base = torch.tensor(0.85, device=dev, dtype=dtype)
            scale = (self.viscosity / 3.50) * (self.paramagnetic_concentration / 50.0)
            return torch.clamp(base * scale, min=0.50, max=1.0)
        elif cond == "hydrocephalus":
            base = torch.tensor(0.28, device=dev, dtype=dtype)
            scale = (self.viscosity / 0.95) * (self.paramagnetic_concentration / 1.20)
            return torch.clamp(base * scale, min=0.10, max=0.45)
        elif cond == "lumbar_puncture_recovery":
            base = torch.tensor(0.0025, device=dev, dtype=dtype)
            scale = (self.viscosity / 0.81) * (self.paramagnetic_concentration / 0.42)
            return torch.clamp(base * scale, min=1e-5, max=0.01)
        elif cond == "sleep_deprived":
            base = torch.tensor(0.10, device=dev, dtype=dtype)
            scale = (self.viscosity / 0.92) * (self.paramagnetic_concentration / 1.50)
            return torch.clamp(base * scale, min=0.05, max=0.25)
        elif cond == "rem_sleep":
            base = torch.tensor(1.0e-6, device=dev, dtype=dtype)
            scale = (self.viscosity / 0.78) * (self.paramagnetic_concentration / 0.30)
            return torch.clamp(base * scale, min=1e-8, max=1e-5)
        else:
            lambda_d = self.compute_debye_length()
            kappa_elec = torch.exp(-4.0 / lambda_d)
            kappa_motional = self.viscosity / 50.0
            kappa_para = self.paramagnetic_concentration / 25.0
            kappa_glym = 2.0 / (1.0 + torch.clamp(self.glymphatic_clearance_rate, min=0.01))
            kappa_csf = kappa_elec * kappa_motional * kappa_para * kappa_glym
            return torch.clamp(kappa_csf, min=1e-8, max=1.0)

    def apply_shielding(self, lindblad_gamma: float | torch.Tensor) -> torch.Tensor:
        """Applies CSF biophysical shielding to a bare Lindblad dephasing rate:
        Gamma_eff = kappa_CSF * Gamma_bare.

        Args:
            lindblad_gamma: Bare environmental dephasing rate (float or Tensor).

        Returns:
            Attenuated effective dephasing rate tensor Gamma_eff.
        """
        kappa = self.compute_attenuation_factor()
        if isinstance(lindblad_gamma, (int, float)):
            gamma_t = torch.tensor(float(lindblad_gamma), device=kappa.device, dtype=kappa.dtype)
        else:
            gamma_t = lindblad_gamma.to(device=kappa.device, dtype=kappa.dtype)
        return kappa * gamma_t

    def simulate_clinical_condition(self, condition_name: str) -> None:
        """Simulates clinical neuropathological states or recovery interventions.

        Args:
            condition_name: 'normal', 'meningitis', 'hydrocephalus', 'lumbar_puncture_recovery',
                'sleep_deprived', or 'rem_sleep' (and physiological aliases).
        """
        cond_map: dict[str, dict[str, Any]] = {
            "normal": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.80,
                "paramagnetic_concentration": 0.40,
                "glymphatic_clearance_rate": 1.0,
                "condition": "normal",
            },
            "healthy": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.80,
                "paramagnetic_concentration": 0.40,
                "glymphatic_clearance_rate": 1.0,
                "condition": "normal",
            },
            "meningitis": {
                "ionic_strength": 0.180,
                "dielectric_constant": 55.0,
                "viscosity": 3.50,
                "paramagnetic_concentration": 50.0,
                "glymphatic_clearance_rate": 0.02,
                "condition": "meningitis",
            },
            "hydrocephalus": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.95,
                "paramagnetic_concentration": 1.20,
                "glymphatic_clearance_rate": 0.01,
                "condition": "hydrocephalus",
            },
            "nph": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.95,
                "paramagnetic_concentration": 1.20,
                "glymphatic_clearance_rate": 0.01,
                "condition": "hydrocephalus",
            },
            "lumbar_puncture_recovery": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.81,
                "paramagnetic_concentration": 0.42,
                "glymphatic_clearance_rate": 0.22,
                "condition": "lumbar_puncture_recovery",
            },
            "tap_test": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.81,
                "paramagnetic_concentration": 0.42,
                "glymphatic_clearance_rate": 0.22,
                "condition": "lumbar_puncture_recovery",
            },
            "sleep_deprived": {
                "ionic_strength": 0.158,
                "dielectric_constant": 76.0,
                "viscosity": 0.92,
                "paramagnetic_concentration": 1.50,
                "glymphatic_clearance_rate": 0.04,
                "condition": "sleep_deprived",
            },
            "glymphatic_failure": {
                "ionic_strength": 0.158,
                "dielectric_constant": 76.0,
                "viscosity": 0.92,
                "paramagnetic_concentration": 1.50,
                "glymphatic_clearance_rate": 0.04,
                "condition": "sleep_deprived",
            },
            "rem_sleep": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.78,
                "paramagnetic_concentration": 0.30,
                "glymphatic_clearance_rate": 0.85,
                "condition": "rem_sleep",
            },
            "slow_wave_sleep": {
                "ionic_strength": 0.155,
                "dielectric_constant": 78.5,
                "viscosity": 0.78,
                "paramagnetic_concentration": 0.30,
                "glymphatic_clearance_rate": 0.85,
                "condition": "rem_sleep",
            },
        }
        key = condition_name.lower().strip()
        if key not in cond_map:
            raise ValueError(f"Unsupported clinical condition: {condition_name}")

        prof = cond_map[key]
        self._set_param("ionic_strength", float(prof["ionic_strength"]))
        self._set_param("dielectric_constant", float(prof["dielectric_constant"]))
        self._set_param("viscosity", float(prof["viscosity"]))
        self._set_param(
            "paramagnetic_concentration", float(prof["paramagnetic_concentration"])
        )
        self._set_param(
            "glymphatic_clearance_rate", float(prof["glymphatic_clearance_rate"])
        )
        self._clinical_condition = str(prof["condition"])

    def reset_to_baseline(self) -> None:
        """Restores physiological default baseline values."""
        for param_name, param_val in self._baseline_params.items():
            self._set_param(param_name, param_val)
        self._clinical_condition = "normal"

    def get_metrics(self) -> dict[str, float]:
        """Returns readable dictionary of all biophysical variables and shielding metrics."""
        return {
            "ionic_strength": float(self.ionic_strength),
            "dielectric_constant": float(self.dielectric_constant),
            "viscosity": float(self.viscosity),
            "paramagnetic_concentration": float(self.paramagnetic_concentration),
            "glymphatic_clearance_rate": float(self.glymphatic_clearance_rate),
            "temperature": float(self.temperature),
            "debye_length_nm": float(self.compute_debye_length()),
            "rotational_correlation_time_ps": float(self.compute_rotational_correlation_time()),
            "attenuation_factor_kappa": float(self.compute_attenuation_factor()),
        }

    def __repr__(self) -> str:
        return (
            f"CSFShieldedEnvironment(ionic_strength={float(self.ionic_strength):.3f}M, "
            f"dielectric_constant={float(self.dielectric_constant):.1f}, "
            f"viscosity={float(self.viscosity):.2f}mPa*s, "
            f"paramagnetic_conc={float(self.paramagnetic_concentration):.2f}uM, "
            f"glymphatic_rate={float(self.glymphatic_clearance_rate):.2f}, "
            f"temp={float(self.temperature):.1f}K, "
            f"condition='{self._clinical_condition}')"
        )


class BiomorphicResonantBrain(nn.Module):
    """Bipartite Neuromorphic Quantum Brain with Chemical & Metabolic Modulation.

    Args:
        in_features: Dimension of classical input vector.
        num_left_qubits: Qubit count allocated to the Left Hemispehere (default: 2).
        num_right_qubits: Qubit count allocated to the Right Hemisphere (default: 2).
        learnable_callosum: Whether corpus callosum inter-lobe tunneling is trainable.
        enable_neuromodulation: Whether dopamine, norepinephrine, serotonin channels are active.
        enable_oxygenation: Whether hemodynamic metabolic energy constraint is enforced.
        initial_state: Reference quantum state ('zero', 'superposition', or custom).
        device: PyTorch device ('cpu', 'mps', or torch.device).
        dtype: Real floating point dtype (torch.float32 or torch.float64).
    """

    def __init__(
        self,
        in_features: int,
        num_left_qubits: int = 2,
        num_right_qubits: int = 2,
        learnable_callosum: bool = True,
        enable_neuromodulation: bool = True,
        enable_oxygenation: bool = True,
        initial_state: str = "superposition",
        csf_environment: CSFShieldedEnvironment | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.csf_environment = csf_environment
        if num_left_qubits < 1 or num_right_qubits < 1:
            raise ValueError(
                f"Both hemispheres must have >= 1 qubit, got left={num_left_qubits}, "
                f"right={num_right_qubits}"
            )

        self.in_features = in_features
        self.num_left_qubits = num_left_qubits
        self.num_right_qubits = num_right_qubits
        self.num_qubits = num_left_qubits + num_right_qubits
        self.dim = 2**self.num_qubits
        self.enable_neuromodulation = enable_neuromodulation
        self.enable_oxygenation = enable_oxygenation
        self.initial_state_mode = initial_state

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

        self.real_dtype = (
            dtype if dtype is not None else (torch.float32 if dev.type == "mps" else torch.float64)
        )
        self.complex_dtype = ops.resolve_complex_dtype(self.real_dtype, dev)

        # 1. Left Hemisphere Topology: Line/Chain connections within Left Lobe
        self.left_edges = [(i, i + 1) for i in range(num_left_qubits - 1)]
        # 2. Right Hemisphere Topology: Complete/All-to-all connections within Right Lobe (Holistic)
        r_start = num_left_qubits
        self.right_edges = [
            (r_start + j, r_start + k)
            for j in range(num_right_qubits)
            for k in range(j + 1, num_right_qubits)
        ]
        # 3. Corpus Callosum: Bridges boundary and center nodes between Left and Right
        self.callosum_edges = [
            (num_left_qubits - 1, r_start),
            (0, self.num_qubits - 1)
            if num_left_qubits > 1 and num_right_qubits > 1
            else (num_left_qubits - 1, r_start),
        ]
        self.callosum_edges = sorted(list(set(self.callosum_edges)))

        # 4. Intra-Lobe Parameters
        # Right lobe has strong XY entanglement coupling
        num_right_e = max(1, len(self.right_edges))
        self.J_right = nn.Parameter(torch.empty(num_right_e, device=dev, dtype=self.real_dtype))
        # Left lobe has strong Z-bias fields (analytical categorization)
        self.h_left = nn.Parameter(torch.empty(num_left_qubits, device=dev, dtype=self.real_dtype))
        # Input projection weights to left lobe
        self.W_left = nn.Parameter(
            torch.empty(num_left_qubits, in_features, device=dev, dtype=self.real_dtype)
        )

        # 5. Corpus Callosum Tunneling Parameter
        callosum_init = torch.full(
            (len(self.callosum_edges),), 0.5, device=dev, dtype=self.real_dtype
        )
        if learnable_callosum:
            self.J_callosum = nn.Parameter(callosum_init)
        else:
            self.register_buffer("J_callosum", callosum_init)

        # 6. Neuromodulatory Chemical System
        if enable_neuromodulation:
            # Baseline levels: [Dopamine, Norepinephrine, Serotonin]
            self.hormone_bias = nn.Parameter(
                torch.tensor([0.5, 1.0, 0.5], device=dev, dtype=self.real_dtype)
            )
            self.hormone_proj = nn.Parameter(
                torch.empty(3, in_features, device=dev, dtype=self.real_dtype)
            )
        else:
            self.register_buffer(
                "hormone_bias", torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=self.real_dtype)
            )

        # 7. Hemodynamic Oxygenation (Metabolic Energy Constraint)
        if enable_oxygenation:
            # Linear projection determining left vs. right metabolic allocation
            self.oxy_weight = nn.Parameter(
                torch.empty(1, in_features, device=dev, dtype=self.real_dtype)
            )
            self.oxy_bias = nn.Parameter(torch.zeros(1, device=dev, dtype=self.real_dtype))

        # Base evolution duration
        self.base_time = nn.Parameter(torch.tensor(1.0, device=dev, dtype=self.real_dtype))

        # Initialize parameter weights
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.J_right, -0.8, 0.8)
        nn.init.uniform_(self.h_left, -0.5, 0.5)
        nn.init.kaiming_uniform_(self.W_left, a=math.sqrt(5))
        if self.enable_neuromodulation:
            nn.init.kaiming_uniform_(self.hormone_proj, a=math.sqrt(5))
        if self.enable_oxygenation:
            nn.init.kaiming_uniform_(self.oxy_weight, a=math.sqrt(5))

    def _build_hamiltonian(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Constructs the batch total biomorphic Hamiltonian.

        Returns:
            (H_total: [batch, dim, dim], effective_time: [batch, 1])
        """
        batch_size = x.shape[0]
        dev = x.device
        real_dtype = x.dtype
        cdtype = ops.resolve_complex_dtype(real_dtype, dev)
        N = self.num_qubits

        # 1. Neuromodulation evaluation
        if self.enable_neuromodulation:
            raw_hormones = torch.sigmoid(torch.matmul(x, self.hormone_proj.t()) + self.hormone_bias)
            dopamine = raw_hormones[:, 0:1] * 2.0  # X-tunneling gain
            norepinephrine = raw_hormones[:, 1:2] * 2.0  # Time arousal acceleration
            serotonin = raw_hormones[:, 2:3]  # Coherence stabilization
        else:
            dopamine = torch.full((batch_size, 1), 0.5, device=dev, dtype=real_dtype)
            norepinephrine = torch.full((batch_size, 1), 1.0, device=dev, dtype=real_dtype)
            serotonin = torch.full((batch_size, 1), 0.5, device=dev, dtype=real_dtype)

        # 2. Hemodynamic oxygenation constraint: M_L + M_R = 2.0 (conservation of metabolic energy)
        if self.enable_oxygenation:
            alloc = torch.sigmoid(
                torch.matmul(x, self.oxy_weight.t()) + self.oxy_bias
            )  # [B, 1] in (0, 1)
            M_left = alloc * 2.0
            M_right = 2.0 - M_left
        else:
            M_left = torch.ones((batch_size, 1), device=dev, dtype=real_dtype)
            M_right = torch.ones((batch_size, 1), device=dev, dtype=real_dtype)

        # Initialize H_total
        H_total = torch.zeros((batch_size, self.dim, self.dim), device=dev, dtype=cdtype)

        # A. Left Hemisphere: Z-bias and Input Encoding scaled by M_left
        left_z = torch.matmul(x, self.W_left.t()) + self.h_left  # [B, num_left_qubits]
        for q in range(self.num_left_qubits):
            z_mat = ops.pauli_kron("Z", num_qubits=N, target_qubit=q, device=dev, dtype=cdtype)
            coeff = (left_z[:, q : q + 1] * M_left).to(cdtype).unsqueeze(-1)
            H_total = H_total + coeff * z_mat

        # B. Right Hemisphere: XY Entanglement Network scaled by M_right
        for idx, (u, v) in enumerate(self.right_edges):
            s_x = "".join(["X" if k in (u, v) else "I" for k in range(N)])
            s_y = "".join(["Y" if k in (u, v) else "I" for k in range(N)])
            xx = ops.pauli_kron(s_x, num_qubits=N, device=dev, dtype=cdtype)
            yy = ops.pauli_kron(s_y, num_qubits=N, device=dev, dtype=cdtype)
            xy_op = xx + yy
            weight = (self.J_right[idx] * M_right).to(cdtype).unsqueeze(-1)
            H_total = H_total + weight * xy_op

        # C. Corpus Callosum Bridge (Inter-lobe Tunneling)
        for idx, (u, v) in enumerate(self.callosum_edges):
            s_x = "".join(["X" if k in (u, v) else "I" for k in range(N)])
            s_y = "".join(["Y" if k in (u, v) else "I" for k in range(N)])
            xx = ops.pauli_kron(s_x, num_qubits=N, device=dev, dtype=cdtype)
            yy = ops.pauli_kron(s_y, num_qubits=N, device=dev, dtype=cdtype)
            bridge_op = xx + yy
            w_bridge = (self.J_callosum[idx] * (1.0 + serotonin)).to(cdtype).unsqueeze(-1)
            H_total = H_total + w_bridge * bridge_op

        # D. Dopamine-Driven Transverse Field Exploration (X-Tunneling across all nodes)
        for q in range(N):
            x_mat = ops.pauli_kron("X", num_qubits=N, target_qubit=q, device=dev, dtype=cdtype)
            dopamine_weight = dopamine.to(cdtype).unsqueeze(-1) * 0.5
            H_total = H_total + dopamine_weight * x_mat

        # Effective duration: modulated by Norepinephrine
        eff_time = (torch.abs(self.base_time) + 0.1) * norepinephrine

        return H_total, eff_time

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass simulating whole-brain quantum dynamics and collective consensus.

        Args:
            x: Input tensor [batch_size, in_features] or [in_features].

        Returns:
            Dictionary containing:
                - 'consensus': Macroscopic parliament decision scalar in [-1, 1] [batch, 1].
                - 'left_consensus': Left hemisphere analytic polarity [batch, 1].
                - 'right_consensus': Right hemisphere holistic polarity [batch, 1].
                - 'readout_vector': Full multi-observable readout [batch, num_qubits].
                - 'state': Evolved quantum statevector [batch, 2^num_qubits].
        """
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)

        dev = self.W_left.device
        real_dtype = self.W_left.dtype
        cdtype = ops.resolve_complex_dtype(real_dtype, dev)
        N = self.num_qubits
        batch_size = x.shape[0]

        if x.device != dev or x.dtype != real_dtype:
            x = x.to(device=dev, dtype=real_dtype)

        # Construct biomorphic Hamiltonian
        H_total, eff_time = self._build_hamiltonian(x)

        # Initial reference state (equal superposition |+>^N if 'superposition')
        init_mode = self.initial_state_mode
        if isinstance(init_mode, str) and init_mode.lower() in ("superposition", "uniform"):
            init_mode = "plus"

        psi_batch = ops.create_initial_state(
            init_mode,
            num_qubits=N,
            batch_size=batch_size,
            device=dev,
            dtype=cdtype,
        )

        # Continuous Unitary Evolution |psi(t)> = exp(-i H t) |psi_0>
        # Diagonalize Hermitian Hamiltonian per batch
        eigvals, eigvecs = torch.linalg.eigh(H_total)  # [B, dim], [B, dim, dim]
        phases = torch.exp(-1j * (eigvals * eff_time))  # [B, dim]
        diag_phases = torch.diag_embed(phases)  # [B, dim, dim]
        U_total = eigvecs @ diag_phases @ eigvecs.mH  # [B, dim, dim]

        psi_t = (U_total @ psi_batch.unsqueeze(-1)).squeeze(-1)  # [B, dim]

        # Holistic Readouts across both Hemispheres via fast O(2^N) projector
        all_z = ops.fast_z_readout(psi_t, N)  # [B, N]

        # Left Hemisphere Consensus: Average of Left Qubits
        left_consensus = all_z[:, : self.num_left_qubits].mean(dim=-1, keepdim=True)
        # Right Hemisphere Consensus: Average of Right Qubits
        right_consensus = all_z[:, self.num_left_qubits :].mean(dim=-1, keepdim=True)
        # Parliament Consensus: Global Unanimous Polarity
        parliament_consensus = all_z.mean(dim=-1, keepdim=True)

        res: dict[str, torch.Tensor] = {
            "consensus": parliament_consensus if not is_1d else parliament_consensus.squeeze(0),
            "left_consensus": left_consensus if not is_1d else left_consensus.squeeze(0),
            "right_consensus": right_consensus if not is_1d else right_consensus.squeeze(0),
            "readout_vector": all_z if not is_1d else all_z.squeeze(0),
            "state": psi_t if not is_1d else psi_t.squeeze(0),
        }
        if self.csf_environment is not None:
            res["kappa_csf"] = self.csf_environment.compute_attenuation_factor()
        return res

    def attach_csf_environment(self, csf_env: CSFShieldedEnvironment) -> None:
        """Attaches a Cerebrospinal Fluid (CSF) biophysical quantum shielding environment."""
        self.csf_environment = csf_env

    def __repr__(self) -> str:
        return (
            f"BiomorphicResonantBrain(in_features={self.in_features}, "
            f"left_qubits={self.num_left_qubits}, right_qubits={self.num_right_qubits}, "
            f"neuromodulation={self.enable_neuromodulation}, oxygenation={self.enable_oxygenation})"
        )


class CSFShieldedResonantLayer(nn.Module):
    """Continuous Resonant Quantum Neural Network Layer with CSF Biophysical Quantum Shielding.

    Combines bipartite continuous-time Hamiltonian graph dynamics with environmental
    open-system Lindblad dephasing attenuated by Cerebrospinal Fluid (CSF) biophysics:
        Gamma_eff = kappa_CSF * Gamma_bare
        C(t) = exp(-Gamma_eff * t / 2)

    Args:
        in_features: Input feature dimension.
        num_left_qubits: Left hemisphere qubit count (default: 2).
        num_right_qubits: Right hemisphere qubit count (default: 2).
        bare_dephasing_rate: Unshielded bare dephasing rate Gamma_bare (default: 1.0).
        csf_environment: Optional custom CSFShieldedEnvironment instance.
        env: Optional alias for csf_environment.
        return_dict: Whether to return full diagnostic dictionary or consensus scalar.
        learnable_callosum: Whether corpus callosum inter-lobe tunneling is trainable.
        enable_neuromodulation: Whether dopamine, norepinephrine, serotonin channels are active.
        enable_oxygenation: Whether metabolic energy constraint is enforced.
        initial_state: Reference quantum state ('superposition', 'zero', etc.).
        device: Target PyTorch execution device.
        dtype: Real floating-point precision.
    """

    def __init__(
        self,
        in_features: int,
        num_left_qubits: int = 2,
        num_right_qubits: int = 2,
        bare_dephasing_rate: float = 1.0,
        csf_environment: CSFShieldedEnvironment | None = None,
        env: CSFShieldedEnvironment | None = None,
        return_dict: bool = True,
        learnable_callosum: bool = True,
        enable_neuromodulation: bool = True,
        enable_oxygenation: bool = True,
        initial_state: str = "superposition",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_left_qubits < 1 or num_right_qubits < 1:
            raise ValueError("Hemisphere qubit count must be >= 1")
        if bare_dephasing_rate < 0.0:
            raise ValueError("bare_dephasing_rate cannot be negative")

        self.in_features = in_features
        self.bare_dephasing_rate = bare_dephasing_rate
        self.return_dict = return_dict

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

        self.real_dtype = dtype if dtype is not None else torch.float32
        self.complex_dtype = ops.resolve_complex_dtype(self.real_dtype, dev)

        self.brain = BiomorphicResonantBrain(
            in_features=in_features,
            num_left_qubits=num_left_qubits,
            num_right_qubits=num_right_qubits,
            learnable_callosum=learnable_callosum,
            enable_neuromodulation=enable_neuromodulation,
            enable_oxygenation=enable_oxygenation,
            initial_state=initial_state,
            device=dev,
            dtype=self.real_dtype,
        )

        selected_env = csf_environment if csf_environment is not None else env
        if selected_env is None:
            selected_env = CSFShieldedEnvironment(device=dev, dtype=self.real_dtype)
        self.csf_environment = selected_env
        self.env = selected_env

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass executing Hamiltonian resonance coupled with CSF dephasing attenuation.

        Args:
            x: Input tensor [batch, in_features], 1D [in_features],
                or 3D [batch, seq_len, in_features].

        Returns:
            Dictionary containing consensus, state, readout_vector, kappa_csf, effective_dephasing,
            coherence_factor, or if return_dict=False, consensus scalar tensor.
        """
        orig_dim = x.dim()
        if orig_dim == 1:
            x_2d = x.unsqueeze(0)
        elif orig_dim == 3:
            b_orig, l_orig, d_orig = x.shape
            x_2d = x.reshape(b_orig * l_orig, d_orig)
        else:
            x_2d = x

        batch_size = x_2d.shape[0]
        dev = self.brain.W_left.device
        real_dtype = self.real_dtype
        cdtype = self.complex_dtype
        n_qubits = self.brain.num_qubits
        dim = 2**n_qubits

        # Handle empty batch
        if batch_size == 0:
            zero_consensus = torch.zeros((0, 1), device=dev, dtype=real_dtype)
            zero_left = torch.zeros((0, 1), device=dev, dtype=real_dtype)
            zero_right = torch.zeros((0, 1), device=dev, dtype=real_dtype)
            zero_readout = torch.zeros((0, n_qubits), device=dev, dtype=real_dtype)
            zero_state = torch.zeros((0, dim), device=dev, dtype=cdtype)
            kappa_csf = self.csf_environment.compute_attenuation_factor()
            gamma_eff = self.csf_environment.apply_shielding(self.bare_dephasing_rate)
            coherence_factor = torch.ones((0, 1), device=dev, dtype=real_dtype)
            if self.return_dict:
                return {
                    "consensus": zero_consensus,
                    "left_consensus": zero_left,
                    "right_consensus": zero_right,
                    "readout_vector": zero_readout,
                    "state": zero_state,
                    "kappa_csf": kappa_csf,
                    "effective_dephasing": gamma_eff,
                    "coherence_factor": coherence_factor,
                }
            return zero_consensus

        if x_2d.device != dev or x_2d.dtype != real_dtype:
            x_2d = x_2d.to(device=dev, dtype=real_dtype)

        # 1. Biomorphic Hamiltonian and effective duration
        h_total, eff_time = self.brain._build_hamiltonian(x_2d)

        # 2. Reference state
        init_mode = self.brain.initial_state_mode
        if isinstance(init_mode, str) and init_mode.lower() in ("superposition", "uniform"):
            init_mode = "plus"
        psi_batch = ops.create_initial_state(
            init_mode,
            num_qubits=n_qubits,
            batch_size=batch_size,
            device=dev,
            dtype=cdtype,
        )

        # 3. Unitary continuous-time state evolution
        eigvals, eigvecs = torch.linalg.eigh(h_total)
        phases = torch.exp(-1j * (eigvals * eff_time))
        diag_phases = torch.diag_embed(phases)
        u_total = eigvecs @ diag_phases @ eigvecs.mH
        psi_t = (u_total @ psi_batch.unsqueeze(-1)).squeeze(-1)

        # Renormalize to maintain exact ||psi|| = 1
        norm = torch.linalg.norm(psi_t, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)
        psi_t = psi_t / norm

        # 4. CSF Dephasing Attenuation
        kappa_csf = self.csf_environment.compute_attenuation_factor()
        gamma_eff = self.csf_environment.apply_shielding(self.bare_dephasing_rate)
        coherence_factor = torch.exp(-0.5 * gamma_eff * eff_time)

        # 5. Readout expectations
        all_z = ops.fast_z_readout(psi_t, n_qubits)  # [B, N]
        readout_shielded = all_z * coherence_factor
        readout_shielded = torch.clamp(readout_shielded, min=-1.0, max=1.0)

        n_left = self.brain.num_left_qubits
        left_consensus = readout_shielded[:, :n_left].mean(dim=-1, keepdim=True)
        right_consensus = readout_shielded[:, n_left:].mean(dim=-1, keepdim=True)
        parliament_consensus = readout_shielded.mean(dim=-1, keepdim=True)

        if orig_dim == 1:
            res_consensus = parliament_consensus.squeeze(0)
            res_left = left_consensus.squeeze(0)
            res_right = right_consensus.squeeze(0)
            res_readout = readout_shielded.squeeze(0)
            res_state = psi_t.squeeze(0)
            res_coherence = coherence_factor.squeeze(0)
        elif orig_dim == 3:
            res_consensus = parliament_consensus.reshape(b_orig, l_orig, 1)
            res_left = left_consensus.reshape(b_orig, l_orig, 1)
            res_right = right_consensus.reshape(b_orig, l_orig, 1)
            res_readout = readout_shielded.reshape(b_orig, l_orig, n_qubits)
            res_state = psi_t.reshape(b_orig, l_orig, dim)
            res_coherence = coherence_factor.reshape(b_orig, l_orig, 1)
        else:
            res_consensus = parliament_consensus
            res_left = left_consensus
            res_right = right_consensus
            res_readout = readout_shielded
            res_state = psi_t
            res_coherence = coherence_factor

        if not self.return_dict:
            return res_consensus

        return {
            "consensus": res_consensus,
            "left_consensus": res_left,
            "right_consensus": res_right,
            "readout_vector": res_readout,
            "state": res_state,
            "kappa_csf": kappa_csf,
            "effective_dephasing": gamma_eff,
            "coherence_factor": res_coherence,
        }

    def __repr__(self) -> str:
        return (
            f"CSFShieldedResonantLayer(in_features={self.in_features}, "
            f"left_qubits={self.brain.num_left_qubits}, "
            f"right_qubits={self.brain.num_right_qubits}, "
            f"bare_dephasing={self.bare_dephasing_rate}, "
            f"csf_env={repr(self.csf_environment)})"
        )


class QuantumREMSleep(nn.Module):
    """Offline closed-system quantum REM sleep annealing module.

    Orthogonalizes stored memory statevectors from prior tasks through uncoupled
    closed-system Hamiltonian evolution (x = 0), preventing catastrophic forgetting
    in continual learning without requiring external data replay.

    Args:
        brain_module: Reference to the underlying BiomorphicResonantBrain module.
        sleep_cycles: Default number of annealing optimization cycles (default: 10).
        learning_rate: Gradient descent step size for sleep consolidation (default: 0.01).
        orthogonalization_weight: Scaling weight for the orthogonalization loss (default: 1.0).
        device: PyTorch device ('cpu', 'mps', or torch.device).
        dtype: Real floating-point dtype.
    """

    def __init__(
        self,
        brain_module: BiomorphicResonantBrain,
        sleep_cycles: int = 10,
        learning_rate: float = 0.01,
        orthogonalization_weight: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.brain_module = brain_module
        self.sleep_cycles = sleep_cycles
        self.learning_rate = learning_rate
        self.orthogonalization_weight = orthogonalization_weight

        target_dev = ops.resolve_device(device) if device is not None else None
        dev = target_dev if target_dev is not None else brain_module.W_left.device
        if (
            target_dev is not None
            and target_dev.type == "mps"
            and dtype in (torch.float64, torch.complex128)
        ):
            raise ops.UnsupportedDtypeError(
                f"Apple Silicon MPS does not support 64-bit precision ({dtype}). "
                f"Use torch.float32 on MPS or switch execution to device='cpu'."
            )
        self.real_dtype = (
            dtype if dtype is not None else (torch.float32 if dev.type == "mps" else torch.float64)
        )
        self.memory_states: list[torch.Tensor] = []

    def register_memory_state(self, state: torch.Tensor) -> None:
        """Stores prototype statevectors from prior tasks for sleep consolidation.

        Args:
            state: Quantum statevector of shape [dim] or batch of statevectors [batch, dim].
        """
        dev = self.brain_module.W_left.device
        cdtype = self.brain_module.complex_dtype
        st = state.detach().to(device=dev, dtype=cdtype)
        if st.dim() == 1:
            norm = torch.linalg.norm(st)
            if norm > 1e-12:
                st = st / norm
            self.memory_states.append(st.clone())
        elif st.dim() == 2:
            for s in st:
                norm = torch.linalg.norm(s)
                if norm > 1e-12:
                    s = s / norm
                self.memory_states.append(s.clone())
        else:
            raise ValueError(f"Expected 1D or 2D state tensor, got shape {tuple(state.shape)}")

    def clear_memory_states(self) -> None:
        """Clears all stored memory prototype statevectors."""
        self.memory_states.clear()

    def sleep(
        self,
        memory_states: Sequence[torch.Tensor] | None = None,
        cycles: int | None = None,
    ) -> dict[str, Any]:
        """Executes closed-system unitary annealing at x = 0 to orthogonalize representations.

        Evaluates pairwise Hilbert-Schmidt overlap:
            L_ortho = sum_{j < k} |<psi_j | psi_k>|^2
        and optimizes brain parameters (J_right, J_callosum, h_left) via gradient
        descent to drive memory representations into mutually orthogonal subspaces.

        Args:
            memory_states: Optional explicit sequence of statevectors. If None, uses
                registered prototype states.
            cycles: Optional override for the number of annealing cycles.

        Returns:
            Dictionary containing:
                - 'initial_overlap': Pairwise overlap before sleep annealing.
                - 'final_overlap': Pairwise overlap after sleep annealing.
                - 'cycles': Number of annealing cycles executed.
                - 'loss_history': List of float losses per cycle.
                - 'retention_estimate': Theoretical task retention estimate in [0, 1].
        """
        active_states: list[torch.Tensor]
        if memory_states is not None:
            active_states = [s.detach() for s in memory_states]
        else:
            active_states = [s.detach() for s in self.memory_states]

        if len(active_states) < 2:
            raise ValueError(
                f"At least two memory states are required for sleep orthogonalization, "
                f"got {len(active_states)}."
            )

        num_cycles = cycles if cycles is not None else self.sleep_cycles
        dev = self.brain_module.W_left.device
        real_dtype = self.brain_module.W_left.dtype
        cdtype = self.brain_module.complex_dtype
        M = len(active_states)

        # Standardize states on device and cdtype
        normed_states = []
        for s in active_states:
            s_t = s.to(device=dev, dtype=cdtype)
            norm = torch.linalg.norm(s_t)
            if norm > 1e-12:
                s_t = s_t / norm
            normed_states.append(s_t)

        # Select trainable parameters in the brain module
        trainable_params: list[nn.Parameter] = []
        for p in [
            self.brain_module.J_right,
            self.brain_module.J_callosum,
            self.brain_module.h_left,
        ]:
            if isinstance(p, nn.Parameter) and p.requires_grad:
                trainable_params.append(p)

        optimizer = (
            torch.optim.Adam(trainable_params, lr=self.learning_rate)
            if trainable_params
            else None
        )

        loss_history: list[float] = []
        initial_overlap = 0.0

        for step in range(num_cycles):
            if optimizer is not None:
                optimizer.zero_grad()

            x_zero = torch.zeros((1, self.brain_module.in_features), device=dev, dtype=real_dtype)
            H_total, eff_time = self.brain_module._build_hamiltonian(x_zero)
            H_free = H_total[0]  # [dim, dim]

            eigvals, eigvecs = torch.linalg.eigh(H_free)  # [dim], [dim, dim]

            # Project states onto eigenbasis: c_j = eigvecs^H @ psi_j
            c_matrix = eigvecs.mH @ torch.stack(normed_states, dim=1)  # [dim, M]
            p_matrix = torch.abs(c_matrix) ** 2  # [dim, M]

            # Dynamic consolidation states
            evolved_states = []
            for j in range(M):
                t_j = eff_time[0, 0] * float(j + 1) / float(M)
                phase_j = torch.exp(-1j * (eigvals * t_j))
                psi_j_evolved = eigvecs @ (phase_j * c_matrix[:, j])
                evolved_states.append(psi_j_evolved)
            evolved_matrix = torch.stack(evolved_states, dim=1)  # [dim, M]

            pair_count = M * (M - 1) // 2
            loss_ortho = torch.tensor(0.0, device=dev, dtype=real_dtype)

            for j in range(M):
                for k in range(j + 1, M):
                    # Ergodic eigenspace overlap (Theorem 1 Eq. 74)
                    ergodic_overlap = torch.sum(p_matrix[:, j] * p_matrix[:, k])
                    # Dynamic state overlap |<psi_j(t) | psi_k(t)>|^2
                    dyn_overlap = (
                        torch.abs(torch.vdot(evolved_matrix[:, j], evolved_matrix[:, k])) ** 2
                    )
                    loss_ortho = loss_ortho + (ergodic_overlap + dyn_overlap) * 0.5

            loss_ortho = (loss_ortho / pair_count) * self.orthogonalization_weight

            if step == 0:
                initial_overlap = float(loss_ortho.item())

            loss_history.append(float(loss_ortho.item()))

            if optimizer is not None and trainable_params:
                loss_ortho.backward()
                optimizer.step()

        # Final evaluation after updates
        with torch.no_grad():
            x_zero = torch.zeros((1, self.brain_module.in_features), device=dev, dtype=real_dtype)
            H_total, eff_time = self.brain_module._build_hamiltonian(x_zero)
            eigvals, eigvecs = torch.linalg.eigh(H_total[0])
            c_matrix = eigvecs.mH @ torch.stack(normed_states, dim=1)
            p_matrix = torch.abs(c_matrix) ** 2

            evolved_states = []
            for j in range(M):
                t_j = eff_time[0, 0] * float(j + 1) / float(M)
                phase_j = torch.exp(-1j * (eigvals * t_j))
                psi_j_evolved = eigvecs @ (phase_j * c_matrix[:, j])
                evolved_states.append(psi_j_evolved)
            evolved_matrix = torch.stack(evolved_states, dim=1)

            final_loss = torch.tensor(0.0, device=dev, dtype=real_dtype)
            for j in range(M):
                for k in range(j + 1, M):
                    ergodic_overlap = torch.sum(p_matrix[:, j] * p_matrix[:, k])
                    dyn_overlap = (
                        torch.abs(torch.vdot(evolved_matrix[:, j], evolved_matrix[:, k])) ** 2
                    )
                    final_loss = final_loss + (ergodic_overlap + dyn_overlap) * 0.5
            final_overlap = float(
                ((final_loss / pair_count) * self.orthogonalization_weight).item()
            )

            if memory_states is None:
                self.memory_states = [evolved_matrix[:, j].clone() for j in range(M)]

        retention_estimate = max(0.0, min(1.0, 1.0 - final_overlap))

        return {
            "initial_overlap": initial_overlap,
            "final_overlap": final_overlap,
            "cycles": num_cycles,
            "loss_history": loss_history,
            "retention_estimate": retention_estimate,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Routes input through the underlying biomorphic quantum brain module.

        Args:
            x: Input tensor of shape [batch_size, in_features] or [in_features].

        Returns:
            Brain module consensus output dictionary.
        """
        return self.brain_module.forward(x)

    def trigger_glymphatic_reset(self, env: CSFShieldedEnvironment) -> None:
        """Triggers nocturnal convective AQP4 glymphatic flush mode resetting entropy."""
        env.simulate_clinical_condition("rem_sleep")

    def consolidate_hippocampus(
        self,
        hippocampus: NoisyHippocampalBuffer,
        cycles: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Consolidates memory engrams replayed from a NoisyHippocampalBuffer.

        Args:
            hippocampus: NoisyHippocampalBuffer containing episodic memory traces.
            cycles: Optional override for sleep annealing cycles.
            batch_size: Optional SWR replay batch size.

        Returns:
            Dictionary containing consolidation diagnostics and hippocampal metrics.
        """
        if hasattr(hippocampus, "csf_environment") and hippocampus.csf_environment is not None:
            env = hippocampus.csf_environment
            old_cond = getattr(env, "_clinical_condition", "normal")
            env.simulate_clinical_condition("rem_sleep")
            try:
                return hippocampus.consolidate_with_sleep(
                    self, cycles=cycles, batch_size=batch_size
                )
            finally:
                env.simulate_clinical_condition(old_cond)
        return hippocampus.consolidate_with_sleep(self, cycles=cycles, batch_size=batch_size)

    def __repr__(self) -> str:
        return (
            f"QuantumREMSleep(sleep_cycles={self.sleep_cycles}, "
            f"learning_rate={self.learning_rate}, "
            f"orthogonalization_weight={self.orthogonalization_weight}, "
            f"stored_states={len(self.memory_states)})"
        )


class NoisyHippocampalBuffer(nn.Module):
    _dummy: torch.Tensor

    """Biologically realistic noisy hippocampal episodic memory buffer (CA3-CA1).

    Simulates mammalian episodic memory buffering under continuous thermal noise,
    Lindblad phase diffusion, and dopaminergic synaptic capture (Frey & Morris, 1997).
    Replaces idealized lossless engram replay with biologically authentic stochastic
    degradation:

    1. Finite Buffer Capacity & Eviction: Holds up to `capacity` episodic engrams.
       When capacity is reached, evicts the oldest engrams (FIFO).
    2. Lindblad Phase Diffusion: Continuous dephasing drift on quantum engram phases:
         psi_k(t + dt) = psi_k(t) * exp(i * d_theta_k), d_theta_k ~ N(0, sigma_phi_eff^2 * dt)
    3. Thermal Amplitude Jitter: Background depolarizing synaptic fluctuations:
         psi(t + dt) = (psi(t) + xi) / ||psi(t) + xi||,  xi ~ CN(0, sigma_noise_eff^2 * dt)
    4. Dopaminergic Synaptic Tagging: Novelty and reward tags scale down effective noise:
         sigma_eff = sigma / (1.0 + lambda_D * max(0.0, dopamine_tag))
    5. Sharp-Wave Ripple (SWR) Replay: Probabilistic sampling of degraded engrams
       during offline sleep consolidation (coupled with QuantumREMSleep).
    6. Objective Ebbinghaus Fidelity Tracking: Computes real-time retention fidelity
       F(t) = |<psi_0 | psi(t)>|^2 to observe biological forgetting curves.

    Args:
        capacity: Maximum number of episodic engram slots (default: 64).
        noise_level: Thermal amplitude jitter standard deviation sigma_noise (default: 0.05).
        phase_diffusion_rate: Phase diffusion rate sigma_phi (default: 0.02).
        temporal_decay_rate: Ebbinghaus Lindblad decay constant gamma (default: 0.01).
        dopamine_protection: Sensitivity parameter lambda_D for dopamine stabilization (1.0).
        device: PyTorch device ('cpu', 'mps', or torch.device).
        dtype: Real floating-point dtype (torch.float32 or torch.float64).
    """

    def __init__(
        self,
        capacity: int = 64,
        noise_level: float = 0.05,
        phase_diffusion_rate: float = 0.02,
        temporal_decay_rate: float = 0.01,
        dopamine_protection: float = 1.0,
        csf_environment: CSFShieldedEnvironment | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.csf_environment = csf_environment
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if noise_level < 0.0:
            raise ValueError(f"noise_level must be non-negative, got {noise_level}")
        if phase_diffusion_rate < 0.0:
            raise ValueError(
                f"phase_diffusion_rate must be non-negative, got {phase_diffusion_rate}"
            )
        if temporal_decay_rate < 0.0:
            raise ValueError(
                f"temporal_decay_rate must be non-negative, got {temporal_decay_rate}"
            )
        if dopamine_protection < 0.0:
            raise ValueError(
                f"dopamine_protection must be non-negative, got {dopamine_protection}"
            )

        self.capacity = capacity
        self.noise_level = float(noise_level)
        self.phase_diffusion_rate = float(phase_diffusion_rate)
        self.temporal_decay_rate = float(temporal_decay_rate)
        self.dopamine_protection = float(dopamine_protection)

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
        self.real_dtype = (
            dtype if dtype is not None else (torch.float32 if dev.type == "mps" else torch.float64)
        )
        self.complex_dtype = ops.resolve_complex_dtype(self.real_dtype, dev)

        # Register a dummy buffer to track module device and dtype transfers automatically
        self.register_buffer("_dummy", torch.empty(0, device=dev, dtype=self.real_dtype))

        self.buffer: list[dict[str, Any]] = []

    @property
    def current_device(self) -> torch.device:
        return self._dummy.device

    @property
    def current_real_dtype(self) -> torch.dtype:
        return self._dummy.dtype

    @property
    def current_complex_dtype(self) -> torch.dtype:
        return ops.resolve_complex_dtype(self._dummy.dtype, self._dummy.device)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        res = super()._apply(fn, recurse=recurse)
        dev = self.current_device
        cdtype = self.current_complex_dtype
        for e in self.buffer:
            e["pristine_state"] = e["pristine_state"].to(device=dev, dtype=cdtype)
            e["degraded_state"] = e["degraded_state"].to(device=dev, dtype=cdtype)
        return res

    def attach_csf_environment(self, csf_env: CSFShieldedEnvironment) -> None:
        """Attaches a Cerebrospinal Fluid (CSF) biophysical quantum shielding environment."""
        self.csf_environment = csf_env

    def store(
        self,
        state: torch.Tensor,
        dopamine_tag: float | torch.Tensor = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> list[int] | int:
        """Stores engram statevector(s) in hippocampal buffer with dopaminergic tags.

        Args:
            state: Statevector of shape [dim] or batch [B, dim].
            dopamine_tag: Dopaminergic salience tag (scalar or [B] tensor).
            metadata: Optional dictionary with custom contextual tags.

        Returns:
            Index or list of indices assigned to stored engram(s).
        """
        if state.dim() == 1:
            return self._store_single(state, dopamine_tag, metadata)
        elif state.dim() == 2:
            B = state.shape[0]
            indices_list: list[int] = []
            if isinstance(dopamine_tag, (int, float)):
                d_tags = [float(dopamine_tag)] * B
            elif torch.is_tensor(dopamine_tag):
                if dopamine_tag.dim() == 0:
                    d_tags = [float(dopamine_tag.item())] * B
                else:
                    d_tags = [float(d.item()) for d in dopamine_tag.flatten()[:B]]
                    if len(d_tags) < B:
                        d_tags.extend([1.0] * (B - len(d_tags)))
            else:
                d_tags = [1.0] * B

            for i in range(B):
                idx = self._store_single(state[i], d_tags[i], metadata)
                indices_list.append(idx)
            return indices_list
        else:
            raise ValueError(f"Expected 1D or 2D state tensor, got shape {tuple(state.shape)}")

    def _store_single(
        self,
        state: torch.Tensor,
        dopamine_tag: float | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        dev = self.current_device
        cdtype = self.current_complex_dtype

        st = state.detach().to(device=dev, dtype=cdtype)
        norm = torch.linalg.norm(st)
        if norm > 1e-12:
            st = st / norm
        else:
            raise ValueError("Cannot store null/zero statevector in hippocampal buffer.")

        d_val = float(dopamine_tag.item()) if torch.is_tensor(dopamine_tag) else float(dopamine_tag)
        meta = dict(metadata) if metadata is not None else {}

        # FIFO eviction if capacity is reached
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        engram = {
            "pristine_state": st.clone(),
            "degraded_state": st.clone(),
            "dopamine_tag": max(0.0, d_val),
            "age": 0.0,
            "metadata": meta,
        }
        self.buffer.append(engram)
        return len(self.buffer) - 1

    def step(self, dt: float = 1.0) -> None:
        """Simulates biological time progression with Lindblad phase diffusion & thermal noise.

        Args:
            dt: Biological time step duration (default: 1.0).
        """
        if dt <= 0.0 or not self.buffer:
            return

        dev = self.current_device
        rdtype = self.current_real_dtype
        cdtype = self.current_complex_dtype

        for engram in self.buffer:
            d_tag = engram["dopamine_tag"]
            protection = 1.0 + self.dopamine_protection * d_tag

            # Effective noise scaled by dopamine protection and sqrt(dt)
            sigma_noise_eff = (self.noise_level / protection) * math.sqrt(dt)
            sigma_phi_eff = (self.phase_diffusion_rate / protection) * math.sqrt(dt)
            gamma_decay_eff = (self.temporal_decay_rate / protection) * dt

            if self.csf_environment is not None:
                kappa = self.csf_environment.compute_attenuation_factor()
                sqrt_kappa = float(torch.sqrt(kappa).item())
                sigma_noise_eff *= sqrt_kappa
                sigma_phi_eff *= sqrt_kappa
                gamma_decay_eff *= float(kappa.item())

            current_state = engram["degraded_state"]
            dim = current_state.shape[0]

            # 1. Phase diffusion: psi_k -> psi_k * exp(i * delta_theta_k)
            if sigma_phi_eff > 1e-9:
                delta_theta = torch.randn(dim, device=dev, dtype=rdtype) * sigma_phi_eff
                phase_factor = torch.exp(1j * delta_theta.to(dtype=cdtype))
                current_state = current_state * phase_factor

            # 2. Thermal amplitude noise: xi ~ CN(0, sigma_noise_eff^2 * I)
            if sigma_noise_eff > 1e-9:
                scale = sigma_noise_eff / math.sqrt(2.0)
                noise_r = torch.randn(dim, device=dev, dtype=rdtype) * scale
                noise_i = torch.randn(dim, device=dev, dtype=rdtype) * scale
                xi = torch.complex(noise_r, noise_i)
                current_state = current_state + xi

            # 3. Temporal Lindblad dephasing damping (Ebbinghaus drift toward maximally mixed noise)
            if gamma_decay_eff > 1e-9:
                decay_factor = math.exp(-gamma_decay_eff)
                rand_r = torch.randn(dim, device=dev, dtype=rdtype)
                rand_i = torch.randn(dim, device=dev, dtype=rdtype)
                thermal_bath_state = torch.complex(rand_r, rand_i)
                bath_norm = torch.linalg.norm(thermal_bath_state)
                if bath_norm > 1e-12:
                    thermal_bath_state = thermal_bath_state / bath_norm
                current_state = (
                    decay_factor * current_state + (1.0 - decay_factor) * thermal_bath_state
                )

            # 4. Renormalize to maintain unitary physical validity
            norm = torch.linalg.norm(current_state)
            if norm > 1e-12:
                current_state = current_state / norm

            engram["degraded_state"] = current_state
            engram["age"] += dt

    def replay(
        self,
        batch_size: int | None = None,
        temperature: float = 1.0,
        apply_degradation: bool = True,
    ) -> torch.Tensor:
        """Samples episodic engrams (Sharp-Wave Ripples) for offline consolidation.

        Args:
            batch_size: Number of engrams to sample. If None or >= len(buffer), returns all.
            temperature: Sampling temperature T. Lower T concentrates on highest dopamine engrams;
                T <= 1e-4 selects deterministically by dopamine rank.
            apply_degradation: If True, returns biologically degraded states;
                if False, returns pristine.

        Returns:
            Tensor of shape [batch_size, dim] containing complex engram statevectors.
        """
        if not self.buffer:
            raise ValueError("Hippocampal buffer is empty. Cannot replay engrams.")

        N = len(self.buffer)
        target_k = min(N, batch_size) if batch_size is not None and batch_size > 0 else N
        key = "degraded_state" if apply_degradation else "pristine_state"

        if target_k == N and temperature >= 1.0:
            return torch.stack([e[key] for e in self.buffer], dim=0)

        d_tags = torch.tensor([e["dopamine_tag"] for e in self.buffer], dtype=torch.float32)

        if temperature <= 1e-4:
            _, top_indices = torch.topk(d_tags, k=target_k)
            selected_indices = top_indices.tolist()
        else:
            logits = d_tags / max(1e-4, float(temperature))
            probs = torch.softmax(logits, dim=0)
            selected_indices = torch.multinomial(
                probs, num_samples=target_k, replacement=False
            ).tolist()

        return torch.stack([self.buffer[i][key] for i in selected_indices], dim=0)

    def consolidate_with_sleep(
        self,
        rem_sleep: QuantumREMSleep,
        cycles: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Consolidates buffered degraded engrams into neocortex via QuantumREMSleep.

        Replaces idealized clean states with biologically degraded engrams replayed
        from the hippocampus.

        Args:
            rem_sleep: QuantumREMSleep module instance to perform neocortical orthogonalization.
            cycles: Optional override for sleep annealing cycles.
            batch_size: Optional batch size for SWR replay sampling.

        Returns:
            Sleep consolidation result dictionary with added hippocampal metadata.
        """
        replayed_states = self.replay(batch_size=batch_size, apply_degradation=True)
        rem_sleep.clear_memory_states()
        rem_sleep.register_memory_state(replayed_states)
        sleep_result = rem_sleep.sleep(cycles=cycles)

        fidelities = self.get_fidelities()
        mean_fidelity = float(sum(fidelities) / len(fidelities)) if fidelities else 0.0

        sleep_result["hippocampal_mean_fidelity"] = mean_fidelity
        sleep_result["hippocampal_fidelities"] = fidelities
        sleep_result["hippocampal_engram_count"] = len(self.buffer)
        return sleep_result

    def get_fidelities(self) -> list[float]:
        """Returns the current state fidelity F = |<psi_0 | psi(t)>|^2 for all engrams."""
        res: list[float] = []
        for e in self.buffer:
            overlap = torch.vdot(e["pristine_state"], e["degraded_state"])
            fid = float((torch.abs(overlap) ** 2).item())
            res.append(max(0.0, min(1.0, fid)))
        return res

    def get_mean_fidelity(self) -> float:
        """Returns the mean engram fidelity across all active engrams."""
        fids = self.get_fidelities()
        return float(sum(fids) / len(fids)) if fids else 1.0

    def clear(self) -> None:
        """Clears all stored engrams from buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        mean_fid = self.get_mean_fidelity()
        return (
            f"NoisyHippocampalBuffer(capacity={self.capacity}, "
            f"stored={len(self.buffer)}, "
            f"mean_fidelity={mean_fid:.3f}, "
            f"noise_level={self.noise_level}, "
            f"phase_diffusion={self.phase_diffusion_rate})"
        )


class QuantumZenoAttention(nn.Module):
    """Quantum Zeno and Anti-Zeno dynamic cognitive attention mechanism.

    Balances attentional focus pinning (Quantum Zeno Effect) and divergent
    exploratory tunneling with phase kickback (Anti-Zeno Effect) modulated by
    internal or external dopamine signals.

    Args:
        dim: Feature dimension / Hilbert space state dimension.
        observation_frequency: Projective monitoring frequency nu_obs (default: 10.0).
        dopamine_coupling: Coupling strength lambda_D to dopamine surges (default: 0.5).
        num_heads: Number of attention heads (default: 1, must divide dim).
        learnable_frequency: Whether observation_frequency is trainable parameter.
        device: PyTorch device ('cpu', 'mps', or torch.device).
        dtype: Real floating-point dtype.
    """

    def __init__(
        self,
        dim: int,
        observation_frequency: float = 10.0,
        dopamine_coupling: float = 0.5,
        num_heads: int = 1,
        learnable_frequency: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

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
        self.real_dtype = (
            dtype if dtype is not None else (torch.float32 if dev.type == "mps" else torch.float64)
        )

        # 1. Observation frequency parameter
        if learnable_frequency:
            self.obs_freq = nn.Parameter(
                torch.tensor(float(observation_frequency), device=dev, dtype=self.real_dtype)
            )
        else:
            self.register_buffer(
                "obs_freq",
                torch.tensor(float(observation_frequency), device=dev, dtype=self.real_dtype),
            )

        # 2. Dopamine coupling parameter
        self.dopamine_coupling = nn.Parameter(
            torch.tensor(float(dopamine_coupling), device=dev, dtype=self.real_dtype)
        )

        # 3. Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)
        self.k_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)
        self.v_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)
        self.kickback_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)

        # 4. Internal dopamine arousal sensor
        self.dopamine_sensor = nn.Linear(dim, 1, device=dev, dtype=self.real_dtype)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.orthogonal_(self.kickback_proj.weight)
        nn.init.kaiming_uniform_(self.dopamine_sensor.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dopamine_sensor.bias)

    def forward(
        self,
        x: torch.Tensor,
        dopamine: torch.Tensor | float | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass implementing Zeno focus pinning vs Anti-Zeno phase tunneling.

        Args:
            x: Input tensor of shape [dim], [batch, dim], or [batch, seq_len, dim].
            dopamine: Optional external dopamine signal tensor or scalar. If None,
                estimated internally via dopamine_sensor.
            return_diagnostics: If True, returns full diagnostic dictionary.

        Returns:
            Output tensor of same shape as x, or dictionary if return_diagnostics=True.
        """
        orig_dim = x.dim()
        dev = self.q_proj.weight.device
        real_dtype = self.q_proj.weight.dtype

        if x.device != dev or x.dtype != real_dtype:
            x = x.to(device=dev, dtype=real_dtype)

        if orig_dim == 1:
            x_3d = x.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        elif orig_dim == 2:
            x_3d = x.unsqueeze(1)  # [B, 1, dim]
        elif orig_dim == 3:
            x_3d = x
        else:
            raise ValueError(f"Expected 1D, 2D, or 3D tensor, got shape {tuple(x.shape)}")

        B, L, _ = x_3d.shape

        # 1. Evaluate dopamine arousal d
        if dopamine is None:
            d = torch.sigmoid(self.dopamine_sensor(x_3d))  # [B, L, 1]
        elif isinstance(dopamine, (int, float)):
            d = torch.full((B, L, 1), float(dopamine), device=dev, dtype=real_dtype)
        elif torch.is_tensor(dopamine):
            d_in = dopamine.to(device=dev, dtype=real_dtype)
            if d_in.dim() == 0:
                d = d_in.expand(B, L, 1)
            elif d_in.dim() == 1:
                if d_in.shape[0] == B:
                    d = d_in.unsqueeze(1).unsqueeze(2).expand(B, L, 1)
                else:
                    d = d_in.expand(B, L, 1)
            elif d_in.dim() == 2:
                d = d_in.unsqueeze(1).expand(B, L, 1) if d_in.shape[1] == 1 else d_in.unsqueeze(-1)
            else:
                d = d_in
        else:
            raise TypeError(f"Unsupported dopamine type: {type(dopamine)}")

        # 2. Dynamic effective observation frequency
        # nu_eff = softplus(nu_obs) * clamp(1.0 - tanh(|lambda_D| * d), min=0.01)
        nu_obs_pos = torch.clamp(nn.functional.softplus(self.obs_freq), min=1e-5)
        dopamine_damping = torch.clamp(
            1.0 - torch.tanh(torch.abs(self.dopamine_coupling) * d), min=0.01
        )
        nu_eff = nu_obs_pos * dopamine_damping  # [B, L, 1]

        # 3. Zeno pinning probability
        # P_zeno = exp(-1.0 / nu_eff)
        P_zeno = torch.exp(-1.0 / nu_eff)  # [B, L, 1]

        # 4. Pinned state: working memory baseline h_pinned = Q(x)
        h_pinned = self.q_proj(x_3d)  # [B, L, dim]

        # 5. Exploratory candidate state via multi-head attention
        Q = self.q_proj(x_3d)  # [B, L, dim]
        K = self.k_proj(x_3d)  # [B, L, dim]
        V = self.v_proj(x_3d)  # [B, L, dim]

        Q_h = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K_h = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V_h = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V_h)
        h_explore = attn_out.transpose(1, 2).contiguous().view(B, L, self.dim)

        # 6. Phase kickback: phi = pi / nu_eff
        phi = math.pi / nu_eff  # [B, L, 1]
        kickback_term = self.kickback_proj(h_explore)  # [B, L, dim]
        h_tunnel = torch.cos(phi) * h_explore + torch.sin(phi) * kickback_term  # [B, L, dim]

        # 7. Coherent superposition
        # h_out = P_zeno * h_pinned + (1.0 - P_zeno) * h_tunnel
        h_out = P_zeno * h_pinned + (1.0 - P_zeno) * h_tunnel  # [B, L, dim]

        res_out: torch.Tensor
        res_P_zeno: torch.Tensor
        res_nu_eff: torch.Tensor
        res_phi: torch.Tensor
        res_d: torch.Tensor
        res_pinned: torch.Tensor
        res_tunnel: torch.Tensor
        res_explore: torch.Tensor

        # Restore original dimension
        if orig_dim == 1:
            res_out = h_out.squeeze(0).squeeze(0)
            res_P_zeno = P_zeno.squeeze(0).squeeze(0)
            res_nu_eff = nu_eff.squeeze(0).squeeze(0)
            res_phi = phi.squeeze(0).squeeze(0)
            res_d = d.squeeze(0).squeeze(0)
            res_pinned = h_pinned.squeeze(0).squeeze(0)
            res_tunnel = h_tunnel.squeeze(0).squeeze(0)
            res_explore = h_explore.squeeze(0).squeeze(0)
        elif orig_dim == 2:
            res_out = h_out.squeeze(1)
            res_P_zeno = P_zeno.squeeze(1)
            res_nu_eff = nu_eff.squeeze(1)
            res_phi = phi.squeeze(1)
            res_d = d.squeeze(1)
            res_pinned = h_pinned.squeeze(1)
            res_tunnel = h_tunnel.squeeze(1)
            res_explore = h_explore.squeeze(1)
        else:
            res_out = h_out
            res_P_zeno = P_zeno
            res_nu_eff = nu_eff
            res_phi = phi
            res_d = d
            res_pinned = h_pinned
            res_tunnel = h_tunnel
            res_explore = h_explore

        if return_diagnostics:
            return {
                "output": res_out,
                "P_zeno": res_P_zeno,
                "nu_eff": res_nu_eff,
                "phi": res_phi,
                "dopamine": res_d,
                "h_pinned": res_pinned,
                "h_tunnel": res_tunnel,
                "h_explore": res_explore,
            }

        return res_out

    def __repr__(self) -> str:
        return (
            f"QuantumZenoAttention(dim={self.dim}, "
            f"obs_freq={float(self.obs_freq.detach()):.2f}, "
            f"dopamine_coupling={float(self.dopamine_coupling.detach()):.2f}, "
            f"num_heads={self.num_heads})"
        )

