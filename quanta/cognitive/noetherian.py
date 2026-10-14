"""Noetherian Invariant Cognitive Arbiter (Mizan Hakemi) Core Engine.

This module formalizes continuous variational mechanics and information-theoretic
conservation laws for AI cognitive decision trajectories in Quanta SDK.

Theoretical Pillars:
1. Hamilton's Principle of Least Action (Discrete Action Integral):
   S_d = sum_{k=1}^T (T_k - V_k), balancing semantic kinetic momentum
   T_k = 0.5 * ||Delta q_k||^2 against cognitive potential V_k (goal deviation,
   consequence cost, constraint penalty). Action efficiency:
   A_eff = exp(-mean_cost / sigma_A^2) in [0.0, 1.0].

2. Epistemic Entropy Bound & Dispersion Control:
   Shannon / Von Neumann entropy production Delta S_k = H(p_k) - H(p_{k-1})
   monitoring cognitive divergence, hallucination spurts, and decoherence.
   Stability score E_eff = exp(-kappa * max(0, Delta S - theta)) in [0.0, 1.0].

3. Noether's First Theorem & Semantic Charge Conservation:
   Continuous Lie generator symmetries phi_j in R^d define conserved semantic charges
   Q_j(q) = <phi_j, q>. Trajectory transitions generate discrete Noether flux
   Phi_Noether = sum_j w_j |Delta Q_j|, measuring rule erosion and context leakage.
   Conservation score C_eff = exp(-gamma * Phi_Noether) in [0.0, 1.0].

4. Tikhonov-Regularized Invariant Manifold Projection:
   States drifting from invariant constraints are projected back onto the intersection
   of the affine subspace C q = b and the unit Hilbert sphere S^{d-1}:
   Pi_M(q) = q - C^T (C C^T + lambda I)^{-1} (C q - b) with spherical retraction
   q* = q / ||q||_2. Compliance score P_eff in [0.0, 1.0].

5. Composite Kozmik Mizan Score:
   M_mizan = w_A * A_eff + w_E * E_eff + w_C * C_eff + w_P * P_eff in [0.0, 1.0],
   governing harmonic equilibrium, mild drift, flux alert, and critical breach gating.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "ActionIntegralMetric",
    "ConservedChargeDefinition",
    "EpistemicEntropyMetric",
    "InvariantViolation",
    "ManifoldProjectionResult",
    "MizanHakemi",
    "NoetherianEngineConfig",
    "NoetherianEvaluation",
    "NoetherianInvariantEngine",
    "evaluate_epistemic_entropy",
    "evaluate_noether_charges",
    "evaluate_trajectory_action",
    "evaluate_trajectory_noetherian",
    "project_to_invariant_manifold",
    "project_to_manifold",
]


def _clamp01(val: Any, default: float = 0.0) -> float:
    """Clamps a numeric value safely to [0.0, 1.0], handling NaNs and Infinities."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return max(0.0, min(1.0, f))
    except (ValueError, TypeError):
        return default


def _ensure_real_vector(
    tensor_like: torch.Tensor | Sequence[float] | None,
    dim: int = 64,
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Safely coerces input into a 1D real float32 tensor of specified dimension."""
    if tensor_like is None:
        v = torch.zeros(dim, dtype=torch.float32, device=device)
        v[0] = 1.0
        return v

    if not isinstance(tensor_like, torch.Tensor):
        tensor_like = torch.tensor(tensor_like, dtype=torch.float32, device=device)

    # If complex, project to real part
    if tensor_like.is_complex():
        tensor_like = tensor_like.real

    tensor_like = tensor_like.to(dtype=torch.float32, device=device).flatten()

    # Dimension match: pad or truncate
    if tensor_like.numel() < dim:
        padded = torch.zeros(dim, dtype=torch.float32, device=tensor_like.device)
        padded[: tensor_like.numel()] = tensor_like
        tensor_like = padded
    elif tensor_like.numel() > dim:
        tensor_like = tensor_like[:dim]

    # Clean NaNs and Infs
    tensor_like = torch.nan_to_num(tensor_like, nan=0.0, posinf=1.0, neginf=-1.0)

    if normalize:
        norm = torch.norm(tensor_like, p=2)
        if norm.item() > 1e-12:
            tensor_like = tensor_like / norm
        else:
            tensor_like = torch.zeros(dim, dtype=torch.float32, device=tensor_like.device)
            tensor_like[0] = 1.0

    return tensor_like


@dataclass
class ConservedChargeDefinition:
    """Definition of an immutable semantic invariant rule modeled as a Lie generator.

    Attributes:
        charge_id: Unique string identifier for the conserved charge.
        rule_name: Human-readable constraint or policy title.
        description: Detailed ontological or operational explanation.
        generator_vector: Lie generator vector phi_j in R^d with unit norm ||phi_j|| = 1.
        target_value: Desired scalar charge b_j = <phi_j, q>. Default 0.0.
        weight: Importance weighting w_j >= 0.0 in composite Noether flux.
        tolerance: Permissible drift epsilon before declaring a rule breach.
        is_hard_constraint: If True, exceeding tolerance triggers invalidity.
    """

    charge_id: str
    rule_name: str
    description: str = ""
    generator_vector: torch.Tensor = field(
        default_factory=lambda: torch.zeros(64, dtype=torch.float32)
    )
    target_value: float = 0.0
    weight: float = 1.0
    tolerance: float = 0.05
    is_hard_constraint: bool = True

    def __post_init__(self) -> None:
        """Enforces real unit-norm generator vector."""
        dim = 64 if self.generator_vector.numel() == 0 else self.generator_vector.numel()
        self.generator_vector = _ensure_real_vector(
            self.generator_vector, dim=dim, normalize=True
        )

    @property
    def name(self) -> str:
        """Alias for rule_name."""
        return self.rule_name

    @property
    def target_charge(self) -> float:
        """Alias for target_value."""
        return self.target_value

    def to_dict(self) -> dict[str, Any]:
        """Serializes charge definition to dictionary."""
        return {
            "charge_id": self.charge_id,
            "rule_name": self.rule_name,
            "description": self.description,
            "target_value": round(float(self.target_value), 6),
            "weight": round(float(self.weight), 4),
            "tolerance": round(float(self.tolerance), 6),
            "is_hard_constraint": self.is_hard_constraint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConservedChargeDefinition:
        """Instantiates charge definition from dictionary with safe fallbacks."""
        if not data or not isinstance(data, dict):
            return cls(
                charge_id="default_charge",
                rule_name="Default Invariant",
                generator_vector=torch.zeros(64, dtype=torch.float32),
            )
        gen = data.get("generator_vector")
        if gen is not None:
            if isinstance(gen, list):
                gen_tensor = torch.tensor(gen, dtype=torch.float32)
            elif isinstance(gen, torch.Tensor):
                gen_tensor = gen
            else:
                gen_tensor = torch.zeros(64, dtype=torch.float32)
        else:
            gen_tensor = torch.zeros(64, dtype=torch.float32)

        return cls(
            charge_id=str(data.get("charge_id", "custom_charge")),
            rule_name=str(data.get("rule_name", data.get("name", "Custom Rule"))),
            description=str(data.get("description", "")),
            generator_vector=gen_tensor,
            target_value=float(data.get("target_value", data.get("target_charge", 0.0))),
            weight=float(data.get("weight", 1.0)),
            tolerance=float(data.get("tolerance", 0.05)),
            is_hard_constraint=bool(data.get("is_hard_constraint", True)),
        )

    @classmethod
    def from_text(
        cls,
        charge_id: str,
        rule_name: str,
        description: str,
        dim: int = 64,
        target_value: float = 0.0,
        weight: float = 1.0,
        tolerance: float = 0.05,
        is_hard_constraint: bool = True,
    ) -> ConservedChargeDefinition:
        """Deterministically generates a pseudo-orthogonal unit generator from text."""
        seed_bytes = hashlib.sha256(f"{charge_id}:{rule_name}:{description}".encode()).digest()
        # Generate reproducible pseudo-random vector from SHA-256 seed
        generator_list = []
        for i in range(dim):
            byte_val = seed_bytes[i % len(seed_bytes)]
            val = ((byte_val / 255.0) * 2.0) - 1.0
            generator_list.append(val)

        tensor_v = torch.tensor(generator_list, dtype=torch.float32)
        tensor_v = _ensure_real_vector(tensor_v, dim=dim, normalize=True)

        return cls(
            charge_id=charge_id,
            rule_name=rule_name,
            description=description,
            generator_vector=tensor_v,
            target_value=target_value,
            weight=weight,
            tolerance=tolerance,
            is_hard_constraint=is_hard_constraint,
        )


class InvariantViolation(str):
    """Immutable string subclass carrying strongly-typed invariant breach provenance.

    Provides 100% backward compatibility with list[str] consumers, JSON serialization,
    and legacy test assertions, while guaranteeing zero substring collision during
    hard-constraint validity arbitration.
    """

    charge_id: str
    rule_name: str
    step: int
    observed_drift: float
    tolerance: float
    is_hard_constraint: bool

    def __new__(
        cls,
        message: str,
        *,
        charge_id: str = "",
        rule_name: str = "",
        step: int = 0,
        observed_drift: float = 0.0,
        tolerance: float = 0.0,
        is_hard_constraint: bool = False,
    ) -> InvariantViolation:
        instance = super().__new__(cls, message)
        instance.charge_id = str(charge_id)
        instance.rule_name = str(rule_name)
        instance.step = int(step)
        instance.observed_drift = float(observed_drift)
        instance.tolerance = float(tolerance)
        instance.is_hard_constraint = bool(is_hard_constraint)
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Serializes violation metadata to structured dictionary."""
        return {
            "message": str(self),
            "charge_id": self.charge_id,
            "rule_name": self.rule_name,
            "step": self.step,
            "observed_drift": round(self.observed_drift, 6),
            "tolerance": round(self.tolerance, 6),
            "is_hard_constraint": self.is_hard_constraint,
        }


@dataclass
class ActionIntegralMetric:
    """Action integral diagnostics along a decision trajectory.

    Attributes:
        kinetic_energy_sum: Cumulative discrete kinetic energy T = 0.5 * sum ||Delta q_k||^2.
        potential_energy_sum: Cumulative discrete potential energy V.
        discrete_action: Discrete Hamilton action S_d = sum (T_k - V_k).
        action_efficiency: Action efficiency score A_eff in [0.0, 1.0].
    """

    kinetic_energy_sum: float = 0.0
    potential_energy_sum: float = 0.0
    discrete_action: float = 0.0
    action_efficiency: float = 1.0

    @property
    def kinetic_energy(self) -> float:
        """Alias for kinetic_energy_sum."""
        return self.kinetic_energy_sum

    @property
    def potential_energy(self) -> float:
        """Alias for potential_energy_sum."""
        return self.potential_energy_sum

    @property
    def lagrangian(self) -> float:
        """Step or trajectory Lagrangian L = T - V."""
        return self.kinetic_energy_sum - self.potential_energy_sum

    @property
    def action(self) -> float:
        """Alias for discrete_action."""
        return self.discrete_action

    def to_dict(self) -> dict[str, float]:
        """Serializes action metric to dictionary."""
        return {
            "kinetic_energy_sum": round(self.kinetic_energy_sum, 6),
            "potential_energy_sum": round(self.potential_energy_sum, 6),
            "discrete_action": round(self.discrete_action, 6),
            "action_efficiency": round(self.action_efficiency, 6),
            "lagrangian": round(self.lagrangian, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ActionIntegralMetric:
        """Instantiates action metric from dictionary."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            kinetic_energy_sum=float(data.get("kinetic_energy_sum", data.get("kinetic_energy", 0.0))),
            potential_energy_sum=float(data.get("potential_energy_sum", data.get("potential_energy", 0.0))),
            discrete_action=float(data.get("discrete_action", data.get("action", 0.0))),
            action_efficiency=_clamp01(data.get("action_efficiency", 1.0), 1.0),
        )


@dataclass
class EpistemicEntropyMetric:
    """Epistemic entropy production diagnostics across trajectory steps.

    Attributes:
        initial_entropy: Normalized Shannon entropy H(p_0) in [0.0, 1.0].
        final_entropy: Normalized Shannon entropy H(p_T) in [0.0, 1.0].
        max_entropy_spurt: Peak positive entropy jump max_k (H_k - H_{k-1}).
        cumulative_entropy_flux: Mean magnitude of step entropy fluctuations.
        entropy_stability_score: Stability factor E_eff in [0.0, 1.0].
        is_decoherent: True if max_entropy_spurt exceeds safety threshold.
    """

    initial_entropy: float = 0.0
    final_entropy: float = 0.0
    max_entropy_spurt: float = 0.0
    cumulative_entropy_flux: float = 0.0
    entropy_stability_score: float = 1.0
    is_decoherent: bool = False

    @property
    def entropy_production(self) -> float:
        """Net entropy production Delta S = H_final - H_initial."""
        return self.final_entropy - self.initial_entropy

    @property
    def dispersion_rate(self) -> float:
        """Alias for max_entropy_spurt."""
        return self.max_entropy_spurt

    @property
    def stability_score(self) -> float:
        """Alias for entropy_stability_score."""
        return self.entropy_stability_score

    @property
    def entropy_stability(self) -> float:
        """Alias for entropy_stability_score."""
        return self.entropy_stability_score

    def to_dict(self) -> dict[str, Any]:
        """Serializes entropy metric to dictionary."""
        return {
            "initial_entropy": round(self.initial_entropy, 6),
            "final_entropy": round(self.final_entropy, 6),
            "max_entropy_spurt": round(self.max_entropy_spurt, 6),
            "cumulative_entropy_flux": round(self.cumulative_entropy_flux, 6),
            "entropy_stability_score": round(self.entropy_stability_score, 6),
            "entropy_production": round(self.entropy_production, 6),
            "is_decoherent": self.is_decoherent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EpistemicEntropyMetric:
        """Instantiates entropy metric from dictionary."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            initial_entropy=float(data.get("initial_entropy", 0.0)),
            final_entropy=float(data.get("final_entropy", 0.0)),
            max_entropy_spurt=float(data.get("max_entropy_spurt", 0.0)),
            cumulative_entropy_flux=float(data.get("cumulative_entropy_flux", 0.0)),
            entropy_stability_score=_clamp01(
                data.get("entropy_stability_score", data.get("stability_score", 1.0)), 1.0
            ),
            is_decoherent=bool(data.get("is_decoherent", False)),
        )


@dataclass
class ManifoldProjectionResult:
    """Outcome of projecting a statevector onto the invariant manifold.

    Attributes:
        drift_distance_before: Euclidean distance ||C q - b||_2 prior to projection.
        drift_distance_after: Residual distance ||C q* - b||_2 following projection.
        projection_applied: True if projection adjustment was performed.
        iterations_run: Number of regularized Gram iterations executed.
        projected_state: Projected statevector q* in R^d on unit sphere.
        charge_restoration_error: Norm of constraint violation after projection.
        compliance_score: Manifold compliance score P_eff in [0.0, 1.0].
    """

    drift_distance_before: float = 0.0
    drift_distance_after: float = 0.0
    projection_applied: bool = False
    iterations_run: int = 0
    projected_state: torch.Tensor | None = None
    charge_restoration_error: float = 0.0
    compliance_score: float = 1.0

    @property
    def projected_vector(self) -> torch.Tensor | None:
        """Alias for projected_state."""
        return self.projected_state

    @property
    def residual_error(self) -> float:
        """Alias for charge_restoration_error."""
        return self.charge_restoration_error

    @property
    def projection_distance(self) -> float:
        """Alias for drift_distance_before."""
        return self.drift_distance_before

    def to_dict(self) -> dict[str, Any]:
        """Serializes projection result to dictionary."""
        return {
            "drift_distance_before": round(self.drift_distance_before, 6),
            "drift_distance_after": round(self.drift_distance_after, 6),
            "projection_applied": self.projection_applied,
            "iterations_run": self.iterations_run,
            "charge_restoration_error": round(self.charge_restoration_error, 6),
            "compliance_score": round(self.compliance_score, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ManifoldProjectionResult:
        """Instantiates projection result from dictionary."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            drift_distance_before=float(data.get("drift_distance_before", 0.0)),
            drift_distance_after=float(data.get("drift_distance_after", 0.0)),
            projection_applied=bool(data.get("projection_applied", False)),
            iterations_run=int(data.get("iterations_run", 0)),
            charge_restoration_error=float(data.get("charge_restoration_error", 0.0)),
            compliance_score=_clamp01(data.get("compliance_score", 1.0), 1.0),
        )


@dataclass
class NoetherianEvaluation:
    """Comprehensive evaluation record produced by the Noetherian Invariant Engine.

    Attributes:
        mizan_score: Composite Kozmik Mizan score M_mizan in [0.0, 1.0].
        action_efficiency: Variational action efficiency A_eff in [0.0, 1.0].
        entropy_stability: Information stability score E_eff in [0.0, 1.0].
        noether_flux: Total divergence Phi_Noether >= 0.0 across the trajectory.
        charge_conservation: Semantic charge preservation score C_eff in [0.0, 1.0].
        charge_drift: Maximum Euclidean distance in charge space ||Q(q_k) - b||_2.
        projection_compliance: Manifold compliance factor P_eff in [0.0, 1.0].
        action_metric: Detailed ActionIntegralMetric.
        entropy_metric: Detailed EpistemicEntropyMetric.
        projection_result: Detailed ManifoldProjectionResult.
        invariant_violations: List of specific invariant constraint breach descriptions.
        is_valid: True if M_mizan >= breach_threshold and no critical constraints breached.
        regime: Invariant regime classification string.
    """

    mizan_score: float = 1.0
    action_efficiency: float = 1.0
    entropy_stability: float = 1.0
    noether_flux: float = 0.0
    charge_conservation: float = 1.0
    charge_drift: float = 0.0
    projection_compliance: float = 1.0
    action_metric: ActionIntegralMetric = field(default_factory=ActionIntegralMetric)
    entropy_metric: EpistemicEntropyMetric = field(default_factory=EpistemicEntropyMetric)
    projection_result: ManifoldProjectionResult = field(default_factory=ManifoldProjectionResult)
    invariant_violations: list[str] = field(default_factory=list)
    is_valid: bool = True
    regime: str = "HARMONIC_EQUILIBRIUM"

    @property
    def violation_details(self) -> list[str]:
        """Alias for invariant_violations."""
        return self.invariant_violations

    def to_dict(self) -> dict[str, Any]:
        """Serializes Noetherian evaluation to dictionary."""
        return {
            "mizan_score": round(self.mizan_score, 6),
            "action_efficiency": round(self.action_efficiency, 6),
            "entropy_stability": round(self.entropy_stability, 6),
            "noether_flux": round(self.noether_flux, 6),
            "charge_conservation": round(self.charge_conservation, 6),
            "charge_drift": round(self.charge_drift, 6),
            "projection_compliance": round(self.projection_compliance, 6),
            "action": self.action_metric.to_dict(),
            "entropy": self.entropy_metric.to_dict(),
            "projection": self.projection_result.to_dict(),
            "invariant_violations": list(self.invariant_violations),
            "is_valid": self.is_valid,
            "regime": self.regime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NoetherianEvaluation:
        """Instantiates NoetherianEvaluation from dictionary."""
        if not data or not isinstance(data, dict):
            return cls()

        action_raw = data.get("action")
        entropy_raw = data.get("entropy")
        projection_raw = data.get("projection")

        return cls(
            mizan_score=_clamp01(data.get("mizan_score", 1.0), 1.0),
            action_efficiency=_clamp01(data.get("action_efficiency", 1.0), 1.0),
            entropy_stability=_clamp01(data.get("entropy_stability", 1.0), 1.0),
            noether_flux=max(0.0, float(data.get("noether_flux", 0.0))),
            charge_conservation=_clamp01(data.get("charge_conservation", 1.0), 1.0),
            charge_drift=max(0.0, float(data.get("charge_drift", 0.0))),
            projection_compliance=_clamp01(data.get("projection_compliance", 1.0), 1.0),
            action_metric=ActionIntegralMetric.from_dict(action_raw),
            entropy_metric=EpistemicEntropyMetric.from_dict(entropy_raw),
            projection_result=ManifoldProjectionResult.from_dict(projection_raw),
            invariant_violations=list(data.get("invariant_violations", [])),
            is_valid=bool(data.get("is_valid", True)),
            regime=str(data.get("regime", "HARMONIC_EQUILIBRIUM")),
        )


@dataclass
class NoetherianEngineConfig:
    """Hyperparameters and configuration for the Noetherian Invariant Engine.

    Attributes:
        dim: Dimensionality of cognitive Hilbert space (default: 64).
        weight_action: Importance weight w_A for Action Integral efficiency.
        weight_entropy: Importance weight w_E for Epistemic Entropy stability.
        weight_charge: Importance weight w_C for Noether Charge conservation.
        weight_manifold: Importance weight w_P for Invariant Manifold compliance.
        action_scale: Scaling parameter sigma_A for Lagrangian cost exponent.
        charge_sigma: Scaling parameter sigma_Q for Noether drift exponent.
        charge_gamma: Sensitivity multiplier gamma for Noether flux penalty.
        max_allowed_entropy_flux: Allowed epistemic entropy spurt theta_allowed.
        entropy_kappa: Sensitivity multiplier kappa for entropy spurt penalty.
        drift_tolerance: Manifold drift threshold epsilon_tol before projection.
        tikhonov_reg: Tikhonov regularization parameter lambda for Gram inversion.
        max_projection_iters: Iteration count for manifold geodesic retraction.
        critical_breach_threshold: Lower threshold for validity and pruning.
        flux_alert_threshold: Threshold below which flux warnings are emitted.
        mild_drift_threshold: Threshold for mild drift regime.
        harmonic_threshold: Threshold for harmonic equilibrium regime.
        default_charges: Initial list of ConservedChargeDefinition instances.
    """

    dim: int = 64
    weight_action: float = 0.25
    weight_entropy: float = 0.20
    weight_charge: float = 0.35
    weight_manifold: float = 0.20
    action_scale: float = 1.0
    charge_sigma: float = 0.25
    charge_gamma: float = 2.0
    max_allowed_entropy_flux: float = 0.15
    entropy_kappa: float = 3.0
    drift_tolerance: float = 0.02
    tikhonov_reg: float = 1e-6
    max_projection_iters: int = 3
    critical_breach_threshold: float = 0.40
    flux_alert_threshold: float = 0.40
    mild_drift_threshold: float = 0.65
    harmonic_threshold: float = 0.85
    default_charges: list[ConservedChargeDefinition] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validates and normalizes weights."""
        total_w = (
            self.weight_action
            + self.weight_entropy
            + self.weight_charge
            + self.weight_manifold
        )
        if abs(total_w - 1.0) > 1e-6 and total_w > 0:
            self.weight_action /= total_w
            self.weight_entropy /= total_w
            self.weight_charge /= total_w
            self.weight_manifold /= total_w

    @property
    def w_A(self) -> float:
        """Alias for weight_action."""
        return self.weight_action

    @property
    def w_E(self) -> float:
        """Alias for weight_entropy."""
        return self.weight_entropy

    @property
    def w_C(self) -> float:
        """Alias for weight_charge."""
        return self.weight_charge

    @property
    def w_P(self) -> float:
        """Alias for weight_manifold."""
        return self.weight_manifold

    @property
    def epsilon(self) -> float:
        """Alias for drift_tolerance."""
        return self.drift_tolerance

    @property
    def beta(self) -> float:
        """Inverse temperature / scaling for action efficiency."""
        return 1.0 / max(1e-6, self.action_scale**2)

    @property
    def hard_constraint_names(self) -> set[str]:
        """Set of rule names for hard invariant constraints."""
        return {c.rule_name for c in self.default_charges if c.is_hard_constraint and c.rule_name}

    @property
    def hard_charge_ids(self) -> set[str]:
        """Set of charge IDs for hard invariant constraints."""
        return {c.charge_id for c in self.default_charges if c.is_hard_constraint and c.charge_id}

    def to_dict(self) -> dict[str, Any]:
        """Serializes configuration to dictionary."""
        return {
            "dim": self.dim,
            "weight_action": round(self.weight_action, 4),
            "weight_entropy": round(self.weight_entropy, 4),
            "weight_charge": round(self.weight_charge, 4),
            "weight_manifold": round(self.weight_manifold, 4),
            "action_scale": self.action_scale,
            "charge_sigma": self.charge_sigma,
            "charge_gamma": self.charge_gamma,
            "max_allowed_entropy_flux": self.max_allowed_entropy_flux,
            "entropy_kappa": self.entropy_kappa,
            "drift_tolerance": self.drift_tolerance,
            "tikhonov_reg": self.tikhonov_reg,
            "max_projection_iters": self.max_projection_iters,
            "critical_breach_threshold": self.critical_breach_threshold,
            "flux_alert_threshold": self.flux_alert_threshold,
            "mild_drift_threshold": self.mild_drift_threshold,
            "harmonic_threshold": self.harmonic_threshold,
            "num_default_charges": len(self.default_charges),
        }


# ============================================================================
# Core Mathematical Algorithms (Pure Functions)
# ============================================================================


def evaluate_trajectory_action(
    path_states: Sequence[torch.Tensor],
    goal_state: torch.Tensor | None = None,
    consequence_costs: Sequence[float] | float | None = None,
    config: NoetherianEngineConfig | None = None,
) -> ActionIntegralMetric:
    """Computes the discrete Hamilton Action Integral S_d along a decision trajectory.

    Discrete Formulation:
        v_k = (q_k - q_{k-1}) / Delta t_k (with Delta t = 1.0)
        T_k = 0.5 * ||v_k||_2^2
        V_k = w_goal * V_goal(q_k) + w_panel * V_consequence(q_k)
        S_d = sum_{k=1}^D (T_k - V_k)
        A_eff = exp( - sum(T_k + V_k) / (D * sigma_A^2) ) in [0.0, 1.0]

    Args:
        path_states: Sequence of statevectors q_0, ..., q_D in R^d.
        goal_state: Optional target statevector q_goal.
        consequence_costs: Optional sequence of consequence penalties. Supports:
            - Transition-indexed sequence of length T (index k-1 maps to transition k)
            - State-indexed sequence of length T+1 (index k maps to arrival state q_k)
            - Scalar float or length-1 sequence broadcast uniformly across all transitions
        config: NoetherianEngineConfig for scaling parameters.

    Returns:
        ActionIntegralMetric containing energy sums, action, and normalized efficiency.
    """
    if config is None:
        config = NoetherianEngineConfig()

    dim = config.dim
    D = len(path_states) - 1

    if D <= 0:
        return ActionIntegralMetric(
            kinetic_energy_sum=0.0,
            potential_energy_sum=0.0,
            discrete_action=0.0,
            action_efficiency=1.0,
        )

    # Sanitize goal state
    goal_v = _ensure_real_vector(goal_state, dim=dim, normalize=True) if goal_state is not None else None

    kinetic_sum = 0.0
    potential_sum = 0.0

    sanitized_states = [_ensure_real_vector(q, dim=dim, normalize=True) for q in path_states]

    if isinstance(consequence_costs, (int, float)):
        consequence_costs = [float(consequence_costs)]

    for k in range(1, len(sanitized_states)):
        q_prev = sanitized_states[k - 1]
        q_curr = sanitized_states[k]

        # 1. Kinetic energy T_k = 0.5 * ||q_curr - q_prev||^2
        diff = q_curr - q_prev
        t_k = 0.5 * torch.sum(diff * diff).item()
        kinetic_sum += t_k

        # 2. Potential energy V_k = V_goal + V_consequence
        v_k_total = 0.0
        if goal_v is not None:
            cos_sim = torch.dot(q_curr, goal_v).item()
            cos_sim = max(-1.0, min(1.0, cos_sim))
            v_goal = 1.0 - cos_sim
            v_k_total += 0.5 * v_goal

        if consequence_costs is not None and len(consequence_costs) > 0:
            if len(consequence_costs) >= len(sanitized_states):
                # State-indexed: aligned with path_states [q_0, ..., q_T]
                idx = k
            else:
                # Transition-indexed or scalar broadcast across all transitions
                idx = 0 if len(consequence_costs) == 1 else k - 1

            if 0 <= idx < len(consequence_costs):
                v_panel = _clamp01(consequence_costs[idx], 0.0)
                v_k_total += 0.5 * v_panel

        potential_sum += v_k_total

    # Discrete action: S_d = sum(T_k - V_k)
    s_d = kinetic_sum - potential_sum

    # Action efficiency: exp(-mean_cost / sigma_A^2)
    mean_cost = (kinetic_sum + potential_sum) / max(1, D)
    scale_sq = max(1e-6, config.action_scale**2)
    eff = math.exp(-mean_cost / scale_sq)
    eff = max(0.0, min(1.0, eff))

    return ActionIntegralMetric(
        kinetic_energy_sum=kinetic_sum,
        potential_energy_sum=potential_sum,
        discrete_action=s_d,
        action_efficiency=eff,
    )


def evaluate_epistemic_entropy(
    probs_sequence: Sequence[Sequence[float] | torch.Tensor | float] | None,
    config: NoetherianEngineConfig | None = None,
) -> EpistemicEntropyMetric:
    """Monitors epistemic entropy flux and detects cognitive decoherence.

    Formulation:
        For each step distribution p_k with N >= 2:
            H(p_k) = - sum_{i=1}^N p_{k,i} ln(p_{k,i} + eps)
            bar{H}(p_k) = H(p_k) / ln(max(2, N)) in [0.0, 1.0]
        Delta S_k = bar{H}(p_k) - bar{H}(p_{k-1})
        E_eff = exp( - kappa * max(0, max_spurt - theta_allowed) )

    Args:
        probs_sequence: Sequence of probability distributions or scalar confidence values.
        config: NoetherianEngineConfig for thresholds and multipliers.

    Returns:
        EpistemicEntropyMetric containing entropy values, spurt rate, and stability score.
    """
    if config is None:
        config = NoetherianEngineConfig()

    if not probs_sequence or len(probs_sequence) == 0:
        return EpistemicEntropyMetric(
            initial_entropy=0.0,
            final_entropy=0.0,
            max_entropy_spurt=0.0,
            cumulative_entropy_flux=0.0,
            entropy_stability_score=1.0,
            is_decoherent=False,
        )

    entropies: list[float] = []

    for item in probs_sequence:
        if isinstance(item, (int, float)):
            # Scalar probability: binary entropy
            p = max(1e-6, min(1.0 - 1e-6, float(item)))
            h = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
            norm_h = h / math.log(2.0)
            entropies.append(max(0.0, min(1.0, norm_h)))
        else:
            if isinstance(item, torch.Tensor):
                p_t = item.detach().cpu().flatten().to(torch.float32)
            else:
                p_t = torch.tensor(list(item), dtype=torch.float32)

            p_t = torch.nan_to_num(p_t, nan=0.0, posinf=0.0, neginf=0.0)
            sum_p = torch.sum(p_t).item()
            p_t = p_t / sum_p if sum_p > 1e-12 else torch.ones_like(p_t) / max(1, p_t.numel())

            n = max(2, p_t.numel())
            # Shannon entropy
            h = -torch.sum(p_t * torch.log(p_t + 1e-12)).item()
            norm_h = h / math.log(float(n))
            entropies.append(max(0.0, min(1.0, norm_h)))

    if len(entropies) == 1:
        return EpistemicEntropyMetric(
            initial_entropy=entropies[0],
            final_entropy=entropies[0],
            max_entropy_spurt=0.0,
            cumulative_entropy_flux=0.0,
            entropy_stability_score=1.0,
            is_decoherent=False,
        )

    fluxes: list[float] = []
    max_spurt = 0.0

    for k in range(1, len(entropies)):
        delta = entropies[k] - entropies[k - 1]
        fluxes.append(abs(delta))
        if delta > max_spurt:
            max_spurt = delta

    cum_flux = sum(fluxes) / max(1, len(fluxes))
    spurt_excess = max(0.0, max_spurt - config.max_allowed_entropy_flux)
    stability = math.exp(-config.entropy_kappa * spurt_excess)
    stability = max(0.0, min(1.0, stability))

    is_decoherent = max_spurt > (config.max_allowed_entropy_flux * 2.0)

    return EpistemicEntropyMetric(
        initial_entropy=entropies[0],
        final_entropy=entropies[-1],
        max_entropy_spurt=max_spurt,
        cumulative_entropy_flux=cum_flux,
        entropy_stability_score=stability,
        is_decoherent=is_decoherent,
    )


def evaluate_noether_charges(
    path_states: Sequence[torch.Tensor],
    charges: Sequence[ConservedChargeDefinition] | None = None,
    initial_target_charges: torch.Tensor | Sequence[float] | None = None,
    config: NoetherianEngineConfig | None = None,
) -> tuple[float, float, float, list[str]]:
    """Evaluates semantic charge conservation, discrete Noether flux, and rule breaches.

    Formulation:
        For charge definitions with generators phi_j:
            Q_j(q) = <phi_j, q>
            Discrete Noether Flux: Phi_Noether = sum_{k=1}^D || W (Q(q_k) - Q(q_{k-1})) ||_1
            Endpoint Drift: Delta Q_max = max_k || Q(q_k) - b ||_2
            Conservation score: C_eff = exp( - gamma * (Phi_Noether / D + Delta Q_max) )

    Args:
        path_states: Sequence of statevectors q_0, ..., q_D in R^d.
        charges: List of ConservedChargeDefinition instances.
        initial_target_charges: Optional override for target values b in R^m.
        config: NoetherianEngineConfig for parameters.

    Returns:
        Tuple of (noether_flux, max_charge_drift, charge_conservation_score, violations).
    """
    if config is None:
        config = NoetherianEngineConfig()

    if charges is None:
        charges = config.default_charges

    if not charges or len(path_states) < 1:
        return 0.0, 0.0, 1.0, []

    dim = config.dim
    m = len(charges)
    device = path_states[0].device if isinstance(path_states[0], torch.Tensor) else None

    # Stack generator matrix C: shape [m, dim]
    C_rows = [_ensure_real_vector(c.generator_vector, dim=dim, normalize=True, device=device) for c in charges]
    C = torch.stack(C_rows, dim=0)

    weights = torch.tensor([max(0.0, float(c.weight)) for c in charges], dtype=torch.float32, device=device)
    tolerances = [max(1e-6, float(c.tolerance)) for c in charges]

    # Target vector b in R^m
    if initial_target_charges is not None:
        if isinstance(initial_target_charges, torch.Tensor):
            target_b = initial_target_charges.detach().to(dtype=torch.float32, device=device).flatten()[:m]
        else:
            target_b = torch.tensor(list(initial_target_charges)[:m], dtype=torch.float32, device=device)
        if target_b.numel() < m:
            padded_b = torch.zeros(m, dtype=torch.float32, device=device)
            padded_b[: target_b.numel()] = target_b
            target_b = padded_b
    else:
        # Default target vector from charges definitions
        target_b = torch.tensor([float(c.target_value) for c in charges], dtype=torch.float32, device=device)

    sanitized_states = [_ensure_real_vector(q, dim=dim, normalize=True, device=device) for q in path_states]

    total_flux = 0.0
    violations: list[InvariantViolation] = []

    def _record_breach(step: int, idx: int, charge: ConservedChargeDefinition, drift_err: float) -> None:
        msg = (
            f"Charge '{charge.rule_name}' breached at step {step}: "
            f"drift={drift_err:.4f} > tol={tolerances[idx]:.4f}"
        )
        if not any(v == msg for v in violations):
            violations.append(
                InvariantViolation(
                    msg,
                    charge_id=charge.charge_id,
                    rule_name=charge.rule_name,
                    step=step,
                    observed_drift=drift_err,
                    tolerance=tolerances[idx],
                    is_hard_constraint=charge.is_hard_constraint,
                )
            )

    # Step 0 evaluation: initialize drift and verify initial state constraints for all N >= 1
    Q_prev = torch.matmul(C, sanitized_states[0])
    max_charge_drift = torch.norm(Q_prev - target_b, p=2).item()

    for idx, c in enumerate(charges):
        err = abs(Q_prev[idx].item() - target_b[idx].item())
        if err > tolerances[idx]:
            _record_breach(step=0, idx=idx, charge=c, drift_err=err)

    D = len(sanitized_states) - 1

    for k in range(1, len(sanitized_states)):
        q_curr = sanitized_states[k]
        Q_curr = torch.matmul(C, q_curr)

        # Step flux: || W (Q_curr - Q_prev) ||_1
        step_diff = torch.abs(Q_curr - Q_prev)
        step_flux = torch.sum(weights * step_diff).item()
        total_flux += step_flux

        # Drift from target b: Delta Q_max = max_{k >= 0} || Q(q_k) - b ||_2
        drift_curr = torch.norm(Q_curr - target_b, p=2).item()
        if drift_curr > max_charge_drift:
            max_charge_drift = drift_curr

        # Check tolerance violations for each charge at step k
        for idx, c in enumerate(charges):
            err = abs(Q_curr[idx].item() - target_b[idx].item())
            if err > tolerances[idx]:
                _record_breach(step=k, idx=idx, charge=c, drift_err=err)

        Q_prev = Q_curr

    # Normalized conservation score: C_eff = exp(-gamma * (mean_flux + max_drift))
    mean_flux = total_flux / max(1, D)
    penalty = mean_flux + max_charge_drift
    c_eff = math.exp(-config.charge_gamma * penalty)
    c_eff = max(0.0, min(1.0, c_eff))

    return total_flux, max_charge_drift, c_eff, violations


def project_to_manifold(
    state: torch.Tensor,
    charges: Sequence[ConservedChargeDefinition] | torch.Tensor | None = None,
    target_b: torch.Tensor | Sequence[float] | None = None,
    config: NoetherianEngineConfig | None = None,
) -> tuple[torch.Tensor, ManifoldProjectionResult]:
    """Projects a statevector onto the invariant manifold via regularized Gram inversion.

    Algorithm:
        Let C in R^{m x d} be the stacked generator matrix and b in R^m the target charges.
        1. Gram matrix G = C C^T + lambda I_m is symmetric positive definite (SPD).
        2. Closed-form affine correction:
           r = C q - b
           delta = G^{-1} r = solve(C C^T + lambda I, r)
           q_affine = q - C^T delta
        3. Spherical geodesic retraction:
           q* = q_affine / ||q_affine||_2
        4. Repeat for max_projection_iters (default 3) to tighten non-linear intersection.

    Args:
        state: Input cognitive statevector in R^d or C^d.
        charges: List of ConservedChargeDefinition or pre-stacked C matrix [m, dim].
        target_b: Target charge vector b in R^m.
        config: NoetherianEngineConfig for regularization and iteration settings.

    Returns:
        Tuple of (projected_state, ManifoldProjectionResult).
    """
    if config is None:
        config = NoetherianEngineConfig()

    dim = config.dim
    device = state.device if isinstance(state, torch.Tensor) else None
    q = _ensure_real_vector(state, dim=dim, normalize=True, device=device)

    # Handle C matrix and target vector
    if charges is None:
        charges = config.default_charges

    if isinstance(charges, torch.Tensor):
        C = charges.to(dtype=torch.float32, device=q.device)
        m = C.shape[0]
    elif isinstance(charges, Sequence) and len(charges) > 0:
        C_rows = [_ensure_real_vector(c.generator_vector, dim=dim, normalize=True, device=q.device) for c in charges]
        C = torch.stack(C_rows, dim=0)
        m = len(charges)
    else:
        # No charges defined: projection is standard spherical retraction
        return q, ManifoldProjectionResult(
            drift_distance_before=0.0,
            drift_distance_after=0.0,
            projection_applied=False,
            iterations_run=0,
            projected_state=q,
            charge_restoration_error=0.0,
            compliance_score=1.0,
        )

    # Determine target vector b
    if target_b is not None:
        if isinstance(target_b, torch.Tensor):
            b = target_b.detach().to(dtype=torch.float32, device=q.device).flatten()[:m]
        else:
            b = torch.tensor(list(target_b)[:m], dtype=torch.float32, device=q.device)
        if b.numel() < m:
            padded_b = torch.zeros(m, dtype=torch.float32, device=q.device)
            padded_b[: b.numel()] = b
            b = padded_b
    elif isinstance(charges, Sequence) and len(charges) > 0 and isinstance(charges[0], ConservedChargeDefinition):
        b = torch.tensor([float(c.target_value) for c in charges], dtype=torch.float32, device=q.device)
    else:
        b = torch.zeros(m, dtype=torch.float32, device=q.device)

    # Initial drift
    initial_drift = torch.norm(torch.matmul(C, q) - b, p=2).item()

    if initial_drift <= config.drift_tolerance:
        comp_score = math.exp(-initial_drift / max(1e-6, config.charge_sigma))
        return q, ManifoldProjectionResult(
            drift_distance_before=initial_drift,
            drift_distance_after=initial_drift,
            projection_applied=False,
            iterations_run=0,
            projected_state=q,
            charge_restoration_error=initial_drift,
            compliance_score=max(0.0, min(1.0, comp_score)),
        )

    # Tikhonov-regularized Gram matrix: G = C C^T + lambda I_m
    gram = torch.matmul(C, C.T) + config.tikhonov_reg * torch.eye(m, dtype=torch.float32, device=q.device)

    q_curr = q.clone()
    iters_run = 0

    for _ in range(config.max_projection_iters):
        iters_run += 1
        residual = torch.matmul(C, q_curr) - b
        # Solve G * delta = residual via Cholesky / LU
        try:
            delta = torch.linalg.solve(gram, residual)
        except RuntimeError:
            # Fallback to pseudoinverse if singular
            delta = torch.matmul(torch.linalg.pinv(gram), residual)

        # Affine correction
        q_affine = q_curr - torch.matmul(C.T, delta)

        # Spherical geodesic retraction
        norm_affine = torch.norm(q_affine, p=2)
        q_curr = q_affine / norm_affine if norm_affine.item() > 1e-12 else q_affine

        # Early break if converged within tolerance
        curr_res = torch.norm(torch.matmul(C, q_curr) - b, p=2).item()
        if curr_res <= config.drift_tolerance:
            break

    final_drift = torch.norm(torch.matmul(C, q_curr) - b, p=2).item()
    comp_score = math.exp(-initial_drift / max(1e-6, config.charge_sigma))
    comp_score = max(0.0, min(1.0, comp_score))

    return q_curr, ManifoldProjectionResult(
        drift_distance_before=initial_drift,
        drift_distance_after=final_drift,
        projection_applied=True,
        iterations_run=iters_run,
        projected_state=q_curr,
        charge_restoration_error=final_drift,
        compliance_score=comp_score,
    )


# Alias for interface compatibility
project_to_invariant_manifold = project_to_manifold


def evaluate_trajectory_noetherian(
    states: Sequence[torch.Tensor],
    probs: Sequence[Sequence[float] | torch.Tensor | float] | None = None,
    config: NoetherianEngineConfig | None = None,
    charges: Sequence[ConservedChargeDefinition] | None = None,
    goal_state: torch.Tensor | None = None,
    consequence_costs: Sequence[float] | float | None = None,
) -> NoetherianEvaluation:
    """Comprehensive evaluation of a decision trajectory across all 4 Noetherian pillars.

    Composite Kozmik Mizan Score:
        M_mizan = w_A * A_eff + w_E * E_eff + w_C * C_eff + w_P * P_eff in [0.0, 1.0]

    Regimes:
        M_mizan >= 0.85: HARMONIC_EQUILIBRIUM
        0.65 <= M_mizan < 0.85: MILD_DRIFT
        0.40 <= M_mizan < 0.65: FLUX_ALERT
        M_mizan < 0.40: CRITICAL_BREACH

    Args:
        states: Sequence of statevectors along trajectory [q_0, ..., q_T].
        probs: Sequence of branching epistemic distributions or scalar confidences.
        config: NoetherianEngineConfig with weights, thresholds, and parameters.
        charges: Optional list of ConservedChargeDefinition instances.
        goal_state: Optional target statevector for action potential evaluation.
        consequence_costs: Optional sequence of consequence operational costs.

    Returns:
        NoetherianEvaluation populated with all metrics, scores, violations, and regime.
    """
    if config is None:
        config = NoetherianEngineConfig()

    if charges is None:
        charges = config.default_charges

    if not states:
        return NoetherianEvaluation(
            mizan_score=1.0,
            action_efficiency=1.0,
            entropy_stability=1.0,
            noether_flux=0.0,
            charge_conservation=1.0,
            charge_drift=0.0,
            projection_compliance=1.0,
            action_metric=ActionIntegralMetric(),
            entropy_metric=EpistemicEntropyMetric(),
            projection_result=ManifoldProjectionResult(),
            invariant_violations=[],
            is_valid=True,
            regime="HARMONIC_EQUILIBRIUM",
        )

    # 1. Action Integral Pillar
    action_metric = evaluate_trajectory_action(
        path_states=states,
        goal_state=goal_state,
        consequence_costs=consequence_costs,
        config=config,
    )

    # 2. Epistemic Entropy Pillar
    entropy_metric = evaluate_epistemic_entropy(
        probs_sequence=probs,
        config=config,
    )

    # 3. Semantic Charge Conservation Pillar
    noether_flux, max_charge_drift, charge_eff, violations = evaluate_noether_charges(
        path_states=states,
        charges=charges,
        config=config,
    )

    # 4. Invariant Manifold Projection Pillar (evaluated at trajectory terminus)
    terminus_state = states[-1]
    _, proj_result = project_to_manifold(
        state=terminus_state,
        charges=charges,
        config=config,
    )

    # 5. Composite Kozmik Mizan Score
    mizan_score = (
        config.w_A * action_metric.action_efficiency
        + config.w_E * entropy_metric.entropy_stability_score
        + config.w_C * charge_eff
        + config.w_P * proj_result.compliance_score
    )
    mizan_score = max(0.0, min(1.0, float(mizan_score)))

    # Regime Determination
    if mizan_score >= config.harmonic_threshold:
        regime = "HARMONIC_EQUILIBRIUM"
    elif mizan_score >= config.mild_drift_threshold:
        regime = "MILD_DRIFT"
    elif mizan_score >= config.critical_breach_threshold:
        regime = "FLUX_ALERT"
    else:
        regime = "CRITICAL_BREACH"

    # Validity Check: Score above critical threshold and no hard constraint breaches
    is_valid = bool(mizan_score >= config.critical_breach_threshold)
    if is_valid and violations:
        # Canonical sets of hard constraints
        hard_rule_names = {c.rule_name for c in charges if c.is_hard_constraint and c.rule_name}
        hard_charge_ids = {c.charge_id for c in charges if c.is_hard_constraint and c.charge_id}

        for v in violations:
            # Tier 1: Strongly-typed object attribute (O(1) exact boolean check)
            if getattr(v, "is_hard_constraint", None) is not None:
                if v.is_hard_constraint:
                    is_valid = False
                    break
                continue

            # Tier 2: Exact anchored regex extraction on formatted Quanta message template
            m = re.match(r"^Charge '(?P<name>.*)' breached at step \d+:", v)
            if m:
                extracted_name = m.group("name")
                if extracted_name in hard_rule_names:
                    is_valid = False
                    break
                continue

            # Tier 3: Structured prefix templates fallback
            if any(
                v.startswith(f"Charge '{name}' breached at step ") or v.startswith(f"Violation in {name}:")
                for name in hard_rule_names
            ):
                is_valid = False
                break
            if any(v.startswith(f"Charge [{cid}]") for cid in hard_charge_ids):
                is_valid = False
                break

    return NoetherianEvaluation(
        mizan_score=mizan_score,
        action_efficiency=action_metric.action_efficiency,
        entropy_stability=entropy_metric.entropy_stability_score,
        noether_flux=noether_flux,
        charge_conservation=charge_eff,
        charge_drift=max_charge_drift,
        projection_compliance=proj_result.compliance_score,
        action_metric=action_metric,
        entropy_metric=entropy_metric,
        projection_result=proj_result,
        invariant_violations=violations,
        is_valid=is_valid,
        regime=regime,
    )


# ============================================================================
# High-Level Object-Oriented Engine Class
# ============================================================================


class NoetherianInvariantEngine:
    """Core Noetherian Invariant Engine (Mizan Hakemi) for Quanta Cognitive Architecture.

    Manages immutable semantic charges, computes variational trajectory action,
    monitors epistemic entropy flux, executes Tikhonov-regularized manifold projections,
    and produces composite Kozmik Mizan scores.
    """

    def __init__(
        self,
        config: NoetherianEngineConfig | None = None,
        charges: Sequence[ConservedChargeDefinition] | None = None,
        auto_register_canonical: bool = True,
    ) -> None:
        """Initializes the Noetherian Invariant Engine.

        Args:
            config: Optional NoetherianEngineConfig instance.
            charges: Optional initial list of ConservedChargeDefinition instances.
            auto_register_canonical: If True and charges is empty, registers 4 canonical charges.
        """
        self.config = config if config is not None else NoetherianEngineConfig()
        self._charges: dict[str, ConservedChargeDefinition] = {}

        if charges:
            for c in charges:
                self.add_charge(c)
        elif self.config.default_charges:
            for c in self.config.default_charges:
                self.add_charge(c)
        elif auto_register_canonical:
            self.register_canonical_charges()

    @property
    def charges(self) -> list[ConservedChargeDefinition]:
        """Returns the list of currently registered conserved charges."""
        return list(self._charges.values())

    def add_charge(self, charge: ConservedChargeDefinition) -> None:
        """Registers a conserved charge definition into the engine."""
        self._charges[charge.charge_id] = charge

    def remove_charge(self, charge_id: str) -> bool:
        """Removes a registered charge definition by identifier."""
        return self._charges.pop(charge_id, None) is not None

    def register_canonical_charges(self) -> None:
        """Registers the 4 canonical Noetherian invariant charges in 64-dimensional Hilbert space.

        1. Core Intent Alignment: Preserves orientation along primary user mission.
        2. Safety & Non-Harm Boundary: Enforces hard ethical and operational guardrails.
        3. Epistemic Factuality: Restrains hallucination and unfounded conjecture.
        4. Contextual Coherence: Prevents long-horizon context erosion across turns.
        """
        dim = self.config.dim

        c_intent = ConservedChargeDefinition.from_text(
            charge_id="canonical_intent",
            rule_name="Core Intent Alignment",
            description="Preserves semantic fidelity to original user directive and mission goal.",
            dim=dim,
            target_value=0.0,
            weight=1.2,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        c_safety = ConservedChargeDefinition.from_text(
            charge_id="canonical_safety",
            rule_name="Safety & Non-Harm Boundary",
            description="Enforces strict operational boundaries, data security, and ethical constraints.",
            dim=dim,
            target_value=0.0,
            weight=1.5,
            tolerance=0.03,
            is_hard_constraint=True,
        )

        c_truth = ConservedChargeDefinition.from_text(
            charge_id="canonical_truth",
            rule_name="Epistemic Factuality",
            description="Restrains hallucination drift, unwarranted extrapolation, and phantom entities.",
            dim=dim,
            target_value=0.0,
            weight=1.0,
            tolerance=0.06,
            is_hard_constraint=True,
        )

        c_coherence = ConservedChargeDefinition.from_text(
            charge_id="canonical_coherence",
            rule_name="Contextual Coherence",
            description="Guarantees temporal consistency and prevents rule decay over multi-step DAG rollouts.",
            dim=dim,
            target_value=0.0,
            weight=0.8,
            tolerance=0.08,
            is_hard_constraint=False,
        )

        self.add_charge(c_intent)
        self.add_charge(c_safety)
        self.add_charge(c_truth)
        self.add_charge(c_coherence)

    def get_charge_matrix(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Returns the stacked generator matrix C in R^{m x d}."""
        if not self._charges:
            return torch.zeros((0, self.config.dim), dtype=torch.float32, device=device)
        rows = [
            _ensure_real_vector(c.generator_vector, dim=self.config.dim, normalize=True, device=device)
            for c in self._charges.values()
        ]
        return torch.stack(rows, dim=0)

    def get_target_vector(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Returns the target charge vector b in R^m."""
        if not self._charges:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.tensor(
            [float(c.target_value) for c in self._charges.values()],
            dtype=torch.float32,
            device=device,
        )

    def compute_noether_flux(
        self,
        q_curr: torch.Tensor,
        q_prev: torch.Tensor,
    ) -> float:
        """Computes the instantaneous discrete Noether flux between two consecutive states."""
        if not self._charges:
            return 0.0
        C = self.get_charge_matrix(device=q_curr.device)
        weights = torch.tensor(
            [max(0.0, float(c.weight)) for c in self._charges.values()],
            dtype=torch.float32,
            device=q_curr.device,
        )
        q_c = _ensure_real_vector(q_curr, dim=self.config.dim, normalize=True, device=q_curr.device)
        q_p = _ensure_real_vector(q_prev, dim=self.config.dim, normalize=True, device=q_curr.device)

        Q_c = torch.matmul(C, q_c)
        Q_p = torch.matmul(C, q_p)
        return float(torch.sum(weights * torch.abs(Q_c - Q_p)).item())

    def project_state(
        self,
        state: torch.Tensor,
        target_b: torch.Tensor | Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, ManifoldProjectionResult]:
        """Projects a single cognitive statevector onto the invariant manifold."""
        return project_to_manifold(
            state=state,
            charges=self.charges,
            target_b=target_b,
            config=self.config,
        )

    def evaluate_state(
        self,
        state: torch.Tensor,
        previous_state: torch.Tensor | None = None,
        probs: Sequence[float] | torch.Tensor | None = None,
        goal_state: torch.Tensor | None = None,
        consequence_cost: float = 0.0,
    ) -> NoetherianEvaluation:
        """Evaluates a single state step or static node."""
        states = [previous_state, state] if previous_state is not None else [state]
        probs_seq = [probs] if probs is not None else None
        costs_seq = [consequence_cost] if consequence_cost > 0.0 else None

        return evaluate_trajectory_noetherian(
            states=states,
            probs=probs_seq,
            config=self.config,
            charges=self.charges,
            goal_state=goal_state,
            consequence_costs=costs_seq,
        )

    def evaluate_trajectory(
        self,
        states: Sequence[torch.Tensor],
        probs: Sequence[Sequence[float] | torch.Tensor | float] | None = None,
        goal_state: torch.Tensor | None = None,
        consequence_costs: Sequence[float] | float | None = None,
    ) -> NoetherianEvaluation:
        """Evaluates an entire multi-step decision trajectory."""
        return evaluate_trajectory_noetherian(
            states=states,
            probs=probs,
            config=self.config,
            charges=self.charges,
            goal_state=goal_state,
            consequence_costs=consequence_costs,
        )


# Canonical alias reflecting the Turkish ontological nomenclature
MizanHakemi = NoetherianInvariantEngine
