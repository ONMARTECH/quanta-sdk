"""Comprehensive Scientific Integrity, Anti-Metric Hacking & Biophysical Rigor Audit Suite.

Verifies all five pillars of scientific integrity across Quanta's cognitive,
lattice CVP, biophysical resonance, and documentation layers:
  1. CVP Solver & Adversarial 3-Arm Ablation:
     - Active-coordinate engram slicing (`e.z_vector[active_indices]`).
     - Zero hidden Babai clamping when `allow_babai_fallback=False`.
     - Honest `0.0%` marginal lift in easy BDD vs positive lift in hard Voronoi-boundary regimes.
     - Honest `CLASSICAL_SV_RAM_LIMIT` status (zero false `BARREN_PLATEAU_LIMIT` labels).
  2. Symmetric Feature & Horizon Observability (`quanta.cognitive.ablation`):
     - Zero ground-truth `breaches_invariant` peeking in `OptionSpec.from_raw`.
     - Zero title-keyword regex leakage across variants.
     - Internal entropy-derived `consensus_score` invariant to `is_optimal`.
     - Genuine seeded Gaussian perturbation across multi-trial runs with win/tie/loss tracking.
  3. Mathematical Fixes & Empirical Counterfactuals (`noetherian.py` & `arbiter.py`):
     - Post-projection `final_drift` used for `ManifoldProjectionResult.compliance_score`.
     - Consistent Legendre-dual Euclidean action derived from discrete Lagrangian `S_d`.
     - Empirically executed control baseline for `counterfactual_ab["effect_verified"]`.
  4. Continuous Biophysical Scaling & Lindblad Coherence Decay (`quanta.torch.brain`):
     - Smooth, differentiable, label-independent `CSFShieldedEnvironment` attenuation law
       with non-zero partial derivatives wrt temperature, viscosity, paramagnetic ions,
       and relative permittivity.
     - Proper Lindblad off-diagonal coherence damping in `CSFShieldedResonantLayer`
       preserving `Tr(rho) = 1` and `<Z_q> = Tr(rho Z_q)`.
     - Genuine mixed-state purity `Tr(rho^2) in [1/d, 1]` decay in `NoisyHippocampalBuffer`
       and honest multi-timescale phase dispersion documentation in `QuantumREMSleep`.
  5. Documentation Synchronization & Confidentiality Guard:
     - `README.md` synchronized with `pyproject.toml` (`1.2.0`), `25` MCP tools,
       `~20x-34x` verified Apple Silicon MLX speedup (zero `404x` or stale `0.9.3` claims),
       working `run(..., method="mlx")` snippet, and zero internal leaks across owned files.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cvp_benchmark import (
    generate_adversarial_voronoi_cvp,
    run_benchmark_suite,
    run_three_arm_ablation,
)
from benchmarks.noetherian_ablation_benchmark import (
    NoetherianAblationBenchmark,
    build_periphrastic_rule_erosion_scenario,
    evaluate_variant_naive,
)
from quanta.cognitive.ablation import (
    AblationEngine,
    AblationVariant,
    HardDecisionInstance,
    OptionSpec,
    ZenoAblationEngine,
    build_stochastic_jitter_instance,
    build_trojan_distractor_instance,
)
from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.cvp_arbiter import LatticeEngramMemory
from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    NoetherianEngineConfig,
    evaluate_trajectory_action,
    project_to_manifold,
)
from quanta.solvers.cognitive_qaoa_cvp import CognitiveQAOACVPSolver, CVPSolveResult
from quanta.torch.brain import (
    CSFShieldedEnvironment,
    CSFShieldedResonantLayer,
    NoisyHippocampalBuffer,
    QuantumREMSleep,
)
from tests.test_confidentiality_guard import _scan_text_for_violations

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# 1. CVP Solver Isolation, Active-Subspace Slicing & 3-Arm Adversarial Ablation
# ============================================================================


def test_cvp_engram_warm_start_slices_by_active_indices() -> None:
    """Verifies warm-start priors slice `e.z_vector` by `active_indices`, not `[:n_dim]`."""
    memory = LatticeEngramMemory()
    basis_id = "audit_basis_6d"
    # Coordinates 0, 2, 4 are -4; coordinates 1, 3, 5 are +3
    z_stored = np.array([-4, 3, -4, 3, -4, 3], dtype=np.int64)
    memory.store_solution(
        basis_id=basis_id,
        z=z_stored,
        distance=0.05,
        is_ground_truth=True,
        salience=1.0,
    )

    bits_per_coord = 3  # span=8, half_span=4 -> z_min = 0 - 4 = -4 => val = z + 4
    center = np.zeros(6, dtype=np.int64)
    active_indices = [1, 3, 5]  # All active coordinates have z=+3 -> encoded val = 3 + 4 = 7 (111_2)

    ws = memory.recall_warm_start(
        basis_id=basis_id,
        num_qubits=len(active_indices) * bits_per_coord,
        bits_per_coord=bits_per_coord,
        center=center,
        delta_regularization=0.05,
        active_indices=active_indices,
    )

    assert ws.recalled_engram_count == 1
    # Because val=7 has binary bits (1, 1, 1), every active bit prior must be 0.95 (not ~0.5 or 0.05)
    for p in ws.bit_priors:
        assert p == pytest.approx(0.95, abs=1e-6)


def test_cvp_solver_allow_babai_fallback_false_eliminates_babai_clamping() -> None:
    """Verifies `allow_babai_fallback=False` returns raw QAOA output without clamping to Babai."""
    basis, target, _, _, diag = generate_adversarial_voronoi_cvp(
        dim=4,
        bits_per_coord=2,
        perturbation_ratio=0.64,
        shear=0.47,
        seed=5401,
    )
    assert diag["babai_failed_hyperplane"] is True
    assert 0.58 <= diag["epsilon_over_lambda1"] <= 0.85

    solver_raw = CognitiveQAOACVPSolver(
        p=1,
        shots=8,
        use_zeno=False,
        use_swr=False,
        min_active_coords=3,
        max_iter=8,
        seed=7,
        allow_babai_fallback=False,
    )
    res_raw = solver_raw.solve(basis=basis, target=target, bits_per_coord=2)
    assert isinstance(res_raw, CVPSolveResult)
    assert res_raw.allow_babai_fallback is False
    assert res_raw.distance == pytest.approx(res_raw.qaoa_raw_distance, rel=1e-9, abs=1e-9)
    expected_rel = (
        (res_raw.babai_distance - res_raw.distance) / res_raw.babai_distance * 100.0
    )
    assert res_raw.relative_improvement_pct == pytest.approx(expected_rel, abs=1e-6)
    assert res_raw.strictly_improved_over_babai is bool(
        res_raw.distance < res_raw.babai_distance - 1e-9
    )

    # With allow_babai_fallback=True on identical seed, distance is upper-bounded by babai_distance
    res_hyb = solver_raw.solve(
        basis=basis,
        target=target,
        bits_per_coord=2,
        allow_babai_fallback=True,
    )
    assert res_hyb.allow_babai_fallback is True
    assert res_hyb.distance <= res_hyb.babai_distance + 1e-9
    assert res_hyb.relative_improvement_pct >= -1e-9


def test_cvp_three_arm_ablation_and_honest_ram_cutoff_label(tmp_path: Path) -> None:
    """Verifies honest 0.0% lift in easy BDD, positive lift in hard Voronoi, and RAM cutoff label."""
    ablation = run_three_arm_ablation(
        dimensions=[4],
        num_trials=3,
        bits_per_coord=2,
        regimes=("easy_bdd", "hard_voronoi"),
        output_path=tmp_path / "cvp_3arm_audit.json",
    )

    easy_d4 = ablation["ablation_by_regime"]["easy_bdd"]["4"]
    assert easy_d4["classical_lll_babai"]["exact_recovery_rate"] == pytest.approx(1.0)
    assert easy_d4["hybrid_babai_qaoa"]["marginal_recovery_lift_pct"] == pytest.approx(0.0)
    assert "0.0% marginal recovery lift" in easy_d4["honest_assessment"]

    hard_d4 = ablation["ablation_by_regime"]["hard_voronoi"]["4"]
    assert hard_d4["classical_lll_babai"]["exact_recovery_rate"] < 1.0
    assert (
        hard_d4["hybrid_babai_qaoa"]["exact_recovery_rate"]
        > hard_d4["classical_lll_babai"]["exact_recovery_rate"]
    )
    assert hard_d4["hybrid_babai_qaoa"]["marginal_recovery_lift_pct"] > 0.0

    # At dim=6, bits_per_coord=3 -> total_qubits=18 > 16: must report CLASSICAL_SV_RAM_LIMIT
    suite = run_benchmark_suite(
        dimensions=[6],
        num_trials=1,
        bits_per_coord=3,
        output_path=tmp_path / "cvp_suite_audit.json",
    )
    vanilla_status = suite["results_by_dimension"]["6"]["vanilla_qaoa"]["status"]
    assert "CLASSICAL_SV_RAM_LIMIT" in vanilla_status
    assert "BARREN_PLATEAU_LIMIT" not in vanilla_status


# ============================================================================
# 2. Symmetric Observability, Zero Regex Leakage & Internal Consensus Entropy
# ============================================================================


def test_option_spec_from_raw_zero_ground_truth_peeking() -> None:
    """Verifies `OptionSpec.from_raw` derives `cognitive_panel` without peeking at `breaches`."""
    raw_safe = {
        "id": "cand_alpha",
        "label": "Architecture Candidate Alpha",
        "nominal_reward": 0.82,
        "risk": 0.35,
        "latency": 0.25,
        "maintenance": 0.30,
        "metabolic": 0.15,
        "breaches_invariant": False,
    }
    raw_breach = dict(raw_safe, breaches_invariant=True)

    spec_safe = OptionSpec.from_raw(raw_safe, idx=0)
    spec_breach = OptionSpec.from_raw(raw_breach, idx=0)

    assert spec_safe.cognitive_panel is not None
    assert spec_breach.cognitive_panel is not None
    assert spec_safe.cognitive_panel.aggregate_penalty() == pytest.approx(
        spec_breach.cognitive_panel.aggregate_penalty(), abs=1e-12
    )


def test_ablation_zero_title_regex_leakage_and_symmetric_features() -> None:
    """Verifies renaming option labels with alarm words causes zero hidden regex penalty."""
    inst_base = build_trojan_distractor_instance(dim=64, seed=101)
    norm_opts = inst_base.get_normalized_options()

    renamed_options: list[OptionSpec] = []
    for i, opt in enumerate(norm_opts):
        is_gt = (
            opt.id == inst_base.ground_truth_optimal
            or opt.label == inst_base.ground_truth_optimal
        )
        new_label = (
            "Reckless Monolithic Bypass Quick-Hack"
            if is_gt
            else f"Clean Modular Verified Service {i}"
        )
        renamed_options.append(
            OptionSpec(
                id=opt.id,
                label=new_label,
                nominal_reward=opt.nominal_reward,
                risk=opt.risk,
                latency=opt.latency,
                maintenance=opt.maintenance,
                metabolic=opt.metabolic,
                breaches_invariant=opt.breaches_invariant,
                invariant_penalty=opt.invariant_penalty,
                consequence=opt.consequence,
                cognitive_panel=opt.cognitive_panel,
                statevector=opt.statevector,
                metadata=dict(opt.metadata),
            )
        )

    inst_renamed = HardDecisionInstance(
        instance_id="audit_title_invariance",
        name="Title-Invariance Audit",
        goal=inst_base.goal,
        description="Title-invariance audit instance",
        dilemma=inst_base.dilemma,
        options=renamed_options,
        ground_truth_optimal=inst_base.ground_truth_optimal,
        dilemma_type=inst_base.dilemma_type,
        hard_invariant=inst_base.hard_invariant,
    )

    engine = AblationEngine(seed=101)
    res_orig = engine.run_instance(inst_base, num_trials=1, log_telemetry=False)
    res_ren = engine.run_instance(inst_renamed, num_trials=1, log_telemetry=False)

    # Greedy and Rule-Based depend strictly on symmetric structured numeric attributes
    for variant in (AblationVariant.GREEDY.value, AblationVariant.RULE_BASED.value):
        assert (
            res_orig.variant_results[variant].chosen_id
            == res_ren.variant_results[variant].chosen_id
        )
        assert res_orig.variant_results[variant].net_utility == pytest.approx(
            res_ren.variant_results[variant].net_utility, abs=1e-9
        )

    # Noetherian naive guardrail is also strictly free of uppercase keyword matching
    scen_orig = build_periphrastic_rule_erosion_scenario(dim=64)
    res_naive_orig = evaluate_variant_naive(scen_orig)
    scen_alarm = build_periphrastic_rule_erosion_scenario(dim=64)
    for nd in scen_alarm.dag.nodes.values():
        if nd.depth > 0:
            nd.label = "BYPASS UNSAFE HACK OVERRIDE " + nd.label
    res_naive_alarm = evaluate_variant_naive(scen_alarm)
    assert res_naive_orig.chosen_trajectory_id == res_naive_alarm.chosen_trajectory_id


def test_consensus_score_purely_internal_entropy_and_never_peeks_is_optimal() -> None:
    """Verifies `consensus_score` depends strictly on internal entropy, never `is_optimal`."""
    peaked = ZenoAblationEngine._compute_internal_consensus(
        {"a": 0.91, "b": 0.03, "c": 0.03, "d": 0.03}
    )
    uniform = ZenoAblationEngine._compute_internal_consensus(
        {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    )
    assert peaked > 0.60
    assert uniform == pytest.approx(0.0, abs=1e-9)

    inst_true = build_trojan_distractor_instance(dim=64, seed=55)
    norm_opts = inst_true.get_normalized_options()
    wrong_id = next(
        o.id
        for o in norm_opts
        if o.id != inst_true.ground_truth_optimal and o.label != inst_true.ground_truth_optimal
    )
    inst_flipped = HardDecisionInstance(
        instance_id="audit_consensus_no_peek",
        name="Flipped Ground-Truth Audit",
        goal=inst_true.goal,
        description="Flipped ground-truth label audit",
        dilemma=inst_true.dilemma,
        options=inst_true.options,
        ground_truth_optimal=wrong_id,
        dilemma_type=inst_true.dilemma_type,
        hard_invariant=inst_true.hard_invariant,
    )

    engine = AblationEngine(seed=55)
    r_true = engine.run_instance(inst_true, num_trials=1, log_telemetry=False)
    r_flip = engine.run_instance(inst_flipped, num_trials=1, log_telemetry=False)

    # is_optimal flips from True to False, yet consensus_score MUST remain bit-for-bit identical!
    assert (
        r_true.variant_results["zeno_arbiter"].is_optimal
        != r_flip.variant_results["zeno_arbiter"].is_optimal
    )
    assert r_true.consensus_score == pytest.approx(r_flip.consensus_score, abs=1e-12)
    assert r_true.consensus_score != 0.95
    assert r_flip.consensus_score != 0.50


def test_real_stochastic_jitter_and_multitrial_statistics() -> None:
    """Verifies real seeded Gaussian perturbation and win/tie/loss statistics across trials."""
    j_zero_a = build_stochastic_jitter_instance(dim=64, seed=42, noise_std=0.0)
    j_zero_b = build_stochastic_jitter_instance(dim=64, seed=99, noise_std=0.0)
    j_noisy_42 = build_stochastic_jitter_instance(dim=64, seed=42, noise_std=0.07)
    j_noisy_43 = build_stochastic_jitter_instance(dim=64, seed=43, noise_std=0.07)

    opts_za = j_zero_a.get_normalized_options()
    opts_zb = j_zero_b.get_normalized_options()
    opts_n42 = j_noisy_42.get_normalized_options()
    opts_n43 = j_noisy_43.get_normalized_options()

    # Zero noise produces identical option rewards across seeds; positive noise differs by seed
    assert [o.nominal_reward for o in opts_za] == pytest.approx(
        [o.nominal_reward for o in opts_zb], abs=1e-12
    )
    assert [o.nominal_reward for o in opts_n42] != pytest.approx(
        [o.nominal_reward for o in opts_n43], abs=1e-6
    )

    engine = AblationEngine(seed=42)
    multi = engine.run_instance(j_noisy_42, num_trials=8, log_telemetry=False, noise_std=0.08)
    assert multi.metadata["num_trials"] == 8
    assert multi.metadata["noise_std"] == pytest.approx(0.08)

    # Across 8 noisy trials, standard deviation of evaluated utility/Mizan metrics is non-zero
    zeno_stats = multi.metadata["trial_stats"]["zeno_arbiter"]
    assert zeno_stats["std_net_utility"] > 0.0 or zeno_stats["std_mizan_score"] > 0.0

    for comp_key in ("vs_greedy", "vs_rule_based"):
        wtl = multi.metadata[comp_key]
        total_rate = wtl["win_rate"] + wtl["tie_rate"] + wtl["loss_rate"]
        assert total_rate == pytest.approx(1.0, abs=1e-9)

    # Noetherian benchmark also exposes honest win/tie/loss decomposition
    nb_report = NoetherianAblationBenchmark(seed=42, runs=1, verbose=False).run_suite()
    wtl_base = nb_report["global_metrics"]["vs_baseline_win_tie_loss"]
    assert (
        wtl_base["win_rate"] + wtl_base["tie_rate"] + wtl_base["loss_rate"]
        == pytest.approx(1.0, abs=1e-9)
    )


# ============================================================================
# 3. Noetherian Manifold Post-Projection Compliance & Empirical Counterfactuals
# ============================================================================


def test_project_to_manifold_compliance_uses_post_projection_final_drift() -> None:
    """Verifies `project_to_manifold` computes `compliance_score` from `final_drift`."""
    cfg = NoetherianEngineConfig(dim=16, charge_sigma=0.35)
    q_vec = torch.zeros(16, dtype=torch.float64)
    q_vec[0] = 1.0
    charges = [
        ConservedChargeDefinition(
            charge_id="q0",
            rule_name="Primary Axis Invariant",
            generator_vector=q_vec,
            target_value=0.6,
        )
    ]

    # Off-manifold state: <q0, state> = -0.5 != 0.6 -> large initial drift = 1.1
    off_state = torch.zeros(16, dtype=torch.float64)
    off_state[0] = -0.5
    off_state[1] = math.sqrt(0.75)

    proj_state, res = project_to_manifold(off_state, charges=charges, config=cfg)
    assert res.drift_distance_before > 0.8
    assert res.drift_distance_after < 0.05
    expected_post_compliance = math.exp(-res.drift_distance_after / cfg.charge_sigma)
    assert res.compliance_score == pytest.approx(expected_post_compliance, rel=1e-6, abs=1e-6)
    assert res.compliance_score > 0.90
    assert torch.linalg.norm(proj_state).item() == pytest.approx(1.0, abs=1e-6)


def test_evaluate_trajectory_action_consistent_legendre_dual() -> None:
    """Verifies discrete Lagrangian `S_d = T - V` and Euclidean action `T + V` consistency."""
    cfg = NoetherianEngineConfig(dim=4, action_scale=0.60)
    s0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    s1 = torch.tensor([0.8, 0.6, 0.0, 0.0], dtype=torch.float64)
    goal = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)

    metric = evaluate_trajectory_action([s0, s1], goal_state=goal, consequence_costs=[0.3], config=cfg)
    s_lagrangian = metric.kinetic_energy - metric.potential_energy
    expected_euclidean = s_lagrangian + 2.0 * metric.potential_energy
    assert metric.discrete_action == pytest.approx(s_lagrangian, rel=1e-9, abs=1e-9)
    assert metric.euclidean_action == pytest.approx(expected_euclidean, rel=1e-9, abs=1e-9)
    expected_eff = math.exp(-expected_euclidean / (cfg.action_scale**2))
    assert metric.action_efficiency == pytest.approx(expected_eff, rel=1e-6, abs=1e-6)


def test_arbiter_counterfactual_ab_empirically_verified_vs_false() -> None:
    """Verifies `counterfactual_ab['effect_verified']` is False unless baseline actually differs."""
    arbiter = QuantumDecisionArbiter()

    # Case A: Option 1 is both goal-identical (wins raw cosine baseline) and low-penalty
    # (wins panel-regularized Zeno arbiter) -> effect_verified MUST be False!
    goal_str = "Transactional database migration with schema validation"
    res_same = arbiter.arbitrate(
        goal=goal_str,
        options=[
            {
                "name": goal_str,
                "cognitive_panel": {
                    "neurobiology": {"synaptic_saturation": 0.02, "energy_expenditure": 0.02},
                    "psychiatry": {"rumination_risk": 0.02, "threat_distortion": 0.02},
                    "sociology": {"social_misalignment": 0.02, "user_fatigue_impact": 0.02},
                },
            },
            {
                "name": "Unrelated graphics shader tweak",
                "cognitive_panel": {
                    "neurobiology": {"synaptic_saturation": 0.98, "energy_expenditure": 0.98},
                    "psychiatry": {"rumination_risk": 0.98, "threat_distortion": 0.98},
                    "sociology": {"social_misalignment": 0.98, "user_fatigue_impact": 0.98},
                },
            },
        ],
        panel_weight=0.85,
        exploration_drive=0.0,
        log_telemetry=False,
    )
    cf_same = res_same["counterfactual_ab"]
    assert cf_same["baseline_executed"] is True
    assert cf_same["baseline_choice"] == cf_same["quanta_choice"]
    assert cf_same["effect_verified"] is False

    # Case A2: Goal-identical lure carries severe cognitive panel hazard -> baseline picks lure,
    # regularized arbiter switches to safe option -> effect_verified MUST be True!
    res_diff_flat = arbiter.arbitrate(
        goal=goal_str,
        options=[
            {
                "name": goal_str,
                "cognitive_panel": {
                    "neurobiology": {"synaptic_saturation": 0.99, "energy_expenditure": 0.99},
                    "psychiatry": {"rumination_risk": 0.99, "threat_distortion": 0.99},
                    "sociology": {"social_misalignment": 0.99, "user_fatigue_impact": 0.99},
                },
            },
            {
                "name": "Safe verified database schema rollout",
                "cognitive_panel": {
                    "neurobiology": {"synaptic_saturation": 0.01, "energy_expenditure": 0.01},
                    "psychiatry": {"rumination_risk": 0.01, "threat_distortion": 0.01},
                    "sociology": {"social_misalignment": 0.01, "user_fatigue_impact": 0.01},
                },
            },
        ],
        panel_weight=0.90,
        exploration_drive=0.0,
        log_telemetry=False,
    )
    cf_diff_flat = res_diff_flat["counterfactual_ab"]
    assert cf_diff_flat["baseline_executed"] is True
    assert cf_diff_flat["baseline_choice"] != cf_diff_flat["quanta_choice"]
    assert cf_diff_flat["effect_verified"] is True

    # Case B: Multibranch DAG where greedy step-1 lure differs from full-horizon Zeno path
    res_dag = arbiter.arbitrate_multibranch(
        goal="Select sustainable long-horizon architecture",
        branches={
            "Fast shortcut with severe step-2 debt": {
                "immediate_utility": 1.50,
                "consequences": {"risk": 0.25, "latency": 0.10, "maintenance": 0.20, "metabolic": 0.10},
                "sub_branches": {
                    "Cascading production outage": {
                        "immediate_utility": 0.05,
                        "consequences": {
                            "risk": 0.99,
                            "latency": 0.95,
                            "maintenance": 0.99,
                            "metabolic": 0.95,
                        },
                        "cognitive_panel": {
                            "neurobiology": {"synaptic_saturation": 0.95, "energy_expenditure": 0.95},
                            "psychiatry": {"rumination_risk": 0.95, "threat_distortion": 0.95},
                            "sociology": {"social_misalignment": 0.95, "user_fatigue_impact": 0.95},
                        },
                    }
                },
            },
            "Disciplined modular architecture": {
                "immediate_utility": 0.40,
                "consequences": {"risk": 0.04, "latency": 0.08, "maintenance": 0.04, "metabolic": 0.05},
                "sub_branches": {
                    "Sustainable long-horizon architecture steady state": {
                        "immediate_utility": 0.90,
                        "consequences": {
                            "risk": 0.03,
                            "latency": 0.05,
                            "maintenance": 0.03,
                            "metabolic": 0.04,
                        },
                    }
                },
            },
        },
        panel_weight=0.6,
        log_telemetry=False,
    )
    cf_dag = res_dag["counterfactual_ab"]
    assert cf_dag["baseline_executed"] is True
    assert cf_dag["effect_verified"] is bool(
        cf_dag["baseline_choice"] != cf_dag["quanta_choice"]
    )
    assert cf_dag["effect_verified"] is True


# ============================================================================
# 4. Continuous Biophysical Scaling Law & Lindblad Coherence Evolution
# ============================================================================


def test_csf_shielded_environment_continuous_partial_derivatives_and_zero_magic_branches() -> None:
    """Verifies continuous, monotonic sensitivity to T, viscosity, paramagnetic ions, and eps_r."""
    env = CSFShieldedEnvironment(
        learnable_params=True,
        temperature_k=310.15,
        viscosity_mpa_s=0.72,
        paramagnetic_ion_mM=0.0005,  # 0.5 uM
        relative_permittivity=79.5,
    )
    kappa = env.compute_attenuation_factor()
    kappa.backward()

    # Every physical parameter must have a strictly non-zero analytical gradient
    assert env.temperature.grad is not None and abs(env.temperature.grad.item()) > 1e-10
    assert env.viscosity.grad is not None and abs(env.viscosity.grad.item()) > 1e-8
    assert (
        env.paramagnetic_concentration.grad is not None
        and abs(env.paramagnetic_concentration.grad.item()) > 1e-8
    )
    assert (
        env.dielectric_constant.grad is not None
        and abs(env.dielectric_constant.grad.item()) > 1e-8
    )

    # Verify changing internal condition label alone NEVER alters attenuation (zero magic branches)
    env_a = CSFShieldedEnvironment()
    env_a.simulate_clinical_condition("normal")
    kappa_a = float(env_a.compute_attenuation_factor().item())
    env_a._clinical_condition = "meningitis"
    kappa_b = float(env_a.compute_attenuation_factor().item())
    assert kappa_a == pytest.approx(kappa_b, rel=1e-12, abs=1e-12)

    # Monotonic directional physics checks:
    # 1. Higher paramagnetic ion concentration -> weaker shielding (larger kappa)
    k_low_mag = float(
        CSFShieldedEnvironment(paramagnetic_ion_mM=0.0002).compute_attenuation_factor().item()
    )
    k_high_mag = float(
        CSFShieldedEnvironment(paramagnetic_ion_mM=0.0100).compute_attenuation_factor().item()
    )
    assert k_high_mag > k_low_mag

    # 2. Higher relative permittivity -> stronger Debye screening (smaller kappa)
    k_low_eps = float(
        CSFShieldedEnvironment(relative_permittivity=40.0).compute_attenuation_factor().item()
    )
    k_high_eps = float(
        CSFShieldedEnvironment(relative_permittivity=85.0).compute_attenuation_factor().item()
    )
    assert k_high_eps < k_low_eps

    # 3. Higher viscosity -> slower rotational tumbling / faster dipolar dephasing (larger kappa)
    k_low_visc = float(
        CSFShieldedEnvironment(viscosity_mpa_s=0.65).compute_attenuation_factor().item()
    )
    k_high_visc = float(
        CSFShieldedEnvironment(viscosity_mpa_s=2.40).compute_attenuation_factor().item()
    )
    assert k_high_visc > k_low_visc

    # 4. Continuous sensitivity to temperature (monotonic decrease of tau_c = eta / T with T)
    k_t305 = float(
        CSFShieldedEnvironment(temperature_k=305.0).compute_attenuation_factor().item()
    )
    k_t315 = float(
        CSFShieldedEnvironment(temperature_k=315.0).compute_attenuation_factor().item()
    )
    assert k_t305 > k_t315
    assert k_t305 != pytest.approx(k_t315, rel=1e-3)


def test_csf_resonant_layer_lindblad_density_matrix_and_hippocampal_purity() -> None:
    """Verifies Lindblad off-diagonal damping preserves `Tr(rho)=1` and decays purity physically."""
    torch.manual_seed(17)
    x = torch.randn(2, 4, dtype=torch.float64)

    env_norm = CSFShieldedEnvironment()
    env_norm.simulate_clinical_condition("normal")
    env_men = CSFShieldedEnvironment()
    env_men.simulate_clinical_condition("meningitis")

    layer_norm = CSFShieldedResonantLayer(
        in_features=4,
        num_left_qubits=1,
        num_right_qubits=1,
        bare_dephasing_rate=1.5,
        csf_environment=env_norm,
    )
    layer_men = CSFShieldedResonantLayer(
        in_features=4,
        num_left_qubits=1,
        num_right_qubits=1,
        bare_dephasing_rate=1.5,
        csf_environment=env_men,
    )
    layer_men.brain.load_state_dict(layer_norm.brain.state_dict())

    out_norm = layer_norm(x)
    out_men = layer_men(x)
    assert isinstance(out_norm, dict) and isinstance(out_men, dict)

    rho_men = out_men["density_matrix"]
    # Trace preservation Tr(rho) == 1 and Hermiticity
    traces = torch.diagonal(rho_men, dim1=-2, dim2=-1).real.sum(dim=-1)
    assert torch.allclose(traces, torch.ones_like(traces), atol=1e-6)
    assert torch.allclose(rho_men, rho_men.conj().transpose(-2, -1), atol=1e-6)

    # Meningitis has higher effective dephasing -> strictly lower mixed-state purity Tr(rho^2)
    assert torch.all(out_men["purity"] < out_norm["purity"])
    assert torch.all(out_men["purity"] >= 0.25 - 1e-6)

    # Hippocampal buffer tracks genuine mixed-state purity Tr(rho^2) in [1/d, 1]
    buf = NoisyHippocampalBuffer(
        capacity=8,
        noise_level=0.05,
        phase_diffusion_rate=0.25,
        temporal_decay_rate=0.10,
    )
    psi_plus = torch.ones(4, dtype=torch.complex128) / 2.0
    buf.store(psi_plus, dopamine_tag=1.0)
    assert buf.get_purities()[0] == pytest.approx(1.0, abs=1e-6)

    buf.step(dt=1.5)
    purity_after = buf.get_purities()[0]
    rho_buf = buf.get_density_matrices()[0]
    assert 0.25 <= purity_after < 0.99
    assert float(torch.trace(rho_buf).real.item()) == pytest.approx(1.0, abs=1e-6)

    # QuantumREMSleep docstring honestly documents differential multi-timescale U(t_j)
    assert QuantumREMSleep.__doc__ is not None
    assert "multi-timescale" in QuantumREMSleep.__doc__.lower()
    assert "preserves all" in QuantumREMSleep.__doc__.lower()


# ============================================================================
# 5. README Synchronization & Owned-Files Zero-Leak Confidentiality Audit
# ============================================================================


def test_readme_honesty_and_owned_files_confidentiality() -> None:
    """Verifies README version/speedup/MCP/QEC accuracy and zero confidentiality leaks."""
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "version-1.2.0-blue.svg" in readme_text
    assert "MCP-25%20tools-teal.svg" in readme_text
    assert "**25 MCP tools**" in readme_text
    assert 'method="mlx"' in readme_text
    assert "Edmonds Blossom MWPM" in readme_text
    assert "Delfosse–Nickerson Union-Find" in readme_text
    assert "Kubica–Delfosse 6.6.6 Color Code" in readme_text
    assert "[[144, 12, 12]]" in readme_text
    assert "BP-OSD" in readme_text

    # Unverified speedups and stale footer must be absent
    assert "404x" not in readme_text
    assert "52.09" not in readme_text
    assert "Version:     0.9.3" not in readme_text
    assert "Tests:       888" not in readme_text

    owned_files = [
        "quanta/cognitive/ablation.py",
        "quanta/cognitive/arbiter.py",
        "quanta/cognitive/noetherian.py",
        "quanta/cognitive/cvp_arbiter.py",
        "quanta/solvers/cognitive_qaoa_cvp.py",
        "quanta/torch/brain.py",
        "benchmarks/cvp_benchmark.py",
        "benchmarks/noetherian_ablation_benchmark.py",
        "README.md",
        "tests/test_scientific_integrity_audit.py",
    ]
    violations: list[str] = []
    for rel in owned_files:
        content = (REPO_ROOT / rel).read_text(encoding="utf-8")
        violations.extend(_scan_text_for_violations(rel, content))
    assert not violations, "Confidentiality violations in Slice-4 files:\n" + "\n".join(violations)
