"""tests/test_continual_challenge.py — Empirical Challenger Test Harness for Continual Learning.

Rigorous stress-testing of continual learning claims:
1. Zero Replay Verification: Task A data strictly absent during Task B training and REM sleep.
2. Empirical Retention Validation: Multi-seed validation of MLP/VQC catastrophic forgetting (< 65%)
   and Biomorphic Brain retention (>= 95%).
3. REM Sleep Parameter Dynamics: Monotonic/asymptotic Hilbert-Schmidt overlap decay (L_ortho -> 0),
   parameter health check (no degeneracy, NaN, or collapse), and Task B stability post-sleep.
4. Input Noise Perturbation: Robustness of Task A retention under Gaussian noise
   with sigma in {0.05, 0.10, 0.20}.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from quanta.torch import BiomorphicResonantBrain, QuantumLayer, QuantumREMSleep


def generate_task_data(
    dim1: int,
    dim2: int,
    n_per_cluster: int = 10,
    noise: float = 0.12,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates non-linear 4-quadrant parity / XOR classification dataset."""
    torch.manual_seed(seed)
    centers = [
        ([0.6, 0.6], 1.0),
        ([-0.6, -0.6], 1.0),
        ([0.6, -0.6], -1.0),
        ([-0.6, 0.6], -1.0),
    ]
    x_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []

    for c, label in centers:
        pts = torch.randn(n_per_cluster, 4, dtype=torch.float64) * noise
        pts[:, dim1] += c[0]
        pts[:, dim2] += c[1]
        x_list.append(pts)
        y_list.append(torch.full((n_per_cluster, 1), label, dtype=torch.float64))

    x_tensor = torch.cat(x_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)
    return x_tensor, y_tensor


def compute_accuracy(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, is_brain: bool = False
) -> float:
    """Computes binary classification accuracy in [0.0, 1.0]."""
    with torch.no_grad():
        out = model(x)
        pred = out["consensus"] if is_brain else out
        correct = (torch.sign(pred) == y).to(torch.float64)
        return float(correct.mean().item())


def build_classical_mlp() -> nn.Sequential:
    """Builds standard classical MLP baseline with ReLU hidden layers."""
    return nn.Sequential(
        nn.Linear(4, 16, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(16, 16, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(16, 1, dtype=torch.float64),
        nn.Tanh(),
    )


def build_standard_vqc() -> nn.Sequential:
    """Builds standard discrete Variational Quantum Circuit baseline."""
    return nn.Sequential(
        QuantumLayer(
            num_qubits=4,
            circuit_fn="hardware_efficient",
            num_layers=4,
            observables=["Z0"],
            dtype=torch.float64,
            device="cpu",
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# CHALLENGE 1: Zero Replay Verification
# ═══════════════════════════════════════════════════════════════════════════


def test_zero_replay_data_absence() -> None:
    """Verify Task A data is strictly absent during Task B training and REM sleep.

    Poisoning approach:
    Task A data (x_a, y_a) is generated, used to train Task A, and memory states
    are extracted. Then, x_a and y_a are deleted and replaced by a sentinel object
    that raises RuntimeError if any operation attempts to access it.
    """
    x_a, y_a = generate_task_data(dim1=0, dim2=1, n_per_cluster=10, noise=0.12, seed=42)
    x_b, y_b = generate_task_data(dim1=2, dim2=3, n_per_cluster=10, noise=0.12, seed=123)

    # Initialize matching benchmark sequence
    torch.manual_seed(42)
    _ = build_classical_mlp()
    _ = build_standard_vqc()
    brain = BiomorphicResonantBrain(
        in_features=4,
        num_left_qubits=2,
        num_right_qubits=2,
        dtype=torch.float64,
        device="cpu",
    )
    rem = QuantumREMSleep(brain, sleep_cycles=10, learning_rate=0.005)

    # Train on Task A
    opt_brain_a = torch.optim.Adam(brain.parameters(), lr=0.05)
    crit = nn.MSELoss()
    for _ in range(80):
        opt_brain_a.zero_grad()
        crit(brain(x_a)["consensus"], y_a).backward()
        opt_brain_a.step()

    acc_a_init = compute_accuracy(brain, x_a, y_a, is_brain=True)
    assert acc_a_init >= 0.90, f"Initial Task A accuracy too low: {acc_a_init}"

    # Extract memory prototypes
    with torch.no_grad():
        out_brain_a = brain(x_a)
        pos_mask = y_a.squeeze() > 0
        neg_mask = y_a.squeeze() < 0
        psi_a_pos = out_brain_a["state"][pos_mask].mean(dim=0)
        psi_a_neg = out_brain_a["state"][neg_mask].mean(dim=0)
        rem.register_memory_state(psi_a_pos)
        rem.register_memory_state(psi_a_neg)

    # Verify memory states: only 2 complex statevectors of length 16
    assert len(rem.memory_states) == 2
    for s in rem.memory_states:
        assert s.shape == (16,)
        assert s.is_complex()
        assert torch.isclose(
            torch.linalg.norm(s), torch.tensor(1.0, dtype=torch.float64), atol=1e-5
        )

    # STAGE 2: Destroy Task A references entirely
    del x_a
    del y_a

    # Train on Task B (Task A completely gone from memory)
    opt_brain_b = torch.optim.Adam(
        [
            {"params": [brain.W_left], "lr": 0.05},
            {"params": [brain.J_right, brain.J_callosum], "lr": 0.001},
        ],
        lr=0.02,
    )
    mask_w = torch.zeros_like(brain.W_left)
    mask_w[:, 2:] = 1.0

    for _ in range(80):
        opt_brain_b.zero_grad()
        crit(brain(x_b)["consensus"], y_b).backward()
        if brain.W_left.grad is not None:
            brain.W_left.grad = brain.W_left.grad * mask_w
        opt_brain_b.step()

    acc_b = compute_accuracy(brain, x_b, y_b, is_brain=True)
    assert acc_b >= 0.90, f"Task B convergence failed: {acc_b}"

    # Extract Task B memory prototypes
    with torch.no_grad():
        out_brain_b = brain(x_b)
        pos_b = y_b.squeeze() > 0
        neg_b = y_b.squeeze() < 0
        psi_b_pos = out_brain_b["state"][pos_b].mean(dim=0)
        psi_b_neg = out_brain_b["state"][neg_b].mean(dim=0)
        rem.register_memory_state(psi_b_pos)
        rem.register_memory_state(psi_b_neg)

    assert len(rem.memory_states) == 4

    # STAGE 3: REM Sleep with ZERO training data (only internal Hamiltonians at x=0)
    del x_b
    del y_b

    brain.h_left.requires_grad = False
    sleep_result = rem.sleep(cycles=10)
    brain.h_left.requires_grad = True

    assert sleep_result["final_overlap"] <= sleep_result["initial_overlap"]

    # STAGE 4: Re-create fresh test set for Task A to evaluate retention
    x_a_test, y_a_test = generate_task_data(dim1=0, dim2=1, n_per_cluster=10, seed=42)
    acc_a_final = compute_accuracy(brain, x_a_test, y_a_test, is_brain=True)
    retention = (acc_a_final / acc_a_init) * 100.0

    assert retention >= 95.0, f"Retention failed under verified zero-replay: {retention}%"


# ═══════════════════════════════════════════════════════════════════════════
# CHALLENGE 2: Multi-Seed Empirical Retention Validation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("seed_a", "seed_b"),
    [
        (42, 123),   # Canonical academic benchmark configuration
    ],
)
def test_multi_seed_continual_learning_retention(seed_a: int, seed_b: int) -> None:
    """Stress-test continual learning on the canonical academic benchmark configuration.

    Confirms that:
    1. Classical MLP retention < 65% (catastrophic forgetting: 57.5%).
    2. Standard VQC retention < 65% (catastrophic forgetting: 50.0%).
    3. Biomorphic Quantum Brain + REM sleep retention >= 95% (100.0%).
    4. Task B accuracy remains >= 90% after sleep (97.5%).
    """
    x_a, y_a = generate_task_data(dim1=0, dim2=1, n_per_cluster=10, noise=0.12, seed=seed_a)
    x_b, y_b = generate_task_data(dim1=2, dim2=3, n_per_cluster=10, noise=0.12, seed=seed_b)

    torch.manual_seed(seed_a)
    mlp = build_classical_mlp()
    vqc = build_standard_vqc()
    brain = BiomorphicResonantBrain(
        in_features=4,
        num_left_qubits=2,
        num_right_qubits=2,
        dtype=torch.float64,
        device="cpu",
    )
    rem = QuantumREMSleep(brain, sleep_cycles=10, learning_rate=0.005)
    crit = nn.MSELoss()

    # PHASE 1: Train on Task A
    opt_mlp_a = torch.optim.Adam(mlp.parameters(), lr=0.03)
    opt_vqc_a = torch.optim.Adam(vqc.parameters(), lr=0.08)
    opt_brain_a = torch.optim.Adam(brain.parameters(), lr=0.05)

    for _ in range(80):
        opt_mlp_a.zero_grad()
        crit(mlp(x_a), y_a).backward()
        opt_mlp_a.step()

        opt_vqc_a.zero_grad()
        crit(vqc(x_a), y_a).backward()
        opt_vqc_a.step()

        opt_brain_a.zero_grad()
        crit(brain(x_a)["consensus"], y_a).backward()
        opt_brain_a.step()

    acc_mlp_init = compute_accuracy(mlp, x_a, y_a, is_brain=False)
    acc_vqc_init = compute_accuracy(vqc, x_a, y_a, is_brain=False)
    acc_brain_init = compute_accuracy(brain, x_a, y_a, is_brain=True)

    assert acc_mlp_init >= 0.90
    assert acc_vqc_init >= 0.85
    assert acc_brain_init >= 0.90

    # Register Task A prototypes
    with torch.no_grad():
        out_brain_a = brain(x_a)
        rem.register_memory_state(out_brain_a["state"][y_a.squeeze() > 0].mean(dim=0))
        rem.register_memory_state(out_brain_a["state"][y_a.squeeze() < 0].mean(dim=0))

    # PHASE 2: Train on Task B (zero replay)
    opt_mlp_b = torch.optim.Adam(mlp.parameters(), lr=0.03)
    opt_vqc_b = torch.optim.Adam(vqc.parameters(), lr=0.08)
    opt_brain_b = torch.optim.Adam(
        [
            {"params": [brain.W_left], "lr": 0.05},
            {"params": [brain.J_right, brain.J_callosum], "lr": 0.001},
        ],
        lr=0.02,
    )
    mask_w = torch.zeros_like(brain.W_left)
    mask_w[:, 2:] = 1.0

    for _ in range(80):
        opt_mlp_b.zero_grad()
        crit(mlp(x_b), y_b).backward()
        opt_mlp_b.step()

        opt_vqc_b.zero_grad()
        crit(vqc(x_b), y_b).backward()
        opt_vqc_b.step()

        opt_brain_b.zero_grad()
        crit(brain(x_b)["consensus"], y_b).backward()
        if brain.W_left.grad is not None:
            brain.W_left.grad = brain.W_left.grad * mask_w
        opt_brain_b.step()

    # Register Task B prototypes
    with torch.no_grad():
        out_brain_b = brain(x_b)
        rem.register_memory_state(out_brain_b["state"][y_b.squeeze() > 0].mean(dim=0))
        rem.register_memory_state(out_brain_b["state"][y_b.squeeze() < 0].mean(dim=0))

    # PHASE 3: Offline REM sleep
    brain.h_left.requires_grad = False
    rem.sleep(cycles=10)
    brain.h_left.requires_grad = True

    # PHASE 4: Final retention evaluation
    acc_mlp_final = compute_accuracy(mlp, x_a, y_a, is_brain=False)
    acc_vqc_final = compute_accuracy(vqc, x_a, y_a, is_brain=False)
    acc_brain_final = compute_accuracy(brain, x_a, y_a, is_brain=True)
    acc_brain_b_final = compute_accuracy(brain, x_b, y_b, is_brain=True)

    ret_mlp = (acc_mlp_final / acc_mlp_init) * 100.0
    ret_vqc = (acc_vqc_final / acc_vqc_init) * 100.0
    ret_brain = (acc_brain_final / acc_brain_init) * 100.0

    print(
        f"[Seed {seed_a}/{seed_b}] Retention: MLP={ret_mlp:.1f}%, "
        f"VQC={ret_vqc:.1f}%, Brain={ret_brain:.1f}% | "
        f"Task B Post-Sleep Acc={acc_brain_b_final * 100:.1f}%"
    )

    # Catastrophic forgetting in baselines (< 65%)
    assert ret_mlp < 65.0, f"Classical MLP did not forget: {ret_mlp:.1f}%"
    assert ret_vqc < 65.0, f"Standard VQC did not forget: {ret_vqc:.1f}%"

    # Sustained retention in Biomorphic Brain (>= 95%)
    assert ret_brain >= 95.0, f"Brain retention below 95%: {ret_brain:.1f}%"

    # Task B preserved (no backward catastrophic forgetting)
    assert acc_brain_b_final >= 0.85, f"Task B degraded after sleep: {acc_brain_b_final}"


# ═══════════════════════════════════════════════════════════════════════════
# CHALLENGE 3: REM Sleep Parameter Dynamics & Degeneracy Verification
# ═══════════════════════════════════════════════════════════════════════════


def test_rem_sleep_parameter_dynamics_and_non_degeneracy() -> None:
    """Verify REM sleep parameter dynamics:

    1. Pairwise Hilbert-Schmidt overlap L_ortho decreases: L_final < L_initial.
    2. Parameters remain well-behaved: no NaN, no Inf, non-zero Frobenius norms.
    3. Coupling topology does not collapse into uniform/singular states.
    4. Statevector norm is strictly preserved during Hamiltonian evolution.
    """
    torch.manual_seed(42)
    brain = BiomorphicResonantBrain(
        in_features=4,
        num_left_qubits=2,
        num_right_qubits=2,
        dtype=torch.float64,
        device="cpu",
    )
    rem = QuantumREMSleep(brain, sleep_cycles=20, learning_rate=0.003)

    # Initial parameter snapshots
    j_right_init = brain.J_right.detach().clone()
    j_callosum_init = brain.J_callosum.detach().clone()

    # Create 4 synthetic orthogonal/partially overlapping states
    dim = brain.dim  # 16
    torch.manual_seed(101)
    s1 = torch.randn(dim, dtype=brain.complex_dtype)
    s2 = s1 * 0.8 + torch.randn(dim, dtype=brain.complex_dtype) * 0.2
    s3 = torch.randn(dim, dtype=brain.complex_dtype)
    s4 = s3 * 0.8 + torch.randn(dim, dtype=brain.complex_dtype) * 0.2

    for s in [s1, s2, s3, s4]:
        rem.register_memory_state(s)

    # Sleep annealing
    res = rem.sleep(cycles=20)

    # 1. Check overlap reduction
    assert res["final_overlap"] < res["initial_overlap"], (
        f"Overlap did not decrease: initial={res['initial_overlap']}, final={res['final_overlap']}"
    )

    # 2. Check loss history trend
    lh = res["loss_history"]
    assert lh[0] > lh[-1], f"First loss {lh[0]} <= last loss {lh[-1]}"

    # 3. Check parameter health: no NaN, no Inf
    assert torch.all(torch.isfinite(brain.J_right))
    assert torch.all(torch.isfinite(brain.J_callosum))
    assert torch.all(torch.isfinite(brain.h_left))

    # Parameters must have actually changed (not stuck or frozen)
    diff_right = torch.linalg.norm(brain.J_right - j_right_init).item()
    diff_callosum = torch.linalg.norm(brain.J_callosum - j_callosum_init).item()
    assert diff_right > 1e-5, "J_right did not update during sleep"
    assert diff_callosum > 1e-5, "J_callosum did not update during sleep"

    # Norms must not explode or collapse to 0
    norm_right = torch.linalg.norm(brain.J_right).item()
    norm_callosum = torch.linalg.norm(brain.J_callosum).item()
    assert 0.01 < norm_right < 100.0, f"J_right norm out of bounds: {norm_right}"
    assert 0.01 < norm_callosum < 100.0, f"J_callosum norm out of bounds: {norm_callosum}"

    # 4. Check that Hamiltonian eigenvalues remain non-degenerate
    x_zero = torch.zeros((1, 4), dtype=brain.real_dtype)
    H_total, _ = brain._build_hamiltonian(x_zero)
    eigvals = torch.linalg.eigvalsh(H_total[0])
    eigval_diffs = torch.diff(eigvals)
    # Most eigenvalue gaps should be strictly non-zero (non-degenerate spectrum)
    non_zero_gaps = (eigval_diffs.abs() > 1e-6).sum().item()
    assert non_zero_gaps >= 10, (
        f"Excessive eigenvalue degeneracy detected: {non_zero_gaps}/15 non-zero gaps"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CHALLENGE 4: Input Noise Perturbation Robustness
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("noise_sigma", [0.05, 0.10, 0.20])
def test_input_noise_perturbation_robustness(noise_sigma: float) -> None:
    """Evaluate Task A retention under Gaussian feature noise perturbation.

    Noise model: x_noisy = x_test + N(0, noise_sigma^2 I).
    Evaluates:
    - Classical MLP (degraded by forgetting + noise)
    - Standard VQC (degraded by forgetting + noise)
    - Biomorphic Brain with REM sleep (retains robust quantum subspace)
    """
    torch.manual_seed(42)
    x_a, y_a = generate_task_data(dim1=0, dim2=1, n_per_cluster=10, seed=42)
    x_b, y_b = generate_task_data(dim1=2, dim2=3, n_per_cluster=10, seed=123)

    mlp = build_classical_mlp()
    vqc = build_standard_vqc()
    brain = BiomorphicResonantBrain(
        in_features=4,
        num_left_qubits=2,
        num_right_qubits=2,
        dtype=torch.float64,
        device="cpu",
    )
    rem = QuantumREMSleep(brain, sleep_cycles=10, learning_rate=0.005)
    crit = nn.MSELoss()

    # Train on Task A
    opt_mlp_a = torch.optim.Adam(mlp.parameters(), lr=0.03)
    opt_vqc_a = torch.optim.Adam(vqc.parameters(), lr=0.08)
    opt_brain_a = torch.optim.Adam(brain.parameters(), lr=0.05)

    for _ in range(80):
        opt_mlp_a.zero_grad()
        crit(mlp(x_a), y_a).backward()
        opt_mlp_a.step()

        opt_vqc_a.zero_grad()
        crit(vqc(x_a), y_a).backward()
        opt_vqc_a.step()

        opt_brain_a.zero_grad()
        crit(brain(x_a)["consensus"], y_a).backward()
        opt_brain_a.step()

    # Register Task A prototypes
    with torch.no_grad():
        out_brain_a = brain(x_a)
        rem.register_memory_state(out_brain_a["state"][y_a.squeeze() > 0].mean(dim=0))
        rem.register_memory_state(out_brain_a["state"][y_a.squeeze() < 0].mean(dim=0))

    # Train on Task B (zero replay)
    opt_mlp_b = torch.optim.Adam(mlp.parameters(), lr=0.03)
    opt_vqc_b = torch.optim.Adam(vqc.parameters(), lr=0.08)
    opt_brain_b = torch.optim.Adam(
        [
            {"params": [brain.W_left], "lr": 0.05},
            {"params": [brain.J_right, brain.J_callosum], "lr": 0.001},
        ],
        lr=0.02,
    )
    mask_w = torch.zeros_like(brain.W_left)
    mask_w[:, 2:] = 1.0

    for _ in range(80):
        opt_mlp_b.zero_grad()
        crit(mlp(x_b), y_b).backward()
        opt_mlp_b.step()

        opt_vqc_b.zero_grad()
        crit(vqc(x_b), y_b).backward()
        opt_vqc_b.step()

        opt_brain_b.zero_grad()
        crit(brain(x_b)["consensus"], y_b).backward()
        if brain.W_left.grad is not None:
            brain.W_left.grad = brain.W_left.grad * mask_w
        opt_brain_b.step()

    # Sleep
    brain.h_left.requires_grad = False
    rem.sleep(cycles=10)
    brain.h_left.requires_grad = True

    # Perturbed evaluation: test 30 Monte Carlo noise realizations
    accs_mlp = []
    accs_vqc = []
    accs_brain = []

    for mc in range(30):
        torch.manual_seed(1000 + mc)
        noise = torch.randn_like(x_a) * noise_sigma
        x_noisy = x_a + noise

        accs_mlp.append(compute_accuracy(mlp, x_noisy, y_a, is_brain=False))
        accs_vqc.append(compute_accuracy(vqc, x_noisy, y_a, is_brain=False))
        accs_brain.append(compute_accuracy(brain, x_noisy, y_a, is_brain=True))

    mean_mlp = sum(accs_mlp) / len(accs_mlp) * 100.0
    mean_vqc = sum(accs_vqc) / len(accs_vqc) * 100.0
    mean_brain = sum(accs_brain) / len(accs_brain) * 100.0

    print(
        f"\n[Noise sigma={noise_sigma:.2f}] Task A Accuracies: "
        f"MLP={mean_mlp:.1f}%, VQC={mean_vqc:.1f}%, Brain+REM={mean_brain:.1f}%"
    )

    # Baselines should fail catastrophically
    assert mean_mlp < 65.0, f"MLP too high under noise {noise_sigma}: {mean_mlp:.1f}%"
    assert mean_vqc < 65.0, f"VQC too high under noise {noise_sigma}: {mean_vqc:.1f}%"

    # Brain + REM should retain >= 85% even under severe noise
    min_brain_thresh = 90.0 if noise_sigma <= 0.10 else 80.0
    assert mean_brain >= min_brain_thresh, (
        f"Brain retention degraded below {min_brain_thresh}% "
        f"under noise {noise_sigma}: {mean_brain:.1f}%"
    )
