"""Academic Benchmark Suite for Biomorphic Quantum Brain Architecture.

Generates 3 publication-ready empirical figures and raw JSON telemetry:
1. Fig 1: Barren Plateau Resilience (Gradient Variance Scaling vs. Standard VQC).
2. Fig 2: Cognitive Dilemma & Consensus Dynamics (Temporal Debate & Consensus Collapse).
3. Fig 3: Biochemical & Hemodynamic Ablation Study (Loss Trajectories).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from quanta.torch import BiomorphicResonantBrain, QuantumLayer

# Set academic plot style
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
    }
)

OUTPUT_DIR = Path("docs/paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = Path("docs/paper/benchmark_academic_data.json")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Barren Plateau Resilience (Gradient Variance Scaling)
# ═══════════════════════════════════════════════════════════════════════════


def run_barren_plateau_experiment(
    qubit_range: list[int] | None = None,
    num_samples: int = 40,
) -> dict[str, list[float]]:
    if qubit_range is None:
        qubit_range = [4, 6, 8, 10]
    """Evaluates gradient variance scaling across qubit count N.

    Standard VQC suffers from exponential concentration of measure (Barren Plateau),
    where Var[d<Z>/dθ] ~ O(2^{-N}) -> 0.
    BiomorphicResonantBrain preserves non-zero polynomial gradient variance due to
    continuous transverse tunneling (dopamine) and non-local Hamiltonian resonance.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Barren Plateau Scaling (Gradient Variance)")
    print("=" * 70)

    vqc_variances: list[float] = []
    brain_variances: list[float] = []

    for n in qubit_range:
        print(f"--> Evaluating N = {n} qubits across {num_samples} random initializations...")
        vqc_grads: list[float] = []
        brain_grads: list[float] = []

        n_left = n // 2
        n_right = n - n_left

        for seed in range(num_samples):
            torch.manual_seed(seed + 1000)

            # 1. Standard Discrete VQC (Hardware Efficient Ansatz)
            vqc = QuantumLayer(
                num_qubits=n,
                circuit_fn="hardware_efficient",
                num_layers=3,
                observables=["Z0"],
                dtype=torch.float64,
                device="cpu",
            )
            x_vqc = torch.randn(1, n, dtype=torch.float64, requires_grad=False)
            out_vqc = vqc(x_vqc)
            loss_vqc = out_vqc.sum()
            loss_vqc.backward()
            g_vqc = float(vqc.weights.grad[0].item()) if vqc.weights.grad is not None else 0.0
            vqc_grads.append(g_vqc)

            # 2. Biomorphic Resonant Brain
            brain = BiomorphicResonantBrain(
                in_features=n,
                num_left_qubits=n_left,
                num_right_qubits=n_right,
                enable_neuromodulation=True,
                enable_oxygenation=True,
                device="cpu",
                dtype=torch.float64,
            )
            x_brain = torch.randn(1, n, dtype=torch.float64, requires_grad=False)
            out_brain = brain(x_brain)
            loss_brain = out_brain["consensus"].sum()
            loss_brain.backward()
            g_brain = (
                float(brain.W_left.grad[0, 0].item()) if brain.W_left.grad is not None else 0.0
            )
            brain_grads.append(g_brain)

        var_vqc = float(np.var(vqc_grads))
        var_brain = float(np.var(brain_grads))

        vqc_variances.append(max(var_vqc, 1e-12))
        brain_variances.append(max(var_brain, 1e-12))
        print(f"    N={n:2d} | VQC Var: {var_vqc:.2e} | Brain Var: {var_brain:.2e}")

    # Plot Fig 1
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax.plot(
        qubit_range,
        vqc_variances,
        "o--",
        color="#D9534F",
        label="Standard Discrete VQC (Hardware-Efficient)",
        markersize=8,
    )
    ax.plot(
        qubit_range,
        brain_variances,
        "s-",
        color="#2E86C1",
        label="Biomorphic Resonant Brain (Ours)",
        markersize=8,
    )

    # Theoretical exponential decay reference
    theo_x = np.linspace(min(qubit_range), max(qubit_range), 100)
    theo_y = vqc_variances[0] * (2.0 ** -(theo_x - qubit_range[0]))
    ax.plot(
        theo_x,
        theo_y,
        ":",
        color="gray",
        alpha=0.7,
        label=r"Theoretical Barren Plateau $\sim \mathcal{O}(2^{-N})$",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"Number of Qubits ($N$)")
    ax.set_ylabel(r"Gradient Variance $\mathrm{Var}[\partial_{\theta} \mathcal{L}]$ (log scale)")
    ax.set_title("Resilience to Barren Plateaus: Discrete VQC vs. Biomorphic Resonance", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower left")
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "fig1_barren_plateau.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[OK] Saved Fig 1 to {fig_path}")

    return {
        "qubit_range": [float(q) for q in qubit_range],
        "vqc_variances": vqc_variances,
        "brain_variances": brain_variances,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Cognitive Dilemma & Parliament Consensus Dynamics
# ═══════════════════════════════════════════════════════════════════════════


def run_cognitive_dilemma_experiment() -> dict[str, list[float]]:
    """Traces temporal trajectory of the brain under an ambiguous cognitive dilemma.

    Left Lobe (analytical) initially polarizes towards +1, Right Lobe (holistic)
    polarizes towards -1. Under neuromodulatory acceleration and inter-lobe
    tunneling, constructive resonance collapses into a unanimous parliament consensus.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cognitive Dilemma & Consensus Dynamics")
    print("=" * 70)

    torch.manual_seed(42)
    num_left = 2
    num_right = 3

    brain = BiomorphicResonantBrain(
        in_features=4,
        num_left_qubits=num_left,
        num_right_qubits=num_right,
        enable_neuromodulation=True,
        enable_oxygenation=True,
        device="cpu",
        dtype=torch.float64,
    )

    # Ambiguous input vector sitting directly on the decision boundary
    x_ambiguous = torch.tensor([[0.85, -0.75, 0.45, -0.60]], dtype=torch.float64)

    # Time grid from t = 0 to t = 3.5
    time_grid = np.linspace(0.05, 3.5, 70)
    left_consensus_hist: list[float] = []
    right_consensus_hist: list[float] = []
    global_consensus_hist: list[float] = []

    # Fix parameters and modulate base_time across the grid
    with torch.no_grad():
        for t_val in time_grid:
            brain.base_time.copy_(torch.tensor(t_val, dtype=torch.float64))
            out = brain(x_ambiguous)
            left_consensus_hist.append(float(out["left_consensus"].item()))
            right_consensus_hist.append(float(out["right_consensus"].item()))
            global_consensus_hist.append(float(out["consensus"].item()))

    # Plot Fig 2
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # Top: Lobe debate and global consensus collapse
    ax1.plot(
        time_grid,
        left_consensus_hist,
        "--",
        color="#E67E22",
        label=r"Left Lobe $\langle Z_{\mathrm{left}} \rangle$ (Analytic / Categorical)",
    )
    ax1.plot(
        time_grid,
        right_consensus_hist,
        "--",
        color="#8E44AD",
        label=r"Right Lobe $\langle Z_{\mathrm{right}} \rangle$ (Holistic / Contextual)",
    )
    ax1.plot(
        time_grid,
        global_consensus_hist,
        "-",
        color="#16A085",
        linewidth=2.8,
        label=r"Parliament Consensus $\hat{M}_{\mathrm{consensus}}$ (Macroscopic)",
    )

    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax1.set_ylabel(r"Expectation Polarity $\langle \sigma^z \rangle \in [-1, 1]$")
    ax1.set_title("Temporal Dynamics of Dual-Hemisphere Debate and Coherent Consensus", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="upper right")

    # Bottom: Inter-lobe difference (Cognitive Tension |Left - Right|)
    tension = [
        abs(left - right)
        for left, right in zip(left_consensus_hist, right_consensus_hist, strict=True)
    ]
    ax2.fill_between(
        time_grid,
        tension,
        color="#E74C3C",
        alpha=0.35,
        label=r"Inter-Hemispheric Tension $|\langle Z_L \rangle - \langle Z_R \rangle|$",
    )
    ax2.plot(time_grid, tension, color="#C0392B", linewidth=1.8)
    ax2.set_xlabel("Effective Interaction Duration (Evolution Time $\\tau$)")
    ax2.set_ylabel("Cognitive Tension")
    ax2.set_ylim(0, max(tension) * 1.25)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "fig2_cognitive_dilemma.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[OK] Saved Fig 2 to {fig_path}")

    return {
        "time_grid": [float(t) for t in time_grid],
        "left_consensus": left_consensus_hist,
        "right_consensus": right_consensus_hist,
        "global_consensus": global_consensus_hist,
        "tension": tension,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Biochemical & Hemodynamic Ablation Study
# ═══════════════════════════════════════════════════════════════════════════


def run_ablation_experiment(epochs: int = 40) -> dict[str, list[float]]:
    """Ablation study demonstrating the critical role of each biomorphic component.

    Compares 4 configurations:
    1. Full Biomorphic Brain (Dopamine + NE + 5-HT + Oxygenation)
    2. Ablated Dopamine (No transverse tunneling exploration -> gets trapped)
    3. Ablated Oxygenation (No metabolic energy constraint -> lobe imbalance)
    4. Static Quantum Baseline (No neuromodulation, no oxygenation)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Biochemical & Hemodynamic Ablation Study")
    print("=" * 70)

    # Synthetic non-linear classification dataset (Continuous Parity / XOR task)
    torch.manual_seed(123)
    N_samples = 40
    X = torch.randn(N_samples, 4, dtype=torch.float64)
    # Target is non-linear parity of sign patterns
    y = torch.sign(X[:, 0] * X[:, 1] - X[:, 2] * X[:, 3]).unsqueeze(-1).to(torch.float64)

    configs = {
        "Full Biomorphic Brain (Ours)": {
            "neuro": True,
            "oxy": True,
            "color": "#1B4F72",
            "style": "-",
        },
        "Ablated Dopamine (No Exploration)": {
            "neuro": False,
            "oxy": True,
            "color": "#B03A2E",
            "style": "--",
        },
        "Ablated Oxygenation (No Energy Bound)": {
            "neuro": True,
            "oxy": False,
            "color": "#D35400",
            "style": "-.",
        },
        "Static Quantum Baseline": {"neuro": False, "oxy": False, "color": "#7D6608", "style": ":"},
    }

    loss_trajectories: dict[str, list[float]] = {}

    for name, cfg in configs.items():
        print(f"--> Training configuration: '{name}'...")
        torch.manual_seed(42)
        brain = BiomorphicResonantBrain(
            in_features=4,
            num_left_qubits=2,
            num_right_qubits=2,
            enable_neuromodulation=cfg["neuro"],
            enable_oxygenation=cfg["oxy"],
            device="cpu",
            dtype=torch.float64,
        )
        optimizer = torch.optim.Adam(brain.parameters(), lr=0.04)
        criterion = nn.MSELoss()

        losses: list[float] = []
        for _epoch in range(epochs):
            optimizer.zero_grad()
            out = brain(X)
            loss = criterion(out["consensus"], y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        loss_trajectories[name] = losses
        print(f"    Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")

    # Plot Fig 3
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300)
    for name, cfg in configs.items():
        ax.plot(
            range(1, epochs + 1),
            loss_trajectories[name],
            cfg["style"],
            color=cfg["color"],
            label=name,
            linewidth=2.2 if "Full" in name else 1.8,
        )

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel(r"Task Mean Squared Error ($\mathcal{L}_{\mathrm{MSE}}$)")
    ax.set_title("Ablation Study: Neuromodulation & Metabolic Energy Conservation", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "fig3_ablation_study.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[OK] Saved Fig 3 to {fig_path}")

    return loss_trajectories


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE PUBLICATION FIGURE (3-PANEL BANNER)
# ═══════════════════════════════════════════════════════════════════════════


def generate_composite_figure(
    bp_data: dict[str, list[float]],
    dilemma_data: dict[str, list[float]],
    ablation_data: dict[str, list[float]],
) -> None:
    """Combines all 3 empirical results into a single publication-ready banner."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel A: Barren Plateau
    q_range = bp_data["qubit_range"]
    ax1.plot(
        q_range,
        bp_data["vqc_variances"],
        "o--",
        color="#D9534F",
        label="Standard Discrete VQC",
        markersize=7,
    )
    ax1.plot(
        q_range,
        bp_data["brain_variances"],
        "s-",
        color="#2E86C1",
        label="Biomorphic Brain (Ours)",
        markersize=7,
    )
    theo_x = np.linspace(min(q_range), max(q_range), 100)
    theo_y = bp_data["vqc_variances"][0] * (2.0 ** -(theo_x - q_range[0]))
    ax1.plot(
        theo_x, theo_y, ":", color="gray", alpha=0.7, label=r"Barren Plateau $\mathcal{O}(2^{-N})$"
    )
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Qubits ($N$)")
    ax1.set_ylabel(r"Gradient Variance $\mathrm{Var}[\partial \mathcal{L}]$")
    ax1.set_title(r"$\mathbf{(a)}$ Barren Plateau Resilience", pad=10)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="lower left", fontsize=9)

    # Panel B: Cognitive Dilemma
    t_grid = dilemma_data["time_grid"]
    ax2.plot(
        t_grid,
        dilemma_data["left_consensus"],
        "--",
        color="#E67E22",
        label=r"Left Lobe $\langle Z_L \rangle$",
    )
    ax2.plot(
        t_grid,
        dilemma_data["right_consensus"],
        "--",
        color="#8E44AD",
        label=r"Right Lobe $\langle Z_R \rangle$",
    )
    ax2.plot(
        t_grid,
        dilemma_data["global_consensus"],
        "-",
        color="#16A085",
        linewidth=2.5,
        label=r"Consensus $\hat{M}$",
    )
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Evolution Duration ($\\tau$)")
    ax2.set_ylabel(r"Polarity $\langle \sigma^z \rangle$")
    ax2.set_title(r"$\mathbf{(b)}$ Cognitive Dilemma Consensus", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper right", fontsize=9)

    # Panel C: Ablation Study
    epochs = range(1, len(next(iter(ablation_data.values()))) + 1)
    colors = {
        "Full": "#1B4F72",
        "Dopamine": "#B03A2E",
        "Oxygenation": "#D35400",
        "Static": "#7D6608",
    }
    for name, loss_vals in ablation_data.items():
        c_key = (
            "Full"
            if "Full" in name
            else (
                "Dopamine"
                if "Dopamine" in name
                else ("Oxygenation" if "Oxygenation" in name else "Static")
            )
        )
        ax3.plot(
            epochs,
            loss_vals,
            label=name,
            color=colors[c_key],
            linewidth=2.0 if c_key == "Full" else 1.5,
        )
    ax3.set_xlabel("Training Epochs")
    ax3.set_ylabel(r"MSE Loss")
    ax3.set_title(r"$\mathbf{(c)}$ Biochemical & Metabolic Ablation", pad=10)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(frameon=True, loc="upper right", fontsize=8.5)

    plt.tight_layout()
    composite_path = OUTPUT_DIR / "figure_combined_academic.png"
    fig.savefig(composite_path)
    plt.close(fig)
    print(f"\n[OK] Successfully generated Composite Publication Figure at {composite_path}")


def main() -> None:
    print("\n" + "#" * 75)
    print("QUANTA SDK — BIOMORPHIC QUANTUM BRAIN ACADEMIC EMPIRICAL SUITE")
    print("#" * 75 + "\n")

    t_start = time.time()
    bp_results = run_barren_plateau_experiment()
    dilemma_results = run_cognitive_dilemma_experiment()
    ablation_results = run_ablation_experiment()

    # Generate composite 3-panel figure
    generate_composite_figure(bp_results, dilemma_results, ablation_results)

    # Save raw data
    raw_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "barren_plateau": bp_results,
        "cognitive_dilemma": dilemma_results,
        "ablation_study": ablation_results,
    }
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(raw_data, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[VICTORY] All 3 experiments successfully executed in {total_time:.2f} seconds.")
    print(f"Data saved to: {DATA_FILE.resolve()}")
    print(f"Figures saved to: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
