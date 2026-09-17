"""Biologically Realistic Noisy Hippocampal Buffer Benchmark.

Evaluates:
1. Experiment A: Temporal Engram Degradation under Thermal Noise & Lindblad Phase Diffusion
   - Traces fidelity F(t) = |<psi_0 | psi(t)>|^2 over 24h of biological wakefulness.
   - Evaluates dopamine tagging protection: Low (D=0.1), Medium (D=1.0), High (D=3.0).
2. Experiment B: Quantum Phase Diffusion Angle Distribution
   - Tracks the angular dispersion of quantum phase fluctuations across engram coordinates.
3. Experiment C: Sharp-Wave Ripple (SWR) Replay Sampling vs. Salience
   - Evaluates Boltzmann probability sampling of engrams across dopamine tags at
     varying temperatures.
4. Experiment D: Sleep Consolidation Resilience under Noisy vs. Pristine Replay
   - Benchmarks QuantumREMSleep orthogonalization on:
     a) No Sleep (Baseline)
     b) Idealized / Pristine Replay (Lossless)
     c) Noisy Hippocampal Replay (Biologically Realistic Degraded States)

Generates:
- Publication Figure 7: `docs/paper/figures/fig7_hippocampal_buffer.png` (300 DPI, 4 panels)
- Telemetry addition: `docs/paper/benchmark_academic_data.json`
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from quanta.torch import (
    BiomorphicResonantBrain,
    NoisyHippocampalBuffer,
    QuantumREMSleep,
)

# Plot styling matching Figs 1-6
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
        "lines.linewidth": 2.2,
    }
)


def run_hippocampal_benchmark() -> dict[str, Any]:
    print("=" * 70)
    print("BIOMORPHIC QUANTUM BRAIN: NOISY HIPPOCAMPAL BUFFER BENCHMARK")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Experiment A: Engram Degradation Trajectory (Fidelity vs. Time)
    # -------------------------------------------------------------------------
    print("\n[1/4] Running Experiment A: Engram Decay Trajectories (Dopamine Tagging)...")
    time_steps = np.linspace(0, 24, 49)  # 0 to 24 hours in 30-min steps
    dt = 0.5  # 30 minutes

    # Set up buffer with 3 engrams of differing dopamine valence
    dim = 16
    buf = NoisyHippocampalBuffer(
        capacity=10,
        noise_level=0.03,
        phase_diffusion_rate=0.04,
        temporal_decay_rate=0.02,
        dopamine_protection=2.0,
    )

    psi_base = torch.randn(3, dim, dtype=torch.float32)
    psi_base = psi_base / torch.linalg.norm(psi_base, dim=-1, keepdim=True)
    psi_c = psi_base.to(dtype=buf.complex_dtype)

    # Store: 0: Low Dopamine (D=0.1), 1: Moderate (D=1.0), 2: Salient (D=3.0)
    dopamine_levels = [0.1, 1.0, 3.0]
    for i, d in enumerate(dopamine_levels):
        buf.store(psi_c[i], dopamine_tag=d, metadata={"level": d})

    fidelities_history: dict[str, list[float]] = {
        "Low (D=0.1)": [1.0],
        "Moderate (D=1.0)": [1.0],
        "Salient (D=3.0)": [1.0],
    }

    # Track angular phase dispersion
    phase_spreads: list[float] = [0.0]

    for _ in time_steps[1:]:
        buf.step(dt=dt)
        fids = buf.get_fidelities()
        fidelities_history["Low (D=0.1)"].append(fids[0])
        fidelities_history["Moderate (D=1.0)"].append(fids[1])
        fidelities_history["Salient (D=3.0)"].append(fids[2])

        # Compute phase drift of the low dopamine state
        st0 = buf.buffer[0]["pristine_state"]
        stt = buf.buffer[0]["degraded_state"]
        phases = torch.angle(stt * torch.conj(st0)).cpu().numpy()
        phase_spreads.append(float(np.std(phases)))

    r_low = fidelities_history["Low (D=0.1)"][-1] * 100
    r_mod = fidelities_history["Moderate (D=1.0)"][-1] * 100
    r_high = fidelities_history["Salient (D=3.0)"][-1] * 100
    print(f"  24h Retention Low Dopamine  (D=0.1): {r_low:.2f}%")
    print(f"  24h Retention Mod Dopamine  (D=1.0): {r_mod:.2f}%")
    print(f"  24h Retention High Dopamine (D=3.0): {r_high:.2f}%")

    # -------------------------------------------------------------------------
    # Experiment B: SWR Replay Sampling vs. Temperature
    # -------------------------------------------------------------------------
    print("\n[2/4] Running Experiment B: Sharp-Wave Ripple Sampling Dynamics...")
    # Buffer with 10 engrams having linear dopamine tags from 0.2 to 4.0
    swr_buf = NoisyHippocampalBuffer(capacity=20)
    tags = np.linspace(0.2, 4.0, 10)
    for t in tags:
        vec = torch.randn(dim, dtype=torch.float32)
        vec = vec / torch.linalg.norm(vec)
        swr_buf.store(vec.to(dtype=swr_buf.complex_dtype), dopamine_tag=float(t))

    temperatures = [0.3, 1.0, 3.0]
    sampling_distributions: dict[str, list[float]] = {}

    for T in temperatures:
        counts = np.zeros(10)
        n_samples = 2000
        for _ in range(n_samples):
            # Sample 1 engram
            replayed = swr_buf.replay(batch_size=1, temperature=T)
            # Find matching index
            for idx, e in enumerate(swr_buf.buffer):
                if torch.allclose(replayed[0], e["degraded_state"]):
                    counts[idx] += 1
                    break
        sampling_distributions[f"T = {T}"] = (counts / n_samples).tolist()

    # -------------------------------------------------------------------------
    # Experiment C: Sleep Consolidation Resilience under Biological Noise
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Experiment C: Sleep Consolidation Robustness...")
    brain = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)

    # Generate 4 task prototype memory representations
    task_inputs = [
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=brain.real_dtype),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=brain.real_dtype),
        torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=brain.real_dtype),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=brain.real_dtype),
    ]

    pristine_states = []
    with torch.no_grad():
        for x_in in task_inputs:
            out = brain(x_in)
            pristine_states.append(out["state"][0].clone())

    # Initial baseline overlap across the 4 tasks
    pair_count = 6
    init_overlap = 0.0
    for j in range(4):
        for k in range(j + 1, 4):
            ov = float(
                (torch.abs(torch.vdot(pristine_states[j], pristine_states[k])) ** 2).item()
            )
            init_overlap += ov
    init_overlap /= pair_count

    # Scenario 1: Sleep consolidation with Pristine states (Lossless replay)
    sleep_clean = QuantumREMSleep(brain, sleep_cycles=15, learning_rate=0.02)
    clean_res = sleep_clean.sleep(memory_states=pristine_states, cycles=15)
    clean_history = clean_res["loss_history"]

    # Scenario 2: Sleep consolidation with Noisy Hippocampal buffer states
    # Store into buffer, degrade over 6 hours of wakefulness, then consolidate
    noisy_buf = NoisyHippocampalBuffer(
        capacity=10,
        noise_level=0.08,
        phase_diffusion_rate=0.08,
        temporal_decay_rate=0.03,
        dopamine_protection=1.5,
    )
    for s in pristine_states:
        noisy_buf.store(s, dopamine_tag=1.5)

    # Advance 6 biological hours
    for _ in range(12):
        noisy_buf.step(dt=0.5)

    mean_fid_before_sleep = noisy_buf.get_mean_fidelity()
    print(f"  Hippocampal mean engram fidelity before sleep: {mean_fid_before_sleep*100:.2f}%")

    brain_noisy = BiomorphicResonantBrain(in_features=4, num_left_qubits=2, num_right_qubits=2)
    sleep_noisy = QuantumREMSleep(brain_noisy, sleep_cycles=15, learning_rate=0.02)
    noisy_res = noisy_buf.consolidate_with_sleep(sleep_noisy, cycles=15)
    noisy_history = noisy_res["loss_history"]

    print(f"  Initial Mean Task Overlap: {init_overlap:.4f}")
    print(f"  Clean Replay Final Overlap: {clean_history[-1]:.4f}")
    print(f"  Noisy Replay Final Overlap: {noisy_history[-1]:.4f}")

    # -------------------------------------------------------------------------
    # Generate 4-Panel Publication Figure 7
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating 300 DPI Publication Figure 7...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Biological Noisy Hippocampal Buffer: Lindblad Phase Diffusion & SWR Consolidation",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Engram Fidelity vs Time
    ax_a = axes[0, 0]
    ax_a.plot(
        time_steps,
        [f * 100 for f in fidelities_history["Low (D=0.1)"]],
        color="#d9534f",
        linestyle="--",
        label="Incidental / Noise (D=0.1)",
    )
    ax_a.plot(
        time_steps,
        [f * 100 for f in fidelities_history["Moderate (D=1.0)"]],
        color="#f0ad4e",
        linestyle="-.",
        label="Standard Event (D=1.0)",
    )
    ax_a.plot(
        time_steps,
        [f * 100 for f in fidelities_history["Salient (D=3.0)"]],
        color="#2b6cb0",
        linestyle="-",
        label="Dopamine Tagged / Salient (D=3.0)",
    )
    ax_a.axhline(50, color="gray", linestyle=":", alpha=0.7, label="50% Coherence Half-life")
    ax_a.set_title("(a) Engram Fidelity Decay over 24h Wakefulness", fontweight="bold")
    ax_a.set_xlabel("Elapsed Time t (Hours)")
    ax_a.set_ylabel(r"Engram Fidelity $\mathcal{F}(t) = |\langle\psi_0|\psi(t)\rangle|^2$ (%)")
    ax_a.set_ylim(20, 105)
    ax_a.legend(loc="lower left", frameon=True)

    # Panel B: Phase Diffusion Dispersion
    ax_b = axes[0, 1]
    ax_b.plot(
        time_steps,
        phase_spreads,
        color="#805ad5",
        label=r"Phase Standard Deviation $\sigma_\phi(t)$",
    )
    # Fit theoretical sqrt(t) Brownian curve
    sqrt_fit = phase_spreads[-1] * np.sqrt(time_steps / 24.0)
    ax_b.plot(
        time_steps,
        sqrt_fit,
        color="#4a5568",
        linestyle=":",
        label=r"Brownian Diffusion $\propto \sqrt{\Delta t}$",
    )
    ax_b.fill_between(time_steps, 0, phase_spreads, color="#805ad5", alpha=0.15)
    ax_b.set_title(r"(b) Quantum Phase Angular Dispersion $\Delta \theta$", fontweight="bold")
    ax_b.set_xlabel("Elapsed Time t (Hours)")
    ax_b.set_ylabel(r"Phase Angle Dispersion $\sigma_\theta$ (rad)")
    ax_b.legend(loc="upper left", frameon=True)

    # Panel C: SWR Replay Sampling vs Dopamine
    ax_c = axes[1, 0]
    x_pos = np.arange(10)
    width = 0.25
    ax_c.bar(
        x_pos - width,
        sampling_distributions["T = 0.3"],
        width,
        label="T = 0.3 (Focused / Salient)",
        color="#2b6cb0",
    )
    ax_c.bar(
        x_pos,
        sampling_distributions["T = 1.0"],
        width,
        label="T = 1.0 (Balanced)",
        color="#48bb78",
    )
    ax_c.bar(
        x_pos + width,
        sampling_distributions["T = 3.0"],
        width,
        label="T = 3.0 (Uniform Diffusion)",
        color="#ecc94b",
    )
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([f"{d:.1f}" for d in tags])
    ax_c.set_title("(c) SWR Replay Probability vs. Dopamine Salience Tag", fontweight="bold")
    ax_c.set_xlabel("Dopaminergic Salience Tag D")
    ax_c.set_ylabel("Replay Sampling Probability")
    ax_c.legend(loc="upper left", frameon=True)

    # Panel D: Sleep Consolidation Resilience
    ax_d = axes[1, 1]
    cycles = list(range(len(clean_history)))
    ax_d.plot(
        cycles,
        clean_history,
        color="#2b6cb0",
        marker="o",
        label="Idealized Lossless Replay (Clean)",
    )
    ax_d.plot(
        cycles,
        noisy_history,
        color="#e53e3e",
        marker="s",
        label=f"Noisy Hippocampal Replay (F={mean_fid_before_sleep:.2f})",
    )
    ax_d.axhline(
        init_overlap,
        color="black",
        linestyle="--",
        label=f"Wake Baseline Overlap ({init_overlap:.3f})",
    )
    ax_d.set_title("(d) Sleep Orthogonalization Convergence", fontweight="bold")
    ax_d.set_xlabel("Sleep Annealing Cycle")
    ax_d.set_ylabel(r"Pairwise Subspace Overlap $\mathcal{L}_{\text{ortho}}$")
    ax_d.legend(loc="upper right", frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_dir = Path("docs/paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fig7_hippocampal_buffer.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure saved to: {fig_path}")

    # Copy to artifacts directory
    artifact_dir = Path("/Users/aes/.gemini/antigravity/brain/5551c804-c729-424c-a867-fd3c7b746f0c")
    if artifact_dir.exists():
        shutil.copy(fig_path, artifact_dir / "fig7_hippocampal_buffer.png")

    # Update telemetry JSON
    telemetry_path = Path("docs/paper/benchmark_academic_data.json")
    if telemetry_path.exists():
        with open(telemetry_path) as f:
            data = json.load(f)
    else:
        data = {}

    data["hippocampal_buffer"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_grid_hours": time_steps.tolist(),
        "fidelity_trajectories": fidelities_history,
        "phase_spread_rad": phase_spreads,
        "swr_sampling_probabilities": sampling_distributions,
        "consolidation_comparison": {
            "initial_overlap": init_overlap,
            "clean_history": clean_history,
            "noisy_history": noisy_history,
            "mean_fidelity_pre_sleep": mean_fid_before_sleep,
        },
    }

    with open(telemetry_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Telemetry updated in: {telemetry_path}")

    return data["hippocampal_buffer"]


if __name__ == "__main__":
    run_hippocampal_benchmark()
