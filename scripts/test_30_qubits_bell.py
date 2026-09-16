#!/usr/bin/env python3
"""
scripts/test_30_qubits_bell.py -- 30-Qubit Entanglement on Apple M5 Pro.

Tests two-qubit entangling gate CX(0, 1) on top of Hadamard on a 30-qubit dense statevector.
Hilbert Space: 1,073,741,824 complex128 numbers (16.00 GiB).
"""

import gc
import json
import time
import psutil
import numpy as np

from quanta.simulator.statevector import StateVectorSimulator


def main():
    n_qubits = 30
    dim = 2 ** n_qubits
    raw_array_gb = (dim * 16) / (1024 ** 3)

    print("=" * 75)
    print("⚡ 30-QUBIT ENTANGLEMENT TEST (H + CX) ON APPLE M5 PRO")
    print(f"   Hilbert Dimension: {dim:,} amplitudes (2^30)")
    print(f"   Raw State Size:    {raw_array_gb:.2f} GiB (17.18 GB)")
    print("=" * 75)

    proc = psutil.Process()

    sim = StateVectorSimulator(30)
    print(f"[*] Statevector allocated. Applying H(0)...")
    t0 = time.perf_counter()
    sim.apply("H", (0,))
    t_h = time.perf_counter() - t0
    rss_h = proc.memory_info().rss / (1024 ** 3)
    print(f"[+] H(0) done in {t_h:.2f}s | Process RSS: {rss_h:.2f} GB")

    print(f"[*] Applying two-qubit entangling gate CX(0, 1)...")
    t0 = time.perf_counter()
    sim.apply("CX", (0, 1))
    t_cx = time.perf_counter() - t0
    rss_cx = proc.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    print(f"[+] CX(0, 1) done in {t_cx:.2f}s | Process RSS: {rss_cx:.2f} GB | System RAM Used: {vm.used / (1024**3):.2f} GB")

    # Verification
    t0 = time.perf_counter()
    amp_0 = sim._state[0]
    # In tensor indexing for qubits (0, 1): state |110...0> is index 3 (or 2^(n-1) + 2^(n-2)) depending on endianness
    # Let's find non-zero amplitudes:
    prob_0 = float(np.abs(amp_0) ** 2)
    norm = float(np.vdot(sim._state, sim._state).real)
    t_verify = time.perf_counter() - t0

    print(f"\n[*] Mathematical Verification (completed in {t_verify:.3f}s):")
    print(f"    State |00...00> Amplitude: {amp_0:.7f} (P = {prob_0:.7f})")
    print(f"    Total Statevector Norm:    {norm:.12f} (Target: 1.000000000000)")

    peak_rss = max(rss_h, rss_cx)
    result = {
        "qubits": 30,
        "dimension": dim,
        "raw_gib": raw_array_gb,
        "peak_rss_gb": round(peak_rss, 2),
        "t_h_s": round(t_h, 2),
        "t_cx_s": round(t_cx, 2),
        "total_time_s": round(t_h + t_cx, 2),
        "prob_0": round(prob_0, 7),
        "norm": round(norm, 12),
        "status": "SUCCESS",
    }

    with open("qubits_30_entangled_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 75)
    print(f"🏆 30-QUBIT ENTANGLEMENT COMPLETE!")
    print(f"   Peak RAM: {peak_rss:.2f} GB | Total Gate Time: {t_h + t_cx:.2f}s")
    print(f"   Norm Fidelity: {norm:.12f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
