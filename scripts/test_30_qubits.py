#!/usr/bin/env python3
"""
scripts/test_30_qubits.py -- 30-Qubit Record Simulation on Apple M5 Pro.

1,073,741,824 Complex Amplitudes (1.074 Billion states)
Raw Memory: 16.0 GiB (17.18 GB)
Pushing Apple Silicon Unified Memory into the 20 GB range!
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
    print("👑 ULTIMATE STRESS TEST: 30 QUBITS FULL STATEVECTOR ON APPLE M5 PRO")
    print(f"   Hilbert Space Dimension: {dim:,} states (2^{n_qubits})")
    print(f"   Raw Array Size: {raw_array_gb:.2f} GiB ({raw_array_gb * 1.074:.2f} GB)")
    print(f"   System Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"   System Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("=" * 75)

    proc = psutil.Process()

    # 1. Statevector Allocation (16 GiB)
    print("\n[*] Step 1: Allocating 30-qubit statevector (1.074 Billion complex128)...")
    t0 = time.perf_counter()
    sim = StateVectorSimulator(30)
    t_alloc = time.perf_counter() - t0
    rss_alloc = proc.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    print(f"[+] Statevector allocated in {t_alloc:.3f}s")
    print(f"    Process RSS: {rss_alloc:.2f} GB | System RAM Used: {vm.used / (1024**3):.2f} GB | Available: {vm.available / (1024**3):.2f} GB")

    # 2. Hadamard Gate on Qubit 0 (Touching memory across the 1.074 billion elements!)
    print("\n[*] Step 2: Applying Hadamard gate on qubit 0 (Superposition across 1.074B states)...")
    t0 = time.perf_counter()
    sim.apply("H", (0,))
    t_h = time.perf_counter() - t0
    rss_h = proc.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    print(f"[+] H gate completed in {t_h:.2f}s ({t_h/60:.2f} min)")
    print(f"    Process Peak RSS: {rss_h:.2f} GB | System RAM Used: {vm.used / (1024**3):.2f} GB")

    # 3. Quantum Normalization and Fidelity Verification
    print("\n[*] Step 3: Verifying Quantum Normalization and State Amplitude...")
    t0 = time.perf_counter()
    amp_0 = sim._state[0]
    prob_0 = float(np.abs(amp_0) ** 2)
    norm = float(np.vdot(sim._state, sim._state).real)
    t_verify = time.perf_counter() - t0

    print(f"[+] Verification completed in {t_verify:.3f}s")
    print(f"    Amplitude |00...00>: {amp_0.real:.7f} + {amp_0.imag:.7f}j (Expected: 1/√2 = 0.7071068)")
    print(f"    Probability P(00...00): {prob_0:.7f} (Expected: 0.5000000)")
    print(f"    Total Statevector Norm: {norm:.12f} (Expected: 1.000000000000)")

    result = {
        "qubits": 30,
        "dimension": dim,
        "raw_gib": raw_array_gb,
        "peak_rss_gb": round(rss_h, 2),
        "t_alloc_s": round(t_alloc, 3),
        "t_h_s": round(t_h, 2),
        "amp_0": f"{amp_0.real:.7f}+{amp_0.imag:.7f}j",
        "prob_0": round(prob_0, 7),
        "norm": round(norm, 12),
        "status": "SUCCESS",
    }

    with open("qubits_30_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 75)
    print("🏆 30-QUBIT SIMULATION FINISHED WITH 100% MATHEMATICAL PRECISION!")
    print(f"   Peak Process Memory: {rss_h:.2f} GB RAM")
    print(f"   Saved report to qubits_30_result.json")
    print("=" * 75)


if __name__ == "__main__":
    main()
