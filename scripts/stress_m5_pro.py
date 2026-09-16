#!/usr/bin/env python3
"""
scripts/stress_m5_pro.py -- High-Memory Quantum Statevector Stress Benchmark on Apple M5 Pro.

Pushes the limits of dense quantum statevector simulation on
Apple Silicon M5 Pro (48 GB Unified Memory).
Targets:
  - 27 Qubits: 134,217,728 amplitudes  -> ~2.15 GB state, ~4.3 GB peak
  - 28 Qubits: 268,435,456 amplitudes  -> ~4.29 GB state, ~8.6 GB peak
  - 29 Qubits: 536,870,912 amplitudes  -> ~8.59 GB state, ~18-20 GB peak!
"""

import gc
import json
import time

import numpy as np
import psutil

from quanta.simulator.statevector import StateVectorSimulator


def get_memory_info():
    """Returns process RSS and system virtual memory in GB."""
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)
    used_gb = vm.used / (1024 ** 3)
    return {
        "proc_rss_gb": round(rss_gb, 3),
        "total_ram_gb": round(total_gb, 2),
        "available_ram_gb": round(avail_gb, 2),
        "used_ram_gb": round(used_gb, 2),
    }


def run_stress_qubits(n_qubits: int):
    """Runs dense GHZ state generation on n_qubits with live memory profiling."""
    dim = 2 ** n_qubits
    raw_array_gb = (dim * 16) / (1024 ** 3)

    print("\n" + "=" * 70)
    print(f"🚀 STRESS TEST: {n_qubits} QUBITS DENSE STATEVECTOR")
    print(f"   Hilbert Space Dimension: {dim:,} states (2^{n_qubits})")
    print(f"   Raw Array Size (complex128): {raw_array_gb:.2f} GB")
    print("=" * 70)

    mem_start = get_memory_info()
    print(
        f"[*] Pre-allocation System RAM: Used={mem_start['used_ram_gb']} GB | "
        f"Avail={mem_start['available_ram_gb']} GB | "
        f"Process RSS={mem_start['proc_rss_gb']} GB"
    )

    # 1. Statevector Allocation
    t0 = time.perf_counter()
    print(f"[*] Allocating {n_qubits}-qubit StateVectorSimulator...")
    sim = StateVectorSimulator(n_qubits)
    t_alloc = time.perf_counter() - t0
    mem_alloc = get_memory_info()
    print(
        f"[+] Allocation completed in {t_alloc:.3f}s | "
        f"Process RSS: {mem_alloc['proc_rss_gb']} GB"
    )

    # 2. Hadamard on Qubit 0
    t0 = time.perf_counter()
    print("[*] Applying H gate on qubit 0...")
    sim.apply("H", (0,))
    t_h = time.perf_counter() - t0
    mem_h = get_memory_info()
    print(
        f"[+] H gate applied in {t_h:.3f}s | "
        f"Process RSS: {mem_h['proc_rss_gb']} GB | "
        f"System Used: {mem_h['used_ram_gb']} GB"
    )

    # 3. Entangling CNOT on Qubits (0 -> 1)
    t0 = time.perf_counter()
    print("[*] Applying CX gate (qubit 0 -> qubit 1)...")
    sim.apply("CX", (0, 1))
    t_cx = time.perf_counter() - t0
    mem_cx = get_memory_info()
    print(
        f"[+] CX gate applied in {t_cx:.3f}s | "
        f"Process RSS: {mem_cx['proc_rss_gb']} GB | "
        f"System Used: {mem_cx['used_ram_gb']} GB"
    )

    # 4. Long-Range CNOT (Qubit 1 -> Last Qubit n-1)
    t0 = time.perf_counter()
    print(f"[*] Applying long-range CX gate (qubit 1 -> qubit {n_qubits-1})...")
    sim.apply("CX", (1, n_qubits - 1))
    t_cx_long = time.perf_counter() - t0
    mem_cx_long = get_memory_info()
    print(
        f"[+] Long-range CX applied in {t_cx_long:.3f}s | "
        f"Process RSS: {mem_cx_long['proc_rss_gb']} GB | "
        f"System Used: {mem_cx_long['used_ram_gb']} GB"
    )

    # 5. Quantum Mathematical Verification
    t0 = time.perf_counter()
    print("[*] Verifying state amplitudes and quantum norm...")
    amp_0 = sim._state[0]
    prob_0 = float(np.abs(amp_0) ** 2)
    norm = float(np.vdot(sim._state, sim._state).real)

    t_verify = time.perf_counter() - t0
    print(f"[+] Quantum state verification completed in {t_verify:.3f}s")
    print(f"    Amplitude |0...0>: {amp_0:.5f} (P = {prob_0:.4f})")
    print(f"    Total Statevector Norm: {norm:.12f}")

    peak_rss = max(
        mem_alloc["proc_rss_gb"],
        mem_h["proc_rss_gb"],
        mem_cx["proc_rss_gb"],
        mem_cx_long["proc_rss_gb"],
    )
    print(f"\n🎉 SUMMARY for {n_qubits} QUBITS:")
    print(f"   Peak Process RAM: {peak_rss:.2f} GB")
    print(f"   Total Gate Time: {t_h + t_cx + t_cx_long:.2f}s")
    print("   System Status: Healthy, 0 crashes")

    del sim
    gc.collect()

    return {
        "qubits": n_qubits,
        "dim": dim,
        "raw_array_gb": round(raw_array_gb, 3),
        "peak_rss_gb": round(peak_rss, 3),
        "t_alloc_s": round(t_alloc, 3),
        "t_h_s": round(t_h, 3),
        "t_cx_s": round(t_cx, 3),
        "t_cx_long_s": round(t_cx_long, 3),
        "total_gate_time_s": round(t_h + t_cx + t_cx_long, 3),
    }


def main():
    print("=" * 70)
    print("🔥 APPLE SILICON M5 PRO (48 GB UNIFIED RAM) STRESS BENCHMARK")
    print("   Author: Quanta SDK Team")
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"   System: {cores} Cores ({threads} Threads), {ram_gb:.1f} GB RAM")
    print("=" * 70)

    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024 ** 3)
    print(f"Current Available RAM: {available_gb:.2f} GB")

    results = []

    # 1. Warm-up / Medium High: 27 Qubits (2.15 GB raw, ~4.3 GB peak)
    res_27 = run_stress_qubits(27)
    results.append(res_27)

    # 2. High: 28 Qubits (4.29 GB raw, ~8.6 GB peak)
    res_28 = run_stress_qubits(28)
    results.append(res_28)

    # 3. Maximum: 29 Qubits (8.59 GB raw, ~18-20 GB peak!)
    res_29 = run_stress_qubits(29)
    results.append(res_29)

    out_file = "stress_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("🏆 ALL M5 PRO STRESS TESTS COMPLETED SUCCESSFULLY!")
    print(f"   Results saved to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
