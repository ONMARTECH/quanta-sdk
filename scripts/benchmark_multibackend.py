#!/usr/bin/env python3
"""
scripts/benchmark_multibackend.py -- Comparative Benchmark across Quantum Platforms.

Platforms tested:
1. Quanta SDK on Apple M5 Pro (Dense Statevector, MPS up to 200 qubits, Sparse)
2. Google Cirq (Local Sycamore / QASM engine simulation on M5 Pro)
3. IonQ Cloud (29-qubit cloud simulator via REST API)
4. IBM Quantum Cloud (IAM auth, QASM 3.0 compilation pipeline)
"""

import json
import os
import time
import tracemalloc

from dotenv import load_dotenv

from quanta import CX, H, circuit, measure, run
from quanta.backends.google import GoogleBackend
from quanta.backends.ionq import IonQBackend
from quanta.simulator.mps import MPSSimulator
from quanta.simulator.statevector import StateVectorSimulator

load_dotenv()


def benchmark_quanta_statevector():
    """Benchmark Quanta Dense Statevector on M5 Pro."""
    print("=" * 60)
    print("1. Quanta Dense StateVector Simulator (Apple M5 Pro)")
    print("=" * 60)

    results = []
    qubit_counts = [2, 4, 8, 12, 16, 20, 24, 26]

    for n in qubit_counts:
        tracemalloc.start()
        start = time.perf_counter()

        sim = StateVectorSimulator(n)
        sim.apply("H", [0])
        for q in range(n - 1):
            sim.apply("CX", [q, q + 1])

        elapsed = time.perf_counter() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        results.append({
            "platform": "Quanta Dense (M5 Pro)",
            "qubits": n,
            "gate_count": n,
            "elapsed_ms": round(elapsed * 1000, 2),
            "peak_mem_mb": round(peak_mb, 2),
            "state_size": f"2^{n} = {2**n:,}",
        })
        dim_str = f"{2**n:,}"
        print(
            f"  {n:2d} qubits: {elapsed*1000:7.2f} ms | "
            f"Peak RAM: {peak_mb:6.2f} MB | Dimension: {dim_str}"
        )

    return results


def benchmark_quanta_mps():
    """Benchmark Quanta Matrix Product State (MPS) on M5 Pro."""
    print("\n" + "=" * 60)
    print("2. Quanta Matrix Product State (MPS) Simulator (Apple M5 Pro)")
    print("=" * 60)

    results = []
    qubit_counts = [10, 25, 50, 100, 200]

    for n in qubit_counts:
        tracemalloc.start()
        start = time.perf_counter()

        mps = MPSSimulator(n, chi_max=64)
        mps.apply("H", [0])
        for q in range(n - 1):
            mps.apply("CX", [q, q + 1])

        elapsed = time.perf_counter() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        results.append({
            "platform": "Quanta MPS (M5 Pro)",
            "qubits": n,
            "gate_count": n,
            "elapsed_ms": round(elapsed * 1000, 2),
            "peak_mem_mb": round(peak_mb, 2),
        })
        print(f"  {n:3d} qubits: {elapsed*1000:7.2f} ms | Peak RAM: {peak_mb:6.2f} MB")

    return results


def make_ghz_circuit(num_qubits: int):
    """Creates a parameterized GHZ circuit."""
    @circuit(qubits=num_qubits)
    def ghz(q):
        H(q[0])
        for i in range(num_qubits - 1):
            CX(q[i], q[i + 1])
        return measure(q)

    return ghz


def benchmark_google_cirq():
    """Benchmark Google Cirq local simulation on M5 Pro."""
    print("\n" + "=" * 60)
    print("3. Google Cirq Local Simulation (Sycamore / QASM Engine)")
    print("=" * 60)

    results = []
    qubit_counts = [2, 4, 8, 12, 16, 20]
    backend = GoogleBackend(simulate_locally=True)

    for n in qubit_counts:
        ghz = make_ghz_circuit(n)

        start = time.perf_counter()
        res = run(ghz, shots=1024, backend=backend)
        elapsed = time.perf_counter() - start

        results.append({
            "platform": "Google Cirq (Local M5 Pro)",
            "qubits": n,
            "gate_count": n,
            "elapsed_ms": round(elapsed * 1000, 2),
            "shots": 1024,
            "num_states_measured": len(res.counts),
        })
        states_sample = list(res.counts.keys())[:2]
        print(f"  {n:2d} qubits: {elapsed*1000:7.2f} ms | Shots: 1024 | States: {states_sample}")

    return results


def benchmark_ionq_cloud():
    """Benchmark IonQ Cloud Simulator via REST API."""
    print("\n" + "=" * 60)
    print("4. IonQ Cloud Simulator (29 Qubits Cloud REST API)")
    print("=" * 60)

    results = []
    backend = IonQBackend(target="simulator")

    test_qubits = [2, 3, 5]
    for n in test_qubits:
        ghz = make_ghz_circuit(n)

        print(f"  Submitting {n}-qubit GHZ circuit to IonQ API...")
        start = time.perf_counter()
        res = run(ghz, shots=500, backend=backend)
        elapsed = time.perf_counter() - start

        all_zeros = "0" * n
        all_ones = "1" * n
        c_zeros = res.counts.get(all_zeros, 0)
        c_ones = res.counts.get(all_ones, 0)
        fidelity = (c_zeros + c_ones) / 500

        results.append({
            "platform": "IonQ Cloud Simulator",
            "qubits": n,
            "gate_count": n,
            "elapsed_ms": round(elapsed * 1000, 2),
            "shots": 500,
            "fidelity": fidelity,
            "counts": res.counts,
        })
        print(
            f"  {n:2d} qubits: {elapsed*1000:7.2f} ms | "
            f"Fidelity: {fidelity*100:.1f}% | Counts: {res.counts}"
        )

    return results


def main():
    print("Starting Multi-Backend Quantum Benchmark...\n")
    all_results = {}

    all_results["quanta_dense"] = benchmark_quanta_statevector()
    all_results["quanta_mps"] = benchmark_quanta_mps()
    all_results["google_cirq"] = benchmark_google_cirq()

    if os.environ.get("IONQ_API_KEY"):
        try:
            all_results["ionq_cloud"] = benchmark_ionq_cloud()
        except Exception as e:
            print(f"IonQ benchmark error: {e}")
            all_results["ionq_cloud"] = [{"error": str(e)}]
    else:
        print("\nSkipping IonQ: IONQ_API_KEY not found.")

    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()
