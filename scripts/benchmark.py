#!/usr/bin/env python3
"""
Quanta SDK Benchmark Suite

Compares Quanta vs Qiskit/Cirq performance on standard quantum tasks.
Outputs results to docs/BENCHMARK.md and stdout.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, ".")

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    quanta_ms: float
    qiskit_ms: float | None = None
    cirq_ms: float | None = None
    note: str = ""


def _time_fn(fn, runs=5) -> float:
    """Returns median time in ms."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return float(np.median(times))


# ═══════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════


def bench_bell_state() -> BenchmarkResult:
    """Build + simulate Bell state."""
    from quanta import circuit, H, CX, measure, run

    @circuit(qubits=2)
    def bell(q):
        H(q[0])
        CX(q[0], q[1])
        return measure(q)

    ms = _time_fn(lambda: run(bell, shots=1024))
    return BenchmarkResult("Bell State (2q, 1024 shots)", ms, note="Build+DAG+sim+sample")


def bench_ghz_10() -> BenchmarkResult:
    """10-qubit GHZ state."""
    from quanta import circuit, H, CX, measure, run

    @circuit(qubits=10)
    def ghz10(q):
        H(q[0])
        for i in range(9):
            CX(q[i], q[i + 1])
        return measure(q)

    ms = _time_fn(lambda: run(ghz10, shots=1024))
    return BenchmarkResult("GHZ State (10q)", ms, note="2^10 = 1024 amplitudes")


def bench_ghz_20() -> BenchmarkResult:
    """20-qubit GHZ state — stress test."""
    from quanta import circuit, H, CX, measure, run

    @circuit(qubits=20)
    def ghz20(q):
        H(q[0])
        for i in range(19):
            CX(q[i], q[i + 1])
        return measure(q)

    ms = _time_fn(lambda: run(ghz20, shots=1024), runs=3)
    return BenchmarkResult("GHZ State (20q)", ms, note="2^20 = 1M amplitudes")


def bench_random_circuit() -> BenchmarkResult:
    """Random 8-qubit circuit with 50 gates."""
    from quanta import circuit, H, CX, RZ, S, T, measure, run

    @circuit(qubits=8)
    def random8(q):
        for i in range(8):
            H(q[i])
        for i in range(0, 7, 2):
            CX(q[i], q[i + 1])
        for i in range(8):
            RZ(0.5)(q[i])
        for i in range(1, 8, 2):
            CX(q[i - 1], q[i])
        for i in range(4):
            S(q[i])
            T(q[i + 4])
        for i in range(0, 6, 2):
            CX(q[i], q[i + 2])
        return measure(q)

    ms = _time_fn(lambda: run(random8, shots=4096))
    return BenchmarkResult("Random Circuit (8q, ~50 gates)", ms, note="4096 shots")


def bench_grover_4() -> BenchmarkResult:
    """Grover search on 4 qubits."""
    from quanta.layer3.search import search

    ms = _time_fn(lambda: search(num_bits=4, target=7, seed=42))
    return BenchmarkResult("Grover Search (4q)", ms, note="Target=7")


def bench_grover_8() -> BenchmarkResult:
    """Grover search on 8 qubits."""
    from quanta.layer3.search import search

    ms = _time_fn(lambda: search(num_bits=8, target=42, seed=42), runs=3)
    return BenchmarkResult("Grover Search (8q)", ms, note="Target=42, 256 states")


def bench_vqe_h2() -> BenchmarkResult:
    """VQE for H₂ molecule."""
    from quanta.layer3.vqe import vqe

    hamiltonian = [("ZZ", -1.0), ("XX", 0.5), ("YY", 0.5)]
    ms = _time_fn(lambda: vqe(num_qubits=2, hamiltonian=hamiltonian, layers=2, seed=42), runs=3)
    return BenchmarkResult("VQE H₂ (2q, 2 layers)", ms, note="3-term Hamiltonian")


def bench_estimator() -> BenchmarkResult:
    """Estimator primitive: Bell state ⟨ZZ⟩."""
    from quanta import circuit, H, CX, measure
    from quanta.primitives import Estimator

    @circuit(qubits=2)
    def bell(q):
        H(q[0])
        CX(q[0], q[1])
        return measure(q)

    est = Estimator(seed=42)
    ms = _time_fn(lambda: est.run(bell, observables=[[("ZZ", 1.0)]]))
    return BenchmarkResult("Estimator ⟨ZZ⟩ (2q)", ms, note="Exact statevector")


def bench_parameter_shift() -> BenchmarkResult:
    """Parameter-shift gradient."""
    from quanta import H, CX, RY, measure
    from quanta.core.quantum import quantum

    @quantum(qubits=2, observable=[("ZZ", 1.0)])
    def ansatz(q, theta=0.0, phi=0.0):
        RY(theta)(q[0])
        RY(phi)(q[1])
        CX(q[0], q[1])
        return measure(q)

    ms = _time_fn(lambda: ansatz.gradient(theta=0.5, phi=0.3), runs=3)
    return BenchmarkResult("Parameter-Shift Gradient (2q, 2 params)", ms, note="5 circuit evals")


def bench_sampler_batch() -> BenchmarkResult:
    """Sampler: 10 circuits batch."""
    from quanta import circuit, H, CX, RZ, measure
    from quanta.primitives import Sampler

    circuits = []
    for i in range(10):
        @circuit(qubits=2)
        def c(q, angle=float(i)):
            H(q[0])
            RZ(angle)(q[0])
            CX(q[0], q[1])
            return measure(q)
        circuits.append(c)

    sampler = Sampler(seed=42)
    ms = _time_fn(lambda: sampler.run(circuits, shots=1024), runs=3)
    return BenchmarkResult("Sampler Batch (10×2q)", ms, note="10 circuits × 1024 shots")


# ═══════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════


def main():
    print("⚛️  Quanta SDK Benchmark Suite")
    print(f"   Python {sys.version.split()[0]}")
    print(f"   NumPy {np.__version__}")
    print("=" * 60)

    benchmarks = [
        bench_bell_state,
        bench_ghz_10,
        bench_ghz_20,
        bench_random_circuit,
        bench_grover_4,
        bench_grover_8,
        bench_vqe_h2,
        bench_estimator,
        bench_parameter_shift,
        bench_sampler_batch,
    ]

    results: list[BenchmarkResult] = []
    for fn in benchmarks:
        try:
            r = fn()
            results.append(r)
            print(f"  ✅ {r.name:45s} {r.quanta_ms:8.2f} ms")
        except Exception as e:
            print(f"  ❌ {fn.__name__:45s} ERROR: {e}")

    # Generate markdown
    md_lines = [
        "# Quanta SDK Benchmark Results\n",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"> Python {sys.version.split()[0]}, NumPy {np.__version__}\n",
        "## Results\n",
        "| Benchmark | Time (ms) | Notes |",
        "|-----------|----------|-------|",
    ]

    for r in results:
        md_lines.append(f"| {r.name} | {r.quanta_ms:.2f} | {r.note} |")

    md_lines.extend([
        "",
        "## Methodology\n",
        "- Each benchmark runs 3–5 iterations, median reported",
        "- Includes full pipeline: circuit build → DAG → simulation → sampling",
        "- No hardware; pure statevector simulation",
        "- Parameter-shift gradient includes 5 circuit evaluations (cost + 2×2 shifts)",
        "",
        "## Hardware\n",
        f"- Python: {sys.version.split()[0]}",
        f"- NumPy: {np.__version__}",
        "- All tests on CPU (no GPU acceleration)",
        "",
    ])

    with open("docs/BENCHMARK.md", "w") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 60)
    print(f"📊 Results written to docs/BENCHMARK.md")
    print(f"   {len(results)}/{len(benchmarks)} benchmarks completed")


if __name__ == "__main__":
    main()
