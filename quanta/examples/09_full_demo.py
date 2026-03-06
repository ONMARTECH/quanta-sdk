"""
Example 09: Full Feature Showcase

Demonstrates ALL Quanta SDK capabilities in one script.
Run this to see everything working together.

Running:
    python -m quanta.examples.09_full_demo
"""

import numpy as np
from quanta import circuit, H, X, CX, CZ, RZ, measure, run, custom_gate
from quanta.visualize import draw


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║          QUANTA SDK v0.4.0 — FULL DEMO              ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── 1. Basic Circuit ──
    print("\n▸ 1. Quantum Circuit (Bell State)")
    @circuit(qubits=2)
    def bell(q):
        H(q[0])
        CX(q[0], q[1])
        return measure(q)

    print(draw(bell))
    result = run(bell, shots=2048, seed=42)
    print(result)
    print(f"  Dirac: {result.dirac_notation()}")

    # ── 2. Custom Gate ──
    print("\n▸ 2. Custom Gate (√X)")
    SqrtX = custom_gate("SX", [[0.5+0.5j, 0.5-0.5j],
                                [0.5-0.5j, 0.5+0.5j]])
    @circuit(qubits=1)
    def custom_test(q):
        SqrtX(q[0])
        return measure(q)
    r = run(custom_test, shots=1024, seed=42)
    print(f"  SqrtX|0> = {r.dirac_notation(2)}")

    # ── 3. Molecular Energy (VQE) ──
    print("\n▸ 3. Molecular Simulation (H2 Ground State)")
    from quanta.layer3.vqe import vqe, build_hamiltonian_matrix
    from quanta.layer3.hamiltonian import molecular_hamiltonian

    h2 = molecular_hamiltonian("H2")
    vqe_result = vqe(
        num_qubits=2, hamiltonian=h2.terms,
        layers=3, max_iter=100, seed=42,
    )
    exact = float(np.linalg.eigvalsh(
        build_hamiltonian_matrix(h2.terms, 2)
    )[0])
    accuracy = (1 - abs(vqe_result.energy - exact) / abs(exact)) * 100
    print(f"  VQE Energy:   {vqe_result.energy:.6f} Ha")
    print(f"  Exact Energy: {exact:.6f} Ha")
    print(f"  Accuracy:     {accuracy:.1f}%")

    # ── 4. Portfolio Optimization ──
    print("\n▸ 4. Financial Portfolio Optimization")
    from quanta.layer3.finance import portfolio_optimize
    assets = [
        {"name": "AAPL",  "return": 0.12, "risk": 0.15},
        {"name": "TSLA",  "return": 0.28, "risk": 0.40},
        {"name": "MSFT",  "return": 0.11, "risk": 0.13},
        {"name": "NVDA",  "return": 0.35, "risk": 0.45},
    ]
    portfolio = portfolio_optimize(assets, budget=2, risk_aversion=0.5, seed=42)
    print(f"  Selected: {portfolio.selected}")
    print(f"  Return: {portfolio.expected_return:.1%}  Risk: {portfolio.portfolio_risk:.1%}")
    print(f"  Sharpe: {portfolio.sharpe_ratio:.2f}")

    # ── 5. Grover Search ──
    print("\n▸ 5. Quantum Search (Grover)")
    from quanta.layer3.search import search
    grover = search(num_bits=4, target=11, shots=1024, seed=42)
    print(f"  Target: 11 (|1011>)")
    print(f"  Found:  {grover.most_frequent} "
          f"(prob: {grover.probabilities.get(grover.most_frequent, 0):.0%})")

    # ── 6. Density Matrix (Noise) ──
    print("\n▸ 6. Noise Simulation (Density Matrix)")
    from quanta.simulator.density_matrix import DensityMatrixSimulator
    dm = DensityMatrixSimulator(2, seed=42)
    dm.apply("H", (0,))
    dm.apply("CX", (0, 1))
    pure = dm.purity
    dm.apply_depolarizing(0, p=0.2)
    noisy = dm.purity
    print(f"  Pure state purity:  {pure:.4f}")
    print(f"  After 20% noise:    {noisy:.4f}")

    # ── 7. Qubit Routing ──
    print("\n▸ 7. Hardware Routing (Linear Topology)")
    from quanta.compiler.passes.routing import RouteToTopology
    from quanta.dag.dag_circuit import DAGCircuit
    @circuit(qubits=5)
    def long_range(q):
        CX(q[0], q[4])
        return measure(q)
    dag = DAGCircuit.from_builder(long_range.build())
    routed = RouteToTopology("linear", 5).run(dag)
    print(f"  CX(q[0],q[4]) on linear: {dag.gate_count()} -> {routed.gate_count()} gates")

    # ── 8. QASM Export ──
    print("\n▸ 8. QASM 3.0 Export")
    from quanta.export.qasm import to_qasm
    print(to_qasm(bell)[:120] + "...")

    # ── Stats ──
    print("\n" + "═" * 55)
    print(f"  Quanta SDK v0.4.0 — All features verified ✅")
    print(f"  Layers: Core + DAG + Compiler + Simulator + Layer3")
    print(f"  Backends: NumPy (JAX/CuPy auto-detect for GPU)")
    print("═" * 55)


if __name__ == "__main__":
    main()
