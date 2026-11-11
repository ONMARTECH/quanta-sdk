"""
Example 10: Quantum SDK Quality Benchmark (Litmus Test)

This is the "turnusol test" — 8 fundamental quantum computing
benchmarks that any serious quantum SDK must pass correctly.

Each test validates a core quantum principle:
  1. Bell Fidelity         → Entanglement works correctly
  2. CHSH Inequality       → Quantum mechanics is real (S > 2)
  3. Teleportation         → Full mid-circuit measurement + feedforward transfer
  4. Grover Amplification  → Gate-compiled quadratic speedup verified
  5. VQE Convergence       → Variational algorithms work
  6. Shor Factoring        → Semiclassical QPE period finding & number theory
  7. QSVM Classification   → Dual QP Quantum ML kernel works
  8. Surface Code          → Exact GF(2) homology fault tolerance suppresses errors

Running:
    python -m quanta.examples.10_quantum_benchmark
"""

import time

import numpy as np

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def test_bell_fidelity() -> tuple[bool, str]:
    """Test 1: Bell state has perfect entanglement (F = 1.0)."""
    from quanta import CX, H, circuit, measure, run

    @circuit(qubits=2)
    def bell(q):
        H(q[0])
        CX(q[0], q[1])
        return measure(q)

    result = run(bell, shots=10000, seed=42)
    p00 = result.probabilities.get("00", 0)
    p11 = result.probabilities.get("11", 0)
    p01 = result.probabilities.get("01", 0)
    p10 = result.probabilities.get("10", 0)

    fidelity = p00 + p11
    purity = p01 + p10

    passed = fidelity > 0.99 and purity < 0.01
    detail = f"F={fidelity:.4f}, |01>+|10>={purity:.4f}"
    return passed, detail


def test_chsh_violation() -> tuple[bool, str]:
    """Test 2: CHSH inequality violation (S > 2 proves quantum)."""
    from quanta.simulator.statevector import StateVectorSimulator

    sim = StateVectorSimulator(2, seed=42)
    sim.apply("H", (0,))
    sim.apply("CX", (0, 1))
    state = sim.state

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    B0 = (Z + X) / np.sqrt(2)
    B1 = (Z - X) / np.sqrt(2)

    def expectation(A, B):
        op = np.kron(A, B)
        return float(np.real(state.conj() @ op @ state))

    S = (
        expectation(Z, B0)
        + expectation(Z, B1)
        + expectation(X, B0)
        - expectation(X, B1)
    )

    passed = abs(S) > 2.0
    detail = f"S={abs(S):.4f} (classical limit=2.0, Tsirelson=2.828)"
    return passed, detail


def test_teleportation() -> tuple[bool, str]:
    """Test 3: Quantum teleportation with mid-circuit Bell measurement & feedforward."""
    from quanta.simulator.statevector import StateVectorSimulator

    # Reference target state |psi> = R_Y(0.9) R_Z(0.5) |0>
    sim_ref = StateVectorSimulator(1, seed=42)
    sim_ref.apply("RZ", (0,), (0.5,))
    sim_ref.apply("RY", (0,), (0.9,))
    psi_target = sim_ref.state  # shape (2,)
    rho_target = np.outer(psi_target, psi_target.conj())

    rng = np.random.default_rng(42)
    n_trials = 64
    fidelities = []

    for trial in range(n_trials):
        sim = StateVectorSimulator(3, seed=42 + trial)
        # 1. Prepare |psi> = R_Y(0.9) R_Z(0.5) |0> on q[0]
        sim.apply("RZ", (0,), (0.5,))
        sim.apply("RY", (0,), (0.9,))

        # 2. Entangle Bell pair on (q[1], q[2])
        sim.apply("H", (1,))
        sim.apply("CX", (1, 2))

        # 3. Alice's Bell-basis rotation on (q[0], q[1])
        sim.apply("CX", (0, 1))
        sim.apply("H", (0,))

        # 4. Projective mid-circuit measurement of q[0] and q[1] (MSB indices: 4*m0 + 2*m1 + q2)
        state = sim.state.reshape(2, 2, 2)  # axes: (q0, q1, q2)
        branch_probs = np.sum(np.abs(state) ** 2, axis=2).ravel()
        branch_idx = int(rng.choice(4, p=branch_probs / np.sum(branch_probs)))
        m0, m1 = divmod(branch_idx, 2)

        collapsed = np.zeros_like(state)
        collapsed[m0, m1, :] = state[m0, m1, :]
        collapsed /= np.linalg.norm(collapsed)
        sim.state = collapsed.reshape(-1)

        # 5. Classical feedforward Pauli corrections on Bob's qubit q[2]
        if m1 == 1:
            sim.apply("X", (2,))
        if m0 == 1:
            sim.apply("Z", (2,))

        # 6. Marginal reduced density matrix rho_2 = Tr_{0,1}(|Psi><Psi|) on q[2]
        psi_3 = sim.state.reshape(4, 2)
        rho_q2 = psi_3.T @ psi_3.conj()
        fid = float(np.real(np.trace(rho_q2 @ rho_target)))
        fidelities.append(fid)

    mean_fidelity = float(np.mean(fidelities))
    min_fidelity = float(np.min(fidelities))
    passed = mean_fidelity > 0.98 and min_fidelity > 0.98
    detail = f"F_q2={mean_fidelity:.6f} (min={min_fidelity:.6f}, need >0.98)"
    return passed, detail


def test_grover_amplification() -> tuple[bool, str]:
    """Test 4: Gate-compiled Grover's search finds target with high probability."""
    from quanta.layer3.search import search

    result = search(num_bits=5, targets=[23], shots=2048, seed=42)
    target_prob = result.probabilities.get("10111", 0)

    passed = target_prob > 0.50 and result.classical_setup_queries == 0
    detail = (
        f"Target |10111> prob={target_prob:.1%}, "
        f"oracle_queries={result.oracle_queries}, gates={result.gate_count}"
    )
    return passed, detail


def test_vqe_convergence() -> tuple[bool, str]:
    """Test 5: VQE converges to H2 ground state energy."""
    from quanta.layer3.hamiltonian import molecular_hamiltonian
    from quanta.layer3.vqe import build_hamiltonian_matrix, vqe

    h2 = molecular_hamiltonian("H2")
    result = vqe(
        num_qubits=2,
        hamiltonian=h2.terms,
        layers=3,
        max_iter=200,
        seed=42,
    )
    exact = float(np.linalg.eigvalsh(build_hamiltonian_matrix(h2.terms, 2))[0])

    error = abs(result.energy - exact)
    passed = error < 0.01
    detail = f"E_vqe={result.energy:.6f}, E_exact={exact:.6f}, error={error:.6f} Ha"
    return passed, detail


def test_shor_factoring() -> tuple[bool, str]:
    """Test 6: Semiclassical Shor's algorithm factors 15 = 3 × 5 without fallback."""
    from quanta.layer3.shor import factor

    result = factor(15, seed=42)
    f1, f2 = sorted(result.factors)

    passed = (
        f1 * f2 == 15
        and f1 > 1
        and f2 > 1
        and result.method == "quantum"
        and result.period > 0
    )
    detail = f"15 = {f1} × {f2} (period={result.period}, method={result.method})"
    return passed, detail


def test_qsvm_classification() -> tuple[bool, str]:
    """Test 7: Dual-QP QSVM classifies linearly separable data."""
    from quanta.layer3.qsvm import qsvm_classify

    X_train = [
        [0.1, 0.2],
        [0.2, 0.1],
        [0.15, 0.15],
        [0.8, 0.9],
        [0.9, 0.8],
        [0.85, 0.85],
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    X_test = [[0.1, 0.1], [0.9, 0.9]]
    expected = [0, 1]

    result = qsvm_classify(X_train, y_train, X_test)

    passed = result.predictions == expected and result.accuracy >= 0.8
    detail = (
        f"Pred={result.predictions}, Acc={result.accuracy:.0%}, "
        f"SVs={len(result.support_indices)}"
    )
    return passed, detail


def test_surface_code() -> tuple[bool, str]:
    """Test 8: Surface code suppresses errors below threshold with exact GF(2) homology."""
    from quanta.qec.surface_code import SurfaceCode

    code = SurfaceCode(distance=5)
    result = code.simulate_error_correction(error_rate=0.005, rounds=2000, seed=42)

    suppression = (
        result.physical_error_rate / result.logical_error_rate
        if result.logical_error_rate > 0
        else float("inf")
    )

    passed = result.logical_error_rate < result.physical_error_rate
    detail = (
        f"p_phys={result.physical_error_rate:.3%}, "
        f"p_log={result.logical_error_rate:.4%}, "
        f"suppression={suppression:.0f}x"
    )
    return passed, detail


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║     QUANTA SDK — QUANTUM QUALITY BENCHMARK          ║")
    print("║     (Turnusol Test / Litmus Test)                   ║")
    print("╠══════════════════════════════════════════════════════╣")

    tests = [
        ("Bell State Fidelity", test_bell_fidelity),
        ("CHSH Inequality (S>2)", test_chsh_violation),
        ("Quantum Teleportation", test_teleportation),
        ("Grover Amplification", test_grover_amplification),
        ("VQE Convergence (H2)", test_vqe_convergence),
        ("Shor Factoring (15=3×5)", test_shor_factoring),
        ("QSVM Classification", test_qsvm_classification),
        ("Surface Code QEC", test_surface_code),
    ]

    results = []
    total_time = 0.0

    for i, (name, test_fn) in enumerate(tests, 1):
        start = time.perf_counter()
        try:
            passed, detail = test_fn()
        except Exception as e:
            passed, detail = False, f"ERROR: {e}"
        elapsed = time.perf_counter() - start
        total_time += elapsed

        status = PASS if passed else FAIL
        results.append(passed)
        print(f"║  {i}. {status} {name:<28} {elapsed:.2f}s  ║")
        print(f"║     {detail:<48}║")

    score = sum(results)
    total = len(results)

    print("╠══════════════════════════════════════════════════════╣")
    print(
        f"║  SCORE: {score}/{total}  "
        f"{'🏆 PERFECT' if score == total else '⚠️  NEEDS WORK'}"
        f"{'':>24}║"
    )
    print(f"║  Total time: {total_time:.2f}s{'':>35}║")
    print("╚══════════════════════════════════════════════════════╝")

    if score == total:
        print("\n  This SDK correctly implements all fundamental")
        print("  quantum computing principles. Production-ready. 🎯")


if __name__ == "__main__":
    main()
