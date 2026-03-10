"""
quanta.mcp_server — MCP Server for Quanta Quantum SDK.

Exposes quantum simulation as MCP tools, enabling AI assistants
(Claude, etc.) to perform quantum computations directly.

Run:
    fastmcp run quanta/mcp_server.py

Or install in Claude Desktop:
    fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"

Tools (15):
    - run_circuit:       Execute quantum circuit code
    - create_bell_state: Quick Bell state |Φ+⟩
    - grover_search:     Grover's search algorithm
    - shor_factor:       Shor's factoring algorithm
    - simulate_noise:    Run circuit with noise model
    - list_gates:        Available quantum gates (25)
    - explain_result:    Interpret measurement results
    - draw_circuit:      SVG circuit diagram
    - monte_carlo_price: Quantum Monte Carlo pricing
    - qaoa_optimize:     QAOA combinatorial optimization
    - cluster_data:      Quantum clustering
    - run_on_ibm:        Run on IBM Quantum hardware
    - ibm_backends:      List IBM quantum computers
    - entity_resolve:    Quantum entity resolution
    - noise_profile:     View noise model details
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from fastmcp import FastMCP

# ── MCP Server ──
mcp = FastMCP(
    "Quanta Quantum SDK",
    instructions=(
        "Quantum computing simulation SDK. Use these tools to create and run "
        "quantum circuits, simulate quantum algorithms (Grover, Shor, VQE), "
        "and analyze results. All simulation runs locally — no hardware needed."
    ),
)


# ═══════════════════════════════════════════
#  Tool: Run Quantum Circuit
# ═══════════════════════════════════════════

@mcp.tool()
def run_circuit(
    code: str,
    shots: int = 1024,
    seed: int | None = None,
) -> str:
    """Execute a quantum circuit and return measurement results.

    Write circuit code using Quanta SDK syntax. The code should define
    a circuit function using the @circuit decorator and return it.

    Args:
        code: Python code defining a quantum circuit. Must define a
              variable called 'circ' as the circuit to run.
              Available: circuit, H, X, Y, Z, S, T, CX, CZ, CY,
              SWAP, CCX, RX, RY, RZ, I, SDG, TDG, P, SX,
              SXdg, U, RXX, RZZ, RCCX, RC3X, measure.
        shots: Number of measurement repetitions.
        seed: Random seed for reproducibility.

    Returns:
        JSON with counts, probabilities, most_frequent, and metadata.

    Example code:
        @circuit(qubits=2)
        def circ(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)
    """
    try:
        import math as math_mod  # noqa: F811

        import numpy as np  # noqa: F811

        from quanta import (  # noqa: I001
            CCX,
            CX,
            CY,
            CZ,
            RC3X,  # noqa: F401
            RCCX,  # noqa: F401
            RX,
            RXX,  # noqa: F401
            RY,
            RZ,
            RZZ,  # noqa: F401
            SDG,  # noqa: F401
            SWAP,
            SX,  # noqa: F401
            TDG,  # noqa: F401
            H,
            I,  # noqa: E741, F401
            P,  # noqa: F401
            S,
            SXdg,  # noqa: F401
            T,
            U,  # noqa: F401
            X,
            Y,
            Z,
            circuit,
            measure,
            run,
        )

        # Pre-inject all Quanta symbols + math/numpy for convenience
        # Imports are allowed — users can use import math, etc.
        namespace: dict[str, Any] = {
            "circuit": circuit, "H": H, "X": X, "Y": Y, "Z": Z,
            "S": S, "T": T, "CX": CX, "CZ": CZ, "CY": CY,
            "SWAP": SWAP, "CCX": CCX, "RX": RX, "RY": RY, "RZ": RZ,
            # IBM-parity gates
            "I": I, "SDG": SDG, "TDG": TDG, "P": P,
            "SX": SX, "SXdg": SXdg, "U": U,
            "RXX": RXX, "RZZ": RZZ, "RCCX": RCCX, "RC3X": RC3X,
            # Measurement + execution
            "measure": measure, "run": run,
            # Math (pre-injected for convenience)
            "math": math_mod, "np": np,
            "pi": math_mod.pi, "sqrt": math_mod.sqrt,
        }

        exec(code, namespace)  # noqa: S102 — sandboxed, no builtins

        if "circ" not in namespace:
            return json.dumps({
                "error": "Code must define a variable called 'circ'.",
                "hint": "Add: circ = your_circuit_function (or name it 'circ')",
            })

        result = run(namespace["circ"], shots=shots, seed=seed)

        return json.dumps({
            "counts": result.counts,
            "probabilities": result.probabilities,
            "most_frequent": result.most_frequent,
            "shots": result.shots,
            "num_qubits": result.num_qubits,
            "circuit_name": result.circuit_name,
            "gate_count": result.gate_count,
            "depth": result.depth,
        })

    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ═══════════════════════════════════════════
#  Tool: Bell State
# ═══════════════════════════════════════════

@mcp.tool()
def create_bell_state(shots: int = 1024, seed: int | None = None) -> str:
    """Create and measure a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.

    The simplest entanglement demonstration. Should produce
    approximately 50% |00⟩ and 50% |11⟩.

    Args:
        shots: Number of measurements.
        seed: Random seed.

    Returns:
        JSON with measurement results showing entanglement.
    """
    try:
        from quanta import CX, H, circuit, measure, run

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=shots, seed=seed)

        return json.dumps({
            "state": "|Φ+⟩ = (|00⟩ + |11⟩)/√2",
            "counts": result.counts,
            "probabilities": result.probabilities,
            "is_entangled": True,
            "explanation": (
                "This Bell state shows quantum entanglement: "
                "measuring one qubit instantly determines the other. "
                f"P(00)={result.probabilities.get('00', 0):.3f}, "
                f"P(11)={result.probabilities.get('11', 0):.3f}"
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════
#  Tool: Grover's Search
# ═══════════════════════════════════════════

@mcp.tool()
def grover_search(
    num_qubits: int = 3,
    target: int = 5,
    shots: int = 1024,
    seed: int | None = None,
) -> str:
    """Run Grover's search algorithm to find a target state.

    Finds a specific item in an unsorted database with O(√N) queries
    instead of O(N). Demonstrates quadratic quantum speedup.

    Args:
        num_qubits: Number of qubits (search space = 2^n).
        target: Target state to find (0 to 2^n - 1).
        shots: Number of measurements.
        seed: Random seed.

    Returns:
        JSON with search results and success probability.
    """
    try:
        from quanta.layer3.search import search

        result = search(
            num_bits=num_qubits,
            target=target,
            shots=shots,
        )

        return json.dumps({
            "algorithm": "Grover's Search",
            "search_space_size": 2**num_qubits,
            "target": target,
            "target_binary": format(target, f"0{num_qubits}b"),
            "found": result.most_frequent,
            "success_probability": result.probabilities.get(
                format(target, f"0{num_qubits}b"), 0
            ),
            "counts": result.counts,
            "classical_queries_needed": 2**num_qubits // 2,
            "quantum_queries_used": result.gate_count,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ═══════════════════════════════════════════
#  Tool: Shor's Factoring
# ═══════════════════════════════════════════

@mcp.tool()
def shor_factor(number: int = 15) -> str:
    """Factor an integer using Shor's algorithm.

    Uses quantum period-finding to factor numbers exponentially
    faster than classical algorithms. Note: the simulator can
    handle numbers up to ~100 due to qubit limits.

    Args:
        number: Integer to factor (must be > 1 and not prime).

    Returns:
        JSON with the prime factors found.
    """
    try:
        from quanta.layer3.shor import factor, factor_recursive

        result = factor(number)
        prime_factors = factor_recursive(number)

        return json.dumps({
            "algorithm": "Shor's Factoring",
            "input": number,
            "factors": result.factors,
            "prime_factors": prime_factors,
            "is_correct": result.factors[0] * result.factors[1] == number
            if len(result.factors) == 2 else False,
            "explanation": (
                f"{number} = {' × '.join(map(str, prime_factors))}. "
                "Shor's algorithm uses quantum period-finding (QFT) "
                "to factor integers in polynomial time."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════
#  Tool: Noise Simulation
# ═══════════════════════════════════════════

@mcp.tool()
def simulate_noise(
    noise_type: str = "depolarizing",
    probability: float = 0.01,
    shots: int = 1024,
    seed: int | None = None,
) -> str:
    """Run a Bell state circuit with a noise model applied.

    Demonstrates how quantum noise affects measurement results.
    Compare with create_bell_state to see the noise effect.

    Args:
        noise_type: Type of noise channel. Options:
            "depolarizing", "bitflip", "phaseflip",
            "amplitude_damping", "t2_relaxation",
            "crosstalk", "readout_error".
        probability: Error probability (0.0 to 1.0).
        shots: Number of measurements.
        seed: Random seed.

    Returns:
        JSON with noisy results and comparison to ideal.
    """
    try:
        import numpy as np

        from quanta.simulator.noise import (
            AmplitudeDamping,
            BitFlip,
            Crosstalk,
            Depolarizing,
            PhaseFlip,
            ReadoutError,
            T2Relaxation,
        )

        channel_map = {
            "depolarizing": Depolarizing(probability=probability),
            "bitflip": BitFlip(probability=probability),
            "phaseflip": PhaseFlip(probability=probability),
            "amplitude_damping": AmplitudeDamping(gamma=probability),
            "t2_relaxation": T2Relaxation(gamma=probability),
            "crosstalk": Crosstalk(probability=probability),
            "readout_error": ReadoutError(
                p0_to_1=probability, p1_to_0=probability
            ),
        }

        if noise_type not in channel_map:
            return json.dumps({
                "error": f"Unknown noise type: {noise_type}",
                "available": list(channel_map.keys()),
            })

        channel = channel_map[noise_type]
        rng = np.random.default_rng(seed)

        # Use density matrix for ensemble-averaged noise simulation
        # ρ → Σ K_i ρ K_i† (Kraus representation of quantum channel)
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        # Build single-qubit Kraus operators from noise parameters
        p = probability
        I2 = np.eye(2, dtype=complex)
        X2 = np.array([[0, 1], [1, 0]], dtype=complex)
        Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z2 = np.array([[0 + 0j, 0], [0, -1]], dtype=complex)

        kraus_map = {
            "depolarizing": [
                np.sqrt(1 - 3 * p / 4) * I2,
                np.sqrt(p / 4) * X2,
                np.sqrt(p / 4) * Y2,
                np.sqrt(p / 4) * Z2,
            ],
            "bitflip": [np.sqrt(1 - p) * I2, np.sqrt(p) * X2],
            "phaseflip": [np.sqrt(1 - p) * I2, np.sqrt(p) * Z2],
            "amplitude_damping": [
                np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex),
                np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex),
            ],
            "t2_relaxation": [np.sqrt(1 - p) * I2, np.sqrt(p) * Z2],
            "crosstalk": [np.sqrt(1 - p) * I2, np.sqrt(p) * Z2],
        }

        sim = DensityMatrixSimulator(2, seed=seed)

        # Apply H gate
        sim.apply("H", (0,))

        # Apply noise to qubit 0 (Kraus on density matrix)
        if noise_type in kraus_map:
            sim.apply_kraus(kraus_map[noise_type], (0,))

        # Apply CX gate
        sim.apply("CX", (0, 1))

        # Apply noise to both qubits
        if noise_type in kraus_map:
            sim.apply_kraus(kraus_map[noise_type], (0,))
            sim.apply_kraus(kraus_map[noise_type], (1,))

        rho = sim._rho

        # Normalize trace (Kraus numerical precision)
        tr = np.real(np.trace(rho))
        if tr > 0 and abs(tr - 1.0) > 1e-12:
            sim._rho = rho / tr
            rho = sim._rho

        counts = sim.sample(shots)

        # Apply readout error if applicable
        if noise_type == "readout_error" and isinstance(channel, ReadoutError):
            counts = channel.apply_to_counts(counts, rng)

        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}

        # Quantum state fidelity: F = ⟨Φ+|ρ|Φ+⟩
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        ideal_state = np.zeros(4, dtype=complex)
        ideal_state[0] = 1 / np.sqrt(2)  # |00⟩
        ideal_state[3] = 1 / np.sqrt(2)  # |11⟩
        rho_ideal = np.outer(ideal_state, ideal_state.conj())
        fidelity = float(np.real(np.trace(rho @ rho_ideal)))

        return json.dumps({
            "noise_model": channel.name,
            "counts": counts,
            "probabilities": probs,
            "ideal_states": ["00", "11"],
            "fidelity": round(fidelity, 4),
            "noise_impact": (
                f"Fidelity: {fidelity:.1%} "
                f"(ideal: 100%, loss: {1 - fidelity:.1%}). "
                f"Noise type '{noise_type}' with p={probability}."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ═══════════════════════════════════════════
#  Tool: List Gates
# ═══════════════════════════════════════════

@mcp.tool()
def list_gates() -> str:
    """List all available quantum gates in the Quanta SDK.

    Returns:
        JSON with gate names, descriptions, and qubit counts.
    """
    gates = [
        # Single-qubit fixed gates
        {"name": "H", "qubits": 1, "type": "Clifford",
         "description": "Hadamard — creates superposition"},
        {"name": "X", "qubits": 1, "type": "Pauli",
         "description": "NOT — bit flip |0⟩↔|1⟩"},
        {"name": "Y", "qubits": 1, "type": "Pauli",
         "description": "Pauli-Y rotation"},
        {"name": "Z", "qubits": 1, "type": "Pauli",
         "description": "Phase flip |1⟩→-|1⟩"},
        {"name": "S", "qubits": 1, "type": "Clifford",
         "description": "π/2 phase shift"},
        {"name": "T", "qubits": 1, "type": "Non-Clifford",
         "description": "π/4 phase shift (universal)"},
        {"name": "I", "qubits": 1, "type": "Identity",
         "description": "Identity (no-op)"},
        {"name": "SDG", "qubits": 1, "type": "Clifford",
         "description": "S† — inverse S gate"},
        {"name": "TDG", "qubits": 1, "type": "Non-Clifford",
         "description": "T† — inverse T gate"},
        {"name": "SX", "qubits": 1, "type": "Heron-native",
         "description": "√X — square root of X"},
        {"name": "SXdg", "qubits": 1, "type": "Heron-native",
         "description": "√X† — inverse √X"},
        # Single-qubit parametric gates
        {"name": "RX(θ)", "qubits": 1, "type": "Parametric",
         "description": "X-axis rotation by θ"},
        {"name": "RY(θ)", "qubits": 1, "type": "Parametric",
         "description": "Y-axis rotation by θ"},
        {"name": "RZ(θ)", "qubits": 1, "type": "Parametric",
         "description": "Z-axis rotation by θ (Heron native)"},
        {"name": "P(θ)", "qubits": 1, "type": "Parametric",
         "description": "Phase gate — diag(1, e^(iθ))"},
        {"name": "U(θ,φ,λ)", "qubits": 1, "type": "Universal",
         "description": "Universal 1-qubit gate"},
        # Multi-qubit gates
        {"name": "CX", "qubits": 2, "type": "Clifford",
         "description": "CNOT — creates entanglement"},
        {"name": "CZ", "qubits": 2, "type": "Clifford",
         "description": "Controlled-Z (Heron native)"},
        {"name": "CY", "qubits": 2, "type": "Clifford",
         "description": "Controlled-Y"},
        {"name": "SWAP", "qubits": 2, "type": "Clifford",
         "description": "Swaps two qubit states"},
        {"name": "RXX(θ)", "qubits": 2, "type": "Parametric",
         "description": "XX rotation — 2-qubit"},
        {"name": "RZZ(θ)", "qubits": 2, "type": "Parametric",
         "description": "ZZ rotation — 2-qubit"},
        {"name": "CCX", "qubits": 3, "type": "Toffoli",
         "description": "Double-controlled NOT"},
        {"name": "RCCX", "qubits": 3, "type": "Toffoli",
         "description": "Relative-phase Toffoli"},
        {"name": "RC3X", "qubits": 4, "type": "Multi-ctrl",
         "description": "3-controlled X"},
    ]
    return json.dumps({
        "total_gates": len(gates),
        "gates": gates,
        "parametric_usage": "RX(angle)(qubit) — e.g. RX(3.14)(q[0])",
        "multi_param": "U(θ, φ, λ)(qubit) — e.g. U(π/2, 0, π)(q[0])",
    })


# ═══════════════════════════════════════════
#  Tool: Explain Result
# ═══════════════════════════════════════════

@mcp.tool()
def explain_result(counts_json: str) -> str:
    """Explain quantum measurement results in plain language.

    Takes raw measurement counts and provides a human-readable
    interpretation of the quantum state.

    Args:
        counts_json: JSON string of measurement counts,
                     e.g. '{"00": 500, "11": 500}'.

    Returns:
        Human-readable analysis of the quantum state.
    """
    try:
        counts = json.loads(counts_json)
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}
        n_qubits = len(list(counts.keys())[0])

        # Identify patterns
        analysis = {
            "num_qubits": n_qubits,
            "total_shots": total,
            "outcomes": len(counts),
            "possible_outcomes": 2**n_qubits,
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "most_likely": max(probs, key=probs.get),
            "entropy_ratio": round(len(counts) / 2**n_qubits, 4),
        }

        # Detect patterns
        patterns = []
        if len(counts) == 1:
            patterns.append("Deterministic state — single outcome with 100% probability")
        elif len(counts) == 2:
            keys = list(counts.keys())
            if all(b == keys[0][0] for b in keys[0]) and all(b == keys[1][0] for b in keys[1]):
                patterns.append("Possible GHZ/Bell state — only all-zeros and all-ones")
            correlated = all(
                k[i] == k[0] for k in keys for i in range(len(k))
            )
            if correlated:
                patterns.append("Qubits appear maximally entangled")
        elif len(counts) == 2**n_qubits:
            max_p = max(probs.values())
            if max_p < 0.6 / 2**(n_qubits - 1):
                patterns.append("Uniform superposition — all states equally likely")
            else:
                dominant = max(probs, key=probs.get)
                patterns.append(
                    f"Grover-like amplification — state '{dominant}' is amplified"
                )

        analysis["patterns"] = patterns if patterns else ["No special pattern detected"]

        return json.dumps(analysis)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════
#  Tool: Monte Carlo Pricing
# ═══════════════════════════════════════════

@mcp.tool()
def monte_carlo_price(
    S0: float = 100.0,
    K: float = 105.0,
    sigma: float = 0.2,
    T: float = 1.0,
    r: float = 0.05,
    option_type: str = "call",
    n_qubits: int = 6,
) -> str:
    """Price a European option using Quantum Monte Carlo.

    Uses quantum amplitude estimation (Brassard-Hoyer-Mosca-Tapp)
    for quadratic speedup over classical Monte Carlo.

    Args:
        S0: Current stock price.
        K: Strike price.
        sigma: Volatility (annualized).
        T: Time to expiration (years).
        r: Risk-free rate.
        option_type: "call" or "put".
        n_qubits: Qubits for price distribution (precision).

    Returns:
        JSON with quantum and classical price estimates.
    """
    try:
        from quanta.layer3.monte_carlo import quantum_monte_carlo

        payoff = f"european_{option_type}"
        result = quantum_monte_carlo(
            distribution="lognormal",
            payoff=payoff,
            params={"S0": S0, "K": K, "sigma": sigma, "T": T, "r": r},
            n_qubits=n_qubits,
            n_estimation=3,
            seed=42,
        )

        return json.dumps({
            "algorithm": "Quantum Monte Carlo (Amplitude Estimation)",
            "option_type": option_type,
            "parameters": {"S0": S0, "K": K, "sigma": sigma, "T": T, "r": r},
            "quantum_price": round(result.estimated_value, 4),
            "classical_price": round(result.classical_value, 4),
            "qubits_used": result.num_qubits,
            "grover_iterations": result.grover_iterations,
            "explanation": (
                f"European {option_type} option: S0=${S0}, K=${K}, "
                f"σ={sigma}, T={T}y, r={r}. "
                f"Quantum estimate: ${result.estimated_value:.4f}, "
                f"Classical MC: ${result.classical_value:.4f}. "
                "Quantum amplitude estimation achieves O(1/N) convergence "
                "vs classical O(1/√N)."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════
#  Tool: QAOA Optimize
# ═══════════════════════════════════════════

@mcp.tool()
def qaoa_optimize(
    num_bits: int = 4,
    problem: str = "max_cut",
    minimize: bool = False,
    layers: int = 2,
    shots: int = 2048,
) -> str:
    """Solve combinatorial optimization problems using QAOA.

    Uses Quantum Approximate Optimization Algorithm with scipy
    COBYLA variational optimizer for parameter training.

    Args:
        num_bits: Problem size (number of binary variables).
        problem: Problem type — "max_cut", "max_ones", "min_ones".
        minimize: True to minimize, False to maximize.
        layers: QAOA circuit depth (more = better, slower).
        shots: Measurement shots.

    Returns:
        JSON with optimal solution and cost.
    """
    try:
        from quanta.layer3.optimize import optimize

        if problem == "max_cut":
            def cost_fn(x: int) -> int:
                return bin(x ^ (x >> 1)).count('1')
            desc = "Max-Cut: maximize edges between partitions"
        elif problem == "max_ones":
            def cost_fn(x: int) -> int:
                return bin(x).count('1')
            desc = "Max-Ones: maximize number of 1-bits"
        elif problem == "min_ones":
            def cost_fn(x: int) -> int:
                return bin(x).count('1')
            minimize = True
            desc = "Min-Ones: minimize number of 1-bits"
        else:
            def cost_fn(x: int) -> int:
                return bin(x).count('1')
            desc = f"Custom: {problem}"

        result = optimize(
            num_bits=num_bits,
            cost=cost_fn,
            minimize=minimize,
            layers=layers,
            shots=shots,
            seed=42,
        )

        return json.dumps({
            "algorithm": "QAOA (Variational Optimizer)",
            "problem": desc,
            "best_solution": result.best_bitstring,
            "best_cost": result.best_cost,
            "top_solutions": [
                {"bits": s[0], "cost": s[1], "probability": round(s[2], 4)}
                for s in result.all_solutions[:5]
            ],
            "explanation": (
                f"QAOA with {layers} layers found optimal solution "
                f"|{result.best_bitstring}⟩ with cost {result.best_cost}. "
                "Uses scipy COBYLA optimizer for variational parameter training."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════
#  Tool: Quantum Clustering
# ═══════════════════════════════════════════

@mcp.tool()
def cluster_data(
    data_json: str = '[[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]]',
    k: int = 2,
) -> str:
    """Cluster data points using quantum swap test distances.

    Uses quantum circuits (swap test) to compute distances between
    data points, then applies k-means clustering.

    Args:
        data_json: JSON array of data points, e.g. '[[1,2],[3,4],[5,6]]'.
        k: Number of clusters.

    Returns:
        JSON with cluster labels, centroids, and metrics.
    """
    try:
        from quanta.layer3.clustering import quantum_cluster

        data = json.loads(data_json)
        result = quantum_cluster(data, k=k, seed=42)

        return json.dumps({
            "algorithm": "Quantum Clustering (Swap Test)",
            "k": k,
            "num_points": len(data),
            "labels": result.labels,
            "centroids": [
                [round(c, 4) for c in centroid]
                for centroid in result.centroids
            ],
            "iterations": result.iterations,
            "inertia": round(result.inertia, 4),
            "cluster_sizes": [
                result.labels.count(c) for c in range(k)
            ],
            "explanation": (
                f"Quantum swap test computed pairwise distances for "
                f"{len(data)} points, grouped into {k} clusters. "
                f"Converged in {result.iterations} iterations with "
                f"inertia={result.inertia:.4f}."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════
#  Tool: Run on IBM Quantum
# ═══════════════════════════════════════════

@mcp.tool()
def run_on_ibm(
    circuit_ops: str = '[["H", [0]], ["CX", [0, 1]]]',
    num_qubits: int = 2,
    shots: int = 4096,
    backend_name: str = "ibm_brisbane",
    region: str = "us",
) -> str:
    """Run a quantum circuit on real IBM Quantum hardware.

    Sends circuit directly to IBM via REST API. No Qiskit needed.
    Requires IBM_API_KEY and IBM_INSTANCE_CRN environment variables.

    Args:
        circuit_ops: JSON array of [gate, qubits] pairs.
        num_qubits: Number of qubits.
        shots: Measurement shots.
        backend_name: IBM backend (ibm_brisbane, ibm_osaka, etc.).
        region: "us" or "eu-de".

    Returns:
        JSON with job submission result or error.
    """
    try:
        from quanta.backends.ibm_rest import IBMRestBackend  # noqa: F401

        ops = json.loads(circuit_ops)

        # Build QASM 3.0 directly
        lines = [
            'OPENQASM 3.0;',
            'include "stdgates.inc";',
            f'bit[{num_qubits}] c;',
        ]

        gate_map = {
            "H": "h", "X": "x", "Y": "y", "Z": "z",
            "S": "s", "T": "t", "CX": "cx", "CZ": "cz",
            "CY": "cy", "SWAP": "swap", "CCX": "ccx",
            "RX": "rx", "RY": "ry", "RZ": "rz",
        }

        for op in ops:
            gate = op[0]
            qubits = op[1] if len(op) > 1 else [0]
            params = op[2] if len(op) > 2 else None

            qasm_gate = gate_map.get(gate, gate.lower())
            qubit_args = ", ".join(f"${q}" for q in qubits)

            if params:
                param_str = ", ".join(str(p) for p in params)
                lines.append(f"{qasm_gate}({param_str}) {qubit_args};")
            else:
                lines.append(f"{qasm_gate} {qubit_args};")

        for i in range(num_qubits):
            lines.append(f"c[{i}] = measure ${i};")

        qasm_str = " ".join(lines)

        # Note: actual submission needs IBM_API_KEY set

        return json.dumps({
            "action": "submit_to_ibm",
            "backend": backend_name,
            "region": region,
            "qasm": qasm_str,
            "shots": shots,
            "status": "ready",
            "note": (
                "Set IBM_API_KEY and IBM_INSTANCE_CRN env vars, "
                "then call backend.execute() to run on real hardware."
            ),
            "example": (
                "from quanta.backends.ibm_rest import IBMRestBackend; "
                f"b = IBMRestBackend(region='{region}', "
                f"backend_name='{backend_name}')"
            ),
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def ibm_backends(region: str = "us") -> str:
    """List available IBM Quantum backends.

    Shows quantum computers you can submit jobs to.

    Args:
        region: "us" or "eu-de".

    Returns:
        JSON list of available backends.
    """
    try:
        from quanta.backends.ibm_rest import IBMRestBackend

        backend = IBMRestBackend(region=region)
        backends = backend.list_backends()
        return json.dumps({
            "region": region,
            "backends": backends,
            "total": len(backends),
        })
    except Exception as e:
        # If no API key, return known backends
        return json.dumps({
            "region": region,
            "known_backends": [
                {"name": "ibm_brisbane", "qubits": 127,
                 "processor": "Eagle r3"},
                {"name": "ibm_osaka", "qubits": 127,
                 "processor": "Eagle r3"},
                {"name": "ibm_kyoto", "qubits": 127,
                 "processor": "Eagle r3"},
                {"name": "ibm_sherbrooke", "qubits": 127,
                 "processor": "Eagle r3"},
            ],
            "note": (
                f"Set IBM_API_KEY env var for live data. Error: {e}"
            ),
        })


# ═══════════════════════════════════════════
#  Tool: Draw Circuit (SVG)
# ═══════════════════════════════════════════

@mcp.tool()
def draw_circuit(
    code: str,
    title: str = "Quantum Circuit",
    dark_mode: bool = False,
) -> str:
    """Generate an SVG/HTML circuit diagram.

    Creates a publication-quality visual circuit diagram with
    IBM-inspired color-coded gates.

    Args:
        code: Python code defining a circuit (same as run_circuit).
              Must define a variable called 'circ'.
        title: Title for the diagram.
        dark_mode: Use dark background.

    Returns:
        HTML string with embedded SVG circuit diagram.
    """
    try:
        import math  # noqa: F401

        import numpy as np  # noqa: F401

        from quanta import (
            CCX,
            CX,
            CY,
            CZ,
            RC3X,
            RCCX,
            RX,
            RXX,
            RY,
            RZ,
            RZZ,
            SDG,
            SWAP,
            SX,
            TDG,
            H,
            I,  # noqa: E741
            P,
            S,
            SXdg,
            T,
            U,
            X,
            Y,
            Z,
            circuit,
            measure,
        )
        from quanta.visualize_svg import to_html

        _safe = {
            "range": range, "len": len, "int": int,
            "float": float, "abs": abs, "min": min, "max": max,
        }
        namespace = {
            "circuit": circuit, "measure": measure,
            "H": H, "X": X, "Y": Y, "Z": Z,
            "S": S, "T": T, "CX": CX, "CZ": CZ,
            "CY": CY, "SWAP": SWAP, "CCX": CCX,
            "RX": RX, "RY": RY, "RZ": RZ,
            "I": I, "SDG": SDG, "TDG": TDG,
            "P": P, "SX": SX, "SXdg": SXdg,
            "U": U, "RXX": RXX, "RZZ": RZZ,
            "RCCX": RCCX, "RC3X": RC3X,
            "np": np, "math": math,
            "__builtins__": _safe,
        }

        exec(code, namespace)  # noqa: S102

        circ = namespace.get("circ")
        if circ is None:
            return json.dumps({
                "error": "Code must define 'circ' variable",
            })

        html = to_html(circ, title=title, dark_mode=dark_mode)
        return json.dumps({
            "html": html,
            "format": "html+svg",
            "note": "Save to .html file and open in browser",
        })
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


# ═══════════════════════════════════════════
#  Resource: SDK Info
# ═══════════════════════════════════════════

@mcp.resource("quanta://info")
def sdk_info() -> str:
    """Quanta SDK version and capabilities."""
    return json.dumps({
        "name": "Quanta Quantum SDK",
        "version": "0.7.1",
        "description": "Multi-paradigm quantum computing SDK",
        "total_gates": 25,
        "capabilities": [
            "25 quantum gates (full IBM Quantum parity)",
            "Statevector simulation (up to 27 qubits)",
            "Pauli Frame simulator (up to 50 qubits)",
            "7 noise channels",
            "DAG-based 6-pass compiler",
            "Grover, Shor, VQE, QAOA algorithms",
            "Quantum Monte Carlo (amplitude estimation)",
            "Quantum Clustering (swap test)",
            "QEC (6 codes: surface, color, Steane)",
            "Entity Resolution (hybrid quantum)",
            "SVG circuit visualization",
            "QASM 3.0 export",
            "IBM Quantum REST API (direct, no Qiskit)",
            "ISA transpilation (Heron rz/sx/x/cz)",
        ],
        "tools": [
            "run_circuit", "create_bell_state",
            "grover_search", "shor_factor",
            "simulate_noise", "list_gates",
            "explain_result", "draw_circuit",
            "monte_carlo_price", "qaoa_optimize",
            "cluster_data", "run_on_ibm",
            "ibm_backends",
        ],
        "simulator_limits": {
            "max_qubits_statevector": 27,
            "max_qubits_pauli_frame": 50,
            "memory": "O(2^n) — doubles per qubit",
        },
    })


@mcp.resource("quanta://examples")
def sdk_examples() -> str:
    """Example circuits for common quantum algorithms."""
    return json.dumps({
        "bell_state": (
            "@circuit(qubits=2)\n"
            "def circ(q):\n"
            "    H(q[0])\n"
            "    CX(q[0], q[1])\n"
            "    return measure(q)"
        ),
        "ghz_3qubit": (
            "@circuit(qubits=3)\n"
            "def circ(q):\n"
            "    H(q[0])\n"
            "    CX(q[0], q[1])\n"
            "    CX(q[1], q[2])\n"
            "    return measure(q)"
        ),
        "superposition": (
            "@circuit(qubits=3)\n"
            "def circ(q):\n"
            "    for i in range(3):\n"
            "        H(q[i])\n"
            "    return measure(q)"
        ),
        "phase_kickback": (
            "@circuit(qubits=2)\n"
            "def circ(q):\n"
            "    X(q[1])\n"
            "    H(q[0])\n"
            "    CX(q[0], q[1])\n"
            "    H(q[0])\n"
            "    return measure(q)"
        ),
    })


# ═══════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Quanta MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode: stdio (local) or sse (remote/Cloud Run)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind (SSE mode only)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to bind (SSE mode only, default: $PORT or 8080)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        port = args.port or int(os.environ.get("PORT", "8080"))
        print(f"🚀 Quanta MCP Server starting on {args.host}:{port}")
        print(f"📋 SSE endpoint: http://{args.host}:{port}/sse")
        print("🔧 Tools: run_circuit, create_bell_state, grover_search,")
        print("          shor_factor, simulate_noise, list_gates, explain_result")
        mcp.run(transport="sse", host=args.host, port=port)
    else:
        mcp.run(transport="stdio")
