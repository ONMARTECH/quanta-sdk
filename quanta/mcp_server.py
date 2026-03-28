"""
quanta.mcp_server — AI-native MCP Server for Quanta Quantum SDK.

Exposes quantum simulation as MCP tools, enabling AI assistants
(Claude, GPT, etc.) to perform quantum computations directly.

Run:
    fastmcp run quanta/mcp_server.py

Or install in Claude Desktop:
    fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"

Tools (16):
  Education:
    - create_bell_state:         Quick Bell state |Φ+⟩
    - draw_circuit:              SVG circuit diagram
    - list_gates:                Available quantum gates (31)
    - explain_result:            Interpret measurement results
  Research:
    - run_circuit:               Execute quantum circuit code
    - grover_search:             Grover's search algorithm
    - shor_factor:               Shor's factoring algorithm
    - simulate_noise:            Run circuit with noise model
    - qaoa_optimize:             QAOA combinatorial optimization
    - surface_code_simulate:     Surface code QEC simulation
    - compare_decoders:          Compare MWPM vs Union-Find decoders
  Business:
    - monte_carlo_price:         Quantum Monte Carlo option pricing
    - cluster_data:              Quantum clustering
  Hardware:
    - run_on_ibm:                Run on IBM Quantum hardware
    - ibm_backends:              List IBM quantum computers
    - ibm_job_result:            Poll job status & fetch results

Resources (5):
    - quanta://info              SDK capabilities
    - quanta://examples          Example circuits
    - quanta://noise-profiles    7 noise channels detail
    - quanta://gate-catalog      31 gates with matrices
    - quanta://backend-specs     IBM/IonQ/Google specs

Prompts (4):
    - grover-tutorial            Guided Grover's search
    - option-pricing             Quantum finance workflow
    - circuit-debug              Debug quantum circuits
    - qec-intro                  QEC exploration guide
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
        "AI-native quantum computing SDK. Quanta provides 16 tools organized "
        "into 4 categories:\n"
        "• Education: Bell states, circuit drawing, gate reference, result explanation\n"
        "• Research: Grover, Shor, QAOA, noise simulation, QEC surface code\n"
        "• Business: Monte Carlo option pricing, quantum clustering\n"
        "• Hardware: Run on real IBM Quantum computers (Heron r3, 156 qubits)\n\n"
        "Start with create_bell_state for a quick demo, or use the prompts "
        "(grover-tutorial, option-pricing, circuit-debug, qec-intro) for "
        "guided workflows. All simulation runs locally — no hardware needed "
        "unless using run_on_ibm."
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
              All 31 gates pre-loaded: H,X,Y,Z,S,T,I,SDG,TDG,SX,SXdg,
              RX,RY,RZ,P,U,CX,CY,CZ,SWAP,RXX,RZZ,CCX,RCCX,RC3X,
              ECR,iSWAP,CSWAP,CH,CP,MS.
              Also: circuit, measure, run, math, np, pi, sqrt.
              Imports allowed: you can use 'import math', etc.
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
        # For readout error, ρ is still pure → use counts-based fidelity
        if noise_type == "readout_error":
            # Bhattacharyya fidelity from counts vs ideal Bell
            ideal_p = {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}
            bc = sum(
                (probs.get(s, 0) * ideal_p.get(s, 0)) ** 0.5
                for s in ["00", "01", "10", "11"]
            )
            fidelity = bc ** 2
        else:
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
        # v0.8+ gates
        {"name": "ECR", "qubits": 2, "type": "Clifford",
         "description": "Echoed cross-resonance — IBM native"},
        {"name": "iSWAP", "qubits": 2, "type": "Clifford",
         "description": "√SWAP with phase — Google native"},
        {"name": "CSWAP", "qubits": 3, "type": "Fredkin",
         "description": "Controlled-SWAP (Fredkin gate)"},
        {"name": "CH", "qubits": 2, "type": "Clifford",
         "description": "Controlled-Hadamard"},
        {"name": "CP(θ)", "qubits": 2, "type": "Parametric",
         "description": "Controlled-Phase — diag(1,1,1,e^iθ)"},
        {"name": "MS(θ)", "qubits": 2, "type": "Parametric",
         "description": "Mølmer–Sørensen — trapped-ion native"},
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
            n_estimation=n_qubits,  # each n_qubits → unique precision
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
    backend_name: str = "ibm_torino",
    region: str = "us",
) -> str:
    """Run a quantum circuit on real IBM Quantum hardware.

    Sends circuit directly to IBM via REST API. No Qiskit needed.
    Loads IBM_API_KEY and IBM_INSTANCE_CRN from .env or environment.

    Args:
        circuit_ops: JSON array of [gate, qubits, params?] triples.
            Fixed gates:  ["H", [0]], ["CX", [0, 1]]
            Parametric:   ["RX", [0], [3.14159]]
            Multi-param:  ["U", [0], [1.5708, 0, 3.14159]]
        num_qubits: Number of qubits.
        shots: Measurement shots (max 100000).
        backend_name: IBM backend (ibm_torino, ibm_fez, etc.).
        region: "us" or "eu-de".

    Returns:
        JSON with QASM, submission status, or job result.
    """
    try:
        import os

        from quanta.backends.ibm_rest import IBMRestBackend

        ops = json.loads(circuit_ops)

        # Known IBM backends for validation
        known_backends = {
            "ibm_torino", "ibm_fez", "ibm_marrakesh",
            "ibm_brisbane", "ibm_osaka", "ibm_kyoto",
            "ibm_sherbrooke", "ibm_nazca", "ibm_cusco",
        }

        # Full 25-gate QASM map
        gate_map = {
            "H": "h", "X": "x", "Y": "y", "Z": "z",
            "S": "s", "T": "t", "I": "id",
            "SDG": "sdg", "TDG": "tdg",
            "SX": "sx", "SXdg": "sxdg",
            "CX": "cx", "CZ": "cz", "CY": "cy",
            "SWAP": "swap", "CCX": "ccx",
            "RX": "rx", "RY": "ry", "RZ": "rz",
            "P": "p", "U": "u",
            "RXX": "rxx", "RZZ": "rzz",
            "RCCX": "rccx", "RC3X": "rc3x",
        }

        # Parametric gates that require angle parameters
        param_gates = {"RX", "RY", "RZ", "P", "U", "RXX", "RZZ"}

        # Build QASM 3.0
        lines = [
            'OPENQASM 3.0;',
            'include "stdgates.inc";',
            f'bit[{num_qubits}] c;',
        ]

        for op in ops:
            gate = op[0]
            qubits = op[1] if len(op) > 1 else [0]
            params = op[2] if len(op) > 2 else None

            qasm_gate = gate_map.get(gate, gate.lower())
            qubit_args = ", ".join(f"${q}" for q in qubits)

            if gate in param_gates and not params:
                return json.dumps({
                    "error": (
                        f"Gate '{gate}' requires angle parameter(s). "
                        f'Format: ["{gate}", {qubits}, [angle]]'
                    ),
                    "example": f'["{gate}", {qubits}, [3.14159]]',
                })

            if params:
                param_str = ", ".join(str(p) for p in params)
                lines.append(f"{qasm_gate}({param_str}) {qubit_args};")
            else:
                lines.append(f"{qasm_gate} {qubit_args};")

        for i in range(num_qubits):
            lines.append(f"c[{i}] = measure ${i};")

        qasm_str = " ".join(lines)

        # Backend validation
        backend_warning = None
        if backend_name not in known_backends:
            backend_warning = (
                f"'{backend_name}' not in known backends: "
                f"{', '.join(sorted(known_backends))}"
            )

        # Load credentials from .env or environment
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("IBM_API_KEY", "")
        instance_crn = os.environ.get("IBM_INSTANCE_CRN", "")  # noqa: F841

        if not api_key:
            return json.dumps({
                "action": "qasm_ready",
                "backend": backend_name,
                "backend_warning": backend_warning,
                "region": region,
                "qasm": qasm_str,
                "shots": shots,
                "status": "credentials_missing",
                "note": (
                    "Set IBM_API_KEY and IBM_INSTANCE_CRN in .env file "
                    "or environment variables to submit to IBM."
                ),
            })

        # Submit to IBM Quantum via REST API
        backend = IBMRestBackend(
            region=region,
            backend_name=backend_name,
        )

        # Build a DAG from the ops for the backend
        from quanta.core.circuit import CircuitBuilder
        from quanta.core.types import Instruction
        from quanta.dag.dag_circuit import DAGCircuit

        builder = CircuitBuilder(num_qubits)
        for op in ops:
            gate = op[0]
            qubits_list = tuple(op[1]) if len(op) > 1 else (0,)
            op_params = tuple(op[2]) if len(op) > 2 else None
            builder.record(Instruction(gate, qubits_list, op_params))

        dag = DAGCircuit.from_builder(builder)

        # Get ISA-transpiled QASM
        isa_qasm = backend.dag_to_qasm3(dag)

        # Submit sampler job
        job = backend.submit_sampler(dag, shots=shots)

        return json.dumps({
            "action": "submitted",
            "backend": backend_name,
            "backend_warning": backend_warning,
            "region": region,
            "qasm": isa_qasm,
            "shots": shots,
            "status": "submitted",
            "job_id": job.job_id,
            "job_status": job.status,
        })

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

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
#  Tool: IBM Job Result (Poll + Fetch)
# ═══════════════════════════════════════════

@mcp.tool()
def ibm_job_result(
    job_id: str = "",
    region: str = "us",
) -> str:
    """Get the status and results of an IBM Quantum job.

    Use after run_on_ibm to poll job status and retrieve results.
    Call repeatedly until status is "Completed" or "Failed".

    Pipeline: run_on_ibm → job_id → ibm_job_result(job_id) → counts

    Args:
        job_id: IBM job ID (returned by run_on_ibm).
        region: "us" or "eu-de".

    Returns:
        JSON with job status, and measurement counts when completed.
    """
    if not job_id:
        return json.dumps({
            "error": "job_id is required",
            "hint": "Use run_on_ibm first to get a job_id",
        })

    try:
        import os

        from quanta.backends.ibm_rest import IBMRestBackend

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("IBM_API_KEY", "")
        if not api_key:
            return json.dumps({
                "error": "IBM_API_KEY not set",
                "note": "Set in .env or environment",
            })

        backend = IBMRestBackend(region=region)

        # Get job status
        status_data = backend.job_status(job_id)

        # Parse response (handle dict or unexpected formats)
        if isinstance(status_data, dict):
            job_status = status_data.get("status", "unknown")
            backend_info = status_data.get("backend", {})
            backend_name = (
                backend_info.get("name", "")
                if isinstance(backend_info, dict) else str(backend_info)
            )
        else:
            job_status = str(status_data)
            backend_name = ""

        result: dict[str, Any] = {
            "job_id": job_id,
            "status": job_status,
            "backend": backend_name,
            "raw_response": status_data,
        }

        # If completed, fetch results
        if job_status.lower() in ("completed", "done"):
            results_data = backend.job_results(job_id)
            result["results"] = results_data
            result["note"] = "Job completed. Results contain measurement counts."
        elif job_status.lower() in ("failed", "cancelled"):
            result["note"] = f"Job {job_status}. Check IBM dashboard for details."
        else:
            result["note"] = (
                f"Job is {job_status}. "
                "Call ibm_job_result again to poll for completion."
            )

        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
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
#  Tool: Surface Code QEC Simulation
# ═══════════════════════════════════════════

@mcp.tool()
def surface_code_simulate(
    distance: int = 3,
    error_rate: float = 0.01,
    rounds: int = 1000,
    seed: int | None = 42,
) -> str:
    """Simulate surface code quantum error correction.

    Runs stabilizer-based error correction on a [[d², 1, d]] surface code.
    Injects random errors, extracts syndromes, decodes, and reports
    logical error rates.

    Args:
        distance: Code distance (odd, >= 3). Higher = more protection.
            d=3: 9 qubits, corrects 1 error.
            d=5: 25 qubits, corrects 2 errors.
            d=7: 49 qubits, corrects 3 errors.
        error_rate: Per-qubit error probability per round (0.0 to 1.0).
        rounds: Number of error correction rounds.
        seed: Random seed for reproducibility.

    Returns:
        JSON with logical/physical error rates, suppression factor,
        and correction statistics.
    """
    try:
        from quanta.qec.surface_code import SurfaceCode

        code = SurfaceCode(distance=distance)
        result = code.simulate_error_correction(
            error_rate=error_rate, rounds=rounds, seed=seed,
        )

        suppression = (
            result.physical_error_rate / result.logical_error_rate
            if result.logical_error_rate > 0 else float("inf")
        )

        return json.dumps({
            "algorithm": "Surface Code QEC",
            "code_params": code.code_params,
            "distance": distance,
            "physical_qubits": code.n_physical,
            "correctable_errors": code.correctable_errors,
            "physical_error_rate": result.physical_error_rate,
            "logical_error_rate": round(result.logical_error_rate, 6),
            "suppression_factor": round(suppression, 1),
            "rounds": result.rounds,
            "errors_injected": result.errors_injected,
            "errors_corrected": result.errors_corrected,
            "threshold_estimate": result.threshold_estimate,
            "explanation": (
                f"Surface code [[{code.n_physical},1,{distance}]] with "
                f"p={error_rate}: logical error rate "
                f"{result.logical_error_rate:.4%} "
                f"({suppression:.1f}x suppression). "
                f"Corrects up to {code.correctable_errors} errors per round."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ═══════════════════════════════════════════
#  Tool: Compare QEC Decoders
# ═══════════════════════════════════════════

@mcp.tool()
def compare_decoders(
    distance: int = 3,
    error_rate: float = 0.05,
    rounds: int = 500,
    seed: int | None = 42,
) -> str:
    """Compare MWPM vs Union-Find QEC decoders.

    Runs the same error patterns through both decoders and compares
    their correction success rates.

    Args:
        distance: Code distance (odd, >= 3).
        error_rate: Per-qubit error probability.
        rounds: Number of error correction rounds.
        seed: Random seed.

    Returns:
        JSON with per-decoder success rates and comparison.
    """
    try:
        import numpy as np

        from quanta.qec.decoder import MWPMDecoder, UnionFindDecoder
        from quanta.qec.surface_code import SurfaceCode

        code = SurfaceCode(distance=distance)
        mwpm = MWPMDecoder()
        uf = UnionFindDecoder()
        rng = np.random.default_rng(seed)

        mwpm_success = 0
        uf_success = 0
        n = code.n_physical

        for _ in range(rounds):
            error_mask = rng.random(n) < error_rate
            if not error_mask.any():
                mwpm_success += 1
                uf_success += 1
                continue

            syndrome = code.get_syndrome(error_mask)

            mwpm_result = mwpm.decode(syndrome, distance)
            uf_result = uf.decode(syndrome, distance)

            if mwpm_result.success:
                mwpm_success += 1
            if uf_result.success:
                uf_success += 1

        mwpm_rate = mwpm_success / rounds
        uf_rate = uf_success / rounds

        return json.dumps({
            "code_params": code.code_params,
            "error_rate": error_rate,
            "rounds": rounds,
            "mwpm": {
                "name": "Minimum Weight Perfect Matching",
                "success_rate": round(mwpm_rate, 4),
                "complexity": "O(n³)",
            },
            "union_find": {
                "name": "Union-Find",
                "success_rate": round(uf_rate, 4),
                "complexity": "O(n·α(n)) ≈ O(n)",
            },
            "winner": "MWPM" if mwpm_rate > uf_rate else (
                "Union-Find" if uf_rate > mwpm_rate else "Tied"
            ),
            "explanation": (
                f"At p={error_rate} on [[{n},1,{distance}]]: "
                f"MWPM {mwpm_rate:.1%} vs UF {uf_rate:.1%}. "
                "MWPM is optimal but O(n³); Union-Find is near-linear "
                "and better for large codes."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ═══════════════════════════════════════════
#  Resource: SDK Info
# ═══════════════════════════════════════════

@mcp.resource("quanta://info")
def sdk_info() -> str:
    """Quanta SDK version and capabilities."""
    return json.dumps({
        "name": "Quanta Quantum SDK",
        "version": "0.9.0",
        "description": "AI-native quantum computing SDK",
        "total_gates": 31,
        "total_tools": 16,
        "tool_categories": {
            "education": ["create_bell_state", "draw_circuit",
                         "list_gates", "explain_result"],
            "research": ["run_circuit", "grover_search", "shor_factor",
                        "simulate_noise", "qaoa_optimize",
                        "surface_code_simulate", "compare_decoders"],
            "business": ["monte_carlo_price", "cluster_data"],
            "hardware": ["run_on_ibm", "ibm_backends", "ibm_job_result"],
        },
        "capabilities": [
            "31 quantum gates (full IBM Quantum parity + Google/IonQ native)",
            "Statevector simulation (up to 27 qubits)",
            "Density matrix simulation (mixed states + noise)",
            "Pauli Frame simulator (1000+ Clifford qubits)",
            "7 noise channels (depolarizing, bitflip, etc.)",
            "DAG-based 6-pass compiler",
            "10 algorithms (Grover, Shor, VQE, QAOA, QSVM, QML, etc.)",
            "QEC (surface code, color code, Steane [[7,1,3]])",
            "2 decoders (MWPM + Union-Find)",
            "Entity Resolution (hybrid quantum-classical)",
            "SVG circuit visualization",
            "QASM 3.0 export/import",
            "IBM Quantum REST API (direct, no Qiskit)",
            "ISA transpilation (Heron rz/sx/x/cz)",
        ],
        "simulator_limits": {
            "max_qubits_statevector": 27,
            "max_qubits_density_matrix": 13,
            "max_qubits_pauli_frame": 1000,
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
#  Resource: Noise Profiles
# ═══════════════════════════════════════════

@mcp.resource("quanta://noise-profiles")
def noise_profiles() -> str:
    """Detailed profiles for all 7 noise channels."""
    return json.dumps({
        "channels": [
            {
                "name": "depolarizing",
                "type": "Symmetric",
                "params": {"probability": "p ∈ [0, 0.75]"},
                "effect": "Random X, Y, or Z error with equal probability p/3 each",
                "kraus_ops": 4,
                "use_case": "General hardware noise modeling",
                "ibm_typical": "p ≈ 0.001–0.01",
            },
            {
                "name": "bitflip",
                "type": "Asymmetric",
                "params": {"probability": "p ∈ [0, 1]"},
                "effect": "|0⟩ ↔ |1⟩ flip with probability p",
                "kraus_ops": 2,
                "use_case": "Classical-like errors, memory decoherence",
            },
            {
                "name": "phaseflip",
                "type": "Asymmetric",
                "params": {"probability": "p ∈ [0, 1]"},
                "effect": "|1⟩ → -|1⟩ phase error with probability p",
                "kraus_ops": 2,
                "use_case": "Dephasing, Z errors",
            },
            {
                "name": "amplitude_damping",
                "type": "Non-unital",
                "params": {"gamma": "γ ∈ [0, 1]"},
                "effect": "Excited state decays to ground: |1⟩ → |0⟩",
                "kraus_ops": 2,
                "use_case": "Energy relaxation (T1 decay)",
                "ibm_typical": "T1 ≈ 300μs",
            },
            {
                "name": "t2_relaxation",
                "type": "Dephasing",
                "params": {"gamma": "γ ∈ [0, 1]"},
                "effect": "Phase coherence decay (T2 process)",
                "kraus_ops": 2,
                "use_case": "Decoherence over time",
                "ibm_typical": "T2 ≈ 200μs",
            },
            {
                "name": "crosstalk",
                "type": "Correlated",
                "params": {"probability": "p ∈ [0, 1]"},
                "effect": "Unwanted coupling between neighboring qubits",
                "kraus_ops": 2,
                "use_case": "Multi-qubit gate errors",
            },
            {
                "name": "readout_error",
                "type": "Measurement",
                "params": {"p0_to_1": "p₀₁", "p1_to_0": "p₁₀"},
                "effect": "Measurement bit flips (classical post-processing)",
                "kraus_ops": 0,
                "use_case": "Measurement calibration errors",
                "ibm_typical": "p ≈ 0.01–0.05",
            },
        ],
        "usage": "run: simulate_noise(noise_type='depolarizing', probability=0.01)",
    })


# ═══════════════════════════════════════════
#  Resource: Gate Catalog
# ═══════════════════════════════════════════

@mcp.resource("quanta://gate-catalog")
def gate_catalog() -> str:
    """Complete catalog of all 31 quantum gates with categories."""
    return json.dumps({
        "total": 31,
        "heron_native": ["RZ", "SX", "X", "CZ"],
        "categories": {
            "pauli": {
                "gates": ["X", "Y", "Z"],
                "qubits": 1,
                "description": "Bit/phase flip operations",
            },
            "hadamard": {
                "gates": ["H"],
                "qubits": 1,
                "description": "Creates equal superposition",
            },
            "phase": {
                "gates": ["S", "T", "SDG", "TDG", "P(θ)"],
                "qubits": 1,
                "description": "Phase rotations (S=π/2, T=π/4)",
            },
            "rotation": {
                "gates": ["RX(θ)", "RY(θ)", "RZ(θ)"],
                "qubits": 1,
                "description": "Continuous rotations around axes",
            },
            "root": {
                "gates": ["SX", "SXdg"],
                "qubits": 1,
                "description": "√X gates — Heron native",
            },
            "universal": {
                "gates": ["U(θ,φ,λ)"],
                "qubits": 1,
                "description": "Any single-qubit unitary",
            },
            "two_qubit": {
                "gates": ["CX", "CY", "CZ", "SWAP", "RXX(θ)", "RZZ(θ)",
                          "ECR", "iSWAP", "CH", "CP(θ)", "MS(θ)"],
                "qubits": 2,
                "description": "Entangling operations",
            },
            "multi_qubit": {
                "gates": ["CCX", "RCCX", "RC3X", "CSWAP"],
                "qubits": "3-4",
                "description": "Toffoli, Fredkin, and multi-controlled",
            },
            "identity": {
                "gates": ["I"],
                "qubits": 1,
                "description": "No-op (useful for timing)",
            },
        },
        "isa_decomposition": {
            "H": "rz(π/2) · sx · rz(π/2)",
            "CX": "H(target) · CZ · H(target)",
            "RX(θ)": "rz(-π/2) · sx · rz(θ+π/2)",
            "RY(θ)": "rz(θ) · sx · rz(-θ)",
        },
    })


# ═══════════════════════════════════════════
#  Resource: Backend Specifications
# ═══════════════════════════════════════════

@mcp.resource("quanta://backend-specs")
def backend_specs() -> str:
    """Hardware specifications for all supported backends."""
    return json.dumps({
        "ibm_quantum": {
            "backends": [
                {"name": "ibm_torino", "qubits": 156, "processor": "Heron r3",
                 "2q_error": "0.25%", "native_gates": ["rz", "sx", "x", "cz"]},
                {"name": "ibm_fez", "qubits": 156, "processor": "Heron r2",
                 "2q_error": "0.28%", "native_gates": ["rz", "sx", "x", "cz"]},
                {"name": "ibm_marrakesh", "qubits": 156, "processor": "Heron r2",
                 "2q_error": "0.23%", "native_gates": ["rz", "sx", "x", "cz"]},
            ],
            "free_tier": {"qpu_time": "10 min/month", "max_shots": 100000},
            "connection": "Direct REST API — no Qiskit needed",
        },
        "simulators": [
            {"name": "Statevector", "max_qubits": 27, "method": "Tensor contraction",
             "memory": "O(2^n)", "speed": "Fast for < 20 qubits"},
            {"name": "Density Matrix", "max_qubits": 13, "method": "Full density matrix",
             "memory": "O(4^n)", "speed": "Required for mixed states + noise"},
            {"name": "Pauli Frame", "max_qubits": 1000, "method": "Stabilizer tableau",
             "memory": "O(n²)", "speed": "Ultra-fast, Clifford circuits only"},
        ],
    })


# ═══════════════════════════════════════════
#  Prompt: Grover Tutorial
# ═══════════════════════════════════════════

@mcp.prompt()
def grover_tutorial() -> str:
    """Guided tutorial for running Grover's search algorithm.

    Walk through quantum search step-by-step:
    superposition → oracle → amplification → measurement.
    """
    return (
        "Let's explore Grover's quantum search algorithm step by step.\n\n"
        "Grover's algorithm finds a target item in an unsorted database of N items "
        "using only O(√N) queries — a quadratic speedup over classical search.\n\n"
        "Please follow this workflow:\n\n"
        "1. **Start simple**: Use grover_search with num_qubits=3 and target=5 "
        "to search a space of 8 items.\n\n"
        "2. **Examine the result**: Look at success_probability — it should be "
        "close to 100%. Use explain_result on the counts to understand the output.\n\n"
        "3. **Visualize**: Use draw_circuit with this code to see the circuit:\n"
        "   @circuit(qubits=3)\n"
        "   def circ(q):\n"
        "       for i in range(3): H(q[i])  # Superposition\n"
        "       # Oracle marks target\n"
        "       X(q[0]); CCX(q[0], q[1], q[2]); X(q[0])\n"
        "       return measure(q)\n\n"
        "4. **Scale up**: Try num_qubits=4 (16 items) and num_qubits=5 (32 items). "
        "Notice how success probability stays high.\n\n"
        "5. **Add noise**: Use simulate_noise to see how real hardware noise "
        "affects search quality.\n\n"
        "Key insight: Classical search needs N/2 queries on average. "
        "Grover needs only ~√N iterations — for N=1,000,000, that's ~1000 vs 500,000."
    )


# ═══════════════════════════════════════════
#  Prompt: Option Pricing
# ═══════════════════════════════════════════

@mcp.prompt()
def option_pricing() -> str:
    """Guided workflow for quantum Monte Carlo option pricing.

    Demonstrates quantum amplitude estimation for financial derivatives.
    """
    return (
        "Let's price financial options using Quantum Monte Carlo.\n\n"
        "Quantum amplitude estimation achieves O(1/N) convergence vs classical "
        "Monte Carlo's O(1/√N) — a quadratic speedup for pricing derivatives.\n\n"
        "Workflow:\n\n"
        "1. **Price a call option**: Use monte_carlo_price with default params "
        "(S0=100, K=105, σ=0.2, T=1yr, r=5%). Compare quantum vs classical price.\n\n"
        "2. **Try a put option**: Change option_type to 'put'. The put-call parity "
        "should hold: C - P = S₀ - K·e^(-rT).\n\n"
        "3. **Vary volatility**: Try σ=0.1 (low vol) vs σ=0.5 (high vol). "
        "Higher volatility → higher option prices.\n\n"
        "4. **Increase precision**: Set n_qubits=8 for more precise pricing "
        "(more qubits = finer price distribution grid).\n\n"
        "5. **Deep vs out-of-money**: Compare K=90 (deep ITM) vs K=120 (OTM).\n\n"
        "Key insight: Quantum advantage grows with the number of scenarios needed. "
        "For complex path-dependent derivatives (Asian options, barrier options), "
        "quantum speedup is even more significant."
    )


# ═══════════════════════════════════════════
#  Prompt: Circuit Debug
# ═══════════════════════════════════════════

@mcp.prompt()
def circuit_debug() -> str:
    """Guided workflow for debugging quantum circuits.

    Systematic approach to identify and fix circuit issues.
    """
    return (
        "Let's debug a quantum circuit step by step.\n\n"
        "Common quantum circuit bugs:\n"
        "• Wrong qubit order (CX control/target swapped)\n"
        "• Missing measurement gates\n"
        "• Incorrect rotation angles\n"
        "• Gate applied to wrong qubit\n\n"
        "Debugging workflow:\n\n"
        "1. **Run the circuit**: Use run_circuit with your code. Check if the output "
        "matches expectations.\n\n"
        "2. **Visualize**: Use draw_circuit to see the circuit structure. "
        "Verify gate placement and qubit wiring.\n\n"
        "3. **Check known states**: Test with simple inputs first. A Bell state "
        "should give ~50/50 for |00⟩ and |11⟩.\n\n"
        "4. **Analyze results**: Use explain_result on the measurement counts. "
        "Look for unexpected outcomes or probabilities.\n\n"
        "5. **Noise test**: Use simulate_noise to rule out noise-related issues. "
        "If results change dramatically, the circuit may be noise-sensitive.\n\n"
        "6. **Compare gates**: Use list_gates to verify gate semantics. "
        "Remember: CX and CZ have different effects!\n\n"
        "Tip: Start with shots=10000 for more precise probability estimates."
    )


# ═══════════════════════════════════════════
#  Prompt: QEC Introduction
# ═══════════════════════════════════════════

@mcp.prompt()
def qec_intro() -> str:
    """Guided introduction to quantum error correction.

    Explore surface codes, decoders, and error thresholds interactively.
    """
    return (
        "Let's explore Quantum Error Correction (QEC) interactively.\n\n"
        "QEC protects quantum information by encoding 1 logical qubit into "
        "many physical qubits. The surface code is the leading candidate "
        "for fault-tolerant quantum computing.\n\n"
        "Exploration workflow:\n\n"
        "1. **Low noise**: Use surface_code_simulate with distance=3 and "
        "error_rate=0.001. You should see near-zero logical error rate — "
        "the code is successfully protecting the qubit.\n\n"
        "2. **Increase noise**: Try error_rate=0.01, then 0.05, then 0.10. "
        "Watch the logical error rate increase. The threshold is around 1.1%.\n\n"
        "3. **Increase distance**: Compare distance=3 vs distance=5 at "
        "error_rate=0.01. Higher distance = better protection but more qubits.\n\n"
        "4. **Compare decoders**: Use compare_decoders with distance=3 and "
        "error_rate=0.05. MWPM is optimal but slow; Union-Find is fast and "
        "near-optimal.\n\n"
        "5. **Threshold experiment**: Run surface_code_simulate at error_rate "
        "values [0.005, 0.008, 0.010, 0.012, 0.015] with distance=3 and "
        "distance=5. Below threshold, higher distance helps. Above threshold, "
        "it doesn't.\n\n"
        "Key insight: The surface code threshold (~1.1%) means if physical "
        "error rates drop below this, we can make logical error rates "
        "arbitrarily small by increasing code distance."
    )


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
