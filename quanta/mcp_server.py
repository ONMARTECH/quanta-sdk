"""
quanta.mcp_server — MCP Server for Quanta Quantum SDK.

Exposes quantum simulation as MCP tools, enabling AI assistants
(Claude, etc.) to perform quantum computations directly.

Run:
    fastmcp run quanta/mcp_server.py

Or install in Claude Desktop:
    fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"

Tools:
    - run_circuit:       Execute quantum circuit code
    - create_bell_state: Quick Bell state |Φ+⟩
    - grover_search:     Grover's search algorithm
    - shor_factor:       Shor's factoring algorithm
    - simulate_noise:    Run circuit with noise model
    - list_gates:        Available quantum gates
    - explain_result:    Interpret measurement results
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
              SWAP, CCX, RX, RY, RZ, measure.
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
        from quanta import (
            circuit, H, X, Y, Z, S, T, CX, CZ, CY, SWAP, CCX,
            RX, RY, RZ, measure, run,
        )

        namespace: dict[str, Any] = {
            "circuit": circuit, "H": H, "X": X, "Y": Y, "Z": Z,
            "S": S, "T": T, "CX": CX, "CZ": CZ, "CY": CY,
            "SWAP": SWAP, "CCX": CCX, "RX": RX, "RY": RY, "RZ": RZ,
            "measure": measure,
        }

        exec(code, namespace)

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
        from quanta import circuit, H, CX, measure, run

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
        from quanta.layer3.search import search, SearchResult

        result = search(
            num_qubits=num_qubits,
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
        from quanta.layer3.shor import factor

        result = factor(number)

        return json.dumps({
            "algorithm": "Shor's Factoring",
            "input": number,
            "factors": result.factors,
            "is_correct": result.factors[0] * result.factors[1] == number
            if len(result.factors) == 2 else False,
            "explanation": (
                f"{number} = {' × '.join(map(str, result.factors))}. "
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
        from quanta import circuit, H, CX, measure
        from quanta.simulator.statevector import StateVectorSimulator
        from quanta.simulator.noise import (
            NoiseModel, Depolarizing, BitFlip, PhaseFlip,
            AmplitudeDamping, T2Relaxation, Crosstalk, ReadoutError,
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

        # Build Bell state manually with noise
        sim = StateVectorSimulator(2, seed=seed)
        model = NoiseModel().add(channel)

        sim.apply("H", (0,))
        state = model.apply_noise(sim.state, (0,), 2, rng)
        sim._state = state

        sim.apply("CX", (0, 1))
        state = model.apply_noise(sim.state, (0, 1), 2, rng)
        sim._state = state

        counts = sim.sample(shots)

        # Apply readout error if applicable
        if noise_type == "readout_error" and isinstance(channel, ReadoutError):
            counts = channel.apply_to_counts(counts, rng)

        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}

        ideal_fidelity = probs.get("00", 0) + probs.get("11", 0)

        return json.dumps({
            "noise_model": channel.name,
            "counts": counts,
            "probabilities": probs,
            "ideal_states": ["00", "11"],
            "fidelity": round(ideal_fidelity, 4),
            "noise_impact": (
                f"Fidelity: {ideal_fidelity:.1%} "
                f"(ideal: 100%, loss: {1 - ideal_fidelity:.1%}). "
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
        {"name": "H", "qubits": 1, "type": "Clifford",
         "description": "Hadamard gate — creates superposition"},
        {"name": "X", "qubits": 1, "type": "Pauli",
         "description": "Pauli-X (NOT) — bit flip |0⟩↔|1⟩"},
        {"name": "Y", "qubits": 1, "type": "Pauli",
         "description": "Pauli-Y — rotation around Y axis"},
        {"name": "Z", "qubits": 1, "type": "Pauli",
         "description": "Pauli-Z — phase flip |1⟩→-|1⟩"},
        {"name": "S", "qubits": 1, "type": "Clifford",
         "description": "S gate — π/2 phase shift"},
        {"name": "T", "qubits": 1, "type": "Non-Clifford",
         "description": "T gate — π/4 phase shift (universal)"},
        {"name": "RX(θ)", "qubits": 1, "type": "Parametric",
         "description": "Rotation around X axis by angle θ"},
        {"name": "RY(θ)", "qubits": 1, "type": "Parametric",
         "description": "Rotation around Y axis by angle θ"},
        {"name": "RZ(θ)", "qubits": 1, "type": "Parametric",
         "description": "Rotation around Z axis by angle θ"},
        {"name": "CX", "qubits": 2, "type": "Clifford",
         "description": "CNOT — controlled NOT, creates entanglement"},
        {"name": "CZ", "qubits": 2, "type": "Clifford",
         "description": "Controlled-Z — phase flip on |11⟩"},
        {"name": "CY", "qubits": 2, "type": "Clifford",
         "description": "Controlled-Y"},
        {"name": "SWAP", "qubits": 2, "type": "Clifford",
         "description": "Swaps two qubit states"},
        {"name": "CCX", "qubits": 3, "type": "Toffoli",
         "description": "Toffoli — double-controlled NOT gate"},
    ]
    return json.dumps({
        "total_gates": len(gates),
        "gates": gates,
        "parametric_usage": "RX(angle)(qubit) — e.g. RX(3.14)(q[0])",
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
#  Resource: SDK Info
# ═══════════════════════════════════════════

@mcp.resource("quanta://info")
def sdk_info() -> str:
    """Quanta SDK version and capabilities."""
    return json.dumps({
        "name": "Quanta Quantum SDK",
        "version": "0.6.0",
        "description": "Multi-paradigm quantum computing SDK",
        "capabilities": [
            "Statevector simulation (up to 27 qubits)",
            "7 noise channels (Depolarizing, BitFlip, PhaseFlip, "
            "AmplitudeDamping, T2Relaxation, Crosstalk, ReadoutError)",
            "DAG-based 6-pass compiler",
            "Grover search, Shor factoring, VQE, QAOA",
            "Quantum Error Correction (surface, color, bit/phase-flip)",
            "Entity Resolution (hybrid classical-quantum)",
            "QASM 3.0 export",
            "Google/IBM/IonQ backend support",
        ],
        "simulator_limits": {
            "max_qubits": 27,
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
