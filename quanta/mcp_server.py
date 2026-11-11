"""
quanta.mcp_server — AI-native MCP Server for Quanta Quantum SDK.

Exposes quantum simulation as MCP tools, enabling AI assistants
(Claude, GPT, etc.) to perform quantum computations directly.

Run:
    fastmcp run quanta/mcp_server.py

Or install in Claude Desktop:
    fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"

Tools (25):
  Education:
    - create_bell_state:         Quick Bell state |Φ+⟩
    - draw_circuit:              SVG circuit diagram
    - list_gates:                Available quantum gates (31)
    - explain_result:            Interpret measurement results
  Research & Fault-Tolerance:
    - run_circuit:               Execute quantum circuit code
    - mps_circuit_simulate:      1D tensor-network MPS simulation (15-30+ qubits)
    - grover_search:             Grover's search algorithm
    - shor_factor:               Shor's factoring algorithm
    - simulate_noise:            Run circuit with noise model
    - qaoa_optimize:             QAOA combinatorial optimization
    - optimize_circuit:          Compiler optimization with metrics
    - surface_code_simulate:     Surface code QEC simulation
    - compare_decoders:          Compare MWPM vs Union-Find decoders
    - qec_diagnose:              Syndrome diagnosis & error lookup
    - estimate_fault_tolerant_cost: Surface code / Willow resource estimation
  Agentic AI & Compilation:
    - quanta_reasoning_eval:     Reasoning complexity feedback for AI agents
    - transpile_for_target:      Transpile to Heron, Willow or IonQ native ISA
  Machine Learning:
    - qml_classify:              Quantum ML classification
  Business:
    - monte_carlo_price:         Quantum Monte Carlo option pricing
    - option_greeks:             Option sensitivity Greeks (Δ,Γ,ν,Θ,ρ)
    - portfolio_optimize:        Gate-based Ising QAOA Markowitz portfolio selection
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
import logging
import re as _re
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── Security: Code Validation & Sandboxing ──

_BLOCKED_PATTERNS = _re.compile(
    r"\b("
    r"__import__|importlib|__subclasses__|__globals__|__code__"
    r"|__builtins__|__loader__|__spec__"
    r"|exec\s*\(|eval\s*\("
    r"|open\s*\(|compile\s*\("
    r"|os\.system|os\.popen|os\.exec"
    r"|subprocess|shutil\.rmtree"
    r"|socket\.|http\.client|urllib"
    r"|getattr\s*\(|setattr\s*\(|delattr\s*\("
    r"|globals\s*\(|locals\s*\("
    r")\b",
    _re.IGNORECASE,
)


def _validate_code(code: str) -> None:
    """Pre-scan code for dangerous patterns before execution.

    Raises:
        ValueError: If code contains blocked patterns.
    """
    match = _BLOCKED_PATTERNS.search(code)
    if match:
        raise ValueError(
            f"Blocked: code contains disallowed pattern '{match.group()}'. "
            f"Only Quanta SDK quantum operations are permitted."
        )

    # Block raw import statements (only 'from quanta' is allowed)
    for line in code.splitlines():
        stripped = line.strip()
        is_import = stripped.startswith("import ") or stripped.startswith("from ")
        if is_import and not stripped.startswith("from quanta"):
            raise ValueError(
                f"Blocked: external imports not allowed. "
                f"Only 'from quanta ...' is permitted. Got: {stripped!r}"
            )


_SAFE_BUILTINS: dict[str, Any] = {
    "range": range, "len": len, "int": int, "float": float,
    "str": str, "bool": bool, "list": list, "dict": dict,
    "tuple": tuple, "set": set, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "sorted": sorted,
    "reversed": reversed, "abs": abs, "min": min, "max": max,
    "sum": sum, "round": round, "any": any, "all": all,
    "isinstance": isinstance, "type": type,
    "print": print, "repr": repr,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError,
}


def _safe_error(exc: Exception) -> str:
    """Return sanitized error message without tracebacks."""
    logger.exception("MCP tool error")  # log full trace server-side
    return json.dumps({
        "error": str(exc),
        "type": type(exc).__name__,
    })


def _build_sandbox_namespace() -> dict[str, Any]:
    """Build restricted execution sandbox pre-loaded with all 31 Quanta gates."""
    import math as math_mod

    import numpy as np_mod

    from quanta import (  # noqa: I001
        CCX,
        CH,
        CP,
        CSWAP,
        CX,
        CY,
        CZ,
        ECR,
        MS,
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
        iSWAP,
        measure,
        run,
    )

    return {
        "__builtins__": _SAFE_BUILTINS,
        "circuit": circuit,
        "measure": measure,
        "run": run,
        # All 31 quantum gates (IBM Heron + Google Willow + IonQ Aria parity)
        "H": H, "X": X, "Y": Y, "Z": Z,
        "S": S, "T": T, "I": I, "SDG": SDG, "TDG": TDG,
        "SX": SX, "SXdg": SXdg,
        "CX": CX, "CZ": CZ, "CY": CY, "SWAP": SWAP,
        "CCX": CCX, "RCCX": RCCX, "RC3X": RC3X,
        "ECR": ECR, "iSWAP": iSWAP, "CSWAP": CSWAP, "CH": CH,
        "RX": RX, "RY": RY, "RZ": RZ, "P": P, "U": U,
        "RXX": RXX, "RZZ": RZZ, "CP": CP, "MS": MS,
        # Math conveniences
        "math": math_mod,
        "np": np_mod,
        "pi": math_mod.pi,
        "sqrt": math_mod.sqrt,
    }


def _is_multiple_of(angle: float, period: float, tol: float = 1e-9) -> bool:
    """Check whether angle is an integer multiple of period within tol."""
    ratio = float(angle) / period
    return abs(ratio - round(ratio)) < tol


def _classify_gate_clifford_and_t(
    gate_name: str,
    params: tuple[float, ...] | None,
    eps_synth: float | None = None,
) -> tuple[bool, int, int]:
    """Classify a gate as Clifford vs non-Clifford and compute its T-count.

    Args:
        gate_name: Gate identifier from DAG node.
        params: Optional rotation angle tuple.
        eps_synth: Per-rotation Ross-Selinger synthesis precision. If None,
            counts arbitrary-angle synthesis rotations separately without
            expanding to T-gates yet.

    Returns:
        Tuple ``(is_clifford, exact_t_gates, arb_synth_rotations)``.
    """
    import math as _m

    g = gate_name.upper()
    ps = tuple(float(x) for x in params) if params else ()

    # Pure fixed Clifford gates (including ECR and iSWAP)
    if g in {
        "I", "H", "X", "Y", "Z", "S", "SDG", "SX", "SXDG",
        "CX", "CY", "CZ", "SWAP", "ECR", "ISWAP",
    }:
        return (True, 0, 0)

    # Fixed non-Clifford gates (Litinski 2019 / Jones-Gidney T-counts)
    if g in {"T", "TDG"}:
        return (False, 1, 0)
    if g == "CH":
        return (False, 2, 0)
    if g == "RCCX":
        return (False, 3, 0)  # Margolus relative-phase Toffoli = 3 T
    if g == "CCX":
        return (False, 4, 0)  # T-optimal measurement-assisted Toffoli = 4 T
    if g == "CSWAP":
        return (False, 7, 0)  # CX + Toffoli + CX
    if g == "RC3X":
        return (False, 8, 0)

    # Controlled-Phase CP(theta) = diag(1, 1, 1, e^{i*theta}):
    # Clifford iff theta is an integer multiple of pi (I or CZ).
    # Odd multiple of pi/2 (Controlled-S) decomposes into 3 P(+-pi/4) = 3 T.
    if g == "CP":
        theta = ps[0] if ps else 0.0
        if _is_multiple_of(theta, _m.pi):
            return (True, 0, 0)
        if _is_multiple_of(theta, _m.pi / 2.0):
            return (False, 3, 0)
        if eps_synth is not None:
            t_rot = max(1, _m.ceil(3.0 * _m.log2(1.0 / max(1e-30, eps_synth))))
            return (False, t_rot, 1)
        return (False, 0, 1)

    # General 1-qubit Euler gate U(theta, phi, lam)
    if g == "U":
        if all(_is_multiple_of(a, _m.pi / 2.0) for a in ps):
            return (True, 0, 0)
        exact_t = 0
        arb_rots = 0
        for a in ps:
            if _is_multiple_of(a, _m.pi / 2.0):
                continue
            if _is_multiple_of(a, _m.pi / 4.0):
                exact_t += 1
            else:
                arb_rots += 1
        if eps_synth is not None and arb_rots > 0:
            t_rot = max(1, _m.ceil(3.0 * _m.log2(1.0 / max(1e-30, eps_synth))))
            return (False, exact_t + arb_rots * t_rot, arb_rots)
        return (False, exact_t, arb_rots)

    # Single-parameter Pauli/Ising rotations: RX, RY, RZ, P, RXX, RZZ, MS
    if g in {"RX", "RY", "RZ", "P", "RXX", "RZZ", "MS"}:
        theta = ps[0] if ps else 0.0
        if _is_multiple_of(theta, _m.pi / 2.0):
            return (True, 0, 0)
        if _is_multiple_of(theta, _m.pi / 4.0):
            return (False, 1, 0)
        if eps_synth is not None:
            t_rot = max(1, _m.ceil(3.0 * _m.log2(1.0 / max(1e-30, eps_synth))))
            return (False, t_rot, 1)
        return (False, 0, 1)

    return (True, 0, 0)


# ── MCP Server ──
mcp = FastMCP(
    "Quanta Quantum SDK",
    instructions=(
        "AI-native quantum computing SDK. Quanta provides 25 tools organized "
        "into 6 categories:\n"
        "• Education: Bell states, circuit drawing, gate reference, result explanation\n"
        "• Research & QEC: Circuit execution, MPS simulation, Grover, Shor, QAOA, "
        "noise, compiler passes, surface code QEC, decoder comparison, FTQC cost\n"
        "• Agentic AI: Circuit reasoning evaluator, ISA target transpilation\n"
        "• Machine Learning: Quantum classifier with feature maps and ansatz presets\n"
        "• Business: Quantum Monte Carlo pricing, CRN Greeks, Ising QAOA portfolio, clustering\n"
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
    method: str = "statevector",
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
              Also: circuit, measure, run, math, np, pi, sqrt, MPSSimulator, CircuitRunner.
               Only 'from quanta ...' imports are allowed.
        shots: Number of measurement repetitions.
        seed: Random seed for reproducibility.
        method: Simulator method ('statevector', 'mps', 'auto'). Default 'statevector'.

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
        from quanta import run
        from quanta.runner import CircuitResult, CircuitRunner
        from quanta.simulator.mps import MPSSimulator

        # Security: validate code before execution
        _validate_code(code)

        # Pre-inject all 31 Quanta gates & safe symbols — NO raw __builtins__
        namespace = _build_sandbox_namespace()
        namespace.update({
            "MPSSimulator": MPSSimulator,
            "CircuitRunner": CircuitRunner,
            "CircuitResult": CircuitResult,
        })

        exec(code, namespace)  # noqa: S102 — sandboxed, restricted builtins

        if "circ" not in namespace:
            return json.dumps({
                "error": "Code must define a variable called 'circ'.",
                "hint": "Add: circ = your_circuit_function (or name it 'circ')",
            })

        sim_method = "dense" if method.lower() in ("statevector", "dense") else method
        result = run(namespace["circ"], shots=shots, seed=seed, method=sim_method)

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
        return _safe_error(e)


# ═══════════════════════════════════════════
#  Tool: MPS Circuit Simulation (High-Qubit)
# ═══════════════════════════════════════════

@mcp.tool()
def mps_circuit_simulate(
    code: str,
    num_qubits: int = 15,
    chi_max: int = 64,
    shots: int = 1024,
    seed: int | None = None,
) -> str:
    """Execute a high-qubit (15-30+ qubits) 1D quantum circuit using Matrix Product States.

    Tensor-network simulator with O(N * chi^2) memory containment, capable of
    simulating 15 to 30+ qubits without dense statevector exponential O(2^N) explosion.

    Args:
        code: Python code defining a quantum circuit. Can define variable 'circ'
              using @circuit or build an MPSSimulator directly.
        num_qubits: Number of qubits in the circuit (default 15).
        chi_max: Maximum MPS bond dimension (default 64).
        shots: Number of measurement repetitions (default 1024).
        seed: Random seed for reproducibility.

    Returns:
        JSON with counts, bond_dimensions, max_bond_dim, truncation_error,
        entanglement_entropy, memory_bytes, num_qubits, shots, and execution_time_ms.
    """
    try:
        import time as time_mod

        t0 = time_mod.perf_counter()

        _validate_code(code)


        from quanta.core.circuit import CircuitDefinition
        from quanta.dag.dag_circuit import DAGCircuit
        from quanta.runner import CircuitResult, CircuitRunner, _filter_measured_qubits
        from quanta.simulator.mps import MPSSimulator

        namespace = _build_sandbox_namespace()
        namespace.update({
            "MPSSimulator": MPSSimulator,
            "CircuitRunner": CircuitRunner,
            "CircuitResult": CircuitResult,
            "num_qubits": num_qubits,
            "chi_max": chi_max,
        })

        exec(code, namespace)  # noqa: S102

        sim: MPSSimulator | None = None
        counts: dict[str, int] = {}
        nq = num_qubits

        if "circ" in namespace:
            target = namespace["circ"]
            if isinstance(target, CircuitDefinition):
                nq = getattr(target, "num_qubits", num_qubits)
                builder = target.build()
                sim = MPSSimulator(nq, seed=seed, chi_max=chi_max)
                for inst in builder.instructions:
                    sim.apply(inst.gate_name, inst.qubits, inst.params)
                counts = sim.sample(shots)
                if builder.measurement and builder.measurement.qubits:
                    counts = _filter_measured_qubits(counts, builder.measurement.qubits, nq)
            elif isinstance(target, DAGCircuit):
                nq = target.num_qubits
                sim = MPSSimulator(nq, seed=seed, chi_max=chi_max)
                for op in target.op_nodes():
                    sim.apply(op.gate_name, op.qubits, op.params)
                counts = sim.sample(shots)
                if target.measurement and target.measurement.qubits:
                    counts = _filter_measured_qubits(counts, target.measurement.qubits, nq)
            elif isinstance(target, MPSSimulator):
                sim = target
                nq = sim.num_qubits
                counts = sim.sample(shots)
        elif "sim" in namespace and isinstance(namespace["sim"], MPSSimulator):
            sim = namespace["sim"]
            nq = sim.num_qubits
            counts = sim.sample(shots)
        else:
            return json.dumps({
                "error": "Code must define a circuit 'circ' (@circuit) or MPSSimulator 'sim'.",
            })

        execution_time_ms = round((time_mod.perf_counter() - t0) * 1000.0, 3)

        entropy = 0.0
        if nq >= 2 and sim is not None:
            try:
                entropy = round(float(sim.entanglement_entropy(cut=nq // 2)), 6)
            except Exception:
                entropy = 0.0

        if sim is None:
            return json.dumps({"error": "Failed to initialize MPSSimulator."})

        return json.dumps({
            "num_qubits": nq,
            "shots": shots,
            "counts": counts,
            "bond_dimensions": sim.bond_dimensions,
            "max_bond_dim": sim.max_bond_dim,
            "truncation_error": float(sim.truncation_error),
            "entanglement_entropy": entropy,
            "memory_bytes": sim.memory_bytes,
            "execution_time_ms": execution_time_ms,
        })

    except Exception as e:
        return _safe_error(e)


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
            targets=[target],
            shots=shots,
            seed=seed,
        )

        target_bin = format(target, f"0{num_qubits}b")
        return json.dumps({
            "algorithm": "Grover's Search",
            "search_space_size": 2**num_qubits,
            "target": target,
            "target_binary": target_bin,
            "found": result.most_frequent,
            "success_probability": result.probabilities.get(target_bin, 0.0),
            "counts": result.counts,
            "classical_queries_needed": 2**num_qubits // 2,
            "classical_setup_queries": result.classical_setup_queries,
            "oracle_queries": result.oracle_queries,
            "quantum_queries_used": result.oracle_queries,
            "gate_count": result.gate_count,
            "depth": result.depth,
        })
    except Exception as e:
        return _safe_error(e)


# ═══════════════════════════════════════════
#  Tool: Shor's Factoring
# ═══════════════════════════════════════════

@mcp.tool()
def shor_factor(number: int = 15, seed: int | None = 42) -> str:
    """Factor an integer using Shor's algorithm (Beauregard n+1 QPE).

    Uses quantum period-finding via Griffiths-Niu / Beauregard semiclassical
    iterative Quantum Phase Estimation on n + 1 qubits without silent
    classical trial-division fallbacks.

    Args:
        number: Integer to factor (must be > 1 and composite).
        seed: Random seed for coprime base selection and QPE sampling.

    Returns:
        JSON with the prime factors, quantum period r, and method metadata.
    """
    try:
        from quanta.layer3.shor import factor, factor_recursive

        result = factor(number, seed=seed)
        prime_factors = factor_recursive(number, seed=seed)

        return json.dumps({
            "algorithm": "Shor's Factoring",
            "input": number,
            "factors": list(result.factors),
            "prime_factors": prime_factors,
            "method": result.method,
            "period": result.period,
            "attempts": result.attempts,
            "is_correct": result.factors[0] * result.factors[1] == number
            if len(result.factors) == 2 else False,
            "explanation": (
                f"{number} = {' × '.join(map(str, prime_factors))} "
                f"(period r={result.period}, method={result.method}). "
                "Shor's algorithm uses semiclassical iterative QPE "
                "to find modular multiplicative order."
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

        # Build exact CPTP Kraus operators from physical channel parameters
        p = float(np.clip(probability, 0.0, 1.0))
        I2 = np.eye(2, dtype=complex)
        X2 = np.array([[0, 1], [1, 0]], dtype=complex)
        Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z2 = np.array([[1 + 0j, 0], [0, -1]], dtype=complex)
        I4 = np.eye(4, dtype=complex)
        ZZ4 = np.kron(Z2, Z2)

        kraus_1q_map = {
            # Symmetric depolarizing: X, Y, Z each with probability p/3
            # sum K_i^dag K_i = (1 - p) I + 3 * (p/3) I = I
            "depolarizing": [
                np.sqrt(max(0.0, 1.0 - p)) * I2,
                np.sqrt(p / 3.0) * X2,
                np.sqrt(p / 3.0) * Y2,
                np.sqrt(p / 3.0) * Z2,
            ],
            "bitflip": [np.sqrt(max(0.0, 1.0 - p)) * I2, np.sqrt(p) * X2],
            "phaseflip": [np.sqrt(max(0.0, 1.0 - p)) * I2, np.sqrt(p) * Z2],
            "amplitude_damping": [
                np.array([[1.0, 0.0], [0.0, np.sqrt(max(0.0, 1.0 - p))]], dtype=complex),
                np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=complex),
            ],
            # Exact T2 pure phase damping Kraus representation:
            # K0 = diag(1, sqrt(1 - gamma)), K1 = diag(0, sqrt(gamma))
            "t2_relaxation": [
                np.array([[1.0, 0.0], [0.0, np.sqrt(max(0.0, 1.0 - p))]], dtype=complex),
                np.array([[0.0, 0.0], [0.0, np.sqrt(p)]], dtype=complex),
            ],
        }
        # Correlated 2-qubit ZZ crosstalk Kraus operators on (q0, q1)
        kraus_2q_crosstalk = [
            np.sqrt(max(0.0, 1.0 - p)) * I4,
            np.sqrt(p) * ZZ4,
        ]

        sim = DensityMatrixSimulator(2, seed=seed)

        # Apply H gate on q[0]
        sim.apply("H", (0,))

        # Apply post-1Q gate noise
        if noise_type in kraus_1q_map:
            sim.apply_kraus(kraus_1q_map[noise_type], (0,))
        elif noise_type == "crosstalk":
            sim.apply_kraus(kraus_2q_crosstalk, (0, 1))

        # Apply CX gate on (q[0], q[1])
        sim.apply("CX", (0, 1))

        # Apply post-2Q gate noise
        if noise_type in kraus_1q_map:
            sim.apply_kraus(kraus_1q_map[noise_type], (0,))
            sim.apply_kraus(kraus_1q_map[noise_type], (1,))
        elif noise_type == "crosstalk":
            sim.apply_kraus(kraus_2q_crosstalk, (0, 1))

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
        return _safe_error(e)


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
    if problem not in ("max_cut", "max_ones", "min_ones"):
        raise ValueError(
            f"Unknown problem {problem!r}. Supported problems: 'max_cut', 'max_ones', 'min_ones'."
        )

    try:
        from quanta.layer3.optimize import optimize

        if problem == "max_cut":
            def cost_fn(x: int) -> int:
                return bin(x ^ (x >> 1)).count("1")
            desc = "Max-Cut: maximize edges between partitions"
        elif problem == "max_ones":
            def cost_fn(x: int) -> int:
                return bin(x).count("1")
            desc = "Max-Ones: maximize number of 1-bits"
        else:  # problem == "min_ones"
            def cost_fn(x: int) -> int:
                return bin(x).count("1")
            minimize = True
            desc = "Min-Ones: minimize number of 1-bits"

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

        # Full 31-gate QASM map
        gate_map = {
            "H": "h", "X": "x", "Y": "y", "Z": "z",
            "S": "s", "T": "t", "I": "id",
            "SDG": "sdg", "TDG": "tdg",
            "SX": "sx", "SXdg": "sxdg",
            "CX": "cx", "CZ": "cz", "CY": "cy",
            "SWAP": "swap", "CCX": "ccx",
            "ECR": "ecr", "iSWAP": "iswap", "CSWAP": "cswap", "CH": "ch",
            "RX": "rx", "RY": "ry", "RZ": "rz",
            "P": "p", "U": "u", "CP": "cp", "MS": "rxx",
            "RXX": "rxx", "RZZ": "rzz",
            "RCCX": "rccx", "RC3X": "rc3x",
        }

        # Parametric gates that require angle parameters
        param_gates = {"RX", "RY", "RZ", "P", "U", "RXX", "RZZ", "CP", "MS"}

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
        return _safe_error(e)


# ═══════════════════════════════════════════
#  Tool: QML Classify
# ═══════════════════════════════════════════

@mcp.tool()
def qml_classify(
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    n_qubits: int = 4,
    feature_map: str = "angle",
    ansatz: str = "hardware_efficient",
    epochs: int = 10,
    seed: int = 42,
) -> str:
    """Run quantum machine learning classification.

    Trains a variational quantum classifier on your data
    and predicts labels for test samples.

    Args:
        X_train: Training feature vectors (list of lists).
        y_train: Training labels (0 or 1).
        X_test: Test feature vectors to classify.
        n_qubits: Number of qubits (default: 4).
        feature_map: "angle", "zz", or "amplitude".
        ansatz: "hardware_efficient", "strongly_entangling", or "reuploading".
        epochs: Training epochs (default: 10).
        seed: Random seed.

    Returns:
        JSON with predictions, accuracy, and training loss.
    """
    try:
        from quanta.qml import Classifier

        clf = Classifier(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            seed=seed,
        )
        result = clf.fit(X_train, y_train, epochs=epochs)
        predictions = clf.predict(X_test).tolist()

        return json.dumps({
            "predictions": predictions,
            "training_accuracy": f"{result.accuracy:.2%}",
            "loss_history": [round(v, 4) for v in result.loss_history],
            "n_qubits": n_qubits,
            "feature_map": feature_map,
            "ansatz": ansatz,
            "n_params": result.n_params,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════
#  Tool: Optimize Circuit
# ═══════════════════════════════════════════

@mcp.tool()
def optimize_circuit(
    circuit_code: str,
    passes: list[str] | None = None,
) -> str:
    """Optimize a quantum circuit using compiler passes.

    Runs optimization passes (gate cancellation, rotation merging,
    identity removal) and reports before/after metrics.

    Args:
        circuit_code: Python code defining a @circuit function.
        passes: List of pass names to run. Default: all available.
            Options: "CancelInverses", "MergeRotations", "IdentityRemoval"

    Returns:
        JSON with original and optimized gate counts, depth, and
        the list of passes applied.
    """
    try:
        from quanta.compiler.passes.optimize import (
            CancelInverses,
            CommutationPass,
            IdentityRemoval,
            MergeRotations,
        )
        from quanta.compiler.pipeline import CompilerPipeline
        from quanta.dag.dag_circuit import DAGCircuit

        # Security: validate code before execution
        _validate_code(circuit_code)

        # Execute circuit code in restricted sandbox with all 31 gates
        sandbox = _build_sandbox_namespace()
        exec(  # noqa: S102
            circuit_code,
            sandbox,
        )

        # Find the circuit function
        circ_fn = None
        for v in sandbox.values():
            if hasattr(v, "build"):
                circ_fn = v
                break

        if circ_fn is None:
            return json.dumps({"error": "No @circuit function found in code."})

        dag = DAGCircuit.from_builder(circ_fn.build())
        original_gates = dag.gate_count()
        original_depth = dag.depth()

        # Select passes
        available = {
            "CancelInverses": CancelInverses,
            "MergeRotations": MergeRotations,
            "IdentityRemoval": IdentityRemoval,
            "CommutationPass": CommutationPass,
        }

        if passes is None:
            selected = [CancelInverses(), MergeRotations(), IdentityRemoval()]
        else:
            selected = []
            for p in passes:
                if p in available:
                    selected.append(available[p]())
                else:
                    return json.dumps({
                        "error": f"Unknown pass '{p}'. Available: {list(available.keys())}",
                    })

        pipeline = CompilerPipeline(selected)
        result = pipeline.run(dag)

        return json.dumps({
            "original": {
                "gates": original_gates,
                "depth": original_depth,
            },
            "optimized": {
                "gates": result.gate_count(),
                "depth": result.depth(),
            },
            "reduction": {
                "gates_removed": original_gates - result.gate_count(),
                "depth_reduced": original_depth - result.depth(),
            },
            "passes_applied": [type(p).__name__ for p in selected],
        })
    except Exception as e:
        return _safe_error(e)

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
        return _safe_error(e)


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
        from quanta.visualize_svg import to_html

        # Security: validate code before execution
        _validate_code(code)

        namespace = _build_sandbox_namespace()

        exec(code, namespace)  # noqa: S102 — sandboxed, restricted builtins

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
        return _safe_error(e)


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
        return _safe_error(e)


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
        z_stabs = code._z_stabilizers

        def _is_logical_success(err_u8: np.ndarray, corr_indices: list[int]) -> bool:
            corr_u8 = np.zeros(n, dtype=np.uint8)
            for q in corr_indices:
                if 0 <= q < n:
                    corr_u8[q] ^= 1
            residual = err_u8 ^ corr_u8
            # Check zero residual Z-stabilizer syndrome AND trivial F2 homology
            if np.any((code.H_Z @ residual) % 2):
                return False
            return int((code.logical_z @ residual) % 2) == 0

        for _ in range(rounds):
            error_mask = (rng.random(n) < error_rate).astype(np.uint8)
            if not error_mask.any():
                mwpm_success += 1
                uf_success += 1
                continue

            syndrome_z = (code.H_Z @ error_mask) % 2

            mwpm_result = mwpm.decode(
                syndrome_z, distance, stabilizers=z_stabs, error_type="X"
            )
            uf_result = uf.decode(
                syndrome_z, distance, stabilizers=z_stabs, error_type="X"
            )

            if _is_logical_success(error_mask, mwpm_result.correction):
                mwpm_success += 1
            if _is_logical_success(error_mask, uf_result.correction):
                uf_success += 1

        mwpm_rate = mwpm_success / rounds if rounds > 0 else 0.0
        uf_rate = uf_success / rounds if rounds > 0 else 0.0

        return json.dumps({
            "code_params": code.code_params,
            "error_rate": error_rate,
            "rounds": rounds,
            "mwpm": {
                "name": "Minimum Weight Perfect Matching",
                "success_rate": round(mwpm_rate, 4),
                "logical_error_rate": round(1.0 - mwpm_rate, 4),
                "complexity": "O(n³)",
            },
            "union_find": {
                "name": "Union-Find (Delfosse-Nickerson)",
                "success_rate": round(uf_rate, 4),
                "logical_error_rate": round(1.0 - uf_rate, 4),
                "complexity": "O(n·α(n)) ≈ O(n)",
            },
            "winner": "MWPM" if mwpm_rate > uf_rate else (
                "Union-Find" if uf_rate > mwpm_rate else "Tied"
            ),
            "explanation": (
                f"At p={error_rate} on [[{n},1,{distance}]]: "
                f"MWPM {mwpm_rate:.1%} vs UF {uf_rate:.1%} logical fidelity "
                "(verified via residual syndrome + F2 logical homology). "
                "MWPM is optimal; Union-Find is near-linear O(n·α(n))."
            ),
        })
    except Exception as e:
        return _safe_error(e)


# ═══════════════════════════════════════════
#  Tool 19: Option Greeks
# ═══════════════════════════════════════════

@mcp.tool()
def option_greeks(
    spot: float = 100.0,
    strike: float = 105.0,
    volatility: float = 0.2,
    rate: float = 0.05,
    time_to_expiry: float = 1.0,
    payoff: str = "european_call",
) -> str:
    """Compute option Greeks (delta, gamma, vega, theta, rho).

    Uses finite-difference Monte Carlo for sensitivity analysis.

    Args:
        spot: Current asset price.
        strike: Strike price.
        volatility: Annualized volatility (σ).
        rate: Risk-free interest rate.
        time_to_expiry: Years to expiry.
        payoff: "european_call" or "european_put".
    """
    try:
        from quanta.layer3.monte_carlo import compute_greeks as _greeks
        result = _greeks(
            spot=spot, strike=strike, volatility=volatility,
            rate=rate, time_to_expiry=time_to_expiry,
            payoff=payoff, n_samples=50_000, seed=42,
        )
        return json.dumps({
            "method": result.method,
            "delta": round(result.delta, 6),
            "gamma": round(result.gamma, 6),
            "vega": round(result.vega, 6),
            "theta": round(result.theta, 6),
            "rho": round(result.rho, 6),
            "summary": result.summary(),
            "params": {
                "spot": spot, "strike": strike,
                "volatility": volatility, "payoff": payoff,
            },
        })
    except Exception as exc:
        return _safe_error(exc)


# ═══════════════════════════════════════════
#  Tool: Portfolio Optimize (Ising QAOA)
# ═══════════════════════════════════════════

@mcp.tool()
def portfolio_optimize(
    assets: list[dict[str, Any]] | None = None,
    budget: int = 2,
    risk_aversion: float = 0.5,
    layers: int = 2,
    shots: int = 2048,
    seed: int = 42,
) -> str:
    """Optimize asset selection via Markowitz QUBO mapped to a gate-based Ising QAOA circuit.

    Args:
        assets: List of asset dicts with ``name``, ``return``, and ``volatility``
            (defaults to a 4-asset benchmark universe when omitted).
        budget: Exact number of assets to select (cardinality constraint K).
        risk_aversion: Risk-aversion weight q >= 0.
        layers: Number of Ising QAOA layers p.
        shots: Measurement shots for portfolio sampling.
        seed: Random seed.
    """
    try:
        from quanta.layer3.finance import portfolio_optimize as _portfolio_opt

        if assets is None:
            assets = [
                {"name": "TECH", "return": 0.16, "risk": 0.24},
                {"name": "BONDS", "return": 0.05, "risk": 0.06},
                {"name": "GOLD", "return": 0.09, "risk": 0.14},
                {"name": "ENERGY", "return": 0.12, "risk": 0.20},
            ]

        res = _portfolio_opt(
            assets=assets,
            budget=budget,
            risk_aversion=risk_aversion,
            layers=layers,
            shots=shots,
            seed=seed,
        )
        return json.dumps({
            "algorithm": "Markowitz Ising QAOA",
            "selected_assets": res.selected,
            "bitstring": res.selection_bits,
            "expected_return": round(res.expected_return, 6),
            "portfolio_risk": round(res.portfolio_risk, 6),
            "sharpe_ratio": round(res.sharpe_ratio, 4),
            "top_portfolios": [
                {
                    "bitstring": p["bits"],
                    "score": round(float(p["objective"]), 6),
                    "probability": round(float(p["probability"]), 4),
                }
                for p in res.all_portfolios[:5]
            ],
        })
    except Exception as exc:
        return _safe_error(exc)


# ═══════════════════════════════════════════
#  Tool 20: QEC Diagnose
# ═══════════════════════════════════════════

@mcp.tool()
def qec_diagnose(
    code: str = "bitflip",
    syndrome: str = "11",
) -> str:
    """Diagnose a quantum error from syndrome measurement.

    Given a QEC code and syndrome bitstring, returns the
    correction action needed.

    Args:
        code: "bitflip", "phaseflip", "steane", or "shor".
        syndrome: Measured syndrome bitstring (e.g., "11", "001011", "10000000").
    """
    try:
        from quanta.qec.codes import (
            BitFlipCode,
            PhaseFlipCode,
            ShorCode,
            SteaneCode,
            correct_error,
        )
        code_map = {
            "bitflip": BitFlipCode,
            "phaseflip": PhaseFlipCode,
            "steane": SteaneCode,
            "shor": ShorCode,
        }
        code_lower = code.lower().replace(" ", "").replace("-", "")
        if code_lower not in code_map:
            return json.dumps({
                "error": f"Unknown code: {code}",
                "available": list(code_map.keys()),
            })

        qec = code_map[code_lower]()
        correction = correct_error(qec, syndrome)
        info = qec.info

        return json.dumps({
            "code": info.name,
            "notation": f"[[{info.n},{info.k},{info.d}]]",
            "syndrome": syndrome,
            "correction": correction,
            "correctable_errors": info.correctable_errors,
            "lookup_table": qec.lookup_table(),
        })
    except Exception as exc:
        return _safe_error(exc)


# ═══════════════════════════════════════════
#  Tool 21: Fault-Tolerant Resource Estimation
# ═══════════════════════════════════════════

def _extract_circuit_def(circuit_code: str) -> Any:
    """Safely executes code in sandbox and returns the CircuitDefinition."""
    _validate_code(circuit_code)
    sandbox = _build_sandbox_namespace()
    exec(circuit_code, sandbox)  # noqa: S102
    for v in sandbox.values():
        if hasattr(v, "build"):
            return v
    raise ValueError("No @circuit function found in code.")


@mcp.tool()
def estimate_fault_tolerant_cost(
    circuit_code: str,
    target_logical_error_rate: float = 1e-10,
    physical_error_rate: float = 1e-3,
) -> str:
    """Estimates fault-tolerant quantum hardware resources (Litinski 2019 / Fowler 2012).

    Calculates required surface code distance via spacetime volume V = N_patches * D,
    Ross-Selinger optimal rotation synthesis T-counts, 1-level vs 2-level
    (15-to-1)^2 magic-state factory selection, routing overhead, and runtime.

    Args:
        circuit_code: Python code defining a @circuit function.
        target_logical_error_rate: Target error rate per algorithm run (default: 1e-10).
        physical_error_rate: Physical 2-qubit gate error rate (default: 1e-3, Willow standard).
    """
    try:
        import math as _m

        from quanta.dag.dag_circuit import DAGCircuit

        p_th = 0.01  # Circuit-level rotated surface code threshold (Fowler et al. 2012)
        if physical_error_rate >= p_th:
            return json.dumps({
                "error": (
                    f"physical_error_rate={physical_error_rate} >= threshold ({p_th}); "
                    "surface code cannot suppress logical errors above threshold."
                ),
            })

        circ_fn = _extract_circuit_def(circuit_code)
        dag = DAGCircuit.from_builder(circ_fn.build())

        n_logical = dag.num_qubits
        total_gates = dag.gate_count()
        depth = dag.depth()

        # Pass 1: Classify Clifford vs fixed non-Clifford vs arbitrary synthesis rotations
        fixed_t_count = 0
        arb_rotations = 0
        clifford_count = 0
        for node in dag.op_nodes():
            is_cliff, exact_t, n_arb = _classify_gate_clifford_and_t(
                node.gate_name, node.params, eps_synth=None
            )
            if is_cliff:
                clifford_count += 1
            else:
                fixed_t_count += exact_t
                arb_rotations += n_arb

        # Pass 2: Ross-Selinger optimal Clifford+T synthesis budget for arbitrary rotations
        # Allocate 1/3 of total error budget to synthesis, 1/3 to distillation, 1/3 to memory
        if arb_rotations > 0:
            eps_synth = max(1e-30, target_logical_error_rate / (3.0 * arb_rotations))
            t_per_rotation = max(1, _m.ceil(3.0 * _m.log2(1.0 / eps_synth)))
        else:
            t_per_rotation = 0

        t_count = fixed_t_count + arb_rotations * t_per_rotation

        # Litinski (Quantum 3, 128, 2019) routing & ancilla patch overhead: ~1.5x + 1 patches
        routing_logical_patches = max(1, _m.ceil(0.5 * n_logical) + 1)
        total_logical_patches = n_logical + routing_logical_patches

        # Logical spacetime volume V = N_patches * D_logical (Fowler 2012 / Litinski 2019)
        logical_time_steps = max(1, depth, t_count)
        spacetime_volume = total_logical_patches * logical_time_steps
        topo_budget = (
            target_logical_error_rate / 3.0 if t_count > 0 else target_logical_error_rate
        )

        # Solve V * 0.1 * (p_phys / p_th)^((d + 1) / 2) <= topo_budget for odd d >= 3
        d = 3
        if physical_error_rate > 0:
            while d <= 55:
                p_l_per_patch_cycle = 0.1 * (
                    (physical_error_rate / p_th) ** ((d + 1) / 2.0)
                )
                if spacetime_volume * p_l_per_patch_cycle <= topo_budget:
                    break
                d += 2

        # Physical qubits per rotated surface code patch = 2 * d^2 (d^2 data + (d^2-1) syndrome)
        data_physical_qubits = 2 * (d ** 2) * n_logical
        routing_physical_qubits = 2 * (d ** 2) * routing_logical_patches

        # Magic-state distillation factory selection (1-level 15-to-1 vs 2-level (15-to-1)^2)
        if t_count == 0:
            factory_levels = 0
            factory_type = "none (Clifford-only circuit)"
            t_factory_qubits = 0
            factory_cycles_per_t = 0
        else:
            eps_per_t = target_logical_error_rate / (3.0 * t_count)
            p_l1 = 35.0 * (physical_error_rate ** 3)
            d_l1 = max(3, (d // 2) | 1)
            if p_l1 <= eps_per_t:
                factory_levels = 1
                factory_type = "15-to-1 Reed-Muller (1-level)"
                # 15 logical patches at distance d_l1 (or d if d_l1 topological error > eps_per_t)
                while d_l1 < d and 15 * 0.1 * ((physical_error_rate / p_th) ** ((d_l1 + 1) / 2.0)) > eps_per_t:
                    d_l1 += 2
                t_factory_qubits = 15 * 2 * (d_l1 ** 2)
                factory_cycles_per_t = 6 * d_l1
            else:
                factory_levels = 2
                factory_type = "(15-to-1)^2 Reed-Muller (2-level)"
                # Litinski 2-level factory: two L1 blocks at d_l1 feeding one L2 block at d
                t_factory_qubits = 2 * (15 * 2 * (d_l1 ** 2)) + 15 * 2 * (d ** 2)
                factory_cycles_per_t = 6 * d

        total_physical_qubits = (
            data_physical_qubits + routing_physical_qubits + t_factory_qubits
        )

        # Surface code syndrome cycle time (~1.0 microsecond on superconducting Willow/Starling)
        cycle_time_us = 1.0
        total_code_cycles = max(depth * d, t_count * factory_cycles_per_t)
        total_runtime_seconds = (total_code_cycles * cycle_time_us) / 1e6

        return json.dumps({
            "logical_qubits": n_logical,
            "routing_logical_patches": routing_logical_patches,
            "spacetime_volume": spacetime_volume,
            "total_gates": total_gates,
            "circuit_depth": depth,
            "clifford_gates": clifford_count,
            "magic_t_gates_required": t_count,
            "synthesis_t_per_rotation": t_per_rotation,
            "surface_code_distance": d,
            "factory_levels": factory_levels,
            "factory_type": factory_type,
            "physical_qubits_data": data_physical_qubits,
            "physical_qubits_routing": routing_physical_qubits,
            "physical_qubits_t_factory": t_factory_qubits,
            "total_physical_qubits": total_physical_qubits,
            "estimated_runtime_seconds": round(total_runtime_seconds, 8),
            "willow_compatible": total_physical_qubits <= 105,
            "architecture": "Rotated Surface Code (Litinski 2019 / Fowler 2012)",
        })
    except Exception as exc:
        return _safe_error(exc)


# ═══════════════════════════════════════════
#  Tool 22: AI Reasoning Circuit Evaluator
# ═══════════════════════════════════════════

@mcp.tool()
def quanta_reasoning_eval(circuit_code: str) -> str:
    """Evaluates circuit complexity and provides reasoning feedback for AI agents.

    Analyzes gate cancellation potential, 2-qubit entangling capacity,
    exact Clifford vs non-Clifford classification, simulation tractability,
    and suggests architectural self-corrections.

    Args:
        circuit_code: Python code defining a @circuit function.
    """
    try:
        from quanta.compiler.passes.optimize import CancelInverses
        from quanta.dag.dag_circuit import DAGCircuit

        circ_fn = _extract_circuit_def(circuit_code)
        dag = DAGCircuit.from_builder(circ_fn.build())

        num_qubits = dag.num_qubits
        total_gates = dag.gate_count()
        depth = dag.depth()

        # Exact Clifford vs non-Clifford gate classification
        two_qubit_count = 0
        non_clifford_count = 0
        for node in dag.op_nodes():
            if len(node.qubits) >= 2:
                two_qubit_count += 1
            is_cliff, _, _ = _classify_gate_clifford_and_t(
                node.gate_name, node.params, eps_synth=None
            )
            if not is_cliff:
                non_clifford_count += 1

        entangling_ratio = (two_qubit_count / total_gates) if total_gates > 0 else 0.0

        # Optimization check
        cancel_pass = CancelInverses()
        optimized_dag = cancel_pass.run(dag)
        cancellable_gates = total_gates - optimized_dag.gate_count()

        # Simulation regime recommendation
        if non_clifford_count == 0:
            regime = "Clifford Stabilizer (efficient up to 1,000+ qubits via PauliFrame)"
        elif num_qubits <= 30:
            regime = "Dense Statevector (exact simulation supported on M5 Pro 48GB)"
        elif entangling_ratio < 0.2:
            regime = "Matrix Product State (low entanglement, scalable to 100+ qubits)"
        else:
            regime = "High-entanglement non-Clifford (classical intractable, requires QPU)"

        suggestions = []
        if cancellable_gates > 0:
            suggestions.append(
                f"Found {cancellable_gates} redundant gates that can be eliminated."
            )
        if entangling_ratio > 0.6:
            suggestions.append(
                "High 2-qubit gate density may cause significant decoherence on NISQ QPUs."
            )
        if num_qubits > 30 and non_clifford_count > 0 and entangling_ratio >= 0.2:
            suggestions.append(
                "Circuit exceeds statevector limits. Consider tensor networks or Clifford."
            )

        return json.dumps({
            "num_qubits": num_qubits,
            "total_gates": total_gates,
            "circuit_depth": depth,
            "entangling_gate_ratio": round(entangling_ratio, 3),
            "non_clifford_gates": non_clifford_count,
            "cancellable_gates": cancellable_gates,
            "recommended_simulation_regime": regime,
            "agent_feedback": (
                suggestions if suggestions else ["Circuit structure is clean."]
            ),
        })
    except Exception as exc:
        return _safe_error(exc)


# ═══════════════════════════════════════════
#  Tool 23: Transpile for Target Architecture
# ═══════════════════════════════════════════

@mcp.tool()
def transpile_for_target(
    circuit_code: str,
    target: str = "ibm_heron",
) -> str:
    """Transpiles a circuit into native hardware gate set and ISA.

    Supports:
      - 'ibm_heron': Native RZ, SX, X, CZ, ECR
      - 'google_willow': Native PhasedXZ, CZ, iSWAP
      - 'ionq_aria': Native GPI, GPI2, MS

    Args:
        circuit_code: Python code defining a @circuit function.
        target: Target hardware platform ("ibm_heron", "google_willow", "ionq_aria").
    """
    try:
        from quanta.compiler.passes.optimize import CancelInverses, MergeRotations
        from quanta.compiler.passes.translate import TranslateToTarget
        from quanta.compiler.pipeline import CompilerPipeline
        from quanta.dag.dag_circuit import DAGCircuit
        from quanta.export.qasm import to_qasm

        circ_fn = _extract_circuit_def(circuit_code)
        dag = DAGCircuit.from_builder(circ_fn.build())

        orig_count = dag.gate_count()
        orig_depth = dag.depth()

        # Translation map based on target
        target_lower = target.lower()
        if "ibm" in target_lower or "heron" in target_lower:
            target_key = "ibm"
            native_gates = {"RZ", "SX", "X", "CZ", "ECR", "MEASURE"}
        elif "google" in target_lower or "willow" in target_lower:
            target_key = "google"
            native_gates = {"RZ", "SX", "CZ", "iSWAP", "MEASURE"}
        else:
            target_key = "universal"
            native_gates = {"RX", "RY", "RZ", "MS", "MEASURE"}

        pipeline = CompilerPipeline([
            TranslateToTarget(target_gates=target_key),
            CancelInverses(),
            MergeRotations(),
        ])
        transpiled_dag = pipeline.run(dag)

        qasm_code = to_qasm(transpiled_dag)

        return json.dumps({
            "target": target,
            "original_gate_count": orig_count,
            "original_depth": orig_depth,
            "transpiled_gate_count": transpiled_dag.gate_count(),
            "transpiled_depth": transpiled_dag.depth(),
            "target_native_gates": list(native_gates),
            "qasm3_output": qasm_code,
        })
    except Exception as exc:
        return _safe_error(exc)


# ═══════════════════════════════════════════
#  Resource: SDK Info
# ═══════════════════════════════════════════

@mcp.resource("quanta://info")
def sdk_info() -> str:
    """Quanta SDK version and capabilities."""
    return json.dumps({
        "name": "Quanta Quantum SDK",
        "version": "1.2.0",
        "description": "AI-native quantum computing SDK",
        "total_gates": 31,
        "total_tools": 25,
        "tool_categories": {
            "education": ["create_bell_state", "draw_circuit",
                         "list_gates", "explain_result"],
            "research": ["run_circuit", "mps_circuit_simulate", "grover_search",
                        "shor_factor", "simulate_noise", "qaoa_optimize",
                        "optimize_circuit", "surface_code_simulate",
                        "compare_decoders", "qec_diagnose",
                        "estimate_fault_tolerant_cost"],
            "agentic_ai": ["quanta_reasoning_eval", "transpile_for_target"],
            "machine_learning": ["qml_classify"],
            "business": ["monte_carlo_price", "option_greeks",
                        "portfolio_optimize", "cluster_data"],
            "hardware": ["run_on_ibm", "ibm_backends", "ibm_job_result"],
        },
        "capabilities": [
            "31 quantum gates (full IBM Quantum parity + Google/IonQ native)",
            "Apple Silicon Metal/MLX GPU acceleration (up to 400x speedup, 30+ qubits)",
            "Statevector simulation (up to 30 qubits)",
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
            "RX(θ)": "rz(π/2) · sx · rz(θ + π) · sx · rz(π/2)",
            "RY(θ)": "sx · rz(θ + π) · sx · rz(π)",
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
