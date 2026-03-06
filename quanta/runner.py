"""
quanta.runner -- Main execution and parameter sweep functions.

The run() function is the main entry point of the SDK.
The sweep() function enables batch execution with varying parameters.

Pipeline:
  1. CircuitDefinition.build() -> CircuitBuilder
  2. DAGCircuit.from_builder() -> DAG
  3. CompilerPipeline (optional) -> optimization
  4. Simulator -> statevector computation
  5. Sampling -> measurement results
  6. Result construction

Example:
    >>> from quanta import circuit, H, CX, measure, run, sweep
    >>> @circuit(qubits=2)
    ... def bell(q):
    ...     H(q[0])
    ...     CX(q[0], q[1])
    ...     return measure(q)
    >>> result = run(bell, shots=1024)
    >>> results = sweep(bell, params={"theta": [0, 1.57, 3.14]})
"""

from __future__ import annotations

import numpy as np

from quanta.core.circuit import CircuitDefinition
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result
from quanta.simulator.statevector import StateVectorSimulator

# -- Public API --
__all__ = ["run", "sweep"]


def run(
    circuit: CircuitDefinition,
    shots: int = 1024,
    seed: int | None = None,
) -> Result:
    """Executes a quantum circuit and returns results.

    This function is the SDK's main orchestrator. It runs all stages
    sequentially and returns a clean Result.

    Args:
        circuit: Circuit defined with @circuit.
        shots: Number of measurement repetitions. Default 1024.
        seed: Random seed for reproducibility.

    Returns:
        Result: Measurement results, probabilities, and circuit metadata.

    Raises:
        QuantaError: If circuit is invalid or simulation fails.
    """
    if not isinstance(circuit, CircuitDefinition):
        raise QuantaError(
            f"run() expects a @circuit-defined circuit. "
            f"Given type: {type(circuit).__name__}"
        )

    if shots < 1:
        raise QuantaError(f"Shot count must be positive, given: {shots}")

    # Stage 1: Build circuit (lazy instructions)
    builder = circuit.build()

    # Stage 2: Build DAG
    dag = DAGCircuit.from_builder(builder)

    # Stage 3: Compile (to be added in v0.2)
    # compiled = CompilerPipeline().run(dag)

    # Stage 4: Simulate
    simulator = StateVectorSimulator(dag.num_qubits, seed=seed)

    # Apply gates in topological order
    for op in dag.op_nodes():
        simulator.apply(op.gate_name, op.qubits, op.params)

    # Stage 5: Sample
    counts = simulator.sample(shots)

    # Stage 6: Build result
    # Filter by measured qubits (partial measurement support)
    if dag.measurement and dag.measurement.qubits:
        measured = dag.measurement.qubits
        counts = _filter_measured_qubits(counts, measured, dag.num_qubits)

    return Result(
        counts=counts,
        shots=shots,
        num_qubits=dag.num_qubits,
        circuit_name=circuit.name,
        gate_count=dag.gate_count(),
        depth=dag.depth(),
        statevector=simulator.state,
    )


def _filter_measured_qubits(
    counts: dict[str, int],
    measured_qubits: tuple[int, ...],
    num_qubits: int,
) -> dict[str, int]:
    """Returns results for only the measured qubits.

    Args:
        counts: Full measurement results.
        measured_qubits: Measured qubit indices.
        num_qubits: Total qubit count.

    Returns:
        Filtered counts dict.
    """
    filtered: dict[str, int] = {}

    for bitstring, count in counts.items():
        # Extract only the bits of measured qubits
        measured_bits = "".join(bitstring[q] for q in measured_qubits)
        filtered[measured_bits] = filtered.get(measured_bits, 0) + count

    return filtered


def sweep(
    circuit: CircuitDefinition,
    params: dict[str, list[float] | np.ndarray],
    shots: int = 1024,
    seed: int | None = None,
) -> list[Result]:
    """Runs a circuit with multiple parameter values (batch execution).

    Efficient for VQE/QAOA: builds circuit structure once, then
    re-executes with different parameter values.

    Args:
        circuit: Parameterized circuit defined with @circuit.
        params: Dict of {param_name: [values]}. All lists must be same length.
        shots: Number of shots per parameter set.
        seed: Random seed for reproducibility.

    Returns:
        List of Result objects, one per parameter combination.

    Example:
        >>> import numpy as np
        >>> @circuit(qubits=1)
        ... def rotation(q, theta=0.0):
        ...     RZ(q[0], theta)
        ...     return measure(q)
        >>> results = sweep(rotation, params={"theta": np.linspace(0, 3.14, 10)})
        >>> for r in results:
        ...     print(r.most_frequent)
    """
    # Validate params
    lengths = [len(v) for v in params.values()]
    if not lengths:
        return [run(circuit, shots=shots, seed=seed)]
    if len(set(lengths)) > 1:
        raise QuantaError(
            f"All parameter lists must have the same length. Got: {lengths}"
        )

    num_runs = lengths[0]
    param_names = list(params.keys())
    results = []

    for i in range(num_runs):
        kwargs = {name: float(params[name][i]) for name in param_names}
        result = run(circuit, shots=shots, seed=seed, **kwargs)
        results.append(result)

    return results
