"""
quanta.backends.google -- Google Quantum Engine backend.

Runs Quanta circuits on Google quantum hardware.

Strategy:
  Quanta -> QASM 3.0 -> Cirq -> Google Quantum Engine
  QASM is used as the common language -- Quanta does NOT depend on Cirq.
  Cirq is only loaded at runtime (lazy import).

Requirements:
    pip install cirq-google

Setup:
    gcloud auth application-default login

Example:
    >>> from quanta.backends.google import GoogleBackend
    >>> backend = GoogleBackend(
    ...     project_id="my-project",
    ...     processor_id="rainbow",
    ... )
    >>> result = backend.execute(bell, shots=1024)
"""

from __future__ import annotations

from typing import Any

from quanta.backends.base import Backend
from quanta.core.circuit import CircuitDefinition
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.export.qasm import to_qasm
from quanta.result import Result

# -- Public API --
__all__ = ["GoogleBackend"]


class GoogleBackendError(QuantaError):
    """Google Quantum backend error."""


class GoogleBackend(Backend):
    """Runs circuits on Google Quantum Engine.

    Uses the QASM bridge: Quanta -> QASM -> Cirq -> Google Quantum Engine.

    Args:
        project_id: Google Cloud project ID.
        processor_id: Processor ID (e.g. "rainbow", "weber").
        simulate_locally: If True, uses local Cirq simulator (for testing).

    Example:
        >>> backend = GoogleBackend("my-project", "rainbow")
        >>> result = backend.execute(bell, shots=1024)
    """

    name = "google_quantum"

    def __init__(
        self,
        project_id: str = "",
        processor_id: str = "",
        simulate_locally: bool = False,
    ) -> None:
        self._project_id = project_id
        self._processor_id = processor_id
        self._simulate_locally = simulate_locally
        self._engine: Any = None
        self._cirq: Any = None
        self._cirq_google: Any = None

    def _ensure_cirq(self) -> None:
        """Lazy imports Cirq -- loaded only on first use."""
        if self._cirq is not None:
            return

        try:
            import cirq
            self._cirq = cirq
        except ImportError as e:
            raise GoogleBackendError(
                "Google backend requires the 'cirq' package.\n"
                "Install: pip install cirq-google\n"
                "See: docs/INSTALL_TR.md"
            ) from e

        if not self._simulate_locally:
            try:
                import cirq_google
                self._cirq_google = cirq_google
            except ImportError as e:
                raise GoogleBackendError(
                    "Google Quantum Engine requires 'cirq-google'.\n"
                    "Install: pip install cirq-google"
                ) from e

    def _quanta_to_cirq(self, circuit: CircuitDefinition) -> Any:
        """Converts Quanta circuit to Cirq circuit (via QASM bridge).

        Strategy:
          Quanta -> QASM 3.0 -> cirq.contrib.qasm_import -> Cirq Circuit
          Quanta never directly depends on the Cirq API.
        """
        self._ensure_cirq()
        cirq = self._cirq

        # 1. Quanta -> QASM
        qasm_str = to_qasm(circuit)

        # 2. QASM -> Cirq
        try:
            from cirq.contrib.qasm_import import circuit_from_qasm
            cirq_circuit = circuit_from_qasm(qasm_str)
        except (ImportError, Exception):
            # Fallback: manual conversion
            cirq_circuit = self._manual_convert(circuit)

        return cirq_circuit

    def _manual_convert(self, circuit: CircuitDefinition) -> Any:
        """Creates Cirq circuit manually if QASM import fails."""
        cirq = self._cirq
        builder = circuit.build()
        dag = DAGCircuit.from_builder(builder)

        # Qubit mapping
        qubits = [cirq.LineQubit(i) for i in range(dag.num_qubits)]

        # Gate mapping
        gate_map = {
            "H": lambda qs: cirq.H(qubits[qs[0]]),
            "X": lambda qs: cirq.X(qubits[qs[0]]),
            "Y": lambda qs: cirq.Y(qubits[qs[0]]),
            "Z": lambda qs: cirq.Z(qubits[qs[0]]),
            "S": lambda qs: cirq.S(qubits[qs[0]]),
            "T": lambda qs: cirq.T(qubits[qs[0]]),
            "CX": lambda qs: cirq.CNOT(qubits[qs[0]], qubits[qs[1]]),
            "CZ": lambda qs: cirq.CZ(qubits[qs[0]], qubits[qs[1]]),
            "SWAP": lambda qs: cirq.SWAP(qubits[qs[0]], qubits[qs[1]]),
        }

        ops = []
        for op in dag.op_nodes():
            converter = gate_map.get(op.gate_name)
            if converter:
                ops.append(converter(op.qubits))

        # Add measurement
        if builder.measurement is not None:
            measured = builder.measurement.qubits or tuple(range(dag.num_qubits))
            ops.append(cirq.measure(*[qubits[q] for q in measured], key="result"))

        return cirq.Circuit(ops)

    def execute(
        self,
        circuit: CircuitDefinition,
        shots: int = 1024,
        seed: int | None = None,
    ) -> Result:
        """Runs circuit on Google Quantum Engine.

        Args:
            circuit: Quanta circuit.
            shots: Number of shots.
            seed: Seed for local simulation.

        Returns:
            Quanta Result object.
        """
        self._ensure_cirq()
        cirq = self._cirq
        cirq_circuit = self._quanta_to_cirq(circuit)

        if self._simulate_locally:
            # Local Cirq simulator (for testing)
            simulator = cirq.Simulator(seed=seed)
            result = simulator.run(cirq_circuit, repetitions=shots)
        else:
            # Run on Google Quantum Engine
            engine = self._cirq_google.Engine(project_id=self._project_id)
            sampler = engine.get_sampler(processor_id=self._processor_id)
            result = sampler.run(cirq_circuit, repetitions=shots)

        # Convert Cirq results to Quanta Result
        counts = self._cirq_result_to_counts(result)
        builder = circuit.build()
        dag = DAGCircuit.from_builder(builder)

        return Result(
            counts=counts,
            shots=shots,
            num_qubits=dag.num_qubits,
            circuit_name=circuit.name,
            gate_count=dag.gate_count(),
            depth=dag.depth(),
        )

    def _cirq_result_to_counts(self, result: Any) -> dict[str, int]:
        """Converts Cirq Result to Quanta counts."""
        try:
            hist = result.histogram(key="result")
            n_qubits = max(1, max(v.bit_length() for v in hist.keys()) if hist else 1)
            return {format(k, f"0{n_qubits}b"): v for k, v in hist.items()}
        except Exception:
            # Fallback: from DataFrame
            data = result.data
            counts: dict[str, int] = {}
            for _, row in data.iterrows():
                bits = "".join(str(int(v)) for v in row.values)
                counts[bits] = counts.get(bits, 0) + 1
            return counts

    def __repr__(self) -> str:
        mode = "local" if self._simulate_locally else "hardware"
        return (
            f"GoogleBackend(project='{self._project_id}', "
            f"processor='{self._processor_id}', mode={mode})"
        )
