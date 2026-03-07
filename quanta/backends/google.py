"""
quanta.backends.google -- Google Quantum Engine backend.

Runs Quanta circuits on Google quantum hardware via Cirq.
Uses the QASM bridge: Quanta DAG -> QASM 3.0 -> Cirq -> Google Quantum Engine.

Requirements:
    pip install cirq-google

Setup:
    gcloud auth application-default login

Example:
    >>> from quanta.backends.google import GoogleBackend
    >>> backend = GoogleBackend(simulate_locally=True)
    >>> result = run(bell, shots=1024, backend=backend)

    >>> backend = GoogleBackend(
    ...     project_id="my-project",
    ...     processor_id="rainbow",
    ... )
    >>> result = run(bell, shots=1024, backend=backend)

Available processors (2026):
    rainbow     23 qubits   Sycamore
    weber       53 qubits   Sycamore
    Willow      105 qubits  (limited access)
"""

from __future__ import annotations

from typing import Any

from quanta.backends.base import Backend
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result

__all__ = ["GoogleBackend"]


class GoogleBackendError(QuantaError):
    """Google Quantum backend error."""


# Quanta gate -> QASM gate name
_QASM_GATE_MAP: dict[str, str] = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t", "CX": "cx", "CZ": "cz",
    "CY": "cy", "SWAP": "swap", "CCX": "ccx",
    "RX": "rx", "RY": "ry", "RZ": "rz",
}

# Quanta gate -> Cirq gate (for manual conversion fallback)
_CIRQ_GATE_BUILDERS: dict[str, str] = {
    "H": "H", "X": "X", "Y": "Y", "Z": "Z",
    "S": "S", "T": "T",
    "CX": "CNOT", "CZ": "CZ", "SWAP": "SWAP",
}


class GoogleBackend(Backend):
    """Runs circuits on Google Quantum Engine.

    Uses Cirq as the transport layer. Cirq is loaded lazily --
    only imported when execute() is called.

    Args:
        project_id: Google Cloud project ID.
        processor_id: Processor ID (e.g. "rainbow", "weber").
        simulate_locally: If True, uses local Cirq simulator (for testing).
    """

    def __init__(
        self,
        project_id: str = "",
        processor_id: str = "",
        simulate_locally: bool = False,
    ) -> None:
        self._project_id = project_id
        self._processor_id = processor_id
        self._simulate_locally = simulate_locally
        self._cirq: Any = None
        self._cirq_google: Any = None

    @property
    def name(self) -> str:
        mode = "local" if self._simulate_locally else self._processor_id
        return f"google_{mode}"

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
                "Install: pip install cirq-google"
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

    def _dag_to_qasm(self, dag: DAGCircuit) -> str:
        """Converts DAG to QASM 3.0 string for Cirq import."""
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "",
            f"qreg q[{dag.num_qubits}];",
            f"creg c[{dag.num_qubits}];",
            "",
        ]

        for op in dag.op_nodes():
            qasm_name = _QASM_GATE_MAP.get(op.gate_name)
            if qasm_name is None:
                raise GoogleBackendError(
                    f"Gate '{op.gate_name}' is not supported by Google backend. "
                    f"Supported: {list(_QASM_GATE_MAP.keys())}"
                )
            qubit_args = ", ".join(f"q[{q}]" for q in op.qubits)
            if op.params:
                param_str = ", ".join(f"{p:.10f}" for p in op.params)
                lines.append(f"{qasm_name}({param_str}) {qubit_args};")
            else:
                lines.append(f"{qasm_name} {qubit_args};")

        lines.append("")
        measured = range(dag.num_qubits)
        if dag.measurement and dag.measurement.qubits:
            measured = dag.measurement.qubits
        for i, q in enumerate(measured):
            lines.append(f"measure q[{q}] -> c[{i}];")

        return "\n".join(lines)

    def _dag_to_cirq_manual(self, dag: DAGCircuit) -> Any:
        """Converts DAG to Cirq circuit via direct gate mapping (fallback)."""
        cirq = self._cirq
        qubits = [cirq.LineQubit(i) for i in range(dag.num_qubits)]
        ops = []

        for op in dag.op_nodes():
            cirq_name = _CIRQ_GATE_BUILDERS.get(op.gate_name)
            if cirq_name:
                gate = getattr(cirq, cirq_name)
                cirq_qubits = [qubits[q] for q in op.qubits]
                ops.append(gate(*cirq_qubits))
            elif op.gate_name in ("RX", "RY", "RZ") and op.params:
                gate_cls = getattr(cirq, f"r{'xyz'['XYZ'.index(op.gate_name[1])]}")
                ops.append(gate_cls(op.params[0])(qubits[op.qubits[0]]))

        # Add measurement
        measured = list(range(dag.num_qubits))
        if dag.measurement and dag.measurement.qubits:
            measured = list(dag.measurement.qubits)
        ops.append(cirq.measure(*[qubits[q] for q in measured], key="result"))

        return cirq.Circuit(ops)

    def execute(
        self,
        dag: DAGCircuit,
        shots: int = 1024,
        seed: int | None = None,
    ) -> Result:
        """Runs circuit on Google Quantum Engine.

        Args:
            dag: Compiled DAG circuit.
            shots: Number of measurement repetitions.
            seed: Random seed (only used for local simulation).

        Returns:
            Result with measurement counts.
        """
        self._ensure_cirq()
        cirq = self._cirq

        # Try QASM bridge first, fall back to manual conversion
        qasm_str = self._dag_to_qasm(dag)
        try:
            from cirq.contrib.qasm_import import circuit_from_qasm
            cirq_circuit = circuit_from_qasm(qasm_str)
        except (ImportError, Exception):
            cirq_circuit = self._dag_to_cirq_manual(dag)

        if self._simulate_locally:
            simulator = cirq.Simulator(seed=seed)
            result = simulator.run(cirq_circuit, repetitions=shots)
        else:
            engine = self._cirq_google.Engine(project_id=self._project_id)
            sampler = engine.get_sampler(processor_id=self._processor_id)
            result = sampler.run(cirq_circuit, repetitions=shots)

        counts = self._cirq_result_to_counts(result)

        return Result(
            counts=counts,
            shots=shots,
            num_qubits=dag.num_qubits,
        )

    @staticmethod
    def _cirq_result_to_counts(result: Any) -> dict[str, int]:
        """Converts Cirq Result to measurement counts dict."""
        try:
            hist = result.histogram(key="result")
            n_qubits = max(1, max(v.bit_length() for v in hist.keys()) if hist else 1)
            return {format(k, f"0{n_qubits}b"): v for k, v in hist.items()}
        except Exception:
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
