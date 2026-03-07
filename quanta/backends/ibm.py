"""
quanta.backends.ibm -- IBM Quantum backend.

Runs Quanta circuits on IBM Quantum hardware via qiskit-ibm-runtime.
Uses the QASM bridge: Quanta DAG -> QASM 3.0 -> Qiskit -> IBM hardware.

Requirements:
    pip install qiskit-ibm-runtime

Setup:
    Export your IBM Quantum API token:
    export IBM_QUANTUM_TOKEN="your-token-here"

    Or get one at: https://quantum.ibm.com/

Example:
    >>> from quanta.backends.ibm import IBMBackend
    >>> backend = IBMBackend(simulate_locally=True)
    >>> result = run(bell, shots=1024, backend=backend)

    >>> backend = IBMBackend(backend_name="ibm_brisbane")
    >>> result = run(bell, shots=1024, backend=backend)
"""

from __future__ import annotations

import os

from quanta.backends.base import Backend
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result

__all__ = ["IBMBackend"]


class IBMBackendError(QuantaError):
    """IBM Quantum backend error."""


# Gate name mapping: Quanta -> QASM
_QASM_GATE_MAP: dict[str, str] = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t", "CX": "cx", "CZ": "cz",
    "CY": "cy", "SWAP": "swap", "CCX": "ccx",
    "RX": "rx", "RY": "ry", "RZ": "rz",
}


class IBMBackend(Backend):
    """Runs circuits on IBM Quantum hardware.

    Uses QASM 3.0 as the bridge between Quanta and Qiskit.
    Qiskit is loaded lazily -- only imported when execute() is called.

    Args:
        backend_name: IBM backend name (e.g. "ibm_brisbane", "ibm_osaka").
        token: IBM Quantum API token. Falls back to IBM_QUANTUM_TOKEN env var.
        simulate_locally: If True, uses Qiskit's local Aer simulator.

    Available backends (2026):
        ibm_brisbane    127 qubits  Eagle r3
        ibm_osaka       127 qubits  Eagle r3
        ibm_kyoto       127 qubits  Eagle r3
        ibm_sherbrooke  127 qubits  Eagle r3
    """

    def __init__(
        self,
        backend_name: str = "ibm_brisbane",
        token: str = "",
        simulate_locally: bool = False,
    ) -> None:
        self._backend_name = backend_name
        self._token = token or os.environ.get("IBM_QUANTUM_TOKEN", "")
        self._simulate_locally = simulate_locally

    @property
    def name(self) -> str:
        mode = "local" if self._simulate_locally else self._backend_name
        return f"ibm_{mode}"

    def _dag_to_qasm(self, dag: DAGCircuit) -> str:
        """Converts DAG to QASM 2.0 string for Qiskit compatibility."""
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
                raise IBMBackendError(
                    f"Gate '{op.gate_name}' is not supported by IBM backend. "
                    f"Supported: {list(_QASM_GATE_MAP.keys())}"
                )
            qubit_args = ", ".join(f"q[{q}]" for q in op.qubits)
            if op.params:
                param_str = ", ".join(f"{p:.10f}" for p in op.params)
                lines.append(f"{qasm_name}({param_str}) {qubit_args};")
            else:
                lines.append(f"{qasm_name} {qubit_args};")

        # Measurement
        lines.append("")
        measured = range(dag.num_qubits)
        if dag.measurement and dag.measurement.qubits:
            measured = dag.measurement.qubits
        for i, q in enumerate(measured):
            lines.append(f"measure q[{q}] -> c[{i}];")

        return "\n".join(lines)

    def execute(
        self,
        dag: DAGCircuit,
        shots: int = 1024,
        seed: int | None = None,
    ) -> Result:
        """Runs circuit on IBM Quantum hardware.

        Args:
            dag: Compiled DAG circuit.
            shots: Number of measurement repetitions.
            seed: Random seed (only used for local simulation).

        Returns:
            Result with measurement counts.
        """
        qasm_str = self._dag_to_qasm(dag)

        if self._simulate_locally:
            counts = self._run_local(qasm_str, shots, seed)
        else:
            counts = self._run_hardware(qasm_str, shots)

        return Result(
            counts=counts,
            shots=shots,
            num_qubits=dag.num_qubits,
        )

    def _run_local(
        self, qasm_str: str, shots: int, seed: int | None
    ) -> dict[str, int]:
        """Runs on local Qiskit Aer simulator."""
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
        except ImportError as e:
            raise IBMBackendError(
                "Local IBM simulation requires qiskit-aer.\n"
                "Install: pip install qiskit qiskit-aer"
            ) from e

        qc = QuantumCircuit.from_qasm_str(qasm_str)
        simulator = AerSimulator()
        result = simulator.run(qc, shots=shots, seed_simulator=seed).result()
        raw_counts = result.get_counts()

        # Qiskit returns bitstrings in reverse order, normalize
        return {k.replace(" ", ""): v for k, v in raw_counts.items()}

    def _run_hardware(
        self, qasm_str: str, shots: int
    ) -> dict[str, int]:
        """Runs on real IBM Quantum hardware via qiskit-ibm-runtime."""
        if not self._token:
            raise IBMBackendError(
                "IBM Quantum API token is required for hardware execution.\n"
                "Set IBM_QUANTUM_TOKEN environment variable or pass token= parameter.\n"
                "Get a token at: https://quantum.ibm.com/"
            )

        try:
            from qiskit import QuantumCircuit
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        except ImportError as e:
            raise IBMBackendError(
                "IBM hardware backend requires qiskit-ibm-runtime.\n"
                "Install: pip install qiskit-ibm-runtime"
            ) from e

        service = QiskitRuntimeService(channel="ibm_quantum", token=self._token)
        backend = service.backend(self._backend_name)

        qc = QuantumCircuit.from_qasm_str(qasm_str)
        sampler = SamplerV2(backend)
        job = sampler.run([qc], shots=shots)
        result = job.result()

        # Extract counts from SamplerV2 result
        pub_result = result[0]
        raw_counts = pub_result.data.c.get_counts()
        return {k.replace(" ", ""): v for k, v in raw_counts.items()}

    def __repr__(self) -> str:
        return (
            f"IBMBackend(backend='{self._backend_name}', "
            f"local={self._simulate_locally})"
        )
