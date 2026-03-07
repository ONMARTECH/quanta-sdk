"""
quanta.benchmark.benchpress_adapter -- Benchpress-compatible backend.

Implements the API pattern used by Benchpress (Nation et al., 2025)
so Quanta can be benchmarked alongside Qiskit, Cirq, and Braket.

API:
    backend = QuantaBenchpressBackend()
    circ = backend.new_circuit(num_qubits=4)
    backend.apply_gate(circ, "h", [0])
    backend.apply_gate(circ, "cx", [0, 1])
    backend.optimize(circ)
    qasm = backend.export_qasm(circ)
    result = backend.simulate(circ, shots=1024)
    metrics = backend.metrics(circ)

Example:
    >>> backend = QuantaBenchpressBackend()
    >>> report = backend.run_workout("small_circuits")
    >>> print(report)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from quanta.compiler.pipeline import CompilerPipeline
from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.export.qasm_import import from_qasm
from quanta.simulator.statevector import StateVectorSimulator

__all__ = ["QuantaBenchpressBackend"]


@dataclass
class CircuitHandle:
    """Internal circuit representation for Benchpress API."""
    builder: CircuitBuilder
    dag: DAGCircuit | None = None
    compiled_dag: DAGCircuit | None = None
    num_qubits: int = 0


@dataclass
class BenchpressMetrics:
    """Standard Benchpress metrics for a circuit."""
    num_qubits: int
    gate_count: int
    depth: int
    two_qubit_gates: int
    transpile_time_ms: float
    simulate_time_ms: float

    def __repr__(self) -> str:
        return (
            f"Metrics(qubits={self.num_qubits}, gates={self.gate_count}, "
            f"depth={self.depth}, 2q={self.two_qubit_gates}, "
            f"transpile={self.transpile_time_ms:.1f}ms)"
        )


class QuantaBenchpressBackend:
    """Benchpress-compatible backend for Quanta SDK.

    Follows the Benchpress API pattern for cross-SDK benchmarking.
    """

    SDK_NAME = "Quanta"
    SDK_VERSION = "0.4.0"

    def new_circuit(self, num_qubits: int) -> CircuitHandle:
        """Creates a new empty circuit."""
        builder = CircuitBuilder(num_qubits)
        return CircuitHandle(builder=builder, num_qubits=num_qubits)

    def from_qasm(self, qasm_str: str) -> CircuitHandle:
        """Creates a circuit from QASM string."""
        dag = from_qasm(qasm_str)
        handle = CircuitHandle(
            builder=CircuitBuilder(dag.num_qubits),
            dag=dag,
            num_qubits=dag.num_qubits,
        )
        return handle

    def apply_gate(
        self,
        circuit: CircuitHandle,
        gate: str,
        qubits: list[int],
        params: list[float] | None = None,
    ) -> None:
        """Applies a gate to the circuit."""
        gate_map = {
            "h": "H", "x": "X", "y": "Y", "z": "Z",
            "s": "S", "t": "T",
            "cx": "CX", "cz": "CZ", "swap": "SWAP", "ccx": "CCX",
            "rx": "RX", "ry": "RY", "rz": "RZ",
        }
        quanta_gate = gate_map.get(gate.lower(), gate.upper())
        circuit.builder.record(Instruction(
            gate_name=quanta_gate,
            qubits=tuple(qubits),
            params=tuple(params or []),
        ))
        circuit.dag = None  # Invalidate cached DAG
        circuit.compiled_dag = None

    def build(self, circuit: CircuitHandle) -> None:
        """Builds the DAG from recorded instructions."""
        if circuit.dag is None:
            circuit.dag = DAGCircuit.from_builder(circuit.builder)

    def optimize(self, circuit: CircuitHandle) -> float:
        """Runs the 3-pass compiler. Returns transpile time in ms."""
        self.build(circuit)
        t0 = time.perf_counter()
        pipeline = CompilerPipeline()
        circuit.compiled_dag = pipeline.run(circuit.dag)
        return (time.perf_counter() - t0) * 1000

    def simulate(
        self,
        circuit: CircuitHandle,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Simulates the circuit and returns measurement counts."""
        dag = circuit.compiled_dag or circuit.dag
        if dag is None:
            self.build(circuit)
            dag = circuit.dag

        sim = StateVectorSimulator(circuit.num_qubits, seed=seed)
        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)
        return sim.sample(shots)

    def export_qasm(self, circuit: CircuitHandle) -> str:
        """Exports the circuit to QASM 3.0 string."""
        dag = circuit.compiled_dag or circuit.dag
        if dag is None:
            self.build(circuit)
            dag = circuit.dag

        lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]
        lines.append(f"qubit[{circuit.num_qubits}] q;")
        lines.append("")

        gate_map = {
            "H": "h", "X": "x", "Y": "y", "Z": "z",
            "S": "s", "T": "t",
            "CX": "cx", "CZ": "cz", "SWAP": "swap", "CCX": "ccx",
            "RX": "rx", "RY": "ry", "RZ": "rz",
        }

        for op in dag.op_nodes():
            name = gate_map.get(op.gate_name, op.gate_name.lower())
            qubits = ", ".join(f"q[{q}]" for q in op.qubits)
            if op.params:
                params = ", ".join(f"{p:.6f}" for p in op.params)
                lines.append(f"{name}({params}) {qubits};")
            else:
                lines.append(f"{name} {qubits};")

        return "\n".join(lines)

    def metrics(self, circuit: CircuitHandle) -> BenchpressMetrics:
        """Returns Benchpress-standard metrics."""
        dag = circuit.compiled_dag or circuit.dag
        if dag is None:
            self.build(circuit)
            dag = circuit.dag

        two_q = sum(1 for op in dag.op_nodes() if len(op.qubits) >= 2)

        return BenchpressMetrics(
            num_qubits=circuit.num_qubits,
            gate_count=dag.gate_count(),
            depth=dag.depth(),
            two_qubit_gates=two_q,
            transpile_time_ms=0,
            simulate_time_ms=0,
        )

    def info(self) -> dict[str, str]:
        """Returns SDK information."""
        return {
            "sdk": self.SDK_NAME,
            "version": self.SDK_VERSION,
            "simulator": "statevector (tensor contraction)",
            "max_qubits": "27",
            "compiler": "3-pass (cancel, merge, translate)",
            "qasm_support": "2.0 + 3.0",
        }
