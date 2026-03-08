"""
tests/test_qasm_import.py — QASM import + round-trip tests.

Validates:
  - QASM 2.0 and 3.0 parsing
  - Register handling (qreg, qubit)
  - Gate mapping
  - Parametric gates
  - Round-trip: QASM → DAG → verify
"""


from quanta.export.qasm_import import from_qasm


class TestQASMImport:
    """QASM import pipeline tests."""

    def test_basic_qasm2(self):
        """Parse simple QASM 2.0 circuit."""
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""
        dag = from_qasm(qasm)
        assert dag.num_qubits == 2
        assert dag.gate_count() == 2

    def test_basic_qasm3(self):
        """Parse simple QASM 3.0 circuit."""
        qasm = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
cx q[0],q[1];
cx q[1],q[2];
"""
        dag = from_qasm(qasm)
        assert dag.num_qubits == 3
        assert dag.gate_count() == 3

    def test_parametric_gates(self):
        """Parse parametric gates (rx, ry, rz)."""
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(1.5708) q[0];
ry(3.1416) q[0];
rz(0.7854) q[0];
"""
        dag = from_qasm(qasm)
        assert dag.gate_count() == 3

    def test_pi_parameter(self):
        """Parse pi expressions in parameters."""
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rz(pi) q[0];
"""
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        assert len(ops) == 1
        assert abs(ops[0].params[0] - 3.14159) < 0.01

    def test_multi_qubit_gates(self):
        """Parse multi-qubit gates: cx, ccx, swap."""
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[0],q[1];
ccx q[0],q[1],q[2];
swap q[0],q[2];
"""
        dag = from_qasm(qasm)
        assert dag.gate_count() == 3
        ops = list(dag.op_nodes())
        assert ops[0].gate_name == "CX"
        assert ops[1].gate_name == "CCX"
        assert ops[2].gate_name == "SWAP"

    def test_bell_state_simulation(self):
        """QASM → DAG → simulate → verify Bell state probs."""
        from quanta.simulator.statevector import StateVectorSimulator

        dag = from_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
""")
        sim = StateVectorSimulator(2, seed=42)
        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        counts = sim.sample(1000)
        assert counts.get("00", 0) > 400
        assert counts.get("11", 0) > 400
        assert counts.get("01", 0) == 0
        assert counts.get("10", 0) == 0

    def test_empty_qasm_returns_dag(self):
        dag = from_qasm("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];")
        assert dag.num_qubits == 1
        assert dag.gate_count() == 0

    def test_comments_ignored(self):
        """Comments should be ignored."""
        qasm = """
OPENQASM 2.0;
// This is a comment
include "qelib1.inc";
qreg q[2];
h q[0]; // Apply H
cx q[0],q[1]; // Entangle
"""
        dag = from_qasm(qasm)
        assert dag.gate_count() == 2

    def test_barrier_ignored(self):
        dag = from_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
barrier q[0],q[1];
cx q[0],q[1];
""")
        assert dag.gate_count() == 2


class TestQASMBenchmarkSuite:
    """Tests for QASMBench benchmark runner."""

    def test_qasmbench_runs(self):
        from quanta.benchmark.qasmbench import run_qasmbench
        result = run_qasmbench()
        assert len(result.circuits) > 0

    def test_all_circuits_pass(self):
        from quanta.benchmark.qasmbench import run_qasmbench
        result = run_qasmbench()
        for c in result.circuits:
            assert c.round_trip_ok, f"{c.name} failed"

    def test_report_is_string(self):
        from quanta.benchmark.qasmbench import run_qasmbench
        result = run_qasmbench()
        report = result.report()
        assert isinstance(report, str)
        assert "QASMBench" in report

    def test_benchpress_adapter(self):
        from quanta.benchmark.benchpress_adapter import QuantaBenchpressBackend
        backend = QuantaBenchpressBackend()
        circ = backend.new_circuit(2)
        backend.apply_gate(circ, "h", [0])
        backend.apply_gate(circ, "cx", [0, 1])
        backend.optimize(circ)
        counts = backend.simulate(circ, shots=100, seed=42)
        assert len(counts) > 0
        metrics = backend.metrics(circ)
        assert metrics.gate_count == 2

    def test_benchpress_qasm_export(self):
        from quanta.benchmark.benchpress_adapter import QuantaBenchpressBackend
        backend = QuantaBenchpressBackend()
        circ = backend.new_circuit(2)
        backend.apply_gate(circ, "h", [0])
        qasm = backend.export_qasm(circ)
        assert "OPENQASM 3.0" in qasm
        assert "h q[0]" in qasm
