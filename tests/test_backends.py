"""
tests/test_backends.py -- Backend integration tests.

Tests IBM and IonQ backends without requiring real API keys or hardware.
Uses mock/patching for external calls, direct testing for internal logic.
"""

import json
import pytest
import numpy as np

from quanta import circuit, H, CX, measure, run
from quanta.backends.base import Backend
from quanta.backends.local import LocalSimulator
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result


# ═══════════════════════════════════════════
#  run() with backend parameter
# ═══════════════════════════════════════════

class TestRunWithBackend:
    """Tests run() function's backend delegation."""

    def test_run_with_local_backend(self):
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        backend = LocalSimulator(seed=42)
        result = run(bell, shots=1024, backend=backend)
        assert result.shots == 1024
        assert result.most_frequent in ("00", "11")

    def test_run_without_backend_uses_default(self):
        @circuit(qubits=1)
        def simple(q):
            H(q[0])
            return measure(q)

        result = run(simple, shots=1000, seed=42)
        assert result.shots == 1000
        assert result.statevector is not None  # default path sets statevector

    def test_run_with_backend_sets_metadata(self):
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        backend = LocalSimulator(seed=42)
        result = run(bell, shots=100, backend=backend)
        assert result.circuit_name == "bell"
        assert result.gate_count == 2
        assert result.depth >= 1


# ═══════════════════════════════════════════
#  IBM Backend internal logic
# ═══════════════════════════════════════════

class TestIBMBackendInternal:
    """Tests IBM backend QASM generation without Qiskit dependency."""

    def test_dag_to_qasm_basic(self):
        from quanta.backends.ibm import IBMBackend

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        dag = DAGCircuit.from_builder(bell.build())
        backend = IBMBackend(simulate_locally=True)
        qasm = backend._dag_to_qasm(dag)

        assert "OPENQASM 2.0;" in qasm
        assert "qelib1.inc" in qasm
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm
        assert "measure q[0]" in qasm
        assert "measure q[1]" in qasm

    def test_dag_to_qasm_parametric(self):
        from quanta.backends.ibm import IBMBackend
        from quanta import RZ

        @circuit(qubits=1)
        def rot(q):
            RZ(1.5707963)(q[0])
            return measure(q)

        dag = DAGCircuit.from_builder(rot.build())
        backend = IBMBackend(simulate_locally=True)
        qasm = backend._dag_to_qasm(dag)

        assert "rz(" in qasm
        assert "q[0];" in qasm

    def test_dag_to_qasm_unsupported_gate_raises(self):
        from quanta.backends.ibm import IBMBackend, IBMBackendError
        from quanta.core.circuit import CircuitBuilder
        from quanta.core.types import Instruction

        builder = CircuitBuilder(1)
        builder.record(Instruction("UNSUPPORTED_XYZ", (0,), ()))
        dag = DAGCircuit.from_builder(builder)

        backend = IBMBackend(simulate_locally=True)
        with pytest.raises(IBMBackendError, match="not supported"):
            backend._dag_to_qasm(dag)

    def test_name_property_local(self):
        from quanta.backends.ibm import IBMBackend
        backend = IBMBackend(simulate_locally=True)
        assert backend.name == "ibm_local"

    def test_name_property_hardware(self):
        from quanta.backends.ibm import IBMBackend
        backend = IBMBackend(backend_name="ibm_osaka")
        assert backend.name == "ibm_ibm_osaka"

    def test_repr(self):
        from quanta.backends.ibm import IBMBackend
        backend = IBMBackend(backend_name="ibm_brisbane", simulate_locally=True)
        assert "ibm_brisbane" in repr(backend)
        assert "local=True" in repr(backend)

    def test_hardware_without_token_raises(self):
        from quanta.backends.ibm import IBMBackend, IBMBackendError
        import os

        # Ensure no env var
        old_token = os.environ.pop("IBM_QUANTUM_TOKEN", None)
        try:
            backend = IBMBackend(backend_name="ibm_brisbane", token="")

            @circuit(qubits=1)
            def simple(q):
                H(q[0])
                return measure(q)

            dag = DAGCircuit.from_builder(simple.build())
            with pytest.raises(IBMBackendError, match="token"):
                backend.execute(dag, shots=100)
        finally:
            if old_token:
                os.environ["IBM_QUANTUM_TOKEN"] = old_token


# ═══════════════════════════════════════════
#  IonQ Backend internal logic
# ═══════════════════════════════════════════

class TestIonQBackendInternal:
    """Tests IonQ backend gate conversion and result parsing."""

    def test_dag_to_ionq_circuit_bell(self):
        from quanta.backends.ionq import IonQBackend

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        dag = DAGCircuit.from_builder(bell.build())
        backend = IonQBackend()
        gates = backend._dag_to_ionq_circuit(dag)

        assert len(gates) == 2
        assert gates[0] == {"gate": "h", "target": 0}
        assert gates[1] == {"gate": "cnot", "control": 0, "target": 1}

    def test_dag_to_ionq_circuit_parametric(self):
        from quanta.backends.ionq import IonQBackend
        from quanta import RZ

        @circuit(qubits=1)
        def rot(q):
            RZ(1.5)(q[0])
            return measure(q)

        dag = DAGCircuit.from_builder(rot.build())
        backend = IonQBackend()
        gates = backend._dag_to_ionq_circuit(dag)

        assert len(gates) == 1
        assert gates[0]["gate"] == "rz"
        assert gates[0]["target"] == 0
        assert abs(gates[0]["rotation"] - 1.5) < 1e-6

    def test_dag_to_ionq_circuit_3qubit(self):
        from quanta.backends.ionq import IonQBackend
        from quanta import CCX

        @circuit(qubits=3)
        def toffoli(q):
            CCX(q[0], q[1], q[2])
            return measure(q)

        dag = DAGCircuit.from_builder(toffoli.build())
        backend = IonQBackend()
        gates = backend._dag_to_ionq_circuit(dag)

        assert gates[0]["gate"] == "ccx"
        assert gates[0]["controls"] == [0, 1]
        assert gates[0]["target"] == 2

    def test_build_job_body(self):
        from quanta.backends.ionq import IonQBackend

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        dag = DAGCircuit.from_builder(bell.build())
        backend = IonQBackend(target="simulator")
        body = backend._build_job_body(dag, shots=100)

        assert body["target"] == "simulator"
        assert body["shots"] == 100
        assert body["input"]["qubits"] == 2
        assert body["input"]["format"] == "ionq.circuit.v0"
        assert len(body["input"]["circuit"]) == 2

        # Validate JSON serializable
        json_str = json.dumps(body)
        assert len(json_str) > 0

    def test_parse_results_bell(self):
        from quanta.backends.ionq import IonQBackend

        job = {
            "data": {
                "probabilities": {"0": 0.5, "3": 0.5}
            }
        }
        counts = IonQBackend._parse_results(job, num_qubits=2, shots=1000)

        assert "00" in counts
        assert "11" in counts
        assert counts["00"] + counts["11"] == 1000

    def test_parse_results_single_state(self):
        from quanta.backends.ionq import IonQBackend

        job = {"data": {"probabilities": {"0": 1.0}}}
        counts = IonQBackend._parse_results(job, num_qubits=1, shots=500)

        assert counts == {"0": 500}

    def test_parse_results_three_states(self):
        from quanta.backends.ionq import IonQBackend

        job = {
            "data": {
                "probabilities": {"0": 0.25, "1": 0.25, "3": 0.5}
            }
        }
        counts = IonQBackend._parse_results(job, num_qubits=2, shots=1000)

        assert sum(counts.values()) == 1000
        assert counts["11"] >= 400  # ~500
        assert "00" in counts
        assert "01" in counts

    def test_name_property(self):
        from quanta.backends.ionq import IonQBackend
        assert IonQBackend(target="simulator").name == "ionq_simulator"
        assert IonQBackend(target="qpu.aria-1").name == "ionq_qpu.aria-1"

    def test_repr(self):
        from quanta.backends.ionq import IonQBackend
        assert "simulator" in repr(IonQBackend(target="simulator"))

    def test_api_request_without_key_raises(self):
        from quanta.backends.ionq import IonQBackend, IonQBackendError
        import os

        old_key = os.environ.pop("IONQ_API_KEY", None)
        try:
            backend = IonQBackend(api_key="")
            with pytest.raises(IonQBackendError, match="API key"):
                backend._api_request("GET", "/jobs")
        finally:
            if old_key:
                os.environ["IONQ_API_KEY"] = old_key

    def test_unsupported_gate_raises(self):
        from quanta.backends.ionq import IonQBackend, IonQBackendError
        from quanta.core.circuit import CircuitBuilder
        from quanta.core.types import Instruction

        builder = CircuitBuilder(1)
        builder.record(Instruction("UNSUPPORTED_XYZ", (0,), ()))
        dag = DAGCircuit.from_builder(builder)

        backend = IonQBackend()
        with pytest.raises(IonQBackendError, match="not supported"):
            backend._dag_to_ionq_circuit(dag)
