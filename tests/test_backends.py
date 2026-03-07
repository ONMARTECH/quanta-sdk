"""
tests/test_backends.py -- Mock tests for hardware backends.

Tests Google/IBM backend adapters using mocks
(no real API tokens or hardware needed).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════
#  Google Cirq Backend (mocked)
# ═══════════════════════════════════════════

class TestGoogleBackend:
    """Tests for Google Cirq backend adapter with mocked cirq."""

    def test_backend_name_local(self):
        with patch.dict("sys.modules", {"cirq": MagicMock()}):
            from quanta.backends.google import GoogleBackend
            backend = GoogleBackend(simulate_locally=True)
            assert backend.name == "google_local"

    def test_backend_name_processor(self):
        with patch.dict("sys.modules", {"cirq": MagicMock()}):
            from quanta.backends.google import GoogleBackend
            backend = GoogleBackend(
                project_id="my-project",
                processor_id="rainbow"
            )
            assert backend.name == "google_rainbow"

    def test_cirq_result_to_counts(self):
        with patch.dict("sys.modules", {"cirq": MagicMock()}):
            from quanta.backends.google import GoogleBackend
            mock_result = MagicMock()
            mock_result.histogram.return_value = {0: 500, 3: 500}
            counts = GoogleBackend._cirq_result_to_counts(mock_result)
            assert "00" in counts
            assert "11" in counts
            assert counts["00"] == 500
            assert counts["11"] == 500

    def test_repr(self):
        with patch.dict("sys.modules", {"cirq": MagicMock()}):
            from quanta.backends.google import GoogleBackend
            backend = GoogleBackend(simulate_locally=True)
            r = repr(backend)
            assert "GoogleBackend" in r
            assert "local" in r

    def test_dag_to_qasm(self):
        with patch.dict("sys.modules", {"cirq": MagicMock()}):
            from quanta.backends.google import GoogleBackend
            from quanta.core.circuit import circuit, CircuitBuilder
            from quanta.core.gates import H, CX
            from quanta.core.measure import measure
            from quanta.dag.dag_circuit import DAGCircuit

            @circuit(qubits=2)
            def bell(q):
                H(q[0])
                CX(q[0], q[1])
                return measure(q)

            builder = bell.build()
            dag = DAGCircuit.from_builder(builder)
            backend = GoogleBackend(simulate_locally=True)
            qasm = backend._dag_to_qasm(dag)
            assert "OPENQASM" in qasm
            assert "h q[0]" in qasm
            assert "cx q[0]" in qasm


# ═══════════════════════════════════════════
#  IBM Qiskit Backend (mocked)
# ═══════════════════════════════════════════

class TestIBMBackend:
    """Tests for IBM Qiskit backend adapter with mocked qiskit."""

    def test_backend_name_local(self):
        backend = self._make_backend(simulate_locally=True)
        assert backend.name == "ibm_local"

    def test_backend_name_hardware(self):
        backend = self._make_backend()
        assert backend.name == "ibm_ibm_brisbane"

    def test_repr(self):
        backend = self._make_backend()
        r = repr(backend)
        assert "IBMBackend" in r
        assert "ibm_brisbane" in r

    def test_dag_to_qasm(self):
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, CX
        from quanta.core.measure import measure
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        builder = bell.build()
        dag = DAGCircuit.from_builder(builder)
        backend = self._make_backend()
        qasm = backend._dag_to_qasm(dag)
        assert "OPENQASM" in qasm
        assert "h q[0]" in qasm

    def test_hardware_without_token_raises(self):
        from quanta.backends.ibm import IBMBackend, IBMBackendError
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        backend = IBMBackend(token="", simulate_locally=False)
        dag = DAGCircuit.from_builder(c.build())
        with pytest.raises(IBMBackendError, match="token"):
            backend.execute(dag, shots=10)

    @staticmethod
    def _make_backend(simulate_locally=False):
        from quanta.backends.ibm import IBMBackend
        return IBMBackend(
            backend_name="ibm_brisbane",
            token="test-token",
            simulate_locally=simulate_locally,
        )


# ═══════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════

class TestRunner:
    """Tests for runner module backend delegation."""

    def test_run_default_simulator(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, CX
        from quanta.core.measure import measure

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=100)
        assert result.shots == 100
        assert sum(result.counts.values()) == 100

    def test_run_preserves_circuit_name(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import X
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def my_circ(q):
            X(q[0])
            return measure(q)

        result = run(my_circ, shots=10)
        assert result.circuit_name == "my_circ"

    def test_run_gate_count_depth(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, CX
        from quanta.core.measure import measure

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=10)
        assert result.gate_count == 2
        assert result.depth >= 2

    def test_run_invalid_circuit_raises(self):
        from quanta.runner import run
        from quanta.core.types import QuantaError
        with pytest.raises(QuantaError, match="@circuit"):
            run("not_a_circuit", shots=10)

    def test_run_negative_shots_raises(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure
        from quanta.core.types import QuantaError

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        with pytest.raises(QuantaError, match="positive"):
            run(c, shots=0)

    def test_sweep_basic(self):
        from quanta.runner import sweep
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        # sweep with empty params returns single result
        results = sweep(c, params={}, shots=10)
        assert len(results) == 1
        assert results[0].shots == 10

    def test_sweep_empty_params(self):
        from quanta.runner import sweep
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        results = sweep(c, params={}, shots=10)
        assert len(results) == 1
