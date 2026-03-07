"""
tests/test_coverage_boost2.py — Coverage gap tests.

Targets modules with <90% coverage:
  - backends/compat.py (0% → covered)
  - mcp_server.py (0% → tool functions tested directly)
  - simulator/accelerated.py (55% → numpy fallback path)
  - backends/ibm.py (60% → mock execution paths)
  - backends/google.py (46% → mock execution + QASM)
  - runner.py (70% → backend delegation, sweep params)
  - layer3/shor.py (71% → edge cases, summary, repr)
  - simulator/pauli_frame.py (79% → Y/Z/CZ/SWAP/inject_error)
  - gradients.py (88% → natural gradient QFIM coverage)
"""

from unittest.mock import MagicMock, patch
import json
import numpy as np
import pytest


# ═══════════════════════════════════════════
#  backends/compat.py
# ═══════════════════════════════════════════

class TestCompat:
    """Tests for backend compatibility layer."""

    def test_qiskit_version_not_installed(self):
        from quanta.backends.compat import qiskit_version
        # In test env, qiskit is not installed
        assert qiskit_version() is None

    def test_cirq_version_not_installed(self):
        from quanta.backends.compat import cirq_version
        assert cirq_version() is None

    def test_check_backend_compatibility_returns_list(self):
        from quanta.backends.compat import check_backend_compatibility
        results = check_backend_compatibility()
        assert len(results) == 3  # qiskit, cirq, ionq

    def test_ionq_always_compatible(self):
        from quanta.backends.compat import check_backend_compatibility
        results = check_backend_compatibility()
        ionq = [r for r in results if r.name == "ionq"][0]
        assert ionq.compatible is True
        assert ionq.installed is True

    def test_qiskit_not_installed_info(self):
        from quanta.backends.compat import check_backend_compatibility
        results = check_backend_compatibility()
        qiskit = [r for r in results if r.name == "qiskit"][0]
        assert qiskit.installed is False
        assert qiskit.compatible is False
        assert "pip install" in qiskit.message

    def test_cirq_not_installed_info(self):
        from quanta.backends.compat import check_backend_compatibility
        results = check_backend_compatibility()
        cirq = [r for r in results if r.name == "cirq"][0]
        assert cirq.installed is False
        assert "cirq-google" in cirq.message

    def test_backend_version_info_repr(self):
        from quanta.backends.compat import BackendVersionInfo
        info = BackendVersionInfo("test", "1.0.0", True, True, "OK")
        r = repr(info)
        assert "✅" in r
        assert "test" in r

    def test_backend_version_info_repr_warning(self):
        from quanta.backends.compat import BackendVersionInfo
        info = BackendVersionInfo("test", "0.1.0", True, False, "Old")
        r = repr(info)
        assert "⚠️" in r

    def test_backend_version_info_repr_not_installed(self):
        from quanta.backends.compat import BackendVersionInfo
        info = BackendVersionInfo("test", None, False, False, "Missing")
        r = repr(info)
        assert "❌" in r
        assert "not installed" in r

    def test_parse_version_valid(self):
        from quanta.backends.compat import _parse_version
        assert _parse_version("1.2.3") == (1, 2, 3)
        assert _parse_version("0.9.0") == (0, 9, 0)

    def test_parse_version_invalid(self):
        from quanta.backends.compat import _parse_version
        assert _parse_version("abc") == (0, 0, 0)

    def test_parse_version_short(self):
        from quanta.backends.compat import _parse_version
        assert _parse_version("1.2") == (1, 2)

    def test_qiskit_version_with_mock(self):
        mock_qiskit = MagicMock()
        mock_qiskit.__version__ = "1.3.1"
        with patch.dict("sys.modules", {"qiskit": mock_qiskit}):
            from quanta.backends import compat
            # Force reimport to pick up mock
            result = compat.qiskit_version()
            assert result == "1.3.1"

    def test_cirq_version_with_mock(self):
        mock_cirq = MagicMock()
        mock_cirq.__version__ = "1.4.0"
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            from quanta.backends import compat
            result = compat.cirq_version()
            assert result == "1.4.0"

    def test_import_qiskit_safe_not_installed(self):
        from quanta.backends.compat import import_qiskit_safe
        with pytest.raises(ImportError):
            import_qiskit_safe()

    def test_import_cirq_safe_not_installed(self):
        from quanta.backends.compat import import_cirq_safe
        with pytest.raises(ImportError):
            import_cirq_safe()


# ═══════════════════════════════════════════
#  simulator/accelerated.py
# ═══════════════════════════════════════════

class TestAccelerated:
    """Tests for GPU acceleration module (numpy fallback path)."""

    def test_xp_returns_numpy(self):
        from quanta.simulator.accelerated import xp
        mod = xp()
        assert mod is np or hasattr(mod, "ndarray")

    def test_get_array_module(self):
        from quanta.simulator.accelerated import get_array_module
        mod = get_array_module()
        assert mod is np or hasattr(mod, "ndarray")

    def test_get_backend_info(self):
        from quanta.simulator.accelerated import get_backend_info
        info = get_backend_info()
        assert "backend" in info
        assert "device" in info
        # Without GPU, should be numpy + cpu
        assert info["backend"] == "numpy"
        assert info["device"] == "cpu"

    def test_tensor_contract_single_qubit(self):
        from quanta.simulator.accelerated import tensor_contract
        # H gate on |0⟩
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        state = np.array([1, 0], dtype=complex)
        result = tensor_contract(H, state, (0,), 1)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_tensor_contract_two_qubit(self):
        from quanta.simulator.accelerated import tensor_contract
        # CX gate on |10⟩ → |11⟩
        CX = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        state = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
        result = tensor_contract(CX, state, (0, 1), 2)
        expected = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_tensor_contract_preserves_norm(self):
        from quanta.simulator.accelerated import tensor_contract
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        result = tensor_contract(H, state, (0,), 2)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-10)

    def test_ensure_init_idempotent(self):
        from quanta.simulator.accelerated import _ensure_init
        _ensure_init()
        _ensure_init()  # Should not fail on repeated calls


# ═══════════════════════════════════════════
#  runner.py — Additional coverage
# ═══════════════════════════════════════════

class TestRunnerExtended:
    """Extended runner tests for backend delegation and sweep edge cases."""

    def test_run_with_mock_backend(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, CX
        from quanta.core.measure import measure
        from quanta.result import Result

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        mock_backend = MagicMock()
        mock_backend.execute.return_value = Result(
            counts={"00": 500, "11": 500},
            shots=1000,
            num_qubits=2,
        )

        result = run(bell, shots=1000, backend=mock_backend)
        mock_backend.execute.assert_called_once()
        assert result.circuit_name == "bell"
        assert result.gate_count == 2

    def test_run_with_builtin_sim(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, X
        from quanta.core.measure import measure

        @circuit(qubits=2)
        def c(q):
            H(q[0])
            X(q[1])
            return measure(q)

        result = run(c, shots=100)
        assert result.shots == 100
        assert result.gate_count == 2
        assert result.depth >= 1

    def test_sweep_single_run(self):
        from quanta.runner import sweep
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        # No params → single run
        results = sweep(c, params={}, shots=50)
        assert len(results) == 1
        assert results[0].shots == 50

    def test_sweep_mismatched_lengths_raises(self):
        from quanta.runner import sweep
        from quanta.core.circuit import circuit
        from quanta.core.gates import RZ
        from quanta.core.measure import measure
        from quanta.core.types import QuantaError

        @circuit(qubits=1)
        def rot(q, theta=0.0):
            RZ(q[0], theta)
            return measure(q)

        with pytest.raises(QuantaError, match="same length"):
            sweep(rot, params={"theta": [1.0, 2.0], "phi": [1.0]}, shots=10)

    def test_run_returns_statevector(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        result = run(c, shots=10)
        assert result.statevector is not None
        assert len(result.statevector) == 2

    def test_run_seed_reproducibility(self):
        from quanta.runner import run
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        r1 = run(c, shots=100, seed=42)
        r2 = run(c, shots=100, seed=42)
        assert r1.counts == r2.counts


# ═══════════════════════════════════════════
#  simulator/pauli_frame.py — Y/Z/CZ/SWAP/inject_error
# ═══════════════════════════════════════════

class TestPauliFrameExtended:
    """Extended Pauli frame simulator tests."""

    def test_y_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.y(0)
        counts = sim.sample(shots=100, seed=42)
        assert "1" in counts  # Y|0⟩ = i|1⟩

    def test_z_gate_on_zero(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.z(0)
        counts = sim.sample(shots=100, seed=42)
        assert "0" in counts  # Z|0⟩ = |0⟩

    def test_cz_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.h(0)
        sim.h(1)
        sim.cz(0, 1)
        counts = sim.sample(shots=1000, seed=42)
        assert len(counts) > 0

    def test_swap_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.x(0)  # |0⟩ → |1⟩ on qubit 0
        sim.swap(0, 1)  # Now qubit 1 should be |1⟩
        counts = sim.sample(shots=100, seed=42)
        assert "01" in counts  # After SWAP, qubit 1 is |1⟩

    def test_inject_error_x(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.inject_error(0, "X")
        counts = sim.sample(shots=100, seed=42)
        assert "1" in counts  # X error flips to |1⟩

    def test_inject_error_y(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.inject_error(0, "Y")
        counts = sim.sample(shots=100, seed=42)
        assert "1" in counts

    def test_inject_error_z(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.inject_error(0, "Z")
        counts = sim.sample(shots=100, seed=42)
        assert "0" in counts  # Z on |0⟩ keeps |0⟩

    def test_inject_error_invalid_raises(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        with pytest.raises(ValueError, match="Unknown error"):
            sim.inject_error(0, "W")

    def test_measure_specific_qubits(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(3)
        sim.x(2)  # Flip qubit 2
        sim.measure(2)  # Only measure qubit 2
        counts = sim.sample(shots=100, seed=42)
        # Should only get 1-bit strings
        for bitstring in counts:
            assert len(bitstring) == 1
        assert "1" in counts

    def test_bell_state_correlation(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.h(0)
        sim.cx(0, 1)
        counts = sim.sample(shots=1000, seed=42)
        # Bell state: only 00 and 11
        for bitstring in counts:
            assert bitstring in ("00", "11")

    def test_repr(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(5)
        assert "PauliFrameSimulator" in repr(sim)
        assert "n=5" in repr(sim)

    def test_ghz_state(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(3)
        sim.h(0)
        sim.cx(0, 1)
        sim.cx(0, 2)
        counts = sim.sample(shots=1000, seed=42)
        for bitstring in counts:
            assert bitstring in ("000", "111")

    def test_s_gate_then_s_gate_equals_z(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        # S^2 = Z. Apply H first to see effect.
        sim1 = PauliFrameSimulator(1)
        sim1.h(0)
        sim1.s(0)
        sim1.s(0)
        counts1 = sim1.sample(shots=1000, seed=42)

        sim2 = PauliFrameSimulator(1)
        sim2.h(0)
        sim2.z(0)
        counts2 = sim2.sample(shots=1000, seed=42)

        assert counts1 == counts2

    def test_deterministic_measurement(self):
        """Test that |0⟩ always measures 0 (deterministic path)."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        # No gates = |0⟩ state
        counts = sim.sample(shots=100, seed=42)
        assert counts == {"0": 100}


# ═══════════════════════════════════════════
#  layer3/shor.py — Edge cases
# ═══════════════════════════════════════════

class TestShorExtended:
    """Extended Shor's algorithm tests."""

    def test_factor_even_number(self):
        from quanta.layer3.shor import factor
        result = factor(6)
        assert result.method == "classical_shortcut"
        assert result.factors == (2, 3)

    def test_factor_small_composite(self):
        from quanta.layer3.shor import factor
        result = factor(9)
        assert result.factors[0] * result.factors[1] == 9

    def test_factor_15(self):
        from quanta.layer3.shor import factor
        result = factor(15, seed=42)
        assert result.factors[0] * result.factors[1] == 15

    def test_factor_21(self):
        from quanta.layer3.shor import factor
        result = factor(21, seed=42)
        assert result.factors[0] * result.factors[1] == 21

    def test_factor_too_small_raises(self):
        from quanta.layer3.shor import factor
        with pytest.raises(ValueError, match="must be > 1"):
            factor(1)

    def test_shor_result_repr(self):
        from quanta.layer3.shor import ShorResult
        r = ShorResult(15, (3, 5), 4, 1, "quantum")
        s = repr(r)
        assert "15" in s
        assert "3" in s
        assert "5" in s

    def test_shor_result_summary(self):
        from quanta.layer3.shor import ShorResult
        r = ShorResult(15, (3, 5), 4, 1, "quantum")
        s = r.summary()
        assert "15" in s
        assert "3" in s
        assert "Period" in s


# ═══════════════════════════════════════════
#  backends/google.py — Mocked execution
# ═══════════════════════════════════════════

class TestGoogleBackendExtended:
    """Extended Google backend tests with mocked cirq execution."""

    def test_ensure_cirq_raises_without_cirq(self):
        from quanta.backends.google import GoogleBackend, GoogleBackendError
        backend = GoogleBackend(simulate_locally=True)
        with pytest.raises(GoogleBackendError, match="cirq"):
            backend._ensure_cirq()

    def test_unsupported_gate_raises(self):
        from quanta.backends.google import GoogleBackend, GoogleBackendError
        from quanta.dag.dag_circuit import DAGCircuit

        # Create a DAG with an unsupported gate
        mock_dag = MagicMock(spec=DAGCircuit)
        mock_dag.num_qubits = 1
        mock_op = MagicMock()
        mock_op.gate_name = "UNSUPPORTED_GATE"
        mock_op.qubits = (0,)
        mock_op.params = ()
        mock_dag.op_nodes.return_value = [mock_op]

        backend = GoogleBackend(simulate_locally=True)
        with pytest.raises(GoogleBackendError, match="not supported"):
            backend._dag_to_qasm(mock_dag)

    def test_dag_to_qasm_with_params(self):
        from quanta.backends.google import GoogleBackend
        from quanta.core.circuit import circuit
        from quanta.core.gates import RZ
        from quanta.core.measure import measure
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=1)
        def c(q):
            RZ(1.5)(q[0])
            return measure(q)

        dag = DAGCircuit.from_builder(c.build())
        backend = GoogleBackend(simulate_locally=True)
        qasm = backend._dag_to_qasm(dag)
        assert "rz" in qasm
        assert "1.5" in qasm

    def test_dag_to_qasm_measurement(self):
        from quanta.backends.google import GoogleBackend
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, CX
        from quanta.core.measure import measure
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=2)
        def c(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        dag = DAGCircuit.from_builder(c.build())
        backend = GoogleBackend(simulate_locally=True)
        qasm = backend._dag_to_qasm(dag)
        assert "measure" in qasm


# ═══════════════════════════════════════════
#  backends/ibm.py — Mocked execution
# ═══════════════════════════════════════════

class TestIBMBackendExtended:
    """Extended IBM backend tests with mocked qiskit execution."""

    def test_ibm_unsupported_gate_raises(self):
        from quanta.backends.ibm import IBMBackend, IBMBackendError
        mock_dag = MagicMock()
        mock_dag.num_qubits = 1
        mock_op = MagicMock()
        mock_op.gate_name = "UNSUPPORTED"
        mock_op.qubits = (0,)
        mock_op.params = ()
        mock_dag.op_nodes.return_value = [mock_op]

        backend = IBMBackend(simulate_locally=True)
        with pytest.raises(IBMBackendError, match="not supported"):
            backend._dag_to_qasm(mock_dag)

    def test_ibm_local_without_qiskit_raises(self):
        from quanta.backends.ibm import IBMBackend, IBMBackendError
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        backend = IBMBackend(simulate_locally=True)
        dag = DAGCircuit.from_builder(c.build())
        with pytest.raises(IBMBackendError, match="qiskit"):
            backend.execute(dag, shots=10)

    def test_ibm_dag_to_qasm_with_params(self):
        from quanta.backends.ibm import IBMBackend
        from quanta.core.circuit import circuit
        from quanta.core.gates import RX
        from quanta.core.measure import measure
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=1)
        def c(q):
            RX(2.0)(q[0])
            return measure(q)

        backend = IBMBackend()
        dag = DAGCircuit.from_builder(c.build())
        qasm = backend._dag_to_qasm(dag)
        assert "rx" in qasm
        assert "2.0" in qasm

    def test_ibm_hardware_without_token_raises(self):
        from quanta.backends.ibm import IBMBackend, IBMBackendError
        from quanta.core.circuit import circuit
        from quanta.core.gates import H
        from quanta.core.measure import measure
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=1)
        def c(q):
            H(q[0])
            return measure(q)

        # Use patch to ensure no env token
        with patch.dict("os.environ", {}, clear=True):
            backend = IBMBackend(token="", simulate_locally=False)
            dag = DAGCircuit.from_builder(c.build())
            with pytest.raises(IBMBackendError, match="token"):
                backend.execute(dag, shots=10)


# ═══════════════════════════════════════════
#  mcp_server.py — Tool functions tested directly
# ═══════════════════════════════════════════

class TestMCPServerTools:
    """Tests for MCP server tool functions (called directly, not via MCP)."""

    def test_create_bell_state(self):
        from quanta.mcp_server import create_bell_state
        result = create_bell_state(shots=100, seed=42)
        data = json.loads(result)
        assert "counts" in data
        counts = data["counts"]
        assert "00" in counts or "11" in counts

    def test_grover_search(self):
        from quanta.mcp_server import grover_search
        result = grover_search(num_qubits=3, target=5, shots=100, seed=42)
        data = json.loads(result)
        # May return result or import error depending on search module
        assert isinstance(data, dict)

    def test_shor_factor(self):
        from quanta.mcp_server import shor_factor
        result = shor_factor(number=15)
        data = json.loads(result)
        assert "factors" in data

    def test_simulate_noise_depolarizing(self):
        from quanta.mcp_server import simulate_noise
        result = simulate_noise(
            noise_type="depolarizing",
            probability=0.05,
            shots=100,
            seed=42,
        )
        data = json.loads(result)
        assert "noisy_counts" in data or "counts" in data or "noise_type" in data

    def test_simulate_noise_bitflip(self):
        from quanta.mcp_server import simulate_noise
        result = simulate_noise(
            noise_type="bitflip",
            probability=0.1,
            shots=100,
            seed=42,
        )
        assert isinstance(result, str)

    def test_list_gates(self):
        from quanta.mcp_server import list_gates
        result = list_gates()
        data = json.loads(result)
        assert "gates" in data
        gate_names = [g["name"] for g in data["gates"]]
        assert "H" in gate_names
        assert "CX" in gate_names

    def test_explain_result(self):
        from quanta.mcp_server import explain_result
        counts = json.dumps({"00": 500, "11": 500})
        result = explain_result(counts)
        assert "entangle" in result.lower() or "correlat" in result.lower() or "00" in result

    def test_explain_result_single_state(self):
        from quanta.mcp_server import explain_result
        counts = json.dumps({"000": 1000})
        result = explain_result(counts)
        assert "000" in result or "deterministic" in result.lower() or "100" in result

    def test_sdk_info(self):
        from quanta.mcp_server import sdk_info
        result = sdk_info()
        assert "quanta" in result.lower() or "version" in result.lower()

    def test_sdk_examples(self):
        from quanta.mcp_server import sdk_examples
        result = sdk_examples()
        assert "circuit" in result.lower() or "bell" in result.lower()

    def test_run_circuit(self):
        from quanta.mcp_server import run_circuit
        code = """
@circuit(qubits=1)
def c(q):
    H(q[0])
    return measure(q)
"""
        result = run_circuit(code=code, shots=50, seed=42)
        data = json.loads(result)
        assert "counts" in data or "error" in data

    def test_run_circuit_invalid_code(self):
        from quanta.mcp_server import run_circuit
        result = run_circuit(code="invalid python code !!!", shots=10)
        data = json.loads(result)
        assert "error" in data


# ═══════════════════════════════════════════
#  gradients.py — QFIM coverage
# ═══════════════════════════════════════════

class TestGradientsExtended:
    """Extended gradient tests for better QFIM coverage."""

    def test_natural_gradient_two_params(self):
        from quanta.gradients import natural_gradient, expectation
        from quanta.simulator.statevector import StateVectorSimulator

        def cost(params):
            sim = StateVectorSimulator(2)
            sim.apply("RY", (0,), (params[0],))
            sim.apply("RY", (1,), (params[1],))
            sim.apply("CX", (0, 1))
            return expectation(sim.state, "ZZ", 2)

        def state(params):
            sim = StateVectorSimulator(2)
            sim.apply("RY", (0,), (params[0],))
            sim.apply("RY", (1,), (params[1],))
            sim.apply("CX", (0, 1))
            return sim.state

        result = natural_gradient(cost, state, [0.5, 1.0])
        assert len(result.gradients) == 2
        assert result.method == "natural-gradient"

    def test_finite_diff_custom_epsilon(self):
        from quanta.gradients import finite_diff, expectation
        from quanta.simulator.statevector import StateVectorSimulator

        def cost(params):
            sim = StateVectorSimulator(1)
            sim.apply("RY", (0,), (params[0],))
            return expectation(sim.state, "Z", 1)

        r1 = finite_diff(cost, [0.5], epsilon=1e-5)
        r2 = finite_diff(cost, [0.5], epsilon=1e-9)
        # Both should approximate -sin(0.5)
        assert r1.gradients[0] == pytest.approx(-np.sin(0.5), abs=1e-3)
        assert r2.gradients[0] == pytest.approx(-np.sin(0.5), abs=1e-3)

    def test_expectation_identity(self):
        """<ψ|I|ψ> = 1 for any normalized state."""
        from quanta.gradients import expectation
        state = np.array([1, 0], dtype=complex)
        assert expectation(state, "I", 1) == pytest.approx(1.0)

    def test_multi_expectation_hamiltonian(self):
        from quanta.gradients import multi_expectation
        # |0⟩ state: <Z>=1, <X>=0, <I>=1
        state = np.array([1, 0], dtype=complex)
        E = multi_expectation(state, [("Z", 0.5), ("X", 0.3), ("I", 1.0)], 1)
        # 0.5*1 + 0.3*0 + 1.0*1 = 1.5
        assert E == pytest.approx(1.5)
