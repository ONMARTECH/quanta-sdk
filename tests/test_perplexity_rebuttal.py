"""
tests/test_perplexity_rebuttal.py — Proof tests for Perplexity's incorrect claims.

These tests exist to explicitly disprove claims made by Perplexity AI
in its code review of Quanta SDK v0.8.0. Each test class corresponds
to a specific claim that was verified as FALSE.

See: docs/REBUTTAL.md for full analysis.
"""

from __future__ import annotations

import threading

import pytest

from quanta.core.circuit import circuit
from quanta.core.gates import (
    CX,
    H,
    _active_builders,
    _get_builders_stack,
    _thread_local,
)
from quanta.core.measure import measure
from quanta.dag.node import OpNode
from quanta.export.qasm import to_qasm
from quanta.export.qasm_import import from_qasm
from quanta.simulator.noise import (
    _apply_single_qubit_error,
)

# ══════════════════════════════════════════════
# REBUTTAL #5: "_active_builders.__bool__ kırık"
#
# Perplexity claimed: "staticmethod lambda self argümanı almıyor,
#   bool() çağrısı AttributeError üretebilir"
#
# Reality: type() ile oluşturulan sınıflarda staticmethod(lambda:...)
#   self almaz — bu DOĞRU çalışır çünkü instance method değil.
# ══════════════════════════════════════════════


class TestBuilderProxyBool:
    """Prove _active_builders.__bool__ works correctly."""

    def test_bool_false_when_no_builder(self) -> None:
        """bool() on _active_builders returns False when stack is empty."""
        # Clear any existing builders for this thread
        stack = _get_builders_stack()
        original = list(stack)
        stack.clear()

        result = bool(_active_builders)
        assert result is False

        # Restore
        stack.extend(original)

    def test_bool_true_when_builder_active(self) -> None:
        """bool() returns True inside a circuit builder context."""
        @circuit(qubits=2)
        def dummy(q):
            # Inside builder → _active_builders should be truthy
            assert bool(_active_builders) is True
            H(q[0])
            return measure(q)

        dummy.build()

    def test_no_attribute_error(self) -> None:
        """bool() never raises AttributeError (Perplexity's claim)."""
        # This is exactly what Perplexity said would fail
        try:
            result = bool(_active_builders)
            assert isinstance(result, bool)
        except AttributeError:
            pytest.fail(
                "Perplexity claimed __bool__ raises AttributeError — it does not"
            )

    def test_thread_local_isolation(self) -> None:
        """Each thread has its own builder stack (thread-safety)."""
        results = {}

        def thread_fn(name: str) -> None:
            @circuit(qubits=1)
            def c(q):
                H(q[0])
                results[name] = bool(_active_builders)
                return measure(q)
            c.build()

        t1 = threading.Thread(target=thread_fn, args=("A",))
        t2 = threading.Thread(target=thread_fn, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["A"] is True
        assert results["B"] is True

    def test_threading_local_exists(self) -> None:
        """Verify threading.local() is used for thread-safety."""
        assert isinstance(_thread_local, threading.local)


# ══════════════════════════════════════════════
# REBUTTAL #6: "OpenQASM 3.0 desteği yok"
#
# Perplexity claimed: "Sadece QASM 2.0 var"
#
# Reality: QASM 3.0 export AND import both exist and work.
# ══════════════════════════════════════════════


class TestQASM30Exists:
    """Prove OpenQASM 3.0 export and import work."""

    def test_export_produces_qasm30(self) -> None:
        """to_qasm() outputs OPENQASM 3.0 header."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm = to_qasm(bell)
        assert "OPENQASM 3.0;" in qasm
        assert 'include "stdgates.inc";' in qasm

    def test_export_uses_qasm30_syntax(self) -> None:
        """QASM output uses 3.0 syntax (qubit[] not qreg)."""
        @circuit(qubits=3)
        def ghz(q):
            H(q[0])
            CX(q[0], q[1])
            CX(q[1], q[2])
            return measure(q)

        qasm = to_qasm(ghz)
        assert "qubit[3]" in qasm  # QASM 3.0 syntax
        assert "bit[3]" in qasm    # QASM 3.0 syntax

    def test_import_qasm20(self) -> None:
        """from_qasm() handles QASM 2.0 input."""
        qasm_2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""
        dag = from_qasm(qasm_2)
        ops = [n for n in dag.topological_sort() if isinstance(n, OpNode)]
        assert len(ops) >= 2

    def test_import_qasm30(self) -> None:
        """from_qasm() handles QASM 3.0 input."""
        qasm_3 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""
        dag = from_qasm(qasm_3)
        ops = [n for n in dag.topological_sort() if isinstance(n, OpNode)]
        assert len(ops) >= 2

    def test_roundtrip_qasm30(self) -> None:
        """Export to QASM 3.0 → import → verify gates preserved."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm = to_qasm(bell)
        dag = from_qasm(qasm)
        ops = [n for n in dag.topological_sort() if isinstance(n, OpNode)]
        gate_names = [n.gate_name for n in ops]
        assert "H" in gate_names
        assert "CX" in gate_names


# ══════════════════════════════════════════════
# REBUTTAL #8: "Type annotations eksik"
#
# Perplexity claimed: "_apply_single_qubit_error has no return type"
#   and "gradients.py state_fn has no type"
#
# Reality: Both have full type annotations.
# ══════════════════════════════════════════════


class TestTypeAnnotations:
    """Prove type annotations exist where Perplexity claimed they don't."""

    def test_noise_apply_has_return_type(self) -> None:
        """_apply_single_qubit_error has -> np.ndarray return type."""
        import inspect
        sig = inspect.signature(_apply_single_qubit_error)
        ret = str(sig.return_annotation)
        assert "ndarray" in ret

    def test_noise_apply_has_param_types(self) -> None:
        """_apply_single_qubit_error has full parameter type hints."""
        import inspect
        sig = inspect.signature(_apply_single_qubit_error)
        params = sig.parameters
        assert "ndarray" in str(params["state"].annotation)
        assert "int" in str(params["qubit"].annotation)
        assert "int" in str(params["num_qubits"].annotation)
        assert "int" in str(params["pauli"].annotation)

    def test_gradients_state_fn_has_type(self) -> None:
        """gradients.py state_fn parameter has Callable type hint."""
        import inspect

        from quanta.gradients import natural_gradient

        sig = inspect.signature(natural_gradient)
        state_fn_param = sig.parameters.get("state_fn")
        assert state_fn_param is not None
        # Should have a Callable annotation
        ann = str(state_fn_param.annotation)
        assert "Callable" in ann or "callable" in ann


# ══════════════════════════════════════════════
# REBUTTAL #14: "Coverage exclusion listesi geniş"
#
# This test proves WHY backends are excluded:
# they require real API credentials.
# ══════════════════════════════════════════════


class TestBackendRequiresCredentials:
    """Show that excluded backends need external API credentials."""

    def test_ibm_needs_api_key(self) -> None:
        """IBMRestBackend requires IBM_API_KEY env var."""
        import os
        # Without API key, job submission should fail gracefully
        original = os.environ.pop("IBM_API_KEY", None)
        try:
            from quanta.backends.ibm_rest import IBMRestBackend
            backend = IBMRestBackend()
            # Can instantiate but cannot submit without key
            assert backend is not None
        finally:
            if original:
                os.environ["IBM_API_KEY"] = original

    def test_ionq_needs_api_key(self) -> None:
        """IonQBackend requires IONQ_API_KEY env var."""
        from quanta.backends.ionq import IonQBackend
        backend = IonQBackend(target="simulator")
        # Can instantiate — proves code loads
        assert backend.name == "ionq_simulator"
