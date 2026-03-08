"""
tests/test_compiler.py — Compiler pipeline ve optimizasyon testleri.

Gate iptal, rotasyon birleştirme ve pipeline istatistiklerini test eder.
"""

import numpy as np

from quanta import CX, RZ, H, X, circuit, measure, run
from quanta.compiler.passes.optimize import CancelInverses, MergeRotations
from quanta.compiler.pipeline import CompilerPipeline
from quanta.dag.dag_circuit import DAGCircuit

# ═══════════════════════════════════════════
#  CancelInverses Testleri
# ═══════════════════════════════════════════

class TestCancelInverses:
    """Ters kapıların iptal edilmesi testleri."""

    def test_hh_cancels_to_empty(self):
        """H·H = I → 0 kapı kalmalı."""
        @circuit(qubits=1)
        def hh(q):
            H(q[0])
            H(q[0])

        dag = DAGCircuit.from_builder(hh.build())
        assert dag.gate_count() == 2

        optimized = CancelInverses().run(dag)
        assert optimized.gate_count() == 0

    def test_xx_cancels_to_empty(self):
        """X·X = I → 0 kapı kalmalı."""
        @circuit(qubits=1)
        def xx(q):
            X(q[0])
            X(q[0])

        dag = DAGCircuit.from_builder(xx.build())
        optimized = CancelInverses().run(dag)
        assert optimized.gate_count() == 0

    def test_hxh_keeps_x(self):
        """H·X·H → H ve X iptal edilmez (farklı kapılar)."""
        @circuit(qubits=1)
        def hxh(q):
            H(q[0])
            X(q[0])
            H(q[0])

        dag = DAGCircuit.from_builder(hxh.build())
        optimized = CancelInverses().run(dag)
        assert optimized.gate_count() == 3  # Hiçbiri iptal edilmemeli

    def test_different_qubits_not_cancelled(self):
        """H(q0)·H(q1) iptal edilmemeli (farklı qubit'ler)."""
        @circuit(qubits=2)
        def diff(q):
            H(q[0])
            H(q[1])

        dag = DAGCircuit.from_builder(diff.build())
        optimized = CancelInverses().run(dag)
        assert optimized.gate_count() == 2

    def test_cx_cx_cancels(self):
        """CX·CX = I → iptal edilmeli."""
        @circuit(qubits=2)
        def cxcx(q):
            CX(q[0], q[1])
            CX(q[0], q[1])

        dag = DAGCircuit.from_builder(cxcx.build())
        optimized = CancelInverses().run(dag)
        assert optimized.gate_count() == 0


# ═══════════════════════════════════════════
#  MergeRotations Testleri
# ═══════════════════════════════════════════

class TestMergeRotations:
    """Rotasyon birleştirme testleri."""

    def test_rz_rz_merges(self):
        """RZ(π/4)·RZ(π/4) = RZ(π/2)."""
        @circuit(qubits=1)
        def rz_twice(q):
            RZ(np.pi / 4)(q[0])
            RZ(np.pi / 4)(q[0])

        dag = DAGCircuit.from_builder(rz_twice.build())
        assert dag.gate_count() == 2

        optimized = MergeRotations().run(dag)
        assert optimized.gate_count() == 1

    def test_rz_opposite_cancels(self):
        """RZ(π)·RZ(-π) = RZ(0) ≈ I → kaldırılır."""
        @circuit(qubits=1)
        def rz_cancel(q):
            RZ(np.pi)(q[0])
            RZ(-np.pi)(q[0])

        dag = DAGCircuit.from_builder(rz_cancel.build())
        optimized = MergeRotations().run(dag)
        assert optimized.gate_count() == 0


# ═══════════════════════════════════════════
#  Pipeline Testleri
# ═══════════════════════════════════════════

class TestCompilerPipeline:
    """Pipeline entegrasyon testleri."""

    def test_empty_pipeline_no_change(self):
        @circuit(qubits=1)
        def simple(q):
            H(q[0])

        dag = DAGCircuit.from_builder(simple.build())
        pipeline = CompilerPipeline()
        result = pipeline.run(dag)
        assert result.gate_count() == 1

    def test_full_pipeline_optimizes(self):
        """Pipeline birden fazla pass çalıştırmalı."""
        @circuit(qubits=1)
        def redundant(q):
            H(q[0])
            H(q[0])
            RZ(np.pi / 4)(q[0])
            RZ(np.pi / 4)(q[0])

        dag = DAGCircuit.from_builder(redundant.build())
        assert dag.gate_count() == 4

        pipeline = CompilerPipeline([CancelInverses(), MergeRotations()])
        result = pipeline.run(dag)
        assert result.gate_count() == 1  # H·H iptal, RZ birleşti

    def test_pipeline_stats_recorded(self):
        @circuit(qubits=1)
        def test_circ(q):
            H(q[0])
            H(q[0])

        dag = DAGCircuit.from_builder(test_circ.build())
        pipeline = CompilerPipeline([CancelInverses()])
        pipeline.run(dag)

        assert "CancelInverses" in pipeline.stats
        assert pipeline.stats["CancelInverses"]["gates_removed"] == 2

    def test_pipeline_summary_returns_string(self):
        @circuit(qubits=1)
        def test_circ(q):
            H(q[0])

        dag = DAGCircuit.from_builder(test_circ.build())
        pipeline = CompilerPipeline([CancelInverses()])
        pipeline.run(dag)
        assert isinstance(pipeline.summary(), str)


# ═══════════════════════════════════════════
#  Algoritma Doğruluk Testleri
# ═══════════════════════════════════════════

class TestAlgorithms:
    """Kuantum algoritma doğruluk testleri — gerçek iş durumları."""

    def test_deutsch_jozsa_constant_returns_all_zeros(self):
        """Sabit oracle → tüm ölçümler |00⟩ olmalı."""
        @circuit(qubits=3)
        def dj_constant(q):
            X(q[2])
            H(q[0])
            H(q[1])
            H(q[2])
            # Sabit oracle: identity
            H(q[0])
            H(q[1])
            return measure(q[0], q[1])

        result = run(dj_constant, shots=100, seed=42)
        assert result.counts == {"00": 100}

    def test_deutsch_jozsa_balanced_never_all_zeros(self):
        """Dengeli oracle → |00⟩ ASLA çıkmamalı."""
        @circuit(qubits=3)
        def dj_balanced(q):
            X(q[2])
            H(q[0])
            H(q[1])
            H(q[2])
            CX(q[0], q[2])
            CX(q[1], q[2])
            H(q[0])
            H(q[1])
            return measure(q[0], q[1])

        result = run(dj_balanced, shots=1000, seed=42)
        assert "00" not in result.counts
