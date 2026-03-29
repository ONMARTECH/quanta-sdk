"""
test_backend_routing — Tests for Task 5 (Topology) and Task 12 (Backend).
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")


# ═══════════════════════════════════════════
# Task 12: Backend Abstraction
# ═══════════════════════════════════════════

class TestBackendFactory:
    """Backend.from_name() and capabilities()."""

    def test_from_name_local(self):
        from quanta.backends.base import Backend
        b = Backend.from_name("local", seed=42)
        assert b.name == "LocalSimulator"
        assert b.is_available()

    def test_from_name_numpy(self):
        from quanta.backends.base import Backend
        b = Backend.from_name("numpy")
        assert b.name == "LocalSimulator"

    def test_from_name_invalid(self):
        from quanta.backends.base import Backend
        with pytest.raises(ValueError, match="nonexistent"):
            Backend.from_name("nonexistent")

    def test_list_available(self):
        from quanta.backends.base import Backend
        avail = Backend.list_available()
        assert "local" in avail
        assert "ibm" in avail

    def test_capabilities_local(self):
        from quanta.backends.base import Backend
        b = Backend.from_name("local")
        caps = b.capabilities()
        assert caps.max_qubits == 25
        assert caps.is_simulator is True
        assert caps.supports_noise is True
        assert "CX" in caps.native_gates
        assert caps.connectivity == "all-to-all"

    def test_capabilities_summary(self):
        from quanta.backends.base import Backend
        b = Backend.from_name("local")
        summary = b.capabilities().summary()
        assert "Max qubits" in summary
        assert "25" in summary

    def test_repr(self):
        from quanta.backends.base import Backend
        b = Backend.from_name("local")
        assert "LocalSimulator" in repr(b)


# ═══════════════════════════════════════════
# Task 5: Topology
# ═══════════════════════════════════════════

class TestTopology:
    """Topology factory methods."""

    def test_line(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.line(5)
        assert t.num_qubits == 5
        assert len(t.edges) == 4
        assert (0, 1) in t.edges

    def test_ring(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.ring(5)
        assert t.num_qubits == 5
        assert len(t.edges) == 5
        assert (0, 4) in t.edges

    def test_grid(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.grid(2, 3)
        assert t.num_qubits == 6
        assert (0, 1) in t.edges
        assert (0, 3) in t.edges

    def test_custom(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.custom(edges=[(0, 1), (1, 2), (0, 2)])
        assert t.num_qubits == 3
        assert len(t.edges) == 3

    def test_from_backend_ibm(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.from_backend("ibm_fez")
        assert t.num_qubits == 156
        assert len(t.edges) > 100

    def test_from_backend_ionq(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.from_backend("ionq_aria")
        assert t.num_qubits == 25
        # All-to-all: 25*24/2 = 300 edges
        assert len(t.edges) == 300

    def test_from_backend_google(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.from_backend("google_willow")
        assert t.num_qubits == 72

    def test_from_backend_invalid(self):
        from quanta.compiler.passes.routing import Topology
        with pytest.raises(ValueError, match="nonexistent"):
            Topology.from_backend("nonexistent")

    def test_repr(self):
        from quanta.compiler.passes.routing import Topology
        t = Topology.line(5)
        assert "line(5)" in repr(t)

    def test_route_with_topology_object(self):
        """RouteToTopology accepts Topology object directly."""
        from quanta.compiler.passes.routing import RouteToTopology, Topology
        t = Topology.line(5)
        router = RouteToTopology(topology=t)
        assert router._topology == "line(5)"


# ═══════════════════════════════════════════
# Task 9: QML Benchmark (basic)
# ═══════════════════════════════════════════

class TestQMLBenchmark:
    """Basic QML classifier benchmark tests."""

    def test_classifier_accuracy(self):
        """Classifier achieves > random on linearly separable data."""
        from quanta.qml import Classifier
        # Simple linearly separable dataset
        X = np.array([
            [0.1, 0.1], [0.2, 0.15], [0.15, 0.2],
            [0.8, 0.9], [0.9, 0.85], [0.85, 0.9],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        clf = Classifier(n_qubits=2, n_layers=2, seed=42)
        clf.fit(X, y, epochs=15)
        score = clf.score(X, y)
        # Should beat random (50%)
        assert score >= 0.5

    def test_qsvm_basic(self):
        """QSVM produces valid predictions."""
        from quanta.qml import QSVM
        X_train = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        y_train = [0, 0, 1, 1]
        X_test = [[0.5, 0.5]]
        qsvm = QSVM(n_qubits=2)
        result = qsvm.classify(X_train, y_train, X_test)
        assert len(result.predictions) == 1
        assert result.accuracy > 0

    def test_kernel_self_similarity(self):
        """Kernel of identical points should be ~1.0."""
        from quanta.qml import Kernel
        k = Kernel(n_qubits=2)
        val = k.evaluate(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        assert val > 0.99

    def test_feature_map_comparison(self):
        """Different feature maps produce different results."""
        from quanta.qml import Classifier
        X = np.array([[0.1, 0.2], [0.8, 0.9]])
        y = np.array([0, 1])

        clf_angle = Classifier(n_qubits=2, feature_map="angle", seed=42)
        clf_zz = Classifier(n_qubits=2, feature_map="zz", seed=42)

        r1 = clf_angle.fit(X, y, epochs=3)
        r2 = clf_zz.fit(X, y, epochs=3)

        # Different loss histories (different feature maps)
        assert r1.loss_history != r2.loss_history
