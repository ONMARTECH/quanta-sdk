"""
tests/test_routing.py -- Tests for topology-aware qubit routing.

Tests all topology types and SWAP insertion logic.
"""

import pytest

from quanta.compiler.passes.routing import (
    RouteToTopology,
    Topology,
)
from quanta.core.circuit import CircuitBuilder, circuit
from quanta.core.gates import CX, H, X
from quanta.core.measure import measure
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit

# ═══════════════════════════════════════════
#  Topology generation
# ═══════════════════════════════════════════

class TestTopologyGeneration:
    def test_linear_topology(self):
        t = Topology.line(5)
        assert (0, 1) in t.edges
        assert (3, 4) in t.edges
        assert len(t.edges) == 4  # n-1 edges

    def test_linear_no_wrap(self):
        t = Topology.line(4)
        assert (0, 3) not in t.edges  # no wrap-around

    def test_ring_topology(self):
        t = Topology.ring(5)
        assert (0, 4) in t.edges  # wrap-around edge
        assert len(t.edges) == 5  # n edges

    def test_ring_has_all_linear_edges(self):
        ring = Topology.ring(4)
        linear = Topology.line(4)
        assert linear.edges.issubset(ring.edges)

    def test_grid_topology_2x2(self):
        t = Topology.grid(2, 2)
        # 0-1, 2-3 (horizontal) + 0-2, 1-3 (vertical)
        assert (0, 1) in t.edges
        assert (2, 3) in t.edges
        assert (0, 2) in t.edges
        assert (1, 3) in t.edges
        assert len(t.edges) == 4

    def test_grid_topology_3x3(self):
        t = Topology.grid(3, 3)
        assert (0, 1) in t.edges   # horizontal
        assert (0, 3) in t.edges   # vertical
        assert (4, 5) in t.edges   # center-right
        assert (4, 7) in t.edges   # center-down
        assert len(t.edges) == 12  # 3*2 + 3*2 per direction


# ═══════════════════════════════════════════
#  RouteToTopology pass
# ═══════════════════════════════════════════

class TestRouteToTopology:
    def test_invalid_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            RouteToTopology(topology="hexagonal", num_qubits=5)

    def test_adjacent_gate_no_swap(self):
        """CX on adjacent qubits should not insert any SWAPs."""
        @circuit(qubits=3)
        def c(q):
            CX(q[0], q[1])  # adjacent on linear
            return measure(q)

        dag = DAGCircuit.from_builder(c.build())
        router = RouteToTopology(topology="linear", num_qubits=3)
        routed = router.run(dag)

        # Count SWAP gates
        swaps = [op for op in routed.op_nodes() if op.gate_name == "SWAP"]
        assert len(swaps) == 0

    def test_non_adjacent_inserts_swap_linear(self):
        """CX(0,2) on linear topology must insert SWAP."""
        builder = CircuitBuilder(3)
        builder.record(Instruction("CX", (0, 2), ()))

        dag = DAGCircuit.from_builder(builder)
        router = RouteToTopology(topology="linear", num_qubits=3)
        routed = router.run(dag)

        swaps = [op for op in routed.op_nodes() if op.gate_name == "SWAP"]
        assert len(swaps) >= 1

    def test_ring_shorter_path(self):
        """Ring topology: CX(0,3) on 5 qubits should use short path via 4."""
        builder = CircuitBuilder(5)
        builder.record(Instruction("CX", (0, 4), ()))

        dag = DAGCircuit.from_builder(builder)

        linear_router = RouteToTopology(topology="linear", num_qubits=5)
        ring_router = RouteToTopology(topology="ring", num_qubits=5)

        linear_routed = linear_router.run(dag)
        ring_routed = ring_router.run(dag)

        linear_swaps = sum(1 for op in linear_routed.op_nodes() if op.gate_name == "SWAP")
        ring_swaps = sum(1 for op in ring_routed.op_nodes() if op.gate_name == "SWAP")

        assert ring_swaps <= linear_swaps  # ring should need fewer/equal SWAPs

    def test_grid_routing(self):
        """Grid topology should route non-adjacent qubits."""
        builder = CircuitBuilder(4)
        builder.record(Instruction("CX", (0, 3), ()))

        dag = DAGCircuit.from_builder(builder)
        router = RouteToTopology(topology="grid", num_qubits=4,
                                  grid_rows=2, grid_cols=2)
        routed = router.run(dag)

        # On 2x2 grid, 0-3 are diagonal (not adjacent), needs SWAP
        ops = list(routed.op_nodes())
        assert any(op.gate_name == "SWAP" for op in ops)

    def test_custom_edges(self):
        """Custom topology."""
        custom = {(0, 2), (1, 2)}
        router = RouteToTopology(
            topology="custom", num_qubits=3,
            custom_edges=custom
        )
        assert router._are_adjacent(0, 2)
        assert router._are_adjacent(2, 0)  # bidirectional
        assert not router._are_adjacent(0, 1)

    def test_single_qubit_gates_preserved(self):
        """Single-qubit gates should pass through unchanged."""
        @circuit(qubits=3)
        def c(q):
            H(q[0])
            X(q[2])
            return measure(q)

        dag = DAGCircuit.from_builder(c.build())
        router = RouteToTopology(topology="linear", num_qubits=3)
        routed = router.run(dag)

        gates = [op.gate_name for op in routed.op_nodes()]
        assert "H" in gates
        assert "X" in gates
        assert "SWAP" not in gates

    def test_shortest_path(self):
        """BFS shortest path on linear topology."""
        router = RouteToTopology(topology="linear", num_qubits=5)
        path = router._shortest_path(0, 4, 5)
        assert path == [0, 1, 2, 3, 4]

    def test_shortest_path_ring(self):
        """BFS shortest path on ring topology (should wrap)."""
        router = RouteToTopology(topology="ring", num_qubits=5)
        path = router._shortest_path(0, 4, 5)
        # Should take the short way: 0 -> 4 (direct)
        assert len(path) == 2  # [0, 4]

    def test_measurement_preserved(self):
        """Measurement spec should be preserved after routing."""
        @circuit(qubits=3)
        def c(q):
            CX(q[0], q[2])
            return measure(q)

        dag = DAGCircuit.from_builder(c.build())
        router = RouteToTopology(topology="linear", num_qubits=3)
        routed = router.run(dag)

        assert routed.measurement is not None
