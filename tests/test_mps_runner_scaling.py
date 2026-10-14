"""
tests/test_mps_runner_scaling.py -- Tests for MPS Simulator Deep Integration,
CircuitRunner, High-Qubit Safeguard, MCP Server Exposure, and 15–30 Qubit Scaling.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quanta import CX, H, circuit, measure, run
from quanta.mcp_server import mps_circuit_simulate, run_circuit
from quanta.runner import CircuitResult, CircuitRunner
from quanta.simulator.mps import MPSSimulator
from quanta.simulator.statevector import StateVectorSimulator


class TestCircuitRunnerIntegration:
    """Tests for CircuitRunner architecture and execution routing."""

    def test_circuit_runner_init_and_defaults(self) -> None:
        """Verifies default parameters and class structure of CircuitRunner."""
        runner = CircuitRunner()
        assert runner.default_method == "auto"
        assert runner.max_dense_qubits == 27
        assert runner.chi_max == 64
        assert runner.cutoff == 1e-12
        assert runner.shots == 1024

    def test_circuit_runner_dense_execution(self) -> None:
        """Verifies execution of low-qubit circuit with default CircuitRunner."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        runner = CircuitRunner()
        result = runner.run(bell, shots=500)

        assert isinstance(result, CircuitResult)
        assert result.shots == 500
        assert result.num_qubits == 2
        assert set(result.counts.keys()).issubset({"00", "11"})
        assert result.statevector is not None
        assert len(result.statevector) == 4

    def test_circuit_runner_explicit_mps(self) -> None:
        """Verifies execution with method='mps' through CircuitRunner."""
        @circuit(qubits=3)
        def ghz3(q):
            H(q[0])
            CX(q[0], q[1])
            CX(q[1], q[2])
            return measure(q)

        runner = CircuitRunner(default_method="mps")
        result = runner.run(ghz3, shots=256)

        assert result.shots == 256
        assert result.num_qubits == 3
        assert set(result.counts.keys()).issubset({"000", "111"})
        # High-qubit safeguard: statevector=None when using MPS
        assert result.statevector is None

    def test_circuit_runner_override_method(self) -> None:
        """Verifies per-run method override in CircuitRunner."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        runner = CircuitRunner(default_method="dense")
        result = runner.run(bell, method="mps")
        assert result.statevector is None
        assert set(result.counts.keys()).issubset({"00", "11"})


class TestHighQubitSafeguardAndAutoRouting:
    """Tests for high-qubit memory safeguard and auto-routing."""

    def test_runner_method_mps_dispatch(self) -> None:
        """Verifies quanta.runner.run(..., method='mps') executes cleanly on 16 qubits."""
        @circuit(qubits=16)
        def ghz_16(q):
            H(q[0])
            for i in range(15):
                CX(q[i], q[i + 1])
            return measure(q)

        result = run(ghz_16, shots=500, method="mps")
        assert result.shots == 500
        assert result.num_qubits == 16
        # GHZ only produces all-0 or all-1 bitstrings
        assert set(result.counts.keys()).issubset({"0" * 16, "1" * 16})
        # MPS safeguard
        assert result.statevector is None

    def test_high_qubit_safeguard_for_n_greater_than_20(self) -> None:
        """When N > 20, statevector MUST be None to prevent 2^N allocation."""
        @circuit(qubits=22)
        def ghz_22(q):
            H(q[0])
            for i in range(21):
                CX(q[i], q[i + 1])
            return measure(q)

        result = run(ghz_22, shots=100, method="mps")
        assert result.num_qubits == 22
        assert result.statevector is None
        assert set(result.counts.keys()).issubset({"0" * 22, "1" * 22})

    def test_auto_routing_to_mps_above_threshold(self) -> None:
        """Verifies method='auto' routes to MPS when N > max_dense_qubits."""
        @circuit(qubits=28)
        def ghz_28(q):
            H(q[0])
            for i in range(27):
                CX(q[i], q[i + 1])
            return measure(q)

        # 28 qubits with dense would allocate 4 GB; auto-routing to MPS keeps it < 5 KB
        result = run(ghz_28, shots=100, method="auto", max_dense_qubits=27)
        assert result.num_qubits == 28
        assert result.statevector is None
        assert set(result.counts.keys()).issubset({"0" * 28, "1" * 28})

    def test_result_display_methods_without_statevector(self) -> None:
        """Verifies that dirac_notation, histogram, and summary work when statevector is None."""
        @circuit(qubits=15)
        def ghz_15(q):
            H(q[0])
            for i in range(14):
                CX(q[i], q[i + 1])
            return measure(q)

        result = run(ghz_15, shots=200, method="mps")
        assert result.statevector is None

        dirac = result.dirac_notation()
        assert isinstance(dirac, str)
        assert "|" in dirac

        hist = result.histogram()
        assert isinstance(hist, str)
        assert "█" in hist

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Quanta Result" in summary


class TestMPSScalingAndFidelity:
    """Tests for 15-30 qubit scaling, memory containment, and statevector fidelity."""

    def test_15_qubit_mps_vs_dense_exact_fidelity(self) -> None:
        """Cross-validates 15-qubit MPS against dense StateVectorSimulator (Fidelity >= 0.99999)."""
        n = 15
        dense = StateVectorSimulator(n, seed=42)
        mps = MPSSimulator(n, seed=42, chi_max=32)

        # Apply 1D nearest-neighbor entangling layer
        for q in range(n):
            dense.apply("H", (q,), ())
            mps.apply("H", (q,), ())
        for q in range(n - 1):
            dense.apply("CX", (q, q + 1), ())
            mps.apply("CX", (q, q + 1), ())

        state_dense = dense.state
        state_mps = mps.state
        overlap = np.vdot(state_mps, state_dense)
        fidelity = float(np.abs(overlap) ** 2)
        assert fidelity >= 0.99999, f"Fidelity was {fidelity:.6f} (expected >= 0.99999)"

    @pytest.mark.parametrize("num_qubits", [15, 20, 25, 30])
    def test_mps_15_to_30_qubits_memory_containment_and_zero_truncation(self, num_qubits: int) -> None:
        """Demonstrates memory containment (<500 KB) and near-zero truncation error for 15-30 qubits."""
        sim = MPSSimulator(num_qubits=num_qubits, chi_max=16)

        # 1D GHZ circuit (exact at chi=2)
        sim.apply("H", (0,), ())
        for i in range(num_qubits - 1):
            sim.apply("CX", (i, i + 1), ())

        # Truncation error must be near-zero
        assert sim.truncation_error <= 1e-12

        # Memory containment assertion: < 500 KB even at 30 qubits
        assert sim.memory_bytes < 500_000, f"Memory footprint exceeded limit: {sim.memory_bytes} bytes"

        # Measurement sampling produces valid GHZ bitstrings
        counts = sim.sample(50)
        assert set(counts.keys()).issubset({"0" * num_qubits, "1" * num_qubits})

    def test_shallow_ising_scaling_30_qubits(self) -> None:
        """Verifies shallow 1D transverse-field Ising circuit on 30 qubits without OOM."""
        sim = MPSSimulator(num_qubits=30, chi_max=32, seed=42)

        for q in range(30):
            sim.apply("H", (q,), ())
        for q in range(29):
            sim.apply("CX", (q, q + 1), ())
            sim.apply("RZ", (q + 1,), (0.2,))
            sim.apply("CX", (q, q + 1), ())
        for q in range(30):
            sim.apply("RX", (q,), (0.1,))

        assert sim.memory_bytes < 500_000
        assert sim.truncation_error <= 1e-6
        counts = sim.sample(20)
        assert len(counts) > 0


class TestMCPServerExposure:
    """Tests for MCP server tool updates and exposure."""

    def test_mcp_run_circuit_mps_support(self) -> None:
        """Verifies Quanta MCP tool run_circuit accepts method='mps'."""
        code = """
@circuit(qubits=18)
def circ(q):
    H(q[0])
    for i in range(17):
        CX(q[i], q[i + 1])
    return measure(q)
"""
        raw_json = run_circuit(code=code, shots=256, method="mps")
        data = json.loads(raw_json)
        assert "counts" in data
        assert data["num_qubits"] == 18
        assert set(data["counts"].keys()).issubset({"0" * 18, "1" * 18})

    def test_mcp_run_circuit_statevector_support(self) -> None:
        """Verifies Quanta MCP tool run_circuit maintains default statevector method."""
        code = """
@circuit(qubits=2)
def circ(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)
"""
        raw_json = run_circuit(code=code, shots=100, method="statevector")
        data = json.loads(raw_json)
        assert "counts" in data
        assert data["num_qubits"] == 2

    def test_mcp_mps_circuit_simulate_tool(self) -> None:
        """Verifies dedicated mps_circuit_simulate MCP tool."""
        code = """
@circuit(qubits=20)
def circ(q):
    H(q[0])
    for i in range(19):
        CX(q[i], q[i + 1])
    return measure(q)
"""
        raw_json = mps_circuit_simulate(code=code, num_qubits=20, chi_max=16, shots=100)
        data = json.loads(raw_json)

        assert "counts" in data
        assert data["num_qubits"] == 20
        assert data["max_bond_dim"] <= 2
        assert data["truncation_error"] <= 1e-12
        assert data["memory_bytes"] < 500_000
        assert "execution_time_ms" in data
        assert "entanglement_entropy" in data
        assert set(data["counts"].keys()).issubset({"0" * 20, "1" * 20})

    def test_mcp_mps_circuit_simulate_error_handling(self) -> None:
        """Verifies sanitized error reporting on bad code."""
        raw_json = mps_circuit_simulate(code="non_existent_symbol", num_qubits=15)
        data = json.loads(raw_json)
        assert "error" in data
