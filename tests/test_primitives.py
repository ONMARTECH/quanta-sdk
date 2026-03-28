"""Tests for Estimator/Sampler primitives, @quantum decorator, run_async."""

from __future__ import annotations

import asyncio
import math

import pytest

from quanta import CX, RY, H, circuit, measure, quantum, run_async
from quanta.core.types import QuantaError
from quanta.primitives import Estimator, EstimatorResult, Sampler, SamplerResult

# ── Fixtures ──

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)


@circuit(qubits=1)
def x_gate(q):
    from quanta import X
    X(q[0])
    return measure(q)


# ══════════════════════════════════════════
# Sampler Tests
# ══════════════════════════════════════════


class TestSampler:
    def test_single_circuit(self):
        sampler = Sampler(seed=42)
        result = sampler.run(bell, shots=1024)
        assert isinstance(result, SamplerResult)
        assert len(result.counts) == 1
        assert sum(result.counts[0].values()) == 1024
        # Bell state: only |00⟩ and |11⟩
        assert set(result.counts[0].keys()).issubset({"00", "11"})

    def test_batch_circuits(self):
        sampler = Sampler(seed=42)
        result = sampler.run([bell, x_gate], shots=512)
        assert len(result.counts) == 2
        assert sum(result.counts[0].values()) == 512
        assert sum(result.counts[1].values()) == 512

    def test_quasi_dists(self):
        sampler = Sampler(seed=42)
        result = sampler.run(bell, shots=4096)
        probs = result.quasi_dists[0]
        assert abs(probs.get("00", 0) - 0.5) < 0.05
        assert abs(probs.get("11", 0) - 0.5) < 0.05

    def test_result_shortcut(self):
        sampler = Sampler(seed=42)
        result = sampler.run(bell, shots=1024)
        assert result.result == result.counts[0]

    def test_metadata(self):
        sampler = Sampler(seed=42)
        result = sampler.run(bell, shots=1024)
        assert result.metadata[0]["shots"] == 1024
        assert result.metadata[0]["circuit_name"] == "bell"

    def test_repr(self):
        sampler = Sampler(seed=42)
        result = sampler.run(bell, shots=100)
        assert "SamplerResult" in repr(result)


class TestSamplerAsync:
    def test_async_single(self):
        sampler = Sampler(seed=42)
        result = asyncio.run(sampler.run_async(bell, shots=512))
        assert isinstance(result, SamplerResult)
        assert sum(result.counts[0].values()) == 512

    def test_async_batch(self):
        sampler = Sampler(seed=42)
        result = asyncio.run(sampler.run_async([bell, x_gate], shots=256))
        assert len(result.counts) == 2


# ══════════════════════════════════════════
# Estimator Tests
# ══════════════════════════════════════════


class TestEstimator:
    def test_zz_bell_state(self):
        """Bell state ⟨ZZ⟩ = 1.0 (perfect correlation)."""
        estimator = Estimator(seed=42)
        result = estimator.run(bell, observables=[[("ZZ", 1.0)]])
        assert isinstance(result, EstimatorResult)
        assert abs(result.value - 1.0) < 1e-6

    def test_xx_bell_state(self):
        """Bell state ⟨XX⟩ = 1.0."""
        estimator = Estimator()
        result = estimator.run(bell, observables=[[("XX", 1.0)]])
        assert abs(result.value - 1.0) < 1e-6

    def test_zi_bell_state(self):
        """Bell state ⟨ZI⟩ = 0.0 (no polarization)."""
        estimator = Estimator()
        result = estimator.run(bell, observables=[[("ZI", 1.0)]])
        assert abs(result.value) < 1e-6

    def test_multi_observable(self):
        """Hamiltonian: H = ZZ + 0.5*XX."""
        estimator = Estimator()
        result = estimator.run(
            bell,
            observables=[[("ZZ", 1.0), ("XX", 0.5)]],
        )
        # Bell state: ⟨ZZ⟩=1, ⟨XX⟩=1 → total = 1.5
        assert abs(result.value - 1.5) < 1e-6

    def test_multi_circuit(self):
        """Run multiple circuits with different observables."""
        estimator = Estimator()
        result = estimator.run(
            [bell, bell],
            observables=[
                [("ZZ", 1.0)],
                [("XX", 1.0)],
            ],
        )
        assert len(result.values) == 2
        assert abs(result.values[0] - 1.0) < 1e-6
        assert abs(result.values[1] - 1.0) < 1e-6

    def test_single_circuit_broadcast(self):
        """Single circuit with multiple observables."""
        estimator = Estimator()
        result = estimator.run(
            bell,  # single circuit
            observables=[
                [("ZZ", 1.0)],
                [("XX", 1.0)],
                [("ZI", 1.0)],
            ],
        )
        assert len(result.values) == 3

    def test_variance(self):
        estimator = Estimator()
        result = estimator.run(bell, observables=[[("ZZ", 1.0)]])
        # Pure eigenstate → variance = 0
        assert result.variances[0] < 1e-6

    def test_mismatch_raises(self):
        estimator = Estimator()
        with pytest.raises(QuantaError):
            estimator.run(
                [bell, bell, bell],
                observables=[[("ZZ", 1.0)], [("XX", 1.0)]],
            )

    def test_repr(self):
        estimator = Estimator()
        result = estimator.run(bell, observables=[[("ZZ", 1.0)]])
        assert "EstimatorResult" in repr(result)


class TestEstimatorAsync:
    def test_async_estimator(self):
        estimator = Estimator()
        result = asyncio.run(
            estimator.run_async(
                [bell, bell],
                observables=[
                    [("ZZ", 1.0)],
                    [("XX", 1.0)],
                ],
            )
        )
        assert len(result.values) == 2


# ══════════════════════════════════════════
# @quantum Decorator Tests
# ══════════════════════════════════════════


class TestQuantumDecorator:
    def test_basic_call(self):
        @quantum(qubits=2, seed=42)
        def bell_q(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = bell_q()
        assert sum(result.counts.values()) == 1024
        assert set(result.counts.keys()).issubset({"00", "11"})

    def test_parametric(self):
        @quantum(qubits=1, seed=42)
        def rotation(q, theta=0.0):
            RY(theta)(q[0])
            return measure(q)

        # theta=0 → |0⟩
        r0 = rotation(theta=0.0)
        assert r0.most_frequent == "0"

        # theta=π → |1⟩
        r_pi = rotation(theta=math.pi)
        assert r_pi.most_frequent == "1"

    def test_expectation(self):
        @quantum(qubits=2, observable=[("ZZ", 1.0)])
        def bell_q(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        exp = bell_q.expectation()
        assert abs(exp - 1.0) < 1e-6

    def test_gradient(self):
        @quantum(qubits=1, observable=[("Z", 1.0)])
        def rotation(q, theta=0.0):
            RY(theta)(q[0])
            return measure(q)

        grad = rotation.gradient(theta=0.5)
        assert "theta" in grad.gradients
        assert isinstance(grad.cost, float)
        # d/dθ ⟨Z⟩ = d/dθ cos(θ) = -sin(θ)
        expected = -math.sin(0.5)
        assert abs(grad.gradients["theta"] - expected) < 0.05

    def test_multi_param_gradient(self):
        @quantum(qubits=2, observable=[("ZZ", 1.0)])
        def two_param(q, theta=0.0, phi=0.0):
            RY(theta)(q[0])
            RY(phi)(q[1])
            CX(q[0], q[1])
            return measure(q)

        grad = two_param.gradient(theta=0.5, phi=0.3)
        assert "theta" in grad.gradients
        assert "phi" in grad.gradients

    def test_properties(self):
        @quantum(qubits=3, diff_method="parameter-shift")
        def test_fn(q):
            H(q[0])
            return measure(q)

        assert test_fn.num_qubits == 3
        assert test_fn.diff_method == "parameter-shift"

    def test_repr(self):
        @quantum(qubits=2)
        def my_circuit(q):
            H(q[0])
            return measure(q)

        assert "QuantumFunction" in repr(my_circuit)
        assert "my_circuit" in repr(my_circuit)


# ══════════════════════════════════════════
# run_async Tests
# ══════════════════════════════════════════


class TestRunAsync:
    def test_single_circuit(self):
        results = asyncio.run(run_async(bell, shots=512))
        assert len(results) == 1
        assert sum(results[0].counts.values()) == 512

    def test_batch(self):
        results = asyncio.run(run_async([bell, x_gate], shots=256))
        assert len(results) == 2
        assert results[0].most_frequent in ("00", "11")
        assert results[1].most_frequent == "1"

    def test_preserves_seed(self):
        r1 = asyncio.run(run_async(bell, shots=100, seed=42))
        r2 = asyncio.run(run_async(bell, shots=100, seed=42))
        assert r1[0].counts == r2[0].counts
