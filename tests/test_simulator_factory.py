"""Tests for quanta.simulator.factory — Simulator creation and selection."""

import pytest

from quanta.simulator.factory import METHODS, create_simulator
from quanta.simulator.mps import MPSSimulator
from quanta.simulator.pauli_frame import PauliFrameSimulator
from quanta.simulator.sparse import SparseSimulator
from quanta.simulator.statevector import StateVectorSimulator


def test_factory_methods_list():
    assert "auto" in METHODS
    assert "dense" in METHODS
    assert "sparse" in METHODS
    assert "mps" in METHODS
    assert "clifford" in METHODS
    assert "custatevec" in METHODS


def test_create_dense():
    sim = create_simulator(num_qubits=2, method="dense", seed=42)
    assert isinstance(sim, StateVectorSimulator)
    assert sim.num_qubits == 2


def test_create_sparse():
    sim = create_simulator(num_qubits=4, method="sparse", seed=42)
    assert isinstance(sim, SparseSimulator)
    assert sim.num_qubits == 4


def test_create_mps():
    sim = create_simulator(num_qubits=5, method="mps", seed=42)
    assert isinstance(sim, MPSSimulator)
    assert sim.num_qubits == 5


def test_create_clifford():
    sim = create_simulator(num_qubits=3, method="clifford")
    assert isinstance(sim, PauliFrameSimulator)
    assert sim.num_qubits == 3


def test_create_auto():
    # Small qubit count selects dense
    sim_small = create_simulator(num_qubits=5, method="auto")
    assert isinstance(sim_small, StateVectorSimulator)

    # Medium qubit count selects sparse
    sim_med = create_simulator(num_qubits=30, method="auto")
    assert isinstance(sim_med, SparseSimulator)

    # Large qubit count selects MPS
    sim_large = create_simulator(num_qubits=60, method="auto")
    assert isinstance(sim_large, MPSSimulator)


def test_create_unknown_method():
    with pytest.raises(ValueError, match="Unknown simulator method"):
        create_simulator(num_qubits=2, method="quantum_computer")


def test_create_custatevec_without_gpu():
    with pytest.raises(ValueError, match="cuQuantum"):
        create_simulator(num_qubits=2, method="custatevec")
