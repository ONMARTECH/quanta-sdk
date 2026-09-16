"""Tests for quanta.simulator.router."""


from quanta.simulator.mps import MPSSimulator
from quanta.simulator.pauli_frame import PauliFrameSimulator
from quanta.simulator.router import analyze_circuit, select_simulator
from quanta.simulator.sparse import SparseSimulator
from quanta.simulator.statevector import StateVectorSimulator


def test_analyze_circuit_defaults():
    res = analyze_circuit()
    assert res["is_clifford"] is True
    assert res["gate_count"] == 0
    assert res["has_parametric"] is False
    assert res["recommended_method"] == "dense"


def test_analyze_circuit_clifford_large():
    res = analyze_circuit(gate_names=["H", "CX", "S", "CZ"], num_qubits=100)
    assert res["is_clifford"] is True
    assert res["recommended_method"] == "clifford"


def test_analyze_circuit_sparse():
    res = analyze_circuit(gate_names=["H", "RX", "CX"], num_qubits=40)
    assert res["is_clifford"] is False
    assert res["has_parametric"] is True
    assert res["recommended_method"] == "sparse"


def test_analyze_circuit_mps():
    res = analyze_circuit(gate_names=["H", "RY", "CX"], num_qubits=100)
    assert res["is_clifford"] is False
    assert res["recommended_method"] == "mps"


def test_select_simulator_instances():
    # Dense
    sim_dense = select_simulator(num_qubits=5, gate_names=["H", "CX"])
    assert isinstance(sim_dense, StateVectorSimulator)

    # Clifford
    sim_cliff = select_simulator(num_qubits=50, gate_names=["H", "CX", "S"])
    assert isinstance(sim_cliff, PauliFrameSimulator)

    # Sparse
    sim_sparse = select_simulator(num_qubits=40, gate_names=["H", "T", "CX"])
    assert isinstance(sim_sparse, SparseSimulator)

    # MPS
    sim_mps = select_simulator(num_qubits=80, gate_names=["H", "T", "CX"], chi_max=32)
    assert isinstance(sim_mps, MPSSimulator)
    assert sim_mps.chi_max == 32


def test_select_simulator_dense_mlx():
    from quanta.simulator.mlx import MLXSimulator, is_mlx_available

    sim = select_simulator(num_qubits=22, gate_names=["H", "T", "CX"])
    if is_mlx_available():
        assert isinstance(sim, MLXSimulator)
    else:
        assert isinstance(sim, StateVectorSimulator)

