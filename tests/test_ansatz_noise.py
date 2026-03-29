"""
test_ansatz_noise — Tests for Task 8 (Ansatz) and Task 14 (Noise Builder).
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")


# ═══════════════════════════════════════════
# Task 8: Ansatz presets
# ═══════════════════════════════════════════


class TestAnsatz:
    """Tests for quanta.qml.ansatz module."""

    def test_list_available(self):
        from quanta.qml import Ansatz
        available = Ansatz.list_available()
        assert "hardware_efficient" in available
        assert "strongly_entangling" in available
        assert "reuploading" in available

    def test_get_valid(self):
        from quanta.qml import Ansatz, AnsatzPreset
        preset = Ansatz.get("hardware_efficient")
        assert isinstance(preset, AnsatzPreset)

    def test_get_invalid(self):
        from quanta.qml import Ansatz
        with pytest.raises(ValueError, match="nonexistent"):
            Ansatz.get("nonexistent")

    def test_describe(self):
        from quanta.qml import Ansatz
        desc = Ansatz.describe("hardware_efficient")
        assert "hardware" in desc.lower()

    def test_param_count_hardware_efficient(self):
        from quanta.qml import Ansatz
        hw = Ansatz.get("hardware_efficient")
        assert hw.param_count(4, 2) == 16  # 2 params/qubit * 4 qubits * 2 layers

    def test_param_count_strongly_entangling(self):
        from quanta.qml import Ansatz
        se = Ansatz.get("strongly_entangling")
        assert se.param_count(4, 2) == 24  # 3 params/qubit * 4 qubits * 2 layers

    def test_param_count_reuploading(self):
        from quanta.qml import Ansatz
        ru = Ansatz.get("reuploading")
        assert ru.param_count(4, 2) == 8   # 1 param/qubit * 4 qubits * 2 layers

    def test_apply_hardware_efficient(self):
        from quanta.qml import Ansatz
        from quanta.simulator.statevector import StateVectorSimulator
        hw = Ansatz.get("hardware_efficient")
        sim = StateVectorSimulator(2)
        params = np.random.default_rng(42).random(hw.param_count(2, 1))
        hw.apply(sim, params, n_qubits=2, n_layers=1)
        # State should be normalized
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10

    def test_apply_strongly_entangling(self):
        from quanta.qml import Ansatz
        from quanta.simulator.statevector import StateVectorSimulator
        se = Ansatz.get("strongly_entangling")
        sim = StateVectorSimulator(3)
        params = np.random.default_rng(42).random(se.param_count(3, 2))
        se.apply(sim, params, n_qubits=3, n_layers=2)
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10

    def test_apply_reuploading(self):
        from quanta.qml import Ansatz
        from quanta.simulator.statevector import StateVectorSimulator
        ru = Ansatz.get("reuploading")
        sim = StateVectorSimulator(2)
        params = np.random.default_rng(42).random(ru.param_count(2, 3))
        ru.apply(sim, params, n_qubits=2, n_layers=3)
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10

    def test_repr(self):
        from quanta.qml import Ansatz
        r = repr(Ansatz.get("reuploading"))
        assert "reuploading" in r

    def test_classifier_with_ansatz(self):
        from quanta.qml import Classifier
        clf = Classifier(n_qubits=2, ansatz="strongly_entangling", seed=42)
        assert clf.ansatz == "strongly_entangling"
        assert clf.get_params()["ansatz"] == "strongly_entangling"

    def test_classifier_invalid_ansatz(self):
        from quanta.qml import Classifier
        with pytest.raises(ValueError, match="bad_ansatz"):
            Classifier(ansatz="bad_ansatz")


# ═══════════════════════════════════════════
# Task 14: Noise Model Builder + Presets
# ═══════════════════════════════════════════


class TestNoiseBuilder:
    """Tests for NoiseModel builder pattern."""

    def test_builder_basic(self):
        from quanta.simulator.noise import NoiseModel
        model = NoiseModel.builder().depolarizing(0.01).build()
        assert len(model.channels) == 1

    def test_builder_chaining(self):
        from quanta.simulator.noise import NoiseModel
        model = (
            NoiseModel.builder()
            .depolarizing(0.01)
            .with_bit_flip(0.005)
            .with_phase_flip(0.003)
            .with_amplitude_damping(0.002)
            .with_t2(0.004)
            .with_crosstalk(0.001)
            .with_readout(0.01, 0.02)
            .build()
        )
        assert len(model.channels) == 7

    def test_describe(self):
        from quanta.simulator.noise import NoiseModel
        model = NoiseModel.builder().depolarizing(0.01).build()
        desc = model.describe()
        assert "Noise Model" in desc
        assert "Depolarizing" in desc

    def test_ibm_heron(self):
        from quanta.simulator.noise import NoiseModel
        ibm = NoiseModel.ibm_heron()
        assert len(ibm.channels) == 4

    def test_ionq_aria(self):
        from quanta.simulator.noise import NoiseModel
        ionq = NoiseModel.ionq_aria()
        assert len(ionq.channels) == 2

    def test_google_willow(self):
        from quanta.simulator.noise import NoiseModel
        gw = NoiseModel.google_willow()
        assert len(gw.channels) == 4

    def test_preset_channels_correct_types(self):
        from quanta.simulator.noise import (
            AmplitudeDamping,
            Depolarizing,
            NoiseModel,
            ReadoutError,
            T2Relaxation,
        )
        ibm = NoiseModel.ibm_heron()
        types = [type(ch) for ch in ibm.channels]
        assert Depolarizing in types
        assert AmplitudeDamping in types
        assert T2Relaxation in types
        assert ReadoutError in types
