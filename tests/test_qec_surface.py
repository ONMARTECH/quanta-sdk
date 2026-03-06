"""
tests/test_qec_surface.py — Surface code tests.

Validates:
  - Surface code parameters ([[n,k,d]])
  - Error correction simulation
  - Threshold behavior
"""

import pytest
import numpy as np

from quanta.qec.surface_code import SurfaceCode


class TestSurfaceCode:
    """Surface code error correction tests."""

    def test_distance_3_params(self):
        """[[9,1,3]]: 9 physical qubits, 1 logical, distance 3."""
        code = SurfaceCode(distance=3)
        assert code.n_physical == 9
        assert code.n_logical == 1
        assert code.correctable_errors == 1
        assert "[[9, 1, 3]]" in code.code_params

    def test_distance_5_params(self):
        """[[25,1,5]]: 25 physical qubits, corrects 2 errors."""
        code = SurfaceCode(distance=5)
        assert code.n_physical == 25
        assert code.correctable_errors == 2

    def test_distance_7_params(self):
        code = SurfaceCode(distance=7)
        assert code.n_physical == 49
        assert code.correctable_errors == 3

    def test_invalid_distance_raises(self):
        """Even or < 3 distance should raise error."""
        with pytest.raises(ValueError):
            SurfaceCode(distance=2)
        with pytest.raises(ValueError):
            SurfaceCode(distance=4)

    def test_error_correction_below_threshold(self):
        """Below threshold: logical error rate < physical error rate."""
        code = SurfaceCode(distance=5)
        result = code.simulate_error_correction(
            error_rate=0.001, rounds=2000, seed=42
        )
        assert result.logical_error_rate <= result.physical_error_rate

    def test_error_correction_rounds(self):
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(rounds=500, seed=42)
        assert result.rounds == 500

    def test_summary_string(self):
        code = SurfaceCode(distance=3)
        s = code.summary()
        assert "Surface Code" in s
        assert "Physical qubits" in s

    def test_repr(self):
        code = SurfaceCode(distance=3)
        assert "SurfaceCode" in repr(code)

    def test_correction_result_summary(self):
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(rounds=100, seed=42)
        assert "Surface Code" in result.summary()

    def test_zero_error_rate(self):
        """Zero errors → zero logical errors."""
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(error_rate=0.0, rounds=100, seed=42)
        assert result.logical_error_rate == 0
        assert result.errors_injected == 0
