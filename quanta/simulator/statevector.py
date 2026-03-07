"""
quanta.simulator.statevector -- NumPy-based statevector simulator.

v0.2: Rewritten with tensor contraction method.

OLD (v0.1): Kronecker expansion O(4^n) -- max 12 qubits, 1.8s
NEW (v0.2): Tensor contraction O(2^n) -- max 26 qubits, <20s

Method:
  Keep statevector as [2,2,...,2] tensor.
  Apply gate only to relevant axes (np.tensordot).

Example:
    >>> sim = StateVectorSimulator(num_qubits=20)
    >>> sim.apply("H", (0,))
    >>> sim.apply("CX", (0, 1))
    >>> sim.probabilities()[:4]
    array([0.5, 0. , 0. , 0.5])
"""

from __future__ import annotations

import numpy as np

from quanta.core.gates import GATE_REGISTRY, ParametricGate
from quanta.core.types import QuantaError

# -- Public API --
__all__ = ["StateVectorSimulator"]


class SimulatorError(QuantaError):
    """Simulator runtime error."""


class StateVectorSimulator:
    """Tensor-based statevector simulator.

    Simulates quantum circuits on a 2^n dimensional complex vector.
    v0.2: O(2^n) performance with np.tensordot -- supports 26 qubits.

    Args:
        num_qubits: Number of qubits to simulate.
        seed: Random seed for reproducibility.
    """

    MAX_QUBITS = 27  # Memory limit: ~2 GB

    __slots__ = ("num_qubits", "_state", "_rng")

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        if num_qubits > self.MAX_QUBITS:
            raise SimulatorError(
                f"Max {self.MAX_QUBITS} qubits supported, "
                f"requested: {num_qubits}"
            )

        self.num_qubits = num_qubits
        self._rng = np.random.default_rng(seed)

        # Initial state: |00...0>
        dim = 2 ** num_qubits
        self._state = np.zeros(dim, dtype=complex)
        self._state[0] = 1.0

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a gate to the statevector via tensor contraction.

        Args:
            gate_name: Gate name (e.g., "H", "CX").
            qubits: Target qubit indices.
            params: Angles for parametric gates.
        """
        matrix = self._get_gate_matrix(gate_name, params)
        self._apply_tensor(matrix, qubits)

    def _apply_tensor(
        self, gate: np.ndarray, qubits: tuple[int, ...]
    ) -> None:
        """Gate application via tensor contraction.

        Uses JAX/CuPy acceleration if available, falls back to NumPy.
        """
        try:
            from quanta.simulator.accelerated import tensor_contract
            self._state = tensor_contract(
                gate, self._state, qubits, self.num_qubits
            )
            return
        except ImportError:
            pass

        # NumPy fallback (always works)
        n = self.num_qubits
        num_gate_qubits = len(qubits)

        state_tensor = self._state.reshape([2] * n)
        gate_tensor = gate.reshape([2] * (2 * num_gate_qubits))

        gate_axes = list(range(num_gate_qubits, 2 * num_gate_qubits))
        state_axes = list(qubits)

        result = np.tensordot(gate_tensor, state_tensor, axes=(gate_axes, state_axes))
        result = np.moveaxis(result, list(range(num_gate_qubits)), list(qubits))

        self._state = result.reshape(-1)

    def _get_gate_matrix(
        self, name: str, params: tuple[float, ...]
    ) -> np.ndarray:
        """Gets gate matrix from name."""
        gate = GATE_REGISTRY.get(name)

        if gate is None:
            # Might be a parametric gate (RX, RY, RZ)
            from quanta.core import gates as gates_module
            parametric = getattr(gates_module, name, None)
            if parametric and isinstance(parametric, ParametricGate):
                if not params:
                    raise SimulatorError(f"{name} gate requires parameters")
                return parametric(params[0]).matrix
            raise SimulatorError(f"Unknown gate: {name}")

        return gate.matrix

    def probabilities(self) -> np.ndarray:
        """Returns measurement probability of each state: P = |a|^2."""
        return np.abs(self._state) ** 2

    def sample(self, shots: int) -> dict[str, int]:
        """Performs measurement sampling based on probabilities.

        Args:
            shots: Number of samples.

        Returns:
            dict: Result -> count. E.g. {'00': 512, '11': 512}
        """
        probs = self.probabilities()
        dim = len(probs)
        n = self.num_qubits

        # Sample
        indices = self._rng.choice(dim, size=shots, p=probs)

        # Count
        counts: dict[str, int] = {}
        for idx in indices:
            bitstring = format(idx, f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    @property
    def state(self) -> np.ndarray:
        """Copy of the current statevector."""
        return self._state.copy()

    @state.setter
    def state(self, new_state: np.ndarray) -> None:
        """Sets the statevector. Used by Layer 3 algorithms (Grover, etc.)."""
        if len(new_state) != 2 ** self.num_qubits:
            raise SimulatorError(
                f"State dimension mismatch: expected {2 ** self.num_qubits}, "
                f"got {len(new_state)}"
            )
        self._state = np.asarray(new_state, dtype=complex)

    def apply_phase(self, index: int, phase: complex) -> None:
        """Applies a phase factor to a specific basis state.

        Used by Grover oracle and other state-manipulation algorithms.

        Args:
            index: Basis state index (0 to 2^n - 1).
            phase: Phase factor to multiply (e.g., -1 for phase flip).
        """
        self._state[index] *= phase
