"""
quanta.simulator.custatevec -- NVIDIA cuQuantum GPU statevector simulator.

Leverages NVIDIA cuStateVec (cuQuantum SDK) for extreme GPU acceleration.
Requires CUDA environment with `cuquantum-python` and `cupy`.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from quanta.core.gates import GATE_REGISTRY, MultiParametricGate, ParametricGate
from quanta.core.types import QuantaError
from quanta.simulator.base import SimulatorBackend

try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    HAS_CUQUANTUM = True
except ImportError:
    HAS_CUQUANTUM = False


class SimulatorError(QuantaError):
    """Simulator runtime error related to GPU/cuQuantum."""
    pass


class CuStateVecSimulator(SimulatorBackend):
    """GPU-accelerated statevector simulator using NVIDIA cuQuantum.

    Simulates quantum circuits natively on GPU memory using CuPy
    and cuStateVec explicit bindings. Can handle very deep circuits
    efficiently up to the VRAM limits.

    Args:
        num_qubits: Number of qubits to simulate.
        seed: Random seed for reproducibility.
    """

    MAX_QUBITS = 32  # Limited by GPU memory (e.g., 30 qubits requires ~16GB VRAM)

    __slots__ = ("num_qubits", "_state_cp", "_rng", "_handle", "_workspace")

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        if not HAS_CUQUANTUM:
            raise ImportError(
                "cuQuantum or CuPy is not installed/reachable. "
                "Ensure you have a CUDA-enabled GPU and have installed quanta-sdk[gpu]."
            )

        if num_qubits > self.MAX_QUBITS:
            raise SimulatorError(
                f"Max {self.MAX_QUBITS} qubits supported on GPU, "
                f"requested: {num_qubits}"
            )

        self.num_qubits = num_qubits
        self._rng = np.random.default_rng(seed)

        # Initial state: |00...0> stored on GPU
        dim = 2 ** num_qubits
        self._state_cp = cp.zeros(dim, dtype=cp.complex128)
        self._state_cp[0] = 1.0

        # Initialize cuStateVec handle
        self._handle = cusv.create()
        self._workspace = None

    def __del__(self) -> None:
        """Safely clean up cuStateVec handle from GPU."""
        if HAS_CUQUANTUM and hasattr(self, "_handle") and self._handle is not None:
            with contextlib.suppress(Exception):
                cusv.destroy(self._handle)

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a gate using cuStateVec tensor manipulation.

        Args:
            gate_name: Gate name (e.g., "H", "CX").
            qubits: Target qubit indices.
            params: Angles for parametric gates.
        """
        gate_matrix = self._get_gate_matrix(gate_name, params)
        gate_cp = cp.asarray(gate_matrix, dtype=cp.complex128, order="C")

        targets = list(qubits)
        n_targets = len(targets)

        # Calculate required workspace size
        workspace_size = cusv.apply_matrix_get_workspace_size(
            self._handle,
            cusv.cudaDataType.CUDA_C_64F,
            self.num_qubits,
            gate_cp.data.ptr,
            cusv.cudaDataType.CUDA_C_64F,
            cusv.MatrixLayout.ROW,
            0,  # adjoint
            n_targets,
            targets,
            0,  # n_controls
            [],  # controls
            [],  # control_bit_values
            cusv.ComputeType.COMPUTE_64F,
        )

        # Re-allocate workspace only if it needs to grow
        if self._workspace is None or self._workspace.size < workspace_size:
            self._workspace = cp.cuda.alloc(workspace_size)

        workspace_ptr = self._workspace.ptr if self._workspace is not None else 0

        # Apply matrix using low-level GPU acceleration
        cusv.apply_matrix(
            self._handle,
            self._state_cp.data.ptr,
            cusv.cudaDataType.CUDA_C_64F,
            self.num_qubits,
            gate_cp.data.ptr,
            cusv.cudaDataType.CUDA_C_64F,
            cusv.MatrixLayout.ROW,
            0,
            n_targets,
            targets,
            0,
            [],
            [],
            cusv.ComputeType.COMPUTE_64F,
            workspace_ptr,
            workspace_size,
        )

    def _get_gate_matrix(
        self, name: str, params: tuple[float, ...]
    ) -> np.ndarray:
        """Gets gate matrix from registry."""
        gate = GATE_REGISTRY.get(name)

        if gate is None:
            raise SimulatorError(f"Unknown gate: {name}")

        if isinstance(gate, MultiParametricGate):
            if not params:
                raise SimulatorError(f"{name} gate requires parameters")
            return gate(*params).matrix
        if isinstance(gate, ParametricGate):
            if not params:
                raise SimulatorError(f"{name} gate requires parameters")
            return gate(params[0]).matrix

        return gate.matrix

    def probabilities(self) -> np.ndarray:
        """Returns measurement probability of each state: P = |a|^2.

        Fetches the state back to CPU host for probability calculation.
        """
        # compute magnitude globally on GPU, then transfer back to CPU memory
        probs_cp = cp.abs(self._state_cp) ** 2
        return np.asarray(cp.asnumpy(probs_cp))

    def sample(self, shots: int) -> dict[str, int]:
        """Performs measurement sampling on GPU probabilities."""
        probs = self.probabilities()
        dim = len(probs)
        n = self.num_qubits

        indices = self._rng.choice(dim, size=shots, p=probs)
        unique, unique_counts = np.unique(indices, return_counts=True)

        fmt = f"0{n}b"
        return {format(idx, fmt): int(cnt) for idx, cnt in zip(unique, unique_counts, strict=True)}

    @property
    def state(self) -> np.ndarray:
        """Copy of the current statevector pulled to CPU."""
        return np.asarray(cp.asnumpy(self._state_cp).copy())

    @state.setter
    def state(self, new_state: np.ndarray) -> None:
        """Sets the statevector, automatically moving it to GPU memory."""
        if len(new_state) != 2 ** self.num_qubits:
            raise SimulatorError(
                f"State dimension mismatch: expected {2 ** self.num_qubits}, "
                f"got {len(new_state)}"
            )
        self._state_cp = cp.asarray(new_state, dtype=cp.complex128)

    def apply_phase(self, index: int, phase: complex) -> None:
        """Applies a phase factor. Fired natively on GPU device."""
        self._state_cp[index] *= phase

    def apply_noise(
        self,
        noise_model: Any,
        qubits: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        """Translates state back to CPU briefly for noise, returning it to GPU."""
        state_cpu = cp.asnumpy(self._state_cp)
        state_cpu = noise_model.apply_noise(
            state_cpu, qubits, self.num_qubits, rng,
        )
        self._state_cp = cp.asarray(state_cpu, dtype=cp.complex128)

