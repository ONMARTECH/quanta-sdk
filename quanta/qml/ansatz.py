"""
quanta.qml.ansatz -- Variational ansatz presets for QML.

Provides ready-to-use parameterized circuit templates:
  - HardwareEfficient: RY-RZ layers + CX entanglement (fastest on real hardware)
  - StronglyEntangling: Full rotation + all-to-all entanglement (most expressive)
  - Reuploading: Data re-encoding between layers (best for small datasets)

Example:
    >>> from quanta.qml.ansatz import Ansatz
    >>> Ansatz.list_available()
    ['hardware_efficient', 'strongly_entangling', 'reuploading']
    >>> hw = Ansatz.get("hardware_efficient")
    >>> hw.param_count(n_qubits=4, n_layers=2)
    16
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanta.simulator.statevector import StateVectorSimulator

__all__ = ["Ansatz", "AnsatzPreset"]


@dataclass(frozen=True)
class AnsatzPreset:
    """Describes a variational ansatz template.

    Attributes:
        name: Short identifier (e.g. "hardware_efficient").
        description: Human-readable explanation.
        params_per_layer: Function(n_qubits) → params per layer.
    """

    name: str
    description: str
    params_per_qubit_per_layer: int

    def param_count(self, n_qubits: int, n_layers: int) -> int:
        """Total trainable parameters."""
        return self.params_per_qubit_per_layer * n_qubits * n_layers

    def apply(
        self,
        sim: StateVectorSimulator,
        params: np.ndarray,
        n_qubits: int,
        n_layers: int,
    ) -> None:
        """Apply the ansatz to a simulator.

        Args:
            sim: StateVectorSimulator instance.
            params: Flat parameter array of length param_count().
            n_qubits: Number of qubits.
            n_layers: Number of variational layers.
        """
        fn = _ANSATZ_FUNCTIONS[self.name]
        fn(sim, params, n_qubits, n_layers)

    def __repr__(self) -> str:
        return f"AnsatzPreset('{self.name}')"


# ── Ansatz implementations ──


def _hardware_efficient(
    sim: StateVectorSimulator,
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> None:
    """Hardware-efficient ansatz: RY + RZ per qubit + linear CX chain.

    Most hardware-native: uses only nearest-neighbor entanglement.
    Params per layer: 2 * n_qubits (RY + RZ).
    """
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            sim.apply("RY", (q,), (float(params[idx]),))
            idx += 1
            sim.apply("RZ", (q,), (float(params[idx]),))
            idx += 1
        for q in range(n_qubits - 1):
            sim.apply("CX", (q, q + 1))


def _strongly_entangling(
    sim: StateVectorSimulator,
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> None:
    """Strongly-entangling ansatz: RX + RY + RZ per qubit + shifted CX.

    Higher expressibility through full rotation and varying entanglement
    patterns across layers.
    Params per layer: 3 * n_qubits (RX + RY + RZ).
    """
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            sim.apply("RX", (q,), (float(params[idx]),))
            idx += 1
            sim.apply("RY", (q,), (float(params[idx]),))
            idx += 1
            sim.apply("RZ", (q,), (float(params[idx]),))
            idx += 1
        # Entanglement with shifted pattern per layer
        shift = layer % max(1, n_qubits - 1)
        for q in range(n_qubits):
            target = (q + shift + 1) % n_qubits
            if target != q:
                sim.apply("CX", (q, target))


def _reuploading(
    sim: StateVectorSimulator,
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> None:
    """Data-reuploading ansatz: RY rotation + CX per layer.

    Designed for small datasets where data is re-encoded into each layer.
    Params per layer: 1 * n_qubits (single RY rotation).
    """
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            sim.apply("RY", (q,), (float(params[idx]),))
            idx += 1
        for q in range(n_qubits - 1):
            sim.apply("CX", (q, q + 1))


# ── Registry ──

_PRESETS: dict[str, AnsatzPreset] = {
    "hardware_efficient": AnsatzPreset(
        name="hardware_efficient",
        description=(
            "Hardware-efficient ansatz with RY+RZ rotations and "
            "linear CX entanglement. Fastest on real quantum hardware."
        ),
        params_per_qubit_per_layer=2,
    ),
    "strongly_entangling": AnsatzPreset(
        name="strongly_entangling",
        description=(
            "Strongly-entangling ansatz with full RX+RY+RZ rotations "
            "and shifted CX pattern. Highest expressibility."
        ),
        params_per_qubit_per_layer=3,
    ),
    "reuploading": AnsatzPreset(
        name="reuploading",
        description=(
            "Data-reuploading ansatz with single RY rotation per qubit. "
            "Best for small datasets with data re-encoding."
        ),
        params_per_qubit_per_layer=1,
    ),
}

_ANSATZ_FUNCTIONS = {
    "hardware_efficient": _hardware_efficient,
    "strongly_entangling": _strongly_entangling,
    "reuploading": _reuploading,
}


class Ansatz:
    """Ansatz discovery and access.

    Example:
        >>> Ansatz.list_available()
        ['hardware_efficient', 'strongly_entangling', 'reuploading']
        >>> preset = Ansatz.get("hardware_efficient")
        >>> preset.param_count(n_qubits=4, n_layers=2)
        16
    """

    @classmethod
    def list_available(cls) -> list[str]:
        """Returns list of available ansatz names."""
        return list(_PRESETS.keys())

    @classmethod
    def get(cls, name: str) -> AnsatzPreset:
        """Returns ansatz preset by name.

        Args:
            name: Ansatz name.

        Raises:
            ValueError: If name is not recognized.
        """
        if name not in _PRESETS:
            available = ", ".join(_PRESETS.keys())
            raise ValueError(
                f"Unknown ansatz '{name}'. Available: {available}"
            )
        return _PRESETS[name]

    @classmethod
    def describe(cls, name: str) -> str:
        """Returns human-readable description."""
        return cls.get(name).description
