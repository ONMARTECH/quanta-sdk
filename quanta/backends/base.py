"""
quanta.backends.base — Backend abstract interface.

All quantum execution targets implement this interface:
  - LocalSimulator: Built-in NumPy statevector
  - IBMQuantum: IBM Quantum via Qiskit Runtime
  - IonQBackend: IonQ via REST API
  - GoogleQuantum: Google Quantum Engine

Example:
    >>> from quanta.backends.base import Backend
    >>> backend = Backend.from_name("local")
    >>> print(backend.capabilities())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quanta.dag.dag_circuit import DAGCircuit
    from quanta.result import Result

# ── Public API ──
__all__ = ["Backend", "BackendCapabilities"]


@dataclass(frozen=True)
class BackendCapabilities:
    """Describes what a backend can do.

    Attributes:
        max_qubits: Maximum supported qubit count.
        native_gates: Set of natively supported gate names.
        connectivity: "all-to-all", "linear", "grid", or "custom".
        supports_noise: Whether noise simulation is available.
        is_simulator: True for simulators, False for real hardware.
    """

    max_qubits: int = 25
    native_gates: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "H", "X", "Y", "Z", "S", "T", "CX", "RX", "RY", "RZ",
        }),
    )
    connectivity: str = "all-to-all"
    supports_noise: bool = False
    is_simulator: bool = True

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"  Max qubits:    {self.max_qubits}",
            f"  Native gates:  {len(self.native_gates)}",
            f"  Connectivity:  {self.connectivity}",
            f"  Noise support: {'Yes' if self.supports_noise else 'No'}",
            f"  Type:          {'Simulator' if self.is_simulator else 'Hardware'}",
        ]
        return "\n".join(lines)


class Backend(ABC):
    """Abstract interface for quantum execution backends.

    All backends must implement:
      - name: Human-readable identifier
      - execute(): Run a circuit
      - capabilities(): Describe supported features

    Example:
        >>> backend = Backend.from_name("local")
        >>> caps = backend.capabilities()
        >>> print(f"{backend.name}: {caps.max_qubits} qubits")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @abstractmethod
    def execute(
        self,
        dag: DAGCircuit,
        shots: int,
        seed: int | None = None,
    ) -> Result:
        """Runs circuit on this backend.

        Args:
            dag: Circuit in DAG form.
            shots: Number of measurement samples.
            seed: Random seed.

        Returns:
            Result with counts and statevector.
        """
        ...

    def capabilities(self) -> BackendCapabilities:
        """Returns what this backend supports.

        Override in subclasses for accurate specs.
        """
        return BackendCapabilities()

    def is_available(self) -> bool:
        """Check if backend is currently usable.

        Returns True for local simulators, checks connectivity
        for cloud backends.
        """
        return True

    # ── Factory ──

    @staticmethod
    def from_name(name: str, **kwargs) -> Backend:
        """Create a backend instance by name.

        Args:
            name: Backend identifier. Options:
                - "local": Built-in NumPy simulator
                - "ibm" / "ibm_fez": IBM Quantum
                - "ionq" / "ionq_aria": IonQ Aria
                - "google" / "google_willow": Google Quantum

        Returns:
            Backend instance.

        Example:
            >>> backend = Backend.from_name("local", seed=42)
        """
        name_lower = name.lower().replace("-", "_")

        if name_lower in ("local", "local_simulator", "numpy"):
            from quanta.backends.local import LocalSimulator
            return LocalSimulator(**kwargs)

        if name_lower in ("ibm", "ibm_fez", "ibm_heron", "ibm_quantum"):
            from quanta.backends.ibm import IBMQuantumBackend
            return IBMQuantumBackend(**kwargs)

        if name_lower in ("ionq", "ionq_aria"):
            from quanta.backends.ionq import IonQBackend
            return IonQBackend(**kwargs)

        if name_lower in ("google", "google_willow", "gcp"):
            from quanta.backends.google import GoogleQuantumBackend
            return GoogleQuantumBackend(**kwargs)

        available = ["local", "ibm", "ionq", "google"]
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(available)}"
        )

    @staticmethod
    def list_available() -> list[str]:
        """List all registered backend names."""
        return ["local", "ibm", "ionq", "google"]

    def __repr__(self) -> str:
        return f"Backend({self.name})"

