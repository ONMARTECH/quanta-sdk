"""
quanta.backends.base — Backend abstract interface.


Available backends:
  - (planned) GCPQuantum: Google Quantum Engine
  - (planned) IBMQuantum: IBM Quantum via Qiskit

Example:
    >>> from quanta.backends.local import LocalSimulator
    >>> result = run(bell, backend=LocalSimulator(seed=42))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quanta.dag.dag_circuit import DAGCircuit
    from quanta.result import Result

# ── Public API ──
__all__ = ["Backend"]

class Backend(ABC):
    """Abstract interface for quantum execution backends.


        - name: Backend name (property)
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
            seed: Random seed.

        Returns:
        """
        ...

    def __repr__(self) -> str:
        return f"Backend({self.name})"
