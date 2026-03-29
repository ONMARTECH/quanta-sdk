"""
quanta.backends.local — Built-in NumPy-based simulator backend.


  - Max ~25 qubit

Example:
    >>> from quanta.backends.local import LocalSimulator
    >>> backend = LocalSimulator(seed=42)
    >>> result = run(bell, backend=backend)
"""

from __future__ import annotations

from quanta.backends.base import Backend, BackendCapabilities
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result
from quanta.simulator.statevector import StateVectorSimulator

# ── Public API ──
__all__ = ["LocalSimulator"]

class LocalSimulator(Backend):
    """Built-in NumPy statevector simulator.

    Args:
        seed: Default random seed.

    Example:
        >>> sim = LocalSimulator(seed=42)
        >>> result = run(my_circuit, backend=sim, shots=1024)
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "LocalSimulator"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=25,
            native_gates=frozenset({
                "H", "X", "Y", "Z", "S", "T", "SDG", "TDG",
                "CX", "CY", "CZ", "SWAP", "CCX", "CSWAP",
                "RX", "RY", "RZ", "P", "U", "SX",
                "RXX", "RZZ", "ECR", "iSWAP", "CH", "CP", "MS",
            }),
            connectivity="all-to-all",
            supports_noise=True,
            is_simulator=True,
        )

    def is_available(self) -> bool:
        return True

    def execute(
        self,
        dag: DAGCircuit,
        shots: int,
        seed: int | None = None,
    ) -> Result:
        """Runs circuit on local simulator.

        Args:
            seed: Random seed (backend seed'ini override eder).

        Returns:
        """
        effective_seed = seed if seed is not None else self._seed
        simulator = StateVectorSimulator(dag.num_qubits, seed=effective_seed)

        for op in dag.op_nodes():
            simulator.apply(op.gate_name, op.qubits, op.params)

        # Sample
        counts = simulator.sample(shots)

        if dag.measurement and dag.measurement.qubits:
            counts = self._filter_measured(counts, dag.measurement.qubits, dag.num_qubits)

        return Result(
            counts=counts,
            shots=shots,
            num_qubits=dag.num_qubits,
            statevector=simulator.state,
        )

    @staticmethod
    def _filter_measured(
        counts: dict[str, int],
        measured: tuple[int, ...],
        num_qubits: int,
    ) -> dict[str, int]:
        filtered: dict[str, int] = {}
        for bitstring, count in counts.items():
            bits = "".join(bitstring[q] for q in measured)
            filtered[bits] = filtered.get(bits, 0) + count
        return filtered
