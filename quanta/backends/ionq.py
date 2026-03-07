"""
quanta.backends.ionq -- IonQ backend via REST API.

Runs Quanta circuits on IonQ trapped-ion quantum hardware.
Uses IonQ's native JSON gate format via their REST API.
No external dependencies -- uses only Python stdlib (urllib).

Setup:
    Export your IonQ API key:
    export IONQ_API_KEY="your-key-here"

    Or get one at: https://cloud.ionq.com/

Example:
    >>> from quanta.backends.ionq import IonQBackend
    >>> backend = IonQBackend(target="simulator")
    >>> result = run(bell, shots=1024, backend=backend)

    >>> backend = IonQBackend(target="qpu.aria-1")
    >>> result = run(bell, shots=1024, backend=backend)

Available targets:
    simulator       Free cloud simulator
    qpu.harmony     11 qubits (trapped ion)
    qpu.aria-1      25 qubits (trapped ion)
    qpu.aria-2      25 qubits (trapped ion)
    qpu.forte       36 qubits (trapped ion)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from quanta.backends.base import Backend
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result

__all__ = ["IonQBackend"]

_IONQ_API = "https://api.ionq.co/v0.3"

# Quanta gate name -> IonQ native gate format
_GATE_MAP: dict[str, str] = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t", "CX": "cnot", "CZ": "cz",
    "SWAP": "swap", "CCX": "ccx",
    "RX": "rx", "RY": "ry", "RZ": "rz",
}


class IonQBackendError(QuantaError):
    """IonQ backend error."""


class IonQBackend(Backend):
    """Runs circuits on IonQ trapped-ion quantum hardware.

    Uses IonQ's REST API with native JSON gate format.
    No external dependencies -- only Python stdlib.

    Args:
        target: IonQ target ("simulator", "qpu.harmony", "qpu.aria-1", etc.).
        api_key: IonQ API key. Falls back to IONQ_API_KEY env var.
        poll_interval: Seconds between status checks for QPU jobs.
    """

    def __init__(
        self,
        target: str = "simulator",
        api_key: str = "",
        poll_interval: float = 2.0,
    ) -> None:
        self._target = target
        self._api_key = api_key or os.environ.get("IONQ_API_KEY", "")
        self._poll_interval = poll_interval

    @property
    def name(self) -> str:
        return f"ionq_{self._target}"

    def _dag_to_ionq_circuit(self, dag: DAGCircuit) -> list[dict[str, Any]]:
        """Converts DAG to IonQ native gate list."""
        gates: list[dict[str, Any]] = []

        for op in dag.op_nodes():
            ionq_name = _GATE_MAP.get(op.gate_name)
            if ionq_name is None:
                raise IonQBackendError(
                    f"Gate '{op.gate_name}' is not supported by IonQ backend. "
                    f"Supported: {list(_GATE_MAP.keys())}"
                )

            gate: dict[str, Any] = {"gate": ionq_name}

            if len(op.qubits) == 1:
                gate["target"] = op.qubits[0]
            elif len(op.qubits) == 2:
                gate["control"] = op.qubits[0]
                gate["target"] = op.qubits[1]
            elif len(op.qubits) == 3:
                gate["controls"] = [op.qubits[0], op.qubits[1]]
                gate["target"] = op.qubits[2]

            if op.params:
                gate["rotation"] = float(op.params[0])

            gates.append(gate)

        return gates

    def _build_job_body(
        self, dag: DAGCircuit, shots: int
    ) -> dict[str, Any]:
        """Builds IonQ API job submission body."""
        return {
            "target": self._target,
            "shots": shots,
            "input": {
                "format": "ionq.circuit.v0",
                "gateset": "qis",
                "qubits": dag.num_qubits,
                "circuit": self._dag_to_ionq_circuit(dag),
            },
        }

    def _api_request(
        self, method: str, path: str, body: dict | None = None
    ) -> dict[str, Any]:
        """Makes an authenticated request to IonQ API."""
        if not self._api_key:
            raise IonQBackendError(
                "IonQ API key is required.\n"
                "Set IONQ_API_KEY environment variable or pass api_key= parameter.\n"
                "Get a key at: https://cloud.ionq.com/"
            )

        url = f"{_IONQ_API}{path}"
        headers = {
            "Authorization": f"apiKey {self._api_key}",
            "Content-Type": "application/json",
        }

        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.readable() else ""
            raise IonQBackendError(
                f"IonQ API error {e.code}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise IonQBackendError(
                f"Cannot reach IonQ API: {e.reason}"
            ) from e

    def execute(
        self,
        dag: DAGCircuit,
        shots: int = 1024,
        seed: int | None = None,
    ) -> Result:
        """Runs circuit on IonQ hardware.

        For simulator target, results are returned immediately.
        For QPU targets, polls until the job completes.

        Args:
            dag: Compiled DAG circuit.
            shots: Number of measurement repetitions.
            seed: Not used (IonQ controls randomness server-side).

        Returns:
            Result with measurement counts.
        """
        job_body = self._build_job_body(dag, shots)
        job = self._api_request("POST", "/jobs", job_body)
        job_id = job["id"]

        # Poll until complete
        while job.get("status") not in ("completed", "failed", "canceled"):
            time.sleep(self._poll_interval)
            job = self._api_request("GET", f"/jobs/{job_id}")

        if job["status"] != "completed":
            raise IonQBackendError(
                f"IonQ job {job_id} ended with status: {job['status']}"
            )

        # Parse results -- IonQ returns probability distribution
        counts = self._parse_results(job, dag.num_qubits, shots)

        return Result(
            counts=counts,
            shots=shots,
            num_qubits=dag.num_qubits,
        )

    @staticmethod
    def _parse_results(
        job: dict[str, Any], num_qubits: int, shots: int
    ) -> dict[str, int]:
        """Converts IonQ probability distribution to measurement counts.

        IonQ returns {"probabilities": {"0": 0.5, "3": 0.5}} where
        keys are decimal state indices. We convert to bitstring counts.
        """
        probs = job.get("data", {}).get("probabilities", {})

        counts: dict[str, int] = {}
        remaining = shots

        # Sort by probability descending for deterministic rounding
        sorted_states = sorted(probs.items(), key=lambda x: -x[1])

        for i, (state_idx, prob) in enumerate(sorted_states):
            bitstring = format(int(state_idx), f"0{num_qubits}b")
            if i == len(sorted_states) - 1:
                # Last state gets remaining shots to ensure exact total
                count = remaining
            else:
                count = round(prob * shots)
                remaining -= count
            if count > 0:
                counts[bitstring] = count

        return counts

    def __repr__(self) -> str:
        return f"IonQBackend(target='{self._target}')"
