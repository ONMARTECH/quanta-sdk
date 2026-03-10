"""
quanta.backends.ibm_rest — Direct REST API backend for IBM Quantum.

Runs Quanta circuits on IBM Quantum hardware via direct HTTP calls.
No Qiskit dependency required — only uses `requests` (or urllib).

Authentication flow:
  1. Exchange IBM Cloud API key for IAM bearer token
  2. Include bearer token + instance CRN in every request
  3. Submit QASM 3.0 circuits as sampler/estimator jobs
  4. Poll for results

Setup:
  export IBM_API_KEY="your-ibm-cloud-api-key"
  export IBM_INSTANCE_CRN="crn:v1:bluemix:public:quantum-computing:..."

  Or pass them as constructor arguments.

Example:
    >>> from quanta.backends.ibm_rest import IBMRestBackend
    >>> backend = IBMRestBackend()
    >>> print(backend.list_backends())
    >>> result = backend.execute(dag, shots=1024)

    # Using sessions:
    >>> with backend.session() as session:
    ...     r1 = session.run(dag1)
    ...     r2 = session.run(dag2)
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from quanta.backends.base import Backend
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result

__all__ = ["IBMRestBackend", "IBMJob", "IBMSession"]

# ── Constants ──

_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

_API_REGIONS: dict[str, str] = {
    "us": "https://quantum.cloud.ibm.com",
    "eu-de": "https://eu-de.quantum.cloud.ibm.com",
}

_API_VERSION = "2026-02-15"

# Gate mapping: Quanta → QASM 3.0
_QASM3_GATE_MAP: dict[str, str] = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t", "CX": "cx", "CZ": "cz",
    "CY": "cy", "SWAP": "swap", "CCX": "ccx",
    "RX": "rx", "RY": "ry", "RZ": "rz",
}


class IBMRestError(QuantaError):
    """IBM Quantum REST API error."""


@dataclass
class IBMJob:
    """Represents a submitted IBM Quantum job.

    Attributes:
        job_id: IBM job identifier.
        backend: Backend name.
        status: Current job status.
        result: Job result (populated after completion).
    """
    job_id: str
    backend: str
    status: str = "pending"
    result: dict = field(default_factory=dict)

    def is_done(self) -> bool:
        return self.status in ("Completed", "Failed", "Cancelled")


class IBMRestBackend(Backend):
    """Runs circuits on IBM Quantum via direct REST API.

    No Qiskit dependency — uses only urllib for HTTP calls.

    Args:
        api_key: IBM Cloud API key. Falls back to IBM_API_KEY env var.
        instance_crn: IBM Quantum instance CRN. Falls back to IBM_INSTANCE_CRN.
        region: "us" or "eu-de".
        backend_name: Target backend (e.g. "ibm_brisbane").
    """

    def __init__(
        self,
        api_key: str = "",
        instance_crn: str = "",
        region: str = "us",
        backend_name: str = "ibm_brisbane",
    ) -> None:
        self._api_key = api_key or os.environ.get("IBM_API_KEY", "")
        self._instance_crn = instance_crn or os.environ.get("IBM_INSTANCE_CRN", "")
        self._region = region
        self._backend_name = backend_name
        self._bearer_token: str = ""
        self._token_expiry: float = 0

        if region not in _API_REGIONS:
            raise IBMRestError(
                f"Unknown region: {region}. Supported: {list(_API_REGIONS.keys())}"
            )

        self._base_url = _API_REGIONS[region]

    @property
    def name(self) -> str:
        return f"ibm_rest_{self._backend_name}"

    # ── Authentication ──

    def _ensure_token(self) -> str:
        """Gets or refreshes the IAM bearer token."""
        if self._bearer_token and time.time() < self._token_expiry - 60:
            return self._bearer_token

        if not self._api_key:
            raise IBMRestError(
                "IBM Cloud API key is required.\n"
                "Set IBM_API_KEY environment variable or pass api_key= parameter.\n"
                "Get one at: https://cloud.ibm.com/iam/apikeys"
            )

        data = (
            "grant_type=urn:ibm:params:oauth:grant-type:apikey"
            f"&apikey={self._api_key}"
        ).encode()

        req = urllib.request.Request(
            _IAM_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
                self._bearer_token = body["access_token"]
                self._token_expiry = body.get("expiration", time.time() + 3600)
                return self._bearer_token
        except urllib.error.URLError as e:
            raise IBMRestError(f"IAM token exchange failed: {e}") from e

    def _headers(self) -> dict[str, str]:
        """Returns authenticated headers for API calls."""
        token = self._ensure_token()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "IBM-API-Version": _API_VERSION,
        }
        if self._instance_crn:
            headers["Service-CRN"] = self._instance_crn
        return headers

    def _api_call(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
    ) -> dict:
        """Makes an authenticated API call."""
        url = f"{self._base_url}/api/v1{endpoint}"
        body = json.dumps(data).encode() if data else None

        req = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            raise IBMRestError(
                f"IBM API error {e.code}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise IBMRestError(f"IBM API request failed: {e}") from e

    # ── Backend Discovery ──

    def list_backends(self) -> list[dict[str, Any]]:
        """Lists available IBM Quantum backends.

        Returns:
            List of backend info dicts with name, qubits, status.
        """
        result = self._api_call("GET", "/backends")
        backends = []
        for b in result.get("backends", result if isinstance(result, list) else []):
            backends.append({
                "name": b.get("name", "unknown"),
                "num_qubits": b.get("num_qubits", 0),
                "status": b.get("status", "unknown"),
                "version": b.get("version", ""),
            })
        return backends

    # ── QASM 3.0 Conversion ──

    def dag_to_qasm3(self, dag: DAGCircuit) -> str:
        """Converts a DAG circuit to QASM 3.0 for IBM submission.

        Uses the compact QASM 3.0 format compatible with IBM Runtime:
          - $N qubit addressing (physical qubits)
          - bit[N] c for classical registers
          - c[i] = measure $i for measurements
        """
        n = dag.num_qubits
        lines = [
            'OPENQASM 3.0;',
            'include "stdgates.inc";',
            f'bit[{n}] c;',
        ]

        for op in dag.op_nodes():
            qasm_name = _QASM3_GATE_MAP.get(op.gate_name)
            if qasm_name is None:
                raise IBMRestError(
                    f"Gate '{op.gate_name}' not supported. "
                    f"Supported: {list(_QASM3_GATE_MAP.keys())}"
                )
            qubit_args = ", ".join(f"${q}" for q in op.qubits)
            if op.params:
                param_str = ", ".join(f"{p:.10f}" for p in op.params)
                lines.append(f"{qasm_name}({param_str}) {qubit_args};")
            else:
                lines.append(f"{qasm_name} {qubit_args};")

        # Measurements
        measured = range(n)
        if dag.measurement and dag.measurement.qubits:
            measured = dag.measurement.qubits
        for i, q in enumerate(measured):
            lines.append(f"c[{i}] = measure ${q};")

        return " ".join(lines)

    # ── Job Submission ──

    def submit_sampler(
        self,
        dag: DAGCircuit,
        shots: int = 4096,
        session_id: str | None = None,
    ) -> IBMJob:
        """Submits a sampler job to IBM Quantum.

        Args:
            dag: Compiled DAG circuit.
            shots: Number of measurement shots.
            session_id: Optional session ID for batched execution.

        Returns:
            IBMJob with job_id for status tracking.
        """
        qasm_str = self.dag_to_qasm3(dag)

        payload: dict[str, Any] = {
            "program_id": "sampler",
            "backend": self._backend_name,
            "params": {
                "pubs": [[qasm_str]],
                "options": {},
                "version": 2,
            },
        }

        if session_id:
            payload["session_id"] = session_id

        result = self._api_call("POST", "/jobs", payload)
        return IBMJob(
            job_id=result.get("id", ""),
            backend=self._backend_name,
            status=result.get("status", "Queued"),
        )

    def submit_estimator(
        self,
        dag: DAGCircuit,
        observables: list[str],
        session_id: str | None = None,
    ) -> IBMJob:
        """Submits an estimator job to IBM Quantum.

        Computes ⟨ψ|O|ψ⟩ for given observables.

        Args:
            dag: Compiled DAG circuit.
            observables: Pauli string observables (e.g. ["ZZ", "XI"]).
            session_id: Optional session ID.

        Returns:
            IBMJob with job_id.
        """
        qasm_str = self.dag_to_qasm3(dag)

        pubs = [[qasm_str, obs] for obs in observables]

        payload: dict[str, Any] = {
            "program_id": "estimator",
            "backend": self._backend_name,
            "params": {
                "pubs": pubs,
                "options": {"dynamical_decoupling": {"enable": True}},
                "version": 2,
                "resilience_level": 1,
            },
        }

        if session_id:
            payload["session_id"] = session_id

        result = self._api_call("POST", "/jobs", payload)
        return IBMJob(
            job_id=result.get("id", ""),
            backend=self._backend_name,
            status=result.get("status", "Queued"),
        )

    # ── Job Management ──

    def job_status(self, job_id: str) -> dict:
        """Gets the status of a submitted job."""
        return self._api_call("GET", f"/jobs/{job_id}")

    def job_results(self, job_id: str) -> dict:
        """Gets the results of a completed job."""
        return self._api_call("GET", f"/jobs/{job_id}/results")

    def wait_for_job(
        self,
        job: IBMJob,
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> IBMJob:
        """Waits for a job to complete.

        Args:
            job: IBMJob to wait for.
            timeout: Max wait time in seconds.
            poll_interval: Seconds between status checks.

        Returns:
            Updated IBMJob with results.
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.job_status(job.job_id)
            job.status = status.get("status", "Unknown")

            if job.status == "Completed":
                job.result = self.job_results(job.job_id)
                return job
            if job.status in ("Failed", "Cancelled"):
                raise IBMRestError(
                    f"Job {job.job_id} {job.status}: "
                    f"{status.get('error', 'Unknown error')}"
                )

            time.sleep(poll_interval)

        raise IBMRestError(
            f"Job {job.job_id} timed out after {timeout}s "
            f"(status: {job.status})"
        )

    # ── Session Management ──

    def create_session(self, max_ttl: int = 28800) -> str:
        """Creates a new session for batched job execution.

        Args:
            max_ttl: Maximum session lifetime in seconds (default 8h).

        Returns:
            Session ID string.
        """
        result = self._api_call("POST", "/sessions", {
            "backend": self._backend_name,
            "max_ttl": max_ttl,
        })
        return result.get("id", "")

    def close_session(self, session_id: str) -> None:
        """Closes an active session."""
        self._api_call("DELETE", f"/sessions/{session_id}")

    def session(self, max_ttl: int = 28800) -> IBMSession:
        """Context manager for session-based execution.

        Example:
            >>> with backend.session() as s:
            ...     r1 = s.run(dag1)
            ...     r2 = s.run(dag2)
        """
        return IBMSession(self, max_ttl)

    # ── Backend Interface ──

    def execute(
        self,
        dag: DAGCircuit,
        shots: int = 4096,
        seed: int | None = None,
    ) -> Result:
        """Runs circuit on IBM Quantum and returns results.

        This is the synchronous high-level interface. It submits,
        waits, and returns the parsed result.

        Args:
            dag: Compiled DAG circuit.
            shots: Number of measurement shots.
            seed: Ignored (hardware is inherently random).

        Returns:
            Result with measurement counts.
        """
        job = self.submit_sampler(dag, shots=shots)
        completed = self.wait_for_job(job)

        # Parse IBM result format into Quanta counts
        counts = self._parse_counts(completed.result)

        return Result(
            counts=counts,
            shots=shots,
            num_qubits=dag.num_qubits,
        )

    def _parse_counts(self, raw_result: dict) -> dict[str, int]:
        """Parses IBM job results into measurement counts."""
        counts: dict[str, int] = {}

        # IBM returns results in various formats depending on version
        results = raw_result.get("results", [])
        if results:
            for pub_result in results:
                data = pub_result.get("data", {})
                # SamplerV2 format: data.c.samples or data.meas.samples
                for key in ("c", "meas", "cr"):
                    if key in data:
                        raw_counts = data[key].get("counts", {})
                        for bitstring, count in raw_counts.items():
                            clean = bitstring.replace(" ", "").replace("0x", "")
                            counts[clean] = counts.get(clean, 0) + count
                        break

        if not counts:
            # Fallback: try flat counts
            flat = raw_result.get("counts", {})
            for k, v in flat.items():
                counts[k.replace(" ", "")] = v

        return counts

    def __repr__(self) -> str:
        return (
            f"IBMRestBackend(backend='{self._backend_name}', "
            f"region='{self._region}')"
        )


class IBMSession:
    """Context manager for IBM Quantum sessions.

    Groups multiple jobs into a single session for priority
    queue access and reduced latency between executions.

    Example:
        >>> backend = IBMRestBackend()
        >>> with backend.session() as s:
        ...     r1 = s.run(dag1, shots=1024)
        ...     r2 = s.run(dag2, shots=1024)
    """

    def __init__(self, backend: IBMRestBackend, max_ttl: int = 28800) -> None:
        self._backend = backend
        self._max_ttl = max_ttl
        self._session_id: str = ""

    def __enter__(self) -> IBMSession:
        self._session_id = self._backend.create_session(self._max_ttl)
        return self

    def __exit__(self, *args: object) -> None:
        if self._session_id:
            with contextlib.suppress(IBMRestError):
                self._backend.close_session(self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    def run(
        self,
        dag: DAGCircuit,
        shots: int = 4096,
    ) -> Result:
        """Runs a circuit within this session.

        Args:
            dag: Compiled DAG circuit.
            shots: Measurement shots.

        Returns:
            Result with counts.
        """
        job = self._backend.submit_sampler(
            dag, shots=shots, session_id=self._session_id
        )
        completed = self._backend.wait_for_job(job)
        counts = self._backend._parse_counts(completed.result)

        return Result(
            counts=counts,
            shots=shots,
            num_qubits=dag.num_qubits,
        )

    def estimate(
        self,
        dag: DAGCircuit,
        observables: list[str],
    ) -> dict:
        """Runs an estimator job within this session.

        Args:
            dag: Compiled DAG circuit.
            observables: Pauli observables (e.g. ["ZZ", "XI"]).

        Returns:
            Raw estimator results from IBM.
        """
        job = self._backend.submit_estimator(
            dag, observables, session_id=self._session_id
        )
        completed = self._backend.wait_for_job(job)
        return completed.result
