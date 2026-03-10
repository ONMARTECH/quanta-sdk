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
import ssl
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

# SSL context for macOS compatibility
def _ssl_context() -> ssl.SSLContext:
    """Creates SSL context with proper CA certificates."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

_USER_AGENT = "quanta-sdk/0.7.1 (Python; IBM-Quantum-Client)"

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
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
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
            "User-Agent": _USER_AGENT,
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
            with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
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
            List of backend info dicts with name, qubits, status,
            processor family, queue length, and error rates.
        """
        result = self._api_call("GET", "/backends")
        backends = []

        # IBM returns { "devices": [...] }
        devices = result.get("devices", result.get("backends", []))
        if isinstance(result, list):
            devices = result

        for b in devices:
            status = b.get("status", {})
            processor = b.get("processor_type", {})
            metrics = b.get("performance_metrics", {})

            backends.append({
                "name": b.get("name", "unknown"),
                "num_qubits": b.get("qubits", b.get("num_qubits", 0)),
                "status": (
                    status.get("name", "unknown")
                    if isinstance(status, dict) else str(status)
                ),
                "processor": f"{processor.get('family', '')} r{processor.get('revision', '')}",
                "queue_length": b.get("queue_length", 0),
                "two_q_error": metrics.get("two_q_error_median", {}).get("value", None),
                "readout_error": metrics.get("readout_error_median", {}).get("value", None),
            })
        return backends

    # ── ISA Transpilation (Heron Native Gates) ──

    @staticmethod
    def _gate_to_isa(
        gate: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] | None,
    ) -> list[str]:
        """Transpiles a gate to IBM Heron ISA instructions.

        Heron native gate set: rz, sx, x, cz, measure.
        All other gates are decomposed into these.

        Decompositions:
          H  = rz(π/2) · sx · rz(π/2)
          Y  = x · rz(π)  (global phase ignored)
          Z  = rz(π)
          S  = rz(π/2)
          T  = rz(π/4)
          CX = H(target) · CZ · H(target)
          RX(θ) = rz(-π/2) · sx · rz(π-θ) · sx · rz(-π/2)
          RY(θ) = rz(θ) · sx · rz(π) · sx  (simplified)
          RZ(θ) = rz(θ)  (native)
        """
        import math
        pi = math.pi
        half = pi / 2

        q0 = qubits[0]
        ops: list[str] = []

        def _h(q: int) -> list[str]:
            return [
                f"rz({half}) ${q};",
                f"sx ${q};",
                f"rz({half}) ${q};",
            ]

        if gate == "H":
            ops.extend(_h(q0))

        elif gate == "X":
            ops.append(f"x ${q0};")

        elif gate == "Y":
            ops.append(f"x ${q0};")
            ops.append(f"rz({pi}) ${q0};")

        elif gate == "Z":
            ops.append(f"rz({pi}) ${q0};")

        elif gate == "S":
            ops.append(f"rz({half}) ${q0};")

        elif gate == "T":
            ops.append(f"rz({pi / 4}) ${q0};")

        elif gate == "RZ":
            theta = params[0] if params else 0
            ops.append(f"rz({theta}) ${q0};")

        elif gate == "RX":
            theta = params[0] if params else 0
            ops.append(f"rz({-half}) ${q0};")
            ops.append(f"sx ${q0};")
            ops.append(f"rz({pi - theta}) ${q0};")
            ops.append(f"sx ${q0};")
            ops.append(f"rz({-half}) ${q0};")

        elif gate == "RY":
            theta = params[0] if params else 0
            ops.extend(_h(q0))
            ops.append(f"rz({theta}) ${q0};")
            ops.extend(_h(q0))

        elif gate == "CX":
            q1 = qubits[1]
            # CX = H(target) · CZ · H(target)
            ops.extend(_h(q1))
            ops.append(f"cz ${q0}, ${q1};")
            ops.extend(_h(q1))

        elif gate == "CZ":
            q1 = qubits[1]
            ops.append(f"cz ${q0}, ${q1};")

        elif gate == "CY":
            q1 = qubits[1]
            # CY = S†(target) · CX · S(target)
            ops.append(f"rz({-half}) ${q1};")
            ops.extend(_h(q1))
            ops.append(f"cz ${q0}, ${q1};")
            ops.extend(_h(q1))
            ops.append(f"rz({half}) ${q1};")

        elif gate == "CCX":
            # Toffoli — simplified decomposition
            q1, q2 = qubits[1], qubits[2]
            # H target
            ops.extend(_h(q2))
            # CX(q1, q2)
            ops.extend(_h(q2))
            ops.append(f"cz ${q1}, ${q2};")
            ops.extend(_h(q2))
            # T† target
            ops.append(f"rz({-pi / 4}) ${q2};")
            # CX(q0, q2)
            ops.extend(_h(q2))
            ops.append(f"cz ${q0}, ${q2};")
            ops.extend(_h(q2))
            # T target
            ops.append(f"rz({pi / 4}) ${q2};")
            # CX(q1, q2)
            ops.extend(_h(q2))
            ops.append(f"cz ${q1}, ${q2};")
            ops.extend(_h(q2))
            # T† target
            ops.append(f"rz({-pi / 4}) ${q2};")
            # CX(q0, q2)
            ops.extend(_h(q2))
            ops.append(f"cz ${q0}, ${q2};")
            ops.extend(_h(q2))
            # T q1, T target, T† q0
            ops.append(f"rz({pi / 4}) ${q1};")
            ops.append(f"rz({pi / 4}) ${q2};")
            ops.append(f"rz({-pi / 4}) ${q0};")
            # CX(q0, q1)
            ops.extend(_h(q1))
            ops.append(f"cz ${q0}, ${q1};")
            ops.extend(_h(q1))
            # T q0, T† q1
            ops.append(f"rz({pi / 4}) ${q0};")
            ops.append(f"rz({-pi / 4}) ${q1};")
            # H target
            ops.extend(_h(q2))

        elif gate == "SWAP":
            q1 = qubits[1]
            # SWAP = CX(0,1) · CX(1,0) · CX(0,1)
            for a, b in [(q0, q1), (q1, q0), (q0, q1)]:
                ops.extend(_h(b))
                ops.append(f"cz ${a}, ${b};")
                ops.extend(_h(b))

        # ── New gates ──

        elif gate == "I":
            # Identity: no operation (skip)
            pass

        elif gate == "SDG":
            ops.append(f"rz({-half}) ${q0};")

        elif gate == "TDG":
            ops.append(f"rz({-pi / 4}) ${q0};")

        elif gate == "P":
            theta = params[0] if params else 0
            ops.append(f"rz({theta}) ${q0};")

        elif gate == "SX":
            ops.append(f"sx ${q0};")

        elif gate == "SXdg":
            # SXdg = rz(π) · sx · rz(π)
            ops.append(f"rz({pi}) ${q0};")
            ops.append(f"sx ${q0};")
            ops.append(f"rz({pi}) ${q0};")

        elif gate == "U":
            # U(θ, φ, λ) = rz(φ) · sx · rz(θ+π) · sx · rz(λ)
            theta = params[0] if params else 0
            phi = params[1] if params and len(params) > 1 else 0
            lam = params[2] if params and len(params) > 2 else 0
            ops.append(f"rz({lam}) ${q0};")
            ops.append(f"sx ${q0};")
            ops.append(f"rz({theta + pi}) ${q0};")
            ops.append(f"sx ${q0};")
            ops.append(f"rz({phi + pi}) ${q0};")

        elif gate == "RXX":
            q1 = qubits[1]
            theta = params[0] if params else 0
            # RXX(θ) = (H⊗H) · CX · RZ(θ) · CX · (H⊗H)
            ops.extend(_h(q0))
            ops.extend(_h(q1))
            ops.extend(_h(q1))
            ops.append(f"cz ${q0}, ${q1};")
            ops.extend(_h(q1))
            ops.append(f"rz({theta}) ${q1};")
            ops.extend(_h(q1))
            ops.append(f"cz ${q0}, ${q1};")
            ops.extend(_h(q1))
            ops.extend(_h(q0))
            ops.extend(_h(q1))

        elif gate == "RZZ":
            q1 = qubits[1]
            theta = params[0] if params else 0
            # RZZ(θ) = CX · RZ(θ) · CX
            ops.extend(_h(q1))
            ops.append(f"cz ${q0}, ${q1};")
            ops.extend(_h(q1))
            ops.append(f"rz({theta}) ${q1};")
            ops.extend(_h(q1))
            ops.append(f"cz ${q0}, ${q1};")
            ops.extend(_h(q1))

        elif gate == "RCCX":
            # Relative-phase Toffoli — same as CCX up to phase
            q1, q2 = qubits[1], qubits[2]
            ops.extend(_h(q2))
            ops.extend(_h(q2))
            ops.append(f"cz ${q1}, ${q2};")
            ops.extend(_h(q2))
            ops.append(f"rz({-pi / 4}) ${q2};")
            ops.extend(_h(q2))
            ops.append(f"cz ${q0}, ${q2};")
            ops.extend(_h(q2))
            ops.append(f"rz({pi / 4}) ${q2};")
            ops.extend(_h(q2))
            ops.append(f"cz ${q1}, ${q2};")
            ops.extend(_h(q2))
            ops.extend(_h(q2))

        elif gate == "RC3X":
            # 3-controlled X — decompose via RCCX chain
            q1, q2, q3 = qubits[1], qubits[2], qubits[3]
            # Use CCX-like decomposition for 4 qubits
            ops.extend(_h(q3))
            ops.extend(_h(q3))
            ops.append(f"cz ${q2}, ${q3};")
            ops.extend(_h(q3))
            ops.append(f"rz({-pi / 4}) ${q3};")
            ops.extend(_h(q3))
            ops.append(f"cz ${q1}, ${q3};")
            ops.extend(_h(q3))
            ops.append(f"rz({pi / 4}) ${q3};")
            ops.extend(_h(q3))
            ops.append(f"cz ${q0}, ${q3};")
            ops.extend(_h(q3))
            ops.extend(_h(q3))

        else:
            raise IBMRestError(
                f"Gate '{gate}' cannot be transpiled to ISA. "
                f"Supported: H, X, Y, Z, S, T, CX, CZ, CY, CCX, "
                f"SWAP, RX, RY, RZ, I, SDG, TDG, P, SX, SXdg, "
                f"U, RXX, RZZ, RCCX, RC3X"
            )

        return ops

    def dag_to_qasm3(self, dag: DAGCircuit) -> str:
        """Converts a DAG circuit to ISA-transpiled QASM 3.0.

        Automatically decomposes all gates into Heron native
        gate set (rz, sx, x, cz). No manual transpilation needed.
        """
        n = dag.num_qubits
        lines = [
            'OPENQASM 3.0;',
            'include "stdgates.inc";',
            f'bit[{n}] c;',
        ]

        for op in dag.op_nodes():
            isa_ops = self._gate_to_isa(
                op.gate_name, op.qubits, op.params
            )
            lines.extend(isa_ops)

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

        results = raw_result.get("results", [])
        if results:
            for pub_result in results:
                data = pub_result.get("data", {})
                for key in ("c", "meas", "cr"):
                    if key not in data:
                        continue

                    reg = data[key]

                    # Format 1: SamplerV2 hex samples ["0x0", "0x3", ...]
                    if "samples" in reg:
                        samples = reg["samples"]
                        n_bits = reg.get("num_bits", 2)
                        for sample in samples:
                            val = int(sample, 16) if isinstance(sample, str) else sample
                            bits = format(val, f"0{n_bits}b")
                            counts[bits] = counts.get(bits, 0) + 1
                        break

                    # Format 2: Pre-counted {"00": 500, "11": 500}
                    if "counts" in reg:
                        for bitstring, count in reg["counts"].items():
                            clean = bitstring.replace(" ", "").replace("0x", "")
                            counts[clean] = counts.get(clean, 0) + count
                        break

        if not counts:
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
