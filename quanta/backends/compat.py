"""
quanta.backends.compat — Backend compatibility layer.

Detects installed versions of external quantum SDKs and provides
version-resilient import shims. This protects Quanta from breaking
API changes in Qiskit (IBM) and Cirq (Google).

Strategy:
    1. QASM is the primary isolation layer (stable standard)
    2. Version detection at import time
    3. Adapter functions that work across version ranges
    4. Clear error messages with upgrade/downgrade guidance

Known API breaks:
    Qiskit 1.x:  qiskit.execute() removed, use SamplerV2
    Qiskit 0.x:  qiskit.execute() was the default
    Cirq 1.4+:   cirq.contrib.qasm_import may be relocated
    Cirq 1.3:    Stable API for qasm_import

Example:
    >>> from quanta.backends.compat import qiskit_version, cirq_version
    >>> print(qiskit_version())  # "1.3.1" or None
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any

__all__ = [
    "qiskit_version",
    "cirq_version",
    "BackendVersionInfo",
    "check_backend_compatibility",
    "import_qiskit_safe",
    "import_cirq_safe",
]


# ═══════════════════════════════════════════
#  Version Info
# ═══════════════════════════════════════════

# Supported version ranges
_QISKIT_MIN = "1.0.0"
_QISKIT_MAX = "2.99.0"
_CIRQ_MIN = "1.3.0"
_CIRQ_MAX = "1.99.0"


@dataclass
class BackendVersionInfo:
    """Version and compatibility info for an external backend SDK."""

    name: str
    version: str | None
    installed: bool
    compatible: bool
    message: str

    def __repr__(self) -> str:
        status = "✅" if self.compatible else "⚠️" if self.installed else "❌"
        ver = self.version or "not installed"
        return f"{status} {self.name} {ver}"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse version string to comparable tuple."""
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def qiskit_version() -> str | None:
    """Get installed Qiskit version, or None if not installed."""
    try:
        import qiskit  # type: ignore[import-untyped]
        return getattr(qiskit, "__version__", "unknown")
    except ImportError:
        return None


def cirq_version() -> str | None:
    """Get installed Cirq version, or None if not installed."""
    try:
        import cirq  # type: ignore[import-untyped]
        return getattr(cirq, "__version__", "unknown")
    except ImportError:
        return None


# ═══════════════════════════════════════════
#  Compatibility Checks
# ═══════════════════════════════════════════

def check_backend_compatibility() -> list[BackendVersionInfo]:
    """Check all backend SDK versions and compatibility.

    Returns list of BackendVersionInfo for each external SDK.
    Useful for diagnostics and CI health checks.

    Example:
        >>> for info in check_backend_compatibility():
        ...     print(info)
        ✅ qiskit 1.3.1
        ⚠️ cirq 1.5.0 (untested version)
        ❌ ionq-sdk not installed (uses REST API — no SDK needed)
    """
    results: list[BackendVersionInfo] = []

    # Qiskit
    qv = qiskit_version()
    if qv is None:
        results.append(BackendVersionInfo(
            "qiskit", None, False, False,
            "Not installed. Install: pip install qiskit qiskit-ibm-runtime",
        ))
    else:
        ver = _parse_version(qv)
        compatible = _parse_version(_QISKIT_MIN) <= ver <= _parse_version(_QISKIT_MAX)
        msg = "Compatible" if compatible else (
            f"Version {qv} outside tested range [{_QISKIT_MIN}, {_QISKIT_MAX}]. "
            "API differences may cause errors."
        )
        results.append(BackendVersionInfo("qiskit", qv, True, compatible, msg))

    # Cirq
    cv = cirq_version()
    if cv is None:
        results.append(BackendVersionInfo(
            "cirq", None, False, False,
            "Not installed. Install: pip install cirq-google",
        ))
    else:
        ver = _parse_version(cv)
        compatible = _parse_version(_CIRQ_MIN) <= ver <= _parse_version(_CIRQ_MAX)
        msg = "Compatible" if compatible else (
            f"Version {cv} outside tested range [{_CIRQ_MIN}, {_CIRQ_MAX}]. "
            "API differences may cause errors."
        )
        results.append(BackendVersionInfo("cirq", cv, True, compatible, msg))

    # IonQ — stdlib only, always compatible
    results.append(BackendVersionInfo(
        "ionq", "REST API v0.3", True, True,
        "Uses Python stdlib (urllib). No external SDK needed.",
    ))

    return results


# ═══════════════════════════════════════════
#  Safe Import Shims
# ═══════════════════════════════════════════

def import_qiskit_safe() -> dict[str, Any]:
    """Import Qiskit components with version-adaptive API resolution.

    Handles known API breaks between Qiskit versions:
      - v1.x: SamplerV2, no qiskit.execute()
      - v0.x: qiskit.execute(), BasicAer/Aer split

    Returns dict with resolved components:
        {
            "QuantumCircuit": <class>,
            "from_qasm": <function>,  # version-appropriate QASM parser
            "simulator": <class or None>,
            "sampler": <class or None>,
            "version": <str>,
        }

    Raises:
        ImportError: If Qiskit is not installed.
    """
    import qiskit  # type: ignore[import-untyped]  # noqa: F811
    from qiskit import QuantumCircuit  # type: ignore[import-untyped]

    ver = _parse_version(qiskit.__version__)
    result: dict[str, Any] = {
        "QuantumCircuit": QuantumCircuit,
        "version": qiskit.__version__,
    }

    # QASM import — consistent across versions
    result["from_qasm"] = QuantumCircuit.from_qasm_str

    # Simulator — try multiple import paths
    result["simulator"] = None
    for sim_path in [
        "qiskit_aer.AerSimulator",
        "qiskit.providers.aer.AerSimulator",
        "qiskit.providers.basicaer.QasmSimulatorPy",
    ]:
        module_path, class_name = sim_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            result["simulator"] = getattr(mod, class_name)
            break
        except (ImportError, AttributeError):
            continue

    if result["simulator"] is None:
        warnings.warn(
            "No Qiskit simulator found. Install qiskit-aer for local simulation: "
            "pip install qiskit-aer",
            UserWarning,
            stacklevel=2,
        )

    # Sampler — SamplerV2 (Qiskit 1.x) or Sampler (0.x fallback)
    result["sampler"] = None
    for sampler_path in [
        "qiskit_ibm_runtime.SamplerV2",
        "qiskit_ibm_runtime.Sampler",
        "qiskit.primitives.Sampler",
    ]:
        module_path, class_name = sampler_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            result["sampler"] = getattr(mod, class_name)
            break
        except (ImportError, AttributeError):
            continue

    # Runtime service
    result["service"] = None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import-untyped]
        result["service"] = QiskitRuntimeService
    except ImportError:
        pass

    # Version warning
    if ver < _parse_version(_QISKIT_MIN) or ver > _parse_version(_QISKIT_MAX):
        warnings.warn(
            f"Qiskit {qiskit.__version__} outside tested range "
            f"[{_QISKIT_MIN}, {_QISKIT_MAX}]. "
            "Some features may not work correctly.",
            UserWarning,
            stacklevel=2,
        )

    return result


def import_cirq_safe() -> dict[str, Any]:
    """Import Cirq components with version-adaptive API resolution.

    Handles known API breaks between Cirq versions:
      - v1.4+: qasm_import may move out of contrib
      - v1.3:  Stable cirq.contrib.qasm_import

    Returns dict with resolved components:
        {
            "cirq": <module>,
            "from_qasm": <function or None>,
            "cirq_google": <module or None>,
            "version": <str>,
        }

    Raises:
        ImportError: If Cirq is not installed.
    """
    import cirq  # type: ignore[import-untyped]  # noqa: F811

    ver = _parse_version(cirq.__version__)
    result: dict[str, Any] = {
        "cirq": cirq,
        "version": cirq.__version__,
    }

    # QASM import — try multiple locations
    result["from_qasm"] = None
    for qasm_path in [
        "cirq.contrib.qasm_import.circuit_from_qasm",
        "cirq.qasm.circuit_from_qasm",
        "cirq_core.qasm.circuit_from_qasm",
    ]:
        parts = qasm_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(parts[0])
            result["from_qasm"] = getattr(mod, parts[1])
            break
        except (ImportError, AttributeError):
            continue

    # Google backend
    result["cirq_google"] = None
    try:
        import cirq_google  # type: ignore[import-untyped]
        result["cirq_google"] = cirq_google
    except ImportError:
        pass

    # Version warning
    if ver < _parse_version(_CIRQ_MIN) or ver > _parse_version(_CIRQ_MAX):
        warnings.warn(
            f"Cirq {cirq.__version__} outside tested range "
            f"[{_CIRQ_MIN}, {_CIRQ_MAX}]. "
            "Some features may not work correctly.",
            UserWarning,
            stacklevel=2,
        )

    return result
