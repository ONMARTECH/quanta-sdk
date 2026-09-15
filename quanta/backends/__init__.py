"""quanta.backends — Quantum execution backends.

Provides execution targets for Quanta circuits:
- LocalBackend: Local CPU/GPU simulation
- IBMRestBackend: Real IBM Quantum hardware via direct REST API
- GoogleBackend: Google Quantum computing backend
- IonQBackend: IonQ trapped-ion quantum computing backend
"""

from quanta.backends.base import Backend
from quanta.backends.google import GoogleBackend
from quanta.backends.ibm import IBMBackend
from quanta.backends.ibm_rest import IBMJob, IBMRestBackend, IBMSession
from quanta.backends.ionq import IonQBackend
from quanta.backends.local import LocalSimulator

# Alias for naming consistency
LocalBackend = LocalSimulator

__all__ = [
    "Backend",
    "LocalSimulator",
    "LocalBackend",
    "IBMBackend",
    "IBMRestBackend",
    "IBMJob",
    "IBMSession",
    "GoogleBackend",
    "IonQBackend",
]
