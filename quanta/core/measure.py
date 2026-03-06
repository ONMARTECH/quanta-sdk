"""

aktif devreye kaydeder.

Example:
    >>> @circuit(qubits=2)
    ... def bell(q):
    ...     H(q[0])
    ...     CX(q[0], q[1])
    ...     return measure(q)
    >>>
    >>> @circuit(qubits=3)
    ... def partial(q):
    ...     H(q[0])
    ...     return measure(q[0], q[1])
"""

from __future__ import annotations

from quanta.core.types import MeasureSpec, QubitRef

# ── Public API ──
__all__ = ["measure"]

def measure(*args: QubitRef) -> MeasureSpec:
    """Measures the specified qubits.


    Args:
            - measure(q[0])  → tek qubit

    Returns:
    """
    from quanta.core.gates import _get_active_builder

    qubit_indices: list[int] = []
    for arg in args:
        if isinstance(arg, QubitRef):
            qubit_indices.append(arg.index)
        elif hasattr(arg, "__iter__"):
            for item in arg:
                if isinstance(item, QubitRef):
                    qubit_indices.append(item.index)

    spec = MeasureSpec(qubits=tuple(qubit_indices))

    # Aktif builder varsa kaydet
    try:
        builder = _get_active_builder()
        builder.set_measurement(spec)
    except Exception:
        pass

    return spec
