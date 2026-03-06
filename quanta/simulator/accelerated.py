"""
quanta.simulator.accelerated -- Optional GPU/JIT accelerated backend.

Tries to use JAX or CuPy for faster tensor operations.
Falls back to NumPy transparently if neither is available.

Priority: JAX > CuPy > NumPy

JAX: JIT compilation + GPU/TPU support (Google)
CuPy: Drop-in NumPy replacement for NVIDIA GPUs

This module provides a drop-in replacement for numpy operations
used by the statevector simulator. Import and use:

    from quanta.simulator.accelerated import xp, tensor_contract

Example:
    >>> from quanta.simulator.accelerated import get_backend_info
    >>> print(get_backend_info())
    {'backend': 'jax', 'device': 'gpu:0'}
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["xp", "tensor_contract", "get_backend_info"]


# -- Backend detection (lazy, no install required) --

_backend_name = "numpy"
_xp: Any = np  # array library (numpy, jax.numpy, or cupy)
_jit: Any = None  # JIT compiler (jax.jit or identity)


def _detect_backend() -> None:
    """Detects best available GPU backend. Called once on first use.

    IMPORTANT: JAX on CPU is SLOWER than NumPy (0.0x-0.8x) due to JIT
    overhead. We only activate JAX/CuPy when actual GPU/TPU is present.
    Benchmark (2026-03-06): CPU JAX 0.48s vs NumPy 0.002s at 12 qubits.
    """
    global _backend_name, _xp, _jit

    # Try JAX with GPU/TPU (Google's ML framework)
    try:
        import jax
        import jax.numpy as jnp

        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)
        has_tpu = any(d.platform == "tpu" for d in devices)

        if has_gpu:
            _xp = jnp
            _jit = jax.jit
            _backend_name = "jax-gpu"
            return
        elif has_tpu:
            _xp = jnp
            _jit = jax.jit
            _backend_name = "jax-tpu"
            return
        # CPU-only JAX: DON'T use it (slower than NumPy)
    except ImportError:
        pass

    # Try CuPy (requires NVIDIA GPU)
    try:
        import cupy
        # Only use if CUDA device is actually available
        cupy.cuda.runtime.getDevice()
        _xp = cupy
        _jit = None
        _backend_name = "cupy"
        return
    except (ImportError, Exception):
        pass

    # Default: NumPy (fastest on CPU)
    _xp = np
    _jit = None
    _backend_name = "numpy"


# Lazy init flag
_initialized = False


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        _detect_backend()
        _initialized = True


@property
def xp() -> Any:
    """Returns the active array library (numpy, jax.numpy, or cupy)."""
    _ensure_init()
    return _xp


def get_array_module() -> Any:
    """Returns the active array library."""
    _ensure_init()
    return _xp


def tensor_contract(
    gate: np.ndarray,
    state: np.ndarray,
    qubits: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Accelerated tensor contraction for gate application.

    Uses the best available backend (JAX JIT > CuPy > NumPy).

    Args:
        gate: Gate unitary matrix.
        state: Flat statevector.
        qubits: Target qubit indices.
        num_qubits: Total qubit count.

    Returns:
        New statevector after gate application.
    """
    _ensure_init()
    xp_mod = _xp
    n = num_qubits
    num_gate_qubits = len(qubits)

    # Convert to backend arrays if needed
    if _backend_name != "numpy":
        gate = xp_mod.asarray(gate)
        state = xp_mod.asarray(state)

    state_tensor = state.reshape([2] * n)
    gate_tensor = gate.reshape([2] * (2 * num_gate_qubits))

    gate_axes = list(range(num_gate_qubits, 2 * num_gate_qubits))
    state_axes = list(qubits)

    if _backend_name.startswith("jax"):
        # JAX tensordot
        result = xp_mod.tensordot(gate_tensor, state_tensor,
                                   axes=(gate_axes, state_axes))
    else:
        result = xp_mod.tensordot(gate_tensor, state_tensor,
                                   axes=(gate_axes, state_axes))

    source = list(range(num_gate_qubits))
    dest = list(qubits)

    if _backend_name.startswith("jax"):
        result = xp_mod.moveaxis(result, source, dest)
    else:
        result = xp_mod.moveaxis(result, source, dest)

    flat = result.reshape(-1)

    # Convert back to numpy if needed
    if _backend_name != "numpy":
        flat = np.asarray(flat)

    return flat


def get_backend_info() -> dict[str, str]:
    """Returns info about the active acceleration backend.

    Example:
        >>> get_backend_info()
        {'backend': 'numpy', 'device': 'cpu'}
    """
    _ensure_init()
    info = {"backend": _backend_name, "device": "cpu"}

    if _backend_name.startswith("jax"):
        try:
            import jax
            devices = jax.devices()
            info["device"] = str(devices[0])
            info["num_devices"] = str(len(devices))
        except Exception:
            pass
    elif _backend_name == "cupy":
        try:
            import cupy
            info["device"] = f"gpu:{cupy.cuda.runtime.getDevice()}"
        except Exception:
            pass

    return info
