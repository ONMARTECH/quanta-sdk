"""quanta.torch — PyTorch Native Quantum Layer & Tensor Operations.

Pillar 2 of Quanta SDK: High-performance, autograd-differentiable quantum circuits,
continuous-time quantum resonance, and device-accelerated quantum tensors on CPU
and Apple Silicon Metal (MPS).
"""

from __future__ import annotations

from quanta.torch import ops
from quanta.torch.brain import (
    BiomorphicResonantBrain,
    NoisyHippocampalBuffer,
    QuantumREMSleep,
    QuantumZenoAttention,
)
from quanta.torch.continuous import (
    ContinuousResonantLayer,
    GraphTopologyParser,
    _ContinuousResonantFunction,
)
from quanta.torch.layer import (
    ObservableParser,
    ParsedObservable,
    PauliTerm,
    QuantumLayer,
    _QuantumLayerFunction,
    get_ansatz_param_count,
)
from quanta.torch.ops import (
    ResonantInteractionBasis,
    UnsupportedDtypeError,
    apply_gate,
    batch_expectation,
    batch_multi_expectation,
    batch_pauli_kron,
    build_batch_resonant_hamiltonian,
    cnot_gate,
    create_initial_state,
    cz_gate,
    daleckii_krein_spectral_derivative,
    ehrenfest_time_gradient,
    fast_x_readout,
    fast_y_readout,
    fast_z_readout,
    get_complex_dtype,
    get_pauli_matrix,
    get_real_dtype,
    hamiltonian_expectation,
    pauli_kron,
    pauli_matrices,
    resolve_complex_dtype,
    resolve_device,
    rotation_x,
    rotation_y,
    rotation_z,
    simultaneous_readout,
    to_numpy_state,
    to_torch_state,
    unitary_evolution,
)

__all__ = [
    # Biomorphic Quantum Brain
    "BiomorphicResonantBrain",
    "NoisyHippocampalBuffer",
    "QuantumREMSleep",
    "QuantumZenoAttention",
    # Continuous Quantum Resonance (Milestone 3)
    "ContinuousResonantLayer",
    "_ContinuousResonantFunction",
    "GraphTopologyParser",
    # Layer & Autograd
    "QuantumLayer",
    "_QuantumLayerFunction",
    "ParsedObservable",
    "PauliTerm",
    "ObservableParser",
    "get_ansatz_param_count",
    # Ops Submodule
    "ops",
    # Core Ops & Types
    "UnsupportedDtypeError",
    "resolve_device",
    "resolve_complex_dtype",
    "get_real_dtype",
    "get_complex_dtype",
    "to_torch_state",
    "to_numpy_state",
    "create_initial_state",
    "get_pauli_matrix",
    "pauli_matrices",
    "pauli_kron",
    "batch_pauli_kron",
    "batch_expectation",
    "batch_multi_expectation",
    "fast_z_readout",
    "fast_x_readout",
    "fast_y_readout",
    "simultaneous_readout",
    "hamiltonian_expectation",
    "ResonantInteractionBasis",
    "build_batch_resonant_hamiltonian",
    "unitary_evolution",
    "ehrenfest_time_gradient",
    "daleckii_krein_spectral_derivative",
    "apply_gate",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "cnot_gate",
    "cz_gate",
]
