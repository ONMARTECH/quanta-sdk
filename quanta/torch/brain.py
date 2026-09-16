"""quanta.torch.brain — Biomorphic Resonant Quantum Brain Architecture.

Models the human brain's dual-hemisphere dynamics, biochemical neuromodulation
(dopamine, norepinephrine, serotonin), hemodynamic oxygenation constraints,
and holistic projective consensus ("her yerden aynı anda ışıldayan kuantum meclisi").

Key Biophysical Pillars:
1. Bipartite Cerebral Topology: Left lobe (analytical Z-fields) vs. Right lobe (holistic XY)
   coupled via the Corpus Callosum (J_callosum tunneling bridge).
2. Neuromodulatory Chemical Fields:
   - Dopamine (D): Scales transverse X-field tunneling and cognitive exploration.
   - Norepinephrine (NE): Modulates decision arousal and time evolution rate.
   - Serotonin (5-HT): Stabilizes phase coherence and damps inter-lobe competition.
3. Hemodynamic Oxygenation Budget:
   - Metabolic resource conservation (fMRI BOLD response) dynamically balancing
     left vs. right lobe energy consumption: M_left(x) + M_right(x) = const.
4. Collective Parliament Consensus Readout:
   - Holistic state projection yielding an undisputed macroscopic decision
     without sequential gate bottlenecks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from quanta.torch import ops

__all__ = ["BiomorphicResonantBrain"]


class BiomorphicResonantBrain(nn.Module):
    """Bipartite Neuromorphic Quantum Brain with Chemical & Metabolic Modulation.

    Args:
        in_features: Dimension of classical input vector.
        num_left_qubits: Qubit count allocated to the Left Hemispehere (default: 2).
        num_right_qubits: Qubit count allocated to the Right Hemisphere (default: 2).
        learnable_callosum: Whether corpus callosum inter-lobe tunneling is trainable.
        enable_neuromodulation: Whether dopamine, norepinephrine, serotonin channels are active.
        enable_oxygenation: Whether hemodynamic metabolic energy constraint is enforced.
        initial_state: Reference quantum state ('zero', 'superposition', or custom).
        device: PyTorch device ('cpu', 'mps', or torch.device).
        dtype: Real floating point dtype (torch.float32 or torch.float64).
    """

    def __init__(
        self,
        in_features: int,
        num_left_qubits: int = 2,
        num_right_qubits: int = 2,
        learnable_callosum: bool = True,
        enable_neuromodulation: bool = True,
        enable_oxygenation: bool = True,
        initial_state: str = "superposition",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_left_qubits < 1 or num_right_qubits < 1:
            raise ValueError(
                f"Both hemispheres must have >= 1 qubit, got left={num_left_qubits}, "
                f"right={num_right_qubits}"
            )

        self.in_features = in_features
        self.num_left_qubits = num_left_qubits
        self.num_right_qubits = num_right_qubits
        self.num_qubits = num_left_qubits + num_right_qubits
        self.dim = 2**self.num_qubits
        self.enable_neuromodulation = enable_neuromodulation
        self.enable_oxygenation = enable_oxygenation
        self.initial_state_mode = initial_state

        target_dev = ops.resolve_device(device) if device is not None else None
        dev = target_dev if target_dev is not None else torch.device("cpu")
        if (
            target_dev is not None
            and target_dev.type == "mps"
            and dtype in (torch.float64, torch.complex128)
        ):
            raise ops.UnsupportedDtypeError(
                f"Apple Silicon MPS does not support 64-bit precision ({dtype}). "
                f"Use torch.float32 on MPS or switch execution to device='cpu'."
            )

        self.real_dtype = (
            dtype if dtype is not None else (torch.float32 if dev.type == "mps" else torch.float64)
        )
        self.complex_dtype = ops.resolve_complex_dtype(self.real_dtype, dev)

        # 1. Left Hemisphere Topology: Line/Chain connections within Left Lobe
        self.left_edges = [(i, i + 1) for i in range(num_left_qubits - 1)]
        # 2. Right Hemisphere Topology: Complete/All-to-all connections within Right Lobe (Holistic)
        r_start = num_left_qubits
        self.right_edges = [
            (r_start + j, r_start + k)
            for j in range(num_right_qubits)
            for k in range(j + 1, num_right_qubits)
        ]
        # 3. Corpus Callosum: Bridges boundary and center nodes between Left and Right
        self.callosum_edges = [
            (num_left_qubits - 1, r_start),
            (0, self.num_qubits - 1)
            if num_left_qubits > 1 and num_right_qubits > 1
            else (num_left_qubits - 1, r_start),
        ]
        self.callosum_edges = sorted(list(set(self.callosum_edges)))

        # 4. Intra-Lobe Parameters
        # Right lobe has strong XY entanglement coupling
        num_right_e = max(1, len(self.right_edges))
        self.J_right = nn.Parameter(torch.empty(num_right_e, device=dev, dtype=self.real_dtype))
        # Left lobe has strong Z-bias fields (analytical categorization)
        self.h_left = nn.Parameter(torch.empty(num_left_qubits, device=dev, dtype=self.real_dtype))
        # Input projection weights to left lobe
        self.W_left = nn.Parameter(
            torch.empty(num_left_qubits, in_features, device=dev, dtype=self.real_dtype)
        )

        # 5. Corpus Callosum Tunneling Parameter
        callosum_init = torch.full(
            (len(self.callosum_edges),), 0.5, device=dev, dtype=self.real_dtype
        )
        if learnable_callosum:
            self.J_callosum = nn.Parameter(callosum_init)
        else:
            self.register_buffer("J_callosum", callosum_init)

        # 6. Neuromodulatory Chemical System
        if enable_neuromodulation:
            # Baseline levels: [Dopamine, Norepinephrine, Serotonin]
            self.hormone_bias = nn.Parameter(
                torch.tensor([0.5, 1.0, 0.5], device=dev, dtype=self.real_dtype)
            )
            self.hormone_proj = nn.Parameter(
                torch.empty(3, in_features, device=dev, dtype=self.real_dtype)
            )
        else:
            self.register_buffer(
                "hormone_bias", torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=self.real_dtype)
            )

        # 7. Hemodynamic Oxygenation (Metabolic Energy Constraint)
        if enable_oxygenation:
            # Linear projection determining left vs. right metabolic allocation
            self.oxy_weight = nn.Parameter(
                torch.empty(1, in_features, device=dev, dtype=self.real_dtype)
            )
            self.oxy_bias = nn.Parameter(torch.zeros(1, device=dev, dtype=self.real_dtype))

        # Base evolution duration
        self.base_time = nn.Parameter(torch.tensor(1.0, device=dev, dtype=self.real_dtype))

        # Initialize parameter weights
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.J_right, -0.8, 0.8)
        nn.init.uniform_(self.h_left, -0.5, 0.5)
        nn.init.kaiming_uniform_(self.W_left, a=math.sqrt(5))
        if self.enable_neuromodulation:
            nn.init.kaiming_uniform_(self.hormone_proj, a=math.sqrt(5))
        if self.enable_oxygenation:
            nn.init.kaiming_uniform_(self.oxy_weight, a=math.sqrt(5))

    def _build_hamiltonian(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Constructs the batch total biomorphic Hamiltonian.

        Returns:
            (H_total: [batch, dim, dim], effective_time: [batch, 1])
        """
        batch_size = x.shape[0]
        dev = x.device
        real_dtype = x.dtype
        cdtype = ops.resolve_complex_dtype(real_dtype, dev)
        N = self.num_qubits

        # 1. Neuromodulation evaluation
        if self.enable_neuromodulation:
            raw_hormones = torch.sigmoid(torch.matmul(x, self.hormone_proj.t()) + self.hormone_bias)
            dopamine = raw_hormones[:, 0:1] * 2.0  # X-tunneling gain
            norepinephrine = raw_hormones[:, 1:2] * 2.0  # Time arousal acceleration
            serotonin = raw_hormones[:, 2:3]  # Coherence stabilization
        else:
            dopamine = torch.full((batch_size, 1), 0.5, device=dev, dtype=real_dtype)
            norepinephrine = torch.full((batch_size, 1), 1.0, device=dev, dtype=real_dtype)
            serotonin = torch.full((batch_size, 1), 0.5, device=dev, dtype=real_dtype)

        # 2. Hemodynamic oxygenation constraint: M_L + M_R = 2.0 (conservation of metabolic energy)
        if self.enable_oxygenation:
            alloc = torch.sigmoid(
                torch.matmul(x, self.oxy_weight.t()) + self.oxy_bias
            )  # [B, 1] in (0, 1)
            M_left = alloc * 2.0
            M_right = 2.0 - M_left
        else:
            M_left = torch.ones((batch_size, 1), device=dev, dtype=real_dtype)
            M_right = torch.ones((batch_size, 1), device=dev, dtype=real_dtype)

        # Initialize H_total
        H_total = torch.zeros((batch_size, self.dim, self.dim), device=dev, dtype=cdtype)

        # A. Left Hemisphere: Z-bias and Input Encoding scaled by M_left
        left_z = torch.matmul(x, self.W_left.t()) + self.h_left  # [B, num_left_qubits]
        for q in range(self.num_left_qubits):
            z_mat = ops.pauli_kron("Z", num_qubits=N, target_qubit=q, device=dev, dtype=cdtype)
            coeff = (left_z[:, q : q + 1] * M_left).to(cdtype).unsqueeze(-1)
            H_total = H_total + coeff * z_mat

        # B. Right Hemisphere: XY Entanglement Network scaled by M_right
        for idx, (u, v) in enumerate(self.right_edges):
            s_x = "".join(["X" if k in (u, v) else "I" for k in range(N)])
            s_y = "".join(["Y" if k in (u, v) else "I" for k in range(N)])
            xx = ops.pauli_kron(s_x, num_qubits=N, device=dev, dtype=cdtype)
            yy = ops.pauli_kron(s_y, num_qubits=N, device=dev, dtype=cdtype)
            xy_op = xx + yy
            weight = (self.J_right[idx] * M_right).to(cdtype).unsqueeze(-1)
            H_total = H_total + weight * xy_op

        # C. Corpus Callosum Bridge (Inter-lobe Tunneling)
        for idx, (u, v) in enumerate(self.callosum_edges):
            s_x = "".join(["X" if k in (u, v) else "I" for k in range(N)])
            s_y = "".join(["Y" if k in (u, v) else "I" for k in range(N)])
            xx = ops.pauli_kron(s_x, num_qubits=N, device=dev, dtype=cdtype)
            yy = ops.pauli_kron(s_y, num_qubits=N, device=dev, dtype=cdtype)
            bridge_op = xx + yy
            w_bridge = (self.J_callosum[idx] * (1.0 + serotonin)).to(cdtype).unsqueeze(-1)
            H_total = H_total + w_bridge * bridge_op

        # D. Dopamine-Driven Transverse Field Exploration (X-Tunneling across all nodes)
        for q in range(N):
            x_mat = ops.pauli_kron("X", num_qubits=N, target_qubit=q, device=dev, dtype=cdtype)
            dopamine_weight = dopamine.to(cdtype).unsqueeze(-1) * 0.5
            H_total = H_total + dopamine_weight * x_mat

        # Effective duration: modulated by Norepinephrine
        eff_time = (torch.abs(self.base_time) + 0.1) * norepinephrine

        return H_total, eff_time

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass simulating whole-brain quantum dynamics and collective consensus.

        Args:
            x: Input tensor [batch_size, in_features] or [in_features].

        Returns:
            Dictionary containing:
                - 'consensus': Macroscopic parliament decision scalar in [-1, 1] [batch, 1].
                - 'left_consensus': Left hemisphere analytic polarity [batch, 1].
                - 'right_consensus': Right hemisphere holistic polarity [batch, 1].
                - 'readout_vector': Full multi-observable readout [batch, num_qubits].
                - 'state': Evolved quantum statevector [batch, 2^num_qubits].
        """
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)

        dev = self.W_left.device
        real_dtype = self.W_left.dtype
        cdtype = ops.resolve_complex_dtype(real_dtype, dev)
        N = self.num_qubits
        batch_size = x.shape[0]

        if x.device != dev or x.dtype != real_dtype:
            x = x.to(device=dev, dtype=real_dtype)

        # Construct biomorphic Hamiltonian
        H_total, eff_time = self._build_hamiltonian(x)

        # Initial reference state (equal superposition |+>^N if 'superposition')
        init_mode = self.initial_state_mode
        if isinstance(init_mode, str) and init_mode.lower() in ("superposition", "uniform"):
            init_mode = "plus"

        psi_batch = ops.create_initial_state(
            init_mode,
            num_qubits=N,
            batch_size=batch_size,
            device=dev,
            dtype=cdtype,
        )

        # Continuous Unitary Evolution |psi(t)> = exp(-i H t) |psi_0>
        # Diagonalize Hermitian Hamiltonian per batch
        eigvals, eigvecs = torch.linalg.eigh(H_total)  # [B, dim], [B, dim, dim]
        phases = torch.exp(-1j * (eigvals * eff_time))  # [B, dim]
        diag_phases = torch.diag_embed(phases)  # [B, dim, dim]
        U_total = eigvecs @ diag_phases @ eigvecs.mH  # [B, dim, dim]

        psi_t = (U_total @ psi_batch.unsqueeze(-1)).squeeze(-1)  # [B, dim]

        # Holistic Readouts across both Hemispheres via fast O(2^N) projector
        all_z = ops.fast_z_readout(psi_t, N)  # [B, N]

        # Left Hemisphere Consensus: Average of Left Qubits
        left_consensus = all_z[:, : self.num_left_qubits].mean(dim=-1, keepdim=True)
        # Right Hemisphere Consensus: Average of Right Qubits
        right_consensus = all_z[:, self.num_left_qubits :].mean(dim=-1, keepdim=True)
        # Parliament Consensus: Global Unanimous Polarity
        parliament_consensus = all_z.mean(dim=-1, keepdim=True)

        res = {
            "consensus": parliament_consensus if not is_1d else parliament_consensus.squeeze(0),
            "left_consensus": left_consensus if not is_1d else left_consensus.squeeze(0),
            "right_consensus": right_consensus if not is_1d else right_consensus.squeeze(0),
            "readout_vector": all_z if not is_1d else all_z.squeeze(0),
            "state": psi_t if not is_1d else psi_t.squeeze(0),
        }
        return res

    def __repr__(self) -> str:
        return (
            f"BiomorphicResonantBrain(in_features={self.in_features}, "
            f"left_qubits={self.num_left_qubits}, right_qubits={self.num_right_qubits}, "
            f"neuromodulation={self.enable_neuromodulation}, oxygenation={self.enable_oxygenation})"
        )
