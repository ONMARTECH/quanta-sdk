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
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from quanta.torch import ops

__all__ = ["BiomorphicResonantBrain", "QuantumREMSleep", "QuantumZenoAttention"]


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


class QuantumREMSleep(nn.Module):
    """Offline closed-system quantum REM sleep annealing module.

    Orthogonalizes stored memory statevectors from prior tasks through uncoupled
    closed-system Hamiltonian evolution (x = 0), preventing catastrophic forgetting
    in continual learning without requiring external data replay.

    Args:
        brain_module: Reference to the underlying BiomorphicResonantBrain module.
        sleep_cycles: Default number of annealing optimization cycles (default: 10).
        learning_rate: Gradient descent step size for sleep consolidation (default: 0.01).
        orthogonalization_weight: Scaling weight for the orthogonalization loss (default: 1.0).
        device: PyTorch device ('cpu', 'mps', or torch.device).
        dtype: Real floating-point dtype.
    """

    def __init__(
        self,
        brain_module: BiomorphicResonantBrain,
        sleep_cycles: int = 10,
        learning_rate: float = 0.01,
        orthogonalization_weight: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.brain_module = brain_module
        self.sleep_cycles = sleep_cycles
        self.learning_rate = learning_rate
        self.orthogonalization_weight = orthogonalization_weight

        target_dev = ops.resolve_device(device) if device is not None else None
        dev = target_dev if target_dev is not None else brain_module.W_left.device
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
        self.memory_states: list[torch.Tensor] = []

    def register_memory_state(self, state: torch.Tensor) -> None:
        """Stores prototype statevectors from prior tasks for sleep consolidation.

        Args:
            state: Quantum statevector of shape [dim] or batch of statevectors [batch, dim].
        """
        dev = self.brain_module.W_left.device
        cdtype = self.brain_module.complex_dtype
        st = state.detach().to(device=dev, dtype=cdtype)
        if st.dim() == 1:
            norm = torch.linalg.norm(st)
            if norm > 1e-12:
                st = st / norm
            self.memory_states.append(st.clone())
        elif st.dim() == 2:
            for s in st:
                norm = torch.linalg.norm(s)
                if norm > 1e-12:
                    s = s / norm
                self.memory_states.append(s.clone())
        else:
            raise ValueError(f"Expected 1D or 2D state tensor, got shape {tuple(state.shape)}")

    def clear_memory_states(self) -> None:
        """Clears all stored memory prototype statevectors."""
        self.memory_states.clear()

    def sleep(
        self,
        memory_states: Sequence[torch.Tensor] | None = None,
        cycles: int | None = None,
    ) -> dict[str, Any]:
        """Executes closed-system unitary annealing at x = 0 to orthogonalize representations.

        Evaluates pairwise Hilbert-Schmidt overlap:
            L_ortho = sum_{j < k} |<psi_j | psi_k>|^2
        and optimizes brain parameters (J_right, J_callosum, h_left) via gradient
        descent to drive memory representations into mutually orthogonal subspaces.

        Args:
            memory_states: Optional explicit sequence of statevectors. If None, uses
                registered prototype states.
            cycles: Optional override for the number of annealing cycles.

        Returns:
            Dictionary containing:
                - 'initial_overlap': Pairwise overlap before sleep annealing.
                - 'final_overlap': Pairwise overlap after sleep annealing.
                - 'cycles': Number of annealing cycles executed.
                - 'loss_history': List of float losses per cycle.
                - 'retention_estimate': Theoretical task retention estimate in [0, 1].
        """
        active_states: list[torch.Tensor]
        if memory_states is not None:
            active_states = [s.detach() for s in memory_states]
        else:
            active_states = [s.detach() for s in self.memory_states]

        if len(active_states) < 2:
            raise ValueError(
                f"At least two memory states are required for sleep orthogonalization, "
                f"got {len(active_states)}."
            )

        num_cycles = cycles if cycles is not None else self.sleep_cycles
        dev = self.brain_module.W_left.device
        real_dtype = self.brain_module.W_left.dtype
        cdtype = self.brain_module.complex_dtype
        M = len(active_states)

        # Standardize states on device and cdtype
        normed_states = []
        for s in active_states:
            s_t = s.to(device=dev, dtype=cdtype)
            norm = torch.linalg.norm(s_t)
            if norm > 1e-12:
                s_t = s_t / norm
            normed_states.append(s_t)

        # Select trainable parameters in the brain module
        trainable_params: list[nn.Parameter] = []
        for p in [
            self.brain_module.J_right,
            self.brain_module.J_callosum,
            self.brain_module.h_left,
        ]:
            if isinstance(p, nn.Parameter) and p.requires_grad:
                trainable_params.append(p)

        optimizer = (
            torch.optim.Adam(trainable_params, lr=self.learning_rate)
            if trainable_params
            else None
        )

        loss_history: list[float] = []
        initial_overlap = 0.0

        for step in range(num_cycles):
            if optimizer is not None:
                optimizer.zero_grad()

            x_zero = torch.zeros((1, self.brain_module.in_features), device=dev, dtype=real_dtype)
            H_total, eff_time = self.brain_module._build_hamiltonian(x_zero)
            H_free = H_total[0]  # [dim, dim]

            eigvals, eigvecs = torch.linalg.eigh(H_free)  # [dim], [dim, dim]

            # Project states onto eigenbasis: c_j = eigvecs^H @ psi_j
            c_matrix = eigvecs.mH @ torch.stack(normed_states, dim=1)  # [dim, M]
            p_matrix = torch.abs(c_matrix) ** 2  # [dim, M]

            # Dynamic consolidation states
            evolved_states = []
            for j in range(M):
                t_j = eff_time[0, 0] * float(j + 1) / float(M)
                phase_j = torch.exp(-1j * (eigvals * t_j))
                psi_j_evolved = eigvecs @ (phase_j * c_matrix[:, j])
                evolved_states.append(psi_j_evolved)
            evolved_matrix = torch.stack(evolved_states, dim=1)  # [dim, M]

            pair_count = M * (M - 1) // 2
            loss_ortho = torch.tensor(0.0, device=dev, dtype=real_dtype)

            for j in range(M):
                for k in range(j + 1, M):
                    # Ergodic eigenspace overlap (Theorem 1 Eq. 74)
                    ergodic_overlap = torch.sum(p_matrix[:, j] * p_matrix[:, k])
                    # Dynamic state overlap |<psi_j(t) | psi_k(t)>|^2
                    dyn_overlap = (
                        torch.abs(torch.vdot(evolved_matrix[:, j], evolved_matrix[:, k])) ** 2
                    )
                    loss_ortho = loss_ortho + (ergodic_overlap + dyn_overlap) * 0.5

            loss_ortho = (loss_ortho / pair_count) * self.orthogonalization_weight

            if step == 0:
                initial_overlap = float(loss_ortho.item())

            loss_history.append(float(loss_ortho.item()))

            if optimizer is not None and trainable_params:
                loss_ortho.backward()
                optimizer.step()

        # Final evaluation after updates
        with torch.no_grad():
            x_zero = torch.zeros((1, self.brain_module.in_features), device=dev, dtype=real_dtype)
            H_total, eff_time = self.brain_module._build_hamiltonian(x_zero)
            eigvals, eigvecs = torch.linalg.eigh(H_total[0])
            c_matrix = eigvecs.mH @ torch.stack(normed_states, dim=1)
            p_matrix = torch.abs(c_matrix) ** 2

            evolved_states = []
            for j in range(M):
                t_j = eff_time[0, 0] * float(j + 1) / float(M)
                phase_j = torch.exp(-1j * (eigvals * t_j))
                psi_j_evolved = eigvecs @ (phase_j * c_matrix[:, j])
                evolved_states.append(psi_j_evolved)
            evolved_matrix = torch.stack(evolved_states, dim=1)

            final_loss = torch.tensor(0.0, device=dev, dtype=real_dtype)
            for j in range(M):
                for k in range(j + 1, M):
                    ergodic_overlap = torch.sum(p_matrix[:, j] * p_matrix[:, k])
                    dyn_overlap = (
                        torch.abs(torch.vdot(evolved_matrix[:, j], evolved_matrix[:, k])) ** 2
                    )
                    final_loss = final_loss + (ergodic_overlap + dyn_overlap) * 0.5
            final_overlap = float(
                ((final_loss / pair_count) * self.orthogonalization_weight).item()
            )

            if memory_states is None:
                self.memory_states = [evolved_matrix[:, j].clone() for j in range(M)]

        retention_estimate = max(0.0, min(1.0, 1.0 - final_overlap))

        return {
            "initial_overlap": initial_overlap,
            "final_overlap": final_overlap,
            "cycles": num_cycles,
            "loss_history": loss_history,
            "retention_estimate": retention_estimate,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Routes input through the underlying biomorphic quantum brain module.

        Args:
            x: Input tensor of shape [batch_size, in_features] or [in_features].

        Returns:
            Brain module consensus output dictionary.
        """
        return self.brain_module.forward(x)

    def __repr__(self) -> str:
        return (
            f"QuantumREMSleep(sleep_cycles={self.sleep_cycles}, "
            f"learning_rate={self.learning_rate}, "
            f"orthogonalization_weight={self.orthogonalization_weight}, "
            f"stored_states={len(self.memory_states)})"
        )


class QuantumZenoAttention(nn.Module):
    """Quantum Zeno and Anti-Zeno dynamic cognitive attention mechanism.

    Balances attentional focus pinning (Quantum Zeno Effect) and divergent
    exploratory tunneling with phase kickback (Anti-Zeno Effect) modulated by
    internal or external dopamine signals.

    Args:
        dim: Feature dimension / Hilbert space state dimension.
        observation_frequency: Projective monitoring frequency nu_obs (default: 10.0).
        dopamine_coupling: Coupling strength lambda_D to dopamine surges (default: 0.5).
        num_heads: Number of attention heads (default: 1, must divide dim).
        learnable_frequency: Whether observation_frequency is trainable parameter.
        device: PyTorch device ('cpu', 'mps', or torch.device).
        dtype: Real floating-point dtype.
    """

    def __init__(
        self,
        dim: int,
        observation_frequency: float = 10.0,
        dopamine_coupling: float = 0.5,
        num_heads: int = 1,
        learnable_frequency: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

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

        # 1. Observation frequency parameter
        if learnable_frequency:
            self.obs_freq = nn.Parameter(
                torch.tensor(float(observation_frequency), device=dev, dtype=self.real_dtype)
            )
        else:
            self.register_buffer(
                "obs_freq",
                torch.tensor(float(observation_frequency), device=dev, dtype=self.real_dtype),
            )

        # 2. Dopamine coupling parameter
        self.dopamine_coupling = nn.Parameter(
            torch.tensor(float(dopamine_coupling), device=dev, dtype=self.real_dtype)
        )

        # 3. Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)
        self.k_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)
        self.v_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)
        self.kickback_proj = nn.Linear(dim, dim, bias=False, device=dev, dtype=self.real_dtype)

        # 4. Internal dopamine arousal sensor
        self.dopamine_sensor = nn.Linear(dim, 1, device=dev, dtype=self.real_dtype)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.orthogonal_(self.kickback_proj.weight)
        nn.init.kaiming_uniform_(self.dopamine_sensor.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dopamine_sensor.bias)

    def forward(
        self,
        x: torch.Tensor,
        dopamine: torch.Tensor | float | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass implementing Zeno focus pinning vs Anti-Zeno phase tunneling.

        Args:
            x: Input tensor of shape [dim], [batch, dim], or [batch, seq_len, dim].
            dopamine: Optional external dopamine signal tensor or scalar. If None,
                estimated internally via dopamine_sensor.
            return_diagnostics: If True, returns full diagnostic dictionary.

        Returns:
            Output tensor of same shape as x, or dictionary if return_diagnostics=True.
        """
        orig_dim = x.dim()
        dev = self.q_proj.weight.device
        real_dtype = self.q_proj.weight.dtype

        if x.device != dev or x.dtype != real_dtype:
            x = x.to(device=dev, dtype=real_dtype)

        if orig_dim == 1:
            x_3d = x.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        elif orig_dim == 2:
            x_3d = x.unsqueeze(1)  # [B, 1, dim]
        elif orig_dim == 3:
            x_3d = x
        else:
            raise ValueError(f"Expected 1D, 2D, or 3D tensor, got shape {tuple(x.shape)}")

        B, L, _ = x_3d.shape

        # 1. Evaluate dopamine arousal d
        if dopamine is None:
            d = torch.sigmoid(self.dopamine_sensor(x_3d))  # [B, L, 1]
        elif isinstance(dopamine, (int, float)):
            d = torch.full((B, L, 1), float(dopamine), device=dev, dtype=real_dtype)
        elif torch.is_tensor(dopamine):
            d_in = dopamine.to(device=dev, dtype=real_dtype)
            if d_in.dim() == 0:
                d = d_in.expand(B, L, 1)
            elif d_in.dim() == 1:
                if d_in.shape[0] == B:
                    d = d_in.unsqueeze(1).unsqueeze(2).expand(B, L, 1)
                else:
                    d = d_in.expand(B, L, 1)
            elif d_in.dim() == 2:
                d = d_in.unsqueeze(1).expand(B, L, 1) if d_in.shape[1] == 1 else d_in.unsqueeze(-1)
            else:
                d = d_in
        else:
            raise TypeError(f"Unsupported dopamine type: {type(dopamine)}")

        # 2. Dynamic effective observation frequency
        # nu_eff = softplus(nu_obs) * clamp(1.0 - tanh(|lambda_D| * d), min=0.01)
        nu_obs_pos = torch.clamp(nn.functional.softplus(self.obs_freq), min=1e-5)
        dopamine_damping = torch.clamp(
            1.0 - torch.tanh(torch.abs(self.dopamine_coupling) * d), min=0.01
        )
        nu_eff = nu_obs_pos * dopamine_damping  # [B, L, 1]

        # 3. Zeno pinning probability
        # P_zeno = exp(-1.0 / nu_eff)
        P_zeno = torch.exp(-1.0 / nu_eff)  # [B, L, 1]

        # 4. Pinned state: working memory baseline h_pinned = Q(x)
        h_pinned = self.q_proj(x_3d)  # [B, L, dim]

        # 5. Exploratory candidate state via multi-head attention
        Q = self.q_proj(x_3d)  # [B, L, dim]
        K = self.k_proj(x_3d)  # [B, L, dim]
        V = self.v_proj(x_3d)  # [B, L, dim]

        Q_h = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K_h = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V_h = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V_h)
        h_explore = attn_out.transpose(1, 2).contiguous().view(B, L, self.dim)

        # 6. Phase kickback: phi = pi / nu_eff
        phi = math.pi / nu_eff  # [B, L, 1]
        kickback_term = self.kickback_proj(h_explore)  # [B, L, dim]
        h_tunnel = torch.cos(phi) * h_explore + torch.sin(phi) * kickback_term  # [B, L, dim]

        # 7. Coherent superposition
        # h_out = P_zeno * h_pinned + (1.0 - P_zeno) * h_tunnel
        h_out = P_zeno * h_pinned + (1.0 - P_zeno) * h_tunnel  # [B, L, dim]

        res_out: torch.Tensor
        res_P_zeno: torch.Tensor
        res_nu_eff: torch.Tensor
        res_phi: torch.Tensor
        res_d: torch.Tensor
        res_pinned: torch.Tensor
        res_tunnel: torch.Tensor
        res_explore: torch.Tensor

        # Restore original dimension
        if orig_dim == 1:
            res_out = h_out.squeeze(0).squeeze(0)
            res_P_zeno = P_zeno.squeeze(0).squeeze(0)
            res_nu_eff = nu_eff.squeeze(0).squeeze(0)
            res_phi = phi.squeeze(0).squeeze(0)
            res_d = d.squeeze(0).squeeze(0)
            res_pinned = h_pinned.squeeze(0).squeeze(0)
            res_tunnel = h_tunnel.squeeze(0).squeeze(0)
            res_explore = h_explore.squeeze(0).squeeze(0)
        elif orig_dim == 2:
            res_out = h_out.squeeze(1)
            res_P_zeno = P_zeno.squeeze(1)
            res_nu_eff = nu_eff.squeeze(1)
            res_phi = phi.squeeze(1)
            res_d = d.squeeze(1)
            res_pinned = h_pinned.squeeze(1)
            res_tunnel = h_tunnel.squeeze(1)
            res_explore = h_explore.squeeze(1)
        else:
            res_out = h_out
            res_P_zeno = P_zeno
            res_nu_eff = nu_eff
            res_phi = phi
            res_d = d
            res_pinned = h_pinned
            res_tunnel = h_tunnel
            res_explore = h_explore

        if return_diagnostics:
            return {
                "output": res_out,
                "P_zeno": res_P_zeno,
                "nu_eff": res_nu_eff,
                "phi": res_phi,
                "dopamine": res_d,
                "h_pinned": res_pinned,
                "h_tunnel": res_tunnel,
                "h_explore": res_explore,
            }

        return res_out

    def __repr__(self) -> str:
        return (
            f"QuantumZenoAttention(dim={self.dim}, "
            f"obs_freq={float(self.obs_freq.detach()):.2f}, "
            f"dopamine_coupling={float(self.dopamine_coupling.detach()):.2f}, "
            f"num_heads={self.num_heads})"
        )

