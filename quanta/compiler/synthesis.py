"""quanta.compiler.synthesis — Exact 1-Qubit ZYZ/Heron & 2-Qubit Cartan/KAK Synthesis.

Provides analytical, singularity-safe unitary synthesis routines:
  - ``zyz_decompose(U)``: Exact Euler decomposition U = exp(i*alpha) * RZ(beta) * RY(gamma) * RZ(delta)
    for any U in U(2), handling cos(gamma/2) -> 0 and sin(gamma/2) -> 0 gimbal-lock singularities.
  - ``heron_1q_decompose(U)``: Exact IBM Heron native decomposition into {RZ, SX, X} with at most
    2 SX and 3 RZ gates.
  - ``kak_decompose(U)``: Exact 2-qubit Cartan / KAK SU(4) decomposition using the Magic Basis and
    Vatan-Williams PRA 69, 032315 (2004) construction, synthesizing any U in U(4) with <= 3 CX gates
    (and 0, 1, or 2 CX gates when algebraically optimal) to machine precision (~1e-15).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quanta.core.types import Instruction

__all__ = [
    "KAKDecomposition",
    "zyz_decompose",
    "heron_1q_decompose",
    "kak_coefficients",
    "kak_decompose",
    "synthesize_unitary_matrix",
]

_SQRT2_INV = 1.0 / math.sqrt(2.0)

_I2 = np.eye(2, dtype=np.complex128)
_X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_H2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) * _SQRT2_INV
_S2 = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
_SDG2 = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
_SX2 = 0.5 * np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=np.complex128)
_CX4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.complex128,
)

# Magic basis B mapping SU(2) x SU(2) <-> SO(4):
# B^dagger (X x X) B = diag(1, 1, -1, -1)
# B^dagger (Y x Y) B = diag(-1, 1, -1, 1)
# B^dagger (Z x Z) B = diag(1, -1, -1, 1)
_MAGIC_B = _SQRT2_INV * np.array(
    [
        [1.0, 0.0, 0.0, 1.0j],
        [0.0, 1.0j, 1.0, 0.0],
        [0.0, 1.0j, -1.0, 0.0],
        [1.0, 0.0, 0.0, -1.0j],
    ],
    dtype=np.complex128,
)
_MAGIC_B_DAG = _MAGIC_B.conj().T


@dataclass(frozen=True)
class KAKDecomposition:
    """Structured Cartan / KAK decomposition of U in U(4).

    Satisfies:
        U == exp(i * global_phase) * kron(left_q0, left_q1)
             @ exp(i * (cx * XX + cy * YY + cz * ZZ))
             @ kron(right_q0, right_q1)
    """

    global_phase: float
    left_q0: np.ndarray
    left_q1: np.ndarray
    interaction: tuple[float, float, float]
    right_q0: np.ndarray
    right_q1: np.ndarray


def _rx_mat(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def _ry_mat(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz_mat(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-0.5j * theta), 0.0],
            [0.0, np.exp(0.5j * theta)],
        ],
        dtype=np.complex128,
    )


def _wrap_angle(angle: float) -> float:
    """Wraps an angle into (-pi, pi], mapping -pi cleanly to +pi."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi + 1e-15:
        return math.pi
    return wrapped


def _validate_unitary(U: np.ndarray, dim: int, tol: float = 1e-7) -> np.ndarray:
    mat = np.asarray(U, dtype=np.complex128)
    if mat.shape != (dim, dim):
        raise ValueError(f"Expected ({dim}, {dim}) unitary matrix, got shape {mat.shape}.")
    err = float(np.max(np.abs(mat.conj().T @ mat - np.eye(dim, dtype=np.complex128))))
    if err > tol:
        raise ValueError(
            f"Input matrix is not unitary within tolerance {tol}: max|U^dagger U - I| = {err:.3e}"
        )
    return mat


def zyz_decompose(U: np.ndarray) -> tuple[float, float, float, float]:
    """Computes exact ZYZ Euler angles (alpha, beta, gamma, delta) for any U in U(2).

    Satisfies:
        U = exp(i * alpha) * RZ(beta) @ RY(gamma) @ RZ(delta)

    Handles both gimbal-lock singularities:
      - sin(gamma / 2) ~ 0 (diagonal U, gamma = 0)
      - cos(gamma / 2) ~ 0 (off-diagonal U, gamma = pi)

    Args:
        U: 2x2 unitary matrix.

    Returns:
        Tuple (alpha, beta, gamma, delta) in radians with gamma in [0, pi].
    """
    mat = _validate_unitary(U, 2)

    det_u = np.linalg.det(mat)
    alpha = 0.5 * float(np.angle(det_u))
    V = np.exp(-1.0j * alpha) * mat  # V in SU(2), det(V) = +1

    c = math.sqrt(max(0.0, 0.5 * (abs(V[0, 0]) ** 2 + abs(V[1, 1]) ** 2)))
    s = math.sqrt(max(0.0, 0.5 * (abs(V[1, 0]) ** 2 + abs(V[0, 1]) ** 2)))
    gamma = 2.0 * math.atan2(s, c)

    if s < 1e-14:
        # Gimbal lock at gamma = 0: V = diag(e^{-i*beta/2}, e^{i*beta/2})
        gamma = 0.0
        delta = 0.0
        phi_plus = float(np.angle(V[1, 1] + np.conj(V[0, 0])))
        beta = _wrap_angle(2.0 * phi_plus)
    elif c < 1e-14:
        # Gimbal lock at gamma = pi: V = [[0, -e^{-i*beta/2}], [e^{i*beta/2}, 0]]
        gamma = math.pi
        delta = 0.0
        phi_minus = float(np.angle(V[1, 0] - np.conj(V[0, 1])))
        beta = _wrap_angle(2.0 * phi_minus)
    else:
        phi_plus = float(np.angle(V[1, 1] + np.conj(V[0, 0])))
        phi_minus = float(np.angle(V[1, 0] - np.conj(V[0, 1])))
        beta = _wrap_angle(phi_plus + phi_minus)
        delta = _wrap_angle(phi_plus - phi_minus)

    # Align alpha exactly with wrapped (beta, gamma, delta) to account for SU(2) 4*pi spinor double-cover
    v_wrapped = _rz_mat(beta) @ _ry_mat(gamma) @ _rz_mat(delta)
    alpha = float(np.angle(np.trace(v_wrapped.conj().T @ mat) / 2.0))

    return (alpha, beta, gamma, delta)


def _zyz_instructions(U: np.ndarray, qubit: int, tol: float = 1e-12) -> list[Instruction]:
    """Emits left-to-right {RZ, RY} instructions implementing U up to global phase."""
    _, beta, gamma, delta = zyz_decompose(U)
    if abs(gamma) < tol:
        merged = _wrap_angle(beta + delta)
        if abs(merged) > tol:
            return [Instruction("RZ", (qubit,), (merged,))]
        return []

    insts: list[Instruction] = []
    if abs(_wrap_angle(delta)) > tol:
        insts.append(Instruction("RZ", (qubit,), (_wrap_angle(delta),)))
    insts.append(Instruction("RY", (qubit,), (gamma,)))
    if abs(_wrap_angle(beta)) > tol:
        insts.append(Instruction("RZ", (qubit,), (_wrap_angle(beta),)))
    return insts


def heron_1q_decompose(
    U: np.ndarray,
    qubit: int = 0,
    tol: float = 1e-12,
) -> list[Instruction]:
    """Decomposes any U in U(2) into IBM Heron native 1Q gates {RZ, SX, X} (up to global phase).

    Uses the exact identity:
        RY(gamma) ~= RZ(pi) @ SX @ RZ(gamma + pi) @ SX @ RZ(0)
    merged with outer Z rotations RZ(beta) @ RY(gamma) @ RZ(delta):
        U ~= RZ(beta + pi) @ SX @ RZ(gamma + pi) @ SX @ RZ(delta)
    producing at most 2 SX and 3 RZ gates (and fewer at special angles gamma in {0, pi/2, pi}).

    Args:
        U: 2x2 unitary matrix.
        qubit: Target qubit index for emitted Instructions.
        tol: Angle pruning threshold.

    Returns:
        List of Instruction objects in left-to-right circuit execution order.
    """
    _, beta, gamma, delta = zyz_decompose(U)
    insts: list[Instruction] = []

    def _emit_rz(angle: float) -> None:
        # Check modulo 2*pi identity up to global phase (-1 global phase at +-2*pi)
        mod_2pi = (angle + math.pi) % (2.0 * math.pi) - math.pi
        if abs(mod_2pi) > tol:
            insts.append(Instruction("RZ", (qubit,), (_wrap_angle(angle),)))

    if abs(gamma) < tol:
        # Pure Z rotation: 0 SX, <= 1 RZ
        _emit_rz(beta + delta)
        return insts

    if abs(gamma - math.pi) < tol:
        # RY(pi) ~= RZ(pi) @ X => RZ(beta) @ RY(pi) @ RZ(delta) = RZ(beta - delta) @ RY(pi)
        # ~= RZ(beta - delta + pi) @ X: 0 SX, 1 X, <= 1 RZ
        insts.append(Instruction("X", (qubit,), ()))
        _emit_rz(beta - delta + math.pi)
        return insts

    if abs(gamma - 0.5 * math.pi) < tol:
        # RY(pi/2) = RZ(pi/2) @ RX(pi/2) @ RZ(-pi/2) ~= RZ(pi/2) @ SX @ RZ(-pi/2)
        # Merged: RZ(beta + pi/2) @ SX @ RZ(delta - pi/2): 1 SX, <= 2 RZ
        _emit_rz(delta - 0.5 * math.pi)
        insts.append(Instruction("SX", (qubit,), ()))
        _emit_rz(beta + 0.5 * math.pi)
        return insts

    # General U(2): U ~= RZ(beta + pi) @ SX @ RZ(gamma + pi) @ SX @ RZ(delta): 2 SX, <= 3 RZ
    _emit_rz(delta)
    insts.append(Instruction("SX", (qubit,), ()))
    _emit_rz(gamma + math.pi)
    insts.append(Instruction("SX", (qubit,), ()))
    _emit_rz(beta + math.pi)
    return insts


def _simultaneous_orthogonal_takagi(M: np.ndarray) -> np.ndarray:
    """Finds Q in SO(4) such that Q^T M Q is diagonal for symmetric unitary M = U_B^T U_B.

    Since M is complex symmetric and unitary:
        M = R + i*S  with R = Re(M), S = Im(M) real symmetric, [R, S] = 0.
    For any real angle eta, T(eta) = cos(eta)*R + sin(eta)*S = Re(exp(-i*eta)*M) is real symmetric
    with eigenvalues cos(phi_k - eta). Distinct complex eigenvalues exp(i*phi_a) != exp(i*phi_b)
    collide in T(eta) only at the single angle eta == (phi_a + phi_b)/2 (mod pi), of which there
    are at most C(4, 2) = 6 bad angles. By the Pigeonhole Principle over 19 uniform angles in
    [0, pi), at least one T(eta) separates all distinct complex eigenvalues while any truly
    degenerate subspace of M is automatically an eigenspace of both R and S.
    """
    R = 0.5 * (np.real(M) + np.real(M).T)
    S = 0.5 * (np.imag(M) + np.imag(M).T)

    best_Q: np.ndarray | None = None
    best_off = float("inf")
    for m in range(19):
        eta = m * math.pi / 19.0
        T = math.cos(eta) * R + math.sin(eta) * S
        _, Q_cand = np.linalg.eigh(T)
        D = Q_cand.T @ M @ Q_cand
        off = float(np.linalg.norm(D - np.diag(np.diag(D))))
        if off < best_off:
            best_off = off
            best_Q = Q_cand
            if best_off < 1e-14:
                break

    assert best_Q is not None
    Q = best_Q.copy()
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _kronecker_factor_su2(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factors W = A x B in SU(2) x SU(2) via rank-1 reshaped SVD."""
    # W[2*i + k, 2*j + l] = A[i, j] * B[k, l]
    reshaped = W.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    u_svd, s_svd, vh_svd = np.linalg.svd(reshaped)
    scale = math.sqrt(float(s_svd[0]))
    A = u_svd[:, 0].reshape(2, 2) * scale
    B = vh_svd[0, :].reshape(2, 2) * scale

    det_a = np.linalg.det(A)
    phase_a = np.exp(-0.5j * float(np.angle(det_a))) / math.sqrt(max(abs(det_a), 1e-15))
    A = A * phase_a
    B = B / phase_a

    # Enforce det(B) = +1 up to +-1 sign ambiguity between A and B
    if np.real(np.linalg.det(B)) < 0:
        B = -B
        A = -A
    return A, B


def kak_coefficients(U: np.ndarray) -> KAKDecomposition:
    """Computes the exact Magic-Basis Cartan / KAK decomposition of any U in U(4).

    Args:
        U: 4x4 unitary matrix.

    Returns:
        KAKDecomposition with exact local SU(2) factors and interaction coordinates.
    """
    mat = _validate_unitary(U, 4)

    # 1. Normalize determinant into SU(4): det(V) = +1
    det_u = np.linalg.det(mat)
    alpha = 0.25 * float(np.angle(det_u))
    V = np.exp(-1.0j * alpha) * mat
    if np.real(np.linalg.det(V)) < 0:
        # Correct branch of 4th root if det(V) == -1
        alpha += 0.5 * math.pi
        V = np.exp(-1.0j * alpha) * mat
    # Refine phase via trace of det(V)^(1/4) for roundoff
    det_v = np.linalg.det(V)
    d_alpha = 0.25 * float(np.angle(det_v))
    alpha += d_alpha
    V = np.exp(-1.0j * d_alpha) * V

    # 2. Transform into Magic Basis: U_B = B^dagger @ V @ B
    U_B = _MAGIC_B_DAG @ V @ _MAGIC_B
    M = U_B.T @ U_B  # Complex symmetric unitary

    # 3. Diagonalize M = Q D_2 Q^T with Q in SO(4), set K2 = Q^T in SO(4)
    Q = _simultaneous_orthogonal_takagi(M)
    K2 = Q.T
    D2_diag = np.diag(K2 @ M @ K2.T)
    thetas = 0.5 * np.angle(D2_diag)

    # 4. Recover K1 in O(4) via W = U_B @ K2^T; since W^* = W @ D2^dagger, W @ D0^dagger is purely real!
    W = U_B @ K2.T
    K1 = np.real(W @ np.diag(np.exp(-1.0j * thetas)))
    # Re-orthonormalize K1 via polar SVD in case of tiny numerical drift
    u_k, _, vt_k = np.linalg.svd(K1)
    K1 = u_k @ vt_k
    if np.linalg.det(K1) < 0:
        K1[:, 3] = -K1[:, 3]
        thetas[3] += math.pi

    # 5. Enforce sum(thetas) == 0 mod 2*pi (since det(U_B) = det(V) = 1 = det(K1)*det(D)*det(K2) = exp(i*sum(thetas)))
    mean_theta = 0.25 * float(np.sum(thetas))
    # exp(i * mean_theta) is a 4th root of unity (+1, -1, +i, -i); absorb into global_phase
    thetas_su4 = thetas - mean_theta
    alpha += mean_theta

    # Magic basis diagonal relation:
    # B^dagger exp(i*(cx*XX + cy*YY + cz*ZZ)) B = diag(e^{i*(cx - cy + cz)}, e^{i*(cx + cy - cz)},
    #                                                  e^{i*(-cx - cy - cz)}, e^{i*(-cx + cy + cz)})
    cx = 0.5 * float(thetas_su4[0] + thetas_su4[1])
    cy = 0.5 * float(thetas_su4[1] + thetas_su4[3])
    cz = 0.5 * float(thetas_su4[0] + thetas_su4[3])

    # 6. Map K1, K2 in SO(4) back to SU(2) x SU(2)
    L_prod = _MAGIC_B @ K1 @ _MAGIC_B_DAG
    R_prod = _MAGIC_B @ K2 @ _MAGIC_B_DAG
    left_q0, left_q1 = _kronecker_factor_su2(L_prod)
    right_q0, right_q1 = _kronecker_factor_su2(R_prod)

    # 7. Reduce each coordinate into (-pi/4, pi/4] using exp(i*(pi/2)*(P x P)) = i*(P x P)
    paulis = (_X2, _Y2, _Z2)
    coords = [cx, cy, cz]
    for axis in range(3):
        while coords[axis] > 0.25 * math.pi + 1e-12:
            coords[axis] -= 0.5 * math.pi
            alpha += 0.5 * math.pi
            left_q0 = left_q0 @ paulis[axis]
            left_q1 = left_q1 @ paulis[axis]
        while coords[axis] <= -0.25 * math.pi - 1e-12:
            coords[axis] += 0.5 * math.pi
            alpha -= 0.5 * math.pi
            left_q0 = left_q0 @ paulis[axis]
            left_q1 = left_q1 @ paulis[axis]

    # Exact global phase alignment against original U via trace inner product
    cx_r, cy_r, cz_r = coords[0], coords[1], coords[2]
    d_magic = np.diag(
        [
            np.exp(1.0j * (cx_r - cy_r + cz_r)),
            np.exp(1.0j * (cx_r + cy_r - cz_r)),
            np.exp(1.0j * (-cx_r - cy_r - cz_r)),
            np.exp(1.0j * (-cx_r + cy_r + cz_r)),
        ]
    )
    entangler = _MAGIC_B @ d_magic @ _MAGIC_B_DAG
    recon_no_phase = np.kron(left_q0, left_q1) @ entangler @ np.kron(right_q0, right_q1)
    overlap = np.trace(recon_no_phase.conj().T @ mat) / 4.0
    exact_phase = float(np.angle(overlap))

    return KAKDecomposition(
        global_phase=exact_phase,
        left_q0=left_q0,
        left_q1=left_q1,
        interaction=(cx_r, cy_r, cz_r),
        right_q0=right_q0,
        right_q1=right_q1,
    )


def kak_decompose(
    U: np.ndarray,
    qubits: tuple[int, int] = (0, 1),
    tol: float = 1e-12,
) -> list[Instruction]:
    """Synthesizes any 2-qubit unitary U in U(4) into <= 3 CX gates + 1Q {RZ, RY} rotations.

    Achieves optimal entangling gate counts across Weyl chamber strata:
      - 0 CX for local product unitaries (cx = cy = cz = 0)
      - 1 CX for CNOT-equivalent unitaries (e.g. CX, CZ, CY, CH, ECR)
      - 2 CX when at least one Cartan coordinate vanishes (e.g. iSWAP, RXX, RZZ, CRX, CRY, CRZ, CP)
      - 3 CX for general Haar-random SU(4) unitaries via Vatan-Williams PRA 69, 032315 (2004)

    Adjacent single-qubit gates on each wire are fused into minimal 1Q ZYZ rotations.

    Args:
        U: 4x4 unitary matrix.
        qubits: Physical/logical qubit pair (q0, q1).
        tol: Numerical zero threshold for Cartan coordinates and rotation angles.

    Returns:
        List of Instruction objects in left-to-right circuit execution order.
    """
    q0, q1 = qubits
    kak = kak_coefficients(U)
    cx, cy, cz = kak.interaction
    L0, L1 = kak.left_q0, kak.left_q1
    R0, R1 = kak.right_q0, kak.right_q1

    zero_x = abs(cx) < tol
    zero_y = abs(cy) < tol
    zero_z = abs(cz) < tol

    # Case 0: Local unitary (0 CX)
    if zero_x and zero_y and zero_z:
        return _zyz_instructions(L0 @ R0, q0, tol) + _zyz_instructions(L1 @ R1, q1, tol)

    # Case 1: Single-CNOT equivalent (two coordinates 0, remaining coordinate == +-pi/4)
    # Using exp(i*s*(pi/4)*(Z x Z)) ~= (S^{-s} x (S^{-s} H)) @ CX(0,1) @ (I x H) for s in {+1, -1}
    if zero_x and zero_y and abs(abs(cz) - 0.25 * math.pi) < tol:
        s_gate = _SDG2 if cz > 0 else _S2
        pre0, pre1 = R0, _H2 @ R1
        post0, post1 = L0 @ s_gate, L1 @ s_gate @ _H2
        return (
            _zyz_instructions(pre0, q0, tol)
            + _zyz_instructions(pre1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(post0, q0, tol)
            + _zyz_instructions(post1, q1, tol)
        )

    if zero_y and zero_z and abs(abs(cx) - 0.25 * math.pi) < tol:
        # X = H Z H on both qubits; (I x H) @ (H x H) = H x I
        s_gate = _SDG2 if cx > 0 else _S2
        pre0, pre1 = _H2 @ R0, R1
        post0, post1 = L0 @ _H2 @ s_gate, L1 @ _H2 @ s_gate @ _H2
        return (
            _zyz_instructions(pre0, q0, tol)
            + _zyz_instructions(pre1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(post0, q0, tol)
            + _zyz_instructions(post1, q1, tol)
        )

    if zero_x and zero_z and abs(abs(cy) - 0.25 * math.pi) < tol:
        # Y = RX(-pi/2) Z RX(pi/2) on both qubits
        rx_pos = _rx_mat(0.5 * math.pi)
        rx_neg = _rx_mat(-0.5 * math.pi)
        s_gate = _SDG2 if cy > 0 else _S2
        pre0, pre1 = rx_pos @ R0, _H2 @ rx_pos @ R1
        post0, post1 = L0 @ rx_neg @ s_gate, L1 @ rx_neg @ s_gate @ _H2
        return (
            _zyz_instructions(pre0, q0, tol)
            + _zyz_instructions(pre1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(post0, q0, tol)
            + _zyz_instructions(post1, q1, tol)
        )

    # Case 2: At least one Cartan coordinate is 0 -> exact 2-CNOT synthesis!
    # Identity: CX(0,1) @ (RX(-2*a) x RZ(-2*b)) @ CX(0,1) == exp(i * (a * XX + b * ZZ))
    if zero_y:
        pre0, pre1 = R0, R1
        mid0, mid1 = _rx_mat(-2.0 * cx), _rz_mat(-2.0 * cz)
        post0, post1 = L0, L1
        return (
            _zyz_instructions(pre0, q0, tol)
            + _zyz_instructions(pre1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(mid0, q0, tol)
            + _zyz_instructions(mid1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(post0, q0, tol)
            + _zyz_instructions(post1, q1, tol)
        )

    if zero_z:
        # Map YY -> ZZ via RX(pi/2) on both qubits (leaves XX invariant)
        rx_pos = _rx_mat(0.5 * math.pi)
        rx_neg = _rx_mat(-0.5 * math.pi)
        pre0, pre1 = rx_pos @ R0, rx_pos @ R1
        mid0, mid1 = _rx_mat(-2.0 * cx), _rz_mat(-2.0 * cy)
        post0, post1 = L0 @ rx_neg, L1 @ rx_neg
        return (
            _zyz_instructions(pre0, q0, tol)
            + _zyz_instructions(pre1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(mid0, q0, tol)
            + _zyz_instructions(mid1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(post0, q0, tol)
            + _zyz_instructions(post1, q1, tol)
        )

    if zero_x:
        # Map YY -> XX via RZ(pi/2) on both qubits: RZ(-pi/2) X RZ(pi/2) = -Y => (-Y)x(-Y) = +YY
        rz_pos = _rz_mat(0.5 * math.pi)
        rz_neg = _rz_mat(-0.5 * math.pi)
        pre0, pre1 = rz_pos @ R0, rz_pos @ R1
        mid0, mid1 = _rx_mat(-2.0 * cy), _rz_mat(-2.0 * cz)
        post0, post1 = L0 @ rz_neg, L1 @ rz_neg
        return (
            _zyz_instructions(pre0, q0, tol)
            + _zyz_instructions(pre1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(mid0, q0, tol)
            + _zyz_instructions(mid1, q1, tol)
            + [Instruction("CX", (q0, q1), ())]
            + _zyz_instructions(post0, q0, tol)
            + _zyz_instructions(post1, q1, tol)
        )

    # Case 3: General SU(4) Vatan-Williams 3-CNOT canonical synthesis (PRA 69, 032315, Fig. 6)
    # Using Vatan-Williams Rz_vw(t) = exp(+i*t*Z/2) = RZ(-t), Ry_vw(t) = exp(+i*t*Y/2) = RY(-t),
    # and CX(1, 0) = (H x H) @ CX(0, 1) @ (H x H) so all 3 entanglers are canonical CX(q0, q1):
    #   Circuit order (left-to-right):
    #     pre:   q0: H @ R0,                      q1: H @ RZ(pi/2) @ R1
    #     CX(q0, q1)
    #     mid1:  q0: RZ(-2*cz - pi/2) @ H,        q1: RY(-2*cx - pi/2) @ H
    #     CX(q0, q1)
    #     mid2:  q0: H,                           q1: H @ RY(2*cy + pi/2)
    #     CX(q0, q1)
    #     post:  q0: L0 @ RZ(-pi/2) @ H,          q1: L1 @ H
    pre0 = _H2 @ R0
    pre1 = _H2 @ _rz_mat(0.5 * math.pi) @ R1

    mid1_0 = _rz_mat(-2.0 * cz - 0.5 * math.pi) @ _H2
    mid1_1 = _ry_mat(-2.0 * cx - 0.5 * math.pi) @ _H2

    mid2_0 = _H2
    mid2_1 = _H2 @ _ry_mat(2.0 * cy + 0.5 * math.pi)

    post0 = L0 @ _rz_mat(-0.5 * math.pi) @ _H2
    post1 = L1 @ _H2

    return (
        _zyz_instructions(pre0, q0, tol)
        + _zyz_instructions(pre1, q1, tol)
        + [Instruction("CX", (q0, q1), ())]
        + _zyz_instructions(mid1_0, q0, tol)
        + _zyz_instructions(mid1_1, q1, tol)
        + [Instruction("CX", (q0, q1), ())]
        + _zyz_instructions(mid2_0, q0, tol)
        + _zyz_instructions(mid2_1, q1, tol)
        + [Instruction("CX", (q0, q1), ())]
        + _zyz_instructions(post0, q0, tol)
        + _zyz_instructions(post1, q1, tol)
    )


def synthesize_unitary_matrix(
    instructions: Sequence[Instruction],
    num_qubits: int,
) -> np.ndarray:
    """Computes the exact 2^n x 2^n unitary matrix produced by an instruction sequence."""
    from quanta.simulator.statevector import StateVectorSimulator

    dim = 2 ** num_qubits
    cols = np.zeros((dim, dim), dtype=np.complex128)
    for basis_idx in range(dim):
        sim = StateVectorSimulator(num_qubits)
        psi = np.zeros(dim, dtype=np.complex128)
        psi[basis_idx] = 1.0
        sim.state = psi
        for inst in instructions:
            sim.apply(inst.gate_name, inst.qubits, inst.params)
        cols[:, basis_idx] = sim.state
    return cols
