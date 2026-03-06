"""
Example 06: Molecular Ground State Energy (H2 molecule)

Uses VQE and Hamiltonian simulation to find the ground-state
energy of the hydrogen molecule (H2).

This is the #1 use case for quantum computers in chemistry:
simulating molecular behavior to discover new drugs and materials.

Running:
    python -m quanta.examples.06_molecule_energy
"""

import numpy as np
from quanta.layer3.vqe import vqe
from quanta.layer3.hamiltonian import molecular_hamiltonian, evolve


def demo_h2_vqe():
    """Find H2 ground-state energy using VQE."""
    print("=" * 55)
    print("  H2 Molecule — VQE Ground State Energy")
    print("=" * 55)

    # H2 Hamiltonian (STO-3G basis, bond length 0.735 Angstrom)
    h2 = molecular_hamiltonian("H2")
    print(f"\n  Molecule: {h2.name}")
    print(f"  Qubits:   {h2.num_qubits}")
    print(f"  Terms:    {len(h2.terms)} Pauli terms")
    print(f"  Basis:    {h2.description}")

    # Run VQE
    result = vqe(
        num_qubits=h2.num_qubits,
        hamiltonian=h2.terms,
        layers=3,
        max_iter=150,
        learning_rate=0.15,
        seed=42,
    )

    # Exact solution for comparison
    from quanta.layer3.vqe import build_hamiltonian_matrix
    H_mat = build_hamiltonian_matrix(h2.terms, h2.num_qubits)
    exact = float(np.linalg.eigvalsh(H_mat)[0])

    print(f"\n  VQE Energy:    {result.energy:.6f} Ha")
    print(f"  Exact Energy:  {exact:.6f} Ha")
    print(f"  Error:         {abs(result.energy - exact):.6f} Ha")
    print(f"  Accuracy:      {(1 - abs(result.energy - exact) / abs(exact)) * 100:.2f}%")
    print(f"  Iterations:    {result.num_iterations}")


def demo_h2_evolution():
    """Time evolution of H2 molecule."""
    print("\n" + "=" * 55)
    print("  H2 Molecule — Time Evolution")
    print("=" * 55)

    h2 = molecular_hamiltonian("H2")
    result = evolve(h2, time=2.0, steps=50)

    print(f"\n  Evolution time: {result.time:.1f}")
    print(f"  Steps:          {len(result.energy_history)}")
    print(f"  Initial energy: {result.energy_history[0]:.6f} Ha")
    print(f"  Final energy:   {result.energy:.6f} Ha")
    print(f"  Energy stable:  {abs(result.energy_history[0] - result.energy) < 0.01}")


def demo_heh_plus():
    """HeH+ cation ground state."""
    print("\n" + "=" * 55)
    print("  HeH+ Ion — VQE Ground State")
    print("=" * 55)

    heh = molecular_hamiltonian("HeH+")
    result = vqe(
        num_qubits=heh.num_qubits,
        hamiltonian=heh.terms,
        layers=3,
        max_iter=100,
        seed=42,
    )

    from quanta.layer3.vqe import build_hamiltonian_matrix
    exact = float(np.linalg.eigvalsh(
        build_hamiltonian_matrix(heh.terms, heh.num_qubits)
    )[0])

    print(f"\n  Molecule:     {heh.name}")
    print(f"  VQE Energy:   {result.energy:.6f} Ha")
    print(f"  Exact Energy: {exact:.6f} Ha")
    print(f"  Accuracy:     {(1 - abs(result.energy - exact) / abs(exact)) * 100:.2f}%")


if __name__ == "__main__":
    demo_h2_vqe()
    demo_h2_evolution()
    demo_heh_plus()
