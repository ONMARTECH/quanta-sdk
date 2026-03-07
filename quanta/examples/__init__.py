"""
quanta.examples -- Quantum computing example scripts.

Lists available examples for discoverability:

    >>> from quanta.examples import list_examples
    >>> list_examples()
"""

from __future__ import annotations



__all__ = ["list_examples", "EXAMPLES"]

EXAMPLES: dict[str, str] = {
    "01_bell_state":              "Bell state — quantum entanglement basics",
    "02_ghz_state":               "GHZ state — multi-qubit entanglement",
    "03_teleportation":           "Quantum teleportation protocol",
    "04_deutsch_jozsa":           "Deutsch-Jozsa algorithm",
    "05_grover":                  "Grover's search algorithm",
    "06_molecule_energy":         "Molecular ground state energy (VQE)",
    "07_portfolio_optimization":  "Financial portfolio optimization (QAOA)",
    "08_qkd_bb84":                "BB84 quantum key distribution",
    "09_full_demo":               "All SDK features in one script",
    "10_quantum_benchmark":       "QASMBench quality benchmark",
    "11_entity_resolution":       "Quantum entity resolution (dedup)",
}


def list_examples() -> None:
    """Prints all available examples with descriptions.

    Example:
        >>> from quanta.examples import list_examples
        >>> list_examples()
        01_bell_state             Bell state — quantum entanglement basics
        ...

    Run an example with:
        python -m quanta.examples.01_bell_state
    """
    print("Quanta SDK — Available Examples")
    print("=" * 55)
    for name, desc in EXAMPLES.items():
        print(f"  {name:<28} {desc}")
    print("-" * 55)
    print("Run: python -m quanta.examples.<name>")
