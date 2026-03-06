"""
quanta.compiler.pipeline — Compiler pass pipeline management.


  3. Schedule: Paralel katmanlara grupla

Example:
    >>> from quanta.compiler.pipeline import CompilerPipeline
    >>> from quanta.compiler.passes.optimize import CancelInverses
    >>> pipeline = CompilerPipeline([CancelInverses()])
    >>> optimized_dag = pipeline.run(dag)
"""

from __future__ import annotations

from typing import Protocol

from quanta.dag.dag_circuit import DAGCircuit

# ── Public API ──
__all__ = ["CompilerPass", "CompilerPipeline"]

class CompilerPass(Protocol):
    """Interface for a single compiler pass.

    """

    name: str

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Runs the pass on a DAG.

        Args:

        Returns:
        """
        ...

class CompilerPipeline:
    """Pipeline that runs compiler passes sequentially.


    Args:

    Example:
        >>> pipeline = CompilerPipeline([
        ...     CancelInverses(),
        ...     MergeRotations(),
        ...     ScheduleMoments(),
        ... ])
        >>> optimized = pipeline.run(dag)
        >>> print(pipeline.stats)
    """

    def __init__(self, passes: list[CompilerPass] | None = None) -> None:
        self._passes = passes or []
        self.stats: dict[str, dict] = {}

    def add_pass(self, compiler_pass: CompilerPass) -> CompilerPipeline:
        """Adds a new pass to the pipeline. Returns self for chaining.

        Args:
            compiler_pass: Eklenecek pass.

        Returns:
            self (for chaining).
        """
        self._passes.append(compiler_pass)
        return self

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Runs all passes sequentially.

        Args:

        Returns:
        """
        current_dag = dag
        self.stats = {}

        for compiler_pass in self._passes:
            before_gates = current_dag.gate_count()
            before_depth = current_dag.depth()

            current_dag = compiler_pass.run(current_dag)

            self.stats[compiler_pass.name] = {
                "gates_before": before_gates,
                "gates_after": current_dag.gate_count(),
                "depth_before": before_depth,
                "depth_after": current_dag.depth(),
                "gates_removed": before_gates - current_dag.gate_count(),
            }

        return current_dag

    def summary(self) -> str:
        """Returns pipeline execution summary."""
        if not self.stats:
            return "Pipeline has not been run yet."

        lines = ["=== Pipeline Summary ==="]
        total_removed = 0

        for pass_name, info in self.stats.items():
            removed = info["gates_removed"]
            total_removed += removed
            lines.append(
                f"║ {pass_name}: "
                f"{info['gates_before']}→{info['gates_after']} gates "
                f"({'-' + str(removed) if removed > 0 else 'no change'})"
            )

        lines.append("╚" + "═" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        names = [p.name for p in self._passes]
        return f"CompilerPipeline(passes={names})"
