#!/usr/bin/env python3
"""
scripts/test_tutorials.py — Extract and test Python code from documentation.

Parses all markdown files in docs/getting-started/, docs/tutorials/,
docs/migration/, and docs/cookbook/, extracts fenced Python code blocks,
and executes them.

Designed for CI: exits with code 1 if any code block fails.

Usage:
    python scripts/test_tutorials.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def extract_python_blocks(md_path: Path) -> list[tuple[int, str]]:
    """Extract Python code blocks from a markdown file.

    Returns list of (line_number, code_string) tuples.
    Skips external SDK comparison blocks (Qiskit, Cirq, PennyLane).
    """
    text = md_path.read_text(encoding="utf-8")
    blocks: list[tuple[int, str]] = []

    for match in re.finditer(
        r"```python\n(.*?)```",
        text,
        re.DOTALL,
    ):
        code = match.group(1).strip()
        line_no = text[: match.start()].count("\n") + 1

        # Skip external SDK comparison blocks (migration guides)
        if "# ── Qiskit" in code or "# ── Cirq" in code or "# ── PennyLane" in code:
            continue

        # Skip blocks that import external SDKs (comparison/migration code)
        if "from qiskit" in code or "import qiskit" in code:
            continue
        if "import cirq" in code or "from cirq" in code:
            continue
        if "import pennylane" in code or "from pennylane" in code:
            continue

        # Skip blocks that only import or are purely comments (no executable statements)
        lines = [
            ln
            for ln in code.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not lines or all(ln.startswith(("from ", "import ")) for ln in lines):
            continue

        blocks.append((line_no, code))

    return blocks


def test_code_block(code: str, file: str, line: int) -> bool:
    """Run a Python code block and return True if it succeeds."""
    wrapper = f"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('QUANTA_TEST_MODE', '1')
{code}
"""
    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).parent.parent),
    )

    if result.returncode != 0:
        print(f"  ❌ {file}:{line}")
        err_lines = result.stderr.strip().splitlines()
        for line_text in err_lines[-3:]:
            print(f"     {line_text}")
        return False

    print(f"  ✅ {file}:{line}")
    return True


def main() -> int:
    """Run all documentation Python code blocks."""
    root = Path(__file__).parent.parent
    dirs = [
        root / "docs" / "getting-started",
        root / "docs" / "tutorials",
        root / "docs" / "migration",
        root / "docs" / "cookbook",
    ]

    total = 0
    passed = 0
    failed = 0

    for d in dirs:
        if not d.exists():
            continue
        for md in sorted(d.glob("*.md")):
            blocks = extract_python_blocks(md)
            if not blocks:
                continue

            rel = md.relative_to(root)
            print(f"\n📄 {rel} ({len(blocks)} blocks)")

            for line_no, code in blocks:
                total += 1
                if test_code_block(code, str(rel), line_no):
                    passed += 1
                else:
                    failed += 1

    print(f"\n{'='*40}")
    print(f"Tutorial tests: {passed}/{total} passed, {failed} failed")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
