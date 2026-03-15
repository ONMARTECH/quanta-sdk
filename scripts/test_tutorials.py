#!/usr/bin/env python3
"""
scripts/test_tutorials.py — Extract and test Python code from tutorials.

Parses all markdown files in docs/tutorials/ and docs/migration/,
extracts fenced Python code blocks, and executes them.

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
    Skips blocks that contain '#' comments with 'Qiskit' (comparison blocks).
    """
    text = md_path.read_text()
    blocks: list[tuple[int, str]] = []

    for match in re.finditer(
        r"```python\n(.*?)```",
        text,
        re.DOTALL,
    ):
        code = match.group(1).strip()
        line_no = text[: match.start()].count("\n") + 1

        # Skip Qiskit comparison blocks (migration guides)
        if "# ── Qiskit ──" in code:
            continue

        # Skip blocks that import qiskit modules
        if "from qiskit" in code or "import qiskit" in code:
            continue

        # Skip blocks that only import (no actual test)
        lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if all(ln.startswith(("from ", "import ")) for ln in lines):
            continue

        blocks.append((line_no, code))

    return blocks


def test_code_block(code: str, file: str, line: int) -> bool:
    """Run a Python code block and return True if it succeeds."""
    # Add project root to path
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
        # Show first 3 lines of error
        err_lines = result.stderr.strip().splitlines()
        for line_text in err_lines[-3:]:
            print(f"     {line_text}")
        return False

    print(f"  ✅ {file}:{line}")
    return True


def main() -> int:
    """Run all tutorial code blocks."""
    root = Path(__file__).parent.parent
    dirs = [
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
