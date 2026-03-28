#!/usr/bin/env python3
"""
Convert Quanta SDK tutorials from Markdown to Jupyter Notebooks.

Extracts markdown text and Python code blocks, creates .ipynb files
with Google Colab badges and proper cell structure.

Usage:
    python scripts/md_to_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GITHUB_REPO = "ONMARTECH/quanta-sdk"
NOTEBOOK_DIR = Path("notebooks")
TUTORIAL_DIR = Path("docs/tutorials")

# Colab badge + install cell
COLAB_BADGE_MD = (
    "[![Open In Colab]"
    "(https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/{repo}/blob/main/notebooks/{filename})"
)

INSTALL_CODE = """# Install Quanta SDK (run once)
!pip install -q quanta-sdk"""


def make_cell(cell_type: str, source: list[str], execution_count=None) -> dict:
    """Create a Jupyter notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell


def md_to_notebook(md_path: Path) -> dict:
    """Convert a markdown tutorial to a Jupyter notebook."""
    content = md_path.read_text(encoding="utf-8")
    basename = md_path.stem
    nb_filename = f"{basename}.ipynb"

    cells = []

    # 1. Colab badge cell
    badge = COLAB_BADGE_MD.format(repo=GITHUB_REPO, filename=nb_filename)
    cells.append(make_cell("markdown", [badge]))

    # 2. Install cell
    cells.append(make_cell("code", [INSTALL_CODE]))

    # 3. Parse markdown content into cells
    lines = content.split("\n")
    current_md: list[str] = []
    in_code = False
    code_lines: list[str] = []
    code_lang = ""
    skip_block = False

    for line in lines:
        if line.startswith("```python"):
            # Flush markdown
            if current_md:
                cells.append(make_cell("markdown", ["\n".join(current_md)]))
                current_md = []
            in_code = True
            code_lines = []
            code_lang = "python"
            skip_block = False
            continue
        elif line.startswith("```") and not in_code:
            # Non-python code block (yaml, bash, etc.) — keep as markdown
            if current_md:
                current_md.append(line)
            else:
                current_md = [line]
            # Check if it's a skip-worthy block
            if any(kw in line for kw in ["qiskit", "cirq", "pennylane"]):
                skip_block = True
            continue
        elif line.startswith("```") and in_code:
            # End of code block
            in_code = False
            if code_lang == "python" and code_lines and not skip_block:
                # Check if block has qiskit/cirq imports (skip those)
                first_lines = "\n".join(code_lines[:3]).lower()
                if "import qiskit" in first_lines or "import cirq" in first_lines:
                    # Add as markdown instead
                    cells.append(make_cell("markdown", [
                        "```python\n" + "\n".join(code_lines) + "\n```"
                    ]))
                else:
                    cells.append(make_cell("code", ["\n".join(code_lines)]))
            skip_block = False
            continue

        if in_code:
            code_lines.append(line)
        else:
            current_md.append(line)

    # Flush remaining markdown
    if current_md:
        text = "\n".join(current_md).strip()
        if text:
            cells.append(make_cell("markdown", [text]))

    # Build notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
            "colab": {
                "provenance": [],
                "name": f"Quanta SDK — {basename}",
            },
        },
        "cells": cells,
    }

    return notebook


def main():
    NOTEBOOK_DIR.mkdir(exist_ok=True)

    tutorials = sorted(TUTORIAL_DIR.glob("*.md"))

    if not tutorials:
        print("❌ No tutorials found in docs/tutorials/")
        sys.exit(1)

    print(f"⚛️  Converting {len(tutorials)} tutorials to Jupyter notebooks")
    print(f"   Output: {NOTEBOOK_DIR}/")
    print("=" * 50)

    for md_path in tutorials:
        nb = md_to_notebook(md_path)
        nb_path = NOTEBOOK_DIR / f"{md_path.stem}.ipynb"

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"  ✅ {md_path.name:35s} → {nb_path.name:35s} ({n_code} code, {n_md} md)")

    # Create README for notebooks dir
    readme = [
        "# Quanta SDK Notebooks\n",
        "Interactive Jupyter notebooks for learning Quanta SDK.\n",
        "## Quick Start\n",
        "Click any badge below to open in Google Colab (no install needed):\n",
    ]

    for md_path in tutorials:
        nb_name = f"{md_path.stem}.ipynb"
        title = md_path.stem.replace("-", " ").title()
        badge = f"[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/notebooks/{nb_name})"
        readme.append(f"- **{title}** {badge}")

    readme.extend([
        "",
        "## Local Usage\n",
        "```bash",
        "pip install quanta-sdk jupyter",
        "cd notebooks/",
        "jupyter notebook",
        "```",
        "",
        "## Auto-Generated\n",
        "These notebooks are auto-generated from `docs/tutorials/` using:",
        "```bash",
        "python scripts/md_to_notebook.py",
        "```",
    ])

    (NOTEBOOK_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"📓 {len(tutorials)} notebooks created + README.md")


if __name__ == "__main__":
    main()
