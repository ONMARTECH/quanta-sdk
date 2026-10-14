"""Automated Confidentiality & Zero-Fabrication Citation Guard Suite.

Ensures that no tracked file in the repository (including docs, papers,
notebooks, source code, scripts, state files, and tests) ever leaks internal
framework identifiers, personal filesystem paths, or private project names.
"""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_ENCODED_FORBIDDEN_PATTERNS: list[tuple[str, int, str]] = [('L1VzZXJzL1thLXpBLVowLTlfLV0r', 0, 'Hardcoded macOS user path'), ('ZW5lc3NhcmlAZ21haWxcLmNvbQ==', 2, 'Personal email'), ('ZW5lc3Nhcmk=', 2, 'Personal username identifier'), ('XGJHRU1JTklcLm1kXGI=', 0, 'Internal rule file reference'), ('XC5nZW1pbmkv', 0, 'Internal config directory'), ('XC5hZ2VudHMv', 0, 'Internal agent directory'), ('XGJhbnRpZ3Jhdml0eVxi', 2, 'Internal framework identifier'), ('Z29vZ2xlLWFudGlncmF2aXR5LXNkaw==', 2, 'Internal SDK identifier'), ('cXVhbnRhX3N1YmNvbnNjaW91c19ob29r', 2, 'Legacy internal hook filename'), ('XGIoZW5lc3Nhcmktd2VifGV0c2dyb3VwX21kbXxvbm1hcnRlY2gtcGFja2FnZXN8YWktbWRtLXBsYXRmb3JtfGRhZ3N0ZXItbWNwLXNlcnZlcnxUdXJuYSBXb3Jrc3xUdXJuYSBKSVJBfHR1cm5hX3dvcmtzfE1laXJvQ0RQfG1laXJvY2RwfExlZ2FsUG9ydHxsZWdhbHBvcnR8SHVrdWtQb3J0fGh1a3VrcG9ydHxNZXRyaWNhc2V8bWV0cmljYXNlfEV0c3R1cnxldHN0dXIpXGI=', 2, 'Sibling private project identifier')]

FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(base64.b64decode(b64).decode("utf-8"), flags), label)
    for b64, flags, label in _ENCODED_FORBIDDEN_PATTERNS
]

PLACEHOLDER_CITATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"arXiv:\d{4}\.x{3,5}", re.IGNORECASE), "Unverified placeholder arXiv ID (arXiv:YYMM.xxxx)"),
    (re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]*x{4,}", re.IGNORECASE), "Placeholder DOI containing xxxx"),
]


def _get_all_tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT).decode("utf-8")
    paths: list[Path] = []
    for rel in out.splitlines():
        p = REPO_ROOT / rel
        if p.exists() and p.is_file():
            paths.append(p)
    return paths


def test_entire_repo_has_zero_internal_or_personal_leaks() -> None:
    """Verifies that 100% of git-tracked files contain zero internal or personal leaks."""
    violations: list[str] = []
    for path in _get_all_tracked_files():
        rel_path = path.relative_to(REPO_ROOT)
        if str(rel_path).startswith("docs/rfcs/RFC_"):
            violations.append(f"{rel_path}: Auto-generated internal RFC file must not be tracked")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern, description in FORBIDDEN_PATTERNS:
            match = pattern.search(content)
            if match:
                line_num = content[: match.start()].count("\n") + 1
                violations.append(
                    f"{rel_path}:{line_num} -> {description} (matched: {match.group(0)!r})"
                )
    assert not violations, "Confidentiality leaks detected in repository:\n" + "\n".join(violations[:50])


def test_cognitive_state_has_zero_leaks() -> None:
    """Verifies that quanta_cognitive_state.json contains no user paths or private project names."""
    state_file = REPO_ROOT / "quanta_cognitive_state.json"
    if not state_file.exists():
        return
    content = state_file.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern, description in FORBIDDEN_PATTERNS:
        match = pattern.search(content)
        if match:
            line_num = content[: match.start()].count("\n") + 1
            violations.append(
                f"quanta_cognitive_state.json:{line_num} -> {description} (matched: {match.group(0)!r})"
            )
    assert not violations, "Confidentiality leaks detected in quanta_cognitive_state.json:\n" + "\n".join(violations)


def test_public_docs_have_zero_placeholder_citations() -> None:
    """Verifies that academic papers and docs have no placeholder arXiv/DOI citations."""
    violations: list[str] = []
    for path in _get_all_tracked_files():
        rel_path = path.relative_to(REPO_ROOT)
        if not str(rel_path).startswith(("docs/", "papers/")):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern, description in PLACEHOLDER_CITATION_PATTERNS:
            match = pattern.search(content)
            if match:
                line_num = content[: match.start()].count("\n") + 1
                violations.append(
                    f"{rel_path}:{line_num} -> {description} (matched: {match.group(0)!r})"
                )
    assert not violations, "Placeholder citations detected:\n" + "\n".join(violations)
