"""Permanent Confidentiality & Anti-Leakage Firewall for Quanta SDK.

Enforces the Global Confidentiality and Zero-Fabrication Citation Protocols:
1. Public documentation (`docs/`, `README.md`, `notebooks/`) must NEVER reference
   internal IDE/agent frameworks (`Antigravity`, `google-antigravity-sdk`, `GEMINI.md`,
   `.gemini/`, `.agents/`, `quanta_subconscious_hook`), personal paths (`/Users/aes`),
   personal emails, or private sibling repository names.
2. Codebase (`quanta/`, `scripts/`, `tests/`, `quanta_cognitive_state.json`) must
   NEVER contain hardcoded `/Users/aes` filesystem paths.
3. Academic papers (`docs/paper/`, `docs/papers/`) must NEVER contain placeholder
   citations (e.g., `arXiv:2412.xxxx`).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_DOC_PATTERNS = [
    re.compile(r"/Users/aes", re.IGNORECASE),
    re.compile(r"enessari@gmail\.com", re.IGNORECASE),
    re.compile(r"\bGEMINI\.md\b"),
    re.compile(r"\.gemini/"),
    re.compile(r"\.agents/"),
    re.compile(r"\bantigravity\b", re.IGNORECASE),
    re.compile(r"google-antigravity-sdk", re.IGNORECASE),
    re.compile(r"quanta_subconscious_hook", re.IGNORECASE),
    re.compile(
        r"\b(enessari-web|etsgroup_mdm|onmartech-packages|ai-mdm-platform|"
        r"dagster-mcp-server|Turna Works|Turna JIRA|turna_works|MeiroCDP|meirocdp|"
        r"LegalPort|legalport|HukukPort|hukukport|Metricase|metricase|Etstur|etstur)\b"
    ),
]

FORBIDDEN_CITATION_PLACEHOLDERS = [
    re.compile(r"arXiv:\d{4}\.x+", re.IGNORECASE),
    re.compile(r"10\.\d{4}/x+", re.IGNORECASE),
]


def _get_tracked_files(*prefixes: str) -> list[Path]:
    """Return git-tracked files under the given path prefixes."""
    cmd = ["git", "ls-files", "--", *prefixes]
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    files: list[Path] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        p = REPO_ROOT / line
        if p.exists() and p.is_file():
            files.append(p)
    return files


class TestConfidentialityAndAntiLeakageGuard:
    """Automated firewall preventing internal framework, path, or citation leaks."""

    def test_public_docs_and_notebooks_zero_internal_leaks(self) -> None:
        """Verifies zero internal framework, path, or private project leaks in docs/ and notebooks/."""
        tracked = _get_tracked_files("docs", "README.md", "notebooks")
        assert len(tracked) > 0, "Expected tracked documentation files"

        violations: list[str] = []
        for path in tracked:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico"}:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(REPO_ROOT)
            for pattern in FORBIDDEN_DOC_PATTERNS:
                match = pattern.search(content)
                if match:
                    line_no = content[: match.start()].count("\n") + 1
                    violations.append(f"{rel}:{line_no} matched '{match.group(0)}'")

        assert not violations, (
            "Confidentiality violation detected in public documentation:\n"
            + "\n".join(violations[:20])
        )

    def test_codebase_zero_hardcoded_user_paths(self) -> None:
        """Verifies zero hardcoded /Users/aes paths across quanta/, scripts/, tests/, and state JSON."""
        tracked = _get_tracked_files(
            "quanta", "scripts", "tests", "quanta_cognitive_state.json"
        )
        user_path_re = re.compile(r"/Users/aes\b", re.IGNORECASE)

        violations: list[str] = []
        for path in tracked:
            # Skip this guard file itself which defines the regex literal
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.suffix.lower() in {".pyc", ".png", ".so"}:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            match = user_path_re.search(content)
            if match:
                rel = path.relative_to(REPO_ROOT)
                line_no = content[: match.start()].count("\n") + 1
                violations.append(f"{rel}:{line_no}")

        assert not violations, (
            "Hardcoded /Users/aes path detected in tracked codebase files:\n"
            + "\n".join(violations[:20])
        )

    def test_academic_papers_zero_citation_placeholders(self) -> None:
        """Verifies zero placeholder DOIs or arXiv IDs (e.g. arXiv:2412.xxxx) in papers."""
        tracked = _get_tracked_files("docs/paper", "docs/papers", "docs/arxiv", "docs/theory")
        violations: list[str] = []
        for path in tracked:
            content = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(REPO_ROOT)
            for pattern in FORBIDDEN_CITATION_PLACEHOLDERS:
                match = pattern.search(content)
                if match:
                    line_no = content[: match.start()].count("\n") + 1
                    violations.append(f"{rel}:{line_no} matched '{match.group(0)}'")

        assert not violations, (
            "Placeholder citation detected in academic papers:\n" + "\n".join(violations)
        )
