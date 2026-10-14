"""Automated Confidentiality & Zero-Fabrication Citation Guard Suite.

Ensures that no tracked file in the repository (including docs, papers,
notebooks, source code, scripts, state files, and tests) ever leaks internal
framework identifiers, personal filesystem paths, or private project names.
Uses one-way SHA-256 token/phrase digests so zero sensitive strings exist in the test file itself.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Structural patterns that never expose private project/user names
STRUCTURAL_FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/Users/[a-zA-Z0-9_-]+"), "Hardcoded macOS user path"),
    (re.compile(r"\bGEMINI\.md\b"), "Internal rule file reference"),
    (re.compile(r"(?<!\w)\.gemini\b"), "Internal config directory"),
    (re.compile(r"(?<!\w)\.agents\b"), "Internal agent directory"),
]

# One-way SHA-256 digests (first 24 hex chars) of forbidden private identifiers & phrases
FORBIDDEN_SHA256_DIGESTS: frozenset[str] = frozenset(
    ['0520cd17259d9f507b26327a', '0821de6b61d98c6a6446642a', '0a362e39fa835caece630934', '0a86e250b49c590d3e3531d6', '11c559bb31db3a897efb970d', '1c153f101d77f6802d4319e9', '27b85e9eb7a009f10f2e43cf', '2821b5dc469d7d01a71dc485', '28b4451f6ece2fa7acca9d01', '28d3e3f0a36158b5055a0837', '2b648cb62be8c53fb5e76c6a', '33c721dfd15e3d1c4e7711e3', '39bf14dbe948dcea02b5d6c3', '3ab1943b1bc062131012125d', '3b0c75b8896ea99139a4e75a', '3d6b972d645d877d1a3a969a', '3e0f89b3282ac9751bbfa4c5', '4fe5603960ca2260c6775ea7', '557cca998a646259a30e56bd', '56284d93879dd64b3740b1bb', '5ed5746aec2757f1332e6489', '5f4b0aabc521a2369018d12c', '6115651b1bd60b1a881778a0', '6727a62ce4715acbd30cac8e', '68315c5d161f7ec06e53a19b', '6fc87cc52be9f28afd7e41c8', '7822cd3f5977337945322b1f', '7f4abee2d172c751dbb40448', '851d1b14bd6bfa5f75360055', '85eda0a048004c83a0603fac', '8732c56c5920bd81e6cb63b6', '8ed0d8fc920b049d95d66b34', '8ee3899b19b4e5e38f012df2', '921c4d42f3111c9935928ca8', '96ef0fc38ea345fc7ba447ce', '9e4fd6ca996a765290f7868c', 'a01604cefa70e767f65ef1dd', 'a06a337659c6cdb706a4552b', 'a4ec0864401210a08853bb76', 'aabaeb6c833f7b8f246b4719', 'ac0a3dfd6dddb20962cecff6', 'af0452a625222a47caea04f1', 'b4baa1d5af187ff4ad17c6ec', 'b7f2443a4b805156efa30380', 'c038ac9022d62131f99b97f8', 'c19df91a898ed8562aa2dd3f', 'c3cb329c31e57c6660a1e583', 'c469e187c7d7813ccf601015', 'c6c3023289fe67594e2d5a5b', 'ceaf20333c6d66a732a4d3e6', 'd02c406fea87d288f4dfada3', 'd17db77a104796a2dfb302ec', 'dcee6cf213e275e396b268a9', 'ddca2ea9ba25badb987f8930', 'de20f441423b6f948d0b6dbd', 'de9a06977be574613d6b5b6e', 'e2677bacc91e200354fab289', 'e40f8b4a5e210fade4eebfb4', 'e631b39effe928312b4af54d', 'e634e7a75df57357af318293', 'fa0f0403c52be33ea736669e', 'fa472000098b9f2bb33b894e', 'fda8df999f6821b95be145dd']
)

PLACEHOLDER_CITATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"arXiv:\d{4}\.x{3,5}", re.IGNORECASE), "Unverified placeholder arXiv ID (arXiv:YYMM.xxxx)"),
    (re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]*x{4,}", re.IGNORECASE), "Placeholder DOI containing xxxx"),
]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_.@/-]+")


def _get_all_tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT).decode("utf-8")
    paths: list[Path] = []
    for rel in out.splitlines():
        p = REPO_ROOT / rel
        if p.exists() and p.is_file():
            paths.append(p)
    return paths


def _scan_text_for_violations(rel_path: Path | str, content: str) -> list[str]:
    violations: list[str] = []
    for pattern, description in STRUCTURAL_FORBIDDEN_PATTERNS:
        if str(rel_path) == "tests/test_confidentiality_guard.py":
            continue
        match = pattern.search(content)
        if match:
            line_num = content[: match.start()].count("\n") + 1
            violations.append(
                f"{rel_path}:{line_num} -> {description} (matched: {match.group(0)!r})"
            )

    for line_idx, line in enumerate(content.splitlines(), start=1):
        tokens = [t.strip("._-/") for t in _TOKEN_RE.findall(line.lower()) if len(t.strip("._-/")) >= 3]
        candidates = list(tokens)
        # Also check 2-word and 3-word consecutive phrases (e.g. domain cluster names)
        for i in range(len(tokens) - 1):
            candidates.append(f"{tokens[i]} {tokens[i + 1]}")
        for i in range(len(tokens) - 2):
            candidates.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")

        for cand in candidates:
            digest = hashlib.sha256(cand.encode("utf-8")).hexdigest()[:24]
            if digest in FORBIDDEN_SHA256_DIGESTS:
                violations.append(
                    f"{rel_path}:{line_idx} -> Forbidden private identifier digest match ({cand!r})"
                )
                break
    return violations


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
        violations.extend(_scan_text_for_violations(rel_path, content))
    assert not violations, "Confidentiality leaks detected in repository:\n" + "\n".join(violations[:50])


def test_cognitive_state_has_zero_leaks() -> None:
    """Verifies that quanta_cognitive_state.json contains no user paths or private project names."""
    state_file = REPO_ROOT / "quanta_cognitive_state.json"
    if not state_file.exists():
        return
    content = state_file.read_text(encoding="utf-8")
    violations = _scan_text_for_violations("quanta_cognitive_state.json", content)
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
