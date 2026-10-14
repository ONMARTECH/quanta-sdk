"""Local Offline Vector & Lexical Retrieval Index for CognitiveAgent Central Vaults.

This module implements a zero-external-dependency local search indexer for CognitiveAgent's
central knowledge and assets vaults (`knowledge_vault.json` and `assets_vault.json`).
Features:
- 100% offline execution using local PyTorch statevectors in C^64 and BM25 lexical ranking.
- Strict security isolation: hard blacklist preventing reading, indexing, or touching
  `credentials_vault.json` or any file containing `credential`, `secret`, `key`, or `token` in its name.
- Sub-50ms query latency via batched complex matrix multiplication (torch.matmul).
- Dual hybrid search combining Quantum Hilbert-space transition fidelity and BM25 lexical scores.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import torch

from quanta.cognitive.memory import text_to_statevector

logger = logging.getLogger(__name__)

# Security Guardrails: Strict Blacklist & Whitelist
SECURITY_BLACKLIST_KEYWORDS: tuple[str, ...] = (
    "credential",
    "secret",
    "key",
    "token",
    "env",
    ".env",
    "password",
    "id_rsa",
    "id_ed25519",
)
ALLOWED_VAULT_FILES: set[str] = {"knowledge_vault.json", "assets_vault.json"}
DEFAULT_VAULT_DIR: Path = Path("~/.quanta/agent/vault").expanduser()


def validate_vault_security(path: str | Path) -> None:
    """Validates that a path does not target sensitive credential or secret files.

    Checks filename, string representation, path segments, and resolved targets
    against blacklisted security keywords.

    Raises:
        PermissionError: If the filename or path contains blacklisted security keywords.
    """
    p = Path(path)
    check_targets: set[str] = {p.name.lower(), str(p).lower()}
    check_targets.update(part.lower() for part in p.parts)

    try:
        resolved = p.resolve()
        check_targets.update({
            resolved.name.lower(),
            str(resolved).lower(),
        })
        check_targets.update(part.lower() for part in resolved.parts)
    except Exception:
        pass

    for target in check_targets:
        for kw in SECURITY_BLACKLIST_KEYWORDS:
            if kw in target:
                raise PermissionError(
                    f"Security Guardrail Violation: Access to '{path}' is strictly forbidden "
                    f"(contains blacklisted keyword '{kw}')."
                )


def is_blacklisted_path(path: str | Path) -> bool:
    """Checks whether a path contains blacklisted security keywords."""
    try:
        validate_vault_security(path)
        return False
    except PermissionError:
        return True


def tokenize_text(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    if not text:
        return []
    return [w for w in re.split(r"[^\w]+", text.lower()) if len(w) > 1]


def _extract_text_from_obj(obj: Any) -> str:
    """Recursively extracts searchable strings from dict/list structures."""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, (int, float, bool)):
        return str(obj)
    elif isinstance(obj, list):
        return " ".join(_extract_text_from_obj(item) for item in obj)
    elif isinstance(obj, dict):
        parts: list[str] = []
        for k, v in obj.items():
            if k in ("is_core_anchor", "timestamp", "created_at", "updated_at", "added_at"):
                continue
            parts.append(str(k))
            parts.append(_extract_text_from_obj(v))
        return " ".join(parts)
    return ""


class LocalVaultIndex:
    """Local offline vector and lexical search index for Central Vaults.

    Indexes `knowledge_vault.json` and `assets_vault.json` locally using PyTorch
    complex statevectors and BM25 lexical ranking with strict credential isolation.
    """

    def __init__(
        self,
        vault_dir: str | Path | None = None,
        auto_build: bool = True,
        vector_dim: int = 64,
        alpha: float = 0.50,
    ) -> None:
        """Initializes the LocalVaultIndex.

        Args:
            vault_dir: Directory containing vault JSON files. Defaults to ~/.quanta/agent/vault.
            auto_build: If True, automatically builds index upon initialization.
            vector_dim: Dimension of complex statevector embedding (default: 64).
            alpha: Weight for vector similarity in [0.0, 1.0]; (1 - alpha) is lexical weight.
        """
        raw_dir = Path(vault_dir).expanduser() if vault_dir is not None else DEFAULT_VAULT_DIR
        validate_vault_security(raw_dir)

        self.vault_dir = raw_dir
        self.vector_dim = vector_dim
        self.alpha = max(0.0, min(1.0, alpha))

        # Index state
        self._documents: list[dict[str, Any]] = []
        self._vector_matrix: torch.Tensor | None = None
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._is_built: bool = False

        if auto_build:
            self.build_index()

    @property
    def is_built(self) -> bool:
        """Returns whether the index has been constructed."""
        return self._is_built

    @property
    def document_count(self) -> int:
        """Returns the total number of indexed documents."""
        return len(self._documents)

    def build_index(self, force_refresh: bool = False) -> int:
        """Parses vault JSON files and constructs offline vector and BM25 indices.

        Args:
            force_refresh: If True, forces rebuilding even if already built.

        Returns:
            Number of indexed documents.

        Raises:
            PermissionError: If any blacklisted file is accessed or attempted.
        """
        if self._is_built and not force_refresh:
            return len(self._documents)

        # Enforce security validation
        validate_vault_security(self.vault_dir)

        docs: list[dict[str, Any]] = []

        if not self.vault_dir.exists() or not self.vault_dir.is_dir():
            logger.warning("Vault directory does not exist or is not a directory: %s", self.vault_dir)
            self._reset_index()
            return 0

        allowed_dir_resolved = self.vault_dir.resolve()
        default_vault_resolved = DEFAULT_VAULT_DIR.resolve()

        # Discover and strictly filter files
        for entry in self.vault_dir.iterdir():
            # Strict blacklist check on entry
            if is_blacklisted_path(entry):
                logger.debug("Strict security guardrail: skipping blacklisted file '%s'", entry.name)
                continue

            # Strict whitelist check on entry name
            if entry.name not in ALLOWED_VAULT_FILES:
                continue

            target_file: Path
            if entry.is_symlink():
                try:
                    resolved_target = entry.resolve()
                except Exception as exc:
                    logger.warning("Security Guardrail: Symlink resolution failed for '%s': %s", entry, exc)
                    continue

                # Verify target against blacklisted security keywords
                if is_blacklisted_path(resolved_target):
                    logger.warning(
                        "Security Guardrail: Symlink '%s' points to blacklisted target '%s'",
                        entry.name,
                        resolved_target,
                    )
                    continue

                # Verify that resolved target is genuinely an allowed vault file
                if resolved_target.name not in ALLOWED_VAULT_FILES:
                    logger.warning(
                        "Security Guardrail: Symlink '%s' points to non-whitelisted file '%s'",
                        entry.name,
                        resolved_target.name,
                    )
                    continue

                # Verify target is a regular file
                if not resolved_target.is_file():
                    logger.warning(
                        "Security Guardrail: Symlink '%s' target '%s' is not a regular file",
                        entry.name,
                        resolved_target,
                    )
                    continue

                # Verify target is located inside an allowed vault directory
                target_parent = resolved_target.parent.resolve()
                is_in_allowed_dir = (
                    target_parent in (allowed_dir_resolved, default_vault_resolved)
                    or allowed_dir_resolved in resolved_target.parents
                    or default_vault_resolved in resolved_target.parents
                )
                if not is_in_allowed_dir:
                    logger.warning(
                        "Security Guardrail: Symlink '%s' targets location outside allowed vault directory: '%s'",
                        entry.name,
                        resolved_target,
                    )
                    continue

                try:
                    validate_vault_security(resolved_target)
                except PermissionError:
                    logger.warning(
                        "Security Guardrail: Symlink target '%s' failed security validation",
                        resolved_target,
                    )
                    continue

                target_file = resolved_target
            else:
                if not entry.is_file():
                    continue

                validate_vault_security(entry)
                target_file = entry

            if entry.name == "knowledge_vault.json":
                docs.extend(self._parse_knowledge_vault(target_file))
            elif entry.name == "assets_vault.json":
                docs.extend(self._parse_assets_vault(target_file))

        self._documents = docs
        n_docs = len(docs)

        if n_docs == 0:
            self._reset_index()
            self._is_built = True
            return 0

        # 1. Build PyTorch Statevector Matrix (offline, local)
        vectors: list[torch.Tensor] = [
            text_to_statevector(doc["searchable_text"], dim=self.vector_dim)
            for doc in docs
        ]
        self._vector_matrix = torch.stack(vectors)  # Shape: (N, dim) complex64

        # 2. Build BM25 / Lexical Index
        self._doc_tokens = [tokenize_text(doc["searchable_text"]) for doc in docs]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(self._doc_lens) / float(n_docs) if n_docs > 0 else 1.0

        df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df

        self._idf = {
            t: math.log(1.0 + (n_docs - c + 0.5) / (c + 0.5))
            for t, c in df.items()
        }

        self._is_built = True
        logger.info(
            "LocalVaultIndex built successfully: %d items indexed from %s",
            n_docs,
            self.vault_dir,
        )
        return n_docs

    def build_or_load_index(self, force_refresh: bool = False) -> int:
        """Alias for build_index to maintain full compatibility."""
        return self.build_index(force_refresh=force_refresh)

    def _reset_index(self) -> None:
        """Resets the index state to empty."""
        self._documents = []
        self._vector_matrix = None
        self._doc_tokens = []
        self._doc_lens = []
        self._avgdl = 0.0
        self._df = {}
        self._idf = {}
        self._is_built = True

    def _parse_knowledge_vault(self, file_path: Path) -> list[dict[str, Any]]:
        """Parses knowledge_vault.json into structured searchable documents."""
        docs: list[dict[str, Any]] = []
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", file_path, exc)
            return docs

        if not isinstance(data, dict):
            return docs

        for key, val in data.items():
            if key == "_metadata":
                continue

            if key == "links" and isinstance(val, list):
                for idx, link in enumerate(val):
                    if not isinstance(link, dict):
                        continue
                    file_key = str(link.get("file_key") or "")
                    url = str(link.get("url") or "")
                    link_id = str(link.get("id") or file_key or url or f"knowledge_link_{idx}")
                    title = str(link.get("title") or "")
                    domain = str(link.get("domain") or "")
                    project_id = str(link.get("project_id") or "")
                    summary = str(
                        link.get("summary")
                        or link.get("purpose")
                        or link.get("solved_problem")
                        or ""
                    )
                    snippet = summary if summary else title
                    full_text = " ".join([
                        link_id,
                        title,
                        domain,
                        project_id,
                        summary,
                        url,
                        _extract_text_from_obj(link),
                    ])
                    docs.append({
                        "id": link_id,
                        "vault": "knowledge",
                        "title": title or link_id,
                        "snippet": snippet,
                        "content": summary,
                        "path": None,
                        "url": url or None,
                        "searchable_text": full_text,
                        "raw": link,
                    })
            elif isinstance(val, dict):
                title = str(val.get("title") or key.replace("_", " ").title())
                purpose = str(val.get("purpose") or val.get("description") or "")
                snippet = purpose if purpose else title
                full_text = " ".join([
                    key,
                    title,
                    purpose,
                    _extract_text_from_obj(val),
                ])
                docs.append({
                    "id": str(key),
                    "vault": "knowledge",
                    "title": title,
                    "snippet": snippet,
                    "content": purpose,
                    "path": None,
                    "url": None,
                    "searchable_text": full_text,
                    "raw": val,
                })

        return docs

    def _parse_assets_vault(self, file_path: Path) -> list[dict[str, Any]]:
        """Parses assets_vault.json into structured searchable documents."""
        docs: list[dict[str, Any]] = []
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", file_path, exc)
            return docs

        if not isinstance(data, dict):
            return docs

        for key, val in data.items():
            if key == "_metadata":
                continue

            if key == "assets" and isinstance(val, list):
                for idx, asset in enumerate(val):
                    if not isinstance(asset, dict):
                        continue
                    file_name = str(asset.get("file_name") or asset.get("name") or "")
                    asset_id = str(asset.get("id") or file_name or f"asset_item_{idx}")
                    title = str(asset.get("title") or file_name or asset_id)
                    desc = str(asset.get("description") or "")
                    path_str = str(asset.get("path") or asset.get("saved_path") or "")
                    snippet = desc if desc else title
                    full_text = " ".join([
                        asset_id,
                        title,
                        desc,
                        path_str,
                        _extract_text_from_obj(asset),
                    ])
                    docs.append({
                        "id": asset_id,
                        "vault": "assets",
                        "title": title,
                        "snippet": snippet,
                        "content": desc,
                        "path": path_str or None,
                        "url": None,
                        "searchable_text": full_text,
                        "raw": asset,
                    })
            elif isinstance(val, dict):
                desc = str(val.get("description") or "")
                path_str = str(val.get("path") or val.get("original_path") or "")
                title = str(val.get("title") or key.replace("_", " ").title())
                snippet = desc if desc else title
                full_text = " ".join([
                    key,
                    title,
                    desc,
                    path_str,
                    _extract_text_from_obj(val),
                ])
                docs.append({
                    "id": str(key),
                    "vault": "assets",
                    "title": title,
                    "snippet": snippet,
                    "content": desc,
                    "path": path_str or None,
                    "url": None,
                    "searchable_text": full_text,
                    "raw": val,
                })

        return docs

    def search(
        self,
        query: str,
        top_k: int = 5,
        vault_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Executes fast dual offline search across indexed vault items.

        Combines Quantum Hilbert-space transition fidelity of complex statevectors
        with normalized BM25 keyword matching and exact match bonuses.

        Args:
            query: Search query text.
            top_k: Maximum number of ranked results to return (default: 5).
            vault_filter: Optional filter ('knowledge', 'assets', or None for all).

        Returns:
            List of result dictionaries containing id, title, score, snippets, and latency_ms.

        Raises:
            PermissionError: If vault_filter targets blacklisted terms.
            ValueError: If vault_filter is unrecognized.
        """
        t0 = time.perf_counter()

        # Guard against security violations in filters
        if vault_filter is not None:
            vf_lower = vault_filter.strip().lower()
            if is_blacklisted_path(vf_lower):
                raise PermissionError(
                    f"Security Guardrail Violation: vault_filter '{vault_filter}' is prohibited."
                )
            if vf_lower not in ("knowledge", "assets", "all"):
                raise ValueError(
                    f"Invalid vault_filter '{vault_filter}'. Must be 'knowledge', 'assets', 'all', or None."
                )
            if vf_lower == "all":
                vf_lower = ""
        else:
            vf_lower = ""

        if not self._is_built:
            self.build_index()

        clean_query = query.strip()
        if not clean_query or len(self._documents) == 0 or top_k <= 0:
            return []

        n_docs = len(self._documents)
        assert self._vector_matrix is not None

        # 1. Quantum Statevector Similarity (C^64)
        q_vec = text_to_statevector(clean_query, dim=self.vector_dim)
        # Inner product <psi_q | psi_d> via batch matmul
        overlaps = torch.matmul(self._vector_matrix, torch.conj(q_vec))
        # Quantum transition fidelity F = |<psi_q | psi_d>|^2 in [0, 1]
        fidelity_scores = torch.abs(overlaps) ** 2

        # 2. BM25 Lexical Ranking
        q_tokens = tokenize_text(clean_query)
        k1 = 1.2
        b = 0.75
        max_possible_bm25 = (
            sum(self._idf.get(t, 0.0) * (k1 + 1.0) for t in q_tokens)
            if q_tokens
            else 1.0
        )
        if max_possible_bm25 <= 0.0:
            max_possible_bm25 = 1.0

        q_lower = clean_query.lower()
        scored_candidates: list[tuple[float, float, float, dict[str, Any]]] = []

        for idx in range(n_docs):
            doc = self._documents[idx]

            # Apply vault filter if specified
            if vf_lower and doc["vault"] != vf_lower:
                continue

            v_score = float(fidelity_scores[idx])

            # Compute BM25 score
            doc_toks = self._doc_tokens[idx]
            dlen = self._doc_lens[idx]
            bm25_score = 0.0
            for t in q_tokens:
                if t in doc_toks:
                    cnt = doc_toks.count(t)
                    tf_norm = (cnt * (k1 + 1.0)) / (
                        cnt + k1 * (1.0 - b + b * (dlen / self._avgdl))
                    )
                    bm25_score += self._idf.get(t, 0.0) * tf_norm

            l_score = min(1.0, bm25_score / max_possible_bm25) if max_possible_bm25 > 0.0 else 0.0

            # Exact match boost for keywords, title, or ID
            exact_boost = 0.0
            doc_id_lower = doc["id"].lower()
            doc_title_lower = doc["title"].lower()

            # Token-in-title / Token-in-id check
            for term in q_tokens:
                if len(term) >= 3 and (term in doc_title_lower or term in doc_id_lower):
                    exact_boost += 0.05

            # Substring check
            if q_lower in doc_title_lower or q_lower in doc_id_lower:
                exact_boost += 0.20

            final_score = min(
                1.0,
                self.alpha * v_score + (1.0 - self.alpha) * l_score + exact_boost,
            )
            scored_candidates.append((final_score, v_score, l_score, doc))

        # Sort descending by final score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        dt_ms = (time.perf_counter() - t0) * 1000.0

        results: list[dict[str, Any]] = []
        for f_score, v_score, l_score, doc in top_candidates:
            results.append({
                "id": doc["id"],
                "vault": doc["vault"],
                "title": doc["title"],
                "score": round(float(f_score), 4),
                "vector_score": round(float(v_score), 4),
                "lexical_score": round(float(l_score), 4),
                "snippet": doc["snippet"],
                "content": doc["content"],
                "path": doc["path"],
                "url": doc["url"],
                "metadata": doc["raw"],
                "latency_ms": round(dt_ms, 3),
            })

        return results

    def search_knowledge(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Searches exclusively within knowledge_vault.json."""
        return self.search(query=query, top_k=top_k, vault_filter="knowledge")

    def search_assets(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Searches exclusively within assets_vault.json."""
        return self.search(query=query, top_k=top_k, vault_filter="assets")


# Alias for backwards and survey compatibility
LocalVaultIndexer = LocalVaultIndex
