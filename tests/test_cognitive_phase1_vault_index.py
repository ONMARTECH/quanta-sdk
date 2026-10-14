"""Comprehensive Test Suite for Local Offline Vector Retrieval for Central Vaults.

Verifies:
1. Zero external network calls (socket monkeypatch).
2. Strict security isolation of credentials_vault.json and secret tokens.
3. Sub-50ms query latency performance benchmarks.
4. Top-k recall accuracy on known items in knowledge_vault and assets_vault.
5. Vault filtering, edge cases, and robustness.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from quanta.cognitive.vault_index import (
    DEFAULT_VAULT_DIR,
    LocalVaultIndex,
    LocalVaultIndexer,
    is_blacklisted_path,
    validate_vault_security,
)


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strictly prevents any external socket connection during offline vault search."""
    def guarded_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("CRITICAL SECURITY VIOLATION: Network call attempted during offline vault search!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def isolated_vault_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates a temporary isolated vault directory with sample knowledge, assets, and credentials."""
    base_dir = tmp_path_factory.mktemp("vbox")
    vault_dir = base_dir / "vaults"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # 1. Knowledge Vault
    kv_data = {
        "_metadata": {
            "description": "Test Knowledge Vault",
        },
        "links": [
            {
                "url": "https://www.figma.com/design/test-figma-key",
                "project_id": "CortexAnalytics",
                "domain": "UI/UX & Design System",
                "title": "CortexAnalytics v2 — Core Design System & UI Kit",
                "file_key": "c3a3IQloD7hfo9row3JL40",
                "summary": "Canonical Figma design file tracking Design Tokens and atomic components.",
            },
            {
                "url": "https://support.atlassian.com/atlassian-ai-gateway/docs/mcp",
                "project_id": "Enterprise Issue Tracker",
                "domain": "AI Agent MCP & Ecosystem",
                "title": "Atlassian Remote MCP Server v2 Documentation",
                "summary": "Official setup and specification for Atlassian Remote MCP Server v2.",
            },
        ],
        "booking_tracking_architecture_1791164740": {
            "title": "Turna Booking Conversion and Click Tracking Architecture",
            "purpose": "End-to-end pixel and server-side tracking architecture for flight booking funnel.",
            "topics": ["tracking", "gtm", "turna", "analytics"],
        },
    }
    with open(vault_dir / "knowledge_vault.json", "w", encoding="utf-8") as f:
        json.dump(kv_data, f)

    # 2. Assets Vault
    av_data = {
        "_metadata": {
            "description": "Test Assets Vault",
        },
        "user_feedback_date_filters_not_working": {
            "path": "/mock/path/user_feedback_date_filters_not_working.png",
            "description": "User report regarding date filters (Bugun, Son 7 Gun) not functioning on dashboard header",
            "is_core_anchor": True,
        },
        "user_feedback_billing_table_button_wrap": {
            "path": "/mock/path/user_feedback_billing_table_button_wrap.png",
            "description": "User report regarding action buttons layout shift / wrapping on billing table",
            "is_core_anchor": True,
        },
        "assets": [
            {
                "file_name": "dagster_turna_observability_job_ui.png",
                "path": "/mock/path/dagster_turna_observability_job_ui.png",
                "project_id": "Enterprise OTA",
                "domain": "Data Platform / Dagster UI",
                "title": "Dagster turna_observability_job UI & Asset Graph",
                "description": "Screenshot showing turna_observability_job in Dagster UI.",
            }
        ],
    }
    with open(vault_dir / "assets_vault.json", "w", encoding="utf-8") as f:
        json.dump(av_data, f)

    # 3. Credentials Vault (MUST NEVER BE READ OR INDEXED)
    cred_data = {
        "api_keys": {
            "openai_api_key": "sk-PROD-ULTRA-SECRET-KEY-9999",
            "gemini_api_key": "AIzaSy-TEST-SECRET-TOKEN-ABCD",
            "jira_api_token": "secret_bearer_token_xyz_123",
        },
        "master_password_hash": "$2b$12$e80MvQkK1xX...",
    }
    with open(vault_dir / "credentials_vault.json", "w", encoding="utf-8") as f:
        json.dump(cred_data, f)

    # 4. Another secret file with blacklisted keyword
    with open(vault_dir / "service_account_secret_key.json", "w", encoding="utf-8") as f:
        json.dump({"private_key": "-----BEGIN PRIVATE KEY-----MIIEvgIBADANBgk..."}, f)

    return vault_dir


class TestLocalVaultIndexSecurity:
    """Test battery validating strict security isolation and zero leakage."""

    def test_credentials_vault_strictly_rejected_and_never_indexed(
        self,
        isolated_vault_dir: Path,
        block_network: None,
    ) -> None:
        """Asserts credentials_vault.json and secret tokens are never read, indexed, or exposed."""
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)
        assert indexer.is_built
        assert indexer.document_count > 0

        # Verify no indexed document came from credentials
        for doc in indexer._documents:
            assert "credential" not in doc["id"].lower()
            assert "secret" not in doc["id"].lower()
            assert "sk-PROD-ULTRA-SECRET-KEY-9999" not in doc["searchable_text"]
            assert "AIzaSy-TEST-SECRET-TOKEN-ABCD" not in doc["searchable_text"]
            assert "secret_bearer_token_xyz_123" not in doc["searchable_text"]
            assert "BEGIN PRIVATE KEY" not in doc["searchable_text"]

        # Search specifically for credential terms - must yield 0 matches containing secrets
        secret_queries = [
            "sk-PROD-ULTRA-SECRET-KEY-9999",
            "openai_api_key",
            "jira_api_token",
            "AIzaSy-TEST-SECRET-TOKEN-ABCD",
            "BEGIN PRIVATE KEY",
        ]
        for sq in secret_queries:
            results = indexer.search(query=sq, top_k=5)
            for res in results:
                assert "sk-PROD" not in res.get("snippet", "")
                assert "AIzaSy" not in res.get("snippet", "")
                assert "PRIVATE KEY" not in res.get("snippet", "")
                assert "credentials" not in res.get("id", "").lower()

    def test_validate_path_security_raises_permission_error_on_blacklisted_files(self) -> None:
        """Asserts that attempting to validate or point to blacklisted files raises PermissionError."""
        blacklisted_names = [
            "credentials_vault.json",
            "secret_key.json",
            "user_api_token.txt",
            "/path/to/my_credentials.json",
            "oauth_token_cache.json",
        ]
        for name in blacklisted_names:
            assert is_blacklisted_path(name)
            with pytest.raises(PermissionError, match="Security Guardrail Violation"):
                validate_vault_security(name)

    def test_vault_filter_rejects_blacklisted_keyword(self, isolated_vault_dir: Path) -> None:
        """Asserts that searching with a blacklisted vault_filter raises PermissionError."""
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)
        with pytest.raises(PermissionError, match="Security Guardrail Violation"):
            indexer.search("test", vault_filter="credentials")

        with pytest.raises(PermissionError, match="Security Guardrail Violation"):
            indexer.search("test", vault_filter="secret_tokens")

    def test_init_raises_if_vault_dir_contains_blacklisted_name(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        """Asserts that LocalVaultIndex raises PermissionError if vault_dir contains blacklisted words."""
        base_dir = tmp_path_factory.mktemp("init_sec")
        secret_dir = base_dir / "credentials_vault"
        secret_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(PermissionError, match="Security Guardrail Violation"):
            LocalVaultIndex(vault_dir=secret_dir)

    def test_symlink_traversal_to_credentials_rejected(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        """Asserts that symlinks targeting credentials or outside vault are strictly rejected."""
        base_dir = tmp_path_factory.mktemp("symtest")
        secret_file = base_dir / "credentials_vault.json"
        secret_file.write_text(json.dumps({"api_keys": {"leak": "sk-MALICIOUS-TOKEN-999"}}), encoding="utf-8")

        target_vault = base_dir / "vaults"
        target_vault.mkdir(parents=True, exist_ok=True)
        symlink_target = target_vault / "knowledge_vault.json"
        import os
        os.symlink(secret_file, symlink_target)

        indexer = LocalVaultIndex(vault_dir=target_vault, auto_build=True)
        assert indexer.document_count == 0
        for doc in indexer._documents:
            assert "sk-MALICIOUS-TOKEN-999" not in doc.get("searchable_text", "")

    def test_expanded_blacklist_keywords(self) -> None:
        """Asserts that all expanded blacklist keywords are recognized and blocked."""
        expanded_keywords = [
            ".env",
            "production.env",
            "id_rsa",
            "id_ed25519",
            "password.txt",
            "db_password.json",
        ]
        for name in expanded_keywords:
            assert is_blacklisted_path(name)
            with pytest.raises(PermissionError, match="Security Guardrail Violation"):
                validate_vault_security(name)

    def test_path_traversal_attacks_blocked(self) -> None:
        """Asserts that path traversal attacks containing credentials are blocked."""
        traversal_attacks = [
            "../../.env",
            "vault/credentials/../",
            "/etc/credentials/..",
            "credentials_vault.json/.",
            "./sub/../../private_key.pem",
        ]
        for attack in traversal_attacks:
            assert is_blacklisted_path(attack)
            with pytest.raises(PermissionError, match="Security Guardrail Violation"):
                validate_vault_security(attack)


class TestLocalVaultIndexOfflineAndPerformance:
    """Test battery validating zero network calls and sub-50ms query latency."""

    def test_zero_network_calls_during_build_and_search(
        self,
        isolated_vault_dir: Path,
        block_network: None,
    ) -> None:
        """Asserts 100% offline execution with socket monkeypatch preventing network calls."""
        # Build index with network blocked
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)
        assert indexer.is_built

        # Perform searches with network blocked
        res1 = indexer.search("Figma design system", top_k=5)
        assert len(res1) > 0

        res2 = indexer.search_knowledge("Turna tracking architecture", top_k=3)
        assert len(res2) > 0

        res3 = indexer.search_assets("date filters", top_k=3)
        assert len(res3) > 0

    def test_query_latency_sub_50ms_benchmark(
        self,
        block_network: None,
    ) -> None:
        """Benchmarks query latency against default central vaults or isolated mock, asserting <50ms."""
        # Use existing default vault if available, otherwise test isolated vault
        vault_path = DEFAULT_VAULT_DIR if DEFAULT_VAULT_DIR.exists() else None
        indexer = LocalVaultIndex(vault_dir=vault_path, auto_build=True)
        assert indexer.is_built

        queries = [
            "Figma design system tokens",
            "date filter bug",
            "billing table layout",
            "Turna Dengage API",
            "CortexAnalytics merchant screen",
            "knowledge vault persistent links",
        ]

        latencies_ms: list[float] = []
        for q in queries * 20:  # 120 total iterations
            t0 = time.perf_counter()
            results = indexer.search(query=q, top_k=5)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            if len(indexer._documents) > 0:
                assert len(results) > 0
                assert "id" in results[0]
                assert "score" in results[0]
                assert "latency_ms" in results[0]
                assert results[0]["latency_ms"] < 50.0

        mean_latency = sum(latencies_ms) / len(latencies_ms)
        p95_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]

        # Acceptance Criteria: p95 < 50ms and mean < 25ms
        assert p95_latency < 50.0, f"p95 latency was {p95_latency:.2f}ms (must be < 50ms)"
        assert mean_latency < 25.0, f"Mean latency was {mean_latency:.2f}ms (must be < 25ms)"


class TestLocalVaultIndexTopKRecall:
    """Test battery validating semantic & lexical retrieval accuracy."""

    def test_top_k_recall_on_isolated_vault(
        self,
        isolated_vault_dir: Path,
        block_network: None,
    ) -> None:
        """Verifies top-k retrieval accuracy for Figma, date filter, and tracking queries."""
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)

        # 1. Query: Figma design system tokens
        res_figma = indexer.search("Figma design system tokens components", top_k=3)
        assert len(res_figma) > 0
        top_titles = [r["title"] for r in res_figma]
        assert any("CortexAnalytics" in t or "Figma" in t for t in top_titles)
        assert any(r["id"] == "c3a3IQloD7hfo9row3JL40" for r in res_figma)

        # 2. Query: date filters not working
        res_date = indexer.search("date filters not working", top_k=3)
        assert len(res_date) > 0
        top_ids = [r["id"] for r in res_date]
        assert "user_feedback_date_filters_not_working" in top_ids

        # 3. Query: billing table button wrap
        res_billing = indexer.search("billing table action buttons wrapping", top_k=3)
        assert len(res_billing) > 0
        top_ids_billing = [r["id"] for r in res_billing]
        assert "user_feedback_billing_table_button_wrap" in top_ids_billing

    def test_top_k_recall_on_default_central_vaults(
        self,
        block_network: None,
    ) -> None:
        """Verifies recall on actual central vaults if present on machine."""
        if not DEFAULT_VAULT_DIR.exists():
            pytest.skip("Central vault dir ~/.quanta/agent/vault not present")

        indexer = LocalVaultIndex(auto_build=True)
        assert indexer.document_count >= 40

        # Query 1: Figma design kit
        res_figma = indexer.search("Figma design system tokens components", top_k=3)
        assert any("CortexAnalytics" in r.get("title", "") or "Figma" in r.get("title", "") for r in res_figma)

        # Query 2: Date filter feedback
        res_date = indexer.search("date filters not working", top_k=3)
        top_ids = [r["id"] for r in res_date]
        assert "user_feedback_date_filters_not_working" in top_ids

    def test_vault_filtering_knowledge_vs_assets(
        self,
        isolated_vault_dir: Path,
        block_network: None,
    ) -> None:
        """Verifies vault_filter and search_knowledge / search_assets APIs."""
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)

        # Knowledge only
        k_results = indexer.search_knowledge("tracking", top_k=5)
        assert len(k_results) > 0
        for r in k_results:
            assert r["vault"] == "knowledge"

        # Assets only
        a_results = indexer.search_assets("date filters", top_k=5)
        assert len(a_results) > 0
        for r in a_results:
            assert r["vault"] == "assets"

        # Invalid vault filter
        with pytest.raises(ValueError, match="Invalid vault_filter"):
            indexer.search("test", vault_filter="invalid_category")


class TestLocalVaultIndexEdgeCases:
    """Test battery validating edge cases, empty directories, and refresh mechanisms."""

    def test_empty_vault_directory(self, tmp_path: Path) -> None:
        """Verifies handling of empty vault directory."""
        empty_dir = tmp_path / "empty_vaults"
        empty_dir.mkdir(parents=True, exist_ok=True)

        indexer = LocalVaultIndex(vault_dir=empty_dir, auto_build=True)
        assert indexer.document_count == 0
        results = indexer.search("anything", top_k=5)
        assert results == []

    def test_nonexistent_vault_directory(self, tmp_path: Path) -> None:
        """Verifies handling of non-existent vault directory."""
        missing_dir = tmp_path / "does_not_exist"
        indexer = LocalVaultIndex(vault_dir=missing_dir, auto_build=True)
        assert indexer.document_count == 0
        assert indexer.search("test") == []

    def test_empty_and_whitespace_query(self, isolated_vault_dir: Path) -> None:
        """Verifies empty or whitespace queries return empty lists."""
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)
        assert indexer.search("") == []
        assert indexer.search("   ") == []
        assert indexer.search("\t\n") == []

    def test_top_k_bounds(self, isolated_vault_dir: Path) -> None:
        """Verifies top_k boundary values (0, negative, larger than count)."""
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)
        assert indexer.search("Figma", top_k=0) == []
        assert indexer.search("Figma", top_k=-5) == []
        res = indexer.search("Figma", top_k=100)
        assert len(res) <= indexer.document_count

    def test_force_refresh(self, isolated_vault_dir: Path) -> None:
        """Verifies force_refresh re-reads disk and updates document count."""
        indexer = LocalVaultIndex(vault_dir=isolated_vault_dir, auto_build=True)
        initial_count = indexer.document_count

        # Add a new link to knowledge_vault.json
        kv_file = isolated_vault_dir / "knowledge_vault.json"
        with open(kv_file, encoding="utf-8") as f:
            data = json.load(f)
        data["links"].append({
            "url": "https://example.com/new-doc",
            "title": "Brand New Dynamic Knowledge Item",
            "summary": "Added dynamically for force_refresh verification.",
        })
        with open(kv_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Without force_refresh: returns cached count
        count_cached = indexer.build_index(force_refresh=False)
        assert count_cached == initial_count

        # With force_refresh: reloads and increments count
        count_refreshed = indexer.build_index(force_refresh=True)
        assert count_refreshed == initial_count + 1

        # Search new item
        res = indexer.search("Brand New Dynamic Knowledge Item", top_k=1)
        assert len(res) > 0
        assert res[0]["title"] == "Brand New Dynamic Knowledge Item"

    def test_compatibility_alias_local_vault_indexer(self, isolated_vault_dir: Path) -> None:
        """Verifies LocalVaultIndexer alias and build_or_load_index function identically."""
        indexer = LocalVaultIndexer(vault_dir=isolated_vault_dir, auto_build=False)
        assert not indexer.is_built
        count = indexer.build_or_load_index()
        assert count > 0
        assert indexer.is_built
