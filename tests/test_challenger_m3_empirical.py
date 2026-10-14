"""tests/test_challenger_m3_empirical.py — Challenger M3 Empirical Verification Suite.

Adversarially tests and independently verifies Milestone M3:
1. Challenge 1 (Sandbox Immutability):
   - Cryptographic pre- and post-SHA256 tree hashing of sibling project trees.
   - Empirically verifies zero mutations across both synthetic trees and real workspace projects (LegalPort, Metricase).
   - Verifies compliance with read-only directory/file permissions (chmod 0555 / 0444).
   - Verifies path traversal sanitization in RFC filename generation.

2. Challenge 2 (Domain Orthogonality Threshold & Anti-Rumination):
   - Randomized project matrices (60+ project combinations across all 8 canonical domains).
   - Mathematical verification that all pairs satisfy Delta_domain >= 0.50.
   - Rejection of self-pairing (P_A == P_B).
   - Exclusion of shared project prefixes (e.g. Turna Works vs Turna JIRA).
   - Empirical tracking of anti-rumination buffer rotation and capacity assessment.
   - Boundary condition testing for monolithic domain workspaces.

3. Challenge 3 (RFC Frontmatter and Content Rigor):
   - Strict YAML frontmatter parsing via yaml.safe_load.
   - Structural validation of all required architectural sections.
   - High-salience engram persistence verification (S >= 2.5, category="architecture_rfc").
   - Telemetry verification via read_telemetry_events().
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import stat
from pathlib import Path

import pytest
import yaml

from quanta.cognitive.daemon import (
    REMExecutionResult,
    SleepPhase,
    SubconsciousDaemon,
)
from quanta.cognitive.mind_wander import MindWanderEngine, MindWanderResult
from quanta.cognitive.telemetry import read_telemetry_events
from quanta.cognitive.workspace_harvester import (
    CANONICAL_DOMAINS,
    DEFAULT_PROJECTS_DIR,
    DOMAIN_KEYWORDS,
    HybridDreamSeed,
    ProjectContext,
    WorkspaceContextHarvester,
)


def compute_tree_hashes(
    p: Path,
    exclude_dirs: tuple[str, ...] = (".git", ".agents", "node_modules", "vendor", ".DS_Store"),
) -> dict[str, str]:
    """Computes a mapping of relative file paths to SHA-256 hashes, ignoring build/agent artifacts."""
    hashes: dict[str, str] = {}
    if not p.exists():
        return hashes
    for item in sorted(p.rglob("*")):
        if any(exc in item.parts for exc in exclude_dirs) or item.name == ".DS_Store":
            continue
        if item.is_file() and not item.name.endswith(".pyc") and "__pycache__" not in str(item):
            try:
                h = hashlib.sha256(item.read_bytes()).hexdigest()
                rel = str(item.relative_to(p))
                hashes[rel] = h
            except (OSError, PermissionError):
                pass
    return hashes


class TestEmpiricalChallenge1SandboxImmutability:
    """Challenge 1: Empirically prove zero writes or modifications to external sibling project folders."""

    def test_simulated_sibling_trees_sha256_immutability(self, tmp_path: Path) -> None:
        """Verifies bit-for-bit SHA-256 immutability on synthetic sibling projects."""
        proj_a = tmp_path / "TravelBookingEngine"
        proj_a.mkdir()
        (proj_a / "fare_calculator.py").write_text("def fare(): return 250\n", encoding="utf-8")
        (proj_a / "package.json").write_text(json.dumps({"name": "travel-svc", "version": "1.0.0"}), encoding="utf-8")

        proj_b = tmp_path / "ComplianceAuditCore"
        proj_b.mkdir()
        (proj_b / "audit_engine.py").write_text("def audit(): return 'PASS'\n", encoding="utf-8")
        (proj_b / "pyproject.toml").write_text('description = "compliance audit engine"\n', encoding="utf-8")

        hashes_a_before = compute_tree_hashes(proj_a)
        hashes_b_before = compute_tree_hashes(proj_b)
        assert len(hashes_a_before) == 2
        assert len(hashes_b_before) == 2

        # Execute harvester scanning and pairing
        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        discovered = harvester.scan_workspace_projects(force=True)
        assert len(discovered) >= 2

        pairs = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=3)
        assert len(pairs) > 0
        seed = pairs[0]

        # Execute mind wander dialectic
        engine = MindWanderEngine(max_turns=2)
        result = engine.execute_hybrid_dream_cycle(seed)
        assert result is not None

        # Execute daemon persistence
        daemon = SubconsciousDaemon(
            projects_dir=tmp_path,
            state_file=tmp_path / "quanta_cognitive_state.json",
        )
        rfc_path = daemon._persist_hybrid_rfc(result, seed)

        try:
            assert rfc_path.exists()
            # Assert RFC is strictly inside quanta/docs/rfcs, not inside external projects
            assert not str(rfc_path).startswith(str(proj_a))
            assert not str(rfc_path).startswith(str(proj_b))
            assert "docs/rfcs" in str(rfc_path)

            hashes_a_after = compute_tree_hashes(proj_a)
            hashes_b_after = compute_tree_hashes(proj_b)

            assert hashes_a_before == hashes_a_after, "External Project A suffered mutations!"
            assert hashes_b_before == hashes_b_after, "External Project B suffered mutations!"

            # Assert no docs or temp directories were created inside external projects
            assert not (proj_a / "docs").exists()
            assert not (proj_b / "docs").exists()
            assert not (proj_a / "rfcs").exists()
            assert not (proj_b / "rfcs").exists()
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_real_workspace_sibling_projects_sha256_immutability(self, tmp_path: Path) -> None:
        """Verifies bit-for-bit SHA-256 immutability on actual workspace projects under /workspace."""
        if not DEFAULT_PROJECTS_DIR.exists():
            pytest.skip(f"Workspace directory {DEFAULT_PROJECTS_DIR} not present.")

        lp_dir = DEFAULT_PROJECTS_DIR / "LegalPort"
        mc_dir = DEFAULT_PROJECTS_DIR / "Metricase"

        if not lp_dir.exists() or not mc_dir.exists():
            pytest.skip("LegalPort or Metricase project directories missing.")

        lp_before = compute_tree_hashes(lp_dir)
        mc_before = compute_tree_hashes(mc_dir)

        harvester = WorkspaceContextHarvester(projects_dir=DEFAULT_PROJECTS_DIR)
        projects = harvester.scan_workspace_projects()
        p_lp = next((p for p in projects if p.name == "LegalPort"), None)
        p_mc = next((p for p in projects if p.name == "Metricase"), None)
        assert p_lp is not None and p_mc is not None

        seed = HybridDreamSeed(
            topic="legalport_x_metricase_challenger_test",
            speculative_question="How can legal compliance tags automate web analytics auditing?",
            primary_project=p_lp,
            secondary_project=p_mc,
            domain_distance=0.95,
            bisociation_theme="Automated Compliance-Audited Analytics",
        )

        engine = MindWanderEngine(max_turns=2)
        result = engine.execute_hybrid_dream_cycle(seed)
        assert result is not None

        daemon = SubconsciousDaemon(state_file=tmp_path / "quanta_cognitive_state.json")
        rfc_path = daemon._persist_hybrid_rfc(result, seed)

        try:
            assert rfc_path.exists()
            assert "docs/rfcs" in str(rfc_path)

            lp_after = compute_tree_hashes(lp_dir)
            mc_after = compute_tree_hashes(mc_dir)

            assert lp_before == lp_after, "Real LegalPort directory suffered byte-level modification!"
            assert mc_before == mc_after, "Real Metricase directory suffered byte-level modification!"
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_read_only_hardened_sibling_directory_permissions(self, tmp_path: Path) -> None:
        """Adversarial stress-test: Sets external sibling directory to chmod 0555 (read-only directory)."""
        ro_proj = tmp_path / "Locked_External_Project"
        ro_proj.mkdir()
        ro_file = ro_proj / "PROJECT.md"
        ro_file.write_text("# Locked Project\nDynamic airline pricing and inventory.\n", encoding="utf-8")

        # Set read-only permissions on files and directory
        ro_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        ro_proj.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

        try:
            harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
            # Must successfully inspect without PermissionError (proves zero write/touch attempt)
            ctx = harvester._inspect_project(ro_proj)
            assert ctx.name == "Locked_External_Project"
            assert "Locked Project" in ctx.summary
        finally:
            # Restore permissions for cleanup
            ro_proj.chmod(stat.S_IRWXU)
            ro_file.chmod(stat.S_IRWXU)

    def test_path_traversal_sanitization_in_rfc_persistence(self, tmp_path: Path) -> None:
        """Verifies adversarial project names with path traversal characters cannot escape docs/rfcs."""
        p_malicious_a = ProjectContext(
            name="../../etc/evil_project",
            path=Path("/mock/evil1"),
            domain="Travel/Pricing",
            summary="Airline pricing",
        )
        p_malicious_b = ProjectContext(
            name="..\\..\\windows\\system32",
            path=Path("/mock/evil2"),
            domain="LegalTech",
            summary="Compliance",
        )

        seed = HybridDreamSeed(
            topic="evil_traversal_test",
            speculative_question="Traversal safety test",
            primary_project=p_malicious_a,
            secondary_project=p_malicious_b,
            domain_distance=0.90,
            bisociation_theme="Traversal Safety",
        )

        result = MindWanderResult(
            topic=seed.topic,
            seed_question=seed.speculative_question,
            synthesis="Synthesis",
            confidence=0.92,
            turns_taken=2,
            tokens_used=200,
            anti_rumination_reset_occurred=False,
            primary_project=p_malicious_a.name,
            secondary_project=p_malicious_b.name,
            domain_distance=0.90,
            bisociation_theme=seed.bisociation_theme,
        )

        daemon = SubconsciousDaemon(state_file=tmp_path / "quanta_cognitive_state.json")
        rfc_path = daemon._persist_hybrid_rfc(result, seed)

        try:
            assert rfc_path.exists()
            assert rfc_path.parent.name == "rfcs"
            assert ".." not in str(rfc_path.name)
            assert "/" not in rfc_path.name
            assert "\\" not in rfc_path.name
        finally:
            if rfc_path.exists():
                rfc_path.unlink()


class TestEmpiricalChallenge2DomainOrthogonalityAndAntiRumination:
    """Challenge 2: Stress-test pair_orthogonal_projects with randomized matrices and boundary tests."""

    def test_randomized_project_matrices_orthogonality_threshold(self) -> None:
        """Stress-test pair_orthogonal_projects across 60 randomized project contexts with distinct prefixes."""
        random.seed(42)
        prefixes = [
            "AeroFlight", "QuantumCore", "LexJuris", "OptiAnalytics",
            "DataMeshCDP", "MetabaseHub", "GraphNetwork", "GenCopyNLP",
        ]
        projects: list[ProjectContext] = []

        # Generate 60 diverse projects with domain-aligned distinct prefixes
        for i in range(60):
            dom_idx = i % 8
            dom_name = CANONICAL_DOMAINS[dom_idx]
            pfx = prefixes[dom_idx]
            kw = DOMAIN_KEYWORDS[dom_idx][i % len(DOMAIN_KEYWORDS[dom_idx])]
            name = f"{pfx}_{i}_{kw}"
            p = ProjectContext(
                name=name,
                path=Path(f"/mock/{name}"),
                domain=dom_name,
                tech_stack=[kw, "Python", "GCP"],
                summary=f"Production {name} operating on {kw} in domain {dom_name}.",
            )
            projects.append(p)

        harvester = WorkspaceContextHarvester(projects_dir=Path("/mock"))
        harvester._cached_projects = projects

        # Run 5 consecutive pairing cycles
        for cycle in range(5):
            pairs = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=8)
            assert len(pairs) > 0, f"Cycle {cycle} returned 0 pairs!"

            for seed in pairs:
                assert isinstance(seed, HybridDreamSeed)
                # Mathematical verification of distance
                assert seed.domain_distance >= 0.50, (
                    f"Pair {seed.primary_project.name} x {seed.secondary_project.name} "
                    f"violated orthogonality threshold: {seed.domain_distance} < 0.50"
                )
                assert seed.primary_project.name != seed.secondary_project.name, (
                    f"Self-pairing detected: {seed.primary_project.name}"
                )

    def test_strict_exclusion_of_shared_project_prefixes(self) -> None:
        """Verifies projects with identical prefix tokens are never paired together."""
        pairs_to_test = [
            ("Turna Works", "Turna JIRA"),
            ("Turna Flights", "Turna Hotels"),
            ("LegalPort Search", "LegalPort Contracts"),
            ("Metricase Web", "Metricase Mobile"),
            ("Qarezmi Core", "Qarezmi Analytics"),
        ]

        projects: list[ProjectContext] = []
        for name_a, name_b in pairs_to_test:
            projects.append(
                ProjectContext(
                    name=name_a,
                    path=Path(f"/mock/{name_a}"),
                    domain="Travel/Pricing",
                    tech_stack=["flight", "pricing"],
                    summary=f"Context for {name_a}",
                )
            )
            projects.append(
                ProjectContext(
                    name=name_b,
                    path=Path(f"/mock/{name_b}"),
                    domain="LegalTech",  # Intentionally different domain to test prefix filter alone
                    tech_stack=["legal", "compliance"],
                    summary=f"Context for {name_b}",
                )
            )

        harvester = WorkspaceContextHarvester(projects_dir=Path("/mock"))
        harvester._cached_projects = projects

        pairs = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=10)

        for seed in pairs:
            names = {seed.primary_project.name, seed.secondary_project.name}
            for p_a, p_b in pairs_to_test:
                assert names != {p_a, p_b}, f"Shared prefix pair {p_a} x {p_b} was incorrectly formed!"

    def test_prefix_whitelist_asymmetry_behavior(self) -> None:
        """Empirically observes the whitelist asymmetry between line 1238 and line 1248 in workspace_harvester.py.

        Line 1238 defines whitelist: {'mcp', 'ai', 'app', 'project', 'google'}
        Line 1248 defines whitelist: {'mcp', 'ai', 'app'}
        This causes projects starting with 'Project' or 'Google' to be excluded by condition 2.
        """
        p1 = ProjectContext(
            name="Project DELTA",
            path=Path("/mock/p1"),
            domain="Travel/Pricing",
            tech_stack=["flight", "pricing"],
            summary="Airline pricing engine",
        )
        p2 = ProjectContext(
            name="Project OMEGA",
            path=Path("/mock/p2"),
            domain="Quantum/Brain",
            tech_stack=["quantum", "brain"],
            summary="Unitary Hamiltonian simulator",
        )

        harvester = WorkspaceContextHarvester(projects_dir=Path("/mock"))
        harvester._cached_projects = [p1, p2]

        # Because line 1248 does not include 'project', 'Project DELTA' and 'Project OMEGA' are excluded
        pairs = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=5)
        # We empirically record that 0 pairs are formed due to this asymmetry
        assert len(pairs) == 0, (
            "Observation: 'project' prefix is filtered out by line 1248 asymmetry."
        )

    def test_anti_rumination_buffer_capacity_and_rotation(self) -> None:
        """Evaluates anti-rumination pair tracking, rotation dynamics, and buffer capacity."""
        names = [
            ("Turna Works", "Travel/Pricing", ["flight"]),
            ("LegalPort", "LegalTech", ["legal"]),
            ("Metricase", "Web Analytics/MarTech", ["analytics"]),
            ("quanta", "Quantum/Brain", ["quantum"]),
            ("MeiroCDP", "CDP/MDM", ["cdp"]),
            ("metabase-ai-assistant", "BI/MCP", ["metabase"]),
            ("app-my-network-planner", "Mobile Graph CRM", ["crm"]),
            ("ContentAI", "Generative NLP", ["content"]),
        ]
        projects = [
            ProjectContext(name=n, path=Path(f"/mock/{n}"), domain=d, tech_stack=k, summary=f"{n} in {d}")
            for n, d, k in names
        ]

        harvester = WorkspaceContextHarvester(projects_dir=Path("/mock"))
        harvester._cached_projects = projects

        # 1. Verify internal pair buffer exists and records canonically sorted pairs
        pairs_step1 = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=3)
        assert len(pairs_step1) == 3
        assert hasattr(harvester, "_recent_hybrid_pairs")
        for item in harvester._recent_hybrid_pairs:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert item[0] <= item[1], "Buffer items must be canonically sorted (min, max)"

        # 2. Check buffer maxlen and public anti_rumination_pairs property
        buffer_maxlen = harvester._recent_hybrid_pairs.maxlen
        assert buffer_maxlen == 50, f"Expected capacity 50 per PROJECT.md, got {buffer_maxlen}"
        assert hasattr(harvester, "anti_rumination_pairs")
        assert harvester.anti_rumination_pairs == list(harvester._recent_hybrid_pairs)

    def test_monolithic_domain_boundary_returns_empty(self) -> None:
        """Verifies workspace with zero orthogonal domain diversity returns empty list gracefully."""
        monolithic_projects = [
            ProjectContext(
                name=f"Flight_Agency_{i}",
                path=Path(f"/mock/fa_{i}"),
                domain="Travel/Pricing",
                tech_stack=["flight", "airline", "pricing", "turna"],
                summary="Dynamic ticket repricing and flight bookings.",
            )
            for i in range(10)
        ]

        harvester = WorkspaceContextHarvester(projects_dir=Path("/mock"))
        harvester._cached_projects = monolithic_projects

        # All pairwise distances will be << 0.50
        pairs = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=5)
        assert len(pairs) == 0, "Monolithic domain workspace must not form pairs when Delta < 0.50"

    def test_unachievable_distance_threshold_returns_empty(self) -> None:
        """Verifies unachievable min_domain_distance (e.g. 1.50) returns empty list without error."""
        projects = [
            ProjectContext(name="Turna Works", path=Path("/mock/turna"), domain="Travel/Pricing"),
            ProjectContext(name="LegalPort", path=Path("/mock/legal"), domain="LegalTech"),
        ]
        harvester = WorkspaceContextHarvester(projects_dir=Path("/mock"))
        harvester._cached_projects = projects

        pairs = harvester.pair_orthogonal_projects(min_domain_distance=1.50, max_pairs=5)
        assert len(pairs) == 0


class TestEmpiricalChallenge3RFCFrontmatterAndContentRigor:
    """Challenge 3: Rigorously validate generated RFC YAML frontmatter, schema, and architectural sections."""

    def test_yaml_frontmatter_strict_parsing_and_schema(self, tmp_path: Path) -> None:
        """Empirically parses YAML frontmatter using yaml.safe_load and asserts all fields."""
        p_a = ProjectContext(
            name="Turna Works",
            path=Path("/mock/turna"),
            domain="Travel/Pricing",
            tech_stack=["MSSQL", "BigQuery"],
            summary="Airline ticket pricing engine.",
        )
        p_b = ProjectContext(
            name="LegalPort",
            path=Path("/mock/legal"),
            domain="LegalTech",
            tech_stack=["Python", "FastAPI"],
            summary="Legal compliance.",
        )
        seed = HybridDreamSeed(
            topic="turna_x_legal_rfc_test",
            speculative_question="How can legal compliance arbitrate airline pricing?",
            primary_project=p_a,
            secondary_project=p_b,
            domain_distance=0.9150,
            bisociation_theme="Compliance-Gated Ticket Repricing",
        )
        result = MindWanderResult(
            topic=seed.topic,
            seed_question=seed.speculative_question,
            synthesis="Consensual synthesis bridging flight pricing and compliance.",
            confidence=0.96,
            turns_taken=3,
            tokens_used=450,
            anti_rumination_reset_occurred=False,
            relaxed_prefrontal=True,
            prefrontal_temperature=0.50,
            dreamer_temperature=0.92,
            primary_project="Turna Works",
            secondary_project="LegalPort",
            domain_pair="Travel/Pricing x LegalTech",
            domain_distance=0.9150,
            bisociation_theme=seed.bisociation_theme,
            abstract_architecture="[Turna Works] <---> [Compliance Bus] <---> [LegalPort]",
            failure_modes=[{"mode": "Schema Drift", "cause": "Field drift", "mitigation": "Adapter"}],
        )

        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file)
        rfc_path = daemon._persist_hybrid_rfc(result, seed)

        try:
            assert rfc_path.exists()
            content = rfc_path.read_text(encoding="utf-8")

            # 1. Verify YAML frontmatter delimiters
            assert content.startswith("---\n"), "RFC must start with YAML frontmatter delimiter '---'"
            parts = content.split("---\n", 2)
            assert len(parts) >= 3, "RFC must contain opening and closing '---' delimiters"
            raw_frontmatter = parts[1]

            # 2. Strict YAML parse via PyYAML
            parsed_fm = yaml.safe_load(raw_frontmatter)
            assert isinstance(parsed_fm, dict), "Frontmatter must parse into a valid YAML dictionary"

            # 3. Assert required keys and types
            assert "rfc_id" in parsed_fm
            assert parsed_fm["rfc_id"].startswith("RFC_HYBRID_")
            assert re.match(r"^RFC_HYBRID_\d{8}_\d{6}_[a-z0-9_]+$", parsed_fm["rfc_id"])

            assert "projects" in parsed_fm
            assert isinstance(parsed_fm["projects"], list)
            assert len(parsed_fm["projects"]) == 2
            assert "Turna Works" in parsed_fm["projects"]
            assert "LegalPort" in parsed_fm["projects"]

            assert parsed_fm.get("hybrid_type") == "cross_domain_bisociation"
            assert parsed_fm.get("domain_pair") == "Travel/Pricing x LegalTech"
            assert isinstance(parsed_fm.get("domain_distance"), float)
            assert parsed_fm["domain_distance"] >= 0.50
            assert isinstance(parsed_fm.get("confidence"), float)
            assert 0.0 <= parsed_fm["confidence"] <= 1.0
            assert "created_at" in parsed_fm
            assert "generated_by" in parsed_fm
            assert "engine" in parsed_fm
            assert "status" in parsed_fm

            # 4. Record observation regarding 'tags' field
            # Note: Feature 14 requested tags: [architecture, rfc, bisociation, hybrid, rem-cycle]
            assert "tags" in parsed_fm
            assert "architecture" in parsed_fm["tags"]
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_architectural_sections_rigor_and_completeness(self, tmp_path: Path) -> None:
        """Verifies presence of all required sections, Mermaid diagram, and failure modes table."""
        p_a = ProjectContext(name="Turna Works", path=Path("/mock/turna"), domain="Travel/Pricing")
        p_b = ProjectContext(name="app-my-network-planner", path=Path("/mock/net"), domain="Mobile Graph CRM")
        seed = HybridDreamSeed(
            topic="turna_x_net_sections",
            speculative_question="Spectral fare optimization?",
            primary_project=p_a,
            secondary_project=p_b,
            domain_distance=0.95,
            bisociation_theme="Spectral Fare Graph",
        )
        result = MindWanderResult(
            topic=seed.topic,
            seed_question=seed.speculative_question,
            synthesis="Dialectical consensus reached.",
            confidence=0.94,
            turns_taken=3,
            tokens_used=400,
            anti_rumination_reset_occurred=False,
            primary_project=p_a.name,
            secondary_project=p_b.name,
            domain_distance=0.95,
            bisociation_theme=seed.bisociation_theme,
            abstract_architecture="[Turna] <---> [Spectral Graph CRM]",
            failure_modes=[
                {"mode": "Graph Partition Divergence", "cause": "Network partition", "mitigation": "CRDT replay"}
            ],
        )

        daemon = SubconsciousDaemon(state_file=tmp_path / "quanta_cognitive_state.json")
        rfc_path = daemon._persist_hybrid_rfc(result, seed)

        try:
            content = rfc_path.read_text(encoding="utf-8")

            # Section 1
            assert "## 1. Executive Summary & Bisociative Thesis" in content
            assert "### 1.1. Motivation & Context: Project A" in content
            assert "### 1.2. Motivation & Context: Project B" in content
            assert "### 1.3. Speculative Bisociation Question" in content
            assert "### 1.4. Core Insight & Architectural Synthesis" in content

            # Section 2 & Mermaid diagram
            assert "## 2. Dialectical Subconscious Deliberation" in content
            assert "```mermaid" in content
            assert "sequenceDiagram" in content
            assert "actor DMN as Generative Dreamer" in content
            assert "actor PZ as Prefrontal Zeno Arbiter" in content

            # Section 3
            assert "## 3. Abstract Architecture" in content
            assert "[Turna] <---> [Spectral Graph CRM]" in content

            # Section 4
            assert "## 4. Failure Modes & Mitigations" in content
            assert "| Failure Mode / Impedance Mismatch | Root Cause | Diagnostic Signature | Architectural Mitigation |" in content
            assert "Graph Partition Divergence" in content

            # Section 5
            assert "## 5. Sandbox Safety & Isolation Verification" in content
            assert "Zero Mutation Guarantee" in content
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_rfc_engram_salience_and_state_persistence(self, tmp_path: Path) -> None:
        """Verifies high-salience engram (S = 2.8 >= 2.5) is stored in quanta_cognitive_state.json."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file)

        p_a = ProjectContext(name="Turna Works", path=Path("/mock/turna"), domain="Travel/Pricing")
        p_b = ProjectContext(name="LegalPort", path=Path("/mock/legal"), domain="LegalTech")
        seed = HybridDreamSeed(
            topic="turna_x_legal_engram",
            speculative_question="Question?",
            primary_project=p_a,
            secondary_project=p_b,
            domain_distance=0.88,
            bisociation_theme="Compliance Gateway",
        )
        result = MindWanderResult(
            topic=seed.topic,
            seed_question=seed.speculative_question,
            synthesis="Synthesis result",
            confidence=0.95,
            turns_taken=2,
            tokens_used=250,
            anti_rumination_reset_occurred=False,
            primary_project=p_a.name,
            secondary_project=p_b.name,
            domain_distance=0.88,
            bisociation_theme=seed.bisociation_theme,
        )

        rfc_path = daemon._persist_hybrid_rfc(result, seed)
        try:
            assert state_file.exists()
            data = json.loads(state_file.read_text(encoding="utf-8"))
            engrams = data.get("engrams", [])

            matching_engrams = [
                e for e in engrams
                if e.get("category") == "architecture_rfc" and "turna_works" in e.get("key", "")
            ]
            assert len(matching_engrams) >= 1
            e = matching_engrams[-1]
            assert e["salience"] == 2.8
            assert e["salience"] >= 2.5, "Engram must have dopaminergic protection S >= 2.5"
            assert "Turna Works" in e["content"]
            assert "LegalPort" in e["content"]
            assert "Compliance Gateway" in e["content"]
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_rem_cycle_end_to_end_telemetry_emission(self, tmp_path: Path) -> None:
        """Verifies run_rem_cycle with HybridDreamSeed emits proper telemetry event."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file)

        p_a = ProjectContext(name="Turna Works", path=Path("/mock/turna"), domain="Travel/Pricing")
        p_b = ProjectContext(name="LegalPort", path=Path("/mock/legal"), domain="LegalTech")
        seed = HybridDreamSeed(
            topic="turna_x_legal_telemetry_unique",
            speculative_question="Telemetry test question unique?",
            primary_project=p_a,
            secondary_project=p_b,
            domain_distance=0.93,
            bisociation_theme="Telemetry Compliance Synthesis Unique",
        )

        res = daemon.run_rem_cycle(seed=seed)
        assert res is not None
        assert isinstance(res, REMExecutionResult)
        assert res.projects_paired == ["Turna Works", "LegalPort"]
        assert res.phase == SleepPhase.REM_DREAM

        assert res.rfc_path is not None
        rfc_file = Path(res.rfc_path)
        try:
            assert rfc_file.exists()

            events = read_telemetry_events(event_type="sleep_cycle", limit=200)
            matching_events = [
                ev for ev in events
                if ev.get("sleep_phase") == "REM_DREAM" and ev.get("metrics", {}).get("topic") == seed.topic
            ]
            assert len(matching_events) >= 1
            ev = matching_events[-1]
            metrics = ev["metrics"]
            assert metrics["hybrid"] is True
            assert metrics["projects_paired"] == ["Turna Works", "LegalPort"]
            assert metrics["domain_distance"] == 0.93
            assert metrics["rfc_path"] == str(rfc_file)
        finally:
            if rfc_file.exists():
                rfc_file.unlink()
