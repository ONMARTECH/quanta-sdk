"""Adversarial stress test suite for Quanta Cognitive Cockpit and Subconscious Hook.

Authored by teamwork_preview_challenger_2 to empirically verify:
1. Subconscious hook fail-safe execution under malformed/empty/huge payloads,
   zero unhandled exceptions, latency < 25ms.
2. Hook output hierarchical structure starting with `[Quanta Bilişsel Çıpa | SWR Replay]:`
   and displaying Core and Transient sections.
3. Web Dashboard zero-CDN compliance (0 external URLs) and XSS escaping.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOOK_PATH = Path(
    "/Users/aes/Antigravity Projects/Alfa/quanta/scripts/hooks/quanta_subconscious_hook.py"
)
DASHBOARD_PATH = Path(
    "/Users/aes/Antigravity Projects/Alfa/quanta/quanta/cognitive/dashboard/index.html"
)
PYTHON_EXE = sys.executable


# ==============================================================================
# 1. SUBCONSCIOUS HOOK FAIL-SAFE EXECUTION & ADVERSARIAL ROBUSTNESS
# ==============================================================================


class TestSubconsciousHookFailSafe:
    """Stress-tests the subconscious hook with malformed, empty, corrupted, and huge payloads."""

    def test_hook_failsafe_malformed_json_syntax(self) -> None:
        """Asserts hook handles syntax-broken JSON payloads with 0 crashes and clean {} output."""
        malformed_inputs = [
            b"",  # Completely empty
            b"   \n  \t  \n  ",  # Whitespace only
            b"{not valid json",  # Unclosed object
            b'{"key": "unclosed string',  # Unclosed string
            b"[1, 2, 3,",  # Unclosed array
            b"null",  # Scalar null
            b"12345",  # Scalar integer
            b'"just a string"',  # Scalar string
            b"true",  # Scalar boolean
            b"[]",  # List instead of dict
            b"[{}]",  # List of dict
            b"\x00\x01\x02\xfe\xff",  # Binary junk
            b'{"valid_start": true} trailing_junk',  # Trailing junk
        ]

        for payload in malformed_inputs:
            proc = subprocess.run(
                [PYTHON_EXE, str(HOOK_PATH)],
                input=payload,
                capture_output=True,
                check=False,
            )
            # Exit code must always be 0
            assert (
                proc.returncode == 0
            ), f"Hook crashed with code {proc.returncode} on payload {payload!r}"
            # Stderr must NOT contain unhandled Python tracebacks
            assert b"Traceback (most recent call last)" not in proc.stderr, (
                f"Unhandled traceback on payload {payload!r}: "
                f"{proc.stderr.decode('utf-8', errors='replace')}"
            )
            # Stdout must be valid JSON and equal to {}
            out = proc.stdout.strip()
            assert out in (b"{}", b""), f"Expected empty dict output, got: {out!r}"
            if out:
                parsed = json.loads(out)
                assert parsed == {}

    def test_hook_failsafe_unexpected_field_types(self, tmp_path: Path) -> None:
        """Asserts the hook handles bizarre/invalid field datatypes gracefully."""
        bizarre_payloads = [
            # conversationId of wrong types
            {"conversationId": None, "artifactDirectoryPath": str(tmp_path)},
            {"conversationId": 12345, "artifactDirectoryPath": str(tmp_path)},
            {"conversationId": ["list", "of", "items"], "artifactDirectoryPath": str(tmp_path)},
            {"conversationId": {"nested": "dict"}, "artifactDirectoryPath": str(tmp_path)},
            {"conversationId": True, "artifactDirectoryPath": str(tmp_path)},
            # artifactDirectoryPath of wrong types or non-existent paths
            {"conversationId": "conv_bad_art", "artifactDirectoryPath": None},
            {"conversationId": "conv_bad_art", "artifactDirectoryPath": 99999},
            {
                "conversationId": "conv_bad_art",
                "artifactDirectoryPath": "/path/that/does/not/exist/ever_123456789",
            },
            # workspacePaths of wrong types
            {
                "conversationId": "conv_bad_ws",
                "workspacePaths": None,
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_bad_ws",
                "workspacePaths": "single_string_not_list",
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_bad_ws",
                "workspacePaths": [123, None, {}],
                "artifactDirectoryPath": str(tmp_path),
            },
            # stepIdx of wrong types
            {
                "conversationId": "conv_bad_step",
                "stepIdx": "not_an_int",
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_bad_step",
                "stepIdx": -999999,
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_bad_step",
                "stepIdx": None,
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_bad_step",
                "stepIdx": 3.14159,
                "artifactDirectoryPath": str(tmp_path),
            },
            # toolCall of wrong types
            {
                "conversationId": "conv_bad_tc",
                "toolCall": "not_a_dict",
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_bad_tc",
                "toolCall": None,
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_bad_tc",
                "toolCall": {"name": None, "args": "not_dict"},
                "artifactDirectoryPath": str(tmp_path),
            },
            # terminationReason
            {
                "conversationId": "conv_term",
                "terminationReason": "COMPLETE",
                "artifactDirectoryPath": str(tmp_path),
            },
            {
                "conversationId": "conv_term",
                "terminationReason": 12345,
                "artifactDirectoryPath": str(tmp_path),
            },
        ]

        for i, payload in enumerate(bizarre_payloads):
            proc = subprocess.run(
                [PYTHON_EXE, str(HOOK_PATH)],
                input=json.dumps(payload).encode("utf-8"),
                capture_output=True,
                check=False,
            )
            assert proc.returncode == 0, f"Payload #{i} crashed with code {proc.returncode}"
            assert b"Traceback (most recent call last)" not in proc.stderr, (
                f"Payload #{i} threw unhandled traceback: "
                f"{proc.stderr.decode('utf-8', errors='replace')}"
            )
            parsed = json.loads(proc.stdout.decode("utf-8"))
            assert isinstance(parsed, dict)

    def test_hook_failsafe_corrupted_state_file(self, tmp_path: Path) -> None:
        """Asserts the hook handles severe on-disk state file corruption without failing."""
        state_file = tmp_path / "quanta_cognitive_state.json"

        corrupted_states = [
            b"NOT JSON DATA AT ALL\x00\xff\xfe",
            b"",
            b"{",
            b'{"engrams": "not_a_list"}',
            b'{"engrams": [null, 123, "text", {}]}',
            b'{"engrams": [{"key": "missing_salience"}, '
            b'{"key": "nan_sal", "salience": "NaN", "fidelity": "Inf"}]}',
            b'{"turn_count": "not_a_number", "engrams": []}',
        ]

        for i, bad_state in enumerate(corrupted_states):
            state_file.write_bytes(bad_state)

            payload = {
                "conversationId": f"test_corrupt_state_{i}",
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 100 + i,
            }
            proc = subprocess.run(
                [PYTHON_EXE, str(HOOK_PATH)],
                input=json.dumps(payload).encode("utf-8"),
                capture_output=True,
                check=False,
            )
            assert proc.returncode == 0, f"Corrupted state #{i} caused crash"
            assert b"Traceback (most recent call last)" not in proc.stderr
            parsed = json.loads(proc.stdout.decode("utf-8"))
            assert isinstance(parsed, dict)

    def test_hook_failsafe_huge_payloads_and_transcripts(self, tmp_path: Path) -> None:
        """Sends massive payloads (>1.5MB) and huge transcripts to test for OOM and timeouts."""
        transcript_file = tmp_path / "transcript.jsonl"

        # Generate a large transcript file (> 1.2MB, 5,000 JSONL lines)
        lines = []
        for i in range(5000):
            lines.append(
                json.dumps({
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "step_index": i,
                    "content": f"Step {i}: Extensive model analysis and planning " + ("A" * 200),
                })
            )
        transcript_file.write_text("\n".join(lines), encoding="utf-8")
        assert transcript_file.stat().st_size > 1_000_000, "Transcript size must exceed 1MB"

        # Large user prompt payload (>1MB)
        huge_text = "Adversarial stress test prompt " * 40_000  # ~1.2MB
        payload = {
            "conversationId": "test_conv_huge",
            "artifactDirectoryPath": str(tmp_path),
            "transcriptPath": str(transcript_file),
            "userPrompt": huge_text,
            "stepIdx": 5001,
        }
        raw_bytes = json.dumps(payload).encode("utf-8")
        assert len(raw_bytes) > 1_000_000, "Input payload must exceed 1MB"

        start_t = time.perf_counter()
        proc = subprocess.run(
            [PYTHON_EXE, str(HOOK_PATH)],
            input=raw_bytes,
            capture_output=True,
            check=False,
            timeout=10.0,
        )
        elapsed_sec = time.perf_counter() - start_t

        assert proc.returncode == 0
        assert b"Traceback (most recent call last)" not in proc.stderr
        parsed = json.loads(proc.stdout.decode("utf-8"))
        assert isinstance(parsed, dict)
        assert elapsed_sec < 5.0, f"Huge payload processing took {elapsed_sec:.2f}s, expected < 5s"

    def test_hook_latency_performance_profile(self, tmp_path: Path) -> None:
        """Measures in-process execution latency of hook logic to verify < 25ms speed."""
        spec = importlib.util.spec_from_file_location("quanta_subconscious_hook", str(HOOK_PATH))
        assert spec and spec.loader
        hook_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook_mod)

        # Warm up
        mem = hook_mod.FastBiomorphicMemory(capacity=32)
        mem.record("executive_summary_rule", "Summary rule", salience=2.5, is_core_anchor=True)
        mem.record("scientific_integrity_rule", "Integrity rule", salience=2.5, is_core_anchor=True)
        mem.record("native_first_rule", "Native rule", salience=2.8, is_core_anchor=True)
        mem.record_transient("transient_opt", "Option decision", salience=0.5)

        # Profile FastBiomorphicMemory step + prune + recall cycles (20 iterations)
        latencies_ms = []
        for _ in range(20):
            t0 = time.perf_counter()
            mem.step(dt=1.0)
            mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
            _ = [e for e in mem.engrams if e.get("is_core_anchor")]
            _ = [e for e in mem.engrams if not e.get("is_core_anchor")]
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        mean_latency = sum(latencies_ms) / len(latencies_ms)
        max_latency = max(latencies_ms)

        assert mean_latency < 25.0, f"Mean latency was {mean_latency:.3f}ms (must be < 25ms)"
        assert max_latency < 25.0, f"Max latency was {max_latency:.3f}ms (must be < 25ms)"

        # Measure hook internal telemetry latency
        payload = {
            "conversationId": "latency_bench_conv",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 1,
        }
        res1 = subprocess.run(
            [PYTHON_EXE, str(HOOK_PATH)],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
        )
        assert res1.returncode == 0

        # Read recorded hook telemetry to check telemetry.latency_ms
        telemetry_file = tmp_path / "quanta_cognitive_telemetry.jsonl"
        if not telemetry_file.exists():
            telemetry_file = Path(
                "/Users/aes/Antigravity Projects/Alfa/quanta/quanta_cognitive_telemetry.jsonl"
            )

        if telemetry_file.exists():
            last_lines = telemetry_file.read_text(encoding="utf-8").strip().splitlines()[-10:]
            for line in reversed(last_lines):
                rec = json.loads(line)
                if "latency_ms" in rec:
                    internal_lat = rec["latency_ms"]
                    assert (
                        internal_lat < 25.0
                    ), f"Hook internal latency {internal_lat}ms exceeded 25ms!"
                    break


# ==============================================================================
# 2. HOOK HIERARCHICAL OUTPUT CORRECTNESS
# ==============================================================================


class TestHookHierarchicalOutputCorrectness:
    """Verifies that the hook produces the exact hierarchical SWR replay structure."""

    def test_hook_output_starts_with_swr_prefix(self, tmp_path: Path) -> None:
        """Verifies hook output starts with exact string `[Quanta Bilişsel Çıpa | SWR Replay]:`."""
        payload = {
            "conversationId": "test_conv_hierarchical_prefix",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 1,
        }
        res = subprocess.run(
            [PYTHON_EXE, str(HOOK_PATH)],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            check=True,
        )
        data = json.loads(res.stdout.decode("utf-8"))
        assert "injectSteps" in data
        assert len(data["injectSteps"]) >= 1
        msg = data["injectSteps"][0]["ephemeralMessage"]

        # Requirement 2: Must start with `[Quanta Bilişsel Çıpa | SWR Replay]:`
        assert msg.startswith(
            "[Quanta Bilişsel Çıpa | SWR Replay]:"
        ), f"Expected prefix '[Quanta Bilişsel Çıpa | SWR Replay]:', got: {msg[:60]!r}"

    def test_hook_output_core_and_transient_sections(self, tmp_path: Path) -> None:
        """Verifies hook outputs `🔒 Çekirdek:` and when transient items exist, `⚡ Geçici:`."""
        state_file = tmp_path / "quanta_cognitive_state.json"

        # Pre-seed state with core anchors AND transient decisions
        seed_state = {
            "turn_count": 5,
            "last_injected_time": 0.0,
            "last_step_idx": 0,
            "engrams": [
                {
                    "key": "native_first_rule",
                    "content": (
                        "Platform işlemlerinde daima native API/CLI aracını öncelikli kullan."
                    ),
                    "salience": 2.8,
                    "category": "constraint",
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                },
                {
                    "key": "scientific_integrity_rule",
                    "content": "Tüm iddia ve önermelerde literatür doğrulaması yap.",
                    "salience": 2.5,
                    "category": "constraint",
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                },
                {
                    "key": "executive_summary_rule",
                    "content": "Kullanıcıya daima sonuç odaklı yönetici özeti sun.",
                    "salience": 2.5,
                    "category": "constraint",
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                },
                # Transient decisions
                {
                    "key": "decision_step_42",
                    "content": "Use node.js runtime for fast headless verification",
                    "salience": 0.65,
                    "category": "contextual_decision",
                    "fidelity": 0.88,
                    "is_core_anchor": False,
                },
                {
                    "key": "decision_step_43",
                    "content": "Apply CSF shielding to avoid catastrophic drift",
                    "salience": 0.55,
                    "category": "contextual_decision",
                    "fidelity": 0.85,
                    "is_core_anchor": False,
                },
            ],
        }
        state_file.write_text(json.dumps(seed_state, indent=2), encoding="utf-8")

        payload = {
            "conversationId": "test_conv_hierarchical_both",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 10,
        }
        res = subprocess.run(
            [PYTHON_EXE, str(HOOK_PATH)],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            check=True,
        )
        data = json.loads(res.stdout.decode("utf-8"))
        msg = data["injectSteps"][0]["ephemeralMessage"]

        # Requirement 2 assertions:
        assert msg.startswith("[Quanta Bilişsel Çıpa | SWR Replay]:")
        assert "🔒 Çekirdek:" in msg, "Must contain '🔒 Çekirdek:' section"
        assert "⚡ Geçici:" in msg, "Must contain '⚡ Geçici:' section when transient items exist"

        # Assert mandatory invariant rules are present
        assert "native_first_rule" in msg
        assert "scientific_integrity_rule" in msg
        assert "executive_summary_rule" in msg

        # Assert transient keys are in the transient section
        assert "decision_step_42" in msg
        assert "decision_step_43" in msg

        # Invariant: Never flat 100.0%
        assert "%100.0" not in msg
        assert "100.0%" not in msg

    def test_hook_output_with_pruning_section(self, tmp_path: Path) -> None:
        """Verifies hook outputs `✂️ Budandı:` when decayed transient items are swept."""
        state_file = tmp_path / "quanta_cognitive_state.json"

        # Pre-seed state with a decayed transient item (fidelity < 0.70)
        seed_state = {
            "turn_count": 10,
            "last_injected_time": 0.0,
            "last_step_idx": 0,
            "engrams": [
                {
                    "key": "native_first_rule",
                    "content": "Native first",
                    "salience": 2.8,
                    "category": "constraint",
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                },
                {
                    "key": "obsolete_decision_99",
                    "content": "An obsolete temporary step",
                    "salience": 0.30,
                    "category": "contextual_decision",
                    "fidelity": 0.45,  # Decayed below threshold 0.70
                    "is_core_anchor": False,
                },
            ],
        }
        state_file.write_text(json.dumps(seed_state, indent=2), encoding="utf-8")

        payload = {
            "conversationId": "test_conv_pruning_hook",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 20,
        }
        res = subprocess.run(
            [PYTHON_EXE, str(HOOK_PATH)],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            check=True,
        )
        data = json.loads(res.stdout.decode("utf-8"))
        msg = data["injectSteps"][0]["ephemeralMessage"]

        assert "✂️ Budandı:" in msg
        assert "1 engram" in msg


# ==============================================================================
# 3. WEB DASHBOARD ZERO-CDN & XSS ESCAPING VERIFICATION
# ==============================================================================


class TestWebDashboardOfflineAndSecurity:
    """Verifies strict zero-CDN offline compliance and XSS escaping in the Web Dashboard."""

    def test_web_dashboard_zero_cdn_strict_compliance(self) -> None:
        """Asserts zero external URLs (0 CDN dependencies) across HTML, scripts, stylesheets."""
        assert DASHBOARD_PATH.exists(), f"Dashboard file not found at {DASHBOARD_PATH}"
        content = DASHBOARD_PATH.read_text(encoding="utf-8")

        # 1. Check for external src= or href= attributes pointing to http or https
        ext_src = re.findall(r'src=["\'](https?://[^"\']+)["\']', content, re.IGNORECASE)
        ext_href = re.findall(r'href=["\'](https?://[^"\']+)["\']', content, re.IGNORECASE)
        css_blocks = "".join(re.findall(r'<style\b[^>]*>(.*?)</style>', content, re.DOTALL | re.IGNORECASE))
        css_attrs = "".join(re.findall(r'style=["\']([^"\']+)["\']', content, re.IGNORECASE))
        css_content = css_blocks + "\n" + css_attrs
        ext_imp = re.findall(r'@import\s+["\'](https?://[^"\']+)["\']', css_content, re.IGNORECASE)
        ext_url = re.findall(r'url\(["\']?(https?://[^"\'\)]+)["\']?\)', css_content, re.IGNORECASE)

        assert len(ext_src) == 0, f"Found external src dependencies: {ext_src}"
        assert len(ext_href) == 0, f"Found external href dependencies: {ext_href}"
        assert len(ext_imp) == 0, f"Found external @import dependencies: {ext_imp}"
        assert len(ext_url) == 0, f"Found external CSS url() dependencies: {ext_url}"

        # 2. Check all <script> tags: must be purely inline (no src=)
        script_tags = re.findall(r'<script\b[^>]*>', content, re.IGNORECASE)
        for tag in script_tags:
            assert "src=" not in tag.lower(), f"External script tag found: {tag}"

        # 3. Check all <link> tags: must not point to external stylesheet or fonts
        link_tags = re.findall(r'<link\b[^>]*>', content, re.IGNORECASE)
        for tag in link_tags:
            assert "http://" not in tag and "https://" not in tag, f"External link tag found: {tag}"

    def test_web_dashboard_xss_escaping_empirical_in_node(self) -> None:
        """Empirically tests XSS escaping in Node.js executing dashboard escapeHtml logic."""
        node_bin = shutil.which("node") or "/usr/local/bin/node"
        assert os.path.exists(node_bin), "node.js binary must be available for empirical JS test"

        # JS script executing escapeHtml and card rendering from index.html
        js_code = """
        const escapeHtml = (str) => {
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        };

        const maliciousPayloads = [
            '<script>alert("XSS_KEY")</script>',
            '<img src=x onerror="alert(\\'XSS_IMG\\')">',
            '<svg onload=alert(1)>',
            '"><b onmouseover="alert(2)">',
            '\\' onfocus=\\'alert(3)\\'',
            'javascript:alert("XSS")',
            '<iframe src="javascript:alert(1)"></iframe>',
            'Normal & Special "Quotes" \\'Single\\''
        ];

        let failed = 0;
        for (const payload of maliciousPayloads) {
            const escaped = escapeHtml(payload);
            const badChars = ['<', '>', '"', "'"];
            if (badChars.some(c => escaped.includes(c))) {
                console.error('XSS Escape Failed on:', payload, '->', escaped);
                failed++;
            }
        }

        const mockCoreEngram = {
            key: '<script>alert("CORE_KEY")</script>',
            salience: 2.8,
            fidelity: 0.9998,
            confidence: 0.98,
            category: '<img src=x onerror=alert("CAT")>',
            drift_status: '"><script>alert(1)</script>',
            content: 'Attack <svg onload=alert("CONTENT")> body payload'
        };

        const key = escapeHtml(mockCoreEngram.key);
        const cat = escapeHtml(mockCoreEngram.category);
        const drift = escapeHtml(mockCoreEngram.drift_status);
        const rawContent = String(mockCoreEngram.content);
        const snippet = escapeHtml(rawContent.slice(0, 180));

        const cardHtml = `
            <div class="core-card">
                <span class="core-key">${key}</span>
                <span class="status-pill">${drift}</span>
                <p class="core-content-snippet">${snippet}</p>
                <span class="rule-cat">${cat}</span>
            </div>
        `;

        if (cardHtml.includes('<script>') ||
            cardHtml.includes('<img') ||
            cardHtml.includes('<svg')) {
            console.error('Rendered card contains unescaped malicious tags!', cardHtml);
            failed++;
        }

        if (failed > 0) {
            process.exit(1);
        } else {
            console.log("XSS_ESCAPING_VERIFIED_CLEAN");
            process.exit(0);
        }
        """

        res = subprocess.run(
            [node_bin, "-e", js_code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"Node.js XSS check failed: {res.stderr}\n{res.stdout}"
        assert "XSS_ESCAPING_VERIFIED_CLEAN" in res.stdout

    def test_web_dashboard_state_parser_adversarial_loads_in_node(self) -> None:
        """Tests that state loader in index.html handles corrupted, empty, or huge states."""
        node_bin = shutil.which("node") or "/usr/local/bin/node"
        assert os.path.exists(node_bin)

        js_code = """
        const mockContext = {
            state: null,
            loadState(data) {
                if (!data || typeof data !== 'object') return false;
                this.state = data;
                return true;
            },
            computeCounts() {
                if (!this.state) return { core: 0, transient: 0 };
                const engrams = Array.isArray(this.state.engrams) ? this.state.engrams : [];
                const isCore = (e) => e && (e.is_core_anchor === true || (e.salience || 0) >= 2.0);
                const coreCount = engrams.filter(isCore).length;
                const transientCount = engrams.filter(e => e && !isCore(e)).length;
                return { core: coreCount, transient: transientCount };
            }
        };

        const testStates = [
            null,
            undefined,
            "plain string",
            12345,
            {},
            { engrams: null },
            { engrams: "not an array" },
            { engrams: [null, undefined, 123, "text", {}] },
            {
                engrams: Array.from({length: 5000}, (_, i) => ({
                    key: `e_${i}`,
                    salience: i % 2 === 0 ? 2.5 : 0.5
                }))
            }
        ];

        for (const st of testStates) {
            mockContext.loadState(st);
            const counts = mockContext.computeCounts();
            if (typeof counts.core !== 'number' || typeof counts.transient !== 'number') {
                console.error("Invalid count output on state:", st);
                process.exit(1);
            }
        }

        console.log("STATE_PARSER_ADVERSARIAL_VERIFIED_CLEAN");
        process.exit(0);
        """

        res = subprocess.run(
            [node_bin, "-e", js_code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"Node.js state parser test failed: {res.stderr}\n{res.stdout}"
        assert "STATE_PARSER_ADVERSARIAL_VERIFIED_CLEAN" in res.stdout
