"""
test_tasks_6_10_13_15 — Tests for QASM round-trip, Greeks, Config, QEC decode.
"""

import sys

sys.path.insert(0, ".")


# ═══════════════════════════════════════════
# Task 6: QASM Round-Trip
# ═══════════════════════════════════════════

class TestQASMRoundTrip:
    """QASM export → import round-trip."""

    def test_bell_roundtrip(self):
        """Export bell state to QASM, import back, verify gates match."""
        from quanta.core.circuit import circuit
        from quanta.core.gates import CX, H
        from quanta.core.measure import measure
        from quanta.export.qasm import to_qasm
        from quanta.export.qasm_import import from_qasm

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm_str = to_qasm(bell)
        assert "OPENQASM 3.0" in qasm_str
        assert "h q[0]" in qasm_str
        assert "cx q[0], q[1]" in qasm_str

        # Re-import
        dag = from_qasm(qasm_str)
        ops = list(dag.op_nodes())
        assert len(ops) == 2
        assert ops[0].gate_name == "H"
        assert ops[1].gate_name == "CX"

    def test_parametric_roundtrip(self):
        """Parametric gates preserve params through export/import."""
        from quanta.core.circuit import circuit
        from quanta.core.gates import RX, RZ
        from quanta.export.qasm import to_qasm
        from quanta.export.qasm_import import from_qasm

        @circuit(qubits=1)
        def parametric(q):
            RX(1.5707963)(q[0])
            RZ(0.7853982)(q[0])

        qasm_str = to_qasm(parametric)
        assert "rx(" in qasm_str
        assert "rz(" in qasm_str

        dag = from_qasm(qasm_str)
        ops = list(dag.op_nodes())
        assert len(ops) == 2
        assert abs(ops[0].params[0] - 1.5707963) < 0.01

    def test_ghz_roundtrip(self):
        """3-qubit GHZ round-trip."""
        from quanta.core.circuit import circuit
        from quanta.core.gates import CX, H
        from quanta.core.measure import measure
        from quanta.export.qasm import to_qasm
        from quanta.export.qasm_import import from_qasm

        @circuit(qubits=3)
        def ghz(q):
            H(q[0])
            CX(q[0], q[1])
            CX(q[1], q[2])
            return measure(q)

        qasm_str = to_qasm(ghz)
        dag = from_qasm(qasm_str)
        assert dag.num_qubits == 3
        assert dag.gate_count() == 3

    def test_qasm2_import(self):
        """QASM 2.0 format with qreg/creg."""
        from quanta.export.qasm_import import from_qasm

        qasm2 = '''
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        '''
        dag = from_qasm(qasm2)
        assert dag.num_qubits == 2
        assert dag.gate_count() == 2

    def test_qasm_from_gates(self):
        """from_qasm_gates extracts gate list."""
        from quanta.export.qasm import from_qasm_gates

        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
"""
        gates = from_qasm_gates(qasm)
        assert len(gates) == 2
        assert gates[0] == ("H", (0,))
        assert gates[1] == ("CX", (0, 1))


# ═══════════════════════════════════════════
# Task 10: Monte Carlo Greeks
# ═══════════════════════════════════════════

class TestGreeks:
    """Option Greeks computation."""

    def test_call_delta_positive(self):
        """European call delta should be positive."""
        from quanta.layer3.monte_carlo import compute_greeks
        g = compute_greeks(
            spot=100, strike=100, volatility=0.2,
            payoff="european_call", n_samples=50_000, seed=42,
        )
        assert g.delta > 0  # Calls have positive delta

    def test_put_delta_negative(self):
        """European put delta should be negative."""
        from quanta.layer3.monte_carlo import compute_greeks
        g = compute_greeks(
            spot=100, strike=100, volatility=0.2,
            payoff="european_put", n_samples=50_000, seed=42,
        )
        assert g.delta < 0  # Puts have negative delta

    def test_vega_positive(self):
        """Vega should be positive for both calls and puts."""
        from quanta.layer3.monte_carlo import compute_greeks
        g = compute_greeks(
            spot=100, strike=100, volatility=0.2,
            payoff="european_call", n_samples=50_000, seed=42,
        )
        assert g.vega > 0  # More vol → more option value

    def test_greeks_summary(self):
        """Greeks summary should contain all symbols."""
        from quanta.layer3.monte_carlo import compute_greeks
        g = compute_greeks(spot=100, strike=105, n_samples=10_000, seed=42)
        summary = g.summary()
        assert "Δ" in summary
        assert "Γ" in summary
        assert "ν" in summary

    def test_greeks_repr(self):
        """GreeksResult has informative repr."""
        from quanta.layer3.monte_carlo import compute_greeks
        g = compute_greeks(spot=100, strike=105, n_samples=10_000, seed=42)
        assert "delta" in repr(g)


# ═══════════════════════════════════════════
# Task 13: Config & Credentials
# ═══════════════════════════════════════════

class TestConfig:
    """QuantaConfig TOML management."""

    def test_create_and_save(self, tmp_path, monkeypatch):
        """Create, set, save, and load config."""
        monkeypatch.setenv("QUANTA_CONFIG_DIR", str(tmp_path))
        from quanta.config import QuantaConfig

        config = QuantaConfig()
        config.set("ibm", "api_key", "test-token-12345")
        config.set("ionq", "api_key", "ionq-secret")
        config.set_default("backend", "local")
        path = config.save()

        assert path.exists()
        content = path.read_text()
        assert "test-token-12345" in content
        assert "ionq-secret" in content

    def test_load_roundtrip(self, tmp_path, monkeypatch):
        """Save then load should preserve values."""
        monkeypatch.setenv("QUANTA_CONFIG_DIR", str(tmp_path))
        from quanta.config import QuantaConfig

        config = QuantaConfig()
        config.set("ibm", "api_key", "my-ibm-token")
        config.set("ibm", "instance", "crn:v1:bluemix")
        config.set_default("backend", "ibm")
        config.save()

        loaded = QuantaConfig.load()
        assert loaded.get("ibm", "api_key") == "my-ibm-token"
        assert loaded.get("ibm", "instance") == "crn:v1:bluemix"
        assert loaded.get_default("backend") == "ibm"

    def test_list_backends(self, tmp_path, monkeypatch):
        """list_backends returns configured backends."""
        monkeypatch.setenv("QUANTA_CONFIG_DIR", str(tmp_path))
        from quanta.config import QuantaConfig

        config = QuantaConfig()
        config.set("ibm", "api_key", "x")
        config.set("google", "project_id", "y")
        assert sorted(config.list_backends()) == ["google", "ibm"]

    def test_describe_masks_credentials(self, tmp_path, monkeypatch):
        """describe() should mask long credentials."""
        monkeypatch.setenv("QUANTA_CONFIG_DIR", str(tmp_path))
        from quanta.config import QuantaConfig

        config = QuantaConfig()
        config.set("ibm", "api_key", "super-secret-long-token-12345")
        desc = config.describe()
        assert "super-secret-long-token-12345" not in desc
        assert "***" in desc

    def test_empty_config(self, tmp_path, monkeypatch):
        """Loading non-existent config returns empty."""
        monkeypatch.setenv("QUANTA_CONFIG_DIR", str(tmp_path))
        from quanta.config import QuantaConfig

        config = QuantaConfig.load()
        assert config.list_backends() == []
        assert config.get("ibm", "api_key") == ""

    def test_get_default_fallback(self, tmp_path, monkeypatch):
        """get/get_default returns fallback when not set."""
        monkeypatch.setenv("QUANTA_CONFIG_DIR", str(tmp_path))
        from quanta.config import QuantaConfig

        config = QuantaConfig()
        assert config.get("nonexistent", "key", "fallback") == "fallback"
        assert config.get_default("missing", "default_val") == "default_val"


# ═══════════════════════════════════════════
# Task 15: QEC Decode & Correct
# ═══════════════════════════════════════════

class TestQECDecode:
    """QEC decode, lookup_table, correct_error."""

    def test_bitflip_decode_exists(self):
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        dec = code.decode()
        assert dec is not None

    def test_bitflip_lookup_table(self):
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        table = code.lookup_table()
        assert table["00"] == "No error detected"
        assert "qubit 0" in table["11"]

    def test_bitflip_correct_error(self):
        from quanta.qec.codes import BitFlipCode, correct_error
        code = BitFlipCode()
        assert "qubit 0" in correct_error(code, "11")
        assert "qubit 1" in correct_error(code, "10")
        assert "qubit 2" in correct_error(code, "01")
        assert "No error" in correct_error(code, "00")

    def test_phaseflip_decode(self):
        from quanta.qec.codes import PhaseFlipCode
        code = PhaseFlipCode()
        dec = code.decode()
        assert dec is not None

    def test_phaseflip_lookup(self):
        from quanta.qec.codes import PhaseFlipCode
        code = PhaseFlipCode()
        table = code.lookup_table()
        assert "Z" in table["11"]

    def test_shor_code_info(self):
        from quanta.qec.codes import ShorCode
        code = ShorCode()
        info = code.info
        assert info.n == 9
        assert info.k == 1
        assert info.d == 3
        assert info.correctable_errors == 1

    def test_shor_encode(self):
        from quanta.qec.codes import ShorCode
        code = ShorCode()
        enc = code.encode()
        assert enc is not None

    def test_shor_decode(self):
        from quanta.qec.codes import ShorCode
        code = ShorCode()
        dec = code.decode()
        assert dec is not None

    def test_shor_lookup(self):
        from quanta.qec.codes import ShorCode
        code = ShorCode()
        table = code.lookup_table()
        assert "No error" in table["0000"]

    def test_correct_error_unknown(self):
        """Unknown syndrome returns 'No error detected'."""
        from quanta.qec.codes import BitFlipCode, correct_error
        code = BitFlipCode()
        assert "No error" in correct_error(code, "99")
