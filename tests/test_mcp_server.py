"""Tests for quanta.mcp_server — AI Agent MCP tool validation."""

import json

from quanta.mcp_server import (
    cluster_data,
    compare_decoders,
    create_bell_state,
    draw_circuit,
    estimate_fault_tolerant_cost,
    explain_result,
    grover_search,
    ibm_backends,
    list_gates,
    monte_carlo_price,
    optimize_circuit,
    option_greeks,
    qaoa_optimize,
    qec_diagnose,
    quanta_reasoning_eval,
    run_circuit,
    sdk_info,
    shor_factor,
    simulate_noise,
    surface_code_simulate,
    transpile_for_target,
)


def test_sdk_info_resource():
    info_json = sdk_info()
    data = json.loads(info_json)
    assert data["name"] == "Quanta Quantum SDK"
    assert data["version"] == "0.9.2"
    assert data["total_tools"] == 23
    assert data["total_gates"] == 31


def test_create_bell_state():
    res = create_bell_state(shots=500, seed=42)
    assert "Bell" in res or "00" in res
    assert "11" in res


def test_list_gates():
    gates_json = list_gates()
    data = json.loads(gates_json)
    assert data["total_gates"] == 31
    gate_names = [g["name"] for g in data["gates"]]
    assert any("H" in name for name in gate_names)
    assert any("CX" in name for name in gate_names)


def test_explain_result():
    counts = json.dumps({"00": 512, "11": 512})
    explanation = explain_result(counts)
    exp_lower = explanation.lower()
    assert (
        "50" in explanation
        or "Bell" in explanation
        or "entangle" in exp_lower
        or "00" in explanation
    )


def test_grover_search():
    res = grover_search(num_qubits=3, target=5, shots=500)
    assert "target" in res.lower() or "101" in res or "grover" in res.lower()


def test_shor_factor():
    res = shor_factor(number=15)
    assert "3" in res and "5" in res


def test_simulate_noise():
    res = simulate_noise(noise_type="depolarizing", probability=0.05, shots=100)
    assert "noise" in res.lower() or "fidelity" in res.lower() or "00" in res


def test_monte_carlo_and_greeks():
    price_res = monte_carlo_price(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert "price" in price_res.lower() or "call" in price_res.lower()

    greeks_res = option_greeks(
        spot=100.0, strike=100.0, volatility=0.2, rate=0.05, time_to_expiry=1.0
    )
    assert "delta" in greeks_res.lower()


def test_qaoa_optimize():
    res = qaoa_optimize(num_bits=3, problem="max_cut")
    assert "qaoa" in res.lower() or "bitstring" in res.lower() or "cost" in res.lower()


def test_cluster_data():
    res = cluster_data()
    assert "cluster" in res.lower() or "labels" in res.lower() or "centroid" in res.lower()


def test_draw_circuit():
    code = (
        "@circuit(qubits=2)\n"
        "def circ(q):\n"
        "    H(q[0])\n"
        "    CX(q[0], q[1])\n"
        "    return measure(q)\n"
    )
    svg_res = draw_circuit(code=code)
    assert "<svg" in svg_res or "svg" in svg_res.lower()


def test_qec_tools():
    surf_res = surface_code_simulate(distance=3, error_rate=0.01)
    assert "surface" in surf_res.lower() or "logical" in surf_res.lower()

    comp_res = compare_decoders(distance=3, error_rate=0.01)
    comp_lower = comp_res.lower()
    assert "mwpm" in comp_lower or "union-find" in comp_lower or "decoder" in comp_lower

    diag_res = qec_diagnose(code="bitflip", syndrome="1")
    diag_lower = diag_res.lower()
    assert "bitflip" in diag_lower or "error" in diag_lower or "syndrome" in diag_lower


def test_optimize_circuit():
    code = (
        "@circuit(qubits=1)\n"
        "def circ(q):\n"
        "    H(q[0])\n"
        "    H(q[0])\n"
        "    return measure(q)\n"
    )
    res = optimize_circuit(circuit_code=code)
    assert "optimized" in res.lower() or "gates" in res.lower() or "depth" in res.lower()


def test_run_circuit_sandboxed():
    valid_code = (
        "@circuit(qubits=2)\n"
        "def circ(q):\n"
        "    H(q[0])\n"
        "    CX(q[0], q[1])\n"
        "    return measure(q)\n"
    )
    res = run_circuit(valid_code, shots=100)
    assert "00" in res or "counts" in res.lower()

    # Disallowed code should raise safe error
    bad_code = "import os; os.system('echo hacked')"
    bad_res = run_circuit(bad_code)
    assert "blocked" in bad_res.lower() or "error" in bad_res.lower()


def test_ibm_backends():
    res = ibm_backends()
    assert "ibm" in res.lower() or "backends" in res.lower()


def test_estimate_fault_tolerant_cost():
    code = (
        "@circuit(qubits=2)\n"
        "def circ(q):\n"
        "    H(q[0])\n"
        "    CX(q[0], q[1])\n"
        "    return measure(q)\n"
    )
    res = estimate_fault_tolerant_cost(
        circuit_code=code,
        target_logical_error_rate=1e-10,
        physical_error_rate=1e-3,
    )
    data = json.loads(res)
    assert data["logical_qubits"] == 2
    assert "surface_code_distance" in data
    assert data["total_physical_qubits"] > 0
    assert "physical_qubits_t_factory" in data


def test_quanta_reasoning_eval():
    code = (
        "@circuit(qubits=2)\n"
        "def circ(q):\n"
        "    H(q[0])\n"
        "    CX(q[0], q[1])\n"
        "    return measure(q)\n"
    )
    res = quanta_reasoning_eval(circuit_code=code)
    data = json.loads(res)
    assert data["num_qubits"] == 2
    assert data["total_gates"] == 2
    assert "entangling_gate_ratio" in data
    assert "agent_feedback" in data


def test_transpile_for_target():
    code = (
        "@circuit(qubits=2)\n"
        "def circ(q):\n"
        "    H(q[0])\n"
        "    CX(q[0], q[1])\n"
        "    return measure(q)\n"
    )
    res = transpile_for_target(circuit_code=code, target="ibm_heron")
    data = json.loads(res)
    assert data["target"] == "ibm_heron"
    assert "transpiled_gate_count" in data
    assert "target_native_gates" in data
    assert "qasm3_output" in data

