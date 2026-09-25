# Quanta SDK Empirical Benchmark Results (v1.2.0-production)

> Benchmark Date: September 2026  
> Test Platform: Apple Silicon M-Series (macOS Darwin ARM64), Python 3.12+, NumPy 2.x, PyTorch 2.x, MLX  
> Verification Suite: `benchmarks/run_paper_benchmarks.py` · 2,076 Passing Automated Tests  
> Software DOI: [10.5281/zenodo.22952779](https://doi.org/10.5281/zenodo.22952779)

## 1. Simulation & Hardware Acceleration Benchmarks

| Benchmark Scenario | Scale / Configuration | Execution Time | Throughput / Speedup | Notes & Verifications |
|--------------------|-----------------------|----------------|----------------------|-----------------------|
| **SIMD Clifford Engine** | 100,000 Random Clifford Gates | **31.91 ms** | **3,133,813 gates/s** | Aaronson-Gottesman binary tableau |
| **Apple Silicon MLX (Metal)** | 24-Qubit Dense Statevector | **78.42 ms** | **52.09× Peak Speedup** | Zero-copy Unified Memory (vs CPU 4.08s) |
| **Matrix Product States (MPS)** | 250-Qubit GHZ State Preparation | **3.42 ms** | Low-rank tensor network | Schmidt truncation $\chi=64$ |
| **Statevector Bell State** | 2 Qubits, 1024 Shots | **0.14 ms** | End-to-end pipeline | Build + DAG + sim + measurement |
| **Statevector GHZ State** | 10 Qubits ($2^{10}$ amplitudes) | **0.28 ms** | Exact statevector | $\|\|\psi\|\|_2 = 1.000000$ |
| **Statevector GHZ State** | 20 Qubits ($2^{20}$ amplitudes) | **64.12 ms** | Dense tensor contraction | 1,048,576 complex128 amplitudes |
| **Grover Search** | 8 Qubits, Target=42 | **0.18 ms** | Quadratic oracle speedup | $P(\text{target}) > 0.99$ |
| **VQE Ground State ($H_2$)** | 2 Qubits, 2-Layer Ansatz | **184.50 ms** | Exact convergence | 3-term Pauli decomposition |

---

## 2. Quantum Error Correction (FTQC) Decoding Latencies

Empirical execution times for syndrome extraction and decoding:

| QEC Decoder & Code | Architecture / Parameters | Decode Time (ms) | Success / Fidelity |
|-------------------|---------------------------|------------------|---------------------|
| **Gross qLDPC Normalized BP-OSD-0** | Canonical $[[144, 12, 12]]$ Bivariate Bicycle | **1.54 ms** | 12 Logical Qubits, $12\times$ qubit saving |
| **Edmonds Blossom MWPM** | Rotated Surface Code $d=3$ | **0.29 ms** | Exact minimum weight matching |
| **Edmonds Blossom MWPM** | Rotated Surface Code $d=5$ | **0.82 ms** | Sub-millisecond FTQC loop |
| **Edmonds Blossom MWPM** | Rotated Surface Code $d=7$ | **2.41 ms** | Full spacetime graph matching |
| **Steane Code Lookup Decoder** | Steane $[[7, 1, 3]]$ | **0.04 ms** | Single-qubit error correction |
| **15-to-1 Bravyi-Kitaev Distillation** | Magic State $|T\rangle$ Factory | **0.62 ms** | Error suppression $\epsilon_{\text{out}} \le 35 p^3$ |

---

## 3. Autograd & Mathematical Gradient Benchmarks (`quanta.torch`)

| Gradient Method | Mathematical Precision | Latency | Baseline Error |
|-----------------|------------------------|---------|----------------|
| **Daleckii-Krein Fréchet Autograd** | Machine Precision ($\epsilon = 9.99\times 10^{-16}$) | **0.12 ms** | Padé approximation error: $>1.3\times 10^{-6}$ |
| **Parameter-Shift Rule (Exact)** | Analytically Exact ($\epsilon = 0$) | **0.38 ms** | 5 circuit evaluations ($2n+1$) |
| **Natural Gradient (QFIM)** | Fubini-Study Riemannian metric | **0.84 ms** | Exact Fisher information inversion |

---

## Methodology & Reproducibility

1. **Zero Mock Policy**: All benchmark figures are measured on actual silicon execution paths; no synthetic sleeps or interpolated values are permitted.
2. **Timing Protocol**: Median of 5 consecutive runs with garbage collection suppression.
3. **Execution Script**: Run `python benchmarks/run_paper_benchmarks.py` to reproduce the full benchmark matrix.

---

## Authorship & Attribution

- **Principal Architect & Author**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Affiliation**: ONMARTECH Quantum Computing Initiative (`info@onmartech.com`)

