# Quanta SDK Benchmark Results

> Generated: 2026-03-29 00:20
> Python 3.13.7, NumPy 2.3.2

## Results

| Benchmark | Time (ms) | Notes |
|-----------|----------|-------|
| Bell State (2q, 1024 shots) | 0.45 | Build+DAG+sim+sample |
| GHZ State (10q) | 0.69 | 2^10 = 1024 amplitudes |
| GHZ State (20q) | 82.77 | 2^20 = 1M amplitudes |
| Random Circuit (8q, ~50 gates) | 2.30 | 4096 shots |
| Grover Search (4q) | 0.38 | Target=7 |
| Grover Search (8q) | 0.49 | Target=42, 256 states |
| VQE H₂ (2q, 2 layers) | 301.08 | 3-term Hamiltonian |
| Estimator ⟨ZZ⟩ (2q) | 0.09 | Exact statevector |
| Parameter-Shift Gradient (2q, 2 params) | 0.51 | 5 circuit evals |
| Sampler Batch (10×2q) | 4.16 | 10 circuits × 1024 shots |

## Methodology

- Each benchmark runs 3–5 iterations, median reported
- Includes full pipeline: circuit build → DAG → simulation → sampling
- No hardware; pure statevector simulation
- Parameter-shift gradient includes 5 circuit evaluations (cost + 2×2 shifts)

## Hardware

- Python: 3.13.7
- NumPy: 2.3.2
- All tests on CPU (no GPU acceleration)
