# Quanta SDK Benchmark Results

> Generated: 2026-03-29 02:06
> Python 3.13.7, NumPy 2.3.2

## Results

| Benchmark | Time (ms) | Notes |
|-----------|----------|-------|
| Bell State (2q, 1024 shots) | 0.18 | Build+DAG+sim+sample |
| GHZ State (10q) | 0.35 | 2^10 = 1024 amplitudes |
| GHZ State (20q) | 81.49 | 2^20 = 1M amplitudes |
| Random Circuit (8q, ~50 gates) | 1.27 | 4096 shots |
| Grover Search (4q) | 0.12 | Target=7 |
| Grover Search (8q) | 0.21 | Target=42, 256 states |
| VQE H₂ (2q, 2 layers) | 277.23 | 3-term Hamiltonian |
| Estimator ⟨ZZ⟩ (2q) | 0.08 | Exact statevector |
| Parameter-Shift Gradient (2q, 2 params) | 0.46 | 5 circuit evals |
| Sampler Batch (10×2q) | 1.40 | 10 circuits × 1024 shots |

## Methodology

- Each benchmark runs 3–5 iterations, median reported
- Includes full pipeline: circuit build → DAG → simulation → sampling
- No hardware; pure statevector simulation
- Parameter-shift gradient includes 5 circuit evaluations (cost + 2×2 shifts)

## Hardware

- Python: 3.13.7
- NumPy: 2.3.2
- All tests on CPU (no GPU acceleration)
