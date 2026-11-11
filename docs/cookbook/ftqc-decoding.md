# Fault-Tolerant Quantum Error Correction (FTQC) — Quick Recipe

> Implement syndrome extraction and optimal decoding using Edmonds' Blossom MWPM and modern Bivariate Bicycle qLDPC codes.

---

## 1. Surface Code Decoding with Edmonds' Blossom MWPM

Quanta provides a native, globally optimal **Minimum Weight Perfect Matching (MWPM)** decoder based on Edmonds' Blossom algorithm paired with virtual boundary node replication. Use `simulate_error_correction(...)` for single-shot **code-capacity** bit-flip decoding (~10.3% threshold), or `simulate_dynamic(...)` for multi-round **phenomenological space-time** syndrome extraction with noisy readout (~2.9% threshold).

### Python Code

```python
from quanta.qec.surface_code import SurfaceCode
from quanta.qec.decoder import MWPMDecoder

# 1. Initialize distance-3 Surface Code [[9, 1, 3]]
code = SurfaceCode(distance=3)
print(code.summary())

# 2. Simulate 100 rounds of code-capacity error correction with true Blossom matching
# (For multi-round phenomenological readout noise, use code.simulate_dynamic(...))
decoder = MWPMDecoder()
result = code.simulate_error_correction(
    error_rate=0.01,
    rounds=100,
    decoder=decoder,
    seed=42,
)

print(f"Logical Error Rate:  {result.logical_error_rate:.4f}")
print(f"Physical Error Rate: {result.physical_error_rate:.4f}")
print(f"Errors Corrected:    {result.errors_corrected} / {result.errors_injected}")
print(f"Threshold Estimate:  {result.threshold_estimate:.1%}")
```

---

## 2. Bivariate Bicycle qLDPC [[144, 12, 12]] Code

Beyond 2D surface codes, Quanta natively supports the canonical Gross $[[144, 12, 12]]$ Bivariate Bicycle quantum LDPC code over the group ring $\mathbb{F}_2[x, y] / \langle x^{12}-1, y^6-1 \rangle$.

### Python Code

```python
import numpy as np
from quanta.qec.qldpc import BivariateBicycleCode, BPOSDDecoder

# 1. Construct Gross [[144, 12, 12]] code
# Encodes k=12 logical qubits into n=144 physical qubits with code distance d=12
code = BivariateBicycleCode.gross_144_12_12()
print(f"Physical Qubits (n): {code.n}")          # 144
print(f"Logical Qubits (k):  {code.k}")          # 12
print(f"Encoding Rate (k/n): {code.k / code.n:.4f}")  # 0.0833 (12x higher than d=12 surface code)

# 2. Decode a syndrome using native BP-OSD on the H_Z parity-check matrix
decoder = BPOSDDecoder(code.H_Z, max_bp_iter=30)

# Simulate a physical X error on physical qubit 7
error_x = np.zeros(code.n, dtype=int)
error_x[7] = 1

# Calculate Z-stabilizer syndrome: s_z = H_Z * e_x mod 2
syndrome_z = (code.H_Z @ error_x) % 2

# Decode syndrome into physical correction
correction = decoder.decode(syndrome_z)
print(f"Decoding Success: {correction.success}")
print(f"Corrected Qubits: {correction.correction}")
```

---

## 3. 15-to-1 Bravyi-Kitaev Magic State Distillation

For non-Clifford logical gates ($T$ gate, $CCZ$), Quanta provides executable distillation factories:

```python
from quanta.qec.distillation import BravyiKitaev15to1Factory

# Initialize 15-to-1 Bravyi-Kitaev factory
factory = BravyiKitaev15to1Factory()

# Distill raw noisy magic states (input error rate p = 0.01)
result = factory.distill(input_error_rate=0.01, shots=1000, seed=42)

print(f"Input Error Rate:  {result.input_error_rate:.4f}")
print(f"Output Error Rate: {result.output_error_rate:.6f}")  # Cubic suppression: ~35 * p^3
print(f"Acceptance Prob:   {result.acceptance_probability * 100:.1f}%")
```

---

## See Also
- [2026 Scientific Audit Report](../scientific_audit_september_2026.md)
- [API Reference — QEC Decoder](../api/qec/decoder.md)
- [API Reference — qLDPC Codes](../api/qec/qldpc.md)
