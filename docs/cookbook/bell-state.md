# Bell State — Quick Recipe

> Copy-paste and run. No explanation needed.

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result.summary())
# |00⟩ ≈ 50%, |11⟩ ≈ 50%
```
