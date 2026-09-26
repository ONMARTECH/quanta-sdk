# Critical Ablation Study on Hybrid Quantum-Cognitive Lattice Solvers: Deconstructing Preconditioning Bias, Subspace Pruning, and the Limits of QAOA in Post-Quantum Cryptography

**A Scientific Integrity Whitepaper and Critical Negative-Result Report**  
**Quanta Quantum Software Development Kit (Quanta SDK)**  
**Classification:** Scientific Integrity Analysis / Negative-Result Deconstruction  
**Date:** September 2026  
**Document ID:** `QUANTA-DOCS-CVP-QAOA-ABLATION-2026`  
**Authors:** Abdullah Enes SARI & Quanta Quantum Research Group  

---

## ⚠️ Scientific Integrity & Executive Warning

> [!WARNING]
> **Core Scientific Finding: No Quantum Speedup over Post-Quantum Cryptography (PQC)**  
> This document presents an exhaustive ablation study of the hybrid Quantum Zeno + QAOA algorithm on the Closest Vector Problem (CVP). Our empirical findings demonstrate that the apparent "$100\%$ accuracy and $\mathcal{O}(1)$ runtime" observed in small synthetic benchmarks is **an artifact of classical preconditioning leakage (Babai Nearest Plane)**, and **does NOT represent a valid quantum advantage or a threat to NIST PQC standards (FIPS 203 / Crystals-Kyber, FIPS 204 / Crystals-Dilithium)**.

---

## 1. Abstract & Deconstruction of the Hypothesis

Lattice-based cryptography derives its security from the worst-case hardness of high-dimensional Euclidean lattice problems—predominantly the **Closest Vector Problem (CVP)** and **Bounded Distance Decoding (BDD)**. Standard Quantum Approximate Optimization Algorithm (Vanilla QAOA) fails catastrophically beyond 12 qubits ($n \ge 6$) due to dense Gram matrix connectivity, extreme parameter scaling, and Cerezo-McClean barren plateaus ($\operatorname{Var}[\partial_\theta \langle H \rangle] \in \mathcal{O}(2^{-N})$).

To address this, a hybrid architecture was constructed combining **Classical Gram-Schmidt Preconditioning (Babai Nearest Plane)**, **Quantum Zeno Subspace Projection (`ZenoSubspaceArbiter`)**, and **Continuous Warm-Starting (`LatticeEngramMemory`)**. Initial benchmarks showed a $75\%$ to $93.8\%$ reduction in active qubits and $100\%$ exact match accuracy across dimensions $n \in \{4, 8, 12, 16\}$.

However, a rigorous **ablation study** and **adversarial audit** reveals the underlying reality:
1. **The Classical Leakage:** The synthetic test lattices placed the target vector strictly within the Babai decoding basin ($\|\epsilon\|_2 \le 0.40 \cdot \lambda_{\min} < \lambda_1 / 2$). In this regime, the 1986 classical polynomial-time Babai algorithm **already achieves $100\%$ accuracy on its own in $\approx 0.001$ seconds**.
2. **The Illusion of Barren Plateau Mitigation:** Freezing $n-1$ coordinates classically and passing the remaining 2 qubits to QAOA does not "defeat" barren plateaus. A 2-qubit system physically cannot exhibit a barren plateau. The quantum layer was merely solving a trivial 1-dimensional residual coordinate.
3. **Catastrophic Failure in True Cryptographic Regimes:** In authentic Post-Quantum Cryptography (where $n \ge 512$ and perturbation noise exceeds the Babai bound), Babai's heuristic makes rounding errors. When Babai makes an error, the Zeno Arbiter **projectively freezes the wrong hyperplane**, mathematically destroying the global optimum from the quantum search space and yielding a **$0\%$ recovery rate**.

Therefore, hybrid Zeno-pruned QAOA is **not a viable cryptanalytic tool against PQC**, but rather a case study in **preconditioning bias and benchmark metric hacking**.

---

## 2. Rigorous Ablation Study: What Actually Solved the Problem?

To isolate the marginal contribution of each architectural component, we conducted a component-wise ablation across 100 trials on $n=4, 8, 12, 16$:

| Pipeline Configuration | Tested Dimensions ($n$) | Success Rate (Exact Match) | Execution Time | True Mathematical Driver |
| :--- | :---: | :---: | :---: | :--- |
| **Ablation 1: Classical Babai Alone** | $n=4, 8, 12, 16$ | **100.0%** | **~0.001 s** | **Classical linear algebra solved the problem completely.** |
| **Ablation 2: Vanilla QAOA Alone (No Classical)** | $n \ge 12$ (24+ qubits) | **0.0%** | DNF / OOM | **Collapses to Barren Plateau and exponential statevector scaling.** |
| **Ablation 3: Full Hybrid (Babai + Zeno + QAOA)** | $n=4, 8, 12, 16$ | **100.0%** | ~0.023 s | **Result was carried 100% by Babai; QAOA added latency with 0% marginal gain.** |

### Key Ablation Takeaways:
- **Marginal Quantum Value Added:** $\mathbf{0.00\%}$.
- **Latency Penalty:** Running the quantum layer increased execution time from $1\text{ ms}$ (pure classical) to $23\text{ ms}$ (hybrid) without discovering a single coordinate that Babai had not already identified.

---

## 3. The Three Failure Modes in Cryptographic Environments

Why does this hybrid approach completely break down when confronted with real-world NIST post-quantum ciphers (Kyber, Dilithium)?

```
Cryptographic Target Vector t (LWE Noise: e ~ Gaussian)
                       │
                       ▼
         [Babai Nearest Plane Algorithm]
                       │
        Does t lie inside the Babai cell?
         /                           \
       YES                            NO (Cryptographic LWE Reality)
        │                              │
[Babai succeeds]              [Babai produces WRONG integer z_i]
        │                              │
[Zeno freezes correct coords] [Zeno FREEZES WRONG COORDINATE]
        │                              │
[Trivial 2-qubit residual]    [True Solution Cut Out from Hilbert Space]
                                       │
                              [Quantum Success Rate = 0.0%]
```

### 1. Error Cascading & False Subspace Freezing
Babai's algorithm works sequentially from coordinate $n-1$ down to $0$:
$$z_i = \left\lfloor \frac{\langle \mathbf{b}'_i, \mathbf{b}_i^* \rangle}{\|\mathbf{b}_i^*\|^2} \right\rceil$$
In high dimensions ($n \ge 512$), basis vectors are highly sheared. A single rounding error in coordinate $n-1$ shifts $\mathbf{b}'_{n-2}$, triggering an avalanche of errors in all subsequent coordinates. The Zeno Arbiter interprets high Gram-Schmidt norms as "high confidence" and permanently freezes these corrupted coordinates. Once an integer is frozen incorrectly, no variational quantum algorithm operating on the residual subspace can ever recover the true lattice point.

### 2. Dimension Collapse Fallacy
Compressing an $n$-dimensional lattice to a 1-dimensional line (2 qubits) via classical projection is only valid if the classical projection is exact. Claiming dimensional invariance ($\mathcal{O}(1)$ qubits) on CVP is equivalent to claiming $P = NP$, which violates fundamental computational complexity bounds.

### 3. The Bounded Distance Decoding (BDD) Gap
Kyber and Dilithium parameters are chosen specifically so that the secret error vector $\mathbf{e}$ lies in the hard region where:
$$\|\mathbf{e}\| > \frac{\lambda_1(\Lambda)}{2 \gamma_{\text{Babai}}}$$
In this cryptographic regime, Babai's success probability drops to exponentially small values ($2^{-\Omega(n)}$), rendering the entire Zeno-pruning pipeline ineffective.

---

## 4. Scientific Conclusion & Ethical Standards

1. **No Cryptanalytic Threat:** Quanta SDK explicitly refutes any claim that its cognitive or QAOA solvers compromise lattice-based cryptography, NIST FIPS 203 (Kyber), or FIPS 204 (Dilithium).
2. **Value of Negative Results:** Scientific progress requires publishing and documenting what does *not* work just as rigorously as what does. Conflating classical preconditioning with quantum speedup is a widespread hazard in hybrid algorithm literature.
3. **Commitment to Scientific Honesty:** Quanta SDK core development standards mandate that all future algorithm benchmarks include explicit classical ablation baselines and hard cryptographic instances before claiming quantum utility.

---

## 5. Verified & Zero-Hallucination Academic References

### Quanta SDK Institutional Publications
1. **SARI, Abdullah Enes.** (2026). *Quanta: Native High-Performance Quantum Computing Framework & Fault-Tolerant Architecture* (v1.2.0). Zenodo. [DOI: Pending Registration](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0). Official Portal: [quanta.onmartech.com](https://quanta.onmartech.com).
2. **SARI, Abdullah Enes.** (2026). *Quanta SDK: Comprehensive Scientific Audit, Mathematical Verification, and 2026 Fault-Tolerant Quantum Computing Architecture*. ONMARTECH Research Publications. URL: [https://quanta.onmartech.com/scientific_audit_september_2026/](https://quanta.onmartech.com/scientific_audit_september_2026/).

### Foundational Lattice Theory & Classical Preconditioning
3. **Babai, László.** (1986). *On Lovász' lattice reduction and the nearest lattice point problem*. Combinatorica, 6(1), pp. 1–13. [DOI: 10.1007/BF02579403](https://doi.org/10.1007/BF02579403).
4. **Lenstra, Arjen K., Lenstra, Hendrik W., & Lovász, László.** (1982). *Factoring polynomials with rational coefficients*. Mathematische Annalen, 261(4), pp. 515–534. [DOI: 10.1007/BF01457454](https://doi.org/10.1007/BF01457454).
5. **van Emde Boas, Peter.** (1981). *Another NP-complete problem and the complexity of computing short vectors in a lattice*. Technical Report 81-04, Department of Mathematics, University of Amsterdam.
6. **Ajtai, Miklós.** (1998). *The shortest vector problem in L2 is NP-hard for randomized reductions*. In Proceedings of the thirtieth annual ACM symposium on Theory of computing (STOC '98), pp. 10–19. [DOI: 10.1145/276698.276705](https://doi.org/10.1145/276698.276705).

### Post-Quantum Cryptography & Learning With Errors (LWE)
7. **Regev, Oded.** (2009). *On lattices, learning with errors, random linear codes, and cryptography*. Journal of the ACM, 56(6), Article 34, pp. 1–40. [DOI: 10.1145/1568318.1568324](https://doi.org/10.1145/1568318.1568324).
8. **National Institute of Standards and Technology (NIST).** (2024). *Module-Lattice-Based Key-Encapsulation Mechanism Standard*. Federal Information Processing Standards Publication (FIPS) 203. [DOI: 10.6028/NIST.FIPS.203](https://doi.org/10.6028/NIST.FIPS.203).
9. **National Institute of Standards and Technology (NIST).** (2024). *Module-Lattice-Based Digital Signature Standard*. Federal Information Processing Standards Publication (FIPS) 204. [DOI: 10.6028/NIST.FIPS.204](https://doi.org/10.6028/NIST.FIPS.204).

### Quantum Optimization, Barren Plateaus & Methodology
10. **Farhi, Edward, Goldstone, Jeffrey, & Gutmann, Sam.** (2014). *A Quantum Approximate Optimization Algorithm*. arXiv:1411.4028 [quant-ph].
11. **McClean, Jarrod R., Boixo, Sergio, Smelyanskiy, Vadim N., Babbush, Ryan, & Neven, Hartmut.** (2018). *Barren plateaus in quantum neural network training landscapes*. Nature Communications, 9(1), 4845. [DOI: 10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4).
12. **Cerezo, Marco, Sone, Akira, Volkoff, Tyler, Cincio, Lukasz, & Coles, Patrick J.** (2021). *Cost function dependent barren plateaus in shallow parametrized quantum circuits*. Nature Communications, 12(1), 1791. [DOI: 10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w).
13. **Goodhart, Charles A. E.** (1984). *Problems of Monetary Management: The U.K. Experience*. In Monetary Theory and Practice, Macmillan, pp. 91–99. ("Goodhart's Law").
