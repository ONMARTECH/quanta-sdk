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

## 2. Rigorous Two-Regime 3-Arm Ablation (`benchmarks/cvp_benchmark.py`)

To isolate the marginal contribution of each architectural component without ground-truth or fallback leakage, `benchmarks/cvp_benchmark.py` (`run_three_arm_ablation()`) evaluates **three independent arms** across **two distinct geometric regimes**:

1. **Arm 1 — Classical LLL + Babai Alone (`classical_lll_babai`)**: Polynomial-time LLL reduction ($\delta=0.75$) followed by Babai Nearest Plane rounding.
2. **Arm 2 — Standalone Subspace QAOA (`standalone_subspace_qaoa`, `allow_babai_fallback=False`)**: Raw decoded QAOA coordinates over active subspace coordinates with **zero Babai fallback clamping**.
3. **Arm 3 — Hybrid Babai + Subspace QAOA (`hybrid_babai_qaoa`, `allow_babai_fallback=True`)**: Classical LLL+Babai preconditioning paired with subspace QAOA refinement.

### Regime A: Easy Bounded Distance Decoding (`easy_bdd`, $\|\epsilon\|_2 = 0.40 \cdot \lambda_{\min} < 0.5 \cdot \lambda_1(B)$)

| Pipeline Configuration | Tested Dimensions ($n$) | Success Rate (Exact Match) | Execution Time | Status / True Mathematical Driver |
| :--- | :---: | :---: | :---: | :--- |
| **Arm 1: Classical LLL + Babai Alone** | $n=4, 8, 12, 16$ | **100.0%** | **~0.001 s** | **Classical linear algebra solved the unique-decoding Voronoi cell completely.** |
| **Vanilla Full-Space QAOA (No Subspace)** | $n \ge 12$ ($N > 16$ qubits) | **N/A** | Skipped (`CLASSICAL_SV_RAM_LIMIT`) | **`CLASSICAL_SV_RAM_LIMIT` ($2^N > 65,536$ classical dense statevector RAM ceiling, distinct from a quantum barren plateau).** |
| **Arm 3: Hybrid (Babai + Zeno + QAOA)** | $n=4, 8, 12, 16$ | **100.0%** | ~0.023 s | **Carried 100% by Babai; QAOA added latency with `0.0%` honest marginal lift.** |

### Regime B: Hard Adversarial Voronoi-Boundary (`hard_voronoi`, $\|\epsilon\|_2 \in [0.58, 0.85] \cdot \lambda_1(B)$)

In `generate_adversarial_voronoi_cvp()`, bases are constructed with coupled off-diagonal Gram-Schmidt shear ($\mu_{ij} \in [0.44, 0.485]$) and top-hyperplane boundary perturbations ($|c_{n-1}| \in [0.54, 0.57]$) where greedy Babai rounding provably selects a suboptimal hyperplane on low-dimensional instances ($n \in \{4, 6, 8\}$):
- **Arm 1 (Classical LLL + Babai)**: Fails hyperplane rounding on sheared Voronoi-boundary targets.
- **Arm 2 (Standalone Subspace QAOA, `allow_babai_fallback=False`)**: Evaluates pure variational subspace search without Babai clamping, measuring raw variational coordinate recovery.
- **Arm 3 (Hybrid Babai + Subspace QAOA, `allow_babai_fallback=True`)**: Can recover local multi-coordinate integer offsets only within the small active subspace window ($k \le 4$ active coordinates), while still failing whenever high-dimensional error cascading ($n \ge 512$) freezes corrupted hyperplanes outside the active window.

### Key Ablation Takeaways:
- **Easy BDD Marginal Quantum Lift:** **`0.0%`** over classical LLL + Babai (`marginal_lift_over_babai_pct = 0.0`), with a $1\text{ ms} \to 23\text{ ms}$ simulation latency penalty.
- **Classical Simulation Cutoff Honesty:** Dense statevector cutoffs above 16 qubits ($2^N > 65,536$) are explicitly tagged **`CLASSICAL_SV_RAM_LIMIT`** in `benchmarks/cvp_benchmark.py` rather than conflating classical RAM exhaustion with asymptotic quantum barren plateaus.
- **Adversarial Voronoi Scope:** Local low-dimensional boundary improvements in `hard_voronoi` ($n \le 8$) reflect small-window discrete neighborhood search and **do not scale** to cryptographic LWE dimensions ($n \ge 512$).

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
1. **SARI, Abdullah Enes.** (2026). *Quanta: Native High-Performance Quantum Computing Framework & Fault-Tolerant Architecture* (v1.2.0). [GitHub Release v1.2.0](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0). Official Portal: [quanta.onmartech.com](https://quanta.onmartech.com).
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
11. **McClean, Jarrod R., Boixo, Sergio, Smelyanskiy, Vadim N., Babbush, Ryan, & Neven, Hartmut.** (2018). *Barren plateaus in quantum neural network training landscapes*. Nature Communications, 9(1), 4812. [DOI: 10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4).
12. **Cerezo, Marco, Sone, Akira, Volkoff, Tyler, Cincio, Lukasz, & Coles, Patrick J.** (2021). *Cost function dependent barren plateaus in shallow parametrized quantum circuits*. Nature Communications, 12(1), 1791. [DOI: 10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w).
13. **Goodhart, Charles A. E.** (1984). *Problems of Monetary Management: The U.K. Experience*. In Monetary Theory and Practice, Macmillan, pp. 91–99. ("Goodhart's Law").
