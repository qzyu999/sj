# Paper Strategy: Full SJ Generalization + Practical Recommendations

## The Paper's Argument (in order)

### Section 1: Introduction
- KDE bandwidth selection is the fundamental problem
- SJ (1991) solved it definitively for 1D → became the default for 35 years
- No comparable method exists for d > 1 (people use Silverman or cross-validation)
- We provide the complete d-D generalization

### Section 2: Background
- AMISE in whitened coordinates
- The roughness functional Ψ₄ and why it's the key quantity
- SJ's architecture: normal reference → kernel functional estimates → adaptive pilot → STE

### Section 3: The Full d-D Sheather-Jones Algorithm
Present the COMPLETE generalization, mirroring SJ's structure:

**3.1 The Roughness Polynomial P_d (estimates Ψ₄)**
- Theorem: closed-form pairwise contribution
- P_d(t) = t²/16 − (d+2)t/4 + d(d+2)/4
- Proof via Gaussian product lemma + moment expansion
- Verification: d=1 recovers SJ's φ⁽⁴⁾ exactly

**3.2 The Next-Order Polynomial R_d (estimates Ψ₆)**
- Theorem: gradient-of-Laplacian pairwise contribution
- R_d(t) = −t³/64 + 3(d+4)t²/32 − 3(d+2)(d+4)t/16 + d(d+2)(d+4)/8
- Derivation: dot product of vector fields → degree-3 polynomial
- Verification: d=1 recovers φ⁽⁶⁾ in the integral-of-products framework

**3.3 Normal References**
- Ψ₄^NR(σ) = d(d+2) / (4(4π)^{d/2} σ^{d+4})
- Ψ₆^NR(σ) = d(d+2)(d+4) / (8(4π)^{d/2} σ^{d+6})
- Verified at d=1 against known values

**3.4 Bias Cancellation and Pilot Bandwidth**
- Diagonal contribution (positive): P_d(0)/(n(4π)^{d/2}g^{d+4}) 
- Smoothing bias (negative): g² × Ψ₆
- Cancellation → g^{d+6} = d(d+2) / (4n(4π)^{d/2} Ψ₆)
- Normal-reference pilot: g = σ̂(2/(n(d+4)))^{1/(d+6)}

**3.5 The Complete Algorithm (Three Variants)**

1. **Two-Stage DPI** (direct plug-in — recommended):
   - NR pilot → Ψ̂₆ → adaptive pilot → Ψ̂₄ → h*
   - Most accurate, moderate cost

2. **STE** (solve-the-equation — SJ's original):
   - Root-find h = h_AMISE(Ψ̂₄(α(h)))
   - Theoretically optimal rate, but coupled iteration

3. **One-Stage DPI** (simplified):
   - Silverman pilot → Ψ̂₄ → h*
   - Fastest, slightly less accurate on multimodal data

### Section 4: Computational Methods
- Exact: O(n² d) — feasible for n ≤ 3000
- Subsampling: O(m d) — constant-time in n, 1-2% bandwidth error at m=80K
- KD-tree: O(n log n) with truncation at 8h₀
- FFT: O(nd + M^d log M) for d ≤ 3
- GPU: trivially parallelizable pairwise computation

**Key argument for practice:** The subsampling strategy means ALL variants (one-stage, two-stage, STE) have the same computational cost: O(m·d) per roughness evaluation. The difference is only how many evaluations:
- One-stage: 1 evaluation
- Two-stage: 2 evaluations (Ψ₆ + Ψ₄)  
- STE: ~5-10 evaluations (Newton iterations)

At m=80K, each evaluation takes ~100ms. So:
- One-stage: ~100ms
- Two-stage: ~200ms  
- STE: ~500ms-1s

ALL are fast enough for modern use. The question is only accuracy vs simplicity.

### Section 5: When to Use Which Variant

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Quick exploration, n < 1000 | One-stage | Silverman pilot is fine at small n |
| Production, need best accuracy | Two-stage DPI | Best balance of accuracy and simplicity |
| Formal statistical analysis | STE | Provably optimal rate (in 1D; conjectured in d-D) |
| Streaming/real-time monitoring | One-stage + subsample | Fastest path to a reasonable answer |
| Anomaly detection | Two-stage | Tighter bandwidth → sharper density boundary |

### Section 6: Empirical Evaluation
- ISE on synthetic mixtures (2D, 3D, 5D)
- Anomaly detection AUC (structured data + embeddings)
- Timing comparison across methods
- Comparison with LSCV (cross-validation)

### Section 7: Discussion
- When Silverman is (surprisingly) fine: unimodal data, high d (concentration of measure)
- When adaptive methods matter: multimodal data, d=2-15 (the embedding sweet spot)
- The clustering caveat (density-optimal ≠ mode-optimal)
- Future: diagonal bandwidths, formal rate proof

---

## What Goes in the gsj Package (v0.2.0)

### New Public API

```python
from gsj import bandwidth

# Current (one-stage DPI — stays as default for backward compat)
h = bandwidth(X)

# New: select algorithm variant
h = bandwidth(X, algorithm='one-stage')    # Current behavior (default)
h = bandwidth(X, algorithm='two-stage')    # NEW: full SJ generalization
h = bandwidth(X, algorithm='ste')          # NEW: solve-the-equation

# Low-level access to functionals
from gsj import roughness, roughness_gradient

psi4 = roughness(X, h)           # Ψ₄ estimate at bandwidth h
psi6 = roughness_gradient(X, h)  # Ψ₆ estimate at bandwidth h
```

### Implementation Plan

1. Add `R_d` polynomial to `_core.py` (alongside existing `P_d`)
2. Add `_compute_psi6()` function (same structure as `_roughness_*_nd`)
3. Add `bandwidth_nd_two_stage()` and `bandwidth_nd_ste()` 
4. Add `algorithm` parameter to `bandwidth()` entry point
5. Expose `roughness()` and `roughness_gradient()` as public functions
6. Update `__init__.py` exports
7. Version bump to 0.2.0
8. Tests: verify d=1 against scipy's `bw_method='scott'` and our own 1D implementation

### What NOT to Change
- Default behavior (`bandwidth(X)` stays one-stage for backward compat)
- The method/backend/subsample_size parameters
- The GPU backends (they just compute pairwise distances — same for all algorithms)

---

## Paper Framing: Why This Matters

The key argument is NOT "our method is slightly better than Silverman." It's:

> **SJ (1991) has been the gold standard for bandwidth selection for 35 years but was limited to 1D. We provide the complete multivariate generalization — including the adaptive two-stage pilot that gives SJ its superior convergence properties — for the first time. We then show that for modern big-data applications (n > 10⁵, d > 5), the simpler one-stage variant achieves nearly identical accuracy due to the (d+4)-th root dampening of pilot errors, making the full SJ algorithm a theoretical contribution that simplifies beautifully in practice.**

This is a stronger story than either:
- "Here's a better bandwidth selector" (incremental)
- "Here's the full SJ in d-D" (theoretical only)

It's both: the complete theory + the practical simplification + the empirical evidence that the simplification costs almost nothing.
