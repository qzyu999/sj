# Analysis of Sheather & Jones (1991) vs Our Work

## The Original Paper's Structure

The paper is 8 pages in JRSS-B. It is NOT a theory paper — it's a **practical algorithm paper** that happens to have good theory behind it. The structure:

1. Introduction (motivation: improve on Park-Marron 1990)
2. Park-Marron's method (the baseline to beat)
3. The improved method (the SJ contribution)
4. Simulation results (the evidence it works)
5. Implementation details (the recipe)

The theory is mostly cited from companion papers (Jones & Sheather 1991, Hall & Marron 1987). The JRSS-B paper itself is the *packaging* — "here's a method that works reliably."

---

## What SJ (1991) Actually Proposes

### The Algorithm (ĥ_{2S})

```
INPUT: Data X_1, ..., X_n

STAGE 3 (Normal reference):
  λ̂ = interquartile range of data
  a = 0.920 * λ̂ * n^{-1/7}      (bandwidth for estimating R(f''))
  b = 0.912 * λ̂ * n^{-1/9}      (bandwidth for estimating R(f'''))

STAGE 2 (Kernel functional estimates):
  Ŝ_D(a)  = n^{-2} * a^{-5} * Σ_{i,j} φ^(4)((Xi-Xj)/a)     ← R(f'') estimate
  T̂_D(b)  = -n^{-2} * b^{-7} * Σ_{i,j} φ^(6)((Xi-Xj)/b)    ← R(f''') estimate

  where φ^(4)(z) = (z⁴ - 6z² + 3)φ(z)     [4th Hermite function]
        φ^(6)(z) = (z⁶ - 15z⁴ + 45z² - 15)φ(z)  [6th Hermite function]

STAGE 1 (Adaptive pilot):
  α̂₂(h) = 1.357 * (Ŝ_D(a) / T̂_D(b))^{1/7} * h^{5/7}

STAGE 0 (STE — Solve The Equation):
  Find h solving:  h = [R(K) / (σ_K⁴ * Ŝ_D(α̂₂(h)))]^{1/5} * n^{-1/5}
  (Use Newton-Raphson)

OUTPUT: ĥ_{2S}
```

### Key Formula: The Roughness Estimator

The core computation is:

$$\hat{S}_D(\alpha) = \frac{1}{n^2 \alpha^5} \sum_{i=1}^n \sum_{j=1}^n \phi^{(4)}\left(\frac{X_i - X_j}{\alpha}\right)$$

where $\phi^{(4)}(z) = (z^4 - 6z^2 + 3)\phi(z)$ and this is the SAME as our formula at d=1:

$$\frac{1}{n^2 (4\pi)^{1/2} h_0^5} \sum_{i,j} e^{-r_{ij}^2/4} \cdot P_1(r_{ij}^2)$$

with $P_1(t) = t^2/16 - 3t/4 + 3/4$.

Verification: $\phi^{(4)}(z) = (z^4 - 6z^2 + 3)\phi(z)$, evaluated at the PAIR difference scaled by $\alpha$, and the double-sum with the normalization gives exactly our formula. ✓

---

## Three Key Ideas in the Original Paper

### Idea 1: "Diagonals In" — Makes Ŝ Always Positive

Park-Marron used $\hat{S}_{ND}$ (no diagonals: $i \neq j$ only). This can go NEGATIVE for small bandwidths, creating discontinuities in the STE function.

SJ adds the $i = j$ terms back. Since $\phi^{(4)}(0) = 3 > 0$, the diagonal adds a guaranteed positive contribution:

$$\hat{S}_D(\alpha) = \hat{S}_{ND}(\alpha) + \frac{1}{n\alpha^5}\phi^{(4)}(0) = \hat{S}_{ND}(\alpha) + \frac{3}{n\alpha^5\sqrt{2\pi}}$$

This is ALWAYS positive → the STE function is smooth → Newton-Raphson works reliably.

**Our analog**: We ALREADY include diagonals (our sum runs $i,j = 1..n$ including $i=j$). At $r=0$: $P_d(0) = d(d+2)/4 > 0$. So our estimator is also always positive. ✓

### Idea 2: Bias Cancellation — The Core Insight

The diagonal term introduces a non-stochastic positive bias to $\hat{S}_D$:
$$E[\hat{S}_D(\alpha)] - R(f'') = \underbrace{\frac{\phi^{(4)}(0)}{n\alpha^5}}_{\text{positive (diagonal bias)}} + \underbrace{\frac{1}{2}\alpha^2 \sigma_K^2 R(f''')}_{\text{negative (smoothing bias)}} + O(\cdot)$$

By choosing $\alpha$ to cancel these two terms, you get a bias of $O(\alpha^4)$ instead of $O(\alpha^2)$, which is what gives the improved rate.

**The cancellation condition** gives $\alpha_2$:
$$\frac{\phi^{(4)}(0)}{n\alpha^5} \approx -\frac{1}{2}\alpha^2 \sigma_K^2 R(f''')$$

Solving: $\alpha^7 \propto n^{-1} / R(f''')$, hence $\alpha_2 \propto R(f''')^{-1/7} n^{-1/7}$.

### Idea 3: Solve-The-Equation (not just Plug-In)

Instead of computing $\hat{h}$ from equation (4) with a FIXED $\alpha$, SJ makes $\alpha$ depend on $h$ through the relation $\alpha_2(h) \propto h^{5/7}$, then solves the fixed-point equation numerically. This couples the bandwidth selection with the functional estimation, giving the $O(n^{-5/14})$ rate.

---

## Comparison: What We Do vs What SJ Does

| Aspect | SJ (1991) | Our Method | Gap |
|--------|-----------|------------|-----|
| **Roughness formula** | $\phi^{(4)}(z) = (z^4-6z^2+3)\phi(z)$ | $P_d(t) = t^2/16 - (d+2)t/4 + d(d+2)/4$ | None — exact generalization |
| **Diagonals** | Yes (the innovation) | Yes (included) | None |
| **Always positive** | Yes (the practical win) | Yes ($P_d(0) = d(d+2)/4 > 0$) | None |
| **Pilot bandwidth** | Adaptive two-stage: $\alpha_2(h) = 1.357(\hat{S}/\hat{T})^{1/7}h^{5/7}$ | Fixed Silverman: $h_0 = (4/(n(d+2)))^{1/(d+4)}$ | **MAJOR GAP** |
| **Higher-order functional** | $\hat{T}_D(b)$ for $R(f''')$ via $\phi^{(6)}$ | Not computed | **MISSING** |
| **Solve-the-equation** | Yes (Newton-Raphson) | No (one-shot plug-in) | **MISSING** |
| **Rate** | $O(n^{-5/14})$ ≈ $O(n^{-0.357})$ | Unknown (probably $O(n^{-2/(d+4)})$) | Theory gap |

---

## What's ACTUALLY Needed for a True d-D SJ

### Step 1: Derive the $\Psi_6$ estimator polynomial (analog of $\phi^{(6)}$)

In 1D:
- $\hat{S}_D(\alpha)$ uses $\phi^{(4)}(z)$ = kernel with 4th derivative → estimates $R(f'') = \int(f'')^2$
- $\hat{T}_D(b)$ uses $\phi^{(6)}(z)$ = kernel with 6th derivative → estimates $R(f''') = \int(f''')^2$

In d-D:
- $\hat{\Psi}_4(h_0)$ uses $P_d(t)$ → estimates $\Psi_4 = \int(\nabla^2 f)^2 dx$ ← DONE
- $\hat{\Psi}_6(b)$ uses $Q_d(t)$ → estimates $\Psi_6 = \int(\Delta^2 f)^2 dx$ or similar ← NEEDED

The polynomial $Q_d(t)$ arises from computing:
$$\int (\nabla^2)^2 K_b(\mathbf{x} - \mathbf{Y}_i) \cdot (\nabla^2)^2 K_b(\mathbf{x} - \mathbf{Y}_j) \, d\mathbf{x}$$

This is the same kind of Gaussian product integral we already did for $P_d$, just with the bi-Laplacian instead of the Laplacian. The calculation structure is identical — compute $(\nabla^2)^2 K_h(\mathbf{t}) = h^{-4} K_h(\mathbf{t}) \cdot Q_d(\|\mathbf{t}\|^2/h^2)$ where $Q_d$ is the bi-Laplacian analog of the Laplacian polynomial, then integrate the product.

### Step 2: Normal reference for $\Psi_6$

For $f = N(0, \sigma^2 I_d)$:
$$\Psi_6^{NR}(\sigma) = \frac{C_d}{\sigma^{d+8}}$$

where $C_d$ is a closed-form constant. This gives the Stage 3 bandwidths $a$ and $b$.

### Step 3: Derive the adaptive pilot $\alpha_2(h)$ in d-D

By the bias cancellation argument:
$$\alpha_2(h) \propto \left(\frac{\hat{\Psi}_4}{\hat{\Psi}_6}\right)^{1/(d+6)} \cdot h^{(d+4)/(d+6)}$$

(The exponent $5/7$ in 1D becomes $(d+4)/(d+6)$ — at $d=1$: $5/7$ ✓)

### Step 4: STE wrapper

```python
from scipy.optimize import brentq

def sj_bandwidth_ste(X, ...):
    Y = whiten(X)
    # Stage 3: normal reference bandwidths a, b
    # Stage 2: compute Ψ̂₄(a), Ψ̂₆(b) using P_d and Q_d
    # Stage 1: adaptive pilot α₂(h) = C * (Ψ̂₄/Ψ̂₆)^{1/(d+6)} * h^{(d+4)/(d+6)}
    # Stage 0: solve h = h_AMISE(Ψ̂₄(α₂(h)))
    def ste_eq(h):
        alpha = adaptive_pilot(h, psi4_est, psi6_est)
        psi4_at_alpha = compute_psi4(Y, alpha)
        h_opt = (d * (4*pi)**(-d/2) / (n * psi4_at_alpha)) ** (1/(d+4))
        return h_opt - h
    return brentq(ste_eq, h_low, h_high)
```

---

## The Bottom Line

**Our paper's honest claim**: "We derived the first closed-form expression for multivariate Gaussian KDE roughness ($P_d$ polynomial), enabling non-iterative roughness estimation in any dimension. Used with a Silverman pilot, this gives a practical one-stage plug-in selector."

**What a TRUE generalization would add**: The full two-stage STE with adaptive pilot — requiring the $Q_d$ polynomial for $\Psi_6$ estimation. This would give:
- Better convergence rate in $d$-D (analog of the $O(n^{-5/14})$ improvement)
- Adaptive pilot that doesn't assume normality
- The complete SJ algorithm in any dimension

**The gap is well-defined and achievable**: It's literally the SAME calculation we already did for $P_d$, but for the bi-Laplacian kernel instead of the Laplacian kernel. The structure is:

1. Compute $(\nabla^2)^2 K_h(\mathbf{t})$ in closed form → get $Q_d$ polynomial
2. Integrate the product of two such terms → get the pairwise formula
3. Verify at $d=1$: should recover $\phi^{(6)}(z) = (z^6 - 15z^4 + 45z^2 - 15)\phi(z)$

This is a focused calculation — probably 2-3 pages of algebra, same techniques we already used.

---

## Summary Table: What the Original Paper's Message Is

| Paper Says | Translation |
|-----------|-------------|
| "Reliable" | The STE function is always positive → Newton-Raphson never fails |
| "Improves Park-Marron" | Better rate ($n^{-5/14}$ vs $n^{-4/13}$) + better constant (0.23 vs 0.50) |
| "Bias cancellation" | Use bandwidth $\alpha$ to cancel the diagonal bias → net bias drops |
| "Three stages" | Normal reference → kernel functional estimates → STE |
| "Simulation evidence" | Works well on normal, mixtures; consistently good across shapes |

The paper's main selling point isn't the math — it's the **reliability**. SJ works on every dataset without parameter tuning, the equation always has a unique solution, and the method never produces absurd results. That's what made it the default in R/scipy for 35 years.
