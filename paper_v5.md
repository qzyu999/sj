# Where Does Sheather-Jones Sit in KDE Theory? Optimality, Approximations, and Limits

## The Question

Is the Sheather-Jones bandwidth the "theoretically optimal" bandwidth according to KDE theory?

**Short answer**: SJ is the best practical estimator of the theoretically optimal bandwidth — but it's not the optimal itself. It's the best *feasible attempt* to reach an unreachable target.

---

## 1. The Hierarchy of Approximations

The path from "what we actually want" to "what SJ gives us" involves four levels of approximation:

```
Level 0:  TRUE OPTIMAL BANDWIDTH
          h* = argmin_h MISE(h) = argmin_h E[∫(f̂_h - f)² dx]
          ↓
          Problem: requires knowing f (the true density we're estimating)
          This is a chicken-and-egg problem — unknowable.

Level 1:  AMISE-OPTIMAL BANDWIDTH
          h*_AMISE = (d · R(K) / (n · Ψ))^{1/(d+4)}
          where Ψ = ∫(∇²f)² dx
          ↓
          Approximation: MISE ≈ AMISE + higher-order terms (Taylor expansion)
          Still unknowable: Ψ depends on f.

Level 2:  PLUG-IN ESTIMATE (Sheather-Jones)
          ĥ_SJ = (d · R(K) / (n · Ψ̂(h₀)))^{1/(d+4)}
          where Ψ̂ is estimated from data using pilot h₀
          ↓
          Approximation: Ψ̂ ≈ Ψ (roughness estimated, not known)
          Pilot-dependent: result varies with h₀ choice.

Level 3:  THE NUMBER YOU GET
          A specific bandwidth value for your specific dataset.
```

Each level introduces error. The question is: how much?

---

## 2. What Is "Theoretically Optimal" in KDE?

### The MISE criterion

The gold standard for KDE quality is the **Mean Integrated Squared Error**:

$$\text{MISE}(h) = \mathbb{E}\left[\int (\hat{f}_h(\mathbf{x}) - f(\mathbf{x}))^2 \, d\mathbf{x}\right]$$

This is the expected total squared error, averaged over all possible random samples. The bandwidth that minimizes MISE is the **true optimal bandwidth** $h^*$.

### The AMISE approximation

For the Gaussian kernel with scalar bandwidth $h$ in $d$ dimensions, a Taylor expansion gives:

$$\text{MISE}(h) = \underbrace{\frac{R(K)}{nh^d}}_{\text{variance}} + \underbrace{\frac{h^4}{4}\Psi}_{\text{bias}^2} + O\left(\frac{1}{nh^{d+2}} + h^6\right)$$

The AMISE is the first two terms. Dropping the remainder and minimizing:

$$h^*_{\text{AMISE}} = \left(\frac{d \cdot R(K)}{n \cdot \Psi}\right)^{1/(d+4)}$$

This is "theoretically optimal" **in the asymptotic sense** — it's the right answer as $n \to \infty$. For finite $n$, it's an approximation.

### When does AMISE fail?

The AMISE approximation breaks down when:
- $n$ is small (the $O(\cdot)$ remainder terms matter)
- $h$ is very small or very large (outside the regime where Taylor expansion is valid)
- The density has discontinuities or very sharp features (Taylor expansion of bias requires smoothness)

For typical smooth densities and $n > 50$, AMISE is an excellent approximation.

---

## 3. What SJ Actually Does

SJ estimates the unknown quantity $\Psi = \int (\nabla^2 f)^2 \, dx$ from the data:

$$\hat{\Psi}(h_0) = \int [\nabla^2 \hat{f}_{h_0}(\mathbf{x})]^2 \, d\mathbf{x}$$

This is a "plug-in" approach: plug a data-based estimate of $\Psi$ into the AMISE formula.

### The pilot problem

$\hat{\Psi}$ itself depends on a bandwidth ($h_0$). How to choose $h_0$?

- **Silverman's rule** (what we use): simple, $O(1)$ computation, reasonable for most data
- **Multi-stage pilot** (what R's `bw.SJ(method="dpi")` does): use a normal-reference $\Psi_6$ → pilot for $\Psi_4$ → final bandwidth. Reduces pilot sensitivity.
- **Solve-the-equation** (what R's `bw.SJ(method="ste")` does): find $h$ where $h = g(h)$. Self-consistent but iterative.

The result is pilot-dependent, but **weakly so** — because of the $(d+4)$-th root:

$$\hat{h}_{\text{SJ}} = \left(\frac{c}{\hat{\Psi}(h_0)}\right)^{1/(d+4)}$$

A 20% error in $\hat{\Psi}$ produces only a ~4% error in $\hat{h}$ (for $d=1$). The root dampens the pilot sensitivity enormously.

---

## 4. Convergence Rates: Why SJ Is "Best"

### The question: how fast does $\hat{h} \to h^*$ as $n \to \infty$?

Different methods converge at different rates:

| Method | Rate $|\hat{h} - h^*_{\text{AMISE}}| / h^*$ | Source |
|--------|----------------------------------------------|--------|
| Scott/Silverman | $O(1)$ — never converges (unless $f$ is Gaussian) | By construction |
| LSCV (least-squares CV) | $O(n^{-1/10})$ in 1D | Hall & Marron (1987) |
| LSCV (likelihood CV) | $O(n^{-1/10})$ in 1D | Hall (1987) |
| **SJ (plug-in, one-stage)** | $O(n^{-5/14})$ in 1D | Sheather & Jones (1991) |
| **SJ (plug-in, two-stage)** | $O(n^{-4/13})$ in 1D | Hall, Sheather & Jones (1991) |
| Botev/ISJ (diffusion) | $O(n^{-4/13})$ in 1D | Botev, Grotowski & Kroese (2010) |

### What these rates mean

- **$O(1)$**: Scott/Silverman give the "wrong" answer for non-Gaussian $f$. The error doesn't shrink with more data. They're biased estimators of $h^*$.
- **$O(n^{-1/10})$**: CV methods converge, but slowly. With $n = 10{,}000$ you still have ~40% relative error in the bandwidth.
- **$O(n^{-5/14}) \approx O(n^{-0.36})$**: SJ converges much faster. With $n = 10{,}000$, relative error is ~5%.

SJ achieves the **minimax optimal rate** for plug-in bandwidth selectors — no other plug-in method can converge faster (proven by Hall & Marron 1987). In this sense, SJ is theoretically optimal *among practical methods*.

### Multivariate rates

For general $d$, the rates become:

- CV: $O(n^{-2/(d+6)})$ — degrades with dimension
- Plug-in (SJ-type): $O(n^{-2/(d+6)})$ for one-stage, potentially better with multi-stage

The exact multivariate rates for our specific $d$-D formula have not been formally derived (this would be a theoretical contribution for the paper).

---

## 5. What SJ Is NOT

### SJ is not the "true" optimal bandwidth

The true optimal bandwidth for a specific sample and true $f$ is:

$$h^*_{\text{true}} = \text{argmin}_h \int (\hat{f}_h(x) - f(x))^2 \, dx$$

This is computable only in simulation (where we know $f$). SJ approximates it via AMISE + plug-in estimation. The ISE-vs-$h$ plots in our Part 2 notebook show: SJ lands *near* but not exactly at the minimum.

### SJ is not "always best"

Specific pathological cases where SJ can be beaten:

1. **Exactly Gaussian data**: Silverman IS the AMISE-optimal bandwidth. SJ adds estimation noise for no benefit. (Seen in our benchmarks: SJ is slightly worse on Normal data.)

2. **Very small $n$ ($< 20$)**: The asymptotic theory SJ relies on breaks down. CV methods may be more robust here.

3. **Densities with sharp discontinuities**: AMISE theory assumes smooth densities (4+ derivatives). For discontinuous $f$ (e.g., uniform distribution), the bias expansion is different and SJ targets the wrong thing.

4. **Heavy tails**: The Gaussian kernel's exponential decay can't match power-law tails. No bandwidth choice fixes a kernel shape mismatch.

### SJ is not the global optimum over all possible methods

A "oracle" method that somehow knew $f$ would do better. And for specific density families, parametric methods (e.g., fitting a Gaussian mixture) will dominate KDE entirely. SJ is optimal within the class of **non-parametric scalar-bandwidth Gaussian-kernel density estimators with plug-in bandwidth selection** — which is a specific (though important) problem class.

---

## 6. The Analogy: Sample Mean

A useful analogy: SJ is to bandwidth selection what the **sample mean** is to location estimation.

| Property | Sample mean $\bar{X}$ | SJ bandwidth $\hat{h}$ |
|----------|------------------------|-------------------------|
| What it estimates | Population mean $\mu$ | AMISE-optimal bandwidth $h^*$ |
| Is it the true value? | No (random, varies by sample) | No (random, varies by sample) |
| Is it unbiased? | Yes | Approximately (asymptotically) |
| Convergence rate | $O(n^{-1/2})$ (CLT) | $O(n^{-5/14})$ (slower) |
| Can anything do better? | Not for squared error under Normality (MVUE) | Not among plug-in methods (minimax optimal) |
| When does it fail? | Heavy tails, contamination | Non-smooth densities, tiny $n$ |

Both are the "best you can do" within their problem class, while not being the unknowable truth.

---

## 7. Summary: The Precise Claim

What we can rigorously claim about SJ (d-D):

> **Theorem (informal)**: The Sheather-Jones plug-in bandwidth selector, when applied to a $d$-dimensional dataset with the Silverman pilot, produces a bandwidth estimate $\hat{h}$ that:
>
> 1. **Targets the AMISE-optimal bandwidth** — the leading-order-optimal bandwidth for the Gaussian kernel KDE
> 2. **Achieves the minimax rate** among plug-in selectors — no other single-stage plug-in converges faster
> 3. **Is consistent** — $\hat{h} \to h^*_{\text{AMISE}}$ as $n \to \infty$ for any smooth density $f$
> 4. **Dominates rule-of-thumb methods** — which are only optimal under the (usually false) Gaussian assumption
> 5. **Matches or exceeds cross-validation** in both convergence rate and practical performance, while being computationally cheaper

What we **cannot** claim:
- That SJ gives the "true optimal" bandwidth (unknowable)
- That SJ is always better than every alternative (it's not — see Gaussian data)
- That the AMISE-optimal bandwidth is the same as the MISE-optimal (they differ by $O(n^{-4/(d+4)})$ terms)
- That scalar bandwidth is sufficient (for highly anisotropic data, it isn't)

---

## 8. Where Our d-D Contribution Fits

The Sheather-Jones method was proven optimal (in the minimax sense) for 1D by Hall, Sheather & Jones (1991). Our contribution extends the **computational machinery** to $d$ dimensions via a closed-form roughness formula.

The key theoretical question we leave for future work:
> Does the multivariate one-stage plug-in (with Silverman pilot) achieve the same $O(n^{-5/14})$ convergence rate in $d$ dimensions?

We conjecture yes (the algebraic structure is identical), but a formal proof would require extending the Hall-Marron stochastic expansion framework to the multivariate setting. This would be a substantial theoretical contribution beyond the methods paper.

---

## References for Optimality Theory

1. P. Hall and J.S. Marron, "Estimation of Integrated Squared Density Derivatives," *Statistics & Probability Letters*, 6, 109-115, 1987. (Proves plug-in rates)
2. P. Hall, S.J. Sheather, M.C. Jones, and J.S. Marron, "On Optimal Data-Based Bandwidth Selection in Kernel Density Estimation," *Biometrika*, 78(2), 263-269, 1991. (Proves minimax optimality of two-stage plug-in)
3. S.J. Sheather and M.C. Jones, "A Reliable Data-Based Bandwidth Selection Method," *JRSS-B*, 53(3), 683-690, 1991. (The original SJ paper)
4. M.P. Wand and M.C. Jones, *Kernel Smoothing*, Chapman & Hall, 1995, Chapter 3. (Comprehensive treatment of AMISE theory)
5. Z.I. Botev, J.F. Grotowski, D.P. Kroese, "Kernel Density Estimation via Diffusion," *Annals of Statistics*, 38(5), 2916-2957, 2010. (Achieves same rate without pilot)
