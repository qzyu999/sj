# What Exactly Did We Generalize? Naming and Positioning

## The Problem

The GPT-5 review correctly identifies that calling our method "Generalized Sheather-Jones" implies we replicated the *full* SJ algorithm in $d$ dimensions. We didn't. We generalized one component — the roughness computation — and wrapped it in a simpler algorithm than what SJ actually proposed.

So what IS our method, precisely? And what should we call it?

---

## Anatomy of the Original Sheather-Jones (1991) Paper

The SJ paper proposes a bandwidth selector. It has multiple parts:

### Part 1: The AMISE formula (standard theory, not unique to SJ)

$$h^* = \left(\frac{R(K)}{n \sigma_K^4 \psi_4}\right)^{1/5}$$

where $\psi_4 = R(f'') = \int [f''(x)]^2 dx$.

This is textbook material (Silverman 1986, Wand & Jones 1995). SJ did NOT invent this.

### Part 2: The roughness estimation strategy (the actual SJ contribution)

SJ's key insight: to estimate $\psi_4$ well, you need a pilot bandwidth $g_2$. But $g_2$ itself depends on $\psi_6$ (the 6th derivative roughness). SJ proposed:

1. **Stage 0**: Estimate $\psi_6$ using a **normal reference** (no data-driven pilot needed for this):
   $$\hat{\psi}_6^{NR} = \frac{-15}{16\sqrt{\pi}} \hat{\sigma}^{-7}$$

2. **Stage 1**: Derive a pilot $g_2$ from $\hat{\psi}_6$:
   $$g_2 = \left(\frac{-6}{\sqrt{2\pi} \hat{\psi}_6 n}\right)^{1/7}$$

3. **Stage 2**: Estimate $\psi_4$ using pilot $g_2$:
   $$\hat{\psi}_4(g_2) = \frac{1}{n^2} \sum_{i,j} K^{(4)}_{g_2}(X_i - X_j)$$

4. **Stage 3**: Plug into AMISE formula:
   $$\hat{h}_{DPI} = \left(\frac{R(K)}{n \hat{\psi}_4}\right)^{1/5}$$

The multi-stage pilot (using ψ₆ to derive ψ₄'s pilot) is what gives SJ its superior convergence rate: $O(n^{-4/13})$ vs $O(n^{-5/14})$ for one-stage methods.

### Part 3: The STE (Solve-the-Equation) variant

SJ also proposed solving the self-consistent equation:
$$h = \left(\frac{R(K)}{n \hat{\psi}_4(h)}\right)^{1/5}$$

where the pilot IS the solution itself. This removes pilot dependence entirely but requires iterative root-finding.

### Part 4: The Gaussian kernel specialization

For the Gaussian kernel, $K^{(4)}$ has a closed form involving Hermite polynomials. This makes the roughness sum tractable. SJ noted this but the closed-form evaluation of pairwise Gaussian integrals predates them (it's standard Gaussian calculus).

---

## What WE Actually Did

### We generalized Part 4 (the closed-form evaluation) to $d$ dimensions.

Specifically:
- We showed that the pairwise integral $\int \nabla^2 K_h(x-X_i) \cdot \nabla^2 K_h(x-X_j) \, dx$ has a closed form in any $d$
- We derived the polynomial $P_d(t)$ that governs it
- We used a **one-stage** plug-in with Silverman pilot (simpler than SJ's two-stage approach)

### What we did NOT generalize:
- The two-stage pilot chain (ψ₆ → g₂ → ψ₄) — we use a simpler Silverman pilot
- The STE variant — we don't iterate
- The general-kernel theory — we assume Gaussian throughout

---

## So What Should We Call It?

### Options:

| Name | Pros | Cons |
|------|------|------|
| "Generalized Sheather-Jones" (GSJ) | Recognizable, connects to known method | Over-claims; implies full algorithm generalization |
| "Multivariate Plug-In Bandwidth" | Accurate, generic | Too bland; doesn't indicate what's new |
| "Closed-Form Laplacian Roughness" | Describes what we actually compute | Doesn't mention bandwidth or KDE |
| "Isotropic Plug-In Selector" (IPS) | Accurate | Bland |
| "Direct Plug-In with Closed-Form Roughness" (DPI-CFR) | Precise | Ugly |
| "Laplacian Plug-In Bandwidth" (LPB) | Describes the functional + the approach | Not immediately recognizable |

### My recommendation: keep "gsj" as the package name but reframe the paper

The package name `gsj` is already published on PyPI and has recognition value. But the paper's title and framing should be more precise:

**Paper title**: "Closed-Form Plug-In Bandwidth Selection for Multivariate Kernel Density Estimation"

**In the abstract**: "...extending the roughness-estimation principle of Sheather & Jones (1991) to $d$ dimensions via a novel closed-form expression..."

**Package name**: `gsj` — fine to keep. Think of it like how "Adam" (the optimizer) has a catchy name that doesn't precisely describe the algorithm. The `g` can stand for "generalized" in the informal sense.

---

## The Precise Relationship (Concrete Proof)

### What the original SJ paper contributes (vs us):

```
                ORIGINAL SJ (1991)              OUR METHOD
                ─────────────────               ──────────
AMISE formula:  standard (not SJ's)             same
                
Roughness:      ψ₄ via pairwise                ψ₄ via pairwise
                K^(4) sum (1D)                  ∇²K product (d-D)
                Gaussian → closed form          Gaussian → closed form P_d(t)
                
Pilot:          TWO-STAGE                       ONE-STAGE
                ψ₆^NR → g₂ → ψ₄               Silverman → ψ₄
                
Iteration:      STE variant available           None (one-shot)
                
Rate:           O(n^{-4/13}) [2-stage DPI]     O(n^{-5/14}) [conjectured]
                O(n^{-5/14}) [1-stage DPI]
                
Kernel:         General K (theory)              Gaussian only
                Gaussian (practice)
```

### The overlap:
- Same functional being estimated (ψ₄ = integrated squared second derivative / Laplacian)
- Same plug-in principle (estimate ψ₄, plug into AMISE formula)
- Same closed-form Gaussian kernel evaluation technique

### The difference:
- Pilot selection strategy (their main theoretical contribution) is NOT replicated
- We use a simpler, weaker (but practical) pilot

---

## What Does locfit::sjpi Actually Do?

Since our method matches `locfit::sjpi` numerically (to binning precision), let's be explicit:

```r
sjpi <- function(x, a) {
  # x: data
  # a: user-provided pilot bandwidth (single value)
  
  # 1. Bin data (for efficiency)
  # 2. Compute ψ₄ using the Gaussian 4th derivative kernel with bandwidth 'a'
  # 3. Return h = (R(K)/(n * ψ₄))^(1/5)
}
```

That's it. `locfit::sjpi` is a **one-stage plug-in** with user-provided pilot. It does NOT do the two-stage ψ₆ chain. It does NOT do STE iteration.

`locfit::sjpi` is called "SJ" because it computes the same roughness functional that SJ proposed — but it uses a simpler pilot strategy.

**Our method is the d-dimensional generalization of `locfit::sjpi`.** That is a precise, defensible, concrete statement.

---

## The Fair Criticism and Fair Response

### The criticism (valid):
> "Calling this 'Generalized Sheather-Jones' implies you replicated the full 1991 algorithm (two-stage pilot, STE, convergence rate proof) in d dimensions. You didn't."

### The response (also valid):
> "What practitioners call 'SJ bandwidth' in software (locfit, scipy, statsmodels) is typically the one-stage DPI — not the two-stage algorithm from the 1991 paper. We generalize what people actually use under the name 'SJ'. The name has drifted from the paper."

### The resolution:
In the paper, be explicit:
> "We generalize the direct plug-in roughness computation — the core calculation underlying practical SJ implementations such as R's `locfit::sjpi` — to $d$ dimensions. We do not replicate the multi-stage pilot hierarchy or solve-the-equation construction of the full Sheather & Jones (1991) algorithm, which represents a separate (and potentially complementary) contribution."

---

## Suggested Revised Naming Convention

| Context | Name to use |
|---------|-------------|
| PyPI package | `gsj` (already published, keep it) |
| Paper title | "Closed-Form Plug-In Bandwidth Selection for Multivariate KDE" |
| In-text reference | "the multivariate plug-in selector" or "our d-D roughness estimator" |
| Comparing to SJ | "SJ-inspired" or "in the spirit of Sheather-Jones" |
| The specific claim | "generalizes the roughness calculation of locfit::sjpi to d dimensions" |
| What NOT to say | "the multivariate Sheather-Jones method" |

---

## Does This Diminish the Contribution?

**No.** The contribution is real and stands on its own:

1. **The polynomial $P_d(t)$ is new** — nobody has published this closed-form expression for the isotropic multivariate Laplacian roughness.

2. **The fact that it's SIMPLER in d-D than in 1D** (3 terms vs 16) is a surprising and publishable result.

3. **A practical, non-iterative scalar bandwidth selector for d > 1** fills a real gap in the toolbox — between crude rules (Scott/Silverman) and expensive matrix methods (Duong-Hazelton).

4. **The software package** makes it immediately usable.

The contribution doesn't need to be "we solved the full multivariate SJ problem." It can be "we found a remarkably simple closed-form roughness expression that enables practical plug-in bandwidth selection in any dimension." That's a cleaner, more honest, and equally publishable story.
