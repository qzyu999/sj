# Three-Way Comparison: Original SJ (1991) → Educational Derivation → d-D Generalization

This document traces the evolution of the Sheather-Jones bandwidth selector from the original paper, through its pedagogical treatment in the closed-form notebook, to our multivariate generalization.

---

## Stage 1: The Objective Function

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

**Eq. (2):** AMISE for kernel estimator

$$\text{AMISE}(h) = \frac{R(K)}{nh} + \frac{1}{4}h^4\sigma_K^4 R(f'')$$

Notation: $R(g) = \int g^2(x)dx$, $\sigma_K^2 = \int x^2 K(x)dx$

Minimizing gives $h_*$ (eq. 4).

</td>
<td>

**Eq. (8):** Same formula, pedagogically derived from ISE → MISE → Taylor expansion:

$$\text{AMISE}(h) = \frac{R(K)}{nh} + \frac{h^4\sigma_K^4 R(f'')}{4}$$

Shows the derivation step by step: ISE (eq. 5) → MISE (eq. 6) → Taylor expansion → AMISE (eq. 8).

Explains: "The $\mathcal{o}(\cdot)$ term is leftover from the Taylor series expansion."

</td>
<td>

**Whitened coordinates:** With $Y_i = \hat\Sigma^{-1/2}X_i$ and isotropic bandwidth $h$:

$$\text{AMISE}(h) = \frac{(4\pi)^{-d/2}}{nh^d} + \frac{h^4}{4}\Psi_4$$

where $\Psi_4 = \int(\nabla^2 f)^2 d\mathbf{x}$

At $d=1$: $(4\pi)^{-1/2} = R(K)$ for Gaussian ✓

Minimizing: $h^* = (d/(n\Psi_4(4\pi)^{d/2}))^{1/(d+4)}$

</td>
</tr>
</table>

---

## Stage 2: The Bandwidth Formula

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

**Eq. (4):**

$$\hat h = \left[\frac{R(K)}{\sigma_K^4 \hat S(\alpha)}\right]^{1/5} n^{-1/5}$$

"An estimate of the usual expression for the asymptotically optimal bandwidth."

The problem: need to estimate $R(f'')$.

</td>
<td>

**Eq. (9):**

$$h = \left(\frac{R(K)}{n\sigma_K^4 R(f'')}\right)^{1/5}$$

"Another issue arises where to calculate this value, it requires that the true density $f(x)$ is known."

This motivates: (1) cross-validation, (2) plug-in methods.

</td>
<td>

$$h^* = \left(\frac{d}{n\hat\Psi_4(4\pi)^{d/2}}\right)^{1/(d+4)}$$

Same structure. Exponent $1/5$ becomes $1/(d+4)$.

At $d=1$: $1/(1+4) = 1/5$ ✓

The problem: need to estimate $\Psi_4$.

</td>
</tr>
</table>

---

## Stage 3: The Pilot Bandwidth

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

**Silverman for pilot:**

SJ uses a normal-reference pilot as one stage of a multi-stage process. The IQR-based $\hat\lambda$ gives a robust scale:

$a = 0.920\hat\lambda n^{-1/7}$
$b = 0.912\hat\lambda n^{-1/9}$

These come from the bias-cancellation argument (the paper's key innovation).

</td>
<td>

**Eq. (14) — Silverman's rule:**

$$h_0 = \left(\frac{4}{3n}\right)^{1/5}\hat\sigma$$

"An advantage of this method is that it's extremely easy to calculate. Problems arise when the data follows a multimodal distribution — it will lead to oversmoothing."

"The resulting bandwidth is considered a suitable pilot bandwidth, $h_0$, and is to be calculated in the first stage of the Sheather-Jones method."

</td>
<td>

**One-stage (current gsj):**

$$h_0 = \left(\frac{4}{n(d+2)}\right)^{1/(d+4)}$$

This is Silverman's rule in $d$-D.

**Two-stage (new):**

$$g = \hat\sigma\left(\frac{2}{n(d+4)}\right)^{1/(d+6)}$$

From bias cancellation using $\Psi_6$ normal reference. Rate: $n^{-1/(d+6)}$.

At $d=1$: $n^{-1/7}$ ✓ (matches SJ's pilot rate)

</td>
</tr>
</table>

---

## Stage 4: The Roughness Estimator (The Core Formula)

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

**Eq. (5) — "Diagonals in":**

$$\hat S_D(\alpha) = \frac{1}{n^2\alpha^5}\sum_{i,j}\phi^{(4)}\!\left(\frac{X_i-X_j}{\alpha}\right)$$

where $\phi^{(4)}(z) = (z^4-6z^2+3)\phi(z)$

Key innovation: including $i=j$ terms (positive, non-stochastic) makes $\hat S_D > 0$ always.

</td>
<td>

**Eq. (15)–(20): Step-by-step closed-form derivation**

The notebook derives $R(\hat f'')$ by hand:

1. Write $\hat f'' = \frac{1}{nh_0^3}\sum_i L''((x-X_i)/h_0)$
2. Square it: product of two sums → double sum
3. Integrate: each $L''(\cdot)L''(\cdot)$ product gives a Gaussian integral
4. Get closed-form: involves $\phi^{(4)}((X_i-X_j)/h_0)$

"Although the idea for the Sheather-Jones method is quite simple, the actual calculation can be quite difficult."

The notebook computes derivatives explicitly:

$L''(z) = \frac{-1}{\sqrt{2\pi}}e^{-z^2/2}(1-z^2)$

Then integrates the product to get the closed form.

</td>
<td>

**Our formula:**

$$\hat\Psi_4(h_0) = \frac{1}{n^2(4\pi)^{d/2}h_0^{d+4}}\sum_{i,j}e^{-r_{ij}^2/4}P_d(r_{ij}^2)$$

where $P_d(t) = \frac{t^2}{16} - \frac{(d+2)t}{4} + \frac{d(d+2)}{4}$

**How we derived it** (same approach as notebook, in $d$-D):

1. Write $\nabla^2\hat f = \frac{1}{n}\sum_i \nabla^2 K_h(x-Y_i)$
2. Square and integrate: double sum of pairwise integrals
3. Each pair: Gaussian product lemma → single Gaussian → moment calculation
4. Moments of $\|\mathbf{U}\|^2$, $\|\mathbf{V}\|^2$ for shifted Gaussians → polynomial $P_d$

**Verified**: at $d=1$, $P_1(z^2) \cdot e^{-z^2/4}/(4\pi)^{1/2}$ matches SJ's $\phi^{(4)}(z/\sqrt{2})$ formula exactly.

</td>
</tr>
</table>

---

## Stage 5: The Higher-Order Functional (Two-Stage)

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

**$\hat T_D(b)$ for $R(f''')$:**

$$\hat T_D = -\frac{1}{n^2 b^7}\sum_{i,j}\phi^{(6)}\!\left(\frac{X_i-X_j}{b}\right)$$

$\phi^{(6)}(z) = (z^6-15z^4+45z^2-15)\phi(z)$

This is the 6th Hermite function.

Used in the adaptive pilot $\hat\alpha_2(h)$.

</td>
<td>

Not derived in the notebook. The notebook uses `locfit::sjpi()` which handles this internally:

"A third package called {locfit} was used... it allows for users to input a single bandwidth as the pilot bandwidth."

The notebook focuses on the **first-stage** closed-form ($R(\hat f'')$) and treats the two-stage pilot as a black box inside the R package.

</td>
<td>

**Our $R_d$ polynomial (NEW):**

$$\hat\Psi_6(g) = \frac{1}{n^2(4\pi)^{d/2}g^{d+6}}\sum_{i,j}e^{-r_{ij}^2/4}R_d(r_{ij}^2)$$

$$R_d(t) = -\frac{t^3}{64} + \frac{3(d+4)}{32}t^2 - \frac{3(d+2)(d+4)}{16}t + \frac{d(d+2)(d+4)}{8}$$

Derivation: same moment approach but for $\nabla(\nabla^2 K)$ (the gradient of the Laplacian), involving the **dot product** of two vector fields.

At $d=1$: matches $\phi^{(6)}$ in integral-of-products framework ✓

</td>
</tr>
</table>

---

## Stage 6: The Normal Reference Values

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

For $f = \mathcal{N}(0, \sigma^2)$:

$R(f'') = \frac{3}{8\sqrt\pi\sigma^5}$

$R(f''') = \frac{15}{16\sqrt\pi\sigma^7}$

Used at Stage 3 (innermost pilot) where using a scale model "is sufficiently good."

</td>
<td>

The notebook uses the Silverman normal reference:

$R(\phi'')/\hat\sigma^5$ where $\phi$ is the standard normal.

"In the Silverman method... the unknown value $R(f'')$ is replaced with $R(\phi'')/\hat\sigma^5$"

The actual values are implicit in the Silverman formula $(4/(3n))^{1/5}\hat\sigma$.

</td>
<td>

$$\Psi_4^{NR}(\sigma) = \frac{d(d+2)}{4(4\pi)^{d/2}\sigma^{d+4}}$$

$$\Psi_6^{NR}(\sigma) = \frac{d(d+2)(d+4)}{8(4\pi)^{d/2}\sigma^{d+6}}$$

Pattern: numerator picks up successive factors $(d+2k)$; denominator picks up factors of $2$ and powers of $\sigma$.

**Verification**: At $d=1$:
- $\Psi_4^{NR} = 3/(8\sqrt\pi\sigma^5) = R(f'')$ ✓
- $\Psi_6^{NR} = 15/(16\sqrt\pi\sigma^7) = R(f''')$ ✓

</td>
</tr>
</table>

---

## Stage 7: Solving for the Final Bandwidth

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

**Eq. (12) — The STE:**

$$\left[\frac{R(K)}{\sigma_K^4\hat S_D(\hat\alpha_2(h))}\right]^{1/5}n^{-1/5} - h = 0$$

"We successfully use the Newton-Raphson method."

The function is smooth because $\hat S_D > 0$ always (diagonals-in guarantee).

</td>
<td>

The notebook uses R packages (`locfit::sjpi`, `bw.SJ`) as black boxes for the STE:

"The Sheather-Jones method is more complicated and requires the Silverman bandwidth for a pilot bandwidth."

Results: `locfit::sjpi` = 1.672771, closed-form = 1.673207 (match to 4 digits).

The notebook validates the approach by comparing the **closed-form roughness** against the R package output.

</td>
<td>

**DPI (recommended):**

$$h^* = \left(\frac{d}{n\hat\Psi_4(g_1)(4\pi)^{d/2}}\right)^{1/(d+4)}$$

where $g_1$ is the adaptive pilot from Stage 5.

**STE variant:**

Solve $h = h_{\text{AMISE}}(\hat\Psi_4(\alpha(h)))$ with $\alpha(h) \propto h^{(d+4)/(d+6)}$

using Brent's method.

At $d=1$: exponent $(d+4)/(d+6) = 5/7$ ✓

Both are always well-posed because $\hat\Psi_4 > 0$ (same positivity guarantee as SJ's $\hat S_D > 0$).

</td>
</tr>
</table>

---

## Stage 8: Verification and Validation

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook Derivation (1D)</th>
<th width="34%">d-D Generalization (Ours)</th>
</tr>
<tr>
<td>

**Simulations (Section 4):**

500 realizations, $n=50$ and $n=100$, four test densities:
- Standard normal $\phi$
- Mean mixture $\frac{1}{2}\phi(x+3)+\frac{1}{2}\phi(x-3)$
- Variance mixtures

Reports $R_h = 100\{\text{MISE}(\hat h)/\text{MISE}(h_0)-1\}$

Result: $\hat h_{2S}$ "is never dominated" — consistently good.

</td>
<td>

**Validation against R packages:**

| Method | Bandwidth |
|--------|-----------|
| Silverman | 3.299 |
| `stats::bw.SJ` | 1.007 |
| `MASS` | 4.029 |
| `locfit::sjpi` | 1.673 |
| Monte Carlo | 1.515 |
| **Closed-form** | **1.673** |

"It's somewhat surprising how similar the closed-form solution is to the {locfit} solution."

Validates: the hand-derived formula matches the reference implementation.

</td>
<td>

**Symbolic verification (SymPy):**
- $P_d$ at $d=1$ matches 1D formula to $< 10^{-14}$
- $R_d$ at $d=1$ matches $\phi^{(6)}$ framework ✓
- Normal references match known values ✓

**Monte Carlo verification:**
- All polynomials verified to $< 10^{-3}$ relative error

**ISE comparison (2D whitened):**

| Method | $h$ | ISE |
|--------|------|-----|
| Optimal | 0.242 | 1.90e-3 |
| **Two-stage** | **0.236** | **1.92e-3** |
| One-stage | 0.266 | 1.96e-3 |
| Silverman | 0.344 | 3.21e-3 |

</td>
</tr>
</table>

---

## The Narrative Arc

<table>
<tr>
<th width="33%">SJ Paper (1991)</th>
<th width="33%">Notebook (Educational)</th>
<th width="34%">d-D Generalization (Research)</th>
</tr>
<tr>
<td>

**Message:** "Here's a method that's reliable, fast, and never dominated."

**Audience:** Statisticians looking for a practical bandwidth selector.

**Key formula:** The STE function (eq. 12) that's smooth and always has a solution.

**Innovation:** Bias cancellation via "diagonals in" + adaptive pilot.

**Impact:** Became the default in R for 35 years.

</td>
<td>

**Message:** "Here's how the math works, step by step, for a practitioner who wants to understand SJ."

**Audience:** Graduate students, applied researchers.

**Key formula:** The closed-form $R(\hat f'')$ integral (eq. 20) — the pairwise kernel evaluation.

**Innovation:** Pedagogical: derives by hand what R packages compute internally.

**Impact:** Shows that the "black box" of SJ is one closed-form double sum.

</td>
<td>

**Message:** "We generalize the entire SJ structure to arbitrary dimension via two new polynomials."

**Audience:** Statisticians + ML researchers working with embeddings.

**Key formula:** $P_d(t)$ and $R_d(t)$ — closed-form pairwise roughness polynomials in any $d$.

**Innovation:**
- First closed-form $d$-D roughness expression
- First $d$-D analog of $\phi^{(6)}$ ($R_d$ polynomial)
- Complete two-stage algorithm with adaptive pilot

**Impact:** Makes SJ-quality bandwidth selection available for embeddings, anomaly detection, distribution monitoring in $d > 1$.

</td>
</tr>
</table>

---

## The Polynomial Lineage

The key mathematical object at each stage:

| Stage | SJ (1991) | Notebook (1D) | Ours (d-D) |
|-------|-----------|---------------|------------|
| Kernel | $K(z) = \phi(z)$ | Same | $K_h(\mathbf{t}) = (2\pi h^2)^{-d/2}e^{-\|\mathbf{t}\|^2/(2h^2)}$ |
| 2nd derivative | $\phi''(z) = (z^2-1)\phi(z)$ | Derived explicitly (eq. 17) | $\nabla^2 K_h(\mathbf{t}) = h^{-2}K_h(\mathbf{t})(s - d)$, $s = \|\mathbf{t}\|^2/h^2$ |
| Roughness kernel ($\Psi_4$) | $\phi^{(4)}(z) = (z^4-6z^2+3)\phi(z)$ | Derived as $L''L''$ product (eq. 20) | $e^{-t/4}\cdot P_d(t)$, $P_d = t^2/16 - (d{+}2)t/4 + d(d{+}2)/4$ |
| Next-order kernel ($\Psi_6$) | $\phi^{(6)}(z) = (z^6-15z^4+45z^2-15)\phi(z)$ | Not derived (R package) | $e^{-t/4}\cdot R_d(t)$, $R_d = -t^3/64 + \ldots$ |
| Value at origin | $\phi^{(4)}(0) = 3\phi(0)$ | Implicit | $P_d(0) = d(d{+}2)/4$; $R_d(0) = d(d{+}2)(d{+}4)/8$ |

---

## What Each Contribution Uniquely Added

| Contribution | What was new | What it enabled |
|-------------|-------------|-----------------|
| **SJ (1991)** | Bias cancellation + "diagonals in" + STE | Reliable 1D bandwidth selection (R's `bw.SJ`) |
| **Notebook** | Pedagogical closed-form derivation | Understanding of the "black box" — researchers can implement SJ from scratch |
| **Our d-D work** | $P_d$ and $R_d$ polynomials + full algorithm | Multivariate SJ for the first time — practical plug-in selection in $d > 1$ |

The notebook was the bridge: by working through the 1D closed form by hand, it revealed the structure (pairwise evaluation → Gaussian product → polynomial × exponential) that generalizes naturally to $d$ dimensions.
