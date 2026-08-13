# True Generalization of Sheather-Jones (1991) to $d$ Dimensions

## What the Original Paper Proposes (1D)

The SJ paper has three interlocking components that form a complete bandwidth selection algorithm:

1. **Functional estimation**: $\hat{\psi}_r(g) = \frac{1}{n^2 g^{r+1}} \sum_{i,j} K^{(r)}\left(\frac{X_i - X_j}{g}\right)$ for the Gaussian kernel
2. **Two-stage pilot**: use $\psi_6^{NR}$ (normal reference) → derive $g_2$ → estimate $\psi_4(g_2)$
3. **Solve-the-equation (STE)**: find $h$ satisfying $h = h_{\text{AMISE}}(\hat{\psi}_4(h))$

Below, we generalize ALL THREE to $d$ dimensions.

---

## Component 1: Functional Estimation in $d$-D (ALREADY DONE)

### 1D version (SJ equation 3):

$$\hat{\psi}_4(g) = \frac{1}{n^2 g^5} \sum_{i=1}^n \sum_{j=1}^n \phi^{(4)}\left(\frac{X_i - X_j}{g}\right)$$

where $\phi^{(4)}(z) = (z^4 - 6z^2 + 3)\phi(z)$ is the 4th Hermite function.

### $d$-D generalization (our result):

$$\hat{\Psi}_4(h_0) = \frac{1}{n^2 (4\pi)^{d/2} h_0^{d+4}} \sum_{i,j} e^{-r_{ij}^2/4} \cdot P_d(r_{ij}^2)$$

where $P_d(t) = \frac{t^2}{16} - \frac{(d+2)t}{4} + \frac{d(d+2)}{4}$ and $r_{ij}^2 = \|Y_i - Y_j\|^2/h_0^2$.

**Status**: ✅ COMPLETE (verified symbolically and numerically)

---

## Component 2: Normal Reference $\Psi_6$ in $d$-D (NEW DERIVATION)

### What $\Psi_6$ is

In 1D: $\psi_6 = \int [f'''(x)]^2 dx$ is the roughness of the 3rd derivative.

In $d$-D: $\Psi_6 = \int [\nabla^2(\nabla^2 f(\mathbf{x}))]^2 d\mathbf{x} = \int [(\nabla^2)^2 f]^2 d\mathbf{x}$

This is the roughness of the **bi-Laplacian** (4th order differential operator applied, then squared and integrated). Note: this is the functional relevant for the asymptotic bias of estimating $\Psi_4$.

### Normal reference: $f = \phi_d$ (standard $d$-dimensional Gaussian)

For $f(\mathbf{x}) = (2\pi)^{-d/2} e^{-\|\mathbf{x}\|^2/2}$, we need $(\nabla^2)^2 f$.

**Step 1**: $\nabla^2 f(\mathbf{x}) = f(\mathbf{x})(\|\mathbf{x}\|^2 - d)$

(This is our Lemma from the $\Psi_4$ derivation with $h=1$.)

**Step 2**: $(\nabla^2)^2 f = \nabla^2[\nabla^2 f] = \nabla^2[f(\|\mathbf{x}\|^2 - d)]$

Let $g(\mathbf{x}) = f(\mathbf{x})(\|\mathbf{x}\|^2 - d)$. Apply $\nabla^2$ to $g$:

$$\nabla^2 g = \nabla^2[f \cdot L_1]$$

where $L_1(\mathbf{x}) = \|\mathbf{x}\|^2 - d$.

Using the product rule for the Laplacian: $\nabla^2(uv) = u\nabla^2 v + v\nabla^2 u + 2\nabla u \cdot \nabla v$

- $u = f$, $v = L_1 = \|\mathbf{x}\|^2 - d$
- $\nabla^2 v = \nabla^2(\|\mathbf{x}\|^2 - d) = 2d$
- $\nabla^2 u = \nabla^2 f = f \cdot L_1$
- $\nabla u = \nabla f = -f \cdot \mathbf{x}$
- $\nabla v = 2\mathbf{x}$
- $2\nabla u \cdot \nabla v = 2(-f\mathbf{x}) \cdot (2\mathbf{x}) = -4f\|\mathbf{x}\|^2$

Therefore:
$$(\nabla^2)^2 f = f \cdot 2d + L_1 \cdot f \cdot L_1 + (-4f\|\mathbf{x}\|^2)$$
$$= f[2d + L_1^2 - 4\|\mathbf{x}\|^2]$$
$$= f[2d + (\|\mathbf{x}\|^2 - d)^2 - 4\|\mathbf{x}\|^2]$$
$$= f[\|\mathbf{x}\|^4 - 2d\|\mathbf{x}\|^2 + d^2 + 2d - 4\|\mathbf{x}\|^2]$$
$$= f[\|\mathbf{x}\|^4 - (2d+4)\|\mathbf{x}\|^2 + d^2 + 2d]$$
$$= f[\|\mathbf{x}\|^4 - 2(d+2)\|\mathbf{x}\|^2 + d(d+2)]$$

Define the **bi-Laplacian polynomial**:
$$Q_d(\|\mathbf{x}\|^2) = \|\mathbf{x}\|^4 - 2(d+2)\|\mathbf{x}\|^2 + d(d+2)$$

So: $(\nabla^2)^2 f = f(\mathbf{x}) \cdot Q_d(\|\mathbf{x}\|^2)$

**Verification for $d=1$**: $Q_1(x^2) = x^4 - 6x^2 + 3$. This is exactly the 4th Hermite polynomial $He_4(x) = x^4 - 6x^2 + 3$. ✓

(In 1D, the bi-Laplacian of $f$ is $f^{(4)}(x) = f(x) \cdot He_4(x)$.)

**Step 3**: Compute $\Psi_6^{NR} = \int [(\nabla^2)^2 f]^2 d\mathbf{x}$

$$\Psi_6^{NR} = \int f(\mathbf{x})^2 \cdot [Q_d(\|\mathbf{x}\|^2)]^2 \, d\mathbf{x}$$

Since $f(\mathbf{x})^2 = (2\pi)^{-d} e^{-\|\mathbf{x}\|^2}$, this integral is:

$$\Psi_6^{NR} = (2\pi)^{-d} \int e^{-\|\mathbf{x}\|^2} [Q_d(\|\mathbf{x}\|^2)]^2 \, d\mathbf{x}$$

Convert to polar: let $r^2 = \|\mathbf{x}\|^2$, surface area of $d$-sphere is $S_d = 2\pi^{d/2}/\Gamma(d/2)$:

$$= (2\pi)^{-d} \cdot S_d \int_0^\infty e^{-r^2} [Q_d(r^2)]^2 r^{d-1} \, dr$$

Substituting $u = r^2$, $dr = du/(2\sqrt{u})$:

$$= (2\pi)^{-d} \cdot \frac{\pi^{d/2}}{\Gamma(d/2)} \int_0^\infty e^{-u} [Q_d(u)]^2 u^{d/2-1} \, du$$

This is an expectation with respect to a Gamma$(d/2, 1)$ distribution! Let $U \sim \text{Gamma}(d/2, 1)$:

$$\Psi_6^{NR} = \frac{(2\pi)^{-d} \cdot \pi^{d/2}}{\Gamma(d/2)} \cdot \Gamma(d/2) \cdot \mathbb{E}[Q_d(U)^2]$$

$$= (2\pi)^{-d} \cdot \pi^{d/2} \cdot \mathbb{E}[Q_d(U)^2]$$

$$= \frac{1}{(4\pi)^{d/2}} \cdot \frac{1}{(\sqrt{\pi})^d} \cdot \pi^{d/2} \cdot \mathbb{E}[Q_d(U)^2]$$

Wait, let me redo this more carefully:

$(2\pi)^{-d} \cdot \pi^{d/2} = \frac{\pi^{d/2}}{(2\pi)^d} = \frac{1}{2^d \pi^{d/2}} = (4\pi)^{-d/2}$

So:
$$\boxed{\Psi_6^{NR} = (4\pi)^{-d/2} \cdot \mathbb{E}_{U \sim \text{Gamma}(d/2,1)}[Q_d(U)^2]}$$

**Step 4**: Compute $\mathbb{E}[Q_d(U)^2]$ where $U \sim \text{Gamma}(d/2, 1)$

$Q_d(u) = u^2 - 2(d+2)u + d(d+2)$

$Q_d(u)^2 = u^4 - 4(d+2)u^3 + [4(d+2)^2 + 2d(d+2)]u^2 - 4d(d+2)^2 u + d^2(d+2)^2$

Wait, let me expand carefully:
$(a - b + c)^2 = a^2 + b^2 + c^2 - 2ab + 2ac - 2bc$ where $a=u^2$, $b=2(d+2)u$, $c=d(d+2)$:

$Q_d^2 = u^4 + 4(d+2)^2 u^2 + d^2(d+2)^2 - 4(d+2)u^3 + 2d(d+2)u^2 - 4d(d+2)^2 u$

$= u^4 - 4(d+2)u^3 + [4(d+2)^2 + 2d(d+2)]u^2 - 4d(d+2)^2 u + d^2(d+2)^2$

$= u^4 - 4(d+2)u^3 + 2(d+2)(2d+4+d)u^2 - 4d(d+2)^2 u + d^2(d+2)^2$

$= u^4 - 4(d+2)u^3 + 2(d+2)(3d+4)u^2 - 4d(d+2)^2 u + d^2(d+2)^2$

For $U \sim \text{Gamma}(\alpha, 1)$ with $\alpha = d/2$, the moments are:
- $\mathbb{E}[U^k] = \alpha(\alpha+1)\cdots(\alpha+k-1) = \frac{\Gamma(\alpha+k)}{\Gamma(\alpha)}$

So:
- $\mathbb{E}[U] = d/2$
- $\mathbb{E}[U^2] = (d/2)(d/2+1) = d(d+2)/4$
- $\mathbb{E}[U^3] = (d/2)(d/2+1)(d/2+2) = d(d+2)(d+4)/8$
- $\mathbb{E}[U^4] = (d/2)(d/2+1)(d/2+2)(d/2+3) = d(d+2)(d+4)(d+6)/16$

Now:
$$\mathbb{E}[Q_d(U)^2] = \mathbb{E}[U^4] - 4(d+2)\mathbb{E}[U^3] + 2(d+2)(3d+4)\mathbb{E}[U^2] - 4d(d+2)^2\mathbb{E}[U] + d^2(d+2)^2$$

Substituting:
$$= \frac{d(d+2)(d+4)(d+6)}{16} - 4(d+2)\frac{d(d+2)(d+4)}{8} + 2(d+2)(3d+4)\frac{d(d+2)}{4} - 4d(d+2)^2\frac{d}{2} + d^2(d+2)^2$$

Let me factor out $d(d+2)$:

$$= d(d+2)\left[\frac{(d+4)(d+6)}{16} - \frac{4(d+2)(d+4)}{8} + \frac{2(d+2)(3d+4)}{4} - \frac{4d(d+2)}{2} + d(d+2)\right] + [\text{correction}]$$

Actually let me just compute term by term and simplify:

**Term 1**: $\frac{d(d+2)(d+4)(d+6)}{16}$

**Term 2**: $-4(d+2) \cdot \frac{d(d+2)(d+4)}{8} = -\frac{d(d+2)^2(d+4)}{2}$

**Term 3**: $2(d+2)(3d+4) \cdot \frac{d(d+2)}{4} = \frac{d(d+2)^2(3d+4)}{2}$

**Term 4**: $-4d(d+2)^2 \cdot \frac{d}{2} = -2d^2(d+2)^2$

**Term 5**: $d^2(d+2)^2$

Combine with common denominator 16:

$$= \frac{1}{16}\left[d(d+2)(d+4)(d+6) - 8d(d+2)^2(d+4) + 8d(d+2)^2(3d+4) - 32d^2(d+2)^2 + 16d^2(d+2)^2\right]$$

$$= \frac{d(d+2)}{16}\left[(d+4)(d+6) - 8(d+2)(d+4) + 8(d+2)(3d+4) - 32d(d+2) + 16d(d+2)\right]$$

Wait, let me not factor and just substitute $d=1$ to verify, then use a general pattern.

**Verification for $d=1$**: $\alpha = 1/2$
- $\mathbb{E}[U] = 1/2$
- $\mathbb{E}[U^2] = (1/2)(3/2) = 3/4$
- $\mathbb{E}[U^3] = (1/2)(3/2)(5/2) = 15/8$
- $\mathbb{E}[U^4] = (1/2)(3/2)(5/2)(7/2) = 105/16$

$Q_1(u) = u^2 - 6u + 3$
$\mathbb{E}[Q_1(U)^2] = \mathbb{E}[U^4 - 12U^3 + 42U^2 - 36U + 9]$
$= 105/16 - 12(15/8) + 42(3/4) - 36(1/2) + 9$
$= 105/16 - 180/8 + 126/4 - 18 + 9$
$= 105/16 - 360/16 + 504/16 - 288/16 + 144/16$
$= (105 - 360 + 504 - 288 + 144)/16$
$= 105/16$

And $\Psi_6^{NR}|_{d=1} = (4\pi)^{-1/2} \cdot 105/16 = \frac{105}{32\sqrt{\pi}}$

**Check against known 1D result**: For the standard normal, $\psi_8 = \int [f^{(4)}]^2 dx = \frac{105}{32\sqrt{\pi}}$.

Hmm — this is $\psi_8$ not $\psi_6$. Let me reconsider what functional we actually need.

---

### Correcting the functional identification

In SJ's notation:
- $\psi_r = \int [f^{(r/2)}(x)]^2 dx$ where the subscript $r$ indicates the order of derivative roughness
- $\psi_4 = \int [f''(x)]^2 dx$ (what we estimate for the final bandwidth)
- $\psi_6 = \int [f'''(x)]^2 dx$ (what we need for the pilot)

In $d$-D, the analog of $f''$ is $\nabla^2 f$ (Laplacian, 2nd order). The analog of $f'''$ is $\nabla(\nabla^2 f)$ — the gradient of the Laplacian, a VECTOR quantity.

Actually the correct multivariate generalization follows the AMISE theory. What we need for the pilot is the roughness functional ONE ORDER HIGHER than what we're estimating. Since we estimate $\Psi_4 = \int (\nabla^2 f)^2 dx$, the next-order functional is:

$$\Psi_6 = \int \|\nabla(\nabla^2 f)\|^2 d\mathbf{x} = \int \|\nabla^3 f\|^2 d\mathbf{x}$$

where $\nabla^3 f = \nabla(\nabla^2 f)$ is the gradient of the Laplacian.

### Normal reference for $\Psi_6 = \int \|\nabla(\nabla^2 f)\|^2 dx$

For $f = (2\pi\sigma^2)^{-d/2} e^{-\|\mathbf{x}\|^2/(2\sigma^2)}$:

$\nabla^2 f = \frac{1}{\sigma^2} f \cdot \left(\frac{\|\mathbf{x}\|^2}{\sigma^2} - d\right)$

$\nabla(\nabla^2 f) = \frac{1}{\sigma^2}\left[\nabla f \cdot L + f \cdot \nabla L\right]$

where $L = \|\mathbf{x}\|^2/\sigma^2 - d$, $\nabla L = 2\mathbf{x}/\sigma^2$, $\nabla f = -f\mathbf{x}/\sigma^2$.

$$\nabla(\nabla^2 f) = \frac{1}{\sigma^2}\left[-\frac{f\mathbf{x}}{\sigma^2} \cdot L + f \cdot \frac{2\mathbf{x}}{\sigma^2}\right] = \frac{f\mathbf{x}}{\sigma^4}\left[-L + 2\right] = \frac{f\mathbf{x}}{\sigma^4}\left[d + 2 - \frac{\|\mathbf{x}\|^2}{\sigma^2}\right]$$

So:
$$\|\nabla(\nabla^2 f)\|^2 = \frac{f^2 \|\mathbf{x}\|^2}{\sigma^8}\left[d + 2 - \frac{\|\mathbf{x}\|^2}{\sigma^2}\right]^2$$

And (for unit variance $\sigma=1$):
$$\Psi_6^{NR} = \int f^2 \|\mathbf{x}\|^2 \left[(d+2) - \|\mathbf{x}\|^2\right]^2 d\mathbf{x}$$

$$= (4\pi)^{-d/2} \cdot \mathbb{E}_{U \sim \text{Gamma}(d/2,1)}\left[U \cdot ((d+2) - 2U)^2 \cdot 2\right]$$

Actually let me redo with the chi-squared approach. Let $\mathbf{W} \sim N(0, I_d)$:

$$\Psi_6^{NR} = (4\pi)^{-d/2} \cdot \mathbb{E}\left[\|\mathbf{W}\|^2 \cdot (d+2-\|\mathbf{W}\|^2)^2\right]$$

Let $S = \|\mathbf{W}\|^2 \sim \chi^2_d$. Then:

$$\Psi_6^{NR} = (4\pi)^{-d/2} \cdot \mathbb{E}[S(d+2-S)^2]$$

$$= (4\pi)^{-d/2} \cdot \mathbb{E}[S(d+2)^2 - 2S^2(d+2) + S^3]$$

$$= (4\pi)^{-d/2} \cdot [(d+2)^2 \mathbb{E}[S] - 2(d+2)\mathbb{E}[S^2] + \mathbb{E}[S^3]]$$

For $S \sim \chi^2_d$:
- $\mathbb{E}[S] = d$
- $\mathbb{E}[S^2] = d(d+2)$
- $\mathbb{E}[S^3] = d(d+2)(d+4)$

Substituting:
$$= (4\pi)^{-d/2}[d(d+2)^2 - 2(d+2) \cdot d(d+2) + d(d+2)(d+4)]$$
$$= (4\pi)^{-d/2} \cdot d(d+2)[(d+2) - 2(d+2) + (d+4)]$$
$$= (4\pi)^{-d/2} \cdot d(d+2)[d+2 - 2d - 4 + d + 4]$$
$$= (4\pi)^{-d/2} \cdot d(d+2) \cdot 2$$

$$\boxed{\Psi_6^{NR} = \frac{2d(d+2)}{(4\pi)^{d/2}}}$$

For general $\sigma$: scale by $\sigma^{-(d+8)}$ (each derivative adds one power, $\nabla^3$ adds 3, times 2 for squaring gives 6; integration adds $d$ from the volume element; total = $d + 6 + 2 = d + 8$... let me verify dimensionally).

Actually, for $f_\sigma(\mathbf{x}) = \sigma^{-d} f_1(\mathbf{x}/\sigma)$:
$\nabla^2 f_\sigma = \sigma^{-(d+2)} (\nabla^2 f_1)(\mathbf{x}/\sigma)$
$\nabla(\nabla^2 f_\sigma) = \sigma^{-(d+3)} [\nabla(\nabla^2 f_1)](\mathbf{x}/\sigma)$
$\|\nabla(\nabla^2 f_\sigma)\|^2 = \sigma^{-2(d+3)} \|\cdots\|^2$
$\int = \sigma^{-2(d+3)} \cdot \sigma^d = \sigma^{-(d+6)}$

So: $\Psi_6^{NR}(\sigma) = \frac{2d(d+2)}{(4\pi)^{d/2} \sigma^{d+6}}$

**Verification for $d=1$**: $\Psi_6^{NR} = \frac{2 \cdot 1 \cdot 3}{2\sqrt{\pi} \cdot \sigma^7} = \frac{6}{2\sqrt{\pi}\sigma^7} = \frac{3}{\sqrt{\pi}\sigma^7}$

The known 1D result is $\psi_6 = \frac{15}{16\sqrt{\pi}\sigma^7}$... These don't match. The discrepancy suggests my functional identification is wrong.

Let me reconsider. In 1D, $\psi_6 = \int [f'''(x)]^2 dx = \frac{-15}{16\sqrt{\pi}} \sigma^{-7}$ (from SJ). For a standard normal ($\sigma=1$): $\psi_6 = \frac{15}{16\sqrt{\pi}}$.

My calculation gives $3/\sqrt{\pi} = 48/(16\sqrt{\pi})$. Off by a factor. The issue might be in the multivariate functional identification.

The correct $d$-D functional for the two-stage SJ chain needs more careful asymptotic analysis. Let me record what we have and note this needs further work:

---

## Summary of True Generalization Status

### COMPLETE:
- $\hat{\Psi}_4$ closed-form estimation: $P_d(t)$ polynomial ✅
- AMISE-optimal bandwidth from $\hat{\Psi}_4$ ✅
- One-stage DPI (Silverman pilot → $\hat{\Psi}_4$ → $h^*$) ✅
- STE wrapper (root-finding around our formula) ✅ (trivial code)

### DERIVED BUT NEEDS VERIFICATION:
- $(\nabla^2)^2 f = f \cdot Q_d(\|\mathbf{x}\|^2)$ with $Q_d(s) = s^2 - 2(d+2)s + d(d+2)$
- $\nabla(\nabla^2 f)$ for the Gaussian: computed above
- Normal reference $\Psi_6^{NR}$: formula derived but d=1 cross-check doesn't match → needs correction

### THE GAP:
The exact multivariate analog of SJ's $\psi_6$ depends on which functional appears in the asymptotic bias of the $\hat{\Psi}_4$ estimator. In 1D this is straightforward ($\psi_6 = R(f''')$). In $d$-D, the relevant functional may be:
- $\int \|\nabla(\nabla^2 f)\|^2 dx$ (gradient of Laplacian)
- $\int (\nabla^4 f)^2 dx$ (bi-Laplacian squared, i.e., $(\nabla^2)^2$)
- Or a mixed functional involving the full Hessian

**Resolving this requires deriving the asymptotic bias of $\hat{\Psi}_4(h_0)$ as an estimator of $\Psi_4$ in $d$ dimensions** — essentially a multivariate Taylor expansion of $\mathbb{E}[\hat{\Psi}_4(h_0)] - \Psi_4$ to leading order in $h_0$.

This is a focused ~1 week calculation. The answer determines:
1. Which $\Psi_6$ functional to estimate
2. The correct normal reference formula
3. The pilot bandwidth $g_2$ scaling

### WHAT WE CAN PUBLISH NOW (without the full chain):
1. The one-stage DPI (already done, already publishable)
2. The STE variant using our formula (5 lines of code, publishable as an extension)
3. Note the two-stage derivation as future work with the partial results above

### WHAT WOULD MAKE IT A COMPLETE GENERALIZATION:
1. Derive the asymptotic bias $\mathbb{E}[\hat{\Psi}_4(h_0)] - \Psi_4 = c \cdot h_0^4 \cdot \Psi_6 + o(h_0^4)$ identifying which $\Psi_6$
2. Compute its normal reference
3. Derive the pilot $g_2$ from it
4. Implement and test two-stage DPI

This is feasible but is genuinely a separate paper's contribution — it's the theoretical core that Hall, Sheather & Jones worked on for the 1991 convergence rate paper.
