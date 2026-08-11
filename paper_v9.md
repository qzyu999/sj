# Whitened Coordinates: What They Are and Why We Need Them

## The Problem

Our derivation produces a **scalar** bandwidth $h$ — the same number in every direction. But real data isn't spherical. Features have different scales (age in years vs income in dollars) and correlations (height and weight move together).

A single number $h$ can't be simultaneously correct for a feature with variance 100 and a feature with variance 0.01. You'd oversmooth one and undersmooth the other.

## The Solution: Whiten First, Then Use Scalar Bandwidth

**Whitening** = a linear transformation that makes the data look spherical.

### Before whitening:
```
Feature 1: mean=170, std=10, units=cm (height)
Feature 2: mean=70,  std=15, units=kg (weight)  
Correlation: 0.7

Data cloud is elongated and tilted:

    ╱
   ╱  •  • •
  ╱  • • • • •
 ╱  • • • •
╱   • •
```

### The whitening transform:

$$Y_i = \hat{\Sigma}^{-1/2} X_i$$

where $\hat{\Sigma}$ is the sample covariance matrix. This simultaneously:
1. Scales each axis to unit variance
2. Removes correlations between features
3. Makes the covariance matrix = identity

### After whitening:
```
Feature 1: mean=0, std=1 (unitless)
Feature 2: mean=0, std=1 (unitless)
Correlation: 0

Data cloud is spherical:

    •  •
  •  •  •  •
   •  • •  •
  •  •  •
    •  •
```

Now a single scalar $h$ makes sense — all directions are equivalent.

## Concretely: The Matrix Square Root

The covariance matrix $\hat{\Sigma}$ can be decomposed as:

$$\hat{\Sigma} = Q \Lambda Q^T$$

where $Q$ is the matrix of eigenvectors (rotation) and $\Lambda$ is diagonal (eigenvalues = variances along principal axes).

Then:

$$\hat{\Sigma}^{-1/2} = Q \Lambda^{-1/2} Q^T$$

This means:
- Rotate to principal axes ($Q^T$)
- Scale each axis by $1/\sqrt{\lambda_k}$ (makes variance = 1)
- Rotate back ($Q$)

Multiplying data by $\hat{\Sigma}^{-1/2}$ applies all three operations at once.

## Why Our Derivation Requires This

Our polynomial $P_d(t)$ is derived under the assumption of an **isotropic** kernel — same bandwidth in all directions. The AMISE formula:

$$\text{AMISE}(h) = \frac{R(K)}{nh^d} + \frac{h^4}{4}\Psi$$

with $\Psi = \int [\nabla^2 f]^2 dx$ (the Laplacian roughness) is only valid when the bandwidth is $H = h^2 I_d$ (identity matrix scaled by $h^2$).

If instead we had $H = h^2 \Sigma$ for some non-identity $\Sigma$, the bias functional would be:

$$\text{Bias}[\hat{f}(x)] \propto \text{tr}\{\Sigma \nabla^2 f(x)\}$$

and the integrated squared bias would be:

$$\int [\text{tr}\{\Sigma \nabla^2 f\}]^2 dx \neq \int [\nabla^2 f]^2 dx$$

These are the same only when $\Sigma = I$. **Whitening makes $\Sigma = I$.**

## The Algorithm Flow

```python
# 1. Compute sample covariance
Sigma_hat = np.cov(X, rowvar=False)  # d × d matrix

# 2. Whiten the data
Y = (inv(sqrtm(Sigma_hat)) @ X.T).T  # Now cov(Y) ≈ I_d

# 3. Compute scalar bandwidth on whitened data
h_star = gsj_formula(Y)  # Uses P_d polynomial, pairwise sums

# 4. The effective bandwidth in ORIGINAL coordinates is:
#    H = h_star² · Sigma_hat
#    (wider in directions with more spread, narrower where data is tight)
```

## What scipy Does Internally

When you call `stats.gaussian_kde(X.T, bw_method=h_factor)`:
- scipy computes the sample covariance internally
- Multiplies: `covariance = factor² * data_covariance`
- This IS the $H = h^2 \hat{\Sigma}$ parameterization

So our `gsj.bandwidth(X)` returns the factor that scipy needs — it's already designed to work with scipy's convention.

## Why the Reviewer Flagged This

The original paper wrote:
> "The bandwidth is $H = h^2\hat{\Sigma}$" and "The roughness is $\Psi = \int (\nabla^2 f)^2 dx$"

A reader could object: "These two statements don't go together! If $H = h^2\Sigma$ then the correct roughness involves tr{$\Sigma\nabla^2 f$}, not the plain Laplacian!"

The fix: be explicit that the derivation happens **in whitened coordinates** where $\Sigma = I$, and then the scalar factor maps back via $H = h^2\hat{\Sigma}$. The revised paper now has a Remark stating this clearly.

## Intuition Summary

| Concept | Analogy |
|---------|---------|
| Raw data | A photo taken at an angle — perspective distortion |
| Whitening | Correcting the perspective — making circles look like circles |
| Scalar bandwidth | Choosing one "blur radius" — only works if everything is round |
| Final bandwidth $H = h^2\hat{\Sigma}$ | Applying the blur, then un-correcting the perspective |


---

## Critical Clarification: Whitening Is Internal — You Always Work on Original Data

### The common concern

> "If the derivation assumes whitened data, do I have to whiten my data before using the KDE? Do my plots and analyses only work on the whitened version?"

**No.** Whitening is an internal computational step that the user never sees. You always analyze your original data.

### The actual flow

```
YOUR ORIGINAL DATA (height in cm, weight in kg, age in years)
        │
        ▼
    gsj.bandwidth(X)
        │
        │  [INTERNALLY: whiten → compute P_d formula → get scalar h*]
        │  [User never sees the whitened data]
        │
        ▼
    Returns: h* = 0.35  (just a number)
        │
        ▼
    scipy.gaussian_kde(X.T, bw_method=h*)  ← feeds ORIGINAL data
        │
        ▼
    kde(eval_points)  ← density evaluated in ORIGINAL space
        │
        ▼
    YOUR PLOTS, ANOMALY SCORES, ANALYSIS — all in original units
```

The whitening happens invisibly inside `gsj.bandwidth()`. Everything else — the KDE, the density evaluation, the visualization — operates on the original untransformed data.

### Why this works: scipy already handles the covariance

When you pass a factor to scipy's `gaussian_kde`, it does NOT use the same bandwidth in all directions. It internally computes:

$$H = \text{factor}^2 \times \hat{\Sigma}_{\text{data}}$$

So if your data has height (std=10cm) and weight (std=15kg):
- Direction with more spread (weight) → wider kernel in that direction
- Direction with less spread (height) → narrower kernel in that direction

scipy does this automatically. Our job is just to find the right scalar factor. The whitening inside `gsj.bandwidth()` is how we compute that factor correctly — but once we have it, we hand it to scipy which applies the full covariance-aware kernel to the original data.

### Concrete example

```python
import numpy as np
from scipy.stats import gaussian_kde
from gsj import bandwidth

# Original data — height(cm), weight(kg), age(years)
# Very different scales! 
X = np.column_stack([
    np.random.normal(170, 10, 1000),   # height: std=10
    np.random.normal(70, 15, 1000),    # weight: std=15
    np.random.normal(35, 12, 1000),    # age: std=12
])

# GSJ computes the optimal factor (whitening is INTERNAL)
h = bandwidth(X)  # Returns e.g. 0.35

# Build KDE on ORIGINAL data — scipy handles the covariance
kde = gaussian_kde(X.T, bw_method=h)

# Evaluate density at a point in ORIGINAL UNITS
point = np.array([[175, 80, 40]]).T  # 175cm, 80kg, 40 years old
density = kde(point)  # Works perfectly in original coordinates

# Plot in original coordinates — nothing is whitened
import matplotlib.pyplot as plt
x_grid = np.linspace(140, 200, 100)  # heights in cm
plt.plot(x_grid, kde(np.vstack([x_grid, np.full(100, 70), np.full(100, 35)])))
plt.xlabel('Height (cm)')  # Original units!
```

### The analogy: GPS navigation

Think of it like computing a driving route:
- Your GPS internally converts your location to a flat projection (Mercator), computes the shortest path, then displays the route on a curved Earth map.
- You never interact with the Mercator coordinates — you see everything on the normal map in lat/long.
- The internal projection is a computational convenience that makes the math work.

Whitening is the same: it makes the bandwidth math work (by making the kernel isotropic), but the result is expressed back in your original coordinate system automatically.

### Why it's mathematically valid

The density estimator $\hat{f}_H(x)$ with bandwidth matrix $H = h^2 \hat{\Sigma}$ applied to data $X_1, ..., X_n$ in original coordinates is **mathematically identical** to the estimator $\hat{f}_{h^2 I}(y)$ applied to whitened data $Y_i = \hat{\Sigma}^{-1/2} X_i$ — they're the same function, just expressed in different coordinates.

$$\hat{f}_{h^2\hat{\Sigma}}(x) = |\hat{\Sigma}|^{-1/2} \cdot \hat{f}_{h^2 I}(\hat{\Sigma}^{-1/2} x)$$

This means:
- Computing $h^*$ in whitened space → applying it in original space (via scipy's $H = h^2\hat{\Sigma}$)
- Is identical to computing the full bandwidth matrix directly on original data
- But much simpler (one scalar vs one matrix)

### Bottom line

| What you do | What happens internally |
|-------------|------------------------|
| Call `gsj.bandwidth(X)` | Whitens, computes polynomial, returns scalar |
| Pass factor to scipy | scipy uses $H = \text{factor}^2 \times \text{cov}(X)$ on original data |
| Evaluate/plot/score | Everything in original coordinates and units |
| **You never see whitened data** | **It's a computational implementation detail** |


---

## Isotropic Bandwidth: What, Why, and How

### What "Isotropic" Means

**Isotropic** = "the same in all directions." An isotropic bandwidth means the kernel has the same width along every axis.

In a KDE, the bandwidth controls the kernel shape:

$$K_H(\mathbf{t}) = \frac{1}{(2\pi)^{d/2}|H|^{1/2}} \exp\left(-\frac{1}{2}\mathbf{t}^T H^{-1} \mathbf{t}\right)$$

The bandwidth matrix $H$ is $d \times d$ and determines the kernel's shape:

| $H$ | Shape of kernel | Parameters | Name |
|-----|----------------|------------|------|
| $h^2 I_d$ | Sphere (same width in all directions) | 1 scalar | **Isotropic** |
| $\text{diag}(h_1^2, ..., h_d^2)$ | Axis-aligned ellipsoid (different width per axis) | $d$ scalars | **Diagonal** |
| General positive-definite matrix | Rotated ellipsoid (any orientation) | $d(d+1)/2$ | **Full matrix** |

Visually in 2D:

```
Isotropic (h²I):         Diagonal:              Full matrix:
    ○                      ⬯                      ⬮
  (circle)          (axis-aligned ellipse)   (rotated ellipse)
  1 parameter        2 parameters            3 parameters
```

### Why We Restrict to Isotropic

Three reasons:

**1. Mathematical tractability**

The AMISE for a general matrix $H$ has the bias term:

$$\text{Bias}[\hat{f}(x)] = \frac{1}{2}\text{tr}\{H \nabla^2 f(x)\} + O(\|H\|^2)$$

where $\nabla^2 f$ is the Hessian matrix (not the scalar Laplacian). The integrated squared bias is:

$$\int [\text{tr}\{H \nabla^2 f(x)\}]^2 dx$$

For general $H$, optimizing this requires inverting a $d(d+1)/2$-dimensional system — the Duong-Hazelton approach, which is iterative and expensive.

For isotropic $H = h^2 I_d$:

$$\text{tr}\{h^2 I \nabla^2 f\} = h^2 \text{tr}\{\nabla^2 f\} = h^2 \nabla^2 f$$

The trace of the Hessian IS the Laplacian (a scalar). The integrated squared bias becomes:

$$h^4 \int [\nabla^2 f(x)]^2 dx = h^4 \Psi$$

Now optimizing over a single scalar $h$ is trivial (take derivative, set to zero).

**2. One parameter instead of many**

| Bandwidth type | Parameters | Estimation difficulty |
|----------------|-----------|----------------------|
| Isotropic | 1 | Closed form (our method) |
| Diagonal | $d$ | $d$ coupled equations |
| Full matrix | $d(d+1)/2$ | Iterative optimization |

For $d = 10$:
- Isotropic: 1 parameter (instant)
- Diagonal: 10 parameters (tractable but needs iteration)
- Full matrix: 55 parameters (expensive, requires large $n$)

With limited data, estimating 55 bandwidth parameters reliably is very hard. Isotropic is robust — it can't overfit because there's nothing to overfit.

**3. After whitening, isotropic IS appropriate**

This is the key insight that makes the restriction acceptable:

If the data has been whitened ($\text{cov}(\mathbf{Y}) = I_d$), then the "natural" scales in all directions are already equalized. An isotropic kernel in whitened space corresponds to a **covariance-adapted kernel** in original space:

$$H_{\text{original}} = h^2 \hat{\Sigma}$$

So the restriction isn't really "same bandwidth in all directions of the raw data." It's "same bandwidth in all directions of the *standardized* data" — which maps back to a full covariance-shaped kernel on the raw data.

### The Rigorous Derivation: AMISE for Isotropic Bandwidth

Starting from the bias-variance decomposition of the MISE.

**Setup:** Data $\mathbf{Y}_1, ..., \mathbf{Y}_n \in \mathbb{R}^d$ (already whitened, covariance = $I_d$). Kernel: $K_h(\mathbf{t}) = (2\pi h^2)^{-d/2}\exp(-\|\mathbf{t}\|^2/(2h^2))$. KDE: $\hat{f}_h(\mathbf{x}) = n^{-1}\sum_i K_h(\mathbf{x} - \mathbf{Y}_i)$.

**Step 1: Bias of the KDE at a point $\mathbf{x}$**

$$\text{Bias}[\hat{f}_h(\mathbf{x})] = \mathbb{E}[\hat{f}_h(\mathbf{x})] - f(\mathbf{x})$$

Expand $\mathbb{E}[\hat{f}_h(\mathbf{x})] = \int K_h(\mathbf{x} - \mathbf{y}) f(\mathbf{y}) d\mathbf{y} = (K_h * f)(\mathbf{x})$.

Taylor-expand $f(\mathbf{y})$ around $\mathbf{x}$. Let $\mathbf{t} = \mathbf{y} - \mathbf{x}$:

$$f(\mathbf{x} + \mathbf{t}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^T \mathbf{t} + \frac{1}{2}\mathbf{t}^T \nabla^2 f(\mathbf{x}) \mathbf{t} + O(\|\mathbf{t}\|^3)$$

Then:

$$\mathbb{E}[\hat{f}_h(\mathbf{x})] = \int K_h(\mathbf{t})\left[f(\mathbf{x}) + \nabla f^T \mathbf{t} + \frac{1}{2}\mathbf{t}^T \nabla^2 f \mathbf{t}\right] d\mathbf{t} + O(h^4)$$

Using properties of the Gaussian kernel:
- $\int K_h(\mathbf{t}) d\mathbf{t} = 1$
- $\int K_h(\mathbf{t}) t_k d\mathbf{t} = 0$ (odd function)
- $\int K_h(\mathbf{t}) t_k t_l d\mathbf{t} = h^2 \delta_{kl}$ (second moment of Gaussian)

So:

$$\mathbb{E}[\hat{f}_h(\mathbf{x})] = f(\mathbf{x}) + \frac{h^2}{2}\sum_{k=1}^d \frac{\partial^2 f}{\partial x_k^2}(\mathbf{x}) + O(h^4) = f(\mathbf{x}) + \frac{h^2}{2}\nabla^2 f(\mathbf{x}) + O(h^4)$$

Therefore:

$$\text{Bias}[\hat{f}_h(\mathbf{x})] = \frac{h^2}{2}\nabla^2 f(\mathbf{x}) + O(h^4)$$

**Step 2: Variance of the KDE at a point $\mathbf{x}$**

$$\text{Var}[\hat{f}_h(\mathbf{x})] = \frac{1}{n}\text{Var}[K_h(\mathbf{x} - \mathbf{Y})] \approx \frac{1}{n}\mathbb{E}[K_h(\mathbf{x} - \mathbf{Y})^2]$$

For small $h$:

$$\mathbb{E}[K_h(\mathbf{x} - \mathbf{Y})^2] \approx f(\mathbf{x}) \int K_h(\mathbf{t})^2 d\mathbf{t} = f(\mathbf{x}) \cdot (4\pi h^2)^{-d/2}$$

So:

$$\text{Var}[\hat{f}_h(\mathbf{x})] \approx \frac{f(\mathbf{x})}{n(4\pi h^2)^{d/2}}$$

**Step 3: MISE = Integrated (Bias² + Variance)**

$$\text{MISE}(h) = \int \text{Bias}^2 d\mathbf{x} + \int \text{Var} \, d\mathbf{x}$$

$$= \int \left[\frac{h^2}{2}\nabla^2 f\right]^2 d\mathbf{x} + \int \frac{f(\mathbf{x})}{n(4\pi h^2)^{d/2}} d\mathbf{x} + \text{higher order}$$

$$= \frac{h^4}{4}\underbrace{\int [\nabla^2 f(\mathbf{x})]^2 d\mathbf{x}}_{\Psi} + \frac{1}{n(4\pi h^2)^{d/2}}\underbrace{\int f(\mathbf{x}) d\mathbf{x}}_{= 1} + O(\cdots)$$

$$\text{AMISE}(h) = \frac{(4\pi)^{-d/2}}{nh^d} + \frac{h^4}{4}\Psi$$

**Step 4: Minimize AMISE**

$$\frac{\partial}{\partial h}\text{AMISE} = -\frac{d(4\pi)^{-d/2}}{nh^{d+1}} + h^3\Psi = 0$$

$$h^{d+4} = \frac{d(4\pi)^{-d/2}}{n\Psi}$$

$$h^* = \left(\frac{d}{n\Psi(4\pi)^{d/2}}\right)^{1/(d+4)}$$

**This is the AMISE-optimal isotropic bandwidth.**

### Why "Isotropic" Appears in Step 1

The crucial moment: in Step 1, the integral $\int K_h(\mathbf{t}) t_k t_l \, d\mathbf{t} = h^2 \delta_{kl}$.

This is because our kernel is $K_h(\mathbf{t}) = (2\pi h^2)^{-d/2}\exp(-\|\mathbf{t}\|^2/(2h^2))$ — isotropic. Its second moment is $h^2$ in EVERY direction and ZERO for cross-terms.

If instead we had a non-isotropic kernel with bandwidth matrix $H$:

$$\int K_H(\mathbf{t}) t_k t_l \, d\mathbf{t} = H_{kl}$$

Then the bias would be:

$$\text{Bias} = \frac{1}{2}\sum_{k,l} H_{kl} \frac{\partial^2 f}{\partial x_k \partial x_l} = \frac{1}{2}\text{tr}\{H \nabla^2 f\}$$

For $H = h^2 I$: $\text{tr}\{h^2 I \cdot \nabla^2 f\} = h^2 \sum_k \frac{\partial^2 f}{\partial x_k^2} = h^2 \nabla^2 f$.

So: **isotropic kernel → bias involves only the Laplacian (sum of second derivatives) → the roughness functional is $\int (\nabla^2 f)^2 dx$ → which has a closed-form pairwise evaluation (our $P_d$ polynomial).**

With a non-isotropic kernel, the bias involves the full Hessian, and the roughness functional becomes much more complex — requiring matrix optimization (Duong-Hazelton approach).

### When Isotropic Is Sufficient vs Insufficient

**Sufficient (our method works well):**
- Data has been whitened (covariance ≈ identity) → all directions are equivalent
- Data is roughly spherical after standardization
- Features are of similar "importance" for density estimation
- You have limited data (not enough to reliably estimate $d(d+1)/2$ parameters)

**Insufficient (diagonal or full matrix may be needed):**
- Data has strong axis-aligned structure that whitening doesn't capture (e.g., one feature is discrete/categorical while others are continuous)
- Very large $n$ and low $d$ where you can afford to estimate more parameters
- Prior knowledge that certain directions need very different smoothing

**In practice:** For the PCA-reduced embeddings in our experiments (d=5-20), the data is already fairly isotropic after standardization. The isotropic assumption is reasonable and our experiments confirm it works well.

### Summary: The Isotropic Restriction Is a Feature, Not a Bug

```
Full matrix H (55 params for d=10):
  + Captures every possible directional structure
  - Requires massive data to estimate reliably
  - Iterative optimization
  - Only available in R (ks package)

Diagonal H (10 params for d=10):
  + Per-axis control
  - Still requires solving coupled equations
  - Not available as a closed form

Isotropic h²I (1 param):
  + Closed-form solution (our P_d polynomial)
  + Robust — can't overfit the bandwidth
  + Fast — no iteration
  + After whitening, captures covariance structure anyway
  - Can't capture directional differences BEYOND what covariance explains
```

The isotropic restriction combined with whitening gives you the covariance-shaped bandwidth $H = h^2\hat{\Sigma}$ in original space — which is a very reasonable default that captures the dominant source of anisotropy (the covariance structure). Residual anisotropy beyond covariance is typically small and requires much more data to estimate.
