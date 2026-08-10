# The Sheather-Jones Method in $d$ Dimensions: A Closed-Form Derivation

## Table of Contents

1. [Background: The 1D Derivation Revisited](#1-background)
2. [Generalizing to $d$ Dimensions: Setup](#2-generalizing)
3. [The Multivariate Kernel and its Laplacian](#3-laplacian)
4. [The Roughness Integral: Pairwise Decomposition](#4-roughness)
5. [Combining the Exponential Terms (Completing the Square)](#5-completing-square)
6. [Recognizing the Expectation](#6-expectation)
7. [Computing the Polynomial Expectation (Moments)](#7-moments)
8. [Assembling the Final Formula](#8-final)
9. [The Optimal Bandwidth](#9-optimal)
10. [Verification: Recovering the 1D Case](#10-verification)
11. [Summary](#11-summary)

---

## 1. Background: The 1D Derivation Revisited
<a id="1-background"></a>

In the original 1D derivation (see *Intro to KDE* notebook, Section 4.2.1.2), the Sheather-Jones method was derived step by step. The key idea was:

1. Start with the AMISE formula (equation 8 in the notebook):

$$AMISE(h) = \frac{R(K)}{nh} + \frac{h^4 \sigma_K^4 R(f'')}{4}$$

2. Minimize with respect to $h$ to get the optimal bandwidth (equation 9):

$$h^* = \left(\frac{R(K)}{n\sigma_K^4 R(f'')}\right)^{1/5}$$

3. Since $R(f'') = \int [f''(x)]^2 dx$ is unknown, estimate it using the KDE itself with a pilot bandwidth $h_0$ (the Silverman estimate, equation 14):

$$\hat{R}(f'') = R(\hat{f}'') = \int [\hat{f}''(x)]^2 dx$$

4. Expand $\hat{f}''(x) = \frac{1}{nh_0^3}\sum_{i=1}^n L''\left(\frac{x - X_i}{h_0}\right)$, then compute the roughness integral by examining pairwise terms.

5. The critical step was to **combine the two Gaussian exponentials** (completing the square), which produced a new Gaussian centered at $\frac{x_i + x_j}{2}$ with variance $\frac{h_0^2}{2}$. This transformed the integral into an **expectation** of a 4th-degree polynomial $A$ with respect to a Normal distribution.

6. The moments of that Normal distribution gave the closed-form answer.

The derivation below follows this exact same pipeline — step by step, with explicit algebra — but in $d$ dimensions. Every step in 1D has a direct multivariate analog, and we will show each one.

---

## 2. Generalizing to $d$ Dimensions: Setup
<a id="2-generalizing"></a>

### Data and KDE

We have data $X_1, \ldots, X_n \in \mathbb{R}^d$. The Gaussian kernel density estimator with bandwidth parameter $h$ is:

(equation 1)

$$\hat{f}(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n K_h(\mathbf{x} - X_i)$$

where the multivariate Gaussian kernel is:

(equation 2)

$$K_h(\mathbf{t}) = \frac{1}{(2\pi h^2)^{d/2}} \exp\left(-\frac{\|\mathbf{t}\|^2}{2h^2}\right)$$

Here $\|\mathbf{t}\|^2 = t_1^2 + t_2^2 + \cdots + t_d^2$ is the squared Euclidean norm. This is the "isotropic" (spherically symmetric) kernel — the same bandwidth $h$ is used in all directions.

### The AMISE in $d$ dimensions

In the 1D case, the AMISE was $\frac{R(K)}{nh} + \frac{h^4}{4}R(f'')$ (using the Normal kernel where $\sigma_K = 1$). In $d$ dimensions the analog is:

(equation 3)

$$AMISE(h) = \frac{R(K)}{nh^d} + \frac{h^4}{4}\Psi$$

where:
- $R(K) = \int K(\mathbf{t})^2 d\mathbf{t} = \frac{1}{(4\pi)^{d/2}}$ is the roughness of the standard $d$-dimensional Gaussian kernel
- $\Psi = \int [\nabla^2 f(\mathbf{x})]^2 d\mathbf{x}$ is the roughness of the **Laplacian** of the true density

The Laplacian $\nabla^2 f = \sum_{k=1}^d \frac{\partial^2 f}{\partial x_k^2}$ is the multivariate generalization of $f''(x)$. When $d=1$, $\nabla^2 f = f''$ and we recover the 1D formula.

### The optimal bandwidth

Setting $\frac{\partial}{\partial h} AMISE = 0$:

(equation 4)

$$-\frac{d \cdot R(K)}{nh^{d+1}} + h^3 \Psi = 0 \implies h^{d+4} = \frac{d \cdot R(K)}{n \cdot \Psi}$$

(equation 5)

$$\boxed{h^* = \left(\frac{d \cdot R(K)}{n \cdot \Psi}\right)^{1/(d+4)}}$$

When $d=1$: $h^5 = \frac{R(K)}{n\Psi}$, recovering equation 9 from the notebook. ✓

### What we need

Just as in 1D, we need to estimate $\Psi = \int [\nabla^2 f(\mathbf{x})]^2 d\mathbf{x}$ using the KDE with a pilot bandwidth $h_0$. The Silverman pilot in $d$ dimensions is:

(equation 6)

$$h_0 = \left(\frac{4}{n(d+2)}\right)^{1/(d+4)} \cdot \hat{\sigma}$$

where $\hat{\sigma}$ is the sample standard deviation (or, after whitening the data to have identity covariance, $\hat{\sigma} = 1$). When $d = 1$: $h_0 = (4/(3n))^{1/5} \hat{\sigma}$, which is equation 14 from the notebook. ✓

---

## 3. The Multivariate Kernel and its Laplacian
<a id="3-laplacian"></a>

In 1D, the key ingredient was $L''(z) = \frac{-1}{\sqrt{2\pi}} e^{-z^2/2}(1 - z^2)$ (equation 17 in the notebook). We need the $d$-dimensional analog.

### Step 3.1: First partial derivative

Starting from $K_h(\mathbf{t}) = \frac{1}{(2\pi h^2)^{d/2}} \exp\left(-\frac{\|\mathbf{t}\|^2}{2h^2}\right)$, take the partial derivative with respect to coordinate $t_k$:

(equation 7)

$$\frac{\partial K_h}{\partial t_k} = K_h(\mathbf{t}) \cdot \left(-\frac{t_k}{h^2}\right)$$

This follows from the chain rule applied to $\exp(-\|\mathbf{t}\|^2/(2h^2))$, since $\frac{\partial}{\partial t_k}\|\mathbf{t}\|^2 = 2t_k$.

### Step 3.2: Second partial derivative

Apply the product rule to differentiate $K_h(\mathbf{t}) \cdot (-t_k/h^2)$ with respect to $t_k$:

(equation 8)

$$\frac{\partial^2 K_h}{\partial t_k^2} = \frac{\partial K_h}{\partial t_k} \cdot \left(-\frac{t_k}{h^2}\right) + K_h(\mathbf{t}) \cdot \left(-\frac{1}{h^2}\right)$$

$$= K_h(\mathbf{t}) \cdot \frac{t_k^2}{h^4} - K_h(\mathbf{t}) \cdot \frac{1}{h^2} = K_h(\mathbf{t})\left(\frac{t_k^2}{h^4} - \frac{1}{h^2}\right)$$

### Step 3.3: The Laplacian (sum over all dimensions)

(equation 9)

$$\nabla^2 K_h(\mathbf{t}) = \sum_{k=1}^d \frac{\partial^2 K_h}{\partial t_k^2} = K_h(\mathbf{t}) \sum_{k=1}^d \left(\frac{t_k^2}{h^4} - \frac{1}{h^2}\right) = K_h(\mathbf{t}) \left(\frac{\|\mathbf{t}\|^2}{h^4} - \frac{d}{h^2}\right)$$

Factoring out $1/h^2$:

(equation 10)

$$\boxed{\nabla^2 K_h(\mathbf{t}) = \frac{1}{h^2} K_h(\mathbf{t}) \left(\frac{\|\mathbf{t}\|^2}{h^2} - d\right)}$$

### Comparison with 1D

When $d = 1$: $\nabla^2 K_h(t) = \frac{1}{h^2} K_h(t)(t^2/h^2 - 1)$. Setting $z = t/h$, this is $\frac{1}{h^2} K_h(t)(z^2 - 1)$. The factor $(z^2 - 1)$ is exactly $(1 - z^2)$ with a sign flip, and when we multiply by $-1$ from the normalization (the $\frac{-1}{\sqrt{2\pi}}$ in equation 17 of the notebook), we get the same $L''(z) = \frac{-1}{\sqrt{2\pi}}e^{-z^2/2}(1 - z^2)$. ✓

### Notation

Define the "Laplacian polynomial":

$$L_h(\mathbf{t}) \equiv \frac{\|\mathbf{t}\|^2}{h^2} - d$$

so that $\nabla^2 K_h(\mathbf{t}) = \frac{1}{h^2} K_h(\mathbf{t}) \cdot L_h(\mathbf{t})$.

When $d = 1$: $L_h(t) = t^2/h^2 - 1 = z^2 - 1$, which corresponds to the $(1 - z^2)$ polynomial in the notebook (differing by a sign that cancels in the square). ✓

---

## 4. The Roughness Integral: Pairwise Decomposition
<a id="4-roughness"></a>

This follows the same logic as equations 18–20 in the notebook. We expand the square of the sum into pairwise terms.

### Step 4.1: The KDE Laplacian

(equation 11)

$$\nabla^2 \hat{f}_{h_0}(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n \nabla^2 K_{h_0}(\mathbf{x} - X_i) = \frac{1}{nh_0^2} \sum_{i=1}^n K_{h_0}(\mathbf{x} - X_i) \cdot L_{h_0}(\mathbf{x} - X_i)$$

In 1D this was $\hat{f}''(x) = \frac{1}{nh_0^3}\sum_i L''(\frac{x-X_i}{h_0})$ (equation 15 in the notebook). The power $h_0^3$ in 1D comes from $h_0^2 \cdot h_0$ (the $h_0^2$ from the Laplacian factor, the $h_0$ from the kernel normalization $(2\pi h_0^2)^{-1/2}$). In $d$-D, the normalization is $(2\pi h_0^2)^{-d/2}$, which is already inside $K_{h_0}$.

### Step 4.2: The roughness (squaring the sum)

(equation 12)

$$\hat{\Psi}(h_0) = \int [\nabla^2 \hat{f}_{h_0}(\mathbf{x})]^2 d\mathbf{x} = \frac{1}{n^2 h_0^4} \int \left[\sum_{i=1}^n K_{h_0}(\mathbf{x} - X_i) \cdot L_{h_0}(\mathbf{x} - X_i)\right]^2 d\mathbf{x}$$

### Step 4.3: Expanding the square (same as equation 19 in the notebook)

Just as in 1D where $[\sum_i x_i]^2 = \sum_i x_i^2 + \sum_{i \neq j} x_i x_j$ (the matrix analogy from the notebook), we expand:

(equation 13)

$$\hat{\Psi}(h_0) = \frac{1}{n^2 h_0^4} \sum_{i=1}^n \sum_{j=1}^n \underbrace{\int K_{h_0}(\mathbf{x} - X_i) \cdot L_{h_0}(\mathbf{x} - X_i) \cdot K_{h_0}(\mathbf{x} - X_j) \cdot L_{h_0}(\mathbf{x} - X_j) \, d\mathbf{x}}_{I_{ij}}$$

This is the multivariate analog of equation 22 in the notebook. Each $I_{ij}$ term is the integral that needs to be computed. In the notebook, this was the expression inside the `outer()` function in R — applied to all $(i,j)$ pairs of data points.

---

## 5. Combining the Exponential Terms (Completing the Square)
<a id="5-completing-square"></a>

This is the critical step that made the 1D derivation tractable (equations 26–30 in the notebook). We combine the two Gaussian exponentials and complete the square.

### Step 5.1: Write out the integral $I_{ij}$

(equation 14)

$$I_{ij} = \int K_{h_0}(\mathbf{x} - X_i) \cdot K_{h_0}(\mathbf{x} - X_j) \cdot L_{h_0}(\mathbf{x} - X_i) \cdot L_{h_0}(\mathbf{x} - X_j) \, d\mathbf{x}$$

Substituting the kernel formula $K_{h_0}(\mathbf{t}) = \frac{1}{(2\pi h_0^2)^{d/2}} \exp(-\|\mathbf{t}\|^2/(2h_0^2))$:

(equation 15)

$$I_{ij} = \frac{1}{(2\pi h_0^2)^d} \int \exp\left(-\frac{\|\mathbf{x} - X_i\|^2 + \|\mathbf{x} - X_j\|^2}{2h_0^2}\right) \cdot L_{h_0}(\mathbf{x} - X_i) \cdot L_{h_0}(\mathbf{x} - X_j) \, d\mathbf{x}$$

This is the $d$-dimensional analog of equation 26 in the notebook, where the two $\exp$ terms were combined.

### Step 5.2: Complete the square in the exponent

We need to simplify $\|\mathbf{x} - X_i\|^2 + \|\mathbf{x} - X_j\|^2$. Expanding:

(equation 16)

$$\|\mathbf{x} - X_i\|^2 + \|\mathbf{x} - X_j\|^2 = 2\|\mathbf{x}\|^2 - 2\mathbf{x}^T(X_i + X_j) + \|X_i\|^2 + \|X_j\|^2$$

To complete the square, write this as $2\|\mathbf{x} - \boldsymbol{\mu}\|^2 + \text{constant}$, where $\boldsymbol{\mu} = \frac{X_i + X_j}{2}$:

(equation 17)

$$= 2\left\|\mathbf{x} - \frac{X_i + X_j}{2}\right\|^2 + \frac{\|X_i - X_j\|^2}{2}$$

**Verification of equation 17:** Expand $2\|\mathbf{x} - \frac{X_i+X_j}{2}\|^2 = 2\|\mathbf{x}\|^2 - 2\mathbf{x}^T(X_i+X_j) + \frac{\|X_i+X_j\|^2}{2}$. And $\frac{\|X_i-X_j\|^2}{2} = \frac{\|X_i\|^2 - 2X_i^TX_j + \|X_j\|^2}{2}$. Adding: $2\|\mathbf{x}\|^2 - 2\mathbf{x}^T(X_i+X_j) + \frac{\|X_i\|^2 + 2X_i^TX_j + \|X_j\|^2}{2} + \frac{\|X_i\|^2 - 2X_i^TX_j + \|X_j\|^2}{2} = 2\|\mathbf{x}\|^2 - 2\mathbf{x}^T(X_i+X_j) + \|X_i\|^2 + \|X_j\|^2$. ✓

This is the exact same "completing the square" technique from equations 27-28 in the notebook, but now with vectors instead of scalars. In the notebook, the result was $(x - \frac{x_i + x_j}{2})^2$ plus a constant involving $x_i^2 + x_j^2$. Here it's $\|\mathbf{x} - \frac{X_i + X_j}{2}\|^2$ plus a constant involving $\|X_i - X_j\|^2$.

### Step 5.3: Substitute back into the exponential

(equation 18)

$$\exp\left(-\frac{\|\mathbf{x} - X_i\|^2 + \|\mathbf{x} - X_j\|^2}{2h_0^2}\right) = \exp\left(-\frac{1}{h_0^2}\left\|\mathbf{x} - \frac{X_i + X_j}{2}\right\|^2\right) \cdot \exp\left(-\frac{\|X_i - X_j\|^2}{4h_0^2}\right)$$

### Step 5.4: Separate the constant (same as equation 29 in the notebook)

Define $\boldsymbol{\mu}_{ij} = \frac{X_i + X_j}{2}$ and let:

(equation 19)

$$B = \frac{1}{(2\pi h_0^2)^d} \cdot \exp\left(-\frac{\|X_i - X_j\|^2}{4h_0^2}\right)$$

This is the constant that gets pulled out of the integral (just like the term $B$ in equation 29 of the notebook). Then:

(equation 20)

$$I_{ij} = B \cdot \underbrace{\int \exp\left(-\frac{1}{h_0^2}\left\|\mathbf{x} - \boldsymbol{\mu}_{ij}\right\|^2\right) \cdot L_{h_0}(\mathbf{x} - X_i) \cdot L_{h_0}(\mathbf{x} - X_j) \, d\mathbf{x}}_C$$

This is the $d$-dimensional version of equation 29's structure: $I_{ij} = B \cdot C$.

---

## 6. Recognizing the Expectation
<a id="6-expectation"></a>

This step mirrors equations 30-31 in the notebook, where the integral was recognized as an expectation with respect to a Normal distribution.

### Step 6.1: Rewrite $C$ to look like an expectation

The exponential $\exp\left(-\frac{1}{h_0^2}\|\mathbf{x} - \boldsymbol{\mu}_{ij}\|^2\right)$ looks like the kernel of a Gaussian with mean $\boldsymbol{\mu}_{ij}$ and covariance $\frac{h_0^2}{2}I_d$. Specifically, for $\mathbf{Z} \sim \mathcal{N}(\boldsymbol{\mu}_{ij}, \frac{h_0^2}{2}I_d)$:

$$p(\mathbf{z}) = \frac{1}{(2\pi \cdot h_0^2/2)^{d/2}} \exp\left(-\frac{\|\mathbf{z} - \boldsymbol{\mu}_{ij}\|^2}{2 \cdot h_0^2/2}\right) = \frac{1}{(\pi h_0^2)^{d/2}} \exp\left(-\frac{\|\mathbf{z} - \boldsymbol{\mu}_{ij}\|^2}{h_0^2}\right)$$

So the exponential in $C$ equals $(\pi h_0^2)^{d/2} \cdot p(\mathbf{x})$. This means:

(equation 21)

$$C = (\pi h_0^2)^{d/2} \int p(\mathbf{x}) \cdot L_{h_0}(\mathbf{x} - X_i) \cdot L_{h_0}(\mathbf{x} - X_j) \, d\mathbf{x} = (\pi h_0^2)^{d/2} \cdot \mathbb{E}_{\mathbf{Z}}\left[L_{h_0}(\mathbf{Z} - X_i) \cdot L_{h_0}(\mathbf{Z} - X_j)\right]$$

where $\mathbf{Z} \sim \mathcal{N}\left(\frac{X_i + X_j}{2}, \frac{h_0^2}{2}I_d\right)$.

This is exactly the same trick as in equation 31 of the notebook (the $C'$ step), where we recognized the integral as $E_g[A]$ with $g \sim Normal(\frac{x_i+x_j}{2}, \frac{h_0^2}{2})$. The only difference is that now $g$ is a **multivariate** Normal and $A$ involves **norms** instead of scalars.

### Step 6.2: Putting $I_{ij}$ together

(equation 22)

$$I_{ij} = B \cdot (\pi h_0^2)^{d/2} \cdot \mathbb{E}_{\mathbf{Z}}\left[L_{h_0}(\mathbf{Z} - X_i) \cdot L_{h_0}(\mathbf{Z} - X_j)\right]$$

Substituting $B$ from equation 19:

$$I_{ij} = \frac{(\pi h_0^2)^{d/2}}{(2\pi h_0^2)^d} \cdot \exp\left(-\frac{\|X_i - X_j\|^2}{4h_0^2}\right) \cdot \mathbb{E}_{\mathbf{Z}}\left[L_{h_0}(\mathbf{Z} - X_i) \cdot L_{h_0}(\mathbf{Z} - X_j)\right]$$

Simplify the constant: $\frac{(\pi h_0^2)^{d/2}}{(2\pi h_0^2)^d} = \frac{1}{2^d (\pi h_0^2)^{d/2}} = \frac{1}{(4\pi h_0^2)^{d/2}}$

(equation 23)

$$\boxed{I_{ij} = \frac{1}{(4\pi h_0^2)^{d/2}} \cdot \exp\left(-\frac{\|X_i - X_j\|^2}{4h_0^2}\right) \cdot \mathbb{E}_{\mathbf{Z}}\left[L_{h_0}(\mathbf{Z} - X_i) \cdot L_{h_0}(\mathbf{Z} - X_j)\right]}$$

where $\mathbf{Z} \sim \mathcal{N}\left(\frac{X_i + X_j}{2}, \frac{h_0^2}{2}I_d\right)$.

The problem is now reduced to computing the expectation of the product $L_{h_0}(\mathbf{Z} - X_i) \cdot L_{h_0}(\mathbf{Z} - X_j)$. In the notebook, this was "computing $E_g[A]$ where $A$ is a polynomial of degree 4." We will now do the same, but the polynomial involves **norms** rather than scalar powers.

---

## 7. Computing the Polynomial Expectation (Moments)
<a id="7-moments"></a>

In the 1D notebook, $A = (1 - z_i^2)(1 - z_j^2)$ was a degree-4 polynomial in $x$, and the expectation $E_g[A]$ was computed using the first four moments of the Normal distribution $g$. In $d$-D, the analog is:

$$A_d = L_{h_0}(\mathbf{Z} - X_i) \cdot L_{h_0}(\mathbf{Z} - X_j) = \left(\frac{\|\mathbf{Z} - X_i\|^2}{h_0^2} - d\right)\left(\frac{\|\mathbf{Z} - X_j\|^2}{h_0^2} - d\right)$$

This is a polynomial in $\|\mathbf{Z} - X_i\|^2$ and $\|\mathbf{Z} - X_j\|^2$. The key insight is that since $\mathbf{Z}$ has a known distribution, we can compute expectations of powers of these norms using **moments of quadratic forms under Gaussian distributions**.

### Step 7.1: Define variables

Let $\mathbf{U} = \mathbf{Z} - X_i$ and $\mathbf{V} = \mathbf{Z} - X_j$.

Since $\mathbf{Z} \sim \mathcal{N}\left(\frac{X_i + X_j}{2}, \frac{h_0^2}{2}I_d\right)$:

- $\mathbf{U} = \mathbf{Z} - X_i$ has mean $\frac{X_i + X_j}{2} - X_i = \frac{X_j - X_i}{2}$
- $\mathbf{V} = \mathbf{Z} - X_j$ has mean $\frac{X_i + X_j}{2} - X_j = \frac{X_i - X_j}{2}$
- Both have covariance $\frac{h_0^2}{2}I_d$

Define $\boldsymbol{\delta} = X_j - X_i$ (the pairwise difference). Then:

(equation 24)

$$\mathbf{U} \sim \mathcal{N}\left(\frac{\boldsymbol{\delta}}{2}, \frac{h_0^2}{2}I_d\right), \qquad \mathbf{V} \sim \mathcal{N}\left(-\frac{\boldsymbol{\delta}}{2}, \frac{h_0^2}{2}I_d\right)$$

Also note: $\mathbf{V} = \mathbf{U} - \boldsymbol{\delta}$ (they are not independent — both are functions of the same $\mathbf{Z}$).

### Step 7.2: Expand the product $A_d$

(equation 25)

$$A_d = \left(\frac{\|\mathbf{U}\|^2}{h_0^2} - d\right)\left(\frac{\|\mathbf{V}\|^2}{h_0^2} - d\right) = \frac{\|\mathbf{U}\|^2 \|\mathbf{V}\|^2}{h_0^4} - \frac{d\|\mathbf{U}\|^2}{h_0^2} - \frac{d\|\mathbf{V}\|^2}{h_0^2} + d^2$$

So the expectation is:

(equation 26)

$$\mathbb{E}[A_d] = \frac{\mathbb{E}[\|\mathbf{U}\|^2\|\mathbf{V}\|^2]}{h_0^4} - \frac{d \cdot \mathbb{E}[\|\mathbf{U}\|^2]}{h_0^2} - \frac{d \cdot \mathbb{E}[\|\mathbf{V}\|^2]}{h_0^2} + d^2$$

We need three types of moments. Compare with the 1D case: the notebook needed $E[x], E[x^2], E[x^3], E[x^4]$ — the first four moments. Here we need $\mathbb{E}[\|\mathbf{U}\|^2]$, $\mathbb{E}[\|\mathbf{V}\|^2]$, and $\mathbb{E}[\|\mathbf{U}\|^2\|\mathbf{V}\|^2]$.

### Step 7.3: Computing $\mathbb{E}[\|\mathbf{U}\|^2]$ and $\mathbb{E}[\|\mathbf{V}\|^2]$

For any random vector $\mathbf{W} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma^2 I_d)$, we have:

$$\mathbb{E}[\|\mathbf{W}\|^2] = \mathbb{E}[W_1^2 + W_2^2 + \cdots + W_d^2] = \sum_{k=1}^d (\mu_k^2 + \sigma^2) = \|\boldsymbol{\mu}\|^2 + d\sigma^2$$

This uses the fact that for each component $W_k \sim \mathcal{N}(\mu_k, \sigma^2)$, $\mathbb{E}[W_k^2] = \text{Var}(W_k) + (\mathbb{E}[W_k])^2 = \sigma^2 + \mu_k^2$.

For $\mathbf{U}$: $\boldsymbol{\mu}_U = \boldsymbol{\delta}/2$, $\sigma^2 = h_0^2/2$

(equation 27)

$$\mathbb{E}[\|\mathbf{U}\|^2] = \frac{\|\boldsymbol{\delta}\|^2}{4} + \frac{dh_0^2}{2}$$

For $\mathbf{V}$: $\boldsymbol{\mu}_V = -\boldsymbol{\delta}/2$, $\sigma^2 = h_0^2/2$

(equation 28)

$$\mathbb{E}[\|\mathbf{V}\|^2] = \frac{\|\boldsymbol{\delta}\|^2}{4} + \frac{dh_0^2}{2}$$

These are equal (by symmetry: $\|-\boldsymbol{\delta}/2\|^2 = \|\boldsymbol{\delta}/2\|^2$).

### Step 7.4: Computing $\mathbb{E}[\|\mathbf{U}\|^2\|\mathbf{V}\|^2]$

This is the harder term — analogous to needing $E[x^4]$ in the 1D case. Since $\mathbf{V} = \mathbf{U} - \boldsymbol{\delta}$:

(equation 29)

$$\|\mathbf{V}\|^2 = \|\mathbf{U} - \boldsymbol{\delta}\|^2 = \|\mathbf{U}\|^2 - 2\mathbf{U}^T\boldsymbol{\delta} + \|\boldsymbol{\delta}\|^2$$

Therefore:

(equation 30)

$$\|\mathbf{U}\|^2\|\mathbf{V}\|^2 = \|\mathbf{U}\|^2(\|\mathbf{U}\|^2 - 2\mathbf{U}^T\boldsymbol{\delta} + \|\boldsymbol{\delta}\|^2) = \|\mathbf{U}\|^4 - 2\|\mathbf{U}\|^2(\mathbf{U}^T\boldsymbol{\delta}) + \|\boldsymbol{\delta}\|^2\|\mathbf{U}\|^2$$

We need: (a) $\mathbb{E}[\|\mathbf{U}\|^4]$, (b) $\mathbb{E}[\|\mathbf{U}\|^2(\mathbf{U}^T\boldsymbol{\delta})]$, and (c) $\mathbb{E}[\|\mathbf{U}\|^2]$ (already computed).

### Step 7.5: Computing $\mathbb{E}[\|\mathbf{U}\|^4]$

Write $\mathbf{U} = \boldsymbol{\mu} + \sigma\mathbf{W}$ where $\boldsymbol{\mu} = \boldsymbol{\delta}/2$, $\sigma^2 = h_0^2/2$, and $\mathbf{W} \sim \mathcal{N}(0, I_d)$:

$$\|\mathbf{U}\|^2 = \|\boldsymbol{\mu}\|^2 + 2\sigma(\boldsymbol{\mu}^T\mathbf{W}) + \sigma^2\|\mathbf{W}\|^2$$

Square it:

(equation 31)

$$\|\mathbf{U}\|^4 = \|\boldsymbol{\mu}\|^4 + 4\sigma^2(\boldsymbol{\mu}^T\mathbf{W})^2 + \sigma^4\|\mathbf{W}\|^4 + 4\sigma\|\boldsymbol{\mu}\|^2(\boldsymbol{\mu}^T\mathbf{W}) + 2\sigma^2\|\boldsymbol{\mu}\|^2\|\mathbf{W}\|^2 + 4\sigma^3(\boldsymbol{\mu}^T\mathbf{W})\|\mathbf{W}\|^2$$

Now take expectations term by term. For $\mathbf{W} \sim \mathcal{N}(0, I_d)$:

- $\mathbb{E}[\mathbf{W}] = \mathbf{0}$, so $\mathbb{E}[\boldsymbol{\mu}^T\mathbf{W}] = 0$
- $\mathbb{E}[(\boldsymbol{\mu}^T\mathbf{W})^2] = \|\boldsymbol{\mu}\|^2$ (since $\boldsymbol{\mu}^T\mathbf{W} \sim \mathcal{N}(0, \|\boldsymbol{\mu}\|^2)$)
- $\mathbb{E}[\|\mathbf{W}\|^2] = d$ (sum of $d$ standard normals squared)
- $\mathbb{E}[\|\mathbf{W}\|^4] = d(d+2)$ (derived below)
- $\mathbb{E}[(\boldsymbol{\mu}^T\mathbf{W})\|\mathbf{W}\|^2] = 0$ (odd function of $\mathbf{W}$)

**Deriving $\mathbb{E}[\|\mathbf{W}\|^4]$:** We have $\|\mathbf{W}\|^4 = (\sum_k W_k^2)^2 = \sum_k W_k^4 + 2\sum_{k<l} W_k^2 W_l^2$.

- $\mathbb{E}[W_k^4] = 3$ (fourth moment of standard normal)
- $\mathbb{E}[W_k^2 W_l^2] = \mathbb{E}[W_k^2]\mathbb{E}[W_l^2] = 1$ (independent)
- Number of diagonal terms: $d$. Number of off-diagonal pairs: $\binom{d}{2} = d(d-1)/2$.

$$\mathbb{E}[\|\mathbf{W}\|^4] = 3d + 2 \cdot \frac{d(d-1)}{2} = 3d + d(d-1) = d^2 + 2d = d(d+2)$$

When $d=1$: $\mathbb{E}[W^4] = 3$. ✓ (The fourth moment of the standard normal.)

**Deriving $\mathbb{E}[(\boldsymbol{\mu}^T\mathbf{W})\|\mathbf{W}\|^2] = 0$:** Decompose $\mathbf{W} = (\hat{\boldsymbol{\mu}}^T\mathbf{W})\hat{\boldsymbol{\mu}} + \mathbf{W}_\perp$ where $\hat{\boldsymbol{\mu}} = \boldsymbol{\mu}/\|\boldsymbol{\mu}\|$. Then $\hat{\boldsymbol{\mu}}^T\mathbf{W} \sim \mathcal{N}(0,1)$ is independent of $\mathbf{W}_\perp$. The expression $(\boldsymbol{\mu}^T\mathbf{W})\|\mathbf{W}\|^2 = \|\boldsymbol{\mu}\|(\hat{\boldsymbol{\mu}}^T\mathbf{W})((\hat{\boldsymbol{\mu}}^T\mathbf{W})^2 + \|\mathbf{W}_\perp\|^2)$. Since $\hat{\boldsymbol{\mu}}^T\mathbf{W}$ is a standard normal, its odd moments are zero: $\mathbb{E}[(\hat{\boldsymbol{\mu}}^T\mathbf{W})^3] = 0$ and $\mathbb{E}[\hat{\boldsymbol{\mu}}^T\mathbf{W}] = 0$. So the entire expression has expectation 0.

**Collecting terms in equation 31:**

(equation 32)

$$\mathbb{E}[\|\mathbf{U}\|^4] = \|\boldsymbol{\mu}\|^4 + 4\sigma^2\|\boldsymbol{\mu}\|^2 + \sigma^4 d(d+2) + 0 + 2\sigma^2\|\boldsymbol{\mu}\|^2 d + 0$$

$$= \|\boldsymbol{\mu}\|^4 + 2(d+2)\sigma^2\|\boldsymbol{\mu}\|^2 + d(d+2)\sigma^4$$

Substituting $\|\boldsymbol{\mu}\|^2 = \|\boldsymbol{\delta}\|^2/4$ and $\sigma^2 = h_0^2/2$:

(equation 33)

$$\mathbb{E}[\|\mathbf{U}\|^4] = \frac{\|\boldsymbol{\delta}\|^4}{16} + 2(d+2) \cdot \frac{h_0^2}{2} \cdot \frac{\|\boldsymbol{\delta}\|^2}{4} + d(d+2) \cdot \frac{h_0^4}{4}$$

$$= \frac{\|\boldsymbol{\delta}\|^4}{16} + \frac{(d+2)\|\boldsymbol{\delta}\|^2 h_0^2}{4} + \frac{d(d+2) h_0^4}{4}$$

### Step 7.6: Computing $\mathbb{E}[\|\mathbf{U}\|^2(\mathbf{U}^T\boldsymbol{\delta})]$

Again write $\mathbf{U} = \boldsymbol{\mu} + \sigma\mathbf{W}$ and expand:

$$\|\mathbf{U}\|^2(\mathbf{U}^T\boldsymbol{\delta}) = (\|\boldsymbol{\mu}\|^2 + 2\sigma\boldsymbol{\mu}^T\mathbf{W} + \sigma^2\|\mathbf{W}\|^2)(\boldsymbol{\mu}^T\boldsymbol{\delta} + \sigma\mathbf{W}^T\boldsymbol{\delta})$$

This expands to 6 terms. Taking expectations (using $\mathbb{E}[\mathbf{W}] = 0$, $\mathbb{E}[\mathbf{W}\mathbf{W}^T] = I_d$, and odd moments vanishing):

- $\|\boldsymbol{\mu}\|^2 \cdot \boldsymbol{\mu}^T\boldsymbol{\delta}$ → contributes $\|\boldsymbol{\mu}\|^2 \boldsymbol{\mu}^T\boldsymbol{\delta}$
- $\|\boldsymbol{\mu}\|^2 \cdot \sigma\mathbf{W}^T\boldsymbol{\delta}$ → expectation $= 0$
- $2\sigma(\boldsymbol{\mu}^T\mathbf{W}) \cdot \boldsymbol{\mu}^T\boldsymbol{\delta}$ → expectation $= 0$
- $2\sigma^2(\boldsymbol{\mu}^T\mathbf{W})(\mathbf{W}^T\boldsymbol{\delta})$ → $= 2\sigma^2 \boldsymbol{\mu}^T\mathbb{E}[\mathbf{W}\mathbf{W}^T]\boldsymbol{\delta} = 2\sigma^2\boldsymbol{\mu}^T\boldsymbol{\delta}$
- $\sigma^2\|\mathbf{W}\|^2 \cdot \boldsymbol{\mu}^T\boldsymbol{\delta}$ → $= \sigma^2 d \cdot \boldsymbol{\mu}^T\boldsymbol{\delta}$
- $\sigma^3\|\mathbf{W}\|^2(\mathbf{W}^T\boldsymbol{\delta})$ → expectation $= 0$ (odd in $\mathbf{W}$)

Collecting:

(equation 34)

$$\mathbb{E}[\|\mathbf{U}\|^2(\mathbf{U}^T\boldsymbol{\delta})] = (\boldsymbol{\mu}^T\boldsymbol{\delta})[\|\boldsymbol{\mu}\|^2 + (d+2)\sigma^2]$$

Now: $\boldsymbol{\mu} = \boldsymbol{\delta}/2$, so $\boldsymbol{\mu}^T\boldsymbol{\delta} = \|\boldsymbol{\delta}\|^2/2$ and $\|\boldsymbol{\mu}\|^2 = \|\boldsymbol{\delta}\|^2/4$:

(equation 35)

$$\mathbb{E}[\|\mathbf{U}\|^2(\mathbf{U}^T\boldsymbol{\delta})] = \frac{\|\boldsymbol{\delta}\|^2}{2}\left[\frac{\|\boldsymbol{\delta}\|^2}{4} + (d+2)\frac{h_0^2}{2}\right] = \frac{\|\boldsymbol{\delta}\|^4}{8} + \frac{(d+2)\|\boldsymbol{\delta}\|^2 h_0^2}{4}$$

### Step 7.7: Assembling $\mathbb{E}[\|\mathbf{U}\|^2\|\mathbf{V}\|^2]$

From equation 30:

$$\mathbb{E}[\|\mathbf{U}\|^2\|\mathbf{V}\|^2] = \mathbb{E}[\|\mathbf{U}\|^4] - 2\mathbb{E}[\|\mathbf{U}\|^2(\mathbf{U}^T\boldsymbol{\delta})] + \|\boldsymbol{\delta}\|^2\mathbb{E}[\|\mathbf{U}\|^2]$$

Substituting from equations 33, 35, and 27:

(equation 36)

$$= \left[\frac{\|\boldsymbol{\delta}\|^4}{16} + \frac{(d+2)\|\boldsymbol{\delta}\|^2 h_0^2}{4} + \frac{d(d+2)h_0^4}{4}\right] - 2\left[\frac{\|\boldsymbol{\delta}\|^4}{8} + \frac{(d+2)\|\boldsymbol{\delta}\|^2 h_0^2}{4}\right] + \|\boldsymbol{\delta}\|^2\left[\frac{\|\boldsymbol{\delta}\|^2}{4} + \frac{dh_0^2}{2}\right]$$

Collecting by powers of $\|\boldsymbol{\delta}\|$ and $h_0$:

**$\|\boldsymbol{\delta}\|^4$ terms:** $\frac{1}{16} - \frac{2}{8} + \frac{1}{4} = \frac{1}{16} - \frac{4}{16} + \frac{4}{16} = \frac{1}{16}$

**$\|\boldsymbol{\delta}\|^2 h_0^2$ terms:** $\frac{d+2}{4} - \frac{2(d+2)}{4} + \frac{d}{2} = \frac{d+2 - 2(d+2) + 2d}{4} = \frac{d - 2}{4}$

**$h_0^4$ terms:** $\frac{d(d+2)}{4}$ (only from first bracket)

(equation 37)

$$\boxed{\mathbb{E}[\|\mathbf{U}\|^2\|\mathbf{V}\|^2] = \frac{\|\boldsymbol{\delta}\|^4}{16} + \frac{(d-2)\|\boldsymbol{\delta}\|^2 h_0^2}{4} + \frac{d(d+2)h_0^4}{4}}$$

### Step 7.8: The final expectation $\mathbb{E}[A_d]$

From equation 26:

$$\mathbb{E}[A_d] = \frac{\mathbb{E}[\|\mathbf{U}\|^2\|\mathbf{V}\|^2]}{h_0^4} - \frac{d}{h_0^2}\mathbb{E}[\|\mathbf{U}\|^2] - \frac{d}{h_0^2}\mathbb{E}[\|\mathbf{V}\|^2] + d^2$$

Substituting:

(equation 38)

$$= \frac{1}{h_0^4}\left[\frac{\|\boldsymbol{\delta}\|^4}{16} + \frac{(d-2)\|\boldsymbol{\delta}\|^2 h_0^2}{4} + \frac{d(d+2)h_0^4}{4}\right] - \frac{2d}{h_0^2}\left[\frac{\|\boldsymbol{\delta}\|^2}{4} + \frac{dh_0^2}{2}\right] + d^2$$

Expand:

$$= \frac{\|\boldsymbol{\delta}\|^4}{16h_0^4} + \frac{(d-2)\|\boldsymbol{\delta}\|^2}{4h_0^2} + \frac{d(d+2)}{4} - \frac{d\|\boldsymbol{\delta}\|^2}{2h_0^2} - d^2 + d^2$$

Group the $\|\boldsymbol{\delta}\|^2/h_0^2$ terms: $\frac{d-2}{4} - \frac{d}{2} = \frac{d-2-2d}{4} = \frac{-(d+2)}{4}$

Group the constants: $\frac{d(d+2)}{4} - d^2 + d^2 = \frac{d(d+2)}{4}$

(equation 39)

$$\boxed{\mathbb{E}[A_d] = \frac{\|\boldsymbol{\delta}\|^4}{16h_0^4} - \frac{(d+2)\|\boldsymbol{\delta}\|^2}{4h_0^2} + \frac{d(d+2)}{4}}$$

### Step 7.9: Introducing scaled distance $r^2$

Define $r_{ij}^2 = \|\boldsymbol{\delta}\|^2/h_0^2 = \|X_i - X_j\|^2/h_0^2$. Then:

(equation 40)

$$\mathbb{E}[A_d] = \frac{r_{ij}^4}{16} - \frac{(d+2)r_{ij}^2}{4} + \frac{d(d+2)}{4} \equiv P_d(r_{ij}^2)$$

This is the **dimension-dependent polynomial** — the multivariate analog of the 4th-degree polynomial $A = (1-z_i^2)(1-z_j^2)$ from the notebook. Note how much simpler it is: only 3 terms instead of the full expansion!

### Step 7.10: Verification for $d = 1$

When $d = 1$, $r^2 = (X_i - X_j)^2/h_0^2 = \Delta^2/h_0^2$:

$$P_1(r^2) = \frac{r^4}{16} - \frac{3r^2}{4} + \frac{3}{4}$$

Let's check this against the 1D formula. In the notebook, $A = (1 - z_i^2)(1 - z_j^2)$ where $z_i = (x - X_i)/h_0$ and the expectation was taken over $x \sim \mathcal{N}(\frac{X_i+X_j}{2}, \frac{h_0^2}{2})$. Setting $U = (x-X_i)/h_0$ which has mean $\Delta/(2h_0)$ and variance $1/2$:

$\mathbb{E}[(1-U^2)(1-V^2)]$ where $V = U - \Delta/h_0$... this should give the same $P_1(r^2)$ when properly accounting for the sign convention (recall $L_h(t) = t^2/h^2 - 1 = -(1 - t^2/h^2)$, so $L_{h_0}(U \cdot h_0) = U^2 - 1 = -(1 - U^2)$, meaning $L \cdot L = (1-z_i^2)(1-z_j^2)$). ✓

---

## 8. Assembling the Final Formula
<a id="8-final"></a>

### Step 8.1: The pairwise integral $I_{ij}$

From equation 23 and equation 40:

(equation 41)

$$I_{ij} = \frac{1}{(4\pi h_0^2)^{d/2}} \cdot \exp\left(-\frac{r_{ij}^2}{4}\right) \cdot P_d(r_{ij}^2)$$

where $r_{ij}^2 = \|X_i - X_j\|^2/h_0^2$.

### Step 8.2: The full roughness estimate

From equation 13:

(equation 42)

$$\hat{\Psi}(h_0) = \frac{1}{n^2 h_0^4} \sum_{i=1}^n \sum_{j=1}^n I_{ij} = \frac{1}{n^2 h_0^4 (4\pi h_0^2)^{d/2}} \sum_{i=1}^n \sum_{j=1}^n \exp\left(-\frac{r_{ij}^2}{4}\right) \cdot P_d(r_{ij}^2)$$

Simplifying the prefactor: $h_0^4 \cdot (4\pi h_0^2)^{d/2} = h_0^4 \cdot (4\pi)^{d/2} \cdot h_0^d = (4\pi)^{d/2} \cdot h_0^{d+4}$

(equation 43)

$$\boxed{\hat{\Psi}(h_0) = \frac{1}{n^2 (4\pi)^{d/2} h_0^{d+4}} \sum_{i=1}^n \sum_{j=1}^n \exp\left(-\frac{r_{ij}^2}{4}\right) \cdot P_d(r_{ij}^2)}$$

where:

$$P_d(t) = \frac{t^2}{16} - \frac{(d+2)t}{4} + \frac{d(d+2)}{4}$$

and $r_{ij}^2 = \|X_i - X_j\|^2 / h_0^2$.

This is the closed-form solution. Compare with the structure from the notebook: $R(\hat{f}'') = \frac{1}{n^2 h_0^6} \sum_{i,j} (\text{outer function applied to pairs})$. The outer function there was the complicated expression involving B_prime; here it is $\exp(-r^2/4) \cdot P_d(r^2)$ — a Gaussian weight times a simple polynomial.

---

## 9. The Optimal Bandwidth
<a id="9-optimal"></a>

### Step 9.1: Plug into the AMISE-optimal formula

From equation 5:

(equation 44)

$$h^* = \left(\frac{d \cdot R(K)}{n \cdot \hat{\Psi}(h_0)}\right)^{1/(d+4)}$$

where $R(K) = (4\pi)^{-d/2}$.

### Step 9.2: Simplification

(equation 45)

$$h^* = \left(\frac{d}{n \cdot \hat{\Psi}(h_0) \cdot (4\pi)^{d/2}}\right)^{1/(d+4)}$$

### Step 9.3: The pilot bandwidth

Using Silverman's rule in $d$ dimensions (the analog of equation 14 from the notebook):

(equation 46)

$$h_0 = \left(\frac{4}{n(d+2)}\right)^{1/(d+4)}$$

(This is for whitened data with unit covariance. For raw data, multiply by the standard deviation or whiten first.)

### Step 9.4: Complete algorithm

1. **Whiten** the data: compute sample covariance $\hat{\Sigma}$, transform $Y_i = \hat{\Sigma}^{-1/2}X_i$
2. **Compute pilot**: $h_0 = (4/(n(d+2)))^{1/(d+4)}$
3. **Compute pairwise distances**: $r_{ij}^2 = \|Y_i - Y_j\|^2/h_0^2$ for all pairs $(i,j)$
4. **Evaluate polynomial**: $P_d(r_{ij}^2) = \frac{r_{ij}^4}{16} - \frac{(d+2)r_{ij}^2}{4} + \frac{d(d+2)}{4}$
5. **Sum with Gaussian weights**: $S = \sum_{i,j} \exp(-r_{ij}^2/4) \cdot P_d(r_{ij}^2)$
6. **Roughness**: $\hat{\Psi} = S / (n^2 (4\pi)^{d/2} h_0^{d+4})$
7. **Optimal bandwidth**: $h^* = (d / (n \hat{\Psi} (4\pi)^{d/2}))^{1/(d+4)}$
8. **Final bandwidth matrix**: $H = (h^*)^2 \hat{\Sigma}$ (the "factor" for scipy is $h^*$)

---

## 10. Verification: Recovering the 1D Case
<a id="10-verification"></a>

Let's verify step by step that when $d = 1$, the formulas reduce to the notebook's results.

### The polynomial

$P_1(t) = \frac{t^2}{16} - \frac{3t}{4} + \frac{3}{4}$

### The roughness

$\hat{\Psi}(h_0) = \frac{1}{n^2 \cdot 2\sqrt{\pi} \cdot h_0^5} \sum_{i,j} \exp\left(-\frac{(X_i-X_j)^2}{4h_0^2}\right) \cdot P_1\left(\frac{(X_i-X_j)^2}{h_0^2}\right)$

since $(4\pi)^{1/2} = 2\sqrt{\pi}$ and $d + 4 = 5$. ✓

### The optimal bandwidth

$h^* = \left(\frac{1}{n \cdot \hat{\Psi} \cdot 2\sqrt{\pi}}\right)^{1/5} = \left(\frac{R(K)}{n\hat{\Psi}}\right)^{1/5}$

since $R(K)_{d=1} = (4\pi)^{-1/2} = \frac{1}{2\sqrt{\pi}}$ and the factor $d = 1$ in the numerator cancels with the way the exponent works. ✓

### The pilot

$h_0 = (4/(3n))^{1/5} = (4/(n \cdot 3))^{1/5}$, which is Silverman's rule (equation 14). ✓

---

## 11. Summary
<a id="11-summary"></a>

### The derivation pipeline (identical in 1D and $d$-D)

| Step | 1D (notebook) | $d$-D (this paper) |
|------|---------------|---------------------|
| 1. AMISE | $\frac{R(K)}{nh} + \frac{h^4}{4}R(f'')$ | $\frac{R(K)}{nh^d} + \frac{h^4}{4}\Psi$ |
| 2. Optimal $h$ | $(R(K)/(n\Psi))^{1/5}$ | $(dR(K)/(n\Psi))^{1/(d+4)}$ |
| 3. Expand roughness | Double sum over pairs | Double sum over pairs |
| 4. Combine exponentials | Complete the square in $x$ | Complete the square in $\mathbf{x}$ |
| 5. Recognize expectation | $E_g[A]$ where $g \sim N(\frac{x_i+x_j}{2}, \frac{h_0^2}{2})$ | $E_{\mathbf{Z}}[A_d]$ where $\mathbf{Z} \sim N(\frac{X_i+X_j}{2}, \frac{h_0^2}{2}I_d)$ |
| 6. Compute moments | $E[x^k]$ for $k=1,\ldots,4$ | $E[\|\mathbf{U}\|^{2k}]$ and cross terms |
| 7. Final formula | 16-term polynomial | 3-term polynomial $P_d(r^2)$ |

### The key results

**Roughness estimate** (equation 43):
$$\hat{\Psi}(h_0) = \frac{1}{n^2 (4\pi)^{d/2} h_0^{d+4}} \sum_{i,j} e^{-r_{ij}^2/4} \cdot P_d(r_{ij}^2)$$

**Polynomial** (equation 40):
$$P_d(t) = \frac{t^2}{16} - \frac{(d+2)t}{4} + \frac{d(d+2)}{4}$$

**Optimal bandwidth** (equation 45):
$$h^* = \left(\frac{d}{n \cdot \hat{\Psi}(h_0) \cdot (4\pi)^{d/2}}\right)^{1/(d+4)}$$

### What's gained

The entire derivation follows the **exact same pipeline** as the 1D case in the notebook:
1. Set up the AMISE
2. Expand the roughness into pairwise terms
3. Combine exponentials by completing the square
4. Recognize the integral as an expectation
5. Compute moments to get a closed form

The only differences are:
- Scalars become vectors ($x \to \mathbf{x}$, $(x-X_i)^2 \to \|\mathbf{x} - X_i\|^2$)
- The polynomial $A$ involves norms instead of individual coordinates
- Moments use $\mathbb{E}[\|\mathbf{W}\|^4] = d(d+2)$ instead of $\mathbb{E}[W^4] = 3$

The result is actually **simpler** in the general case because working with norms avoids the 16-term coordinate-specific expansion that appeared in the 1D implementation.
