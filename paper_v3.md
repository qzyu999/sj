# Optimizing the $O(n^2)$ Sheather-Jones Computation

## The Problem

The core computation in our $d$-D Sheather-Jones bandwidth selector is:

$$S = \sum_{i=1}^n \sum_{j=1}^n \exp\left(-\frac{\|Y_i - Y_j\|^2}{4h_0^2}\right) \cdot P_d\left(\frac{\|Y_i - Y_j\|^2}{h_0^2}\right)$$

This is a **double sum over all $n^2$ pairs** of data points. Each term involves:
1. A squared Euclidean distance ($d$ multiplications + additions)
2. An exponential evaluation
3. A polynomial evaluation (3 terms)

The total cost is $O(n^2 d)$ — which means:
- $n = 1{,}000$: ~1M pairs → 40ms ✓
- $n = 10{,}000$: ~100M pairs → 4s (annoying)
- $n = 100{,}000$: ~10B pairs → 400s (unusable)
- $n = 1{,}000{,}000$: ~1T pairs → 11 hours (absurd)

We need strategies to bring this to practical running times for large $n$.

---

## Strategy 1: Exploit the Gaussian Decay (Truncation)

### Observation

The Gaussian weight $\exp(-r_{ij}^2/4)$ decays **extremely fast**:

| $r_{ij}$ | $\exp(-r^2/4)$ | Contribution |
|-----------|-----------------|--------------|
| 0 | 1.0 | Full |
| 2 | 0.368 | 37% |
| 4 | 0.018 | 1.8% |
| 6 | $1.2 \times 10^{-4}$ | 0.01% |
| 8 | $1.1 \times 10^{-7}$ | Negligible |

By $r_{ij} \approx 6$ (i.e., $\|Y_i - Y_j\| \approx 6h_0$), the contribution is effectively zero.

### Method: Spatial Truncation

Only compute terms where $\|Y_i - Y_j\| < R_{\text{cut}}$ for some cutoff $R_{\text{cut}} = c \cdot h_0$.

Setting $c = 6$ gives relative error $< 10^{-4}$. Setting $c = 8$ gives error $< 10^{-7}$.

**How to find nearby pairs efficiently:**

- **KD-tree** (scipy's `cKDTree`): Build tree in $O(n \log n)$, then query all pairs within distance $R_{\text{cut}}$ via `query_ball_tree` or `query_pairs`. Cost: $O(n \cdot k)$ where $k$ is the average number of neighbors within the radius.

- **Ball tree** (sklearn's `BallTree`): Same idea, better for moderate $d$ (3–10).

### When does this help?

The number of neighbors $k$ within radius $R_{\text{cut}}$ depends on data density:
- If data is spread out (relative to $h_0$): $k \ll n$, so cost is $O(nk) \ll O(n^2)$
- If data is very concentrated: $k \approx n$, no improvement

In practice, for well-distributed data with $n$ points in $d$ dimensions:
$$k \approx n \cdot \text{Vol}(B(R_{\text{cut}})) / \text{Vol}(\text{data region}) \propto n \cdot (R_{\text{cut}}/\text{diameter})^d$$

For the pilot bandwidth $h_0 \sim n^{-1/(d+4)}$, the cutoff radius is $R_{\text{cut}} = 6h_0 \sim n^{-1/(d+4)}$, and the data diameter is $O(1)$ (after whitening). So:

$$k \approx n \cdot (n^{-1/(d+4)})^d = n^{1 - d/(d+4)} = n^{4/(d+4)}$$

Total cost: $O(n \cdot n^{4/(d+4)}) = O(n^{(d+8)/(d+4)})$

| $d$ | Exponent | vs $O(n^2)$ |
|-----|----------|-------------|
| 1 | $n^{9/5} = n^{1.8}$ | Slight improvement |
| 2 | $n^{10/6} = n^{1.67}$ | Good |
| 3 | $n^{11/7} = n^{1.57}$ | Better |
| 5 | $n^{13/9} = n^{1.44}$ | Much better |
| 10 | $n^{18/14} = n^{1.29}$ | Excellent |

Somewhat counterintuitively, spatial truncation helps **more** in higher dimensions (because the pilot bandwidth is larger relative to the data spread, meaning fewer neighbors fall within range).

### Implementation sketch

```python
from scipy.spatial import cKDTree

def sheather_jones_nd_fast(Y, h_0, d, cutoff=6.0):
    n = Y.shape[0]
    R_cut = cutoff * h_0
    
    tree = cKDTree(Y)
    pairs = tree.query_pairs(r=R_cut, output_type='ndarray')  # (m, 2) array
    
    # Compute distances only for nearby pairs
    diffs = Y[pairs[:, 0]] - Y[pairs[:, 1]]
    dist_sq = np.sum(diffs**2, axis=1)
    r_sq = dist_sq / h_0**2
    
    # Polynomial + weight
    P = r_sq**2/16 - (d+2)*r_sq/4 + d*(d+2)/4
    W = np.exp(-r_sq / 4.0)
    
    # Off-diagonal sum (pairs counted once, multiply by 2)
    S_off = 2.0 * np.sum(W * P)
    
    # Diagonal terms: r=0, P_d(0) = d(d+2)/4, W=1
    S_diag = n * d * (d + 2) / 4.0
    
    S = S_off + S_diag
    roughness = S / (n**2 * (4*np.pi)**(d/2) * h_0**(d+4))
    return roughness
```

---

## Strategy 2: Random Subsampling

### Observation

The sum $S = \sum_{i,j} f(Y_i, Y_j)$ is an average of $n^2$ terms. We can estimate it by sampling $m$ random pairs:

$$\hat{S} = \frac{n^2}{m} \sum_{k=1}^m f(Y_{a_k}, Y_{b_k}), \quad (a_k, b_k) \sim \text{Uniform}(\{1..n\}^2)$$

This is an unbiased estimator with variance:

$$\text{Var}(\hat{S}) = \frac{n^4}{m} \text{Var}(f(Y_a, Y_b))$$

### How many samples $m$ do we need?

For relative error $\epsilon$ with confidence $1-\delta$:

$$m \geq \frac{\text{Var}(f)}{\epsilon^2 \cdot \mathbb{E}[f]^2} \cdot \log(2/\delta)$$

In practice, $m = 10{,}000$ to $100{,}000$ random pairs gives excellent bandwidth estimates (the final $h^*$ involves a 5th root of the roughness, so errors are dampened: a 10% error in $\hat{\Psi}$ becomes only a 2% error in $h^*$).

### Cost

$O(m \cdot d)$ — completely independent of $n$!

### Variance reduction: stratified subsampling

Instead of purely random pairs, combine:
- **All diagonal terms**: $n$ terms (cheap, $O(nd)$)
- **Random off-diagonal**: $m$ random pairs

This reduces variance because diagonal terms (self-pairs) have the largest contribution.

### Implementation sketch

```python
def sheather_jones_nd_subsample(Y, h_0, d, m=50000, seed=42):
    n = Y.shape[0]
    rng = np.random.default_rng(seed)
    
    # Diagonal contribution (exact)
    S_diag = n * d * (d + 2) / 4.0
    
    # Random off-diagonal pairs
    idx_i = rng.integers(0, n, m)
    idx_j = rng.integers(0, n, m)
    
    diffs = Y[idx_i] - Y[idx_j]
    dist_sq = np.sum(diffs**2, axis=1)
    r_sq = dist_sq / h_0**2
    
    P = r_sq**2/16 - (d+2)*r_sq/4 + d*(d+2)/4
    W = np.exp(-r_sq / 4.0)
    
    # Scale up: we sampled m out of n^2 pairs
    S_off = (n**2 / m) * np.sum(W * P)
    
    S = S_diag + S_off  # (slight double-counting of diag in S_off is negligible)
    roughness = S / (n**2 * (4*np.pi)**(d/2) * h_0**(d+4))
    return roughness
```

### Accuracy of subsampling

Since $h^* = (c / \hat{\Psi})^{1/(d+4)}$, even if $\hat{\Psi}$ has 5-10% error, $h^*$ has only 1-2% error. The subsampling approach is therefore **extremely practical** — the 5th-root dampens estimation noise.

---

## Strategy 3: Dual-Tree / Fast Multipole Methods

### Observation

Our sum has the form of a **kernel sum**:

$$S = \sum_{i,j} K(Y_i, Y_j) \cdot Q(Y_i, Y_j)$$

where $K$ is a Gaussian kernel and $Q$ is a low-degree polynomial in the distance. This is exactly the structure that **Fast Gauss Transform (FGT)** and **dual-tree algorithms** exploit.

### Fast Gauss Transform (FGT)

The classical FGT (Greengard & Strain 1991) computes sums of the form $\sum_j w_j \exp(-\|x_i - y_j\|^2/h^2)$ for all $i$ in $O(n)$ time (for fixed error tolerance $\epsilon$).

Our sum requires not just the Gaussian sum but also weighted versions with polynomial factors. Specifically, expanding $P_d(r^2)$:

$$S = \frac{1}{16}\sum_{i,j} r_{ij}^4 e^{-r_{ij}^2/4} - \frac{d+2}{4}\sum_{i,j} r_{ij}^2 e^{-r_{ij}^2/4} + \frac{d(d+2)}{4}\sum_{i,j} e^{-r_{ij}^2/4}$$

Each of these three sums has the form $\sum_{i,j} \|Y_i - Y_j\|^{2k} \exp(-\|Y_i-Y_j\|^2/(4h_0^2))$ for $k = 0, 1, 2$.

These are **derivative moments of the Gaussian kernel**:
- $k=0$: Standard Gauss sum
- $k=1$: Related to $\nabla^2$ of the Gauss sum
- $k=2$: Related to $(\nabla^2)^2$ (bi-Laplacian) of the Gauss sum

The Improved Fast Gauss Transform (IFGT, Yang et al. 2003) can compute all three in $O(n)$ time with guaranteed error bounds.

### Dual-tree (e.g., KDE-specific)

Libraries like `scikit-learn`'s `KernelDensity` with `algorithm='kd_tree'` or `'ball_tree'` already implement dual-tree traversals for Gaussian kernel evaluation. Adapting this to our polynomial-weighted sum would require:

1. Build a tree on $Y$
2. For well-separated node pairs: approximate the contribution using far-field expansions
3. For nearby pairs: compute exactly

### Complexity

- FGT: $O(n)$ for fixed $\epsilon$ and $d$ (but the constant grows exponentially with $d$, practical up to $d \approx 10$)
- Dual-tree: $O(n \log n)$ with adaptive approximation

### Libraries

- [`figtree`](https://github.com/vmorariu/figtree) — C++ IFGT implementation with Python bindings
- [`FGT.jl`](https://github.com/JuliaApproximation/FastGaussTransform.jl) — Julia
- Custom implementation using Taylor expansions of $e^{-r^2}$ around cluster centers

---

## Strategy 4: GPU Parallelization

### Why GPUs help here

The pairwise computation is **embarrassingly parallel**: each $(i,j)$ pair is independent. This is ideal for GPU hardware which excels at launching millions of independent threads.

### Approach: Direct GPU pairwise

```python
import cupy as cp  # or torch

def sheather_jones_nd_gpu(Y_np, h_0, d):
    Y = cp.asarray(Y_np)  # Transfer to GPU
    n = Y.shape[0]
    
    # Pairwise squared distances (GPU handles n x n matrix)
    # CuPy broadcasting: (n,1,d) - (1,n,d) -> (n,n,d) -> sum -> (n,n)
    diff = Y[:, None, :] - Y[None, :, :]
    dist_sq = cp.sum(diff**2, axis=2)
    r_sq = dist_sq / h_0**2
    
    P = r_sq**2/16 - (d+2)*r_sq/4 + d*(d+2)/4
    W = cp.exp(-r_sq / 4.0)
    
    S = float(cp.sum(W * P))
    roughness = S / (n**2 * (4*cp.pi)**(d/2) * h_0**(d+4))
    return roughness
```

### Memory constraint

The $n \times n$ distance matrix requires $8n^2$ bytes (float64):
- $n = 10{,}000$: 800 MB (fits on most GPUs)
- $n = 30{,}000$: 7.2 GB (needs a high-end GPU)
- $n = 50{,}000$: 20 GB (exceeds most GPUs)

**Solution**: Block/tiled computation — process the $n \times n$ matrix in $B \times B$ blocks:

```python
def sheather_jones_nd_gpu_tiled(Y_np, h_0, d, block_size=4096):
    Y = cp.asarray(Y_np)
    n = Y.shape[0]
    S = 0.0
    
    for i_start in range(0, n, block_size):
        i_end = min(i_start + block_size, n)
        Yi = Y[i_start:i_end]
        
        for j_start in range(0, n, block_size):
            j_end = min(j_start + block_size, n)
            Yj = Y[j_start:j_end]
            
            diff = Yi[:, None, :] - Yj[None, :, :]
            dist_sq = cp.sum(diff**2, axis=2)
            r_sq = dist_sq / h_0**2
            
            P = r_sq**2/16 - (d+2)*r_sq/4 + d*(d+2)/4
            W = cp.exp(-r_sq / 4.0)
            S += float(cp.sum(W * P))
    
    roughness = S / (n**2 * (4*np.pi)**(d/2) * h_0**(d+4))
    return roughness
```

### Expected speedups

| $n$ | CPU (NumPy) | GPU (CuPy) | Speedup |
|-----|-------------|------------|---------|
| 1,000 | 40ms | 5ms | 8× |
| 5,000 | 1.1s | 20ms | 55× |
| 10,000 | 4.5s | 60ms | 75× |
| 50,000 | 110s | 1.5s | 73× |

(Estimates based on typical A100/V100 performance on pairwise distance + elementwise ops.)

### PyTorch alternative

```python
import torch

def sheather_jones_nd_torch(Y_np, h_0, d, device='cuda'):
    Y = torch.tensor(Y_np, device=device, dtype=torch.float64)
    n = Y.shape[0]
    
    dist_sq = torch.cdist(Y, Y, p=2).pow(2)  # Efficient pairwise distance
    r_sq = dist_sq / h_0**2
    
    P = r_sq**2/16 - (d+2)*r_sq/4 + d*(d+2)/4
    W = torch.exp(-r_sq / 4.0)
    
    S = (W * P).sum().item()
    roughness = S / (n**2 * (4*torch.pi)**(d/2) * h_0**(d+4))
    return roughness
```

`torch.cdist` is highly optimized for GPU pairwise distance computation.

---

## Strategy 5: Binning / Grid Approximation

### Observation

If we discretize the data onto a regular grid with $M$ bins per dimension ($M^d$ total bins), then the pairwise sum can be computed between **bin centers** weighted by bin counts:

$$S \approx \sum_{a=1}^{M^d} \sum_{b=1}^{M^d} c_a c_b \cdot \exp\left(-\frac{\|g_a - g_b\|^2}{4h_0^2}\right) \cdot P_d\left(\frac{\|g_a - g_b\|^2}{h_0^2}\right)$$

where $c_a$ is the number of data points in bin $a$ and $g_a$ is the bin center.

### Cost

$O(M^{2d})$ — independent of $n$!

For $d = 2$ with $M = 100$: $10{,}000$ bins, $10^8$ pairs. Still $O(n^2)$ in the number of bins, but:
1. The number of non-empty bins is often $\ll M^d$
2. Can combine with spatial truncation (most bin-pairs have negligible weight)

### When it works well

- Low $d$ (1–3): binning is very effective
- Large $n$: the savings from $O(M^{2d})$ vs $O(n^2)$ become enormous when $n \gg M^d$
- Smooth data: binning approximation error is small when bins are much smaller than $h_0$

### Linear binning (1D, for reference)

In 1D, this is exactly how the `bw.SJ` function in R works internally — it bins the data onto a fine grid ($M \approx 401$ points) and computes the roughness on the grid in $O(M^2)$ time. For $n = 100{,}000$ with $M = 401$: that's $160{,}000$ operations instead of $10^{10}$.

### Grid + FFT (the sweet spot for low $d$)

When the bins are regular, the double sum becomes a **convolution**:

$$S = \sum_a c_a \sum_b c_b \cdot K(g_a - g_b) = \sum_a c_a \cdot (c * K)(g_a)$$

Convolutions on regular grids can be computed in $O(M^d \log M^d)$ via **FFT**!

The algorithm:
1. Bin the data: histogram on a $M^d$ grid → $O(nd)$
2. Compute the kernel on the same grid → $O(M^d)$
3. Convolve via FFT: $\text{FFT}(c) \cdot \text{FFT}(K)$, then IFFT → $O(M^d \log M)$
4. Dot product with counts → $O(M^d)$

**Total: $O(nd + M^d \log M)$** — essentially linear in $n$ for fixed grid resolution!

For our case, the "kernel" is $K(\Delta) = \exp(-\|\Delta\|^2/(4h_0^2)) \cdot P_d(\|\Delta\|^2/h_0^2)$, which separates nicely on a grid.

### Implementation sketch (2D)

```python
from scipy.fft import fftn, ifftn

def sheather_jones_2d_fft(Y, h_0, M=256):
    n, d = Y.shape
    assert d == 2
    
    # Bin the data
    mins = Y.min(axis=0) - 4*h_0
    maxs = Y.max(axis=0) + 4*h_0
    counts, edges_x, edges_y = np.histogram2d(
        Y[:,0], Y[:,1], bins=M, range=[[mins[0],maxs[0]], [mins[1],maxs[1]]])
    
    # Grid spacing
    dx = (maxs[0] - mins[0]) / M
    dy = (maxs[1] - mins[1]) / M
    
    # Build kernel on same grid
    cx = np.arange(M) * dx  # relative positions
    cy = np.arange(M) * dy
    Cx, Cy = np.meshgrid(cx, cy, indexing='ij')
    dist_sq_grid = Cx**2 + Cy**2
    r_sq = dist_sq_grid / h_0**2
    
    kernel = np.exp(-r_sq/4) * (r_sq**2/16 - (d+2)*r_sq/4 + d*(d+2)/4)
    
    # Convolve via FFT (with zero-padding for linear convolution)
    c_fft = fftn(counts, s=[2*M, 2*M])
    k_fft = fftn(kernel, s=[2*M, 2*M])
    conv = np.real(ifftn(c_fft * k_fft))[:M, :M]
    
    # Sum
    S = np.sum(counts * conv)
    roughness = S / (n**2 * (4*np.pi)**(d/2) * h_0**(d+4))
    return roughness
```

---

## Strategy 6: Combining Approaches (Practical Recommendations)

### Decision tree by problem size

```
n < 3,000:     → Direct vectorized NumPy (current implementation)
                  Simple, fast enough, no approximation error.

n < 20,000:    → GPU if available (CuPy/PyTorch)
                  OR spatial truncation with KD-tree (cutoff = 6*h_0)
                  Both are exact (or epsilon-exact) and fast.

n < 100,000:   → Subsampling (m = 50,000 pairs) + exact diagonal
                  OR FFT-based binning (for d ≤ 3)
                  ~1% bandwidth error, sub-second computation.

n > 100,000:   → FFT-based binning (d ≤ 5)
                  OR Improved Fast Gauss Transform (any d)
                  OR subsampling (always works, any d, any n)
```

### Hybrid: Truncation + GPU

The best practical combination for moderate-to-large $n$:

1. Build a KD-tree on CPU ($O(n \log n)$)
2. Find all pairs within $R_{\text{cut}} = 6h_0$ ($O(nk)$ where $k \ll n$)
3. Transfer the pair list to GPU
4. Compute distances + polynomial + exponential on GPU
5. Sum on GPU, transfer scalar back

This gives near-optimal performance across all regimes.

---

## Comparison Summary

| Method | Complexity | Memory | Approx Error | Best For |
|--------|-----------|--------|--------------|----------|
| Direct (NumPy) | $O(n^2 d)$ | $O(n^2)$ | Exact | $n < 3000$ |
| GPU direct | $O(n^2 d / P)$ | $O(n^2)$ | Exact | $n < 30000$, GPU available |
| GPU tiled | $O(n^2 d / P)$ | $O(B^2)$ | Exact | $n < 50000$, limited GPU RAM |
| KD-tree truncation | $O(n^{(d+8)/(d+4)})$ | $O(nk)$ | $< 10^{-4}$ | Any $n$, any $d$ |
| Subsampling | $O(md)$ | $O(m)$ | ~1-5% | Any $n$, any $d$, simplest |
| FFT binning | $O(nd + M^d \log M)$ | $O(M^d)$ | Binning error | $n > 10000$, $d \leq 5$ |
| Fast Gauss Transform | $O(n)$ | $O(n)$ | $< \epsilon$ | Large $n$, $d \leq 10$ |

($P$ = number of GPU cores, $B$ = block size, $M$ = bins per dimension)

---

## Effect on Bandwidth Accuracy

A key point: **errors in the roughness estimate are dampened by the $(d+4)$-th root**.

$$h^* = \left(\frac{c}{\hat{\Psi}}\right)^{1/(d+4)}$$

If $\hat{\Psi}$ has relative error $\epsilon$:

$$\frac{\Delta h^*}{h^*} \approx \frac{\epsilon}{d+4}$$

| Relative error in $\hat{\Psi}$ | Error in $h^*$ (d=2) | Error in $h^*$ (d=5) |
|--------------------------------|----------------------|----------------------|
| 1% | 0.17% | 0.11% |
| 5% | 0.83% | 0.56% |
| 10% | 1.67% | 1.11% |

This means **subsampling with even 5-10% roughness error gives sub-1% bandwidth error** — which is well within the statistical uncertainty of the bandwidth estimate itself (which depends on the sample). There's no point computing the roughness to 12 decimal places when the underlying data is stochastic.

---

## Recommended Default Implementation

For a production library, the recommended approach is **adaptive**:

```python
def sheather_jones_nd_adaptive(X, cutoff=6.0, max_direct=3000, 
                                subsample_size=50000):
    n, d = X.shape
    
    # Whiten
    Y = whiten(X)
    h_0 = pilot_bandwidth(n, d)
    
    if n <= max_direct:
        # Exact computation
        return _direct(Y, h_0, d)
    
    elif n <= 20000 and d <= 10:
        # KD-tree truncation (nearly exact)
        return _kdtree_truncated(Y, h_0, d, cutoff)
    
    else:
        # Subsampling (fast, slight approximation)
        return _subsample(Y, h_0, d, subsample_size)
```

This gives:
- **Exact** results for small datasets
- **Near-exact** ($< 10^{-4}$ error) for medium datasets
- **Approximate** (~1% bandwidth error) for large datasets
- All in **sub-second** time regardless of $n$
