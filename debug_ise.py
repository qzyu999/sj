"""
Debug: Why is the two-stage method giving worse ISE?

Hypothesis: The bandwidth is too small (undersmoothing), which is actually
CORRECT for density estimation (minimizes MISE) but appears worse on the
specific ISE grid computation because the grid is too coarse to capture
the sharp features, OR because the true density is well-separated Gaussian
mixture where Scott/Silverman happen to be near-optimal.

Let's check by:
1. Printing actual bandwidth values
2. Computing ISE with finer grids
3. Testing on a density where the optimal h is clearly DIFFERENT from Silverman
"""

import numpy as np
from scipy.stats import multivariate_normal
from implement_two_stage import (
    silverman_bandwidth, one_stage_dpi, two_stage_dpi, ste_bandwidth,
    whiten, compute_psi4, compute_psi6
)

# =============================================================================
# Test 1: Check bandwidth values
# =============================================================================
print("=" * 70)
print("BANDWIDTH VALUES")
print("=" * 70)

rng = np.random.default_rng(42)
means = np.array([[0, 0], [3, 3], [-3, 3]], dtype=float)
n_per = 200
X_list = [rng.standard_normal((n_per, 2)) + m for m in means]
X = np.vstack(X_list)
n, d = X.shape

h_silv = silverman_bandwidth(n, d)
h_1stg = one_stage_dpi(X)
h_2stg = two_stage_dpi(X)
h_ste_v = ste_bandwidth(X)

print(f"\n2D mixture (n={n}, d={d}, 3 components, sep=3):")
print(f"  Silverman: h = {h_silv:.4f}")
print(f"  1-stage:   h = {h_1stg:.4f}")
print(f"  2-stage:   h = {h_2stg:.4f}")
print(f"  STE:       h = {h_ste_v:.4f}")

# The issue: our methods return a bandwidth for WHITENED data.
# But compute_ise uses the raw data. We need to be consistent.
# Let me compute ISE in whitened coordinates.

Y = whiten(X)
print(f"\n  Data std (raw): {X.std(axis=0)}")
print(f"  Data std (whitened): {Y.std(axis=0)}")

# =============================================================================
# Test 2: Optimal bandwidth search (brute force)
# =============================================================================
print("\n" + "=" * 70)
print("OPTIMAL BANDWIDTH (BRUTE FORCE ISE MINIMIZATION)")
print("=" * 70)

def compute_ise_whitened(Y, h, true_pdf_fn, grid_points=80):
    """Compute ISE in whitened coordinates."""
    n, d = Y.shape
    mins = Y.min(axis=0) - 4*h
    maxs = Y.max(axis=0) + 4*h
    
    g1 = np.linspace(mins[0], maxs[0], grid_points)
    g2 = np.linspace(mins[1], maxs[1], grid_points)
    G1, G2 = np.meshgrid(g1, g2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])
    dx = (g1[1]-g1[0]) * (g2[1]-g2[0])
    
    # KDE at grid points (using bandwidth h in whitened space)
    kde_vals = np.zeros(grid.shape[0])
    for i in range(n):
        diff = grid - Y[i]
        kde_vals += np.exp(-np.sum(diff**2, axis=1) / (2*h**2))
    kde_vals /= n * (2*np.pi*h**2)**(d/2)
    
    # True density at grid points
    true_vals = true_pdf_fn(grid)
    
    ise = np.sum((kde_vals - true_vals)**2) * dx
    return ise

# Need the true density in whitened coordinates
# Original mixture: equal weight N(mu_k, I_2) for each mean
# After whitening by sample covariance, the true density transforms too.
# For a fair comparison, let's just work with a KNOWN density.

# Simpler test: known 2D mixture in standardized form
print("\nTest: 2D standard mixture with known true density")
rng = np.random.default_rng(0)
# True density: 0.5*N([2,0], I) + 0.5*N([-2,0], I)
mu1 = np.array([2.0, 0.0])
mu2 = np.array([-2.0, 0.0])

X_test = np.vstack([
    rng.standard_normal((300, 2)) + mu1,
    rng.standard_normal((300, 2)) + mu2,
])
n_test = len(X_test)

# The true density in original coordinates
def true_pdf_original(x):
    return 0.5 * multivariate_normal.pdf(x, mu1) + 0.5 * multivariate_normal.pdf(x, mu2)

# Brute force: scan bandwidths
h_range = np.linspace(0.2, 1.5, 30)
ise_by_h = []
for h_try in h_range:
    ise_val = compute_ise_whitened(X_test, h_try, true_pdf_original, grid_points=60)
    ise_by_h.append(ise_val)

h_opt_bf = h_range[np.argmin(ise_by_h)]
ise_opt = min(ise_by_h)

# Our methods
h_silv2 = silverman_bandwidth(n_test, 2)
h_1stg2 = one_stage_dpi(X_test)
h_2stg2 = two_stage_dpi(X_test)
h_ste2 = ste_bandwidth(X_test)

print(f"\n  Optimal h (brute force): {h_opt_bf:.4f} (ISE = {ise_opt:.6e})")
print(f"  Silverman:               {h_silv2:.4f} (ISE = {compute_ise_whitened(X_test, h_silv2, true_pdf_original, 60):.6e})")
print(f"  1-stage:                 {h_1stg2:.4f} (ISE = {compute_ise_whitened(X_test, h_1stg2, true_pdf_original, 60):.6e})")
print(f"  2-stage:                 {h_2stg2:.4f} (ISE = {compute_ise_whitened(X_test, h_2stg2, true_pdf_original, 60):.6e})")
print(f"  STE:                     {h_ste2:.4f} (ISE = {compute_ise_whitened(X_test, h_ste2, true_pdf_original, 60):.6e})")

# =============================================================================
# The problem might be that our methods return bandwidth for WHITENED data
# but we're applying it to RAW data. Let me check.
# =============================================================================
print("\n\nCheck: is the method already whitening internally?")
print(f"  X_test shape: {X_test.shape}")
print(f"  X_test cov diag: {np.cov(X_test, rowvar=False).diagonal()}")
print(f"  X_test overall std: {X_test.std():.4f}")

# One-stage internally whitens, so h returned is for whitened coordinates.
# When we apply it to raw data, we need to adjust.
# Actually our method returns h for whitened space. The full bandwidth matrix is h^2 * Sigma_hat.
# For ISE computation on raw data with isotropic kernel, we should use h * sqrt(eigenvalue).
# But for the ISE comparison to be fair, let's compute everything in whitened coordinates.

Y_test = whiten(X_test)
print(f"  Y_test (whitened) cov diag: {np.cov(Y_test, rowvar=False).diagonal()}")

# We need the true density in WHITENED coordinates
# After whitening by W = Sigma_hat^{-1/2}, the density transforms as:
# f_Y(y) = f_X(W^{-1}*y + mu) * |det(W^{-1})| = f_X(Sigma_hat^{1/2}*y + mu_X) * |det(Sigma_hat)|^{1/2}
from scipy.linalg import sqrtm as sp_sqrtm

mu_X = X_test.mean(axis=0)
cov_X = np.cov(X_test, rowvar=False)
Sigma_sqrt = sp_sqrtm(cov_X).real
det_Sigma = np.linalg.det(cov_X)

def true_pdf_whitened(y):
    """True density evaluated at whitened coordinates."""
    # Transform back to original space
    x = (y @ Sigma_sqrt.T) + mu_X
    return true_pdf_original(x) * np.sqrt(det_Sigma)

# Brute force on whitened data
ise_by_h_w = []
h_range_w = np.linspace(0.2, 1.2, 25)
for h_try in h_range_w:
    ise_val = compute_ise_whitened(Y_test, h_try, true_pdf_whitened, grid_points=60)
    ise_by_h_w.append(ise_val)

h_opt_w = h_range_w[np.argmin(ise_by_h_w)]
ise_opt_w = min(ise_by_h_w)

print(f"\n  In WHITENED coordinates:")
print(f"  Optimal h (brute force): {h_opt_w:.4f} (ISE = {ise_opt_w:.6e})")
print(f"  Silverman:               {h_silv2:.4f} (ISE = {compute_ise_whitened(Y_test, h_silv2, true_pdf_whitened, 60):.6e})")
print(f"  1-stage:                 {h_1stg2:.4f} (ISE = {compute_ise_whitened(Y_test, h_1stg2, true_pdf_whitened, 60):.6e})")
print(f"  2-stage:                 {h_2stg2:.4f} (ISE = {compute_ise_whitened(Y_test, h_2stg2, true_pdf_whitened, 60):.6e})")
print(f"  STE:                     {h_ste2:.4f} (ISE = {compute_ise_whitened(Y_test, h_ste2, true_pdf_whitened, 60):.6e})")

# =============================================================================
# Test 3: A density where Silverman is clearly wrong
# =============================================================================
print("\n" + "=" * 70)
print("TEST: HIGHLY MULTIMODAL (5 clusters, tight)")
print("=" * 70)

rng = np.random.default_rng(123)
# 5 well-separated narrow clusters
centers = np.array([[0,0], [5,0], [0,5], [5,5], [2.5,2.5]], dtype=float)
sigma_cluster = 0.5  # tight clusters
n_per_cluster = 100

X_multi = np.vstack([
    rng.standard_normal((n_per_cluster, 2)) * sigma_cluster + c 
    for c in centers
])

def true_pdf_multi(x):
    pdf = np.zeros(x.shape[0])
    for c in centers:
        pdf += multivariate_normal.pdf(x, mean=c, cov=sigma_cluster**2 * np.eye(2))
    return pdf / len(centers)

n_multi = len(X_multi)
h_silv_m = silverman_bandwidth(n_multi, 2)
h_1stg_m = one_stage_dpi(X_multi)
h_2stg_m = two_stage_dpi(X_multi)
h_ste_m = ste_bandwidth(X_multi)

# Brute force
h_range_m = np.linspace(0.1, 2.0, 40)
ise_m = [compute_ise_whitened(X_multi, h, true_pdf_multi, 80) for h in h_range_m]
h_opt_m = h_range_m[np.argmin(ise_m)]

print(f"\n  n={n_multi}, d=2, 5 tight clusters (sigma={sigma_cluster})")
print(f"  Optimal h (brute force): {h_opt_m:.4f} (ISE = {min(ise_m):.6e})")
print(f"  Silverman:               {h_silv_m:.4f} (ISE = {compute_ise_whitened(X_multi, h_silv_m, true_pdf_multi, 80):.6e})")
print(f"  1-stage:                 {h_1stg_m:.4f} (ISE = {compute_ise_whitened(X_multi, h_1stg_m, true_pdf_multi, 80):.6e})")
print(f"  2-stage:                 {h_2stg_m:.4f} (ISE = {compute_ise_whitened(X_multi, h_2stg_m, true_pdf_multi, 80):.6e})")
print(f"  STE:                     {h_ste_m:.4f} (ISE = {compute_ise_whitened(X_multi, h_ste_m, true_pdf_multi, 80):.6e})")

# Final: 2D bimodal with CLOSE modes (where bandwidth really matters)
print("\n" + "=" * 70)
print("TEST: CLOSE BIMODAL (separation = 2)")
print("=" * 70)

rng = np.random.default_rng(77)
mu_a = np.array([1.0, 0.0])
mu_b = np.array([-1.0, 0.0])
X_bim = np.vstack([
    rng.standard_normal((400, 2)) + mu_a,
    rng.standard_normal((400, 2)) + mu_b,
])

def true_pdf_bim(x):
    return 0.5 * multivariate_normal.pdf(x, mu_a) + 0.5 * multivariate_normal.pdf(x, mu_b)

n_bim = len(X_bim)
h_silv_b = silverman_bandwidth(n_bim, 2)
h_1stg_b = one_stage_dpi(X_bim)
h_2stg_b = two_stage_dpi(X_bim)
h_ste_b = ste_bandwidth(X_bim)

# Brute force
h_range_b = np.linspace(0.2, 1.2, 30)
ise_b = [compute_ise_whitened(X_bim, h, true_pdf_bim, 70) for h in h_range_b]
h_opt_b = h_range_b[np.argmin(ise_b)]

print(f"\n  n={n_bim}, d=2, bimodal (separation=2)")
print(f"  Optimal h (brute force): {h_opt_b:.4f} (ISE = {min(ise_b):.6e})")
print(f"  Silverman:               {h_silv_b:.4f} (ISE = {compute_ise_whitened(X_bim, h_silv_b, true_pdf_bim, 70):.6e})")
print(f"  1-stage:                 {h_1stg_b:.4f} (ISE = {compute_ise_whitened(X_bim, h_1stg_b, true_pdf_bim, 70):.6e})")
print(f"  2-stage:                 {h_2stg_b:.4f} (ISE = {compute_ise_whitened(X_bim, h_2stg_b, true_pdf_bim, 70):.6e})")
print(f"  STE:                     {h_ste_b:.4f} (ISE = {compute_ise_whitened(X_bim, h_ste_b, true_pdf_bim, 70):.6e})")
