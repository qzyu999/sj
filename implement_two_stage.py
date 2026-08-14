"""
Implementation and verification of the full two-stage d-D Sheather-Jones algorithm.

Compares:
  1. Silverman rule-of-thumb
  2. One-stage DPI (our current gsj method — Silverman pilot)
  3. Two-stage DPI (new — data-driven Psi_6 pilot)
  4. STE variant (full SJ analog)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.linalg import sqrtm, inv

# =============================================================================
# POLYNOMIALS
# =============================================================================

def P_d(t, d):
    """Psi_4 pairwise polynomial (Laplacian roughness)."""
    return t**2 / 16 - (d + 2) * t / 4 + d * (d + 2) / 4

def R_d(t, d):
    """Psi_6 pairwise polynomial (gradient-of-Laplacian roughness)."""
    return -t**3 / 64 + 3*(d + 4) * t**2 / 32 - 3*(d + 2)*(d + 4) * t / 16 + d*(d + 2)*(d + 4) / 8

# =============================================================================
# CORE ROUTINES
# =============================================================================

def whiten(X):
    """Whiten data to have identity covariance."""
    n, d = X.shape
    mu = X.mean(axis=0)
    X_centered = X - mu
    cov = np.cov(X_centered, rowvar=False)
    # Use eigendecomposition for numerical stability
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return X_centered @ W.T

def compute_roughness(Y, h, d, poly_fn, poly_order):
    """
    Compute roughness functional using given polynomial.
    
    poly_order: d+4 for Psi_4, d+6 for Psi_6
    """
    n = Y.shape[0]
    
    if n <= 3000:
        # Exact computation
        diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        r_sq = dist_sq / h**2
        P = poly_fn(r_sq, d)
        W = np.exp(-r_sq / 4.0)
        S = np.sum(W * P)
    else:
        # Subsample
        rng = np.random.default_rng(42)
        m = 80000
        idx_i = rng.integers(0, n, m)
        idx_j = rng.integers(0, n, m)
        diffs = Y[idx_i] - Y[idx_j]
        dist_sq = np.sum(diffs**2, axis=1)
        r_sq = dist_sq / h**2
        P = poly_fn(r_sq, d)
        W = np.exp(-r_sq / 4.0)
        # Diagonal contribution
        S_diag = n * poly_fn(0.0, d)
        S = (n**2 / m) * np.sum(W * P) + S_diag
    
    return S / (n**2 * (4 * np.pi)**(d/2) * h**poly_order)

def compute_psi4(Y, h, d):
    """Estimate Psi_4 = R(nabla^2 f) using P_d polynomial."""
    return compute_roughness(Y, h, d, P_d, d + 4)

def compute_psi6(Y, h, d):
    """Estimate Psi_6 = R(grad(nabla^2 f)) using R_d polynomial."""
    return compute_roughness(Y, h, d, R_d, d + 6)

# =============================================================================
# BANDWIDTH SELECTORS
# =============================================================================

def silverman_bandwidth(n, d):
    """Silverman rule of thumb."""
    return (4.0 / (n * (d + 2)))**(1.0 / (d + 4))

def one_stage_dpi(X):
    """Our current method: Silverman pilot -> Psi_4 -> h*."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    
    Y = whiten(X)
    h_0 = silverman_bandwidth(n, d)
    psi4 = compute_psi4(Y, h_0, d)
    
    if psi4 <= 0:
        return h_0  # fallback
    
    h_star = (d / (n * psi4 * (4 * np.pi)**(d/2)))**(1.0 / (d + 4))
    return h_star

def two_stage_dpi(X):
    """
    Full two-stage DPI (true SJ generalization):
    Normal reference -> Psi_6 estimate -> adaptive pilot -> Psi_4 -> h*
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    
    Y = whiten(X)
    
    # Robust scale estimate (in whitened space, should be ~1 for normal data)
    norms = np.sqrt(np.sum(Y**2, axis=1))
    sigma_hat = np.median(norms) / np.sqrt(d)
    
    # Stage 3: Normal-reference pilot bandwidth for estimating Psi_6
    # (This is the bandwidth for estimating Psi_4 via bias cancellation)
    g_2 = sigma_hat * (2.0 / (n * (d + 4)))**(1.0 / (d + 6))
    
    # Stage 2: Estimate Psi_6 using R_d at bandwidth g_2
    psi6_hat = compute_psi6(Y, g_2, d)
    
    if psi6_hat <= 0:
        # Fallback to one-stage if Psi_6 estimate is non-positive
        return one_stage_dpi(X)
    
    # Stage 1: Adaptive pilot from data-driven Psi_6
    g_1 = (d * (d + 2) / (4 * n * (4 * np.pi)**(d/2) * psi6_hat))**(1.0 / (d + 6))
    
    # Stage 0: Estimate Psi_4 using P_d at adaptive pilot g_1
    psi4_hat = compute_psi4(Y, g_1, d)
    
    if psi4_hat <= 0:
        return one_stage_dpi(X)
    
    # Final bandwidth
    h_star = (d / (n * psi4_hat * (4 * np.pi)**(d/2)))**(1.0 / (d + 4))
    return h_star

def ste_bandwidth(X):
    """
    Solve-the-equation variant: find h satisfying h = h_AMISE(Psi_4(alpha(h))).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    
    Y = whiten(X)
    
    # Get Psi_6 estimate (same as two-stage Stage 2)
    sigma_hat = np.median(np.sqrt(np.sum(Y**2, axis=1))) / np.sqrt(d)
    g_2 = sigma_hat * (2.0 / (n * (d + 4)))**(1.0 / (d + 6))
    psi6_hat = compute_psi6(Y, g_2, d)
    
    if psi6_hat <= 0:
        return one_stage_dpi(X)
    
    # Get initial Psi_4 estimate for the ratio
    g_1_init = (d * (d + 2) / (4 * n * (4 * np.pi)**(d/2) * psi6_hat))**(1.0 / (d + 6))
    psi4_init = compute_psi4(Y, g_1_init, d)
    
    if psi4_init <= 0:
        return one_stage_dpi(X)
    
    # STE: solve h = h_AMISE(Psi_4(alpha(h)))
    # alpha(h) = C * (psi4/psi6)^{1/(d+6)} * h^{(d+4)/(d+6)}
    # We use a simplified form: alpha(h) proportional to h^{(d+4)/(d+6)}
    
    ratio = (psi4_init / psi6_hat)**(1.0 / (d + 6))
    C_ste = ratio * g_1_init / g_1_init**((d+4.0)/(d+6.0))  # calibrate constant
    
    def ste_equation(h):
        alpha = C_ste * h**((d + 4.0) / (d + 6.0))
        alpha = max(alpha, 1e-10)  # safety
        psi4 = compute_psi4(Y, alpha, d)
        if psi4 <= 0:
            return h - silverman_bandwidth(n, d)
        h_opt = (d / (n * psi4 * (4 * np.pi)**(d/2)))**(1.0 / (d + 4))
        return h_opt - h
    
    # Find bounds
    h_low = silverman_bandwidth(n, d) * 0.1
    h_high = silverman_bandwidth(n, d) * 5.0
    
    try:
        # Check if equation changes sign
        f_low = ste_equation(h_low)
        f_high = ste_equation(h_high)
        
        if f_low * f_high > 0:
            # No sign change — use two-stage result instead
            return two_stage_dpi(X)
        
        h_ste = brentq(ste_equation, h_low, h_high, xtol=1e-8, maxiter=50)
        return h_ste
    except (ValueError, RuntimeError):
        return two_stage_dpi(X)

# =============================================================================
# TEST: Compare methods on synthetic data
# =============================================================================

def true_mise_bandwidth(X_train, X_test, method_fn, true_density_fn=None):
    """Compute ISE for a given bandwidth method."""
    h = method_fn(X_train)
    return h

def generate_mixture(n, d, n_components=3, separation=3.0, rng=None):
    """Generate a Gaussian mixture in d dimensions."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Random means separated by `separation`
    means = rng.standard_normal((n_components, d)) * separation
    
    # Equal weights
    labels = rng.integers(0, n_components, n)
    X = np.zeros((n, d))
    for k in range(n_components):
        mask = labels == k
        X[mask] = rng.standard_normal((mask.sum(), d)) + means[k]
    
    return X

print("=" * 70)
print("COMPARISON: One-Stage vs Two-Stage vs STE")
print("=" * 70)

np.random.seed(42)

# Test scenarios
scenarios = [
    ("Normal d=2, n=200", lambda rng: rng.standard_normal((200, 2))),
    ("Normal d=5, n=500", lambda rng: rng.standard_normal((500, 5))),
    ("Normal d=10, n=1000", lambda rng: rng.standard_normal((1000, 10))),
    ("Mixture d=2, n=500", lambda rng: generate_mixture(500, 2, 3, 3.0, rng)),
    ("Mixture d=5, n=1000", lambda rng: generate_mixture(1000, 5, 3, 3.0, rng)),
    ("Mixture d=3, n=300", lambda rng: generate_mixture(300, 3, 5, 4.0, rng)),
]

print(f"\n{'Scenario':<28} {'Silverman':>10} {'1-stage':>10} {'2-stage':>10} {'STE':>10}")
print("-" * 70)

for name, gen_fn in scenarios:
    rng = np.random.default_rng(42)
    X = gen_fn(rng)
    n, d = X.shape
    
    h_silv = silverman_bandwidth(n, d)
    h_1stage = one_stage_dpi(X)
    h_2stage = two_stage_dpi(X)
    h_ste = ste_bandwidth(X)
    
    print(f"{name:<28} {h_silv:10.6f} {h_1stage:10.6f} {h_2stage:10.6f} {h_ste:10.6f}")

# =============================================================================
# ISE comparison on known distributions
# =============================================================================
print("\n" + "=" * 70)
print("ISE COMPARISON (known true density)")
print("=" * 70)

from scipy.stats import multivariate_normal

def compute_ise(X, h, true_pdf_fn, grid_points=50):
    """Approximate ISE = int (f_hat - f)^2 dx via grid."""
    n, d = X.shape
    
    # Create grid in the data range
    mins = X.min(axis=0) - 3*h
    maxs = X.max(axis=0) + 3*h
    
    if d == 2:
        g1 = np.linspace(mins[0], maxs[0], grid_points)
        g2 = np.linspace(mins[1], maxs[1], grid_points)
        G1, G2 = np.meshgrid(g1, g2)
        grid = np.column_stack([G1.ravel(), G2.ravel()])
        dx = (g1[1]-g1[0]) * (g2[1]-g2[0])
    elif d == 3:
        gp = 25  # fewer points for 3D
        g1 = np.linspace(mins[0], maxs[0], gp)
        g2 = np.linspace(mins[1], maxs[1], gp)
        g3 = np.linspace(mins[2], maxs[2], gp)
        G1, G2, G3 = np.meshgrid(g1, g2, g3, indexing='ij')
        grid = np.column_stack([G1.ravel(), G2.ravel(), G3.ravel()])
        dx = (g1[1]-g1[0]) * (g2[1]-g2[0]) * (g3[1]-g3[0])
    else:
        return None  # skip higher dims
    
    # KDE at grid points
    kde_vals = np.zeros(grid.shape[0])
    for i in range(n):
        diff = grid - X[i]
        kde_vals += np.exp(-np.sum(diff**2, axis=1) / (2*h**2))
    kde_vals /= n * (2*np.pi*h**2)**(d/2)
    
    # True density at grid points
    true_vals = true_pdf_fn(grid)
    
    # ISE
    ise = np.sum((kde_vals - true_vals)**2) * dx
    return ise

# Test: 2D mixture
print("\n2D Gaussian Mixture (3 components, separation=3):")
print(f"{'Trial':<8} {'ISE(Silv)':>12} {'ISE(1-stg)':>12} {'ISE(2-stg)':>12} {'ISE(STE)':>12}")
print("-" * 60)

n_trials = 10
ise_results = {k: [] for k in ['silv', '1stage', '2stage', 'ste']}

for trial in range(n_trials):
    rng = np.random.default_rng(trial)
    
    # Fixed mixture
    means = np.array([[0, 0], [3, 3], [-3, 3]], dtype=float)
    n_per = 200
    X_list = [rng.standard_normal((n_per, 2)) + m for m in means]
    X = np.vstack(X_list)
    n, d = X.shape
    
    def true_pdf(x):
        pdf = np.zeros(x.shape[0])
        for m in means:
            pdf += multivariate_normal.pdf(x, mean=m)
        return pdf / len(means)
    
    h_silv = silverman_bandwidth(n, d)
    h_1stg = one_stage_dpi(X)
    h_2stg = two_stage_dpi(X)
    h_ste_v = ste_bandwidth(X)
    
    ise_silv = compute_ise(X, h_silv, true_pdf)
    ise_1stg = compute_ise(X, h_1stg, true_pdf)
    ise_2stg = compute_ise(X, h_2stg, true_pdf)
    ise_ste = compute_ise(X, h_ste_v, true_pdf)
    
    ise_results['silv'].append(ise_silv)
    ise_results['1stage'].append(ise_1stg)
    ise_results['2stage'].append(ise_2stg)
    ise_results['ste'].append(ise_ste)
    
    print(f"{trial:<8} {ise_silv:12.6e} {ise_1stg:12.6e} {ise_2stg:12.6e} {ise_ste:12.6e}")

print("-" * 60)
print(f"{'MEAN':<8} {np.mean(ise_results['silv']):12.6e} "
      f"{np.mean(ise_results['1stage']):12.6e} "
      f"{np.mean(ise_results['2stage']):12.6e} "
      f"{np.mean(ise_results['ste']):12.6e}")

# Relative improvement
baseline = np.mean(ise_results['silv'])
print(f"\nRelative to Silverman:")
print(f"  1-stage: {(np.mean(ise_results['1stage'])/baseline - 1)*100:+.1f}%")
print(f"  2-stage: {(np.mean(ise_results['2stage'])/baseline - 1)*100:+.1f}%")
print(f"  STE:     {(np.mean(ise_results['ste'])/baseline - 1)*100:+.1f}%")

# =============================================================================
# 3D test
# =============================================================================
print("\n\n3D Gaussian Mixture (4 components, separation=4):")
print(f"{'Trial':<8} {'ISE(Silv)':>12} {'ISE(1-stg)':>12} {'ISE(2-stg)':>12} {'ISE(STE)':>12}")
print("-" * 60)

ise_results3d = {k: [] for k in ['silv', '1stage', '2stage', 'ste']}

for trial in range(n_trials):
    rng = np.random.default_rng(trial + 100)
    
    means3d = np.array([[0,0,0], [4,0,0], [0,4,0], [0,0,4]], dtype=float)
    n_per = 150
    X_list = [rng.standard_normal((n_per, 3)) + m for m in means3d]
    X = np.vstack(X_list)
    n, d = X.shape
    
    def true_pdf_3d(x):
        pdf = np.zeros(x.shape[0])
        for m in means3d:
            pdf += multivariate_normal.pdf(x, mean=m)
        return pdf / len(means3d)
    
    h_silv = silverman_bandwidth(n, d)
    h_1stg = one_stage_dpi(X)
    h_2stg = two_stage_dpi(X)
    h_ste_v = ste_bandwidth(X)
    
    ise_silv = compute_ise(X, h_silv, true_pdf_3d)
    ise_1stg = compute_ise(X, h_1stg, true_pdf_3d)
    ise_2stg = compute_ise(X, h_2stg, true_pdf_3d)
    ise_ste = compute_ise(X, h_ste_v, true_pdf_3d)
    
    if ise_silv is not None:
        ise_results3d['silv'].append(ise_silv)
        ise_results3d['1stage'].append(ise_1stg)
        ise_results3d['2stage'].append(ise_2stg)
        ise_results3d['ste'].append(ise_ste)
        
        print(f"{trial:<8} {ise_silv:12.6e} {ise_1stg:12.6e} {ise_2stg:12.6e} {ise_ste:12.6e}")

if ise_results3d['silv']:
    print("-" * 60)
    print(f"{'MEAN':<8} {np.mean(ise_results3d['silv']):12.6e} "
          f"{np.mean(ise_results3d['1stage']):12.6e} "
          f"{np.mean(ise_results3d['2stage']):12.6e} "
          f"{np.mean(ise_results3d['ste']):12.6e}")
    
    baseline3d = np.mean(ise_results3d['silv'])
    print(f"\nRelative to Silverman:")
    print(f"  1-stage: {(np.mean(ise_results3d['1stage'])/baseline3d - 1)*100:+.1f}%")
    print(f"  2-stage: {(np.mean(ise_results3d['2stage'])/baseline3d - 1)*100:+.1f}%")
    print(f"  STE:     {(np.mean(ise_results3d['ste'])/baseline3d - 1)*100:+.1f}%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
