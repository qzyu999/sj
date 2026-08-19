"""
Rigorous benchmark: GSJ vs alternatives on known densities.

Methodology:
- Known true density f(x) → can compute exact ISE for any h
- For each method, select h, compute ISE = int (f_hat - f)^2 dx
- Repeat over many trials (different random samples from same f)
- Report: mean ISE, std, paired t-test (GSJ vs each alternative)

Methods compared:
1. Silverman (normal reference)
2. Scott (similar to Silverman)  
3. LSCV (likelihood cross-validation via scipy)
4. GSJ one-stage (our Silverman pilot → P_d)
5. GSJ two-stage (full SJ generalization)

Densities tested (d=2 and d=3):
A. Standard normal (where Silverman is OPTIMAL)
B. Well-separated bimodal (separation = 4σ)
C. Close bimodal (separation = 2σ)
D. Skewed mixture (unequal weights)
E. 5-component mixture (complex multimodal)
"""

import sys
sys.path.insert(0, 'gsj/src')

import numpy as np
from scipy.stats import multivariate_normal, ttest_rel
from scipy.optimize import minimize_scalar
import time

from gsj import bandwidth


# =============================================================================
# ISE COMPUTATION (numerical integration on grid)
# =============================================================================

def kde_evaluate(X, h, grid):
    """Evaluate Gaussian KDE at grid points."""
    n, d = X.shape
    ng = grid.shape[0]
    
    # Vectorized: compute all distances at once in chunks to avoid memory blow-up
    chunk_size = 5000
    kde_vals = np.zeros(ng)
    
    for start in range(0, ng, chunk_size):
        end = min(start + chunk_size, ng)
        g = grid[start:end]  # (chunk, d)
        # (chunk, n, d) differences
        diff = g[:, np.newaxis, :] - X[np.newaxis, :, :]  # broadcast: (chunk, n, d)
        sq_dist = np.sum(diff**2, axis=2)  # (chunk, n)
        kde_vals[start:end] = np.mean(np.exp(-sq_dist / (2 * h**2)), axis=1)
    
    kde_vals /= (2 * np.pi * h**2) ** (d / 2)
    return kde_vals


def compute_ise_2d(X, h, true_pdf_fn, n_grid=80):
    """Compute ISE on a 2D grid."""
    n, d = X.shape
    assert d == 2
    
    # Grid covers data range + padding
    margin = 4 * h
    mins = X.min(axis=0) - margin
    maxs = X.max(axis=0) + margin
    
    g1 = np.linspace(mins[0], maxs[0], n_grid)
    g2 = np.linspace(mins[1], maxs[1], n_grid)
    G1, G2 = np.meshgrid(g1, g2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])
    dx = (g1[1] - g1[0]) * (g2[1] - g2[0])
    
    kde_vals = kde_evaluate(X, h, grid)
    true_vals = true_pdf_fn(grid)
    
    ise = np.sum((kde_vals - true_vals)**2) * dx
    return ise


def compute_ise_3d(X, h, true_pdf_fn, n_grid=35):
    """Compute ISE on a 3D grid."""
    n, d = X.shape
    assert d == 3
    
    margin = 3.5 * h
    mins = X.min(axis=0) - margin
    maxs = X.max(axis=0) + margin
    
    g1 = np.linspace(mins[0], maxs[0], n_grid)
    g2 = np.linspace(mins[1], maxs[1], n_grid)
    g3 = np.linspace(mins[2], maxs[2], n_grid)
    G1, G2, G3 = np.meshgrid(g1, g2, g3, indexing='ij')
    grid = np.column_stack([G1.ravel(), G2.ravel(), G3.ravel()])
    dx = (g1[1] - g1[0]) * (g2[1] - g2[0]) * (g3[1] - g3[0])
    
    kde_vals = kde_evaluate(X, h, grid)
    true_vals = true_pdf_fn(grid)
    
    ise = np.sum((kde_vals - true_vals)**2) * dx
    return ise


def oracle_bandwidth(X, true_pdf_fn, d):
    """Find the ISE-optimal bandwidth by brute-force search."""
    compute_ise = compute_ise_2d if d == 2 else compute_ise_3d
    
    def neg_ise(log_h):
        h = np.exp(log_h)
        return compute_ise(X, h, true_pdf_fn)
    
    # Search over a reasonable range
    n = X.shape[0]
    h_silv = (4 / (n * (d + 2))) ** (1 / (d + 4))
    result = minimize_scalar(neg_ise, bounds=(np.log(h_silv * 0.2), np.log(h_silv * 3)),
                             method='bounded')
    return np.exp(result.x), result.fun


# =============================================================================
# BANDWIDTH METHODS
# =============================================================================

def silverman_h(X):
    n, d = X.shape
    return (4 / (n * (d + 2))) ** (1 / (d + 4))


def scott_h(X):
    n, d = X.shape
    return n ** (-1 / (d + 4))


def lscv_h(X):
    """Likelihood cross-validation (leave-one-out)."""
    n, d = X.shape
    
    def neg_loocv(log_h):
        h = np.exp(log_h)
        # LOO log-likelihood
        total = 0.0
        # Vectorized: for each point, compute KDE without that point
        # Use pairwise distances
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)  # (n, n)
        kernel_vals = np.exp(-sq_dist / (2 * h**2)) / (2 * np.pi * h**2) ** (d / 2)
        np.fill_diagonal(kernel_vals, 0)  # leave-one-out
        loo_densities = kernel_vals.sum(axis=1) / (n - 1)
        # Avoid log(0)
        loo_densities = np.maximum(loo_densities, 1e-300)
        return -np.mean(np.log(loo_densities))
    
    h_silv = silverman_h(X)
    result = minimize_scalar(neg_loocv, bounds=(np.log(h_silv * 0.1), np.log(h_silv * 5)),
                             method='bounded')
    return np.exp(result.x)


def gsj_one_stage_h(X):
    return bandwidth(X, algorithm='one-stage')


def gsj_two_stage_h(X):
    return bandwidth(X, algorithm='two-stage')


# =============================================================================
# TEST DENSITIES
# =============================================================================

def make_density_2d(name):
    """Return (sampler, pdf_fn, description)."""
    
    if name == "normal":
        def sample(n, rng):
            return rng.standard_normal((n, 2))
        def pdf(x):
            return multivariate_normal.pdf(x, mean=[0, 0])
        return sample, pdf, "Standard Normal (Silverman optimal)"
    
    elif name == "bimodal_far":
        mu1, mu2 = np.array([2, 0]), np.array([-2, 0])
        def sample(n, rng):
            labels = rng.integers(0, 2, n)
            X = rng.standard_normal((n, 2))
            X[labels == 0] += mu1
            X[labels == 1] += mu2
            return X
        def pdf(x):
            return 0.5 * multivariate_normal.pdf(x, mu1) + 0.5 * multivariate_normal.pdf(x, mu2)
        return sample, pdf, "Bimodal (sep=4)"
    
    elif name == "bimodal_close":
        mu1, mu2 = np.array([1, 0]), np.array([-1, 0])
        def sample(n, rng):
            labels = rng.integers(0, 2, n)
            X = rng.standard_normal((n, 2))
            X[labels == 0] += mu1
            X[labels == 1] += mu2
            return X
        def pdf(x):
            return 0.5 * multivariate_normal.pdf(x, mu1) + 0.5 * multivariate_normal.pdf(x, mu2)
        return sample, pdf, "Bimodal (sep=2)"
    
    elif name == "skewed":
        mu1 = np.array([0, 0])
        mu2 = np.array([3, 2])
        cov2 = np.array([[0.5, 0.2], [0.2, 0.5]])
        def sample(n, rng):
            n1 = int(0.7 * n)
            X1 = rng.standard_normal((n1, 2)) + mu1
            X2 = rng.multivariate_normal(mu2, cov2, n - n1)
            return np.vstack([X1, X2])
        def pdf(x):
            return (0.7 * multivariate_normal.pdf(x, mu1) + 
                    0.3 * multivariate_normal.pdf(x, mu2, cov2))
        return sample, pdf, "Skewed (70/30 mix, different covariances)"
    
    elif name == "five_cluster":
        centers = np.array([[0,0], [3,0], [0,3], [3,3], [1.5,1.5]])
        sigma = 0.6
        def sample(n, rng):
            k = len(centers)
            labels = rng.integers(0, k, n)
            X = rng.standard_normal((n, 2)) * sigma
            for i in range(k):
                X[labels == i] += centers[i]
            return X
        def pdf(x):
            p = np.zeros(x.shape[0])
            for c in centers:
                p += multivariate_normal.pdf(x, c, sigma**2 * np.eye(2))
            return p / len(centers)
        return sample, pdf, "5-cluster (tight, σ=0.6)"
    
    raise ValueError(f"Unknown density: {name}")


def make_density_3d(name):
    """3D test densities."""
    
    if name == "normal":
        def sample(n, rng):
            return rng.standard_normal((n, 3))
        def pdf(x):
            return multivariate_normal.pdf(x, mean=[0,0,0])
        return sample, pdf, "3D Standard Normal"
    
    elif name == "trimodal":
        centers = np.array([[2,0,0], [-2,0,0], [0,2,0]])
        def sample(n, rng):
            k = len(centers)
            labels = rng.integers(0, k, n)
            X = rng.standard_normal((n, 3))
            for i in range(k):
                X[labels == i] += centers[i]
            return X
        def pdf(x):
            p = np.zeros(x.shape[0])
            for c in centers:
                p += multivariate_normal.pdf(x, c)
            return p / len(centers)
        return sample, pdf, "3D Trimodal (sep=4)"
    
    raise ValueError(f"Unknown density: {name}")


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(density_name, dim, n_samples, n_trials=50):
    """Run a full benchmark for one density."""
    
    if dim == 2:
        sample_fn, pdf_fn, desc = make_density_2d(density_name)
        compute_ise = compute_ise_2d
    else:
        sample_fn, pdf_fn, desc = make_density_3d(density_name)
        compute_ise = compute_ise_3d
    
    methods = {
        'Silverman': silverman_h,
        'Scott': scott_h,
        'LSCV': lscv_h,
        'GSJ-1stage': gsj_one_stage_h,
        'GSJ-2stage': gsj_two_stage_h,
    }
    
    results = {name: [] for name in methods}
    h_values = {name: [] for name in methods}
    oracle_ise_vals = []
    
    print(f"\n{'='*70}")
    print(f"  {desc} | d={dim}, n={n_samples}, trials={n_trials}")
    print(f"{'='*70}")
    
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        X = sample_fn(n_samples, rng)
        
        for name, method_fn in methods.items():
            try:
                if name == 'LSCV' and n_samples > 2000:
                    # LSCV is O(n²) and slow for large n — skip
                    h = np.nan
                    ise = np.nan
                else:
                    h = method_fn(X)
                    ise = compute_ise(X, h, pdf_fn)
            except Exception:
                h = np.nan
                ise = np.nan
            
            results[name].append(ise)
            h_values[name].append(h)
        
        if trial % 10 == 0:
            print(f"  trial {trial}/{n_trials}...", flush=True)
    
    # Report
    print(f"\n  {'Method':<12} {'Mean ISE':>12} {'Std ISE':>12} {'Mean h':>10} {'vs Oracle':>10}")
    print(f"  {'-'*58}")
    
    # Find oracle for reference (use median trial)
    rng_oracle = np.random.default_rng(0)
    X_oracle = sample_fn(n_samples, rng_oracle)
    h_oracle, ise_oracle = oracle_bandwidth(X_oracle, pdf_fn, dim)
    
    for name in methods:
        ises = np.array(results[name])
        hs = np.array(h_values[name])
        valid = ~np.isnan(ises)
        if valid.sum() == 0:
            print(f"  {name:<12} {'N/A':>12}")
            continue
        mean_ise = np.nanmean(ises)
        std_ise = np.nanstd(ises)
        mean_h = np.nanmean(hs)
        ratio = mean_ise / ise_oracle if ise_oracle > 0 else np.nan
        print(f"  {name:<12} {mean_ise:12.4e} {std_ise:12.4e} {mean_h:10.4f} {ratio:10.2f}x")
    
    print(f"  {'Oracle':<12} {ise_oracle:12.4e} {'':>12} {h_oracle:10.4f} {'1.00x':>10}")
    
    # Paired t-tests: GSJ-2stage vs each other
    print(f"\n  Paired t-tests (GSJ-2stage vs others):")
    gsj2_ises = np.array(results['GSJ-2stage'])
    
    for name in ['Silverman', 'Scott', 'LSCV', 'GSJ-1stage']:
        other_ises = np.array(results[name])
        valid = ~(np.isnan(gsj2_ises) | np.isnan(other_ises))
        if valid.sum() < 5:
            print(f"    vs {name:<10}: insufficient data")
            continue
        stat, pval = ttest_rel(gsj2_ises[valid], other_ises[valid])
        diff = np.mean(other_ises[valid]) - np.mean(gsj2_ises[valid])
        winner = "GSJ-2stage" if diff > 0 else name
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"    vs {name:<10}: diff={diff:+.4e}, t={stat:.3f}, p={pval:.4f} {sig} → {winner}")
    
    return results


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("RIGOROUS BANDWIDTH SELECTION BENCHMARK")
    print("=" * 70)
    print("Methodology: known density → sample → select h → compute ISE")
    print("             50 trials per condition, paired t-test for significance")
    print()
    
    # 2D benchmarks
    for density in ["normal", "bimodal_far", "bimodal_close", "skewed", "five_cluster"]:
        run_benchmark(density, dim=2, n_samples=500, n_trials=50)
    
    # 3D benchmarks
    for density in ["normal", "trimodal"]:
        run_benchmark(density, dim=3, n_samples=800, n_trials=30)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
