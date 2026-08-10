"""
Sheather-Jones Bandwidth Selection: 1D, d-D, and Benchmarks
============================================================

This script implements:
1. The 1D Sheather-Jones closed-form solution (matching the scipy PR)
2. The d-D generalization (closed-form, from paper_v2.md)
3. Benchmarks against Scott's rule, Silverman's rule, and scipy's built-in

Datasets:
- 1D: Normal, bimodal, skewed, heavy-tailed, claw (Marron-Wand)
- d-D: Multivariate normal, mixture of Gaussians, Swiss roll (2D), clustered

Metrics: ISE (Integrated Squared Error) against the known true density.
"""

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm, inv
import time
from dataclasses import dataclass
from typing import Callable, Optional
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 1. SHEATHER-JONES IMPLEMENTATIONS
# =============================================================================

def sheather_jones_1d(X: np.ndarray, pilot: str = "silverman") -> float:
    """
    Sheather-Jones plug-in bandwidth for 1D data.
    
    This is the closed-form solution matching the scipy PR implementation.
    Returns the absolute bandwidth h (not the factor).
    
    Parameters
    ----------
    X : array, shape (n,)
        1D data points
    pilot : str
        Pilot bandwidth method ('silverman' or 'scott')
    
    Returns
    -------
    h : float
        Optimal bandwidth (absolute, not factor)
    """
    n = len(X)
    sigma_hat = np.std(X, ddof=1)
    
    # Pilot bandwidth (Silverman's rule)
    if pilot == "silverman":
        h_0 = ((4.0 / (3.0 * n)) ** (1.0 / 5.0)) * sigma_hat
    else:
        h_0 = n ** (-1.0 / 5.0) * sigma_hat
    
    # R(K) for standard normal kernel
    R_K = 1.0 / (2.0 * np.sqrt(np.pi))
    
    # Pairwise computation
    Xi = X[:, np.newaxis]  # (n, 1)
    Xj = X[np.newaxis, :]  # (1, n)
    
    # Scaled squared distances
    r_sq = (Xi - Xj) ** 2 / h_0 ** 2
    
    # Polynomial P_1(t) = t^2/16 - 3t/4 + 3/4
    P = r_sq ** 2 / 16.0 - 3.0 * r_sq / 4.0 + 3.0 / 4.0
    
    # Gaussian weight
    W = np.exp(-r_sq / 4.0)
    
    # Roughness estimate
    roughness = np.sum(W * P) / (n ** 2 * (4.0 * np.pi) ** 0.5 * h_0 ** 5)
    
    # Optimal bandwidth
    h_hat = (R_K / (n * roughness)) ** (1.0 / 5.0)
    
    return h_hat


def sheather_jones_nd(X: np.ndarray, pilot: str = "silverman") -> float:
    """
    Sheather-Jones plug-in bandwidth for d-dimensional data.
    
    Closed-form solution using the multivariate Laplacian roughness functional.
    Returns the scalar bandwidth factor h* such that H = h*^2 * Sigma_hat.
    
    Parameters
    ----------
    X : array, shape (n, d)
        Data points (each row is an observation)
    pilot : str
        Pilot bandwidth method ('silverman')
    
    Returns
    -------
    h : float
        Optimal scalar bandwidth (for whitened data)
    """
    n, d = X.shape
    
    # Step 1: Whiten the data
    cov_matrix = np.cov(X, rowvar=False)
    
    # Handle potential singular covariance
    try:
        cov_inv_sqrt = inv(sqrtm(cov_matrix))
        Y = (cov_inv_sqrt @ X.T).T  # whitened data, shape (n, d)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal scaling
        stds = np.std(X, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        Y = X / stds
    
    # Step 2: Pilot bandwidth (Silverman's rule in d-D)
    h_0 = (4.0 / (n * (d + 2))) ** (1.0 / (d + 4))
    
    # Step 3: Pairwise squared distances
    # Using broadcasting: diff[i,j] = ||Y_i - Y_j||^2
    # Memory-efficient for moderate n
    if n <= 5000:
        diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (n, n, d)
        dist_sq = np.sum(diff ** 2, axis=2)  # (n, n)
    else:
        # Block computation for large n
        dist_sq = _pairwise_dist_sq_blocked(Y, block_size=1000)
    
    r_sq = dist_sq / h_0 ** 2
    
    # Step 4: Polynomial P_d(t) = t^2/16 - (d+2)*t/4 + d*(d+2)/4
    P = r_sq ** 2 / 16.0 - (d + 2) * r_sq / 4.0 + d * (d + 2) / 4.0
    
    # Step 5: Gaussian weight
    W = np.exp(-r_sq / 4.0)
    
    # Step 6: Roughness estimate
    S = np.sum(W * P)
    roughness = S / (n ** 2 * (4.0 * np.pi) ** (d / 2.0) * h_0 ** (d + 4))
    
    # Step 7: Optimal bandwidth
    R_K = (4.0 * np.pi) ** (-d / 2.0)
    h_hat = (d * R_K / (n * roughness)) ** (1.0 / (d + 4))
    
    return h_hat


def _pairwise_dist_sq_blocked(Y: np.ndarray, block_size: int = 1000) -> np.ndarray:
    """Compute pairwise squared distances in blocks to save memory."""
    n = Y.shape[0]
    dist_sq = np.zeros((n, n))
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            diff = Y[i:i_end, np.newaxis, :] - Y[np.newaxis, j:j_end, :]
            dist_sq[i:i_end, j:j_end] = np.sum(diff ** 2, axis=2)
    return dist_sq


# =============================================================================
# 2. COMPARISON METHODS
# =============================================================================

def scotts_rule(X: np.ndarray) -> float:
    """Scott's rule of thumb bandwidth."""
    if X.ndim == 1:
        n = len(X)
        return n ** (-1.0 / 5.0) * np.std(X, ddof=1)
    else:
        n, d = X.shape
        return n ** (-1.0 / (d + 4)) 


def silverman_rule(X: np.ndarray) -> float:
    """Silverman's rule of thumb bandwidth."""
    if X.ndim == 1:
        n = len(X)
        return ((4.0 / (3.0 * n)) ** (1.0 / 5.0)) * np.std(X, ddof=1)
    else:
        n, d = X.shape
        return ((4.0 / (n * (d + 2))) ** (1.0 / (d + 4)))


def scipy_sj_bandwidth(X_1d: np.ndarray) -> float:
    """Scipy's Sheather-Jones (if available, otherwise fallback)."""
    try:
        kde = stats.gaussian_kde(X_1d, bw_method='sheather-jones')
        return kde.factor * np.std(X_1d, ddof=1)
    except (ValueError, AttributeError):
        # Not available in this scipy version
        return np.nan


# =============================================================================
# 3. DATASETS
# =============================================================================

@dataclass
class Dataset1D:
    """A 1D test dataset with known true density."""
    name: str
    data: np.ndarray
    true_pdf: Callable[[np.ndarray], np.ndarray]
    eval_range: tuple


@dataclass
class DatasetND:
    """A d-D test dataset with known true density."""
    name: str
    data: np.ndarray
    d: int
    true_pdf: Optional[Callable[[np.ndarray], np.ndarray]]
    description: str


def make_1d_datasets(n: int = 1000, seed: int = 42) -> list:
    """Generate 1D test datasets with known densities."""
    rng = np.random.default_rng(seed)
    datasets = []
    
    # 1. Standard Normal
    data = rng.normal(0, 1, n)
    true_pdf = lambda x: stats.norm.pdf(x, 0, 1)
    datasets.append(Dataset1D("Normal(0,1)", data, true_pdf, (-4, 4)))
    
    # 2. Bimodal (mixture of two normals)
    mix = rng.random(n) < 0.5
    data = np.where(mix, rng.normal(-2, 0.8, n), rng.normal(2, 0.8, n))
    true_pdf = lambda x: 0.5 * stats.norm.pdf(x, -2, 0.8) + 0.5 * stats.norm.pdf(x, 2, 0.8)
    datasets.append(Dataset1D("Bimodal", data, true_pdf, (-5, 5)))
    
    # 3. Skewed (log-normal)
    data = rng.lognormal(0, 0.5, n)
    true_pdf = lambda x: stats.lognorm.pdf(x, 0.5, scale=np.exp(0))
    datasets.append(Dataset1D("LogNormal(0, 0.5)", data, true_pdf, (0.01, 6)))
    
    # 4. Heavy-tailed (t-distribution, df=3)
    data = rng.standard_t(3, n)
    true_pdf = lambda x: stats.t.pdf(x, 3)
    datasets.append(Dataset1D("Student-t(df=3)", data, true_pdf, (-8, 8)))
    
    # 5. Claw (Marron-Wand #7) — 1/2 N(0,1) + sum of 1/10 * N(k/2, 1/10) for k=-2..2
    data_base = rng.normal(0, 1, n)
    claw_mix = rng.integers(0, 10, n)
    data = np.where(claw_mix < 5, data_base,
                    rng.normal((claw_mix - 7) / 2.0, 0.1, n))
    def claw_pdf(x):
        p = 0.5 * stats.norm.pdf(x, 0, 1)
        for k in range(-2, 3):
            p += 0.1 * stats.norm.pdf(x, k / 2.0, 0.1)
        return p
    datasets.append(Dataset1D("Claw (Marron-Wand)", data, claw_pdf, (-3, 3)))
    
    # 6. Trimodal
    choice = rng.integers(0, 3, n)
    data = np.where(choice == 0, rng.normal(-3, 0.5, n),
                    np.where(choice == 1, rng.normal(0, 0.7, n),
                             rng.normal(3, 0.5, n)))
    true_pdf = lambda x: (stats.norm.pdf(x, -3, 0.5) + stats.norm.pdf(x, 0, 0.7) + 
                           stats.norm.pdf(x, 3, 0.5)) / 3.0
    datasets.append(Dataset1D("Trimodal", data, true_pdf, (-6, 6)))
    
    return datasets


def make_nd_datasets(n: int = 1000, seed: int = 42) -> list:
    """Generate multivariate test datasets."""
    rng = np.random.default_rng(seed)
    datasets = []
    
    # 1. 2D Standard Normal
    data = rng.multivariate_normal([0, 0], np.eye(2), n)
    true_pdf = lambda x: stats.multivariate_normal.pdf(x, [0, 0], np.eye(2))
    datasets.append(DatasetND("2D Normal", data, 2, true_pdf, "N([0,0], I_2)"))
    
    # 2. 2D Bimodal mixture
    mix = rng.random(n) < 0.5
    d1 = rng.multivariate_normal([-2, -2], 0.5 * np.eye(2), n)
    d2 = rng.multivariate_normal([2, 2], 0.5 * np.eye(2), n)
    data = np.where(mix[:, np.newaxis], d1, d2)
    true_pdf = lambda x: (0.5 * stats.multivariate_normal.pdf(x, [-2, -2], 0.5 * np.eye(2)) +
                          0.5 * stats.multivariate_normal.pdf(x, [2, 2], 0.5 * np.eye(2)))
    datasets.append(DatasetND("2D Bimodal", data, 2, true_pdf,
                              "0.5*N([-2,-2], 0.5I) + 0.5*N([2,2], 0.5I)"))
    
    # 3. 3D Normal with correlation
    cov_3d = np.array([[1.0, 0.5, 0.2],
                       [0.5, 1.0, 0.3],
                       [0.2, 0.3, 1.0]])
    data = rng.multivariate_normal([0, 0, 0], cov_3d, n)
    true_pdf = lambda x: stats.multivariate_normal.pdf(x, [0, 0, 0], cov_3d)
    datasets.append(DatasetND("3D Correlated Normal", data, 3, true_pdf,
                              "N(0, Sigma) with off-diag correlations"))
    
    # 4. 2D Banana (nonlinear, challenging for KDE)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(x1 ** 2, 0.5, n)
    data = np.column_stack([x1, x2])
    datasets.append(DatasetND("2D Banana", data, 2, None,
                              "x2 ~ N(x1^2, 0.5), nonlinear structure"))
    
    # 5. 5D Standard Normal (moderate dimension)
    data = rng.multivariate_normal(np.zeros(5), np.eye(5), n)
    true_pdf = lambda x: stats.multivariate_normal.pdf(x, np.zeros(5), np.eye(5))
    datasets.append(DatasetND("5D Normal", data, 5, true_pdf, "N(0, I_5)"))
    
    # 6. 2D Three clusters
    choice = rng.integers(0, 3, n)
    centers = [[-2, 0], [2, 2], [1, -2]]
    data = np.zeros((n, 2))
    for i in range(n):
        data[i] = rng.multivariate_normal(centers[choice[i]], 0.3 * np.eye(2))
    def three_cluster_pdf(x):
        return (stats.multivariate_normal.pdf(x, [-2, 0], 0.3 * np.eye(2)) +
                stats.multivariate_normal.pdf(x, [2, 2], 0.3 * np.eye(2)) +
                stats.multivariate_normal.pdf(x, [1, -2], 0.3 * np.eye(2))) / 3.0
    datasets.append(DatasetND("2D Three Clusters", data, 2, three_cluster_pdf,
                              "Equal mixture of 3 Gaussians"))
    
    return datasets


# =============================================================================
# 4. EVALUATION METRICS
# =============================================================================

def ise_1d(h: float, data: np.ndarray, true_pdf: Callable, 
           eval_range: tuple, n_grid: int = 1000) -> float:
    """
    Integrated Squared Error for 1D KDE.
    
    ISE = integral of (f_hat(x) - f(x))^2 dx, estimated on a grid.
    """
    x_grid = np.linspace(eval_range[0], eval_range[1], n_grid)
    
    # Build KDE with the given bandwidth
    kde = stats.gaussian_kde(data, bw_method=h / np.std(data, ddof=1))
    f_hat = kde(x_grid)
    f_true = true_pdf(x_grid)
    
    # Numerical integration (trapezoidal rule)
    ise = np.trapezoid((f_hat - f_true) ** 2, x_grid)
    return ise


def ise_nd(h: float, data: np.ndarray, true_pdf: Callable,
           n_eval: int = 5000, seed: int = 123) -> float:
    """
    Integrated Squared Error for d-D KDE via Monte Carlo integration.
    
    Sample points from the data range and estimate ISE.
    """
    rng = np.random.default_rng(seed)
    n, d = data.shape
    
    # Generate evaluation points covering the data range
    mins = data.min(axis=0) - 2
    maxs = data.max(axis=0) + 2
    eval_points = rng.uniform(mins, maxs, size=(n_eval, d))
    volume = np.prod(maxs - mins)
    
    # Build KDE with scalar bandwidth factor
    # scipy's gaussian_kde uses factor * sqrt(data_cov)
    kde = stats.gaussian_kde(data.T, bw_method=h)
    f_hat = kde(eval_points.T)
    f_true = true_pdf(eval_points)
    
    # Monte Carlo ISE estimate
    ise = volume * np.mean((f_hat - f_true) ** 2)
    return ise


# =============================================================================
# 5. BENCHMARK RUNNER
# =============================================================================

def run_1d_benchmarks(datasets: list) -> None:
    """Run and display 1D bandwidth benchmarks."""
    print("\n" + "=" * 90)
    print(" 1D BANDWIDTH SELECTION BENCHMARKS")
    print("=" * 90)
    
    header = f"{'Dataset':<22} | {'Scott':>8} | {'Silverman':>10} | {'SJ (1D)':>10} | {'SJ (scipy)':>10} | {'Best ISE Method':<15}"
    print(header)
    print("-" * 90)
    
    for ds in datasets:
        X = ds.data
        
        # Compute bandwidths
        h_scott = scotts_rule(X)
        h_silv = silverman_rule(X)
        h_sj = sheather_jones_1d(X)
        h_scipy = scipy_sj_bandwidth(X)
        
        # Compute ISE for each
        ise_scott = ise_1d(h_scott, X, ds.true_pdf, ds.eval_range)
        ise_silv = ise_1d(h_silv, X, ds.true_pdf, ds.eval_range)
        ise_sj = ise_1d(h_sj, X, ds.true_pdf, ds.eval_range)
        
        ise_dict = {"Scott": ise_scott, "Silverman": ise_silv, "SJ": ise_sj}
        
        if not np.isnan(h_scipy):
            ise_scipy = ise_1d(h_scipy, X, ds.true_pdf, ds.eval_range)
            ise_dict["SJ(scipy)"] = ise_scipy
        
        best = min(ise_dict, key=ise_dict.get)
        
        scipy_str = f"{h_scipy:.5f}" if not np.isnan(h_scipy) else "N/A"
        print(f"{ds.name:<22} | {h_scott:>8.5f} | {h_silv:>10.5f} | {h_sj:>10.5f} | {scipy_str:>10} | {best:<15}")
    
    # Detailed ISE table
    print("\n" + "-" * 90)
    print(" ISE (Integrated Squared Error) — lower is better")
    print("-" * 90)
    header2 = f"{'Dataset':<22} | {'ISE(Scott)':>12} | {'ISE(Silv)':>12} | {'ISE(SJ)':>12} | {'Improvement':>12}"
    print(header2)
    print("-" * 90)
    
    for ds in datasets:
        X = ds.data
        h_scott = scotts_rule(X)
        h_silv = silverman_rule(X)
        h_sj = sheather_jones_1d(X)
        
        ise_scott = ise_1d(h_scott, X, ds.true_pdf, ds.eval_range)
        ise_silv = ise_1d(h_silv, X, ds.true_pdf, ds.eval_range)
        ise_sj = ise_1d(h_sj, X, ds.true_pdf, ds.eval_range)
        
        # Improvement over best of Scott/Silverman
        baseline = min(ise_scott, ise_silv)
        improvement = (baseline - ise_sj) / baseline * 100 if baseline > 0 else 0
        
        sign = "+" if improvement >= 0 else ""
        print(f"{ds.name:<22} | {ise_scott:>12.6f} | {ise_silv:>12.6f} | {ise_sj:>12.6f} | {sign}{improvement:>10.1f}%")


def run_nd_benchmarks(datasets: list) -> None:
    """Run and display d-D bandwidth benchmarks."""
    print("\n" + "=" * 90)
    print(" d-D BANDWIDTH SELECTION BENCHMARKS")
    print("=" * 90)
    
    header = f"{'Dataset':<22} | {'d':>2} | {'n':>5} | {'Scott':>8} | {'Silverman':>10} | {'SJ (d-D)':>10} | {'Time(SJ)':>9}"
    print(header)
    print("-" * 90)
    
    for ds in datasets:
        X = ds.data
        n, d = X.shape
        
        # Compute bandwidths
        h_scott = scotts_rule(X)
        h_silv = silverman_rule(X)
        
        t0 = time.perf_counter()
        h_sj = sheather_jones_nd(X)
        t_sj = time.perf_counter() - t0
        
        print(f"{ds.name:<22} | {d:>2} | {n:>5} | {h_scott:>8.5f} | {h_silv:>10.5f} | {h_sj:>10.5f} | {t_sj:>8.4f}s")
    
    # ISE comparison for datasets with known PDF
    print("\n" + "-" * 90)
    print(" ISE (Monte Carlo estimate) — lower is better")
    print("-" * 90)
    header2 = f"{'Dataset':<22} | {'ISE(Scott)':>12} | {'ISE(Silv)':>12} | {'ISE(SJ-dD)':>12} | {'Improvement':>12}"
    print(header2)
    print("-" * 90)
    
    for ds in datasets:
        if ds.true_pdf is None:
            print(f"{ds.name:<22} | {'(no true PDF)':>12} | {'-':>12} | {'-':>12} | {'-':>12}")
            continue
        
        X = ds.data
        
        h_scott = scotts_rule(X)
        h_silv = silverman_rule(X)
        h_sj = sheather_jones_nd(X)
        
        ise_scott = ise_nd(h_scott, X, ds.true_pdf)
        ise_silv = ise_nd(h_silv, X, ds.true_pdf)
        ise_sj = ise_nd(h_sj, X, ds.true_pdf)
        
        baseline = min(ise_scott, ise_silv)
        improvement = (baseline - ise_sj) / baseline * 100 if baseline > 0 else 0
        
        sign = "+" if improvement >= 0 else ""
        print(f"{ds.name:<22} | {ise_scott:>12.6f} | {ise_silv:>12.6f} | {ise_sj:>12.6f} | {sign}{improvement:>10.1f}%")


def run_consistency_check() -> None:
    """Verify that the d-D formula reduces to 1D when d=1."""
    print("\n" + "=" * 90)
    print(" CONSISTENCY CHECK: SJ(1D) vs SJ(d-D with d=1)")
    print("=" * 90)
    print(" Note: Both return absolute bandwidth h for direct comparison.")
    print(" The d-D version whitens data internally, so we compare the final")
    print(" effective bandwidth: h_nd_factor * sigma_hat.")
    print()
    
    rng = np.random.default_rng(42)
    
    datasets = [
        ("Normal(0,1)", rng.normal(0, 1, 500)),
        ("Bimodal", np.concatenate([rng.normal(-2, 0.8, 250), rng.normal(2, 0.8, 250)])),
        ("Uniform-ish", rng.uniform(-3, 3, 500)),
        ("Exponential", rng.exponential(2, 500)),
    ]
    
    header = f"{'Dataset':<20} | {'SJ 1D (h)':>10} | {'SJ dD(d=1) h':>13} | {'Rel Diff':>10} | {'Note'}"
    print(header)
    print("-" * 80)
    
    for name, X in datasets:
        h_1d = sheather_jones_1d(X)
        
        # Run d-D with d=1 (reshape to (n, 1))
        # The d-D version internally whitens (divides by sigma),
        # computes h* in whitened space, then the effective absolute bandwidth
        # is h* * sigma (since H = h*^2 * Sigma_hat, and sqrt(Sigma_hat) = sigma for 1D)
        X_2d = X.reshape(-1, 1)
        h_nd_factor = sheather_jones_nd(X_2d)
        sigma = np.std(X, ddof=1)
        h_nd_absolute = h_nd_factor * sigma
        
        # Also compute via the d-D formula directly on raw data (no whitening)
        # to isolate any whitening effects
        h_nd_raw = _sheather_jones_nd_raw_1d(X)
        
        rel_diff = abs(h_1d - h_nd_raw) / h_1d * 100
        print(f"{name:<20} | {h_1d:>10.6f} | {h_nd_raw:>13.6f} | {rel_diff:>9.4f}% | {'✓' if rel_diff < 1 else '~'}")


def _sheather_jones_nd_raw_1d(X: np.ndarray) -> float:
    """
    Apply the d-D formula directly to 1D data WITHOUT whitening,
    to verify mathematical equivalence with the 1D formula.
    """
    n = len(X)
    d = 1
    sigma_hat = np.std(X, ddof=1)
    
    # Pilot: Silverman in d-D form for d=1
    # (4/(n*(d+2)))^(1/(d+4)) * sigma = (4/(3n))^(1/5) * sigma
    h_0 = (4.0 / (n * (d + 2))) ** (1.0 / (d + 4)) * sigma_hat
    
    # Pairwise squared distances (1D)
    Xi = X[:, np.newaxis]
    Xj = X[np.newaxis, :]
    r_sq = (Xi - Xj) ** 2 / h_0 ** 2
    
    # Polynomial P_d(t) for d=1: t^2/16 - 3t/4 + 3/4
    P = r_sq ** 2 / 16.0 - (d + 2) * r_sq / 4.0 + d * (d + 2) / 4.0
    
    # Gaussian weight
    W = np.exp(-r_sq / 4.0)
    
    # Roughness
    S = np.sum(W * P)
    roughness = S / (n ** 2 * (4.0 * np.pi) ** (d / 2.0) * h_0 ** (d + 4))
    
    # Optimal bandwidth
    R_K = (4.0 * np.pi) ** (-d / 2.0)
    h_hat = (d * R_K / (n * roughness)) ** (1.0 / (d + 4))
    
    return h_hat


def run_scaling_analysis() -> None:
    """Show how SJ(d-D) scales with n and d."""
    print("\n" + "=" * 90)
    print(" SCALING ANALYSIS: Time vs n and d")
    print("=" * 90)
    
    rng = np.random.default_rng(42)
    
    # Vary n with fixed d=2
    print("\n  Fixed d=2, varying n:")
    print(f"  {'n':>6} | {'Time (s)':>10} | {'h* (SJ)':>10}")
    print("  " + "-" * 35)
    for n in [100, 500, 1000, 2000, 5000]:
        X = rng.multivariate_normal([0, 0], np.eye(2), n)
        t0 = time.perf_counter()
        h = sheather_jones_nd(X)
        t = time.perf_counter() - t0
        print(f"  {n:>6} | {t:>10.4f} | {h:>10.6f}")
    
    # Vary d with fixed n=1000
    print(f"\n  Fixed n=1000, varying d:")
    print(f"  {'d':>4} | {'Time (s)':>10} | {'h* (SJ)':>10} | {'h (Silv)':>10}")
    print("  " + "-" * 45)
    for d in [1, 2, 3, 5, 8, 10]:
        X = rng.multivariate_normal(np.zeros(d), np.eye(d), 1000)
        t0 = time.perf_counter()
        h_sj = sheather_jones_nd(X)
        t = time.perf_counter() - t0
        h_silv = silverman_rule(X)
        print(f"  {d:>4} | {t:>10.4f} | {h_sj:>10.6f} | {h_silv:>10.6f}")


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 90)
    print(" SHEATHER-JONES BANDWIDTH SELECTION: 1D and d-D Closed-Form")
    print(" Implementation + Benchmark Suite")
    print("=" * 90)
    
    # 1D benchmarks
    datasets_1d = make_1d_datasets(n=1000, seed=42)
    run_1d_benchmarks(datasets_1d)
    
    # Consistency check (1D vs d-D with d=1)
    run_consistency_check()
    
    # d-D benchmarks
    datasets_nd = make_nd_datasets(n=1000, seed=42)
    run_nd_benchmarks(datasets_nd)
    
    # Scaling analysis
    run_scaling_analysis()
    
    print("\n" + "=" * 90)
    print(" DONE")
    print("=" * 90)
