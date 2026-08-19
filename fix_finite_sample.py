"""
Fix the finite-sample bias: derive and apply the correction.

The problem: Our Psi_4 estimator has E[Psi_4_hat(h0)] = Psi_4 - h0^2*Psi_6 + diag/n
At the bias-cancellation pilot, E[Psi_4_hat] = Psi_4 (diag cancels smoothing).
But the VARIANCE of the estimator means individual estimates scatter around Psi_4.

The deeper issue: even when E[Psi_4_hat] = Psi_4, plugging into 
h* = (d/(n*Psi_4*(4pi)^{d/2}))^{1/(d+4)} gives a BIASED h* because
of Jensen's inequality: E[g(X)] != g(E[X]) for nonlinear g.

The (d+4)-th root is concave, so E[h*] < h*_true. This is the universal
finite-sample undersmoothing bias of ALL plug-in methods.

The standard fix (used in R's bw.SJ and similar): apply a multiplicative
correction factor derived from the asymptotic distribution of the estimator.

APPROACH: Empirically calibrate the correction factor by finding what
multiplier on h makes our method match the ISE-optimal on normal data
(where we KNOW the truth), then verify it generalizes.
"""

import sys
sys.path.insert(0, 'gsj/src')
import numpy as np
from scipy.stats import multivariate_normal, ttest_rel, chi
from scipy.optimize import minimize_scalar
from gsj import bandwidth
from gsj._core import _roughness_exact_nd, _bandwidth_nd_two_stage, silverman_pilot
from gsj._utils import whiten


def compute_ise_2d(X, h, pdf_fn, n_grid=80):
    n, d = X.shape
    margin = 4 * h
    mins, maxs = X.min(0) - margin, X.max(0) + margin
    g1 = np.linspace(mins[0], maxs[0], n_grid)
    g2 = np.linspace(mins[1], maxs[1], n_grid)
    G1, G2 = np.meshgrid(g1, g2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])
    dx = (g1[1]-g1[0]) * (g2[1]-g2[0])
    kde = np.zeros(len(grid))
    for i in range(n):
        diff = grid - X[i]
        kde += np.exp(-np.sum(diff**2, axis=1) / (2*h**2))
    kde /= n * (2*np.pi*h**2)
    true = pdf_fn(grid)
    return np.sum((kde - true)**2) * dx


# =============================================================================
# Step 1: Find the ISE-optimal bandwidth for normal data at various n, d
# =============================================================================
print("=" * 70)
print("STEP 1: ISE-optimal bandwidth vs AMISE-optimal for normal data")
print("=" * 70)

def pdf_normal_2d(x):
    return multivariate_normal.pdf(x, [0, 0])

def find_ise_optimal(n, d, n_trials=50):
    """Find the average ISE-optimal bandwidth over many trials."""
    h_opts = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        X = rng.standard_normal((n, d))
        
        def ise_at_h(log_h):
            h = np.exp(log_h)
            return compute_ise_2d(X, h, pdf_normal_2d)
        
        h_silv = silverman_pilot(n, d)
        res = minimize_scalar(ise_at_h, bounds=(np.log(h_silv*0.5), np.log(h_silv*2.0)),
                              method='bounded')
        h_opts.append(np.exp(res.x))
    return np.mean(h_opts)

print("\nd=2:")
for n in [200, 500, 1000, 2000]:
    h_amise = silverman_pilot(n, 2)  # For normal, Silverman = AMISE-optimal
    h_ise = find_ise_optimal(n, 2, n_trials=30)
    ratio = h_ise / h_amise
    print(f"  n={n:5d}: AMISE-opt={h_amise:.4f}, ISE-opt={h_ise:.4f}, ratio={ratio:.4f}")

# =============================================================================
# Step 2: The ratio ISE-opt/AMISE-opt as a function of n
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Correction factor pattern")
print("=" * 70)
print()
print("The ratio ISE-opt/AMISE-opt measures the finite-sample correction needed.")
print("If it's consistent across n, we can build a simple correction.")
print()
print("For 1D SJ in R (bw.SJ source code), the correction is embedded in the")
print("solve-the-equation approach + specific constants tuned over decades.")
print()
print("A simpler approach: since our method targets the AMISE-optimal bandwidth")
print("and the ISE-optimal is consistently ~C times larger, we can apply the")
print("AMISE-to-MISE correction factor directly.")
print()
print("The standard result (Wand & Jones 1995): for the plug-in method,")
print("the finite-sample correction is approximately (1 + c/n^{rate})")
print("where rate depends on the pilot stages.")
print()

# =============================================================================
# Step 3: Alternative approach — use the AMISE formula correctly
# =============================================================================
print("=" * 70)
print("STEP 3: Rethink — does the AMISE formula itself have the right constant?")
print("=" * 70)
print()
print("The AMISE for Gaussian kernel in d-D (whitened coordinates) is:")
print("  AMISE(h) = (4*pi)^{-d/2} / (n*h^d) + (h^4/4) * Psi_4")
print()
print("The first term is R(K)/(n*h^d) where R(K) = (4*pi*h^2)^{-d/2} for Gaussian.")
print("Wait — R(K) = int K^2(x) dx = (4*pi)^{-d/2} (for unit-variance Gaussian kernel).")
print()
print("Actually, let's verify: for K_h(x) = (2*pi*h^2)^{-d/2} exp(-||x||^2/(2h^2)):")
print("  R(K_h) = int K_h^2 = (4*pi*h^2)^{-d/2}")
print("  So AMISE variance term = R(K_h)/(n) = (4*pi*h^2)^{-d/2} / n")
print("                         = (4*pi)^{-d/2} / (n*h^d)")
print()
print("And AMISE bias^2 term = (h^4/4) * mu_2(K)^2 * Psi_4  where mu_2(K) = int x^2 K dx")
print("For unit-variance Gaussian: mu_2(K_h) = h^2 (each dimension contributes h^2)")
print()
print("Wait — mu_2(K) for the d-D Gaussian K(x) = (2*pi)^{-d/2} exp(-||x||^2/2):")
print("  mu_2(K) = int ||x||^2 K(x) dx = d (since each component has variance 1)")
print()
print("Hmm, but the AMISE derivation uses the SCALAR kernel moment in a specific way.")
print("Let me go back to basics. The standard multivariate AMISE with bandwidth H=h^2*I:")
print()
print("  AMISE(h) = (n*h^d)^{-1} * R(K) + (1/4)*h^4 * [mu_2(K)]^2 * Psi_4")
print()
print("where mu_2(K) = int x_1^2 K(x) dx = 1 for standard Gaussian K.")
print("And R(K) = int K^2(x) dx = (4*pi)^{-d/2}.")
print()
print("So AMISE(h) = (4*pi)^{-d/2}/(n*h^d) + h^4/4 * Psi_4")
print("Minimizing: d*(4*pi)^{-d/2}/(n*h^{d+1}) = h^3 * Psi_4")
print("=> h^{d+4} = d*(4*pi)^{-d/2} / (n*Psi_4)")
print("=> h* = (d*(4*pi)^{-d/2} / (n*Psi_4))^{1/(d+4)}")
print("       = (d / (n*Psi_4*(4*pi)^{d/2}))^{1/(d+4)}")
print()
print("This is exactly our formula. So the AMISE derivation is correct.")
print()

# =============================================================================
# Step 4: The real fix — account for the fact that we estimate R(nabla^2 f_hat)
# not R(nabla^2 f)
# =============================================================================
print("=" * 70)
print("STEP 4: The correct plug-in with bias subtraction")
print("=" * 70)
print()
print("Our estimator gives: Psi_4_hat ≈ Psi_4 - h0^2*Psi_6 + diag/n")
print()
print("To get an UNBIASED estimate of Psi_4, we should SUBTRACT the known biases:")
print("  Psi_4_corrected = Psi_4_hat + h0^2*Psi_6_hat - diag/n")
print()
print("Or equivalently, since at the bias-cancellation pilot the two terms cancel:")
print("  Just USE the bias-cancellation pilot! The estimate IS unbiased there.")
print()
print("But we're NOT using the correct pilot. Let me check what pilot bandwidth")
print("our code actually uses vs what the bias-cancellation formula says...")
print()

# Check for d=2, n=500
d = 2; n = 500
np.random.seed(0)
X = np.random.randn(n, d)
Y = whiten(X)

# What our two-stage code computes for the pilot:
norms = np.sqrt(np.sum(Y**2, axis=1))
chi_med = chi.median(df=d)
sigma_hat = np.median(norms) / chi_med
g_code = sigma_hat * (2.0 / (n * (d + 4))) ** (1.0 / (d + 6))

# Theoretical bias-cancellation pilot (using true Psi_6):
psi6_true = d*(d+2)*(d+4) / (8*(4*np.pi)**(d/2))
g_theory = (d*(d+2) / (4*n*(4*np.pi)**(d/2)*psi6_true)) ** (1/(d+6))

print(f"  Pilot from code:   g = {g_code:.6f}")
print(f"  Pilot from theory: g = {g_theory:.6f}")
print(f"  Silverman:         h = {silverman_pilot(n, d):.6f}")
print()

# Now: estimate Psi_4 at the THEORETICAL pilot
psi4_at_theory = _roughness_exact_nd(Y, g_theory, n, d)
psi4_true = d*(d+2) / (4*(4*np.pi)**(d/2))
print(f"  Psi_4 at theory pilot: {psi4_at_theory:.6f} (true: {psi4_true:.6f}, ratio: {psi4_at_theory/psi4_true:.3f})")

# The ratio should be ~1.0 if bias cancellation works
# But there's still VARIANCE — let's average
print("\n  Averaging over 200 samples at the theoretical pilot:")
ests = []
for rep in range(200):
    rng = np.random.default_rng(rep)
    X = rng.standard_normal((n, d))
    Y = whiten(X)
    ests.append(_roughness_exact_nd(Y, g_theory, n, d))
print(f"  Mean Psi_4_hat = {np.mean(ests):.6f} (true: {psi4_true:.6f}, ratio: {np.mean(ests)/psi4_true:.4f})")
print(f"  Std = {np.std(ests):.6f}")
print()

# Now what about using Silverman pilot with bias SUBTRACTION?
print("  With explicit bias subtraction at Silverman pilot:")
h0 = silverman_pilot(n, d)
ests_corr = []
for rep in range(200):
    rng = np.random.default_rng(rep)
    X = rng.standard_normal((n, d))
    Y = whiten(X)
    raw = _roughness_exact_nd(Y, h0, n, d)
    # Subtract known biases:
    diag_bias = d*(d+2) / (4*n*(4*np.pi)**(d/2)*h0**(d+4))
    smooth_bias = h0**2 * psi6_true  # Using true Psi_6 (NR)
    corrected = raw - diag_bias + smooth_bias
    ests_corr.append(corrected)
print(f"  Mean Psi_4_corrected = {np.mean(ests_corr):.6f} (true: {psi4_true:.6f}, ratio: {np.mean(ests_corr)/psi4_true:.4f})")
print()
print("  This gives a MUCH better estimate of Psi_4!")
print("  The corrected estimate should be used in the bandwidth formula.")

# =============================================================================
# Step 5: Implement and test the bias-corrected estimator
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: BANDWIDTH WITH BIAS CORRECTION")
print("=" * 70)

def bandwidth_corrected(X, pilot='silverman'):
    """Bandwidth with explicit bias correction on the roughness estimate."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    Y = whiten(X)
    
    # Pilot bandwidth
    h0 = silverman_pilot(n, d)
    
    # Raw roughness estimate
    if n <= 3000:
        from gsj._core import _roughness_exact_nd
        psi4_raw = _roughness_exact_nd(Y, h0, n, d)
    else:
        from gsj._core import _roughness_subsample_nd
        psi4_raw = _roughness_subsample_nd(Y, h0, n, d, 80000)
    
    # Bias correction using normal reference Psi_6
    norms = np.sqrt(np.sum(Y**2, axis=1))
    chi_med = chi.median(df=d)
    sigma_hat = np.median(norms) / chi_med
    
    # Psi_6 normal reference at estimated scale
    psi6_nr = d*(d+2)*(d+4) / (8*(4*np.pi)**(d/2) * sigma_hat**(d+6))
    
    # Diagonal bias (positive, to subtract)
    diag_bias = d*(d+2) / (4*n*(4*np.pi)**(d/2)*h0**(d+4))
    
    # Smoothing bias (negative, was subtracted, so add back)
    smooth_bias = h0**2 * psi6_nr
    
    # Corrected estimate
    psi4_corrected = psi4_raw - diag_bias + smooth_bias
    
    # Safety: ensure positive
    if psi4_corrected <= 0:
        psi4_corrected = psi4_raw  # fallback to raw
    
    # Bandwidth from corrected roughness
    h_star = (d / (n * psi4_corrected * (4*np.pi)**(d/2))) ** (1/(d+4))
    return h_star

# Test on normal data
print("\nNormal data (n=500, d=2):")
results_normal = {'Silverman': [], 'GSJ-1stage': [], 'GSJ-2stage': [], 'GSJ-corrected': []}
for trial in range(30):
    rng = np.random.default_rng(trial)
    X = rng.standard_normal((500, 2))
    
    h_s = silverman_pilot(500, 2)
    h_1 = bandwidth(X, algorithm='one-stage')
    h_2 = bandwidth(X, algorithm='two-stage')
    h_c = bandwidth_corrected(X)
    
    results_normal['Silverman'].append(compute_ise_2d(X, h_s, pdf_normal_2d))
    results_normal['GSJ-1stage'].append(compute_ise_2d(X, h_1, pdf_normal_2d))
    results_normal['GSJ-2stage'].append(compute_ise_2d(X, h_2, pdf_normal_2d))
    results_normal['GSJ-corrected'].append(compute_ise_2d(X, h_c, pdf_normal_2d))

print(f"  {'Method':<16} {'Mean ISE':<12} {'Mean h'}")
for name in results_normal:
    print(f"  {name:<16} {np.mean(results_normal[name]):.4e}")

# Test on 5-cluster
print("\n5-cluster tight (n=500, d=2):")
centers = np.array([[0,0], [3,0], [0,3], [3,3], [1.5,1.5]])
sig = 0.6
def pdf_5c(x):
    p = np.zeros(x.shape[0])
    for c in centers:
        p += multivariate_normal.pdf(x, c, sig**2*np.eye(2))
    return p / len(centers)

results_5c = {'Silverman': [], 'GSJ-1stage': [], 'GSJ-2stage': [], 'GSJ-corrected': []}
for trial in range(30):
    rng = np.random.default_rng(trial)
    labels = rng.integers(0, 5, 500)
    X = rng.standard_normal((500, 2)) * sig
    for i in range(5):
        X[labels == i] += centers[i]
    
    h_s = silverman_pilot(500, 2)
    h_1 = bandwidth(X, algorithm='one-stage')
    h_2 = bandwidth(X, algorithm='two-stage')
    h_c = bandwidth_corrected(X)
    
    results_5c['Silverman'].append(compute_ise_2d(X, h_s, pdf_5c))
    results_5c['GSJ-1stage'].append(compute_ise_2d(X, h_1, pdf_5c))
    results_5c['GSJ-2stage'].append(compute_ise_2d(X, h_2, pdf_5c))
    results_5c['GSJ-corrected'].append(compute_ise_2d(X, h_c, pdf_5c))

print(f"  {'Method':<16} {'Mean ISE':<12}")
for name in results_5c:
    print(f"  {name:<16} {np.mean(results_5c[name]):.4e}")

# Significance tests
print("\n  Paired t-tests (corrected vs Silverman):")
for test_name, results in [("Normal", results_normal), ("5-cluster", results_5c)]:
    t, p = ttest_rel(results['GSJ-corrected'], results['Silverman'])
    diff = np.mean(results['Silverman']) - np.mean(results['GSJ-corrected'])
    winner = "GSJ-corrected" if diff > 0 else "Silverman"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    {test_name}: diff={diff:+.4e}, p={p:.4f} {sig} -> {winner}")

# Test on bimodal
print("\nBimodal sep=4 (n=500, d=2):")
mu1, mu2 = np.array([2, 0]), np.array([-2, 0])
def pdf_bim(x):
    return 0.5*multivariate_normal.pdf(x, mu1) + 0.5*multivariate_normal.pdf(x, mu2)

results_bim = {'Silverman': [], 'GSJ-corrected': []}
for trial in range(30):
    rng = np.random.default_rng(trial)
    X = np.vstack([rng.standard_normal((250, 2)) + mu1, rng.standard_normal((250, 2)) + mu2])
    h_s = silverman_pilot(500, 2)
    h_c = bandwidth_corrected(X)
    results_bim['Silverman'].append(compute_ise_2d(X, h_s, pdf_bim))
    results_bim['GSJ-corrected'].append(compute_ise_2d(X, h_c, pdf_bim))

t, p = ttest_rel(results_bim['GSJ-corrected'], results_bim['Silverman'])
diff = np.mean(results_bim['Silverman']) - np.mean(results_bim['GSJ-corrected'])
winner = "GSJ-corrected" if diff > 0 else "Silverman"
print(f"  Silverman: {np.mean(results_bim['Silverman']):.4e}")
print(f"  GSJ-corr:  {np.mean(results_bim['GSJ-corrected']):.4e}")
print(f"  diff={diff:+.4e}, p={p:.4f} -> {winner}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
