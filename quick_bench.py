"""Quick benchmark after sigma_hat fix."""
import sys
sys.path.insert(0, 'gsj/src')
import numpy as np
from scipy.stats import multivariate_normal, ttest_rel
from gsj import bandwidth

def silverman_h(n, d):
    return (4 / (n * (d + 2))) ** (1 / (d + 4))

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

# ============================================================
# Test 1: Standard Normal (Silverman is theoretically optimal)
# ============================================================
print("=" * 60)
print("TEST 1: 2D Standard Normal (n=500, 30 trials)")
print("  Silverman should be best here")
print("=" * 60)

def pdf_normal(x):
    return multivariate_normal.pdf(x, [0, 0])

results = {'Silverman': [], 'GSJ-1stage': [], 'GSJ-2stage': []}
for trial in range(30):
    rng = np.random.default_rng(trial)
    X = rng.standard_normal((500, 2))
    h_s = silverman_h(500, 2)
    h_1 = bandwidth(X, algorithm='one-stage')
    h_2 = bandwidth(X, algorithm='two-stage')
    results['Silverman'].append(compute_ise_2d(X, h_s, pdf_normal))
    results['GSJ-1stage'].append(compute_ise_2d(X, h_1, pdf_normal))
    results['GSJ-2stage'].append(compute_ise_2d(X, h_2, pdf_normal))

for name in results:
    print(f"  {name:<14}: mean ISE = {np.mean(results[name]):.4e}")
t, p = ttest_rel(results['GSJ-2stage'], results['Silverman'])
print(f"  GSJ-2stage vs Silverman: t={t:.3f}, p={p:.4f}")

# ============================================================
# Test 2: Bimodal (sep=4) — where adaptive methods SHOULD win
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: 2D Bimodal sep=4 (n=500, 30 trials)")
print("  Adaptive methods should win here")
print("=" * 60)

mu1, mu2 = np.array([2, 0]), np.array([-2, 0])
def pdf_bimodal(x):
    return 0.5*multivariate_normal.pdf(x, mu1) + 0.5*multivariate_normal.pdf(x, mu2)

results2 = {'Silverman': [], 'GSJ-1stage': [], 'GSJ-2stage': []}
for trial in range(30):
    rng = np.random.default_rng(trial)
    X = np.vstack([rng.standard_normal((250, 2)) + mu1, rng.standard_normal((250, 2)) + mu2])
    h_s = silverman_h(500, 2)
    h_1 = bandwidth(X, algorithm='one-stage')
    h_2 = bandwidth(X, algorithm='two-stage')
    results2['Silverman'].append(compute_ise_2d(X, h_s, pdf_bimodal))
    results2['GSJ-1stage'].append(compute_ise_2d(X, h_1, pdf_bimodal))
    results2['GSJ-2stage'].append(compute_ise_2d(X, h_2, pdf_bimodal))

for name in results2:
    print(f"  {name:<14}: mean ISE = {np.mean(results2[name]):.4e}")
t, p = ttest_rel(results2['GSJ-2stage'], results2['Silverman'])
diff = np.mean(results2['Silverman']) - np.mean(results2['GSJ-2stage'])
print(f"  GSJ-2stage vs Silverman: t={t:.3f}, p={p:.4f}, diff={diff:+.4e}")

# ============================================================
# Test 3: 5-cluster tight (sigma=0.6)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: 2D 5-cluster tight sigma=0.6 (n=500, 30 trials)")
print("  Strongly multimodal — adaptive should dominate")
print("=" * 60)

centers = np.array([[0,0], [3,0], [0,3], [3,3], [1.5,1.5]])
sig = 0.6
def pdf_5cluster(x):
    p = np.zeros(x.shape[0])
    for c in centers:
        p += multivariate_normal.pdf(x, c, sig**2 * np.eye(2))
    return p / len(centers)

results3 = {'Silverman': [], 'GSJ-1stage': [], 'GSJ-2stage': []}
for trial in range(30):
    rng = np.random.default_rng(trial)
    labels = rng.integers(0, 5, 500)
    X = rng.standard_normal((500, 2)) * sig
    for i in range(5):
        X[labels == i] += centers[i]
    h_s = silverman_h(500, 2)
    h_1 = bandwidth(X, algorithm='one-stage')
    h_2 = bandwidth(X, algorithm='two-stage')
    results3['Silverman'].append(compute_ise_2d(X, h_s, pdf_5cluster))
    results3['GSJ-1stage'].append(compute_ise_2d(X, h_1, pdf_5cluster))
    results3['GSJ-2stage'].append(compute_ise_2d(X, h_2, pdf_5cluster))

for name in results3:
    print(f"  {name:<14}: mean ISE = {np.mean(results3[name]):.4e}")
t, p = ttest_rel(results3['GSJ-2stage'], results3['Silverman'])
diff = np.mean(results3['Silverman']) - np.mean(results3['GSJ-2stage'])
print(f"  GSJ-2stage vs Silverman: t={t:.3f}, p={p:.4f}, diff={diff:+.4e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
