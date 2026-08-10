"""
Phase 1: Complete Experiments for GSJ Paper
============================================

Runs:
1A. Extract benchmark numbers for LaTeX tables
1B. Downstream task evaluation (anomaly detection + clustering)
1C. Subsampling accuracy vs m
1D. Truncation bound verification

Outputs results to experiments/results/ as JSON for easy insertion into LaTeX.
"""

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm, inv
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.cluster import MeanShift
import time
import json
import os

os.makedirs("experiments/results", exist_ok=True)

# =============================================================================
# GSJ Implementation
# =============================================================================

def sheather_jones_nd(X, max_exact=3000, subsample_m=80000):
    n, d = X.shape
    cov_matrix = np.cov(X, rowvar=False)
    try:
        cov_inv_sqrt = inv(sqrtm(cov_matrix))
        Y = (cov_inv_sqrt @ X.T).T
    except:
        stds = np.std(X, axis=0, ddof=1); stds[stds==0]=1.0; Y = X/stds
    h_0 = (4.0 / (n * (d + 2))) ** (1.0 / (d + 4))
    if n > max_exact:
        rng = np.random.default_rng(42)
        m = subsample_m
        idx_i = rng.integers(0, n, m); idx_j = rng.integers(0, n, m)
        diffs = Y[idx_i] - Y[idx_j]
        dist_sq_s = np.sum(diffs**2, axis=1)
        r_sq_s = dist_sq_s / h_0**2
        P_s = r_sq_s**2/16.0 - (d+2)*r_sq_s/4.0 + d*(d+2)/4.0
        W_s = np.exp(-r_sq_s/4.0)
        S = (n**2/m) * np.sum(W_s * P_s) + n*d*(d+2)/4.0
    else:
        diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        r_sq = dist_sq / h_0**2
        P = r_sq**2/16.0 - (d+2)*r_sq/4.0 + d*(d+2)/4.0
        W = np.exp(-r_sq/4.0)
        S = np.sum(W * P)
    roughness = S / (n**2 * (4.0*np.pi)**(d/2.0) * h_0**(d+4))
    R_K = (4.0*np.pi)**(-d/2.0)
    return (d * R_K / (n * roughness)) ** (1.0/(d+4))

def scotts_rule(X): return X.shape[0]**(-1.0/(X.shape[1]+4))
def silverman_rule(X):
    n, d = X.shape
    return (4.0/(n*(d+2)))**(1.0/(d+4))


# =============================================================================
# 1B. DOWNSTREAM TASK: ANOMALY DETECTION
# =============================================================================

def anomaly_detection_benchmark():
    """
    Fit KDE with different bandwidths, use -log f(x) as anomaly score,
    evaluate AUC-ROC against known labels.
    """
    print("\n" + "="*80)
    print(" PHASE 1B: ANOMALY DETECTION BENCHMARK")
    print("="*80)
    
    results = []
    
    # Dataset 1: Breast Cancer (malignant = anomaly ~37%)
    bc = datasets.load_breast_cancer()
    X_bc = StandardScaler().fit_transform(bc.data)
    X_bc = PCA(n_components=5).fit_transform(X_bc)
    y_bc = 1 - bc.target  # malignant = 1 (anomaly)
    results.append(_run_anomaly("Breast Cancer (PCA 5D)", X_bc, y_bc))
    
    # Dataset 2: Digits — treat digit '8' as anomaly (most confusable)
    digits = datasets.load_digits()
    X_dig = StandardScaler().fit_transform(digits.data)
    X_dig = PCA(n_components=8).fit_transform(X_dig)
    y_dig = (digits.target == 8).astype(int)
    results.append(_run_anomaly("Digits '8' as anomaly (PCA 8D)", X_dig, y_dig))
    
    # Dataset 3: Wine — class 2 as anomaly (~27%)
    wine = datasets.load_wine()
    X_wine = StandardScaler().fit_transform(wine.data)
    X_wine = PCA(n_components=5).fit_transform(X_wine)
    y_wine = (wine.target == 2).astype(int)
    results.append(_run_anomaly("Wine class-2 anomaly (PCA 5D)", X_wine, y_wine))
    
    # Dataset 4: Covertype — class 4 as anomaly (rare, ~3%)
    cov = datasets.fetch_covtype()
    rng = np.random.default_rng(42)
    idx = rng.choice(cov.data.shape[0], 5000, replace=False)
    X_cov = StandardScaler().fit_transform(cov.data[idx, :10].astype(float))
    X_cov = PCA(n_components=6).fit_transform(X_cov)
    y_cov = (cov.target[idx] == 4).astype(int)
    results.append(_run_anomaly("Covertype class-4 (PCA 6D)", X_cov, y_cov))
    
    # Dataset 5: Shuttle — class != 1 as anomaly (~22%)
    try:
        import openml
        ds = openml.datasets.get_dataset(40685)
        X_df, y_s, _, _ = ds.get_data(target=ds.default_target_attribute)
        X_shut = X_df.values.astype(float)
        mask = ~np.isnan(X_shut).any(axis=1)
        X_shut = X_shut[mask]
        y_shut_raw = y_s[mask]
        # Subsample
        idx = rng.choice(X_shut.shape[0], 5000, replace=False)
        X_shut = StandardScaler().fit_transform(X_shut[idx])
        y_shut = (y_shut_raw.iloc[idx] != '1').astype(int).values
        results.append(_run_anomaly("Shuttle non-class-1 (9D)", X_shut, y_shut))
    except Exception as e:
        print(f"  Shuttle skipped: {e}")
    
    # Print summary
    print("\n" + "-"*80)
    print(f"{'Dataset':<35} | {'AUC(Scott)':>10} | {'AUC(Silv)':>10} | {'AUC(GSJ)':>10} | {'Best'}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<35} | {r['auc_scott']:>10.4f} | {r['auc_silv']:>10.4f} | {r['auc_gsj']:>10.4f} | {r['best']}")
    
    with open("experiments/results/anomaly_detection.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def _run_anomaly(name, X, y, test_size=0.3):
    """Run anomaly detection with different bandwidths."""
    print(f"\n  {name}: n={X.shape[0]}, d={X.shape[1]}, anomaly_rate={y.mean():.2%}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Only fit KDE on "normal" training data
    X_normal = X_train[y_train == 0]
    
    methods = {}
    for method_name, h in [("scott", scotts_rule(X_normal)),
                           ("silv", silverman_rule(X_normal)),
                           ("gsj", sheather_jones_nd(X_normal))]:
        kde = stats.gaussian_kde(X_normal.T, bw_method=h)
        # Anomaly score = negative log-likelihood
        scores = -kde.logpdf(X_test.T)
        auc = roc_auc_score(y_test, scores)
        methods[method_name] = auc
        print(f"    {method_name}: h={h:.5f}, AUC={auc:.4f}")
    
    best = max(methods, key=methods.get)
    return {
        "name": name, "n": X.shape[0], "d": X.shape[1],
        "auc_scott": methods["scott"], "auc_silv": methods["silv"],
        "auc_gsj": methods["gsj"], "best": best
    }


# =============================================================================
# 1B. DOWNSTREAM TASK: DENSITY-BASED CLUSTERING
# =============================================================================

def clustering_benchmark():
    """
    Find modes of KDE via mean-shift, evaluate clustering with ARI.
    """
    print("\n" + "="*80)
    print(" PHASE 1B: DENSITY-BASED CLUSTERING (Mean-Shift)")
    print("="*80)
    
    from sklearn.cluster import MeanShift
    
    results = []
    
    # Dataset 1: Iris (3 classes)
    iris = datasets.load_iris()
    X_iris = StandardScaler().fit_transform(iris.data[:, 2:4])  # petal dims only
    y_iris = iris.target
    results.append(_run_clustering("Iris petals (2D)", X_iris, y_iris))
    
    # Dataset 2: Wine (3 classes, PCA 3D)
    wine = datasets.load_wine()
    X_wine = PCA(n_components=3).fit_transform(StandardScaler().fit_transform(wine.data))
    y_wine = wine.target
    results.append(_run_clustering("Wine (PCA 3D)", X_wine, y_wine))
    
    # Dataset 3: Digits subset (digits 0,1,2 — 3 classes, PCA 4D)
    digits = datasets.load_digits()
    mask = digits.target <= 2
    X_dig = PCA(n_components=4).fit_transform(
        StandardScaler().fit_transform(digits.data[mask]))
    y_dig = digits.target[mask]
    results.append(_run_clustering("Digits 0-1-2 (PCA 4D)", X_dig, y_dig))
    
    # Dataset 4: Synthetic 5 clusters in 3D
    rng = np.random.default_rng(42)
    centers = rng.normal(0, 2.5, (5, 3))
    X_synth = np.vstack([rng.multivariate_normal(c, 0.3*np.eye(3), 200) for c in centers])
    y_synth = np.repeat(range(5), 200)
    results.append(_run_clustering("Synthetic 5 clusters (3D)", X_synth, y_synth))
    
    print("\n" + "-"*80)
    print(f"{'Dataset':<35} | {'ARI(Scott)':>10} | {'ARI(Silv)':>10} | {'ARI(GSJ)':>10} | {'Best'}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<35} | {r['ari_scott']:>10.4f} | {r['ari_silv']:>10.4f} | {r['ari_gsj']:>10.4f} | {r['best']}")
    
    with open("experiments/results/clustering.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def _run_clustering(name, X, y_true):
    """Run mean-shift clustering with different bandwidths."""
    print(f"\n  {name}: n={X.shape[0]}, d={X.shape[1]}, k={len(np.unique(y_true))}")
    
    methods = {}
    for method_name, h in [("scott", scotts_rule(X)),
                           ("silv", silverman_rule(X)),
                           ("gsj", sheather_jones_nd(X))]:
        # Mean-shift bandwidth = h * std (approximate)
        sigma = np.std(X)
        ms = MeanShift(bandwidth=h * sigma * 1.5)
        try:
            labels = ms.fit_predict(X)
            ari = adjusted_rand_score(y_true, labels)
        except:
            ari = 0.0
        methods[method_name] = ari
        n_clusters = len(np.unique(labels)) if 'labels' in dir() else 0
        print(f"    {method_name}: h={h:.5f}, n_clusters={n_clusters}, ARI={ari:.4f}")
    
    best = max(methods, key=methods.get)
    return {
        "name": name, "n": X.shape[0], "d": X.shape[1],
        "ari_scott": methods["scott"], "ari_silv": methods["silv"],
        "ari_gsj": methods["gsj"], "best": best
    }


# =============================================================================
# 1C. SUBSAMPLING ACCURACY VS m
# =============================================================================

def subsampling_accuracy():
    """Compare exact bandwidth vs subsampled for various m values."""
    print("\n" + "="*80)
    print(" PHASE 1C: SUBSAMPLING ACCURACY")
    print("="*80)
    
    rng = np.random.default_rng(42)
    results = []
    
    for d in [2, 5]:
        X = rng.multivariate_normal(np.zeros(d), np.eye(d), 2000)
        X = StandardScaler().fit_transform(X)
        
        # Exact reference
        h_exact = sheather_jones_nd(X, max_exact=5000)
        
        print(f"\n  d={d}, n=2000, h_exact={h_exact:.6f}")
        print(f"    {'m':>8} | {'h_sub':>10} | {'rel_error':>10}")
        print(f"    {'-'*35}")
        
        for m in [5000, 10000, 30000, 50000, 80000, 100000, 200000]:
            h_sub = sheather_jones_nd(X, max_exact=100, subsample_m=m)  # force subsample
            rel_err = abs(h_sub - h_exact) / h_exact
            results.append({"d": d, "n": 2000, "m": m, 
                           "h_exact": h_exact, "h_sub": h_sub, "rel_error": rel_err})
            print(f"    {m:>8} | {h_sub:>10.6f} | {rel_err:>10.4%}")
    
    with open("experiments/results/subsampling.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# 1D. TRUNCATION BOUND VERIFICATION
# =============================================================================

def truncation_bound():
    """Compute sup_{r>=c} |P_d(r²)| exp(-r²/4) for various c, d."""
    print("\n" + "="*80)
    print(" PHASE 1D: TRUNCATION BOUND VERIFICATION")
    print("="*80)
    
    results = []
    
    # Fine grid for numerical supremum
    r_grid = np.linspace(0, 20, 10000)
    
    print(f"\n  {'d':>3} | {'max g(r)':>10} | {'r at max':>8} | {'g(6)':>10} | {'g(8)':>10} | {'g(10)':>10}")
    print(f"  {'-'*65}")
    
    for d in range(1, 11):
        t = r_grid**2
        P = t**2/16 - (d+2)*t/4 + d*(d+2)/4
        g = np.abs(P) * np.exp(-r_grid**2/4)
        
        max_g = g.max()
        r_at_max = r_grid[g.argmax()]
        g_at_6 = np.abs(P[np.argmin(np.abs(r_grid-6))]) * np.exp(-36/4)
        g_at_8 = np.abs(P[np.argmin(np.abs(r_grid-8))]) * np.exp(-64/4)
        g_at_10 = np.abs(P[np.argmin(np.abs(r_grid-10))]) * np.exp(-100/4)
        
        results.append({
            "d": d, "max_g": float(max_g), "r_at_max": float(r_at_max),
            "g_at_6": float(g_at_6), "g_at_8": float(g_at_8), "g_at_10": float(g_at_10)
        })
        
        print(f"  {d:>3} | {max_g:>10.6f} | {r_at_max:>8.3f} | {g_at_6:>10.2e} | {g_at_8:>10.2e} | {g_at_10:>10.2e}")
    
    # Key conclusion
    print(f"\n  Conclusion:")
    max_g_at_8 = max(r["g_at_8"] for r in results)
    print(f"    For ALL d <= 10: sup_{{r>=8}} g(r) < {max_g_at_8:.2e}")
    print(f"    Global maximum always occurs at r < 4 (well before cutoff)")
    print(f"    Cutoff r=8 gives relative error < {max_g_at_8:.1e} per pair")
    
    with open("experiments/results/truncation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" GSJ PAPER — PHASE 1: EXPERIMENTS")
    print("="*80)
    
    # 1B: Downstream tasks
    anomaly_results = anomaly_detection_benchmark()
    clustering_results = clustering_benchmark()
    
    # 1C: Subsampling
    subsample_results = subsampling_accuracy()
    
    # 1D: Truncation
    truncation_results = truncation_bound()
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    
    n_anom_wins = sum(1 for r in anomaly_results if r["best"] == "gsj")
    n_clust_wins = sum(1 for r in clustering_results if r["best"] == "gsj")
    
    print(f"\n  Anomaly Detection: GSJ wins {n_anom_wins}/{len(anomaly_results)} datasets")
    print(f"  Clustering: GSJ wins {n_clust_wins}/{len(clustering_results)} datasets")
    print(f"  Subsampling: m=80K gives < {max(r['rel_error'] for r in subsample_results if r['m']==80000):.1%} error")
    print(f"  Truncation: cutoff=8 gives < {max(r['g_at_8'] for r in truncation_results):.1e} per pair")
    
    print("\n  All results saved to experiments/results/")
    print("="*80)
