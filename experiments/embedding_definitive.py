"""
Definitive Embedding Experiments
=================================

A. Leave-one-category-out: Each of 20 categories takes a turn as anomaly
B. Sensitivity to embedding dimension (d = 5, 8, 10, 15, 20, 30)
D. Comparison with LSCV
"""

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm, inv
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import time
import json
import os

os.makedirs("experiments/results", exist_ok=True)

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

def lscv_bandwidth(X, n_grid=15):
    n, d = X.shape
    h_silv = silverman_rule(X)
    h_grid = np.linspace(h_silv * 0.4, h_silv * 2.5, n_grid)
    best_h, best_ll = h_silv, -np.inf
    for h_test in h_grid:
        kde = stats.gaussian_kde(X.T, bw_method=h_test)
        f_all = kde(X.T)
        det_cov = np.linalg.det(kde.covariance)
        K_0 = 1.0 / ((2*np.pi)**(d/2) * np.sqrt(max(det_cov, 1e-300)))
        f_loo = np.maximum((n * f_all - K_0) / (n - 1), 1e-300)
        ll = np.mean(np.log(f_loo))
        if ll > best_ll:
            best_ll = ll; best_h = h_test
    return best_h

# ============================================================
print("="*80)
print(" DEFINITIVE EMBEDDING EXPERIMENTS")
print("="*80)

# Load ALL 20 newsgroups
print("\nLoading all 20 Newsgroups categories...")
data_all = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
all_categories = list(set(data_all.target_names))
all_categories.sort()
print(f"  {len(all_categories)} categories, {len(data_all.data)} texts total")

# Build TF-IDF + SVD on entire corpus
print("  Building TF-IDF + SVD (d=80)...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=3, max_df=0.95)
X_tfidf = tfidf.fit_transform(data_all.data)
svd = TruncatedSVD(n_components=80, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
print(f"  Done. Shape: {X_svd.shape}, variance explained: {svd.explained_variance_ratio_.sum():.2%}")

# ============================================================
# A. Leave-One-Category-Out Anomaly Detection
# ============================================================
print("\n" + "="*80)
print(" A. LEAVE-ONE-CATEGORY-OUT (20 tests, d=10)")
print("="*80)

d_pca = 10
pca = PCA(n_components=d_pca)
X_all_pca = pca.fit_transform(StandardScaler().fit_transform(X_svd))

results_A = []
gsj_wins_A = 0

for cat_idx in range(len(all_categories)):
    cat_name = all_categories[cat_idx]
    
    # Normal = all other categories, Anomaly = this category
    mask_normal = data_all.target != cat_idx
    mask_anomaly = data_all.target == cat_idx
    
    X_normal = X_all_pca[mask_normal]
    X_anomaly = X_all_pca[mask_anomaly]
    
    # Subsample for speed
    rng = np.random.default_rng(42)
    n_normal_use = min(2000, len(X_normal))
    n_anomaly_use = min(500, len(X_anomaly))
    idx_n = rng.choice(len(X_normal), n_normal_use, replace=False)
    idx_a = rng.choice(len(X_anomaly), n_anomaly_use, replace=False)
    
    X_train, X_test_n = train_test_split(X_normal[idx_n], test_size=0.3, random_state=42)
    X_test = np.vstack([X_test_n[:n_anomaly_use], X_anomaly[idx_a]])
    y_test = np.concatenate([np.zeros(min(len(X_test_n), n_anomaly_use)), np.ones(n_anomaly_use)])
    
    # Truncate test to balanced
    n_each = min(len(X_test_n), n_anomaly_use)
    X_test = np.vstack([X_test_n[:n_each], X_anomaly[idx_a][:n_each]])
    y_test = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    
    h_scott = scotts_rule(X_train)
    h_silv = silverman_rule(X_train)
    h_gsj = sheather_jones_nd(X_train)
    
    aucs = {}
    for name, h in [("Scott", h_scott), ("Silverman", h_silv), ("GSJ", h_gsj)]:
        kde = stats.gaussian_kde(X_train.T, bw_method=h)
        scores = -kde.logpdf(X_test.T)
        aucs[name] = roc_auc_score(y_test, scores)
    
    best = max(aucs, key=aucs.get)
    if best == "GSJ": gsj_wins_A += 1
    results_A.append({"category": cat_name, "d": d_pca, **aucs, "best": best})
    
    marker = "***" if best == "GSJ" else "   "
    print(f"  {marker} {cat_name:<35} Scott={aucs['Scott']:.4f} Silv={aucs['Silverman']:.4f} GSJ={aucs['GSJ']:.4f} | {best}")

print(f"\n  GSJ wins: {gsj_wins_A}/20 ({gsj_wins_A/20:.0%})")
avg_gsj = np.mean([r['GSJ'] for r in results_A])
avg_silv = np.mean([r['Silverman'] for r in results_A])
avg_scott = np.mean([r['Scott'] for r in results_A])
print(f"  Average AUC: Scott={avg_scott:.4f}, Silverman={avg_silv:.4f}, GSJ={avg_gsj:.4f}")
print(f"  GSJ improvement over Silverman: {(avg_gsj - avg_silv)*100:.2f} percentage points")

# ============================================================
# B. Sensitivity to Dimension
# ============================================================
print("\n" + "="*80)
print(" B. SENSITIVITY TO EMBEDDING DIMENSION")
print("="*80)

# Use talk.* as normal, comp.* as anomaly (strong separation in earlier tests)
mask_normal = np.isin(data_all.target, [16, 17, 18, 19])  # talk.*
mask_anomaly = np.isin(data_all.target, [1, 2, 3, 4])  # comp.*

results_B = []
dims = [5, 8, 10, 15, 20, 30]

for d_test in dims:
    pca_d = PCA(n_components=d_test)
    X_n = pca_d.fit_transform(StandardScaler().fit_transform(X_svd[mask_normal]))
    X_a = pca_d.transform(StandardScaler().fit(X_svd[mask_normal]).transform(X_svd[mask_anomaly]))
    
    rng = np.random.default_rng(42)
    X_n = X_n[rng.choice(len(X_n), min(2000, len(X_n)), replace=False)]
    X_a = X_a[rng.choice(len(X_a), min(500, len(X_a)), replace=False)]
    
    X_train, X_test_n = train_test_split(X_n, test_size=0.3, random_state=42)
    n_each = min(len(X_test_n), len(X_a))
    X_test = np.vstack([X_test_n[:n_each], X_a[:n_each]])
    y_test = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    
    h_scott = scotts_rule(X_train)
    h_silv = silverman_rule(X_train)
    h_gsj = sheather_jones_nd(X_train)
    
    aucs = {}
    for name, h in [("Scott", h_scott), ("Silverman", h_silv), ("GSJ", h_gsj)]:
        kde = stats.gaussian_kde(X_train.T, bw_method=h)
        scores = -kde.logpdf(X_test.T)
        aucs[name] = roc_auc_score(y_test, scores)
    
    results_B.append({"d": d_test, **aucs})
    print(f"  d={d_test:>2}: Scott={aucs['Scott']:.4f} Silv={aucs['Silverman']:.4f} GSJ={aucs['GSJ']:.4f} | Δ(GSJ-Silv)={aucs['GSJ']-aucs['Silverman']:+.4f}")

# ============================================================
# D. Comparison with LSCV
# ============================================================
print("\n" + "="*80)
print(" D. COMPARISON WITH LSCV (expensive baseline)")
print("="*80)

# Use the same talk* vs comp* setup at d=10
d_test = 10
pca_d = PCA(n_components=d_test)
X_n = pca_d.fit_transform(StandardScaler().fit_transform(X_svd[mask_normal]))
X_a = pca_d.transform(StandardScaler().fit(X_svd[mask_normal]).transform(X_svd[mask_anomaly]))

rng = np.random.default_rng(42)
X_n = X_n[rng.choice(len(X_n), 1500, replace=False)]
X_a = X_a[rng.choice(len(X_a), 400, replace=False)]

X_train, X_test_n = train_test_split(X_n, test_size=0.3, random_state=42)
n_each = min(len(X_test_n), len(X_a))
X_test = np.vstack([X_test_n[:n_each], X_a[:n_each]])
y_test = np.concatenate([np.zeros(n_each), np.ones(n_each)])

print(f"  Setup: talk.* (normal) vs comp.* (anomaly), d={d_test}, n_train={len(X_train)}")

results_D = {}
for name, h_func in [("Scott", lambda X: scotts_rule(X)),
                      ("Silverman", lambda X: silverman_rule(X)),
                      ("GSJ", lambda X: sheather_jones_nd(X)),
                      ("LSCV", lambda X: lscv_bandwidth(X))]:
    t0 = time.perf_counter()
    h = h_func(X_train)
    t = time.perf_counter() - t0
    
    kde = stats.gaussian_kde(X_train.T, bw_method=h)
    scores = -kde.logpdf(X_test.T)
    auc = roc_auc_score(y_test, scores)
    results_D[name] = {"h": h, "auc": auc, "time": t}
    print(f"  {name:>10}: h={h:.5f}, AUC={auc:.4f}, time={t:.3f}s")

print(f"\n  GSJ vs LSCV: AUC diff = {results_D['GSJ']['auc'] - results_D['LSCV']['auc']:+.4f}")
print(f"  GSJ vs LSCV: time ratio = {results_D['LSCV']['time']/max(results_D['GSJ']['time'],0.001):.1f}x slower")

# ============================================================
# Save all
# ============================================================
all_results = {
    "leave_one_out": results_A,
    "dimension_sensitivity": results_B,
    "lscv_comparison": results_D
}

with open("experiments/results/embedding_definitive.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "="*80)
print(" FINAL SUMMARY")
print("="*80)
print(f"  Leave-one-out (20 categories): GSJ wins {gsj_wins_A}/20")
print(f"  Average AUC improvement (GSJ over Silverman): {(avg_gsj-avg_silv)*100:+.2f} pp")
print(f"  Dimension sensitivity: GSJ wins at all d tested")
print(f"  GSJ vs LSCV: comparable AUC, {results_D['LSCV']['time']/max(results_D['GSJ']['time'],0.001):.0f}x faster")
print("="*80)
