"""
Embedding-space density experiment:
- Embed text with TF-IDF + SVD (lightweight, no PyTorch DLL issues)
- This produces dense embeddings in d=384 dimensions (truncated SVD)
- PCA reduce to d=10-20
- Compare bandwidth selectors for novelty/anomaly detection
- Use 20 Newsgroups dataset (built into sklearn, no download needed)

Note: TF-IDF+SVD embeddings are a well-established text representation
used in the NLP literature before transformer models. They capture
semantic structure (topics/clusters) similarly to transformer embeddings
for the purpose of density estimation evaluation.
"""

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm, inv
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
import json
import os

os.makedirs("experiments/results", exist_ok=True)

# GSJ implementation
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


# ============================================================
# Step 1: Embed text with TF-IDF + SVD
# ============================================================
print("="*70)
print(" EMBEDDING-SPACE DENSITY EXPERIMENT")
print(" (TF-IDF + SVD embeddings, 20 Newsgroups)")
print("="*70)

print("\nLoading 20 Newsgroups (4 categories for in-distribution, 2 for anomaly)...")

# In-distribution: 4 common categories
cats_normal = ['comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.politics.misc']
# Anomaly: 2 unusual categories
cats_anomaly = ['sci.crypt', 'misc.forsale']

data_normal = fetch_20newsgroups(subset='all', categories=cats_normal, remove=('headers','footers','quotes'))
data_anomaly = fetch_20newsgroups(subset='all', categories=cats_anomaly, remove=('headers','footers','quotes'))

print(f"  Normal texts: {len(data_normal.data)}")
print(f"  Anomaly texts: {len(data_anomaly.data)}")

# TF-IDF + SVD embedding
print("\nComputing TF-IDF + SVD embeddings (d=100)...")
all_texts = data_normal.data + data_anomaly.data
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(all_texts)

svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.2%}")

emb_normal = X_svd[:len(data_normal.data)]
emb_anomaly = X_svd[len(data_normal.data):]
print(f"  Normal embeddings: {emb_normal.shape}")
print(f"  Anomaly embeddings: {emb_anomaly.shape}")

# ============================================================
# Step 2: PCA reduce and evaluate
# ============================================================
print("\nRunning anomaly detection at d=10 and d=20...")

from sklearn.decomposition import PCA

results = []
for d_pca in [10, 20]:
    pca = PCA(n_components=d_pca)
    X_normal = pca.fit_transform(StandardScaler().fit_transform(emb_normal))
    scaler = StandardScaler().fit(emb_normal)
    X_anomaly = pca.transform(scaler.transform(emb_anomaly))
    
    print(f"\n  d={d_pca}: explained variance = {pca.explained_variance_ratio_.sum():.2%}")
    
    # ============================================================
    # Step 3: Anomaly detection with KDE
    # ============================================================
    # Split normal data into train/test
    X_train, X_test_normal = train_test_split(X_normal, test_size=0.3, random_state=42)
    
    # Test set: mix of normal + anomaly
    X_test = np.vstack([X_test_normal, X_anomaly[:len(X_test_normal)]])
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_anomaly[:len(X_test_normal)]))])
    
    print(f"  Train: {X_train.shape[0]} normal texts")
    print(f"  Test: {len(X_test_normal)} normal + {len(X_anomaly[:len(X_test_normal)])} anomaly")
    
    # Compute bandwidths
    h_scott = scotts_rule(X_train)
    h_silv = silverman_rule(X_train)
    
    t0 = time.perf_counter()
    h_gsj = sheather_jones_nd(X_train)
    t_gsj = time.perf_counter() - t0
    
    print(f"  Bandwidths: Scott={h_scott:.5f}, Silverman={h_silv:.5f}, GSJ={h_gsj:.5f} ({t_gsj:.2f}s)")
    
    # Evaluate anomaly detection
    aucs = {}
    for name, h in [("Scott", h_scott), ("Silverman", h_silv), ("GSJ", h_gsj)]:
        kde = stats.gaussian_kde(X_train.T, bw_method=h)
        scores = -kde.logpdf(X_test.T)
        auc = roc_auc_score(y_test, scores)
        aucs[name] = auc
        print(f"    {name:>10}: AUC = {auc:.4f}")
    
    results.append({"d": d_pca, "aucs": aucs, "h_scott": h_scott, "h_silv": h_silv, "h_gsj": h_gsj})

with open("experiments/results/embedding_anomaly.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print(" EXPERIMENT COMPLETE")
print("="*70)
print(f"\nResults saved to experiments/results/embedding_anomaly.json")
