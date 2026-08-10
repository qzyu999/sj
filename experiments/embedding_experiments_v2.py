"""
Extended Embedding-Space Density Experiments
=============================================

Multiple scenarios testing GSJ on text embeddings:
1. Topic anomaly: one topic is OOD among several others
2. Domain shift: train on one domain, test on another
3. Novelty detection: rare category within a broad corpus
4. Embedding-space HOLL: held-out log-likelihood on text embeddings

All use TF-IDF + SVD (lightweight, no PyTorch DLL needed).
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

# GSJ
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

def held_out_loglik(X, h, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lls = []
    for tr, te in kf.split(X):
        kde = stats.gaussian_kde(X[tr].T, bw_method=h)
        d = np.maximum(kde(X[te].T), 1e-300)
        lls.append(np.mean(np.log(d)))
    return np.mean(lls)


def embed_newsgroups(categories, max_features=5000, svd_dim=100):
    """Fetch and embed newsgroups categories."""
    data = fetch_20newsgroups(subset='all', categories=categories,
                              remove=('headers', 'footers', 'quotes'))
    return data.data, data.target, data.target_names


def build_embeddings(texts_list, max_features=5000, svd_dim=100):
    """Build TF-IDF + SVD embeddings for a list of text collections."""
    all_texts = []
    boundaries = [0]
    for texts in texts_list:
        all_texts.extend(texts)
        boundaries.append(len(all_texts))
    
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english',
                            min_df=3, max_df=0.95)
    X_tfidf = tfidf.fit_transform(all_texts)
    
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    
    # Split back
    result = []
    for i in range(len(texts_list)):
        result.append(X_svd[boundaries[i]:boundaries[i+1]])
    
    return result, svd.explained_variance_ratio_.sum()


# ============================================================
print("="*80)
print(" EXTENDED EMBEDDING-SPACE EXPERIMENTS")
print("="*80)

all_results = []

# ============================================================
# Experiment 1: Single-topic anomaly detection
# Each topic takes turns being the "anomaly" among 3 normal topics
# ============================================================
print("\n" + "="*80)
print(" EXPERIMENT 1: Single-Topic Anomaly Detection")
print("="*80)

topic_groups = [
    ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
    ['rec.autos', 'rec.motorcycles', 'rec.sport.hockey'],
    ['sci.electronics', 'sci.med', 'sci.space'],
    ['talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc'],
]

exp1_results = []
for group_idx, normal_cats in enumerate(topic_groups):
    # Use one category from DIFFERENT group as anomaly
    anomaly_groups = [g for i, g in enumerate(topic_groups) if i != group_idx]
    anomaly_cat = anomaly_groups[0][0]  # First cat from next group
    
    texts_normal, _, _ = embed_newsgroups(normal_cats)
    texts_anomaly, _, _ = embed_newsgroups([anomaly_cat])
    
    [emb_normal, emb_anomaly], var_exp = build_embeddings(
        [texts_normal, texts_anomaly], svd_dim=80)
    
    for d_pca in [8, 15]:
        pca = PCA(n_components=d_pca)
        X_n = pca.fit_transform(StandardScaler().fit_transform(emb_normal))
        X_a = pca.transform(StandardScaler().fit(emb_normal).transform(emb_anomaly))
        
        X_train, X_test_n = train_test_split(X_n, test_size=0.3, random_state=42)
        n_anom = min(len(X_a), len(X_test_n))
        X_test = np.vstack([X_test_n[:n_anom], X_a[:n_anom]])
        y_test = np.concatenate([np.zeros(n_anom), np.ones(n_anom)])
        
        h_scott = scotts_rule(X_train)
        h_silv = silverman_rule(X_train)
        h_gsj = sheather_jones_nd(X_train)
        
        aucs = {}
        for name, h in [("Scott", h_scott), ("Silverman", h_silv), ("GSJ", h_gsj)]:
            kde = stats.gaussian_kde(X_train.T, bw_method=h)
            scores = -kde.logpdf(X_test.T)
            auc = roc_auc_score(y_test, scores)
            aucs[name] = auc
        
        desc = f"Normal={normal_cats[0].split('.')[0]}* | Anomaly={anomaly_cat}"
        exp1_results.append({"desc": desc, "d": d_pca, **aucs})
        best = max(aucs, key=aucs.get)
        print(f"  d={d_pca:>2} | {desc:<55} | Scott={aucs['Scott']:.4f} Silv={aucs['Silverman']:.4f} GSJ={aucs['GSJ']:.4f} | {best}")

all_results.append({"experiment": "single_topic_anomaly", "results": exp1_results})

# ============================================================
# Experiment 2: Domain Shift Detection
# Train on one broad domain, test on another
# ============================================================
print("\n" + "="*80)
print(" EXPERIMENT 2: Domain Shift Detection")
print("="*80)

domain_pairs = [
    ({'train': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'],
      'shift': ['sci.electronics', 'sci.med', 'sci.space', 'sci.crypt']},
     "Computers → Science"),
    ({'train': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
      'shift': ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']},
     "Recreation → Politics/Religion"),
]

exp2_results = []
for pair, desc in domain_pairs:
    texts_train, _, _ = embed_newsgroups(pair['train'])
    texts_shift, _, _ = embed_newsgroups(pair['shift'])
    
    [emb_train, emb_shift], var_exp = build_embeddings(
        [texts_train, texts_shift], svd_dim=80)
    
    for d_pca in [8, 15]:
        pca = PCA(n_components=d_pca)
        X_tr = pca.fit_transform(StandardScaler().fit_transform(emb_train))
        X_sh = pca.transform(StandardScaler().fit(emb_train).transform(emb_shift))
        
        # Train KDE on in-domain, test on mix of in-domain + shifted
        X_fit, X_test_id = train_test_split(X_tr, test_size=0.3, random_state=42)
        n_test = min(len(X_test_id), len(X_sh))
        X_test = np.vstack([X_test_id[:n_test], X_sh[:n_test]])
        y_test = np.concatenate([np.zeros(n_test), np.ones(n_test)])
        
        h_scott = scotts_rule(X_fit)
        h_silv = silverman_rule(X_fit)
        h_gsj = sheather_jones_nd(X_fit)
        
        aucs = {}
        for name, h in [("Scott", h_scott), ("Silverman", h_silv), ("GSJ", h_gsj)]:
            kde = stats.gaussian_kde(X_fit.T, bw_method=h)
            scores = -kde.logpdf(X_test.T)
            auc = roc_auc_score(y_test, scores)
            aucs[name] = auc
        
        exp2_results.append({"desc": desc, "d": d_pca, **aucs})
        best = max(aucs, key=aucs.get)
        print(f"  d={d_pca:>2} | {desc:<30} | Scott={aucs['Scott']:.4f} Silv={aucs['Silverman']:.4f} GSJ={aucs['GSJ']:.4f} | {best}")

all_results.append({"experiment": "domain_shift", "results": exp2_results})

# ============================================================
# Experiment 3: Held-Out Log-Likelihood on Embeddings
# Which bandwidth gives best predictive density?
# ============================================================
print("\n" + "="*80)
print(" EXPERIMENT 3: Held-Out Log-Likelihood on Text Embeddings")
print("="*80)

cat_collections = [
    (['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'], "Computers (3 cats)"),
    (['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], "Recreation (4 cats)"),
    (['sci.electronics', 'sci.med', 'sci.space'], "Science (3 cats)"),
    (['alt.atheism', 'soc.religion.christian', 'talk.religion.misc'], "Religion (3 cats)"),
]

exp3_results = []
for cats, desc in cat_collections:
    texts, _, _ = embed_newsgroups(cats)
    [emb], var_exp = build_embeddings([texts], svd_dim=80)
    
    for d_pca in [8, 12]:
        X = PCA(n_components=d_pca).fit_transform(StandardScaler().fit_transform(emb))
        
        h_scott = scotts_rule(X)
        h_silv = silverman_rule(X)
        h_gsj = sheather_jones_nd(X)
        
        holls = {}
        for name, h in [("Scott", h_scott), ("Silverman", h_silv), ("GSJ", h_gsj)]:
            holls[name] = held_out_loglik(X, h)
        
        exp3_results.append({"desc": desc, "d": d_pca, **holls})
        best = max(holls, key=holls.get)
        print(f"  d={d_pca:>2} | {desc:<25} | Scott={holls['Scott']:.4f} Silv={holls['Silverman']:.4f} GSJ={holls['GSJ']:.4f} | {best}")

all_results.append({"experiment": "holl_embeddings", "results": exp3_results})

# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print(" SUMMARY")
print("="*80)

# Count wins
total_exp1 = len(exp1_results)
gsj_wins_1 = sum(1 for r in exp1_results if max({"S": r["Scott"], "V": r["Silverman"], "G": r["GSJ"]}, key={"S": r["Scott"], "V": r["Silverman"], "G": r["GSJ"]}.get) == "G")

total_exp2 = len(exp2_results)
gsj_wins_2 = sum(1 for r in exp2_results if max({"S": r["Scott"], "V": r["Silverman"], "G": r["GSJ"]}, key={"S": r["Scott"], "V": r["Silverman"], "G": r["GSJ"]}.get) == "G")

total_exp3 = len(exp3_results)
gsj_wins_3 = sum(1 for r in exp3_results if max({"S": r["Scott"], "V": r["Silverman"], "G": r["GSJ"]}, key={"S": r["Scott"], "V": r["Silverman"], "G": r["GSJ"]}.get) == "G")

print(f"\n  Topic Anomaly Detection: GSJ wins {gsj_wins_1}/{total_exp1}")
print(f"  Domain Shift Detection:  GSJ wins {gsj_wins_2}/{total_exp2}")
print(f"  Embedding HOLL:          GSJ wins {gsj_wins_3}/{total_exp3}")
print(f"\n  Total: GSJ wins {gsj_wins_1+gsj_wins_2+gsj_wins_3}/{total_exp1+total_exp2+total_exp3}")

with open("experiments/results/embedding_experiments_v2.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n  Results saved to experiments/results/embedding_experiments_v2.json")
print("="*80)
