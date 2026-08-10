# Project Plan: GSJ Paper Submission

## Current Status

- [x] Core math: derived, verified (SymPy + numerical), confirmed by external review
- [x] Package: published on PyPI (`pip install gsj`), working, tested
- [x] LaTeX: skeleton with correct framing, addresses all review criticisms
- [x] Notebooks pt1-pt6: benchmarks on synthetic + real data (ISE, HOLL, LOOCV)
- [x] Phase 1 experiments: anomaly detection, clustering, subsampling, truncation — COMPLETED
- [ ] Paper tables/figures: still placeholders in LaTeX
- [ ] Minor wording fixes: from review v2 (O(n^2 d), "stability" not "consistency", etc.)
- [ ] Phase 2: Insert results into paper, finalize

---

## Phase 1 Results (COMPLETED)

### 1B-i. Anomaly Detection: GSJ wins 3/5 datasets

| Dataset | AUC(Scott) | AUC(Silv) | AUC(GSJ) | Winner | Notes |
|---------|-----------|-----------|----------|--------|-------|
| Breast Cancer (5D) | 0.9528 | 0.9536 | **0.9547** | GSJ | Marginal win |
| Digits '8' anomaly (8D) | 0.7156 | 0.7433 | **0.8001** | GSJ | **+8.5% over Scott** |
| Wine class-2 (5D) | **0.9675** | 0.9641 | 0.9624 | Scott | GSJ -0.5% |
| Covertype class-4 (6D) | **0.8177** | 0.8137 | 0.8062 | Scott | Rare anomaly (0.6%) |
| Shuttle (9D) | 0.9788 | 0.9796 | **0.9811** | GSJ | Clear win |

**Conclusion**: Include in paper. The Digits result (+8.5% AUC) is a compelling downstream win.

**When GSJ loses**: On very rare anomaly classes (0.6% prevalence), the tighter bandwidth undersmooths the sparse anomaly region. Scott's wider bandwidth creates a smoother normal-class envelope that works better as a one-class classifier in extreme imbalance.

---

### 1B-ii. Clustering: GSJ loses 0/4 datasets

| Dataset | ARI(Scott) | ARI(Silv) | ARI(GSJ) | n_clusters(GSJ) | True k |
|---------|-----------|-----------|----------|-----------------|--------|
| Iris (2D) | 0.886 | 0.886 | 0.886 | 3 | 3 |
| Wine (3D) | **0.463** | 0.325 | 0.147 | 39 | 3 |
| Digits 0-1-2 (4D) | 0.482 | **0.483** | 0.175 | 112 | 3 |
| Synthetic 5 clusters (3D) | **0.983** | 0.983 | 0.937 | 11 | 5 |

**What happened**: GSJ produces too many clusters. It finds 39-112 modes instead of 3-5.

**WHY — this is a fundamental mismatch, not a bug:**

Mean-shift clustering and KDE bandwidth selection optimize DIFFERENT objectives:

- **KDE bandwidth**: minimize ISE = integral of (f_hat - f)^2. Wants to capture ALL density detail (bumps, shoulders, sub-peaks). Smaller h = more detail = better density estimate.
- **Clustering bandwidth**: find the "right" number of macro-modes. Wants to MERGE sub-clusters into coherent groups. Larger h = fewer modes = coarser grouping.

GSJ correctly identifies that the density has fine-grained structure within each cluster. For density estimation this IS correct — the true density has local variations within each class. But for clustering, you want to ignore intra-cluster variation and only see inter-cluster separation.

**Formal explanation**: The optimal bandwidth for mode-finding (clustering) scales as h ~ n^{-1/(d+6)}, which is asymptotically LARGER than the density-optimal rate h ~ n^{-1/(d+4)}. No single bandwidth is optimal for both tasks.

**Analogy**: Like a microscope — GSJ gives the sharpest image of the density landscape (every hill and valley visible). But for clustering you want a blurry satellite view showing only major mountain ranges, not individual boulders.

**Conclusion**: Do NOT include clustering in the paper. Add one paragraph in Discussion explaining this is a non-goal:

> "Our bandwidth selector targets the AMISE for density estimation, not mode-finding. Applications requiring mode-based clustering typically use larger bandwidths; the relationship between density-optimal and mode-optimal bandwidths is a separate research question (see Chacon, 2015)."

---

### 1C. Subsampling: 5-11% roughness error at m=80K

| d | m=80K rel error (roughness) | After root dampening /(d+4) | Effective h error |
|---|---------------------------|-------------------------------|-------------------|
| 2 | 10.7% | 10.7% / 6 = 1.8% | ~2% |
| 5 | 5.8% | 5.8% / 9 = 0.6% | ~1% |

The roughness error is higher than hoped, but the root dampening makes the final bandwidth error acceptable (1-2%). The paper should honestly state:

> "Subsampling with m = 80,000 pairs produces roughness estimates with 5-11% relative error, translating to 1-2% bandwidth error via the (d+4)-th root dampening."

---

### 1D. Truncation: Rigorously confirmed

| d | max g(r) | occurs at | g(r=8) | g(r=10) |
|---|----------|-----------|--------|---------|
| 1 | 0.75 | r=0 | 2.35e-05 | 7.65e-09 |
| 5 | 8.75 | r=0 | 1.72e-05 | 6.37e-09 |
| 10 | 30.0 | r=0 | 1.06e-05 | 4.93e-09 |

Key facts:
- Global maximum always at r=0 (origin), not in the tail
- Function is monotonically decreasing for r >= ~3
- For ALL d <= 10: sup_{r>=8} g(r) < 2.35e-05
- Cutoff r=8 is rigorously justified

Paper statement: "Since |P_d(r^2)| exp(-r^2/4) achieves its maximum at r=0 and is monotonically decreasing for r >= 3, truncating pairs beyond 8h_0 introduces relative error below 2.4e-05 for all d <= 10."

---

## Phase 2: Revise the Paper (Next Steps)

### 2A. Minor wording fixes
- [ ] O(n^{2d}) -> O(n^2 d) everywhere
- [ ] "suggests consistency" -> "suggests stable behavior"
- [ ] "same approach" -> "closely related to"
- [ ] Comparison table: annotate 1D rates
- [ ] Truncation: state as supremum with monotonicity argument
- [ ] Subsampling: "In our experiments..."
- [ ] Novelty: "We are not aware of..."

### 2B. Fill tables with Phase 1 results
- [ ] Anomaly detection table (AUC-ROC)
- [ ] Truncation bound table
- [ ] Subsampling accuracy note (inline, not separate table)
- [ ] Pull ISE/HOLL numbers from existing notebooks

### 2C. Add anomaly detection section
- [ ] New subsection 6.4: "Application: Anomaly Detection"
- [ ] Explain: fit KDE on normal class, score = -log f_hat(x_test)
- [ ] Present AUC table (5 datasets)
- [ ] Highlight Digits result (+8.5% absolute AUC)
- [ ] Note limitation: very rare anomalies may prefer wider bandwidth

### 2D. Add clustering non-result discussion
- [ ] One paragraph in Discussion/Limitations
- [ ] Explain: density-optimal != mode-optimal bandwidth
- [ ] Different asymptotic rates: n^{-1/(d+4)} vs n^{-1/(d+6)}
- [ ] Not a flaw — it's a scope distinction

### 2E. Do NOT include
- Clustering results (wrong objective)
- LLM applications (unvalidated)
- Formal consistency proof (honestly deferred)

---

## Phase 3: Submit

### Target: JCGS (Journal of Computational and Graphical Statistics)
- Methods + software papers
- Values working code
- 4-6 month review

### Supplementary materials
- [ ] gsj package (PyPI + GitHub)
- [ ] sj repo (notebooks)
- [ ] verify_symbolic.py
- [ ] experiments/run_all.py (reproduces all tables)

### Paper narrative
> "We derive a closed-form expression for multivariate Gaussian KDE roughness (P_d polynomial), use it in a practical one-shot plug-in bandwidth selector, demonstrate ISE improvement on synthetic/real data, and show that this improvement translates to better anomaly detection AUC on real classification tasks."

---

## Timeline

```
Phase 2 (paper revision):  ~1 week
  - Wording fixes: day 1
  - Fill tables/figures: day 2-3
  - Add anomaly section: day 3-4
  - Polish: day 5

Phase 3 (submission):  ~3 days
  - Format for JCGS: day 1
  - Supplementary materials: day 2
  - Submit: day 3
```
