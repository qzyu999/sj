# Applications of Efficient $d$-D KDE in Modern LLM Development

## Overview

Large Language Model development involves massive datasets (trillions of tokens), high-dimensional representations (d=768–4096 embeddings), and GPU-heavy infrastructure. Several stages of the LLM pipeline involve density estimation problems — often solved with ad-hoc thresholds or expensive learned models. A principled, GPU-accelerated KDE with proper bandwidth selection (SJ) can replace hand-tuned hyperparameters with data-driven decisions.

This document maps the connections between our closed-form $d$-D Sheather-Jones bandwidth selector and concrete LLM development tasks.

---

## 1. Training Data Deduplication and Quality Filtering

### The problem

Modern LLM training corpora are massive (15T+ tokens for Llama 3, 13T for Gemini). Before training, pipelines aggressively deduplicate and filter:

- **Near-duplicate detection**: Remove passages that are semantically identical (paraphrases, copy-paste, boilerplate)
- **Quality filtering**: Remove low-quality, toxic, or garbage text
- **Domain balancing**: Ensure proportional representation across topics

### Current approach

1. Embed passages into d-dimensional space via a sentence transformer (d=384–768)
2. Reduce to d=10–50 via PCA (KDE in d=768 is infeasible due to curse of dimensionality)
3. Apply similarity thresholds: "if two embeddings are within distance $\epsilon$, consider them duplicates"

**The problem**: $\epsilon$ is hand-tuned. Too small → misses paraphrases. Too large → removes legitimately distinct content.

### How SJ helps

KDE with SJ bandwidth gives a **data-driven density estimate** at each point. Instead of a fixed distance threshold:

$$\text{duplicate}(x_i, x_j) = \mathbf{1}[\|x_i - x_j\| < \epsilon]$$

Use density-adaptive filtering:

$$\text{redundant}(x_i) = \mathbf{1}[\hat{f}(x_i) > \tau]$$

where $\hat{f}$ is the KDE with SJ-selected bandwidth. Points in high-density regions are redundant; points in low-density regions are unique/novel. The SJ bandwidth adapts to the local structure of the embedding space — tight where embeddings cluster (common text patterns), wide where they're sparse (rare/specialized content).

### Scale requirements

- 1B embedded passages × d=50 (after PCA) = 200 GB of float32 vectors
- Full $O(n^2)$ KDE: impossible ($10^{18}$ operations)
- Subsampled SJ (paper_v3 Strategy 2): sample 100K pairs → bandwidth in seconds
- Apply KDE to full dataset with tree-based truncation (Strategy 1): $O(n \cdot k)$ where $k \ll n$

### GPU pipeline

```
Text corpus
  → Batch embed on GPU (sentence-transformers, already standard)
    → PCA reduce to d=50 on GPU (torch.pca_lowrank)
      → SJ bandwidth computation (subsampled, 80K pairs, <1s on GPU)
        → Density evaluation per point (tree-truncated or tiled GPU)
          → Filter decisions
```

The entire pipeline stays on GPU — no CPU-GPU transfers needed.

---

## 2. Training Data Mixture and Domain Weighting

### The problem

Not all training data is equally valuable. Papers like DoReMi (Xie et al., 2023) and DSIR (Xie et al., 2023) show that reweighting domains during training significantly impacts downstream performance.

The key idea: **importance sampling**. Weight each training sample by:

$$w(x) = \frac{f_{\text{target}}(x)}{f_{\text{source}}(x)}$$

where $f_{\text{target}}$ is the distribution of "high-quality" data (e.g., Wikipedia, books) and $f_{\text{source}}$ is the full corpus distribution.

### Current approach

DSIR uses n-gram features + logistic regression to approximate the density ratio. This works but:
- Requires a labeled "target" set
- Captures only surface-level features (n-grams), not semantics
- Doesn't adapt to the actual geometry of the embedding space

### How SJ helps

Estimate both densities directly via KDE in embedding space:

$$\hat{w}(x) = \frac{\hat{f}_{\text{target}}(x; h^*_{\text{target}})}{\hat{f}_{\text{source}}(x; h^*_{\text{source}})}$$

where each bandwidth $h^*$ is selected by SJ from its respective dataset. This gives:
- Semantically meaningful weights (based on embedding similarity, not n-grams)
- Automatic adaptation to each domain's density structure
- No need for a separate classifier — just KDE + bandwidth selection

### Why SJ matters here specifically

The target distribution (Wikipedia) is highly multimodal — clusters of topics (science, history, geography, etc.). Silverman's rule would oversmooth, treating all of Wikipedia as one blob. SJ detects the topic clusters and assigns higher weight to training samples that match specific Wikipedia topics, not just the average.

---

## 3. Embedding Space Analysis and Representation Quality

### The problem

During training, practitioners monitor embedding quality:
- **Isotropy**: Are embeddings spread uniformly or collapsed to a low-dimensional manifold?
- **Cluster structure**: Do semantically related texts form distinct groups?
- **Uniformity**: Is the embedding space being "used" efficiently?

### Current metrics

- Average cosine similarity (crude measure of isotropy)
- Alignment and Uniformity (Wang & Isola, 2020)
- Cluster indices (silhouette, etc.)

### How KDE adds value

KDE with proper bandwidth gives a continuous density map of the embedding space. From this:

**Effective dimensionality**: The SJ bandwidth $h^*$ itself encodes information about the data's intrinsic structure:

$$h^* = \left(\frac{d \cdot R(K)}{n \cdot \hat{\Psi}}\right)^{1/(d+4)}$$

The roughness $\hat{\Psi}$ measures how "non-uniform" the density is. High roughness → multimodal → SJ gives small $h$. Low roughness → nearly uniform → SJ gives large $h$ (close to Silverman). Tracking $h^*_{\text{SJ}} / h_{\text{Silverman}}$ over training is a scalar metric for "how much structure has the model learned."

**Entropy estimation**: $H(\hat{f}) = -\int \hat{f} \log \hat{f} \, dx$ gives the differential entropy of the embedding distribution. Maximum entropy = uniform = good uniformity. KDE provides the density estimate; SJ ensures the bandwidth doesn't artificially inflate or deflate the entropy estimate.

**Mode counting**: The number of modes in $\hat{f}$ indicates how many distinct "concepts" the embedding space represents. With SJ bandwidth, modes aren't artificially merged (as with Silverman) or split (as with undersmoothing).

---

## 4. Retrieval-Augmented Generation (RAG) Calibration

### The problem

RAG systems retrieve documents by finding nearest neighbors in embedding space. A critical question: **how many documents should we retrieve, and how far should we search?**

Current approaches use fixed top-$k$ or fixed distance threshold. Both are suboptimal:
- Fixed $k$: retrieves $k$ documents regardless of whether they're relevant
- Fixed threshold: doesn't adapt to dense vs sparse regions of the corpus

### Density-adaptive retrieval

With KDE of the corpus in embedding space:

1. **In high-density regions** (common topics): many similar documents exist → can be selective (tight threshold)
2. **In low-density regions** (rare topics): few documents available → need wider search

The SJ bandwidth naturally provides this adaptation — it's the scale at which the corpus has meaningful density variation. Use $c \cdot h^*_{\text{SJ}}$ as the retrieval radius, where $c$ is a small constant (e.g., 2–3).

### Confidence calibration

When the query falls in a low-density region of the corpus:
- Few retrieved documents are available
- The LLM should express uncertainty ("I'm not sure about this")
- KDE density at the query point → confidence score

This is exactly the "know what you don't know" problem, solved with density estimation.

---

## 5. RLHF: Out-of-Distribution Detection for Reward Models

### The problem

In RLHF (Reinforcement Learning from Human Feedback), the reward model scores LLM outputs. A well-known failure: **reward hacking** — the policy generates outputs that get high reward scores but are actually garbage (exploiting reward model blindspots).

The blindspots occur where the reward model was never trained — out-of-distribution (OOD) inputs.

### KDE-based OOD detection

1. Estimate the density of the reward model's training data in embedding space: $\hat{f}_{\text{RM-train}}$
2. For each policy output, compute its density under this distribution
3. If density is low → the reward model's score is unreliable → apply penalty

$$r_{\text{safe}}(x) = r(x) \cdot \sigma(\log \hat{f}_{\text{RM-train}}(x) - \log \tau)$$

where $\sigma$ is a sigmoid that smoothly penalizes OOD outputs.

### Why SJ matters

The threshold $\tau$ is usually hand-tuned. With SJ bandwidth, the density estimate $\hat{f}$ is calibrated — its magnitude has meaning relative to the data structure. A natural threshold: $\tau = \hat{f}$ at the 5th percentile of the training data's own density values.

---

## 6. Synthetic Data Generation and Diversity

### The problem

Synthetic data is increasingly central to LLM training (Phi-4, Orca, etc.). Key questions:
- Is the synthetic data diverse enough?
- Are we generating redundant examples?
- Where in the distribution should we generate more?

### KDE for generation guidance

1. Fit KDE to existing (real + synthetic) data embeddings
2. Identify low-density regions → these are "gaps" that need more synthetic examples
3. Generate targeted synthetic data to fill gaps
4. Re-estimate KDE → check if gaps are filled

The SJ bandwidth determines the resolution at which "gaps" are detected. Too wide → can't see gaps. Too narrow → everything looks like a gap.

### Diversity metric

$$\text{Diversity} = H(\hat{f}_{\text{synthetic}}) = -\int \hat{f} \log \hat{f} \, dx$$

Estimated via KDE with SJ bandwidth. Track over generation rounds to ensure diversity doesn't collapse.

---

## 7. Speculative Decoding in Continuous-Token Architectures

### The problem

Speculative decoding (Leviathan et al., 2023) accelerates inference by using a small "draft" model to propose tokens, accepted/rejected by the full model:

$$P(\text{accept}) = \min\left(1, \frac{p_{\text{target}}(x)}{p_{\text{draft}}(x)}\right)$$

For discrete tokens, this is a ratio of probability masses. But emerging architectures use **continuous token representations** (diffusion LLMs, MDLM, etc.). There, it's a ratio of probability *densities*.

### KDE for continuous speculative decoding

When the target model's distribution over continuous tokens is intractable to compute exactly:
- Collect samples from the target model → fit KDE with SJ bandwidth
- Use KDE density as $p_{\text{target}}(x)$ for acceptance decisions
- The draft model's density $p_{\text{draft}}(x)$ is known (it's your model)

This enables speculative decoding in architectures where the target distribution is only available via samples.

---

## 8. Practical Considerations

### What dimension range is feasible?

| Application | Embedding dim | After PCA | KDE feasible? |
|-------------|--------------|-----------|---------------|
| Sentence dedup | 384–768 | 10–50 | ✓ (d ≤ 50 with subsampling) |
| Domain weighting | 384–768 | 10–30 | ✓ |
| Embedding analysis | 768–4096 | 5–20 | ✓ |
| RAG calibration | 768–1536 | 10–50 | ✓ |
| Reward OOD | 768 | 10–30 | ✓ |
| Token-level (speculative) | 64–512 | 10–30 | ✓ |

The key: **always PCA first**, then KDE. Direct KDE in d=768 is theoretically and practically impossible — the curse of dimensionality makes all non-parametric methods fail above d≈15-20. But PCA to the effective dimensionality (typically 10–50 for embedding spaces) is standard practice.

### Computation budget

| Scenario | n | d (post-PCA) | SJ time (subsampled GPU) | Full KDE time |
|----------|---|---|---|---|
| Dedup 1M passages | 1M | 30 | ~2s | ~30s (tiled GPU) |
| Domain weight 10M | 10M | 20 | ~2s (subsample) | ~5min (blocked) |
| Embedding analysis 100K | 100K | 50 | ~5s | ~2min |
| RAG corpus 10M | 10M | 30 | ~2s (subsample) | Build tree: 30s, query: O(1) per point |

All of these are negligible relative to the actual LLM training time (days–weeks on thousands of GPUs).

### What SJ specifically adds vs. just using Silverman

In the LLM context, the data is almost always multimodal (multiple topics, multiple writing styles, multiple quality levels). Silverman's Normal-reference rule:
- Assumes unimodal data → oversmooths
- Blurs topic boundaries → dedup misses paraphrases within topics
- Gives one-size-fits-all threshold → misses local structure

SJ's advantage (20-70% ISE improvement on multimodal data) translates to:
- **Better dedup recall**: catches more near-duplicates within topic clusters
- **More precise importance weights**: distinguishes between topics rather than averaging
- **Tighter confidence bounds**: OOD detection is more sensitive to distribution boundaries

---

## 9. Honest Limitations

1. **KDE in high-d is fundamentally limited**: Even with SJ, KDE estimates become unreliable above d≈15-20. The curse of dimensionality is inescapable for non-parametric methods. PCA preprocessing is mandatory.

2. **LLM pipelines don't currently use KDE**: Current practice uses SimHash/MinHash (dedup), learned classifiers (quality filtering), and cosine similarity thresholds (RAG). KDE would be a more principled replacement but requires integration into existing tooling.

3. **The marginal value may be small**: In a pipeline with 100 design decisions, replacing one threshold with a data-driven one helps — but it's not transformative. The value is in removing one hyperparameter, not in revolutionizing training.

4. **Better alternatives may exist for specific tasks**: For dedup specifically, locality-sensitive hashing (LSH) is faster and well-studied. For OOD detection, ensemble disagreement or dropout uncertainty may be more calibrated than KDE.

5. **Non-isotropic embedding spaces**: LLM embeddings are typically anisotropic (Ethayarajh 2019). Our scalar SJ bandwidth assumes isotropy (after whitening). For highly anisotropic spaces, a diagonal or full bandwidth matrix would be more appropriate — but then we lose the closed-form simplicity.

---

## 10. Summary: Where This Matters

| Application | Impact | Readiness |
|-------------|--------|-----------|
| Data dedup threshold | Medium — removes a hyperparameter | High (drop-in) |
| Domain weighting | High — principled importance sampling | Medium (needs integration) |
| Embedding analysis | Medium — better metrics during training | High (monitoring tool) |
| RAG adaptive retrieval | Medium — query-dependent radius | Medium |
| RLHF OOD detection | High — prevents reward hacking | Low (research stage) |
| Synthetic data diversity | Medium — guides generation | Medium |
| Speculative decoding (continuous) | Speculative — emerging architectures only | Low (future) |

The strongest near-term application is **data dedup/weighting** — it's where density estimation is already implicitly used (via similarity thresholds), the data is already embedded on GPUs, and SJ directly replaces a hand-tuned constant with a data-driven one.

---

## References

1. Xie et al., "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining," NeurIPS 2023.
2. Xie et al., "Data Selection for Language Models via Importance Resampling," NeurIPS 2023.
3. Ethayarajh, "How Contextual are Contextualized Word Representations?," EMNLP 2019.
4. Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity," ICML 2020.
5. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023.
6. Touvron et al., "Llama 3," Meta AI Technical Report, 2024.
7. Lee et al., "Deduplicating Training Data Makes Language Models Better," ACL 2022.
