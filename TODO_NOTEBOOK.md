# TODO: Definitive Data Scientist Notebook

## What to build (single comprehensive notebook)

**File**: `gsj_complete_guide.ipynb`

**Dataset**: Real MiniLM transformer embeddings (embeddings.npz, 18846 docs, 384-dim)

## Sections (all use cases):

### Part 1: Density Estimation Fundamentals
- What is KDE, why bandwidth matters
- Compare Scott/Silverman/GSJ on real embeddings
- Show the density landscape at different bandwidths
- Marginal density plots per PC dimension

### Part 2: Anomaly/Novelty Detection (ML)
- Train KDE on one domain, detect OOD from another
- ROC curves comparing methods
- Precision-recall at various thresholds
- Leave-one-category-out comprehensive benchmark

### Part 3: Distribution Shift / Data Drift (MLOps)
- Reference distribution roughness
- Detect when incoming data changes
- Quantify shift magnitude via roughness ratio
- Streaming monitoring simulation

### Part 4: Embedding Space Analysis (Foundation Models)
- Roughness as a measure of representation quality
- Per-class/cluster density analysis
- Intra-cluster vs inter-cluster density contrast
- Dimensionality sensitivity (d=5 to 30)

### Part 5: Unsupervised Exploration
- Density-colored t-SNE
- Outlier identification (bottom-k density)
- Pre-filtering for improved clustering
- Per-cluster quality validation

### Part 6: Synthetic Data / Sampling
- KDE-based resampling with different bandwidths
- Quality of synthetic samples (downstream classifier test)
- Diversity measurement via roughness of generated data

### Part 7: Feature Selection / Importance
- Per-feature marginal roughness as importance metric
- Which PCA components carry the most structure?
- Roughness decay curve (effective dimensionality)

### Part 8: Model Comparison
- Benchmark table: all methods x all metrics x all scenarios
- Timing comparison
- When to use what (decision tree)

### Part 9: Production Integration
- Code snippets for sklearn/scipy integration
- Streaming/batch patterns
- Monitoring dashboard mockup

## Dependencies needed:
- numpy, scipy, sklearn, matplotlib (all available in .venv)
- embeddings.npz (already computed)
- gsj package concepts (inline implementation)

## Build approach:
- Single build script -> single .ipynb
- Execute with jupyter nbconvert
- Target: 20-30 code cells, thorough markdown between each
