Name: Sayantika Chakraborty   Roll No.: ME22B190 

# Data Veracity Analysis using t-SNE and Isomap

This notebook performs **manifold learning and data veracity inspection** on a multi-label dataset using **t-SNE** and **Isomap**.
The goal is to uncover **hidden structure**, identify **label noise**, **outliers**, and **hard-to-learn regions**, and understand their impact on **classification model performance**.

## Contents

1. **Data Preparation**
   - Scaling of features (`X_scaled`)
   - Label filtering to reduce visualization complexity
     → keeps only the top 2 most frequent single-label classes and the most common multi-label combination.

2. **Dimensionality Reduction**
   - **t-SNE** (local structure preservation)
   - **Isomap** (global manifold preservation)

3. **Veracity Inspection**
   - Identification of:
     - Noisy / Ambiguous samples
     - Outliers
     - Hard-to-learn regions (based on label entropy)
   - Visual highlighting of these points on both embeddings.

4. **Interpretations**
   - Local vs global patterns from t-SNE and Isomap.
   - Explanation of how these patterns affect classifier performance.

## Theoretical Difference — t-SNE vs Isomap

| Aspect | t-SNE | Isomap |
|--------|--------|--------|
| **Goal** | Preserve **local neighborhood structure** | Preserve **global manifold geometry** |
| **Distance Type** | Pairwise **Euclidean** converted to **probabilistic similarity** | **Manifold distance** (shortest path along manifold) |
| **Neighborhood Parameter** | `perplexity` (typ. 5–50) | `n_neighbors` |
| **When it works best** | When data has **local clusters** and nonlinear separations | When data lies on a **smooth low-dimensional manifold** |
| **Failure Mode** | Global distances distorted | Sensitive to disconnected or sparse manifolds |

## Veracity Inspection — t-SNE Interpretation

The t-SNE embedding emphasizes **local cluster formation** and **neighborhood purity**.

- **Noisy / Ambiguous Samples:**
  Points of one color embedded inside another cluster indicate potential mislabeling or overlapping feature spaces.

- **Outliers:**
  Distant isolated points or mini-clusters; may represent rare biological variations or measurement errors.

- **Hard-to-Learn Samples:**
  Dense mixed-color regions with high label entropy where simple classifiers cannot define clear boundaries.

**Summary:**
t-SNE effectively reveals **local label inconsistencies**, **noisy samples**, and **regions of overlap** — ideal for detecting fine-grained veracity issues.

## Veracity Inspection — Isomap Interpretation

The Isomap embedding (n_neighbors = 5) captures **global manifold continuity**.

- **Noisy / Ambiguous Samples:**
  Scattered through cluster boundaries — signal overlapping or inconsistent labeling.

- **Outliers:**
  Far-off samples with weak connectivity — could represent rare cases or errors.

- **Hard-to-Learn Samples:**
  Located along smooth transitions between larger groups — regions of high label entropy.

**Summary:**
Isomap exposes the **global connectivity** of the data manifold, revealing smooth transitions and uneven sampling density that affect classifier generalization.

## Impact of Veracity Issues on Classification Performance

Both embeddings highlight veracity issues that directly affect model behavior:

- **Noisy / Ambiguous Labels:** reduce cluster purity → inconsistent training signals.
- **Outliers:** can confuse models that depend on distances (like k-NN or SVM) by pulling their decisions away from the main data patterns. 
- **Hard-to-Learn Regions:** high local entropy → increased misclassification probability.
- **Global vs Local Geometry:** t-SNE captures *local confusion*; Isomap reveals *global manifold continuity* — together providing a complete data quality picture.

**Conclusion:**
Combining t-SNE and Isomap visualizations allows both **local** and **global** inspection of the data’s integrity.

## Key Parameters

| Method | Parameter | Description | Typical Value |
|---------|------------|--------------|----------------|
| t-SNE | `perplexity` | Balances local vs global structure | 30 |
| Isomap | `n_neighbors` | Controls manifold connectivity | 5 |
| Veracity Analysis | `k` | Neighborhood size for entropy/outlier detection | 10 |

## Visualization Summary

- **t-SNE:** Fine-grained clusters with visible local mixing → useful for detecting noisy points.
- **Isomap:** Smooth manifold continuity → reveals global relationships between clusters.
- **Combined View:** Provides a complete diagnostic of data veracity for robust classification model design.

