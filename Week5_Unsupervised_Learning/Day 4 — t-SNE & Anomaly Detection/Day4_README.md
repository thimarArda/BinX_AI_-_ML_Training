#  t-SNE & Anomaly Detection


## Overview

The previous notebook used PCA to compress high-dimensional data while preserving global variance. This
notebook introduces **t-SNE**, a technique built specifically for *visualizing* high-dimensional data by
preserving local neighborhoods instead — making it especially good at revealing clusters that PCA can't
show clearly. It then moves to a different unsupervised problem: **anomaly detection**, finding data
points that don't fit the pattern of the rest — and covers **Isolation Forest**, a standard algorithm for
that task, including how it connects back to DBSCAN's "noise" points from the clustering notebook.

## Structure

| Section | Contents |
|---|---|
| Introduction | Recaps PCA and introduces t-SNE and anomaly detection |
| Section 1 — t-SNE for Visualization | What t-SNE is, how it works, implementation with syntax explained, and a PCA vs. t-SNE comparison table |
| Section 2 — Anomaly Detection | What anomaly detection is, Isolation Forest and how it works, the `contamination` parameter, and its overlap with DBSCAN |
| Section 3 — Hands-on Lab | Runs t-SNE on the dataset (colored by K-Means clusters), compares it side-by-side with a PCA plot, runs Isolation Forest, and inspects the two most anomalous points |
| References | Further reading links |

## Dataset

Uses the same **Breast Cancer dataset from Scikit-learn** as the PCA notebook (30 features, binary
malignant/benign target), specifically so the t-SNE plot can be compared directly against the earlier PCA
plot on identical data.

## How to Use This Notebook

1. Install the required libraries if not already available: `numpy`, `pandas`, `matplotlib`,
   `scikit-learn`.
2. No external dataset file is needed — the Breast Cancer dataset loads directly from `scikit-learn`.
3. Sections 1 and 2 are conceptual (markdown only) and can be read in any order.
4. Run Section 3's code cells top to bottom — later steps (the PCA comparison, Isolation Forest, and the
   anomaly inspection) reuse `X_scaled`, `cluster_labels`, and `X_tsne` from the first two code cells.

## Note

t-SNE has no `.transform()` for new data, it must be re-run on the full dataset any time it's used, and
its output changes with the `perplexity` and `random_state` settings. Treat it as a visualization tool
only, not as a feature-reduction step for downstream modeling.
