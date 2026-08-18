# Dimensionality Reduction with PCA


## Overview

After covering unsupervised learning and clustering in the previous two notebooks, this one turns to
another common problem: datasets with a large number of features. As the number of features grows, data
gets harder to analyze and models need more compute to process it — especially true for images or wide
tabular datasets. This notebook introduces **Principal Component Analysis (PCA)**, a technique that
transforms the original features into a smaller set of principal components while preserving as much of
the data's meaningful variation as possible, making high-dimensional data more manageable with minimal
information loss.

## Structure

| Section | Contents |
|---|---|
| Introduction | Recap of past notebooks (they are in the same week 5 folder) and the goal of this one |
| Section 1 — The Curse of Dimensionality | High-dimensional data, its challenges, and ways to manage it |
| Section 2 — Principal Component Analysis (PCA) | The core linear algebra concepts PCA is built on |
| Section 3 — Hands-on Lab | Scaling the data, fitting PCA, choosing the number of components, 2D visualization, and a written summary |
| References | Further reading links |

## Dataset

The hands-on lab uses the **Breast Cancer dataset from Scikit-learn**, a binary classification dataset
(malignant vs. benign) with 30 numeric features. It's a good fit for demonstrating dimensionality
reduction precisely because of that high feature count.

## How to Use This Notebook

1. Install the required libraries if not already available: `numpy`, `pandas`, `matplotlib`,
   `scikit-learn`.
2. Open the notebook in Jupyter (`jupyter notebook` or `jupyter lab`) or in an editor like VS Code with
   the Jupyter extension.
3. No external dataset file is needed — the Breast Cancer dataset loads directly from `scikit-learn`.
4. Run all cells top to bottom; each step in the hands-on lab (scaling → fitting PCA → choosing components
   → visualizing) builds on the previous one.
