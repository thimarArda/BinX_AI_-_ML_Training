# Day 2 — DBSCAN & Hierarchical Clustering


## Overview

This notebook picks up from Day 1's K-Means work by covering two alternative clustering algorithms —
**DBSCAN** and **Hierarchical Clustering** — built specifically to handle the shapes and situations where
K-Means falls short. It explains how each algorithm works, when to choose one over the other, and applies
both to a hands-on lab, comparing their results directly against K-Means on the same dataset.

## Structure

| Section | Contents |
|---|---|
| Introduction | Recaps Day 1 (K-Means) and introduces the goal of this notebook |
| Section 1 — K-Means Limitations | Why K-Means struggles with non-spherical clusters, varying density, and outliers |
| Section 2 — DBSCAN Algorithm | What DBSCAN is and how density-based clustering works |
| Section 3 — Hierarchical Clustering Algorithm | What hierarchical clustering is, dendrograms, and the `scipy` implementation |
| Which Algorithm to Choose? | A decision guide matching data shape/situation to the best-fit algorithm |
| Section 4 — Hands-on Lab | Runs DBSCAN and Hierarchical Clustering on the Iris dataset, builds a dendrogram, compares results against K-Means, and concludes which method fits best |
| References | Further reading links |

## Dataset

The hands-on lab uses the **Iris dataset from Scikit-learn**, a classic dataset of 150 flower samples
with four numeric measurements each (sepal/petal length and width) split across three species. It's small,
well-separated, and commonly used to compare clustering algorithms side by side.

> **Note:** the same code from Section 4 will be re-tested later using a heart disease dataset.

## How to Use This Notebook

1. Install the required libraries if not already available: `numpy`, `pandas`, `matplotlib`,
   `scikit-learn`, `scipy`.
2. Sections 1–3 and the "Which Algorithm to Choose?" section are conceptual (markdown only) and can be
   read in any order.
3. Run Section 4's code cells top to bottom — each step builds on variables (`X_scaled_df`, `labels`, `Z`)
   defined in the cells before it.
