# t-SNE & Anomaly Detection

## Overview

In the previous notebook, we used PCA to reduce high-dimensional data while keeping as much of the original variance as possible.

In this notebook, we introduce **t-SNE**, another dimensionality reduction technique mainly used for **visualizing high-dimensional data**. Unlike PCA, t-SNE focuses more on keeping similar data points close to each other, which can make clusters and local groups easier to see.

We will also learn about **anomaly detection**, which focuses on finding data points that are unusual or different from the rest of the dataset. For this, we will use **Isolation Forest**, an algorithm designed to detect these unusual points.

## Structure

| Section | Contents |
|---|---|
| Introduction | A quick review of PCA and an introduction to t-SNE and anomaly detection |
| Section 1 — t-SNE | What t-SNE is, how it works, and how to use it for visualization |
| Section 2 — Anomaly Detection | What anomaly detection is and how Isolation Forest works |
| Section 3 — Hands-on Lab | Applying t-SNE, comparing it with PCA, and using Isolation Forest to detect unusual points |
| References | Further reading and resources |

## Dataset

This notebook uses the same **Breast Cancer dataset from Scikit-learn** that we used in the PCA notebook. The dataset contains 30 features and two classes: **Malignant** and **Benign**.

Using the same dataset allows us to directly compare the PCA and t-SNE visualizations.

## How to Use This Notebook

1. Make sure the required libraries are installed: `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

2. No external dataset is needed because the Breast Cancer dataset can be loaded directly from `scikit-learn`.

3. Sections 1 and 2 explain the main concepts behind t-SNE and anomaly detection.

4. Run the code in Section 3 from top to bottom because some later steps use variables created in earlier cells.

## Note

t-SNE is mainly used for visualization. Unlike PCA, it does not provide a simple `.transform()` method for applying the same reduction to new data. Its results can also change depending on settings such as `perplexity` and `random_state`.

For this reason, we will use t-SNE to explore and visualize the structure of the data rather than as a preprocessing step for building a machine learning model.