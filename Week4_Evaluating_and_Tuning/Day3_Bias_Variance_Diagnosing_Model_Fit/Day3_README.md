#  Bias, Variance & Regularization



## Overview

This notebook covers Day 3 of Week 4 ("Evaluation, Tuning & Pipelines"): **bias, variance, overfitting,
underfitting, and regularization**.

It focuses on understanding why a machine learning model may perform poorly, how to diagnose whether the
problem is **high bias or high variance**, and how model complexity affects its ability to generalize to
unseen data. The notebook also introduces **L1 and L2 regularization** as techniques for controlling model
complexity and reducing overfitting.



## Structure

The notebook is organized into four main sections, along with an introduction and a references list at
the end.

| Section | Contents |
| --- | --- |
| **Introduction** | Introduces the purpose of the notebook . |
| **Section 1 — Bias & Variance** | Explains high bias and high variance, their relationship with model complexity, and how they lead to underfitting and overfitting. |
| **Section 2 — Diagnosing Overfitting & Underfitting** | Explains how to use the gap between training and validation performance to diagnose whether a model is underfitting, overfitting, or achieving a better balance. |
| **Section 3 — Bias–Variance Trade-Off** | Explains the trade-off between bias and variance, how model complexity affects both, and why finding the right level of complexity is important for good generalization. |
| **Section 4 — Regularization** | Introduces regularization as a way to control model complexity and reduce overfitting. Covers **L2 (Ridge)** and **L1 (Lasso)** regularization, their equations, the regularization parameter, and how they affect model coefficients. |
| **Section 5 — Hands-On Lab** | Implements and compares overfitting, underfitting, and a better-balanced model using different model configurations. The results are evaluated using training and validation accuracy and documented with score evidence. |
| **References** | Contains references and additional resources used to study bias, variance, model fitting, and regularization. |

## Dataset

The hands-on experiments use a **synthetically generated binary classification dataset** created with
scikit-learn's `make_classification`.

The dataset was generated specifically for this notebook so that the classification problem could be
controlled in terms of the number of samples, features, informative features, redundant features, class
separation, and noise. This provides a more suitable environment for observing differences between
underfitting, overfitting, and better model generalization.

The dataset is generated directly in the notebook, so no external dataset file is required.

## How to Use This Notebook

1. Open `Day3_Bias_Variance_Diagnosing_Model_Fit.ipynb`.
2. Install the required libraries if they are not already available: `numpy`, `pandas`, `matplotlib`,
   `seaborn`, and `scikit-learn`.
3. Run the notebook from top to bottom.
4. Read **Section 1** and **Section 2** to understand bias, variance, and how training and validation
   performance can be used to diagnose model fit.
5. Review **Section 3** to understand the bias–variance trade-off and how model complexity affects
   generalization.
6. Review **Section 4** to understand L1 and L2 regularization and how regularization controls model
   complexity.
7. Run all the codes in **Section 5 — Hands-On Lab** from top to buttom beacuse they depend on eachother.