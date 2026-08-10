# Day 2 — Cross-Validation

**BinX Tech | AI & Machine Learning Internship — Phase 2, Week 4**

## Overview

This notebook covers Day 2 of Week 4 ("Evaluation, Tuning & Pipelines"): **cross-validation**. It picks up
directly from Day 1, where a model was evaluated using a single train/validation/test split. That approach
has a weakness — the validation score depends on which rows happen to land in that one slice of data, so
it can be misleadingly optimistic or pessimistic.


## Structure

The notebook is organized into three main sections, plus an introduction and a references list at the end.

| Section                                        | Contents                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Introduction**                               | Recaps the Day 1 single-split approach and introduces cross-validation.                                            |
| **Section 1 — Types of Cross-Validation**      | Explains K-Fold, Stratified K-Fold, their differences, and other CV variants.                                      |
| **Section 2 — Implementing K-Fold Validation** | Loads the dataset, checks class imbalance, splits the data, trains Logistic Regression, and applies 5-fold CV.     |
| **Section 3 — Hands-On Lab**                   | Cross-validates the Week 3 model, reports mean ± std, compares CV with Day 1 results, and explains stratification. |
| **References**                                 | Further reading on cross-validation and K-Fold.                                                                    |


## Dataset

The notebook uses **`heart.csv`**, a heart disease classification dataset [Link on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

The file `heart.csv` must be present in the same directory as the notebook for the data-loading cells to
run. I Uploaded it in this folder

## How to Use This Notebook

1. Place `heart.csv` in the same folder as `Day2_Cross_Validation.ipynb`.
2. Install the required libraries if not already available: `numpy`, `pandas`, `matplotlib`, `seaborn`,
   `scikit-learn`.
3. Run the cells top to bottom — Section 1 is conceptual (markdown only) and can be read without running
   anything; Section 2 and Section 3 contain the executable code and depend on the dataset being loaded
   first.

