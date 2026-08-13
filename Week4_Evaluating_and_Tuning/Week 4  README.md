# Week 4 — Evaluation, Tuning & Pipelines

**BinX Tech | AI & Machine Learning Internship Program — Phase 2**

## Overview

This folder covers **Week 4 of Phase 2**, focusing on how to evaluate, improve, tune, and build reliable machine learning models.

The week starts with **train/validation/test splitting and cross-validation**, then moves to **bias and variance, regularization, feature engineering, and hyperparameter tuning**. It concludes with a mini-project that combines everything into a **professional, leak-free Scikit-learn pipeline**.

Each day builds on the previous one, leading to a complete workflow for training and evaluating machine learning models.

## Table of Contents

| Day                                                     | Main Contents                                                                                                                               | Dataset                                                               |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Day 1 — Train / Validation / Test Splits**            | Three-way data splitting and why the test set should remain untouched until final evaluation.                                               | Scikit-learn Breast Cancer dataset                                    |
| **Day 2 — Cross-Validation**                            | K-Fold and Stratified K-Fold cross-validation, `cross_val_score`, and comparing single-split and cross-validation results.                  | Kaggle Heart Disease dataset (`heart.csv`)                            |
| **Day 3 — Bias, Variance & Model Fit**                  | Underfitting, overfitting, the bias-variance trade-off, and regularization.                                                                 | Scikit-learn synthetic classification dataset (`make_classification`) |
| **Day 4 — Feature Engineering & Hyperparameter Tuning** | Feature creation, encoding, binning, scaling, and `GridSearchCV` / `RandomizedSearchCV`.                                                    | Titanic dataset (`Titanic-Dataset.csv`)                               |
| **Day 5 — Pipelines & Mini-Project**                    | Full EDA, feature engineering, preprocessing with `ColumnTransformer`, `Pipeline`, `GridSearchCV`, Stratified K-Fold, and final evaluation. | Titanic dataset (`Titanic-Dataset.csv`)                               |


## How to Use This Folder

1. **Follow the notebooks in order (Day 1 → Day 5)** because each day builds on the concepts from the previous one.
2. **Place the required datasets** in the same folder as their notebooks:
   - Day 2 → `heart.csv`
   - Day 4 & Day 5 → `Titanic-Dataset.csv`
3. **Install the required libraries:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

4. **Run each notebook from top to bottom** to avoid missing or outdated variables.
5. **Day 5 is the final mini-project**, combining the main concepts learned throughout the week into one complete machine learning workflow.

