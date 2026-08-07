# Mini Project: Breast Tumor Classifier

## Overview

This notebook is the **Week 3 mini-project** of the **BinX training program**. It implements a full,  machine learning pipeline to build a **breast cancer tumor classifier** — predicting whether a tumor is **malignant** or **benign** from 30 numeric measurements derived from cell nucleus images (a binary classification problem on tabular/numerical data, not on raw images).

## Structure 

The notebook is organized into three phases:

1. **Phase 1 — EDA:** all exploratory data analysis procedures — inspecting the dataset's structure and features, checking for missing values and duplicates, examining class balance, correlations, and feature distributions.
2. **Phase 2 — Modeling:** preprocessing (scaling) followed by training two classification models — **Logistic Regression** and **Random Forest**.
3. **Phase 3 — Evaluation:** evaluating both trained models, alongside an additional **baseline** model, then selecting and justifying the best-performing model.

## Files

| File | Description |
|---|---|
| `Mini_Project_Full_ML_Pipeline.ipynb` | The full narrated notebook — every stage explained in Markdown, every choice justified, all cells pre-executed with outputs. |
| `README.md` | This file. |



## How to Use This Notebook

1. Make sure the required libraries are installed:
   ```bash
   pip install scikit-learn pandas matplotlib seaborn jupyter
   ```
2. Open the notebook and start by running the **first coding cell**, which imports all the required libraries used throughout the pipeline.
3. Run **every cell in order, from top to bottom**. The cells depend on each other (later cells reuse variables like `df`, `X_train`, `model`, etc. created earlier), so running them out of order — or skipping one — will cause errors.

No external data files are needed; the dataset loads directly from scikit-learn.



