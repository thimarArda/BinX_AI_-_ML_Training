# Day 4: Trees, Forests, SVMs & k-NN

## Overview

This notebook is part of a supervised machine learning series (following Day 1–3, which covered Linear and Logistic Regression). It introduces and compares four classic supervised learning algorithms:

- **Decision Trees**
- **Random Forests**
- **Support Vector Machines (SVMs)**
- **k-Nearest Neighbors (k-NN)**

Each algorithm is trained and evaluated on the same dataset — the **Breast Cancer Wisconsin dataset** (`sklearn.datasets.load_breast_cancer`) — a binary classification problem where the goal is to predict whether a tumor is **malignant** or **benign** based on 30 numeric features computed from digitized images of breast mass cell nuclei.

The notebook explains how each model works conceptually, walks through building and evaluating it in code, and discusses its practical limitations. It closes with a hands-on lab that directly compares all four models on the same train/test split.

## Notebook Structure

The notebook is organized into five main sections, each following a consistent pattern:

1. **Section 1 — Decision Trees**
   - How the model works (root node, splitting, impurity measures)
   - Building the model: loading data, splitting into train/test, fitting with `DecisionTreeClassifier`, and evaluating with a confusion matrix and classification report
   - Limitations (e.g., overfitting)

2. **Section 2 — Random Forest**
   - How the model works (bootstrap sampling, ensemble of trees)
   - Building the model with `RandomForestClassifier`, including feature importances
   - Evaluation via confusion matrix and classification report

3. **Section 3 — Support Vector Machines (SVM)**
   - How the model works (hyperplanes, margins, kernels)
   - Building the model with `SVC`, key hyperparameters (`kernel`, `C`, `gamma`)
   - Evaluation and limitations (computational cost, sensitivity to parameters and feature scaling)

4. **Section 4 — k-Nearest Neighbors (k-NN)**
   - How the model works (lazy/instance-based learning, distance-based voting)
   - Building the model with `KNeighborsClassifier`, key hyperparameters (`n_neighbors`, `weights`, `metric`)
   - Limitations (slow prediction on large datasets, sensitivity to feature scaling)

5. **Section 5 — Hands-On Lab (Model Comparison)**
   - Trains all four models on the same train/test split
   - Evaluates them using a consistent metric (F1-score) and assembles the results into a single comparison table
   - Reports the Random Forest's top feature importances and interprets them
   - Identifies the best-performing model and explains why it likely won

The notebook ends with a **References** section linking to further reading and video summaries for each algorithm.

## How to Use This Notebook

1. **Environment**: Make sure you have Python with `scikit-learn` and `matplotlib` installed (e.g., `pip install scikit-learn matplotlib`). No external dataset download is needed — the Breast Cancer dataset is loaded directly from `sklearn.datasets`.
2. **Run order matters**: Cells are meant to be run **top to bottom**, since later sections (especially Section 5) reuse variables such as `model`, `X_train`, `X_test`, `y_train`, and `y_test` defined earlier. Skipping cells may cause `NameError`s.
3. **Reading8*: Markdown cells explain the theory behind each algorithm

