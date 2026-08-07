# Week 3 — Supervised Learning 

## Overview

Week 3 of the BinX training program focuses on **supervised learning** through a five-day progression.Starting with the core concepts of supervised learning, then move into the two foundational algorithms: **Linear Regression** and **Logistic Regression**. After that, we explore more advanced classification models, including trees, forests, SVMs, and k-NN.

The week ends with a complete end-to-end mini project that combines all the concepts learned throughout the week into one documented machine learning pipeline.

## Objectives

- Understand what supervised learning is and distinguish between **regression** and **classification** problems.
- Implement and interpret **Linear Regression** for predicting continuous numerical targets.
- Implement and interpret **Logistic Regression** for binary classification and understand classification evaluation metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Implement and compare additional classification models, including **Decision Trees, Random Forests, SVMs, and k-NN**.
- Build a complete machine learning pipeline:
  
  **EDA → Data Preprocessing → Train/Test Split → Modeling → Evaluation against a baseline**


## Structure



| Day | Folder | What It Covers |
| --- | --- | --- |
| Day 1 | `Day1_Supervised_Learning_Concepts` | Introduces the foundations of supervised learning |
| Day 2 | `Day2_Linear_Regression` | Implements **Linear Regression** for predicting continuous numerical values. This notebook covers training a linear model, interpreting coefficients, and evaluating regression performance using different metrics. |
| Day 3 | `Day3_Logistic_Regression_and_Classification_Metrics` | Moves from regression to **classification** by implementing **Logistic Regression** for binary classification. It introduces evaluation metrics. |
| Day 4 | `Day4_Trees_Forests_SVMs_kNN` | Implementing and comparing four binary classification algorithms: **Decision Trees, Random Forests, SVMs, and k-NN**. |
| Day 5 | `Day5_Full_ML_Pipeline` | **Mini-project.** Combines everything learned throughout the week into a complete end-to-end machine learning pipeline: **EDA → preprocessing → train/test split → model training → evaluation against a baseline → selecting the final model** using the Breast Cancer Wisconsin dataset. |

## How to Use This Folder

1. **Go through the notebooks in order (Day 1 → Day 5).**  
   Each notebook builds on the concepts from the previous day, and the final mini project in **Day 5** brings together everything learned throughout the week.

2. **Run each notebook from top to bottom.**  
   Start with the import cell, then execute the remaining cells in order. Some cells depend on variables created earlier, so skipping or running cells out of order may cause errors.

3. **Install the required libraries before running the notebooks:**

```bash
pip install scikit-learn pandas matplotlib seaborn jupyter
