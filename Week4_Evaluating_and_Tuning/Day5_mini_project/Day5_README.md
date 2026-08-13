# Day 5 — Scikit-learn Pipelines & Tuned Mini-Project


## Overview

This notebook is brings together a complete machine learning workflow, starting from Exploratory Data Analysis (EDA) and feature engineering and ending with model tuning and final evaluation.

The objective of this project is to build a classification model using the Titanic dataset to predict whether a passenger survived. Before building the model. We will investigate which characteristics are associated with survival by exploring the dataset and its features, understand the passengers, and examine the distribution of survival. These observations will guide our feature engineering. The project will then follow a complete machine learning pipeline, from data analysis and feature engineering to preprocessing, model training, tuning, and final evaluation

## Structure

| Phase                             | Contents                                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Introduction**                  | Project overview, dataset, and objective.                                                                                     |
| **Phase 1 — EDA**                 | Dataset overview, class balance, correlations, feature distributions, and observations.                                       |
| **Phase 2 — Feature Engineering** | Creates `Deck`, `Title`, `FamilySize`, and `IsAlone`, then analyzes the new features and removes unnecessary columns.         |
| **Phase 3 — Data Splitting**      | Splits the data into 80% training and 20% testing.                                                                            |
| **Phase 4 — Preprocessing**       | Separates numerical and categorical features and applies imputation, scaling, and one-hot encoding using `ColumnTransformer`. |
| **Phase 5 — Pipeline**            | Combines preprocessing and Random Forest into one pipeline and evaluates the baseline model.                                  |
| **Phase 6 — Model Tuning**        | Uses `GridSearchCV` with 5-fold `StratifiedKFold` to tune the Random Forest hyperparameters.                                  |
| **Phase 7 — Final Evaluation**    | Evaluates the tuned model on the test set and compares it with the baseline.                                                  |
| **References**                    | Additional resources and references.                                                                                          |


## Dataset

**[Titanic dataset from Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)**
(`Titanic-Dataset.csv`, 891 rows). Target: `Survived` (binary).
.

## How to Use This Notebook

1. Place `Titanic-Dataset.csv` in the same folder as this notebook.
2. Install the required libraries if not already available: `numpy`, `pandas`, `matplotlib`, `seaborn`,
   `scikit-learn`.
3. Run the notebook **top to bottom, in order**. without skipping cells to avoid missing or outdated variables.Phase 3 reloads and re-engineers the dataset
  
   

##  Visualization Accessibility:
The Visuals in this notebook are represented useing yellow and blue to distinguish between passengers who did not survive and those who survived. These colors were chosen to provide clear contrast and make the charts easier to interpret for a wider range of viewers, including those with color-vision deficiencies.
