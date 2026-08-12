#  Feature Engineering & Hyperparameter Tuning


## Overview

This notebook talkes about feature engineering and   how to improve a model by changing **what information** it sees  and **how it's configured** (hyperparameter tuning).

The notebook walks through the standard feature engineering toolkit (encoding, binning, feature creation,
scaling, datetime extraction), the parameter-vs-hyperparameter distinction, `GridSearchCV` /
`RandomizedSearchCV`, and then applies all of it  in a the hands-on lab: cleaning the Titanic
dataset, engineering new features, tuning a Random Forest, and comparing it against an untuned Model.

## Structure

| Section | Contents |
|---|---|
| **Section 1 — Feature Engineering** | What feature engineering is, why it matters, and the general steps (cleaning, transformation, extraction, selection, iteration). |
| **Section 2 — Common Techniques in Feature Engineering** | One-Hot Encoding, Binning, Creating Features, Feature Scaling, and Datetime Extraction — each with how it works, why it's used, and a code implementation. |
| **Section 3 — Hyperparameters vs. Parameters** | The distinction between what a model *learns* (parameters) vs. what's *set beforehand* (hyperparameters), with examples per model type. |
| **Section 4 — Hyperparameter Tuning with Scikit-Learn** | `GridSearchCV` (exhaustive search) and `RandomizedSearchCV` (sampled search for large search spaces), demonstrated on the Breast Cancer dataset. |
| **Section 5 — Hands-on Lab** | The full applied workflow on the Titanic dataset: load & inspect, clean missing values, extract `Deck`/`Title`, engineer `FamilySize`/`IsAlone`, tune a Random Forest with `GridSearchCV`, compare tuned vs. untuned CV accuracy, and rank feature importances. |
| **References** | Further reading links for the techniques covered. |

## Dataset

Two datasets appear in this notebook, used for different purposes:


- **Section 5 (Hands-on Lab):** the **[Titanic dataset from Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)**
  (`Titanic-Dataset.csv`, 891 rows) — the real dataset the lab is built around. Target: `Survived`
  (binary). It requires cleaning before use: `Cabin` is mostly missing (77%) and is converted into a
  `Deck` feature instead of being dropped outright; `Age` is imputed with the median; `Embarked` is
  imputed with the mode; and `PassengerId`, `Ticket`, `Cabin`, and `Name` are dropped once their useful
  information has been extracted into `Deck`, `Title`, `FamilySize`, and `IsAlone`.

## How to Use This Notebook

1. Place `Titanic-Dataset.csv` in the same folder as this notebook (required for Section 5).
2. Install the required libraries if not already available: `numpy`, `pandas`, `matplotlib`, `seaborn`,
   `scikit-learn`.
3. Sections 1–3 are conceptual (markdown only, no execution needed)
4. **Section 5 must be run top to bottom, in order, without skipping cells.** Several code
   cells reuse or overwrite shared variables from earlier cells in the same section.

