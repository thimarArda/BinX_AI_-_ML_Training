#  Train / Validation / Test Splits

## Overview
Introduces the **validation set** as a third data split alongside train/test, and why it's needed to tune models honestly without leaking information from the test set. Walks through the theory, a guided code implementation on the Breast Cancer dataset, then a hands-on lab comparing 5 classifiers.

## Structure

| Section | Contents |
|---|---|
| **Section 1: Core Concepts** | What a train/val/test split is, common split ratios, why the validation set exists, using it for model selection & hyperparameter tuning, detecting overfitting |
| **Section 2: Code Implementation** | Explaining the code syntax on how to split the data into | train/validation/test sets and tunes the models after
| **Section 3: Hands-on Lab** | Code practice on 5 models (Logistic Regression, SVM, KNN, Decision Tree, Random Forest), each tuned on one hyperparameter, compared on validation then test |
| **References** | Overfitting reading + links to previous notebooks |

## Notes
- Section 2's dataset and baseline models (SVM, Decision Tree) build on **Week3-Day4** and **Week3-Day5** notebooks from the same repo.
- Section 3 is meant to be done independently, All the codes there can work without going back to Section 2.

## How to Use this Notebok
1. Read Section 1 for the concept before touching code.
2.  Section 2's codes are written inside markdown cells, but you can copy them into code cells and they will work no problem.
3.  Section 3 has codes inside code cells run them all from top to buttom, then compare your results/observations to the ones documented in the notebook.
4. Requires `scikit-learn`; dataset (`load_breast_cancer`) is built in, no external files needed except `intro.png` / `Overfitting.png` for the images in Section 1, make sure they are inside the same folder when you run the notebook on your device.
