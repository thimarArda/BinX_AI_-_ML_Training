# Day 2 — Linear Regression

## What this notebook is

A teaching notebook that introduces **Linear Regression**, the first regression
algorithm covered after the intro to supervised learning and the scikit-learn API.
It mixes conceptual explanations, small illustrative diagrams, and a full
hands-on lab on a real dataset.

## Structure

- **Introduction & Learning Objectives** — what the notebook covers and what
  you should be able to do by the end.

- **Section 1 : Key Topics** — the concepts, built up in phases:
  - *Phase 1 : How does Linear Regression work*`
  - *Phase 2 : Interpreting Coefficients* 
  - *Phase 3 : Linear regression - Loss* 


- **Section 2 : Hands-on Lab** — applies everything in section1 on the
  California housing dataset: split the data, train a model, inspect
  coefficients, evaluate with MAE/RMSE/R², and compare against a baseline
  that just predicts the mean.

- **References** — sources the explanations and diagrams were adapted from.

## How to use it

1. Open the file in Jupyter (`jupyter notebook Day2__Linear_Regression.ipynb`)
   or JupyterLab.
2. Make sure `scikit-learn`, `pandas`, `numpy`, and `matplotlib` are installed
   (`pip install scikit-learn pandas numpy matplotlib`).
3. Run the cells top to bottom — Section 1 is has syntax cells and diagrams functions. Section 3 is the lab,
   where each code cell builds on the ones before it (dataset load → split →
   train → evaluate), so run those in order.
4. Read the markdown commentary after each code cell it explains what the
   output means, which is the main point of the notebook.
