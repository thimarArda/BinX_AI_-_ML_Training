# Day 1 — Supervised Learning Concepts & the Scikit-learn API

## About this notebook

This notebook is an introduction to **supervised learning** and the **scikit-learn** library. It covers the theory behind classification and regression, walks through the basic scikit-learn model-building workflow, and ends with a short hands-on lab on a real dataset.

## Contents

| Section | What's inside |
|---|---|
| **Section 1: Supervised Learning** | Definition and key features of supervised learning; Classification (definition + types: binary, multi-class, multi-label); Regression (definition + types: simple linear, multiple linear, polynomial, regularized); a Classification vs Regression comparison table |
| **Section 2: Scikit-learn** | What scikit-learn is, the general ML flow, the steps to build a model, and a first code example using the built-in California Housing dataset with `KNeighborsRegressor` |
| **Section 3: Hands-on Lab** | Loading the Heart Disease dataset, splitting it into features (`X`) and target (`y`), performing an 80/20 train/test split, and a written explanation of why the model must never see the test set during training |
| **Section 4: References** | Articles and tutorials used as sources for this notebook |

## Requirements

- Python 3.x
- Jupyter Notebook / JupyterLab
- Libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`

Install everything with:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Data files needed

- `heart.csv` — the Heart Disease dataset used in Section 3. Place it in the same folder as the notebook before running that section.
- The California Housing dataset used in Section 2 is fetched automatically via `sklearn.datasets.fetch_california_housing()` — no download needed.

Optional images referenced in the markdown cells (not required to run the code):
- `classification_types.png`
- `ScikitLearn_in_ML.PNG`
- `Step_of_building_with_scikitlearn.PNG`

## How to use this notebook

1. Install the required libraries listed above.
2. Place `heart.csv` in the same directory as the notebook.
3. Open the notebook with `jupyter notebook` or `jupyter lab`.
4. Read Section 1 and 2 for the theory before running any code — the concepts build on each other.
5. Run the cells in order from top to bottom (the model-building steps in Section 2 and the lab in Section 3 depend on variables with the same names).
