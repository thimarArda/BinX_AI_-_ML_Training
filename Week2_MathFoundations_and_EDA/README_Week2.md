# Week 2 — Math Foundations & EDA
### BinX Tech AI & Machine Learning Internship Program



## Contents

| Day | Topic | What It Covers | Dataset |
|---|---|---|---|
| **Day 1** | Descriptive Statistics | Central tendency (mean, median, mode) and spread (range, variance, std, IQR); when to use each; mean vs. median lab | `heart.csv` |
| **Day 2** | Probability & Distributions | Probability rules, conditional probability, Bayes' theorem, normal/binomial/uniform distributions; coin-flip & normal-distribution simulations | none — simulation-based (NumPy) |
| **Day 3** | Linear Algebra for ML | Vectors, matrices, dot product, matrix multiplication; how models predict; shape-mismatch debugging | none — concept-based (NumPy) |
| **Day 4** | EDA Part 1: Distributions & Outliers | Univariate statistics, Seaborn visualization, IQR outlier detection | `insurance.csv` |
| **Day 5** | EDA Part 2: Correlation & Data Storytelling | Pearson correlation, ANOVA, p-values & null hypothesis, correlation heatmap, pairplot | `kc_house_data.csv` |

### Mini Project — Complete Heart Disease EDA Notebook
A fully narrated EDA notebook that assembles everything from
Days 1–5 — descriptive statistics, univariate distributions, IQR outlier handling, bivariate analysis, and a
correlation study with heatmap and pairplot — into  analysis on the Heart Disease dataset. 


## Datasets Used This Week

| Dataset | Used In | Description |
|---|---|---|
| `heart.csv` | Day 1, Mini Project | Heart Disease dataset — 1,025 patients, 13 clinical features + binary disease label |
| `insurance.csv` | Day 4 | Medical Insurance Cost dataset — used for univariate distribution and outlier analysis |
| `kc_house_data.csv` | Day 5 | King County House Sales dataset — numeric target (`price`) used for Pearson correlation and ANOVA |



## Tools Used
Python 3.10+ • NumPy • Pandas • Matplotlib • Seaborn • SciPy • Jupyter Notebook • Git & GitHub

## How to Use This Folder
1. Work through the notebooks in order, Day 1 → Day 5, then the mini project — each README above documents
   that day's specific setup and dataset requirements.
2. Keep each day's dataset file in the same folder as its notebook.
3. Run every notebook top to bottom; later cells generally depend on the DataFrame loaded earlier in that
   same notebook.
4. Commit each notebook to GitHub as it's completed, with a clear, descriptive commit message.
