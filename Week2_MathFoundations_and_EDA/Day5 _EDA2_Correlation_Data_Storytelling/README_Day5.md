# Day 5 — EDA Part 2: Correlation & Data Storytelling

## Overview
This notebook covers Day 5 of Week 2 (Math Foundations & EDA) from the BinX Tech AI & ML Internship Program.
It goes deeper into **bivariate and correlation analysis** than the Week 2 syllabus alone requires — beyond
Pearson correlation and heatmaps, it also introduces statistical significance testing (p-values, ANOVA) and a
proper decision process for choosing the right technique depending on variable type. The notebook uses the
King County House Sales dataset (`kc_house_data.csv`) to demonstrate every technique on a real numeric target
(`price`).



## Notebook Contents

**1. Data Storytelling Concepts**
- What bivariate analysis is, and the features-vs-labels (independent vs. dependent variable) distinction
- A decision table matching label/feature data types to the correct effect-size statistic and visualization
  (Pearson correlation + scatterplot, ANOVA + bar chart, chi-square + cross-tab)
- Continuous vs. discrete data types and why the distinction matters for choosing a technique

**2. Numerical vs. Numerical — Pearson Correlation**
- Visualizing the relationship with a scatter plot and a regression line (`sns.regplot`)
- Computing Pearson's *r* two ways: `scipy.stats.pearsonr` (with p-value) and `df.corr()`
- Worked example on `sqft_living` vs. `price` (r ≈ 0.70 — a strong positive relationship: larger homes cost more)

**3. Numerical vs. Categorical — ANOVA**
- Visualizing with box plots and grouped bar charts (average price by number of floors)
- Running a one-way ANOVA (`scipy.stats.f_oneway`) across floor-count groups
- Interpreting the resulting p-value against the 0.05 significance threshold

**4. P-values, Correlation & the Null Hypothesis**
- What the null hypothesis represents and how a p-value tests it
- How this connects back to whether an observed correlation is likely real or due to chance

**5. Correlation Heatmap & Pairplot**
- What a correlation heatmap shows (−1 to +1 scale, color intensity for relationship strength)
- What a pairplot reveals across multiple numeric variables at once (scatter grid + diagonal distributions)
- A pairplot of `bedrooms`, `sqft_living`, `sqft_lot`, `floors`, and `price`

## Dataset
`kc_house_data.csv` — King County (Seattle area) house sales data, used here because it has a continuous
numeric target (`price`), which makes it a clean way to demonstrate Pearson correlation before moving to
categorical-vs-numeric tests like ANOVA.

## Tools Used
Python • Pandas • NumPy • Matplotlib • Seaborn • SciPy (`pearsonr`, `f_oneway`) • Jupyter Notebook

## How to Run It
1. Ensure `kc_house_data.csv` is in the same folder as the notebook.
2. Referenced images (`Correllation_pearson.png`, `spearman_correlation.png`) should also sit next to the
   notebook — if they don't load, the surrounding markdown text still explains the same concept.
3. Open the notebook in Jupyter Notebook or JupyterLab and run the cells sequentially from top to bottom.
4. Read the markdown explanation before each code cell — this notebook is concept-first, code-second.

## Note
This notebook focuses on teaching the **statistical reasoning** behind correlation and bivariate analysis
(including significance testing). For the applied, fully narrated end-to-end EDA required by the Week 2
mini-project deliverable, see the separate `heart_disease_eda.ipynb` notebook and its README.
