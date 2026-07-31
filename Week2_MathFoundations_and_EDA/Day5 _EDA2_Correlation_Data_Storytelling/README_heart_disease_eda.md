# Heart Disease EDA — Week 2 Mini Project

## Overview
This notebook is the **capstone deliverable for Week 2 (Math Foundations & EDA)** of the BinX Tech AI &
Machine Learning Internship Program. It assembles everything covered across Days 1–5 — descriptive statistics,
univariate analysis, outlier detection, bivariate analysis, and correlation — into a single, narrated,
 Data Analysis on one real dataset, exactly as required by the Day 5 "Complete EDA


## Dataset
`heart.csv` — the Heart Disease dataset: **1,025 patient records, 13 clinical features + 1 target label**.

| Column | Meaning |
|---|---|
| age | age in years |
| sex | 1 = male, 0 = female |
| cp | chest pain type (0–3) |
| trestbps | resting blood pressure (mm Hg) |
| chol | serum cholesterol (mg/dl) |
| fbs | fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | resting ECG results (0–2) |
| thalach | maximum heart rate achieved |
| exang | exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | slope of the peak exercise ST segment (0–2) |
| ca | number of major vessels colored by fluoroscopy (0–4) |
| thal | thalassemia result (0–3) |
| target | 1 = heart disease present, 0 = not present |

There are no missing values in this dataset, confirmed directly in the notebook.

## Notebook Contents

**1. Setup & Data Loading** — imports, `df.head()`, `df.info()`, missing-value check, column reference table.

**2. Descriptive Statistics** — mean, median, mode, standard deviation, variance, and IQR for `age`, `trestbps`,
`chol`, `thalach`, and `oldpeak`, with an interpretation of which columns are skewed and why the median is the
more honest summary for those.

**3. Univariate Analysis & Outlier Detection**
- Histograms with KDE for every numeric variable
- Box plots to visually spot outliers
- IQR-rule outlier flagging (`Q1 - 1.5×IQR` / `Q3 + 1.5×IQR`) with a table of bounds and outlier counts
- A written verdict on why the flagged outliers (mostly in `chol`, `trestbps`, `oldpeak`) are kept rather than
  deleted
- Count plots for every categorical/coded variable, split by `target`, with a class-balance note

**4. Bivariate Analysis**
- Scatter plots: age vs. max heart rate, age vs. cholesterol, max heart rate vs. ST depression — colored by
  `target`
- Grouped box plots: max heart rate, ST depression, and age, each split by `target`

**5. Correlation Analysis**
- Full annotated correlation heatmap across all numeric variables
- Correlation-with-`target` ranking table
- Pairplot of the most relevant variables, colored by `target`
- A written identification of the two–three strongest relationships and what they imply for a future model
  (see the separate heatmap write-up for the exact numbers)




## Tools Used
Pandas • NumPy • Seaborn • Matplotlib • Jupyter Notebook • Git & GitHub

## How to Run It
1. Place `heart.csv` in the same folder as `heart_disease_eda.ipynb`.
2. Open the notebook in Jupyter Notebook or JupyterLab.
3. Run all cells top to bottom — every section depends on the DataFrame (`df`) loaded in Section 1.
4. Read the markdown narrative alongside each code cell.
