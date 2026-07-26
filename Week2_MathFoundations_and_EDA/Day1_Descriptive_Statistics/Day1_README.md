# Day 1 - Descriptive Statistics

This notebook covers Day 1 of Week 2 (Math Foundations & EDA) from the BinX Tech AI & ML
Internship Program. It introduces descriptive statistics and applies them to the Heart
Disease dataset.

## Learning Objectives

- Compute mean, median, and mode, and choose the appropriate one for a given dataset.
- Compute and interpret variance, standard deviation, and IQR.
- Explain how outliers affect each measure differently.

## Dataset

The notebook uses `heart.csv`, the Heart Disease dataset (1025 rows, 14 columns). The
`target` column indicates whether a patient has heart disease (1 = disease, 0 = no disease).

## Structure

**Section 1: Data Loading and Preparation**
Loads `heart.csv` into a Pandas DataFrame and checks for missing values.

**Section 2: Learnings**
- Topic 1: Measures of Central Tendency (mean, median, mode)
- Topic 2: Measures of Spread (range, variance, standard deviation, IQR)
- Topic 3: Percentiles and Quartiles

Each topic includes a short explanation, the formula, and a code example applied to
real columns from the dataset (age, chol, oldpeak).

**Section 3: Hands-On Lab**
- Step 1: Load a numeric column as a Pandas Series
- Step 2: Compute mean, median, mode, standard deviation, and IQR for that column
- Step 3: Compare mean vs. median and justify which one better represents a typical
  value, using histograms and box plots as evidence

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib

Install with:

```
pip install pandas numpy matplotlib
```

## How to Use This Notebook

1. Place `heart.csv` in the same folder as the notebook.
2. Open the notebook with Jupyter:
   ```
   jupyter notebook "Day_1_Descriptive_Statistics.ipynb"
   ```
3. Run the cells in order from top to bottom, since later cells depend on the
   DataFrame (`df`) loaded in Section 1.
4. Read the markdown cells alongside the code — they explain the concept before
   each calculation.
5. Feel free to change the column names in the code cells (e.g., swap `"chol"` for
   `"trestbps"` or `"thalach"`) to practice the same statistics on a different column.
