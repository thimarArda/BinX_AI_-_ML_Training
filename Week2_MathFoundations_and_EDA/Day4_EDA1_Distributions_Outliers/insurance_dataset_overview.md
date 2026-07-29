# Insurance Dataset — Overview

## What This Dataset Is

This is the classic **Medical Insurance Cost** dataset (`insurance.csv`), a widely used benchmark dataset in data science and machine learning courses. It contains **1,338 records** of individual health insurance beneficiaries in the United States, along with basic demographic and lifestyle attributes and the medical insurance charges billed to them.

It's most commonly used for:
- **Regression modeling** — predicting `charges` (the target variable) from the other features
- **Exploratory data analysis (EDA)** — practicing grouping, visualization, and correlation analysis
- **Feature engineering practice** — working with a mix of numeric and categorical variables
- **Teaching** — a go-to dataset for intro-level statistics, pandas, and ML tutorials because it's small, clean, and intuitive

## Key Characteristics

- **Rows:** 1,338
- **Columns:** 7 (4 numeric, 3 categorical)
- **Target variable:** `charges` (continuous, right-skewed)
- **Missing values:** None — the dataset is clean and ready to use
- **File format:** CSV

## Column Descriptions

| Column | Data Type | Description |
|---|---|---|
| `age` | Integer | Age of the primary policyholder, in years |
| `sex` | Categorical | Biological sex of the policyholder — `male` or `female` |
| `bmi` | Float | Body Mass Index — weight (kg) divided by height² (m²); an indicator of body fat and health risk |
| `children` | Integer | Number of dependents/children covered under the insurance plan |
| `smoker` | Categorical | Whether the policyholder smokes — `yes` or `no` |
| `region` | Categorical | The policyholder's residential area in the US — `northeast`, `northwest`, `southeast`, or `southwest` |
| `charges` | Float | Individual medical costs billed by the insurance provider (in USD) — the target/outcome variable |

## Typical Use Cases

- Predicting `charges` using linear regression, random forests, or gradient boosting
- Investigating how `smoker` status and `bmi` drive higher medical costs
- Comparing average charges across `region` or `sex` groups
- Practicing one-hot encoding of categorical variables (`sex`, `smoker`, `region`) before modeling
