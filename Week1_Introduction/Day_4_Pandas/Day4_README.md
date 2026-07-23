# Day 4 - Dealing With Tabular Data Using Pandas 🐼

## What this notebook is about
This notebook is for practicing Pandas, using the King County house sales dataset (the same one used in the Week 1 mini-project). It covers loading data, cleaning it, filtering it, and grouping/aggregating it.

## About the dataset
- **Title:** House Sales in King County, USA
- **Source:** [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- Contains house sale prices for King County, for homes sold between May 2014 and May 2015. No missing values.

## What the notebook does
1. **Loads the dataset** (`kc_house_data.csv`) and inspects it with `.head()`, `.shape`, `.info()`, and `.describe()`.
2. **Checks for missing values and duplicates**, and goes over the different ways to handle them (`fillna`, `dropna`, `drop_duplicates`), even though this dataset turned out to have none.
3. **Filters the data** to find things like houses that cost more than $6 million, and houses that have a waterfront view.
4. **Groups and aggregates the data** using `groupby()`:
   - Groups by a single column (like `floors`) and by multiple columns (like `waterfront` and `view`).
   - Sums up prices per group, and iterates through each group to look at it individually.
   - Uses aggregation functions (`sum`, `mean`, `std`) to get summary statistics per group.
5. **Filters for a bigger family** — houses with at least 4 bedrooms, 2 bathrooms, and 2000 sqft of living area.
6. **Groups by number of floors** to find the average price per floor category, and interprets the result: houses with more floors tend to have higher average prices, though it's not a perfectly straight relationship since other things like location and condition matter too.

## Tools used
- Python
- Pandas
- NumPy
- Jupyter Notebook

## How to run it
1. Make sure `kc_house_data.csv` is in the same folder as the notebook.
2. Open the notebook and run the cells from top to bottom.
