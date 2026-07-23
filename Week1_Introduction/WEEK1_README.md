# Week 1 - Python & Data Science Foundations 

## What this is about
This is the first week of the BinX AI & ML Internship Program (Phase 1: Foundations). The goal of the week was to get comfortable with the core tools every AI/ML project is built on: Python itself, NumPy, Pandas, and Matplotlib, all inside the Jupyter Notebook workflow. The week wraps up with a mini-project that uses all of them together.

## Notebooks in this folder

| Day | Notebook | What it covers |
|-----|----------|-----------------|
| Day 1 | `Day1_env_setup.ipynb` | Setting up the Python environment and checking that NumPy, Pandas, and Matplotlib are installed correctly. |
| Day 2 | `Day2_python_basics.ipynb` | Core Python practice: writing functions, list comprehensions, and a small intro to classes (OOP). |
| Day 3 | `Day3_NumPy_Numerical_Computing.ipynb` | NumPy basics: creating arrays, indexing, slicing, boolean masking, and broadcasting. |
| Day 4 | `Day4_Panda_Tabular_Data.ipynb` | Pandas basics: loading a real dataset, cleaning it, filtering it, and grouping/aggregating it. |
| Day 5 | `Day5_Matplotlib_&_Week_1_Mini-Notebook.ipynb` | Matplotlib basics (bar, histogram, scatter plots) and the Week 1 mini-project: helping a family find a suitable house using everything learned this week. |

## The Week 1 mini-project (Day 5)
The mini-project is a small case study: a family of 7 is looking for a house in King County within a $300,000 budget. The notebook:
- Loads and cleans the King County housing dataset with Pandas.
- Filters it down to the houses that actually fit the family's needs.
- Derives new features (like total area and price per sqft) using NumPy.
- Visualizes the results with a bar chart, histogram, and scatter plot using Matplotlib.
- Ranks the suitable houses from cheapest to most expensive as a conclusion.

## Tools used across the week
- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Git & GitHub

## How to run everything
1. Make sure `kc_house_data.csv` is in the same folder as the Day 4 and Day 5 notebooks (they're the ones that use it).
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Open the notebooks in order (Day 1 → Day 5) and run the cells from top to bottom.
