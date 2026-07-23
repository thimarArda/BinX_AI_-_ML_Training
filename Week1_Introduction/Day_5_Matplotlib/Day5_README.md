# Dream House 

## What this is about
Solving a test case of a family that is looking for a home to buy by applying data cleaning with Pandas, deriving new features with NumPy, and visualizing the results with Matplotlib (bar charts, histograms, and scatter plots) inside a Jupyter Notebook.

## What the family is looking for
- Budget: up to $300,000
- At least 5 bedrooms and 3 bathrooms (enough space for everyone)
- A lot big enough to have a garden

## What the notebook does
1. **Loads the dataset** (`kc_house_data.csv`) and checks it for missing values and duplicates.
2. **Cleans the data** by removing houses with weird/impossible values, like 0 bedrooms or a 33-bedroom house (probably data errors, and definitely not what this family needs).
3. **Filters the houses** down to the ones that actually match the family's budget and space requirements — ends up with 15 suitable houses.
4. **Creates new features** using NumPy and Pandas, like the total area (`Whole_area` = living area + lot area) and price per square foot, to compare the houses more fairly.
5. **Visualizes the data** with a bar chart, a histogram, and a scatter plot to see how bedrooms, area, and price relate to each other.
6. **Ranks the suitable houses** from cheapest to most expensive, so the family knows where to start looking.

## Tools used
- Python
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## How to run it
1. Make sure `kc_house_data.csv` is in the same folder as the notebook.
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Open the notebook and run the cells from top to bottom.
