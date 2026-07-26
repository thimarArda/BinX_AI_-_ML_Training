# Day 3 - NumPy: Numerical Computing 

## What the notebook is about
This notebook is for practicing NumPy basics. It covers creating arrays, indexing, slicing, boolean masking, and broadcasting.

## What the notebook does
1. **Creates arrays** of different dimensions (1D, 2D, 3D, and a 4x4 2D array) using `np.array()`, and checks their `ndim` and `dtype`.
2. **Indexes arrays** to grab individual elements from both 1D and 2D arrays.
3. **Slices arrays** to pull out specific rows, columns, or smaller squares from inside a bigger array.
4. **Builds a 4x4 array filled with 1-16** two different ways: once with a plain nested loop (just for practicing Python syntax), and once the proper NumPy way using `np.arange()` and `.reshape()`.
5. **Uses slicing again** to grab the second column and the last row of that array.
6. **Uses a boolean mask** to filter out only the values greater than the array's mean.
7. **Uses broadcasting** to add a 1D row array to every row of a 2D array, and checks the result manually.

## Tools used
- Python
- NumPy
- Jupyter Notebook

## How to run it
Just open the notebook and run the cells from top to bottom, no dataset needed for this one, just NumPy.
