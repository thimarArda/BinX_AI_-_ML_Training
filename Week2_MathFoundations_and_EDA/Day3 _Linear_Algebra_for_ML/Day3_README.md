# Day 3 - Linear Algebra for Machine Learning

## What this is about
This notebook covers the linear algebra concepts that machine learning is built on: vectors, matrices, the dot product, and matrix multiplication. It explains each concept from a computer science, physics, and mathematical point of view, then ties it back to how these operations are actually used in ML models like linear and logistic regression.

## Notebook Contents
1. **Vectors** - what they and how they represent data.
2. **Vectorization** - processing whole vectors at once with NumPy instead of looping.
3. **Matrices** - representing a dataset (samples x features) and reading its shape.
4. **The dot product** - calculating it by hand and with `np.dot()`, and how it drives predictions.
5. **Matrix multiplication** - using `@` to predict on many samples at once, and the shape-matching rule.
6. **Hands-on lab** - building a matrix, checking a dot product by hand, predicting with `@`, and fixing a shape-mismatch error.

## Tools used
- Python
- NumPy
- Jupyter Notebook

## How to run it
Open the notebook and run the cells from top to bottom, no dataset needed, this one is concept and math based rather than dataset based.

## If the images do not load
This notebook has two images embedded with plain HTML `<img>` tags (`physics_vectors.PNG` and `vectors_pov.PNG`), not with Markdown image syntax. This means the images are not stored inside the notebook file itself, they are separate image files that the notebook expects to find in the same folder as the `.ipynb` file.

If the images do not show up when you open the notebook, it almost always means the image files were not placed next to the notebook. To fix this:
- Make sure `physics_vectors.PNG` and `vectors_pov.PNG` are in the exact same folder as `Day3__Linear_Algebra_for_ML.ipynb`.
- Check that the file names match exactly, including capitalization (`.PNG` vs `.png` matters on some systems, especially on GitHub and Linux).
- If the images are still missing entirely, the explanations before and after each image still describe the same idea in text, so the notebook can still be followed without them, you would just be missing the visual reference for that section.

