# Day 2 - Probability and Distributions

This notebook covers Day 2 of Week 2 (Math Foundations & EDA) from the BinX Tech AI & ML
Internship Program. It introduces the core probability concepts used in machine learning
and applies them through code simulations.

## Learning Objectives

- Apply the complement, addition, and multiplication rules of probability.
- Explain conditional probability and Bayes' theorem and where they appear in ML.
- Recognize the normal, binomial, and uniform distributions.

## Structure

**Section 1: Learnings**
- Probability basics and definition
- Core probability rules: addition, multiplication, complement (with worked examples)
- Conditional probability
- Bayes' theorem, including its application in Naive Bayes classifiers
- Common distributions: normal, binomial, uniform

**Section 2: Hands-On Lab**
- Step 1: Simulate 10,000 coin flips with NumPy and confirm the proportion of heads
  approaches 0.5, using both a loop and `np.mean()`.
- Step 2: Sample from a normal distribution with `np.random.normal` and plot a histogram
  to confirm the bell shape.
- Step 3: Solve a conditional probability problem by hand, then verify the result with
  a simulation.

## Requirements

- Python 3.10+
- numpy
- matplotlib

Install with:

```
pip install numpy matplotlib
```

## How to Use This Notebook

1. Open the notebook with Jupyter:
   ```
   jupyter notebook "Day2_Probability_and_Distributions.ipynb"
   ```
2. Run the cells in order from top to bottom, since the hands-on lab in Section 2
   depends on NumPy and Matplotlib being imported earlier in the notebook.
3. Read the markdown cells before each code cell — they explain the rule or concept
   before it is applied in code.
4. Try changing the simulation parameters (e.g., number of coin flips, or `loc`/`scale`
   in `np.random.normal`) to see how the results and plots change.
