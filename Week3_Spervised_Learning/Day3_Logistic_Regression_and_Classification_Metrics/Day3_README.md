# Day 3 – Logistic Regression & Classification Metrics

This notebook  teaches **Logistic Regression** which is a machine learning model for classification and how to check if it's performing well using Evaluation Metrecies.

## Structure of the Notebook

**Section 1 – How Logistic Regression Works**
The theory: what logistic regression is, how it's different from linear regression, and the math behind it (weights, bias, the sigmoid function).

**Section 2 – Training the Model**
Hands-on code using the built-in **Breast Cancer** dataset from scikit-learn. Trains a logistic regression model, and shows why scaling your data (`StandardScaler`) helps the model converge properly.

**Section 3 – Evaluating the Model**
Explains and calculates the key metrics used to judge a classifier:
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Log Loss
- AUC-ROC Curve

**Section 4 – Hands-on Lab**
Practice exercise using a diabetes dataset — train a model, generate a confusion matrix, compute precision/recall/F1, and interpret what the results  mean in a medical context(Breast cancer).

## How to use it

1. **Open the file** in Jupyter Notebook, JupyterLab, or Google Colab (upload the `.ipynb` file).
2. Make sure you have these Python packages installed:
   ```
   pip install scikit-learn matplotlib
   ```
3. Run the cells **from top to bottom, in order**, later cells depend on variables (like `model`, `X_train`, `predictions`) created earlier.
4. Read the markdown (text) cells for explanations, then run the code cells right after them to see the results.


