# Neural Networks Fundamentals

## Overview

This notebook introduces the fundamental concepts of **Neural Networks**, focusing on activation functions, forward propagation, and loss functions. It then applies these concepts in a hands-on experiment using the **Heart Disease dataset** previously used in Project One.

The practical section demonstrates how real patient data can pass through a simple neural network using **NumPy**, from the input features through the hidden layer and finally to a prediction.

---

## Contents

| Section | Topic | Description |
|---|---|---|
| **1** | Why Activation Functions Matter | Explains why neural networks need non-linearity and how activation functions allow them to learn complex patterns. |
| **2** | Common Activation Functions | Introduces ReLU, Sigmoid, Softmax, and Tanh, including their formulas, output ranges, and typical use cases. |
| **3** | Forward Propagation | Explains how input data moves through the network using weights, biases, and activation functions to produce a prediction. |
| **4** | The Loss Function | Explains how prediction error is measured and introduces common loss functions for regression and classification tasks. |
| **5** | Hands-On Lab | Applies the concepts using the Heart Disease dataset, including preprocessing, activation function visualization, and a manual forward pass using 10 patient samples. |

---

## Dataset

The notebook uses the:

```text
heart_2020_raw.csv
```

dataset, which contains health-related information used to predict whether a patient has heart disease.

The target variable is **`HeartDisease`**, making this a **binary classification** problem.

For the practical implementation:

- **ReLU** is used in the hidden layer.
- **Sigmoid** is used in the output layer.
- **Binary Cross-Entropy** is the appropriate loss function.

---

## How to Use This Notebook

### 1. Prepare the Dataset

Make sure `heart_2020_raw.csv` is available before running the notebook.

If the dataset is included with the repository, keep it in the same directory as the notebook:

```text
project/
├── Neural_Networks.ipynb
├── heart_2020_raw.csv
└── README.md
```

If it is not included, download the dataset from the source provided with the training materials and place the CSV file in the notebook's directory.

### 2. Install the Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Open the Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open the neural network notebook and run the cells **in order from top to bottom**.

### 4. Follow the Hands-On Lab

The practical section will:

1. Load and preprocess the Heart Disease dataset.
2. Remove duplicate records and encode categorical features.
3. Split and scale the data.
4. Visualize ReLU, Sigmoid, and Tanh.
5. Select the appropriate activation and loss functions for the binary classification task.
6. Perform a manual forward propagation using 10 real training samples.

> **Note:** The manually initialized weights are random and have not been trained. Therefore, the resulting probabilities demonstrate the mechanics of forward propagation and should not be interpreted as actual medical predictions.