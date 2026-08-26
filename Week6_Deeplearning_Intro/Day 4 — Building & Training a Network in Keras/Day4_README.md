 # Building & Training a Network in Keras

## Overview

This notebook is the hands-on lab for **Day 4: Building & Training a Network in Keras**. It walks through building a first Artificial Neural Network (ANN) with **TensorFlow/Keras**, training it, reading its training history, and applying **Dropout** and **Early Stopping** to fight overfitting — then comparing the result to see whether deep learning can outperform a classic ML baseline (Day 1) on the same problem.

## Learning Objectives

- Build a neural network with the Keras **Sequential API**.
- **Compile**, **train**, and **evaluate** the network, and read its training history (loss/accuracy curves).
- Apply **batch normalization** and **dropout** to stabilize training and reduce overfitting.
- Compare deep learning performance against a classic ML baseline.

## Dataset

The notebook uses a **cardiac (heart disease) dataset** (`heart_2020_raw.csv`). The task is **binary classification**: predict whether a person has heart disease based on health indicators (smoking, alcohol use, stroke history, physical activity, sex, and other features). The dataset is **imbalanced** — roughly 90% of records are "No Disease" — which is central to how the results should be interpreted.

## Notebook Structure

1. **Loading libraries** — NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and TensorFlow/Keras.
2. **Data preprocessing and encoding** — dropping duplicates and mapping categorical/binary columns to numeric values.
3. **Data splitting** — into training, validation, and test sets (stratified on the target).
4. **Feature scaling** — standardizing features with `StandardScaler`.
5. **Model 1 — Baseline network** — a `Dense(64) → Dense(32) → Dense(1, sigmoid)` network, compiled with `adam` and `binary_crossentropy`, trained for 50 epochs.
6. **Training curves & evaluation** — plotting loss/accuracy history, evaluating on the test set, and reviewing the classification report/confusion matrix.
7. **Model 2 — Dropout** — the same architecture with a `Dropout(0.3)` layer added to reduce overfitting.
8. **Model 3 — Dropout + Early Stopping** — adds an `EarlyStopping` callback (monitoring validation loss) on top of Model 2.
9. **Model comparison** — a summary table comparing accuracy, macro F1-score, and heart-disease recall/F1 across the three models.

## Key Takeaways

- All three models reach **~91% overall accuracy**, but accuracy is misleading here because of the class imbalance.
- The models struggle to detect the **minority class (Heart Disease)**: recall for that class ranges from only **4–10%**, meaning most true heart disease cases are missed.
- **Model 1 (Baseline)** turned out to be the best of the three, with the highest macro F1-score (0.56) and the best heart-disease recall (10%) — Dropout and Early Stopping did not fix the underlying class-imbalance problem in this run.
- **Next steps** identified in the notebook: address class imbalance directly (e.g., `class_weight='balanced'`, oversampling/undersampling like SMOTE), lower the decision threshold, and optimize for recall rather than accuracy.

## Tools Used

- TensorFlow / Keras
- Scikit-learn (preprocessing, splitting, metrics)
- Matplotlib / Seaborn
- Jupyter / Colab (GPU recommended)

## How to Run 

This notebook was written and run on **Google Colab**, and the dataset was accessed through a path pointing to a copy of the CSV uploaded to **Google Drive** (`/content/drive/MyDrive/Datasets/heart_2020_raw.csv`). Running this notebook on **Colab is recommended**, since it gives free access to cloud compute/GPU resources and avoids any local setup.

**If you want to run it locally instead:**
> Follow the requirement text file for a proper environment settings
1. Download the dataset (`heart_2020_raw.csv`).
2. Place it in the **same folder** as this notebook.
3. Replace the data-loading path in the code with just the file name, e.g.:
```python
   df = pd.read_csv("heart_2020_raw.csv")
```
