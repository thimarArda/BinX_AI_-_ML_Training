# ECG Heartbeat Classification Using RNNs and LSTMs

## Overview

This notebook explores **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** networks for classifying ECG heartbeat signals.

Unlike ordinary tabular data, ECG signals are **sequential**. The order of the signal values matters because each value represents a point in the heartbeat waveform. This makes ECG data a suitable example for understanding why sequence-aware architectures such as RNNs and LSTMs are useful.

The main goal of this notebook is to build an LSTM model that can classify individual ECG heartbeat signals and compare its performance with a simpler RNN model.


---

## Dataset

### ECG Heartbeat Categorization Dataset

The dataset used in this notebook is the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) from Kaggle.

It contains preprocessed and segmented ECG heartbeat signals derived from two well-known datasets:

- MIT-BIH Arrhythmia Database
- PTB Diagnostic ECG Database

For this notebook, we use **only the MIT-BIH Arrhythmia portion** because it provides a 5-class heartbeat classification problem that is suitable for comparing RNN and LSTM models.

The MIT-BIH portion contains **109,446 heartbeat samples** divided into five categories. Each heartbeat is represented using 187 signal values, followed by its class label. The original dataset has 188 columns per row, with the final column representing the class.

### Classes

| Class | Label | Description |
|---|---:|---|
| N | 0 | Normal heartbeat |
| S | 1 | Supraventricular heartbeat |
| V | 2 | Ventricular heartbeat |
| F | 3 | Fusion heartbeat |
| Q | 4 | Unknown / unclassifiable heartbeat |

The dataset is highly imbalanced, with normal heartbeats representing the majority of the samples. Because of this, class weights are used during training and metrics such as precision, recall, and F1-score are considered alongside accuracy.

---

## Models Built

### 1. LSTM Model


```text
Input ECG Sequence
       ↓
LSTM (64 units)
       ↓
Dropout
       ↓
Dense (32 units, ReLU)
       ↓
Dense (5 units, Softmax)
       ↓
Heartbeat Class
```

### 2. Simple RNN Model

A plain `SimpleRNN` model is also built using a similar architecture.

It is used as a comparison model to investigate the difference between a basic RNN and an LSTM when working with sequential ECG data.

---


# How to Use This Notebook in Google Colab

## 1. Open the Notebook in Google Colab

Upload or open the `.ipynb` notebook in Google Colab.

You can also open a notebook directly from GitHub using:

**File → Open notebook → GitHub**

---

## 2. Download the Dataset

Download the Kaggle dataset from:

[ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

The downloaded file should be a ZIP archive.

Because the dataset is relatively large, it is recommended to store the ZIP file in **Google Drive** rather than uploading it directly to Colab every time.

---

## 3. Connect Google Drive

Run the Google Drive mounting cell in the notebook:

```python
from google.colab import drive

drive.mount('/content/drive')
```

Authorize Colab when prompted.

---

## 4. Set the Dataset Path

After placing the ZIP file in Google Drive, update the path in the notebook.

For example:

```python
zip_path = '/content/drive/MyDrive/heartbeat.zip'
```

If the ZIP file is inside a folder:

```python
zip_path = '/content/drive/MyDrive/datasets/heartbeat.zip'
```

---

## 5. Extract the Dataset

The notebook extracts the ZIP file into Colab's local storage.

This is recommended because repeatedly reading large CSV files directly from Google Drive can slow down the workflow.

The relevant files for this notebook are:

```text
mitbih_train.csv
mitbih_test.csv
```

---

## 6. Run the Notebook

Run the cells in order


## Requirements

The notebook uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Google Colab
- Google Drive

Most of these libraries are already available in Google Colab.

---

