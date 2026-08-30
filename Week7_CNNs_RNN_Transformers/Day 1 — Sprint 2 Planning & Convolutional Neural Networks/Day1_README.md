# Day 1 — Convolutional Neural Networks

---

## Overview

This notebook introduces the fundamentals of **Convolutional Neural Networks (CNNs)** and explains why they are well suited for image processing. It covers the limitations of Dense networks when working with images, how convolution filters detect patterns, the basic architecture of a CNN, and the efficiency gained through parameter sharing. The notebook also includes a hands-on exercise where a manually defined edge-detection filter is applied to a sample image and its feature map is visualized.

---

## Table of Contents


| Section                                          | Description                                                                                                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Introduction**                                 | An introduction to CNNs and why different types of data require different neural network architectures.                                                          |
| **Section 1: Why Dense Networks Fail on Images** | Explains why flattening large images and processing them with Dense layers results in a very large number of parameters and loses important spatial information. |
| **Section 2: Convolution**                       | Explains convolution, filters, feature maps, stride, padding, parameter sharing, translation invariance, and feature hierarchy.                                  |
| **Section 3: CNN Architecture**                  | Introduces the main components of a CNN, including convolutional layers, activation functions, pooling layers, flattening, and Dense layers.                     |
| **Section 4: Convolution vs Dense Layers**       | Compares Dense and convolutional layers using a **740 × 740 RGB image**, showing how parameter sharing makes CNNs more efficient.                                |
| **Section 5: Hands-On Lab**                      | Applies a manually defined edge-detection filter to a sample **zebra image**, visualizes the feature map, and compares the number of weights with a Dense layer. |


---

## Running the Notebook in Google Colab

To run this notebook in **Google Colab**:

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook** and upload the `.ipynb` file.
3. Upload the required image using the **Files** panel on the left, or update the image path in the code.
4. Run the cells from top to bottom using **Shift + Enter**.
5. Make sure the required libraries such as **NumPy, Matplotlib, and Pillow** are available in the Colab environment.

Or you can run it on you VSC if you have the proper python setups
