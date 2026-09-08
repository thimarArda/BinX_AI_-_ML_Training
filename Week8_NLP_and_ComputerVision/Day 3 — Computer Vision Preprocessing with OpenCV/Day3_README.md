# Image Preprocessing with OpenCV and Keras

## Overview

This notebook is a practical guide for preparing image data for computer vision and deep learning tasks. It covers basic image manipulations with OpenCV, image augmentation techniques using Keras to prevent model overfitting, and the correct preprocessing steps required for transfer learning with pre-trained models like `MobileNetV2`.

## Table of Contents

| Section Number | Content Description |
| --- | --- |
| Section 1 | Covers the importance of standardizing image inputs to improve model convergence and consistency. |
| Section 2 | Demonstrates basic OpenCV techniques: loading images, resizing, handling BGR-to-RGB conversion, and pixel scaling. |
| Section 3 | Shows how to configure Keras `ImageDataGenerator` for transformations like rotation, zoom, and shifts. |
| Section 4 | Explains how to match image formats and scaling to specific pre-trained model requirements (e.g., `MobileNetV2`). |
| Hands-On Lab | Contains practical code blocks to execute each preprocessing step on sample images. |

## Tips for Using This Notebook

* **Run Cells Sequentially:** Execute cells in order top-to-bottom so all required libraries and variables remain defined.
* **Experiment with Parameters:** Tweak values in `ImageDataGenerator` and custom image functions to see visual output changes.
* **Verify Image URLs:** Ensure direct image links ending in formats like `.jpg` or `.png` are used, as web page links trigger a `ValueError`.
* **Understand BGR vs. RGB:** Remember that OpenCV reads images in BGR format, which must be converted to RGB before passing them to Keras or standard visualizer tools.
* **Review Model Preprocessing:** Check model documentation for specific `preprocess_input` requirements if changing the base model architecture.