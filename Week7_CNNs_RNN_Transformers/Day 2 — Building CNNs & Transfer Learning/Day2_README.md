# Melanoma Detection with CNNs

## Overview

This notebook explores image classification for melanoma skin cancer using **Convolutional Neural Networks (CNNs)**. The goal is to classify skin lesions into two classes: **Benign** and **Malignant**.

I trained and compared three different approaches:

1. **Baseline CNN** – a simple CNN built from scratch.
2. **CNN with Data Augmentation** – the same idea with random image transformations to improve generalization.
3. **Transfer Learning with MobileNetV2** – using a pre-trained model to extract useful image features.

## Models

### 1. Baseline CNN

The first model is a basic CNN using:

* `Conv2D` for feature extraction
* `MaxPooling2D` for reducing the image size
* `Flatten` to prepare the features for classification
* `Dense` layers for the final prediction

The model learns all of its features directly from the melanoma dataset.

### 2. CNN with Data Augmentation

The second model adds data augmentation using:

* `RandomFlip`
* `RandomRotation`
* `RandomZoom`

These transformations create slightly different versions of the training images and can help reduce overfitting.

### 3. Transfer Learning with MobileNetV2

The third approach uses **MobileNetV2**, a model that was already trained on ImageNet. Instead of learning all image features from the beginning, the model uses its existing learned features and adapts them to the melanoma classification task.

## Results

| Model                   | Validation Accuracy |
| ----------------------- | ------------------: |
| Baseline CNN            |          **87.16%** |
| CNN + Data Augmentation |          **86.32%** |
| MobileNetV2             |          **87.92%** |

### Conclusion

**MobileNetV2 performed the best**, achieving a validation accuracy of about **87.92%**.

The baseline CNN performed slightly better than the augmented CNN in this experiment. Although augmentation can help with generalization, it did not improve the validation accuracy here. This could be related to the augmentation settings or the limited training time.

Overall, the results show that **transfer learning was the most effective approach among the three models tested**.

## How to Run the Notebook

The notebook was created in **Google Colab**.

1. Open the notebook in Google Colab.
2. Mount Google Drive when prompted.
3. Make sure the dataset is available at:

```text
/content/drive/MyDrive/Datasets/Skin Cancer Dataset/skin cancer dataset.zip
```

4. Run the cells in order, or use **Runtime → Run all**.

The notebook extracts the dataset and creates the training and testing folders needed to load the images.

The zip file of the dataset is uploaded alongside the notebook in the same folder.
If the dataset is stored somewhere else, update the `zip_path` variable with its correct location.
