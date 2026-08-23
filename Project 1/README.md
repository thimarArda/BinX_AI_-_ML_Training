# Heart Disease Classification System 

This project is part of the AI & Machine Learning track. The previous version of this project used a smaller heart disease dataset with around 1,000 samples. For this version, I decided to work with a larger dataset containing around 320,000 responses.



## Dataset

**Personal Key Indicators of Heart Disease** is based on the CDC BRFSS 2020 survey. The dataset contains **319,795 respondents** and features related to health, lifestyle, and demographics. The target variable, `HeartDisease`, indicates whether a respondent reported having heart disease.


## Why this dataset?

The previous dataset was smaller and more balanced. In this project, I wanted to work with a larger and more challenging dataset. Heart disease cases represent only around **9%** of the data, creating a class imbalance problem. Because of this, accuracy alone is not enough to evaluate the models.

The dataset contains health conditions, lifestyle information, and demographic characteristics based mainly on self-reported survey responses rather than detailed clinical measurements.


## What I Did

The project follows a complete machine learning workflow:

* Data cleaning and preparation
* Exploratory Data Analysis
* Feature engineering
* Feature preprocessing and encoding
* Model training and comparison
* Cross-validation and final evaluation

Because of the class imbalance, I focused on **precision, recall, F1-score, balanced accuracy, and ROC-AUC**, rather than accuracy alone.

I compared **Logistic Regression** and a **shallow Random Forest**, and also tested whether engineered features improved the results.

## Results

| Metric            | Logistic Regression | Random Forest | Random Forest + Engineered Features |
| ----------------- | ------------------: | ------------: | ----------------------------------: |
| Accuracy          |               0.743 |         0.719 |                               0.731 |
| Balanced Accuracy |               0.755 |         0.756 |                               0.754 |
| Precision         |               0.227 |         0.216 |                               0.221 |
| Recall            |               0.770 |         0.802 |                               0.782 |
| F1-score          |               0.351 |         0.340 |                               0.345 |
| ROC-AUC           |               0.832 |         0.828 |                               0.827 |

The models performed similarly. Logistic Regression achieved the highest precision, F1-score, and ROC-AUC, while Random Forest achieved the highest recall.

The engineered features also did not clearly improve the results, showing that adding more features does not always improve model performance.

## Limitations

The dataset is based mainly on self-reported survey responses, which may contain reporting errors or bias. Also, because there is no respondent ID, identical rows may not always represent true duplicates.

**Disclaimer :** This project is for educational purposes and is not intended for medical diagnosis or clinical decision-making.
