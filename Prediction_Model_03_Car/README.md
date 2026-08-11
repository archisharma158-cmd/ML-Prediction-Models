# Car Price Prediction Model using Logistic Regression

## Overview
This project predicts whether a used car belongs to the **higher-price** or **lower-price** category based on its features using a **Logistic Regression Classification Model**.

The dataset contains information about used cars such as company, manufacturing year, kilometers driven, and fuel type. Since Logistic Regression is a classification algorithm, the continuous car prices were converted into two classes using the dataset's median price.

---

## Problem Statement

The objective of this project is to classify a used car as either:

- **0 → Below Median Price**
- **1 → Above or Equal to Median Price**

using various car attributes.

---

## Dataset Information

- **Total Records:** 815
- **Features Used:**
  - Car Name
  - Company
  - Manufacturing Year
  - Kilometers Driven
  - Fuel Type

### Sample Features

| Feature | Description |
|----------|-------------|
| name | Car model |
| company | Manufacturer |
| year | Manufacturing year |
| kms_driven | Distance travelled |
| fuel_type | Petrol/Diesel/CNG/LPG |

---

## Target Variable

Since the original dataset contains continuous prices, the target was converted into binary classes.

**Median Price:** `₹299,999`

| Class | Meaning |
|-------|---------|
| 0 | Price below ₹299,999 |
| 1 | Price greater than or equal to ₹299,999 |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Machine Learning Pipeline

1. Data Loading
2. Data Cleaning
3. Feature Selection
4. Label Creation using Median Price
5. Encoding Categorical Features
6. Train-Test Split
7. Model Training using Logistic Regression
8. Model Evaluation
9. Hyperparameter Tuning

---

## Model Used

**Logistic Regression**

Logistic Regression was selected because the problem was converted into a binary classification task by dividing car prices into two categories based on the median price.

---

## Hyperparameter Tuning

The Logistic Regression model was optimized by tuning important hyperparameters to improve prediction performance.

Examples include:

- Regularization Strength (`C`)
- Solver
- Maximum Iterations (`max_iter`)
- Penalty

---

## Model Performance

### Accuracy

**80.49%**

### Confusion Matrix

```
[[32 11]
 [ 5 34]]
```

### Classification Report

| Class | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| 0 | 0.86 | 0.74 | 0.80 |
| 1 | 0.76 | 0.87 | 0.81 |

### Overall Metrics

- **Accuracy:** 80.49%
- **Macro Precision:** 0.81
- **Macro Recall:** 0.81
- **Macro F1-Score:** 0.80

---

## Project Structure

```
Car_Price_Prediction/
│
├── data/
│   └── car.csv
│
├── notebook/
│   └── Car_Price_Prediction.ipynb
│
├── models/
│   └── logistic_regression_model.pkl
│
├── README.md
│
└── requirements.txt
```

---

## Future Improvements

- Train additional machine learning models such as:
  - Decision Tree
  - Random Forest
  - XGBoost
  - Support Vector Machine

- Perform extensive hyperparameter tuning using GridSearchCV.
- Add feature engineering techniques.
- Deploy the model using Flask or Streamlit.
- Build a user-friendly web application for real-time predictions.

---

## Conclusion

This project demonstrates the complete workflow of a machine learning classification problem using Logistic Regression. By converting continuous car prices into binary categories based on the median price, the model achieved an accuracy of approximately **80.5%**, making it effective for classifying used cars into higher or lower price groups.

---

## Author

**Archi Sharma**

GitHub: *https://github.com/archisharma158-cmd*

LinkedIn: *(https://www.linkedin.com/in/archi-sharma-05a3a2371?utm_source=share_via&utm_content=profile&utm_medium=member_android)*
