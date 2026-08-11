# Telco Customer Churn Prediction using Random Forest Classifier

## Project Overview

Customer churn is one of the most significant challenges faced by subscription-based businesses. Accurately identifying customers who are likely to leave enables companies to implement targeted retention strategies, improve customer satisfaction, and reduce revenue loss.

This project develops a **Machine Learning-based Customer Churn Prediction Model** using the **Random Forest Classifier**. The model analyzes customer demographics, account information, and subscribed services to predict whether a customer is likely to churn.

---

## Problem Statement

Build a binary classification model that predicts whether a customer will churn based on customer demographics, service usage, billing information, and account details.

---

## Objectives

- Load and preprocess the Telco Customer Churn dataset.
- Handle missing values and encode categorical variables.
- Identify the most influential features affecting customer churn.
- Train a Random Forest Classifier.
- Evaluate the model using multiple classification metrics.
- Provide insights that can help businesses improve customer retention.

---

## Dataset

- **Dataset:** Telco Customer Churn Dataset
- **Total Records:** 7,043
- **Target Variable:** `Churn`

### Features Used

The following features were selected based on their importance in predicting customer churn:

- Contract
- Tenure
- Online Security
- Tech Support
- Online Backup
- Monthly Charges
- Paperless Billing
- Device Protection
- Dependents
- Senior Citizen

The complete dataset contains **19 input features** including:

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

---

## Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Checked for missing values.
- Converted `TotalCharges` to numeric format.
- Handled missing values.
- Encoded categorical variables using Label Encoding.
- Selected relevant features for model training.
- Split the dataset into training (80%) and testing (20%) sets.

---

## Machine Learning Model

**Algorithm Used**

- Random Forest Classifier

### Why Random Forest?

Random Forest was selected because it:

- Handles binary classification effectively.
- Captures complex and non-linear relationships.
- Performs well on structured tabular datasets.
- Reduces overfitting by combining multiple decision trees.
- Provides robust and reliable predictions.

---

## Model Training

The model was trained using:

- **Training Data:** 80%
- **Testing Data:** 20%

Target Variable:

```
Churn
```

---

## Model Performance

### Accuracy

**79.84%**

### Confusion Matrix

```
[[944  92]
 [192 181]]
```

### Classification Report

| Metric | Class 0 | Class 1 |
|---------|---------|---------|
| Precision | 0.83 | 0.66 |
| Recall | 0.91 | 0.49 |
| F1-Score | 0.87 | 0.56 |

### Overall Performance

| Metric | Score |
|---------|-------|
| Accuracy | **0.7984** |
| Precision (Churn) | **0.66** |
| Recall | **0.4853** |
| F1-Score | **0.5604** |
| ROC-AUC Score | **0.8348** |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

```
Load Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Handle Missing Values
        │
        ▼
Encode Categorical Variables
        │
        ▼
Feature Selection
        │
        ▼
Train-Test Split (80:20)
        │
        ▼
Random Forest Classifier
        │
        ▼
Model Evaluation
        │
        ▼
Performance Analysis
```

---

## Key Insights

- Customers with shorter tenure are more likely to churn.
- Contract type plays a major role in predicting churn.
- Customers without Online Security and Tech Support show higher churn tendencies.
- Monthly Charges significantly influence customer retention.
- Service-related features contribute strongly to churn prediction.

---

## Future Improvements

- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Address class imbalance using SMOTE or class weighting.
- Compare performance with XGBoost, LightGBM, and Gradient Boosting.
- Deploy the model as a web application using Flask or Streamlit.
- Implement explainable AI techniques such as SHAP or LIME for model interpretability.

---

## Repository Structure

```
Telco_Customer_Churn_Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── Telco_Customer_Churn_Prediction.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Results Summary

- Successfully preprocessed and prepared the dataset.
- Built a Random Forest-based customer churn prediction model.
- Achieved an **Accuracy of 79.84%**.
- Obtained a **ROC-AUC Score of 0.8348**, indicating good discriminative performance.
- Identified important customer attributes that influence churn, enabling businesses to make informed customer retention decisions.

---

## Author

**Archi Sharma**

Aspiring AI & Machine Learning Engineer | Passionate about Machine Learning, Data Science, and Predictive Analytics.

GitHub: *https://github.com/archisharma158-cmd*

LinkedIn: *https://www.linkedin.com/in/archi-sharma-05a3a2371?utm_source=share_via&utm_content=profile&utm_medium=member_android*
