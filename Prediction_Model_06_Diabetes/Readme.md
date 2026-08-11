# Diabetes Prediction Model using Support Vector Machine (SVM)

## Overview

This project is a **Machine Learning-based Diabetes Prediction System** that predicts whether a person is likely to have diabetes based on various medical attributes. The model is trained using the **Pima Indians Diabetes Dataset** and utilizes a **Support Vector Machine (SVM)** classifier for binary classification.

The objective of this project is to demonstrate the complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and prediction.

---

## Problem Statement

Diabetes is one of the most common chronic diseases worldwide. Early prediction can help healthcare professionals identify high-risk individuals and recommend timely medical intervention.

This project aims to build a classification model that predicts whether a patient is diabetic based on diagnostic measurements.

---

## Dataset Information

The dataset contains **768 patient records** with **8 medical features** and **1 target variable**.

### Features

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times the patient has been pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age of the patient |

### Target Variable

| Value | Meaning |
|-------|---------|
| 0 | Non-Diabetic |
| 1 | Diabetic |

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Machine Learning Workflow

1. Data Collection
2. Data Exploration
3. Data Cleaning
4. Feature Selection
5. Data Standardization
6. Train-Test Split
7. Model Training using Support Vector Machine (SVM)
8. Model Evaluation
9. Performance Analysis
10. Prediction on New Data

---

## Model Used

### Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm widely used for binary classification tasks. It works by finding the optimal hyperplane that best separates the two classes while maximizing the margin between them.

SVM performs well on datasets with multiple numerical features and is effective in handling high-dimensional data.

---

## Model Performance

### Classification Report

| Metric | Class 0 | Class 1 |
|---------|---------|---------|
| Precision | 0.80 | 0.75 |
| Recall | 0.89 | 0.59 |
| F1-Score | 0.84 | 0.66 |

### Overall Accuracy

**Accuracy: 79%**

### Additional Metrics

| Metric | Score |
|---------|-------|
| Macro Average Precision | 0.77 |
| Macro Average Recall | 0.74 |
| Macro Average F1-Score | 0.75 |
| Weighted Average Precision | 0.78 |
| Weighted Average Recall | 0.79 |
| Weighted Average F1-Score | 0.78 |

---

## Project Structure

```
Diabetes-Prediction/
│
├── dataset/
│   └── diabetes.csv
│
├── notebooks/
│   └── Diabetes_Prediction.ipynb
│
├── models/
│   └── diabetes_model.pkl
│
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Diabetes-Prediction.git
```

Move into the project directory

```bash
cd Diabetes-Prediction
```

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the notebook or Python script to train the model.

---

## Required Libraries

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

---

## Example Prediction

Input:

```text
Pregnancies: 6
Glucose: 148
BloodPressure: 72
SkinThickness: 35
Insulin: 0
BMI: 33.6
DiabetesPedigreeFunction: 0.627
Age: 50
```

Output:

```text
Prediction: Diabetic
```

---

## Key Learnings

- Data preprocessing and feature engineering
- Data standardization
- Binary classification using SVM
- Model evaluation using Accuracy, Precision, Recall, and F1-score
- Understanding healthcare datasets
- Machine Learning workflow from data preparation to prediction

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Compare with Logistic Regression, Random Forest, XGBoost, and Decision Trees
- Handle missing or zero-valued medical measurements more effectively
- Perform feature importance analysis
- Build an interactive web application using Streamlit or Flask
- Deploy the model on a cloud platform

---

## Results

The Support Vector Machine classifier achieved an overall accuracy of **79%**, demonstrating its effectiveness in predicting diabetes based on patient health indicators. While the model performs well in identifying non-diabetic cases, future improvements can focus on increasing the recall for diabetic patients to reduce false negatives.

---

## Author

**Archi Sharma**

If you found this project helpful, consider giving the repository a ⭐.

GitHub: *https://github.com/archisharma158-cmd*

LinkedIn: *(https://www.linkedin.com/in/archisharma158/)*
