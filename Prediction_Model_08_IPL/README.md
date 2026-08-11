# 🏏 IPL First Innings Score Prediction

> **Predicting the first innings score of an IPL match using Machine Learning and historical ball-by-ball match data.**

This project applies **Regression-based Machine Learning** techniques to estimate the **final first innings score** during an IPL match. The model is trained on **ball-by-ball data from IPL Seasons 1–10 (2008–2017)**, learning patterns from thousands of deliveries to generate accurate score predictions based on real-time match conditions.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Requirements](#-requirements)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Pipeline Overview](#-pipeline-overview)
- [Models Used](#-models-used)
- [Results](#-results)
- [Predictions](#-predictions)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)

---

## 📌 Project Overview

Cricket is a game of strategy, momentum, and data. This project demonstrates how **predictive analytics** can transform sports data into meaningful insights by building a **machine learning model** that predicts the **first innings total score** of an IPL match in real-time.

The model uses match statistics available at any point during the first innings (such as current runs, wickets, overs, and recent scoring rate) to estimate the final total.

---

## 📊 Dataset

The dataset contains **ball-by-ball information** for IPL matches played between **2008 and 2017 (Seasons 1–10)**.

### Original Dataset Columns

| Column           | Description                        |
| ---------------- | ---------------------------------- |
| `mid`            | Match ID                           |
| `date`           | Date of the match                  |
| `venue`          | Stadium where the match was played |
| `batting_team`   | Team batting                       |
| `bowling_team`   | Team bowling                       |
| `batsman`        | Batsman facing the delivery        |
| `bowler`         | Bowler delivering the ball         |
| `runs`           | Runs scored on that delivery       |
| `wickets`        | Total wickets lost so far          |
| `overs`          | Overs completed so far             |
| `runs_last_5`    | Runs scored in the last 5 overs    |
| `wickets_last_5` | Wickets lost in the last 5 overs   |
| `striker`        | Striker batsman ID                 |
| `non-striker`    | Non-striker batsman ID             |
| `total`          | Final total score of the innings   |

> **Note:** Place the IPL ball-by-ball dataset as `data/players.xlsx` (must be an Excel file with the columns listed above).

### Consistent Teams Used

The model only considers matches played by these 8 consistent IPL teams:

- Chennai Super Kings
- Delhi Daredevils (now Delhi Capitals)
- Kings XI Punjab (now Punjab Kings)
- Kolkata Knight Riders
- Mumbai Indians
- Rajasthan Royals
- Royal Challengers Bangalore
- Sunrisers Hyderabad

---

## ✨ Features

The model uses the following features after preprocessing and encoding:

**Numerical Features (5):**

- `runs` - Current runs scored
- `wickets` - Wickets lost
- `overs` - Overs bowled
- `runs_last_5` - Runs in last 5 overs
- `wickets_last_5` - Wickets in last 5 overs

**Categorical Features (16 one-hot encoded):**

- 8 batting team indicators
- 8 bowling team indicators

**Target Variable:**

- `total` - Final first innings score

---

## 🛠️ Tech Stack

| Technology       | Purpose                                     |
| ---------------- | ------------------------------------------- |
| **Python 3**     | Primary programming language                |
| **Pandas**       | Data manipulation and analysis              |
| **NumPy**        | Numerical computing                         |
| **Scikit-learn** | Machine Learning algorithms & preprocessing |
| **Matplotlib**   | Data visualization                          |
| **Seaborn**      | Statistical data visualization              |
| **Joblib**       | Model persistence and loading               |
| **Openpyxl**     | Excel file handling                         |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/IPL-Prediction-Model.git
cd IPL-Prediction-Model
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the Dataset

Place your IPL ball-by-ball dataset Excel file at:

```
data/
└── players.xlsx
```

---

## 📚 Requirements

All Python dependencies are listed in [`requirements.txt`](requirements.txt):

```
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0
openpyxl>=3.0.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
IPL_Prediction_Model/
│
├── data/
│   └── players.xlsx                     # IPL ball-by-ball dataset (Excel)
│
├── notebooks/
│   └── original_notebook.ipynb          # Original Colab notebook (reference)
│
├── src/
│   ├── __init__.py                      # Package initializer
│   ├── data_loader.py                   # Data loading & EDA functions
│   ├── preprocessing.py                 # Data cleaning pipeline
│   ├── feature_engineering.py           # Feature encoding & transformation
│   ├── train.py                         # Model training (6 algorithms)
│   ├── evaluate.py                      # Model evaluation & visualization
│   ├── predict.py                       # Prediction function & test cases
│   └── utils.py                         # Constants & helper functions
│
├── models/
│   ├── forest_model.pkl                 # Random Forest model
│   ├── tree_model.pkl                   # Decision Tree model
│   ├── neural_nets_model.pkl            # Neural Network model
│   ├── lasso_model.pkl                  # Lasso Regression model
│   ├── linreg_model.pkl                 # Linear Regression model
│   ├── svm_model.pkl                    # Support Vector Regression model
│   └── trained_model.pkl                # Best model (Random Forest) alias
│
├── outputs/
│   ├── plots/
│   │   ├── correlation_matrix.png       # Feature correlation heatmap
│   │   └── model_comparison.png         # Model performance barplot
│   ├── predictions.csv                  # Test case predictions
│   └── metrics.txt                      # Evaluation metrics report
│
├── main.py                              # Main pipeline orchestrator
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── .gitignore                           # Git ignore rules
```

---

## 🚀 How to Run

### Run the Complete Pipeline

```bash
python main.py
```

This will execute the entire ML pipeline:

1. Load the dataset from `data/players.xlsx`
2. Preprocess the data (clean, filter, transform)
3. Engineer features (encode, encode, build DataFrame)
4. Split data into training (80%) and testing (20%)
5. Train all 6 regression models
6. Evaluate models and save metrics
7. Generate and save visualizations
8. Save trained models to `models/` directory
9. Run test predictions and save results

### Skip Training (if models already exist)

```bash
python main.py --skip-train
```

### Custom Data Path

```bash
python main.py --data-path "path/to/your/dataset.xlsx"
```

### Make a Single Prediction

```python
from src.predict import predict_score, load_model

# Load the trained model
model = load_model("models/forest_model.pkl")

# Predict score for a match situation
predicted_score = predict_score(
    batting_team="Mumbai Indians",
    bowling_team="Chennai Super Kings",
    overs=10.0,
    runs=75,
    wickets=2,
    runs_last_5=35,
    wickets_last_5=1,
    model=model
)

print(f"Predicted Score: {predicted_score}")
```

### Run the Original Notebook

The original Colab notebook is preserved at `notebooks/original_notebook.ipynb` for reference.

---

## 🔄 Pipeline Overview

The project follows a structured ML pipeline:

```text
IPL Ball-by-Ball Dataset (data/players.xlsx)
                    │
                    ▼
         ┌─────────────────────┐
         │    DATA LOADER      │  ← Load Excel, display info (shape, head, describe, etc.)
         └─────────┬───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   PREPROCESSING     │  ← Drop irrelevant columns, filter consistent teams,
         └─────────┬───────────┘     remove deliveries before 5th over
                    │
                    ▼
         ┌─────────────────────┐
         │ FEATURE ENGINEERING │  ← Label encode teams, one-hot encode,
         └─────────┬───────────┘     build 21-feature DataFrame
                    │
                    ▼
         ┌─────────────────────┐
         │    TRAIN/TEST SPLIT │  ← 80% training, 20% testing
         └─────────┬───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   MODEL TRAINING    │  ← 6 regression models trained
         └─────────┬───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    EVALUATION       │  ← R², MAE, MSE, RMSE, plots
         └─────────┬───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   SAVE MODELS       │  ← joblib.dump() all models
         └─────────┬───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   PREDICTIONS       │  ← 6 test cases, save to CSV
         └─────────────────────┘
```

---

## 🤖 Models Used

Six different regression algorithms are trained and compared:

| #   | Model                               | Description                                                            |
| --- | ----------------------------------- | ---------------------------------------------------------------------- |
| 1   | **Decision Tree Regressor**         | Tree-based model that learns decision rules from features              |
| 2   | **Linear Regression**               | Linear approach to model relationship between features and target      |
| 3   | **Random Forest Regressor**         | Ensemble of decision trees (best performer)                            |
| 4   | **Lasso Regression (LassoCV)**      | Linear regression with L1 regularization and automatic alpha selection |
| 5   | **Support Vector Regression (SVR)** | SVR with RBF kernel                                                    |
| 6   | **Neural Network (MLPRegressor)**   | Multi-layer perceptron with logistic activation (500 max iterations)   |

---

## 📈 Results

### Model Performance (R² Scores)

| Model             | Train Score | Test Score | MAE      | RMSE     |
| ----------------- | ----------- | ---------- | -------- | -------- |
| **Random Forest** | **99.07%**  | **93.08%** | **4.61** | **7.82** |
| Decision Tree     | 99.98%      | 86.13%     | 3.97     | 11.07    |
| Neural Network    | 86.27%      | 84.68%     | 8.25     | 11.64    |
| Linear Regression | 65.91%      | 65.91%     | 13.10    | 17.36    |
| Lasso Regression  | 64.89%      | 64.96%     | 13.12    | 17.60    |
| SVR               | 57.48%      | 57.45%     | 14.69    | 19.40    |

> **Random Forest** was selected as the best model due to its highest test R² score (93.08%) and lowest RMSE (7.82).

### Test Predictions

| Test        | Batting Team          | Bowling Team        | Predicted | Actual |
| ----------- | --------------------- | ------------------- | --------- | ------ |
| Test 1      | Delhi Daredevils      | Chennai Super Kings | 147       | 147    |
| Test 2      | Mumbai Indians        | Kings XI Punjab     | 189       | 176    |
| Live Test 1 | Kings XI Punjab       | Rajasthan Royals    | 178       | 185    |
| Live Test 2 | Kolkata Knight Riders | Chennai Super Kings | 175       | 172    |
| Live Test 3 | Delhi Daredevils      | Mumbai Indians      | 107       | 110    |
| Live Test 4 | Kings XI Punjab       | Chennai Super Kings | 147       | 153    |

---

## 🔮 Predictions

The `predict_score()` function uses **manual one-hot encoding** to transform team names into the feature vector expected by the model. This approach is identical to the original notebook implementation.

### Features Required for a Single Prediction

- `batting_team` - Name of the batting team
- `bowling_team` - Name of the bowling team
- `overs` - Overs completed in the innings
- `runs` - Current runs scored
- `wickets` - Wickets lost
- `runs_last_5` - Runs scored in the last 5 overs
- `wickets_last_5` - Wickets lost in the last 5 overs

---

## 🚀 Future Improvements

- [ ] **Deep Learning Integration** - Implement LSTM/RNN models for sequence-based prediction
- [ ] **Second Innings Prediction** - Predict winning probability and second innings score
- [ ] **Live API Integration** - Connect to live IPL APIs for real-time predictions
- [ ] **Web Application** - Deploy as an interactive Streamlit or Flask web app
- [ ] **Advanced Ensemble Models** - Implement XGBoost, LightGBM, and CatBoost
- [ ] **Cloud Deployment** - Deploy as a cloud-based prediction service (AWS/GCP/Azure)
- [ ] **Hyperparameter Tuning** - Use GridSearchCV/RandomizedSearchCV for optimization

---

## 🤝 Contributing

Contributions, ideas, and improvements are always welcome!

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

It helps the project reach more developers and motivates further improvements!

---

## 📢 Keywords

`Machine Learning` `Data Science` `IPL Analytics` `Cricket Analytics` `Regression Model` `Sports Analytics` `Predictive Analytics` `Python` `Scikit-learn` `Data Visualization` `Artificial Intelligence` `Sports AI` `Portfolio Project` `IPL Score Prediction` `Cricket Prediction` `Data Analytics`

---

<p align="center">Made with ❤️ and 🏏</p>
