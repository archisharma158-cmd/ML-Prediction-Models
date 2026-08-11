"""
Prediction module for the IPL Score Prediction project.

This module handles:
- The predict_score function with manual one-hot encoding (exact replica from notebook).
- Loading a saved model using joblib.
- Running all 6 test cases from the notebook.
- Saving predictions to a CSV file.

Manual one-hot encoding is used instead of the ColumnTransformer
to make predictions on new single-instance data straightforward.
The encoding order matches the original notebook exactly.
"""

import os
import numpy as np
import pandas as pd
from joblib import load, dump


def predict_score(
    batting_team,
    bowling_team,
    runs,
    wickets,
    overs,
    runs_last_5,
    wickets_last_5,
    model=None,
):
    """
    Predict the final first innings score using the trained model.

    This function performs manual one-hot encoding for the batting and
    bowling teams, exactly as implemented in the original notebook.

    Parameters
    ----------
    batting_team : str
        Name of the batting team.
    bowling_team : str
        Name of the bowling team.
    runs : float
        Current runs scored.
    wickets : float
        Current wickets lost.
    overs : float
        Current overs bowled.
    runs_last_5 : float
        Runs scored in the last 5 overs.
    wickets_last_5 : float
        Wickets lost in the last 5 overs.
    model : sklearn model, optional
        Trained model to use for prediction. If None, uses the default
        Random Forest model loaded from "models/forest_model.pkl".

    Returns
    -------
    int
        Predicted final score (rounded to the nearest integer).
    """
    prediction_array = []

    # One-hot encoding for Batting Team
    if batting_team == "Chennai Super Kings":
        prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == "Delhi Daredevils":
        prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif batting_team == "Kings XI Punjab":
        prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif batting_team == "Kolkata Knight Riders":
        prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif batting_team == "Mumbai Indians":
        prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif batting_team == "Rajasthan Royals":
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif batting_team == "Royal Challengers Bangalore":
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif batting_team == "Sunrisers Hyderabad":
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]

    # One-hot encoding for Bowling Team
    if bowling_team == "Chennai Super Kings":
        prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == "Delhi Daredevils":
        prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif bowling_team == "Kings XI Punjab":
        prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif bowling_team == "Kolkata Knight Riders":
        prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif bowling_team == "Mumbai Indians":
        prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif bowling_team == "Rajasthan Royals":
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif bowling_team == "Royal Challengers Bangalore":
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif bowling_team == "Sunrisers Hyderabad":
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]

    # Add numerical features
    prediction_array = prediction_array + [
        runs,
        wickets,
        overs,
        runs_last_5,
        wickets_last_5,
    ]

    # Convert to numpy array and predict
    prediction_array = np.array([prediction_array])
    pred = model.predict(prediction_array)

    return int(round(pred[0]))


def load_model(model_path="models/forest_model.pkl"):
    """
    Load a saved model from disk using joblib.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved model file (default: "models/forest_model.pkl").

    Returns
    -------
    sklearn model
        The loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Please run the training pipeline first (main.py) to generate the model."
        )
    model = load(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def run_all_tests(model):
    """
    Run all 6 test cases from the original notebook.

    Executes the 2 historical test cases and 4 live test cases,
    displaying each prediction alongside the actual score.

    Parameters
    ----------
    model : sklearn model
        Trained model to use for predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all test results.
    """
    test_cases = [
        # (test_name, batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5, actual_score)
        (
            "Test 1",
            "Delhi Daredevils",
            "Chennai Super Kings",
            10.2,
            68,
            3,
            29,
            1,
            147,
        ),
        (
            "Test 2",
            "Mumbai Indians",
            "Kings XI Punjab",
            12.3,
            113,
            2,
            55,
            0,
            176,
        ),
        (
            "Live Test 1 (2020)",
            "Kings XI Punjab",
            "Rajasthan Royals",
            14.0,
            118,
            1,
            45,
            0,
            185,
        ),
        (
            "Live Test 2 (2020)",
            "Kolkata Knight Riders",
            "Chennai Super Kings",
            18.0,
            150,
            4,
            57,
            1,
            172,
        ),
        (
            "Live Test 3 (2020)",
            "Delhi Daredevils",
            "Mumbai Indians",
            18.0,
            96,
            8,
            18,
            4,
            110,
        ),
        (
            "Live Test 4 (2020)",
            "Kings XI Punjab",
            "Chennai Super Kings",
            18.0,
            129,
            6,
            34,
            2,
            153,
        ),
    ]

    print("\n" + "=" * 70)
    print("PREDICTION TEST RESULTS")
    print("=" * 70)

    results = []
    for (
        test_name,
        batting_team,
        bowling_team,
        overs,
        runs,
        wickets,
        runs_last_5,
        wickets_last_5,
        actual_score,
    ) in test_cases:
        predicted_score = predict_score(
            batting_team,
            bowling_team,
            overs=overs,
            runs=runs,
            wickets=wickets,
            runs_last_5=runs_last_5,
            wickets_last_5=wickets_last_5,
            model=model,
        )
        results.append(
            {
                "Test": test_name,
                "Batting Team": batting_team,
                "Bowling Team": bowling_team,
                "Overs": overs,
                "Runs": runs,
                "Wickets": wickets,
                "Runs (Last 5)": runs_last_5,
                "Wickets (Last 5)": wickets_last_5,
                "Predicted Score": predicted_score,
                "Actual Score": actual_score,
            }
        )
        print(
            f"\n{test_name}:"
            f"\n  {batting_team} vs {bowling_team}"
            f"\n  Predicted Score : {predicted_score} || Actual Score : {actual_score}"
        )

    results_df = pd.DataFrame(results)
    return results_df


def save_predictions(results_df, save_path="outputs/predictions.csv"):
    """
    Save prediction results to a CSV file.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing test results.
    save_path : str, optional
        Path to save the CSV file (default: "outputs/predictions.csv").

    Returns
    -------
    str
        Path where the CSV was saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"\nPredictions saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    # Load the trained model and run predictions
    try:
        model = load_model("models/forest_model.pkl")
        results = run_all_tests(model)
        save_predictions(results, "outputs/predictions.csv")
        print("\nPredictions completed successfully!")
    except FileNotFoundError as e:
        print(e)
        print("\nPlease run 'python main.py' first to train and save the models.")

