"""
Model evaluation module for the IPL Score Prediction project.

This module handles:
- Computing R-squared scores for training and testing sets.
- Computing MAE, MSE, and RMSE evaluation metrics.
- Generating and saving a correlation heatmap.
- Generating and saving a model comparison barplot.
- Saving evaluation metrics to a text file.
- Selecting the best performing model (Random Forest).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse


def evaluate_model(model, train_features, train_labels, test_features, test_labels):
    """
    Evaluate a trained model on both training and testing sets.

    Computes R² score, MAE, MSE, and RMSE for the test set.

    Parameters
    ----------
    model : sklearn model
        Trained machine learning model.
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.
    test_features : pd.DataFrame
        Testing feature set.
    test_labels : pd.Series
        Testing labels.

    Returns
    -------
    dict
        Dictionary containing:
        - "train_score": R² score on training set (as percentage string)
        - "test_score": R² score on testing set (as percentage string)
        - "mae": Mean Absolute Error
        - "mse": Mean Squared Error
        - "rmse": Root Mean Squared Error
        - "test_score_float": R² score on testing set (float)
    """
    train_score = model.score(train_features, train_labels) * 100
    test_score = model.score(test_features, test_labels) * 100

    test_predictions = model.predict(test_features)
    mae_value = mae(test_labels, test_predictions)
    mse_value = mse(test_labels, test_predictions)
    rmse_value = np.sqrt(mse_value)

    return {
        "train_score": f"{train_score:.2f}",
        "test_score": f"{test_score:.2f}",
        "mae": mae_value,
        "mse": mse_value,
        "rmse": rmse_value,
        "test_score_float": test_score,
    }


def evaluate_all_models(
    models, train_features, train_labels, test_features, test_labels
):
    """
    Evaluate all trained models and track their performance.

    Parameters
    ----------
    models : dict
        Dictionary of trained models.
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.
    test_features : pd.DataFrame
        Testing feature set.
    test_labels : pd.Series
        Testing labels.

    Returns
    -------
    tuple
        (results_dict, models_tracker)
        - results_dict: dict mapping model names to their evaluation results.
        - models_tracker: dict mapping model names to their test R² scores (as strings).
    """
    results = {}
    models_tracker = {}

    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    for model_name, model in models.items():
        print(f"\n----- {model_name.upper()} -----")
        result = evaluate_model(
            model, train_features, train_labels, test_features, test_labels
        )
        results[model_name] = result

        print(f"Train Score : {result['train_score']}%")
        print(f"Test Score : {result['test_score']}%")
        print(f"Mean Absolute Error (MAE): {result['mae']}")
        print(f"Mean Squared Error (MSE): {result['mse']}")
        print(f"Root Mean Squared Error (RMSE): {result['rmse']}")

        models_tracker[model_name] = result["test_score"]

    return results, models_tracker


def plot_correlation_matrix(df, save_path="outputs/plots/correlation_matrix.png"):
    """
    Generate and save a correlation heatmap of the dataset features.

    Parameters
    ----------
    df : pd.DataFrame
        The feature-engineered DataFrame.
    save_path : str, optional
        Path to save the plot image (default: "outputs/plots/correlation_matrix.png").

    Returns
    -------
    str
        Path where the plot was saved.
    """
    from seaborn import heatmap

    plt.figure(figsize=(12, 10))
    heatmap(data=df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix of IPL Dataset Features", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Correlation matrix saved to: {save_path}")

    return save_path


def plot_model_comparison(models_tracker, save_path="outputs/plots/model_comparison.png"):
    """
    Generate and save a barplot comparing model performance (R² scores).

    Parameters
    ----------
    models_tracker : dict
        Dictionary mapping model names to their test R² scores (as strings).
    save_path : str, optional
        Path to save the plot image (default: "outputs/plots/model_comparison.png").

    Returns
    -------
    str
        Path where the plot was saved.
    """
    from seaborn import barplot

    model_names = list(models_tracker.keys())
    accuracy = list(map(float, models_tracker.values()))

    plt.figure(figsize=(10, 6))
    barplot(x=model_names, y=accuracy)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Test R² Score (%)", fontsize=12)
    plt.title("Model Performance Comparison", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    # Add value labels on bars
    for i, v in enumerate(accuracy):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=10)

    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison plot saved to: {save_path}")

    return save_path


def save_metrics(results, save_path="outputs/metrics.txt"):
    """
    Save all evaluation metrics to a text file.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their evaluation result dicts.
    save_path : str, optional
        Path to save the metrics file (default: "outputs/metrics.txt").

    Returns
    -------
    str
        Path where the metrics were saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("IPL FIRST INNINGS SCORE PREDICTION - MODEL EVALUATION METRICS\n")
        f.write("=" * 70 + "\n\n")

        for model_name, result in results.items():
            f.write(f"Model: {model_name.upper()}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Train R² Score : {result['train_score']}%\n")
            f.write(f"Test R² Score  : {result['test_score']}%\n")
            f.write(f"MAE            : {result['mae']:.4f}\n")
            f.write(f"MSE            : {result['mse']:.4f}\n")
            f.write(f"RMSE           : {result['rmse']:.4f}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("BEST MODEL: Random Forest Regressor\n")
        f.write("=" * 70 + "\n")

    print(f"Metrics saved to: {save_path}")
    return save_path


def select_best_model(models):
    """
    Select the best model based on the original notebook's analysis.

    The notebook concluded that Random Forest performed best,
    closely followed by Decision Tree and Neural Networks.

    Parameters
    ----------
    models : dict
        Dictionary of all trained models.

    Returns
    -------
    sklearn model
        The Random Forest model (key: "forest").
    """
    print("\nBest Model Selected: Random Forest Regressor")
    return models["forest"]


if __name__ == "__main__":
    # Quick test when run directly
    from src.data_loader import load_data
    from src.preprocessing import (
        drop_irrelevant_columns,
        filter_consistent_teams,
        remove_first_5_overs,
    )
    from src.feature_engineering import build_features
    from src.train import prepare_train_test_split, train_all_models

    df = load_data("data/players.xlsx")
    df = drop_irrelevant_columns(df)
    df = filter_consistent_teams(df)
    df = remove_first_5_overs(df)
    df = build_features(df)
    train_features, test_features, train_labels, test_labels = prepare_train_test_split(
        df
    )
    models = train_all_models(train_features, train_labels)
    results, models_tracker = evaluate_all_models(
        models, train_features, train_labels, test_features, test_labels
    )
    plot_correlation_matrix(df)
    plot_model_comparison(models_tracker)
    save_metrics(results)

