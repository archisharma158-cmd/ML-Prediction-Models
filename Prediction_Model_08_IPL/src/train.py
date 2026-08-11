"""
Model training module for the IPL Score Prediction project.

This module handles:
- Splitting the data into training and testing sets (80:20 split).
- Training 6 different regression models:
  1. Decision Tree Regressor
  2. Linear Regression
  3. Random Forest Regressor
  4. Lasso Regression (LassoCV with automatic alpha selection)
  5. Support Vector Regression (SVR with RBF kernel)
  6. Neural Network Regression (MLPRegressor with logistic activation)
- Returning a dictionary of trained models for evaluation and comparison.

All hyperparameters are preserved exactly as in the original notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def prepare_train_test_split(df, target_col="total", test_size=0.20, shuffle=True):
    """
    Split the feature-engineered DataFrame into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The feature DataFrame (21 features + 1 target column 'total').
    target_col : str, optional
        Name of the target column (default: "total").
    test_size : float, optional
        Proportion of data to use for testing (default: 0.20).
    shuffle : bool, optional
        Whether to shuffle the data before splitting (default: True).

    Returns
    -------
    tuple
        (train_features, test_features, train_labels, test_labels)
    """
    features = df.drop([target_col], axis=1)
    labels = df[target_col]

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, shuffle=shuffle
    )

    print(f"Training Set : {train_features.shape}")
    print(f"Testing Set : {test_features.shape}")

    return train_features, test_features, train_labels, test_labels


def train_decision_tree(train_features, train_labels):
    """
    Train a Decision Tree Regressor.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.

    Returns
    -------
    DecisionTreeRegressor
        Trained Decision Tree model.
    """
    print("\n----- Training Decision Tree Regressor -----")
    tree = DecisionTreeRegressor()
    tree.fit(train_features, train_labels)
    print("Decision Tree training complete.")
    return tree


def train_linear_regression(train_features, train_labels):
    """
    Train a Linear Regression model.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.

    Returns
    -------
    LinearRegression
        Trained Linear Regression model.
    """
    print("\n----- Training Linear Regression -----")
    linreg = LinearRegression()
    linreg.fit(train_features, train_labels)
    print("Linear Regression training complete.")
    return linreg


def train_random_forest(train_features, train_labels):
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.

    Returns
    -------
    RandomForestRegressor
        Trained Random Forest model.
    """
    print("\n----- Training Random Forest Regressor -----")
    forest = RandomForestRegressor()
    forest.fit(train_features, train_labels)
    print("Random Forest training complete.")
    return forest


def train_lasso_regression(train_features, train_labels):
    """
    Train a Lasso Regression model with automatic alpha selection (LassoCV).

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.

    Returns
    -------
    LassoCV
        Trained Lasso regression model.
    """
    print("\n----- Training Lasso Regression -----")
    lasso = LassoCV()
    lasso.fit(train_features, train_labels)
    print("Lasso Regression training complete.")
    return lasso


def train_svr(train_features, train_labels):
    """
    Train a Support Vector Regression model with RBF kernel.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.

    Returns
    -------
    SVR
        Trained SVR model.
    """
    print("\n----- Training Support Vector Regression -----")
    svm = SVR()
    svm.fit(train_features, train_labels)
    print("SVR training complete.")
    return svm


def train_neural_network(train_features, train_labels):
    """
    Train a Neural Network (MLP Regressor) with logistic activation.

    Uses 500 max iterations as in the original notebook.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.

    Returns
    -------
    MLPRegressor
        Trained Neural Network model.
    """
    print("\n----- Training Neural Network (MLP Regressor) -----")
    neural_net = MLPRegressor(activation="logistic", max_iter=500)
    neural_net.fit(train_features, train_labels)
    print("Neural Network training complete.")
    return neural_net


def train_all_models(train_features, train_labels):
    """
    Train all 6 regression models and return them in a dictionary.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature set.
    train_labels : pd.Series
        Training labels.

    Returns
    -------
    dict
        Dictionary containing all trained models with keys:
        "tree", "linreg", "forest", "lasso", "svm", "neural_net".
    """
    models = {}

    print("=" * 50)
    print("Starting model training pipeline...")
    print("=" * 50)

    # 1. Decision Tree
    models["tree"] = train_decision_tree(train_features, train_labels)

    # 2. Linear Regression
    models["linreg"] = train_linear_regression(train_features, train_labels)

    # 3. Random Forest
    models["forest"] = train_random_forest(train_features, train_labels)

    # 4. Lasso Regression
    models["lasso"] = train_lasso_regression(train_features, train_labels)

    # 5. Support Vector Regression
    models["svm"] = train_svr(train_features, train_labels)

    # 6. Neural Network
    models["neural_net"] = train_neural_network(train_features, train_labels)

    print("\n" + "=" * 50)
    print("All models trained successfully!")
    print("=" * 50)

    return models


if __name__ == "__main__":
    # Quick test when run directly
    from src.data_loader import load_data
    from src.preprocessing import (
        drop_irrelevant_columns,
        filter_consistent_teams,
        remove_first_5_overs,
    )
    from src.feature_engineering import build_features

    df = load_data("data/players.xlsx")
    df = drop_irrelevant_columns(df)
    df = filter_consistent_teams(df)
    df = remove_first_5_overs(df)
    df = build_features(df)
    train_features, test_features, train_labels, test_labels = prepare_train_test_split(
        df
    )
    models = train_all_models(train_features, train_labels)

