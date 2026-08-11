"""
Feature engineering module for the IPL Score Prediction project.

This module handles:
- Label encoding of team names (batting_team, bowling_team).
- One-hot encoding using sklearn's ColumnTransformer.
- Converting the transformed numpy array back into a named DataFrame.

The final feature set includes 21 features:
- 8 one-hot encoded batting team columns
- 8 one-hot encoded bowling team columns
- 5 numerical features: runs, wickets, overs, runs_last_5, wickets_last_5
And the target variable: total
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.utils import COLUMN_NAMES


def label_encode_teams(df):
    """
    Apply Label Encoding to the batting_team and bowling_team columns.

    Converts team names into integer labels for subsequent one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'batting_team' and 'bowling_team' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with team columns label-encoded.
    """
    le = LabelEncoder()
    for col in ["batting_team", "bowling_team"]:
        df[col] = le.fit_transform(df[col])
    return df


def one_hot_encode_features(df):
    """
    Apply One-Hot Encoding to batting_team and bowling_team columns.

    Uses sklearn's ColumnTransformer to one-hot encode columns at indices
    0 (batting_team) and 1 (bowling_team), while passing through all other
    columns unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with label-encoded team columns.

    Returns
    -------
    np.ndarray
        NumPy array of the transformed dataset.
    """
    column_transformer = ColumnTransformer(
        [("encoder", OneHotEncoder(), [0, 1])], remainder="passthrough"
    )
    data_transformed = np.array(column_transformer.fit_transform(df))
    return data_transformed


def create_feature_dataframe(data_array):
    """
    Convert the transformed numpy array into a named pandas DataFrame.

    Parameters
    ----------
    data_array : np.ndarray
        NumPy array from one-hot encoding step.

    Returns
    -------
    pd.DataFrame
        DataFrame with proper column names for all features and target.
    """
    df = pd.DataFrame(data_array, columns=COLUMN_NAMES)
    return df


def build_features(df):
    """
    Complete feature engineering pipeline.

    Executes the full feature engineering process:
    1. Label encode team names
    2. One-hot encode team columns
    3. Create a properly named DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with columns:
        batting_team, bowling_team, runs, wickets, overs,
        runs_last_5, wickets_last_5, total

    Returns
    -------
    pd.DataFrame
        Final feature DataFrame with 22 columns (21 features + 1 target).
    """
    print("Starting feature engineering pipeline...")

    # Step 1: Label Encode
    df_encoded = label_encode_teams(df)
    print("Label encoding complete.")

    # Step 2: One-Hot Encode
    data_transformed = one_hot_encode_features(df_encoded)
    print("One-hot encoding complete.")

    # Step 3: Create DataFrame
    df_final = create_feature_dataframe(data_transformed)
    print(f"Feature DataFrame created with shape: {df_final.shape}")

    return df_final


if __name__ == "__main__":
    # Quick test when run directly
    from src.data_loader import load_data
    from src.preprocessing import (
        drop_irrelevant_columns,
        filter_consistent_teams,
        remove_first_5_overs,
    )

    df = load_data("data/players.xlsx")
    df = drop_irrelevant_columns(df)
    df = filter_consistent_teams(df)
    df = remove_first_5_overs(df)
    df = build_features(df)
    print("\nEncoded Data (first 5 rows):")
    print(df.head())

