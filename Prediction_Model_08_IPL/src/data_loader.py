"""
Data loading module for the IPL Score Prediction project.

This module handles:
- Loading the IPL ball-by-ball dataset from an Excel file.
- Displaying basic information about the dataset (shape, head, describe, info, unique values, dtypes).

The dataset should contain the following columns:
mid, date, venue, batting_team, bowling_team, batsman, bowler, runs, wickets,
overs, runs_last_5, wickets_last_5, striker, non-striker, total
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load the IPL ball-by-ball dataset from an Excel file.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the Excel file (e.g., "data/players.xlsx").

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    df = pd.read_excel(filepath, engine="openpyxl")
    return df


def display_dataset_info(df):
    """
    Display basic information about the dataset.

    Prints:
        - Shape of the dataset
        - First 5 rows (head)
        - Statistical description of numerical columns
        - Data types and non-null counts (info)
        - Number of unique values per column
        - Data types of each column

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to display information for.
    """
    print(f"Dataset Shape: {df.shape}")
    print("\n" + "=" * 70)
    print("First 5 rows of the dataset:")
    print("=" * 70)
    print(df.head())
    print("\n" + "=" * 70)
    print("Statistical Description of Numerical Columns:")
    print("=" * 70)
    print(df.describe())
    print("\n" + "=" * 70)
    print("Dataset Info (Data Types & Non-Null Counts):")
    print("=" * 70)
    # Capture info output
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    print(buf.getvalue())
    print("\n" + "=" * 70)
    print("Number of Unique Values per Column:")
    print("=" * 70)
    print(df.nunique())
    print("\n" + "=" * 70)
    print("Data Types of All Columns:")
    print("=" * 70)
    print(df.dtypes)


if __name__ == "__main__":
    # Quick test when run directly
    df = load_data("data/players.xlsx")
    display_dataset_info(df)

