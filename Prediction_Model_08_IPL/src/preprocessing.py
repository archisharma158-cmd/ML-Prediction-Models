"""
Data preprocessing module for the IPL Score Prediction project.

This module handles:
- Dropping irrelevant columns that don't contribute to the prediction.
- Filtering the dataset to only include consistent IPL teams.
- Removing data from the first 5 overs of each match.

These preprocessing steps ensure the model trains on relevant,
high-quality data from the middle and death overs of matches
played by established IPL teams.
"""

import pandas as pd
from src.utils import CONST_TEAMS, IRRELEVANT_COLUMNS


def drop_irrelevant_columns(df):
    """
    Drop columns that are irrelevant for the prediction model.

    Columns dropped: ['mid', 'date', 'venue', 'batsman', 'bowler',
                      'striker', 'non-striker']

    These columns represent match metadata and player-specific information
    that does not provide generalizable predictive value for scoring.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset containing all columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with irrelevant columns removed.
    """
    print(f"Before Removing Irrelevant Columns : {df.shape}")
    df_clean = df.drop(IRRELEVANT_COLUMNS, axis=1)
    print(f"After Removing Irrelevant Columns : {df_clean.shape}")
    return df_clean


def filter_consistent_teams(df):
    """
    Filter the dataset to only include matches between consistent IPL teams.

    Consistent teams are those that have participated across seasons
    without major changes: Kolkata Knight Riders, Chennai Super Kings,
    Rajasthan Royals, Mumbai Indians, Kings XI Punjab,
    Royal Challengers Bangalore, Delhi Daredevils, Sunrisers Hyderabad.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with batting_team and bowling_team columns.

    Returns
    -------
    pd.DataFrame
        DataFrame filtered to only consistent teams.
    """
    print(f"Before Removing Inconsistent Teams : {df.shape}")
    df_filtered = df[
        (df["batting_team"].isin(CONST_TEAMS))
        & (df["bowling_team"].isin(CONST_TEAMS))
    ]
    print(f"After Removing Inconsistent Teams : {df_filtered.shape}")
    print(f"Consistent Teams :\n{df_filtered['batting_team'].unique()}")
    return df_filtered


def remove_first_5_overs(df):
    """
    Remove data from the first 5 overs of each match.

    Only keep deliveries where overs >= 5.0, as the model is
    designed to predict scores based on the middle and death overs
    when a more stable pattern has emerged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an 'overs' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with only deliveries from overs >= 5.0.
    """
    print(f"Before Removing Overs < 5.0 : {df.shape}")
    df_filtered = df[df["overs"] >= 5.0]
    print(f"After Removing Overs < 5.0 : {df_filtered.shape}")
    return df_filtered


if __name__ == "__main__":
    # Quick test when run directly
    from src.data_loader import load_data

    df = load_data("data/players.xlsx")
    df = drop_irrelevant_columns(df)
    df = filter_consistent_teams(df)
    df = remove_first_5_overs(df)
    print("\nPreprocessed Data (first 5 rows):")
    print(df.head())

