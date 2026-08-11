"""
Utility constants and helper functions for the IPL Score Prediction project.

This module contains:
- Consistent team names used throughout the project.
- Column names for the one-hot encoded DataFrame.
- Helper functions for manual one-hot encoding of batting and bowling teams.
"""

# List of consistent IPL teams (teams that remained across seasons)
CONST_TEAMS = [
    "Kolkata Knight Riders",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Mumbai Indians",
    "Kings XI Punjab",
    "Royal Challengers Bangalore",
    "Delhi Daredevils",
    "Sunrisers Hyderabad",
]

# Column names for the final one-hot encoded DataFrame
# Batting team one-hot columns (8)
BATTING_TEAM_COLS = [
    "batting_team_Chennai Super Kings",
    "batting_team_Delhi Daredevils",
    "batting_team_Kings XI Punjab",
    "batting_team_Kolkata Knight Riders",
    "batting_team_Mumbai Indians",
    "batting_team_Rajasthan Royals",
    "batting_team_Royal Challengers Bangalore",
    "batting_team_Sunrisers Hyderabad",
]

# Bowling team one-hot columns (8)
BOWLING_TEAM_COLS = [
    "bowling_team_Chennai Super Kings",
    "bowling_team_Delhi Daredevils",
    "bowling_team_Kings XI Punjab",
    "bowling_team_Kolkata Knight Riders",
    "bowling_team_Mumbai Indians",
    "bowling_team_Rajasthan Royals",
    "bowling_team_Royal Challengers Bangalore",
    "bowling_team_Sunrisers Hyderabad",
]

# All column names for the transformed DataFrame
COLUMN_NAMES = (
    BATTING_TEAM_COLS
    + BOWLING_TEAM_COLS
    + ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5", "total"]
)

# Irrelevant columns to drop during data cleaning
IRRELEVANT_COLUMNS = [
    "mid",
    "date",
    "venue",
    "batsman",
    "bowler",
    "striker",
    "non-striker",
]


def get_batting_team_onehot(batting_team):
    """
    Generate one-hot encoded array for a batting team.

    Parameters
    ----------
    batting_team : str
        Name of the batting team (must be one of CONST_TEAMS).

    Returns
    -------
    list
        A list of 8 integers representing the one-hot encoding of the batting team.
    """
    encoding_map = {
        "Chennai Super Kings": [1, 0, 0, 0, 0, 0, 0, 0],
        "Delhi Daredevils": [0, 1, 0, 0, 0, 0, 0, 0],
        "Kings XI Punjab": [0, 0, 1, 0, 0, 0, 0, 0],
        "Kolkata Knight Riders": [0, 0, 0, 1, 0, 0, 0, 0],
        "Mumbai Indians": [0, 0, 0, 0, 1, 0, 0, 0],
        "Rajasthan Royals": [0, 0, 0, 0, 0, 1, 0, 0],
        "Royal Challengers Bangalore": [0, 0, 0, 0, 0, 0, 1, 0],
        "Sunrisers Hyderabad": [0, 0, 0, 0, 0, 0, 0, 1],
    }
    return encoding_map.get(batting_team, [0, 0, 0, 0, 0, 0, 0, 0])


def get_bowling_team_onehot(bowling_team):
    """
    Generate one-hot encoded array for a bowling team.

    Parameters
    ----------
    bowling_team : str
        Name of the bowling team (must be one of CONST_TEAMS).

    Returns
    -------
    list
        A list of 8 integers representing the one-hot encoding of the bowling team.
    """
    encoding_map = {
        "Chennai Super Kings": [1, 0, 0, 0, 0, 0, 0, 0],
        "Delhi Daredevils": [0, 1, 0, 0, 0, 0, 0, 0],
        "Kings XI Punjab": [0, 0, 1, 0, 0, 0, 0, 0],
        "Kolkata Knight Riders": [0, 0, 0, 1, 0, 0, 0, 0],
        "Mumbai Indians": [0, 0, 0, 0, 1, 0, 0, 0],
        "Rajasthan Royals": [0, 0, 0, 0, 0, 1, 0, 0],
        "Royal Challengers Bangalore": [0, 0, 0, 0, 0, 0, 1, 0],
        "Sunrisers Hyderabad": [0, 0, 0, 0, 0, 0, 0, 1],
    }
    return encoding_map.get(bowling_team, [0, 0, 0, 0, 0, 0, 0, 0])

