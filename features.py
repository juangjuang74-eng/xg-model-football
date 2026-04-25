"""
features.py
Feature engineering for the xG model covering four groups:
  1. Geometric   -- shot location relative to goal
  2. Player behaviour -- historical shooting patterns per player
  3. Player x ShotType interaction -- player accuracy per body part / zone
  4. Temporal    -- match phase and timing signals

All player behaviour and interaction features are built with
expanding windows, using only data prior to the current shot,
ensuring no future leakage into training.
"""
import numpy as np
import pandas as pd

# StatsBomb pitch dimensions (in metres)
GOAL_X        = 120.0
GOAL_Y_LEFT   = 36.0
GOAL_Y_RIGHT  = 44.0
GOAL_CENTER_Y = (GOAL_Y_LEFT + GOAL_Y_RIGHT) / 2.0


# -- 1. Geometric features -----------------------------------------------------

def _distance(x: pd.Series, y: pd.Series) -> pd.Series:
    """Euclidean distance from shot location to goal centre."""
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_CENTER_Y - y) ** 2)


def _angle(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    Visible angle of the goal from the shot location (radians).
    Computed via the cross-product of vectors to each post.
    """
    a1, b1 = GOAL_X - x, GOAL_Y_LEFT  - y
    a2, b2 = GOAL_X - x, GOAL_Y_RIGHT - y
    dot   = a1 * a2 + b1 * b2
    cross = a1 * b2 - b1 * a2
    return np.arctan2(np.abs(cross), dot)


def add_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance"]         = _distance(df["x"], df["y"])
    df["angle"]            = _angle(df["x"], df["y"])
    df["distance_sq"]      = df["distance"] ** 2
    df["angle_x_distance"] = df["angle"] * df["distance"]
    df["in_box"]           = (
        (df["x"] >= 102) & (df["x"] <= 120) &
        (df["y"] >= 18)  & (df["y"] <= 62)
    ).astype(int)
    df["y_center_offset"]  = (df["y"] - GOAL_CENTER_Y).abs()
    return df


# -- 2. Player behaviour features ----------------------------------------------

def add_player_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding-window statistics per player, computed in chronological order.
    Only information available before the current match is used.
    """
    df = df.sort_values(["player", "match_index", "minute"]).copy()

    # Cumulative shot count per player
    df["player_shot_count"] = df.groupby("player").cumcount()

    # Shot frequency over the last 5 matches
    df["player_shots_in_last_5"] = (
        df.groupby("player")["match_index"]
          .transform(lambda s: s.expanding().apply(
              lambda x: (x >= x.iloc[-1] - 5).sum() - 1, raw=False
          ))
    )

    # Historical average xG per shot
    if "statsbomb_xg" in df.columns:
        df["player_avg_xg_hist"] = (
            df.groupby("player")["statsbomb_xg"]
              .transform(lambda s: s.expanding().mean().shift(1))
              .fillna(df["statsbomb_xg"].mean())
        )
        df["player_total_xg_hist"] = (
            df.groupby("player")["statsbomb_xg"]
              .transform(lambda s: s.expanding().sum().shift(1))
              .fillna(0)
        )

    # Historical goals and conversion rate
    df["player_goals_hist"] = (
        df.groupby("player")["goal"]
          .transform(lambda s: s.expanding().sum().shift(1))
          .fillna(0)
    )
    df["player_conversion_hist"] = (
        df["player_goals_hist"] / (df["player_shot_count"] + 1)
    )

    return df


# -- 3. Player x ShotType interaction features ---------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture per-player accuracy broken down by body part and shot zone.
    All statistics are expanding to avoid leakage.
    """
    df = df.copy()

    # Shot count per player x body_part
    df["player_bp_count"] = df.groupby(["player", "body_part"]).cumcount()

    # Goal count per player x body_part (expanding, shifted to avoid leakage)
    df["player_bp_goals"] = (
        df.groupby(["player", "body_part"])["goal"]
          .transform(lambda s: s.expanding().sum().shift(1))
          .fillna(0)
    )
    df["player_bp_rate"] = df["player_bp_goals"] / (df["player_bp_count"] + 1)

    # Shot count per player x in_box zone
    df["player_inbox_count"] = df.groupby(["player", "in_box"]).cumcount()

    return df


# -- 4. Temporal features ------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based signals capturing match phase and momentum."""
    df = df.copy()

    df["match_phase"] = pd.cut(
        df["minute"],
        bins=[-1, 15, 45, 60, 75, 120],
        labels=["early", "first_half", "early_2nd", "late_2nd", "extra"]
    ).astype(str)

    df["is_late_game"] = (df["minute"] >= 75).astype(int)

    # Shot number within the current match (proxy for momentum)
    df["shot_number_in_match"] = df.groupby("match_id").cumcount() + 1

    return df


# -- Master builder ------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple:
    """
    Apply all four feature groups and return (enriched_df, feature_col_list).
    """
    df = add_geometric_features(df)
    df = add_player_behaviour_features(df)
    df = add_interaction_features(df)
    df = add_temporal_features(df)

    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=["body_part", "technique", "match_phase"],
        prefix=["bp", "tech", "phase"],
        drop_first=False,
    )

    # Convert boolean columns to int
    for col in df.select_dtypes(include=bool).columns:
        df[col] = df[col].astype(int)

    # Collect numeric feature columns (exclude identifiers and target)
    exclude = {
        "id", "match_id", "match_index", "match_date",
        "player", "team", "home_team", "away_team",
        "goal", "statsbomb_xg",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype != object
    ]

    return df, feature_cols
