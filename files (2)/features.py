"""
features.py
Feature engineering terinspirasi dari Farm-to-Feed methodology:
  1. Geometris (distance, angle, in_box)
  2. Player behaviour (shot frequency, recency, avg xG)
  3. Player–ShotType interaction (player × body_part history)
  4. Temporal (minute, match phase)
  5. Kategorikal one-hot
"""
import numpy as np
import pandas as pd

GOAL_X = 120.0
GOAL_Y_LEFT = 36.0
GOAL_Y_RIGHT = 44.0
GOAL_CENTER_Y = (GOAL_Y_LEFT + GOAL_Y_RIGHT) / 2.0


# ── 1. Geometris ─────────────────────────────────────────────────────────────

def _distance(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_CENTER_Y - y) ** 2)


def _angle(x, y):
    a1 = GOAL_X - x;  b1 = GOAL_Y_LEFT - y
    a2 = GOAL_X - x;  b2 = GOAL_Y_RIGHT - y
    dot   = a1 * a2 + b1 * b2
    cross = a1 * b2 - b1 * a2
    return np.arctan2(np.abs(cross), dot)


def add_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance"]        = _distance(df["x"], df["y"])
    df["angle"]           = _angle(df["x"], df["y"])
    df["distance_sq"]     = df["distance"] ** 2
    df["angle_x_distance"]= df["angle"] * df["distance"]
    df["in_box"]          = (
        (df["x"] >= 102) & (df["x"] <= 120) &
        (df["y"] >= 18)  & (df["y"] <= 62)
    ).astype(int)
    df["y_center_offset"] = (df["y"] - GOAL_CENTER_Y).abs()
    return df


# ── 2. Player Behaviour Features (analog: Customer Behaviour) ────────────────

def add_player_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dihitung secara expanding window — hanya memakai data sebelum
    match_index saat ini (mencegah data leakage).
    """
    df = df.sort_values(["player", "match_index", "minute"]).copy()

    # Shot frequency per player (jumlah shot kumulatif sebelum match ini)
    df["player_shot_count"] = (
        df.groupby("player").cumcount()  # shot ke-N untuk player ini
    )

    # Recency: berapa match yang lalu player ini terakhir shot
    df["player_shots_in_last_5_matches"] = (
        df.groupby("player")["match_index"]
          .transform(lambda s: s.expanding().apply(
              lambda x: (x >= x.iloc[-1] - 5).sum() - 1, raw=False
          ))
    )

    # Rata-rata xG historis player (pakai statsbomb_xg sebagai proxy)
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

    # Conversion rate historis
    df["player_goals_hist"] = (
        df.groupby("player")["goal"]
          .transform(lambda s: s.expanding().sum().shift(1))
          .fillna(0)
    )
    df["player_conversion_hist"] = (
        df["player_goals_hist"] / (df["player_shot_count"] + 1)
    )

    return df


# ── 3. Player–ShotType Interaction Features (analog: Customer-Product) ───────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berapa kali player X melakukan shot dengan body_part Y?
    Seberapa akurat player X dengan teknik Z?
    """
    df = df.copy()

    # Shot count per player × body_part
    df["player_bp_count"] = (
        df.groupby(["player", "body_part"]).cumcount()
    )

    # Goal rate per player × body_part
    df["player_bp_goals"] = (
        df.groupby(["player", "body_part"])["goal"]
          .transform(lambda s: s.expanding().sum().shift(1))
          .fillna(0)
    )
    df["player_bp_rate"] = df["player_bp_goals"] / (df["player_bp_count"] + 1)

    # Shot count per player × in_box
    df["player_inbox_count"] = (
        df.groupby(["player", "in_box"]).cumcount()
    )

    return df


# ── 4. Temporal Features (analog: Temporal Purchasing Patterns) ──────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Match phase
    df["match_phase"] = pd.cut(
        df["minute"],
        bins=[-1, 15, 45, 60, 75, 120],
        labels=["early", "first_half", "early_2nd", "late_2nd", "extra"]
    ).astype(str)

    # Is late in the game? (desperate shooting)
    df["is_late_game"] = (df["minute"] >= 75).astype(int)

    # Shot number in this match (momentum proxy)
    df["shot_number_in_match"] = df.groupby("match_id").cumcount() + 1

    return df


# ── Master builder ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple:
    """
    Jalankan semua feature groups. Return (df_enriched, feature_cols).
    """
    df = add_geometric_features(df)
    df = add_player_behaviour_features(df)
    df = add_interaction_features(df)
    df = add_temporal_features(df)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["body_part", "technique", "match_phase"],
                        prefix=["bp", "tech", "phase"], drop_first=False)

    # Bool → int
    for c in df.select_dtypes(include=bool).columns:
        df[c] = df[c].astype(int)

    # Pilih feature columns (numerik, bukan ID/target/meta)
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
