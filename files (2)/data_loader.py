"""
data_loader.py
Mengambil shot data dari StatsBomb + menyimpan match_index
untuk gap-aware time-based cross-validation.
"""
import pandas as pd
import numpy as np
from statsbombpy import sb


GOAL_X = 120.0
GOAL_Y_LEFT = 36.0
GOAL_Y_RIGHT = 44.0
GOAL_CENTER_Y = (GOAL_Y_LEFT + GOAL_Y_RIGHT) / 2.0


def load_shots(competition_id: int, season_id: int, verbose: bool = True) -> pd.DataFrame:
    """
    Load semua shot events. Setiap shot diberi match_index (urutan kronologis)
    untuk digunakan sebagai pengganti timestamp di time-based CV.
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    # Urutkan match berdasarkan tanggal supaya match_index kronologis
    matches = matches.sort_values("match_date").reset_index(drop=True)
    matches["match_index"] = matches.index  # 0, 1, 2, ... N_matches-1

    all_shots = []
    for _, row in matches.iterrows():
        mid = row["match_id"]
        midx = row["match_index"]
        mdate = row["match_date"]
        home = row["home_team"]
        away = row["away_team"]

        events = sb.events(match_id=mid)
        shots = events[events["type"] == "Shot"].copy()
        shots["match_id"] = mid
        shots["match_index"] = midx
        shots["match_date"] = mdate
        shots["home_team"] = home
        shots["away_team"] = away
        all_shots.append(shots)

    df = pd.concat(all_shots, ignore_index=True)
    if verbose:
        print(f"Loaded {len(df):,} shots from {len(matches)} matches")
    return _parse_shots(df)


def _parse_shots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["x"] = df["location"].apply(lambda l: l[0] if isinstance(l, list) else np.nan)
    df["y"] = df["location"].apply(lambda l: l[1] if isinstance(l, list) else np.nan)
    df["goal"] = (df["shot_outcome"] == "Goal").astype(int)
    df["body_part"] = df["shot_body_part"].fillna("Unknown")
    df["technique"] = df["shot_technique"].fillna("Unknown")
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["first_time"] = df.get("shot_first_time", pd.Series(False, index=df.index)).fillna(False).astype(int)
    df["statsbomb_xg"] = df.get("shot_statsbomb_xg", np.nan)

    df = df.dropna(subset=["x", "y"])

    keep = [
        "id", "match_id", "match_index", "match_date",
        "player", "team", "home_team", "away_team",
        "x", "y", "goal", "statsbomb_xg",
        "body_part", "technique", "under_pressure", "first_time",
        "minute", "second",
    ]
    available = [c for c in keep if c in df.columns]
    return df[available].reset_index(drop=True)
