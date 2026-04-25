"""
data_loader.py
Fetch and clean shot event data from StatsBomb Open Data.
Each shot is assigned a chronological match_index used as
the time axis for gap-aware cross-validation.
"""
import pandas as pd
import numpy as np
from statsbombpy import sb


def get_competitions() -> pd.DataFrame:
    """Return all free competitions available from StatsBomb."""
    return sb.competitions()


def load_shots(competition_id: int, season_id: int, verbose: bool = True) -> pd.DataFrame:
    """
    Load all shot events for a given competition and season.

    Popular free competitions:
      La Liga (Messi seasons) : competition_id=11, season_id=27
      FIFA World Cup 2018     : competition_id=43, season_id=3
      Women's Super League    : competition_id=37, season_id=42
      UEFA Euro 2020          : competition_id=55, season_id=43
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches = matches.sort_values("match_date").reset_index(drop=True)
    matches["match_index"] = matches.index  # chronological integer index

    all_shots = []
    for _, row in matches.iterrows():
        mid  = row["match_id"]
        midx = row["match_index"]

        events = sb.events(match_id=mid)
        shots  = events[events["type"] == "Shot"].copy()
        shots["match_id"]    = mid
        shots["match_index"] = midx
        shots["match_date"]  = row["match_date"]
        shots["home_team"]   = row["home_team"]
        shots["away_team"]   = row["away_team"]
        all_shots.append(shots)

    df = pd.concat(all_shots, ignore_index=True)
    if verbose:
        print(f"Loaded {len(df):,} shots from {len(matches)} matches")
    return _parse_shots(df)


def _parse_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Extract nested fields and create binary goal label."""
    df = df.copy()

    df["x"]            = df["location"].apply(lambda l: l[0] if isinstance(l, list) else np.nan)
    df["y"]            = df["location"].apply(lambda l: l[1] if isinstance(l, list) else np.nan)
    df["goal"]         = (df["shot_outcome"] == "Goal").astype(int)
    df["body_part"]    = df["shot_body_part"].fillna("Unknown")
    df["technique"]    = df["shot_technique"].fillna("Unknown")
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["first_time"]   = (
        df.get("shot_first_time", pd.Series(False, index=df.index))
          .fillna(False).astype(int)
    )
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
