"""
tests/test_features.py
Unit tests for feature engineering and the gap-aware CV splitter.

Run with:
    pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import (
    add_geometric_features,
    add_temporal_features,
    build_features,
    GOAL_X,
    GOAL_CENTER_Y,
)
from src.cv import GapAwareTimeSeriesCV


# -- Fixtures ------------------------------------------------------------------

def make_shot_df(n: int = 50, n_matches: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic shot DataFrame for testing."""
    rng = np.random.default_rng(seed)
    match_ids = rng.integers(0, n_matches, size=n)
    return pd.DataFrame({
        "id":             [f"shot_{i}" for i in range(n)],
        "match_id":       match_ids,
        "match_index":    match_ids,
        "match_date":     pd.date_range("2015-08-01", periods=n, freq="6h"),
        "player":         rng.choice(["Messi", "Suarez", "Neymar"], size=n),
        "team":           rng.choice(["Barcelona", "Real Madrid"], size=n),
        "home_team":      "Barcelona",
        "away_team":      "Real Madrid",
        "x":              rng.uniform(80, 120, size=n),
        "y":              rng.uniform(20, 60, size=n),
        "goal":           rng.integers(0, 2, size=n),
        "statsbomb_xg":   rng.uniform(0.01, 0.9, size=n),
        "body_part":      rng.choice(["Foot", "Head"], size=n),
        "technique":      rng.choice(["Normal", "Volley"], size=n),
        "under_pressure": rng.integers(0, 2, size=n),
        "first_time":     rng.integers(0, 2, size=n),
        "minute":         rng.integers(1, 90, size=n),
        "second":         rng.integers(0, 60, size=n),
    })


# -- Geometric feature tests ---------------------------------------------------

class TestGeometricFeatures:

    def test_distance_at_goal_is_zero(self):
        df = make_shot_df()
        # Shot placed exactly at goal centre
        point = pd.DataFrame({**{c: [df[c].iloc[0]] for c in df.columns},
                               "x": [GOAL_X], "y": [GOAL_CENTER_Y]})
        out = add_geometric_features(point)
        assert out["distance"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_distance_is_non_negative(self):
        out = add_geometric_features(make_shot_df())
        assert (out["distance"] >= 0).all()

    def test_angle_within_valid_range(self):
        out = add_geometric_features(make_shot_df())
        assert (out["angle"] >= 0).all()
        assert (out["angle"] <= np.pi).all()

    def test_in_box_is_binary(self):
        out = add_geometric_features(make_shot_df())
        assert set(out["in_box"].unique()).issubset({0, 1})

    def test_required_columns_exist(self):
        out = add_geometric_features(make_shot_df())
        for col in ["distance", "angle", "distance_sq", "in_box", "y_center_offset"]:
            assert col in out.columns, f"Missing column: {col}"


# -- Temporal feature tests ----------------------------------------------------

class TestTemporalFeatures:

    def test_late_game_flag_correct(self):
        out = add_temporal_features(make_shot_df())
        assert (out.loc[out["minute"] >= 75, "is_late_game"] == 1).all()
        assert (out.loc[out["minute"] < 75,  "is_late_game"] == 0).all()

    def test_match_phase_values_valid(self):
        out = add_temporal_features(make_shot_df())
        valid = {"early", "first_half", "early_2nd", "late_2nd", "extra", "nan"}
        assert set(out["match_phase"].unique()).issubset(valid)

    def test_shot_number_starts_at_one(self):
        out = add_temporal_features(make_shot_df())
        assert (out["shot_number_in_match"] >= 1).all()


# -- Build features tests ------------------------------------------------------

class TestBuildFeatures:

    def test_no_object_dtype_in_feature_cols(self):
        df_feat, feature_cols = build_features(make_shot_df())
        for col in feature_cols:
            assert df_feat[col].dtype != object, f"Object dtype in feature: {col}"

    def test_no_nan_in_feature_cols(self):
        df_feat, feature_cols = build_features(make_shot_df())
        assert df_feat[feature_cols].isnull().sum().sum() == 0

    def test_target_excluded_from_features(self):
        _, feature_cols = build_features(make_shot_df())
        assert "goal" not in feature_cols
        assert "statsbomb_xg" not in feature_cols

    def test_minimum_feature_count(self):
        _, feature_cols = build_features(make_shot_df())
        assert len(feature_cols) >= 10, f"Too few features: {len(feature_cols)}"


# -- Gap-aware CV tests --------------------------------------------------------

class TestGapAwareCV:

    def test_correct_number_of_splits(self):
        df = make_shot_df(n=200, n_matches=20)
        cv = GapAwareTimeSeriesCV(n_splits=3, gap=2, val_size=3)
        assert len(list(cv.split(df))) == 3

    def test_no_temporal_leakage(self):
        """Train match indices must always precede validation match indices."""
        df = make_shot_df(n=200, n_matches=20)
        cv = GapAwareTimeSeriesCV(n_splits=3, gap=2, val_size=3)
        for train_idx, val_idx in cv.split(df):
            max_train = df.loc[train_idx, "match_index"].max()
            min_val   = df.loc[val_idx,   "match_index"].min()
            assert max_train < min_val, "Leakage: train overlaps with val in time"

    def test_gap_size_respected(self):
        """The gap between train end and val start must be >= cv.gap."""
        df  = make_shot_df(n=200, n_matches=20)
        gap = 2
        cv  = GapAwareTimeSeriesCV(n_splits=3, gap=gap, val_size=3)
        for train_idx, val_idx in cv.split(df):
            max_train = df.loc[train_idx, "match_index"].max()
            min_val   = df.loc[val_idx,   "match_index"].min()
            assert (min_val - max_train) >= gap

    def test_train_and_val_do_not_overlap(self):
        """Train and validation sets must share no match indices."""
        df = make_shot_df(n=200, n_matches=20)
        cv = GapAwareTimeSeriesCV(n_splits=3, gap=2, val_size=3)
        for train_idx, val_idx in cv.split(df):
            train_matches = set(df.loc[train_idx, "match_index"])
            val_matches   = set(df.loc[val_idx,   "match_index"])
            assert train_matches.isdisjoint(val_matches), "Train and val share match indices"

    def test_summary_returns_dataframe(self):
        df = make_shot_df(n=200, n_matches=20)
        cv = GapAwareTimeSeriesCV(n_splits=3, gap=2, val_size=3)
        summary = cv.summary(df)
        assert isinstance(summary, pd.DataFrame)
        assert "Fold" in summary.columns
