"""
cv.py
Gap-aware time-based cross-validation for temporal football data.

Matches are treated as an ordered time series. A configurable gap
between training and validation windows prevents leakage from
temporally adjacent matches, simulating realistic deployment conditions.
"""
import numpy as np
import pandas as pd
from typing import Iterator


class GapAwareTimeSeriesCV:
    """
    Time-based cross-validation with an explicit gap between splits.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    gap : int
        Number of matches to skip between the end of training and
        the start of validation (prevents temporal leakage).
    val_size : int
        Number of matches per validation window.

    Example
    -------
    Timeline:
        [TRAIN fold 1] [GAP] [VAL fold 1]
              [TRAIN fold 2]  [GAP] [VAL fold 2]
                    [TRAIN fold 3]   [GAP] [VAL fold 3]
    """

    def __init__(self, n_splits: int = 3, gap: int = 2, val_size: int = 5):
        self.n_splits = n_splits
        self.gap      = gap
        self.val_size = val_size

    def split(self, df: pd.DataFrame) -> Iterator[tuple]:
        """
        Yield (train_indices, val_indices) for each fold.
        Requires a 'match_index' column in df.
        """
        match_indices = sorted(df["match_index"].unique())
        n_matches     = len(match_indices)
        total_needed  = self.gap + self.val_size
        max_train_end = n_matches - total_needed

        if max_train_end < self.n_splits:
            raise ValueError(
                f"Not enough matches ({n_matches}) for {self.n_splits} folds "
                f"with gap={self.gap} and val_size={self.val_size}."
            )

        fold_points = np.linspace(
            n_matches // (self.n_splits + 1),
            max_train_end,
            self.n_splits,
            dtype=int,
        )

        for train_end_pos in fold_points:
            train_matches = match_indices[:train_end_pos]
            val_start_pos = train_end_pos + self.gap
            val_end_pos   = min(val_start_pos + self.val_size, n_matches)
            val_matches   = match_indices[val_start_pos:val_end_pos]

            if not val_matches:
                continue

            train_idx = df[df["match_index"].isin(train_matches)].index.tolist()
            val_idx   = df[df["match_index"].isin(val_matches)].index.tolist()

            yield train_idx, val_idx

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame summarising the size of each fold."""
        rows = []
        for fold, (tr, vl) in enumerate(self.split(df)):
            tr_df = df.loc[tr]
            vl_df = df.loc[vl]
            rows.append({
                "Fold":           fold + 1,
                "Train matches":  tr_df["match_index"].nunique(),
                "Train shots":    len(tr),
                "Gap (matches)":  self.gap,
                "Val matches":    vl_df["match_index"].nunique(),
                "Val shots":      len(vl),
                "Val goals":      int(vl_df["goal"].sum()),
            })
        return pd.DataFrame(rows)
