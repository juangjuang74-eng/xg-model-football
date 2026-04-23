"""
cv.py
Gap-aware time-based cross-validation — adaptasi langsung dari
metodologi Farm-to-Feed:
  - Pakai match_index sebagai pengganti timestamp
  - Gap antar train dan val untuk mencegah data leakage
  - N folds dengan expanding window (train tumbuh, val geser)
"""
import numpy as np
import pandas as pd
from typing import Iterator


class GapAwareTimeSeriesCV:
    """
    Time-based CV dengan gap, mirip strategi di Farm-to-Feed challenge.

    Params
    ------
    n_splits : jumlah fold
    gap      : jumlah match yang di-skip antara train dan val
               (mencegah leakage dari info pertandingan terbaru)
    val_size : jumlah match per validation window
    """

    def __init__(self, n_splits: int = 3, gap: int = 2, val_size: int = 5):
        self.n_splits = n_splits
        self.gap = gap
        self.val_size = val_size

    def split(self, df: pd.DataFrame) -> Iterator[tuple]:
        """
        Yield (train_idx, val_idx) untuk setiap fold.
        df harus punya kolom 'match_index'.
        """
        match_indices = sorted(df["match_index"].unique())
        n_matches = len(match_indices)

        # Total match yang diperlukan untuk 1 fold: gap + val_size
        total_needed = self.gap + self.val_size
        max_train_end = n_matches - total_needed

        if max_train_end < self.n_splits:
            raise ValueError(
                f"Tidak cukup matches ({n_matches}) untuk {self.n_splits} folds "
                f"dengan gap={self.gap}, val_size={self.val_size}"
            )

        # Titik potong train end untuk setiap fold
        fold_points = np.linspace(
            n_matches // (self.n_splits + 1),
            max_train_end,
            self.n_splits,
            dtype=int
        )

        for fold, train_end_pos in enumerate(fold_points):
            train_matches = match_indices[:train_end_pos]
            val_start_pos = train_end_pos + self.gap
            val_end_pos = min(val_start_pos + self.val_size, n_matches)
            val_matches = match_indices[val_start_pos:val_end_pos]

            if len(val_matches) == 0:
                continue

            train_idx = df[df["match_index"].isin(train_matches)].index.tolist()
            val_idx   = df[df["match_index"].isin(val_matches)].index.tolist()

            yield train_idx, val_idx

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Print ringkasan setiap fold."""
        rows = []
        for fold, (tr, vl) in enumerate(self.split(df)):
            tr_df = df.loc[tr]
            vl_df = df.loc[vl]
            rows.append({
                "Fold": fold + 1,
                "Train matches": tr_df["match_index"].nunique(),
                "Train shots": len(tr),
                "Gap (matches)": self.gap,
                "Val matches": vl_df["match_index"].nunique(),
                "Val shots": len(vl),
                "Val goals": vl_df["goal"].sum(),
            })
        return pd.DataFrame(rows)
