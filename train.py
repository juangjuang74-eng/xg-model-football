"""
train.py -- xG Model Pipeline

Steps:
  1. Load StatsBomb shot data (with chronological match_index)
  2. Feature engineering: geometric + player behaviour + interaction + temporal
  3. Gap-aware time-based CV setup (3 folds, 2-match gap)
  4. Train: LightGBM + CatBoost + Ensemble
  5. Evaluate: LogLoss, AUC, Brier Score per fold
  6. Train final model on full dataset
  7. Generate plots and save model artifacts

Usage:
    python train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd

from src.data_loader import load_shots
from src.features import build_features
from src.cv import GapAwareTimeSeriesCV
from src.model import (
    train_with_gap_cv, print_cv_summary,
    train_final_model,
    plot_cv_results, plot_shap_lgb, save_model,
)

# -- Config --------------------------------------------------------------------
COMPETITION_ID = 11   # La Liga
SEASON_ID      = 27   # 2015/16
CV_N_SPLITS    = 3
CV_GAP         = 2    # number of matches skipped between train and val
CV_VAL_SIZE    = 6    # number of matches per validation window
LGB_WEIGHT     = 0.5  # ensemble weight for LightGBM (CatBoost = 1 - LGB_WEIGHT)
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("  xG Model -- Expected Goals Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading StatsBomb Open Data...")
    df = load_shots(COMPETITION_ID, SEASON_ID)
    print(f"      Goals: {df['goal'].sum()} / {len(df):,} shots "
          f"({df['goal'].mean() * 100:.1f}% conversion rate)")
    print(f"      Matches: {df['match_index'].nunique()}")

    # 2. Feature engineering
    print("\n[2/6] Building features (geometric + behaviour + interaction + temporal)...")
    df_feat, feature_cols = build_features(df)
    print(f"      {len(feature_cols)} features built")

    # 3. Gap-aware CV setup
    print(f"\n[3/6] Gap-Aware Time-Based CV (n_splits={CV_N_SPLITS}, gap={CV_GAP})")
    cv = GapAwareTimeSeriesCV(n_splits=CV_N_SPLITS, gap=CV_GAP, val_size=CV_VAL_SIZE)
    print(cv.summary(df_feat).to_string(index=False))

    # 4. Train with CV
    print(f"\n[4/6] Training LightGBM + CatBoost + Ensemble...")
    cv_output = train_with_gap_cv(df_feat, feature_cols, cv, lgb_weight=LGB_WEIGHT)
    fold_results = cv_output["fold_results"]

    # 5. CV results
    print("\n[5/6] CV Results:")
    cv_summary = print_cv_summary(fold_results)
    fold_results.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)

    print("\n  Model comparison (average across folds):")
    print(f"  {'Model':<14} | {'LogLoss':>8} | {'AUC':>8} | {'Brier':>8}")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for _, r in cv_summary.iterrows():
        print(
            f"  {r['Model']:<14} | "
            f"{r['logloss_mean']:>8.4f} | "
            f"{r['auc_mean']:>8.4f} | "
            f"{r['brier_mean']:>8.4f}"
        )

    # 6. Plot CV results
    plot_cv_results(fold_results)

    # 7. Train final model on full dataset
    print("\n[6/6] Training final ensemble on full dataset...")
    final_model = train_final_model(df_feat, feature_cols, lgb_weight=LGB_WEIGHT)

    # Add predictions to dataframe
    X_all = df_feat[feature_cols]
    df_feat["xg_pred"]     = final_model.predict_proba(X_all)
    df_feat["xg_lgb"]      = final_model.lgb_clf.predict_proba(X_all)[:, 1]
    df_feat["xg_catboost"] = final_model.cb_clf.predict_proba(X_all)[:, 1]

    # SHAP feature importance plot
    sample = df_feat[feature_cols].sample(min(500, len(df_feat)), random_state=42)
    plot_shap_lgb(final_model, sample)

    # Save model and predictions
    save_model(final_model)
    df_feat[[
        "id", "player", "match_index", "goal",
        "statsbomb_xg", "xg_lgb", "xg_catboost", "xg_pred"
    ]].to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    print("\n" + "=" * 60)
    print("  Done! Artifacts saved to outputs/")
    print("  xg_ensemble.pkl   -- final model (LGB + CatBoost)")
    print("  predictions.csv   -- per-shot xG (LGB, CatBoost, Ensemble)")
    print("  fold_results.csv  -- metrics per CV fold")
    print("  cv_results.png    -- AUC / LogLoss per fold plot")
    print("  shap_lgb.png      -- SHAP feature importance")
    print("=" * 60)


if __name__ == "__main__":
    main()
