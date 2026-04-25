"""
model.py
Stacked ensemble pipeline for xG modelling:
  - LightGBM classifier  -- goal probability
  - CatBoost classifier  -- goal probability
  - Weighted ensemble    -- averaged xG prediction
  - Separate regressor   -- xG value (optional continuous target)

Evaluation metrics: LogLoss, AUC, Brier Score (classification)
                    MAE (regression)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import (
    log_loss, roc_auc_score, mean_absolute_error, brier_score_loss,
)
import matplotlib.pyplot as plt
import shap
import pickle
from pathlib import Path


# -- Model hyperparameters -----------------------------------------------------

LGB_CLF_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary",
    metric="binary_logloss",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

LGB_REG_PARAMS = dict(
    **{k: v for k, v in LGB_CLF_PARAMS.items() if k not in ("objective", "metric")},
    objective="regression",
    metric="mae",
)

CB_CLF_PARAMS = dict(
    iterations=600,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=0,
)

CB_REG_PARAMS = dict(
    **{k: v for k, v in CB_CLF_PARAMS.items() if k not in ("loss_function", "eval_metric")},
    loss_function="MAE",
    eval_metric="MAE",
)


# -- Ensemble class ------------------------------------------------------------

class XGEnsemble:
    """
    LightGBM + CatBoost ensemble with tuneable blending weight.

    Two separate model pairs:
      - Classifier pair  -> predict_proba()   (goal probability)
      - Regressor pair   -> predict_quantity() (xG value)
    """

    def __init__(self, lgb_weight: float = 0.5):
        self.lgb_weight = lgb_weight
        self.cb_weight  = 1.0 - lgb_weight

        self.lgb_clf = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
        self.cb_clf  = cb.CatBoostClassifier(**CB_CLF_PARAMS)
        self.lgb_reg = lgb.LGBMRegressor(**LGB_REG_PARAMS)
        self.cb_reg  = cb.CatBoostRegressor(**CB_REG_PARAMS)

        self.feature_cols: list = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGEnsemble":
        self.lgb_clf.fit(X_train, y_train)
        self.cb_clf.fit(X_train, y_train)
        self.lgb_reg.fit(X_train, y_train)
        self.cb_reg.fit(X_train, y_train)
        self.feature_cols = list(X_train.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return blended goal probability."""
        lgb_p = self.lgb_clf.predict_proba(X)[:, 1]
        cb_p  = self.cb_clf.predict_proba(X)[:, 1]
        return self.lgb_weight * lgb_p + self.cb_weight * cb_p

    def predict_quantity(self, X: pd.DataFrame) -> np.ndarray:
        """Return blended continuous xG estimate."""
        lgb_q = self.lgb_reg.predict(X)
        cb_q  = self.cb_reg.predict(X)
        return self.lgb_weight * lgb_q + self.cb_weight * cb_q


# -- Gap-aware CV training loop ------------------------------------------------

def train_with_gap_cv(
    df: pd.DataFrame,
    feature_cols: list,
    cv,
    lgb_weight: float = 0.5,
) -> dict:
    """
    Train and evaluate LightGBM, CatBoost, and Ensemble on each CV fold.
    Returns a dict with fold metrics and fitted model objects.
    """
    results        = []
    fitted_models  = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
        X_tr = df.loc[train_idx, feature_cols]
        y_tr = df.loc[train_idx, "goal"]
        X_vl = df.loc[val_idx,   feature_cols]
        y_vl = df.loc[val_idx,   "goal"]

        model = XGEnsemble(lgb_weight=lgb_weight)
        model.fit(X_tr, y_tr)

        lgb_prob = model.lgb_clf.predict_proba(X_vl)[:, 1]
        cb_prob  = model.cb_clf.predict_proba(X_vl)[:, 1]
        ens_prob = model.predict_proba(X_vl)

        row = {
            "fold":       fold + 1,
            "train_shots": len(train_idx),
            "val_shots":   len(val_idx),
            "val_goals":   int(y_vl.sum()),
        }

        for name, probs in [("LightGBM", lgb_prob), ("CatBoost", cb_prob), ("Ensemble", ens_prob)]:
            row[f"{name}_logloss"] = log_loss(y_vl, probs)
            row[f"{name}_auc"]     = roc_auc_score(y_vl, probs)
            row[f"{name}_brier"]   = brier_score_loss(y_vl, probs)

        results.append(row)
        fitted_models.append(model)

        print(
            f"  Fold {fold + 1} | "
            f"LGB AUC={row['LightGBM_auc']:.4f} | "
            f"CB AUC={row['CatBoost_auc']:.4f} | "
            f"Ensemble AUC={row['Ensemble_auc']:.4f}"
        )

    return {"fold_results": pd.DataFrame(results), "models": fitted_models}


def print_cv_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Print a summary table (mean +/- std) across all folds."""
    metrics     = ["logloss", "auc", "brier"]
    model_names = ["LightGBM", "CatBoost", "Ensemble"]
    rows = []

    for mname in model_names:
        row = {"Model": mname}
        for metric in metrics:
            col = f"{mname}_{metric}"
            row[f"{metric}_mean"] = fold_results[col].mean()
            row[f"{metric}_std"]  = fold_results[col].std()
        rows.append(row)

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 68)
    print("  CV SUMMARY  (mean +/- std across folds)")
    print("=" * 68)
    for _, r in summary.iterrows():
        print(
            f"  {r['Model']:<14} | "
            f"LogLoss={r['logloss_mean']:.4f}+/-{r['logloss_std']:.4f} | "
            f"AUC={r['auc_mean']:.4f}+/-{r['auc_std']:.4f} | "
            f"Brier={r['brier_mean']:.4f}+/-{r['brier_std']:.4f}"
        )
    print("=" * 68)
    return summary


# -- Final model training ------------------------------------------------------

def train_final_model(
    df: pd.DataFrame,
    feature_cols: list,
    lgb_weight: float = 0.5,
) -> XGEnsemble:
    """Train the ensemble on the full dataset after CV is complete."""
    X = df[feature_cols]
    y = df["goal"]
    model = XGEnsemble(lgb_weight=lgb_weight)
    model.fit(X, y)
    print(f"Final model trained on {len(df):,} shots")
    return model


# -- Plotting ------------------------------------------------------------------

def plot_cv_results(
    fold_results: pd.DataFrame,
    save_path: str = "outputs/cv_results.png",
):
    """Line plot of AUC, LogLoss, and Brier Score per fold for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [("auc", "AUC (higher is better)"),
               ("logloss", "LogLoss (lower is better)"),
               ("brier", "Brier Score (lower is better)")]
    colors = {"LightGBM": "#4477aa", "CatBoost": "#ee6677", "Ensemble": "#228833"}

    for ax, (metric, label) in zip(axes, metrics):
        for mname, color in colors.items():
            col   = f"{mname}_{metric}"
            vals  = fold_results[col].values
            folds = fold_results["fold"].values
            ax.plot(folds, vals, "o-", label=mname, color=color, linewidth=2, markersize=7)
            ax.axhline(vals.mean(), linestyle="--", color=color, alpha=0.4, linewidth=1)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Fold")
        ax.set_xticks(fold_results["fold"].values)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Gap-Aware Time-Based CV -- xG Model", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_lgb(
    model: XGEnsemble,
    X: pd.DataFrame,
    save_path: str = "outputs/shap_lgb.png",
):
    """SHAP summary dot plot for the LightGBM classifier."""
    explainer   = shap.TreeExplainer(model.lgb_clf)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # positive class

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, show=False, plot_type="dot", max_display=15)
    plt.title("SHAP Feature Importance -- LightGBM xG Model", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# -- Persistence ---------------------------------------------------------------

def save_model(model: XGEnsemble, path: str = "outputs/xg_ensemble.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")


def load_model(path: str = "outputs/xg_ensemble.pkl") -> XGEnsemble:
    with open(path, "rb") as f:
        return pickle.load(f)
