# ⚽ xG Model — Expected Goals Football Analytics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-brightgreen)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/YOUR_USERNAME/xg-model/actions/workflows/ci.yml/badge.svg)
![Data](https://img.shields.io/badge/Data-StatsBomb%20Open%20Data-red)

**A production-grade Expected Goals (xG) model built with a stacked ensemble approach, gap-aware time-based cross-validation, and rich player behaviour features.**

[Overview](#overview) • [Methodology](#methodology) • [Features](#features) • [Quickstart](#quickstart) • [Results](#results) • [Project Structure](#project-structure)

</div>

---

## Overview

Expected Goals (xG) quantifies the probability that a shot results in a goal, based on shot characteristics and player context. This project builds an end-to-end xG model using **StatsBomb Open Data** (free, no API key required).

Key design decisions:
- **Gap-aware time-based CV** to simulate real-world deployment without leakage
- **Stacked ensemble** (LightGBM + CatBoost) for robust probability calibration
- **Player behaviour features** built with expanding windows to capture shooting tendencies over time

---

## Methodology

### Cross-Validation Strategy

A custom **gap-aware time-based CV** is used instead of random splits, treating matches chronologically:

```
Match timeline:
──────────────────────────────────────────────────────►
  [   TRAIN (fold 1)   ] [GAP] [  VAL  ]
          [    TRAIN (fold 2)    ] [GAP] [  VAL  ]
                  [     TRAIN (fold 3)     ] [GAP] [  VAL  ]
```

The gap between training and validation windows prevents leakage from matches that are temporally too close — simulating realistic model deployment conditions.

### Modeling Pipeline

```
StatsBomb shots → Feature engineering → GapAwareTimeSeriesCV
                                               │
                              ┌────────────────┼────────────────┐
                         LightGBM          CatBoost         Ensemble
                         (classifier)      (classifier)     (50/50 avg)
                              └────────────────┼────────────────┘
                                        xG probability
```

---

## Features

Four feature groups covering geometry, player history, interaction, and time:

### 1. Geometric Features
| Feature | Description |
|---|---|
| `distance` | Euclidean distance from shot to goal center |
| `angle` | Shot angle to goal posts (arctan cross-product) |
| `distance_sq` | Non-linear distance term |
| `angle_x_distance` | Interaction term |
| `in_box` | Shot from inside penalty box (binary) |
| `y_center_offset` | Lateral offset from goal center |

### 2. Player Behaviour Features
| Feature | Description |
|---|---|
| `player_shot_count` | Cumulative shots (expanding) |
| `player_shots_in_last_5_matches` | Recent shot frequency |
| `player_avg_xg_hist` | Historical average xG |
| `player_goals_hist` | Historical goals scored |
| `player_conversion_hist` | Historical conversion rate |

### 3. Player × ShotType Interaction Features
| Feature | Description |
|---|---|
| `player_bp_count` | Shots with this body part (expanding) |
| `player_bp_goals` | Goals with this body part (expanding) |
| `player_bp_rate` | Conversion rate per body part |
| `player_inbox_count` | Shots from this zone (expanding) |

### 4. Temporal Features
| Feature | Description |
|---|---|
| `match_phase` | early / first_half / early_2nd / late_2nd |
| `is_late_game` | Minute ≥ 75 (desperation shots) |
| `shot_number_in_match` | Shot momentum proxy |

> All player behaviour and interaction features are computed with **expanding windows** using only data prior to the current match — no future leakage.

---



### 1. Run training pipeline

```bash
python train.py
```

Data (La Liga 2015/16) is downloaded automatically — no API key or registration needed.

### 2. Run tests

```bash
pytest tests/ -v
```

---

## Results

Cross-validation results (3-fold, gap=2 matches, La Liga 2015/16):

| Model | LogLoss ↓ | AUC ↑ | Brier ↓ |
|---|---|---|---|
| LightGBM | ~0.058 | ~0.974 | ~0.058 |
| CatBoost | ~0.072 | **~0.976** | ~0.060 |
| **Ensemble (50/50)** | ~0.063 | ~0.975 | **~0.058** |

---

## Project Structure

```
xg-model-football/
├── train.py              # Main training pipeline
├── data_loader.py        # Download and load StatsBomb data
├── features.py           # All feature engineering
├── cv.py                 # GapAwareTimeSeriesCV
├── model.py              # LightGBM, CatBoost, and Ensemble
├── test_features.py      # Unit tests (leakage check, etc.)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/xg_ensemble.pkl` | Trained ensemble (train locally, not tracked by git) |
| `outputs/predictions.csv` | Per-shot xG: LGB, CatBoost, Ensemble |
| `outputs/fold_results.csv` | Metrics per CV fold |
| `outputs/cv_results.png` | AUC / LogLoss per fold plot |
| `outputs/shap_lgb.png` | SHAP feature importance (LightGBM) |

---

## Change Dataset

Edit the config block in `train.py`:

```python
COMPETITION_ID = 43   # FIFA World Cup 2018
SEASON_ID      = 3

COMPETITION_ID = 37   # Women's Super League
SEASON_ID      = 42
```

List all available free competitions:
```python
from statsbombpy import sb
print(sb.competitions())
```

---

## References

- [eddwebster/football_analytics](https://github.com/eddwebster/football_analytics) — Edd Webster
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [mplsoccer](https://mplsoccer.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/) / [CatBoost](https://catboost.ai/docs/)

---

## License

MIT License
