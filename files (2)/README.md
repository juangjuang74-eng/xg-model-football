# ⚽ xG Model — Farm-to-Feed Methodology Adaptation

Mengadaptasi **metodologi kompetisi** dari
[koleshjr/Farm_to_Feed_Shopping_Basket_Recommendation_Challenge](https://github.com/koleshjr/Farm_to_Feed_Shopping_Basket_Recommendation_Challenge)
ke dalam konteks **Expected Goals (xG) modeling** football analytics.

---

## 🔄 Mapping Metodologi

| Farm-to-Feed | xG Model (adaptasi) |
|---|---|
| Gap-aware time CV (1-week gap) | Gap-aware CV per **match gap** |
| 7-day / 14-day horizon | Prediksi per shot dalam konteks pertandingan |
| LightGBM + CatBoost + Ensemble | LightGBM + CatBoost + **50/50 Ensemble** |
| Classification (beli/tidak) | Classification (gol/tidak = xG prob) |
| Regression (jumlah beli) | Regression (xG value) |
| Customer behaviour features | **Player behaviour** features |
| Customer–Product interaction | **Player–ShotType** interaction |
| Temporal (hari, bulan) | Temporal (menit, fase pertandingan) |

---

## 📁 Struktur Project

```
xg_model/
├── src/
│   ├── data_loader.py   # StatsBomb → shots DataFrame + match_index kronologis
│   ├── features.py      # 4 kelompok fitur (geometris, behaviour, interaksi, temporal)
│   ├── cv.py            # GapAwareTimeSeriesCV — adaptasi langsung dari Farm-to-Feed
│   └── model.py         # LightGBM + CatBoost + XGEnsemble + CV training loop
├── outputs/             # Model, prediksi, plot
├── train.py             # Entry point
└── requirements.txt
```

---

## 🚀 Cara Mulai

```bash
pip install -r requirements.txt
python train.py
```

---

## 🔧 Feature Groups (4 kelompok)

### 1. Geometris (dari koordinat shot)
| Fitur | Deskripsi |
|---|---|
| `distance` | Jarak ke tengah gawang |
| `angle` | Sudut pandang ke gawang (arctan) |
| `distance_sq` | Non-linear term |
| `in_box` | Dalam kotak penalti? |
| `y_center_offset` | Seberapa ke samping dari tengah? |

### 2. Player Behaviour (analog: Customer Behaviour)
| Fitur | Deskripsi |
|---|---|
| `player_shot_count` | Total shot historis pemain |
| `player_shots_in_last_5_matches` | Shot dalam 5 laga terakhir |
| `player_avg_xg_hist` | Rata-rata xG historis |
| `player_goals_hist` | Total gol historis |
| `player_conversion_hist` | Conversion rate historis |

### 3. Player–ShotType Interaction (analog: Customer–Product)
| Fitur | Deskripsi |
|---|---|
| `player_bp_count` | Shot dengan body part ini |
| `player_bp_goals` | Gol dengan body part ini |
| `player_bp_rate` | Conversion rate per body part |
| `player_inbox_count` | Shot dari dalam box |

### 4. Temporal
| Fitur | Deskripsi |
|---|---|
| `match_phase` | early / first_half / early_2nd / late_2nd |
| `is_late_game` | Menit >= 75 |
| `shot_number_in_match` | Shot ke-N dalam pertandingan |

---

## 📊 Model & Evaluasi

### CV Strategy: Gap-Aware Time-Based (3 fold, gap 2 match)
```
Train [match 0..N]  → GAP (2 match) → Val [match N+3..N+8]
```
Gap mencegah **data leakage** dari pertandingan yang terlalu dekat.

### Models
| Model | Peran |
|---|---|
| LightGBM | Fast, good baseline, SHAP-friendly |
| CatBoost | Strong AUC, robust kategorik |
| **Ensemble (50/50)** | **Stable, terbaik secara rata-rata** |

### Metrics
- **LogLoss** — primary metric (sama dengan Farm-to-Feed)
- **AUC** — ranking quality
- **Brier Score** — calibration quality

---

## 📈 Output

| File | Isi |
|---|---|
| `xg_ensemble.pkl` | Model final (LGB + CB) |
| `predictions.csv` | xG per shot: LGB, CB, Ensemble |
| `fold_results.csv` | Metrik tiap fold |
| `cv_results.png` | Plot AUC/LogLoss per fold |
| `shap_lgb.png` | Feature importance (SHAP) |

---

## 📚 Referensi
- [Farm-to-Feed Challenge (2nd place solution)](https://github.com/koleshjr/Farm_to_Feed_Shopping_Basket_Recommendation_Challenge)
- [eddwebster/football_analytics](https://github.com/eddwebster/football_analytics)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [LightGBM docs](https://lightgbm.readthedocs.io/)
- [CatBoost docs](https://catboost.ai/docs/)
