# PROJECT 1: store-sales-time-series-forecasting

**Competition:** Kaggle — Store Sales Time Series Forecasting
**Metric:** RMSLE (Root Mean Squared Logarithmic Error)
**Leaderboard Rank:** #765
**Validation RMSLE:** 0.3853

##  ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA INGESTION                           │
│  train.csv · test.csv · stores.csv · oil.csv · holidays · txn      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING                                 │
│  • Oil: linear interpolation for gaps (weekends/holidays)           │
│  • Holidays: filter transferred=True, locale-aware flags            │
│  • Transactions: lag-16/28, rolling-7/28 (safe lags only)          │
│  • Unified dataframe: train + test concatenated                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING (67 features)                │
│  Calendar · Fourier (3 harmonics) · Earthquake decay               │
│  Target encoding · Days-to-holiday · Promo aggregates              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│           LAG & ROLLING FEATURES  (all anchored at lag≥16)         │
│  sales_lag16/21/28/35/42/56/84/364                                  │
│  rolling mean/std (w=7,14,28,56) · EWM α=0.2/0.5                  │
│  sales_lag7x4 (same-weekday 4-week mean) · sales_trend             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
          ┌──────────────┐  ┌──────────────┐
          │  LightGBM    │  │   CatBoost   │
          │  (Tweedie)   │  │   (RMSE/log) │
          │  RMSLE:0.387 │  │  RMSLE:0.387 │
          └──────┬───────┘  └──────┬───────┘
                 └────────┬─────────┘
                          ▼
              ┌─────────────────────┐
              │  ENSEMBLE (50/50)   │
              │  RMSLE: 0.3853      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  RETRAIN ON         │
              │  FULL DATA          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  ZERO-SALES RULE    │
              │  (90-day inactive)  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  submission.csv     │
              │  28,512 rows        │
              └─────────────────────┘
```

---

## Feature Engineering Details

### Calendar Features
Year, month, week, day, dayofweek, quarter, dayofyear, is_weekend, is_month_start, is_month_end, is_payday (15th + month-end), near_payday (±2d window).

### Fourier Seasonality
Three annual harmonics (sin/cos) + one weekly harmonic. Encodes cyclical patterns as smooth continuous signals rather than noisy dummy variables. Captures sub-seasonal patterns better than month dummies.

### Earthquake Features
The 2016-04-16 Ecuador earthquake disrupted supply chains for ~60 days. Three features encode this: `quake_30` (0–30 days post-quake), `quake_60` (30–60 days), and `quake_decay` (exponential with 20-day half-life). The decay provides a smooth signal instead of a hard binary cutoff.

### Oil & Macro
Ecuador is heavily oil-dependent. Seven oil features: raw price, lag-7, lag-14, 7/14/28-day moving averages, and daily change. Linear interpolation fills weekend/holiday gaps in the oil price series.

### Holiday Features (locale-aware)
- `transferred=True` rows are dropped (the original date is cancelled)
- `type='Transfer'` rows are kept (the rescheduled date is the actual holiday)
- National scope: 4 flags (holiday, event, bridge, transfer)
- Regional/Local: one presence flag + days_to_nearest_holiday (vectorised)

### Transaction Features
Daily store-level transaction counts are a strong foot-traffic proxy. Since test dates have no transaction data, all features use lags ≥16 to guarantee availability at inference. NaN for test rows is filled with 0 — the model learns during training that 0 = unavailable.

### Lag & Rolling Features (the most important group)
All anchored at `shift(16)` to match the forecast horizon. This enables **direct multi-step prediction** — the model predicts all 16 test days in a single pass without recursive error compounding.

| Feature | Description |
|---------|-------------|
| `sales_lag16/21/28/35/42/56/84` | Raw sales at fixed lags |
| `sales_lag364` | Same week 1 year ago (annual seasonality) |
| `sales_rmean7/14/28/56` | Rolling mean anchored at lag-16 |
| `sales_rstd7/14/28/56` | Rolling standard deviation |
| `sales_ewm2/5` | Exponential weighted mean (α=0.2, 0.5) |
| `sales_lag7x4` | Average of 4 prior same-weekday values |
| `sales_trend` | (recent_14d_mean − older_14d_mean) / older |

### Target Encoding
- `fam_cluster_mean_sales`: average sales per family-cluster pair (computed on train only)
- `store_fam_mean_sales`: historical mean sales per store-family pair

---

## Validation Strategy

### Primary Holdout
Val window = last 16 days of training (Aug 1–15, 2017), exactly matching the 16-day test horizon. Models are early-stopped on this window, then retrained on all data at the discovered `best_iteration_`.

**Why 16 days?** The test period is Aug 16–31 (exactly 16 days). Using the same-length preceding window as validation creates the most realistic proxy for test performance.

### Walk-Forward Cross-Validation (4 Folds)
Four additional validation folds at offsets of 16, 32, 64, and 96 days from training end. Scores are reported per fold and as mean±std. This confirms whether the model's RMSLE is stable across different time periods — a model that only works on one window is likely overfit.

### Lag Safety Guarantee
All lag features are anchored at `shift(HORIZON)` where `HORIZON=16`. This guarantees that every feature used at inference time is drawn from data that existed before the forecast window, with no data leakage and no requirement for recursive prediction.

---

## Model Architecture

### LightGBM
```
objective             = tweedie (variance_power=1.1)
n_estimators          = 6000 + early_stopping(150 rounds)
learning_rate         = 0.015
num_leaves            = 255
min_child_samples     = 50
feature/bagging frac  = 0.75
reg_alpha / lambda    = 0.1 / 0.5
```

**Why Tweedie?** Sales data is zero-inflated — many store-family pairs sell zero units on many days. The Tweedie distribution has a natural zero-mass component that MSE/RMSE objectives cannot model well. Variance power 1.1 is close to Poisson, giving stronger zero-mass.

### CatBoost
```
loss_function         = RMSE (on log1p-transformed target)
iterations            = 6000 + early_stopping(150 rounds)
learning_rate         = 0.04
depth                 = 8
l2_leaf_reg           = 3.0
min_data_in_leaf      = 50
task_type             = GPU
```

### Ensemble
Both models predict `log1p(sales)`, which is `expm1`-transformed back to sales space. The ensemble uses **inverse-RMSLE weighting** — when both models score nearly identically (as they do here), this converges to 50/50.

### Full-Data Retraining
After early stopping identifies `best_iteration_`, both models are retrained on all training data (including validation) at that iteration count. This ensures the models see the most recent 16 days of history, which are the closest to the test period.

---

## Feature Importance Analysis

<img width="1500" height="1050" alt="feature_importance" src="https://github.com/user-attachments/assets/51ee8870-f038-43ae-97b3-ddbbd5071f54" />

Top 10 features by LightGBM split importance:

| Rank | Feature | Category | Interpretation |
|------|---------|----------|----------------|
| 1 | sales_lag16 | Lag | Most direct predictor — what sold 16 days ago |
| 2 | sales_rmean14 | Rolling | 2-week rolling average (recent trend) |
| 3 | sales_ewm5 | EWM | Recency-sensitive exponential weighted mean |
| 4 | sales_lag28 | Lag | 4-week lag — monthly seasonality anchor |
| 5 | sales_lag364 | Lag | Same week last year — annual seasonality |
| 6 | sales_rmean28 | Rolling | 4-week rolling average |
| 7 | txn_ma28 | Transactions | 28-day store transaction rolling mean |
| 8 | store_fam_mean_sales | Target Enc. | Historical baseline per store-family |
| 9 | onpromotion | Promotion | Current promotion flag |
| 10 | dayofweek | Calendar | Weekly seasonality |

**Key insight:** The top 6 features are all lag/rolling features of sales itself, confirming that historical sales patterns are the dominant signal. Transaction features rank 7th, validating their inclusion despite the test-time availability caveat.

---

## Error Analysis
<img width="2700" height="750" alt="error_analysis" src="https://github.com/user-attachments/assets/e192ffe6-e8f8-4626-b379-77202855b329" />

### By Product Family (hardest to easiest)

| Family | RMSLE | Why Difficult |
|--------|-------|---------------|
| SCHOOL & OFFICE SUPPLIES | 0.61 | Extreme seasonal spikes (back-to-school) |
| BOOKS | 0.54 | Low volume, irregular demand |
| BABY CARE | 0.49 | Highly volatile, promotional sensitivity |
| HARDWARE | 0.44 | Low frequency, high variance |
| HOME CARE | 0.41 | Seasonal + promotion-driven |
| GROCERY I | 0.31 | High volume, stable, well-captured |
| BEVERAGES | 0.28 | Most stable, highest volume |

### By Store
Higher-error stores tend to be smaller-format or newer stores with shorter sales history, leading to sparser lag features and weaker rolling averages.

## Limitations
* No Short-Horizon Lag Signal
* The variance power of 1.1 was set manually. Optimal values likely vary by product family — SCHOOL AND OFFICE SUPPLIES may need a higher power than BEVERAGES.
* No Neural Models

## Improvement 
* Optuna hyperparameter tuning (50 trials, both models)
* Per-family models for top-error familie
* Recursive lag-1/7 imputation for test window

## Results Summary

| Model | Val RMSLE |
|-------|-----------|
| LightGBM (Tweedie) | 0.3871 |
| CatBoost (RMSE/log) | 0.3875 |
| **Ensemble (50/50)** | **0.3853** |
| Kaggle Leaderboard | **Rank #765** |

