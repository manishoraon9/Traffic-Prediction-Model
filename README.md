# Flipkart Gridlock 2.0 - Spatiotemporal Demand Forecasting

Round 1 solution for the Flipkart Gridlock 2.0 Machine Learning challenge (hosted on HackerEarth with the Bengaluru Traffic Police). The goal is to predict traffic/travel **demand** for city zones across time, using a LightGBM model.

## Problem

Given historical demand readings for grid cells (`geohash`) across timestamps, predict `demand` for a held-out set of future (geohash, timestamp) pairs. Demand is a continuous value roughly in `[0, 1]`.

## Data

| File | Rows | Description |
|---|---|---|
| `train.csv` | 77,299 | Historical rows with the true `demand` label |
| `test.csv` | 41,778 | Rows to predict `demand` for |
| `sample_submission.csv` | — | Output format template (`Index, demand`) |

**Columns:** `Index`, `geohash` (6-char geohash cell, ~1.2km x 0.6km), `day`, `timestamp` (15-min resolution, `H:M`), `demand` (train only), `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`.

### The critical insight: this is a forecasting problem, not a random split

EDA on the actual files showed:

- **`train`** covers **day 48 in full** (all 96 x 15-minute slots) plus the **first 9 slots of day 49** (00:00 → 02:00).
- **`test`** covers **day 49, slots 02:15 → 13:45** (47 slots) — the **direct continuation in time** of the train set.
- There is **zero timestamp overlap** between train and test.
- `RoadType`, `Weather`, `NumberofLanes`, `LargeVehicles`, and `Landmarks` are **not fixed per geohash** — they vary row to row, so they behave as noisy contextual features rather than static road attributes.

This means naive row-wise features (or a random train/val split) would both underperform and risk leaking future information. The whole feature set below is built to respect time order.

## Approach

### 1. Time features
- `minute_of_day`, `hour`, cyclical `sin`/`cos` encoding of time-of-day
- Rush-hour / night flags (morning rush, evening rush, late night)

### 2. Spatial features
- `lat` / `lon` decoded directly from the geohash string (standard base32 geohash decoding, no external dependency)
- `geohash_id` — integer-coded geohash for LightGBM

### 3. Leakage-safe temporal features (the strongest signal)
- **`demand_prev_day`** — demand for the *same geohash, same time-of-day* on day 48 (day 48 has full 24h coverage, so every day-49 slot has a matching "yesterday, same time" value)
- **`geo_last_demand`** — most recent known demand for that geohash, using only strictly earlier timestamps
- **`geo_roll4_mean`** — rolling mean of the last 4 known observations per geohash (shifted so the current row is excluded)
- **`geo_expanding_mean` / `geo_expanding_count`** — running average demand for that geohash using only prior timestamps
- **`geo_time_since_last`** — minutes since the last known observation for that geohash
- **`tod_global_mean`** — city-wide average demand at that time-of-day, learned from day 48, with **leave-one-out** so a day-48 row never sees its own value

### 4. Contextual / categorical features
`RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Weather`, `Temperature` — passed to LightGBM as native categoricals (missing values handled natively, no imputation needed).

## Model

**LightGBM** regression (`objective: regression`, RMSE metric).

- **Validation:** a *time-based* holdout — train on day 48, validate on day 49 (mirrors the real train/test time gap) — used to pick hyperparameters and early-stopping rounds.
- **Final model:** 5-fold CV bagging over the full training set; test predictions are the average of the 5 fold models (more robust than a single split). Predictions are clipped to `[0, 1]`.

### Results

| Metric | Score |
|---|---|
| Time-based holdout RMSE | 0.0299 |
| Time-based holdout MAE | 0.0207 |
| 5-fold OOF RMSE | 0.0268 |

(For reference, the target `demand` has a standard deviation of ~0.142 across train.)

**Top features by gain:** `RoadType`, `geo_last_demand`, `geo_roll4_mean`, `geo_expanding_mean`, `tod_global_mean` — confirming that the time-continuation features carry most of the predictive power.

## Files

| File | Description |
|---|---|
| `solution.py` | End-to-end script: loads data, builds all features, trains + validates the model, generates `submission.csv` |
| `submission.csv` | Final predictions in `Index, demand` format, ready to submit |

## How to run

```bash
pip install lightgbm pandas numpy scikit-learn
python solution.py
```

Place `train.csv`, `test.csv`, and `sample_submission.csv` in the same directory as `solution.py` before running. The script prints validation RMSE/MAE, per-fold CV scores, and feature importances, then writes `submission.csv`.

## Ideas for improvement

- Try `objective: tweedie` — demand looks right-skewed / zero-inflated, which Tweedie loss often handles better than plain regression.
- Add neighboring-geohash features (spatial lag: average demand of adjacent grid cells at the same time).
- Blend with a simple seasonal-naive baseline (`demand_prev_day`) as a sanity check / ensemble component.
- Hyperparameter tuning (`num_leaves`, `min_data_in_leaf`, `learning_rate`) via Optuna on the time-based validation split.
