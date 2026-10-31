# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts `posted_rate` for 12,000 future loads (2025-11-01 → 2025-12-31) from
48,000 labelled loads (2025-01-01 → 2025-10-31), plus the fixed
Lexington → Fort Wayne December chart.

**Headline result (forward holdout, train Jan–Aug → test Sep–Oct):**
RMSE **$52.56**, MAE **$36.23**, MAPE **1.57%** on clean test loads
(naive median-rate baseline: RMSE $295.92, MAPE 9.27%).

## The three decisions that matter most

1. **`quote_signal` is excluded — it is an unstable leak.** On clean training
   rows its correlation with the true rate-per-mile is **+0.996** in some
   months (it literally is the target, `std(rpm − quote) = $0.022/mi`),
   **−0.994** in others (it is the mirror image `4.15 − rpm`), and **≈ 0**
   in August. A label-free fingerprint — `corr(quote, log distance)`, which
   is ±0.80 in informative months — is **+0.009 / +0.013 in Nov / Dec**, i.e.
   the quote is pure noise in the scoring period. Using it made the future
   holdout *worse* ($63.91 → $77.51 RMSE), and the naive `quote × distance`
   predictor goes from RMSE $29.91 in September to $481.75 in October.
2. **`market_index` is excluded — it duplicates the calendar.** 97% of its
   daily variation is explained by annual Fourier terms + day-of-week, and
   adding it raw degraded the forward holdout ($63.91 → $91.83) because its
   autumn *levels* collide with winter levels that have different rates.
   The model uses Fourier/calendar features instead, which extrapolate.
   (Neither field exists in the December chart inputs — the production
   feature set only uses city, coordinates, distance, equipment, weight
   and date.)
3. **~1.4% of labels are corrupted and are dropped from training.** The
   rate-per-mile histogram has empty gaps at ~($1.0, $1.5) and ($3.7, $4.0);
   outside them sit 329 loads below $1/mi and 340 above $4/mi — rates
   divided/multiplied by ~2–6.5×. Training keeps rpm ∈ [$1.50, $3.70].

Other data-quality handling: negative weights are sign errors → `abs()`
(their |weight| distribution matches the positives exactly); missing weight
gets an indicator and LightGBM-native NaN handling; 8 cities in the scoring
set never appear in training (12.1% of rows) and are priced from their
coordinates (city effects are ~75% explained by latitude) with
global-mean target encodings as fallback.

## Approach in one paragraph

The model predicts **rate per mile** and multiplies by distance. A LightGBM
gradient-boosted ensemble (3 seeds averaged) uses distance/haversine and
their log/ratio, equipment, weight, pickup/delivery coordinates and diffs,
day-of-week, and annual Fourier harmonics (K=6), plus smoothed
out-of-fold city target encodings. Validation is **forward in time**
(train ≤ 2025-08-31, test = Sep–Oct), mirroring the Nov–Dec prediction
task; a random split is shown to flatter leaky features and is not used
for any decision. The final model refits on all clean Jan–Oct data.

## Repository layout

```
├── data/                  # provided assessment files
├── src/
│   ├── features.py        # cleaning, features, target encoding, Dec frame
│   ├── analyze.py         # reproduces all EDA exhibits -> outputs/figures
│   ├── train.py           # forward-holdout evaluation + final fit
│   └── predict.py         # submission files
├── score.py               # provided by Spotter (validator + Dec chart)
├── validation_predictions.csv
├── outputs/               # metrics, figures, candidate_december.png, ...
└── docs/                  # PDF report
```

## How to run

```bash
python -m pip install -r requirements.txt

python src/analyze.py     # optional: EDA figures + eda_summary.json
python src/train.py       # holdout evaluation, then final 3-seed fit (~2 min)
python src/predict.py     # validation_predictions.csv + December predictions

python score.py --predictions validation_predictions.csv \
    --december-predictions outputs/december_predictions.csv \
    --output-dir outputs
```

Expected scorer output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: outputs/candidate_december.png
```

## December chart

December prices come from the same final model. The December file has no
coordinates, so Lexington/Fort Wayne take their fixed training coordinates
(coordinates are constant per city); no `market_index`/`quote_signal`
values are needed because the model never uses them. Predicted rates run
$793–$816 (implied $2.20–$2.27/mile vs. a $2.23/mile historical median for
this lane/equipment), with the learned weekly cycle (Wed–Thu peaks,
weekend troughs) over a gentle seasonal easing into late December:
see `outputs/candidate_december.png` and the report in `docs/`.
