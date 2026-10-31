"""Cleaning, feature engineering and target encoding for freight-rate prediction.

Design rules that follow from the EDA (see docs / report):

* The model predicts **rate per mile** (``rpm = posted_rate / distance``) and the
  final rate is ``predicted_rpm * distance``.  This matches how the target is
  generated and stops a tree model from having to re-learn proportionality
  to distance.
* ``quote_signal`` is deliberately **never** used: in some training months it
  is an almost exact copy of the target rate-per-mile, in others it is a
  mirrored copy (``4.15 - rpm``), and in August -- and in the Nov/Dec scoring
  period -- it is independent noise.  ``src/analyze.py`` reproduces the
  evidence.
* ``market_index`` is also excluded: it is ~97% explained by the calendar
  (annual Fourier terms + day-of-week), it degrades a future holdout when
  used raw, and -- like ``quote_signal`` -- it is not even present in the
  December chart inputs.
* Only inputs that exist for *every* scoring surface (validation loads and
  the fixed December lane) are used: pickup/delivery city, coordinates,
  distance, equipment, weight and date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Clean-target window.  The rpm histogram has empty gaps at ~(1.0, 1.5) and
# (3.7, 4.0): ~0.7% of rates are multiplied and ~0.7% divided by a factor of
# ~2-6.5.  Those rows are treated as corrupted labels and dropped from
# training (they are kept when reporting "all rows" holdout metrics).
CLEAN_RPM_MIN = 1.5
CLEAN_RPM_MAX = 3.7

FOURIER_K = 6        # annual harmonics (swept K=3..8 on the time holdout)
TE_SMOOTHING = 50    # pseudo-count for city target encoding
TE_FOLDS = 5         # chronological folds for out-of-fold encodings


def is_clean_rpm(rpm: pd.Series) -> pd.Series:
    return (rpm >= CLEAN_RPM_MIN) & (rpm <= CLEAN_RPM_MAX)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality issues that can be fixed without the target.

    * Negative weights are sign errors: |weight| of the negatives has exactly
      the distribution of the positives (5,000-47,500 lb), so take abs().
    * Missing weight gets an indicator; the value itself is left as NaN and
      handled natively by LightGBM.
    * ``posted_rate`` (training only) also yields the rpm modelling target.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].abs()
    if "posted_rate" in df.columns:
        df["rpm"] = df["posted_rate"] / df["distance"]
    return df


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    radius = 3958.8
    p = np.pi / 180.0
    a = (
        np.sin((lat2 - lat1) * p / 2) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_features(df: pd.DataFrame, fourier_k: int = FOURIER_K) -> pd.DataFrame:
    """Add model features. Requires coordinate columns to be present."""
    df = df.copy()
    df["log_dist"] = np.log(df["distance"])
    df["haversine"] = haversine_miles(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    df["dist_haversine_ratio"] = df["distance"] / (df["haversine"] + 1.0)
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["lat_sum"] = df["pickup_lat"] + df["delivery_lat"]
    df["lon_sum"] = df["pickup_lon"] + df["delivery_lon"]

    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Annual seasonality as Fourier terms: unlike a raw month/day-of-year
    # encoding, these extrapolate smoothly into the unseen Nov/Dec period.
    day_of_year = df["date"].dt.dayofyear
    for k in range(1, fourier_k + 1):
        df[f"sin{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    df["equipment"] = df["equipment"].astype("category")
    return df


def feature_columns(fourier_k: int = FOURIER_K, with_te: bool = True) -> list[str]:
    cols = [
        "distance", "log_dist", "weight", "weight_missing", "equipment",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "haversine", "dist_haversine_ratio", "lat_diff", "lon_diff",
        "lat_sum", "lon_sum",
        "dow", "is_weekend", "dow_sin", "dow_cos",
    ]
    cols += [f"{fn}{k}" for k in range(1, fourier_k + 1) for fn in ("sin", "cos")]
    if with_te:
        cols += ["pickup_te", "delivery_te"]
    return cols


# --------------------------------------------------------------------------
# City target encoding (smoothed means of rpm).
#
# Latitude/longitude already capture most of the city effect (city effects
# are ~75% explained by latitude), but the encodings add a small, consistent
# gain on the time holdout.  Cities that never appear in training (8 cities,
# 12.1% of validation rows) fall back to the global mean and rely on their
# coordinates.
# --------------------------------------------------------------------------

def _smoothed_map(df: pd.DataFrame, key: str, global_mean: float,
                  smoothing: int = TE_SMOOTHING) -> pd.Series:
    stats = df.groupby(key)["rpm"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)


def fit_te_maps(train_df: pd.DataFrame) -> dict:
    global_mean = float(train_df["rpm"].mean())
    return {
        "global_mean": global_mean,
        "pickup": _smoothed_map(train_df, "pickup", global_mean).to_dict(),
        "delivery": _smoothed_map(train_df, "delivery", global_mean).to_dict(),
    }


def apply_te(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    df = df.copy()
    global_mean = maps["global_mean"]
    df["pickup_te"] = df["pickup"].map(maps["pickup"]).fillna(global_mean)
    df["delivery_te"] = df["delivery"].map(maps["delivery"]).fillna(global_mean)
    return df


def add_oof_te(train_df: pd.DataFrame, n_folds: int = TE_FOLDS) -> pd.DataFrame:
    """Out-of-fold encodings for the training rows themselves.

    Folds are chronological blocks so a row's encoding never uses its own
    target (ordinary in-sample target encoding would leak).
    """
    df = train_df.sort_values("date").reset_index(drop=True).copy()
    global_mean = float(df["rpm"].mean())
    folds = np.array_split(np.arange(len(df)), n_folds)
    for out_col, key in [("pickup_te", "pickup"), ("delivery_te", "delivery")]:
        encoded = np.full(len(df), global_mean)
        for fold in folds:
            rest = df.drop(index=fold)
            mapping = _smoothed_map(rest, key, global_mean)
            encoded[fold] = df.iloc[fold][key].map(mapping).fillna(global_mean).to_numpy()
        df[out_col] = encoded
    return df


# --------------------------------------------------------------------------
# December chart inputs
# --------------------------------------------------------------------------

def city_coordinate_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city. Coordinates are constant per city in the data
    and identical whether the city appears as pickup or delivery."""
    pieces = []
    for frame in frames:
        pieces.append(
            frame[["pickup", "pickup_lat", "pickup_lon"]]
            .rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"})
        )
        pieces.append(
            frame[["delivery", "delivery_lat", "delivery_lon"]]
            .rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})
        )
    lookup = pd.concat(pieces).drop_duplicates("city").set_index("city")
    return lookup


def december_model_frame(december_inputs: pd.DataFrame,
                         lookup: pd.DataFrame) -> pd.DataFrame:
    """Turn the 7-column December file into a model-ready frame.

    The December file has no coordinates (and no market/quote columns, which
    the model does not use anyway).  Lexington and Fort Wayne both appear in
    the training data, so their fixed coordinates come from the lookup.
    """
    df = december_inputs.copy()
    for prefix, city_col in [("pickup", "pickup"), ("delivery", "delivery")]:
        df[f"{prefix}_lat"] = df[city_col].map(lookup["lat"])
        df[f"{prefix}_lon"] = df[city_col].map(lookup["lon"])
    if df[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].isna().any().any():
        raise ValueError("December city missing from coordinate lookup")
    return df
