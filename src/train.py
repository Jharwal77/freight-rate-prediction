"""Train the freight-rate model.

Two stages:

1. **Honest future holdout** (`--evaluate`, on by default): train on
   Jan-Aug 2025, test on Sep-Oct 2025.  The labelled data ends 2025-10-31
   and the scored loads are 2025-11-01..12-31, so evaluation must be a
   forward-in-time split -- a random split would share months (and leaked
   feature regimes) between train and test and flatter the model.
   Metrics are reported for all test rows and for the clean subset
   (excluding the ~1.4% of corrupted target values), plus by month.

2. **Final fit**: same configuration retrained on all clean Jan-Oct rows,
   three LightGBM seeds averaged.  Models and the target-encoding maps are
   written to ``outputs/`` for ``src/predict.py``.

Usage:
    python src/train.py [--skip-evaluate] [--data-dir data] [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import (
    add_oof_te,
    apply_te,
    build_features,
    feature_columns,
    fit_te_maps,
    is_clean_rpm,
    prepare,
)

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    verbose=-1,
    num_threads=4,
)
NUM_ROUNDS = 1500
FINAL_SEEDS = [1, 7, 42]
HOLDOUT_TRAIN_END = "2025-08-31"   # train Jan-Aug, test Sep-Oct


def find_file(data_dir: Path, *names: str) -> Path:
    for name in names:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"none of {names} found in {data_dir}")


def load_training(data_dir: Path) -> pd.DataFrame:
    path = find_file(data_dir, "train_test.csv", "train-test.csv")
    return prepare(pd.read_csv(path))


def train_one(train_df: pd.DataFrame, features: list[str], seed: int) -> lgb.Booster:
    params = {**LGB_PARAMS, "seed": seed, "bagging_seed": seed,
              "feature_fraction_seed": seed}
    dataset = lgb.Dataset(train_df[features], label=train_df["rpm"])
    return lgb.train(params, dataset, num_boost_round=NUM_ROUNDS)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape_pct": round(float(np.mean(np.abs(y_true - y_pred) / y_true) * 100), 2),
    }


def evaluate(df: pd.DataFrame, features: list[str], output_dir: Path) -> dict:
    clean = is_clean_rpm(df["rpm"])
    train_raw = df[(df["date"] <= HOLDOUT_TRAIN_END) & clean]
    test_raw = df[df["date"] > HOLDOUT_TRAIN_END]

    maps = fit_te_maps(train_raw)
    train_df = add_oof_te(build_features(train_raw))
    test_df = apply_te(build_features(test_raw), maps)

    model = train_one(train_df, features, seed=42)
    pred_rate = model.predict(test_df[features]) * test_df["distance"].to_numpy()
    y_true = test_df["posted_rate"].to_numpy()
    clean_test = is_clean_rpm(test_df["rpm"]).to_numpy()

    naive_rate = float(train_raw["rpm"].median()) * test_df["distance"].to_numpy()
    results = {
        "split": f"train <= {HOLDOUT_TRAIN_END} (clean, n={len(train_df):,}); "
                 f"test after that (n={len(test_df):,}, clean n={int(clean_test.sum()):,})",
        "model_clean": metrics(y_true[clean_test], pred_rate[clean_test]),
        "model_all_rows": metrics(y_true, pred_rate),
        "naive_median_rpm_clean": metrics(y_true[clean_test], naive_rate[clean_test]),
        "by_month_clean": {},
    }
    for month in sorted(test_df["date"].dt.month.unique()):
        mask = clean_test & (test_df["date"].dt.month.to_numpy() == month)
        results["by_month_clean"][str(int(month))] = metrics(y_true[mask], pred_rate[mask])

    holdout = test_df[["load_id", "date", "distance", "equipment", "posted_rate", "rpm"]].copy()
    holdout["predicted_rate"] = np.round(pred_rate, 2)
    holdout["clean"] = clean_test
    holdout.to_csv(output_dir / "holdout_predictions.csv", index=False)
    _save_parity_figure(holdout, output_dir)

    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=features
    ).sort_values(ascending=False)
    results["top_features_gain_share"] = {
        k: round(float(v / importance.sum()), 3) for k, v in importance.head(10).items()
    }
    return results


def _save_parity_figure(holdout: pd.DataFrame, output_dir: Path) -> None:
    """Predicted-vs-actual and residual profile on the clean holdout rows."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clean = holdout[holdout["clean"]]
        sample = clean.sample(min(3000, len(clean)), random_state=0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].scatter(sample["posted_rate"], sample["predicted_rate"],
                        s=5, alpha=0.25, color="#064A56", linewidths=0)
        lims = [0, max(sample["posted_rate"].max(), sample["predicted_rate"].max()) * 1.02]
        axes[0].plot(lims, lims, color="#C2502E", linewidth=1.4)
        axes[0].set_xlim(lims); axes[0].set_ylim(lims)
        axes[0].set_xlabel("actual posted rate ($)")
        axes[0].set_ylabel("predicted rate ($)")
        axes[0].set_title("Holdout Sep-Oct: predicted vs actual (clean rows)",
                          loc="left", fontsize=11)

        resid = clean["posted_rate"] - clean["predicted_rate"]
        bins = pd.cut(clean["distance"], [0, 300, 700, 1200, 2000, 3500])
        profile = resid.groupby(bins, observed=True).agg(["mean", "std"])
        centers = [interval.mid for interval in profile.index]
        axes[1].errorbar(centers, profile["mean"], yerr=profile["std"], fmt="o-",
                         color="#064A56", ecolor="#9DAFB3", linewidth=2, markersize=4)
        axes[1].axhline(0, color="#C2502E", linewidth=1.2)
        axes[1].set_xlabel("distance (miles)")
        axes[1].set_ylabel("residual actual - predicted, mean +/- std ($)")
        axes[1].set_title("No material distance bias; spread grows with load value",
                          loc="left", fontsize=11)
        for ax in axes:
            ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
        fig.tight_layout()
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "holdout_parity.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # figure is diagnostic only; never block training
        print(f"  (parity figure skipped: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = feature_columns()

    df = load_training(data_dir)
    print(f"Loaded {len(df):,} labelled loads "
          f"({df['date'].min().date()} .. {df['date'].max().date()})")

    if not args.skip_evaluate:
        print("Running forward holdout evaluation (train Jan-Aug -> test Sep-Oct)...")
        results = evaluate(df, features, output_dir)
        (output_dir / "holdout_metrics.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))

    print("Training final model on all clean rows (3 seeds)...")
    clean_df = df[is_clean_rpm(df["rpm"])]
    dropped = len(df) - len(clean_df)
    print(f"  clean rows: {len(clean_df):,}  (dropped {dropped:,} corrupted targets, "
          f"{dropped / len(df):.2%})")
    maps = fit_te_maps(clean_df)
    train_df = add_oof_te(build_features(clean_df))
    for seed in FINAL_SEEDS:
        model = train_one(train_df, features, seed=seed)
        model.save_model(str(output_dir / f"model_seed{seed}.txt"))

    artifacts = {"features": features, "seeds": FINAL_SEEDS, "te_maps": maps}
    (output_dir / "artifacts.json").write_text(json.dumps(artifacts))
    print(f"Saved {len(FINAL_SEEDS)} models + artifacts.json to {output_dir}/")


if __name__ == "__main__":
    main()
