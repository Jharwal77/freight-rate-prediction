"""Reproduce the exploratory findings behind the modelling decisions.

Writes figures to ``outputs/figures/`` and the key numbers to
``outputs/eda_summary.json``.  Highlights:

1. The ``quote_signal`` regimes (exact leak / mirrored leak / pure noise)
   and the target-free "fingerprint" showing Nov-Dec is the noise regime.
2. The corrupted-label gaps in the rate-per-mile histogram.
3. Distance curve, weekly + annual seasonality, city/latitude effects.

Usage:  python src/analyze.py [--data-dir data] [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import is_clean_rpm, prepare
from train import find_file

COLOR = "#064A56"
ACCENT = "#C2502E"
plt.rcParams.update({
    "axes.edgecolor": "#9DAFB3", "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#D9E2E4", "font.size": 10,
})


def quote_exhibits(train: pd.DataFrame, valid: pd.DataFrame, fig_dir: Path,
                   summary: dict) -> None:
    """quote_signal: monthly correlations in training + validation fingerprint."""
    clean = train[is_clean_rpm(train["rpm"])]
    rows = []
    for month, sub in clean.groupby(clean["date"].dt.month):
        rows.append({
            "month": int(month),
            "corr_quote_rpm": float(sub["quote_signal"].corr(sub["rpm"])),
            "std_rpm_minus_quote": float((sub["rpm"] - sub["quote_signal"]).std()),
            "std_rpm_plus_quote": float((sub["rpm"] + sub["quote_signal"]).std()),
            "mean_rpm_plus_quote": float((sub["rpm"] + sub["quote_signal"]).mean()),
        })
    monthly = pd.DataFrame(rows)
    summary["quote_by_training_month"] = monthly.round(3).to_dict("records")

    # Target-free regime fingerprint: rate-per-mile falls with distance, so
    # corr(quote, log distance) reveals the regime without any labels.
    def fingerprint(frame: pd.DataFrame, label: str) -> dict:
        return {label: round(float(np.corrcoef(np.log(frame["distance"]),
                                                frame["quote_signal"])[0, 1]), 3)}
    fp = {}
    for month, sub in train.groupby(train["date"].dt.month):
        fp.update(fingerprint(sub, f"train_month_{int(month)}"))
    for month, sub in valid.groupby(valid["date"].dt.month):
        fp.update(fingerprint(sub, f"validation_month_{int(month)}"))
    summary["quote_logdistance_corr"] = fp

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = [COLOR if c > 0.5 else ACCENT if c < -0.5 else "#9DAFB3"
              for c in monthly["corr_quote_rpm"]]
    ax.bar(monthly["month"], monthly["corr_quote_rpm"], color=colors)
    ax.axhline(0, color="#455A60", linewidth=0.8)
    ax.set_xlabel("2025 month (training, clean rows)")
    ax.set_ylabel("corr(quote_signal, rate per mile)")
    ax.set_title("quote_signal flips between copy, mirror and noise", loc="left", fontsize=11)
    ax.set_xticks(range(1, 11))

    ax = axes[1]
    labels = [str(m) for m in range(1, 13)]
    keys = [f"train_month_{m}" for m in range(1, 11)] + \
           ["validation_month_11", "validation_month_12"]
    values = [fp[k] for k in keys]
    ax.bar(labels, values,
           color=[COLOR] * 10 + [ACCENT, ACCENT])
    ax.axhline(0, color="#455A60", linewidth=0.8)
    ax.set_xlabel("2025 month (last two bars = scoring period)")
    ax.set_ylabel("corr(quote_signal, log distance)")
    ax.set_title("Label-free fingerprint: Nov-Dec quote is noise", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "quote_signal_regimes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Cost of trusting the quote naively, by month (clean test rows).
    holdout_start = train["date"] >= "2025-09-01"
    sub = train[holdout_start & is_clean_rpm(train["rpm"])]
    trap = {}
    for month, part in sub.groupby(sub["date"].dt.month):
        y = part["posted_rate"].to_numpy()
        naive = part["quote_signal"].to_numpy() * part["distance"].to_numpy()
        mirrored = (4.15 - part["quote_signal"].to_numpy()) * part["distance"].to_numpy()
        trap[f"month_{int(month)}"] = {
            "rmse_quote_x_distance": round(float(np.sqrt(np.mean((y - naive) ** 2))), 2),
            "rmse_mirrored_x_distance": round(float(np.sqrt(np.mean((y - mirrored) ** 2))), 2),
        }
    summary["naive_quote_predictor_rmse"] = trap


def target_exhibits(train: pd.DataFrame, fig_dir: Path, summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bins = np.linspace(0, 8, 81)
    ax.hist(train["rpm"], bins=bins, color=COLOR, alpha=0.9)
    for lo, hi in [(0, 1.5), (3.7, 8)]:
        ax.axvspan(lo, hi, color=ACCENT, alpha=0.15)
    ax.set_xlabel("posted_rate / distance  ($ per mile)")
    ax.set_ylabel("loads")
    ax.set_title("~1.4% corrupted labels sit outside empty gaps", loc="left", fontsize=11)
    ax.set_xlim(0, 8)
    ax.text(0.97, 0.92, "329 rows < $1.00/mi\n340 rows > $4.00/mi (max $14.13)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#7A2E1A")

    clean = train[is_clean_rpm(train["rpm"])]
    binned = clean.groupby(pd.cut(clean["distance"],
                           [0, 150, 250, 400, 600, 850, 1100, 1400, 1800, 2200, 2700, 3500]),
                           observed=True)["rpm"].median()
    centers = [interval.mid for interval in binned.index]
    ax = axes[1]
    ax.plot(centers, binned.values, color=COLOR, marker="o", markersize=3.5, linewidth=2.2)
    ax.set_xlabel("distance (miles)")
    ax.set_ylabel("median rate per mile ($)")
    ax.set_title("Rate per mile declines with distance (log-like)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "target_structure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary["label_corruption"] = {
        "rows_rpm_below_1": int((train["rpm"] < 1).sum()),
        "rows_rpm_above_4": int((train["rpm"] > 4).sum()),
        "clean_share": round(float(is_clean_rpm(train["rpm"]).mean()), 4),
    }


def seasonality_exhibits(train: pd.DataFrame, fig_dir: Path, summary: dict) -> None:
    clean = train[is_clean_rpm(train["rpm"])].copy()
    slope, intercept = np.polyfit(np.log(clean["distance"]), clean["rpm"], 1)
    clean["rpm_adj"] = clean["rpm"] - (intercept + slope * np.log(clean["distance"]))
    daily = clean.groupby("date").agg(adj=("rpm_adj", "mean"),
                                       market=("market_index", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(daily.index, daily["adj"].rolling(14, center=True).mean(),
            color=COLOR, linewidth=2.2, label="distance-adjusted $/mile")
    market_scaled = (daily["market"] - daily["market"].mean()) * 0.28
    ax.plot(daily.index, market_scaled.rolling(14, center=True).mean(),
            color=ACCENT, linewidth=1.6, linestyle="--", label="market_index (scaled)")
    ax.set_ylabel("$/mile deviation")
    ax.set_title("market_index shadows seasonality, then diverges in autumn",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("2025")

    dow = clean.groupby(clean["date"].dt.dayofweek)["rpm_adj"].mean()
    ax = axes[1]
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color=COLOR)
    ax.set_ylabel("$/mile deviation")
    ax.set_title("Weekly cycle (Wed-Thu peak)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "seasonality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # How much of daily market_index is just the calendar?
    d = daily.dropna().copy()
    doy, dow_idx = d.index.dayofyear.to_numpy(), d.index.dayofweek.to_numpy()
    cols = [np.ones(len(d))]
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    cols += [(dow_idx == w).astype(float) for w in range(1, 7)]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, d["market"], rcond=None)
    resid = d["market"] - X @ coef
    summary["market_index_calendar_r2"] = round(float(1 - resid.var() / d["market"].var()), 3)


def quality_summary(train_raw: pd.DataFrame, valid_raw: pd.DataFrame,
                      train: pd.DataFrame, valid: pd.DataFrame, summary: dict) -> None:
    def rates(raw):
        return {
            "weight_missing_pct": round(float(raw["weight"].isna().mean() * 100), 2),
            "weight_negative_pct": round(float((raw["weight"] < 0).mean() * 100), 2),
            "market_index_missing_pct": round(float(raw["market_index"].isna().mean() * 100), 2),
        }
    summary["missingness"] = {"train": rates(train_raw), "validation": rates(valid_raw)}
    new_cities = (set(valid["pickup"]) | set(valid["delivery"])) - \
                 (set(train["pickup"]) | set(train["delivery"]))
    affected = valid["pickup"].isin(new_cities) | valid["delivery"].isin(new_cities)
    summary["unseen_cities_in_validation"] = {
        "cities": sorted(new_cities),
        "rows_affected": int(affected.sum()),
        "share_pct": round(float(affected.mean() * 100), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(find_file(data_dir, "train_test.csv", "train-test.csv"))
    valid_raw = pd.read_csv(find_file(data_dir, "validation.csv"))
    train, valid = prepare(train_raw), prepare(valid_raw)

    summary: dict = {}
    quote_exhibits(train, valid, fig_dir, summary)
    target_exhibits(train, fig_dir, summary)
    seasonality_exhibits(train, fig_dir, summary)
    quality_summary(train_raw, valid_raw, train, valid, summary)
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Figures written to {fig_dir}/")


if __name__ == "__main__":
    main()
