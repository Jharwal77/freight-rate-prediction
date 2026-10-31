"""Produce the two submission files from the trained models.

Outputs:
  * ``validation_predictions.csv`` (repo root) -- load_id,predicted_rate for
    the 12,000 Nov/Dec loads, in the template's row order.
  * ``outputs/december_predictions.csv`` -- the December chart inputs with
    ``predicted_rate`` filled in (original seven columns, unchanged).

Run ``src/train.py`` first.  Then validate + chart with the provided scorer:

    python score.py --predictions validation_predictions.csv \
        --december-predictions outputs/december_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (
    apply_te,
    build_features,
    city_coordinate_lookup,
    december_model_frame,
    prepare,
)
from train import find_file


def load_models(output_dir: Path) -> tuple[list[lgb.Booster], list[str], dict]:
    artifacts = json.loads((output_dir / "artifacts.json").read_text())
    models = [lgb.Booster(model_file=str(output_dir / f"model_seed{s}.txt"))
              for s in artifacts["seeds"]]
    return models, artifacts["features"], artifacts["te_maps"]


def predict_rates(models, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    rpm = np.mean([m.predict(frame[features]) for m in models], axis=0)
    return rpm * frame["distance"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--predictions-out", default="validation_predictions.csv")
    args = parser.parse_args()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)

    models, features, maps = load_models(output_dir)

    # ---- 12,000 validation loads, filled into the provided template -------
    validation = prepare(pd.read_csv(find_file(data_dir, "validation.csv")))
    scored = apply_te(build_features(validation), maps)
    validation["predicted_rate"] = np.round(predict_rates(models, scored, features), 2)
    assert (validation["predicted_rate"] > 0).all()

    template = pd.read_csv(find_file(
        data_dir, "validation_predictions_template.csv", "validation-predictions-template.csv"))
    filled = template.drop(columns=["predicted_rate"]).merge(
        validation[["load_id", "predicted_rate"]], on="load_id", how="left", validate="one_to_one")
    assert len(filled) == 12_000 and filled["predicted_rate"].notna().all()
    filled.to_csv(args.predictions_out, index=False)
    print(f"Wrote {args.predictions_out}: {len(filled):,} rows, "
          f"predicted rate ${filled['predicted_rate'].min():,.0f}"
          f" - ${filled['predicted_rate'].max():,.0f}"
          f" (median ${filled['predicted_rate'].median():,.0f})")

    # ---- Fixed December lane ----------------------------------------------
    december = pd.read_csv(find_file(
        data_dir, "december_chart_inputs.csv", "december-chart-inputs.csv"))
    original_columns = list(december.columns)
    lookup = city_coordinate_lookup(
        pd.read_csv(find_file(data_dir, "train_test.csv", "train-test.csv")),
        pd.read_csv(find_file(data_dir, "validation.csv")),
    )
    scored_dec = apply_te(build_features(prepare(december_model_frame(december, lookup))), maps)
    december["predicted_rate"] = np.round(predict_rates(models, scored_dec, features), 2)
    december = december[original_columns]
    assert len(december) == 31 and (december["predicted_rate"] > 0).all()
    out_path = output_dir / "december_predictions.csv"
    december.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: Dec rates "
          f"${december['predicted_rate'].min():,.0f} - ${december['predicted_rate'].max():,.0f}")


if __name__ == "__main__":
    main()
