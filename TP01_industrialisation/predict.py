"""Prédiction : recharge les deux modèles d'un run et écrit appid, pct_positive_pred, estimated_owners_pred."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import RAW_FEATURE_COLUMNS, check_raw_columns


def predict(run_dir, input_csv, output_csv):
    run_dir = Path(run_dir)
    frame = pd.read_csv(input_csv)
    check_raw_columns(frame, ["appid"] + RAW_FEATURE_COLUMNS)
    regression = joblib.load(run_dir / "regression.joblib")
    classification = joblib.load(run_dir / "classification.joblib")
    result = pd.DataFrame({
        "appid": frame["appid"],
        "pct_positive_pred": np.clip(regression.predict(frame[RAW_FEATURE_COLUMNS]), 0, 100),
        "estimated_owners_pred": classification.predict(frame[RAW_FEATURE_COLUMNS]),
    })
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire avec les deux modèles d'un run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True, help="CSV de métadonnées sans les cibles")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(predict(args.run_dir, args.input, args.output).head())
