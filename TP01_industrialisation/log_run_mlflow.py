"""Envoyer un run d'entraînement (dossier runs/...) dans MLflow : paramètres, métriques, artefacts.

Usage :
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    uv run python log_run_mlflow.py --run-dir runs/<dossier_du_run>
"""
import argparse
import json
import os
from pathlib import Path

import mlflow

DEFAULT_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "Steam_TP03"


def flat(prefix, mapping):
    """Aplatir un dictionnaire d'hyperparamètres en paramètres MLflow."""
    return {f"{prefix}.{key}": value for key, value in mapping.items()}


def log_run(run_dir, experiment=DEFAULT_EXPERIMENT, run_name=None):
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest.get("provenance") or {}

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name or run_dir.name) as run:
        reg, clf = metrics["regression"], metrics["classification"]

        mlflow.log_params({
            "source_type": config.get("source", {}).get("type"),
            "random_state": config.get("random_state"),
            "test_size": config.get("test_size"),
            **flat("regression", reg.get("hyperparameters", {})),
            **flat("classification", clf.get("hyperparameters", {})),
        })
        mlflow.log_metrics({
            "reg_mae": reg["mae"], "reg_r2": reg["r2"],
            "reg_baseline_mae": reg["baseline_mae"], "reg_baseline_r2": reg["baseline_r2"],
            "reg_n_train": reg["n_train"], "reg_n_test": reg["n_test"],
            "clf_accuracy": clf["accuracy"], "clf_f1_macro": clf["f1_macro"],
            "clf_baseline_accuracy": clf["baseline_accuracy"],
            "clf_baseline_f1_macro": clf["baseline_f1_macro"],
            "clf_n_train": clf["n_train"], "clf_n_test": clf["n_test"],
            "clf_n_classes": clf["n_classes"],
        })
        mlflow.set_tags({
            "snapshot_sha256": provenance.get("snapshot_sha256", ""),
            "snapshot_rows": str(provenance.get("snapshot_rows", "")),
            "scikit_learn": manifest.get("scikit_learn", ""),
        })

        # Modèles, rapports et traçabilité : mêmes fichiers que le dossier du run.
        for name in ("regression.joblib", "classification.joblib"):
            mlflow.log_artifact(str(run_dir / name), artifact_path="models")
        for name in ("metrics.json", "config.json", "manifest.json",
                     "split_regression.csv", "split_classification.csv"):
            mlflow.log_artifact(str(run_dir / name), artifact_path="run")
        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Envoyer un run dans MLflow")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    print("Run MLflow :", log_run(args.run_dir, args.experiment, args.run_name))
