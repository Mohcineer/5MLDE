"""Évaluation sur test et publication atomique des artefacts d'un run."""
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_absolute_error, r2_score)

from src.preprocessing import RAW_FEATURE_COLUMNS


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regression_metrics(info):
    test = info["test"]
    y = test[info["target"]]
    pred = np.clip(info["model"].predict(test[RAW_FEATURE_COLUMNS]), 0, 100)
    base = info["baseline"].predict(test[RAW_FEATURE_COLUMNS])
    return {
        "n_train": len(info["train"]), "n_test": len(test),
        "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
        "baseline_mae": float(mean_absolute_error(y, base)),
        "baseline_r2": float(r2_score(y, base)),
        "unit": "points de pourcentage",
        "hyperparameters": info["params"],
    }


def _classification_metrics(info):
    test = info["test"]
    y = test[info["target"]]
    pred = info["model"].predict(test[RAW_FEATURE_COLUMNS])
    base = info["baseline"].predict(test[RAW_FEATURE_COLUMNS])
    return {
        "n_train": len(info["train"]), "n_test": len(test),
        "n_classes": int(info["train"][info["target"]].nunique()),
        "excluded_classes": info["excluded_classes"],
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "baseline_accuracy": float(accuracy_score(y, base)),
        "baseline_f1_macro": float(f1_score(y, base, average="macro", zero_division=0)),
        "per_class": classification_report(y, pred, zero_division=0, output_dict=True),
        "hyperparameters": info["params"],
    }


def resolve_run_dir(config, paths):
    """Dossier de run : `output.run_dir` explicite, sinon horodaté sous `runs_dir`."""
    explicit = config.get("output", {}).get("run_dir")
    if explicit:
        path = Path(explicit).expanduser()
        return (path if path.is_absolute() else paths["config"].parent / path).resolve()
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    return paths["runs_dir"] / stamp


def evaluate_models(trained_models):
    """Prédire sur test, comparer aux baselines, publier les artefacts après succès des deux tâches."""
    config, paths = trained_models["config"], trained_models["paths"]
    tasks = trained_models["tasks"]

    metrics = {
        "regression": _regression_metrics(tasks["regression"]),
        "classification": _classification_metrics(tasks["classification"]),
    }

    run_dir = Path(paths.get("run_dir") or resolve_run_dir(config, paths))
    if run_dir.exists():
        raise FileExistsError(f"Le dossier de run existe déjà : {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / f".{run_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        for task, info in tasks.items():
            joblib.dump(info["model"], staging / f"{task}.joblib")
            split = pd.concat([
                pd.DataFrame({"appid": info["train"]["appid"].to_numpy(), "set": "train"}),
                pd.DataFrame({"appid": info["test"]["appid"].to_numpy(), "set": "test"}),
            ])
            split.to_csv(staging / f"split_{task}.csv", index=False)
        (staging / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (staging / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        snapshot = trained_models.get("snapshot")
        if snapshot is not None and config.get("source", {}).get("type") != "csv":
            snapshot.to_csv(staging / "snapshot.csv", index=False)

        files = {p.name: sha256_file(p) for p in sorted(staging.iterdir()) if p.is_file()}
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__, "random_state": config.get("random_state"),
            "snapshot_sha256": (trained_models.get("provenance") or {}).get("snapshot_sha256"),
            "provenance": trained_models.get("provenance"),
            "validation": trained_models.get("validation"),
            "files_sha256": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.rename(staging, run_dir)  # publication atomique ; échoue si le dossier existe
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({"run_dir": str(run_dir),
                      "regression_mae": metrics["regression"]["mae"],
                      "classification_f1_macro": metrics["classification"]["f1_macro"]}))
    return {"run_dir": run_dir, "metrics": metrics,
            "artifacts": sorted(p.name for p in run_dir.iterdir())}
