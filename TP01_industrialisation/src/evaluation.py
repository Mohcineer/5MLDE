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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

from src import tracking
from src.preprocessing import (CLASSIFICATION_TARGET, RAW_FEATURE_COLUMNS, eligible_regression)


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


def _validation_metrics(task, info, corpus, config):
    """Qualification commune : même corpus réservé pour toutes les variantes, baseline incluse."""
    frame = corpus["validation"]
    if task == "regression":
        frame = eligible_regression(frame, config.get("thresholds", {}).get("regression_min_reviews", 10))
        y = pd.to_numeric(frame[info["target"]])
        x = frame[RAW_FEATURE_COLUMNS]
        pred = np.clip(info["model"].predict(x), 0, 100)
        base = info["baseline"].predict(x)
        return {"n": len(frame), "mae": float(mean_absolute_error(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred)),
                "baseline_mae": float(mean_absolute_error(y, base)),
                "baseline_rmse": float(np.sqrt(mean_squared_error(y, base))),
                "baseline_r2": float(r2_score(y, base))}
    labels = list(corpus["manifest"]["classification_labels"])
    frame = frame[frame[CLASSIFICATION_TARGET].isin(labels)]
    y, x = frame[info["target"]], frame[RAW_FEATURE_COLUMNS]
    pred, base = info["model"].predict(x), info["baseline"].predict(x)
    return {"n": len(frame), "labels": labels,
            "accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "baseline_accuracy": float(accuracy_score(y, base)),
            "baseline_f1_macro": float(f1_score(y, base, labels=labels, average="macro", zero_division=0)),
            "per_class": classification_report(y, pred, labels=labels, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "classes_absent_from_train": sorted(set(labels) - set(info["train"][info["target"]]))}


def _log_run_results(ctx, tasks, metrics, validation, corpus):
    """Métriques de test interne et de validation commune, séparées ; modèles rechargeables."""
    for task, info in tasks.items():
        scalar = lambda d: {k: v for k, v in d.items() if isinstance(v, (int, float))}
        tracking.log_metrics(ctx, {f"{task}.test.{k}": v for k, v in scalar(metrics[task]).items()})
        if task in validation:
            tracking.log_metrics(ctx, {f"{task}.validation.{k}": v
                                       for k, v in scalar(validation[task]).items()})
        example = info["test"][RAW_FEATURE_COLUMNS].head(5)
        uri = tracking.log_model(ctx, info["model"], f"{task}-model", example)
        tracking.set_tags(ctx, {f"{task}.model_uri": uri})
    tracking.log_json(ctx, metrics, "reports/test_metrics.json")
    if validation:
        tracking.log_json(ctx, {"protocol_id": corpus["manifest"].get("protocol_id"),
                                "validation_corpus_sha256":
                                    corpus["manifest"]["files"]["validation.csv"]["sha256"],
                                "results": validation}, "reports/validation_metrics.json")


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
    ctx = trained_models.get("tracking")
    with tracking.stage(ctx, "evaluation"):
        result = _evaluate(trained_models, ctx)
    tracking.finish(ctx)
    return result


def _evaluate(trained_models, ctx):
    config, paths = trained_models["config"], trained_models["paths"]
    tasks = trained_models["tasks"]

    metrics = {
        "regression": _regression_metrics(tasks["regression"]),
        "classification": _classification_metrics(tasks["classification"]),
    }

    corpus = trained_models.get("validation_corpus")
    validation = ({task: _validation_metrics(task, info, corpus, config) for task, info in tasks.items()}
                  if corpus is not None else {})
    _log_run_results(ctx, tasks, metrics, validation, corpus)

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
