"""Déclencher le même entraînement que le TP1 sur un lot PUSH figé.

Chaque lot a un dossier stable `<state_dir>/batches/batch_<id>/` : CSV immuable des 500 jeux,
`config.json`, `diagnostics/` (rapports GX, snapshot rejeté) et `run/` (modèles, publié seulement
après succès). Un `run/` déjà publié pour la même empreinte de CSV est reconnu : pas de second
entraînement (reprise après arrêt pendant RUNNING).
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from api_trigger_passif import storage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_TIMEOUT_SECONDS = 3600
GAME_COLUMNS = ["appid", "name", "release_date", "required_age", "price", "dlc_count", "windows",
                "mac", "linux", "achievements", "genres", "pct_pos_total", "num_reviews_total"]


class TrainingFailed(RuntimeError):
    """Échec du lot (validation GX ou entraînement) : le lot passe en FAILED, sans rejeu auto."""

    def __init__(self, message, report_path=None, kind="business", flow_run_id=None, mlflow_run_id=None):
        super().__init__(message)
        self.report_path = report_path
        # business (donnée/validation), timeout, infra (MLflow/Prefect indisponible), publication, crash
        self.kind = kind
        self.flow_run_id = flow_run_id
        self.mlflow_run_id = mlflow_run_id


def batch_dir(batch_id):
    return storage.state_dir() / "batches" / f"batch_{batch_id}"


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def freeze_batch(batch):
    """Écrire le CSV immuable des jeux du lot ; refuser s'il existe avec un contenu différent."""
    events = batch["events"]
    rows = []
    for event in events:
        game = event["game"]
        rows.append({
            **{column: game[column] for column in GAME_COLUMNS if column != "genres"},
            "genres": str(game["genres"]) if game["genres"] else "Unknown",
            "estimated_owners": game["estimated_owners"],
            "origin": event["source"],
        })
    frame = pd.DataFrame(rows)
    content = frame.to_csv(index=False).encode("utf-8")
    directory = batch_dir(batch["batch_id"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "games.csv"
    if path.exists():
        if path.read_bytes() != content:
            raise TrainingFailed(f"Le CSV figé du lot {batch['batch_id']} a changé : refus de l'écraser.")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
    return path, sha256_bytes(content)


def tracking_settings(batch_id, digest):
    """Suivi MLflow du lot : un run par lot (jamais par POST), publication stable = lot + empreinte."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return None
    return {"enabled": True, "batch_id": batch_id, "publication_id": f"push-batch-{batch_id}-{digest[:12]}",
            "experiment": os.getenv("MLFLOW_EXPERIMENT", "5mlde-steam")}


def write_config(batch_id, csv_path, digest=None):
    directory = batch_dir(batch_id)
    config = {
        "source": {"type": "push_batch", "path": csv_path.name},
        "output": {"runs_dir": "runs", "run_dir": "run", "diagnostics_dir": "diagnostics"},
        "random_state": 42,
        "test_size": 0.2,
        "thresholds": {"regression_min_reviews": 10, "classification_min_class_count": 5},
    }
    tracking_cfg = tracking_settings(batch_id, digest) if digest else None
    if tracking_cfg:
        config["tracking"] = tracking_cfg
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def read_flow_info(directory):
    path = Path(directory) / "flow_info.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def classify_failure(stderr, returncode):
    """Distinguer échec métier, panne d'infrastructure et arrêt brutal (jamais rejoué automatiquement)."""
    if returncode < 0 or returncode in (3221225786, 137):  # signal / Ctrl+C Windows / SIGKILL
        return "crash"
    if any(word in stderr for word in ("TrackingUnavailable", "ConnectionError", "TransientError",
                                       "RetryLaterError")):
        return "infra"
    return "business"


def publish_models(batch_id, csv_path, digest):
    """Relier lot -> run et enregistrer les deux modèles de DÉMONSTRATION (idempotent).

    Exécuté comme un flow Prefect distinct : le flow de reprise est visible et relié au lot.
    """
    settings = tracking_settings(batch_id, digest)
    if settings is None:
        return None
    from prefect import flow
    return flow(name="Publication des modèles de démonstration", retries=0)(_publish_models)(
        batch_id, csv_path, digest, settings)


def _publish_models(batch_id, csv_path, digest, settings):
    """Reprise d'un lot déjà publié : la version existante pour cette publication est réutilisée."""
    from src import registry, tracking
    from src.orchestration import runtime_ids
    from src.preprocessing import RAW_FEATURE_COLUMNS
    tracking.configure({})
    client = tracking.client()
    experiment = client.get_experiment_by_name(settings["experiment"])
    runs = client.search_runs(
        [experiment.experiment_id], max_results=1, order_by=["attributes.start_time DESC"],
        filter_string=f"tags.publication_id = '{settings['publication_id']}' and attributes.status = 'FINISHED'")
    if not runs:
        raise TrainingFailed(f"Aucun run MLflow terminé pour la publication {settings['publication_id']}.")
    run_id = runs[0].info.run_id
    example = pd.read_csv(csv_path)[RAW_FEATURE_COLUMNS].head(5)
    versions = {}
    for task in ("regression", "classification"):
        if task == "classification" and os.getenv("STEAM_FAULT_AFTER_FIRST_VERSION"):
            # Panne simulée (TP4 partie 5) : publication partielle, identifiée et non qualifiée
            raise TrainingFailed(
                f"Panne simulée après la version {versions['regression']['version']} du modèle de régression.",
                kind="publication", mlflow_run_id=run_id)
        version = registry.register(run_id, task, settings["publication_id"], demo=True, example=example)
        report = registry.qualify(registry.model_name(task, demo=True), version.version)
        versions[task] = {"name": registry.model_name(task, demo=True), "version": version.version,
                          "qualification": report["status"]}
    link = {"batch_id": batch_id, "run_id": run_id, "publication_id": settings["publication_id"],
            "publication_flow_run_id": runtime_ids()["flow_run_id"],
            "snapshot_sha256": digest, "models": versions}
    (batch_dir(batch_id) / "mlflow.json").write_text(json.dumps(link, indent=2), encoding="utf-8")
    return link


def published_run(run_dir, csv_sha256):
    """Un run publié pour exactement ce CSV figé (empreinte du manifeste) ?"""
    manifest = run_dir / "manifest.json"
    if not (run_dir / "regression.joblib").exists() or not manifest.exists():
        return False
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (info.get("provenance") or {}).get("base_sha256") == csv_sha256


def latest_rejection_report(directory):
    reports = sorted((directory / "diagnostics").glob("validation_*.json"))
    for path in reversed(reports):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("success") is False:
                return str(path)
        except json.JSONDecodeError:
            continue
    return None


def trigger_training(batch):
    """Figer les 500 jeux, produire la config TP1 et appeler `main.py --config` (même Python)."""
    batch_id = batch["batch_id"]
    csv_path, digest = freeze_batch(batch)
    run_dir = batch_dir(batch_id) / "run"

    if published_run(run_dir, digest):
        info = read_flow_info(batch_dir(batch_id))  # ids du flow d'entraînement initial
        try:
            link = publish_models(batch_id, csv_path, digest)
        except TrainingFailed as exc:
            exc.flow_run_id = exc.flow_run_id or info.get("flow_run_id")
            raise
        return {"run_dir": str(run_dir), "snapshot_sha256": digest, "already_published": True,
                "mlflow": link, "flow_run_id": info.get("flow_run_id"),
                "mlflow_run_id": (link or {}).get("run_id") or info.get("mlflow_run_id")}
    if run_dir.exists():
        raise TrainingFailed(f"Le dossier de run {run_dir} existe avec une autre empreinte.")

    config_path = write_config(batch_id, csv_path, digest)
    info_path = batch_dir(batch_id) / "flow_info.json"
    info_path.unlink(missing_ok=True)
    try:
        completed = subprocess.run(
            [sys.executable, "main.py", "--config", str(config_path)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "STEAM_FLOW_INFO": str(info_path)},
            timeout=TRAINING_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        info = read_flow_info(batch_dir(batch_id))
        raise TrainingFailed(f"Entraînement interrompu après {TRAINING_TIMEOUT_SECONDS} s (timeout).",
                             kind="timeout", flow_run_id=info.get("flow_run_id"),
                             mlflow_run_id=info.get("mlflow_run_id")) from exc
    info = read_flow_info(batch_dir(batch_id))
    if completed.returncode != 0:
        lines = [line for line in completed.stderr.strip().splitlines() if line.strip()]
        reason = lines[-1] if lines else f"code de sortie {completed.returncode}"
        raise TrainingFailed(reason, latest_rejection_report(batch_dir(batch_id)),
                             kind=classify_failure(completed.stderr, completed.returncode),
                             flow_run_id=info.get("flow_run_id"), mlflow_run_id=info.get("mlflow_run_id"))
    try:
        link = publish_models(batch_id, csv_path, digest)
    except TrainingFailed as exc:
        exc.flow_run_id = exc.flow_run_id or info.get("flow_run_id")
        raise
    return {"run_dir": str(run_dir), "snapshot_sha256": digest, "already_published": False,
            "mlflow": link, "flow_run_id": info.get("flow_run_id"),
            "mlflow_run_id": (link or {}).get("run_id") or info.get("mlflow_run_id")}
