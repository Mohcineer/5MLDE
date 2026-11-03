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

    def __init__(self, message, report_path=None):
        super().__init__(message)
        self.report_path = report_path


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


def write_config(batch_id, csv_path):
    directory = batch_dir(batch_id)
    config = {
        "source": {"type": "push_batch", "path": csv_path.name},
        "output": {"runs_dir": "runs", "run_dir": "run", "diagnostics_dir": "diagnostics"},
        "random_state": 42,
        "test_size": 0.2,
        "thresholds": {"regression_min_reviews": 10, "classification_min_class_count": 5},
    }
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


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
        return {"run_dir": str(run_dir), "snapshot_sha256": digest, "already_published": True}
    if run_dir.exists():
        raise TrainingFailed(f"Le dossier de run {run_dir} existe avec une autre empreinte.")

    config_path = write_config(batch_id, csv_path)
    completed = subprocess.run(
        [sys.executable, "main.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=TRAINING_TIMEOUT_SECONDS)
    if completed.returncode != 0:
        lines = [line for line in completed.stderr.strip().splitlines() if line.strip()]
        reason = lines[-1] if lines else f"code de sortie {completed.returncode}"
        raise TrainingFailed(reason, latest_rejection_report(batch_dir(batch_id)))
    return {"run_dir": str(run_dir), "snapshot_sha256": digest, "already_published": False}
