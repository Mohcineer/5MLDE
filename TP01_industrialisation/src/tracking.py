"""Suivi MLflow (TP3) : module technique utilisé par les quatre étapes métier.

Principes :
- un run par tentative d'entraînement des deux modèles, ouvert AVANT l'acquisition ;
- suivi explicite : le `run_id` voyage dans le dictionnaire échangé entre étapes (`data["tracking"]`),
  jamais dans un « run actif » implicite (un processus enfant n'en hérite pas) ;
- toute exception d'une étape ferme le run en `FAILED` (étape + cause) puis est relancée ;
- si le serveur est injoignable, le workflow échoue explicitement : aucune publication annoncée.

Le suivi est activé par `config["tracking"]["enabled"]` ; absent, les TP1/TP2 fonctionnent comme avant.
"""
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ("main.py", "predict.py", "pyproject.toml", "uv.lock")
CODE_DIRS = ("src", "api_trigger_passif")
MAX_PARAM = 500
DEFAULT_EXPERIMENT = "5mlde-steam"


class TrackingUnavailable(RuntimeError):
    """Serveur MLflow absent ou injoignable : le workflow s'arrête avant toute publication."""


def enabled(config):
    return bool((config or {}).get("tracking", {}).get("enabled"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mlflow():
    import mlflow  # import tardif : le projet reste utilisable sans suivi
    return mlflow


def client():
    return _mlflow().MlflowClient()


def configure(config):
    """Fixer les URI depuis l'environnement (ou la config) ; refuser un `mlruns` local involontaire."""
    tracking = config.get("tracking", {})
    uri = os.getenv("MLFLOW_TRACKING_URI") or tracking.get("uri")
    if not uri:
        raise TrackingUnavailable(
            "MLFLOW_TRACKING_URI n'est pas défini : refus d'écrire dans un dossier local mlruns.")
    mlflow = _mlflow()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI") or tracking.get("registry_uri") or uri)
    return uri


def _git(*args):
    try:
        done = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=20)
        return done.stdout.strip() if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_state():
    """Révision du code et état propre/modifié (un commit seul ne décrit pas du code modifié)."""
    status = _git("status", "--porcelain", "--", ".")
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


def open_run(config, config_file, workflow, run_name=None):
    """Ouvrir le run de la tentative (avant l'acquisition) et fixer son identité."""
    tracking = config.get("tracking", {})
    uri = configure(config)
    experiment = tracking.get("experiment", DEFAULT_EXPERIMENT)
    mlflow_client = client()
    try:
        found = mlflow_client.get_experiment_by_name(experiment)
        # Le serveur choisit l'emplacement des artefacts : aucun chemin client n'est imposé.
        experiment_id = found.experiment_id if found else mlflow_client.create_experiment(experiment)
        now = datetime.now(timezone.utc)
        batch_id = tracking.get("batch_id")
        tags = {
            "workflow": workflow,
            "execution_id": uuid.uuid4().hex,
            "started_utc": now.isoformat(),
            "attempt": str(tracking.get("attempt", 1)),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyproject_sha256": sha256_file(PROJECT_ROOT / "pyproject.toml"),
            "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
            "config_sha256": sha256_text(json.dumps(config, sort_keys=True)),
            "config_file": str(config_file),
            **code_state(),
        }
        from src.orchestration import runtime_ids
        ids = runtime_ids()  # lien MLflow -> Prefect ; absent hors d'un flow
        if ids["flow_run_id"]:
            tags["prefect_flow_run_id"] = str(ids["flow_run_id"])
        if batch_id is not None:
            tags["batch_id"] = str(batch_id)
        if tracking.get("publication_id"):
            tags["publication_id"] = tracking["publication_id"]
        run = mlflow_client.create_run(
            experiment_id, tags=tags,
            run_name=run_name or tracking.get("run_name") or f"{workflow}-{now:%Y%m%dT%H%M%S}")
    except Exception as exc:
        raise TrackingUnavailable(f"Serveur MLflow indisponible ({uri}) : {exc}") from exc
    ctx = {"run_id": run.info.run_id, "experiment": experiment, "experiment_id": experiment_id,
           "workflow": workflow, "uri": uri, "files": {}, "config": config,
           "diagnostics_dir": None}
    log_code(ctx)
    return ctx


# --- écriture explicite (toujours avec un run_id) -----------------------------------------------

def _clip(value):
    return str(value)[:MAX_PARAM]


def log_params(ctx, params):
    if ctx is None:
        return
    from mlflow.entities import Param
    client().log_batch(ctx["run_id"], params=[
        Param(key, _clip(value)) for key, value in params.items() if value is not None])


def log_metrics(ctx, metrics):
    if ctx is None:
        return
    from mlflow.entities import Metric
    now = int(time.time() * 1000)
    client().log_batch(ctx["run_id"], metrics=[
        Metric(key, float(value), now, 0) for key, value in metrics.items()
        if value is not None and _finite(value)])


def _finite(value):
    try:
        return value == value and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False


def set_tags(ctx, tags):
    if ctx is None:
        return
    from mlflow.entities import RunTag
    client().log_batch(ctx["run_id"], tags=[
        RunTag(key, _clip(value)) for key, value in tags.items() if value is not None])


def log_json(ctx, payload, artifact_file):
    if ctx is not None:
        client().log_dict(ctx["run_id"], payload, artifact_file)
        ctx["files"][artifact_file] = sha256_text(json.dumps(payload, sort_keys=True, default=str))


def log_file(ctx, path, artifact_path=None):
    if ctx is None:
        return
    path = Path(path)
    client().log_artifact(ctx["run_id"], str(path), artifact_path)
    name = f"{artifact_path}/{path.name}" if artifact_path else path.name
    ctx["files"][name] = sha256_file(path)


def archive_dataframe(ctx, frame, filename, artifact_path, columns=None):
    """Archiver une projection en CSV compressé (récupérable) ; retourne son SHA-256."""
    if ctx is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        (frame[columns] if columns else frame).to_csv(path, index=False, compression="gzip")
        log_file(ctx, path, artifact_path)
        return sha256_file(path)


def log_dataset(ctx, frame, name, context, source, targets=None):
    """Associer un jeu au run (digest MLflow ≠ empreinte du fichier archivé)."""
    if ctx is None:
        return None
    from mlflow.data import from_pandas
    from mlflow.entities import DatasetInput, InputTag
    dataset = from_pandas(frame.reset_index(drop=True), source=str(source), name=name,
                          targets=targets)
    client().log_inputs(ctx["run_id"], datasets=[DatasetInput(
        dataset=dataset._to_mlflow_entity(),
        tags=[InputTag("mlflow.data.context", context)])])
    return dataset.digest


def log_model(ctx, pipeline, name, example, code_paths=None):
    """Journaliser un pipeline complet rechargeable ; retourne `ModelInfo.model_uri` (non reconstruit).

    Format cloudpickle : la construction des features est une fonction de `src`, archivée avec le
    modèle (`code_paths`) pour rester importable après récupération. Signature inférée de l'exemple.
    """
    if ctx is None:
        return None
    import mlflow
    from mlflow.models import infer_signature
    mlflow.set_tracking_uri(ctx["uri"])
    with mlflow.start_run(run_id=ctx["run_id"]):  # porté par le run_id explicite, jamais implicite
        info = mlflow.sklearn.log_model(
            pipeline, name=name, serialization_format="cloudpickle",
            signature=infer_signature(example, pipeline.predict(example)),
            input_example=example, code_paths=code_paths or [str(PROJECT_ROOT / "src")])
    return info.model_uri


def log_code(ctx):
    """Archiver les sources nécessaires (sans caches ni environnements virtuels)."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "code"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "artifacts", "runs")
        for name in CODE_DIRS:
            if (PROJECT_ROOT / name).is_dir():
                shutil.copytree(PROJECT_ROOT / name, target / name, ignore=ignore)
        for name in CODE_FILES:
            if (PROJECT_ROOT / name).is_file():
                target.mkdir(exist_ok=True)
                shutil.copy2(PROJECT_ROOT / name, target / name)
        client().log_artifacts(ctx["run_id"], str(target), "code")


# --- cycle de vie -------------------------------------------------------------------------------

@contextlib.contextmanager
def stage(ctx, name):
    """Mesure l'étape ; sur exception, ferme le run en FAILED (étape + cause) puis relance."""
    if ctx is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        fail(ctx, name, exc)
        raise
    else:
        log_metrics(ctx, {f"duration.{name}_s": time.perf_counter() - started})


def fail(ctx, stage_name, exc):
    """Marquer le run FAILED. Si le serveur est tombé, garder un diagnostic local à réconcilier."""
    info = {"run_id": ctx["run_id"], "failed_stage": stage_name,
            "error_type": type(exc).__name__, "error": str(exc)[:2000],
            "date": datetime.now(timezone.utc).isoformat(), "uri": ctx["uri"]}
    try:
        set_tags(ctx, {"failed_stage": stage_name, "error_type": info["error_type"],
                       "error": info["error"][:MAX_PARAM]})
        client().set_terminated(ctx["run_id"], "FAILED")
    except Exception as tracking_error:  # serveur indisponible : pas d'écriture à distance possible
        info["tracking_error"] = str(tracking_error)[:500]
        directory = Path(ctx.get("diagnostics_dir") or PROJECT_ROOT / "diagnostics")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"mlflow_unreconciled_{ctx['run_id']}.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def finish(ctx):
    """Terminer le run après succès (manifeste SHA-256 des fichiers journalisés inclus)."""
    if ctx is None:
        return
    log_json(ctx, {"files_sha256": dict(sorted(ctx["files"].items()))}, "manifest_sha256.json")
    client().set_terminated(ctx["run_id"], "FINISHED")


def reconcile_unclosed_runs(diagnostics_dir):
    """Après reprise du serveur : fermer en FAILED les runs dont l'échec n'avait pu être publié."""
    done = []
    for path in sorted(Path(diagnostics_dir).glob("mlflow_unreconciled_*.json")):
        info = json.loads(path.read_text(encoding="utf-8"))
        mlflow_client = client()
        mlflow_client.set_tag(info["run_id"], "failed_stage", info["failed_stage"])
        mlflow_client.set_tag(info["run_id"], "error", info["error"][:MAX_PARAM])
        mlflow_client.set_terminated(info["run_id"], "FAILED")
        path.rename(path.with_suffix(".reconciled"))
        done.append(info["run_id"])
    return done


def find_open_runs(experiment=DEFAULT_EXPERIMENT):
    """Runs restés RUNNING (arrêt brutal) à traiter manuellement ou via `reconcile`."""
    found = client().get_experiment_by_name(experiment)
    if found is None:
        return []
    runs = client().search_runs([found.experiment_id], filter_string="attributes.status = 'RUNNING'")
    return [run.info.run_id for run in runs]
