"""Acquisition : CSV historique, collecte PULL (APIs) ou lot PUSH figé -> snapshot validé.

    load_data : acquisition -> normalisation du format -> snapshot candidat -> validation GX

Le contrat stable entre acquisition et entraînement est un DataFrame aux colonnes du CSV TP1
(+ `origin`, colonne de provenance qui n'est jamais une feature). La validation GX est la dernière
étape de `load_data` : ce qui est retourné est exactement ce qui a été validé (pas de relecture).
Avec `tracking.enabled`, le run MLflow est ouvert avant l'acquisition (voir src/tracking.py).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src import tracking
from prefect import task
from prefect.cache_policies import NO_CACHE

from src.orchestration import logger, record_flow_info, step
from src.client import AcquisitionError, SteamClient, TransientError
from src.data_validation import (HISTORICAL_ORIGIN, snapshot_hash, validate_dataframe,
                                 validate_snapshot)
from src.evaluation import resolve_run_dir
from src.normalisation import normaliser_ligne

REAL_ORIGIN = "steam-store+steamspy"
NUMERIC_COLUMNS = ["price", "required_age", "dlc_count", "achievements", "positive",
                   "negative", "pct_pos_total", "num_reviews_total"]
MAX_PULL_GAMES = 10
WORKFLOWS = {"csv": "csv-baseline", "api": "pull", "push_batch": "push"}


def _resolve(base_dir, path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def read_config(config_path):
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_file}")
    with config_file.open("r", encoding="utf-8") as file:
        return config_file, json.load(file)


def read_csv_source(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")
    data = pd.read_csv(csv_path)
    if "origin" not in data.columns:
        data["origin"] = HISTORICAL_ORIGIN
    return data


# --- PULL ---------------------------------------------------------------------------------------

def normalize_new_rows(rows):
    """Lignes normalisées -> DataFrame typé. Convertit seulement le format, jamais une valeur
    invalide en null : une valeur non numérique lève une erreur."""
    frame = pd.DataFrame(rows)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("dlc_count", "num_reviews_total", "positive", "negative"):
        if column in frame.columns and frame[column].notna().all():
            frame[column] = frame[column].astype("int64")
    frame["appid"] = frame["appid"].astype("int64")
    frame["origin"] = REAL_ORIGIN
    return frame


def _only_transient(task_obj, task_run, state):
    """Retry uniquement sur panne transitoire : jamais sur JSON/schéma invalide, 404 ou pause trop longue."""
    error = state.result(raise_on_failure=False)
    return isinstance(error, TransientError)


# Seule couche de retry HTTP : 2 retries = 3 tentatives, le client n'en fait qu'une (budget inchangé).
@task(name="Appel HTTP Steam", retries=2, retry_delay_seconds=[2, 5], retry_condition_fn=_only_transient,
      timeout_seconds=90, cache_policy=NO_CACHE, persist_result=False)
def fetch_source(client, source, appid):
    return client.get(source, appid)


def collect_games(appids, client, fetch=None):
    """Collecter 1 à 10 jeux. Un seul échec interrompt toute la collecte (pas de lot partiel)."""
    appids = list(dict.fromkeys(appids))
    if not 1 <= len(appids) <= MAX_PULL_GAMES:
        raise ValueError(f"La collecte PULL porte sur 1 à {MAX_PULL_GAMES} jeux (reçu {len(appids)}).")
    fetch = fetch or (lambda source, appid: client.get(source, appid))
    rows, sources = [], []
    for appid in appids:
        try:
            store = fetch("store", appid)
            spy = fetch("spy", appid)
            reviews = fetch("reviews", appid)
            rows.append(normaliser_ligne(appid, store, spy, reviews))
        except AcquisitionError:
            raise
        except ValueError as exc:
            raise AcquisitionError(f"Appid {appid} rejeté à la normalisation : {exc}") from exc
        sources.append({"appid": appid, **{
            s: client.provenance(s, appid) for s in ("store", "spy", "reviews")}})
    return normalize_new_rows(rows), sources


def upsert_snapshot(base, new_rows):
    """Remplacer les appids connus, ajouter les nouveaux ; une observation par appid."""
    if new_rows["appid"].duplicated().any():
        raise ValueError("Doublon d'appid parmi les jeux collectés.")
    known = set(base["appid"])
    updated = sorted(int(a) for a in new_rows["appid"] if a in known)
    added = sorted(int(a) for a in new_rows["appid"] if a not in known)
    kept = base[~base["appid"].isin(new_rows["appid"])]
    snapshot = pd.concat([kept, new_rows], ignore_index=True)
    columns = list(dict.fromkeys(list(base.columns) + list(new_rows.columns)))
    for column in ("dlc_count", "num_reviews_total", "positive", "negative"):
        if column in snapshot.columns and snapshot[column].notna().all():
            snapshot[column] = snapshot[column].astype("int64")
    return snapshot[columns], {"added": added, "updated": updated}


# --- corpus de validation réservé (TP3) ---------------------------------------------------------

def load_validation_corpus(config, base_dir):
    """Corpus figé de qualification : empreintes vérifiées, appids réservés exclus de l'apprentissage."""
    settings = config.get("validation")
    if not settings:
        return None
    directory = _resolve(base_dir, settings.get("dir", "validation"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if tracking.sha256_file(directory / name) != info["sha256"]:
            raise ValueError(f"Corpus de validation modifié (empreinte différente) : {name}")
    validation = pd.read_csv(directory / "validation.csv")
    final = pd.read_csv(directory / "final_test.csv")
    return {"validation": validation, "manifest": manifest, "dir": directory,
            "reserved_appids": set(validation["appid"]) | set(final["appid"])}


# --- fonction mère -----------------------------------------------------------------------------

def _acquire(config_file, config, ctx):
    base_dir = config_file.parent
    source = config.get("source", {})
    output = config.get("output", {})
    source_type = source.get("type")

    if source_type not in ("csv", "api", "push_batch"):
        raise ValueError("Source supportée : csv, api ou push_batch.")
    if not source.get("path"):
        raise ValueError("Le chemin du fichier source est absent de la configuration.")

    csv_path = _resolve(base_dir, source["path"])
    paths = {
        "config": config_file, "source": csv_path,
        "runs_dir": _resolve(base_dir, output.get("runs_dir", "runs")),
        "diagnostics_dir": _resolve(base_dir, output.get("diagnostics_dir", "diagnostics")),
    }
    paths["run_dir"] = resolve_run_dir(config, paths)
    if ctx is not None:
        ctx["diagnostics_dir"] = paths["diagnostics_dir"]
    if paths["run_dir"].exists():
        raise FileExistsError(f"Le dossier de run existe déjà : {paths['run_dir']}")

    provenance = {"source_type": source_type, "source_file": str(csv_path)}
    data = read_csv_source(csv_path)
    provenance["base_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    provenance["base_rows"] = len(data)

    corpus = load_validation_corpus(config, base_dir)
    if corpus is not None:
        reserved = corpus["reserved_appids"]
        before = len(data)
        data = data[~data["appid"].isin(reserved)].reset_index(drop=True)
        provenance.update(
            protocol_id=corpus["manifest"]["protocol_id"],
            validation_corpus_sha256=corpus["manifest"]["files"]["validation.csv"]["sha256"],
            reserved_appids=len(reserved), excluded_reserved=before - len(data))
        validate_dataframe(corpus["validation"].assign(origin=HISTORICAL_ORIGIN),
                           "steam_historical_v1", paths["diagnostics_dir"],
                           source="corpus de validation")

    if source_type == "api":
        acquisition = config.get("acquisition", {})
        cache_dir = _resolve(base_dir, acquisition.get("cache_dir", "cache"))
        client = SteamClient(cache_dir, refresh=bool(acquisition.get("refresh", False)), max_attempts=1)
        try:
            new_rows, sources = collect_games(
                acquisition.get("appids", []), client,
                fetch=lambda source, appid: fetch_source(client, source, appid))
        finally:
            client.close()
        if corpus is not None and set(new_rows["appid"]) & corpus["reserved_appids"]:
            raise AcquisitionError("Un appid collecté est réservé au corpus de validation.")
        # Nouveaux jeux validés AVANT l'upsert (règles strictes), puis snapshot complet.
        validate_dataframe(new_rows, "steam_new_games_v1", paths["diagnostics_dir"],
                           source="collecte PULL avant upsert")
        data, changes = upsert_snapshot(data, new_rows)
        provenance.update(changes, sources=sources, http_calls=client.http_calls)

    # Snapshot candidat -> validation GX (barrière) ; exceptions = arrêt avant le prétraitement.
    provenance["origins"] = {k: int(v) for k, v in data["origin"].value_counts().items()} \
        if "origin" in data.columns else {}
    reports = validate_snapshot(
        data, config, report_dir=paths["diagnostics_dir"], source=str(csv_path),
        push_batch=(source_type == "push_batch"))
    provenance["snapshot_sha256"] = snapshot_hash(data)
    provenance["snapshot_rows"] = len(data)

    return {"data": data, "config": config, "paths": paths, "validation": reports,
            "provenance": provenance, "validation_corpus": corpus, "tracking": ctx}


def _track_ingestion(ctx, result, config):
    """Configuration, origine, snapshot récupérable, rapports de validation, schéma."""
    provenance, paths, data = result["provenance"], result["paths"], result["data"]
    tracking_cfg = config.get("tracking", {})
    publication = tracking_cfg.get("publication_id") or (
        f"{ctx['workflow']}-{provenance['snapshot_sha256'][:16]}-"
        f"{tracking.sha256_text(json.dumps(config, sort_keys=True))[:8]}")
    ctx["publication_id"] = publication
    is_pull = provenance["source_type"] == "api"
    tracking.log_params(ctx, {
        "source.type": provenance["source_type"], "random_state": config.get("random_state"),
        "test_size": config.get("test_size"),
        "cache.refresh": config.get("acquisition", {}).get("refresh") if is_pull else None,
        "acquisition.appids": config.get("acquisition", {}).get("appids") if is_pull else None,
        "thresholds": json.dumps(config.get("thresholds", {})),
    })
    tracking.set_tags(ctx, {
        "publication_id": publication, "snapshot_sha256": provenance["snapshot_sha256"],
        "base_sha256": provenance["base_sha256"],
        "origins": json.dumps(provenance["origins"]),
        "protocol_id": provenance.get("protocol_id"),
        "validation_corpus_sha256": provenance.get("validation_corpus_sha256"),
        "ge_suites": ",".join(sorted({r["suite"] for r in result["validation"]})),
        "synthetic": str("synthetic-course" in provenance["origins"]).lower(),
    })
    tracking.log_metrics(ctx, {
        "data.rows_base": provenance["base_rows"], "data.rows_snapshot": len(data),
        "data.rows_excluded_reserved": provenance.get("excluded_reserved", 0),
        "data.rows_added": len(provenance.get("added", [])),
        "data.rows_updated": len(provenance.get("updated", [])),
    })
    tracking.log_json(ctx, config, "config_resolved.json")
    tracking.log_json(ctx, provenance, "provenance.json")
    tracking.log_json(ctx, {"columns": {c: str(t) for c, t in data.dtypes.items()}},
                      "schema.json")
    for report in result["validation"]:
        if report.get("report_file"):
            tracking.log_file(ctx, report["report_file"], "validation")
    meta = paths["source"].parent / "events_meta.json"
    if meta.exists():
        tracking.log_file(ctx, meta, "provenance")
    snapshot_sha = tracking.archive_dataframe(ctx, data, "snapshot.csv.gz", "data")
    tracking.set_tags(ctx, {"snapshot_file_sha256": snapshot_sha})
    tracking.log_dataset(ctx, data, f"snapshot-{provenance['snapshot_sha256'][:12]}",
                         "snapshot", paths["source"])


@step("Ingestion")
def load_data(config_path):
    """Charger la configuration, acquérir les données et les valider avant tout prétraitement."""
    config_file, config = read_config(config_path)
    workflow = WORKFLOWS.get(config.get("source", {}).get("type"), "csv-baseline")
    # Le run est ouvert AVANT l'acquisition ; serveur injoignable -> erreur explicite ici.
    ctx = tracking.open_run(config, config_file, workflow) if tracking.enabled(config) else None
    record_flow_info(ctx["run_id"] if ctx else None)
    log = logger()
    log.info("Ingestion : workflow=%s, config=%s, run MLflow=%s", workflow, config_file,
             ctx["run_id"] if ctx else "indisponible (suivi désactivé)")
    with tracking.stage(ctx, "ingestion"):
        result = _acquire(config_file, config, ctx)
        if ctx is not None:
            _track_ingestion(ctx, result, config)
    log.info("Ingestion terminée : %d jeux, snapshot sha256=%s", len(result["data"]),
             (result.get("provenance") or {}).get("snapshot_sha256", "indisponible"))
    return result
