"""Acquisition : CSV historique, collecte PULL (APIs) ou lot PUSH figé -> snapshot validé.

    load_data : acquisition -> normalisation du format -> snapshot candidat -> validation GX

Le contrat stable entre acquisition et entraînement est un DataFrame aux colonnes du CSV TP1
(+ `origin`, colonne de provenance qui n'est jamais une feature). La validation GX est la dernière
étape de `load_data` : ce qui est retourné est exactement ce qui a été validé (pas de relecture).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.client import AcquisitionError, SteamClient
from src.data_validation import (HISTORICAL_ORIGIN, snapshot_hash, validate_dataframe,
                                 validate_snapshot)
from src.evaluation import resolve_run_dir
from src.normalisation import normaliser_ligne

REAL_ORIGIN = "steam-store+steamspy"
NUMERIC_COLUMNS = ["price", "required_age", "dlc_count", "achievements", "positive",
                   "negative", "pct_pos_total", "num_reviews_total"]
MAX_PULL_GAMES = 10


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


def collect_games(appids, client):
    """Collecter 1 à 10 jeux. Un seul échec interrompt toute la collecte (pas de lot partiel)."""
    appids = list(dict.fromkeys(appids))
    if not 1 <= len(appids) <= MAX_PULL_GAMES:
        raise ValueError(f"La collecte PULL porte sur 1 à {MAX_PULL_GAMES} jeux (reçu {len(appids)}).")
    rows, sources = [], []
    for appid in appids:
        try:
            store = client.get("store", appid)
            spy = client.get("spy", appid)
            reviews = client.get("reviews", appid)
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


# --- fonction mère -----------------------------------------------------------------------------

def load_data(config_path):
    """Charger la configuration, acquérir les données et les valider avant tout prétraitement."""
    config_file, config = read_config(config_path)
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
    if paths["run_dir"].exists():
        raise FileExistsError(f"Le dossier de run existe déjà : {paths['run_dir']}")

    provenance = {"source_type": source_type, "source_file": str(csv_path)}
    data = read_csv_source(csv_path)
    provenance["base_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    provenance["base_rows"] = len(data)

    if source_type == "api":
        acquisition = config.get("acquisition", {})
        cache_dir = _resolve(base_dir, acquisition.get("cache_dir", "cache"))
        client = SteamClient(cache_dir, refresh=bool(acquisition.get("refresh", False)))
        try:
            new_rows, sources = collect_games(acquisition.get("appids", []), client)
        finally:
            client.close()
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

    return {"data": data, "config": config, "paths": paths,
            "validation": reports, "provenance": provenance}
