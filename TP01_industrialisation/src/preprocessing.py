"""Prétraitement : éligibilité, features, split et ajustement sur train uniquement.

Décisions (reprises du notebook, voir data/DICTIONNAIRE.md) :
- genres nettoyés (espaces, casse, doublons) sans changer l'ordre ; premier genre = catégorie
  principale, nombre de genres et trois indicateurs multi-genres ;
- `required_age = -1` est une sentinelle historique : traitée comme manquante ;
- `log1p` sur prix, DLC et succès (conserve les zéros ; les valeurs extrêmes sont gardées) ;
- année + mois codé sur un cercle (pas d'ancienneté calculée avec la date du jour) ;
- imputation par la médiane du train avec indicateurs de manque ; genres rares regroupés
  (min_frequency=10) sur le train uniquement ; catégorie inconnue -> groupe rare ;
- appid, cibles, compteurs d'avis, positive/negative et provenance ne sont jamais des features.

Les transformations déterministes (`build_features`) font partie du pipeline sauvegardé : le
modèle accepte donc directement un CSV de métadonnées brutes à l'inférence.
"""
import ast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src import tracking

OWNERS_EXCLUDED = "0 - 0"
REGRESSION_TARGET = "pct_pos_total"
CLASSIFICATION_TARGET = "estimated_owners"

# Colonnes brutes nécessaires pour construire les features (contrat d'inférence).
RAW_FEATURE_COLUMNS = [
    "price", "required_age", "dlc_count", "achievements",
    "windows", "mac", "linux", "release_date", "genres",
]

NUMERIC_FEATURES = [
    "log_price", "required_age", "log_dlc_count", "log_achievements",
    "windows", "mac", "linux", "platform_count", "release_year",
    "month_sin", "month_cos", "is_free", "genre_count",
    "is_indie", "is_action", "is_adventure",
]
CATEGORICAL_FEATURES = ["genre"]


def _numeric(frame, column):
    """Convertir strictement : une valeur non numérique est un incident, pas un NaN silencieux."""
    series = frame[column]
    try:
        return pd.to_numeric(series, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Colonne {column} non numérique : {exc}") from exc


def parse_genres(value):
    """Nettoyer une liste de genres sérialisée ; vide ou illisible -> liste vide."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        text = str(value).strip()
        if text in ("", "Unknown"):
            return []
        try:
            raw = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Liste de genres illisible : {text!r}") from exc
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"Liste de genres illisible : {text!r}")
    return list(dict.fromkeys(str(g).strip().casefold() for g in raw if str(g).strip()))


def check_raw_columns(frame, columns=RAW_FEATURE_COLUMNS):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes indispensables absentes : {missing}")


def build_features(frame):
    """Transformations déterministes (sans paramètre appris) : brut -> features."""
    check_raw_columns(frame)
    out = pd.DataFrame(index=frame.index)

    genres = frame["genres"].map(parse_genres)
    out["genre"] = genres.map(lambda gs: gs[0] if gs else "unknown")
    out["genre_count"] = genres.map(len).astype(float)
    for genre in ("indie", "action", "adventure"):
        out["is_" + genre] = genres.map(lambda gs, g=genre: float(g in gs))

    price = _numeric(frame, "price")
    age = _numeric(frame, "required_age").mask(lambda s: s < 0)
    dlc = _numeric(frame, "dlc_count")
    achievements = _numeric(frame, "achievements")
    out["log_price"] = np.log1p(price)
    out["required_age"] = age
    out["log_dlc_count"] = np.log1p(dlc)
    out["log_achievements"] = np.log1p(achievements)

    platforms = pd.DataFrame({
        p: frame[p].map(_as_bool) for p in ("windows", "mac", "linux")
    })
    for p in platforms:
        out[p] = platforms[p].astype(float)
    out["platform_count"] = platforms.sum(axis=1).astype(float)

    dates = pd.to_datetime(frame["release_date"], errors="coerce")
    out["release_year"] = dates.dt.year.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    out["is_free"] = (price == 0).astype(float).mask(price.isna())

    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ("True", "true", 1):
        return True
    if value in ("False", "false", 0):
        return False
    raise ValueError(f"Booléen de plateforme invalide : {value!r}")


def make_preprocessor():
    """Imputation + encodage : appris sur train uniquement lors du fit du pipeline."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
         NUMERIC_FEATURES),
        ("genre", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                sparse_output=False), CATEGORICAL_FEATURES),
    ])


def make_pipeline(estimator):
    """Pipeline complet : brut -> features -> imputation/encodage -> estimateur."""
    return Pipeline([
        ("features", FunctionTransformer(build_features, validate=False)),
        ("prep", make_preprocessor()),
        ("model", estimator),
    ])


# --- éligibilité (mêmes règles pour l'audit de validation et le prétraitement) ---------------

def eligible_regression(frame, min_reviews=10):
    pct = pd.to_numeric(frame[REGRESSION_TARGET], errors="coerce")
    reviews = pd.to_numeric(frame["num_reviews_total"], errors="coerce")
    mask = pct.between(0, 100) & (reviews >= min_reviews)
    return frame[mask]


def eligible_classification(frame, min_class_count=5):
    """Exclut `0 - 0` puis les classes de moins de `min_class_count` jeux (hors périmètre)."""
    kept = frame[frame[CLASSIFICATION_TARGET].notna()
                 & (frame[CLASSIFICATION_TARGET] != OWNERS_EXCLUDED)]
    counts = kept[CLASSIFICATION_TARGET].value_counts()
    excluded = counts[counts < min_class_count].to_dict()
    kept = kept[kept[CLASSIFICATION_TARGET].isin(counts[counts >= min_class_count].index)]
    return kept, excluded


def split_sizes(n_rows, test_size):
    """Effectifs train/test qu'obtiendrait `train_test_split` (sans transformer de données)."""
    n_test = int(np.ceil(test_size * n_rows))
    return n_rows - n_test, n_test


def build_training_audit(frame, config):
    """Petite table d'audit : effectifs éligibles et split, sans rien ajuster."""
    thresholds = config.get("thresholds", {})
    test_size = config.get("test_size", 0.2)
    reg = eligible_regression(frame, thresholds.get("regression_min_reviews", 10))
    clf, _ = eligible_classification(frame, thresholds.get("classification_min_class_count", 5))
    rows = []
    for task, subset, classes in (
        ("regression", reg, 0),
        ("classification", clf, clf[CLASSIFICATION_TARGET].nunique()),
    ):
        train, test = split_sizes(len(subset), test_size)
        rows.append({
            "task": task, "eligible_count": len(subset), "train_count": train,
            "test_count": test, "class_count": int(classes),
        })
    return pd.DataFrame(rows)


# --- fonction mère ---------------------------------------------------------------------------

def _split_task(subset, target, config, stratify):
    cfg = config
    stratified = subset[target] if stratify else None
    train, test = train_test_split(
        subset, test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42), stratify=stratified,
    )
    if set(train["appid"]) & set(test["appid"]):
        raise ValueError("Un appid apparaît à la fois dans train et test.")
    return train, test


PREPROCESSING_VERSION = "features-v1"


def preprocess_data(data):
    """Contrôler doublons et colonnes, filtrer, séparer train/test pour les deux tâches.

    L'ajustement (médianes, catégories rares) se fait ensuite sur train seul, dans le pipeline
    sauvegardé avec chaque modèle (voir training.py).
    """
    ctx = data.get("tracking")
    with tracking.stage(ctx, "preprocessing"):
        prepared = _preprocess(data)
        if ctx is not None:
            _track_preprocessing(ctx, prepared, data.get("validation_corpus"))
    return prepared


def _track_preprocessing(ctx, prepared, corpus):
    """Filtres, partitions, effectifs et jeux d'entrée du run (sans fuite : rien n'est appris ici)."""
    config = prepared["config"]
    thresholds = config.get("thresholds", {})
    tracking.set_tags(ctx, {"preprocessing_version": PREPROCESSING_VERSION})
    tracking.log_params(ctx, {
        "preprocessing.version": PREPROCESSING_VERSION,
        "regression.filter": (f"pct_pos_total in [0,100] et num_reviews_total >= "
                              f"{thresholds.get('regression_min_reviews', 10)}"),
        "classification.filter": (f"estimated_owners != '0 - 0' et classes >= "
                                  f"{thresholds.get('classification_min_class_count', 5)} jeux"),
        "features": ",".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
    })
    tracking.log_json(ctx, {
        "version": PREPROCESSING_VERSION, "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES, "imputation": "médiane du train + indicateurs",
        "encoding": "OneHot, min_frequency=10, infrequent_if_exist",
        "excluded_from_features": ["appid", "cibles", "num_reviews_total", "positive", "negative",
                                   "origin"],
    }, "preprocessing.json")
    for task, info in prepared["tasks"].items():
        train, test = info["train"], info["test"]
        tracking.log_metrics(ctx, {
            f"{task}.n_eligible": len(train) + len(test), f"{task}.n_train": len(train),
            f"{task}.n_test": len(test),
            f"{task}.n_excluded_classes": len(info["excluded_classes"]),
        })
        split = pd.concat([pd.DataFrame({"appid": train["appid"].to_numpy(), "set": "train"}),
                           pd.DataFrame({"appid": test["appid"].to_numpy(), "set": "test"})])
        tracking.archive_dataframe(ctx, split, f"split_{task}.csv.gz", "partitions")
        tracking.log_dataset(ctx, train, f"{task}-train", "training", "snapshot",
                             targets=info["target"])
        if info["excluded_classes"]:
            tracking.log_json(ctx, info["excluded_classes"], f"partitions/{task}_excluded_classes.json")
        if corpus is not None:
            subset = (eligible_regression(corpus["validation"],
                                          thresholds.get("regression_min_reviews", 10))
                      if task == "regression" else
                      corpus["validation"][corpus["validation"][CLASSIFICATION_TARGET].isin(
                          corpus["manifest"]["classification_labels"])])
            tracking.log_dataset(ctx, subset, f"{task}-validation", "validation",
                                 corpus["dir"] / "validation.csv", targets=info["target"])


def _preprocess(data):
    frame = data["data"]
    config = data["config"]
    thresholds = config.get("thresholds", {})

    check_raw_columns(frame, RAW_FEATURE_COLUMNS + [
        "appid", REGRESSION_TARGET, CLASSIFICATION_TARGET, "num_reviews_total"])
    if frame["appid"].duplicated().any():
        raise ValueError("Doublons d'appid : une seule observation par jeu avant le split.")
    # Valeur non numérique = incident bloquant (jamais un zéro ou un NaN silencieux).
    for column in ("price", "dlc_count", "achievements", "required_age"):
        _numeric(frame, column)

    reg = eligible_regression(frame, thresholds.get("regression_min_reviews", 10))
    clf, excluded = eligible_classification(
        frame, thresholds.get("classification_min_class_count", 5))
    if reg.empty:
        raise ValueError("Aucune ligne éligible pour la régression.")
    if clf[CLASSIFICATION_TARGET].nunique() < 2:
        raise ValueError("Classification impossible : moins de deux classes retenues.")

    reg_train, reg_test = _split_task(reg, REGRESSION_TARGET, config, stratify=False)
    clf_train, clf_test = _split_task(clf, CLASSIFICATION_TARGET, config, stratify=True)

    return {
        "config": config,
        "paths": data["paths"],
        "validation": data.get("validation"),
        "snapshot": frame,
        "provenance": data.get("provenance"),
        "tracking": data.get("tracking"),
        "validation_corpus": data.get("validation_corpus"),
        "tasks": {
            "regression": {
                "target": REGRESSION_TARGET, "train": reg_train, "test": reg_test,
                "excluded_classes": {},
            },
            "classification": {
                "target": CLASSIFICATION_TARGET, "train": clf_train, "test": clf_test,
                "excluded_classes": {str(k): int(v) for k, v in excluded.items()},
            },
        },
    }
