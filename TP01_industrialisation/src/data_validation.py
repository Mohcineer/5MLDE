"""Validation des données avec Great Expectations (GX Core 1.x) : contrat, barrière et rapports.

Suites versionnées (créées par le SDK Python, voir `build_suite`) :

| Suite                    | Périmètre                                   | Particularités                              |
|--------------------------|---------------------------------------------|---------------------------------------------|
| steam_historical_v1      | corpus complet (CSV + nouveaux jeux)        | sentinelles historiques admises (-1, 0 - 0) |
| steam_new_games_v1       | nouveaux jeux (origin != historical-csv)    | sentinelles interdites, nulls requis        |
| steam_push_batch_v1      | lot PUSH de 500 jeux                        | new_games + exactement 500 lignes           |
| steam_training_audit_v1  | table d'audit des effectifs éligibles       | classes >= 2, effectifs compatibles split   |

Toutes les expectations sont bloquantes (jamais de `mostly` < 1). Les sentinelles historiques
admises ne concernent que `required_age=-1`, `pct_pos_total=-1`, `num_reviews_total=-1` et la
tranche `0 - 0` (exclue ensuite par le prétraitement) : elles ne sont jamais étendues aux nouveaux jeux.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import great_expectations as gx
import pandas as pd

CONTRACT_VERSION = "2.0.0"
HISTORICAL_ORIGIN = "historical-csv"

SUITE_NAMES = (
    "steam_historical_v1",
    "steam_new_games_v1",
    "steam_push_batch_v1",
    "steam_training_audit_v1",
)
STRICT_SUITES = ("steam_new_games_v1", "steam_push_batch_v1")

# Features et cibles indispensables au contrat TP1 (positive/negative sont informatives).
REQUIRED_COLUMNS = [
    "appid", "name", "release_date", "required_age", "price",
    "dlc_count", "windows", "mac", "linux", "achievements",
    "genres", "pct_pos_total", "num_reviews_total", "estimated_owners",
]
NUMBER_TYPES = ["int64", "float64"]
MAX_REPORTED_APPIDS = 20


class DataValidationError(ValueError):
    """Rejet métier : un contrôle requis a échoué (rapport conservé)."""


class DataValidationTechnicalError(RuntimeError):
    """Erreur technique GX ou absence de résultat : jamais une autorisation d'entraîner."""


def create_validation_context():
    return gx.get_context(mode="ephemeral")


# --- construction des suites ------------------------------------------------------------------

def _e():
    return gx.expectations


def _add_structure(suite):
    for column in REQUIRED_COLUMNS:
        suite.add_expectation(_e().ExpectColumnToExist(column=column))
    suite.add_expectation(_e().ExpectTableRowCountToBeBetween(min_value=1))


def _add_identity(suite):
    suite.add_expectation(_e().ExpectColumnValuesToNotBeNull(column="appid"))
    suite.add_expectation(_e().ExpectColumnValuesToBeOfType(column="appid", type_="int64"))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(column="appid", min_value=1))
    suite.add_expectation(_e().ExpectColumnValuesToBeUnique(column="appid"))


def _add_numeric(suite, strict):
    for column in ("price", "required_age", "dlc_count", "achievements",
                   "pct_pos_total", "num_reviews_total"):
        suite.add_expectation(_e().ExpectColumnValuesToBeInTypeList(
            column=column, type_list=NUMBER_TYPES))
    # Bornes hautes finies : rejettent l'infini sans écarter les valeurs extrêmes réelles.
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="price", min_value=0, max_value=1_000_000))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="dlc_count", min_value=0, max_value=1_000_000))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="achievements", min_value=0, max_value=1_000_000))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="required_age", min_value=-1 if not strict else 0, max_value=100))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="pct_pos_total", min_value=-1 if not strict else 0, max_value=100))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="num_reviews_total", min_value=-1 if not strict else 10, max_value=1_000_000_000))
    if strict:
        # Une règle de plage ignore les nulls : la présence est donc exigée explicitement.
        for column in ("pct_pos_total", "num_reviews_total", "dlc_count"):
            suite.add_expectation(_e().ExpectColumnValuesToNotBeNull(column=column))


def _add_formats(suite, strict):
    for column in ("windows", "mac", "linux"):
        suite.add_expectation(_e().ExpectColumnValuesToNotBeNull(column=column))
        suite.add_expectation(_e().ExpectColumnValuesToBeOfType(column=column, type_="bool"))
        suite.add_expectation(_e().ExpectColumnValuesToBeInSet(
            column=column, value_set=[True, False]))
    suite.add_expectation(_e().ExpectColumnValuesToMatchRegex(
        column="genres", regex=r"^(\[.*\]|Unknown)$"))
    suite.add_expectation(_e().ExpectColumnValuesToMatchStrftimeFormat(
        column="release_date", strftime_format="%Y-%m-%d"))
    suite.add_expectation(_e().ExpectColumnValuesToMatchRegex(
        column="estimated_owners", regex=r"^\d+ - \d+$"))
    # Ordre des bornes (une regex seule n'attrape pas « 50000 - 20000 ») sur colonnes dérivées.
    suite.add_expectation(_e().ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="owners_max", column_B="owners_min", or_equal=not strict,
        ignore_row_if="either_value_is_missing"))
    if strict:
        for column in ("name", "release_date", "genres", "estimated_owners"):
            suite.add_expectation(_e().ExpectColumnValuesToNotBeNull(column=column))
        suite.add_expectation(_e().ExpectColumnValuesToNotBeInSet(
            column="estimated_owners", value_set=["0 - 0"]))


def _add_training_audit(suite):
    for column in ("task", "eligible_count", "train_count", "test_count", "class_count"):
        suite.add_expectation(_e().ExpectColumnToExist(column=column))
    suite.add_expectation(_e().ExpectColumnValuesToBeInSet(
        column="task", value_set=["regression", "classification"]))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(column="train_count", min_value=2))
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(column="test_count", min_value=2))
    # train + test == eligible
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="split_gap", min_value=0, max_value=0))
    # Classification : au moins deux classes et effectifs compatibles avec le split stratifié.
    classification = {"row_condition": 'task=="classification"', "condition_parser": "pandas"}
    suite.add_expectation(_e().ExpectColumnValuesToBeBetween(
        column="class_count", min_value=2, **classification))
    suite.add_expectation(_e().ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="test_count", column_B="class_count", or_equal=True, **classification))
    suite.add_expectation(_e().ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="train_count", column_B="class_count", or_equal=True, **classification))


def build_suite(name):
    if name not in SUITE_NAMES:
        raise ValueError(f"Suite inconnue : {name}")
    suite = gx.ExpectationSuite(name=name)
    if name == "steam_training_audit_v1":
        _add_training_audit(suite)
        return suite
    strict = name in STRICT_SUITES
    _add_structure(suite)
    _add_identity(suite)
    _add_numeric(suite, strict)
    _add_formats(suite, strict)
    if name == "steam_push_batch_v1":
        suite.add_expectation(_e().ExpectTableRowCountToBeBetween(min_value=500, max_value=500))
    return suite


def create_expectation_suites(context, names=SUITE_NAMES):
    suites = {}
    for name in names:
        suite = build_suite(name)
        context.suites.add(suite)
        suites[name] = suite
    return suites


# --- exécution --------------------------------------------------------------------------------

def _with_derived_columns(data, suite_name):
    """Colonnes dérivées pour les contrôles non exprimables par une regex (copie, jamais l'original)."""
    derived = data.copy()
    if suite_name == "steam_training_audit_v1":
        if {"train_count", "test_count", "eligible_count"} <= set(derived.columns):
            derived["split_gap"] = (
                derived["train_count"] + derived["test_count"] - derived["eligible_count"]).abs()
        return derived
    if "estimated_owners" in derived.columns:
        bounds = derived["estimated_owners"].astype("string").str.extract(r"^(\d+) - (\d+)$")
        derived["owners_min"] = pd.to_numeric(bounds[0])
        derived["owners_max"] = pd.to_numeric(bounds[1])
    return derived


def snapshot_hash(data):
    try:
        hashed = pd.util.hash_pandas_object(data, index=True).values.tobytes()
    except TypeError:  # colonne non hachable (listes) : repli sur la sérialisation
        hashed = data.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(hashed).hexdigest()


def _failed_controls(data, result):
    failed = []
    for item in result.results:
        if item.success:
            continue
        kwargs = dict(item.expectation_config.kwargs)
        kwargs.pop("batch_id", None)
        detail = item.result or {}
        control = {
            "expectation": item.expectation_config.type,
            "kwargs": kwargs,
            "unexpected_count": detail.get("unexpected_count"),
            "examples": [str(v) for v in (detail.get("partial_unexpected_list") or [])[:5]],
        }
        info = item.exception_info or {}
        if any(entry.get("raised_exception") for entry in info.values() if isinstance(entry, dict)):
            control["not_evaluated"] = True
            control["exception"] = next(
                (e.get("exception_message") for e in info.values() if isinstance(e, dict)
                 and e.get("raised_exception")), None)
        column = kwargs.get("column")
        if (column in data.columns and "appid" in data.columns
                and control["examples"] and column != "appid"):
            wanted = detail.get("partial_unexpected_list") or []
            rows = data.loc[data[column].isin(wanted), "appid"]
            control["appids"] = [str(a) for a in rows.head(MAX_REPORTED_APPIDS)]
        failed.append(control)
    return failed


def _save_json(directory, name, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


def validate_dataframe(data, suite_name, report_dir=None, source="unknown"):
    """Exécuter une suite GX sur le DataFrame complet ; lever une exception si elle échoue.

    - échec d'un contrôle -> `DataValidationError` (ValueError) après écriture du rapport ;
    - erreur GX ou résultat absent -> `DataValidationTechnicalError` (RuntimeError), rapport conservé.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("La validation attend un pandas.DataFrame.")
    if data.empty:
        raise ValueError("Validation refusée : DataFrame vide.")
    if suite_name not in SUITE_NAMES:
        raise ValueError(f"Suite inconnue : {suite_name}")

    digest = snapshot_hash(data)
    base = {
        "date": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "suite": suite_name,
        "suite_version": suite_name.rsplit("_", 1)[-1],
        "contract_version": CONTRACT_VERSION,
        "great_expectations_version": gx.__version__,
        "snapshot_sha256": digest,
        "row_count": len(data),
    }

    def save(report):
        if report_dir is not None:
            report["report_file"] = str(_save_json(report_dir, "validation", report))
        return report

    try:
        context = create_validation_context()
        suite = build_suite(suite_name)
        context.suites.add(suite)
        datasource = context.data_sources.add_pandas(name="steam_validation_source")
        asset = datasource.add_dataframe_asset(name="steam_validation_asset")
        batch_definition = asset.add_batch_definition_whole_dataframe(name="steam_validation_batch")
        validation_definition = gx.ValidationDefinition(
            name=f"validation_{suite_name}", data=batch_definition, suite=suite)
        context.validation_definitions.add(validation_definition)
        result = validation_definition.run(
            batch_parameters={"dataframe": _with_derived_columns(data, suite_name)})
        if result is None or not hasattr(result, "success"):
            raise RuntimeError("Great Expectations n'a retourné aucun résultat exploitable.")
        failed = _failed_controls(data, result)
        success = bool(result.success)
    except Exception as exc:
        save({**base, "success": False, "failed_controls": [], "error": str(exc),
              "error_type": type(exc).__name__,
              "not_evaluated": "tous les contrôles de la suite"})
        raise DataValidationTechnicalError(f"Échec technique de la validation : {exc}") from exc

    report = save({**base, "success": success, "failed_controls": failed,
                   "rejected_appids": sorted({a for c in failed for a in c.get("appids", [])})[
                       :MAX_REPORTED_APPIDS]})
    if not success:
        raise DataValidationError(
            f"Validation refusée avec {suite_name}. {len(failed)} contrôle(s) en échec.")
    return report


def validate_snapshot(data, config=None, report_dir=None, source="unknown", push_batch=False):
    """Valider un snapshot candidat complet : toutes les suites requises doivent réussir.

    - corpus historique/complet : `steam_historical_v1` sur toutes les lignes ;
    - nouveaux jeux (origin != historical-csv) : `steam_new_games_v1` sur ce sous-ensemble,
      pour qu'une règle stricte ne disparaisse pas dans le mélange des sources ;
    - lot PUSH : `steam_push_batch_v1` (500 lignes exactement) ;
    - aptitude à entraîner : `steam_training_audit_v1` sur les effectifs éligibles.
    """
    from src.preprocessing import build_training_audit  # import tardif : évite un cycle

    config = config or {}
    reports = []
    rejected_dir = report_dir

    def run(frame, suite, label):
        try:
            reports.append(validate_dataframe(frame, suite, report_dir, f"{source} [{label}]"))
        except (DataValidationError, DataValidationTechnicalError):
            if rejected_dir is not None:
                _save_rejected_snapshot(data, rejected_dir)
            raise

    if push_batch:
        run(data, "steam_push_batch_v1", "lot PUSH")
    else:
        run(data, "steam_historical_v1", "snapshot complet")
        if "origin" in data.columns:
            new_rows = data[data["origin"] != HISTORICAL_ORIGIN]
            if not new_rows.empty:
                run(new_rows, "steam_new_games_v1", "nouveaux jeux")
    run(build_training_audit(data, config), "steam_training_audit_v1", "aptitude à entraîner")
    return reports


def _save_rejected_snapshot(data, directory):
    directory = Path(directory) / "rejected_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rejected_{snapshot_hash(data)[:16]}.csv"
    if not path.exists():
        data.to_csv(path, index=False)
    return path
