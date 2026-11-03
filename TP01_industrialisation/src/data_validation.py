
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import great_expectations as gx


CONTRACT_VERSION = "1.0.0"

REQUIRED_COLUMNS = [
    "appid", "name", "release_date", "required_age", "price",
    "dlc_count", "windows", "mac", "linux", "achievements",
    "genres", "positive", "negative", "pct_pos_total",
    "num_reviews_total", "estimated_owners",
]


def create_validation_context():
    return gx.get_context()


def add_base_expectations(suite):
    for column in REQUIRED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=column)
        )

    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="appid")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="appid", type_="int64"
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="appid", min_value=1
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="appid")
    )


def add_numeric_expectations(suite, historical=False):
    for column in [
        "price",
        "dlc_count",
        "achievements",
        "positive",
        "negative",
    ]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
                mostly=1.0,
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="num_reviews_total",
            min_value=-1 if historical else 0,
            mostly=1.0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="required_age",
            min_value=-1 if historical else 0,
            mostly=1.0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="pct_pos_total",
            min_value=-1 if historical else 0,
            max_value=100,
            mostly=1.0,
        )
    )

    if not historical:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="num_reviews_total",
                min_value=10,
                mostly=1.0,
            )
        )


def add_format_expectations(suite, strict=False):
    for column in ["windows", "mac", "linux"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=[True, False],
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="genres",
            type_="object",
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="release_date",
            regex=r"^\d{4}-\d{2}-\d{2}$",
            mostly=1.0,
        )
    )

    if strict:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column="name"
            )
        )


def add_owners_expectations(suite, strict=False):
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="estimated_owners",
            regex=r"^\d+ - \d+$",
            mostly=1.0,
        )
    )

    if strict:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeInSet(
                column="estimated_owners",
                value_set=["0 - 0"],
            )
        )


def validate_owners_ranges(data, strict=False):
    owners = data["estimated_owners"].dropna()

    for value in owners:
        parts = value.split(" - ")

        if len(parts) != 2:
            raise ValueError(
                f"Format estimated_owners invalide : {value}"
            )

        try:
            minimum, maximum = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Bornes estimated_owners invalides : {value}"
            ) from exc

        if minimum < 0 or maximum < 0 or minimum > maximum:
            raise ValueError(
                f"Tranche estimated_owners incohérente : {value}"
            )

        if strict and minimum == 0 and maximum == 0:
            raise ValueError(
                "La tranche 0 - 0 est interdite pour les nouveaux jeux."
            )

    return True


def add_training_audit_expectations(suite):
    audit_columns = [
        "task",
        "eligible_count",
        "train_count",
        "test_count",
        "class_count",
    ]

    for column in audit_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=column)
        )

    for column in [
        "eligible_count",
        "train_count",
        "test_count",
        "class_count",
    ]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="task",
            value_set=["regression", "classification"],
        )
    )


def validate_training_audit(data):
    """Vérifier la cohérence des effectifs de l'audit d'entraînement."""

    for _, row in data.iterrows():
        task = row["task"]
        eligible = int(row["eligible_count"])
        train = int(row["train_count"])
        test = int(row["test_count"])
        class_count = int(row["class_count"])

        if train + test != eligible:
            raise ValueError(
                f"Effectifs incohérents pour {task} : "
                f"train ({train}) + test ({test}) != eligible ({eligible})."
            )

        if train <= 0 or test <= 0:
            raise ValueError(
                f"Effectifs train/test insuffisants pour {task}."
            )

        if task == "classification" and class_count < 2:
            raise ValueError(
                "Classification impossible : au moins deux classes sont nécessaires."
            )

    return True


def create_expectation_suites(context):
    suite_names = [
        "steam_historical_v1",
        "steam_new_games_v1",
        "steam_push_batch_v1",
        "steam_training_audit_v1",
    ]

    suites = {}

    for name in suite_names:
        suite = gx.ExpectationSuite(name=name)

        if name == "steam_training_audit_v1":
            add_training_audit_expectations(suite)

        else:
            add_base_expectations(suite)

            historical = name == "steam_historical_v1"
            strict = name in [
                "steam_new_games_v1",
                "steam_push_batch_v1",
            ]

            add_numeric_expectations(suite, historical)
            add_format_expectations(suite, strict)
            add_owners_expectations(suite, strict)

            if name == "steam_push_batch_v1":
                suite.add_expectation(
                    gx.expectations.ExpectTableRowCountToBeBetween(
                        min_value=500,
                        max_value=500,
                    )
                )

        context.suites.add(suite)
        suites[name] = suite

    return suites


def validate_dataframe(
    data,
    suite_name,
    report_dir=None,
    source="unknown",
):
    """Exécuter une suite GX et bloquer le traitement si elle échoue."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "La validation attend un pandas.DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Validation refusée : DataFrame vide."
        )

    allowed_suites = [
        "steam_historical_v1",
        "steam_new_games_v1",
        "steam_push_batch_v1",
        "steam_training_audit_v1",
    ]

    if suite_name not in allowed_suites:
        raise ValueError(f"Suite inconnue : {suite_name}")

    snapshot_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            data,
            index=True,
        ).values.tobytes()
    ).hexdigest()

    def save_report(report):
        if report_dir is not None:
            directory = Path(report_dir)
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            report_path = directory / (
                f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            )

            report_path.write_text(
                json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    try:
        # Contrôles complémentaires.
        if suite_name == "steam_training_audit_v1":
            validate_training_audit(data)
        else:
            validate_owners_ranges(
                data,
                strict=suite_name in [
                    "steam_new_games_v1",
                    "steam_push_batch_v1",
                ],
            )

        # Création du contexte Great Expectations.
        context = create_validation_context()
        suites = create_expectation_suites(context)

        datasource = context.data_sources.add_pandas(
            name="steam_validation_source"
        )

        asset = datasource.add_dataframe_asset(
            name="steam_validation_asset"
        )

        batch_definition = asset.add_batch_definition_whole_dataframe(
            name="steam_validation_batch"
        )

        validation_definition = gx.ValidationDefinition(
            name=f"validation_{suite_name}",
            data=batch_definition,
            suite=suites[suite_name],
        )

        context.validation_definitions.add(
            validation_definition
        )

        result = validation_definition.run(
            batch_parameters={"dataframe": data}
        )

        if result is None or not hasattr(result, "success"):
            raise RuntimeError(
                "Great Expectations n'a retourné aucun résultat exploitable."
            )

        failed_controls = []

        for item in result.results:
            if not item.success:
                failed_controls.append({
                    "expectation": item.expectation_config.type,
                    "kwargs": item.expectation_config.kwargs,
                })

        report = {
            "date": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "suite": suite_name,
            "contract_version": CONTRACT_VERSION,
            "great_expectations_version": gx.__version__,
            "snapshot_sha256": snapshot_hash,
            "success": bool(result.success),
            "row_count": len(data),
            "failed_controls": failed_controls,
            "rejected_appids": (
                data["appid"].dropna().astype(str).tolist()
                if not result.success and "appid" in data.columns
                else []
            ),
        }

        save_report(report)

        if not result.success:
            raise ValueError(
                f"Validation refusée avec {suite_name}. "
                f"{len(failed_controls)} contrôle(s) en échec."
            )

        return report

    except ValueError:
        # Les rejets métier sont déjà documentés.
        raise

    except Exception as exc:
        error_report = {
            "date": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "suite": suite_name,
            "contract_version": CONTRACT_VERSION,
            "great_expectations_version": gx.__version__,
            "snapshot_sha256": snapshot_hash,
            "success": False,
            "row_count": len(data),
            "failed_controls": [],
            "rejected_appids": (
                data["appid"].dropna().astype(str).tolist()
                if "appid" in data.columns
                else []
            ),
            "error": str(exc),
        }

        save_report(error_report)

        raise RuntimeError(
            f"Échec technique de la validation : {exc}"
        ) from exc
