
import great_expectations as gx


# ============================================================
# VERSION DU CONTRAT
# ============================================================

CONTRACT_VERSION = "1.0.0"


# Schéma brut du dataset Steam
REQUIRED_COLUMNS = [
    "appid",
    "name",
    "release_date",
    "required_age",
    "price",
    "dlc_count",
    "windows",
    "mac",
    "linux",
    "achievements",
    "genres",
    "positive",
    "negative",
    "pct_pos_total",
    "num_reviews_total",
    "estimated_owners",
]


# ============================================================
# CONTEXTE GREAT EXPECTATIONS
# ============================================================

def create_validation_context():
    """Créer le contexte Great Expectations."""
    return gx.get_context()


# ============================================================
# EXPECTATIONS COMMUNES
# ============================================================

def add_base_expectations(suite):
    """Vérifier la structure et l'identité des jeux."""

    for column in REQUIRED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=column)
        )

    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="appid"
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="appid",
            type_="int64"
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="appid",
            min_value=1
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="appid"
        )
    )


# ============================================================
# EXPECTATIONS NUMÉRIQUES
# ============================================================

def add_numeric_expectations(suite, historical=False):
    """Vérifier les valeurs numériques."""

    numeric_columns = [
        "price",
        "dlc_count",
        "achievements",
        "positive",
        "negative",
        "num_reviews_total",
    ]

    for column in numeric_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
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


# ============================================================
# EXPECTATIONS DE TYPES ET DE FORMATS
# ============================================================

def add_format_expectations(suite, strict=False):
    """Vérifier les types et les formats des colonnes."""

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


# ============================================================
# EXPECTATIONS ESTIMATED_OWNERS
# ============================================================

def add_owners_expectations(suite, strict=False):
    """Vérifier le format des tranches de propriétaires."""

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
    """
    Vérification complémentaire :
    les bornes de chaque tranche doivent être cohérentes.
    """

    owners = data["estimated_owners"].dropna()

    for value in owners:
        parts = value.split(" - ")

        if len(parts) != 2:
            raise ValueError(
                f"Format estimated_owners invalide : {value}"
            )

        try:
            minimum = int(parts[0])
            maximum = int(parts[1])
        except ValueError:
            raise ValueError(
                f"Bornes estimated_owners invalides : {value}"
            )

        if minimum < 0 or maximum < 0 or minimum > maximum:
            raise ValueError(
                f"Tranche estimated_owners incohérente : {value}"
            )

        if strict and minimum == 0 and maximum == 0:
            raise ValueError(
                "La tranche 0 - 0 est interdite pour les nouveaux jeux."
            )

    return True


# ============================================================
# EXPECTATIONS DU RAPPORT D'ENTRAÎNEMENT
# ============================================================

def add_training_audit_expectations(suite):
    """Vérifier la structure du rapport d'éligibilité."""

    audit_columns = [
        "task",
        "eligible_count",
        "train_count",
        "test_count",
        "class_count",
    ]

    for column in audit_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(
                column=column
            )
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


# ============================================================
# CRÉATION DES QUATRE SUITES
# ============================================================

def create_expectation_suites(context):
    """Créer les suites de validation du projet."""

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

            add_numeric_expectations(
                suite,
                historical=historical,
            )

            add_format_expectations(
                suite,
                strict=strict,
            )

            add_owners_expectations(
                suite,
                strict=strict,
            )

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
