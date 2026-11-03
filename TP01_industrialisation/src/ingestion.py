
import json
from pathlib import Path

import pandas as pd

from src.data_validation import validate_dataframe


def load_data(config_path):
    """
    Charger la configuration et le CSV Steam.
    Valider les données avant de les transmettre au prétraitement.
    """

    # 1. Lire la configuration
    config_file = Path(config_path).expanduser().resolve()

    if not config_file.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {config_file}"
        )

    with config_file.open("r", encoding="utf-8") as file:
        config = json.load(file)

    # 2. Récupérer les chemins
    base_dir = config_file.parent

    source_config = config.get("source", {})
    output_config = config.get("output", {})

    if source_config.get("type") != "csv":
        raise ValueError("Seule une source de type CSV est supportée.")

    if not source_config.get("path"):
        raise ValueError("Le chemin du fichier CSV est absent de la configuration.")

    def resolve_path(path):
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()

    csv_path = resolve_path(source_config["path"])
    runs_dir = resolve_path(output_config.get("runs_dir", "runs"))
    diagnostics_dir = resolve_path(
        output_config.get("diagnostics_dir", "diagnostics")
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {csv_path}"
        )

    # 3. Charger les données
    data = pd.read_csv(csv_path)

    # 4. Valider les données historiques avant le prétraitement
    report = validate_dataframe(
        data=data,
        suite_name="steam_historical_v1",
        report_dir=diagnostics_dir,
        source=str(csv_path),
    )

    # 5. Retourner les données et la configuration
    return {
        "data": data,
        "config": config,
        "paths": {
            "config": config_file,
            "source": csv_path,
            "runs_dir": runs_dir,
            "diagnostics_dir": diagnostics_dir,
        },
        "validation": report,
    }
