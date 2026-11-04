"""Vérifier dans un NOUVEAU processus qu'un alias et un numéro de version donnent le même modèle.

    uv run python verify_reload.py steam-regression [--alias champion]

Compare `models:/<nom>@<alias>` et `models:/<nom>/<version concrète>` sur quelques jeux, plus un
genre inconnu et une valeur manquante autorisée. Aucun objet n'est hérité d'un processus précédent.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import tracking
from src.preprocessing import RAW_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    tracking.configure({})
    import mlflow

    client = tracking.client()
    version = client.get_model_version_by_alias(args.name, args.alias)
    print(f"alias '{args.alias}' -> version concrète {version.version} (run {version.run_id})")
    print("tags de version :", json.dumps(dict(version.tags), ensure_ascii=False, indent=2))

    by_alias = mlflow.sklearn.load_model(f"models:/{args.name}@{args.alias}")
    by_number = mlflow.sklearn.load_model(f"models:/{args.name}/{version.version}")

    sample = pd.read_csv(ROOT / "validation" / "validation.csv")[RAW_FEATURE_COLUMNS].head(10)
    edge = sample.head(2).copy()
    edge.loc[edge.index[0], "genres"] = "['GenreInconnuXYZ']"   # genre jamais vu
    edge.loc[edge.index[1], "dlc_count"] = np.nan              # valeur manquante autorisée
    frame = pd.concat([sample, edge], ignore_index=True)

    a, b = by_alias.predict(frame), by_number.predict(frame)
    same = np.allclose(a, b, atol=1e-9) if np.issubdtype(np.asarray(a).dtype, np.number) else list(a) == list(b)
    print(f"{len(frame)} jeux (dont genre inconnu et valeur manquante) -> prédictions identiques : {same}")
    print("exemple :", list(a[:3]))
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
