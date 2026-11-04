"""Déploiement local du flow « Pipeline Steam » avec `flow.serve` (TP4 partie 6).

    # sans planning : lancer ensuite un run depuis l'interface Prefect (paramètre config_path)
    uv run python deployment.py
    # avec planning toutes les 10 minutes
    uv run python deployment.py --interval-minutes 10

Ce processus reste actif et exécute les runs dans des sous-processus locaux : les trois variables
d'environnement (PREFECT_API_URL, MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_URI) doivent y être définies.
Aucune logique d'entraînement ici : le flow est celui de `main.py`.
"""
import argparse
import subprocess
from datetime import timedelta
from pathlib import Path

from main import main as pipeline

ROOT = Path(__file__).resolve().parent
DEPLOYMENT_NAME = "pipeline-steam-binome"


def code_version():
    """Version liée au code : révision Git courte, avec « -modifie » si l'arbre est modifié."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                        timeout=20).stdout.strip()
        dirty = "-modifie" if run("status", "--porcelain", "--", ".") else ""
        return (run("rev-parse", "--short", "HEAD") or "inconnue") + dirty
    except (OSError, subprocess.SubprocessError):
        return "inconnue"


if __name__ == "__main__":  # indispensable sous Windows (sous-processus)
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-minutes", type=int, default=None,
                        help="planning à intervalle ; absent = aucun planning")
    parser.add_argument("--config", default=str(ROOT / "config_deploy.json"))
    args = parser.parse_args()
    options = {}
    if args.interval_minutes:
        options["interval"] = timedelta(minutes=args.interval_minutes)
    pipeline.serve(
        name=DEPLOYMENT_NAME,
        parameters={"config_path": str(Path(args.config).resolve())},  # chemin absolu
        description="Pipeline Steam (PULL, 4 étapes, suivi MLflow) lancé depuis un déploiement local.",
        version=code_version(),
        tags=["tp4", "pull"],
        limit=1,  # une exécution simultanée au plus sur ce runner
        **options,
    )
