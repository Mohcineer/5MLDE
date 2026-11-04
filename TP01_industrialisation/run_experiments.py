"""Partie 5-6 du TP3 : deux variantes comparables, enregistrement, qualification, tableau.

Prérequis : serveur MLflow joignable (MLFLOW_TRACKING_URI défini).
    uv run python run_experiments.py --config config_pull.json

Les deux variantes partagent snapshot, partitions (même graine) et corpus de validation ;
seuls les hyperparamètres changent. Un `publication_id` stable rend la reprise idempotente.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import registry, tracking
from src.preprocessing import RAW_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "v-a": {"regression": {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 5},
            "classification": {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 5}},
    "v-b": {"regression": {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 3},
            "classification": {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 3}},
}


def run_variant(base_config, label):
    config = json.loads(json.dumps(base_config))
    config["models"] = VARIANTS[label]
    config["validation"] = {"dir": "validation"}
    publication = f"{config.get('tracking', {}).get('publication_prefix', 'pull')}-{label}"
    config["tracking"] = {**config.get("tracking", {}), "enabled": True, "publication_id": publication,
                          "run_name": publication}
    path = ROOT / f"config_{label}.generated.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    client = tracking.client()
    experiment = client.get_experiment_by_name(config["tracking"].get("experiment", tracking.DEFAULT_EXPERIMENT))
    if experiment:  # reprise technique : ne pas réentraîner une publication déjà terminée
        done = client.search_runs([experiment.experiment_id], max_results=1,
                                  filter_string=f"tags.publication_id = '{publication}' and attributes.status = 'FINISHED'")
        if done:
            return done[0].info.run_id, publication
    subprocess.run([sys.executable, "main.py", "--config", str(path)], cwd=ROOT, check=True)
    experiment = client.get_experiment_by_name(config["tracking"].get("experiment", tracking.DEFAULT_EXPERIMENT))
    run = client.search_runs([experiment.experiment_id], max_results=1,
                             filter_string=f"tags.publication_id = '{publication}' and attributes.status = 'FINISHED'",
                             order_by=["attributes.start_time DESC"])[0]
    return run.info.run_id, publication


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_pull.json")
    parser.add_argument("--out", default="comparison.md")
    args = parser.parse_args()
    base = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    tracking.configure(base)
    client = tracking.client()
    example = pd.read_csv(ROOT / "validation" / "validation.csv")[RAW_FEATURE_COLUMNS].head(5)

    rows = []
    for label in VARIANTS:
        run_id, publication = run_variant(base, label)
        for task in registry_tasks():
            version = registry.register(run_id, task, publication, example=example)
            report = registry.qualify(registry.model_name(task), version.version)
            metrics = json.loads(Path(client.download_artifacts(
                run_id, "reports/validation_metrics.json")).read_text(encoding="utf-8"))["results"][task]
            rows.append({"variante": label, "tâche": task, "run_id": run_id,
                         "modèle": registry.model_name(task), "version": version.version,
                         "métrique": metrics.get("mae", metrics.get("f1_macro")),
                         "baseline": metrics.get("baseline_mae", metrics.get("baseline_f1_macro")),
                         "n_validation": metrics["n"], "décision": report["status"],
                         "motif": "; ".join(report["reasons"])})
    table = pd.DataFrame(rows)
    corpus = registry.load_policy()["validation_corpus_sha256"]
    Path(args.out).write_text(
        f"Corpus de validation sha256 : `{corpus}`  \nPolitique : `{registry.load_policy()['policy_id']}`  \n"
        "Régression : MAE (plus bas = mieux). Classification : F1 macro (plus haut = mieux).\n\n"
        + markdown(table), encoding="utf-8")
    print(table.to_string(index=False))


def markdown(table):
    head = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False))


def registry_tasks():
    return ("regression", "classification")


if __name__ == "__main__":
    main()
