"""Aides Prefect (TP4) : tâches des quatre étapes et logs utilisables avec ou sans contexte Prefect.

Premier passage : zéro retry, pas de cache, pas de persistance de résultat (voir partie 2 de l'énoncé).
"""
import json
import logging
import os
from pathlib import Path

from prefect import get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.context import FlowRunContext, TaskRunContext


def step(name, **options):
    """Décorateur de tâche : nom lisible, zéro retry, NO_CACHE, persist_result=False."""
    settings = {"retries": 0, "cache_policy": NO_CACHE, "persist_result": False, **options}
    return task(name=name, **settings)


def logger():
    """Logger Prefect dans un contexte d'exécution ; logger standard sinon (tests, appels directs)."""
    if TaskRunContext.get() is not None or FlowRunContext.get() is not None:
        return get_run_logger()
    return logging.getLogger("steam")


def runtime_ids():
    """Identifiants Prefect disponibles dans le contexte courant (None en dehors d'un flow)."""
    try:
        from prefect.runtime import flow_run, task_run
        return {"flow_run_id": flow_run.id, "task_run_id": task_run.id}
    except Exception:
        return {"flow_run_id": None, "task_run_id": None}


def record_flow_info(mlflow_run_id=None):
    """Écrire `flow_run_id` / `mlflow_run_id` si l'appelant l'a demandé (`STEAM_FLOW_INFO`).

    Le worker PUSH lance le flow dans un sous-processus : ce fichier est le seul canal fiable pour
    persister les identifiants dans le job, y compris quand le flow échoue juste après.
    """
    target = os.getenv("STEAM_FLOW_INFO")
    if not target:
        return
    info = {**runtime_ids(), "mlflow_run_id": mlflow_run_id}
    Path(target).write_text(json.dumps(info), encoding="utf-8")


def annotate_flow_run(**refs):
    """Exposer des références (mode, batch_id, run MLflow, étape en erreur) dans les TAGS du flow run.

    Le modèle de notification d'une automation ne voit pas les variables du processus : seules les
    métadonnées du run lui sont accessibles. Sans contexte Prefect, ou si l'API est injoignable,
    l'annotation est ignorée (elle ne doit jamais faire échouer ni masquer l'échec du pipeline).
    """
    try:
        from prefect.client.orchestration import get_client
        from prefect.runtime import flow_run
        run_id = flow_run.id
        if not run_id:
            return
        with get_client(sync_client=True) as client:
            current = client.read_flow_run(run_id).tags or []  # tags COURANTS côté serveur, pas ceux du démarrage
            existing = [t for t in current if t.split(":", 1)[0] not in refs]
            tags = existing + [f"{key}:{value}" for key, value in refs.items() if value is not None]
            client.update_flow_run(run_id, tags=tags)
    except Exception:
        return
