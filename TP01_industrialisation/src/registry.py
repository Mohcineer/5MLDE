"""Registre MLflow (TP3 parties 6-7) : enregistrement idempotent, qualification, sélection par alias.

La décision (`decide`) est une fonction pure : elle consomme des résultats déjà calculés et ne
réentraîne rien. Les effets MLflow (tags, alias, rapports) sont appliqués séparément.
"""
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import tracking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_FILE = PROJECT_ROOT / "policy.json"


def load_policy(path=POLICY_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_name(task, demo=False):
    return f"steam-{task}" + ("-demo" if demo else "")


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _check(name, ok, observed, expected):
    return {"check": name, "ok": bool(ok), "observed": observed, "expected": expected}


def decide(task, evidence, policy):
    """Décision de qualification depuis des résultats déjà calculés.

    `evidence` : {"workflow", "metrics" (validation commune), "protocol_id", "corpus_sha256",
    "reload_ok"}. Retourne {"status", "reasons", "checks"}. Démo > refus > attente > accepté.
    """
    metrics = evidence.get("metrics") or {}
    rules = policy[task]
    primary, secondary = rules["primary"], rules["secondary"]
    checks, pending, rejected = [], [], []

    if evidence.get("workflow") in policy["demo_workflows"]:
        return {"status": "demo-only", "checks": [],
                "reasons": ["entraîné sur données synthétiques du PUSH : jamais qualifié"]}

    same = (evidence.get("protocol_id") == policy["protocol_id"]
            and evidence.get("corpus_sha256") == policy["validation_corpus_sha256"])
    checks.append(_check("same_protocol", same, [evidence.get("protocol_id"), evidence.get("corpus_sha256")],
                         [policy["protocol_id"], policy["validation_corpus_sha256"]]))
    if not same:
        pending.append("protocole ou corpus de validation incompatible")

    needed = [primary["metric"], secondary["metric"], f"baseline_{primary['metric']}",
              f"baseline_{secondary['metric']}"]
    finite = all(_finite(metrics.get(key)) for key in needed)
    checks.append(_check("metrics_finite", finite, {k: metrics.get(k) for k in needed}, "finies"))
    if not finite:
        pending.append("métrique manquante ou non finie")

    reload_ok = evidence.get("reload_ok")
    checks.append(_check("reload_ok", reload_ok is True, reload_ok, True))
    if reload_ok is not True:
        rejected.append("modèle non rechargeable (ou rechargement non vérifié)")

    if finite:
        value, base = metrics[primary["metric"]], metrics[f"baseline_{primary['metric']}"]
        gain = (base - value) if primary["direction"] == "lower" else (value - base)
        ok_gain = gain >= primary["min_gain_vs_baseline"]
        checks.append(_check(f"{primary['metric']}_gain_vs_baseline", ok_gain, round(gain, 4),
                             f">= {primary['min_gain_vs_baseline']}"))
        if not ok_gain:
            rejected.append(f"gain insuffisant sur la baseline ({primary['metric']})")
        if "max_absolute" in primary:
            ok = value <= primary["max_absolute"]
            checks.append(_check(f"{primary['metric']}_absolute", ok, value, f"<= {primary['max_absolute']}"))
            if not ok:
                rejected.append(f"{primary['metric']} trop élevée pour l'usage visé")
        if "min_absolute" in primary:
            ok = value >= primary["min_absolute"]
            checks.append(_check(f"{primary['metric']}_absolute", ok, value, f">= {primary['min_absolute']}"))
            if not ok:
                rejected.append(f"{primary['metric']} insuffisante pour l'usage visé")
        sec, sec_base = metrics[secondary["metric"]], metrics[f"baseline_{secondary['metric']}"]
        if "min" in secondary:
            ok = sec >= secondary["min"]
            checks.append(_check(f"{secondary['metric']}_min", ok, sec, f">= {secondary['min']}"))
        else:
            ok = (sec - sec_base) >= secondary["min_gain_vs_baseline"]
            checks.append(_check(f"{secondary['metric']}_gain_vs_baseline", ok, round(sec - sec_base, 4),
                                 f">= {secondary['min_gain_vs_baseline']}"))
        if not ok:
            rejected.append(f"métrique secondaire {secondary['metric']} dégradée")
        # une classe effondrée ne doit pas être masquée par la moyenne
        if task == "classification":
            collapsed = [c for c, r in (metrics.get("per_class_recall") or {}).items() if r == 0]
            checks.append(_check("no_collapsed_class", not collapsed, len(collapsed), "0 classe à rappel nul"))
            if collapsed:
                rejected.append(f"{len(collapsed)} classe(s) jamais reconnue(s)")

    if rejected:
        return {"status": "rejected", "reasons": rejected + pending, "checks": checks}
    if pending:
        return {"status": "pending", "reasons": pending, "checks": checks}
    return {"status": "production-ready", "reasons": ["tous les contrôles de la politique sont satisfaits"],
            "checks": checks}


# --- effets MLflow -----------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


def reload_check(model_uri, example):
    """Recharger dans un NOUVEAU processus et prédire (indépendant de l'état du processus parent)."""
    code = ("import sys, json, pandas as pd, mlflow;"
            "m = mlflow.sklearn.load_model(sys.argv[1]);"
            "print(list(m.predict(pd.DataFrame(json.loads(sys.argv[2])))))")
    done = subprocess.run([sys.executable, "-I", "-c", code, model_uri, example.to_json(orient="records")],
                          capture_output=True, text=True, timeout=300)
    return done.returncode == 0


def _exists(client, name):
    return bool(client.search_registered_models(f"name = '{name}'"))


def find_versions(name, publication_id):
    return [v for v in tracking.client().search_model_versions(f"name = '{name}'")
            if v.tags.get("publication_id") == publication_id]


def register(run_id, task, publication_id, demo=False, example=None):
    """Enregistrer le modèle d'un run ; idempotent par `publication_id` (reprise ≠ nouvel essai)."""
    client = tracking.client()
    name = model_name(task, demo)
    if _exists(client, name):
        existing = find_versions(name, publication_id)
        if existing:
            return existing[0]
    else:
        client.create_registered_model(name)
    run = client.get_run(run_id)
    uri = run.data.tags[f"{task}.model_uri"]
    reloaded = reload_check(uri, example) if example is not None else None
    return client.create_model_version(
        name, uri, run_id=run_id,
        tags={"publication_id": publication_id, "run_id": run_id, "task": task,
              "workflow": run.data.tags.get("workflow", ""), "model_uri": uri,
              "reload_ok": "unknown" if reloaded is None else str(reloaded).lower(),
              "qualification": "pending"})


def _evidence(client, version):
    task = version.tags["task"]
    local = client.download_artifacts(version.run_id, "reports/validation_metrics.json")
    report = json.loads(Path(local).read_text(encoding="utf-8"))
    metrics = dict(report["results"][task])
    metrics["per_class_recall"] = {c: v["recall"] for c, v in (metrics.get("per_class") or {}).items()
                                   if isinstance(v, dict) and v.get("support", 0) > 0}
    run = client.get_run(version.run_id)
    return {"workflow": run.data.tags.get("workflow"), "metrics": metrics,
            "protocol_id": report.get("protocol_id"),
            "corpus_sha256": report.get("validation_corpus_sha256"),
            "reload_ok": version.tags.get("reload_ok") == "true"}


def qualify(name, number, policy=None):
    """Évaluer une version, taguer la VERSION (pas seulement le run) et archiver un rapport daté."""
    policy, client = policy or load_policy(), tracking.client()
    version = client.get_model_version(name, str(number))
    try:
        decision = decide(version.tags["task"], _evidence(client, version), policy)
    except Exception as exc:  # rapport absent/illisible : attente motivée, jamais de qualification positive
        decision = {"status": "pending", "checks": [], "reasons": [f"rapport incomplet : {exc}"]}
    stamp = _now()
    report = {"model": name, "version": str(number), "policy_id": policy["policy_id"], "date": stamp,
              **decision}
    safe = stamp.replace(":", "")
    client.log_dict(version.run_id, report, f"qualification/{name}_v{number}_{safe}.json")
    for key, value in {"qualification": decision["status"], "qualification_policy": policy["policy_id"],
                       "qualification_date": stamp,
                       "qualification_reason": "; ".join(decision["reasons"])[:500]}.items():
        client.set_model_version_tag(name, str(number), key, value)
    return report


def select(name, number, alias="champion", actor="unknown", reason="", policy=None):
    """Positionner l'alias seulement si la version est qualifiée selon la politique COURANTE.

    Le tag saisi ne suffit pas : la décision est recalculée depuis les preuves du run.
    """
    policy, client = policy or load_policy(), tracking.client()
    version = client.get_model_version(name, str(number))
    if alias == "demo-current":
        if not name.endswith("-demo"):
            raise PermissionError("l'alias demo-current est réservé aux modèles -demo")
        fresh = {"status": "demo-only", "reasons": ["mécanique de démonstration"]}
    else:
        fresh = decide(version.tags.get("task"), _evidence(client, version), policy)
        if not (fresh["status"] == "production-ready"
                and version.tags.get("qualification") == "production-ready"
                and version.tags.get("qualification_policy") == policy["policy_id"]):
            raise PermissionError(f"{name} v{number} non éligible à '{alias}' : {fresh['reasons']}")
    try:
        previous = client.get_model_version_by_alias(name, alias).version
    except Exception:
        previous = None
    client.set_registered_model_alias(name, alias, str(number))
    entry = {"alias": alias, "previous": previous, "new": str(number), "actor": actor,
             "date": _now(), "reason": reason, "policy_id": policy["policy_id"], "report": fresh}
    client.log_dict(version.run_id, entry,
                    f"selection/{name}_{alias}_{entry['date'].replace(':', '')}.json")
    client.set_registered_model_tag(name, f"selection.{alias}.last", json.dumps(
        {k: entry[k] for k in ("previous", "new", "actor", "date", "reason")}))
    return entry


def rollback(name, alias="champion", actor="unknown"):
    """Revenir à la version précédemment sélectionnée (lue dans l'historique du registre)."""
    last = json.loads(tracking.client().get_registered_model(name).tags[f"selection.{alias}.last"])
    if not last["previous"]:
        raise ValueError("aucune version précédente à restaurer")
    return select(name, last["previous"], alias, actor, f"retour arrière depuis v{last['new']}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Qualifier / sélectionner une version de modèle")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("qualify", "select"):
        p = sub.add_parser(cmd)
        p.add_argument("name")
        p.add_argument("version")
        if cmd == "select":
            p.add_argument("--alias", default="champion")
            p.add_argument("--actor", default="cli")
            p.add_argument("--reason", default="")
    p = sub.add_parser("rollback")
    p.add_argument("name")
    p.add_argument("--alias", default="champion")
    p.add_argument("--actor", default="cli")
    args = parser.parse_args(argv)
    tracking.configure({})
    if args.cmd == "qualify":
        out = qualify(args.name, args.version)
    elif args.cmd == "select":
        out = select(args.name, args.version, args.alias, args.actor, args.reason)
    else:
        out = rollback(args.name, args.alias, args.actor)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
