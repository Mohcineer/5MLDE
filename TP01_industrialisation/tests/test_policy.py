"""Décision de qualification : cas accepté, refusés et en attente (résultats de test contrôlés)."""
import copy

from src.registry import decide, load_policy

POLICY = load_policy()


def evidence(task, **override):
    metrics = ({"mae": 10.0, "baseline_mae": 13.9, "r2": 0.2, "baseline_r2": 0.0}
               if task == "regression" else
               {"f1_macro": 0.40, "baseline_f1_macro": 0.07, "accuracy": 0.8, "baseline_accuracy": 0.73,
                "per_class_recall": {"a": 0.5}})
    base = {"workflow": "pull", "metrics": metrics, "protocol_id": POLICY["protocol_id"],
            "corpus_sha256": POLICY["validation_corpus_sha256"], "reload_ok": True}
    return {**base, **override}


def test_accepted_regression():
    assert decide("regression", evidence("regression"), POLICY)["status"] == "production-ready"


def test_accepted_classification():
    assert decide("classification", evidence("classification"), POLICY)["status"] == "production-ready"


def test_rejected_below_baseline_gain():
    ev = evidence("regression")
    ev["metrics"]["mae"] = 13.7
    assert decide("regression", ev, POLICY)["status"] == "rejected"


def test_rejected_good_score_but_not_reloadable():
    assert decide("classification", evidence("classification", reload_ok=False), POLICY)["status"] == "rejected"


def test_rejected_collapsed_class_despite_good_f1():
    ev = evidence("classification")
    ev["metrics"]["per_class_recall"] = {"a": 0.9, "b": 0.0}
    assert decide("classification", ev, POLICY)["status"] == "rejected"


def test_pending_on_non_finite_metric():
    ev = evidence("regression")
    ev["metrics"]["mae"] = float("nan")
    assert decide("regression", ev, POLICY)["status"] == "pending"


def test_pending_on_incompatible_protocol():
    assert decide("regression", evidence("regression", corpus_sha256="x"), POLICY)["status"] == "pending"


def test_push_is_demo_only_whatever_the_scores():
    assert decide("regression", evidence("regression", workflow="push"), POLICY)["status"] == "demo-only"


def test_decision_does_not_mutate_policy():
    before = copy.deepcopy(POLICY)
    decide("regression", evidence("regression"), POLICY)
    assert POLICY == before
