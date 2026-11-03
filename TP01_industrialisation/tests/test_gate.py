"""Barrière de validation (Q2/Q3) : blocage avant prétraitement dans PULL et PUSH, effets persistants."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import main
from helpers import make_games, write_config
from src.data_validation import (DataValidationError, DataValidationTechnicalError,
                                 validate_dataframe, validate_snapshot)

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reports(directory):
    return sorted(Path(directory).glob("validation_*.json"))


def setup_old_model(tmp_path):
    old = tmp_path / "runs" / "old"
    old.mkdir(parents=True)
    (old / "regression.joblib").write_bytes(b"ancien modele")
    (old / "metrics.json").write_text("{}")
    return {p: sha(p) for p in old.iterdir()}


def bad_csv(tmp_path, mutate):
    frame = make_games(200, origin="historical-csv")
    mutate(frame)
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("mutate", [
    lambda f: f.loc.__setitem__((1, "appid"), f.loc[0, "appid"]),          # appid dupliqué
    lambda f: f.loc.__setitem__((3, "price"), "gratuit"),                  # prix texte
    lambda f: f.drop(columns=["price"], inplace=True),                     # colonne absente
    lambda f: f.loc.__setitem__((3, "estimated_owners"), "50000 - 20000"),  # bornes inversées
])
def test_pull_rejection_blocks_every_stage_and_keeps_old_models(tmp_path, mutate):
    csv = bad_csv(tmp_path, mutate)
    config = write_config(tmp_path, csv)
    old = setup_old_model(tmp_path)

    with patch("main.preprocess_data") as pre, patch("main.train_models") as train, \
            patch("main.evaluate_models") as evaluate:
        with pytest.raises((DataValidationError, DataValidationTechnicalError)):
            main.main(str(config))
    pre.assert_not_called(); train.assert_not_called(); evaluate.assert_not_called()

    assert {p: sha(p) for p in (tmp_path / "runs" / "old").iterdir()} == old
    assert [p.name for p in (tmp_path / "runs").iterdir()] == ["old"]  # aucun nouveau run
    found = reports(tmp_path / "diag")
    assert found and json.loads(found[-1].read_text())["success"] is False


def test_pull_exit_code_is_non_zero_and_report_exists(tmp_path):
    csv = bad_csv(tmp_path, lambda f: f.loc.__setitem__((3, "price"), "gratuit"))
    config = write_config(tmp_path, csv)
    done = subprocess.run([sys.executable, "main.py", "--config", str(config)],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode != 0
    assert reports(tmp_path / "diag")
    assert not (tmp_path / "runs").exists()


def test_rejection_report_is_readable(tmp_path):
    data = make_games(50, origin="steam-store+steamspy")
    data.loc[2, "pct_pos_total"] = 120
    with pytest.raises(DataValidationError):
        validate_dataframe(data, "steam_new_games_v1", tmp_path, source="essai")
    report = json.loads(reports(tmp_path)[0].read_text())
    assert report["success"] is False and report["suite"] == "steam_new_games_v1"
    assert report["suite_version"] == "v1" and report["snapshot_sha256"] and report["date"]
    control = next(c for c in report["failed_controls"] if c["kwargs"].get("column") == "pct_pos_total")
    assert control["unexpected_count"] == 1 and str(data.loc[2, "appid"]) in control["appids"]
    assert len(report["rejected_appids"]) <= 20  # quelques appids, jamais tout le snapshot


def test_price_text_and_infinity_have_identifiable_cause(tmp_path):
    for value, label in (("gratuit", "price"), (float("inf"), "price")):
        data = make_games(30, origin="steam-store+steamspy")
        data["price"] = data["price"].astype(object)
        data.loc[1, "price"] = value
        with pytest.raises((DataValidationError, DataValidationTechnicalError)):
            validate_dataframe(data, "steam_new_games_v1", tmp_path / label)
        text = reports(tmp_path / label)[-1].read_text()
        assert '"price"' in text


def test_historical_sentinels_accepted_but_not_in_new_games(tmp_path):
    frame = make_games(200, origin="historical-csv")
    frame.loc[0, ["required_age", "pct_pos_total", "num_reviews_total"]] = -1
    frame.loc[1, "estimated_owners"] = "0 - 0"
    assert validate_dataframe(frame, "steam_historical_v1")["success"]

    new = make_games(30, origin="steam-store+steamspy")
    new.loc[0, "pct_pos_total"] = -1
    with pytest.raises(DataValidationError):
        validate_dataframe(new, "steam_new_games_v1")
    new = make_games(30, origin="steam-store+steamspy")
    new.loc[0, "estimated_owners"] = "0 - 0"
    with pytest.raises(DataValidationError):
        validate_dataframe(new, "steam_new_games_v1")


def test_strict_rule_survives_source_mixing(tmp_path):
    """Un nouveau jeu avec sentinelle ne doit pas être excusé par le mélange avec l'historique."""
    snapshot = pd.concat([make_games(200, origin="historical-csv", start_appid=10),
                          make_games(3, origin="steam-store+steamspy")], ignore_index=True)
    snapshot.loc[snapshot.index[-1], "pct_pos_total"] = -1
    with pytest.raises(DataValidationError):
        validate_snapshot(snapshot, {"test_size": 0.2}, report_dir=tmp_path)
    assert list((tmp_path / "rejected_snapshots").glob("*.csv"))  # snapshot rejeté conservé à part


def test_gx_exception_and_missing_result_are_blocking(tmp_path, monkeypatch):
    data = make_games(30)
    monkeypatch.setattr("src.data_validation.create_validation_context",
                        lambda: (_ for _ in ()).throw(RuntimeError("panne GX")))
    with pytest.raises(DataValidationTechnicalError):
        validate_dataframe(data, "steam_new_games_v1", tmp_path)
    assert json.loads(reports(tmp_path)[0].read_text())["error"] == "panne GX"


# --- PUSH : lot de 500 -----------------------------------------------------------------------

def test_push_batch_499_rows_and_single_class_are_rejected(tmp_path):
    with pytest.raises(DataValidationError):
        validate_snapshot(make_games(499), {"test_size": 0.2}, tmp_path, push_batch=True)
    one_class = make_games(500, classes=["0 - 20000"])
    with pytest.raises(DataValidationError) as raised:
        validate_snapshot(one_class, {"test_size": 0.2}, tmp_path, push_batch=True)
    assert "steam_training_audit_v1" in str(raised.value)
    assert validate_snapshot(make_games(500), {"test_size": 0.2}, tmp_path, push_batch=True)


def make_event(i, owners="20000 - 50000"):
    return {"event_id": f"evt-{i}", "source": "synthetic-course",
            "observed_at": "2026-11-03T10:00:00Z",
            "game": {"appid": 8_000_000_000_000 + i, "name": f"Jeu {i}", "price": 9.99,
                     "required_age": 0, "dlc_count": 0, "achievements": 5, "windows": True,
                     "mac": False, "linux": False, "release_date": "2024-01-15",
                     "genres": ["Action"], "pct_pos_total": 70.0 + i % 20,
                     "num_reviews_total": 100 + i,
                     "estimated_owners": owners}}


@pytest.fixture
def push_state(tmp_path, monkeypatch):
    from api_trigger_passif import storage
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "state" / "push.sqlite3")
    return storage


def fill(storage, count, offset=0, owners=None):
    from api_trigger_passif.validation import validate_event
    results = []
    for i in range(offset, offset + count):
        owner = owners or ["0 - 20000", "20000 - 50000", "50000 - 100000", "100000 - 200000"][i % 4]
        results.append(storage.store_event(validate_event(make_event(i, owner))))
    return results


def test_push_single_class_batch_fails_with_report_and_api_keeps_accepting(push_state):
    from api_trigger_passif import worker
    storage = push_state
    old = push_state.state_dir() / "old_model.joblib"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"ancien")
    digest = sha(old)

    fill(storage, 500, owners="20000 - 50000")  # 500 jeux valides mais d'une seule classe
    assert worker.process_pending_batches() == 1  # main.py tourne en sous-processus
    batch = storage.get_batch(1)
    assert batch["status"] == "FAILED" and batch["error"] and batch["report_path"]
    report = json.loads(Path(batch["report_path"]).read_text())
    assert report["success"] is False and report["suite"] == "steam_training_audit_v1"
    assert not (storage.state_dir() / "batches" / "batch_1" / "run").exists()
    assert sha(old) == digest
    assert len(storage.get_status()["batches"]) == 1

    # Le lot invalide reste réservé et n'est pas rejoué ; l'API accepte la suite.
    assert worker.process_pending_batches() == 0
    assert fill(storage, 1, offset=500)[0]["status"] == "accepted"
    assert storage.get_status()["pending_games"] == 1


def test_push_valid_batch_trains_and_restart_reuses_published_run(push_state):
    from api_trigger_passif import training_trigger, worker
    storage = push_state
    fill(storage, 500)
    assert worker.process_pending_batches() == 1
    batch = storage.get_batch(1)
    assert batch["status"] == "SUCCEEDED"
    run = Path(batch["run_dir"])
    assert (run / "regression.joblib").exists() and (run / "classification.joblib").exists()
    assert len(pd.read_csv(run / "snapshot.csv")) == 500
    digest = sha(run / "regression.joblib")

    # Arrêt pendant RUNNING simulé : le lot repasse PENDING ; le run publié est reconnu.
    storage.update_batch_status(1, "RUNNING")
    assert storage.recover_interrupted_batches() == 1
    with patch.object(training_trigger.subprocess, "run") as run_mock:
        assert worker.process_pending_batches() == 1
    run_mock.assert_not_called()  # aucun second entraînement
    assert storage.get_batch(1)["status"] == "SUCCEEDED" and sha(run / "regression.joblib") == digest
