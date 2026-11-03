"""API PUSH : contrat HTTP, idempotence, seuil de 500, concurrence, redémarrage (B1, B2, B4)."""
import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from api_trigger_passif import storage
from api_trigger_passif.main import EXAMPLE_EVENT, create_app
from api_trigger_passif.validation import validate_event


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "push.sqlite3")


@pytest.fixture
def client():
    with TestClient(create_app(start_worker=False)) as test_client:
        yield test_client


def event(i):
    payload = copy.deepcopy(EXAMPLE_EVENT)
    payload["event_id"] = f"evt-{i}"
    payload["game"]["appid"] = 9_000_000_000_000 + i
    payload["game"]["estimated_owners"] = ["0 - 20000", "20000 - 50000"][i % 2]
    return payload


def accepted(client):
    return client.get("/status").json()["accepted_games"]


def test_replaying_same_post_ten_times_adds_one_game(client):
    codes = [client.post("/games", json=EXAMPLE_EVENT).status_code for _ in range(10)]
    assert codes == [202] + [200] * 9
    assert accepted(client) == 1


def test_invalid_payload_gives_422_without_write(client):
    bad = copy.deepcopy(EXAMPLE_EVENT)
    bad["game"]["price"] = "gratuit"
    response = client.post("/games", json=bad)
    assert response.status_code == 422 and "price" in response.json()["detail"]
    assert accepted(client) == 0
    bad["game"]["price"], bad["game"]["pct_pos_total"] = 1, 120
    assert client.post("/games", json=bad).status_code == 422


def test_conflicts_and_identical_game_under_other_event_id(client):
    client.post("/games", json=EXAMPLE_EVENT)
    other_content = copy.deepcopy(EXAMPLE_EVENT)
    other_content["game"]["price"] = 5.0
    assert client.post("/games", json=other_content).status_code == 409  # même event_id
    other_content["event_id"] = "autre"
    assert client.post("/games", json=other_content).status_code == 409  # même appid, valeurs ≠
    same_game = copy.deepcopy(EXAMPLE_EVENT)
    same_game["event_id"] = "autre-id"
    assert client.post("/games", json=same_game).status_code == 200      # même jeu identique
    assert accepted(client) == 1


def test_threshold_499_500_501_and_1000(client):
    for i in range(499):
        assert client.post("/games", json=event(i)).status_code == 202
    status = client.get("/status").json()
    assert status["pending_games"] == 499 and status["batches"] == []

    response = client.post("/games", json=event(499))
    assert response.status_code == 202 and response.json()["batch_id"] == 1
    status = client.get("/status").json()
    assert status["pending_games"] == 0 and len(status["batches"]) == 1

    assert client.post("/games", json=event(500)).json()["batch_id"] is None
    assert client.get("/status").json()["pending_games"] == 1
    assert len(client.get("/status").json()["batches"]) == 1

    for i in range(501, 1000):
        client.post("/games", json=event(i))
    batches = client.get("/status").json()["batches"]
    assert [b["batch_id"] for b in batches] == [1, 2]
    first, second = (storage.get_batch(i, with_events=True) for i in (1, 2))
    ids = [{e["game"]["appid"] for e in b["events"]} for b in (first, second)]
    assert len(ids[0]) == len(ids[1]) == 500 and not ids[0] & ids[1]
    assert client.get("/batches/1").json()["status"] == "PENDING"
    assert client.get("/batches/99").status_code == 404


def test_concurrent_posts_create_disjoint_batches():
    storage.init_storage()
    events = [validate_event(event(i)) for i in range(1000)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(storage.store_event, events))
    assert sum(r["batch_id"] is not None for r in results) == 2
    status = storage.get_status()
    assert status["accepted_games"] == 1000 and status["pending_games"] == 0
    reserved = [storage.get_batch(i)["game_count"] for i in (1, 2)]
    assert reserved == [500, 500]


def test_restart_at_499_still_triggers_next_batch(tmp_path):
    with TestClient(create_app(start_worker=False)) as first:
        for i in range(499):
            first.post("/games", json=event(i))
    storage._initialized.clear()  # simule un nouveau processus
    with TestClient(create_app(start_worker=False)) as second:
        assert second.get("/status").json()["pending_games"] == 499
        assert second.post("/games", json=event(499)).json()["batch_id"] == 1


def test_interrupted_running_batch_is_recoverable(client):
    for i in range(500):
        client.post("/games", json=event(i))
    assert storage.claim_next_batch() == 1
    assert storage.get_batch(1)["status"] == "RUNNING"
    assert storage.claim_next_batch() is None  # un job RUNNING n'est pas réattribué
    assert storage.recover_interrupted_batches() == 1
    assert storage.get_batch(1)["status"] == "PENDING"
    assert threading.active_count() >= 1
