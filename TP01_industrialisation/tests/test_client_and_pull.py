"""Client HTTP (A2), normalisation hors ligne (A1), upsert (A3) et incidents d'acquisition (A4).

Toutes les réponses sont SYNTHÉTIQUES (construites d'après le contrat de l'énoncé), pas réelles.
"""
import hashlib
import json

import httpx
import pandas as pd
import pytest

import main
from helpers import FakeSteam, fake_reviews, fake_spy, fake_store, make_games, write_config
from src import ingestion
from src.client import (AcquisitionError, PermanentError, RetryLaterError, SteamClient,
                        TransientError)
from src.normalisation import normaliser_ligne


def make_client(tmp_path, steam, **kwargs):
    sleeps = []
    client = SteamClient(tmp_path / "cache", transport=steam.transport(), sleep=sleeps.append,
                         **kwargs)
    return client, sleeps


# --- A1 : normalisation -------------------------------------------------------------------------

def test_normalise_regular_free_and_missing_price_games():
    row = normaliser_ligne(620, fake_store(620), fake_spy(620), fake_reviews(80, 20))
    assert row["price"] == 19.99 and row["pct_pos_total"] == 80.0
    assert row["estimated_owners"] == "20000 - 50000" and row["release_date"] == "2023-05-15"
    assert row["genres"] == "['Action', 'Indie']"
    free = normaliser_ligne(400, fake_store(400, is_free=True), fake_spy(400), fake_reviews())
    assert free["price"] == 0.0
    no_price = normaliser_ligne(550, fake_store(550, price_cents=None), fake_spy(550), fake_reviews())
    assert no_price["price"] is None


def test_unavailable_label_is_rejected_not_invented():
    with pytest.raises(ValueError):
        normaliser_ligne(620, fake_store(620), fake_spy(620, owners="999999"), fake_reviews())
    with pytest.raises(ValueError, match="10 avis"):
        normaliser_ligne(620, fake_store(620), fake_spy(620), fake_reviews(5, 3))
    with pytest.raises(ValueError):
        normaliser_ligne(620, fake_store(620), fake_spy(620, owners="0 .. 0"), fake_reviews())


# --- A2 : client --------------------------------------------------------------------------------

def test_429_then_success_respects_retry_after(tmp_path):
    steam = FakeSteam(script={("spy", 620): [httpx.Response(429, headers={"Retry-After": "5"})]})
    client, sleeps = make_client(tmp_path, steam)
    assert client.get("spy", 620)["owners"] == "20,000 .. 50,000"
    assert 5.0 in sleeps and client.http_calls == 2


def test_three_503_fail_after_three_attempts(tmp_path):
    steam = FakeSteam(script={("store", 620): [httpx.Response(503)] * 3})
    client, _ = make_client(tmp_path, steam)
    with pytest.raises(TransientError):
        client.get("store", 620)
    assert client.http_calls == 3


def test_permanent_errors_are_not_retried(tmp_path):
    steam = FakeSteam(script={
        ("store", 1): [httpx.Response(404)],
        ("store", 2): [httpx.Response(200, text="<html>pas du json</html>")],
        ("store", 3): [httpx.Response(200, json={"3": {"success": False}})],
        ("reviews", 4): [httpx.Response(200, json={"success": 1})],
    })
    client, _ = make_client(tmp_path, steam)
    for source, appid in [("store", 1), ("store", 2), ("store", 3), ("reviews", 4)]:
        before = client.http_calls
        with pytest.raises(PermanentError):
            client.get(source, appid)
        assert client.http_calls == before + 1


def test_long_retry_after_interrupts_for_later(tmp_path):
    steam = FakeSteam(script={("spy", 620): [httpx.Response(429, headers={"Retry-After": "120"})]})
    client, _ = make_client(tmp_path, steam)
    with pytest.raises(RetryLaterError):
        client.get("spy", 620)


def test_calls_are_spaced_and_cache_avoids_network(tmp_path):
    steam = FakeSteam()
    ticks = iter(range(0, 1000))
    client, sleeps = make_client(tmp_path, steam, clock=lambda: next(ticks) * 0.1)
    for source in ("store", "spy", "reviews"):
        client.get(source, 620)
    assert sleeps and all(s > 0 for s in sleeps)
    calls = client.http_calls
    fresh = SteamClient(tmp_path / "cache", transport=steam.transport())  # nouvelle exécution
    for source in ("store", "spy", "reviews"):
        fresh.get(source, 620)
    assert fresh.http_calls == 0 and calls == 3
    record = json.loads((tmp_path / "cache" / "store_620.json").read_text())
    assert record["url"].startswith("https://") and record["fetched_at"]
    reviews = json.loads((tmp_path / "cache" / "reviews_620.json").read_text())["response"]
    assert set(reviews) == {"success", "query_summary"}  # aucun texte d'avis conservé


def test_refresh_renews_store_but_not_recent_spy(tmp_path):
    steam = FakeSteam()
    client, _ = make_client(tmp_path, steam)
    client.get("store", 620)
    client.get("spy", 620)
    refreshing, _ = make_client(tmp_path, steam, refresh=True)
    refreshing.get("store", 620)
    refreshing.get("spy", 620)
    assert refreshing.http_calls == 1  # SteamSpy < 24 h : cache conservé


# --- A3 : upsert --------------------------------------------------------------------------------

def pull_config(tmp_path, base_csv, appids, run_name):
    return write_config(
        tmp_path, base_csv, source={"type": "api", "path": str(base_csv)},
        output={"runs_dir": str(tmp_path / "runs"), "run_dir": str(tmp_path / "runs" / run_name),
                "diagnostics_dir": str(tmp_path / "diag")},
        acquisition={"appids": appids, "cache_dir": str(tmp_path / "cache"), "refresh": False})


@pytest.fixture
def fake_steam(monkeypatch):
    steam = FakeSteam()

    class Client(SteamClient):
        def __init__(self, cache_dir, refresh=False):
            super().__init__(cache_dir, refresh=refresh, transport=steam.transport(),
                             sleep=lambda s: None)

    monkeypatch.setattr(ingestion, "SteamClient", Client)
    return steam


@pytest.fixture
def base_csv(tmp_path):
    frame = make_games(200, origin="historical-csv", start_appid=10)
    path = tmp_path / "base.csv"
    frame.to_csv(path, index=False)
    return path


def test_add_then_update_without_duplicates(tmp_path, base_csv, fake_steam):
    first = ingestion.load_data(str(pull_config(tmp_path, base_csv, [620, 400, 550], "r1")))
    assert first["provenance"]["added"] == [400, 550, 620] and first["provenance"]["updated"] == []
    assert len(first["data"]) == 203 and not first["data"]["appid"].duplicated().any()
    assert (first["data"]["origin"] == "steam-store+steamspy").sum() == 3
    assert all(r["success"] for r in first["validation"])

    snapshot = tmp_path / "snapshot1.csv"
    first["data"].to_csv(snapshot, index=False)
    calls_before = len(fake_steam.calls)
    second = ingestion.load_data(str(pull_config(tmp_path, snapshot, [620, 400, 550], "r2")))
    assert second["provenance"]["updated"] == [400, 550, 620] and second["provenance"]["added"] == []
    assert len(second["data"]) == 203
    assert len(fake_steam.calls) == calls_before  # tout vient du cache
    assert second["provenance"]["http_calls"] == 0


def test_pull_end_to_end_publishes_two_models(tmp_path, base_csv, fake_steam):
    result = main.main(str(pull_config(tmp_path, base_csv, [620], "r1")))
    assert (result["run_dir"] / "regression.joblib").exists()
    assert (result["run_dir"] / "snapshot.csv").exists()
    manifest = json.loads((result["run_dir"] / "manifest.json").read_text())
    assert manifest["provenance"]["added"] == [620]
    assert manifest["provenance"]["sources"][0]["store"]["url"].startswith("https://")


def test_collection_is_limited_to_ten_games(tmp_path, base_csv, fake_steam):
    with pytest.raises(ValueError):
        ingestion.load_data(str(pull_config(tmp_path, base_csv, list(range(1, 12)), "r1")))


# --- A4 : incident d'acquisition ----------------------------------------------------------------

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("script", [
    {("spy", 620): [httpx.Response(503)] * 3},
    {("spy", 620): [httpx.Response(200, json=fake_spy(620, owners="0 .. 0"))]},
    {("store", 620): [httpx.Response(200, json=fake_store(620, success=False))]},
])
def test_acquisition_incident_publishes_nothing(tmp_path, base_csv, monkeypatch, script):
    steam = FakeSteam(script=script)

    class Client(SteamClient):
        def __init__(self, cache_dir, refresh=False):
            super().__init__(cache_dir, refresh=refresh, transport=steam.transport(),
                             sleep=lambda s: None)

    monkeypatch.setattr(ingestion, "SteamClient", Client)
    old = tmp_path / "runs" / "old"
    old.mkdir(parents=True)
    (old / "regression.joblib").write_bytes(b"ancien modele")
    before = sha(old / "regression.joblib")

    with pytest.raises(AcquisitionError):
        main.main(str(pull_config(tmp_path, base_csv, [620], "nouveau")))

    assert not (tmp_path / "runs" / "nouveau").exists()  # aucun modèle publié
    assert sha(old / "regression.joblib") == before
    assert [p.name for p in (tmp_path / "runs").iterdir() if p.name.startswith(".")] == []


def test_429_then_success_still_trains(tmp_path, base_csv, monkeypatch):
    steam = FakeSteam(script={("store", 620): [httpx.Response(429, headers={"Retry-After": "1"})]})

    class Client(SteamClient):
        def __init__(self, cache_dir, refresh=False):
            super().__init__(cache_dir, refresh=refresh, transport=steam.transport(),
                             sleep=lambda s: None)

    monkeypatch.setattr(ingestion, "SteamClient", Client)
    result = main.main(str(pull_config(tmp_path, base_csv, [620], "r1")))
    assert (result["run_dir"] / "classification.joblib").exists()


def test_pandas_frames_unchanged_by_failed_upsert_inputs(base_csv):
    base = pd.read_csv(base_csv)
    snapshot, changes = ingestion.upsert_snapshot(base, ingestion.normalize_new_rows(
        [normaliser_ligne(620, fake_store(620), fake_spy(620), fake_reviews())]))
    assert changes == {"added": [620], "updated": []} and len(base) == 200 and len(snapshot) == 201
