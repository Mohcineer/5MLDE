"""Jeux de données et réponses d'API SYNTHÉTIQUES pour les tests (aucun réseau, aucun vrai jeu)."""
import json

import httpx
import numpy as np
import pandas as pd

OWNER_CLASSES = ["0 - 20000", "20000 - 50000", "50000 - 100000", "100000 - 200000"]
GENRES = ["Action", "Indie", "Adventure", "RPG"]


def make_games(n=200, seed=0, start_appid=1_000_001, origin="synthetic-course", classes=None):
    """Snapshot conforme (nouveaux jeux) : 4 classes équilibrées par défaut."""
    rng = np.random.default_rng(seed)
    classes = classes or OWNER_CLASSES
    rows = []
    for i in range(n):
        rows.append({
            "appid": start_appid + i,
            "name": f"Jeu synthétique {start_appid + i}",
            "release_date": f"20{10 + i % 15:02d}-{1 + i % 12:02d}-{1 + i % 28:02d}",
            "required_age": int(rng.choice([0, 12, 16, 18])),
            "price": round(float(rng.uniform(0, 60)), 2),
            "dlc_count": int(rng.integers(0, 5)),
            "windows": True,
            "mac": bool(i % 2),
            "linux": bool(i % 3 == 0),
            "achievements": int(rng.integers(0, 80)),
            "genres": str([GENRES[i % 4], GENRES[(i + 1) % 4]][: 1 + i % 2]),
            "pct_pos_total": round(float(rng.uniform(20, 99)), 1),
            "num_reviews_total": int(rng.integers(10, 5000)),
            "estimated_owners": classes[i % len(classes)],
            "origin": origin,
        })
    return pd.DataFrame(rows)


def write_config(tmp_path, csv_path, **overrides):
    config = {
        "source": {"type": "csv", "path": str(csv_path)},
        "output": {"runs_dir": str(tmp_path / "runs"), "diagnostics_dir": str(tmp_path / "diag")},
        "random_state": 42, "test_size": 0.2,
        "thresholds": {"regression_min_reviews": 10, "classification_min_class_count": 5},
        "models": {"regression": {"n_estimators": 5}, "classification": {"n_estimators": 5}},
    }
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


# --- réponses d'API synthétiques, conformes au contrat décrit dans l'énoncé --------------------

def fake_store(appid, name="Jeu fictif", price_cents=1999, is_free=False, coming_soon=False,
               success=True):
    data = {
        "steam_appid": appid, "name": name, "type": "game", "is_free": is_free,
        "required_age": 0, "dlc": [1, 2], "achievements": {"total": 12},
        "platforms": {"windows": True, "mac": False, "linux": True},
        "release_date": {"coming_soon": coming_soon, "date": "May 15, 2023"},
        "genres": [{"id": "1", "description": "Action"}, {"id": "23", "description": "Indie"}],
    }
    if price_cents is not None and not is_free:
        data["price_overview"] = {"currency": "USD", "final": price_cents}
    entry = {"success": success}
    if success:
        entry["data"] = data
    return {str(appid): entry}


def fake_spy(appid, name="Jeu fictif", owners="20,000 .. 50,000"):
    return {"appid": appid, "name": name, "owners": owners}


def fake_reviews(positive=80, negative=20):
    return {"success": 1, "query_summary": {
        "total_positive": positive, "total_negative": negative,
        "total_reviews": positive + negative, "review_score": 8}}


class FakeSteam:
    """Transport httpx simulé ; `script[(source, appid)]` = liste de réponses consommées en ordre."""

    def __init__(self, appids=(), script=None):
        self.calls = []
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.appids = list(appids)

    @staticmethod
    def source_of(request):
        host = request.url.host
        if host == "steamspy.com":
            return "spy", int(request.url.params["appid"])
        if "input_json" in request.url.params:
            return "reviews", int(json.loads(request.url.params["input_json"])["appid"])
        return "store", int(request.url.params["appids"])

    def default(self, source, appid):
        body = {"store": fake_store, "spy": fake_spy}.get(source, lambda a: fake_reviews())(appid)
        return httpx.Response(200, json=body)

    def __call__(self, request):
        source, appid = self.source_of(request)
        self.calls.append((source, appid))
        queue = self.script.get((source, appid))
        if queue:
            item = queue.pop(0)
            return item(request) if callable(item) else item
        return self.default(source, appid)

    def transport(self):
        return httpx.MockTransport(self)
