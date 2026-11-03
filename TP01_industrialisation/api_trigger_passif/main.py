"""API d'entrée PUSH : POST /games, GET /status, GET /batches/{id}.

Lancer une seule instance (sans --reload ni plusieurs workers) :
    uv run uvicorn api_trigger_passif.main:create_app --factory --host 127.0.0.1 --port 8000
Le 202 du 500e POST signifie « accepté et lot réservé », pas « modèles entraînés ».
"""
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from api_trigger_passif import storage
from api_trigger_passif.validation import validate_event
from api_trigger_passif.worker import BackgroundWorker

EXAMPLE_EVENT = {
    "event_id": "exemple-manuel-001",
    "source": "synthetic-course",
    "observed_at": "2026-10-03T12:00:00Z",
    "game": {
        "appid": 8000000000001, "name": "Jeu fictif de démonstration", "price": 19.99,
        "required_age": 12, "dlc_count": 2, "achievements": 45,
        "windows": True, "mac": False, "linux": True, "release_date": "2023-05-15",
        "genres": ["Action", "Indie"], "pct_pos_total": 84.5, "num_reviews_total": 200,
        "estimated_owners": "20000 - 50000",
    },
}


def create_app(start_worker=True):
    worker = BackgroundWorker()

    @asynccontextmanager
    async def lifespan(app):
        storage.init_storage()
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="API PUSH Steam", lifespan=lifespan)
    app.state.worker = worker

    @app.post("/games", status_code=202)
    def post_game(payload: dict = Body(..., examples=[EXAMPLE_EVENT])):
        try:
            event = validate_event(payload)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)  # aucune écriture

        result = storage.store_event(event)  # stocké durablement AVANT de répondre
        if result["status"] == "accepted":
            if result["batch_id"] is not None:
                worker.notify()
            return {"status": "accepted", "batch_id": result["batch_id"],
                    "pending_games": result["pending_count"]}
        if result["status"] == "duplicate":
            return JSONResponse({"status": "duplicate", "pending_games": result["pending_count"]},
                                status_code=200)
        return JSONResponse(
            {"status": "conflict", "detail": "event_id ou appid déjà reçu avec un autre contenu."},
            status_code=409)

    @app.get("/status")
    def status():
        return storage.get_status()

    @app.get("/batches/{batch_id}")
    def batch(batch_id: int):
        found = storage.get_batch(batch_id)
        if found is None:
            return JSONResponse({"detail": "Lot inconnu."}, status_code=404)
        return found

    return app
