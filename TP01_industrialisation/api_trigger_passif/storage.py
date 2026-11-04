"""Stockage durable SQLite : événements, unicité, réservation atomique des lots, jobs.

Le compteur porte sur les appids distincts valides non encore affectés. L'insertion, le calcul du
nombre de jeux en attente et la réservation des 500 jeux se font dans UNE transaction
`BEGIN IMMEDIATE` : deux POST concurrents ne peuvent pas réserver deux fois les mêmes jeux, ni
lancer deux entraînements pour le même seuil (un `if compteur == 500` hors transaction le pourrait).
"""
import json
import os
import sqlite3
from pathlib import Path

BATCH_SIZE = 500
STATUSES = ("PENDING", "RUNNING", "SUCCEEDED", "FAILED")

DB_PATH = Path(os.getenv("PUSH_DB_PATH", "artifacts/push.sqlite3"))
_initialized = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    finished_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    report_path TEXT,
    run_dir TEXT,
    snapshot_sha256 TEXT,
    flow_run_id TEXT,
    mlflow_run_id TEXT,
    failure_kind TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    appid INTEGER NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    game TEXT NOT NULL,
    batch_id INTEGER REFERENCES batches(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_pending ON events(batch_id) WHERE batch_id IS NULL;
"""


def state_dir():
    return DB_PATH.parent


def get_connection():
    """Connexion sans transaction implicite (les transactions sont explicites)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_storage():
    """Créer les tables une seule fois par base."""
    key = str(DB_PATH.resolve())
    if key in _initialized and DB_PATH.exists():
        return
    connection = get_connection()
    try:
        connection.executescript(SCHEMA)
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(batches)")}
        for column in ("flow_run_id", "mlflow_run_id", "failure_kind"):  # bases créées avant le TP4
            if column not in existing:
                connection.execute(f"ALTER TABLE batches ADD COLUMN {column} TEXT")
    finally:
        connection.close()
    _initialized.add(key)


def _pending_count(connection):
    return connection.execute(
        "SELECT COUNT(*) FROM events WHERE batch_id IS NULL").fetchone()[0]


def store_event(event):
    """Stocker un événement (202), reconnaître un doublon (200) ou signaler un conflit (409)."""
    init_storage()
    payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else event
    event_id, game = payload["event_id"], payload["game"]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    serialized_game = json.dumps(game, ensure_ascii=False, sort_keys=True)

    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")

        by_event = connection.execute(
            "SELECT payload FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if by_event is not None:
            status = "duplicate" if by_event["payload"] == serialized else "conflict"
            result = {"status": status, "batch_id": None, "pending_count": _pending_count(connection)}
            connection.execute("ROLLBACK")
            return result

        by_appid = connection.execute(
            "SELECT game FROM events WHERE appid = ?", (game["appid"],)).fetchone()
        if by_appid is not None:
            # Même jeu identique sous un autre event_id : reconnu ; valeurs différentes : conflit.
            status = "duplicate" if by_appid["game"] == serialized_game else "conflict"
            result = {"status": status, "batch_id": None, "pending_count": _pending_count(connection)}
            connection.execute("ROLLBACK")
            return result

        connection.execute(
            "INSERT INTO events (event_id, appid, payload, game) VALUES (?, ?, ?, ?)",
            (event_id, game["appid"], serialized, serialized_game))

        batch_id = None
        if _pending_count(connection) >= BATCH_SIZE:
            batch_id = connection.execute("INSERT INTO batches DEFAULT VALUES").lastrowid
            connection.execute(
                """UPDATE events SET batch_id = ?
                   WHERE id IN (SELECT id FROM events WHERE batch_id IS NULL ORDER BY id LIMIT ?)""",
                (batch_id, BATCH_SIZE))

        result = {"status": "accepted", "batch_id": batch_id,
                  "pending_count": _pending_count(connection)}
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def get_pending_count():
    init_storage()
    connection = get_connection()
    try:
        return _pending_count(connection)
    finally:
        connection.close()


def _batch_row(row):
    return {key: row[key] for key in row.keys()}


def get_batch(batch_id, with_events=False):
    """État d'un lot ; les 500 événements ne sont chargés que si demandé (worker)."""
    init_storage()
    connection = get_connection()
    try:
        row = connection.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        batch = {**_batch_row(row), "batch_id": row["id"]}
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE batch_id = ?", (batch_id,)).fetchone()[0]
        batch["game_count"] = count
        if with_events:
            rows = connection.execute(
                "SELECT payload FROM events WHERE batch_id = ? ORDER BY id", (batch_id,)).fetchall()
            batch["events"] = [json.loads(r["payload"]) for r in rows]
        return batch
    finally:
        connection.close()


def claim_next_batch():
    """Passer le plus ancien lot PENDING en RUNNING (une seule fois, même en concurrence)."""
    init_storage()
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT id FROM batches WHERE status = 'PENDING' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        connection.execute(
            """UPDATE batches SET status = 'RUNNING', attempts = attempts + 1,
                   started_at = CURRENT_TIMESTAMP, error = NULL WHERE id = ?""", (row["id"],))
        connection.execute("COMMIT")
        return row["id"]
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def update_batch_status(batch_id, status, error=None, report_path=None, run_dir=None,
                        snapshot_sha256=None, flow_run_id=None, mlflow_run_id=None,
                        failure_kind=None):
    if status not in STATUSES:
        raise ValueError("Statut de lot invalide")
    init_storage()
    connection = get_connection()
    try:
        connection.execute(
            """UPDATE batches SET status = ?, error = ?,
                   report_path = COALESCE(?, report_path), run_dir = COALESCE(?, run_dir),
                   snapshot_sha256 = COALESCE(?, snapshot_sha256),
                   flow_run_id = COALESCE(?, flow_run_id),
                   mlflow_run_id = COALESCE(?, mlflow_run_id),
                   failure_kind = ?,
                   finished_at = CASE WHEN ? IN ('SUCCEEDED', 'FAILED')
                                      THEN CURRENT_TIMESTAMP ELSE finished_at END
               WHERE id = ?""",
            (status, error, report_path, run_dir, snapshot_sha256, flow_run_id, mlflow_run_id,
             failure_kind, status, batch_id))
    finally:
        connection.close()


def requeue_failed_batch(batch_id):
    """Reprise contrôlée, demandée explicitement : FAILED -> PENDING (jamais automatique).

    Les jeux restent ceux du lot ; le déclencheur reconnaît un run déjà publié par son empreinte et
    la publication MLflow retrouve ses versions par `publication_id` (aucun doublon).
    """
    init_storage()
    connection = get_connection()
    try:
        cursor = connection.execute(
            "UPDATE batches SET status = 'PENDING' WHERE id = ? AND status = 'FAILED'", (batch_id,))
        return cursor.rowcount == 1
    finally:
        connection.close()


def recover_interrupted_batches():
    """Reprise après arrêt : un lot RUNNING (processus interrompu) redevient PENDING.

    Les jeux restent réservés à leur lot (aucun mélange). Le worker retrouvera un éventuel run
    déjà publié par son empreinte avant de réentraîner. Les lots FAILED ne sont jamais rejoués
    automatiquement : une non-conformité inchangée ne se corrige pas par une boucle de retry.
    """
    init_storage()
    connection = get_connection()
    try:
        cursor = connection.execute(
            "UPDATE batches SET status = 'PENDING' WHERE status = 'RUNNING'")
        return cursor.rowcount
    finally:
        connection.close()


def get_status():
    """Jeux distincts acceptés, jeux en attente, seuil et états des lots."""
    init_storage()
    connection = get_connection()
    try:
        total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        batches = connection.execute(
            """SELECT id AS batch_id, status, attempts, created_at, started_at, finished_at,
                      error, run_dir, flow_run_id, mlflow_run_id, failure_kind
               FROM batches ORDER BY id""").fetchall()
        return {
            "accepted_games": total,
            "pending_games": _pending_count(connection),
            "threshold": BATCH_SIZE,
            "batches": [dict(b) for b in batches],
        }
    finally:
        connection.close()
