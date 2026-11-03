
import json
import os
import sqlite3
from pathlib import Path

BATCH_SIZE = 500

DB_PATH = Path(
    os.getenv("PUSH_DB_PATH", "artifacts/push.sqlite3")
)


def get_connection():
    """Ouvrir une connexion SQLite et activer les clés étrangères."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    return connection


def init_storage():
    """Créer les tables si elles n'existent pas."""
    with get_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                appid INTEGER NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                batch_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
            );

            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                finished_at TEXT,
                error TEXT
            );
        """)


def store_event(event):
    """
    Enregistrer un événement ou reconnaître un doublon.
    Réserver atomiquement un lot dès que 500 jeux sont disponibles.
    """
    init_storage()

    payload = event.model_dump(mode="json") if hasattr(
        event, "model_dump"
    ) else event

    event_id = payload["event_id"]
    game = payload["game"]
    appid = game["appid"]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    connection = get_connection()

    try:
        connection.execute("BEGIN IMMEDIATE")

        existing_event = connection.execute(
            "SELECT event_id, appid, payload FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()

        if existing_event:
            if existing_event["payload"] == serialized:
                connection.commit()
                return {
                    "status": "duplicate",
                    "batch_id": None,
                    "pending_count": get_pending_count(),
                }

            connection.rollback()
            return {"status": "conflict", "batch_id": None}

        existing_app = connection.execute(
            "SELECT event_id, payload FROM events WHERE appid = ?",
            (appid,),
        ).fetchone()

        if existing_app:
            connection.rollback()
            return {"status": "conflict", "batch_id": None}

        connection.execute(
            """
            INSERT INTO events (event_id, appid, payload)
            VALUES (?, ?, ?)
            """,
            (event_id, appid, serialized),
        )

        pending = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM events
            WHERE batch_id IS NULL
            """
        ).fetchone()["total"]

        batch_id = None

        if pending >= BATCH_SIZE:
            connection.execute(
                "INSERT INTO batches DEFAULT VALUES"
            )
            batch_id = connection.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]

            connection.execute(
                """
                UPDATE events
                SET batch_id = ?
                WHERE id IN (
                    SELECT id
                    FROM events
                    WHERE batch_id IS NULL
                    ORDER BY id
                    LIMIT ?
                )
                """,
                (batch_id, BATCH_SIZE),
            )

        connection.commit()

        return {
            "status": "accepted",
            "batch_id": batch_id,
            "pending_count": get_pending_count(),
        }

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_pending_count():
    """Retourner le nombre de jeux qui attendent encore un lot."""
    with get_connection() as connection:
        result = connection.execute(
            "SELECT COUNT(*) AS total FROM events WHERE batch_id IS NULL"
        ).fetchone()

        return result["total"]


def get_batch(batch_id):
    """Récupérer les 500 événements d'un lot."""
    with get_connection() as connection:
        batch = connection.execute(
            "SELECT * FROM batches WHERE id = ?",
            (batch_id,),
        ).fetchone()

        if batch is None:
            return None

        events = connection.execute(
            """
            SELECT payload
            FROM events
            WHERE batch_id = ?
            ORDER BY id
            """,
            (batch_id,),
        ).fetchall()

        return {
            "batch_id": batch_id,
            "status": batch["status"],
            "events": [json.loads(row["payload"]) for row in events],
            "error": batch["error"],
        }


def update_batch_status(batch_id, status, error=None):
    """Mettre à jour l'état d'un lot."""
    if status not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED"}:
        raise ValueError("Statut de lot invalide")

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE batches
            SET status = ?,
                error = ?,
                started_at = CASE
                    WHEN ? = 'RUNNING' THEN CURRENT_TIMESTAMP
                    ELSE started_at
                END,
                finished_at = CASE
                    WHEN ? IN ('SUCCEEDED', 'FAILED') THEN CURRENT_TIMESTAMP
                    ELSE finished_at
                END
            WHERE id = ?
            """,
            (status, error, status, status, batch_id),
        )


def get_status():
    """Retourner les compteurs et les états des lots."""
    init_storage()

    with get_connection() as connection:
        pending = connection.execute(
            "SELECT COUNT(*) AS total FROM events WHERE batch_id IS NULL"
        ).fetchone()["total"]

        total_events = connection.execute(
            "SELECT COUNT(*) AS total FROM events"
        ).fetchone()["total"]

        batches = connection.execute(
            """
            SELECT id AS batch_id, status, created_at,
                   started_at, finished_at, error
            FROM batches
            ORDER BY id
            """
        ).fetchall()

        return {
            "total_events": total_events,
            "pending_games": pending,
            "batches": [dict(batch) for batch in batches],
        }
