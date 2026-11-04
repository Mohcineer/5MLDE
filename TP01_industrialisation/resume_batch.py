"""Reprise contrôlée d'un lot PUSH en échec (TP4 partie 5) : FAILED -> PENDING, puis traitement.

    uv run python resume_batch.py <batch_id>

Jamais automatique. Si le run d'entraînement est déjà publié, il est reconnu par l'empreinte du CSV
et la publication MLflow retrouve ses versions par `publication_id` : aucune version en double.
"""
import sys

from api_trigger_passif import storage, worker


def main(batch_id):
    storage.init_storage()
    before = storage.get_batch(batch_id)
    if before is None:
        raise SystemExit(f"Lot {batch_id} inconnu.")
    if not storage.requeue_failed_batch(batch_id):
        raise SystemExit(f"Lot {batch_id} en état {before['status']} : seule une reprise de FAILED est permise.")
    worker.process_pending_batches()
    after = storage.get_batch(batch_id)
    print({k: after[k] for k in ("batch_id", "status", "attempts", "failure_kind", "flow_run_id",
                                 "mlflow_run_id", "error")})


if __name__ == "__main__":
    main(int(sys.argv[1]))
