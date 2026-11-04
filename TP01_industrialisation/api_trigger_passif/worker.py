"""Worker local : traite les jobs de lots hors de la requête HTTP.

États : PENDING -> RUNNING -> SUCCEEDED | FAILED (cause consultable). Un seul worker pour le socle.
Au démarrage, les lots RUNNING laissés par un arrêt brutal repassent en PENDING ; le déclencheur
reconnaît alors un run déjà publié (empreinte) au lieu de réentraîner.
"""
import threading

from api_trigger_passif import storage, training_trigger


def process_pending_batches():
    """Traiter tous les lots PENDING ; retourne le nombre de lots terminés (succès ou échec)."""
    processed = 0
    while (batch_id := storage.claim_next_batch()) is not None:
        batch = storage.get_batch(batch_id, with_events=True)
        try:
            result = training_trigger.trigger_training(batch)
        except training_trigger.TrainingFailed as exc:
            storage.update_batch_status(
                batch_id, "FAILED", error=str(exc), report_path=exc.report_path,
                flow_run_id=exc.flow_run_id, mlflow_run_id=exc.mlflow_run_id, failure_kind=exc.kind)
        except Exception as exc:  # erreur imprévue : lot FAILED, diagnostic conservé, pas de boucle
            storage.update_batch_status(batch_id, "FAILED", error=f"{type(exc).__name__}: {exc}",
                                        failure_kind="unexpected")
        else:
            storage.update_batch_status(
                batch_id, "SUCCEEDED", run_dir=result["run_dir"],
                snapshot_sha256=result["snapshot_sha256"], flow_run_id=result.get("flow_run_id"),
                mlflow_run_id=result.get("mlflow_run_id"))
        processed += 1
    return processed


class BackgroundWorker:
    """Thread démon réveillé par `notify()` après la réservation d'un lot."""

    def __init__(self, poll_seconds=2.0):
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="push-worker", daemon=True)
        self._poll = poll_seconds

    def start(self):
        storage.init_storage()
        storage.recover_interrupted_batches()
        self._thread.start()
        self._wake.set()  # traiter d'emblée les lots restés PENDING

    def notify(self):
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=10)

    def _loop(self):
        while not self._stop.is_set():
            self._wake.wait(self._poll)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                process_pending_batches()
            except Exception:  # le worker ne doit jamais mourir silencieusement avec l'API
                continue
