"""Les tâches Prefect appelées hors d'un flow exigent une API.

Serveur temporaire de session si le SDK peut le démarrer ; sinon (version de FastAPI incompatible avec le
serveur embarqué), les tests utilisent le serveur commun défini par `PREFECT_API_URL`.
"""
import contextlib
import os

import pytest

try:
    from prefect.testing.utilities import prefect_test_harness
except Exception:  # ex. TypeError de FastAPI : serveur embarqué inutilisable dans cet environnement
    prefect_test_harness = None


@pytest.fixture(autouse=True, scope="session")
def prefect_api():
    if prefect_test_harness is not None:
        with prefect_test_harness():
            yield
        return
    if not os.getenv("PREFECT_API_URL"):
        pytest.exit("Serveur Prefect de test indisponible : définir PREFECT_API_URL "
                    "(ex. http://localhost:4200/api) et lancer `docker compose up -d prefect`.", returncode=2)
    with contextlib.nullcontext():
        yield
