"""Les tâches Prefect appelées hors d'un flow exigent une API : un serveur temporaire pour la session."""
import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_api():
    with prefect_test_harness():
        yield
