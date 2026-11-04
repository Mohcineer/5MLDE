"""Politique de retry (TP4 partie 4) : 503 puis succès, entrée invalide sans retry, échec définitif."""
import pytest
from prefect import flow

from src.client import PermanentError, TransientError
from src.ingestion import fetch_source


class FlakyClient:
    def __init__(self, failures, error):
        self.failures, self.error, self.calls = failures, error, 0

    def get(self, source, appid):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("HTTP 503")
        return {"ok": True}


def run(client):
    quick = fetch_source.with_options(retry_delay_seconds=[0, 0])

    @flow
    def one():
        return quick(client, "store", 1)
    return one()


def test_503_then_success_is_retried_and_succeeds():
    client = FlakyClient(2, TransientError)
    assert run(client) == {"ok": True}
    assert client.calls == 3  # deux retries = trois tentatives au plus


def test_permanent_error_is_not_retried():
    client = FlakyClient(99, PermanentError)
    with pytest.raises(PermanentError):
        run(client)
    assert client.calls == 1


def test_persistent_transient_error_fails_after_bounded_attempts():
    client = FlakyClient(99, TransientError)
    with pytest.raises(TransientError):
        run(client)
    assert client.calls == 3
