"""Clé de cache des partitions : stable pour un même contenu, invalidée par contenu, paramètre, version, code."""
import pandas as pd

from src import preprocessing
from src.preprocessing import partition_cache_key

CONFIG = {"thresholds": {"regression_min_reviews": 10}, "test_size": 0.2, "random_state": 42}


def frame():
    return pd.DataFrame({"appid": [1, 2, 3], "price": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})


def test_same_content_same_key_even_with_different_index():
    assert partition_cache_key(frame(), CONFIG) == partition_cache_key(frame().set_index([[9, 8, 7]]), CONFIG)


def test_content_change_under_same_columns_changes_key():
    other = frame()
    other.loc[0, "price"] = 9.0
    assert partition_cache_key(other, CONFIG) != partition_cache_key(frame(), CONFIG)


def test_parameter_change_changes_key():
    assert partition_cache_key(frame(), {**CONFIG, "test_size": 0.3}) != partition_cache_key(frame(), CONFIG)
    assert partition_cache_key(frame(), {**CONFIG, "random_state": 1}) != partition_cache_key(frame(), CONFIG)


def test_preprocessing_version_change_changes_key(monkeypatch):
    before = partition_cache_key(frame(), CONFIG)
    monkeypatch.setattr(preprocessing, "PREPROCESSING_VERSION", "features-v2")
    assert partition_cache_key(frame(), CONFIG) != before


def test_key_does_not_depend_on_flow_run():
    # aucune date ni identifiant d'exécution : deux appels successifs donnent la même clé
    assert partition_cache_key(frame(), CONFIG) == partition_cache_key(frame(), CONFIG)
