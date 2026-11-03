
import json
import pytest
import pandas as pd
from unittest.mock import patch

import main

from src.data_validation import validate_dataframe
from test_data_contract import create_valid_game


def test_pipeline_stops_when_ingestion_fails():
    with patch(
        "main.load_data",
        side_effect=ValueError("Validation refusée"),
    ) as mock_load, patch(
        "main.preprocess_data"
    ) as mock_preprocess, patch(
        "main.train_models"
    ) as mock_train, patch(
        "main.evaluate_models"
    ) as mock_evaluate:

        with pytest.raises(ValueError, match="Validation refusée"):
            main.main("config.json")

        mock_load.assert_called_once_with("config.json")
        mock_preprocess.assert_not_called()
        mock_train.assert_not_called()
        mock_evaluate.assert_not_called()


def test_rejection_report_is_generated(tmp_path):
    data = pd.DataFrame([create_valid_game()])
    data.loc[0, "pct_pos_total"] = 120

    with pytest.raises(ValueError, match="Validation refusée"):
        validate_dataframe(
            data=data,
            suite_name="steam_historical_v1",
            report_dir=tmp_path,
            source="test_invalid.csv",
        )

    reports = list(tmp_path.glob("validation_*.json"))

    assert len(reports) == 1

    report = json.loads(reports[0].read_text(encoding="utf-8"))

    assert report["success"] is False
    assert report["source"] == "test_invalid.csv"
    assert len(report["failed_controls"]) > 0

def test_great_expectations_exception_is_blocked_and_reported(
    tmp_path, monkeypatch
):
    def simulate_gx_error():
        raise RuntimeError("Erreur simulée Great Expectations")

    monkeypatch.setattr(
        "src.data_validation.create_validation_context",
        simulate_gx_error,
    )

    data = pd.DataFrame([create_valid_game()])

    with pytest.raises(RuntimeError, match="Erreur simulée"):
        validate_dataframe(
            data=data,
            suite_name="steam_historical_v1",
            report_dir=tmp_path,
            source="test_gx_exception.csv",
        )

    reports = list(tmp_path.glob("validation_*.json"))
    assert len(reports) == 1

    report = json.loads(reports[0].read_text(encoding="utf-8"))

    assert report["success"] is False
    assert "Erreur simulée Great Expectations" in report["error"]

def test_great_expectations_none_result_is_blocked_and_reported(
    tmp_path, monkeypatch
):
    class FakeValidationDefinition:
        def run(self, batch_parameters):
            return None

    monkeypatch.setattr(
        "src.data_validation.create_validation_context",
        lambda: type(
            "FakeContext",
            (),
            {
                "data_sources": type(
                    "FakeDataSources",
                    (),
                    {
                        "add_pandas": lambda self, name: type(
                            "FakeDatasource",
                            (),
                            {
                                "add_dataframe_asset": lambda self, name: type(
                                    "FakeAsset",
                                    (),
                                    {
                                        "add_batch_definition_whole_dataframe":
                                            lambda self, name: object()
                                    },
                                )()
                            },
                        )()
                    },
                )(),
                "suites": type(
                    "FakeSuites",
                    (),
                    {"add": lambda self, suite: None},
                )(),
                "validation_definitions": type(
                    "FakeDefinitions",
                    (),
                    {"add": lambda self, definition: None},
                )(),
            },
        )(),
    )

    monkeypatch.setattr(
        "src.data_validation.gx.ValidationDefinition",
        lambda **kwargs: FakeValidationDefinition(),
    )

    data = pd.DataFrame([create_valid_game()])

    with pytest.raises(RuntimeError, match="aucun résultat exploitable"):
        validate_dataframe(
            data=data,
            suite_name="steam_historical_v1",
            report_dir=tmp_path,
            source="test_none_result.csv",
        )

    reports = list(tmp_path.glob("validation_*.json"))
    assert len(reports) == 1

    report = json.loads(reports[0].read_text(encoding="utf-8"))

    assert report["success"] is False
    assert "error" in report

def test_validation_rejection_preserves_existing_artifacts(tmp_path):
    import hashlib

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    model_path = artifacts_dir / "modele.joblib"
    results_path = artifacts_dir / "resultats.csv"

    model_path.write_bytes(b"modele existant")
    results_path.write_text("accuracy\n0.95\n", encoding="utf-8")

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    model_hash_before = file_hash(model_path)
    results_hash_before = file_hash(results_path)

    data = pd.DataFrame([create_valid_game()])
    data.loc[0, "pct_pos_total"] = 120

    with pytest.raises(ValueError):
        validate_dataframe(
            data=data,
            suite_name="steam_new_games_v1",
            report_dir=tmp_path / "diagnostics",
            source="test_rejection.csv",
        )

    assert model_path.exists()
    assert results_path.exists()

    assert file_hash(model_path) == model_hash_before
    assert file_hash(results_path) == results_hash_before
