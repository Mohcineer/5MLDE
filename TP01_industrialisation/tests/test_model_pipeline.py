"""Comportement du prétraitement, de l'entraînement, de l'évaluation et de la prédiction (TP1 §4)."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import main
from helpers import make_games, write_config
from predict import predict
from src.preprocessing import RAW_FEATURE_COLUMNS, preprocess_data
from src.training import train_models


def prepared(frame, tmp_path, **overrides):
    config = json.loads(write_config(tmp_path, tmp_path / "x.csv", **overrides).read_text())
    return preprocess_data({"data": frame, "config": config,
                            "paths": {"config": tmp_path / "config.json", "runs_dir": tmp_path / "runs"}})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trained")
    frame = make_games(300)
    return frame, train_models(prepared(frame, tmp))


def test_missing_required_column_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Colonnes indispensables absentes"):
        prepared(make_games(100).drop(columns=["price"]), tmp_path)


def test_non_numeric_price_refuses_training(tmp_path):
    frame = make_games(100)
    frame["price"] = frame["price"].astype(object)
    frame.loc[3, "price"] = "gratuit"
    with pytest.raises(ValueError, match="price"):
        prepared(frame, tmp_path)


def test_duplicate_appid_is_blocked(tmp_path):
    frame = pd.concat([make_games(100), make_games(1)], ignore_index=True)
    with pytest.raises(ValueError, match="Doublons"):
        prepared(frame, tmp_path)


def test_no_identifier_shared_between_train_and_test(trained):
    _, models = trained
    for info in models["tasks"].values():
        assert not set(info["train"]["appid"]) & set(info["test"]["appid"])


def test_unseen_genre_can_be_predicted(trained):
    frame, models = trained
    probe = frame.head(3).copy()
    probe["genres"] = "['Genre Jamais Vu']"
    for task in ("regression", "classification"):
        assert len(models["tasks"][task]["model"].predict(probe[RAW_FEATURE_COLUMNS])) == 3


def test_missing_rare_and_unknown_values_keep_dimensions(trained):
    frame, models = trained
    probe = frame.head(4).copy()
    probe.loc[probe.index[0], ["price", "required_age", "achievements"]] = np.nan
    probe.loc[probe.index[1], "release_date"] = "pas une date"
    probe.loc[probe.index[2], "genres"] = "[]"
    for task in ("regression", "classification"):
        model = models["tasks"][task]["model"]
        width = model.named_steps["prep"].transform(
            model.named_steps["features"].transform(frame.head(4)[RAW_FEATURE_COLUMNS])).shape[1]
        out = model.named_steps["prep"].transform(
            model.named_steps["features"].transform(probe[RAW_FEATURE_COLUMNS]))
        assert out.shape == (4, width) and np.isfinite(out).all()


def test_preprocessing_is_fitted_on_train_only(trained):
    _, models = trained
    info = models["tasks"]["regression"]
    imputer = info["model"].named_steps["prep"].named_transformers_["num"]
    features = info["model"].named_steps["features"].transform(info["train"][RAW_FEATURE_COLUMNS])
    train_median = features["log_price"].median()
    assert imputer.statistics_[0] == pytest.approx(train_median)


def test_predictions_identical_after_serialization(trained, tmp_path):
    frame, models = trained
    for task, info in models["tasks"].items():
        path = tmp_path / f"{task}.joblib"
        joblib.dump(info["model"], path)
        reloaded = joblib.load(path)
        sample = info["test"][RAW_FEATURE_COLUMNS]
        before, after = info["model"].predict(sample), reloaded.predict(sample)
        if task == "regression":
            np.testing.assert_allclose(before, after, atol=1e-10)
        else:
            assert (before == after).all()


def test_end_to_end_run_artifacts_and_prediction(tmp_path):
    frame = make_games(300)
    csv = tmp_path / "games.csv"
    frame.to_csv(csv, index=False)
    config = write_config(tmp_path, csv, output={
        "runs_dir": str(tmp_path / "runs"), "run_dir": str(tmp_path / "runs" / "run_a"),
        "diagnostics_dir": str(tmp_path / "diag")})
    result = main.main(str(config))
    run = result["run_dir"]
    assert set(result["artifacts"]) >= {
        "regression.joblib", "classification.joblib", "metrics.json", "split_regression.csv",
        "split_classification.csv", "config.json", "manifest.json"}
    metrics = json.loads((run / "metrics.json").read_text())
    assert {"mae", "baseline_mae", "n_train", "n_test"} <= metrics["regression"].keys()
    assert {"f1_macro", "accuracy", "per_class", "excluded_classes"} <= metrics["classification"].keys()

    probe = frame.head(5).drop(columns=["pct_pos_total", "num_reviews_total", "estimated_owners",
                                        "origin", "name"])
    probe_csv = tmp_path / "probe.csv"
    probe.to_csv(probe_csv, index=False)
    out = predict(run, probe_csv, tmp_path / "pred.csv")
    assert list(out.columns) == ["appid", "pct_positive_pred", "estimated_owners_pred"]
    assert out["pct_positive_pred"].between(0, 100).all()
    assert set(out["estimated_owners_pred"]) <= set(frame["estimated_owners"])

    # Un dossier de run déjà existant est refusé sans modification.
    digest = hashlib.sha256((run / "regression.joblib").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        main.main(str(config))
    assert hashlib.sha256((run / "regression.joblib").read_bytes()).hexdigest() == digest


def test_same_config_gives_same_splits_and_metrics(tmp_path):
    frame = make_games(300)
    csv = tmp_path / "games.csv"
    frame.to_csv(csv, index=False)
    runs = []
    for name in ("a", "b"):
        config = write_config(tmp_path, csv, output={
            "runs_dir": str(tmp_path / "runs"), "run_dir": str(tmp_path / "runs" / name),
            "diagnostics_dir": str(tmp_path / "diag")})
        runs.append(main.main(str(config))["run_dir"])
    for task in ("regression", "classification"):
        a, b = (pd.read_csv(r / f"split_{task}.csv") for r in runs)
        pd.testing.assert_frame_equal(a, b)
    m = [json.loads((r / "metrics.json").read_text()) for r in runs]
    assert m[0]["regression"]["mae"] == pytest.approx(m[1]["regression"]["mae"], abs=1e-10)
    assert m[0]["classification"]["f1_macro"] == pytest.approx(
        m[1]["classification"]["f1_macro"], abs=1e-10)
