
import pytest
import pandas as pd
from src.data_validation import validate_dataframe
from test_data_contract import create_test_datasets


def test_valid_game_is_accepted():
    datasets = create_test_datasets()
    result = validate_dataframe(
        datasets["valid"],
        suite_name="steam_new_games_v1",
    )
    assert result["success"] is True


def test_missing_name_is_accepted_in_historical_data():
    datasets = create_test_datasets()
    result = validate_dataframe(
        datasets["missing_optional"],
        suite_name="steam_historical_v1",
    )
    assert result["success"] is True


def test_duplicate_appid_is_rejected():
    datasets = create_test_datasets()
    with pytest.raises(ValueError):
        validate_dataframe(
            datasets["duplicate_appid"],
            suite_name="steam_historical_v1",
        )


def test_invalid_percentage_is_rejected():
    datasets = create_test_datasets()
    with pytest.raises(ValueError):
        validate_dataframe(
            datasets["invalid_percentage"],
            suite_name="steam_new_games_v1",
        )


def test_invalid_owners_is_rejected():
    datasets = create_test_datasets()
    with pytest.raises(ValueError):
        validate_dataframe(
            datasets["invalid_owners"],
            suite_name="steam_new_games_v1",
        )

def test_missing_required_column_is_rejected():
    datasets = create_test_datasets()
    data = datasets["valid"].drop(columns=["price"])

    with pytest.raises((ValueError, KeyError)):
        validate_dataframe(data, suite_name="steam_historical_v1")


def test_null_appid_is_rejected():
    datasets = create_test_datasets()
    data = datasets["valid"].copy()
    data.loc[0, "appid"] = None

    with pytest.raises((ValueError, TypeError)):
        validate_dataframe(data, suite_name="steam_historical_v1")


def test_invalid_price_text_is_rejected():
    datasets = create_test_datasets()
    data = datasets["valid"].copy()
    data["price"] = "gratuit"

    with pytest.raises(Exception):
        validate_dataframe(data, suite_name="steam_historical_v1")


def test_historical_sentinels_are_accepted():
    datasets = create_test_datasets()
    data = datasets["valid"].copy()

    data["required_age"] = -1
    data["pct_pos_total"] = -1
    data["num_reviews_total"] = -1
    data["estimated_owners"] = "0 - 0"

    result = validate_dataframe(data, suite_name="steam_historical_v1")
    assert result["success"] is True


def test_new_game_sentinels_are_rejected():
    datasets = create_test_datasets()
    data = datasets["valid"].copy()
    data["pct_pos_total"] = -1

    with pytest.raises(ValueError):
        validate_dataframe(data, suite_name="steam_new_games_v1")

def test_push_batch_with_500_rows_is_accepted():
    datasets = create_test_datasets()
    data = pd.concat(
        [datasets["valid"]] * 500,
        ignore_index=True,
    )
    data["appid"] = range(100001, 100501)

    result = validate_dataframe(
        data,
        suite_name="steam_push_batch_v1",
    )

    assert result["success"] is True


def test_push_batch_with_499_rows_is_rejected():
    datasets = create_test_datasets()
    data = pd.concat(
        [datasets["valid"]] * 499,
        ignore_index=True,
    )
    data["appid"] = range(100001, 100500)

    with pytest.raises(ValueError):
        validate_dataframe(
            data,
            suite_name="steam_push_batch_v1",
        )


def test_valid_training_audit_is_accepted():
    data = pd.DataFrame([
        {
            "task": "regression",
            "eligible_count": 100,
            "train_count": 80,
            "test_count": 20,
            "class_count": 0,
        },
        {
            "task": "classification",
            "eligible_count": 100,
            "train_count": 80,
            "test_count": 20,
            "class_count": 12,
        },
    ])

    result = validate_dataframe(
        data,
        suite_name="steam_training_audit_v1",
    )

    assert result["success"] is True


def test_classification_with_one_class_is_rejected():
    data = pd.DataFrame([
        {
            "task": "classification",
            "eligible_count": 100,
            "train_count": 80,
            "test_count": 20,
            "class_count": 1,
        }
    ])

    with pytest.raises(ValueError):
        validate_dataframe(
            data,
            suite_name="steam_training_audit_v1",
        )
