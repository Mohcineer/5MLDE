
import pytest

from api_trigger_passif.validation import validate_event


def valid_event():
    return {
        "event_id": "event-001",
        "source": "synthetic-course",
        "observed_at": "2026-11-03T10:00:00Z",
        "game": {
            "appid": 100001,
            "name": "Test Game",
            "price": 9.99,
            "required_age": 0,
            "dlc_count": 0,
            "achievements": 10,
            "windows": True,
            "mac": True,
            "linux": False,
            "release_date": "2025-01-15",
            "genres": ["Action"],
            "pct_pos_total": 83.0,
            "num_reviews_total": 120,
            "estimated_owners": "20000 - 50000",
        },
    }


def test_valid_event_is_accepted():
    result = validate_event(valid_event())
    assert result.event_id == "event-001"
    assert result.game.appid == 100001


def test_nullable_fields_are_accepted():
    event = valid_event()
    event["game"]["price"] = None
    event["game"]["required_age"] = None
    event["game"]["achievements"] = None
    assert validate_event(event)


def test_invalid_appid_is_rejected():
    event = valid_event()
    event["game"]["appid"] = -1
    with pytest.raises(ValueError):
        validate_event(event)


def test_invalid_price_is_rejected():
    event = valid_event()
    event["game"]["price"] = -5
    with pytest.raises(ValueError):
        validate_event(event)


def test_invalid_owners_range_is_rejected():
    event = valid_event()
    event["game"]["estimated_owners"] = "50000 - 20000"
    with pytest.raises(ValueError):
        validate_event(event)


def test_unknown_field_is_rejected():
    event = valid_event()
    event["unexpected"] = "value"
    with pytest.raises(ValueError):
        validate_event(event)


def test_missing_timezone_is_rejected():
    event = valid_event()
    event["observed_at"] = "2026-11-03T10:00:00"
    with pytest.raises(ValueError):
        validate_event(event)


def test_invalid_source_is_rejected():
    event = valid_event()
    event["source"] = "unknown"
    with pytest.raises(ValueError):
        validate_event(event)
