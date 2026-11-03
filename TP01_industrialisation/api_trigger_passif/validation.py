
from datetime import date, datetime
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator


class GamePayload(BaseModel):
    """Données d'un jeu Steam reçu par PUSH."""

    model_config = ConfigDict(extra="forbid")

    appid: StrictInt = Field(gt=0)
    name: str = Field(min_length=1)
    price: float | None
    required_age: StrictInt | None
    dlc_count: StrictInt = Field(ge=0)
    achievements: StrictInt | None

    windows: StrictBool
    mac: StrictBool
    linux: StrictBool

    release_date: str
    genres: list[str]

    pct_pos_total: float = Field(ge=0, le=100)
    num_reviews_total: StrictInt = Field(ge=10)
    estimated_owners: str

    @field_validator("price", mode="before")
    @classmethod
    def validate_price_type(cls, value):
        if value is not None and (
            isinstance(value, (bool, str)) or not isinstance(value, (int, float))
        ):
            raise ValueError("price doit être un nombre ou null")
        return value

    @field_validator("price")
    @classmethod
    def validate_price(cls, value):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("price doit être fini et positif ou nul")
        return value

    @field_validator("pct_pos_total", mode="before")
    @classmethod
    def validate_percentage_type(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("pct_pos_total doit être un nombre")
        if not math.isfinite(value):
            raise ValueError("pct_pos_total doit être fini")
        return value

    @field_validator("release_date")
    @classmethod
    def validate_release_date(cls, value):
        try:
            date.fromisoformat(value)
        except ValueError:
            raise ValueError("release_date doit être une date ISO valide")
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if not value.strip():
            raise ValueError("name ne peut pas être vide")
        return value

    @field_validator("genres")
    @classmethod
    def validate_genres(cls, value):
        if any(not genre.strip() for genre in value):
            raise ValueError("Les genres ne peuvent pas être vides")
        return value

    @field_validator("estimated_owners")
    @classmethod
    def validate_owners(cls, value):
        pattern = r"^(\d+) - (\d+)$"
        match = re.fullmatch(pattern, value)

        if not match:
            raise ValueError(
                "estimated_owners doit respecter le format 'minimum - maximum'"
            )

        minimum, maximum = map(int, match.groups())

        if minimum >= maximum:
            raise ValueError("La borne minimale doit être strictement inférieure à la borne maximale")

        return value


class EventPayload(BaseModel):
    """Enveloppe d'un événement PUSH."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    source: Literal["synthetic-course", "steam-store+steamspy"]
    observed_at: datetime
    game: GamePayload

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value):
        if not value.strip():
            raise ValueError("event_id ne peut pas être vide")
        return value

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at doit contenir un fuseau horaire")
        return value


def validate_event(payload):
    """Valider l'enveloppe et les données d'un jeu."""

    if isinstance(payload, EventPayload):
        return payload

    if not isinstance(payload, dict):
        raise ValueError("L'événement doit être un dictionnaire.")

    try:
        return EventPayload.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Événement PUSH invalide : {exc}") from exc
