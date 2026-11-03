import time
import json
import httpx
import pytest

from normalisation import (
    normaliser_ligne,
    normaliser_prix,
    normaliser_owners
)


@pytest.mark.parametrize(
    "appid, nom_attendu",
    [
        (620, "Portal 2"),
        (400, "Portal"),
        (550, "Left 4 Dead 2"),
    ]
)
def test_normalisation_avec_apis_reelles(appid, nom_attendu):

    with httpx.Client(timeout=20, follow_redirects=True) as client:

        # 1. Steam Store
        store_response = client.get(
            "https://store.steampowered.com/api/appdetails",
            params={"appids": appid, "cc": "us", "l": "english"}
        )
        store_response.raise_for_status()
        steam_store = store_response.json()

        time.sleep(1.1)

        # 2. SteamSpy
        steamspy_response = client.get(
            "https://steamspy.com/api.php",
            params={"request": "appdetails", "appid": appid}
        )
        steamspy_response.raise_for_status()
        steamspy = steamspy_response.json()

        time.sleep(1.1)

        # 3. Résumé des avis Steam
        params = {
            "appid": appid,
            "filter": 0,
            "languages": ["all"],
            "review_type": 0,
            "purchase_type": 1,
            "num_per_page": 1
        }

        reviews_response = client.get(
            "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/",
            params={"input_json": json.dumps(params)}
        )
        reviews_response.raise_for_status()
        reviews = reviews_response.json()

    # Normalisation
    ligne = normaliser_ligne(appid, steam_store, steamspy, reviews)

    assert ligne["appid"] == appid
    assert ligne["name"] == nom_attendu
    assert 0 <= ligne["pct_pos_total"] <= 100
    assert ligne["num_reviews_total"] >= 10

    print(f"\nJeu normalisé : {ligne['name']}")
    print(ligne)


def test_normaliser_prix():

    # Jeu gratuit
    assert normaliser_prix({"is_free": True}) == 0.0

    # Prix absent
    assert normaliser_prix({
        "is_free": False,
        "price_overview": None
    }) is None


def test_normaliser_owners():

    # Tranche valide
    assert normaliser_owners({
        "owners": "5,000,000 .. 10,000,000"
    }) == "5000000 - 10000000"

    # Tranche indisponible
    with pytest.raises(ValueError):
        normaliser_owners({"owners": "999999"})

    # Tranche 0 - 0 exclue
    with pytest.raises(ValueError):
        normaliser_owners({"owners": "0 .. 0"})