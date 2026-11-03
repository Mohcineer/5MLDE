
import pandas as pd


def create_valid_game():
    """Créer un jeu Steam valide pour les tests."""

    return {
        "appid": 100001,
        "name": "Test Game",
        "release_date": "2025-01-15",
        "required_age": 0,
        "price": 9.99,
        "dlc_count": 0,
        "windows": True,
        "mac": True,
        "linux": False,
        "achievements": 10,
        "genres": "['Action']",
        "positive": 100,
        "negative": 20,
        "pct_pos_total": 83,
        "num_reviews_total": 120,
        "estimated_owners": "20000 - 50000",
    }


def create_test_datasets():
    """Créer les exemples indépendants pour Q1."""

    valid = create_valid_game()

    # Cas 1 : jeu valide
    valid_dataset = pd.DataFrame([valid])

    # Cas 2 : champ facultatif name manquant
    missing_optional = valid.copy()
    missing_optional["name"] = None
    missing_optional_dataset = pd.DataFrame([missing_optional])

    # Cas 3 : appid dupliqué
    duplicate_dataset = pd.DataFrame([
        valid,
        valid.copy(),
    ])

    # Cas 4 : nouveau jeu avec un pourcentage invalide
    invalid_percentage = valid.copy()
    invalid_percentage["pct_pos_total"] = 120
    invalid_percentage_dataset = pd.DataFrame([
        invalid_percentage
    ])

    # Cas 5 : nouveau jeu avec une tranche de propriétaires invalide
    invalid_owners = valid.copy()
    invalid_owners["estimated_owners"] = "0 - 0"
    invalid_owners_dataset = pd.DataFrame([
        invalid_owners
    ])

    return {
        "valid": valid_dataset,
        "missing_optional": missing_optional_dataset,
        "duplicate_appid": duplicate_dataset,
        "invalid_percentage": invalid_percentage_dataset,
        "invalid_owners": invalid_owners_dataset,
    }


if __name__ == "__main__":
    datasets = create_test_datasets()

    for name, dataset in datasets.items():
        print(f"\n{name}")
        print(dataset.to_string(index=False))
