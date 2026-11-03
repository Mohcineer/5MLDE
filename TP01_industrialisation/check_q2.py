
import pandas as pd
from src.data_validation import validate_dataframe

# Jeu de données volontairement simple et valide.
data = pd.DataFrame({
    "appid": [1001],
    "name": ["Jeu test"],
    "release_date": ["2024-01-01"],
    "required_age": [0],
    "price": [9.99],
    "dlc_count": [0],
    "windows": [True],
    "mac": [False],
    "linux": [False],
    "achievements": [10],
    "genres": ["['Action']"],
    "positive": [100],
    "negative": [10],
    "pct_pos_total": [90],
    "num_reviews_total": [110],
    "estimated_owners": ["20000 - 50000"],
})

print("Test 1 : données valides")
report = validate_dataframe(
    data,
    "steam_historical_v1",
    report_dir="diagnostics",
    source="test_q2",
)
print("Résultat :", report["success"])

print("\nTest 2 : prix invalide")
invalid_data = data.copy()
invalid_data.loc[0, "price"] = "gratuit"

try:
    validate_dataframe(
        invalid_data,
        "steam_historical_v1",
        report_dir="diagnostics",
        source="test_q2_invalid",
    )
    print("ERREUR : le prix invalide a été accepté.")
except Exception as error:
    print("Rejet attendu :", error)
