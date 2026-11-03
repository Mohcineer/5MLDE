
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement du dataset historique
df = pd.read_csv("data/games.csv")

print("===== 1. DONNÉES =====")
print("Nombre de lignes :", len(df))

# Vérification des tranches estimated_owners
print("\n===== 2. ESTIMATED_OWNERS =====")

invalid_format = []
invalid_ranges = []

for value in df["estimated_owners"].dropna().unique():
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", str(value))

    if not match:
        invalid_format.append(value)
        continue

    minimum = int(match.group(1))
    maximum = int(match.group(2))

    if minimum > maximum:
        invalid_ranges.append(value)

print("Formats invalides :", len(invalid_format))
print("Bornes incohérentes :", len(invalid_ranges))
print("Sentinelles 0 - 0 :", (df["estimated_owners"] == "0 - 0").sum())

# Régression : filtres exacts du TP1
print("\n===== 3. RÉGRESSION =====")

regression = df[
    (df["pct_pos_total"] >= 0)
    & (df["pct_pos_total"] <= 100)
    & (df["num_reviews_total"] >= 10)
].copy()

print("Jeux admissibles :", len(regression))

try:
    train, test = train_test_split(
        regression,
        test_size=0.2
    )
    print("Split régression : OK")
    print("Train :", len(train))
    print("Test :", len(test))
except ValueError as error:
    print("Split régression impossible :", error)

# Classification : filtres exacts du TP1
print("\n===== 4. CLASSIFICATION =====")

classification = df[
    df["estimated_owners"] != "0 - 0"
].copy()

value_counts = classification["estimated_owners"].value_counts()

classification = classification[
    classification["estimated_owners"].isin(
        value_counts[value_counts >= 5].index
    )
].copy()

print("Jeux admissibles :", len(classification))
print("Nombre de classes :", classification["estimated_owners"].nunique())
print("Effectif minimal par classe :",
      classification["estimated_owners"].value_counts().min())

try:
    train, test = train_test_split(
        classification,
        test_size=0.2,
        stratify=classification["estimated_owners"]
    )
    print("Split classification : OK")
    print("Train :", len(train))
    print("Test :", len(test))
except ValueError as error:
    print("Split classification impossible :", error)

print("\n===== FIN DU CONTRÔLE Q1 =====")
