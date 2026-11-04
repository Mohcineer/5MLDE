"""Trois essais du cache des partitions (TP4 partie 7) sur un petit snapshot, dans un flow Prefect.

    uv run python demo_cache.py

1. même snapshot, mêmes paramètres : calcul (miss) puis réutilisation (hit), résultats identiques ;
2. contenu du snapshot modifié sous le même nom : nouveau calcul ;
3. paramètre, puis version du prétraitement modifiés : invalidation.
"""
import random

import pandas as pd
from prefect import flow

from src import preprocessing
from src.preprocessing import partitions_with_state

CONFIG = {"thresholds": {"regression_min_reviews": 10, "classification_min_class_count": 5},
          "test_size": 0.2}


def appids(parts):
    return sorted(parts["reg_train"]["appid"]) + sorted(parts["clf_test"]["appid"])


@flow(name="Essais du cache des partitions", log_prints=True)
def trials():
    frame = pd.read_csv("validation/validation.csv").head(3000)
    seed = random.randint(10_000, 99_999)  # graine neuve : le premier appel est un vrai calcul
    config = {**CONFIG, "random_state": seed}
    results = []

    first, a = partitions_with_state(frame, config)
    second, b = partitions_with_state(frame, config)
    results.append(("1. mêmes entrées, 1er appel", a["status"], "miss"))
    results.append(("1. mêmes entrées, 2e appel", b["status"], "hit"))
    results.append(("1. résultats identiques", appids(first) == appids(second), True))

    changed = frame.copy()
    changed.loc[changed.index[0], "price"] = float(changed["price"].iloc[0]) + 1.0
    _, c = partitions_with_state(changed, config)  # même nom de fichier, contenu différent
    results.append(("2. contenu du snapshot modifié", c["status"], "miss"))

    _, d = partitions_with_state(frame, {**config, "test_size": 0.25})
    results.append(("3. paramètre modifié (test_size)", d["status"], "miss"))

    original = preprocessing.PREPROCESSING_VERSION
    preprocessing.PREPROCESSING_VERSION = original + "-essai"
    try:
        _, e = partitions_with_state(frame, config)
    finally:
        preprocessing.PREPROCESSING_VERSION = original
    results.append(("3. version du prétraitement modifiée", e["status"], "miss"))

    _, f = partitions_with_state(frame, config)  # retour à l'état d'origine : réutilisé
    results.append(("   retour aux entrées d'origine", f["status"], "hit"))

    ok = True
    for label, observed, expected in results:
        good = observed == expected
        ok &= good
        print(f"{'OK ' if good else 'KO '} {label} : {observed} (attendu {expected})")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if trials() else 1)
