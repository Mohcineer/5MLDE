"""Figer le corpus de validation (qualification) et un jeu de bilan final, issus des données réelles.

    uv run python make_validation_corpus.py --csv data/games.csv --out validation

Deux ensembles disjoints d'appids sont réservés (graine fixe, donc reproductibles) :
- `validation.csv` : sert à comparer et qualifier les candidats (ce n'est plus un test indépendant) ;
- `final_test.csv` : bilan final, à ne pas utiliser pour ajuster les seuils.
Leurs appids sont exclus de l'apprentissage, de l'ajustement du prétraitement et des collectes.
Le manifeste fixe empreintes, effectifs et périmètre de classes (le même pour tous les candidats).
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocessing import (CLASSIFICATION_TARGET, eligible_classification, eligible_regression)
from src.tracking import sha256_file

PROTOCOL_ID = "steam-validation-v1"
LF = chr(10)  # fins de ligne identiques sur tous les postes (empreintes stables)


def build(csv_path, out_dir, seed=2026, validation_frac=0.10, final_frac=0.10,
          min_reviews=10, min_class_count=5):
    frame = pd.read_csv(csv_path)
    if frame["appid"].duplicated().any():
        raise ValueError("Doublons d'appid dans le CSV source.")
    shuffled = frame.sample(frac=1.0, random_state=seed)
    n_val, n_final = int(len(frame) * validation_frac), int(len(frame) * final_frac)
    validation, final = shuffled.iloc[:n_val], shuffled.iloc[n_val:n_val + n_final]

    # Périmètre de classes fixé sur le corpus complet, identique pour toutes les versions.
    scope_frame, excluded = eligible_classification(frame, min_class_count)
    labels = sorted(scope_frame[CLASSIFICATION_TARGET].unique(),
                    key=lambda label: int(label.split(" - ")[0]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, part in (("validation.csv", validation), ("final_test.csv", final)):
        part.to_csv(out / name, index=False, lineterminator=LF)
        files[name] = {"rows": len(part), "sha256": sha256_file(out / name),
                       "regression_eligible": len(eligible_regression(part, min_reviews)),
                       "classification_eligible": int(part[CLASSIFICATION_TARGET].isin(labels).sum())}
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "source_csv": Path(csv_path).name, "source_sha256": sha256_file(csv_path),
        "source_rows": len(frame), "files": files,
        "regression_filter": f"pct_pos_total in [0,100] et num_reviews_total >= {min_reviews}",
        "classification_labels": labels,
        "classification_excluded": {str(k): int(v) for k, v in excluded.items()} | {"0 - 0": "exclue"},
        "note": ("validation.csv = sélection/qualification (n'est plus un test indépendant) ; "
                 "final_test.csv = bilan final sans ajustement de seuils."),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="validation")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = build(args.csv, args.out, args.seed)
    print(json.dumps({k: result[k] for k in ("protocol_id", "files")}, indent=2))
