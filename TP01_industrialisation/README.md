# TP 01 + TP 02 — Deux modèles Steam, collecte PULL et lots PUSH

Projet unique (uv, Python 3.12). Énoncés : [TP01](ENONCE.md) et [TP02](../TP02_donnees_dynamiques/ENONCE.md).
Le TP02 étend ce projet : `src/client.py`, `src/ingestion.py`, `src/data_validation.py` et le dossier `api_trigger_passif/`.

## Démarrage (reprise par un binôme)

```sh
uv python install 3.12
uv sync --locked
uv run --locked python -m pytest -q          # tests hors réseau (~5 min : Great Expectations est lent)
uv run --locked python main.py --config config.json      # entraîne les deux modèles
uv run --locked python predict.py --run-dir runs/<run> --input data/exemple_inference.csv --output pred.csv
```

`data/games.csv` (94 948 lignes) n'est pas versionné : voir `../GUIDE_INSTALLATION.md`.
Les sorties (`runs/`, `diagnostics/`, `cache/`, `artifacts/`) sont ignorées par Git.

## Architecture

```text
main.py            # seulement 4 appels : load_data -> preprocess_data -> train_models -> evaluate_models
src/ingestion.py   # acquisition (csv | api | push_batch) -> snapshot candidat -> VALIDATION GX (dernier pas de load_data)
src/client.py      # client HTTP : timeout, cache, 1,1 s d'espacement, 3 tentatives, Retry-After
src/normalisation.py  # réponses Steam Store / SteamSpy / avis -> ligne du contrat TP1
src/data_validation.py # suites GX, rapports, barrière
src/preprocessing.py   # éligibilité, features, split (ajustement ensuite sur train uniquement)
src/training.py, src/evaluation.py  # forêts + baselines ; publication atomique du run
predict.py         # recharge les deux pipelines dans un nouveau processus
api_trigger_passif/  # API PUSH : main (FastAPI), validation (Pydantic), storage (SQLite), worker, training_trigger
```

**Contrat stable acquisition → entraînement** : un DataFrame aux colonnes du CSV TP1
(`appid, name, release_date, required_age, price, dlc_count, windows, mac, linux, achievements, genres, pct_pos_total, num_reviews_total, estimated_owners`)
plus `origin` (provenance, jamais une feature). `genres` est une liste Python sérialisée en texte.
`load_data` retourne exactement le DataFrame validé : rien n'est relu depuis un fichier mutable.

## Configuration (`config.json`)

| Clé | Rôle |
|---|---|
| `source.type` | `csv` (TP1), `api` (PULL : CSV de base + appids collectés, upsert) ou `push_batch` (CSV figé de 500 jeux) |
| `source.path` | CSV de base / lot (résolu depuis le dossier du fichier de config) |
| `output.runs_dir`, `output.run_dir` | dossier de runs ; `run_dir` explicite sinon `runs/run_<horodatage>`. Un run existant est **refusé** |
| `output.diagnostics_dir` | rapports de validation et snapshots rejetés (distinct des modèles) |
| `random_state`, `test_size`, `thresholds`, `models.*` | graine, split, seuils (10 avis, classes ≥ 5), hyperparamètres des forêts |
| `acquisition.appids / cache_dir / refresh` | PULL : 1 à 10 appids, cache, renouvellement (SteamSpy ≥ 24 h) |

Artefacts d'un run : `regression.joblib`, `classification.joblib` (pipelines complets, prétraitement inclus,
acceptent des métadonnées brutes), `metrics.json`, `split_*.csv`, `config.json`, `manifest.json` (empreintes, provenance,
rapports de validation) et `snapshot.csv` pour les sources non CSV. Le run est écrit dans un dossier temporaire puis publié
par `rename` : un échec ne laisse ni modèle ni métrique.

## Contrat de données — règle → expectation → périmètre → nulls/sentinelles

Suites créées par le SDK Python (`src/data_validation.py::build_suite`), toutes **bloquantes** (aucun `mostly` < 1).

| Règle métier | Expectation GX | Périmètre | Nulls / sentinelles |
|---|---|---|---|
| Features et cibles présentes | `ExpectColumnToExist` ×14 | toutes suites | colonne absente = rejet ; `positive/negative` non requises |
| Corpus non vide | `ExpectTableRowCountToBeBetween(min=1)` | toutes | — |
| `appid` entier positif, non nul, unique | `NotBeNull`, `OfType int64`, `Between(min=1)`, `BeUnique` | toutes, **snapshot complet** | jamais de null |
| Prix, DLC, succès ≥ 0 et finis | `InTypeList(int64,float64)` + `Between(0, 1e6)` | toutes | null autorisé (imputé au TP1) sauf `dlc_count` des nouveaux ; texte « gratuit » et ±inf rejetés |
| `required_age` | `Between` | historique : ≥ **-1** ; nouveaux : ≥ 0 | `-1` = sentinelle historique (manquant) |
| `pct_pos_total` | `Between` + `NotBeNull` (nouveaux) | historique : -1..100 ; nouveaux : 0..100 | `-1` historique seulement |
| `num_reviews_total` | `Between` + `NotBeNull` (nouveaux) | historique ≥ -1 ; nouveaux ≥ 10 | idem |
| Plateformes booléennes | `NotBeNull`, `OfType bool`, `InSet(True, False)` | toutes | pas de chaînes |
| Genres, dates | regex `^(\[.*\]\|Unknown)$`, `MatchStrftimeFormat %Y-%m-%d` | toutes | null toléré (historique), exigé (nouveaux) |
| Tranche de propriétaires | regex `^\d+ - \d+$` + `PairValuesAToBeGreaterThanB(owners_max, owners_min)` | toutes | historique : `or_equal` (`0 - 0` admis) ; nouveaux : strictement croissant et `NotBeInSet("0 - 0")` |
| Nom présent | `NotBeNull(name)` | nouveaux seulement | 2 noms manquants admis dans l'historique |
| Lot PUSH = exactement 500 lignes | `TableRowCountToBeBetween(500, 500)` | `steam_push_batch_v1` | — |
| Aptitude à entraîner | table d'audit : `train+test == eligible`, `train/test ≥ 2`, `class_count ≥ 2`, `train/test ≥ class_count` | `steam_training_audit_v1`, mêmes filtres que le prétraitement | classification seulement (`row_condition`) |

Sentinelles historiques admises : `required_age=-1`, `pct_pos_total=-1`, `num_reviews_total=-1`, `0 - 0` (exclue ensuite par le
prétraitement). Elles ne sont **jamais** étendues aux nouveaux jeux : le snapshot mixte est validé par `steam_historical_v1`
sur tout le corpus **et** par `steam_new_games_v1` sur les lignes dont `origin != historical-csv`.
Les nouveaux jeux sont aussi validés **avant** l'upsert.

Rapport (`diagnostics/validation_*.json`) : date, source, empreinte SHA-256 du snapshot, suite et version, version GX, succès,
contrôles en échec (exemples, ≤ 20 appids), cause et contrôles non évalués en cas d'erreur technique. Un échec de contrôle lève
`DataValidationError`, une erreur GX ou un résultat absent `DataValidationTechnicalError` : aucun des deux n'autorise l'entraînement.

## Workflows

| | PULL (actif) | PUSH (passif) |
|---|---|---|
| Lancer | `main.py --config <config api>` | API + `simulateur_push.py` |
| Initiateur | notre commande | producteur externe (POST par jeu) |
| Condition | collecte complète + dataset valide | 500 appids distincts valides non affectés |
| Latence | minutes (appels espacés de 1,1 s) | accepté immédiatement (202) ; entraînement ensuite en arrière-plan |
| Contrôle | codes de sortie, rapports GX | état du job `PENDING/RUNNING/SUCCEEDED/FAILED`, `/status`, `/batches/{id}` |
| Risques | quotas, schémas qui changent, labels non comparables au CSV | lots d'une seule classe, labels artificiels, producteur qui renvoie |
| Reprise | cache + même commande ; run existant refusé | lots `FAILED` conservés (jamais rejoués seuls) ; `RUNNING` interrompu → `PENDING` puis run déjà publié reconnu par empreinte |

La décision de déclencher est dans `storage.store_event` (une transaction `BEGIN IMMEDIATE`) ; le producteur n'a décidé que
d'envoyer des événements. « Exactement une fois » est plus fort qu'un POST idempotent : il faut aussi qu'un lot ne soit réservé qu'une
fois (transaction) et qu'un entraînement ne soit publié qu'une fois (dossier de run stable + empreinte).

### PULL

```sh
# config : "source": {"type": "api", "path": "data/games.csv"}, "acquisition": {"appids": [620, 400, 550], ...}
uv run --locked python main.py --config config_pull.json
```

Le premier run ajoute 3 jeux (94 948 → 94 951) ; avec `snapshot.csv` du run comme `source.path`, un second run les actualise
(94 951 inchangé). Les réponses sont mises en cache (URL et horodatage conservés) : sans `refresh`, aucun appel HTTP.

### PUSH

```sh
uv run --locked uvicorn api_trigger_passif.main:create_app --factory --host 127.0.0.1 --port 8000   # une seule instance
uv run --locked python simulateur_push.py 499      # puis 1, 1, 499 : /status = 499, 500, 501, 1000
```

Docs Swagger : <http://127.0.0.1:8000/docs>. État durable : `artifacts/push.sqlite3` (`PUSH_DB_PATH` pour changer) ; lots :
`artifacts/batches/batch_<id>/{games.csv, config.json, diagnostics/, run/}`.

Codes : 202 accepté (avec `batch_id` au 500ᵉ), 200 doublon identique (même événement ou même jeu), 409 conflit, 422 invalide.

## TP3 — Suivi MLflow, qualification et sélection

Serveur : `docker compose up -d --wait mlflow` depuis `TP00_setup_des_postes` (MLflow 3.16.1). Dans chaque terminal :

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_REGISTRY_URI = "http://localhost:5000"
```

Sans `MLFLOW_TRACKING_URI`, le suivi refuse d'écrire (jamais de dossier `mlruns` local involontaire). Le suivi s'active par
`"tracking": {"enabled": true, ...}` dans la configuration ; sans ce bloc, les TP1/TP2 fonctionnent comme avant.

| Élément | Où |
|---|---|
| Module technique MLflow (runs explicites, `run_id` transmis entre étapes, échec -> `FAILED` avec étape et cause) | `src/tracking.py` |
| Politique de qualification versionnée (seuils, empreinte du corpus, provenance) | `policy.json` |
| Décision testable, enregistrement idempotent, tags de version, alias, retour arrière | `src/registry.py` |
| Corpus de validation réservé (appids exclus de l'apprentissage), `steam-validation-v1` | `validation/`, `make_validation_corpus.py` |
| Deux variantes comparables + enregistrement + qualification + `comparison.md` | `run_experiments.py` |
| Rechargement par alias et par numéro dans un nouveau processus | `verify_reload.py` |

Un run MLflow par tentative d'entraînement (les deux modèles) ; en PUSH, un run par lot de 500 jeux (`batch_id` -> `run_id` dans
`artifacts/batches/batch_<id>/mlflow.json`). Modèles du registre : `steam-regression`, `steam-classification` ; `-demo` pour les
modèles synthétiques du PUSH (toujours `demo-only`).

```powershell
uv run python run_experiments.py --config config_pull.json                      # variantes v-a / v-b, qualification
uv run python -m src.registry qualify steam-regression 1                        # nouvelle évaluation (rapport daté)
uv run python -m src.registry select steam-regression 1 --actor moi --reason "..."   # refusé si non qualifié
uv run python -m src.registry rollback steam-regression                         # version précédemment sélectionnée
uv run python verify_reload.py steam-regression                                 # models:/<nom>@champion == models:/<nom>/<n>
```

Reprise : une publication a un `publication_id` stable ; relancer `run_experiments.py` ou rappeler le déclencheur d'un lot déjà
publié ne crée aucune version en double. Après une panne du serveur pendant un run, un diagnostic local
`diagnostics/mlflow_unreconciled_<run_id>.json` est écrit ; après reprise :
`tracking.reconcile_unclosed_runs("diagnostics")`. Les runs restés `RUNNING` après un arrêt brutal se listent avec
`tracking.find_open_runs()`.

Résultats constatés (corpus `steam-validation-v1`) : régression qualifiée (MAE 12,82 contre 13,92 pour la baseline) ; classification
refusée (F1 macro 0,11 contre 0,07, seuil 0,25 ; classes jamais reconnues) : aucun alias `champion` pour la classification.

## Limites

- Les labels du simulateur sont artificiels : les scores des lots PUSH ne disent rien sur Steam. Les scores PULL/PUSH ne sont pas
  comparables (corpus et splits différents) ; aucun modèle n'est promu automatiquement.
- `pct_pos_total` recalculé par le TP2 (périmètre d'avis explicite) peut différer de celui du CSV historique ; devise du CSV non confirmée.
- Le split est aléatoire (évaluation rétrospective) ; les classes rares restent difficiles (F1 macro faible, accuracy trompeuse).
- Un seul worker/instance d'API est supporté ; la reprise après arrêt repose sur SQLite et l'empreinte du CSV de lot.
