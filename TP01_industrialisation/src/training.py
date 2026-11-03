"""Entraînement : forêt + baseline par tâche, prétraitement ajusté sur train uniquement."""
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.preprocessing import RAW_FEATURE_COLUMNS, make_pipeline

DEFAULT_PARAMS = {
    "regression": {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 5},
    "classification": {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 5},
}


def _params(config, task):
    return {**DEFAULT_PARAMS[task], **config.get("models", {}).get(task, {})}


def train_models(prepared_data):
    """Ajuster les deux modèles et leurs baselines sur train ; conserver test et configuration."""
    config = prepared_data["config"]
    seed = config.get("random_state", 42)
    trained = {}
    for task, info in prepared_data["tasks"].items():
        train = info["train"]
        x_train, y_train = train[RAW_FEATURE_COLUMNS], train[info["target"]]
        if task == "regression":
            model = make_pipeline(RandomForestRegressor(
                random_state=seed, n_jobs=1, **_params(config, task)))
            baseline = make_pipeline(DummyRegressor(strategy="mean"))
        else:
            model = make_pipeline(RandomForestClassifier(
                random_state=seed, n_jobs=1, **_params(config, task)))
            baseline = make_pipeline(DummyClassifier(strategy="most_frequent"))
        model.fit(x_train, y_train)
        baseline.fit(x_train, y_train)
        trained[task] = {**info, "model": model, "baseline": baseline,
                         "params": _params(config, task)}
    return {
        "config": config, "paths": prepared_data["paths"],
        "validation": prepared_data.get("validation"),
        "snapshot": prepared_data["snapshot"],
        "provenance": prepared_data.get("provenance"),
        "tasks": trained,
    }
