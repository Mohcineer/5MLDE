"""Entraînement : forêt + baseline par tâche, prétraitement ajusté sur train uniquement."""
import time

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src import tracking
from src.orchestration import logger, step
from src.preprocessing import RAW_FEATURE_COLUMNS, make_pipeline

DEFAULT_PARAMS = {
    "regression": {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 5},
    "classification": {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 5},
}


def _params(config, task):
    return {**DEFAULT_PARAMS[task], **config.get("models", {}).get(task, {})}


@step("Entraînement")
def train_models(prepared_data):
    """Ajuster les deux modèles et leurs baselines sur train ; conserver test et configuration."""
    ctx = prepared_data.get("tracking")
    logger().info("Entraînement des deux modèles (graine %s)", prepared_data["config"].get("random_state", 42))
    with tracking.stage(ctx, "training"):
        return _train(prepared_data, ctx)


def _train(prepared_data, ctx):
    config = prepared_data["config"]
    seed = config.get("random_state", 42)
    trained = {}
    for task, info in prepared_data["tasks"].items():
        started = time.perf_counter()
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
        tracking.log_params(ctx, {f"{task}.{key}": value for key, value in _params(config, task).items()}
                            | {f"{task}.random_state": seed, f"{task}.algorithm": type(model[-1]).__name__,
                               f"{task}.baseline": type(baseline[-1]).__name__})
        tracking.log_metrics(ctx, {f"{task}.duration.training_s": time.perf_counter() - started})
    return {
        "config": config, "paths": prepared_data["paths"],
        "validation": prepared_data.get("validation"),
        "snapshot": prepared_data["snapshot"],
        "provenance": prepared_data.get("provenance"),
        "cache": prepared_data.get("cache"), "tracking": ctx, "validation_corpus": prepared_data.get("validation_corpus"),
        "tasks": trained,
    }
