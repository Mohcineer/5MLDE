
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("TP02_Steam")

with mlflow.start_run(run_name="Premier_test_TP02"):
    mlflow.log_param("modele", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("mae", 12.5)
    mlflow.log_metric("r2", 0.85)

    mlflow.set_tag("projet", "TP02")

print("Run enregistré avec succès !")
