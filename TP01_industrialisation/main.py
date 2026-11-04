"""Orchestration uniquement : la logique métier appartient aux quatre modules."""
from prefect import flow

from src.ingestion import load_data
from src.preprocessing import preprocess_data
from src.training import train_models
from src.evaluation import evaluate_models


@flow(name="Pipeline Steam", retries=0, log_prints=True)
def main(config_path):
    data = load_data(config_path)
    prepared_data = preprocess_data(data)
    trained_models = train_models(prepared_data)
    return evaluate_models(trained_models)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Entraîner et évaluer les deux modèles Steam')
    parser.add_argument('--config', required=True, help='Chemin du fichier de configuration JSON')
    main(parser.parse_args().config)
