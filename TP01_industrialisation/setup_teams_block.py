"""Enregistrer l'URL du webhook Teams dans un block Prefect et envoyer un message de test (TP4 partie 8).

    uv run python setup_teams_block.py            # saisie masquée de l'URL
    uv run python setup_teams_block.py --test-only

L'URL est un secret : elle n'est jamais écrite dans le dépôt, ni dans un fichier, ni affichée.
Elle est lue en saisie masquée (ou via la variable TEAMS_WEBHOOK_URL), puis stockée dans le block
côté serveur Prefect. Nécessite PREFECT_API_URL.
"""
import argparse
import getpass
import os

from prefect.blocks.notifications import MicrosoftTeamsWebhook

BLOCK_NAME = "teams-tp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="ne pas réenregistrer, seulement tester")
    args = parser.parse_args()
    if not os.getenv("PREFECT_API_URL"):
        raise SystemExit("Définir PREFECT_API_URL (ex. http://localhost:4200/api).")

    if not args.test_only:
        url = os.getenv("TEAMS_WEBHOOK_URL") or getpass.getpass("URL du webhook Teams (saisie masquée) : ")
        if not url.startswith("https://"):
            raise SystemExit("L'URL doit commencer par https://")
        MicrosoftTeamsWebhook(url=url).save(BLOCK_NAME, overwrite=True)
        print(f"Block '{BLOCK_NAME}' enregistré (l'URL n'est pas affichée).")

    block = MicrosoftTeamsWebhook.load(BLOCK_NAME)
    block.notify(body="TEST TP4 : message de vérification du canal d'alertes Prefect. "
                      "Aucune action requise.", subject="TEST TP4")
    print("Message de test envoyé : vérifie qu'il apparaît dans le canal Teams.")


if __name__ == "__main__":
    main()
