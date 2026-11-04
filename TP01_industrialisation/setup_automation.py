"""Créer ou mettre à jour l'automation d'alerte Teams (TP4 partie 8) : point d'entrée de configuration.

    uv run python setup_automation.py

Chemin : flow run Failed/Crashed -> événement Prefect -> automation -> block `teams-tp4` -> canal Teams.
Réactive, limitée aux flows du binôme. Les tentatives de tâche et `AwaitingRetry` ne déclenchent rien :
un incident transitoire résolu par les retries reste silencieux. Idempotent : l'automation existante
(même nom) est modifiée, jamais dupliquée.

Le poste est mentionné dans le message : `localhost:4200` n'ouvre la bonne interface que sur ce poste.
"""
import os
import socket
from datetime import timedelta

from prefect.automations import Automation, EventTrigger, Posture
from prefect.blocks.notifications import MicrosoftTeamsWebhook
from prefect.events.actions import SendNotification

NAME = "TP4 alerte Teams : echec du pipeline Steam"
BLOCK_NAME = "teams-tp4"
BINOME = os.getenv("BINOME", "binome-tp4")
FLOWS = ["Pipeline Steam", "Publication des modèles de démonstration"]  # PULL/déploiement et PUSH

SUBJECT = "ALERTE TP4 : {{ flow.name }} en échec ({{ flow_run.state.name }})"
BODY = (
    "Flow : {{ flow.name }}\n"
    "Flow run : {{ flow_run.name }} (ID {{ flow_run.id }})\n"
    "État terminal : {{ flow_run.state.name }} à {{ flow_run.state.timestamp }}\n"
    "Cause : {{ flow_run.state.message or 'indisponible' }}\n"
    "Références (mode, batch_id, run MLflow, étape en erreur) : "
    "{{ flow_run.tags | join(', ') or 'indisponible' }}\n"
    "Binôme / poste : " + BINOME + " / " + socket.gethostname() + "\n"
    "Ouvrir l'exécution (uniquement depuis ce poste) : {{ flow_run|ui_url }}\n"
    "Aucun modèle n'est publié après un échec définitif du pipeline."
)


def main():
    if not os.getenv("PREFECT_API_URL"):
        raise SystemExit("Définir PREFECT_API_URL (ex. http://localhost:4200/api).")
    block = MicrosoftTeamsWebhook.load(BLOCK_NAME)
    trigger = EventTrigger(
        expect={"prefect.flow-run.Failed", "prefect.flow-run.Crashed"},
        match_related={"prefect.resource.role": "flow", "prefect.resource.name": FLOWS},
        posture=Posture.Reactive, threshold=1, within=timedelta(0),
    )
    actions = [SendNotification(block_document_id=block._block_document_id, subject=SUBJECT, body=BODY)]

    try:
        automation = Automation.read(name=NAME)
    except Exception:
        automation = None
    if automation is None:
        automation = Automation(name=NAME, description="Alerte Teams sur échec définitif du pipeline",
                                trigger=trigger, actions=actions).create()
        print(f"Automation créée : {automation.id}")
    else:
        automation.trigger, automation.actions = trigger, actions
        automation.update()
        print(f"Automation mise à jour (même identifiant) : {automation.id}")


if __name__ == "__main__":
    main()
