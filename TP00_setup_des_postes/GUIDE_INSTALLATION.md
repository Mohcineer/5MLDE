# Guide d'installation des prérequis — TPs 5MLDE

© SUPINFO — Auteur : Baptiste DAVID

Préparer votre poste avant de commencer les TPs. Suivre, dans chaque chapitre, les instructions
correspondant à votre système. Une installation existante qui passe les vérifications peut être conservée.

| Prérequis | Utilité |
|---|---|
| Docker et Docker Compose | Exécuter MLflow, Prefect, Grafana et PostgreSQL |
| uv | Installer Python, les dépendances et lancer les commandes du projet |
| Python 3.12 | Exécuter le code des TPs, dans l'environnement géré par uv |
| Dataset Steam téléchargé depuis Kaggle | Fournir les données initiales, absentes du dépôt Git |
| Git déjà disponible | Versionner le code, les configurations et les dépendances |
| Éditeur et navigateur | Développer, ouvrir le notebook initial et utiliser les interfaces locales |
| Compte Teams avec Workflows | Réaliser l'exercice d'alerting du TP4 |

Utiliser **PowerShell sous Windows**, **Terminal sous macOS**, et **Bash sous Linux** pour les exemples.
Les commandes ne comprennent pas le symbole de l'invite du terminal. Dans les exemples, remplacer
`DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE` par le chemin du dossier `5MLDE` créé par le clonage.

Prévoir un accès Internet pour les téléchargements, de l'espace pour les images Docker, les données
et les environnements Python, ainsi que les droits d'installation nécessaires sur votre poste.
Un GPU n'est pas requis. Sur une machine gérée par l'établissement, faire installer les outils
manquants par son support si les droits administrateur sont indisponibles.

## 0 - Clonage du répo

Clonez le repository sur votre poste à l'aide de la commande suivante : 

```sh
git clone https://github.com/SUPINFO-Organization/5MLDE.git
```

## 1 — Docker et Docker Compose

Docker exécute les services du cours dans des conteneurs Linux. Compose lit le fichier commun
`compose.yml`. La commande attendue est **`docker compose`**, avec une espace ; installer un plugin
Compose actuel qui prend en charge `up --wait`.

### Windows

1. Vérifier que votre version de Windows et votre processeur figurent dans les prérequis de
   [Docker Desktop pour Windows](https://docs.docker.com/desktop/setup/install/windows-install/).
   La virtualisation matérielle doit être activée dans le BIOS/UEFI.
2. Dans **PowerShell administrateur**, préparer WSL si nécessaire :

```powershell
wsl --install
wsl --update
wsl --set-default-version 2
```

Redémarrer si Windows le demande, puis vérifier :

```powershell
wsl --version
wsl --status
```

Si WSL est déjà installé, commencer par sa mise à jour. En cas de difficulté, suivre la
[procédure WSL de Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install).

3. Télécharger l'installateur Docker Desktop adapté au processeur depuis la page officielle.
   Lancer l'installation et choisir le moteur WSL 2 lorsqu'il est proposé.
4. Ouvrir Docker Desktop et attendre que son moteur soit démarré. Utiliser le mode **conteneurs Linux**.
5. Fermer puis rouvrir un PowerShell normal et vérifier :

```powershell
docker version
docker compose version
docker run --rm hello-world
```

Pour ce guide, uv, Python et le code s'exécutent **dans Windows**, depuis PowerShell. WSL sert au
moteur Docker. Ne pas partager un même `.venv` entre Windows et une distribution WSL.

### macOS

1. Dans le menu Apple → À propos de ce Mac, identifier **Apple Silicon** ou **Intel**.
2. Télécharger le paquet correspondant depuis
   [Docker Desktop pour Mac](https://docs.docker.com/desktop/setup/install/mac-install/).
3. Ouvrir le fichier `.dmg`, déplacer Docker dans Applications et lancer Docker.
4. Terminer la configuration demandée et attendre le démarrage du moteur.
5. Dans Terminal :

```sh
docker version
docker compose version
docker run --rm hello-world
```

Docker Desktop inclut Compose. Conserver l'architecture native de votre Mac ; ne pas imposer
`linux/amd64` dans le Compose sur Apple Silicon pour résoudre un problème sans en identifier la cause.

### Linux

Utiliser **Docker Engine et le plugin Compose**. Les commandes suivantes concernent une installation
sur une distribution prise en charge, avec systemd. Si Docker existe déjà, vérifier son fonctionnement
avant de modifier ses paquets. En cas de paquets conflictuels, suivre la procédure officielle de
votre distribution, sans supprimer ses données Docker.

#### Ubuntu ou Debian

Ces commandes ciblent Ubuntu ou Debian directement, pas automatiquement leurs distributions dérivées.
Ouvrir Bash, identifier le système et préparer le dépôt :

```sh
cat /etc/os-release
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
```

**Ubuntu seulement :**

```sh
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
```

**Debian seulement :**

```sh
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: $(. /etc/os-release && echo "$VERSION_CODENAME")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
```

Puis, sur les deux distributions :

```sh
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
```

Références : [Ubuntu](https://docs.docker.com/engine/install/ubuntu/) et
[Debian](https://docs.docker.com/engine/install/debian/).

#### Fedora

Sur une version prise en charge utilisant DNF 5 :

```sh
sudo dnf install dnf-plugins-core
sudo dnf config-manager addrepo --from-repofile https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
```

Référence : [Docker Engine sur Fedora](https://docs.docker.com/engine/install/fedora/).
Pour une autre distribution, utiliser sa procédure depuis
[les installations Docker Engine](https://docs.docker.com/engine/install/) ; ne pas y copier les dépôts Ubuntu.

#### Accéder à Docker depuis votre compte

Vous pouvez conserver `sudo docker ...` pour les commandes Docker. Pour employer les commandes des
TPs sans `sudo`, sur votre poste de travail autorisé :

```sh
sudo usermod -aG docker "$USER"
```

Fermer la session utilisateur et se reconnecter, puis lancer `docker run --rm hello-world`.
Le groupe `docker` donne des privilèges comparables à ceux de root : sur un poste partagé ou géré,
utiliser le mode d'accès prévu par l'administrateur. Ne pas rendre le socket Docker accessible à tous.
[Référence : configuration après installation](https://docs.docker.com/engine/install/linux-postinstall/).

### Résultat attendu

`docker version` affiche un client **et un serveur**, `docker compose version` fonctionne et
`hello-world` s'exécute. Le moteur doit rester actif pendant l'utilisation des services.

## 2 — uv, gestionnaire du projet Python

uv s'installe sans Python préalable. Exécuter son installation avec votre compte habituel,
pas dans un terminal administrateur ni avec `sudo`.

### Windows

Dans PowerShell :

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cette commande utilise l'installateur officiel et limite l'option de politique d'exécution au
processus lancé. Fermer puis rouvrir PowerShell :

```powershell
uv --version
Get-Command uv
```

### macOS

Dans Terminal :

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Fermer puis rouvrir Terminal, puis vérifier :

```sh
uv --version
command -v uv
```

### Linux

Installer `curl` s'il manque : `sudo apt install curl` sur Ubuntu/Debian, ou `sudo dnf install curl`
sur Fedora. Dans votre terminal utilisateur :

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Rouvrir le terminal, puis exécuter `uv --version` et `command -v uv`.
Si uv reste introuvable, suivre l'indication de PATH affichée par l'installateur.

Référence pour les trois systèmes : [installation de uv](https://docs.astral.sh/uv/getting-started/installation/).

## 3 — Python 3.12 et environnement du projet

Le TP1 exige **Python 3.12** et fournit `pyproject.toml`, `uv.lock` et `.python-version`.
uv installe l'interpréteur et crée `.venv` : aucune installation globale de pandas, sklearn,
Jupyter ou des SDK MLOps n'est nécessaire.
[Référence : gérer Python avec uv](https://docs.astral.sh/uv/guides/install-python/).

### Windows

Depuis le dépôt cloné à l’étape précédente, adapter le chemin si nécessaire :

```powershell
uv python install 3.12
Set-Location "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE\TP01_industrialisation"
uv sync --locked
uv run --locked python --version
uv run --locked python -c "import pandas, sklearn; print('Environnement prêt')"
uv run --locked jupyter lab exploration_steam.ipynb
```

### macOS

Dans Terminal, après le clonage du dépôt :

```sh
uv python install 3.12
cd "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE/TP01_industrialisation"
uv sync --locked
uv run --locked python --version
uv run --locked python -c "import pandas, sklearn; print('Environnement prêt')"
uv run --locked jupyter lab exploration_steam.ipynb
```

### Linux

Remplacer le placeholder par le chemin du dépôt cloné :

```sh
uv python install 3.12
cd "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE/TP01_industrialisation"
uv sync --locked
uv run --locked python --version
uv run --locked python -c "import pandas, sklearn; print('Environnement prêt')"
uv run --locked jupyter lab exploration_steam.ipynb
```

### Résultat attendu et suite du parcours

L'interpréteur affiche `Python 3.12.x`, les imports réussissent et Jupyter ouvre le notebook initial.
Si le navigateur ne s'ouvre pas automatiquement, utiliser l'URL affichée dans le terminal.
Arrêter Jupyter avec Ctrl+C. Choisir le kernel de ce projet si plusieurs sont proposés.

Ne pas activer manuellement `.venv` : préfixer les commandes par `uv run --locked` suffit.
Ne pas copier un environnement `.venv` entre systèmes ; le recréer avec le verrou.
À partir du TP2, conserver **le projet que vous avez industrialisé au TP1**, sans nouveau notebook.
Les fichiers incomplets du squelette ne deviennent pas une solution après installation.

Installer les dépendances supplémentaires au début du TP concerné, dans ce même projet.
Ces commandes s'appliquent dans PowerShell, Terminal macOS et Bash :

| TP | Commande d'ajout |
|---|---|
| TP2 | `uv add httpx fastapi uvicorn "great-expectations==1.23.2"` |
| TP3 | `uv add "mlflow==3.16.1"` |
| TP4 | `uv add "prefect==3.8.7"` |
| TP5 | `uv add fastapi uvicorn "psycopg[binary,pool]" "locust==2.46.6"` |

Versionner les modifications de `pyproject.toml` et `uv.lock` après chaque ajout.
`uv sync --locked` refuse un verrou obsolète : ne pas supprimer le verrou pour masquer un conflit.
Ne pas lancer une mise à jour globale des dépendances au milieu d'un TP sans en examiner les effets.

## 3 bis — Télécharger et préparer le dataset Steam

**Les données ne sont pas incluses dans le dépôt Git.** Réaliser cette étape après l'installation
de l'environnement Python et **avant d'exécuter les cellules du notebook du TP1**.

1. Ouvrir [Steam Games Dataset sur Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download).
2. Cliquer sur **Download**, puis **Download dataset as zip**. Si Kaggle demande une connexion,
   se connecter ou créer un compte. Aucun SDK Kaggle ni clé API n'est nécessaire pour ce téléchargement manuel.
3. Extraire l'archive et repérer **`games_march2025_full.csv`**. Utiliser la version complète,
   pas `games_march2025_cleaned.csv`. Si la source évolue, retrouver la version contenant ce fichier.
4. Copier ce CSV dans **`TP01_industrialisation/data/`**, en conservant son nom.
   Les procédures ci-dessous détaillent cette copie selon votre système.

### Windows

Dans l'Explorateur de fichiers, faire un clic droit sur l'archive téléchargée → **Extraire tout**.
Copier `games_march2025_full.csv` depuis le dossier extrait vers
`TP01_industrialisation\data\`. Dans PowerShell, adapter le chemin des supports et vérifier :

```powershell
Set-Location "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE\TP01_industrialisation"
Get-Item .\data\games_march2025_full.csv
```

### macOS

Dans Finder, ouvrir l'archive ZIP pour l'extraire. Copier `games_march2025_full.csv` depuis le
dossier extrait vers `TP01_industrialisation/data/`. Dans Terminal :

```sh
cd "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE/TP01_industrialisation"
ls -lh data/games_march2025_full.csv
```

### Linux

Dans le gestionnaire de fichiers, extraire l'archive ZIP. Copier `games_march2025_full.csv`
depuis le dossier extrait vers `TP01_industrialisation/data/`. Dans Bash, adapter le chemin :

```sh
cd "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE/TP01_industrialisation"
ls -lh data/games_march2025_full.csv
```

### Générer le fichier attendu par le TP1 — les trois systèmes

Le notebook lit **`data/games.csv`**. Depuis `TP01_industrialisation`, exécuter cette commande
avec l'environnement installé au chapitre précédent. Elle sélectionne les colonnes du manifeste,
conserve toutes les lignes dans leur ordre et écrit le fichier de travail (en remplaçant une éventuelle
copie précédente de `data/games.csv`). Le CSV complet reste intact.

```sh
uv run --locked python -c "import json; from pathlib import Path; import pandas as pd; m = json.loads(Path('data/manifest.json').read_text(encoding='utf-8')); df = pd.read_csv('data/games_march2025_full.csv', usecols=m['columns']); df[m['columns']].to_csv('data/games.csv', index=False); print('Fichier prêt : data/games.csv —', len(df), 'lignes')"
```

Pour le fichier de référence du cours, le résultat attendu est **94 948 lignes et 16 colonnes**.
Vérifier la lecture avant de lancer le notebook :

```sh
uv run --locked python -c "import pandas as pd; df = pd.read_csv('data/games.csv'); print(df.shape); print(df.columns.tolist()); assert {'pct_pos_total', 'estimated_owners'}.issubset(df.columns)"
```

Le manifeste décrit le fichier de référence : si le téléchargement ne correspond pas à cette version,
ne pas présenter ses empreintes comme celles de votre copie. Signaler l'écart avant de poursuivre.
Conserver les CSV hors de Git : le `.gitignore` du TP1 exclut déjà `data/*.csv`.

## 4 — Éditeur de code et navigateur

Utiliser votre éditeur habituel s'il sait travailler sur un projet Python. La procédure ci-dessous
propose Visual Studio Code ; il ne remplace ni Python ni uv.

### Windows

Télécharger le **User Installer** depuis [VS Code pour Windows](https://code.visualstudio.com/docs/setup/windows),
le lancer puis ouvrir le dossier complet du projet avec Fichier → Ouvrir le dossier.
Utiliser PowerShell dans le terminal intégré. Choisir un navigateur à jour, par exemple Edge ou Firefox.

### macOS

Télécharger l'archive adaptée depuis [VS Code pour macOS](https://code.visualstudio.com/docs/setup/mac),
extraire l'application et la déplacer dans Applications. Ouvrir le dossier du projet.
Utiliser le terminal intégré et un navigateur à jour, par exemple Safari, Chrome ou Firefox.

### Linux

Télécharger le paquet adapté depuis [VS Code pour Linux](https://code.visualstudio.com/docs/setup/linux).
Dans le dossier du téléchargement, remplacer le nom ci-dessous par celui du fichier reçu :

```sh
# Ubuntu/Debian
sudo apt install ./code_VERSION_ARCH.deb
```

```sh
# Fedora
sudo dnf install ./code-VERSION.ARCH.rpm
```

Ouvrir le dossier du projet et utiliser Bash dans le terminal intégré. Un navigateur à jour tel
que Firefox ou Chrome permet d'accéder aux interfaces des TPs.

### Configurer Python dans l'éditeur

Sur les trois systèmes, installer l'extension **Python de Microsoft** si vous utilisez VS Code.
Après `uv sync`, sélectionner l'interpréteur `.venv\Scripts\python.exe` sous Windows ou
`.venv/bin/python` sous macOS/Linux. Le notebook du TP1 peut être ouvert dans JupyterLab :
l'extension notebook de l'éditeur n'est pas indispensable au parcours.

**Vérification :** le terminal intégré exécute `uv run --locked python --version` depuis le bon dossier.

## 5 — Teams et Workflows pour l'alerting du TP4

Prévoir un compte de l'établissement, l'accès à un canal Teams standard dédié au cours et
l'autorisation d'utiliser **Workflows / Power Automate**. Les droits et licences dépendent du tenant.
Faire vérifier cet accès avant l'exercice ; le compte personnel Microsoft ne garantit pas ces capacités.

### Windows

Ouvrir [Teams dans le navigateur](https://teams.microsoft.com/) avec le compte de l'établissement,
ou utiliser le client Teams déjà installé. Vérifier l'accès au canal et à l'application Workflows.

### macOS

Ouvrir Teams dans un navigateur pris en charge par votre établissement, ou dans le client Teams
déjà installé. Se connecter avec le même compte scolaire et vérifier le canal et Workflows.

### Linux

Utiliser Teams dans le navigateur, avec le compte scolaire. Vérifier que le canal et Workflows
sont accessibles ; aucun client de bureau Linux n'est requis pour ce TP.

Le TP4 guidera la création du webhook et du bot de publication. Ne pas mettre son URL secrète dans
Git. Si Workflows ou le mode d'authentification nécessaire est interdit, utiliser le canal préparé
par l'enseignant ou le repli HTTP décrit dans le TP4, en signalant que Teams n'a pas été validé.
[Référence : webhooks Teams avec Workflows](https://support.microsoft.com/en-us/workflows/send-messages-in-teams-using-incoming-webhooks).

## 6 — Installer et vérifier les services communs aux TPs

Les services s'installent via Docker : il n'est pas nécessaire d'installer les serveurs MLflow,
Prefect, Grafana ou PostgreSQL directement sur votre système. Les SDK Python restent dans uv.

Les fichiers du TP5 sont nécessaires aux montages du Compose, même pour un premier lancement.
Les versions sont fixées dans ce fichier : MLflow 3.16.1, Prefect 3.8.7, Grafana 13.2.3 et PostgreSQL 18.6.
`docker compose pull` télécharge ces versions sans les remplacer par une nouvelle version majeure.

### Windows

Docker Desktop doit être démarré. Dans PowerShell, se placer dans le dossier `TP00_setup_des_postes` du dépôt :

```powershell
Set-Location "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE\TP00_setup_des_postes"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose config --quiet
docker compose pull
docker compose up -d --wait --wait-timeout 180
docker compose ps
```

Dans chaque terminal qui exécute le code Python, définir :

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_REGISTRY_URI = "http://localhost:5000"
$env:PREFECT_API_URL = "http://localhost:4200/api"
```

### macOS

Docker Desktop doit être démarré. Dans Terminal :

```sh
cd "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE/TP00_setup_des_postes"
test -f .env || cp .env.example .env
docker compose config --quiet
docker compose pull
docker compose up -d --wait --wait-timeout 180
docker compose ps
```

Dans chaque terminal du projet Python :

```sh
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_REGISTRY_URI=http://localhost:5000
export PREFECT_API_URL=http://localhost:4200/api
```

### Linux

Vérifier que le service Docker est démarré. Dans Bash :

```sh
cd "DOSSIER_DANS_LEQUEL_LE_PROJET_EST_CLONE/TP00_setup_des_postes"
test -f .env || cp .env.example .env
docker compose config --quiet
docker compose pull
docker compose up -d --wait --wait-timeout 180
docker compose ps
```

Si votre accès Docker nécessite `sudo`, le préfixer uniquement sur les commandes Docker.
Dans chaque terminal du projet Python, sans `sudo` :

```sh
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_REGISTRY_URI=http://localhost:5000
export PREFECT_API_URL=http://localhost:4200/api
```

### Accès et contrôles sur les trois systèmes

| Service | Adresse locale par défaut | Accès |
|---|---|---|
| MLflow | [localhost:5000](http://localhost:5000) | Sans authentification dans cette stack locale |
| Prefect | [localhost:4200](http://localhost:4200) | Sans authentification dans cette stack locale |
| Grafana | [localhost:3000](http://localhost:3000) | `admin` / `5mlde-local-admin` |
| PostgreSQL | `127.0.0.1:5432`, base `steam_api` | Comptes ci-dessous |

Les services sont liés à `127.0.0.1`. Adapter les ports dans `.env` **avant le lancement** s'ils sont
déjà occupés ; reporter les mêmes ports dans les variables des clients. `.env` est lu par Compose,
mais ne définit pas automatiquement les variables du terminal. Redémarrer un worker applicatif
après changement de ses variables d'environnement.

Vérifier les tables avec le client psql présent dans le conteneur :

```sh
docker compose exec -T postgres psql -U postgres -d steam_api -c "\dt tracing.*"
```

Les quatre tables attendues sont `model_versions`, `api_requests`, `predictions` et `feedbacks`.
Grafana doit proposer la source **Steam API - PostgreSQL**. Le dashboard sera créé au TP5.

| Compte PostgreSQL | Mot de passe par défaut | Utilisation |
|---|---|---|
| `postgres` | `5mlde-local-postgres` | Administration seulement |
| `api_writer` | `5mlde-local-api` | Lecture et insertion depuis l'API |
| `grafana_reader` | `5mlde-local-grafana` | Lecture seule depuis Grafana |

Ces mots de passe de démonstration sont configurables dans `.env`. Au TP5, configurer l'API avec
`api_writer`, l'hôte `127.0.0.1`, le port choisi et la base `steam_api`, selon le format de connexion
que vous implémentez. Grafana utilise `postgres:5432` depuis son conteneur. Un `localhost` dans un
conteneur désigne ce conteneur, pas votre poste.

L'initialisation PostgreSQL exécute [le script fourni](TP05_api_supervision/init-db.sql) uniquement
sur un volume vide. Pour réappliquer cette même version sans effacer les traces :

```sh
docker compose exec -T postgres psql -U postgres -d steam_api -f /docker-entrypoint-initdb.d/010-tracing.sql
```

Si vous modifiez `API_DB_PASSWORD` ou `GRAFANA_DB_PASSWORD`, recréer les conteneurs avec
`docker compose up -d postgres grafana`, puis réappliquer ce script et adapter les clients.
Modifier `POSTGRES_PASSWORD` dans `.env` ne change pas le mot de passe du superutilisateur déjà
créé. De même, le mot de passe d'initialisation Grafana ne remplace pas celui d'un compte existant.

### Arrêt et diagnostic

Depuis la racine des supports, sur Windows, macOS ou Linux :

```sh
docker compose logs --tail=100 mlflow prefect grafana postgres
docker compose stop
```

Pour reprendre : `docker compose up -d --wait`.
`docker compose down` conserve les volumes ; **ne pas ajouter `--volumes` pour un arrêt normal**,
car cela effacerait modèles, historiques, dashboards et traces.

| Symptôme | Vérification |
|---|---|
| `docker` existe, mais aucun serveur ne répond | Démarrer Docker Desktop ou le service Docker Engine |
| `docker compose` absent | Installer le plugin Compose, ou compléter l'installation Docker Desktop |
| `uv` introuvable | Rouvrir le terminal et vérifier le PATH indiqué par l'installateur |
| Python n'est pas en 3.12 | Se placer dans le projet et utiliser `uv run --locked python`, pas le Python global |
| Port occupé | Modifier le port hôte dans `.env`, recréer le service et adapter les clients |
| Montage SQL/Grafana absent | Cloner le dépôt complet et lancer Compose depuis sa racine |
| Un service reste `unhealthy` | Examiner `docker compose ps` et les logs du service concerné |
| Modèles ou runs absents de l'interface | Vérifier les URI des clients et éviter un second serveur local |
| Teams/Workflows inaccessible | Vérifier le compte, les droits du tenant et le repli du TP4 |

## Poste prêt pour les TPs

- Docker répond avec un serveur actif, Compose fonctionne et `hello-world` s'exécute.
- uv trouve Python 3.12 ; `uv sync --locked` et les imports du TP1 réussissent.
- Le CSV complet a été téléchargé depuis Kaggle et `TP01_industrialisation/data/games.csv` a été généré et vérifié.
- Git fonctionne, l'éditeur ouvre le bon projet et utilise son interpréteur.
- Les interfaces MLflow, Prefect et Grafana sont accessibles ; les quatre tables PostgreSQL existent.
- L'accès au canal Teams et à Workflows est confirmé, ou son repli est identifié.

Les procédures sont basées sur les documentations officielles liées dans chaque chapitre.
Les installations sur les trois systèmes et le démarrage Docker n'ont pas été exécutés sur le
poste de rédaction ; les vérifications ci-dessus doivent être réalisées sur chaque poste étudiant.
