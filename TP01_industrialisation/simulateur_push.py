"""Producteur artificiel : exactement N tentatives POST, sans fichier de données.

Usage : uv run python simulateur_push.py 500
L'API à construire doit écouter sur http://127.0.0.1:8000/games.
"""
import argparse
import concurrent.futures
from datetime import date, datetime, timedelta, timezone
import json
import random
import secrets
import urllib.error
import urllib.request
import uuid

API_URL = 'http://127.0.0.1:8000/games'
MAX_WORKERS = 8
TIMEOUT_SECONDS = 15
OWNER_RANGES = ('0 - 20000', '20000 - 50000', '50000 - 100000', '100000 - 200000')


def generate_event(appid, rng):
    """Aucune fiche réelle : les données et leurs labels sont artificiels."""
    genres = rng.sample(['Action', 'Adventure', 'Indie', 'Strategy', 'RPG', 'Simulation'], rng.randint(1, 3))
    total = rng.randint(10, 50000)
    positive = rng.randint(0, total)
    released = date(2000, 1, 1) + timedelta(days=rng.randrange(9000))
    return {
        'event_id': str(uuid.uuid4()),
        'source': 'synthetic-course',
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'game': {
            'appid': appid, 'name': f'Jeu fictif {appid}',
            'price': rng.choice([0.0, 4.99, 9.99, 19.99, 29.99, 59.99]),
            'required_age': rng.choice([0, 12, 16, 18]),
            'dlc_count': rng.randint(0, 15), 'achievements': rng.randint(0, 150),
            'windows': True, 'mac': rng.choice([True, False]), 'linux': rng.choice([True, False]),
            'release_date': released.isoformat(), 'genres': genres,
            'pct_pos_total': round(100 * positive / total, 4),
            'num_reviews_total': total, 'estimated_owners': rng.choice(OWNER_RANGES),
        },
    }


def post_event(event):
    request = urllib.request.Request(API_URL, data=json.dumps(event).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            response.read()
            return str(response.status)
    except urllib.error.HTTPError as exc:
        exc.close()
        return str(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        # Pas de retry automatique : N signifie N tentatives HTTP, pas N succès.
        return 'network_error'


def run(count):
    rng = random.Random(secrets.randbits(128))
    # Espace synthétique de grands entiers, toujours dans un entier SQLite signé 64 bits.
    # Préfixe tiré au hasard par lancement : collisions inter-lancements très improbables.
    prefix = secrets.randbelow(2 ** 40 - 1) + 1
    base = prefix * 1_000_000
    counters = {}
    next_id = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pending = set()
        while next_id < count or pending:
            while next_id < count and len(pending) < MAX_WORKERS:
                event = generate_event(base + next_id, rng)
                pending.add(pool.submit(post_event, event))
                next_id += 1
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                status = future.result()
                counters[status] = counters.get(status, 0) + 1
    print(json.dumps({'attempted': count, 'responses': counters}, ensure_ascii=False, indent=2))
    return 0 if sum(counters.get(str(s), 0) for s in range(200, 300)) == count else 1


def number_of_calls(value):
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Un nombre entier est attendu') from exc
    if not 1 <= count <= 1_000_000:
        raise argparse.ArgumentTypeError('Choisir un nombre entre 1 et 1 000 000')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nombre_appels', type=number_of_calls)
    raise SystemExit(run(parser.parse_args().nombre_appels))
