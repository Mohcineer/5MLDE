"""Client HTTP des trois sources Steam : timeout, cache, espacement, retries bornés.

Règles (voir ENONCE TP2, « Débit et cache ») :
- 1,1 s minimum entre deux appels réseau, timeout de 20 s ;
- 3 tentatives au plus, uniquement sur erreur réseau, 429 ou 5xx ; `Retry-After` respecté ;
  une pause demandée de plus de 60 s interrompt le parcours (`RetryLaterError`) ;
- aucun retry sur 4xx permanent, JSON invalide ou erreur de schéma (la même réponse reviendrait) ;
- cache relu par défaut ; `refresh` renouvelle Steam Store/avis mais SteamSpy au plus 1 fois / 24 h ;
- seuls les champs utiles sont conservés (ni texte des avis, ni profils).
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

MIN_INTERVAL = 1.1
TIMEOUT = 20.0
MAX_ATTEMPTS = 3
MAX_RETRY_AFTER = 60.0
SPY_MIN_AGE = timedelta(hours=24)

STORE_FIELDS = (
    "steam_appid", "name", "type", "is_free", "price_overview", "required_age", "dlc",
    "achievements", "platforms", "release_date", "genres",
)
SUMMARY_FIELDS = ("total_positive", "total_negative", "total_reviews", "review_score")


class AcquisitionError(RuntimeError):
    """Échec de la collecte demandée : aucun entraînement partiel ne doit suivre."""


class TransientError(AcquisitionError):
    """Panne transitoire persistante après les tentatives autorisées."""


class PermanentError(AcquisitionError):
    """404, réponse non JSON ou schéma inattendu : inutile de réessayer."""


class RetryLaterError(AcquisitionError):
    """Pause demandée trop longue : reprendre plus tard (le cache conserve l'acquis)."""


def source_request(source, appid):
    if source == "store":
        return ("https://store.steampowered.com/api/appdetails",
                {"appids": appid, "cc": "us", "l": "english"})
    if source == "spy":
        return "https://steamspy.com/api.php", {"request": "appdetails", "appid": appid}
    if source == "reviews":
        # /appreviews/<id>?json=1 (endpoint de l'énoncé) redirige vers la page d'accueil Steam lors
        # de nos essais ; le service équivalent ci-dessous renvoie le même résumé en JSON.
        # purchase_type=1 donne le total le plus large (équivalent de « all » constaté).
        query = {"appid": appid, "filter": 0, "languages": ["all"], "review_type": 0,
                 "purchase_type": 1, "num_per_page": 1}
        return ("https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/",
                {"input_json": json.dumps(query)})
    raise ValueError(f"Source inconnue : {source}")


def reduce_response(source, appid, payload):
    """Garder les seuls champs utiles ; refuser un schéma inattendu (erreur permanente)."""
    if not isinstance(payload, dict):
        raise PermanentError(f"{source} {appid} : JSON de forme inattendue.")
    if source == "store":
        entry = payload.get(str(appid))
        if not isinstance(entry, dict) or "success" not in entry:
            raise PermanentError(f"store {appid} : schéma inattendu.")
        if entry["success"] is not True:
            raise PermanentError(f"store {appid} : fiche indisponible (success=false).")
        data = entry.get("data")
        if not isinstance(data, dict):
            raise PermanentError(f"store {appid} : champ data absent.")
        return {str(appid): {"success": True,
                             "data": {k: data[k] for k in STORE_FIELDS if k in data}}}
    if source == "spy":
        if not {"appid", "owners"} <= payload.keys():
            raise PermanentError(f"spy {appid} : schéma inattendu.")
        return {k: payload[k] for k in ("appid", "name", "owners") if k in payload}
    body = payload.get("response", payload)  # forme enveloppée ou à plat
    summary = body.get("query_summary") if isinstance(body, dict) else None
    if not isinstance(summary, dict):
        raise PermanentError(f"reviews {appid} : query_summary absent.")
    reduced = {"query_summary": {k: summary[k] for k in SUMMARY_FIELDS if k in summary}}
    if "success" in body:
        reduced["success"] = body["success"]
    return reduced


class SteamClient:
    def __init__(self, cache_dir, refresh=False, transport=None, sleep=time.sleep,
                 clock=time.monotonic, now=lambda: datetime.now(timezone.utc),
                 min_interval=MIN_INTERVAL):
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self._sleep, self._clock, self._now = sleep, clock, now
        self.min_interval = min_interval
        self._last_call = None
        self.http_calls = 0
        self._http = httpx.Client(timeout=TIMEOUT, follow_redirects=True, transport=transport)

    def close(self):
        self._http.close()

    # -- cache --
    def _cache_path(self, source, appid):
        return self.cache_dir / f"{source}_{appid}.json"

    def _read_cache(self, source, appid):
        path = self._cache_path(source, appid)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache(self, source, appid, url, response):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {"source": source, "appid": appid, "url": url,
                  "fetched_at": self._now().isoformat(), "response": response}
        path = self._cache_path(source, appid)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        return record

    def _must_refresh(self, source, record):
        if not self.refresh:
            return False
        if source != "spy":
            return True
        age = self._now() - datetime.fromisoformat(record["fetched_at"])
        return age >= SPY_MIN_AGE

    # -- réseau --
    def _throttle(self):
        if self._last_call is not None:
            wait = self.min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    @staticmethod
    def _retry_after(response, default):
        value = response.headers.get("Retry-After")
        try:
            return float(value) if value is not None else default
        except ValueError:
            return default

    def _request_json(self, url, params):
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            self.http_calls += 1
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                last_error, delay = f"erreur réseau : {exc}", self.min_interval * attempt
            else:
                status = response.status_code
                if status == 429 or status >= 500:
                    last_error = f"HTTP {status}"
                    delay = self._retry_after(response, self.min_interval * attempt)
                    if delay > MAX_RETRY_AFTER:
                        raise RetryLaterError(
                            f"{url} : pause de {delay:.0f} s demandée (> {MAX_RETRY_AFTER:.0f} s).")
                elif status >= 400:
                    raise PermanentError(f"{url} : HTTP {status} (permanent, pas de retry).")
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise PermanentError(f"{url} : réponse non JSON.") from exc
            if attempt < MAX_ATTEMPTS:
                self._sleep(delay)
        raise TransientError(f"{url} : {last_error} après {MAX_ATTEMPTS} tentatives.")

    def get(self, source, appid):
        """Réponse réduite d'une source, depuis le cache ou le réseau."""
        record = self._read_cache(source, appid)
        if record is not None and not self._must_refresh(source, record):
            return record["response"]
        url, params = source_request(source, appid)
        payload = self._request_json(url, params)
        response = reduce_response(source, appid, payload)
        return self._write_cache(source, appid, url, response)["response"]

    def provenance(self, source, appid):
        record = self._read_cache(source, appid)
        return {"source": source, "url": record["url"], "fetched_at": record["fetched_at"]}
