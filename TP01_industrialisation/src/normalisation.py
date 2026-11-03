def normaliser_appid(appid, steam_store, steamspy, reviews):
    """Vérifie l'identité du jeu et prépare les données sources."""

    if not isinstance(appid, int) or isinstance(appid, bool) or appid <= 0:
        raise ValueError("L'appid doit être un entier positif.")

    store_data = steam_store.get("data")
    if not store_data or steam_store.get("success") is not True:
        raise ValueError("Réponse Steam Store invalide.")

    if store_data.get("steam_appid") != appid:
        raise ValueError("L'appid Steam Store ne correspond pas.")

    if store_data.get("type", "game") != "game":
        raise ValueError("L'application n'est pas un jeu (DLC, démo, outil…).")

    if steamspy.get("appid") != appid:
        raise ValueError("L'appid SteamSpy ne correspond pas.")

    # Les noms ne sont pas comparés : SteamSpy et Steam Store diffèrent souvent (™, ®, casse) ;
    # l'identité est portée par l'appid, vérifié dans les trois sources.
    inner = reviews.get("response", reviews)
    if not isinstance(inner, dict) or inner.get("success", 1) not in (1, True):
        raise ValueError("Réponse des avis non réussie.")

    if steamspy.get("owners") == "999999":
        raise ValueError("Identifiant SteamSpy caché ou invalide.")

    return True

def normaliser_prix(store_data):
    """Convertit le prix Steam en dollars, à partir des centimes."""

    if store_data.get("is_free") is True:
        return 0.0

    price = store_data.get("price_overview")

    if price is None:
        return None

    if not isinstance(price, dict):
        raise ValueError("Format du prix invalide.")

    final_price = price.get("final")
    currency = price.get("currency")

    if not isinstance(final_price, int) or isinstance(final_price, bool):
        raise ValueError("Prix final invalide.")

    if final_price < 0:
        raise ValueError("Le prix ne peut pas être négatif.")

    if currency != "USD":
        raise ValueError("La devise reçue n'est pas USD.")

    return final_price / 100

def normaliser_date(store_data):
    """Convertit la date Steam en format ISO."""

    release = store_data.get("release_date")

    if not isinstance(release, dict):
        raise ValueError("Date de sortie absente.")

    if release.get("coming_soon") is True:
        raise ValueError("Le jeu n'est pas encore publié.")

    date_brute = release.get("date")

    if not isinstance(date_brute, str) or not date_brute.strip():
        raise ValueError("Date de sortie invalide.")

    from datetime import datetime

    formats = ["%b %d, %Y", "%d %b, %Y", "%b %Y", "%Y"]

    for format_date in formats:
        try:
            return datetime.strptime(date_brute.strip(), format_date).date().isoformat()
        except ValueError:
            continue

    raise ValueError(f"Format de date non reconnu : {date_brute}")

def normaliser_plateformes(store_data):
    """Vérifie et récupère les compatibilités des plateformes."""

    platforms = store_data.get("platforms")

    if not isinstance(platforms, dict):
        raise ValueError("Informations sur les plateformes absentes.")

    resultat = {}

    for plateforme in ("windows", "mac", "linux"):
        valeur = platforms.get(plateforme)

        if not isinstance(valeur, bool):
            raise ValueError(f"Valeur invalide pour {plateforme}.")

        resultat[plateforme] = valeur

    return resultat

def normaliser_genres(store_data):
    """Récupère les genres et les sérialise au format texte."""

    genres = store_data.get("genres", [])

    if not isinstance(genres, list):
        raise ValueError("Format des genres invalide.")

    resultat = []

    for genre in genres:
        if not isinstance(genre, dict):
            raise ValueError("Un genre possède un format invalide.")

        description = genre.get("description")

        if not isinstance(description, str) or not description.strip():
            raise ValueError("Description de genre invalide.")

        resultat.append(description.strip())

    return str(resultat) if resultat else "Unknown"

def normaliser_age(store_data):
    """Récupère et vérifie l'âge requis."""

    age = store_data.get("required_age")

    if age is None:
        return None

    if not isinstance(age, int) or isinstance(age, bool) or age < 0:
        raise ValueError("Âge requis invalide.")

    return age

def normaliser_statistiques(store_data):
    """Récupère le nombre de DLC et de succès."""

    dlc = store_data.get("dlc", [])

    if not isinstance(dlc, list):
        raise ValueError("Liste des DLC invalide.")

    dlc_count = len(dlc)

    achievements_data = store_data.get("achievements")
    achievements = None

    if achievements_data is not None:
        if not isinstance(achievements_data, dict):
            raise ValueError("Format des succès invalide.")

        achievements = achievements_data.get("total")

        if achievements is not None:
            if not isinstance(achievements, int) or isinstance(achievements, bool) or achievements < 0:
                raise ValueError("Nombre de succès invalide.")

    return {
        "dlc_count": dlc_count,
        "achievements": achievements
    }

def normaliser_avis(reviews):
    """Vérifie le résumé des avis et calcule le pourcentage positif."""

    # Deux formes : /appreviews/<id>?json=1 (à plat) ou GetAppReviews/v1 (sous "response").
    response = reviews.get("response", reviews)

    if not isinstance(response, dict):
        raise ValueError("Réponse des avis invalide.")

    summary = response.get("query_summary")

    if not isinstance(summary, dict):
        raise ValueError("Résumé des avis absent.")

    positive = summary.get("total_positive")
    negative = summary.get("total_negative")
    total = summary.get("total_reviews")

    valeurs = (positive, negative, total)

    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in valeurs):
        raise ValueError("Compteurs d'avis invalides.")

    if total < 10:
        raise ValueError("Le jeu doit avoir au moins 10 avis.")

    if positive + negative != total:
        raise ValueError("Les compteurs d'avis ne correspondent pas au total.")

    pct_pos_total = positive / total * 100

    return {
        "positive": positive,
        "negative": negative,
        "num_reviews_total": total,
        "pct_pos_total": pct_pos_total
    }

def normaliser_owners(steamspy):
    """Convertit la tranche SteamSpy au format attendu par TP1."""

    owners = steamspy.get("owners")

    if not isinstance(owners, str) or not owners.strip():
        raise ValueError("Tranche de propriétaires absente.")

    if owners == "999999":
        raise ValueError("Tranche de propriétaires indisponible.")

    import re

    match = re.fullmatch(
        r"\s*([\d,]+)\s*\.\.\s*([\d,]+)\s*",
        owners
    )

    if not match:
        raise ValueError("Format de tranche de propriétaires invalide.")

    minimum = int(match.group(1).replace(",", ""))
    maximum = int(match.group(2).replace(",", ""))

    if minimum < 0 or maximum < minimum or (minimum == 0 and maximum == 0):
        raise ValueError("Bornes de propriétaires invalides.")

    return f"{minimum} - {maximum}"

def normaliser_ligne(appid, steam_store, steamspy, reviews):

    store_response = steam_store.get(str(appid))

    if not isinstance(store_response, dict):
        raise ValueError("Réponse Steam Store absente.")

    # Vérification de l'identité.
    normaliser_appid(appid, store_response, steamspy, reviews)

    store_data = store_response["data"]

    # Normalisation des différentes colonnes.
    statistiques = normaliser_statistiques(store_data)
    avis = normaliser_avis(reviews)
    plateformes = normaliser_plateformes(store_data)

    ligne = {
        "appid": appid,
        "name": store_data["name"],
        "release_date": normaliser_date(store_data),
        "required_age": normaliser_age(store_data),
        "price": normaliser_prix(store_data),
        "dlc_count": statistiques["dlc_count"],
        "achievements": statistiques["achievements"],
        "genres": normaliser_genres(store_data),
        "estimated_owners": normaliser_owners(steamspy),
        **plateformes,
        **avis,
    }

    return ligne