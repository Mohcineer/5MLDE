def store_event(event):
    """Stocker durablement ou reconnaître un doublon ; réserver atomiquement un lot de 500."""
    raise NotImplementedError('SQLite : unicité, transaction et association jeu/lot')


def get_status():
    """Retourner les jeux en attente et les états des lots."""
    raise NotImplementedError('Lire les compteurs dans le stockage durable')
