"""
Récupération des actualités crypto/finance via flux RSS.
Utilise feedparser pour parser CoinDesk, CoinTelegraph, Yahoo Finance.
Les titres seront ensuite analysés par FinBERT (Phase 3 — features/nlp.py).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import feedparser
import pandas as pd

from config.settings import NEWS_KEYWORDS, RSS_FEEDS

logger = logging.getLogger(__name__)


def _parse_published_date(entry: dict) -> Optional[datetime]:
    """Extrait la date de publication d'une entrée RSS."""
    for date_field in ("published_parsed", "updated_parsed"):
        parsed = entry.get(date_field)
        if parsed:
            try:
                from time import mktime
                return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
            except Exception:
                continue

    # Fallback : parser la string directement
    for date_field in ("published", "updated"):
        date_str = entry.get(date_field)
        if date_str:
            try:
                return pd.to_datetime(date_str, utc=True).to_pydatetime()
            except Exception:
                continue

    return None


def _matches_keywords(text: str, keywords: list[str] = NEWS_KEYWORDS) -> bool:
    """Vérifie si le texte contient au moins un mot-clé pertinent."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def fetch_news(
    feeds: list[str] = RSS_FEEDS,
    keywords: list[str] = NEWS_KEYWORDS,
    filter_by_keywords: bool = True,
    max_articles: Optional[int] = None,
) -> pd.DataFrame:
    """
    Récupère les titres d'articles depuis les flux RSS.

    Args:
        feeds: Liste des URLs de flux RSS
        keywords: Mots-clés pour filtrer les articles pertinents
        filter_by_keywords: Si True, ne garde que les articles avec mots-clés
        max_articles: Nombre max d'articles à retourner (None = tous)

    Returns:
        DataFrame avec colonnes: timestamp, title, source, url, has_keyword
    """
    all_articles = []

    for feed_url in feeds:
        try:
            logger.info(f"Parsing RSS: {feed_url}")
            feed = feedparser.parse(feed_url)

            if feed.bozo and not feed.entries:
                logger.warning(f"Flux RSS invalide ou vide: {feed_url}")
                continue

            # Extraire le nom de la source depuis l'URL
            source = _extract_source_name(feed_url)

            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                published = _parse_published_date(entry)
                link = entry.get("link", "")
                has_keyword = _matches_keywords(title, keywords)

                # Filtrer par mots-clés si demandé
                if filter_by_keywords and not has_keyword:
                    continue

                all_articles.append({
                    "timestamp": published,
                    "title": title,
                    "source": source,
                    "url": link,
                    "has_keyword": has_keyword,
                })

            logger.info(
                f"  → {len([a for a in all_articles if a.get('source') == source])} "
                f"articles depuis {source}"
            )

        except Exception as e:
            logger.error(f"Erreur parsing RSS {feed_url}: {e}")
            continue

    if not all_articles:
        logger.warning("Aucun article récupéré")
        return pd.DataFrame(
            columns=["timestamp", "title", "source", "url", "has_keyword"]
        )

    df = pd.DataFrame(all_articles)

    # Supprimer les articles sans date
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Trier par date décroissante (plus récent en premier)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Limiter le nombre d'articles si demandé
    if max_articles:
        df = df.head(max_articles)

    logger.info(f"Total: {len(df)} articles récupérés")
    return df


def fetch_news_titles_for_hour(
    target_hour: Optional[datetime] = None,
    feeds: list[str] = RSS_FEEDS,
    keywords: list[str] = NEWS_KEYWORDS,
) -> list[str]:
    """
    Récupère les titres d'articles pour une heure donnée.
    Utilisé par le pipeline NLP pour obtenir un score de sentiment horaire.

    Args:
        target_hour: L'heure cible (None = heure actuelle)
        feeds: Flux RSS à parser
        keywords: Mots-clés de filtrage

    Returns:
        Liste de titres d'articles pour cette heure.
    """
    if target_hour is None:
        target_hour = datetime.now(tz=timezone.utc)

    df = fetch_news(feeds, keywords, filter_by_keywords=True)

    if df.empty:
        return []

    # Filtrer les articles de l'heure cible
    hour_start = target_hour.replace(minute=0, second=0, microsecond=0)
    hour_end = hour_start + pd.Timedelta(hours=1)

    mask = (df["timestamp"] >= pd.Timestamp(hour_start, tz="UTC")) & (
        df["timestamp"] < pd.Timestamp(hour_end, tz="UTC")
    )
    hour_articles = df[mask]

    return hour_articles["title"].tolist()


def _extract_source_name(url: str) -> str:
    """Extrait un nom lisible depuis l'URL du flux RSS."""
    url_lower = url.lower()
    if "coindesk" in url_lower:
        return "CoinDesk"
    elif "cointelegraph" in url_lower:
        return "CoinTelegraph"
    elif "yahoo" in url_lower:
        return "Yahoo Finance"
    else:
        # Extraire le domaine
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
