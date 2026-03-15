"""
Pipeline d'ingestion des données — Orchestrateur.
Assemble toutes les sources (crypto, macro, sentiment, news) en un seul DataFrame
aligné temporellement sur le timeframe 1h.
"""

import logging
from typing import Optional

import pandas as pd

from config.settings import (
    EXCHANGE,
    MACRO_SYMBOLS,
    SYMBOL,
    TIMEFRAME,
    TRAIN_END,
    TRAIN_START,
)
from data.crypto_fetcher import fetch_funding_rate, fetch_ohlcv
from data.macro_fetcher import fetch_macro_daily
from data.sentiment_fetcher import fetch_fear_greed_history
from data.news_fetcher import fetch_news

logger = logging.getLogger(__name__)


def build_dataset(
    symbol: str = SYMBOL,
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_news: bool = True,
    exchange_id: str = EXCHANGE,
) -> pd.DataFrame:
    """
    Construit le dataset complet en assemblant toutes les sources de données.

    Processus:
    1. Récupère les données OHLCV crypto (1h)
    2. Récupère les données macro (daily, forward-fill sur 1h)
    3. Récupère le Fear & Greed Index (daily, forward-fill sur 1h)
    4. Récupère les titres des news (pour NLP en Phase 3)
    5. Merge tout sur le timestamp 1h

    Args:
        symbol: Paire de trading
        start: Date de début (ISO "YYYY-MM-DD")
        end: Date de fin (ISO "YYYY-MM-DD"), None = aujourd'hui
        include_news: Si True, inclut les titres des news
        exchange_id: Exchange ccxt à utiliser

    Returns:
        DataFrame aligné temporellement avec toutes les sources.
    """
    if start is None:
        start = TRAIN_START
    if end is None:
        end = TRAIN_END

    logger.info(f"=== Construction du dataset {symbol} ({start} → {end}) ===")

    # -------------------------------------------------------------------------
    # 1. Données crypto OHLCV (1h) — Source principale
    # -------------------------------------------------------------------------
    logger.info("1/4 — Récupération OHLCV crypto...")
    df_crypto = fetch_ohlcv(symbol, TIMEFRAME, since=start, until=end,
                            exchange_id=exchange_id)

    if df_crypto.empty:
        logger.error("Aucune donnée crypto. Impossible de construire le dataset.")
        return pd.DataFrame()

    # Arrondir les timestamps à l'heure (pour alignement)
    df_crypto["timestamp"] = df_crypto["timestamp"].dt.floor("h")
    df_crypto = df_crypto.drop_duplicates(subset=["timestamp"], keep="last")

    logger.info(f"   → {len(df_crypto)} bougies crypto")

    # -------------------------------------------------------------------------
    # 2. Données macro (daily → forward-fill sur grille 1h)
    # -------------------------------------------------------------------------
    logger.info("2/4 — Récupération données macro...")
    df_macro = fetch_macro_daily(MACRO_SYMBOLS, start=start, end=end)

    if not df_macro.empty:
        df_macro["timestamp"] = df_macro["timestamp"].dt.floor("D")
        df_macro = df_macro.drop_duplicates(subset=["timestamp"], keep="last")
        df_macro = _resample_daily_to_hourly(df_macro, df_crypto["timestamp"])
        logger.info(f"   → {len(df_macro)} lignes macro (resamplées)")
    else:
        logger.warning("   → Aucune donnée macro disponible")

    # -------------------------------------------------------------------------
    # 3. Fear & Greed Index (daily → forward-fill sur grille 1h)
    # -------------------------------------------------------------------------
    logger.info("3/4 — Récupération Fear & Greed Index...")
    df_sentiment = fetch_fear_greed_history(limit=0, start=start, end=end)

    if not df_sentiment.empty:
        df_sentiment["timestamp"] = df_sentiment["timestamp"].dt.floor("D")
        df_sentiment = df_sentiment.drop_duplicates(
            subset=["timestamp"], keep="last"
        )
        # Ne garder que les colonnes numériques utiles
        sentiment_cols = ["timestamp", "fear_greed_value", "fear_greed_normalized"]
        df_sentiment = df_sentiment[[
            c for c in sentiment_cols if c in df_sentiment.columns
        ]]
        df_sentiment = _resample_daily_to_hourly(
            df_sentiment, df_crypto["timestamp"]
        )
        logger.info(f"   → {len(df_sentiment)} lignes sentiment (resamplées)")
    else:
        logger.warning("   → Aucune donnée sentiment disponible")

    # -------------------------------------------------------------------------
    # 4. Funding rate (si disponible)
    # -------------------------------------------------------------------------
    logger.info("4/4 — Récupération Funding Rate...")
    df_funding = fetch_funding_rate(symbol, since=start, exchange_id=exchange_id)

    if not df_funding.empty:
        df_funding["timestamp"] = df_funding["timestamp"].dt.floor("h")
        df_funding = df_funding.drop_duplicates(
            subset=["timestamp"], keep="last"
        )
        logger.info(f"   → {len(df_funding)} entrées funding rate")
    else:
        logger.warning("   → Aucun funding rate disponible")

    # -------------------------------------------------------------------------
    # Merge final : tout sur la grille temporelle crypto (1h)
    # -------------------------------------------------------------------------
    logger.info("Merge de toutes les sources...")
    dataset = df_crypto.copy()

    if not df_macro.empty:
        dataset = pd.merge(dataset, df_macro, on="timestamp", how="left")

    if not df_sentiment.empty:
        dataset = pd.merge(dataset, df_sentiment, on="timestamp", how="left")

    if not df_funding.empty:
        dataset = pd.merge(dataset, df_funding, on="timestamp", how="left")

    # Ajouter le flag weekend
    dataset["is_weekend"] = dataset["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    # Forward-fill les colonnes avec données manquantes (macro, sentiment)
    cols_to_fill = [c for c in dataset.columns
                    if c not in ("timestamp", "is_weekend")]
    dataset[cols_to_fill] = dataset[cols_to_fill].ffill()

    # Backward-fill les premières lignes si nécessaire
    dataset[cols_to_fill] = dataset[cols_to_fill].bfill()

    dataset = dataset.reset_index(drop=True)

    logger.info(
        f"=== Dataset final: {len(dataset)} lignes, "
        f"{len(dataset.columns)} colonnes ==="
    )
    logger.info(f"    Colonnes: {list(dataset.columns)}")
    logger.info(
        f"    Période: {dataset['timestamp'].min()} → {dataset['timestamp'].max()}"
    )

    return dataset


def _resample_daily_to_hourly(
    df_daily: pd.DataFrame,
    hourly_index: pd.Series,
) -> pd.DataFrame:
    """
    Resample des données daily sur la grille horaire crypto.
    Utilise merge_asof pour assigner chaque heure à sa valeur daily la plus récente.

    Args:
        df_daily: DataFrame avec colonne 'timestamp' (daily)
        hourly_index: Series de timestamps horaires (référence crypto)

    Returns:
        DataFrame avec les mêmes colonnes, aligné sur les timestamps horaires.
    """
    df_hourly = pd.DataFrame({"timestamp": hourly_index})
    df_daily = df_daily.sort_values("timestamp")
    df_hourly = df_hourly.sort_values("timestamp")

    result = pd.merge_asof(
        df_hourly,
        df_daily,
        on="timestamp",
        direction="backward",
    )
    return result


def get_news_for_dataset(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Récupère les news et les agrège par heure (nombre d'articles).
    Les titres sont conservés pour l'analyse NLP en Phase 3.

    Returns:
        DataFrame avec colonnes: timestamp, news_count, news_titles (liste)
    """
    df_news = fetch_news(filter_by_keywords=True)

    if df_news.empty:
        return pd.DataFrame(columns=["timestamp", "news_count", "news_titles"])

    # Arrondir à l'heure
    df_news["timestamp"] = df_news["timestamp"].dt.floor("h")

    # Agréger par heure
    grouped = df_news.groupby("timestamp").agg(
        news_count=("title", "count"),
        news_titles=("title", list),
    ).reset_index()

    # Filtrer par dates
    if start:
        start_dt = pd.Timestamp(start, tz="UTC")
        grouped = grouped[grouped["timestamp"] >= start_dt]
    if end:
        end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        grouped = grouped[grouped["timestamp"] < end_dt]

    return grouped.reset_index(drop=True)


def build_full_pipeline(
    symbol: str = SYMBOL,
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_nlp: bool = True,
    fit_scaler: bool = True,
    exchange_id: str = EXCHANGE,
) -> tuple:
    """
    Pipeline complet : données brutes → features → normalisation.
    Relie Phase 2 (data) et Phase 3 (features) en un seul appel.

    Args:
        symbol: Paire de trading
        start: Date de début
        end: Date de fin
        include_nlp: Si True, analyse NLP FinBERT sur les news
        fit_scaler: Si True, ajuste le scaler (train). False = réutilise (live).
        exchange_id: Exchange ccxt

    Returns:
        Tuple (DataFrame normalisé, FeatureScaler)
    """
    from features.nlp import add_sentiment_to_dataframe
    from features.scaler import FeatureScaler, normalize_features
    from features.technical import add_all_indicators

    # Étape 1 : Construire le dataset brut (Phase 2)
    logger.info("=== Pipeline complet : données → features → normalisation ===")
    dataset = build_dataset(symbol, start, end, exchange_id=exchange_id)

    if dataset.empty:
        logger.error("Dataset vide, pipeline interrompu")
        return pd.DataFrame(), FeatureScaler()

    # Étape 2 : Ajouter les indicateurs techniques (Phase 3)
    logger.info("Ajout des indicateurs techniques...")
    dataset = add_all_indicators(dataset)

    # Étape 3 : Ajouter le sentiment NLP (Phase 3)
    if include_nlp:
        logger.info("Analyse NLP des news...")
        news_df = fetch_news(filter_by_keywords=True)
        dataset = add_sentiment_to_dataframe(dataset, news_df)
    else:
        dataset["sentiment_score"] = 0.0
        dataset["n_articles"] = 0

    # Étape 4 : Supprimer les lignes avec trop de NaN (début de série)
    # Les indicateurs comme SMA_200 ont besoin de 200 bougies avant d'être valides
    initial_len = len(dataset)
    dataset = dataset.dropna(thresh=len(dataset.columns) - 5)
    dataset = dataset.reset_index(drop=True)
    dropped = initial_len - len(dataset)
    if dropped > 0:
        logger.info(f"Supprimé {dropped} lignes avec trop de NaN (warmup indicateurs)")

    # Sauvegarder le prix brut avant normalisation (pour le live trading)
    dataset["raw_close"] = dataset["close"].copy()

    # Étape 5 : Normalisation (Phase 3)
    logger.info("Normalisation des features...")
    if fit_scaler:
        dataset, scaler = normalize_features(dataset, fit=True)
    else:
        # Charger le scaler sauvegardé lors de l'entraînement
        saved_scaler = FeatureScaler()
        try:
            saved_scaler.load()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Scaler non trouvé dans models/feature_scaler.pkl. "
                "Entraînez d'abord le modèle avec: python main.py train"
            )
        dataset, scaler = normalize_features(dataset, scaler=saved_scaler, fit=False)

    logger.info(
        f"=== Pipeline terminé: {len(dataset)} lignes, "
        f"{len(dataset.columns)} colonnes ==="
    )
    return dataset, scaler
