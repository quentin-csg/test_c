"""
Récupération des données macroéconomiques via yfinance.
Nasdaq (QQQ) et S&P500 (SPY) — gestion du weekend incluse.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import MACRO_SYMBOLS

logger = logging.getLogger(__name__)


def fetch_macro_data(
    symbols: list[str] = MACRO_SYMBOLS,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Récupère les données macro (QQQ, SPY) via yfinance.

    Le weekend et les jours fériés, les marchés sont fermés.
    On forward-fill les valeurs manquantes et on ajoute un flag is_weekend.

    Args:
        symbols: Liste des tickers (ex: ["QQQ", "SPY"])
        start: Date de début (ISO format "YYYY-MM-DD")
        end: Date de fin (ISO format "YYYY-MM-DD"), None = aujourd'hui
        interval: Intervalle des données ("1h", "1d")

    Returns:
        DataFrame avec colonnes par symbol + is_weekend flag
    """
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    all_data = []

    for symbol in symbols:
        logger.info(f"Fetching macro {symbol} ({start} → {end})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                logger.warning(f"Aucune donnée macro pour {symbol}")
                continue

            # Normaliser les colonnes
            df = df.reset_index()
            # yfinance utilise "Datetime" pour intraday, "Date" pour daily
            time_col = "Datetime" if "Datetime" in df.columns else "Date"
            df = df.rename(columns={time_col: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            prefix = symbol.lower()
            result = pd.DataFrame({
                "timestamp": df["timestamp"],
                f"{prefix}_close": df["Close"].values,
                f"{prefix}_volume": df["Volume"].values,
                f"{prefix}_high": df["High"].values,
                f"{prefix}_low": df["Low"].values,
            })

            all_data.append(result)
            logger.info(f"Récupéré {len(result)} lignes pour {symbol}")

        except Exception as e:
            logger.error(f"Erreur fetching {symbol}: {e}")
            continue

    if not all_data:
        logger.warning("Aucune donnée macro récupérée")
        return pd.DataFrame(columns=["timestamp"])

    # Merge toutes les sources macro sur le timestamp
    merged = all_data[0]
    for df in all_data[1:]:
        merged = pd.merge(merged, df, on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Ajouter le flag weekend (samedi=5, dimanche=6)
    merged["is_weekend"] = merged["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    # Forward-fill les données manquantes (weekend/jours fériés)
    data_cols = [c for c in merged.columns if c not in ("timestamp", "is_weekend")]
    merged[data_cols] = merged[data_cols].ffill()

    logger.info(f"Données macro finales: {len(merged)} lignes, {len(merged.columns)} colonnes")
    return merged


def fetch_macro_daily(
    symbols: list[str] = MACRO_SYMBOLS,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Récupère les données macro en daily (plus fiable pour le long terme).
    Utile pour l'entraînement sur 2020-2023 (yfinance limite l'intraday à ~2 ans).

    Returns:
        DataFrame avec données daily, forward-filled.
    """
    return fetch_macro_data(symbols, start, end, interval="1d")
