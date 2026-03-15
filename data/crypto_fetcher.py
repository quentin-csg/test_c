"""
Récupération des données OHLCV crypto via ccxt.
Utilise l'API publique Binance/Bybit, timeframe 1h.
"""

import logging
from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd

from config.settings import (
    API_KEY,
    API_SECRET,
    EXCHANGE,
    SYMBOL,
    TIMEFRAME,
    TIMEFRAME_SECONDARY,
)

logger = logging.getLogger(__name__)


def _get_exchange(exchange_id: str = EXCHANGE) -> ccxt.Exchange:
    """Instancie l'exchange ccxt (API publique ou authentifiée)."""
    exchange_class = getattr(ccxt, exchange_id)
    config = {"enableRateLimit": True}
    if API_KEY and API_SECRET:
        config["apiKey"] = API_KEY
        config["secret"] = API_SECRET
    return exchange_class(config)


def fetch_ohlcv(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    since: Optional[str] = None,
    until: Optional[str] = None,
    exchange_id: str = EXCHANGE,
) -> pd.DataFrame:
    """
    Récupère les données OHLCV (Open, High, Low, Close, Volume) pour un symbol.

    Args:
        symbol: Paire de trading (ex: "BTC/USDT")
        timeframe: Timeframe des bougies (ex: "1h", "4h")
        since: Date de début (ISO format "YYYY-MM-DD")
        until: Date de fin (ISO format "YYYY-MM-DD"), None = maintenant
        exchange_id: Identifiant de l'exchange ccxt

    Returns:
        DataFrame avec colonnes: timestamp, open, high, low, close, volume
    """
    exchange = _get_exchange(exchange_id)

    since_ms = None
    if since:
        since_ms = exchange.parse8601(f"{since}T00:00:00Z")

    until_ms = None
    if until:
        until_ms = exchange.parse8601(f"{until}T23:59:59Z")

    all_ohlcv = []
    logger.info(f"Fetching {symbol} {timeframe} depuis {since or 'le début'}...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=1000
            )
        except Exception as e:
            logger.error(f"Erreur fetch_ohlcv {symbol}: {e}")
            break
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)

        # Avancer le curseur après la dernière bougie récupérée
        last_timestamp = ohlcv[-1][0]
        if since_ms and last_timestamp == since_ms:
            break
        since_ms = last_timestamp + 1

        # Arrêter si on dépasse la date de fin
        if until_ms and last_timestamp >= until_ms:
            break

        # Moins de 1000 résultats = on a tout récupéré
        if len(ohlcv) < 1000:
            break

    if not all_ohlcv:
        logger.warning(f"Aucune donnée OHLCV pour {symbol} {timeframe}")
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    # Filtrer jusqu'à la date de fin
    if until_ms:
        until_dt = pd.Timestamp(until_ms, unit="ms", tz="UTC")
        df = df[df["timestamp"] <= until_dt]

    logger.info(f"Récupéré {len(df)} bougies {symbol} {timeframe}")
    return df


def fetch_multi_timeframe(
    symbol: str = SYMBOL,
    since: Optional[str] = None,
    until: Optional[str] = None,
    exchange_id: str = EXCHANGE,
) -> dict[str, pd.DataFrame]:
    """
    Récupère les données OHLCV sur les deux timeframes (1h et 4h).

    Returns:
        Dict avec clés "1h" et "4h", chaque valeur est un DataFrame OHLCV.
    """
    return {
        TIMEFRAME: fetch_ohlcv(symbol, TIMEFRAME, since, until, exchange_id),
        TIMEFRAME_SECONDARY: fetch_ohlcv(
            symbol, TIMEFRAME_SECONDARY, since, until, exchange_id
        ),
    }


def fetch_funding_rate(
    symbol: str = SYMBOL,
    since: Optional[str] = None,
    exchange_id: str = EXCHANGE,
) -> pd.DataFrame:
    """
    Récupère le funding rate (taux de financement) des contrats à terme.
    Convertit automatiquement le symbole spot en symbole futures si nécessaire.

    Returns:
        DataFrame avec colonnes: timestamp, funding_rate
    """
    exchange = _get_exchange(exchange_id)

    # Convertir le symbole spot en futures (ex: BTC/USDT → BTC/USDT:USDT)
    futures_symbol = symbol
    if ":" not in symbol:
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        futures_symbol = f"{symbol}:{quote}"

    since_ms = None
    if since:
        since_ms = exchange.parse8601(f"{since}T00:00:00Z")

    try:
        if hasattr(exchange, "fetch_funding_rate_history"):
            rates = exchange.fetch_funding_rate_history(
                futures_symbol, since=since_ms, limit=1000
            )
        else:
            logger.warning(
                f"{exchange_id} ne supporte pas fetch_funding_rate_history"
            )
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
    except Exception as e:
        logger.error(f"Erreur funding rate: {e}")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    if not rates:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
            "funding_rate": r.get("fundingRate", 0.0),
        }
        for r in rates
    ])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def fetch_order_book(
    symbol: str = SYMBOL,
    exchange_id: str = EXCHANGE,
    depth: int = 20,
) -> dict:
    """
    Récupère le carnet d'ordres L2 et calcule l'imbalance.

    Returns:
        Dict avec: bid_volume, ask_volume, imbalance (-1 à +1), mid_price
    """
    exchange = _get_exchange(exchange_id)

    try:
        ob = exchange.fetch_order_book(symbol, limit=depth)
    except Exception as e:
        logger.error(f"Erreur order book: {e}")
        return {
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "imbalance": 0.0,
            "mid_price": 0.0,
        }

    bid_volume = sum(bid[1] for bid in ob.get("bids", []))
    ask_volume = sum(ask[1] for ask in ob.get("asks", []))

    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total if total > 0 else 0.0

    best_bid = ob["bids"][0][0] if ob.get("bids") else 0.0
    best_ask = ob["asks"][0][0] if ob.get("asks") else 0.0
    mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0

    return {
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "imbalance": imbalance,
        "mid_price": mid_price,
    }


def fetch_open_interest(
    symbol: str = SYMBOL,
    exchange_id: str = EXCHANGE,
) -> float:
    """
    Récupère l'Open Interest (positions ouvertes sur les contrats à terme).
    Convertit automatiquement le symbole spot en symbole futures si nécessaire.

    Returns:
        Open interest en unités de base (ex: BTC)
    """
    exchange = _get_exchange(exchange_id)

    # Convertir le symbole spot en futures (ex: BTC/USDT → BTC/USDT:USDT)
    futures_symbol = symbol
    if ":" not in symbol:
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        futures_symbol = f"{symbol}:{quote}"

    try:
        if hasattr(exchange, "fetch_open_interest"):
            oi = exchange.fetch_open_interest(futures_symbol)
            return float(oi.get("openInterestAmount", 0.0))
        else:
            logger.warning(f"{exchange_id} ne supporte pas fetch_open_interest")
            return 0.0
    except Exception as e:
        logger.error(f"Erreur open interest: {e}")
        return 0.0
