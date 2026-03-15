"""
Récupération du Fear & Greed Index via l'API publique Alternative.me.
Retourne un score de sentiment global du marché crypto (0=peur extrême, 100=avidité extrême).
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from config.settings import FEAR_GREED_URL

logger = logging.getLogger(__name__)

# URL pour récupérer l'historique complet
FEAR_GREED_HISTORY_URL = "https://api.alternative.me/fng/?limit={limit}&format=json"


def fetch_fear_greed_current() -> dict:
    """
    Récupère le Fear & Greed Index actuel.

    Returns:
        Dict avec: value (0-100), label (ex: "Fear"), timestamp
    """
    try:
        response = requests.get(FEAR_GREED_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            logger.warning("Réponse Fear & Greed vide")
            return {"value": 50, "label": "Neutral", "timestamp": None}

        entry = data["data"][0]
        return {
            "value": int(entry["value"]),
            "label": entry["value_classification"],
            "timestamp": pd.to_datetime(
                int(entry["timestamp"]), unit="s", utc=True
            ),
        }

    except Exception as e:
        logger.error(f"Erreur Fear & Greed API: {e}")
        return {"value": 50, "label": "Neutral", "timestamp": None}


def fetch_fear_greed_history(
    limit: int = 0,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Récupère l'historique du Fear & Greed Index.

    Args:
        limit: Nombre de jours (0 = tout l'historique disponible)
        start: Date de début (ISO format "YYYY-MM-DD"), filtre après récupération
        end: Date de fin (ISO format "YYYY-MM-DD"), filtre après récupération

    Returns:
        DataFrame avec colonnes: timestamp, fear_greed_value, fear_greed_label
    """
    try:
        url = FEAR_GREED_HISTORY_URL.format(limit=limit)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            logger.warning("Historique Fear & Greed vide")
            return pd.DataFrame(
                columns=["timestamp", "fear_greed_value", "fear_greed_label"]
            )

        records = []
        for entry in data["data"]:
            records.append({
                "timestamp": pd.to_datetime(
                    int(entry["timestamp"]), unit="s", utc=True
                ),
                "fear_greed_value": int(entry["value"]),
                "fear_greed_label": entry["value_classification"],
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filtrer par dates si spécifié
        if start:
            start_dt = pd.Timestamp(start, tz="UTC")
            df = df[df["timestamp"] >= start_dt]
        if end:
            end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            df = df[df["timestamp"] < end_dt]

        # Normaliser le score de 0-100 à -1/+1 pour cohérence
        df["fear_greed_normalized"] = (df["fear_greed_value"] - 50) / 50.0

        logger.info(f"Récupéré {len(df)} jours de Fear & Greed Index")
        return df.reset_index(drop=True)

    except Exception as e:
        logger.error(f"Erreur historique Fear & Greed: {e}")
        return pd.DataFrame(
            columns=["timestamp", "fear_greed_value", "fear_greed_label",
                      "fear_greed_normalized"]
        )
