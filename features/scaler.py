"""
Normalisation des features via RobustScaler.
Lisse toutes les données entre -1 et 1 pour l'entrée dans le modèle RL.
RobustScaler est résistant aux outliers (utilise médiane et IQR).
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

# Chemin de sauvegarde du scaler (pour réutiliser en live)
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

# Colonnes à NE PAS normaliser (timestamps, flags binaires, identifiants)
EXCLUDE_COLUMNS = [
    "timestamp",
    "is_weekend",
    "n_articles",
    "n_positive",
    "n_negative",
    "n_neutral",
    "raw_close",
]


class FeatureScaler:
    """
    Wrapper autour de RobustScaler pour normaliser les features du bot.
    Supporte la sauvegarde/chargement pour réutiliser en mode live.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns: list[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """
        Ajuste le scaler sur les données d'entraînement.

        Args:
            df: DataFrame avec les features à normaliser

        Returns:
            self (pour chaînage)
        """
        self.feature_columns = [
            c for c in df.columns if c not in EXCLUDE_COLUMNS
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        ]

        if not self.feature_columns:
            logger.warning("Aucune colonne numérique à normaliser")
            return self

        data = df[self.feature_columns].values
        # Remplacer inf par NaN avant fit
        data = np.where(np.isinf(data), np.nan, data)

        self.scaler.fit(data)
        self.is_fitted = True
        logger.info(f"Scaler ajusté sur {len(self.feature_columns)} colonnes")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les features du DataFrame.

        Args:
            df: DataFrame avec les mêmes colonnes que le fit

        Returns:
            DataFrame avec les features normalisées (clampées entre -1 et +1)
        """
        if not self.is_fitted:
            raise RuntimeError("Le scaler n'a pas été ajusté (appeler fit d'abord)")

        df = df.copy()

        # Ne transformer que les colonnes connues
        cols_to_scale = [c for c in self.feature_columns if c in df.columns]
        if not cols_to_scale:
            logger.warning("Aucune colonne à normaliser dans le DataFrame")
            return df

        data = df[cols_to_scale].values
        data = np.where(np.isinf(data), np.nan, data)

        scaled = self.scaler.transform(data)
        # Clamper entre -1 et +1
        scaled = np.clip(scaled, -1, 1)
        # Remplacer NaN par 0 après normalisation
        scaled = np.nan_to_num(scaled, nan=0.0)

        df[cols_to_scale] = scaled
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajuste et normalise en une seule étape."""
        self.fit(df)
        return self.transform(df)

    def save(self, path: Optional[Path] = None) -> None:
        """Sauvegarde le scaler sur disque."""
        path = path or SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Scaler sauvegardé dans {path}")

    def load(self, path: Optional[Path] = None) -> "FeatureScaler":
        """Charge le scaler depuis le disque."""
        path = path or SCALER_PATH
        if not path.exists():
            raise FileNotFoundError(f"Scaler non trouvé: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.scaler = state["scaler"]
        self.feature_columns = state["feature_columns"]
        self.is_fitted = state["is_fitted"]
        logger.info(
            f"Scaler chargé depuis {path} ({len(self.feature_columns)} colonnes)"
        )
        return self


def normalize_features(
    df: pd.DataFrame,
    scaler: Optional[FeatureScaler] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, FeatureScaler]:
    """
    Fonction utilitaire pour normaliser un DataFrame de features.

    Args:
        df: DataFrame avec les features à normaliser
        scaler: Scaler existant (None = en créer un nouveau)
        fit: Si True, ajuste le scaler sur les données (mode entraînement).
             Si False, utilise le scaler existant (mode live/test).

    Returns:
        Tuple (DataFrame normalisé, FeatureScaler utilisé)
    """
    if scaler is None:
        scaler = FeatureScaler()

    if fit:
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = scaler.transform(df)

    return df_scaled, scaler
