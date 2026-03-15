"""
Configuration de l'agent PPO avec Stable-Baselines3.

Fournit les fonctions pour :
  - Créer des environnements vectorisés (SubprocVecEnv / DummyVecEnv)
  - Empiler les observations (VecFrameStack)
  - Instancier, sauvegarder et charger un agent PPO
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
)

from config.settings import (
    FRAME_STACK_SIZE,
    MODELS_DIR,
    N_ENVS,
    PPO_HYPERPARAMS,
    TENSORBOARD_LOG_DIR,
)
from env.trading_env import TradingEnv

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE SETS pour l'entraînement progressif
# =============================================================================

# V1 : OHLCV + indicateurs techniques de base
FEATURES_V1 = [
    "close", "open", "high", "low", "volume",
    "rsi", "rsi_normalized",
    "sma_50", "sma_200", "sma_trend",
]

# V2 : V1 + macro + sentiment global
FEATURES_V2 = FEATURES_V1 + [
    "qqq_close", "spy_close",
    "fear_greed_normalized", "funding_rate",
    "atr", "atr_pct",
    "bb_position", "bb_bandwidth",
    "zscore",
    "volume_ratio", "volume_direction",
    "log_return", "log_return_5h", "log_return_24h",
    "is_weekend",
]

# V3 : V2 + NLP sentiment
FEATURES_V3 = FEATURES_V2 + [
    "sentiment_score", "n_articles",
    "price_to_sma_long",
]


def make_env(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    **env_kwargs: Any,
) -> callable:
    """
    Factory function qui retourne un callable créant un TradingEnv.
    Nécessaire pour SubprocVecEnv (chaque process crée son propre env).

    Args:
        df: DataFrame avec les données de marché
        feature_columns: colonnes à utiliser comme features (None = auto)
        **env_kwargs: paramètres additionnels pour TradingEnv

    Returns:
        Callable qui crée un TradingEnv
    """
    def _init():
        return TradingEnv(
            df=df,
            feature_columns=feature_columns,
            **env_kwargs,
        )
    return _init


def make_vec_env(
    df: pd.DataFrame,
    n_envs: int = N_ENVS,
    feature_columns: Optional[list[str]] = None,
    use_subproc: bool = False,
    frame_stack: int = FRAME_STACK_SIZE,
    **env_kwargs: Any,
) -> VecFrameStack:
    """
    Crée un environnement vectorisé avec frame stacking.

    Args:
        df: DataFrame avec les données de marché
        n_envs: nombre d'environnements parallèles
        feature_columns: colonnes à utiliser comme features
        use_subproc: True = SubprocVecEnv (multi-process), False = DummyVecEnv
        frame_stack: nombre de frames empilées (24 = 1 jour H1)
        **env_kwargs: paramètres additionnels pour TradingEnv

    Returns:
        VecFrameStack wrappant les environnements vectorisés
    """
    env_fns = [
        make_env(df, feature_columns=feature_columns, **env_kwargs)
        for _ in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
        logger.info(f"SubprocVecEnv créé avec {n_envs} environnements")
    else:
        vec_env = DummyVecEnv(env_fns)
        logger.info(f"DummyVecEnv créé avec {n_envs} environnements")

    # Empiler les observations pour donner de la mémoire à l'agent
    stacked_env = VecFrameStack(vec_env, n_stack=frame_stack)
    logger.info(f"VecFrameStack: {frame_stack} frames empilées")

    return stacked_env


def create_agent(
    env: VecFrameStack,
    hyperparams: Optional[dict] = None,
    tensorboard_log: Optional[str] = TENSORBOARD_LOG_DIR,
    seed: Optional[int] = None,
) -> PPO:
    """
    Instancie un agent PPO avec les hyperparamètres configurés.

    Args:
        env: environnement vectorisé (VecFrameStack)
        hyperparams: dict d'hyperparamètres (None = utiliser config/settings.py)
        tensorboard_log: chemin pour les logs TensorBoard (None = pas de logs)
        seed: seed pour la reproductibilité

    Returns:
        Agent PPO prêt à être entraîné
    """
    params = dict(PPO_HYPERPARAMS)
    if hyperparams:
        params.update(hyperparams)

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,
        **params,
    )

    obs_shape = env.observation_space.shape
    logger.info(
        f"Agent PPO créé | obs_shape={obs_shape} | "
        f"lr={params['learning_rate']} | batch_size={params['batch_size']}"
    )

    return agent


def save_agent(agent: PPO, name: str = "ppo_trading", path: Optional[Path] = None) -> Path:
    """
    Sauvegarde l'agent PPO sur disque.

    Args:
        agent: agent PPO à sauvegarder
        name: nom du fichier (sans extension)
        path: dossier de destination (None = models/)

    Returns:
        Chemin complet du fichier sauvegardé
    """
    save_dir = path or MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / name
    agent.save(str(filepath))
    logger.info(f"Agent sauvegardé: {filepath}.zip")
    return filepath


def load_agent(
    env: VecFrameStack,
    name: str = "ppo_trading",
    path: Optional[Path] = None,
) -> PPO:
    """
    Charge un agent PPO depuis le disque.

    Args:
        env: environnement vectorisé (doit avoir le même obs_space)
        name: nom du fichier (sans extension)
        path: dossier source (None = models/)

    Returns:
        Agent PPO chargé
    """
    load_dir = path or MODELS_DIR
    filepath = load_dir / name

    zip_path = Path(f"{filepath}.zip")
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {zip_path}. "
            f"Entraînez d'abord avec: python main.py train --model {name}"
        )

    agent = PPO.load(str(filepath), env=env)
    logger.info(f"Agent chargé: {filepath}.zip")
    return agent


def get_feature_set(version: str = "v1") -> list[str]:
    """
    Retourne le set de features pour l'entraînement progressif.

    Args:
        version: "v1" (OHLCV+tech), "v2" (+macro), "v3" (+NLP)

    Returns:
        Liste des noms de colonnes
    """
    sets = {"v1": FEATURES_V1, "v2": FEATURES_V2, "v3": FEATURES_V3}
    if version not in sets:
        raise ValueError(f"Version inconnue: {version}. Choix: {list(sets.keys())}")
    return sets[version]
