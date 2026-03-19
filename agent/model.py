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
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from config.settings import (
    CNN_FEATURES_DIM,
    FRAME_STACK_SIZE,
    MODELS_DIR,
    N_ENVS,
    PPO_HYPERPARAMS,
    TENSORBOARD_LOG_DIR,
    USE_CNN,
    VECNORM_PATH,
)
from env.trading_env import TradingEnv

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE SETS — Entraînement progressif (sans doublons)
# =============================================================================

# V1 : Tech de base — momentum + tendance + volatilité (7 features)
FEATURES_V1 = [
    "rsi_normalized",        # Momentum / surachat-survente (-1 à +1)
    "sma_trend",             # Direction de tendance long terme (+1/-1)
    "price_to_sma_long",     # Distance au trend long terme
    "atr_pct",               # Volatilité normalisée (% du prix)
    "bb_position",           # Position dans les bandes de Bollinger
    "bb_bandwidth",          # Compression de volatilité (squeeze)
    "log_return",            # Return 1h (momentum court terme)
]

# V2 : + Volume + Momentum multi-horizon + Macro (14 features)
FEATURES_V2 = FEATURES_V1 + [
    "volume_ratio",          # Anomalies de volume (vs moyenne 20p)
    "volume_direction",      # Volume × direction du prix
    "log_return_5h",         # Return 5h (momentum moyen terme)
    "log_return_24h",        # Return 24h (momentum long terme)
    "fear_greed_normalized", # Sentiment global du marché (-1 à +1)
    "funding_rate",          # Positionnement long/short du marché
    "is_weekend",            # Régime de marché (weekend/semaine)
]

# V3 : V2 + MACD + ADX + Multi-timeframe + Candles + Price position (23 features)
FEATURES_V3 = FEATURES_V2 + [
    "macd_hist_normalized",  # Momentum + signal de retournement (scale-free)
    "adx_normalized",        # Force de la tendance (0-1, indépendant de direction)
    "rsi_4h_normalized",     # RSI 4h — contexte macro momentum
    "sma_trend_4h",          # Tendance long terme 4h (+1/-1)
    "candle_body",           # Corps de bougie normalisé (-1 bearish, +1 bullish)
    "upper_wick_ratio",      # Rejet de résistance (0 à 1)
    "lower_wick_ratio",      # Rejet de support (0 à 1)
    "price_to_high_20",      # Distance au plus haut 20p — résistances (≤ 0)
    "price_to_low_20",       # Distance au plus bas 20p — supports (≥ 0)
]

# Alias pour accès direct au set complet (V3 = toutes les features)
MODEL_FEATURES = FEATURES_V3

# Features live-only (pas d'historique → ne pas inclure dans le training par défaut)
FEATURES_LIVE_ONLY = [
    "orderbook_imbalance",  # Imbalance bid/ask en temps réel
    "oi_change_pct",        # Variation d'Open Interest (futures)
]


# =============================================================================
# CNN 1D — Feature Extractor pour patterns temporels
# =============================================================================

class CustomCNN1D(BaseFeaturesExtractor):
    """
    CNN 1D feature extractor pour détecter des patterns temporels
    dans les observations empilées par VecFrameStack.

    Input:  (batch, n_stack * n_features) — vecteur aplati par VecFrameStack
    Output: (batch, features_dim) — features extraites pour le policy head PPO
    """

    def __init__(self, observation_space, n_features: int, n_stack: int,
                 features_dim: int = CNN_FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        self.n_features = n_features
        self.n_stack = n_stack

        # Conv1d: in_channels=n_features, séquence=n_stack
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pool temporel → (batch, 64, 1)
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # Reshape: (batch, n_stack * n_features) → (batch, n_stack, n_features)
        x = observations.reshape(batch_size, self.n_stack, self.n_features)
        # Conv1d attend (batch, channels, seq_len) → transpose
        x = x.permute(0, 2, 1)  # → (batch, n_features, n_stack)
        x = self.cnn(x)
        x = self.fc(x)
        return x


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
    normalize: bool = True,
    **env_kwargs: Any,
) -> VecFrameStack:
    """
    Crée un environnement vectorisé avec normalisation des rewards et frame stacking.

    Pipeline: DummyVecEnv/SubprocVecEnv → VecNormalize → VecFrameStack

    Args:
        df: DataFrame avec les données de marché
        n_envs: nombre d'environnements parallèles
        feature_columns: colonnes à utiliser comme features
        use_subproc: True = SubprocVecEnv (multi-process), False = DummyVecEnv
        frame_stack: nombre de frames empilées (24 = 1 jour H1)
        normalize: True = ajouter VecNormalize (norm_obs=False, norm_reward=True)
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

    # Normalisation des rewards (obs déjà normalisées par FeatureScaler)
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        # SB3 >= 2.x ne crée obs_rms que si norm_obs=True.
        # Forcer l'attribut à None pour éviter AttributeError dans getattr_recursive.
        if not hasattr(vec_env, "obs_rms"):
            vec_env.obs_rms = None
        logger.info("VecNormalize appliqué (norm_obs=False, norm_reward=True)")

    # Empiler les observations pour donner de la mémoire à l'agent
    stacked_env = VecFrameStack(vec_env, n_stack=frame_stack)
    logger.info(f"VecFrameStack: {frame_stack} frames empilées")

    return stacked_env


def create_agent(
    env: VecFrameStack,
    hyperparams: Optional[dict] = None,
    tensorboard_log: Optional[str] = TENSORBOARD_LOG_DIR,
    seed: Optional[int] = None,
    use_cnn: bool = USE_CNN,
    n_features: Optional[int] = None,
    frame_stack: int = FRAME_STACK_SIZE,
) -> PPO:
    """
    Instancie un agent PPO avec les hyperparamètres configurés.

    Args:
        env: environnement vectorisé (VecFrameStack)
        hyperparams: dict d'hyperparamètres (None = utiliser config/settings.py)
        tensorboard_log: chemin pour les logs TensorBoard (None = pas de logs)
        seed: seed pour la reproductibilité
        use_cnn: True = CNN 1D extractor, False = MLP classique
        n_features: nombre de features par observation (pour CNN reshape)
        frame_stack: nombre de frames empilées (pour CNN reshape)

    Returns:
        Agent PPO prêt à être entraîné
    """
    params = dict(PPO_HYPERPARAMS)
    if hyperparams:
        params.update(hyperparams)

    policy_kwargs = {}
    if use_cnn:
        # Déduire n_features depuis l'obs space si non fourni
        obs_dim = env.observation_space.shape[0]
        if n_features is None:
            n_features = obs_dim // frame_stack
        policy_kwargs = dict(
            features_extractor_class=CustomCNN1D,
            features_extractor_kwargs=dict(
                n_features=n_features,
                n_stack=frame_stack,
                features_dim=CNN_FEATURES_DIM,
            ),
        )
        logger.info(
            f"CNN 1D activé: {n_features} features × {frame_stack} frames "
            f"→ {CNN_FEATURES_DIM}d output"
        )

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        **params,
    )

    obs_shape = env.observation_space.shape
    arch = "CNN 1D" if use_cnn else "MLP"
    logger.info(
        f"Agent PPO créé [{arch}] | obs_shape={obs_shape} | "
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


def _get_vec_normalize(vec_env) -> Optional[VecNormalize]:
    """Traverse la wrapper chain pour trouver le VecNormalize."""
    env = vec_env
    while env is not None:
        if isinstance(env, VecNormalize):
            return env
        env = getattr(env, "venv", None)
    return None


def save_vec_normalize(vec_env, path: Optional[Path] = None) -> Path:
    """
    Sauvegarde les stats de VecNormalize (running mean/var des rewards).

    Args:
        vec_env: environnement vectorisé (VecFrameStack wrappant VecNormalize)
        path: chemin du fichier (None = VECNORM_PATH)

    Returns:
        Chemin du fichier sauvegardé
    """
    save_path = path or VECNORM_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vn = _get_vec_normalize(vec_env)
    if vn is None:
        logger.warning("Pas de VecNormalize trouvé dans la wrapper chain")
        return save_path

    vn.save(str(save_path))
    logger.info(f"VecNormalize stats sauvegardées: {save_path}")
    return save_path


def load_vec_normalize(vec_env, path: Optional[Path] = None) -> None:
    """
    Charge les stats de VecNormalize et configure en mode eval.

    Args:
        vec_env: environnement vectorisé contenant un VecNormalize
        path: chemin du fichier (None = VECNORM_PATH)
    """
    load_path = path or VECNORM_PATH
    if not load_path.exists():
        logger.warning(f"VecNormalize stats introuvables: {load_path}")
        return

    vn = _get_vec_normalize(vec_env)
    if vn is None:
        logger.warning("Pas de VecNormalize trouvé dans la wrapper chain")
        return

    # Charger les stats sauvegardées
    saved_vn = VecNormalize.load(str(load_path), vn.venv)
    vn.obs_rms = saved_vn.obs_rms
    vn.ret_rms = saved_vn.ret_rms

    # Mode évaluation : pas de mise à jour des stats, pas de normalisation des rewards
    vn.training = False
    vn.norm_reward = False

    logger.info(f"VecNormalize stats chargées (mode eval): {load_path}")


def get_feature_set(version: str = "v1") -> list[str]:
    """
    Retourne le set de features pour l'entraînement.

    Args:
        version: "v1" = features curées (14), "v1_legacy"/"v2_legacy"/"v3_legacy" = anciens sets

    Returns:
        Liste des noms de colonnes
    """
    sets = {
        "v1": FEATURES_V1,
        "v2": FEATURES_V2,
        "v3": FEATURES_V3,
    }
    if version not in sets:
        raise ValueError(f"Version inconnue: {version}. Choix: {list(sets.keys())}")
    return sets[version]
