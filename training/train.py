"""
Script d'entraînement PPO.

Utilise les données historiques (2020-2023 par défaut) pour entraîner
l'agent via SubprocVecEnv multi-core avec callbacks TensorBoard et checkpoints.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from agent.model import create_agent, get_feature_set, load_agent, make_vec_env, save_agent, save_vec_normalize
from config.settings import (
    CHECKPOINT_FREQ,
    EARLY_STOPPING_PATIENCE,
    FRAME_STACK_SIZE,
    MODELS_DIR,
    N_ENVS,
    TENSORBOARD_LOG_DIR,
    TOTAL_TIMESTEPS,
    TRAIN_END,
    TRAIN_START,
)
from data.pipeline import build_full_pipeline
from training.logger import print_stats

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """Arrête l'entraînement si le reward moyen stagne pendant N évaluations."""

    def __init__(self, check_freq: int = 10000, patience: int = EARLY_STOPPING_PATIENCE,
                 min_delta: float = 0.0, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.n_calls_since_check = 0

    def _on_step(self) -> bool:
        self.n_calls_since_check += 1
        if self.n_calls_since_check >= self.check_freq:
            self.n_calls_since_check = 0
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                if mean_reward > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        logger.info(f"Early stopping: nouveau best reward = {mean_reward:.4f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        logger.info(
                            f"Early stopping: pas d'amélioration ({self.no_improvement_count}/{self.patience})"
                        )

                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print(f"\n=== Early stopping: pas d'amélioration depuis {self.patience} checks ===")
                    return False
        return True


def train(
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = N_ENVS,
    frame_stack: int = FRAME_STACK_SIZE,
    feature_columns: Optional[list[str]] = None,
    model_name: Optional[str] = None,
    include_nlp: bool = False,
    use_subproc: bool = True,
    seed: Optional[int] = None,
    warm_start_model: Optional[str] = None,
) -> Path:
    """
    Lance l'entraînement PPO.

    Args:
        train_start: date de début des données d'entraînement
        train_end: date de fin des données d'entraînement
        total_timesteps: nombre total de steps d'entraînement
        n_envs: nombre d'environnements parallèles
        frame_stack: nombre de frames empilées
        feature_columns: features à utiliser (None = V3 par défaut)
        model_name: nom du modèle sauvegardé (auto-généré si None)
        include_nlp: inclure l'analyse NLP FinBERT dans les données
        use_subproc: utiliser SubprocVecEnv (True) ou DummyVecEnv (False)
        seed: seed pour la reproductibilité
        warm_start_model: nom d'un modèle existant à charger comme point de départ
                          (warm start pour walk-forward — None = entraînement from scratch)

    Returns:
        Chemin du modèle sauvegardé
    """
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ppo_trading_{timestamp}"

    logger.info(f"=== Début de l'entraînement: {model_name} ===")
    logger.info(f"Données: {train_start} → {train_end}")
    logger.info(f"Steps: {total_timesteps}, Envs: {n_envs}, Frame stack: {frame_stack}")

    # 1. Récupérer et préparer les données
    print(f"[1/4] Chargement des données {train_start} → {train_end}...")
    dataset, scaler = build_full_pipeline(
        start=train_start,
        end=train_end,
        include_nlp=include_nlp,
    )

    if dataset.empty:
        raise RuntimeError(
            f"Pipeline a retourné un dataset vide pour {train_start} → {train_end}. "
            "Vérifiez la connexion internet et les dates."
        )

    print(f"  → {len(dataset)} bougies, {len(dataset.columns)} colonnes")

    # Sauvegarder le scaler pour réutilisation en backtest/live
    scaler.save()
    print("  → Scaler sauvegardé")

    # 2. Créer l'environnement vectorisé
    # Aligner les features avec l'executor live : V2 par défaut (14 features curées)
    # IMPORTANT : ne pas laisser None (auto-détection de toutes colonnes) car
    # cela créerait un obs space incompatible avec _build_observation() de l'executor.
    if feature_columns is None:
        feature_columns = get_feature_set("v3")
    print(f"[2/4] Création de l'environnement ({n_envs} envs, stack={frame_stack})...")
    print(f"  → Features: {len(feature_columns)} colonnes ({feature_columns})")
    vec_env = make_vec_env(
        df=dataset,
        n_envs=n_envs,
        feature_columns=feature_columns,
        use_subproc=use_subproc,
        frame_stack=frame_stack,
    )
    print(f"  → Observation space: {vec_env.observation_space.shape}")

    # 3. Créer (ou charger) l'agent et les callbacks
    print(f"[3/4] Création de l'agent PPO...")

    # Run name pour TensorBoard
    run_name = model_name

    if warm_start_model is not None:
        try:
            agent = load_agent(env=vec_env, name=warm_start_model)
            print(f"  → Warm start depuis: {warm_start_model}.zip")
        except FileNotFoundError:
            logger.warning(f"Warm start '{warm_start_model}' introuvable — démarrage à froid")
            agent = create_agent(env=vec_env, tensorboard_log=TENSORBOARD_LOG_DIR,
                                 seed=seed, frame_stack=frame_stack)
    else:
        agent = create_agent(
            env=vec_env,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            seed=seed,
            frame_stack=frame_stack,
        )

    # Callbacks
    checkpoint_dir = MODELS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=model_name,
        verbose=1,
    )

    early_stop = EarlyStoppingCallback(
        check_freq=max(CHECKPOINT_FREQ // n_envs, 1),
        patience=EARLY_STOPPING_PATIENCE,
    )

    callbacks = CallbackList([checkpoint_callback, early_stop])

    # 4. Entraîner
    print(f"[4/4] Entraînement en cours ({total_timesteps} steps)...")
    print(f"  → TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )

    # Sauvegarder le modèle final et les stats VecNormalize
    model_path = save_agent(agent, name=model_name)
    save_vec_normalize(vec_env)
    print(f"\n=== Entraînement terminé ===")
    print(f"Modèle sauvegardé: {model_path}.zip")
    print(f"  → VecNormalize stats sauvegardées")

    vec_env.close()
    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
