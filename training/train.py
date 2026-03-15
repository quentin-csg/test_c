"""
Script d'entraînement PPO.

Utilise les données historiques (2020-2023 par défaut) pour entraîner
l'agent via SubprocVecEnv multi-core avec callbacks TensorBoard et checkpoints.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)

from agent.model import create_agent, make_vec_env, save_agent
from config.settings import (
    CHECKPOINT_FREQ,
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
) -> Path:
    """
    Lance l'entraînement PPO.

    Args:
        train_start: date de début des données d'entraînement
        train_end: date de fin des données d'entraînement
        total_timesteps: nombre total de steps d'entraînement
        n_envs: nombre d'environnements parallèles
        frame_stack: nombre de frames empilées
        feature_columns: features à utiliser (None = toutes colonnes numériques)
        model_name: nom du modèle sauvegardé (auto-généré si None)
        include_nlp: inclure l'analyse NLP FinBERT dans les données
        use_subproc: utiliser SubprocVecEnv (True) ou DummyVecEnv (False)
        seed: seed pour la reproductibilité

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
    print(f"[2/4] Création de l'environnement ({n_envs} envs, stack={frame_stack})...")
    vec_env = make_vec_env(
        df=dataset,
        n_envs=n_envs,
        feature_columns=feature_columns,
        use_subproc=use_subproc,
        frame_stack=frame_stack,
    )
    print(f"  → Observation space: {vec_env.observation_space.shape}")

    # 3. Créer l'agent et les callbacks
    print(f"[3/4] Création de l'agent PPO...")

    # Run name pour TensorBoard
    run_name = model_name

    agent = create_agent(
        env=vec_env,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        seed=seed,
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

    callbacks = CallbackList([checkpoint_callback])

    # 4. Entraîner
    print(f"[4/4] Entraînement en cours ({total_timesteps} steps)...")
    print(f"  → TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )

    # Sauvegarder le modèle final
    model_path = save_agent(agent, name=model_name)
    print(f"\n=== Entraînement terminé ===")
    print(f"Modèle sauvegardé: {model_path}.zip")

    vec_env.close()
    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
