"""
Script de backtest.

Charge un modèle entraîné et l'évalue sur des données de test (2024-aujourd'hui).
L'apprentissage est désactivé — on vérifie la rentabilité réelle.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from agent.model import load_agent, make_vec_env
from config.settings import (
    FRAME_STACK_SIZE,
    MODELS_DIR,
    TEST_END,
    TEST_START,
)
from data.pipeline import build_full_pipeline
from training.logger import log_backtest_result, print_stats

logger = logging.getLogger(__name__)


def backtest(
    model_name: str = "ppo_trading",
    test_start: str = TEST_START,
    test_end: Optional[str] = TEST_END,
    frame_stack: int = FRAME_STACK_SIZE,
    feature_columns: Optional[list[str]] = None,
    include_nlp: bool = False,
    save_results: bool = True,
) -> dict:
    """
    Exécute un backtest sur les données de test.

    Args:
        model_name: nom du modèle à charger (sans .zip)
        test_start: date de début des données de test
        test_end: date de fin (None = aujourd'hui)
        frame_stack: nombre de frames empilées (doit correspondre à l'entraînement)
        feature_columns: features à utiliser (doit correspondre à l'entraînement)
        include_nlp: inclure l'analyse NLP FinBERT
        save_results: sauvegarder les résultats en JSON

    Returns:
        Dict de métriques du backtest
    """
    logger.info(f"=== Backtest: {model_name} ===")
    logger.info(f"Données: {test_start} → {test_end or 'aujourd hui'}")

    # 1. Charger les données de test
    print(f"[1/4] Chargement des données {test_start} → {test_end or 'maintenant'}...")

    dataset, scaler = build_full_pipeline(
        start=test_start,
        end=test_end,
        include_nlp=include_nlp,
        fit_scaler=True,
    )
    print(f"  → {len(dataset)} bougies, {len(dataset.columns)} colonnes")

    # 2. Créer l'environnement
    print(f"[2/4] Création de l'environnement (stack={frame_stack})...")
    vec_env = make_vec_env(
        df=dataset,
        n_envs=1,
        feature_columns=feature_columns,
        use_subproc=False,
        frame_stack=frame_stack,
    )

    # 3. Charger le modèle
    print(f"[3/4] Chargement du modèle: {model_name}...")
    agent = load_agent(vec_env, name=model_name)

    # 4. Exécuter le backtest
    print(f"[4/4] Exécution du backtest...")
    obs = vec_env.reset()
    terminated = False
    total_steps = 0
    rewards = []

    while not terminated:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        rewards.append(reward[0])
        total_steps += 1
        terminated = dones[0]

    # Récupérer les stats du portfolio (unwrap VecFrameStack → DummyVecEnv → TradingEnv)
    inner_vec = vec_env.venv  # DummyVecEnv inside VecFrameStack
    trading_env = inner_vec.envs[0]
    stats = trading_env.get_portfolio_stats()

    # Ajouter des métriques supplémentaires
    stats["model_name"] = model_name
    stats["test_start"] = test_start
    stats["test_end"] = test_end or "now"
    stats["total_steps"] = total_steps
    stats["avg_reward"] = float(np.mean(rewards))
    stats["total_reward"] = float(np.sum(rewards))
    stats["feature_columns"] = feature_columns or list(trading_env.feature_columns)

    # Afficher les résultats
    print_stats(stats, title=f"Backtest: {model_name}")

    # Sauvegarder les résultats
    if save_results:
        log_backtest_result(stats, model_name=model_name)

    vec_env.close()
    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    name = sys.argv[1] if len(sys.argv) > 1 else "ppo_trading"
    backtest(model_name=name)
