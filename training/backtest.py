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
        fit_scaler=False,
    )

    if dataset.empty:
        raise RuntimeError(
            f"Pipeline a retourné un dataset vide pour {test_start} → {test_end}. "
            "Vérifiez la connexion internet et les dates."
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
    try:
        agent = load_agent(vec_env, name=model_name)
    except FileNotFoundError:
        vec_env.close()
        raise

    # 4. Exécuter le backtest
    print(f"[4/4] Exécution du backtest ({len(dataset)} steps)...")
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
        if terminated:
            terminal_info = infos[0]

    # Récupérer les stats depuis l'info terminale (AVANT auto-reset du VecEnv)
    # VecEnv auto-reset l'env quand done=True, donc env.get_portfolio_stats()
    # retournerait les stats de l'épisode reset, pas celles du backtest.
    stats = terminal_info.get("portfolio_stats", {})
    if not stats:
        logger.warning(
            "portfolio_stats absent de l'info terminale, "
            "fallback sur l'info de base"
        )
        stats = {
            "net_worth": terminal_info.get("net_worth", 0),
            "total_trades": terminal_info.get("total_trades", 0),
            "total_fees": terminal_info.get("total_fees", 0),
            "total_return_pct": 0.0,
        }

    # Ajouter des métriques supplémentaires
    stats["model_name"] = model_name
    stats["test_start"] = test_start
    stats["test_end"] = test_end or "now"
    stats["total_steps"] = total_steps
    stats["avg_reward"] = float(np.mean(rewards))
    stats["total_reward"] = float(np.sum(rewards))
    stats["feature_columns"] = feature_columns or list(dataset.columns)

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
