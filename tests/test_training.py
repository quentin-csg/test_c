"""
Tests Phase 6 — Training, Backtest et Logger.

Ces tests vérifient les fonctionnalités sans lancer de vrai entraînement long.
On utilise des micro-datasets et des timesteps très courts.
"""

import csv
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from agent.model import create_agent, make_vec_env, save_agent
from config.settings import LOGS_DIR, MODELS_DIR
from env.trading_env import TradingEnv
from training.logger import (
    BACKTEST_LOG_DIR,
    MONTHLY_CSV_COLUMNS,
    WEEKLY_CSV_COLUMNS,
    WEEKLY_LOG_DIR,
    append_monthly_csv,
    append_weekly_csv,
    log_backtest_result,
    log_weekly_summary,
    load_backtest_results,
    print_stats,
    _serialize,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_df():
    """Crée un DataFrame simulé minimal pour les tests."""
    np.random.seed(42)
    n = 200
    prices = 30000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        "open": prices + np.random.randn(n) * 10,
        "high": prices + abs(np.random.randn(n) * 30),
        "low": prices - abs(np.random.randn(n) * 30),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
        "rsi_14": np.random.uniform(20, 80, n),
        "sma_50": prices + np.random.randn(n) * 5,
        "atr_14": np.random.uniform(50, 200, n),
        "sentiment_score": np.random.uniform(-0.5, 0.5, n),
        "fear_greed_value": np.random.uniform(20, 80, n),
    })
    return df


@pytest.fixture
def micro_vec_env(sample_df):
    """Crée un environnement vectorisé minimal."""
    env = make_vec_env(
        df=sample_df,
        n_envs=1,
        use_subproc=False,
        frame_stack=4,
    )
    yield env
    env.close()


@pytest.fixture
def trained_micro_agent(micro_vec_env):
    """Crée et entraîne un agent avec très peu de steps."""
    agent = create_agent(micro_vec_env, tensorboard_log=None, seed=42)
    agent.learn(total_timesteps=64)
    return agent


@pytest.fixture
def tmp_logs_dir():
    """Crée un dossier temporaire pour les logs de test."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# Tests Logger
# ============================================================================


class TestLogger:
    """Tests pour training/logger.py."""

    def test_log_weekly_summary(self, tmp_logs_dir):
        """Sauvegarde et lit un résumé hebdomadaire."""
        with patch("training.logger.WEEKLY_LOG_DIR", tmp_logs_dir):
            stats = {"pnl": 150.5, "sharpe": 1.2, "trades": 42}
            filepath = log_weekly_summary(stats, week_label="2024_W01")

            assert filepath.exists()
            with open(filepath, "r") as f:
                data = json.load(f)
            assert data["pnl"] == 150.5
            assert data["sharpe"] == 1.2
            assert data["trades"] == 42
            assert data["week"] == "2024_W01"

    def test_log_weekly_auto_label(self, tmp_logs_dir):
        """Génère automatiquement le label de semaine."""
        with patch("training.logger.WEEKLY_LOG_DIR", tmp_logs_dir):
            filepath = log_weekly_summary({"pnl": 0})
            assert filepath.exists()
            assert "week_" in filepath.name

    def test_log_backtest_result(self, tmp_logs_dir):
        """Sauvegarde un résultat de backtest."""
        with patch("training.logger.BACKTEST_LOG_DIR", tmp_logs_dir):
            stats = {
                "total_return_pct": 5.2,
                "sharpe_ratio": 1.1,
                "max_drawdown_pct": 3.5,
            }
            filepath = log_backtest_result(
                stats, model_name="test_model", run_name="run_001"
            )
            assert filepath.exists()
            with open(filepath, "r") as f:
                data = json.load(f)
            assert data["total_return_pct"] == 5.2
            assert data["model_name"] == "test_model"

    def test_load_backtest_results(self, tmp_logs_dir):
        """Charge et filtre les résultats de backtest."""
        with patch("training.logger.BACKTEST_LOG_DIR", tmp_logs_dir):
            # Créer 2 résultats
            log_backtest_result({"pnl": 100}, model_name="modelA", run_name="r1")
            log_backtest_result({"pnl": 200}, model_name="modelB", run_name="r2")

            # Charger tous
            all_results = load_backtest_results()
            assert len(all_results) == 2

            # Filtrer par modèle
            filtered = load_backtest_results(model_name="modelA")
            assert len(filtered) == 1
            assert filtered[0]["pnl"] == 100

    def test_serialize_numpy(self):
        """Convertit correctement les types numpy."""
        assert _serialize(np.int64(42)) == 42
        assert isinstance(_serialize(np.int64(42)), int)
        assert _serialize(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_serialize(np.float64(3.14)), float)
        assert _serialize(np.array([1, 2, 3])) == [1, 2, 3]

    def test_print_stats(self, capsys):
        """print_stats affiche les métriques de manière formatée."""
        stats = {"pnl": 123.4567, "trades": 42}
        print_stats(stats, title="Test")
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "pnl" in captured.out
        assert "123.4567" in captured.out

    def test_append_weekly_csv_creates_file(self, tmp_logs_dir):
        """Crée le CSV avec header à la première écriture."""
        csv_path = tmp_logs_dir / "weekly_summary.csv"
        stats = {
            "model_name": "v1_test",
            "pnl_usdt": 150.0,
            "pnl_cumul_usdt": 150.0,
            "net_worth": 10150.0,
            "total_return_pct": 1.5,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "max_drawdown_pct": 0.8,
            "total_trades": 12,
            "total_fees": 3.5,
            "mode": "paper",
        }
        append_weekly_csv(stats, week_label="2026_W11", csv_path=csv_path)

        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["week"] == "2026_W11"
        assert rows[0]["pnl_usdt"] == "150.0"
        assert rows[0]["model_name"] == "v1_test"

    def test_append_weekly_csv_cumulative(self, tmp_logs_dir):
        """Ajoute des lignes sans écraser les précédentes."""
        csv_path = tmp_logs_dir / "weekly_summary.csv"
        for i, week in enumerate(["2026_W10", "2026_W11", "2026_W12"]):
            stats = {
                "model_name": "v1",
                "pnl_usdt": 100.0 * (i + 1),
                "pnl_cumul_usdt": sum(100.0 * (j + 1) for j in range(i + 1)),
                "net_worth": 10000 + 100.0 * (i + 1),
                "total_return_pct": 1.0 * (i + 1),
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 5,
                "total_fees": 1.0,
                "mode": "paper",
            }
            append_weekly_csv(stats, week_label=week, csv_path=csv_path)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["week"] == "2026_W10"
        assert rows[2]["week"] == "2026_W12"
        assert rows[2]["pnl_cumul_usdt"] == "600.0"

    def test_append_monthly_csv_creates_file(self, tmp_logs_dir):
        """Crée le CSV mensuel avec header à la première écriture."""
        csv_path = tmp_logs_dir / "monthly_summary.csv"
        stats = {
            "model_name": "v1_test",
            "pnl_usdt": 500.0,
            "pnl_cumul_usdt": 500.0,
            "net_worth": 10500.0,
            "total_return_pct": 5.0,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.3,
            "max_drawdown_pct": 2.1,
            "total_trades": 45,
            "total_fees": 12.0,
            "mode": "paper",
        }
        append_monthly_csv(stats, month_label="2026_03", csv_path=csv_path)

        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["month"] == "2026_03"
        assert rows[0]["pnl_usdt"] == "500.0"

    def test_append_monthly_csv_cumulative(self, tmp_logs_dir):
        """Ajoute des lignes mensuelles sans écraser."""
        csv_path = tmp_logs_dir / "monthly_summary.csv"
        for i, month in enumerate(["2026_01", "2026_02", "2026_03"]):
            stats = {
                "model_name": "v1",
                "pnl_usdt": 200.0 * (i + 1),
                "pnl_cumul_usdt": sum(200.0 * (j + 1) for j in range(i + 1)),
                "net_worth": 10000 + 200.0 * (i + 1),
                "total_return_pct": 2.0 * (i + 1),
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 20,
                "total_fees": 5.0,
                "mode": "paper",
            }
            append_monthly_csv(stats, month_label=month, csv_path=csv_path)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["month"] == "2026_01"
        assert rows[2]["month"] == "2026_03"
        assert rows[2]["pnl_cumul_usdt"] == "1200.0"


# ============================================================================
# Tests Training
# ============================================================================


class TestTraining:
    """Tests pour training/train.py."""

    def test_train_micro(self, sample_df, tmp_path):
        """Entraîne un agent avec très peu de steps (validation de la pipeline)."""
        from training.train import train

        # Mocker build_full_pipeline pour éviter les appels réseau
        mock_scaler = MagicMock()
        with patch("training.train.build_full_pipeline") as mock_pipeline, \
             patch("training.train.MODELS_DIR", tmp_path / "models"), \
             patch("training.train.TENSORBOARD_LOG_DIR", str(tmp_path / "tb")):

            mock_pipeline.return_value = (sample_df, mock_scaler)

            model_path = train(
                total_timesteps=128,
                n_envs=1,
                frame_stack=4,
                model_name="test_micro",
                use_subproc=False,
                seed=42,
            )

            assert model_path is not None
            # Vérifier que le modèle est sauvegardé
            mock_pipeline.assert_called_once()
            mock_scaler.save.assert_called_once()

    def test_agent_save_load(self, trained_micro_agent, micro_vec_env, tmp_path):
        """Sauvegarde et recharge un agent PPO."""
        with patch("agent.model.MODELS_DIR", tmp_path):
            saved_path = save_agent(trained_micro_agent, name="test_save")
            assert (tmp_path / "test_save.zip").exists()


# ============================================================================
# Tests Backtest
# ============================================================================


class TestBacktest:
    """Tests pour training/backtest.py."""

    def test_backtest_runs(self, sample_df, tmp_path):
        """Exécute un backtest complet sur des données simulées."""
        from training.backtest import backtest

        # 1. Créer et entraîner un micro-agent
        vec_env = make_vec_env(
            df=sample_df, n_envs=1, use_subproc=False, frame_stack=4
        )
        agent = create_agent(vec_env, tensorboard_log=None, seed=42)
        agent.learn(total_timesteps=64)

        with patch("agent.model.MODELS_DIR", tmp_path):
            save_agent(agent, name="backtest_test")
        vec_env.close()

        # 2. Lancer le backtest
        mock_scaler = MagicMock()
        with patch("training.backtest.build_full_pipeline") as mock_pipeline, \
             patch("agent.model.MODELS_DIR", tmp_path), \
             patch("training.backtest.log_backtest_result") as mock_log:

            mock_pipeline.return_value = (sample_df, mock_scaler)

            stats = backtest(
                model_name="backtest_test",
                frame_stack=4,
                save_results=True,
            )

        # 3. Vérifier les métriques
        assert "total_return_pct" in stats
        assert "net_worth" in stats
        assert "sharpe_ratio" in stats
        assert "sortino_ratio" in stats
        assert "max_drawdown_pct" in stats
        assert "total_trades" in stats
        assert "total_steps" in stats
        assert stats["total_steps"] > 0
        mock_log.assert_called_once()

    def test_backtest_deterministic(self, sample_df, tmp_path):
        """Le backtest est déterministe (mode deterministic=True)."""
        from training.backtest import backtest

        vec_env = make_vec_env(
            df=sample_df, n_envs=1, use_subproc=False, frame_stack=4
        )
        agent = create_agent(vec_env, tensorboard_log=None, seed=42)
        agent.learn(total_timesteps=64)

        with patch("agent.model.MODELS_DIR", tmp_path):
            save_agent(agent, name="det_test")
        vec_env.close()

        results = []
        for _ in range(2):
            mock_scaler = MagicMock()
            with patch("training.backtest.build_full_pipeline") as mock_pipeline, \
                 patch("agent.model.MODELS_DIR", tmp_path):

                mock_pipeline.return_value = (sample_df.copy(), mock_scaler)

                stats = backtest(
                    model_name="det_test",
                    frame_stack=4,
                    save_results=False,
                )
                results.append(stats["total_return_pct"])

        assert results[0] == pytest.approx(results[1], abs=0.01)


# ============================================================================
# Tests d'intégration légers
# ============================================================================


class TestIntegration:
    """Tests d'intégration rapides entre les modules."""

    def test_train_then_backtest_pipeline(self, sample_df, tmp_path):
        """Enchaîne entraînement → sauvegarde → backtest."""
        # Entraîner
        vec_env = make_vec_env(
            df=sample_df, n_envs=1, use_subproc=False, frame_stack=4
        )
        agent = create_agent(vec_env, tensorboard_log=None, seed=42)
        agent.learn(total_timesteps=128)

        with patch("agent.model.MODELS_DIR", tmp_path):
            save_agent(agent, name="integ_test")
        vec_env.close()

        # Backtest avec le même modèle
        from training.backtest import backtest

        mock_scaler = MagicMock()
        with patch("training.backtest.build_full_pipeline") as mock_pipeline, \
             patch("agent.model.MODELS_DIR", tmp_path):

            mock_pipeline.return_value = (sample_df.copy(), mock_scaler)

            stats = backtest(
                model_name="integ_test",
                frame_stack=4,
                save_results=False,
            )

        # L'agent a été entraîné sur les mêmes données, doit avoir des trades
        assert stats["net_worth"] > 0
        assert stats["total_steps"] > 0

    def test_logger_with_backtest_stats(self, sample_df, tmp_path):
        """Les stats de backtest sont correctement logguées."""
        stats = {
            "total_return_pct": np.float64(2.5),
            "net_worth": np.float64(10250.0),
            "sharpe_ratio": np.float64(0.8),
            "total_trades": np.int64(15),
        }

        with patch("training.logger.BACKTEST_LOG_DIR", tmp_path):
            filepath = log_backtest_result(
                stats, model_name="test", run_name="run1"
            )

        with open(filepath, "r") as f:
            loaded = json.load(f)

        assert loaded["total_return_pct"] == 2.5
        assert loaded["total_trades"] == 15
        assert isinstance(loaded["total_trades"], int)
