"""
Tests d'integration end-to-end.

Verifie le pipeline complet : donnees synthetiques -> features -> env -> train -> save -> load -> backtest -> log.
Aucun appel reseau. Tout est mocke ou synthetique.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.model import create_agent, load_agent, make_vec_env, save_agent
from env.trading_env import TradingEnv
from features.scaler import FeatureScaler, normalize_features
from features.technical import add_all_indicators
from training.logger import (
    append_monthly_csv,
    append_weekly_csv,
    log_backtest_result,
    log_weekly_summary,
    load_backtest_results,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Genere un dataset synthetique realiste (OHLCV + colonnes macro)."""
    rng = np.random.RandomState(seed)
    prices = 30000 + np.cumsum(rng.randn(n) * 50)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "open": prices + rng.randn(n) * 10,
        "high": prices + abs(rng.randn(n) * 30),
        "low": prices - abs(rng.randn(n) * 30),
        "close": prices,
        "volume": rng.uniform(100, 1000, n),
        "fear_greed_value": rng.uniform(20, 80, n),
        "fear_greed_normalized": rng.uniform(-1, 1, n),
        "is_weekend": 0,
        "funding_rate": rng.uniform(-0.001, 0.001, n),
    })
    # Add macro
    df["QQQ_close"] = 350 + rng.randn(n) * 2
    df["SPY_close"] = 450 + rng.randn(n) * 2
    return df


def _make_featured_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Genere un dataset avec indicateurs techniques calcules."""
    df = _make_synthetic_dataset(n, seed)
    df = add_all_indicators(df)
    # Supprimer les lignes NaN (warmup indicateurs)
    df = df.dropna(thresh=len(df.columns) - 5).reset_index(drop=True)
    # Ajouter sentiment
    df["sentiment_score"] = 0.0
    df["n_articles"] = 0
    # Multi-timeframe features (pas disponibles en test — remplir avec 0)
    for col in ("rsi_4h_normalized", "sma_trend_4h"):
        if col not in df.columns:
            df[col] = 0.0
    # Sauvegarder les prix bruts avant normalisation pour TradingEnv
    df["raw_close"] = df["close"].copy()
    return df


# ============================================================================
# Tests
# ============================================================================


class TestEndToEnd:
    """Pipeline complet : donnees -> features -> scaler -> env -> train -> save -> load -> predict."""

    def test_full_pipeline_train_and_predict(self, tmp_path):
        """Entrainer un agent, le sauvegarder, le recharger et predire."""
        # 1. Preparer les donnees
        df = _make_featured_dataset(n=500, seed=42)
        assert len(df) > 100, "Pas assez de donnees apres warmup"

        # 2. Normaliser (mode train)
        df_scaled, scaler = normalize_features(df, fit=True)
        assert scaler.is_fitted
        assert len(scaler.feature_columns) > 0

        # 3. Sauvegarder le scaler
        scaler_path = tmp_path / "scaler.pkl"
        scaler.save(scaler_path)
        assert scaler_path.exists()

        # 4. Creer l'environnement
        vec_env = make_vec_env(
            df=df_scaled,
            n_envs=1,
            use_subproc=False,
            frame_stack=4,
        )
        obs_shape = vec_env.observation_space.shape
        assert len(obs_shape) == 1
        assert obs_shape[0] > 0

        # 5. Entrainer l'agent
        agent = create_agent(vec_env, tensorboard_log=None, seed=42, use_cnn=False)
        agent.learn(total_timesteps=64)

        # 6. Sauvegarder le modele
        model_path = save_agent(agent, name="test_integration", path=tmp_path)
        assert (tmp_path / "test_integration.zip").exists()

        # 7. Recharger le modele
        vec_env2 = make_vec_env(
            df=df_scaled,
            n_envs=1,
            use_subproc=False,
            frame_stack=4,
        )
        loaded_agent = load_agent(vec_env2, name="test_integration", path=tmp_path)
        assert loaded_agent is not None

        # 8. Predire
        obs = vec_env2.reset()
        action, _ = loaded_agent.predict(obs, deterministic=True)
        assert action.shape == (1, 1)
        assert -1.0 <= action[0][0] <= 1.0

        vec_env.close()
        vec_env2.close()

    def test_scaler_save_load_consistency(self, tmp_path):
        """Le scaler charge donne les memes resultats que l'original."""
        df = _make_featured_dataset(n=300, seed=99)
        df_scaled, scaler = normalize_features(df, fit=True)

        # Sauvegarder
        scaler_path = tmp_path / "scaler.pkl"
        scaler.save(scaler_path)

        # Recharger
        scaler2 = FeatureScaler()
        scaler2.load(scaler_path)

        # Transformer les memes donnees
        df_raw = _make_featured_dataset(n=300, seed=99)
        df_scaled2, _ = normalize_features(df_raw, scaler=scaler2, fit=False)

        # Comparer les colonnes numeriques
        for col in scaler.feature_columns:
            if col in df_scaled.columns and col in df_scaled2.columns:
                np.testing.assert_array_almost_equal(
                    df_scaled[col].values, df_scaled2[col].values,
                    decimal=6,
                    err_msg=f"Colonne {col} differe apres reload du scaler",
                )


class TestBacktestIntegration:
    """Backtest end-to-end avec agent micro-entraine."""

    def test_backtest_produces_valid_stats(self, tmp_path):
        """Le backtest retourne des stats valides depuis l'info terminale."""
        # Train
        df = _make_featured_dataset(n=400, seed=42)
        df_scaled, scaler = normalize_features(df, fit=True)

        vec_env = make_vec_env(
            df=df_scaled, n_envs=1, use_subproc=False, frame_stack=4,
        )
        agent = create_agent(vec_env, tensorboard_log=None, seed=42, use_cnn=False)
        agent.learn(total_timesteps=64)
        save_agent(agent, name="bt_test", path=tmp_path)
        vec_env.close()

        # Backtest
        vec_env_bt = make_vec_env(
            df=df_scaled, n_envs=1, use_subproc=False, frame_stack=4,
        )
        bt_agent = load_agent(vec_env_bt, name="bt_test", path=tmp_path)

        obs = vec_env_bt.reset()
        terminated = False
        total_steps = 0
        rewards = []
        terminal_info = None
        while not terminated:
            action, _ = bt_agent.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env_bt.step(action)
            rewards.append(reward[0])
            total_steps += 1
            terminated = dones[0]
            if terminated:
                terminal_info = infos[0]

        # Stats MUST come from terminal info (VecEnv auto-resets the env)
        assert terminal_info is not None, "Backtest n'a jamais termine"
        stats = terminal_info.get("portfolio_stats", {})
        assert stats, "portfolio_stats absent de l'info terminale"

        assert "net_worth" in stats
        assert "total_return_pct" in stats
        assert "sharpe_ratio" in stats
        assert "sortino_ratio" in stats
        assert "max_drawdown_pct" in stats
        assert stats["net_worth"] > 0
        assert total_steps > 10

        vec_env_bt.close()

    def test_backtest_terminal_stats_not_reset(self, tmp_path):
        """Verifie que portfolio_stats est dans l'info terminale et non vide."""
        df = _make_featured_dataset(n=400, seed=42)
        df_scaled, scaler = normalize_features(df, fit=True)

        vec_env = make_vec_env(
            df=df_scaled, n_envs=1, use_subproc=False, frame_stack=4,
        )
        agent = create_agent(vec_env, tensorboard_log=None, seed=42, use_cnn=False)
        agent.learn(total_timesteps=128)
        save_agent(agent, name="bt_reset_test", path=tmp_path)
        vec_env.close()

        vec_env_bt = make_vec_env(
            df=df_scaled, n_envs=1, use_subproc=False, frame_stack=4,
        )
        bt_agent = load_agent(vec_env_bt, name="bt_reset_test", path=tmp_path)

        obs = vec_env_bt.reset()
        terminated = False
        terminal_info = None
        while not terminated:
            action, _ = bt_agent.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env_bt.step(action)
            terminated = dones[0]
            if terminated:
                terminal_info = infos[0]

        # Verifier que portfolio_stats est present et complet
        assert "portfolio_stats" in terminal_info
        stats = terminal_info["portfolio_stats"]
        required_keys = [
            "net_worth", "total_return_pct", "max_drawdown_pct",
            "sharpe_ratio", "sortino_ratio", "total_trades", "total_fees",
        ]
        for key in required_keys:
            assert key in stats, f"Cle manquante dans portfolio_stats: {key}"

        # L'env interne est auto-reset → ses stats sont initiales
        inner_env = vec_env_bt.venv.envs[0]
        assert inner_env.total_trades == 0, "L'env devrait etre reset"

        # Mais les stats terminales doivent avoir des valeurs calculees
        assert stats["net_worth"] > 0
        assert isinstance(stats["sharpe_ratio"], float)

        vec_env_bt.close()


class TestLoggerIntegration:
    """Tests d'integration du systeme de logs."""

    def test_full_log_cycle(self, tmp_path):
        """Weekly JSON + CSV + monthly CSV + backtest JSON, puis lecture."""
        import config.settings as cfg
        original_train = cfg.LOGS_TRAIN_DIR
        cfg.LOGS_TRAIN_DIR = tmp_path / "train"

        try:
            stats = {
                "net_worth": 10500.0,
                "total_return_pct": 5.0,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "max_drawdown_pct": 3.0,
                "total_trades": 42,
                "total_fees": 12.5,
                "model_name": "test_model",
                "pnl_usdt": 500.0,
                "pnl_cumul_usdt": 500.0,
            }

            # 1. Weekly JSON
            weekly_path = log_weekly_summary(stats, week_label="2025_W01", mode="train")
            assert weekly_path.exists()
            with open(weekly_path) as f:
                data = json.load(f)
            assert data["net_worth"] == 10500.0

            # 2. Weekly CSV
            csv_dir = tmp_path / "train"
            csv_path = append_weekly_csv(
                stats, week_label="2025_W01", mode="train",
                csv_path=csv_dir / "weekly_summary.csv",
            )
            assert csv_path.exists()

            # 3. Monthly CSV
            monthly_path = append_monthly_csv(
                stats, month_label="2025_01", mode="train",
                csv_path=csv_dir / "monthly_summary.csv",
            )
            assert monthly_path.exists()

            # 4. Backtest JSON
            bt_path = log_backtest_result(
                stats, model_name="test_model", run_name="run001", mode="train",
            )
            assert bt_path.exists()

            # 5. Load backtest results
            results = load_backtest_results(model_name="test_model", mode="train")
            assert len(results) >= 1
            assert results[0]["net_worth"] == 10500.0

        finally:
            cfg.LOGS_TRAIN_DIR = original_train


class TestFeaturesPipeline:
    """Tests d'integration du pipeline de features."""

    def test_indicators_then_scale(self):
        """add_all_indicators + normalize_features bout a bout."""
        df = _make_synthetic_dataset(n=300, seed=42)
        df = add_all_indicators(df)

        # Verifier que les indicateurs sont presents
        expected_cols = ["rsi", "sma_50", "sma_200", "atr", "bb_upper", "bb_lower"]
        for col in expected_cols:
            assert col in df.columns, f"Colonne manquante: {col}"

        # Drop NaN warmup
        df_clean = df.dropna(thresh=len(df.columns) - 5).reset_index(drop=True)
        assert len(df_clean) > 50

        # Scale
        df_scaled, scaler = normalize_features(df_clean, fit=True)
        assert scaler.is_fitted

        # Toutes les valeurs scalees entre -1 et +1 (hors timestamp et flags)
        for col in scaler.feature_columns:
            vals = df_scaled[col].values
            assert np.all(vals >= -1.0), f"{col} a des valeurs < -1"
            assert np.all(vals <= 1.0), f"{col} a des valeurs > 1"

    def test_env_accepts_scaled_data(self):
        """L'environnement Gymnasium fonctionne avec des donnees scalees."""
        df = _make_featured_dataset(n=200, seed=42)
        df_scaled, scaler = normalize_features(df, fit=True)

        env = TradingEnv(df_scaled)
        obs, info = env.reset()
        assert obs.shape[0] > 0

        # Faire quelques steps
        for _ in range(10):
            action = np.array([np.random.uniform(-1, 1)])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        stats = env.get_portfolio_stats()
        assert "net_worth" in stats
        assert stats["net_worth"] > 0


class TestPaperTradingIntegration:
    """Tests d'integration du paper trading."""

    def test_paper_portfolio_multi_trades(self):
        """Serie d'achats/ventes avec verification de coherence."""
        from live.executor import PaperPortfolio

        portfolio = PaperPortfolio(initial_balance=10000.0)
        prices = [50000, 51000, 49000, 52000, 48000]
        actions = [0.3, -0.5, 0.8, -1.0, 0.1]

        for price, action in zip(prices, actions):
            portfolio.execute_order(action, current_price=price)

        stats = portfolio.get_stats(current_price=48000)

        # Verifications de coherence
        assert stats["balance_usdt"] >= 0
        assert stats["position_btc"] >= 0
        assert stats["total_trades"] > 0
        assert stats["total_fees"] > 0
        assert stats["net_worth"] > 0
        assert len(portfolio.trade_history) == 5

    def test_paper_portfolio_entry_price_averaging(self):
        """Achats successifs moyennent le prix d'entree."""
        from live.executor import PaperPortfolio

        portfolio = PaperPortfolio(initial_balance=10000.0)

        # Premier achat a 50,000
        portfolio.execute_order(0.5, current_price=50000.0)
        first_entry = portfolio.entry_price
        assert first_entry == 50000.0

        # Deuxieme achat a 60,000 -> prix d'entree moyen
        portfolio.execute_order(0.5, current_price=60000.0)
        avg_entry = portfolio.entry_price
        assert avg_entry > 50000.0
        assert avg_entry < 60000.0

    def test_paper_portfolio_last_price_tracking(self):
        """last_price est mis a jour a chaque ordre."""
        from live.executor import PaperPortfolio

        portfolio = PaperPortfolio(initial_balance=10000.0)
        assert portfolio.last_price == 0.0

        portfolio.execute_order(0.3, current_price=50000.0)
        assert portfolio.last_price == 50000.0

        portfolio.execute_order(0.0, current_price=55000.0)
        assert portfolio.last_price == 55000.0

    def test_raw_close_preserved_through_scaling(self):
        """raw_close est preserve apres normalisation."""
        df = _make_featured_dataset(n=200, seed=42)
        original_close = df["close"].copy()

        df["raw_close"] = df["close"].copy()
        df_scaled, scaler = normalize_features(df, fit=True)

        # raw_close doit etre identique a l'original (non normalisee)
        assert "raw_close" in df_scaled.columns
        pd.testing.assert_series_equal(
            df_scaled["raw_close"].reset_index(drop=True),
            original_close.reset_index(drop=True),
            check_names=False,
        )

        # close doit etre normalisee (differente de l'original)
        assert not np.allclose(df_scaled["close"].values, original_close.values)

    def test_raw_close_excluded_from_env_features(self):
        """raw_close ne doit PAS etre dans les features de l'agent."""
        df = _make_featured_dataset(n=200, seed=42)
        df["raw_close"] = df["close"].copy()
        df_scaled, scaler = normalize_features(df, fit=True)

        env = TradingEnv(df_scaled)
        assert "raw_close" not in env.feature_columns
        assert "close" in env.feature_columns

    def test_circuit_breaker_detection(self):
        """Circuit breaker detecte un crash simule."""
        from unittest.mock import MagicMock, patch
        from live.circuit_breaker import CircuitBreaker

        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(live_mode=False)
            assert not cb.triggered

            # Trigger manuellement
            cb.trigger("Price crash -5% in 3 min")
            assert cb.triggered
            assert "crash" in cb.trigger_reason.lower()

            # Reset
            cb.reset()
            assert not cb.triggered


class TestErrorHandling:
    """Tests de gestion d'erreurs aux points critiques."""

    def test_load_agent_missing_model(self, tmp_path):
        """load_agent leve FileNotFoundError si le modele n'existe pas."""
        from agent.model import load_agent, make_vec_env
        df = _make_featured_dataset(n=200, seed=42)
        df_scaled, _ = normalize_features(df, fit=True)
        vec_env = make_vec_env(df=df_scaled, n_envs=1, use_subproc=False, frame_stack=2)

        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_agent(vec_env, name="modele_inexistant", path=tmp_path)
        vec_env.close()

    def test_scaler_load_missing_file(self, tmp_path):
        """FeatureScaler.load leve FileNotFoundError si le fichier n'existe pas."""
        from features.scaler import FeatureScaler
        scaler = FeatureScaler()
        with pytest.raises(FileNotFoundError):
            scaler.load(tmp_path / "inexistant.pkl")

    def test_scaler_transform_before_fit(self):
        """FeatureScaler.transform leve RuntimeError si pas fitted."""
        from features.scaler import FeatureScaler
        scaler = FeatureScaler()
        df = _make_featured_dataset(n=50, seed=42)
        with pytest.raises(RuntimeError, match="pas .* ajust"):
            scaler.transform(df)


class TestMaxDrawdownAccuracy:
    """Verifie la precision du calcul du max drawdown."""

    def test_max_drawdown_cumulative(self):
        """Le drawdown cumule sur plusieurs steps est correctement mesure."""
        # Creer un prix qui monte puis descend de 10% en 2 etapes
        n = 20
        prices = [100.0] * n
        # Prix monte a 110 au step 5, puis descend a 99 sur les steps 6-7
        prices[5] = 110.0
        prices[6] = 104.5   # -5% depuis le pic
        prices[7] = 99.0    # -10% depuis le pic (cumulatif)
        prices[8] = 99.0
        for i in range(9, n):
            prices[i] = 100.0

        df = pd.DataFrame({
            "close": prices,
            "rsi": [0.0] * n,
        })

        env = TradingEnv(df=df, initial_balance=10000.0,
                         slippage_min=0.0, slippage_max=0.0)
        env.reset(seed=42)

        # Acheter 100% au step 0
        env.step(np.array([1.0]))

        # Holdover les steps suivants
        for _ in range(n - 2):
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            if terminated:
                break

        stats = env.get_portfolio_stats()
        # Avec MAX_POSITION_PCT=30%, un crash de -10% du prix → ~3% drawdown portfolio
        assert stats["max_drawdown_pct"] > 2.5, (
            f"Le drawdown cumule devrait etre > 2.5%, got {stats['max_drawdown_pct']:.2f}%"
        )

    def test_no_drawdown_when_price_only_rises(self):
        """Pas de drawdown si le prix monte continuellement."""
        n = 20
        prices = [100.0 + i * 1.0 for i in range(n)]
        df = pd.DataFrame({"close": prices, "rsi": [0.0] * n})

        env = TradingEnv(df=df, initial_balance=10000.0,
                         slippage_min=0.0, slippage_max=0.0,
                         trading_fee=0.0)
        env.reset(seed=42)

        # Acheter et hold
        env.step(np.array([1.0]))
        for _ in range(n - 2):
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            if terminated:
                break

        stats = env.get_portfolio_stats()
        # Drawdown devrait etre 0 (prix monte toujours)
        assert stats["max_drawdown_pct"] == pytest.approx(0.0, abs=0.01)

    def test_drawdown_resets_after_new_peak(self):
        """Le drawdown se recalcule correctement apres un nouveau pic."""
        n = 30
        prices = [100.0] * n
        # Premier pic a 110, descente a 100, puis nouveau pic a 120, descente a 108
        for i in range(5):
            prices[i] = 100.0 + i * 2  # monte a 108
        prices[5] = 110.0  # pic 1
        prices[6] = 100.0  # -9.1% depuis pic 1
        prices[7] = 105.0
        prices[8] = 120.0  # pic 2 (nouveau peak)
        prices[9] = 108.0  # -10% depuis pic 2
        for i in range(10, n):
            prices[i] = 115.0

        df = pd.DataFrame({"close": prices, "rsi": [0.0] * n})
        env = TradingEnv(df=df, initial_balance=10000.0,
                         slippage_min=0.0, slippage_max=0.0,
                         trading_fee=0.0)
        env.reset(seed=42)

        env.step(np.array([1.0]))
        for _ in range(n - 2):
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            if terminated:
                break

        stats = env.get_portfolio_stats()
        # Avec MAX_POSITION_PCT=30%, un crash de -10% du prix → ~3% drawdown portfolio
        assert stats["max_drawdown_pct"] > 2.5
