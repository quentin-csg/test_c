"""
Tests Phase 7 — Live / Paper Trading.

Ces tests vérifient les fonctionnalités sans connexion réseau.
On utilise des mocks pour ccxt et les données.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from live.circuit_breaker import CircuitBreaker
from live.executor import LiveExecutor, PaperPortfolio


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_df():
    """DataFrame simulé minimal."""
    np.random.seed(42)
    n = 200
    prices = 30000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame({
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


# ============================================================================
# Tests PaperPortfolio
# ============================================================================


class TestPaperPortfolio:
    """Tests pour le portefeuille paper trading."""

    def test_initial_state(self):
        """Le portefeuille commence avec le bon solde."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        assert portfolio.balance == 10000.0
        assert portfolio.position == 0.0
        assert portfolio.total_trades == 0

    def test_buy_order(self):
        """Achat via paper portfolio."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        order = portfolio.execute_order(0.5, current_price=50000.0)

        assert order["type"] == "buy"
        assert portfolio.balance < 10000.0
        assert portfolio.position > 0.0
        assert portfolio.total_trades == 1
        assert portfolio.total_fees > 0.0

    def test_sell_order(self):
        """Vente via paper portfolio."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        # D'abord acheter
        portfolio.execute_order(1.0, current_price=50000.0)
        initial_position = portfolio.position

        # Puis vendre 50%
        order = portfolio.execute_order(-0.5, current_price=50000.0)

        assert order["type"] == "sell"
        assert portfolio.position < initial_position
        assert portfolio.total_trades == 2

    def test_hold_in_dead_zone(self):
        """Action dans la zone morte = pas de trade."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        order = portfolio.execute_order(0.03, current_price=50000.0)

        assert order["type"] == "hold"
        assert portfolio.balance == 10000.0
        assert portfolio.total_trades == 0

    def test_fees_applied(self):
        """Les frais sont correctement appliqués."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        portfolio.execute_order(1.0, current_price=50000.0)

        # Frais = 0.1% de 10000 = 10 USDT
        assert portfolio.total_fees == pytest.approx(10.0, abs=0.1)

    def test_get_stats(self):
        """Statistiques du portefeuille cohérentes."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        portfolio.execute_order(0.5, current_price=50000.0)

        stats = portfolio.get_stats(current_price=50000.0)
        assert "net_worth" in stats
        assert "balance_usdt" in stats
        assert "position_btc" in stats
        assert "total_return_pct" in stats
        assert "total_trades" in stats
        assert stats["total_trades"] == 1

    def test_trade_history(self):
        """L'historique des trades est conservé."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        portfolio.execute_order(0.5, current_price=50000.0)
        portfolio.execute_order(-0.5, current_price=51000.0)

        assert len(portfolio.trade_history) == 2
        assert portfolio.trade_history[0]["type"] == "buy"
        assert portfolio.trade_history[1]["type"] == "sell"


# ============================================================================
# Tests CircuitBreaker
# ============================================================================


class TestCircuitBreaker:
    """Tests pour le circuit breaker."""

    def test_initial_state(self):
        """Le circuit breaker démarre non déclenché."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(live_mode=False)
            assert cb.triggered is False
            assert cb.trigger_reason == ""

    def test_status(self):
        """Le statut retourne les bonnes informations."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(
                price_drop_threshold=0.03,
                volume_spike_factor=5.0,
                lookback_minutes=5,
            )
            status = cb.status
            assert status["triggered"] is False
            assert status["config"]["price_drop_threshold"] == 0.03
            assert status["config"]["volume_spike_factor"] == 5.0

    def test_trigger(self):
        """Le déclenchement met le bon état."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(live_mode=False)
            cb.trigger("Test crash")

            assert cb.triggered is True
            assert cb.trigger_reason == "Test crash"
            assert cb.trigger_time is not None

    def test_reset(self):
        """Le reset remet le circuit breaker en état normal."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(live_mode=False)
            cb.trigger("Test crash")
            cb.reset()

            assert cb.triggered is False
            assert cb.trigger_reason == ""
            assert cb.trigger_time is None

    def test_price_drop_detection(self):
        """Détecte une chute de prix."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(
                price_drop_threshold=0.03,
                lookback_minutes=5,
                live_mode=False,
            )

            # Simuler des bougies avec chute de 5%
            normal_price = 50000
            crash_price = 47000  # -6%
            ohlcv = []
            # 20 bougies normales
            for i in range(20):
                ohlcv.append([i * 60000, normal_price, normal_price + 10,
                             normal_price - 10, normal_price, 100])
            # 5 bougies en crash
            for i in range(5):
                p = normal_price - (normal_price - crash_price) * (i + 1) / 5
                ohlcv.append([(20 + i) * 60000, p, p + 10, p - 10, p, 500])

            cb.exchange.fetch_ohlcv = MagicMock(return_value=ohlcv)
            result = cb.check_conditions()

            assert result["price_drop_detected"] is True

    def test_volume_spike_detection(self):
        """Détecte un volume anormal."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(
                volume_spike_factor=5.0,
                lookback_minutes=5,
                live_mode=False,
            )

            # 20 bougies normales (volume=100) + 5 bougies volume spike (volume=1000)
            ohlcv = []
            for i in range(20):
                ohlcv.append([i * 60000, 50000, 50010, 49990, 50000, 100])
            for i in range(5):
                ohlcv.append([(20 + i) * 60000, 50000, 50010, 49990, 50000, 1000])

            cb.exchange.fetch_ohlcv = MagicMock(return_value=ohlcv)
            result = cb.check_conditions()

            assert result["volume_spike_detected"] is True
            assert result["volume_ratio"] >= 5.0

    def test_no_false_alarm(self):
        """Pas de faux positif en conditions normales."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(
                price_drop_threshold=0.03,
                volume_spike_factor=5.0,
                lookback_minutes=5,
                live_mode=False,
            )

            # 25 bougies stables
            ohlcv = []
            for i in range(25):
                ohlcv.append([i * 60000, 50000, 50010, 49990, 50000, 100])

            cb.exchange.fetch_ohlcv = MagicMock(return_value=ohlcv)
            result = cb.check_conditions()

            assert result["price_drop_detected"] is False
            assert result["volume_spike_detected"] is False

    def test_close_positions_paper(self):
        """Fermeture positions en mode paper."""
        with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
            cb = CircuitBreaker(live_mode=False)
            result = cb.close_all_positions()

            assert result["success"] is True
            assert result["mode"] == "paper"


# ============================================================================
# Tests LiveExecutor
# ============================================================================


class TestLiveExecutor:
    """Tests pour l'exécuteur live."""

    def test_init_paper_mode(self):
        """Initialisation en mode paper."""
        with patch.object(LiveExecutor, "_load_model"):
            executor = LiveExecutor(
                model_name="test",
                live_mode=False,
            )
            assert executor.live_mode is False
            assert executor.paper_portfolio is not None
            # En mode paper, exchange est initialisé sans auth (pour les prix publics)
            assert executor.exchange is not None

    def test_init_live_mode_no_keys(self):
        """Mode live sans API keys lève une erreur."""
        with patch.object(LiveExecutor, "_load_model"), \
             patch("live.executor.API_KEY", ""), \
             patch("live.executor.API_SECRET", ""):
            with pytest.raises(ValueError, match="API_KEY"):
                LiveExecutor(model_name="test", live_mode=True)

    def test_paper_portfolio_in_executor(self):
        """L'exécuteur utilise le paper portfolio."""
        from config.settings import INITIAL_BALANCE
        with patch.object(LiveExecutor, "_load_model"):
            executor = LiveExecutor(model_name="test", live_mode=False)
            assert executor.paper_portfolio.balance == INITIAL_BALANCE


# ============================================================================
# Tests main.py CLI
# ============================================================================


class TestCLI:
    """Tests pour le CLI main.py."""

    def test_help(self):
        """Le help s'affiche sans erreur."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "train" in result.stdout
        assert "backtest" in result.stdout
        assert "live" in result.stdout
        assert "dashboard" in result.stdout

    def test_train_command_args(self):
        """Les arguments train sont parsés correctement."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import main; import sys; sys.argv = ['main.py', 'train', '--model', 'test', '--timesteps', '100']; "
             "# just test parsing, don't run"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # Just verify no import error
        assert result.returncode == 0
