"""
Tests unitaires pour l'environnement Gymnasium et les fonctions de reward.
"""

import unittest

import numpy as np
import pandas as pd


class TestReward(unittest.TestCase):
    """Tests des fonctions de récompense."""

    def test_import(self):
        from agent.reward import (
            compute_reward,
            drawdown_penalty,
            log_return_reward,
            position_size_penalty,
            sharpe_reward,
            sortino_reward,
            transaction_cost_penalty,
        )
        self.assertTrue(callable(compute_reward))

    def test_log_return_positive(self):
        from agent.reward import log_return_reward
        reward = log_return_reward(10500, 10000)
        self.assertGreater(reward, 0)
        self.assertAlmostEqual(reward, np.log(10500 / 10000), places=6)

    def test_log_return_negative(self):
        from agent.reward import log_return_reward
        reward = log_return_reward(9500, 10000)
        self.assertLess(reward, 0)

    def test_log_return_zero_worth(self):
        from agent.reward import log_return_reward
        reward = log_return_reward(0, 10000)
        self.assertEqual(reward, -1.0)

    def test_sharpe_not_enough_data(self):
        from agent.reward import sharpe_reward
        result = sharpe_reward([0.01], window=24)
        self.assertEqual(result, 0.0)

    def test_sharpe_with_data(self):
        from agent.reward import sharpe_reward
        returns = list(np.random.randn(30) * 0.01 + 0.001)
        result = sharpe_reward(returns, window=24)
        self.assertIsInstance(result, float)

    def test_sortino_positive_returns(self):
        from agent.reward import sortino_reward
        returns = [0.01, 0.02, 0.005, 0.01, 0.015, 0.008, 0.012]
        result = sortino_reward(returns, window=24)
        self.assertGreater(result, 0)

    def test_drawdown_no_loss(self):
        from agent.reward import drawdown_penalty
        penalty = drawdown_penalty(10000, 10000)
        self.assertEqual(penalty, 0.0)

    def test_drawdown_small_loss(self):
        from agent.reward import drawdown_penalty
        penalty = drawdown_penalty(9500, 10000, threshold=0.15)
        self.assertLess(penalty, 0)

    def test_drawdown_exponential(self):
        from agent.reward import drawdown_penalty
        # Au-delà du seuil, la pénalité doit être plus sévère
        mild = drawdown_penalty(9000, 10000, threshold=0.15)
        severe = drawdown_penalty(7000, 10000, threshold=0.15)
        self.assertLess(severe, mild)

    def test_position_size_penalty(self):
        from agent.reward import position_size_penalty
        # En dessous du seuil (40%) → aucune pénalité
        self.assertEqual(position_size_penalty(0.1), 0.0)
        self.assertEqual(position_size_penalty(0.4), 0.0)
        # Au-dessus du seuil → pénalité négative
        large = position_size_penalty(0.6)
        self.assertLess(large, 0.0)
        # Plus la position est grande, plus la pénalité est grande
        larger = position_size_penalty(1.0)
        self.assertLess(larger, large)

    def test_transaction_cost_penalty(self):
        from agent.reward import transaction_cost_penalty
        penalty = transaction_cost_penalty(10, 10000)
        self.assertAlmostEqual(penalty, -0.001, places=4)

    def test_compute_reward_returns_tuple(self):
        from agent.reward import compute_reward
        total, components = compute_reward(
            net_worth=10100,
            prev_net_worth=10000,
            peak_net_worth=10100,
            position_ratio=0.5,
            trade_cost=5.0,
            returns_history=[0.01, 0.005, -0.002, 0.003],
        )
        self.assertIsInstance(total, float)
        self.assertIsInstance(components, dict)
        self.assertIn("log_return", components)
        self.assertIn("sortino", components)   # Sortino remplace Sharpe
        self.assertNotIn("sharpe", components) # Sharpe supprimé (redondant)
        self.assertIn("drawdown", components)
        self.assertIn("position_size", components)
        self.assertIn("transaction", components)


class TestTradingEnv(unittest.TestCase):
    """Tests de l'environnement Gymnasium TradingEnv."""

    @classmethod
    def setUpClass(cls):
        """Crée un DataFrame de test avec des données synthétiques."""
        np.random.seed(42)
        n = 200
        prices = 42000 + np.cumsum(np.random.randn(n) * 100)
        cls.test_df = pd.DataFrame({
            "close": prices,
            "raw_close": prices.copy(),  # ADD THIS LINE
            "open": prices + np.random.randn(n) * 50,
            "high": prices + abs(np.random.randn(n) * 100),
            "low": prices - abs(np.random.randn(n) * 100),
            "volume": np.abs(np.random.randn(n) * 500 + 1000),
            "rsi": np.random.rand(n) * 2 - 1,  # déjà normalisé
            "sma_trend": np.random.choice([-1, 1], n).astype(float),
        })

    def _make_env(self, **kwargs):
        from env.trading_env import TradingEnv
        return TradingEnv(df=self.test_df.copy(), **kwargs)

    def test_import(self):
        from env.trading_env import TradingEnv
        self.assertTrue(callable(TradingEnv))

    def test_init(self):
        env = self._make_env()
        self.assertIsNotNone(env.observation_space)
        self.assertIsNotNone(env.action_space)

    def test_reset(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (env.n_obs,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(info["balance"], 5000.0)
        self.assertEqual(info["position"], 0.0)

    def test_step_hold(self):
        """Action ~0 = hold, pas de trade."""
        env = self._make_env()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        self.assertEqual(info["total_trades"], 0)
        self.assertAlmostEqual(info["balance"], 5000.0)

    def test_step_buy(self):
        """Achat avec action positive."""
        env = self._make_env()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
        self.assertEqual(info["total_trades"], 1)
        self.assertGreater(info["position"], 0)
        self.assertLess(info["balance"], 5000.0)

    def test_step_buy_then_sell(self):
        """Achat puis vente complète."""
        env = self._make_env()
        env.reset(seed=42)

        # Acheter 100%
        env.step(np.array([1.0]))
        self.assertGreater(env.position, 0)

        # Vendre 100%
        env.step(np.array([-1.0]))
        self.assertAlmostEqual(env.position, 0.0, places=8)
        self.assertEqual(env.total_trades, 2)

    def test_fees_applied(self):
        """Vérifie que les frais sont bien déduits."""
        env = self._make_env(trading_fee=0.001, slippage_min=0.0, slippage_max=0.0)
        env.reset(seed=42)

        # Achat avec action=1.0, capé à 30% du cash par MAX_POSITION_PCT
        env.step(np.array([1.0]))

        # Les frais doivent avoir été payés
        self.assertGreater(env.total_fees_paid, 0)
        # Après un achat (30% max), le balance doit avoir baissé mais pas à zéro
        self.assertLess(env.balance, 5000.0)   # balance a diminué
        self.assertGreater(env.balance, 3000.0)  # ~70% du cash reste
        # La position BTC doit être non-nulle
        self.assertGreater(env.position, 0)

    def test_slippage_applied(self):
        """Vérifie que le slippage affecte le prix d'exécution."""
        env = self._make_env(slippage_min=0.01, slippage_max=0.01)
        env.reset(seed=42)
        env.step(np.array([1.0]))
        # Avec 1% de slippage, l'entry price doit être > prix marché
        self.assertGreater(env.entry_price, env.prices[env.start_step])

    def test_episode_complete(self):
        """Vérifie qu'un épisode se termine correctement."""
        env = self._make_env()
        env.reset(seed=42)

        terminated = False
        steps = 0
        while not terminated:
            action = np.array([env.np_random.uniform(-0.3, 0.3)])
            _, _, terminated, truncated, info = env.step(action)
            steps += 1

        self.assertGreater(steps, 0)
        self.assertEqual(steps, len(self.test_df) - 1 - env.start_step)

    def test_portfolio_stats(self):
        """Vérifie le calcul des statistiques de portfolio."""
        env = self._make_env()
        env.reset(seed=42)

        for _ in range(50):
            action = np.array([env.np_random.uniform(-0.5, 0.5)])
            env.step(action)

        stats = env.get_portfolio_stats()
        self.assertIn("total_return_pct", stats)
        self.assertIn("max_drawdown_pct", stats)
        self.assertIn("sharpe_ratio", stats)
        self.assertIn("total_trades", stats)

    def test_ruin_terminates(self):
        """Vérifie que l'épisode se termine si le portfolio est ruiné."""
        # Créer un env avec prix qui chute massivement (90%+ de perte)
        n = 100
        prices = [10000 * (0.95 ** i) for i in range(n)]  # -5% par step
        df = pd.DataFrame({
            "close": prices,
            "rsi": [0.0] * n,
        })
        from env.trading_env import TradingEnv
        env = TradingEnv(df=df, initial_balance=10000.0)
        env.reset(seed=42)

        # Acheter 100% et laisser le prix chuter
        env.step(np.array([1.0]))

        terminated = False
        for _ in range(90):
            _, _, terminated, _, info = env.step(np.array([0.0]))
            if terminated:
                break

        self.assertTrue(terminated)
        # Nouveau seuil de ruine = 80% du capital initial
        self.assertLess(info["net_worth"], 10000.0 * 0.80)

    def test_observation_space_check(self):
        """Vérifie que l'observation est dans l'espace défini."""
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))

    def test_action_space_check(self):
        """Vérifie que les actions sont dans l'espace défini."""
        env = self._make_env()
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            self.assertTrue(env.action_space.contains(action))
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
            self.assertTrue(env.observation_space.contains(obs))

    def test_render_human(self):
        """Vérifie que le render en mode human ne crashe pas."""
        env = self._make_env(render_mode="human")
        env.reset(seed=42)
        env.step(np.array([0.5]))  # Doit afficher dans la console

    def test_dead_zone(self):
        """Action < 5% = pas de trade."""
        env = self._make_env()
        env.reset(seed=42)
        env.step(np.array([0.03]))
        self.assertEqual(env.total_trades, 0)

    def test_custom_feature_columns(self):
        """Vérifie qu'on peut spécifier les colonnes manuellement."""
        env = self._make_env(feature_columns=["close", "rsi"])
        obs, _ = env.reset(seed=42)
        # 2 features marché + 3 portfolio = 5
        self.assertEqual(obs.shape, (5,))

    def test_gymnasium_api_check(self):
        """Vérifie la conformité basique avec l'API Gymnasium."""
        env = self._make_env()
        obs, info = env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)

        result = env.step(env.action_space.sample())
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_raw_close_used_for_prices(self):
        """raw_close doit être utilisé comme prix si disponible."""
        n = 50
        raw_prices = 40000 + np.arange(n, dtype=float)
        normalized_prices = np.linspace(-0.5, 0.5, n)  # prices normalisées incorrectes
        df = pd.DataFrame({
            "close": normalized_prices,
            "raw_close": raw_prices,
            "rsi": [0.0] * n,
        })
        from env.trading_env import TradingEnv
        env = TradingEnv(df=df, min_episode_steps=10)
        # Les prix utilisés doivent être les raw_close
        self.assertAlmostEqual(env.prices[0], 40000.0)
        self.assertNotAlmostEqual(env.prices[0], -0.5, places=3)

    def test_random_start_within_bounds(self):
        """Le start aléatoire doit être dans une plage valide."""
        env = self._make_env()
        for seed in range(10):
            env.reset(seed=seed)
            self.assertGreaterEqual(env.start_step, 0)
            self.assertLessEqual(
                env.start_step,
                max(0, len(self.test_df) - env.min_episode_steps - 1)
            )

    def test_random_start_deterministic_with_seed(self):
        """Même seed → même start_step."""
        env = self._make_env()
        env.reset(seed=99)
        start1 = env.start_step
        env.reset(seed=99)
        start2 = env.start_step
        self.assertEqual(start1, start2)

    def test_unrealized_pnl_clipped(self):
        """unrealized_pnl doit être clippé entre -3 et +3."""
        n = 50
        prices = [10000.0] * n
        prices[0] = 1.0  # prix d'entrée très bas pour créer un PnL énorme
        df = pd.DataFrame({
            "close": prices,
            "rsi": [0.0] * n,
        })
        from env.trading_env import TradingEnv
        env = TradingEnv(df=df, initial_balance=5000.0, min_episode_steps=10)
        env.reset(seed=42)
        # Forcer un entry_price très bas
        env.entry_price = 1.0
        env.position = 0.1
        obs = env._get_obs()
        # unrealized_pnl est la 3ème feature portfolio (derniers 3 éléments)
        unrealized_pnl = obs[-1]
        self.assertLessEqual(unrealized_pnl, 3.0)
        self.assertGreaterEqual(unrealized_pnl, -3.0)

    def test_position_size_no_small_penalty(self):
        """Petites positions (< 40%) ne doivent pas être pénalisées."""
        from agent.reward import position_size_penalty
        self.assertEqual(position_size_penalty(0.0), 0.0)
        self.assertEqual(position_size_penalty(0.3), 0.0)
        self.assertEqual(position_size_penalty(0.4), 0.0)
        self.assertLess(position_size_penalty(0.5), 0.0)


class TestRewardFixes(unittest.TestCase):
    """Tests pour les corrections de rewards (Steps 1)."""

    def test_drawdown_penalty_capped(self):
        """La pénalité ne dépasse jamais -DRAWDOWN_PENALTY_CAP."""
        from agent.reward import drawdown_penalty
        from config.settings import DRAWDOWN_PENALTY_CAP
        # 80% drawdown — sans cap: exp(0.8/0.15)-1 ≈ 205
        penalty = drawdown_penalty(2000, 10000, threshold=0.15)
        self.assertGreaterEqual(penalty, -DRAWDOWN_PENALTY_CAP)
        self.assertLess(penalty, 0)

    def test_sharpe_annualized(self):
        """Le Sharpe reward est annualisé (facteur sqrt(8760))."""
        from agent.reward import sharpe_reward
        # Returns avec variation non-nulle (std > 0) et moyenne positive
        returns = [0.001 + 0.0002 * (i % 5 - 2) for i in range(30)]
        result = sharpe_reward(returns, window=24)
        # Avec annualisation sqrt(8760) ≈ 93.6, le résultat doit être > 1
        self.assertGreater(result, 1.0)
        # Vérifier l'annualisation : Sharpe non-annualisé serait bien plus petit
        raw_returns = np.array(returns[-24:])
        raw_sharpe = float(np.mean(raw_returns) / np.std(raw_returns))
        self.assertAlmostEqual(result, raw_sharpe * float(np.sqrt(8760)), places=4)

    def test_sortino_no_downside_capped(self):
        """Sortino sans returns négatifs est cappé à 3.0."""
        from agent.reward import sortino_reward
        # Tous les returns positifs
        returns = [0.001] * 30
        result = sortino_reward(returns, window=24)
        self.assertLessEqual(result, 3.0)
        self.assertGreater(result, 0)

    def test_sortino_no_downside_zero_mean(self):
        """Sortino retourne 0 si mean est 0 et pas de downside."""
        from agent.reward import sortino_reward
        returns = [0.0] * 30
        result = sortino_reward(returns, window=24)
        self.assertEqual(result, 0.0)

    def test_position_size_threshold(self):
        """La pénalité position ne s'applique qu'au-delà du seuil."""
        from agent.reward import position_size_penalty
        from config.settings import POSITION_SIZE_THRESHOLD
        # Exactement au seuil → 0
        self.assertEqual(position_size_penalty(POSITION_SIZE_THRESHOLD), 0.0)
        # Légèrement au-dessus → pénalité
        above = position_size_penalty(POSITION_SIZE_THRESHOLD + 0.1)
        self.assertLess(above, 0.0)


if __name__ == "__main__":
    unittest.main()
