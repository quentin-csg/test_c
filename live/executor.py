"""
Exécuteur live / paper trading.

Boucle principale qui s'exécute toutes les heures :
  1. Fetch données récentes
  2. Calcul des features + normalisation
  3. Prédiction du modèle PPO
  4. Passage d'ordre (paper ou live via ccxt)
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from agent.model import load_agent, make_vec_env
from config.settings import (
    API_KEY,
    API_SECRET,
    EXCHANGE,
    EXECUTION_INTERVAL_SECONDS,
    FRAME_STACK_SIZE,
    INITIAL_BALANCE,
    LIVE_MODE,
    SYMBOL,
    TRADING_FEE,
)
from data.pipeline import build_full_pipeline
from training.logger import (
    append_weekly_csv,
    log_weekly_summary,
    print_stats,
)

logger = logging.getLogger(__name__)


class PaperPortfolio:
    """Simule un portefeuille pour le paper trading."""

    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0  # quantité BTC détenue
        self.entry_price = 0.0
        self.total_trades = 0
        self.total_fees = 0.0
        self.trade_history: list[dict] = []
        self.last_price = 0.0  # dernier prix connu (pour arrêt propre)

    def net_worth(self, current_price: float) -> float:
        """Valeur totale du portefeuille au prix courant."""
        return self.balance + self.position * current_price

    def execute_order(self, action: float, current_price: float) -> dict:
        """
        Exécute un ordre paper.

        Args:
            action: entre -1 (vendre 100%) et +1 (acheter 100%)
            current_price: prix actuel du BTC

        Returns:
            Dict avec les détails de l'ordre
        """
        self.last_price = current_price

        order_info = {
            "timestamp": datetime.now().isoformat(),
            "action": float(action),
            "price": current_price,
            "type": "hold",
            "amount": 0.0,
            "fee": 0.0,
        }

        if action > 0.05:  # Acheter (zone morte 5%)
            buy_amount_usdt = self.balance * action
            fee = buy_amount_usdt * TRADING_FEE
            net_amount = buy_amount_usdt - fee
            btc_bought = net_amount / current_price

            # Moyenner le prix d'entrée si position existante
            if self.position > 0:
                total_value = self.position * self.entry_price + btc_bought * current_price
                self.entry_price = total_value / (self.position + btc_bought)
            else:
                self.entry_price = current_price

            self.balance -= buy_amount_usdt
            self.position += btc_bought
            self.total_fees += fee
            self.total_trades += 1

            order_info.update({
                "type": "buy",
                "amount": btc_bought,
                "fee": fee,
            })

        elif action < -0.05:  # Vendre (zone morte 5%)
            sell_ratio = abs(action)
            btc_to_sell = self.position * sell_ratio
            sell_value = btc_to_sell * current_price
            fee = sell_value * TRADING_FEE

            self.balance += sell_value - fee
            self.position -= btc_to_sell
            self.total_fees += fee
            self.total_trades += 1

            order_info.update({
                "type": "sell",
                "amount": btc_to_sell,
                "fee": fee,
            })

        self.trade_history.append(order_info)
        return order_info

    def get_stats(self, current_price: float) -> dict:
        """Retourne les statistiques du portefeuille."""
        net_worth = self.balance + self.position * current_price
        return {
            "net_worth": net_worth,
            "balance_usdt": self.balance,
            "position_btc": self.position,
            "total_return_pct": (net_worth / self.initial_balance - 1) * 100,
            "total_trades": self.total_trades,
            "total_fees": self.total_fees,
        }


class LiveExecutor:
    """Boucle principale de trading live/paper."""

    def __init__(
        self,
        model_name: str = "ppo_trading",
        live_mode: bool = LIVE_MODE,
        symbol: str = SYMBOL,
        interval_seconds: int = EXECUTION_INTERVAL_SECONDS,
        frame_stack: int = FRAME_STACK_SIZE,
        include_nlp: bool = False,
    ):
        self.model_name = model_name
        self.live_mode = live_mode
        self.symbol = symbol
        self.interval = interval_seconds
        self.frame_stack = frame_stack
        self.include_nlp = include_nlp
        self.running = False

        # Paper portfolio (utilisé en mode paper)
        self.paper_portfolio = PaperPortfolio()

        # Exchange ccxt (utilisé en mode live)
        self.exchange = None
        if self.live_mode:
            self._init_exchange()

        mode_str = "LIVE" if self.live_mode else "PAPER"
        logger.info(f"LiveExecutor initialisé en mode {mode_str}")

    def _init_exchange(self):
        """Initialise la connexion à l'exchange pour le mode live."""
        if not API_KEY or not API_SECRET:
            raise ValueError(
                "API_KEY et API_SECRET requis pour le mode live. "
                "Définir EXCHANGE_API_KEY et EXCHANGE_API_SECRET."
            )

        exchange_class = getattr(ccxt, EXCHANGE)
        self.exchange = exchange_class({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
        })
        logger.info(f"Exchange {EXCHANGE} connecté")

    def _fetch_recent_data(self) -> tuple:
        """
        Récupère les données récentes et prépare les features.

        Returns:
            Tuple (DataFrame normalisé, FeatureScaler)

        Raises:
            RuntimeError: si les données sont vides ou insuffisantes
        """
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            dataset, scaler = build_full_pipeline(
                start=start,
                end=end,
                include_nlp=self.include_nlp,
                fit_scaler=False,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "Scaler non trouvé. Entraînez d'abord le modèle "
                "avec: python main.py train"
            ) from e

        if dataset.empty:
            raise RuntimeError(
                f"Aucune donnée récupérée pour {start} → {end}. "
                "Vérifiez la connexion internet."
            )

        return dataset, scaler

    def _get_current_price(self, df: pd.DataFrame) -> float:
        """Récupère le prix réel (non normalisé) depuis le DataFrame."""
        if "raw_close" in df.columns:
            return float(df["raw_close"].iloc[-1])
        return float(df["close"].iloc[-1])

    def _execute_live_order(self, action: float, current_price: float) -> dict:
        """Exécute un ordre réel sur l'exchange."""
        order_info = {
            "timestamp": datetime.now().isoformat(),
            "action": float(action),
            "price": current_price,
            "type": "hold",
        }

        if action > 0.05:
            # Acheter : market order
            balance = self.exchange.fetch_balance()
            usdt_available = balance["USDT"]["free"]
            buy_amount_usdt = usdt_available * action
            btc_amount = buy_amount_usdt / current_price

            if btc_amount > 0.00001:  # minimum order size
                order = self.exchange.create_market_buy_order(
                    self.symbol, btc_amount
                )
                order_info.update({"type": "buy", "order": order})
                logger.info(f"LIVE BUY: {btc_amount:.6f} BTC @ {current_price}")

        elif action < -0.05:
            # Vendre : market order
            balance = self.exchange.fetch_balance()
            btc_available = balance["BTC"]["free"]
            btc_to_sell = btc_available * abs(action)

            if btc_to_sell > 0.00001:
                order = self.exchange.create_market_sell_order(
                    self.symbol, btc_to_sell
                )
                order_info.update({"type": "sell", "order": order})
                logger.info(f"LIVE SELL: {btc_to_sell:.6f} BTC @ {current_price}")

        return order_info

    def tick(self) -> dict:
        """
        Exécute un cycle de trading (1 tick).

        Returns:
            Dict avec les détails de l'action effectuée
        """
        logger.info(f"=== Tick {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

        # 1. Fetch données récentes
        dataset, scaler = self._fetch_recent_data()
        if dataset.empty or len(dataset) < self.frame_stack + 1:
            logger.warning("Pas assez de données, tick ignoré")
            return {"action": "skip", "reason": "insufficient_data"}

        current_price = self._get_current_price(dataset)

        # 2. Créer l'environnement et charger le modèle
        vec_env = make_vec_env(
            df=dataset,
            n_envs=1,
            use_subproc=False,
            frame_stack=self.frame_stack,
        )

        try:
            agent = load_agent(vec_env, name=self.model_name)
        except Exception as e:
            vec_env.close()
            raise RuntimeError(
                f"Impossible de charger le modèle '{self.model_name}'. "
                f"Vérifiez que models/{self.model_name}.zip existe. "
                f"Erreur: {e}"
            ) from e

        # 3. Prédiction : on prend la dernière observation
        obs = vec_env.reset()
        # Avancer jusqu'à la fin des données
        for _ in range(len(dataset) - self.frame_stack - 1):
            obs, _, dones, _ = vec_env.step(np.array([[0.0]]))
            if dones[0]:
                break

        action, _ = agent.predict(obs, deterministic=True)
        action_value = float(action[0][0])
        vec_env.close()

        # 4. Exécuter l'ordre
        if self.live_mode:
            order_info = self._execute_live_order(action_value, current_price)
        else:
            order_info = self.paper_portfolio.execute_order(
                action_value, current_price
            )

        # Log
        mode_str = "LIVE" if self.live_mode else "PAPER"
        logger.info(
            f"[{mode_str}] Action: {action_value:+.4f} | "
            f"Type: {order_info['type']} | Prix: {current_price:.2f}"
        )

        if not self.live_mode:
            stats = self.paper_portfolio.get_stats(current_price)
            logger.info(
                f"  Portfolio: {stats['net_worth']:.2f} USDT | "
                f"Position: {stats['position_btc']:.6f} BTC | "
                f"Trades: {stats['total_trades']}"
            )

        return {
            "action": action_value,
            "price": current_price,
            "order": order_info,
            "timestamp": datetime.now().isoformat(),
        }

    def run(self):
        """Boucle principale — exécute un tick toutes les heures."""
        mode_str = "LIVE" if self.live_mode else "PAPER"
        print(f"=== Trading Bot démarré en mode {mode_str} ===")
        print(f"Modèle: {self.model_name}")
        print(f"Symbole: {self.symbol}")
        print(f"Intervalle: {self.interval}s ({self.interval // 3600}h)")
        print(f"Appuyez sur Ctrl+C pour arrêter.\n")

        self.running = True
        tick_count = 0

        try:
            while self.running:
                tick_count += 1
                try:
                    result = self.tick()
                    print(
                        f"[Tick {tick_count}] "
                        f"Action: {result['action']:+.4f} | "
                        f"Prix: {result['price']:.2f} | "
                        f"Type: {result['order'].get('type', 'skip')}"
                    )

                    if not self.live_mode:
                        stats = self.paper_portfolio.get_stats(result["price"])
                        print(
                            f"  Net worth: {stats['net_worth']:.2f} | "
                            f"Return: {stats['total_return_pct']:+.2f}%"
                        )

                except Exception as e:
                    logger.error(f"Erreur tick {tick_count}: {e}")
                    print(f"  ⚠ Erreur: {e}")

                # Attendre le prochain tick
                if self.running:
                    next_tick = datetime.now() + timedelta(seconds=self.interval)
                    print(f"  Prochain tick: {next_tick.strftime('%H:%M:%S')}")
                    time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n=== Arrêt du bot demandé ===")

        finally:
            self.running = False
            if not self.live_mode:
                # Sauvegarder les stats finales
                stats = self.paper_portfolio.get_stats(
                    self.paper_portfolio.last_price
                )
                print_stats(stats, title=f"Résultat final ({mode_str})")

                # Log dans les fichiers live
                log_weekly_summary(stats, mode="live")
                append_weekly_csv(stats, mode="live")

        print("Bot arrêté.")


def run_live(
    model_name: str = "ppo_trading",
    live_mode: bool = False,
    include_nlp: bool = False,
):
    """Point d'entrée pour lancer le trading live/paper."""
    executor = LiveExecutor(
        model_name=model_name,
        live_mode=live_mode,
        include_nlp=include_nlp,
    )
    executor.run()
