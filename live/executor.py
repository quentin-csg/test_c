"""
Exécuteur live / paper trading.

Boucle principale qui s'exécute toutes les heures :
  1. Fetch données récentes
  2. Calcul des features + normalisation
  3. Prédiction du modèle PPO
  4. Passage d'ordre (paper ou live via ccxt)
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from agent.model import MODEL_FEATURES
from config.settings import (
    API_KEY,
    API_SECRET,
    EXCHANGE,
    EXECUTION_INTERVAL_SECONDS,
    FRAME_STACK_SIZE,
    INITIAL_BALANCE,
    LIVE_MODE,
    LOGS_LIVE_DIR,
    MODELS_DIR,
    SYMBOL,
    TRADING_FEE,
)
from data.pipeline import build_full_pipeline
from features.scaler import FeatureScaler
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

        # Modèle et scaler chargés une seule fois
        self.agent = None
        self.scaler = None
        self.feature_columns = MODEL_FEATURES
        self._load_model()

        # Exchange ccxt (live: avec auth, paper: public uniquement)
        self.exchange = None
        if self.live_mode:
            self._init_exchange()
        else:
            # Exchange public (sans auth) pour la mise à jour du prix toutes les 10s
            exchange_class = getattr(ccxt, EXCHANGE)
            self.exchange = exchange_class({"enableRateLimit": True})

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

    def _load_model(self):
        """Charge le modèle PPO et le scaler une seule fois au démarrage."""
        # Charger le scaler sauvegardé pendant l'entraînement
        self.scaler = FeatureScaler()
        self.scaler.load()
        logger.info("Scaler chargé")

        # Charger le modèle PPO (sans env — on construit les obs manuellement)
        model_path = MODELS_DIR / self.model_name
        zip_path = Path(f"{model_path}.zip")
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable: {zip_path}. "
                f"Entraînez d'abord avec: python main.py train --model {self.model_name}"
            )
        self.agent = PPO.load(str(model_path))
        logger.info(f"Modèle PPO chargé: {zip_path}")

    def _build_observation(self, dataset: pd.DataFrame) -> np.ndarray:
        """
        Construit l'observation empilée (frame_stack) à partir des données récentes.

        Reproduit exactement la logique de TradingEnv._get_obs() + VecFrameStack :
        - Extrait les features du marché + 3 features portfolio
        - Empile les `frame_stack` dernières observations

        Returns:
            np.ndarray de shape (1, frame_stack * n_features)
        """
        n_rows = len(dataset)
        if n_rows < self.frame_stack:
            raise ValueError(
                f"Pas assez de données ({n_rows}) pour frame_stack={self.frame_stack}"
            )

        # Sélectionner les colonnes de features disponibles
        available_cols = [c for c in self.feature_columns if c in dataset.columns]

        # Construire les observations empilées
        frames = []
        for i in range(n_rows - self.frame_stack, n_rows):
            # Features du marché
            market = dataset.iloc[i][available_cols].values.astype(np.float32)

            # Features portfolio (depuis PaperPortfolio)
            current_price = self._get_current_price_at(dataset, i)
            nw = self.paper_portfolio.net_worth(current_price)
            balance_ratio = self.paper_portfolio.balance / self.paper_portfolio.initial_balance - 1.0
            position_value = self.paper_portfolio.position * current_price
            position_ratio = position_value / max(nw, 1e-8)
            unrealized_pnl = 0.0
            if self.paper_portfolio.position > 0 and self.paper_portfolio.entry_price > 0:
                unrealized_pnl = (current_price - self.paper_portfolio.entry_price) / self.paper_portfolio.entry_price

            portfolio = np.array([balance_ratio, position_ratio, unrealized_pnl], dtype=np.float32)
            obs = np.concatenate([market, portfolio])
            frames.append(obs)

        # Empiler et aplatir comme VecFrameStack
        stacked = np.concatenate(frames)
        return stacked.reshape(1, -1)

    def _get_current_price_at(self, df: pd.DataFrame, idx: int) -> float:
        """Récupère le prix réel à un index donné."""
        if "raw_close" in df.columns:
            return float(df["raw_close"].iloc[idx])
        return float(df["close"].iloc[idx])

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

    def _save_live_state(self, dataset: pd.DataFrame, current_price: float) -> None:
        """
        Sauvegarde l'état courant du portfolio dans live_state.json.
        Utilisé par le dashboard pour afficher le graphe BTC et les positions.
        """
        LOGS_LIVE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = LOGS_LIVE_DIR / "live_state.json"

        stats = self.paper_portfolio.get_stats(current_price)
        position_usdt = stats["position_btc"] * current_price

        # Historique de prix : dernières 168 bougies (1 semaine H1) — prix réel
        price_col = "raw_close" if "raw_close" in dataset.columns else "close"
        ts_col = "timestamp" if "timestamp" in dataset.columns else None
        price_history = []
        for _, row in dataset[[ts_col, price_col]].tail(168).iterrows() if ts_col else dataset[[price_col]].tail(168).iterrows():
            entry = {"price": float(row[price_col])}
            if ts_col:
                entry["timestamp"] = str(row[ts_col])
            price_history.append(entry)

        # Position ouverte courante
        open_positions = []
        if self.paper_portfolio.position > 1e-8:
            entry_price = self.paper_portfolio.entry_price
            unrealized_pnl_pct = (
                (current_price - entry_price) / entry_price * 100
                if entry_price > 0 else 0.0
            )
            open_positions.append({
                "entry_price": entry_price,
                "amount_btc": self.paper_portfolio.position,
                "value_usdt": position_usdt,
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 4),
            })

        # Trades fermés (toutes les ventes)
        closed_trades = [
            t for t in self.paper_portfolio.trade_history
            if t.get("type") == "sell"
        ]

        state = {
            "last_updated": datetime.now().isoformat(),
            "current_price": current_price,
            "price_history": price_history,
            "portfolio": {
                "net_worth": stats["net_worth"],
                "balance_usdt": stats["balance_usdt"],
                "position_btc": stats["position_btc"],
                "position_usdt": position_usdt,
                "total_return_pct": stats["total_return_pct"],
                "total_trades": stats["total_trades"],
                "total_fees": stats["total_fees"],
            },
            "open_positions": open_positions,
            "closed_trades": closed_trades,
        }

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

    def tick(self) -> dict:
        """
        Exécute un cycle de trading (1 tick).

        Le modèle est chargé une seule fois dans __init__. À chaque tick,
        on fetch les données récentes, construit l'observation directement,
        et prédit l'action — sans recréer d'environnement.

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

        # 2. Construire l'observation directement (sans env)
        obs = self._build_observation(dataset)

        # 3. Prédiction
        action, _ = self.agent.predict(obs, deterministic=True)
        action_value = float(action[0][0]) if action.ndim > 1 else float(action[0])

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
            position_usdt = stats["position_btc"] * current_price
            logger.info(
                f"  Portfolio: {stats['net_worth']:.2f} USDT | "
                f"Position: {stats['position_btc']:.6f} BTC "
                f"(≈ {position_usdt:.2f} USDT) | "
                f"Trades: {stats['total_trades']}"
            )
            self._save_live_state(dataset, current_price)

        return {
            "action": action_value,
            "price": current_price,
            "order": order_info,
            "timestamp": datetime.now().isoformat(),
        }

    def _price_update_loop(self, interval_seconds: int = 10) -> None:
        """
        Thread background : met à jour le prix BTC dans live_state.json
        toutes les `interval_seconds` secondes, sans déclencher de trade.
        """
        state_path = LOGS_LIVE_DIR / "live_state.json"
        while self.running:
            try:
                current_price = float(
                    self.exchange.fetch_ticker(self.symbol)["last"]
                )
                if state_path.exists():
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f)

                    # Mettre à jour le prix et l'historique
                    state["current_price"] = current_price
                    state["last_updated"] = datetime.now().isoformat()

                    price_history = state.get("price_history", [])
                    price_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                    })
                    # Garder seulement les 7 derniers jours (168h × 6 points/h = ~1000)
                    state["price_history"] = price_history[-1000:]

                    # Recalculer les métriques portfolio avec le prix live
                    portfolio = state.get("portfolio", {})
                    position_btc = portfolio.get("position_btc", 0.0)
                    balance_usdt = portfolio.get("balance_usdt", 0.0)
                    if position_btc > 0:
                        position_usdt = position_btc * current_price
                        net_worth = balance_usdt + position_usdt
                        portfolio["position_usdt"] = round(position_usdt, 4)
                        portfolio["net_worth"] = round(net_worth, 4)
                        portfolio["total_return_pct"] = round(
                            (net_worth - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 4
                        )
                        state["portfolio"] = portfolio

                    # Recalculer PnL latent des positions ouvertes
                    open_positions = state.get("open_positions", [])
                    for pos in open_positions:
                        entry = pos.get("entry_price", 0)
                        if entry > 0:
                            pos["unrealized_pnl_pct"] = round(
                                (current_price - entry) / entry * 100, 4
                            )
                            pos["value_usdt"] = round(
                                pos.get("amount_btc", 0) * current_price, 4
                            )
                    state["open_positions"] = open_positions

                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2, default=str)

            except Exception as e:
                logger.debug(f"Price updater: {e}")

            time.sleep(interval_seconds)

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

        # Thread background : mise à jour du prix toutes les 10s
        price_thread = threading.Thread(
            target=self._price_update_loop,
            args=(10,),
            daemon=True,
        )
        price_thread.start()

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
                        position_usdt = stats["position_btc"] * result["price"]
                        print(
                            f"  Net worth: {stats['net_worth']:.2f} USDT | "
                            f"Position: {stats['position_btc']:.6f} BTC "
                            f"(≈ {position_usdt:.2f} USDT) | "
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
