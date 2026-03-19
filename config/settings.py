"""
Configuration centralisée du Trading Bot.
Tous les paramètres modifiables sont ici. Aucun magic number dans le code.
"""

import os
from pathlib import Path


# =============================================================================
# CHEMINS DU PROJET
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
VECNORM_PATH = MODELS_DIR / "vec_normalize.pkl"

# =============================================================================
# DONNÉES CRYPTO (ccxt)
# =============================================================================
EXCHANGE = "binance"               # Exchange à utiliser (binance, bybit)
SYMBOL = "BTC/USDT"                # Paire de trading
TIMEFRAME = "1h"                   # Timeframe principal
TIMEFRAME_SECONDARY = "4h"         # Timeframe secondaire (pour VecFrameStack)

# Clés API (à remplir pour le mode live — JAMAIS commiter en clair)
API_KEY = os.getenv("EXCHANGE_API_KEY", "")
API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")

# =============================================================================
# DONNÉES MACRO (yfinance)
# =============================================================================
MACRO_SYMBOLS = ["QQQ", "SPY"]     # ETFs macro à suivre (NASDAQ, S&P500)

# =============================================================================
# SENTIMENT (Alternative.me)
# =============================================================================
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"

# =============================================================================
# NEWS (RSS)
# =============================================================================
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://finance.yahoo.com/news/rssindex",
]

# Mots-clés pour filtrer les articles pertinents
NEWS_KEYWORDS = ["BTC", "Bitcoin", "FED", "ETF", "SEC", "crypto", "inflation",
                 "interest rate", "FOMC", "regulation"]

# =============================================================================
# NLP — FinBERT
# =============================================================================
FINBERT_MODEL = "ProsusAI/finbert"
NLP_MAX_ARTICLES = 20              # Nombre max d'articles à analyser par heure
NLP_BATCH_SIZE = 8                 # Batch size pour l'inférence FinBERT

# =============================================================================
# INDICATEURS TECHNIQUES (pandas-ta)
# =============================================================================
SMA_SHORT = 50
SMA_LONG = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14

# =============================================================================
# ENVIRONNEMENT GYMNASIUM
# =============================================================================
INITIAL_BALANCE = 5_000.0          # Capital initial en USDT
TRADING_FEE = 0.001                # Frais par trade (0.1%)
SLIPPAGE_MIN = 0.0001              # Slippage minimum (0.01%)
SLIPPAGE_MAX = 0.001               # Slippage maximum (0.1%)

# =============================================================================
# AGENT RL — PPO (Stable-Baselines3)
# =============================================================================
PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# VecFrameStack — nombre de bougies empilées en mémoire
FRAME_STACK_SIZE = 24              # 24 bougies H1 = 1 jour de mémoire

# Architecture réseau
USE_CNN = True                     # True = CNN 1D, False = MLP classique
CNN_FEATURES_DIM = 128             # Dimension de sortie du CNN feature extractor

# =============================================================================
# REWARD (récompense)
# =============================================================================
SHARPE_WINDOW = 72                 # Fenêtre glissante pour Sharpe/Sortino (3 jours H1)
MAX_DRAWDOWN_PENALTY = 0.15        # Seuil de drawdown max avant pénalité exponentielle
DRAWDOWN_PENALTY_CAP = 2.0         # Cap maximum de la pénalité de drawdown
POSITION_SIZE_PENALTY_FACTOR = 0.1 # Facteur de pénalité pour grosses positions
POSITION_SIZE_THRESHOLD = 0.4      # position threshold above which penalty applies (40%)
PRICE_POSITION_WINDOW = 20         # window for price vs high/low features

# =============================================================================
# ENTRAÎNEMENT
# =============================================================================
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = None                    # None = jusqu'à aujourd'hui

TOTAL_TIMESTEPS = 1_000_000        # Nombre total de pas d'entraînement
N_ENVS = 4                        # Nombre d'environnements parallèles (SubprocVecEnv)
CHECKPOINT_FREQ = 50_000           # Sauvegarde du modèle tous les N pas
EARLY_STOPPING_PATIENCE = 5        # Arrêter si pas d'amélioration après N checks

# Walk-Forward Validation
WF_TRAIN_MONTHS = 24               # Durée initiale du train (mois)
WF_TEST_MONTHS = 3                 # Durée du test (mois)
WF_STEP_MONTHS = 3                 # Pas d'avancement (mois)

# Logs séparés train / live
LOGS_TRAIN_DIR = LOGS_DIR / "train"
LOGS_LIVE_DIR = LOGS_DIR / "live"
TENSORBOARD_LOG_DIR = str(LOGS_TRAIN_DIR / "tensorboard")

# =============================================================================
# LIVE / PAPER TRADING
# =============================================================================
LIVE_MODE = False                  # True = trading réel, False = paper trading
EXECUTION_INTERVAL_SECONDS = 3600  # Exécution toutes les heures (3600s)

# =============================================================================
# CIRCUIT BREAKER
# =============================================================================
CB_PRICE_DROP_THRESHOLD = 0.03    # Coupe si chute > 3% sur la fenêtre
CB_VOLUME_SPIKE_FACTOR = 5.0      # Coupe si volume > 5x la moyenne
CB_CHECK_INTERVAL = "1m"          # Timeframe de surveillance
CB_LOOKBACK_MINUTES = 30          # Fenêtre de détection (30 dernières minutes)
STOP_LOSS_PCT = 3.0               # Stop-loss par position : fermer si PnL latent < -3%
MAX_POSITION_PCT = 0.30           # Taille max d'un achat : 30% du cash disponible

# =============================================================================
# DASHBOARD (Streamlit)
# =============================================================================
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_SECONDS = 10    # Rafraîchissement auto du dashboard
