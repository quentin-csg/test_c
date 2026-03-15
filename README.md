# Trading Bot RL — Bot de trading crypto par Reinforcement Learning

Bot de trading crypto basé sur **PPO** (Proximal Policy Optimization) avec données multi-sources (OHLCV, macro, sentiment, news), analyse NLP via **FinBERT**, et trading live/paper. 100% gratuit et local.

---

## Architecture

```
Trading-bot/
├── config/
│   └── settings.py              # Configuration centralisee (exchange, hyperparams, chemins)
├── data/
│   ├── crypto_fetcher.py        # Prix & volumes via ccxt (Binance/Bybit)
│   ├── macro_fetcher.py         # QQQ, SPY via yfinance (forward-fill weekend)
│   ├── sentiment_fetcher.py     # Fear & Greed Index (Alternative.me)
│   ├── news_fetcher.py          # Flux RSS (CoinDesk, CoinTelegraph, Yahoo Finance)
│   └── pipeline.py              # Orchestrateur : merge toutes les sources
├── features/
│   ├── technical.py             # SMA, RSI, ATR, Bollinger, Funding Rate, OI
│   ├── nlp.py                   # FinBERT : sentiment des news (-1 a +1)
│   └── scaler.py                # RobustScaler : normalisation [-1, +1]
├── env/
│   └── trading_env.py           # Environnement Gymnasium (action continue [-1,+1])
├── agent/
│   ├── model.py                 # PPO + VecFrameStack + SubprocVecEnv
│   └── reward.py                # Sharpe, Sortino, drawdown, penalites
├── training/
│   ├── train.py                 # Entrainement PPO avec callbacks
│   ├── backtest.py              # Backtest sur donnees de test
│   └── logger.py                # Logs hebdo/mensuel + TensorBoard
├── live/
│   ├── executor.py              # Boucle horaire paper/live trading
│   ├── circuit_breaker.py       # Surveillance temps reel (coupe si crash)
│   └── dashboard.py             # Dashboard Streamlit
├── tests/                       # 114 tests unitaires
├── models/                      # Modeles sauvegardes (.zip)
├── logs/
│   ├── train/                   # Logs d'entrainement + TensorBoard
│   └── live/                    # Logs de trading live/paper
├── main.py                      # CLI principal
└── requirements.txt             # Dependances
```

---

## Installation

### Prerequis
- Python 3.10+
- pip

### Installation des dependances

```bash
pip install -r requirements.txt
```

**Dependances principales :**

| Package | Usage |
|---------|-------|
| ccxt | Donnees crypto + ordres live |
| yfinance | Donnees macro (QQQ, SPY) |
| pandas / pandas-ta | Indicateurs techniques |
| stable-baselines3 | Agent PPO |
| gymnasium | Environnement RL |
| transformers / torch | FinBERT (NLP) |
| streamlit | Dashboard local |
| scikit-learn | RobustScaler |
| feedparser | Flux RSS news |
| tensorboard | Visualisation entrainement |

> **Note FinBERT** : Le modele `ProsusAI/finbert` (~500 Mo) est telecharge automatiquement au premier lancement avec `--nlp`. Il necessite ~2 Go de RAM.

---

## Utilisation

### Entrainement

```bash
# Entrainement par defaut (1M steps, donnees 2020-2023)
python main.py train

# Avec options
python main.py train --model mon_modele --timesteps 500000

# Avec analyse NLP FinBERT
python main.py train --model v3_nlp --nlp
```

Le modele est sauvegarde dans `models/` et les logs TensorBoard dans `logs/train/tensorboard/`.

```bash
# Visualiser l'entrainement
tensorboard --logdir logs/train/tensorboard
```

### Backtest

```bash
# Backtest sur donnees 2024+ (utilise le scaler de l'entrainement)
python main.py backtest

# Avec un modele specifique
python main.py backtest --model mon_modele
```

Les resultats sont sauvegardes en JSON dans `logs/train/backtests/`.

### Paper Trading

```bash
# Paper trading (simulation, pas d'argent reel)
python main.py live --model ppo_trading
```

Le bot execute un tick toutes les heures :
1. Fetch donnees recentes (500 bougies)
2. Calcul des features + normalisation
3. Prediction du modele PPO
4. Execution de l'ordre (simulation)

Arret : `Ctrl+C`

### Live Trading

```bash
# ATTENTION : argent reel !
# Definir les cles API en variables d'environnement
$env:EXCHANGE_API_KEY = "votre_cle"
$env:EXCHANGE_API_SECRET = "votre_secret"

python main.py live --model ppo_trading --live-mode
```

> **Avertissement** : Toujours tester en paper trading avant de passer en live. Le bot est experimental.

### Dashboard

```bash
python main.py dashboard
```

Ouvre un dashboard Streamlit local (port 8501) avec 3 onglets :
- **Live/Paper** : KPIs (net worth, PnL, return) + graphique
- **Backtests** : Historique des resultats
- **Modeles** : Liste des modeles sauvegardes

---

## Pipeline de donnees

```
                    ┌─────────────────┐
                    │   ccxt          │  OHLCV 1h (BTC/USDT)
                    │   Binance/Bybit │  + Funding Rate
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   yfinance      │  QQQ, SPY (daily → forward-fill 1h)
                    │                 │  + flag is_weekend
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  Alternative.me │  Fear & Greed Index (daily → 1h)
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  RSS Feeds      │  News → FinBERT → score sentiment
                    │  (CoinDesk...)  │  (-1 negatif, +1 positif)
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  pipeline.py    │  Merge + Indicateurs + Scaler
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  TradingEnv     │  Gymnasium (obs → action → reward)
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  PPO Agent      │  stable-baselines3
                    └─────────────────┘
```

---

## Indicateurs techniques

| Indicateur | Periode | Description |
|------------|---------|-------------|
| SMA 50/200 | 50, 200 | Moyennes mobiles simples |
| RSI | 14 | Relative Strength Index |
| ATR | 14 | Average True Range (volatilite) |
| Bollinger Bands | 20, 2σ | Bandes + Z-Score |
| Funding Rate | — | Taux de financement futures |
| Open Interest | — | Interet ouvert (si disponible) |

---

## Reward (recompense RL)

La recompense combine :
- **Log-return** : variation du net worth (r = log(NW_t / NW_{t-1}))
- **Sharpe ratio** : rendement ajuste au risque sur fenetre glissante (24h)
- **Sortino ratio** : comme Sharpe mais penalise seulement la volatilite negative
- **Penalite drawdown** : exponentielle si drawdown > 15%
- **Penalite position sizing** : empeche les positions trop grandes
- **Penalite couts** : frais 0.1% + slippage [0%, 0.05%]

---

## Circuit Breaker

Surveillance continue du marche (toutes les minutes) :
- **Chute de prix** : coupe les positions si chute > 3% en 5 minutes
- **Volume anormal** : coupe si volume > 5x la moyenne

```python
from live.circuit_breaker import run_circuit_breaker
run_circuit_breaker(live_mode=False)  # paper mode
```

---

## Configuration

Tous les parametres sont dans `config/settings.py` :

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| EXCHANGE | binance | Exchange ccxt |
| SYMBOL | BTC/USDT | Paire de trading |
| TIMEFRAME | 1h | Timeframe principal |
| INITIAL_BALANCE | 10,000 USDT | Capital initial |
| TRADING_FEE | 0.1% | Frais par trade |
| FRAME_STACK_SIZE | 24 | Bougies en memoire |
| TOTAL_TIMESTEPS | 1,000,000 | Steps d'entrainement |
| N_ENVS | 4 | Envs paralleles |

Cles API (variables d'environnement) :
```
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...
```

---

## Tests

```bash
# Tous les tests (114)
python -m pytest tests/ -v

# Par module
python -m pytest tests/test_data.py -v         # 16 tests (data pipeline)
python -m pytest tests/test_features.py -v     # 22 tests (indicateurs + NLP)
python -m pytest tests/test_env.py -v          # 30 tests (environnement RL)
python -m pytest tests/test_agent.py -v        # 10 tests (PPO agent)
python -m pytest tests/test_training.py -v     # 16 tests (training + backtest)
python -m pytest tests/test_live.py -v         # 20 tests (live + circuit breaker)
```

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.10+ |
| RL Framework | Stable-Baselines3 (PPO) |
| Environnement | Gymnasium |
| Donnees crypto | ccxt (Binance, Bybit) |
| Donnees macro | yfinance |
| NLP | FinBERT (ProsusAI/finbert) |
| Indicateurs | pandas-ta |
| Normalisation | scikit-learn (RobustScaler) |
| Dashboard | Streamlit |
| Logs | TensorBoard + JSON/CSV |
| Tests | pytest (114 tests) |

---

## Licence

Projet personnel — usage libre.
