# Trading Bot RL — Bot de trading crypto par Reinforcement Learning

Bot de trading crypto basé sur **PPO** (Proximal Policy Optimization) avec données multi-sources (OHLCV, macro, sentiment, news), analyse NLP via **FinBERT**, CNN 1D feature extractor, walk-forward validation, et trading live/paper. 100% gratuit et local.

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
│   └── pipeline.py              # Orchestrateur : merge toutes les sources + 4h
├── features/
│   ├── technical.py             # SMA, RSI, ATR, Bollinger, MACD, ADX, multi-TF 4h
│   ├── nlp.py                   # FinBERT : sentiment des news (-1 a +1)
│   └── scaler.py                # RobustScaler : normalisation [-1, +1]
├── env/
│   └── trading_env.py           # Environnement Gymnasium (action continue [-1,+1])
├── agent/
│   ├── model.py                 # PPO + CNN 1D + VecNormalize + VecFrameStack
│   └── reward.py                # Sharpe annualise, Sortino, drawdown cappe, penalites
├── training/
│   ├── train.py                 # Entrainement PPO + early stopping
│   ├── backtest.py              # Backtest sur donnees de test
│   ├── walk_forward.py          # Walk-forward validation (expanding window)
│   └── logger.py                # Logs hebdo/mensuel/walk-forward + TensorBoard
├── live/
│   ├── executor.py              # Boucle horaire paper/live (modele charge 1x)
│   ├── circuit_breaker.py       # Surveillance temps reel (coupe si crash)
│   └── dashboard.py             # Dashboard Streamlit
├── tests/                       # Tests unitaires
├── test_phases/                 # Documentation par phase
├── models/                      # Modeles sauvegardes (.zip + vec_normalize.pkl)
├── logs/
│   ├── train/                   # Logs d'entrainement + TensorBoard
│   ├── live/                    # Logs de trading live/paper
│   └── walk_forward/            # Resultats walk-forward par fold
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
| transformers / torch | FinBERT (NLP) + CNN 1D |
| streamlit | Dashboard local |
| scikit-learn | RobustScaler |
| feedparser | Flux RSS news |
| tensorboard | Visualisation entrainement |
| python-dateutil | Walk-forward date arithmetic |

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

Le modele est sauvegarde dans `models/` avec son `vec_normalize.pkl`. Les logs TensorBoard sont dans `logs/train/tensorboard/`.

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

### Walk-Forward Validation

```bash
# Validation statistique sur fenetres glissantes
python main.py walk-forward
```

Entraine et backteste sur `N` folds avec expanding window (defaut : train=24 mois, test=3 mois, step=3 mois). Les metriques agregees (mean/std Sharpe, return, drawdown) sont sauvegardees dans `logs/walk_forward/`.

### Paper Trading

```bash
# Paper trading (simulation, pas d'argent reel)
python main.py live --model ppo_trading
```

Le bot execute un tick toutes les heures :
1. Fetch donnees recentes (500 bougies)
2. Calcul des features + normalisation
3. Construction de l'observation stackee (frame_stack × n_features)
4. Prediction du modele PPO (charge une seule fois)
5. Execution de l'ordre (simulation)

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

Ouvre un dashboard Streamlit local (port 8501) avec les onglets :
- **Live/Paper** : KPIs (net worth, PnL, return) + graphique
- **Backtests** : Historique des resultats
- **Modeles** : Liste des modeles sauvegardes

---

## Pipeline de donnees

```
                    ┌─────────────────┐
                    │   ccxt          │  OHLCV 1h + 4h (BTC/USDT)
                    │   Binance/Bybit │  + Funding Rate + OB imbalance + OI
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
                    │  pipeline.py    │  Merge + MACD/ADX/multi-TF + Scaler
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  TradingEnv     │  Gymnasium (obs → action → reward)
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  PPO + CNN 1D   │  stable-baselines3 + extractor custom
                    └─────────────────┘
```

---

## Feature sets progressifs

| Set | N features | Contenu |
| --- | ---------- | ------- |
| **V1** | 7 | RSI normalisé, SMA trend, price_to_sma_long, ATR%, BB position, BB bandwidth, log_return |
| **V2** | 14 | V1 + volume_ratio, volume_direction, log_return_5h, log_return_24h, fear_greed_normalized, funding_rate, is_weekend |
| **V3** ✅ | 23 | V2 + macd_hist_normalized, adx_normalized, rsi_4h_normalized, sma_trend_4h, candle_body, upper_wick_ratio, lower_wick_ratio, price_to_high_20, price_to_low_20 — **set par défaut** |

**Features live-only** (signal temps réel, poids ~0 au training) : `orderbook_imbalance`, `oi_change_pct`

Avec `frame_stack=24` et V3 : 23 × 24 = 552 dimensions d'entrée pour le CNN 1D (+ 3 portfolio = 26 × 24 = 624 total).

---

## Indicateurs techniques

| Indicateur | Periode | Feature normalisée |
| ---------- | ------- | ------------------ |
| SMA 50/200 | 50, 200 | `sma_trend` (+1/-1), `price_to_sma_long` |
| RSI | 14 | `rsi_normalized` (-1 à +1) |
| ATR | 14 | `atr_pct` (% du prix) |
| Bollinger Bands | 20, 2σ | `bb_position` (-1 à +1), `bb_bandwidth` |
| MACD | 12/26/9 | `macd_hist_normalized` (histogram / close) |
| ADX | 14 | `adx_normalized` (0 à 1) |
| RSI 4h | 14 | `rsi_4h_normalized` (-1 à +1) |
| SMA trend 4h | 50/200 | `sma_trend_4h` (+1/-1) |
| Funding Rate | — | `funding_rate` |
| Bougies japonaises | — | `candle_body` (-1 à +1), `upper_wick_ratio` (0-1), `lower_wick_ratio` (0-1) |
| High/Low 20p | 20 | `price_to_high_20` (≤ 0), `price_to_low_20` (≥ 0) |

---

## Architecture réseau

```
Observations stacked (frame_stack=24 × n_features)
  → Reshape (batch, n_features, 24)
  → Conv1d(n_features→32, kernel=3) → ReLU
  → Conv1d(32→64, kernel=3) → ReLU
  → AdaptiveAvgPool1d(1) → Flatten
  → Linear(64→128) → ReLU
  → PPO policy head (actor + critic)
```

Le CNN 1D détecte des patterns temporels dans les observations empilées (meilleur que MLP flat).

---

## Reward (recompense RL)

La recompense combine :

- **Log-return** : variation du net worth (`log(NW_t / NW_{t-1})`)
- **Sharpe ratio** : rendement ajuste au risque, annualisé (`× sqrt(8760)` pour returns horaires)
- **Sortino ratio** : comme Sharpe mais penalise seulement la volatilite negative
- **Penalite drawdown** : exponentielle si drawdown > 15%, cappée à -2.0 (stabilite)
- **Penalite position sizing** : empeche les positions trop grandes
- **Penalite couts** : frais 0.1% + slippage [0.01%, 0.1%]

Les rewards sont normalisés par **VecNormalize** (`norm_obs=False, norm_reward=True`).

---

## Circuit Breaker

Surveillance continue du marche (toutes les minutes) :
- **Chute de prix** : coupe les positions si chute > 3% en 5 minutes
- **Volume anormal** : coupe si volume > 5x la moyenne

---

## Configuration

Tous les parametres sont dans `config/settings.py` :

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| EXCHANGE | binance | Exchange ccxt |
| SYMBOL | BTC/USDT | Paire de trading |
| TIMEFRAME | 1h | Timeframe principal |
| TIMEFRAME_SECONDARY | 4h | Timeframe contexte macro |
| INITIAL_BALANCE | 10,000 USDT | Capital initial |
| TRADING_FEE | 0.1% | Frais par trade (fee sur valeur nominale) |
| FRAME_STACK_SIZE | 24 | Bougies en memoire |
| TOTAL_TIMESTEPS | 1,000,000 | Steps d'entrainement |
| N_ENVS | 4 | Envs paralleles |
| USE_CNN | True | CNN 1D feature extractor |
| CNN_FEATURES_DIM | 128 | Dimension sortie CNN |
| DRAWDOWN_PENALTY_CAP | 2.0 | Cap max penalite drawdown |
| EARLY_STOPPING_PATIENCE | 5 | Checks avant early stop |
| WF_TRAIN_MONTHS | 24 | Mois de train par fold |
| WF_TEST_MONTHS | 3 | Mois de test par fold |
| WF_STEP_MONTHS | 3 | Pas d'avancement par fold |

Cles API (variables d'environnement) :
```
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...
```

---

## Tests

```bash
# Tous les tests
python -m pytest tests/ -v

# Par module
python -m pytest tests/test_data.py -v             # Data pipeline
python -m pytest tests/test_features.py -v         # Indicateurs + NLP + multi-TF
python -m pytest tests/test_env.py -v              # Environnement RL + rewards
python -m pytest tests/test_agent.py -v            # PPO + CNN + VecNormalize
python -m pytest tests/test_training.py -v         # Training + backtest + logger
python -m pytest tests/test_walk_forward.py -v     # Walk-forward validation
python -m pytest tests/test_live.py -v             # Live executor + circuit breaker
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
| Feature extractor | CNN 1D (PyTorch) |
| Normalisation | scikit-learn (RobustScaler) + VecNormalize |
| Dashboard | Streamlit |
| Logs | TensorBoard + JSON/CSV |
| Tests | pytest |

---

## Licence

Projet personnel — usage libre.
