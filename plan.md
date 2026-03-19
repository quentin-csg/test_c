# Plan d'implementation — 4 corrections majeures

## 1. Reconstruire `env/trading_env.py` (fichier vide)

Creer la classe `TradingEnv(gymnasium.Env)` en se basant sur :
- Les tests existants dans `test_env.py` et `test_integration.py` (qui definissent exactement l'API attendue)
- L'`instruction.md` (specs du projet)
- Les imports et usages dans `agent/model.py`, `training/backtest.py`, `live/executor.py`

**Contrat deduit des tests :**
- `TradingEnv(df, feature_columns=None, initial_balance=10000, trading_fee=0.001, slippage_min=0, slippage_max=0.0005, render_mode=None)`
- `observation_space` : Box(-inf, inf, shape=(n_obs,)) ou n_obs = len(feature_columns) + 3 (balance_norm, position_norm, net_worth_norm)
- `action_space` : Box(-1, 1, shape=(1,))
- `reset(seed=None)` -> (obs, info) avec info contenant balance, position
- `step(action)` -> (obs, reward, terminated, truncated, info)
  - Zone morte : |action| <= 0.05 = hold
  - Achat : action > 0.05, vente : action < -0.05
  - Frais deduits, slippage aleatoire applique
  - Terminated si : fin des donnees OU net_worth < 80% du capital initial (drawdown > 20%)
  - Info contient : net_worth, balance, position, total_trades, total_fees_paid
  - A la terminaison, info contient `portfolio_stats` (pour le VecEnv auto-reset)
- `get_portfolio_stats()` -> dict avec net_worth, total_return_pct, max_drawdown_pct, sharpe_ratio, sortino_ratio, total_trades, total_fees
- Proprietes : `prices`, `current_step`, `position`, `balance`, `entry_price`, `total_trades`, `total_fees_paid`, `n_obs`, `feature_columns`, `np_random`
- `render_mode="human"` affiche dans la console
- Exclut `raw_close` et `timestamp` des features

**Reward** : utilise `agent/reward.py` -> `compute_reward()` avec les composantes existantes.

**Fichiers impactes** : `env/trading_env.py` (creation)

---

## 2. Ajouter `VecNormalize` pour normaliser obs et rewards

Integrer `VecNormalize` de SB3 dans le pipeline de vectorisation. `VecNormalize` normalise dynamiquement les observations et rewards pendant le training, ce qui stabilise l'apprentissage PPO.

**Modifications :**

### `agent/model.py`
- Importer `VecNormalize`
- Dans `make_vec_env()` : ajouter un parametre `normalize=True`
- La chaine devient : DummyVecEnv/SubprocVecEnv -> VecNormalize -> VecFrameStack
- Ajouter fonctions `save_vec_normalize()` et `load_vec_normalize()` pour persister les stats de normalisation (fichier `.pkl` dans `models/`)

### `training/train.py`
- Apres creation du vec_env, sauvegarder les stats VecNormalize avec le modele
- Sauvegarder `vec_normalize.pkl` a cote du modele

### `training/backtest.py`
- Charger le VecNormalize sauvegarde (mode eval : `training=False`, `norm_reward=False`)

### `live/executor.py`
- Charger le VecNormalize sauvegarde en mode eval

### `config/settings.py`
- Ajouter `VECNORM_PATH` pour le chemin du fichier de normalisation

---

## 3. Corriger `executor.py` — Ne plus recreer env/modele a chaque tick

**Probleme actuel :** A chaque tick horaire, le code :
1. Refait `build_full_pipeline()` (30 jours de donnees)
2. Cree un nouveau `VecFrameStack`
3. Recharge le modele PPO depuis le disque
4. Joue des centaines d'actions `hold` pour arriver au dernier step

**Solution :**
- Charger le modele et le VecNormalize **une seule fois** dans `__init__()` ou au premier tick
- A chaque tick, fetcher uniquement les donnees recentes, construire l'observation directement (sans creer un env complet)
- Utiliser directement le scaler + les dernieres `frame_stack` observations pour construire le vecteur d'input
- Appeler `agent.predict(obs)` directement sans passer par un env Gymnasium

**Modifications dans `live/executor.py` :**
- `__init__()` : charger le modele PPO, le scaler, et le VecNormalize une seule fois
- `_build_observation()` : nouvelle methode qui construit l'observation a partir des donnees recentes
  - Fetch frame_stack + marge bougies recentes
  - Calcule les features
  - Normalise avec le scaler sauvegarde
  - Empile les frame_stack dernieres observations
  - Applique VecNormalize.normalize_obs()
- `tick()` : appeler `_build_observation()` puis `agent.predict(obs)`
- Supprimer la boucle de hold fictive

---

## 4. Walk-Forward Validation au lieu du split fixe

**Probleme actuel :** Split fixe train=2020-2023, test=2024+. Un seul run ne valide rien statistiquement.

**Solution : Walk-forward avec fenetres glissantes**

Creer `training/walk_forward.py` :
- Diviser les donnees en fenetres glissantes (ex: 12 mois train, 3 mois test)
- Pour chaque fold :
  1. Entrainer un modele sur la fenetre train
  2. Backtester sur la fenetre test
  3. Sauvegarder les metriques
- Agreger les resultats : mean/std du Sharpe, return, drawdown sur tous les folds
- Logger chaque fold dans `logs/train/walk_forward/`

**Schema des folds (par defaut) :**
```
Fold 1: Train 2020-01 -> 2021-12  |  Test 2022-01 -> 2022-03
Fold 2: Train 2020-01 -> 2022-03  |  Test 2022-04 -> 2022-06
Fold 3: Train 2020-01 -> 2022-06  |  Test 2022-07 -> 2022-09
...
Fold N: Train 2020-01 -> 2024-09  |  Test 2024-10 -> 2024-12
```
(Expanding window : le train grandit, le test avance)

**Modifications :**

### `training/walk_forward.py` (nouveau fichier)
- `generate_folds()` : genere les fenetres train/test
- `walk_forward_validate()` : boucle train+backtest sur chaque fold
- `aggregate_results()` : calcule mean/std des metriques

### `config/settings.py`
- `WF_TRAIN_MONTHS = 24` (fenetre train initiale)
- `WF_TEST_MONTHS = 3` (fenetre test)
- `WF_STEP_MONTHS = 3` (pas d'avancement)

### `main.py`
- Ajouter la commande `walk-forward`

### `training/logger.py`
- Ajouter `log_walk_forward_result()` et `load_walk_forward_results()`
- Dossier `logs/train/walk_forward/`

### `live/dashboard.py`
- Ajouter un onglet "Walk-Forward" dans le dashboard avec les resultats agreges

---

## 5. Multi-timeframe :

Le 1h est un bon choix pour BTC. Pas trop de bruit (comme le 1m/5m), assez reactif.
Le 4h (TIMEFRAME_SECONDARY) est defini dans les settings mais jamais utilise dans le pipeline. Si tu veux du multi-timeframe, il faut l'integrer.
Suggestion : le multi-timeframe (1h + 4h + 1d) via des features derivees (RSI en 4h, SMA en daily) apporterait de la robustesse.

---

## 6. Features

LISTE COMPLETE DES FEATURES
Features V1 (OHLCV + tech de base) — 10 features
#	Feature	Source	Description	Verdict
1	close	OHLCV	Prix de cloture de la bougie	DOUBLON — Deja encode dans log_return, price_to_sma_long, bb_position, zscore. Le prix brut apres normalisation RobustScaler perd son sens (le modele ne peut pas comparer 20k vs 60k)
2	open	OHLCV	Prix d'ouverture	DOUBLON — Meme probleme. La difference open-close est deja capturee par log_return
3	high	OHLCV	Plus haut de la bougie	DOUBLON — Deja encode dans atr. Apres normalisation, la valeur absolue n'a pas de sens
4	low	OHLCV	Plus bas de la bougie	DOUBLON — Idem que high
5	volume	OHLCV	Volume brut en BTC/USDT	DOUBLON — Deja encode dans volume_ratio et volume_direction. Le volume brut varie enormement d'une annee a l'autre, le scaler ne corrige pas ca
6	rsi	pandas-ta	RSI sur 14 periodes (0-100)	DOUBLON avec rsi_normalized — les deux portent exactement la meme info, juste une transformation lineaire
7	rsi_normalized	pandas-ta	RSI normalise (-1 a +1)	GARDER — Version propre du RSI, deja dans le bon range
8	sma_50	pandas-ta	Moyenne mobile 50 periodes	DOUBLON — Le prix absolu de la SMA est inutile apres normalisation. L'info utile est sma_trend et price_to_sma_long
9	sma_200	pandas-ta	Moyenne mobile 200 periodes	DOUBLON — Meme raison
10	sma_trend	pandas-ta	SMA50 > SMA200 ? (+1 bullish / -1 bearish)	GARDER — Signal binaire de tendance, tres pertinent
Features V2 (+ macro, sentiment, tech avancee) — 16 features ajoutees (26 total)
#	Feature	Source	Description	Verdict
11	qqq_close	yfinance	Prix de cloture de l'ETF QQQ (Nasdaq 100)	A TRANSFORMER — Le prix brut est inutile. Il faudrait le log_return du QQQ, pas le prix
12	spy_close	yfinance	Prix de cloture de l'ETF SPY (S&P 500)	A TRANSFORMER — Idem. Et QQQ/SPY sont tres correles entre eux (~0.95), un seul suffit
13	fear_greed_normalized	Alternative.me	Fear & Greed Index normalise (-1/+1)	GARDER — Bon proxy du sentiment global, donnees historiques disponibles
14	funding_rate	ccxt	Taux de financement des contrats perpetuels	GARDER — Excellent indicateur de positionnement du marche. Funding positif = trop de longs, negatif = trop de shorts
15	atr	pandas-ta	Average True Range brut (14p)	DOUBLON avec atr_pct — L'ATR brut depend de l'echelle du prix, inutile apres normalisation
16	atr_pct	pandas-ta	ATR en % du prix	GARDER — Mesure de volatilite normalisee, independante de l'echelle du prix
17	bb_position	pandas-ta	Position du prix dans les bandes de Bollinger (-1 en bas, +1 en haut)	GARDER — Signal de surachat/survente, complementaire au RSI
18	bb_bandwidth	pandas-ta	Largeur des bandes de Bollinger	GARDER — Mesure de volatilite. Bandes serrees = explosion a venir
19	zscore	calcul	Nombre d'ecarts-types vs moyenne 20p	DOUBLON avec bb_position — Le Z-score et bb_position mesurent exactement la meme chose (position relative du prix par rapport a sa distribution recente). Le Z-score utilise 20 periodes, comme Bollinger
20	volume_ratio	calcul	Volume actuel / moyenne mobile 20p	GARDER — Volume relatif, detecte les anomalies
21	volume_direction	calcul	sign(delta_price) * volume_ratio	GARDER — Combine direction et intensite du volume
22	log_return	calcul	ln(close/close[-1]) — return 1h	GARDER — Feature fondamentale, return a court terme
23	log_return_5h	calcul	ln(close/close[-5]) — return 5h	GARDER — Momentum moyen terme
24	log_return_24h	calcul	ln(close/close[-24]) — return 24h	GARDER — Momentum long terme
25	is_weekend	calcul	1 si samedi/dimanche	GARDER — Le comportement du marche crypto change le weekend (moins de volume, absence des institutionnels)
Features V3 (+ NLP) — 3 features ajoutees (29 total)
#	Feature	Source	Description	Verdict
26	sentiment_score	FinBERT	Score NLP moyen des news (-1/+1)	PROBLEMATIQUE — Pas de donnees historiques (RSS = derniers articles). Sera toujours 0 pendant le training. Inutile sauf en live
27	n_articles	RSS	Nombre d'articles par heure	PROBLEMATIQUE — Meme raison. Et le nombre d'articles n'est pas un bon predicteur de direction
28	price_to_sma_long	calcul	(close - SMA200) / SMA200	GARDER — Distance relative au trend long terme, tres informatif
Features dans le pipeline mais PAS dans les feature sets
Feature	Source	Description	Verdict
bb_lower	pandas-ta	Bande de Bollinger basse	INUTILE — Prix brut, deja capture par bb_position
bb_middle	pandas-ta	Bande de Bollinger mediane (=SMA20)	INUTILE — Redondant avec SMA et bb_position
bb_upper	pandas-ta	Bande de Bollinger haute	INUTILE — Redondant avec bb_position
bb_percent	pandas-ta	%B de Bollinger	DOUBLON DIRECT avec bb_position
fear_greed_value	Alternative.me	Score brut 0-100	DOUBLON avec fear_greed_normalized
raw_close	pipeline	Prix brut sauvegarde avant normalisation	NE PAS DONNER AU MODELE — utilise uniquement pour le live
RESUME : CE QU'IL FAUT GARDER
Features recommandees (14 features) :

#	Feature	Pourquoi
1	rsi_normalized	Momentum / surachat-survente
2	sma_trend	Direction de tendance long terme
3	price_to_sma_long	Distance au trend, detecte les ecarts
4	atr_pct	Volatilite normalisee
5	bb_position	Position dans les bandes, reversal signal
6	bb_bandwidth	Compression de volatilite (squeeze)
7	volume_ratio	Anomalies de volume
8	volume_direction	Volume + direction du prix
9	log_return	Return 1h (momentum court terme)
10	log_return_5h	Return 5h (momentum moyen terme)
11	log_return_24h	Return 24h (momentum long terme)
12	fear_greed_normalized	Sentiment global du marche
13	funding_rate	Positionnement long/short du marche
14	is_weekend	Regime de marche
Avec un frame stack de 8 : 14 x 8 = 112 dimensions — parfait pour un MLP 256x256.

FEATURES MANQUANTES IMPORTANTES
Feature	Description	Pourquoi c'est important
MACD (12/26/9)	Moving Average Convergence Divergence	Momentum + signal de croisement. Complementaire au RSI : le RSI mesure la force, le MACD mesure le changement de momentum
MACD histogram	MACD - Signal line	La divergence entre MACD et signal est un des meilleurs predicteurs de retournement
ADX (14p)	Average Directional Index (0-100)	Mesure la force de la tendance (pas la direction). Un RSI a 70 + ADX > 25 = forte tendance haussiere. RSI a 70 + ADX < 20 = probable reversal
OBV normalized	On-Balance Volume (derive)	Flux de volume cumule directionnel. Si le prix monte mais l'OBV baisse = divergence = signal de reversal imminent. Ton volume_direction est une version simplifiee mais rate les divergences
Orderbook imbalance	(bid_volume - ask_volume) / total	Tu as fetch_order_book() dans ton code mais tu ne l'utilises jamais dans le pipeline. C'est un signal tres puissant en crypto pour predire les mouvements a court terme
Open Interest change	Variation de l'OI (futures)	Tu as fetch_open_interest() mais pas utilise non plus. L'OI qui monte avec le prix = tendance forte. L'OI qui monte et le prix qui baisse = liquidations a venir
QQQ log_return	Return du QQQ au lieu du prix brut	Au lieu de qqq_close (prix brut inutile), utiliser le return. Et un seul suffit (QQQ ou SPY, pas les deux)
Stochastic RSI	RSI du RSI (0-1)	Plus reactif que le RSI classique pour les retournements rapides en crypto

## 7. Mise a jour tests, docs, dashboard

### Tests a modifier :
- `tests/test_env.py` : les tests TradingEnv passeront maintenant (env reconstruit)
- `tests/test_agent.py` : adapter si l'obs shape change avec VecNormalize
- `tests/test_training.py` : ajouter tests pour VecNormalize save/load
- `tests/test_integration.py` : ajouter tests walk-forward
- `tests/test_live.py` : adapter les tests LiveExecutor pour la nouvelle architecture (modele charge une seule fois)
- Nouveau : `tests/test_walk_forward.py`

### Docs :
- `README.md` : ajouter walk-forward dans usage et description
- `config/settings.py` : documenter les nouveaux parametres

### Dashboard :
- `live/dashboard.py` : ajouter onglet walk-forward

---

## Ordre d'implementation

1. **`env/trading_env.py`** — tout depend de ca
2. **`VecNormalize`** dans `agent/model.py` + `train.py` + `backtest.py`
3. **`executor.py`** — fix du tick (depend de VecNormalize)
4. **`walk_forward.py`** — nouveau module
5. **Tests + docs + dashboard** — en continu avec chaque etape


utiliser cnn 1d

1. Le changement prioritaire : L'Architecture du Réseau
Actuellement, ton MLP reçoit les 24 bougies "aplaties" (flattened) en un seul long vecteur. Le réseau ne "comprend" pas nativement que la bougie t-1 vient avant la bougie t. Il doit réapprendre cette relation temporelle à partir de zéro, ce qui est très inefficace.

Amélioration A (La plus recommandée) : Passer à un CNN 1D

Pourquoi ? Les réseaux convolutifs 1D excellent pour détecter des "motifs" (patterns locaux comme des figures chartistes) dans une série temporelle fixe (tes 24 bougies).

Comment ? Dans SB3, tu peux utiliser une CnnPolicy mais tu devras coder un BaseFeaturesExtractor personnalisé en PyTorch qui applique des couches Conv1d sur ta fenêtre temporelle avant de passer au MLP.

Amélioration B : Passer au LSTM / GRU

Pourquoi ? Le LSTM possède une mémoire à court et long terme native. Il peut repérer des tendances sans avoir besoin d'aplatir les données.

Comment ? Passer sur sb3-contrib et utiliser RecurrentPPO avec MlpLstmPolicy.

Attention : L'entraînement sera beaucoup plus lent et parfois plus instable qu'un CNN 1D, mais cela retire le besoin du VecFrameStack.

2. Dimensionnement du réseau (Si tu gardes le MLP)
Par défaut, la MlpPolicy de SB3 utilise 2 couches cachées de 64 neurones ([64, 64]).
Si chaque bougie possède par exemple 10 features (OHLCV + 5 indicateurs techniques), ton VecFrameStack de 24 produit un vecteur d'entrée de 240 valeurs.
Faire passer 240 valeurs dans 64 neurones crée une perte d'information immédiate (bottleneck).