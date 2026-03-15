# Phase 8 -- Checklist de validation

## README.md

- [X] Architecture du projet documentee
- [X] Instructions d'installation completes
- [X] Guide d'utilisation (train, backtest, live, dashboard)
- [X] Pipeline de donnees explique
- [X] Indicateurs techniques listes
- [X] Configuration centralisee documentee
- [X] Stack technique resumee

## Revue de code + nettoyage

### Bugs corriges (passe 1)
- [X] `pipeline.py` : scaler charge depuis le disque quand `fit_scaler=False`
- [X] `backtest.py` : utilise `fit_scaler=False` (charge le scaler d'entrainement)
- [X] `executor.py` : `PaperPortfolio.net_worth` corrige (methode avec current_price)

### Bugs corriges (passe 2 -- revue approfondie)
- [X] `executor.py` : `_get_current_price` retournait le prix normalise au lieu du prix reel -- corrige via `raw_close`
- [X] `executor.py` : `get_stats(0)` dans le `finally` valorisait la position a 0 -- utilise `last_price`
- [X] `executor.py` : `entry_price` ecrase au lieu d'etre moyenne sur achats successifs -- corrige comme `TradingEnv`
- [X] `crypto_fetcher.py` : pas de try-except dans la boucle `fetch_ohlcv` -- erreurs reseau crashaient le pipeline
- [X] `test_data.py` : test `test_fetch_ohlcv_empty_on_invalid_symbol` adapte au nouveau comportement gracieux

### Imports nettoyes (passe 1)
- [X] `training/train.py` : suppression `sys` et `EvalCallback` inutilises
- [X] `live/executor.py` : suppression `append_monthly_csv` inutilise

### Imports nettoyes (passe 2)
- [X] `agent/model.py` : suppression `VecNormalize` inutilise
- [X] `training/train.py` : suppression `make_env` inutilise (seul `make_vec_env` est utilise)

### Ajouts pipeline
- [X] `pipeline.py` : ajout colonne `raw_close` preservee a travers la normalisation
- [X] `scaler.py` : `raw_close` ajoute a `EXCLUDE_COLUMNS`

### Bugs corriges (passe 3 -- revue approfondie)
- [X] CRITICAL `backtest.py` : auto-reset VecEnv effacait les stats terminales -- extraction depuis `infos[0]["portfolio_stats"]`
- [X] CRITICAL `trading_env.py` : `raw_close` inclus dans les features (data leakage) -- ajoute au set `exclude`
- [X] CRITICAL `trading_env.py` : ajout `portfolio_stats` dans le dict `info` quand `terminated=True`
- [X] `backtest.py` : suppression import inutilise `MODELS_DIR`

### Gestion d'erreurs ajoutee (passe 3)
- [X] `agent/model.py` : `load_agent` verifie l'existence du fichier avec message clair
- [X] `data/pipeline.py` : `build_full_pipeline` try/except sur le chargement scaler avec FileNotFoundError
- [X] `training/backtest.py` : erreur si dataset vide apres pipeline + try/except model loading
- [X] `training/train.py` : erreur si dataset vide apres pipeline
- [X] `live/executor.py` : try/except dans `_fetch_recent_data` et chargement modele

## Tests d'integration (`tests/test_integration.py`)

### Tests automatiques (pytest)
```
python -m pytest tests/test_integration.py -v
```
**16 tests a verifier :**

- [X] Pipeline complet train + predict (donnees -> features -> scaler -> env -> train -> save -> load -> predict)
- [X] Coherence scaler save/load (memes valeurs apres rechargement)
- [X] Backtest produit des stats valides via terminal info (net_worth, return, trades)
- [X] Backtest terminal stats : portfolio_stats present et complet (7 cles requises)
- [X] Cycle complet de logs (weekly JSON, CSV, monthly CSV, backtest JSON, lecture)
- [X] Pipeline features + normalisation (indicateurs presents, valeurs entre -1 et +1)
- [X] Environnement accepte les donnees scalees
- [X] Paper trading multi-trades coherent
- [X] Paper trading : prix d'entree moyenne sur achats successifs
- [X] Paper trading : `last_price` mis a jour a chaque ordre
- [X] `raw_close` preserve a travers la normalisation
- [X] `raw_close` exclue des features de l'environnement
- [X] Circuit breaker detection fonctionne
- [X] Error handling : `load_agent` fichier manquant → FileNotFoundError
- [X] Error handling : scaler manquant → FileNotFoundError
- [X] Error handling : `transform` avant `fit` → RuntimeError

## Tests manuels (REPL Python)

### Test 1 -- Scaler save/load
```python
import numpy as np
import pandas as pd
from features.scaler import FeatureScaler, normalize_features
np.random.seed(42)
n = 100
df = pd.DataFrame({"close": np.random.randn(n) * 1000 + 30000, "volume": np.random.uniform(100, 1000, n), "rsi": np.random.uniform(20, 80, n)})
df_scaled, scaler = normalize_features(df, fit=True)
print(f"Fitted: {scaler.is_fitted}, Columns: {scaler.feature_columns}")
from pathlib import Path
import tempfile
tmp = Path(tempfile.mkdtemp()) / "scaler.pkl"
scaler.save(tmp)
scaler2 = FeatureScaler()
scaler2.load(tmp)
df_scaled2, _ = normalize_features(df.copy(), scaler=scaler2, fit=False)
diff = (df_scaled["close"] - df_scaled2["close"]).abs().max()
print(f"Max difference apres reload: {diff}")
print(f"Identique: {diff < 1e-10}")
```
- [X] Scaler fitted avec les bonnes colonnes
- [X] Difference < 1e-10 apres rechargement
- [X] Identique = True

### Test 2 -- Pipeline complet build_full_pipeline (mode train)
```python
from data.pipeline import build_full_pipeline
df, scaler = build_full_pipeline(start="2025-01-01", end="2025-01-10", include_nlp=False, fit_scaler=True)
print(f"Shape: {df.shape}")
print(f"Colonnes: {list(df.columns[:10])}...")
print(f"raw_close present: {'raw_close' in df.columns}")
print(f"raw_close echantillon: {df['raw_close'].iloc[-1]:.2f}")
print(f"close normalise echantillon: {df['close'].iloc[-1]:.4f}")
print(f"Prix reel vs normalise different: {abs(df['raw_close'].iloc[-1] - df['close'].iloc[-1]) > 100}")
print(f"Scaler fitted: {scaler.is_fitted}")
print(f"Nb colonnes scalees: {len(scaler.feature_columns)}")
```
- [X] Shape > (0, 0) -- donnees recuperees
- [X] `raw_close` present dans les colonnes
- [X] `raw_close` contient le vrai prix BTC (ex: ~60000-100000)
- [X] `close` normalise est tres different de `raw_close` (entre -1 et 1)
- [X] Scaler fitted = True

## Regression totale
```
python -m pytest tests/ -v
```
- [X] 130/130 tests passent (dont 16 integration, 8 nouveaux dans la passe 3)
