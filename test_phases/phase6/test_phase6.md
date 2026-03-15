# Phase 6 — Checklist de validation

## training/logger.py

### Tests automatiques (pytest)
```
python -m pytest tests/test_training.py::TestLogger -v
```
**6 tests a verifier :**

- [X] log_weekly_summary sauvegarde un JSON avec metriques + timestamp
- [X] log_weekly_summary genere automatiquement le label de semaine
- [X] log_backtest_result sauvegarde les resultats avec model_name et run_name
- [X] load_backtest_results charge et filtre par modele
- [X] _serialize convertit correctement np.int64, np.float64, np.ndarray
- [X] print_stats affiche les metriques de maniere formatee

## training/train.py

### Tests automatiques (pytest)
```
python -m pytest tests/test_training.py::TestTraining -v
```
**2 tests a verifier :**

- [X] train() execute un entrainement complet (micro: 128 steps)
- [X] save/load d'un agent PPO fonctionne correctement

## training/backtest.py

### Tests automatiques (pytest)
```
python -m pytest tests/test_training.py::TestBacktest -v
```
**2 tests a verifier :**

- [X] backtest() charge un modele et l'evalue sur des donnees
- [X] backtest() est deterministe (memes donnees = memes resultats)

## Integration

### Tests automatiques (pytest)
```
python -m pytest tests/test_training.py::TestIntegration -v
```
**2 tests a verifier :**

- [X] Pipeline complet: train -> save -> backtest sur memes donnees
- [X] Les stats numpy sont correctement serialisees en JSON

## Tests manuels (REPL Python)

Lancer `python` puis copier-coller chaque bloc.

### Test 1 — Logger : sauvegarder et relire un log hebdo
```python
import json, tempfile, os
from pathlib import Path
from training.logger import log_weekly_summary, WEEKLY_LOG_DIR

WEEKLY_LOG_DIR.mkdir(parents=True, exist_ok=True)
stats = {"pnl": 256.78, "sharpe_ratio": 1.35, "total_trades": 18}
filepath = log_weekly_summary(stats, week_label="2024_W10")
print(f"Fichier cree: {filepath}")
print(f"Existe: {filepath.exists()}")

with open(filepath, "r") as f:
    data = json.load(f)
    print(f"PnL: {data['pnl']}")
    print(f"Sharpe: {data['sharpe_ratio']}")
    print(f"Week: {data['week']}")
```
- [X] Fichier cree dans logs/weekly/
- [X] PnL = 256.78, Sharpe = 1.35, Week = 2024_W10

### Test 2 — Logger : sauvegarder et relire un backtest
```python
from training.logger import log_backtest_result, load_backtest_results, BACKTEST_LOG_DIR
import json

BACKTEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
stats1 = {"total_return_pct": 5.2, "sharpe_ratio": 1.1}
stats2 = {"total_return_pct": -2.0, "sharpe_ratio": -0.3}
log_backtest_result(stats1, model_name="modelA", run_name="run01")
log_backtest_result(stats2, model_name="modelB", run_name="run02")

results = load_backtest_results()
print(f"Nombre de resultats: {len(results)}")
for r in results:
    print(f"  {r['model_name']}: return={r['total_return_pct']}%")

filtered = load_backtest_results(model_name="modelA")
print(f"Filtre modelA: {len(filtered)} resultat(s)")
```
- [X] 2 resultats charges
- [X] Filtrage par modele fonctionne

### Test 3 — Micro-entrainement complet avec train()
```python
import numpy as np, pandas as pd
from unittest.mock import patch, MagicMock
from training.train import train

np.random.seed(42)
n = 200
prices = 30000 + np.cumsum(np.random.randn(n) * 50)
df = pd.DataFrame({
    'open': prices, 'high': prices + 30, 'low': prices - 30,
    'close': prices, 'volume': np.random.uniform(100, 1000, n),
    'rsi_14': np.random.uniform(20, 80, n),
    'sma_50': prices, 'atr_14': np.random.uniform(50, 200, n),
    'sentiment_score': np.random.uniform(-0.5, 0.5, n),
    'fear_greed_value': np.random.uniform(20, 80, n),
})

mock_scaler = MagicMock()
with patch("training.train.build_full_pipeline", return_value=(df, mock_scaler)):
    model_path = train(
        total_timesteps=256,
        n_envs=1,
        frame_stack=4,
        model_name="test_manual",
        use_subproc=False,
        seed=42,
    )

print(f"Modele sauvegarde: {model_path}")
print(f"Fichier existe: {model_path.with_suffix('.zip').exists()}")
```
- [X] Entrainement sans crash (barre de progression visible)
- [X] Modele .zip sauvegarde dans models/

### Test 4 — Backtest apres entrainement
```python
import numpy as np, pandas as pd
from unittest.mock import patch, MagicMock
from agent.model import create_agent, make_vec_env, save_agent
from training.backtest import backtest

np.random.seed(42)
n = 200
prices = 30000 + np.cumsum(np.random.randn(n) * 50)
df = pd.DataFrame({
    'open': prices, 'high': prices + 30, 'low': prices - 30,
    'close': prices, 'volume': np.random.uniform(100, 1000, n),
    'rsi_14': np.random.uniform(20, 80, n),
    'sma_50': prices, 'atr_14': np.random.uniform(50, 200, n),
    'sentiment_score': np.random.uniform(-0.5, 0.5, n),
    'fear_greed_value': np.random.uniform(20, 80, n),
})

vec_env = make_vec_env(df, n_envs=1, use_subproc=False, frame_stack=4)
agent = create_agent(vec_env, tensorboard_log=None, seed=42)
agent.learn(total_timesteps=128)
save_agent(agent, name="backtest_manual")
vec_env.close()

mock_scaler = MagicMock()
with patch("training.backtest.build_full_pipeline", return_value=(df.copy(), mock_scaler)):
    stats = backtest(model_name="backtest_manual", frame_stack=4, save_results=False)

print(f"Return: {stats['total_return_pct']:.2f}%")
print(f"Net worth: {stats['net_worth']:.2f}")
print(f"Sharpe: {stats['sharpe_ratio']:.4f}")
print(f"Sortino: {stats['sortino_ratio']:.4f}")
print(f"Max drawdown: {stats['max_drawdown_pct']:.2f}%")
print(f"Trades: {stats['total_trades']}")
print(f"Steps: {stats['total_steps']}")
```
- [X] Tableau de stats affiche (via print_stats dans backtest)
- [X] Toutes les metriques presentes (return, net_worth, sharpe, sortino, drawdown, trades)
- [X] total_steps > 0

### Test 5 — print_stats affichage formate
```python
from training.logger import print_stats

stats = {
    "total_return_pct": 3.75,
    "net_worth": 10375.0,
    "sharpe_ratio": 0.92,
    "sortino_ratio": 1.15,
    "max_drawdown_pct": 4.2,
    "total_trades": 27,
}
print_stats(stats, title="Backtest V1 - 2024")
```
- [X] Affichage avec bordures (====)
- [X] Titre visible
- [X] Valeurs alignees

### Test 6 — Serialisation numpy dans les logs
```python
import numpy as np
from training.logger import _serialize

print(_serialize(np.int64(42)), type(_serialize(np.int64(42))))
print(_serialize(np.float64(3.14)), type(_serialize(np.float64(3.14))))
print(_serialize(np.array([1, 2, 3])), type(_serialize(np.array([1, 2, 3]))))
print(_serialize("texte"), type(_serialize("texte")))
```
- [X] np.int64 -> int
- [X] np.float64 -> float
- [X] np.array -> list
- [X] str reste str

## Tous les tests (regression)
```
python -m pytest tests/ -v
```
- [X] 90/90 tests passent (16 data + 22 features + 30 env + 10 agent + 12 training)
