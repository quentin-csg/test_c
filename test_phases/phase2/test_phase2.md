# =============================================================================
# Phase 2 — Checklist de validation
# =============================================================================
#
# Toutes les commandes sont à copier-coller dans le REPL Python :
#   1. Ouvrir un terminal PowerShell dans le dossier Trading-bot
#   2. Lancer `python`
#   3. Copier-coller chaque bloc ci-dessous
#
# Seul le test pytest (dernier bloc) se lance depuis PowerShell directement.
#

## Imports

- [X] **Tous les modules data s'importent sans erreur** :
  ```python
  from data.crypto_fetcher import fetch_ohlcv, fetch_multi_timeframe, fetch_funding_rate, fetch_order_book, fetch_open_interest
  from data.macro_fetcher import fetch_macro_data, fetch_macro_daily
  from data.sentiment_fetcher import fetch_fear_greed_current, fetch_fear_greed_history
  from data.news_fetcher import fetch_news, fetch_news_titles_for_hour
  from data.pipeline import build_dataset, get_news_for_dataset
  print('OK: Tous les modules importés')
  ```

## Crypto Fetcher (ccxt)

- [X] **Récupération OHLCV BTC/USDT 1h (1er au 2 janv inclus = ~48 bougies)** :
  ```python
  from data.crypto_fetcher import fetch_ohlcv
  df = fetch_ohlcv('BTC/USDT', '1h', since='2024-01-01', until='2024-01-02')
  print(f'{len(df)} bougies')
  print(df.head())
  ```
  → ~48 bougies (1er + 2 janv), colonnes: timestamp, open, high, low, close, volume

- [X] **Récupération multi-timeframe (1h + 4h)** :
  ```python
  from data.crypto_fetcher import fetch_multi_timeframe
  data = fetch_multi_timeframe(since='2024-01-01', until='2024-01-03')
  print(f"1h: {len(data['1h'])} bougies")
  print(f"4h: {len(data['4h'])} bougies")
  ```
  → ~72 bougies 1h, ~18 bougies 4h

- [X] **Order Book avec imbalance** :
  ```python
  from data.crypto_fetcher import fetch_order_book
  ob = fetch_order_book('BTC/USDT')
  print(f"Bid Vol: {ob['bid_volume']:.4f}")
  print(f"Ask Vol: {ob['ask_volume']:.4f}")
  print(f"Imbalance: {ob['imbalance']:.4f}")
  print(f"Mid Price: {ob['mid_price']:.2f}")
  ```
  → Imbalance entre -1 et +1

- [X] **Funding Rate** :
  ```python
  from data.crypto_fetcher import fetch_funding_rate
  df = fetch_funding_rate('BTC/USDT', since='2024-01-01')
  print(f'{len(df)} entrées funding rate')
  print(df.head())
  ```

- [X] **Open Interest** :
  ```python
  from data.crypto_fetcher import fetch_open_interest
  oi = fetch_open_interest('BTC/USDT')
  print(f'Open Interest: {oi}')
  ```

## Macro Fetcher (yfinance)

- [X] **Données macro QQQ + SPY** :
  ```python
  from data.macro_fetcher import fetch_macro_daily
  df = fetch_macro_daily(['QQQ', 'SPY'], start='2024-01-01', end='2024-01-31')
  print(f'{len(df)} lignes')
  print(df.columns.tolist())
  print(df.head())
  ```
  → Colonnes: timestamp, qqq_close, qqq_volume, spy_close, spy_volume, is_weekend

- [X] **Flag weekend présent et correct** :
  ```python
  from data.macro_fetcher import fetch_macro_daily
  df = fetch_macro_daily(['SPY'], start='2024-01-01', end='2024-01-15')
  weekends = df[df['is_weekend'] == 1]
  print(f'{len(weekends)} lignes weekend sur {len(df)} total')
  ```

## Sentiment Fetcher (Fear & Greed)

- [X] **Fear & Greed actuel** :
  ```python
  from data.sentiment_fetcher import fetch_fear_greed_current
  result = fetch_fear_greed_current()
  print(f"Valeur: {result['value']} ({result['label']})")
  ```
  → Valeur entre 0 et 100

- [X] **Historique Fear & Greed normalisé** :
  ```python
  from data.sentiment_fetcher import fetch_fear_greed_history
  df = fetch_fear_greed_history(limit=30)
  print(f'{len(df)} jours')
  print(f"Min normalisé: {df['fear_greed_normalized'].min():.2f}")
  print(f"Max normalisé: {df['fear_greed_normalized'].max():.2f}")
  ```
  → Valeurs normalisées entre -1 et +1

## News Fetcher (RSS)

- [X] **Récupération des news avec filtrage** :
  ```python
  from data.news_fetcher import fetch_news
  df = fetch_news(filter_by_keywords=True, max_articles=10)
  print(f'{len(df)} articles filtrés')
  print(df[['title', 'source']].to_string())
  ```

- [X] **Récupération sans filtrage** :
  ```python
  from data.news_fetcher import fetch_news
  df = fetch_news(filter_by_keywords=False, max_articles=5)
  print(f'{len(df)} articles total')
  ```

## Pipeline (Orchestrateur)

- [X] **Resampling daily → hourly fonctionne** :
  ```python
  from data.pipeline import _resample_daily_to_hourly
  import pandas as pd
  daily = pd.DataFrame({'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02'], utc=True), 'val': [100, 200]})
  hourly = pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC')
  result = _resample_daily_to_hourly(daily, pd.Series(hourly))
  print(f"{len(result)} heures, val[0]={result.iloc[0]['val']}, val[24]={result.iloc[24]['val']}")
  ```
  → 48 heures, val[0]=100, val[24]=200

## Tests unitaires (depuis PowerShell, PAS le REPL Python)

- [ ] **Tous les tests passent** :
  ```
  python -m pytest tests/test_data.py -v
  ```
  → 16 tests doivent passer (16 passed)

---

✅ **Phase 2 validée** quand toutes les cases sont cochées.
Passe ensuite à la **Phase 3 — Feature Engineering**.
