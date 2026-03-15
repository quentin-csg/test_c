# Phase 2 — Résumé

## Objectif
Implémenter le pipeline complet d'ingestion des données : crypto, macro, sentiment et actualités (BLOC 1 du instruction.md).

## Ce qui a été implémenté

### 📊 `data/crypto_fetcher.py` — Données Crypto (ccxt)
| Fonction | Description |
|----------|-------------|
| `fetch_ohlcv()` | Récupère les bougies OHLCV (Open, High, Low, Close, Volume) via ccxt. Supporte la pagination automatique pour les gros historiques. |
| `fetch_multi_timeframe()` | Récupère les données sur les 2 timeframes (1h + 4h) simultanément. |
| `fetch_funding_rate()` | Récupère le taux de financement des contrats à terme (Futures). |
| `fetch_order_book()` | Récupère le carnet d'ordres L2 et calcule l'**imbalance** (ratio bid/ask, entre -1 et +1). |
| `fetch_open_interest()` | Récupère l'Open Interest (positions ouvertes sur les contrats à terme). |

- Exchange : Binance (configurable)
- Paire : BTC/USDT (configurable)
- Timeframes : 1h (principal) + 4h (secondaire)
- API : publique par défaut, authentifiée si clés fournies

### 📈 `data/macro_fetcher.py` — Données Macroéconomiques (yfinance)
| Fonction | Description |
|----------|-------------|
| `fetch_macro_data()` | Récupère QQQ (NASDAQ) et SPY (S&P500) en intraday. |
| `fetch_macro_daily()` | Récupère les données daily (plus fiable pour le long terme 2020-2023). |

- **Gestion du weekend** : Flag `is_weekend` ajouté automatiquement (samedi=5, dimanche=6)
- **Forward-fill** : Les données manquantes (weekend, jours fériés) sont propagées automatiquement
- Colonnes générées : `{symbol}_close`, `{symbol}_volume`, `{symbol}_high`, `{symbol}_low`

### 😱 `data/sentiment_fetcher.py` — Fear & Greed Index (Alternative.me)
| Fonction | Description |
|----------|-------------|
| `fetch_fear_greed_current()` | Récupère le score actuel (0-100 + label). |
| `fetch_fear_greed_history()` | Récupère l'historique complet avec normalisation. |

- Score brut : 0 (peur extrême) → 100 (avidité extrême)
- **Normalisation incluse** : `fear_greed_normalized` = score ramené entre -1 et +1
- Filtrage par dates (start/end)

### 📰 `data/news_fetcher.py` — Actualités (feedparser)
| Fonction | Description |
|----------|-------------|
| `fetch_news()` | Récupère les titres d'articles via flux RSS avec filtrage par mots-clés. |
| `fetch_news_titles_for_hour()` | Récupère les titres pour une heure donnée (pour NLP Phase 3). |

- Sources : CoinDesk, CoinTelegraph, Yahoo Finance
- Mots-clés de filtrage : BTC, Bitcoin, FED, ETF, SEC, crypto, inflation, interest rate, FOMC, regulation
- Les titres sont conservés pour l'analyse FinBERT (Phase 3)

### 🔗 `data/pipeline.py` — Orchestrateur
| Fonction | Description |
|----------|-------------|
| `build_dataset()` | Assemble crypto + macro + sentiment + funding rate en un seul DataFrame aligné sur la grille 1h. |
| `get_news_for_dataset()` | Agrège les news par heure (count + titres). |
| `_resample_daily_to_hourly()` | Resample les données daily sur la grille horaire via `merge_asof`. |

- Merge via `merge_asof` (backward) pour aligner daily → hourly
- Forward-fill + backward-fill des valeurs manquantes
- Flag `is_weekend` automatique

### 🧪 `tests/test_data.py` — 16 Tests unitaires
- 5 tests CryptoFetcher (import, exchange, OHLCV, invalid symbol, order book)
- 3 tests MacroFetcher (import, daily, weekend flag)
- 3 tests SentimentFetcher (import, current, history + normalisation)
- 3 tests NewsFetcher (import, fetch, keyword filtering)
- 2 tests Pipeline (import, resampling daily→hourly)

## Technologies utilisées dans cette phase
| Technologie | Usage |
|------------|-------|
| **ccxt** | API unifiée pour exchanges crypto (Binance, Bybit) |
| **yfinance** | Données boursières macro (QQQ, SPY) |
| **feedparser** | Parsing des flux RSS (news crypto/finance) |
| **requests** | Appels HTTP vers API Alternative.me |
| **pandas** | Manipulation, merge et alignement temporel des données |
| **pytest** | Tests unitaires |

## Dépendances installées
Toutes les dépendances de `requirements.txt` ont été installées avec succès (16 packages + dépendances transitives).

## Statut
✅ **Phase 2 validée** — 16/16 tests passent. Tous les fetchers et le pipeline sont opérationnels.
