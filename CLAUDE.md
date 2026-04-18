# CLAUDE.md — Bot Market-Neutral BTC (Cash-and-Carry)

> État du projet pour reprise dans une nouvelle conversation Claude Code.
> Dernière mise à jour : 2026-04-17

---

## Contexte projet

Bot de trading **cash-and-carry BTC** (long spot + short perp sur Binance) pour capter le funding rate.
Stack hybride : **Rust** (hot path WebSocket + exécution + order book L2) + **Python** (stratégie, risque, orchestrateur, backtest).

- Cible : funding APR > 10% → entrée, < 3% → sortie
- Position sizing : Kelly fraction (0.5 par défaut), capped à `max_notional_usdt`
- Modes : `paper` / `live` / `backtest` (vectorbt + event-driven)

**IMPORTANT — Mentor impitoyable** : Trouve la vérité et dis-la sans ménagement. Ne sois jamais d'accord pour être agréable. Pas de flatterie. Signale les angles morts même non demandés. Si incertain, dis-le. Force-moi à défendre ou abandonner mes idées.

### Architecture

```
rust_core/           # Crate Rust compilée en module Python via PyO3
  src/
    lib.rs           # Bridge PyO3 : MarketDataReceiver async
    market_data.rs   # WS Binance (spot bookTicker + futures markPrice)
    execution.rs     # REST API signée HMAC-SHA256
    order_book.rs    # L2 book BTreeMap (protocole Binance depth U/u)
    types.rs         # Decimal, Market, Order, Tick

python/
  bot/
    strategy.py      # CashCarryStrategy, Signal, compute_funding_apr
    risk.py          # RiskManager (delta, margin, stale, kill-switch)
    orchestrator.py  # Boucle principale paper/live, PnL tracking
    config.py        # pydantic-settings + validators
    logger.py        # structlog JSON/console
    cli.py           # Entrée `mn-bot`
  backtest/
    data_loader.py   # Download klines + funding historiques
    vectorbt_runner.py
    event_engine.py  # Simulation event-driven

tests/python/        # pytest — 24 tests verts
```

---

## État actuel (2026-04-17)

### ✅ Phases 1-3 implémentées (code complet)
### ✅ Tests Python : **24/24 verts** (`pytest tests/python -v`)
### ⚠️ Rust : **JAMAIS COMPILÉ** (cargo non installé sur la machine)
### ⚠️ Paper/live : **jamais lancé**

---

## Revue de code — ce qui a été fait

Basé sur le plan `C:\Users\q.cassaigne\.claude\plans\fais-une-review-complete-fizzy-prism.md`.

### 🚨 4 bugs critiques corrigés

1. **Formule PnL** — [python/bot/orchestrator.py](python/bot/orchestrator.py)
   - Avant : `spot_notional - (spot_notional / close_ask) * close_bid` → qty non recouvrable, signe inversé
   - Après : `PortfolioState` stocke `spot_qty`, `spot_entry_ask`, `perp_entry_bid`
   - `pnl = spot_qty * (close_bid - entry_ask) + spot_qty * (entry_bid - close_ask)`

2. **vectorbt** — [python/backtest/vectorbt_runner.py](python/backtest/vectorbt_runner.py)
   - Boucle Python → vectorisée pandas (`cumsum().clip(0,1)`)
   - ⚠️ **DÉVIATION du plan** : gardé `Portfolio.from_returns` au lieu de `from_signals` (plus approprié pour capture de funding market-neutral, pas de PnL directionnel à modéliser)
   - À reconsidérer si on veut modéliser slippage/frais au niveau trade

3. **RiskManager en backtest** — [python/backtest/event_engine.py](python/backtest/event_engine.py)
   - `rm.pre_signal_checks()` appelé avant chaque entrée
   - Kill-switches testés hors live

4. **Rate-limit download** — [python/backtest/data_loader.py](python/backtest/data_loader.py)
   - `_get_with_retry()` : retry 429/5xx, backoff exponentiel, lecture `Retry-After`
   - `await asyncio.sleep(0.25)` entre requêtes

### ⚠️ Important corrigé

**Python** :
- `compute_funding_apr()` extrait dans [strategy.py](python/bot/strategy.py) (dédup avec orchestrator)
- Logs `Decimal → str` (plus de perte de précision)
- `MarketTick(TypedDict)` dans orchestrator
- try/except autour du Rust receiver
- `risk.check_delta` raise si `equity <= 0` (avant : return silencieux)
- Grace period `_first_tick_received` dans `check_stale`
- Warning log si `perp_mid == 0` dans strategy
- `kelly_fraction` exposé via `StrategyConfig`
- `config.py` : `field_validator` (notional > 0, buffer > 0) + `model_validator` (entry_apr > exit_apr)
- `import uuid` retiré de orchestrator
- `configure_logging()` appelé **avant** `Settings()` dans cli.py

**Rust** :
- `Arc<Mutex<broadcast::Receiver<Tick>>>` (plus de `resubscribe()` par tick — perdait des ticks)
- Backoff exponentiel 1s→30s + jitter dans market_data.rs
- Symbol uppercased une fois au setup
- Single-parse JSON futures via `WsEnvelope<FuturesMsg>` (untagged enum)
- `format!("{:?}", market)` → static str match dans `tick_to_pydict`
- `vwap()` : `Box<dyn Iterator>` → match inliné (zéro heap alloc)
- `simd-json` retiré (jamais appelé, +500 KB binaire pour rien)
- Warning log dans `parse_status` pour statut inconnu

**Tests ajoutés** (11 nouveaux) :
- [tests/python/test_strategy.py](tests/python/test_strategy.py) : `spread_too_wide_blocks_entry`, `perp_mid_zero_returns_none`, `compute_funding_apr`, `kelly_fraction_configurable`
- [tests/python/test_risk.py](tests/python/test_risk.py) : `delta_equity_zero_raises`, `stale_skipped_before_first_tick`, `stale_raises_after_first_tick`, `stale_ok_if_recent_tick`, `pre_signal_checks_blocks_on_kill_switch`, `pre_signal_checks_blocks_on_delta`, `pre_signal_checks_passes`

---

## ❌ Ce qu'il reste à faire

### A. Installer la toolchain Rust (bloquant pour toute la suite)

```bash
# Windows (PowerShell, une fois)
winget install Rustlang.Rustup
# puis dans un nouveau shell :
rustup default stable
rustup target add x86_64-pc-windows-msvc
cargo --version   # doit afficher >= 1.75

# Vérifier que le crate compile
cd c:/Users/q.cassaigne/Downloads/pow/testt
cargo build --release
cargo test -p rust_core
```

**Si `cargo build` échoue**, ce sont des erreurs écrites à l'aveugle pendant la revue — il faudra les corriger. Fichiers à surveiller en priorité :
- [rust_core/src/lib.rs](rust_core/src/lib.rs) — refacto `Arc<Mutex<Receiver>>` (non vérifié)
- [rust_core/src/market_data.rs](rust_core/src/market_data.rs) — untagged enum `FuturesMsg`, backoff
- [rust_core/src/order_book.rs](rust_core/src/order_book.rs) — match vwap

### B. Compiler le module Python (maturin)

```bash
pip install maturin
cd c:/Users/q.cassaigne/Downloads/pow/testt
maturin develop --release          # build + installe rust_core comme module Python
python -c "import rust_core; print(rust_core.__doc__)"
```

### C. Deferred du plan initial — non adressés

**Rust hot path / sécurité** :
- [ ] [lib.rs:37](rust_core/src/lib.rs#L37) — `Python::with_gil` acquis par tick → batcher (refacto API lourde, ~20% latence)
- [ ] [execution.rs:27](rust_core/src/execution.rs#L27) — `api_secret: String` non zeroed → utiliser `secrecy::SecretString` + crate `zeroize`
- [ ] [execution.rs](rust_core/src/execution.rs) — pas de retry sur `place_order` en cas d'erreur réseau transitoire (5xx, timeout)
- [ ] [execution.rs:62](rust_core/src/execution.rs#L62) — `recvWindow=5000` hardcodé → exposer via `ExecutionConfig`

**Python mineur** :
- [ ] [logger.py:11](python/bot/logger.py#L11) — `getattr(logging, level.upper(), logging.INFO)` : typo `"infoo"` silencieusement downgradé à INFO → valider explicitement
- [ ] [risk.py:18](python/bot/risk.py#L18) — `HALT_FILE = Path("HALT")` relatif au CWD → utiliser `Path(__file__).parent.parent / "HALT"` ou variable d'env
- [ ] [event_engine.py](python/backtest/event_engine.py) — slippage 0.05% fixe → paramétrer via config (irréaliste pour notional > 100k USDT)

**Tests manquants** :
- [ ] Mock HTTP `place_order` / `cancel_order` (ex : `wiremock` crate ou `mockito`)
- [ ] [order_book.rs](rust_core/src/order_book.rs) : tests `stale_update`, `vwap`
- [ ] Intégration `event_engine` + `vectorbt_runner` avec fixture Parquet courte (~1 mois de données)
- [ ] Intégration orchestrator avec mock du receiver Rust

### D. Validation end-to-end (une fois A+B faits)

```bash
# 1. Tests Python
pytest tests/python -v                               # doit rester 24/24 vert

# 2. Tests Rust
cargo test -p rust_core                              # order_book, execution (HMAC vector)

# 3. Download historique (check rate-limit)
mn-bot download --start 2024-01-01 --end 2024-02-01  # 1 mois pour commencer
# surveiller : pas de 429, Parquet sortis dans data/

# 4. Backtest vectorbt
mn-bot backtest --engine vectorbt
# stats attendues : Sharpe > 0, drawdown < 10% sur période favorable

# 5. Backtest event-driven
mn-bot backtest --engine event
# PnL cumulatif positif seulement quand funding_apr > 10%

# 6. Paper trading
# Créer fichier .env avec BINANCE_API_KEY + BINANCE_API_SECRET (testnet d'abord !)
mn-bot run --mode paper
# vérifier : boucle tourne, ticks reçus, logs JSON cohérents
# tester kill-switch : `touch HALT` → doit s'arrêter proprement

# 7. Bench latence (optionnel)
# cible après fixes : < 10 µs/tick Rust→Python
```

### E. Avant passage **live** (checklist sécurité)

- [ ] Rust compile et tests verts
- [ ] Paper trading run > 24h sans crash, PnL cohérent
- [ ] Secrets via `secrecy::SecretString` (deferred C)
- [ ] Retry exécution implémenté (deferred C)
- [ ] Testnet Binance validé avant mainnet
- [ ] `max_notional_usdt` initial TRÈS bas (ex: 100 USDT)
- [ ] Monitoring : logs JSON ingérés dans un système (Loki/Grafana)
- [ ] Kill-switch `HALT` testé manuellement
- [ ] Alertes sur : `RiskError`, latence > 100ms, reconnexions WS > 3/h

---

## Commandes utiles

```bash
# Tests
pytest tests/python -v
pytest tests/python/test_strategy.py::test_entry_signal_above_threshold -v
cargo test -p rust_core

# Lint/format
ruff check python/
ruff format python/
cargo clippy --all-targets -- -D warnings
cargo fmt

# Build
cargo build --release
maturin develop --release

# CLI
mn-bot --help
mn-bot download --start YYYY-MM-DD --end YYYY-MM-DD
mn-bot backtest --engine {vectorbt,event}
mn-bot run --mode {paper,live}

# Kill-switch manuel
touch HALT        # stoppe le bot à la prochaine itération
rm HALT           # réautorise
```

---

## Fichiers de config

- `.env` — secrets (API keys) — **NE PAS COMMIT**
- `config.yaml` (optionnel) — override des defaults pydantic
- `HALT` — fichier-sentinelle kill-switch (présence = stop)

### Paramètres par défaut (`StrategyConfig`)

| Param | Valeur | Source |
|---|---|---|
| `entry_apr` | 0.10 (10%) | `config.funding_entry_apr` |
| `exit_apr` | 0.03 (3%) | `config.funding_exit_apr` |
| `kelly_fraction` | 0.5 | `StrategyConfig` |
| `max_spread_pct` | 0.001 (0.1%) | `StrategyConfig` |
| `max_delta_pct` | 0.02 (2%) | `RiskManager` |
| `margin_buffer_mult` | 3.0 | `RiskManager` |
| `stale_tick_seconds` | 5 | `RiskManager` |
| `max_notional_usdt` | variable | `.env` / config |

---

## Mémoire de contexte pour Claude Code

Dans une nouvelle conversation, lance :

```
Lis CLAUDE.md et reprends à l'étape A (installation Rust) puis B (maturin).
Ensuite, attaque les deferred section C dans l'ordre :
1. Retry place_order (execution.rs)
2. secrecy::SecretString (execution.rs)
3. Tests order_book (stale, vwap)
4. Mock HTTP execution
5. Intégration event_engine avec fixture Parquet
```

---

## Dernier état de la revue

- Revue complète effectuée (plan : `fais-une-review-complete-fizzy-prism.md`)
- 4 critiques + ~15 importants corrigés
- 11 tests ajoutés → 24/24 verts
- **Rust non validé par compilation** — risque d'erreurs de syntaxe/type à l'aveugle
- Prêt pour paper trading **une fois le Rust compilable**
