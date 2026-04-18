# TODO — mn-bot

## ✅ Fait

### Infrastructure

- Rust toolchain + `cargo build --release` OK
- `maturin develop --release` — module Python buildé
- **37/37 tests Python verts**, **16/16 tests Rust verts**
- CI GitHub Actions : `pytest` + `cargo test` + `ruff` + `cargo clippy` sur push/PR

### P1 (avant paper trading)

- HALT path absolu (`Path(__file__).resolve().parents[2] / "HALT"`), frais paper trading, position state sync
- Logger : validation explicite du log level (raise `ValueError` si invalide)
- Backtest event-driven lit les paramètres depuis `.env`
- Log spam `reverse_funding_trigger` supprimé

### P2 (sans clé API)

- Retry HTTP avec backoff exponentiel (500ms→1s→2s, max 3 tentatives, retry sur 5xx + timeout)
- `Zeroizing<String>` pour `api_secret` — zéroïsé en mémoire à la destruction
- Nouvelle méthode `futures_account_info` (prête pour la synchro equity)
- Tests wiremock : 200, retry 500, fail fast 400, cancel, futures account

### P3 — Tests Python

- `test_orchestrator.py` — PnL round-trip avec FakeReceiver (ENTER/EXIT/force_exit/kill-switch)
- `test_event_engine.py` — fixtures Parquet synthétiques, funding capturé, empty data raises
- `test_data_loader.py` — retry 429/5xx via pytest-httpx, fail-fast 400, max retries

### P3 partiel (antérieur)

- Tests `order_book.rs` : `stale_update`, `vwap_bid/ask`, `vwap_empty`, `vwap_depth`
- Tests `execution.rs` via wiremock (5 tests)

### P4 — Améliorations mineures

- `recvWindow` exposé via `Settings.recv_window_ms` + `ExecutionConfig.recv_window_ms`
- Slippage backtest paramétrable via `Settings.backtest_slippage_pct` (défaut 0.05%)
- Delta recalculé sur mark-to-market (`check_delta` accepte qty + prix mark courants)
- GIL batching : `BatchReceiver` dans `rust_core/src/lib.rs`, orchestrator consomme via `receiver.batches(32)` — une acquisition GIL pour jusqu'à 32 ticks

### P5 — Observabilité

- Log sink fichier JSON rotatif (`Settings.log_file`, `RotatingFileHandler`)
- Events WS reconnect structurés dans `market_data.rs` (champ `market` pour filtrage Loki)
- Stack monitoring `monitoring/` : Loki + Promtail + Grafana (docker compose)
- Règles d'alerte Loki : `RiskError` rate > 0 sur 5min, WS reconnects > 3/h

### Divers

- Backtest validé : +10.59% sur 2021, +2.42% sur 2024 avec seuil 10% APR
- README.md créé

---

## ❌ Reste à faire

### P1 — Paper trading (en cours)

- [ ] Laisser `mn-bot run --mode paper` tourner 24-48h sans interruption

---

### P2 — Nécessite clé API Binance

#### P2.1 — Synchro equity live via `futures_account_info`

La méthode Rust existe déjà (`execution.rs:82`). Ce qui manque :

**Étape A — Exposer `ExecutionClient` à Python via PyO3 (`rust_core/src/lib.rs`)**

Ajouter un `#[pyclass]` wrapper minimal :

```rust
#[pyclass]
pub struct PyExecutionClient(ExecutionClient);

#[pymethods]
impl PyExecutionClient {
    /// Returns (equity_usdt, maintenance_margin_usdt)
    fn get_equity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // appel futures_account_info + parse totalWalletBalance + totalMaintMargin
    }
}

#[pyfunction]
fn create_execution_client(
    api_key: &str, api_secret: &str, testnet: bool, recv_window_ms: u64
) -> PyResult<PyExecutionClient> { ... }
```

Le JSON `/fapi/v2/account` retourne (entre autres) :

```json
{ "totalWalletBalance": "1234.56", "totalMaintMargin": "12.34", ... }
```

Extraire `data["totalWalletBalance"]` et `data["totalMaintMargin"]` → convertir en `Decimal`.

**Étape B — Modifier `Orchestrator.__init__` (`python/bot/orchestrator.py`)**

```python
from mn_bot._rust import create_execution_client  # ajout

class Orchestrator:
    def __init__(self, settings):
        ...
        self._exec_client = None
        if settings.bot_mode == BotMode.live:
            self._exec_client = create_execution_client(
                settings.binance_api_key,
                settings.binance_api_secret,
                settings.binance_testnet,
                settings.recv_window_ms,
            )
```

**Étape C — Ajouter `_refresh_equity` et la tâche périodique (`orchestrator.py`)**

```python
async def _refresh_equity(self, interval_s: int = 30) -> None:
    while True:
        await asyncio.sleep(interval_s)
        if self._exec_client is None:
            return
        try:
            equity, maint = await self._exec_client.get_equity()
            self.portfolio.equity = equity
            self.portfolio.maintenance_margin = maint
            log.debug("equity_refreshed", equity=str(equity), maint=str(maint))
        except Exception as e:
            log.warning("equity_refresh_failed", error=str(e))

async def run(self) -> None:
    ...
    # Lancer la synchro en parallèle de la boucle ticks
    refresh_task = asyncio.create_task(self._refresh_equity(interval_s=30))
    try:
        async for batch in receiver.batches(32):
            ...
    finally:
        refresh_task.cancel()
```

**Tests à ajouter** : mock `get_equity` dans `test_orchestrator.py` (retourne `(Decimal("1050"), Decimal("5"))`) et vérifier que `portfolio.equity` est mis à jour après la synchro.

#### P2.2 — Compte Binance Testnet

- Action externe : créer un compte sur [testnet.binancefuture.com](https://testnet.binancefuture.com)
- Générer API key + secret, les mettre dans `.env` :

  ```env
  BINANCE_API_KEY=xxx
  BINANCE_API_SECRET=yyy
  BINANCE_TESTNET=true
  ```

- Valider avec : `mn-bot run --mode live` (les ordres vont sur testnet, aucun risque)

#### P2.3 — Limiter l'exposition initiale

Dans `.env` avant le premier live :

```env
MAX_NOTIONAL_USDT=100
```

(`Settings.max_notional_usdt` est déjà lu depuis `.env`)

---

### Post-paper (nécessite données live)

#### Alertes latence > 100ms

La latence WS→Python peut être mesurée sans infra supplémentaire. `ts_ms` dans chaque tick est déjà le timestamp Rust au moment de la réception WS (`types::now_ms()` dans `market_data.rs:114`).

Ce qu'il faut ajouter dans `orchestrator.py` (dans `_on_both_ticks`) :

```python
import time
latency_ms = int(time.time() * 1000) - int(perp["ts_ms"])
if latency_ms > 100:
    log.warning("high_latency", latency_ms=latency_ms)
```

Le seuil 100ms est une hypothèse — calibrer d'abord sur 1 semaine de paper pour voir la distribution réelle.

Ajouter ensuite une règle Loki dans `monitoring/loki/rules.yml` :

```yaml
- alert: HighLatency
  expr: count_over_time({app="mn-bot"} |= "high_latency" [5m]) > 3
```

#### Réévaluer Kelly fraction

Une fois 1 mois de données live disponibles, calculer le Sharpe ratio et le win-rate réel. Si Sharpe > 1.5 et drawdown < 5%, monter `kelly_fraction` de `0.5` à `0.7` dans `.env` :

```env
KELLY_FRACTION=0.7
```

(`StrategyConfig.kelly_fraction` est déjà lu depuis `Settings` → pas de code à modifier)

#### Routing alertes Loki

Remplacer dans `monitoring/loki-config.yml` :

```yaml
alertmanager_url: http://localhost:9093
```

par l'URL Alertmanager réel, puis configurer `alertmanager.yml` avec un receiver Slack/PagerDuty. Aucune modification du code bot.
