"""Tests for Orchestrator — PnL round-trip with a fake Rust receiver stub."""
import sys
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.config import BotMode, Settings
from bot.orchestrator import Orchestrator


# ── Fake async receiver ────────────────────────────────────────────────────────

class FakeReceiver:
    def __init__(self, ticks):
        self._it = iter(ticks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _tick(market: str, price: str, fr: str) -> dict:
    p = Decimal(price)
    return {
        "market": market,
        "symbol": "BTCUSDT",
        "ts_ms": 1_700_000_000_000,
        "best_bid": str(p * Decimal("0.9999")),
        "best_bid_qty": "1.0",
        "best_ask": price,
        "best_ask_qty": "1.0",
        "funding_rate": fr,
        "next_funding_ms": 1_700_000_000_000,
        "mark_price": price,
    }


def spot(price: str = "50000.0", fr: str = "0.0001") -> dict:
    return _tick("Spot", price, fr)


def perp(price: str = "50000.0", fr: str = "0.0001") -> dict:
    return _tick("UsdtPerpetual", price, fr)


# fr = 0.0001  → APR = 0.0001 * 3 * 365 = 0.1095  > 10% entry threshold
HIGH_FR = "0.0001"
# fr = 0.000001 → APR = 0.001095 < 3% exit threshold
LOW_FR = "0.000001"
# fr = -0.0001 → APR = -0.1095 < -2% reverse floor
REV_FR = "-0.0001"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_rust(monkeypatch):
    """Inject a fake mn_bot._rust module so the compiled extension is not needed."""
    mod = MagicMock()
    monkeypatch.setitem(sys.modules, "mn_bot", MagicMock())
    monkeypatch.setitem(sys.modules, "mn_bot._rust", mod)
    return mod


def _settings(**kw) -> Settings:
    defaults = dict(
        binance_api_key="x",
        binance_api_secret="x",
        bot_mode=BotMode.paper,
        funding_entry_apr=Decimal("0.10"),
        funding_exit_apr=Decimal("0.03"),
        max_notional_usdt=Decimal("500"),
        margin_buffer_mult=Decimal("3"),
        max_delta_pct=Decimal("0.02"),
        stale_tick_seconds=5,
        log_level="WARNING",
        binance_testnet=True,
    )
    defaults.update(kw)
    return Settings.model_construct(**defaults)


async def _run(mock_rust, ticks, settings=None) -> Orchestrator:
    if settings is None:
        settings = _settings()
    mock_rust.create_market_data_receiver = AsyncMock(return_value=FakeReceiver(ticks))
    orc = Orchestrator(settings)
    await orc.run()
    return orc


# ── Tests ──────────────────────────────────────────────────────────────────────

async def test_enters_on_high_apr(mock_rust):
    # Spot tick fills _spot_tick, perp tick fires _on_both_ticks → ENTER
    orc = await _run(mock_rust, [spot(fr=HIGH_FR), perp(fr=HIGH_FR)])
    assert orc.portfolio.spot_qty > Decimal("0"), "should have entered a position"


async def test_pnl_round_trip(mock_rust):
    """Enter at high APR, exit at low APR — both legs at same price, only fees paid."""
    PRICE = "50000.0"
    ticks = [
        spot(PRICE, HIGH_FR),
        perp(PRICE, HIGH_FR),   # → ENTER
        spot(PRICE, LOW_FR),
        perp(PRICE, LOW_FR),    # → EXIT
    ]
    orc = await _run(mock_rust, ticks)

    assert orc.portfolio.spot_qty == Decimal("0"), "should be flat after EXIT"
    assert orc.portfolio.equity < Decimal("1000"), "fees should have reduced equity"
    # Sanity: two legs × open + close fees on notional ~500 USDT should be < $3
    assert orc.portfolio.equity > Decimal("997"), "total fees should be under $3"


async def test_force_exit_on_reverse_funding(mock_rust):
    """Deeply negative funding triggers immediate close even without EXIT signal."""
    PRICE = "50000.0"
    ticks = [
        spot(PRICE, HIGH_FR),
        perp(PRICE, HIGH_FR),   # → ENTER
        spot(PRICE, REV_FR),
        perp(PRICE, REV_FR),    # → reverse_funding_force_exit
    ]
    orc = await _run(mock_rust, ticks)
    assert orc.portfolio.spot_qty == Decimal("0"), "position should be closed on reverse funding"


async def test_skips_entry_on_kill_switch(mock_rust, tmp_path, monkeypatch):
    """HALT file present → pre_signal_checks raises → no entry despite high APR."""
    halt = tmp_path / "HALT"
    halt.touch()
    monkeypatch.setattr("bot.risk.HALT_FILE", halt)

    orc = await _run(mock_rust, [spot(fr=HIGH_FR), perp(fr=HIGH_FR)])
    assert orc.portfolio.spot_qty == Decimal("0"), "kill-switch should block entry"
