from decimal import Decimal

import pytest

from bot.strategy import CashCarryStrategy, Signal, StrategyConfig, compute_funding_apr


@pytest.fixture()
def strategy():
    return CashCarryStrategy(StrategyConfig(entry_apr=Decimal("0.10"), exit_apr=Decimal("0.03")))


def tick(strategy, funding_rate: str, spread_mult: str = "1") -> Signal:
    price = Decimal("60000")
    # spread_mult > 1 widens the spread beyond the default 0.1% max
    return strategy.on_tick(
        spot_bid=price * Decimal("0.9995"),
        spot_ask=price * Decimal("1.0005"),
        perp_bid=price * Decimal("0.9998") / Decimal(spread_mult),
        perp_ask=price * Decimal("1.0002") * Decimal(spread_mult),
        funding_rate=Decimal(funding_rate),
    )


def test_no_signal_below_entry(strategy):
    # funding_rate * 3 * 365 = 0.0001 * 1095 = 10.95% but let's use 0.0002 => 21.9% APR
    assert tick(strategy, "0.00005") is Signal.NONE   # ~5.5% APR < 10%


def test_entry_signal_above_threshold(strategy):
    # 0.0003 * 3 * 365 = 32.85% APR > 10%
    assert tick(strategy, "0.0003") is Signal.ENTER


def test_exit_signal_when_in_position(strategy):
    tick(strategy, "0.0003")  # enter
    assert strategy.state.in_position
    # 0.00002 * 1095 = 2.19% APR < 3% exit threshold
    assert tick(strategy, "0.00002") is Signal.EXIT
    assert not strategy.state.in_position


def test_no_double_entry(strategy):
    tick(strategy, "0.0003")   # enter
    result = tick(strategy, "0.0004")  # already in position
    assert result is Signal.NONE


def test_position_sizing_capped(strategy):
    equity = Decimal("10000")
    max_notional = Decimal("500")
    notional = strategy.position_sizing(equity, max_notional)
    assert notional == max_notional


def test_position_sizing_kelly(strategy):
    equity = Decimal("800")
    max_notional = Decimal("500")
    notional = strategy.position_sizing(equity, max_notional)
    assert notional == Decimal("400")  # 50% Kelly of 800


def test_spread_too_wide_blocks_entry(strategy):
    # spread_mult=10 → spread ~1% which is >> max_spread_pct=0.1%
    result = tick(strategy, "0.0003", spread_mult="10")
    assert result is Signal.NONE
    assert not strategy.state.in_position


def test_perp_mid_zero_returns_none(strategy):
    result = strategy.on_tick(
        spot_bid=Decimal("60000"),
        spot_ask=Decimal("60001"),
        perp_bid=Decimal("0"),
        perp_ask=Decimal("0"),
        funding_rate=Decimal("0.001"),
    )
    assert result is Signal.NONE


def test_compute_funding_apr():
    # 0.0003 * 3 * 365 = 0.3285
    assert compute_funding_apr(Decimal("0.0003")) == Decimal("0.3285")


def test_kelly_fraction_configurable():
    cfg = StrategyConfig(kelly_fraction=Decimal("0.25"))
    s = CashCarryStrategy(cfg)
    notional = s.position_sizing(Decimal("1000"), Decimal("10000"))
    assert notional == Decimal("250")  # 25% of 1000
