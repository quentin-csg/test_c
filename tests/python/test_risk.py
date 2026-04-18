import time
from decimal import Decimal
from pathlib import Path

import pytest

from bot.risk import RiskError, RiskManager


@pytest.fixture()
def rm():
    return RiskManager(
        max_delta_pct=Decimal("0.02"),
        margin_buffer_mult=Decimal("3"),
        stale_tick_seconds=5,
    )


def test_delta_ok(rm):
    # spot_qty=1, perp_qty=1, mark=500 each → delta_notional=0 < 2% of 1000
    rm.check_delta(Decimal("1"), Decimal("1"), Decimal("500"), Decimal("500"), Decimal("1000"))


def test_delta_exceeds(rm):
    # spot: 1 BTC × 600, perp: 1 BTC × 400 → delta=200 = 20% > 2% limit
    with pytest.raises(RiskError, match="delta imbalance"):
        rm.check_delta(Decimal("1"), Decimal("1"), Decimal("600"), Decimal("400"), Decimal("1000"))


def test_delta_equity_zero_raises(rm):
    with pytest.raises(RiskError, match="equity is zero"):
        rm.check_delta(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"))


def test_margin_ok(rm):
    rm.check_margin(Decimal("10"), Decimal("100"))  # 10*3=30 < 100 free → ok


def test_margin_insufficient(rm):
    with pytest.raises(RiskError, match="insufficient margin"):
        rm.check_margin(Decimal("50"), Decimal("100"))  # 50*3=150 > 100 free


def test_kill_switch(tmp_path, rm, monkeypatch):
    halt = tmp_path / "HALT"
    halt.touch()
    monkeypatch.setattr("bot.risk.HALT_FILE", halt)
    with pytest.raises(RiskError, match="HALT"):
        rm.check_kill_switch()


def test_funding_floor_trigger(rm):
    assert rm.check_funding_floor(Decimal("-0.05")) is True


def test_funding_floor_ok(rm):
    assert rm.check_funding_floor(Decimal("0.05")) is False


def test_stale_skipped_before_first_tick(rm):
    # No record_tick() called yet — grace period: must not raise
    rm.check_stale()


def test_stale_raises_after_first_tick(rm, monkeypatch):
    rm.record_tick()
    # Simulate 10 seconds passing by backdating the last tick timestamp
    rm._last_tick_ts -= 10
    with pytest.raises(RiskError, match="stale data"):
        rm.check_stale()


def test_stale_ok_if_recent_tick(rm):
    rm.record_tick()
    rm.check_stale()  # should not raise: just recorded


def test_pre_signal_checks_blocks_on_kill_switch(tmp_path, rm, monkeypatch):
    halt = tmp_path / "HALT"
    halt.touch()
    monkeypatch.setattr("bot.risk.HALT_FILE", halt)
    with pytest.raises(RiskError, match="HALT"):
        rm.pre_signal_checks(
            spot_qty=Decimal("0"),
            perp_qty=Decimal("0"),
            spot_mark=Decimal("50000"),
            perp_mark=Decimal("50000"),
            equity=Decimal("1000"),
            maintenance_margin=Decimal("0"),
            free_margin=Decimal("1000"),
            funding_apr=Decimal("0.15"),
        )


def test_pre_signal_checks_blocks_on_delta(rm):
    rm.record_tick()
    # spot: 1 BTC × 600, perp: 1 BTC × 400 → delta=200 = 20% > 2% limit
    with pytest.raises(RiskError, match="delta imbalance"):
        rm.pre_signal_checks(
            spot_qty=Decimal("1"),
            perp_qty=Decimal("1"),
            spot_mark=Decimal("600"),
            perp_mark=Decimal("400"),
            equity=Decimal("1000"),
            maintenance_margin=Decimal("0"),
            free_margin=Decimal("1000"),
            funding_apr=Decimal("0.15"),
        )


def test_pre_signal_checks_passes(rm):
    rm.record_tick()
    rm.pre_signal_checks(
        spot_qty=Decimal("1"),
        perp_qty=Decimal("1"),
        spot_mark=Decimal("500"),
        perp_mark=Decimal("499"),
        equity=Decimal("1000"),
        maintenance_margin=Decimal("5"),
        free_margin=Decimal("900"),
        funding_apr=Decimal("0.15"),
    )
