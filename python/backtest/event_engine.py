"""Event-driven backtest engine — shares strategy.py with the live orchestrator.

Simulates:
  - Funding rate payments every 8h.
  - Maker fees on entry/exit (spot ask + perp bid at time of signal).
  - Simple slippage: entry at ask, exit at bid.
  - Risk checks (delta, margin, kill-switch) via shared RiskManager.
  - Margin calls / stop if equity < 10% of initial.

Run via CLI:
  mn-bot backtest --engine event --start 2023-01-01 --end 2024-01-01
"""

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pandas as pd

from bot.logger import log
from bot.risk import RiskError, RiskManager
from bot.strategy import CashCarryStrategy, Signal, StrategyConfig

DATA_DIR = Path("data")
MAKER_FEE = Decimal("0.0002")
TAKER_FEE = Decimal("0.0004")


@dataclass
class SimPortfolio:
    equity: Decimal = Decimal("1000")
    spot_qty: Decimal = Decimal("0")       # BTC held long
    spot_entry_price: Decimal = Decimal("0")
    perp_entry_price: Decimal = Decimal("0")
    perp_qty: Decimal = Decimal("0")       # BTC short on perp
    cumulative_funding: Decimal = Decimal("0")
    trades: list = field(default_factory=list)


async def run_event_backtest(start: str, end: str) -> None:
    from bot.config import Settings
    settings = Settings()

    funding_df = _load(DATA_DIR / "btcusdt_funding.parquet", start, end)
    klines_df = _load(DATA_DIR / "btcusdt_klines_1m.parquet", start, end)

    if funding_df.empty:
        raise ValueError(f"No funding data for {start} → {end}. Run 'mn-bot download' for this period.")

    # Resample to 8h candles for event resolution.
    price_8h = (
        klines_df["close"]
        .resample("8h")
        .last()
        .reindex(funding_df.index, method="ffill")
    )

    slippage = settings.backtest_slippage_pct

    strategy = CashCarryStrategy(StrategyConfig(
        entry_apr=settings.funding_entry_apr,
        exit_apr=settings.funding_exit_apr,
    ))
    risk = RiskManager(
        max_delta_pct=settings.max_delta_pct,
        margin_buffer_mult=settings.margin_buffer_mult,
        stale_tick_seconds=300,  # relaxed for 8h event cadence
    )
    portfolio = SimPortfolio()
    initial_equity = portfolio.equity

    for ts, funding_rate in funding_df["fundingRate"].items():
        price = Decimal(str(price_8h.loc[ts]))
        fr = Decimal(str(funding_rate))

        # Mark tick so stale check passes.
        risk.record_tick()

        # Funding payment if in position: we receive as short perp holder.
        if portfolio.perp_qty > 0:
            payment = portfolio.perp_qty * portfolio.perp_entry_price * fr
            portfolio.equity += payment
            portfolio.cumulative_funding += payment

        spot_bid = price * (1 - slippage)
        spot_ask = price * (1 + slippage)
        perp_bid = price * (1 - slippage / 2)
        perp_ask = price * (1 + slippage / 2)

        signal = strategy.on_tick(
            spot_bid=spot_bid,
            spot_ask=spot_ask,
            perp_bid=perp_bid,
            perp_ask=perp_ask,
            funding_rate=fr,
        )

        # Risk gate: skip entry if any check fails.
        if signal is Signal.ENTER and portfolio.spot_qty == 0:
            try:
                risk.pre_signal_checks(
                    spot_qty=Decimal("0"),
                    perp_qty=Decimal("0"),
                    spot_mark=price,
                    perp_mark=price,
                    equity=portfolio.equity,
                    maintenance_margin=Decimal("0"),
                    free_margin=portfolio.equity,
                    funding_apr=strategy.state.last_funding_apr,
                )
            except RiskError as e:
                log.warning("backtest_risk_blocked", ts=str(ts), reason=str(e))
                signal = Signal.NONE

        if signal is Signal.ENTER and portfolio.spot_qty == 0:
            notional = min(portfolio.equity * Decimal("0.5"), Decimal("500"))
            qty = (notional / spot_ask).quantize(Decimal("0.00001"))
            fee = qty * spot_ask * MAKER_FEE * 2   # spot + perp entry
            portfolio.equity -= fee
            portfolio.spot_qty = qty
            portfolio.perp_qty = qty
            portfolio.spot_entry_price = spot_ask
            portfolio.perp_entry_price = perp_bid
            log.debug("event_enter", ts=str(ts), price=float(spot_ask), qty=float(qty))

        elif signal is Signal.EXIT and portfolio.spot_qty > 0:
            pnl_spot = portfolio.spot_qty * (spot_bid - portfolio.spot_entry_price)
            pnl_perp = portfolio.perp_qty * (portfolio.perp_entry_price - perp_ask)
            fee = portfolio.spot_qty * spot_bid * TAKER_FEE * 2
            net_pnl = pnl_spot + pnl_perp - fee
            portfolio.equity += net_pnl
            portfolio.trades.append({
                "exit_ts": ts, "pnl": float(net_pnl),
                "equity": float(portfolio.equity),
            })
            portfolio.spot_qty = Decimal("0")
            portfolio.perp_qty = Decimal("0")
            log.debug("event_exit", ts=str(ts), net_pnl=float(net_pnl), equity=float(portfolio.equity))

        # Margin call guard.
        if portfolio.equity < initial_equity * Decimal("0.1"):
            log.warning("margin_call", equity=float(portfolio.equity))
            break

    total_return = (portfolio.equity - initial_equity) / initial_equity
    log.info(
        "event_backtest_result",
        total_return_pct=round(float(total_return) * 100, 2),
        cumulative_funding=float(portfolio.cumulative_funding),
        num_trades=len(portfolio.trades),
        final_equity=float(portfolio.equity),
    )


def _load(path: Path, start: str, end: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Run 'mn-bot download' first. Missing: {path}")
    return pd.read_parquet(path).loc[start:end]
