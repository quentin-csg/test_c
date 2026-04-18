"""Vectorbt-based exploration of the cash-and-carry strategy.

Reads data/btcusdt_funding.parquet and data/btcusdt_klines_1m.parquet.
Produces a performance summary and equity curve.

Run via CLI:
  mn-bot backtest --engine vectorbt --start 2023-01-01 --end 2024-01-01
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bot.logger import log

DATA_DIR = Path("data")
FUNDINGS_PER_DAY = 3
DAYS_PER_YEAR = 365
MAKER_FEE = 0.0002   # 0.02% Binance VIP0
TAKER_FEE = 0.0004


def run_vectorbt(start: str, end: str) -> None:
    try:
        import vectorbt as vbt
    except ImportError:
        log.error("vectorbt not installed — run: pip install 'mn-bot[backtest]'")
        return

    funding = _load_funding(start, end)
    klines = _load_klines(start, end)

    # Resample klines to 8h to align with funding periods.
    price_8h = klines["close"].resample("8h").last().reindex(funding.index).ffill()

    funding_apr = funding["fundingRate"] * FUNDINGS_PER_DAY * DAYS_PER_YEAR

    entry_threshold = 0.10
    exit_threshold = 0.03

    entries = (funding_apr > entry_threshold) & (funding_apr.shift(1) <= entry_threshold)
    exits = (funding_apr < exit_threshold) & (funding_apr.shift(1) >= exit_threshold)

    # Vectorized in-position mask: cumsum of entries minus exits, clipped to [0, 1].
    # Entry and exit thresholds are disjoint (entry=10% > exit=3%) so they can't
    # fire at the same timestamp — no cancellation issue.
    pos_changes = entries.astype(int) - exits.astype(int)
    in_position = pos_changes.cumsum().clip(0, 1).astype(bool)

    # Per-period return when in position (funding collected minus amortised entry fee).
    period_return = (
        funding_apr / (FUNDINGS_PER_DAY * DAYS_PER_YEAR)
        - MAKER_FEE / (FUNDINGS_PER_DAY * DAYS_PER_YEAR)
    )
    returns_series = period_return.where(in_position, 0.0)

    equity = (1 + returns_series).cumprod()
    total_return = equity.iloc[-1] - 1
    log.info("backtest_result", total_return_pct=round(total_return * 100, 2))

    try:
        pf = vbt.Portfolio.from_returns(returns_series, init_cash=1000.0)
        stats = pf.stats()
        log.info("vectorbt_stats", stats=stats.to_dict())
    except Exception as e:
        log.warning("vectorbt_stats_failed", error=str(e))
        log.info("manual_stats", sharpe=_sharpe(returns_series))


def _load_funding(start: str, end: str) -> pd.DataFrame:
    path = DATA_DIR / "btcusdt_funding.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run 'mn-bot download' first. Missing: {path}")
    df = pd.read_parquet(path)
    return df.loc[start:end]


def _load_klines(start: str, end: str) -> pd.DataFrame:
    path = DATA_DIR / "btcusdt_klines_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run 'mn-bot download' first. Missing: {path}")
    df = pd.read_parquet(path)
    return df.loc[start:end]


def _sharpe(returns: pd.Series, periods_per_year: int = FUNDINGS_PER_DAY * DAYS_PER_YEAR) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * (periods_per_year ** 0.5))
