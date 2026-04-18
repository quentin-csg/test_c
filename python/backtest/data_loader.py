"""Download historical klines (1m) and funding rates from Binance REST API.

Outputs:
  data/btcusdt_klines_1m.parquet   — OHLCV for spot BTCUSDT
  data/btcusdt_funding.parquet     — funding rate history (perp)
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from bot.logger import log

SPOT_REST = "https://api.binance.com"
FUTURES_REST = "https://fapi.binance.com"

KLINES_LIMIT = 1000      # max per request
FUNDING_LIMIT = 1000

# Conservative: klines weight=2, funding weight=5; stay well under 1200/min.
_SLEEP_BETWEEN_REQS = 0.25   # 4 req/sec max per stream = 240/min per stream
_MAX_RETRIES = 5


async def download_all(start: str, end: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)

    async with httpx.AsyncClient(timeout=30) as client:
        await asyncio.gather(
            _download_klines(client, start_ms, end_ms, out_dir),
            _download_funding(client, start_ms, end_ms, out_dir),
        )


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


async def _get_with_retry(client: httpx.AsyncClient, url: str, params: dict) -> list:
    """GET with exponential backoff on 429 / 5xx errors."""
    backoff = 1.0
    for attempt in range(_MAX_RETRIES):
        resp = await client.get(url, params=params)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", str(int(backoff * 60))))
            log.warning("rate_limited", url=url, retry_after=retry_after, attempt=attempt)
            await asyncio.sleep(retry_after)
            backoff = min(backoff * 2, 60.0)
            continue
        if resp.status_code >= 500:
            log.warning("server_error", status=resp.status_code, attempt=attempt)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed after {_MAX_RETRIES} attempts: {url}")


async def _download_klines(client: httpx.AsyncClient, start_ms: int, end_ms: int, out_dir: Path) -> None:
    log.info("download_klines_start")
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        data = await _get_with_retry(
            client,
            f"{SPOT_REST}/api/v3/klines",
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": KLINES_LIMIT,
            },
        )
        if not data:
            break
        rows.extend(data)
        cursor = data[-1][0] + 60_000  # next minute
        log.debug("klines_progress", n=len(rows))
        await asyncio.sleep(_SLEEP_BETWEEN_REQS)

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df = df.astype({c: "float64" for c in ["open", "high", "low", "close", "volume", "quote_volume"]})
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    path = out_dir / "btcusdt_klines_1m.parquet"
    df.to_parquet(path)
    log.info("klines_saved", path=str(path), rows=len(df))


async def _download_funding(client: httpx.AsyncClient, start_ms: int, end_ms: int, out_dir: Path) -> None:
    log.info("download_funding_start")
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        data = await _get_with_retry(
            client,
            f"{FUTURES_REST}/fapi/v1/fundingRate",
            {
                "symbol": "BTCUSDT",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": FUNDING_LIMIT,
            },
        )
        if not data:
            break
        rows.extend(data)
        cursor = data[-1]["fundingTime"] + 1
        log.debug("funding_progress", n=len(rows))
        await asyncio.sleep(_SLEEP_BETWEEN_REQS)

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype("float64")
    df.set_index("fundingTime", inplace=True)
    path = out_dir / "btcusdt_funding.parquet"
    df.to_parquet(path)
    log.info("funding_saved", path=str(path), rows=len(df))
