"""Tests for data_loader._get_with_retry — HTTP retry logic via pytest-httpx."""
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from backtest.data_loader import _MAX_RETRIES, _get_with_retry

_URL = "https://api.binance.com/api/v3/klines"
_OK_BODY = [{"open_time": 1, "close": "50000"}]


async def test_success_on_first_attempt(httpx_mock):
    httpx_mock.add_response(json=_OK_BODY)

    async with httpx.AsyncClient() as client:
        result = await _get_with_retry(client, _URL, {"symbol": "BTCUSDT"})

    assert result == _OK_BODY


async def test_retry_on_429_then_success(httpx_mock, monkeypatch):
    """Two 429s followed by a 200: should succeed after retrying."""
    monkeypatch.setattr("backtest.data_loader.asyncio", _fast_asyncio())

    httpx_mock.add_response(status_code=429, headers={"Retry-After": "0"})
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "0"})
    httpx_mock.add_response(json=_OK_BODY)

    async with httpx.AsyncClient() as client:
        result = await _get_with_retry(client, _URL, {})

    assert result == _OK_BODY
    assert len(httpx_mock.get_requests()) == 3


async def test_retry_on_500_then_success(httpx_mock, monkeypatch):
    """Two 5xx responses followed by a 200: should succeed after retrying."""
    monkeypatch.setattr("backtest.data_loader.asyncio", _fast_asyncio())

    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(json=_OK_BODY)

    async with httpx.AsyncClient() as client:
        result = await _get_with_retry(client, _URL, {})

    assert result == _OK_BODY
    assert len(httpx_mock.get_requests()) == 3


async def test_fails_after_max_retries(httpx_mock, monkeypatch):
    """MAX_RETRIES consecutive 5xx → RuntimeError."""
    monkeypatch.setattr("backtest.data_loader.asyncio", _fast_asyncio())

    for _ in range(_MAX_RETRIES):
        httpx_mock.add_response(status_code=503)

    with pytest.raises(RuntimeError, match="Failed after"):
        async with httpx.AsyncClient() as client:
            await _get_with_retry(client, _URL, {})

    assert len(httpx_mock.get_requests()) == _MAX_RETRIES


async def test_fail_fast_on_400(httpx_mock):
    """4xx client error is not retried — raises immediately."""
    httpx_mock.add_response(status_code=400)

    with pytest.raises(httpx.HTTPStatusError):
        async with httpx.AsyncClient() as client:
            await _get_with_retry(client, _URL, {})

    assert len(httpx_mock.get_requests()) == 1


# ── Helper ────────────────────────────────────────────────────────────────────

def _fast_asyncio():
    """Return a fake asyncio-like namespace where sleep is a no-op AsyncMock."""
    import types
    import asyncio as _asyncio

    fake = types.ModuleType("asyncio")
    # Copy everything from the real asyncio, then replace sleep
    fake.__dict__.update(_asyncio.__dict__)
    fake.sleep = AsyncMock()
    return fake
