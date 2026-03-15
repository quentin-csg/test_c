"""
Tests unitaires pour le module data (Phase 2 — Ingestion des données).
"""

import pandas as pd
import pytest


class TestCryptoFetcher:
    """Tests pour data/crypto_fetcher.py"""

    def test_import(self):
        """Le module s'importe correctement."""
        from data.crypto_fetcher import (
            fetch_ohlcv,
            fetch_multi_timeframe,
            fetch_funding_rate,
            fetch_order_book,
            fetch_open_interest,
        )
        assert callable(fetch_ohlcv)
        assert callable(fetch_multi_timeframe)
        assert callable(fetch_funding_rate)
        assert callable(fetch_order_book)
        assert callable(fetch_open_interest)

    def test_get_exchange(self):
        """L'exchange s'instancie correctement."""
        from data.crypto_fetcher import _get_exchange
        exchange = _get_exchange("binance")
        assert exchange is not None
        assert exchange.id == "binance"

    def test_fetch_ohlcv_returns_dataframe(self):
        """fetch_ohlcv retourne un DataFrame avec les bonnes colonnes."""
        from data.crypto_fetcher import fetch_ohlcv
        df = fetch_ohlcv("BTC/USDT", "1h", since="2024-01-01", until="2024-01-02")
        assert isinstance(df, pd.DataFrame)
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_cols
        if not df.empty:
            assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
            assert len(df) > 0

    def test_fetch_ohlcv_empty_on_invalid_symbol(self):
        """fetch_ohlcv retourne un DataFrame vide pour un symbol invalide."""
        from data.crypto_fetcher import fetch_ohlcv
        result = fetch_ohlcv("INVALID/PAIR", "1h", since="2024-01-01", until="2024-01-02")
        assert result.empty

    def test_order_book_structure(self):
        """fetch_order_book retourne un dict avec les clés attendues."""
        from data.crypto_fetcher import fetch_order_book
        result = fetch_order_book("BTC/USDT")
        assert isinstance(result, dict)
        assert "bid_volume" in result
        assert "ask_volume" in result
        assert "imbalance" in result
        assert "mid_price" in result
        assert -1 <= result["imbalance"] <= 1


class TestMacroFetcher:
    """Tests pour data/macro_fetcher.py"""

    def test_import(self):
        """Le module s'importe correctement."""
        from data.macro_fetcher import fetch_macro_data, fetch_macro_daily
        assert callable(fetch_macro_data)
        assert callable(fetch_macro_daily)

    def test_fetch_macro_daily_returns_dataframe(self):
        """fetch_macro_daily retourne un DataFrame."""
        from data.macro_fetcher import fetch_macro_daily
        df = fetch_macro_daily(["SPY"], start="2024-01-01", end="2024-01-15")
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "timestamp" in df.columns
            assert "spy_close" in df.columns
            assert "is_weekend" in df.columns

    def test_weekend_flag(self):
        """Le flag is_weekend est correctement calculé."""
        from data.macro_fetcher import fetch_macro_daily
        df = fetch_macro_daily(["SPY"], start="2024-01-01", end="2024-01-15")
        if not df.empty and "is_weekend" in df.columns:
            # Vérifier que le flag est 0 ou 1
            assert df["is_weekend"].isin([0, 1]).all()


class TestSentimentFetcher:
    """Tests pour data/sentiment_fetcher.py"""

    def test_import(self):
        """Le module s'importe correctement."""
        from data.sentiment_fetcher import (
            fetch_fear_greed_current,
            fetch_fear_greed_history,
        )
        assert callable(fetch_fear_greed_current)
        assert callable(fetch_fear_greed_history)

    def test_fear_greed_current_structure(self):
        """fetch_fear_greed_current retourne un dict avec les bonnes clés."""
        from data.sentiment_fetcher import fetch_fear_greed_current
        result = fetch_fear_greed_current()
        assert isinstance(result, dict)
        assert "value" in result
        assert "label" in result
        assert 0 <= result["value"] <= 100

    def test_fear_greed_history_returns_dataframe(self):
        """fetch_fear_greed_history retourne un DataFrame."""
        from data.sentiment_fetcher import fetch_fear_greed_history
        df = fetch_fear_greed_history(limit=7)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "timestamp" in df.columns
            assert "fear_greed_value" in df.columns
            assert "fear_greed_normalized" in df.columns
            # Vérifier la normalisation entre -1 et +1
            assert df["fear_greed_normalized"].between(-1, 1).all()


class TestNewsFetcher:
    """Tests pour data/news_fetcher.py"""

    def test_import(self):
        """Le module s'importe correctement."""
        from data.news_fetcher import fetch_news, fetch_news_titles_for_hour
        assert callable(fetch_news)
        assert callable(fetch_news_titles_for_hour)

    def test_fetch_news_returns_dataframe(self):
        """fetch_news retourne un DataFrame avec les bonnes colonnes."""
        from data.news_fetcher import fetch_news
        df = fetch_news(filter_by_keywords=False, max_articles=5)
        assert isinstance(df, pd.DataFrame)
        expected_cols = ["timestamp", "title", "source", "url", "has_keyword"]
        for col in expected_cols:
            assert col in df.columns

    def test_keyword_filtering(self):
        """Le filtrage par mots-clés fonctionne."""
        from data.news_fetcher import _matches_keywords
        assert _matches_keywords("Bitcoin price surges to new ATH")
        assert _matches_keywords("FED announces rate cut")
        assert _matches_keywords("New crypto ETF approved by SEC")
        assert not _matches_keywords("Weather forecast for tomorrow")


class TestPipeline:
    """Tests pour data/pipeline.py"""

    def test_import(self):
        """Le module s'importe correctement."""
        from data.pipeline import build_dataset, get_news_for_dataset
        assert callable(build_dataset)
        assert callable(get_news_for_dataset)

    def test_resample_daily_to_hourly(self):
        """Le resampling daily → hourly fonctionne correctement."""
        from data.pipeline import _resample_daily_to_hourly

        # Créer des données daily de test
        daily = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "value": [100, 200],
        })

        # Créer un index horaire
        hourly_index = pd.date_range(
            "2024-01-01", periods=48, freq="h", tz="UTC"
        )

        result = _resample_daily_to_hourly(daily, pd.Series(hourly_index))
        assert len(result) == 48
        # Les 24 premières heures doivent avoir la valeur du jour 1
        assert result.iloc[0]["value"] == 100
        # Les 24 dernières heures doivent avoir la valeur du jour 2
        assert result.iloc[24]["value"] == 200
