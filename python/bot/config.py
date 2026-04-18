from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotMode(str, Enum):
    backtest = "backtest"
    paper = "paper"
    live = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Exchange credentials
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    # Runtime mode
    bot_mode: BotMode = BotMode.paper

    # Strategy thresholds (APR as decimal: 0.10 = 10%)
    funding_entry_apr: Decimal = Decimal("0.10")
    funding_exit_apr: Decimal = Decimal("0.03")

    # Risk
    max_notional_usdt: Decimal = Decimal("500")
    margin_buffer_mult: Decimal = Decimal("3.0")
    max_delta_pct: Decimal = Decimal("0.02")
    stale_tick_seconds: int = Field(default=5, gt=0)

    # Execution
    recv_window_ms: int = Field(default=5000, gt=0, le=60000)
    backtest_slippage_pct: Decimal = Decimal("0.0005")

    # Strategy
    kelly_fraction: Decimal = Decimal("0.5")

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None  # if set, JSON logs are tee'd here (rotated at 50 MB)

    # State persistence: equity + position survives restarts
    state_file: Path | None = None  # e.g. Path("portfolio_state.json")

    @field_validator("binance_api_key", "binance_api_secret", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("max_notional_usdt", "margin_buffer_mult", "max_delta_pct")
    @classmethod
    def must_be_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("must be positive (> 0)")
        return v

    @model_validator(mode="after")
    def check_apr_ordering(self) -> "Settings":
        if self.funding_entry_apr <= self.funding_exit_apr:
            raise ValueError(
                f"funding_entry_apr ({self.funding_entry_apr}) must be greater than "
                f"funding_exit_apr ({self.funding_exit_apr})"
            )
        return self

    def require_credentials(self) -> None:
        if not self.binance_api_key or not self.binance_api_secret:
            raise RuntimeError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env"
            )


settings = Settings()
