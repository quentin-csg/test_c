"""Cash-and-carry signal: Long spot BTCUSDT + Short perp BTCUSDT.

Signal logic:
  funding_apr = funding_rate * 3 * 365   (3 funding events per day on Binance)
  ENTER  : funding_apr > entry_threshold AND spread < max_spread_pct
  EXIT   : funding_apr < exit_threshold  OR  risk manager fires stop
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto

from bot.logger import log


class Signal(Enum):
    NONE = auto()
    ENTER = auto()
    EXIT = auto()


FUNDINGS_PER_DAY = Decimal("3")
DAYS_PER_YEAR = Decimal("365")


def compute_funding_apr(funding_rate: Decimal) -> Decimal:
    """Annualise a single Binance perp funding rate (3 payments/day)."""
    return funding_rate * FUNDINGS_PER_DAY * DAYS_PER_YEAR


@dataclass
class StrategyState:
    in_position: bool = False
    last_funding_apr: Decimal = Decimal("0")


@dataclass
class StrategyConfig:
    entry_apr: Decimal = Decimal("0.10")
    exit_apr: Decimal = Decimal("0.03")
    max_spread_pct: Decimal = Decimal("0.001")   # 0.1% max spread on perp
    kelly_fraction: Decimal = Decimal("0.5")


class CashCarryStrategy:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.state = StrategyState()

    def on_tick(
        self,
        spot_bid: Decimal,
        spot_ask: Decimal,
        perp_bid: Decimal,
        perp_ask: Decimal,
        funding_rate: Decimal,
    ) -> Signal:
        funding_apr = compute_funding_apr(funding_rate)
        self.state.last_funding_apr = funding_apr

        perp_mid = (perp_bid + perp_ask) / Decimal("2")
        if not perp_mid:
            log.warning("perp_mid_zero", perp_bid=str(perp_bid), perp_ask=str(perp_ask))
            return Signal.NONE
        spread_pct = (perp_ask - perp_bid) / perp_mid
        if spread_pct < 0:
            return Signal.NONE

        if not self.state.in_position:
            if funding_apr >= self.cfg.entry_apr and spread_pct <= self.cfg.max_spread_pct:
                self.state.in_position = True
                return Signal.ENTER
        else:
            if funding_apr <= self.cfg.exit_apr:
                self.state.in_position = False
                return Signal.EXIT

        return Signal.NONE

    def position_sizing(self, equity_usdt: Decimal, max_notional_usdt: Decimal) -> Decimal:
        """Return target notional in USDT (capped at max_notional_usdt)."""
        raw = equity_usdt * self.cfg.kelly_fraction
        return min(raw, max_notional_usdt)
