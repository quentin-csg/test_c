"""Main event loop: consumes ticks → strategy → risk → execution."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import TypedDict

from bot.config import BotMode, Settings
from bot.logger import log
from bot.risk import RiskError, RiskManager
from bot.strategy import CashCarryStrategy, Signal, StrategyConfig, compute_funding_apr

SPOT_TAKER_FEE = Decimal("0.001")      # 0.10% Binance spot taker
FUTURES_TAKER_FEE = Decimal("0.0005")  # 0.05% Binance futures taker


class MarketTick(TypedDict):
    market: str
    symbol: str
    ts_ms: int
    best_bid: str
    best_bid_qty: str
    best_ask: str
    best_ask_qty: str
    funding_rate: str
    next_funding_ms: int
    mark_price: str


@dataclass
class PortfolioState:
    spot_notional: Decimal = Decimal("0")    # long spot cost basis (USDT)
    perp_notional: Decimal = Decimal("0")    # short perp notional (USDT)
    spot_qty: Decimal = Decimal("0")         # BTC held long
    spot_entry_ask: Decimal = Decimal("0")   # spot ask price at entry
    perp_entry_bid: Decimal = Decimal("0")   # perp bid price at entry (short)
    equity: Decimal = Decimal("1000")        # starting paper equity
    free_margin: Decimal = Decimal("1000")
    maintenance_margin: Decimal = Decimal("0")


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.strategy = CashCarryStrategy(
            StrategyConfig(
                entry_apr=settings.funding_entry_apr,
                exit_apr=settings.funding_exit_apr,
            )
        )
        self.risk = RiskManager(
            max_delta_pct=settings.max_delta_pct,
            margin_buffer_mult=settings.margin_buffer_mult,
            stale_tick_seconds=settings.stale_tick_seconds,
        )
        self.portfolio = PortfolioState()
        self._spot_tick: MarketTick | None = None
        self._perp_tick: MarketTick | None = None

    async def run(self) -> None:
        try:
            from mn_bot._rust import create_market_data_receiver  # type: ignore[import]
            testnet = self.settings.binance_testnet
            receiver = await create_market_data_receiver("btcusdt", testnet=testnet)
        except Exception as exc:
            log.error("market_data_init_failed", error=str(exc))
            raise

        log.info("orchestrator_started", mode=self.settings.bot_mode.value, testnet=self.settings.binance_testnet)

        try:
            async for tick in receiver:
                self.risk.record_tick()

                if tick["market"] == "Spot":
                    self._spot_tick = tick
                else:
                    self._perp_tick = tick

                if self._spot_tick is None or self._perp_tick is None:
                    continue

                await self._on_both_ticks()
        except Exception as exc:
            log.error("orchestrator_loop_error", error=str(exc))
            raise

    async def _on_both_ticks(self) -> None:
        spot = self._spot_tick
        perp = self._perp_tick

        funding_rate = Decimal(perp["funding_rate"])
        funding_apr = compute_funding_apr(funding_rate)

        # Check reverse funding → force exit
        if self.risk.check_funding_floor(funding_apr):
            if not self.strategy.state.in_position:
                return
            log.warning(
                "reverse_funding_force_exit",
                funding_apr=str(funding_apr),
                floor=str(self.risk.exit_apr_floor),
            )
            try:
                await self._close_position(spot, perp, reason="reverse_funding")
            except Exception as exc:
                log.error("close_position_failed", reason="reverse_funding", error=str(exc))
            return

        # Run standard risk checks (delta computed from current mark prices)
        try:
            self.risk.pre_signal_checks(
                spot_qty=self.portfolio.spot_qty,
                perp_qty=self.portfolio.spot_qty,  # always equal in cash-and-carry
                spot_mark=Decimal(spot["mark_price"]),
                perp_mark=Decimal(perp["mark_price"]),
                equity=self.portfolio.equity,
                maintenance_margin=self.portfolio.maintenance_margin,
                free_margin=self.portfolio.free_margin,
                funding_apr=funding_apr,
            )
        except RiskError as e:
            log.warning("risk_check_blocked", reason=str(e))
            return

        signal = self.strategy.on_tick(
            spot_bid=Decimal(spot["best_bid"]),
            spot_ask=Decimal(spot["best_ask"]),
            perp_bid=Decimal(perp["best_bid"]),
            perp_ask=Decimal(perp["best_ask"]),
            funding_rate=funding_rate,
        )

        if signal is Signal.ENTER:
            try:
                await self._open_position(spot, perp)
            finally:
                self.strategy.state.in_position = self.portfolio.spot_qty > Decimal("0")
        elif signal is Signal.EXIT:
            try:
                await self._close_position(spot, perp, reason="signal")
            finally:
                self.strategy.state.in_position = self.portfolio.spot_qty > Decimal("0")

    async def _open_position(self, spot: MarketTick, perp: MarketTick) -> None:
        notional = self.strategy.position_sizing(
            self.portfolio.equity, self.settings.max_notional_usdt
        )
        if notional <= Decimal("0"):
            log.warning("open_position_skipped", reason="notional_zero")
            return

        spot_ask = Decimal(spot["best_ask"])
        qty = (notional / spot_ask).quantize(Decimal("0.00001"))

        log.info(
            "open_position",
            funding_apr=str(self.strategy.state.last_funding_apr),
            notional_usdt=str(notional),
            qty_btc=str(qty),
            spot_ask=str(spot_ask),
        )

        if self.settings.bot_mode == BotMode.paper:
            perp_bid = Decimal(perp["best_bid"])
            self.portfolio.spot_qty = qty
            self.portfolio.spot_entry_ask = spot_ask
            self.portfolio.perp_entry_bid = perp_bid
            self.portfolio.spot_notional = qty * spot_ask
            self.portfolio.perp_notional = qty * perp_bid
            fees = notional * (SPOT_TAKER_FEE + FUTURES_TAKER_FEE)
            self.portfolio.equity -= fees
            log.info("paper_fill_open", spot_notional=str(self.portfolio.spot_notional), fees=str(fees))
        else:
            # Live execution — delegate to Rust ExecutionClient (wired in later)
            log.info("live_order_not_yet_wired")

    async def _close_position(self, spot: MarketTick, perp: MarketTick, *, reason: str) -> None:
        log.info("close_position", reason=reason)

        if self.settings.bot_mode == BotMode.paper:
            spot_bid = Decimal(spot["best_bid"])
            perp_ask = Decimal(perp["best_ask"])
            # Long spot: bought at entry_ask, selling at close bid
            pnl_spot = self.portfolio.spot_qty * (spot_bid - self.portfolio.spot_entry_ask)
            # Short perp: sold at entry_bid, buying back at close ask
            pnl_perp = self.portfolio.spot_qty * (self.portfolio.perp_entry_bid - perp_ask)
            close_notional = self.portfolio.spot_qty * (spot_bid + perp_ask) / Decimal("2")
            fees = close_notional * (SPOT_TAKER_FEE + FUTURES_TAKER_FEE)
            pnl = pnl_spot + pnl_perp - fees
            self.portfolio.equity += pnl
            self.portfolio.spot_qty = Decimal("0")
            self.portfolio.spot_entry_ask = Decimal("0")
            self.portfolio.perp_entry_bid = Decimal("0")
            self.portfolio.spot_notional = Decimal("0")
            self.portfolio.perp_notional = Decimal("0")
            log.info("paper_fill_close", pnl=str(pnl), equity=str(self.portfolio.equity))
        else:
            log.info("live_order_not_yet_wired")
