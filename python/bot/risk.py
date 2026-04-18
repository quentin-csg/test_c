"""Risk manager — gates every signal before it reaches the execution layer.

Checks (in order):
  1. Stale data guard           — no tick for > N seconds → HALT signals.
  2. Delta neutrality           — |notional_long - notional_short| / equity < threshold.
  3. Margin buffer              — futures maintenance margin * buffer_mult < free margin.
  4. Reverse funding            — if APR turns deeply negative, exit immediately.
  5. Kill-switch file           — presence of ./HALT file blocks all new orders.
"""

import time
from decimal import Decimal
from pathlib import Path

from bot.logger import log


HALT_FILE = Path(__file__).resolve().parents[2] / "HALT"


class RiskError(Exception):
    """Raised when a check blocks an action."""


class RiskManager:
    def __init__(
        self,
        max_delta_pct: Decimal,
        margin_buffer_mult: Decimal,
        stale_tick_seconds: int,
        exit_apr_floor: Decimal = Decimal("-0.02"),  # -2% APR → emergency exit
    ) -> None:
        self.max_delta_pct = max_delta_pct
        self.margin_buffer_mult = margin_buffer_mult
        self.stale_tick_seconds = stale_tick_seconds
        self.exit_apr_floor = exit_apr_floor

        self._last_tick_ts: float = time.monotonic()
        self._first_tick_received: bool = False

    # ── Called by market-data layer ──────────────────────────────────────────

    def record_tick(self) -> None:
        self._last_tick_ts = time.monotonic()
        self._first_tick_received = True

    # ── Gate checks (raise RiskError to block) ───────────────────────────────

    def check_stale(self) -> None:
        if not self._first_tick_received:
            return
        age = time.monotonic() - self._last_tick_ts
        if age > self.stale_tick_seconds:
            raise RiskError(f"stale data: last tick {age:.1f}s ago")

    def check_kill_switch(self) -> None:
        if HALT_FILE.exists():
            raise RiskError("HALT file present — all trading suspended")

    def check_delta(
        self,
        spot_qty: Decimal,
        perp_qty: Decimal,
        spot_mark: Decimal,
        perp_mark: Decimal,
        equity: Decimal,
    ) -> None:
        if equity <= 0:
            raise RiskError("equity is zero or negative — cannot compute delta")
        delta_notional = abs(spot_qty * spot_mark - perp_qty * perp_mark)
        delta_pct = delta_notional / equity
        if delta_pct > self.max_delta_pct:
            raise RiskError(
                f"delta imbalance {delta_pct:.2%} > limit {self.max_delta_pct:.2%}"
            )

    def check_margin(
        self,
        maintenance_margin: Decimal,
        free_margin: Decimal,
    ) -> None:
        required = maintenance_margin * self.margin_buffer_mult
        if free_margin < required:
            raise RiskError(
                f"insufficient margin buffer: free={free_margin:.2f}, "
                f"required={required:.2f} ({self.margin_buffer_mult}× maint.)"
            )

    def check_funding_floor(self, funding_apr: Decimal) -> bool:
        """Return True if funding is so negative we must exit immediately."""
        return funding_apr < self.exit_apr_floor

    def pre_signal_checks(
        self,
        spot_qty: Decimal,
        perp_qty: Decimal,
        spot_mark: Decimal,
        perp_mark: Decimal,
        equity: Decimal,
        maintenance_margin: Decimal,
        free_margin: Decimal,
        funding_apr: Decimal,
    ) -> None:
        """Run all pre-signal checks. Raises RiskError on any failure."""
        self.check_kill_switch()
        self.check_stale()
        self.check_delta(spot_qty, perp_qty, spot_mark, perp_mark, equity)
        self.check_margin(maintenance_margin, free_margin)
        # Reverse funding is handled separately by the orchestrator (triggers EXIT).
