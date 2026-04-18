# mn-bot — Market-Neutral BTC Cash-and-Carry Bot

Automated cash-and-carry trading bot for BTC on Binance. Captures perpetual funding rates by simultaneously holding a long spot position and a short perpetual futures position, staying delta-neutral while collecting funding payments every 8 hours.

**Status: paper trading ready — not yet validated for live trading.**

---

## How it works

When BTC perpetual funding is high (longs pay shorts), the bot:

1. **Enters**: buys spot BTC + shorts the same notional on BTCUSDT perp
2. **Holds**: collects funding payments every 8h (annualized target > 10% APR)
3. **Exits**: when funding drops below 3% APR or reverses

Net exposure to BTC price is near zero. Profit comes from the funding spread minus trading fees.

---

## Architecture

```
rust_core/          Rust crate compiled as Python extension (PyO3)
  market_data.rs    WebSocket streams: spot bookTicker + perp markPrice
  execution.rs      Binance REST API, HMAC-SHA256 signed orders
  order_book.rs     L2 order book, BTreeMap, VWAP
  types.rs          Shared types: Tick, Order, Market

python/
  bot/
    strategy.py     Entry/exit signals, Kelly position sizing
    risk.py         5 safety gates (stale data, HALT, delta, margin, reverse funding)
    orchestrator.py Main event loop, paper/live fills, PnL tracking
    config.py       Pydantic settings with validation
    cli.py          mn-bot CLI entry point
  backtest/
    data_loader.py  Binance klines + funding history download
    event_engine.py Event-driven backtest (shares strategy.py with live)
    vectorbt_runner.py Vectorized backtest for quick exploration

tests/python/       24 pytest tests, all green
```

---

## Requirements

- Python 3.10+
- Rust stable (1.75+) — for building the WebSocket/execution core
- Windows, macOS, or Linux

### Install Rust

```bash
# Windows
winget install Rustlang.Rustup

# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

---

## Setup

```bash
git clone https://github.com/your-username/mn-bot.git
cd mn-bot

# Create virtualenv and install Python dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install maturin

# Build the Rust extension and install the package
maturin develop --release

pip install -e "python/[dev]"
```

---

## Configuration

Create a `.env` file at the project root:

```env
# Required for live mode only — paper trading does not need API keys
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=false

BOT_MODE=paper

# Strategy thresholds (APR as decimal)
FUNDING_ENTRY_APR=0.07     # enter when funding > 7% APR
FUNDING_EXIT_APR=0.03      # exit when funding < 3% APR

# Risk
MAX_NOTIONAL_USDT=500      # max position size in USDT
MAX_DELTA_PCT=0.02          # max allowed imbalance between legs
MARGIN_BUFFER_MULT=3.0      # required free margin = maintenance * 3

LOG_LEVEL=INFO
```

---

## Usage

### Paper trading (no account needed)

Connects to live Binance WebSocket streams, simulates fills locally without placing any real orders.

```bash
mn-bot run --mode paper
```

### Backtest

Download historical data first (public Binance API, no auth required):

```bash
mn-bot download --start 2024-10-01 --end 2025-01-01
```

Run the event-driven backtest (canonical, simulates fees and funding payments):

```bash
mn-bot backtest --engine event
```

Quick vectorized exploration (no fees, approximate):

```bash
mn-bot backtest --engine vectorbt
```

### Live trading

Requires Binance API keys and testnet validation first.

```bash
mn-bot run --mode live
```

---

## Kill switch

Drop a `HALT` file at the project root to stop the bot after the current tick:

```bash
touch HALT      # bot stops immediately
rm HALT         # re-enables trading
```

---

## Risk gates

Every signal passes through 5 checks before execution:

| Gate | What it checks |
|---|---|
| Stale data | No tick received for > 5 seconds |
| HALT file | Presence of `./HALT` stops all orders |
| Delta neutrality | `\|long - short\| / equity < 2%` |
| Margin buffer | `free_margin > maintenance_margin × 3` |
| Reverse funding | APR < -2% forces immediate exit |

---

## Default parameters

| Parameter | Default | Description |
|---|---|---|
| `FUNDING_ENTRY_APR` | 0.07 | Enter above this APR |
| `FUNDING_EXIT_APR` | 0.03 | Exit below this APR |
| `MAX_NOTIONAL_USDT` | 500 | Max position size |
| `MAX_DELTA_PCT` | 0.02 | Max delta imbalance |
| `MARGIN_BUFFER_MULT` | 3.0 | Margin safety multiplier |
| `STALE_TICK_SECONDS` | 5 | Data freshness threshold |
| Kelly fraction | 0.5 | Position sizing conservatism |

---

## Development

```bash
# Run tests
pytest tests/python -v

# Rust tests
cargo test -p rust_core

# Lint
ruff check python/
cargo clippy --all-targets -- -D warnings

# Format
ruff format python/
cargo fmt
```

---

## Disclaimer

This software is for educational purposes. Trading perpetual futures carries significant risk of loss. The authors are not responsible for any financial losses incurred by using this software.
