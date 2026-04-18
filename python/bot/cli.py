import asyncio
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(help="Market-neutral BTC cash-and-carry bot", add_completion=False)


@app.command()
def run(
    mode: str = typer.Option("paper", envvar="BOT_MODE", help="Execution mode: backtest | paper | live"),
) -> None:
    """Start the bot in paper or live mode."""
    from bot.config import BotMode, Settings
    from bot.logger import configure_logging
    from bot.orchestrator import Orchestrator

    configure_logging()  # early logging before Settings instantiation
    settings = Settings(bot_mode=BotMode(mode))
    configure_logging(settings.log_level)  # reconfigure with correct level from env

    if settings.bot_mode == BotMode.live:
        settings.require_credentials()

    orchestrator = Orchestrator(settings)
    asyncio.run(orchestrator.run())


@app.command()
def backtest(
    start: str = typer.Option("2023-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2024-01-01", help="End date YYYY-MM-DD"),
    engine: str = typer.Option("vectorbt", help="vectorbt | event"),
) -> None:
    """Run a backtest over historical data."""
    from bot.logger import configure_logging
    configure_logging()

    for date_str, name in [(start, "start"), (end, "end")]:
        try:
            datetime.fromisoformat(date_str)
        except ValueError:
            raise typer.BadParameter(
                f"{name} date must be YYYY-MM-DD, got: {date_str!r}", param_hint=f"--{name}"
            )

    if engine == "vectorbt":
        from backtest.vectorbt_runner import run_vectorbt
        run_vectorbt(start, end)
    else:
        from backtest.event_engine import run_event_backtest
        asyncio.run(run_event_backtest(start, end))


@app.command()
def download(
    start: str = typer.Option("2022-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2024-12-31", help="End date YYYY-MM-DD"),
    out_dir: Path = typer.Option(Path("data"), help="Output directory for Parquet files"),
) -> None:
    """Download historical klines and funding rates from Binance."""
    from bot.logger import configure_logging
    configure_logging()
    from backtest.data_loader import download_all
    asyncio.run(download_all(start, end, out_dir))


if __name__ == "__main__":
    app()
