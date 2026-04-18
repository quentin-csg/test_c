use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::broadcast;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::types::{Market, Tick};

const SPOT_WS_BASE: &str = "wss://stream.binance.com:9443/stream?streams=";
const FUTURES_WS_BASE: &str = "wss://fstream.binance.com/stream?streams=";

const SPOT_WS_TESTNET: &str = "wss://testnet.binance.vision/stream?streams=";
const FUTURES_WS_TESTNET: &str = "wss://stream.binancefuture.com/stream?streams=";

const MAX_BACKOFF: Duration = Duration::from_secs(30);

pub struct MarketDataConfig {
    pub symbol: String,    // e.g. "btcusdt"
    pub testnet: bool,
}

/// Spawn both spot and futures WebSocket streams and merge ticks onto a single channel.
///
/// Returns a broadcast receiver of Tick. The sender is kept alive internally
/// inside the spawned tasks.
pub async fn spawn_market_data(
    cfg: MarketDataConfig,
    capacity: usize,
) -> broadcast::Receiver<Tick> {
    let (tx, rx) = broadcast::channel::<Tick>(capacity);

    let sym_lower = cfg.symbol.to_lowercase();
    // Uppercase computed once here; passed to tasks to avoid per-tick allocation.
    let sym_upper = cfg.symbol.to_uppercase();

    // Spot: bookTicker gives best bid/ask with negligible overhead.
    let spot_streams = format!("{}@bookTicker", sym_lower);
    let spot_base = if cfg.testnet { SPOT_WS_TESTNET } else { SPOT_WS_BASE };
    let spot_url = format!("{}{}", spot_base, spot_streams);
    tokio::spawn(run_spot_stream(spot_url, sym_upper.clone(), tx.clone()));

    // Futures: combined stream of markPrice (has funding) + bookTicker.
    let futures_streams = format!("{}@markPrice@1s/{}@bookTicker", sym_lower, sym_lower);
    let futures_base = if cfg.testnet { FUTURES_WS_TESTNET } else { FUTURES_WS_BASE };
    let futures_url = format!("{}{}", futures_base, futures_streams);
    tokio::spawn(run_futures_stream(futures_url, sym_upper, tx.clone()));

    rx
}

// ── Spot ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct WsEnvelope<T> {
    data: T,
}

#[derive(Debug, Deserialize)]
struct BookTickerMsg {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "b")]
    best_bid: Decimal,
    #[serde(rename = "B")]
    best_bid_qty: Decimal,
    #[serde(rename = "a")]
    best_ask: Decimal,
    #[serde(rename = "A")]
    best_ask_qty: Decimal,
}

async fn run_spot_stream(url: String, symbol: String, tx: broadcast::Sender<Tick>) {
    let mut backoff = Duration::from_secs(1);
    loop {
        match connect_and_stream_spot(&url, &symbol, &tx).await {
            Ok(()) => {
                tracing::info!(market = "spot", "ws_reconnect: closed cleanly");
                backoff = Duration::from_secs(1);
            }
            Err(e) => {
                tracing::warn!(market = "spot", backoff_ms = backoff.as_millis() as u64,
                               "ws_reconnect: {e:#}");
                // Jitter: mix low bits of current ms into the delay to avoid
                // thundering-herd when multiple tasks reconnect simultaneously.
                let jitter_ms = crate::types::now_ms() % 500;
                sleep(backoff + Duration::from_millis(jitter_ms)).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
                continue;
            }
        }
        sleep(Duration::from_secs(1)).await;
    }
}

async fn connect_and_stream_spot(
    url: &str,
    symbol: &str,
    tx: &broadcast::Sender<Tick>,
) -> Result<()> {
    let (mut ws, _) = connect_async(url).await.context("spot WS connect")?;
    tracing::info!("spot WS connected: {url}");

    while let Some(msg) = ws.next().await {
        let msg = msg.context("spot WS recv")?;
        if let Message::Text(text) = msg {
            if let Ok(env) = serde_json::from_str::<WsEnvelope<BookTickerMsg>>(&text) {
                let m = env.data;
                let tick = Tick {
                    market: Market::Spot,
                    symbol: symbol.to_string(),
                    ts_ms: crate::types::now_ms(),
                    best_bid: m.best_bid,
                    best_bid_qty: m.best_bid_qty,
                    best_ask: m.best_ask,
                    best_ask_qty: m.best_ask_qty,
                    funding_rate: Decimal::ZERO,
                    next_funding_ms: 0,
                    mark_price: (m.best_bid + m.best_ask) / Decimal::TWO,
                };
                let _ = tx.send(tick);
            }
        } else if let Message::Ping(data) = msg {
            ws.send(Message::Pong(data)).await?;
        } else if let Message::Close(_) = msg {
            break;
        }
    }
    Ok(())
}

// ── Futures ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct MarkPriceMsg {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "p")]
    mark_price: Decimal,
    #[serde(rename = "r")]
    funding_rate: Decimal,
    #[serde(rename = "T")]
    next_funding_ms: u64,
}

/// Untagged enum: serde tries MarkPrice first (has field `r`), then BookTicker (has field `b`).
/// Single parse — no serde_json::Value intermediate.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FuturesMsg {
    MarkPrice(MarkPriceMsg),
    BookTicker(BookTickerMsg),
}

#[derive(Default)]
struct FuturesState {
    mark_price: Decimal,
    funding_rate: Decimal,
    next_funding_ms: u64,
}

async fn run_futures_stream(url: String, symbol: String, tx: broadcast::Sender<Tick>) {
    let mut backoff = Duration::from_secs(1);
    loop {
        match connect_and_stream_futures(&url, &symbol, &tx).await {
            Ok(()) => {
                tracing::info!(market = "futures", "ws_reconnect: closed cleanly");
                backoff = Duration::from_secs(1);
            }
            Err(e) => {
                tracing::warn!(market = "futures", backoff_ms = backoff.as_millis() as u64,
                               "ws_reconnect: {e:#}");
                let jitter_ms = crate::types::now_ms() % 500;
                sleep(backoff + Duration::from_millis(jitter_ms)).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
                continue;
            }
        }
        sleep(Duration::from_secs(1)).await;
    }
}

async fn connect_and_stream_futures(
    url: &str,
    symbol: &str,
    tx: &broadcast::Sender<Tick>,
) -> Result<()> {
    let (mut ws, _) = connect_async(url).await.context("futures WS connect")?;
    tracing::info!("futures WS connected: {url}");

    let mut state = FuturesState::default();

    while let Some(msg) = ws.next().await {
        let msg = msg.context("futures WS recv")?;
        if let Message::Text(text) = msg {
            // Single parse via untagged enum — no serde_json::Value intermediate.
            if let Ok(env) = serde_json::from_str::<WsEnvelope<FuturesMsg>>(&text) {
                match env.data {
                    FuturesMsg::MarkPrice(m) => {
                        state.mark_price = m.mark_price;
                        state.funding_rate = m.funding_rate;
                        state.next_funding_ms = m.next_funding_ms;
                    }
                    FuturesMsg::BookTicker(m) => {
                        let tick = Tick {
                            market: Market::UsdtPerpetual,
                            symbol: symbol.to_string(),
                            ts_ms: crate::types::now_ms(),
                            best_bid: m.best_bid,
                            best_bid_qty: m.best_bid_qty,
                            best_ask: m.best_ask,
                            best_ask_qty: m.best_ask_qty,
                            funding_rate: state.funding_rate,
                            next_funding_ms: state.next_funding_ms,
                            mark_price: state.mark_price,
                        };
                        let _ = tx.send(tick);
                    }
                }
            }
        } else if let Message::Ping(data) = msg {
            ws.send(Message::Pong(data)).await?;
        } else if let Message::Close(_) = msg {
            break;
        }
    }
    Ok(())
}
