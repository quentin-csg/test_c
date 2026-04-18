use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_millis() as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Market {
    #[default]
    Spot,
    UsdtPerpetual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderType {
    Limit,
    Market,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
    Expired,
}

/// A tick emitted by the market-data layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub market: Market,
    pub symbol: String,
    pub ts_ms: u64,
    pub best_bid: Decimal,
    pub best_bid_qty: Decimal,
    pub best_ask: Decimal,
    pub best_ask_qty: Decimal,
    /// Current funding rate (perpetual only, else 0).
    pub funding_rate: Decimal,
    /// Next funding timestamp in ms (perpetual only, else 0).
    pub next_funding_ms: u64,
    /// Mark price (perpetual) or last trade price (spot).
    pub mark_price: Decimal,
}

impl Tick {
    pub fn mid(&self) -> Decimal {
        (self.best_bid + self.best_ask) / Decimal::TWO
    }

    pub fn spread(&self) -> Decimal {
        self.best_ask - self.best_bid
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub client_order_id: String,
    pub exchange_order_id: Option<String>,
    pub market: Market,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub qty: Decimal,
    pub price: Option<Decimal>,
    pub status: OrderStatus,
    pub ts_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub exchange_order_id: String,
    pub client_order_id: String,
    pub market: Market,
    pub symbol: String,
    pub side: Side,
    pub qty: Decimal,
    pub price: Decimal,
    pub fee: Decimal,
    pub fee_asset: String,
    pub ts_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Position {
    pub market: Market,
    pub symbol: String,
    pub qty: Decimal,   // positive = long, negative = short
    pub avg_price: Decimal,
    pub unrealised_pnl: Decimal,
    pub ts_ms: u64,
}
