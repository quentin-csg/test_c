use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use hmac::{Hmac, Mac};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::types::{Market, Order, OrderStatus, OrderType, Side};

type HmacSha256 = Hmac<Sha256>;

const SPOT_BASE: &str = "https://api.binance.com";
const FUTURES_BASE: &str = "https://fapi.binance.com";
const SPOT_TESTNET: &str = "https://testnet.binance.vision";
const FUTURES_TESTNET: &str = "https://testnet.binancefuture.com";

const MAX_RETRIES: u32 = 3;

pub struct ExecutionConfig {
    pub api_key: String,
    pub api_secret: String,   // plain String in public API; wrapped internally
    pub testnet: bool,
    pub recv_window_ms: u64,
}

pub struct ExecutionClient {
    http: Client,
    api_key: String,
    api_secret: Zeroizing<String>,  // zeroed in memory on drop
    spot_base: String,
    futures_base: String,
    recv_window_ms: u64,
}

impl ExecutionClient {
    pub fn new(cfg: ExecutionConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_millis(5_000))
            .connection_verbose(false)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .context("build HTTP client")?;

        Ok(Self {
            http,
            api_key: cfg.api_key,
            api_secret: Zeroizing::new(cfg.api_secret),
            spot_base: if cfg.testnet { SPOT_TESTNET } else { SPOT_BASE }.to_string(),
            futures_base: if cfg.testnet { FUTURES_TESTNET } else { FUTURES_BASE }.to_string(),
            recv_window_ms: cfg.recv_window_ms,
        })
    }

    /// Sign a query string with HMAC-SHA256.
    fn sign(&self, query: &str) -> Result<String> {
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .map_err(|e| anyhow!("HMAC init failed: {e}"))?;
        mac.update(query.as_bytes());
        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    fn ts_query(&self, timestamp_ms: u64) -> String {
        format!("timestamp={timestamp_ms}&recvWindow={}", self.recv_window_ms)
    }

    pub async fn spot_account_info(&self, ts_ms: u64) -> Result<serde_json::Value> {
        let qs = self.ts_query(ts_ms);
        let sig = self.sign(&qs)?;
        let url = format!("{}/api/v3/account?{}&signature={}", self.spot_base, qs, sig);
        let resp = send_with_retry(|| {
            self.http.get(&url).header("X-MBX-APIKEY", &self.api_key)
        }).await?;
        check_status(&resp)?;
        Ok(resp.json().await?)
    }

    /// GET /fapi/v2/account — futures wallet balance and positions.
    /// Use to refresh equity in PortfolioState (live mode).
    pub async fn futures_account_info(&self, ts_ms: u64) -> Result<serde_json::Value> {
        let qs = self.ts_query(ts_ms);
        let sig = self.sign(&qs)?;
        let url = format!("{}/fapi/v2/account?{}&signature={}", self.futures_base, qs, sig);
        let resp = send_with_retry(|| {
            self.http.get(&url).header("X-MBX-APIKEY", &self.api_key)
        }).await?;
        check_status(&resp)?;
        Ok(resp.json().await?)
    }

    pub async fn place_order(
        &self,
        ts_ms: u64,
        market: Market,
        symbol: &str,
        side: Side,
        order_type: OrderType,
        qty: Decimal,
        price: Option<Decimal>,
        client_order_id: &str,
    ) -> Result<Order> {
        let side_str = match side { Side::Buy => "BUY", Side::Sell => "SELL" };
        let type_str = match order_type { OrderType::Limit => "LIMIT", OrderType::Market => "MARKET" };

        let (base_url, path) = match market {
            Market::Spot => (&self.spot_base, "/api/v3/order"),
            Market::UsdtPerpetual => (&self.futures_base, "/fapi/v1/order"),
        };

        let mut qs = format!(
            "symbol={symbol}&side={side_str}&type={type_str}&quantity={qty}&newClientOrderId={client_order_id}&{}",
            self.ts_query(ts_ms)
        );
        if let Some(p) = price {
            qs.push_str(&format!("&price={p}&timeInForce=GTC"));
        }
        let sig = self.sign(&qs)?;
        let url = format!("{base_url}{path}?{qs}&signature={sig}");

        let resp = send_with_retry(|| {
            self.http.post(&url).header("X-MBX-APIKEY", &self.api_key)
        }).await?;
        check_status(&resp)?;

        let body: PlaceOrderResponse = resp.json().await?;
        Ok(Order {
            client_order_id: client_order_id.to_string(),
            exchange_order_id: Some(body.order_id.to_string()),
            market,
            symbol: symbol.to_uppercase(),
            side,
            order_type,
            qty,
            price,
            status: parse_status(&body.status),
            ts_ms,
        })
    }

    pub async fn cancel_order(
        &self,
        ts_ms: u64,
        market: Market,
        symbol: &str,
        exchange_order_id: &str,
    ) -> Result<()> {
        let (base_url, path) = match market {
            Market::Spot => (&self.spot_base, "/api/v3/order"),
            Market::UsdtPerpetual => (&self.futures_base, "/fapi/v1/order"),
        };
        let qs = format!("symbol={symbol}&orderId={exchange_order_id}&{}", self.ts_query(ts_ms));
        let sig = self.sign(&qs)?;
        let url = format!("{base_url}{path}?{qs}&signature={sig}");
        let resp = send_with_retry(|| {
            self.http.delete(&url).header("X-MBX-APIKEY", &self.api_key)
        }).await?;
        check_status(&resp)?;
        Ok(())
    }

    pub async fn cancel_all_orders(&self, ts_ms: u64, market: Market, symbol: &str) -> Result<()> {
        let (base_url, path) = match market {
            Market::Spot => (&self.spot_base, "/api/v3/openOrders"),
            Market::UsdtPerpetual => (&self.futures_base, "/fapi/v1/allOpenOrders"),
        };
        let qs = format!("symbol={symbol}&{}", self.ts_query(ts_ms));
        let sig = self.sign(&qs)?;
        let url = format!("{base_url}{path}?{qs}&signature={sig}");
        let resp = send_with_retry(|| {
            self.http.delete(&url).header("X-MBX-APIKEY", &self.api_key)
        }).await?;
        check_status(&resp)?;
        Ok(())
    }

    pub async fn internal_transfer(
        &self,
        ts_ms: u64,
        asset: &str,
        amount: Decimal,
        to_futures: bool,
    ) -> Result<()> {
        let transfer_type = if to_futures { 1u8 } else { 2u8 };
        let qs = format!(
            "asset={asset}&amount={amount}&type={transfer_type}&{}",
            self.ts_query(ts_ms)
        );
        let sig = self.sign(&qs)?;
        let url = format!("{}/sapi/v1/futures/transfer?{qs}&signature={sig}", self.spot_base);
        let resp = send_with_retry(|| {
            self.http.post(&url).header("X-MBX-APIKEY", &self.api_key)
        }).await?;
        check_status(&resp)?;
        Ok(())
    }

    #[cfg(test)]
    pub fn new_for_test(api_key: &str, api_secret: &str, spot_base: String, futures_base: String) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_millis(5_000))
            .build()
            .expect("test http client");
        Self {
            http,
            api_key: api_key.to_string(),
            api_secret: Zeroizing::new(api_secret.to_string()),
            spot_base,
            futures_base,
            recv_window_ms: 5000,
        }
    }
}

/// Retry transient errors (network failures, 5xx). Fail fast on 4xx.
async fn send_with_retry(
    make_req: impl Fn() -> reqwest::RequestBuilder,
) -> Result<reqwest::Response> {
    let mut delay = Duration::from_millis(500);
    for attempt in 1..=MAX_RETRIES {
        match make_req().send().await {
            Ok(resp) if resp.status().is_server_error() => {
                if attempt == MAX_RETRIES {
                    anyhow::bail!("server error {} after {} attempts", resp.status(), MAX_RETRIES);
                }
                tracing::warn!(
                    "HTTP {} on attempt {}/{}, retrying in {:?}",
                    resp.status(), attempt, MAX_RETRIES, delay
                );
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(10));
            }
            Ok(resp) => return Ok(resp),
            Err(e) if is_transient(&e) => {
                if attempt == MAX_RETRIES {
                    return Err(e.into());
                }
                tracing::warn!(
                    "network error on attempt {}/{}: {}, retrying in {:?}",
                    attempt, MAX_RETRIES, e, delay
                );
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(10));
            }
            Err(e) => return Err(e.into()),
        }
    }
    unreachable!()
}

fn is_transient(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect() || e.is_request()
}

fn check_status(resp: &reqwest::Response) -> Result<()> {
    let status = resp.status();
    if status.is_client_error() {
        anyhow::bail!("Binance client error {status}");
    }
    Ok(())
}

fn parse_status(s: &str) -> OrderStatus {
    match s {
        "NEW" => OrderStatus::New,
        "PARTIALLY_FILLED" => OrderStatus::PartiallyFilled,
        "FILLED" => OrderStatus::Filled,
        "CANCELED" => OrderStatus::Canceled,
        "REJECTED" => OrderStatus::Rejected,
        "EXPIRED" => OrderStatus::Expired,
        _ => {
            tracing::warn!("unknown order status: {s:?}, treating as Expired");
            OrderStatus::Expired
        }
    }
}

#[derive(Deserialize)]
struct PlaceOrderResponse {
    #[serde(rename = "orderId")]
    order_id: u64,
    status: String,
}
