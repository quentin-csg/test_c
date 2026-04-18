pub mod execution;
pub mod market_data;
pub mod order_book;
pub mod types;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use tokio::sync::{broadcast, Mutex};

use crate::market_data::{MarketDataConfig, spawn_market_data};
use crate::types::{Market, Tick};

/// Python-facing wrapper around a `broadcast::Receiver<Tick>`.
///
/// Usage from Python:
///   receiver = await create_market_data_receiver("btcusdt", testnet=True)
///   async for tick in receiver:
///       print(tick)
#[pyclass]
struct MarketDataReceiver {
    rx: Arc<Mutex<broadcast::Receiver<Tick>>>,
}

#[pymethods]
impl MarketDataReceiver {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Clone the Arc (cheap) so the async block owns its own handle to the receiver.
        let rx = Arc::clone(&self.rx);
        future_into_py(py, async move {
            loop {
                let mut guard = rx.lock().await;
                match guard.recv().await {
                    Ok(tick) => return Python::with_gil(|py| tick_to_pydict(py, &tick)),
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("Python consumer lagged by {n} ticks");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        return Err(pyo3::exceptions::PyStopAsyncIteration::new_err("channel closed"))
                    }
                }
            }
        })
    }
}

fn tick_to_pydict(py: Python<'_>, tick: &Tick) -> PyResult<PyObject> {
    let d = pyo3::types::PyDict::new_bound(py);
    // Use static str for the enum variant — avoids a heap allocation on every tick.
    d.set_item("market", match tick.market {
        Market::Spot => "Spot",
        Market::UsdtPerpetual => "UsdtPerpetual",
    })?;
    d.set_item("symbol", &tick.symbol)?;
    d.set_item("ts_ms", tick.ts_ms)?;
    // Prices and quantities serialised as strings to preserve full Decimal precision.
    d.set_item("best_bid", tick.best_bid.to_string())?;
    d.set_item("best_bid_qty", tick.best_bid_qty.to_string())?;
    d.set_item("best_ask", tick.best_ask.to_string())?;
    d.set_item("best_ask_qty", tick.best_ask_qty.to_string())?;
    d.set_item("funding_rate", tick.funding_rate.to_string())?;
    d.set_item("next_funding_ms", tick.next_funding_ms)?;
    d.set_item("mark_price", tick.mark_price.to_string())?;
    Ok(d.into())
}

/// Create and start market-data WebSocket streams.
///
/// Returns an async iterable that yields tick dicts.
#[pyfunction]
#[pyo3(signature = (symbol, testnet=false, capacity=1024))]
fn create_market_data_receiver<'py>(
    py: Python<'py>,
    symbol: &str,
    testnet: bool,
    capacity: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let symbol = symbol.to_string();
    future_into_py(py, async move {
        let rx = spawn_market_data(MarketDataConfig { symbol, testnet }, capacity).await;
        let rx = Arc::new(Mutex::new(rx));
        Python::with_gil(|py| Ok(MarketDataReceiver { rx }.into_py(py)))
    })
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialise tracing once for the extension.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .try_init();

    m.add_class::<MarketDataReceiver>()?;
    m.add_function(wrap_pyfunction!(create_market_data_receiver, m)?)?;
    Ok(())
}
