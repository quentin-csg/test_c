use std::collections::BTreeMap;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Incremental L2 order book following the Binance depth-update protocol.
///
/// Invariants maintained:
///   - Bids sorted descending (best bid = last key of bids BTreeMap when iterated ascending).
///   - Asks sorted ascending (best ask = first key).
///   - Levels with qty == 0 are removed immediately.
///   - `last_update_id` is tracked to detect gaps and trigger a full snapshot refresh.
#[derive(Debug, Default)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: BTreeMap<Decimal, Decimal>, // price → qty
    pub asks: BTreeMap<Decimal, Decimal>,
    pub last_update_id: u64,
}

impl OrderBook {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self { symbol: symbol.into(), ..Default::default() }
    }

    /// Apply a full snapshot from the REST depth endpoint.
    pub fn apply_snapshot(&mut self, update_id: u64, bids: &[(Decimal, Decimal)], asks: &[(Decimal, Decimal)]) {
        self.bids.clear();
        self.asks.clear();
        for (p, q) in bids {
            if *q > dec!(0) { self.bids.insert(*p, *q); }
        }
        for (p, q) in asks {
            if *q > dec!(0) { self.asks.insert(*p, *q); }
        }
        self.last_update_id = update_id;
    }

    /// Apply an incremental depth update.
    /// Returns `Err` if the update is stale or creates a gap, signalling a snapshot refresh.
    pub fn apply_update(
        &mut self,
        first_update_id: u64,
        final_update_id: u64,
        bids: &[(Decimal, Decimal)],
        asks: &[(Decimal, Decimal)],
    ) -> Result<(), BookError> {
        if final_update_id <= self.last_update_id {
            return Err(BookError::StaleUpdate);
        }
        if first_update_id > self.last_update_id + 1 {
            return Err(BookError::Gap {
                expected: self.last_update_id + 1,
                got: first_update_id,
            });
        }
        for (p, q) in bids {
            if *q == dec!(0) { self.bids.remove(p); } else { self.bids.insert(*p, *q); }
        }
        for (p, q) in asks {
            if *q == dec!(0) { self.asks.remove(p); } else { self.asks.insert(*p, *q); }
        }
        self.last_update_id = final_update_id;
        Ok(())
    }

    pub fn best_bid(&self) -> Option<(Decimal, Decimal)> {
        self.bids.iter().next_back().map(|(p, q)| (*p, *q))
    }

    pub fn best_ask(&self) -> Option<(Decimal, Decimal)> {
        self.asks.iter().next().map(|(p, q)| (*p, *q))
    }

    pub fn mid(&self) -> Option<Decimal> {
        let (bp, _) = self.best_bid()?;
        let (ap, _) = self.best_ask()?;
        Some((bp + ap) / Decimal::TWO)
    }

    pub fn spread(&self) -> Option<Decimal> {
        let (bp, _) = self.best_bid()?;
        let (ap, _) = self.best_ask()?;
        Some(ap - bp)
    }

    /// Volume-weighted average price for a given notional side up to `depth` levels.
    pub fn vwap(&self, side: VwapSide, depth: usize) -> Option<Decimal> {
        let mut total_qty = dec!(0);
        let mut total_notional = dec!(0);
        match side {
            VwapSide::Bid => {
                for (p, q) in self.bids.iter().rev().take(depth) {
                    total_qty += q;
                    total_notional += p * q;
                }
            }
            VwapSide::Ask => {
                for (p, q) in self.asks.iter().take(depth) {
                    total_qty += q;
                    total_notional += p * q;
                }
            }
        }
        if total_qty == dec!(0) { None } else { Some(total_notional / total_qty) }
    }
}

pub enum VwapSide { Bid, Ask }

#[derive(Debug, thiserror::Error)]
pub enum BookError {
    #[error("stale update (already applied)")]
    StaleUpdate,
    #[error("gap in updates: expected first_update_id={expected}, got {got}")]
    Gap { expected: u64, got: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn book_with_snapshot() -> OrderBook {
        let mut book = OrderBook::new("BTCUSDT");
        book.apply_snapshot(
            100,
            &[(dec!(99000), dec!(1.5)), (dec!(98900), dec!(2.0))],
            &[(dec!(99100), dec!(1.0)), (dec!(99200), dec!(0.5))],
        );
        book
    }

    #[test]
    fn best_bid_ask() {
        let book = book_with_snapshot();
        assert_eq!(book.best_bid().unwrap().0, dec!(99000));
        assert_eq!(book.best_ask().unwrap().0, dec!(99100));
    }

    #[test]
    fn mid_spread() {
        let book = book_with_snapshot();
        assert_eq!(book.mid().unwrap(), dec!(99050));
        assert_eq!(book.spread().unwrap(), dec!(100));
    }

    #[test]
    fn apply_update_removes_zero() {
        let mut book = book_with_snapshot();
        book.apply_update(
            101, 101,
            &[(dec!(99000), dec!(0))],
            &[],
        ).unwrap();
        assert_eq!(book.best_bid().unwrap().0, dec!(98900));
    }

    #[test]
    fn gap_returns_error() {
        let mut book = book_with_snapshot();
        let res = book.apply_update(200, 201, &[], &[]);
        assert!(matches!(res, Err(BookError::Gap { .. })));
    }

    #[test]
    fn stale_update_rejected() {
        let mut book = book_with_snapshot(); // last_update_id = 100
        let res = book.apply_update(99, 100, &[], &[]); // final_update_id <= last_update_id
        assert!(matches!(res, Err(BookError::StaleUpdate)));
        assert_eq!(book.last_update_id, 100); // unchanged
    }

    #[test]
    fn vwap_bid_weighted() {
        let book = book_with_snapshot();
        // bids: 99000 × 1.5, 98900 × 2.0
        // vwap = (99000*1.5 + 98900*2.0) / (1.5 + 2.0) = (148500 + 197800) / 3.5 = 346300 / 3.5
        let expected = (dec!(99000) * dec!(1.5) + dec!(98900) * dec!(2.0))
            / (dec!(1.5) + dec!(2.0));
        assert_eq!(book.vwap(VwapSide::Bid, 5).unwrap(), expected);
    }

    #[test]
    fn vwap_ask_weighted() {
        let book = book_with_snapshot();
        // asks: 99100 × 1.0, 99200 × 0.5
        let expected = (dec!(99100) * dec!(1.0) + dec!(99200) * dec!(0.5))
            / (dec!(1.0) + dec!(0.5));
        assert_eq!(book.vwap(VwapSide::Ask, 5).unwrap(), expected);
    }

    #[test]
    fn vwap_empty_returns_none() {
        let book = OrderBook::new("BTCUSDT");
        assert!(book.vwap(VwapSide::Bid, 5).is_none());
        assert!(book.vwap(VwapSide::Ask, 5).is_none());
    }

    #[test]
    fn vwap_depth_limits_levels() {
        let book = book_with_snapshot();
        // depth=1 on bids: only best bid (99000 × 1.5) → vwap = 99000
        assert_eq!(book.vwap(VwapSide::Bid, 1).unwrap(), dec!(99000));
    }
}
