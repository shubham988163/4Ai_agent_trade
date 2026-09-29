"""Transaction cost model for NSE intraday equity (Zerodha schedule).

Why this is not a flat percentage: brokerage is `min(0.03% of turnover, Rs20)`
per order, so cost as a fraction of position value is NOT constant. It is
~0.11% on a Rs 8,000 position and ~0.06% on a Rs 2,00,000 one. A flat-percent
model therefore misprices exactly the thing the strategy is most sensitive to
-- whether a given move clears its own costs -- and it mismodels it in
opposite directions at the two ends of the size range.

Measured against `CHARGES_PCT_ROUND_TRIP * (entry + exit) * qty` (the model in
strategy.py): ~1.13x too high at Rs 25k positions, ~2.04x too high at Rs 2L.

RATES ARE AS OF 2026 AND CHANGE. Verify against Zerodha's brokerage calculator
before trusting a live P&L projection: https://zerodha.com/brokerage-calculator
"""
from __future__ import annotations

from dataclasses import dataclass

# --- rate card: NSE equity intraday (MIS), Zerodha ---
BROKERAGE_PCT = 0.0003        # 0.03% of order turnover...
BROKERAGE_CAP = 20.0          # ...capped at Rs 20 per executed order
STT_PCT_SELL = 0.00025        # 0.025%, sell side only
EXCHANGE_TXN_PCT = 0.0000297  # 0.00297% of turnover, NSE, both sides
SEBI_PCT = 0.000001           # Rs 10 per crore, both sides
STAMP_PCT_BUY = 0.00003       # 0.003%, buy side only
GST_PCT = 0.18                # on brokerage + exchange txn + SEBI


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst

    def __str__(self) -> str:
        return (f"brokerage {self.brokerage:.2f} + STT {self.stt:.2f} + exch {self.exchange:.2f}"
                f" + SEBI {self.sebi:.2f} + stamp {self.stamp:.2f} + GST {self.gst:.2f}"
                f" = {self.total:.2f}")


def breakdown(entry: float, exit_price: float, qty: int) -> CostBreakdown:
    """Itemised round-trip cost for one intraday equity position.

    Direction-agnostic: STT lands on the sell leg and stamp duty on the buy
    leg whichever way round the trade was, so both legs are always priced.
    """
    buy_value, sell_value = entry * qty, exit_price * qty
    turnover = buy_value + sell_value
    brokerage = (min(BROKERAGE_PCT * buy_value, BROKERAGE_CAP)
                 + min(BROKERAGE_PCT * sell_value, BROKERAGE_CAP))
    exchange = EXCHANGE_TXN_PCT * turnover
    sebi = SEBI_PCT * turnover
    return CostBreakdown(
        brokerage=brokerage,
        stt=STT_PCT_SELL * sell_value,
        exchange=exchange,
        sebi=sebi,
        stamp=STAMP_PCT_BUY * buy_value,
        gst=GST_PCT * (brokerage + exchange + sebi),
    )


def round_trip(entry: float, exit_price: float, qty: int) -> float:
    """Total round-trip charges in INR."""
    return breakdown(entry, exit_price, qty).total


def breakeven_move(price: float, qty: int) -> float:
    """Points of favourable move needed just to cover costs.

    The single most useful number for a scalper: if the average winner is
    smaller than this, the strategy cannot be profitable at this size no
    matter how accurate the entries are.
    """
    if qty < 1 or price <= 0:
        return float("inf")
    return round_trip(price, price, qty) / qty


def breakeven_pct(price: float, qty: int) -> float:
    """`breakeven_move` as a fraction of price."""
    return breakeven_move(price, qty) / price if price > 0 else float("inf")


def compute_cost_risk_ratio(
    entry_price: float,
    stop_loss: float,
    qty: int = 1,
    slippage_pct: float | None = None,
) -> float:
    """Compute round-trip friction as a fraction of stop distance using exact broker round_trip charges.

    Friction = (entry_price * slippage_pct * 2 * qty) + round_trip(entry_price, entry_price, qty)
    Risk = abs(entry_price - stop_loss) * qty
    Returns: friction / risk
    """
    if entry_price <= 0 or qty <= 0:
        return float("inf")
    stop_dist = abs(entry_price - stop_loss)
    if stop_dist <= 0:
        return float("inf")

    from trading.config import SLIPPAGE_PCT
    slip = slippage_pct if slippage_pct is not None else SLIPPAGE_PCT
    slippage_cost = entry_price * slip * 2 * qty
    statutory_charges = round_trip(entry_price, entry_price, qty)
    total_friction = slippage_cost + statutory_charges
    total_risk = stop_dist * qty
    return total_friction / total_risk


def check_cost_floor(
    entry_price: float,
    stop_loss: float,
    qty: int = 1,
    max_ratio: float | None = None,
    slippage_pct: float | None = None,
) -> tuple[bool, float]:
    """Single shared gate check: is friction <= max_ratio of stop distance using round_trip() for actual quantity?

    Returns:
        (passed: bool, cost_risk_ratio: float)
    """
    from trading.config import MAX_COST_RISK_RATIO
    limit = max_ratio if max_ratio is not None else MAX_COST_RISK_RATIO
    ratio = compute_cost_risk_ratio(entry_price, stop_loss, qty=qty, slippage_pct=slippage_pct)
    return (ratio <= limit), ratio


if __name__ == "__main__":
    print(f"{'position':>12} {'qty':>5} {'charges':>9} {'as %':>7} {'breakeven move':>15}")
    for price, qty in [(1300, 6), (1300, 19), (1323, 151), (1300, 400)]:
        c = round_trip(price, price, qty)
        print(f"{price*qty:>12,.0f} {qty:>5} {c:>9.2f} {c/(price*qty):>7.3%}"
              f" {breakeven_move(price, qty):>13.2f} pts")
    print()
    print("sample:", breakdown(1323, 1323, 151))
