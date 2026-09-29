"""End-to-end demo of the pipeline with fake prices — no market data, no API key
needed for the ledger/router part.

Runs: signal -> risk kernel -> paper fill -> ledger -> exit -> P&L,
plus one deliberately oversized signal to show the risk kernel rejecting it.
If ANTHROPIC_API_KEY is set, also runs the supervisor over the new trades.

Usage:  python -m trading.demo
"""
from __future__ import annotations

import random
import time

from trading.costs import round_trip as round_trip_charges
from trading.execution_router import ExecutionRouter
from trading.ledger import Ledger

FAKE_LTP = {"RELIANCE": 2950.0, "INFY": 1890.0, "SBIN": 815.0}


def get_ltp(symbol: str) -> float:
    return FAKE_LTP[symbol] * (1 + random.uniform(-0.001, 0.001))


def main():
    ledger = Ledger()
    router = ExecutionRouter(mode="paper", ledger=ledger, get_ltp=get_ltp)
    print(f"day config: {router.day_config}\n")

    # 1. Normal signal -> should fill
    signal = {"symbol": "RELIANCE", "side": "BUY", "qty": 5,
              "price": get_ltp("RELIANCE"), "ts": time.time(),
              "stop_loss": 2935.0, "target": 2975.0,
              "strategy_id": "ema_x_v3", "regime": router.day_config["regime"],
              "trend_state": "up"}
    trade_id = router.execute(signal)
    print(f"signal 1 (RELIANCE BUY x5): trade_id={trade_id}")

    # 2. Oversized signal -> risk kernel must reject
    big = dict(signal, symbol="INFY", qty=500, price=get_ltp("INFY"))
    rejected = router.execute(big)
    print(f"signal 2 (INFY BUY x500, oversized): trade_id={rejected} (expected None)")

    # 3. Close trade 1 with a simulated exit
    if trade_id:
        exit_price = get_ltp("RELIANCE") * 1.004  # pretend it went our way
        charges = round_trip_charges(signal["price"], exit_price, signal["qty"])
        pnl = ledger.record_exit(trade_id, exit_price, charges=charges,
                                 mae=-4.5, mfe=13.2)
        print(f"closed trade {trade_id}: net pnl = {pnl:.2f} INR")

    print(f"\nday realized P&L: {ledger.day_realized_pnl():.2f} INR")
    print(f"trades today: {len(ledger.trades_for_date())}")

    # 4. Supervisor agent (only if credentials available)
    from trading.agents.llm import credentials_available, provider
    if credentials_available():
        print(f"\nrunning supervisor agent ({provider()}) over unreviewed trades...")
        from trading.agents.supervisor import run as supervise
        supervise(loop=False)
    else:
        print("\nno LLM credentials (GEMINI_API_KEY / ANTHROPIC_API_KEY) — skipping supervisor demo")


if __name__ == "__main__":
    main()
