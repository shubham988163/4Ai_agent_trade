"""Unit tests for ORBMA200Strategy and Engine integration."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from trading.strategies.orb_ma200 import ORBMA200Strategy
from trading.strategy import Strategy, Signal, STRATEGIES, Engine
from trading.execution_router import ExecutionRouter
from trading.ledger import Ledger


def test_orb_ma200_signal_generation():
    ist = ZoneInfo("Asia/Kolkata")
    
    # Generate 35 trading days of 5-min bars (75 bars per day = ~2625 bars)
    # Target day: 2026-09-28
    base_dates = []
    curr_date = datetime(2026, 8, 10, 9, 15, tzinfo=ist)
    while len(base_dates) < 2500:
        if curr_date.weekday() < 5:  # Mon-Fri
            if "09:15" <= curr_date.strftime("%H:%M") <= "15:25":
                base_dates.append(curr_date)
        curr_date += timedelta(minutes=5)

    n_bars = len(base_dates)
    # Uptrend so that 1H and 30m 200 MAs are below current price
    prices = np.linspace(100.0, 200.0, n_bars)
    
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 2.0,
        "Low": prices - 2.0,
        "Close": prices,
        "Volume": 1000.0,
    }, index=pd.DatetimeIndex(base_dates))

    # Identify the last day
    last_day_date = df.index[-1].date()
    day_mask = df.index.date == last_day_date
    day_indices = np.where(day_mask)[0]

    # Structure Opening Range (first 3 bars: 09:15, 09:20, 09:25)
    # Range pct between 0.3% and 1.5%
    or_idx = day_indices[:3]
    df.iloc[or_idx, df.columns.get_loc("Open")] = 200.0
    df.iloc[or_idx, df.columns.get_loc("High")] = 201.0
    df.iloc[or_idx, df.columns.get_loc("Low")] = 198.5
    df.iloc[or_idx, df.columns.get_loc("Close")] = 200.2
    df.iloc[or_idx, df.columns.get_loc("Volume")] = 1000.0

    # Next bar (bar index 3: 09:30) closes above 201.0 with high volume -> Breakout trigger
    breakout_idx = day_indices[3]
    df.iloc[breakout_idx, df.columns.get_loc("Open")] = 200.5
    df.iloc[breakout_idx, df.columns.get_loc("High")] = 203.0
    df.iloc[breakout_idx, df.columns.get_loc("Low")] = 198.0
    df.iloc[breakout_idx, df.columns.get_loc("Close")] = 202.0  # Fresh close > 201.0
    df.iloc[breakout_idx, df.columns.get_loc("Volume")] = 5000.0  # > 1.5x avg 1000.0

    # Slice df up to breakout_idx
    eval_df = df.iloc[: breakout_idx + 1]

    strat = ORBMA200Strategy()
    sig = strat.evaluate(eval_df, "TESTSYM")

    assert sig is not None
    assert sig.symbol == "TESTSYM"
    assert sig.side == "BUY"
    assert sig.price == 202.0
    assert sig.strategy_id == "orb_ma200"
    assert sig.stop_loss < 202.0
    assert sig.target > 202.0

    # Test Engine integration with orb_ma200 using isolated test ledger
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_trading.db"
        ledger = Ledger(db_path=test_db)
        router = ExecutionRouter(mode="paper", ledger=ledger)
        engine = Engine(router=router, ledger=ledger, strategy="orb_ma200")
        assert engine.strat["id"] == "orb_ma200"
        
        # Enter trade
        engine.try_enter("TESTSYM", eval_df, ts=eval_df.index[-1].timestamp())
        assert "TESTSYM" in engine.open
        assert engine.open["TESTSYM"]["side"] == "BUY"


if __name__ == "__main__":
    test_orb_ma200_signal_generation()
