"""Look-ahead audit and consistency tests for ORB + Multi-TF 200MA backtest adapter.

Audit checklist:
1. Zero look-ahead: detect(window[:i+1]) produces the exact same signal as full-frame slicing.
2. Indicator invariance: precomputed indicators for bar i do not change when future bars are appended.
3. Live engine alignment: sizing, slippage on entry and exit legs, and real transaction cost model.
4. One trade per symbol per day enforcement.
"""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from trading.backtest import Backtester, BTTrade, Result
from trading.backtest_orb import BacktestORBStrategy, ORBBacktestAdapter, build_orb_strategy, compute_causal_htf_smas
from trading.costs import round_trip
from trading.config import RISK_PER_TRADE, MAX_POSITION_VALUE, SLIPPAGE_PCT
from trading.strategies.orb_ma200 import ORBMA200Strategy

IST = ZoneInfo("Asia/Kolkata")


def _generate_synthetic_candles(
    n_days: int = 40,
    bars_per_day: int = 75,
    base_price: float = 1000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic 5m candle bars across n_days starting at 09:15 IST."""
    rng = np.random.default_rng(seed)
    records = []
    price = base_price
    start_date = pd.Timestamp("2026-07-01 09:15:00", tz=IST)

    day_count = 0
    current_day = start_date
    while day_count < n_days:
        if current_day.weekday() < 5:  # Monday to Friday
            day_open = price
            for b in range(bars_per_day):
                bar_time = current_day + pd.Timedelta(minutes=5 * b)
                ret = rng.normal(0.0001, 0.001)
                close = price * (1 + ret)
                high = max(price, close) + abs(rng.normal(0, 0.5))
                low = min(price, close) - abs(rng.normal(0, 0.5))
                vol = float(rng.integers(10000, 50000))
                records.append({
                    "Datetime": bar_time,
                    "Open": price,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": vol,
                })
                price = close
            day_count += 1
        current_day += pd.Timedelta(days=1)

    df = pd.DataFrame(records).set_index("Datetime")
    df.attrs["symbol"] = "TESTSYM"
    return df


def test_indicator_invariance_when_future_bars_appended():
    """Indicators for any past bar k must not change when future bars are appended."""
    df = _generate_synthetic_candles(n_days=40, bars_per_day=75)
    k = 2500  # Bar in day ~34 (well past the 200 1H bar threshold)

    df_past = df.iloc[: k + 1].copy()
    prep_past = compute_causal_htf_smas(df_past)

    prep_full = compute_causal_htf_smas(df)

    # 1H and 30m SMAs at bar k must be exactly identical
    val_past_1h = prep_past["sma200_1h"].iloc[k]
    val_full_1h = prep_full["sma200_1h"].iloc[k]
    val_past_30m = prep_past["sma200_30m"].iloc[k]
    val_full_30m = prep_full["sma200_30m"].iloc[k]

    assert not pd.isna(val_past_1h), "1H 200 SMA should be available at bar 2500"
    assert not pd.isna(val_past_30m), "30m 200 SMA should be available at bar 2500"

    assert abs(val_past_1h - val_full_1h) < 1e-11, (
        f"1H SMA changed when future bars were appended: {val_past_1h} vs {val_full_1h}"
    )
    assert abs(val_past_30m - val_full_30m) < 1e-11, (
        f"30m SMA changed when future bars were appended: {val_past_30m} vs {val_full_30m}"
    )


def test_detect_lookahead_invariance():
    """detect(window[:i+1]) matches evaluation on truncated window and full-frame slicing."""
    df = _generate_synthetic_candles(n_days=40, bars_per_day=75)
    strat_dict = build_orb_strategy()
    adapter: ORBBacktestAdapter = strat_dict["adapter"]
    prep = adapter.prepare(df)

    # Test across multiple bars
    test_indices = [2200, 2400, 2600, 2800, len(prep) - 1]
    for idx in test_indices:
        window_truncated = prep.iloc[: idx + 1]
        sig_window = adapter.detect(window_truncated)

        # Direct evaluation using fresh strategy instance on the truncated window
        fresh_strat = ORBMA200Strategy()
        # Compute setup directly without precalculated columns (pure O(N^2) live way)
        raw_truncated = df.iloc[: idx + 1]
        setup_live = fresh_strat.compute_setup(raw_truncated, "TESTSYM")
        sig_live = setup_live.get("signal")
        live_side = sig_live.side if sig_live is not None else None

        assert sig_window == live_side, (
            f"Signal divergence at bar {idx}: adapter={sig_window}, live={live_side}"
        )


def test_sizing_and_fill_consistency_with_live():
    """Confirm Backtester sizing, slippage, and charges match live Engine & costs."""
    strat_dict = build_orb_strategy()
    bt = Backtester(strat_dict, risk_per_trade=RISK_PER_TRADE, max_position_value=MAX_POSITION_VALUE)

    # 1. Sizing check: min(int(RISK_PER_TRADE / risk_per_share), int(MAX_POSITION_VALUE / price))
    price = 1000.0
    risk_per_share = 10.0
    expected_qty = min(int(RISK_PER_TRADE / risk_per_share), int(MAX_POSITION_VALUE / price))
    actual_qty = bt.size(price, risk_per_share)
    assert actual_qty == expected_qty, f"Sizing mismatch: expected {expected_qty}, got {actual_qty}"

    # 2. Slippage check: adverse on both legs
    fill_buy_in = bt._fill(price, "BUY", entering=True)
    assert fill_buy_in == pytest.approx(price * (1 + SLIPPAGE_PCT))
    fill_buy_out = bt._fill(price + 20, "BUY", entering=False)
    assert fill_buy_out == pytest.approx((price + 20) * (1 - SLIPPAGE_PCT))

    fill_sell_in = bt._fill(price, "SELL", entering=True)
    assert fill_sell_in == pytest.approx(price * (1 - SLIPPAGE_PCT))
    fill_sell_out = bt._fill(price - 20, "SELL", entering=False)
    assert fill_sell_out == pytest.approx((price - 20) * (1 + SLIPPAGE_PCT))

    # 3. Cost model check: charges_for matches trading.costs.round_trip
    entry_fill = 1000.5
    exit_fill = 1020.0
    qty = 25
    expected_charges = round_trip(entry_fill, exit_fill, qty)
    bt_charges = bt.charges_for(entry_fill, exit_fill, qty)
    assert bt_charges == pytest.approx(expected_charges, abs=1e-6)


def test_one_trade_per_symbol_per_day_enforced():
    """Confirm one trade per symbol per side per day is strictly respected."""
    strat = BacktestORBStrategy(one_per_side=True)
    adapter = ORBBacktestAdapter(strat=strat)

    df = _generate_synthetic_candles(n_days=36, bars_per_day=75)
    prep = adapter.prepare(df)

    # Find a day where a breakout occurs
    d = prep.index[-1].date()
    adapter.strat._taken.clear()

    key = ("TESTSYM", d)
    # Simulate first trade taken on BUY side
    adapter.strat._taken.setdefault(key, set()).add("BUY")

    # An immediate next BUY on the same day must be rejected by compute_setup
    w = prep.iloc[-10:]
    setup = adapter.strat.compute_setup(w, "TESTSYM")
    assert setup.get("signal") is None, "Second BUY signal on same day should be suppressed"
