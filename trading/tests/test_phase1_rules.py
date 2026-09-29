"""Unit tests for Phase 1: Deterministic rules and risk gates in code.

Covers:
1. Trend gate in ExecutionRouter: rejects wrong-side, mixed, unavailable, and missing trend states.
2. Structural stop requirement: rejects missing or invalid stop loss (no entry*0.01 fallback).
3. Daily profit lock and loss limit: triggers at +/- (+1000.0 and -500.0).
4. Volume ratio calculation: drops incomplete bars and uses completed bars only.
5. Deterministic pre-council gate & side constraint: ensures trigger decides side and filters invalid setups.
6. Ledger candidate signals logging: verifies candidate setups and decisions are logged for shadow P&L.
"""
from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from trading.config import DAILY_LOSS_LIMIT, DAILY_PROFIT_TARGET
from trading.execution_router import ExecutionRouter
from trading.ledger import Ledger
from trading.strategies.orb_ma200 import ORBMA200Strategy, drop_unclosed_bars
from trading.agents.agent_trader import compute_indicators, AgentTradeRecommendation, execute_recommendation


IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def temp_ledger():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ledger.db"
        ledger = Ledger(db_path=db_path)
        yield ledger


def test_trend_gate_in_execution_router(temp_ledger):
    """Router trend_gate must reject wrong-side, mixed, and unavailable trends."""
    router = ExecutionRouter(mode="paper", ledger=temp_ledger)
    base_sig = {
        "symbol": "SBIN",
        "qty": 3,
        "price": 800.0,
        "stop_loss": 790.0,
        "target": 820.0,
        "strategy_id": "orb_ma200",
        "ts": time.time(),
    }

    # 1. Missing or None trend_state -> reject
    sig_no_trend = dict(base_sig, side="BUY")
    sig_no_trend.pop("trend_state", None)
    err = router.risk_check(sig_no_trend)
    assert err is not None
    assert "trend_gate_failed" in err

    # 2. Mixed trend state -> reject
    sig_mixed = dict(base_sig, side="BUY", trend_state="mixed")
    err = router.risk_check(sig_mixed)
    assert err is not None
    assert "trend_gate_failed" in err
    assert "mixed" in err

    # 3. Unavailable trend state -> reject
    sig_unavail = dict(base_sig, side="SELL", stop_loss=810.0, target=780.0, trend_state="unavailable")
    err = router.risk_check(sig_unavail)
    assert err is not None
    assert "trend_gate_failed" in err
    assert "unavailable" in err

    # 4. BUY with downtrend -> reject
    sig_buy_down = dict(base_sig, side="BUY", trend_state="down")
    err = router.risk_check(sig_buy_down)
    assert err is not None
    assert "trend_gate_failed" in err
    assert "BUY requires trend_state 'up'" in err

    # 5. SELL with uptrend -> reject
    sig_sell_up = dict(base_sig, side="SELL", price=800.0, stop_loss=810.0, trend_state="up")
    err = router.risk_check(sig_sell_up)
    assert err is not None
    assert "trend_gate_failed" in err
    assert "SELL requires trend_state 'down'" in err

    # 6. Aligned BUY (up) and SELL (down) -> pass trend gate
    sig_buy_ok = dict(base_sig, side="BUY", trend_state="up")
    assert router.risk_check(sig_buy_ok) is None

    sig_sell_ok = dict(base_sig, side="SELL", price=800.0, stop_loss=810.0, target=780.0, trend_state="down")
    assert router.risk_check(sig_sell_ok) is None


def test_structural_stop_required_no_fallback(temp_ledger):
    """Missing or invalid stop loss must be rejected, not filled with entry*0.01 fallback."""
    router = ExecutionRouter(mode="paper", ledger=temp_ledger)

    # 1. Missing stop in router
    sig_no_sl = {
        "symbol": "SBIN",
        "side": "BUY",
        "qty": 3,
        "price": 800.0,
        "stop_loss": None,
        "target": 820.0,
        "trend_state": "up",
    }
    err = router.risk_check(sig_no_sl)
    assert err is not None
    assert "missing_stop_loss" in err

    # 2. Stop equal to entry (risk = 0)
    sig_zero_risk = dict(sig_no_sl, stop_loss=800.0)
    err = router.risk_check(sig_zero_risk)
    assert err is not None
    assert "invalid_stop_loss" in err

    # 3. Missing stop in execute_recommendation
    rec_no_sl = AgentTradeRecommendation(
        symbol="SBIN",
        action="BUY",
        entry_price=800.0,
        stop_loss=0.0,
        target=820.0,
        conviction=8,
        technical_analysis="Test",
        bull_case="Test",
        bear_case="Test",
        decision_rationale="Test",
        trend_state="up",
    )
    res = execute_recommendation(rec_no_sl, router)
    assert not res["executed"]
    assert "missing_stop_loss" in res["reason"]


def test_profit_lock_and_loss_limit(temp_ledger):
    """Router must enforce DAILY_PROFIT_TARGET (+1000) and DAILY_LOSS_LIMIT (-500)."""
    router = ExecutionRouter(mode="paper", ledger=temp_ledger)
    sig = {
        "symbol": "SBIN",
        "side": "BUY",
        "qty": 3,
        "price": 800.0,
        "stop_loss": 790.0,
        "target": 820.0,
        "trend_state": "up",
    }

    # Simulate hitting DAILY_PROFIT_TARGET
    temp_ledger.day_realized_pnl = lambda: DAILY_PROFIT_TARGET
    err_profit = router.risk_check(sig)
    assert err_profit is not None
    assert "daily_profit_target_hit" in err_profit

    # Simulate exceeding DAILY_PROFIT_TARGET
    temp_ledger.day_realized_pnl = lambda: DAILY_PROFIT_TARGET + 150.0
    err_profit_over = router.risk_check(sig)
    assert err_profit_over is not None
    assert "daily_profit_target_hit" in err_profit_over

    # Simulate hitting DAILY_LOSS_LIMIT (-500)
    temp_ledger.day_realized_pnl = lambda: DAILY_LOSS_LIMIT
    err_loss = router.risk_check(sig)
    assert err_loss is not None
    assert "daily_loss_limit_hit" in err_loss

    # Simulate exceeding DAILY_LOSS_LIMIT (-600)
    temp_ledger.day_realized_pnl = lambda: DAILY_LOSS_LIMIT - 100.0
    err_loss_over = router.risk_check(sig)
    assert err_loss_over is not None
    assert "daily_loss_limit_hit" in err_loss_over

    # Normal day P&L (+200) -> passes
    temp_ledger.day_realized_pnl = lambda: 200.0
    assert router.risk_check(sig) is None


def test_drop_unclosed_bars_and_volume_calculation():
    """Bars not yet closed (bar_end > now) must be dropped and not corrupt volume ratio."""
    # Bar 1: 09:15-09:20, Bar 2: 09:20-09:25, Bar 3: 09:25-09:30, Bar 4: 09:30-09:35 (forming)
    times = [
        datetime(2026, 9, 28, 9, 15, tzinfo=IST),
        datetime(2026, 9, 28, 9, 20, tzinfo=IST),
        datetime(2026, 9, 28, 9, 25, tzinfo=IST),
        datetime(2026, 9, 28, 9, 30, tzinfo=IST),  # end is 09:35
    ]
    df = pd.DataFrame({
        "Open": [100.0, 100.2, 100.5, 101.0],
        "High": [100.3, 100.6, 100.8, 101.5],
        "Low": [99.8, 100.1, 100.3, 100.9],
        "Close": [100.2, 100.5, 100.7, 101.2],
        "Volume": [1000.0, 1200.0, 1100.0, 50.0],  # Bar 4 only has 50 volume (incomplete)
    }, index=pd.DatetimeIndex(times))

    # At 09:32 IST, the 09:30 bar has NOT closed yet (closes at 09:35)
    now_eval = datetime(2026, 9, 28, 9, 32, tzinfo=IST)
    cleaned_df = drop_unclosed_bars(df, interval_minutes=5, now=now_eval)

    assert len(cleaned_df) == 3
    assert cleaned_df.index[-1] == datetime(2026, 9, 28, 9, 25, tzinfo=IST)

    # At 09:35:05 with default grace_seconds=30, the bar ending at 09:35 is still in grace (closes at 09:35:30)
    now_in_grace = datetime(2026, 9, 28, 9, 35, 5, tzinfo=IST)
    cleaned_grace = drop_unclosed_bars(df, interval_minutes=5, now=now_in_grace, grace_seconds=30)
    assert len(cleaned_grace) == 3

    # At 09:35:35, the 30s grace has elapsed -> bar is finalized and included
    now_post_grace = datetime(2026, 9, 28, 9, 35, 35, tzinfo=IST)
    cleaned_df2 = drop_unclosed_bars(df, interval_minutes=5, now=now_post_grace, grace_seconds=30)
    assert len(cleaned_df2) == 4


    # Indicators on completed bars exposes bar_complete and uses prior 20 bars
    # Create 25 completed bars
    ts_list = [datetime(2026, 9, 28, 9, 15, tzinfo=IST) + timedelta(minutes=5 * i) for i in range(25)]
    df_25 = pd.DataFrame({
        "Open": [100.0] * 25,
        "High": [101.0] * 25,
        "Low": [99.5] * 25,
        "Close": [100.5] * 25,
        "Volume": [1000.0] * 24 + [2500.0],  # 24 bars @ 1000, 25th bar (breakout) @ 2500
    }, index=pd.DatetimeIndex(ts_list))

    inds = compute_indicators(df_25)
    assert inds["bar_complete"] is True
    # Volume ratio: 2500 / 1000 = 2.5x
    assert inds["vol_ratio"] == 2.5
    assert inds["vol_confirmed"] is True


def test_candidate_signals_ledger_persistence(temp_ledger):
    """Candidate setups and council decisions must be recorded in SQLite for shadow P&L."""
    cand_id = temp_ledger.record_candidate({
        "symbol": "RELIANCE",
        "side": "BUY",
        "price": 3000.0,
        "stop_loss": 2980.0,
        "target": 3040.0,
        "strategy_id": "orb_ma200",
        "trend_state": "up",
        "gate_passed": True,
        "facts_json": {"test": "data"},
    })
    assert cand_id > 0

    # Verify candidate recorded
    cands = temp_ledger.candidate_signals_for_date()
    assert len(cands) == 1
    assert cands[0]["symbol"] == "RELIANCE"
    assert cands[0]["side"] == "BUY"
    assert cands[0]["trend_state"] == "up"
    assert cands[0]["shadow_status"] == "open"

    # Update with council decision
    temp_ledger.update_candidate_decision(cand_id, {
        "action": "BUY",
        "conviction": 8,
        "reasons": "Strong trend and volume alignment",
        "executed": True,
        "trade_id": 101,
        "rejection_reason": None,
    })

    cands_updated = temp_ledger.candidate_signals_for_date()
    assert cands_updated[0]["council_action"] == "BUY"
    assert cands_updated[0]["council_conviction"] == 8
    assert cands_updated[0]["executed"] == 1
    assert cands_updated[0]["trade_id"] == 101

    # Record shadow exit for hypothetical tracking
    pnl = temp_ledger.record_shadow_exit(cand_id, exit_price=3040.0)
    assert pnl == 40.0

    cands_closed = temp_ledger.candidate_signals_for_date()
    assert cands_closed[0]["shadow_status"] == "closed"
    assert cands_closed[0]["shadow_pnl"] == 40.0


def test_deterministic_precouncil_gate_and_side_constraint(temp_ledger, monkeypatch):
    """Signals that fail quant rules must not reach council, and trigger strictly decides side."""
    from trading.strategy import Signal
    from trading.agents.agent_trader import evaluate_stock
    import trading.agents.agent_trader as at_mod

    # 1. Quant rules fail -> evaluate_stock returns None
    monkeypatch.setattr(at_mod, "fetch_symbol_context", lambda sym: {
        "symbol": sym, "ltp": 800.0, "signal": None, "or_valid": False,
    })
    rec_blocked = evaluate_stock("SBIN", ledger=temp_ledger)
    assert rec_blocked is None

    # 2. Breakdown signal -> candidate side is SELL
    sell_sig = Signal(
        symbol="SBIN", side="SELL", price=800.0, stop_loss=810.0,
        target=780.0, strategy_id="orb_ma200", trend_state="down",
    )
    monkeypatch.setattr(at_mod, "fetch_symbol_context", lambda sym: {
        "symbol": sym, "ltp": 800.0, "signal": sell_sig, "or_valid": True,
        "or_high": 815.0, "or_low": 805.0, "or_pct": 1.2,
        "ma_200_1h": 830.0, "ma_200_30m": 825.0, "trend_gate": "DOWNTREND",
        "vol_ratio": 1.8, "vol_confirmed": True, "bar_complete": True,
        "session_vwap": 808.0, "vwap_slope": "FALLING", "atr": 5.0,
        "realtime_news": "None", "quarterly_growth": "None", "market_direction": "Neutral",
    })

    # If LLM hallucinates BUY on a SELL breakdown, code must coerce to HOLD
    hallucinated_rec = AgentTradeRecommendation(
        symbol="SBIN", action="BUY", entry_price=800.0, stop_loss=790.0,
        target=820.0, conviction=9, technical_analysis="Test",
        bull_case="Hallucinated buy thesis", bear_case="None",
        decision_rationale="Hallucinated BUY",
    )
    monkeypatch.setattr(at_mod, "call_structured", lambda **kwargs: hallucinated_rec)

    rec_coerced = evaluate_stock("SBIN", ledger=temp_ledger)
    assert rec_coerced is not None
    assert rec_coerced.action == "HOLD"
    assert rec_coerced.conviction == 1
    assert "Cannot BUY on a SELL breakdown" in rec_coerced.decision_rationale
    assert rec_coerced.trend_state == "down"

    # If LLM agrees with SELL breakdown, side is preserved with trend_state from strategy
    valid_sell_rec = AgentTradeRecommendation(
        symbol="SBIN", action="SELL", entry_price=800.0, stop_loss=810.0,
        target=780.0, conviction=8, technical_analysis="Test",
        bull_case="Not applicable", bear_case="Strong breakdown",
        decision_rationale="Valid SELL breakdown aligned with downtrend",
    )
    monkeypatch.setattr(at_mod, "call_structured", lambda **kwargs: valid_sell_rec)

    rec_valid = evaluate_stock("SBIN", ledger=temp_ledger)
    assert rec_valid is not None
    assert rec_valid.action == "SELL"
    assert rec_valid.conviction == 8
    assert rec_valid.trend_state == "down"


def test_trend_gate_applies_only_to_orb_ma200(temp_ledger):
    """Router trend_gate must strictly gate orb_ma200 strategies but permit EMA / AVWAP."""
    router = ExecutionRouter(mode="paper", ledger=temp_ledger)

    # 1. orb_ma200 without trend_state -> rejected
    orb_sig = {
        "symbol": "INFY", "side": "BUY", "qty": 1, "price": 1500.0,
        "stop_loss": 1480.0, "target": 1540.0, "strategy_id": "orb_ma200",
        "trend_state": "unavailable",
    }
    err = router.risk_check(orb_sig)
    assert err is not None
    assert "trend_gate_failed" in err

    # 2. Non-orb strategy (ema_9_21) without trend_state -> passes
    ema_sig = {
        "symbol": "INFY", "side": "BUY", "qty": 1, "price": 1500.0,
        "stop_loss": 1480.0, "target": 1540.0, "strategy_id": "ema_9_21",
        "trend_state": "unavailable",
    }
    assert router.risk_check(ema_sig) is None

    # 3. AVWAP scalp without trend_state -> passes
    avwap_sig = {
        "symbol": "INFY", "side": "SELL", "qty": 1, "price": 1500.0,
        "stop_loss": 1520.0, "target": 1460.0, "strategy_id": "avwap_scalp",
    }
    assert router.risk_check(avwap_sig) is None


def test_stop_vs_entry_validation(temp_ledger):
    """Router must reject BUY if stop >= entry and SELL if stop <= entry."""
    router = ExecutionRouter(mode="paper", ledger=temp_ledger)

    # BUY with stop above entry (invalid)
    buy_bad = {
        "symbol": "TCS", "side": "BUY", "qty": 1, "price": 3000.0,
        "stop_loss": 3050.0, "target": 3100.0, "strategy_id": "orb_ma200", "trend_state": "up",
    }
    err = router.risk_check(buy_bad)
    assert err is not None
    assert "BUY stop_loss 3050.00 must be below entry price 3000.00" in err

    # BUY with stop equal to entry (invalid)
    buy_eq = dict(buy_bad, stop_loss=3000.0)
    err = router.risk_check(buy_eq)
    assert err is not None
    assert "BUY stop_loss 3000.00 must be below entry price 3000.00" in err

    # SELL with stop below entry (invalid)
    sell_bad = {
        "symbol": "TCS", "side": "SELL", "qty": 1, "price": 3000.0,
        "stop_loss": 2950.0, "target": 2900.0, "strategy_id": "orb_ma200", "trend_state": "down",
    }
    err = router.risk_check(sell_bad)
    assert err is not None
    assert "SELL stop_loss 2950.00 must be above entry price 3000.00" in err

    # SELL with stop equal to entry (invalid)
    sell_eq = dict(sell_bad, stop_loss=3000.0)
    err = router.risk_check(sell_eq)
    assert err is not None
    assert "SELL stop_loss 3000.00 must be above entry price 3000.00" in err


def test_cost_floor_gate(temp_ledger):
    """Router and strategy must skip trades where round-trip friction > 30% of risk."""
    router = ExecutionRouter(mode="paper", ledger=temp_ledger)

    # 1. Narrow stop: price=1000, stop=998 (risk=2 pts = 0.20%)
    # Friction = 0.16%, ratio = 0.16 / 0.20 = 80% > 30% -> REJECTED
    bad_signal = {
        "symbol": "SBIN", "side": "BUY", "qty": 1, "price": 1000.0,
        "stop_loss": 998.0, "target": 1006.0, "strategy_id": "orb_ma200", "trend_state": "up",
    }
    # Test through execute(): must return None and log rejection in ledger
    trade_id_bad = router.execute(bad_signal)
    assert trade_id_bad is None
    assert router.last_rejection is not None
    assert "cost_floor_breached" in router.last_rejection
    rejections = temp_ledger.recent_rejections()
    assert len(rejections) >= 1
    assert "cost_floor_breached" in rejections[0]["reason"]

    # 2. Healthy stop: price=1000, stop=990 (risk=10 pts = 1.0%)
    # Friction = 0.16%, ratio = 0.16 / 1.0 = 16% <= 30% -> PASSES
    good_signal = {
        "symbol": "SBIN", "side": "BUY", "qty": 1, "price": 1000.0,
        "stop_loss": 990.0, "target": 1020.0, "strategy_id": "orb_ma200", "trend_state": "up",
    }
    assert router.risk_check(good_signal) is None
    trade_id_good = router.execute(good_signal)
    assert trade_id_good is not None
    assert trade_id_good > 0


def test_shadow_exit_resolution_job(temp_ledger):
    """Verify hypothetical exit resolution with 15:15 bar open, paper slippage, and charges."""
    from trading.shadow import resolve_candidate_from_bars

    candidate = {
        "id": 42,
        "symbol": "SBIN",
        "side": "BUY",
        "price": 800.0,
        "stop_loss": 790.0,
        "target": 820.0,
        "ts": datetime(2026, 9, 28, 10, 0, tzinfo=IST).timestamp(),
        "date": "2026-09-28",
    }

    times = [
        datetime(2026, 9, 28, 10, 5, tzinfo=IST),
        datetime(2026, 9, 28, 10, 10, tzinfo=IST),
        datetime(2026, 9, 28, 10, 15, tzinfo=IST),
    ]

    # Case A: Target hit at 10:15
    df_tgt = pd.DataFrame({
        "Open": [801.0, 808.0, 815.0],
        "High": [805.0, 812.0, 822.0],  # hits target 820
        "Low": [798.0, 804.0, 814.0],
        "Close": [804.0, 811.0, 821.0],
        "Volume": [1000.0, 1000.0, 1000.0],
    }, index=pd.DatetimeIndex(times))

    res_tgt = resolve_candidate_from_bars(candidate, df_tgt)
    assert res_tgt is not None
    assert res_tgt["exit_reason"] == "target_hit"
    assert res_tgt["raw_exit_price"] == 820.0
    # Slippage applied: exit_fill = 820 * (1 - 0.0005) = 819.59, entry_fill = 800 * (1 + 0.0005) = 800.40
    assert res_tgt["gross_pnl"] < 20.0
    # Net shadow P&L deducts statutory round-trip charges
    assert res_tgt["shadow_pnl"] < res_tgt["gross_pnl"]
    assert res_tgt["charges"] > 0

    # Case B: Stop loss hit
    df_sl = pd.DataFrame({
        "Open": [800.0, 795.0],
        "High": [802.0, 796.0],
        "Low": [796.0, 788.0],  # breaks stop 790
        "Close": [797.0, 789.0],
        "Volume": [1000.0, 1000.0],
    }, index=pd.DatetimeIndex(times[:2]))
    res_sl = resolve_candidate_from_bars(candidate, df_sl)
    assert res_sl is not None
    assert res_sl["exit_reason"] == "stop_loss_hit"
    assert res_sl["raw_exit_price"] == 790.0
    # Loss includes slippage and charges
    assert res_sl["shadow_pnl"] < -10.0

    # Case C: Square-off at 15:15 resolves at bar Open (not Close)
    times_sq = [datetime(2026, 9, 28, 15, 15, tzinfo=IST)]
    df_sq = pd.DataFrame({
        "Open": [805.0], "High": [808.0], "Low": [803.0], "Close": [806.5], "Volume": [1000.0]
    }, index=pd.DatetimeIndex(times_sq))
    res_sq = resolve_candidate_from_bars(candidate, df_sq)
    assert res_sq is not None
    assert res_sq["exit_reason"] == "squareoff_time"
    # Must resolve at 15:15 bar open (805.0), NOT close (806.5)
    assert res_sq["raw_exit_price"] == 805.0


def test_drop_unclosed_bars_deterministic():
    """Verify drop_unclosed_bars with deterministic mock timestamps and exact grace boundary."""
    times = [
        datetime(2026, 9, 28, 9, 15, tzinfo=IST),
        datetime(2026, 9, 28, 9, 20, tzinfo=IST),
        datetime(2026, 9, 28, 9, 25, tzinfo=IST),
    ]
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=pd.DatetimeIndex(times))

    # At 09:29:50 IST, the 09:25 candle (end 09:30:00 + 30s grace = 09:30:30) is NOT closed
    now_before = datetime(2026, 9, 28, 9, 29, 50, tzinfo=IST)
    cleaned_before = drop_unclosed_bars(df, interval_minutes=5, now=now_before, grace_seconds=30)
    assert len(cleaned_before) == 2  # Only 09:15 and 09:20 bars included

    # At 09:30:35 IST, the 09:25 candle HAS closed (09:30:30 <= 09:30:35)
    now_after = datetime(2026, 9, 28, 9, 30, 35, tzinfo=IST)
    cleaned_after = drop_unclosed_bars(df, interval_minutes=5, now=now_after, grace_seconds=30)
    assert len(cleaned_after) == 3  # All 3 bars included


@pytest.mark.integration
def test_drop_unclosed_bars_with_real_yahoo_data():
    """Verify drop_unclosed_bars on actual live Yahoo data with IST timezone and grace (network integration)."""
    import yfinance as yf
    t = yf.Ticker("RELIANCE.NS")
    df = t.history(period="1d", interval="5m")
    if df.empty:
        pytest.skip("Yahoo Finance unreachable or market data unavailable")

    now = datetime.now(IST)
    cleaned = drop_unclosed_bars(df, interval_minutes=5, now=now, grace_seconds=30)
    assert len(cleaned) <= len(df)
    if not cleaned.empty:
        last_bar_end = cleaned.index[-1] + timedelta(minutes=5) + timedelta(seconds=30)
        assert last_bar_end <= now


