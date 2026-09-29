"""Unit tests for Phase 2: Structured MarketFacts Object & Single-Source Technicals.

Enforces:
1. MarketFacts fields are populated directly from ORBMA200Strategy.compute_setup.
2. No duplicate logic: volume ratio, VWAP, ORB, and 200MA match strategy output exactly.
3. Missing data serializes to JSON null (never invented or estimated numbers).
4. Prompt received by LLM council contains valid JSON adhering to MarketFacts schema.
5. Candidate setup logging to SQLite ledger persists full MarketFacts JSON.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from trading.ledger import Ledger
from trading.market_facts import (
    MarketFacts,
    SignalFacts,
    ORBFacts,
    TrendFacts,
    VolumeFacts,
    VWAPFacts,
    MarketDirectionFacts,
    EventFacts,
    FundamentalFacts,
    NewsItem,
    build_market_facts,
)
from trading.strategies.orb_ma200 import ORBMA200Strategy
from trading.agents.agent_trader import (
    AgentTradeRecommendation,
    compute_indicators,
    evaluate_stock,
)

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def temp_ledger(tmp_path):
    db_file = tmp_path / "test_facts_ledger.db"
    return Ledger(db_path=db_file)


def create_mock_intraday_data(symbol: str = "SBIN") -> tuple[pd.DataFrame, datetime]:
    """Create 35 days of 5m candles with a valid fresh breakout on the final day."""
    base_dates = []
    curr_date = datetime(2026, 8, 10, 9, 15, tzinfo=IST)
    while len(base_dates) < 2500:
        if curr_date.weekday() < 5:
            if "09:15" <= curr_date.strftime("%H:%M") <= "15:25":
                base_dates.append(curr_date)
        curr_date += timedelta(minutes=5)

    n_bars = len(base_dates)
    prices = np.linspace(800.0, 900.0, n_bars)

    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 2.0,
        "Low": prices - 2.0,
        "Close": prices,
        "Volume": 1000.0,
    }, index=pd.DatetimeIndex(base_dates))

    last_day_date = df.index[-1].date()
    day_indices = np.where(df.index.date == last_day_date)[0]

    # Opening Range (09:15, 09:20, 09:25) -> High 902, Low 895 (0.78% width)
    or_idx = day_indices[:3]
    df.iloc[or_idx, df.columns.get_loc("Open")] = 898.0
    df.iloc[or_idx, df.columns.get_loc("High")] = 902.0
    df.iloc[or_idx, df.columns.get_loc("Low")] = 895.0
    df.iloc[or_idx, df.columns.get_loc("Close")] = 900.0
    df.iloc[or_idx, df.columns.get_loc("Volume")] = 1000.0

    # Breakout bar (09:30): Close 904.5 > 902.0 with high volume and structural stop
    breakout_idx = day_indices[3]
    df.iloc[breakout_idx, df.columns.get_loc("Open")] = 901.0
    df.iloc[breakout_idx, df.columns.get_loc("High")] = 905.0
    df.iloc[breakout_idx, df.columns.get_loc("Low")] = 900.45  # Stop ~899.43, risk ~5.07 pts (friction 28.5% <= 30% and risk <= 1.5 ATR)
    df.iloc[breakout_idx, df.columns.get_loc("Close")] = 904.5
    df.iloc[breakout_idx, df.columns.get_loc("Volume")] = 3500.0

    eval_df = df.iloc[: breakout_idx + 1]
    eval_now = eval_df.index[-1] + timedelta(minutes=5, seconds=35)
    return eval_df, eval_now


def test_market_facts_schema_and_serialization():
    """Verify MarketFacts strict Pydantic model serialization with nulls for missing values."""
    facts = MarketFacts(
        symbol="SBIN",
        company_name="State Bank of India",
        asof="2026-09-28T09:35:00+05:30",
        signal=SignalFacts(side="BUY", price=904.5, stop_loss=901.86, target=909.79, rr=2.0),
        orb=ORBFacts(or_high=902.0, or_low=895.0, or_width_pct=0.78, or_valid=True),
        trend=TrendFacts(price=904.5, sma200_1h=850.0, sma200_30m=860.0, state="up"),
        volume=VolumeFacts(breakout_bar_vol=3500.0, avg20=1000.0, ratio=3.5, bar_complete=True),
        vwap=VWAPFacts(value=902.15, slope="up", aligned=True),
        market=MarketDirectionFacts(nifty_pct=0.45, direction="supports"),
        events=EventFacts(asm_gsm=False, near_circuit=False),
        news=[NewsItem(headline="SBI Q1 Results Beat Street Estimates", source="Mint", published_at="2026-09-28 08:30")],
        fundamentals=FundamentalFacts(period="Jun 2026", basis="consolidated", yoy_net_profit_pct=14.2),
    )

    json_str = facts.to_json()
    parsed = json.loads(json_str)

    assert parsed["symbol"] == "SBIN"
    assert parsed["company_name"] == "State Bank of India"
    assert parsed["signal"]["side"] == "BUY"
    assert parsed["signal"]["price"] == 904.5
    assert parsed["orb"]["or_valid"] is True
    assert parsed["volume"]["ratio"] == 3.5
    assert parsed["trend"]["state"] == "up"

    # Missing fields must serialize to JSON null, not invented numbers
    assert parsed["events"]["gap_pct"] is None
    assert parsed["market"]["india_vix"] is None
    assert parsed["fundamentals"]["yoy_revenue_pct"] is None


def test_build_market_facts_from_strategy_computed_values():
    """Ensure build_market_facts pulls technicals directly from ORBMA200Strategy.compute_setup."""
    df, now = create_mock_intraday_data("SBIN")
    strat = ORBMA200Strategy(max_cost_risk_ratio=1.0)

    # Strategy computes setup once
    setup = strat.compute_setup(df, "SBIN", now=now)
    assert setup["signal"] is not None
    sig = setup["signal"]

    facts = build_market_facts(
        symbol="SBIN",
        strategy_setup=setup,
        market_direction={"nifty_pct": 0.5, "direction": "supports"},
        events={"asm_gsm": False, "near_circuit": False},
        news=[{"headline": "Test Headline", "source": "ET", "published_at": "2026-09-28 09:00"}],
        fundamentals={"period": "Jun 2026", "yoy_net_profit_pct": 12.5},
        asof=now,
    )

    # Invariants: Facts must match strategy computed values exactly
    assert facts.symbol == "SBIN"
    assert facts.company_name == "State Bank of India"
    assert facts.signal is not None
    assert facts.signal.price == sig.price
    assert facts.signal.stop_loss == sig.stop_loss
    assert facts.signal.target == sig.target
    assert facts.orb.or_high == setup["orb"]["or_high"]
    assert facts.orb.or_low == setup["orb"]["or_low"]
    assert facts.orb.or_width_pct == setup["orb"]["or_width_pct"]
    assert facts.volume.ratio == setup["volume"]["ratio"]
    assert facts.vwap.value == setup["vwap"]["value"]
    assert facts.trend.state == setup["trend"]["state"]
    assert facts.trend.sma200_1h == setup["trend"]["sma200_1h"]


def test_compute_indicators_delegation():
    """Verify compute_indicators delegates to ORBMA200Strategy without duplicating indicator logic."""
    df, now = create_mock_intraday_data("SBIN")
    strat = ORBMA200Strategy()
    setup = strat.compute_setup(df, "SBIN", now=now)
    ind = compute_indicators(df)

    assert ind["ltp"] == setup["price"]
    assert ind["or_high"] == setup["orb"]["or_high"]
    assert ind["or_low"] == setup["orb"]["or_low"]
    assert ind["vol_ratio"] == setup["volume"]["ratio"]
    assert ind["session_vwap"] == setup["vwap"]["value"]
    assert ind["trend_gate"] == setup["trend"]["state"]
    assert ind["buy_stop"] == setup["buy_stop"]
    assert ind["buy_target"] == setup["buy_target"]


def test_evaluate_stock_passes_market_facts_json_to_council_and_ledger(temp_ledger, monkeypatch):
    """Verify evaluate_stock serializes MarketFacts to JSON for prompt and ledger."""
    df, now = create_mock_intraday_data("SBIN")
    strat = ORBMA200Strategy(max_cost_risk_ratio=1.0)
    setup = strat.compute_setup(df, "SBIN", now=now)

    facts = build_market_facts(
        symbol="SBIN",
        strategy_setup=setup,
        market_direction={"nifty_pct": 0.35, "direction": "supports"},
        news=[{"headline": "SBI Loan Growth Accelerates", "source": "Mint", "published_at": "2026-09-28 09:00"}],
        asof=now,
    )

    ctx = {
        "symbol": "SBIN",
        "signal": setup["signal"],
        "ltp": setup["price"],
        "facts": facts,
        "facts_json": facts.to_json(),
    }

    captured_prompt = None

    def mock_call_structured(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs.get("prompt")
        return AgentTradeRecommendation(
            symbol="SBIN",
            action="BUY",
            entry_price=setup["signal"].price,
            stop_loss=setup["signal"].stop_loss,
            target=setup["signal"].target,
            conviction=8,
            technical_analysis="OR breakout confirmed with volume expansion.",
            bull_case="Strong bullish thesis.",
            bear_case="Risk monitored.",
            decision_rationale="Council approves BUY based on MarketFacts.",
            trend_state="up",
        )

    import trading.agents.agent_trader as at_mod
    monkeypatch.setattr(at_mod, "call_structured", mock_call_structured)

    rec = evaluate_stock("SBIN", ledger=temp_ledger, ctx=ctx)
    assert rec is not None
    assert rec.action == "BUY"
    assert rec.conviction == 8

    # Verify prompt contains valid MarketFacts JSON
    assert captured_prompt is not None
    assert "MARKET FACTS (VERIFIED DETERMINISTIC AUDIT):" in captured_prompt
    assert '"symbol": "SBIN"' in captured_prompt
    assert '"ratio": 3.5' in captured_prompt

    # Verify candidate setup in SQLite ledger persisted the MarketFacts dictionary
    cands = temp_ledger.candidate_signals_for_date()
    assert len(cands) == 1
    cand = cands[0]
    assert cand["symbol"] == "SBIN"
    assert cand["side"] == "BUY"

    facts_in_db = json.loads(cand["facts_json"])
    assert facts_in_db["symbol"] == "SBIN"
    assert facts_in_db["volume"]["ratio"] == 3.5
    assert facts_in_db["orb"]["or_valid"] is True
    assert facts_in_db["market"]["direction"] == "supports"
