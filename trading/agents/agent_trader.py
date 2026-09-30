"""Multi-Agent Stock Evaluator and Paper Trader.

Combines the multi-role reasoning of TradingAgents (Technical Analyst, Bullish
Researcher, Bearish Researcher, and Trader Decision Agent) with shubya_ai's
deterministic ExecutionRouter and SQLite ledger.

Workflow:
  1. For each symbol in the watchlist/universe, fetch intraday and daily candles.
  2. Compute key technical indicators (EMAs, RSI, ATR, Support/Resistance).
  3. Run multi-agent debate and synthesis via Gemini:
     - Technical Analyst: evaluates price momentum, trend, moving average alignment.
     - Bullish Researcher: establishes the strongest upside thesis.
     - Bearish Researcher: identifies downside risks, false breakouts, resistance.
     - Trader Agent: synthesizes debate into a concrete BUY / SELL / HOLD action,
       with entry price, stop loss, profit target, and conviction score (1-10).
  4. If action is BUY or SELL and conviction >= AGENT_MIN_CONVICTION:
     - Sizing calculated via RISK_PER_TRADE and the pre-market risk multiplier.
     - Dispatches order to ExecutionRouter (hard risk kernel).
     - Trade is recorded in the SQLite ledger and immediately visible on the
       web dashboard (http://localhost:8080).

Usage:
  python -m trading.agents.agent_trader                     # Scan core watchlist
  python -m trading.agents.agent_trader --symbol RELIANCE   # Analyze single symbol
  python -m trading.agents.agent_trader --all               # Scan full Nifty-50
  python -m trading.agents.agent_trader --from-scanner --dry-run
      Take candidates from the F&O long scanner instead. Needs
      AGENT_SCANNER_ENABLED = True in trading/config.py; without it the flags
      are ignored and the watchlist is scanned as usual.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from trading.agents.llm import call_structured
from trading.config import (
    AGENT_MAX_TRADES_PER_SCAN,
    AGENT_MIN_CONVICTION,
    AGENT_SCANNER_ENABLED,
    AGENT_SCANNER_MAX,
    AGENT_SCANNER_REQUIRE_ENTRY_BAND,
    AGENT_SCANNER_STRATEGY_ID,
    AGENT_SCANNER_TIERS,
    MAX_POSITION_VALUE,
    NIFTY50,
    RISK_PER_TRADE,
    SCAN_UNIVERSE,
    WATCHLIST,
    YF_SUFFIX,
)
from trading.execution_router import ExecutionRouter
from trading.ledger import Ledger
from trading.market_facts import MarketFacts, build_market_facts
from trading.notify import notify
from trading.strategies.orb_ma200 import ORBMA200Strategy, drop_unclosed_bars
from trading.strategy import Signal

IST = ZoneInfo("Asia/Kolkata")

AGENT_SYSTEM = (
    "You are a disciplined, professional Multi-Agent Trading Council specializing in the "
    "Opening Range Breakout (ORB) + Multi-Timeframe 200MA Trend-Filter Strategy for Indian NSE equities.\n"
    "Your panel consists of:\n"
    "1. Technical Analyst: Evaluates the Opening Range (09:15-09:30), Dual Multi-Timeframe 200 SMAs (1H & 30m), "
    "Session VWAP slope, ATR(14) volatility, and breakout volume ratio.\n"
    "2. Bullish Researcher: Formulates the Long thesis ONLY when price is above BOTH the 1H 200MA and 30m 200MA, "
    "Opening Range is between 0.3% and 1.5%, and fresh breakout above OR-High is confirmed.\n"
    "3. Bearish Researcher: Formulates the Short thesis when price is below BOTH 200 MAs with OR-Low breakdown, "
    "and rigorously audits for false breakouts, tight/wide ranges, and lack of volume expansion.\n"
    "4. Trader Decision Agent: Synthesizes the debate into a strict, disciplined trading decision adhering to the rulebook.\n\n"
    "MANDATORY STRATEGY RULES (ORB + Multi-TF 200MA):\n"
    "1. Opening Range Gate: Opening range is 09:15-09:30 high/low. Range % = (OR_High - OR_Low) / OR_Low * 100. "
    "Must be between 0.3% and 1.5%. If range is <0.3% (tight fakeout) or >1.5% (extended move), MUST CHOOSE HOLD.\n"
    "2. Trigger: Candle must fresh CLOSE beyond the OR (above OR-High for BUY, below OR-Low for SELL). Wicks without closes are ignored.\n"
    "3. Trend Gate (Core Rule): Price above 200 MA on BOTH 1H and 30m = Uptrend (BUY permitted). "
    "Price below 200 MA on BOTH 1H and 30m = Downtrend (SELL permitted). "
    "If trend is mixed or price not on correct side of both 200 MAs -> STRICTLY HOLD (No trade).\n"
    "4. Volume & VWAP: Breakout volume must be >= 1.5x of 20-bar average. Price must be on the right side of sloping session VWAP.\n"
    "5. Structural Stop & Target: Stop loss = breakout bar extreme +/- 0.25x ATR. Target = price +/- 2.0x Risk. "
    "Risk distance must NOT exceed 1.5x ATR (if risk > 1.5x ATR, trade is skipped -> HOLD).\n"
    "6. Execution Threshold: Only setups satisfying all rules receive high conviction (>=7/10). If any rule fails, action MUST be HOLD with conviction <=5.\n"
    "7. Data Integrity Invariant: Use only supplied data; null = not provided. Never invent or extrapolate missing figures."
)


class AgentTradeRecommendation(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="Final action: BUY (if uptrend gate PASS + OR breakout PASS), SELL (if downtrend gate PASS + OR breakdown PASS), or HOLD"
    )
    entry_price: float = Field(description="Exact recommended entry price level in INR")
    stop_loss: float = Field(description="Strict structural stop-loss price level in INR (breakout extreme +/- 0.25x ATR)")
    target: float = Field(description="Profit target price level in INR (exact 2.0x risk)")
    conviction: int = Field(
        description="Conviction score from 1 (weakest) to 10 (highest). Must be >=7 for entries meeting all ORB + 200MA rules", ge=1, le=10
    )
    technical_analysis: str = Field(
        description="Technical analyst view on ORB breakout, 1H/30m 200 MAs, VWAP, and volume expansion"
    )
    bull_case: str = Field(description="Bullish researcher thesis aligning with ORB long breakout criteria")
    bear_case: str = Field(description="Bearish researcher counter-thesis or breakdown short criteria")
    decision_rationale: str = Field(
        description="Trader agent synthesis strictly adhering to the ORB + Multi-TF 200MA rulebook"
    )
    trend_state: str = Field(
        default="unavailable",
        description="Deterministic trend state computed from strategy 200 SMAs (up/down/mixed/unavailable)",
    )
    candidate_id: int | None = Field(
        default=None,
        description="Ledger candidate_signals tracking ID for shadow P&L",
    )


def compute_indicators(
    df_5m: pd.DataFrame,
    df_30m: pd.DataFrame | None = None,
    df_1h: pd.DataFrame | None = None,
) -> dict:
    """Compute indicators delegated to ORBMA200Strategy.compute_setup to avoid duplicate math."""
    if df_5m.empty or len(df_5m) < 10:
        return {}

    strat = ORBMA200Strategy()
    setup = strat.compute_setup(df_5m, "INDICATORS")
    close = df_5m["Close"]
    ema9 = round(float(close.ewm(span=9, adjust=False).mean().iloc[-1]), 2)
    ema21 = round(float(close.ewm(span=21, adjust=False).mean().iloc[-1]), 2)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

    return {
        "ltp": setup["price"],
        "session_time": setup["ts"].strftime("%H:%M") if setup.get("ts") else "N/A",
        "or_high": setup["orb"]["or_high"],
        "or_low": setup["orb"]["or_low"],
        "or_pct": setup["orb"]["or_width_pct"],
        "or_valid": setup["orb"]["or_valid"],
        "or_trigger_status": "BREAKOUT" if setup.get("signal") else ("VALID_OR" if setup["orb"]["or_valid"] else "INSIDE_OR"),
        "ma_200_1h": setup["trend"]["sma200_1h"],
        "ma_200_30m": setup["trend"]["sma200_30m"],
        "trend_gate": setup["trend"]["state"],
        "session_vwap": setup["vwap"]["value"],
        "vwap_slope": setup["vwap"]["slope"],
        "atr": setup["atr"],
        "max_risk_atr": round(1.5 * setup["atr"], 2),
        "vol_ratio": setup["volume"]["ratio"],
        "vol_confirmed": setup["volume"]["ratio"] >= 1.5,
        "bar_complete": True,
        "buy_stop": setup["buy_stop"],
        "buy_risk": round(setup["price"] - setup["buy_stop"], 2),
        "buy_target": setup["buy_target"],
        "sell_stop": setup["sell_stop"],
        "sell_risk": round(setup["sell_stop"] - setup["price"], 2),
        "sell_target": setup["sell_target"],
        "ema9": ema9,
        "ema21": ema21,
        "rsi": round(rsi, 1),
    }



def facts_setup_with_plan(setup: dict, plan: Signal) -> dict:
    """A shallow copy of `setup` whose signal is `plan`, for `build_market_facts`.

    Everything the scanner cannot know — the opening range, the 1H/30m 200MAs,
    VWAP, the volume ratio, ATR — stays exactly as the strategy computed it.
    Only the trade plan is substituted, so the facts and the routed order
    cannot disagree about entry, stop or target.
    """
    risk = plan.price - plan.stop_loss
    return {**setup, "signal": {
        "side": plan.side,
        "price": plan.price,
        "stop_loss": plan.stop_loss,
        "target": plan.target,
        # Derived from the plan itself rather than assumed, so the R:R the
        # council sees is the one the stop and target actually imply.
        "rr": round((plan.target - plan.price) / risk, 2) if risk > 0 else 2.0,
    }}


def fetch_symbol_context(symbol: str, *, want_news: bool = False,
                         plan: Signal | None = None) -> dict | None:
    """Fetch market data, news, fundamentals, and compute ORB + Multi-TF 200MA indicators via MarketFacts.

    ``want_news`` forces the news/fundamentals fetch even when the ORB strategy
    has not fired. The watchlist path leaves it False -- it only pays for the
    fetch once a signal exists -- but a scanner-sourced candidate needs the
    same news even though the ORB trigger belongs to a different rulebook.

    ``plan`` is a `Signal` to trade instead of the ORB strategy's own. The
    scanner supplies its levels this way. Everything else on the returned
    context -- OR block, 1H/30m 200MAs, VWAP, volume, ATR -- is still the
    strategy's own computation over real bars, so the council evaluates a
    scanner name against genuine technical facts rather than a bare claim.
    """
    yf_symbol = symbol + YF_SUFFIX if not symbol.endswith(YF_SUFFIX) else symbol
    try:
        ticker = yf.Ticker(yf_symbol)
        df_5m = ticker.history(period="60d", interval="5m")
        if df_5m.empty or len(df_5m) < 10:
            df_5m = ticker.history(period="5d", interval="5m")
        if df_5m.empty:
            return None
        df_5m = df_5m.tz_convert(IST)

        # Drop any bar not yet closed (bar end time + grace > now)
        now = datetime.now(IST)
        df_5m = drop_unclosed_bars(df_5m, interval_minutes=5, now=now, grace_seconds=30)
        if df_5m.empty:
            return None

        # Compute setup directly via ORBMA200Strategy (single source of truth)
        strat = ORBMA200Strategy()
        setup = strat.compute_setup(df_5m, symbol, now=now)
        orb_sig = setup["signal"]

        # The signal the agent will actually trade: the scanner's plan when one
        # is supplied, otherwise the strategy's own.
        sig = plan if plan is not None else orb_sig
        side = sig.side if sig is not None else None

        # 1. Day High & Day Low
        fast_info = getattr(ticker, "fast_info", None)
        day_high = getattr(fast_info, "day_high", None) or float(df_5m["High"].max())
        day_low = getattr(fast_info, "day_low", None) or float(df_5m["Low"].min())

        # 2. Structured Real-Time News Headlines & 3. Structured Quarterly Fundamentals
        # Phase 3 Lazy fetch: executed ONLY for gated candidates (quant strategy + cost-floor passed)
        structured_news = []
        news_summary = "No gated signal — news fetch skipped (lazy fetch mode)."
        fund_dict = {}
        quarterly_summary = "No gated signal — fundamentals fetch skipped (lazy fetch mode)."

        if orb_sig is not None or want_news:
            from trading.indian_market_data import lazy_fetch_indian_context
            structured_news, news_summary, fund_dict, quarterly_summary = lazy_fetch_indian_context(
                symbol=symbol, ticker=ticker
            )

        # 4. Structured Market Direction (Nifty 50 Trend)
        mkt_dict = {}
        market_direction = "Nifty 50 neutral"
        try:
            nifty_df = yf.Ticker("^NSEI").history(period="2d", interval="1d")
            if len(nifty_df) >= 2:
                n_chg = (nifty_df["Close"].iloc[-1] - nifty_df["Close"].iloc[-2]) / nifty_df["Close"].iloc[-2] * 100
                mkt_dict["nifty_pct"] = round(float(n_chg), 2)
                if n_chg > 0.2:
                    m_label = "BULLISH"
                    mkt_dict["direction"] = "supports" if side == "BUY" else ("opposes" if side == "SELL" else "neutral")
                elif n_chg < -0.2:
                    m_label = "BEARISH"
                    mkt_dict["direction"] = "supports" if side == "SELL" else ("opposes" if side == "BUY" else "neutral")
                else:
                    m_label = "RANGEBOUND"
                    mkt_dict["direction"] = "neutral"
                market_direction = f"Nifty 50 {n_chg:+.2f}% ({m_label})"
        except Exception:
            pass

        # Build MarketFacts directly from strategy computed values. When the
        # scanner supplied the plan, its levels replace the signal inside a
        # shallow copy so the facts and the trade agree on entry, stop and
        # target; the ORB/trend/volume/VWAP facts stay the strategy's own.
        facts_setup = setup
        if plan is not None:
            facts_setup = facts_setup_with_plan(setup, plan)
        facts = build_market_facts(
            symbol=symbol,
            strategy_setup=facts_setup,
            market_direction=mkt_dict,
            news=structured_news,
            fundamentals=fund_dict,
            asof=now,
        )

        return {
            "symbol": symbol,
            "feed": "NSE Real-Time / yfinance (5m + Multi-TF 200MA)",
            "day_high": round(float(day_high), 2) if day_high else None,
            "day_low": round(float(day_low), 2) if day_low else None,
            "realtime_news": news_summary,
            "quarterly_growth": quarterly_summary,
            "market_direction": market_direction,
            "signal": sig,
            "facts": facts,
            "facts_json": facts.to_json(),
            "ltp": setup["price"],
            "session_time": setup["ts"].strftime("%H:%M") if setup.get("ts") else "N/A",
            "or_high": setup["orb"]["or_high"],
            "or_low": setup["orb"]["or_low"],
            "or_pct": setup["orb"]["or_width_pct"],
            "or_valid": setup["orb"]["or_valid"],
            "or_trigger_status": "BREAKOUT" if orb_sig else ("INSIDE_OR" if not setup["orb"]["or_valid"] else "PENDING"),
            "ma_200_1h": setup["trend"]["sma200_1h"],
            "ma_200_30m": setup["trend"]["sma200_30m"],
            "trend_gate": setup["trend"]["state"],
            "session_vwap": setup["vwap"]["value"],
            "vwap_slope": setup["vwap"]["slope"],
            "vol_ratio": setup["volume"]["ratio"],
            "vol_confirmed": setup["volume"]["ratio"] >= 1.5,
            "bar_complete": True,
            "atr": setup["atr"],
            "max_risk_atr": round(1.5 * setup["atr"], 2),
            "buy_stop": setup["buy_stop"],
            "buy_risk": round(setup["price"] - setup["buy_stop"], 2),
            "buy_target": setup["buy_target"],
            "sell_stop": setup["sell_stop"],
            "sell_risk": round(setup["sell_stop"] - setup["price"], 2),
            "sell_target": setup["sell_target"],
            # Kept so callers never have to recompute the OR block themselves.
            "_setup": setup,
            "_orb_signal": orb_sig,
        }
    except Exception as exc:
        print(f"[{symbol}] Failed to fetch market data: {exc}", file=sys.stderr)
        return None


# --- Scanner-sourced candidates -------------------------------------------
# The F&O long scanner (trading/fno/) and this agent were disjoint pipelines:
# the scanner's BUY/WATCH cards were display-only, read by nothing but its own
# web page. These helpers are the whole of the join. The scanner supplies the
# NAME and the PLAN; everything the council reasons over is still computed here
# from real bars, so "the council's ORB rules still apply" is a genuine
# evaluation rather than a formality.

@dataclass
class ScannerSource:
    """What one scan contributes to one agent pass."""
    candidates: list = field(default_factory=list)     # fno.models.Candidate
    refusal: str | None = None                         # why the WHOLE scan is unusable
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (symbol, why)


def scanner_candidates(res, tiers=None, limit: int | None = None) -> ScannerSource:
    """Turn a `ScanResult` into the names this pass may consider.

    `res.candidates` already arrives sorted BUY-first (tradeable, then score,
    then relative strength), so the cap keeps the best of the chosen tiers.
    A name with `trade is None` has no structural stop to place, so it is
    skipped and said out loud rather than handed an invented level.
    """
    tiers = tuple(tiers) if tiers else tuple(AGENT_SCANNER_TIERS)
    cap = AGENT_SCANNER_MAX if limit is None else limit

    # A stale, delayed or out-of-window scan is not a candidate list. Refusing
    # the source is the point of data_ok -- a scan taken before the opening
    # range closed would otherwise look like four perfectly good names.
    if not res.data_ok:
        return ScannerSource(refusal="scan data is not usable — "
                             + " | ".join(res.data_notes[-3:] or ["no reason given"]))
    if not res.signals_allowed:
        return ScannerSource(refusal=f"{res.window}: {res.window_note} — "
                                     "the scanner is not emitting signals in this window")

    usable, skipped = [], []
    for c in res.candidates:
        if c.verdict not in tiers:
            continue
        if c.trade is None:
            skipped.append((c.symbol,
                            f"no plan ({c.blocker_summary or 'the scanner built no trade plan'})"))
            continue
        usable.append(c)

    for c in usable[cap:]:
        skipped.append((c.symbol, f"beyond the per-pass cap of {cap}"))
    return ScannerSource(usable[:cap], skipped=skipped)


def scanner_signal(symbol: str, setup: dict, trade) -> Signal:
    """The scanner's plan, expressed as the agent's `Signal`.

    `trend_state` is the ORB strategy's real 1H/30m read over actual bars.
    It is never fabricated and never taken from the scanner -- the scanner
    computes no 200MA at all, which is exactly why the agent recomputes it.
    """
    return Signal(
        symbol=symbol,
        side="BUY",                      # the F&O scanner is long-only
        price=trade.entry,
        stop_loss=trade.stop,
        target=trade.target1,
        strategy_id=AGENT_SCANNER_STRATEGY_ID,
        trend_state=setup["trend"]["state"],
    )


def scanner_preflight(cand, ctx: dict, *,
                      require_entry_band: bool | None = None) -> str | None:
    """Deterministic gates on a scanner candidate, before the council is paid for.

    Returns a rejection reason, or None to proceed. Two of these carry weight:

    * **The 200MA trend gate.** `fno_scanner` is deliberately not an
      `orb_ma200*` id, so `ExecutionRouter.risk_check` does NOT apply the trend
      gate to it -- the same tested exemption `ema_9_21` and `avwap_scalp` use.
      It is re-applied here, in code, so that exemption stays a design choice
      instead of becoming a hole.
    * **The entry band.** A scanner WATCH name is usually already above its own
      band -- that is part of why it is only on watch -- so without this check
      the agent would be chasing. The price is compared against the agent's
      live LTP rather than the price the scanner froze at scan time.
    """
    if cand.trade is None:
        return "no plan (the scanner built no structural stop for this name)"

    sig = (ctx or {}).get("signal")
    if sig is None:
        return "no plan (no signal on the fetched context)"

    if sig.trend_state != "up":
        return f"scanner_trend_gate_failed (trend '{sig.trend_state}')"

    band = AGENT_SCANNER_REQUIRE_ENTRY_BAND if require_entry_band is None else require_entry_band
    if band:
        t = cand.trade
        ltp = (ctx or {}).get("ltp") or cand.levels.price
        if not (t.entry_low <= ltp <= t.entry_high):
            return (f"outside entry band {t.entry_low:.2f}-{t.entry_high:.2f} "
                    f"(ltp {ltp:.2f}) — not chasing")
    return None


def evaluate_stock(
    symbol: str,
    ledger: Ledger | None = None,
    ctx: dict | None = None,
) -> AgentTradeRecommendation | None:
    """Run the multi-agent panel evaluation on a symbol using ORB + Multi-TF 200MA strategy."""
    if ctx is None:
        ctx = fetch_symbol_context(symbol)
    if not ctx:
        return None

    # Deterministic Pre-Council Gate:
    # Only signals that pass all quant strategy rules (OR width, fresh breakout close,
    # 200 MA trend on 1H+30m, volume ratio >=1.5x, VWAP alignment, structural stop)
    # are sent to the council.
    sig = ctx.get("signal")
    if sig is None:
        return None

    candidate_side = sig.side  # BUY or SELL only
    trend_state = getattr(sig, "trend_state", "unavailable")

    facts = ctx.get("facts")
    facts_json_data = facts.model_dump() if hasattr(facts, "model_dump") else {k: v for k, v in ctx.items() if k != "signal"}
    facts_str = facts.to_json() if hasattr(facts, "to_json") else json.dumps(facts_json_data, indent=2, default=str)

    # Record candidate signal in ledger for shadow P&L tracking
    candidate_id = None
    if ledger:
        candidate_id = ledger.record_candidate({
            "symbol": symbol,
            "side": candidate_side,
            "price": sig.price,
            "stop_loss": sig.stop_loss,
            "target": sig.target,
            "strategy_id": sig.strategy_id,
            "trend_state": trend_state,
            "gate_passed": True,
            "facts_json": facts_json_data,
        })

    ltp = ctx.get("ltp", sig.price)
    prompt = f"""Evaluate trading opportunity for NSE stock: {symbol} using the ORB + Multi-TF 200MA Strategy.
MARKET FACTS (VERIFIED DETERMINISTIC AUDIT):
```json
{facts_str}
```

PANEL INSTRUCTIONS:
- The deterministic quant gate has ALREADY verified technical rules for this {candidate_side} setup.
- Ground all arguments strictly in the verified MarketFacts JSON above. Use only supplied data; null = not provided. Never invent or extrapolate missing figures.
- If Candidate Side is BUY: Bullish Researcher builds upside thesis; Bearish Researcher audits downside risks.
- If Candidate Side is SELL: Bullish Researcher must state 'not applicable' for short breakdown; Bearish Researcher details downside continuation.
- Trader Decision Agent: You may decide {candidate_side} or HOLD. You CANNOT select {'SELL' if candidate_side == 'BUY' else 'BUY'}. Conviction >= 7 only if news, sector, and risk profile align."""

    fallback = AgentTradeRecommendation(
        symbol=symbol,
        action="HOLD",
        entry_price=sig.price,
        stop_loss=sig.stop_loss,
        target=sig.target,
        conviction=1,
        technical_analysis=f"Deterministic gate passed for {candidate_side}. Trend: {trend_state}.",
        bull_case="Bull case evaluated." if candidate_side == "BUY" else "Not applicable for short breakdown.",
        bear_case="Risk audit evaluated.",
        decision_rationale="Fallback: Hold mandated due to LLM failure.",
        trend_state=trend_state,
        candidate_id=candidate_id,
    )

    rec = call_structured(
        agent_name="agent_trader",
        system=AGENT_SYSTEM,
        prompt=prompt,
        output_model=AgentTradeRecommendation,
        fallback=fallback,
        ledger=ledger,
    )
    return _enforce_code_owned_levels(rec, sig, candidate_side, trend_state, candidate_id)


def _enforce_code_owned_levels(rec: AgentTradeRecommendation, sig, candidate_side: str,
                               trend_state: str, candidate_id: int | None
                               ) -> AgentTradeRecommendation:
    """Post-council enforcement, shared by the watchlist and scanner paths.

    The council advises; it does not set prices. Levels, trend read and
    candidate id are overwritten from the deterministic signal, and a flipped
    side is coerced to HOLD. Living in one function is what keeps that
    invariant true on both paths rather than only the one that has a test.
    """
    rec.trend_state = trend_state
    rec.candidate_id = candidate_id
    rec.entry_price = sig.price
    rec.stop_loss = sig.stop_loss
    rec.target = sig.target

    if candidate_side == "BUY" and rec.action == "SELL":
        rec.action = "HOLD"
        rec.conviction = 1
        rec.decision_rationale += " [Code Override: Cannot SELL on a BUY breakout candidate]"
    elif candidate_side == "SELL" and rec.action == "BUY":
        rec.action = "HOLD"
        rec.conviction = 1
        rec.decision_rationale += " [Code Override: Cannot BUY on a SELL breakdown candidate]"
    return rec


def evaluate_scanner_candidate(
    cand,
    ledger: Ledger | None = None,
    ctx: dict | None = None,
) -> AgentTradeRecommendation | None:
    """Run the same council on a candidate the F&O scanner chose.

    The rulebook is unchanged -- `AGENT_SYSTEM` still mandates the ORB and
    multi-timeframe 200MA rules, because you asked to keep them. What differs
    is the prompt: it says which rulebook picked this name and shows the
    scanner's own evidence, so the council is never told the ORB gate already
    passed when the ORB gate is not what selected it.
    """
    if ctx is None:
        return None
    sig = ctx.get("signal")
    t = cand.trade
    if sig is None or t is None:
        return None

    candidate_side = sig.side
    trend_state = getattr(sig, "trend_state", "unavailable")

    facts = ctx.get("facts")
    facts_json_data = facts.model_dump() if hasattr(facts, "model_dump") else {}
    facts_str = (facts.to_json() if hasattr(facts, "to_json")
                 else json.dumps(facts_json_data, indent=2, default=str))

    # Record candidate signal in ledger for shadow P&L tracking — same call the
    # watchlist path makes, so scanner names are shadow-tracked identically.
    candidate_id = None
    if ledger:
        candidate_id = ledger.record_candidate({
            "symbol": cand.symbol,
            "side": candidate_side,
            "price": sig.price,
            "stop_loss": sig.stop_loss,
            "target": sig.target,
            "strategy_id": sig.strategy_id,
            "trend_state": trend_state,
            "gate_passed": True,
            "facts_json": facts_json_data,
        })

    oi, st = cand.oi, cand.structure
    sector_note = (f"{cand.sector} ({cand.sector_strength:+.2f}% vs NIFTY)"
                   if cand.sector_strength is not None else cand.sector)
    scanner_block = "\n".join([
        f"- Scanner verdict: {cand.verdict}  (score {cand.score:.0f}/100, grade {cand.grade})",
        f"- Scanner plan (FIXED — code-owned, not yours to change): entry {t.entry:.2f} "
        f"inside band {t.entry_low:.2f}-{t.entry_high:.2f}; stop {t.stop:.2f} "
        f"({t.stop_basis}); targets {t.target1:.2f} / {t.target2:.2f}; "
        f"R:R {t.rr1:.2f} / {t.rr2:.2f}",
        f"- Open interest: {oi.classification} (reliable={oi.reliable}) — {oi.detail}",
        f"- Relative strength vs NIFTY: {cand.rel_strength:+.2f}%  |  Sector: {sector_note}",
        f"- RVOL: {cand.levels.rvol:.2f}" if cand.levels.rvol is not None else "- RVOL: not provided",
        f"- Breakout: {'; '.join(st.notes) if st.notes else 'no structural notes'}",
        "- Why the scanner did not mark this a BUY: "
        + ("; ".join(cand.rejections) if cand.rejections else "no rejections recorded"),
    ])

    prompt = f"""Evaluate trading opportunity for NSE stock: {cand.symbol} — SOURCED FROM THE F&O LONG SCANNER.
MARKET FACTS (VERIFIED DETERMINISTIC AUDIT — the opening range, 200MA, VWAP and volume facts below are computed here from real bars by the ORB + Multi-TF 200MA strategy):
```json
{facts_str}
```

SCANNER CONTEXT (the scanner's own read — additional evidence, not a substitute for your rules):
{scanner_block}

PANEL INSTRUCTIONS:
- Candidate Side is BUY only; the F&O scanner is a long scanner. You may decide BUY or HOLD. You CANNOT select SELL.
- The entry, stop-loss and target in MarketFacts are the SCANNER'S structural plan. They are code-owned: judge whether to take the trade, but do not re-derive or propose different levels.
- YOUR ORB + MULTI-TIMEFRAME 200MA RULES REMAIN MANDATORY. This name was picked by a different rulebook (opening range, VWAP, EMA20/50, RVOL, futures OI, sector), so treat your rules as a second, independent gate — not as already satisfied.
- In particular, this candidate did NOT necessarily produce an ORB breakout, and its opening-range width and 200MA trend have NOT been pre-screened for you. If the Opening Range gate, the 200MA trend gate, the volume rule or the VWAP rule fails on the facts above, action MUST be HOLD with conviction <= 5.
- Bullish Researcher builds the upside thesis; Bearish Researcher audits downside risk and false breakout risk.
- Ground all arguments strictly in the supplied data. Use only supplied data; null = not provided. Never invent or extrapolate missing figures.
- Conviction >= 7 only if the ORB + 200MA rules hold AND the scanner evidence and the risk profile align."""

    fallback = AgentTradeRecommendation(
        symbol=cand.symbol,
        action="HOLD",
        entry_price=sig.price,
        stop_loss=sig.stop_loss,
        target=sig.target,
        conviction=1,
        technical_analysis=(f"Scanner-sourced candidate. Trend: {trend_state}. "
                            f"Scanner score {cand.score:.0f}/100 ({cand.verdict})."),
        bull_case="Bull case evaluated.",
        bear_case="Risk audit evaluated.",
        decision_rationale="Fallback: Hold mandated due to LLM failure.",
        trend_state=trend_state,
        candidate_id=candidate_id,
    )

    rec = call_structured(
        agent_name="agent_trader",
        system=AGENT_SYSTEM,
        prompt=prompt,
        output_model=AgentTradeRecommendation,
        fallback=fallback,
        ledger=ledger,
    )
    return _enforce_code_owned_levels(rec, sig, candidate_side, trend_state, candidate_id)


def execute_recommendation(
    rec: AgentTradeRecommendation,
    router: ExecutionRouter,
    min_conviction: int = AGENT_MIN_CONVICTION,
    strategy_id: str = "orb_ma200_agents",
) -> dict:
    """Route an approved recommendation through ExecutionRouter.

    `strategy_id` is a code-owned argument, never an LLM-readable field: it
    decides which of the router's gates apply (see `risk_check`), so letting
    the council influence it would let the council choose its own regulator.
    """
    ledger = router.ledger
    result = {
        "symbol": rec.symbol,
        "action": rec.action,
        "conviction": rec.conviction,
        "executed": False,
        "trade_id": None,
        "reason": None,
        "details": rec.model_dump(),
    }

    def _record_outcome(reason: str | None, trade_id: int | None = None, executed: bool = False):
        result["reason"] = reason
        result["trade_id"] = trade_id
        result["executed"] = executed
        if ledger and getattr(rec, "candidate_id", None):
            ledger.update_candidate_decision(rec.candidate_id, {
                "action": rec.action,
                "conviction": rec.conviction,
                "reasons": rec.decision_rationale,
                "executed": executed,
                "trade_id": trade_id,
                "rejection_reason": reason,
            })

    # Missing stop-loss check (Amendment 3: no fallback)
    if not rec.stop_loss or rec.stop_loss <= 0:
        _record_outcome("missing_stop_loss (trade rejected: structural stop required)")
        return result

    entry = rec.entry_price
    sl = rec.stop_loss
    per_share_risk = abs(entry - sl)
    if per_share_risk <= 0.05:
        _record_outcome("invalid_stop_loss (stop loss too close to entry)")
        return result

    if rec.action == "HOLD":
        _record_outcome(f"Agent recommended HOLD (conviction={rec.conviction})")
        return result

    if rec.conviction < min_conviction:
        _record_outcome(f"Conviction {rec.conviction}/10 below threshold {min_conviction}/10")
        return result

    # Size calculation (0.5% risk per trade)
    risk_budget = RISK_PER_TRADE * router.day_config.get("risk_multiplier", 0.5)
    qty = int(risk_budget / per_share_risk)
    if qty <= 0:
        qty = 1

    # Cap by max position value
    if qty * entry > MAX_POSITION_VALUE:
        qty = int(MAX_POSITION_VALUE / entry)

    if qty <= 0:
        _record_outcome("Position value cap exceeded with minimum 1 share")
        return result

    # Trend gate value comes strictly from strategy (NEVER LLM output)
    signal = {
        "symbol": rec.symbol,
        "side": rec.action,
        "qty": qty,
        "price": entry,
        "ts": time.time(),
        "stop_loss": rec.stop_loss,
        "target": rec.target,
        "strategy_id": strategy_id,
        "regime": router.day_config.get("regime", "trending"),
        "trend_state": getattr(rec, "trend_state", "unavailable"),
    }

    trade_id = router.execute(signal)
    if trade_id is not None:
        _record_outcome(None, trade_id=trade_id, executed=True)
        result["qty"] = qty
        notify(
            f"AI Trader Executed {rec.action} on {rec.symbol}",
            f"Qty: {qty} @ {entry:.2f} | SL: {rec.stop_loss:.2f} | Target: {rec.target:.2f} (Conviction: {rec.conviction}/10)",
        )
    else:
        _record_outcome(router.last_rejection or "Risk kernel rejection")

    return result



def _get_live_ltp(symbol: str) -> float:
    """Fetch live LTP from Fyers, falling back to yfinance."""
    try:
        from trading.fno.fyers import FyersClient
        fyers = FyersClient()
        fyers_sym = f"NSE:{symbol}-EQ" if not symbol.startswith("NSE:") else symbol
        q = fyers.quotes([fyers_sym])
        if q and len(q) > 0 and "v" in q[0] and "lp" in q[0]["v"]:
            return float(q[0]["v"]["lp"])
    except Exception:
        pass
    try:
        t = yf.Ticker(symbol + YF_SUFFIX)
        hist = t.history(period="1d", interval="5m")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def _run_scanner_pass(
    src: ScannerSource,
    router: ExecutionRouter,
    ledger: Ledger | None,
    *,
    min_conviction: int,
    budget: int,
    dry_run: bool,
    require_entry_band: bool | None = None,
) -> tuple[list[dict], int]:
    """Work the scanner's candidate list. Returns (results, executed_count).

    Every name prints an outcome — chosen, plan, verdict, or the exact reason
    it was not bought — so a run that buys nothing says why, per name, instead
    of looking like a silent failure.
    """
    print(f"\n{'-' * 70}")
    print(f"  SOURCE: F&O LONG SCANNER — {len(src.candidates)} candidate(s) in tier")
    if dry_run:
        print("  DRY RUN — nothing reaches the risk kernel or the ledger")
    print(f"{'-' * 70}")

    for sym, why in src.skipped:
        print(f"[-] {sym:12s} SKIPPED: {why}")

    results: list[dict] = []
    executed = 0
    for cand in src.candidates:
        sym, t = cand.symbol, cand.trade
        print(f"[*] {sym:12s} {cand.verdict:5s} score {cand.score:3.0f} | plan: "
              f"entry {t.entry:.2f} in {t.entry_low:.2f}-{t.entry_high:.2f}, "
              f"stop {t.stop:.2f}, target {t.target1:.2f}", flush=True)

        if sym in router.day_config.get("blocked_symbols", []):
            print("    SKIPPED: blocked by pre-market analyst")
            continue

        # Fetch with the scanner's plan so MarketFacts carries the scanner's
        # levels; the trend state on this placeholder is overwritten below,
        # once the strategy has actually read the 200MAs.
        plan = Signal(symbol=sym, side="BUY", price=t.entry, stop_loss=t.stop,
                      target=t.target1, strategy_id=AGENT_SCANNER_STRATEGY_ID,
                      trend_state="unavailable")
        ctx = fetch_symbol_context(sym, want_news=True, plan=plan)
        if not ctx:
            print("    FAILED (no market data)")
            continue

        # The scanner computes no 200MA. Computing a real one here is what
        # makes "the council's ORB rules still apply" an evaluation rather
        # than an automatic HOLD on trend 'unavailable'.
        setup = ctx["_setup"]
        ctx["signal"] = scanner_signal(sym, setup, t)

        or_pct, vol = ctx.get("or_pct"), ctx.get("vol_ratio")
        or_txt = f"{or_pct:.2f}%" if isinstance(or_pct, (int, float)) else "n/a"
        vol_txt = f"{vol:.2f}x" if isinstance(vol, (int, float)) else "n/a"
        print(f"    real read: trend {setup['trend']['state']}, OR {or_txt}, "
              f"vol {vol_txt}, ltp {ctx.get('ltp')}")

        why = scanner_preflight(cand, ctx, require_entry_band=require_entry_band)
        if why:
            print(f"    REJECTED: {why}")
            continue

        rec = evaluate_scanner_candidate(cand, ledger=ledger, ctx=ctx)
        if not rec:
            print("    GATED (failed quant rules)")
            continue

        if dry_run:
            print(f"    DRY RUN [{rec.action} {rec.conviction}/10] — not routed")
            print(f"    Thesis: {rec.decision_rationale}")
            results.append({"symbol": sym, "action": rec.action,
                            "conviction": rec.conviction, "executed": False,
                            "trade_id": None, "reason": "dry_run",
                            "details": rec.model_dump()})
            continue

        exec_res = execute_recommendation(rec, router, min_conviction=min_conviction,
                                          strategy_id=AGENT_SCANNER_STRATEGY_ID)
        results.append(exec_res)
        if exec_res["executed"]:
            executed += 1
            print(f"    EXECUTED trade #{exec_res['trade_id']} | Qty: {exec_res['qty']} "
                  f"@ {rec.entry_price} | SL: {rec.stop_loss} | Tgt: {rec.target}")
            print(f"    Thesis: {rec.decision_rationale}")
        else:
            print(f"    [{rec.action} {rec.conviction}/10] - {exec_res['reason']}")

        if executed >= budget:
            print(f"\n[!] Reached max execution cap for this pass.")
            break

    return results, executed


def scan_and_trade(
    symbols: list[str] | None = None,
    min_conviction: int = AGENT_MIN_CONVICTION,
    max_trades: int = AGENT_MAX_TRADES_PER_SCAN,
    *,
    source: str = "watchlist",
    dry_run: bool = False,
    tiers=None,
    require_entry_band: bool | None = None,
) -> list[dict]:
    """Scan candidate symbols, run multi-agent evaluation, and execute qualifying setups.

    ``source`` picks where the names come from:
      ``"watchlist"`` (default) — today's path, unchanged;
      ``"scanner"``   — only names the F&O long scanner chose;
      ``"both"``      — scanner names first, then the watchlist.

    With ``AGENT_SCANNER_ENABLED = False`` the scanner is a strict no-op: no
    scan is run and the symbol list is exactly today's, whatever ``source``
    says. ``dry_run`` runs the whole decision path and writes no trades.
    """
    if source not in ("watchlist", "scanner", "both"):
        raise ValueError(f"source must be watchlist/scanner/both, got {source!r}")

    want_scanner = source in ("scanner", "both")
    want_watchlist = source in ("watchlist", "both")
    if want_scanner and not AGENT_SCANNER_ENABLED:
        print("[scanner] AGENT_SCANNER_ENABLED = False — ignoring the scanner "
              "source and scanning the watchlist. Flip it in trading/config.py "
              "to enable.")
        want_scanner, want_watchlist = False, True

    candidates = symbols or WATCHLIST
    router = ExecutionRouter(mode="paper", get_ltp=_get_live_ltp)
    ledger = router.ledger

    feed_status = "yfinance (Delayed fallback)"
    try:
        from trading.fno.fyers import FyersClient
        fc = FyersClient()
        connected, msg = fc.status()
        if connected:
            feed_status = f"Fyers API v3 Real-Time ({msg})"
    except Exception:
        pass

    print(f"\n{'='*70}")
    if want_watchlist:
        print(f"  AI TRADING AGENTS SCANNER: Evaluating {len(candidates)} symbols")
    else:
        print(f"  AI TRADING AGENTS SCANNER")
    print(f"  Live Market Feed: {feed_status}")
    print(f"  Market Regime: {router.day_config.get('regime')} | Risk Multiplier: {router.day_config.get('risk_multiplier')}")
    print(f"  Min Conviction: {min_conviction}/10 | Max Trades per Scan: {max_trades}")
    if source != "watchlist":
        print(f"  Candidate Source: {source}")
    print(f"{'='*70}\n")

    results: list[dict] = []
    executed_count = 0

    # --- scanner arm -----------------------------------------------------
    if want_scanner:
        from trading.fno.live import run_scan
        # One scan per pass. with_options=False skips the NIFTY chain read --
        # the agent wants stock candidates, not the chain.
        res = run_scan(with_options=False)
        src = scanner_candidates(res, tiers=tiers)
        if src.refusal:
            print(f"[scanner] REFUSED: {src.refusal}")
        else:
            scan_results, executed_count = _run_scanner_pass(
                src, router, ledger,
                min_conviction=min_conviction,
                budget=max_trades - executed_count,
                dry_run=dry_run,
                require_entry_band=require_entry_band,
            )
            results.extend(scan_results)

    # --- watchlist arm (unchanged) ---------------------------------------
    for symbol in (candidates if want_watchlist else []):
        if symbol in router.day_config.get("blocked_symbols", []):
            print(f"[-] {symbol:12s} SKIPPED: Blocked by pre-market analyst")
            continue

        print(f"[*] Evaluating {symbol:10s} ... ", end="", flush=True)
        ctx = fetch_symbol_context(symbol)
        if not ctx:
            print("FAILED (no market data)")
            continue
        if not ctx.get("signal"):
            print(f"NO SETUP (quant gate: {ctx.get('or_trigger_status', 'No trigger')}, trend: {ctx.get('trend_gate', 'N/A')})")
            continue

        rec = evaluate_stock(symbol, ledger=ledger, ctx=ctx)
        if not rec:
            print("GATED (failed quant rules)")
            continue

        if dry_run:
            # The whole decision path ran; only the order is withheld.
            print(f"DRY RUN [{rec.action} {rec.conviction}/10] — not routed")
            print(f"    Thesis: {rec.decision_rationale}")
            results.append({"symbol": symbol, "action": rec.action,
                            "conviction": rec.conviction, "executed": False,
                            "trade_id": None, "reason": "dry_run",
                            "details": rec.model_dump()})
            continue

        exec_res = execute_recommendation(rec, router, min_conviction=min_conviction)
        results.append(exec_res)

        action_tag = f"[{rec.action} {rec.conviction}/10]"
        if exec_res["executed"]:
            executed_count += 1
            print(f"EXECUTED trade #{exec_res['trade_id']} | Qty: {exec_res['qty']} @ {rec.entry_price} | SL: {rec.stop_loss} | Tgt: {rec.target}")
            print(f"    Thesis: {rec.decision_rationale}")
        else:
            print(f"{action_tag:14s} - {exec_res['reason']}")

        if executed_count >= max_trades:
            print(f"\n[!] Reached max execution cap of {max_trades} trades for this pass.")
            break

    print(f"\n{'='*70}")
    if dry_run:
        print(f"  DRY RUN COMPLETE: {len(results)} setup(s) evaluated, 0 trades written")
        print(f"  Re-run without --dry-run to route them through the risk kernel")
    else:
        print(f"  SCAN COMPLETE: {executed_count} new paper trades executed into ledger.db")
        print(f"  View live on dashboard: http://localhost:8080")
    print(f"{'='*70}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Stock Evaluator and Paper Trader")
    parser.add_argument("--symbol", type=str, help="Evaluate a specific symbol (e.g. RELIANCE)")
    parser.add_argument("--all", action="store_true", help="Scan full Nifty-50 universe")
    parser.add_argument("--min-conviction", type=int, default=AGENT_MIN_CONVICTION, help="Minimum conviction (1-10)")
    parser.add_argument("--max-trades", type=int, default=AGENT_MAX_TRADES_PER_SCAN, help="Max new trades to execute")

    # --- F&O scanner as a candidate source (AGENT_SCANNER_ENABLED must be on) ---
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--from-scanner", action="store_true",
                     help="Take candidates from the F&O long scanner only")
    src.add_argument("--with-scanner", action="store_true",
                     help="Scanner candidates first, then the watchlist")
    tiers = parser.add_mutually_exclusive_group()
    tiers.add_argument("--include-watch", dest="tiers", action="store_const",
                       const=("BUY", "WATCH"),
                       help="Include scanner WATCH names as well as BUY picks (default)")
    tiers.add_argument("--picks-only", dest="tiers", action="store_const",
                       const=("BUY",),
                       help="Only the scanner's BUY picks, no WATCH names")
    parser.set_defaults(tiers=None)
    parser.add_argument("--no-entry-band", dest="require_entry_band", action="store_false",
                        default=None,
                        help="Enter even when price is above the scanner's entry band "
                             "(i.e. allow chasing an extended setup)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the whole decision path, print the call per name, write no trades")
    args = parser.parse_args()

    symbols = None
    if args.symbol:
        symbols = [args.symbol.upper().replace(YF_SUFFIX, "")]
    elif args.all:
        symbols = NIFTY50

    source = "watchlist"
    if args.from_scanner:
        source = "scanner"
    elif args.with_scanner:
        source = "both"

    scan_and_trade(
        symbols=symbols,
        min_conviction=args.min_conviction,
        max_trades=args.max_trades,
        source=source,
        dry_run=args.dry_run,
        tiers=args.tiers,
        require_entry_band=args.require_entry_band,
    )


if __name__ == "__main__":
    main()
