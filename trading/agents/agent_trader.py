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
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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



def fetch_symbol_context(symbol: str) -> dict | None:
    """Fetch market data, news, fundamentals, and compute ORB + Multi-TF 200MA indicators via MarketFacts."""
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
        sig = setup["signal"]

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

        if sig is not None:
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
                    mkt_dict["direction"] = "supports" if sig and sig.side == "BUY" else ("opposes" if sig and sig.side == "SELL" else "neutral")
                elif n_chg < -0.2:
                    m_label = "BEARISH"
                    mkt_dict["direction"] = "supports" if sig and sig.side == "SELL" else ("opposes" if sig and sig.side == "BUY" else "neutral")
                else:
                    m_label = "RANGEBOUND"
                    mkt_dict["direction"] = "neutral"
                market_direction = f"Nifty 50 {n_chg:+.2f}% ({m_label})"
        except Exception:
            pass

        # Build MarketFacts directly from strategy computed values
        facts = build_market_facts(
            symbol=symbol,
            strategy_setup=setup,
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
            "or_trigger_status": "BREAKOUT" if sig else ("INSIDE_OR" if not setup["orb"]["or_valid"] else "PENDING"),
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
        }
    except Exception as exc:
        print(f"[{symbol}] Failed to fetch market data: {exc}", file=sys.stderr)
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

    # Code-level post-council enforcement:
    # 1. Trend state MUST come from the strategy, NEVER from the LLM.
    rec.trend_state = trend_state
    rec.candidate_id = candidate_id
    rec.entry_price = sig.price
    rec.stop_loss = sig.stop_loss
    rec.target = sig.target

    # 2. Side constraint: if LLM flipped the direction opposite to candidate_side, coerce to HOLD
    if candidate_side == "BUY" and rec.action == "SELL":
        rec.action = "HOLD"
        rec.conviction = 1
        rec.decision_rationale += " [Code Override: Cannot SELL on a BUY breakout candidate]"
    elif candidate_side == "SELL" and rec.action == "BUY":
        rec.action = "HOLD"
        rec.conviction = 1
        rec.decision_rationale += " [Code Override: Cannot BUY on a SELL breakdown candidate]"

    return rec


def execute_recommendation(
    rec: AgentTradeRecommendation,
    router: ExecutionRouter,
    min_conviction: int = AGENT_MIN_CONVICTION,
) -> dict:
    """Route an approved recommendation through ExecutionRouter."""
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
        "strategy_id": "orb_ma200_agents",
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


def scan_and_trade(
    symbols: list[str] | None = None,
    min_conviction: int = AGENT_MIN_CONVICTION,
    max_trades: int = AGENT_MAX_TRADES_PER_SCAN,
) -> list[dict]:
    """Scan candidate symbols, run multi-agent evaluation, and execute qualifying setups."""
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
    print(f"  AI TRADING AGENTS SCANNER: Evaluating {len(candidates)} symbols")
    print(f"  Live Market Feed: {feed_status}")
    print(f"  Market Regime: {router.day_config.get('regime')} | Risk Multiplier: {router.day_config.get('risk_multiplier')}")
    print(f"  Min Conviction: {min_conviction}/10 | Max Trades per Scan: {max_trades}")
    print(f"{'='*70}\n")


    results = []
    executed_count = 0

    for symbol in candidates:
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
    args = parser.parse_args()

    symbols = None
    if args.symbol:
        symbols = [args.symbol.upper().replace(YF_SUFFIX, "")]
    elif args.all:
        symbols = NIFTY50

    scan_and_trade(
        symbols=symbols,
        min_conviction=args.min_conviction,
        max_trades=args.max_trades,
    )


if __name__ == "__main__":
    main()
