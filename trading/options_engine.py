"""Options Trading Engine — converts equity & index strategy signals into option trades.

Supports:
  1. NIFTY 50 index options (weekly/monthly)
  2. BANK NIFTY index options (weekly/monthly)
  3. Stock options for F&O watchlist (RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN, etc.)

Rules:
  - BUY signal on underlying  -> BUY Call Option (CE)
  - SELL signal on underlying -> BUY Put Option (PE)
  - Directional buyers pay premium: strictly defined risk (max loss = premium paid).
  - Strike selection: At-The-Money (ATM) or nearest liquid strike.
  - Sizing: 1 lot (or sized per risk budget) based on exchange lot sizes.
  - Structural stop on option: premium at underlying's stop level, or 30% premium stop loss.
  - Target on option: premium at underlying's target 1 (1:2 R:R) and target 2 (1:3 R:R).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, date
from zoneinfo import ZoneInfo

from trading.config import (
    INDEX_LOT_SIZES,
    STOCK_LOT_SIZES,
    DEFAULT_STOCK_OPTION_LOT,
    MAX_OPTION_POSITION_VALUE,
    MAX_OPTION_RISK_PER_TRADE,
)
from trading.costs import options_round_trip
from trading.fno import options as opt_mod
from trading.fno.options import OptionQuote
from trading.fno.fyers import FyersClient

IST = ZoneInfo("Asia/Kolkata")


def get_lot_size(symbol: str) -> int:
    """Return standard contract lot size for an index or stock."""
    clean = symbol.upper().replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "").strip()
    if clean in INDEX_LOT_SIZES:
        return INDEX_LOT_SIZES[clean]
    if clean in STOCK_LOT_SIZES:
        return STOCK_LOT_SIZES[clean]
    for k, v in INDEX_LOT_SIZES.items():
        if k in clean:
            return v
    for k, v in STOCK_LOT_SIZES.items():
        if k in clean:
            return v
    return DEFAULT_STOCK_OPTION_LOT


def is_index_symbol(symbol: str) -> bool:
    clean = symbol.upper()
    return any(x in clean for x in ("NIFTY", "BANKNIFTY", "FINNIFTY", "^NSEI", "^NSEBANK"))


def get_fyers_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    if s in ("^NSEI", "NIFTY", "NIFTY50", "NIFTY 50"):
        return "NSE:NIFTY50-INDEX"
    if s in ("^NSEBANK", "BANKNIFTY", "BANK NIFTY"):
        return "NSE:NIFTYBANK-INDEX"
    if s.startswith("NSE:"):
        return s
    return f"NSE:{s}-EQ"


def build_option_order_from_plan(
    underlying: str,
    plan: opt_mod.OptionPlan,
    side: str = "BUY",
    lot_multiplier: int = 1,
) -> dict | None:
    """Build an executable option trade dict from a calculated OptionPlan."""
    if not plan or not plan.quote:
        return None
    q = plan.quote
    lot_size = get_lot_size(underlying) * max(1, lot_multiplier)
    premium = q.last_price
    if premium <= 0:
        return None

    # Calculate option stop-loss:
    # 1. Modelled stop if available
    # 2. Else 30% of premium
    stop_premium = plan.premium_at_stop if plan.premium_at_stop is not None else round(premium * 0.70, 2)
    # Ensure stop loss is below premium and > 0
    if stop_premium >= premium or stop_premium <= 0:
        stop_premium = max(0.05, round(premium * 0.70, 2))

    # Calculate option target:
    # 1. Modelled target1 if available
    # 2. Else 1:2 R:R on premium
    target_premium = plan.premium_at_t1 if plan.premium_at_t1 is not None else round(premium + 2 * (premium - stop_premium), 2)
    if target_premium <= premium:
        target_premium = round(premium + 2 * max(0.5, premium - stop_premium), 2)

    opt_symbol = q.identifier
    return {
        "symbol": opt_symbol,
        "underlying": underlying,
        "side": "BUY",  # Buyer pays premium
        "option_type": "CE" if q.option_type == "Call" else "PE",
        "strike": q.strike,
        "expiry": q.expiry,
        "qty": lot_size,
        "price": premium,
        "stop_loss": stop_premium,
        "target": target_premium,
        "is_option": True,
        "breakeven": plan.breakeven,
        "rr": plan.option_rr or round((target_premium - premium) / max(0.1, premium - stop_premium), 2),
    }


def select_option_for_underlying(
    symbol: str,
    side: str,
    spot_price: float,
    stop_loss: float | None = None,
    target: float | None = None,
    client: FyersClient | None = None,
) -> dict | None:
    """Select the best option (Call for BUY, Put for SELL) for an underlying stock or index.
    
    Returns an option order dictionary ready for execution, or None.
    """
    fyers_sym = get_fyers_symbol(symbol)
    c = client or FyersClient()
    chain_data = c.option_chain(fyers_sym, strikecount=8)
    if not chain_data:
        return None

    chain = chain_data.get("optionsChain") or []
    expiry_data = chain_data.get("expiryData") or []
    front_expiry = expiry_data[0].get("date") if expiry_data else ""
    now = datetime.now(IST)

    quotes: list[OptionQuote] = []
    spot_from_chain = None
    for it in chain:
        if it.get("strike_price") == -1:
            spot_from_chain = float(it.get("ltp") or 0)
            break
    spot = spot_price or spot_from_chain or 1.0

    for it in chain:
        strike = it.get("strike_price")
        opt_type = it.get("option_type")
        if strike is None or strike == -1 or not opt_type:
            continue
        sym_id = it.get("symbol", "")
        kind = "Call" if opt_type == "CE" else "Put"
        ltp = float(it.get("ltp") or 0)
        oi = float(it.get("oi") or 0)
        vol = float(it.get("volume") or 0)
        pchange = float(it.get("ltpchp") or 0)
        from trading.fno.models import Provenance
        quotes.append(OptionQuote(
            underlying=symbol,
            identifier=sym_id,
            option_type=kind,
            strike=float(strike),
            expiry=front_expiry or "near",
            last_price=ltp,
            pct_change=pchange,
            open_interest=oi,
            volume=vol,
            underlying_value=spot,
            prov=Provenance("fyers", sym_id, "option_chain", now, now)
        ))

    if not quotes:
        return None

    # Calculate default targets if not provided
    is_buy = side.upper() in ("BUY", "LONG", "CALL", "CE")
    if is_buy:
        sl = stop_loss if stop_loss is not None else round(spot * 0.99, 2)
        risk = max(0.5, spot - sl)
        tg1 = target if target is not None else round(spot + 2 * risk, 2)
        tg2 = round(spot + 3 * risk, 2)
        plan = opt_mod.plan_call(symbol, quotes, spot=spot, target1=tg1, target2=tg2, stop=sl, now=now, today=now.date())
    else:
        sl = stop_loss if stop_loss is not None else round(spot * 1.01, 2)
        risk = max(0.5, sl - spot)
        tg1 = target if target is not None else round(spot - 2 * risk, 2)
        tg2 = round(spot - 3 * risk, 2)
        plan = opt_mod.plan_put(symbol, quotes, spot=spot, target1=tg1, target2=tg2, stop=sl, now=now, today=now.date())

    # Fallback to ATM strike if strict plan checks filtered everything
    if not plan.quote:
        target_kind = "Call" if is_buy else "Put"
        matching = [q for q in quotes if q.option_type == target_kind and q.last_price > 0]
        if matching:
            # Pick strike closest to spot
            matching.sort(key=lambda q: abs(q.strike - spot))
            best = matching[0]
            plan.quote = best
            plan.breakeven = best.strike + best.last_price if is_buy else best.strike - best.last_price

    return build_option_order_from_plan(symbol, plan, side=side)
