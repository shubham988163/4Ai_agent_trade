"""NIFTY 50 index options — a directional call on the index itself.

This is a different instrument from the stock scanner, and deliberately a
separate module, because **an index has no volume**. NSE publishes no traded
quantity for NIFTY spot, so three of the stock scanner's confirmations simply
do not exist here:

    VWAP · relative volume · breakout-bar volume expansion

Removing three of nine checks makes an index signal weaker evidence than a
stock one, not merely different. What is left is price structure (the opening
range, the EMA stack) plus the market context the scanner already computes —
breadth and sector participation — which for the index *is* the instrument
rather than background. The output says so rather than presenting a
five-factor read as if it were eight.

Only long/call setups are produced, matching the scanner's scope. A NIFTY
breakdown would be a put trade and is not inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from trading.fno import config as C
from trading.fno import indicators as ind
from trading.fno import options as opt_mod
from trading.fno.models import MarketContext

# The index needs a cleaner break than a stock: with no volume to confirm it,
# structure has to carry more weight.
BREAKOUT_BUFFER_ATR = 0.10
EXTENDED_ATR = 1.25


@dataclass
class IndexPlan:
    """An option plan on the index (Call or Put), or the reason there is not one."""
    spot: float
    or_high: float
    or_low: float
    atr: float
    ema_fast: float | None
    ema_slow: float | None
    day_high: float
    day_low: float
    breakout: bool
    retested: bool
    holding: bool
    extended: bool
    name: str = "NIFTY 50"
    index_symbol: str = "NIFTY"
    direction: str = "CALL"               # "CALL" or "PUT"
    entry_low: float | None = None
    entry_high: float | None = None
    stop: float | None = None
    target1: float | None = None
    target2: float | None = None
    option: object | None = None          # options.OptionPlan
    reasons: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        tag = "BREAKDOWN" if self.direction == "PUT" else "BREAKOUT"
        if not self.breakout:
            return f"NO {tag}"
        if self.extended:
            return f"{tag} — EXTENDED"
        if self.retested and self.holding:
            return f"{tag} → RETEST → HOLD"
        return f"{tag} (no retest yet)" if self.holding else f"{tag} LOST"

    @property
    def tradeable(self) -> bool:
        return (not self.rejections and self.option is not None
                and getattr(self.option, "tradeable", False))


def analyse(df: pd.DataFrame, now: datetime, market: MarketContext,
            chain: list, *, index_symbol: str = "NIFTY",
            name: str = "NIFTY 50", side: str | None = None) -> IndexPlan | None:
    """Read index opening-range structure and price a Call or Put option on it."""
    bars = ind.closed_bars(df, now)
    if bars.empty:
        return None
    day = ind.sessions(bars)[-1]
    if day != now.date():
        return None
    today = ind.session_of(bars, day)
    or_high, or_low, or_bars = ind.opening_range(today)
    if or_bars == 0 or or_high != or_high:
        return None

    atr = float(ind.atr(bars).iloc[-1])
    if not atr or atr != atr:
        return None
    closes = bars["Close"].astype(float)
    ema_f = float(ind.ema(closes, C.EMA_FAST).iloc[-1]) if len(closes) >= C.EMA_FAST else None
    ema_s = float(ind.ema(closes, C.EMA_SLOW).iloc[-1]) if len(closes) >= C.EMA_SLOW else None
    spot = float(today["Close"].iloc[-1])
    post = ind.after_opening_range(today)

    # Determine whether we evaluate a PUT breakdown or CALL breakout
    is_put = (side == "PUT" or (side is None and market.classification in ("BEARISH", "STRONGLY BEARISH") and spot < or_low))
    direction = "PUT" if is_put else "CALL"

    if is_put:
        trigger = or_low - BREAKOUT_BUFFER_ATR * atr
        broke = post[post["Close"] < trigger]
        breakout = not broke.empty
        holding = breakout and spot < or_low
        after = post[post.index >= broke.index[0]] if breakout else post.iloc[0:0]
        band = or_low - C.RETEST_TOLERANCE_ATR * atr
        touched = after.iloc[1:][(after.iloc[1:]["High"] >= band)
                                 & (after.iloc[1:]["Close"] < or_low)] if len(after) > 1 else after.iloc[0:0]
        retested = not touched.empty
        extended = breakout and (or_low - spot) > EXTENDED_ATR * atr

        plan = IndexPlan(spot=spot, or_high=or_high, or_low=or_low, atr=atr,
                         ema_fast=ema_f, ema_slow=ema_s,
                         day_high=float(today["High"].max()),
                         day_low=float(today["Low"].min()),
                         breakout=breakout, retested=retested, holding=holding,
                         extended=extended, name=name, index_symbol=index_symbol,
                         direction="PUT")
        plan.caveats.append("an index publishes no volume, so VWAP, relative volume "
                            "and breakout-volume confirmation are unavailable")

        if not breakout:
            plan.rejections.append(f"{name} has not closed below its 09:15–09:30 low ({or_low:.2f})")
            return plan
        if not holding:
            plan.rejections.append(f"the breakdown of {or_low:.2f} is not holding")
            return plan
        if extended:
            plan.rejections.append(f"{name} is {or_low - spot:.0f} points below the level — chase")
        if market.classification in ("BULLISH", "STRONGLY BULLISH"):
            plan.rejections.append(f"market context is {market.classification} — put fights the tape")
        if ema_f and ema_s and ema_f > ema_s:
            plan.rejections.append("the 20 EMA is above the 50 EMA — trend does not support a short")

        entry_high, entry_low = or_low, or_low - 0.40 * atr
        stop_base = (float(touched["High"].max()) if retested
                     else float(after["High"].iloc[0]) if len(after) else or_high)
        stop = stop_base + C.SL_BUFFER_ATR * atr
        entry = max(min(spot, entry_high), entry_low)
        risk = stop - entry
        if risk <= 0:
            plan.rejections.append("no structural stop above the entry")
            return plan
        plan.entry_low, plan.entry_high = entry_low, entry_high
        plan.stop = stop
        plan.target1 = entry - C.TARGET_RR_1 * risk
        plan.target2 = entry - C.TARGET_RR_2 * risk

        plan.reasons.append(f"closed below the 09:15–09:30 low {or_low:.2f}"
                            + (" and retested it" if retested else " (no retest yet)"))
        if ema_f and ema_s and spot < ema_f < ema_s:
            plan.reasons.append(f"EMA stack is bearish — {spot:.0f} < {ema_f:.0f} < {ema_s:.0f}")

        plan.option = opt_mod.plan_put(index_symbol, chain, spot=spot,
                                       target1=plan.target1, target2=plan.target2,
                                       stop=plan.stop, now=now, today=now.date())
        return plan

    # Call breakout logic
    trigger = or_high + BREAKOUT_BUFFER_ATR * atr
    broke = post[post["Close"] > trigger]
    breakout = not broke.empty
    holding = breakout and spot > or_high
    after = post[post.index >= broke.index[0]] if breakout else post.iloc[0:0]
    band = or_high + C.RETEST_TOLERANCE_ATR * atr
    touched = after.iloc[1:][(after.iloc[1:]["Low"] <= band)
                             & (after.iloc[1:]["Close"] > or_high)] if len(after) > 1 else after.iloc[0:0]
    retested = not touched.empty
    extended = breakout and (spot - or_high) > EXTENDED_ATR * atr

    plan = IndexPlan(spot=spot, or_high=or_high, or_low=or_low, atr=atr,
                     ema_fast=ema_f, ema_slow=ema_s,
                     day_high=float(today["High"].max()),
                     day_low=float(today["Low"].min()),
                     breakout=breakout, retested=retested, holding=holding,
                     extended=extended, name=name, index_symbol=index_symbol,
                     direction="CALL")
    plan.caveats.append("an index publishes no volume, so VWAP, relative volume "
                        "and breakout-volume confirmation are unavailable — this "
                        "is a five-factor read, not the stock scanner's eight")

    if not breakout:
        plan.rejections.append(
            f"{name} has not closed above its 09:15–09:30 high ({or_high:.2f})")
        return plan
    if not holding:
        plan.rejections.append(f"the break of {or_high:.2f} is not holding")
        return plan
    if extended:
        plan.rejections.append(
            f"{name} is {spot - or_high:.0f} points above the level — more than "
            f"{EXTENDED_ATR:g} ATR, so this is a chase")
    if market.classification in ("BEARISH", "STRONGLY BEARISH"):
        plan.rejections.append(
            f"market context is {market.classification} — a long on the index "
            "fights the tape it is made of")
    if ema_f and ema_s and ema_f < ema_s:
        plan.rejections.append("the 20 EMA is below the 50 EMA — the trend does "
                               "not support a long")

    # Levels, on the same rules the stock setups use.
    entry_low, entry_high = or_high, or_high + 0.40 * atr
    stop_base = (float(touched["Low"].min()) if retested
                 else float(after["Low"].iloc[0]) if len(after) else or_low)
    stop = stop_base - C.SL_BUFFER_ATR * atr
    entry_low = max(entry_low, stop + C.ENTRY_FLOOR_ATR * atr)
    entry = min(max(spot, entry_low), entry_high)
    risk = entry - stop
    if risk <= 0:
        plan.rejections.append("no structural stop below the entry")
        return plan
    plan.entry_low, plan.entry_high = entry_low, entry_high
    plan.stop = stop
    plan.target1 = entry + C.TARGET_RR_1 * risk
    plan.target2 = entry + C.TARGET_RR_2 * risk

    plan.reasons.append(
        f"closed above the 09:15–09:30 high {or_high:.2f}"
        + (" and retested it" if retested else " (no retest yet)"))
    if ema_f and ema_s and spot > ema_f > ema_s:
        plan.reasons.append(f"EMA stack is bullish — {spot:.0f} > {ema_f:.0f} > {ema_s:.0f}")
    if market.breadth_pct is not None:
        plan.reasons.append(f"breadth {market.breadth_pct:.0f}% green")
    strong = [k for k, v in (market.sector_pct or {}).items() if v > 0.3]
    if strong:
        plan.reasons.append("sectors participating: " + ", ".join(sorted(strong)[:4]))

    plan.option = opt_mod.plan_call(index_symbol, chain, spot=spot,
                                    target1=plan.target1, target2=plan.target2,
                                    stop=plan.stop, now=now, today=now.date())
    return plan


def render(plan: IndexPlan | None) -> str:
    """Terminal block for the index option setup."""
    if plan is None:
        return ("NIFTY 50 OPTIONS\n  no intraday index candles for today — "
                "nothing to read")
    kind = "CE" if plan.direction == "CALL" else "PE"
    kind_label = "Call option" if plan.direction == "CALL" else "Put option"
    L = [f"{plan.name.upper()} OPTIONS",
         f"  Spot:              {plan.spot:.2f}",
         f"  09:15–09:30:       {plan.or_low:.2f} – {plan.or_high:.2f}   "
         f"(ATR {plan.atr:.2f}, day {plan.day_low:.2f}–{plan.day_high:.2f})",
         f"  Structure:         {plan.status}"]
    if plan.ema_fast and plan.ema_slow:
        L.append(f"  EMA 20/50:         {plan.ema_fast:.2f} / {plan.ema_slow:.2f}")
    if plan.target1:
        L += [f"  Index entry:       {plan.entry_low:.2f} – {plan.entry_high:.2f}",
              f"  Index stop:        {plan.stop:.2f}",
              f"  Index targets:     {plan.target1:.2f} (1:2) / {plan.target2:.2f} (1:3)"]
    o = plan.option
    if o is None:
        L.append(f"  {kind_label}:       not evaluated — no index levels to price against")
    elif o.quote is None:
        L.append(f"  {kind_label}:       NO {kind} — "
                 + (o.rejections[0] if o.rejections else "no candidate strike"))
    else:
        q = o.quote
        L += [f"  {kind_label}:       {q.strike:g} {kind} @ {q.last_price:.2f} "
              f"({q.expiry}, {q.moneyness})",
              f"    Breakeven:       {o.breakeven:.2f} if held to expiry",
              f"    Contract:        OI {q.open_interest:,.0f}, "
              f"volume {q.volume:,.0f}"]
        if o.premium_at_t1 is not None:
            L.append(f"    On the move:     pay {q.last_price:.2f} → "
                     f"{o.premium_at_t1:.2f} at T1, {o.premium_at_t2:.2f} at T2"
                     + (f", {o.premium_at_stop:.2f} at the index stop"
                        if o.premium_at_stop is not None else ""))
            if o.option_rr is not None:
                L.append(f"    Option R:R:      {o.option_rr:.2f}:1")
        L += [f"    ! {w}" for w in o.warnings]
    if plan.reasons:
        L.append("  Why:")
        L += [f"    • {r}" for r in plan.reasons]
    if plan.rejections:
        L.append("  Blocking:")
        L += [f"    ✗ {r}" for r in plan.rejections]
    L.append("  Caveat:")
    L += [f"    ! {c}" for c in plan.caveats]
    return "\n".join(L)
