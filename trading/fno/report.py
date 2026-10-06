"""Rendering: the §16 candidate block, the §19 alert, and the §17 no-trade line.

Reasons and invalidation text are generated from the same objects the score was
computed from, so the prose cannot drift from the numbers — if a reason says
"OI up 4.2%", that string came out of the OI read that scored the setup.
"""
from __future__ import annotations

from trading.fno import config as C
from trading.fno.models import Candidate, ScanResult

NO_TRADE = "NO HIGH-QUALITY LONG SETUP"


def _f(x, nd: int = 2, dash: str = "n/a") -> str:
    return dash if x is None else f"{x:,.{nd}f}"


def volume_status(cand: Candidate) -> str:
    st, lv = cand.structure, cand.levels
    bits = []
    if st.breakout_vol_mult is not None:
        word = ("expansion" if st.breakout_vol_mult >= C.BREAKOUT_VOL_MULT
                else "adequate" if st.breakout_vol_mult >= C.WEAK_VOL_MULT
                else "WEAK")
        bits.append(f"breakout bar {st.breakout_vol_mult:.1f}x prior bars ({word})")
    bits.append(f"RVOL {_f(lv.rvol, 2)}" if lv.rvol is not None
                else "RVOL unavailable")
    bits.append(f"last bar {lv.last_bar_vol:,.0f} vs session avg "
                f"{lv.avg_bar_vol:,.0f}")
    return "; ".join(bits)


def rel_strength_text(cand: Candidate) -> str:
    rs = cand.rel_strength
    word = "stronger than" if rs > 0.1 else "in line with" if rs > -0.1 else "weaker than"
    out = f"{rs:+.2f}% vs NIFTY ({word} the index)"
    if cand.sector_strength is not None:
        out += f"; sector {cand.sector} median {cand.sector_strength:+.2f}% (peer proxy)"
    return out


def reasons(cand: Candidate) -> list[str]:
    lv, st = cand.levels, cand.structure
    out: list[str] = []
    if st.breakout:
        out.append(f"closed above the 09:15–09:30 resistance at {lv.resistance:.2f}"
                   + (f" at {st.breakout_time:%H:%M}" if st.breakout_time else "")
                   + (" and is holding above it" if st.holding else ""))
    if st.breakout_vol_mult and st.breakout_vol_mult >= C.WEAK_VOL_MULT:
        out.append(f"breakout carried {st.breakout_vol_mult:.1f}x the prior-bar "
                   f"volume" + (f", RVOL {lv.rvol:.2f}x" if lv.rvol else ""))
    if st.retested and st.holding:
        out.append(f"retest into {_f(st.retest_low)} held the level and closed back above")
    if lv.price > lv.vwap:
        slope = ("rising" if lv.vwap_slope > 0.01
                 else "flat" if lv.vwap_slope > -0.01 else "falling")
        out.append(f"price {lv.price:.2f} is above a {slope} VWAP ({lv.vwap:.2f})")
    if lv.ema_fast and lv.ema_slow and lv.price > lv.ema_fast > lv.ema_slow:
        out.append(f"EMA stack is bullish — price > 20 EMA ({lv.ema_fast:.2f}) "
                   f"> 50 EMA ({lv.ema_slow:.2f})")
    if cand.oi.bullish:
        out.append(f"futures OI: {cand.oi.classification} — {cand.oi.detail}")
    if cand.rel_strength > 0.3:
        out.append(f"outperforming NIFTY by {cand.rel_strength:+.2f}%")
    if st.higher_highs and st.higher_lows:
        out.append("higher highs and higher lows on the 5-minute chart")
    return out[:5]


def invalidation(cand: Candidate) -> list[str]:
    lv, st = cand.levels, cand.structure
    out = [f"a 5-minute close back below the breakout level {lv.resistance:.2f}"]
    if cand.trade:
        out.append(f"price losing the stop at {cand.trade.stop:.2f} (structure broken)")
    out.append(f"loss of VWAP ({lv.vwap:.2f}) — §8 setup score no longer holds")
    if cand.oi.classification == "SHORT COVERING":
        out.append("OI turning down further while price stalls (covering exhausted, "
                   "no fresh longs)")
    else:
        out.append("OI rolling over into long unwinding while price fades")
    if st.retest_low:
        out.append(f"a break of the retest low {st.retest_low:.2f}")
    return out


def option_lines(cand: Candidate) -> list[str]:
    """How the long could be expressed in calls — or why it should not be."""
    plan = cand.option_plan
    if plan is None:
        return []
    if plan.quote is None:
        return ["Option (call):       none — "
                + (plan.rejections[0] if plan.rejections else "no candidate strike")]
    q, t = plan.quote, cand.trade
    out = [f"Option (call):       {q.strike:g} CE @ {q.last_price:.2f}  "
           f"({q.expiry}, {q.moneyness})",
           f"  Breakeven:         {plan.breakeven:.2f} — "
           + (f"below target 1 ({t.target1:.2f}), so T1 pays"
              if plan.clears_t1 else
              f"ABOVE target 1 ({t.target1:.2f}); only the 1:3 target pays"
              if plan.clears_t2 else "above both targets"),
           f"  Contract:          OI {q.open_interest:,.0f}, "
           f"volume {q.volume:,.0f}"]
    if plan.premium_at_t1 is not None:
        out.append(f"  Option plan:       pay {q.last_price:.2f} → "
                   f"{plan.premium_at_t1:.2f} at T1, {plan.premium_at_t2:.2f} at T2"
                   + (f", {plan.premium_at_stop:.2f} at the stock's stop"
                      if plan.premium_at_stop is not None else ""))
        rr = plan.option_rr
        if rr is not None:
            out.append(f"  Option R:R:        {rr:.2f}:1 on the option "
                       f"(the stock trade is 1:{t.rr1:g})")
        out.append(f"  Model:             Black-Scholes at "
                   f"{plan.implied_vol * 100:.0f}% IV, IV assumed unchanged, ~2h "
                   "hold — an estimate, not a quote")
    out += [f"  ! {w}" for w in plan.warnings]
    return out


def candidate_block(cand: Candidate, rank: int, market_label: str,
                    when=None) -> str:
    lv, st, fut = cand.levels, cand.structure, cand.fut
    L = [
        f"Rank:                {rank}",
        f"Stock:               {cand.symbol}",
        f"Futures Contract:    {fut.contract}"
        + (f"  (expiry {fut.expiry})" if fut.expiry else ""),
        f"Current Price:       {lv.price:.2f}"
        + (f"  ({lv.pct_change:+.2f}%)" if lv.pct_change is not None else ""),
        f"Market Trend:        {market_label}",
        f"Sector:              {cand.sector}"
        + (f"  ({cand.sector_strength:+.2f}% peer median)"
           if cand.sector_strength is not None else "  (unmapped)"),
        f"9:15–9:30 High:      {lv.or_high:.2f}",
        f"9:15–9:30 Low:       {lv.or_low:.2f}",
        f"VWAP:                {lv.vwap:.2f}  (slope {lv.vwap_slope:+.3f}%/bar)",
        f"20 EMA:              {_f(lv.ema_fast)}",
        f"50 EMA:              {_f(lv.ema_slow)}",
        f"Volume Status:       {volume_status(cand)}",
        f"OI Change:           {_f(fut.oi_change_pct)}%"
        + (f"  (OI {fut.open_interest:,.0f})" if fut.open_interest else ""),
        f"OI Interpretation:   {cand.oi.classification} — {cand.oi.detail}",
        f"Prev Day H/L/C:      {_f(lv.prev_high)} / {_f(lv.prev_low)} / "
        f"{_f(lv.prev_close)}",
        f"Intraday S/R:        support {_f(lv.support)} / resistance "
        f"{lv.resistance:.2f}   (day range {lv.day_low:.2f}–{lv.day_high:.2f}, "
        f"ATR {lv.atr:.2f})",
        f"Breakout Status:     {st.status}",
        f"Retest Status:       " + ("held" if st.retested and st.holding
                                    else "retested and lost" if st.retested
                                    else "no retest yet"),
        f"Relative Strength:   {rel_strength_text(cand)}",
        f"Score:               {cand.score:.0f}/100  "
        + " ".join(f"{k}={v:g}" for k, v in cand.scores.items()),
        f"Setup Grade:         {cand.grade}",
        "",
    ]
    if cand.trade:
        t = cand.trade
        L += [
            f"Entry Zone:          {t.entry_low:.2f} – {t.entry_high:.2f}"
            f"   (ref {t.entry:.2f})",
            f"Stop Loss:           {t.stop:.2f}   ({t.stop_basis}, "
            f"risk {t.risk:.2f} = {t.risk / t.entry * 100:.2f}%)",
            f"Target 1:            {t.target1:.2f}   (1:{t.rr1:g})",
            f"Target 2:            {t.target2:.2f}   (1:{t.rr2:g})",
            f"Risk/Reward:         1:{t.rr1:g} to 1:{t.rr2:g}"
            + (f"   [next resistance {lv.overhead:.2f}]" if lv.overhead else ""),
        ]
    else:
        L.append("Entry Zone:          NONE — no valid entry (see rejections)")
    L += option_lines(cand)
    L.append("")
    L.append("Reason:")
    L += [f"  • {r}" for r in reasons(cand)] or ["  • (none)"]
    L.append("")
    L.append("Invalidation:")
    L += [f"  • {r}" for r in invalidation(cand)]
    if cand.warnings:
        L.append("")
        L.append("Data caveats:")
        L += [f"  ! {w}" for w in cand.warnings]
    if cand.rejections:
        L.append("")
        L.append("Rejected because:")
        L += [f"  ✗ {r}" for r in cand.rejections]
    L.append("")
    L.append("Sources:")
    L += [f"  – {p.label(when)}" for p in cand.sources]
    return "\n".join(L)


def alert(cand: Candidate, when=None) -> str:
    t = cand.trade
    if t is None:
        return ""
    conf = ("High" if cand.score >= 90 and not cand.warnings
            else "Medium" if cand.score >= 80 else "Low")
    lines = [
        "🚨 NSE F&O LONG SETUP",
        "",
        f"Stock:   {cand.symbol}  ({cand.fut.contract})",
        f"Price:   ₹{cand.levels.price:,.2f}",
        f"Score:   {cand.score:.0f}/100",
        f"Grade:   {cand.grade}",
        "",
        "WHY:",
    ]
    lines += [f"• {r}" for r in reasons(cand)]
    lines += [
        "",
        f"ENTRY:\n₹{t.entry_low:,.2f} – ₹{t.entry_high:,.2f}",
        "",
        f"STOP:\n₹{t.stop:,.2f}",
        "",
        f"TARGET:\n₹{t.target1:,.2f}\n₹{t.target2:,.2f}",
        "",
        f"R:R:\n1:{t.rr1:g} to 1:{t.rr2:g}",
        "",
        "INVALIDATION:",
    ]
    lines += [f"• {r}" for r in invalidation(cand)[:3]]
    lines += [
        "",
        f"CONFIDENCE:\n{conf}  (signal agreement across factors — not a "
        "probability of profit)",
        "",
        "DATA TIME:",
    ]
    lines += [f"{p.label(when)}" for p in cand.sources]
    return "\n".join(lines)


def as_dict(res: ScanResult) -> dict:
    """The whole scan as plain data — the one payload the CLI's --json and the
    web UI both render, so the two can never disagree about a level."""
    return {
        "when": res.when.isoformat(),
        "when_label": f"{res.when:%d %b %Y, %H:%M:%S} IST",
        "window": res.window,
        "signals_allowed": res.signals_allowed,
        "window_note": res.window_note,
        "data_ok": res.data_ok,
        "stale_banner": None if res.data_ok else
        "REAL-TIME DATA UNAVAILABLE — SCAN NOT RELIABLE.",
        # Surfaced separately from data_notes because a loosened bar has to be
        # impossible to miss: reading a relaxed board as a strict one is exactly
        # the mistake this flag exists to prevent.
        "relaxed_banner": None if not C.RELAXED else
        ("RELAXED GRADING — OI-flat setups shown as WATCH instead of rejected. "
         "The score floor is unchanged and nothing relaxed is a BUY."),
        "data_notes": res.data_notes,
        "considered": res.considered,
        "market": {
            "classification": res.market.classification,
            "nifty_pct": res.market.nifty_pct,
            "banknifty_pct": res.market.banknifty_pct,
            "nifty_above_vwap": res.market.nifty_above_vwap,
            "breadth_pct": res.market.breadth_pct,
            "advances": res.market.advances,
            "declines": res.market.declines,
            "breadth_source": res.market.breadth_source,
            "sectors": res.market.sector_pct,
            "sector_source": res.market.sector_source,
            "sources": [_prov(p, res.when) for p in res.market.sources],
        },
        "index_options": _index_plan(res.index_plan),
        "banknifty_options": _index_plan(getattr(res, "banknifty_plan", None)),
        "chain": _chain_read(res.chain_read),
        "pivots": _pivots(res.pivots, res.gap),
        "picks": [_cand(c, res.when, i + 1) for i, c in enumerate(res.picks)],
        "candidates": [_cand(c, res.when) for c in res.candidates],
        "no_trade": NO_TRADE if not res.picks else None,
    }


def _chain_read(r) -> dict | None:
    if r is None:
        return None
    return {"underlying": r.underlying, "expiry": r.expiry, "spot": r.spot,
            "days_to_expiry": r.days_to_expiry, "call_oi": r.call_oi,
            "put_oi": r.put_oi, "pcr": r.pcr, "pcr_label": r.pcr_label,
            "max_pain": r.max_pain, "max_pain_useful": r.max_pain_useful,
            "resistance_strike": r.resistance_strike, "resistance_oi": r.resistance_oi,
            "support_strike": r.support_strike, "support_oi": r.support_oi,
            "strikes": r.strikes, "notes": r.notes}


def _pivots(p, g) -> dict | None:
    if p is None:
        return None
    out = {"pivot": p.pivot, "bc": p.bc, "tc": p.tc,
           "cpr_width_pct": p.cpr_width_pct(), "cpr_label": p.cpr_label,
           "r1": p.r1, "r2": p.r2, "r3": p.r3,
           "s1": p.s1, "s2": p.s2, "s3": p.s3,
           "h3": p.h3, "h4": p.h4, "l3": p.l3, "l4": p.l4,
           "prev_high": p.prev_high, "prev_low": p.prev_low,
           "prev_close": p.prev_close, "gap": None}
    if g is not None:
        out["gap"] = {"open": g.open_price, "prev_close": g.prev_close,
                      "gap_pct": g.gap_pct, "filled": g.filled, "label": g.label}
    return out


def _index_plan(plan) -> dict | None:
    """The NIFTY index-option read as plain data."""
    if plan is None:
        return None
    o = plan.option
    return {
        "spot": plan.spot, "or_high": plan.or_high, "or_low": plan.or_low,
        "atr": plan.atr, "ema20": plan.ema_fast, "ema50": plan.ema_slow,
        "day_high": plan.day_high, "day_low": plan.day_low,
        "status": plan.status, "tradeable": plan.tradeable,
        "entry_low": plan.entry_low, "entry_high": plan.entry_high,
        "stop": plan.stop, "target1": plan.target1, "target2": plan.target2,
        "reasons": plan.reasons, "rejections": plan.rejections,
        "caveats": plan.caveats,
        "option": None if o is None else {
            "tradeable": o.tradeable,
            "quote": None if o.quote is None else {
                "strike": o.quote.strike, "expiry": o.quote.expiry,
                "ltp": o.quote.last_price, "moneyness": o.quote.moneyness,
                "open_interest": o.quote.open_interest, "volume": o.quote.volume,
            },
            "breakeven": o.breakeven, "premium_at_stop": o.premium_at_stop,
            "premium_at_t1": o.premium_at_t1, "premium_at_t2": o.premium_at_t2,
            "decay": o.decay,
            "option_rr": o.option_rr, "implied_vol": o.implied_vol,
            "rejections": o.rejections, "warnings": o.warnings,
        },
    }


def _options(c: Candidate) -> dict | None:
    """The call plan as plain data — None when options were not looked at."""
    plan = c.option_plan
    if plan is None:
        return None

    def q(x):
        return None if x is None else {
            "identifier": x.identifier, "strike": x.strike,
            "type": x.option_type, "expiry": x.expiry, "ltp": x.last_price,
            "pct_change": x.pct_change, "open_interest": x.open_interest,
            "volume": x.volume, "moneyness": x.moneyness,
            "breakeven": x.breakeven,
        }
    return {
        "tradeable": plan.tradeable,
        "quote": q(plan.quote),
        "breakeven": plan.breakeven,
        "clears_t1": plan.clears_t1,
        "clears_t2": plan.clears_t2,
        "implied_vol": plan.implied_vol,
        "premium_at_stop": plan.premium_at_stop,
        "premium_at_t1": plan.premium_at_t1,
        "premium_at_t2": plan.premium_at_t2,
        "option_rr": plan.option_rr,
        "decay": plan.decay,
        "rejections": plan.rejections,
        "warnings": plan.warnings,
        "chain": [q(x) for x in plan.considered],
    }


def _prov(p, when) -> dict:
    return {"label": p.label(when), "source": p.source, "instrument": p.instrument,
            "timeframe": p.timeframe, "data_time": p.data_time.isoformat(),
            "age_min": round(p.age_min(when), 1), "delayed": p.delayed}


def _cand(c: Candidate, when, rank: int | None = None) -> dict:
    lv, st, t = c.levels, c.structure, c.trade
    return {
        "rank": rank,
        "symbol": c.symbol,
        "sector": c.sector,
        "contract": c.fut.contract,
        "expiry": c.fut.expiry,
        "verdict": c.verdict,
        "score": c.score,
        "grade": c.grade,
        "components": c.scores,
        "tradeable": c.tradeable,
        "price": lv.price,
        "pct_change": lv.pct_change,
        "or_high": lv.or_high, "or_low": lv.or_low,
        "vwap": lv.vwap, "vwap_slope": lv.vwap_slope,
        "ema20": lv.ema_fast, "ema50": lv.ema_slow,
        "atr": lv.atr,
        "prev_high": lv.prev_high, "prev_low": lv.prev_low,
        "prev_close": lv.prev_close,
        "day_high": lv.day_high, "day_low": lv.day_low,
        "support": lv.support, "resistance": lv.resistance, "overhead": lv.overhead,
        "rvol": lv.rvol,
        "volume_status": volume_status(c),
        "breakout_vol_mult": st.breakout_vol_mult,
        "structure": {
            "status": st.status, "breakout": st.breakout, "holding": st.holding,
            "retested": st.retested, "retest_low": st.retest_low,
            "extended": st.extended, "failed": st.failed_breakout,
            "breakout_time": st.breakout_time.strftime("%H:%M")
            if st.breakout_time else None,
        },
        "oi": {
            "change_pct": c.fut.oi_change_pct,
            "open_interest": c.fut.open_interest,
            "classification": c.oi.classification,
            "detail": c.oi.detail,
            "bullish": c.oi.bullish,
        },
        "rel_strength": c.rel_strength,
        "sector_strength": c.sector_strength,
        "trade": None if t is None else {
            "entry_low": t.entry_low, "entry_high": t.entry_high, "entry": t.entry,
            "stop": t.stop, "stop_basis": t.stop_basis,
            "target1": t.target1, "target2": t.target2,
            "risk": t.risk, "risk_pct": t.risk / t.entry * 100,
            "rr1": t.rr1, "rr2": t.rr2,
        },
        "spark": c.spark,
        "options": _options(c),
        "reasons": reasons(c),
        "invalidation": invalidation(c),
        "blockers": [{"kind": k, "text": txt}
                     for k, txt in zip(c.blockers, c.rejections)],
        "warnings": c.warnings,
        "sources": [_prov(p, when) for p in c.sources],
    }


def market_block(res: ScanResult) -> str:
    m = res.market
    L = [
        "MARKET CONTEXT",
        f"  Classification:  {m.classification}",
        f"  NIFTY 50:        {_f(m.nifty_pct)}%"
        + ("" if m.nifty_above_vwap is None
           else "  (above VWAP)" if m.nifty_above_vwap else "  (below VWAP)"),
        f"  BANK NIFTY:      {_f(m.banknifty_pct)}%",
        f"  Breadth:         "
        + (f"{m.breadth_pct:.0f}% green ({m.advances:.0f} up / "
           f"{m.declines:.0f} down, {m.breadth_source})"
           if m.breadth_pct is not None and m.advances is not None
           else f"{m.breadth_pct:.0f}% green ({m.breadth_source})"
           if m.breadth_pct is not None else "unavailable"),
    ]
    if m.sector_pct:
        top = sorted(m.sector_pct.items(), key=lambda kv: kv[1], reverse=True)
        n = min(3, len(top) // 2) or len(top)
        L.append("  Strong sectors:  "
                 + ", ".join(f"{k} {v:+.2f}%" for k, v in top[:n])
                 + f"   ({m.sector_source})")
        if len(top) > n:
            L.append("  Weak sectors:    "
                     + ", ".join(f"{k} {v:+.2f}%" for k, v in top[-n:]))
    for p in m.sources:
        L.append(f"  Source:          {p.label(res.when)}")
    return "\n".join(L)


def render(res: ScanResult) -> str:
    out = [
        "=" * 62,
        f"NSE F&O INTRADAY LONG SCANNER — {res.when:%Y-%m-%d %H:%M:%S} IST",
        f"Window: {res.window}",
        "=" * 62,
        "",
    ]
    if res.data_notes:
        out.append("DATA / WINDOW NOTES")
        out += [f"  · {n}" for n in res.data_notes]
        out.append("")
    if not res.data_ok:
        out += [f"*** {'REAL-TIME DATA UNAVAILABLE — SCAN NOT RELIABLE.'} ***",
                "No setup is graded and no entry, stop or target is produced.",
                ""]
        return "\n".join(out)

    out += [market_block(res), ""]
    if res.index_plan is not None:
        from trading.fno import index_options as ix
        out += [ix.render(res.index_plan), ""]
    if res.chain_read is not None:
        from trading.fno import chain_analytics as ca
        out += [ca.render(res.chain_read), ""]
    if res.pivots is not None:
        from trading.fno import pivots as pv
        spot = res.index_plan.spot if res.index_plan else None
        out += [pv.render(res.pivots, res.gap, spot), ""]
    out += [f"Screened {res.considered} liquid F&O names.", ""]

    picks = res.picks
    if not picks:
        out += [NO_TRADE, ""]
        near = [c for c in res.candidates if c.score >= 60][:3]
        if near:
            out.append("Closest misses (not tradeable — shown so you can see why):")
            for c in near:
                why = c.rejections[0] if c.rejections else "no valid entry"
                out.append(f"  · {c.symbol:<12} {c.score:>5.0f}/100  {c.grade:<5} ✗ {why}")
            out.append("")
        return "\n".join(out)

    out.append(f"TOP {len(picks)} LONG CANDIDATE{'S' if len(picks) > 1 else ''}")
    out.append("")
    for i, cand in enumerate(picks, 1):
        out += ["-" * 62,
                candidate_block(cand, i, res.market.classification, res.when), ""]

    alerts = [c for c in picks if c.score >= C.ALERT_SCORE]
    for cand in alerts:
        out += ["=" * 62, alert(cand, res.when), ""]

    out += ["=" * 62,
            "Decision support only. Not investment advice; no profit is implied "
            "or guaranteed. Verify every level on your own terminal before acting.",
            "No order is placed by this tool."]
    return "\n".join(out)
