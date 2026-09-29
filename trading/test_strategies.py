"""Correctness tests for the strategy pack: chain analytics and pivot levels.

These are textbook formulas, so the tests are arithmetic ones — a case whose
answer can be worked out by hand, and the edge cases where the maths degenerates
(an empty chain, a session with no predecessor, a zero-range day).

Run with:

    .venv/bin/python -m trading.test_strategies
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from trading.fno import chain_analytics as ca
from trading.fno import pivots as pv
from trading.fno.models import IST, Provenance
from trading.fno.options import OptionQuote

FAILURES: list[str] = []
PROV = Provenance("test", "x", "snapshot",
                  datetime(2026, 8, 26, 10, 0, tzinfo=IST),
                  datetime(2026, 8, 26, 10, 0, tzinfo=IST))


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")
        print(f"  FAIL {name} {detail}")


def close_to(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


def opt(strike, kind, oi, expiry="01-Sep-2026", spot=24400.0, ltp=100.0):
    return OptionQuote(underlying="NIFTY", identifier=f"N{strike}{kind}",
                       option_type=kind, strike=strike, expiry=expiry,
                       last_price=ltp, pct_change=0.0, open_interest=oi,
                       volume=1_000_000, underlying_value=spot, prov=PROV)


# --- chain analytics -------------------------------------------------------

def test_pcr_and_walls():
    print("chain analytics")
    # Call OI 100k against put OI 60k -> PCR 0.60, inside the call-heavy band.
    board = [
        opt(24300, "Call", 20_000), opt(24400, "Call", 30_000), opt(24500, "Call", 50_000),
        opt(24300, "Put", 40_000), opt(24400, "Put", 15_000), opt(24500, "Put", 5_000),
    ]
    r = ca.analyse(board, "NIFTY", date(2026, 8, 26))
    check("put/call ratio is put OI over call OI",
          close_to(r.pcr, 60_000 / 100_000), str(r.pcr))
    check("a low ratio reads as call-heavy", r.pcr_label.startswith("call-heavy"),
          f"{r.pcr:.2f} -> {r.pcr_label}")
    check("the heaviest call OI above spot is the resistance wall",
          r.resistance_strike == 24500 and r.resistance_oi == 50_000)
    check("the heaviest put OI below spot is the support wall",
          r.support_strike == 24300 and r.support_oi == 40_000)
    check("strikes are counted", r.strikes == 3)
    check("open interest is always labelled non-directional",
          any("not directional" in n for n in r.notes))

    heavy_puts = [opt(24300, "Call", 10_000), opt(24300, "Put", 40_000)]
    check("a high ratio reads as put-heavy",
          ca.analyse(heavy_puts, "NIFTY", date(2026, 8, 26)).pcr_label.startswith("put-heavy"))

    check("an empty chain is None, not a zero-filled read",
          ca.analyse([], "NIFTY", date(2026, 8, 26)) is None)
    check("an unknown underlying is None",
          ca.analyse(board, "BANKNIFTY", date(2026, 8, 26)) is None)


def test_max_pain():
    print("max pain")
    # Worked by hand. 200k calls at 24400, 100k puts at 24600:
    #   settle 24400 -> calls pay 0,  puts pay 200*100k = 20.0m   <- least
    #   settle 24500 -> calls pay 100*200k = 20m, puts 100*100k = 10m = 30.0m
    #   settle 24600 -> calls pay 200*200k = 40m, puts 0          = 40.0m
    calls = {24400: 200_000.0, 24500: 0.0, 24600: 0.0}
    puts = {24400: 0.0, 24500: 0.0, 24600: 100_000.0}
    mp = ca.max_pain(calls, puts)
    check("max pain lands where writers pay least", mp == 24400, str(mp))

    # Mirror it: heavy puts below pull max pain up to their strike instead.
    mp2 = ca.max_pain({24400: 0.0, 24500: 0.0, 24600: 100_000.0},
                      {24400: 0.0, 24500: 0.0, 24600: 200_000.0})
    check("heavy put OI pulls max pain to its strike", mp2 == 24600, str(mp2))

    check("too few strikes yields no max pain",
          ca.max_pain({24400: 1.0}, {24400: 1.0}) is None)

    far = ca.analyse([opt(24400, "Call", 10, expiry="29-Dec-2026"),
                      opt(24300, "Put", 10, expiry="29-Dec-2026"),
                      opt(24500, "Put", 10, expiry="29-Dec-2026")],
                     "NIFTY", date(2026, 8, 26))
    check("max pain far from expiry is flagged as not an intraday level",
          not far.max_pain_useful
          and any("expiry-week magnet" in n for n in far.notes), str(far.notes))


# --- pivots ----------------------------------------------------------------

def test_pivots():
    print("pivots")
    # H=110, L=90, C=100 -> P=100, BC=100, TC=100, range 20.
    p = pv.compute(110, 90, 100)
    check("the pivot is the average of high, low and close", close_to(p.pivot, 100))
    check("R1 = 2P - low", close_to(p.r1, 110))
    check("S1 = 2P - high", close_to(p.s1, 90))
    check("R2 = P + range", close_to(p.r2, 120))
    check("S2 = P - range", close_to(p.s2, 80))
    check("a symmetric day gives a zero-width CPR", close_to(p.cpr_width, 0))
    check("and that reads as narrow", p.cpr_label.startswith("narrow"))

    # A close far from the midpoint widens the CPR.
    wide = pv.compute(110, 90, 109)
    check("a close near the high widens the CPR", wide.cpr_width > p.cpr_width)
    check("TC is always the upper central line", wide.tc >= wide.bc)

    check("Camarilla bands straddle the close",
          wide.l4 < wide.l3 < wide.prev_close < wide.h3 < wide.h4)

    near = p.nearest_above(105)
    check("the nearest level above a price is found",
          near is not None and near[1] >= 105, str(near))
    below = p.nearest_below(105)
    check("and the nearest below", below is not None and below[1] <= 105, str(below))


def _bars(day, rows):
    idx = pd.DatetimeIndex([pd.Timestamp(f"{day}T{t}:00+05:30") for t, *_ in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows],
         "Volume": [1000.0] * len(rows)}, index=idx)


def test_gap():
    print("gap read")
    prev = _bars("2026-08-25", [("15:20", 100, 102, 99, 100)])
    up = _bars("2026-08-26", [("09:15", 103, 104, 102.5, 103.5),
                              ("09:20", 103.5, 104, 103, 103.8)])
    df = pd.concat([prev, up])
    g = pv.gap(df, date(2026, 8, 26))
    check("a gap up is measured against the previous close",
          close_to(g.gap_pct, 3.0), str(g.gap_pct))
    check("an unfilled gap is reported as such", not g.filled and "gap up" in g.label)

    filled = pd.concat([prev, _bars("2026-08-26", [("09:15", 103, 104, 99.5, 100.2)])])
    check("a gap that trades back through the close is filled",
          pv.gap(filled, date(2026, 8, 26)).filled)

    flat = pd.concat([prev, _bars("2026-08-26", [("09:15", 100.1, 101, 99.8, 100.5)])])
    check("a small move is a flat open, not a gap",
          pv.gap(flat, date(2026, 8, 26)).label == "flat open")

    check("no previous session means no gap read",
          pv.gap(up, date(2026, 8, 26)) is None)
    check("and no pivots either", pv.from_candles(up, date(2026, 8, 26)) is None)
    check("pivots come from the previous session",
          close_to(pv.from_candles(df, date(2026, 8, 26)).prev_close, 100))


def test_render():
    print("rendering")
    p = pv.compute(110, 90, 100)
    out = pv.render(p, None, 105)
    for word in ("Pivot", "CPR", "Resistance", "Support", "Camarilla"):
        check(f"pivot report shows '{word}'", word in out)
    check("pivot report never claims to be a signal",
          "levels, not signals" in out)
    check("render survives a missing pivot set", "no previous session" in pv.render(None, None))
    check("chain report survives a missing chain", "no chain" in ca.render(None))


def test_orb_ma200():
    print("orb ma200 strategy")
    from trading.strategies.orb_ma200 import ORBMA200Strategy
    import numpy as np

    ist = ZoneInfo("Asia/Kolkata")
    base_dates = []
    curr_date = datetime(2026, 8, 10, 9, 15, tzinfo=ist)
    while len(base_dates) < 2500:
        if curr_date.weekday() < 5:
            if "09:15" <= curr_date.strftime("%H:%M") <= "15:25":
                base_dates.append(curr_date)
        curr_date += timedelta(minutes=5)

    n_bars = len(base_dates)
    prices = np.linspace(100.0, 200.0, n_bars)
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": 1000.0,
    }, index=pd.DatetimeIndex(base_dates))

    last_day_date = df.index[-1].date()
    day_mask = df.index.date == last_day_date
    day_indices = np.where(day_mask)[0]

    # Structure Opening Range
    or_idx = day_indices[:3]
    df.iloc[or_idx, df.columns.get_loc("Open")] = 200.0
    df.iloc[or_idx, df.columns.get_loc("High")] = 201.0
    df.iloc[or_idx, df.columns.get_loc("Low")] = 199.5
    df.iloc[or_idx, df.columns.get_loc("Close")] = 200.2
    df.iloc[or_idx, df.columns.get_loc("Volume")] = 1000.0

    # Breakout bar
    breakout_idx = day_indices[3]
    df.iloc[breakout_idx, df.columns.get_loc("Open")] = 200.8
    df.iloc[breakout_idx, df.columns.get_loc("High")] = 202.2
    df.iloc[breakout_idx, df.columns.get_loc("Low")] = 201.2
    df.iloc[breakout_idx, df.columns.get_loc("Close")] = 202.0
    df.iloc[breakout_idx, df.columns.get_loc("Volume")] = 5000.0

    eval_df = df.iloc[: breakout_idx + 1]
    strat = ORBMA200Strategy()
    sig = strat.evaluate(eval_df, "INFY")
    check("ORB MA 200 generates breakout buy signal", sig is not None and sig.side == "BUY")
    if sig:
        check("signal has correct symbol", sig.symbol == "INFY")
        check("signal has strategy_id orb_ma200", sig.strategy_id == "orb_ma200")
        check("signal has stop_loss and target", sig.stop_loss < sig.price < sig.target)


def main() -> int:
    for fn in (test_pcr_and_walls, test_max_pain, test_pivots, test_gap, test_render, test_orb_ma200):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

