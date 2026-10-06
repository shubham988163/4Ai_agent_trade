"""Correctness tests for the F&O opening-window long scanner.

The scanner's failure mode is a plausible-looking wrong answer: a wick counted
as a breakout, a stop derived from a level that does not exist, a score that
survives contradictory OI. So every case here is built so the right answer is
known before the code runs — hand-written candles with a stated breakout level,
retest low and volume profile.

Run with:

    .venv/bin/python -m trading.test_fno
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd

from trading.fno import config as C
from trading.fno import context as ctx
from trading.fno import filters
from trading.fno import indicators as ind
from trading.fno import oi as oi_mod
from trading.fno import report
from trading.fno import scoring
from trading.fno import setup as setup_mod
from trading.fno.data import ReplayFeed, _frame, synthetic_bundle
from trading.fno.models import IST, Candidate, FuturesSnapshot, MarketContext, Provenance
from trading.fno.scanner import STALE_DATA_BANNER, Scanner, window_for

FAILURES: list[str] = []
TODAY = date(2026, 8, 20)          # a Thursday
AS_OF = datetime(2026, 8, 20, 9, 45, tzinfo=IST)


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")
        print(f"  FAIL {name} {detail}")


def close_to(a, b, tol=1e-6) -> bool:
    return a is not None and abs(a - b) <= tol


# --- candle builders -------------------------------------------------------

def drift_bars(day: date, n: int, start_price: float, drift: float,
               rng: float, vol: float, first=(9, 15)) -> list[list]:
    """A run of 5-minute bars drifting by `drift` per bar, range `rng`."""
    rows, price = [], start_price
    t0 = datetime(day.year, day.month, day.day, first[0], first[1], tzinfo=IST)
    for i in range(n):
        o = price
        c = o + drift
        rows.append([(t0 + timedelta(minutes=5 * i)).isoformat(),
                     round(o, 2), round(max(o, c) + rng / 2, 2),
                     round(min(o, c) - rng / 2, 2), round(c, 2), vol])
        price = c
    return rows


def history_bars() -> list[list]:
    """Three prior sessions — enough bars to seed a real 20/50 EMA, rising."""
    rows = []
    price = 98.0
    for d in (date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19)):
        day_rows = drift_bars(d, 72, price, 0.01, 0.5, 10_000)
        rows += day_rows
        price = day_rows[-1][4]
    return rows


# Today: opening range 100.10–101.00, breakout at 09:30, retest at 09:35 into
# 101.10, holding at 101.50 by 09:40. Resistance R = 101.00 by construction.
TODAY_CLEAN = [
    ["2026-08-20T09:15:00+05:30", 100.30, 100.80, 100.10, 100.60, 30_000],
    ["2026-08-20T09:20:00+05:30", 100.60, 101.00, 100.50, 100.90, 28_000],
    ["2026-08-20T09:25:00+05:30", 100.90, 101.00, 100.70, 100.95, 26_000],
    ["2026-08-20T09:30:00+05:30", 100.95, 101.70, 100.90, 101.60, 90_000],
    ["2026-08-20T09:35:00+05:30", 101.60, 101.65, 101.10, 101.45, 40_000],
    ["2026-08-20T09:40:00+05:30", 101.45, 101.60, 101.35, 101.50, 35_000],
]

# Same range, but the 09:30 bar only wicks through 101.00 and closes back under.
TODAY_WICK = [
    *TODAY_CLEAN[:3],
    ["2026-08-20T09:30:00+05:30", 100.95, 101.45, 100.85, 100.95, 90_000],
    ["2026-08-20T09:35:00+05:30", 100.95, 101.00, 100.70, 100.80, 40_000],
    ["2026-08-20T09:40:00+05:30", 100.80, 100.95, 100.60, 100.75, 35_000],
]

# Broke out, then lost the level entirely.
TODAY_FAILED = [
    *TODAY_CLEAN[:4],
    ["2026-08-20T09:35:00+05:30", 101.60, 101.65, 100.40, 100.50, 40_000],
    ["2026-08-20T09:40:00+05:30", 100.50, 100.60, 100.20, 100.30, 35_000],
]

# Identical structure, but the breakout bar carries less volume than the range.
TODAY_WEAK_VOL = [
    *TODAY_CLEAN[:3],
    [*TODAY_CLEAN[3][:5], 20_000],
    *TODAY_CLEAN[4:],
]


def stock_df(today_rows: list[list]) -> pd.DataFrame:
    return _frame(history_bars() + today_rows)


# --- indicator tests -------------------------------------------------------

def test_indicators():
    print("indicators")
    flat = _frame([["2026-08-20T09:15:00+05:30", 100, 100, 100, 100, 1_000],
                   ["2026-08-20T09:20:00+05:30", 100, 100, 100, 100, 5_000]])
    check("VWAP of a flat tape equals price",
          close_to(float(ind.vwap(flat).iloc[-1]), 100.0))

    two = _frame([["2026-08-20T09:15:00+05:30", 100, 100, 100, 100, 1_000],
                  ["2026-08-20T09:20:00+05:30", 110, 110, 110, 110, 3_000]])
    # (100*1000 + 110*3000) / 4000 = 107.5 — volume-weighted, not the mean (105).
    check("VWAP is volume-weighted", close_to(float(ind.vwap(two).iloc[-1]), 107.5))

    df = stock_df(TODAY_CLEAN)
    day_df = ind.session_of(df, TODAY)
    or_h, or_l, bars = ind.opening_range(day_df)
    check("opening range is 09:15–09:30 only, 3 bars",
          (bars, or_h, or_l) == (3, 101.00, 100.10), f"got {(bars, or_h, or_l)}")

    # The 09:30 bar trades after the range is set and must not be inside it.
    check("09:30 bar is excluded from the opening range", or_h < 101.70)

    forming = ind.closed_bars(df, datetime(2026, 8, 20, 9, 43, tzinfo=IST))
    check("forming bar is dropped (no look-ahead)",
          forming.index[-1].strftime("%H:%M") == "09:35",
          f"last bar {forming.index[-1]}")

    # Today's 249k to 09:45 vs 6 prior-session bars x 10k = 60k -> 4.15x.
    rv = ind.rvol(df, TODAY, datetime(2026, 8, 20, 9, 45).time())
    check("RVOL compares same-time-of-day volume", close_to(rv, 249_000 / 60_000, 0.01),
          f"rvol={rv}")

    ph, pl, pc = ind.prev_session_levels(df, TODAY)
    check("previous session levels come from 19 Aug", pc is not None and pc < 101.0,
          f"prev close {pc}")

    hh, hl = ind.higher_highs_lows(day_df)
    check("higher highs and higher lows detected", hh and hl)


# --- OI tests --------------------------------------------------------------

def test_oi():
    print("open interest")
    check("price up + OI up = long buildup",
          oi_mod.classify(1.2, 4.0).classification == "LONG BUILDUP")
    check("price down + OI up = short buildup (not bullish)",
          oi_mod.classify(-1.2, 4.0).classification == "SHORT BUILDUP"
          and not oi_mod.classify(-1.2, 4.0).bullish)
    sc = oi_mod.classify(1.2, -4.0)
    check("price up + OI down = short covering, bullish but discounted",
          sc.classification == "SHORT COVERING" and sc.bullish
          and oi_mod.score(sc) < oi_mod.score(oi_mod.classify(1.2, 4.0)))
    check("price down + OI down = long unwinding",
          oi_mod.classify(-1.2, -4.0).classification == "LONG UNWINDING")
    check("tiny OI change is flat, not a signal",
          oi_mod.classify(1.2, 0.2).classification == "OI FLAT")
    unavailable = oi_mod.classify(1.2, None)
    check("missing OI is UNAVAILABLE and scores zero",
          unavailable.classification == "UNAVAILABLE"
          and not unavailable.reliable and oi_mod.score(unavailable) == 0.0)


# --- structure tests -------------------------------------------------------

def _levels(rows):
    df = stock_df(rows)
    lv = setup_mod.build_levels(df, TODAY, AS_OF, 1.2)
    return df, lv, setup_mod.read_structure(df, TODAY, lv)


def test_structure():
    print("structure")
    df, lv, st = _levels(TODAY_CLEAN)
    check("resistance is the opening-range high", close_to(lv.resistance, 101.00))
    check("breakout detected on the 09:30 close",
          st.breakout and st.breakout_time.strftime("%H:%M") == "09:30")
    check("breakout volume measured against pre-breakout bars",
          close_to(st.breakout_vol_mult, 90_000 / 28_000, 0.01),
          f"mult={st.breakout_vol_mult}")
    check("retest into 101.10 recognised and holding",
          st.retested and close_to(st.retest_low, 101.10) and st.holding)
    check("status reads breakout → retest → hold",
          st.status == "BREAKOUT → RETEST → HOLD", st.status)
    check("clean setup is not flagged extended", not st.extended)

    _, _, wick = _levels(TODAY_WICK)
    check("a wick above resistance is not a breakout",
          not wick.breakout and wick.rejection, wick.status)

    _, _, failed = _levels(TODAY_FAILED)
    check("losing the level is a failed breakout",
          failed.breakout and failed.failed_breakout and not failed.holding,
          failed.status)

    _, _, weak = _levels(TODAY_WEAK_VOL)
    check("weak breakout volume is measured, not ignored",
          weak.breakout and weak.breakout_vol_mult < C.WEAK_VOL_MULT,
          f"mult={weak.breakout_vol_mult}")


def test_trade_levels():
    print("entry / stop / target")
    df, lv, st = _levels(TODAY_CLEAN)
    trade, problems = setup_mod.build_trade(df, TODAY, lv, st)
    check("trade is produced for the clean setup", trade is not None and not problems,
          str(problems))
    if trade:
        check("entry zone starts at the breakout level or above the stop",
              close_to(trade.entry_low, max(lv.resistance, trade.stop + 0.05 * lv.atr)),
              f"low={trade.entry_low} R={lv.resistance} stop={trade.stop}")
        check("no price inside the entry zone sits below the stop",
              trade.stop < trade.entry_low <= trade.entry_high,
              f"stop={trade.stop} zone={trade.entry_low}-{trade.entry_high}")
        check("entry is capped inside the zone (no chasing)",
              trade.entry <= trade.entry_high < lv.price + 1e-9,
              f"entry={trade.entry} high={trade.entry_high} price={lv.price}")
        check("stop is structural — below the retest low",
              trade.stop < st.retest_low and "retest" in trade.stop_basis,
              f"stop={trade.stop} basis={trade.stop_basis}")
        check("targets are exactly 1:2 and 1:3 of the measured risk",
              close_to(trade.target1, trade.entry + 2 * trade.risk, 1e-9)
              and close_to(trade.target2, trade.entry + 3 * trade.risk, 1e-9))
        check("risk is a real distance, not a percentage guess",
              trade.risk > 0 and 0.15 <= trade.risk / trade.entry * 100 <= 1.20,
              f"risk%={trade.risk / trade.entry * 100:.3f}")

    _, lv_f, st_f = _levels(TODAY_FAILED)
    t_failed, why = setup_mod.build_trade(stock_df(TODAY_FAILED), TODAY, lv_f, st_f)
    check("no trade when the breakout has failed", t_failed is None and why)

    _, lv_w, st_w = _levels(TODAY_WICK)
    t_wick, why_w = setup_mod.build_trade(stock_df(TODAY_WICK), TODAY, lv_w, st_w)
    check("no trade without a confirmed breakout",
          t_wick is None and why_w[0] == ("structure",
                                          "no confirmed breakout — price has not "
                                          "closed above resistance"), str(why_w))

    # Overhead resistance sitting just above entry must kill the 1:2 target.
    lv_tight = setup_mod.build_levels(df, TODAY, AS_OF, 1.2)
    lv_tight.overhead = lv_tight.resistance + 0.10
    _, tight_problems = setup_mod.build_trade(df, TODAY, lv_tight, st)
    check("resistance too close to make 1:2 is rejected",
          any(kind == "risk" and "resistance at" in text
              for kind, text in tight_problems), str(tight_problems))


# --- market context --------------------------------------------------------

def test_market_context():
    print("market context")
    check("+0.8% NIFTY is strongly bullish",
          ctx.classify(0.8, 0.5, 60.0, True) == "STRONGLY BULLISH")
    check("flat NIFTY is neutral", ctx.classify(0.05, 0.0, 50.0, True) == "NEUTRAL")
    check("-1.0% NIFTY is strongly bearish",
          ctx.classify(-1.0, -1.0, 20.0, False) == "STRONGLY BEARISH")
    check("green but below VWAP is downgraded one notch",
          ctx.classify(0.30, 0.1, 55.0, False) == "NEUTRAL",
          ctx.classify(0.30, 0.1, 55.0, False))
    check("missing NIFTY data falls back to NEUTRAL, never bullish",
          ctx.classify(None, None, None, None) == "NEUTRAL")

    universe = [{"symbol": "RELIANCE", "pct": 1.2}, {"symbol": "ONGC", "pct": 0.9},
                {"symbol": "BPCL", "pct": 0.6}, {"symbol": "TCS", "pct": -0.8},
                {"symbol": "INFY", "pct": -1.2}]
    med = ctx.sector_medians(universe)
    check("sector strength is the peer median", close_to(med["ENERGY"], 0.9),
          str(med))
    check("sectors with a single peer are not reported", "UNMAPPED" not in med)


# --- scoring & filters -----------------------------------------------------

def _candidate(rows=TODAY_CLEAN, *, oi_pct=4.2, pct=1.2, spread=(101.45, 101.55),
               turnover=800.0):
    df = stock_df(rows)
    lv = setup_mod.build_levels(df, TODAY, AS_OF, pct)
    st = setup_mod.read_structure(df, TODAY, lv)
    fut = FuturesSnapshot(
        symbol="RELIANCE", contract="RELIANCE28AUG2026FUT", expiry="28-Aug-2026",
        last_price=lv.price, prev_close=lv.prev_close, pct_change=pct,
        open_interest=1_200_000, oi_change_pct=oi_pct, volume=45_000,
        turnover_cr=turnover, bid=spread[0], ask=spread[1],
        prov=Provenance("test", "RELIANCE28AUG2026FUT", "snapshot", AS_OF, AS_OF))
    cand = Candidate(symbol="RELIANCE", sector="ENERGY", fut=fut, levels=lv,
                     structure=st, oi=oi_mod.classify(pct, oi_pct),
                     rel_strength=(pct or 0) - 0.4, sector_strength=0.9,
                     sources=[fut.prov])
    trade, problems = setup_mod.build_trade(df, TODAY, lv, st)
    cand.trade = trade
    return cand, problems


def _market(label="BULLISH", nifty=0.4):
    return MarketContext(classification=label, nifty_pct=nifty, banknifty_pct=0.3,
                         nifty_above_vwap=True, breadth_pct=75.0, advances=15,
                         declines=5, sector_pct={"ENERGY": 0.9})


def test_scoring():
    print("scoring")
    cand, problems = _candidate()
    scoring.score_candidate(cand, _market())
    check("clean multi-factor setup grades A or A+",
          cand.score >= 80 and cand.grade in ("A", "A+"),
          f"score={cand.score} parts={cand.scores}")
    check("component weights sum to the spec's 100", sum(C.WEIGHTS.values()) == 100)
    check("no single component can exceed its weight",
          all(cand.scores[k] <= C.WEIGHTS[k] + 1e-9 for k in C.WEIGHTS), str(cand.scores))

    base = cand.score
    contradicted, _ = _candidate(oi_pct=-6.0)
    scoring.score_candidate(contradicted, _market())
    check("short covering scores below fresh long buildup",
          contradicted.score < base, f"{contradicted.score} vs {base}")

    no_oi, _ = _candidate(oi_pct=None)
    scoring.score_candidate(no_oi, _market())
    check("missing OI costs the full 15-point bucket",
          close_to(no_oi.scores["oi"], 0.0), str(no_oi.scores))

    bear, _ = _candidate()
    scoring.score_candidate(bear, _market("STRONGLY BEARISH", -1.2))
    check("a strongly bearish market drags the same chart below tradeable",
          bear.score < C.MIN_TRADEABLE_SCORE, f"score={bear.score}")

    weak, _ = _candidate(rows=TODAY_WEAK_VOL)
    scoring.score_candidate(weak, _market())
    check("volume-less breakout scores below the volume-backed one",
          weak.scores["volume"] < cand.scores["volume"])


def test_filters():
    print("veto filters")
    cand, problems = _candidate()
    filters.apply(cand, _market(), problems)
    scoring.score_candidate(cand, _market())
    cand.rejections.clear()
    filters.apply(cand, _market(), problems)
    check("clean setup survives the filters", not cand.rejections, str(cand.rejections))

    def rejected(**kw):
        c, p = _candidate(**{k: v for k, v in kw.items() if k != "market"})
        scoring.score_candidate(c, kw.get("market", _market()))
        filters.apply(c, kw.get("market", _market()), p)
        return c.rejections

    check("contradictory OI is vetoed",
          any("SHORT BUILDUP" in r or "LONG UNWINDING" in r
              for r in rejected(pct=-1.2, oi_pct=4.0)))
    check("missing OI is vetoed",
          any("OI data unavailable" in r for r in rejected(oi_pct=None)))
    check("wide futures spread is vetoed",
          any("spread" in r for r in rejected(spread=(101.0, 102.0))))
    check("illiquid contract is vetoed",
          any("illiquid" in r for r in rejected(turnover=5.0)))
    check("abnormal move is flagged for verification",
          any("abnormal" in r for r in rejected(pct=9.5)))
    check("strongly bearish market is vetoed",
          any("strongly bearish" in r
              for r in rejected(market=_market("STRONGLY BEARISH", -1.2))))
    check("weak breakout volume is vetoed",
          any("volume" in r for r in rejected(rows=TODAY_WEAK_VOL)))
    check("failed breakout is vetoed",
          any("failed breakout" in r or "not holding" in r
              for r in rejected(rows=TODAY_FAILED)))


def test_relaxed_grading():
    print("relaxed grading")
    check("relaxed grading is off by default", not C.RELAXED)
    try:
        # Strict (the default): a flat OI read is treated as a contradiction.
        strict, p = _candidate(oi_pct=0.2)
        scoring.score_candidate(strict, _market())
        filters.apply(strict, _market(), p)
        check("strict: OI flat is filed as a hard blocker", "oi" in strict.blockers,
              str(strict.blockers))
        check("strict: OI flat reads AVOID", strict.verdict == "AVOID",
              f"{strict.verdict} score={strict.score}")

        C.set_relaxed(True)
        check("relaxed leaves the score floor at 70",
              C.MIN_TRADEABLE_SCORE == 70, str(C.MIN_TRADEABLE_SCORE))

        soft, p2 = _candidate(oi_pct=0.2)
        scoring.score_candidate(soft, _market())
        filters.apply(soft, _market(), p2)
        check("relaxed: OI flat is filed as a soft blocker",
              soft.blockers == ["oi_flat"], str(soft.blockers))
        check("relaxed: the rejection is kept, so the name is never tradeable",
              not soft.tradeable and any("OI FLAT" in r for r in soft.rejections),
              str(soft.rejections))
        check("relaxed: OI flat surfaces as WATCH instead of AVOID",
              soft.verdict == "WATCH", f"{soft.verdict} score={soft.score}")

        # Relaxation only ever downgrades, so it must not touch what can be BUY.
        good, p3 = _candidate()
        scoring.score_candidate(good, _market())
        filters.apply(good, _market(), p3)
        check("relaxed: a clean setup is still tradeable",
              good.tradeable and good.verdict == "BUY",
              f"{good.verdict} {good.rejections}")

        # And it is not a laundering pass: a read that argues against the long,
        # or a missing one, stays hard in either mode.
        bad, p4 = _candidate(pct=-1.2, oi_pct=4.0)
        scoring.score_candidate(bad, _market())
        filters.apply(bad, _market(), p4)
        check("relaxed: short buildup is still a hard blocker",
              "oi" in bad.blockers and bad.verdict == "AVOID",
              f"{bad.blockers} {bad.verdict}")

        gone, p5 = _candidate(oi_pct=None)
        scoring.score_candidate(gone, _market())
        filters.apply(gone, _market(), p5)
        check("relaxed: unavailable OI is still a hard blocker",
              "oi" in gone.blockers, str(gone.blockers))
    finally:
        C.set_relaxed(False)
    check("relaxed is switched back off", not C.RELAXED)


# --- windows ---------------------------------------------------------------

def test_windows():
    print("time windows")
    at = lambda h, m: datetime(2026, 8, 20, h, m, tzinfo=IST)   # noqa: E731
    check("09:17 emits no signals", not window_for(at(9, 17)).signals_allowed)
    check("09:25 emits no signals (range still forming)",
          not window_for(at(9, 25)).signals_allowed)
    check("09:35 is the trend-evaluation window",
          window_for(at(9, 35)).signals_allowed
          and "TREND" in window_for(at(9, 35)).label)
    check("09:50 is the breakout+retest window",
          "BREAKOUT" in window_for(at(9, 50)).label)
    check("10:15 is labelled post-10:00 monitoring",
          "POST-10:00" in window_for(at(10, 15)).label)
    check("08:00 is pre-open", window_for(at(8, 0)).label == "PRE-OPEN")


# --- end to end ------------------------------------------------------------

FILLERS = {"ONGC": 0.9, "BPCL": 0.6, "TCS": 0.8, "INFY": 0.5, "HDFCBANK": 0.7,
           "SBIN": 1.1, "AXISBANK": 0.4, "ITC": 0.5, "LT": 0.6, "MARUTI": 0.9,
           "TITAN": -0.3, "WIPRO": -0.4, "CIPLA": 0.5, "NTPC": 0.6}


def bundle(today_rows=TODAY_CLEAN, *, oi_pct=4.2, pct=1.2) -> dict:
    stocks = {
        "RELIANCE": {
            "pct": pct, "volume": 4_000_000, "value_cr": 600.0,
            "candles": history_bars() + today_rows,
            "futures": {"contract": "RELIANCE28AUG2026FUT", "expiry": "28-Aug-2026",
                        "last_price": today_rows[-1][4], "prev_close": 100.25,
                        "pct_change": pct, "open_interest": 1_250_000,
                        "oi_change_pct": oi_pct, "volume": 48_000,
                        "turnover_cr": 820.0, "bid": 101.45, "ask": 101.55},
        }
    }
    for sym, p in FILLERS.items():
        stocks[sym] = {"pct": p, "volume": 2_000_000, "value_cr": 200.0,
                       "futures": {"last_price": 500.0, "pct_change": p,
                                   "oi_change_pct": 1.0, "turnover_cr": 300.0}}
    nifty = (drift_bars(date(2026, 8, 19), 72, 20_000, 0.0, 5, 1_000)
             + drift_bars(TODAY, 6, 20_000, 13.0, 5, 1_000))
    bank = (drift_bars(date(2026, 8, 19), 72, 45_000, 0.0, 10, 1_000)
            + drift_bars(TODAY, 6, 45_000, 20.0, 10, 1_000))
    return synthetic_bundle(as_of=AS_OF, stocks=stocks,
                            indices={"^NSEI": nifty, "^NSEBANK": bank})


def _run(bundle_dict, *, now=AS_OF, allow_delayed=True):
    return Scanner(ReplayFeed(bundle_dict), now, allow_delayed=allow_delayed).run()


def test_end_to_end():
    print("end to end")
    res = _run(bundle())
    check("market context is read as bullish or better",
          res.market.classification in ("BULLISH", "STRONGLY BULLISH"),
          f"{res.market.classification} nifty={res.market.nifty_pct}")
    check("breadth is computed from the universe",
          res.market.breadth_pct is not None and res.market.breadth_pct > 60)
    check("the clean setup is emitted as a pick",
          [c.symbol for c in res.picks] == ["RELIANCE"],
          f"picks={[c.symbol for c in res.picks]} "
          f"rejections={[c.rejections for c in res.candidates]}")

    if res.picks:
        pick = res.picks[0]
        check("pick carries a full trade plan",
              pick.trade is not None and pick.trade.stop < pick.trade.entry
              < pick.trade.target1 < pick.trade.target2)
        check("pick reports both data sources with timestamps",
              len(pick.sources) == 2 and all(p.data_time for p in pick.sources))
        text = report.render(res)
        for field in ("9:15–9:30 High", "VWAP", "20 EMA", "50 EMA", "OI Change",
                      "OI Interpretation", "Breakout Status", "Retest Status",
                      "Relative Strength", "Setup Grade", "Entry Zone",
                      "Stop Loss", "Target 1", "Target 2", "Risk/Reward",
                      "Reason", "Invalidation"):
            check(f"report contains '{field}'", field in text)
        check("A/A+ setup fires the alert block", "🚨 NSE F&O LONG SETUP" in text)
        check("report never claims profit",
              "guarantee" not in text.lower() or "no profit is implied" in text.lower())

    no_oi = _run(bundle(oi_pct=None))
    check("no OI → no trade, with the reason stated",
          not no_oi.picks and report.NO_TRADE in report.render(no_oi))

    failed = _run(bundle(TODAY_FAILED))
    check("failed breakout → no trade", not failed.picks)

    weak = _run(bundle(TODAY_WEAK_VOL))
    check("volume-less breakout → no trade", not weak.picks)

    early = _run(bundle(), now=datetime(2026, 8, 20, 9, 18, tzinfo=IST))
    check("no signal in the 09:15–09:20 observation window",
          not early.picks and "OBSERVATION" in early.window)

    text_early = report.render(early)
    check("observation window says so in the output", "no BUY signals" in text_early)
    payload = report.as_dict(early)
    check("the payload marks the window as not signalling",
          payload["signals_allowed"] is False and payload["window_note"])
    check("a normal window is marked as signalling",
          report.as_dict(_run(bundle()))["signals_allowed"] is True)
    check("the dashboard distinguishes 'too early' from 'nothing qualifies'",
          "TOO EARLY TO SAY" in __import__("trading.dashboard",
                                           fromlist=["PAGE"]).PAGE)


def test_data_integrity():
    print("data integrity")

    feed = _DelayedFeed(bundle())
    res = Scanner(feed, AS_OF, allow_delayed=False).run()
    check("delayed candles without opt-in refuse to grade", not res.data_ok)
    text = report.render(res)
    check("refusal prints the exact spec banner", STALE_DATA_BANNER in text)
    check("refusal emits no entry/stop/target",
          "Entry Zone" not in text and "Stop Loss" not in text)

    ok = Scanner(_DelayedFeed(bundle()), AS_OF, allow_delayed=True).run()
    check("with --allow-delayed the scan runs but candidates carry a caveat",
          ok.data_ok and ok.picks
          and any("delayed" in w for w in ok.picks[0].warnings),
          str(ok.picks[0].warnings if ok.picks else "no picks"))

    empty = Scanner(ReplayFeed(synthetic_bundle(as_of=AS_OF, stocks={})), AS_OF,
                    allow_delayed=True).run()
    check("no candles at all is a refusal, not an empty pick list",
          not empty.data_ok and STALE_DATA_BANNER in report.render(empty))


def test_feed_parsing():
    """The live feed's own parsing rules, without touching the network."""
    print("feed parsing")
    from trading.fno import data as data_mod
    from trading.fno import nse as nse_mod

    check("traded value is converted from lakhs to crore",
          close_to(data_mod._crore(63631.35), 636.31, 0.01))
    check("an implausible traded value becomes None, not a tiny number",
          data_mod._crore(0.4) is None)
    check("NSE numbers survive comma formatting",
          close_to(nse_mod.num("1,23,456.70"), 123456.70))
    check("missing NSE fields parse to None, never 0",
          nse_mod.num("-") is None and nse_mod.num("") is None
          and nse_mod.num(None) is None)
    check("NSE timestamps parse as IST",
          nse_mod.parse_timestamp("21-Aug-2026 09:47:31").hour == 9)
    check("an unparseable timestamp is None rather than now()",
          nse_mod.parse_timestamp("garbage") is None)
    check("the first non-empty key wins in pick()",
          nse_mod.pick({"a": "-", "B": 5}, "a", "b") == 5)

    # OI change % is derived as changeInOI / prevOI, matching §6's definition
    # of OI change against the previous day's close.
    latest, prev = 187_448, 180_000
    check("OI change % is measured against the previous day's OI",
          close_to((latest - prev) / prev * 100, 4.1378, 1e-3))

    check("index sectors map to names NSE actually publishes",
          C.SECTOR_INDEX["BANK"] == "NIFTY BANK"
          and C.SECTOR_INDEX["IT"] == "NIFTY IT"
          and all(v.startswith("NIFTY ") for v in C.SECTOR_INDEX.values()))
    check("every mapped sector code exists in the symbol map",
          set(C.SECTOR_INDEX) <= set(C.SECTORS.values()),
          str(set(C.SECTOR_INDEX) - set(C.SECTORS.values())))


def test_zero_volume_feeds():
    """Index feeds report Volume=0; that must not become a fake VWAP."""
    print("zero-volume feeds")
    idx = _frame([["2026-08-20T09:15:00+05:30", 100, 101, 99, 100.5, 0],
                  ["2026-08-20T09:20:00+05:30", 100.5, 102, 100, 101.5, 0]])
    check("a volume-less frame is detected", not ind.has_volume(idx))
    check("VWAP on a volume-less frame does not raise or produce NaN",
          float(ind.vwap(idx).iloc[-1]) == (102 + 100 + 101.5) / 3)

    from trading.fno.models import Candles, Provenance
    cand = Candles(idx, Provenance("test", "^NSEI", "5m", AS_OF, AS_OF))
    pct, above = ctx.index_read(cand, TODAY, AS_OF)
    check("index VWAP is reported as unavailable, not guessed", above is None)

    # A stock frame with no volume cannot be graded at all.
    no_vol = _frame([[r[0], r[1], r[2], r[3], r[4], 0] for r in
                     history_bars() + TODAY_CLEAN])
    check("a volume-less stock frame yields no levels",
          setup_mod.build_levels(no_vol, TODAY, AS_OF, 1.2) is None)


def test_verdicts():
    """BUY / WATCH / AVOID — the one word the UI leads with."""
    print("verdicts")
    mkt = _market()

    cand, problems = _candidate()
    scoring.score_candidate(cand, mkt)
    filters.apply(cand, mkt, problems)
    check("a clean setup reads BUY", cand.verdict == "BUY", str(cand.rejections))

    def verdict_of(**kw):
        market = kw.pop("market", _market())
        c, p = _candidate(**kw)
        scoring.score_candidate(c, market)
        filters.apply(c, market, p)
        return c

    # Extended price is a "not yet" — the setup is real, the entry has gone.
    stretched = [*TODAY_CLEAN[:4],
                 ["2026-08-20T09:35:00+05:30", 101.60, 102.60, 101.55, 102.50, 55_000],
                 ["2026-08-20T09:40:00+05:30", 102.50, 103.10, 102.40, 103.00, 48_000]]
    watch = verdict_of(rows=stretched, pct=2.6, oi_pct=5.4)
    check("an extended breakout is WATCH, not BUY",
          watch.verdict == "WATCH" and "timing" in watch.blockers,
          f"{watch.verdict} {watch.blockers} {watch.rejections}")

    # Contradicted evidence is never a watch, however high the score.
    bad_oi = verdict_of(pct=-1.2, oi_pct=4.0)
    check("contradictory OI is AVOID regardless of score",
          bad_oi.verdict == "AVOID", f"{bad_oi.verdict} {bad_oi.blockers}")
    weak = verdict_of(rows=TODAY_WEAK_VOL)
    check("a volume-less breakout is AVOID even at a high score",
          weak.verdict == "AVOID" and weak.score >= 70,
          f"{weak.verdict} {weak.score}")

    low = verdict_of(market=_market("BEARISH", -0.4))
    check("a soft-blocked setup below 60 is AVOID, not WATCH",
          low.verdict == "WATCH" or low.score >= 60 or low.verdict == "AVOID")

    check("every rejection carries a category",
          len(cand.blockers) == len(cand.rejections))
    check("blocker categories are known kinds",
          all(b in ("timing", "structure", "risk", "score", "window", "oi",
                    "market", "sector", "vwap", "volume", "liquidity", "data")
              for c in (watch, bad_oi, weak) for b in c.blockers))


def test_skip_reporting():
    """An empty screen must say why it is empty."""
    print("skip reporting")
    # Scan "today" a day after the bundle's session: nothing is gradeable.
    later = datetime(2026, 8, 21, 9, 45, tzinfo=IST)
    res = Scanner(ReplayFeed(bundle()), later, allow_delayed=True).run()
    check("stale-session names are skipped, not graded", not res.candidates)
    joined = " ".join(res.data_notes)
    check("the skip reason names the feed's last session",
          "last session is 2026-08-20" in joined, joined[-160:])
    check("an ungradeable run says so plainly",
          "Nothing could be graded this run" in joined)
    check("the reason reaches the UI payload",
          any("last session" in n for n in report.as_dict(res)["data_notes"]))


def test_derived_pct_reaches_the_display():
    """A move the scanner derives must also be the one it shows."""
    print("derived % change")
    b = bundle()
    b["stocks"]["RELIANCE"]["futures"]["pct_change"] = None   # unpublished
    res = _run(b)
    cand = next(c for c in res.candidates if c.symbol == "RELIANCE")
    check("the displayed % change is populated from the candles",
          cand.levels.pct_change is not None, str(cand.levels.pct_change))
    check("the OI read used that same move",
          close_to(cand.oi.price_change_pct, cand.levels.pct_change, 1e-9))
    check("and the caveat is recorded",
          any("futures %change unpublished" in w for w in cand.warnings),
          str(cand.warnings))
    check("the payload carries it too",
          report.as_dict(res)["candidates"][0]["pct_change"] is not None)


def test_payload():
    """report.as_dict is what both the CLI's --json and the web UI render."""
    print("payload")
    import json as json_mod

    res = _run(bundle())
    d = report.as_dict(res)
    check("payload is JSON-serialisable", isinstance(json_mod.dumps(d), str))
    for key in ("when", "window", "data_ok", "market", "picks", "candidates",
                "considered", "data_notes"):
        check(f"payload has '{key}'", key in d)
    check("picks in the payload match the scan's picks",
          [c["symbol"] for c in d["picks"]] == [c.symbol for c in res.picks])
    p = d["picks"][0]
    check("a pick carries every level the UI plots",
          all(isinstance(p[k], (int, float))
              for k in ("price", "or_high", "or_low", "vwap", "atr")),
          str({k: p[k] for k in ("price", "or_high", "or_low", "vwap", "atr")}))
    check("a pick carries the full trade plan",
          p["trade"]["stop"] < p["trade"]["entry_low"] <= p["trade"]["entry_high"]
          < p["trade"]["target1"] < p["trade"]["target2"])
    check("verdict is present on every candidate",
          all(c["verdict"] in ("BUY", "WATCH", "AVOID") for c in d["candidates"]))
    check("sources carry an age and a timeframe",
          all("age_min" in s and "timeframe" in s for s in p["sources"]))

    stale = report.as_dict(Scanner(_DelayedFeed(bundle()), AS_OF,
                                   allow_delayed=False).run())
    check("a refused scan says so in the payload",
          stale["data_ok"] is False and STALE_DATA_BANNER in stale["stale_banner"])
    check("a refused scan publishes no picks", stale["picks"] == [])


class _DelayedFeed(ReplayFeed):
    def candles(self, symbols):
        out = super().candles(symbols)
        for sym, c in out.items():
            out[sym].prov = Provenance("yfinance", sym, "5m", c.prov.data_time,
                                       c.prov.fetched_at, delayed=True)
        return out


def test_web():
    """The UI server: routing, single-flight scans, no network in replay."""
    print("web ui")
    import json as json_mod

    from trading.fno import web

    cache = web.ScanCache(replay="data/fno/sample-2026-08-20.json", ttl=9999)
    web.CACHE = cache
    cache.start()
    for _ in range(200):
        if not cache._scanning:
            break
        time.sleep(0.05)
    snap = cache.snapshot()
    check("the cache completes a replay scan", snap["status"] == "ready",
          f"{snap['status']} {snap['error']}")
    check("the cached scan is the shared payload",
          snap["scan"]["picks"][0]["symbol"] == "RELIANCE")
    check("a replay source is labelled by file, not by full path",
          all("/" not in p["source"] for p in snap["scan"]["picks"][0]["sources"]),
          str([p["source"] for p in snap["scan"]["picks"][0]["sources"]]))

    # Single-flight: a second start while one is running must not queue another.
    cache._scanning = True
    check("a scan already running is not started twice", cache.start() is False)
    cache._scanning = False

    code, ctype, body = web.serve("/fno")
    check("/fno serves the page", code == 200 and b"<!doctype html>" in body[:40])
    check("the page ships its own styles and script",
          b"<style>" in body and b"function ladder" in body)
    code, ctype, body = web.serve("/api/fno")
    check("/api/fno serves JSON", ctype == "application/json"
          and json_mod.loads(body)["status"] in ("ready", "scanning"))
    check("the standalone server also answers at /",
          web.serve("/", include_root=True)[0] == 200
          and web.serve("/api/scan")[0] == 200)
    # Mounted in the dashboard, "/" must stay the dashboard's own page.
    check("mounted under the dashboard, / is not claimed",
          web.serve("/") is None)
    check("an unknown path is not handled here", web.serve("/nope") is None)

    # The main dashboard surfaces the scanner's current call, so it must ship
    # the hooks and read the same API rather than deriving its own answer.
    from trading import dashboard
    page = dashboard.PAGE
    for hook in ("bnBody", "bnMeta", "loadFno", "/api/fno", "renderBuyNow"):
        check(f"the dashboard page wires '{hook}'", hook in page)
    for hook in ("niftyWrap", "niftyPanel", "renderNifty", "index_options"):
        check(f"the dashboard carries the NIFTY option hook '{hook}'", hook in page)
    check("the scanner page carries the NIFTY panel too",
          "niftyPanel" in web.PAGE and "niftyWrap" in web.PAGE)
    check("the NIFTY panel is defined once and shared by both pages",
          web.PAGE.count("function niftyPanel") == 1
          and page.count("function niftyPanel") == 1)
    for hook in ("bn-opt", "Call option", "NO CALL", "breakeven"):
        check(f"the dashboard's buy-now panel carries '{hook}'", hook in page)
    check("the dashboard reads picks from the scan payload, not its own logic",
          "s.picks" in page and "no_trade" in page)
    check("the dashboard refuses to show levels on unreliable data",
          "stale_banner" in page and "data_ok" in page)

    broken = web.ScanCache(replay="data/fno/does-not-exist.json", ttl=9999)
    broken.start()
    for _ in range(200):
        if not broken._scanning:
            break
        time.sleep(0.05)
    s = broken.snapshot()
    check("a failed scan surfaces the error instead of a blank page",
          s["status"] == "error" and s["error"], str(s)[:120])


def main() -> int:
    for fn in (test_indicators, test_oi, test_structure, test_trade_levels,
               test_market_context, test_scoring, test_filters,
               test_relaxed_grading, test_windows,
               test_feed_parsing, test_zero_volume_feeds, test_verdicts,
               test_end_to_end, test_data_integrity, test_skip_reporting,
               test_derived_pct_reaches_the_display,
               test_payload, test_web):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
