"""Correctness tests for the backtest harness, on synthetic candles.

The point of these is that a backtest that silently peeks at future bars will
still produce a plausible-looking equity curve, so the harness has to be
checked against cases whose answer is known by construction. Run with:

    .venv/bin/python -m trading.test_backtest
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from trading import strategy as sm
from trading.backtest import Backtester, build_strategy, metrics, params
from trading.costs import round_trip

IST = ZoneInfo("Asia/Kolkata")
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILURES.append(name)


# --- fixtures ------------------------------------------------------------

def session_index(n: int, *, start_day: str = "2026-08-03", step: int = 15,
                  open_hhmm: str = "09:15", close_hhmm: str = "15:15") -> pd.DatetimeIndex:
    """`n` bar timestamps laid across consecutive weekday NSE sessions.

    Bars must fall inside real session hours or the harness will (correctly)
    treat them as post-square-off and refuse entries — which is what made the
    first version of these tests silently trade nothing.
    """
    o = datetime.fromisoformat(f"2000-01-01T{open_hhmm}")
    c = datetime.fromisoformat(f"2000-01-01T{close_hhmm}")
    per_day = int((c - o).total_seconds() // 60 // step) + 1
    day = datetime.fromisoformat(f"{start_day}T{open_hhmm}").replace(tzinfo=IST)
    out: list[datetime] = []
    while len(out) < n:
        for b in range(min(per_day, n - len(out))):
            out.append(day + timedelta(minutes=step * b))
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
    return pd.DatetimeIndex(out)


def candles(closes: list[float], *, high_pad: float = 0.0, low_pad: float = 0.0,
            **kw) -> pd.DataFrame:
    idx = session_index(len(closes), **kw)
    return pd.DataFrame({
        "Open": closes,
        "High": [c + high_pad for c in closes],
        "Low": [c - low_pad for c in closes],
        "Close": closes,
        "Volume": [100_000.0] * len(closes),
    }, index=idx)


# A long decline (EMA9 below EMA21) followed by a sharp rise, so EMA9 crosses
# up a few bars into the rise. 50 warm-up bars = two full 15m sessions, so the
# cross lands early in session 3 with plenty of room before square-off.
DECLINE = [100.0 - i * 0.4 for i in range(50)]


def rising(n: int = 20, slope: float = 2.0) -> list[float]:
    base = DECLINE[-1]
    return [base + (i + 1) * slope for i in range(n)]


def bt(strategy: str = "ema", **kw) -> Backtester:
    kw.setdefault("slippage", 0.0)
    kw.setdefault("charges", 0.0)
    kw.setdefault("risk_per_trade", 500.0)
    return Backtester(build_strategy(strategy), **kw)


def entered(res) -> list:
    return list(res.trades)


# --- 1. no look-ahead ----------------------------------------------------

def test_no_lookahead_on_entry_bar():
    """A position opened at bar i's close may only be tested against bar i+1
    onward. The signal bar gets a spike low far below any stop: if the harness
    peeks at the entry bar's own wick it will report an instant stop-out."""
    df = candles(DECLINE + rising())
    with params(SWING_LOOKBACK=10):
        res = bt().run({"X": df})
    trades = entered(res)
    check("a trade was actually entered", bool(trades), f"rejections={res.rejections}")
    if not trades:
        return
    sig_ts = trades[0].entry_ts
    df.loc[sig_ts, "Low"] = 1.0          # spike only on the signal bar
    with params(SWING_LOOKBACK=10):
        res2 = bt().run({"X": df})
    t = entered(res2)[0]
    check("entry bar's own wick is not used for exits",
          t.exit is None or t.exit_ts > t.entry_ts,
          f"entry={t.entry_ts} exit={t.exit_ts} reason={t.reason}")
    check("spiking the entry bar does not change the entry price",
          abs(t.entry - trades[0].entry) < 1e-9, f"{t.entry} vs {trades[0].entry}")


# --- 2. exit fills -------------------------------------------------------

def test_stop_fill():
    """After entry, collapse the price so the swing stop is taken out."""
    closes = DECLINE + rising(6) + [40.0] * 6
    with params(SWING_LOOKBACK=10):
        res = bt().run({"X": candles(closes)})
    stops = [t for t in res.closed if t.reason == "stop"]
    check("stop-out recorded", bool(stops),
          f"reasons={[t.reason for t in res.closed]} rejections={res.rejections}")
    if stops:
        t = stops[0]
        check("stop fills at the stop price, not the bar low",
              abs(t.exit - t.sl) < 1e-9, f"exit={t.exit} sl={t.sl}")
        check("stopped trade is a loser", t.gross < 0, f"gross={t.gross}")


def test_target_fill():
    with params(SWING_LOOKBACK=10, RR_TARGET=2.0):
        res = bt().run({"X": candles(DECLINE + rising(20, slope=3.0))})
    hits = [t for t in res.closed if t.reason == "target"]
    check("target exit recorded", bool(hits),
          f"reasons={[t.reason for t in res.closed]} rejections={res.rejections}")
    if hits:
        t = hits[0]
        check("target fills at the target price", abs(t.exit - t.target) < 1e-9,
              f"exit={t.exit} target={t.target}")
        rr = (t.target - t.entry) / (t.entry - t.sl)
        check("target sits at RR_TARGET x risk", abs(rr - 2.0) < 0.02, f"rr={rr:.3f}")
        check("target trade is a winner", t.gross > 0, f"gross={t.gross}")


def test_stop_wins_ties():
    """When one bar spans both stop and target the stop must be assumed first
    — the conservative assumption the live engine also makes."""
    closes = DECLINE + rising(8)
    df = candles(closes)
    with params(SWING_LOOKBACK=10):
        probe = entered(bt().run({"X": df}))
    check("probe trade entered for tie test", bool(probe))
    if not probe:
        return
    # Make the bar AFTER entry straddle both levels.
    nxt = df.index[df.index.get_loc(probe[0].entry_ts) + 1]
    df.loc[nxt, "Low"] = 1.0
    df.loc[nxt, "High"] = 9_999.0
    with params(SWING_LOOKBACK=10):
        res = bt().run({"X": df})
    t = entered(res)[0]
    check("stop wins when a bar hits both levels", t.reason == "stop",
          f"reason={t.reason} exit={t.exit}")


# --- 3. risk kernel and sizing ------------------------------------------

def test_position_value_cap():
    with params(SWING_LOOKBACK=10):
        res = bt(risk_per_trade=1e9, max_position_value=50_000.0).run(
            {"X": candles(DECLINE + rising())})
    trades = entered(res)
    check("trade entered for sizing check", bool(trades), f"rejections={res.rejections}")
    if trades:
        t = trades[0]
        check("qty respects MAX_POSITION_VALUE", t.qty * t.entry <= 50_000.0,
              f"qty={t.qty} value={t.qty * t.entry:.0f}")
        check("cap actually bound the size (risk would have allowed more)",
              t.qty == int(50_000.0 / t.entry), f"qty={t.qty}")


def test_risk_per_trade_sizing():
    """When the position cap is not binding, qty must come from risk/share."""
    with params(SWING_LOOKBACK=10):
        res = bt(risk_per_trade=10_000.0, max_position_value=10_000_000.0).run(
            {"X": candles(DECLINE + rising())})
    trades = entered(res)
    check("trade entered for risk sizing check", bool(trades))
    if trades:
        t = trades[0]
        expected = int(10_000.0 / (t.entry - t.sl))
        check("qty = risk_per_trade / risk_per_share", t.qty == expected,
              f"qty={t.qty} expected={expected} risk/share={t.entry - t.sl:.4f}")


def test_max_open_positions():
    frames = {f"S{i}": candles(DECLINE + rising()) for i in range(5)}
    with params(SWING_LOOKBACK=10):
        res = bt(max_open=2).run(frames)
    per_bar: dict[pd.Timestamp, int] = {}
    for t in entered(res):
        per_bar[t.entry_ts] = per_bar.get(t.entry_ts, 0) + 1
    check("some trades entered", bool(per_bar), f"rejections={res.rejections}")
    check("max_open_positions caps concurrent entries",
          all(v <= 2 for v in per_bar.values()),
          f"per-bar={ {str(k): v for k, v in per_bar.items()} }")
    check("the cap was exercised", res.rejections.get("max_open_positions", 0) > 0,
          f"rejections={res.rejections}")


def test_daily_loss_limit():
    closes = DECLINE + rising(6) + [30.0] * 8
    frames = {f"S{i}": candles(closes) for i in range(4)}
    with params(SWING_LOOKBACK=10):
        res = bt(max_open=4, risk_per_trade=50_000.0, daily_loss_limit=-1_000.0).run(frames)
    check("daily loss limit blocked further entries",
          res.rejections.get("daily_loss_limit", 0) > 0, f"rejections={res.rejections}")


def test_squareoff():
    """No position may survive past the square-off time.

    Fixture: a sharp 6-bar rise crosses EMA9 up early in session 3, then price
    goes flat so neither the stop nor the 1:2 target can trigger — the only way
    out is the 15:15 square-off. 75 bars = exactly three 25-bar sessions, so
    the last bar IS 15:15 (a truncated session would exit on end_of_data).
    """
    up = rising(6, slope=2.0)
    closes = DECLINE + up + [up[-1]] * 19
    with params(SWING_LOOKBACK=10):
        res = bt(squareoff="15:15").run({"X": candles(closes)})
    late = [t for t in res.closed
            if t.exit_ts is not None and t.exit_ts.strftime("%H:%M") > "15:15"
            and t.reason not in ("end_of_data", "day_rollover")]
    check("some trades closed", bool(res.closed), f"rejections={res.rejections}")
    check("nothing held past square-off", not late,
          f"{[(str(t.exit_ts), t.reason) for t in late]}")
    check("square-off exits are recorded as such",
          any(t.reason == "squareoff" for t in res.closed) or not res.closed,
          f"reasons={[t.reason for t in res.closed]}")


def test_one_position_per_symbol():
    closes = DECLINE + rising(10) + [DECLINE[-1]] * 10 + rising(10)
    with params(SWING_LOOKBACK=10):
        res = bt().run({"X": candles(closes)})
    spans = sorted((t.entry_ts, t.exit_ts) for t in res.trades)
    overlap = [(a, b) for (a, ax), (b, _) in zip(spans, spans[1:]) if ax is None or b < ax]
    check("no overlapping positions in one symbol", not overlap, f"{overlap}")


# --- 4. costs and slippage ----------------------------------------------

def test_slippage_is_always_adverse():
    df = candles(DECLINE + rising())
    with params(SWING_LOOKBACK=10):
        clean = bt().run({"X": df})
        slipped = bt(slippage=0.01).run({"X": df})
    check("both runs traded", bool(clean.closed) and bool(slipped.closed),
          f"clean={len(clean.closed)} slipped={len(slipped.closed)}")
    if not (clean.closed and slipped.closed):
        return
    a, b = metrics(clean), metrics(slipped)
    check("slippage can only reduce gross P&L", b["gross"] < a["gross"],
          f"clean={a['gross']:.2f} slipped={b['gross']:.2f}")
    t, c = slipped.trades[0], clean.trades[0]
    worse = t.entry > c.entry if t.side == "BUY" else t.entry < c.entry
    check("entry slips against us", worse, f"{t.side} {t.entry:.4f} vs clean {c.entry:.4f}")
    check("entry slips by exactly the configured pct",
          abs(abs(t.entry / c.entry - 1) - 0.01) < 1e-9, f"{t.entry / c.entry - 1:.6f}")


def test_charges_match_cost_model():
    with params(SWING_LOOKBACK=10):
        res = Backtester(build_strategy("ema"), slippage=0.0).run(
            {"X": candles(DECLINE + rising())})
    t = next(iter(res.closed), None)
    check("closed trade exists for charge check", t is not None,
          f"rejections={res.rejections}")
    if t:
        check("charges equal trading.costs.round_trip",
              abs(t.charges - round_trip(t.entry, t.exit, t.qty)) < 1e-9,
              f"{t.charges} vs {round_trip(t.entry, t.exit, t.qty)}")
        check("net = gross - charges", abs(t.net - (t.gross - t.charges)) < 1e-9)
        check("charges are positive", t.charges > 0, f"{t.charges}")


def test_flat_charge_model_still_available():
    with params(SWING_LOOKBACK=10):
        res = Backtester(build_strategy("ema"), slippage=0.0, charges=0.0006).run(
            {"X": candles(DECLINE + rising())})
    t = next(iter(res.closed), None)
    check("flat model closed a trade", t is not None)
    if t:
        check("flat model = qty x (entry+exit) x pct",
              abs(t.charges - t.qty * (t.entry + t.exit) * 0.0006) < 1e-9, f"{t.charges}")


# --- 5. metrics ----------------------------------------------------------

def test_metrics_significance():
    """A CI that straddles zero must never be reported as an edge."""
    from trading.backtest import BTTrade, Result
    ts = pd.Timestamp("2026-08-03T10:00", tz=IST)

    def synth(grosses):
        r = Result(label="synthetic")
        for g in grosses:
            r.trades.append(BTTrade("X", "BUY", 1, ts, 100.0, 99.0, 102.0,
                                    exit_ts=ts, exit=100.0, gross=g, charges=0.0))
        return metrics(r)

    m = synth([500.0, -500.0] * 20)                     # mean exactly zero
    check("zero-mean sample is not called an edge",
          not m["significant"] and m["lo"] < 0 < m["hi"],
          f"exp={m['expectancy']} ci=({m['lo']:.2f},{m['hi']:.2f})")
    m2 = synth([100.0] * 200)                            # constant winner
    check("consistent winner is flagged as an edge", m2["significant"] and m2["lo"] > 0,
          f"ci=({m2['lo']:.2f},{m2['hi']:.2f})")
    m3 = synth([-100.0] * 200)                           # constant loser
    check("consistent loser is flagged negative", m3["significant"] and m3["hi"] < 0,
          f"ci=({m3['lo']:.2f},{m3['hi']:.2f})")
    m4 = synth([300.0, -100.0, -100.0, -100.0] * 25)     # PF and drawdown
    check("profit factor = gross win / gross loss",
          abs(m4["profit_factor"] - (300 * 25) / (100 * 75)) < 1e-9,
          f"{m4['profit_factor']}")
    check("max drawdown is negative or zero", m4["max_dd"] <= 0, f"{m4['max_dd']}")


# --- 6. pivot indicator --------------------------------------------------

def test_pivots_use_previous_day_only():
    """Pivot levels must come from yesterday's range, never today's."""
    closes = [100.0 + i for i in range(25)] + [200.0 + i for i in range(25)]
    prepared = sm.add_pivots(candles(closes))
    days = sorted({d for d in prepared.index.date})
    check("fixture spans two sessions", len(days) == 2, f"{days}")
    d1 = prepared.loc[prepared.index.date == days[0]]
    d2 = prepared.loc[prepared.index.date == days[1]]
    check("first session has no pivots (no prior day)", d1["pp"].isna().all())
    expected = (d1["High"].max() + d1["Low"].min() + d1["Close"].iloc[-1]) / 3
    check("day 2 pivot = day 1 H/L/C", abs(float(d2["pp"].iloc[0]) - expected) < 1e-9,
          f"got {float(d2['pp'].iloc[0]):.4f} want {expected:.4f}")
    check("pivot is constant within the session", d2["pp"].nunique() == 1)
    check("levels are ordered s3<s2<s1<pp<r1<r2<r3",
          bool(d2[["s3", "s2", "s1", "pp", "r1", "r2", "r3"]].iloc[0].is_monotonic_increasing),
          f"{d2[['s3', 's2', 's1', 'pp', 'r1', 'r2', 'r3']].iloc[0].to_dict()}")


def test_pivot_filter_is_a_subset_of_ema():
    """ema_pivot only gates the plain EMA cross, so it can never fire on a bar
    where plain ema would not."""
    df = candles(DECLINE + rising(20) + [DECLINE[-1]] * 25 + rising(20))
    with params(SWING_LOOKBACK=10):
        plain = {t.entry_ts for t in bt("ema").run({"X": df}).trades}
        gated = {t.entry_ts for t in bt("ema_pivot").run({"X": df}).trades}
    check("pivot-filtered entries are a subset of unfiltered ones",
          gated <= plain, f"extra={sorted(str(x) for x in gated - plain)}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {e!r}")
            FAILURES.append(f"{fn.__name__} raised {e!r}")
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILURE(S):\n  " + "\n  ".join(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
