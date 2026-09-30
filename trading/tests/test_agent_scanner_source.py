"""The F&O long scanner as a candidate source for the AI agent.

These tests are deliberately offline: no network, no LLM, no broker. They pin
the properties that make the join safe rather than merely functional --

* the scanner supplies the NAME and the PLAN, and nothing else. The 200MA
  trend, the opening range, VWAP and volume are still computed from real bars,
  so "the council's ORB rules still apply" is a real evaluation;
* a WATCH card with no structural stop is skipped, never given an invented one;
* the 200MA trend gate the router gives up for `fno_scanner` is re-applied in
  code -- the test proves the router really does skip it, so the agent-side
  check cannot be quietly deleted later as redundant;
* with the feature off, nothing changes at all.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from trading.agents import agent_trader as A
from trading.config import WATCHLIST
from trading.execution_router import ExecutionRouter
from trading.fno import live as L
from trading.fno.models import (
    Candidate, FuturesSnapshot, IST, Levels, MarketContext, OIRead, Provenance,
    ScanResult, Structure, Trade,
)
from trading.market_facts import build_market_facts

WHEN = datetime(2026, 9, 29, 15, 30, tzinfo=IST)


@pytest.fixture
def temp_ledger():
    from trading.ledger import Ledger
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Ledger(db_path=Path(tmpdir) / "test_ledger.db")


# --- synthetic scan objects ------------------------------------------------

def make_setup(*, trend="up", price=2441.5, or_high=2343.10, or_pct=1.87):
    """A strategy_setup shaped exactly like ORBMA200Strategy.compute_setup.

    `signal` is None throughout: the whole point of the scanner path is that
    the ORB strategy has NOT fired, so every fact the council sees about the
    trend and the opening range has to come from the bars, not the scanner.
    """
    return {
        "signal": None,
        "symbol": "GLENMARK",
        "price": price,
        "ts": WHEN,
        "orb": {"or_high": or_high, "or_low": 2300.0, "or_width_pct": or_pct,
                "or_valid": True},
        "trend": {"price": price, "sma200_1h": 2200.0, "sma200_30m": 2250.0,
                  "state": trend},
        "volume": {"breakout_bar_vol": 120000.0, "avg20": 80000.0, "ratio": 1.5,
                   "bar_complete": True},
        "vwap": {"value": 2380.0, "slope": "up", "aligned": True},
        "atr": 18.0,
        "buy_stop": 2320.0, "buy_target": 2500.0,
        "sell_stop": 2310.0, "sell_target": 2200.0,
    }


def make_trade(*, entry=2360.0, stop=2340.0, t1=2400.0, t2=2440.0,
               entry_low=2350.0, entry_high=2370.0):
    risk = entry - stop
    return Trade(entry_low=entry_low, entry_high=entry_high, entry=entry, stop=stop,
                 stop_basis="below the breakout bar low", target1=t1, target2=t2,
                 risk=risk, rr1=(t1 - entry) / risk, rr2=(t2 - entry) / risk)


def make_candidate(*, symbol="GLENMARK", trade=None, score=79.0,
                   rejections=(), blockers=(), rvol=1.8, rel=1.2, price=2441.5):
    prov = Provenance(source="test", instrument=symbol, timeframe="5m",
                      data_time=WHEN, fetched_at=WHEN)
    fut = FuturesSnapshot(symbol=symbol, contract=f"{symbol}FUT", expiry="2026-10-29",
                          last_price=price, prev_close=2400.0, pct_change=1.73,
                          open_interest=1_000_000.0, oi_change_pct=4.2,
                          volume=500_000.0, turnover_cr=1200.0,
                          bid=price - 0.5, ask=price + 0.5, prov=prov)
    levels = Levels(price=price, or_high=2343.10, or_low=2300.0, vwap=2380.0,
                    vwap_slope=0.4, ema_fast=2400.0, ema_slow=2380.0,
                    ema_fast_slope=0.2, ema_slow_slope=0.1, atr=18.0,
                    prev_high=2400.0, prev_low=2300.0, prev_close=2400.0,
                    day_high=2450.0, day_low=2310.0, support=2343.10,
                    resistance=2343.10, overhead=2500.0, rvol=rvol,
                    last_bar_vol=100000.0, avg_bar_vol=60000.0, pct_change=1.73)
    structure = Structure(higher_highs=True, higher_lows=True, consolidating=False,
                          breakout=True, breakout_time=WHEN, breakout_vol_mult=1.7,
                          holding=True, retested=False, retest_low=None,
                          failed_breakout=False, rejection=False, extended=True,
                          notes=["breakout bar closed above the level"])
    oi = OIRead(classification="LONG BUILDUP", bullish=True, reliable=True,
                detail="price up, open interest up", oi_change_pct=4.2,
                price_change_pct=1.73)
    return Candidate(symbol=symbol, sector="PHARMA", fut=fut, levels=levels,
                     structure=structure, oi=oi, rel_strength=rel,
                     sector_strength=0.8, score=score, grade="B",
                     trade=trade, rejections=list(rejections),
                     blockers=list(blockers))


def make_result(cands, *, data_ok=True, signals_allowed=True,
                window="POST-10:00 MONITORING", window_note="", notes=()):
    return ScanResult(
        when=WHEN, window=window,
        market=MarketContext(classification="BEARISH", nifty_pct=-0.28,
                             banknifty_pct=-0.39, nifty_above_vwap=True,
                             breadth_pct=40.0, advances=18, declines=27),
        candidates=list(cands), considered=len(cands),
        data_ok=data_ok, data_notes=list(notes),
        signals_allowed=signals_allowed, window_note=window_note,
    )


def make_ctx(plan, *, setup=None, ltp=None):
    """The subset of fetch_symbol_context's return that the agent reads."""
    setup = setup or make_setup()
    facts = build_market_facts(
        symbol=plan.symbol,
        strategy_setup=A.facts_setup_with_plan(setup, plan),
        market_direction={"nifty_pct": -0.28, "direction": "opposes"},
        asof=WHEN,
    )
    return {"symbol": plan.symbol, "signal": plan, "facts": facts,
            "ltp": setup["price"] if ltp is None else ltp,
            "or_pct": setup["orb"]["or_width_pct"],
            "vol_ratio": setup["volume"]["ratio"],
            "_setup": setup}


class _SpyRouter:
    """Records what would have been routed, without a risk kernel."""

    def __init__(self, ledger=None):
        self.ledger = ledger
        self.day_config = {"risk_multiplier": 0.5, "regime": "trending",
                           "blocked_symbols": []}
        self.last_rejection = None
        self.signals: list[dict] = []

    def execute(self, signal):
        self.signals.append(signal)
        return 42


def _buy_rec(model, **over):
    fields = dict(symbol="GLENMARK", action="BUY", entry_price=1.0, stop_loss=0.5,
                  target=2.0, conviction=9, technical_analysis="t", bull_case="b",
                  bear_case="r", decision_rationale="d")
    fields.update(over)
    return model(**fields)


# --- 1. the plan travels; the facts do not -------------���-------------------

def test_scanner_plan_reaches_market_facts_unchanged():
    setup = make_setup(trend="mixed")
    trade = make_trade(entry=2360.0, stop=2340.0, t1=2400.0, t2=2440.0)

    plan = A.scanner_signal("GLENMARK", setup, trade)
    facts = build_market_facts(symbol="GLENMARK",
                               strategy_setup=A.facts_setup_with_plan(setup, plan),
                               asof=WHEN)

    # The scanner's plan is what the council is asked to judge, to the paisa.
    assert facts.signal.side == "BUY"
    assert facts.signal.price == trade.entry
    assert facts.signal.stop_loss == trade.stop
    assert facts.signal.target == trade.target1
    assert facts.signal.rr == pytest.approx(trade.rr1, abs=0.01)

    # Everything else is the strategy's own reading of real bars -- the scanner
    # supplies no 200MA at all, so a wrong trend here would be fabricated.
    assert facts.trend.state == "mixed"
    assert facts.trend.sma200_1h == 2200.0
    assert facts.trend.sma200_30m == 2250.0
    assert facts.orb.or_high == 2343.10
    assert facts.vwap.value == 2380.0
    assert facts.volume.ratio == 1.5
    assert plan.trend_state == setup["trend"]["state"]


def test_facts_setup_leaves_the_original_setup_untouched():
    setup = make_setup()
    A.facts_setup_with_plan(setup, A.scanner_signal("GLENMARK", setup, make_trade()))
    assert setup["signal"] is None, "the ORB setup must not be mutated in place"


# --- 2. no plan, no trade --------------------------------------------------

def test_watch_candidate_without_a_plan_is_skipped():
    no_plan = make_candidate(trade=None, score=79.0,
                              rejections=["extended: 4.2% above the breakout level"],
                              blockers=["timing"])
    assert no_plan.verdict == "WATCH"          # the fixture is the case we mean

    src = A.scanner_candidates(make_result([no_plan]))
    assert src.candidates == []
    assert src.refusal is None
    assert [s for s, _ in src.skipped] == ["GLENMARK"]
    assert "no plan" in src.skipped[0][1]


def test_watch_candidate_with_a_plan_is_kept():
    with_plan = make_candidate(trade=make_trade(), score=79.0,
                               rejections=["extended: 4.2% above the breakout level"],
                               blockers=["timing"])
    assert with_plan.verdict == "WATCH"

    src = A.scanner_candidates(make_result([with_plan]))
    assert [c.symbol for c in src.candidates] == ["GLENMARK"]


def test_avoid_candidates_are_not_picked_up():
    avoid = make_candidate(trade=make_trade(), score=63.0,
                           rejections=["weak volume"], blockers=["volume"])
    assert avoid.verdict == "AVOID"
    assert A.scanner_candidates(make_result([avoid])).candidates == []


# --- 3. the trend gate the router does not apply ---------------------------

def test_router_really_does_skip_the_trend_gate_for_fno_scanner(temp_ledger):
    """The premise of the agent-side check. If this ever fails, the router has
    grown the gate for fno_scanner too -- good news, but the agent check must
    then be re-justified rather than silently kept or deleted."""
    router = ExecutionRouter(mode="paper", ledger=temp_ledger, get_ltp=lambda s: 1000.0)
    router.day_config = {"risk_multiplier": 0.5, "regime": "trending",
                         "blocked_symbols": []}
    base = {"symbol": "GLENMARK", "side": "BUY", "qty": 1, "price": 1000.0,
            "stop_loss": 950.0, "target": 1100.0}

    # Both the "not a permitted state" branch and the side-specific branch fire
    # for an orb_ma200 id...
    assert router.risk_check({**base, "strategy_id": "orb_ma200_agents",
                              "trend_state": "mixed"}) == \
        "trend_gate_failed (trend state 'mixed' is not permitted)"
    assert router.risk_check({**base, "strategy_id": "orb_ma200_agents",
                              "trend_state": "down"}) == \
        "trend_gate_failed (BUY requires trend_state 'up', got 'down')"

    # ...and neither fires for fno_scanner.
    for state in ("mixed", "down"):
        scanner = router.risk_check({**base, "strategy_id": "fno_scanner",
                                     "trend_state": state})
        assert not str(scanner).startswith("trend_gate_failed")


@pytest.mark.parametrize("state", ["mixed", "down", "unavailable"])
def test_scanner_trend_gate_is_reapplied_in_code(state):
    setup = make_setup(trend=state)
    cand = make_candidate(trade=make_trade())
    plan = A.scanner_signal("GLENMARK", setup, cand.trade)

    why = A.scanner_preflight(cand, make_ctx(plan, setup=setup))
    assert why == f"scanner_trend_gate_failed (trend '{state}')"


# --- 4. the strategy tag survives all the way to the order -----------------

def test_scanner_trades_are_tagged_and_watchlist_trades_are_not(monkeypatch, temp_ledger):
    trade = make_trade()
    cand = make_candidate(trade=trade)
    setup = make_setup(trend="up", price=trade.entry)     # LTP inside the band
    plan = A.scanner_signal(cand.symbol, setup, trade)
    ctx = make_ctx(plan, setup=setup)

    monkeypatch.setattr(A, "fetch_symbol_context", lambda s, **kw: ctx)
    monkeypatch.setattr(A, "call_structured",
                        lambda **kw: _buy_rec(kw["output_model"]))
    monkeypatch.setattr(A, "notify", lambda *a, **kw: None)

    spy = _SpyRouter(temp_ledger)
    results, executed = A._run_scanner_pass(
        A.ScannerSource([cand]), spy, temp_ledger,
        min_conviction=7, budget=3, dry_run=False)

    assert executed == 1
    assert len(spy.signals) == 1
    sig = spy.signals[0]
    assert sig["strategy_id"] == "fno_scanner"
    assert sig["trend_state"] == "up"          # from the strategy, not the LLM
    assert sig["price"] == trade.entry
    assert sig["stop_loss"] == trade.stop
    assert sig["target"] == trade.target1
    assert results[0]["executed"] is True

    # The watchlist path keeps its own tag: execute_recommendation defaults to it.
    rec = _buy_rec(A.AgentTradeRecommendation, symbol="GLENMARK")
    A.execute_recommendation(rec, spy, min_conviction=7)
    assert spy.signals[1]["strategy_id"] == "orb_ma200_agents"


def test_dry_run_routes_nothing(monkeypatch, temp_ledger):
    trade = make_trade()
    cand = make_candidate(trade=trade)
    setup = make_setup(trend="up", price=trade.entry)
    plan = A.scanner_signal(cand.symbol, setup, trade)

    monkeypatch.setattr(A, "fetch_symbol_context",
                        lambda s, **kw: make_ctx(plan, setup=setup))
    monkeypatch.setattr(A, "call_structured",
                        lambda **kw: _buy_rec(kw["output_model"]))

    spy = _SpyRouter(temp_ledger)
    results, executed = A._run_scanner_pass(
        A.ScannerSource([cand]), spy, temp_ledger,
        min_conviction=7, budget=3, dry_run=True)

    assert executed == 0
    assert spy.signals == []
    assert results[0]["reason"] == "dry_run"


# --- 5. off means off ------------------------------------------------------

def test_disabled_scanner_is_a_strict_no_op(monkeypatch):
    monkeypatch.setattr(A, "AGENT_SCANNER_ENABLED", False)

    scanned = []
    monkeypatch.setattr("trading.fno.live.run_scan",
                        lambda **kw: scanned.append(kw) or make_result([]))

    fetched = []
    monkeypatch.setattr(A, "fetch_symbol_context",
                        lambda s, **kw: fetched.append(s) or None)
    monkeypatch.setattr(A, "ExecutionRouter", lambda **kw: _SpyRouter(None))

    A.scan_and_trade(source="scanner")

    assert scanned == [], "run_scan must not run while the feature is off"
    assert set(fetched) == set(WATCHLIST), "the symbol list is exactly today's"


def test_enabled_scanner_but_no_candidates_still_scans_the_watchlist(monkeypatch):
    monkeypatch.setattr(A, "AGENT_SCANNER_ENABLED", True)
    monkeypatch.setattr("trading.fno.live.run_scan",
                        lambda **kw: make_result([], data_ok=True))

    fetched = []
    monkeypatch.setattr(A, "fetch_symbol_context",
                        lambda s, **kw: fetched.append(s) or None)
    monkeypatch.setattr(A, "ExecutionRouter", lambda **kw: _SpyRouter(None))

    A.scan_and_trade(source="both")
    assert set(fetched) == set(WATCHLIST)


def test_dry_run_also_holds_back_the_watchlist_arm(monkeypatch, temp_ledger):
    """--dry-run is about not writing trades, so it cannot apply to only one arm."""
    setup = make_setup(trend="up")
    plan = A.scanner_signal("GLENMARK", setup, make_trade())
    ctx = make_ctx(plan, setup=setup)

    monkeypatch.setattr(A, "fetch_symbol_context", lambda s, **kw: ctx)
    monkeypatch.setattr(A, "call_structured",
                        lambda **kw: _buy_rec(kw["output_model"]))
    monkeypatch.setattr(A, "notify", lambda *a, **kw: None)

    spy = _SpyRouter(temp_ledger)
    monkeypatch.setattr(A, "ExecutionRouter", lambda **kw: spy)

    results = A.scan_and_trade(symbols=["GLENMARK"], dry_run=True)

    assert spy.signals == [], "dry run must not route anything"
    assert all(r["reason"] == "dry_run" for r in results if r["reason"])


def test_watchlist_arm_is_untouched_when_no_scanner_flag_is_given(monkeypatch, temp_ledger):
    """No scanner flag means today's code path: the same names, in the same
    order, reached with the same (unforced) news fetch."""
    monkeypatch.setattr(A, "AGENT_SCANNER_ENABLED", True)
    monkeypatch.setattr("trading.fno.live.run_scan",
                        lambda **kw: pytest.fail("the scanner must not be scanned"))

    seen = []

    def _ctx(symbol, **kwargs):
        seen.append((symbol, kwargs))
        return None

    monkeypatch.setattr(A, "fetch_symbol_context", _ctx)
    monkeypatch.setattr(A, "ExecutionRouter", lambda **kw: _SpyRouter(None))

    A.scan_and_trade()

    assert [s for s, _ in seen] == list(WATCHLIST)
    assert all(kw == {} for _, kw in seen), "the watchlist path passes no kwargs"


# --- 6. an unusable scan is refused, with the scanner's own reason ---------

def test_stale_scan_data_refuses_the_whole_source():
    res = make_result([make_candidate(trade=make_trade())], data_ok=False,
                      notes=["⚠ STALE DATA — the feed's last session is 2026-09-28"])
    src = A.scanner_candidates(res)
    assert src.candidates == []
    assert "not usable" in src.refusal
    assert "last session is 2026-09-28" in src.refusal


def test_out_of_window_scan_refuses_the_whole_source():
    res = make_result([make_candidate(trade=make_trade())],
                      signals_allowed=False, window="PRE-OPEN",
                      window_note="no signals are emitted before 09:15")
    src = A.scanner_candidates(res)
    assert src.candidates == []
    assert "PRE-OPEN" in src.refusal
    assert "no signals are emitted before 09:15" in src.refusal


def test_scan_and_trade_prints_the_refusal_and_buys_nothing(monkeypatch, capsys):
    monkeypatch.setattr(A, "AGENT_SCANNER_ENABLED", True)
    monkeypatch.setattr("trading.fno.live.run_scan",
                        lambda **kw: make_result([make_candidate(trade=make_trade())],
                                                 data_ok=False,
                                                 notes=["⚠ STALE DATA — delayed feed"]))
    monkeypatch.setattr(A, "ExecutionRouter", lambda **kw: _SpyRouter(None))

    A.scan_and_trade(source="scanner")

    out = capsys.readouterr().out
    assert "[scanner] REFUSED" in out
    assert "delayed feed" in out


def test_the_cap_keeps_the_best_and_says_what_it_dropped():
    cands = [make_candidate(symbol=f"SYM{i}", trade=make_trade(), score=90 - i)
             for i in range(5)]
    src = A.scanner_candidates(make_result(cands), limit=2)
    assert [c.symbol for c in src.candidates] == ["SYM0", "SYM1"]
    assert [s for s, _ in src.skipped] == ["SYM2", "SYM3", "SYM4"]
    assert "cap of 2" in src.skipped[0][1]


# --- 7. the entry band: do not chase ---------------------------------------

def test_entry_band_guard_blocks_a_chase_and_can_be_switched_off():
    # The trade's band is 2350-2370; LTP is 2441.5, i.e. GLENMARK's real
    # situation -- 4.2% above its breakout level.
    setup = make_setup(trend="up", price=2441.5)
    trade = make_trade(entry=2360.0, stop=2340.0)
    cand = make_candidate(trade=trade)
    plan = A.scanner_signal("GLENMARK", setup, trade)
    ctx = make_ctx(plan, setup=setup)

    why = A.scanner_preflight(cand, ctx)
    assert why is not None and "outside entry band 2350.00-2370.00" in why
    assert "not chasing" in why

    # Explicitly allowing the chase lets it through -- this is the flag that
    # bounds the effect of including WATCH names at all.
    assert A.scanner_preflight(cand, ctx, require_entry_band=False) is None

    # And inside the band it passes without any override.
    ctx["ltp"] = 2360.0
    assert A.scanner_preflight(cand, ctx) is None


def test_entry_band_uses_the_live_price_not_the_scanners_frozen_one():
    """The scanner's own price was inside the band when it scanned; by the time
    the agent acts the price has run. The guard must follow the live number."""
    setup = make_setup(trend="up", price=2360.0)          # in-band at scan time
    trade = make_trade(entry=2360.0, stop=2340.0)
    cand = make_candidate(trade=trade, price=2360.0)
    plan = A.scanner_signal("GLENMARK", setup, trade)

    assert A.scanner_preflight(cand, make_ctx(plan, setup=setup)) is None
    moved = make_ctx(plan, setup=setup, ltp=2441.5)
    assert "outside entry band" in A.scanner_preflight(cand, moved)


# --- 8. feed selection -----------------------------------------------------

def _stub_client(connected):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def status(self):
            return connected, "stub"

    return _Client


def test_pick_feed_prefers_a_replay():
    feed, now, allow = L.pick_feed(replay={"as_of": WHEN.isoformat()})
    from trading.fno.data import ReplayFeed
    assert isinstance(feed, ReplayFeed)
    assert now == feed.as_of == WHEN
    assert allow is True


def test_pick_feed_uses_fyers_when_a_token_exists(monkeypatch):
    from trading.fno import fyers as F
    import trading.fno.data as D

    monkeypatch.setattr(F, "load_token", lambda: ("token", None))
    monkeypatch.setattr(F, "FyersClient", _stub_client(False))
    feed, _, allow = L.pick_feed()
    assert isinstance(feed, F.FyersFeed) and isinstance(feed, D.LiveFeed)
    assert allow is True


def test_pick_feed_uses_fyers_when_the_bridge_is_connected(monkeypatch):
    from trading.fno import fyers as F

    monkeypatch.setattr(F, "load_token", lambda: (None, None))
    monkeypatch.setattr(F, "FyersClient", _stub_client(True))
    feed, _, _ = L.pick_feed()
    assert isinstance(feed, F.FyersFeed)


def test_pick_feed_falls_back_to_live_and_carries_allow_delayed(monkeypatch):
    from trading.fno import fyers as F
    from trading.fno.data import LiveFeed

    monkeypatch.setattr(F, "load_token", lambda: (None, None))
    monkeypatch.setattr(F, "FyersClient", _stub_client(False))

    feed, _, allow = L.pick_feed(allow_delayed=False)
    assert isinstance(feed, LiveFeed) and not isinstance(feed, F.FyersFeed)
    assert allow is False

    # --fyers forces the broker feed even with no token on disk and no bridge.
    feed2, _, _ = L.pick_feed(fyers=True)
    assert isinstance(feed2, F.FyersFeed)


def test_run_scan_returns_a_scan_result_not_a_dict(monkeypatch):
    """The agent needs the objects; only the web layer renders JSON."""
    from trading.fno.scanner import Scanner

    monkeypatch.setattr(Scanner, "run", lambda self: make_result([]))
    out = L.run_scan(replay={"as_of": WHEN.isoformat()}, with_options=False)
    assert isinstance(out, ScanResult)
