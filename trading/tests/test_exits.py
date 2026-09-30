"""Unit tests for the deterministic trailing-stop engine (trading/exits.py).

Synthetic price paths only -- no network, no ledger, no LLM. The five behaviours the
design spec requires, plus the no-op guarantee that makes ``enabled=False`` a true
control arm in the backtest comparison.
"""
import pytest

from trading.costs import breakeven_move, round_trip
from trading.exits import TrailConfig, TrailState, breakeven_buffer, update_exit

SLIP = 0.0005  # mirrors config.SLIPPAGE_PCT; injected so tests never depend on .env


def cfg(**kw) -> TrailConfig:
    base = dict(enabled=True, slippage_pct=SLIP)
    base.update(kw)
    return TrailConfig(**base)


def _net_buy(entry: float, stop: float, qty: int) -> float:
    """Net P&L of a BUY entered at `entry` and stopped at `stop`, with real fills/costs."""
    entry_fill = entry * (1 + SLIP)
    exit_fill = stop * (1 - SLIP)
    gross = (exit_fill - entry_fill) * qty
    return gross - round_trip(entry_fill, exit_fill, qty)


# --- 1. breakeven triggers exactly at 1R, net of round-trip cost -------------------

def test_breakeven_arms_at_exactly_1R_and_nets_flat():
    entry, qty, risk = 1300.0, 6, 10.0
    stop0 = entry - risk
    c = cfg()

    # Just short of 1R: must NOT arm.
    s, st, pq = update_exit(entry, "BUY", stop0, entry + 9.99, 0.0, TrailState(),
                            qty=qty, risk_per_share=risk, config=c)
    assert s == stop0
    assert st.armed is False
    assert pq == 0

    # Exactly 1R: arms, and the stop moves to entry + the cost-derived buffer.
    s, st, _ = update_exit(entry, "BUY", stop0, entry + risk, 0.0, TrailState(),
                           qty=qty, risk_per_share=risk, config=c)
    assert st.armed is True
    # The cost-derived base is a FLOOR, not the final answer: STT and brokerage scale with
    # the sell leg's value, so a small fixed-point correction sits on top. It is a
    # paise-level nudge, not a redesign -- anything larger means the model moved.
    base = entry + breakeven_move(entry, qty) + 2 * SLIP * entry
    assert s >= base - 1e-12
    assert s - base < 0.01, f"buffer correction unexpectedly large: {s - base:.6f}"

    # The real claim: an exit at that stop is genuinely flat, not a disguised loss.
    assert _net_buy(entry, s, qty) >= -1e-9

    # ...and that a round_trip-only buffer would NOT have been enough (the bug this guards).
    naive = entry + breakeven_move(entry, qty)
    assert _net_buy(entry, naive, qty) < 0


def test_breakeven_buffer_is_the_round_trip_cost_plus_slippage():
    """Not a guessed constant: the buffer is derived from costs.round_trip() for the
    actual quantity, plus slippage on both legs."""
    entry, qty = 1300.0, 6
    base = breakeven_move(entry, qty) + 2 * SLIP * entry
    buf = breakeven_buffer(entry, qty, SLIP)
    assert breakeven_move(entry, qty) == round_trip(entry, entry, qty) / qty
    assert buf >= base - 1e-12
    assert buf - base < 0.01, f"buffer correction unexpectedly large: {buf - base:.6f}"


def test_breakeven_buffer_unpriceable_returns_inf():
    assert breakeven_buffer(0.0, 6, SLIP) == float("inf")
    assert breakeven_buffer(1300.0, 0, SLIP) == float("inf")


# --- 2. the stop never moves backward ---------------------------------------------

def test_stop_is_monotonic_on_a_rising_then_falling_path_buy():
    entry, qty, risk = 100.0, 100, 2.0
    stop = 98.0
    state = TrailState()
    c = cfg(trail_atr_mult=2.0)
    closes = [100, 101, 102, 103, 104, 105, 106, 103, 101, 99, 99.5, 98.5]
    seen = [stop]
    for cl in closes:
        stop, state, _ = update_exit(entry, "BUY", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)
        seen.append(stop)
    assert seen == sorted(seen), f"stop went backward: {seen}"


def test_stop_is_monotonic_on_a_falling_then_rising_path_sell():
    entry, qty, risk = 100.0, 100, 2.0
    stop = 102.0
    state = TrailState()
    c = cfg(trail_atr_mult=2.0)
    closes = [100, 99, 98, 97, 96, 95, 94, 97, 99, 101, 100.5, 101.5]
    seen = [stop]
    for cl in closes:
        stop, state, _ = update_exit(entry, "SELL", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)
        seen.append(stop)
    assert seen == sorted(seen, reverse=True), f"stop went backward: {seen}"


# --- 3. a reversal locks the trailed level, not the original stop ------------------

def test_reversal_after_runup_locks_trailed_level_not_original_stop():
    entry, qty, risk = 100.0, 100, 2.0
    original_stop = 98.0
    stop = original_stop
    state = TrailState()
    c = cfg(trail_atr_mult=2.0)

    # Run up hard so the trail clears entry comfortably.
    for cl in (101, 103, 106, 109, 112):
        stop, state, _ = update_exit(entry, "BUY", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)
    peak, trailed = state.highest_close, stop
    assert peak == 112.0
    assert trailed == pytest.approx(112.0 - 2.0 * 1.0, abs=1e-9)  # peak - mult*ATR
    assert trailed > entry > original_stop

    # Reverse hard. The stop must HOLD the trailed level, not fall back.
    for cl in (110, 108, 106):
        stop, state, _ = update_exit(entry, "BUY", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)
    assert stop == pytest.approx(trailed, abs=1e-9)
    assert stop > original_stop


def test_trailing_stop_strictly_above_breakeven_on_pullback():
    """Price rises significantly and then falls back: the stop must be locked strictly
    ABOVE the breakeven level, proving active chandelier ratcheting (not just a breakeven move)."""
    entry, qty, risk = 100.0, 100, 2.0
    stop = 98.0
    state = TrailState()
    c = cfg(trail_atr_mult=2.0)
    be_level = entry + breakeven_buffer(entry, qty, SLIP)

    # Price advances to 110 (5R excursion) with ATR=1.0.
    # Chandelier level reaches 110 - 2*1.0 = 108.0.
    for cl in (101, 103, 106, 110):
        stop, state, _ = update_exit(entry, "BUY", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)

    assert state.highest_close == 110.0
    assert stop == pytest.approx(108.0, abs=1e-9)
    assert stop > be_level, f"stop {stop} should be strictly above breakeven {be_level}"

    # Price falls back towards breakeven: stop must stay firmly at 108.0.
    for cl in (107, 104, 101, 99):
        stop, state, _ = update_exit(entry, "BUY", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)
    assert stop == pytest.approx(108.0, abs=1e-9)
    assert stop > be_level


def test_reversal_after_rundown_locks_trailed_level_not_original_stop_sell():
    """SELL-side mirror: price runs down favourably, locking the chandelier trail
    (lowest_close + mult*ATR) which holds firmly when price reverses upward."""
    entry, qty, risk = 100.0, 100, 2.0
    original_stop = 102.0
    stop = original_stop
    state = TrailState()
    c = cfg(trail_atr_mult=2.0)

    # Run down hard so the trail clears entry comfortably.
    for cl in (99, 97, 94, 91, 88):
        stop, state, _ = update_exit(entry, "SELL", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)
    trough, trailed = state.lowest_close, stop
    assert trough == 88.0
    assert trailed == pytest.approx(88.0 + 2.0 * 1.0, abs=1e-9)  # trough + mult*ATR = 90.0
    assert trailed < entry < original_stop

    # Reverse hard upward. The stop must HOLD the trailed level, not rise back.
    for cl in (90, 92, 94, 98):
        stop, state, _ = update_exit(entry, "SELL", stop, float(cl), 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)
    assert stop == pytest.approx(trailed, abs=1e-9)
    assert stop < original_stop


def test_atr_zero_and_nan_skips_chandelier():
    """When ATR is 0 or NaN, the chandelier trail must be skipped so the stop never
    lands on the bar's close. Breakeven floor still holds once armed."""
    import math

    entry, qty, risk = 100.0, 100, 2.0
    stop = 98.0
    c = cfg(trail_atr_mult=2.0)
    be_level = entry + breakeven_buffer(entry, qty, SLIP)

    # 1. ATR = 0.0: Price arms at 104 (2R). If chandelier were close - 2*0, stop would be 104.0.
    # It must be skipped, keeping the stop at breakeven (~100.21).
    s_zero, st_zero, _ = update_exit(entry, "BUY", stop, 104.0, 0.0, TrailState(),
                                     qty=qty, risk_per_share=risk, config=c)
    assert st_zero.armed is True
    assert s_zero < 104.0, f"stop landed on close: {s_zero}"
    assert s_zero == pytest.approx(be_level, abs=1e-9)

    # 2. ATR = NaN: Must not produce NaN stop or blow up; must stay at breakeven.
    s_nan, st_nan, _ = update_exit(entry, "BUY", stop, 104.0, float("nan"), TrailState(),
                                   qty=qty, risk_per_share=risk, config=c)
    assert st_nan.armed is True
    assert not math.isnan(s_nan)
    assert s_nan == pytest.approx(be_level, abs=1e-9)

    # 3. Negative ATR: Must also skip chandelier.
    s_neg, _, _ = update_exit(entry, "BUY", stop, 104.0, -1.0, TrailState(),
                              qty=qty, risk_per_share=risk, config=c)
    assert s_neg == pytest.approx(be_level, abs=1e-9)



# --- 4. partial booking ------------------------------------------------------------

def test_partial_booking_reduces_qty_and_remainder_still_trails():
    entry, qty, risk = 100.0, 100, 2.0
    stop = 98.0
    state = TrailState()
    c = cfg(partial_at_r=1.5, partial_pct=0.5, trail_atr_mult=2.0)

    # Below the partial threshold: nothing booked.
    stop, state, pq = update_exit(entry, "BUY", stop, 102.0, 1.0, state,
                                  qty=qty, risk_per_share=risk, config=c)
    assert pq == 0 and state.partial_taken is False

    # At 1.5R: exactly half the position.
    stop, state, pq = update_exit(entry, "BUY", stop, 103.0, 1.0, state,
                                  qty=qty, risk_per_share=risk, config=c)
    assert pq == 50
    assert state.partial_taken is True

    # Sticky: a second call must not book again.
    stop, state, pq = update_exit(entry, "BUY", stop, 104.0, 1.0, state,
                                  qty=qty, risk_per_share=risk, config=c)
    assert pq == 0

    # The remainder still trails as price keeps rising.
    before = stop
    stop, state, _ = update_exit(entry, "BUY", stop, 108.0, 1.0, state,
                                 qty=qty, risk_per_share=risk, config=c)
    assert stop > before


def test_partial_never_closes_the_whole_position():
    entry, qty, risk = 100.0, 1, 2.0
    c = cfg(partial_at_r=1.5, partial_pct=0.5)
    _, _, pq = update_exit(entry, "BUY", 98.0, 110.0, 1.0, TrailState(),
                           qty=qty, risk_per_share=risk, config=c)
    assert pq == 0  # a 1-share position cannot be halved


def test_partial_disabled_when_threshold_is_none():
    c = cfg(partial_at_r=None)
    _, state, pq = update_exit(100.0, "BUY", 98.0, 120.0, 1.0, TrailState(),
                               qty=100, risk_per_share=2.0, config=c)
    assert pq == 0 and state.partial_taken is False


# --- 5. square-off still overrides an untriggered trail ----------------------------

def test_squareoff_overrides_untriggered_trail():
    """exits.py never blocks square-off: a position that never moved favourably keeps
    its structural stop, so the caller's 15:15 force-close still fires."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from trading.config import SQUAREOFF_TIME

    ist = ZoneInfo("Asia/Kolkata")
    entry, qty, risk = 100.0, 100, 2.0
    stop, target, original = 98.0, 104.0, 98.0
    state = TrailState()
    c = cfg()

    base = datetime(2026, 9, 28, 9, 15, tzinfo=ist)
    exit_price = exit_reason = None
    for i in range(73):  # 09:15 .. 15:15
        ts = base + timedelta(minutes=5 * i)
        hhmm = ts.strftime("%H:%M")
        # Choppy around entry: never reaches 1R, so the trail never arms.
        lo, hi, cl, op = 99.5, 100.6, 100.1, 100.0

        stop, state, _ = update_exit(entry, "BUY", stop, cl, 1.0, state,
                                     qty=qty, risk_per_share=risk, config=c)

        # Mirror shadow.py's BUY ordering: stop, then target, then square-off.
        if lo <= stop:
            exit_price, exit_reason = stop, "stop_loss_hit"
        elif hi >= target:
            exit_price, exit_reason = target, "target_hit"
        elif hhmm >= SQUAREOFF_TIME:
            exit_price, exit_reason = op, "squareoff_time"
        if exit_price is not None:
            break

    assert state.armed is False
    assert stop == original, "stop should be untouched when the trail never armed"
    assert exit_reason == "squareoff_time"
    assert exit_price == 100.0


# --- no-op guarantee: enabled=False is a true control arm --------------------------

@pytest.mark.parametrize("side,stop0", [("BUY", 98.0), ("SELL", 102.0)])
@pytest.mark.parametrize("close", [100.0, 105.0, 95.0, 120.0, 80.0])
def test_disabled_config_is_a_strict_noop(side, stop0, close):
    state = TrailState()
    s, st, pq = update_exit(100.0, side, stop0, close, 1.0, state,
                            qty=100, risk_per_share=2.0, config=TrailConfig(enabled=False))
    assert s == stop0
    assert st == state
    assert pq == 0


@pytest.mark.parametrize("risk", [0.0, -1.0])
def test_nonpositive_risk_is_a_noop(risk):
    s, _, pq = update_exit(100.0, "BUY", 98.0, 120.0, 1.0, TrailState(),
                           qty=100, risk_per_share=risk, config=cfg())
    assert s == 98.0 and pq == 0


def test_option_trailing_stop_breakeven_and_lock():
    from trading.costs import options_round_trip
    entry_prem = 35.0
    qty = 250
    risk = 10.5  # 30% stop: stop at 24.5
    stop0 = entry_prem - risk
    c = cfg(breakeven_at_r=1.0)

    # 1. Below 1R: stop stays at initial stop
    s, st, _ = update_exit(entry_prem, "BUY", stop0, 40.0, 0.0, TrailState(),
                           qty=qty, risk_per_share=risk, config=c, is_option=True)
    assert s == stop0
    assert st.armed is False

    # 2. At 1R (prem = 45.5): arms and moves to breakeven
    s, st, _ = update_exit(entry_prem, "BUY", stop0, 45.5, 0.0, TrailState(),
                           qty=qty, risk_per_share=risk, config=c, is_option=True)
    assert st.armed is True
    assert s > entry_prem, "Breakeven stop for option must be strictly above entry to cover flat brokerage"
    entry_fill = entry_prem * (1 + SLIP)
    exit_fill = s * (1 - SLIP)
    net_opt = (exit_fill - entry_fill) * qty - options_round_trip(entry_fill, exit_fill, qty)
    assert net_opt >= -1e-9, f"Option breakeven exit must not be negative: {net_opt}"

    # 3. At 2R (prem = 56.0): profit-lock kicks in, locking in 50% of the gain above 1R
    s_high, st_high, _ = update_exit(entry_prem, "BUY", s, 56.0, 0.0, st,
                                     qty=qty, risk_per_share=risk, config=c, is_option=True)
    assert s_high > s, "Option stop must trail upward as premium expands"
    assert s_high > entry_prem + 4.0, "Substantial profit must be locked in"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
