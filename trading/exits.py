"""Deterministic trailing-stop engine: breakeven move, ATR chandelier trail, partial booking.

Pure calculation. This module reads nothing from the network, the ledger, or the LLM,
so it is fast, reproducible, and backtestable -- and unit-testable against synthetic
price paths with no fixtures at all. Every exit site in the system (shadow resolution,
research harness, backtest engine, live paper loop) calls the SAME function here, so
backtest and live behaviour cannot drift apart.

Rules
-----
1. Breakeven move. Once favourable excursion reaches ``breakeven_at_r`` (1.0R default),
   the stop moves to entry +/- ``breakeven_buffer``. The buffer is derived from the real
   round-trip cost via :func:`trading.costs.breakeven_move`, NOT a guessed constant.
2. ATR chandelier trail. Once armed, the stop trails at
   ``highest_close_since_entry - trail_atr_mult * ATR`` for a BUY (mirrored for a SELL).
   The stop is MONOTONIC: it only ever moves in the favourable direction.
3. Optional partial booking at ``partial_at_r``: the caller is told how many shares to
   close; it decides the fill price and books the charges.
4. Square-off is the caller's business. This module never overrides it -- a caller that
   still force-closes at SQUAREOFF_TIME does so regardless of trail state.

Why the signature differs from the obvious one
---------------------------------------------
A natural signature is ``(entry, side, current_stop, bars_since_entry, atr_series, config)``.
That cannot implement rule 2: ``highest_close_since_entry`` is not derivable from those
arguments. So the caller owns a small :class:`TrailState` accumulator and passes the
current bar's close and ATR value. ``risk_per_share`` is likewise required -- 1R is the
ORIGINAL entry-to-stop distance and cannot be recovered from a stop that has already moved.

Why the breakeven buffer includes slippage
------------------------------------------
``round_trip()`` covers statutory charges (brokerage, STT, exchange, SEBI, stamp, GST).
Every fill in this system is ALSO slippage-adjusted on both legs, so a stop placed at
``entry + round_trip_buffer`` still exits at a small net loss. The buffer therefore adds
``2 * slippage_pct * entry`` on top, making a breakeven exit genuinely flat. Using
``round_trip()`` alone would quietly book a loss on every "breakeven" trade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from trading.costs import breakeven_move, round_trip


@dataclass(frozen=True)
class TrailConfig:
    """Trailing-stop parameters. ``enabled=False`` is a strict no-op."""

    enabled: bool = False
    breakeven_at_r: float = 1.0
    trail_atr_mult: float = 2.0
    # ``None`` disables partial booking entirely.
    partial_at_r: float | None = 1.5
    partial_pct: float = 0.5
    # ``None`` resolves to trading.config.SLIPPAGE_PCT at call time. Tests inject here.
    slippage_pct: float | None = None


@dataclass(frozen=True)
class TrailState:
    """Per-position accumulator owned by the caller and threaded bar to bar."""

    highest_close: float | None = None
    lowest_close: float | None = None
    partial_taken: bool = False
    armed: bool = False  # breakeven reached -> trail is live


def _resolve_slippage(slippage_pct: float | None) -> float:
    if slippage_pct is not None:
        return slippage_pct
    from trading.config import SLIPPAGE_PCT
    return SLIPPAGE_PCT


def trail_config_from_settings(**overrides) -> TrailConfig:
    """Build a :class:`TrailConfig` from ``trading.config``'s toggle and knobs.

    Every exit site calls this so a single config flag moves all of them together and
    backtest/live cannot drift. ``overrides`` exists for the sensitivity grid in the
    research harness. Still pure -- config constants only, no network, ledger or LLM.
    """
    from trading.config import (
        BREAKEVEN_AT_R,
        PARTIAL_AT_R,
        PARTIAL_PCT,
        TRAIL_ATR_MULT,
        TRAIL_ENABLED,
    )

    base = dict(
        enabled=TRAIL_ENABLED,
        breakeven_at_r=BREAKEVEN_AT_R,
        trail_atr_mult=TRAIL_ATR_MULT,
        partial_at_r=PARTIAL_AT_R,
        partial_pct=PARTIAL_PCT,
    )
    base.update(overrides)
    return TrailConfig(**base)


def breakeven_buffer(entry: float, qty: int, slippage_pct: float | None = None) -> float:
    """Points of favourable move needed for a truly flat exit.

    Base is the real round-trip cost priced for the actual quantity
    (``breakeven_move`` is ``round_trip()/qty``, as the design spec requires), plus
    slippage on both legs.

    That base alone is a few paise short: ``breakeven_move`` prices the round trip at the
    ENTRY price, but STT and brokerage scale with order value, so the sell leg costs more
    once the stop sits above entry. The residual is closed by a short fixed-point
    refinement -- costs are near-linear in price, so this converges in 2-3 passes. Without
    it every "breakeven" exit books a small loss, which is exactly what rule 1 forbids.
    """
    if qty < 1 or entry <= 0:
        return float("inf")
    slip = _resolve_slippage(slippage_pct)
    buffer = breakeven_move(entry, qty) + 2.0 * slip * entry

    for _ in range(4):
        stop = entry + buffer
        entry_fill = entry * (1 + slip)
        exit_fill = stop * (1 - slip)
        net = (exit_fill - entry_fill) * qty - round_trip(entry_fill, exit_fill, qty)
        if net >= 0:
            break
        buffer += (-net) / qty
    return buffer


def update_exit(
    entry: float,
    side: str,
    current_stop: float,
    bar_close: float,
    atr: float,
    state: TrailState,
    *,
    qty: int,
    risk_per_share: float,
    config: TrailConfig,
) -> tuple[float, TrailState, int]:
    """Advance the exit for one closed bar.

    Returns ``(new_stop, new_state, partial_qty_to_close)``.

    ``new_stop`` is monotonic -- it never moves against the position. ``partial_qty_to_close``
    is 0 when no partial is due this bar; when non-zero the caller closes that many shares
    at its own market price and keeps the remainder open, still trailing.

    ``risk_per_share`` is 1R in price terms (the ORIGINAL |entry - stop|). It cannot be
    recovered from ``current_stop`` once the stop has moved.
    """
    if not config.enabled or qty < 1 or risk_per_share <= 0 or entry <= 0:
        return current_stop, state, 0

    buy = side == "BUY"

    # 1. Accumulate the favourable extreme. Closes, not highs/lows: the trail is defined
    #    against closes so a single wick cannot ratchet the stop.
    if buy:
        highest = bar_close if state.highest_close is None else max(state.highest_close, bar_close)
        lowest = state.lowest_close
    else:
        lowest = bar_close if state.lowest_close is None else min(state.lowest_close, bar_close)
        highest = state.highest_close

    # 2. Favourable excursion in R.
    favourable = (bar_close - entry) if buy else (entry - bar_close)
    r_multiple = favourable / risk_per_share

    # 3. Arm at breakeven. Sticky: once armed it stays armed even if price falls back.
    armed = state.armed or (r_multiple >= config.breakeven_at_r)

    # 4. Compute candidate stops and combine monotonically.
    # Candidates always include current_stop to guarantee monotonicity.
    candidates = [current_stop]
    if armed:
        buffer = breakeven_buffer(entry, qty, config.slippage_pct)
        if buffer != float("inf"):
            candidates.append(entry + buffer if buy else entry - buffer)

        # Chandelier trail: active once armed. Must be skipped when ATR is non-positive or NaN.
        if atr is not None and not math.isnan(atr) and atr > 0:
            if buy and highest is not None:
                candidates.append(highest - config.trail_atr_mult * atr)
            elif not buy and lowest is not None:
                candidates.append(lowest + config.trail_atr_mult * atr)

    new_stop = max(candidates) if buy else min(candidates)

    # 4. Partial booking -- instruction only, the caller fills and charges it.
    partial_qty = 0
    partial_taken = state.partial_taken
    if (
        not partial_taken
        and config.partial_at_r is not None
        and r_multiple >= config.partial_at_r
    ):
        # Never book the whole position, and never round up into one.
        wanted = int(qty * config.partial_pct)
        partial_qty = max(0, min(wanted, qty - 1))
        if partial_qty > 0:
            partial_taken = True

    new_state = replace(
        state,
        highest_close=highest,
        lowest_close=lowest,
        partial_taken=partial_taken,
        armed=armed,
    )
    return new_stop, new_state, partial_qty
