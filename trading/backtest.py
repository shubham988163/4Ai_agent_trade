"""Cost-aware backtest harness — test a change without burning a live session.

Why this exists: `strategy.py --replay` writes real paper trades into the
ledger, so it can only ever test one config per day and it pollutes the
performance record it is supposed to measure. This module runs the *same*
signal functions from `trading.strategy` over historical candles with no
ledger, no LLM, and no side effects, so any number of configurations can be
compared against each other in seconds.

What it reproduces from the live path (deliberately, line for line):
  * the same prepare/detect/stop/target functions in `strategy.STRATEGIES`
  * `Engine.size()` sizing — RISK_PER_TRADE x multiplier, MAX_POSITION_VALUE cap
  * the `ExecutionRouter` risk kernel — max open positions, daily loss limit
  * conservative fills — stop assumed to fill first when a bar hits both levels
  * slippage on entry AND exit, plus the round-trip charge model
  * 15:15 IST square-off, one position per symbol

What it deliberately does NOT model (so read results as optimistic):
  * intrabar path — a bar that touches the stop is a stop, no tick sequencing
  * real spreads and queue position; SLIPPAGE_PCT is a flat assumption
  * the pre-market agent's blocked_symbols (no historical record of them)

The headline number is expectancy with a 95% confidence interval. A strategy
whose interval straddles zero has not been shown to have an edge, however
good the total looks — that distinction is the whole point of the harness.

Usage:
  python -m trading.backtest                          # default strategy, cached history
  python -m trading.backtest --strategy avwap
  python -m trading.backtest --interval 5m --period 60d
  python -m trading.backtest --symbols RELIANCE,INFY
  python -m trading.backtest --compare                # sweep strategies x intervals
  python -m trading.backtest --refresh                # force re-download
  python -m trading.backtest --charges 0 --slippage 0 # frictionless upper bound
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from trading import strategy as strat_mod
from trading.costs import round_trip as real_charges
from trading.exits import TrailState, trail_config_from_settings, update_exit
from trading.strategies.orb_ma200 import _atr
from trading.config import (
    SCAN_UNIVERSE, YF_SUFFIX, CANDLE_INTERVAL, RISK_PER_TRADE,
    MAX_POSITION_VALUE, MAX_OPEN_POSITIONS, DAILY_LOSS_LIMIT,
    SLIPPAGE_PCT, CHARGES_PCT_ROUND_TRIP, SQUAREOFF_TIME, PROJECT_ROOT,
)

IST = ZoneInfo("Asia/Kolkata")
CACHE_DIR = PROJECT_ROOT / "data" / "candles"

# yfinance intraday history limits (days) — the hard ceiling on a cold start.
YF_MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730}


# --- history: fetch + accumulating cache ---------------------------------

def _cache_path(symbol: str, interval: str):
    return CACHE_DIR / f"{symbol}_{interval}.pkl"


def _download(symbols: list[str], interval: str, period: str) -> dict[str, pd.DataFrame]:
    # threads=False: yfinance leaks a tz-cache sqlite fd per worker thread.
    # See the note in strategy.fetch_all_candles — a sweep is exactly the
    # workload that would exhaust the fd limit fastest.
    data = yf.download(
        tickers=[s + YF_SUFFIX for s in symbols],
        period=period, interval=interval,
        group_by="ticker", threads=False, progress=False, auto_adjust=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = data[sym + YF_SUFFIX] if len(symbols) > 1 else data
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        out[sym] = df.tz_convert(IST)
    return out


def live_engine_running() -> bool:
    """True when a live strategy/paper session is polling Yahoo right now.

    Matters because yfinance shares one rate limit and one sqlite tz-cache per
    machine. A sweep that re-downloads on every config will throttle the
    running session's data feed and lock its cache — which is exactly what
    happened on 2026-08-10 (six symbols went to 'possibly delisted', then
    OperationalError, mid-session). Backtests must reuse the cache by default.
    """
    try:
        r = subprocess.run(["pgrep", "-f", "trading.strategy"],
                           capture_output=True, text=True, timeout=5)
        pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
        return bool(pids)
    except Exception:  # noqa: BLE001 - pgrep absence must not block a backtest
        return False


def _cache_is_fresh(df: pd.DataFrame, interval: str) -> bool:
    """A cache is fresh when it already holds the newest bar that could have
    closed — i.e. re-downloading would add nothing."""
    bar_min = int(interval.rstrip("m")) if interval.endswith("m") else 60
    now = datetime.now(IST)
    last_close = datetime.combine(now.date(), datetime.strptime("15:30", "%H:%M").time(),
                                  tzinfo=IST)
    edge = min(now, last_close) - timedelta(minutes=bar_min)
    # Walk back over weekends/holidays: any bar within one session is fine.
    return bool(len(df)) and df.index[-1] >= edge - timedelta(days=4)


def load_history(symbols: list[str], interval: str = CANDLE_INTERVAL,
                 period: str | None = None, refresh: bool = False,
                 quiet: bool = False, download: str = "auto") -> dict[str, pd.DataFrame]:
    """Candles for `symbols`, merging a fresh download into the on-disk cache.

    yfinance only serves ~60 days of intraday history, so the cache is what
    lets the usable backtest window grow past that: every run appends the
    newest bars to whatever was stored before. Run it periodically and the
    history deepens on its own.

    download: "auto"   — download only when the cache is missing or stale, and
                         never while a live session is running (default)
              "always" — always refetch (use outside market hours)
              "never"  — cache only; fails loudly if the cache is empty
    """
    period = period or f"{YF_MAX_DAYS.get(interval, 60)}d"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = _cache_path(sym, interval)
        if path.exists() and not refresh:
            try:
                cache[sym] = pd.read_pickle(path)
            except Exception:  # noqa: BLE001 - a corrupt cache should not be fatal
                pass

    if download == "never":
        want = False
    elif download == "always" or refresh:
        want = True
    else:
        have_all = all(s in cache and _cache_is_fresh(cache[s], interval) for s in symbols)
        want = not have_all
        if want and live_engine_running():
            want = False
            if not quiet:
                print("! live session is running — using cached candles only so the\n"
                      "  backtest cannot throttle its Yahoo feed. Re-run with\n"
                      "  --download always after 15:30 IST to extend history.")

    fresh: dict[str, pd.DataFrame] = {}
    if want:
        try:
            fresh = _download(symbols, interval, period)
        except Exception as e:  # noqa: BLE001 - offline/rate-limited is expected
            if not quiet:
                print(f"download failed ({e!r}) — using cache only")

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        cached = cache.get(sym)
        new = fresh.get(sym)
        if cached is not None and new is not None:
            merged = pd.concat([cached, new])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new if new is not None else cached
        if merged is None or merged.empty:
            continue
        # Drop a still-forming final bar so we never backtest a partial candle.
        bar_min = int(interval.rstrip("m")) if interval.endswith("m") else 60
        if merged.index[-1] + timedelta(minutes=bar_min) > datetime.now(IST):
            merged = merged.iloc[:-1]
        if merged.empty:
            continue
        merged.to_pickle(path)
        out[sym] = merged
    return out


# --- simulation ----------------------------------------------------------

@dataclass
class BTTrade:
    symbol: str
    side: str
    qty: int
    entry_ts: pd.Timestamp
    entry: float
    sl: float
    target: float
    exit_ts: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    gross: float = 0.0
    charges: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0
    # Trailing-stop bookkeeping. ``initial_sl`` is 1R's anchor -- once the stop moves it
    # cannot be recovered from ``sl``. ``trail`` is the caller-owned accumulator.
    initial_sl: float | None = None
    trail: TrailState = field(default_factory=TrailState)

    @property
    def net(self) -> float:
        return self.gross - self.charges

    @property
    def date(self) -> str:
        return self.entry_ts.strftime("%Y-%m-%d")


@dataclass
class Result:
    label: str
    trades: list[BTTrade] = field(default_factory=list)
    bars: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def closed(self) -> list[BTTrade]:
        return [t for t in self.trades if t.exit is not None]


class Backtester:
    """One config, one run. Mirrors Engine + ExecutionRouter without the
    ledger, the notifier, or the agent layer."""

    def __init__(self, strat: dict, *, risk_per_trade: float = RISK_PER_TRADE,
                 risk_multiplier: float = 1.0,
                 max_position_value: float = MAX_POSITION_VALUE,
                 max_open: int = MAX_OPEN_POSITIONS,
                 daily_loss_limit: float = DAILY_LOSS_LIMIT,
                 slippage: float = SLIPPAGE_PCT,
                 charges: float | None = None,
                 squareoff: str = SQUAREOFF_TIME,
                 trail_config=None):
        """charges: None (default) uses the itemised Zerodha model in
        trading.costs, which prices the Rs-20-per-order brokerage cap
        correctly. Pass a float to fall back to the flat-percent model that
        strategy.py currently uses (for comparison), or 0.0 for a
        frictionless upper bound.

        trail_config: None (default) reads the system toggle
        (config.TRAIL_ENABLED, off), so this harness and the live paper loop
        share one implementation and cannot drift."""
        self.strat = strat
        self.risk_inr = risk_per_trade * risk_multiplier
        self.max_position_value = max_position_value
        self.max_open = max_open
        self.daily_loss_limit = daily_loss_limit
        self.slippage = slippage
        self.charges_pct = charges
        self.squareoff = squareoff
        self.trail_config = (
            trail_config if trail_config is not None else trail_config_from_settings()
        )

    def charges_for(self, entry: float, exit_price: float, qty: int) -> float:
        if self.charges_pct is None:
            return real_charges(entry, exit_price, qty)
        return qty * (entry + exit_price) * self.charges_pct

    # sizing: identical to Engine.size()
    def size(self, price: float, risk_per_share: float) -> int:
        if risk_per_share <= 0 or price <= 0:
            return 0
        return min(int(self.risk_inr / risk_per_share),
                   int(self.max_position_value / price))

    def _fill(self, price: float, side: str, entering: bool) -> float:
        """Slippage always works against us, on the way in and the way out."""
        adverse = (side == "BUY") == entering
        return price * (1 + self.slippage) if adverse else price * (1 - self.slippage)

    def run(self, frames: dict[str, pd.DataFrame], label: str = "") -> Result:
        # Indicators are causal (ewm / groupby-cumsum / rolling), so computing
        # them once over full history is identical to recomputing per bar --
        # and orders of magnitude faster than replay()'s O(n^2) slicing.
        for s, df in frames.items():
            if not df.empty and "symbol" not in df.attrs:
                df.attrs["symbol"] = s
        prepared = {s: self.strat["prepare"](df) for s, df in frames.items() if not df.empty}
        for s, df in prepared.items():
            if not df.empty and "symbol" not in df.attrs:
                df.attrs["symbol"] = s
        stamps = sorted({ts for df in prepared.values() for ts in df.index})
        res = Result(label=label or self.strat["id"], bars=len(stamps))

        # ATR(14) for the chandelier trail, from the same definition the strategy uses.
        # Computed once per symbol; only when trailing is on, so the default path is
        # untouched. ``_atr`` needs OHLC, which every prepared frame has.
        atr_by_sym: dict[str, pd.Series] = {}
        if self.trail_config.enabled:
            for s, df in prepared.items():
                if {"High", "Low", "Close"} <= set(df.columns):
                    atr_by_sym[s] = _atr(df, 14)

        open_pos: dict[str, BTTrade] = {}
        day_pnl = 0.0
        cur_day = None

        for ts in stamps:
            day = ts.strftime("%Y-%m-%d")
            if day != cur_day:
                # New session: flat overnight, loss limit resets.
                for sym in list(open_pos):
                    self._close(open_pos, sym, open_pos[sym].entry, ts, "day_rollover", res)
                cur_day, day_pnl = day, 0.0

            hhmm = ts.strftime("%H:%M")
            squared_off = hhmm >= self.squareoff

            for sym, df in prepared.items():
                if ts not in df.index:
                    continue
                i = df.index.get_loc(ts)
                bar = df.iloc[i]

                # 1. manage first — mirrors scan_once()'s ordering, and means a
                #    position opened at bar i's close is only ever tested
                #    against bar i+1 onward. No look-ahead.
                if sym in open_pos:
                    closed = self._manage(open_pos, sym, bar, ts, res, squared_off, i,
                                          atr_by_sym.get(sym))
                    if closed:
                        day_pnl += closed

                # 2. then look for an entry
                if squared_off or sym in open_pos:
                    continue
                if day_pnl <= self.daily_loss_limit:
                    res.rejections["daily_loss_limit"] = res.rejections.get("daily_loss_limit", 0) + 1
                    continue
                if len(open_pos) >= self.max_open:
                    res.rejections["max_open_positions"] = res.rejections.get("max_open_positions", 0) + 1
                    continue
                self._try_enter(open_pos, sym, df.iloc[: i + 1], ts, res)

        for sym in list(open_pos):
            self._close(open_pos, sym, open_pos[sym].entry, stamps[-1] if stamps else None,
                        "end_of_data", res)
        return res

    def _try_enter(self, open_pos, sym, window, ts, res: Result):
        side = self.strat["detect"](window)
        if side is None:
            return
        price = float(window.iloc[-1]["Close"])
        sl = self.strat["stop"](window, side)
        risk = (price - sl) if side == "BUY" else (sl - price)
        if risk <= 0:
            res.rejections["bad_stop"] = res.rejections.get("bad_stop", 0) + 1
            return
        if "target" in self.strat:
            target = self.strat["target"](window, side, price, sl)
            if target is None:
                res.rejections["reward_too_small"] = res.rejections.get("reward_too_small", 0) + 1
                return
        else:
            rr = self.strat["rr"]
            target = price + rr * risk if side == "BUY" else price - rr * risk
        qty = self.size(price, risk)
        if qty < 1:
            res.rejections["qty_zero"] = res.rejections.get("qty_zero", 0) + 1
            return
        entry = self._fill(price, side, entering=True)
        t = BTTrade(symbol=sym, side=side, qty=qty, entry_ts=ts, entry=entry,
                    sl=round(sl, 2), target=round(target, 2))
        # 1R is anchored to the structural stop, before any trailing moves it.
        t.initial_sl = t.sl
        open_pos[sym] = t
        res.trades.append(t)

    def _manage(self, open_pos, sym, bar, ts, res: Result, squared_off: bool,
                bar_idx: int, atr_series=None) -> float:
        """Advance one open position by one bar.

        Returns the net P&L realised this bar -- a partial booking and a final exit on the
        same bar both count, so 0.0 means "still open, nothing realised"."""
        t = open_pos[sym]
        lo, hi = float(bar["Low"]), float(bar["High"])
        d = 1 if t.side == "BUY" else -1
        t.mae = min(t.mae, d * ((lo if d == 1 else hi) - t.entry))
        t.mfe = max(t.mfe, d * ((hi if d == 1 else lo) - t.entry))

        exit_price, reason = None, ""
        if t.side == "BUY":
            if lo <= t.sl:
                exit_price, reason = t.sl, "stop"      # conservative: stop wins ties
            elif hi >= t.target:
                exit_price, reason = t.target, "target"
        else:
            if hi >= t.sl:
                exit_price, reason = t.sl, "stop"
            elif lo <= t.target:
                exit_price, reason = t.target, "target"
        if exit_price is None and squared_off:
            exit_price, reason = float(bar["Close"]), "squareoff"

        if exit_price is not None:
            return self._close(open_pos, sym, exit_price, ts, reason, res)

        # No exit on this bar -- advance the trail on this bar's CLOSE for the next bar.
        # After the exit tests, so bar i is only ever tested against a stop derived from
        # bars strictly before i. No look-ahead.
        realised = 0.0
        if self.trail_config.enabled and atr_series is not None:
            a = atr_series.iloc[bar_idx] if bar_idx < len(atr_series) else 0.0
            a = 0.0 if a != a else float(a)  # NaN -> no chandelier; breakeven floor still holds
            new_sl, t.trail, partial_qty = update_exit(
                t.entry, t.side, t.sl, float(bar["Close"]), a, t.trail,
                qty=t.qty, risk_per_share=abs(t.entry - (t.initial_sl or t.sl)),
                config=self.trail_config,
            )
            t.sl = new_sl
            if partial_qty > 0:
                realised += self._book_partial(t, partial_qty, float(bar["Close"]), ts, res)
        return realised

    def _book_partial(self, t: BTTrade, partial_qty: int, price: float, ts, res: Result) -> float:
        """Realise part of a position as its own closed trade.

        A separate BTTrade (rather than mutating the parent's average) keeps one entry/exit
        per record, which is what the metrics iterate -- and it prices the extra sell leg's
        charges properly instead of understating them.
        """
        part = BTTrade(symbol=t.symbol, side=t.side, qty=partial_qty,
                       entry_ts=t.entry_ts, entry=t.entry, sl=t.sl, target=t.target,
                       initial_sl=t.initial_sl)
        part.exit = self._fill(price, t.side, entering=False)
        part.exit_ts = ts
        part.reason = "partial"
        d = 1 if t.side == "BUY" else -1
        part.gross = d * (part.exit - part.entry) * part.qty
        part.charges = self.charges_for(part.entry, part.exit, part.qty)
        res.trades.append(part)
        t.qty -= partial_qty
        return part.net

    def _close(self, open_pos, sym, price, ts, reason, res: Result) -> float:
        t = open_pos.pop(sym)
        t.exit = self._fill(price, t.side, entering=False)
        t.exit_ts = ts
        t.reason = reason
        d = 1 if t.side == "BUY" else -1
        t.gross = d * (t.exit - t.entry) * t.qty
        t.charges = self.charges_for(t.entry, t.exit, t.qty)
        return t.net


# --- metrics -------------------------------------------------------------

def metrics(res: Result, *, per_position: bool = False) -> dict:
    ts = res.closed
    n_raw = len(ts)
    if n_raw == 0:
        return {"n": 0, "label": res.label, "pos_n": 0}

    # Aggregate positions by (symbol, entry_ts) to prevent partial bookings
    # from artificially inflating sample size n and narrowing the 95% CI.
    pos_groups: dict[tuple[str, object], list[BTTrade]] = {}
    for t in ts:
        key = (t.symbol, t.entry_ts)
        pos_groups.setdefault(key, []).append(t)

    pos_nets = [sum(t.net for t in group) for group in pos_groups.values()]
    pos_n = len(pos_nets)
    pos_mean = sum(pos_nets) / pos_n if pos_n else 0.0
    pos_var = sum((x - pos_mean) ** 2 for x in pos_nets) / (pos_n - 1) if pos_n > 1 else 0.0
    pos_sd = math.sqrt(pos_var)
    pos_se = pos_sd / math.sqrt(pos_n) if pos_n else 0.0
    pos_ci = 1.96 * pos_se
    pos_wins = [x for x in pos_nets if x > 0]
    pos_win_rate = len(pos_wins) / pos_n if pos_n else 0.0

    if per_position:
        nets = pos_nets
        n = pos_n
    else:
        nets = [t.net for t in ts]
        n = n_raw

    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    total = sum(nets)
    mean = total / n
    var = sum((x - mean) ** 2 for x in nets) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n else 0.0
    ci = 1.96 * se

    equity, peak, dd = 0.0, 0.0, 0.0
    for t in sorted(ts, key=lambda x: (x.exit_ts or x.entry_ts)):
        equity += t.net
        peak = max(peak, equity)
        dd = min(dd, equity - peak)

    gross_profit = sum(t.gross for t in ts if t.gross > 0)
    gross_loss = -sum(t.gross for t in ts if t.gross <= 0)
    charges = sum(t.charges for t in ts)

    return {
        "label": res.label, "n": n,
        "net": total, "gross": sum(t.gross for t in ts), "charges": charges,
        "win_rate": len(wins) / n,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "expectancy": mean, "ci95": ci,
        "lo": mean - ci, "hi": mean + ci,
        "significant": (mean - ci > 0) or (mean + ci < 0),
        "profit_factor": (gross_profit / gross_loss) if gross_loss else float("inf"),
        "max_dd": dd,
        # How much of the winning side is eaten by costs. Above ~1.0 the
        # strategy cannot pay for itself no matter how the entries are tuned.
        "cost_drag": charges / gross_profit if gross_profit else float("inf"),
        "days": len({t.date for t in ts}),
        "trades_per_day": n / max(1, len({t.date for t in ts})),
        # Per-position metrics
        "pos_n": pos_n,
        "pos_expectancy": pos_mean,
        "pos_ci95": pos_ci,
        "pos_lo": pos_mean - pos_ci,
        "pos_hi": pos_mean + pos_ci,
        "pos_win_rate": pos_win_rate,
    }


def _fmt_money(x: float) -> str:
    return f"{x:>10,.2f}"


def report(res: Result, verbose: bool = False, *, per_position: bool = False) -> str:
    m = metrics(res, per_position=per_position)
    if m["n"] == 0:
        return f"{res.label}: no trades over {res.bars} bars · rejections={res.rejections}"

    verdict = ("EDGE (95% CI above zero)" if m["lo"] > 0 else
               "NEGATIVE (95% CI below zero)" if m["hi"] < 0 else
               "NOT PROVEN — CI straddles zero, result is indistinguishable from noise")

    lines = [
        f"=== {res.label} ===",
        f"  window        {m['days']} sessions · {m['n']} trades ({m['trades_per_day']:.1f}/day)",
        f"  net P&L      {_fmt_money(m['net'])}",
        f"  gross        {_fmt_money(m['gross'])}   charges {_fmt_money(m['charges'])}",
        f"  win rate      {m['win_rate']:>9.1%}   avg win {_fmt_money(m['avg_win'])}  avg loss {_fmt_money(m['avg_loss'])}",
        f"  profit factor {m['profit_factor']:>9.2f}   cost drag {m['cost_drag']:>8.2f}  max DD {_fmt_money(m['max_dd'])}",
        f"  expectancy   {_fmt_money(m['expectancy'])} +/- {m['ci95']:.2f} per trade"
        f"  (95% CI {m['lo']:+.2f} .. {m['hi']:+.2f})",
        f"  verdict       {verdict}",
    ]
    if m.get("pos_n") and m["pos_n"] != m["n"]:
        lines.append(
            f"  per-position  {m['pos_n']} positions · expectancy {_fmt_money(m['pos_expectancy'])} +/- {m['pos_ci95']:.2f}"
            f"  (95% CI {m['pos_lo']:+.2f} .. {m['pos_hi']:+.2f})"
        )
    if res.rejections:
        lines.append(f"  skipped       {dict(sorted(res.rejections.items()))}")

    if verbose:
        by_day: dict[str, float] = {}
        for t in res.closed:
            by_day[t.date] = by_day.get(t.date, 0.0) + t.net
        lines.append("  per session:")
        for d in sorted(by_day):
            lines.append(f"    {d}  {by_day[d]:+9.2f}")
        by_sym: dict[str, list[float]] = {}
        for t in res.closed:
            by_sym.setdefault(t.symbol, []).append(t.net)
        lines.append("  per symbol:")
        for s in sorted(by_sym, key=lambda k: -sum(by_sym[k])):
            v = by_sym[s]
            lines.append(f"    {s:<12} n={len(v):>3}  net {sum(v):+9.2f}  win {sum(1 for x in v if x > 0)/len(v):>5.0%}")
        by_reason: dict[str, int] = {}
        for t in res.closed:
            by_reason[t.reason] = by_reason.get(t.reason, 0) + 1
        lines.append(f"  exits         {dict(sorted(by_reason.items()))}")
    return "\n".join(lines)


# --- parameter overrides -------------------------------------------------

@contextlib.contextmanager
def params(**overrides):
    """Temporarily rebind module globals in `trading.strategy`.

    The signal functions read FAST_EMA / SWING_LOOKBACK / etc. as globals at
    call time, so patching them here changes the strategy under test without
    editing config.py or duplicating the signal logic.
    """
    saved = {}
    try:
        for k, v in overrides.items():
            if not hasattr(strat_mod, k):
                raise AttributeError(f"trading.strategy has no parameter {k!r}")
            saved[k] = getattr(strat_mod, k)
            setattr(strat_mod, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(strat_mod, k, v)


def build_strategy(key: str) -> dict:
    """Rebuild a strategy spec so it picks up any patched params."""
    if key == "orb_ma200":
        from trading.backtest_orb import build_orb_strategy
        return build_orb_strategy()
    if key not in strat_mod.STRATEGIES:
        raise SystemExit(f"unknown strategy {key!r} — choose from {list(strat_mod.STRATEGIES)}")
    spec = dict(strat_mod.STRATEGIES[key])
    if key == "ema":
        spec["id"] = f"ema_{strat_mod.FAST_EMA}_{strat_mod.SLOW_EMA}"
        spec["rr"] = strat_mod.RR_TARGET
    return spec


# --- CLI -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m trading.backtest")
    p.add_argument("--strategy", default="ema")
    p.add_argument("--interval", default=CANDLE_INTERVAL)
    p.add_argument("--period", default=None, help="yfinance period, e.g. 60d")
    p.add_argument("--symbols", default=None, help="comma-separated; default SCAN_UNIVERSE")
    p.add_argument("--risk-multiplier", type=float, default=1.0)
    p.add_argument("--slippage", type=float, default=SLIPPAGE_PCT)
    p.add_argument("--charges", type=float, default=None,
                   help=f"flat charge pct instead of the itemised Zerodha model "
                        f"(strategy.py currently uses {CHARGES_PCT_ROUND_TRIP}; 0 = frictionless)")
    p.add_argument("--refresh", action="store_true", help="ignore cache, re-download")
    p.add_argument("--download", choices=("auto", "always", "never"), default="auto",
                   help="auto (default) skips the fetch when the cache is fresh or a "
                        "live session is running; never = cache only")
    p.add_argument("--compare", action="store_true", help="sweep strategies x intervals")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    symbols = a.symbols.split(",") if a.symbols else list(SCAN_UNIVERSE)

    def load(interval):
        frames = load_history(symbols, interval, a.period, refresh=a.refresh,
                              download=a.download)
        if not frames:
            raise SystemExit(f"no candle data for {interval} — check network or run --refresh")
        return frames

    if a.compare:
        rows = []
        for interval in ("5m", "15m"):
            frames = load(interval)
            span = f"{min(df.index[0] for df in frames.values()):%Y-%m-%d}..{max(df.index[-1] for df in frames.values()):%Y-%m-%d}"
            for key in strat_mod.STRATEGIES:
                bt = Backtester(build_strategy(key), risk_multiplier=a.risk_multiplier,
                                slippage=a.slippage, charges=a.charges)
                rows.append((f"{key}@{interval}", metrics(bt.run(frames, f"{key}@{interval}"))))
            print(f"loaded {interval}: {len(frames)} symbols, {span}")
        print()
        hdr = f"{'config':<22}{'n':>5}{'net':>12}{'win%':>7}{'PF':>7}{'exp/trade':>12}{'95% CI':>22}  verdict"
        print(hdr); print("-" * len(hdr))
        for name, m in sorted(rows, key=lambda r: -(r[1].get("expectancy") or -9e9)):
            if not m["n"]:
                print(f"{name:<22}{0:>5}  (no trades)"); continue
            v = "EDGE" if m["lo"] > 0 else "negative" if m["hi"] < 0 else "not proven"
            ci = f"{m['lo']:+.2f} .. {m['hi']:+.2f}"
            print(f"{name:<22}{m['n']:>5}{m['net']:>12,.2f}{m['win_rate']:>7.1%}"
                  f"{m['profit_factor']:>7.2f}{m['expectancy']:>12,.2f}{ci:>22}  {v}")
        return

    frames = load(a.interval)
    span = (f"{min(df.index[0] for df in frames.values()):%Y-%m-%d}"
            f"..{max(df.index[-1] for df in frames.values()):%Y-%m-%d}")
    print(f"{len(frames)} symbols · {a.interval} · {span}\n")
    bt = Backtester(build_strategy(a.strategy), risk_multiplier=a.risk_multiplier,
                    slippage=a.slippage, charges=a.charges)
    res = bt.run(frames, f"{a.strategy}@{a.interval}")
    print(report(res, verbose=a.verbose))


if __name__ == "__main__":
    main(sys.argv[1:])
