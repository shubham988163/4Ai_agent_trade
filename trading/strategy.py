"""Strategy engine — EMA 9/21 crossover on 5-minute candles (delayed data).

Rules:
  Entry  : EMA(9) crosses above EMA(21) on a closed candle -> BUY
           EMA(9) crosses below EMA(21)                     -> SELL (intraday short)
  Stop   : swing low (long) / swing high (short) of the last SWING_LOOKBACK bars
  Target : entry +/- RR_TARGET x risk  (1:2 by default)
  Sizing : RISK_PER_TRADE / per-share risk, capped by the day's position cap
  Exits  : stop or target checked against candle low/high (SL assumed to fill
           first if both hit in one bar — conservative), square-off at 15:15 IST

Data comes from yfinance (delayed ~15 min) — fine for paper validation; swap in
the Kite WebSocket later without touching the rules. Exit checks against candle
extremes are optimistic vs. live ticks; treat paper results accordingly.

Usage:
  python -m trading.strategy               # live polling loop during market hours
  python -m trading.strategy --once        # single scan (cron-able)
  python -m trading.strategy --replay      # replay the last session bar-by-bar
  python -m trading.strategy --replay 2026-07-04
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from trading.config import (
    SCAN_UNIVERSE, YF_SUFFIX, FAST_EMA, SLOW_EMA, CANDLE_INTERVAL, SWING_LOOKBACK,
    RR_TARGET, RISK_PER_TRADE, POLL_SECONDS, SQUAREOFF_TIME, MARKET_OPEN,
    MARKET_CLOSE, MAX_POSITION_VALUE,
    AVWAP_RR, AVWAP_ATR_MULT, AVWAP_RSI_LEN, AVWAP_EMA_LEN, AVWAP_VOL_MULT,
)
from trading.costs import round_trip as round_trip_charges
from trading.exits import TrailState, trail_config_from_settings, update_exit
from trading.execution_router import ExecutionRouter
from trading.ledger import Ledger
from trading.notify import notify

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class Signal:
    symbol: str
    side: str
    price: float
    stop_loss: float
    target: float
    strategy_id: str
    trend_state: str = "unavailable"


class Strategy:
    strategy_id: str = "base"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        raise NotImplementedError

# Consecutive no-data scans before the engine gives up and asks to be restarted.
# At POLL_SECONDS=60 that is five minutes of blindness.
MAX_BLIND_SCANS = 5
# Exit code meaning "transient data failure, please restart me" (EX_TEMPFAIL).
EXIT_DATA_BLACKOUT = 75


def _bar_minutes() -> int:
    return int(CANDLE_INTERVAL.rstrip("m"))


# --- data & indicators ---

def fetch_candles(symbol: str, period: str = "5d") -> pd.DataFrame:
    """Closed intraday candles (CANDLE_INTERVAL) for an NSE symbol, IST-indexed."""
    df = yf.Ticker(symbol + YF_SUFFIX).history(period=period, interval=CANDLE_INTERVAL)
    if df.empty:
        return df
    df = df.tz_convert(IST)
    # Drop the still-forming candle so we only ever act on closed bars.
    now = datetime.now(IST)
    if df.index[-1] + timedelta(minutes=_bar_minutes()) > now:
        df = df.iloc[:-1]
    return df


def fetch_all_candles(symbols: list[str], period: str = "5d") -> dict[str, pd.DataFrame]:
    """Batched candle download for the whole universe — one HTTP call for all
    symbols instead of one per symbol (matters at Nifty-50 scale).

    threads=False is deliberate and load-bearing. With threads=True, yfinance
    opens a peewee sqlite connection to its timezone cache per worker thread
    and never closes it (close_db() only reaches the main thread's). That leaks
    ~2 fds per symbol per call, so a 60s poll loop crosses macOS's 256-fd soft
    limit inside one session and every subsequent fetch dies with
    OperationalError('unable to open database file') — the engine goes blind
    mid-session and square-off then prices positions at entry. Measured
    2026-08-10: +6 fds per 3-symbol call threaded, flat at 4 unthreaded.
    Sequential is comfortably fast enough at a 60s poll.
    """
    data = yf.download(
        tickers=[s + YF_SUFFIX for s in symbols],
        period=period, interval=CANDLE_INTERVAL,
        group_by="ticker", threads=False, progress=False, auto_adjust=True,
    )
    now = datetime.now(IST)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = data[sym + YF_SUFFIX] if len(symbols) > 1 else data
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        df = df.tz_convert(IST)
        if df.index[-1] + timedelta(minutes=_bar_minutes()) > now:
            df = df.iloc[:-1]
        if not df.empty:
            out[sym] = df
    return out


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=FAST_EMA, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=SLOW_EMA, adjust=False).mean()
    return df


def detect_cross(df: pd.DataFrame) -> str | None:
    """BUY/SELL if the fast EMA crossed the slow EMA on the last closed bar."""
    if len(df) < SLOW_EMA + 2:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
        return "BUY"
    if prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
        return "SELL"
    return None


def swing_stop(df: pd.DataFrame, side: str) -> float:
    window = df.iloc[-SWING_LOOKBACK:]
    return float(window["Low"].min()) if side == "BUY" else float(window["High"].max())


# --- indicators & signals: AVWAP scalp (port of tradingview/avwap_scalp.pine) ---

def add_avwap(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-anchored VWAP with ±1σ bands, RSI, EMA trend filter, volume
    surge flag, and ATR — same math as the Pine script."""
    df = df.copy()
    day = df.index.date
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].astype(float)
    cum_vol = vol.groupby(day).cumsum()
    avwap = (tp * vol).groupby(day).cumsum() / cum_vol
    variance = ((tp * tp * vol).groupby(day).cumsum() / cum_vol - avwap**2).clip(lower=0)
    sd = variance**0.5
    df["avwap"] = avwap
    df["sd1_up"] = avwap + sd
    df["sd1_dn"] = avwap - sd
    df["sd2_up"] = avwap + 2 * sd
    df["sd2_dn"] = avwap - 2 * sd

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / AVWAP_RSI_LEN, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / AVWAP_RSI_LEN, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)

    df["ema_trend"] = df["Close"].ewm(span=AVWAP_EMA_LEN, adjust=False).mean()
    df["vol_surge"] = vol > vol.rolling(AVWAP_EMA_LEN).mean() * AVWAP_VOL_MULT

    prev_close = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def detect_avwap(df: pd.DataFrame) -> str | None:
    """AVWAP entry on the last completed bar. Three long setups (AVWAP
    reclaim with volume + trend, lower-band bounce, lower-band cross) and
    their short mirrors; requires the signal to be fresh (none on the
    previous bar), mirroring the Pine `and not signal[1]` gate."""
    if len(df) < AVWAP_EMA_LEN + 3:
        return None
    c, o = df["Close"], df["Open"]
    rsi, hot = df["rsi"], df["vol_surge"]
    up_x = (c > df["avwap"]) & (c.shift() <= df["avwap"].shift())
    dn_x = (c < df["avwap"]) & (c.shift() >= df["avwap"].shift())
    up_x_dn_band = (c > df["sd1_dn"]) & (c.shift() <= df["sd1_dn"].shift())
    dn_x_up_band = (c < df["sd1_up"]) & (c.shift() >= df["sd1_up"].shift())

    buy = ((up_x & (rsi > 45) & hot & (c > df["ema_trend"]))
           | ((df["Low"] <= df["sd1_dn"]) & (c > o) & (c > df["sd1_dn"]) & (rsi > 45))
           | (up_x_dn_band & (rsi > 45) & hot))
    sell = ((dn_x & (rsi < 55) & hot & (c < df["ema_trend"]))
            | ((df["High"] >= df["sd1_up"]) & (c < o) & (c < df["sd1_up"]) & (rsi < 55))
            | (dn_x_up_band & (rsi < 55) & hot))

    if bool(buy.iloc[-1]) and not bool(buy.iloc[-2]):
        return "BUY"
    if bool(sell.iloc[-1]) and not bool(sell.iloc[-2]):
        return "SELL"
    return None


def avwap_stop(df: pd.DataFrame, side: str) -> float:
    bar = df.iloc[-1]
    pad = float(bar["atr"]) * AVWAP_ATR_MULT
    return float(bar["Low"]) - pad if side == "BUY" else float(bar["High"]) + pad


# --- indicators & signals: VWAP 2-sigma mean reversion ---
# The opposite temperament to the EMA/AVWAP chasers that bled in chop: act
# only when price is stretched to the 2-sigma band of the daily anchored VWAP
# with RSI at an extreme AND the candle rejecting the band, then target a
# reversion to VWAP itself. Few signals, full-ATR stop, mean as the target.

def detect_revert(df: pd.DataFrame) -> str | None:
    if len(df) < AVWAP_EMA_LEN + 3:
        return None
    bar = df.iloc[-1]
    c, o = float(bar["Close"]), float(bar["Open"])
    if float(bar["Low"]) <= float(bar["sd2_dn"]) and c > float(bar["sd2_dn"]) \
            and c > o and float(bar["rsi"]) < 35:
        return "BUY"
    if float(bar["High"]) >= float(bar["sd2_up"]) and c < float(bar["sd2_up"]) \
            and c < o and float(bar["rsi"]) > 65:
        return "SELL"
    return None


def revert_stop(df: pd.DataFrame, side: str) -> float:
    bar = df.iloc[-1]
    pad = float(bar["atr"])                      # full ATR: room to breathe
    return float(bar["Low"]) - pad if side == "BUY" else float(bar["High"]) + pad


def revert_target(df: pd.DataFrame, side: str, price: float, sl: float) -> float | None:
    """Target the anchored VWAP; skip the trade when the reversion distance
    doesn't pay at least 1R."""
    target = float(df.iloc[-1]["avwap"])
    risk = (price - sl) if side == "BUY" else (sl - price)
    reward = (target - price) if side == "BUY" else (price - target)
    if risk <= 0 or reward < risk:
        return None
    return target


# --- indicators & signals: 9 EMA + standard pivot points -----------------
# Floor-trader pivots off the *previous* session's H/L/C, used two ways:
#   pivot_ema    - the "9 EMA + pivot point" intraday setup: price reclaims the
#                  9 EMA on the correct side of the daily pivot, stop at the
#                  nearest pivot behind, target the next pivot ahead.
#   ema_pivot    - the existing EMA 9/21 cross with the pivot only as a
#                  directional filter (longs above P, shorts below P), keeping
#                  the swing stop and 1:2 target. Isolates what the filter adds.
# Pivots are computed from intraday bars, so H/L/C span market hours only —
# a few paise off an official daily bar, immaterial at these level widths.

PIVOT_LEVELS_UP = ["pp", "r1", "r2", "r3"]
PIVOT_LEVELS_DN = ["pp", "s1", "s2", "s3"]


def add_pivots(df: pd.DataFrame) -> pd.DataFrame:
    df = add_emas(df)
    day = pd.Series(df.index.date, index=df.index)
    daily = df.groupby(day).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    prev = daily.shift(1)                       # yesterday's range — never today's
    pp = (prev["H"] + prev["L"] + prev["C"]) / 3
    rng = prev["H"] - prev["L"]
    levels = {
        "pp": pp,
        "r1": 2 * pp - prev["L"], "s1": 2 * pp - prev["H"],
        "r2": pp + rng,           "s2": pp - rng,
        "r3": prev["H"] + 2 * (pp - prev["L"]),
        "s3": prev["L"] - 2 * (prev["H"] - pp),
    }
    for name, series in levels.items():
        df[name] = day.map(series)

    prev_close = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def _pivot_bias(bar: pd.Series) -> int:
    """+1 above the daily pivot, -1 below, 0 when pivots aren't available yet
    (first session of the frame)."""
    pp = bar.get("pp")
    if pp is None or pd.isna(pp):
        return 0
    return 1 if float(bar["Close"]) > float(pp) else -1


def detect_pivot_ema(df: pd.DataFrame) -> str | None:
    """Close reclaims/loses the 9 EMA, taken only in the pivot's direction."""
    if len(df) < FAST_EMA + 2:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    if pd.isna(last.get("pp")):
        return None
    up = prev["Close"] <= prev["ema_fast"] and last["Close"] > last["ema_fast"]
    dn = prev["Close"] >= prev["ema_fast"] and last["Close"] < last["ema_fast"]
    bias = _pivot_bias(last)
    if up and bias > 0:
        return "BUY"
    if dn and bias < 0:
        return "SELL"
    return None


def _nearest_level(bar: pd.Series, names: list[str], price: float, below: bool) -> float | None:
    vals = [float(bar[n]) for n in names if not pd.isna(bar.get(n))]
    side = [v for v in vals if (v < price if below else v > price)]
    if not side:
        return None
    return max(side) if below else min(side)


def pivot_stop(df: pd.DataFrame, side: str) -> float:
    """Nearest pivot level behind the entry, padded by 0.25 ATR so a level
    tag doesn't stop us out. Falls back to a 1-ATR stop past the level set."""
    bar = df.iloc[-1]
    price = float(bar["Close"])
    pad = float(bar["atr"]) * 0.25
    if side == "BUY":
        lvl = _nearest_level(bar, PIVOT_LEVELS_DN, price, below=True)
        return (lvl - pad) if lvl is not None else price - float(bar["atr"])
    lvl = _nearest_level(bar, PIVOT_LEVELS_UP, price, below=False)
    return (lvl + pad) if lvl is not None else price + float(bar["atr"])


def pivot_target(df: pd.DataFrame, side: str, price: float, sl: float) -> float | None:
    """Next pivot level ahead. Skipped when it doesn't pay at least 1R —
    the level structure, not a fixed RR, decides whether the trade is worth it."""
    bar = df.iloc[-1]
    lvl = _nearest_level(bar, PIVOT_LEVELS_UP if side == "BUY" else PIVOT_LEVELS_DN,
                         price, below=(side != "BUY"))
    if lvl is None:
        return None
    risk = (price - sl) if side == "BUY" else (sl - price)
    reward = (lvl - price) if side == "BUY" else (price - lvl)
    if risk <= 0 or reward < risk:
        return None
    return lvl


def detect_ema_pivot(df: pd.DataFrame) -> str | None:
    """The existing EMA 9/21 cross, gated on the pivot bias."""
    side = detect_cross(df)
    if side is None:
        return None
    bias = _pivot_bias(df.iloc[-1])
    if (side == "BUY" and bias > 0) or (side == "SELL" and bias < 0):
        return side
    return None


# Pluggable strategies: prepare(df) adds indicators, detect(df) yields a side,
# stop(df, side) places the SL, rr sets the target, bias(bar) feeds the scan
# summary line. The risk kernel in ExecutionRouter applies identically to all.
STRATEGIES = {
    "ema": {
        "id": f"ema_{FAST_EMA}_{SLOW_EMA}_{CANDLE_INTERVAL}",
        "prepare": add_emas, "detect": detect_cross,
        "stop": swing_stop, "rr": RR_TARGET,
        "bias": lambda bar: bar["ema_fast"] > bar["ema_slow"],
    },
    "avwap": {
        "id": "avwap_scalp_5m",
        "prepare": add_avwap, "detect": detect_avwap,
        "stop": avwap_stop, "rr": AVWAP_RR,
        "bias": lambda bar: bar["Close"] > bar["avwap"],
    },
    "revert": {
        "id": "vwap_revert_5m",
        "prepare": add_avwap, "detect": detect_revert,
        "stop": revert_stop, "target": revert_target,
        "bias": lambda bar: bar["Close"] > bar["avwap"],
    },
    # The Instagram reel's setup: 9 EMA reclaim + standard pivot points.
    "pivot": {
        "id": f"pivot_ema{FAST_EMA}",
        "prepare": add_pivots, "detect": detect_pivot_ema,
        "stop": pivot_stop, "target": pivot_target,
        "bias": lambda bar: _pivot_bias(bar) > 0,
    },
    # Minimal change to the live system: existing EMA 9/21 + pivot filter.
    "ema_pivot": {
        "id": f"ema_{FAST_EMA}_{SLOW_EMA}_pivotfilter",
        "prepare": add_pivots, "detect": detect_ema_pivot,
        "stop": swing_stop, "rr": RR_TARGET,
        "bias": lambda bar: bar["ema_fast"] > bar["ema_slow"],
    },
    # ORB + Multi-TF 200 MA trend-filter strategy
    "orb_ma200": {
        "id": "orb_ma200",
        "instance": None,
        "prepare": lambda df: df,
        "bias": lambda bar: True,
    },
}


# --- engine ---

def market_open_now() -> bool:
    """True only during a live NSE session (Mon–Fri, open → square-off, IST)."""
    now = datetime.now(IST)
    return now.weekday() < 5 and MARKET_OPEN <= now.strftime("%H:%M") < SQUAREOFF_TIME


class Engine:
    def __init__(self, router: ExecutionRouter, ledger: Ledger, strategy: str = "ema",
                 trail_config=None):
        self.strat = STRATEGIES[strategy]
        if "instance" in self.strat and self.strat["instance"] is None:
            from trading.strategies.orb_ma200 import ORBMA200Strategy
            self.strat["instance"] = ORBMA200Strategy()
        self.router = router
        self.ledger = ledger
        # Trailing-stop toggle. Defaults to the system setting (off), so backtest and live
        # run one implementation from one flag.
        self.trail_config = trail_config or trail_config_from_settings()
        self.prices: dict[str, float] = {}
        router.get_ltp = lambda sym: self.prices[sym]
        # open positions: symbol -> state (re-synced from ledger on start)
        self.open: dict[str, dict] = {}
        for t in ledger.open_trades():
            self.open[t["symbol"]] = {
                "trade_id": t["id"], "side": t["side"], "qty": t["qty"],
                "entry": t["entry_price"], "sl": t["stop_loss"],
                # initial_stop is 1R's anchor; rows predating that column fall back to
                # the stop as stored, which is the pre-trailing behaviour.
                "initial_sl": t["initial_stop"] or t["stop_loss"],
                "target": t["target"], "mae": 0.0, "mfe": 0.0,
            }
        self.last_bar: dict[str, object] = {}
        self.blind_scans = 0        # consecutive scans that returned no data

    def _bar_atr(self, df: pd.DataFrame) -> float:
        """ATR(14) at the latest bar, for the chandelier trail. 0.0 when unavailable --
        the breakeven floor still holds, there is simply no trail leg."""
        if not self.trail_config.enabled or df.empty:
            return 0.0
        if not {"High", "Low", "Close"} <= set(df.columns):
            return 0.0
        from trading.strategies.orb_ma200 import _atr
        series = _atr(df, 14)
        if series.empty:
            return 0.0
        val = series.iloc[-1]
        return 0.0 if val != val else float(val)  # NaN -> no trail leg

    def size(self, price: float, risk_per_share: float) -> int:
        if risk_per_share <= 0 or price <= 0:
            return 0
        # Cautious days shrink the rupees we risk, not the buying power —
        # a scaled position cap priced the whole watchlist out for a small
        # account. MAX_POSITION_VALUE itself stays a hard, unscaled ceiling.
        risk_inr = RISK_PER_TRADE * self.router.day_config["risk_multiplier"]
        qty = int(risk_inr / risk_per_share)
        return min(qty, int(MAX_POSITION_VALUE / price))

    def try_enter(self, symbol: str, df: pd.DataFrame, ts: float):
        if symbol in self.open:
            return
        if "instance" in self.strat and self.strat["instance"] is not None:
            sig = self.strat["instance"].evaluate(df, symbol)
            if sig is None:
                return
            side = sig.side
            price = sig.price
            sl = sig.stop_loss
            target = sig.target
            risk = (price - sl) if side == "BUY" else (sl - price)
        else:
            side = self.strat["detect"](df)
            if side is None:
                return
            price = float(df.iloc[-1]["Close"])
            sl = self.strat["stop"](df, side)
            risk = (price - sl) if side == "BUY" else (sl - price)
            if risk <= 0:
                return
            if "target" in self.strat:
                target = self.strat["target"](df, side, price, sl)
                if target is None:
                    return              # reversion doesn't pay enough — skip
            else:
                rr = self.strat["rr"]
                target = price + rr * risk if side == "BUY" else price - rr * risk
        if risk <= 0:
            return
        qty = self.size(price, risk)
        if qty < 1:
            return
        self.prices[symbol] = price
        trend_state = getattr(sig, "trend_state", "up" if side == "BUY" else "down") if "instance" in self.strat and self.strat["instance"] is not None else ("up" if side == "BUY" else "down")
        signal = {"symbol": symbol, "side": side, "qty": qty, "price": price,
                  "ts": ts, "stop_loss": round(sl, 2), "initial_stop": round(sl, 2),
                  "target": round(target, 2),
                  "strategy_id": self.strat["id"],
                  "regime": self.router.day_config.get("regime"),
                  "trend_state": trend_state}
        trade_id = self.router.execute(signal)
        if trade_id:
            self.open[symbol] = {"trade_id": trade_id, "side": side, "qty": qty,
                                 "entry": price, "sl": sl, "initial_sl": sl,
                                 "target": target, "mae": 0.0, "mfe": 0.0}
            print(f"ENTER {side} {symbol} x{qty} @ ~{price:.2f} sl={sl:.2f} tgt={target:.2f}")
            notify(f"📈 ENTRY {side} {symbol}",
                   f"×{qty} @ ~{price:.2f} · SL {sl:.2f} · TGT {target:.2f} (python engine)")
        else:
            reason = getattr(self.router, "last_rejection", None) or "risk kernel"
            print(f"rejected {side} {symbol} x{qty} ({reason})")
            # Routine cap/rate-limit rejections are expected many times per day
            # with a 49-symbol scan — log them but don't spam notifications.
            if not reason.startswith(("max_open_positions", "rate_limit")):
                notify(f"🚫 REJECTED {side} {symbol}", f"×{qty} @ {price:.2f} — {reason}")

    def manage_open(self, symbol: str, bar: pd.Series, atr: float | None = None):
        """Advance one open position by the latest closed bar.

        The trailing stop is applied BEFORE the exit tests using the previous bars' data
        only, so a bar is never tested against a stop derived from its own close. This is
        the same ordering as backtest.py's _manage() and the research harness.
        """
        pos = self.open.get(symbol)
        if pos is None:
            return

        trail_config = self.trail_config
        if trail_config.enabled:
            risk_per_share = abs(pos["entry"] - pos.get("initial_sl", pos["sl"]))
            pos.setdefault("trail", TrailState())
            new_sl, pos["trail"], partial_qty = update_exit(
                pos["entry"], pos["side"], pos["sl"], float(bar["Close"]), float(atr or 0.0),
                pos["trail"], qty=pos["qty"], risk_per_share=risk_per_share,
                config=trail_config,
            )
            if new_sl != pos["sl"]:
                pos["sl"] = new_sl
                # Mirror the move into the open ledger row so a restart, the supervisor
                # agent, and the dashboard all see the stop the engine is actually using.
                self.ledger.update_open_stop(pos["trade_id"], new_sl)
            if partial_qty > 0:
                self._book_partial(symbol, partial_qty, float(bar["Close"]))

        pos = self.open.get(symbol)
        if pos is None:
            return  # a partial cannot close the whole position, but stay defensive
        lo, hi = float(bar["Low"]), float(bar["High"])
        direction = 1 if pos["side"] == "BUY" else -1
        pos["mae"] = min(pos["mae"], direction * ((lo if direction == 1 else hi) - pos["entry"]))
        pos["mfe"] = max(pos["mfe"], direction * ((hi if direction == 1 else lo) - pos["entry"]))

        exit_price = None
        if pos["side"] == "BUY":
            if lo <= pos["sl"]:
                exit_price = pos["sl"]          # conservative: stop fills first
            elif hi >= pos["target"]:
                exit_price = pos["target"]
        else:
            if hi >= pos["sl"]:
                exit_price = pos["sl"]
            elif lo <= pos["target"]:
                exit_price = pos["target"]
        if exit_price is not None:
            self._close(symbol, exit_price)

    def _book_partial(self, symbol: str, partial_qty: int, price: float):
        """Realise part of an open position at market, leaving the rest trailing.

        Recorded as its own trade row (entry at the parent's entry, exit at the booking
        price) so each row stays a complete, readable record and the extra sell leg's
        charges are priced properly. The parent row's qty is reduced to match, otherwise
        its final exit would count the booked shares a second time.
        """
        pos = self.open.get(symbol)
        if pos is None:
            return
        partial_qty = min(partial_qty, pos["qty"] - 1)
        if partial_qty < 1:
            return
        child_signal = {
            "symbol": symbol, "side": pos["side"], "qty": partial_qty,
            "ts": time.time(), "stop_loss": pos["sl"], "target": pos["target"],
            "strategy_id": self.strat["id"], "regime": self.router.day_config.get("regime"),
        }
        child_id = self.ledger.record_entry(child_signal, pos["entry"])
        charges = round_trip_charges(pos["entry"], price, partial_qty)
        pnl = self.ledger.record_exit(child_id, round(price, 2), charges=round(charges, 2))
        self.ledger.reduce_open_qty(pos["trade_id"], pos["qty"] - partial_qty)
        pos["qty"] -= partial_qty
        print(f"PARTIAL {pos['side']} {symbol} x{partial_qty} @ {price:.2f} -> pnl {pnl:.2f} "
              f"({pos['qty']} left trailing, sl {pos['sl']:.2f})")
        notify(f"💰 PARTIAL {pos['side']} {symbol}",
               f"×{partial_qty} @ {price:.2f} → P&L {pnl:+.2f} INR · {pos['qty']} left, SL {pos['sl']:.2f}")

    def _close(self, symbol: str, exit_price: float):
        pos = self.open.pop(symbol)
        # Itemised Zerodha charges, not a flat percentage: brokerage is capped
        # at Rs 20/order, so the flat model overstated costs by ~2x once
        # position sizes reached Rs 2L (and by ~13% at the old Rs 25k sizes).
        charges = round_trip_charges(pos["entry"], exit_price, pos["qty"])
        pnl = self.ledger.record_exit(pos["trade_id"], round(exit_price, 2),
                                      charges=round(charges, 2),
                                      mae=round(pos["mae"], 2), mfe=round(pos["mfe"], 2))
        print(f"EXIT  {pos['side']} {symbol} @ {exit_price:.2f} -> pnl {pnl:.2f}")
        notify(f"{'✅' if pnl >= 0 else '🔻'} EXIT {pos['side']} {symbol}",
               f"@ {exit_price:.2f} → P&L {pnl:+.2f} INR · day {self.ledger.day_realized_pnl():+.2f}")

    def squareoff(self, note: str = "squareoff"):
        for symbol in list(self.open):
            price = self.prices.get(symbol, self.open[symbol]["entry"])
            print(f"{note}: closing {symbol}")
            self._close(symbol, price)

    # --- modes ---

    def scan_once(self):
        try:
            frames = fetch_all_candles(SCAN_UNIVERSE)
        except Exception as e:  # noqa: BLE001
            print(f"batch data error {e!r}")
            self.blind_scans += 1
            return
        # A data blackout does NOT raise — yfinance returns an empty/partial dict
        # per symbol. On 2026-08-10 the engine scanned 0/6 for ~40 minutes and
        # kept reporting success, so a crash-only watchdog would never fire.
        # Count consecutive shutouts; live() escalates.
        if not frames:
            self.blind_scans += 1
            print(f"data blackout — 0/{len(SCAN_UNIVERSE)} symbols "
                  f"({self.blind_scans} consecutive)")
            return
        self.blind_scans = 0
        # Outside a live session the latest candle is Friday's/yesterday's close;
        # it may hold an unprocessed crossover that would fill at a stale price
        # and sit open all weekend. Manage exits, but take no new entries.
        entries_ok = market_open_now()
        if not entries_ok:
            print("market closed — managing open positions only, no new entries")
        bull = bear = 0
        for symbol, df in frames.items():
            df = self.strat["prepare"](df)
            bar = df.iloc[-1]
            self.prices[symbol] = float(bar["Close"])
            if self.last_bar.get(symbol) == df.index[-1]:
                continue  # already acted on this bar
            self.last_bar[symbol] = df.index[-1]
            self.manage_open(symbol, bar, atr=self._bar_atr(df))
            if entries_ok:
                self.try_enter(symbol, df, ts=df.index[-1].timestamp())
            if self.strat["bias"](bar):
                bull += 1
            else:
                bear += 1
        open_desc = ", ".join(f"{s}({p['side']})" for s, p in self.open.items()) or "none"
        print(f"scanned {len(frames)}/{len(SCAN_UNIVERSE)} symbols · "
              f"{bull} bull / {bear} bear · open: {open_desc}")

    def live(self):
        print(f"live loop: {CANDLE_INTERVAL} candles, poll {POLL_SECONDS}s, "
              f"squareoff {SQUAREOFF_TIME} IST")
        while True:
            now = datetime.now(IST)
            hhmm = now.strftime("%H:%M")
            if hhmm >= SQUAREOFF_TIME:
                if self.blind_scans and self.open:
                    # No fresh prices means squareoff() would close at the entry
                    # price and book a fake ~0 P&L. Say so loudly in the log.
                    print(f"WARNING: squaring off during a data blackout "
                          f"({self.blind_scans} blind scans) — exit prices are stale")
                self.squareoff("EOD")
                if hhmm >= MARKET_CLOSE:
                    print("market closed — exiting")
                    return
            elif hhmm >= MARKET_OPEN:
                self.scan_once()
                if self.blind_scans >= MAX_BLIND_SCANS:
                    # Exit non-zero so cron_day.sh restarts us with a fresh
                    # process. The 2026-08-10 failure was unrecoverable in-process:
                    # yfinance had already exhausted the fd limit.
                    print(f"FATAL: {self.blind_scans} consecutive blind scans — "
                          f"exiting {EXIT_DATA_BLACKOUT} for supervisor restart")
                    sys.exit(EXIT_DATA_BLACKOUT)
            else:
                print(f"{hhmm} IST — waiting for open")
            time.sleep(POLL_SECONDS)

    def replay(self, date: str | None = None, period: str = "5d"):
        """Replay one session bar-by-bar through the same rules — a dry run on
        real market data that writes real (paper) trades to the ledger.
        Bars from earlier days in the fetched period serve as indicator
        warm-up (like the live engine's history); only the target date
        trades."""
        raw = fetch_all_candles(SCAN_UNIVERSE, period=period)
        if not raw:
            print("no candle data to replay")
            return
        target = date or max(df.index[-1].strftime("%Y-%m-%d") for df in raw.values())
        stamps = sorted({ts for df in raw.values() for ts in df.index
                         if ts.strftime("%Y-%m-%d") == target})
        if not stamps:
            print(f"no bars for {target}")
            return
        print(f"replaying {target} for {', '.join(raw)}")
        for ts in stamps:
            for symbol, df in raw.items():
                if ts not in df.index:
                    continue
                idx = df.index.get_loc(ts)
                if idx < SLOW_EMA + 2:
                    continue
                window = self.strat["prepare"](df.iloc[: idx + 1])
                bar = window.iloc[-1]
                self.prices[symbol] = float(bar["Close"])
                self.manage_open(symbol, bar, atr=self._bar_atr(window))
                self.try_enter(symbol, window, ts=ts.timestamp())
        self.squareoff("replay EOD")
        print(f"\nreplay done — day P&L: {self.ledger.day_realized_pnl():.2f} INR "
              f"({len(self.ledger.trades_for_date())} trades today in ledger)")


def raise_fd_limit(target: int = 8192):
    """Lift the file-descriptor soft limit (macOS defaults to 256).

    Belt and braces behind the threads=False fix in fetch_all_candles: if any
    dependency leaks fds again, a session should degrade slowly rather than go
    blind two hours before square-off. The hard limit is unlimited, so raising
    the soft limit needs no privileges.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = min(target, hard) if hard != resource.RLIM_INFINITY else target
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            print(f"fd soft limit raised {soft} -> {want}")
    except Exception as e:  # noqa: BLE001 - never block a session over this
        print(f"could not raise fd limit ({e!r}) — continuing")


def main():
    raise_fd_limit()
    ledger = Ledger()
    router = ExecutionRouter(mode="paper", ledger=ledger)
    args = sys.argv[1:]
    # Default = EMA on the core watchlist: the exact Jul-7 setup, the only
    # configuration that has been net-profitable. AVWAP looked good in the
    # ledger (+222 over 6 TV trades) but systematic replays lose every day —
    # run it with `--strategy avwap` if you want to keep testing it.
    strategy = args[args.index("--strategy") + 1] if "--strategy" in args else "ema"
    if strategy not in STRATEGIES:
        print(f"unknown strategy {strategy!r} — choose from {list(STRATEGIES)}")
        return
    engine = Engine(router, ledger, strategy=strategy)
    print(f"strategy: {engine.strat['id']}")
    print(f"day config: {router.day_config}")
    if "--replay" in args:
        date = next((a for a in args if a[0].isdigit()), None)
        engine.replay(date)
    elif "--once" in args:
        engine.scan_once()
    else:
        engine.live()


if __name__ == "__main__":
    main()
