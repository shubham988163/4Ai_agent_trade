"""ORB + Multi-TF 200MA trend-filter strategy  (v2 - upgraded).

Rules
-----
1. Opening range = 09:15-09:30 high/low. Skip the day if the range is
   < min_or_pct or > max_or_pct of price.
2. Trigger = a bar CLOSES beyond the OR, previous bar had not (fresh breakout).
3. Trend gate: price above 200 MA = uptrend, below = downtrend, on BOTH 1H and
   30min. Longs only in uptrend, shorts only in downtrend.
4. Confirmation:
   a. Volume: breakout-bar volume >= vol_mult x baseline. Baseline = average
      volume of the SAME time-of-day bar over the last tod_lookback sessions
      (falls back to prior-20-bar average if fewer than tod_min_sessions).
   b. Price on right side of a sloping session VWAP.
   c. Breakout-bar quality: close near the extreme of the bar and a real body.
   d. Optional index filter: Nifty (or any index_df you pass) on the same side
      of its own session VWAP.
5. Structural stop = breakout bar extreme +/- atr_buffer x ATR. Target = rr x
   risk. Skip if stop wider than max_risk_atr x ATR.
6. No entries after last_entry (default 11:30 - ORB edge fades later).
   One long + one short per symbol per day.

NEW exit helper (engine must call it every closed bar while in a trade):
   should_exit(...) -> "failed_breakout" | "time_stop" | None

NEW taken-side handling:
   Set auto_mark_taken=False and call confirm_taken(symbol, side, ts) from the
   engine when the order is actually FILLED. Then a router rejection no longer
   burns the side for the day.

Position sizing is NOT done here (engine: RISK_PER_TRADE; router: limits).
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from trading.strategy import Strategy, Signal

IST = ZoneInfo("Asia/Kolkata")


# --------------------------------------------------------------------------
# module helpers
# --------------------------------------------------------------------------
def drop_unclosed_bars(
    df: pd.DataFrame,
    interval_minutes: int = 5,
    now: datetime | None = None,
    grace_seconds: int = 30,
) -> pd.DataFrame:
    """Drop any bar not yet closed (bar end time + grace > now).

    A 5-minute bar stamped 14:35 covers 14:35:00 to 14:40:00. With
    grace_seconds=30 it counts as closed only when now >= 14:40:30.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if now is None:
        now = datetime.now(IST)
    elif getattr(now, "tzinfo", None) is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)

    bar_end = idx + pd.Timedelta(minutes=interval_minutes) + pd.Timedelta(seconds=grace_seconds)
    mask = (bar_end <= now)
    return df.iloc[mask]


def _ist_idx(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    return idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)


def _ist_ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift()
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _session_vwap(day: pd.DataFrame) -> pd.Series:
    typical = (day["High"] + day["Low"] + day["Close"]) / 3
    vol_sum = day["Volume"].cumsum()
    return (typical * day["Volume"]).cumsum() / vol_sum.where(vol_sum > 0, 1.0)


def _empty_setup(symbol: str) -> dict:
    return {
        "signal": None,
        "symbol": symbol,
        "price": 0.0,
        "ts": None,
        "orb": {"or_high": None, "or_low": None, "or_width_pct": None, "or_valid": False},
        "trend": {"price": 0.0, "sma200_1h": None, "sma200_30m": None, "state": "unavailable"},
        "volume": {"breakout_bar_vol": 0.0, "avg20": 0.0, "ratio": 0.0,
                   "bar_complete": False, "baseline": 0.0, "source": "none"},
        "vwap": {"value": 0.0, "slope": "flat", "aligned": False},
        "quality": {"close_pos": 0.0, "body_ratio": 0.0, "ok": False},
        "index_bias": None,
        "atr": 0.0,
        "buy_stop": 0.0,
        "buy_target": 0.0,
        "sell_stop": 0.0,
        "sell_target": 0.0,
    }


class ORBMA200Strategy(Strategy):
    strategy_id = "orb_ma200"

    def __init__(
        self,
        or_minutes: int = 15,
        min_or_pct: float = 0.3,
        max_or_pct: float = 1.5,
        ma_len: int = 200,
        use_1h: bool = True,
        use_30m: bool = True,
        vol_mult: float = 1.5,
        use_vwap: bool = True,
        atr_len: int = 14,
        atr_buffer: float = 0.25,
        max_risk_atr: float = 1.5,
        rr: float = 2.0,
        max_cost_risk_ratio: float = 0.30,
        last_entry: str = "11:30",
        one_per_side: bool = True,
        # ---- v2 additions ----
        use_tod_volume: bool = True,
        tod_lookback: int = 10,
        tod_min_sessions: int = 5,
        use_quality: bool = True,
        min_close_pos: float = 0.70,
        min_body_ratio: float = 0.50,
        use_index_filter: bool = False,
        use_fail_exit: bool = True,
        fail_exit_buffer_atr: float = 0.0,
        time_stop_bars: int = 12,
        time_stop_min_r: float = 1.0,
        auto_mark_taken: bool = True,
    ):
        self.or_minutes = or_minutes
        self.min_or_pct = min_or_pct
        self.max_or_pct = max_or_pct
        self.ma_len = ma_len
        self.use_1h = use_1h
        self.use_30m = use_30m
        self.vol_mult = vol_mult
        self.use_vwap = use_vwap
        self.atr_len = atr_len
        self.atr_buffer = atr_buffer
        self.max_risk_atr = max_risk_atr
        self.rr = rr
        self.max_cost_risk_ratio = max_cost_risk_ratio
        self.last_entry = last_entry
        self.one_per_side = one_per_side

        self.use_tod_volume = use_tod_volume
        self.tod_lookback = tod_lookback
        self.tod_min_sessions = tod_min_sessions
        self.use_quality = use_quality
        self.min_close_pos = min_close_pos
        self.min_body_ratio = min_body_ratio
        self.use_index_filter = use_index_filter
        self.use_fail_exit = use_fail_exit
        self.fail_exit_buffer_atr = fail_exit_buffer_atr
        self.time_stop_bars = time_stop_bars
        self.time_stop_min_r = time_stop_min_r
        self.auto_mark_taken = auto_mark_taken

        self._taken: dict[tuple[str, object], set[str]] = {}

    # ---- helpers -------------------------------------------------------

    def _htf_ma(self, df: pd.DataFrame, rule: str) -> float | None:
        """200-period SMA of close on a higher timeframe, resampled from df.
        offset=15min aligns bins to the 09:15 NSE open."""
        bars = (
            df.resample(rule, offset="15min")
            .agg({"Open": "first", "High": "max", "Low": "min",
                  "Close": "last", "Volume": "sum"})
            .dropna()
        )
        if len(bars) < self.ma_len:
            return None
        return float(bars["Close"].rolling(self.ma_len).mean().iloc[-1])

    def _trend(self, df: pd.DataFrame, price: float) -> str:
        mas = []
        if self.use_1h:
            mas.append(self._htf_ma(df, "60min"))
        if self.use_30m:
            mas.append(self._htf_ma(df, "30min"))
        if not mas or any(m is None for m in mas):
            return "unavailable"
        if all(price > m for m in mas):
            return "up"
        if all(price < m for m in mas):
            return "down"
        return "mixed"

    def _or_levels(self, day: pd.DataFrame):
        """(or_high, or_low, or_end) for a single-session frame, or None."""
        if day.empty:
            return None
        or_end = day.index[0].normalize() + pd.Timedelta(
            hours=9, minutes=15 + self.or_minutes
        )
        or_bars = day[day.index < or_end]
        if or_bars.empty:
            return None
        return (
            round(float(or_bars["High"].max()), 2),
            round(float(or_bars["Low"].min()), 2),
            or_end,
        )

    def _tod_avg_volume(self, df: pd.DataFrame, ts) -> float | None:
        """Mean volume of the same time-of-day bar over prior sessions."""
        prior = df[(df.index.date < ts.date()) & (df.index.time == ts.time())]
        vols = prior["Volume"].iloc[-self.tod_lookback:]
        if len(vols) < self.tod_min_sessions:
            return None
        m = float(vols.mean())
        return m if m > 0 else None

    def _index_bias(self, index_df: pd.DataFrame | None, ts) -> str | None:
        """'up' if index closes above its session VWAP, 'down' below, else None."""
        if index_df is None or index_df.empty:
            return None
        idx = _ist_idx(index_df)
        t = _ist_ts(ts)
        mask = (idx.date == t.date()) & (idx <= t)
        iday = index_df[mask]
        if len(iday) < 3:
            return None
        vwap = float(_session_vwap(iday).iloc[-1])
        close = float(iday["Close"].iloc[-1])
        if close > vwap:
            return "up"
        if close < vwap:
            return "down"
        return None

    def _prune_taken(self, today) -> None:
        for k in [k for k in self._taken if k[1] < today]:
            del self._taken[k]

    def confirm_taken(self, symbol: str, side: str, ts) -> None:
        """Call from the engine when an order is FILLED (auto_mark_taken=False)."""
        d = pd.Timestamp(ts).date()
        self._prune_taken(d)
        self._taken.setdefault((symbol, d), set()).add(side)

    # ---- setup ---------------------------------------------------------

    def compute_setup(
        self,
        df: pd.DataFrame,
        symbol: str,
        now: datetime | None = None,
        index_df: pd.DataFrame | None = None,
    ) -> dict:
        """Compute all indicators and trigger checks in one pass.

        Unclosed bars are ALWAYS dropped (now=None -> wall clock). Historical
        bars are all closed, so this is harmless in backtests.
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return _empty_setup(symbol)

        df = drop_unclosed_bars(df, interval_minutes=5, now=now)
        if len(df) == 0:
            return _empty_setup(symbol)

        ts = df.index[-1]
        price = round(float(df["Close"].iloc[-1]), 2)

        # 1. Multi-TF 200 SMA trend
        sma200_1h = self._htf_ma(df, "60min") if self.use_1h else None
        sma200_30m = self._htf_ma(df, "30min") if self.use_30m else None
        trend_state = self._trend(df, price)

        # 2. Opening range
        day = df[df.index.date == ts.date()]
        or_high, or_low, or_pct = None, None, None
        or_valid = False
        if len(day) >= 3:
            lv = self._or_levels(day)
            if lv is not None and ts >= lv[2]:
                or_high, or_low = lv[0], lv[1]
                if or_low > 0:
                    or_pct = round((or_high - or_low) / or_low * 100, 2)
                    or_valid = (self.min_or_pct <= or_pct <= self.max_or_pct)

        # 3. Volume (time-of-day baseline, fallback prior-20-bar)
        current_vol = round(float(df["Volume"].iloc[-1]), 2)
        if len(df) >= 21:
            avg20 = round(float(df["Volume"].iloc[-21:-1].mean()), 2)
        elif len(df) > 1:
            avg20 = round(float(df["Volume"].iloc[:-1].mean()), 2)
        else:
            avg20 = current_vol

        tod_avg = self._tod_avg_volume(df, ts) if self.use_tod_volume else None
        if tod_avg is not None:
            vol_base, vol_src = round(tod_avg, 2), "tod"
        else:
            vol_base, vol_src = avg20, "prev20"
        vol_ratio = round(current_vol / vol_base, 2) if vol_base > 0 else 1.0

        # 4. VWAP side + slope
        if len(day) > 0:
            vwap_series = _session_vwap(day)
            vwap_val = round(float(vwap_series.iloc[-1]), 2)
            slope_ref = vwap_series.iloc[-4] if len(vwap_series) >= 4 else vwap_series.iloc[0]
            if vwap_val > slope_ref:
                vwap_slope = "up"
            elif vwap_val < slope_ref:
                vwap_slope = "down"
            else:
                vwap_slope = "flat"
        else:
            vwap_val = price
            vwap_slope = "flat"

        # 5. ATR & structural stops
        atr_series = _atr(df, self.atr_len)
        atr_raw = atr_series.iloc[-1] if not atr_series.empty else None
        if atr_raw is not None and not pd.isna(atr_raw) and atr_raw > 0:
            atr = round(float(atr_raw), 2)
        else:
            atr = round(price * 0.005, 2)

        last_bar = day.iloc[-1] if len(day) > 0 else df.iloc[-1]
        buy_stop = round(float(last_bar["Low"]) - self.atr_buffer * atr, 2)
        buy_risk = round(price - buy_stop, 2)
        buy_target = round(price + self.rr * buy_risk, 2)

        sell_stop = round(float(last_bar["High"]) + self.atr_buffer * atr, 2)
        sell_risk = round(sell_stop - price, 2)
        sell_target = round(price - self.rr * sell_risk, 2)

        # 6. Bar quality
        bar_rng = float(last_bar["High"]) - float(last_bar["Low"])
        if bar_rng > 0:
            close_pos = round((float(last_bar["Close"]) - float(last_bar["Low"])) / bar_rng, 2)
            body_ratio = round(abs(float(last_bar["Close"]) - float(last_bar["Open"])) / bar_rng, 2)
        else:
            close_pos, body_ratio = 0.5, 0.0

        # 7. Index bias
        idx_bias = self._index_bias(index_df, ts) if self.use_index_filter else None

        # 8. Breakout trigger
        trigger_side = None
        vwap_aligned = False
        if (
            len(day) >= 2
            and or_high is not None
            and or_low is not None
            and ts.strftime("%H:%M") <= self.last_entry
        ):
            prev_cl = float(day.iloc[-2]["Close"])
            if price > or_high and prev_cl <= or_high:
                trigger_side = "BUY"
                vwap_aligned = (price > vwap_val and vwap_slope == "up")
            elif price < or_low and prev_cl >= or_low:
                trigger_side = "SELL"
                vwap_aligned = (price < vwap_val and vwap_slope == "down")

        quality_ok = True
        if trigger_side == "BUY":
            quality_ok = (close_pos >= self.min_close_pos and body_ratio >= self.min_body_ratio)
        elif trigger_side == "SELL":
            quality_ok = (close_pos <= 1 - self.min_close_pos and body_ratio >= self.min_body_ratio)

        # 9. Gate evaluation
        sig = None
        if len(df) >= 30 and trigger_side is not None and or_valid:
            trend_ok = (trigger_side == "BUY" and trend_state == "up") or (
                trigger_side == "SELL" and trend_state == "down"
            )
            vol_ok = (vol_base > 0 and current_vol >= self.vol_mult * vol_base)
            vwap_ok = (not self.use_vwap) or vwap_aligned
            qual_ok = (not self.use_quality) or quality_ok
            if self.use_index_filter:
                idx_ok = (trigger_side == "BUY" and idx_bias == "up") or (
                    trigger_side == "SELL" and idx_bias == "down"
                )
            else:
                idx_ok = True
            risk = buy_risk if trigger_side == "BUY" else sell_risk
            risk_ok = (risk > 0 and risk <= self.max_risk_atr * atr)

            from trading.costs import check_cost_floor
            cand_stop = buy_stop if trigger_side == "BUY" else sell_stop
            cost_floor_ok, _ = check_cost_floor(price, cand_stop, max_ratio=self.max_cost_risk_ratio)

            key = (symbol, ts.date())
            taken = self.one_per_side and (trigger_side in self._taken.get(key, set()))

            if (trend_ok and vol_ok and vwap_ok and qual_ok and idx_ok
                    and risk_ok and cost_floor_ok and not taken):
                stop_val = buy_stop if trigger_side == "BUY" else sell_stop
                target_val = buy_target if trigger_side == "BUY" else sell_target
                sig = Signal(
                    symbol=symbol,
                    side=trigger_side,
                    price=price,
                    stop_loss=stop_val,
                    target=target_val,
                    strategy_id=self.strategy_id,
                    trend_state=trend_state,
                )

        return {
            "signal": sig,
            "symbol": symbol,
            "price": price,
            "ts": ts,
            "orb": {
                "or_high": or_high,
                "or_low": or_low,
                "or_width_pct": or_pct,
                "or_valid": or_valid,
            },
            "trend": {
                "price": price,
                "sma200_1h": round(sma200_1h, 2) if sma200_1h is not None else None,
                "sma200_30m": round(sma200_30m, 2) if sma200_30m is not None else None,
                "state": trend_state,
            },
            "volume": {
                "breakout_bar_vol": current_vol,
                "avg20": avg20,
                "ratio": vol_ratio,
                "bar_complete": True,
                "baseline": vol_base,
                "source": vol_src,
            },
            "vwap": {
                "value": vwap_val,
                "slope": vwap_slope,
                "aligned": vwap_aligned,
            },
            "quality": {
                "close_pos": close_pos,
                "body_ratio": body_ratio,
                "ok": quality_ok,
            },
            "index_bias": idx_bias,
            "atr": atr,
            "buy_stop": buy_stop,
            "buy_target": buy_target,
            "sell_stop": sell_stop,
            "sell_target": sell_target,
        }

    # ---- entry point ---------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        now: datetime | None = None,
        index_df: pd.DataFrame | None = None,
    ) -> Signal | None:
        setup = self.compute_setup(df, symbol, now=now, index_df=index_df)
        sig = setup.get("signal")
        if sig is not None and self.auto_mark_taken and setup.get("ts") is not None:
            self.confirm_taken(symbol, sig.side, setup["ts"])
        return sig

    # ---- exit helper ---------------------------------------------------

    def should_exit(
        self,
        df: pd.DataFrame,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        entry_ts,
        now: datetime | None = None,
    ) -> str | None:
        """Discretionary exit check. Engine calls this each closed bar while a
        position is open. Returns a reason string to exit at market, else None.

        failed_breakout : bar closed back inside the opening range
        time_stop       : N bars passed and trade never reached time_stop_min_r
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return None
        df = drop_unclosed_bars(df, interval_minutes=5, now=now)
        if df.empty:
            return None

        ts = df.index[-1]
        idx = _ist_idx(df)
        e_ts = _ist_ts(entry_ts)
        bars = df[idx > e_ts]
        if bars.empty:
            return None

        day = df[df.index.date == ts.date()]
        lv = self._or_levels(day)

        atr_series = _atr(df, self.atr_len)
        atr_raw = atr_series.iloc[-1] if not atr_series.empty else None
        atr = float(atr_raw) if (atr_raw is not None and not pd.isna(atr_raw) and atr_raw > 0) \
            else float(df["Close"].iloc[-1]) * 0.005

        if self.use_fail_exit and lv is not None and ts >= lv[2]:
            or_high, or_low = lv[0], lv[1]
            c = float(bars["Close"].iloc[-1])
            buf = self.fail_exit_buffer_atr * atr
            if side == "BUY" and c < or_high - buf:
                return "failed_breakout"
            if side == "SELL" and c > or_low + buf:
                return "failed_breakout"

        risk = abs(entry_price - stop_loss)
        if self.time_stop_bars > 0 and risk > 0 and len(bars) >= self.time_stop_bars:
            if side == "BUY":
                mfe = float(bars["High"].max()) - entry_price
            else:
                mfe = entry_price - float(bars["Low"].min())
            if mfe < self.time_stop_min_r * risk:
                return "time_stop"

        return None
