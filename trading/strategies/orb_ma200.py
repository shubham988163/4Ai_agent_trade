"""ORB + Multi-TF 200MA trend-filter strategy.

Python port of the TradingView strategy (orb_ma_strategy.pine), written for the
plug-and-play `Strategy` / `Signal` interface in trading/strategies/.

Rules
-----
1. Opening range = 09:15-09:30 high/low. Skip the day if the range is
   < min_or_pct or > max_or_pct of price (too tight = fake breakouts,
   too wide = move already happened).
2. Trigger = a bar CLOSES beyond the OR, and the previous bar had not
   (fresh breakout, no chasing). Wicks that don't close are ignored.
3. Trend gate (the core rule): price above the 200 MA = uptrend, below =
   downtrend, on BOTH 1H and 30min. Longs only in uptrend, shorts only in
   downtrend. If there isn't enough history to build a 200 MA, no trade.
4. Confirmation: breakout-bar volume >= vol_mult x average of prior 20 bars,
   and price on the right side of a sloping session VWAP.
5. Structural stop = breakout bar extreme +/- atr_buffer x ATR. Target = rr x
   risk. Trades with a stop wider than max_risk_atr x ATR are skipped.
6. No entries after last_entry; one long + one short per symbol per day.

Data requirement: a 200-bar 1H MA needs ~32 sessions of intraday bars
(about 2,400 five-minute bars). Feed the strategy that much history or it
will (correctly) return None for every symbol.

Position sizing is NOT done here. The engine sizes qty from RISK_PER_TRADE
and the stop distance, and the router enforces DAILY_LOSS_LIMIT,
DAILY_PROFIT_TARGET and MAX_POSITION_VALUE.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from trading.strategy import Strategy, Signal

IST = ZoneInfo("Asia/Kolkata")


def drop_unclosed_bars(
    df: pd.DataFrame,
    interval_minutes: int = 5,
    now: datetime | None = None,
    grace_seconds: int = 30,
) -> pd.DataFrame:
    """Drop any bar not yet closed (bar end time + grace > now).
    
    A 5-minute bar stamped 14:35 covers 14:35:00 to 14:40:00.
    With grace_seconds=30, the bar is considered closed only when
    now >= 14:40:30, ensuring data vendor volume has finalized and
    eliminating partial-bar false triggers.
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
        last_entry: str = "14:30",
        one_per_side: bool = True,
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
        """'up' if price is above every enabled 200MA, 'down' if below every
        enabled 200MA, 'unavailable' if not enough history, else 'mixed'."""
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

    def compute_setup(
        self, df: pd.DataFrame, symbol: str, now: datetime | None = None
    ) -> dict:
        """Compute the full strategy indicators and trigger checks in a single pass.

        Single source of truth for ORB, Multi-TF 200MA trend, Volume, VWAP, and
        structural stops/targets. Returns a dictionary consumed directly by
        MarketFacts and evaluate().
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return {
                "signal": None,
                "symbol": symbol,
                "price": 0.0,
                "ts": None,
                "orb": {"or_high": None, "or_low": None, "or_width_pct": None, "or_valid": False},
                "trend": {"price": 0.0, "sma200_1h": None, "sma200_30m": None, "state": "unavailable"},
                "volume": {"breakout_bar_vol": 0.0, "avg20": 0.0, "ratio": 0.0, "bar_complete": False},
                "vwap": {"value": 0.0, "slope": "flat", "aligned": False},
                "atr": 0.0,
                "buy_stop": 0.0,
                "buy_target": 0.0,
                "sell_stop": 0.0,
                "sell_target": 0.0,
            }

        # Drop any bar not yet closed
        if now is not None:
            df = drop_unclosed_bars(df, interval_minutes=5, now=now)

        if len(df) == 0:
            return {
                "signal": None,
                "symbol": symbol,
                "price": 0.0,
                "ts": None,
                "orb": {"or_high": None, "or_low": None, "or_width_pct": None, "or_valid": False},
                "trend": {"price": 0.0, "sma200_1h": None, "sma200_30m": None, "state": "unavailable"},
                "volume": {"breakout_bar_vol": 0.0, "avg20": 0.0, "ratio": 0.0, "bar_complete": False},
                "vwap": {"value": 0.0, "slope": "flat", "aligned": False},
                "atr": 0.0,
                "buy_stop": 0.0,
                "buy_target": 0.0,
                "sell_stop": 0.0,
                "sell_target": 0.0,
            }

        ts = df.index[-1]
        price = round(float(df["Close"].iloc[-1]), 2)

        # 1. Multi-TF 200 SMA trend
        sma200_1h = self._htf_ma(df, "60min") if self.use_1h else None
        sma200_30m = self._htf_ma(df, "30min") if self.use_30m else None
        trend_state = self._trend(df, price)

        # 2. Opening range (09:15 to 09:15 + or_minutes)
        day = df[df.index.date == ts.date()]
        or_high, or_low, or_pct = None, None, None
        or_valid = False

        if len(day) >= 3:
            or_end = day.index[0].normalize() + pd.Timedelta(
                hours=9, minutes=15 + self.or_minutes
            )
            or_bars = day[day.index < or_end]
            if not or_bars.empty and ts >= or_end:
                or_high = round(float(or_bars["High"].max()), 2)
                or_low = round(float(or_bars["Low"].min()), 2)
                if or_low > 0:
                    or_pct = round((or_high - or_low) / or_low * 100, 2)
                    or_valid = (self.min_or_pct <= or_pct <= self.max_or_pct)

        # 3. Volume confirmation
        current_vol = round(float(df["Volume"].iloc[-1]), 2)
        if len(df) >= 21:
            avg20 = round(float(df["Volume"].iloc[-21:-1].mean()), 2)
        elif len(df) > 1:
            avg20 = round(float(df["Volume"].iloc[:-1].mean()), 2)
        else:
            avg20 = current_vol

        vol_ratio = round(current_vol / avg20, 2) if avg20 > 0 else 1.0

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

        # 5. ATR & Structural Stops
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

        # 6. Breakout Trigger
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

        # 7. Quant Gate Evaluation
        sig = None
        if len(df) >= 30 and trigger_side is not None and or_valid:
            trend_ok = (trigger_side == "BUY" and trend_state == "up") or (
                trigger_side == "SELL" and trend_state == "down"
            )
            vol_ok = (avg20 > 0 and current_vol >= self.vol_mult * avg20)
            vwap_ok = (not self.use_vwap) or vwap_aligned
            risk = buy_risk if trigger_side == "BUY" else sell_risk
            risk_ok = (risk > 0 and risk <= self.max_risk_atr * atr)

            # Cost-floor gate: friction % of entry / stop distance % (shared helper)
            from trading.costs import check_cost_floor
            cand_stop = buy_stop if trigger_side == "BUY" else sell_stop
            cost_floor_ok, _ = check_cost_floor(price, cand_stop, max_ratio=self.max_cost_risk_ratio)

            key = (symbol, ts.date())
            taken = self.one_per_side and (trigger_side in self._taken.get(key, set()))

            if trend_ok and vol_ok and vwap_ok and risk_ok and cost_floor_ok and not taken:
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
            },
            "vwap": {
                "value": vwap_val,
                "slope": vwap_slope,
                "aligned": vwap_aligned,
            },
            "atr": atr,
            "buy_stop": buy_stop,
            "buy_target": buy_target,
            "sell_stop": sell_stop,
            "sell_target": sell_target,
        }

    # ---- main entry point ---------------------------------------------

    def evaluate(self, df: pd.DataFrame, symbol: str, now: datetime | None = None) -> Signal | None:
        setup = self.compute_setup(df, symbol, now=now)
        sig = setup.get("signal")
        if sig is not None and self.one_per_side and setup.get("ts"):
            key = (symbol, setup["ts"].date())
            self._taken.setdefault(key, set()).add(sig.side)
        return sig


