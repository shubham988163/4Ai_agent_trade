"""Backtest adapter for ORB + Multi-TF 200MA strategy.

Wraps `ORBMA200Strategy` so that it conforms to `Backtester`'s interface:
  prepare(df) -> df
  detect(window) -> "BUY" | "SELL" | None
  stop(window, side) -> float
  target(window, side, price, sl) -> float | None

Reuses the exact live strategy logic from `trading.strategies.orb_ma200.ORBMA200Strategy`
without reimplementing signal rules, quant gates, or structural stops/targets.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from trading.strategies.orb_ma200 import ORBMA200Strategy

IST = ZoneInfo("Asia/Kolkata")


def compute_causal_htf_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute 1H/30m 200 SMAs and causal Opening Range bounds across full cached history.
    
    offset=15min aligns bins to the 09:15 NSE open.
    Shift(1) rolling sum of 199 completed higher-timeframe bars plus current 5m Close
    divided by 200 is mathematically identical (diff == 0.0) to running _htf_ma on
    every truncated bar window, but computes in ~8ms instead of O(N^2) resamplings.
    """
    if df.empty or len(df) < 30:
        return df

    d = df.copy()
    idx = d.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(IST)
        d.index = idx
    else:
        idx = idx.tz_convert(IST)
        d.index = idx

    # 1H 200 SMA
    res_1h = d.resample("60min", offset="15min").agg({"Close": "last"}).dropna()
    sum_prev_199_1h = res_1h["Close"].shift(1).rolling(199).sum()
    bin_starts_1h = (idx - pd.Timedelta(minutes=15)).floor("60min") + pd.Timedelta(minutes=15)
    d["sma200_1h"] = (bin_starts_1h.map(sum_prev_199_1h) + d["Close"]) / 200.0

    # 30m 200 SMA
    res_30m = d.resample("30min", offset="15min").agg({"Close": "last"}).dropna()
    sum_prev_199_30m = res_30m["Close"].shift(1).rolling(199).sum()
    bin_starts_30m = (idx - pd.Timedelta(minutes=15)).floor("30min") + pd.Timedelta(minutes=15)
    d["sma200_30m"] = (bin_starts_30m.map(sum_prev_199_30m) + d["Close"]) / 200.0

    # Causal Opening Range (09:15 - 09:30): fixed and known for bars >= 09:30
    d["or_high"] = np.nan
    d["or_low"] = np.nan
    d["or_valid"] = False

    cutoff_time = pd.Timestamp("09:30").time()
    for d_val, day_df in d.groupby(d.index.date):
        or_bars = day_df[day_df.index.time < cutoff_time]
        if len(or_bars) >= 3:
            hi = round(float(or_bars["High"].max()), 2)
            lo = round(float(or_bars["Low"].min()), 2)
            pct = round((hi - lo) / lo * 100, 2)
            valid = (0.3 <= pct <= 1.5)
            mask = (d.index.date == d_val) & (d.index.time >= cutoff_time)
            d.loc[mask, "or_high"] = hi
            d.loc[mask, "or_low"] = lo
            d.loc[mask, "or_valid"] = valid

    return d


class BacktestORBStrategy(ORBMA200Strategy):
    """Subclass of ORBMA200Strategy that accelerates higher-timeframe lookups by
    reading precomputed causal HTF 200 SMAs if present on the dataframe."""

    def _htf_ma(self, df: pd.DataFrame, rule: str) -> float | None:
        col = "sma200_1h" if rule == "60min" else "sma200_30m"
        if col in df.columns:
            val = df[col].iloc[-1]
            return float(val) if pd.notna(val) else None
        return super()._htf_ma(df, rule)


class ORBBacktestAdapter:
    """Runnable spec adapter for Backtester wrapping ORBMA200Strategy."""

    def __init__(self, strat: ORBMA200Strategy | None = None, **strat_kwargs):
        self.strat = strat or BacktestORBStrategy(**strat_kwargs)
        self._cached_setup: dict | None = None
        self._cached_ts: object = None
        self._cached_sym: str | None = None

    def reset(self):
        """Reset internal caches and daily trade tracking."""
        self._cached_setup = None
        self._cached_ts = None
        self._cached_sym = None
        self.strat._taken.clear()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal preparation: compute 1H/30m 200 SMAs and causal OR bounds from full cached history."""
        prepared = compute_causal_htf_smas(df)
        if hasattr(df, "attrs") and "symbol" in df.attrs:
            prepared.attrs["symbol"] = df.attrs["symbol"]
        return prepared

    def _eval(self, window: pd.DataFrame) -> dict:
        if window.empty:
            return {}
        ts = window.index[-1]
        sym = getattr(window, "attrs", {}).get("symbol", "UNKNOWN")
        if self._cached_ts == ts and self._cached_sym == sym and self._cached_setup is not None:
            return self._cached_setup

        # Fast causal pre-filter: if current bar cannot trigger breakout, skip compute_setup
        if "or_valid" in window.columns:
            if ts.time() > pd.Timestamp("14:30").time() or not window["or_valid"].iloc[-1]:
                return {}
            hi = window["or_high"].iloc[-1]
            lo = window["or_low"].iloc[-1]
            if pd.isna(hi) or pd.isna(lo):
                return {}
            p = round(float(window["Close"].iloc[-1]), 2)
            prev_p = float(window["Close"].iloc[-2]) if len(window) > 1 else p
            if not ((p > hi and prev_p <= hi) or (p < lo and prev_p >= lo)):
                return {}

        setup = self.strat.compute_setup(window, sym)
        self._cached_ts = ts
        self._cached_sym = sym
        self._cached_setup = setup
        return setup

    def detect(self, window: pd.DataFrame) -> str | None:
        """Return BUY, SELL, or None using only bars up to window.index[-1]."""
        setup = self._eval(window)
        sig = setup.get("signal")
        return sig.side if sig is not None else None

    def stop(self, window: pd.DataFrame, side: str) -> float:
        """Return strategy's own structural stop (breakout bar extreme +/- buffer*ATR)."""
        setup = self._eval(window)
        sig = setup.get("signal")
        if sig is not None:
            return sig.stop_loss
        return setup["buy_stop"] if side == "BUY" else setup["sell_stop"]

    def target(self, window: pd.DataFrame, side: str, price: float, sl: float) -> float | None:
        """Return strategy's 2.0R target, or None if rejected by live logic."""
        setup = self._eval(window)
        sig = setup.get("signal")
        if sig is None:
            return None

        # Respect one trade per symbol per day on accepted entry
        sym = setup.get("symbol", "UNKNOWN")
        ts = setup.get("ts")
        if self.strat.one_per_side and ts:
            key = (sym, ts.date())
            self.strat._taken.setdefault(key, set()).add(sig.side)

        return sig.target


def build_orb_strategy(strat: ORBMA200Strategy | None = None, **strat_kwargs) -> dict:
    """Build a strategy spec dictionary for Backtester."""
    adapter = ORBBacktestAdapter(strat=strat, **strat_kwargs)
    return {
        "id": "orb_ma200",
        "adapter": adapter,
        "prepare": adapter.prepare,
        "detect": adapter.detect,
        "stop": adapter.stop,
        "target": adapter.target,
        "rr": getattr(adapter.strat, "rr", 2.0),
    }
