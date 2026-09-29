"""Shadow P&L resolution engine for candidate signals.

Resolves hypothetical trades for candidate setups (whether approved, rejected by
council, or gated) to measure council alpha and quantify opportunity costs.

Resolution Rules:
1. Long (BUY):
   - Stop hit if bar['Low'] <= stop_loss (assumed to fill first if both hit in bar)
   - Target hit if bar['High'] >= target
2. Short (SELL):
   - Stop hit if bar['High'] >= stop_loss (assumed to fill first if both hit in bar)
   - Target hit if bar['Low'] <= target
3. Intraday Square-off:
   - If neither hit and bar time reaches >= 15:15 IST, close at bar['Close'].
"""
from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

from trading.config import YF_SUFFIX, SQUAREOFF_TIME
from trading.exits import TrailState, trail_config_from_settings, update_exit
from trading.ledger import Ledger
from trading.strategies.orb_ma200 import _atr

IST = ZoneInfo("Asia/Kolkata")


def resolve_candidate_from_bars(
    candidate: dict,
    df_bars: pd.DataFrame,
    trail_config=None,
) -> dict | None:
    """Evaluate an open candidate setup sequentially against subsequent bars.

    ``trail_config`` defaults to the system toggle (``config.TRAIL_ENABLED``, off by
    default), so shadow resolution and the live paper loop move together.
    """
    if df_bars.empty or not isinstance(df_bars.index, pd.DatetimeIndex):
        return None

    trail_config = trail_config or trail_config_from_settings()

    cand_ts = float(candidate.get("ts", 0.0))
    entry = float(candidate["price"])
    stop = float(candidate["stop_loss"])
    initial_stop = stop
    target = float(candidate["target"])
    side = str(candidate["side"]).upper()
    qty = int(candidate.get("qty") or 1)
    risk_per_share = abs(entry - initial_stop)

    # Filter bars strictly occurring on or after candidate entry ts
    if df_bars.index.tz is None:
        idx_ts = df_bars.index.tz_localize(IST).astype("int64") // 10**9
    else:
        idx_ts = df_bars.index.tz_convert(IST).astype("int64") // 10**9

    subsequent_bars = df_bars[idx_ts >= cand_ts]
    if subsequent_bars.empty:
        # If no bar has ts >= cand_ts, check if the last day bars match candidate date
        cand_date = candidate.get("date")
        if cand_date:
            subsequent_bars = df_bars[df_bars.index.strftime("%Y-%m-%d") == cand_date]
        if subsequent_bars.empty:
            return None

    # ATR for the chandelier trail, from the same definition the strategy itself uses.
    atr_vals = None
    if trail_config.enabled and risk_per_share > 0:
        atr_series = _atr(df_bars, 14)
        atr_vals = atr_series.reindex(subsequent_bars.index).to_numpy(dtype=float)

    trail_state = TrailState()
    remaining_qty = qty
    # (price, qty_closed, reason) -- one leg per exit, plus one per partial booking.
    legs: list[tuple[float, int, str]] = []

    for i in range(len(subsequent_bars)):
        bar = subsequent_bars.iloc[i]
        bar_dt = subsequent_bars.index[i]
        if bar_dt.tzinfo is None:
            bar_dt = bar_dt.replace(tzinfo=IST)
        else:
            bar_dt = bar_dt.astimezone(IST)

        op = float(bar["Open"])
        lo = float(bar["Low"])
        hi = float(bar["High"])
        cl = float(bar["Close"])
        time_str = bar_dt.strftime("%H:%M")

        exit_price = None
        exit_reason = None

        if side == "BUY":
            # Conservative: stop checked first
            if lo <= stop:
                exit_price = stop
                exit_reason = "stop_loss_hit"
            elif hi >= target:
                exit_price = target
                exit_reason = "target_hit"
            elif time_str >= SQUAREOFF_TIME:
                exit_price = op  # Resolve at 15:15 bar open
                exit_reason = "squareoff_time"
        elif side == "SELL":
            # Conservative: stop checked first
            if hi >= stop:
                exit_price = stop
                exit_reason = "stop_loss_hit"
            elif lo <= target:
                exit_price = target
                exit_reason = "target_hit"
            elif time_str >= SQUAREOFF_TIME:
                exit_price = op  # Resolve at 15:15 bar open
                exit_reason = "squareoff_time"

        if exit_price is not None:
            # Apply identical slippage and statutory charges as paper fills
            from trading.config import SLIPPAGE_PCT
            from trading.costs import round_trip as round_trip_charges

            legs.append((exit_price, remaining_qty, exit_reason))

            # Charges are priced PER LEG: a partial booking is a separate sell order, so
            # brokerage/STT/GST apply to it again. One round_trip() over the whole position
            # would understate cost. With trailing off there is exactly one leg, and the
            # arithmetic below reduces to the original single-exit calculation.
            slip_entry = SLIPPAGE_PCT * entry
            entry_fill = entry + slip_entry if side == "BUY" else entry - slip_entry

            gross_pnl = 0.0
            charges = 0.0
            for leg_price, leg_qty, _leg_reason in legs:
                if leg_qty <= 0:
                    continue
                slip_exit = SLIPPAGE_PCT * leg_price
                exit_fill = leg_price - slip_exit if side == "BUY" else leg_price + slip_exit
                pnl_per_share = (exit_fill - entry_fill) if side == "BUY" else (entry_fill - exit_fill)
                gross_pnl += pnl_per_share * leg_qty
                charges += round_trip_charges(entry_fill, exit_fill, leg_qty)
            net_pnl = gross_pnl - charges

            label = exit_reason if len(legs) == 1 else "partial_exit+" + exit_reason
            return {
                "candidate_id": candidate["id"],
                "raw_exit_price": round(exit_price, 2),
                "exit_price": round(exit_fill, 2),
                "exit_ts": bar_dt.timestamp(),
                "exit_reason": label,
                "gross_pnl": round(gross_pnl, 2),
                "charges": round(charges, 2),
                "shadow_pnl": round(net_pnl, 2),
            }

        # No exit on this bar -- advance the trail on this bar's close for the next bar.
        # Updating AFTER the exit tests is what keeps this look-ahead free.
        if atr_vals is not None:
            a = atr_vals[i]
            a = 0.0 if a != a else float(a)  # NaN -> no chandelier, breakeven floor still holds
            stop, trail_state, partial_qty = update_exit(
                entry, side, stop, cl, a, trail_state,
                qty=remaining_qty, risk_per_share=risk_per_share, config=trail_config,
            )
            if partial_qty > 0:
                legs.append((cl, partial_qty, "partial_exit"))
                remaining_qty -= partial_qty

    return None


def resolve_open_shadow_candidates(
    ledger: Ledger,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """Fetch open candidates and resolve them using provided or fetched candles."""
    open_cands = ledger.open_shadow_candidates()
    if not open_cands:
        return []

    resolved = []
    for cand in open_cands:
        sym = cand["symbol"]
        df_bars = None
        if bars_by_symbol and sym in bars_by_symbol:
            df_bars = bars_by_symbol[sym]
        else:
            try:
                yf_sym = sym + YF_SUFFIX if not sym.endswith(YF_SUFFIX) else sym
                t = yf.Ticker(yf_sym)
                df_bars = t.history(period="5d", interval="5m")
                if not df_bars.empty:
                    df_bars = df_bars.tz_convert(IST)
            except Exception:
                continue

        if df_bars is None or df_bars.empty:
            continue

        res = resolve_candidate_from_bars(cand, df_bars)
        if res:
            ledger.record_shadow_exit(
                candidate_id=res["candidate_id"],
                exit_price=res["exit_price"],
                exit_ts=res["exit_ts"],
                shadow_pnl=res["shadow_pnl"],
            )
            resolved.append(res)

    return resolved


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("  Resolving Open Shadow Candidates (Paper Mode)")
    print("=" * 60)
    ledger = Ledger()
    resolved = resolve_open_shadow_candidates(ledger)
    print(f"Total candidate setups resolved: {len(resolved)}")
    for r in resolved:
        print(f"  Candidate #{r['candidate_id']}: exit={r['exit_price']} ({r['exit_reason']}) | P&L: Rs {r['shadow_pnl']:+.2f}")

