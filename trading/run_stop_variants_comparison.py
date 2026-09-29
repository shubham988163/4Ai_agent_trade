"""Comparative analysis of 4 stop variants under the cost-floor gate using FyersClient.history().

Variants evaluated:
(a) Current breakout-bar stop (5m candle)
(b) Opposite OR boundary (5m candle)
(c) OR midpoint (5m candle)
(d) 15-minute candles with breakout-bar stop

All variants enforce:
- Valid OR width (0.3% to 1.5%)
- Multi-TF 200 SMA trend gate (1H and 30m)
- Breakout volume >= 1.5x prior 20-bar avg
- VWAP alignment (slope and position)
- Cost-floor gate: friction <= 30% of stop distance
- 2.0 R:R target
- Realistic shadow fills: 0.05% slippage on entry and exit, Zerodha round-trip charges, 15:15 bar open square-off.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from trading.config import WATCHLIST, SLIPPAGE_PCT, CHARGES_PCT_ROUND_TRIP, MAX_COST_RISK_RATIO
from trading.costs import round_trip, check_cost_floor
from trading.fno.fyers import FyersClient

IST = ZoneInfo("Asia/Kolkata")


def fetch_fyers_df(client: FyersClient, symbol: str, resolution: str = "5") -> pd.DataFrame:
    """Fetch candles from Fyers API and convert to IST DatetimeIndex DataFrame."""
    fyers_sym = f"NSE:{symbol}-EQ"
    candles = client.history(fyers_sym, resolution=resolution, days=100)
    if not candles:
        return pd.DataFrame()
    rows = []
    for c in candles:
        dt = datetime.fromtimestamp(c["timestamp"], tz=IST)
        rows.append({
            "datetime": dt,
            "Open": float(c["open"]),
            "High": float(c["high"]),
            "Low": float(c["low"]),
            "Close": float(c["close"]),
            "Volume": float(c["volume"]),
        })
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return df


def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def calc_vwap(day_df: pd.DataFrame) -> pd.Series:
    typical = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
    vol_sum = day_df["Volume"].cumsum()
    return (typical * day_df["Volume"]).cumsum() / vol_sum.where(vol_sum > 0, 1.0)


def simulate_variant(
    df: pd.DataFrame,
    symbol: str,
    variant_type: str,  # 'breakout_bar', 'opposite_or', 'or_midpoint', '15m_bar'
    interval_minutes: int = 5,
) -> dict:
    """Simulate a specific stop variant over the dataset."""
    days = sorted(list(set(df.index.date)))
    trades = []
    raw_signals_count = 0
    filtered_by_cost_count = 0

    # Resample for HTF 200 SMAs
    df_1h = df.resample("60min", offset="15min").agg({"Close": "last"}).dropna()
    ma1h_series = df_1h["Close"].rolling(200).mean().reindex(df.index, method="ffill")

    df_30m = df.resample("30min", offset="15min").agg({"Close": "last"}).dropna()
    ma30m_series = df_30m["Close"].rolling(200).mean().reindex(df.index, method="ffill")

    atr_series = calc_atr(df, n=14)

    for d in days:
        day_df = df[df.index.date == d]
        if len(day_df) < (3 if interval_minutes == 5 else 2):
            continue

        # Opening Range: 09:15 to 09:30
        or_end = day_df.index[0].normalize() + pd.Timedelta(hours=9, minutes=30)
        or_bars = day_df[day_df.index < or_end]
        if or_bars.empty:
            continue

        or_high = float(or_bars["High"].max())
        or_low = float(or_bars["Low"].min())
        if or_low <= 0:
            continue
        or_width_pct = (or_high - or_low) / or_low * 100
        if not (0.30 <= or_width_pct <= 1.50):
            continue

        day_vwap = calc_vwap(day_df)

        taken_side = set()

        for idx in range(len(day_df)):
            bar_dt = day_df.index[idx]
            if bar_dt < or_end:
                continue
            time_str = bar_dt.strftime("%H:%M")
            if time_str > "14:30":
                break

            sub_df = df.loc[:bar_dt]
            if len(sub_df) < 30:
                continue

            close = float(day_df["Close"].iloc[idx])
            prev_close = float(day_df["Close"].iloc[idx - 1]) if idx > 0 else close

            trigger_side = None
            if close > or_high and prev_close <= or_high:
                trigger_side = "BUY"
            elif close < or_low and prev_close >= or_low:
                trigger_side = "SELL"

            if not trigger_side or trigger_side in taken_side:
                continue

            # Multi-TF Trend check
            ma1h = ma1h_series.get(bar_dt)
            ma30m = ma30m_series.get(bar_dt)
            if pd.isna(ma1h) or pd.isna(ma30m):
                continue
            if trigger_side == "BUY" and not (close > ma1h and close > ma30m):
                continue
            if trigger_side == "SELL" and not (close < ma1h and close < ma30m):
                continue

            # Volume check (breakout bar volume >= 1.5x prior 20-bar avg)
            bar_vol = float(day_df["Volume"].iloc[idx])
            prior_vol = sub_df["Volume"].iloc[-21:-1]
            if len(prior_vol) < 20:
                continue
            avg20_vol = float(prior_vol.mean())
            if avg20_vol <= 0 or bar_vol < 1.5 * avg20_vol:
                continue

            # VWAP check
            vwap_val = float(day_vwap.get(bar_dt, 0))
            # VWAP slope (compare to 3 bars ago)
            prev_vwap_idx = max(0, idx - 3)
            prev_vwap_val = float(day_vwap.iloc[prev_vwap_idx])
            vwap_slope_up = (vwap_val > prev_vwap_val)
            vwap_slope_down = (vwap_val < prev_vwap_val)

            if trigger_side == "BUY" and not (close > vwap_val and vwap_slope_up):
                continue
            if trigger_side == "SELL" and not (close < vwap_val and vwap_slope_down):
                continue

            # Calculate Stop Loss according to variant
            atr = float(atr_series.get(bar_dt, 0))
            if atr <= 0:
                continue

            bar_high = float(day_df["High"].iloc[idx])
            bar_low = float(day_df["Low"].iloc[idx])

            if variant_type in ("breakout_bar", "15m_bar"):
                if trigger_side == "BUY":
                    stop_loss = round(bar_low - 0.25 * atr, 2)
                else:
                    stop_loss = round(bar_high + 0.25 * atr, 2)
            elif variant_type == "opposite_or":
                if trigger_side == "BUY":
                    stop_loss = round(or_low - 0.25 * atr, 2)
                else:
                    stop_loss = round(or_high + 0.25 * atr, 2)
            elif variant_type == "or_midpoint":
                or_mid = round((or_high + or_low) / 2.0, 2)
                stop_loss = or_mid
            else:
                raise ValueError(f"Unknown variant: {variant_type}")

            risk = abs(close - stop_loss)
            if risk <= 0:
                continue

            raw_signals_count += 1

            # Cost-Floor Gate Check
            passed, cost_ratio = check_cost_floor(close, stop_loss, max_ratio=MAX_COST_RISK_RATIO)
            if not passed:
                filtered_by_cost_count += 1
                continue

            # Compute Target (2.0 R:R)
            target = round(close + 2.0 * risk if trigger_side == "BUY" else close - 2.0 * risk, 2)
            taken_side.add(trigger_side)

            # Resolve trade over remaining bars of the day
            remaining_bars = day_df.iloc[idx + 1 :]
            exit_price = None
            exit_reason = None

            for r_idx in range(len(remaining_bars)):
                r_bar = remaining_bars.iloc[r_idx]
                r_time = remaining_bars.index[r_idx].strftime("%H:%M")
                r_high = float(r_bar["High"])
                r_low = float(r_bar["Low"])

                if trigger_side == "BUY":
                    if r_high >= target:
                        exit_price = target
                        exit_reason = "target"
                        break
                    if r_low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "stop"
                        break
                else:
                    if r_low <= target:
                        exit_price = target
                        exit_reason = "target"
                        break
                    if r_high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "stop"
                        break

                if r_time >= "15:15":
                    exit_price = float(r_bar["Open"])
                    exit_reason = "1515_squareoff"
                    break

            if exit_price is None and not remaining_bars.empty:
                exit_price = float(remaining_bars["Close"].iloc[-1])
                exit_reason = "eod"

            if exit_price is not None:
                # Apply slippage & charges
                fill_entry = close * (1 + SLIPPAGE_PCT) if trigger_side == "BUY" else close * (1 - SLIPPAGE_PCT)
                fill_exit = exit_price * (1 - SLIPPAGE_PCT) if trigger_side == "BUY" else exit_price * (1 + SLIPPAGE_PCT)
                gross_pnl = round(fill_exit - fill_entry if trigger_side == "BUY" else fill_entry - fill_exit, 2)
                charges = round(round_trip(fill_entry, fill_exit, 1), 2)
                shadow_pnl = round(gross_pnl - charges, 2)

                trades.append({
                    "symbol": symbol,
                    "date": bar_dt.strftime("%Y-%m-%d"),
                    "time": time_str,
                    "side": trigger_side,
                    "entry": close,
                    "stop": stop_loss,
                    "target": target,
                    "risk": round(risk, 2),
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "gross_pnl": gross_pnl,
                    "charges": charges,
                    "shadow_pnl": shadow_pnl,
                })

    return {
        "raw_signals": raw_signals_count,
        "filtered_by_cost": filtered_by_cost_count,
        "executed_trades": trades,
    }


def run_comparison():
    print("=" * 80)
    print("  STOP VARIANTS COMPARISON UNDER COST-FLOOR GATE (FYERS 100-DAY HISTORY)")
    print("=" * 80)

    client = FyersClient()
    connected, reason = client.status()
    print(f"Fyers Client Connection: {connected} ({reason})")
    if not connected:
        print("ERROR: Cannot run Fyers comparison without active Fyers session.")
        return

    symbols = WATCHLIST
    variants = [
        ("variant_a_breakout_bar_5m", "breakout_bar", "5", 5, "Variant (a): Current Breakout-Bar Stop (5m)"),
        ("variant_b_opposite_or_5m", "opposite_or", "5", 5, "Variant (b): Opposite OR Boundary (5m)"),
        ("variant_c_or_midpoint_5m", "or_midpoint", "5", 5, "Variant (c): OR Midpoint (5m)"),
        ("variant_d_breakout_bar_15m", "15m_bar", "15", 15, "Variant (d): 15-Minute Candles Breakout-Bar Stop"),
    ]

    # Pre-fetch data for all symbols
    data_5m = {}
    data_15m = {}
    print("\n[+] Fetching 100-day intraday histories from Fyers...")
    for sym in symbols:
        df5 = fetch_fyers_df(client, sym, resolution="5")
        data_5m[sym] = df5
        df15 = fetch_fyers_df(client, sym, resolution="15")
        data_15m[sym] = df15
        print(f"    {sym:<10}: {len(df5)} 5m candles | {len(df15)} 15m candles")

    comparison_results = []

    for v_key, v_type, res_code, interval, label in variants:
        print(f"\n" + "-" * 75)
        print(f"  Evaluating {label}...")
        print("-" * 75)

        total_raw = 0
        total_filtered = 0
        all_trades = []

        for sym in symbols:
            df = data_5m[sym] if interval == 5 else data_15m[sym]
            if df.empty:
                continue
            res = simulate_variant(df, sym, variant_type=v_type, interval_minutes=interval)
            total_raw += res["raw_signals"]
            total_filtered += res["filtered_by_cost"]
            all_trades.extend(res["executed_trades"])

        n_exec = len(all_trades)
        rejection_rate = (total_filtered / total_raw * 100) if total_raw > 0 else 0.0

        if n_exec > 0:
            df_trades = pd.DataFrame(all_trades)
            wins = df_trades[df_trades["shadow_pnl"] > 0]
            losses = df_trades[df_trades["shadow_pnl"] <= 0]
            wr = len(wins) / n_exec * 100
            gross = df_trades["gross_pnl"].sum()
            charges = df_trades["charges"].sum()
            net = df_trades["shadow_pnl"].sum()
            targets_hit = len(df_trades[df_trades["exit_reason"] == "target"])
            stops_hit = len(df_trades[df_trades["exit_reason"] == "stop"])
            squareoffs = len(df_trades[df_trades["exit_reason"] == "1515_squareoff"])
        else:
            wr, gross, charges, net = 0.0, 0.0, 0.0, 0.0
            targets_hit, stops_hit, squareoffs = 0, 0, 0

        summary = {
            "variant": label,
            "raw_signals": total_raw,
            "cost_filtered": total_filtered,
            "rejection_rate": rejection_rate,
            "executed_signals": n_exec,
            "win_rate": wr,
            "gross_pnl": gross,
            "charges": charges,
            "net_shadow_pnl": net,
            "targets_hit": targets_hit,
            "stops_hit": stops_hit,
            "squareoffs": squareoffs,
            "all_trades": all_trades,
        }
        comparison_results.append(summary)

        print(f"  Raw Signals:           {total_raw}")
        print(f"  Cost-Floor Filtered:   {total_filtered} ({rejection_rate:.1f}%)")
        print(f"  Executed Signals:      {n_exec}")
        if n_exec > 0:
            print(f"  Exits breakdown:       {targets_hit} targets, {stops_hit} stops, {squareoffs} 15:15 squareoffs")
            print(f"  Win Rate:              {wr:.1f}%")
            print(f"  Gross P&L:             Rs {gross:+.2f}")
            print(f"  Charges:               Rs {charges:.2f}")
            print(f"  Net Shadow P&L:        Rs {net:+.2f}")

    print("\n" + "=" * 80)
    print("  SUMMARY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Variant':<40} {'Raw':>4} {'CostFilt':>9} {'Exec':>5} {'WinRate':>8} {'Gross(Rs)':>11} {'Charges':>9} {'Net(Rs)':>11}")
    print("-" * 105)
    for r in comparison_results:
        print(f"{r['variant']:<40} {r['raw_signals']:>4} {r['cost_filtered']:>9} {r['executed_signals']:>5} "
              f"{r['win_rate']:>7.1f}% {r['gross_pnl']:>+11.2f} {r['charges']:>9.2f} {r['net_shadow_pnl']:>+11.2f}")


if __name__ == "__main__":
    run_comparison()
