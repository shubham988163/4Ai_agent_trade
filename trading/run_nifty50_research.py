"""High-performance Nifty 50 Strategy-Viability & Baseline Research.

Research questions addressed:
1. Realism: Enter at the NEXT bar's open after the signal bar (not the signal close).
   Measure and report distribution of signal-close vs next-open slippage.
2. Cost Consistency: Exact round_trip() for actual quantity (R-based sizing: 1R = Rs 75 target risk).
3. Metric standard: R-multiples net of slippage and charges, plus % return per trade.
4. Baselines:
   (i) Random direction at same entry times with same exits (500 bootstrap iterations)
   (ii) Plain ORB (no MA/vol/VWAP filters) held to 15:15
   (iii) Full rule set
   Report trades, win rate, net expectancy in R, and bootstrap 95% CI.
5. Scope & Out-of-Sample Discipline:
   Nifty 50 constituent universe over longest Fyers history (100 days).
   Chronological split: First 70% of sessions for exploration, last 30% held out and evaluated once.
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from trading.config import SLIPPAGE_PCT, MAX_COST_RISK_RATIO, REPORTS_DIR
from trading.costs import round_trip, check_cost_floor
from trading.exits import TrailConfig, TrailState, update_exit
from trading.fno.fyers import FyersClient

IST = ZoneInfo("Asia/Kolkata")
CACHE_DIR = Path("data/fyers_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN", "LT", "HINDUNILVR",
    "KOTAKBANK", "AXISBANK", "BAJFINANCE", "M&M", "MARUTI", "SUNPHARMA", "NTPC", "ONGC", "POWERGRID",
    "TITAN", "ADANIENT", "ADANIPORTS", "TATASTEEL", "COALINDIA", "BAJAJFINSV", "JSWSTEEL", "HCLTECH", "ASIANPAINT", "ULTRACEMCO",
    "GRASIM", "TECHM", "WIPRO", "NESTLEIND", "SBILIFE", "HDFCLIFE", "BPCL", "CIPLA", "DRREDDY", "EICHERMOT",
    "TATACONSUM", "BRITANNIA", "HINDALCO", "APOLLOHOSP", "DIVISLAB", "SHRIRAMFIN", "HEROMOTOCO", "BEL", "TRENT", "INDUSINDBK"
]


def get_cached_symbol_data(client: FyersClient, symbol: str) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{symbol}_100d_5m.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    fsym = f"NSE:{symbol}-EQ"
    candles = None
    for attempt in range(5):
        try:
            candles = client.history(fsym, resolution="5", days=100)
            time.sleep(1.2)  # Respect Fyers rate limits
            break
        except Exception as e:
            if "limit" in str(e).lower() and attempt < 4:
                time.sleep(3.0)
            else:
                print(f"    [!] Error fetching {symbol}: {e}", flush=True)
                break
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
    if not df.empty:
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
    return df


def precompute_indicators(df: pd.DataFrame) -> dict:
    df_1h = df.resample("60min", offset="15min").agg({"Close": "last"}).dropna()
    ma1h = df_1h["Close"].rolling(200).mean().reindex(df.index, method="ffill")

    df_30m = df.resample("30min", offset="15min").agg({"Close": "last"}).dropna()
    ma30m = df_30m["Close"].rolling(200).mean().reindex(df.index, method="ffill")

    prev_close = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    return {"ma1h": ma1h, "ma30m": ma30m, "atr": atr}


def calc_vwap(day_df: pd.DataFrame) -> pd.Series:
    typical = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
    vol_sum = day_df["Volume"].cumsum()
    return (typical * day_df["Volume"]).cumsum() / vol_sum.where(vol_sum > 0, 1.0)


def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    arr = np.array(values)
    n = len(arr)
    rng = np.random.default_rng(42)
    means = [float(np.mean(rng.choice(arr, size=n, replace=True))) for _ in range(n_boot)]
    lower = float(np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lower, upper


def run_strategy_simulation(
    all_data: dict[str, pd.DataFrame],
    precomputed: dict[str, dict],
    target_days: list,
    strategy_mode: str,  # 'full_rules', 'plain_orb_1515'
    stop_variant: str = "breakout_bar",  # 'breakout_bar', 'opposite_or', 'or_midpoint'
    target_risk_rs: float = 75.0,
    trail_config: TrailConfig | None = None,
) -> dict:
    all_trades = []
    close_to_open_slippages = []
    cost_floor_rejections = 0
    raw_signals_count = 0
    trail_config = trail_config or TrailConfig(enabled=False)

    for sym, df in all_data.items():
        if df.empty:
            continue
        ind = precomputed[sym]
        ma1h = ind["ma1h"]
        ma30m = ind["ma30m"]
        atr_series = ind["atr"]

        for d in target_days:
            day_df = df[df.index.date == d]
            if len(day_df) < 5:
                continue

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

            for idx in range(len(day_df) - 1):
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

                if strategy_mode != "plain_orb_1515":
                    # Multi-TF 200 SMA Trend
                    m1 = ma1h.get(bar_dt)
                    m30 = ma30m.get(bar_dt)
                    if pd.isna(m1) or pd.isna(m30):
                        continue
                    if trigger_side == "BUY" and not (close > m1 and close > m30):
                        continue
                    if trigger_side == "SELL" and not (close < m1 and close < m30):
                        continue

                    # Volume >= 1.5x prior 20-bar avg
                    bar_vol = float(day_df["Volume"].iloc[idx])
                    prior_vol = sub_df["Volume"].iloc[-21:-1]
                    if len(prior_vol) < 20:
                        continue
                    avg_vol = float(prior_vol.mean())
                    if avg_vol <= 0 or bar_vol < 1.5 * avg_vol:
                        continue

                    # VWAP alignment
                    vwap_val = float(day_vwap.get(bar_dt, 0))
                    prev_v_idx = max(0, idx - 3)
                    prev_v = float(day_vwap.iloc[prev_v_idx])
                    vwap_up = vwap_val > prev_v
                    vwap_down = vwap_val < prev_v
                    if trigger_side == "BUY" and not (close > vwap_val and vwap_up):
                        continue
                    if trigger_side == "SELL" and not (close < vwap_val and vwap_down):
                        continue

                # Signal confirmed! Next bar open is actual entry
                next_open = float(day_df["Open"].iloc[idx + 1])
                co_gap_pct = (next_open - close) / close * 100
                close_to_open_slippages.append({
                    "symbol": sym, "date": d, "time": time_str,
                    "side": trigger_side, "gap_pct": co_gap_pct,
                })

                atr = float(atr_series.get(bar_dt, 0))
                if atr <= 0:
                    continue

                bar_high = float(day_df["High"].iloc[idx])
                bar_low = float(day_df["Low"].iloc[idx])

                if strategy_mode == "plain_orb_1515":
                    stop_loss = or_low if trigger_side == "BUY" else or_high
                else:
                    if stop_variant == "breakout_bar":
                        stop_loss = round(bar_low - 0.25 * atr, 2) if trigger_side == "BUY" else round(bar_high + 0.25 * atr, 2)
                    elif stop_variant == "opposite_or":
                        stop_loss = round(or_low - 0.25 * atr, 2) if trigger_side == "BUY" else round(or_high + 0.25 * atr, 2)
                    elif stop_variant == "or_midpoint":
                        stop_loss = round((or_high + or_low) / 2.0, 2)
                    else:
                        stop_loss = round(bar_low - 0.25 * atr, 2) if trigger_side == "BUY" else round(bar_high + 0.25 * atr, 2)

                raw_risk = abs(next_open - stop_loss)
                if raw_risk <= 0:
                    continue

                raw_signals_count += 1
                qty = max(1, int(target_risk_rs / raw_risk))

                # Exact Cost-Floor Gate Check
                if strategy_mode != "plain_orb_1515":
                    passed, cost_ratio = check_cost_floor(next_open, stop_loss, qty=qty, max_ratio=MAX_COST_RISK_RATIO)
                    if not passed:
                        cost_floor_rejections += 1
                        continue

                target = round(next_open + 2.0 * raw_risk if trigger_side == "BUY" else next_open - 2.0 * raw_risk, 2)
                taken_side.add(trigger_side)

                fill_entry = next_open * (1 + SLIPPAGE_PCT) if trigger_side == "BUY" else next_open * (1 - SLIPPAGE_PCT)

                remaining_bars = day_df.iloc[idx + 1 :]
                exit_price = None
                exit_reason = None

                # Trailing applies to the rule-based variants only: the plain-ORB baseline
                # holds to 15:15 by definition (no stop or target to trail), so it stays a
                # pure entry-side control.
                trail_state = TrailState()
                trail_active = trail_config.enabled and strategy_mode != "plain_orb_1515"
                initial_stop = stop_loss
                remaining_qty = qty
                # (price, qty_closed, reason) -- one leg per exit, plus one per partial.
                legs: list[tuple[float, int, str]] = []

                for r_idx in range(len(remaining_bars)):
                    r_bar = remaining_bars.iloc[r_idx]
                    r_dt = remaining_bars.index[r_idx]
                    r_time = r_dt.strftime("%H:%M")
                    r_high = float(r_bar["High"])
                    r_low = float(r_bar["Low"])

                    if strategy_mode == "plain_orb_1515":
                        if r_time >= "15:15":
                            exit_price = float(r_bar["Open"])
                            exit_reason = "1515_squareoff"
                            break
                    else:
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

                    # No exit this bar, so advance the trail on this bar's CLOSE for the next
                    # bar. Updating AFTER the checks is what keeps it look-ahead free: bar i
                    # is always tested against a stop derived from bars strictly before i.
                    if trail_active:
                        r_atr = atr_series.get(r_dt, 0.0)
                        r_atr = 0.0 if r_atr is None or r_atr != r_atr else float(r_atr)
                        stop_loss, trail_state, partial_qty = update_exit(
                            next_open, trigger_side, stop_loss, float(r_bar["Close"]), r_atr,
                            trail_state, qty=remaining_qty, risk_per_share=raw_risk,
                            config=trail_config,
                        )
                        if partial_qty > 0:
                            # Booked at this bar's close -- an observable market price right
                            # now, so no look-ahead. The remainder keeps trailing.
                            legs.append((float(r_bar["Close"]), partial_qty, "partial_exit"))
                            remaining_qty -= partial_qty

                if exit_price is None and not remaining_bars.empty:
                    exit_price = float(remaining_bars["Close"].iloc[-1])
                    exit_reason = "eod"

                if exit_price is not None:
                    legs.append((exit_price, remaining_qty, exit_reason))

                    # Charges are priced PER LEG: a partial exit is a second sell order, so
                    # brokerage/STT/GST apply again (brokerage is capped per order, not per
                    # position). Folding the legs into one round_trip() would understate cost
                    # and flatter the partial variant.
                    gross_pnl_rs = 0.0
                    charges_rs = 0.0
                    for leg_price, leg_qty, _leg_reason in legs:
                        if leg_qty <= 0:
                            continue
                        fill_exit = leg_price * (1 - SLIPPAGE_PCT) if trigger_side == "BUY" else leg_price * (1 + SLIPPAGE_PCT)
                        gross_pnl_rs += ((fill_exit - fill_entry) if trigger_side == "BUY" else (fill_entry - fill_exit)) * leg_qty
                        charges_rs += round_trip(fill_entry, fill_exit, leg_qty)
                    net_pnl_rs = gross_pnl_rs - charges_rs

                    # R denominator stays the ORIGINAL risk so trailing-ON and trailing-OFF
                    # are scored on the same denominator and remain comparable.
                    r_denom = raw_risk * qty
                    net_r = net_pnl_rs / r_denom if r_denom > 0 else 0.0
                    capital_invested = fill_entry * qty
                    pct_return = (net_pnl_rs / capital_invested * 100) if capital_invested > 0 else 0.0

                    all_trades.append({
                        "symbol": sym, "date": d, "time": time_str, "side": trigger_side,
                        "entry": next_open, "stop": initial_stop, "stop_final": stop_loss,
                        "target": target, "risk_1r": raw_risk,
                        "qty": qty, "exit_price": exit_price, "exit_reason": exit_reason,
                        "legs": legs, "trail_armed": trail_state.armed,
                        "partial_taken": trail_state.partial_taken,
                        "gross_pnl_rs": gross_pnl_rs, "charges_rs": charges_rs, "net_pnl_rs": net_pnl_rs,
                        "net_r": net_r, "pct_return": pct_return,
                        "remaining_bars": remaining_bars, "raw_risk": raw_risk, "next_open": next_open,
                    })

    n_trades = len(all_trades)
    if n_trades > 0:
        r_vals = [t["net_r"] for t in all_trades]
        pct_vals = [t["pct_return"] for t in all_trades]
        wins = [r for r in r_vals if r > 0]
        win_rate = len(wins) / n_trades * 100
        expectancy_r = float(np.mean(r_vals))
        ci_low, ci_high = bootstrap_ci(r_vals)
        mean_pct_return = float(np.mean(pct_vals))
        total_net_r = float(np.sum(r_vals))
    else:
        win_rate, expectancy_r, ci_low, ci_high, mean_pct_return, total_net_r = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "raw_signals": raw_signals_count,
        "cost_rejections": cost_floor_rejections,
        "trades_count": n_trades,
        "win_rate": win_rate,
        "expectancy_r": expectancy_r,
        "ci_95": (ci_low, ci_high),
        "mean_pct_return": mean_pct_return,
        "total_net_r": total_net_r,
        "trades": all_trades,
        "slippages": close_to_open_slippages,
    }


def simulate_random_baseline(executed_trades: list[dict], n_iter: int = 500) -> tuple[float, float, tuple[float, float], float]:
    """Flip direction on exact same entry bars, resolve with same rules."""
    if not executed_trades:
        return 0.0, 0.0, (0.0, 0.0), 0.0

    iteration_means = []
    iteration_winrates = []
    iteration_pcts = []

    for seed in range(n_iter):
        rng = random.Random(seed)
        iter_rs = []
        iter_pcts = []
        for t in executed_trades:
            rnd_side = rng.choice(["BUY", "SELL"])
            next_open = t["next_open"]
            raw_risk = t["raw_risk"]
            qty = t["qty"]
            stop_loss = round(next_open - raw_risk if rnd_side == "BUY" else next_open + raw_risk, 2)
            target = round(next_open + 2.0 * raw_risk if rnd_side == "BUY" else next_open - 2.0 * raw_risk, 2)
            fill_entry = next_open * (1 + SLIPPAGE_PCT) if rnd_side == "BUY" else next_open * (1 - SLIPPAGE_PCT)

            exit_price = None
            exit_reason = None
            remaining_bars = t["remaining_bars"]

            for r_idx in range(len(remaining_bars)):
                r_bar = remaining_bars.iloc[r_idx]
                r_time = remaining_bars.index[r_idx].strftime("%H:%M")
                r_high = float(r_bar["High"])
                r_low = float(r_bar["Low"])

                if rnd_side == "BUY":
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

            if exit_price is not None:
                fill_exit = exit_price * (1 - SLIPPAGE_PCT) if rnd_side == "BUY" else exit_price * (1 + SLIPPAGE_PCT)
                gross_pnl_rs = (fill_exit - fill_entry if rnd_side == "BUY" else fill_entry - fill_exit) * qty
                charges_rs = round_trip(fill_entry, fill_exit, qty)
                net_pnl_rs = gross_pnl_rs - charges_rs

                r_denom = raw_risk * qty
                net_r = net_pnl_rs / r_denom if r_denom > 0 else 0.0
                capital_invested = fill_entry * qty
                pct_return = (net_pnl_rs / capital_invested * 100) if capital_invested > 0 else 0.0
                iter_rs.append(net_r)
                iter_pcts.append(pct_return)

        if iter_rs:
            iteration_means.append(float(np.mean(iter_rs)))
            iteration_winrates.append(len([r for r in iter_rs if r > 0]) / len(iter_rs) * 100)
            iteration_pcts.append(float(np.mean(iter_pcts)))

    mean_exp_r = float(np.mean(iteration_means))
    mean_wr = float(np.mean(iteration_winrates))
    mean_pct = float(np.mean(iteration_pcts))
    ci = (float(np.percentile(iteration_means, 2.5)), float(np.percentile(iteration_means, 97.5)))
    return mean_exp_r, mean_wr, ci, mean_pct


def run_research():
    print("=" * 85, flush=True)
    print("  NIFTY 50 ORB STRATEGY-VIABILITY & BASELINE RESEARCH (FYERS 100-DAY DATA)", flush=True)
    print("=" * 85, flush=True)

    client = FyersClient()
    connected, reason = client.status()
    print(f"Fyers API Status: {connected} ({reason})", flush=True)
    if not connected:
        print("ERROR: Active Fyers session required.", flush=True)
        return

    # 1. Fetch data for all 49 symbols
    print(f"\n[1] Loading 100-day 5m candles for {len(NIFTY_50_SYMBOLS)} Nifty 50 stocks...", flush=True)
    all_data = {}
    precomputed = {}
    t0 = time.time()
    for s in NIFTY_50_SYMBOLS:
        df = get_cached_symbol_data(client, s)
        if not df.empty:
            all_data[s] = df
            precomputed[s] = precompute_indicators(df)
    print(f"    Loaded {len(all_data)} symbols in {time.time() - t0:.1f}s.", flush=True)

    # 2. Chronological Split (70% exploration, 30% held-out)
    all_dates = sorted(list(set.union(*[set(df.index.date) for df in all_data.values()])))
    n_days = len(all_dates)
    split_idx = int(n_days * 0.70)
    explore_days = all_dates[:split_idx]
    heldout_days = all_dates[split_idx:]

    print(f"\n[2] Chronological Split ({n_days} total sessions):", flush=True)
    print(f"    Exploration (70%): {len(explore_days)} sessions ({explore_days[0]} to {explore_days[-1]})", flush=True)
    print(f"    Held-Out Test (30%): {len(heldout_days)} sessions ({heldout_days[0]} to {heldout_days[-1]})", flush=True)

    # 3. Next-Bar Open Slippage Analysis across exploration dataset
    print(f"\n[3] Signal-Close vs Next-Open Slippage Distribution (Exploration Set):", flush=True)
    res_full_explore = run_strategy_simulation(all_data, precomputed, explore_days, strategy_mode="full_rules", stop_variant="breakout_bar")
    slips = res_full_explore["slippages"]
    if slips:
        gaps = [s["gap_pct"] for s in slips]
        print(f"    Total Signals Tracked:          {len(slips)}", flush=True)
        print(f"    Mean Gap (Close to Next Open):  {np.mean(gaps):+.4f}%", flush=True)
        print(f"    Median Gap:                    {np.median(gaps):+.4f}%", flush=True)
        print(f"    Std Deviation:                 {np.std(gaps):.4f}%", flush=True)
        print(f"    Percentile 25%:                {np.percentile(gaps, 25):+.4f}%", flush=True)
        print(f"    Percentile 75%:                {np.percentile(gaps, 75):+.4f}%", flush=True)
        print(f"    Percentile 95%:                {np.percentile(gaps, 95):+.4f}%", flush=True)
        print(f"    Min Gap / Max Gap:             {np.min(gaps):+.4f}% / {np.max(gaps):+.4f}%", flush=True)

    # 4. Baselines Evaluation on Exploration Dataset (70%)
    print(f"\n[4] Exploration Dataset (70%): Baselines & Stop Variants Evaluated:", flush=True)
    print("-" * 105, flush=True)
    print(f"{'Variant / Baseline':<44} {'Trades':>6} {'WinRate':>8} {'Net Exp (R)':>12} {'Bootstrap 95% CI':>20} {'Return/Trade':>13}", flush=True)
    print("-" * 105, flush=True)

    variants_to_try = [
        ("Full Rules (Breakout-Bar Stop)", "full_rules", "breakout_bar"),
        ("Full Rules (Opposite-OR Stop)", "full_rules", "opposite_or"),
        ("Full Rules (OR-Midpoint Stop)", "full_rules", "or_midpoint"),
        ("Baseline (ii): Plain ORB (Hold to 15:15)", "plain_orb_1515", "opposite_or"),
    ]

    explore_results = {}
    for name, mode, s_var in variants_to_try:
        res = run_strategy_simulation(all_data, precomputed, explore_days, strategy_mode=mode, stop_variant=s_var)
        explore_results[name] = res
        ci_str = f"[{res['ci_95'][0]:+.2f}R, {res['ci_95'][1]:+.2f}R]"
        print(f"{name:<44} {res['trades_count']:>6} {res['win_rate']:>7.1f}% {res['expectancy_r']:>+11.2f}R {ci_str:>20} {res['mean_pct_return']:>+12.2f}%", flush=True)

    # Baseline (i): Random Direction on the Same Bars
    trades_for_random = explore_results["Full Rules (Breakout-Bar Stop)"]["trades"]
    rand_exp_r, rand_wr, rand_ci, rand_pct = simulate_random_baseline(trades_for_random, n_iter=500)
    rand_ci_str = f"[{rand_ci[0]:+.2f}R, {rand_ci[1]:+.2f}R]"
    print(f"{'Baseline (i): Random Direction (Same Exits)':<44} {len(trades_for_random):>6} {rand_wr:>7.1f}% {rand_exp_r:>+11.2f}R {rand_ci_str:>20} {rand_pct:>+12.2f}%", flush=True)

    # 5. Held-Out Evaluation (Last 30% Used Strictly Once)
    print(f"\n[5] Held-Out Test Set (Last 30% - Chronological Out-of-Sample, Evaluated Once):", flush=True)
    print("-" * 105, flush=True)
    print(f"{'Variant / Baseline':<44} {'Trades':>6} {'WinRate':>8} {'Net Exp (R)':>12} {'Bootstrap 95% CI':>20} {'Return/Trade':>13}", flush=True)
    print("-" * 105, flush=True)

    heldout_results = {}
    for name, mode, s_var in variants_to_try:
        res_held = run_strategy_simulation(all_data, precomputed, heldout_days, strategy_mode=mode, stop_variant=s_var)
        heldout_results[name] = res_held
        ci_str = f"[{res_held['ci_95'][0]:+.2f}R, {res_held['ci_95'][1]:+.2f}R]"
        print(f"{name:<44} {res_held['trades_count']:>6} {res_held['win_rate']:>7.1f}% {res_held['expectancy_r']:>+11.2f}R {ci_str:>20} {res_held['mean_pct_return']:>+12.2f}%", flush=True)

    # Baseline (i) on held-out trades
    held_trades_for_random = heldout_results["Full Rules (Breakout-Bar Stop)"]["trades"]
    h_rand_exp_r, h_rand_wr, h_rand_ci, h_rand_pct = simulate_random_baseline(held_trades_for_random, n_iter=500)
    h_rand_ci_str = f"[{h_rand_ci[0]:+.2f}R, {h_rand_ci[1]:+.2f}R]"
    print(f"{'Baseline (i): Random Direction (Same Exits)':<44} {len(held_trades_for_random):>6} {h_rand_wr:>7.1f}% {h_rand_exp_r:>+11.2f}R {h_rand_ci_str:>20} {h_rand_pct:>+12.2f}%", flush=True)

    # 6. Trailing-stop grid: does a trailing + partial-booking exit change the result?
    #    Same harness, same 70/30 split, same seeded bootstrap. The OFF arm is the
    #    control and must reproduce sections [4]/[5] exactly -- if it does not, the
    #    wiring changed behaviour and nothing below is trustworthy.
    print("\n[6] Trailing-Stop Grid (Part 1 of the exit study):", flush=True)
    print("    Same harness, same 70/30 split, same seeded bootstrap. OFF = control arm.", flush=True)

    trail_grid: list[tuple[str, TrailConfig]] = [
        ("Trail OFF (control)", TrailConfig(enabled=False)),
    ]
    for _mult in (1.5, 2.0, 2.5):
        trail_grid.append((f"Trail ON  ATRx{_mult}  no-partial",
                           TrailConfig(enabled=True, trail_atr_mult=_mult, partial_at_r=None)))
        trail_grid.append((f"Trail ON  ATRx{_mult}  partial@1.5R",
                           TrailConfig(enabled=True, trail_atr_mult=_mult, partial_at_r=1.5)))

    grid_rows = []
    for label, tcfg in trail_grid:
        for set_name, days in (("explore", explore_days), ("heldout", heldout_days)):
            for name, mode, s_var in variants_to_try:
                res = run_strategy_simulation(
                    all_data, precomputed, days, strategy_mode=mode,
                    stop_variant=s_var, trail_config=tcfg,
                )
                grid_rows.append({
                    "config": label, "split": set_name, "variant": name,
                    "trades": res["trades_count"], "win_rate": res["win_rate"],
                    "expectancy_r": res["expectancy_r"],
                    "ci_low": res["ci_95"][0], "ci_high": res["ci_95"][1],
                    "mean_pct_return": res["mean_pct_return"],
                    "total_net_r": res["total_net_r"],
                })

    # Print one block per split so the table reads top-to-bottom as a comparison.
    for set_name in ("explore", "heldout"):
        print("-" * 105, flush=True)
        header = "Exploration (70%)" if set_name == "explore" else "Held-Out (30%, out-of-sample)"
        print(f"    {header}", flush=True)
        print(f"{'Config':<34} {'Variant':<36} {'Trades':>6} {'WinRate':>8} {'Net Exp (R)':>12} {'95% CI':>20}", flush=True)
        for row in grid_rows:
            if row["split"] != set_name:
                continue
            ci_str = f"[{row['ci_low']:+.2f}R, {row['ci_high']:+.2f}R]"
            print(f"{row['config']:<34} {row['variant'][:35]:<36} {row['trades']:>6} "
                  f"{row['win_rate']:>7.1f}% {row['expectancy_r']:>+11.2f}R {ci_str:>20}", flush=True)

    # Full-rules aggregate per config, which is the headline comparison.
    print("-" * 105, flush=True)
    print("    Full-Rules variants pooled, per config (the headline):", flush=True)
    print(f"{'Config':<34} {'Split':<9} {'Trades':>6} {'WinRate':>8} {'Net Exp (R)':>12}", flush=True)
    for set_name in ("explore", "heldout"):
        for label, _tcfg in trail_grid:
            rows = [r for r in grid_rows if r["split"] == set_name
                    and r["config"] == label and r["variant"].startswith("Full Rules")]
            trades = sum(r["trades"] for r in rows)
            if trades == 0:
                continue
            exp_r = sum(r["expectancy_r"] * r["trades"] for r in rows) / trades
            wr = sum(r["win_rate"] * r["trades"] for r in rows) / trades
            print(f"{label:<34} {set_name:<9} {trades:>6} {wr:>7.1f}% {exp_r:>+11.2f}R", flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"trail_grid_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"grid": grid_rows, "splits": {
            "explore_sessions": len(explore_days), "heldout_sessions": len(heldout_days),
            "explore_range": [str(explore_days[0]), str(explore_days[-1])],
            "heldout_range": [str(heldout_days[0]), str(heldout_days[-1])],
        }}, fh, indent=2)
    print(f"\n    Grid written to {out_path}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("  RESEARCH RUN COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    run_research()
