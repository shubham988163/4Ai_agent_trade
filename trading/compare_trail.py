"""Compare trailing-stop configurations on the same historical candle frames.

Evaluates three configurations:
1. Trail OFF (baseline control arm)
2. Trail ON (Breakeven @ 1.0R + ATR chandelier @ 2.0x + 50% partial @ 1.5R)
3. Trail ON (No partials: Breakeven @ 1.0R + ATR chandelier @ 2.0x, partial_at_r=None)

Features:
- Unpaired analysis: reports per-position expectancy and 95% CI (partial fills aggregated to parent).
- Paired comparison: matches positions by (symbol, entry_ts), computes net difference (ON - OFF),
  paired 95% CI, matched vs unmatched counts, and total net difference.
- Exploratory sweep (--sweep): grid across ATR multipliers (1.5, 2.0, 3.0) and Breakeven thresholds
  (0.75, 1.0, 1.5R), with session/bar coverage and explicit data-mining warnings.
"""
from __future__ import annotations

import argparse
import math
import sys
from trading import strategy as strat_mod
from trading.backtest import Backtester, build_strategy, load_history, metrics, Result
from trading.config import SCAN_UNIVERSE, CANDLE_INTERVAL
from trading.exits import TrailConfig


def extract_positions(trades_or_result) -> dict[tuple[str, object], float]:
    """Map (symbol, entry_ts) -> aggregated net P&L across all partial and final fills."""
    ts = getattr(trades_or_result, "closed", trades_or_result)
    pos: dict[tuple[str, object], float] = {}
    for t in ts:
        if isinstance(t, dict):
            key = (t["symbol"], t["entry_ts"])
            net = t["net"]
        else:
            key = (t.symbol, t.entry_ts)
            net = t.net
        pos[key] = pos.get(key, 0.0) + net
    return pos


def compute_paired_diffs(pos_off: dict, pos_on: dict) -> dict:
    """Pair positions on (symbol, entry_ts) and calculate paired statistics (ON - OFF),
    including clustered session standard errors and day-level paired differences."""
    matched_keys = sorted(set(pos_off.keys()) & set(pos_on.keys()))
    unmatched_off = set(pos_off.keys()) - set(pos_on.keys())
    unmatched_on = set(pos_on.keys()) - set(pos_off.keys())

    n_matched = len(matched_keys)
    if n_matched == 0:
        return {
            "n_matched": 0,
            "unmatched_off": len(unmatched_off),
            "unmatched_on": len(unmatched_on),
            "mean_diff": 0.0,
            "ci95": 0.0,
            "lo": 0.0,
            "hi": 0.0,
            "ci95_clustered": 0.0,
            "day_mean_diff": 0.0,
            "day_ci95": 0.0,
            "n_days": 0,
            "total_diff": 0.0,
            "diffs": [],
        }

    diffs = [pos_on[k] - pos_off[k] for k in matched_keys]
    total_diff = sum(diffs)
    mean_diff = total_diff / n_matched
    if n_matched > 1:
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n_matched - 1)
        sd_diff = math.sqrt(var_diff)
        se_diff = sd_diff / math.sqrt(n_matched)
        ci95 = 1.96 * se_diff
    else:
        sd_diff = 0.0
        ci95 = 0.0

    # Cluster sensitivity: group differences by session date (YYYY-MM-DD)
    day_diffs: dict[str, list[float]] = {}
    for k in matched_keys:
        ts_val = k[1]
        if hasattr(ts_val, "strftime"):
            d_str = ts_val.strftime("%Y-%m-%d")
        else:
            d_str = str(ts_val)
        diff = pos_on[k] - pos_off[k]
        day_diffs.setdefault(d_str, []).append(diff)

    n_days = len(day_diffs)
    day_totals = [sum(v) for v in day_diffs.values()]
    day_mean = sum(day_totals) / n_days if n_days > 0 else 0.0
    if n_days > 1:
        var_day = sum((dt - day_mean) ** 2 for dt in day_totals) / (n_days - 1)
        se_day = math.sqrt(var_day) / math.sqrt(n_days)
        day_ci95 = 1.96 * se_day

        # Cluster-robust standard error on per-trade mean (Liang-Zeger)
        u_sq_sum = sum((sum(v) - len(v) * mean_diff) ** 2 for v in day_diffs.values())
        var_clustered = (n_days / (n_days - 1)) * (u_sq_sum / (n_matched ** 2))
        se_clustered = math.sqrt(var_clustered)
        ci95_clustered = 1.96 * se_clustered
    else:
        day_ci95 = 0.0
        ci95_clustered = ci95

    return {
        "n_matched": n_matched,
        "unmatched_off": len(unmatched_off),
        "unmatched_on": len(unmatched_on),
        "mean_diff": mean_diff,
        "ci95": ci95,
        "lo": mean_diff - ci95,
        "hi": mean_diff + ci95,
        "ci95_clustered": ci95_clustered,
        "day_mean_diff": day_mean,
        "day_ci95": day_ci95,
        "n_days": n_days,
        "total_diff": total_diff,
        "diffs": diffs,
    }


def run_sweep(frames: dict, strat_dict: dict, res_off: Result):
    """Exploratory sensitivity grid across ATR mults and breakeven thresholds."""
    atr_mults = (1.5, 2.0, 3.0)
    be_rs = (0.75, 1.0, 1.5)
    pos_off = extract_positions(res_off)

    total_bars = sum(len(df) for df in frames.values())
    total_sessions = len(set.union(*(set(df.index.date) for df in frames.values())))

    print("\n" + "=" * 115)
    print("  EXPLORATORY ROBUSTNESS GRID: TRAILING-STOP SENSITIVITY (PARTIALS OFF)")
    print(f"  Coverage: {len(frames)} symbols · {total_sessions} sessions · {total_bars:,} total bars")
    print("  *CAUTION*: Exploratory data sweep. Testing multiple parameters increases false-positive risk.")
    print("             Any candidate must be strictly validated out-of-sample on unseen market periods.")
    print("=" * 115)
    print(f"{'BE (R)':>8} {'ATR Mult':>10} {'Positions':>10} {'WinRate':>9} {'Matched':>8} {'Total Net Diff':>16} {'Paired Mean Diff':>18} {'Paired 95% CI':>20}")
    print("-" * 115)

    for be_r in be_rs:
        for mult in atr_mults:
            t_cfg = TrailConfig(enabled=True, breakeven_at_r=be_r, trail_atr_mult=mult, partial_at_r=None)
            bt = Backtester(strat_dict, trail_config=t_cfg)
            res = bt.run(frames, label=f"BE={be_r}R ATR={mult}x")
            m = metrics(res)
            pos_on = extract_positions(res)
            paired = compute_paired_diffs(pos_off, pos_on)

            pos_n = m["pos_n"]
            wr = m.get("pos_win_rate", m["win_rate"]) * 100
            m_cnt = paired["n_matched"]
            tot_d = paired["total_diff"]
            mean_d = paired["mean_diff"]
            ci = paired["ci95"]
            ci_str = f"[{mean_d - ci:+.2f}, {mean_d + ci:+.2f}]"

            print(f"{be_r:>8.2f} {mult:>10.1f} {pos_n:>10} {wr:>8.1f}% {m_cnt:>8} {tot_d:>15,.2f} {mean_d:>14,.2f} +/- {ci:>3.2f} {ci_str:>20}")

    print("=" * 115 + "\n")


def compute_r_expectancy(res: Result) -> dict:
    """Compute per-position expectancy in R-multiples (net / initial_risk_inr)."""
    closed = getattr(res, "closed", res.trades)
    r_multiples = []
    for t in closed:
        if t.exit is None:
            continue
        init_sl = t.initial_sl if t.initial_sl is not None else t.sl
        risk_per_share = abs(t.entry - init_sl)
        risk_inr = t.qty * risk_per_share
        if risk_inr > 0:
            r_multiples.append(t.net / risk_inr)

    n = len(r_multiples)
    if n == 0:
        return {"mean_r": 0.0, "ci95_r": 0.0, "n": 0, "total_r": 0.0}

    mean_r = sum(r_multiples) / n
    if n > 1:
        var_r = sum((r - mean_r) ** 2 for r in r_multiples) / (n - 1)
        se_r = math.sqrt(var_r) / math.sqrt(n)
        ci95_r = 1.96 * se_r
    else:
        ci95_r = 0.0

    return {
        "mean_r": mean_r,
        "ci95_r": ci95_r,
        "n": n,
        "total_r": sum(r_multiples),
    }


def run_comparison(
    symbols: list[str] | None = None,
    strategy: str = "ema",
    interval: str = CANDLE_INTERVAL,
    period: str = "60d",
    refresh: bool = False,
    sweep: bool = False,
):
    # Strategy interval adaptation
    if strategy == "orb_ma200" and interval == CANDLE_INTERVAL:
        interval = "5m"

    syms = symbols or SCAN_UNIVERSE
    print(f"Loading candle frames for {len(syms)} symbols ({interval}, {period})...", flush=True)
    frames = load_history(syms, interval=interval, period=period, refresh=refresh)
    if not frames:
        print("Error: No candle frames loaded.", file=sys.stderr)
        return

    loaded_symbols = sorted(list(frames.keys()))
    failed_symbols = sorted([s for s in syms if s not in frames])

    strat_dict = build_strategy(strategy)
    configs = [
        ("Trail OFF (Control)", TrailConfig(enabled=False)),
        ("Trail ON (Full: BE + Chandelier + Partial)", TrailConfig(enabled=True, breakeven_at_r=1.0, trail_atr_mult=2.0, partial_at_r=1.5, partial_pct=0.5)),
        ("Trail ON (No Partials: BE + Chandelier)", TrailConfig(enabled=True, breakeven_at_r=1.0, trail_atr_mult=2.0, partial_at_r=None)),
    ]

    total_bars = sum(len(df) for df in frames.values())
    total_sessions = len(set.union(*(set(df.index.date) for df in frames.values())))

    print("\n" + "=" * 115)
    print(f"  TRAILING-STOP UNPAIRED COMPARISON: {strategy.upper()} on {len(frames)} symbols ({interval}, {period})")
    print(f"  Sessions: {total_sessions} · Total Bars: {total_bars:,}")
    print(f"  Symbols Loaded ({len(loaded_symbols)}): {', '.join(loaded_symbols)}")
    if failed_symbols:
        print(f"  Symbols Failed ({len(failed_symbols)}): {', '.join(failed_symbols)}")
    print("=" * 115)
    print(f"{'Configuration':<44} {'Positions':>9} {'Fills':>6} {'WinRate':>8} {'Net P&L':>11} {'Per-Pos Expectancy':>18} {'95% CI':>16}")
    print("-" * 115)

    results = []
    run_objects = []
    for label, t_cfg in configs:
        # Reset strategy adapter tracking if present
        adapter = strat_dict.get("adapter")
        if adapter and hasattr(adapter, "reset"):
            adapter.reset()

        bt = Backtester(strat_dict, trail_config=t_cfg)
        res = bt.run(frames, label=label)
        m = metrics(res)
        results.append((label, m))
        run_objects.append((label, res))

        pos_n = m.get("pos_n", 0)
        fills_n = m.get("n", 0)
        wr = m.get("pos_win_rate", m.get("win_rate", 0.0)) * 100
        net = m.get("net", 0.0)
        exp = m.get("pos_expectancy", m.get("expectancy", 0.0))
        ci = m.get("pos_ci95", m.get("ci95", 0.0))
        ci_str = f"[Rs {exp - ci:+.2f}, {exp + ci:+.2f}]"

        print(f"{label:<44} {pos_n:>9} {fills_n:>6} {wr:>7.1f}% {net:>11,.2f} {exp:>12,.2f} +/- {ci:>4.2f} {ci_str:>16}")

    print("=" * 115)

    # Paired comparison
    res_off = run_objects[0][1]
    pos_off = extract_positions(res_off)

    print("\n" + "=" * 115)
    print("  PAIRED COMPARISON vs. TRAIL OFF CONTROL (MATCHED ON SYMBOL + ENTRY TIME)")
    print("=" * 115)
    print(f"{'Comparison Arm vs. OFF':<38} {'Matched':>7} {'Unm(OFF)':>8} {'Unm(ON)':>7} {'Net Diff':>12} {'Mean Diff':>13} {'Paired 95% CI':>18} {'Clustered 95% CI':>20}")
    print("-" * 115)

    for label, res_on in run_objects[1:]:
        pos_on = extract_positions(res_on)
        p = compute_paired_diffs(pos_off, pos_on)
        m_cnt = p["n_matched"]
        u_off = p["unmatched_off"]
        u_on = p["unmatched_on"]
        tot_d = p["total_diff"]
        mean_d = p["mean_diff"]
        ci = p["ci95"]
        ci_cl = p["ci95_clustered"]
        ci_str = f"[{mean_d - ci:+.2f}, {mean_d + ci:+.2f}]"
        ci_cl_str = f"[{mean_d - ci_cl:+.2f}, {mean_d + ci_cl:+.2f}]"
        print(f"{label:<38} {m_cnt:>7} {u_off:>8} {u_on:>7} {tot_d:>12,.2f} {mean_d:>13,.2f} {ci_str:>18} {ci_cl_str:>20}")

    print("-" * 115)

    # Session-level sensitivity report
    print("\n  SESSION-LEVEL (DAY) SENSITIVITY CHECK:")
    for label, res_on in run_objects[1:]:
        pos_on = extract_positions(res_on)
        p = compute_paired_diffs(pos_off, pos_on)
        if p["n_days"] > 0:
            d_mean = p["day_mean_diff"]
            d_ci = p["day_ci95"]
            print(f"  * {label}:")
            print(f"    Active Sessions with Matched Trades: {p['n_days']}")
            print(f"    Day-Level Net Mean Difference: Rs {d_mean:+.2f} +/- {d_ci:.2f} [Rs {d_mean - d_ci:+.2f}, {d_mean + d_ci:+.2f}]")
        else:
            print(f"  * {label}: No matched trades across sessions.")

    # Control Arm Expectancy in R
    r_off = compute_r_expectancy(res_off)
    print("\n" + "=" * 115)
    print("  OFF ARM EXPECTANCY IN R-MULTIPLES (NET P&L / INITIAL RISK)")
    print("=" * 115)
    print(f"  Positions Evaluated:   {r_off['n']}")
    print(f"  Total R Realised:      {r_off['total_r']:+.2f}R")
    print(f"  Net Expectancy / Pos:  {r_off['mean_r']:+.2f}R +/- {r_off['ci95_r']:.2f}R "
          f"[{r_off['mean_r'] - r_off['ci95_r']:+.2f}R, {r_off['mean_r'] + r_off['ci95_r']:+.2f}R]")
    print("  Reference Benchmark:   Known research figures: -0.23R / -0.54R")
    print("  *Dataset Notice*:      The backtest (yfinance, ~60d of 5m) and the research harness (Fyers, 100d)")
    print("                         are distinct datasets with different session spans, candle constructions,")
    print("                         and vendor feeds. Figures are not expected to match exactly.")
    print("=" * 115)

    # Sample size audit
    min_pos = min(m["pos_n"] for _, m in results)
    if min_pos < 100:
        print("\n  [SAMPLE SIZE NOTICE]:")
        print(f"  Arm position count ({min_pos}) is fewer than ~100 positions.")
        print("  The sample size is too small to draw statistically reliable conclusions.")
        print("  Confidence intervals should NOT be interpreted as conclusive evidence.")
        print("=" * 115 + "\n")

    # Robustness sweep if requested
    if sweep:
        run_sweep(frames, strat_dict, res_off)

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare trailing stop configurations")
    p.add_argument("--strategy", default="ema", help="Strategy to run (default: ema)")
    p.add_argument("--interval", default=CANDLE_INTERVAL, help="Candle interval (default: from config)")
    p.add_argument("--period", default="60d", help="History period (default: 60d)")
    p.add_argument("--refresh", action="store_true", help="Force re-download of candle frames")
    p.add_argument("--sweep", action="store_true", help="Run exploratory sensitivity grid across BE and ATR parameters")
    p.add_argument("--symbols", default=None, help="Comma-separated symbols")
    args = p.parse_args()

    sym_list = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    run_comparison(
        symbols=sym_list,
        strategy=args.strategy,
        interval=args.interval,
        period=args.period,
        refresh=args.refresh,
        sweep=args.sweep,
    )
