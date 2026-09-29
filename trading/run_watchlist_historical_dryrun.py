"""Historical dry-run simulation across the longest available history (60 days) for all 6 watchlist symbols.

Evaluates:
- ORB + Multi-TF 200MA strategy with cost-floor gate (friction <= 30% of risk).
- Sequential hypothetical shadow executions with slippage, Zerodha charges, and 15:15 bar open square-off.
- Per-symbol metrics: signals, win rate, gross P&L, charges, net shadow P&L.
- Data limits and cost-floor filter audit.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

from trading.config import WATCHLIST, YF_SUFFIX, SLIPPAGE_PCT, CHARGES_PCT_ROUND_TRIP, MAX_COST_RISK_RATIO
from trading.strategies.orb_ma200 import ORBMA200Strategy
from trading.shadow import resolve_candidate_from_bars

IST = ZoneInfo("Asia/Kolkata")


def run_watchlist_simulation():
    print("=" * 75, flush=True)
    print("  WATCHLIST 60-DAY HISTORICAL DRY RUN (ORB + 200MA + COST-FLOOR GATE)", flush=True)
    print("=" * 75, flush=True)

    overall_results = []
    signals_filtered_by_cost_floor = 0
    total_raw_signals = 0

    for symbol in WATCHLIST:
        yf_sym = symbol + YF_SUFFIX
        print(f"\n[+] Fetching 60d data for {symbol} ({yf_sym})...", flush=True)
        t = yf.Ticker(yf_sym)
        df = t.history(period="60d", interval="5m")
        if df.empty:
            print(f"    [!] No data returned for {symbol}", flush=True)
            continue

        df = df.tz_convert(IST)
        n_bars = len(df)
        days = sorted(list(set(df.index.date)))
        print(f"    Available: {n_bars} bars across {len(days)} sessions ({days[0]} to {days[-1]})", flush=True)

        # Precompute HTF 200 MAs for speed
        df_1h = df.resample("60min", offset="15min").agg({"Close": "last"}).dropna()
        ma1h_series = df_1h["Close"].rolling(200).mean().reindex(df.index, method="ffill")

        df_30m = df.resample("30min", offset="15min").agg({"Close": "last"}).dropna()
        ma30m_series = df_30m["Close"].rolling(200).mean().reindex(df.index, method="ffill")

        # Track signals for this symbol
        symbol_trades = []
        sym_strat = ORBMA200Strategy(max_cost_risk_ratio=MAX_COST_RISK_RATIO)
        sym_strat_raw = ORBMA200Strategy(max_cost_risk_ratio=10.0)

        # Hook precomputed HTF MAs into strategy instance
        def make_htf_getter(ma1h, ma30m):
            def _get_htf(sub_df, rule):
                ts = sub_df.index[-1]
                if rule == "60min":
                    val = ma1h.get(ts)
                    return float(val) if val is not None and not pd.isna(val) else None
                if rule == "30min":
                    val = ma30m.get(ts)
                    return float(val) if val is not None and not pd.isna(val) else None
                return None
            return _get_htf

        sym_strat._htf_ma = make_htf_getter(ma1h_series, ma30m_series)
        sym_strat_raw._htf_ma = make_htf_getter(ma1h_series, ma30m_series)

        # Run bar by bar across each day
        for d in days:
            day_mask = df.index.date == d
            day_indices = [i for i, val in enumerate(day_mask) if val]
            if len(day_indices) < 5:
                continue

            for idx in day_indices:
                bar_dt = df.index[idx]
                time_str = bar_dt.strftime("%H:%M")
                if time_str < "09:30" or time_str > "14:30":
                    continue

                sub_df = df.iloc[: idx + 1]
                # Check with cost-floor gate
                sig = sym_strat.evaluate(sub_df, symbol)
                # Check without cost-floor gate to measure filter effect
                sig_raw = sym_strat_raw.evaluate(sub_df, symbol)

                if sig_raw is not None:
                    total_raw_signals += 1
                    if sig is None:
                        signals_filtered_by_cost_floor += 1

                if sig is not None:
                    cand = {
                        "id": len(symbol_trades) + 1,
                        "symbol": symbol,
                        "side": sig.side,
                        "price": sig.price,
                        "stop_loss": sig.stop_loss,
                        "target": sig.target,
                        "ts": bar_dt.timestamp(),
                        "date": bar_dt.strftime("%Y-%m-%d"),
                    }
                    # Resolve trade over remaining bars of that day
                    day_df = df[df.index.date == d]
                    res = resolve_candidate_from_bars(cand, day_df)
                    if res:
                        symbol_trades.append({
                            "symbol": symbol,
                            "date": cand["date"],
                            "time": time_str,
                            "side": sig.side,
                            "entry": sig.price,
                            "stop": sig.stop_loss,
                            "target": sig.target,
                            "risk": round(abs(sig.price - sig.stop_loss), 2),
                            "exit_price": res["exit_price"],
                            "exit_reason": res["exit_reason"],
                            "gross_pnl": res["gross_pnl"],
                            "charges": res["charges"],
                            "shadow_pnl": res["shadow_pnl"],
                        })

        print(f"    Signals triggered: {len(symbol_trades)}", flush=True)
        for tr in symbol_trades:
            print(f"      {tr['date']} {tr['time']} | {tr['side']} @ {tr['entry']:.2f} (SL: {tr['stop']:.2f}, Tgt: {tr['target']:.2f}) -> {tr['exit_reason']} @ {tr['exit_price']:.2f} | Net: Rs {tr['shadow_pnl']:+.2f}", flush=True)
        overall_results.extend(symbol_trades)

    # Print Aggregate Performance Summary
    print("\n" + "=" * 75, flush=True)
    print("  AGGREGATE PERFORMANCE & COST-FLOOR REPORT", flush=True)
    print("=" * 75, flush=True)
    print(f"Total raw quant breakout signals (before cost-floor): {total_raw_signals}", flush=True)
    print(f"Signals removed by cost-floor gate (friction > 30% risk): {signals_filtered_by_cost_floor}", flush=True)
    if total_raw_signals > 0:
        print(f"Cost-floor rejection rate: {signals_filtered_by_cost_floor / total_raw_signals * 100:.1f}%", flush=True)
    print(f"Final approved signals executed: {len(overall_results)}", flush=True)

    if overall_results:
        df_res = pd.DataFrame(overall_results)
        wins = df_res[df_res["shadow_pnl"] > 0]
        losses = df_res[df_res["shadow_pnl"] <= 0]
        win_rate = len(wins) / len(df_res) * 100
        gross_pnl = df_res["gross_pnl"].sum()
        total_charges = df_res["charges"].sum()
        net_pnl = df_res["shadow_pnl"].sum()

        print("\nBreakdown by Symbol:", flush=True)
        for sym in WATCHLIST:
            sym_df = df_res[df_res["symbol"] == sym]
            if not sym_df.empty:
                s_wins = len(sym_df[sym_df["shadow_pnl"] > 0])
                s_wr = s_wins / len(sym_df) * 100
                print(f"  {sym:<10}: {len(sym_df):>2} trades | WR: {s_wr:>5.1f}% | Gross: Rs {sym_df['gross_pnl'].sum():>+8.2f} | Charges: Rs {sym_df['charges'].sum():>6.2f} | Net: Rs {sym_df['shadow_pnl'].sum():>+8.2f}", flush=True)
            else:
                print(f"  {sym:<10}:  0 trades", flush=True)

        print("\nPortfolio Totals (Per 1-Share Shadow Unit):", flush=True)
        print(f"  Total Trades:       {len(df_res)}", flush=True)
        print(f"  Winning Trades:     {len(wins)}", flush=True)
        print(f"  Losing Trades:      {len(losses)}", flush=True)
        print(f"  Win Rate:           {win_rate:.1f}%", flush=True)
        print(f"  Total Gross P&L:    Rs {gross_pnl:+.2f}", flush=True)
        print(f"  Total Charges:      Rs {total_charges:.2f}", flush=True)
        print(f"  Total Net P&L:      Rs {net_pnl:+.2f}", flush=True)
        print(f"  Avg Net / Trade:    Rs {net_pnl / len(df_res):+.2f}", flush=True)
    else:
        print("No trades triggered over the 60-day period with the current rule thresholds.", flush=True)


if __name__ == "__main__":
    run_watchlist_simulation()
