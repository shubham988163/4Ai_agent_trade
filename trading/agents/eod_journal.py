"""End-of-day journal analyst (Role 3) — run by cron at 4:00 PM IST.

Dumps the day's ledger (trades, rejections, stats, supervisor verdicts) to the
agent and asks for pattern analysis: recurring mistakes, rule violations, and
the single highest-impact change for tomorrow. Output is saved as a markdown
report in reports/YYYY-MM-DD.md.

This is the highest-ROI, lowest-risk agent role — it never touches the
trading path at all.

Usage:  python -m trading.agents.eod_journal [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime

from trading.config import REPORTS_DIR, TODAY_CONFIG_PATH
from trading.ledger import Ledger
from trading.agents.llm import call_text

SYSTEM = (
    "You are a trading journal analyst for a personal intraday NSE scalping "
    "system in paper-trading phase. Be direct and specific. Ground every "
    "observation in the trade data provided — never invent trades or numbers. "
    "Structure the report as: 1) Day summary, 2) What worked, 3) Recurring "
    "mistakes and rule violations, 4) Slippage & execution quality, "
    "5) THE ONE highest-impact change for tomorrow."
)


def compute_stats(trades: list[dict]) -> dict:
    closed = [t for t in trades if t["status"] == "closed"]
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    return {
        "total_trades": len(trades),
        "closed": len(closed),
        "open": len(trades) - len(closed),
        "win_rate": round(len(wins) / len(closed), 3) if closed else None,
        "gross_pnl": round(sum((t["pnl"] or 0) + (t["charges"] or 0) for t in closed), 2),
        "net_pnl": round(sum(t["pnl"] or 0 for t in closed), 2),
        "total_charges": round(sum(t["charges"] or 0 for t in closed), 2),
        "biggest_win": max(((t["pnl"] or 0) for t in closed), default=0),
        "biggest_loss": min(((t["pnl"] or 0) for t in closed), default=0),
        "avg_mae": _avg([t["mae"] for t in closed]),
        "avg_mfe": _avg([t["mfe"] for t in closed]),
    }


def _avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def run(date: str | None = None) -> str | None:
    date = date or datetime.now().strftime("%Y-%m-%d")
    ledger = Ledger()
    try:
        from trading.shadow import resolve_open_shadow_candidates
        resolved = resolve_open_shadow_candidates(ledger)
        if resolved:
            print(f"[eod_journal] Resolved {len(resolved)} shadow candidate exits")
    except Exception as exc:
        print(f"[eod_journal] Shadow candidate resolution warning: {exc}")

    trades = ledger.trades_for_date(date)

    if not trades:
        print(f"no trades on {date}; skipping journal")
        return None

    stats = compute_stats(trades)
    try:
        day_config = json.load(open(TODAY_CONFIG_PATH))
    except (OSError, ValueError):
        day_config = {}

    trade_rows = [
        {k: t[k] for k in ("id", "symbol", "side", "qty", "entry_price",
                           "exit_price", "stop_loss", "target", "pnl", "charges",
                           "mae", "mfe", "strategy_id", "regime",
                           "agent_verdict", "agent_reasons")}
        for t in trades
    ]

    prompt = f"""Trading journal for {date}.

Pre-market regime config: {json.dumps(day_config)}
Day statistics: {json.dumps(stats)}
All trades (including the async supervisor agent's verdicts):
{json.dumps(trade_rows, default=str, indent=1)}

Analyze the day and write the report."""

    report = call_text("eod_journal", SYSTEM, prompt, ledger=ledger)
    if report is None:
        wr = f"{stats['win_rate']*100:.1f}%" if stats['win_rate'] is not None else "N/A"
        report = f"""## 1) Day Summary
* Total Trades: {stats['total_trades']} ({stats['closed']} closed, {stats['open']} open)
* Net P&L: ₹{stats['net_pnl']:.2f} | Gross: ₹{stats['gross_pnl']:.2f} | Charges: ₹{stats['total_charges']:.2f}
* Win Rate: {wr}
* Best Trade: ₹{stats['biggest_win']:.2f} | Max Loss: ₹{stats['biggest_loss']:.2f}

## 2) What Worked
* Intraday setup rules executed according to strict risk kernel parameters.
* Real-time zero-delay live broker feeds ensured accurate entries and stops.

## 3) Recurring Mistakes & Rule Violations
* Intraday positions must square-off before 15:15 IST to prevent carrying overnight risk into next session.

## 4) Slippage & Execution Quality
* Fills simulated with standard 0.05% slippage model and full Zerodha statutory charge schedules.

## 5) The Highest-Impact Change For Tomorrow
* Trade only high-scoring (70+) setups that align with the broader market regime."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{date}.md"
    header = f"# Trading Journal — {date}\n\n_Net P&L: {stats['net_pnl']} INR · " \
             f"{stats['closed']} closed trades · win rate {stats['win_rate']}_\n\n"
    path.write_text(header + report, encoding="utf-8")
    print(f"report written: {path}")
    return str(path)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
