"""Keep only the last N trading sessions of data on disk.

The dashboard is a working screen, not an archive: five sessions is what you
actually read, and an unbounded ledger makes every query and page load slower
for data nobody opens.

**This deletes.** Pruning is irreversible, so it is deliberate about scope:

  * trades are kept by SESSION, not by calendar day — counting calendar days
    would silently eat a session across a long weekend or an exchange holiday;
  * rejections and the LLM audit log are cut at the timestamp of the oldest
    session kept, so a row is never orphaned from the trades it explains;
  * the report and candle-cache files follow the same window.

Run it:

    .venv/bin/python -m trading.retention --dry-run      # show what would go
    .venv/bin/python -m trading.retention                # prune to RETENTION_DAYS
    .venv/bin/python -m trading.retention --keep-days 10
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from trading.config import DB_PATH, PROJECT_ROOT, REPORTS_DIR, RETENTION_DAYS

IST = ZoneInfo("Asia/Kolkata")
CANDLE_DIR = PROJECT_ROOT / "data" / "candles"


def sessions_in_ledger(conn: sqlite3.Connection) -> list[str]:
    """Distinct trade dates present, newest first."""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM trades WHERE date IS NOT NULL ORDER BY date DESC")]


def plan(keep_days: int = RETENTION_DAYS) -> dict:
    """What a prune would remove — computed without changing anything."""
    if keep_days < 1:
        raise ValueError("keep_days must be at least 1")
    out: dict = {"keep_days": keep_days, "cutoff_date": None, "cutoff_ts": None,
                 "trades": 0, "rejections": 0, "agent_log": 0,
                 "reports": [], "candles": [], "kept_sessions": []}
    if not DB_PATH.exists():
        return out

    conn = sqlite3.connect(DB_PATH)
    try:
        sessions = sessions_in_ledger(conn)
        keep = sessions[:keep_days]
        out["kept_sessions"] = list(reversed(keep))
        if len(sessions) > keep_days:
            cutoff = keep[-1]
            out["cutoff_date"] = cutoff
            # Oldest timestamp inside the oldest session we keep — the boundary
            # for tables that only carry a timestamp.
            row = conn.execute("SELECT MIN(ts) FROM trades WHERE date = ?",
                               (cutoff,)).fetchone()
            cutoff_ts = row[0] if row and row[0] is not None else None
            out["cutoff_ts"] = cutoff_ts
            out["trades"] = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE date < ?", (cutoff,)).fetchone()[0]
            if cutoff_ts is not None:
                for table in ("rejections", "agent_log"):
                    out[table] = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE ts < ?",
                        (cutoff_ts,)).fetchone()[0]
    finally:
        conn.close()

    # Files follow the same window. Reports are dated by filename; the candle
    # cache is unnamed by date, so it goes by modification time.
    if REPORTS_DIR.exists():
        reports = sorted(REPORTS_DIR.glob("*.md"))
        out["reports"] = [p for p in reports[:-keep_days]] if len(reports) > keep_days else []
    # Deliberately exclude data/candles/ from pruning so the backtest cache
    # can accumulate historical candles beyond yfinance's ~60-day window.
    out["candles"] = []
    return out


def prune(keep_days: int = RETENTION_DAYS, *, dry_run: bool = False) -> dict:
    """Delete everything outside the window. Returns the same shape as plan()."""
    p = plan(keep_days)
    if dry_run:
        return p

    if p["cutoff_date"] and DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("DELETE FROM trades WHERE date < ?", (p["cutoff_date"],))
            if p["cutoff_ts"] is not None:
                conn.execute("DELETE FROM rejections WHERE ts < ?", (p["cutoff_ts"],))
                conn.execute("DELETE FROM agent_log WHERE ts < ?", (p["cutoff_ts"],))
            conn.commit()
            conn.execute("VACUUM")          # actually give the space back
        finally:
            conn.close()

    for path in list(p["reports"]) + list(p["candles"]):
        try:
            path.unlink()
        except OSError:
            pass
    return p


def summary(p: dict, dry_run: bool = False) -> str:
    verb = "would remove" if dry_run else "removed"
    rows = p["trades"] + p["rejections"] + p["agent_log"]
    files = len(p["reports"]) + len(p["candles"])
    if not rows and not files:
        kept = len(p["kept_sessions"])
        return (f"retention: nothing to prune — {kept} session"
                f"{'s' if kept != 1 else ''} on file, keeping {p['keep_days']}")
    return (f"retention: {verb} {p['trades']} trades, {p['rejections']} rejections, "
            f"{p['agent_log']} agent-log rows and {files} files older than "
            f"{p['cutoff_date']} (keeping the last {p['keep_days']} sessions)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trading.retention",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--keep-days", type=int, default=RETENTION_DAYS,
                    help=f"sessions to keep (default {RETENTION_DAYS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted, delete nothing")
    args = ap.parse_args(argv)
    p = prune(args.keep_days, dry_run=args.dry_run)
    print(summary(p, args.dry_run))
    if p["kept_sessions"]:
        print("   sessions kept: " + ", ".join(p["kept_sessions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
