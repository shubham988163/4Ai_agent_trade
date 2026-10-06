"""CLI for the F&O opening-window long scanner.

    # one scan against the live feeds
    .venv/bin/python -m trading.fno

    # candles are exchange-delayed on the free feed; opt in explicitly
    .venv/bin/python -m trading.fno --allow-delayed

    # only these names, with the alert pushed to macOS/Telegram
    .venv/bin/python -m trading.fno --symbols RELIANCE,SBIN,TATASTEEL --notify

    # poll every 2 minutes through the 09:15–10:00 window
    .venv/bin/python -m trading.fno --watch --allow-delayed

    # reproduce a saved scan, no network
    .venv/bin/python -m trading.fno --replay data/fno/2026-08-20.json --at 09:47

No order is ever placed: this process has no broker connection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as time_mod
from datetime import datetime

from trading.fno import config as C
from trading.fno import report
from trading.fno.data import LiveFeed, ReplayFeed
from trading.fno.models import IST
from trading.fno.scanner import Scanner


def _at(now: datetime, hhmm: str | None) -> datetime:
    if not hhmm:
        return now
    h, m = hhmm.split(":")
    return now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m trading.fno",
                               description="NSE F&O intraday BUY scanner "
                                           "(09:15–10:00 IST opening window)")
    p.add_argument("--symbols", help="comma-separated list; skips the liquidity screen")
    p.add_argument("--shortlist", type=int, default=C.SHORTLIST_SIZE,
                   help=f"names given the full workup (default {C.SHORTLIST_SIZE})")
    p.add_argument("--at", metavar="HH:MM",
                   help="evaluate as of this IST time instead of now")
    p.add_argument("--allow-delayed", action="store_true",
                   help="proceed even though the candle feed is exchange-delayed")
    p.add_argument("--replay", metavar="BUNDLE.json",
                   help="scan a saved JSON bundle instead of the live feeds")
    p.add_argument("--watch", action="store_true",
                   help="re-scan every --interval seconds until 10:00 IST")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--notify", action="store_true",
                   help="push A/A+ alerts via trading.notify (macOS/Telegram)")
    p.add_argument("--fyers", action="store_true",
                   help="use real-time Fyers candles via the TradeBrahma server "
                        "(needs `npm run server` and a same-day broker login)")
    p.add_argument("--fyers-base", default="http://localhost:3001",
                   help="where that server listens (default http://localhost:3001)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--relaxed", action="store_true",
                   help="opt-in looser grading: an OI-flat read is shown as "
                        "WATCH instead of rejected, so a quiet tape shows the "
                        "near-misses instead of an empty board. The score floor "
                        "is unchanged and nothing relaxed is a BUY.")
    return p


def _as_json(res) -> str:
    return json.dumps(report.as_dict(res), indent=2, default=str)


def scan_once(args) -> int:
    now = _at(datetime.now(IST), args.at)
    if args.replay:
        feed = ReplayFeed(args.replay)
        now = _at(feed.as_of, args.at)
        allow_delayed = True            # a replay is by definition not live
    elif args.fyers:
        from trading.fno.fyers import FyersFeed
        feed = FyersFeed(base=args.fyers_base)
        # Real-time candles need no delayed-data opt-in; a fallback to yfinance
        # still does, and the feed records which happened.
        allow_delayed = args.allow_delayed or feed.connected
    else:
        feed = LiveFeed()
        allow_delayed = args.allow_delayed

    scanner = Scanner(feed, now, allow_delayed=allow_delayed,
                      symbols=(args.symbols.split(",") if args.symbols else None),
                      shortlist=args.shortlist)
    res = scanner.run()

    print(_as_json(res) if args.json else report.render(res))

    if args.notify:
        from trading.notify import notify
        for cand in res.picks:
            if cand.score >= C.ALERT_SCORE:
                notify(f"F&O LONG {cand.symbol} ({cand.grade})",
                       f"{cand.levels.price:.2f} | entry "
                       f"{cand.trade.entry_low:.2f}-{cand.trade.entry_high:.2f} "
                       f"| SL {cand.trade.stop:.2f} | T1 {cand.trade.target1:.2f}")
    return 0 if res.data_ok else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.relaxed:
        C.set_relaxed(True)
    if not args.watch:
        return scan_once(args)

    end = _at(datetime.now(IST), C.SCANNER_WINDOW_END)
    while True:
        code = scan_once(args)
        now = datetime.now(IST)
        if now >= end:
            print(f"\n{C.SCANNER_WINDOW_END} IST reached — opening-window scan "
                  "complete. Existing setups should now be monitored manually.")
            return code
        time_mod.sleep(max(5, args.interval))


if __name__ == "__main__":
    sys.exit(main())
