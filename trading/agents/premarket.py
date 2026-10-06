"""Pre-market analyst agent (Role 1) — run before the open, ~08:45 IST.

Feeds overnight market context to the agent and asks for a structured verdict:
market regime, symbols to avoid today, and a risk multiplier. Writes the result
to data/today_config.json, which the ExecutionRouter reads at startup.

The agent tunes the day's parameters. It never touches orders, and the router
clamps everything it produces.

Usage:  python -m trading.agents.premarket

Exit code is 0 only when the agent itself produced the verdict. Every failure
path (no API key, network error, refusal, unparseable JSON) still writes the
safe fallback config so the day is never left without one — but it exits 1, so
a scheduler cannot read a silent agent as a successful run.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from trading.config import (TODAY_CONFIG_PATH, FALLBACK_DAY_CONFIG,
                            SCAN_UNIVERSE, RISK_CRITICAL_FEEDS,
                            DARK_FEED_RISK_CAP)
from trading.ledger import Ledger
from trading.agents.llm import call_structured

# What gather_overnight_context() returns for a source with no feed behind it.
UNAVAILABLE = "unavailable"

# The date written below is compared against datetime.now(IST) by
# execution_router.load_day_config(). Using naive local time would write a date
# that disagrees with the router on any machine not set to IST — the router
# would then reject a perfectly good verdict as stale and silently halve risk
# for the day, with nothing in the UI explaining why.
IST = ZoneInfo("Asia/Kolkata")

SYSTEM = (
    "You are a risk analyst for a personal intraday NSE equity scalping system. "
    "Your job is to set conservative day-level risk parameters, not to predict "
    "prices. When information is missing or ambiguous, reduce risk. "
    "risk_multiplier scales position size: 1.0 = normal, 0.5 = half size, "
    "0.0 = no trading today. Block a symbol only for a concrete reason "
    "(earnings today, corporate action, circuit risk, extreme news). "
    "A source reported as UNKNOWN means there is NO feed for it. Never restate "
    "an UNKNOWN source as a negative finding — 'no events reported' is not "
    "something you can know when the events feed is UNKNOWN, and saying so "
    "would let a blind spot justify more risk instead of less."
)


class PremarketVerdict(BaseModel):
    regime: Literal["trending", "choppy", "event_risk"]
    risk_multiplier: float = Field(description="0.0 to 1.0 position-size scaler")
    blocked_symbols: list[str] = Field(
        description="Symbols to avoid today, drawn only from the core watchlist")
    rationale: str = Field(description="One or two lines explaining the verdict")


def gather_overnight_context() -> dict[str, str]:
    """Collect overnight inputs. Each source degrades to 'unavailable' rather
    than failing — the agent is told what it doesn't know.

    Only global_markets is wired up today. GIFT Nifty, news headlines and the
    events calendar have no feed behind them: they are passed through as the
    literal string "unavailable" so the agent can see it is blind on them and
    lean conservative, rather than silently reading an empty value as "no news".

    Extend these with real feeds as you build: GIFT Nifty quote, news API,
    NSE events calendar. yfinance is used if installed.
    """
    ctx = {
        "gift_nifty": UNAVAILABLE,
        "global_markets": UNAVAILABLE,
        "news_headlines": UNAVAILABLE,
        "events_calendar": UNAVAILABLE,
    }
    try:
        import yfinance as yf  # optional dependency

        snapshots = []
        for name, ticker in [("Nifty 50", "^NSEI"), ("S&P 500", "^GSPC"),
                             ("Nasdaq", "^IXIC"), ("Nikkei", "^N225"),
                             ("India VIX", "^INDIAVIX")]:
            try:
                # 5d + dropna: the latest row can be NaN while a session is
                # unconsolidated — NaNs here once read as "all data lost" and
                # made the agent halt a normal day.
                h = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
                if len(h) >= 2:
                    chg = (h.iloc[-1] / h.iloc[-2] - 1) * 100
                    snapshots.append(f"{name}: {h.iloc[-1]:.0f} ({chg:+.2f}%)")
            except Exception:  # noqa: BLE001
                continue
        if snapshots:
            ctx["global_markets"] = "; ".join(snapshots)
    except ImportError:
        pass
    return ctx


def _show(key: str, ctx: dict[str, str]) -> str:
    """Render one source for the prompt.

    A missing feed is stated as UNKNOWN and spelled out, because the bare word
    "unavailable" was read as a finding rather than as an absence: on
    2026-10-06 the events line came back in the rationale as "no specific
    earnings or events reported", which is not knowable from a dark feed.
    """
    if ctx.get(key, UNAVAILABLE) == UNAVAILABLE:
        return "UNKNOWN — no feed. You cannot conclude there are no events."
    return ctx[key]


def _dark_feeds(ctx: dict[str, str]) -> list[str]:
    """Risk-critical sources that have no data behind them today."""
    return [k for k in RISK_CRITICAL_FEEDS if ctx.get(k, UNAVAILABLE) == UNAVAILABLE]


def _clamp_multiplier(value: float) -> float:
    """Risk multiplier bounded to 0.0–1.0.

    Anything non-finite (NaN/inf) or unparseable falls back to the safe
    default rather than to full size: max(0.0, min(1.0, nan)) evaluates to
    1.0 in Python, so a NaN straight out of the model would have silently
    promoted the day to maximum position size.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(FALLBACK_DAY_CONFIG["risk_multiplier"])
    if not math.isfinite(v):
        return float(FALLBACK_DAY_CONFIG["risk_multiplier"])
    return max(0.0, min(1.0, v))


def run() -> dict:
    ledger = Ledger()
    ctx = gather_overnight_context()
    now = datetime.now(IST)

    # The prompt must describe the universe the engine ACTUALLY trades. It used
    # to read "Nifty 50 constituents ({len(SCAN_UNIVERSE)} symbols)" while
    # SCAN_UNIVERSE is the 6-name core watchlist — so the agent was invited to
    # block any of 49 names, and everything outside those 6 was then discarded.
    prompt = f"""Date: {now.strftime('%A %Y-%m-%d')} (IST)
Core watchlist — the only symbols this system trades, and the only ones you may block:
  {', '.join(SCAN_UNIVERSE)}

Overnight data:
- Global markets: {ctx['global_markets']}
- GIFT Nifty: {_show('gift_nifty', ctx)}
- News headlines: {_show('news_headlines', ctx)}
- Today's events (earnings/expiry/macro): {_show('events_calendar', ctx)}

Set today's risk parameters for the scalping system."""

    fallback = PremarketVerdict(**FALLBACK_DAY_CONFIG)
    verdict = call_structured("premarket", SYSTEM, prompt,
                              PremarketVerdict, fallback, ledger=ledger)

    # call_structured hands back the *identical object* on every failure path,
    # so identity — not equality — is what separates "the agent spoke" from
    # "the agent was silent and we substituted the fallback". A model that
    # happens to return the fallback's exact values still counts as having run.
    from_agent = verdict is not fallback

    # Clamp everything before it can influence the router — belt and braces.
    universe = {str(s).strip().upper() for s in SCAN_UNIVERSE}
    blocked: list[str] = []
    ignored: list[str] = []
    for raw in verdict.blocked_symbols:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        if sym in universe:
            if sym not in blocked:
                blocked.append(sym)
        elif sym not in ignored:
            ignored.append(sym)

    # Deterministic adjustments are carried as notes so they survive the
    # rationale truncation below: if the agent's own text runs long, the note
    # explaining why risk was cut is precisely the part that must not be lost.
    notes: list[str] = []

    if ignored:
        # Dropping a block the agent asked for is itself a risk decision, so it
        # is recorded rather than swallowed: an unplaceable block means the
        # agent believed it was protecting a name the engine will still trade.
        notes.append(f"ignored {len(ignored)} block(s) outside the traded "
                     f"universe: {', '.join(ignored)}")

    multiplier = _clamp_multiplier(verdict.risk_multiplier)
    dark = _dark_feeds(ctx)
    if dark and multiplier > DARK_FEED_RISK_CAP:
        notes.append(f"capped risk {multiplier:.2f}x -> {DARK_FEED_RISK_CAP}x "
                     f"(no data for {', '.join(dark)})")
        multiplier = DARK_FEED_RISK_CAP

    suffix = f" [{' ; '.join(notes)}]" if notes else ""
    rationale = verdict.rationale.strip()[: max(0, 500 - len(suffix))] + suffix

    config = {
        "date": now.strftime("%Y-%m-%d"),
        # Machine-readable provenance: "agent" means a real verdict, "fallback"
        # means the agent never answered and these are the safe defaults. The
        # two look identical otherwise, and on a real-money day that difference
        # is the whole story.
        "source": "agent" if from_agent else "fallback",
        "regime": verdict.regime,
        "risk_multiplier": multiplier,
        "blocked_symbols": blocked,
        "dark_feeds": dark,
        "rationale": rationale,
    }

    TODAY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TODAY_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"today_config.json written: {json.dumps(config)}")
    if not from_agent:
        print("WARNING: the agent returned no verdict — the fallback config is "
              "in force (half size). Check the API key and the agent_log table.",
              file=sys.stderr)

    from trading.notify import notify      # fire-and-forget; cannot raise
    notify(f"🌅 Pre-market: {config['regime']} · {config['risk_multiplier']}x risk",
           config["rationale"][:180])
    return config


def main() -> int:
    """0 only when the agent's own verdict was written to disk."""
    return 0 if run().get("source") == "agent" else 1


if __name__ == "__main__":
    sys.exit(main())
