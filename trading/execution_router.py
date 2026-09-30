"""Execution router — the paper/live switch and the safety kernel.

Hard risk limits live HERE, not in the strategy and not in the AI agent.
A bug or a bad LLM output can never blow past them. The AI agent only ever
adjusts parameters (risk multiplier, blocked symbols) within bounds enforced
in this file.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from trading.config import (
    DAILY_LOSS_LIMIT,
    DAILY_PROFIT_TARGET,
    MAX_POSITION_VALUE,
    MAX_OPEN_POSITIONS,
    MAX_ORDERS_PER_SEC,
    SLIPPAGE_PCT,
    CHARGES_PCT_ROUND_TRIP,
    MAX_COST_RISK_RATIO,
    TODAY_CONFIG_PATH,
    FALLBACK_DAY_CONFIG,
)
from trading.ledger import Ledger

IST = ZoneInfo("Asia/Kolkata")


class RateLimiter:
    """Sliding one-second window. Keeps us under SEBI's 10 orders-per-second
    registration threshold with a buffer."""

    def __init__(self, max_per_sec: int = MAX_ORDERS_PER_SEC):
        self.max_per_sec = max_per_sec
        self._timestamps: list[float] = []

    def allow(self) -> bool:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < 1.0]
        if len(self._timestamps) >= self.max_per_sec:
            return False
        self._timestamps.append(now)
        return True


def load_day_config() -> dict:
    """Read the config written by the pre-market agent; safe defaults if absent.

    A config whose `date` is not today is REJECTED. The router re-reads this
    file on every order, so a stale file would silently apply an old session's
    regime, risk multiplier and blocked_symbols to today's trades — e.g. after
    a missed pre-market run (2026-08-13 was skipped entirely because the Mac
    was asleep, leaving 08-12's config in place). Falling back to
    FALLBACK_DAY_CONFIG trades at half size instead of trusting stale advice.
    """
    try:
        with open(TODAY_CONFIG_PATH) as f:
            cfg = json.load(f)
        today = datetime.now(IST).strftime("%Y-%m-%d")
        cfg_date = str(cfg.get("date", ""))
        if cfg_date != today:
            stale = dict(FALLBACK_DAY_CONFIG)
            stale["rationale"] = (f"fallback: today_config.json is for {cfg_date or 'an unknown date'}, "
                                  f"not {today} — pre-market agent did not run today")
            return stale
        cfg["risk_multiplier"] = max(0.0, min(1.0, float(cfg.get("risk_multiplier", 0.5))))
        cfg.setdefault("blocked_symbols", [])
        cfg.setdefault("regime", "choppy")
        return cfg
    except (OSError, ValueError, TypeError):
        return dict(FALLBACK_DAY_CONFIG)


class ExecutionRouter:
    def __init__(self, mode: str = "paper", ledger: Ledger | None = None,
                 get_ltp=None):
        """
        mode:    "paper" or "live". Live requires kiteconnect + static IP +
                 broker Algo-ID tagging — see README before ever flipping this.
        get_ltp: callable(symbol) -> float. In paper mode this is your live
                 tick source; the demo passes a stub.
        """
        assert mode in ("paper", "live")
        self.mode = mode
        self.ledger = ledger or Ledger()
        self.get_ltp = get_ltp
        self.rate_limiter = RateLimiter()
        self.day_config = load_day_config()
        self.kite = None  # set externally in live mode

    # --- risk kernel: ALWAYS runs first, both modes ---

    def risk_check(self, signal: dict) -> str | None:
        """Return a rejection reason, or None if the signal passes."""
        if signal["symbol"] in self.day_config.get("blocked_symbols", []):
            return f"symbol_blocked_by_premarket_agent ({self.day_config.get('rationale', '')})"

        if self.day_config["risk_multiplier"] <= 0:
            return "risk_multiplier_zero (pre-market halt)"

        # 1. Structural stop check (no fallback)
        stop_loss = signal.get("stop_loss")
        if stop_loss is None or stop_loss <= 0:
            return "missing_stop_loss (trade rejected: structural stop required)"

        price = float(signal.get("price", 0.0))
        side = signal.get("side")
        sym = str(signal.get("symbol", ""))
        is_option = bool(signal.get("is_option")) or ("CE" in sym or "PE" in sym)

        if is_option or side == "BUY":
            if stop_loss >= price:
                return f"invalid_stop_loss (BUY stop_loss {stop_loss:.2f} must be below entry price {price:.2f})"
        elif side == "SELL" and stop_loss <= price:
            return f"invalid_stop_loss (SELL stop_loss {stop_loss:.2f} must be above entry price {price:.2f})"

        per_share_risk = abs(price - stop_loss)
        if per_share_risk <= 0.05:
            return "invalid_stop_loss (stop loss too close to entry price)"

        # 2. Cost-floor check: round-trip friction cannot exceed MAX_COST_RISK_RATIO of risk
        if is_option:
            from trading.costs import check_option_cost_floor
            passed, cost_ratio = check_option_cost_floor(price, stop_loss, qty=int(signal.get("qty", 1)))
            if not passed:
                return (f"option_cost_floor_breached (round-trip friction {cost_ratio*100:.1f}% of risk "
                        f"exceeds limit {MAX_COST_RISK_RATIO*100:.1f}%)")
        else:
            from trading.costs import check_cost_floor
            passed, cost_ratio = check_cost_floor(price, stop_loss, qty=int(signal.get("qty", 1)))
            if not passed:
                return (f"cost_floor_breached (round-trip friction {cost_ratio*100:.1f}% of risk "
                        f"exceeds limit {MAX_COST_RISK_RATIO*100:.1f}%)")

        # 3. Trend gate check (router hard risk kernel — restricted to orb_ma200 strategies)
        strat_id = str(signal.get("strategy_id", ""))
        if strat_id.startswith("orb_ma200"):
            trend_state = signal.get("trend_state")
            if trend_state not in ("up", "down"):
                return f"trend_gate_failed (trend state '{trend_state}' is not permitted)"
            if is_option:
                opt_type = signal.get("option_type") or ("CE" if "CE" in sym else "PE" if "PE" in sym else "CE")
                if opt_type == "CE" and trend_state != "up":
                    return f"trend_gate_failed (Call Option CE requires trend_state 'up', got '{trend_state}')"
                if opt_type == "PE" and trend_state != "down":
                    return f"trend_gate_failed (Put Option PE requires trend_state 'down', got '{trend_state}')"
            else:
                if side == "BUY" and trend_state != "up":
                    return f"trend_gate_failed (BUY requires trend_state 'up', got '{trend_state}')"
                if side == "SELL" and trend_state != "down":
                    return f"trend_gate_failed (SELL requires trend_state 'down', got '{trend_state}')"

        day_pnl = self.ledger.day_realized_pnl()

        if day_pnl <= DAILY_LOSS_LIMIT:
            return "daily_loss_limit_hit"

        if day_pnl >= DAILY_PROFIT_TARGET:
            return "daily_profit_target_hit (locking in gains for the day)"

        today = datetime.now(IST).strftime("%Y-%m-%d")
        open_positions = [t for t in self.ledger.open_trades() if t.get("date") == today]
        if (len(open_positions) >= MAX_OPEN_POSITIONS
                and signal["symbol"] not in {t["symbol"] for t in open_positions}):
            return f"max_open_positions ({len(open_positions)}/{MAX_OPEN_POSITIONS})"

        position_value = signal["qty"] * signal["price"]
        existing = self.ledger.open_position_value(signal["symbol"])
        if is_option:
            from trading.config import MAX_OPTION_POSITION_VALUE
            cap = MAX_OPTION_POSITION_VALUE
            if existing + position_value > cap:
                return f"max_option_position_value (cap={cap:.0f}, would_be={existing + position_value:.0f})"
        else:
            if existing + position_value > MAX_POSITION_VALUE:
                return f"max_position_value (cap={MAX_POSITION_VALUE:.0f}, would_be={existing + position_value:.0f})"

        return None

    # --- execution ---

    def execute(self, signal: dict) -> int | None:
        """Execute a signal. Returns the ledger trade id, or None if rejected."""
        # Re-read the agent's day config on every order — long-running processes
        # (webhook receiver, live loop) must pick up a config written after they
        # started, e.g. the 8:45 pre-market run or an intraday halt.
        self.day_config = load_day_config()
        self.last_rejection = None
        reason = self.risk_check(signal)
        if reason:
            self.last_rejection = reason
            self.ledger.record_rejection(signal, reason)
            return None

        if not self.rate_limiter.allow():
            self.last_rejection = "rate_limit"
            self.ledger.record_rejection(signal, "rate_limit")
            return None

        if self.mode == "paper":
            fill = self.simulate_fill(signal)
            return self.ledger.record_entry(signal, fill_price=fill, mode="paper")

        # live mode: static IP + Algo-ID tagging handled by broker for
        # sub-10-OPS personal use. Requires self.kite to be configured.
        raise NotImplementedError(
            "Live mode is intentionally not wired up. Complete the go-live "
            "checklist in README.md (static IP, OAuth automation, broker "
            "Algo-ID confirmation) before implementing kite.place_order()."
        )

    def simulate_fill(self, signal: dict) -> float:
        ltp = None
        if self.get_ltp:
            try:
                ltp = self.get_ltp(signal["symbol"])
            except Exception:
                pass
        if not ltp or ltp <= 0:
            try:
                from trading.fno.fyers import FyersClient
                fyers = FyersClient()
                sym = signal["symbol"]
                if "CE" in sym or "PE" in sym or "INDEX" in sym:
                    fyers_sym = sym if sym.startswith("NSE:") else f"NSE:{sym}"
                else:
                    fyers_sym = f"NSE:{sym}-EQ" if not sym.startswith("NSE:") else sym
                q = fyers.quotes([fyers_sym])
                if q and len(q) > 0 and "v" in q[0] and "lp" in q[0]["v"]:
                    ltp = float(q[0]["v"]["lp"])
            except Exception:
                pass
        if not ltp or ltp <= 0:
            ltp = signal["price"]

        slip = SLIPPAGE_PCT * ltp
        return ltp + slip if signal["side"] == "BUY" else ltp - slip

