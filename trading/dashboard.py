"""Local dashboard — see everything the system does in one page.

Zero dependencies (stdlib http.server). Reads the SQLite ledger and serves:
  stat tiles (net P&L, win rate, trades, charges, rejections)
  cumulative P&L line chart with crosshair + tooltip
  day config from the pre-market agent
  trades table with the supervisor's verdicts
  rejections table (risk kernel / rate limiter)
  LLM audit log
  latest EOD journal report

Usage:  python -m trading.dashboard [port]     # default 8080
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from zoneinfo import ZoneInfo

import yfinance as yf

from trading import ui_theme
from trading.config import (DB_PATH, REPORTS_DIR, RETENTION_DAYS, TODAY_CONFIG_PATH)
from trading.costs import round_trip as round_trip_charges
from trading.execution_router import ExecutionRouter
from trading.fno import fyers, web as fno_web
from trading.ledger import Ledger

IST = ZoneInfo("Asia/Kolkata")


def _get_batch_ltp(symbols: list[str]) -> dict[str, float]:
    """Fetch real-time LTP for a list of equity symbols using Fyers, falling back to yfinance."""
    if not symbols:
        return {}
    res: dict[str, float] = {}
    clean_syms = list(set(s.upper().strip() for s in symbols if s))
    try:
        from trading.fno.fyers import FyersClient
        client = FyersClient()
        fyers_syms = [f"NSE:{s}-EQ" if not s.startswith("NSE:") else s for s in clean_syms]
        quotes = client.quotes(fyers_syms)
        for q in quotes:
            if isinstance(q, dict) and "v" in q and "lp" in q["v"]:
                name = q.get("n", "")
                raw = name.replace("NSE:", "").replace("-EQ", "")
                res[raw] = float(q["v"]["lp"])
    except Exception:
        pass

    missing = [s for s in clean_syms if s not in res]
    if missing:
        for s in missing:
            try:
                t = yf.Ticker(s + ".NS")
                hist = t.history(period="1d", interval="5m")
                if not hist.empty and "Close" in hist.columns:
                    res[s] = float(hist["Close"].iloc[-1])
            except Exception:
                pass
    return res


def _get_ltp(symbol: str) -> float:
    batch = _get_batch_ltp([symbol])
    return batch.get(symbol.upper().strip(), 1000.0)


def _breakdown(trades: list[dict], key: str) -> list[dict]:
    """Per-group performance report (used for side and strategy breakdowns), including live floating P&L."""
    groups: dict = {}
    for t in trades:
        k = t[key] or "—"
        g = groups.setdefault(k, {"name": k, "trades": 0, "closed": 0, "open": 0,
                                  "wins": 0, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                                  "net_pnl": 0.0, "charges": 0.0})
        g["trades"] += 1
        if t["status"] == "closed":
            g["closed"] += 1
            g["realized_pnl"] += t["pnl"] or 0
            g["charges"] += t["charges"] or 0
            if (t["pnl"] or 0) > 0:
                g["wins"] += 1
        elif t["status"] == "open":
            g["open"] += 1
            g["unrealized_pnl"] += t.get("unrealized_pnl", 0.0)
            g["charges"] += t.get("est_charges", 0.0)
    out = []
    for g in groups.values():
        g["win_rate"] = round(g["wins"] / g["closed"] * 100, 1) if g["closed"] else None
        g["net_pnl"] = round(g["realized_pnl"] + g["unrealized_pnl"], 2)
        g["realized_pnl"] = round(g["realized_pnl"], 2)
        g["unrealized_pnl"] = round(g["unrealized_pnl"], 2)
        g["charges"] = round(g["charges"], 2)
        out.append(g)
    return sorted(out, key=lambda g: g["name"])


def _handle_trade(data: dict) -> dict:
    ledger = Ledger()
    router = ExecutionRouter(mode="paper", ledger=ledger)
    sym = str(data.get("symbol", "RELIANCE")).upper().strip()
    side = str(data.get("side", "BUY")).upper().strip()
    qty = int(data.get("qty", 10))
    price = float(data["price"]) if data.get("price") else None
    if price is None or price <= 0:
        price = _get_ltp(sym)
    router.get_ltp = lambda s: price

    sl = float(data["stop_loss"]) if data.get("stop_loss") else None
    tg = float(data["target"]) if data.get("target") else None
    if sl is None and price > 0:
        sl = round(price * 0.99, 2) if side == "BUY" else round(price * 1.01, 2)
    if tg is None and price > 0:
        tg = round(price * 1.02, 2) if side == "BUY" else round(price * 0.98, 2)

    signal = {
        "symbol": sym,
        "side": side,
        "qty": qty,
        "price": price,
        "ts": time.time(),
        "stop_loss": sl,
        "target": tg,
        "strategy_id": str(data.get("strategy_id", "manual_paper")),
        "regime": router.day_config.get("regime", "choppy"),
    }
    trade_id = router.execute(signal)
    if trade_id is None:
        reason = getattr(router, "last_rejection", "Risk kernel rejection")
        return {"ok": False, "rejected": True, "reason": reason}
    return {"ok": True, "trade_id": trade_id, "signal": signal}


def _handle_close(data: dict) -> dict:
    ledger = Ledger()
    trade_id = int(data.get("trade_id", 0))
    with ledger._conn() as conn:
        t = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
    if not t:
        return {"ok": False, "error": "Trade not found"}
    if t["status"] == "closed":
        return {"ok": False, "error": "Trade already closed"}

    exit_price = float(data["exit_price"]) if data.get("exit_price") else None
    if exit_price is None or exit_price <= 0:
        exit_price = _get_ltp(t["symbol"])

    charges = round_trip_charges(t["entry_price"], exit_price, t["qty"])
    pnl = ledger.record_exit(trade_id, round(exit_price, 2), charges=round(charges, 2))
    return {"ok": True, "trade_id": trade_id, "exit_price": round(exit_price, 2), "pnl": round(pnl, 2)}


def _handle_ai_scan(data: dict) -> dict:
    sym = str(data.get("symbol", "SBIN")).strip().upper()
    try:
        from trading.agents.agent_trader import evaluate_stock, execute_recommendation, _get_live_ltp, fetch_symbol_context
        from trading.execution_router import ExecutionRouter
        from trading.ledger import Ledger

        ledger = Ledger()
        ctx = fetch_symbol_context(sym)
        rec = evaluate_stock(sym, ledger=ledger)
        if not rec:
            if ctx and not ctx.get("signal"):
                return {
                    "ok": True,
                    "symbol": sym,
                    "context": ctx or {},
                    "recommendation": {
                        "symbol": sym,
                        "action": "HOLD",
                        "conviction": 1,
                        "entry_price": ctx.get("ltp", 0.0),
                        "stop_loss": 0.0,
                        "target": 0.0,
                        "technical_analysis": f"Deterministic quant gate: setup did not satisfy all ORB+200MA conditions (OR valid: {ctx.get('or_valid')}, Trigger: {ctx.get('or_trigger_status')}, Trend: {ctx.get('trend_gate')}, Vol: {ctx.get('vol_ratio')}x). Council not invoked.",
                        "bull_case": "Not applicable — deterministic gate failed.",
                        "bear_case": "Not applicable — deterministic gate failed.",
                        "decision_rationale": "HOLD: Pre-council deterministic gate rejected candidate before LLM invocation.",
                        "trend_state": "unavailable",
                    },
                    "execution": {
                        "symbol": sym,
                        "action": "HOLD",
                        "conviction": 1,
                        "executed": False,
                        "trade_id": None,
                        "reason": "Deterministic quant gate rejected candidate before LLM invocation",
                    },
                }
            return {"ok": False, "error": f"Could not fetch market data for {sym}"}

        min_conv = int(data.get("min_conviction", 7))
        router = ExecutionRouter(mode="paper", get_ltp=_get_live_ltp, ledger=ledger)
        res = execute_recommendation(rec, router, min_conviction=min_conv)
        return {
            "ok": True,
            "symbol": sym,
            "context": ctx or {},
            "recommendation": rec.model_dump(),
            "execution": res,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _handle_premarket_run() -> dict:
    try:
        from trading.agents.premarket import run as run_premarket
        cfg = run_premarket()
        return {"ok": True, "config": cfg}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _handle_supervisor_run() -> dict:
    try:
        from trading.agents.supervisor import run as run_supervisor
        run_supervisor(loop=False)
        return {"ok": True, "message": "Supervisor review completed successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _handle_journal_run(data: dict) -> dict:
    try:
        from trading.agents.eod_journal import run as run_journal
        date = data.get("date")
        report_text = run_journal(date=date)
        return {"ok": True, "report": report_text}
    except Exception as e:
        return {"ok": False, "error": str(e)}



def get_data(date: str | None) -> dict:
    ledger = Ledger()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    dates = [r["date"] for r in conn.execute(
        "SELECT DISTINCT date FROM trades ORDER BY date DESC").fetchall()]
    # Today belongs in the picker (and is the default view) even before the
    # first trade of the day, so a live session never looks "missing".
    today = datetime.now(IST).strftime("%Y-%m-%d")
    if today not in dates:
        dates.insert(0, today)
    if not date:
        date = today

    trades = [dict(r) for r in conn.execute(
        "SELECT * FROM trades WHERE date = ? ORDER BY ts", (date,)).fetchall()] if date else []

    rejections = [dict(r) for r in conn.execute(
        "SELECT * FROM rejections ORDER BY ts DESC LIMIT 50").fetchall()]

    agent_log = [dict(r) for r in conn.execute(
        "SELECT id, ts, agent, model, ok, error, "
        "substr(COALESCE(response,''),1,400) AS response "
        "FROM agent_log ORDER BY ts DESC LIMIT 50").fetchall()]
    conn.close()

    # Enrich open trades with real-time live market quotes and floating P&L
    open_trades = [t for t in trades if t["status"] == "open"]
    if open_trades:
        syms = list({t["symbol"] for t in open_trades if t.get("symbol")})
        ltp_map = _get_batch_ltp(syms)
        for t in open_trades:
            sym = t.get("symbol", "")
            ltp = ltp_map.get(sym) or t.get("entry_price") or 0.0
            t["current_price"] = round(ltp, 2)
            entry = float(t.get("entry_price") or 0.0)
            qty = int(t.get("qty") or 1)
            side = str(t.get("side", "BUY")).upper()
            if side == "BUY":
                pnl = (ltp - entry) * qty
                pnl_pct = ((ltp - entry) / entry * 100) if entry else 0.0
            else:
                pnl = (entry - ltp) * qty
                pnl_pct = ((entry - ltp) / entry * 100) if entry else 0.0
            est_charges = round_trip_charges(entry, ltp, qty)
            t["unrealized_pnl"] = round(pnl, 2)
            t["pnl_pct"] = round(pnl_pct, 2)
            t["est_charges"] = round(est_charges, 2)
            t["net_unrealized_pnl"] = round(pnl - est_charges, 2)

    closed = [t for t in trades if t["status"] == "closed"]
    wins = [t for t in closed if (t["pnl"] or 0) > 0]

    realized_pnl = round(sum(t["pnl"] or 0 for t in closed), 2)
    unrealized_pnl = round(sum(t.get("unrealized_pnl", 0) for t in open_trades), 2)
    total_net_pnl = round(realized_pnl + unrealized_pnl, 2)

    realized_charges = round(sum(t["charges"] or 0 for t in closed), 2)
    est_open_charges = round(sum(t.get("est_charges", 0) for t in open_trades), 2)
    total_charges = round(realized_charges + est_open_charges, 2)

    stats = {
        "net_pnl": total_net_pnl,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "trades": len(trades),
        "open": len(open_trades),
        "charges": realized_charges,
        "est_open_charges": est_open_charges,
        "total_charges": total_charges,
        "rejections_today": sum(1 for r in rejections
                                if date and __import__("datetime").datetime
                                .fromtimestamp(r["ts"]).strftime("%Y-%m-%d") == date),
    }

    try:
        day_config = json.load(open(TODAY_CONFIG_PATH))
    except (OSError, ValueError):
        day_config = None

    report = None
    if date:
        p = REPORTS_DIR / f"{date}.md"
        if p.exists():
            report = p.read_text(encoding="utf-8")

    try:
        from trading.agents.llm import provider
        from trading.config import GEMINI_MODEL, ANTHROPIC_MODEL
        prov = provider()
        model = GEMINI_MODEL if prov == "gemini" else ANTHROPIC_MODEL
    except Exception:  # noqa: BLE001
        prov, model = "?", "?"

    return {"date": date, "dates": dates, "stats": stats, "trades": trades,
            "rejections": rejections, "agent_log": agent_log,
            "day_config": day_config, "report": report,
            "provider": prov, "model": model,
            "retention_days": RETENTION_DAYS,
            "side_report": _breakdown(trades, "side"),
            "strategy_report": _breakdown(trades, "strategy_id"),
            "day_pnl_all_time": ledger.day_realized_pnl(date) if date else 0}


_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Agent Dashboard</title>
<style>
__THEME__
/* Aliases: the chart and a few older blocks were written against the previous
   token names. Mapping them here keeps that code untouched and still theme-aware. */
:root{
  --surface-1:var(--card); --page:var(--bg); --grid:var(--line); --axis:var(--line-2);
  --series-1:var(--accent); --border:var(--line); --warning:var(--warn);
  --critical:var(--bad); --delta-good:var(--up); --delta-bad:var(--down);
}
h1{font-size:13px}
.filters{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;
  padding:14px;box-shadow:var(--shadow)}

/* --- Buy now: the scanner's current call, first thing on the page --- */
#buynow .body{padding:14px}
.bn-hero{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.bn-sym{font-size:26px;font-weight:700;letter-spacing:-.01em}
.bn-px{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.bn-why{color:var(--ink-2);font-size:12.5px;margin-top:10px}
.bn-none{font-size:17px;font-weight:700;letter-spacing:.02em}
.bn-levels{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;
  margin-top:12px}
.bn-levels div{background:var(--cell);padding:9px 12px}
.bn-levels .k{font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.10em}
.bn-levels .v{font-size:16px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.bn-levels .v.sl{color:var(--bad)} .bn-levels .v.tg{color:var(--good)}
.bn-opt{margin-top:11px;border:1px solid var(--line);background:var(--cell);
  border-radius:9px;padding:9px 12px;font-size:12.5px;color:var(--ink-2);
  display:flex;gap:9px;flex-wrap:wrap;align-items:baseline}
.bn-opt .k{font-size:9.5px;letter-spacing:.10em;text-transform:uppercase;
  color:var(--ink-3)}
.bn-opt .strike{font-size:15px;font-weight:650;color:var(--ink);
  font-variant-numeric:tabular-nums}
.bn-opt b{color:var(--ink);font-weight:600}
.bn-opt .nocall{color:var(--bad);font-weight:700;letter-spacing:.06em}
.bn-opt .warn{color:var(--warn);width:100%;font-size:11.5px}
.bn-more{margin-top:11px;font-size:12px;color:var(--ink-2);display:flex;
  gap:8px 10px;flex-wrap:wrap;align-items:center}
.bn-chip{border:1px solid var(--line);background:var(--cell);border-radius:999px;
  padding:2px 9px;font-size:11.5px}
.bn-warn{color:var(--bad);font-size:13px}

/* --- chart --- */
#chartwrap{position:relative}
#tooltip{position:absolute;pointer-events:none;background:var(--card);
  border:1px solid var(--line-2);border-radius:8px;padding:8px 10px;font-size:12px;
  box-shadow:var(--shadow);display:none;min-width:140px;z-index:2}
#tooltip .v{font-weight:600;font-size:14px}
#tooltip .k{display:inline-block;width:12px;height:0;border-top:2px solid var(--accent);
  vertical-align:middle;margin-right:6px}
.cfg{display:flex;flex-wrap:wrap;gap:10px 16px;font-size:12.5px;color:var(--ink-2)}
.cfg b{color:var(--ink);font-weight:600}
pre.report{white-space:pre-wrap;font:12.5px/1.6 ui-sans-serif,system-ui,sans-serif;
  margin:0;color:var(--ink-2)}
.pnl-pos{color:var(--up)} .pnl-neg{color:var(--down)}
.pnl-sub{font-size:10px;opacity:.85;font-weight:600;font-variant-numeric:tabular-nums;margin-top:1px}
.live-pulse{animation:pulse 1.3s ease-in-out infinite}
.small{font-size:11.5px;color:var(--ink-2)}
.overflow{overflow-x:auto}
</style></head>
<body>
<div class="wrap">
__TOPBAR__

  <section class="panel" id="liveTickerPanel" style="margin-top:14px">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 13px;background:var(--cell);border-bottom:1px solid var(--line)">
      <div style="display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase">
        <span class="led" style="background:var(--good);box-shadow:0 0 8px var(--good)"></span>
        <span style="color:var(--ink)">Live Market Rates <span style="font-weight:400;color:var(--ink-3)">(Fyers Zero-Delay Stream)</span></span>
      </div>
      <div style="font-size:11px;color:var(--ink-3);font-family:ui-monospace,monospace" id="liveTickerUpdated">Syncing live stream…</div>
    </div>
    <div id="liveTickerItems" style="display:flex;gap:10px;padding:10px 13px;overflow-x:auto;align-items:stretch">
      <span class="muted" style="font-size:11.5px">Loading live market quotes…</span>
    </div>
  </section>

  <section class="panel" id="buynow">
    <h2>Buy now <span class="sub" id="bnMeta">— checking the F&amp;O scanner…</span></h2>
    <div class="body" id="bnBody"></div>
  </section>

  <div id="niftyWrap"></div>

  <div class="kpis" id="tiles"></div>

  <!-- ==================== THE 4 AI AGENTS COUNCIL & MULTI-AGENT DESK ==================== -->
  <section class="panel" id="agents-desk" style="border:1px solid color-mix(in srgb,var(--accent) 35%,var(--line));box-shadow:0 0 24px rgba(47,159,219,0.07)">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--cell);flex-wrap:wrap;gap:10px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#0284c7,#818cf8);display:grid;place-items:center;font-size:18px;box-shadow:0 0 12px rgba(56,189,248,0.4)">🤖</div>
        <div>
          <h2 style="border:none;padding:0;margin:0;background:none;font-size:14.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink);display:flex;align-items:center;gap:8px">
            The 4 AI Agents Council Desk
          </h2>
          <div style="font-size:11px;color:var(--ink-3);margin-top:2px">
            Technical Analyst • Bullish Researcher • Bearish Researcher • Trader Decision Engine
          </div>
        </div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span class="badge sm ok" id="councilModelBadge">⚡ Gemini 2.5 Flash</span>
        <span class="badge sm" style="background:var(--good-soft);color:var(--good)">● 4 Agents Active &amp; Armed</span>
      </div>
    </div>

    <!-- Agent Command Bar -->
    <div style="padding:12px 16px;background:color-mix(in srgb,var(--cell) 60%,var(--card));border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:11px;font-weight:700;color:var(--ink-2);text-transform:uppercase;letter-spacing:.05em">Select Stock:</span>
        <select id="councilSymSelect" style="padding:6px 11px;border-radius:7px;border:1px solid var(--line-2);background:var(--bg-2);color:var(--ink);font-weight:700;font-size:12px;cursor:pointer;max-width:260px">
          <optgroup label="⭐ NIFTY 50 Pillars">
            <option value="SBIN" selected>SBIN (State Bank of India)</option>
            <option value="RELIANCE">RELIANCE (Reliance Industries)</option>
            <option value="HDFCBANK">HDFCBANK (HDFC Bank)</option>
            <option value="ICICIBANK">ICICIBANK (ICICI Bank)</option>
            <option value="INFY">INFY (Infosys)</option>
            <option value="TCS">TCS (Tata Consultancy)</option>
            <option value="ITC">ITC (ITC Limited)</option>
            <option value="LT">LT (Larsen &amp; Toubro)</option>
            <option value="BHARTIARTL">BHARTIARTL (Bharti Airtel)</option>
            <option value="AXISBANK">AXISBANK (Axis Bank)</option>
            <option value="KOTAKBANK">KOTAKBANK (Kotak Mahindra Bank)</option>
            <option value="HINDUNILVR">HINDUNILVR (Hindustan Unilever)</option>
            <option value="TATAMOTORS">TATAMOTORS (Tata Motors)</option>
            <option value="MARUTI">MARUTI (Maruti Suzuki)</option>
            <option value="BAJFINANCE">BAJFINANCE (Bajaj Finance)</option>
            <option value="M&amp;M">M&amp;M (Mahindra &amp; Mahindra)</option>
            <option value="SUNPHARMA">SUNPHARMA (Sun Pharma)</option>
            <option value="TITAN">TITAN (Titan Company)</option>
            <option value="TATASTEEL">TATASTEEL (Tata Steel)</option>
            <option value="NTPC">NTPC (NTPC Ltd.)</option>
            <option value="POWERGRID">POWERGRID (Power Grid Corp)</option>
            <option value="JSWSTEEL">JSWSTEEL (JSW Steel)</option>
            <option value="ADANIENT">ADANIENT (Adani Enterprises)</option>
            <option value="ADANIPORTS">ADANIPORTS (Adani Ports)</option>
            <option value="COALINDIA">COALINDIA (Coal India)</option>
            <option value="TRENT">TRENT (Trent Ltd)</option>
            <option value="BEL">BEL (Bharat Electronics)</option>
          </optgroup>
          <optgroup label="🏦 Banking Sector">
            <option value="BANKBARODA">BANKBARODA (Bank of Baroda)</option>
            <option value="PNB">PNB (Punjab National Bank)</option>
            <option value="CANBK">CANBK (Canara Bank)</option>
            <option value="INDUSINDBK">INDUSINDBK (IndusInd Bank)</option>
            <option value="FEDERALBNK">FEDERALBNK (Federal Bank)</option>
            <option value="IDFCFIRSTB">IDFCFIRSTB (IDFC First Bank)</option>
            <option value="AUBANK">AUBANK (AU Small Finance Bank)</option>
            <option value="BANDHANBNK">BANDHANBNK (Bandhan Bank)</option>
          </optgroup>
          <optgroup label="💳 Financial Services &amp; Insurance">
            <option value="BAJAJFINSV">BAJAJFINSV (Bajaj Finserv)</option>
            <option value="SHRIRAMFIN">SHRIRAMFIN (Shriram Finance)</option>
            <option value="CHOLAFIN">CHOLAFIN (Cholamandalam Inv)</option>
            <option value="MUTHOOTFIN">MUTHOOTFIN (Muthoot Finance)</option>
            <option value="LICHSGFIN">LICHSGFIN (LIC Housing Finance)</option>
            <option value="SBILIFE">SBILIFE (SBI Life Insurance)</option>
            <option value="HDFCLIFE">HDFCLIFE (HDFC Life Insurance)</option>
            <option value="ICICIGI">ICICIGI (ICICI Lombard Gen)</option>
            <option value="ICICIPRULI">ICICIPRULI (ICICI Prudential Life)</option>
            <option value="PFC">PFC (Power Finance Corp)</option>
            <option value="RECLTD">RECLTD (REC Limited)</option>
            <option value="IRFC">IRFC (Indian Railway Finance)</option>
            <option value="HDFCAMC">HDFCAMC (HDFC AMC)</option>
          </optgroup>
          <optgroup label="💻 Information Technology (IT)">
            <option value="HCLTECH">HCLTECH (HCL Technologies)</option>
            <option value="WIPRO">WIPRO (Wipro Ltd)</option>
            <option value="TECHM">TECHM (Tech Mahindra)</option>
            <option value="LTIM">LTIM (LTIMindtree)</option>
            <option value="PERSISTENT">PERSISTENT (Persistent Systems)</option>
            <option value="COFORGE">COFORGE (Coforge Ltd)</option>
            <option value="MPHASIS">MPHASIS (Mphasis Ltd)</option>
            <option value="NAUKRI">NAUKRI (Info Edge)</option>
          </optgroup>
          <optgroup label="🚗 Auto &amp; Ancillaries">
            <option value="BAJAJ-AUTO">BAJAJ-AUTO (Bajaj Auto)</option>
            <option value="EICHERMOT">EICHERMOT (Eicher Motors)</option>
            <option value="HEROMOTOCO">HEROMOTOCO (Hero MotoCorp)</option>
            <option value="TVSMOTOR">TVSMOTOR (TVS Motor)</option>
            <option value="ASHOKLEY">ASHOKLEY (Ashok Leyland)</option>
            <option value="BHARATFORG">BHARATFORG (Bharat Forge)</option>
            <option value="MOTHERSON">MOTHERSON (Samvardhana Motherson)</option>
            <option value="BALKRISIND">BALKRISIND (Balkrishna Ind)</option>
            <option value="MRF">MRF (MRF Ltd)</option>
            <option value="EXIDEIND">EXIDEIND (Exide Industries)</option>
          </optgroup>
          <optgroup label="⚡ Energy, Oil &amp; Gas, Power">
            <option value="ONGC">ONGC (Oil &amp; Natural Gas Corp)</option>
            <option value="BPCL">BPCL (Bharat Petroleum)</option>
            <option value="IOC">IOC (Indian Oil Corp)</option>
            <option value="HINDPETRO">HINDPETRO (Hindustan Petroleum)</option>
            <option value="GAIL">GAIL (GAIL India)</option>
            <option value="OIL">OIL (Oil India)</option>
            <option value="TATAPOWER">TATAPOWER (Tata Power)</option>
            <option value="ADANIGREEN">ADANIGREEN (Adani Green Energy)</option>
            <option value="ADANIENSOL">ADANIENSOL (Adani Energy Solutions)</option>
            <option value="NHPC">NHPC (NHPC Ltd)</option>
          </optgroup>
          <optgroup label="🏭 Metals &amp; Mining">
            <option value="HINDALCO">HINDALCO (Hindalco Industries)</option>
            <option value="VEDL">VEDL (Vedanta Ltd)</option>
            <option value="SAIL">SAIL (Steel Authority of India)</option>
            <option value="JINDALSTEL">JINDALSTEL (Jindal Steel &amp; Power)</option>
            <option value="NATIONALUM">NATIONALUM (National Aluminium)</option>
            <option value="HINDCOPPER">HINDCOPPER (Hindustan Copper)</option>
            <option value="APLAPOLLO">APLAPOLLO (APL Apollo Tubes)</option>
          </optgroup>
          <optgroup label="💊 Pharma &amp; Healthcare">
            <option value="CIPLA">CIPLA (Cipla Ltd)</option>
            <option value="DRREDDY">DRREDDY (Dr. Reddy's Labs)</option>
            <option value="DIVISLAB">DIVISLAB (Divi's Laboratories)</option>
            <option value="AUROPHARMA">AUROPHARMA (Aurobindo Pharma)</option>
            <option value="LUPIN">LUPIN (Lupin Ltd)</option>
            <option value="ALKEM">ALKEM (Alkem Laboratories)</option>
            <option value="TORNTPHARM">TORNTPHARM (Torrent Pharmaceuticals)</option>
            <option value="ZYDUSLIFE">ZYDUSLIFE (Zydus Lifesciences)</option>
            <option value="APOLLOHOSP">APOLLOHOSP (Apollo Hospitals)</option>
            <option value="LAURUSLABS">LAURUSLABS (Laurus Labs)</option>
            <option value="BIOCON">BIOCON (Biocon Ltd)</option>
          </optgroup>
          <optgroup label="🛒 FMCG, Consumer &amp; Retail">
            <option value="NESTLEIND">NESTLEIND (Nestle India)</option>
            <option value="BRITANNIA">BRITANNIA (Britannia Industries)</option>
            <option value="TATACONSUM">TATACONSUM (Tata Consumer)</option>
            <option value="DABUR">DABUR (Dabur India)</option>
            <option value="GODREJCP">GODREJCP (Godrej Consumer)</option>
            <option value="MARICO">MARICO (Marico Ltd)</option>
            <option value="COLPAL">COLPAL (Colgate-Palmolive)</option>
            <option value="UBL">UBL (United Breweries)</option>
            <option value="DMART">DMART (Avenue Supermarts)</option>
            <option value="JUBLFOOD">JUBLFOOD (Jubilant FoodWorks)</option>
            <option value="PAGEIND">PAGEIND (Page Industries)</option>
            <option value="VBL">VBL (Varun Beverages)</option>
            <option value="ASIANPAINT">ASIANPAINT (Asian Paints)</option>
          </optgroup>
          <optgroup label="🏗️ Infra, Cement &amp; Realty">
            <option value="ULTRACEMCO">ULTRACEMCO (UltraTech Cement)</option>
            <option value="GRASIM">GRASIM (Grasim Industries)</option>
            <option value="SHREECEM">SHREECEM (Shree Cement)</option>
            <option value="AMBUJACEM">AMBUJACEM (Ambuja Cements)</option>
            <option value="ACC">ACC (ACC Limited)</option>
            <option value="DLF">DLF (DLF Limited)</option>
            <option value="GODREJPROP">GODREJPROP (Godrej Properties)</option>
            <option value="OBEROIRLTY">OBEROIRLTY (Oberoi Realty)</option>
            <option value="LODHA">LODHA (Macrotech Developers)</option>
            <option value="PRESTIGE">PRESTIGE (Prestige Estates)</option>
            <option value="CONCOR">CONCOR (Container Corp)</option>
          </optgroup>
          <optgroup label="🛡️ Defense &amp; Capital Goods">
            <option value="HAL">HAL (Hindustan Aeronautics)</option>
            <option value="BDL">BDL (Bharat Dynamics)</option>
            <option value="SIEMENS">SIEMENS (Siemens India)</option>
            <option value="ABB">ABB (ABB India)</option>
            <option value="CUMMINSIND">CUMMINSIND (Cummins India)</option>
            <option value="POLYCAB">POLYCAB (Polycab India)</option>
            <option value="HAVELLS">HAVELLS (Havells India)</option>
          </optgroup>
          <optgroup label="🧪 Chemicals &amp; Telecom">
            <option value="PIDILITIND">PIDILITIND (Pidilite Industries)</option>
            <option value="SRF">SRF (SRF Limited)</option>
            <option value="UPL">UPL (UPL Limited)</option>
            <option value="TATACHEM">TATACHEM (Tata Chemicals)</option>
            <option value="DEEPAKNTR">DEEPAKNTR (Deepak Nitrite)</option>
            <option value="IDEA">IDEA (Vodafone Idea)</option>
            <option value="INDUSTOWER">INDUSTOWER (Indus Towers)</option>
            <option value="ZEEL">ZEEL (Zee Entertainment)</option>
            <option value="PVRINOX">PVRINOX (PVR INOX)</option>
          </optgroup>
          <optgroup label="🚀 New Age Tech &amp; Travel">
            <option value="ZOMATO">ZOMATO (Zomato Ltd)</option>
            <option value="PAYTM">PAYTM (One97 Communications)</option>
            <option value="NYKAA">NYKAA (FSN E-Commerce)</option>
            <option value="POLICYBZR">POLICYBZR (PB Fintech)</option>
            <option value="IRCTC">IRCTC (IRCTC Ltd)</option>
            <option value="INDIGO">INDIGO (InterGlobe Aviation)</option>
          </optgroup>
        </select>
      </div>

      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:11px;font-weight:700;color:var(--ink-2);text-transform:uppercase;letter-spacing:.05em">Custom NSE:</span>
        <input type="text" id="councilSymCustom" placeholder="e.g. TITAN" style="width:85px;padding:6px 9px;border-radius:7px;border:1px solid var(--line-2);background:var(--bg-2);color:var(--ink);font-weight:700;font-size:12px;text-transform:uppercase">
      </div>

      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:11px;font-weight:700;color:var(--ink-2);text-transform:uppercase;letter-spacing:.05em">Min Conviction:</span>
        <select id="councilMinConviction" style="padding:6px 9px;border-radius:7px;border:1px solid var(--line-2);background:var(--bg-2);color:var(--ink);font-size:12px">
          <option value="6">6 / 10 (Moderate)</option>
          <option value="7" selected>7 / 10 (Standard - Auto-Trade ≥7)</option>
          <option value="8">8 / 10 (High Conviction)</option>
          <option value="9">9 / 10 (Strict)</option>
        </select>
      </div>

      <div style="display:flex;align-items:center;gap:8px;margin-left:auto">
        <button class="btn go" id="btnRunCouncilDebate" style="padding:7px 16px;font-size:12px;font-weight:700;background:linear-gradient(135deg,var(--accent),#4f46e5);border:none;color:#fff;box-shadow:0 0 12px rgba(47,159,219,0.35);cursor:pointer;border-radius:7px">
          ⚡ Run 4-Agent Debate &amp; Auto-Trade
        </button>
      </div>
    </div>

    <!-- The 4 Council Agent Cards Grid -->
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:12px;padding:16px" id="agentsGrid">
      
      <!-- Agent 1: Technical Analyst -->
      <div class="agent-card agent-tech">
        <div class="agent-header">
          <div class="agent-avatar">📈</div>
          <div>
            <div class="agent-name">1. Technical Analyst</div>
            <div class="agent-role">ORB &amp; Multi-TF 200MA Gate</div>
          </div>
          <span class="agent-status-pill" id="techStatus" style="background:var(--accent-soft);color:var(--accent)">● Standby</span>
        </div>
        <div style="font-size:11px;color:var(--ink-3);display:flex;gap:5px;flex-wrap:wrap">
          <span class="bn-chip" style="color:#38bdf8">ORB 09:15-09:30</span>
          <span class="bn-chip" style="color:#a855f7">1H &amp; 30m 200 SMA</span>
          <span class="bn-chip" style="color:#eab308">Session VWAP</span>
          <span class="bn-chip">Vol Spike ≥1.5x</span>
        </div>
        <div class="agent-body" id="techOutput">
          Audits the 09:15-09:30 Opening Range (0.3%-1.5% range filter), validates 200 SMA trend gate across 1H and 30m, session VWAP slope, and 5m ATR volatility.
        </div>
      </div>

      <!-- Agent 2: Bullish Researcher -->
      <div class="agent-card agent-bull">
        <div class="agent-header">
          <div class="agent-avatar">🐂</div>
          <div>
            <div class="agent-name">2. Bullish Researcher</div>
            <div class="agent-role">ORB Long Breakout Specialist</div>
          </div>
          <span class="agent-status-pill" id="bullStatus" style="background:var(--good-soft);color:var(--good)">● Standby</span>
        </div>
        <div style="font-size:11px;color:var(--ink-3);display:flex;gap:5px;flex-wrap:wrap">
          <span class="bn-chip" style="color:#22c55e">Uptrend Gate (LTP > 200MA)</span>
          <span class="bn-chip">OR-H Breakout Close</span>
          <span class="bn-chip">Volume Surge ≥1.5x</span>
        </div>
        <div class="agent-body" id="bullOutput">
          Argues the Long breakout thesis ONLY when price is strictly above BOTH 1H &amp; 30m 200 SMAs, with fresh candle close above OR-High and volume expansion.
        </div>
      </div>

      <!-- Agent 3: Bearish Researcher -->
      <div class="agent-card agent-bear">
        <div class="agent-header">
          <div class="agent-avatar">🐻</div>
          <div>
            <div class="agent-name">3. Bearish Researcher</div>
            <div class="agent-role">ORB Short &amp; False Breakout Auditor</div>
          </div>
          <span class="agent-status-pill" id="bearStatus" style="background:var(--bad-soft);color:var(--bad)">● Standby</span>
        </div>
        <div style="font-size:11px;color:var(--ink-3);display:flex;gap:5px;flex-wrap:wrap">
          <span class="bn-chip" style="color:#ef4444">Downtrend Gate (LTP < 200MA)</span>
          <span class="bn-chip">OR-L Breakdown</span>
          <span class="bn-chip">Range Trap Audit (&lt;0.3% / &gt;1.5%)</span>
        </div>
        <div class="agent-body" id="bearOutput">
          Audits for false breakout wicks, range disqualification (&lt;0.3% or &gt;1.5%), overhead 200 MAs, or short breakdown setups below OR-Low.
        </div>
      </div>

      <!-- Agent 4: Trader Decision & Execution Engine -->
      <div class="agent-card agent-trader">
        <div class="agent-header">
          <div class="agent-avatar">⚖️</div>
          <div>
            <div class="agent-name">4. Trader Decision Agent</div>
            <div class="agent-role">ORB + 200MA Synthesis &amp; Risk Kernel</div>
          </div>
          <span class="agent-status-pill" id="traderStatus" style="background:var(--warn-soft);color:var(--warn)">● Arbitrator</span>
        </div>
        <div style="font-size:11px;color:var(--ink-3);display:flex;gap:5px;flex-wrap:wrap">
          <span class="bn-chip" style="color:#eab308">Strict Rule Enforcement</span>
          <span class="bn-chip">Structural ATR Stop</span>
          <span class="bn-chip">2.0 R:R Target</span>
          <span class="bn-chip">Conviction (≥7/10)</span>
        </div>
        <div class="agent-body" id="traderOutput">
          Strictly enforces the 6 ORB + 200MA rules. Dispatches BUY/SELL orders with structural ATR stops and 2.0 R:R only if conviction ≥7/10; otherwise mandates HOLD.
        </div>
      </div>
    </div>

    <!-- Live Debate Deliberation & Execution Stream -->
    <div id="councilDebateResult" style="display:none;margin:0 16px 16px;padding:14px;background:var(--card);border:1px solid var(--line);border-radius:10px"></div>

    <!-- 🔄 The 4 Lifecycle Architecture Agents Pipeline -->
    <div style="border-top:1px solid var(--line);padding:14px 16px;background:color-mix(in srgb,var(--cell) 45%,var(--card))">
      <div style="font-size:11px;font-weight:700;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
        <span>🔄 Full Lifecycle Agent Suite</span>
        <span style="color:var(--ink-2);font-weight:400;text-transform:none">Deterministic Risk Kernel &amp; SQLite Ledger Architecture</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px">
        
        <div style="background:var(--cell);border:1px solid var(--line);border-radius:8px;padding:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span style="font-weight:700;color:var(--ink);font-size:12px">☀️ 1. Pre-Market Analyst</span>
            <button class="btn" id="btnRunPremarket" style="font-size:10px;padding:2px 7px">Run Now</button>
          </div>
          <div style="font-size:11px;color:var(--ink-2)">Runs before 9:15 AM to set daily regime, macro bias, and risk multiplier (0.5x–1.5x).</div>
        </div>

        <div style="background:var(--cell);border:1px solid var(--line);border-radius:8px;padding:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span style="font-weight:700;color:var(--ink);font-size:12px">⚡ 2. 4-Agent Debate Desk</span>
            <span class="badge sm ok" style="font-size:9.5px">Active</span>
          </div>
          <div style="font-size:11px;color:var(--ink-2)">Multi-role intraday council evaluating technicals, bull thesis, bear risks, and execution.</div>
        </div>

        <div style="background:var(--cell);border:1px solid var(--line);border-radius:8px;padding:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span style="font-weight:700;color:var(--ink);font-size:12px">🛡️ 3. Real-Time Supervisor</span>
            <button class="btn" id="btnRunSupervisor" style="font-size:10px;padding:2px 7px">Audit Trades</button>
          </div>
          <div style="font-size:11px;color:var(--ink-2)">Async trade auditor checking for rule violations, revenge trades, or regime mismatch.</div>
        </div>

        <div style="background:var(--cell);border:1px solid var(--line);border-radius:8px;padding:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span style="font-weight:700;color:var(--ink);font-size:12px">📝 4. EOD Journal Agent</span>
            <button class="btn" id="btnRunJournal" style="font-size:10px;padding:2px 7px">Generate</button>
          </div>
          <div style="font-size:11px;color:var(--ink-2)">Post-market reflection identifying recurring mistakes and high-impact optimizations.</div>
        </div>

      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Cumulative net P&amp;L <span class="sub">closed trades through the selected session</span></h2>
    <div class="body" id="chartwrap">
      <svg id="chart" width="100%" height="240" role="img"
           aria-label="Cumulative net P&L line chart; values also in the trades table below"></svg>
      <div id="tooltip"></div>
    </div>
  </section>

  <section class="panel">
    <h2>Buy vs Sell <span class="sub">selected session</span></h2>
    <div class="scroll"><table id="sides"></table></div>
  </section>

  <section class="panel">
    <h2>By strategy <span class="sub">selected session</span></h2>
    <div class="scroll"><table id="strats"></table></div>
  </section>

  <section class="panel">
    <h2>Pre-market agent <span class="sub">today's config</span></h2>
    <div class="body"><div class="cfg" id="cfg"></div></div>
  </section>

  <section class="panel">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:9px 13px;border-bottom:1px solid var(--line);background:var(--cell)">
      <h2 style="border-bottom:none;padding:0;background:none;margin:0">Trades <span class="sub">with the supervisor's verdict</span></h2>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="btn" id="btnAiScan" style="font-size:11.5px;padding:4px 10px;border-color:color-mix(in srgb,var(--accent) 55%,var(--line));color:var(--accent);font-weight:600">⚡ AI Agent Scan</button>
        <button class="btn go" id="btnNewTrade" style="font-size:11.5px;padding:4px 10px">+ Take Paper Trade</button>
      </div>
    </div>
    <div id="aiScanDrawer" style="display:none;padding:14px;background:var(--card);border-bottom:1px solid var(--line);font-size:12px">
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
        <span style="font-weight:700;color:var(--ink);display:flex;align-items:center;gap:6px">
          <span>⚡ Multi-Agent Evaluator</span>
          <span class="badge sm ok">Gemini 3.5 Flash Lite</span>
        </span>
        <label style="margin-left:auto"><b>Symbol:</b>
          <select id="aiSymSelect" style="padding:5px 9px;border-radius:6px;border:1px solid var(--line);background:var(--cell);color:var(--ink);font-weight:600;max-width:240px">
          <optgroup label="⭐ NIFTY 50 Pillars">
            <option value="SBIN" selected>SBIN (State Bank of India)</option>
            <option value="RELIANCE">RELIANCE (Reliance Industries)</option>
            <option value="HDFCBANK">HDFCBANK (HDFC Bank)</option>
            <option value="ICICIBANK">ICICIBANK (ICICI Bank)</option>
            <option value="INFY">INFY (Infosys)</option>
            <option value="TCS">TCS (Tata Consultancy)</option>
            <option value="ITC">ITC (ITC Limited)</option>
            <option value="LT">LT (Larsen &amp; Toubro)</option>
            <option value="BHARTIARTL">BHARTIARTL (Bharti Airtel)</option>
            <option value="AXISBANK">AXISBANK (Axis Bank)</option>
            <option value="KOTAKBANK">KOTAKBANK (Kotak Mahindra Bank)</option>
            <option value="HINDUNILVR">HINDUNILVR (Hindustan Unilever)</option>
            <option value="TATAMOTORS">TATAMOTORS (Tata Motors)</option>
            <option value="MARUTI">MARUTI (Maruti Suzuki)</option>
            <option value="BAJFINANCE">BAJFINANCE (Bajaj Finance)</option>
            <option value="M&amp;M">M&amp;M (Mahindra &amp; Mahindra)</option>
            <option value="SUNPHARMA">SUNPHARMA (Sun Pharma)</option>
            <option value="TITAN">TITAN (Titan Company)</option>
            <option value="TATASTEEL">TATASTEEL (Tata Steel)</option>
            <option value="NTPC">NTPC (NTPC Ltd.)</option>
            <option value="POWERGRID">POWERGRID (Power Grid Corp)</option>
            <option value="JSWSTEEL">JSWSTEEL (JSW Steel)</option>
            <option value="ADANIENT">ADANIENT (Adani Enterprises)</option>
            <option value="ADANIPORTS">ADANIPORTS (Adani Ports)</option>
            <option value="COALINDIA">COALINDIA (Coal India)</option>
            <option value="TRENT">TRENT (Trent Ltd)</option>
            <option value="BEL">BEL (Bharat Electronics)</option>
          </optgroup>
          <optgroup label="🏦 Banking Sector">
            <option value="BANKBARODA">BANKBARODA (Bank of Baroda)</option>
            <option value="PNB">PNB (Punjab National Bank)</option>
            <option value="CANBK">CANBK (Canara Bank)</option>
            <option value="INDUSINDBK">INDUSINDBK (IndusInd Bank)</option>
            <option value="FEDERALBNK">FEDERALBNK (Federal Bank)</option>
            <option value="IDFCFIRSTB">IDFCFIRSTB (IDFC First Bank)</option>
            <option value="AUBANK">AUBANK (AU Small Finance Bank)</option>
            <option value="BANDHANBNK">BANDHANBNK (Bandhan Bank)</option>
          </optgroup>
          <optgroup label="💳 Financial Services &amp; Insurance">
            <option value="BAJAJFINSV">BAJAJFINSV (Bajaj Finserv)</option>
            <option value="SHRIRAMFIN">SHRIRAMFIN (Shriram Finance)</option>
            <option value="CHOLAFIN">CHOLAFIN (Cholamandalam Inv)</option>
            <option value="MUTHOOTFIN">MUTHOOTFIN (Muthoot Finance)</option>
            <option value="LICHSGFIN">LICHSGFIN (LIC Housing Finance)</option>
            <option value="SBILIFE">SBILIFE (SBI Life Insurance)</option>
            <option value="HDFCLIFE">HDFCLIFE (HDFC Life Insurance)</option>
            <option value="ICICIGI">ICICIGI (ICICI Lombard Gen)</option>
            <option value="ICICIPRULI">ICICIPRULI (ICICI Prudential Life)</option>
            <option value="PFC">PFC (Power Finance Corp)</option>
            <option value="RECLTD">RECLTD (REC Limited)</option>
            <option value="IRFC">IRFC (Indian Railway Finance)</option>
            <option value="HDFCAMC">HDFCAMC (HDFC AMC)</option>
          </optgroup>
          <optgroup label="💻 Information Technology (IT)">
            <option value="HCLTECH">HCLTECH (HCL Technologies)</option>
            <option value="WIPRO">WIPRO (Wipro Ltd)</option>
            <option value="TECHM">TECHM (Tech Mahindra)</option>
            <option value="LTIM">LTIM (LTIMindtree)</option>
            <option value="PERSISTENT">PERSISTENT (Persistent Systems)</option>
            <option value="COFORGE">COFORGE (Coforge Ltd)</option>
            <option value="MPHASIS">MPHASIS (Mphasis Ltd)</option>
            <option value="NAUKRI">NAUKRI (Info Edge)</option>
          </optgroup>
          <optgroup label="🚗 Auto &amp; Ancillaries">
            <option value="BAJAJ-AUTO">BAJAJ-AUTO (Bajaj Auto)</option>
            <option value="EICHERMOT">EICHERMOT (Eicher Motors)</option>
            <option value="HEROMOTOCO">HEROMOTOCO (Hero MotoCorp)</option>
            <option value="TVSMOTOR">TVSMOTOR (TVS Motor)</option>
            <option value="ASHOKLEY">ASHOKLEY (Ashok Leyland)</option>
            <option value="BHARATFORG">BHARATFORG (Bharat Forge)</option>
            <option value="MOTHERSON">MOTHERSON (Samvardhana Motherson)</option>
            <option value="BALKRISIND">BALKRISIND (Balkrishna Ind)</option>
            <option value="MRF">MRF (MRF Ltd)</option>
            <option value="EXIDEIND">EXIDEIND (Exide Industries)</option>
          </optgroup>
          <optgroup label="⚡ Energy, Oil &amp; Gas, Power">
            <option value="ONGC">ONGC (Oil &amp; Natural Gas Corp)</option>
            <option value="BPCL">BPCL (Bharat Petroleum)</option>
            <option value="IOC">IOC (Indian Oil Corp)</option>
            <option value="HINDPETRO">HINDPETRO (Hindustan Petroleum)</option>
            <option value="GAIL">GAIL (GAIL India)</option>
            <option value="OIL">OIL (Oil India)</option>
            <option value="TATAPOWER">TATAPOWER (Tata Power)</option>
            <option value="ADANIGREEN">ADANIGREEN (Adani Green Energy)</option>
            <option value="ADANIENSOL">ADANIENSOL (Adani Energy Solutions)</option>
            <option value="NHPC">NHPC (NHPC Ltd)</option>
          </optgroup>
          <optgroup label="🏭 Metals &amp; Mining">
            <option value="HINDALCO">HINDALCO (Hindalco Industries)</option>
            <option value="VEDL">VEDL (Vedanta Ltd)</option>
            <option value="SAIL">SAIL (Steel Authority of India)</option>
            <option value="JINDALSTEL">JINDALSTEL (Jindal Steel &amp; Power)</option>
            <option value="NATIONALUM">NATIONALUM (National Aluminium)</option>
            <option value="HINDCOPPER">HINDCOPPER (Hindustan Copper)</option>
            <option value="APLAPOLLO">APLAPOLLO (APL Apollo Tubes)</option>
          </optgroup>
          <optgroup label="💊 Pharma &amp; Healthcare">
            <option value="CIPLA">CIPLA (Cipla Ltd)</option>
            <option value="DRREDDY">DRREDDY (Dr. Reddy's Labs)</option>
            <option value="DIVISLAB">DIVISLAB (Divi's Laboratories)</option>
            <option value="AUROPHARMA">AUROPHARMA (Aurobindo Pharma)</option>
            <option value="LUPIN">LUPIN (Lupin Ltd)</option>
            <option value="ALKEM">ALKEM (Alkem Laboratories)</option>
            <option value="TORNTPHARM">TORNTPHARM (Torrent Pharmaceuticals)</option>
            <option value="ZYDUSLIFE">ZYDUSLIFE (Zydus Lifesciences)</option>
            <option value="APOLLOHOSP">APOLLOHOSP (Apollo Hospitals)</option>
            <option value="LAURUSLABS">LAURUSLABS (Laurus Labs)</option>
            <option value="BIOCON">BIOCON (Biocon Ltd)</option>
          </optgroup>
          <optgroup label="🛒 FMCG, Consumer &amp; Retail">
            <option value="NESTLEIND">NESTLEIND (Nestle India)</option>
            <option value="BRITANNIA">BRITANNIA (Britannia Industries)</option>
            <option value="TATACONSUM">TATACONSUM (Tata Consumer)</option>
            <option value="DABUR">DABUR (Dabur India)</option>
            <option value="GODREJCP">GODREJCP (Godrej Consumer)</option>
            <option value="MARICO">MARICO (Marico Ltd)</option>
            <option value="COLPAL">COLPAL (Colgate-Palmolive)</option>
            <option value="UBL">UBL (United Breweries)</option>
            <option value="DMART">DMART (Avenue Supermarts)</option>
            <option value="JUBLFOOD">JUBLFOOD (Jubilant FoodWorks)</option>
            <option value="PAGEIND">PAGEIND (Page Industries)</option>
            <option value="VBL">VBL (Varun Beverages)</option>
            <option value="ASIANPAINT">ASIANPAINT (Asian Paints)</option>
          </optgroup>
          <optgroup label="🏗️ Infra, Cement &amp; Realty">
            <option value="ULTRACEMCO">ULTRACEMCO (UltraTech Cement)</option>
            <option value="GRASIM">GRASIM (Grasim Industries)</option>
            <option value="SHREECEM">SHREECEM (Shree Cement)</option>
            <option value="AMBUJACEM">AMBUJACEM (Ambuja Cements)</option>
            <option value="ACC">ACC (ACC Limited)</option>
            <option value="DLF">DLF (DLF Limited)</option>
            <option value="GODREJPROP">GODREJPROP (Godrej Properties)</option>
            <option value="OBEROIRLTY">OBEROIRLTY (Oberoi Realty)</option>
            <option value="LODHA">LODHA (Macrotech Developers)</option>
            <option value="PRESTIGE">PRESTIGE (Prestige Estates)</option>
            <option value="CONCOR">CONCOR (Container Corp)</option>
          </optgroup>
          <optgroup label="🛡️ Defense &amp; Capital Goods">
            <option value="HAL">HAL (Hindustan Aeronautics)</option>
            <option value="BDL">BDL (Bharat Dynamics)</option>
            <option value="SIEMENS">SIEMENS (Siemens India)</option>
            <option value="ABB">ABB (ABB India)</option>
            <option value="CUMMINSIND">CUMMINSIND (Cummins India)</option>
            <option value="POLYCAB">POLYCAB (Polycab India)</option>
            <option value="HAVELLS">HAVELLS (Havells India)</option>
          </optgroup>
          <optgroup label="🧪 Chemicals &amp; Telecom">
            <option value="PIDILITIND">PIDILITIND (Pidilite Industries)</option>
            <option value="SRF">SRF (SRF Limited)</option>
            <option value="UPL">UPL (UPL Limited)</option>
            <option value="TATACHEM">TATACHEM (Tata Chemicals)</option>
            <option value="DEEPAKNTR">DEEPAKNTR (Deepak Nitrite)</option>
            <option value="IDEA">IDEA (Vodafone Idea)</option>
            <option value="INDUSTOWER">INDUSTOWER (Indus Towers)</option>
            <option value="ZEEL">ZEEL (Zee Entertainment)</option>
            <option value="PVRINOX">PVRINOX (PVR INOX)</option>
          </optgroup>
          <optgroup label="🚀 New Age Tech &amp; Travel">
            <option value="ZOMATO">ZOMATO (Zomato Ltd)</option>
            <option value="PAYTM">PAYTM (One97 Communications)</option>
            <option value="NYKAA">NYKAA (FSN E-Commerce)</option>
            <option value="POLICYBZR">POLICYBZR (PB Fintech)</option>
            <option value="IRCTC">IRCTC (IRCTC Ltd)</option>
            <option value="INDIGO">INDIGO (InterGlobe Aviation)</option>
          </optgroup>
          </select>
        </label>
        <label><b>Or custom:</b>
          <input type="text" id="aiSymCustom" placeholder="e.g. TECHM" style="width:95px;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--cell);color:var(--ink);text-transform:uppercase">
        </label>
        <button class="btn go" id="btnStartAiScan" style="padding:5px 14px;font-weight:700">Scan &amp; Auto-Trade</button>
        <button class="btn" id="btnCancelAiScan" style="padding:5px 10px">Close</button>
      </div>
      <div id="aiScanStatus" style="margin-top:10px;display:none"></div>
    </div>
    <div id="tradeForm" style="display:none;padding:12px;background:var(--card);border-bottom:1px solid var(--line);font-size:12px">
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
        <label><b>Symbol:</b> <input type="text" id="tSym" value="RELIANCE" style="width:100px;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--cell);color:var(--ink);font-weight:600;text-transform:uppercase"></label>
        <label><b>Side:</b> <select id="tSide" style="padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--cell);color:var(--ink)"><option value="BUY">BUY (Long)</option><option value="SELL">SELL (Short)</option></select></label>
        <label><b>Qty:</b> <input type="number" id="tQty" value="10" min="1" style="width:65px;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--cell);color:var(--ink)"></label>
        <label><b>Price (INR):</b> <input type="number" id="tPx" placeholder="Market (LTP)" step="0.05" style="width:110px;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--cell);color:var(--ink)"></label>
        <button class="btn go" id="btnSubmitTrade" style="padding:5px 14px">Execute Order</button>
        <button class="btn" id="btnCancelTrade" style="padding:5px 10px">Cancel</button>
        <span id="tradeMsg" style="font-size:12px;font-weight:600"></span>
      </div>
    </div>
    <div class="scroll"><table id="trades"></table></div>
  </section>

  <section class="panel">
    <h2>Rejections <span class="sub">risk kernel &amp; rate limiter, last 50</span></h2>
    <div class="scroll"><table id="rej"></table></div>
  </section>

  <section class="panel">
    <h2>LLM audit log <span class="sub">last 50 calls</span></h2>
    <div class="scroll"><table id="alog"></table></div>
  </section>

  <section class="panel">
    <h2>EOD journal <span class="sub">written by the journal agent</span></h2>
    <div class="body"><pre class="report" id="report"></pre></div>
  </section>

  <div class="foot" id="foot"></div>
</div>

<script>
"use strict";
let DATA=null, selDate=null;

const fmt=(n,d=2)=> n==null ? "—" : Number(n).toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d});
const t2time=ts=> new Date(ts*1000).toLocaleTimeString("en-IN",{hour12:false});
// "10/8/2026" reads as either 10 Aug or 8 Oct depending on the reader; spell the
// month so a timestamp in a log is never ambiguous.
const t2stamp=ts=> new Date(ts*1000).toLocaleString("en-IN",
  {day:"2-digit",month:"short",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});

function el(tag, cls, text){const e=document.createElement(tag); if(cls)e.className=cls;
  if(text!=null)e.textContent=text; return e;}

// The scanner runs its own background scan; this only reads the cached result,
// so polling it is cheap and never blocks this page.
async function loadFno(){
  try{
    const r = await fetch("/api/fno", {cache:"no-store"});
    const st = await r.json();
    renderBuyNow(st);
    renderNifty(st);
  }catch(e){
    document.getElementById("bnMeta").textContent = "— scanner unreachable";
    document.getElementById("bnBody").replaceChildren(
      el("div","bn-warn","Could not reach the scanner API: "+e));
  }
}

__NIFTYJS__
__CONTEXTJS__

function renderNifty(st){
  const wrap = document.getElementById("niftyWrap");
  if(!wrap) return;
  wrap.replaceChildren();
  const s = st ? st.scan : null;
  if(!s) return;
  const ix = s.index_options || null;
  if(!ix && !s.chain && !s.pivots) return;
  if(!ix){
    const only = contextPanel(s.chain, s.pivots, v => fmt(v));
    if(only) wrap.append(only);
    return;
  }
  wrap.append(niftyPanel(ix, v => fmt(v)));
  const ctx = contextPanel(s.chain, s.pivots, v => fmt(v));
  if(ctx) wrap.append(ctx);
}

function renderBuyNow(st){
  const meta=document.getElementById("bnMeta"), box=document.getElementById("bnBody");
  box.replaceChildren();
  const s=st.scan;
  const age = st.age_sec==null ? "" :
    " · updated "+(st.age_sec<60 ? st.age_sec+"s" : Math.round(st.age_sec/60)+"m")+" ago";

  if(!s){
    meta.textContent = st.scanning ? "— scanning the F&O universe…"
      : st.error ? "— scan failed" : "— no scan yet";
    box.append(el("div","muted", st.error || "The first scan is running; this fills in shortly."));
    return;
  }
  meta.textContent = "— " + s.window + (st.scanning ? " · rescanning" : "") + age;

  // Data integrity gates the call, exactly as it does in the scanner itself.
  if(!s.data_ok){
    box.append(el("div","bn-warn", s.stale_banner));
    box.append(el("div","muted","No entry, stop or target is produced. "
      + "Open the scanner for the data notes."));
    const l1=el("div","bn-more"); l1.append(link()); box.append(l1);
    return;
  }

  const picks = s.picks || [];
  if(window.ScannerAlerts && picks.length){
    picks.forEach(p => window.ScannerAlerts.checkAndAlert(p, "BUY"));
  }
  if(!picks.length){
    const watch=(s.candidates||[]).filter(c=>c.verdict==="WATCH").length;
    if(s.signals_allowed === false){
      // Not "nothing qualifies" — the scanner is not permitted to answer yet.
      box.append(el("div","bn-none","TOO EARLY TO SAY"));
      box.append(el("div","muted", s.window_note
        || "this window does not emit signals"));
      box.append(el("div","muted",
        "The 09:15\u201309:30 range has to finish before any breakout can be "
        + "graded, so the first signals are possible from 09:30."));
    } else {
    box.append(el("div","bn-none", s.no_trade || "NO HIGH-QUALITY LONG SETUP"));
    box.append(el("div","muted", "Nothing clears the rules right now"
      + (watch ? " — "+watch+" name"+(watch>1?"s":"")+" on watch." : ".")
      + " Not forcing a trade is the correct output most mornings."));
    }
    const l2=el("div","bn-more"); l2.append(link()); box.append(l2);
    return;
  }

  const c=picks[0], t=c.trade;
  const hero=el("div","bn-hero");
  hero.append(el("span","bn-sym", c.symbol));
  const px=el("span","bn-px", "₹"+fmt(c.price));
  hero.append(px);
  if(c.pct_change!=null){
    const ch=el("span","bn-px "+(c.pct_change>=0?"up":"down"),
      (c.pct_change>=0?"+":"")+c.pct_change.toFixed(2)+"%");
    hero.append(ch);
  }
  hero.append(el("span","badge buy","BUY · "+c.grade+" · "+
    (c.score%1?c.score.toFixed(1):c.score.toFixed(0))+"/100"));
  hero.append(el("span","muted", c.sector+" · "+c.contract));
  box.append(hero);

  if(t){
    const g=el("div","bn-levels");
    const cell=(k,v,cls,note)=>{const d=el("div"); d.append(el("div","k",k));
      d.append(el("div","v "+(cls||""),v)); if(note) d.append(el("div","k",note));
      g.append(d);};
    cell("Entry zone","₹"+fmt(t.entry_low)+" – "+fmt(t.entry_high),"","buy inside this band");
    cell("Stop loss","₹"+fmt(t.stop),"sl",t.stop_basis);
    cell("Target 1","₹"+fmt(t.target1),"tg","1:"+t.rr1);
    cell("Target 2","₹"+fmt(t.target2),"tg","1:"+t.rr2);
    cell("Risk","₹"+fmt(t.risk),"",t.risk_pct.toFixed(2)+"% of price");
    box.append(g);
  }
  // How the same long could be taken in calls — breakeven first, because a
  // call can lose while the stock setup works exactly as planned.
  if(c.options){
    const o=c.options, row=el("div","bn-opt");
    row.append(el("span","k","Call option"));
    if(o.quote){
      const q=o.quote;
      row.append(el("span","strike", q.strike+" CE @ "+fmt(q.ltp)));
      row.append(el("span",null,q.expiry+" · "+q.moneyness));
      const be=el("span"); be.append(document.createTextNode("breakeven "));
      be.append(el("b",null,fmt(o.breakeven)));
      be.append(document.createTextNode(o.clears_t1 ? " — under target 1, so T1 pays"
        : o.clears_t2 ? " — above target 1; only the 1:3 target pays"
        : " — above both targets"));
      row.append(be);
      (o.warnings||[]).forEach(w=>row.append(el("span","warn","⚠ "+w)));
    } else {
      row.append(el("span","nocall","NO CALL"));
      row.append(el("span",null,(o.rejections||["no candidate strike"])[0]));
    }
    box.append(row);
  }

  if(c.reasons && c.reasons.length)
    box.append(el("div","bn-why","Why: "+c.reasons.slice(0,2).join(" · ")));

  const more=el("div","bn-more");
  picks.slice(1).forEach(o=>more.append(el("span","bn-chip",
    o.symbol+"  ₹"+fmt(o.price)+"  "+o.grade)));
  const watch=(s.candidates||[]).filter(x=>x.verdict==="WATCH");
  watch.slice(0,3).forEach(o=>more.append(el("span","bn-chip",
    "watch: "+o.symbol+" "+(o.score%1?o.score.toFixed(1):o.score.toFixed(0)))));
  more.append(link());
  box.append(more);
}

function link(){
  const a=el("a",null,"open the scanner →");
  a.href="/fno";
  return a;
}

async function load(){
  try{
    const q = selDate ? "?date="+encodeURIComponent(selDate) : "";
    const r = await fetch("/api/data"+q, {cache:"no-store"});
    if(!r.ok) return;
    DATA = await r.json();
    selDate = DATA.date;
    render();
    document.getElementById("refreshed").textContent =
      "updated " + new Date().toLocaleTimeString("en-IN",{hour12:false});
    document.getElementById("foot").textContent =
      "Paper trading — no live orders. This dashboard keeps only the last "
      + (DATA.retention_days || 5) + " trading sessions on disk; older trades, "
      + "rejections, agent logs, journal reports and cached candles are pruned "
      + "automatically when the dashboard starts.";
  }catch(e){
    console.warn("Could not reach /api/data:", e);
  }
}

function render(){
  const d=DATA;
  if(!d) return;
  const winEl = document.getElementById("win");
  if(winEl) {
    winEl.textContent = `paper mode · LLM: ${d.provider} (${d.model}) · ${d.date||"no data"}`;
  }

  // date filter
  const sel=document.getElementById("dateSel");
  if(sel) {
    sel.replaceChildren();
    (d.dates.length?d.dates:[d.date||"no data"]).forEach(dt=>{
      const o=el("option",null,dt); o.value=dt; if(dt===d.date)o.selected=true; sel.append(o);
    });
    sel.onchange=()=>{selDate=sel.value; load();};
  }

  renderTiles(d); renderChart(d); renderBreakdown(d); renderCfg(d);
  renderTrades(d); renderRej(d); renderAlog(d);
  const rep=document.getElementById("report");
  if(rep) {
    rep.textContent = d.report || "No journal report for this date yet — run: python -m trading.agents.eod_journal";
  }
}

function renderTiles(d){
  const s=d.stats, box=document.getElementById("tiles"); box.replaceChildren();
  const mk=(label,value,opts={})=>{
    const c=el("div","kpi");
    c.append(el("div","k",label));
    const v=el("div","v mono"+(opts.hero?" hero":""),value);
    if(opts.dir) v.classList.add(opts.dir);
    c.append(v);
    if(opts.delta) {
      const dEl = el("div","f");
      if(opts.deltaHtml) dEl.innerHTML = opts.delta;
      else dEl.textContent = opts.delta;
      c.append(dEl);
    }
    box.append(c);
  };
  const net=s.net_pnl??0;
  const unp=s.unrealized_pnl??0;
  const rlp=s.realized_pnl??0;

  let deltaHtml = "";
  if(s.open > 0){
    deltaHtml = `Realized: ₹${fmt(rlp)} · <span class="live-pulse" style="color:${unp>=0?'var(--up)':'var(--down)'}">● Floating: ${unp>=0?'+':''}${fmt(unp)}</span>`;
  } else {
    deltaHtml = (net>=0?"▲ ":"▼ ") + "vs start of day";
  }

  mk("Total Net P&L (MTM)", (net>=0?"+":"")+fmt(net), {
    hero:true,
    dir:net>=0?"up":"down",
    delta:deltaHtml,
    deltaHtml:true
  });

  if(s.open > 0){
    mk("Live Floating P&L", (unp>=0?"+":"")+fmt(unp), {
      dir:unp>=0?"up":"down",
      delta:`across ${s.open} active open trade${s.open>1?'s':''}`
    });
  }

  mk("Win rate", s.win_rate==null?"—":fmt(s.win_rate,1)+"%");
  mk("Trades", String(s.trades)+(s.open?` (${s.open} open)`:""));
  mk("Charges (INR)", fmt(s.charges)+(s.est_open_charges?` (~₹${fmt(s.total_charges)} est)`:""));
  mk("Rejections today", String(s.rejections_today));
}

function renderCfg(d){
  const c=document.getElementById("cfg"); c.replaceChildren();
  if(!d.day_config){c.append(el("span","muted","No today_config.json — pre-market agent hasn't run.")); return;}
  const cfg=d.day_config;
  const add=(k,v)=>{const s=el("span"); const b=el("b",null,k+": "); s.append(b,String(v)); c.append(s);};
  add("date",cfg.date||"—"); add("regime",cfg.regime);
  add("risk multiplier",cfg.risk_multiplier);
  add("blocked",(cfg.blocked_symbols&&cfg.blocked_symbols.length)?cfg.blocked_symbols.join(", "):"none");
  const r=el("span","muted"); r.textContent="“"+(cfg.rationale||"")+"”"; c.append(r);
}

const VGLY={approve:"\u2713", caution:"\u25D4", veto:"\u2715", ok:"\u2713", err:"\u2715"};
function verdictBadge(v){
  if(!v) return el("span","muted","pending");
  const b=el("span","badge sm "+v);
  b.append(el("span","g", VGLY[v]||""));
  b.append(document.createTextNode(v));
  return b;
}

function renderTrades(d){
  const t=document.getElementById("trades");
  if(!t) return;
  t.replaceChildren();
  if(!d.trades.length){ t.append(el("caption","empty","No trades for this date.")); return; }
  const head=el("tr");
  ["#","time","symbol","side","qty","entry","exit / live LTP","P&L (realized / live)","charges","strategy","verdict","reasons"]
    .forEach((h,i)=>{const th=el("th",[4,5,6,7,8].includes(i)?"num":null,h); head.append(th);});
  t.append(head);
  d.trades.forEach(tr=>{
    const row=el("tr");
    row.append(el("td","mono",String(tr.id)));
    row.append(el("td","mono",t2time(tr.ts)));
    row.append(el("td",null,tr.symbol));
    row.append(el("td",null,tr.side));
    row.append(el("td","num",String(tr.qty)));
    row.append(el("td","num",fmt(tr.entry_price)));
    const isClosed = tr.status === "closed" && tr.exit_price != null;
    const exitTd = el("td","num");
    if(isClosed){
      exitTd.textContent = fmt(tr.exit_price);
    } else {
      const wrap = el("div");
      wrap.style.cssText = "display:flex;align-items:center;justify-content:flex-end;gap:6px";
      const ltpVal = tr.current_price != null ? tr.current_price : tr.entry_price;
      const ltpPill = el("span","badge sm live-pulse", "₹" + fmt(ltpVal));
      ltpPill.style.cssText = "font-family:ui-monospace,monospace;font-weight:700;padding:2px 6px;border-color:color-mix(in srgb,var(--accent) 55%,var(--line));color:var(--ink);background:var(--cell-2)";
      const cBtn = el("button","btn sm","Close");
      cBtn.style.padding = "2px 6px";
      cBtn.style.fontSize = "10.5px";
      cBtn.onclick = async ()=>{
        cBtn.disabled = true;
        cBtn.textContent = "Closing…";
        try{
          const res = await fetch("/api/close", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({trade_id: tr.id})
          });
          const j = await res.json();
          if(j.ok){ load(); } else { alert(j.error || "Failed to close trade"); cBtn.disabled=false; cBtn.textContent="Close"; }
        }catch(e){ alert("Error: "+e); cBtn.disabled=false; cBtn.textContent="Close"; }
      };
      wrap.append(ltpPill, cBtn);
      exitTd.append(wrap);
    }
    row.append(exitTd);

    const pnlTd = el("td","num");
    if(isClosed){
      const p = tr.pnl ?? 0;
      pnlTd.className = "num " + (p >= 0 ? "pnl-pos" : "pnl-neg");
      pnlTd.textContent = (p >= 0 ? "+" : "") + fmt(p);
    } else {
      const up = tr.unrealized_pnl ?? 0;
      const pct = tr.pnl_pct ?? 0;
      pnlTd.className = "num " + (up >= 0 ? "pnl-pos" : "pnl-neg");
      const valDiv = el("div", null, (up >= 0 ? "+" : "") + fmt(up));
      valDiv.style.fontWeight = "700";
      valDiv.style.fontFamily = "ui-monospace,monospace";
      const subDiv = el("div", "pnl-sub");
      subDiv.textContent = (pct >= 0 ? "+" : "") + fmt(pct, 2) + "% live";
      pnlTd.append(valDiv, subDiv);
    }
    row.append(pnlTd);

    const chgTd = el("td","num");
    if(isClosed){
      chgTd.textContent = tr.charges == null ? "—" : fmt(tr.charges);
    } else {
      chgTd.innerHTML = `<span class="muted" style="font-size:11px" title="Estimated round-trip brokerage & charges">~₹${fmt(tr.est_charges || 0)} <span style="font-size:9px">est</span></span>`;
    }
    row.append(chgTd);

    row.append(el("td","small",tr.strategy_id||"—"));
    const vtd=el("td"); vtd.append(verdictBadge(tr.agent_verdict));
    if(tr.agent_confidence!=null) vtd.append(el("span","small"," "+Number(tr.agent_confidence).toFixed(2)));
    row.append(vtd);
    let reasons="—";
    try{const rr=JSON.parse(tr.agent_reasons||"[]"); if(rr.length)reasons=rr.join("; ");}catch(e){}
    const rtd=el("td","small",reasons); rtd.style.maxWidth="320px"; row.append(rtd);
    t.append(row);
  });
}

function renderRej(d){
  const t=document.getElementById("rej"); t.replaceChildren();
  if(!d.rejections.length){t.append(el("caption","empty","No rejections logged."));return;}
  const head=el("tr");
  ["time","symbol","side","qty","reason"].forEach(h=>head.append(el("th",null,h)));
  t.append(head);
  d.rejections.forEach(r=>{
    let sig={}; try{sig=JSON.parse(r.signal_json);}catch(e){}
    const row=el("tr");
    row.append(el("td","mono",t2stamp(r.ts)));
    row.append(el("td",null,sig.symbol||"—"));
    row.append(el("td",null,sig.side||"—"));
    row.append(el("td","num",sig.qty!=null?String(sig.qty):"—"));
    row.append(el("td","small",r.reason));
    t.append(row);
  });
}

function renderAlog(d){
  const t=document.getElementById("alog"); t.replaceChildren();
  if(!d.agent_log.length){t.append(el("caption","empty","No LLM calls logged yet."));return;}
  const head=el("tr");
  ["time","agent","model","status","response / error"].forEach(h=>head.append(el("th",null,h)));
  t.append(head);
  d.agent_log.forEach(a=>{
    const row=el("tr");
    row.append(el("td","mono",t2stamp(a.ts)));
    row.append(el("td",null,a.agent));
    row.append(el("td","small",a.model||"—"));
    const st=el("td"); st.append(el("span","badge "+(a.ok?"ok":"err"), a.ok?"ok":"failed")); row.append(st);
    const msg=a.ok ? (a.response||"") : (a.error||"");
    const m=el("td","small", msg.length>200 ? msg.slice(0,200)+"…" : msg);
    m.style.maxWidth="420px"; row.append(m);
    t.append(row);
  });
}

function renderBreakdown(d){
  const fill=(tableId, rows, label)=>{
    const t=document.getElementById(tableId); t.replaceChildren();
    if(!rows || !rows.length){t.append(el("caption","empty","No trades for this date."));return;}
    const head=el("tr");
    [label,"trades","closed","open","win rate","charges","net P&L (incl. live)"].forEach((h,i)=>
      head.append(el("th", i>0?"num":null, h)));
    t.append(head);
    rows.forEach(g=>{
      const row=el("tr");
      const name=el("td",null, g.name==="BUY" ? "BUY (long)" : g.name==="SELL" ? "SELL (short)" : g.name);
      row.append(name);
      row.append(el("td","num",String(g.trades)));
      row.append(el("td","num",String(g.closed)));
      row.append(el("td","num",String(g.open||0)));
      row.append(el("td","num", g.win_rate==null?"—":fmt(g.win_rate,1)+"%"));
      row.append(el("td","num",fmt(g.charges)));
      const pnlTd = el("td","num "+(g.net_pnl>=0?"pnl-pos":"pnl-neg"));
      const pnlVal = el("div", null, (g.net_pnl>=0?"+":"")+fmt(g.net_pnl));
      pnlVal.style.fontWeight = "700";
      pnlTd.append(pnlVal);
      if(g.open > 0 && g.unrealized_pnl !== 0){
        const sub = el("div", "pnl-sub");
        sub.textContent = (g.unrealized_pnl >= 0 ? "+" : "") + fmt(g.unrealized_pnl) + " live";
        pnlTd.append(sub);
      }
      row.append(pnlTd);
      t.append(row);
    });
  };
  fill("sides", d.side_report, "side");
  fill("strats", d.strategy_report, "strategy");
}

// --- cumulative P&L line chart (SVG) ---
function renderChart(d){
  const svg=document.getElementById("chart");
  svg.replaceChildren();
  const closed=d.trades.filter(t=>t.status==="closed" && t.exit_ts!=null)
                       .sort((a,b)=>a.exit_ts-b.exit_ts);
  const W=svg.clientWidth||1040, H=240, M={t:16,r:20,b:26,l:56};
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const css=getComputedStyle(document.documentElement);
  const C={line:css.getPropertyValue("--series-1").trim(),
           grid:css.getPropertyValue("--grid").trim(),
           axis:css.getPropertyValue("--axis").trim(),
           muted:css.getPropertyValue("--muted").trim(),
           surface:css.getPropertyValue("--surface-1").trim(),
           ink:css.getPropertyValue("--ink").trim()};
  const ns="http://www.w3.org/2000/svg";
  const mk=(tag,attrs)=>{const e=document.createElementNS(ns,tag);
    for(const k in attrs)e.setAttribute(k,attrs[k]); return e;};

  if(!closed.length){
    svg.style.display="none";
    let note=document.getElementById("chartEmpty");
    if(!note){ note=el("div","empty","No closed trades in this session yet.");
      note.id="chartEmpty"; svg.parentNode.append(note); }
    note.style.display="";
    return;
  }
  svg.style.display="";
  const note=document.getElementById("chartEmpty");
  if(note) note.style.display="none";

  let cum=0;
  const pts=[{x:closed[0].ts, y:0, label:"start"}];
  closed.forEach(t=>{cum+=t.pnl||0; pts.push({x:t.exit_ts,y:cum,trade:t,cum});});

  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  const xmin=Math.min(...xs), xmax=Math.max(...xs)||xmin+1;
  let ymin=Math.min(0,...ys), ymax=Math.max(0,...ys);
  if(ymin===ymax){ymax+=1;}
  const pad=(ymax-ymin)*0.1; ymin-=pad; ymax+=pad;
  const X=v=> M.l+(v-xmin)/(xmax-xmin||1)*(W-M.l-M.r);
  const Y=v=> H-M.b-(v-ymin)/(ymax-ymin)*(H-M.t-M.b);

  // gridlines + clean y ticks
  const span=ymax-ymin, rawStep=span/4,
        mag=Math.pow(10,Math.floor(Math.log10(rawStep))),
        step=[1,2,5,10].map(m=>m*mag).find(s=>s>=rawStep)||mag*10;
  for(let v=Math.ceil(ymin/step)*step; v<=ymax; v+=step){
    svg.append(mk("line",{x1:M.l,x2:W-M.r,y1:Y(v),y2:Y(v),stroke:C.grid,"stroke-width":1}));
    const t=mk("text",{x:M.l-8,y:Y(v)+4,"text-anchor":"end",fill:C.muted,
      "font-size":"11","font-variant-numeric":"tabular-nums"});
    t.textContent=(Math.round(v)||0).toLocaleString("en-IN"); svg.append(t);
  }
  // zero baseline emphasized
  if(ymin<0&&ymax>0)
    svg.append(mk("line",{x1:M.l,x2:W-M.r,y1:Y(0),y2:Y(0),stroke:C.axis,"stroke-width":1}));
  // x baseline
  svg.append(mk("line",{x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b,stroke:C.axis,"stroke-width":1}));
  // x labels: first + last time
  [[pts[0],"start"],[pts[pts.length-1],"end"]].forEach(([p])=>{
    const t=mk("text",{x:X(p.x),y:H-8,"text-anchor":"middle",fill:C.muted,"font-size":"11"});
    t.textContent=t2time(p.x); svg.append(t);
  });

  const path=pts.map((p,i)=>(i?"L":"M")+X(p.x).toFixed(1)+","+Y(p.y).toFixed(1)).join(" ");
  // area wash ~10%
  svg.append(mk("path",{d:path+` L${X(xmax).toFixed(1)},${Y(Math.max(ymin,0)).toFixed(1)} L${X(xmin).toFixed(1)},${Y(Math.max(ymin,0)).toFixed(1)} Z`,
    fill:C.line,"fill-opacity":"0.1",stroke:"none"}));
  svg.append(mk("path",{d:path,fill:"none",stroke:C.line,"stroke-width":2,
    "stroke-linejoin":"round","stroke-linecap":"round"}));

  // end marker: ≥8px dot with 2px surface ring + end label (value at the end)
  const last=pts[pts.length-1];
  svg.append(mk("circle",{cx:X(last.x),cy:Y(last.y),r:6,fill:C.line,
    stroke:C.surface,"stroke-width":2}));
  const lbl=mk("text",{x:Math.min(X(last.x)+10,W-M.r),y:Y(last.y)+4,fill:C.ink,
    "font-size":"12","font-weight":"600"});
  lbl.textContent=fmt(last.y);
  if(X(last.x)+70>W-M.r){lbl.setAttribute("x",X(last.x)-10);lbl.setAttribute("text-anchor","end");}
  svg.append(lbl);

  // crosshair + tooltip (snap to nearest point)
  const cross=mk("line",{y1:M.t,y2:H-M.b,stroke:C.axis,"stroke-width":1,visibility:"hidden"});
  const dot=mk("circle",{r:5,fill:C.line,stroke:C.surface,"stroke-width":2,visibility:"hidden"});
  svg.append(cross,dot);
  const tip=document.getElementById("tooltip"), wrap=document.getElementById("chartwrap");
  const hover=mk("rect",{x:M.l,y:M.t,width:W-M.l-M.r,height:H-M.t-M.b,fill:"transparent"});
  svg.append(hover);
  const show=(clientX)=>{
    const box=svg.getBoundingClientRect(), sx=(clientX-box.left)*(W/box.width);
    let best=pts[0],bd=1e18;
    pts.forEach(p=>{const d0=Math.abs(X(p.x)-sx); if(d0<bd){bd=d0;best=p;}});
    cross.setAttribute("x1",X(best.x));cross.setAttribute("x2",X(best.x));
    cross.setAttribute("visibility","visible");
    dot.setAttribute("cx",X(best.x));dot.setAttribute("cy",Y(best.y));
    dot.setAttribute("visibility","visible");
    tip.replaceChildren();
    const v=el("div","v"); v.append(el("span","k"),document.createTextNode(fmt(best.y)+" INR cum."));
    tip.append(v);
    if(best.trade){
      tip.append(el("div","small",t2time(best.x)+" · "+best.trade.symbol+" "+best.trade.side+" ×"+best.trade.qty));
      tip.append(el("div","small","trade P&L: "+fmt(best.trade.pnl)));
    } else tip.append(el("div","small","session start"));
    tip.style.display="block";
    const wb=wrap.getBoundingClientRect();
    let lx=(X(best.x)/W)*wb.width+12;
    if(lx+160>wb.width) lx-=180;
    tip.style.left=lx+"px"; tip.style.top=(Y(best.y)/H)*240-10+"px";
  };
  hover.addEventListener("pointermove",e=>show(e.clientX));
  hover.addEventListener("pointerleave",()=>{tip.style.display="none";
    cross.setAttribute("visibility","hidden");dot.setAttribute("visibility","hidden");});
}

// ---------------- 4 AI AGENTS COUNCIL CONTROLS ----------------
const btnRunCouncilDebate = document.getElementById("btnRunCouncilDebate");
const councilSymSelect = document.getElementById("councilSymSelect");
const councilSymCustom = document.getElementById("councilSymCustom");
const councilMinConviction = document.getElementById("councilMinConviction");
const councilDebateResult = document.getElementById("councilDebateResult");
const councilModelBadge = document.getElementById("councilModelBadge");

const techStatus = document.getElementById("techStatus");
const techOutput = document.getElementById("techOutput");
const bullStatus = document.getElementById("bullStatus");
const bullOutput = document.getElementById("bullOutput");
const bearStatus = document.getElementById("bearStatus");
const bearOutput = document.getElementById("bearOutput");
const traderStatus = document.getElementById("traderStatus");
const traderOutput = document.getElementById("traderOutput");

async function runCouncilDebate(sym, minConv = 7) {
  if (!btnRunCouncilDebate) return;
  btnRunCouncilDebate.disabled = true;
  btnRunCouncilDebate.innerHTML = '<span class="spin"></span> 4-Agents Debating…';

  // Set agents into active thinking states
  techStatus.style.background = "var(--accent-soft)";
  techStatus.style.color = "var(--accent)";
  techStatus.innerHTML = '<span class="spin"></span> Analyzing OHLCV';
  techOutput.innerHTML = `<div style="color:var(--ink-2)"><span class="spin"></span> Fetching live candles & computing EMA 9/21/50, RSI, ATR for <b>${sym}</b>…</div>`;

  bullStatus.style.background = "var(--good-soft)";
  bullStatus.style.color = "var(--good)";
  bullStatus.innerHTML = '<span class="spin"></span> Building Thesis';
  bullOutput.innerHTML = `<div style="color:var(--ink-2)"><span class="spin"></span> Evaluating upside momentum, accumulation triggers, and catalysts…</div>`;

  bearStatus.style.background = "var(--bad-soft)";
  bearStatus.style.color = "var(--bad)";
  bearStatus.innerHTML = '<span class="spin"></span> Stress-Testing';
  bearOutput.innerHTML = `<div style="color:var(--ink-2)"><span class="spin"></span> Checking overhead supply, false breakout patterns, and downside risks…</div>`;

  traderStatus.style.background = "var(--warn-soft)";
  traderStatus.style.color = "var(--warn)";
  traderStatus.innerHTML = '<span class="spin"></span> Synthesizing';
  traderOutput.innerHTML = `<div style="color:var(--ink-2)"><span class="spin"></span> Weighing debate, calculating conviction score and risk-reward ratio…</div>`;

  councilDebateResult.style.display = "block";
  councilDebateResult.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;color:var(--ink-2)">
      <span class="spin"></span>
      <span><b>Council in Session:</b> Technical Analyst, Bullish Researcher, Bearish Researcher, and Trader Decision Agent are evaluating <b>${sym}</b>...</span>
    </div>
  `;

  try {
    const res = await fetch("/api/agents/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: sym, min_conviction: minConv })
    });
    const j = await res.json();
    if (!j.ok) {
      councilDebateResult.innerHTML = `<div class="banner" style="margin:0"><b>Council Evaluation Failed</b>${j.error || "Unknown error"}</div>`;
      techStatus.innerHTML = '● Error';
      bullStatus.innerHTML = '● Error';
      bearStatus.innerHTML = '● Error';
      traderStatus.innerHTML = '● Error';
      return;
    }

    const rec = j.recommendation;
    const ctx = j.context || {};
    const exec = j.execution;
    const isExecuted = exec && exec.executed;
    const conv = rec.conviction || 1;
    const convPct = (conv / 10) * 100;
    const convColor = conv >= 8 ? "var(--good)" : conv >= 6 ? "var(--accent)" : "var(--warn)";
    const actClass = rec.action === "BUY" ? "v-buy" : rec.action === "SELL" ? "veto" : "v-watch";

    // 1. Technical Analyst Card
    techStatus.innerHTML = '● Active • Evaluated';
    techStatus.style.background = "var(--accent-soft)";
    techStatus.style.color = "var(--accent)";
    techOutput.innerHTML = `
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
        <span class="bn-chip" style="color:var(--ink)">LTP: <b class="mono">₹${fmt(ctx.ltp || rec.entry_price)}</b></span>
        <span class="bn-chip" style="color:#38bdf8">OR: <b class="mono">₹${fmt(ctx.or_low)}-${fmt(ctx.or_high)} (${fmt(ctx.or_pct)}%)</b></span>
        <span class="bn-chip" style="color:#a855f7">1H 200MA: <b class="mono">₹${fmt(ctx.ma_200_1h)}</b></span>
        <span class="bn-chip" style="color:#a855f7">30m 200MA: <b class="mono">₹${fmt(ctx.ma_200_30m)}</b></span>
        <span class="bn-chip" style="color:#eab308">VWAP: <b class="mono">₹${fmt(ctx.session_vwap)} (${ctx.vwap_slope || 'FLAT'})</b></span>
        <span class="bn-chip">Vol: <b>${fmt(ctx.vol_ratio)}x (${ctx.vol_confirmed ? 'PASS' : 'LOW'})</b></span>
      </div>
      <div style="font-size:11px;color:var(--ink-2);margin-bottom:6px;padding:4px 8px;background:var(--cell);border-radius:6px;border:1px solid var(--line)">
        <b>Trend Gate:</b> ${ctx.trend_gate || 'N/A'} · <b>Trigger:</b> ${ctx.or_trigger_status || 'N/A'}
      </div>
      <div>${rec.technical_analysis || "Technical structure evaluated."}</div>
    `;

    // 2. Bullish Researcher Card
    bullStatus.innerHTML = '● Bull Thesis Ready';
    bullStatus.style.background = "var(--good-soft)";
    bullStatus.style.color = "var(--good)";
    bullOutput.innerHTML = `
      <div style="color:var(--good);font-weight:700;font-size:11.5px;margin-bottom:4px">🐂 Upside &amp; ORB Continuation Case:</div>
      <div>${rec.bull_case || "No significant upside thesis found."}</div>
    `;

    // 3. Bearish Researcher Card
    bearStatus.innerHTML = '● Risks Scrutinized';
    bearStatus.style.background = "var(--bad-soft)";
    bearStatus.style.color = "var(--bad)";
    bearOutput.innerHTML = `
      <div style="color:var(--bad);font-weight:700;font-size:11.5px;margin-bottom:4px">🐻 Downside Hazards &amp; False Breakout Audit:</div>
      <div>${rec.bear_case || "Downside stress-test complete."}</div>
    `;

    // 4. Trader Decision Agent Card
    traderStatus.innerHTML = `● Verdict: ${rec.action}`;
    traderStatus.style.background = rec.action === "BUY" ? "var(--good-soft)" : rec.action === "SELL" ? "var(--bad-soft)" : "var(--track)";
    traderStatus.style.color = rec.action === "BUY" ? "var(--good)" : rec.action === "SELL" ? "var(--bad)" : "var(--ink-2)";
    traderOutput.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span class="badge sm ${actClass}" style="font-size:11px;font-weight:700">${rec.action}</span>
        <span style="font-size:11px;font-weight:700;color:${convColor}">Conviction: ${conv}/10</span>
      </div>
      <div class="conviction-meter">
        <div class="conviction-fill" style="width:${convPct}%;background:${convColor}"></div>
      </div>
      <div style="margin-top:6px;font-size:11.5px">${rec.decision_rationale || "Multi-agent synthesis complete."}</div>
    `;

    // Council Summary Box
    councilDebateResult.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;padding-bottom:10px;border-bottom:1px solid var(--line)">
        <div style="display:flex;align-items:center;gap:10px">
          <span style="font-size:18px;font-weight:800;letter-spacing:-.01em;color:var(--ink)">${j.symbol}</span>
          <span class="badge ${actClass}" style="font-size:12px;font-weight:800">${rec.action}</span>
          <span class="badge sm ${conv >= minConv ? 'ok' : 'caution'}">Conviction ${conv}/10 (Threshold: ≥${minConv})</span>
          <span class="small mono" style="color:var(--ink-3)">Strategy: ORB + Multi-TF 200MA</span>
        </div>
        <div>
          ${isExecuted 
            ? `<span class="badge ok" style="padding:4px 10px;font-weight:700">✓ Executed Paper Trade #${exec.trade_id} (Qty: ${exec.qty})</span>` 
            : `<span class="badge sm" style="padding:4px 10px">${exec && exec.reason ? exec.reason : "No trade placed (below conviction threshold or HOLD)"}</span>`}
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-top:10px;background:var(--cell);padding:10px;border-radius:8px;border:1px solid var(--line)">
        <div><span class="small">Recommended Entry:</span> <b class="mono" style="font-size:14px;color:var(--ink);display:block;margin-top:2px">₹${fmt(rec.entry_price)}</b></div>
        <div><span class="small">Strict Structural Stop:</span> <b class="mono" style="font-size:14px;color:var(--bad);display:block;margin-top:2px">₹${fmt(rec.stop_loss)}</b></div>
        <div><span class="small">2.0 R:R Target:</span> <b class="mono" style="font-size:14px;color:var(--good);display:block;margin-top:2px">₹${fmt(rec.target)}</b></div>
        <div><span class="small">Risk-Reward Ratio:</span> <b class="mono" style="font-size:14px;color:var(--accent);display:block;margin-top:2px">${rec.stop_loss && rec.target && rec.entry_price ? (Math.abs((rec.target - rec.entry_price) / (rec.entry_price - rec.stop_loss || 1))).toFixed(2) + " : 1" : "2.0 : 1"}</b></div>
      </div>
    `;

    // Refresh dashboard tables and charts
    load();
  } catch (err) {
    councilDebateResult.innerHTML = `<div class="banner" style="margin:0"><b>Error running 4-agent debate:</b> ${err}</div>`;
  } finally {
    btnRunCouncilDebate.disabled = false;
    btnRunCouncilDebate.innerHTML = '⚡ Run 4-Agent Debate &amp; Auto-Trade';
  }
}

if (btnRunCouncilDebate) {
  btnRunCouncilDebate.onclick = () => {
    const custom = (councilSymCustom.value || "").trim().toUpperCase();
    const sym = custom || councilSymSelect.value;
    const minConv = parseInt(councilMinConviction.value, 10) || 7;
    runCouncilDebate(sym, minConv);
  };
}

// Lifecycle Agent Triggers
const btnRunPremarket = document.getElementById("btnRunPremarket");
if (btnRunPremarket) {
  btnRunPremarket.onclick = async () => {
    btnRunPremarket.disabled = true;
    btnRunPremarket.innerHTML = '<span class="spin"></span> Running…';
    try {
      const res = await fetch("/api/agents/premarket", { method: "POST" });
      const j = await res.json();
      if (j.ok) {
        alert("Pre-Market Analyst executed successfully! Daily regime config updated.");
        load();
      } else {
        alert("Pre-Market run failed: " + (j.error || "Unknown error"));
      }
    } catch (e) {
      alert("Error: " + e);
    } finally {
      btnRunPremarket.disabled = false;
      btnRunPremarket.innerHTML = 'Run Now';
    }
  };
}

const btnRunSupervisor = document.getElementById("btnRunSupervisor");
if (btnRunSupervisor) {
  btnRunSupervisor.onclick = async () => {
    btnRunSupervisor.disabled = true;
    btnRunSupervisor.innerHTML = '<span class="spin"></span> Auditing…';
    try {
      const res = await fetch("/api/agents/supervisor", { method: "POST" });
      const j = await res.json();
      if (j.ok) {
        alert("Supervisor audit completed! Trade verdicts updated.");
        load();
      } else {
        alert("Supervisor audit failed: " + (j.error || "Unknown error"));
      }
    } catch (e) {
      alert("Error: " + e);
    } finally {
      btnRunSupervisor.disabled = false;
      btnRunSupervisor.innerHTML = 'Audit Trades';
    }
  };
}

const btnRunJournal = document.getElementById("btnRunJournal");
if (btnRunJournal) {
  btnRunJournal.onclick = async () => {
    btnRunJournal.disabled = true;
    btnRunJournal.innerHTML = '<span class="spin"></span> Generating…';
    try {
      const res = await fetch("/api/agents/journal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date: selDate })
      });
      const j = await res.json();
      if (j.ok) {
        alert("EOD Journal generated successfully!");
        load();
      } else {
        alert("Journal generation failed: " + (j.error || "Unknown error"));
      }
    } catch (e) {
      alert("Error: " + e);
    } finally {
      btnRunJournal.disabled = false;
      btnRunJournal.innerHTML = 'Generate';
    }
  };
}

// Keep the secondary button inside the trades panel hooked up too
const btnNewTrade = document.getElementById("btnNewTrade");
const tradeForm = document.getElementById("tradeForm");
const btnCancelTrade = document.getElementById("btnCancelTrade");
const btnSubmitTrade = document.getElementById("btnSubmitTrade");
const tradeMsg = document.getElementById("tradeMsg");

const btnAiScan = document.getElementById("btnAiScan");
const aiScanDrawer = document.getElementById("aiScanDrawer");
const btnCancelAiScan = document.getElementById("btnCancelAiScan");
const btnStartAiScan = document.getElementById("btnStartAiScan");
const aiScanStatus = document.getElementById("aiScanStatus");

if (btnAiScan) {
  btnAiScan.onclick = () => {
    // Scroll directly to the 4 Agents Council Desk panel
    const desk = document.getElementById("agents-desk");
    if (desk) {
      desk.scrollIntoView({ behavior: "smooth" });
      desk.style.borderColor = "var(--accent)";
      setTimeout(() => { desk.style.borderColor = "color-mix(in srgb,var(--accent) 35%,var(--line))"; }, 1500);
    }
  };
}
if (btnCancelAiScan) {
  btnCancelAiScan.onclick = () => {
    if (aiScanDrawer) aiScanDrawer.style.display = "none";
  };
}
if (btnStartAiScan) {
  btnStartAiScan.onclick = () => {
    const custom = (document.getElementById("aiSymCustom")?.value || "").trim().toUpperCase();
    const sym = custom || document.getElementById("aiSymSelect")?.value || "SBIN";
    runCouncilDebate(sym, 7);
  };
}

if (btnNewTrade) {
  btnNewTrade.onclick = () => {
    tradeForm.style.display = tradeForm.style.display === "none" ? "block" : "none";
    tradeMsg.textContent = "";
  };
}
if (btnCancelTrade) {
  btnCancelTrade.onclick = () => {
    tradeForm.style.display = "none";
    tradeMsg.textContent = "";
  };
}
if (btnSubmitTrade) {
  btnSubmitTrade.onclick = async () => {
    btnSubmitTrade.disabled = true;
    tradeMsg.style.color = "var(--ink-2)";
    tradeMsg.textContent = "Executing order…";
    try {
      const sym = (document.getElementById("tSym").value || "").trim().toUpperCase();
      const side = document.getElementById("tSide").value;
      const qty = parseInt(document.getElementById("tQty").value, 10) || 1;
      const px = parseFloat(document.getElementById("tPx").value) || 0;
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({symbol: sym, side: side, qty: qty, price: px})
      });
      const j = await res.json();
      if (j.ok) {
        tradeMsg.style.color = "var(--good)";
        tradeMsg.textContent = `✓ Trade #${j.trade_id} executed!`;
        setTimeout(() => { tradeForm.style.display = "none"; load(); }, 700);
      } else {
        tradeMsg.style.color = "var(--bad)";
        tradeMsg.textContent = `✗ ${j.reason || j.error || "Order rejected"}`;
      }
    } catch (e) {
      tradeMsg.style.color = "var(--bad)";
      tradeMsg.textContent = `Error: ${e}`;
    } finally {
      btnSubmitTrade.disabled = false;
    }
  };
}

async function loadQuotes(){
  try{
    const r = await fetch("/api/quotes", {cache:"no-store"});
    if(!r.ok) return;
    const res = await r.json();
    const itemsBox = document.getElementById("liveTickerItems");
    const updatedEl = document.getElementById("liveTickerUpdated");
    if(!itemsBox) return;
    if(!res.quotes || !res.quotes.length){
      itemsBox.innerHTML = '<span class="muted" style="font-size:11.5px">Waiting for real-time market stream…</span>';
      return;
    }
    itemsBox.innerHTML = "";
    res.quotes.forEach(q => {
      const v = q.v || {};
      const rawName = (v.short_name || q.n || "").replace("-INDEX","").replace("-EQ","").replace("NSE:","");
      const lp = v.lp || 0;
      const ch = v.ch || 0;
      const chp = v.chp || 0;
      const isUp = ch >= 0;
      const card = el("div", "ticker-item");
      card.style.cssText = "background:var(--cell);border:1px solid var(--line);border-radius:8px;padding:7px 11px;min-width:142px;flex:0 0 auto;display:flex;flex-direction:column;gap:3px;box-shadow:var(--shadow)";
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px">
          <span style="font-weight:700;font-size:11px;letter-spacing:.04em;color:var(--ink)">${rawName}</span>
          <span class="badge sm ${isUp ? 'bull' : 'bear'}" style="font-size:9.5px;padding:1px 5px">${chp >= 0 ? '+' : ''}${chp.toFixed(2)}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-top:2px">
          <span style="font-size:14px;font-weight:700;font-family:ui-monospace,monospace;color:${isUp ? 'var(--up)' : 'var(--down)'}">₹${fmt(lp)}</span>
          <span style="font-size:10px;color:var(--ink-3);font-family:ui-monospace,monospace">${ch >= 0 ? '+' : ''}${fmt(ch)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:9.5px;color:var(--ink-3);margin-top:1px">
          <span>H: ${fmt(v.high_price)}</span>
          <span>L: ${fmt(v.low_price)}</span>
        </div>
      `;
      itemsBox.appendChild(card);
    });
    if(updatedEl){
      updatedEl.textContent = "⚡ Live: " + new Date().toLocaleTimeString("en-IN",{hour12:false});
    }
  }catch(e){
    console.warn("loadQuotes error:", e);
  }
}

load();
loadFno();
loadQuotes();
setInterval(load, 2000);
setInterval(loadFno, 2000);
setInterval(loadQuotes, 2000);
__THEMEJS__
window.addEventListener("resize", ()=>DATA&&renderChart(DATA));
</script>
</body></html>
"""


_TOPBAR_RIGHT = """    <select class="btn" id="dateSel" aria-label="Trading session"></select>
    <div class="live"><span class="led"></span><span id="refreshed">—</span></div>"""

PAGE = (_PAGE_TEMPLATE
        .replace("__THEME__", ui_theme.CSS)
        .replace("__TOPBAR__", ui_theme.topbar("Trading Agent Dashboard", "dashboard",
                                               _TOPBAR_RIGHT))
        .replace("__NIFTYJS__", ui_theme.NIFTY_JS)
        .replace("__CONTEXTJS__", ui_theme.CONTEXT_JS)
        .replace("__THEMEJS__", ui_theme.THEME_JS))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        # The F&O scanner page and its API live in trading.fno.web; they are
        # served from here too so one process covers both views. "/" is not
        # delegated — that is this dashboard's own page.
        fno = fno_web.serve(url.path, parse_qs(url.query))
        if fno is not None:
            return self._send(*fno)
        if url.path == "/api/data":
            q = parse_qs(url.query)
            body = json.dumps(get_data(q.get("date", [None])[0]),
                              default=str).encode()
            self._send(200, "application/json", body)
        elif url.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except Exception:
            body = {}

        if url.path in ("/api/ai-scan", "/api/agents/scan"):
            res = _handle_ai_scan(body)
            self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res).encode())
        elif url.path == "/api/agents/premarket":
            res = _handle_premarket_run()
            self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res).encode())
        elif url.path == "/api/agents/supervisor":
            res = _handle_supervisor_run()
            self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res).encode())
        elif url.path == "/api/agents/journal":
            res = _handle_journal_run(body)
            self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res).encode())
        elif url.path == "/api/trade":
            res = _handle_trade(body)
            self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res).encode())
        elif url.path == "/api/close":
            res = _handle_close(body)
            self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet
        pass


def main(argv: list[str] | None = None):
    import argparse

    from trading import retention

    ap = argparse.ArgumentParser(prog="python -m trading.dashboard")
    ap.add_argument("port", nargs="?", type=int, default=8080)
    ap.add_argument("--keep-days", type=int, default=RETENTION_DAYS,
                    help=f"trading sessions of history to keep (default {RETENTION_DAYS})")
    ap.add_argument("--no-prune", action="store_true",
                    help="leave the ledger alone — keep every session on disk")
    args = ap.parse_args(argv)

    if args.no_prune:
        print("retention: pruning disabled for this run (--no-prune)")
    else:
        print(retention.summary(retention.prune(args.keep_days)))

    fyers.start_background_server()
    fno_web.CACHE.start()
    server = fno_web._bind(args.port, Handler, "the dashboard")
    if server is None:
        return 1
    return fno_web._serve_forever(
        server, f"dashboard: http://localhost:{args.port}")


if __name__ == "__main__":
    main()
