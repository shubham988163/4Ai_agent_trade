"""Web UI for the F&O long scanner — "what can I buy right now, and why not?"

Served two ways:

    python -m trading.fno.web [port]     standalone on 8788
    python -m trading.dashboard          the same page at /fno

Scans are slow (a yfinance batch plus four NSE calls), so one runs in a
background thread and the page renders the cached result immediately, showing
its age. Nothing here re-derives a level: the page renders `report.as_dict`,
the same payload the CLI's --json prints, so the screen and the terminal can
never disagree about an entry or a stop.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from trading import ui_theme
from trading.fno import config as C
from trading.fno import fyers
from trading.fno import report
from trading.fno.live import run_scan
from trading.fno.models import IST

DEFAULT_TTL = 120          # seconds before a cached scan is considered stale


class ScanCache:
    """Runs at most one scan at a time and hands out the latest result.

    The UI must never block on the network, and two browser tabs must not
    trigger two concurrent scans, so state transitions are guarded by a lock
    and the worker is fire-and-forget.
    """

    def __init__(self, *, allow_delayed: bool = True, shortlist: int = C.SHORTLIST_SIZE,
                 symbols: list[str] | None = None, replay: str | None = None,
                 ttl: int = DEFAULT_TTL, fyers: bool = False,
                 fyers_base: str = "http://localhost:3001"):
        self.allow_delayed = allow_delayed
        self.shortlist = shortlist
        self.symbols = symbols
        self.replay = replay
        self.ttl = ttl
        self.fyers = fyers
        self.fyers_base = fyers_base
        self._lock = threading.Lock()
        self._scanning = False
        self._scan: dict | None = None
        self._error: str | None = None
        self._finished: datetime | None = None
        self._started: datetime | None = None

    # --- state ----------------------------------------------------------

    @property
    def age_sec(self) -> float | None:
        if self._finished is None:
            return None
        return (datetime.now(IST) - self._finished).total_seconds()

    def snapshot(self, refresh: bool = False) -> dict:
        stale = self.age_sec is None or self.age_sec > self.ttl
        if refresh or stale:
            self.start()
        with self._lock:
            return {
                "status": ("scanning" if self._scanning
                           else "error" if self._error and self._scan is None
                           else "ready" if self._scan else "idle"),
                "scanning": self._scanning,
                "error": self._error,
                "age_sec": None if self.age_sec is None else round(self.age_sec),
                "ttl": self.ttl,
                "started_at": self._started.isoformat() if self._started else None,
                "finished_at": self._finished.isoformat() if self._finished else None,
                "scan": self._scan,
            }

    def start(self) -> bool:
        with self._lock:
            if self._scanning:
                return False
            self._scanning = True
            self._started = datetime.now(IST)
        threading.Thread(target=self._run, daemon=True).start()
        return True

    # --- worker ---------------------------------------------------------

    def _run(self):
        try:
            scan = self._scan_now()
            with self._lock:
                self._scan, self._error = scan, None
        except Exception as exc:                     # noqa: BLE001
            # A feed outage must degrade to a visible message, never a blank page.
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._scanning = False
                self._finished = datetime.now(IST)

    def _scan_now(self) -> dict:
        res = run_scan(symbols=self.symbols, shortlist=self.shortlist,
                       replay=self.replay, allow_delayed=self.allow_delayed,
                       fyers=self.fyers, fyers_base=self.fyers_base)
        return report.as_dict(res)


CACHE = ScanCache()


def serve(path: str, query: dict | None = None, *,
          include_root: bool = False) -> tuple[int, str, bytes] | None:
    """Route one request. Returns (status, content-type, body) or None.

    Shared by this module's standalone server and trading.dashboard, so the
    page exists at exactly one implementation regardless of which port it is
    reached on. `include_root` is for the standalone server only — when this is
    mounted inside the dashboard, "/" belongs to the dashboard.
    """
    query = query or {}
    roots = ("/", "/fno") if include_root else ("/fno",)
    if path in roots:
        # Standalone, "/" is the scanner itself and there is no dashboard to
        # link to — the page hides that nav item rather than offering a link
        # that goes nowhere.
        mode = "standalone" if include_root else "mounted"
        return 200, "text/html; charset=utf-8", PAGE.replace("__MODE__", mode).encode()
    if path in ("/api/scan", "/api/fno"):
        refresh = query.get("refresh", ["0"])[0] == "1"
        body = json.dumps(CACHE.snapshot(refresh=refresh), default=str).encode()
        return 200, "application/json", body
    if path in ("/api/quotes", "/api/fno/quotes", "/api/fyers/quotes"):
        from trading.fno.fyers import FyersClient
        client = FyersClient()
        syms_param = query.get("symbols", [None])[0]
        if syms_param:
            syms = [s.strip() for s in syms_param.split(",") if s.strip()]
        else:
            default_syms = [
                "NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:RELIANCE-EQ",
                "NSE:INFY-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ",
                "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:ITC-EQ"
            ]
            cand_syms = []
            if CACHE._scan and CACHE._scan.get("candidates"):
                for c in CACHE._scan["candidates"][:15]:
                    s_name = f"NSE:{c['symbol']}-EQ"
                    if s_name not in default_syms and s_name not in cand_syms:
                        cand_syms.append(s_name)
            syms = default_syms + cand_syms
        quotes = client.quotes(syms)
        return 200, "application/json", json.dumps({"ok": True, "quotes": quotes}, default=str).encode()
    if path == "/api/fyers/status":
        from trading.fno.fyers import FyersClient, get_auth_link, FYERS_APP_ID
        client = FyersClient()
        connected, note = client.status()
        prof = {}
        try:
            p = client._get("/api/fyers/status")
            if p.get("profile"):
                prof = p.get("profile")
        except Exception:
            pass
        res = {
            "connected": connected,
            "profile": prof or ({"name": "SHUBHAM NARAYAN PANCHAL"} if connected else {}),
            "auth_url": get_auth_link(),
            "app_id": FYERS_APP_ID,
            "note": note,
        }
        return 200, "application/json", json.dumps(res, default=str).encode()
    if path == "/api/fyers/login":
        from trading.fno.fyers import get_auth_link
        auth_link = get_auth_link()
        html = f"""<!doctype html><html><head><meta http-equiv="refresh" content="0; url={auth_link}"></head><body>Redirecting to Fyers login...</body></html>"""
        return 200, "text/html; charset=utf-8", html.encode()
    if path == "/api/fyers/callback":
        from trading.fno.fyers import exchange_token
        auth_code = query.get("auth_code", [None])[0]
        if not auth_code:
            return 400, "text/html; charset=utf-8", b"<h3>Error: No auth_code received from Fyers.</h3>"
        try:
            data = exchange_token(auth_code)
            name = (data.get("profile") or {}).get("name") or "Trader"
            html = f"""<!doctype html>
<html><head><title>Fyers Connected</title>
<style>body{{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#e6edf3;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:32px;text-align:center;max-width:440px;box-shadow:0 8px 24px rgba(0,0,0,0.4);}}
h2{{color:#3fb950;margin-top:0;}}
a{{display:inline-block;margin-top:20px;background:#238636;color:#fff;text-decoration:none;padding:10px 20px;border-radius:6px;font-weight:600;}}</style>
</head><body><div class="card"><h2>&#10003; Fyers API Connected!</h2>
<p>Welcome, <b>{name}</b>. Zero-delay real-time market data is now active.</p>
<a href="/">Open Dashboard</a></div></body></html>"""
            return 200, "text/html; charset=utf-8", html.encode()
        except Exception as exc:
            return 500, "text/html; charset=utf-8", f"<h3>Failed to exchange token: {exc}</h3>".encode()
    return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        result = serve(url.path, parse_qs(url.query), include_root=True)
        if result is None:
            return self._send(404, "text/plain", b"not found")
        self._send(*result)

    def do_POST(self):  # noqa: N802
        try:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length)) if length > 0 else {}
            except Exception:
                body = {}

            if url.path == "/api/trade":
                from trading.dashboard import _handle_trade
                res = _handle_trade(body)
                self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res, default=str).encode())
            elif url.path == "/api/close":
                from trading.dashboard import _handle_close
                res = _handle_close(body)
                self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res, default=str).encode())
            elif url.path in ("/api/ai-scan", "/api/agents/scan"):
                from trading.dashboard import _handle_ai_scan
                res = _handle_ai_scan(body)
                self._send(200 if res.get("ok") else 400, "application/json", json.dumps(res, default=str).encode())
            else:
                self._send(404, "text/plain", b"not found")
        except Exception as exc:
            err = {"ok": False, "error": f"Internal server error: {exc}"}
            self._send(500, "application/json", json.dumps(err, default=str).encode())

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):   # quiet
        pass


_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>F&amp;O Long Scanner</title>
<style>
__THEME__

/* the hero tile leads the strip, so it gets more room than the rest */
.kpis{grid-template-columns:1.15fr repeat(4,1fr)}
@media (max-width:900px){.kpis{grid-template-columns:1fr 1fr}}
.bars{display:flex;gap:2px;margin-top:7px;height:5px}
.bars i{flex:1;border-radius:1px;background:var(--track)}
.bars i.on{background:var(--good)}

/* ---------- layout ---------- */
.grid{display:grid;grid-template-columns:minmax(0,1fr) 288px;gap:14px;margin-top:14px;
  align-items:start}
@media (max-width:1080px){.grid{grid-template-columns:1fr}}
.rail{display:grid;gap:12px;position:sticky;top:64px}
@media (max-width:1080px){.rail{position:static}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:11px;
  box-shadow:var(--shadow);overflow:hidden}
.panel > h2{margin:0;padding:9px 13px;font-size:10px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-3);border-bottom:1px solid var(--line);
  background:var(--cell);font-weight:600}
.panel .body{padding:11px 13px}

/* ---------- candidate card ---------- */
.card{position:relative;background:var(--card);border:1px solid var(--line);
  border-radius:12px;box-shadow:var(--shadow);margin-bottom:12px;overflow:hidden}
.card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--ink-3)}
.card.v-buy::before{background:var(--good)}
.card.v-watch::before{background:var(--warn)}
.card.v-avoid::before{background:var(--line-2)}
.card .hd{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;
  padding:12px 14px 11px 16px;border-bottom:1px solid var(--line)}
.idw{min-width:0;flex:1 1 220px}
.idw .row1{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.rank{font-size:10px;font-weight:700;color:var(--ink-3);letter-spacing:.08em}
.sym{font-size:17px;font-weight:700;letter-spacing:-.01em}
.chip{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-2);
  border:1px solid var(--line);background:var(--cell);border-radius:5px;padding:2px 6px;
  white-space:nowrap}
.status{font-size:11px;color:var(--ink-2);margin-top:5px}
.status .dotsep{color:var(--ink-3);margin:0 6px}
.pxw{text-align:right;flex:0 0 auto}
.pxw .px{font-size:20px;font-weight:650}
.pxw .chg{font-size:12px;font-weight:600;margin-top:1px}
.vw{flex:0 0 176px;display:flex;flex-direction:column;align-items:flex-end;gap:7px}

/* Verdict badge: glyph + word do the work; the tint only reinforces them. */
.badge{display:inline-flex;align-items:center;gap:6px;border-radius:7px;padding:5px 11px;
  font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  border:1px solid var(--line-2);background:var(--cell);color:var(--ink)}
.badge .g{font-size:12px;line-height:1}
.badge.v-buy{border-color:var(--good);background:var(--good-soft)}
.badge.v-buy .g{color:var(--good)}
.badge.v-watch{border-color:var(--warn);background:var(--warn-soft)}
.badge.v-watch .g{color:var(--warn)}
.badge.v-avoid .g{color:var(--ink-3)}
.badge.sm{padding:3px 8px;font-size:10px}
.badge.mkt.bull{border-color:var(--good);background:var(--good-soft)}
.badge.mkt.bull .g{color:var(--good)}
.badge.mkt.bear{border-color:var(--bad);background:var(--bad-soft)}
.badge.mkt.bear .g{color:var(--bad)}

.meter{width:176px}
.meter .r{display:flex;justify-content:space-between;font-size:10px;color:var(--ink-3);
  letter-spacing:.07em;text-transform:uppercase}
.meter .t{height:5px;border-radius:999px;background:var(--track);margin-top:4px;overflow:hidden}
.meter .f{height:100%;border-radius:999px;background:var(--accent)}

/* ---------- Candlestick & Strategy Chart ---------- */
.chart-container{padding:12px 16px 6px 16px;display:flex;flex-direction:column;gap:10px}
.candle-card{position:relative;background:var(--cell);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow:hidden;box-shadow:inset 0 1px 4px rgba(0,0,0,0.3)}
.candle-hud{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;padding-bottom:8px;border-bottom:1px solid var(--line);font-size:11px}
.candle-hud-title{display:flex;align-items:center;gap:6px;font-weight:700;color:var(--ink);letter-spacing:.04em;text-transform:uppercase;font-size:11px}
.candle-hud-vals{display:flex;align-items:center;gap:8px 12px;flex-wrap:wrap;color:var(--ink-2);font-family:ui-monospace,monospace;font-size:11px}
.candle-hud-vals span{display:inline-flex;align-items:baseline;gap:3px}
.candle-hud-vals b{color:var(--ink);font-weight:650}
.candle-hud-vals b.up{color:var(--good)}
.candle-hud-vals b.down{color:var(--bad)}
.candle-legend{display:flex;align-items:center;gap:10px 16px;flex-wrap:wrap;padding:7px 0 3px 0;font-size:10.5px;color:var(--ink-2);border-top:1px solid rgba(255,255,255,0.04);margin-top:6px}
.candle-legend-item{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.candle-legend-dot{width:8px;height:8px;border-radius:2px;display:inline-block}
.candle-legend-line{width:14px;height:0;border-top:2px solid;display:inline-block;vertical-align:middle}
.candle-legend-dashed{border-top-style:dashed}
.candle-svg-wrap{position:relative;width:100%;user-select:none}
.candle-svg-wrap svg{display:block;width:100%;height:220px;cursor:crosshair}
.candle-crosshair{pointer-events:none}

.ladder-section{padding:9px 12px 11px 12px;background:var(--cell);border:1px solid var(--line);border-radius:9px}
.ladder-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
.cap{font-size:9.5px;letter-spacing:.10em;text-transform:uppercase;color:var(--ink-3);
  margin-bottom:5px}

/* Price-vs-levels ladder — positioned HTML, so circles stay circles. */
.lad{position:relative;height:32px}
.lad .bar{position:absolute;left:0;right:0;top:13px;height:6px;border-radius:999px;
  background:var(--track)}
.lad .zone{position:absolute;top:13px;height:6px}
.lad .zone.risk{background:var(--bad-soft)}
.lad .zone.rew{background:var(--good-soft)}
.lad .zone.entry{top:11px;height:10px;border-radius:2px;background:var(--accent);opacity:.32}
.lad .rule{position:absolute;top:5px;width:2px;height:22px;margin-left:-1px}
.lad .cap-sq{position:absolute;left:-3px;bottom:-6px;width:8px;height:6px;background:currentColor}
.lad .cap-tri{position:absolute;left:-4px;top:-7px;width:0;height:0;
  border-left:5px solid transparent;border-right:5px solid transparent;
  border-bottom:7px solid currentColor}
.lad .dot{position:absolute;top:10px;width:12px;height:12px;margin-left:-6px;border-radius:50%;
  background:var(--accent);box-shadow:0 0 0 2px var(--card)}
.lgnd{display:flex;flex-wrap:wrap;gap:4px 13px;font-size:10.5px;color:var(--ink-2);margin-top:6px}
.lgnd i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;
  vertical-align:-1px}
.lgnd i.rnd{border-radius:50%}

/* ---------- plan ---------- */
.plan{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:1px;
  background:var(--line);border-top:1px solid var(--line);border-bottom:1px solid var(--line);
  margin-top:11px}
.plan .c{background:var(--cell);padding:9px 13px}
.plan .c.hl{background:var(--cell-2)}
.plan .k{font-size:9.5px;letter-spacing:.10em;text-transform:uppercase;color:var(--ink-3)}
.plan .v{font-size:15px;font-weight:650;margin-top:2px}
.plan .v.bad{color:var(--bad)} .plan .v.good{color:var(--good)}
.plan .n{font-size:10px;color:var(--ink-3);margin-top:1px}
.noplan{padding:10px 16px;font-size:12px;color:var(--ink-3);border-top:1px solid var(--line)}

/* ---------- options ---------- */
.opt{margin:12px 14px 0 16px;border:1px solid var(--line);border-radius:9px;
  background:var(--cell);overflow:hidden}
.opt .oh{display:flex;align-items:center;gap:9px;flex-wrap:wrap;padding:8px 11px;
  border-bottom:1px solid var(--line)}
.opt .ot{font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3)}
.opt .strike{font-size:15px;font-weight:650}
.opt .be{font-size:12px;color:var(--ink-2)}
.opt .be b{color:var(--ink);font-weight:600}
.opt .no{padding:9px 11px;font-size:12.5px;color:var(--ink-2)}
.opt .no b{color:var(--bad);font-weight:700;letter-spacing:.06em}
.opt table{font-size:11.5px}
.opt th,.opt td{padding:5px 11px}
.opt .oplan{display:grid;grid-template-columns:repeat(auto-fit,minmax(104px,1fr));
  gap:1px;background:var(--line);border-top:1px solid var(--line)}
.opt .oplan div{background:var(--card);padding:8px 11px}
.opt .oplan .k{font-size:9px;letter-spacing:.10em;text-transform:uppercase;color:var(--ink-3)}
.opt .oplan .v{font-size:14px;font-weight:650;margin-top:2px}
.opt .oplan .v.bad{color:var(--bad)} .opt .oplan .v.good{color:var(--good)}
.opt .oplan .n{font-size:10px;color:var(--ink-3);margin-top:1px}
.opt .model{padding:6px 11px;font-size:10.5px;color:var(--ink-3);
  border-top:1px solid var(--line)}
.opt .warns{padding:7px 11px;font-size:11.5px;color:var(--warn);
  border-top:1px solid var(--line)}
.opt .warns div{margin:2px 0}

/* ---------- reasons ---------- */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:12px 14px 2px 16px}
@media (max-width:760px){.cols{grid-template-columns:1fr}}
.hdr{font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);
  margin-bottom:6px}
ul.rs{margin:0;padding-left:15px}
ul.rs li{margin:3px 0;color:var(--ink-2);font-size:12.5px}
ul.rs.bad li{color:var(--bad)}
.metrics{display:flex;flex-wrap:wrap;gap:5px;padding:11px 14px 12px 16px}
.m{display:inline-flex;align-items:baseline;gap:5px;border:1px solid var(--line);
  background:var(--cell);border-radius:6px;padding:3px 8px;font-size:11px;color:var(--ink-3)}
.m b{color:var(--ink);font-weight:600;font-size:11.5px}
.warns{padding:0 14px 10px 16px;font-size:11.5px;color:var(--warn)}
.srcs{padding:8px 14px 11px 16px;font-size:10.5px;color:var(--ink-3);
  border-top:1px solid var(--line);word-break:break-word}

/* ---------- rail ---------- */
.sec{display:grid;grid-template-columns:64px 1fr 50px;align-items:center;gap:8px;
  font-size:11.5px;padding:3px 0}
.sec .nm{color:var(--ink-2);letter-spacing:.03em;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.sec .bb{position:relative;height:5px;background:var(--track);border-radius:999px}
.sec .bb span{position:absolute;top:0;height:100%;border-radius:999px}
.sec .bb::after{content:"";position:absolute;left:50%;top:-2px;width:1px;height:9px;
  background:var(--line-2)}
.sec .vv{text-align:right;font-size:11px}
.feed{display:flex;justify-content:space-between;gap:8px;font-size:11px;padding:3px 0;
  border-bottom:1px solid var(--line)}
.feed:last-child{border-bottom:0}
.feed .nm{color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.feed .ag{color:var(--ink-3);white-space:nowrap}
.feed .ag.warnc{color:var(--warn)}
.notes{margin:0;padding-left:15px;font-size:11.5px;color:var(--ink-3)}
.notes li{margin:4px 0}

/* ---------- table ---------- */
table{width:100%;border-collapse:collapse;font-size:12px}
th{position:sticky;top:0;background:var(--cell);color:var(--ink-3);font-weight:600;
  text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap;
  font-size:10px;letter-spacing:.09em;text-transform:uppercase}
td{padding:7px 10px;border-bottom:1px solid var(--line);color:var(--ink-2);white-space:nowrap}
td.sym{color:var(--ink);font-weight:600}
td.num,th.num{text-align:right}
tbody tr:hover td{background:var(--cell)}
td.blk{white-space:normal;min-width:220px}
.scroll{overflow-x:auto}
.empty{padding:16px;color:var(--ink-3);font-size:12.5px;text-align:center}
.foot{margin-top:18px;padding-top:12px;border-top:1px solid var(--line);
  font-size:11px;color:var(--ink-3);line-height:1.6}

/* ---------- AI Multi-Agent Intelligence Desk ---------- */
.agent-desk{margin-top:14px;background:linear-gradient(180deg,var(--card) 0%,rgba(14,23,40,0.85) 100%);
  border:1px solid var(--line-2);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
.agent-desk-hdr{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;
  background:var(--cell);border-bottom:1px solid var(--line);flex-wrap:wrap;gap:10px}
.agent-desk-title{display:flex;align-items:center;gap:10px;font-size:13.5px;font-weight:700;
  letter-spacing:0.04em;color:var(--ink)}
.agent-pulse{width:9px;height:9px;border-radius:50%;background:var(--good);
  box-shadow:0 0 10px var(--good);display:inline-block;animation:pulse-dot 2s infinite ease-in-out}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)}}
.agent-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:14px 16px}
@media (max-width:980px){.agent-grid{grid-template-columns:repeat(2,1fr)}}
@media (max-width:580px){.agent-grid{grid-template-columns:1fr}}
.agent-card{background:var(--cell);border:1px solid var(--line);border-radius:10px;padding:12px;
  display:flex;flex-direction:column;gap:8px;position:relative;transition:all .2s ease}
.agent-card:hover{transform:translateY(-2px);border-color:var(--accent);box-shadow:0 6px 16px rgba(0,0,0,0.3)}
.agent-card-hdr{display:flex;align-items:center;justify-content:space-between;gap:6px}
.agent-badge{font-size:9.5px;padding:2px 7px;border-radius:4px;font-weight:700;
  text-transform:uppercase;letter-spacing:0.05em}
.agent-name{font-size:12px;font-weight:700;color:var(--ink);display:flex;align-items:center;gap:6px}
.agent-model{font-size:10px;color:var(--ink-3);font-family:ui-monospace,monospace}
.agent-desc{font-size:11.5px;color:var(--ink-2);line-height:1.45;flex-grow:1}
.agent-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
.agent-tag{font-size:9.5px;padding:2px 5px;border-radius:4px;background:var(--card);
  border:1px solid var(--line);color:var(--ink-3);font-family:ui-monospace,monospace}

.agent-console{padding:12px 16px;border-top:1px solid var(--line);background:rgba(14,23,40,0.45);
  display:flex;flex-direction:column;gap:12px}
.agent-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.agent-input-wrap{display:flex;align-items:center;gap:6px;background:var(--cell);
  border:1px solid var(--line);border-radius:8px;padding:4px 8px}
.agent-input{background:transparent;border:none;color:var(--ink);font-size:13px;
  font-weight:600;width:130px;outline:none;text-transform:uppercase;font-family:ui-monospace,monospace}
.agent-chips{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.agent-chip{padding:3px 8px;border-radius:6px;border:1px solid var(--line);background:var(--cell);
  color:var(--ink-2);font-size:11px;font-weight:600;cursor:pointer;transition:all 0.15s ease;
  font-family:ui-monospace,monospace}
.agent-chip:hover{background:var(--cell-2);border-color:var(--accent);color:var(--ink)}

.agent-debate-output{display:none;background:var(--cell);border:1px solid var(--line-2);
  border-radius:10px;padding:14px;margin-top:6px;animation:fade-in 0.25s ease}
@keyframes fade-in{from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)}}
.debate-verdict-hdr{display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:10px;padding-bottom:12px;border-bottom:1px solid var(--line)}
.debate-scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:12px 0}
.debate-stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.debate-stat .lbl{font-size:10px;text-transform:uppercase;color:var(--ink-3);letter-spacing:0.05em}
.debate-stat .val{font-size:15px;font-weight:700;font-family:ui-monospace,monospace;margin-top:2px}
.debate-perspectives{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:10px}
@media (max-width:800px){.debate-perspectives{grid-template-columns:1fr}}
.perspective-card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:11px 13px}
.perspective-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;
  margin-bottom:6px;display:flex;align-items:center;gap:6px}
.perspective-body{font-size:12px;color:var(--ink-2);line-height:1.55}
</style></head>
<body data-mode="__MODE__">
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

  <div id="banner"></div>
  <div class="kpis" id="kpis"></div>

  <div id="niftyWrap"></div>

  <!-- AI Multi-Agent Intelligence Desk -->
  <section class="agent-desk" id="agentDeskPanel">
    <div class="agent-desk-hdr">
      <div>
        <div class="agent-desk-title">
          <span class="agent-pulse"></span>
          <span>⚡ AI Multi-Agent Intelligence Desk</span>
          <span class="agent-badge" style="background:var(--accent-soft);color:var(--accent);border:1px solid rgba(47,159,219,.3)">Live Consensus Engine</span>
        </div>
        <div style="font-size:11px;color:var(--ink-3);margin-top:3px">
          Autonomous 4-Agent Debate Council powered by <strong>Google Gemini 3.5 Flash Lite</strong> + <strong>Fyers API v3 Tick Stream</strong>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="font-size:11px;color:var(--ink-2);font-family:ui-monospace,monospace">Status: <strong style="color:var(--good)">4 Agents Online</strong></span>
        <button class="btn" id="btnToggleAgents" style="font-size:11px;padding:3px 9px" onclick="toggleAgentDesk()">Minimize Desk</button>
      </div>
    </div>

    <div id="agentDeskBody">
      <div class="agent-grid">
        <!-- Agent 1 -->
        <div class="agent-card">
          <div class="agent-card-hdr">
            <span class="agent-name">📊 Technical Analyst</span>
            <span class="agent-badge" style="background:var(--good-soft);color:var(--good)">Active</span>
          </div>
          <div class="agent-model">Gemini 3.5 · ORB &amp; Trend Engine</div>
          <div class="agent-desc">
            Audits the 09:15-09:30 Opening Range (0.3%-1.5% gate), verifies the Dual Multi-Timeframe 200 SMA trend gate on 1H &amp; 30m, session VWAP slope, and volume surge.
          </div>
          <div class="agent-tags">
            <span class="agent-tag">ORB 09:15-09:30</span>
            <span class="agent-tag">1H/30m 200 SMA</span>
            <span class="agent-tag">VWAP Slope</span>
            <span class="agent-tag">Vol Spike ≥1.5x</span>
          </div>
        </div>

        <!-- Agent 2 -->
        <div class="agent-card">
          <div class="agent-card-hdr">
            <span class="agent-name">🐂 Bullish Researcher</span>
            <span class="agent-badge" style="background:var(--good-soft);color:var(--good)">Active</span>
          </div>
          <div class="agent-model">Gemini 3.5 · ORB Long Specialist</div>
          <div class="agent-desc">
            Argues the Long breakout thesis ONLY when price is strictly above BOTH 1H &amp; 30m 200 SMAs, with fresh candle close above OR-High and volume expansion.
          </div>
          <div class="agent-tags">
            <span class="agent-tag">Uptrend Gate (LTP > 200MA)</span>
            <span class="agent-tag">OR-H Breakout</span>
            <span class="agent-tag">Volume Surge ≥1.5x</span>
          </div>
        </div>

        <!-- Agent 3 -->
        <div class="agent-card">
          <div class="agent-card-hdr">
            <span class="agent-name">🐻 Bearish Researcher</span>
            <span class="agent-badge" style="background:var(--bad-soft);color:var(--bad)">Auditing</span>
          </div>
          <div class="agent-model">Gemini 3.5 · False Breakout Auditor</div>
          <div class="agent-desc">
            Stress-tests setups for false breakout wicks, range disqualification (&lt;0.3% or &gt;1.5%), overhead 200 MAs, or short breakdown setups below OR-Low.
          </div>
          <div class="agent-tags">
            <span class="agent-tag">Downtrend Gate (LTP < 200MA)</span>
            <span class="agent-tag">OR-L Breakdown</span>
            <span class="agent-tag">Range Trap Audit</span>
          </div>
        </div>

        <!-- Agent 4 -->
        <div class="agent-card">
          <div class="agent-card-hdr">
            <span class="agent-name">⚖️ Decision Desk Chair</span>
            <span class="agent-badge" style="background:var(--accent-soft);color:var(--accent)">Supervisor</span>
          </div>
          <div class="agent-model">Gemini 3.5 · ORB + 200MA Kernel</div>
          <div class="agent-desc">
            Strictly enforces the 6 ORB + 200MA rules. Dispatches BUY/SELL orders with structural ATR stops and 2.0 R:R only if conviction ≥7/10; otherwise mandates HOLD.
          </div>
          <div class="agent-tags">
            <span class="agent-tag">ORB Rulebook</span>
            <span class="agent-tag">Structural ATR Stop</span>
            <span class="agent-tag">2.0 R:R Target</span>
            <span class="agent-tag">Conviction ≥ 7/10</span>
          </div>
        </div>
      </div>

      <!-- Interactive Debate Console -->
      <div class="agent-console">
        <div class="agent-bar">
          <span style="font-size:11.5px;font-weight:700;color:var(--ink-2);text-transform:uppercase;letter-spacing:.05em">Summon AI Committee:</span>
          <div class="agent-input-wrap">
            <span style="color:var(--ink-3);font-size:11px">NSE:</span>
            <input type="text" id="aiAgentSymInput" class="agent-input" placeholder="e.g. RELIANCE" value="SBIN" />
          </div>
          <div class="agent-chips" id="agentQuickChips">
            <span class="agent-chip" onclick="setAiSymbol('RELIANCE')">RELIANCE</span>
            <span class="agent-chip" onclick="setAiSymbol('SBIN')">SBIN</span>
            <span class="agent-chip" onclick="setAiSymbol('HDFCBANK')">HDFCBANK</span>
            <span class="agent-chip" onclick="setAiSymbol('ICICIBANK')">ICICIBANK</span>
            <span class="agent-chip" onclick="setAiSymbol('INFY')">INFY</span>
            <span class="agent-chip" onclick="setAiSymbol('TCS')">TCS</span>
          </div>
          <div style="margin-left:auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <label class="chk" style="font-size:11px;color:var(--ink-2);display:flex;align-items:center;gap:5px;cursor:pointer">
              <input type="checkbox" id="chkAutoAiDebate" onchange="toggleAutoAiDebate(this.checked)"> Auto-Debate Scanned Stocks
            </label>
            <button class="btn go" id="btnRunAgentDebate" style="font-weight:600;padding:6px 14px" onclick="triggerAgentDebate()">
              ⚡ Run AI Council Debate
            </button>
          </div>
        </div>

        <!-- Live Debate Progress Status -->
        <div id="aiDebateLoading" style="display:none;padding:12px;background:var(--cell);border-radius:8px;border:1px solid var(--line);align-items:center;gap:12px">
          <div class="led" style="background:var(--accent);box-shadow:0 0 10px var(--accent);animation:pulse-dot 1s infinite"></div>
          <div style="flex-grow:1">
            <div id="aiDebateStep" style="font-size:12px;font-weight:600;color:var(--ink)">Gathering Fyers live quotes and technicals…</div>
            <div style="font-size:10.5px;color:var(--ink-3);margin-top:2px">Consulting Technical Analyst, Bullish Researcher, Bearish Researcher, and Trader Desk</div>
          </div>
        </div>

        <!-- Live Debate Output Result -->
        <div id="aiDebateResult" class="agent-debate-output"></div>
      </div>
    </div>
  </section>

  <div class="grid">
    <main id="list"></main>
    <aside class="rail">
      <section class="panel" id="secpanel" hidden>
        <h2>Sector strength</h2>
        <div class="body"><div class="cap" id="sectitle"></div>
          <div id="sectors"></div></div>
      </section>
      <section class="panel">
        <h2>Data feeds</h2>
        <div class="body" id="feeds"></div>
      </section>
      <section class="panel">
        <h2>Window &amp; data notes</h2>
        <div class="body"><ul class="notes" id="notes"></ul></div>
      </section>
    </aside>
  </div>

  <section class="panel" style="margin-top:14px">
    <h2>All screened names — score, verdict, and what is blocking each</h2>
    <div class="scroll"><table id="tbl"></table></div>
  </section>

  <div class="foot">
    Decision support only — not investment advice, and no profit is implied or
    guaranteed. A score counts how many independent factors currently agree; it is
    not a probability of profit. Verify every level on your own terminal before
    acting. This page places no orders.
  </div>
</div>

<script>
const API = location.pathname.replace(/\/+$/,"") === "/fno" ? "/api/fno" : "/api/scan";
const $ = s => document.querySelector(s);
const el = (t,c,x) => { const n=document.createElement(t); if(c) n.className=c;
  if(x!==undefined) n.textContent=x; return n; };
const num = v => v===null||v===undefined||!isFinite(v) ? "—"
  : v.toLocaleString("en-IN",{minimumFractionDigits:2,maximumFractionDigits:2});
const pc = v => v===null||v===undefined ? "—" : (v>=0?"+":"")+v.toFixed(2)+"%";
const sgn = v => v===null||v===undefined ? "" : v>0 ? "up" : v<0 ? "down" : "";
const sc = v => v%1 ? v.toFixed(1) : v.toFixed(0);
const GLY = {BUY:"✓", WATCH:"◔", AVOID:"✕", bull:"▲",
             bear:"▼", flat:"—"};

let STATE=null, FILTER="all", TIMER=null;

async function load(force){
  try{
    const r = await fetch(API + (force?"?refresh=1":""), {cache:"no-store"});
    STATE = await r.json();
  }catch(e){ STATE = {status:"error", error:String(e), scan:null, scanning:false}; }
  render();
  clearTimeout(TIMER);
  TIMER = setTimeout(()=>load(false), STATE.scanning ? 2000 : ($("#auto").checked ? 2000 : 30000));
}

function render(){
  if(!STATE) return;
  const s = STATE.scan;
  const busy = !!STATE.scanning;
  $("#rescan").disabled = busy;
  $("#rescan").innerHTML = "";
  if(busy){ $("#rescan").append(el("span","spin")); $("#rescan").append(document.createTextNode("Scanning")); }
  else $("#rescan").textContent = "Scan now";
  $("#led").className = "led" + (busy ? " busy" : (s && !s.data_ok) ? " stale" : "");
  $("#stamp").textContent = STATE.age_sec===null ? "no scan yet"
    : busy ? "scanning" : "updated " + (STATE.age_sec<60 ? STATE.age_sec+"s"
        : Math.round(STATE.age_sec/60)+"m") + " ago";

  $("#banner").innerHTML = "";
  if(!s){
    $("#win").textContent = STATE.error ? "scan failed" : "running first scan…";
    if(STATE.error) banner("Scan failed", STATE.error);
    return;
  }
  $("#win").textContent = s.when_label + "  ·  " + s.window;
  if(s.stale_banner) banner(s.stale_banner,
    "No setup is graded and no entry, stop or target is produced.");
  else if(STATE.error) banner("Last refresh failed — showing the previous scan", STATE.error);

  kpis(s); nifty(s); sectors(s); feeds(s); notes(s); counts(s); list(s); table(s); updateQuickChips(s);
  if(window.ScannerAlerts){
    (s.picks||[]).forEach(p => window.ScannerAlerts.checkAndAlert(p, "BUY"));
    (s.candidates||[]).filter(c=>c.verdict==="WATCH").forEach(c => window.ScannerAlerts.checkAndAlert(c, "WATCH"));
  }
}

__NIFTYJS__
__CONTEXTJS__

function nifty(s){
  const wrap = $("#niftyWrap");
  wrap.replaceChildren();
  if(s.index_options) wrap.append(niftyPanel(s.index_options, num));
  if(s.banknifty_options) wrap.append(niftyPanel(s.banknifty_options, num));
  const ctx = contextPanel(s.chain, s.pivots, num);
  if(ctx) wrap.append(ctx);
}

function banner(t,d){
  const b = el("div","banner"); b.append(el("b",null,t));
  if(d) b.append(el("div",null,d)); $("#banner").append(b);
}

function counts(s){
  const all = s.candidates||[];
  $("#n-all").textContent = all.length || "";
  ["BUY","WATCH","AVOID"].forEach(v =>
    $("#n-"+v).textContent = all.filter(c=>c.verdict===v).length || "");
}

/* ---------- KPI strip ---------- */
function kpis(s){
  const k = $("#kpis"); k.innerHTML="";
  const m = s.market, buys = (s.picks||[]).length;
  const watch = (s.candidates||[]).filter(c=>c.verdict==="WATCH").length;

  const hero = el("div","kpi");
  hero.append(el("div","k","Buyable long setups"));
  const early = s.signals_allowed === false;
  const v = el("div","v hero mono",
    !s.data_ok ? "\u2014" : early ? "\u2014" : String(buys));
  if(s.data_ok && !early && buys) v.style.color = "var(--good)";
  hero.append(v);
  hero.append(el("div","f", !s.data_ok ? "data unreliable — nothing graded"
    : early ? (s.window_note || "this window does not emit signals")
    : buys ? "clearing every rule" : (s.no_trade||"no qualifying setup")
      + (watch ? " · " + watch + " on watch" : "")));
  k.append(hero);

  const mk = el("div","kpi");
  mk.append(el("div","k","Market"));
  const holder = el("div"); holder.style.marginTop="7px";
  const cls = m.classification.indexOf("BULL")>=0 ? "bull"
            : m.classification.indexOf("BEAR")>=0 ? "bear" : "flat";
  holder.append(badge(m.classification, "mkt " + cls, GLY[cls]));
  mk.append(holder);
  mk.append(el("div","f", m.nifty_above_vwap===null ? "NIFTY VWAP unavailable"
    : m.nifty_above_vwap ? "NIFTY above VWAP" : "NIFTY below VWAP"));
  k.append(mk);

  k.append(kpi("NIFTY 50", pc(m.nifty_pct), sgn(m.nifty_pct), "kpi-nifty"));
  k.append(kpi("Bank Nifty", pc(m.banknifty_pct), sgn(m.banknifty_pct), "kpi-banknifty"));

  const br = el("div","kpi");
  br.append(el("div","k","Breadth"));
  br.append(el("div","v mono", m.breadth_pct===null ? "—"
    : Math.round(m.breadth_pct)+"%"));
  if(m.breadth_pct!==null){
    const bars = el("div","bars");
    for(let i=0;i<10;i++){ const b=el("i"); if(i < Math.round(m.breadth_pct/10)) b.className="on";
      bars.append(b); }
    br.append(bars);
  }
  br.append(el("div","f", m.advances!==null && m.advances!==undefined
    ? m.advances+" up / "+m.declines+" down" : m.breadth_source));
  k.append(br);
}
function kpi(label, value, cls, id){
  const t = el("div","kpi");
  t.append(el("div","k",label));
  const v = el("div","v mono "+(cls||""),value);
  if(id) v.id = id;
  t.append(v);
  return t;
}
function badge(text, cls, glyph){
  const b = el("span","badge "+(cls||""));
  b.append(el("span","g", glyph || GLY[text] || ""));
  b.append(document.createTextNode(text));
  return b;
}

/* ---------- rail ---------- */
function sectors(s){
  const box = $("#sectors"), m = s.market;
  const entries = Object.entries(m.sectors||{});
  $("#secpanel").hidden = !entries.length;
  if(!entries.length) return;
  $("#sectitle").textContent = m.sector_source;
  box.innerHTML = "";
  const max = Math.max(...entries.map(([,v])=>Math.abs(v)), 0.5);
  entries.sort((a,b)=>b[1]-a[1]).forEach(([name,v])=>{
    const row = el("div","sec");
    row.append(el("div","nm",name));
    const bb = el("div","bb"), f = el("span");
    const w = Math.abs(v)/max*50;
    f.style.width = w+"%"; f.style.left = (v>=0?50:50-w)+"%";
    f.style.background = v>=0 ? "var(--up)" : "var(--down)";
    bb.append(f); row.append(bb);
    row.append(el("div","vv mono "+sgn(v), pc(v)));
    box.append(row);
  });
}

function feeds(s){
  const box = $("#feeds"); box.innerHTML="";
  const all = (s.market.sources||[]).concat(
    ...(s.candidates||[]).map(c=>c.sources||[]));
  // Group by feed, not by instrument: 12 rows of the same source is noise.
  // The age shown is the worst in the group — feed health is its slowest part.
  const groups = new Map();
  all.forEach(p=>{
    const key = p.source + "|" + p.timeframe;
    const g = groups.get(key) || {source:p.source, timeframe:p.timeframe,
                                  delayed:false, age:0, n:0};
    g.delayed = g.delayed || p.delayed;
    g.age = Math.max(g.age, p.age_min);
    g.n += 1;
    groups.set(key, g);
  });
  [...groups.values()].forEach(g=>{
    const row = el("div","feed");
    row.append(el("div","nm", g.source + " · " + g.timeframe
      + (g.n > 1 ? "  ×" + g.n : "")));
    row.append(el("div","ag mono" + (g.delayed ? " warnc" : ""),
      (g.delayed ? "DELAYED " : "") + g.age.toFixed(0) + "m"));
    box.append(row);
  });
  if(!groups.size) box.append(el("div","notes","no feeds reported"));
}

function notes(s){
  const ul = $("#notes"); ul.innerHTML="";
  (s.data_notes||[]).forEach(t=>ul.append(el("li",null,t)));
}

/* ---------- candidate list ---------- */
function list(s){
  const box = $("#list"); box.innerHTML="";
  let items = (s.candidates||[]).slice();
  if(FILTER!=="all") items = items.filter(c=>c.verdict===FILTER);
  if(!items.length){
    const p = el("section","panel");
    p.append(el("div","empty", s.data_ok
      ? (FILTER==="all" ? "No name cleared the liquidity screen this run."
                        : "Nothing in this view.")
      : "Nothing graded — see the banner above."));
    box.append(p); return;
  }
  const ord = {BUY:0, WATCH:1, AVOID:2};
  items.sort((a,b)=>(ord[a.verdict]-ord[b.verdict]) || (b.score-a.score));
  items.forEach(c=>box.append(card(c)));
}

function card(c){
  const v = c.verdict.toLowerCase();
  const box = el("article","card v-"+v);
  box.dataset.cardSym = c.symbol;

  const hd = el("div","hd");
  const idw = el("div","idw");
  const r1 = el("div","row1");
  if(c.rank) r1.append(el("span","rank mono", String(c.rank).padStart(2,"0")));
  r1.append(el("span","sym", c.symbol));
  r1.append(el("span","chip", c.sector));
  r1.append(el("span","chip", c.contract));
  idw.append(r1);
  const st = el("div","status");
  st.append(document.createTextNode(c.structure.status));
  if(c.structure.breakout_time){
    st.append(el("span","dotsep","·"));
    st.append(document.createTextNode("broke out " + c.structure.breakout_time));
  }
  idw.append(st);
  hd.append(idw);

  const pxw = el("div","pxw");
  pxw.append(el("div","px mono", "₹" + num(c.price)));
  pxw.append(el("div","chg mono "+sgn(c.pct_change), pc(c.pct_change)));
  hd.append(pxw);

  const vw = el("div","vw");
  vw.append(badge(c.verdict, "v-"+v));
  const m = el("div","meter");
  const r = el("div","r");
  r.append(el("span",null,"Score"));
  r.append(el("span","mono", sc(c.score)+"/100 · "+c.grade));
  m.append(r);
  const t = el("div","t"), f = el("div","f");
  f.style.width = Math.max(0,Math.min(100,c.score))+"%";
  t.append(f); m.append(t);

  const btnsWrap = el("div");
  btnsWrap.style.display = "flex";
  btnsWrap.style.gap = "6px";
  btnsWrap.style.marginTop = "6px";
  btnsWrap.style.alignItems = "center";
  btnsWrap.style.flexWrap = "wrap";

  const tBtn = el("button","btn go","⚡ Take Trade");
  tBtn.style.fontSize = "11px";
  tBtn.style.padding = "3px 9px";
  tBtn.onclick = async (e)=>{
    e.stopPropagation();
    tBtn.disabled = true;
    tBtn.textContent = "Placing…";
    try{
      const entryPx = c.price || (c.trade ? c.trade.entry : 0);
      const slPx = c.trade ? c.trade.stop : (entryPx * 0.99);
      const tgtPx = c.trade ? (c.trade.target1 || c.trade.target) : (entryPx * 1.02);
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          symbol: c.symbol,
          side: "BUY",
          qty: 15,
          price: entryPx,
          stop_loss: slPx,
          target: tgtPx,
          strategy_id: "fno_scanner"
        })
      });
      const j = await res.json();
      if(j.ok){
        tBtn.style.background = "var(--good)";
        tBtn.textContent = "✓ Trade #" + j.trade_id + " Placed!";
        setTimeout(()=>{ tBtn.textContent="⚡ Take Trade"; tBtn.disabled=false; tBtn.style.background=""; }, 2000);
      } else {
        alert("Trade rejected: " + (j.reason || j.error || "Unknown"));
        tBtn.disabled = false;
        tBtn.textContent = "⚡ Take Trade";
      }
    }catch(err){
      alert("Error: " + err);
      tBtn.disabled = false;
      tBtn.textContent = "⚡ Take Trade";
    }
  };
  btnsWrap.append(tBtn);

  if(c.options && c.options.quote){
    const optQ = c.options.quote;
    const kind = optQ.option_type === "Put" ? "PE" : "CE";
    const optBtn = el("button","btn go","🎯 Buy " + optQ.strike + " " + kind);
    optBtn.style.fontSize = "11px";
    optBtn.style.padding = "3px 9px";
    optBtn.style.background = "linear-gradient(135deg, #10b981, #059669)";
    optBtn.onclick = async (e)=>{
      e.stopPropagation();
      optBtn.disabled = true;
      optBtn.textContent = "Placing…";
      try {
        const res = await fetch("/api/trade", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            symbol: optQ.identifier || c.symbol + optQ.strike + kind,
            underlying: c.symbol,
            side: "BUY",
            is_option: true,
            option_type: kind,
            price: optQ.ltp,
            stop_loss: c.options.premium_at_stop || (optQ.ltp * 0.7),
            target: c.options.premium_at_t1 || (optQ.ltp * 1.5),
            strategy_id: "fno_stock_option"
          })
        });
        const j = await res.json();
        if(j.ok){
          optBtn.style.background = "var(--good)";
          optBtn.textContent = "✓ Option #" + j.trade_id + "!";
          setTimeout(()=>{ optBtn.textContent="🎯 Buy " + optQ.strike + " " + kind; optBtn.disabled=false; optBtn.style.background="linear-gradient(135deg, #10b981, #059669)"; }, 2500);
        } else {
          alert("Option order rejected: " + (j.reason || j.error || "Unknown"));
          optBtn.disabled = false;
          optBtn.textContent = "🎯 Buy " + optQ.strike + " " + kind;
        }
      } catch(err){
        alert("Error: " + err);
        optBtn.disabled = false;
        optBtn.textContent = "🎯 Buy " + optQ.strike + " " + kind;
      }
    };
    btnsWrap.append(optBtn);
  }

  const aiBtn = el("button","btn","🤖 AI Debate");
  aiBtn.style.fontSize = "11px";
  aiBtn.style.padding = "3px 9px";
  aiBtn.style.background = "linear-gradient(135deg, rgba(47,159,219,.15), rgba(99,102,241,.18))";
  aiBtn.style.borderColor = "rgba(47,159,219,.4)";
  aiBtn.style.color = "var(--ink)";
  aiBtn.title = "Summon the 4 AI Agents for deep multi-perspective debate on " + c.symbol;
  aiBtn.onclick = (e)=>{
    e.stopPropagation();
    runAgentAnalysis(c.symbol);
  };
  btnsWrap.append(aiBtn);

  m.append(btnsWrap);

  vw.append(m);
  hd.append(vw);
  box.append(hd);

  const chartSec = el("div","chart-container");
  const cc = candleChart(c);
  chartSec.append(cc || el("div","cap","no intraday bars available"));

  const ladSec = el("div","ladder-section");
  const ladHdr = el("div","ladder-header");
  ladHdr.append(el("div","cap","Price Against Strategy Range & Levels"));
  ladSec.append(ladHdr);
  ladSec.append(ladder(c));
  chartSec.append(ladSec);
  box.append(chartSec);

  if(c.trade) box.append(plan(c));
  else box.append(el("div","noplan","No entry — this setup does not qualify for a "
    + "trade plan. See what is blocking it below."));

  if(c.options) box.append(optionsPanel(c));

  const cols = el("div","cols");
  const why = el("div");
  why.append(el("div","hdr", c.verdict==="BUY" ? "Why this qualifies" : "What is working"));
  const ul = el("ul","rs");
  (c.reasons.length?c.reasons:["nothing notable yet"]).forEach(x=>ul.append(el("li",null,x)));
  why.append(ul); cols.append(why);
  const other = el("div");
  if(c.blockers.length){
    other.append(el("div","hdr","Blocking a buy"));
    const b = el("ul","rs bad");
    c.blockers.forEach(x=>b.append(el("li",null,x.text)));
    other.append(b);
  }else{
    other.append(el("div","hdr","Invalidation — exit if this happens"));
    const b = el("ul","rs");
    c.invalidation.forEach(x=>b.append(el("li",null,x)));
    other.append(b);
  }
  cols.append(other); box.append(cols);

  const met = el("div","metrics");
  const chip = (k,val)=>{ const s=el("span","m"); s.append(document.createTextNode(k));
    s.append(el("b","mono",val)); met.append(s); };
  chip("VWAP", num(c.vwap));
  chip("20 EMA", c.ema20===null?"—":num(c.ema20));
  chip("50 EMA", c.ema50===null?"—":num(c.ema50));
  chip("OR", num(c.or_low)+"–"+num(c.or_high));
  chip("OI", (c.oi.change_pct===null?"—":pc(c.oi.change_pct))+" "+c.oi.classification);
  chip("RVOL", c.rvol===null?"—":c.rvol.toFixed(2)+"x");
  chip("vs NIFTY", pc(c.rel_strength));
  if(c.sector_strength!==null && c.sector_strength!==undefined)
    chip("Sector", pc(c.sector_strength));
  box.append(met);

  if(c.warnings.length)
    box.append(el("div","warns","⚠ " + c.warnings.join("  ·  ")));
  box.append(el("div","srcs", c.sources.map(p=>p.label).join("   |   ")));
  return box;
}

/* The option view. NSE publishes only the 20 most-active option contracts, so
   most names have none — that is stated rather than filled in with a guess.
   Breakeven leads, because a call can lose while the stock setup works. */
function optionsPanel(c){
  const o = c.options, box = el("div","opt");
  const head = el("div","oh");
  head.append(el("span","ot","Call option"));

  if(!o.quote){
    head.append(el("span","be","not available for this setup"));
    box.append(head);
    const no = el("div","no");
    no.append(el("b","NO CALL"));
    no.append(document.createTextNode("  " + (o.rejections[0] || "no candidate strike")));
    box.append(no);
    return box;
  }

  const q = o.quote;
  head.append(el("span","strike mono", q.strike + " CE @ " + num(q.ltp)));
  head.append(el("span","chip", q.expiry));
  head.append(el("span","chip", q.moneyness));
  const be = el("span","be");
  be.append(document.createTextNode("breakeven "));
  be.append(el("b","mono", num(o.breakeven)));
  be.append(document.createTextNode(
    o.clears_t1 ? " — under target 1, so T1 pays"
    : o.clears_t2 ? " — above target 1; only the 1:3 target pays"
    : " — above both targets"));
  head.append(be);
  box.append(head);

  if(o.chain && o.chain.length){
    const t = el("table"), thead = el("thead"), hr = el("tr");
    ["Strike","Type","Expiry","LTP","Breakeven","OI","Volume"]
      .forEach((h,i)=>hr.append(el("th", i>=3?"num":null, h)));
    thead.append(hr); t.append(thead);
    const tb = el("tbody");
    o.chain.forEach(x=>{
      const tr = el("tr");
      if(q && x.identifier === q.identifier) tr.style.background = "var(--accent-soft)";
      tr.append(el("td","mono", String(x.strike)));
      tr.append(el("td", null, x.type));
      tr.append(el("td","mono", x.expiry || "—"));
      tr.append(el("td","num mono", num(x.ltp)));
      tr.append(el("td","num mono", num(x.breakeven)));
      tr.append(el("td","num mono", x.open_interest===null?"—":x.open_interest.toLocaleString("en-IN")));
      tr.append(el("td","num mono", x.volume===null?"—":x.volume.toLocaleString("en-IN")));
      tb.append(tr);
    });
    t.append(tb);
    const wrap = el("div","scroll"); wrap.append(t); box.append(wrap);
  }
  // The option's own plan. Its reward-to-risk is usually worse than the
  // stock's, because premium decays while the stock does not — so it is shown
  // next to the entry rather than left for the reader to work out.
  if(o.premium_at_t1 !== null && o.premium_at_t1 !== undefined){
    const g = el("div","oplan");
    const cell = (k,v,cls,note)=>{const d=el("div");
      d.append(el("div","k",k)); d.append(el("div","v mono "+(cls||""),v));
      if(note) d.append(el("div","n",note)); g.append(d);};
    const pctOf = p => q.ltp>0 ? ((p-q.ltp)/q.ltp*100).toFixed(0)+"%" : "";
    cell("Pay now","₹"+num(q.ltp),"","premium per share");
    if(o.premium_at_stop!==null && o.premium_at_stop!==undefined)
      cell("At stock stop","₹"+num(o.premium_at_stop),"bad",pctOf(o.premium_at_stop));
    cell("At target 1","₹"+num(o.premium_at_t1),"good",pctOf(o.premium_at_t1));
    cell("At target 2","₹"+num(o.premium_at_t2),"good",pctOf(o.premium_at_t2));
    if(o.option_rr!==null && o.option_rr!==undefined)
      cell("Option R:R", o.option_rr.toFixed(2)+":1",
           o.option_rr>=1?"good":"bad", "on the option, not the stock");
    box.append(g);
    box.append(el("div","model",
      "Modelled with Black-Scholes at " +
      (o.implied_vol ? (o.implied_vol*100).toFixed(0)+"% implied volatility" : "the traded IV") +
      ", assuming IV is unchanged and the position is held about two hours. " +
      "An estimate, not a quote — the fill depends on IV and the order book."));
  }

  if(o.decay && o.decay.length){
    const d = el("div","model");
    d.textContent = "If the stock does not move, this premium becomes "
      + o.decay.map(x => (x.days < 1 ? "tonight " : "+" + x.days + "d ")
          + num(x.premium) + (x.pct_left !== null
              ? " (" + Math.round(x.pct_left) + "%)" : "")).join("  ·  ");
    box.append(d);
  }

  if(o.warnings && o.warnings.length){
    const w = el("div","warns");
    o.warnings.forEach(x=>w.append(el("div", null, "⚠ " + x)));
    box.append(w);
  }
  return box;
}

/* 5-Minute Candlestick Chart with Complete Strategy Lines (VWAP, EMA9, EMA21,
   Breakout Resistance, Stop Loss, Target 1, Target 2, Volume, and Interactive HUD). */
function candleChart(c){
  const raw = c.spark || [];
  if(!raw.length) return null;

  const d = raw.map(p => {
    const o = (p.o !== undefined && isFinite(p.o)) ? p.o : p.c;
    const cl = isFinite(p.c) ? p.c : o;
    const h = (p.h !== undefined && isFinite(p.h)) ? p.h : Math.max(o, cl);
    const l = (p.l !== undefined && isFinite(p.l)) ? p.l : Math.min(o, cl);
    const v = (p.v !== undefined && isFinite(p.v)) ? p.v : 0;
    const w = (p.w !== undefined && isFinite(p.w)) ? p.w : cl;
    const e9 = (p.e9 !== undefined && p.e9 !== null && isFinite(p.e9)) ? p.e9 : null;
    const e21 = (p.e21 !== undefined && p.e21 !== null && isFinite(p.e21)) ? p.e21 : null;
    return { t: p.t, o, h, l, c: cl, v, w, e9, e21 };
  }).filter(p => isFinite(p.c));

  if(!d.length) return null;

  const n = d.length;
  const t = c.trade;

  // Collect price data points to set Y-axis scale
  const allPrices = [];
  d.forEach(p => {
    allPrices.push(p.h, p.l, p.o, p.c);
    if(p.w) allPrices.push(p.w);
    if(p.e9) allPrices.push(p.e9);
    if(p.e21) allPrices.push(p.e21);
  });
  if(isFinite(c.or_high)) allPrices.push(c.or_high);
  if(isFinite(c.or_low)) allPrices.push(c.or_low);
  if(isFinite(c.price)) allPrices.push(c.price);
  if(t){
    if(isFinite(t.stop)) allPrices.push(t.stop);
    if(isFinite(t.target1)) allPrices.push(t.target1);
    if(isFinite(t.target2)) allPrices.push(t.target2);
    if(isFinite(t.entry_low)) allPrices.push(t.entry_low);
    if(isFinite(t.entry_high)) allPrices.push(t.entry_high);
  }

  const validPrices = allPrices.filter(v => typeof v === "number" && isFinite(v));
  const minP = Math.min(...validPrices);
  const maxP = Math.max(...validPrices);
  const pad = ((maxP - minP) * 0.05) || 1.0;
  const yMin = minP - pad;
  const yMax = maxP + pad;
  const rng = (yMax - yMin) || 1.0;

  // Viewport dimensions
  const W = 780, H = 220;
  const PL = 8, PR = 66, PT = 14, PB = 28;
  const chartW = W - PL - PR;
  const totalH = H - PT - PB;
  const priceH = totalH * 0.78;
  const volH = totalH * 0.22;
  const volBaseY = H - PB;

  const X = i => n === 1 ? (PL + chartW / 2) : (PL + (i / (n - 1)) * chartW);
  const Y = p => PT + (1 - (p - yMin) / rng) * priceH;

  const candleW = Math.max(2.5, Math.min(10, (chartW / Math.max(n, 12)) * 0.72));
  const maxVol = Math.max(...d.map(p => p.v), 1);

  const wrap = el("div", "candle-card");

  // Dynamic HUD Strip
  const hud = el("div", "candle-hud");
  const hudLeft = el("div", "candle-hud-title");
  hudLeft.innerHTML = `<span style="font-size:12px">🕯️</span> <span>5-Min Candlestick &amp; Strategy Engine</span> <span class="chip" style="font-size:9.5px;padding:1px 5px">${n} Bars</span>`;
  hud.append(hudLeft);

  const hudVals = el("div", "candle-hud-vals");
  const hudTime = el("span", null, "");
  const hudO = el("span", null, "");
  const hudH = el("span", null, "");
  const hudL = el("span", null, "");
  const hudC = el("span", null, "");
  const hudV = el("span", null, "");
  const hudW = el("span", null, "");
  const hudE9 = el("span", null, "");
  const hudE21 = el("span", null, "");

  hudVals.append(hudTime, hudO, hudH, hudL, hudC, hudV, hudW, hudE9, hudE21);
  hud.append(hudVals);
  wrap.append(hud);

  function updateHud(p){
    const isUp = p.c >= p.o;
    const diff = p.c - p.o;
    const diffPct = p.o > 0 ? (diff / p.o * 100) : 0;
    const chgClass = isUp ? "up" : "down";

    hudTime.innerHTML = `<span style="color:var(--ink-3)">Time:</span> <b>${p.t}</b>`;
    hudO.innerHTML = `<span style="color:var(--ink-3)">O:</span> <b>₹${num(p.o)}</b>`;
    hudH.innerHTML = `<span style="color:var(--ink-3)">H:</span> <b>₹${num(p.h)}</b>`;
    hudL.innerHTML = `<span style="color:var(--ink-3)">L:</span> <b>₹${num(p.l)}</b>`;
    hudC.innerHTML = `<span style="color:var(--ink-3)">C:</span> <b class="${chgClass}">₹${num(p.c)} (${isUp ? "+" : ""}${diffPct.toFixed(2)}%)</b>`;
    hudV.innerHTML = `<span style="color:var(--ink-3)">Vol:</span> <b>${p.v >= 1e5 ? (p.v/1e5).toFixed(1)+"L" : p.v >= 1e3 ? (p.v/1e3).toFixed(1)+"K" : p.v.toLocaleString("en-IN")}</b>`;
    hudW.innerHTML = `<span style="color:#eab308">VWAP:</span> <b style="color:#eab308">₹${num(p.w)}</b>`;
    if(p.e9 !== null) hudE9.innerHTML = `<span style="color:#38bdf8">EMA9:</span> <b style="color:#38bdf8">₹${num(p.e9)}</b>`;
    else hudE9.innerHTML = "";
    if(p.e21 !== null) hudE21.innerHTML = `<span style="color:#a855f7">EMA21:</span> <b style="color:#a855f7">₹${num(p.e21)}</b>`;
    else hudE21.innerHTML = "";
  }

  updateHud(d[n - 1]);

  const svgWrap = el("div", "candle-svg-wrap");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");

  const add = (tag, attrs, title) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for(const k in attrs) node.setAttribute(k, attrs[k]);
    if(title){
      const tNode = document.createElementNS("http://www.w3.org/2000/svg", "title");
      tNode.textContent = title;
      node.append(tNode);
    }
    svg.append(node);
    return node;
  };

  // Horizontal Grid Lines & Y-Axis Labels
  for(let i = 0; i <= 4; i++){
    const pVal = yMin + (i / 4) * rng;
    const yPos = Y(pVal);
    add("line", {
      x1: PL, x2: W - PR, y1: yPos, y2: yPos,
      stroke: "rgba(255,255,255,0.06)", "stroke-dasharray": "3,4", "stroke-width": 1
    });
    const txt = add("text", {
      x: W - PR + 6, y: yPos + 3.5, fill: "var(--ink-3)",
      "font-size": 9.5, "font-family": "ui-monospace, monospace"
    });
    txt.textContent = "₹" + num(pVal);
  }

  // Vertical Grid Lines & X-Axis Time Labels
  const step = Math.max(1, Math.round(n / 6));
  for(let i = 0; i < n; i += step){
    const cx = X(i);
    add("line", {
      x1: cx, x2: cx, y1: PT, y2: H - PB,
      stroke: "rgba(255,255,255,0.04)", "stroke-dasharray": "2,4", "stroke-width": 1
    });
    const tTxt = add("text", {
      x: cx, y: H - PB + 13, "text-anchor": "middle", fill: "var(--ink-3)",
      "font-size": 9, "font-family": "ui-monospace, monospace"
    });
    tTxt.textContent = d[i].t;
  }
  if((n - 1) % step !== 0 && ((n - 1) - ((n - 1) % step)) > step / 2){
    const lastX = X(n - 1);
    const tTxt = add("text", {
      x: lastX, y: H - PB + 13, "text-anchor": "middle", fill: "var(--ink-3)",
      "font-size": 9, "font-family": "ui-monospace, monospace"
    });
    tTxt.textContent = d[n - 1].t;
  }

  // Volume baseline separator & label
  add("line", {
    x1: PL, x2: W - PR, y1: volBaseY - volH, y2: volBaseY - volH,
    stroke: "rgba(255,255,255,0.06)", "stroke-width": 1
  });
  const vLbl = add("text", {
    x: PL + 2, y: volBaseY - volH + 9, fill: "var(--ink-3)", "font-size": 8, "font-weight": 700
  });
  vLbl.textContent = "VOL";

  // Volume Bars
  d.forEach((p, i) => {
    const cx = X(i);
    const vHeight = maxVol > 0 ? (p.v / maxVol) * (volH - 3) : 0;
    const isUp = p.c >= p.o;
    const vColor = isUp ? "rgba(52, 211, 153, 0.40)" : "rgba(239, 68, 68, 0.40)";
    add("rect", {
      x: cx - candleW / 2, y: volBaseY - vHeight,
      width: Math.max(1.5, candleW), height: Math.max(0.5, vHeight),
      fill: vColor, rx: 0.5
    }, `${p.t} Vol: ${p.v.toLocaleString("en-IN")}`);
  });

  // Opening Range (OR) Zone Band
  if(isFinite(c.or_high) && isFinite(c.or_low)){
    const topY = Y(c.or_high);
    const botY = Y(c.or_low);
    add("rect", {
      x: PL, y: topY, width: chartW, height: Math.max(2, botY - topY),
      fill: "rgba(56, 189, 248, 0.06)", stroke: "rgba(56, 189, 248, 0.22)",
      "stroke-dasharray": "3,3", "stroke-width": 0.8
    }, `Opening Range (09:15-09:30): ₹${num(c.or_low)} - ₹${num(c.or_high)}`);
  }

  // Entry Zone Band (if trade exists)
  if(t && isFinite(t.entry_low) && isFinite(t.entry_high)){
    const topY = Y(t.entry_high);
    const botY = Y(t.entry_low);
    add("rect", {
      x: PL, y: topY, width: chartW, height: Math.max(2, botY - topY),
      fill: "rgba(47, 159, 219, 0.12)", stroke: "rgba(47, 159, 219, 0.40)",
      "stroke-dasharray": "3,3", "stroke-width": 1
    }, `Entry Zone: ₹${num(t.entry_low)} - ₹${num(t.entry_high)}`);
  }

  // Horizontal Strategy Level Lines & Tags
  const drawLevel = (val, color, bgFill, textFill, label, strokeDash = "4,3", strokeWidth = 1.4) => {
    if(typeof val !== "number" || !isFinite(val)) return;
    const yPos = Y(val);
    if(yPos < PT - 5 || yPos > H - PB + 5) return;

    add("line", {
      x1: PL, x2: W - PR, y1: yPos, y2: yPos,
      stroke: color, "stroke-dasharray": strokeDash, "stroke-width": strokeWidth, opacity: 0.9
    });

    add("rect", {
      x: W - PR + 3, y: yPos - 7, width: 62, height: 14, rx: 3,
      fill: bgFill, stroke: color, "stroke-width": 0.6
    });
    const tag = add("text", {
      x: W - PR + 5, y: yPos + 3.5, fill: textFill,
      "font-size": 8.5, "font-weight": 700, "font-family": "ui-monospace, monospace"
    });
    tag.textContent = label;
  };

  if(isFinite(c.or_high)){
    drawLevel(c.or_high, "#38bdf8", "rgba(14,116,144,0.9)", "#e0f2fe", `OR-H ₹${num(c.or_high)}`, "4,4", 1.3);
  }
  if(isFinite(c.or_low)){
    drawLevel(c.or_low, "#f43f5e", "rgba(159,18,57,0.9)", "#ffe4e6", `OR-L ₹${num(c.or_low)}`, "4,4", 1.3);
  }

  if(t){
    if(isFinite(t.stop)){
      drawLevel(t.stop, "#ef4444", "rgba(127,29,29,0.85)", "#fca5a5", `■ SL ₹${num(t.stop)}`, "4,3", 1.5);
    }
    if(isFinite(t.target1)){
      drawLevel(t.target1, "#10b981", "rgba(6,78,59,0.85)", "#6ee7b7", `▲ T1 ₹${num(t.target1)}`, "4,3", 1.5);
    }
    if(isFinite(t.target2)){
      drawLevel(t.target2, "#34d399", "rgba(6,78,59,0.85)", "#a7f3d0", `▲ T2 ₹${num(t.target2)}`, "4,3", 1.5);
    }
  }

  // Candlesticks (OHLC Bodies & Wicks)
  d.forEach((p, i) => {
    const cx = X(i);
    const isUp = p.c >= p.o;
    const color = isUp ? "#22c55e" : "#ef4444";
    const wickTop = Y(p.h);
    const wickBot = Y(p.l);
    const bodyTop = Y(Math.max(p.o, p.c));
    const bodyBot = Y(Math.min(p.o, p.c));
    const bodyHeight = Math.max(1.8, bodyBot - bodyTop);

    add("line", {
      x1: cx, x2: cx, y1: wickTop, y2: wickBot,
      stroke: color, "stroke-width": 1.2
    });

    add("rect", {
      x: cx - candleW / 2, y: bodyTop,
      width: candleW, height: bodyHeight,
      fill: color, stroke: color, "stroke-width": 0.5, rx: 0.8
    });
  });

  // Strategy Curves: VWAP, EMA9, EMA21
  const vwapPoints = d.filter(p => isFinite(p.w));
  if(vwapPoints.length >= 2){
    const vwapPath = vwapPoints.map((p, i) => (i ? "L" : "M") + X(d.indexOf(p)).toFixed(1) + " " + Y(p.w).toFixed(1)).join(" ");
    add("path", {
      d: vwapPath, fill: "none", stroke: "#eab308", "stroke-width": 1.9,
      "stroke-linecap": "round", "stroke-linejoin": "round"
    }, "VWAP Line");
  }

  const e9Points = d.filter(p => p.e9 !== null && isFinite(p.e9));
  if(e9Points.length >= 2){
    const e9Path = e9Points.map((p, i) => (i ? "L" : "M") + X(d.indexOf(p)).toFixed(1) + " " + Y(p.e9).toFixed(1)).join(" ");
    add("path", {
      d: e9Path, fill: "none", stroke: "#38bdf8", "stroke-width": 1.6,
      "stroke-linecap": "round", "stroke-linejoin": "round"
    }, "EMA 9 Line");
  }

  const e21Points = d.filter(p => p.e21 !== null && isFinite(p.e21));
  if(e21Points.length >= 2){
    const e21Path = e21Points.map((p, i) => (i ? "L" : "M") + X(d.indexOf(p)).toFixed(1) + " " + Y(p.e21).toFixed(1)).join(" ");
    add("path", {
      d: e21Path, fill: "none", stroke: "#a855f7", "stroke-width": 1.6,
      "stroke-dasharray": "4,2", "stroke-linecap": "round", "stroke-linejoin": "round"
    }, "EMA 21 Line");
  }

  // Live Price Marker
  const lastBar = d[n - 1];
  const lastX = X(n - 1);
  const lastY = Y(lastBar.c);
  add("circle", {
    cx: lastX, cy: lastY, r: 3.5, fill: "#38bdf8", stroke: "var(--card)", "stroke-width": 1.5
  });

  add("rect", {
    x: W - PR + 3, y: lastY - 7, width: 59, height: 14, rx: 3,
    fill: "#0284c7", stroke: "#38bdf8", "stroke-width": 0.8
  });
  const curTag = add("text", {
    x: W - PR + 6, y: lastY + 3.5, fill: "#ffffff",
    "font-size": 8.5, "font-weight": 700, "font-family": "ui-monospace, monospace"
  });
  curTag.textContent = "₹" + num(lastBar.c);

  // Interactive Crosshair
  const crossGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  crossGroup.setAttribute("class", "candle-crosshair");
  crossGroup.style.display = "none";

  const vCross = document.createElementNS("http://www.w3.org/2000/svg", "line");
  vCross.setAttribute("y1", String(PT));
  vCross.setAttribute("y2", String(H - PB));
  vCross.setAttribute("stroke", "rgba(255,255,255,0.35)");
  vCross.setAttribute("stroke-dasharray", "3,3");
  vCross.setAttribute("stroke-width", "1");
  crossGroup.append(vCross);

  const hCross = document.createElementNS("http://www.w3.org/2000/svg", "line");
  hCross.setAttribute("x1", String(PL));
  hCross.setAttribute("x2", String(W - PR));
  hCross.setAttribute("stroke", "rgba(255,255,255,0.35)");
  hCross.setAttribute("stroke-dasharray", "3,3");
  hCross.setAttribute("stroke-width", "1");
  crossGroup.append(hCross);

  const hoverDot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  hoverDot.setAttribute("r", "4");
  hoverDot.setAttribute("fill", "var(--accent)");
  hoverDot.setAttribute("stroke", "var(--ink)");
  hoverDot.setAttribute("stroke-width", "1.5");
  crossGroup.append(hoverDot);

  svg.append(crossGroup);

  const overlay = add("rect", {
    x: PL, y: PT, width: chartW, height: totalH,
    fill: "transparent", cursor: "crosshair"
  });

  const handlePointer = (clientX) => {
    const rect = svg.getBoundingClientRect();
    const relX = ((clientX - rect.left) / rect.width) * W;
    const barIdx = Math.max(0, Math.min(n - 1, Math.round(((relX - PL) / chartW) * (n - 1))));
    const bar = d[barIdx];
    if(!bar) return;

    const bx = X(barIdx);
    const by = Y(bar.c);

    vCross.setAttribute("x1", String(bx));
    vCross.setAttribute("x2", String(bx));
    hCross.setAttribute("y1", String(by));
    hCross.setAttribute("y2", String(by));
    hoverDot.setAttribute("cx", String(bx));
    hoverDot.setAttribute("cy", String(by));
    crossGroup.style.display = "";

    updateHud(bar);
  };

  overlay.addEventListener("mousemove", e => handlePointer(e.clientX));
  overlay.addEventListener("touchmove", e => {
    if(e.touches && e.touches.length) handlePointer(e.touches[0].clientX);
  });
  overlay.addEventListener("mouseleave", () => {
    crossGroup.style.display = "none";
    updateHud(d[n - 1]);
  });
  overlay.addEventListener("touchend", () => {
    crossGroup.style.display = "none";
    updateHud(d[n - 1]);
  });

  svgWrap.append(svg);
  wrap.append(svgWrap);

  // Strategy Legend
  const leg = el("div", "candle-legend");
  const legItem = (iconHtml, label, color) => {
    const item = el("div", "candle-legend-item");
    item.innerHTML = iconHtml + `<span style="color:${color || 'inherit'}">${label}</span>`;
    leg.append(item);
  };

  legItem(`<span class="candle-legend-line" style="border-color:#eab308"></span>`, "VWAP (Session)", "#eab308");
  legItem(`<span class="candle-legend-line" style="border-color:#38bdf8"></span>`, "EMA 9", "#38bdf8");
  legItem(`<span class="candle-legend-line candle-legend-dashed" style="border-color:#a855f7"></span>`, "EMA 21", "#a855f7");
  if(isFinite(c.or_high))
    legItem(`<span class="candle-legend-line candle-legend-dashed" style="border-color:#94a3b8"></span>`, `Breakout ₹${num(c.or_high)}`, "#94a3b8");
  if(t){
    legItem(`<span class="candle-legend-line candle-legend-dashed" style="border-color:#ef4444"></span>`, `Stop Loss ₹${num(t.stop)}`, "#fca5a5");
    legItem(`<span class="candle-legend-line candle-legend-dashed" style="border-color:#10b981"></span>`, `Target 1 ₹${num(t.target1)}`, "#6ee7b7");
    legItem(`<span class="candle-legend-line candle-legend-dashed" style="border-color:#34d399"></span>`, `Target 2 ₹${num(t.target2)}`, "#a7f3d0");
  }

  wrap.append(leg);
  return wrap;
}

/* Stop and target differ by SHAPE as well as hue — red/green is exactly the
   pair a colour-blind reader loses, and confusing them is the most expensive
   mistake this page could cause. */
function ladder(c){
  const wrap = el("div");
  const t = c.trade;
  const vals = [c.or_low,c.or_high,c.vwap,c.price,c.day_low,c.day_high]
    .concat(t?[t.stop,t.target1,t.target2]:[])
    .filter(v=>typeof v==="number" && isFinite(v));
  const lo=Math.min(...vals), hi=Math.max(...vals);
  const pad=(hi-lo)*0.06 || 0.5, A=lo-pad, B=hi+pad;
  const X=v=>((v-A)/(B-A))*100;

  const lad = el("div","lad");
  lad.append(el("div","bar"));
  const zone=(cls,from,to,title)=>{const z=el("div","zone "+cls);
    z.style.left=X(from)+"%"; z.style.width=Math.max(X(to)-X(from),0.4)+"%";
    z.title=title; lad.append(z);};
  if(t){
    zone("risk",t.stop,t.entry_low,"risk "+num(t.stop)+" → "+num(t.entry_low));
    zone("rew",t.entry_high,t.target2,"reward "+num(t.entry_high)+" → "+num(t.target2));
    zone("entry",t.entry_low,t.entry_high,"entry zone "+num(t.entry_low)+" – "+num(t.entry_high));
  }
  const rule=(v,color,label,cap)=>{
    if(typeof v!=="number"||!isFinite(v)) return;
    const r=el("div","rule"); r.style.left=X(v)+"%";
    r.style.background=color; r.style.color=color; r.title=label+" "+num(v);
    if(cap) r.append(el("i","cap-"+cap));
    lad.append(r);
  };
  rule(c.or_high,"var(--ink-2)","breakout level (09:15–09:30 high)");
  rule(c.vwap,"var(--ink-3)","VWAP");
  if(t){
    rule(t.stop,"var(--bad)","stop","sq");
    rule(t.target1,"var(--good)","target 1","tri");
    rule(t.target2,"var(--good)","target 2","tri");
  }
  const dot=el("div","dot"); dot.style.left=X(c.price)+"%";
  dot.title="current price "+num(c.price); lad.append(dot);
  wrap.append(lad);

  const lg=el("div","lgnd");
  const key=(color,txt,round)=>{const s=el("span"); const i=el("i",round?"rnd":null);
    i.style.background=color; s.append(i); s.append(document.createTextNode(txt));
    lg.append(s);};
  key("var(--accent)","Price "+num(c.price),true);
  key("var(--ink-2)","Breakout "+num(c.or_high));
  key("var(--ink-3)","VWAP "+num(c.vwap));
  if(t){
    key("var(--bad)","■ Stop "+num(t.stop));
    key("var(--good)","▲ T1 "+num(t.target1));
    key("var(--good)","▲ T2 "+num(t.target2));
  }
  wrap.append(lg);
  return wrap;
}

function plan(c){
  const t=c.trade, g=el("div","plan");
  const cell=(k,v,cls,note,hl)=>{const d=el("div","c"+(hl?" hl":""));
    d.append(el("div","k",k)); d.append(el("div","v mono "+(cls||""),v));
    if(note) d.append(el("div","n",note)); g.append(d);};
  cell("Entry zone","₹"+num(t.entry_low)+" – "+num(t.entry_high),"",
       "buy inside this band only",true);
  cell("Stop loss","₹"+num(t.stop),"bad",t.stop_basis);
  cell("Target 1","₹"+num(t.target1),"good","1:"+t.rr1);
  cell("Target 2","₹"+num(t.target2),"good","1:"+t.rr2);
  cell("Risk","₹"+num(t.risk),"",t.risk_pct.toFixed(2)+"% of price");
  return g;
}

/* ---------- table ---------- */
function table(s){
  const t=$("#tbl"); t.innerHTML="";
  const head=el("tr");
  [["Stock",0],["Verdict",0],["Score",1],["Price",1],["vs NIFTY",1],["Breakout",1],
   ["VWAP",1],["RVOL",1],["OI",0],["Structure",0],["Blocking",0]]
    .forEach(([h,n])=>head.append(el("th",n?"num":null,h)));
  const thead=el("thead"); thead.append(head); t.append(thead);
  const body=el("tbody");
  const rows=(s.candidates||[]).slice().sort((a,b)=>b.score-a.score);
  if(!rows.length){
    const tr=el("tr"), td=el("td",null, s.data_ok
      ? "No names cleared the liquidity screen." : "Nothing graded.");
    td.colSpan=11; tr.append(td); body.append(tr);
  }
  rows.forEach(c=>{
    const tr=el("tr");
    tr.dataset.rowSym = c.symbol;
    const symTd = el("td","sym");
    symTd.style.cursor = "pointer";
    symTd.title = "Click to run AI Council Debate for " + c.symbol;
    symTd.onclick = () => runAgentAnalysis(c.symbol);
    symTd.innerHTML = `<span style="border-bottom:1px dotted var(--link)">${c.symbol}</span> <span style="font-size:10px;opacity:0.8" title="Consult AI Agents">🤖</span>`;
    tr.append(symTd);
    const v=el("td"); v.append(badge(c.verdict,"sm v-"+c.verdict.toLowerCase()));
    tr.append(v);
    tr.append(el("td","num mono", sc(c.score)+" "+c.grade));
    tr.append(el("td","num mono", num(c.price)));
    tr.append(el("td","num mono "+sgn(c.rel_strength), pc(c.rel_strength)));
    tr.append(el("td","num mono", num(c.or_high)));
    tr.append(el("td","num mono", num(c.vwap)));
    tr.append(el("td","num mono", c.rvol===null?"—":c.rvol.toFixed(2)+"x"));
    tr.append(el("td","mono", (c.oi.change_pct===null?"—":pc(c.oi.change_pct))
      +" "+c.oi.classification));
    tr.append(el("td",null,c.structure.status));
    tr.append(el("td","blk", c.blockers.length?c.blockers[0].text:"—"));
    body.append(tr);
  });
  t.append(body);
}

/* ---------- controls ---------- */
$("#seg").addEventListener("click", e=>{
  const b=e.target.closest("button"); if(!b) return;
  FILTER=b.dataset.f;
  [...$("#seg").children].forEach(x=>x.setAttribute("aria-pressed",String(x===b)));
  if(STATE && STATE.scan) list(STATE.scan);
});
$("#rescan").addEventListener("click", ()=>load(true));
$("#auto").addEventListener("change", ()=>load(false));

// No dashboard is mounted next to a standalone scanner, so drop the link.
if(document.body.dataset.mode === "standalone"){
  const a = document.querySelector('.nav a[href="/"]');
  if(a) a.remove();
}

load(false);

async function loadQuotes(){
  try{
    const r = await fetch("/api/quotes", {cache:"no-store"});
    if(!r.ok) return;
    const res = await r.json();
    const itemsBox = document.getElementById("liveTickerItems");
    const updatedEl = document.getElementById("liveTickerUpdated");
    if(itemsBox && res.quotes && res.quotes.length){
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
            <span style="font-size:14px;font-weight:700;font-family:ui-monospace,monospace;color:${isUp ? 'var(--up)' : 'var(--down)'}">₹${num(lp)}</span>
            <span style="font-size:10px;color:var(--ink-3);font-family:ui-monospace,monospace">${ch >= 0 ? '+' : ''}${num(ch)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:9.5px;color:var(--ink-3);margin-top:1px">
            <span>H: ${num(v.high_price)}</span>
            <span>L: ${num(v.low_price)}</span>
          </div>
        `;
        itemsBox.appendChild(card);

        // Update candidate cards on page
        const sym = (v.symbol || q.n || "").replace("NSE:","").replace("-EQ","").replace("-INDEX","");
        const cardEl = document.querySelector(`[data-card-sym="${sym}"]`);
        if(cardEl && v.lp){
          const pxEl = cardEl.querySelector(".pxw .px");
          const chgEl = cardEl.querySelector(".pxw .chg");
          if(pxEl) pxEl.textContent = "₹" + num(v.lp);
          if(chgEl && v.chp != null){
            chgEl.className = "chg mono " + (v.chp > 0 ? "up" : v.chp < 0 ? "down" : "");
            chgEl.textContent = (v.chp >= 0 ? "+" : "") + v.chp.toFixed(2) + "%";
          }
        }
      });
      if(updatedEl){
        updatedEl.textContent = "⚡ Live: " + new Date().toLocaleTimeString("en-IN",{hour12:false});
      }
    }

    if(res.quotes){
      const nq = res.quotes.find(q => (q.n||"").includes("NIFTY50"));
      const bq = res.quotes.find(q => (q.n||"").includes("NIFTYBANK"));
      if(nq && nq.v){
        const elN = document.getElementById("kpi-nifty");
        if(elN){
          elN.innerHTML = `<span style="font-size:17px;font-weight:700;font-family:ui-monospace,monospace;color:${nq.v.ch>=0?'var(--up)':'var(--down)'}">₹${num(nq.v.lp)}</span> <span style="font-size:11px;font-weight:600;color:${nq.v.ch>=0?'var(--up)':'var(--down)'}">(${nq.v.chp>=0?'+':''}${nq.v.chp.toFixed(2)}%)</span>`;
        }
        const spotEl = document.querySelector("#niftyWrap .nfx .spot");
        if(spotEl && nq.v.lp){
          spotEl.textContent = num(nq.v.lp);
        }
      }
      if(bq && bq.v){
        const elB = document.getElementById("kpi-banknifty");
        if(elB){
          elB.innerHTML = `<span style="font-size:17px;font-weight:700;font-family:ui-monospace,monospace;color:${bq.v.ch>=0?'var(--up)':'var(--down)'}">₹${num(bq.v.lp)}</span> <span style="font-size:11px;font-weight:600;color:${bq.v.ch>=0?'var(--up)':'var(--down)'}">(${bq.v.chp>=0?'+':''}${bq.v.chp.toFixed(2)}%)</span>`;
        }
      }
    }
  }catch(e){
    console.warn("loadQuotes error:", e);
  }
}

/* ---------- AI Agent Intelligence Desk ---------- */
function setAiSymbol(sym){
  const inp = document.getElementById("aiAgentSymInput");
  if(inp){
    inp.value = sym.toUpperCase().trim();
  }
}

function toggleAgentDesk(){
  const body = document.getElementById("agentDeskBody");
  const btn = document.getElementById("btnToggleAgents");
  if(!body || !btn) return;
  const isHidden = body.style.display === "none";
  body.style.display = isHidden ? "block" : "none";
  btn.textContent = isHidden ? "Minimize Desk" : "Expand Desk";
}

function updateQuickChips(s){
  const box = document.getElementById("agentQuickChips");
  if(!box || !s || !s.candidates) return;
  const topSyms = s.candidates.slice(0, 8).map(c => c.symbol);
  if(!topSyms.length) return;
  const defaultList = ["RELIANCE", "SBIN", "HDFCBANK", "ICICIBANK", "INFY", "TCS"];
  const combined = Array.from(new Set([...topSyms, ...defaultList])).slice(0, 10);
  box.innerHTML = "";
  combined.forEach(sym => {
    const chip = el("span", "agent-chip", sym);
    chip.onclick = () => setAiSymbol(sym);
    box.appendChild(chip);
  });
}

function runAgentAnalysis(sym){
  setAiSymbol(sym);
  const panel = document.getElementById("agentDeskPanel");
  if(panel){
    panel.scrollIntoView({behavior: "smooth", block: "start"});
    const body = document.getElementById("agentDeskBody");
    const btn = document.getElementById("btnToggleAgents");
    if(body && body.style.display === "none"){
      body.style.display = "block";
      if(btn) btn.textContent = "Minimize Desk";
    }
  }
  triggerAgentDebate();
}

let autoDebateTimer = null;
function toggleAutoAiDebate(enabled){
  clearInterval(autoDebateTimer);
  if(enabled){
    triggerAgentDebate();
    autoDebateTimer = setInterval(() => {
      const chk = document.getElementById("chkAutoAiDebate");
      if(!chk || !chk.checked){
        clearInterval(autoDebateTimer);
        return;
      }
      if(STATE && STATE.scan && STATE.scan.candidates && STATE.scan.candidates.length){
        const syms = STATE.scan.candidates.map(c => c.symbol);
        const currentSym = (document.getElementById("aiAgentSymInput")?.value || "").toUpperCase();
        let nextIdx = (syms.indexOf(currentSym) + 1) % syms.length;
        setAiSymbol(syms[nextIdx]);
      }
      triggerAgentDebate();
    }, 15000);
  }
}

let debateInterval = null;
const debateStages = [
  "⚡ Ingesting live ticks & 15m OHLCV candles via Fyers API v3…",
  "📊 Technical Analyst calculating EMA 9/21/50, RSI(14), ATR & VWAP…",
  "🐂 Bullish Researcher formulating momentum and breakout thesis…",
  "🐻 Bearish Researcher auditing bull traps, fakeouts and supply zones…",
  "⚖️ Trader Decision Desk synthesizing multi-agent debate and risk rules…"
];

async function triggerAgentDebate(){
  const inp = document.getElementById("aiAgentSymInput");
  const btn = document.getElementById("btnRunAgentDebate");
  const loading = document.getElementById("aiDebateLoading");
  const stepText = document.getElementById("aiDebateStep");
  const resBox = document.getElementById("aiDebateResult");

  if(!inp || !btn) return;
  const sym = (inp.value || "").toUpperCase().trim();
  if(!sym){
    alert("Please enter a stock symbol (e.g. RELIANCE, SBIN, TCS)");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Debating…";
  if(resBox) resBox.style.display = "none";
  if(loading) loading.style.display = "flex";

  let stageIdx = 0;
  if(stepText) stepText.textContent = debateStages[0];
  clearInterval(debateInterval);
  debateInterval = setInterval(() => {
    stageIdx = (stageIdx + 1) % debateStages.length;
    if(stepText) stepText.textContent = debateStages[stageIdx];
  }, 1600);

  try {
    const res = await fetch("/api/ai-scan", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ symbol: sym, min_conviction: 7 })
    });
    const data = await res.json();
    clearInterval(debateInterval);
    if(loading) loading.style.display = "none";
    btn.disabled = false;
    btn.textContent = "⚡ Run AI Council Debate";

    if(data.ok && data.recommendation){
      renderAgentDebateResult(data);
    } else {
      if(resBox){
        resBox.style.display = "block";
        resBox.innerHTML = `
          <div style="padding:14px;background:rgba(224,90,112,0.12);border:1px solid var(--bad);border-radius:8px;color:var(--bad)">
            <strong style="display:flex;align-items:center;gap:6px">
              <span>✗ AI Council Analysis Failed for ${sym}</span>
            </strong>
            <p style="margin:6px 0 0 0;font-size:12px;color:var(--ink-2)">
              ${data.error || "Unable to retrieve quotes or generate recommendation."}
            </p>
          </div>
        `;
      }
    }
  } catch(err){
    clearInterval(debateInterval);
    if(loading) loading.style.display = "none";
    btn.disabled = false;
    btn.textContent = "⚡ Run AI Council Debate";
    if(resBox){
      resBox.style.display = "block";
      resBox.innerHTML = `
        <div style="padding:14px;background:rgba(224,90,112,0.12);border:1px solid var(--bad);border-radius:8px;color:var(--bad)">
          <strong>✗ Error communicating with AI Desk:</strong> ${err}
        </div>
      `;
    }
  }
}

function renderAgentDebateResult(data){
  const resBox = document.getElementById("aiDebateResult");
  if(!resBox) return;

  const rec = data.recommendation;
  const sym = data.symbol;
  const act = rec.action || "HOLD";
  const conv = rec.conviction || 5;

  let actColor = "var(--warn)";
  let actBg = "var(--warn-soft)";
  let actBorder = "rgba(201,133,0,.3)";
  if(act === "BUY"){
    actColor = "var(--good)";
    actBg = "var(--good-soft)";
    actBorder = "rgba(52,211,153,.35)";
  } else if(act === "SELL"){
    actColor = "var(--bad)";
    actBg = "var(--bad-soft)";
    actBorder = "rgba(224,90,112,.35)";
  }

  const ep = rec.entry_price || 0;
  const sl = rec.stop_loss || 0;
  const tg = rec.target || 0;
  const risk = ep > 0 && sl > 0 ? Math.abs(ep - sl) : 0;
  const reward = ep > 0 && tg > 0 ? Math.abs(tg - ep) : 0;
  const rr = risk > 0 ? (reward / risk).toFixed(2) : "—";
  const riskPct = ep > 0 && risk > 0 ? ((risk / ep) * 100).toFixed(2) + "%" : "—";
  const rewardPct = ep > 0 && reward > 0 ? ((reward / ep) * 100).toFixed(2) + "%" : "—";

  resBox.style.display = "block";
  resBox.innerHTML = `
    <div class="debate-verdict-hdr">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <span style="font-size:18px;font-weight:800;letter-spacing:-.01em;color:var(--ink)">NSE:${sym}</span>
        <span class="agent-badge" style="font-size:12px;padding:4px 10px;background:${actBg};color:${actColor};border:1px solid ${actBorder}">
          VERDICT: ${act}
        </span>
        <span style="font-size:11px;color:var(--ink-3);font-family:ui-monospace,monospace">
          Evaluated at ${new Date().toLocaleTimeString("en-IN", {hour12:false})} IST
        </span>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <div style="text-align:right">
          <div style="font-size:10px;text-transform:uppercase;color:var(--ink-3);letter-spacing:.05em">Council Conviction</div>
          <div style="font-size:15px;font-weight:700;font-family:ui-monospace,monospace;color:${conv >= 7 ? 'var(--good)' : 'var(--warn)'}">
            ${conv} / 10 · ${conv * 10}%
          </div>
        </div>
        <div style="width:70px;height:7px;background:var(--track);border-radius:99px;overflow:hidden">
          <div style="width:${conv * 10}%;height:100%;background:${conv >= 7 ? 'var(--good)' : 'var(--warn)'};border-radius:99px"></div>
        </div>
      </div>
    </div>

    <!-- Key Trading Matrix -->
    <div class="debate-scores">
      <div class="debate-stat">
        <div class="lbl">Entry Level</div>
        <div class="val" style="color:var(--ink)">₹${num(ep)}</div>
      </div>
      <div class="debate-stat">
        <div class="lbl">Stop Loss</div>
        <div class="val" style="color:var(--bad)">₹${num(sl)} <span style="font-size:10px;color:var(--ink-3)">(${riskPct})</span></div>
      </div>
      <div class="debate-stat">
        <div class="lbl">Target</div>
        <div class="val" style="color:var(--good)">₹${num(tg)} <span style="font-size:10px;color:var(--ink-3)">(${rewardPct})</span></div>
      </div>
      <div class="debate-stat">
        <div class="lbl">Risk / Reward</div>
        <div class="val" style="color:var(--accent)">1 : ${rr}</div>
      </div>
      <div class="debate-stat">
        <div class="lbl">Decision Gate</div>
        <div class="val" style="color:${conv >= 7 ? 'var(--good)' : 'var(--warn)'}">
          ${conv >= 7 ? '✓ Approved' : '✗ Hold Filter'}
        </div>
      </div>
    </div>

    <!-- The 4 Agent Perspectives -->
    <div class="debate-perspectives">
      <div class="perspective-card" style="border-left: 3px solid var(--accent)">
        <div class="perspective-title" style="color:var(--accent)">
          <span>📊 Technical Analyst View</span>
        </div>
        <div class="perspective-body">
          ${rec.technical_analysis || "No technical breakdown available."}
        </div>
      </div>

      <div class="perspective-card" style="border-left: 3px solid var(--good)">
        <div class="perspective-title" style="color:var(--good)">
          <span>🐂 Bullish Researcher Thesis</span>
        </div>
        <div class="perspective-body">
          ${rec.bull_case || "No bullish arguments presented."}
        </div>
      </div>

      <div class="perspective-card" style="border-left: 3px solid var(--bad)">
        <div class="perspective-title" style="color:var(--bad)">
          <span>🐻 Bearish Researcher Counter-Audit</span>
        </div>
        <div class="perspective-body">
          ${rec.bear_case || "No counter-arguments found."}
        </div>
      </div>

      <div class="perspective-card" style="border-left: 3px solid #a855f7">
        <div class="perspective-title" style="color:#c084fc">
          <span>⚖️ Trader Decision Desk Synthesis</span>
        </div>
        <div class="perspective-body">
          ${rec.decision_rationale || "No final rationale recorded."}
        </div>
      </div>
    </div>

    <!-- Execution Action Bar -->
    <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
      <div style="font-size:11.5px;color:var(--ink-2)">
        ${act === 'HOLD'
          ? 'ℹ Committee recommended HOLD — conviction is below 7/10 or risk/reward is unfavorable.'
          : '⚡ Committee issued high-conviction order directive. Ready for execution in ledger.'}
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        ${act !== 'HOLD' ? `
          <button class="btn go" id="btnExecAiTrade" style="padding:6px 14px;font-weight:700" onclick="executeAiTradeDirect('${sym}', '${act}', ${ep}, ${sl}, ${tg})">
            ⚡ Execute ${act} Paper Trade (₹${num(ep)})
          </button>
        ` : ''}
        <button class="btn" style="padding:6px 12px" onclick="triggerAgentDebate()">🔄 Re-debate</button>
      </div>
    </div>
    <div id="aiTradeExecMsg" style="margin-top:8px;font-size:11.5px;font-weight:600"></div>
  `;
}

async function executeAiTradeDirect(sym, side, px, sl, tg){
  const btn = document.getElementById("btnExecAiTrade");
  const msg = document.getElementById("aiTradeExecMsg");
  if(btn) btn.disabled = true;
  if(msg){
    msg.style.color = "var(--ink-2)";
    msg.textContent = "Placing paper trade in SQLite ledger…";
  }

  try {
    const res = await fetch("/api/trade", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        symbol: sym,
        side: side,
        qty: 15,
        price: px,
        stop_loss: sl,
        target: tg,
        strategy_id: "agent_debate_desk"
      })
    });
    const j = await res.json();
    if(j.ok){
      if(msg){
        msg.style.color = "var(--good)";
        msg.textContent = "✓ Order executed! Trade #" + j.trade_id + " recorded in ledger. Live P&L tracked in Dashboard.";
      }
      if(btn){
        btn.textContent = "✓ Trade #" + j.trade_id + " Placed";
        btn.style.background = "var(--good)";
      }
    } else {
      if(msg){
        msg.style.color = "var(--bad)";
        msg.textContent = "✗ Rejected by Risk Kernel: " + (j.reason || j.error || "Order rejected");
      }
      if(btn) btn.disabled = false;
    }
  } catch(err){
    if(msg){
      msg.style.color = "var(--bad)";
      msg.textContent = "✗ Error: " + err;
    }
    if(btn) btn.disabled = false;
  }
}

loadQuotes();
setInterval(loadQuotes, 2000);
__THEMEJS__
</script>
</body></html>
"""


_TOPBAR_RIGHT = """    <div class="live"><span class="led" id="led"></span><span id="stamp">—</span></div>
    <div class="seg" id="seg">
      <button data-f="all" aria-pressed="true">All <span class="n" id="n-all"></span></button>
      <button data-f="BUY" aria-pressed="false">Buy <span class="n" id="n-BUY"></span></button>
      <button data-f="WATCH" aria-pressed="false">Watch <span class="n" id="n-WATCH"></span></button>
      <button data-f="AVOID" aria-pressed="false">Avoid <span class="n" id="n-AVOID"></span></button>
    </div>
    <button class="btn go" id="rescan">Scan now</button>
    <label class="chk"><input type="checkbox" id="auto" checked> auto</label>"""

PAGE = (_PAGE_TEMPLATE
        .replace("__THEME__", ui_theme.CSS)
        .replace("__TOPBAR__", ui_theme.topbar("NSE F&amp;O Long Scanner", "scanner",
                                               _TOPBAR_RIGHT))
        .replace("__NIFTYJS__", ui_theme.NIFTY_JS)
        .replace("__CONTEXTJS__", ui_theme.CONTEXT_JS)
        .replace("__THEMEJS__", ui_theme.THEME_JS))


def _serve_forever(server, url: str) -> int:
    """Run the server, turning the two normal failures into plain sentences."""
    print(f"{url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


def _bind(port: int, handler, what: str):
    """Bind, or explain who already has the port."""
    from http.server import ThreadingHTTPServer

    try:
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        if getattr(exc, "errno", None) not in (48, 98):      # EADDRINUSE
            raise
        print(f"Port {port} is already in use — something is serving there.\n"
              f"  Look:  open http://localhost:{port}\n"
              f"  Who:   lsof -nP -iTCP:{port} -sTCP:LISTEN\n"
              f"  Free:  lsof -ti tcp:{port} | xargs kill\n"
              f"  Or run {what} on another port, e.g. {port + 1}")
        return None


def main(argv: list[str] | None = None) -> int:
    import argparse

    global CACHE
    ap = argparse.ArgumentParser(prog="python -m trading.fno.web",
                                 description="Web UI for the F&O long scanner")
    ap.add_argument("port", nargs="?", type=int, default=8788)
    ap.add_argument("--replay", metavar="BUNDLE.json",
                    help="serve a saved scan instead of the live feeds")
    ap.add_argument("--symbols", help="comma-separated list; skips the screen")
    ap.add_argument("--shortlist", type=int, default=C.SHORTLIST_SIZE)
    ap.add_argument("--ttl", type=int, default=DEFAULT_TTL,
                    help=f"seconds before a cached scan is re-run (default {DEFAULT_TTL})")
    ap.add_argument("--fyers", action="store_true",
                    help="real-time candles via the TradeBrahma Fyers server")
    ap.add_argument("--fyers-base", default="http://localhost:3001")
    ap.add_argument("--live-only", action="store_true",
                    help="refuse to grade delayed candles (same rule as the CLI "
                         "without --allow-delayed)")
    args = ap.parse_args(argv)

    fyers.start_background_server()
    server = _bind(args.port, Handler, "the scanner")
    if server is None:
        return 1

    CACHE = ScanCache(allow_delayed=not args.live_only, shortlist=args.shortlist,
                      symbols=args.symbols.split(",") if args.symbols else None,
                      replay=args.replay, ttl=args.ttl, fyers=args.fyers,
                      fyers_base=args.fyers_base)
    CACHE.start()
    return _serve_forever(server, f"F&O scanner UI: http://localhost:{args.port}")


if __name__ == "__main__":
    sys.exit(main())
