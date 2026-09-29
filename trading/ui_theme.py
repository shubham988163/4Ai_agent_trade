"""One design system for both local pages — the dashboard and the F&O scanner.

Kept in a single module so the two views cannot drift apart: a token added here
appears on both, and neither page carries its own copy of the palette.

The colours were checked with a palette validator against the real card
surfaces rather than by eye. Two results are load-bearing:

  * every status step clears 4.5:1 as text, because these colours label actual
    prices (a stop, a target), not just marks;
  * the stop/target pair is separated by LIGHTNESS as well as hue, because
    red-vs-green is the pair colour-blind readers lose and mistaking a stop for
    a target is the costliest error these pages can cause. On the near-black
    surface that pushes the green above the categorical lightness band — a
    deliberate trade, since the contrast floor and that ceiling cannot both
    hold while keeping the pair separable.

Dark is the designed default (a desk runs dark); light is a fully worked
alternate, not an inverted afterthought.
"""
from __future__ import annotations

CSS = r"""
:root{
  --bg:#04070f; --bg-2:#070b14; --card:#0b1220; --cell:#0e1728; --cell-2:#111c31;
  --line:#1a2439; --line-2:#26344f;
  --ink:#f1f5f9; --ink-2:#9fb0c7; --ink-3:#6b7c94;
  --accent:#2f9fdb; --good:#34d399; --warn:#c98500; --bad:#e05a70;
  --good-soft:rgba(52,211,153,.14); --bad-soft:rgba(224,90,112,.14);
  --warn-soft:rgba(201,133,0,.16); --accent-soft:rgba(47,159,219,.14);
  --track:#16233a; --link:#5cb3ea;
  --up:#34d399; --down:#e05a70;
  --shadow:0 1px 0 rgba(255,255,255,.03) inset, 0 10px 30px rgba(0,0,0,.45);
  color-scheme:dark;
}
:root[data-theme="light"]{
  --bg:#f4f6f9; --bg-2:#ffffff; --card:#ffffff; --cell:#f8fafc; --cell-2:#f1f5f9;
  --line:#e2e8f0; --line-2:#cbd5e1;
  --ink:#0b1220; --ink-2:#475569; --ink-3:#64748b;
  --accent:#0369a1; --good:#15803d; --warn:#a06800; --bad:#9f1239;
  --good-soft:rgba(21,128,61,.10); --bad-soft:rgba(159,18,57,.10);
  --warn-soft:rgba(160,104,0,.10); --accent-soft:rgba(3,105,161,.10);
  --track:#e2e8f0; --link:#1f66c0;
  --up:#15803d; --down:#9f1239;
  --shadow:0 1px 2px rgba(15,23,42,.06);
  color-scheme:light;
}
*{box-sizing:border-box}
a{color:var(--link)}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
/* A faint grid gives the console depth without competing with the data. */
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(148,163,184,.022) 1px,transparent 1px),
    linear-gradient(90deg,rgba(148,163,184,.022) 1px,transparent 1px);
  background-size:44px 44px;
  mask-image:radial-gradient(ellipse 90% 70% at 50% 0%,#000 20%,transparent 100%)}
:root[data-theme="light"] body::before{display:none}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.wrap{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:0 18px 56px}
.muted{color:var(--ink-3)}
.up{color:var(--up)} .down{color:var(--down)}
.scroll{overflow-x:auto}

/* ---------- top bar ---------- */
.top{position:sticky;top:0;z-index:20;margin:0 -18px;padding:10px 18px;
  background:color-mix(in srgb,var(--bg-2) 92%,transparent);
  backdrop-filter:blur(14px);border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;min-width:0}
.mark{width:26px;height:26px;border-radius:7px;flex:none;display:grid;place-items:center;
  background:linear-gradient(145deg,var(--accent),#6366f1);color:#04070f;
  font-weight:800;font-size:11px;letter-spacing:-.5px}
.brand h1{margin:0;font-size:13px;font-weight:700;letter-spacing:.10em;
  text-transform:uppercase;white-space:nowrap}
.brand .win{color:var(--ink-3);font-size:11px;letter-spacing:.04em}
.spacer{flex:1 1 auto}
.nav{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;
  background:var(--cell)}
.nav a{padding:6px 12px;font-size:11px;font-weight:600;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink-3);text-decoration:none}
.nav a+a{border-left:1px solid var(--line)}
.nav a[aria-current=page]{background:var(--accent-soft);color:var(--accent)}
.live{display:inline-flex;align-items:center;gap:7px;font-size:11px;color:var(--ink-2);
  letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.led{width:7px;height:7px;border-radius:50%;background:var(--good);flex:none}
.led.busy{background:var(--accent);animation:pulse 1.1s ease-in-out infinite}
.led.stale{background:var(--bad)}
@keyframes pulse{50%{opacity:.25}}
.spin{width:10px;height:10px;border:2px solid currentColor;border-top-color:transparent;
  border-radius:50%;animation:sp .7s linear infinite;display:inline-block}
@keyframes sp{to{transform:rotate(360deg)}}

/* ---------- controls ---------- */
.btn,select.btn{background:var(--cell);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:6px 11px;font:inherit;font-size:11px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;cursor:pointer;
  display:inline-flex;align-items:center;gap:6px}
.btn:hover{background:var(--cell-2);border-color:var(--line-2)}
.btn[disabled]{opacity:.5;cursor:default}
.btn.go{border-color:color-mix(in srgb,var(--accent) 45%,var(--line));color:var(--accent)}
select.btn{text-transform:none;letter-spacing:0;font-weight:500}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;
  background:var(--cell)}
.seg button{background:transparent;color:var(--ink-3);border:0;padding:6px 11px;
  font:inherit;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed=true]{background:var(--accent-soft);color:var(--accent)}
.seg button .n{color:var(--ink-3);margin-left:5px;font-weight:500}
.seg button[aria-pressed=true] .n{color:inherit;opacity:.75}
.chk{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:var(--ink-2);
  letter-spacing:.05em;text-transform:uppercase;cursor:pointer}

/* ---------- surfaces ---------- */
section{margin-top:14px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:11px;
  box-shadow:var(--shadow);overflow:hidden}
.panel > h2{margin:0;padding:9px 13px;font-size:10px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-3);border-bottom:1px solid var(--line);
  background:var(--cell);font-weight:600;display:flex;align-items:center;gap:8px}
.panel > h2 .sub{color:var(--ink-3);font-weight:400;letter-spacing:.03em;
  text-transform:none;font-size:11px}
.panel .body{padding:12px 13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px;margin-top:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:11px;
  padding:11px 13px;box-shadow:var(--shadow)}
.kpi .k{font-size:10px;letter-spacing:.10em;text-transform:uppercase;color:var(--ink-3)}
.kpi .v{font-size:22px;font-weight:650;margin-top:3px;letter-spacing:-.01em}
.kpi .v.hero{font-size:40px;line-height:1.05;font-weight:700}
.kpi .f{font-size:11px;color:var(--ink-3);margin-top:3px}

/* ---------- badges: glyph + word carry the state, colour reinforces ------- */
.badge{display:inline-flex;align-items:center;gap:6px;border-radius:7px;padding:4px 10px;
  font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  border:1px solid var(--line-2);background:var(--cell);color:var(--ink)}
.badge .g{font-size:12px;line-height:1}
.badge.sm{padding:3px 8px;font-size:10px}
.badge.v-buy,.badge.ok,.badge.approve{border-color:var(--good);background:var(--good-soft)}
.badge.v-buy .g,.badge.ok .g,.badge.approve .g{color:var(--good)}
.badge.v-watch,.badge.caution{border-color:var(--warn);background:var(--warn-soft)}
.badge.v-watch .g,.badge.caution .g{color:var(--warn)}
.badge.v-avoid .g{color:var(--ink-3)}
.badge.veto,.badge.err{border-color:var(--bad);background:var(--bad-soft)}
.badge.veto .g,.badge.err .g{color:var(--bad)}
.badge.bull{border-color:var(--good);background:var(--good-soft)}
.badge.bull .g{color:var(--good)}
.badge.bear{border-color:var(--bad);background:var(--bad-soft)}
.badge.bear .g{color:var(--bad)}

/* ---------- tables ---------- */
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:var(--cell);color:var(--ink-3);font-weight:600;text-align:left;
  padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap;
  font-size:10px;letter-spacing:.09em;text-transform:uppercase}
td{padding:7px 10px;border-bottom:1px solid var(--line);color:var(--ink-2);
  white-space:nowrap;vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.strong{color:var(--ink);font-weight:600}
tbody tr:hover td{background:var(--cell)}
.empty{padding:16px;color:var(--ink-3);font-size:12.5px;text-align:center}

/* ---------- NIFTY 50 index-option panel (shared by both pages) ---------- */
.nfx .hd{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:11px 13px}
.nfx .spot{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.nfx .st{font-size:11px;color:var(--ink-2);letter-spacing:.04em}
.nfx .call{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap;
  padding:0 13px 11px}
.nfx .strike{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}
.nfx .nocall{color:var(--bad);font-weight:700;letter-spacing:.06em;font-size:11px}
.nfx .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));
  gap:1px;background:var(--line);border-top:1px solid var(--line)}
.nfx .grid div{background:var(--cell);padding:8px 11px}
.nfx .grid .k{font-size:9px;letter-spacing:.10em;text-transform:uppercase;color:var(--ink-3)}
.nfx .grid .v{font-size:14px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.nfx .grid .v.bad{color:var(--bad)} .nfx .grid .v.good{color:var(--good)}
.nfx .grid .n{font-size:10px;color:var(--ink-3);margin-top:1px}
.nfx .lvl{padding:9px 13px;font-size:12px;color:var(--ink-2);
  border-top:1px solid var(--line);display:flex;gap:8px 16px;flex-wrap:wrap}
.nfx .lvl b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.nfx ul{margin:0;padding:9px 13px 9px 30px;font-size:12px;color:var(--ink-2);
  border-top:1px solid var(--line)}
.nfx ul.bad li{color:var(--bad)}
.nfx .decay{display:flex;gap:8px 14px;flex-wrap:wrap;align-items:baseline;
  padding:9px 13px;border-top:1px solid var(--line);font-size:12px;
  color:var(--ink-2);font-variant-numeric:tabular-nums}
.nfx .decay .k{font-size:9px;letter-spacing:.10em;text-transform:uppercase;
  color:var(--ink-3)}
.nfx .cav{padding:8px 13px;font-size:10.5px;color:var(--ink-3);
  border-top:1px solid var(--line)}

/* ---------- chain positioning & levels ---------- */
.ctxp .ctx{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
  gap:1px;background:var(--line)}
.ctxp .ctx > div{background:var(--card);padding:9px 12px}
.ctxp .ctx .k{font-size:9px;letter-spacing:.10em;text-transform:uppercase;color:var(--ink-3)}
.ctxp .ctx .v{font-size:15px;font-weight:650;margin-top:2px}
.ctxp .ctx .n{font-size:10px;color:var(--ink-3);margin-top:1px}
.ctxp .cav{padding:8px 12px;font-size:10.5px;color:var(--ink-3);
  border-top:1px solid var(--line)}

/* ---------- 4 AI Agents Council styles ---------- */
.agent-card{background:var(--cell);border:1px solid var(--line);border-radius:10px;padding:14px;transition:all .2s ease;display:flex;flex-direction:column;gap:10px;box-shadow:var(--shadow)}
.agent-card:hover{border-color:var(--line-2);transform:translateY(-2px)}
.agent-header{display:flex;align-items:center;gap:10px}
.agent-avatar{width:34px;height:34px;border-radius:9px;display:grid;place-items:center;font-size:17px;flex:none}
.agent-tech .agent-avatar{background:linear-gradient(135deg,#0284c7,#38bdf8);box-shadow:0 0 10px rgba(56,189,248,.3)}
.agent-bull .agent-avatar{background:linear-gradient(135deg,#059669,#34d399);box-shadow:0 0 10px rgba(52,211,153,.3)}
.agent-bear .agent-avatar{background:linear-gradient(135deg,#e11d48,#fb7185);box-shadow:0 0 10px rgba(251,113,133,.3)}
.agent-trader .agent-avatar{background:linear-gradient(135deg,#d97706,#fbbf24);box-shadow:0 0 10px rgba(251,191,36,.3)}
.agent-name{font-weight:700;font-size:13px;color:var(--ink);letter-spacing:.02em}
.agent-role{font-size:10.5px;color:var(--ink-3);letter-spacing:.04em;text-transform:uppercase}
.agent-status-pill{margin-left:auto;font-size:10px;font-weight:600;padding:2px 7px;border-radius:999px;display:inline-flex;align-items:center;gap:4px}
.agent-body{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:10px;font-size:12px;color:var(--ink-2);line-height:1.55;min-height:90px;flex:1}
.conviction-meter{background:var(--track);height:8px;border-radius:999px;overflow:hidden;margin:6px 0;position:relative}
.conviction-fill{height:100%;border-radius:999px;transition:width .4s ease}

/* ---------- banner + footer ---------- */
.banner{margin-top:14px;border:1px solid var(--bad);background:var(--bad-soft);
  border-radius:10px;padding:11px 14px;font-size:12.5px;color:var(--ink)}
.banner b{display:block;color:var(--bad);letter-spacing:.05em;text-transform:uppercase;
  font-size:11px;margin-bottom:3px}
.foot{margin-top:18px;padding-top:12px;border-top:1px solid var(--line);
  font-size:11px;color:var(--ink-3);line-height:1.6}
@keyframes slideIn{from{opacity:0;transform:translateY(-12px) scale(0.96)}to{opacity:1;transform:translateY(0) scale(1)}}
"""


def topbar(title: str, active: str, right: str = "", subtitle_id: str = "win") -> str:
    """The shared header. `active` is "dashboard" or "scanner"."""
    def cur(name: str) -> str:
        return ' aria-current="page"' if name == active else ""
    return f"""  <header class="top">
    <div class="brand">
      <div class="mark">F&amp;O</div>
      <div>
        <h1>{title}</h1>
        <div class="win mono" id="{subtitle_id}">…</div>
      </div>
    </div>
    <nav class="nav">
      <a href="/"{cur("dashboard")}>Dashboard</a>
      <a href="/#agents-desk" style="border-left:1px solid var(--line);color:var(--accent);font-weight:700">🤖 4 Agents Desk</a>
      <a href="/fno"{cur("scanner")}>Scanner</a>
    </nav>
    <div class="spacer"></div>
    <button class="btn" id="btnAlertToggle" title="Toggle Audio Chimes &amp; Desktop Alerts" onclick="window.ScannerAlerts &amp;&amp; window.ScannerAlerts.toggle()" style="font-size:11px;padding:4px 10px;display:inline-flex;align-items:center;gap:5px">
      <span id="alertIcon">🔔</span> <span id="alertLabel">Alerts ON</span>
    </button>
    <a id="fyersPill" class="btn" style="font-size:11px;padding:3px 9px;text-decoration:none;display:inline-flex;align-items:center;gap:4px" target="_blank" href="/api/fyers/login">Fyers…</a>
{right}
    <button class="btn" id="theme" title="Switch dark / light / system">Dark</button>
  </header>"""


# One implementation of the NIFTY panel, injected into both pages, so the
# dashboard and the scanner cannot drift on how an index trade is presented.
# `nf` is the host page's number formatter — the two pages name theirs
# differently, so it is passed in rather than assumed.
NIFTY_JS = r"""
function niftyPanel(ix, nf){
  const mk = (t,c,x) => { const n=document.createElement(t); if(c) n.className=c;
    if(x!==undefined) n.textContent=x; return n; };
  const box = mk("section","panel nfx");
  const h2 = mk("h2"); h2.append(document.createTextNode("NIFTY 50 option"));
  if(!ix){
    box.append(h2);
    box.append(mk("div","empty","No index read this scan."));
    return box;
  }
  const buyable = ix.tradeable && ix.option && ix.option.quote;
  const badge = mk("span","badge sm " + (buyable ? "v-buy" : "v-avoid"));
  badge.append(mk("span","g", buyable ? "\u2713" : "\u2715"));
  badge.append(document.createTextNode(buyable ? "BUY" : "NO TRADE"));
  h2.append(badge);
  box.append(h2);

  const hd = mk("div","hd");
  hd.append(mk("span","spot mono", nf(ix.spot)));
  hd.append(mk("span","st", ix.status));
  hd.append(mk("span","st", "range " + nf(ix.or_low) + " – " + nf(ix.or_high)));
  box.append(hd);

  const o = ix.option;
  const call = mk("div","call");
  if(o && o.quote){
    const q = o.quote;
    call.append(mk("span","strike mono", q.strike + " CE @ " + nf(q.ltp)));
    call.append(mk("span","chip", q.expiry));
    call.append(mk("span","chip", q.moneyness));
    call.append(mk("span","st", "OI " + (q.open_interest||0).toLocaleString("en-IN")));
  } else {
    call.append(mk("span","nocall","NO CALL"));
    call.append(mk("span","st", (o && o.rejections && o.rejections[0]) ||
      (ix.rejections && ix.rejections[0]) || "no candidate strike"));
  }
  box.append(call);

  if(o && o.quote && o.premium_at_t1 !== null && o.premium_at_t1 !== undefined){
    const g = mk("div","grid");
    const cell=(k,v,cls,n)=>{const d=mk("div"); d.append(mk("div","k",k));
      d.append(mk("div","v "+(cls||""),v)); if(n) d.append(mk("div","n",n)); g.append(d);};
    const pct = p => o.quote.ltp>0 ? ((p-o.quote.ltp)/o.quote.ltp*100).toFixed(0)+"%" : "";
    cell("Pay now","\u20b9"+nf(o.quote.ltp));
    if(o.premium_at_stop!==null && o.premium_at_stop!==undefined)
      cell("At index stop","\u20b9"+nf(o.premium_at_stop),"bad",pct(o.premium_at_stop));
    cell("At target 1","\u20b9"+nf(o.premium_at_t1),"good",pct(o.premium_at_t1));
    cell("At target 2","\u20b9"+nf(o.premium_at_t2),"good",pct(o.premium_at_t2));
    if(o.option_rr!==null && o.option_rr!==undefined)
      cell("Option R:R", o.option_rr.toFixed(2)+":1", o.option_rr>=1?"good":"bad");
    box.append(g);
  }

  if(ix.target1){
    const l = mk("div","lvl");
    const kv=(k,v)=>{const s=mk("span"); s.append(document.createTextNode(k+" "));
      s.append(mk("b",null,nf(v))); l.append(s);};
    kv("index entry", ix.entry_low); kv("–", ix.entry_high);
    kv("stop", ix.stop); kv("T1", ix.target1); kv("T2", ix.target2);
    box.append(l);
  }

  const list = (items, cls) => {
    if(!items || !items.length) return;
    const ul = mk("ul", cls);
    items.forEach(x => ul.append(mk("li", null, x)));
    box.append(ul);
  };
  list(buyable ? ix.reasons : ix.rejections, buyable ? null : "bad");
  if(o && o.warnings && o.warnings.length) list(o.warnings, "bad");
  if(o && o.decay && o.decay.length){
    const d = mk("div","decay");
    d.append(mk("span","k","If NIFTY does not move"));
    o.decay.forEach(x => {
      const s2 = mk("span","st");
      s2.textContent = (x.days < 1 ? "tonight" : "+" + x.days + "d") + " "
        + nf(x.premium) + (x.pct_left !== null ? " (" + Math.round(x.pct_left) + "%)" : "");
      d.append(s2);
    });
    box.append(d);
  }
  if(ix.caveats && ix.caveats.length)
    box.append(mk("div","cav","\u26a0 " + ix.caveats[0]));
  return box;
}
"""

CONTEXT_JS = r"""
// Chain positioning and yesterday's levels. Context for every candidate, so it
// lives in one panel rather than being repeated on each card.
function contextPanel(chain, piv, nf){
  const mk = (t,c,x) => { const n=document.createElement(t); if(c) n.className=c;
    if(x!==undefined) n.textContent=x; return n; };
  if(!chain && !piv) return null;
  const box = mk("section","panel ctxp");
  box.append(mk("h2","", "Chain positioning & levels"));
  const body = mk("div","ctx");

  const cell = (k, v, note) => {
    const d = mk("div");
    d.append(mk("div","k",k));
    d.append(mk("div","v mono",v));
    if(note) d.append(mk("div","n",note));
    body.append(d);
  };
  if(chain){
    if(chain.pcr !== null && chain.pcr !== undefined)
      cell("PCR", chain.pcr.toFixed(2), chain.pcr_label);
    if(chain.max_pain !== null && chain.max_pain !== undefined)
      cell("Max pain", nf(chain.max_pain),
           chain.max_pain_useful ? chain.expiry
             : (chain.days_to_expiry + "d out — not an intraday level"));
    if(chain.resistance_strike)
      cell("OI resistance", chain.resistance_strike + " CE",
           Math.round(chain.resistance_oi).toLocaleString("en-IN") + " OI");
    if(chain.support_strike)
      cell("OI support", chain.support_strike + " PE",
           Math.round(chain.support_oi).toLocaleString("en-IN") + " OI");
  }
  if(piv){
    cell("Pivot", nf(piv.pivot),
         "CPR " + nf(piv.bc) + "–" + nf(piv.tc));
    cell("CPR width", piv.cpr_width_pct.toFixed(2) + "%", piv.cpr_label);
    cell("R1 / S1", nf(piv.r1) + " / " + nf(piv.s1),
         "R2 " + nf(piv.r2) + " · S2 " + nf(piv.s2));
    if(piv.gap) cell("Open", nf(piv.gap.open), piv.gap.label);
  }
  box.append(body);
  box.append(mk("div","cav","\u26a0 positioning and levels are context, not "
    + "signals — open interest is not directional, and pivots work largely "
    + "because many traders watch the same ones"));
  return box;
}
"""

THEME_JS = r"""
const THEMES=["dark","light","system"];
function applyTheme(t){
  if(t==="system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme",t);
  const b=document.getElementById("theme");
  if(b) b.textContent = t.charAt(0).toUpperCase()+t.slice(1);
  try{ localStorage.setItem("fno-theme",t); }catch(e){}
}

window.ScannerAlerts = (function(){
  let audioCtx = null;
  let soundEnabled = localStorage.getItem("fno-sound-alerts") !== "false";
  let desktopEnabled = localStorage.getItem("fno-desktop-alerts") !== "false";
  let seenAlerts = new Set();
  let flashInterval = null;
  let originalTitle = document.title;

  function initAudio(){
    if(!audioCtx){
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if(AudioContext) audioCtx = new AudioContext();
    }
    if(audioCtx && audioCtx.state === 'suspended'){
      audioCtx.resume();
    }
  }

  function playSound(type){
    if(!soundEnabled) return;
    try{
      initAudio();
      if(!audioCtx) return;
      const now = audioCtx.currentTime;

      if(type === 'buy'){
        const notes = [523.25, 659.25, 783.99, 1046.50];
        notes.forEach((freq, idx) => {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(freq, now + idx * 0.08);
          gain.gain.setValueAtTime(0.25, now + idx * 0.08);
          gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.08 + 0.35);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(now + idx * 0.08);
          osc.stop(now + idx * 0.08 + 0.4);
        });
      } else if(type === 'watch'){
        const notes = [587.33, 880.00];
        notes.forEach((freq, idx) => {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, now + idx * 0.12);
          gain.gain.setValueAtTime(0.2, now + idx * 0.12);
          gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.12 + 0.3);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(now + idx * 0.12);
          osc.stop(now + idx * 0.12 + 0.35);
        });
      } else {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.exponentialRampToValueAtTime(1200, now + 0.15);
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.3);
      }
    }catch(e){
      console.warn("Audio error:", e);
    }
  }

  function notifyDesktop(title, body, sym){
    if(!desktopEnabled || !("Notification" in window)) return;
    if(Notification.permission === "granted"){
      try{
        const n = new Notification(title, {
          body: body,
          icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>",
          tag: "fno-" + sym,
          requireInteraction: false
        });
        n.onclick = function(){
          window.focus();
          focusStock(sym);
        };
      }catch(e){}
    } else if(Notification.permission !== "denied"){
      Notification.requestPermission();
    }
  }

  function showToast(opt){
    let container = document.getElementById("toastContainer");
    if(!container){
      container = document.createElement("div");
      container.id = "toastContainer";
      container.style.cssText = "position:fixed;top:60px;right:18px;z-index:99999;display:flex;flex-direction:column;gap:10px;pointer-events:none;max-width:380px;width:calc(100% - 36px);";
      document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = "alert-toast";
    toast.style.cssText = "pointer-events:auto;background:color-mix(in srgb, var(--card) 92%, transparent);backdrop-filter:blur(16px);border:1px solid var(--line-2);border-left:4px solid " + (opt.type==='BUY'?'var(--good)':'var(--warn)') + ";border-radius:10px;padding:12px 14px;box-shadow:0 12px 36px rgba(0,0,0,0.5);display:flex;flex-direction:column;gap:6px;animation:slideIn 0.3s cubic-bezier(0.16,1,0.3,1);transition:all 0.3s ease;";

    const isBuy = opt.type === 'BUY';
    toast.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:${isBuy?'var(--good)':'var(--warn)'}">
          <span>${isBuy?'⚡ NEW BUY SETUP':'◔ WATCH BREAKOUT'}</span>
        </div>
        <button style="background:none;border:none;color:var(--ink-3);cursor:pointer;font-size:14px;padding:0 4px;" onclick="this.closest('.alert-toast').remove()">✕</button>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:2px;">
        <span style="font-size:18px;font-weight:700;color:var(--ink);">${opt.symbol}</span>
        <span style="font-size:16px;font-weight:700;font-family:ui-monospace,monospace;color:var(--ink);">₹${opt.price}</span>
      </div>
      <div style="font-size:11.5px;color:var(--ink-2);line-height:1.4;">
        ${opt.details || ''}
      </div>
      <div style="display:flex;gap:8px;margin-top:4px;">
        <button class="btn go" style="flex:1;padding:4px 8px;font-size:11px;" onclick="window.ScannerAlerts.focusStock('${opt.symbol}'); this.closest('.alert-toast').remove();">View Setup →</button>
      </div>
    `;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      setTimeout(() => toast.remove(), 350);
    }, 8000);
  }

  function flashTitle(msg){
    if(flashInterval) clearInterval(flashInterval);
    let toggle = false;
    flashInterval = setInterval(() => {
      if(!document.hidden){
        clearInterval(flashInterval);
        flashInterval = null;
        document.title = originalTitle;
        return;
      }
      document.title = toggle ? `🔔 ${msg}` : originalTitle;
      toggle = !toggle;
    }, 1000);
  }

  window.addEventListener("focus", () => {
    if(flashInterval){
      clearInterval(flashInterval);
      flashInterval = null;
      document.title = originalTitle;
    }
  });

  function focusStock(sym){
    const el = document.querySelector(`[data-card-sym="${sym}"]`) || document.getElementById("buynow");
    if(el){
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.style.outline = '2px solid var(--accent)';
      setTimeout(() => { el.style.outline = ''; }, 3000);
    }
  }

  function checkAndAlert(item, type){
    if(!item || !item.symbol) return;
    const key = `${item.symbol}_${type}_${item.structure?.status || item.score || ''}`;
    if(seenAlerts.has(key)) return;
    seenAlerts.add(key);

    const isBuy = type === 'BUY';
    const title = isBuy ? `⚡ BUY Signal: ${item.symbol}` : `◔ Watch Alert: ${item.symbol}`;
    const scoreStr = item.score ? `Score: ${Math.round(item.score)}/100 · ${item.grade || ''}` : '';
    const pxStr = item.price ? `₹${Number(item.price).toLocaleString('en-IN')}` : '';
    const details = `${scoreStr} ${item.reasons && item.reasons[0] ? '· ' + item.reasons[0] : ''}`;

    playSound(isBuy ? 'buy' : 'watch');
    notifyDesktop(title, `${pxStr} | ${details}`, item.symbol);
    showToast({
      symbol: item.symbol,
      type: type,
      price: pxStr.replace('₹',''),
      details: details
    });
    flashTitle(`${type}: ${item.symbol} (${pxStr})`);
  }

  function toggleAlerts(){
    initAudio();
    soundEnabled = !soundEnabled;
    desktopEnabled = soundEnabled;
    localStorage.setItem("fno-sound-alerts", soundEnabled);
    localStorage.setItem("fno-desktop-alerts", desktopEnabled);
    updateButtonUI();

    if(soundEnabled){
      playSound('buy');
      showToast({
        symbol: "SYSTEM TEST",
        type: "BUY",
        price: "READY",
        details: "✓ Alerts and Audio Chimes are now active for all new scanner signals."
      });
      if("Notification" in window && Notification.permission !== "granted"){
        Notification.requestPermission();
      }
    }
  }

  function updateButtonUI(){
    const btn = document.getElementById("btnAlertToggle");
    const icon = document.getElementById("alertIcon");
    const label = document.getElementById("alertLabel");
    if(btn && icon && label){
      if(soundEnabled){
        btn.style.borderColor = "var(--good)";
        btn.style.background = "var(--good-soft)";
        btn.style.color = "var(--good)";
        icon.textContent = "🔔";
        label.textContent = "Alerts ON";
      } else {
        btn.style.borderColor = "var(--line)";
        btn.style.background = "var(--cell)";
        btn.style.color = "var(--ink-3)";
        icon.textContent = "🔕";
        label.textContent = "Alerts OFF";
      }
    }
  }

  return {
    init: function(){
      updateButtonUI();
      document.addEventListener("click", initAudio, { once: true });
    },
    toggle: toggleAlerts,
    playSound: playSound,
    showToast: showToast,
    checkAndAlert: checkAndAlert,
    focusStock: focusStock
  };
})();

(function(){
  const btn=document.getElementById("theme");
  if(btn) btn.addEventListener("click", ()=>{
    let t="dark"; try{ t=localStorage.getItem("fno-theme")||"dark"; }catch(e){}
    applyTheme(THEMES[(THEMES.indexOf(t)+1)%THEMES.length]);
  });
  const q=new URLSearchParams(location.search).get("theme");
  let t=THEMES.includes(q)?q:null;
  if(!t){ try{ t=localStorage.getItem("fno-theme"); }catch(e){} }
  applyTheme(t||"dark");

  if(window.ScannerAlerts) window.ScannerAlerts.init();

  async function checkFyers(){
    try{
      const r = await fetch("/api/fyers/status");
      const st = await r.json();
      const p = document.getElementById("fyersPill");
      if(p){
        if(st.connected){
          p.style.borderColor = "var(--good)";
          p.style.background = "var(--good-soft)";
          p.style.color = "var(--good)";
          p.textContent = "⚡ Fyers Live (" + ((st.profile && st.profile.name) ? st.profile.name : "Connected") + ")";
          p.href = "#";
          p.onclick = (e)=>{ e.preventDefault(); alert("Fyers API v3 is Connected and Streaming Real-Time Candles!"); };
        } else {
          p.style.borderColor = "var(--warn)";
          p.style.background = "var(--warn-soft)";
          p.style.color = "var(--warn)";
          p.textContent = "🔑 Connect Fyers";
          p.href = st.auth_url || "http://localhost:3001/api/fyers/login";
          p.onclick = null;
        }
      }
    }catch(e){}
  }
  checkFyers();
  setInterval(checkFyers, 10000);
})();
"""
