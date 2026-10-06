# AI Trade Agent — AI-supervised paper trading for NSE

Paper-trading system for NSE intraday scalping with an advisory AI layer:

```
[Market Data] → [Strategy Engine] → [AI Agent Layer] → [Execution Router] → [SQLite Ledger] → [Dashboard]
 (yfinance /       (EMA 9/21 or       (advisory only)     (hard risk kernel)
  TradingView)      Pine webhook)
```

**Design rule enforced throughout: the AI agent advises and supervises;
deterministic code decides and executes.** Every agent output is schema-validated,
clamped to safe ranges, logged to the audit trail, and replaced with safe
defaults on any failure. Hard risk limits live in `ExecutionRouter` and cannot
be overridden by any agent output. Live trading is intentionally
`NotImplementedError` until the go-live checklist at the bottom is complete.

## Quick start (any machine)

Requires **Python 3.11+** and `git`. Notifications use macOS `osascript` when
available and silently skip elsewhere; Telegram works everywhere (optional, see below).

```bash
git clone git@github.com:shubhambirajdar07/ai-tradeagent.git
cd ai-tradeagent

# 1. Create the virtualenv and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Set your LLM key (either provider works; Gemini is auto-selected when its key is set)
export GEMINI_API_KEY=...             # Google AI Studio key (free tier → gemini-2.5-flash)
# export ANTHROPIC_API_KEY=sk-ant-... # or Claude; force a provider with LLM_PROVIDER=anthropic|gemini
# Put the export in your ~/.zshrc or ~/.bashrc so it survives new terminals.

# 3. Verify the whole pipeline end-to-end (fake prices, no market needed)
.venv/bin/python -m trading.demo

# 4. Open the dashboard
.venv/bin/python -m trading.dashboard      # → http://localhost:8787
```

The `data/`, `logs/`, and ledger database are created automatically on first
run — they are machine-local and never committed.

## One command per trading day

```bash
cd ai-tradeagent
./scripts/start_day.sh
```

This starts everything for the session:

| Process | What it does |
|---|---|
| Dashboard | http://localhost:8787 — P&L chart, buy/sell + per-strategy reports, trades with AI verdicts, rejections, LLM audit log, journal |
| Supervisor `--loop` | Reviews every executed trade asynchronously (approve/caution/veto + reasons), with quota backoff |
| Strategy engine | EMA 9/21 crossover on 5-min candles across all Nifty 50 symbols, swing-low SL, 1:2 target, 15:15 square-off |
| TV bridge | Webhook receiver + Cloudflare tunnel for TradingView alerts (needs `cloudflared`; skip by Ctrl-C if unused) |

Run the pre-market analyst before the open (or cron it, below):

```bash
.venv/bin/python -m trading.agents.premarket   # writes data/today_config.json
```

And the journal after the close:

```bash
.venv/bin/python -m trading.agents.eod_journal # writes reports/YYYY-MM-DD.md
```

## Individual commands

```bash
.venv/bin/python -m trading.demo               # end-to-end smoke test (fake prices)
.venv/bin/python -m trading.dashboard [port]   # web dashboard (default 8787)
.venv/bin/python -m trading.strategy           # live scan loop during market hours
.venv/bin/python -m trading.strategy --once    # one scan pass, then exit
.venv/bin/python -m trading.strategy --replay [YYYY-MM-DD]  # replay a session bar-by-bar
.venv/bin/python -m trading.strategy --strategy avwap       # AVWAP scalp instead of EMA 9/21
.venv/bin/python -m trading.agents.premarket   # Role 1: regime + risk multiplier + blocked symbols
.venv/bin/python -m trading.agents.supervisor --loop   # Role 2: async per-trade review
.venv/bin/python -m trading.agents.eod_journal [date]  # Role 3: daily journal report
.venv/bin/python -m trading.import_tv file.csv # import a TradingView Strategy Tester CSV
./scripts/tv_bridge.sh                         # TradingView webhook receiver + tunnel
.venv/bin/python -m trading.backtest           # cost-aware backtest (no ledger writes)
.venv/bin/python -m trading.test_backtest      # harness correctness tests
.venv/bin/python -m trading.costs              # charge table + breakeven move by size
```

Models and all tunables (watchlist, risk limits, EMA lengths, slippage/charges)
live in `trading/config.py`. Note: `gemini-2.5-pro` requires billing on the
Google project — the free tier only serves the flash models.

### Optional: Telegram notifications

Works on any OS (macOS pop-ups are automatic):

```bash
export TELEGRAM_BOT_TOKEN=...   # from @BotFather
export TELEGRAM_CHAT_ID=...
```

## Components

| File | Role |
|---|---|
| `trading/config.py` | All tunables: universe, risk kernel limits, strategy params, models |
| `trading/ledger.py` | SQLite ledger: trades, rejections, full LLM audit log |
| `trading/execution_router.py` | Paper/live switch, hard risk limits (daily loss, position value, max 5 open positions), SEBI sub-10-OPS rate limiter |
| `trading/strategy.py` | EMA 9/21 engine: Nifty-50 scan, MAE/MFE tracking, SL/target exits, square-off |
| `trading/agents/llm.py` | Dual-provider wrapper (Gemini/Anthropic): structured output via Pydantic, fallbacks, audit logging |
| `trading/agents/premarket.py` | **Role 1** — 8:45 AM: regime + risk multiplier + blocked symbols → `today_config.json` |
| `trading/agents/supervisor.py` | **Role 2** — async per-trade sanity check; verdict logged, never blocks a trade |
| `trading/agents/eod_journal.py` | **Role 3** — 4:00 PM: pattern analysis of the day's ledger → `reports/YYYY-MM-DD.md` |
| `trading/tv_webhook.py` | TradingView alert receiver → same risk kernel and ledger |
| `trading/dashboard.py` | Zero-dependency local web dashboard — leads with the scanner's current call |
| `trading/ui_theme.py` | The one design system both pages import (tokens, top bar, components) |
| `trading/retention.py` | 5-session data retention — prunes the ledger, reports and candle cache |
| `tradingview/*.pine` | Pine v6 strategies (EMA 9/21 and AVWAP scalp) that alert into the webhook |

## Daily schedule (IST)

The pre-market agent must run **before the 09:15 open**. It exits `0` only when
it got a real verdict from the model; if the call fails it still writes the safe
fallback so the day is never left without parameters, but it exits `1` — so a
scheduler's "last result" tells you whether the agent actually spoke. Check
`source` in `data/today_config.json`: `"agent"` or `"fallback"`.

macOS — `./scripts/install_schedule.sh` (launchd, catches up after wake):

```cron
# Pre-market analyst — writes data/today_config.json
45 8 * * 1-5  cd $HOME/myselfproject/ai-tradeagent && .venv/bin/python -m trading.agents.premarket >> logs/premarket.log 2>&1

# EOD journal — writes reports/YYYY-MM-DD.md
0 16 * * 1-5  cd $HOME/myselfproject/ai-tradeagent && .venv/bin/python -m trading.agents.eod_journal >> logs/eod.log 2>&1
```

Windows — run once, from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1
powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Status
```

That registers `RanchoTrade-Premarket` at 08:45 Mon–Fri with *start when
available* (the Windows equivalent of launchd's catch-up). The trigger time is
local, so the machine must be set to IST. The same agent can be run by hand at
any time from the dashboard's **Run Now** button.

## Wiring in your own strategy

Your strategy engine emits signals and hands them to the router:

```python
from trading.execution_router import ExecutionRouter
from trading.ledger import Ledger

router = ExecutionRouter(mode="paper", ledger=Ledger(), get_ltp=my_tick_source)
router.execute({
    "symbol": "RELIANCE", "side": "BUY", "qty": 10, "price": ltp,
    "ts": time.time(), "stop_loss": sl, "target": tgt,
    "strategy_id": "ema_x_v3", "regime": router.day_config["regime"],
})
```

The router re-reads `data/today_config.json` (written by the pre-market agent)
on every order: `risk_multiplier` shrinks the position cap (never grows it) and
`blocked_symbols` are rejected outright.

On exit, call `ledger.record_exit(trade_id, exit_price, charges, mae, mfe)` —
check exits against **live ticks**, not candle closes, and log MAE/MFE so the
journal agent can use them.

## Path A — TradingView Pine Script automation

`tradingview/ema_9_21.pine` is the same EMA strategy for TradingView; its
order-fill alerts POST into `trading/tv_webhook.py`, which routes through the
same risk kernel and ledger (tagged `strategy_id=tv_ema_9_21`), so TV trades
appear on the dashboard and get supervised/journaled like everything else.
`tradingview/avwap_scalp.pine` is an AVWAP mean-reversion scalper wired the
same way.

One command starts the receiver + a free Cloudflare tunnel
(`brew install cloudflared` on macOS) and prints the webhook URL and secret to
paste into TradingView:

```bash
./scripts/tv_bridge.sh
```

(The secret is generated once into `data/tv_secret`; the trycloudflare URL
changes on every restart — update the TradingView alert's webhook URL when you
relaunch. For a stable URL, use a named Cloudflare tunnel or ngrok with an
account.)

TradingView side (paid plan required for webhooks; note TV's own "Paper
Trading" broker is manual-only — Pine strategies can't place orders into it,
which is why this webhook bridge exists):
1. Open a 5-min NSE chart → add the Pine strategy → set its "Webhook secret"
   input to the printed secret.
2. Create ONE alert: condition = the strategy, "Order fills only",
   message `{{strategy.order.alert_message}}`,
   webhook URL = the printed `https://....trycloudflare.com/webhook`.
3. The Strategy Tester panel on the chart is TradingView's simulation record;
   your ledger/dashboard is the authoritative paper record (with charges,
   slippage, and agent verdicts).

Payloads with a wrong/missing secret are rejected (403) and logged to the
rejections table. Alerts fire on bar close, so fidelity ≈ the Python engine.

## Data retention — only the last 5 sessions

The dashboard is a working screen, not an archive. `trading/retention.py` keeps
the last **5 trading sessions** (`RETENTION_DAYS` in `trading/config.py`) of
trades, rejections, agent logs, journal reports and cached candles; everything
older is deleted and the SQLite file is vacuumed so the space comes back.

**Pruning is irreversible.** It runs automatically when the dashboard starts,
and prints exactly what it removed. Preview or override it:

```bash
.venv/bin/python -m trading.retention --dry-run     # show what would go
.venv/bin/python -m trading.retention --keep-days 20
.venv/bin/python -m trading.dashboard --no-prune    # leave history alone
.venv/bin/python -m trading.test_retention          # 18 correctness checks
```

Two details that are easy to get wrong and are therefore tested: the window
counts **sessions, not calendar days** (five calendar days would silently eat a
session across a long weekend), and rejections and agent-log rows are cut at
the timestamp of the oldest session kept, so a row is never orphaned from the
trades it explains.

## F&O opening-window long scanner — `trading/fno/`

A separate, self-contained scanner for the 09:15–10:00 IST opening window. It
ranks NSE F&O stocks for **intraday long setups** using price structure, VWAP,
EMAs, volume, futures open interest, market context and sector strength, and
emits an entry/stop/target only when several of those agree. It is decision
support: it holds no broker connection and places no orders.

```bash
.venv/bin/python -m trading.fno --allow-delayed          # one scan
.venv/bin/python -m trading.fno --watch --allow-delayed  # re-scan every 2 min to 10:00
.venv/bin/python -m trading.fno --symbols VEDL,SBIN --notify --allow-delayed
.venv/bin/python -m trading.fno --replay data/fno/sample-2026-08-20.json
.venv/bin/python -m trading.fno --allow-delayed --json   # machine-readable
.venv/bin/python -m trading.fno --relaxed --allow-delayed  # show near-misses too
.venv/bin/python -m trading.dashboard                    # web UI at :8787/fno
.venv/bin/python -m trading.test_fno                     # the scanner's checks
```

`--relaxed` is an opt-in widening of what is *displayed*, not a strategy change.
On a quiet tape nearly every name is vetoed on flat open interest, so the board
reads empty; with it on, an OI-flat read is filed as a soft blocker and the name
shows as **WATCH** instead of AVOID. It can only ever downgrade — a relaxed name
keeps its rejection, and `tradeable` requires an empty rejection list, so nothing
relaxed can report **BUY**. The score floor and every other gate are unchanged,
and the page carries a `RELAXED GRADING` banner so a loosened board can never be
mistaken for a strict one.

### What it does per scan

1. **Data-integrity gate first.** If the candle feed is exchange-delayed or
   missing, it prints `REAL-TIME DATA UNAVAILABLE — SCAN NOT RELIABLE.` and
   grades nothing. `--allow-delayed` proceeds but stamps the caveat on every
   candidate. Every printed value carries its source, instrument, timeframe and
   age.
2. **Market context** — NIFTY 50 and BANK NIFTY move, NIFTY-50 breadth, and the
   live NSE sector indices → one of STRONGLY BULLISH … STRONGLY BEARISH. A
   strongly bearish tape multiplies every long score by 0.55 *and* vetoes the
   trade.
3. **Liquidity screen** — the F&O gainers board, filtered on traded value,
   volume and a sane move size.
4. **Per-stock workup on 5-minute bars** — 09:15–09:30 opening range, VWAP and
   its slope, 20/50 EMA (seeded from prior sessions, not from six bars),
   ATR, previous-day levels, pivots, relative volume vs the same clock time on
   prior days.
5. **Breakout logic** — a *close* above the range high by a buffer, on expanded
   volume, still holding, ideally with a retest that held. A wick through the
   level is not a breakout; an extended candle is a chase and is rejected.
6. **Futures OI** — long buildup / short covering / short buildup / long
   unwinding, from `latestOI` vs `prevOI`. Long buildup scores highest; short
   covering is allowed but discounted; the other two are vetoed.
7. **Score out of 100** on the documented weights (market 15, sector 10, trend
   10, VWAP 10, EMA 10, breakout 15, volume 10, OI 15, retest 5), then the veto
   filters run — so a 91-point setup with contradictory OI is shown *rejected
   with its score*, not silently dropped.
8. **Structural levels** — stop below the retest low / breakout bar / swing low
   (never a round percentage), targets at 1:2 and 1:3, and rejection if
   overhead resistance leaves no room for 1:2.

If nothing clears the bar it prints `NO HIGH-QUALITY LONG SETUP` plus the
closest misses and why each failed. That is the expected output most mornings.

### The call, on the main dashboard

`http://localhost:8787/` now opens with a **Buy now** panel — the scanner's
current answer, before the P&L tiles:

```
Buy now  — 09:45–10:00 BREAKOUT + RETEST · updated 9s ago

RELIANCE  ₹101.50  +1.20%   ● BUY · A+ · 99.5/100   ENERGY · RELIANCE28AUG2026FUT
  ENTRY ZONE        STOP LOSS        TARGET 1     TARGET 2     RISK
  ₹101.05 – 101.20  ₹101.02          ₹101.56      ₹101.74      ₹0.18
  buy inside band   below retest low 1:2          1:3          0.18% of price
Why: closed above the 09:15–09:30 resistance at 101.00 and is holding · 3.2x volume
[watch: HDFCBANK 95.5]  open the scanner →
```

Most mornings it will instead read **NO HIGH-QUALITY LONG SETUP** with the
number of names on watch — that is the correct output, not a failure. If the
feed is unreliable it shows the data banner and no levels at all, the same gate
the scanner and the CLI apply. The panel derives nothing of its own: it renders
`picks[0]` from the same payload, so the dashboard, the page and the terminal
cannot disagree about an entry.

### The screen you actually check — `/fno`

```bash
.venv/bin/python -m trading.dashboard        # http://localhost:8787/fno
.venv/bin/python -m trading.fno.web          # or standalone on 8788
.venv/bin/python -m trading.fno.web --replay data/fno/demo-mixed.json   # offline demo
```

The page answers one question at the top — **how many long setups are buyable
right now** — and then shows each name as a card you can act on or dismiss in a
few seconds:

- **A verdict, in a word.** `BUY` (clears every rule), `WATCH` (a real setup
  that has not matured — extended, no retest yet, still under the score line),
  `AVOID` (the evidence argues against it: contradictory OI, no volume behind
  the breakout, below VWAP, illiquid). The word and a glyph carry the verdict —
  never the colour alone — so it survives a colour-blind reader, a grey-scale
  print, and a glance.
- **A price-vs-levels bar** per candidate: the breakout level, VWAP, the entry
  band, the stop and both targets on one axis, with the current price as a
  ringed dot. Stop and targets differ by *shape* (square cap vs triangle) as
  well as hue, because red/green is exactly the pair a colour-blind reader
  loses and confusing a stop for a target is the most expensive mistake this
  page could cause. Every value is also printed beneath the bar.
- **The plan as five figures** — entry zone, stop (with the structure it came
  from), T1, T2, risk in rupees and percent.
- **Why, and what is blocking it** side by side. A rejected name shows its
  score *and* its blockers rather than disappearing, so you can see that
  TATASTEEL scored 87 and was still refused for a 0.7x-volume breakout.
- **A full table** underneath with every screened name, and **data notes** with
  the source and age of every feed. Scans run in a background thread, so the
  page never blocks; `Scan now` forces a refresh and the age is always shown.

It is laid out as a **terminal console**: a sticky command bar (window, live
status LED, verdict filters with counts, scan button), a KPI strip, a candidate
column beside a rail of sector strength / feed health / window notes, and a
dense table at the bottom. Dark is the designed default — a desk runs dark —
with light as a fully worked alternate (`?theme=light|dark|system`, or the
button).

Both palettes were run through the data-viz palette validator against their
real card surfaces rather than eyeballed. Two results worth knowing: every step
clears 4.5:1 as text (these colours label target and stop *prices*, not just
marks), and the stop/target pair is separated by lightness as well as hue
(deutan ΔE 11.7 dark, 6.2 light with shape relief) because red-vs-green is the
pair colour-blind readers lose. The score meter is a single hue — magnitude is
one hue, and the grade word already states quality.

### Call options — `trading/fno/options.py`

When a stock setup qualifies, the scanner also shows how it could be expressed
in a **call option**, and the number it leads with is **breakeven = strike +
premium**. That is the point most option buyers miss: the breakeven routinely
sits *above* the scanner's own 1:2 target on the stock, so the chart can play
out perfectly and the call still expires worthless. A strike is offered only if

  * the stock reaching its 1:3 target puts the option in profit,
  * the contract is liquid enough to leave (open interest and volume floors),
  * and the premium is not an unreachable share of spot.

Otherwise it prints **NO CALL** with the arithmetic, e.g. from a live run:

```
KAYNES  ₹3908.00 +1.77%   stock T1 3927.28  T2 3960.20
  NO CALL — the nearest strike (4000 CE at 1.00) breaks even at 4001.00 —
            above the 1:3 target 3960.20, so the stock setup can work and
            the call still lose
```

Same-day expiry, out-of-the-money strikes and "pays only at 1:3" all raise
explicit warnings rather than being buried.

### Option targets — what the call is worth at the stock's target

`trading/fno/pricing.py` answers the other half of the question: the scanner
gives a target on the *stock*, and this gives the premium there. It solves the
implied volatility that reproduces the option's current traded price, then
reprices at the stock's stop and both targets, ageing the option by the assumed
hold. That yields the option's **own** risk/reward — which is usually worse
than the stock's, and the report says so:

```
Option (call):  100 CE @ 1.62  (24-Sep-2026, ITM)
  Breakeven:    101.62 — ABOVE target 1 (101.56); only the 1:3 target pays
  Option plan:  pay 1.62 → 1.67 at T1, 1.83 at T2, 1.24 at the stock's stop
  Option R:R:   0.14:1 on the option (the stock trade is 1:2)
  Model:        Black-Scholes at 5% IV, IV assumed unchanged, ~2h hold
  ! the option's own reward-to-risk is 0.1:1 — worse than the stock trade it
    expresses, because premium decays while the stock does not
```

A 1:2 stock setup became a 0.14:1 option trade. That is the number worth seeing
before buying a call, and it is why the projection exists.

**It is a model, not a quote.** IV is assumed unchanged (an IV crush pays less,
a panic bid more), decay is charged only for the assumed hold, and the fill
depends on an order book the free feed does not publish. A premium that no sane
volatility explains — below intrinsic, usually a stale print — produces no
projection at all rather than a fabricated one.

### Real-time data via Fyers — `trading/fno/fyers.py`

The free candle feed is exchange-delayed ~15 minutes. If you have the
TradeBrahma Fyers server running (`npm run server` in that project) and a
same-day broker login, the scanner can take **real-time** 5-minute candles from
it instead:

```bash
.venv/bin/python -m trading.fno --fyers            # CLI
.venv/bin/python -m trading.fno.web 8790 --fyers   # the web UI
```

It goes through that project's Express server rather than calling Fyers
directly, so **no credential is ever read by this project** — that server owns
the app secret and the OAuth session. Open interest, the movers board, sector
indices and the option board still come from NSE; Fyers supplies only the thing
NSE will not serve in real time, which is intraday OHLCV.

Access tokens last about a day, so reconnecting is a **daily step**: TradeBrahma
→ Broker Settings → Fyers API v3 → Connect. The feed distinguishes "server not
running" from "token expired" because the fix differs, and falls back to the
delayed feed with a visible note rather than pretending.

**The hard limit:** NSE's option-chain endpoints (`option-chain-equities`,
`option-chain-v3`) return an empty object to programmatic clients — a soft
block. The only option data available is the **20 most-active stock option
contracts** market-wide, which rotates through the day. So a name gets an
option view only when its contracts are in that board (typically 1–2 of a
12-name shortlist, and it changes hour to hour); everything else says *no live
option data* rather than inventing a strike. No bid/ask and no implied
volatility are published there, so no greeks are computed and none are shown —
every number on screen is a field NSE returned. A broker feed (Kite/Fyers)
would give the full chain and fix this.

### NIFTY 50 index options — `trading/fno/index_options.py`

The **full NIFTY chain is available** — 664 calls and 664 puts, 133 strikes,
six expiries, live. The endpoint name is the trap: `index=nifty_opt` answers
HTTP 500, `index=nse50_opt` serves the chain.

NIFTY gets its own module because **an index publishes no volume**. VWAP,
relative volume and breakout-volume confirmation do not exist for it, so three
of the stock scanner's nine checks are simply unavailable. What remains is
price structure (opening range, EMA stack) plus breadth and sector
participation — which for the index *is* the instrument rather than context.
Every NIFTY read states that it is a five-factor call, not an eight-factor one.

Two rules matter more here than for stocks:

* **front expiry only.** A back-month contract can model a better reward-to-risk
  while being untradeable — on a deep chain the far months carry a fraction of
  the open interest. A live run picked a 27-Oct strike with 1,012 OI over a
  1-Sep one with 106,858 until this rule was added.
* **exit on the move, not at expiry.** Breakeven (strike + premium) is the right
  test only if you hold to expiry; intraday the call gains from delta well
  before spot reaches it. So breakeven is reported as information and the
  *gate* is the modelled premium at the index target versus at the stop — the
  option's own reward-to-risk.

Only calls are produced, matching the scanner's long-only scope; a NIFTY
breakdown would be a put trade and is not inferred.

```bash
.venv/bin/python -m trading.test_options    # 25 correctness checks
```

### Data sources and their honest limits

| What | Source | Quality |
|---|---|---|
| Futures OI + change | NSE `live-analysis-oi-spurts-underlyings` | live, all F&O names |
| F&O movers, volume, value | NSE `live-analysis-variations` (FOSec) | live, top 20 each way |
| Futures turnover | NSE `live-analysis-most-active-underlying` | live |
| Contract id / futures LTP | NSE `liveEquity-derivatives` | live, 20 most active only |
| Indices, sectors, breadth | NSE `allIndices` | live |
| 5-minute OHLCV | yfinance | **exchange-delayed ~15 min** |

Consequences, stated rather than hidden: candles are delayed, so the scanner
refuses to grade without `--allow-delayed`; NSE no longer serves
`quote-derivative` publicly, so **bid/ask spreads are unavailable and the
spread filter is skipped**; and for names outside the 20 most-active contracts
the futures %change is unpublished, so the OI read pairs futures OI with the
underlying's cash move (flagged per candidate). Wiring a broker feed (Kite
historical + quote) fixes all three — implement `universe`, `futures_board`,
`candles`, `index_candles` and optionally `market_snapshot` on a new feed class
in `trading/fno/data.py`; nothing downstream reads a provider directly.

### Layout

| File | Role |
|---|---|
| `config.py` | every threshold, weight, sector map — tune here, not in code |
| `models.py` | typed carriers; each value travels with its `Provenance` |
| `nse.py` | cookie-bootstrapped NSE public client, tolerant field parsing |
| `data.py` | `LiveFeed` (NSE + yfinance) and `ReplayFeed` (saved JSON bundle) |
| `indicators.py` | VWAP, EMA, ATR, opening range, RVOL, pivots — pure functions |
| `context.py` | market classification and sector strength |
| `oi.py` | the four OI quadrants and their scores |
| `setup.py` | breakout/retest detection, structural entry, stop and targets |
| `scoring.py` | the 0–100 weighted score |
| `filters.py` | the hard veto rules |
| `scanner.py` | time windows and orchestration |
| `pricing.py` | Black-Scholes implied vol + projected premium at the stock's levels |
| `fyers.py` | real-time candles through the TradeBrahma Fyers server (optional) |
| `options.py` | call-option selection: breakeven test, liquidity floors, expiry warnings |
| `report.py` | the candidate block, the 🚨 alert, the no-trade line, and `as_dict` — the one payload both `--json` and the web UI render |
| `web.py` | the `/fno` page: background scan cache, verdict cards, price-vs-levels bars |

**It does not guarantee profit, and a high score is not a probability of
profit** — it is the number of independent factors that currently agree.
Verify every level on your own terminal before acting on it.

## Backtesting — `trading/backtest.py`

`--replay` writes real paper trades into the ledger, so it can only test one
config per day and it pollutes the record it is measuring. The backtest harness
runs the **same** `prepare`/`detect`/`stop`/`target` functions from
`strategy.py` over cached candles with **no ledger, no LLM, no side effects**,
so any number of configs can be compared in seconds.

```bash
.venv/bin/python -m trading.backtest                     # current config
.venv/bin/python -m trading.backtest --strategy pivot -v # per-session + per-symbol detail
.venv/bin/python -m trading.backtest --compare           # all strategies x 5m/15m
.venv/bin/python -m trading.backtest --slippage 0.0001   # slippage sensitivity
.venv/bin/python -m trading.backtest --charges 0 --slippage 0   # frictionless upper bound
.venv/bin/python -m trading.backtest --download always   # extend history (after 15:30 IST)
```

**The headline number is expectancy ± a 95% confidence interval.** A config
whose interval straddles zero has *not* been shown to have an edge no matter
how good its total looks. `--compare` labels each row `EDGE` / `negative` /
`not proven` on that basis.

Use it as a library for sweeps — `params()` rebinds strategy globals so you can
vary EMA lengths or RR without editing `config.py`:

```python
from trading.backtest import Backtester, build_strategy, load_history, metrics, params
frames = load_history(["RELIANCE", "INFY"], "15m")
with params(FAST_EMA=5, SLOW_EMA=20, RR_TARGET=1.5):
    m = metrics(Backtester(build_strategy("ema")).run(frames))
print(m["expectancy"], m["lo"], m["hi"])
```

Two things to know:

* **Candle cache.** yfinance only serves ~60 days of intraday history. Every
  run merges the newest bars into `data/candles/`, so the usable window grows
  past that ceiling the longer you keep running it.
* **It will not download while a live session is running.** yfinance shares one
  rate limit and one sqlite tz-cache per machine; a sweep that refetches per
  config throttles the running engine's feed. `--download auto` (the default)
  detects `trading.strategy` and falls back to cache. Extend history after
  15:30 IST with `--download always`.

`trading/test_backtest.py` checks the harness against synthetic candles whose
answer is known by construction — no look-ahead on the entry bar, stop wins
level ties, sizing caps bind, square-off is enforced, slippage is always
adverse, and a zero-mean sample is never reported as an edge.

## Transaction costs — `trading/costs.py`

Charges are **not** a flat percentage. Zerodha brokerage is
`min(0.03% of turnover, ₹20)` per order, so round-trip cost as a share of
position value falls as size rises:

| Position | Charges | As % | Breakeven move |
|---|---|---|---|
| ₹7,800 | ₹8.27 | 0.106% | 1.38 pts |
| ₹24,700 | ₹26.19 | 0.106% | 1.38 pts |
| ₹1,99,773 | ₹117.61 | 0.059% | 0.78 pts |
| ₹5,20,000 | ₹230.48 | 0.044% | 0.58 pts |

The old flat `CHARGES_PCT_ROUND_TRIP` model overstated costs by **~2x at ₹2L
positions** (and ~13% at ₹25k), so pre-2026-08 ledger rows carry inflated
charges. All ledger-writing paths now use `costs.round_trip()`.
`breakeven_move()` is the number to design entries against: if the average
winner is smaller than it, no amount of entry tuning makes the strategy pay.
**Verify the rate card against Zerodha's brokerage calculator before trusting
a live projection.**

## Data-blackout watchdog

yfinance failures do **not** raise — a blacked-out fetch returns an empty dict,
so the engine keeps logging `scanned 0/6` and looks alive while flying blind. On
2026-08-10 it did that for ~40 minutes, and a blind engine cannot check stops or
targets; worse, `squareoff()` then closes at the *entry* price and books a fake
~0 P&L. A crash-only watchdog would never have fired.

So the engine counts consecutive empty scans (`MAX_BLIND_SCANS = 5`, i.e. five
minutes at a 60s poll) and exits **75** (`EXIT_DATA_BLACKOUT`, EX_TEMPFAIL).
`cron_day.sh` treats 75 as "restart me", up to `MAX_RESTARTS = 6` and never past
15:30. Square-off during a blackout now logs an explicit `WARNING` that the exit
prices are stale.

`raise_fd_limit()` runs at engine startup and lifts the fd soft limit from
macOS's default 256 to 8192. This matters for **cron-launched** runs specifically
— an interactive shell already inherits a high limit, which is why the bug only
ever bit the automated session.

**Known gap:** cron does not fire while the Mac is asleep. No session ran on
2026-08-11 for that reason. Use `caffeinate`, a `launchd` agent with
`StartCalendarInterval` (which catches up after wake), or just check the log.

## Evaluating the supervisor (after 4–6 weeks of paper trading)

The supervisor's verdicts are advisory. Before ever promoting it to a blocking
veto, check whether its vetoes actually predicted losers:

```sql
SELECT agent_verdict, COUNT(*) n, ROUND(AVG(pnl),2) avg_pnl, ROUND(SUM(pnl),2) total_pnl
FROM trades WHERE status='closed' GROUP BY agent_verdict;
```

If `veto` trades are meaningfully worse than `approve` trades, promote it — with
a hard timeout (e.g. 500 ms → default allow) so the LLM can never stall the
trade path.

## Measured status as of 2026-08-10 — no strategy here has a proven edge

Backtested over 58 sessions (2026-05-19 → 08-10), 6-symbol watchlist, itemised
Zerodha charges, `SLIPPAGE_PCT = 0.0005`:

| Strategy | Trades | Net | PF | Expectancy ± 95% CI | Verdict |
|---|---|---|---|---|---|
| `ema` 9/21 15m | 208 | −4,255 | 1.17 | −20 ± 222 | not proven |
| `ema_pivot` (pivot filter) | 180 | −8,282 | 1.11 | −46 ± 246 | not proven |
| `pivot` (9 EMA + pivots) | 217 | −43,856 | 0.80 | −202 ± 122 | **negative** |
| `avwap` scalp | 445 | −134,813 | 0.72 | −303 ± 130 | **negative** |
| `revert` (VWAP 2σ) | 16 | +2,425 | 2.57 | +152 ± 353 | not proven (n too small) |

Three findings that matter more than the totals:

1. **Slippage, not brokerage, decides profitability.** The raw EMA signal is
   worth ~+288/trade frictionless. Correct charges cost ~116/trade; the *assumed*
   0.05% slippage costs ~193/trade. Breakeven is around `SLIPPAGE_PCT ≈ 0.00045`
   — so whether this system makes money hinges on a parameter that has never
   been measured against real fills. Measuring it is the top go-live blocker.
2. **A 64-config sweep produced zero significant results** (best t-stat 0.84).
   The best config would need ~1,130 trades (~316 sessions) to reach p<0.05 at
   its own effect size. Per-trade variance (sd ≈ ₹1,635) dwarfs the edge, so
   picking the top row of a sweep is selection bias, not a finding.
3. **`avwap` and `pivot` are significantly negative** — those are real results,
   not noise. Don't run them. `pivot` is the "9 EMA + pivot points" setup from
   social media; it loses before costs (gross −18,367, PF 0.80).

## Go-live checklist (do NOT flip `mode="live"` before all of these)

0. **Establish positive expectancy first.** As of 2026-08-10 no config clears
   this bar — see the table above. Nothing below matters until one does.
1. Backtest validates the edge; **4–6+ weeks of paper trading** across trending
   and choppy conditions validates execution and the slippage model.
2. Compare simulated fills vs. real spreads; tighten `SLIPPAGE_PCT` if paper
   results look too good.
3. Static IP registered with Zerodha (SEBI mandate for API order placement;
   one IP change per calendar week).
4. Kite Connect subscription (₹500/mo) + automated daily OAuth token refresh.
5. Confirm with Zerodha how your sub-10-OPS personal strategy gets Algo-ID
   tagging (the rate limiter in the router keeps you under the threshold).
6. Keep all agents advisory-only for at least the first month live.
7. Start with minimum quantity for 2 weeks regardless of paper results.
8. Verify current STT/charges rates and update `CHARGES_PCT_ROUND_TRIP`.

## Audit trail

Every LLM call (prompt + response + errors) is stored in the `agent_log` table:

```bash
sqlite3 data/ledger.db "SELECT datetime(ts,'unixepoch','localtime'), agent, ok, COALESCE(error,'') FROM agent_log ORDER BY ts DESC LIMIT 20;"
```
