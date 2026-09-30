"""Central configuration for the trading system.

All paths are anchored to the project root so cron jobs work regardless of CWD.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# --- Trading universe ---
# Core watchlist (used by the pre-market agent's prompt focus).
WATCHLIST = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN"]

# Full scan universe for the strategy engine — Nifty 50 constituents.
# Index composition changes ~semi-annually; edit as needed.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    # TATAMOTORS removed — ticker changed after the 2025 CV/PV demerger; add
    # the successor ticker(s) here if you want it back.
    "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# What the strategy engine scans. The full-Nifty-50 sweep (Jul 9–10) lost
# money in choppy conditions; the 6-stock core watchlist (Jul 7 setup) is the
# only configuration that has been net-profitable so far.
SCAN_UNIVERSE = WATCHLIST
YF_SUFFIX = ".NS"

# --- data retention ---
# The dashboard is a working screen, not an archive. Keep the last N *sessions*
# (not calendar days — a long weekend would otherwise eat one) of trades,
# rejections, agent logs, journal reports and cached candles.
# Pruning is destructive: raise this, or pass --keep-days, before it runs if you
# want a longer history. See trading/retention.py.
RETENTION_DAYS = 5

DB_PATH = PROJECT_ROOT / "data" / "ledger.db"
TODAY_CONFIG_PATH = PROJECT_ROOT / "data" / "today_config.json"
FYERS_TOKEN_PATH = PROJECT_ROOT / "data" / "fyers_token.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# --- Fyers API v3 (Real-time live market feed) ---
FYERS_APP_ID = os.getenv("FYERS_APP_ID", "J8ZMHWBTBW-100")
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID", "")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "http://localhost:3001/api/fyers/callback")

# --- Strategy: EMA crossover ---
FAST_EMA = 9
SLOW_EMA = 21
CANDLE_INTERVAL = "15m"
SWING_LOOKBACK = 10
RR_TARGET = 2.0
# Account sizing: capital set to 15,000 INR.
ACCOUNT_CAPITAL = 15_000.0
RISK_PER_TRADE = 75.0           # INR risked per trade (0.5% of capital)
POLL_SECONDS = 60
SQUAREOFF_TIME = "15:15"
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# --- Strategy: AVWAP scalp ---
AVWAP_RR = 1.5
AVWAP_ATR_MULT = 0.5
AVWAP_RSI_LEN = 14
AVWAP_EMA_LEN = 20
AVWAP_VOL_MULT = 1.2

# --- Risk kernel (hard limits — the AI agent can NEVER override these) ---
# Scaled to ACCOUNT_CAPITAL = 15,000 INR:
DAILY_LOSS_LIMIT = -500.0       # INR; hard stop for the day (~3.3% of capital)
DAILY_PROFIT_TARGET = 1_000.0   # INR; stop opening new trades once hit for the day
MAX_POSITION_VALUE = 3_000.0    # INR per position (20% of capital)
MAX_OPEN_POSITIONS = 5          # portfolio cap (max 100% of capital deployed)
MAX_ORDERS_PER_SEC = 8

# --- Fill simulation (paper mode) ---
SLIPPAGE_PCT = 0.0005           # 0.05% assumed slippage
# Rough intraday cost model per round trip (brokerage + STT + charges).
# Verify against Zerodha's brokerage calculator — STT rates changed in 2026.
CHARGES_PCT_ROUND_TRIP = 0.0006
MAX_COST_RISK_RATIO = 0.30      # Max friction as a fraction of risk (30%)

# --- Exit management: deterministic trailing stop ---
TRAIL_ENABLED = True             # Enabled: trailing stop moves to breakeven & trails to protect profit
BREAKEVEN_AT_R = 1.0            # favourable excursion (in R) that moves the stop to breakeven
OPTION_BREAKEVEN_AT_R = 0.4     # for options: arm breakeven at 0.4R (since option stops are wide)
OPTION_BREAKEVEN_PCT = 0.08     # for options: arm breakeven at +8% gain
TRAIL_ATR_MULT = 2.0            # chandelier trail distance in ATR(14) units
PARTIAL_AT_R = 1.5              # R multiple that triggers partial booking (None disables)
PARTIAL_PCT = 0.5               # fraction of the position booked at the partial
AUTO_EXIT_ON_STOP = True        # Automatically close open positions when trailing stop or target is hit

# --- AI agent ---
# Provider is auto-selected in trading/agents/llm.py: Gemini when
# GEMINI_API_KEY is set, Anthropic otherwise (override with LLM_PROVIDER).
ANTHROPIC_MODEL = "claude-opus-4-8"
# Google Gemini: gemini-3.5-flash-lite is fast, reliable and cost-effective on Google AI Studio
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-3.6-flash"
AGENT_MAX_TOKENS = 16000
SUPERVISOR_POLL_SECONDS = 5     # how often the async supervisor checks for new trades

# Multi-Agent stock evaluator parameters
AGENT_MIN_CONVICTION = 7        # Conviction score out of 10 required to execute a trade
AGENT_MAX_TRADES_PER_SCAN = 3   # Max new trades to execute in a single scan pass

# --- AI agent: the F&O long scanner as a second candidate source ---
# The scanner (trading/fno/) and the agent were two disjoint pipelines: the
# scanner printed BUY/WATCH cards that nothing ever traded. When enabled, the
# agent reads the scanner's verdicts AND its plan (entry band, structural stop,
# 1:2 / 1:3 targets) and runs them through the same council and risk kernel.
# Opt-in and OFF by default, like TRAIL_ENABLED: nothing changes until flipped.
AGENT_SCANNER_ENABLED = False
# Deliberately NOT an "orb_ma200*" id, so ExecutionRouter.risk_check's 200MA
# trend gate does not apply to it -- the same pattern ema_9_21 and avwap_scalp
# already use. The agent re-applies that gate itself before routing; see
# scanner_preflight() in agents/agent_trader.py. Do not retire the agent-side
# check on the assumption the router is covering it.
AGENT_SCANNER_STRATEGY_ID = "fno_scanner"
AGENT_SCANNER_TIERS = ("BUY", "WATCH")   # ("BUY",) would be the BUY picks only
AGENT_SCANNER_MAX = 3                    # names taken from one scan per pass
# Enter only while price is inside the scanner's own entry band. A WATCH name
# is usually sitting above its band already -- that is part of why it is only
# on watch -- so this flag decides whether the agent chases them or skips them.
AGENT_SCANNER_REQUIRE_ENTRY_BAND = True


# Safe defaults used whenever the pre-market agent fails or returns invalid output
FALLBACK_DAY_CONFIG = {
    "regime": "choppy",
    "risk_multiplier": 0.5,
    "blocked_symbols": [],
    "rationale": "fallback: pre-market agent unavailable — trading at half size",
}

# --- Options Trading Settings ---
INDEX_LOT_SIZES: dict[str, int] = {
    "NIFTY": 75,
    "NIFTY 50": 75,
    "NSE:NIFTY50-INDEX": 75,
    "BANKNIFTY": 30,
    "BANK NIFTY": 30,
    "NSE:NIFTYBANK-INDEX": 30,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 120,
}

STOCK_LOT_SIZES: dict[str, int] = {
    "RELIANCE": 250,
    "HDFCBANK": 550,
    "ICICIBANK": 700,
    "INFY": 400,
    "TCS": 175,
    "SBIN": 750,
    "BHARTIARTL": 475,
    "ITC": 1600,
    "LT": 175,
    "AXISBANK": 625,
    "KOTAKBANK": 400,
    "TATAMOTORS": 550,
    "TATASTEEL": 5500,
    "BAJFINANCE": 125,
    "MARUTI": 50,
}

DEFAULT_STOCK_OPTION_LOT = 250
MAX_OPTION_POSITION_VALUE = 75000.0
MAX_OPTION_RISK_PER_TRADE = 2500.0

