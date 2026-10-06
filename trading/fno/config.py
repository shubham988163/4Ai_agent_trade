"""Scanner thresholds, scoring weights, universe and sector mapping.

Every number here is a tunable. They are collected in one place so a change of
policy ("only take 1:2.5 or better") is a config edit, not a code change.
"""
from __future__ import annotations

# --- time windows (IST) -------------------------------------------------
MARKET_OPEN = "09:15"
OPENING_RANGE_END = "09:30"      # first-15-minute high/low window ends here
SCANNER_WINDOW_END = "10:00"     # after this, setups are labelled POST-10:00
NO_SIGNAL_BEFORE = "09:20"       # 09:15-09:20 is observation only

BAR_INTERVAL = "5m"
BAR_MINUTES = 5

# --- data integrity -----------------------------------------------------
# A quote older than this is not "live" for a 5-minute intraday scan.
MAX_DATA_AGE_MIN = 6.0

# --- liquidity screen ---------------------------------------------------
MIN_FUT_VALUE_CR = 25.0          # futures turnover so far today, INR crore
MIN_CASH_VALUE_CR = 15.0         # cash-segment traded value so far today, INR crore
MIN_CASH_VOLUME = 100_000        # today's cash-segment shares traded
MAX_SPREAD_BPS = 15.0            # futures bid/ask spread, basis points
MIN_ABS_MOVE_PCT = 0.35          # needs to actually be moving to be a candidate
MAX_ABS_MOVE_PCT = 7.0           # beyond this it is news-driven; treat as suspect
SHORTLIST_SIZE = 12              # how many names get the full per-symbol workup

# --- structure / setup --------------------------------------------------
ATR_PERIOD = 14
BREAKOUT_BUFFER_ATR = 0.05       # close must clear resistance by this much ATR
RETEST_TOLERANCE_ATR = 0.30      # how deep a pullback still counts as a retest
HOLD_TOLERANCE_ATR = 0.25        # close below R by more than this = lost the level
EXTENDED_ATR = 1.50              # price this far above R = chasing, reject
EXTENDED_VWAP_PCT = 2.0          # price this % above VWAP = extended, reject
VWAP_SLOPE_BARS = 3              # bars used to measure VWAP slope
BREAKOUT_VOL_MULT = 1.5          # breakout bar volume vs prior-bar average
WEAK_VOL_MULT = 1.0              # below this the breakout is volume-less
MIN_RVOL = 1.2                   # today's cumulative volume vs same-time average

# --- risk ---------------------------------------------------------------
SL_BUFFER_ATR = 0.15             # stop sits this far below the structural level
ENTRY_FLOOR_ATR = 0.05           # the entry zone never starts closer than this
                                 # above the stop, so no fill inside the zone
                                 # can be below its own stop
MAX_RISK_PCT = 1.20              # per-share risk above this % of price = no trade
MIN_RISK_PCT = 0.15              # stop this tight is noise, not structure
MIN_RR = 2.0                     # reject anything that cannot make 1:2
TARGET_RR_1 = 2.0
TARGET_RR_2 = 3.0
MIN_RR_TO_RESISTANCE = 1.8       # overhead resistance must leave at least this
                                 # much reward-to-risk of clear air above entry
OVERHEAD_MIN_ATR = 0.50          # the day high only counts as resistance once
                                 # price has pulled this far back below it

# --- open interest ------------------------------------------------------
OI_FLAT_PCT = 0.5                # |OI change| below this = no meaningful signal
OI_STRONG_PCT = 3.0              # OI change at/above this = strong buildup

# --- scoring weights (must total 100; see §11 of the spec) --------------
WEIGHTS = {
    "market": 15,
    "sector": 10,
    "trend": 10,
    "vwap": 10,
    "ema": 10,
    "breakout": 15,
    "volume": 10,
    "oi": 15,
    "retest": 5,
}

GRADES = [(90, "A+"), (80, "A"), (70, "B"), (60, "Weak")]
MIN_TRADEABLE_SCORE = 70         # below this the scanner will not emit a trade
ALERT_SCORE = 80                 # A/A+ setups fire the 🚨 alert block

# --- relaxed (opt-in) grading, for seeing the near-misses ------------------
# Strict stays the default: the spec and the README are explicit that an empty
# board is the correct output most mornings, and these gates are what make it
# so. `--relaxed` stops treating an OI-flat reading as a contradiction, so a
# quiet tape — where nearly every name is vetoed on flat OI — shows what came
# closest instead of nothing.
#
# It can only ever downgrade, never promote. A relaxed name keeps its rejection
# text and `tradeable` requires an empty rejection list, so nothing relaxed can
# report BUY. The score floor is deliberately NOT lowered: `"score"` is already
# a soft blocker, so strict mode already surfaces 60–69 as WATCH, and moving the
# floor to 55 would have let a 55–69 name through with no rejection at all —
# i.e. a real BUY, not a wider display. A read that actively argues against the
# long (short buildup, long unwinding, unavailable) stays vetoed outright.
RELAXED_ALLOW_OI_FLAT = True
RELAXED = False                  # flip via set_relaxed(); do not assign directly


def set_relaxed(on: bool = True) -> None:
    """Turn relaxed grading on or off.

    Every module reads the flag as `C.RELAXED` at call time, so setting the
    module attribute is enough — nothing needs threading through Scanner,
    ScanCache or the web layer.
    """
    global RELAXED
    RELAXED = bool(on)


EMA_FAST = 20
EMA_SLOW = 50

# --- market context ------------------------------------------------------
# NIFTY % change thresholds for the five-way classification.
MARKET_BANDS = [
    (0.60, "STRONGLY BULLISH"),
    (0.20, "BULLISH"),
    (-0.20, "NEUTRAL"),
    (-0.60, "BEARISH"),
]
MARKET_SCORE = {
    "STRONGLY BULLISH": 1.00,
    "BULLISH": 0.80,
    "NEUTRAL": 0.45,
    "BEARISH": 0.15,
    "STRONGLY BEARISH": 0.00,
}
# Longs are heavily penalised, not merely down-weighted, against the market.
STRONGLY_BEARISH_PENALTY = 0.55  # multiplies the whole score

NIFTY_TICKER = "^NSEI"
BANKNIFTY_TICKER = "^NSEBANK"

# --- sectors -------------------------------------------------------------
# Sector strength comes from the live NSE sector index where one maps cleanly
# (names verified against /api/allIndices), and falls back to the median move
# of that sector's F&O peers when it does not. Which of the two was used is
# stated in the report. Symbols absent from SECTORS score neutral and are
# flagged rather than guessed at.
SECTOR_INDEX: dict[str, str] = {
    "BANK": "NIFTY BANK",
    "FIN": "NIFTY FINANCIAL SERVICES",
    "IT": "NIFTY IT",
    "AUTO": "NIFTY AUTO",
    "METAL": "NIFTY METAL",
    "PHARMA": "NIFTY PHARMA",
    "FMCG": "NIFTY FMCG",
    "ENERGY": "NIFTY ENERGY",
    "POWER": "NIFTY ENERGY",          # NTPC/POWERGRID/TATAPOWER sit in this index
    "REALTY": "NIFTY REALTY",
    "MEDIA": "NIFTY MEDIA",
    "CEMENT": "NIFTY CEMENT",
    "CHEMICAL": "NIFTY CHEMICALS",
    "INFRA": "NIFTY INFRASTRUCTURE",
    "CONSUMER": "NIFTY INDIA CONSUMPTION",
    "DEFENCE": "NIFTY INDIA DEFENCE",
    "INTERNET": "NIFTY INDIA INTERNET",
    "TRAVEL": "NIFTY INDIA TOURISM",
    "CAPGOODS": "NIFTY INDIA MANUFACTURING",
}

SECTORS: dict[str, str] = {
    # Banks & financials
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK", "AXISBANK": "BANK",
    "KOTAKBANK": "BANK", "INDUSINDBK": "BANK", "BANKBARODA": "BANK",
    "PNB": "BANK", "CANBK": "BANK", "FEDERALBNK": "BANK", "IDFCFIRSTB": "BANK",
    "AUBANK": "BANK", "BANDHANBNK": "BANK",
    "BAJFINANCE": "FIN", "BAJAJFINSV": "FIN", "SHRIRAMFIN": "FIN",
    "CHOLAFIN": "FIN", "MUTHOOTFIN": "FIN", "LICHSGFIN": "FIN",
    "SBILIFE": "FIN", "HDFCLIFE": "FIN", "ICICIGI": "FIN", "ICICIPRULI": "FIN",
    "PFC": "FIN", "RECLTD": "FIN", "IRFC": "FIN", "HDFCAMC": "FIN",
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT", "MPHASIS": "IT",
    # Auto
    "MARUTI": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO", "EICHERMOT": "AUTO",
    "HEROMOTOCO": "AUTO", "TVSMOTOR": "AUTO", "ASHOKLEY": "AUTO",
    "BHARATFORG": "AUTO", "MOTHERSON": "AUTO", "BALKRISIND": "AUTO",
    "MRF": "AUTO", "EXIDEIND": "AUTO",
    # Metals
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    "VEDL": "METAL", "SAIL": "METAL", "JINDALSTEL": "METAL",
    "NATIONALUM": "METAL", "HINDCOPPER": "METAL", "APLAPOLLO": "METAL",
    # Pharma
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA",
    "DIVISLAB": "PHARMA", "AUROPHARMA": "PHARMA", "LUPIN": "PHARMA",
    "ALKEM": "PHARMA", "TORNTPHARM": "PHARMA", "ZYDUSLIFE": "PHARMA",
    "APOLLOHOSP": "PHARMA", "LAURUSLABS": "PHARMA", "BIOCON": "PHARMA",
    # Energy & oil
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY", "IOC": "ENERGY",
    "HINDPETRO": "ENERGY", "GAIL": "ENERGY", "OIL": "ENERGY",
    "NTPC": "POWER", "POWERGRID": "POWER", "TATAPOWER": "POWER",
    "ADANIGREEN": "POWER", "ADANIENSOL": "POWER", "NHPC": "POWER",
    "COALINDIA": "ENERGY",
    # FMCG & consumer
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "DABUR": "FMCG",
    "GODREJCP": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG", "UBL": "FMCG",
    "TITAN": "CONSUMER", "TRENT": "CONSUMER", "DMART": "CONSUMER",
    "JUBLFOOD": "CONSUMER", "PAGEIND": "CONSUMER", "VBL": "CONSUMER",
    # Infra, cement, realty
    "LT": "INFRA", "ULTRACEMCO": "CEMENT", "GRASIM": "CEMENT",
    "SHREECEM": "CEMENT", "AMBUJACEM": "CEMENT", "ACC": "CEMENT",
    "DLF": "REALTY", "GODREJPROP": "REALTY", "OBEROIRLTY": "REALTY",
    "LODHA": "REALTY", "PRESTIGE": "REALTY",
    # Telecom, media, misc
    "BHARTIARTL": "TELECOM", "IDEA": "TELECOM", "INDUSTOWER": "TELECOM",
    "ZEEL": "MEDIA", "PVRINOX": "MEDIA",
    "ADANIENT": "DIVERSIFIED", "ADANIPORTS": "INFRA", "SIEMENS": "CAPGOODS",
    "ABB": "CAPGOODS", "BEL": "DEFENCE", "HAL": "DEFENCE", "BDL": "DEFENCE",
    "CUMMINSIND": "CAPGOODS", "POLYCAB": "CAPGOODS", "HAVELLS": "CAPGOODS",
    "ASIANPAINT": "CONSUMER", "PIDILITIND": "CHEMICAL", "SRF": "CHEMICAL",
    "UPL": "CHEMICAL", "TATACHEM": "CHEMICAL", "DEEPAKNTR": "CHEMICAL",
    "IRCTC": "TRAVEL", "INDIGO": "TRAVEL", "CONCOR": "INFRA",
    "NAUKRI": "INTERNET", "ZOMATO": "INTERNET", "PAYTM": "INTERNET",
    "NYKAA": "INTERNET", "POLICYBZR": "INTERNET",
}


def grade_for(score: float) -> str:
    for cutoff, label in GRADES:
        if score >= cutoff:
            return label
    return "Avoid"
