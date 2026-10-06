"""Typed carriers for scanner data.

Every market value the scanner touches travels inside something that knows
where it came from and when — §18 of the spec ("timestamp, data source,
instrument, timeframe"). A bare float is never passed around, because a bare
float cannot be checked for staleness and cannot be attributed in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class Provenance:
    """Where a number came from, and how old it is."""
    source: str            # "nse-public", "yfinance", "replay:<file>"
    instrument: str        # "RELIANCE", "RELIANCE28AUG2026FUT", "^NSEI"
    timeframe: str         # "5m", "1d", "snapshot"
    data_time: datetime    # exchange timestamp of the newest datum (IST)
    fetched_at: datetime   # when we pulled it (IST)
    delayed: bool = False  # feed is known to be exchange-delayed

    def age_min(self, now: datetime | None = None) -> float:
        now = now or datetime.now(IST)
        return (now - self.data_time).total_seconds() / 60.0

    def label(self, now: datetime | None = None) -> str:
        tag = "DELAYED" if self.delayed else "live"
        return (f"{self.source}/{tag} {self.instrument} {self.timeframe} "
                f"@ {self.data_time:%H:%M:%S} ({self.age_min(now):.1f}m old)")


@dataclass
class Candles:
    """Intraday OHLCV, IST-indexed, columns Open/High/Low/Close/Volume."""
    df: object             # pandas.DataFrame — untyped to keep import cost out
    prov: Provenance

    def __len__(self) -> int:
        return len(self.df)


@dataclass
class FuturesSnapshot:
    """Current-month stock-futures state for one underlying."""
    symbol: str
    contract: str
    expiry: str | None
    last_price: float
    prev_close: float | None
    pct_change: float | None
    open_interest: float | None          # contracts
    oi_change_pct: float | None          # vs previous day's close OI
    volume: float | None                 # contracts traded today
    turnover_cr: float | None            # INR crore
    bid: float | None
    ask: float | None
    prov: Provenance

    @property
    def spread_bps(self) -> float | None:
        if not self.bid or not self.ask or self.ask <= 0 or self.bid <= 0:
            return None
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid * 10_000


@dataclass
class Levels:
    """Structure read off today's 5-minute bars."""
    price: float
    or_high: float
    or_low: float
    vwap: float
    vwap_slope: float
    ema_fast: float | None
    ema_slow: float | None
    ema_fast_slope: float | None
    ema_slow_slope: float | None
    atr: float
    prev_high: float | None
    prev_low: float | None
    prev_close: float | None
    day_high: float
    day_low: float
    support: float | None
    resistance: float          # the level the setup is built on
    overhead: float | None     # next resistance above the entry
    rvol: float | None
    last_bar_vol: float
    avg_bar_vol: float
    pct_change: float | None


@dataclass
class Structure:
    """Qualitative price-action read."""
    higher_highs: bool
    higher_lows: bool
    consolidating: bool
    breakout: bool
    breakout_time: datetime | None
    breakout_vol_mult: float | None
    holding: bool
    retested: bool
    retest_low: float | None
    failed_breakout: bool
    rejection: bool
    extended: bool
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed_breakout:
            return "FAILED BREAKOUT"
        if self.breakout and self.retested and self.holding:
            return "BREAKOUT → RETEST → HOLD"
        if self.breakout and self.holding:
            return "BREAKOUT (no retest yet)"
        if self.breakout:
            return "BREAKOUT LOST"
        if self.rejection:
            return "REJECTED AT RESISTANCE"
        if self.consolidating:
            return "CONSOLIDATION BELOW RESISTANCE"
        return "NO BREAKOUT"


@dataclass
class OIRead:
    classification: str    # LONG BUILDUP / SHORT COVERING / ...
    bullish: bool
    reliable: bool
    detail: str
    oi_change_pct: float | None
    price_change_pct: float | None


@dataclass
class Trade:
    entry_low: float
    entry_high: float
    entry: float
    stop: float
    stop_basis: str
    target1: float
    target2: float
    risk: float
    rr1: float
    rr2: float


@dataclass
class Candidate:
    symbol: str
    sector: str
    fut: FuturesSnapshot
    levels: Levels
    structure: Structure
    oi: OIRead
    rel_strength: float          # stock %chg minus NIFTY %chg
    sector_strength: float | None
    scores: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    grade: str = "Avoid"
    trade: Trade | None = None
    rejections: list[str] = field(default_factory=list)
    # Category of each rejection ("oi", "timing", "market", …). Kept beside the
    # texts so a UI can tell "not set up yet" from "actively contradicted"
    # without pattern-matching prose.
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: list[Provenance] = field(default_factory=list)
    # Today's closed 5-minute bars as {t, c, w, v} — close and VWAP per bar, so
    # the UI can plot the session path rather than only the final levels.
    spark: list[dict] = field(default_factory=list)
    # How this long could be expressed in calls, or why it cannot be.
    option_plan: object | None = None

    @property
    def tradeable(self) -> bool:
        return not self.rejections and self.trade is not None

    # Blockers that describe a setup which has not matured yet, as opposed to
    # one the evidence argues against. Only these can leave a stock on watch.
    # "oi_flat" is how relaxed mode files an OI-flat read: no positional
    # conviction either way, which is an absence of confirmation rather than a
    # contradiction. "oi" (short buildup, long unwinding) stays hard either way.
    SOFT_BLOCKERS = ("timing", "structure", "risk", "score", "window", "oi_flat")

    @property
    def verdict(self) -> str:
        """BUY / WATCH / AVOID — the one word the scanner exists to produce."""
        if self.tradeable:
            return "BUY"
        if self.score >= 60 and all(b in self.SOFT_BLOCKERS for b in self.blockers):
            return "WATCH"
        return "AVOID"

    @property
    def blocker_summary(self) -> str:
        return self.rejections[0] if self.rejections else ""


@dataclass
class MarketContext:
    classification: str
    nifty_pct: float | None
    banknifty_pct: float | None
    nifty_above_vwap: bool | None
    breadth_pct: float | None      # % of scanned universe positive
    advances: int | None
    declines: int | None
    notes: list[str] = field(default_factory=list)
    sources: list[Provenance] = field(default_factory=list)
    sector_pct: dict[str, float] = field(default_factory=dict)
    # How the sector and breadth numbers were obtained — printed with them, so
    # an index reading and a peer proxy are never mistaken for each other.
    sector_source: str = "unavailable"
    breadth_source: str = "unavailable"


@dataclass
class ScanResult:
    when: datetime
    window: str
    market: MarketContext
    candidates: list[Candidate]
    considered: int
    data_ok: bool
    data_notes: list[str] = field(default_factory=list)
    # Whether this window may emit signals at all, and why not. "No setup" and
    # "too early to say" are different answers and must not render as the same
    # sentence.
    signals_allowed: bool = True
    window_note: str = ""
    # NIFTY 50 index-option read — a separate instrument from the stock
    # candidates, so it travels separately.
    index_plan: object | None = None
    banknifty_plan: object | None = None
    # Whole-chain positioning (PCR, max pain, OI walls) and yesterday's pivot
    # levels — context for every candidate rather than a per-stock reading.
    chain_read: object | None = None
    pivots: object | None = None
    gap: object | None = None

    @property
    def picks(self) -> list[Candidate]:
        """The top 3 tradeable candidates (§16) — often empty, by design."""
        return [c for c in self.candidates if c.tradeable][:3]
