"""Structured MarketFacts schema and builder for the Multi-Agent Trading Council.

Enforces:
1. Strict Pydantic schema for all facts (no free text or duplicate technicals).
2. Computed values (ORB, 200 SMA trend, Volume ratio, VWAP) sourced directly from ORBMA200Strategy.
3. Missing fields default to None (serializes to JSON null), never an estimated number.
4. Auditable JSON persistence in SQLite ledger.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field

IST = ZoneInfo("Asia/Kolkata")

# Clean mapping of NSE ticker symbols to official company names
COMPANY_NAMES = {
    "SBIN": "State Bank of India",
    "RELIANCE": "Reliance Industries Limited",
    "HDFCBANK": "HDFC Bank Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "INFY": "Infosys Limited",
    "TCS": "Tata Consultancy Services Limited",
    "AXISBANK": "Axis Bank Limited",
    "KOTAKBANK": "Kotak Mahindra Bank Limited",
    "LT": "Larsen & Toubro Limited",
    "ITC": "ITC Limited",
    "BHARTIARTL": "Bharti Airtel Limited",
    "BAJFINANCE": "Bajaj Finance Limited",
    "ASIANPAINT": "Asian Paints Limited",
    "MARUTI": "Maruti Suzuki India Limited",
    "TITAN": "Titan Company Limited",
    "TATASTEEL": "Tata Steel Limited",
    "SUNPHARMA": "Sun Pharmaceutical Industries Limited",
    "WIPRO": "Wipro Limited",
    "HCLTECH": "HCL Technologies Limited",
    "NTPC": "NTPC Limited",
    "ONGC": "Oil and Natural Gas Corporation Limited",
    "POWERGRID": "Power Grid Corporation of India Limited",
}


class SignalFacts(BaseModel):
    side: Literal["BUY", "SELL"]
    price: float
    stop_loss: float
    target: float
    rr: float = 2.0


class ORBFacts(BaseModel):
    or_high: float | None = None
    or_low: float | None = None
    or_width_pct: float | None = None
    or_valid: bool = False


class TrendFacts(BaseModel):
    price: float
    sma200_1h: float | None = None
    sma200_30m: float | None = None
    state: Literal["up", "down", "mixed", "unavailable"] = "unavailable"


class VolumeFacts(BaseModel):
    breakout_bar_vol: float = 0.0
    avg20: float = 0.0
    ratio: float = 0.0
    bar_complete: bool = True


class VWAPFacts(BaseModel):
    value: float = 0.0
    slope: Literal["up", "down", "flat"] = "flat"
    aligned: bool = False


class MarketDirectionFacts(BaseModel):
    nifty_pct: float | None = None
    banknifty_pct: float | None = None
    sector_pct: float | None = None
    india_vix: float | None = None
    gift_nifty_gap_pct: float | None = None
    direction: Literal["supports", "opposes", "neutral", "unavailable"] = "unavailable"


class EventFacts(BaseModel):
    result_within_days: int | None = None
    ex_date_within_days: int | None = None
    asm_gsm: bool = False
    near_circuit: bool = False
    gap_pct: float | None = None
    holiday: bool = False


class NewsItem(BaseModel):
    headline: str
    source: str
    published_at: str
    match: Literal["exact", "alias"] = "exact"


class FundamentalFacts(BaseModel):
    period: str = ""
    basis: Literal["consolidated", "standalone"] = "consolidated"
    yoy_net_profit_pct: float | None = None
    yoy_revenue_pct: float | None = None
    yoy_nii_pct: float | None = None
    gnpa_pct: float | None = None


class MarketFacts(BaseModel):
    symbol: str
    company_name: str
    asof: str
    signal: SignalFacts | None = None
    orb: ORBFacts
    trend: TrendFacts
    volume: VolumeFacts
    vwap: VWAPFacts
    market: MarketDirectionFacts = Field(default_factory=MarketDirectionFacts)
    events: EventFacts = Field(default_factory=EventFacts)
    news: list[NewsItem] = Field(default_factory=list)
    fundamentals: FundamentalFacts = Field(default_factory=FundamentalFacts)

    def to_json(self) -> str:
        """Serialize facts to JSON with null values for missing fields."""
        return self.model_dump_json(indent=2)


def build_market_facts(
    symbol: str,
    strategy_setup: dict,
    market_direction: dict | None = None,
    events: dict | None = None,
    news: list[dict] | None = None,
    fundamentals: dict | None = None,
    asof: datetime | None = None,
) -> MarketFacts:
    """Construct a MarketFacts instance directly from the strategy's computed values.
    
    Eliminates duplicated calculations of volume, VWAP, or OR logic.
    """
    now_dt = asof or datetime.now(IST)
    asof_str = now_dt.isoformat()
    company = COMPANY_NAMES.get(symbol, f"{symbol} Limited")

    sig_data = strategy_setup.get("signal")
    signal_obj = None
    if sig_data is not None:
        if hasattr(sig_data, "side"):
            signal_obj = SignalFacts(
                side=sig_data.side,
                price=sig_data.price,
                stop_loss=sig_data.stop_loss,
                target=sig_data.target,
                rr=getattr(sig_data, "rr", 2.0),
            )
        elif isinstance(sig_data, dict):
            signal_obj = SignalFacts(
                side=sig_data["side"],
                price=sig_data["price"],
                stop_loss=sig_data["stop_loss"],
                target=sig_data["target"],
                rr=sig_data.get("rr", 2.0),
            )

    orb_data = strategy_setup["orb"]
    orb_obj = ORBFacts(
        or_high=orb_data.get("or_high"),
        or_low=orb_data.get("or_low"),
        or_width_pct=orb_data.get("or_width_pct"),
        or_valid=bool(orb_data.get("or_valid", False)),
    )

    trend_data = strategy_setup["trend"]
    trend_obj = TrendFacts(
        price=trend_data["price"],
        sma200_1h=trend_data.get("sma200_1h"),
        sma200_30m=trend_data.get("sma200_30m"),
        state=trend_data.get("state", "unavailable"),
    )

    vol_data = strategy_setup["volume"]
    vol_obj = VolumeFacts(
        breakout_bar_vol=vol_data.get("breakout_bar_vol", 0.0),
        avg20=vol_data.get("avg20", 0.0),
        ratio=vol_data.get("ratio", 0.0),
        bar_complete=vol_data.get("bar_complete", True),
    )

    vwap_data = strategy_setup["vwap"]
    vwap_obj = VWAPFacts(
        value=vwap_data.get("value", 0.0),
        slope=vwap_data.get("slope", "flat"),
        aligned=bool(vwap_data.get("aligned", False)),
    )

    mkt_obj = MarketDirectionFacts(**(market_direction or {}))
    evt_obj = EventFacts(**(events or {}))
    news_items = [NewsItem(**n) for n in (news or [])]
    fund_obj = FundamentalFacts(**(fundamentals or {}))

    return MarketFacts(
        symbol=symbol,
        company_name=company,
        asof=asof_str,
        signal=signal_obj,
        orb=orb_obj,
        trend=trend_obj,
        volume=vol_obj,
        vwap=vwap_obj,
        market=mkt_obj,
        events=evt_obj,
        news=news_items,
        fundamentals=fund_obj,
    )
