"""Phase 3: Real Indian news & fundamentals with curated aliases and strict YoY accounting.

Rules:
1. Curated alias table for 6 symbols: SBIN, RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS.
2. Negative keyword filters to eliminate false associations (e.g. SBI Holdings Japan / Ripple,
   Reliance Power / Anil Ambani, HDFC Life, ICICI Prudential, TCS as tax acronym).
3. Whitelist of approved Indian financial publishers (ET, Mint, BS, Moneycontrol, FE, CNBC-TV18, NSE/BSE).
4. Strict YoY calculation for quarterly results: current quarter vs same quarter prior year (Q_t vs Q_{t-4}).
   Return null if YoY unavailable; NEVER fall back to sequential QoQ.
5. Bank-specific metrics (Net Interest Income, GNPA) and consolidated basis flag.
6. Lazy fetch: executed only for candidates passing quant strategy + cost-floor gates.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

from trading.market_facts import MarketFacts, NewsItem, FundamentalFacts, build_market_facts

IST = ZoneInfo("Asia/Kolkata")

# 1. Curated Alias Table for 6-Symbol Watchlist
WATCHLIST_METADATA: dict[str, dict[str, Any]] = {
    "SBIN": {
        "company_name": "State Bank of India",
        "aliases": ["State Bank of India", "SBI", "State Bank", "SBIN"],
        "negative_keywords": [
            "sbi holdings",
            "sbi securities japan",
            "ripple",
            "xrp",
            "state bank of pakistan",
            "sbi cards",
            "sbi life",
            "sbi mutual fund",
            "sbi general insurance",
        ],
        "is_bank": True,
    },
    "RELIANCE": {
        "company_name": "Reliance Industries Limited",
        "aliases": ["Reliance Industries", "RIL", "Reliance", "Mukesh Ambani"],
        "negative_keywords": [
            "reliance power",
            "rpower",
            "reliance infrastructure",
            "rinfra",
            "reliance capital",
            "rcap",
            "anil ambani",
            "reliance naval",
            "reliance communications",
            "rcom",
        ],
        "is_bank": False,
    },
    "HDFCBANK": {
        "company_name": "HDFC Bank Limited",
        "aliases": ["HDFC Bank", "HDFCBank"],
        "negative_keywords": [
            "hdfc life",
            "hdfc amc",
            "hdfc standard life",
            "hdfc mutual fund",
            "hdfc securities",
            "hdfc ergo",
        ],
        "is_bank": True,
    },
    "ICICIBANK": {
        "company_name": "ICICI Bank Limited",
        "aliases": ["ICICI Bank", "ICICIBank"],
        "negative_keywords": [
            "icici prudential",
            "icici pru",
            "icici lombard",
            "icici securities",
            "icici direct",
            "icici mutual fund",
            "icici venture",
        ],
        "is_bank": True,
    },
    "INFY": {
        "company_name": "Infosys Limited",
        "aliases": ["Infosys", "INFY", "Infosys Ltd"],
        "negative_keywords": [
            "infosys bpm",
            "infosys science foundation",
        ],
        "is_bank": False,
    },
    "TCS": {
        "company_name": "Tata Consultancy Services Limited",
        "aliases": ["Tata Consultancy Services", "TCS", "Tata Consultancy"],
        "negative_keywords": [
            "tax collected at source",
            "tax collection at source",
            "tcs rate",
            "tcs on remittance",
            "tcs on foreign",
            "tcs provisions",
            "cbdt",
            "income tax act",
            "section 206c",
        ],
        "is_bank": False,
    },
}

# 2. Approved Indian Financial Publishers Whitelist
APPROVED_PUBLISHERS = [
    "economic times",
    "the economic times",
    "et now",
    "livemint",
    "mint",
    "business standard",
    "moneycontrol",
    "financial express",
    "cnbc-tv18",
    "cnbctv18",
    "cnbc tv18",
    "reuters",
    "bloomberg",
    "nse",
    "bse",
    "sebi",
]


def is_approved_source(source_str: str) -> bool:
    """Validate publisher against curated whitelist."""
    if not source_str:
        return False
    src_lower = source_str.strip().lower()
    return any(appr in src_lower for appr in APPROVED_PUBLISHERS)


def filter_curated_news(
    raw_news: list[dict],
    symbol: str,
    max_items: int = 5,
) -> tuple[list[dict], str]:
    """Filter news items against curated aliases, negative keywords, and publisher whitelist."""
    meta = WATCHLIST_METADATA.get(symbol)
    if not meta:
        return [], "No news available (untracked symbol)"

    aliases = meta["aliases"]
    negative_kws = meta["negative_keywords"]
    filtered_items: list[dict] = []
    formatted_bullets: list[str] = []
    audit_log: list[dict[str, str]] = []

    import logging
    logger = logging.getLogger("indian_market_data")

    for item in raw_news:
        title = item.get("title") or (item.get("content", {}) or {}).get("title")
        if not title:
            continue
        title_str = str(title).strip()
        title_lower = title_str.lower()

        # Check publisher
        pub = (
            item.get("publisher")
            or ((item.get("content", {}) or {}).get("provider", {}) or {}).get("displayName")
            or item.get("source")
            or ""
        )
        pub_str = str(pub).strip()
        if not is_approved_source(pub_str):
            entry = {
                "headline": title_str,
                "source": pub_str,
                "status": "REJECTED",
                "rule": f"unapproved_source: '{pub_str}' not in publisher whitelist",
            }
            audit_log.append(entry)
            logger.info("[NEWS_FILTER] %s", entry)
            continue

        # Check negative keywords first
        matched_neg = next((neg for neg in negative_kws if neg in title_lower), None)
        if matched_neg:
            entry = {
                "headline": title_str,
                "source": pub_str,
                "status": "REJECTED",
                "rule": f"negative_keyword_match: '{matched_neg}'",
            }
            audit_log.append(entry)
            logger.info("[NEWS_FILTER] %s", entry)
            continue

        # Check alias match
        alias_matched = None
        exact_sym_matched = False
        for alias in aliases:
            # Word boundary regex to avoid partial substring false matches
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, title_lower):
                alias_matched = alias
                if alias.upper() == symbol:
                    exact_sym_matched = True
                break

        if not alias_matched:
            entry = {
                "headline": title_str,
                "source": pub_str,
                "status": "REJECTED",
                "rule": f"no_alias_match: no curated alias of {symbol} found in text",
            }
            audit_log.append(entry)
            logger.info("[NEWS_FILTER] %s", entry)
            continue

        pub_time = item.get("providerPublishTime") or item.get("published_at")
        if isinstance(pub_time, (int, float)):
            dt_str = datetime.fromtimestamp(pub_time, tz=IST).strftime("%Y-%m-%d %H:%M")
        elif isinstance(pub_time, str):
            dt_str = pub_time
        else:
            dt_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

        match_type = "exact" if exact_sym_matched else "alias"
        news_entry = {
            "headline": title_str,
            "source": pub_str,
            "published_at": dt_str,
            "match": match_type,
        }
        filtered_items.append(news_entry)
        formatted_bullets.append(f"• {title_str} ({pub_str})")

        accept_entry = {
            "headline": title_str,
            "source": pub_str,
            "status": "ACCEPTED",
            "rule": f"alias_match: '{alias_matched}' (type={match_type})",
        }
        audit_log.append(accept_entry)
        logger.info("[NEWS_FILTER] %s", accept_entry)

        if len(filtered_items) >= max_items:
            break

    summary = (
        "\n".join(formatted_bullets[:3])
        if formatted_bullets
        else "No verified Indian financial headlines for this session."
    )
    return filtered_items, summary, audit_log


def compute_strict_yoy_fundamentals(
    symbol: str,
    quarterly_income_stmt: pd.DataFrame | None,
) -> tuple[dict[str, Any], str]:
    """Compute YoY quarterly fundamentals strictly comparing Q_t vs Q_{t-4}.
    
    NEVER falls back to sequential QoQ (Q_t vs Q_{t-1}).
    Returns null for all growth figures if YoY comparative quarter is absent.
    """
    meta = WATCHLIST_METADATA.get(symbol, {})
    is_bank = meta.get("is_bank", False)

    fund_dict: dict[str, Any] = {
        "period": "",
        "basis": "consolidated",
        "yoy_net_profit_pct": None,
        "yoy_revenue_pct": None,
        "yoy_nii_pct": None,
        "gnpa_pct": None,
    }

    if (
        quarterly_income_stmt is None
        or not isinstance(quarterly_income_stmt, pd.DataFrame)
        or quarterly_income_stmt.empty
    ):
        return fund_dict, "Quarterly financials unavailable (null)"

    cols = list(quarterly_income_stmt.columns)
    if len(cols) < 5:
        # Fewer than 5 quarters means Q_{t-4} cannot be verified
        period_str = cols[0].strftime("%b %Y") if hasattr(cols[0], "strftime") else str(cols[0])
        fund_dict["period"] = period_str
        return (
            fund_dict,
            f"Insufficient history ({len(cols)} quarters) for strict YoY comparison (null). Never fallback to QoQ.",
        )

    # Latest quarter
    col_t = cols[0]
    period_str = col_t.strftime("%b %Y") if hasattr(col_t, "strftime") else str(col_t)
    fund_dict["period"] = period_str

    # Find the matching same quarter from previous year (approx 365 days prior or 4 quarters back)
    col_yoy = None
    if hasattr(col_t, "year") and hasattr(col_t, "month"):
        target_year = col_t.year - 1
        for col in cols[1:]:
            if hasattr(col, "year") and hasattr(col, "month"):
                if col.year == target_year and abs(col.month - col_t.month) <= 1:
                    col_yoy = col
                    break
    if col_yoy is None and len(cols) >= 5:
        # Fall back to index 4 (4 quarters prior) if date math missed due to fiscal alignment
        col_yoy = cols[4]

    if col_yoy is None:
        return (
            fund_dict,
            f"Matching YoY quarter for {period_str} not found in statement (null). QoQ fallback prohibited.",
        )

    yoy_period_str = col_yoy.strftime("%b %Y") if hasattr(col_yoy, "strftime") else str(col_yoy)

    # 1. Net Profit / Net Income
    net_inc_rows = [
        k for k in quarterly_income_stmt.index
        if "Net Income Common Stockholders" in str(k) or "Net Income" in str(k)
    ]
    if net_inc_rows:
        row_key = net_inc_rows[0]
        v_t = quarterly_income_stmt.loc[row_key, col_t]
        v_prior = quarterly_income_stmt.loc[row_key, col_yoy]
        if pd.notna(v_t) and pd.notna(v_prior) and v_prior != 0:
            growth = float((v_t - v_prior) / abs(v_prior) * 100)
            fund_dict["yoy_net_profit_pct"] = round(growth, 2)

    # 2. Revenue (Non-banks)
    if not is_bank:
        rev_rows = [
            k for k in quarterly_income_stmt.index
            if "Total Revenue" in str(k) or "Operating Revenue" in str(k)
        ]
        if rev_rows:
            row_key = rev_rows[0]
            v_t = quarterly_income_stmt.loc[row_key, col_t]
            v_prior = quarterly_income_stmt.loc[row_key, col_yoy]
            if pd.notna(v_t) and pd.notna(v_prior) and v_prior != 0:
                growth = float((v_t - v_prior) / abs(v_prior) * 100)
                fund_dict["yoy_revenue_pct"] = round(growth, 2)

    # 3. Bank-Specific: Net Interest Income (NII)
    if is_bank:
        nii_rows = [
            k for k in quarterly_income_stmt.index
            if "Net Interest Income" in str(k) or "Interest Income" in str(k)
        ]
        if nii_rows:
            row_key = nii_rows[0]
            v_t = quarterly_income_stmt.loc[row_key, col_t]
            v_prior = quarterly_income_stmt.loc[row_key, col_yoy]
            if pd.notna(v_t) and pd.notna(v_prior) and v_prior != 0:
                growth = float((v_t - v_prior) / abs(v_prior) * 100)
                fund_dict["yoy_nii_pct"] = round(growth, 2)

    # Formulate human summary
    summary_parts = [f"Period: {period_str} vs {yoy_period_str} (YoY strict)"]
    if fund_dict["yoy_net_profit_pct"] is not None:
        summary_parts.append(f"PAT YoY: {fund_dict['yoy_net_profit_pct']:+.1f}%")
    if is_bank and fund_dict["yoy_nii_pct"] is not None:
        summary_parts.append(f"NII YoY: {fund_dict['yoy_nii_pct']:+.1f}%")
    if not is_bank and fund_dict["yoy_revenue_pct"] is not None:
        summary_parts.append(f"Rev YoY: {fund_dict['yoy_revenue_pct']:+.1f}%")

    return fund_dict, " | ".join(summary_parts)


def lazy_fetch_indian_context(
    symbol: str,
    ticker: yf.Ticker | None = None,
) -> tuple[list[dict], str, dict[str, Any], str]:
    """Lazy fetcher: executed ONLY when a symbol generates a gated candidate.
    
    Fetches official Indian news and strict YoY fundamentals, never called during initial bar scanning.
    """
    from trading.config import YF_SUFFIX
    if ticker is None:
        yf_sym = symbol + YF_SUFFIX if not symbol.endswith(YF_SUFFIX) else symbol
        ticker = yf.Ticker(yf_sym)

    # News fetch & curated filter
    raw_news = getattr(ticker, "news", []) or []
    filtered_news, news_summary, _ = filter_curated_news(raw_news, symbol)

    # Fundamentals fetch & strict YoY
    try:
        stmt = getattr(ticker, "quarterly_income_stmt", None)
    except Exception:
        stmt = None

    fund_dict, fund_summary = compute_strict_yoy_fundamentals(symbol, stmt)
    return filtered_news, news_summary, fund_dict, fund_summary
