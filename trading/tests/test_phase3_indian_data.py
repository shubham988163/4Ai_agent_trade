"""Unit tests for Phase 3: Indian market data, curated aliases, and strict YoY fundamentals."""
from __future__ import annotations

import pandas as pd
import pytest

from trading.indian_market_data import (
    WATCHLIST_METADATA,
    filter_curated_news,
    is_approved_source,
    compute_strict_yoy_fundamentals,
)


def test_publisher_whitelist():
    """Verify official Indian financial publishers pass and unapproved sources are rejected."""
    assert is_approved_source("The Economic Times") is True
    assert is_approved_source("Mint") is True
    assert is_approved_source("Livemint") is True
    assert is_approved_source("Business Standard") is True
    assert is_approved_source("Moneycontrol") is True
    assert is_approved_source("CNBC-TV18") is True
    assert is_approved_source("NSE Corporate Disclosures") is True

    # Unapproved sources
    assert is_approved_source("CryptoDailyNews") is False
    assert is_approved_source("Reddit /r/IndianStreetBets") is False
    assert is_approved_source("RandomStockBlogspot") is False
    assert is_approved_source("") is False


def test_alias_and_negative_keyword_filtering():
    """Verify curated alias matching and negative keyword rejection."""
    # 1. SBIN
    raw_sbin_news = [
        {
            "title": "State Bank of India Reports 12% Rise in Q1 Net Profit",
            "source": "The Economic Times",
            "published_at": "2026-09-28 09:30",
        },
        {
            "title": "SBI Holdings Japan Partners with Ripple for XRP Cross-Border Remittances",
            "source": "Livemint",
            "published_at": "2026-09-28 09:30",
        },
        {
            "title": "State Bank of Pakistan hikes benchmark interest rate",
            "source": "Business Standard",
            "published_at": "2026-09-28 09:30",
        },
        {
            "title": "SBI Cards shares tumble after RBI unsecured lending circular",
            "source": "Moneycontrol",
            "published_at": "2026-09-28 09:30",
        },
    ]
    filtered_sbin, summary, audit = filter_curated_news(raw_sbin_news, "SBIN")
    assert len(filtered_sbin) == 1
    assert "State Bank of India Reports" in filtered_sbin[0]["headline"]

    # Verify every headline logged with accepted/rejected status and exact rule
    assert len(audit) == 4
    assert audit[0]["status"] == "ACCEPTED"
    assert "alias_match" in audit[0]["rule"]
    assert audit[1]["status"] == "REJECTED"
    assert "negative_keyword_match" in audit[1]["rule"]
    assert audit[2]["status"] == "REJECTED"
    assert "negative_keyword_match" in audit[2]["rule"]
    assert audit[3]["status"] == "REJECTED"
    assert "negative_keyword_match" in audit[3]["rule"]

    # 2. RELIANCE vs Reliance Power / Infrastructure
    raw_ril_news = [
        {
            "title": "Reliance Industries signs multi-billion dollar green energy pact",
            "source": "Economic Times",
            "published_at": "2026-09-28 10:00",
        },
        {
            "title": "Reliance Power shares hit 5% lower circuit amid debt probe",
            "source": "Mint",
            "published_at": "2026-09-28 10:00",
        },
        {
            "title": "Anil Ambani-led Reliance Infrastructure faces arbitration setback",
            "source": "Business Standard",
            "published_at": "2026-09-28 10:00",
        },
    ]
    filtered_ril, _, _ = filter_curated_news(raw_ril_news, "RELIANCE")
    assert len(filtered_ril) == 1
    assert "Reliance Industries" in filtered_ril[0]["headline"]

    # 3. TCS vs Tax Collected at Source (tax acronym false positive)
    raw_tcs_news = [
        {
            "title": "Tata Consultancy Services bags $1B digital transformation deal in UK",
            "source": "Moneycontrol",
            "published_at": "2026-09-28 10:30",
        },
        {
            "title": "CBDT issues clarification on TCS rate of 20% on foreign remittances under LRS",
            "source": "Economic Times",
            "published_at": "2026-09-28 10:30",
        },
    ]
    filtered_tcs, _, _ = filter_curated_news(raw_tcs_news, "TCS")
    assert len(filtered_tcs) == 1
    assert "Tata Consultancy Services" in filtered_tcs[0]["headline"]

    # 4. HDFCBANK vs HDFC Life / AMC
    raw_hdfc_news = [
        {
            "title": "HDFC Bank opens 100 new branches across rural India",
            "source": "Livemint",
            "published_at": "2026-09-28 11:00",
        },
        {
            "title": "HDFC Life announces special bonus for policyholders",
            "source": "Business Standard",
            "published_at": "2026-09-28 11:00",
        },
    ]
    filtered_hdfc, _, _ = filter_curated_news(raw_hdfc_news, "HDFCBANK")
    assert len(filtered_hdfc) == 1
    assert "HDFC Bank" in filtered_hdfc[0]["headline"]


def test_strict_yoy_quarterly_fundamentals_calculation():
    """Verify strict YoY calculation comparing Q_t vs Q_{t-4}."""
    dates = pd.date_range("2026-06-30", periods=5, freq="-1QE")
    # dates: [2026-06-30 (Q1 FY27), 2026-03-31, 2025-12-31, 2025-09-30, 2025-06-30 (Q1 FY26)]
    data = {
        dates[0]: [12000.0, 60000.0],  # Q_t
        dates[1]: [11500.0, 58000.0],  # Q_{t-1} (QoQ - must be ignored)
        dates[2]: [11000.0, 56000.0],
        dates[3]: [10500.0, 54000.0],
        dates[4]: [10000.0, 50000.0],  # Q_{t-4} (YoY comparative quarter)
    }
    stmt = pd.DataFrame(data, index=["Net Income", "Total Revenue"])

    fund, summary = compute_strict_yoy_fundamentals("TCS", stmt)

    # Net Income: (12000 - 10000) / 10000 * 100 = +20.0%
    assert fund["yoy_net_profit_pct"] == 20.0
    # Revenue: (60000 - 50000) / 50000 * 100 = +20.0%
    assert fund["yoy_revenue_pct"] == 20.0
    assert fund["basis"] == "consolidated"
    assert "YoY strict" in summary


def test_strict_yoy_never_falls_back_to_qoq():
    """If YoY (t-4) comparative quarter is missing, return null (never QoQ)."""
    dates = pd.date_range("2026-06-30", periods=3, freq="-1QE")  # Only 3 quarters available!
    data = {
        dates[0]: [12000.0, 60000.0],
        dates[1]: [11500.0, 58000.0],
        dates[2]: [11000.0, 56000.0],
    }
    stmt = pd.DataFrame(data, index=["Net Income", "Total Revenue"])

    fund, summary = compute_strict_yoy_fundamentals("INFY", stmt)

    # Must be None (null), never fall back to (12000 - 11500) / 11500 = +4.35%
    assert fund["yoy_net_profit_pct"] is None
    assert fund["yoy_revenue_pct"] is None
    assert "null" in summary.lower() or "insufficient history" in summary.lower()


def test_bank_specific_nii_metric():
    """Verify bank stocks calculate Net Interest Income YoY."""
    dates = pd.date_range("2026-06-30", periods=5, freq="-1QE")
    data = {
        dates[0]: [15000.0, 42000.0],  # Net Income, Net Interest Income
        dates[1]: [14000.0, 40000.0],
        dates[2]: [13500.0, 39000.0],
        dates[3]: [13000.0, 38000.0],
        dates[4]: [12500.0, 35000.0],  # Q_{t-4}
    }
    stmt = pd.DataFrame(data, index=["Net Income", "Net Interest Income"])

    fund, summary = compute_strict_yoy_fundamentals("SBIN", stmt)

    # PAT YoY: (15000 - 12500) / 12500 * 100 = +20.0%
    assert fund["yoy_net_profit_pct"] == 20.0
    # NII YoY: (42000 - 35000) / 35000 * 100 = +20.0%
    assert fund["yoy_nii_pct"] == 20.0
    assert "NII YoY: +20.0%" in summary
