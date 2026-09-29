"""Unit tests for paired trailing stop comparison logic in trading/compare_trail.py.

Verifies:
1. Partial fill aggregation into parent position prior to matching.
2. Proper detection of matched vs unmatched positions across arms.
3. Paired statistical math: mean difference, sample standard error, and 95% confidence interval.
"""
import math
import pytest
from trading.compare_trail import compute_paired_diffs, extract_positions


def test_extract_positions_aggregates_partials():
    """Verify that multiple trade fills from the same entry are summed together."""
    trades = [
        {"symbol": "INFY", "entry_ts": 1000, "net": 25.5},  # Partial fill
        {"symbol": "INFY", "entry_ts": 1000, "net": 34.5},  # Final remainder
        {"symbol": "TCS", "entry_ts": 2000, "net": -15.0},
    ]
    pos = extract_positions(trades)
    assert len(pos) == 2
    assert pos[("INFY", 1000)] == pytest.approx(60.0)
    assert pos[("TCS", 2000)] == pytest.approx(-15.0)


def test_paired_comparison_matched_and_unmatched_counts():
    """Verify matched/unmatched bookkeeping across two arms."""
    pos_off = {
        ("INFY", 1000): 10.0,
        ("TCS", 2000): -5.0,
        ("SBIN", 3000): 12.0,  # Exists only in OFF
    }
    pos_on = {
        ("INFY", 1000): 14.0,  # Matched (diff +4)
        ("TCS", 2000): -2.0,   # Matched (diff +3)
        ("RELIANCE", 4000): 8.0,  # Exists only in ON
    }

    res = compute_paired_diffs(pos_off, pos_on)
    assert res["n_matched"] == 2
    assert res["unmatched_off"] == 1
    assert res["unmatched_on"] == 1
    assert res["total_diff"] == pytest.approx(7.0)
    assert res["mean_diff"] == pytest.approx(3.5)


def test_paired_comparison_zero_diff_on_identical_arms():
    """When comparing identical arms, difference and CI should be exactly 0."""
    pos = {
        ("INFY", 1000): 10.0,
        ("TCS", 2000): -5.0,
    }
    res = compute_paired_diffs(pos, pos)
    assert res["n_matched"] == 2
    assert res["unmatched_off"] == 0
    assert res["unmatched_on"] == 0
    assert res["total_diff"] == 0.0
    assert res["mean_diff"] == 0.0
    assert res["ci95"] == 0.0


def test_paired_comparison_exact_hand_calculated_math():
    """Verify exact formula calculations for sample variance, standard error, and 95% CI.

    Hand-built test dataset:
    Position 1: OFF = 10.0, ON = 15.0 -> Diff = +5.0
    Position 2: OFF = -4.0, ON = -1.0 -> Diff = +3.0
    Position 3: OFF =  2.0, ON =  6.0 -> Diff = +4.0
    Position 4: OFF =  8.0, ON =  8.0 -> Diff =  0.0

    Diffs = [5.0, 3.0, 4.0, 0.0]
    N = 4
    Sum = 12.0
    Mean = 3.0
    Deviations = [+2.0, 0.0, +1.0, -3.0]
    Sum of Squared Deviations = 4.0 + 0.0 + 1.0 + 9.0 = 14.0
    Sample Variance (N - 1 = 3) = 14.0 / 3 = 4.6666666667
    Sample SD = sqrt(14 / 3) = 2.16024689947
    Standard Error = SD / sqrt(4) = 1.08012344973
    95% CI Half-Width = 1.96 * SE = 2.11704196148
    """
    pos_off = {
        ("P1", 1): 10.0,
        ("P2", 2): -4.0,
        ("P3", 3): 2.0,
        ("P4", 4): 8.0,
    }
    pos_on = {
        ("P1", 1): 15.0,
        ("P2", 2): -1.0,
        ("P3", 3): 6.0,
        ("P4", 4): 8.0,
    }

    res = compute_paired_diffs(pos_off, pos_on)
    assert res["n_matched"] == 4
    assert res["total_diff"] == pytest.approx(12.0)
    assert res["mean_diff"] == pytest.approx(3.0)

    expected_sd = math.sqrt(14.0 / 3.0)
    expected_se = expected_sd / math.sqrt(4.0)
    expected_ci = 1.96 * expected_se

    assert res["ci95"] == pytest.approx(expected_ci, rel=1e-6)
    assert res["lo"] == pytest.approx(3.0 - expected_ci, rel=1e-6)
    assert res["hi"] == pytest.approx(3.0 + expected_ci, rel=1e-6)
