"""What a call is worth if the stock reaches its target.

The scanner already produces a target on the *stock*. Traders buying the option
need the other half: what the premium is likely to be there. That cannot be
read off any feed — it has to be modelled — so this module is explicit about
being a model:

  1. solve the implied volatility that reproduces the option's **current**
     traded premium at the current spot and time to expiry;
  2. reprice at the stock's target with that same volatility, after ageing the
     option by the intended holding period.

Both steps are Black-Scholes with zero carry. The assumptions are stated
wherever the output is shown, because they are the whole story:

  * **implied volatility is assumed unchanged.** A breakout that arrives with
    an IV crush pays less than this says; one that arrives with panic bid pays
    more.
  * **decay is charged for the holding period only.** Hold longer than assumed
    and the estimate is optimistic — brutally so on expiry day.
  * a real fill also depends on the order book, which the free feed does not
    publish.

So the projection is a bound on expectations, not a quote.
"""
from __future__ import annotations

import math
from datetime import date, datetime

# A year of calendar days — options decay on wall-clock time, not sessions.
YEAR_DAYS = 365.0
# Below this the maths degenerates (an option at expiry is pure intrinsic), so
# the model refuses rather than dividing by ~zero.
MIN_YEARS = 1e-6
IV_BOUNDS = (0.01, 5.00)      # 1% to 500% — outside this the quote is nonsense


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_price(spot: float, strike: float, years: float, iv: float) -> float:
    """Black-Scholes call value, zero rates and no dividends."""
    if years <= MIN_YEARS or iv <= 0:
        return max(0.0, spot - strike)          # at expiry: intrinsic only
    vt = iv * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * years) / vt
    d2 = d1 - vt
    return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)


def put_price(spot: float, strike: float, years: float, iv: float) -> float:
    """Black-Scholes put value, zero rates and no dividends."""
    if years <= MIN_YEARS or iv <= 0:
        return max(0.0, strike - spot)          # at expiry: intrinsic only
    vt = iv * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * years) / vt
    d2 = d1 - vt
    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def option_price(spot: float, strike: float, years: float, iv: float, is_call: bool = True) -> float:
    """Black-Scholes value for Call or Put."""
    return call_price(spot, strike, years, iv) if is_call else put_price(spot, strike, years, iv)


def implied_vol(price: float, spot: float, strike: float, years: float,
                is_call: bool = True) -> float | None:
    """The volatility that reproduces `price`, or None if none does.

    Bisection rather than Newton: it cannot diverge, and these are traded
    prices that are sometimes stale or crossed, where Newton wanders off.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or years <= MIN_YEARS:
        return None
    intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
    if price <= intrinsic:
        return None                 # no time value to explain — often a stale print
    lo, hi = IV_BOUNDS
    fn = call_price if is_call else put_price
    if fn(spot, strike, years, hi) < price:
        return None                 # price above what even 500% vol produces
    for _ in range(80):
        mid = (lo + hi) / 2
        if fn(spot, strike, years, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def years_to_expiry(expiry: date, now: datetime, close_hour: float = 15.5) -> float:
    """Calendar years from `now` to the expiry-day close."""
    days = (expiry - now.date()).days
    hours_left_today = max(0.0, close_hour - (now.hour + now.minute / 60))
    return max(0.0, days + hours_left_today / 24.0) / YEAR_DAYS


class Projection:
    """A modelled premium at one stock price."""

    __slots__ = ("spot", "premium", "change_pct")

    def __init__(self, spot: float, premium: float, entry_premium: float):
        self.spot = spot
        self.premium = premium
        self.change_pct = ((premium - entry_premium) / entry_premium * 100
                           if entry_premium > 0 else None)


def decay_curve(*, spot: float, strike: float, premium_now: float,
                expiry: date, now: datetime,
                days: tuple[float, ...] = (0.25, 1.0, 2.0, 3.0),
                is_call: bool = True) -> list[dict]:
    """What the option is worth at each horizon **if the underlying does not move**."""
    years = years_to_expiry(expiry, now)
    iv = implied_vol(premium_now, spot, strike, years, is_call=is_call)
    if iv is None:
        return []
    out = []
    for d in days:
        left = max(years - d / YEAR_DAYS, 0.0)
        px = option_price(spot, strike, left, iv, is_call=is_call)
        out.append({"days": d, "premium": px,
                    "pct_left": (px / premium_now * 100) if premium_now > 0 else None})
    return out


def project(*, spot_now: float, strike: float, premium_now: float,
            expiry: date, now: datetime, targets: list[float],
            hold_hours: float = 2.0,
            is_call: bool = True) -> tuple[float | None, list[Projection]]:
    """(implied vol, projected premiums at each target price)."""
    years_now = years_to_expiry(expiry, now)
    iv = implied_vol(premium_now, spot_now, strike, years_now, is_call=is_call)
    if iv is None:
        return None, []
    years_then = max(0.0, years_now - hold_hours / 24.0 / YEAR_DAYS)
    out = [Projection(t, option_price(t, strike, years_then, iv, is_call=is_call), premium_now)
           for t in targets]
    return iv, out
