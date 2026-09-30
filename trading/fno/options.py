"""Call options for a qualifying long setup (§ beyond the futures spec).

What NSE actually serves us decides what this can honestly do. The full option
chain endpoints (`option-chain-equities`, `option-chain-v3`) answer `{}` to
programmatic clients — a soft block, not an error — so there is **no chain per
stock**. What is live is the *20 most-active stock option contracts* board,
market-wide. So:

  * a candidate gets an option view only if its contracts are in that board,
    which is a minority of names. Everything else says "no live option data",
    never a guessed strike or premium;
  * no bid/ask and no implied volatility are published there, so no greeks are
    computed and none are shown. Nothing here is modelled — every number is a
    field NSE returned.

The one number that matters most is **breakeven = strike + premium**. A call
only pays if the stock clears that by expiry, and it routinely sits *above* the
scanner's own 1:2 target on the underlying — meaning the stock setup can work
perfectly and the call still expire worthless. That comparison is computed for
every strike and is the reason a strike gets rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from trading.fno import pricing
from trading.fno.models import IST, Provenance

# A strike this thin is a quote, not a market you can get out of.
MIN_OI = 100                 # open contracts
MIN_VOLUME = 1_000           # contracts traded today
# A premium above this share of spot ties up too much capital for an intraday
# move, whatever the chart says.
MAX_PREMIUM_PCT = 4.0
# The option must make more on the planned move than it loses at the stop.
MIN_OPTION_RR = 1.0


@dataclass
class OptionQuote:
    """One listed contract, as published — nothing derived except breakeven."""
    underlying: str
    identifier: str
    option_type: str            # "Call" / "Put"
    strike: float
    expiry: str
    last_price: float
    pct_change: float | None
    open_interest: float | None
    volume: float | None
    underlying_value: float | None
    prov: Provenance

    @property
    def breakeven(self) -> float:
        """Where the stock must be by expiry for the buyer to be square."""
        return (self.strike + self.last_price if self.option_type == "Call"
                else self.strike - self.last_price)

    @property
    def moneyness(self) -> str:
        spot = self.underlying_value
        if spot is None:
            return "unknown"
        edge = max(spot * 0.005, 0.01)
        if abs(self.strike - spot) <= edge:
            return "ATM"
        if self.option_type == "Call":
            return "OTM" if self.strike > spot else "ITM"
        return "OTM" if self.strike < spot else "ITM"

    def expires_today(self, today: date | None = None) -> bool:
        d = parse_expiry(self.expiry)
        return d is not None and d == (today or datetime.now(IST).date())

    def liquidity_problem(self) -> str | None:
        if self.open_interest is not None and self.open_interest < MIN_OI:
            return f"open interest {self.open_interest:,.0f} — too thin to exit cleanly"
        if self.volume is not None and self.volume < MIN_VOLUME:
            return f"only {self.volume:,.0f} traded today — thin"
        return None


@dataclass
class OptionPlan:
    """The chosen call, or the reason there isn't one."""
    quote: OptionQuote | None
    breakeven: float | None
    breakeven_vs_t1: float | None      # how far breakeven sits above target 1 (%)
    clears_t1: bool
    clears_t2: bool
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    considered: list[OptionQuote] = field(default_factory=list)
    # Modelled premium if the stock reaches the scanner's own levels. None when
    # the traded price cannot be explained by any sane volatility.
    implied_vol: float | None = None
    premium_at_stop: float | None = None
    premium_at_t1: float | None = None
    premium_at_t2: float | None = None
    # What the option is worth at each horizon if the underlying does NOT move.
    decay: list[dict] = field(default_factory=list)

    @property
    def tradeable(self) -> bool:
        return self.quote is not None and not self.rejections

    @property
    def option_rr(self) -> float | None:
        """Reward-to-risk on the OPTION, which is not the stock's ratio."""
        if (self.quote is None or self.premium_at_stop is None
                or self.premium_at_t1 is None):
            return None
        risk = self.quote.last_price - self.premium_at_stop
        reward = self.premium_at_t1 - self.quote.last_price
        return reward / risk if risk > 0 else None


def parse_expiry(text: str | None) -> date | None:
    if not text:
        return None
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def calls_for(symbol: str, board: list[OptionQuote]) -> list[OptionQuote]:
    """Listed calls on this underlying, nearest expiry first, then by strike."""
    rows = [q for q in board
            if q.underlying == symbol and q.option_type == "Call"]
    return sorted(rows, key=lambda q: (parse_expiry(q.expiry) or date.max, q.strike))


def puts_for(symbol: str, board: list[OptionQuote]) -> list[OptionQuote]:
    rows = [q for q in board if q.underlying == symbol and q.option_type == "Put"]
    return sorted(rows, key=lambda q: (parse_expiry(q.expiry) or date.max, q.strike))


def _front_expiry(calls: list[OptionQuote]) -> list[OptionQuote]:
    """Just the nearest expiry (calls arrive sorted by expiry, then strike)."""
    if not calls:
        return []
    first = parse_expiry(calls[0].expiry)
    return [q for q in calls if parse_expiry(q.expiry) == first]


def plan_call(symbol: str, board: list[OptionQuote], *, spot: float,
              target1: float, target2: float, stop: float | None = None,
              now: datetime | None = None,
              today: date | None = None) -> OptionPlan:
    """Pick the call that a long on this stock could actually be expressed in.

    A strike qualifies only when the stock reaching the scanner's own targets
    would put the option in profit — the breakeven test — and when the contract
    is liquid enough to leave. Otherwise the plan says why not.
    """
    every = calls_for(symbol, board)
    # An intraday trade belongs in the nearest expiry. A far-month contract can
    # model a better reward-to-risk while being untradeable in practice — on a
    # deep chain the back months carry a fraction of the open interest, and you
    # have to be able to get out. Later expiries are only considered if nothing
    # in the front one works.
    listed = _front_expiry(every)
    if not listed:
        return OptionPlan(None, None, None, False, False,
                          rejections=["no live option data for this name — NSE "
                                      "publishes only the 20 most-active option "
                                      "contracts and this is not among them"])

    plan = OptionPlan(None, None, None, False, False, considered=listed)
    viable: list[tuple[float, OptionQuote, dict]] = []
    near_misses: list[str] = []

    for q in listed:
        if q.last_price <= 0:
            continue
        if spot > 0 and (q.last_price / spot * 100) > MAX_PREMIUM_PCT:
            continue
        thin = q.liquidity_problem()
        if thin:
            near_misses.append(f"{q.strike:g} CE: {thin}")
            continue

        # The gate is what the option is worth AT THE TARGET, not at expiry.
        # An intraday trade is closed on the move, where a call gains from delta
        # long before spot reaches strike+premium — so breakeven is reported as
        # information, never used to veto.
        proj = _model(q, spot=spot, stop=stop, target1=target1, target2=target2,
                      now=now, today=today)
        if proj is None:
            # No usable model (stale print). Fall back to the conservative
            # hold-to-expiry test rather than guessing.
            if q.breakeven > target2:
                near_misses.append(
                    f"{q.strike:g} CE at {q.last_price:.2f}: no usable price model, "
                    f"and its expiry breakeven {q.breakeven:.2f} is beyond the "
                    f"1:3 target {target2:.2f}")
                continue
            viable.append((abs(q.strike - spot), q, {}))
            continue

        rr = proj.get("rr")
        if rr is None:
            # No stop means no risk to measure, so the intraday gate cannot be
            # applied — fall back to the conservative hold-to-expiry test rather
            # than letting the strike through ungated.
            if q.breakeven > target2:
                near_misses.append(
                    f"{q.strike:g} CE at {q.last_price:.2f} breaks even at "
                    f"{q.breakeven:.2f} — beyond the 1:3 target {target2:.2f}, and "
                    "with no stop given its reward-to-risk cannot be checked")
                continue
        elif rr < MIN_OPTION_RR:
            near_misses.append(
                f"{q.strike:g} CE: risks {q.last_price - proj['at_stop']:.2f} to make "
                f"{proj['at_t1'] - q.last_price:.2f} on the move — {rr:.2f}:1, worse "
                "than one-to-one")
            continue
        # Among strikes that pay, prefer the best reward-to-risk, then the one
        # nearest the money.
        viable.append((-(rr or 0) + abs(q.strike - spot) / max(spot, 1) * 10, q, proj))

    if not viable:
        later = [q for q in every if q not in listed]
        if later:
            near_misses.append(
                f"nothing works in the {listed[0].expiry} expiry; later expiries "
                "are not offered for an intraday trade")
        plan.rejections = near_misses[:3] or [
            "no listed call makes more on the planned move than it loses at the stop"]
        return plan

    viable.sort(key=lambda x: x[0])
    best = viable[0][1]
    be = best.breakeven
    plan.quote = best
    plan.breakeven = be
    plan.breakeven_vs_t1 = (be - target1) / target1 * 100 if target1 else None
    plan.clears_t1 = be < target1
    plan.clears_t2 = be < target2

    exp = parse_expiry(best.expiry)
    if exp is not None:
        plan.decay = pricing.decay_curve(
            spot=spot, strike=best.strike, premium_now=best.last_price,
            expiry=exp, now=now or datetime.now(IST))

    modelled = viable[0][2]
    if modelled:
        plan.implied_vol = modelled["iv"]
        plan.premium_at_t1 = modelled["at_t1"]
        plan.premium_at_t2 = modelled["at_t2"]
        plan.premium_at_stop = modelled.get("at_stop")
    else:
        plan.warnings.append("premium does not fit any sane volatility — likely a "
                             "stale print; no exit price is modelled")

    if plan.decay:
        tail = plan.decay[-1]
        if tail["pct_left"] is not None and tail["pct_left"] < 25:
            plan.warnings.append(
                f"if the move does not come, this keeps only "
                f"{tail['pct_left']:.0f}% of its value after {tail['days']:g} days "
                "— it is a bet on the move happening now, not soon")

    if best.expires_today(today):
        plan.warnings.append("EXPIRES TODAY — premium decays to intrinsic value by "
                             "the close; an intraday stall loses money even if the "
                             "stock does not fall")
    if best.moneyness == "OTM":
        plan.warnings.append(f"out of the money — the stock must clear {be:.2f} "
                             "for this to pay, not merely hold its breakout")
    if not plan.clears_t1:
        plan.warnings.append(
            f"breakeven {be:.2f} is above target 1 — if you HOLD TO EXPIRY only "
            "the 1:3 target pays. Exiting on the move is priced above.")
    return plan


def plan_put(symbol: str, board: list[OptionQuote], *, spot: float,
             target1: float, target2: float, stop: float | None = None,
             now: datetime | None = None,
             today: date | None = None) -> OptionPlan:
    """Pick the put option that a short/bearish setup on this stock could be expressed in."""
    every = puts_for(symbol, board)
    listed = _front_expiry(every)
    if not listed:
        return OptionPlan(None, None, None, False, False,
                          rejections=["no live option data for this name — no puts available"])

    plan = OptionPlan(None, None, None, False, False, considered=listed)
    viable: list[tuple[float, OptionQuote, dict]] = []
    near_misses: list[str] = []

    for q in listed:
        if q.last_price <= 0:
            continue
        if spot > 0 and (q.last_price / spot * 100) > MAX_PREMIUM_PCT:
            continue
        thin = q.liquidity_problem()
        if thin:
            near_misses.append(f"{q.strike:g} PE: {thin}")
            continue

        proj = _model(q, spot=spot, stop=stop, target1=target1, target2=target2,
                      now=now, today=today)
        if proj is None:
            # Conservative fallback: put breaks even at strike - last_price.
            # Must clear 1:3 target downwards.
            if q.breakeven < target2:
                near_misses.append(
                    f"{q.strike:g} PE at {q.last_price:.2f}: no usable price model, "
                    f"and its expiry breakeven {q.breakeven:.2f} is beyond the "
                    f"1:3 target {target2:.2f}")
                continue
            viable.append((abs(q.strike - spot), q, {}))
            continue

        rr = proj.get("rr")
        if rr is None:
            if q.breakeven < target2:
                near_misses.append(
                    f"{q.strike:g} PE at {q.last_price:.2f} breaks even at "
                    f"{q.breakeven:.2f} — beyond the 1:3 target {target2:.2f}")
                continue
        elif rr < MIN_OPTION_RR:
            near_misses.append(
                f"{q.strike:g} PE: risks {q.last_price - proj['at_stop']:.2f} to make "
                f"{proj['at_t1'] - q.last_price:.2f} on the move — {rr:.2f}:1, worse "
                "than one-to-one")
            continue

        viable.append((-(rr or 0) + abs(q.strike - spot) / max(spot, 1) * 10, q, proj))

    if not viable:
        later = [q for q in every if q not in listed]
        if later:
            near_misses.append(
                f"nothing works in the {listed[0].expiry} expiry; later expiries "
                "are not offered for an intraday trade")
        plan.rejections = near_misses[:3] or [
            "no listed put makes more on the planned move than it loses at the stop"]
        return plan

    viable.sort(key=lambda x: x[0])
    best = viable[0][1]
    be = best.breakeven
    plan.quote = best
    plan.breakeven = be
    plan.breakeven_vs_t1 = (target1 - be) / target1 * 100 if target1 else None
    plan.clears_t1 = be > target1
    plan.clears_t2 = be > target2

    exp = parse_expiry(best.expiry)
    if exp is not None:
        plan.decay = pricing.decay_curve(
            spot=spot, strike=best.strike, premium_now=best.last_price,
            expiry=exp, now=now or datetime.now(IST), is_call=False)

    modelled = viable[0][2]
    if modelled:
        plan.implied_vol = modelled["iv"]
        plan.premium_at_t1 = modelled["at_t1"]
        plan.premium_at_t2 = modelled["at_t2"]
        plan.premium_at_stop = modelled.get("at_stop")
    else:
        plan.warnings.append("premium does not fit any sane volatility — likely a "
                             "stale print; no exit price is modelled")

    if best.expires_today(today):
        plan.warnings.append("EXPIRES TODAY — premium decays to intrinsic value by "
                             "the close; an intraday stall loses money")
    if best.moneyness == "OTM":
        plan.warnings.append(f"out of the money — the stock must drop below {be:.2f} "
                             "for this to pay")
    if not plan.clears_t1:
        plan.warnings.append(
            f"breakeven {be:.2f} is below target 1 — if you HOLD TO EXPIRY only "
            "the 1:3 target pays. Exiting on the move is priced above.")
    return plan


def plan_option(symbol: str, board: list[OptionQuote], *, side: str = "BUY",
                spot: float, target1: float, target2: float, stop: float | None = None,
                now: datetime | None = None, today: date | None = None) -> OptionPlan:
    """Unified helper to choose Call (for BUY/LONG) or Put (for SELL/SHORT)."""
    if side.upper() in ("SELL", "PUT", "PE", "SHORT"):
        return plan_put(symbol, board, spot=spot, target1=target1, target2=target2,
                        stop=stop, now=now, today=today)
    return plan_call(symbol, board, spot=spot, target1=target1, target2=target2,
                     stop=stop, now=now, today=today)


def _model(q: OptionQuote, *, spot: float, stop: float | None,
           target1: float, target2: float, now: datetime | None,
           today: date | None) -> dict | None:
    """Premium at the stop and both targets, or None if unmodellable."""
    expiry = parse_expiry(q.expiry)
    if expiry is None:
        return None
    when = now or datetime.combine(today or datetime.now(IST).date(),
                                   datetime.min.time()).replace(tzinfo=IST)
    levels = [target1, target2] + ([stop] if stop is not None else [])
    is_call = (q.option_type == "Call")
    iv, proj = pricing.project(spot_now=spot, strike=q.strike,
                               premium_now=q.last_price, expiry=expiry,
                               now=when, targets=levels, is_call=is_call)
    if iv is None:
        return None
    out = {"iv": iv, "at_t1": proj[0].premium, "at_t2": proj[1].premium}
    if stop is not None:
        out["at_stop"] = proj[2].premium
        risk = q.last_price - out["at_stop"]
        out["rr"] = ((out["at_t1"] - q.last_price) / risk) if risk > 0 else None
    return out


def describe(plan: OptionPlan) -> str:
    """One-line summary for the terminal report."""
    if plan.quote is None:
        return "options: NO CALL — " + (plan.rejections[0] if plan.rejections
                                        else "no candidate strike")
    q = plan.quote
    kind = "CE" if q.option_type == "Call" else "PE"
    return (f"options: {q.strike:g} {kind} @ {q.last_price:.2f} ({q.expiry}, "
            f"{q.moneyness}) — breakeven {plan.breakeven:.2f}, "
            f"OI {q.open_interest:,.0f}")
