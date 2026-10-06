"""Scan orchestration and the time-window rules (§1, §2, §16, §17, §18).

Order of operations is deliberate:

    data-integrity gate → market context → liquidity screen → per-stock workup
    → score → veto filters → rank

The gate comes first because grading setups on stale or partial data is the
failure mode the spec is most emphatic about; the veto filters come after
scoring so a rejection is visible with its score attached rather than the
candidate silently vanishing.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time

from trading.fno import config as C
from trading.fno import context as ctx
from trading.fno import filters
from trading.fno import indicators as ind
from trading.fno import oi as oi_mod
from trading.fno import chain_analytics as ca_mod
from trading.fno import index_options as ix_mod
from trading.fno import pivots as pv_mod
from trading.fno import options as opt_mod
from trading.fno import scoring
from trading.fno import setup as setup_mod
from trading.fno.models import IST, Candidate, ScanResult

STALE_DATA_BANNER = "REAL-TIME DATA UNAVAILABLE — SCAN NOT RELIABLE."


def _t(text: str) -> time:
    h, m = text.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class Window:
    label: str
    signals_allowed: bool
    note: str


def window_for(now: datetime) -> Window:
    t = now.time()
    if t < _t(C.MARKET_OPEN):
        return Window("PRE-OPEN", False, "market has not opened")
    if t < _t(C.NO_SIGNAL_BEFORE):
        return Window("09:15–09:20 OBSERVATION", False,
                      "opening volatility — observation only, no BUY signals (§2)")
    if t < _t(C.OPENING_RANGE_END):
        return Window("09:20–09:30 CANDIDATE ID", False,
                      "the 09:15–09:30 range is still forming — candidates are "
                      "listed but no breakout can be graded yet")
    if t < _t("09:45"):
        return Window("09:30–09:45 TREND EVALUATION", True,
                      "evaluating trend, VWAP, EMA, volume and OI")
    if t < _t(C.SCANNER_WINDOW_END):
        return Window("09:45–10:00 BREAKOUT + RETEST", True,
                      "looking for confirmed breakout-and-retest setups")
    return Window("POST-10:00 MONITORING", True,
                  "outside the opening scanner window — these are monitoring "
                  "setups, labelled separately (§2)")


class Scanner:
    def __init__(self, feed, now: datetime | None = None, *,
                 allow_delayed: bool = False,
                 symbols: list[str] | None = None,
                 shortlist: int = C.SHORTLIST_SIZE,
                 with_options: bool = True):
        self.feed = feed
        self.now = now or datetime.now(IST)
        self.allow_delayed = allow_delayed
        self.symbols = symbols
        self.shortlist_size = shortlist
        self.with_options = with_options
        self.option_board: list = []
        self.skips: Counter[str] = Counter()

    # --- screening ------------------------------------------------------

    def screen(self, universe: list[dict], board: dict) -> list[str]:
        """Liquid, actually-moving names that have a futures contract (§1)."""
        if self.symbols:
            return [s for s in self.symbols]
        scored = []
        for row in universe:
            sym, pct = row["symbol"], row.get("pct")
            if sym not in board:
                continue
            vol, val = row.get("volume"), row.get("value_cr")
            if vol is not None and vol < C.MIN_CASH_VOLUME:
                continue
            if val is not None and val < C.MIN_CASH_VALUE_CR:
                continue
            if pct is None or not (C.MIN_ABS_MOVE_PCT <= abs(pct) <= C.MAX_ABS_MOVE_PCT):
                continue
            if pct <= 0:
                continue                       # long scanner: skip red names
            fut = board[sym]
            liquidity = fut.turnover_cr or 0
            scored.append((pct + min(liquidity / 500, 1.0), sym))
        scored.sort(reverse=True)
        return [s for _, s in scored[:self.shortlist_size]]

    # --- per stock ------------------------------------------------------

    def attach_options(self, cand: Candidate) -> None:
        """A call plan only makes sense once the stock has real targets."""
        if cand.trade is None:
            return
        board = self.option_board
        if hasattr(self.feed, "stock_option_board"):
            try:
                stock_board = self.feed.stock_option_board(cand.symbol)
                if stock_board:
                    board = stock_board
            except Exception:
                pass
        if not board:
            return
        cand.option_plan = opt_mod.plan_call(
            cand.symbol, board, spot=cand.levels.price,
            target1=cand.trade.target1, target2=cand.trade.target2,
            stop=cand.trade.stop, now=self.now, today=self.now.date())

    def evaluate(self, symbol: str, candles, fut, market) -> Candidate | None:
        """One stock, fully worked up — or None, with the reason recorded.

        Skips are counted rather than silently dropped: a screen showing no
        candidates must be able to say whether nothing set up or nothing was
        gradeable.
        """
        df = ind.closed_bars(candles.df, self.now)
        if df.empty:
            self.skips["no closed bars yet"] += 1
            return None
        day = ind.sessions(df)[-1]
        if day != self.now.date():
            self.skips[f"no candles for today — the feed's last session is {day}"] += 1
            return None

        pct = fut.pct_change
        levels = setup_mod.build_levels(df, day, self.now, pct)
        if levels is None:
            self.skips["opening range or volume missing for today"] += 1
            return None
        derived_pct = False
        if pct is None and levels.prev_close:
            # The futures %change is only published for the most-active
            # contracts; for the rest the underlying's move stands in.
            pct = (levels.price - levels.prev_close) / levels.prev_close * 100
            levels.pct_change = pct        # keep the display and the OI read in step
            derived_pct = True

        structure = setup_mod.read_structure(df, day, levels)
        oi_read = oi_mod.classify(pct, fut.oi_change_pct)
        sector = C.SECTORS.get(symbol, "UNMAPPED")

        cand = Candidate(
            symbol=symbol, sector=sector, fut=fut, levels=levels,
            structure=structure, oi=oi_read,
            rel_strength=(pct or 0) - (market.nifty_pct or 0),
            sector_strength=market.sector_pct.get(sector),
            sources=[candles.prov, fut.prov],
        )

        age = fut.prov.age_min(self.now)
        if age > C.MAX_DATA_AGE_MIN:
            cand.warnings.append(f"futures snapshot is {age:.1f} min old")
        if candles.prov.delayed:
            cand.warnings.append(f"candles are exchange-delayed "
                                 f"({candles.prov.age_min(self.now):.0f} min)")
        if sector == "UNMAPPED":
            cand.warnings.append("sector not mapped — sector alignment unscored")
        if derived_pct:
            cand.warnings.append("futures %change unpublished — the OI read pairs "
                                 "futures OI with the underlying's price move")

        cand.spark = self._spark(df, day)

        trade, problems = setup_mod.build_trade(df, day, levels, structure)
        cand.trade = trade
        self.attach_options(cand)
        scoring.score_candidate(cand, market)
        filters.apply(cand, market, problems)
        if trade is None:
            cand.trade = None
        return cand

    @staticmethod
    def _spark(df, day, limit: int = 75) -> list[dict]:
        """Today's closed bars with running VWAP, EMA9, EMA21, OHLC, and Volume for candlestick chart."""
        import pandas as pd
        day_df = ind.session_of(df, day)
        if day_df.empty:
            return []
        vw = ind.vwap(day_df)
        e9 = ind.ema(df["Close"], 9)
        e21 = ind.ema(df["Close"], 21)
        rows = []
        for ts, bar in day_df.tail(limit).iterrows():
            e9_val = round(float(e9.loc[ts]), 2) if ts in e9.index and pd.notna(e9.loc[ts]) else None
            e21_val = round(float(e21.loc[ts]), 2) if ts in e21.index and pd.notna(e21.loc[ts]) else None
            w_val = round(float(vw.loc[ts]), 2) if ts in vw.index and pd.notna(vw.loc[ts]) else round(float(bar["Close"]), 2)
            rows.append({
                "t": ts.strftime("%H:%M"),
                "o": round(float(bar["Open"]), 2),
                "h": round(float(bar["High"]), 2),
                "l": round(float(bar["Low"]), 2),
                "c": round(float(bar["Close"]), 2),
                "v": float(bar["Volume"]) if "Volume" in bar else 0.0,
                "w": w_val,
                "e9": e9_val,
                "e21": e21_val,
            })
        return rows

    # --- orchestration --------------------------------------------------

    def run(self) -> ScanResult:
        win = window_for(self.now)
        notes: list[str] = [f"{win.label}: {win.note}"]
        if C.RELAXED:
            notes.append(
                "RELAXED GRADING — an OI-flat read is shown as WATCH instead of "
                "rejected. Nothing relaxed can be a BUY; the score floor and "
                "every other gate are unchanged.")

        universe = self.feed.universe()
        board = self.feed.futures_board()
        if self.with_options and hasattr(self.feed, "option_board"):
            self.option_board = self.feed.option_board()
        if not board:
            notes.append("stock-futures open interest could not be fetched — no "
                         "setup can be OI-confirmed")

        shortlist = self.screen(universe, board)
        market = ctx.build(self.feed, universe, self.now.date(), self.now)
        notes.extend(market.notes)

        candles = self.feed.candles(shortlist) if shortlist else {}
        delayed = any(c.prov.delayed for c in candles.values())
        data_ok = True
        if delayed and not self.allow_delayed:
            data_ok = False
            notes.append(f"{STALE_DATA_BANNER} Candle feed is exchange-delayed "
                         "(yfinance). Re-run with --allow-delayed to see the "
                         "analysis anyway, or wire a live broker feed.")
        if not candles:
            data_ok = False
            notes.append(f"{STALE_DATA_BANNER} No intraday candles were returned.")
        if not win.signals_allowed:
            notes.append(f"No BUY signal will be emitted in the {win.label} window.")

        candidates: list[Candidate] = []
        if data_ok:
            for sym in shortlist:
                c = candles.get(sym)
                fut = board.get(sym)
                if c is None or fut is None:
                    continue
                cand = self.evaluate(sym, c, fut, market)
                if cand is None:
                    continue
                if not win.signals_allowed:
                    cand.trade = None
                    cand.rejections.append(
                        f"{win.label}: signals are not emitted in this window")
                    cand.blockers.append("window")
                candidates.append(cand)

        for reason, count in self.skips.most_common():
            notes.append(f"{count} of {len(shortlist)} screened names skipped: {reason}")
        if data_ok and shortlist and not candidates:
            notes.append("Nothing could be graded this run — see the skip reasons "
                         "above. Outside market hours this is expected.")

        index_plan = banknifty_plan = chain_read = piv = gap = None
        if data_ok and self.with_options:
            if hasattr(self.feed, "index_option_board"):
                try:
                    chain = self.feed.index_option_board()
                    nifty = self.feed.index_candles(C.NIFTY_TICKER)
                    if nifty is not None:
                        index_plan = ix_mod.analyse(nifty.df, self.now, market, chain,
                                                    index_symbol="NIFTY", name="NIFTY 50")
                        bars = ind.closed_bars(nifty.df, self.now)
                        if not bars.empty:
                            day = ind.sessions(bars)[-1]
                            piv = pv_mod.from_candles(bars, day)
                            gap = pv_mod.gap(bars, day)
                    chain_read = ca_mod.analyse(chain, "NIFTY", self.now.date())
                except Exception as exc:                    # noqa: BLE001
                    notes.append(f"NIFTY option read failed: {exc}")

            # BANK NIFTY option setup
            try:
                bank_chain = (self.feed.banknifty_option_board()
                              if hasattr(self.feed, "banknifty_option_board")
                              else getattr(self, "option_board", []))
                bank_candles = self.feed.index_candles(C.BANKNIFTY_TICKER)
                if bank_candles is not None:
                    banknifty_plan = ix_mod.analyse(bank_candles.df, self.now, market, bank_chain,
                                                    index_symbol="BANKNIFTY", name="BANK NIFTY")
            except Exception as exc:                        # noqa: BLE001
                notes.append(f"BANK NIFTY option read failed: {exc}")

        candidates.sort(key=lambda c: (c.tradeable, c.score, c.rel_strength),
                        reverse=True)
        notes.extend(getattr(self.feed, "errors", []))
        return ScanResult(when=self.now, window=win.label,
                          signals_allowed=win.signals_allowed,
                          window_note=win.note, market=market,
                          candidates=candidates, considered=len(shortlist),
                          data_ok=data_ok, data_notes=notes,
                          index_plan=index_plan, banknifty_plan=banknifty_plan,
                          chain_read=chain_read, pivots=piv, gap=gap)
