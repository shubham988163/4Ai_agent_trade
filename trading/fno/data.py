"""Feeds: candles, futures OI, indices — each stamped with its provenance.

Two implementations:

  LiveFeed    5-minute OHLCV from yfinance (exchange-delayed, ~15 min) +
              stock-futures OI from NSE's public endpoints (near-live).
              The mix is deliberate and is surfaced honestly: the scanner
              refuses to grade setups on delayed candles unless the operator
              passes --allow-delayed, and every printed value carries its
              source and age.

  ReplayFeed  a saved JSON bundle — used by the tests and for reproducing a
              past scan without hitting the network.

Swapping in a broker feed (Kite historical + quote) means implementing the same
four methods; nothing downstream reads a provider directly.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from trading.fno import config as C
from trading.fno import nse as nse_mod
from trading.fno.models import IST, Candles, FuturesSnapshot, Provenance


def _crore(lakhs: float | None) -> float | None:
    """NSE reports traded value/turnover in lakhs; 1 crore = 100 lakh.

    Values that convert to an implausibly small number are returned as None
    rather than guessed at, so the illiquidity filter never fires on a unit
    mismatch in an undocumented field.
    """
    if lakhs is None:
        return None
    cr = lakhs / 100.0
    return cr if cr > 0.01 else None


def _bar_end(df: pd.DataFrame) -> datetime:
    return (df.index[-1] + pd.Timedelta(minutes=C.BAR_MINUTES)).to_pydatetime()


class LiveFeed:
    """yfinance candles + NSE public open interest."""

    source_candles = "yfinance"
    source_oi = "nse-public"
    # universe() returns the top-20 gainers/losers, not a cross-section.
    universe_is_biased = True

    def __init__(self, period: str = "5d"):
        self.period = period
        self.universe_time: datetime | None = None
        self._nse: nse_mod.NSEClient | None = None
        self.errors: list[str] = []

    @property
    def nse(self) -> nse_mod.NSEClient:
        if self._nse is None:
            self._nse = nse_mod.NSEClient()
        return self._nse

    # --- universe -------------------------------------------------------

    def universe(self) -> list[dict]:
        """Today's F&O movers with % change, cash volume and traded value.

        NSE serves the top 20 gainers and top 20 losers of the F&O segment;
        that is the pool, and it is stated in the report rather than implied to
        be the full 200-name universe.
        """
        try:
            gainers, losers, stamp = self.nse.movers()
        except nse_mod.NSEError as exc:
            self.errors.append(f"F&O movers unavailable ({exc}) — the liquidity "
                               "and movement screen could not run")
            return []
        self.universe_time = nse_mod.parse_timestamp(stamp)
        out = []
        for row in gainers + losers:
            sym = nse_mod.pick(row, "symbol")
            if not sym:
                continue
            pct = nse_mod.num(nse_mod.pick(row, "perChange", "net_price"))
            out.append({
                "symbol": sym,
                "pct": pct,
                "ltp": nse_mod.num(nse_mod.pick(row, "ltp")),
                "prev_close": nse_mod.num(nse_mod.pick(row, "prev_price")),
                "volume": nse_mod.num(nse_mod.pick(row, "trade_quantity")),
                "value_cr": _crore(nse_mod.num(nse_mod.pick(row, "turnover"))),
            })
        return out

    # --- futures --------------------------------------------------------

    def futures_board(self) -> dict[str, FuturesSnapshot]:
        """Per-underlying futures state, assembled from three endpoints.

        OI and its day-on-day change come from the OI-spurts endpoint, futures
        turnover from most-active-underlying, and the real contract identifier
        and futures last price from the active-futures board where the name
        appears in it. Anything missing stays None — the OI classifier treats a
        missing OI change as UNAVAILABLE and the candidate gets vetoed.
        """
        fetched = datetime.now(IST)
        try:
            spurts, stamp = self.nse.oi_spurts()
        except nse_mod.NSEError as exc:
            self.errors.append(f"open interest unavailable ({exc}) — no setup "
                               "can be OI-confirmed (§18)")
            return {}
        data_time = nse_mod.parse_timestamp(stamp) or fetched

        turnover: dict[str, float | None] = {}
        try:
            for row in self.nse.most_active_underlying():
                sym = nse_mod.pick(row, "symbol")
                if sym:
                    turnover[sym] = _crore(nse_mod.num(nse_mod.pick(row, "futTurnover")))
        except nse_mod.NSEError as exc:
            self.errors.append(f"futures turnover unavailable ({exc}) — the "
                               "illiquid-contract filter cannot run")

        contracts: dict[str, dict] = {}
        try:
            rows, _ = self.nse.active_futures()
            for row in rows:
                sym = nse_mod.pick(row, "underlying")
                if sym:
                    contracts.setdefault(sym, row)
        except nse_mod.NSEError as exc:
            self.errors.append(f"active-futures board unavailable ({exc}) — "
                               "contract identifiers fall back to a generic label")

        self.errors.append("futures bid/ask is not served by NSE's public API — "
                           "the spread filter (§1) is skipped this run")

        board: dict[str, FuturesSnapshot] = {}
        for row in spurts:
            sym = nse_mod.pick(row, "symbol")
            if not sym or sym in ("NIFTY", "BANKNIFTY", "FINNIFTY",
                                  "MIDCPNIFTY", "NIFTYNXT50"):
                continue
            latest = nse_mod.num(nse_mod.pick(row, "latestOI"))
            prev = nse_mod.num(nse_mod.pick(row, "prevOI"))
            change = nse_mod.num(nse_mod.pick(row, "changeInOI"))
            oi_pct = (change / prev * 100) if (prev and change is not None and prev > 0) else None

            fut = contracts.get(sym, {})
            last = (nse_mod.num(nse_mod.pick(fut, "lastPrice"))
                    or nse_mod.num(nse_mod.pick(row, "underlyingValue")))
            if last is None:
                continue
            board[sym] = FuturesSnapshot(
                symbol=sym,
                contract=str(nse_mod.pick(fut, "identifier") or f"{sym} near-month FUT"),
                expiry=nse_mod.pick(fut, "expiryDate"),
                last_price=last,
                prev_close=nse_mod.num(nse_mod.pick(fut, "closePrice")),
                pct_change=nse_mod.num(nse_mod.pick(fut, "pChange")),
                open_interest=latest,
                oi_change_pct=oi_pct,
                volume=nse_mod.num(nse_mod.pick(fut, "volume")),
                turnover_cr=turnover.get(sym),
                bid=None, ask=None,
                prov=Provenance(self.source_oi, f"{sym} FUT", "snapshot",
                                data_time, fetched),
            )
        return board

    def futures_detail(self, symbol: str, fallback: FuturesSnapshot | None = None
                       ) -> FuturesSnapshot | None:
        """No per-symbol derivative quote is publicly reachable any more
        (/api/quote-derivative returns 404), so the board value stands."""
        return fallback

    # --- indices --------------------------------------------------------

    def market_snapshot(self) -> dict | None:
        """NIFTY, BANK NIFTY, the sector indices and breadth, in one call."""
        fetched = datetime.now(IST)
        try:
            rows, stamp = self.nse.all_indices()
        except nse_mod.NSEError as exc:
            self.errors.append(f"index snapshot unavailable ({exc}) — market "
                               "context falls back to delayed index candles")
            return None
        pct = {}
        adv = dec = None
        for row in rows:
            name = nse_mod.pick(row, "index", "indexSymbol")
            if not name:
                continue
            pct[name] = nse_mod.num(nse_mod.pick(row, "percentChange"))
            if name == "NIFTY 50":
                adv = nse_mod.num(nse_mod.pick(row, "advances"))
                dec = nse_mod.num(nse_mod.pick(row, "declines"))
        return {
            "pct": pct, "advances": adv, "declines": dec,
            "prov": Provenance(self.source_oi, "NSE indices", "snapshot",
                               nse_mod.parse_timestamp(stamp) or fetched, fetched),
        }

    def index_option_board(self) -> list:
        """The full NIFTY option chain, as OptionQuote objects."""
        return self._options(lambda: self.nse.index_options("nse50_opt"), "NIFTY chain")

    def banknifty_option_board(self) -> list:
        """The full BANK NIFTY option chain, as OptionQuote objects."""
        return self._options(lambda: self.nse.index_options("nifty_bank_opt"), "BANK NIFTY chain")

    def option_board(self) -> list:
        """Most-active stock option contracts, as OptionQuote objects."""
        return self._options(lambda: self.nse.active_options(), "option board")

    def stock_option_board(self, symbol: str) -> list:
        """Options for a specific stock."""
        board = self.option_board()
        return [q for q in board if q.underlying == symbol]

    def _options(self, fetch, label: str) -> list:
        from trading.fno.options import OptionQuote

        fetched = datetime.now(IST)
        try:
            rows, stamp = fetch()
        except nse_mod.NSEError as exc:
            self.errors.append(f"{label} unavailable ({exc})")
            return []
        stamp_dt = nse_mod.parse_timestamp(stamp) or fetched
        out = []
        for row in rows:
            sym = nse_mod.pick(row, "underlying")
            kind = nse_mod.pick(row, "optionType")
            strike = nse_mod.num(nse_mod.pick(row, "strikePrice"))
            ltp = nse_mod.num(nse_mod.pick(row, "lastPrice"))
            if not sym or kind not in ("Call", "Put") or strike is None or ltp is None:
                continue
            ident = str(nse_mod.pick(row, "identifier") or f"{sym}-{kind}-{strike}")
            out.append(OptionQuote(
                underlying=sym, identifier=ident, option_type=kind, strike=strike,
                expiry=nse_mod.pick(row, "expiryDate"), last_price=ltp,
                pct_change=nse_mod.num(nse_mod.pick(row, "pChange")),
                open_interest=nse_mod.num(nse_mod.pick(row, "openInterest")),
                volume=nse_mod.num(nse_mod.pick(row, "volume")),
                underlying_value=nse_mod.num(nse_mod.pick(row, "underlyingValue")),
                prov=Provenance(self.source_oi, ident, "snapshot", stamp_dt, fetched)))
        return out

    # --- candles --------------------------------------------------------

    def candles(self, symbols: list[str]) -> dict[str, Candles]:
        import yfinance as yf

        fetched = datetime.now(IST)
        tickers = [s + ".NS" for s in symbols]
        try:
            raw = yf.download(tickers=tickers, period=self.period,
                              interval=C.BAR_INTERVAL, group_by="ticker",
                              threads=False, progress=False, auto_adjust=True)
        except Exception as exc:                      # noqa: BLE001
            self.errors.append(f"candle download failed: {exc}")
            return {}
        out: dict[str, Candles] = {}
        for sym in symbols:
            df = self._clean(_extract(raw, sym + ".NS"))
            if df is None:
                continue
            out[sym] = Candles(df, Provenance(
                self.source_candles, sym, C.BAR_INTERVAL, _bar_end(df), fetched,
                delayed=True))
        return out

    def index_candles(self, ticker: str) -> Candles | None:
        import yfinance as yf

        fetched = datetime.now(IST)
        try:
            df = yf.Ticker(ticker).history(period=self.period,
                                           interval=C.BAR_INTERVAL)
        except Exception as exc:                      # noqa: BLE001
            self.errors.append(f"{ticker}: index candles failed ({exc})")
            return None
        df = self._clean(df)
        if df is None:
            self.errors.append(f"{ticker}: no index candles returned")
            return None
        return Candles(df, Provenance(self.source_candles, ticker, C.BAR_INTERVAL,
                                      _bar_end(df), fetched, delayed=True))

    @staticmethod
    def _clean(df) -> pd.DataFrame | None:
        if df is None or len(df) == 0 or "Close" not in df.columns:
            return None
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.tz_convert(IST)
        # yfinance stamps a bar with its START time, so the newest row is often
        # still forming. Drop it here: the analysis would discard it anyway, and
        # keeping it made the feed's reported age negative — a freshness claim
        # the data does not support.
        end = df.index[-1] + pd.Timedelta(minutes=C.BAR_MINUTES)
        if end > pd.Timestamp(datetime.now(IST)):
            df = df.iloc[:-1]
        return df if not df.empty else None


class ReplayFeed:
    """Feed backed by a saved JSON bundle — no network, fully deterministic."""

    def __init__(self, path: str | dict):
        if isinstance(path, dict):
            self.bundle, self.name = path, "inline"
        else:
            with open(path) as fh:
                self.bundle = json.load(fh)
            # The basename, not the path — this string is a source label that
            # gets printed in reports and shown in the UI's feed panel.
            self.name = os.path.basename(path)
        self.errors: list[str] = []
        self.as_of = datetime.fromisoformat(self.bundle["as_of"])
        self.source = f"replay:{self.name}"

    def universe(self) -> list[dict]:
        return [{"symbol": s, "pct": v.get("pct"), "volume": v.get("volume"),
                 "value_cr": v.get("value_cr")}
                for s, v in self.bundle["stocks"].items()]

    def futures_board(self) -> dict[str, FuturesSnapshot]:
        out = {}
        for sym, v in self.bundle["stocks"].items():
            f = v.get("futures")
            if not f:
                continue
            out[sym] = FuturesSnapshot(
                symbol=sym, contract=f.get("contract", f"{sym}-FUT"),
                expiry=f.get("expiry"), last_price=f["last_price"],
                prev_close=f.get("prev_close"), pct_change=f.get("pct_change"),
                open_interest=f.get("open_interest"),
                oi_change_pct=f.get("oi_change_pct"), volume=f.get("volume"),
                turnover_cr=f.get("turnover_cr"), bid=f.get("bid"), ask=f.get("ask"),
                prov=Provenance(self.source, f.get("contract", sym), "snapshot",
                                self.as_of, self.as_of))
        return out

    def futures_detail(self, symbol, fallback=None):
        return self.futures_board().get(symbol, fallback)

    def candles(self, symbols: list[str]) -> dict[str, Candles]:
        out = {}
        for sym in symbols:
            rows = (self.bundle["stocks"].get(sym) or {}).get("candles")
            if not rows:
                continue
            out[sym] = Candles(_frame(rows), Provenance(
                self.source, sym, C.BAR_INTERVAL, self.as_of, self.as_of))
        return out

    def option_board(self) -> list:
        from trading.fno.options import OptionQuote

        out = []
        for row in self.bundle.get("options") or []:
            ident = row.get("identifier", f"{row['underlying']}-{row['option_type']}")
            out.append(OptionQuote(
                underlying=row["underlying"], identifier=ident,
                option_type=row["option_type"], strike=row["strike"],
                expiry=row.get("expiry"), last_price=row["last_price"],
                pct_change=row.get("pct_change"), open_interest=row.get("open_interest"),
                volume=row.get("volume"), underlying_value=row.get("underlying_value"),
                prov=Provenance(self.source, ident, "snapshot", self.as_of, self.as_of)))
        return out

    def index_option_board(self) -> list:
        """Replay bundles keep every contract in one list; the planner filters
        by underlying, so the index chain is served from the same place."""
        return self.option_board()

    def index_candles(self, ticker: str) -> Candles | None:
        rows = (self.bundle.get("indices") or {}).get(ticker)
        if not rows:
            return None
        return Candles(_frame(rows), Provenance(
            self.source, ticker, C.BAR_INTERVAL, self.as_of, self.as_of))


def _extract(raw, ticker: str):
    """Pull one ticker's OHLCV out of a yfinance download.

    yfinance returns MultiIndex columns even for a single ticker, and which
    level holds the ticker depends on group_by — so both are checked instead of
    assuming the multi-symbol shape.
    """
    if raw is None or len(raw) == 0:
        return None
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    for level in (0, 1):
        if ticker in raw.columns.get_level_values(level):
            return raw.xs(ticker, axis=1, level=level)
    return None


def _frame(rows) -> pd.DataFrame:
    """[[iso_ts, o, h, l, c, v], ...] → an IST-indexed OHLCV frame."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows],
         "Volume": [float(r[5]) for r in rows]}, index=idx.tz_convert(IST))


def synthetic_bundle(*, as_of: datetime, stocks: dict, indices: dict | None = None) -> dict:
    """Helper used by the tests to build replay bundles inline."""
    return {"as_of": as_of.isoformat(), "stocks": stocks, "indices": indices or {}}
