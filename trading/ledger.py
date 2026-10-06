"""SQLite ledger — the single source of truth for trades, signals, and agent audit logs.

Tables:
  trades     — every paper/live trade with entry/exit, P&L, MAE/MFE, and the
               supervisor agent's verdict (written asynchronously, never blocking).
  rejections — signals the risk kernel or rate limiter refused.
  agent_log  — full prompt + response for every LLM call (the audit trail).
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from trading.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    date            TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    qty             INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    stop_loss       REAL,
    initial_stop    REAL,
    target          REAL,
    exit_price      REAL,
    exit_ts         REAL,
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    pnl             REAL,
    charges         REAL,
    mae             REAL,
    mfe             REAL,
    strategy_id     TEXT,
    regime          TEXT,
    mode            TEXT NOT NULL DEFAULT 'paper',
    agent_verdict   TEXT,
    agent_confidence REAL,
    agent_reasons   TEXT,
    agent_reviewed_at REAL
);

CREATE TABLE IF NOT EXISTS rejections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    reason      TEXT NOT NULL,
    signal_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    agent       TEXT NOT NULL,
    model       TEXT,
    prompt      TEXT,
    response    TEXT,
    ok          INTEGER NOT NULL,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS candidate_signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  REAL NOT NULL,
    date                TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price               REAL NOT NULL,
    stop_loss           REAL NOT NULL,
    target              REAL NOT NULL,
    strategy_id         TEXT NOT NULL,
    trend_state         TEXT,
    gate_passed         INTEGER NOT NULL DEFAULT 1,
    council_action      TEXT,
    council_conviction  INTEGER,
    council_reasons     TEXT,
    facts_json          TEXT,
    executed            INTEGER NOT NULL DEFAULT 0,
    trade_id            INTEGER,
    rejection_reason    TEXT,
    shadow_status       TEXT DEFAULT 'open',
    shadow_exit_price   REAL,
    shadow_exit_ts      REAL,
    shadow_pnl          REAL
);
"""



# IST, to match execution_router.load_day_config() and the pre-market agent.
# Every "today"-keyed row — trades, rejections, the day's realized P&L — has to
# agree on when today started. On a machine already set to IST this is a no-op;
# anywhere else it is the difference between the router's day and the ledger's
# day being the same day.
IST = ZoneInfo("Asia/Kolkata")


def _today() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


class Ledger:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn) -> None:
        """Additive column migrations, applied to existing databases.

        ``initial_stop`` is 1R's anchor for the trailing stop. ``stop_loss`` cannot serve
        that role once the trail has moved it, and a restart mid-position re-syncs open
        trades from this table -- without the original stop the engine would re-derive 1R
        from an already-trailed stop and arm the trail at the wrong level. Nullable, so
        rows written before this column existed fall back to their structural stop.
        """
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(trades)")}
        if "initial_stop" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN initial_stop REAL")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- trades ---

    def record_entry(self, signal: dict, fill_price: float, mode: str = "paper") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (ts, date, symbol, side, qty, entry_price, stop_loss, initial_stop,
                    target, strategy_id, regime, mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.get("ts", time.time()),
                    _today(),
                    signal["symbol"],
                    signal["side"],
                    signal["qty"],
                    fill_price,
                    signal.get("stop_loss"),
                    signal.get("initial_stop", signal.get("stop_loss")),
                    signal.get("target"),
                    signal.get("strategy_id"),
                    signal.get("regime"),
                    mode,
                ),
            )
            return cur.lastrowid

    def update_open_stop(self, trade_id: int, new_stop: float) -> None:
        """Persist a trailing stop that has moved.

        ``initial_stop`` is deliberately NOT touched -- it stays 1R's anchor. Only the
        working stop moves, so a restart re-syncs the engine to the stop it was actually
        using instead of silently reverting to the structural one.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE trades SET stop_loss = ? WHERE id = ? AND status = 'open'",
                (new_stop, trade_id),
            )

    def record_exit(self, trade_id: int, exit_price: float, charges: float,
                    mae: float | None = None, mfe: float | None = None):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT side, qty, entry_price FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"trade {trade_id} not found")
            direction = 1 if row["side"] == "BUY" else -1
            gross = direction * (exit_price - row["entry_price"]) * row["qty"]
            pnl = gross - charges
            conn.execute(
                """UPDATE trades SET exit_price = ?, exit_ts = ?, status = 'closed',
                   pnl = ?, charges = ?, mae = ?, mfe = ? WHERE id = ?""",
                (exit_price, time.time(), pnl, charges, mae, mfe, trade_id),
            )
            return pnl

    def reduce_open_qty(self, trade_id: int, new_qty: int) -> None:
        """Shrink an open trade's quantity, for a partial profit booking.

        A partial booking is recorded as a child trade (entry + exit at the booking price),
        so the parent row must stop claiming those shares or the final exit would count them
        twice. ``record_exit`` sizes gross P&L from the row's stored qty, so this is the one
        place that has to move. Rejects a no-op or a non-positive quantity: a partial must
        always leave something open, and this is not the path that closes a position.
        """
        if new_qty < 1:
            raise ValueError("partial booking must leave at least 1 share open")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT qty, status FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"trade {trade_id} not found")
            if row["status"] != "open":
                raise ValueError(f"trade {trade_id} is not open")
            if new_qty >= row["qty"]:
                raise ValueError(
                    f"new qty {new_qty} must be smaller than current qty {row['qty']}"
                )
            conn.execute("UPDATE trades SET qty = ? WHERE id = ?", (new_qty, trade_id))

    def record_rejection(self, signal: dict, reason: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO rejections (ts, reason, signal_json) VALUES (?, ?, ?)",
                (time.time(), reason, json.dumps(signal, default=str)),
            )

    def recent_rejections(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM rejections ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def day_realized_pnl(self, date: str | None = None) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS pnl FROM trades "
                "WHERE date = ? AND status = 'closed'",
                (date or _today(),),
            ).fetchone()
            return row["pnl"]

    def open_position_value(self, symbol: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(qty * entry_price), 0) AS v FROM trades "
                "WHERE symbol = ? AND status = 'open'",
                (symbol,),
            ).fetchone()
            return row["v"]

    def trades_for_date(self, date: str | None = None) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE date = ? ORDER BY ts", (date or _today(),)
            ).fetchall()
            return [dict(r) for r in rows]

    def unreviewed_trades(self, limit: int = 10) -> list[dict]:
        """Trades the supervisor agent has not yet reviewed."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE agent_verdict IS NULL ORDER BY ts LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_agent_verdict(self, trade_id: int, verdict: str,
                           confidence: float, reasons: list[str]):
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades SET agent_verdict = ?, agent_confidence = ?,
                   agent_reasons = ?, agent_reviewed_at = ? WHERE id = ?""",
                (verdict, confidence, json.dumps(reasons), time.time(), trade_id),
            )

    def open_trades(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY ts").fetchall()
            return [dict(r) for r in rows]

    def recent_closed_trades(self, n: int = 5) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_ts DESC LIMIT ?",
                (n,),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- agent audit log ---

    def log_agent_call(self, agent: str, model: str, prompt: str,
                       response: str | None, ok: bool, error: str | None = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_log (ts, agent, model, prompt, response, ok, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), agent, model, prompt, response, int(ok), error),
            )

    # --- candidate signals & shadow P&L (council audit) ---

    def record_candidate(self, candidate: dict) -> int:
        """Log a candidate setup entering the council pipeline."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO candidate_signals
                   (ts, date, symbol, side, price, stop_loss, target,
                    strategy_id, trend_state, gate_passed, council_action,
                    council_conviction, council_reasons, facts_json, executed,
                    trade_id, rejection_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.get("ts", time.time()),
                    candidate.get("date", _today()),
                    candidate["symbol"],
                    candidate["side"],
                    candidate["price"],
                    candidate["stop_loss"],
                    candidate["target"],
                    candidate.get("strategy_id", "orb_ma200"),
                    candidate.get("trend_state"),
                    1 if candidate.get("gate_passed", True) else 0,
                    candidate.get("council_action"),
                    candidate.get("council_conviction"),
                    candidate.get("council_reasons"),
                    json.dumps(candidate.get("facts_json")) if isinstance(candidate.get("facts_json"), (dict, list)) else candidate.get("facts_json"),
                    1 if candidate.get("executed", False) else 0,
                    candidate.get("trade_id"),
                    candidate.get("rejection_reason"),
                ),
            )
            return int(cur.lastrowid)

    def update_candidate_decision(self, candidate_id: int, decision: dict) -> None:
        """Update a candidate record with council verdict and execution outcome."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE candidate_signals
                   SET council_action = ?, council_conviction = ?, council_reasons = ?,
                       executed = ?, trade_id = ?, rejection_reason = ?
                   WHERE id = ?""",
                (
                    decision.get("action"),
                    decision.get("conviction"),
                    decision.get("reasons"),
                    1 if decision.get("executed") else 0,
                    decision.get("trade_id"),
                    decision.get("rejection_reason"),
                    candidate_id,
                ),
            )

    def record_shadow_exit(
        self,
        candidate_id: int,
        exit_price: float,
        exit_ts: float | None = None,
        shadow_pnl: float | None = None,
    ) -> float:
        """Record hypothetical exit for a candidate to compute shadow P&L."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM candidate_signals WHERE id = ?", (candidate_id,)).fetchone()
            if not row:
                return 0.0
            r = dict(row)
            entry = r["price"]
            side = r["side"]
            if shadow_pnl is not None:
                pnl = shadow_pnl
            else:
                pnl = (exit_price - entry) if side == "BUY" else (entry - exit_price)
            ts = exit_ts or time.time()
            conn.execute(
                """UPDATE candidate_signals
                   SET shadow_status = 'closed', shadow_exit_price = ?, shadow_exit_ts = ?, shadow_pnl = ?
                   WHERE id = ?""",
                (exit_price, ts, round(pnl, 2), candidate_id),
            )
            return round(pnl, 2)

    def candidate_signals_for_date(self, date: str | None = None) -> list[dict]:
        d = date or _today()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_signals WHERE date = ? ORDER BY ts ASC",
                (d,),
            ).fetchall()
            return [dict(r) for r in rows]

    def open_shadow_candidates(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_signals WHERE shadow_status = 'open' ORDER BY ts ASC"
            ).fetchall()
            return [dict(r) for r in rows]
