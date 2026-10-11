"""SQLite 儲存：訊號、模擬持倉／交易、淨值曲線、事件、回測紀錄、狀態值。"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..data.cache import _Ctx

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals(
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, signal_key TEXT, symbol TEXT, timeframe TEXT,
 direction TEXT, candidate_direction TEXT, strategy TEXT, regime TEXT, data_source TEXT, payload TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_signal_key ON signals(signal_key);
CREATE TABLE IF NOT EXISTS positions(
 id INTEGER PRIMARY KEY AUTOINCREMENT, signal_key TEXT UNIQUE, symbol TEXT, timeframe TEXT, side INTEGER,
 strategy TEXT, qty REAL, entry_price REAL, stop REAL, tp1 REAL, notional REAL, margin REAL,
 risk_amount REAL, entry_fee REAL, opened_at TEXT, opened_ms INTEGER, eval_from_ms INTEGER,
 max_hold_bars INTEGER, status TEXT, exit_price REAL, exit_time TEXT, exit_reason TEXT, gross_pnl REAL,
 fees REAL, funding REAL, slippage_cost REAL, net_pnl REAL, reason TEXT, data_source TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_open_symbol ON positions(symbol) WHERE status='open';
CREATE TABLE IF NOT EXISTS equity_points(ts TEXT, equity REAL);
CREATE TABLE IF NOT EXISTS events(ts TEXT, level TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS backtest_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT,
 params TEXT, metrics TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PositionLimitReached(Exception):
    """已達最大持倉數（在資料庫交易內原子判斷，避免多個瀏覽器分頁同時下單造成超限）。"""


class Storage:
    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    def _c(self):
        return _Ctx(sqlite3.connect(self.path, timeout=30))

    # -- kv / events ----------------------------------------------------
    def get(self, key, default=None):
        with self._c() as c:
            r = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, key, value):
        with self._c() as c:
            c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))

    def log(self, level: str, message: str):
        with self._c() as c:
            c.execute("INSERT INTO events VALUES (?,?,?)", (_now(), level, message))

    def events(self, limit=100) -> pd.DataFrame:
        with self._c() as c:
            return pd.read_sql_query("SELECT * FROM events ORDER BY ts DESC LIMIT ?", c, params=(limit,))

    # -- signals --------------------------------------------------------
    def save_signal(self, sig) -> bool:
        """回傳 True=新增；False=同一根 K 線同策略同方向的訊號已存在（去重）。"""
        d = sig.to_dict()
        with self._c() as c:
            try:
                c.execute("INSERT INTO signals(created_at,signal_key,symbol,timeframe,direction,candidate_direction,"
                          "strategy,regime,data_source,payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (_now(), sig.signal_key or f"flat|{sig.symbol}|{sig.timeframe}|{sig.bar_time}",
                           sig.symbol, sig.timeframe, sig.direction, sig.candidate_direction, sig.strategy,
                           sig.regime, sig.data_source, json.dumps(d, default=str, ensure_ascii=False)))
                return True
            except sqlite3.IntegrityError:
                return False

    def signals(self, limit=100) -> pd.DataFrame:
        with self._c() as c:
            return pd.read_sql_query("SELECT id,created_at,symbol,timeframe,direction,candidate_direction,strategy,"
                                     "regime,data_source FROM signals ORDER BY id DESC LIMIT ?", c, params=(limit,))

    # -- positions ------------------------------------------------------
    def insert_position(self, rec: dict, max_open: int | None = None) -> int:
        """原子新增持倉。保證：同 signal_key 不重複（UNIQUE）、同商品最多一筆未平倉（部分唯一索引）、
        未平倉總數不超過 max_open（同一個 SQL 陳述式內判斷）。違反時丟 sqlite3.IntegrityError／PositionLimitReached。"""
        cols = ",".join(rec)
        q = ",".join("?" * len(rec))
        args = list(rec.values())
        if max_open is None:
            sql = f"INSERT INTO positions({cols}) VALUES ({q})"
        else:
            sql = (f"INSERT INTO positions({cols}) SELECT {q} "
                   "WHERE (SELECT COUNT(*) FROM positions WHERE status='open') < ?")
            args.append(int(max_open))
        with self._c() as c:
            cur = c.execute(sql, args)
            if cur.rowcount == 0:
                raise PositionLimitReached(f"未平倉數已達上限 {max_open}")
            return cur.lastrowid

    def positions(self, status: str | None = None) -> pd.DataFrame:
        q, a = "SELECT * FROM positions", ()
        if status:
            q, a = q + " WHERE status=?", (status,)
        with self._c() as c:
            return pd.read_sql_query(q + " ORDER BY id", c, params=a)

    def close_position(self, pid: int, **fields):
        sets = ",".join(f"{k}=?" for k in fields)
        with self._c() as c:
            c.execute(f"UPDATE positions SET {sets}, status='closed' WHERE id=? AND status='open'",
                      (*fields.values(), pid))

    def executed_keys(self) -> set:
        with self._c() as c:
            return {r[0] for r in c.execute("SELECT signal_key FROM positions")}

    # -- equity / backtests --------------------------------------------
    def add_equity_point(self, equity: float):
        with self._c() as c:
            c.execute("INSERT INTO equity_points VALUES (?,?)", (_now(), float(equity)))

    def equity_points(self) -> pd.DataFrame:
        with self._c() as c:
            return pd.read_sql_query("SELECT * FROM equity_points ORDER BY ts", c)

    def save_backtest(self, kind: str, params: dict, metrics: dict) -> int:
        with self._c() as c:
            cur = c.execute("INSERT INTO backtest_runs(created_at,kind,params,metrics) VALUES (?,?,?,?)",
                            (_now(), kind, json.dumps(params, default=str, ensure_ascii=False),
                             json.dumps(metrics, default=str, ensure_ascii=False)))
            return cur.lastrowid

    def backtest_runs(self, limit: int = 50) -> pd.DataFrame:
        with self._c() as c:
            return pd.read_sql_query("SELECT id,created_at,kind,params,metrics FROM backtest_runs ORDER BY id DESC LIMIT ?",
                                     c, params=(limit,))

    def count_backtests(self) -> int:
        with self._c() as c:
            return int(c.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])
