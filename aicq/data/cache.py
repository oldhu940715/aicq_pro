"""OHLCV 本機快取（SQLite）。只存真實行情；合成測試資料不寫入快取。"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd

from ..timeutil import from_ms, index_to_ms

_SCHEMA = """CREATE TABLE IF NOT EXISTS ohlcv(
 exchange TEXT, symbol TEXT, timeframe TEXT, ts_ms INTEGER,
 open REAL, high REAL, low REAL, close REAL, volume REAL,
 PRIMARY KEY(exchange, symbol, timeframe, ts_ms))"""


class OhlcvCache:
    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with self._c() as c:
            c.execute(_SCHEMA)

    def _c(self):
        conn = sqlite3.connect(self.path, timeout=30)
        return _Ctx(conn)

    def upsert(self, exchange: str, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        ms = index_to_ms(df.index)
        rows = [(exchange, symbol, timeframe, int(t), float(o), float(h), float(l), float(c), float(v))
                for t, o, h, l, c, v in zip(ms, df["open"], df["high"], df["low"], df["close"], df["volume"])]
        with self._c() as c:
            c.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def load(self, exchange: str, symbol: str, timeframe: str, since_ms: int | None = None,
             limit: int | None = None) -> pd.DataFrame:
        q = "SELECT ts_ms,open,high,low,close,volume FROM ohlcv WHERE exchange=? AND symbol=? AND timeframe=?"
        args: list = [exchange, symbol, timeframe]
        if since_ms is not None:
            q += " AND ts_ms>=?"
            args.append(int(since_ms))
        q += " ORDER BY ts_ms"
        with self._c() as c:
            rows = c.execute(q, args).fetchall()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                                index=pd.DatetimeIndex([], tz="UTC"))
        df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df.index = from_ms(df.pop("ts_ms").to_numpy())
        df.index.name = "ts"
        return df.iloc[-limit:] if limit else df


class _Ctx:
    """sqlite3 連線 context：離開時 commit 並確實 close。"""
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, et, ev, tb):
        try:
            self.conn.commit() if et is None else self.conn.rollback()
        finally:
            self.conn.close()
