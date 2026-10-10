"""時間工具：全系統內部一律使用 UTC，顯示時才轉為 Asia/Taipei。"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TAIPEI = ZoneInfo("Asia/Taipei")
TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
EIGHT_HOURS_MS = 8 * 3600 * 1000
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def tf_seconds(tf: str) -> int:
    try:
        return TIMEFRAME_SECONDS[tf]
    except KeyError:
        raise ValueError(f"不支援的週期：{tf}（支援 {list(TIMEFRAME_SECONDS)}）") from None


def tf_ms(tf: str) -> int:
    return tf_seconds(tf) * 1000


def tf_delta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(seconds=tf_seconds(tf))


def to_ms(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int((t - _EPOCH) // pd.Timedelta(milliseconds=1))


def index_to_ms(idx) -> np.ndarray:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return np.asarray((idx - _EPOCH) // pd.Timedelta(milliseconds=1), dtype="int64")


def from_ms(ms) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ms, dtype="int64"), unit="ms", utc=True))


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def taipei_day_key(ts_ms):
    """台灣時區的『日』編號（用於單日損失計算）。"""
    return (np.asarray(ts_ms, dtype="int64") + 8 * 3600 * 1000) // 86_400_000


def fmt_taipei(ts) -> str:
    if ts is None:
        return "—"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TAIPEI).strftime("%Y-%m-%d %H:%M:%S (台北)")
