"""K 線完整性、時間戳與新鮮度檢查。假設傳入的 DataFrame 以 UTC 開盤時間為 index。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..timeutil import index_to_ms, now_utc, tf_delta, tf_ms

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class DataQuality:
    ok: bool
    issues: list = field(default_factory=list)      # 會阻擋產生訊號
    warnings: list = field(default_factory=list)
    n_rows: int = 0
    n_missing_bars: int = 0
    n_duplicates: int = 0
    n_invalid: int = 0
    last_bar_open: Optional[pd.Timestamp] = None
    age_seconds: Optional[float] = None
    fresh: bool = False


def drop_incomplete_last_bar(df: pd.DataFrame, timeframe: str, now=None) -> pd.DataFrame:
    """移除仍在形成中的最後一根 K 線（防止用未收盤資料產生訊號）。"""
    if df.empty:
        return df
    now = pd.Timestamp(now) if now is not None else now_utc()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    if df.index[-1] + tf_delta(timeframe) > now:
        return df.iloc[:-1]
    return df


def validate_ohlcv(df: pd.DataFrame, timeframe: str, now=None, max_age_bars: float = 2.5,
                   check_freshness: bool = True) -> DataQuality:
    q = DataQuality(ok=False, n_rows=len(df))
    if df is None or df.empty:
        q.issues.append("沒有任何 K 線資料")
        return q
    missing_cols = [c for c in OHLCV if c not in df.columns]
    if missing_cols:
        q.issues.append(f"缺少欄位：{missing_cols}")
        return q
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        q.issues.append("時間索引必須為含時區的 DatetimeIndex（UTC）")
        return q
    q.n_duplicates = int(df.index.duplicated().sum())
    if q.n_duplicates:
        q.issues.append(f"有 {q.n_duplicates} 筆重複時間戳")
    if not df.index.is_monotonic_increasing:
        q.issues.append("時間戳未按先後排序")
    vals = df[OHLCV].astype(float)
    nan_n = int(vals.isna().any(axis=1).sum())
    pos = vals[["open", "high", "low", "close"]]
    bad = (
        vals.isna().any(axis=1)
        | (pos <= 0).any(axis=1)
        | (vals["high"] < vals[["open", "close"]].max(axis=1) - 1e-12)
        | (vals["low"] > vals[["open", "close"]].min(axis=1) + 1e-12)
        | (vals["high"] < vals["low"])
        | (vals["volume"] < 0)
    )
    q.n_invalid = int(bad.sum())
    if q.n_invalid:
        q.issues.append(f"有 {q.n_invalid} 根 K 線 OHLCV 不合理（NaN {nan_n} 根、價格<=0、高低點矛盾或成交量<0）")
    # 缺口
    if len(df) > 1 and not q.n_duplicates and df.index.is_monotonic_increasing:
        step = tf_ms(timeframe)
        d = np.diff(index_to_ms(df.index))
        miss = np.maximum(d // step - 1, 0)
        q.n_missing_bars = int(miss.sum())
        if q.n_missing_bars:
            recent_gap = bool(miss[-200:].sum() > 0)
            msg = f"缺少 {q.n_missing_bars} 根 K 線"
            if recent_gap:
                q.issues.append(msg + "（近 200 根內有缺口，會影響指標）")
            else:
                q.warnings.append(msg + "（位於較早期）")
    q.last_bar_open = df.index[-1]
    now = pd.Timestamp(now) if now is not None else now_utc()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    q.age_seconds = float((now - (df.index[-1] + tf_delta(timeframe))).total_seconds())
    q.fresh = q.age_seconds <= max_age_bars * tf_delta(timeframe).total_seconds()
    if q.age_seconds < -1:
        q.issues.append("最後一根 K 線尚未收盤或時間在未來（請先移除形成中的 K 線）")
    if check_freshness and not q.fresh:
        q.issues.append(f"行情資料過期：最後一根收盤已過 {q.age_seconds/60:.1f} 分鐘")
    q.ok = not q.issues
    return q
