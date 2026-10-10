"""合成『測試資料』產生器：僅供離線測試與功能示範，絕不可當作真實行情。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutil import from_ms, tf_ms, to_ms, now_utc

# (名稱, 每根漂移, 每根波動, 均值回歸強度)
_REGIMES = [
    ("up", 0.00030, 0.0008, 0.0),
    ("down", -0.00030, 0.0008, 0.0),
    ("range", 0.0, 0.0006, 0.08),
    ("range", 0.0, 0.0006, 0.08),
    ("wild", 0.0, 0.0020, 0.0),
]


def generate_synthetic_ohlcv(n: int, timeframe: str, seed: int = 7, start_price: float = 60_000.0,
                             end=None) -> pd.DataFrame:
    """產生 n 根『已收盤』合成 K 線；最後一根於 end（預設現在）之前收盤。決定性（同 seed 同結果）。"""
    rng = np.random.default_rng(seed)
    step = tf_ms(timeframe)
    end_ms = to_ms(end) if end is not None else to_ms(now_utc())
    last_open = (end_ms // step) * step - step
    opens_ms = last_open - step * np.arange(n - 1, -1, -1, dtype="int64")

    rets = np.empty(n)
    vol_mult = np.empty(n)
    i = 0
    anchor = 0.0
    x = 0.0  # 對數價相對位置（供均值回歸）
    while i < n:
        name, mu, sg, kappa = _REGIMES[rng.integers(0, len(_REGIMES))]
        seg = int(rng.integers(120, 400))
        for _ in range(min(seg, n - i)):
            shock = rng.standard_normal() * sg
            r = mu + shock - kappa * (x - anchor)
            x += r
            rets[i] = r
            vol_mult[i] = 1.0 + 4.0 * abs(shock) / max(sg, 1e-9) * 0.25
            i += 1
        anchor = x
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.standard_normal(n)) * np.abs(rets).clip(min=1e-4) * 0.8
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=n) * vol_mult
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                      index=from_ms(opens_ms))
    df.index.name = "ts"
    return df


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """把較小週期 K 線聚合成較大週期，只保留『完整』的大週期 K 線（用於讓測試資料的多週期彼此一致）。"""
    if df.empty:
        return df
    base_ms = int(np.median(np.diff(df.index.asi8 // 1_000_000))) if len(df) > 1 else tf_ms(target_tf)
    per = tf_ms(target_tf) // base_ms
    rule = f"{tf_ms(target_tf) // 60_000}min"
    g = df.resample(rule, closed="left", label="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    cnt = g["close"].count()
    return out[cnt == per].dropna()
