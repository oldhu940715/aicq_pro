"""市場狀態判斷，並決定各狀態啟用哪些策略（保守：高波動與低流動性一律觀望）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyParams

REGIME_LABELS = {
    "trend": "趨勢",
    "range": "盤整",
    "mixed": "過渡（方向不明）",
    "high_vol": "高波動",
    "low_liquidity": "低流動性",
    "warmup": "指標暖機中",
    "unknown": "未知",
}

STRATEGIES_BY_REGIME = {
    "trend": ["trend"],
    "range": ["mean_reversion", "breakout"],
    "mixed": ["breakout"],
    "high_vol": [],
    "low_liquidity": [],
    "warmup": [],
}


def classify_regime(f: pd.DataFrame, p: StrategyParams) -> pd.Series:
    warm = f[["adx", "atr", "vol_ma20", "atr_pct_q", "ema_slow"]].isna().any(axis=1)
    low_liq = (f["vol_ma20"] <= 0) | (f["vol_ma20"] < p.low_liq_frac * f["vol_med200"])
    high_vol = f["atr_pct"] > f["atr_pct_q"]
    trend = f["adx"] >= p.adx_trend
    rng = f["adx"] < p.adx_range
    out = np.select([warm, low_liq, high_vol, trend, rng],
                    ["warmup", "low_liquidity", "high_vol", "trend", "range"], default="mixed")
    return pd.Series(out.astype(object), index=f.index, dtype=object)
