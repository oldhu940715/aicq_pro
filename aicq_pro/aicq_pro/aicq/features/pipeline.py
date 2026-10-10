"""特徵管線：基準週期特徵 + 高週期（1h）趨勢對齊（避免前視偏差）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyParams
from ..timeutil import index_to_ms, tf_ms
from . import indicators as ind


def compute_htf_trend(htf_df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    """高週期趨勢：+1 多頭、-1 空頭、0 不明。每個值代表該根 K 線『收盤後』才知道的結果。"""
    c = htf_df["close"]
    f, s = ind.ema(c, p.htf_fast), ind.ema(c, p.htf_slow)
    out = pd.Series(0, index=htf_df.index, dtype="int64")
    out[(f > s) & (c > s)] = 1
    out[(f < s) & (c < s)] = -1
    return out


def align_htf(base_index: pd.DatetimeIndex, base_tf: str, htf: pd.Series, htf_tf: str) -> pd.Series:
    """對每根基準 K 線（在其收盤時決策），取『已收盤』的最新高週期值。

    高週期 K 線開盤於 T，須到 T+htf_tf 才算完成；基準 K 線決策時間為其開盤+base_tf。
    """
    if htf is None or len(htf) == 0:
        return pd.Series(0, index=base_index, dtype="int64")
    lk = index_to_ms(base_index) + tf_ms(base_tf)
    order = np.argsort(index_to_ms(htf.index))
    rk = index_to_ms(htf.index)[order] + tf_ms(htf_tf)
    vals = htf.to_numpy()[order]
    pos = np.searchsorted(rk, lk, side="right") - 1
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], 0)
    return pd.Series(out.astype("int64"), index=base_index)


def add_features(df: pd.DataFrame, p: StrategyParams, htf_aligned: pd.Series | None = None) -> pd.DataFrame:
    f = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    c, h, l, v = f["close"], f["high"], f["low"], f["volume"]
    f["ema_fast"] = ind.ema(c, p.ema_fast)
    f["ema_slow"] = ind.ema(c, p.ema_slow)
    f["atr"] = ind.atr(f, p.atr_period)
    f["atr_pct"] = f["atr"] / c
    f["adx"] = ind.adx(f, p.adx_period)
    f["rsi"] = ind.rsi(c, p.rsi_period)
    f["bb_mid"], f["bb_up"], f["bb_lo"], f["bb_z"] = ind.bollinger(c, p.bb_period, p.bb_k)
    f["vwap"] = ind.session_vwap(f)
    f["vol_ma20"] = v.rolling(20, min_periods=20).mean()
    f["vol_ratio"] = v / f["vol_ma20"].shift(1).replace(0, np.nan)      # 對比『之前』20 根均量
    f["vol_med200"] = v.rolling(200, min_periods=50).median().shift(1)
    f["don_hi"] = h.rolling(p.breakout_lookback, min_periods=p.breakout_lookback).max().shift(1)
    f["don_lo"] = l.rolling(p.breakout_lookback, min_periods=p.breakout_lookback).min().shift(1)
    f["atr_ma"] = f["atr"].rolling(20, min_periods=20).mean()
    f["atr_pct_q"] = f["atr_pct"].rolling(200, min_periods=50).quantile(p.high_vol_quantile).shift(1)
    f["close_loc"] = (c - l) / (h - l).replace(0, np.nan)
    f["htf"] = htf_aligned.reindex(f.index).fillna(0).astype("int64") if htf_aligned is not None else 0
    return f
