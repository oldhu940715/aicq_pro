"""策略：趨勢、突破、均值回歸。全部向量化、只用當根及過去資料；訊號在 K 線收盤時產生，
回測／模擬皆於『下一根』才可能成交。不假設任何策略一定有效。

證據評分：把高度相關的指標歸成『一組』只算一票（例如 EMA/ADX/VWAP 同屬趨勢結構），
共 4 組獨立證據，數字為「成立組數」，不是獲勝機率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyParams
from ..features.pipeline import add_features, align_htf, compute_htf_trend
from .regime import STRATEGIES_BY_REGIME, classify_regime

STRATEGY_LABELS = {"trend": "趨勢順勢", "breakout": "區間突破", "mean_reversion": "均值回歸"}
PRIORITY = ["trend", "breakout", "mean_reversion"]

EVIDENCE_NAMES = {
    "trend": ["趨勢結構(EMA/ADX/VWAP視為一組)", "高週期1h趨勢同向", "成交量放大", "波動擴張(ATR>均值)"],
    "breakout": ["趨勢結構(EMA/ADX/VWAP視為一組)", "高週期1h趨勢同向", "成交量放大", "波動擴張(ATR>均值)"],
    "mean_reversion": ["價格過度偏離(布林/RSI視為一組)", "高週期未反向", "市場為盤整", "無爆量(非恐慌/追價)"],
}


def _sig(long: pd.Series, short: pd.Series) -> np.ndarray:
    return np.where(long.fillna(False), 1, np.where(short.fillna(False), -1, 0)).astype("int64")


def _trend(f, p):
    long = (f.ema_fast > f.ema_slow) & (f.close > f.vwap) & (f.adx >= p.adx_trend) & (f.htf >= 0) \
        & (f.close > f.ema_fast) & (f.close.shift(1) <= f.ema_fast.shift(1))
    short = (f.ema_fast < f.ema_slow) & (f.close < f.vwap) & (f.adx >= p.adx_trend) & (f.htf <= 0) \
        & (f.close < f.ema_fast) & (f.close.shift(1) >= f.ema_fast.shift(1))
    return _sig(long, short)


def _breakout(f, p):
    common = (f.vol_ratio >= p.breakout_vol_ratio) & (f.atr > f.atr_ma)
    long = common & (f.close > f.don_hi) & (f.close_loc >= 0.7) & (f.htf >= 0)
    short = common & (f.close < f.don_lo) & (f.close_loc <= 0.3) & (f.htf <= 0)
    return _sig(long, short)


def _mean_rev(f, p):
    long = (f.bb_z <= -p.bb_z_extreme) & (f.rsi <= p.rsi_low)
    short = (f.bb_z >= p.bb_z_extreme) & (f.rsi >= p.rsi_high)
    return _sig(long, short)


def _evidence(name: str, f: pd.DataFrame, s: np.ndarray, p: StrategyParams) -> np.ndarray:
    """回傳 shape (n,4) 布林矩陣；s 為 ±1 方向。"""
    sv = pd.Series(s, index=f.index)
    if name in ("trend", "breakout"):
        g1 = (np.sign(f.ema_fast - f.ema_slow) == sv) & (f.adx >= p.adx_range)
        g2 = f.htf == sv
        g3 = f.vol_ratio >= 1.2
        g4 = f.atr > f.atr_ma
    else:
        g1 = ((-sv * f.bb_z) >= p.bb_z_extreme) & (((sv == 1) & (f.rsi <= p.rsi_low)) | ((sv == -1) & (f.rsi >= p.rsi_high)))
        g2 = f.htf != -sv
        g3 = f.adx < p.adx_range
        g4 = f.vol_ratio <= 2.0
    return np.column_stack([g.fillna(False).to_numpy(bool) for g in (g1, g2, g3, g4)])


def generate_candidates(f: pd.DataFrame, regime: pd.Series, p: StrategyParams) -> pd.DataFrame:
    """產生每根 K 線（收盤時）的候選訊號。"""
    n = len(f)
    side = np.zeros(n, dtype="int64")
    strat = np.full(n, "", dtype=object)
    conflict = np.zeros(n, dtype=bool)
    ev = np.zeros((n, 4), dtype=bool)
    raw = {"trend": _trend(f, p), "breakout": _breakout(f, p), "mean_reversion": _mean_rev(f, p)}
    reg = regime.to_numpy(object)
    for name in PRIORITY:
        enabled = np.array([name in STRATEGIES_BY_REGIME.get(r, []) for r in reg])
        s = np.where(enabled, raw[name], 0)
        clash = (side != 0) & (s != 0) & (s != side)
        conflict |= clash
        take = (side == 0) & (s != 0)
        side = np.where(take, s, side)
        strat = np.where(take, name, strat)
        if take.any():
            e = _evidence(name, f, np.where(take, s, 1), p)
            ev[take] = e[take]
    side = np.where(conflict, 0, side)
    n_ev = ev.sum(axis=1)
    raw_side = side.copy()
    side = np.where((side != 0) & (n_ev >= p.min_evidence_groups), side, 0)

    stop_dist = p.stop_atr_mult * f["atr"].to_numpy()
    tp_dist = np.where(strat == "mean_reversion", np.abs(f["bb_mid"].to_numpy() - f["close"].to_numpy()),
                       p.tp_r * stop_dist)
    return pd.DataFrame({
        "side": side, "raw_side": raw_side, "strategy": strat, "stop_dist": stop_dist, "tp1_dist": tp_dist,
        "evidence": n_ev, "conflict": conflict,
        "g1": ev[:, 0], "g2": ev[:, 1], "g3": ev[:, 2], "g4": ev[:, 3],
    }, index=f.index)


def prepare_dataset(base_df: pd.DataFrame, htf_df: pd.DataFrame | None, base_tf: str, htf_tf: str,
                    p: StrategyParams):
    """一次完成：HTF 對齊 → 特徵 → 市場狀態 → 候選訊號。回傳 (features, candidates)。"""
    htf_aligned = None
    if htf_df is not None and len(htf_df) and base_tf != htf_tf:
        htf_aligned = align_htf(base_df.index, base_tf, compute_htf_trend(htf_df, p), htf_tf)
    f = add_features(base_df, p, htf_aligned)
    f["regime"] = classify_regime(f, p)
    return f, generate_candidates(f, f["regime"], p)
