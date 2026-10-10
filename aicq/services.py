"""服務層：把資料、分析、回測串起來，與 UI 分離以便測試。"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.metrics import buy_hold_curve, compute_metrics
from .backtest.validation import (chronological_split, cost_sensitivity, param_grid, regime_breakdown,
                                  robustness_table, run_segments, walk_forward)
from .config import AppConfig, BacktestConfig, StrategyParams
from .data.cache import OhlcvCache
from .data.provider import MarketData, load_instrument, load_live, load_synthetic
from .data.synthetic import resample_ohlcv
from .data.validation import validate_ohlcv
from .execution.paper import PaperBroker
from .execution.storage import Storage
from .models import Instrument
from .strategy.strategies import prepare_dataset


def get_broker(cfg: AppConfig) -> PaperBroker:
    return PaperBroker(cfg, Storage(cfg.db_path))


def load_market(cfg: AppConfig, symbol: str, base_tf: str, mode: str, bars: int):
    """mode: 'live' 或 'synthetic'。回傳 (base MarketData, htf MarketData | None, Instrument)。"""
    if mode == "synthetic":
        base = load_synthetic(symbol, base_tf, bars, cfg)
        if base_tf == cfg.htf:
            htf = None
        else:
            hdf = resample_ohlcv(base.df, cfg.htf)
            htf = MarketData(symbol, cfg.htf, hdf, "synthetic",
                             validate_ohlcv(hdf, cfg.htf, now=hdf.index[-1] + pd.Timedelta(hours=1), max_age_bars=99) if len(hdf) else None,
                             "⚠ 由測試資料聚合而成")
        return base, htf, load_instrument(symbol, cfg, "synthetic")
    cache = OhlcvCache(cfg.db_path)
    base = load_live(symbol, base_tf, bars, cfg, cache)
    htf = None if base_tf == cfg.htf else load_live(symbol, cfg.htf, 400, cfg, cache)
    return base, htf, load_instrument(symbol, cfg, "live")


def run_validation_suite(cfg: AppConfig, base: MarketData, htf: MarketData | None, base_tf: str, inst: Instrument,
                         params: StrategyParams, bt: BacktestConfig, do_walk_forward: bool = True) -> dict:
    """完整回測流程：訓練/驗證/樣本外三段、成本敏感度、分狀態表現、Walk-forward 與參數穩健性。"""
    df = base.df
    hdf = htf.df if htf is not None else None
    feat, cand = prepare_dataset(df, hdf, base_tf, cfg.htf, params)
    n = len(df)
    tr, va, te = chronological_split(n, 0.5, 0.2, embargo=bt.max_hold_bars)
    seg = {}
    for name, sl in (("訓練", tr), ("驗證", va), ("樣本外", te)):
        r = run_segments(feat, cand, sl, inst, cfg.risk, cfg.cost, bt, base_tf)
        seg[name] = (r, compute_metrics(r, df.iloc[sl], cfg.cost))
    full = run_backtest(feat, cand, inst, cfg.risk, cfg.cost, bt, base_tf)
    out = {"features": feat, "candidates": cand, "segments": seg, "full": full,
           "full_metrics": compute_metrics(full, df, cfg.cost),
           "benchmark": buy_hold_curve(df, bt.initial_equity),
           "cost_sensitivity": cost_sensitivity(feat.iloc[te], cand.iloc[te], inst, cfg.risk, cfg.cost, bt, base_tf),
           "regimes": regime_breakdown(full.trades), "data_source": base.source}
    if do_walk_forward and n >= 3000:
        grid = param_grid(params)
        train_bars, test_bars = int(n * 0.35), int(n * 0.15)
        out["walk_forward"] = walk_forward(df, hdf, base_tf, cfg.htf, grid, inst, cfg.risk, cfg.cost, bt,
                                           train_bars, test_bars, min_train_trades=5)
        out["robustness"] = robustness_table(df, hdf, base_tf, cfg.htf, grid, inst, cfg.risk, cfg.cost, bt)
    return out
