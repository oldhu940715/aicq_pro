"""樣本外驗證工具：時間序列切分、Walk-forward、參數穩健性、成本敏感度、分市場狀態表現。

原則：特徵一律只用過去資料計算，因此可先對全資料計算特徵再按時間切片（切片起點的指標暖機使用的是『更早』的資料，
不會洩漏未來）。參數只在訓練窗選擇，並在其後的測試窗檢驗；絕不挑選全樣本最佳參數當作成果。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

from ..config import BacktestConfig, CostConfig, RiskConfig, StrategyParams
from ..models import Instrument
from ..strategy.strategies import prepare_dataset
from .engine import BacktestResult, run_backtest
from .metrics import compute_metrics, max_drawdown_pct


def chronological_split(n: int, train: float = 0.5, val: float = 0.2, embargo: int = 0):
    """回傳 (train, val, test) 三段 slice，按時間先後且互不重疊；段與段之間留 embargo 根空窗。"""
    if not (0 < train and 0 < val and train + val < 1):
        raise ValueError("train/val 比例不合法")
    a = int(n * train)
    b = int(n * (train + val))
    return slice(0, a), slice(a + embargo, b), slice(b + embargo, n)


def _slice(feat, cand, sl):
    return feat.iloc[sl], cand.iloc[sl]


def param_grid(base: StrategyParams, stop_mults=(1.2, 1.5, 2.0), tp_rs=(1.5, 2.0, 3.0)) -> list:
    return [replace(base, stop_atr_mult=s, tp_r=r) for s, r in product(stop_mults, tp_rs)]


def run_segments(feat, cand, sl, inst, risk, cost, bt, tf, initial_equity=None) -> BacktestResult:
    f, c = _slice(feat, cand, sl)
    return run_backtest(f, c, inst, risk, cost, bt, tf, initial_equity=initial_equity)


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos_trades: pd.DataFrame
    oos_equity: pd.Series
    oos_metrics: dict


def walk_forward(base_df, htf_df, base_tf, htf_tf, grid, inst: Instrument, risk: RiskConfig, cost: CostConfig,
                 bt: BacktestConfig, train_bars: int, test_bars: int, step_bars: int | None = None,
                 min_train_trades: int = 10, metric: str = "expectancy_r") -> WalkForwardResult:
    """滾動：每折在訓練窗從 grid 選參數（需訓練交易數達標），再在『緊接其後』的測試窗用該參數驗證。"""
    step_bars = step_bars or test_bars
    data = [prepare_dataset(base_df, htf_df, base_tf, htf_tf, p) for p in grid]
    n = len(base_df)
    folds, trades, eqs = [], [], []
    equity = bt.initial_equity
    start = 0
    k = 0
    while start + train_bars + test_bars <= n:
        tr = slice(start, start + train_bars)
        te = slice(start + train_bars, start + train_bars + test_bars)
        best, best_score, best_m = None, -np.inf, None
        for gi, (f, c) in enumerate(data):
            m = compute_metrics(run_segments(f, c, tr, inst, risk, cost, bt, base_tf))
            score = m.get(metric, float("nan"))
            if m["n_trades"] >= min_train_trades and np.isfinite(score) and score > best_score:
                best, best_score, best_m = gi, score, m
        row = {"fold": k, "train_start": base_df.index[tr.start], "test_start": base_df.index[te.start],
               "test_end": base_df.index[te.stop - 1], "chosen_param": None, "train_score": np.nan,
               "oos_trades": 0, "oos_net_return_pct": 0.0, "oos_expectancy_r": np.nan}
        if best is not None:
            f, c = data[best]
            r = run_segments(f, c, te, inst, risk, cost, bt, base_tf, initial_equity=equity)
            mo = compute_metrics(r)
            p = grid[best]
            row.update(chosen_param=f"stop×{p.stop_atr_mult}/tp {p.tp_r}R", train_score=best_score,
                       oos_trades=mo["n_trades"], oos_net_return_pct=mo["net_return_pct"],
                       oos_expectancy_r=mo["expectancy_r"])
            equity = r.final_equity
            trades.append(r.trades.assign(fold=k))
            eqs.append(r.equity)
        else:
            row["chosen_param"] = "（訓練窗無合格參數 → 該折不交易）"
            eqs.append(pd.Series(equity, index=base_df.index[te]))
        folds.append(row)
        start += step_bars
        k += 1
    oos_tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    oos_eq = pd.concat(eqs) if eqs else pd.Series(dtype=float)
    oos_eq = oos_eq[~oos_eq.index.duplicated(keep="last")]
    res = BacktestResult(oos_tr if len(oos_tr) else run_backtest(data[0][0].iloc[:0], data[0][1].iloc[:0], inst, risk, cost, bt, base_tf).trades,
                         oos_eq, bt.initial_equity, {}, None, base_tf)
    return WalkForwardResult(pd.DataFrame(folds), oos_tr, oos_eq, compute_metrics(res, base_df.loc[oos_eq.index] if len(oos_eq) else None, cost))


def robustness_table(base_df, htf_df, base_tf, htf_tf, grid, inst, risk, cost, bt, chunks: int = 3) -> pd.DataFrame:
    """每組參數 × 每個時間區段（等長）的結果。不是為了挑最佳，而是看結果是否隨時期與參數穩定。"""
    rows = []
    n = len(base_df)
    edges = np.linspace(0, n, chunks + 1, dtype=int)
    for p in grid:
        f, c = prepare_dataset(base_df, htf_df, base_tf, htf_tf, p)
        row = {"stop_atr_mult": p.stop_atr_mult, "tp_r": p.tp_r}
        pos = 0
        for j in range(chunks):
            m = compute_metrics(run_segments(f, c, slice(edges[j], edges[j + 1]), inst, risk, cost, bt, base_tf))
            row[f"期{j+1}_交易數"] = m["n_trades"]
            row[f"期{j+1}_期望R"] = m["expectancy_r"]
            pos += int(m["n_trades"] > 0 and m["expectancy_r"] > 0)
        row["期望R為正的期數"] = pos
        rows.append(row)
    return pd.DataFrame(rows)


def cost_sensitivity(feat, cand, inst, risk, cost: CostConfig, bt, tf, fee_mults=(1.0, 1.5, 2.0),
                     slip_mults=(1.0, 2.0, 4.0)) -> pd.DataFrame:
    rows = []
    for fm, sm in product(fee_mults, slip_mults):
        c2 = replace(cost, taker_fee_rate=cost.taker_fee_rate * fm, slippage_bps=cost.slippage_bps * sm)
        m = compute_metrics(run_backtest(feat, cand, inst, risk, c2, bt, tf))
        rows.append({"手續費倍數": fm, "滑價倍數": sm, "交易數": m["n_trades"],
                     "淨報酬%": m["net_return_pct"], "期望R": m["expectancy_r"], "最大回撤%": m["max_drawdown_pct"]})
    return pd.DataFrame(rows)


def regime_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["regime", "strategy", "n", "net_pnl", "expectancy_r", "win_rate_pct"])
    g = trades.groupby(["regime", "strategy"])
    out = g.agg(n=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), expectancy_r=("r_multiple", "mean"),
                win_rate_pct=("net_pnl", lambda s: (s > 0).mean() * 100)).reset_index()
    return out
