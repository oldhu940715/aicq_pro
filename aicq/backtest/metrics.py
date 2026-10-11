"""績效指標。Sharpe 有兩種並明確標註：日報酬年化（√365、無風險利率 0）與每筆交易（非年化）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CostConfig
from ..timeutil import tf_seconds
from .engine import BacktestResult


def max_drawdown_pct(eq: pd.Series, initial: float) -> float:
    e = np.concatenate([[initial], eq.dropna().to_numpy()])
    if len(e) == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(((peak - e) / peak).max() * 100)


def sharpe_daily_annualized(eq: pd.Series, min_days: int = 10) -> float:
    d = eq.dropna()
    if d.empty:
        return float("nan")
    daily = d.resample("1D").last().dropna().pct_change().dropna()
    if len(daily) < min_days or daily.std(ddof=1) == 0:
        return float("nan")
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(365))


def compute_metrics(res: BacktestResult, prices: pd.DataFrame | None = None, cost: CostConfig | None = None) -> dict:
    t = res.trades
    n = len(t)
    init = res.initial_equity
    m = {"n_trades": n, "initial_equity": init, "final_equity": res.final_equity,
         "net_return_pct": (res.final_equity / init - 1) * 100,
         "max_drawdown_pct": max_drawdown_pct(res.equity, init),
         "sharpe_daily_annualized(√365,rf=0)": sharpe_daily_annualized(res.equity),
         "halted_at": res.halted_at, "blocked_entries": sum(res.blocked.values())}
    if n:
        w, lo = t[t.net_pnl > 0], t[t.net_pnl <= 0]
        gl = abs(lo.net_pnl.sum())
        sd = t.net_pnl.std(ddof=1) if n > 1 else float("nan")
        m.update({
            "win_rate_pct": len(w) / n * 100,
            "avg_win": float(w.net_pnl.mean()) if len(w) else 0.0,
            "avg_loss": float(lo.net_pnl.mean()) if len(lo) else 0.0,
            "expectancy_usdt": float(t.net_pnl.mean()),
            "expectancy_r": float(t.r_multiple.mean()),
            "profit_factor": float(w.net_pnl.sum() / gl) if gl > 0 else (float("inf") if len(w) else float("nan")),
            "sharpe_per_trade(非年化)": float(t.net_pnl.mean() / sd) if sd and sd > 0 else float("nan"),
            "avg_hold_minutes": float(t.bars_held.mean() * tf_seconds(res.timeframe) / 60),
            "total_fees": float(t.fees.sum()), "total_funding": float(t.funding.sum()),
            "total_slippage": float(t.slippage_cost.sum()), "n_ambiguous_bars": int(t.ambiguous.sum()),
        })
    else:
        m.update({k: float("nan") for k in ("win_rate_pct", "avg_win", "avg_loss", "expectancy_usdt", "expectancy_r",
                                            "profit_factor", "sharpe_per_trade(非年化)", "avg_hold_minutes")})
        m.update({"total_fees": 0.0, "total_funding": 0.0, "total_slippage": 0.0, "n_ambiguous_bars": 0})
    if prices is not None and len(prices):
        bh = prices["close"].iloc[-1] / prices["open"].iloc[0] - 1
        m["buy_hold_return_pct"] = bh * 100
        if cost is not None:
            rt = 2 * (cost.taker_fee_rate + cost.slippage_bps / 1e4)
            m["buy_hold_net_of_costs_pct"] = (bh - rt) * 100
    return m


def buy_hold_curve(prices: pd.DataFrame, initial: float) -> pd.Series:
    return prices["close"] / prices["open"].iloc[0] * initial


def expectancy_ci(trades: pd.DataFrame, z: float = 1.96) -> dict:
    """每筆期望值（R）的近似 95% 信賴區間（常態近似；交易數少時僅供參考）。"""
    r = trades["r_multiple"].dropna().to_numpy(float) if len(trades) else np.array([])
    n = len(r)
    if n < 2:
        return {"n": n, "mean_r": float(r.mean()) if n else float("nan"), "se": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "t_stat": float("nan")}
    se = r.std(ddof=1) / np.sqrt(n)
    m = float(r.mean())
    return {"n": n, "mean_r": m, "se": float(se), "lo": m - z * se, "hi": m + z * se,
            "t_stat": m / se if se > 0 else float("nan")}


def oos_verdict(trades: pd.DataFrame, metrics: dict, min_trades: int = 30, max_dd_pct: float = 10.0) -> dict:
    """樣本外判定。只有『足夠交易數＋期望值信賴區間下界>0＋獲利因子>1＋淨報酬>0＋回撤可接受』才算通過，
    且『通過』也只代表『目前沒有被否定』，不是獲利保證。"""
    ci = expectancy_ci(trades)
    reasons = []
    n = metrics.get("n_trades", 0)
    if n == 0:
        return {"status": "未驗證", "reasons": ["樣本外沒有任何成交交易，無法評估"], "ci": ci}
    if n < min_trades:
        reasons.append(f"交易數 {n} 少於 {min_trades}，統計上不足以判斷")
    if not (ci["mean_r"] > 0):
        reasons.append(f"每筆期望值 {ci['mean_r']:.3f}R ≤ 0")
    elif not (ci["lo"] > 0):
        reasons.append(f"期望值 95% 信賴區間下界 {ci['lo']:.3f}R ≤ 0，無法排除只是運氣")
    pf = metrics.get("profit_factor", float("nan"))
    if not (pf > 1):
        reasons.append(f"獲利因子 {pf:.2f} ≤ 1")
    if not (metrics.get("net_return_pct", 0) > 0):
        reasons.append("扣成本後淨報酬 ≤ 0")
    if metrics.get("max_drawdown_pct", 0) > max_dd_pct:
        reasons.append(f"最大回撤 {metrics['max_drawdown_pct']:.1f}% 超過 {max_dd_pct}%")
    if not reasons:
        return {"status": "通過（尚未被否定）", "reasons": ["仍需更長期、跨行情、跨標的驗證；非獲利保證"], "ci": ci}
    status = "證據不足" if (n < min_trades and ci["mean_r"] > 0) else "未通過"
    return {"status": status, "reasons": reasons, "ci": ci}
