"""事件驅動回測引擎。

時序（無前視）：訊號在第 i 根『收盤』產生 → 第 i+1(+delay) 根『開盤』成交 → 該根起逐根檢查止損/止盈。
成本：手續費（taker，進出各一次）、滑價（進場與止損市價單）、資金費（依持倉期間經過的結算次數）。
風控：與模擬交易共用同一個 RiskManager（單日損失、策略回撤停用、倉位計算、最小下單量等）。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig, CostConfig, RiskConfig
from ..models import AccountState, DataGate, Instrument, TradeRequest
from ..risk.manager import RiskManager
from ..timeutil import index_to_ms, taipei_day_key, tf_ms
from .fills import bar_exit, funding_cost


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    initial_equity: float
    blocked: dict = field(default_factory=dict)
    halted_at: object = None
    timeframe: str = ""

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.initial_equity


TRADE_COLS = ["entry_time", "exit_time", "side", "strategy", "regime", "qty", "entry_price", "exit_price",
              "stop", "tp", "exit_reason", "ambiguous", "gross_pnl", "fees", "slippage_cost", "funding",
              "net_pnl", "r_multiple", "risk_amount", "bars_held", "evidence", "equity_after"]


def run_backtest(feat: pd.DataFrame, cand: pd.DataFrame, inst: Instrument, risk: RiskConfig,
                 cost: CostConfig, bt: BacktestConfig, timeframe: str, funding_series=None,
                 initial_equity: float | None = None) -> BacktestResult:
    n = len(feat)
    eq0 = float(initial_equity if initial_equity is not None else bt.initial_equity)
    if n == 0:
        return BacktestResult(pd.DataFrame(columns=TRADE_COLS), pd.Series(dtype=float), eq0, {}, None, timeframe)
    ms = index_to_ms(feat.index)
    step = tf_ms(timeframe)
    o, h, l, c = (feat[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    side_a = cand["side"].to_numpy()
    stop_a, tp_a = cand["stop_dist"].to_numpy(float), cand["tp1_dist"].to_numpy(float)
    strat_a, ev_a = cand["strategy"].to_numpy(object), cand["evidence"].to_numpy()
    reg_a = feat["regime"].to_numpy(object) if "regime" in feat else np.full(n, "", dtype=object)
    day = taipei_day_key(ms)
    slip, fee = cost.slippage_bps / 1e4, cost.taker_fee_rate
    rm = RiskManager(risk, cost)
    hold_h = bt.max_hold_bars * step / 3_600_000

    cash, peak = eq0, eq0
    halted_at = None
    day_start, cur_day, last_eq = eq0, day[0], eq0
    eq = np.full(n, np.nan)
    pos, pending = None, None
    trades, blocked = [], Counter()

    def close(i, px, raw, reason, amb):
        nonlocal cash, pos
        p = pos
        gross = p["side"] * (px - p["entry"]) * p["qty"]
        fee_x = fee * p["qty"] * px
        fund = funding_cost(p["side"], p["qty"] * p["entry"], p["entry_ms"], int(ms[i] + step),
                            cost.funding_rate_per_8h, funding_series)
        cash += gross - fee_x - fund
        net = gross - p["fee_e"] - fee_x - fund
        slip_cost = p["qty"] * (abs(p["entry"] - p["entry_ref"]) + abs(px - raw))
        trades.append({
            "entry_time": feat.index[p["i"]], "exit_time": feat.index[i], "side": p["side"],
            "strategy": p["strategy"], "regime": p["regime"], "qty": p["qty"], "entry_price": p["entry"],
            "exit_price": px, "stop": p["stop"], "tp": p["tp"], "exit_reason": reason, "ambiguous": amb,
            "gross_pnl": gross, "fees": p["fee_e"] + fee_x, "slippage_cost": slip_cost, "funding": fund,
            "net_pnl": net, "r_multiple": net / p["risk_amount"] if p["risk_amount"] else np.nan,
            "risk_amount": p["risk_amount"], "bars_held": i - p["i"] + 1, "evidence": p["evidence"],
            "equity_after": cash})
        pos = None

    for i in range(n):
        if day[i] != cur_day:
            cur_day, day_start = day[i], last_eq
        # 1) 進場：訊號 bar 的『下一根』開盤成交
        if pending is not None and pending[0] == i:
            sb, pending = pending[1], None
            if pos is None and halted_at is None and side_a[sb] != 0:
                side = int(side_a[sb])
                fill = o[i] * (1 + side * slip)
                req = TradeRequest("BT", side, fill, fill - side * stop_a[sb], fill + side * tp_a[sb],
                                   str(strat_a[sb]), f"bt|{sb}", hold_h)
                acct = AccountState(equity=cash, day_start_equity=day_start)
                plan = rm.evaluate(req, acct, DataGate(True), inst)
                if plan.allowed:
                    fee_e = fee * plan.qty * fill
                    cash -= fee_e
                    pos = {"side": side, "qty": plan.qty, "entry": fill, "entry_ref": o[i], "stop": plan.stop,
                           "tp": plan.tp1, "i": i, "entry_ms": int(ms[i]), "fee_e": fee_e,
                           "strategy": str(strat_a[sb]), "regime": reg_a[sb], "evidence": int(ev_a[sb]),
                           "risk_amount": plan.risk_amount, "liq": plan.liq_price}
                else:
                    for r in plan.reasons:
                        blocked[re.sub(r"[\d.\-]+", "#", r)] += 1
        # 2) 出場
        if pos is not None:
            res = bar_exit(pos["side"], pos["stop"], pos["tp"], o[i], h[i], l[i], slip)
            liq = pos["liq"]
            if res is not None and res[2] == "stop_gap" and liq is not None and \
                    pos["side"] * (res[0] - liq) <= 0:      # 跳空直接越過粗估強平價：以強平價出場（逐倉損失上限≈保證金）
                close(i, liq, liq, "liquidated", False)
            elif res is not None:
                close(i, res[0], res[1], res[2], res[3])
            elif i - pos["i"] + 1 >= bt.max_hold_bars:
                close(i, c[i] * (1 - pos["side"] * slip), c[i], "time", False)
        # 3) 逐根以收盤價計淨值
        unreal = pos["side"] * (c[i] - pos["entry"]) * pos["qty"] if pos is not None else 0.0
        eq[i] = cash + unreal
        last_eq = eq[i]
        peak = max(peak, eq[i])
        if halted_at is None and (peak - eq[i]) / peak * 100 >= risk.strategy_max_drawdown_pct:
            halted_at = feat.index[i]
        # 4) 新訊號（只在空手且無待成交時）
        if pos is None and pending is None and halted_at is None and side_a[i] != 0:
            due = i + 1 + bt.extra_delay_bars
            if due < n:
                pending = (due, i)
    if pos is not None:  # 資料結束，強制平倉
        i = n - 1
        close(i, c[i] * (1 - pos["side"] * slip), c[i], "data_end", False)
        eq[i] = cash
    df = pd.DataFrame(trades, columns=TRADE_COLS)
    return BacktestResult(df, pd.Series(eq, index=feat.index), eq0, dict(blocked), halted_at, timeframe)
