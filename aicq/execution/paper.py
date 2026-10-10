"""模擬交易（紙上交易）。與回測共用 bar_exit／funding_cost／RiskManager，行為一致。

保守約定：模擬單於『分析當下』以最新收盤價＋不利滑價成交；之後只用『進場時間之後才開始』且已收盤的 K 線判斷
止損／止盈（避免以進場前的價格波動誤判）。
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..models import AccountState, DataGate, OpenPosition, TradeSignal
from ..risk.manager import RiskManager
from ..risk.sizing import round_conservative  # noqa: F401  (供外部引用)
from ..timeutil import fmt_taipei, index_to_ms, taipei_day_key, tf_ms, to_ms, now_utc
from ..backtest.fills import bar_exit, funding_cost
from ..models import TradeRequest, Instrument
from .guard import assert_paper_only
from .storage import Storage


class PaperBroker:
    def __init__(self, cfg: AppConfig, storage: Storage):
        assert_paper_only(cfg)
        self.cfg, self.db = cfg, storage
        self.rm = RiskManager(cfg.risk, cfg.cost)
        if self.db.get("initial_equity") is None:
            self.db.set("initial_equity", cfg.initial_equity)

    # -- 帳戶 -----------------------------------------------------------
    @property
    def initial_equity(self) -> float:
        return float(self.db.get("initial_equity", self.cfg.initial_equity))

    def realized_pnl(self) -> float:
        p = self.db.positions("closed")
        return float(p["net_pnl"].sum()) if len(p) else 0.0

    def equity(self, marks: dict) -> float:
        eq = self.initial_equity + self.realized_pnl()
        o = self.db.positions("open")
        for _, r in o.iterrows():
            px = marks.get(r["symbol"], r["entry_price"])
            eq += r["side"] * (px - r["entry_price"]) * r["qty"] - r["entry_fee"]
        return float(eq)

    def strategy_drawdown_pct(self, strategy: str) -> float:
        p = self.db.positions("closed")
        p = p[p["strategy"] == strategy] if len(p) else p
        if p.empty:
            return 0.0
        cum = np.concatenate([[0.0], p["net_pnl"].cumsum().to_numpy()])
        dd = (np.maximum.accumulate(cum) - cum).max()
        return float(dd / self.initial_equity * 100)

    def account_state(self, marks: dict, now=None) -> AccountState:
        now = pd.Timestamp(now) if now is not None else now_utc()
        eq = self.equity(marks)
        key = int(taipei_day_key(to_ms(now)))
        if self.db.get("day_key") != key:      # 台北日換日：重設日初權益
            self.db.set("day_key", key)
            self.db.set("day_start_equity", eq)
        day_start = float(self.db.get("day_start_equity", eq))
        pos = []
        for _, r in self.db.positions("open").iterrows():
            pos.append(OpenPosition(r["symbol"], int(r["side"]), r["qty"], r["entry_price"], r["notional"],
                                    r["margin"], r["strategy"]))
        strategies = {"trend", "breakout", "mean_reversion"}
        halted = {s for s in strategies if self.strategy_drawdown_pct(s) >= self.cfg.risk.strategy_max_drawdown_pct}
        return AccountState(eq, day_start, pos, halted, self.db.executed_keys())

    # -- 開倉 -----------------------------------------------------------
    def try_open(self, sig: TradeSignal, inst: Instrument, marks: dict, gate: DataGate, now=None):
        """回傳 (是否成功, 訊息)。會以『當下帳戶狀態』重新跑一次風控，不信任舊的 plan。"""
        now = pd.Timestamp(now) if now is not None else now_utc()
        if sig.direction not in ("long", "short") or sig.entry_ref is None or sig.stop is None or sig.tp1 is None:
            return False, "此訊號為觀望或條件不完整，不建立模擬單"
        side = 1 if sig.direction == "long" else -1
        slip = self.cfg.cost.slippage_bps / 1e4
        fill = sig.entry_ref * (1 + side * slip)
        acct = self.account_state(marks, now)
        hold_h = 96 * tf_ms(sig.timeframe) / 3_600_000
        req = TradeRequest(sig.symbol, side, fill, sig.stop, sig.tp1, sig.strategy, sig.signal_key, hold_h)
        plan = self.rm.evaluate(req, acct, gate, inst)
        if not plan.allowed:
            msg = "風控拒絕：" + "；".join(plan.reasons)
            self.db.log("WARN", f"{sig.symbol} {msg}")
            return False, msg
        step = tf_ms(sig.timeframe)
        opened_ms = to_ms(now)
        rec = dict(signal_key=sig.signal_key, symbol=sig.symbol, timeframe=sig.timeframe, side=side,
                   strategy=sig.strategy, qty=plan.qty, entry_price=fill, stop=plan.stop, tp1=plan.tp1,
                   notional=plan.notional, margin=plan.margin, risk_amount=plan.risk_amount,
                   entry_fee=self.cfg.cost.taker_fee_rate * plan.qty * fill, opened_at=now.isoformat(),
                   opened_ms=opened_ms, eval_from_ms=-(-opened_ms // step) * step, max_hold_bars=96,
                   status="open", reason=json.dumps({"conditions": sig.conditions, "cons": sig.cons,
                                                      "regime": sig.regime_label}, ensure_ascii=False),
                   data_source=sig.data_source)
        try:
            pid = self.db.insert_position(rec)
        except sqlite3.IntegrityError:
            return False, "此訊號已建立過模擬單（重複執行被阻擋）"
        self.db.log("INFO", f"模擬開倉 #{pid} {sig.symbol} {sig.direction_label} qty={plan.qty} @ {fill:.4f}")
        return True, f"已建立模擬單 #{pid}：{sig.direction_label} {plan.qty} @ {fill:.4f}（風險約 {plan.risk_amount:.2f} USDT）"

    # -- 更新／平倉 -------------------------------------------------------
    def _finish(self, r, exit_px, raw_px, reason, exit_ms):
        cc = self.cfg.cost
        side = int(r["side"])
        gross = side * (exit_px - r["entry_price"]) * r["qty"]
        fee_x = cc.taker_fee_rate * r["qty"] * exit_px
        fund = funding_cost(side, r["qty"] * r["entry_price"], int(r["opened_ms"]), int(exit_ms), cc.funding_rate_per_8h)
        net = gross - r["entry_fee"] - fee_x - fund
        slip = r["qty"] * abs(exit_px - raw_px)
        self.db.close_position(int(r["id"]), exit_price=exit_px, exit_time=pd.Timestamp(exit_ms, unit="ms", tz="UTC").isoformat(),
                               exit_reason=reason, gross_pnl=gross, fees=r["entry_fee"] + fee_x, funding=fund,
                               slippage_cost=slip, net_pnl=net)
        self.db.log("INFO", f"模擬平倉 #{int(r['id'])} {r['symbol']} {reason} 淨損益 {net:.2f}")

    def update(self, candles: dict) -> list:
        """candles: {symbol: 已收盤 K 線 DataFrame}。回傳本次平倉的持倉 id。"""
        closed = []
        slip = self.cfg.cost.slippage_bps / 1e4
        for _, r in self.db.positions("open").iterrows():
            df = candles.get(r["symbol"])
            if df is None or df.empty:
                continue
            step = tf_ms(r["timeframe"])
            ms = index_to_ms(df.index)
            idx = np.nonzero(ms >= r["eval_from_ms"])[0]
            held = 0
            for j in idx:
                held += 1
                o, h, l, c = (float(df[k].iloc[j]) for k in ("open", "high", "low", "close"))
                res = bar_exit(int(r["side"]), r["stop"], r["tp1"], o, h, l, slip)
                if res is not None:
                    self._finish(r, res[0], res[1], res[2], ms[j] + step)
                    closed.append(int(r["id"]))
                    break
                if held >= r["max_hold_bars"]:
                    self._finish(r, c * (1 - int(r["side"]) * slip), c, "time", ms[j] + step)
                    closed.append(int(r["id"]))
                    break
        return closed

    def close_manual(self, pid: int, price: float, now=None):
        r = self.db.positions("open")
        r = r[r["id"] == pid]
        if r.empty:
            return False
        r = r.iloc[0]
        slip = self.cfg.cost.slippage_bps / 1e4
        self._finish(r, price * (1 - int(r["side"]) * slip), price, "manual", to_ms(now or now_utc()))
        return True
