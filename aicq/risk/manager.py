"""風控管理器：獨立於策略引擎。任何一項不通過、或風控計算本身出錯，一律禁止新增交易（fail-closed）。"""
from __future__ import annotations

from ..config import CostConfig, RiskConfig
from ..models import AccountState, DataGate, Instrument, PositionPlan, TradeRequest
from .sizing import size_position


class RiskManager:
    def __init__(self, risk: RiskConfig, cost: CostConfig):
        self.rc, self.cc = risk, cost

    def evaluate(self, req: TradeRequest, account: AccountState, gate: DataGate, inst: Instrument) -> PositionPlan:
        try:
            return self._evaluate(req, account, gate, inst)
        except Exception as e:  # noqa: BLE001 — 風控失敗必須擋單
            return PositionPlan(side=getattr(req, "side", 0),
                                reasons=[f"風控計算失敗，已停止新增交易：{type(e).__name__}: {e}"])

    def _evaluate(self, req, account, gate, inst) -> PositionPlan:
        rc = self.rc
        blocks = []
        if not gate.ok:
            blocks += list(gate.reasons) or ["行情資料不可用"]
        if account.day_start_equity > 0 and account.daily_pnl_pct <= -rc.max_daily_loss_pct:
            blocks.append(f"單日損失 {account.daily_pnl_pct:.2f}% 已達上限 -{rc.max_daily_loss_pct}%，今日停止新增倉位")
        if len(account.open_positions) >= rc.max_open_positions:
            blocks.append(f"持倉數已達上限 {rc.max_open_positions}")
        if any(p.symbol == req.symbol for p in account.open_positions):
            blocks.append("已有同商品持倉（禁止加碼／攤平／重複進場）")
        if req.strategy and req.strategy in account.halted_strategies:
            blocks.append(f"策略 {req.strategy} 因回撤超過 {rc.strategy_max_drawdown_pct}% 已停用")
        if req.signal_key and req.signal_key in account.executed_signal_keys:
            blocks.append("此訊號已執行過（避免重複下單）")
        plan = size_position(req, account.equity, account.free_margin, account.total_notional, inst, rc, self.cc)
        plan.reasons = blocks + plan.reasons
        if gate.is_test_data:
            plan.warnings.append("使用合成測試資料，結果僅供功能驗證")
        plan.allowed = not plan.reasons
        return plan
