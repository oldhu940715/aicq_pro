"""倉位計算、成本估算、強平估算。名目價值、保證金與『止損實際風險』三者分開計算。
槓桿倍數不是獲利倍數：風險＝(止損距離＋成本)×數量，與槓桿無關，槓桿只影響保證金與強平距離。
"""
from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from ..config import CostConfig, RiskConfig
from ..models import Instrument, PositionPlan, TradeRequest


def _q(x: float, step: float, rounding) -> float:
    if step <= 0:
        return float(x)
    d = (Decimal(str(x)) / Decimal(str(step))).to_integral_value(rounding) * Decimal(str(step))
    return float(d)


def floor_to_step(x: float, step: float) -> float:
    return _q(x, step, ROUND_FLOOR)


def ceil_to_step(x: float, step: float) -> float:
    return _q(x, step, ROUND_CEILING)


def round_conservative(price: float, tick: float, side: int) -> float:
    """止損／止盈取整：多單向下、空單向上（止損變遠、止盈變近，皆較保守）。"""
    return floor_to_step(price, tick) if side == 1 else ceil_to_step(price, tick)


def liquidation_price(entry: float, side: int, leverage: float, mmr: float) -> float:
    """逐倉線性合約粗估強平價。未含分級維持保證金、手續費與資金費，實際以交易所為準。"""
    return entry * (1 - 1 / leverage + mmr) if side == 1 else entry * (1 + 1 / leverage - mmr)


def unit_costs(entry: float, exit_price: float, side: int, cost: CostConfig, hold_hours: float):
    """每單位（1 幣）來回成本：(手續費, 滑價, 資金費)。資金費只計『需支付』的方向（保守）。"""
    fee = cost.taker_fee_rate * (entry + exit_price)
    slip = cost.slippage_bps / 1e4 * (entry + exit_price)
    funding = max(side * cost.funding_rate_per_8h, 0.0) * entry * hold_hours / 8.0
    return fee, slip, funding


def size_position(req: TradeRequest, equity: float, free_margin: float, current_notional: float,
                  inst: Instrument, rc: RiskConfig, cc: CostConfig) -> PositionPlan:
    plan = PositionPlan(side=req.side, entry=req.entry, stop=req.stop, tp1=req.tp1)
    nums = [req.entry, req.stop, req.tp1, equity, free_margin]
    if any(not math.isfinite(v) for v in nums) or req.entry <= 0 or equity <= 0:
        plan.reasons.append("輸入數值無效（NaN／非正數），風控無法計算")
        return plan
    if req.side not in (1, -1):
        plan.reasons.append("方向無效")
        return plan
    side = req.side
    stop = round_conservative(req.stop, inst.price_tick, side)
    tp1 = round_conservative(req.tp1, inst.price_tick, side)
    plan.stop, plan.tp1 = stop, tp1
    stop_dist = side * (req.entry - stop)
    tp_dist = side * (tp1 - req.entry)
    if stop_dist <= 0:
        plan.reasons.append("止損價位於進場價錯誤的一側或距離為 0")
        return plan
    if tp_dist <= 0:
        plan.reasons.append("止盈價位於進場價錯誤的一側或距離為 0")
        return plan

    fee_s, slip_s, fund = unit_costs(req.entry, stop, side, cc, req.est_hold_hours)
    fee_t, slip_t, _ = unit_costs(req.entry, tp1, side, cc, req.est_hold_hours)
    risk_unit = stop_dist + fee_s + slip_s + fund
    reward_unit = tp_dist - fee_t - slip_t - fund
    plan.rr_gross = tp_dist / stop_dist
    plan.rr_net = reward_unit / risk_unit
    plan.cost_to_risk = (fee_s + slip_s + fund) / risk_unit

    budget = equity * rc.risk_per_trade_pct / 100.0
    qty_risk = budget / risk_unit
    caps = {
        "槓桿上限": equity * rc.max_leverage / req.entry,
        "總曝險上限": max(rc.max_total_exposure_x * equity - current_notional, 0.0) / req.entry,
        "可用保證金": max(free_margin, 0.0) * rc.max_leverage / req.entry,
    }
    capped = min(qty_risk, *caps.values())
    if capped < qty_risk * 0.999:
        limiter = min(caps, key=caps.get)
        plan.warnings.append(f"倉位受『{limiter}』限制，實際風險低於設定的 {rc.risk_per_trade_pct}%")
    qty = floor_to_step(capped, inst.amount_step)
    if qty < inst.min_amount - 1e-12:
        plan.reasons.append(f"依風險預算算得數量 {qty} 低於最小下單量 {inst.min_amount}（資金過小或止損過遠）")
        return plan
    notional = qty * req.entry
    if notional < inst.min_notional - 1e-9:
        plan.reasons.append(f"名目價值 {notional:.2f} 低於最小名目 {inst.min_notional}")
        return plan

    target_margin = 0.5 * max(free_margin, 1e-9)
    lev = min(rc.max_leverage, max(1.0, float(math.ceil(notional / target_margin - 1e-9))))
    margin = notional / lev
    if margin > free_margin + 1e-9:
        plan.reasons.append(f"保證金 {margin:.2f} 超過可用保證金 {free_margin:.2f}")
    liq = liquidation_price(req.entry, side, lev, rc.maintenance_margin_rate)
    liq_dist = side * (req.entry - liq)
    if stop_dist > rc.liq_safety_ratio * liq_dist:
        plan.reasons.append("止損距離相對強平價過近，止損前可能先被強平")

    plan.qty, plan.notional, plan.leverage, plan.margin, plan.liq_price = qty, notional, lev, margin, liq
    plan.risk_amount = qty * risk_unit
    plan.risk_pct_equity = plan.risk_amount / equity * 100.0
    plan.est_fees = qty * fee_s
    plan.est_slippage = qty * slip_s
    plan.est_funding = qty * fund
    if plan.cost_to_risk > rc.max_cost_to_risk:
        plan.reasons.append(f"預估成本占止損風險 {plan.cost_to_risk:.0%} 超過上限 {rc.max_cost_to_risk:.0%}")
    if plan.rr_net < rc.min_net_rr:
        plan.reasons.append(f"扣成本後風險報酬比 {plan.rr_net:.2f} 低於下限 {rc.min_net_rr}")
    if not inst.verified:
        plan.warnings.append("商品規格／強平價為預設或粗估值，未向交易所驗證，不可視為已完成風控計算")
    if inst.contract_size != 1.0:
        plan.reasons.append("contractSize 非 1，數量換算未支援")
    plan.allowed = not plan.reasons
    return plan
