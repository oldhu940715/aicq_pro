import math

from aicq.config import CostConfig, RiskConfig
from aicq.models import AccountState, DataGate, Instrument, OpenPosition, TradeRequest, default_instrument
from aicq.risk.manager import RiskManager
from aicq.risk.sizing import ceil_to_step, floor_to_step, liquidation_price, round_conservative, size_position
from tests.helpers import approx

BTC = default_instrument("BTC/USDT:USDT")
ZERO = CostConfig(0.0, 0.0, 0.0)
INST = Instrument("T", price_tick=0.01, amount_step=0.01, min_amount=0.01, min_notional=1.0, verified=True)


def req(**k):
    d = dict(symbol="T", side=1, entry=100.0, stop=99.0, tp1=102.0, strategy="trend", signal_key="k1")
    d.update(k)
    return TradeRequest(**d)


def test_step_rounding():
    assert floor_to_step(0.0419, 0.001) == 0.041 and ceil_to_step(0.0411, 0.001) == 0.042
    assert round_conservative(99.123, 0.1, 1) == 99.1 and round_conservative(100.123, 0.1, -1) == 100.2


def test_position_size_matches_risk_budget():
    rm = RiskManager(RiskConfig(), ZERO)
    p = rm.evaluate(req(), AccountState(10_000, 10_000), DataGate(True), INST)
    assert p.allowed, p.reasons
    # 風險 0.25% × 10000 = 25；止損距離 1 → 數量 25
    assert approx(p.qty, 25.0) and approx(p.risk_amount, 25.0) and approx(p.risk_pct_equity, 0.25)
    assert approx(p.notional, 2500.0) and p.margin <= 10_000 and p.rr_gross == 2.0


def test_leverage_is_not_profit_multiplier_and_cap_applies():
    rc = RiskConfig(max_leverage=3.0)
    p = size_position(req(stop=99.99, tp1=100.02), 10_000, 10_000, 0, INST, rc, ZERO)
    assert p.notional <= 3.0 * 10_000 + 1e-6          # 槓桿上限限制名目
    assert any("槓桿上限" in w for w in p.warnings)
    assert p.risk_amount < 25.0                        # 實際止損風險因被限制而低於預算


def test_costs_reduce_rr_and_can_block():
    rc, cc = RiskConfig(), CostConfig(0.0005, 2.0, 0.0001)
    p = size_position(req(entry=100, stop=99.8, tp1=100.4), 10_000, 10_000, 0, INST, rc, cc)
    assert p.rr_net < p.rr_gross and not p.allowed
    assert any("成本" in r or "風險報酬" in r for r in p.reasons)
    assert p.risk_amount > 0 and p.est_fees > 0 and p.est_slippage > 0


def test_invalid_stop_side_and_nan_rejected():
    rm = RiskManager(RiskConfig(), ZERO)
    a = AccountState(10_000, 10_000)
    assert not rm.evaluate(req(stop=101.0), a, DataGate(True), INST).allowed
    assert not rm.evaluate(req(tp1=99.5), a, DataGate(True), INST).allowed
    assert not rm.evaluate(req(entry=float("nan")), a, DataGate(True), INST).allowed
    assert not rm.evaluate(req(), AccountState(0, 0), DataGate(True), INST).allowed


def test_min_amount_and_min_notional():
    small = Instrument("T", 0.01, 0.01, 0.01, 1.0, verified=True)
    p = size_position(req(), 100.0, 100.0, 0, small, RiskConfig(), ZERO)   # 風險 0.25 → 數量 0.25 → OK
    assert p.allowed
    p2 = size_position(req(stop=50.0, tp1=200.0), 100.0, 100.0, 0, small, RiskConfig(), ZERO)  # 止損 50 → 0.005 <0.01
    assert not p2.allowed and any("最小下單量" in r for r in p2.reasons)
    big_min = Instrument("T", 0.01, 0.01, 0.01, 10_000.0, verified=True)
    assert not size_position(req(), 10_000.0, 10_000.0, 0, big_min, RiskConfig(), ZERO).allowed


def test_liquidation_estimate():
    assert approx(liquidation_price(100, 1, 2, 0.005), 100 * (1 - 0.5 + 0.005))
    assert approx(liquidation_price(100, -1, 2, 0.005), 100 * (1 + 0.5 - 0.005))
    p = size_position(req(stop=50.0, tp1=300.0), 1_000_000, 1_000_000, 0, INST, RiskConfig(max_leverage=3), ZERO)
    assert p.liq_price is not None


def test_stop_beyond_liquidation_is_blocked():
    # 3 倍槓桿下強平距離約 33%；止損距離 60% 必須被擋
    p = size_position(req(entry=100, stop=40.0, tp1=300.0), 10_000, 10_000, 0, INST,
                      RiskConfig(max_leverage=3, risk_per_trade_pct=50), ZERO)
    assert not p.allowed


def test_daily_loss_limit_blocks_new_positions():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=1.0), ZERO)
    ok = rm.evaluate(req(), AccountState(9_950, 10_000), DataGate(True), INST)
    assert ok.allowed
    hit = rm.evaluate(req(), AccountState(9_900, 10_000), DataGate(True), INST)
    assert not hit.allowed and any("單日損失" in r for r in hit.reasons)


def test_data_failure_blocks():
    rm = RiskManager(RiskConfig(), ZERO)
    p = rm.evaluate(req(), AccountState(10_000, 10_000), DataGate(False, ["行情過期"]), INST)
    assert not p.allowed and "行情過期" in p.reasons


def test_duplicate_samesymbol_maxpos_halted_blocked():
    rm = RiskManager(RiskConfig(max_open_positions=1), ZERO)
    pos = OpenPosition("T", 1, 1, 100, 100, 50)
    a = AccountState(10_000, 10_000, [pos])
    r = rm.evaluate(req(), a, DataGate(True), INST).reasons
    assert any("持倉數" in x for x in r) and any("同商品" in x for x in r)
    assert any("已執行過" in x for x in rm.evaluate(req(), AccountState(10_000, 10_000, executed_signal_keys={"k1"}), DataGate(True), INST).reasons)
    assert any("停用" in x for x in rm.evaluate(req(), AccountState(10_000, 10_000, halted_strategies={"trend"}), DataGate(True), INST).reasons)


def test_exposure_cap_with_existing_positions():
    rm = RiskManager(RiskConfig(max_total_exposure_x=3.0), ZERO)
    a = AccountState(10_000, 10_000, [OpenPosition("OTHER", 1, 1, 1, 29_900, 10_000 / 3)])
    p = rm.evaluate(req(), a, DataGate(True), INST)
    assert p.notional <= 100 + 1e-6      # 只剩 100 的名目空間


def test_risk_manager_fails_closed_on_exception():
    rm = RiskManager(RiskConfig(), ZERO)
    p = rm.evaluate(req(), None, DataGate(True), INST)   # account=None → 內部例外
    assert not p.allowed and "風控計算失敗" in p.reasons[0]


def test_unverified_instrument_warns():
    p = size_position(req(), 10_000, 10_000, 0, BTC, RiskConfig(), ZERO)
    assert any("未向交易所驗證" in w for w in p.warnings)
