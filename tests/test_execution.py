import os, tempfile

import pandas as pd

from aicq.config import AppConfig, CostConfig, RiskConfig, StrategyParams
from aicq.data.provider import MarketData, load_synthetic
from aicq.execution.guard import LiveTradingDisabled, assert_paper_only, live_status_message, place_live_order
from aicq.execution.paper import PaperBroker
from aicq.execution.storage import Storage
from aicq.models import DataGate, Instrument, TradeSignal
from aicq.strategy.signal_engine import analyze_symbol
from aicq.data.validation import DataQuality
from aicq.models import AccountState, default_instrument
from tests.helpers import make_df, raises

INST = Instrument("T", 0.01, 0.01, 0.01, 1.0, verified=True)
CFG = AppConfig(cost=CostConfig(0.0, 0.0, 0.0), risk=RiskConfig(max_daily_loss_pct=100))
T0 = pd.Timestamp("2026-03-01 00:00", tz="UTC")


def _sig(key="k1", side="long", entry=100.0, stop=99.0, tp=102.0):
    return TradeSignal(symbol="T", exchange="x", timeframe="15m", direction=side, direction_label="做多",
                       signal_key=key, entry_ref=entry, stop=stop, tp1=tp, strategy="trend", data_source="synthetic",
                       bar_time=(T0 - pd.Timedelta(minutes=15)).isoformat())    # 剛收盤的 K 線


def _db():
    d = tempfile.mkdtemp()
    return os.path.join(d, "t.db")


def test_open_persist_and_duplicate_blocked():
    path = _db()
    b = PaperBroker(CFG, Storage(path))
    ok, msg = b.try_open(_sig(), INST, {"T": 100.0}, DataGate(True), now=T0)
    assert ok, msg
    ok2, msg2 = b.try_open(_sig(), INST, {"T": 100.0}, DataGate(True), now=T0)
    assert not ok2                                    # 同訊號不可重複下單
    b2 = PaperBroker(CFG, Storage(path))               # 重新啟動後狀態仍在
    assert len(b2.db.positions("open")) == 1
    assert not b2.try_open(_sig(), INST, {"T": 100.0}, DataGate(True), now=T0)[0]
    ok3, msg3 = b2.try_open(_sig(key="k2"), INST, {"T": 100.0}, DataGate(True), now=T0)
    assert not ok3 and "同商品" in msg3                # 同商品禁止加碼


def test_stop_and_tp_close_with_pnl_recorded():
    b = PaperBroker(CFG, Storage(_db()))
    assert b.try_open(_sig(), INST, {"T": 100.0}, DataGate(True), now=T0)[0]
    pos = b.db.positions("open").iloc[0]
    start = pd.Timestamp(int(pos.eval_from_ms), unit="ms", tz="UTC")
    # 進場當根（進場前）的劇烈下跌不應觸發；之後一根才觸及止盈
    pre = make_df([(100, 100, 90, 95)], start_ms=int(pos.eval_from_ms) - 900_000)
    assert b.update({"T": pre}) == []
    post = make_df([(100, 101, 99.5, 100.5), (100.5, 102.5, 100, 102)], start_ms=int(pos.eval_from_ms))
    assert b.update({"T": post}) == [int(pos.id)]
    closed = b.db.positions("closed").iloc[0]
    assert closed.exit_reason == "tp" and closed.net_pnl > 0
    assert abs(b.equity({}) - (CFG.initial_equity + closed.net_pnl)) < 1e-6


def test_stop_loss_and_same_bar_conservative():
    b = PaperBroker(CFG, Storage(_db()))
    b.try_open(_sig(), INST, {"T": 100.0}, DataGate(True), now=T0)
    pos = b.db.positions("open").iloc[0]
    both = make_df([(100, 103, 98, 100)], start_ms=int(pos.eval_from_ms))
    b.update({"T": both})
    c = b.db.positions("closed").iloc[0]
    assert c.exit_reason == "stop" and c.net_pnl < 0


def test_daily_loss_blocks_paper_open():
    cfg = AppConfig(cost=CostConfig(0, 0, 0), risk=RiskConfig(max_daily_loss_pct=1.0))
    b = PaperBroker(cfg, Storage(_db()))
    b.account_state({}, now=T0)                       # 記錄日初權益 10000
    b.db.set("day_start_equity", cfg.initial_equity * 1.02)   # 等同當日已虧超過 1%
    ok, msg = b.try_open(_sig(), INST, {"T": 100.0}, DataGate(True), now=T0)
    assert not ok and "單日損失" in msg


def test_stale_data_blocks_paper_open():
    b = PaperBroker(CFG, Storage(_db()))
    ok, msg = b.try_open(_sig(), INST, {}, DataGate(False, ["行情資料過期"]), now=T0)
    assert not ok and "過期" in msg and len(b.db.positions()) == 0


def test_flat_signal_never_opens():
    b = PaperBroker(CFG, Storage(_db()))
    s = _sig(side="flat")
    assert not b.try_open(s, INST, {}, DataGate(True), now=T0)[0]


def test_live_trading_cannot_be_enabled():
    cfg = AppConfig(live_requested=True)
    assert cfg.live_trading_enabled is False
    assert_paper_only(cfg)
    assert "忽略" in live_status_message(cfg)
    assert raises(LiveTradingDisabled, place_live_order, "BTC", 1)


def test_signal_storage_dedup():
    s = Storage(_db())
    sig = _sig()
    assert s.save_signal(sig) is True and s.save_signal(sig) is False


def _md(df, source="synthetic", ok=True, issues=None):
    q = DataQuality(ok=ok, issues=issues or [], n_rows=len(df))
    return MarketData("BTC/USDT:USDT", "15m", df, source, q, "t")


def test_signal_engine_offline_stale_and_normal():
    cfg = AppConfig()
    inst = default_instrument("BTC/USDT:USDT")
    acct = AccountState(10_000, 10_000)
    off = analyze_symbol("BTC/USDT:USDT", "15m", MarketData("BTC/USDT:USDT", "15m", make_df([(1, 1, 1, 1)]), "offline", None, "離線"), None, inst, acct, cfg)
    assert off.direction == "flat" and off.rejection_reasons and not off.data_ok
    base = load_synthetic("BTC/USDT:USDT", "15m", 1500, cfg)
    htf = load_synthetic("BTC/USDT:USDT", "1h", 800, cfg)
    stale = _md(base.df, ok=False, issues=["行情資料過期"])
    s = analyze_symbol("BTC/USDT:USDT", "15m", stale, htf, inst, acct, cfg)
    assert s.direction == "flat" and "行情資料過期" in s.rejection_reasons
    found = {"flat": 0, "long": 0, "short": 0}
    for k in range(0, 600, 3):
        d = base.df.iloc[: len(base.df) - k]
        r = analyze_symbol("BTC/USDT:USDT", "15m", _md(d), htf, inst, acct, cfg, params=StrategyParams(min_evidence_groups=2))
        found[r.direction] += 1
        assert r.direction_label in ("做多", "做空", "觀望") and r.uncertainties and r.data_source == "synthetic"
        if r.direction != "flat":
            assert r.plan.allowed and r.stop and r.tp1 and r.plan.rr_net >= cfg.risk.min_net_rr and r.conditions and r.cons
            assert (r.stop < r.entry_ref) == (r.direction == "long")
    assert found["flat"] > 0


def test_long_and_short_signals_pass_full_pipeline_with_relaxed_thresholds():
    """放寬成本門檻的設定下，系統確實能產生做多與做空，且倉位風險貼近 0.25% 預算（確定性資料）。"""
    from dataclasses import replace
    from aicq.data.synthetic import generate_synthetic_ohlcv, resample_ohlcv
    cfg = replace(AppConfig(), risk=RiskConfig(min_net_rr=1.0, max_cost_to_risk=1.0), cost=CostConfig(0.0002, 1.0, 0.0001),
                  params=StrategyParams(min_evidence_groups=2))
    inst = Instrument("BTC/USDT:USDT", 0.1, 0.001, 0.001, 100, verified=True)
    end = pd.Timestamp("2026-01-01", tz="UTC")           # 固定結束時間 → 結果與執行時刻無關
    bdf = generate_synthetic_ohlcv(5000, "15m", seed=7, start_price=60_000.0, end=end)
    base = _md(bdf)
    htf = _md(resample_ohlcv(bdf, "1h"))
    seen = set()
    for k in range(0, 2500, 2):
        d = bdf.iloc[: len(bdf) - k]
        s = analyze_symbol("BTC/USDT:USDT", "15m", _md(d), htf, inst, AccountState(1e4, 1e4), cfg)
        if s.direction != "flat":
            seen.add(s.direction)
            assert s.plan.risk_amount <= 10_000 * 0.25 / 100 + 1e-6 and s.plan.allowed
    assert seen == {"long", "short"}
