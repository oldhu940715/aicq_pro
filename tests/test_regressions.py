"""第二輪獨立審查發現的問題與對抗式驗證，全部固化為回歸測試。"""
import os
import sqlite3
import tempfile
from dataclasses import replace

import numpy as np
import pandas as pd

from aicq.backtest.engine import run_backtest
from aicq.backtest.metrics import compute_metrics, expectancy_ci, oos_verdict
from aicq.backtest.validation import param_grid, three_way_selection, walk_forward
from aicq.config import AppConfig, BacktestConfig, CostConfig, RiskConfig, StrategyParams
from aicq.data.provider import MarketData
from aicq.data.synthetic import generate_synthetic_ohlcv, resample_ohlcv
from aicq.data.validation import DataQuality
from aicq.execution.paper import PaperBroker
from aicq.execution.storage import PositionLimitReached, Storage
from aicq.models import AccountState, DataGate, Instrument, TradeSignal
from aicq.services import data_fingerprint, load_market, run_validation_suite
from aicq.strategy.signal_engine import analyze_symbol
from aicq.strategy.strategies import prepare_dataset
from aicq.timeutil import from_ms, index_to_ms, tf_ms, to_ms
from tests.helpers import make_df, raises

INST = Instrument("T", 0.01, 0.001, 0.001, 5.0, verified=True)
END = pd.Timestamp("2026-01-01", tz="UTC")
RC = RiskConfig(max_cost_to_risk=10, min_net_rr=0, max_daily_loss_pct=100, strategy_max_drawdown_pct=100)
P2 = StrategyParams(min_evidence_groups=2)


def random_walk(n, seed, tf="15m", sig=0.0012):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * sig
    close = 50_000 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[50_000], close[:-1]])
    w = np.abs(rng.standard_normal(n)) * sig * 0.5
    idx = from_ms(to_ms(END) - tf_ms(tf) * np.arange(n, 0, -1))
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + w),
                         "low": np.minimum(open_, close) * (1 - w), "close": close,
                         "volume": rng.lognormal(3, 0.5, n)}, index=idx)


# ---------------------------------------------------------------- 1. 時間解析度
def test_resample_and_ms_conversion_independent_of_datetime_resolution():
    base = generate_synthetic_ohlcv(1200, "15m", seed=21, end=END)
    ref = resample_ohlcv(base, "1h")
    assert len(ref) == 300
    for unit in ("ns", "us", "ms", "s"):
        b = base.copy()
        b.index = b.index.astype(f"datetime64[{unit}, UTC]")
        assert index_to_ms(b.index)[1] - index_to_ms(b.index)[0] == 900_000
        out = resample_ohlcv(b, "1h")
        assert len(out) == 300 and np.allclose(out["close"].to_numpy(), ref["close"].to_numpy()), unit


# ---------------------------------------------------------------- 2. 回測因果性
def test_engine_is_causal_future_data_cannot_change_past_trades():
    base = generate_synthetic_ohlcv(5000, "15m", seed=21, end=END)
    f, c = prepare_dataset(base, resample_ohlcv(base, "1h"), "15m", "1h", P2)
    r1 = run_backtest(f, c, INST, RC, CostConfig(), BacktestConfig(), "15m")
    checked = 0
    for k in (2000, 3500):
        b2 = base.copy()
        rng = np.random.default_rng(k)
        junk = base.iloc[k:].copy()
        m = rng.uniform(0.5, 1.5, len(junk))
        for col in ("open", "high", "low", "close"):
            junk[col] = junk[col].to_numpy() * m
        junk["high"] = junk[["open", "high", "low", "close"]].max(axis=1)
        junk["low"] = junk[["open", "high", "low", "close"]].min(axis=1)
        b2.iloc[k:] = junk.to_numpy()
        f2, c2 = prepare_dataset(b2, resample_ohlcv(b2, "1h"), "15m", "1h", P2)
        r2 = run_backtest(f2, c2, INST, RC, CostConfig(), BacktestConfig(), "15m")
        cut = base.index[k - 1]
        a = r1.trades[r1.trades.exit_time <= cut].reset_index(drop=True)
        b = r2.trades[r2.trades.exit_time <= cut].reset_index(drop=True)
        assert a.equals(b) and r1.equity.iloc[: k - 1].equals(r2.equity.iloc[: k - 1])
        checked += len(a)
    assert checked > 10        # 確實比對了足夠多筆交易，不是空比對


# ---------------------------------------------------------------- 3. 虛無假設：隨機漫步不得出現優勢
def test_null_random_walk_shows_no_phantom_edge_and_costs_hurt():
    rs, rs_cost = [], []
    for seed in range(12):
        b = random_walk(6000, seed)
        f, c = prepare_dataset(b, resample_ohlcv(b, "1h"), "15m", "1h", P2)
        rs += list(run_backtest(f, c, INST, RC, CostConfig(0, 0, 0), BacktestConfig(), "15m").trades.r_multiple)
        rs_cost += list(run_backtest(f, c, INST, RC, CostConfig(), BacktestConfig(), "15m").trades.r_multiple)
    rs, rs_cost = np.array(rs), np.array(rs_cost)
    assert len(rs) > 500
    t = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))
    assert abs(t) < 3.0, f"隨機漫步出現顯著優勢 t={t:.2f}，疑似洩漏或會計錯誤"
    assert rs_cost.mean() < rs.mean() - 0.05          # 加入成本後期望值必須明顯變差


# ---------------------------------------------------------------- 4/5. 帳務獨立重算與止損風險
def test_independent_accounting_recompute_and_stopout_within_budget():
    base = generate_synthetic_ohlcv(6000, "15m", seed=21, end=END)
    f, c = prepare_dataset(base, resample_ohlcv(base, "1h"), "15m", "1h", P2)
    cc = CostConfig()
    r = run_backtest(f, c, INST, RC, cc, BacktestConfig(), "15m")
    t = r.trades
    assert len(t) > 20
    assert np.allclose(cc.taker_fee_rate * t.qty * (t.entry_price + t.exit_price), t.fees)
    assert np.allclose(t.side * (t.exit_price - t.entry_price) * t.qty, t.gross_pnl)
    h8 = 8 * 3600 * 1000
    e = index_to_ms(pd.DatetimeIndex(t.entry_time))
    x = index_to_ms(pd.DatetimeIndex(t.exit_time)) + tf_ms("15m")
    assert np.allclose(t.side * t.qty * t.entry_price * cc.funding_rate_per_8h * (x // h8 - e // h8), t.funding)
    assert np.allclose(t.gross_pnl - t.fees - t.funding, t.net_pnl)
    assert abs(r.final_equity - (10_000 + t.net_pnl.sum())) < 1e-6
    s = t[t.exit_reason == "stop"]
    loss_pct = (-s.net_pnl / (s.equity_after - s.net_pnl) * 100)
    assert loss_pct.max() <= 0.25 + 1e-9               # 止損虧損不超過單筆風險預算 0.25%
    assert (s.r_multiple > -1.0 - 1e-9).all()


# ---------------------------------------------------------------- 6. 高週期資料離線／過期不得靜默使用
def _md(df, src="synthetic", ok=True, issues=None):
    return MarketData("BTC/USDT:USDT", "15m", df, src, DataQuality(ok=ok, issues=issues or [], n_rows=len(df)), "t")


def test_offline_or_insufficient_htf_blocks_executable_signals():
    cfg = replace(AppConfig(), risk=RiskConfig(min_net_rr=1.0, max_cost_to_risk=1.0), cost=CostConfig(0.0002, 1.0, 0.0001),
                  params=P2)
    inst = Instrument("BTC/USDT:USDT", 0.1, 0.001, 0.001, 100, verified=True)
    bdf = generate_synthetic_ohlcv(5000, "15m", seed=7, start_price=60_000.0, end=END)
    hdf = resample_ohlcv(bdf, "1h")
    ok_n = off_n = short_n = none_n = 0
    for k in range(0, 2500, 2):
        d = _md(bdf.iloc[: len(bdf) - k])
        acct = AccountState(1e4, 1e4)
        a = analyze_symbol("BTC/USDT:USDT", "15m", d, _md(hdf), inst, acct, cfg)
        b = analyze_symbol("BTC/USDT:USDT", "15m", d, _md(hdf, "offline", False, ["行情離線"]), inst, acct, cfg)
        c = analyze_symbol("BTC/USDT:USDT", "15m", d, _md(hdf.iloc[:20]), inst, acct, cfg)
        e = analyze_symbol("BTC/USDT:USDT", "15m", d, None, inst, acct, cfg)
        ok_n += a.direction != "flat"; off_n += b.direction != "flat"; short_n += c.direction != "flat"; none_n += e.direction != "flat"
    assert ok_n > 0                       # 高週期正常時確實會有訊號（否則本測試沒有意義）
    assert off_n == 0 and short_n == 0 and none_n == 0
    assert any("高週期" in r for r in b.rejection_reasons)


# ---------------------------------------------------------------- 7. 資料庫層重複／競態防護
def _rec(key, sym="T"):
    return dict(signal_key=key, symbol=sym, timeframe="15m", side=1, strategy="trend", qty=1, entry_price=100, stop=99,
                tp1=102, notional=100, margin=50, risk_amount=1, entry_fee=0, opened_at="x", opened_ms=0,
                eval_from_ms=0, max_hold_bars=96, status="open", reason="{}", data_source="s")


def test_db_blocks_same_symbol_duplicates_and_enforces_max_positions_atomically():
    s = Storage(os.path.join(tempfile.mkdtemp(), "r.db"))
    s.insert_position(_rec("a"), max_open=2)
    assert raises(sqlite3.IntegrityError, s.insert_position, _rec("b"), 2)       # 同商品第二筆（不同訊號）
    assert raises(sqlite3.IntegrityError, s.insert_position, _rec("a", "U"), 2)   # 同訊號重複
    s.insert_position(_rec("c", "U"), max_open=2)
    assert raises(PositionLimitReached, s.insert_position, _rec("d", "V"), 2)     # 第三筆超過上限
    assert len(s.positions("open")) == 2
    s.close_position(1, exit_price=1, exit_time="x", exit_reason="t", gross_pnl=0, fees=0, funding=0, slippage_cost=0, net_pnl=0)
    s.insert_position(_rec("e", "T"), max_open=2)                                  # 平倉後同商品可再進場
    assert len(s.positions("open")) == 2


# ---------------------------------------------------------------- 8. 過期訊號
CFGP = AppConfig(cost=CostConfig(0, 0, 0), risk=RiskConfig(max_daily_loss_pct=100))
INSTP = Instrument("T", 0.01, 0.01, 0.01, 1.0, verified=True)
NOW = pd.Timestamp("2026-03-01 00:00", tz="UTC")


def _sig(bar_open, key="k"):
    return TradeSignal(symbol="T", exchange="x", timeframe="15m", direction="long", direction_label="做多", signal_key=key,
                       entry_ref=100.0, stop=99.0, tp1=102.0, strategy="trend", data_source="synthetic",
                       bar_time=bar_open.isoformat() if bar_open is not None else "")


def test_stale_or_untimed_signal_rejected_fresh_accepted():
    def broker():
        return PaperBroker(CFGP, Storage(os.path.join(tempfile.mkdtemp(), "s.db")))
    ok, msg = broker().try_open(_sig(NOW - pd.Timedelta(hours=2)), INSTP, {}, DataGate(True), now=NOW)
    assert not ok and "過期" in msg
    ok, msg = broker().try_open(_sig(None), INSTP, {}, DataGate(True), now=NOW)
    assert not ok and "缺少" in msg
    ok, msg = broker().try_open(_sig(NOW - pd.Timedelta(minutes=15)), INSTP, {}, DataGate(True), now=NOW + pd.Timedelta(minutes=10))
    assert ok, msg


# ---------------------------------------------------------------- 9. 單日損失斷路器不可因程式晚啟動而失效
def test_daily_loss_limit_counts_losses_before_app_was_opened():
    cfg = AppConfig(cost=CostConfig(0, 0, 0), risk=RiskConfig(max_daily_loss_pct=1.0))
    b = PaperBroker(cfg, Storage(os.path.join(tempfile.mkdtemp(), "d.db")))
    now = pd.Timestamp("2026-03-02 10:00", tz="Asia/Taipei").tz_convert("UTC")
    early = pd.Timestamp("2026-03-02 02:00", tz="Asia/Taipei").tz_convert("UTC")
    b.db.insert_position({**_rec("loss"), "status": "closed", "net_pnl": -200.0, "exit_time": early.isoformat(), "exit_price": 98,
                          "exit_reason": "stop", "gross_pnl": -200, "fees": 0, "funding": 0, "slippage_cost": 0})
    st = b.account_state({}, now=now)           # 程式第一次在 10:00 才被開啟
    assert abs(st.daily_pnl_pct - (-2.0)) < 1e-9
    ok, msg = b.try_open(_sig(now - pd.Timedelta(minutes=15)), INSTP, {}, DataGate(True), now=now)
    assert not ok and "單日損失" in msg
    # 前一個台北日的虧損不得計入今天
    yday = pd.Timestamp("2026-03-01 20:00", tz="Asia/Taipei").tz_convert("UTC")
    b2 = PaperBroker(cfg, Storage(os.path.join(tempfile.mkdtemp(), "d2.db")))
    b2.db.insert_position({**_rec("old"), "status": "closed", "net_pnl": -500.0, "exit_time": yday.isoformat(), "exit_price": 1,
                           "exit_reason": "stop", "gross_pnl": -500, "fees": 0, "funding": 0, "slippage_cost": 0})
    assert abs(b2.account_state({}, now=now).daily_pnl_pct) < 1e-9


def test_equity_curve_and_drawdown_from_closed_trades():
    b = PaperBroker(CFGP, Storage(os.path.join(tempfile.mkdtemp(), "e.db")))
    for i, pnl in enumerate([100.0, -300.0, 50.0]):
        b.db.insert_position({**_rec(f"k{i}", f"S{i}"), "status": "closed", "net_pnl": pnl,
                              "exit_time": (NOW + pd.Timedelta(hours=i)).isoformat(), "exit_price": 1, "exit_reason": "t",
                              "gross_pnl": pnl, "fees": 0, "funding": 0, "slippage_cost": 0})
    assert list(b.equity_curve().round(2)) == [10_100.0, 9_800.0, 9_850.0]
    assert abs(b.max_drawdown_pct() - (300 / 10_100 * 100)) < 1e-9


# ---------------------------------------------------------------- 10/11. 選參邏輯
def test_walk_forward_never_trades_a_parameter_with_nonpositive_train_score():
    for seed in (3, 5):
        b = random_walk(9000, seed)
        wf = walk_forward(b, resample_ohlcv(b, "1h"), "15m", "1h", param_grid(P2), INST, RC, CostConfig(), BacktestConfig(),
                          train_bars=3000, test_bars=1500, min_train_trades=3)
        traded = wf.folds[wf.folds.oos_trades > 0]
        assert (traded.train_score > 0).all()


def test_three_way_selection_respects_train_and_validation_gates():
    b = generate_synthetic_ohlcv(9000, "15m", seed=9, end=END)
    h = resample_ohlcv(b, "1h")
    grid = param_grid(P2)
    data = [prepare_dataset(b, h, "15m", "1h", p) for p in grid]
    for mt, mv in ((20, 10), (1, 1)):
        s = three_way_selection(data, grid, b, INST, RC, CostConfig(), BacktestConfig(), "15m", min_train_trades=mt, min_val_trades=mv)
        assert s["n_trials"] == 9 and len(s["table"]) == 9 and s["message"]
        if s["chosen"] is None:
            assert s["test"] is None
        else:
            assert s["train_metrics"]["expectancy_r"] > 0
            assert (s["test"] is not None) == s["validation_passed"]      # 樣本外只在通過驗證後才被碰
    none = three_way_selection(data, grid, b, INST, RC, CostConfig(), BacktestConfig(), "15m", min_train_trades=10**6)
    assert none["chosen"] is None and none["test"] is None


# ---------------------------------------------------------------- 12. 樣本外統計判定
def _trades(r):
    return pd.DataFrame({"r_multiple": r})


def test_oos_verdict_requires_enough_trades_and_ci_above_zero():
    rng = np.random.default_rng(0)
    good = _trades(rng.normal(0.4, 0.8, 200))
    m = {"n_trades": 200, "profit_factor": 1.5, "net_return_pct": 5.0, "max_drawdown_pct": 3.0}
    assert oos_verdict(good, m)["status"].startswith("通過")
    assert oos_verdict(_trades([]), {"n_trades": 0})["status"] == "未驗證"
    few = _trades([1.0, 2.0, 1.5, 0.5, 1.0])
    assert oos_verdict(few, {"n_trades": 5, "profit_factor": 9, "net_return_pct": 1, "max_drawdown_pct": 1})["status"] == "證據不足"
    noise = _trades(rng.normal(0.02, 1.0, 60))          # 平均略正但信賴區間含 0
    assert oos_verdict(noise, {"n_trades": 60, "profit_factor": 1.05, "net_return_pct": 1, "max_drawdown_pct": 1})["status"] == "未通過"
    assert oos_verdict(good, {**m, "max_drawdown_pct": 25.0})["status"] == "未通過"
    ci = expectancy_ci(good)
    assert ci["lo"] < ci["mean_r"] < ci["hi"]


# ---------------------------------------------------------------- 13. 回測紀錄與可重現
def test_backtest_runs_are_persisted_with_data_fingerprint_and_reproducible():
    cfg = AppConfig()
    base, htf, inst = load_market(cfg, "BTC/USDT:USDT", "15m", "synthetic", 4000)
    stg = Storage(os.path.join(tempfile.mkdtemp(), "b.db"))
    o1 = run_validation_suite(cfg, base, htf, "15m", inst, P2, BacktestConfig(), storage=stg)
    o2 = run_validation_suite(cfg, base, htf, "15m", inst, P2, BacktestConfig(), storage=stg)
    assert o1["fingerprint"] == o2["fingerprint"] == data_fingerprint(base.df)
    assert o2["n_backtests_so_far"] == 2 and len(stg.backtest_runs()) == 2
    pd.testing.assert_frame_equal(o1["full"].trades, o2["full"].trades)       # 同資料同設定 → 同結果
    shifted = base.df.copy(); shifted["close"] = shifted["close"] * 1.0001
    assert data_fingerprint(shifted) != o1["fingerprint"]
    row = stg.backtest_runs().iloc[0]
    assert "data_fingerprint" in row["params"] and "oos_default" in row["metrics"]


# ---------------------------------------------------------------- 14. 跳空越過強平價
def test_gap_through_liquidation_price_exits_at_liquidation_not_beyond():
    rows = [(100, 100.2, 99.9, 100), (100, 100.1, 99.95, 100), (0.3, 0.4, 0.2, 0.3)] + [(0.3, 0.4, 0.2, 0.3)] * 3
    df = make_df(rows)
    cand = pd.DataFrame({"side": 0, "raw_side": 0, "strategy": "trend", "stop_dist": 1.0, "tp1_dist": 2.0, "evidence": 4,
                         "conflict": False}, index=df.index)
    cand.iloc[0, cand.columns.get_loc("side")] = 1
    f = df.copy(); f["regime"] = "trend"
    r = run_backtest(f, cand, INST, RiskConfig(max_daily_loss_pct=100), CostConfig(0, 0, 0), BacktestConfig(), "15m")
    t = r.trades.iloc[0]
    assert t.exit_reason == "liquidated"
    liq = 100 * 0.005          # 1 倍槓桿：entry*(1-1+mmr)
    assert abs(t.exit_price - liq) < 1e-6
    assert t.net_pnl > -(t.qty * 100) - 1e-6      # 損失不超過全部保證金（逐倉）
