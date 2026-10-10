import numpy as np
import pandas as pd

from aicq.backtest.engine import run_backtest
from aicq.backtest.fills import bar_exit, funding_cost
from aicq.backtest.metrics import compute_metrics, max_drawdown_pct
from aicq.backtest.validation import chronological_split, param_grid, regime_breakdown, walk_forward, cost_sensitivity
from aicq.config import BacktestConfig, CostConfig, RiskConfig, StrategyParams
from aicq.data.synthetic import generate_synthetic_ohlcv
from aicq.models import Instrument
from aicq.strategy.strategies import prepare_dataset
from tests.helpers import make_df, approx

INST = Instrument("T", 0.01, 0.01, 0.01, 1.0, verified=True)
ZERO = CostConfig(0.0, 0.0, 0.0)
END = pd.Timestamp("2026-01-01", tz="UTC")


def _cand(df, entries, stop_dist=1.0, tp_dist=2.0, side=1):
    n = len(df)
    c = pd.DataFrame({"side": 0, "raw_side": 0, "strategy": "trend", "stop_dist": stop_dist, "tp1_dist": tp_dist,
                      "evidence": 4, "conflict": False}, index=df.index)
    for i in entries:
        c.iloc[i, c.columns.get_loc("side")] = side
    f = df.copy(); f["regime"] = "trend"
    return f, c


def test_entry_is_next_bar_open_not_signal_bar_close():
    df = make_df([(100, 100.5, 99.5, 100), (105, 105.5, 104.5, 105), (105, 107.5, 104.5, 107)] + [(107, 107.5, 106.5, 107)] * 3)
    f, c = _cand(df, [0], stop_dist=1.0, tp_dist=2.0)
    r = run_backtest(f, c, INST, RiskConfig(), ZERO, BacktestConfig(), "15m")
    t = r.trades.iloc[0]
    assert t.entry_price == 105.0                     # 第 1 根開盤，而非訊號棒收盤 100
    assert t.exit_reason == "tp" and approx(t.exit_price, 107.0)


def test_same_bar_stop_and_tp_is_conservative_stop_first():
    r = bar_exit(1, 99.0, 102.0, 100.0, 103.0, 98.0, 0.0)
    assert r[2] == "stop" and r[3] is True
    r2 = bar_exit(-1, 101.0, 98.0, 100.0, 102.0, 97.0, 0.0)
    assert r2[2] == "stop" and r2[3] is True


def test_gap_through_stop_fills_at_open_and_slippage_adverse():
    r = bar_exit(1, 99.0, 105.0, 97.0, 98.0, 96.0, 0.001)
    assert r[2] == "stop_gap" and r[1] == 97.0 and r[0] < 97.0
    s = bar_exit(-1, 101.0, 95.0, 100.0, 100.5, 99.0, 0.001)
    assert s is None
    s2 = bar_exit(-1, 101.0, 95.0, 100.0, 101.5, 99.0, 0.001)
    assert s2[0] > 101.0


def test_funding_settlement_count():
    h8 = 8 * 3600 * 1000
    assert funding_cost(1, 1000, 1 * h8 - 1, 2 * h8 - 1, 0.0001) == 1000 * 0.0001 * 1    # 只跨過 8h 這一次結算
    assert funding_cost(1, 1000, 1 * h8 - 1, 2 * h8, 0.0001) == 1000 * 0.0001 * 2        # 8h 與 16h 兩次
    assert funding_cost(1, 1000, 1 * h8, 2 * h8 - 1, 0.0001) == 0                    # 未跨過
    assert funding_cost(-1, 1000, 0, 3 * h8, 0.0001) == -1000 * 0.0001 * 3            # 空單收取
    ts, rates = np.array([5, 10, 15]), np.array([0.001, 0.002, -0.001])
    assert approx(funding_cost(1, 100, 4, 12, 0.0, (ts, rates)), 100 * 0.003)


def test_costs_reduce_pnl_and_equity_conserved():
    base = generate_synthetic_ohlcv(6000, "15m", seed=11, end=END)
    htf = generate_synthetic_ohlcv(1600, "1h", seed=11, end=END)
    p = StrategyParams(min_evidence_groups=2)
    f, c = prepare_dataset(base, htf, "15m", "1h", p)
    free = run_backtest(f, c, INST, RiskConfig(max_cost_to_risk=10, min_net_rr=0), ZERO, BacktestConfig(), "15m")
    paid = run_backtest(f, c, INST, RiskConfig(max_cost_to_risk=10, min_net_rr=0), CostConfig(), BacktestConfig(), "15m")
    assert len(free.trades) > 3
    assert free.trades.fees.sum() == 0 and paid.trades.fees.sum() > 0
    assert paid.final_equity < free.final_equity or len(paid.trades) != len(free.trades)
    for r in (free, paid):   # 淨值守恆：期末淨值 = 期初 + 各筆淨損益
        assert approx(r.final_equity, r.initial_equity + r.trades.net_pnl.sum(), 1e-9)
        assert not r.equity.isna().any()


def test_backtest_reproducible():
    base = generate_synthetic_ohlcv(4000, "15m", seed=5, end=END)
    htf = generate_synthetic_ohlcv(1100, "1h", seed=5, end=END)
    f, c = prepare_dataset(base, htf, "15m", "1h", StrategyParams(min_evidence_groups=2))
    a = run_backtest(f, c, INST, RiskConfig(), CostConfig(), BacktestConfig(), "15m")
    b = run_backtest(f, c, INST, RiskConfig(), CostConfig(), BacktestConfig(), "15m")
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert a.equity.equals(b.equity)


def test_daily_loss_limit_stops_entries_same_day():
    # 兩根連續止損：第一筆虧 ~0.25%，把日限額設 0.2% → 同日第二筆應被擋
    rows = [(100, 100.2, 99.9, 100), (100, 100.1, 98.0, 98.5), (98.5, 98.6, 98.4, 98.5), (98.5, 98.6, 98.4, 98.5),
            (98.5, 99, 97, 98), (98, 98.2, 97.9, 98)]
    df = make_df(rows, start_ms=1_700_006_400_000 // 86_400_000 * 86_400_000 - 8 * 3600_000 + 3600_000)  # 台北日初附近
    f, c = _cand(df, [0, 3])
    r = run_backtest(f, c, INST, RiskConfig(max_daily_loss_pct=0.2), ZERO, BacktestConfig(), "15m")
    assert len(r.trades) == 1 and any("單日損失" in k for k in r.blocked)


def test_strategy_drawdown_halts_new_entries():
    n = 200
    rows = []
    for i in range(n):
        rows.append((100, 100.1, 98.0, 99) if i % 2 == 1 else (100, 100.1, 99.9, 100))
    df = make_df(rows)
    f, c = _cand(df, list(range(0, n, 2)))
    r = run_backtest(f, c, INST, RiskConfig(strategy_max_drawdown_pct=0.5, risk_per_trade_pct=0.25, max_daily_loss_pct=100), ZERO, BacktestConfig(), "15m")
    assert r.halted_at is not None and len(r.trades) < 100


def test_max_drawdown_and_metrics_keys():
    assert approx(max_drawdown_pct(pd.Series([100, 120, 90, 110.0]), 100), 25.0)
    base = generate_synthetic_ohlcv(5000, "15m", seed=2, end=END)
    htf = generate_synthetic_ohlcv(1300, "1h", seed=2, end=END)
    f, c = prepare_dataset(base, htf, "15m", "1h", StrategyParams(min_evidence_groups=2))
    r = run_backtest(f, c, INST, RiskConfig(max_cost_to_risk=10, min_net_rr=0), CostConfig(), BacktestConfig(), "15m")
    m = compute_metrics(r, base, CostConfig())
    for k in ("net_return_pct", "max_drawdown_pct", "win_rate_pct", "avg_win", "avg_loss", "expectancy_usdt", "profit_factor",
              "sharpe_daily_annualized(√365,rf=0)", "sharpe_per_trade(非年化)", "n_trades", "avg_hold_minutes",
              "total_fees", "total_funding", "buy_hold_return_pct"):
        assert k in m, k
    assert approx(m["buy_hold_return_pct"], (base.close.iloc[-1] / base.open.iloc[0] - 1) * 100)
    assert not regime_breakdown(r.trades).empty


def test_split_chronological_no_overlap_and_order():
    a, b, c = chronological_split(1000, 0.5, 0.2, embargo=10)
    assert a.stop <= b.start and b.stop <= c.start and a.start == 0 and c.stop == 1000


def test_walk_forward_folds_are_forward_only():
    base = generate_synthetic_ohlcv(9000, "15m", seed=4, end=END)
    htf = generate_synthetic_ohlcv(2300, "1h", seed=4, end=END)
    wf = walk_forward(base, htf, "15m", "1h", param_grid(StrategyParams(min_evidence_groups=2))[:3], INST,
                      RiskConfig(max_cost_to_risk=10, min_net_rr=0), CostConfig(), BacktestConfig(),
                      train_bars=3000, test_bars=1500, min_train_trades=1)
    f = wf.folds
    assert len(f) >= 3
    assert (f.test_start.iloc[1:].to_numpy() > f.test_end.iloc[:-1].to_numpy()).all()   # 測試窗不重疊、往前
    assert (f.test_start > f.train_start).all()
    if len(wf.oos_trades):
        for _, row in f.iterrows():
            tr = wf.oos_trades[wf.oos_trades.fold == row.fold]
            assert ((tr.entry_time >= row.test_start) & (tr.exit_time <= row.test_end)).all()   # 樣本外交易全落在測試窗


def test_cost_sensitivity_shape():
    base = generate_synthetic_ohlcv(3000, "15m", seed=6, end=END)
    htf = generate_synthetic_ohlcv(800, "1h", seed=6, end=END)
    f, c = prepare_dataset(base, htf, "15m", "1h", StrategyParams(min_evidence_groups=2))
    t = cost_sensitivity(f, c, INST, RiskConfig(max_cost_to_risk=10, min_net_rr=0), CostConfig(), BacktestConfig(), "15m")
    assert len(t) == 9
