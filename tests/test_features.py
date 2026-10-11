import numpy as np
import pandas as pd

from aicq.config import StrategyParams
from aicq.data.synthetic import generate_synthetic_ohlcv
from aicq.features import indicators as ind
from aicq.features.pipeline import add_features, align_htf, compute_htf_trend
from aicq.strategy.strategies import prepare_dataset
from tests.helpers import make_df, approx

END = pd.Timestamp("2026-01-01", tz="UTC")


def test_indicator_basics():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert ind.rsi(up, 14).iloc[-1] == 100.0
    assert ind.rsi(-up + 200, 14).iloc[-1] < 1e-6
    flat = pd.Series(np.full(60, 5.0))
    assert approx(ind.ema(flat, 10).iloc[-1], 5.0)
    df = generate_synthetic_ohlcv(400, "15m", end=END)
    a, x = ind.atr(df), ind.adx(df)
    assert (a.dropna() > 0).all() and x.dropna().between(0, 100).all()
    one = make_df([(10, 12, 9, 11, 5)])
    assert approx(ind.session_vwap(one).iloc[0], (12 + 9 + 11) / 3)


def test_no_lookahead_features_and_candidates():
    """用截斷到第 k 根的資料算出的結果，必須與用完整資料算出的前 k 根完全一致。"""
    base = generate_synthetic_ohlcv(1500, "15m", seed=3, end=END)
    htf = generate_synthetic_ohlcv(600, "1h", seed=3, end=END)
    p = StrategyParams()
    f_full, c_full = prepare_dataset(base, htf, "15m", "1h", p)
    for k in (700, 1100):
        f_cut, c_cut = prepare_dataset(base.iloc[:k], htf, "15m", "1h", p)
        pd.testing.assert_frame_equal(f_full.iloc[:k].drop(columns="regime"), f_cut.drop(columns="regime"))
        assert (f_full["regime"].iloc[:k].to_numpy() == f_cut["regime"].to_numpy()).all()
        pd.testing.assert_frame_equal(c_full.iloc[:k], c_cut)


def test_htf_alignment_uses_only_closed_bars():
    htf = make_df([(1, 1, 1, 1)] * 5, tf_ms_=3_600_000, start_ms=3_600_000 * 1000)
    s = pd.Series([1, -1, 1, -1, 1], index=htf.index)
    t0 = htf.index[1]                                    # 第 2 根 1h K 線開盤
    base_idx = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=m) for m in (0, 15, 30, 45)])
    out = align_htf(base_idx, "15m", s, "1h")
    # 第 2 根 1h 在 t0+1h 才收盤；t0+45m 的 15m 棒收盤於 t0+60m → 此時才剛好可用
    assert list(out.iloc[:3]) == [1, 1, 1]        # 仍只能用第 1 根(=1)
    assert out.iloc[3] == -1                      # 第 2 根(=-1)剛收盤可用


def test_breakout_level_is_previous_bars_only_and_breakout_can_trigger():
    """突破位必須是『之前』N 根的高低點（不含當根）；否則收盤價永遠不可能高於包含自己最高價的區間。"""
    p = StrategyParams(breakout_lookback=20)
    rows = [(100, 100.5, 99.5, 100.0, 100.0)] * 25 + [(100, 106, 100, 105.5, 400.0)]
    df = make_df(rows)
    f = add_features(df, p)
    last = f.iloc[-1]
    assert approx(last.don_hi, 100.5) and approx(last.don_lo, 99.5)          # 不含當根的 106
    assert last.close > last.don_hi                                            # 突破可以成立
    assert np.isnan(f["don_hi"].iloc[19]) and not np.isnan(f["don_hi"].iloc[20])   # 暖機長度正確
    # 當根的量比使用『之前』均量，不含當根
    assert approx(last.vol_ratio, 400.0 / 100.0)
