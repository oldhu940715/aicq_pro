import numpy as np
import pandas as pd
import tempfile, os

from aicq.data.cache import OhlcvCache
from aicq.data.exchange import ExchangeClient, ExchangeUnavailable, instrument_from_market, TICK_SIZE, DECIMAL_PLACES
from aicq.data.synthetic import generate_synthetic_ohlcv
from aicq.data.validation import validate_ohlcv, drop_incomplete_last_bar
from aicq.timeutil import from_ms, to_ms
from tests.helpers import make_df, raises

NOW = pd.Timestamp("2026-01-01 00:00", tz="UTC")


def _good(n=300, tf="15m"):
    return generate_synthetic_ohlcv(n, tf, seed=1, end=NOW)


def test_synthetic_deterministic_and_valid():
    a, b = _good(), _good()
    assert a.equals(b)
    q = validate_ohlcv(a, "15m", now=NOW)
    assert q.ok, q.issues


def test_validation_detects_gap_duplicate_invalid_stale():
    df = _good()
    q = validate_ohlcv(df.drop(df.index[-50]), "15m", now=NOW)
    assert not q.ok and q.n_missing_bars == 1
    dup = pd.concat([df, df.iloc[[-1]]])
    assert not validate_ohlcv(dup, "15m", now=NOW).ok
    bad = df.copy(); bad.iloc[-5, bad.columns.get_loc("high")] = bad.iloc[-5]["low"] - 1
    assert not validate_ohlcv(bad, "15m", now=NOW).ok
    stale = validate_ohlcv(df, "15m", now=NOW + pd.Timedelta(hours=3))
    assert not stale.ok and not stale.fresh
    assert not validate_ohlcv(df.iloc[:0], "15m", now=NOW).ok


def test_old_gap_is_warning_not_blocking():
    df = _good(600)
    q = validate_ohlcv(df.drop(df.index[10]), "15m", now=NOW)
    assert q.ok and q.n_missing_bars == 1 and q.warnings


def test_drop_incomplete_bar():
    df = _good()
    forming = df.index[-1] + pd.Timedelta(minutes=15)
    ext = pd.concat([df, df.iloc[[-1]].set_axis([forming])])
    out = drop_incomplete_last_bar(ext, "15m", now=forming + pd.Timedelta(minutes=5))
    assert len(out) == len(df)


class FakeNetErr(Exception):
    pass


class FakeEx:
    precisionMode = TICK_SIZE

    def __init__(self, data, fail_first=0, always_fail=False):
        self.data, self.fails, self.always = data, fail_first, always_fail
        self.calls = 0

    def fetch_ohlcv(self, symbol, tf, since=None, limit=1000):
        self.calls += 1
        if self.always or self.fails > 0:
            self.fails -= 1
            raise FakeNetErr("timeout")
        return [r for r in self.data if r[0] >= since][:limit]


def _rows(n, step=900_000, start=1_700_000_000_000 // 900_000 * 900_000):
    return [[start + i * step, 1, 2, 0.5, 1.5, 10] for i in range(n)]


def test_exchange_pagination_and_retry():
    rows = _rows(2500)
    ex = FakeEx(rows, fail_first=2)
    sleeps = []
    cl = ExchangeClient(ex=ex, sleep=sleeps.append, network_errors=(FakeNetErr,))
    df = cl.fetch_ohlcv_range("X", "15m", rows[0][0], rows[-1][0] + 900_000)
    assert len(df) == 2500 and df.index.is_unique and df.index.is_monotonic_increasing
    assert sleeps == [1.0, 2.0]   # 指數退避


def test_exchange_gives_up_after_retries():
    cl = ExchangeClient(ex=FakeEx([], always_fail=True), sleep=lambda s: None, network_errors=(FakeNetErr,), max_retries=2)
    assert raises(ExchangeUnavailable, cl.fetch_ohlcv_range, "X", "15m", 0, 10**12)


def test_instrument_parsing():
    m = {"symbol": "BTC/USDT:USDT", "precision": {"price": 0.1, "amount": 0.001},
         "limits": {"amount": {"min": 0.001}, "cost": {"min": 100}}, "contractSize": 1}
    i = instrument_from_market(m, TICK_SIZE)
    assert i.verified and i.price_tick == 0.1 and i.amount_step == 0.001 and i.min_notional == 100
    m2 = dict(m, precision={"price": 2, "amount": 3})
    i2 = instrument_from_market(m2, DECIMAL_PLACES)
    assert abs(i2.price_tick - 0.01) < 1e-12 and abs(i2.amount_step - 0.001) < 1e-12
    i3 = instrument_from_market(dict(m, contractSize=10), TICK_SIZE)
    assert not i3.verified


def test_cache_roundtrip_idempotent():
    with tempfile.TemporaryDirectory() as d:
        c = OhlcvCache(os.path.join(d, "c.db"))
        df = _good(100)
        c.upsert("ex", "S", "15m", df); c.upsert("ex", "S", "15m", df)
        out = c.load("ex", "S", "15m")
        assert len(out) == 100 and np.allclose(out["close"], df["close"])
        assert list(out.index) == list(df.index)
        assert len(c.load("ex", "S", "5m")) == 0
