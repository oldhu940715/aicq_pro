import numpy as np
import pandas as pd

from aicq.timeutil import from_ms


def make_df(rows, tf_ms_=900_000, start_ms=1_700_000_000_000 // 900_000 * 900_000):
    """rows: list of (o,h,l,c[,v])"""
    arr = np.array([r if len(r) == 5 else (*r, 100.0) for r in rows], dtype=float)
    idx = from_ms(start_ms + tf_ms_ * np.arange(len(arr)))
    return pd.DataFrame(arr, columns=["open", "high", "low", "close", "volume"], index=idx)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False
