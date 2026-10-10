"""K 線內成交判定（回測與模擬交易共用，確保兩者行為一致）。"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..timeutil import EIGHT_HOURS_MS


def bar_exit(side: int, stop: float, tp: float, o: float, h: float, l: float, slip: float):
    """判斷一根 K 線內是否觸及止損／止盈。回傳 (成交價, 未含滑價的原始價, 原因, 是否同K線雙觸) 或 None。

    K 線內價格先後順序無法從 OHLC 得知：若同一根同時觸及止損與止盈，保守判定為『先止損』。
    開盤跳空越過止損時，以開盤價成交（比止損價更差）。止損為市價單，加上不利滑價；止盈視為限價單，無滑價。
    """
    if side == 1:
        gap, stop_hit, tp_hit = o <= stop, l <= stop, h >= tp
    else:
        gap, stop_hit, tp_hit = o >= stop, h >= stop, l <= tp
    if gap:
        raw, reason, amb = o, "stop_gap", False
    elif stop_hit:
        raw, reason, amb = stop, "stop", bool(tp_hit)
    elif tp_hit:
        return tp, tp, "tp", False
    else:
        return None
    return raw * (1 - side * slip), raw, reason, amb


def funding_cost(side: int, notional: float, entry_ms: int, exit_ms: int, rate_8h: float,
                 series: Optional[tuple] = None) -> float:
    """資金費用（正值＝支付）。固定費率：以經過的 00/08/16 UTC 結算次數計；或傳入 (ms 陣列, 費率陣列)。"""
    if exit_ms <= entry_ms:
        return 0.0
    if series is not None:
        ts, rates = series
        a, b = np.searchsorted(ts, entry_ms, side="right"), np.searchsorted(ts, exit_ms, side="right")
        return float(side * notional * np.sum(rates[a:b]))
    count = exit_ms // EIGHT_HOURS_MS - entry_ms // EIGHT_HOURS_MS
    return float(side * notional * rate_8h * count)
