"""行情提供者：明確區分 live / synthetic（測試資料）/ offline。絕不在即時模式失敗時偷偷改用假資料。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..config import AppConfig
from ..models import Instrument, default_instrument
from ..timeutil import now_utc, tf_ms, to_ms
from .cache import OhlcvCache
from .exchange import ExchangeClient, ExchangeUnavailable
from .synthetic import generate_synthetic_ohlcv
from .validation import DataQuality, drop_incomplete_last_bar, validate_ohlcv


@dataclass
class MarketData:
    symbol: str
    timeframe: str
    df: pd.DataFrame
    source: str                     # live / synthetic / offline
    quality: Optional[DataQuality]
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.source != "offline" and self.quality is not None and self.quality.ok

    @property
    def is_test_data(self) -> bool:
        return self.source == "synthetic"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz="UTC"))


def load_synthetic(symbol: str, timeframe: str, bars: int, cfg: AppConfig, seed: int = 7) -> MarketData:
    start = 60_000.0 if symbol.startswith("BTC") else 3_000.0
    df = generate_synthetic_ohlcv(bars, timeframe, seed=seed + (0 if symbol.startswith("BTC") else 1000),
                                  start_price=start)
    q = validate_ohlcv(df, timeframe, max_age_bars=cfg.risk.max_data_age_bars)
    return MarketData(symbol, timeframe, df, "synthetic", q, "⚠ 合成測試資料，非真實行情")


def load_live(symbol: str, timeframe: str, bars: int, cfg: AppConfig, cache: OhlcvCache,
              client: Optional[ExchangeClient] = None, now=None) -> MarketData:
    now = pd.Timestamp(now) if now is not None else now_utc()
    try:
        client = client or ExchangeClient(cfg.exchange_id)
        step = tf_ms(timeframe)
        now_ms = to_ms(now)
        want_from = now_ms - bars * step
        cached = cache.load(cfg.exchange_id, symbol, timeframe, since_ms=want_from)
        if len(cached) and to_ms(cached.index[0]) <= want_from + step:
            since = max(want_from, to_ms(cached.index[-1]) - 2 * step)   # 只補最新缺的部分
        else:
            since = want_from
        fresh = client.fetch_ohlcv_range(symbol, timeframe, since, now_ms)
        fresh = drop_incomplete_last_bar(fresh, timeframe, now)
        cache.upsert(cfg.exchange_id, symbol, timeframe, fresh)
        df = cache.load(cfg.exchange_id, symbol, timeframe, since_ms=want_from)
        df = drop_incomplete_last_bar(df, timeframe, now)
    except ExchangeUnavailable as e:
        # 離線：若快取有資料仍可顯示，但標示為離線、不產生訊號
        df = cache.load(cfg.exchange_id, symbol, timeframe, limit=bars)
        q = validate_ohlcv(df, timeframe, now=now, max_age_bars=cfg.risk.max_data_age_bars)
        q.issues.append(f"行情離線：{e}")
        q.ok = False
        return MarketData(symbol, timeframe, df, "offline", q, f"離線：{e}")
    q = validate_ohlcv(df, timeframe, now=now, max_age_bars=cfg.risk.max_data_age_bars)
    return MarketData(symbol, timeframe, df, "live", q, "即時行情" if q.ok else "行情品質檢查未通過")


def load_instrument(symbol: str, cfg: AppConfig, mode: str, client: Optional[ExchangeClient] = None) -> Instrument:
    if mode == "live":
        try:
            return (client or ExchangeClient(cfg.exchange_id)).load_instrument(symbol)
        except Exception:
            pass
    return default_instrument(symbol)
