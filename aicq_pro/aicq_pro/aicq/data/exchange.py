"""交易所公開行情封裝（CCXT）。僅使用公開端點，不需要 API 金鑰。

重點：限流（enableRateLimit）、逾時、重試退避、分頁、缺資料偵測、商品規格解析。
本模組在沙盒中無法連網，真實連線路徑『尚未實測』；分頁／重試／規格解析以假交易所單元測試。
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import pandas as pd

from ..models import Instrument
from ..timeutil import from_ms, tf_ms

# ccxt 的 precisionMode 常數值
DECIMAL_PLACES, SIGNIFICANT_DIGITS, TICK_SIZE = 2, 3, 4


class ExchangeUnavailable(RuntimeError):
    """無法使用交易所（未安裝 ccxt、斷線、重試後仍失敗）。"""


class ExchangeClient:
    def __init__(self, exchange_id: str = "binanceusdm", ccxt_module=None, sleep: Callable = time.sleep,
                 max_retries: int = 3, network_errors: Optional[tuple] = None, ex=None):
        self.exchange_id = exchange_id
        self.sleep = sleep
        self.max_retries = max_retries
        if ex is not None:                      # 測試用：直接注入假交易所
            self.ex = ex
            self.network_errors = network_errors or (ConnectionError, TimeoutError)
            return
        try:
            ccxt = ccxt_module or __import__("ccxt")
        except ImportError as e:
            raise ExchangeUnavailable("未安裝 ccxt（請執行 pip install -r requirements.txt）") from e
        try:
            self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15_000})
        except AttributeError as e:
            raise ExchangeUnavailable(f"ccxt 不支援交易所：{exchange_id}") from e
        self.network_errors = network_errors or (ccxt.NetworkError,)
        self._ccxt = ccxt

    # -- 重試 -----------------------------------------------------------
    def _call(self, fn, *args, **kwargs):
        delay = 1.0
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except self.network_errors as e:    # 逾時、限流、斷線
                last = e
                if attempt == self.max_retries:
                    break
                self.sleep(delay)
                delay *= 2
            except Exception as e:              # 非網路錯誤（參數、商品不存在等）不重試
                raise ExchangeUnavailable(f"交易所錯誤：{type(e).__name__}: {e}") from e
        raise ExchangeUnavailable(f"網路錯誤，重試 {self.max_retries} 次後仍失敗：{type(last).__name__}: {last}")

    # -- K 線 -----------------------------------------------------------
    def fetch_ohlcv_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int,
                          page_limit: int = 1000, max_pages: int = 500) -> pd.DataFrame:
        step = tf_ms(timeframe)
        rows: list = []
        cursor = since_ms
        for _ in range(max_pages):
            page = self._call(self.ex.fetch_ohlcv, symbol, timeframe, since=cursor, limit=page_limit)
            if not page:
                break
            rows.extend(page)
            last_ts = page[-1][0]
            if last_ts >= until_ms - step or last_ts < cursor:
                break
            cursor = last_ts + step
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                                index=pd.DatetimeIndex([], tz="UTC"))
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").sort_values("ts")
        df = df[df["ts"] <= until_ms]
        df.index = from_ms(df.pop("ts").to_numpy())
        df.index.name = "ts"
        return df.astype(float)

    # -- 商品規格 ---------------------------------------------------------
    def load_instrument(self, symbol: str) -> Instrument:
        self._call(self.ex.load_markets)
        market = self.ex.market(symbol)
        mode = getattr(self.ex, "precisionMode", TICK_SIZE)
        return instrument_from_market(market, mode)


def _step_from_precision(value, mode) -> Optional[float]:
    if value is None:
        return None
    if mode == TICK_SIZE:
        return float(value)
    if mode == DECIMAL_PLACES:
        return 10.0 ** (-int(value))
    return None  # SIGNIFICANT_DIGITS：無法可靠換算


def instrument_from_market(market: dict, precision_mode: int = TICK_SIZE) -> Instrument:
    prec = market.get("precision", {}) or {}
    limits = market.get("limits", {}) or {}
    tick = _step_from_precision(prec.get("price"), precision_mode)
    step = _step_from_precision(prec.get("amount"), precision_mode)
    min_amt = (limits.get("amount") or {}).get("min")
    min_cost = (limits.get("cost") or {}).get("min")
    contract = market.get("contractSize") or 1.0
    notes = []
    if tick is None or step is None:
        notes.append("精度模式無法可靠換算，已使用保守預設")
    if contract != 1:
        notes.append(f"contractSize={contract}（非 1，數量換算未支援，請勿用於下單）")
    verified = tick is not None and step is not None and contract == 1
    return Instrument(
        symbol=market["symbol"],
        price_tick=tick or 0.01,
        amount_step=step or 0.001,
        min_amount=float(min_amt) if min_amt else (step or 0.001),
        min_notional=float(min_cost) if min_cost else 5.0,
        contract_size=float(contract),
        verified=verified,
        note="；".join(notes) or "規格來自交易所 API",
    )
