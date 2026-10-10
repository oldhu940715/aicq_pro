"""共用資料模型。side：1=做多，-1=做空。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Instrument:
    symbol: str
    price_tick: float = 0.01
    amount_step: float = 0.001
    min_amount: float = 0.001
    min_notional: float = 20.0
    contract_size: float = 1.0
    verified: bool = False        # True 代表規格來自交易所 API
    note: str = "預設規格（未向交易所驗證），請以交易所實際規格為準"


# 未驗證的預設值，僅供離線/測試資料使用；連線成功時會以 ccxt 市場資料覆蓋。
DEFAULT_INSTRUMENTS = {
    "BTC/USDT:USDT": Instrument("BTC/USDT:USDT", 0.1, 0.001, 0.001, 100.0),
    "ETH/USDT:USDT": Instrument("ETH/USDT:USDT", 0.01, 0.001, 0.001, 20.0),
}


def default_instrument(symbol: str) -> Instrument:
    return DEFAULT_INSTRUMENTS.get(symbol, Instrument(symbol=symbol))


@dataclass
class OpenPosition:
    symbol: str
    side: int
    qty: float
    entry_price: float
    notional: float
    margin: float
    strategy: str = ""


@dataclass
class AccountState:
    equity: float
    day_start_equity: float
    open_positions: list = field(default_factory=list)
    halted_strategies: set = field(default_factory=set)
    executed_signal_keys: set = field(default_factory=set)

    @property
    def used_margin(self) -> float:
        return sum(p.margin for p in self.open_positions)

    @property
    def total_notional(self) -> float:
        return sum(p.notional for p in self.open_positions)

    @property
    def free_margin(self) -> float:
        return self.equity - self.used_margin

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.equity - self.day_start_equity) / self.day_start_equity * 100.0


@dataclass
class DataGate:
    ok: bool
    reasons: list = field(default_factory=list)
    is_test_data: bool = False


@dataclass
class TradeRequest:
    symbol: str
    side: int
    entry: float
    stop: float
    tp1: float
    strategy: str = ""
    signal_key: str = ""
    est_hold_hours: float = 4.0


@dataclass
class PositionPlan:
    allowed: bool = False
    reasons: list = field(default_factory=list)    # 阻擋原因
    warnings: list = field(default_factory=list)
    side: int = 0
    qty: float = 0.0
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    notional: float = 0.0
    leverage: float = 0.0
    margin: float = 0.0
    risk_amount: float = 0.0          # 觸及止損時的實際預估虧損（含成本）
    risk_pct_equity: float = 0.0
    est_fees: float = 0.0
    est_slippage: float = 0.0
    est_funding: float = 0.0
    liq_price: Optional[float] = None
    rr_gross: float = 0.0
    rr_net: float = 0.0
    cost_to_risk: float = 0.0


@dataclass
class TradeSignal:
    symbol: str
    exchange: str
    timeframe: str
    analysis_time: str = ""
    bar_time: str = ""                 # 最後一根已收盤 K 線的開盤時間（UTC ISO）
    data_source: str = "offline"       # live / synthetic(測試資料) / offline
    data_status: str = ""
    data_ok: bool = False
    regime: str = "unknown"
    regime_label: str = "未知"
    direction: str = "flat"            # long / short / flat
    direction_label: str = "觀望"
    candidate_direction: str = "flat"  # 風控前的候選方向
    strategy: str = ""
    strategy_label: str = ""
    signal_key: str = ""
    conditions: list = field(default_factory=list)
    entry_ref: Optional[float] = None
    entry_note: str = ""
    stop: Optional[float] = None
    invalidation: str = ""
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    other_exits: list = field(default_factory=list)
    rr_gross: Optional[float] = None
    rr_net: Optional[float] = None
    plan: Optional[PositionPlan] = None
    pros: list = field(default_factory=list)
    cons: list = field(default_factory=list)
    uncertainties: list = field(default_factory=list)
    evidence_groups: dict = field(default_factory=dict)
    evidence_text: str = ""
    rejection_reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
