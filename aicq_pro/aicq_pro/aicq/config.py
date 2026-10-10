"""設定：所有風控與成本參數皆可調整；預設值為『保守的工程測試值』，不代表適合任何人。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.25      # 單筆風險：帳戶權益 %
    max_daily_loss_pct: float = 1.0       # 單日最大損失：日初權益 %
    max_leverage: float = 3.0             # 槓桿上限
    max_open_positions: int = 2
    max_total_exposure_x: float = 3.0     # 總名義價值 / 權益 上限
    min_net_rr: float = 1.5               # 扣成本後最低風險報酬比
    max_cost_to_risk: float = 0.30        # 預估來回成本 / 止損風險 上限
    maintenance_margin_rate: float = 0.005  # 強平估算用（粗估，非交易所實際分級）
    liq_safety_ratio: float = 0.5         # 止損距離須 <= 此比例 * 強平距離
    strategy_max_drawdown_pct: float = 10.0  # 策略回撤達此值即停用新倉
    max_data_age_bars: float = 2.5        # 最後一根 K 線收盤後超過幾根週期視為過期


@dataclass(frozen=True)
class CostConfig:
    # 以下為保守假設，非官方查證數值：請依自己帳戶實際費率調整。
    taker_fee_rate: float = 0.0005        # 0.05%
    slippage_bps: float = 2.0             # 單邊 2 bps
    funding_rate_per_8h: float = 0.0001   # 0.01% / 8h（固定假設；回測可傳入真實序列）


@dataclass(frozen=True)
class StrategyParams:
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    adx_period: int = 14
    adx_trend: float = 25.0
    adx_range: float = 20.0
    stop_atr_mult: float = 1.5
    tp_r: float = 2.0
    breakout_lookback: int = 20
    breakout_vol_ratio: float = 1.5
    rsi_period: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    bb_period: int = 20
    bb_k: float = 2.0
    bb_z_extreme: float = 2.0
    high_vol_quantile: float = 0.90
    low_liq_frac: float = 0.30
    min_evidence_groups: int = 3          # 4 組獨立證據中至少幾組成立
    htf_fast: int = 20
    htf_slow: int = 50


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    extra_delay_bars: int = 0             # 除「下一根開盤成交」外額外延遲
    max_hold_bars: int = 96


@dataclass(frozen=True)
class AppConfig:
    exchange_id: str = "binanceusdm"
    symbols: tuple = ("BTC/USDT:USDT", "ETH/USDT:USDT")
    timeframes: tuple = ("1m", "5m", "15m", "1h")
    htf: str = "1h"
    db_path: str = "data/aicq.db"
    initial_equity: float = 10_000.0
    live_requested: bool = False          # 使用者是否在 .env 要求真實下單（v1 一律拒絕）
    risk: RiskConfig = field(default_factory=RiskConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    params: StrategyParams = field(default_factory=StrategyParams)

    @property
    def live_trading_enabled(self) -> bool:
        """v1 未實作任何真實下單程式碼，因此永遠為 False。"""
        return False


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # python-dotenv 缺少時仍可用預設值
        pass
    env = os.environ
    return AppConfig(
        exchange_id=env.get("AICQ_EXCHANGE", "binanceusdm"),
        db_path=env.get("AICQ_DB_PATH", "data/aicq.db"),
        initial_equity=float(env.get("AICQ_INITIAL_EQUITY", "10000")),
        live_requested=_truthy(env.get("LIVE_TRADING_ENABLED")),
    )
