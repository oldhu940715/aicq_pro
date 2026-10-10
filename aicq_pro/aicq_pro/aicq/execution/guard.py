"""真實下單防護。v1 完全沒有實作任何真實下單程式碼；此模組確保任何路徑都無法意外進入真實模式。"""
from __future__ import annotations

from ..config import AppConfig


class LiveTradingDisabled(RuntimeError):
    pass


def assert_paper_only(cfg: AppConfig) -> None:
    if cfg.live_trading_enabled:  # 結構上永遠為 False，保留為第二道檢查
        raise LiveTradingDisabled("v1 不支援真實下單")


def live_status_message(cfg: AppConfig) -> str:
    if cfg.live_requested:
        return ("偵測到 LIVE_TRADING_ENABLED=true，但 v1 沒有實作真實下單，已忽略此設定；"
                "系統仍為『僅模擬交易』。")
    return "真實下單：關閉（v1 僅提供模擬交易）"


def place_live_order(*_a, **_k):
    raise LiveTradingDisabled("v1 未實作真實下單；請勿嘗試繞過。")
