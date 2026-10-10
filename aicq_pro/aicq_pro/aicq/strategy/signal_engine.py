"""訊號引擎：資料檢查 → 特徵 → 候選訊號 → 風控 → 可解釋輸出。沒有優勢或資料不全時輸出『觀望』。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import AppConfig, StrategyParams
from ..data.provider import MarketData
from ..models import AccountState, DataGate, Instrument, TradeRequest, TradeSignal
from ..risk.manager import RiskManager
from ..timeutil import fmt_taipei, now_utc, to_ms
from .regime import REGIME_LABELS, STRATEGIES_BY_REGIME
from .strategies import EVIDENCE_NAMES, STRATEGY_LABELS, prepare_dataset

DIR_LABEL = {"long": "做多", "short": "做空", "flat": "觀望"}
MIN_ROWS = 260   # EMA/分位數等指標暖機所需的最少根數


def _fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}"


def _conditions(name: str, r, side: int, p: StrategyParams) -> list:
    s = "多" if side == 1 else "空"
    if name == "trend":
        return [f"EMA{p.ema_fast}{'>' if side == 1 else '<'}EMA{p.ema_slow}（{_fmt(r.ema_fast)} / {_fmt(r.ema_slow)}），順勢{s}向",
                f"收盤價{'高於' if side == 1 else '低於'}當日 VWAP（{_fmt(r.close)} vs {_fmt(r.vwap)}）",
                f"ADX={_fmt(r.adx,1)} ≥ {p.adx_trend}（趨勢強度足夠）",
                f"回測 EMA{p.ema_fast} 後重新{'站上' if side == 1 else '跌破'}（再進場觸發）",
                f"1h 趨勢={'多' if r.htf == 1 else '空' if r.htf == -1 else '不明'}"]
    if name == "breakout":
        lvl = r.don_hi if side == 1 else r.don_lo
        return [f"收盤{'突破' if side == 1 else '跌破'}前 {p.breakout_lookback} 根區間（{_fmt(lvl)}）",
                f"成交量為近20根均量的 {_fmt(r.vol_ratio)} 倍（門檻 {p.breakout_vol_ratio}）",
                f"ATR 高於其20根均值（波動擴張）",
                f"收盤位於 K 線{'上' if side == 1 else '下'}段（位置 {_fmt(r.close_loc)}），降低假突破"]
    return [f"價格偏離布林中線 {_fmt(r.bb_z)} 個標準差（門檻 ±{p.bb_z_extreme}）",
            f"RSI={_fmt(r.rsi,1)}（{'≤' if side == 1 else '≥'} {p.rsi_low if side == 1 else p.rsi_high}）",
            f"目標為回歸布林中線 {_fmt(r.bb_mid)}（僅在盤整狀態啟用）"]


def _cons(name: str, r, side: int, p: StrategyParams) -> list:
    out = []
    if r.htf == -side:
        out.append("1h 高週期趨勢與訊號方向相反")
    if np.isfinite(r.rsi) and ((side == 1 and r.rsi > 70) or (side == -1 and r.rsi < 30)) and name != "mean_reversion":
        out.append(f"RSI={_fmt(r.rsi,1)} 已處極端，追價風險高")
    if np.isfinite(r.bb_z) and abs(r.bb_z) > 2 and np.sign(r.bb_z) == side and name != "mean_reversion":
        out.append("價格已貼近／超出布林外軌，容易回檔")
    if np.isfinite(r.vol_ratio) and r.vol_ratio < 0.8:
        out.append("成交量低於均量，訊號參與度不足")
    if name == "mean_reversion" and np.isfinite(r.adx) and r.adx > p.adx_range:
        out.append("ADX 偏高，逆勢摸頂抄底風險較大")
    if np.isfinite(r.vwap) and ((side == 1 and r.close < r.vwap) or (side == -1 and r.close > r.vwap)) and name != "mean_reversion":
        out.append("價格位於 VWAP 的反側")
    return out or ["未偵測到明顯的單項反面證據（不代表沒有風險）"]


def analyze_symbol(symbol: str, base_tf: str, base: MarketData, htf: MarketData | None, inst: Instrument,
                   account: AccountState, cfg: AppConfig, params: StrategyParams | None = None,
                   backtest_note: str | None = None) -> TradeSignal:
    p = params or cfg.params
    rm = RiskManager(cfg.risk, cfg.cost)
    now = now_utc()
    sig = TradeSignal(symbol=symbol, exchange=cfg.exchange_id, timeframe=base_tf, analysis_time=fmt_taipei(now),
                      data_source=base.source, data_status=base.message, data_ok=base.ok)
    sig.uncertainties = [
        "『證據組數』是成立的獨立證據組數量，不是獲勝機率；本系統沒有經驗證的機率校準模型。",
        "同一根 K 線內止損與止盈的先後順序無法由 OHLC 判定，回測採保守的『先止損』。",
        "資金費率採固定假設；實際費率會變動。滑價為固定 bps 假設，實際取決於流動性。",
        backtest_note or "尚未提供樣本外驗證結果：在驗證前不應假設此策略有正期望值。",
    ]
    if not inst.verified:
        sig.warnings.append(f"商品規格未驗證：{inst.note}")
    if base.source == "synthetic":
        sig.warnings.append("目前為合成測試資料，非真實行情。")
    df = base.df
    if base.source == "offline" or df is None or df.empty:
        sig.rejection_reasons = list(base.quality.issues) if base.quality else ["沒有行情資料（離線）"]
        sig.conditions = ["無法取得行情，停止分析"]
        return sig
    if len(df) < MIN_ROWS:
        sig.rejection_reasons = [f"K 線只有 {len(df)} 根，少於指標暖機所需的 {MIN_ROWS} 根"]
        sig.bar_time = df.index[-1].isoformat()
        return sig
    f, cand = prepare_dataset(df, htf.df if htf is not None and len(htf.df) else None, base_tf, cfg.htf, p)
    r, c = f.iloc[-1], cand.iloc[-1]
    sig.bar_time = f.index[-1].isoformat()
    sig.regime = str(r.regime)
    sig.regime_label = REGIME_LABELS.get(sig.regime, sig.regime)
    if htf is None or htf.df.empty:
        sig.uncertainties.append("缺少 1h 高週期資料，多週期判斷以『中性』處理。")
    gate = DataGate(ok=base.ok, reasons=list(base.quality.issues) if base.quality else [], is_test_data=base.is_test_data)
    if not gate.ok:
        sig.rejection_reasons = gate.reasons or ["行情資料不可用"]
        sig.conditions = ["資料未通過檢查，停止產生可執行訊號"]
        return sig

    side = int(c.side)
    if side == 0:
        allowed = STRATEGIES_BY_REGIME.get(sig.regime, [])
        if c.conflict:
            why = "多個策略方向互相矛盾，視為行情矛盾 → 觀望"
        elif int(c.raw_side) != 0:
            why = f"有候選訊號但獨立證據僅 {int(c.evidence)}/4 組，低於門檻 {p.min_evidence_groups} → 觀望"
        elif not allowed:
            why = f"目前市場狀態為『{sig.regime_label}』，系統設定不啟用任何策略 → 觀望"
        else:
            why = f"市場狀態『{sig.regime_label}』（啟用：{'、'.join(STRATEGY_LABELS[s] for s in allowed)}），但沒有任何策略的進場條件成立 → 觀望"
        sig.conditions = [why, f"ADX={_fmt(r.adx,1)}、RSI={_fmt(r.rsi,1)}、成交量倍數={_fmt(r.vol_ratio)}、1h趨勢={int(r.htf)}"]
        return sig

    name = str(c.strategy)
    sig.strategy, sig.strategy_label = name, STRATEGY_LABELS[name]
    sig.candidate_direction = "long" if side == 1 else "short"
    sig.conditions = _conditions(name, r, side, p)
    sig.cons = _cons(name, r, side, p)
    groups = EVIDENCE_NAMES[name]
    flags = [bool(c.g1), bool(c.g2), bool(c.g3), bool(c.g4)]
    sig.evidence_groups = dict(zip(groups, flags))
    sig.evidence_text = f"獨立證據 {int(c.evidence)}/4 組成立（定義：相關指標合併為一組、每組一票；非機率）"
    entry = float(r.close)
    stop = entry - side * float(c.stop_dist)
    tp1 = entry + side * float(c.tp1_dist)
    sig.entry_ref = entry
    sig.entry_note = "下一根 K 線開盤市價進場（參考價＝最新收盤價，實際成交含滑價）"
    sig.stop, sig.tp1 = stop, tp1
    sig.tp2 = entry + side * float(c.tp1_dist) * 1.5
    sig.other_exits = ["第二目標 TP2 僅供參考（回測/模擬以 TP1 全部出場）", f"時間止損：持倉超過 96 根 K 線以收盤價出場",
                       "策略失效條件成立時應主動出場"]
    sig.invalidation = {"trend": f"收盤價{'跌破' if side == 1 else '站上'} EMA{p.ema_slow}，或觸及止損",
                        "breakout": "收盤價回到被突破的區間內（假突破），或觸及止損",
                        "mean_reversion": "價格續創新極端且不回歸中線（趨勢延續），或觸及止損"}[name]
    key = f"{symbol}|{base_tf}|{to_ms(f.index[-1])}|{name}|{side}"
    sig.signal_key = key
    req = TradeRequest(symbol, side, entry, stop, tp1, name, key, 96 * (to_ms(f.index[-1]) - to_ms(f.index[-2])) / 3_600_000)
    plan = rm.evaluate(req, account, gate, inst)
    sig.plan = plan
    sig.rr_gross, sig.rr_net = plan.rr_gross, plan.rr_net
    sig.warnings += plan.warnings
    if plan.allowed:
        sig.direction = sig.candidate_direction
    else:
        sig.rejection_reasons = list(plan.reasons)
    sig.direction_label = DIR_LABEL[sig.direction]
    return sig
