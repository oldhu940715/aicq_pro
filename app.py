"""AI Crypto Quant Pro — Streamlit 儀表板（繁體中文）。啟動：streamlit run app.py

注意：此檔只負責顯示；所有計算在 aicq/ 套件內（已有自動化測試）。
本檔在開發沙盒內因無法安裝 streamlit/plotly，只做過語法與『假 streamlit』煙霧測試，尚未在真實瀏覽器驗證。
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from aicq.config import BacktestConfig, CostConfig, RiskConfig, StrategyParams, load_config
from aicq.execution.guard import live_status_message
from aicq.models import DataGate
from aicq.services import get_broker, load_market, run_validation_suite
from aicq.strategy.signal_engine import analyze_symbol
from aicq.timeutil import fmt_taipei, tf_seconds
from aicq.ui import charts

st.set_page_config(page_title="AI Crypto Quant Pro", page_icon="📈", layout="wide")
BASE_CFG = load_config()

# ---------------------------------------------------------------- 側邊欄
with st.sidebar:
    st.title("📈 AI Crypto Quant Pro")
    st.caption("加密貨幣短線量化分析｜預設僅模擬交易")
    mode_label = st.radio("資料來源", ["即時行情（交易所公開 API）", "測試資料（合成，非真實行情）"], index=0)
    mode = "live" if mode_label.startswith("即時") else "synthetic"
    symbol = st.selectbox("交易標的", list(BASE_CFG.symbols))
    base_tf = st.selectbox("分析週期", ["15m", "5m", "1m", "1h"], index=0)
    bars = st.slider("K 線根數", 500, 5000, 2000, step=250)
    auto = st.checkbox("自動重新整理", value=False)
    auto_s = st.number_input("重新整理間隔（秒）", 15, 600, 60) if auto else 60
    st.divider()
    st.info(live_status_message(BASE_CFG))

# 使用者在『設定』頁調整後的參數（人工調整；系統不會自行放大風險）
ss = st.session_state
risk = RiskConfig(**ss.get("risk_kwargs", {}))
cost = CostConfig(**ss.get("cost_kwargs", {}))
params = StrategyParams(**ss.get("param_kwargs", {}))
cfg = replace(BASE_CFG, risk=risk, cost=cost, params=params)


@st.cache_data(ttl=30, show_spinner=False)
def _market(symbol_: str, tf_: str, mode_: str, bars_: int):
    return load_market(BASE_CFG, symbol_, tf_, mode_, bars_)


def _fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime"):
            out[c] = out[c].astype(str)
    return out


def _metrics_table(m: dict) -> pd.DataFrame:
    rows = []
    for k, v in m.items():
        if isinstance(v, float):
            v = "∞" if np.isinf(v) else ("—" if np.isnan(v) else f"{v:,.4f}")
        rows.append({"指標": k, "數值": str(v)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 取得資料
with st.spinner("讀取行情中…"):
    try:
        base, htf, inst = _market(symbol, base_tf, mode, bars)
    except Exception as e:  # noqa: BLE001
        st.error(f"讀取行情失敗：{type(e).__name__}: {e}")
        st.stop()

if base.source == "synthetic":
    st.warning("⚠ 目前使用【合成測試資料】，不是真實行情。所有訊號與績效僅供功能驗證，沒有任何實際意義。")
elif base.source == "offline":
    st.error(f"📡 行情離線：{base.message}。系統已停止產生交易訊號與新增模擬單。")
elif not base.ok:
    st.warning("行情資料品質檢查未通過：" + "；".join(base.quality.issues))
else:
    st.success(f"行情正常｜{cfg.exchange_id}｜{symbol}｜最後 K 線 {fmt_taipei(base.df.index[-1])}")

broker = get_broker(cfg)
if base.source == "offline" and len(broker.db.positions("open")):
    st.error("⚠ 行情離線，但仍有未平倉的模擬持倉：目前無法監控止損／止盈，恢復連線後請立即更新持倉。")
marks = {symbol: float(base.df["close"].iloc[-1])} if len(base.df) else {}
account = broker.account_state(marks)
signal = analyze_symbol(symbol, base_tf, base, htf, inst, account, cfg)
broker.db.save_signal(signal)

tabs = st.tabs(["市場總覽", "K線與指標", "交易訊號", "策略回測", "模擬帳戶", "交易紀錄", "設定與風控", "說明與狀態"])

# ---------------------------------------------------------------- 市場總覽
with tabs[0]:
    st.subheader("市場總覽")
    rows = []
    for s in cfg.symbols:
        try:
            b, h, i = _market(s, base_tf, mode, bars)
            sg = analyze_symbol(s, base_tf, b, h, i, account, cfg)
            chg = np.nan
            per_day = 86400 // tf_seconds(base_tf)
            if len(b.df) > per_day:
                chg = (b.df.close.iloc[-1] / b.df.close.iloc[-per_day - 1] - 1) * 100
            rows.append({"標的": s, "最新收盤": round(float(b.df.close.iloc[-1]), 4) if len(b.df) else None,
                         "24h 漲跌%": None if np.isnan(chg) else round(chg, 2), "市場狀態": sg.regime_label,
                         "方向": sg.direction_label, "資料來源": b.source,
                         "資料狀態": "正常" if b.ok else "異常/離線", "最後K線": fmt_taipei(b.df.index[-1]) if len(b.df) else "—"})
        except Exception as e:  # noqa: BLE001
            rows.append({"標的": s, "方向": f"錯誤：{e}"})
    st.dataframe(pd.DataFrame(rows))
    c1, c2, c3 = st.columns(3)
    c1.metric("模擬帳戶權益 (USDT)", f"{account.equity:,.2f}")
    c2.metric("今日損益", f"{account.daily_pnl_pct:+.2f}%", help=f"單日損失上限 -{risk.max_daily_loss_pct}%")
    c3.metric("持倉數", f"{len(account.open_positions)} / {risk.max_open_positions}")
    if account.daily_pnl_pct <= -risk.max_daily_loss_pct:
        st.error("已達單日最大損失，今日停止新增倉位。")

# ---------------------------------------------------------------- K 線
with tabs[1]:
    st.subheader(f"{symbol} · {base_tf}")
    if base.df.empty or len(base.df) < 260:
        st.info("K 線資料不足，無法繪製指標（需要至少 260 根）。")
    else:
        from aicq.strategy.strategies import prepare_dataset
        feat, _ = prepare_dataset(base.df, htf.df if htf is not None else None, base_tf, cfg.htf, cfg.params)
        st.plotly_chart(charts.candlestick_figure(feat, signal, title=f"{symbol} {base_tf}"))
    q = base.quality
    if q is not None:
        st.caption(f"資料檢查：{q.n_rows} 根｜缺漏 {q.n_missing_bars}｜重複 {q.n_duplicates}｜異常 {q.n_invalid}｜"
                   f"資料延遲 {0 if q.age_seconds is None else q.age_seconds / 60:.1f} 分鐘｜{'新鮮' if q.fresh else '過期'}")
        for w in q.warnings + q.issues:
            st.warning(w)

# ---------------------------------------------------------------- 訊號
with tabs[2]:
    st.subheader("交易訊號")
    icon = {"long": "🟢", "short": "🔴", "flat": "⚪"}[signal.direction]
    st.markdown(f"### {icon} {signal.direction_label}　{signal.strategy_label}")
    a, b_, c_, d_ = st.columns(4)
    a.metric("標的／交易所", f"{signal.symbol}", signal.exchange)
    b_.metric("市場狀態", signal.regime_label)
    c_.metric("資料來源", {"live": "即時", "synthetic": "測試資料", "offline": "離線"}[signal.data_source])
    d_.metric("分析時間", signal.analysis_time)
    st.caption(f"最後一根已收盤 K 線：{signal.bar_time or '—'}｜{signal.data_status}")
    for w in signal.warnings:
        st.warning(w)
    if signal.rejection_reasons:
        st.error("未產生可執行訊號：\n\n" + "\n".join(f"- {r}" for r in signal.rejection_reasons))
    st.markdown("**訊號依據**\n\n" + "\n".join(f"- {x}" for x in signal.conditions))
    if signal.strategy:
        st.markdown(f"**證據評分（非機率）**：{signal.evidence_text}")
        st.dataframe(pd.DataFrame([{"獨立證據組": k, "成立": "✅" if v else "❌"} for k, v in signal.evidence_groups.items()]))
    if signal.entry_ref is not None:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("參考進場價", f"{signal.entry_ref:,.2f}")
        e2.metric("止損", f"{signal.stop:,.2f}")
        e3.metric("止盈 1", f"{signal.tp1:,.2f}")
        e4.metric("風險報酬比 (毛／淨)", f"{signal.rr_gross:.2f} / {signal.rr_net:.2f}")
        st.caption(signal.entry_note)
        st.markdown(f"**策略失效條件**：{signal.invalidation}")
        st.markdown("**其他出場**：" + "；".join(signal.other_exits) + f"；TP2 參考價 {signal.tp2:,.2f}")
    p = signal.plan
    if p is not None and p.qty > 0:
        st.markdown("**倉位與成本估算**（以風控模組計算）")
        st.dataframe(pd.DataFrame([{
            "數量": p.qty, "名目價值": round(p.notional, 2), "採用槓桿": p.leverage, "保證金占用": round(p.margin, 2),
            "止損實際風險(USDT,含成本)": round(p.risk_amount, 2), "占權益%": round(p.risk_pct_equity, 3),
            "預估手續費": round(p.est_fees, 2), "預估滑價": round(p.est_slippage, 2), "預估資金費": round(p.est_funding, 2),
            "粗估強平價": None if p.liq_price is None else round(p.liq_price, 2)}]))
    st.markdown("**主要反面證據**\n\n" + ("\n".join(f"- {x}" for x in signal.cons) if signal.cons else "- （無候選訊號）"))
    st.markdown("**尚未解決的不確定性**\n\n" + "\n".join(f"- {x}" for x in signal.uncertainties))
    if signal.direction != "flat":
        if st.button("建立模擬單（僅模擬，不會真實下單）"):
            ok, msg = broker.try_open(signal, inst, marks, DataGate(base.ok, list(base.quality.issues) if base.quality else [],
                                                                     base.is_test_data))
            (st.success if ok else st.error)(msg)

# ---------------------------------------------------------------- 回測
with tabs[3]:
    st.subheader("策略回測與樣本外驗證")
    if base.source == "synthetic":
        st.warning("目前為合成測試資料：回測結果只能驗證程式流程，不能用來判斷策略是否有效。")
    st.caption("流程：依時間切成 前50%訓練／中20%驗證／後30%樣本外（間隔空窗）；三段式選參只在訓練段選、驗證段把關、樣本外凍結後只評估一次；"
               "另有 Walk-forward 與成本敏感度。"
               "進場於訊號下一根開盤；同K線雙觸一律先算止損；含手續費、滑價、資金費。")
    if st.button("執行回測（可能需要數十秒）"):
        with st.spinner("回測中…"):
            try:
                ss["bt"] = run_validation_suite(cfg, base, htf, base_tf, inst, cfg.params, BacktestConfig(), storage=broker.db)
                ss["bt_label"] = f"{symbol} {base_tf} {base.source} {len(base.df)} 根"
            except Exception as e:  # noqa: BLE001
                st.error(f"回測失敗：{type(e).__name__}: {e}")
    bt = ss.get("bt")
    if bt:
        st.caption(f"結果對應：{ss.get('bt_label')}")
        fm = bt["full_metrics"]
        n_runs = bt.get("n_backtests_so_far", 1)
        st.caption(f"資料指紋 {bt.get('fingerprint')}｜這是累計第 {n_runs} 次回測紀錄")
        if n_runs >= 5:
            st.warning(f"你已累計執行 {n_runs} 次回測／調參。試驗次數越多，『挑到運氣好的設定』的機率越高，"
                       "樣本外結果會逐漸失去獨立性；請以全新資料（之後才發生的行情）做最終確認。")
        def _show_verdict(title, v):
            fn = st.success if v["status"].startswith("通過") else (st.warning if v["status"] in ("證據不足", "未驗證") else st.error)
            fn(f"**{title}：{v['status']}**\n\n" + "\n".join(f"- {r}" for r in v["reasons"]))
        _show_verdict("預設參數・後段樣本外判定", bt["default_oos_verdict"])
        if "selection" in bt:
            sel = bt["selection"]
            st.markdown(f"**三段式選參（試驗 {sel['n_trials']} 組）**：{sel['message']}")
            st.dataframe(sel["table"])
            _show_verdict("選參後・樣本外判定", bt["selection_verdict"])
            _show_verdict("Walk-forward 合併樣本外判定", bt["wf_verdict"])
        st.plotly_chart(charts.equity_figure({"策略（全期間）": bt["full"].equity, "買入持有": bt["benchmark"]}))
        st.plotly_chart(charts.drawdown_figure(bt["full"].equity))
        st.markdown("**全期間績效**")
        st.dataframe(_metrics_table(fm))
        st.markdown("**分段績效（預設參數，三段都沒有被調校；僅看穩定性）**")
        st.dataframe(pd.DataFrame({k: v[1] for k, v in bt["segments"].items()}).astype(str))
        st.markdown("**成本敏感度（樣本外）**")
        st.dataframe(bt["cost_sensitivity"])
        st.markdown("**分市場狀態表現**")
        st.dataframe(bt["regimes"])
        if bt["full"].blocked:
            st.markdown("**被風控擋下的候選進場（原因彙總）**")
            st.dataframe(pd.DataFrame([{"原因": k, "次數": v} for k, v in bt["full"].blocked.items()]))
        if "walk_forward" in bt:
            wf = bt["walk_forward"]
            st.markdown("**Walk-forward（參數只在訓練窗選擇，於其後測試窗驗證）**")
            st.dataframe(_fmt_df(wf.folds))
            st.dataframe(_metrics_table(wf.oos_metrics))
            st.markdown("**參數／時期穩健性（不是為了挑最佳組合）**")
            st.dataframe(bt["robustness"])
        st.markdown("**交易明細**")
        st.dataframe(_fmt_df(bt["full"].trades))

# ---------------------------------------------------------------- 模擬帳戶
with tabs[4]:
    st.subheader("模擬帳戶")
    if st.button("以最新已收盤 K 線更新持倉（檢查止損／止盈）"):
        if base.source == "offline" or not base.ok:
            st.error("行情離線或資料品質異常，已拒絕更新持倉（避免用過期資料判斷止損／止盈）。")
        else:
            closed = broker.update({symbol: base.df})
            broker.db.add_equity_point(broker.equity(marks))
            st.success(f"已更新；本次平倉 {len(closed)} 筆")
    acct = broker.account_state(marks)
    pos = broker.db.positions("open")
    cl = broker.db.positions("closed")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("權益", f"{acct.equity:,.2f}")
    k2.metric("已實現損益", f"{broker.realized_pnl():+,.2f}")
    k3.metric("今日損益", f"{acct.daily_pnl_pct:+.2f}%")
    eqp = broker.db.equity_points()
    k4.metric("最大回撤（已實現權益）", f"{broker.max_drawdown_pct():.2f}%", help="由已平倉交易推導，不受是否按過『更新』影響")
    if acct.halted_strategies:
        st.error(f"因回撤超過 {risk.strategy_max_drawdown_pct}% 停用的策略：{', '.join(acct.halted_strategies)}")
    st.markdown("**持倉**")
    st.dataframe(_fmt_df(pos[["id", "symbol", "side", "strategy", "qty", "entry_price", "stop", "tp1", "notional",
                              "margin", "risk_amount", "opened_at", "data_source"]]) if len(pos) else pd.DataFrame())
    if len(pos):
        pid = st.selectbox("手動平倉的持倉 ID", list(pos["id"]))
        if st.button("以最新價手動平倉"):
            broker.close_manual(int(pid), marks.get(pos[pos.id == pid].iloc[0].symbol, 0.0))
            st.rerun()
    curve = broker.equity_curve()
    if len(curve) > 1:
        st.plotly_chart(charts.equity_figure({"已實現權益（每筆平倉後）": curve}))
    else:
        st.info("尚無已平倉交易，沒有資金曲線。")

# ---------------------------------------------------------------- 紀錄
with tabs[5]:
    st.subheader("交易歷史")
    cl = broker.db.positions("closed")
    if cl.empty:
        st.info("尚無已平倉的模擬交易。")
    else:
        st.dataframe(_fmt_df(cl[["id", "symbol", "side", "strategy", "qty", "entry_price", "exit_price", "exit_reason",
                                 "gross_pnl", "fees", "funding", "slippage_cost", "net_pnl", "opened_at", "exit_time",
                                 "data_source", "reason"]]))
    st.markdown("**訊號紀錄**")
    st.dataframe(broker.db.signals(100))
    st.markdown("**系統事件／風控拒絕紀錄**")
    st.dataframe(broker.db.events(100))

# ---------------------------------------------------------------- 設定
with tabs[6]:
    st.subheader("策略設定與風控設定")
    st.caption("預設值是『保守的工程測試值』，不代表適合任何人。調高風險參數須由你本人手動確認，系統不會自動放大。")
    r = RiskConfig()
    c1, c2, c3 = st.columns(3)
    rk = {
        "risk_per_trade_pct": c1.number_input("單筆風險（權益 %）", 0.05, 2.0, risk.risk_per_trade_pct, 0.05),
        "max_daily_loss_pct": c2.number_input("單日最大損失（%）", 0.2, 5.0, risk.max_daily_loss_pct, 0.1),
        "max_leverage": c3.number_input("槓桿上限", 1.0, 10.0, risk.max_leverage, 0.5),
        "max_open_positions": int(c1.number_input("最大持倉數", 1, 5, risk.max_open_positions)),
        "min_net_rr": c2.number_input("扣成本後最低風險報酬比", 1.0, 5.0, risk.min_net_rr, 0.1),
        "max_cost_to_risk": c3.number_input("成本占止損風險上限", 0.05, 1.0, risk.max_cost_to_risk, 0.05),
    }
    d1, d2, d3 = st.columns(3)
    ck = {
        "taker_fee_rate": d1.number_input("Taker 手續費率", 0.0, 0.005, cost.taker_fee_rate, 0.0001, format="%.5f"),
        "slippage_bps": d2.number_input("單邊滑價 (bps)", 0.0, 50.0, cost.slippage_bps, 0.5),
        "funding_rate_per_8h": d3.number_input("資金費率 / 8h（假設）", -0.01, 0.01, cost.funding_rate_per_8h, 0.0001, format="%.5f"),
    }
    e1, e2, e3 = st.columns(3)
    pk = {
        "stop_atr_mult": e1.number_input("止損 ATR 倍數", 0.5, 5.0, params.stop_atr_mult, 0.1),
        "tp_r": e2.number_input("止盈 R 倍數（趨勢/突破）", 0.5, 6.0, params.tp_r, 0.25),
        "min_evidence_groups": int(e3.number_input("最少獨立證據組數 (1-4)", 1, 4, params.min_evidence_groups)),
    }
    if st.button("套用設定"):
        ss["risk_kwargs"], ss["cost_kwargs"], ss["param_kwargs"] = rk, ck, pk
        st.cache_data.clear()
        st.rerun()
    st.markdown(f"**真實下單**：{live_status_message(cfg)}")
    st.markdown("**商品規格**：" + ("來自交易所 API" if inst.verified else "⚠ 預設／未驗證（" + inst.note + "）"))

# ---------------------------------------------------------------- 說明
with tabs[7]:
    st.subheader("說明與功能狀態")
    st.markdown("""
**已完成並有自動化測試**：行情驗證與快取、指標與特徵（防前視）、市場狀態、三種策略、風控與倉位計算、回測引擎與樣本外驗證、模擬交易與 SQLite 紀錄、真實下單防護。

**已寫好但本沙盒無法實測**：CCXT 真實連線與下載、Streamlit 介面與 Plotly 圖表（需在你的電腦依 README 啟動確認）。

**尚未實作**：真實下單、機器學習模型、LLM 訊號解說、真實資金費率歷史下載、部分停利（回測以 TP1 全部出場）、多帳戶／多交易所。

**重要限制**：這不是投資建議，也不保證獲利；永續合約有強平風險；預設費率與規格為保守假設，請以交易所實際值為準。
""")

if auto:
    time.sleep(float(auto_s))
    st.cache_data.clear()
    st.rerun()
