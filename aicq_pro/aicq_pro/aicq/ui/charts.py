"""Plotly 圖表。plotly 只在函式內匯入，核心邏輯與測試不依賴它。此檔在沙盒中『未實際渲染驗證』。"""
from __future__ import annotations

import pandas as pd


def candlestick_figure(feat: pd.DataFrame, signal=None, last_n: int = 300, title: str = ""):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    d = feat.iloc[-last_n:]
    x = d.index.tz_convert("Asia/Taipei")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.15, 0.25], vertical_spacing=0.03,
                        subplot_titles=(title, "成交量", "RSI / ADX"))
    fig.add_trace(go.Candlestick(x=x, open=d.open, high=d.high, low=d.low, close=d.close, name="K線"), row=1, col=1)
    for col, name in (("ema_fast", "EMA快"), ("ema_slow", "EMA慢"), ("vwap", "VWAP"), ("bb_up", "布林上"), ("bb_lo", "布林下")):
        fig.add_trace(go.Scatter(x=x, y=d[col], name=name, mode="lines", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=d.volume, name="成交量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d.rsi, name="RSI", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d.adx, name="ADX", mode="lines"), row=3, col=1)
    if signal is not None and signal.direction != "flat":
        for y, label, color in ((signal.entry_ref, "進場參考", "#1f77b4"), (signal.stop, "止損", "#d62728"),
                                (signal.tp1, "止盈1", "#2ca02c")):
            if y is not None:
                fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=label, row=1, col=1)
    fig.update_layout(height=760, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h"))
    return fig


def equity_figure(curves: dict, title: str = "資金曲線"):
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, s in curves.items():
        if s is not None and len(s):
            fig.add_trace(go.Scatter(x=s.index.tz_convert("Asia/Taipei") if getattr(s.index, "tz", None) else s.index,
                                     y=s.values, name=name, mode="lines"))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def drawdown_figure(equity: pd.Series, title: str = "回撤 %"):
    import plotly.graph_objects as go

    dd = (equity.cummax() - equity) / equity.cummax() * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=-dd.values, fill="tozeroy", name="回撤%"))
    fig.update_layout(title=title, height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig
