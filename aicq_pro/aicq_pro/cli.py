"""命令列工具（不需要 Streamlit）。

  python cli.py download --symbol BTC/USDT:USDT --timeframe 15m --days 90    # 下載並快取真實歷史 K 線（需網路與 ccxt）
  python cli.py validate --symbol BTC/USDT:USDT --timeframe 15m               # 檢查快取資料完整性
  python cli.py backtest --synthetic --timeframe 15m --bars 8000              # 用『測試資料』跑完整回測流程
  python cli.py backtest --symbol BTC/USDT:USDT --timeframe 15m --bars 8000   # 用快取的真實資料回測
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from aicq.config import BacktestConfig, load_config
from aicq.data.cache import OhlcvCache
from aicq.data.exchange import ExchangeClient, ExchangeUnavailable
from aicq.data.provider import MarketData
from aicq.data.validation import drop_incomplete_last_bar, validate_ohlcv
from aicq.models import default_instrument
from aicq.services import load_market, run_validation_suite
from aicq.timeutil import now_utc, tf_ms, to_ms


def cmd_download(a, cfg):
    cache = OhlcvCache(cfg.db_path)
    try:
        cl = ExchangeClient(cfg.exchange_id)
        now = now_utc()
        df = cl.fetch_ohlcv_range(a.symbol, a.timeframe, to_ms(now) - a.days * 86_400_000, to_ms(now))
    except ExchangeUnavailable as e:
        print(f"❌ 下載失敗：{e}")
        return 2
    df = drop_incomplete_last_bar(df, a.timeframe, now)
    n = cache.upsert(cfg.exchange_id, a.symbol, a.timeframe, df)
    print(f"✅ 已寫入 {n} 根 {a.symbol} {a.timeframe} K 線到 {cfg.db_path}")
    return cmd_validate(a, cfg)


def cmd_validate(a, cfg):
    df = OhlcvCache(cfg.db_path).load(cfg.exchange_id, a.symbol, a.timeframe)
    q = validate_ohlcv(df, a.timeframe, max_age_bars=cfg.risk.max_data_age_bars, check_freshness=False)
    print(f"根數 {q.n_rows}｜缺漏 {q.n_missing_bars}｜重複 {q.n_duplicates}｜異常 {q.n_invalid}｜結果：{'通過' if q.ok else '未通過'}")
    for m in q.issues + q.warnings:
        print(" -", m)
    return 0 if q.ok else 1


def cmd_backtest(a, cfg):
    if a.synthetic:
        base, htf, inst = load_market(cfg, a.symbol, a.timeframe, "synthetic", a.bars)
        print("⚠ 使用合成測試資料：結果只能驗證流程，不代表任何真實績效。")
    else:
        cache = OhlcvCache(cfg.db_path)
        df = cache.load(cfg.exchange_id, a.symbol, a.timeframe, limit=a.bars)
        hdf = cache.load(cfg.exchange_id, a.symbol, cfg.htf)
        if df.empty:
            print("❌ 快取內沒有資料，請先執行 download。")
            return 2
        q = validate_ohlcv(df, a.timeframe, check_freshness=False)
        if not q.ok:
            print("❌ 資料未通過檢查：", q.issues)
            return 2
        base = MarketData(a.symbol, a.timeframe, df, "live", q, "快取歷史資料")
        htf = MarketData(a.symbol, cfg.htf, hdf, "live", None, "快取") if len(hdf) else None
        inst = default_instrument(a.symbol)
    out = run_validation_suite(cfg, base, htf, a.timeframe, inst, cfg.params, BacktestConfig())
    pd.set_option("display.width", 200)
    print("\n== 分段績效 ==")
    print(pd.DataFrame({k: v[1] for k, v in out["segments"].items()}).to_string())
    print("\n== 成本敏感度（樣本外） ==")
    print(out["cost_sensitivity"].round(4).to_string(index=False))
    if "walk_forward" in out:
        print("\n== Walk-forward 各折 ==")
        print(out["walk_forward"].folds.to_string(index=False))
        print("\n== 樣本外合併：", {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in out["walk_forward"].oos_metrics.items() if k in ("n_trades", "net_return_pct", "max_drawdown_pct", "expectancy_r")})
    print("\n被風控擋下的候選進場：", out["full"].blocked)
    return 0


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "validate", "backtest"):
        p = sp.add_parser(name)
        p.add_argument("--symbol", default=cfg.symbols[0])
        p.add_argument("--timeframe", default="15m")
        if name == "download":
            p.add_argument("--days", type=int, default=90)
        if name == "backtest":
            p.add_argument("--bars", type=int, default=8000)
            p.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    sys.exit({"download": cmd_download, "validate": cmd_validate, "backtest": cmd_backtest}[a.cmd](a, cfg))


if __name__ == "__main__":
    main()
