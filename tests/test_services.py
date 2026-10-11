from aicq.config import AppConfig, BacktestConfig, StrategyParams
from aicq.services import load_market, run_validation_suite


def test_services_synthetic_end_to_end():
    cfg = AppConfig()
    base, htf, inst = load_market(cfg, "BTC/USDT:USDT", "15m", "synthetic", 6000)
    assert base.source == "synthetic" and base.is_test_data and htf is not None and htf.source == "synthetic"
    out = run_validation_suite(cfg, base, htf, "15m", inst, StrategyParams(min_evidence_groups=2), BacktestConfig())
    assert set(out["segments"]) == {"前段(訓練)", "中段(驗證)", "後段(樣本外)"}
    assert out["data_source"] == "synthetic"
    assert "walk_forward" in out and "robustness" in out
    assert out["full_metrics"]["n_trades"] == len(out["full"].trades)
