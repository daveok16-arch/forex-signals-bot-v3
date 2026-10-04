"""Unit tests for the trend / time-series-momentum strategy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.trend_strategy import TrendConfig, TrendStrategy  # noqa: E402


def _frame(closes, start="2020-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": idx,
        "open": closes,
        "high": np.array(closes) * 1.01,
        "low": np.array(closes) * 0.99,
        "close": closes,
        "volume": 1000,
    })


def test_uptrend_is_buy_and_downtrend_is_sell():
    up = list(np.linspace(1.0, 1.5, 400))
    down = list(np.linspace(1.5, 1.0, 400))
    sig = TrendStrategy().generate({"AAA": _frame(up), "BBB": _frame(down)})
    assert sig["AAA"]["direction"] == "BUY"
    assert sig["BBB"]["direction"] == "SELL"
    assert sig["AAA"]["trend_score"] == 1.0
    assert sig["BBB"]["trend_score"] == -1.0


def test_stop_and_target_bracket_entry():
    up = list(np.linspace(1.0, 1.5, 400))
    s = TrendStrategy().generate({"AAA": _frame(up)})["AAA"]
    assert s["stop_loss"] < s["entry"] < s["take_profit"]
    down = list(np.linspace(1.5, 1.0, 400))
    s = TrendStrategy().generate({"BBB": _frame(down)})["BBB"]
    assert s["take_profit"] < s["entry"] < s["stop_loss"]


def test_flat_market_is_hold():
    flat = [1.0] * 400
    s = TrendStrategy().generate({"AAA": _frame(flat)})["AAA"]
    assert s["direction"] == "HOLD"
    assert s["confidence"] == 0.0


def test_confidence_bounds_and_size():
    up = list(np.linspace(1.0, 1.5, 400))
    s = TrendStrategy().generate({"AAA": _frame(up)})["AAA"]
    assert 0.0 <= s["confidence"] <= 95.0
    assert 0.0 < s["vol_target_size"] <= TrendConfig().max_leverage


def test_insufficient_history_is_skipped():
    short = list(np.linspace(1.0, 1.1, 50))
    assert TrendStrategy().generate({"AAA": _frame(short)}) == {}


def test_misaligned_indexes_are_aligned():
    """Pairs with different date ranges must not crash or misalign."""
    a = _frame(list(np.linspace(1.0, 1.5, 400)), start="2020-01-01")
    b = _frame(list(np.linspace(1.5, 1.0, 400)), start="2020-06-01")
    sig = TrendStrategy().generate({"AAA": a, "BBB": b})
    assert set(sig) == {"AAA", "BBB"}
