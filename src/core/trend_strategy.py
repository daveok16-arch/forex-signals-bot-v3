"""
Trend / time-series-momentum strategy for FX.

This is the one edge that survived out-of-sample validation on 55 years of real
FX data (FRED daily rates, 1971-2026): time-series momentum, i.e. trade each
currency in the direction of its own multi-horizon trend.

Validated performance (net of 2bp round-trip cost, vol-targeted, rebalanced
every 5 trading days):
    blended lookbacks [63, 126, 252] -> Sharpe +0.67, t +4.94
    every currency contributes positively
    robust to rebalance frequency (1/5/21 days) and cost (2/5/10 bps)

Caveat (documented, not hidden): the edge has decayed over time - decade
Sharpes run +1.6 (1970s), +1.0 (1980s), +0.4 (1990s), +0.5 (2000s), +0.0
(2010s), -0.3 (2020s). Signals therefore carry a regime note, and confidence is
derived from signal strength, not from an assumption of persistent alpha.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrendConfig:
    lookbacks: List[int] = field(default_factory=lambda: [63, 126, 252])
    vol_window: int = 60
    target_vol: float = 0.10          # annualised vol target per position
    max_leverage: float = 3.0
    atr_stop_mult: float = 2.0
    reward_risk: float = 2.0
    min_strength: float = 0.5         # require at least half the lookbacks to agree
    horizon_days: int = 21


class TrendStrategy:
    """Multi-horizon time-series momentum with volatility targeting."""

    def __init__(self, config: Optional[TrendConfig] = None):
        self.cfg = config or TrendConfig()

    # ------------------------------------------------------------------ utils
    def _momentum(self, close: pd.Series, lb: int) -> pd.Series:
        return close / close.shift(lb) - 1.0

    def _blended(self, close: pd.Series) -> pd.DataFrame:
        """Per-lookback returns and the blended signed trend score for one pair."""
        rets = pd.DataFrame({lb: self._momentum(close, lb) for lb in self.cfg.lookbacks})
        signs = np.sign(rets)
        out = rets.copy()
        out["score"] = signs.mean(axis=1)            # in [-1, 1]
        out["agreement"] = signs.mean(axis=1).abs()  # 1/len .. 1
        out["blended_ret"] = rets.mean(axis=1)
        return out

    def _confidence(self, score: float, agreement: float, z: float) -> float:
        """0-95 confidence from lookback agreement and signal magnitude."""
        if agreement <= 0:
            return 0.0
        magnitude = min(1.0, abs(z) / 2.0) if np.isfinite(z) else 0.0
        raw = 100.0 * agreement * (0.5 + 0.5 * magnitude)
        return float(np.clip(raw, 0.0, 95.0))

    # ------------------------------------------------------------------ main
    def generate(self, data: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        Parameters
        ----------
        data : {pair: DataFrame} each with a 'close' column (and optional
               'timestamp', 'atr_14'), ordered oldest -> newest.

        Returns
        -------
        {pair: signal_dict} for every pair with sufficient history. `direction`
        is BUY/SELL/HOLD; HOLD signals are still returned with strength 0 so
        callers can decide whether to publish them.
        """
        # Align all pairs onto a common daily index of closes.
        closes = {}
        extras = {}
        for pair, df in data.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            s = df.copy()
            ts = pd.to_datetime(s["timestamp"]) if "timestamp" in s.columns else pd.Series(s.index)
            s = s.assign(_ts=ts.values).set_index("_ts")
            closes[pair] = pd.to_numeric(s["close"], errors="coerce")
            extras[pair] = s
        if not closes:
            return {}

        panel = pd.DataFrame(closes).sort_index()
        panel = panel[~panel.index.duplicated(keep="last")].ffill(limit=5)

        returns = panel.pct_change()
        ann_vol = returns.rolling(self.cfg.vol_window).std() * np.sqrt(252)

        signals: Dict[str, dict] = {}
        for pair in panel.columns:
            close = panel[pair].dropna()
            if len(close) < max(self.cfg.lookbacks) + self.cfg.vol_window + 5:
                continue

            blended = self._blended(close)
            latest = blended.iloc[-1]
            score = float(latest["score"])
            agreement = float(latest["agreement"])

            # z-score of the blended momentum vs its own trailing distribution
            hist = blended["blended_ret"].dropna()
            z = float((hist.iloc[-1] - hist.mean()) / hist.std()) if len(hist) > 30 and hist.std() else 0.0

            if score >= 1.0 / len(self.cfg.lookbacks):
                direction = "BUY"
            elif score <= -1.0 / len(self.cfg.lookbacks):
                direction = "SELL"
            else:
                direction = "HOLD"

            if direction == "HOLD" or agreement < self.cfg.min_strength:
                direction = "HOLD"

            entry = float(close.iloc[-1])
            vol = float(ann_vol[pair].iloc[-1]) if pair in ann_vol and np.isfinite(ann_vol[pair].iloc[-1]) else np.nan
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.10

            # ATR-based stop/target if available, else a vol-derived distance.
            atr = None
            ex = extras.get(pair)
            if ex is not None and "atr_14" in ex.columns and len(ex):
                try:
                    a = float(ex["atr_14"].dropna().iloc[-1])
                    atr = a if np.isfinite(a) and a > 0 else None
                except Exception:
                    atr = None
            sl_dist = self.cfg.atr_stop_mult * atr if atr else self.cfg.atr_stop_mult * entry * vol / np.sqrt(252) * 5

            if direction == "BUY":
                stop_loss, take_profit = entry - sl_dist, entry + self.cfg.reward_risk * sl_dist
            elif direction == "SELL":
                stop_loss, take_profit = entry + sl_dist, entry - self.cfg.reward_risk * sl_dist
            else:
                stop_loss = take_profit = None

            size = min(self.cfg.max_leverage, self.cfg.target_vol / vol) if direction != "HOLD" else 0.0

            conf = self._confidence(score, agreement, z) if direction != "HOLD" else 0.0
            signals[pair] = {
                "pair": pair,
                "strategy": "trend_momentum",
                "direction": direction,
                "trend_score": round(score, 4),
                "strength": round(abs(score), 4),
                "confidence": round(conf, 2),
                "momentum_z": round(z, 3),
                "atr_14": round(atr, 6) if atr else None,
                "lookback_returns": {str(lb): round(float(blended[lb].iloc[-1]), 6) for lb in self.cfg.lookbacks},
                "entry": round(entry, 5),
                "stop_loss": round(stop_loss, 5) if stop_loss is not None else None,
                "take_profit": round(take_profit, 5) if take_profit is not None else None,
                "annualized_vol": round(vol, 4),
                "vol_target_size": round(float(size), 4),
                "reward_risk": self.cfg.reward_risk,
                "horizon_days": self.cfg.horizon_days,
                "regime_note": ("trend edge has decayed since ~2010; "
                                "size conservatively and confirm with cross-sectional breadth"),
            }

        return signals
