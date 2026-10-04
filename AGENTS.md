# AGENTS.md

Repository notes for AI agents working in this project.

## Overview
Multi-agent AI/ML forex signal pipeline. Agents live in `src/agents/`
(agent_a..agent_g) and inherit from `src/core/base_agent.py`. The orchestrator
is `src/core/pipeline_orchestrator.py`; `run.py` is the CLI entry point.

Flow: A (data) -> B (training) -> C (signals) -> D (risk) -> F (execution);
E (backtest) runs off B; G (monitoring) runs last.

## Setup
```
pip install -r requirements.txt   # heavy: torch/optuna/cloud SDKs
```
For a local, non-GPU run only the core subset is needed:
```
pip install numpy pandas scipy scikit-learn polars pyyaml joblib pyarrow \
            xgboost lightgbm catboost optuna requests psutil
```

## Running
```
python run.py --mode single --agent agent_a   # data + features
python run.py --mode single --agent agent_b   # train models (long)
python run.py --mode single --agent agent_c   # generate signals
python run.py --mode single --agent agent_d   # risk sizing
python run.py --mode single --agent agent_f   # execution (paper)
python run.py --mode single --agent agent_e   # backtest
python run.py --mode single --agent agent_g   # monitoring
```
`config/system_config.fast.yaml` is a reduced config (`n_trials: 5`) for quick
local runs. Artifacts land in `output/<agent>/`; models in `output/agent_b/`.

## Important behaviours / gotchas
- Agent A loads real data in this order: OANDA (needs `OANDA_API_KEY`/`OANDA_ACCOUNT_ID`)
  -> Yahoo Finance (`src/core/market_data.py`, daily OHLCV, `yahoo_range` in config)
  -> local `data/raw` -> synthetic. Synthetic is **off by default**
  (`allow_synthetic: false`); with no real source available Agent A raises instead
  of silently inventing data.
- Yahoo `*USD=X` FX daily bars are synthesized and overlap adjacent sessions (the
  next close lands inside the current bar's range ~74% of the time). Agent A
  therefore lags all derived features by one bar (`_compute_indicators`) so a model
  predicting bar `t+1` only sees settled bars. Without this, CV accuracy is a
  spurious ~0.80. `XAUUSD` uses real `GC=F` futures bars and does not leak.
- Agent B training is slow: with the shipped config (30-50 Optuna trials/model)
  it takes hours across 8 pairs. Use the fast config for iteration (~32 min).
- Agent C confidence/uncertainty describe the **latest bar** (the one a signal is
  emitted for), not the whole history. On real daily FX the models have no edge
  (CV ~0.50) so most signals are legitimately filtered; occasional passes are
  low-confidence by construction. Zero/few signals is the honest outcome, not a crash.
- Agent G GitHub sync is opt-in (`github.auto_commit: true`); it no longer commits
  or pushes during a normal monitoring run.
- `catboost_info/` is created by CatBoost at the repo root; it is gitignored.

## Tradeable signal: trend / time-series momentum (2026-10)
The ML ensemble (Agent B) has no edge on daily FX (CV ~0.50), so Agent C now emits
signals from a validated **trend / time-series-momentum** strategy instead
(`src/core/trend_strategy.py`). It is the only edge that survived out-of-sample
testing; the full factor search is documented below.

Strategy: per currency, blend the sign of 63/126/252-day momentum; go long when all
three agree up, short when all agree down (HOLD otherwise). Positions are
volatility-targeted (10% annualised, capped 3x) with ATR-based SL/TP (2x ATR stop,
2:1 reward/risk) and a 21-day horizon. Config block `agent_c.trend_strategy`
(`enabled: true`); set `enabled: false` to fall back to the ML ensemble path.

Validated performance (net 2bp cost, rebalanced every 5 days, 1971-2026 real FX):
blended lookbacks Sharpe **+0.67**, t **+4.94**; robust to rebalance frequency
(1/5/21d) and cost (2/5/10bp); every currency contributes positively.

**Honest caveat:** the edge has decayed. Decade Sharpes: 1970s +1.6, 1980s +1.0,
1990s +0.4, 2000s +0.5, 2010s +0.0, 2020s -0.3. Signals carry a `regime_note` and
confidence reflects signal strength, not an assumption of persistent alpha.

### Factor search summary (what did NOT work)
Tested on 55 years of real FRED daily FX (`DEX*` series) and clean ETF proxies:
- **Carry** (rate differential, FRED 3m interbank): no edge on this sample.
- **Cross-sectional momentum**: Sharpe ~+0.3, t<2, unstable across halves.
- **Short-term reversal**: negative (confirming trend) — the apparent `=X` reversal
  edge did not replicate on clean ETFs, i.e. a vendor artifact.
- Beware: a naive backtest on `*USD=X` closes (union of dates, forward-filled) can
  produce spurious positive Sharpes from index misalignment; always align panels.

## Fixes applied (2026-10)
- **Target leakage (critical).** Yahoo FX daily bars overlap, so intra-bar features
  (`adosc`, `upper_shadow`, `lower_shadow`, `cmf_20`, close position) correlated
  ~0.6-0.7 with the next day's return, inflating CV accuracy to ~0.80. Agent A now
  lags derived features one bar; honest accuracy is ~0.50 across all pairs.
- **Data source.** Added `src/core/market_data.py` (Yahoo/OANDA providers) and wired
  it into `agent_a_data._load_data`; synthetic is opt-in. Feature matrix now
  preserves the `timestamp` column so Agent C stamps signals with the real candle time.
- **MRMR selection.** `_mrmr_selection` was O(n^2) Python loops over ~200 features;
  vectorized to correlation-matrix operations.
- **Agent B CV.** Shuffled `StratifiedKFold` (and a purge filter that emptied folds)
  replaced with `TimeSeriesSplit` everywhere; the default splitter is now temporal.
  Also removed the deprecated `use_label_encoder` XGBoost arg.
- **Agent B report.** `results["total_models"]` was set after
  `_generate_training_report()` used it -> `KeyError`. Assignment moved earlier.
- **Agent C timestamp.** Feature parquet has no datetime index, so
  `timestamp.isoformat()` raised `AttributeError`. Now uses the preserved
  `timestamp` column, falling back to `pd.Timestamp.utcnow()`.
- **Agent C confidence.** Was averaged over the entire prediction history
  (`np.mean(abs(p-0.5))`); now evaluated on the latest bar only.
- **Agent E folds.** Expanding train window + 50% step produced ~1,594 folds/pair
  (~12,750 RF fits) and appeared to hang. Bounded to `max_folds` (default 50) with
  progress logging.
- **Agent E returns alignment.** Fold returns were sliced by position from a
  differently-sized `pct_change().dropna()` series, so late folds raised
  "operands could not be broadcast (63,) (62,)". Returns are now reindexed to the
  feature index.
- **Agent E annualization.** Metrics assumed 5-min bars (`252*24*12`) while volatility
  used `sqrt(252)`, giving absurd Sharpes (~-25). Now derives bars-per-year from the
  data (`_set_annualization`); realistic Sharpes are ~-1 to -5 on daily FX.
- **Agent E indexing.** `cumulative[-1]` used label indexing on a RangeIndex
  (pandas >=2 raises); now `.iloc[-1]`. `n_trades` counts actual direction changes.
- **Agent G sync.** `_sync_to_github` is opt-in and no longer runs `git init` or
  surprise commits/pushes.
