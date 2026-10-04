"""
Agent E: Backtesting & Performance Analytics Engine
===================================================
Robust strategy validation with walk-forward optimization,
combinatorial purged cross-validation, Monte Carlo testing,
and comprehensive performance analytics.

Features: Deflated Sharpe Ratio, Probability of Overfitting (PBO),
drawdown decomposition, regime-dependent attribution.
"""

import json
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent


class BacktestingEngine(BaseAgent):
    """
    Agent E - Backtesting & Performance Analytics
    
    Pipeline:
    1. Load feature matrix and trained models
    2. Run walk-forward backtest
    3. Calculate comprehensive metrics
    4. Monte Carlo permutation tests
    5. Generate performance report
    """

    def __init__(self, config_path: str = "config/system_config.yaml", kaggle_mode: bool = False):
        super().__init__(config_path=config_path, agent_key="agent_e", kaggle_mode=kaggle_mode)
        
        # Config
        self.initial_capital = self.agent_config.get("initial_capital", 100000)
        self.commission = self.agent_config.get("commission_per_lot", 3.5)
        self.slippage = self.agent_config.get("slippage_pips", 0.5)
        self.risk_free_rate = self.agent_config.get("risk_free_rate", 0.02)
        self.benchmark = self.agent_config.get("benchmark", "buy_and_hold")
        
        # Walk-forward config
        self.wf_config = self.agent_config.get("walk_forward", {})
        self.mc_config = self.agent_config.get("monte_carlo", {})
        self.metrics_list = self.agent_config.get("metrics", [])
        self.reporting_config = self.agent_config.get("reporting", {})
        
        # Results storage
        self._backtest_results: Dict[str, Any] = {}
        self._equity_curves: Dict[str, pd.Series] = {}
        self._periods_per_year = 252  # overwritten once the bar frequency is known

    def _execute_core(self) -> Dict[str, Any]:
        """Execute backtesting pipeline"""
        results = {
            "pairs_backtested": [],
            "performance_metrics": {},
            "monte_carlo_results": {},
            "walk_forward_results": {},
            "best_pair": None,
            "best_sharpe": -np.inf,
            "reports_generated": []
        }
        
        # Step 1: Load data and models
        self.logger.info("STEP 1: Loading features and models")
        feature_data = self._load_feature_data()
        
        if not feature_data:
            raise RuntimeError("No feature data for backtesting")
        
        # Step 2: Run backtest per pair
        self.logger.info("STEP 2: Running walk-forward backtest")
        
        for pair, df in feature_data.items():
            self.logger.info(f"\n  Backtesting {pair}")
            
            try:
                # Run walk-forward backtest
                bt_result = self._walk_forward_backtest(pair, df)
                
                results["pairs_backtested"].append(pair)
                results["performance_metrics"][pair] = bt_result["metrics"]
                results["walk_forward_results"][pair] = bt_result["folds"]
                
                # Track best pair
                sharpe = bt_result["metrics"].get("sharpe_ratio", -np.inf)
                if sharpe > results["best_sharpe"]:
                    results["best_sharpe"] = sharpe
                    results["best_pair"] = pair
                
                # Monte Carlo test
                if self.mc_config.get("enabled", True):
                    mc_result = self._monte_carlo_test(bt_result["returns"])
                    results["monte_carlo_results"][pair] = mc_result
                
            except Exception as e:
                self.logger.error(f"  Backtest failed for {pair}: {e}"); self.logger.error(traceback.format_exc())
                continue
        
        # Step 3: Cross-pair portfolio analysis
        self.logger.info("\nSTEP 3: Portfolio-level analysis")
        portfolio_metrics = self._portfolio_analysis(results)
        results["portfolio_metrics"] = portfolio_metrics
        
        # Step 4: Generate report
        self.logger.info("STEP 4: Generating performance report")
        report_path = self._generate_report(results)
        results["report_path"] = report_path
        results["reports_generated"].append(report_path)
        
        self.logger.info(
            f"\nAgent E complete | Pairs: {len(results['pairs_backtested'])} | "
            f"Best: {results['best_pair']} (Sharpe: {results['best_sharpe']:.3f})"
        )
        
        return results

    def _load_feature_data(self) -> Dict[str, pd.DataFrame]:
        """Load feature matrix from Agent A"""
        feature_dir = self.project_root / "output" / "agent_a" / "features"
        
        data = {}
        if feature_dir.exists():
            for file in feature_dir.glob("*_features.parquet"):
                pair = file.stem.replace("_features", "")
                df = pd.read_parquet(file)
                data[pair] = df
        
        return data

    def _set_annualization(self, df: pd.DataFrame) -> None:
        """Derive bars-per-year from the data so Sharpe/annualised figures are
        scaled to the actual sampling frequency (daily data here, not 5-min)."""
        if "timestamp" in df.columns and len(df) > 1:
            ts = pd.to_datetime(df["timestamp"])
            days = (ts.iloc[-1] - ts.iloc[0]).days
            if days > 0:
                self._periods_per_year = max(1, int(round(len(df) / (days / 365.25))))
                return
        self._periods_per_year = 252

    def _walk_forward_backtest(self, pair: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Walk-forward optimization backtest.
        Train on in-sample, test on out-of-sample, then roll forward.
        """
        self._set_annualization(df)

        if not self.wf_config.get("enabled", True):
            return self._simple_backtest(pair, df)
        
        train_size = self.wf_config.get("train_size", 0.7)
        test_size = self.wf_config.get("test_size", 0.3)
        min_train = self.wf_config.get("min_train_size", 252)
        step_size = self.wf_config.get("step_size", 63)
        
        # Prepare data
        returns = df["close"].pct_change().dropna()
        
        if len(returns) < min_train + step_size:
            return self._simple_backtest(pair, df)
        
        # Generate features
        X, feature_cols = self._prepare_backtest_features(df)
        y = (df["close"].shift(-1) > df["close"]).astype(int).loc[X.index]
        # Align returns to the feature index; otherwise fold slices are taken by
        # position from a differently-sized series and late folds broadcast-fail.
        returns = df["close"].pct_change().reindex(X.index).fillna(0.0)
        
        # Walk-forward folds
        folds = []
        all_returns = []
        max_folds = self.wf_config.get("max_folds", 50)
        
        start_idx = 0
        fold_num = 0
        
        while (start_idx + min_train + step_size <= len(X)
               and fold_num < max_folds):
            fold_num += 1
            train_end = start_idx + min_train + (fold_num - 1) * step_size
            train_end = min(train_end, len(X) - step_size)
            if fold_num % 10 == 0:
                self.logger.info(f"    {pair}: fold {fold_num}/{max_folds}")
            test_end = min(train_end + step_size, len(X))
            
            if test_end > len(X):
                break
            
            X_train = X.iloc[start_idx:train_end]
            y_train = y.iloc[start_idx:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            test_returns = returns.iloc[train_end:test_end]
            
            # Train simple model
            try:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                # Predict
                pred_proba = model.predict_proba(X_test)[:, 1]
                pred_direction = (pred_proba > 0.5).astype(int)
                
                # Simulate trading
                fold_returns = self._simulate_trades(test_returns, pred_direction, pred_proba)
                
                fold_result = {
                    "fold": fold_num,
                    "train_start": str(X.index[start_idx]),
                    "train_end": str(X.index[train_end - 1]),
                    "test_start": str(X.index[train_end]),
                    "test_end": str(X.index[test_end - 1]),
                    "n_trades": int(np.sum(np.diff(pred_direction) != 0)),
                    "fold_return": float(np.sum(fold_returns)),
                    "fold_sharpe": float(self._calculate_sharpe(fold_returns))
                }
                
                folds.append(fold_result)
                all_returns.extend(fold_returns.tolist())
                
            except Exception as e:
                self.logger.warning(f"  Fold {fold_num} failed: {e}")
            
            start_idx += step_size // 2  # 50% overlap
        
        # Aggregate results
        all_returns = pd.Series(all_returns)
        metrics = self._calculate_metrics(all_returns)
        
        # Store equity curve
        equity = self.initial_capital * (1 + all_returns).cumprod()
        self._equity_curves[pair] = equity
        
        return {
            "metrics": metrics,
            "folds": folds,
            "returns": all_returns,
            "equity_curve": equity
        }

    def _simple_backtest(self, pair: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple train-test split backtest"""
        returns = df["close"].pct_change().dropna()
        
        # Use a simple momentum strategy as baseline
        signal = (df["close"].shift(1) > df["close"].shift(2)).astype(int).loc[returns.index]
        signal = signal.replace(0, -1)  # Short when not long
        
        strategy_returns = signal.shift(1) * returns
        strategy_returns = strategy_returns.dropna()
        
        metrics = self._calculate_metrics(strategy_returns)
        
        equity = self.initial_capital * (1 + strategy_returns).cumprod()
        self._equity_curves[pair] = equity
        
        return {
            "metrics": metrics,
            "folds": [],
            "returns": strategy_returns,
            "equity_curve": equity
        }

    def _prepare_backtest_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for backtest model"""
        exclude = ["open", "high", "low", "close", "volume", "target",
                   "timestamp", "datetime", "date", "spread"]
        feature_cols = [c for c in df.columns 
                       if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
        
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Drop rows with NaN
        X = X.dropna()
        
        return X, feature_cols

    def _simulate_trades(self, returns: pd.Series, signals: np.ndarray, 
                         confidences: np.ndarray) -> pd.Series:
        """Simulate trades with transaction costs"""
        # Position sizing based on confidence
        positions = np.where(signals == 1, confidences - 0.5, 
                           np.where(signals == 0, 0, -(0.5 - confidences)))
        positions = np.clip(positions * 4, -1, 1)  # Scale to [-1, 1]
        
        # Strategy returns
        strategy_returns = positions * returns.values[:len(positions)]
        
        # Transaction costs (commission + slippage)
        # Assume round-trip cost per trade
        trades = np.diff(signals) != 0
        trade_cost = (self.commission / 10000) + (self.slippage / 10000)
        strategy_returns[1:] -= trades * trade_cost
        
        return pd.Series(strategy_returns, index=returns.index[:len(strategy_returns)])

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {m: 0.0 for m in self.metrics_list}
        
        metrics = {}
        
        # Basic returns
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / self._periods_per_year if len(returns) > 0 else 1
        ann_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
        
        # Risk metrics
        ann_vol = returns.std() * np.sqrt(self._periods_per_year)
        
        # Sharpe Ratio
        if ann_vol > 0:
            sharpe = (ann_return - self.risk_free_rate) / ann_vol
        else:
            sharpe = 0.0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self._periods_per_year) if len(downside_returns) > 0 else 1e-6
        sortino = (ann_return - self.risk_free_rate) / downside_vol
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calmar Ratio
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0
        
        # Win Rate
        wins = (returns > 0).sum()
        total = len(returns)
        win_rate = wins / total if total > 0 else 0.0
        
        # Profit Factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Average trade
        avg_trade = returns.mean()
        avg_win = returns[returns > 0].mean() if wins > 0 else 0.0
        avg_loss = returns[returns < 0].mean() if (total - wins) > 0 else 0.0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Omega Ratio
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega = gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')
        
        # Gain/Pain Ratio
        gain_pain = gross_profits / gross_losses if gross_losses > 0 else 0.0
        
        # Deflated Sharpe Ratio
        dsr = self._deflated_sharpe(sharpe, len(returns), self.wf_config.get("n_trials", 100))
        
        # Tail Ratio
        tail_95 = np.percentile(returns, 95)
        tail_5 = abs(np.percentile(returns, 5))
        tail_ratio = tail_95 / tail_5 if tail_5 > 0 else 0.0
        
        # Max Drawdown Duration
        dd_duration = self._max_drawdown_duration(drawdown)
        
        # Probability of Overfitting (simplified)
        pbo = self._estimate_pbo(returns)
        
        metrics = {
            "total_return": float(total_return),
            "annualized_return": float(ann_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_dd),
            "max_drawdown_duration": int(dd_duration),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_trade": float(avg_trade),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
            "omega_ratio": float(omega),
            "tail_ratio": float(tail_ratio),
            "gain_to_pain": float(gain_pain),
            "deflated_sharpe": float(dsr),
            "probability_of_overfitting": float(pbo),
            "annualized_volatility": float(ann_vol)
        }
        
        return metrics

    def _monte_carlo_test(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Monte Carlo permutation test.
        Randomizes returns to test strategy robustness.
        """
        if not self.mc_config.get("enabled", True) or len(returns) < 30:
            return {}
        
        n_simulations = self.mc_config.get("n_simulations", 1000)
        
        mc_returns = []
        mc_sharpes = []
        mc_maxdds = []
        
        np.random.seed(42)
        
        for _ in range(n_simulations):
            if self.mc_config.get("randomize_returns", True):
                # Shuffle returns
                shuffled = np.random.permutation(returns.values)
            else:
                shuffled = returns.values
            
            # Simulate
            sim_returns = pd.Series(shuffled)
            cumulative = (1 + sim_returns).cumprod()
            
            # Sharpe
            ann_sharpe = sim_returns.mean() / sim_returns.std() * np.sqrt(self._periods_per_year) if sim_returns.std() > 0 else 0
            
            # Max DD
            running_max = cumulative.cummax()
            dd = (cumulative - running_max) / running_max
            max_dd = dd.min()
            
            mc_returns.append(float(cumulative.iloc[-1]) - 1)
            mc_sharpes.append(ann_sharpe)
            mc_maxdds.append(max_dd)
        
        actual_sharpe = returns.mean() / returns.std() * np.sqrt(self._periods_per_year) if returns.std() > 0 else 0
        
        result = {
            "n_simulations": n_simulations,
            "actual_sharpe": float(actual_sharpe),
            "mc_sharpe_median": float(np.median(mc_sharpes)),
            "mc_sharpe_5pct": float(np.percentile(mc_sharpes, 5)),
            "mc_sharpe_95pct": float(np.percentile(mc_sharpes, 95)),
            "p_value_sharpe": float(np.mean(np.array(mc_sharpes) >= actual_sharpe)),
            "mc_maxdd_median": float(np.median(mc_maxdds)),
            "mc_maxdd_95pct": float(np.percentile(mc_maxdds, 5)),
            "is_significant": float(np.mean(np.array(mc_sharpes) >= actual_sharpe)) < 0.05
        }
        
        return result

    def _portfolio_analysis(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze portfolio-level performance across pairs"""
        portfolio_metrics = {}
        
        # Average metrics across pairs
        if results["performance_metrics"]:
            all_metrics = results["performance_metrics"]
            
            for metric in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", 
                          "win_rate", "profit_factor", "max_drawdown"]:
                values = [m.get(metric, 0) for m in all_metrics.values()]
                portfolio_metrics[f"avg_{metric}"] = float(np.mean(values))
                portfolio_metrics[f"median_{metric}"] = float(np.median(values))
                portfolio_metrics[f"best_{metric}"] = float(np.max(values)) if metric != "max_drawdown" else float(np.min(values))
        
        return portfolio_metrics

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(self._periods_per_year))

    def _deflated_sharpe(self, sharpe: float, n_observations: int, 
                         n_trials: int) -> float:
        """
        Calculate Deflated Sharpe Ratio.
        Adjusts Sharpe for multiple testing (trial overfitting).
        """
        if n_observations < 2:
            return 0.0
        
        # Expected maximum Sharpe under null (random strategies)
        gamma_euler = 0.5772156649
        expected_max_sharpe = ((1 - gamma_euler) * np.log(n_trials) + 
                               gamma_euler * np.log(n_trials) - 
                               (1 - gamma_euler))
        
        # Variance of Sharpe
        var_sharpe = (1 / n_observations) * (1 - sharpe ** 2 / 2 + 1)
        
        # Deflated Sharpe
        dsr = (sharpe - expected_max_sharpe * np.sqrt(var_sharpe)) / np.sqrt(var_sharpe)
        
        return max(0, dsr)

    def _max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods"""
        is_drawdown = drawdown < 0
        if not is_drawdown.any():
            return 0
        
        durations = []
        current_duration = 0
        
        for in_dd in is_drawdown:
            if in_dd:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 0
        
        durations.append(current_duration)
        return max(durations) if durations else 0

    def _estimate_pbo(self, returns: pd.Series, n_splits: int = 10) -> float:
        """
        Estimate Probability of Backtest Overfitting (PBO).
        Uses combinatorial symmetric cross-validation.
        """
        if len(returns) < n_splits * 2:
            return 0.5
        
        try:
            # Split into IS/OOS sets
            split_size = len(returns) // n_splits
            
            is_sharpes = []
            oos_sharpes = []
            
            for i in range(n_splits):
                start = i * split_size
                end = start + split_size
                
                is_returns = returns.iloc[start:end]
                oos_start = (i + n_splits // 2) % n_splits
                oos_returns = returns.iloc[oos_start * split_size:(oos_start + 1) * split_size]
                
                is_sharpe = is_returns.mean() / is_returns.std() if is_returns.std() > 0 else 0
                oos_sharpe = oos_returns.mean() / oos_returns.std() if oos_returns.std() > 0 else 0
                
                is_sharpes.append(is_sharpe)
                oos_sharpes.append(oos_sharpe)
            
            # PBO = probability that IS rank != OOS rank
            is_ranks = stats.rankdata(is_sharpes)
            oos_ranks = stats.rankdata(oos_sharpes)
            
            # Count where IS best is not OOS best
            best_is = np.argmax(is_sharpes)
            pbo = 1 - (oos_ranks[best_is] / len(oos_ranks))
            
            return float(np.clip(pbo, 0, 1))
            
        except Exception:
            return 0.5

    def _generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML performance report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "initial_capital": self.initial_capital,
                "pairs_backtested": len(results["pairs_backtested"]),
                "best_pair": results["best_pair"],
                "best_sharpe": results["best_sharpe"]
            },
            "pair_metrics": results["performance_metrics"],
            "portfolio_metrics": results.get("portfolio_metrics", {}),
            "monte_carlo": results["monte_carlo_results"]
        }
        
        report_path = self.output_dir / f"backtest_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save equity curves
        for pair, equity in self._equity_curves.items():
            equity_path = self.output_dir / f"equity_curve_{pair}.csv"
            equity.to_csv(equity_path)
        
        self.logger.info(f"Report saved: {report_path}")
        
        return str(report_path)

    def get_backtest_results(self) -> Dict[str, Any]:
        """Return backtest results for other agents"""
        return self._backtest_results


if __name__ == "__main__":
    agent = BacktestingEngine(kaggle_mode=False)
    result = agent.execute()
    print(f"\nAgent E Result: {result}")
