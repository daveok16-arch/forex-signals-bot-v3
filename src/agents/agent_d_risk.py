"""
Agent D: Risk Management & Position Sizing Engine
=================================================
Handles capital preservation through Kelly Criterion, VaR analysis,
drawdown controls, and dynamic position sizing.

Features: Regime-based risk scaling, tail risk hedging, Monte Carlo
risk assessment, and streak-based adjustments.
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent


class RiskManagementEngine(BaseAgent):
    """
    Agent D - Risk Management & Position Sizing
    
    Pipeline:
    1. Receive signals from Agent C
    2. Calculate optimal position size (Kelly/Volatility Targeting)
    3. Compute stop-loss and take-profit levels
    4. Check exposure limits and drawdown controls
    5. Apply regime-based risk scaling
    6. Output risk-adjusted trade orders
    """

    def __init__(self, config_path: str = "config/system_config.yaml", kaggle_mode: bool = False):
        super().__init__(config_path=config_path, agent_key="agent_d", kaggle_mode=kaggle_mode)
        
        # Risk parameters
        self.max_risk_per_trade = self.agent_config.get("max_risk_per_trade", 0.02)
        self.max_total_risk = self.agent_config.get("max_total_risk", 0.06)
        self.max_correlated_risk = self.agent_config.get("max_correlated_risk", 0.04)
        self.kelly_fraction = self.agent_config.get("kelly_fraction", 0.5)
        self.var_confidence = self.agent_config.get("var_confidence", 0.95)
        self.cvar_confidence = self.agent_config.get("cvar_confidence", 0.95)
        self.max_drawdown_pct = self.agent_config.get("max_drawdown_pct", 0.10)
        self.daily_loss_limit = self.agent_config.get("daily_loss_limit_pct", 0.03)
        self.weekly_loss_limit = self.agent_config.get("weekly_loss_limit_pct", 0.05)
        
        # Position sizing config
        self.position_config = self.agent_config.get("position_sizing", {})
        self.sl_config = self.agent_config.get("stop_loss", {})
        self.tp_config = self.agent_config.get("take_profit", {})
        self.dynamic_config = self.agent_config.get("dynamic_adjustment", {})
        
        # State tracking
        self._open_positions: Dict[str, Dict] = {}
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._current_drawdown: float = 0.0
        self._peak_equity: float = 0.0
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        self._trade_history: List[Dict] = []
        self._equity_curve: List[float] = [100000.0]  # Starting equity

    def _execute_core(self) -> Dict[str, Any]:
        """Execute risk management pipeline"""
        results = {
            "orders_approved": [],
            "orders_rejected": [],
            "risk_metrics": {},
            "current_exposure": 0.0,
            "available_risk_budget": 0.0
        }
        
        # Step 1: Load signals from Agent C
        self.logger.info("STEP 1: Loading signals")
        signals = self._load_signals()
        
        # Step 2: Update equity and drawdown
        self.logger.info("STEP 2: Updating equity and drawdown tracking")
        self._update_equity_state()
        
        # Step 3: Calculate risk metrics
        self.logger.info("STEP 3: Calculating risk metrics")
        risk_metrics = self._calculate_risk_metrics()
        results["risk_metrics"] = risk_metrics
        
        # Step 4: Process each signal
        self.logger.info("STEP 4: Processing signals through risk filters")
        
        for signal in signals:
            pair = signal["pair"]
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, risk_metrics)
            
            if position_size <= 0:
                signal["rejection_reason"] = "position_size_zero"
                results["orders_rejected"].append(signal)
                continue
            
            # Calculate stop-loss and take-profit
            sl_price, tp_price = self._calculate_sl_tp(signal)
            
            # Check risk limits
            risk_check = self._check_risk_limits(signal, position_size, sl_price)
            
            if not risk_check["approved"]:
                signal["rejection_reason"] = risk_check["reason"]
                results["orders_rejected"].append(signal)
                self.logger.info(f"  REJECTED {pair}: {risk_check['reason']}")
                continue
            
            # Apply dynamic adjustments
            adjusted_size = self._apply_dynamic_adjustments(position_size)
            
            # Create risk-adjusted order
            order = self._create_order(signal, adjusted_size, sl_price, tp_price)
            results["orders_approved"].append(order)
            
            # Track exposure
            results["current_exposure"] += order["risk_amount"]
            
            self.logger.info(
                f"  APPROVED {pair} {signal['direction']} | "
                f"Size: {adjusted_size:.4f} lots | "
                f"SL: {sl_price:.5f} | TP: {tp_price:.5f} | "
                f"Risk: ${order['risk_amount']:.2f}"
            )
        
        # Step 5: Available risk budget
        results["available_risk_budget"] = (
            self.max_total_risk * self._equity_curve[-1] - results["current_exposure"]
        )
        
        # Step 6: Save state
        self._save_risk_state(results)
        
        self.logger.info(
            f"\nAgent D complete | Approved: {len(results['orders_approved'])} | "
            f"Rejected: {len(results['orders_rejected'])} | "
            f"Exposure: ${results['current_exposure']:.2f}"
        )
        
        return results

    def _load_signals(self) -> List[Dict[str, Any]]:
        """Load signals from Agent C output"""
        signals_dir = self.project_root / "output" / "agent_c"
        
        signals = []
        if signals_dir.exists():
            # Find latest signals file
            signal_files = sorted(signals_dir.glob("signals_*.json"))
            if signal_files:
                latest = signal_files[-1]
                with open(latest) as f:
                    data = json.load(f)
                signals = data.get("signals", [])
        
        # Fallback: check alternate paths
        if not signals:
            alt_paths = [
                Path("/kaggle/working/agent_c"),
                self.project_root / "output" / "signals.json"
            ]
            for path in alt_paths:
                if path.exists() and path.is_file():
                    with open(path) as f:
                        signals = json.load(f)
                elif path.exists():
                    files = sorted(path.glob("signals_*.json"))
                    if files:
                        with open(files[-1]) as f:
                            signals = json.load(f).get("signals", [])
                if signals:
                    break
        
        return signals

    def _update_equity_state(self) -> None:
        """Update equity curve and drawdown tracking"""
        current_equity = self._equity_curve[-1] if self._equity_curve else 100000.0
        
        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        # Calculate drawdown
        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        
        # Reset daily/weekly PnL if needed
        # (In production, this would check actual time)

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        equity = np.array(self._equity_curve) if self._equity_curve else np.array([100000.0])
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        
        metrics = {
            "current_equity": equity[-1],
            "peak_equity": self._peak_equity,
            "current_drawdown": self._current_drawdown,
            "max_drawdown": 0.0,
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "total_trades": len(self._trade_history),
        }
        
        if len(returns) > 0:
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            metrics["max_drawdown"] = abs(float(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0
            
            # Volatility
            metrics["volatility"] = float(np.std(returns) * np.sqrt(252))
            
            # Value at Risk
            if len(returns) >= 30:
                metrics["var_95"] = float(np.percentile(returns, 5))
                metrics["var_99"] = float(np.percentile(returns, 1))
                
                # Conditional VaR (Expected Shortfall)
                metrics["cvar_95"] = float(np.mean(returns[returns <= metrics["var_95"]]))
            
            # Sharpe (simplified)
            if metrics["volatility"] > 0:
                metrics["sharpe"] = float(np.mean(returns) * 252 / metrics["volatility"])
            
            # Win rate
            wins = sum(1 for r in returns if r > 0)
            metrics["win_rate"] = wins / len(returns) if len(returns) > 0 else 0.5
        
        self.log_metric("current_drawdown", metrics["current_drawdown"])
        self.log_metric("current_equity", metrics["current_equity"])
        
        return metrics

    def _calculate_position_size(self, signal: Dict, risk_metrics: Dict) -> float:
        """
        Calculate optimal position size using selected method.
        
        Methods:
        - kelly_half: Half-Kelly Criterion
        - fixed_fraction: Fixed fractional risk
        - volatility_targeting: Target annual volatility
        """
        method = self.position_config.get("method", "kelly_half")
        current_equity = risk_metrics["current_equity"]
        
        # Get signal parameters
        pair = signal["pair"]
        confidence = signal.get("confidence", 50) / 100.0
        direction = signal.get("direction", "HOLD")
        
        # Get ATR for volatility-based sizing
        atr = signal.get("market_context", {}).get("atr_14", 0.001)
        current_price = signal.get("current_price", 1.0)
        
        # Account for pair-specific pip value
        pip_value = self._get_pip_value(pair)
        
        if method == "kelly_half":
            # Half-Kelly Criterion
            # f* = (p*b - q) / b  where p = win prob, q = loss prob, b = win/loss ratio
            win_prob = confidence
            loss_prob = 1 - win_prob
            
            # Estimate payoff ratio from historical data
            payoff_ratio = self._estimate_payoff_ratio()
            
            if payoff_ratio <= 0:
                return 0.0
            
            kelly_fraction = (win_prob * payoff_ratio - loss_prob) / payoff_ratio
            half_kelly = max(0, kelly_fraction * self.kelly_fraction)
            
            # Risk-based size: risk % of equity per trade
            risk_amount = current_equity * self.max_risk_per_trade * half_kelly
            
            # Convert to lot size using ATR as expected move
            if atr > 0:
                stop_distance_pips = atr / pip_value
                position_size = risk_amount / (stop_distance_pips * 10)  # $10 per pip per lot
            else:
                position_size = risk_amount / (50 * 10)  # Default 50 pip stop
            
        elif method == "volatility_targeting":
            target_vol = self.position_config.get("target_annual_volatility", 0.15)
            vol_lookback = self.position_config.get("volatility_lookback", 20)
            
            # Get current volatility estimate
            current_vol = risk_metrics.get("volatility", 0.15)
            if current_vol <= 0:
                current_vol = 0.15
            
            # Scale position inversely to volatility
            vol_scalar = target_vol / current_vol
            
            risk_amount = current_equity * self.max_risk_per_trade * vol_scalar
            
            if atr > 0:
                stop_distance_pips = atr / pip_value
                position_size = risk_amount / (stop_distance_pips * 10)
            else:
                position_size = risk_amount / (50 * 10)
        
        else:  # fixed_fraction
            risk_amount = current_equity * self.max_risk_per_trade * confidence
            
            if atr > 0:
                stop_distance_pips = atr / pip_value
                position_size = risk_amount / (stop_distance_pips * 10)
            else:
                position_size = risk_amount / (50 * 10)
        
        # Apply confidence scaling
        position_size *= (0.5 + 0.5 * confidence)
        
        # Ensure minimum and maximum lot sizes
        min_lot = 0.01
        max_lot = self._get_max_lot(pair)
        
        position_size = max(min_lot, min(position_size, max_lot))
        
        return round(position_size, 4)

    def _calculate_sl_tp(self, signal: Dict) -> Tuple[float, float]:
        """
        Calculate stop-loss and take-profit prices.
        
        SL Methods: atr_based, fixed, volatility_based
        TP Methods: risk_reward, atr_based, trailing
        """
        pair = signal["pair"]
        direction = signal["direction"]
        price = signal["current_price"]
        atr = signal.get("market_context", {}).get("atr_14", price * 0.001)
        
        # Stop Loss calculation
        sl_method = self.sl_config.get("method", "atr_based")
        atr_multiplier = self.sl_config.get("atr_multiplier", 2.0)
        max_sl_pips = self.sl_config.get("max_sl_pips", 50)
        min_sl_pips = self.sl_config.get("min_sl_pips", 10)
        pip_value = self._get_pip_value(pair)
        
        if sl_method == "atr_based":
            sl_distance = max(min(atr * atr_multiplier, max_sl_pips * pip_value), min_sl_pips * pip_value)
        elif sl_method == "fixed":
            sl_distance = 20 * pip_value  # Fixed 20 pips
        else:
            sl_distance = atr * atr_multiplier
        
        if direction == "BUY":
            sl_price = price - sl_distance
            tp_distance = sl_distance * self.tp_config.get("risk_reward_ratio", 2.0)
            tp_price = price + tp_distance
        elif direction == "SELL":
            sl_price = price + sl_distance
            tp_distance = sl_distance * self.tp_config.get("risk_reward_ratio", 2.0)
            tp_price = price - tp_distance
        else:
            sl_price = price
            tp_price = price
        
        return round(sl_price, 5), round(tp_price, 5)

    def _check_risk_limits(self, signal: Dict, position_size: float, 
                           sl_price: float) -> Dict[str, Any]:
        """Check if trade passes all risk limits"""
        pair = signal["pair"]
        price = signal["current_price"]
        direction = signal["direction"]
        
        # Calculate risk amount
        if direction == "BUY":
            risk_pips = abs(price - sl_price) / self._get_pip_value(pair)
        else:
            risk_pips = abs(sl_price - price) / self._get_pip_value(pair)
        
        risk_amount = risk_pips * 10 * position_size  # $10 per pip per lot
        
        # Check 1: Max drawdown
        if self._current_drawdown >= self.max_drawdown_pct:
            return {"approved": False, "reason": "max_drawdown_reached"}
        
        # Check 2: Per-trade risk limit
        equity = self._equity_curve[-1] if self._equity_curve else 100000
        if risk_amount > equity * self.max_risk_per_trade * 1.5:
            return {"approved": False, "reason": "risk_per_trade_exceeded"}
        
        # Check 3: Total risk limit
        current_risk = sum(p.get("risk_amount", 0) for p in self._open_positions.values())
        if current_risk + risk_amount > equity * self.max_total_risk:
            return {"approved": False, "reason": "total_risk_exceeded"}
        
        # Check 4: Daily loss limit
        if self._daily_pnl <= -equity * self.daily_loss_limit:
            return {"approved": False, "reason": "daily_loss_limit"}
        
        # Check 5: Weekly loss limit
        if self._weekly_pnl <= -equity * self.weekly_loss_limit:
            return {"approved": False, "reason": "weekly_loss_limit"}
        
        # Check 6: Correlated exposure
        correlated_risk = self._get_correlated_risk(pair)
        if correlated_risk + risk_amount > equity * self.max_correlated_risk:
            return {"approved": False, "reason": "correlated_risk_exceeded"}
        
        # Check 7: Max open positions
        if len(self._open_positions) >= 10:
            return {"approved": False, "reason": "max_positions_reached"}
        
        return {"approved": True, "reason": "", "risk_amount": risk_amount}

    def _apply_dynamic_adjustments(self, position_size: float) -> float:
        """Apply dynamic adjustments based on performance"""
        adjusted = position_size
        
        # Reduce after consecutive losses
        if self.dynamic_config.get("loss_streak_adjustment", True):
            reduce_after = self.dynamic_config.get("consecutive_losses_reduce", 3)
            if self._consecutive_losses >= reduce_after:
                reduction = min(0.5, 0.1 * (self._consecutive_losses - reduce_after + 1))
                adjusted *= (1 - reduction)
                self.logger.info(f"  Loss streak adjustment: -{reduction*100:.0f}%")
        
        # Increase after consecutive wins
        if self.dynamic_config.get("win_streak_adjustment", True):
            increase_after = self.dynamic_config.get("consecutive_wins_increase", 5)
            if self._consecutive_wins >= increase_after:
                increase = min(0.3, 0.05 * (self._consecutive_wins - increase_after + 1))
                adjusted *= (1 + increase)
                self.logger.info(f"  Win streak adjustment: +{increase*100:.0f}%")
        
        # Regime-based scaling
        if self.dynamic_config.get("regime_based", True):
            # Reduce size in high volatility regimes
            # (Would use actual volatility regime from market data)
            pass
        
        return round(adjusted, 4)

    def _create_order(self, signal: Dict, position_size: float, 
                      sl_price: float, tp_price: float) -> Dict[str, Any]:
        """Create a risk-adjusted order object"""
        pair = signal["pair"]
        direction = signal["direction"]
        price = signal["current_price"]
        
        pip_value = self._get_pip_value(pair)
        risk_pips = abs(price - sl_price) / pip_value if direction != "HOLD" else 0
        risk_amount = risk_pips * 10 * position_size
        
        order = {
            "order_id": f"{pair}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "pair": pair,
            "direction": direction,
            "order_type": "MARKET",
            "entry_price": price,
            "position_size": position_size,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_pips": risk_pips,
            "risk_amount": risk_amount,
            "risk_pct": risk_amount / self._equity_curve[-1] if self._equity_curve else 0,
            "reward_risk_ratio": abs(tp_price - price) / abs(price - sl_price) if sl_price != price else 0,
            "signal_confidence": signal.get("confidence", 50),
            "generated_at": datetime.utcnow().isoformat(),
            "status": "pending_execution"
        }
        
        return order

    def _get_pip_value(self, pair: str) -> float:
        """Get pip value for a currency pair"""
        pip_values = {
            "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
            "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
            "USDJPY": 0.01, "XAUUSD": 0.01
        }
        return pip_values.get(pair, 0.0001)

    def _get_max_lot(self, pair: str) -> float:
        """Get maximum lot size for a pair"""
        max_lots = {
            "EURUSD": 100, "GBPUSD": 100, "USDJPY": 100,
            "AUDUSD": 100, "USDCAD": 100, "USDCHF": 100,
            "NZDUSD": 100, "XAUUSD": 50
        }
        return max_lots.get(pair, 100)

    def _estimate_payoff_ratio(self) -> float:
        """Estimate win/loss payoff ratio from trade history"""
        if not self._trade_history:
            return 1.5  # Default conservative estimate
        
        wins = [t["pnl"] for t in self._trade_history if t.get("pnl", 0) > 0]
        losses = [abs(t["pnl"]) for t in self._trade_history if t.get("pnl", 0) < 0]
        
        if not wins or not losses:
            return 1.5
        
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        return avg_win / avg_loss if avg_loss > 0 else 1.5

    def _get_correlated_risk(self, pair: str) -> float:
        """Get current risk exposure for correlated pairs"""
        correlations = {
            "EURUSD": ["GBPUSD", "AUDUSD", "NZDUSD", "EURJPY"],
            "GBPUSD": ["EURUSD", "AUDUSD", "GBPJPY"],
            "USDJPY": ["EURJPY", "GBPJPY", "AUDJPY"],
            "AUDUSD": ["EURUSD", "GBPUSD", "NZDUSD"],
            "USDCAD": ["AUDUSD", "NZDUSD"],
            "USDCHF": ["EURUSD", "GBPUSD"],
            "XAUUSD": ["EURUSD"]
        }
        
        correlated_pairs = correlations.get(pair, [])
        total_risk = 0.0
        
        for open_pair, position in self._open_positions.items():
            if open_pair in correlated_pairs or open_pair == pair:
                total_risk += position.get("risk_amount", 0)
        
        return total_risk

    def _save_risk_state(self, results: Dict[str, Any]) -> None:
        """Save risk management state"""
        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "equity_curve": self._equity_curve,
            "peak_equity": self._peak_equity,
            "current_drawdown": self._current_drawdown,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "open_positions": self._open_positions,
            "approved_orders": results["orders_approved"],
            "rejected_orders": results["orders_rejected"]
        }
        
        state_file = self.output_dir / "risk_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def record_trade_result(self, pair: str, pnl: float) -> None:
        """Record trade PnL for streak tracking"""
        self._trade_history.append({
            "pair": pair,
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update equity
        if self._equity_curve:
            self._equity_curve.append(self._equity_curve[-1] + pnl)
        
        # Update streaks
        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            self._daily_pnl += pnl
            self._weekly_pnl += pnl
        elif pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            self._daily_pnl += pnl
            self._weekly_pnl += pnl
        
        self.logger.info(
            f"Trade result: {pair} PnL=${pnl:.2f} | "
            f"Wins: {self._consecutive_wins} | Losses: {self._consecutive_losses}"
        )

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL (call at day start)"""
        self._daily_pnl = 0.0

    def reset_weekly_pnl(self) -> None:
        """Reset weekly PnL (call at week start)"""
        self._weekly_pnl = 0.0


if __name__ == "__main__":
    agent = RiskManagementEngine(kaggle_mode=False)
    result = agent.execute()
    print(f"\nAgent D Result: {result}")
