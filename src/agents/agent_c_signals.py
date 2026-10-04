"""
Agent C: Signal Generation & Ensemble Prediction Engine
======================================================
Converts model predictions into actionable trading signals.
Features: dynamic ensemble weighting, uncertainty quantification,
conformal prediction, signal decay, and multi-timeframe correlation.

Kaggle-Adapted: Uses pre-computed model artifacts, optimized inference.
"""

import gc
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent
from core.kaggle_manager import KaggleManager
from core.trend_strategy import TrendConfig, TrendStrategy


class SignalGenerationEngine(BaseAgent):
    """
    Agent C - Signal Generation & Ensemble Prediction
    
    Pipeline:
    1. Load trained models (from Agent B)
    2. Generate predictions from each model
    3. Apply dynamic ensemble weighting
    4. Quantify prediction uncertainty
    5. Create structured signal objects
    6. Apply signal filters and decay
    """

    def __init__(self, config_path: str = "config/system_config.yaml", kaggle_mode: bool = False):
        super().__init__(config_path=config_path, agent_key="agent_c", kaggle_mode=kaggle_mode)
        self.kaggle = KaggleManager()
        
        self.ensemble_method = self.agent_config.get("ensemble_method", "dynamic_weighted")
        self.min_confidence = self.agent_config.get("min_confidence", 0.65)
        self.max_confidence = self.agent_config.get("max_confidence", 0.95)
        self.signal_decay_minutes = self.agent_config.get("signal_decay_minutes", 60)
        self.prediction_horizon = self.agent_config.get("prediction_horizon", "H1")
        
        self.signal_filters = self.agent_config.get("signal_filters", {})
        self.uncertainty_config = self.agent_config.get("uncertainty_quantification", {})

        # Tradeable strategy: multi-horizon trend / time-series momentum. This is
        # the only edge that validated out-of-sample (see core/trend_strategy.py).
        trend_cfg = self.agent_config.get("trend_strategy", {})
        self.strategy_name = trend_cfg.get("strategy", "trend_momentum")
        self.use_strategy = trend_cfg.get("enabled", True)
        self.trend = TrendStrategy(TrendConfig(
            lookbacks=trend_cfg.get("lookbacks", [63, 126, 252]),
            vol_window=trend_cfg.get("vol_window", 60),
            target_vol=trend_cfg.get("target_vol", 0.10),
            max_leverage=trend_cfg.get("max_leverage", 3.0),
            atr_stop_mult=trend_cfg.get("atr_stop_mult", 2.0),
            reward_risk=trend_cfg.get("reward_risk", 2.0),
            min_strength=trend_cfg.get("min_strength", 0.5),
            horizon_days=trend_cfg.get("horizon_days", 21),
        ))
        
        # State
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._model_weights: Dict[str, Dict[str, float]] = {}
        self._signal_history: List[Dict[str, Any]] = []
        self._current_signals: Dict[str, Dict[str, Any]] = {}

    def _execute_core(self) -> Dict[str, Any]:
        """Execute signal generation pipeline"""
        results = {
            "signals_generated": 0,
            "signals_by_pair": {},
            "ensemble_weights": {},
            "uncertainty_estimates": {},
            "filtered_signals": 0,
            "final_signals": []
        }
        
        # Step 1: Load trained models
        self.logger.info("STEP 1: Loading trained models")
        self._load_models()
        
        if not self._models:
            raise RuntimeError("No trained models found. Run Agent B first.")
        
        # Step 2: Load latest feature data
        self.logger.info("STEP 2: Loading latest features")
        feature_data = self._load_latest_features()

        if self.use_strategy:
            self.logger.info(f"STEP 3: Generating {self.strategy_name} signals")
            return self._generate_strategy_signals(feature_data, results)

        # Step 3: Generate predictions per pair
        self.logger.info("STEP 3: Generating predictions")
        
        for pair, df in feature_data.items():
            pair_models = self._get_pair_models(pair)
            if not pair_models:
                continue
            
            self.logger.info(f"  Generating signals for {pair}")
            
            # Prepare features
            X = self._prepare_features(df, pair)
            if X is None or len(X) == 0:
                continue
            
            # Get predictions from each model
            predictions = self._get_model_predictions(pair, pair_models, X, df)
            
            # Ensemble predictions
            ensemble_pred, confidence, weights = self._ensemble_predictions(pair, predictions, X)
            
            # Uncertainty quantification
            uncertainty = self._quantify_uncertainty(pair, predictions, X)
            
            # Generate signals
            latest_idx = -1
            current_price = df["close"].iloc[latest_idx]
            candle_ts = (df["timestamp"].iloc[latest_idx]
                         if "timestamp" in df.columns else df.index[latest_idx])
            
            signal = self._create_signal(
                pair=pair,
                timestamp=candle_ts,
                current_price=current_price,
                prediction=ensemble_pred[latest_idx] if len(ensemble_pred) > 0 else 0.5,
                confidence=confidence,
                uncertainty=uncertainty,
                model_weights=weights,
                features_df=df
            )
            
            # Apply filters
            passed = self._apply_signal_filters(signal)
            self.logger.info(
                f"    {pair}: {signal['direction']} conf={signal['confidence']:.1f} "
                f"unc={signal['uncertainty'].get('mean_uncertainty', 0):.3f} "
                f"-> {'PASS' if passed else 'filtered'}"
            )
            if passed:
                self._current_signals[pair] = signal
                results["signals_generated"] += 1
                results["signals_by_pair"][pair] = signal
                results["ensemble_weights"][pair] = weights
                results["uncertainty_estimates"][pair] = uncertainty
            else:
                results["filtered_signals"] += 1
            
            # Memory cleanup
            gc.collect()
        
        # Step 4: Cross-pair correlation check
        self.logger.info("STEP 4: Cross-pair correlation analysis")
        final_signals = self._cross_pair_analysis(results["signals_by_pair"])
        results["final_signals"] = list(final_signals.values())
        
        # Step 5: Save signals
        self.logger.info("STEP 5: Saving signals")
        signals_path = self._save_signals(results)
        results["signals_file"] = signals_path
        
        self.logger.info(
            f"\nAgent C complete | Signals: {results['signals_generated']} | "
            f"Filtered: {results['filtered_signals']} | Final: {len(results['final_signals'])}"
        )
        
        return results

    def _load_models(self) -> None:
        """Load trained models from Agent B output"""
        model_dir = self.project_root / "output" / "agent_b"
        
        if not model_dir.exists():
            self.logger.warning(f"Model directory not found: {model_dir}")
            return
        
        # Load model files
        for file in model_dir.glob("*.pkl"):
            model_name = file.stem
            if "scaler" in model_name:
                pair = model_name.replace("_scaler", "")
                self._scalers[pair] = joblib.load(file)
                self.logger.info(f"  Loaded scaler for {pair}")
            else:
                try:
                    self._models[model_name] = joblib.load(file)
                    self.logger.info(f"  Loaded model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"  Failed to load {model_name}: {e}")
        
        # Load LSTM models (PyTorch)
        for file in model_dir.glob("*_lstm.pt"):
            pair = file.stem.replace("_lstm.pt", "")
            self._models[f"{pair}_lstm"] = {"state_dict_path": str(file), "type": "lstm"}
            self.logger.info(f"  Loaded LSTM model: {pair}")
        
        # Load optimization results for weighting
        opt_results_path = model_dir / "optimization_results.json"
        if opt_results_path.exists():
            with open(opt_results_path) as f:
                opt_results = json.load(f)
            
            # Initialize weights based on optimization scores
            for key, result in opt_results.items():
                if "best_score" in result:
                    pair_model = key.replace("_", "_", 1) if key.count("_") >= 2 else key
                    parts = key.rsplit("_", 1)
                    if len(parts) == 2:
                        pair, model_type = parts
                        if pair not in self._model_weights:
                            self._model_weights[pair] = {}
                        self._model_weights[pair][model_type] = max(0.1, result["best_score"])

    def _load_latest_features(self) -> Dict[str, pd.DataFrame]:
        """Load latest feature matrix"""
        feature_dir = self.project_root / "output" / "agent_a" / "features"
        
        data = {}
        if feature_dir.exists():
            for file in feature_dir.glob("*_features.parquet"):
                pair = file.stem.replace("_features", "")
                df = pd.read_parquet(file)
                # Keep only last 500 rows for speed
                data[pair] = df.tail(500)
        
        return data

    def _generate_strategy_signals(self, feature_data: Dict[str, pd.DataFrame],
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals from the validated trend strategy.

        Emits the same signal schema downstream agents (D/E/F) already consume,
        enriched with the strategy's entry/SL/TP/size and horizon.
        """
        raw = self.trend.generate(feature_data)
        if not raw:
            raise RuntimeError("Trend strategy produced no signals (insufficient data).")

        min_conf = self.min_confidence * 100
        signals: Dict[str, Dict[str, Any]] = {}
        for pair, s in raw.items():
            ts = feature_data[pair]["timestamp"].iloc[-1] if "timestamp" in feature_data[pair].columns \
                else datetime.utcnow()
            ts = ts if hasattr(ts, "isoformat") else pd.Timestamp.utcnow()
            action = s["direction"] != "HOLD"
            if action and s["confidence"] < min_conf:
                action = False

            self.logger.info(
                f"    {pair}: {s['direction']} conf={s['confidence']:.1f} "
                f"score={s['trend_score']:+.2f} -> {'PASS' if action else 'filtered'}"
            )
            if not action:
                results["filtered_signals"] += 1
                continue

            signal = {
                "pair": pair,
                "timestamp": ts.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "current_price": s["entry"],
                "direction": s["direction"],
                "strength": s["strength"],
                "confidence": s["confidence"],
                "prediction_probability": round(0.5 + s["trend_score"] / 2.0, 6),
                "strategy": s["strategy"],
                "trend_score": s["trend_score"],
                "momentum_z": s["momentum_z"],
                "lookback_returns": s["lookback_returns"],
                "suggested_stop_loss": s["stop_loss"],
                "suggested_take_profit": s["take_profit"],
                "suggested_size": s["vol_target_size"],
                "reward_risk": s["reward_risk"],
                "prediction_horizon_days": s["horizon_days"],
                "uncertainty": {"mean_uncertainty": round(1.0 - s["strength"], 4)},
                "market_context": {
                    "annualized_vol": s["annualized_vol"],
                    "atr_14": s.get("atr_14"),
                    "regime_note": s["regime_note"],
                },
                "model_weights": {self.strategy_name: 1.0},
                "prediction_horizon": self.prediction_horizon,
                "expires_at": (datetime.utcnow() + timedelta(minutes=self.signal_decay_minutes)).isoformat(),
                "status": "active",
            }
            signals[pair] = signal
            results["signals_generated"] += 1

        final = self._cross_pair_analysis(signals)
        results["signals_by_pair"] = signals
        results["final_signals"] = list(final.values())

        self.logger.info("STEP 5: Saving signals")
        results["signals_file"] = self._save_signals(results)
        self.logger.info(
            f"\nAgent C complete | Signals: {results['signals_generated']} | "
            f"Filtered: {results['filtered_signals']} | Final: {len(results['final_signals'])}"
        )
        return results

    def _get_pair_models(self, pair: str) -> Dict[str, Any]:
        """Get all trained models for a specific pair"""
        pair_models = {}
        for name, model in self._models.items():
            if name.startswith(f"{pair}_") and "ensemble" not in name and "scaler" not in name:
                model_type = name.split("_")[-1]
                pair_models[model_type] = model
        return pair_models

    def _prepare_features(self, df: pd.DataFrame, pair: str) -> Optional[pd.DataFrame]:
        """Prepare features for prediction"""
        # Exclude non-feature columns
        exclude = ["open", "high", "low", "close", "volume", "target", "spread",
                  "timestamp", "datetime", "date"]
        
        feature_cols = [c for c in df.columns 
                       if c not in exclude 
                       and df[c].dtype in [np.float64, np.float32, np.int64]]
        
        if len(feature_cols) < 5:
            return None
        
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Apply scaler if available
        if pair in self._scalers:
            try:
                X_scaled = pd.DataFrame(
                    self._scalers[pair].transform(X),
                    columns=X.columns,
                    index=X.index
                )
                return X_scaled
            except Exception as e:
                self.logger.warning(f"Scaler failed for {pair}: {e}")
        
        return X

    def _get_model_predictions(self, pair: str, models: Dict[str, Any], 
                                X: pd.DataFrame, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each model"""
        predictions = {}
        
        for model_type, model in models.items():
            try:
                if isinstance(model, dict) and model.get("type") == "lstm":
                    # LSTM prediction
                    pred = self._predict_lstm(model, df, pair)
                elif hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)[:, 1]
                elif hasattr(model, "predict"):
                    pred = model.predict(X)
                    if pred.ndim == 1 and set(np.unique(pred)).issubset({0, 1}):
                        pred = pred.astype(float)
                    else:
                        # Normalize to [0, 1]
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
                else:
                    continue
                
                predictions[model_type] = pred
                
            except Exception as e:
                self.logger.warning(f"  Prediction failed for {model_type}: {e}")
        
        return predictions

    def _predict_lstm(self, model_info: Dict, df: pd.DataFrame, pair: str) -> np.ndarray:
        """Predict using LSTM model"""
        try:
            import torch
            
            # Load model
            seq_length = model_info.get("sequence_length", 60)
            feature_cols = model_info.get("feature_cols", [])
            
            if not feature_cols:
                # Infer feature columns
                exclude = ["open", "high", "low", "close", "volume", "target"]
                feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32]]
            
            X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
            
            # Apply scaler
            scaler = model_info.get("scaler")
            if scaler:
                X = scaler.transform(X)
            
            # Create sequences
            X_seq = []
            for i in range(len(X) - seq_length + 1):
                X_seq.append(X[i:i + seq_length])
            
            if not X_seq:
                return np.array([0.5] * len(df))
            
            X_seq = np.array(X_seq, dtype=np.float32)
            
            # Load model architecture
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            class LSTMClassifier(torch.nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                             batch_first=True, dropout=0.2)
                    self.fc1 = torch.nn.Linear(hidden_size, 32)
                    self.fc2 = torch.nn.Linear(32, 2)
                
                def forward(self, x):
                    h0 = torch.zeros(2, x.size(0), 64).to(x.device)
                    c0 = torch.zeros(2, x.size(0), 64).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = out[:, -1, :]
                    out = torch.relu(self.fc1(out))
                    out = self.fc2(out)
                    return torch.softmax(out, dim=1)
            
            model = LSTMClassifier(X_seq.shape[2]).to(device)
            model.load_state_dict(torch.load(model_info["state_dict_path"], map_location=device))
            model.eval()
            
            # Predict
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq).to(device)
                outputs = model(X_tensor)
                probs = outputs[:, 1].cpu().numpy()
            
            # Pad to original length
            padded = np.full(len(df), 0.5)
            padded[seq_length - 1:] = probs
            
            return padded
            
        except Exception as e:
            self.logger.warning(f"LSTM prediction failed: {e}")
            return np.array([0.5] * len(df))

    def _ensemble_predictions(self, pair: str, predictions: Dict[str, np.ndarray],
                              X: pd.DataFrame) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Combine predictions using dynamic ensemble weighting.
        
        Methods:
        - dynamic_weighted: Weights based on recent performance
        - equal: Equal weights
        - confidence: Weight by prediction confidence
        """
        if not predictions:
            return np.array([0.5] * len(X)), 0.0, {}
        
        if len(predictions) == 1:
            pred = list(predictions.values())[0]
            confidence = float(np.abs(np.asarray(pred)[-1] - 0.5) * 2)
            return pred, confidence, {list(predictions.keys())[0]: 1.0}
        
        # Get or compute weights
        if pair in self._model_weights and len(self._model_weights[pair]) > 0:
            # Use stored weights
            raw_weights = self._model_weights[pair]
            # Filter to available models
            weights = {k: raw_weights.get(k, 0.5) for k in predictions.keys()}
        else:
            # Equal weights
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        # Compute weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for model_type, pred in predictions.items():
            w = weights.get(model_type, 0)
            ensemble_pred += w * np.array(pred)
        
        # Confidence on the latest bar: distance from 0.5, averaged across models
        latest = np.array([np.asarray(p)[-1] for p in predictions.values()])
        model_confidences = np.abs(latest - 0.5) * 2
        avg_confidence = float(np.mean(model_confidences))
        
        # Disagreement penalty
        disagreement = float(np.std(latest))
        confidence = avg_confidence * (1 - disagreement)
        
        return ensemble_pred, float(confidence), weights

    def _quantify_uncertainty(self, pair: str, predictions: Dict[str, np.ndarray],
                               X: pd.DataFrame) -> Dict[str, float]:
        """
        Quantify prediction uncertainty.
        
        Methods:
        - conformal: Prediction intervals via conformal prediction
        - bootstrap: Bootstrapped confidence intervals
        - ensemble_variance: Variance across ensemble members
        """
        method = self.uncertainty_config.get("method", "ensemble_variance")
        confidence_level = self.uncertainty_config.get("confidence_level", 0.90)
        
        uncertainty = {
            "method": method,
            "confidence_level": confidence_level,
            "mean_uncertainty": 0.0,
            "prediction_interval": (0.0, 1.0)
        }
        
        if not predictions or len(predictions) == 0:
            return uncertainty
        
        pred_array = np.array(list(predictions.values()))
        
        if method == "ensemble_variance":
            # Use variance across models as uncertainty
            variance = np.var(pred_array, axis=0)
            uncertainty["mean_uncertainty"] = float(np.mean(variance))
            uncertainty["prediction_interval"] = (
                float(np.clip(np.mean(pred_array, axis=0)[-1] - 1.96 * np.sqrt(variance[-1]), 0, 1)),
                float(np.clip(np.mean(pred_array, axis=0)[-1] + 1.96 * np.sqrt(variance[-1]), 0, 1))
            )
        
        elif method == "bootstrap":
            # Bootstrap confidence intervals
            n_bootstrap = 100
            bootstraps = []
            for _ in range(n_bootstrap):
                sample_idx = np.random.choice(pred_array.shape[0], size=pred_array.shape[0], replace=True)
                bootstraps.append(np.mean(pred_array[sample_idx, -1]))
            
            lower = np.percentile(bootstraps, (1 - confidence_level) * 50)
            upper = np.percentile(bootstraps, 100 - (1 - confidence_level) * 50)
            uncertainty["mean_uncertainty"] = float(np.std(bootstraps))
            uncertainty["prediction_interval"] = (float(lower), float(upper))
        
        elif method == "conformal":
            # Simplified conformal prediction
            alpha = 1 - confidence_level
            n_models = pred_array.shape[0]
            # Quantile-based interval
            lower = np.percentile(pred_array[:, -1], alpha * 100)
            upper = np.percentile(pred_array[:, -1], (1 - alpha) * 100)
            uncertainty["mean_uncertainty"] = float((upper - lower) / 2)
            uncertainty["prediction_interval"] = (float(lower), float(upper))
        
        return uncertainty

    def _create_signal(self, pair: str, timestamp: pd.Timestamp, current_price: float,
                       prediction: float, confidence: float, uncertainty: Dict,
                       model_weights: Dict[str, float], features_df: pd.DataFrame) -> Dict[str, Any]:
        """Create a structured signal object"""
        # Determine direction and strength
        if prediction > 0.55:
            direction = "BUY"
            strength = min((prediction - 0.5) / 0.5, 1.0)
        elif prediction < 0.45:
            direction = "SELL"
            strength = min((0.5 - prediction) / 0.5, 1.0)
        else:
            direction = "HOLD"
            strength = 0.0
        
        # Calculate confidence score (0-100)
        confidence_score = confidence * 100
        
        # Volatility regime from features
        vol_regime = "unknown"
        if "volatility_regime" in features_df.columns:
            vol_regime = "high" if features_df["volatility_regime"].iloc[-1] == 1 else "low"
        
        # Trend info
        trend = "neutral"
        if "trend_20" in features_df.columns:
            trend_20 = features_df["trend_20"].iloc[-1]
            trend_50 = features_df["trend_50"].iloc[-1] if "trend_50" in features_df.columns else trend_20
            if trend_20 == 1 and trend_50 == 1:
                trend = "bullish"
            elif trend_20 == 0 and trend_50 == 0:
                trend = "bearish"
            else:
                trend = "mixed"
        
        # Feature parquet is saved without an index, so the candle timestamp may
        # not be a datetime - fall back to the current time in that case.
        ts = timestamp if hasattr(timestamp, "isoformat") else pd.Timestamp.utcnow()

        signal = {
            "pair": pair,
            "timestamp": ts.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
            "current_price": round(float(current_price), 5),
            "direction": direction,
            "strength": round(float(strength), 4),
            "confidence": round(float(confidence_score), 2),
            "prediction_probability": round(float(prediction), 6),
            "uncertainty": uncertainty,
            "prediction_interval": uncertainty.get("prediction_interval", (0, 1)),
            "market_context": {
                "volatility_regime": vol_regime,
                "trend": trend,
                "rsi_14": float(features_df["rsi_14"].iloc[-1]) if "rsi_14" in features_df.columns else None,
                "atr_14": float(features_df["atr_14"].iloc[-1]) if "atr_14" in features_df.columns else None
            },
            "model_weights": {k: round(v, 4) for k, v in model_weights.items()},
            "prediction_horizon": self.prediction_horizon,
            "expires_at": (datetime.utcnow() + timedelta(minutes=self.signal_decay_minutes)).isoformat(),
            "status": "active"
        }
        
        return signal

    def _apply_signal_filters(self, signal: Dict[str, Any]) -> bool:
        """Apply filters to determine if signal should be emitted"""
        min_models = self.signal_filters.get("min_models_agree", 3)
        min_confidence = self.min_confidence * 100
        consecutive_min = self.signal_filters.get("consecutive_signal_min", 1)
        
        # Filter 1: Minimum confidence
        if signal["confidence"] < min_confidence:
            self.logger.debug(f"  Filtered: confidence {signal['confidence']:.1f} < {min_confidence}")
            return False
        
        # Filter 2: Minimum models contributing
        n_models = len([w for w in signal["model_weights"].values() if w > 0])
        if n_models < min_models:
            self.logger.debug(f"  Filtered: models {n_models} < {min_models}")
            return False
        
        # Filter 3: Not a HOLD signal
        if signal["direction"] == "HOLD":
            return False
        
        # Filter 4: Uncertainty check
        uncertainty = signal["uncertainty"].get("mean_uncertainty", 0)
        if uncertainty > 0.15:  # High uncertainty
            self.logger.debug(f"  Filtered: uncertainty {uncertainty:.3f} > 0.15")
            return False
        
        return True

    def _cross_pair_analysis(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Analyze signals across pairs for correlation and conflicts.
        Filters signals that conflict with strongly correlated pairs.
        """
        if len(signals) <= 1:
            return signals
        
        correlation_threshold = self.signal_filters.get("correlation_threshold", 0.7)
        
        # Known correlations between forex pairs
        pair_correlations = {
            ("EURUSD", "GBPUSD"): 0.85,
            ("EURUSD", "AUDUSD"): 0.75,
            ("EURUSD", "NZDUSD"): 0.70,
            ("GBPUSD", "AUDUSD"): 0.80,
            ("USDJPY", "USDCHF"): 0.65,
            ("USDCAD", "AUDUSD"): -0.70,
            ("EURUSD", "USDCAD"): -0.65,
            ("XAUUSD", "EURUSD"): 0.60,
        }
        
        final_signals = signals.copy()
        
        # Check for conflicts in correlated pairs
        for (pair1, pair2), corr in pair_correlations.items():
            if abs(corr) < correlation_threshold:
                continue
            
            if pair1 in signals and pair2 in signals:
                sig1 = signals[pair1]
                sig2 = signals[pair2]
                
                # If positively correlated, signals should agree
                if corr > 0:
                    if sig1["direction"] != sig2["direction"]:
                        # Remove the lower confidence signal
                        if sig1["confidence"] < sig2["confidence"]:
                            self.logger.info(f"  Conflict: {pair1} vs {pair2} - removing {pair1}")
                            final_signals.pop(pair1, None)
                        else:
                            self.logger.info(f"  Conflict: {pair1} vs {pair2} - removing {pair2}")
                            final_signals.pop(pair2, None)
                
                # If negatively correlated, signals should disagree
                elif corr < 0:
                    if sig1["direction"] == sig2["direction"]:
                        # Remove the lower confidence signal
                        if sig1["confidence"] < sig2["confidence"]:
                            final_signals.pop(pair1, None)
                        else:
                            final_signals.pop(pair2, None)
        
        return final_signals

    def _save_signals(self, results: Dict[str, Any]) -> str:
        """Save generated signals"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        signals_file = self.output_dir / f"signals_{timestamp}.json"
        with open(signals_file, "w") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "signals": results["final_signals"],
                "metadata": {
                    "total_generated": results["signals_generated"],
                    "filtered_out": results["filtered_signals"],
                    "ensemble_method": self.ensemble_method
                }
            }, f, indent=2, default=str)
        
        # Save CSV for easy viewing
        if results["final_signals"]:
            signals_df = pd.DataFrame([
                {
                    "pair": s["pair"],
                    "direction": s["direction"],
                    "strength": s["strength"],
                    "confidence": s["confidence"],
                    "price": s["current_price"],
                    "uncertainty": s.get("uncertainty", {}).get("mean_uncertainty", 0),
                    "trend": s.get("market_context", {}).get("trend", s.get("strategy", "")),
                    "volatility": s.get("market_context", {}).get("volatility_regime",
                                    s.get("market_context", {}).get("annualized_vol", ""))
                }
                for s in results["final_signals"]
            ])
            csv_file = self.output_dir / f"signals_{timestamp}.csv"
            signals_df.to_csv(csv_file, index=False)
        
        # Update history
        self._signal_history.extend(results["final_signals"])
        history_file = self.output_dir / "signal_history.json"
        with open(history_file, "w") as f:
            json.dump(self._signal_history, f, indent=2, default=str)
        
        return str(signals_file)

    def get_current_signals(self) -> Dict[str, Dict[str, Any]]:
        """Return current active signals"""
        # Remove expired signals
        now = datetime.utcnow()
        expired = []
        for pair, signal in self._current_signals.items():
            expires = datetime.fromisoformat(signal["expires_at"].replace("Z", "+00:00"))
            if now > expires:
                expired.append(pair)
        
        for pair in expired:
            self._current_signals[pair]["status"] = "expired"
            del self._current_signals[pair]
        
        return self._current_signals

    def update_model_weights(self, pair: str, model_type: str, performance_score: float) -> None:
        """Update ensemble weights based on recent performance"""
        if pair not in self._model_weights:
            self._model_weights[pair] = {}
        
        # Exponential moving average update
        alpha = 0.3  # Learning rate for weight updates
        current = self._model_weights[pair].get(model_type, 0.5)
        self._model_weights[pair][model_type] = alpha * performance_score + (1 - alpha) * current
        
        self.logger.info(f"Updated weight for {pair}/{model_type}: {self._model_weights[pair][model_type]:.4f}")


if __name__ == "__main__":
    agent = SignalGenerationEngine(kaggle_mode=False)
    result = agent.execute()
    print(f"\nAgent C Result: {result}")
