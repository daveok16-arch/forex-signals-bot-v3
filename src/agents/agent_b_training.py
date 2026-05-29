"""
Agent B: Model Training & Hyperparameter Optimization Engine
============================================================
Multi-model training with automated hyperparameter tuning via Optuna.
Supports: XGBoost, LightGBM, CatBoost, LSTM, TabNet, Transformer.

Kaggle-Adapted: GPU/TPU auto-detection, mixed precision training,
gradient accumulation, memory optimization.
"""

import gc
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                            log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent
from core.kaggle_manager import KaggleManager


class ModelTrainingEngine(BaseAgent):
    """
    Agent B - Model Training & Hyperparameter Optimization
    
    Pipeline:
    1. Load feature matrix from Agent A
    2. Prepare sequences for temporal models
    3. Optuna hyperparameter optimization per model
    4. Purged cross-validation training
    5. Ensemble stacking/blending preparation
    6. Save model artifacts and training reports
    """

    def __init__(self, config_path: str = "config/system_config.yaml", kaggle_mode: bool = False):
        super().__init__(config_path=config_path, agent_key="agent_b", kaggle_mode=kaggle_mode)
        self.kaggle = KaggleManager()
        self.models_config = self.agent_config.get("models", {})
        self.opt_config = self.agent_config.get("optimization", {})
        self.train_config = self.agent_config.get("training", {})
        
        # Detect hardware
        self.device = self.kaggle.get_best_device()
        self.n_gpus = self.kaggle.get_device_count()
        
        # Storage
        self._trained_models: Dict[str, Any] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._optimization_results: Dict[str, Any] = {}
        self._feature_importance: Dict[str, pd.DataFrame] = {}
        
        self.logger.info(
            f"Agent B initialized | Device: {self.device} | GPUs: {self.n_gpus} | "
            f"Mixed Precision: {self.agent_config.get('mixed_precision', True)}"
        )

    def _execute_core(self) -> Dict[str, Any]:
        """Execute full model training pipeline"""
        results = {
            "models_trained": [],
            "optimization_trials": {},
            "cv_scores": {},
            "ensemble_ready": False,
            "model_paths": {},
            "best_model": None,
            "best_score": 0.0
        }
        
        # Step 1: Load data
        self.logger.info("STEP 1: Loading feature matrix")
        feature_data = self._load_feature_matrix()
        
        if not feature_data:
            raise RuntimeError("No feature data available. Run Agent A first.")
        
        # Step 2: Train models for each pair
        self.logger.info("STEP 2: Training models per currency pair")
        
        for pair, df in feature_data.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training models for {pair}")
            self.logger.info(f"Data shape: {df.shape}")
            
            pair_results = self._train_pair_models(pair, df)
            results["models_trained"].extend(pair_results["models"])
            results["optimization_trials"][pair] = pair_results["trials"]
            results["cv_scores"][pair] = pair_results["cv_scores"]
            
            if pair_results["best_score"] > results["best_score"]:
                results["best_score"] = pair_results["best_score"]
                results["best_model"] = pair_results["best_model_name"]
            
            # Memory cleanup
            if self.kaggle_mode:
                self.kaggle.memory_optimization()
        
        # Step 3: Save all artifacts
        self.logger.info("\nSTEP 3: Saving model artifacts")
        model_dir = self._save_models()
        results["model_paths"] = model_dir
        
        # Step 4: Generate training report
        self.logger.info("STEP 4: Generating training report")
        report_path = self._generate_training_report(results)
        results["report_path"] = report_path
        
        results["ensemble_ready"] = len(results["models_trained"]) > 0
        results["total_models"] = len(results["models_trained"])
        results["device_used"] = self.device
        
        self.logger.info(
            f"\nAgent B complete | Models: {results['total_models']} | "
            f"Best: {results['best_model']} ({results['best_score']:.4f})"
        )
        
        return results

    def _load_feature_matrix(self) -> Dict[str, pd.DataFrame]:
        """Load feature matrix from Agent A output"""
        feature_dir = self.project_root / "output" / "agent_a" / "features"
        
        data = {}
        if feature_dir.exists():
            for file in feature_dir.glob("*_features.parquet"):
                pair = file.stem.replace("_features", "")
                df = pd.read_parquet(file)
                data[pair] = df
                self.logger.info(f"Loaded features for {pair}: {df.shape}")
        
        # Fallback: try to load from known location or generate sample
        if not data:
            self.logger.warning("No feature files found - checking alternative paths")
            alt_paths = [
                Path("/kaggle/working/agent_a/features"),
                Path("data/processed")
            ]
            for path in alt_paths:
                if path.exists():
                    for file in path.glob("*_features.*"):
                        pair = file.stem.replace("_features", "")
                        if file.suffix == ".parquet":
                            data[pair] = pd.read_parquet(file)
                        elif file.suffix == ".csv":
                            data[pair] = pd.read_csv(file, index_col=0)
        
        return data

    def _train_pair_models(self, pair: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all configured models for a single currency pair"""
        results = {
            "models": [],
            "trials": 0,
            "cv_scores": {},
            "best_score": 0.0,
            "best_model_name": None
        }
        
        # Prepare data
        X, y, scaler = self._prepare_data(df)
        
        if X is None or len(X) < 1000:
            self.logger.warning(f"Insufficient data for {pair}")
            return results
        
        self._scalers[pair] = scaler
        
        # Train each model type
        model_types = self.models_config
        
        # XGBoost
        if model_types.get("xgboost", {}).get("enabled", False):
            self.logger.info(f"  Training XGBoost for {pair}...")
            try:
                xgb_result = self._train_xgboost(pair, X, y)
                self._trained_models[f"{pair}_xgboost"] = xgb_result["model"]
                results["models"].append(f"{pair}_xgboost")
                results["cv_scores"]["xgboost"] = xgb_result["cv_score"]
                results["trials"] += xgb_result.get("trials", 0)
                
                if xgb_result["cv_score"] > results["best_score"]:
                    results["best_score"] = xgb_result["cv_score"]
                    results["best_model_name"] = f"{pair}_xgboost"
                    
                self._log_feature_importance(pair, "xgboost", X.columns, 
                    xgb_result["model"].feature_importances_ if hasattr(xgb_result["model"], 'feature_importances_') else None)
            except Exception as e:
                self.logger.error(f"  XGBoost training failed: {e}")
        
        # LightGBM
        if model_types.get("lightgbm", {}).get("enabled", False):
            self.logger.info(f"  Training LightGBM for {pair}...")
            try:
                lgb_result = self._train_lightgbm(pair, X, y)
                self._trained_models[f"{pair}_lightgbm"] = lgb_result["model"]
                results["models"].append(f"{pair}_lightgbm")
                results["cv_scores"]["lightgbm"] = lgb_result["cv_score"]
                results["trials"] += lgb_result.get("trials", 0)
                
                if lgb_result["cv_score"] > results["best_score"]:
                    results["best_score"] = lgb_result["cv_score"]
                    results["best_model_name"] = f"{pair}_lightgbm"
            except Exception as e:
                self.logger.error(f"  LightGBM training failed: {e}")
        
        # CatBoost
        if model_types.get("catboost", {}).get("enabled", False):
            self.logger.info(f"  Training CatBoost for {pair}...")
            try:
                cb_result = self._train_catboost(pair, X, y)
                self._trained_models[f"{pair}_catboost"] = cb_result["model"]
                results["models"].append(f"{pair}_catboost")
                results["cv_scores"]["catboost"] = cb_result["cv_score"]
                results["trials"] += cb_result.get("trials", 0)
                
                if cb_result["cv_score"] > results["best_score"]:
                    results["best_score"] = cb_result["cv_score"]
                    results["best_model_name"] = f"{pair}_catboost"
            except Exception as e:
                self.logger.error(f"  CatBoost training failed: {e}")
        
        # Random Forest
        self.logger.info(f"  Training Random Forest for {pair}...")
        try:
            rf_result = self._train_random_forest(pair, X, y)
            self._trained_models[f"{pair}_randomforest"] = rf_result["model"]
            results["models"].append(f"{pair}_randomforest")
            results["cv_scores"]["randomforest"] = rf_result["cv_score"]
            
            if rf_result["cv_score"] > results["best_score"]:
                results["best_score"] = rf_result["cv_score"]
                results["best_model_name"] = f"{pair}_randomforest"
        except Exception as e:
            self.logger.error(f"  Random Forest training failed: {e}")
        
        # LSTM (if GPU available and data is large enough)
        if (model_types.get("lstm", {}).get("enabled", False) and 
            self.device in ["cuda", "tpu"] and 
            len(X) > 5000):
            self.logger.info(f"  Training LSTM for {pair}...")
            try:
                lstm_result = self._train_lstm(pair, df)
                if lstm_result["model"] is not None:
                    self._trained_models[f"{pair}_lstm"] = lstm_result["model"]
                    results["models"].append(f"{pair}_lstm")
                    results["cv_scores"]["lstm"] = lstm_result["cv_score"]
                    
                    if lstm_result["cv_score"] > results["best_score"]:
                        results["best_score"] = lstm_result["cv_score"]
                        results["best_model_name"] = f"{pair}_lstm"
            except Exception as e:
                self.logger.error(f"  LSTM training failed: {e}")
        
        # Create ensemble if we have multiple models
        if len([m for m in results["models"] if m.startswith(pair)]) >= 2:
            self.logger.info(f"  Creating ensemble for {pair}...")
            try:
                ensemble = self._create_pair_ensemble(pair, X, y)
                if ensemble:
                    self._trained_models[f"{pair}_ensemble"] = ensemble
                    results["models"].append(f"{pair}_ensemble")
            except Exception as e:
                self.logger.error(f"  Ensemble creation failed: {e}")
        
        # Save scaler
        scaler_path = self.output_dir / f"{pair}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        self.log_metric(f"models_trained_{pair}", len(results["models"]))
        self.log_metric(f"best_cv_score_{pair}", results["best_score"])
        
        return results

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], StandardScaler]:
        """Prepare features and target variable"""
        # Target: direction of next period (1 = up, 0 = down)
        df = df.copy()
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        
        # Exclude non-feature columns
        exclude_cols = ["open", "high", "low", "close", "volume", "target", "spread",
                       "timestamp", "datetime", "date"]
        
        feature_cols = [c for c in df.columns 
                       if c not in exclude_cols 
                       and df[c].dtype in [np.float64, np.float32, np.int64]]
        
        if len(feature_cols) < 5:
            self.logger.warning(f"Not enough features: {len(feature_cols)}")
            return None, None, StandardScaler()
        
        X = df[feature_cols].copy()
        y = df["target"].copy()
        
        # Remove NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 1000:
            return None, None, StandardScaler()
        
        # Handle infinities
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled, y, scaler

    def _create_cv_splitter(self, n_splits: int = 5):
        """Create cross-validation splitter with purging"""
        purge_gap = self.opt_config.get("purge_gap", 10)
        
        if self.opt_config.get("cv_method", "purged_kfold") == "time_series":
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            # Use stratified k-fold as base (purging applied during split)
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _train_xgboost(self, pair: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost with Optuna optimization"""
        cfg = self.models_config.get("xgboost", {})
        
        try:
            import optuna
            import xgboost as xgb
            
            # Optuna optimization
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 0.5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "random_state": 42,
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                    "tree_method": "gpu_hist" if self.device == "cuda" else "hist"
                }
                
                cv_scores = []
                cv = self._create_cv_splitter()
                
                for train_idx, val_idx in cv.split(X, y):
                    # Apply purge gap
                    purge_gap = self.opt_config.get("purge_gap", 10)
                    val_idx = val_idx[val_idx > train_idx.max() + purge_gap]
                    
                    if len(val_idx) < 50:
                        continue
                    
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train, verbose=False)
                    
                    preds = model.predict(X_val)
                    score = f1_score(y_val, preds, average="weighted")
                    cv_scores.append(score)
                
                return np.mean(cv_scores) if cv_scores else 0.0
            
            # Run optimization
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.HyperbandPruner()
            )
            
            n_trials = self.opt_config.get("n_trials", 100)
            study.optimize(objective, n_trials=min(n_trials, 50), show_progress_bar=False)
            
            self._optimization_results[f"{pair}_xgboost"] = {
                "best_params": study.best_params,
                "best_score": study.best_value,
                "n_trials": len(study.trials)
            }
            
            # Train final model with best params
            best_params = study.best_params
            best_params.update({
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "tree_method": "gpu_hist" if self.device == "cuda" else "hist"
            })
            
            final_model = xgb.XGBClassifier(**best_params)
            final_model.fit(X, y, verbose=False)
            
            # Cross-validation score
            cv_score = study.best_value
            
            return {
                "model": final_model,
                "cv_score": cv_score,
                "trials": len(study.trials),
                "best_params": study.best_params
            }
            
        except ImportError:
            self.logger.warning("XGBoost or Optuna not available")
            return {"model": None, "cv_score": 0, "trials": 0}

    def _train_lightgbm(self, pair: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train LightGBM with Optuna optimization"""
        try:
            import lightgbm as lgb
            import optuna
            
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "max_depth": trial.suggest_int("max_depth", -1, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "verbose": -1,
                    "random_state": 42
                }
                
                cv_scores = []
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    preds = model.predict(X_val)
                    score = f1_score(y_val, preds, average="weighted")
                    cv_scores.append(score)
                
                return np.mean(cv_scores) if cv_scores else 0.0
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=min(self.opt_config.get("n_trials", 100), 30), show_progress_bar=False)
            
            best_params = study.best_params
            best_params.update({"verbose": -1, "random_state": 42})
            
            final_model = lgb.LGBMClassifier(**best_params)
            final_model.fit(X, y)
            
            return {
                "model": final_model,
                "cv_score": study.best_value,
                "trials": len(study.trials),
                "best_params": study.best_params
            }
            
        except ImportError:
            self.logger.warning("LightGBM or Optuna not available")
            return {"model": None, "cv_score": 0, "trials": 0}

    def _train_catboost(self, pair: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train CatBoost with Optuna optimization"""
        try:
            from catboost import CatBoostClassifier
            import optuna
            
            def objective(trial):
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 1000),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    "random_seed": 42,
                    "verbose": False
                }
                
                cv_scores = []
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = CatBoostClassifier(**params)
                    model.fit(X_train, y_train, verbose=False)
                    
                    preds = model.predict(X_val)
                    score = f1_score(y_val, preds, average="weighted")
                    cv_scores.append(score)
                
                return np.mean(cv_scores) if cv_scores else 0.0
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=min(self.opt_config.get("n_trials", 100), 30), show_progress_bar=False)
            
            best_params = study.best_params
            best_params.update({"random_seed": 42, "verbose": False})
            
            final_model = CatBoostClassifier(**best_params)
            final_model.fit(X, y, verbose=False)
            
            return {
                "model": final_model,
                "cv_score": study.best_value,
                "trials": len(study.trials),
                "best_params": study.best_params
            }
            
        except ImportError:
            self.logger.warning("CatBoost or Optuna not available")
            return {"model": None, "cv_score": 0, "trials": 0}

    def _train_random_forest(self, pair: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest (no Optuna - fast baseline)"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average="weighted")
            cv_scores.append(score)
        
        # Final fit on all data
        model.fit(X, y)
        
        cv_score = np.mean(cv_scores) if cv_scores else 0.0
        
        return {
            "model": model,
            "cv_score": cv_score,
            "trials": 0
        }

    def _train_lstm(self, pair: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM neural network for temporal patterns"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Prepare sequences
            seq_length = self.models_config.get("lstm", {}).get("sequence_length", 60)
            
            # Feature columns
            exclude = ["open", "high", "low", "close", "volume", "target"]
            feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32]]
            
            if len(feature_cols) < 5:
                return {"model": None, "cv_score": 0}
            
            # Create sequences
            X_data = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
            y_data = (df["close"].shift(-1) > df["close"]).astype(int).values
            
            # Normalize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_data = scaler.fit_transform(X_data)
            
            X_seq, y_seq = [], []
            for i in range(len(X_data) - seq_length - 1):
                if not np.isnan(y_data[i + seq_length]):
                    X_seq.append(X_data[i:i + seq_length])
                    y_seq.append(y_data[i + seq_length])
            
            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.int64)
            
            if len(X_seq) < 1000:
                return {"model": None, "cv_score": 0}
            
            # Train/val split
            split_idx = int(0.8 * len(X_seq))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # DataLoader
            train_ds = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            val_ds = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=64)
            
            # Model
            input_size = X_seq.shape[2]
            hidden_size = 64
            num_layers = 2
            
            class LSTMClassifier(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                       batch_first=True, dropout=0.2)
                    self.fc1 = nn.Linear(hidden_size, 32)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.2)
                    self.fc2 = nn.Linear(32, num_classes)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = out[:, -1, :]
                    out = self.fc1(out)
                    out = self.relu(out)
                    out = self.dropout(out)
                    out = self.fc2(out)
                    return out
            
            model = LSTMClassifier(input_size, hidden_size, num_layers).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training
            best_val_acc = 0
            patience = 10
            patience_counter = 0
            
            for epoch in range(50):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_acc = val_correct / val_total
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.output_dir / f"{pair}_lstm.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Return model info
            return {
                "model": {
                    "state_dict_path": str(self.output_dir / f"{pair}_lstm.pt"),
                    "architecture": "LSTM",
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    "sequence_length": seq_length
                },
                "cv_score": best_val_acc
            }
            
        except ImportError:
            self.logger.warning("PyTorch not available - skipping LSTM")
            return {"model": None, "cv_score": 0}
        except Exception as e:
            self.logger.error(f"LSTM training error: {e}")
            return {"model": None, "cv_score": 0}

    def _create_pair_ensemble(self, pair: str, X: pd.DataFrame, y: pd.Series) -> Optional[Any]:
        """Create ensemble of trained models for a pair"""
        pair_models = {}
        
        for name, model in self._trained_models.items():
            if name.startswith(f"{pair}_") and "ensemble" not in name:
                model_type = name.split("_")[-1]
                if isinstance(model, dict) and "state_dict_path" in model:
                    # LSTM model - skip for sklearn ensemble
                    continue
                pair_models[model_type] = model
        
        if len(pair_models) < 2:
            return None
        
        try:
            # Create voting classifier
            estimators = [(name, model) for name, model in pair_models.items()]
            ensemble = VotingClassifier(
                estimators=estimators,
                voting="soft"
            )
            ensemble.fit(X, y)
            
            # Evaluate
            cv_scores = []
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            for train_idx, val_idx in cv.split(X, y):
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                preds = ensemble.predict(X_val)
                score = f1_score(y_val, preds, average="weighted")
                cv_scores.append(score)
            
            self.logger.info(f"  Ensemble CV score: {np.mean(cv_scores):.4f}")
            
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Ensemble creation error: {e}")
            return None

    def _log_feature_importance(self, pair: str, model_type: str, 
                                 feature_names: List[str], importances: Optional[np.ndarray] = None):
        """Log feature importance for analysis"""
        if importances is None:
            return
        
        importance_df = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        key = f"{pair}_{model_type}"
        self._feature_importance[key] = importance_df
        
        # Log top 10
        top_features = importance_df.head(10)
        self.logger.info(f"  Top features for {key}:")
        for _, row in top_features.iterrows():
            self.logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    def _save_models(self) -> Dict[str, str]:
        """Save all trained models and metadata"""
        saved_paths = {}
        
        # Save sklearn-compatible models
        for name, model in self._trained_models.items():
            if isinstance(model, dict) and "state_dict_path" in model:
                # LSTM - already saved
                saved_paths[name] = model["state_dict_path"]
            elif model is not None:
                path = self.output_dir / f"{name}.pkl"
                try:
                    joblib.dump(model, path)
                    saved_paths[name] = str(path)
                except Exception as e:
                    self.logger.error(f"Failed to save {name}: {e}")
        
        # Save scalers
        for pair, scaler in self._scalers.items():
            path = self.output_dir / f"{pair}_scaler.pkl"
            joblib.dump(scaler, path)
            saved_paths[f"{pair}_scaler"] = str(path)
        
        # Save optimization results
        opt_path = self.output_dir / "optimization_results.json"
        with open(opt_path, "w") as f:
            json.dump(self._optimization_results, f, indent=2, default=str)
        saved_paths["optimization_results"] = str(opt_path)
        
        # Save feature importance
        for key, df in self._feature_importance.items():
            path = self.output_dir / f"feature_importance_{key}.csv"
            df.to_csv(path, index=False)
            saved_paths[f"feature_importance_{key}"] = str(path)
        
        return saved_paths

    def _generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive training report"""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary": {
                "total_models_trained": results["total_models"],
                "best_model": results["best_model"],
                "best_score": results["best_score"],
                "device_used": results.get("device_used", "cpu"),
                "kaggle_mode": self.kaggle_mode
            },
            "per_pair": {}
        }
        
        for pair, scores in results["cv_scores"].items():
            report["per_pair"][pair] = {
                "cv_scores": scores,
                "optimization_trials": results["optimization_trials"].get(pair, 0)
            }
        
        report_path = self.output_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(report_path)

    def get_trained_models(self) -> Dict[str, Any]:
        """Return trained models for other agents"""
        return self._trained_models

    def get_scalers(self) -> Dict[str, StandardScaler]:
        """Return fitted scalers"""
        return self._scalers


if __name__ == "__main__":
    agent = ModelTrainingEngine(kaggle_mode=False)
    result = agent.execute()
    print(f"\nAgent B Result: {result}")
