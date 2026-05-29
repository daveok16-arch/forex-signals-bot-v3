"""
Agent A: Data Ingestion & Feature Engineering Engine
====================================================
Handles multi-source forex data collection, advanced technical indicator
computation, statistical feature generation, and intelligent feature selection.

Kaggle-Adapted: Auto-detects Kaggle environment, mounts datasets,
optimizes memory, and uses GPU-accelerated computations where available.
"""

import gc
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent, AgentResult
from core.kaggle_manager import KaggleManager


class DataIngestionEngine(BaseAgent):
    """
    Agent A - Data Ingestion & Feature Engineering
    
    Pipeline:
    1. Fetch raw OHLCV data (multi-source)
    2. Data validation and quality scoring
    3. Technical indicator computation (150+)
    4. Statistical feature engineering
    5. Feature selection (MRMR / Boruta-SHAP)
    6. Output feature matrix
    """

    def __init__(self, config_path: str = "config/system_config.yaml", kaggle_mode: bool = False):
        super().__init__(config_path=config_path, agent_key="agent_a", kaggle_mode=kaggle_mode)
        self.kaggle = KaggleManager()
        self.pairs_config = self.config.get("pairs", {}).get("primary", [])
        self.timeframes = self.config.get("timeframes", [])
        self.indicators_config = self.agent_config.get("technical_indicators", {})
        self.statistical_features = self.agent_config.get("statistical_features", [])
        self.feature_selection_config = self.agent_config.get("feature_selection", {})
        self.data_sources = self.agent_config.get("data_sources", [])
        self.kaggle_datasets = self.agent_config.get("kaggle_datasets", [])
        self.lookback_days = self.agent_config.get("lookback_days", 252)
        self._feature_store: Dict[str, pd.DataFrame] = {}

    def _execute_core(self) -> Dict[str, Any]:
        """Execute the full data pipeline"""
        results = {
            "pairs_processed": [],
            "features_generated": 0,
            "features_selected": 0,
            "data_quality_scores": {},
            "feature_matrix_path": "",
            "metadata": {}
        }
        
        # Step 1: Load or fetch data
        self.logger.info("STEP 1: Loading forex data")
        raw_data = self._load_data()
        
        # Step 2: Validate and clean
        self.logger.info("STEP 2: Data validation and quality scoring")
        clean_data = {}
        for pair, df in raw_data.items():
            quality_score = self._validate_data_quality(df, pair)
            results["data_quality_scores"][pair] = quality_score
            if quality_score >= self.agent_config.get("data_quality_threshold", 0.95):
                clean_data[pair] = self._clean_data(df)
                results["pairs_processed"].append(pair)
            else:
                self.logger.warning(f"Data quality too low for {pair}: {quality_score:.3f}")
        
        # Step 3: Compute technical indicators
        self.logger.info("STEP 3: Computing technical indicators")
        indicator_data = {}
        for pair, df in clean_data.items():
            indicator_data[pair] = self._compute_indicators(df)
            self.logger.info(f"  {pair}: {indicator_data[pair].shape[1] - df.shape[1]} indicators added")
        
        # Step 4: Generate statistical features
        self.logger.info("STEP 4: Generating statistical features")
        enriched_data = {}
        for pair, df in indicator_data.items():
            enriched_data[pair] = self._compute_statistical_features(df)
        
        # Step 5: Feature selection
        self.logger.info("STEP 5: Feature selection")
        final_data = {}
        total_features = 0
        for pair, df in enriched_data.items():
            selected_df, n_selected = self._select_features(df, pair)
            final_data[pair] = selected_df
            total_features += n_selected
        
        results["features_generated"] = sum(df.shape[1] for df in enriched_data.values())
        results["features_selected"] = total_features
        
        # Step 6: Save feature store
        self.logger.info("STEP 6: Saving feature matrix")
        matrix_path = self._save_feature_matrix(final_data)
        results["feature_matrix_path"] = matrix_path
        results["metadata"] = {
            "pairs": list(final_data.keys()),
            "timeframes": [tf["name"] for tf in self.timeframes],
            "lookback_days": self.lookback_days,
            "feature_selection_method": self.feature_selection_config.get("method", "mrmr"),
            "total_rows": sum(len(df) for df in final_data.values()),
        }
        
        # Memory optimization for Kaggle
        if self.kaggle_mode:
            self.kaggle.memory_optimization()
        
        self.logger.info(
            f"Agent A complete | Pairs: {len(final_data)} | "
            f"Features: {total_features} | Rows: {results['metadata']['total_rows']}"
        )
        
        return results

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load forex data from available sources.
        Priority: Kaggle datasets > OANDA API > Local files > Synthetic demo
        """
        raw_data = {}
        
        # Try Kaggle datasets first (if in Kaggle environment)
        if self.kaggle_mode and self.kaggle_datasets:
            self.logger.info("Attempting Kaggle dataset loading")
            for dataset in self.kaggle_datasets:
                try:
                    path = self.kaggle.mount_dataset(dataset)
                    data = self._load_from_directory(path)
                    raw_data.update(data)
                    self.logger.info(f"Loaded from Kaggle dataset: {dataset}")
                except Exception as e:
                    self.logger.warning(f"Failed to load Kaggle dataset {dataset}: {e}")
        
        # Try local data directory
        if not raw_data:
            data_dir = self.project_root / "data" / "raw"
            if data_dir.exists():
                raw_data = self._load_from_directory(data_dir)
        
        # Generate synthetic data as fallback for demo/testing
        if not raw_data:
            self.logger.info("No external data found - generating synthetic forex data for testing")
            raw_data = self._generate_synthetic_data()
        
        return raw_data

    def _load_from_directory(self, path: Path) -> Dict[str, pd.DataFrame]:
        """Load all CSV/Parquet files from a directory"""
        data = {}
        for file in path.glob("*"):
            if file.suffix in [".csv", ".parquet", ".pkl"]:
                pair_name = file.stem.upper().replace("_", "").replace("-", "")
                try:
                    if file.suffix == ".csv":
                        df = pd.read_csv(file, parse_dates=["timestamp"], index_col="timestamp")
                    elif file.suffix == ".parquet":
                        df = pd.read_parquet(file)
                    else:
                        df = pd.read_pickle(file)
                    data[pair_name] = df
                except Exception as e:
                    self.logger.warning(f"Failed to load {file}: {e}")
        return data

    def _generate_synthetic_data(self, n_rows: int = 50000) -> Dict[str, pd.DataFrame]:
        """
        Generate realistic synthetic OHLCV data for testing.
        Uses geometric Brownian motion with realistic forex characteristics.
        """
        np.random.seed(42)
        data = {}
        
        for pair_config in self.pairs_config:
            pair = pair_config["symbol"]
            # Start price around typical forex levels
            base_prices = {
                "EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 149.50,
                "AUDUSD": 0.6550, "USDCAD": 1.3650, "USDCHF": 0.8850,
                "NZDUSD": 0.5950, "XAUUSD": 2025.00
            }
            
            S0 = base_prices.get(pair, 1.0)
            dt = 1 / (24 * 12)  # 5-minute increments
            mu = 0.00002  # Slight drift
            sigma = 0.0003  # Volatility
            
            returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_rows)
            prices = S0 * np.exp(np.cumsum(returns))
            
            # Generate OHLCV from close prices
            timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_rows, freq="5min")
            spread = pair_config.get("spread_avg", 0.0001)
            
            df = pd.DataFrame(index=timestamps)
            df["close"] = prices
            df["open"] = df["close"].shift(1).fillna(S0)
            df["high"] = df[["open", "close"]].max(axis=1) + np.random.uniform(0, spread * 2, n_rows)
            df["low"] = df[["open", "close"]].min(axis=1) - np.random.uniform(0, spread * 2, n_rows)
            df["volume"] = np.random.lognormal(10, 1, n_rows).astype(int)
            df["spread"] = spread
            
            data[pair] = df
            self.logger.info(f"Generated {n_rows} rows of synthetic data for {pair}")
        
        return data

    def _validate_data_quality(self, df: pd.DataFrame, pair: str) -> float:
        """
        Validate data quality and return a score (0-1).
        Checks: missing values, duplicate timestamps, price consistency, outliers.
        """
        scores = []
        
        # Check 1: Missing values (< 1% expected)
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(max(0, 1 - missing_pct * 100))
        
        # Check 2: No duplicate timestamps
        dup_pct = df.index.duplicated().sum() / len(df)
        scores.append(1 - dup_pct)
        
        # Check 3: Price consistency (high >= low, high >= close >= low, etc.)
        price_ok = (
            (df["high"] >= df["low"]).all() and
            (df["high"] >= df["close"]).all() and
            (df["close"] >= df["low"]).all() and
            (df["high"] >= df["open"]).all() and
            (df["open"] >= df["low"]).all()
        )
        scores.append(1.0 if price_ok else 0.0)
        
        # Check 4: No extreme outliers (> 10 sigma from mean return)
        returns = df["close"].pct_change().dropna()
        outlier_pct = (np.abs(returns) > 10 * returns.std()).sum() / len(returns)
        scores.append(max(0, 1 - outlier_pct * 100))
        
        # Check 5: Volume present and non-zero
        vol_ok = (df["volume"] > 0).mean()
        scores.append(vol_ok)
        
        # Check 6: Chronological order
        chronological = (df.index.to_series().diff().dropna() > pd.Timedelta(0)).all()
        scores.append(1.0 if chronological else 0.0)
        
        final_score = np.mean(scores)
        self.log_metric(f"data_quality_{pair}", round(final_score, 4))
        
        return final_score

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw OHLCV data"""
        df = df.copy()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Forward fill small gaps (up to 5 periods)
        df = df.ffill(limit=5)
        
        # Remove rows with any remaining NaN in core columns
        core_cols = ["open", "high", "low", "close", "volume"]
        df = df.dropna(subset=[c for c in core_cols if c in df.columns])
        
        # Ensure numeric types
        for col in core_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive technical indicators.
        Categories: Overlap, Momentum, Volume, Volatility
        """
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # === OVERLAP STUDIES ===
        overlap = self.indicators_config.get("overlap", [])
        
        if "sma" in overlap:
            for period in [5, 10, 20, 50, 100, 200]:
                df[f"sma_{period}"] = close.rolling(period).mean()
        
        if "ema" in overlap:
            for period in [5, 10, 12, 20, 26, 50, 200]:
                df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()
        
        if "wma" in overlap:
            for period in [10, 20, 50]:
                weights = np.arange(1, period + 1)
                df[f"wma_{period}"] = close.rolling(period).apply(
                    lambda x: np.dot(x, weights) / weights.sum(), raw=True
                )
        
        if "bbands" in overlap:
            bb_period = 20
            bb_std = 2.0
            sma = close.rolling(bb_period).mean()
            std = close.rolling(bb_period).std()
            df["bb_upper"] = sma + bb_std * std
            df["bb_middle"] = sma
            df["bb_lower"] = sma - bb_std * std
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        if "sar" in overlap:
            df["sar"] = self._compute_parabolic_sar(high, low, close)
        
        if "midpoint" in overlap:
            for period in [10, 14, 20]:
                df[f"midpoint_{period}"] = (high.rolling(period).max() + low.rolling(period).min()) / 2
        
        # === MOMENTUM INDICATORS ===
        momentum = self.indicators_config.get("momentum", [])
        
        if "rsi" in momentum:
            for period in [6, 10, 14, 21]:
                df[f"rsi_{period}"] = self._compute_rsi(close, period)
        
        if "macd" in momentum:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["macd_line"] = ema12 - ema26
            df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        
        if "stoch" in momentum:
            for k_period, d_period in [(14, 3), (21, 5)]:
                lowest_low = low.rolling(k_period).min()
                highest_high = high.rolling(k_period).max()
                df[f"stoch_k_{k_period}"] = 100 * (close - lowest_low) / (highest_high - lowest_low)
                df[f"stoch_d_{k_period}"] = df[f"stoch_k_{k_period}"].rolling(d_period).mean()
        
        if "cci" in momentum:
            for period in [14, 20]:
                tp = (high + low + close) / 3
                sma_tp = tp.rolling(period).mean()
                mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
                df[f"cci_{period}"] = (tp - sma_tp) / (0.015 * mean_dev)
        
        if "adx" in momentum:
            df["adx"], df["di_plus"], df["di_minus"] = self._compute_adx(high, low, close, 14)
        
        if "willr" in momentum:
            for period in [10, 14]:
                highest = high.rolling(period).max()
                lowest = low.rolling(period).min()
                df[f"willr_{period}"] = -100 * (highest - close) / (highest - lowest)
        
        if "aroon" in momentum:
            for period in [14, 25]:
                df[f"aroon_up_{period}"] = 100 * (
                    high.rolling(period).apply(lambda x: x.argmax(), raw=True) / (period - 1)
                )
                df[f"aroon_down_{period}"] = 100 * (
                    low.rolling(period).apply(lambda x: x.argmin(), raw=True) / (period - 1)
                )
                df[f"aroon_osc_{period}"] = df[f"aroon_up_{period}"] - df[f"aroon_down_{period}"]
        
        if "momentum" in momentum or "mom" in momentum:
            for period in [10, 14, 20]:
                df[f"momentum_{period}"] = close - close.shift(period)
        
        if "roc" in momentum:
            for period in [10, 12, 25]:
                df[f"roc_{period}"] = 100 * (close / close.shift(period) - 1)
        
        if "ppo" in momentum:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["ppo"] = 100 * (ema12 - ema26) / ema26
            df["ppo_signal"] = df["ppo"].ewm(span=9, adjust=False).mean()
            df["ppo_hist"] = df["ppo"] - df["ppo_signal"]
        
        # === VOLUME INDICATORS ===
        vol = self.indicators_config.get("volume", [])
        
        if "obv" in vol:
            df["obv"] = self._compute_obv(close, volume)
        
        if "ad" in vol or "cmf" in vol:
            mfm = ((close - low) - (high - close)) / (high - low)
            mfv = mfm * volume
            df["ad_line"] = mfv.cumsum()
            
            if "cmf" in vol:
                df["cmf_20"] = mfm.rolling(20).mean()
        
        if "adosc" in vol:
            ad = ((close - low) - (high - close)) / (high - low) * volume
            df["adosc"] = ad.ewm(span=3, adjust=False).mean() - ad.ewm(span=10, adjust=False).mean()
        
        # Volume moving averages
        df["vol_sma_20"] = volume.rolling(20).mean()
        df["vol_ema_20"] = volume.ewm(span=20, adjust=False).mean()
        df["vol_ratio"] = volume / df["vol_sma_20"]
        
        # === VOLATILITY INDICATORS ===
        vol_ind = self.indicators_config.get("volatility", [])
        
        if "atr" in vol_ind or "natr" in vol_ind:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            for period in [10, 14, 20]:
                df[f"atr_{period}"] = tr.rolling(period).mean()
            
            df["natr_14"] = 100 * df["atr_14"] / close
            df["trange"] = tr
        
        # Volatility regime
        df["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252 * 24 * 12)
        df["volatility_regime"] = (df["volatility_20"] > df["volatility_20"].rolling(100).mean()).astype(int)
        
        # Keltner Channels
        if "keltner" not in [c.lower() for c in df.columns]:
            ema20 = close.ewm(span=20, adjust=False).mean()
            atr20 = df.get("atr_20", df["atr_14"] if "atr_14" in df.columns else tr.rolling(20).mean())
            df["kc_upper"] = ema20 + 2 * atr20
            df["kc_lower"] = ema20 - 2 * atr20
            df["kc_position"] = (close - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"])
        
        # Drop NaN rows from indicator calculations
        df = df.dropna()
        
        return df

    def _compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate advanced statistical features.
        Includes: returns, fractional differentiation, wavelet features, etc.
        """
        df = df.copy()
        close = df["close"]
        
        stat_features = self.agent_config.get("statistical_features", [])
        
        if "returns" in stat_features:
            df["returns"] = close.pct_change()
            df["log_returns"] = np.log(close / close.shift(1))
        
        if "realized_volatility" in stat_features:
            for window in [5, 10, 20, 60]:
                df[f"realized_vol_{window}"] = df["returns"].rolling(window).std() * np.sqrt(window)
        
        if "skewness" in stat_features:
            for window in [10, 20, 50]:
                df[f"skewness_{window}"] = df["returns"].rolling(window).skew()
        
        if "kurtosis" in stat_features:
            for window in [10, 20, 50]:
                df[f"kurtosis_{window}"] = df["returns"].rolling(window).kurt()
        
        if "z_score" in stat_features:
            for window in [20, 50, 100]:
                rolling_mean = close.rolling(window).mean()
                rolling_std = close.rolling(window).std()
                df[f"z_score_{window}"] = (close - rolling_mean) / rolling_std
        
        if "percentile_rank" in stat_features:
            for window in [20, 50, 100]:
                df[f"pct_rank_{window}"] = close.rolling(window).apply(
                    lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False
                )
        
        # Price-derived features
        df["price_range"] = df["high"] - df["low"]
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["body_to_range"] = df["body_size"] / (df["price_range"] + 1e-10)
        
        # Trend features
        df["trend_20"] = (close > close.rolling(20).mean()).astype(int)
        df["trend_50"] = (close > close.rolling(50).mean()).astype(int)
        df["golden_cross"] = (
            (close.rolling(50).mean() > close.rolling(200).mean()) &
            (close.rolling(50).mean().shift(1) <= close.rolling(200).mean().shift(1))
        ).astype(int)
        
        # Mean reversion features
        for window in [20, 50]:
            rolling_mean = close.rolling(window).mean()
            df[f"dist_from_mean_{window}"] = (close - rolling_mean) / rolling_mean
        
        # Autoregressive features
        for lag in [1, 2, 3, 5, 10]:
            df[f"close_lag_{lag}"] = close.shift(lag)
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
        
        # Moving average crossovers
        df["sma_ratio_10_50"] = close.rolling(10).mean() / close.rolling(50).mean()
        df["ema_ratio_12_26"] = close.ewm(span=12, adjust=False).mean() / close.ewm(span=26, adjust=False).mean()
        
        # Clean up NaN
        df = df.dropna()
        
        return df

    def _select_features(self, df: pd.DataFrame, pair: str) -> Tuple[pd.DataFrame, int]:
        """
        Select most informative features using MRMR (Minimum Redundancy Maximum Relevance).
        Falls back to correlation-based selection if needed.
        """
        method = self.feature_selection_config.get("method", "mrmr")
        n_features = self.feature_selection_config.get("n_features", 80)
        min_importance = self.feature_selection_config.get("min_importance", 0.01)
        
        # Define target: next period direction
        df_features = df.copy()
        df_features["target_direction"] = (df_features["close"].shift(-1) > df_features["close"]).astype(int)
        df_features = df_features.dropna()
        
        if len(df_features) < 1000:
            self.logger.warning(f"Insufficient data for feature selection: {len(df_features)} rows")
            return df, df.shape[1]
        
        # Get feature columns (exclude OHLCV, target, and non-numeric)
        exclude = ["open", "high", "low", "close", "volume", "target_direction", "spread"]
        feature_cols = [c for c in df_features.columns 
                       if c not in exclude and df_features[c].dtype in [np.float64, np.float32, np.int64]]
        
        if len(feature_cols) <= n_features:
            self.logger.info(f"Feature count ({len(feature_cols)}) <= target ({n_features}), keeping all")
            return df, len(feature_cols)
        
        try:
            if method == "mrmr":
                selected = self._mrmr_selection(df_features, feature_cols, n_features)
            elif method == "mutual_info":
                selected = self._mutual_info_selection(df_features, feature_cols, n_features)
            else:
                selected = self._correlation_selection(df_features, feature_cols, n_features)
        except Exception as e:
            self.logger.warning(f"Feature selection failed ({method}): {e}, using correlation fallback")
            selected = self._correlation_selection(df_features, feature_cols, n_features)
        
        # Always keep OHLCV + selected features
        keep_cols = ["open", "high", "low", "close", "volume"] + selected
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        result = df[keep_cols].copy()
        self.log_metric(f"features_selected_{pair}", len(selected))
        
        return result, len(selected)

    def _mrmr_selection(self, df: pd.DataFrame, features: List[str], n_select: int) -> List[str]:
        """Minimum Redundancy Maximum Relevance feature selection"""
        target = "target_direction"
        
        # Calculate relevance (mutual information approximation using correlation)
        relevance = {}
        for f in features:
            relevance[f] = abs(df[f].corr(df[target]))
        
        # Initialize with most relevant feature
        selected = [max(relevance, key=relevance.get)]
        remaining = set(features) - set(selected)
        
        # Iteratively add features
        while len(selected) < n_select and remaining:
            best_score = -np.inf
            best_feature = None
            
            for f in remaining:
                # Relevance term
                rel = relevance[f]
                
                # Redundancy term (average correlation with selected features)
                if selected:
                    red = np.mean([abs(df[f].corr(df[s])) for s in selected])
                else:
                    red = 0
                
                # MRMR score
                score = rel - red
                
                if score > best_score:
                    best_score = score
                    best_feature = f
            
            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
            else:
                break
        
        return selected

    def _mutual_info_selection(self, df: pd.DataFrame, features: List[str], n_select: int) -> List[str]:
        """Mutual information-based feature selection"""
        from sklearn.feature_selection import mutual_info_classif
        
        X = df[features].fillna(0)
        y = df["target_direction"]
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        feature_scores = list(zip(features, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [f for f, s in feature_scores[:n_select]]

    def _correlation_selection(self, df: pd.DataFrame, features: List[str], n_select: int) -> List[str]:
        """Correlation-based feature selection (fallback)"""
        correlations = {}
        for f in features:
            correlations[f] = abs(df[f].corr(df["target_direction"]))
        
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [f for f, s in sorted_features[:n_select]]

    def _save_feature_matrix(self, data: Dict[str, pd.DataFrame]) -> str:
        """Save feature matrices for all pairs"""
        # Save individual pair features
        for pair, df in data.items():
            self.save_artifact(df, f"{pair}_features.parquet", subdir="features")
        
        # Save combined metadata
        metadata = {
            "pairs": list(data.keys()),
            "feature_counts": {p: d.shape[1] for p, d in data.items()},
            "row_counts": {p: len(d) for p, d in data.items()},
            "columns": {p: list(d.columns) for p, d in data.items()},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self.save_artifact(metadata, "feature_metadata.json", subdir="features")
        
        # Store in memory for other agents
        self._feature_store = data
        
        return str(self.output_dir / "features")

    # ===== Helper Methods =====
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Compute Parabolic SAR"""
        sar = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        
        # Initialize
        trend.iloc[0] = 1  # 1 = uptrend, -1 = downtrend
        sar.iloc[0] = low.iloc[0]
        ep = high.iloc[0]  # Extreme point
        current_af = af
        
        for i in range(1, len(close)):
            # SAR calculation
            sar.iloc[i] = sar.iloc[i-1] + current_af * (ep - sar.iloc[i-1])
            
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[max(0, i-2)])
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    current_af = min(current_af + af, max_af)
                if sar.iloc[i] > low.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    current_af = af
                else:
                    trend.iloc[i] = 1
            else:  # Downtrend
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[max(0, i-2)])
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    current_af = min(current_af + af, max_af)
                if sar.iloc[i] < high.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    current_af = af
                else:
                    trend.iloc[i] = -1
        
        return sar

    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
        """Compute ADX, DI+, DI-"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx, plus_di, minus_di

    def _compute_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    def get_feature_store(self) -> Dict[str, pd.DataFrame]:
        """Return the feature store for other agents to consume"""
        return self._feature_store


# ===== Standalone Execution =====

if __name__ == "__main__":
    # Test the agent
    agent = DataIngestionEngine(kaggle_mode=False)
    result = agent.execute()
    print(f"\nAgent A Result: {result}")
