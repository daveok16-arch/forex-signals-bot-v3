"""
Base Agent Class for Forex ML Bot
All 7 agents inherit from this base class.
Provides: config loading, logging, metrics, Kaggle detection, cloud sync hooks
"""

import os
import sys
import time
import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import yaml


@dataclass
class AgentState:
    """Standardized agent state tracking"""
    agent_id: str
    status: str = "idle"  # idle | running | error | completed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    memory_usage_mb: float = 0.0


@dataclass
class AgentResult:
    """Standardized agent execution result"""
    success: bool
    agent_name: str
    duration_seconds: float
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Abstract base class for all Forex ML Bot agents.
    
    Agent naming convention:
    - Agent A: DataIngestionEngine
    - Agent B: ModelTrainingEngine  
    - Agent C: SignalGenerationEngine
    - Agent D: RiskManagementEngine
    - Agent E: BacktestingEngine
    - Agent F: TradeExecutionEngine
    - Agent G: MonitoringCloudSyncEngine
    """

    def __init__(
        self,
        config_path: str = "config/system_config.yaml",
        agent_key: Optional[str] = None,
        kaggle_mode: bool = False
    ):
        self.agent_key = agent_key or self.__class__.__name__.lower()
        self.agent_id = str(uuid.uuid4())[:8]
        self.config = self._load_config(config_path)
        self.agent_config = self.config.get("agents", {}).get(self.agent_key, {})
        self.kaggle_mode = kaggle_mode or self._detect_kaggle()
        self.correlation_id = str(uuid.uuid4())[:12]
        
        # Setup paths
        self.project_root = self._get_project_root()
        self.output_dir = self._setup_output_dir()
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # State tracking
        self.state = AgentState(agent_id=self.agent_id)
        self._execution_history: List[Dict[str, Any]] = []
        
        # Log initialization
        self.logger.info(
            f"Agent initialized | Type: {self.__class__.__name__} | "
            f"ID: {self.agent_id} | Kaggle: {self.kaggle_mode} | "
            f"CorrID: {self.correlation_id}"
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration with environment variable substitution"""
        full_path = self._get_project_root() / config_path
        try:
            with open(full_path, "r") as f:
                config = yaml.safe_load(f)
            config = self._substitute_env_vars(config)
            return config
        except FileNotFoundError:
            logging.warning(f"Config file not found: {full_path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            logging.error(f"YAML parse error: {e}")
            return {}

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${VAR} with environment variables"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(v) for v in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)
        return obj

    def _detect_kaggle(self) -> bool:
        """Detect if running in Kaggle environment"""
        kaggle_indicators = [
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_URL_BASE",
            "KAGGLE_WORKING_DIR"
        ]
        return any(os.getenv(ind) is not None for ind in kaggle_indicators)

    def _get_project_root(self) -> Path:
        """Get project root directory"""
        current = Path(__file__).resolve()
        # Navigate up from src/core/base_agent.py to project root
        return current.parent.parent.parent

    def _setup_output_dir(self) -> Path:
        """Setup output directory with Kaggle awareness"""
        if self.kaggle_mode:
            output = Path("/kaggle/working") / self.agent_key
        else:
            output = self._get_project_root() / "output" / self.agent_key
        output.mkdir(parents=True, exist_ok=True)
        return output

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.agent_id}")
        logger.setLevel(getattr(logging, self.config.get("system", {}).get("log_level", "INFO")))
        
        if not logger.handlers:
            # Console handler
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            console_fmt = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console.setFormatter(console_fmt)
            logger.addHandler(console)
            
            # File handler
            log_file = self.log_dir / f"{self.agent_key}_{self.agent_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)-8s | "
                "corr_id=%(corr_id)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)
        
        # Add correlation ID to extra
        logger = logging.LoggerAdapter(logger, {"corr_id": self.correlation_id})
        return logger

    def log_metric(self, name: str, value: Union[int, float, str]) -> None:
        """Log a metric for monitoring"""
        self.state.metrics[name] = value
        self.logger.info(f"METRIC | {name}={value}")

    def log_artifact(self, path: str) -> None:
        """Log an artifact file"""
        self.state.metrics.setdefault("artifacts", []).append(path)
        self.logger.info(f"ARTIFACT | {path}")

    def get_kaggle_path(self, path_type: str = "working") -> Path:
        """Get Kaggle-specific path or fallback to local"""
        if self.kaggle_mode:
            base = Path("/kaggle") / path_type
        else:
            base = self.project_root / "output"
        return base

    def save_artifact(self, data: Any, filename: str, subdir: str = "") -> str:
        """Save artifact with proper path handling"""
        save_dir = self.output_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename
        
        # Auto-detect format
        if filename.endswith(".json"):
            import json
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif filename.endswith(".pkl") or filename.endswith(".joblib"):
            import joblib
            joblib.dump(data, filepath)
        elif filename.endswith(".parquet"):
            if hasattr(data, "write_parquet"):
                data.write_parquet(filepath)
            elif hasattr(data, "to_parquet"):
                data.to_parquet(filepath, index=False)
        elif filename.endswith(".csv"):
            if hasattr(data, "write_csv"):
                data.write_csv(filepath)
            elif hasattr(data, "to_csv"):
                data.to_csv(filepath, index=False)
        else:
            with open(filepath, "w") as f:
                f.write(str(data))
        
        self.log_artifact(str(filepath))
        return str(filepath)

    def execute(self) -> AgentResult:
        """Execute agent workflow with timing and error handling"""
        start_time = time.time()
        self.state.status = "running"
        self.state.start_time = datetime.utcnow()
        
        self.logger.info(f"=== {self.__class__.__name__} START ===")
        
        try:
            # Pre-execution hook
            self._before_execute()
            
            # Main execution
            result_data = self._execute_core()
            
            # Post-execution hook
            self._after_execute(result_data)
            
            self.state.status = "completed"
            duration = time.time() - start_time
            
            self.logger.info(f"=== {self.__class__.__name__} COMPLETED ({duration:.2f}s) ===")
            
            return AgentResult(
                success=True,
                agent_name=self.__class__.__name__,
                duration_seconds=duration,
                data=result_data,
                artifacts=self.state.metrics.get("artifacts", [])
            )
            
        except Exception as e:
            self.state.status = "error"
            self.state.last_error = str(e)
            duration = time.time() - start_time
            
            self.logger.error(f"=== {self.__class__.__name__} FAILED ({duration:.2f}s) ===")
            self.logger.error(f"Error: {e}", exc_info=True)
            
            return AgentResult(
                success=False,
                agent_name=self.__class__.__name__,
                duration_seconds=duration,
                errors=[str(e)],
                artifacts=self.state.metrics.get("artifacts", [])
            )
        finally:
            self.state.end_time = datetime.utcnow()
            self._execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "status": self.state.status,
                "metrics": self.state.metrics.copy()
            })

    def _before_execute(self) -> None:
        """Hook called before execution - override in subclass"""
        pass

    def _after_execute(self, result: Dict[str, Any]) -> None:
        """Hook called after execution - override in subclass"""
        pass

    @abstractmethod
    def _execute_core(self) -> Dict[str, Any]:
        """
        Core agent logic - MUST be implemented by each agent.
        Returns a dictionary of results.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, status={self.state.status})"
