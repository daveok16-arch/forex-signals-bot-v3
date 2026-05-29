"""
Kaggle Environment Manager
Handles Kaggle-specific operations: GPU detection, dataset mounting,
notebook integration, and output management.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class KaggleManager:
    """
    Manages Kaggle Notebook environment integration.
    Provides GPU/TPU detection, dataset handling, and Kaggle API operations.
    """

    def __init__(self):
        self.is_kaggle = self._detect_kaggle()
        self.gpu_info = None
        self.tpu_info = None
        self._paths = None
        if self.is_kaggle:
            logger.info("Kaggle environment detected - initializing manager")
            self._setup_environment()
        else:
            logger.info("Not in Kaggle environment - using local mode")

    def _detect_kaggle(self) -> bool:
        """Detect if running inside Kaggle notebook"""
        indicators = [
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_URL_BASE",
            "KAGGLE_WORKING_DIR"
        ]
        return any(os.environ.get(ind) is not None for ind in indicators)

    def _setup_environment(self) -> None:
        """Configure Kaggle-specific environment settings"""
        # Set working paths
        os.environ.setdefault("PYTHONPATH", "/kaggle/working")
        
        # Optimize for training
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Detect hardware
        self.gpu_info = self._detect_gpu()
        self.tpu_info = self._detect_tpu()
        
        logger.info(f"GPU: {self.gpu_info}")
        logger.info(f"TPU: {self.tpu_info}")

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect available GPU(s) in Kaggle"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpus = []
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "id": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / 1e9,
                        "multi_processor_count": props.multi_processor_count
                    })
                logger.info(f"Detected {gpu_count} GPU(s): {[g['name'] for g in gpus]}")
                return {"available": True, "count": gpu_count, "devices": gpus}
        except ImportError:
            pass
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                for i, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(",")]
                    gpus.append({"id": i, "name": parts[0], "memory": parts[1] if len(parts) > 1 else "unknown"})
                return {"available": True, "count": len(gpus), "devices": gpus}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return {"available": False, "count": 0, "devices": []}

    def _detect_tpu(self) -> Dict[str, Any]:
        """Detect TPU availability"""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            dev = xm.xla_device()
            return {"available": True, "device": str(dev)}
        except ImportError:
            pass
        return {"available": False, "device": None}

    @property
    def paths(self) -> Dict[str, Path]:
        """Get Kaggle paths"""
        if self._paths is None:
            if self.is_kaggle:
                self._paths = {
                    "working": Path("/kaggle/working"),
                    "input": Path("/kaggle/input"),
                    "temp": Path("/kaggle/temp"),
                    "lib": Path("/kaggle/lib"),
                }
            else:
                root = Path(__file__).parent.parent.parent
                self._paths = {
                    "working": root / "output",
                    "input": root / "data",
                    "temp": root / "tmp",
                    "lib": root / "lib",
                }
        return self._paths

    def get_best_device(self) -> str:
        """Return the best available device string for PyTorch"""
        if self.tpu_info and self.tpu_info.get("available"):
            return "tpu"
        if self.gpu_info and self.gpu_info.get("available"):
            return "cuda"
        return "cpu"

    def get_device_count(self) -> int:
        """Get number of compute devices"""
        if self.tpu_info and self.tpu_info.get("available"):
            return 8  # TPU v3-8
        if self.gpu_info:
            return self.gpu_info.get("count", 0)
        return 0

    def mount_dataset(self, dataset_slug: str, path_alias: Optional[str] = None) -> Path:
        """
        Mount a Kaggle dataset.
        
        Args:
            dataset_slug: Kaggle dataset path (e.g., 'yakashri/export-15-fx-currency-pairs-1m')
            path_alias: Local alias for the mounted dataset
        
        Returns:
            Path to mounted dataset
        """
        target = path_alias or dataset_slug.split("/")[-1]
        
        if self.is_kaggle:
            # In Kaggle, datasets are auto-mounted at /kaggle/input
            mounted = self.paths["input"] / target
            if mounted.exists():
                logger.info(f"Dataset already mounted: {mounted}")
                return mounted
            
            # Try to download via Kaggle API
            try:
                import subprocess
                result = subprocess.run(
                    ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(mounted), "--unzip"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    logger.info(f"Dataset downloaded and mounted: {mounted}")
                    return mounted
                else:
                    logger.error(f"Failed to mount dataset: {result.stderr}")
            except Exception as e:
                logger.error(f"Error mounting dataset: {e}")
        
        # Local fallback
        local_path = self.paths["input"] / target
        local_path.mkdir(parents=True, exist_ok=True)
        return local_path

    def commit_notebook(self, message: str = "Auto-commit") -> bool:
        """Commit notebook version via Kaggle API"""
        if not self.is_kaggle:
            logger.info("Not in Kaggle - skipping notebook commit")
            return False
        
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", "/kaggle/working"],
                capture_output=True, text=True, timeout=60
            )
            success = result.returncode == 0
            if success:
                logger.info(f"Notebook committed: {message}")
            else:
                logger.error(f"Commit failed: {result.stderr}")
            return success
        except Exception as e:
            logger.error(f"Commit error: {e}")
            return False

    def save_output(self, source: str, destination: str = "") -> str:
        """Save file to Kaggle output directory"""
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        
        dst = self.paths["working"] / destination / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        
        logger.info(f"Saved output: {dst}")
        return str(dst)

    def create_submission(self, file_path: str, message: str = "Submission") -> bool:
        """Create a competition submission"""
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "submit", "-f", file_path, "-m", message],
                capture_output=True, text=True, timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Submission failed: {e}")
            return False

    def memory_optimization(self) -> None:
        """Optimize memory for Kaggle's constraints"""
        import gc
        gc.collect()
        
        if self.gpu_info and self.gpu_info.get("available"):
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except ImportError:
                pass
        
        logger.info("Memory optimization completed")

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        import psutil
        
        memory = psutil.virtual_memory()
        resources = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_used_gb": (memory.total - memory.available) / 1e9,
            "memory_available_gb": memory.available / 1e9,
            "memory_percent": memory.percent,
        }
        
        if self.gpu_info and self.gpu_info.get("available"):
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    resources[f"gpu_{i}_memory_allocated_gb"] = allocated
                    resources[f"gpu_{i}_memory_reserved_gb"] = reserved
            except ImportError:
                pass
        
        return resources

    def __repr__(self) -> str:
        return (
            f"KaggleManager(kaggle={self.is_kaggle}, "
            f"gpu={self.gpu_info.get('available', False)}, "
            f"tpu={self.tpu_info.get('available', False)})"
        )
