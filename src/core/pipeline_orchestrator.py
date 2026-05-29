"""
Pipeline Orchestrator for Forex ML Bot
======================================
Manages the execution pipeline across all 7 agents.
Handles: sequential execution, error recovery, state management,
conditional logic, and reporting.

Execution Flow:
Agent A -> Agent B -> Agent C -> Agent D -> Agent E -> Agent F -> Agent G
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core.base_agent import AgentResult


class PipelineOrchestrator:
    """
    Orchestrates the full execution pipeline across all agents.
    
    Features:
    - Sequential agent execution with dependency management
    - Error recovery and rollback capabilities
    - Conditional execution (skip agents if previous failed)
    - Comprehensive pipeline reporting
    - Pipeline state persistence
    """

    def __init__(self, config_path: str = "config/system_config.yaml", kaggle_mode: bool = False):
        self.config_path = config_path
        self.kaggle_mode = kaggle_mode
        self.project_root = Path(__file__).parent.parent
        
        # Pipeline state
        self._pipeline_state: Dict[str, Any] = {
            "status": "idle",
            "current_agent": None,
            "completed_agents": [],
            "failed_agents": [],
            "skipped_agents": [],
            "results": {},
            "start_time": None,
            "end_time": None,
            "errors": []
        }
        
        # Agent registry
        self._agents: Dict[str, Callable] = {}
        self._agent_dependencies: Dict[str, List[str]] = {}
        
        self._register_agents()

    def _register_agents(self) -> None:
        """Register all agents with their dependencies"""
        # Import agents
        from agents.agent_a_data import DataIngestionEngine
        from agents.agent_b_training import ModelTrainingEngine
        from agents.agent_c_signals import SignalGenerationEngine
        from agents.agent_d_risk import RiskManagementEngine
        from agents.agent_e_backtest import BacktestingEngine
        from agents.agent_f_execution import TradeExecutionEngine
        from agents.agent_g_monitoring import MonitoringCloudSyncEngine
        
        # Register with dependencies
        self._agents["agent_a"] = DataIngestionEngine
        self._agent_dependencies["agent_a"] = []
        
        self._agents["agent_b"] = ModelTrainingEngine
        self._agent_dependencies["agent_b"] = ["agent_a"]
        
        self._agents["agent_c"] = SignalGenerationEngine
        self._agent_dependencies["agent_c"] = ["agent_b"]
        
        self._agents["agent_d"] = RiskManagementEngine
        self._agent_dependencies["agent_d"] = ["agent_c"]
        
        self._agents["agent_e"] = BacktestingEngine
        self._agent_dependencies["agent_e"] = ["agent_b"]  # Can run after B
        
        self._agents["agent_f"] = TradeExecutionEngine
        self._agent_dependencies["agent_f"] = ["agent_d"]
        
        self._agents["agent_g"] = MonitoringCloudSyncEngine
        self._agent_dependencies["agent_g"] = []  # Always runs

    def _can_run(self, agent_name: str) -> bool:
        """Check if agent can run based on dependencies"""
        dependencies = self._agent_dependencies.get(agent_name, [])
        
        for dep in dependencies:
            if dep in self._pipeline_state["failed_agents"]:
                return False
            if dep not in self._pipeline_state["completed_agents"]:
                return False
        
        return True

    def _run_agent(self, agent_name: str) -> AgentResult:
        """Execute a single agent"""
        agent_class = self._agents.get(agent_name)
        if not agent_class:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        print(f"\n{'='*60}")
        print(f"  EXECUTING: {agent_name.upper()}")
        print(f"{'='*60}")
        
        self._pipeline_state["current_agent"] = agent_name
        self._save_state()
        
        try:
            # Instantiate and execute
            agent = agent_class(
                config_path=self.config_path,
                kaggle_mode=self.kaggle_mode
            )
            
            result = agent.execute()
            
            print(f"\n  {agent_name.upper()} COMPLETED")
            print(f"  Success: {result.success} | Duration: {result.duration_seconds:.2f}s")
            
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for err in result.errors[:3]:
                    print(f"    - {err}")
            
            return result
            
        except Exception as e:
            print(f"\n  {agent_name.upper()} FAILED: {e}")
            traceback.print_exc()
            
            return AgentResult(
                success=False,
                agent_name=agent_name,
                duration_seconds=0,
                errors=[str(e), traceback.format_exc()]
            )

    def run_pipeline(self, agents: Optional[List[str]] = None,
                     skip_failed: bool = True) -> Dict[str, Any]:
        """
        Execute the full pipeline or a subset of agents.
        
        Args:
            agents: List of agent names to run (None = all)
            skip_failed: Skip agents that fail vs stop pipeline
        
        Returns:
            Pipeline execution summary
        """
        agent_list = agents or list(self._agents.keys())
        
        self._pipeline_state["status"] = "running"
        self._pipeline_state["start_time"] = datetime.utcnow().isoformat()
        
        print(f"\n{'#'*60}")
        print(f"#  FOREX ML BOT - PIPELINE EXECUTION")
        print(f"#  Agents: {len(agent_list)}")
        print(f"#  Kaggle Mode: {self.kaggle_mode}")
        print(f"#  Start: {self._pipeline_state['start_time']}")
        print(f"{'#'*60}")
        
        for agent_name in agent_list:
            # Check dependencies
            if not self._can_run(agent_name):
                print(f"\n  SKIPPING {agent_name}: Dependencies not met")
                self._pipeline_state["skipped_agents"].append(agent_name)
                continue
            
            # Execute agent
            result = self._run_agent(agent_name)
            
            # Update state
            self._pipeline_state["results"][agent_name] = {
                "success": result.success,
                "duration": result.duration_seconds,
                "errors": result.errors,
                "data_summary": {k: str(v)[:100] for k, v in result.data.items()}
            }
            
            if result.success:
                self._pipeline_state["completed_agents"].append(agent_name)
            else:
                self._pipeline_state["failed_agents"].append(agent_name)
                if not skip_failed:
                    print(f"\n  PIPELINE STOPPED: {agent_name} failed")
                    break
            
            self._save_state()
        
        # Finalize
        self._pipeline_state["status"] = (
            "completed" if not self._pipeline_state["failed_agents"] 
            else "completed_with_errors"
        )
        self._pipeline_state["end_time"] = datetime.utcnow().isoformat()
        
        # Calculate total duration
        start = datetime.fromisoformat(self._pipeline_state["start_time"])
        end = datetime.fromisoformat(self._pipeline_state["end_time"])
        total_duration = (end - start).total_seconds()
        
        self._pipeline_state["total_duration"] = total_duration
        
        # Print summary
        print(f"\n{'#'*60}")
        print(f"#  PIPELINE COMPLETE")
        print(f"#  Duration: {total_duration:.2f}s")
        print(f"#  Completed: {len(self._pipeline_state['completed_agents'])}/{len(agent_list)}")
        print(f"#  Failed: {len(self._pipeline_state['failed_agents'])}")
        print(f"#  Skipped: {len(self._pipeline_state['skipped_agents'])}")
        print(f"{'#'*60}")
        
        self._save_state()
        self._save_summary()
        
        return self._pipeline_state

    def run_single_agent(self, agent_name: str) -> AgentResult:
        """Run a single agent"""
        if agent_name not in self._agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        return self._run_agent(agent_name)

    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run only training-related agents (A, B, E)"""
        return self.run_pipeline(["agent_a", "agent_b", "agent_e", "agent_g"])

    def run_inference_pipeline(self) -> Dict[str, Any]:
        """Run only inference-related agents (A, C, D, F, G)"""
        return self.run_pipeline(["agent_a", "agent_c", "agent_d", "agent_f", "agent_g"])

    def run_backtest_only(self) -> Dict[str, Any]:
        """Run only backtesting"""
        return self.run_pipeline(["agent_a", "agent_b", "agent_e"])

    def _save_state(self) -> None:
        """Save pipeline state to disk"""
        state_dir = self.project_root / "output" / "pipeline"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = state_dir / "pipeline_state.json"
        with open(state_file, "w") as f:
            json.dump(self._pipeline_state, f, indent=2, default=str)

    def _save_summary(self) -> None:
        """Save pipeline summary report"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_config": {
                "kaggle_mode": self.kaggle_mode,
                "config_path": self.config_path
            },
            "execution": {
                "status": self._pipeline_state["status"],
                "start_time": self._pipeline_state["start_time"],
                "end_time": self._pipeline_state["end_time"],
                "total_duration_seconds": self._pipeline_state.get("total_duration", 0)
            },
            "agents": {
                "completed": self._pipeline_state["completed_agents"],
                "failed": self._pipeline_state["failed_agents"],
                "skipped": self._pipeline_state["skipped_agents"]
            },
            "results_summary": {
                agent: {
                    "success": data["success"],
                    "duration": data["duration"]
                }
                for agent, data in self._pipeline_state["results"].items()
            }
        }
        
        summary_dir = self.project_root / "output" / "pipeline"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = summary_dir / f"pipeline_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also save latest
        latest = summary_dir / "pipeline_summary_latest.json"
        with open(latest, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        return self._pipeline_state

    def reset_state(self) -> None:
        """Reset pipeline state"""
        self._pipeline_state = {
            "status": "idle",
            "current_agent": None,
            "completed_agents": [],
            "failed_agents": [],
            "skipped_agents": [],
            "results": {},
            "start_time": None,
            "end_time": None,
            "errors": []
        }


def main():
    """Main entry point for pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Forex ML Bot Pipeline")
    parser.add_argument("--mode", choices=["full", "training", "inference", "backtest", "single"],
                       default="full", help="Pipeline execution mode")
    parser.add_argument("--agent", help="Single agent to run (with --mode single)")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle mode")
    parser.add_argument("--config", default="config/system_config.yaml", help="Config file path")
    parser.add_argument("--continue-on-error", action="store_true", 
                       help="Continue pipeline on agent failure")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config_path=args.config,
        kaggle_mode=args.kaggle
    )
    
    # Run pipeline
    if args.mode == "full":
        result = orchestrator.run_pipeline(skip_failed=args.continue_on_error)
    elif args.mode == "training":
        result = orchestrator.run_training_pipeline()
    elif args.mode == "inference":
        result = orchestrator.run_inference_pipeline()
    elif args.mode == "backtest":
        result = orchestrator.run_backtest_only()
    elif args.mode == "single":
        if not args.agent:
            print("Error: --agent required with --mode single")
            sys.exit(1)
        result = orchestrator.run_single_agent(args.agent)
    
    # Exit code based on success
    if isinstance(result, dict):
        success = result.get("status") == "completed"
    else:
        success = result.success
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
