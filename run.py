#!/usr/bin/env python3
"""
Forex ML Bot - Main Entry Point
================================
Runs the full multi-agent pipeline for AI/ML forex signal generation.

Usage:
    python run.py --mode full                    # Run complete pipeline
    python run.py --mode training                # Run training only (A, B, E)
    python run.py --mode inference               # Run inference only (A, C, D, F)
    python run.py --mode backtest                # Run backtesting only
    python run.py --mode single --agent agent_a   # Run single agent
    python run.py --kaggle                       # Enable Kaggle mode
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Forex ML Bot - AI/ML Signal Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Modes:
  full       - Run all agents in sequence (A->B->C->D->E->F->G)
  training   - Run data prep, model training, and backtesting (A->B->E->G)
  inference  - Run data prep, signal generation, risk, execution (A->C->D->F->G)
  backtest   - Run backtesting analysis (A->B->E)
  single     - Run a single specified agent

Agents:
  agent_a    - Data Ingestion & Feature Engineering
  agent_b    - Model Training & Hyperparameter Optimization
  agent_c    - Signal Generation & Ensemble Prediction
  agent_d    - Risk Management & Position Sizing
  agent_e    - Backtesting & Performance Analytics
  agent_f    - Trade Execution & Broker Integration
  agent_g    - Monitoring, Logging & Cloud Sync
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "training", "inference", "backtest", "single"],
        default="full",
        help="Pipeline execution mode (default: full)"
    )
    
    parser.add_argument(
        "--agent",
        choices=["agent_a", "agent_b", "agent_c", "agent_d", "agent_e", "agent_f", "agent_g"],
        help="Single agent to run (required with --mode single)"
    )
    
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Enable Kaggle environment mode"
    )
    
    parser.add_argument(
        "--config",
        default="config/system_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if an agent fails"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Forex ML Bot 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "single" and not args.agent:
        parser.error("--agent is required when using --mode single")
    
    # Setup paths
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "src"))
    
    # Import orchestrator
    from core.pipeline_orchestrator import PipelineOrchestrator
    
    print("=" * 60)
    print("  FOREX ML BOT v1.0.0")
    print("  Multi-Agent AI/ML Signal Generation System")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config_path=args.config,
        kaggle_mode=args.kaggle
    )
    
    # Run pipeline based on mode
    if args.mode == "full":
        print("\n[MODE] Full Pipeline Execution")
        result = orchestrator.run_pipeline(skip_failed=args.continue_on_error)
        
    elif args.mode == "training":
        print("\n[MODE] Training Pipeline")
        result = orchestrator.run_training_pipeline()
        
    elif args.mode == "inference":
        print("\n[MODE] Inference Pipeline")
        result = orchestrator.run_inference_pipeline()
        
    elif args.mode == "backtest":
        print("\n[MODE] Backtest Only")
        result = orchestrator.run_backtest_only()
        
    elif args.mode == "single":
        print(f"\n[MODE] Single Agent: {args.agent}")
        agent_result = orchestrator.run_single_agent(args.agent)
        
        print(f"\n{'=' * 60}")
        print(f"  Agent: {agent_result.agent_name}")
        print(f"  Success: {agent_result.success}")
        print(f"  Duration: {agent_result.duration_seconds:.2f}s")
        print(f"{'=' * 60}")
        
        sys.exit(0 if agent_result.success else 1)
    
    # Print summary
    if isinstance(result, dict):
        print(f"\n{'=' * 60}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Completed: {len(result.get('completed_agents', []))} agents")
        print(f"  Failed: {len(result.get('failed_agents', []))} agents")
        print(f"  Duration: {result.get('total_duration', 0):.2f}s")
        
        # Per-agent results
        print(f"\n  Agent Results:")
        for agent, data in result.get("results", {}).items():
            status = "PASS" if data.get("success") else "FAIL"
            duration = data.get("duration", 0)
            print(f"    [{status}] {agent:<10} ({duration:.1f}s)")
        
        print(f"{'=' * 60}")
        
        # Exit code
        success = result.get("status") == "completed"
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
