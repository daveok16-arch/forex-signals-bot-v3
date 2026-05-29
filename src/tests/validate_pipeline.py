"""
Pipeline Validation Test
========================
Quick validation that all agents execute correctly with minimal data.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_agent_a():
    """Test Agent A - Data Ingestion"""
    from agents.agent_a_data import DataIngestionEngine
    agent = DataIngestionEngine(kaggle_mode=False)
    result = agent.execute()
    assert result.success, f"Agent A failed: {result.errors}"
    assert result.data.get("features_selected", 0) > 0, "No features generated"
    print(f"  PASS | Features: {result.data['features_selected']}")
    return True


def test_agent_b():
    """Test Agent B - Model Training"""
    from agents.agent_b_training import ModelTrainingEngine
    agent = ModelTrainingEngine(kaggle_mode=False)
    result = agent.execute()
    assert result.success, f"Agent B failed: {result.errors}"
    print(f"  PASS | Models: {result.data.get('total_models', 0)}")
    return True


def test_agent_c():
    """Test Agent C - Signal Generation"""
    from agents.agent_c_signals import SignalGenerationEngine
    agent = SignalGenerationEngine(kaggle_mode=False)
    result = agent.execute()
    assert result.success, f"Agent C failed: {result.errors}"
    print(f"  PASS | Signals: {result.data.get('signals_generated', 0)}")
    return True


def test_agent_d():
    """Test Agent D - Risk Management"""
    from agents.agent_d_risk import RiskManagementEngine
    agent = RiskManagementEngine(kaggle_mode=False)
    result = agent.execute()
    assert result.success, f"Agent D failed: {result.errors}"
    print(f"  PASS | Orders: {len(result.data.get('orders_approved', []))}")
    return True


def test_agent_e():
    """Test Agent E - Backtesting"""
    from agents.agent_e_backtest import BacktestingEngine
    agent = BacktestingEngine(kaggle_mode=False)
    result = agent.execute()
    assert result.success, f"Agent E failed: {result.errors}"
    print(f"  PASS | Pairs: {len(result.data.get('pairs_backtested', []))}")
    return True


def test_agent_f():
    """Test Agent F - Trade Execution"""
    from agents.agent_f_execution import TradeExecutionEngine
    agent = TradeExecutionEngine(kaggle_mode=False)
    result = agent.execute()
    assert result.success, f"Agent F failed: {result.errors}"
    print(f"  PASS | Executed: {result.data.get('orders_executed', 0)}")
    return True


def test_agent_g():
    """Test Agent G - Monitoring"""
    from agents.agent_g_monitoring import MonitoringCloudSyncEngine
    agent = MonitoringCloudSyncEngine(kaggle_mode=False)
    result = agent.execute()
    assert result.success, f"Agent G failed: {result.errors}"
    print(f"  PASS | Health: {result.data.get('health_status', 'N/A')}")
    return True


def run_validation():
    """Run all validation tests"""
    print("=" * 60)
    print("  FOREX ML BOT - PIPELINE VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Agent A: Data Ingestion", test_agent_a),
        ("Agent B: Model Training", test_agent_b),
        ("Agent C: Signal Generation", test_agent_c),
        ("Agent D: Risk Management", test_agent_d),
        ("Agent E: Backtesting", test_agent_e),
        ("Agent F: Trade Execution", test_agent_f),
        ("Agent G: Monitoring", test_agent_g),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAIL | {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
