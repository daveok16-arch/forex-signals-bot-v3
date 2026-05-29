# AI/ML Forex Signals Bot v1.0.0

A production-grade, multi-agent AI/ML system for forex market signal generation, built with advanced ensemble modeling, risk management, and automated cloud infrastructure.

## System Architecture

```
Agent A (Data) -> Agent B (Training) -> Agent C (Signals) -> Agent D (Risk) -> Agent F (Execution)
                                      \
                                       -> Agent E (Backtest)
                                        
Agent G (Monitoring & Cloud Sync) oversees all agents
```

### The 7 Agents

| Agent | Name | Purpose |
|-------|------|---------|
| A | Data Ingestion Engine | Multi-source data collection, 150+ technical indicators, feature selection |
| B | Model Training Engine | 6 model types with Optuna HPO, GPU/TPU support, ensemble preparation |
| C | Signal Generation Engine | Dynamic ensemble weighting, uncertainty quantification, signal decay |
| D | Risk Management Engine | Kelly Criterion, VaR, drawdown controls, dynamic position sizing |
| E | Backtesting Engine | Walk-forward optimization, Monte Carlo, Deflated Sharpe Ratio |
| F | Trade Execution Engine | OANDA integration, paper trading, TWAP/VWAP execution |
| G | Monitoring Cloud Sync | Health checks, AWS/GCP/Azure backup, GitHub sync, alerting |

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd forex-ml-bot
bash scripts/setup.sh

# 2. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Run full pipeline
python run.py --mode full

# Or run specific pipeline modes
python run.py --mode training      # Train models only
python run.py --mode inference     # Generate signals and trade
python run.py --mode backtest      # Backtest strategies
python run.py --mode single --agent agent_a  # Run single agent

# 4. Run with Kaggle GPU
python run.py --mode full --kaggle
```

## Project Structure

```
forex-ml-bot/
├── config/
│   └── system_config.yaml          # Master configuration
├── docs/
│   └── ARCHITECTURE.md             # Detailed architecture docs
├── src/
│   ├── core/
│   │   ├── base_agent.py           # Base class for all agents
│   │   ├── kaggle_manager.py       # Kaggle environment manager
│   │   ├── cloud_sync.py           # Cloud backup engine
│   │   └── pipeline_orchestrator.py # Pipeline runner
│   ├── agents/
│   │   ├── agent_a_data.py         # Data & feature engineering
│   │   ├── agent_b_training.py     # Model training & HPO
│   │   ├── agent_c_signals.py      # Signal generation
│   │   ├── agent_d_risk.py         # Risk management
│   │   ├── agent_e_backtest.py     # Backtesting
│   │   ├── agent_f_execution.py    # Trade execution
│   │   └── agent_g_monitoring.py   # Monitoring & cloud sync
│   └── tests/
│       └── validate_pipeline.py    # Integration tests
├── notebooks/
│   └── kaggle/
│       ├── agent_a_data_prep.ipynb
│       └── agent_b_model_train.ipynb
├── .github/
│   └── workflows/
│       └── ci-pipeline.yml         # CI/CD pipeline
├── scripts/
│   └── setup.sh                    # Environment setup
├── requirements.txt
├── run.py                          # Main entry point
└── README.md
```

## Configuration

All configuration is in `config/system_config.yaml`:

### Currency Pairs (8 pairs)
- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, XAUUSD

### Timeframes (5 timeframes)
- M5, M15, H1, H4, D1

### Models (6 types)
- XGBoost, LightGBM, CatBoost, Random Forest, LSTM, TabNet, Transformer

### Risk Parameters
- Max risk per trade: 2%
- Max total risk: 6%
- Kelly fraction: 0.5x
- Max drawdown: 10%

## Kaggle Integration

### Features
- Auto-detects Kaggle environment
- GPU/TPU detection and utilization
- Dataset mounting from Kaggle Datasets
- Mixed precision training
- Memory optimization
- Output saving to `/kaggle/working`

### Running on Kaggle
```python
# In Kaggle notebook
from src.core.pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(kaggle_mode=True)
result = orchestrator.run_training_pipeline()  # or run_pipeline()
```

## Cloud Backup

Supports AWS S3, Google Cloud Storage, and Azure Blob Storage.

```yaml
# config/system_config.yaml
cloud_backup:
  provider: "aws"           # aws | gcp | azure
  s3_bucket: "your-bucket"
  backup_interval_hours: 6
  compression: "gzip"
  encryption: true
```

## GitHub CI/CD

The `.github/workflows/ci-pipeline.yml` provides:
- **Lint & Test**: Code quality checks and unit tests
- **Agent A**: Data ingestion with artifact upload
- **Agent B**: Model training with GPU support
- **Agent E**: Backtesting and performance analysis
- **Deploy**: Automatic model deployment to cloud storage
- **Notify**: Slack notifications for pipeline status

## Environment Variables

```bash
# Required
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account

# Cloud Backup (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your-bucket

# GitHub
GITHUB_REPO=username/repo

# Alerts
SLACK_WEBHOOK=https://hooks.slack.com/...
```

## Key Features

### Agent A - Data Ingestion
- 150+ technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, etc.)
- Statistical features (skewness, kurtosis, z-score)
- MRMR feature selection
- Data quality scoring
- Multi-source data (OANDA, Kaggle, TrueFX)

### Agent B - Model Training
- Optuna hyperparameter optimization (TPE sampler, Hyperband pruner)
- 6 model architectures
- Purged cross-validation
- GPU/TPU acceleration
- Ensemble stacking/blending

### Agent C - Signal Generation
- Dynamic ensemble weighting
- Conformal prediction intervals
- Signal decay and expiration
- Cross-pair correlation analysis
- Multi-timeframe signal correlation

### Agent D - Risk Management
- Half-Kelly Criterion position sizing
- VaR and Conditional VaR
- ATR-based stop-loss
- Regime-based risk scaling
- Drawdown circuit breakers

### Agent E - Backtesting
- Walk-forward optimization
- Monte Carlo permutation tests
- Deflated Sharpe Ratio
- Probability of Overfitting (PBO)
- Regime-dependent attribution

### Agent F - Trade Execution
- Paper and live trading modes
- OANDA REST API integration
- Smart order routing
- Latency monitoring
- TWAP/VWAP execution

### Agent G - Monitoring
- System health checks
- Structured logging with correlation IDs
- Multi-cloud backup (AWS/GCP/Azure)
- GitHub repository sync
- Slack/email alerts

## Performance

The system is designed for production with:
- **Scalability**: Handles 8 currency pairs across 5 timeframes
- **Speed**: GPU-accelerated training, polars for fast data processing
- **Reliability**: Retry logic, circuit breakers, health monitoring
- **Auditability**: Full trade history, model versioning, cloud backups

## Testing

```bash
# Validate all agents
python -m src.tests.validate_pipeline

# Run single agent test
python -m src.agents.agent_a_data

# Run full pipeline test
python run.py --mode full --continue-on-error
```

## License

MIT License - See LICENSE file

## Disclaimer

This software is for educational and research purposes. Trading forex involves significant risk of loss. Past performance does not guarantee future results. Always test thoroughly in paper trading before using real money.
