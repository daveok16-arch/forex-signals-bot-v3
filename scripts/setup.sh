#!/bin/bash
# Forex ML Bot - Environment Setup Script
# Usage: bash scripts/setup.sh [--kaggle]

set -e

echo "=========================================="
echo "  Forex ML Bot - Environment Setup"
echo "=========================================="

# Parse arguments
KAGGLE_MODE=false
for arg in "$@"; do
    case $arg in
        --kaggle)
            KAGGLE_MODE=true
            shift
            ;;
    esac
done

# Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "Python: $PYTHON_VERSION"

# Create virtual environment if not in Kaggle
if [ "$KAGGLE_MODE" = false ]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv || python -m venv venv
    fi
    
    echo "Activating virtual environment..."
    source venv/bin/activate || . venv/Scripts/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p output/{agent_a,agent_b,agent_c,agent_d,agent_e,agent_f,agent_g,pipeline}
mkdir -p notebooks/kaggle
mkdir -p docs

# Copy Kaggle notebooks if in Kaggle mode
if [ "$KAGGLE_MODE" = true ]; then
    echo "Setting up Kaggle environment..."
    mkdir -p /kaggle/working/output
    cp -r config /kaggle/working/
    cp -r src /kaggle/working/
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Set environment variables in .env file"
echo "  2. Run pipeline: python -m src.core.pipeline_orchestrator --mode full"
echo "  3. Or run single agent: python -m src.agents.agent_a_data"
echo ""