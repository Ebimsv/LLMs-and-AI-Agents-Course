#!/bin/bash

# LLMs and Agents in Production - Setup Script
# This script sets up the development environment for the tutorial series

set -e  # Exit on any error

echo "🚀 Setting up LLMs and Agents in Production environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python version $PYTHON_VERSION is too old. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv llms_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source llms_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔐 Creating .env file..."
    cat > .env << EOF
# API Keys (Add your keys here)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
EOF
    echo "📝 Please edit .env file with your API keys"
fi

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p day-1-understanding-llms/exercises
mkdir -p day-2-llm-apis
mkdir -p day-3-ollama-local-llms/streamlit_apps
mkdir -p day-4-model-selection/benchmarks
mkdir -p day-5-prompt-engineering/techniques
mkdir -p day-5-prompt-engineering/evaluation_tools
mkdir -p day-6-website-summarizer/templates
mkdir -p day-6-website-summarizer/static
mkdir -p shared/examples
mkdir -p docs
mkdir -p assets/diagrams
mkdir -p assets/screenshots

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed. You can install it from https://ollama.ai/"
    echo "   For now, you can continue with cloud-based LLMs (OpenAI, Anthropic)"
else
    echo "✅ Ollama is installed"
fi

echo ""
echo "🎉 Setup complete! Your environment is ready."
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source llms_env/bin/activate"
echo "2. Edit .env file with your API keys"
echo "3. Start with Day 1: cd day-1-understanding-llms/"
echo "4. Run Jupyter: jupyter notebook"
echo ""
echo "📚 Happy learning! 🚀" 