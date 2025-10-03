#!/bin/bash

# AI Agents Project Setup Script
# This script helps set up the development environment with Poetry

set -e

echo "🚀 Setting up AI Agents project with Poetry..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    echo "✅ Poetry installed successfully!"
else
    echo "✅ Poetry is already installed"
fi

# Install dependencies
echo "📚 Installing project dependencies..."
poetry install --with dev

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: poetry shell"
echo "2. Run tests: poetry run python backend/run_tests.py --type all"
echo "3. Start development server: poetry run uvicorn backend.app.main:app --reload"
echo ""
echo "For more commands, see the README.md file."