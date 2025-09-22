#!/bin/bash
# Script to activate the Python virtual environment for jhamon-training project

echo "Activating Python virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "To deactivate the virtual environment, run: deactivate"
echo "To run Python scripts, use: python script_name.py"
echo ""
