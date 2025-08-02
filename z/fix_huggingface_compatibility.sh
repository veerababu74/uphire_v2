#!/bin/bash
# Quick Fix Script for HuggingFace Hub Compatibility Issue
# Run this on your remote server to fix the import error

echo "🔧 Fixing HuggingFace Hub compatibility issue..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📁 Activating virtual environment..."
    source venv/bin/activate
fi

echo "📦 Uninstalling conflicting packages..."
pip uninstall -y transformers huggingface_hub sentence-transformers

echo "📥 Installing compatible versions..."
pip install transformers==4.42.4
pip install "huggingface_hub>=0.25.0,<1.0.0"
pip install sentence-transformers==2.2.2

echo "✅ Compatibility fix completed!"
echo "🚀 You can now run: python main.py"
