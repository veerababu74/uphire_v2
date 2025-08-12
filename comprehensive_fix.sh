#!/bin/bash
# Comprehensive HuggingFace Hub Compatibility Fix
# This script fixes the specific 'list_repo_tree' import error

echo "🔧 Starting HuggingFace Hub compatibility fix..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment not detected. Activating..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ No virtual environment found. Please create one first."
        exit 1
    fi
fi

echo "📋 Current package versions:"
pip show transformers huggingface_hub sentence-transformers 2>/dev/null || echo "Some packages not installed"

echo ""
echo "🧹 Step 1: Cleaning up incompatible packages..."
pip uninstall -y transformers huggingface_hub sentence-transformers tokenizers safetensors --quiet

echo "🔄 Step 2: Installing compatible versions in correct order..."

# Install core dependencies first
echo "  📦 Installing tokenizers..."
pip install "tokenizers>=0.19.0,<1.0.0" --quiet

echo "  📦 Installing safetensors..."
pip install "safetensors>=0.4.0" --quiet

echo "  📦 Installing huggingface_hub..."
pip install "huggingface_hub>=0.25.0,<1.0.0" --quiet

echo "  📦 Installing transformers..."
pip install "transformers>=4.42.0,<4.46.0" --quiet

echo "  📦 Installing sentence-transformers..."
pip install "sentence-transformers>=2.2.2,<3.0.0" --quiet

echo ""
echo "🔍 Step 3: Verifying installation..."
python -c "
try:
    import transformers
    import huggingface_hub
    import sentence_transformers
    print('✅ All packages imported successfully!')
    print(f'   transformers: {transformers.__version__}')
    print(f'   huggingface_hub: {huggingface_hub.__version__}')
    print(f'   sentence_transformers: {sentence_transformers.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Fix completed successfully!"
    echo "🚀 You can now run: python main.py"
else
    echo ""
    echo "❌ Fix failed. Please check the error above."
    exit 1
fi
