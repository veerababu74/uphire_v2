# Comprehensive dependency fix script
# Run this script to resolve dependency conflicts

echo 'Starting dependency fix process...'

# Step 1: Backup current environment
pip freeze > requirements_backup.txt
echo 'Current environment backed up to requirements_backup.txt'

# Step 2: Uninstall problematic packages
echo 'Removing potentially conflicting packages...'
pip uninstall -y transformers huggingface_hub tokenizers safetensors sentence-transformers
pip uninstall -y torch torchvision torchaudio
pip uninstall -y langchain langchain-community langchain-core langchain-openai langchain-groq

# Step 3: Clear pip cache
pip cache purge

# Step 4: Install core dependencies in correct order
echo 'Installing core dependencies...'
pip install --upgrade pip setuptools wheel

# Install PyTorch first (CPU version for better compatibility)
pip install 'torch>=2.1.0,<3.0.0' --index-url https://download.pytorch.org/whl/cpu

# Install HuggingFace ecosystem
pip install 'huggingface_hub>=0.25.0,<1.0.0'
pip install 'tokenizers>=0.19.0,<1.0.0'
pip install 'safetensors>=0.4.0,<1.0.0'
pip install 'transformers>=4.42.0,<4.52.0'
pip install 'sentence-transformers>=2.2.2,<5.0.0'

# Install FastAPI ecosystem
pip install 'fastapi>=0.100.0,<0.120.0'
pip install 'uvicorn>=0.23.0,<0.36.0'
pip install 'pydantic>=2.6.0,<3.0.0'

# Install remaining requirements
pip install -r requirements.txt

# Step 5: Verify installation
echo 'Verifying installation...'
python -c "import transformers, sentence_transformers, fastapi, torch; print('All core packages imported successfully!')"

echo 'Dependency fix completed!'