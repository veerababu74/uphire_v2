@echo off
REM Quick fix for UPHire v2 dependency issues on Windows
REM This script resolves common package conflicts

echo Starting dependency fix for UPHire v2...

REM Step 1: Backup current environment
echo Creating backup of current environment...
pip freeze > requirements_backup.txt

REM Step 2: Uninstall problematic packages
echo Removing potentially conflicting packages...
pip uninstall -y transformers huggingface_hub tokenizers safetensors sentence-transformers --quiet
pip uninstall -y torch torchvision torchaudio --quiet

REM Step 3: Clear pip cache
echo Clearing pip cache...
pip cache purge

REM Step 4: Install core dependencies in correct order
echo Installing core dependencies...
pip install --upgrade pip setuptools wheel

REM Install PyTorch (CPU version for better compatibility)
echo Installing PyTorch...
pip install "torch>=2.1.0,<3.0.0" --index-url https://download.pytorch.org/whl/cpu

REM Install HuggingFace ecosystem in correct order
echo Installing HuggingFace packages...
pip install "huggingface_hub>=0.25.0,<1.0.0"
pip install "tokenizers>=0.19.0,<1.0.0"
pip install "safetensors>=0.4.0,<1.0.0"
pip install "transformers>=4.42.0,<4.52.0"
pip install "sentence-transformers>=2.2.2,<5.0.0"

REM Install FastAPI ecosystem
echo Installing FastAPI packages...
pip install "fastapi>=0.100.0,<0.120.0"
pip install "uvicorn>=0.23.0,<0.36.0"
pip install "pydantic>=2.6.0,<3.0.0"

REM Install remaining requirements
echo Installing remaining dependencies...
pip install -r requirements.txt

REM Step 5: Verify installation
echo Verifying installation...
python -c "import transformers, sentence_transformers, fastapi, torch; print('✅ All core packages imported successfully!')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Some packages failed to import. Check the error messages above.
    exit /b 1
) else (
    echo ✅ Dependency fix completed successfully!
)

echo.
echo 🎉 Your environment is now ready!
echo You can now run: python main.py
pause
