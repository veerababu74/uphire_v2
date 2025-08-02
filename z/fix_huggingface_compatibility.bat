@echo off
REM Quick Fix Script for HuggingFace Hub Compatibility Issue
REM Run this to fix the import error

echo 🔧 Fixing HuggingFace Hub compatibility issue...

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 📁 Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo 📦 Uninstalling conflicting packages...
pip uninstall -y transformers huggingface_hub sentence-transformers

echo 📥 Installing compatible versions...
pip install transformers==4.42.4
pip install "huggingface_hub>=0.25.0,<1.0.0"
pip install sentence-transformers==2.2.2

echo ✅ Compatibility fix completed!
echo 🚀 You can now run: python main.py
pause
