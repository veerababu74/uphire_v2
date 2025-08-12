@echo off
echo ================================
echo   EMBEDDING ISSUES FIX SCRIPT
echo ================================
echo.
echo This script will fix SentenceTransformer meta tensor loading issues
echo.

REM Change to script directory
cd /d "%~dp0"

echo [1/3] Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

echo.
echo [2/3] Running embedding fix script...
python fix_embedding_issues.py
if errorlevel 1 (
    echo ERROR: Fix script failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo [3/3] Running integration tests...
python test_embedding_integration.py
if errorlevel 1 (
    echo WARNING: Some integration tests failed. Check the output above.
    echo However, the basic fix may still work.
    pause
    exit /b 1
)

echo.
echo ================================
echo   FIX COMPLETED SUCCESSFULLY!
echo ================================
echo.
echo The embedding issues have been resolved.
echo You can now run your resume parsing application.
echo.
pause
