@echo off
:: ============================================================================
:: Quick launcher — requires Python + dependencies already installed
:: Double-click this file to open the Trading Suite launcher window
:: ============================================================================
cd /d "%~dp0"
python launcher.py
if errorlevel 1 (
    echo.
    echo  Something went wrong. Installing dependencies first...
    pip install -r requirements.txt
    python launcher.py
)
