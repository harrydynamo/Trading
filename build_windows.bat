@echo off
:: ============================================================================
:: Trading Suite — Windows .exe Builder
:: Run this file on a Windows machine to produce TradingSuite.exe
:: ============================================================================

echo.
echo  ============================================================
echo   Trading Suite — Windows Build Script
echo  ============================================================
echo.

:: ── Step 1: Check Python ─────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo  [OK] Python found

:: ── Step 2: Install / upgrade dependencies ───────────────────────────────────
echo.
echo  Installing dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install pyinstaller --upgrade --quiet
echo  [OK] Dependencies installed

:: ── Step 3: Clean old build ───────────────────────────────────────────────────
echo.
echo  Cleaning previous build...
if exist dist\TradingSuite rmdir /s /q dist\TradingSuite
if exist build rmdir /s /q build
echo  [OK] Clean done

:: ── Step 4: Build .exe ────────────────────────────────────────────────────────
echo.
echo  Building TradingSuite.exe  (this takes 2-5 minutes)...
pyinstaller trading_suite.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo  [ERROR] Build failed. See above for details.
    pause
    exit /b 1
)

:: ── Step 5: Done ─────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   BUILD SUCCESSFUL
echo   Location: dist\TradingSuite\TradingSuite.exe
echo  ============================================================
echo.
echo  Copy the entire dist\TradingSuite\ folder to any Windows PC.
echo  Double-click TradingSuite.exe to launch the app menu.
echo.
pause
