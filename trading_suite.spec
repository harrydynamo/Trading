# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Trading Suite (Windows .exe)
#
# Build with (on Windows):
#   pip install pyinstaller
#   pyinstaller trading_suite.spec
#
# Output: dist/TradingSuite/TradingSuite.exe

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Collect all data files needed by each dependency ─────────────────────────
datas = []
datas += collect_data_files("streamlit")
datas += collect_data_files("streamlit_extras",    ignore_package_data_errors=True)
datas += collect_data_files("plotly")
datas += collect_data_files("yfinance")
datas += collect_data_files("pandas")
datas += collect_data_files("altair",              ignore_package_data_errors=True)

# Include the entire project folder
datas += [
    ("trading_ui",    "trading_ui"),
    ("stock_screener","stock_screener"),
    ("live_signals",  "live_signals"),
    ("ui",            "ui"),
    ("simulation",    "simulation"),
    ("config.py",     "."),
]

# ── Hidden imports (modules Streamlit discovers at runtime) ───────────────────
hiddenimports = (
    collect_submodules("streamlit")
    + collect_submodules("streamlit.runtime")
    + collect_submodules("streamlit.web")
    + collect_submodules("streamlit.components")
    + collect_submodules("plotly")
    + collect_submodules("yfinance")
    + collect_submodules("pandas")
    + collect_submodules("numpy")
    + collect_submodules("requests")
    + [
        "pkg_resources.py2_warn",
        "charset_normalizer",
        "rich",
        "tkinter",
        "tkinter.ttk",
        "altair",
        "pydeck",
        "trading_ui.indicators",
        "trading_ui.signals",
        "trading_ui.charts",
        "trading_ui.support_resistance",
        "stock_screener.universe",
        "stock_screener.scorer",
        "live_signals.scanner",
        "live_signals.display",
        "live_signals.portfolio",
        "simulation.backtest",
    ]
)

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pytest", "IPython", "notebook"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TradingSuite",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no black terminal window (GUI only)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # replace with "trading_ui/icon.ico" if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TradingSuite",
)
