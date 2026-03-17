# Indian Stock Trading Strategy

A Python-based toolkit for NSE/BSE mid-cap and small-cap stock trading, consisting of three modules that work together:

1. **Stock Screener** — scores all ~2,100 NSE/BSE stocks using technical indicators and saves the top 50 as a shared watchlist
2. **Simulation** — backtests the strategy on real historical data using that watchlist
3. **Live Signals** — scans the same stocks in real time and suggests buy/sell signals

---

## Project Structure

```
Trading/
├── config.py                   # All strategy parameters (edit this to tune the strategy)
├── trading_strategy.txt        # Human-readable strategy documentation
├── watchlist.py                # Shared watchlist bridge (screener → simulation & live signals)
│
├── stock_screener/
│   ├── run.py                  # Entry point — score all NSE/BSE stocks
│   ├── universe.py             # Fetches live stock list from NSE/BSE
│   ├── indicators.py           # Technical indicator calculations
│   ├── scorer.py               # 100-point scoring engine
│   └── results/                # Output: scores.csv, scores.xlsx, charts, watchlist.json
│
├── simulation/
│   ├── run.py                  # Entry point — run backtest
│   ├── backtest.py             # Walk-forward backtesting engine
│   ├── report.py               # Results report, charts, trade log
│   └── results/                # Output: trades.csv, equity_curve.png, etc.
│
└── live_signals/
    ├── run.py                  # Entry point — scan for signals / manage positions
    ├── scanner.py              # Real-time signal detection
    ├── display.py              # Terminal display (Rich tables)
    └── portfolio.py            # Position tracker (my_positions.json)
```

---

## Requirements

- Python 3.11 or higher
- Internet connection (downloads data from NSE and Yahoo Finance)

---

## Installation

### 1. Clone or download the project

```bash
cd ~/Desktop
# If using git:
git clone <your-repo-url> Trading
cd Trading

# Or just navigate into the folder if you already have it:
cd Trading
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

**Mac / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -c "import yfinance, pandas, numpy, matplotlib, rich, openpyxl; print('All packages installed')"
```

---

## Recommended First Run Order

Run the three modules in this order. The screener feeds the simulation and live signals.

```
[1] Stock Screener  →  [2] Simulation  →  [3] Live Signals
```

---

## Module 1 — Stock Screener

Scores all NSE/BSE mid-cap and small-cap stocks and saves the top 50 to a shared watchlist.

### Run the screener

```bash
# Score all mid-cap stocks (~150 stocks, takes ~10 seconds)
python stock_screener/run.py --cap midcap

# Score all small-cap stocks (~350 stocks, takes ~20 seconds)
python stock_screener/run.py --cap smallcap

# Score both mid and small-cap stocks (~500 stocks)
python stock_screener/run.py --cap midcap --cap smallcap

# Score ALL NSE/BSE stocks (~2100 stocks, takes ~3 minutes)
python stock_screener/run.py
```

### Other options

```bash
# Show only the top 20 results in terminal
python stock_screener/run.py --cap midcap --top 20

# Use more parallel workers (faster, more network load)
python stock_screener/run.py --cap midcap --workers 30

# Force refresh the stock universe list (re-download from NSE)
python stock_screener/run.py --cap midcap --refresh-universe

# Skip local cache and re-download all price data
python stock_screener/run.py --cap midcap --no-cache
```

### Outputs saved to `stock_screener/results/`

| File | Description |
|------|-------------|
| `scores.csv` | Full score breakdown for every stock |
| `scores.xlsx` | Excel file with conditional formatting |
| `top20_chart.png` | Bar chart of top 20 stocks by score |
| `heatmap.png` | Score heatmap across all categories |
| `top_watchlist.json` | Top 50 stocks — used by simulation and live signals |

> **Tip:** Re-run the screener weekly to keep the watchlist fresh. If the file is older than 7 days, the other modules will warn you.

---

## Module 2 — Simulation (Backtest)

Backtests the strategy on real historical data using the screener's top 50 stocks.

### Run a backtest

```bash
# Default: 3-year backtest on screener's top 50 stocks
python simulation/run.py

# Custom time period
python simulation/run.py --years 1                      # 1 year
python simulation/run.py --years 2 --months 6           # 2 years 6 months
python simulation/run.py --months 9                     # 9 months only
python simulation/run.py --years 1 --months 3 --weeks 2 # combined

# Custom starting capital
python simulation/run.py --capital 1000000              # ₹10 lakh

# Test specific stocks instead of the screener watchlist
python simulation/run.py --symbols RELIANCE TCS INFY

# Force re-download all price data (ignore cache)
python simulation/run.py --no-cache
```

### Outputs saved to `simulation/results/`

| File | Description |
|------|-------------|
| `trades.csv` | Every trade: entry/exit dates, P&L, reason |
| `metrics_summary.txt` | Full performance metrics |
| `equity_curve.png` | Portfolio value over time |
| `monthly_returns.png` | Month-by-month return bar chart |
| `trade_distribution.png` | Win/loss distribution histogram |

---

## Module 3 — Live Signals

Scans the screener's top 50 stocks in real time and suggests buy/sell signals based on the same strategy rules as the simulation.

### Scan once and see signals

```bash
python live_signals/run.py
```

### Watch mode — auto-refreshes every 15 minutes during market hours

```bash
python live_signals/run.py --watch
```

### Scan specific stocks only

```bash
python live_signals/run.py --symbols RELIANCE ICICIBANK LUPIN
```

### Manage your positions

```bash
# Log a trade you've taken
# Format: --add SYMBOL QUANTITY ENTRY_PRICE STOP_LOSS TARGET ATR
python live_signals/run.py --add ICICIBANK 38 1300 1241 1450 19.8

# View all open positions and check for exit signals
python live_signals/run.py --positions

# Remove a closed position from tracking
python live_signals/run.py --remove ICICIBANK
```

### Signal types displayed

| Signal | Meaning |
|--------|---------|
| **BUY** (green) | All entry conditions met — Setup A or B triggered |
| **ON RADAR** (yellow) | Weekly filter passes, waiting for price/volume trigger |
| **EXIT** (red) | A tracked position has hit a stop, target, or MA exit |
| **POSITIONS** (blue) | Your currently tracked open positions with live P&L |

---

## Configuration

All strategy parameters are in `config.py`. The key ones:

```python
# Entry
SETUP_A_LOW_DAYS      = 10      # N-day closing low for pullback entry
SETUP_B_DROP_THRESHOLD= 0.07    # 7% drop in 3 days triggers Setup B
PROXIMITY_THRESHOLD   = 0.15    # Price within 15% of weekly 20 MA
VOLUME_CONSECUTIVE_DAYS = 2     # 2 days of top-20% volume required
MIN_REWARD_RISK       = 1.5     # Minimum 1.5:1 R:R before entering

# Exit / Stop
ATR_STOP_MULTIPLIER   = 2       # Hard stop: 2 ATR below entry
ATR_TRAIL_MULTIPLIER  = 1.5     # Trailing stop: 1.5 ATR below price
MIN_HOLD_DAYS         = 10      # Days before MA/trail-stop exit activates

# Risk management
MAX_POSITIONS         = 5       # Max 5 open positions
MAX_POSITION_PCT      = 0.15    # 15% of capital per position
MAX_CAPITAL_DEPLOYED  = 0.75    # Never deploy more than 75%
TOTAL_CAPITAL         = 500_000 # Your starting capital in INR
```

---

## Typical Weekly Workflow

```bash
# Monday morning — refresh the watchlist
python stock_screener/run.py --cap midcap

# Check what signals are live today
python live_signals/run.py

# Log a trade after placing it on Zerodha
python live_signals/run.py --add LUPIN 40 2290 2135 2600 51.5

# End of day — check positions
python live_signals/run.py --positions

# Whenever — re-run backtest with new parameters
python simulation/run.py
```

---

## Troubleshooting

**`ModuleNotFoundError`** — Make sure your virtual environment is activated:
```bash
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

**`No data loaded`** — Check your internet connection. NSE data is fetched live on first run and cached locally for subsequent runs.

**`too little history, skipping`** — The stock was recently listed and doesn't have enough price history for the weekly 200 MA calculation (needs ~4.5 years of data for warmup).

**Screener watchlist is stale** — Re-run `python stock_screener/run.py --cap midcap` to refresh it. The live signals and simulation modules will warn you if the file is older than 7 days.

**VSCode shows red import underlines** — Select the `.venv` interpreter:
`Cmd+Shift+P` → `Python: Select Interpreter` → choose `.venv`
