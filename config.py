"""
Configuration for Indian Stock Market Trading Strategy
Broker: Zerodha Kite Connect
"""

# ─── KITE CONNECT CREDENTIALS ───────────────────────────────────────────────
# Get these from https://developers.kite.trade/
KITE_API_KEY = "your_api_key_here"
KITE_API_SECRET = "your_api_secret_here"
# Access token is generated daily — run `python main.py --login` each morning
ACCESS_TOKEN_FILE = "access_token.txt"

# ─── STRATEGY PARAMETERS ─────────────────────────────────────────────────────
WEEKLY_TREND_MA = 200         # Price must be above this on weekly chart
WEEKLY_PROXIMITY_MA = 20      # Price must be near this on weekly chart
DAILY_EXIT_MA = 20            # Exit when daily close drops below this

ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2       # Hard stop: 2 ATR below entry (was 3 — losses too large)
ATR_TRAIL_MULTIPLIER = 1.5    # Trailing stop: 1.5 ATR below current price (was 2)

# Volume conditions
VOLUME_LOOKBACK_WEEKS = 12    # Rank volume relative to last 12 weeks
VOLUME_PERCENTILE = 80        # Must be in top 20% (80th percentile)
VOLUME_CONSECUTIVE_DAYS = 2   # Elevated volume for 2 consecutive days (was 3 — too rare)
VOLUME_PEAK_DAYS = 10         # Volume must NOT exceed 10-day peak (avoid exhaustion)

# Setup A — N-day low pullback
SETUP_A_LOW_DAYS = 10           # 10-day closing low (wider pullback window)

# Setup B — 3-day drop + hammer
SETUP_B_DROP_DAYS = 3
SETUP_B_DROP_THRESHOLD = 0.07   # 7% drop required (was 10% — never fired)

# Weekly 20 MA proximity: price within this % of the 20 MA to qualify
PROXIMITY_THRESHOLD = 0.15      # within 15% of weekly 20 MA (was 5% — too tight)

# ─── RISK MANAGEMENT ─────────────────────────────────────────────────────────
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.15         # 15% of total capital per position
MAX_CAPITAL_DEPLOYED = 0.75     # Never deploy more than 75%
MIN_CASH_BUFFER = 0.25          # Always keep 25% cash
MIN_REWARD_RISK = 1.5           # Minimum 1.5:1 R:R before entering (was 2.0)
MIN_HOLD_DAYS   = 10            # Days held before MA/trail-stop exit can fire (was 5)

TOTAL_CAPITAL = 500_000         # INR — update to your actual capital

# ─── STOCK UNIVERSE (NSE) ────────────────────────────────────────────────────
# Mix of Nifty 100 (large-cap) + Nifty Midcap 150 — yfinance uses ".NS" suffix
WATCHLIST = [
    # Large-cap (Nifty 50)
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "BAJFINANCE", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID",
    "NTPC", "TECHM", "HCLTECH", "ONGC", "COALINDIA",
    "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT", "ADANIPORTS",
    "BAJAJFINSV", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
    "HEROMOTOCO", "HINDALCO", "ITC", "M&M", "INDUSINDBK",
    "BRITANNIA", "CIPLA", "HDFCLIFE", "SBILIFE", "BPCL",
    # Mid-cap (more volatile — better for both setups)
    "PERSISTENT", "MPHASIS", "COFORGE", "LTIM", "TATAELXSI",
    "ASTRAL", "PIIND", "DEEPAKNTR", "AARTIIND", "APLLTD",
    "ESCORTS", "VOLTAS", "CROMPTON", "HAVELLS", "POLYCAB",
    "PAGEIND", "METROPOLIS", "LALPATHLAB", "MAXHEALTH", "FORTIS",
    "ABCAPITAL", "CHOLAFIN", "MANAPPURAM", "MUTHOOTFIN", "MFSL",
    "FEDERALBNK", "RBLBANK", "BANDHANBNK", "AUBANK", "IDFCFIRSTB",
    "KPRMILL", "APLAPOLLO", "RATNAMANI", "RAYMOND", "KAJARIACER",
    "RVNL", "IRFC", "PFC", "RECLTD", "HUDCO",
]

# ─── EXECUTION SETTINGS ──────────────────────────────────────────────────────
PAPER_TRADE = True              # Set to False for live trading
SCAN_INTERVAL_MINUTES = 15      # How often to scan for signals during market hours
LOG_FILE = "trading_log.txt"

# NSE market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
