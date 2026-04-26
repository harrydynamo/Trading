"""
Live Trading UI — NSE / BSE

Run with:
    streamlit run trading_ui/app.py
"""

import os, re, sys, time, warnings, logging, calendar, json, uuid
from datetime import date, timedelta
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from stock_screener.universe import get_universe
from trading_ui.indicators import compute_all
from trading_ui.signals import compute_signals
from trading_ui.support_resistance import pivot_points, swing_levels, fibonacci_levels
from trading_ui.charts import build_chart


# ─── Portfolio persistence ─────────────────────────────────────────────────────
_PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")

def _load_portfolio() -> list[dict]:
    try:
        if os.path.exists(_PORTFOLIO_FILE):
            with open(_PORTFOLIO_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_portfolio(trades: list[dict]):
    try:
        with open(_PORTFOLIO_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception:
        pass


# ─── Cached heavy-computation wrappers ────────────────────────────────────────
# These run once per (ticker, timeframe) combo and skip re-execution on every
# Streamlit widget interaction, making the UI responsive during searches.

@st.cache_data(ttl=300, show_spinner=False)
def _cached_compute_all(df: pd.DataFrame,
                        use_supertrend: bool = True,
                        use_donchian: bool = True) -> pd.DataFrame:
    return compute_all(df, use_supertrend=use_supertrend, use_donchian=use_donchian)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_compute_signals(df: pd.DataFrame,
                             lookback: int = 50,
                             use_candlestick: bool = True,
                             use_volume: bool = True) -> dict:
    return compute_signals(df, lookback=lookback,
                           use_candlestick=use_candlestick,
                           use_volume=use_volume)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_sr_levels(df: pd.DataFrame,
                      do_pivot: bool, do_swing: bool, do_fib: bool) -> list:
    levels: list = []
    if do_pivot: levels += pivot_points(df)
    if do_swing: levels += swing_levels(df)
    if do_fib:   levels += fibonacci_levels(df)
    return levels


@st.cache_data(ttl=300, show_spinner=False)
def _cached_build_chart(df: pd.DataFrame,
                        indicators_config: dict,
                        sr_levels: list,
                        signals: list,
                        timeframe: str = "1D") -> object:
    return build_chart(df, indicators_config, sr_levels, signals, timeframe=timeframe)


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Live Trading — NSE/BSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Hide Streamlit chrome, white page background ── */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
/* Make header transparent and remove its space, but keep the sidebar toggle button */
header     { background: transparent !important; }
[data-testid="stToolbar"]    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
/* Always show the sidebar open/close arrow button */
header button,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button {
    visibility: visible !important;
    display:    flex   !important;
}
.stApp    { background-color: #ffffff; }
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* ── Stock header card ── */
.header-card {
    background: #ffffff;
    border: 1.5px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.header-symbol  { font-size: 2rem; font-weight: 700; color: #111; margin: 0; }
.header-name    { font-size: 0.85rem; color: #888; margin-top: 2px; }
.header-price   { font-size: 2rem; font-weight: 700; color: #111; margin: 0; }
.header-up      { color: #1a7a3c; font-size: 1rem; font-weight: 600; }
.header-down    { color: #c0392b; font-size: 1rem; font-weight: 600; }
.header-neutral { color: #888;    font-size: 1rem; }

/* ── Stat pills ── */
.stats-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
.stat-pill {
    background: #f8f9fa;
    border: 1.5px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px 16px;
    flex: 1; min-width: 110px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stat-pill .label { font-size: 0.72rem; color: #999; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 3px; }
.stat-pill .value { font-size: 0.95rem; font-weight: 600; color: #111; }

/* ── Bias panel ── */
.bias-panel {
    border-radius: 10px;
    padding: 16px 22px;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.bias-bullish { background: #f0faf3; border: 1.5px solid #1a7a3c; }
.bias-bearish { background: #fff5f5; border: 1.5px solid #c0392b; }
.bias-neutral { background: #f8f9fa; border: 1.5px solid #ccc; }
.bias-label   { font-size: 1.5rem; font-weight: 700; }
.bias-bull-txt { color: #1a7a3c; }
.bias-bear-txt { color: #c0392b; }
.bias-neut-txt { color: #666; }
.bias-score   { font-size: 0.85rem; color: #888; margin-top: 2px; }

/* ── Signal badges ── */
.signal-badge {
    display: inline-block;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 4px;
}
.badge-buy   { background: #e8f5ec; color: #1a7a3c; border: 1px solid #1a7a3c; }
.badge-sell  { background: #fdecea; color: #c0392b; border: 1px solid #c0392b; }
.badge-watch { background: #fff8e1; color: #b8860b; border: 1px solid #b8860b; }

/* ── Trade setup cards ── */
.setup-card {
    background: #fff;
    border: 1.5px solid #e0e0e0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.setup-buy  { border-top: 3px solid #1a7a3c; }
.setup-sell { border-top: 3px solid #c0392b; }
.setup-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.setup-dir-buy  { font-size:1rem; font-weight:700; color:#1a7a3c; }
.setup-dir-sell { font-size:1rem; font-weight:700; color:#c0392b; }
.setup-date { font-size:0.75rem; color:#999; }
.setup-grid {
    display: grid;
    grid-template-columns: 56px 1fr 1fr;
    gap: 5px 10px;
    margin-bottom: 10px;
    align-items: baseline;
}
.setup-lbl   { font-size:0.68rem; color:#aaa; text-transform:uppercase; letter-spacing:.05em; padding-top:2px; }
.setup-val   { font-size:0.92rem; font-weight:600; color:#111; }
.setup-chg   { font-size:0.78rem; }
.setup-green { color:#1a7a3c; }
.setup-red   { color:#c0392b; }
.setup-rr    { font-size:0.78rem; background:#f3f4f6; border-radius:5px;
               padding:3px 8px; display:inline-block; color:#555; margin-bottom:8px; }
.setup-tags  { display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }
.setup-tag   { background:#f3f4f6; border-radius:4px; padding:2px 7px;
               font-size:0.70rem; color:#666; }

/* ── SR level rows ── */
.sr-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 12px;
    border-radius: 6px;
    margin-bottom: 5px;
    background: #fff;
    border: 1px solid #e8e8e8;
    font-size: 0.85rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.sr-label { color: #999; font-size: 0.75rem; }
.sr-price { font-weight: 600; color: #111; }
.sr-dist  { font-size: 0.78rem; }
.sr-res   { border-left: 3px solid #c0392b; }
.sr-sup   { border-left: 3px solid #1a7a3c; }
.sr-piv   { border-left: 3px solid #7c3aed; }

/* ── How-to guide ── */
.guide-box {
    background: #f8f9fa;
    border: 1.5px solid #e0e0e0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 6px;
    font-size: 0.82rem;
    color: #333;
    line-height: 1.7;
}
.guide-step {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 8px;
}
.guide-num {
    background: #1a5aad;
    color: #fff;
    border-radius: 50%;
    width: 20px; height: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}
.guide-num.sell-num { background: #c0392b; }
.guide-text  { flex: 1; }
.guide-price { font-weight: 700; color: #111; }
.guide-green { color: #1a7a3c; font-weight: 600; }
.guide-red   { color: #c0392b; font-weight: 600; }
.guide-rule  {
    background: #fff3cd;
    border-left: 3px solid #f0ad00;
    border-radius: 0 6px 6px 0;
    padding: 7px 12px;
    margin-top: 10px;
    font-size: 0.78rem;
    color: #555;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #999;
    margin: 16px 0 10px 0;
}

/* ── Divider ── */
.thin-divider { border: none; border-top: 1px solid #e8e8e8; margin: 16px 0; }

/* ── Chart wrapper with border ── */
.chart-wrapper {
    border: 1.5px solid #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)


# ─── Instrument universes ─────────────────────────────────────────────────────

INDICES: dict[str, dict] = {
    "NIFTY 50":        {"symbol": "NIFTY50",   "name": "Nifty 50 Index",          "yf_ticker": "^NSEI",      "exchange": "NSE", "cap": "index"},
    "BANK NIFTY":      {"symbol": "BANKNIFTY",  "name": "Bank Nifty Index",        "yf_ticker": "^NSEBANK",   "exchange": "NSE", "cap": "index"},
    "SENSEX":          {"symbol": "SENSEX",     "name": "BSE Sensex",              "yf_ticker": "^BSESN",     "exchange": "BSE", "cap": "index"},
    "NIFTY MIDCAP 50": {"symbol": "NIFTYMID50", "name": "Nifty Midcap 50",         "yf_ticker": "^CNXMIDCAP", "exchange": "NSE", "cap": "index"},
    "NIFTY IT":        {"symbol": "NIFTYIT",    "name": "Nifty IT Index",          "yf_ticker": "^CNXIT",     "exchange": "NSE", "cap": "index"},
    "INDIA VIX":       {"symbol": "INDIAVIX",   "name": "India VIX (Volatility)",  "yf_ticker": "^INDIAVIX",  "exchange": "NSE", "cap": "index"},
}

CURRENCIES: dict[str, dict] = {
    "USD/INR": {"symbol": "USDINR", "name": "US Dollar / Indian Rupee",  "yf_ticker": "USDINR=X", "exchange": "FOREX", "cap": "fx"},
    "EUR/USD": {"symbol": "EURUSD", "name": "Euro / US Dollar",          "yf_ticker": "EURUSD=X", "exchange": "FOREX", "cap": "fx"},
    "GBP/USD": {"symbol": "GBPUSD", "name": "British Pound / US Dollar", "yf_ticker": "GBPUSD=X", "exchange": "FOREX", "cap": "fx"},
    "USD/JPY": {"symbol": "USDJPY", "name": "US Dollar / Japanese Yen",  "yf_ticker": "USDJPY=X", "exchange": "FOREX", "cap": "fx"},
    "EUR/INR": {"symbol": "EURINR", "name": "Euro / Indian Rupee",       "yf_ticker": "EURINR=X", "exchange": "FOREX", "cap": "fx"},
    "GBP/INR": {"symbol": "GBPINR", "name": "British Pound / INR",       "yf_ticker": "GBPINR=X", "exchange": "FOREX", "cap": "fx"},
}


def _detect_itype(yf_ticker: str) -> str:
    t = yf_ticker.upper()
    if t.startswith("^"):                            return "index"
    if t.endswith("=X"):                             return "currency"
    if re.search(r"\d{2}[A-Z]{3}FUT", t):           return "futures"
    if re.search(r"\d{2}[A-Z]{3}\d+[CP]E", t):      return "options"
    return "stock"


_ITYPE_BADGE = {
    "index":    ("#1a5aad", "#e8f0fd", "INDEX"),
    "currency": ("#7c3aed", "#f3e8fd", "FX"),
    "futures":  ("#d97706", "#fef3c7", "FUTURES"),
    "options":  ("#db2777", "#fce7f3", "OPTIONS"),
    "stock":    ("#1a7a3c", "#f0faf3", "STOCK"),
}

def _itype_badge(itype: str) -> str:
    color, bg, label = _ITYPE_BADGE.get(itype, ("#888", "#f0f0f0", "—"))
    return (f'<span style="background:{bg};color:{color};border:1px solid {color};'
            f'border-radius:5px;padding:2px 9px;font-size:0.72rem;font-weight:700;">'
            f'{label}</span>')


# ─── NSE F&O helpers ──────────────────────────────────────────────────────────

_FNO_UNDERLYINGS = {
    "NIFTY":      {"step": 50,  "spot_ticker": "^NSEI"},
    "BANKNIFTY":  {"step": 100, "spot_ticker": "^NSEBANK"},
    "FINNIFTY":   {"step": 50,  "spot_ticker": "NIFTY_FIN_SERVICE.NS"},
    "MIDCPNIFTY": {"step": 25,  "spot_ticker": "^CNXMIDCAP"},
}


def _last_thursday(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:          # 3 = Thursday
        d -= timedelta(days=1)
    return d


def _nse_monthly_expiries(n: int = 3) -> list[tuple[str, str]]:
    """
    Returns list of (label, code) for the next n monthly expiry months.
    E.g. [("April 2025", "25APR"), ("May 2025", "25MAY"), ...]
    Automatically skips months whose last Thursday has already passed.
    """
    today = date.today()
    year, month = today.year, today.month
    if today > _last_thursday(year, month):
        month += 1
        if month > 12:
            month, year = 1, year + 1

    result = []
    for _ in range(n):
        label = f"{calendar.month_name[month]} {year}"
        code  = f"{str(year)[-2:]}{calendar.month_abbr[month].upper()}"
        result.append((label, code))
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return result


def _atm_strikes(spot: float, step: int, n_each_side: int = 10) -> list[int]:
    atm = round(spot / step) * step
    return [atm + i * step for i in range(-n_each_side, n_each_side + 1)]


# ─── Timeframe config ─────────────────────────────────────────────────────────

TIMEFRAMES = {
    "5m":  {"interval": "5m",  "period": "5d",  "max_bars": 390},
    "15m": {"interval": "15m", "period": "60d", "max_bars": 390},
    "1h":  {"interval": "60m", "period": "60d", "max_bars": 500},
    "1D":  {"interval": "1d",  "period": "2y",  "max_bars": 504},
    "1W":  {"interval": "1wk", "period": "10y", "max_bars": 520},
}


# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def scan_tradeable_stocks(include_microcap: bool = False) -> pd.DataFrame:
    """
    Scan stocks and score them for trading readiness (0–100).
    By default scans only NSE midcap + smallcap (~400 stocks) for speed.
    Set include_microcap=True to scan all ~2900 stocks (much slower).
    Cached 30 min.
    """
    universe = load_universe()

    # Filter to liquid stocks by default — midcap + smallcap only
    if not include_microcap:
        universe = {k: v for k, v in universe.items()
                    if v["cap"] in ("midcap", "smallcap") and v["exchange"] == "NSE"}

    tickers = [s["yf_ticker"] for s in universe.values()]
    meta    = {s["yf_ticker"]: s for s in universe.values()}

    if not tickers:
        return pd.DataFrame()

    # Download in batches of 100 to avoid yfinance timeouts
    BATCH = 100
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        try:
            chunk = yf.download(
                batch,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
            frames.append((batch, chunk))
        except Exception:
            # fall back to threads=False for this batch
            try:
                chunk = yf.download(
                    batch,
                    period="6mo",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,
                )
                frames.append((batch, chunk))
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    # Merge all batches into one lookup
    def _extract(raw, ticker):
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return None
            return raw[ticker].copy()
        return raw.copy()

    rows = []
    for batch_tickers, raw in frames:
        for ticker in batch_tickers:
            try:
                df = _extract(raw, ticker)
                if df is None:
                    continue

                df.dropna(how="all", inplace=True)
                if len(df) < 30:
                    continue

                # ── Compute indicators inline (fast, no full compute_all) ────
                close  = df["Close"]
                high   = df["High"]
                low    = df["Low"]

                ema9   = close.ewm(span=9,   adjust=False).mean()
                ema21  = close.ewm(span=21,  adjust=False).mean()
                ema50  = close.ewm(span=50,  adjust=False).mean()
                ema200 = close.ewm(span=200, adjust=False).mean()

                # ATR
                prev_c = close.shift(1)
                tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
                atr_s  = tr.ewm(com=13, min_periods=14).mean()

                # RSI
                delta   = close.diff()
                avg_g   = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
                avg_l   = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
                rsi_s   = 100 - (100 / (1 + avg_g / avg_l.replace(0, np.nan)))

                # ADX
                up_move   = high - high.shift(1)
                dn_move   = low.shift(1) - low
                plus_dm   = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
                minus_dm  = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
                atr14     = tr.ewm(com=13, min_periods=14).mean()
                plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(com=13, min_periods=14).mean() / atr14.replace(0, np.nan)
                minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(com=13, min_periods=14).mean() / atr14.replace(0, np.nan)
                dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
                adx_s     = dx.ewm(com=13, min_periods=14).mean()

                # Supertrend
                hl2        = (high + low) / 2
                mult       = 3.0
                b_upper    = (hl2 + mult * atr_s).values
                b_lower    = (hl2 - mult * atr_s).values
                c_arr      = close.values
                n          = len(df)
                f_upper    = b_upper.copy()
                f_lower    = b_lower.copy()
                for i in range(1, n):
                    if np.isnan(atr_s.iloc[i]): continue
                    f_upper[i] = b_upper[i] if (np.isnan(f_upper[i-1]) or b_upper[i] < f_upper[i-1] or c_arr[i-1] > f_upper[i-1]) else f_upper[i-1]
                    f_lower[i] = b_lower[i] if (np.isnan(f_lower[i-1]) or b_lower[i] > f_lower[i-1] or c_arr[i-1] < f_lower[i-1]) else f_lower[i-1]
                st_dir = np.zeros(n, dtype=int)
                for i in range(1, n):
                    if np.isnan(f_upper[i]) or np.isnan(f_lower[i]): continue
                    pd_ = st_dir[i-1]
                    if pd_ == 0:   st_dir[i] = 1 if c_arr[i] >= (f_upper[i]+f_lower[i])/2 else -1
                    elif pd_ == 1: st_dir[i] = -1 if c_arr[i] < f_lower[i] else 1
                    else:          st_dir[i] =  1 if c_arr[i] > f_upper[i] else -1

                # ── Last bar values ──────────────────────────────────────────
                last_close  = float(close.iloc[-1])
                prev_close  = float(close.iloc[-2]) if len(df) >= 2 else last_close
                last_rsi    = float(rsi_s.iloc[-1])
                last_adx    = float(adx_s.iloc[-1])
                last_pdi    = float(plus_di.iloc[-1])
                last_mdi    = float(minus_di.iloc[-1])
                last_st     = int(st_dir[-1])
                last_e9     = float(ema9.iloc[-1])
                last_e21    = float(ema21.iloc[-1])
                last_e50    = float(ema50.iloc[-1])
                last_e200   = float(ema200.iloc[-1])
                avg_vol     = float(df["Volume"].iloc[-20:].mean())
                vol_today   = float(df["Volume"].iloc[-1])
                vol_ratio   = round(vol_today / avg_vol, 1) if avg_vol > 0 else 0
                chg_pct     = (last_close - prev_close) / prev_close * 100

                if any(np.isnan(v) for v in (last_rsi, last_adx, last_e21)):
                    continue

                # ── Determine direction ──────────────────────────────────────
                if last_e9 > last_e21 > last_e50 and last_close > last_e21 and last_st == 1 and last_pdi > last_mdi:
                    direction = "BUY"
                elif last_e9 < last_e21 < last_e50 and last_close < last_e21 and last_st == -1 and last_mdi > last_pdi:
                    direction = "SELL"
                else:
                    direction = "WATCH"

                # ── Tradability score 0–100 ──────────────────────────────────
                score = 0

                # 1. ADX — trending market
                if last_adx > 35:    score += 25
                elif last_adx > 25:  score += 18
                elif last_adx > 20:  score += 8

                # 2. Supertrend direction alignment
                if direction == "BUY"  and last_st ==  1: score += 20
                if direction == "SELL" and last_st == -1: score += 20
                if direction == "WATCH": score += 5

                # 3. EMA stack
                if last_e9 > last_e21 > last_e50 > last_e200: score += 20
                elif last_e9 < last_e21 < last_e50 < last_e200: score += 20
                elif last_e9 > last_e21 > last_e50: score += 12
                elif last_e9 < last_e21 < last_e50: score += 12

                # 4. Volume
                if vol_ratio >= 2.0:   score += 15
                elif vol_ratio >= 1.5: score += 10
                elif vol_ratio >= 1.0: score += 5

                # 5. RSI in sweet spot (not extreme)
                if 45 <= last_rsi <= 65:   score += 10
                elif 35 <= last_rsi <= 70: score += 5

                # 6. Price vs EMA21
                if direction == "BUY"  and last_close > last_e21: score += 10
                if direction == "SELL" and last_close < last_e21: score += 10

                # ── ATR-based SL/Target ──────────────────────────────────────
                atr_val = float(atr_s.iloc[-1])
                if direction == "BUY":
                    sl     = round(last_close - 1.5 * atr_val, 2)
                    target = round(last_close + 3.0 * atr_val, 2)
                elif direction == "SELL":
                    sl     = round(last_close + 1.5 * atr_val, 2)
                    target = round(last_close - 3.0 * atr_val, 2)
                else:
                    sl = target = None

                # ── Trading execution type ───────────────────────────────────
                # Classified by current market microstructure, not stock fundamentals.
                # Priority: most specific condition first.
                atr_pct   = atr_val / last_close * 100 if last_close > 0 else 0
                _full_bull = last_e9 > last_e21 > last_e50 > last_e200
                _full_bear = last_e9 < last_e21 < last_e50 < last_e200
                _ema_align = last_e9 > last_e21 > last_e50 or last_e9 < last_e21 < last_e50

                # Reversal: RSI at extremes — price likely to mean-revert
                if (direction == "BUY" and last_rsi < 36) or (direction == "SELL" and last_rsi > 74):
                    trade_type = "Reversal Trading"

                # Breakout: surge in volume + price above EMA9 + strong RSI
                elif vol_ratio >= 2.5 and last_rsi >= 58 and last_close > last_e9:
                    trade_type = "Breakout Trading"

                # Momentum: strong vol, overbought-ish RSI, powerful trend
                elif vol_ratio >= 1.8 and last_rsi >= 60 and last_adx > 28 and _ema_align:
                    trade_type = "Momentum Trading"

                # Scalping: ATR% very wide — risk moves too fast for multi-day holds
                elif atr_pct > 4.5:
                    trade_type = "Scalping"

                # Long-term Investing: full EMA stack, low volatility, mild trend
                elif (_full_bull or _full_bear) and atr_pct < 1.5 and last_adx < 22 and 45 <= last_rsi <= 60:
                    trade_type = "Long-term Investing"

                # Positional Trading: full EMA stack, strong trend, multi-week hold
                elif (_full_bull or _full_bear) and last_adx >= 22 and 42 <= last_rsi <= 68:
                    trade_type = "Positional Trading"

                # Swing Trading: partial EMA alignment, moderate trend and RSI
                elif _ema_align and 18 <= last_adx <= 42 and 38 <= last_rsi <= 67:
                    trade_type = "Swing Trading"

                # Intraday Trading: ADX too low for multi-day, or trend unclear
                elif last_adx < 18 or not _ema_align:
                    trade_type = "Intraday Trading"

                # Algorithmic Trading: very tight ATR, no clear trend edge — only algo benefits
                else:
                    trade_type = "Algorithmic Trading"

                s = meta[ticker]
                rows.append({
                    "Symbol":    s["symbol"],
                    "Name":      s["name"],
                    "Cap":       s["cap"].upper(),
                    "Type":      trade_type,
                    "Signal":    direction,
                    "Score":     score,
                    "Price (₹)": round(last_close, 2),
                    "Day Chg %": round(chg_pct, 2),
                    "ADX":       round(last_adx, 1),
                    "RSI":       round(last_rsi, 1),
                    "Vol Ratio": vol_ratio,
                    "Stop (₹)":  sl,
                    "Target (₹)":target,
                    "ATR":       round(atr_val, 2),
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out.sort_values(["Score", "ADX"], ascending=[False, False], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


# ─── Pullback Scanner ──────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def scan_pullback_stocks(include_microcap: bool = False) -> pd.DataFrame:
    """
    Finds stocks in confirmed uptrends that have pulled back into a buy zone.

    Entry rules (all must be true):
      1. Trend intact     — EMA9 > EMA21 > EMA50 AND Supertrend = +1
      2. Pulled back      — price dipped below EMA21 or EMA9 within last 5 bars,
                            OR current price is within 3% above EMA21
      3. Pullback depth   — between 3% and 25% from the 20-bar high
      4. RSI cooling off  — RSI between 35 and 62 (not overbought)
      5. ADX trend        — ADX ≥ 18 (some trend strength, not choppy)

    Stage column:
      "Near EMA9"  — price within 1 ATR above EMA9  (tightest, aggressive entry)
      "Near EMA21" — price within 1 ATR above EMA21 (standard)
      "Deep"       — pulled back >10% from 20d high but structure intact (higher risk)
    """
    universe = load_universe()
    if include_microcap:
        stocks = {k: v for k, v in universe.items() if v.get("exchange") in ("NSE", "BSE")}
    else:
        stocks = {k: v for k, v in universe.items()
                  if v.get("cap", "").lower() in ("largecap", "midcap", "smallcap")
                  and v.get("exchange") in ("NSE", "BSE")}

    tickers = list({v["yf_ticker"] for v in stocks.values()})
    meta    = {v["yf_ticker"]: v for v in stocks.values()}

    BATCH = 60
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            chunk = yf.download(batch, period="6mo", interval="1d",
                                progress=False, auto_adjust=True,
                                group_by="ticker", threads=False)
            frames.append((batch, chunk))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    def _extract(raw, ticker):
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return None
            return raw[ticker].copy()
        return raw.copy()

    rows = []
    for batch_tickers, raw in frames:
        for ticker in batch_tickers:
            try:
                df = _extract(raw, ticker)
                if df is None:
                    continue
                df.dropna(how="all", inplace=True)
                if len(df) < 50:
                    continue

                close  = df["Close"]
                high   = df["High"]
                low    = df["Low"]

                ema9   = close.ewm(span=9,   adjust=False).mean()
                ema21  = close.ewm(span=21,  adjust=False).mean()
                ema50  = close.ewm(span=50,  adjust=False).mean()
                ema200 = close.ewm(span=200, adjust=False).mean()

                prev_c = close.shift(1)
                tr     = pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
                atr_s  = tr.ewm(com=13, min_periods=14).mean()

                delta  = close.diff()
                avg_g  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
                avg_l  = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
                rsi_s  = 100 - (100 / (1 + avg_g / avg_l.replace(0, np.nan)))

                up_move  = high - high.shift(1)
                dn_move  = low.shift(1) - low
                plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
                atr14    = tr.ewm(com=13, min_periods=14).mean()
                plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(com=13, min_periods=14).mean() / atr14.replace(0, np.nan)
                minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(com=13, min_periods=14).mean() / atr14.replace(0, np.nan)
                dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
                adx_s    = dx.ewm(com=13, min_periods=14).mean()

                # Supertrend
                hl2     = (high + low) / 2
                b_upper = (hl2 + 3.0 * atr_s).values
                b_lower = (hl2 - 3.0 * atr_s).values
                c_arr   = close.values
                n       = len(df)
                f_upper = b_upper.copy(); f_lower = b_lower.copy()
                for i in range(1, n):
                    if np.isnan(atr_s.iloc[i]): continue
                    f_upper[i] = b_upper[i] if (np.isnan(f_upper[i-1]) or b_upper[i] < f_upper[i-1] or c_arr[i-1] > f_upper[i-1]) else f_upper[i-1]
                    f_lower[i] = b_lower[i] if (np.isnan(f_lower[i-1]) or b_lower[i] > f_lower[i-1] or c_arr[i-1] < f_lower[i-1]) else f_lower[i-1]
                st_dir = np.zeros(n, dtype=int)
                for i in range(1, n):
                    if np.isnan(f_upper[i]) or np.isnan(f_lower[i]): continue
                    pd_ = st_dir[i-1]
                    if pd_ == 0:   st_dir[i] = 1 if c_arr[i] >= (f_upper[i]+f_lower[i])/2 else -1
                    elif pd_ == 1: st_dir[i] = -1 if c_arr[i] < f_lower[i] else 1
                    else:          st_dir[i] =  1 if c_arr[i] > f_upper[i] else -1

                lc      = float(close.iloc[-1])
                e9      = float(ema9.iloc[-1])
                e21     = float(ema21.iloc[-1])
                e50     = float(ema50.iloc[-1])
                e200    = float(ema200.iloc[-1])
                rsi     = float(rsi_s.iloc[-1])
                adx     = float(adx_s.iloc[-1])
                atr_val = float(atr_s.iloc[-1])
                st      = int(st_dir[-1])
                prev_c_val = float(close.iloc[-2]) if n >= 2 else lc
                chg     = (lc - prev_c_val) / prev_c_val * 100 if prev_c_val else 0
                avg_vol = float(df["Volume"].iloc[-20:].mean())
                vol_ratio = round(float(df["Volume"].iloc[-1]) / avg_vol, 1) if avg_vol > 0 else 0

                if any(np.isnan(v) for v in (rsi, adx, e9, e21, e50, atr_val)):
                    continue

                # ── Rule 1: Uptrend intact ───────────────────────────────────
                uptrend = e9 > e21 > e50 and st == 1

                if not uptrend:
                    continue

                # ── Rule 2: RSI and ADX ──────────────────────────────────────
                if not (35 <= rsi <= 62):
                    continue
                if adx < 18:
                    continue

                # ── Rule 3: Pullback depth from 20-bar high ──────────────────
                high_20d     = float(high.iloc[-20:].max())
                pullback_pct = (high_20d - lc) / high_20d * 100 if high_20d > 0 else 0.0

                if not (3.0 <= pullback_pct <= 25.0):
                    continue

                # ── Rule 4: Price in buy zone — near EMA9 or EMA21 ──────────
                near_ema9  = lc <= e9  + 1.0 * atr_val and lc >= e9  - 0.5 * atr_val
                near_ema21 = lc <= e21 + 1.0 * atr_val and lc >= e21 - 0.5 * atr_val

                # Also accept: touched EMA21 within last 5 bars
                touched_ema21_recent = any(
                    float(close.iloc[j]) <= float(ema21.iloc[j]) * 1.005
                    for j in range(-5, 0) if abs(j) <= n
                )

                if not (near_ema9 or near_ema21 or touched_ema21_recent):
                    continue

                # ── Stage label ──────────────────────────────────────────────
                if near_ema9:
                    stage = "🟢 Near EMA9"
                elif near_ema21 or touched_ema21_recent:
                    stage = "🟡 Near EMA21"
                else:
                    stage = "🔵 Deep"

                # ── Pullback quality score ───────────────────────────────────
                pb_score = 0
                if e9 > e21 > e50 > e200:   pb_score += 30   # full bull stack
                elif e9 > e21 > e50:         pb_score += 20
                if adx > 30:                 pb_score += 20
                elif adx > 20:               pb_score += 12
                if 40 <= rsi <= 55:          pb_score += 20   # RSI cooling, not washed out
                elif 35 <= rsi <= 60:        pb_score += 10
                if near_ema9:                pb_score += 15   # tighter entry = higher quality
                elif near_ema21:             pb_score += 10
                if 5.0 <= pullback_pct <= 12.0: pb_score += 15  # sweet spot depth
                elif pullback_pct <= 5.0:       pb_score += 8

                # ── SL / Target ──────────────────────────────────────────────
                sl     = round(e21 - 1.0 * atr_val, 2)   # below EMA21 by 1 ATR
                target = round(lc  + 3.0 * atr_val, 2)   # 3R target

                s = meta[ticker]
                rows.append({
                    "Symbol":       s["symbol"],
                    "Name":         s["name"],
                    "Cap":          s.get("cap", "").upper(),
                    "Stage":        stage,
                    "PB Score":     pb_score,
                    "Price (₹)":    round(lc, 2),
                    "Day Chg %":    round(chg, 2),
                    "Pullback %":   round(pullback_pct, 1),
                    "EMA9":         round(e9, 2),
                    "EMA21":        round(e21, 2),
                    "EMA50":        round(e50, 2),
                    "RSI":          round(rsi, 1),
                    "ADX":          round(adx, 1),
                    "Vol Ratio":    vol_ratio,
                    "Stop (₹)":     sl,
                    "Target (₹)":   target,
                    "ATR":          round(atr_val, 2),
                    "_yf_ticker":   ticker,
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out.sort_values(["PB Score", "Pullback %"], ascending=[False, True], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


# ── NSE Bhavcopy helpers for SME OHLCV ───────────────────────────────────────

def _nse_bhavcopy_dates(n_days: int = 60) -> list[str]:
    """Return last n_days calendar dates (YYYYMMDD) excluding weekends, newest first."""
    from datetime import date, timedelta
    result = []
    d = date.today()
    while len(result) < n_days:
        d -= timedelta(days=1)
        if d.weekday() < 5:            # Mon-Fri only
            result.append(d.strftime("%Y%m%d"))
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_nse_bhavcopy_day(date_str: str) -> pd.DataFrame:
    """Download one NSE CM bhavcopy ZIP and return SM-series rows."""
    import io, zipfile, urllib.request
    url = (
        f"https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.csv.zip"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        req  = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=20)
        if resp.status != 200:
            return pd.DataFrame()
        raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = z.namelist()[0]
            df   = pd.read_csv(z.open(name))
        # keep only SM-series rows
        srs_col = next((c for c in df.columns if "SctySrs" in c or "Series" in c), None)
        if srs_col:
            df = df[df[srs_col].astype(str).str.strip().str.upper() == "SM"].copy()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _build_sme_ohlcv(n_days: int = 60) -> dict[str, pd.DataFrame]:
    """
    Download the last n_days of NSE bhavcopy files and build a per-symbol
    OHLCV DataFrame (Date-indexed) for every SME stock found.
    Returns {symbol: DataFrame} where DataFrame has columns Open/High/Low/Close/Volume.
    """
    dates  = _nse_bhavcopy_dates(n_days)
    pieces: list[pd.DataFrame] = []

    for ds in dates:
        day_df = _fetch_nse_bhavcopy_day(ds)
        if day_df.empty:
            continue

        # --- Detect column names dynamically ---
        cols = list(day_df.columns)

        sym_col  = next((c for c in cols if c in ("TckrSymb", "Symbol")), None) or \
                   next((c for c in cols if c == "FinInstrmId"), None)
        date_col = next((c for c in cols if c in ("TradDt", "BizDt", "TradeDate", "Date")), None)
        open_col = next((c for c in cols if c in ("OpnPric", "Open", "OPEN")), None)
        high_col = next((c for c in cols if c in ("HghPric", "High", "HIGH")), None)
        low_col  = next((c for c in cols if c in ("LwPric", "Low", "LOW")), None)
        cls_col  = next((c for c in cols if c in ("ClsPric", "Close", "CLOSE")), None)
        vol_col  = next((c for c in cols if c in ("TtlTradgVol", "Volume", "VOLUME", "TotTrdQty")), None)

        if not all([sym_col, open_col, high_col, low_col, cls_col, vol_col]):
            continue

        piece = day_df[[sym_col, open_col, high_col, low_col, cls_col, vol_col]].copy()
        piece.columns = ["Symbol", "Open", "High", "Low", "Close", "Volume"]

        if date_col:
            piece["Date"] = pd.to_datetime(day_df[date_col], dayfirst=False, errors="coerce")
        else:
            piece["Date"] = pd.Timestamp(ds)

        pieces.append(piece)

    if not pieces:
        return {}

    combined = pd.concat(pieces, ignore_index=True)
    combined["Symbol"] = combined["Symbol"].astype(str).str.strip().str.upper()
    for c in ("Open", "High", "Low", "Close"):
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
    combined["Volume"] = pd.to_numeric(combined["Volume"], errors="coerce").fillna(0)
    combined.dropna(subset=["Date", "Close"], inplace=True)
    combined.sort_values(["Symbol", "Date"], inplace=True)

    result: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].copy()
        g = g[~g.index.duplicated(keep="last")]
        result[str(sym)] = g

    return result


@st.cache_data(ttl=1800, show_spinner=False)
def scan_sme_stocks() -> pd.DataFrame:
    """
    Scan NSE Emerge SME stocks for trade setups.
    Uses NSE bhavcopy (last 60 trading days) for OHLCV since yfinance
    has insufficient history for NSE Emerge stocks.
    """
    universe = load_sme_universe()
    if not universe:
        return pd.DataFrame()

    # Build OHLCV from bhavcopy
    ohlcv_map = _build_sme_ohlcv(n_days=60)
    if not ohlcv_map:
        return pd.DataFrame()

    # Map bhavcopy symbols → universe metadata
    # universe keys are NSE symbols (upper-case), bhavcopy symbols match
    sym_to_meta: dict[str, dict] = {}
    for key, meta in universe.items():
        sym = meta["symbol"].upper()
        sym_to_meta[sym] = meta

    rows = []
    for sym, df in ohlcv_map.items():
        try:
            df = df.copy()
            df.dropna(subset=["Close"], inplace=True)
            if len(df) < 20:
                continue

            close  = df["Close"]
            high   = df["High"]
            low    = df["Low"]

            ema9   = close.ewm(span=9,   adjust=False).mean()
            ema21  = close.ewm(span=21,  adjust=False).mean()
            ema50  = close.ewm(span=50,  adjust=False).mean()

            prev_c = close.shift(1)
            tr     = pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
            atr_s  = tr.ewm(com=13, min_periods=14).mean()

            delta  = close.diff()
            avg_g  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
            avg_l  = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
            rsi_s  = 100 - (100 / (1 + avg_g / avg_l.replace(0, np.nan)))

            up_move  = high - high.shift(1)
            dn_move  = low.shift(1) - low
            plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
            atr14    = tr.ewm(com=13, min_periods=14).mean()
            plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(com=13, min_periods=14).mean() / atr14.replace(0, np.nan)
            minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(com=13, min_periods=14).mean() / atr14.replace(0, np.nan)
            dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx_s    = dx.ewm(com=13, min_periods=14).mean()

            # Supertrend
            hl2     = (high + low) / 2
            b_upper = (hl2 + 3.0 * atr_s).values
            b_lower = (hl2 - 3.0 * atr_s).values
            c_arr   = close.values
            n       = len(df)
            f_upper = b_upper.copy(); f_lower = b_lower.copy()
            for _i in range(1, n):
                if np.isnan(atr_s.iloc[_i]): continue
                f_upper[_i] = b_upper[_i] if (np.isnan(f_upper[_i-1]) or b_upper[_i] < f_upper[_i-1] or c_arr[_i-1] > f_upper[_i-1]) else f_upper[_i-1]
                f_lower[_i] = b_lower[_i] if (np.isnan(f_lower[_i-1]) or b_lower[_i] > f_lower[_i-1] or c_arr[_i-1] < f_lower[_i-1]) else f_lower[_i-1]
            st_dir = np.zeros(n, dtype=int)
            for _i in range(1, n):
                if np.isnan(f_upper[_i]) or np.isnan(f_lower[_i]): continue
                pd_ = st_dir[_i-1]
                if pd_ == 0:   st_dir[_i] = 1 if c_arr[_i] >= (f_upper[_i]+f_lower[_i])/2 else -1
                elif pd_ == 1: st_dir[_i] = -1 if c_arr[_i] < f_lower[_i] else 1
                else:          st_dir[_i] =  1 if c_arr[_i] > f_upper[_i] else -1

            lc      = float(close.iloc[-1])
            pc      = float(close.iloc[-2]) if n >= 2 else lc
            e9      = float(ema9.iloc[-1])
            e21     = float(ema21.iloc[-1])
            e50     = float(ema50.iloc[-1])
            rsi     = float(rsi_s.iloc[-1])
            adx     = float(adx_s.iloc[-1])
            pdi     = float(plus_di.iloc[-1])
            mdi     = float(minus_di.iloc[-1])
            st      = int(st_dir[-1])
            atr_val = float(atr_s.iloc[-1])
            avg_vol = float(df["Volume"].iloc[-20:].mean())
            vol_ratio = round(float(df["Volume"].iloc[-1]) / avg_vol, 1) if avg_vol > 0 else 0
            chg     = (lc - pc) / pc * 100 if pc else 0

            if any(np.isnan(v) for v in (rsi, adx, e9, e21)):
                continue

            # SME-specific liquidity guard — skip if avg daily volume < 5,000 shares
            if avg_vol < 5000:
                continue

            # Signal
            if e9 > e21 > e50 and lc > e21 and st == 1 and pdi > mdi:
                direction = "BUY"
            elif e9 < e21 < e50 and lc < e21 and st == -1 and mdi > pdi:
                direction = "SELL"
            else:
                direction = "WATCH"

            # Score
            score = 0
            if adx > 35:    score += 25
            elif adx > 25:  score += 18
            elif adx > 20:  score += 8
            if direction == "BUY"  and st ==  1: score += 20
            if direction == "SELL" and st == -1: score += 20
            if direction == "WATCH": score += 5
            if e9 > e21 > e50: score += 15
            elif e9 < e21 < e50: score += 15
            if vol_ratio >= 2.0:   score += 15
            elif vol_ratio >= 1.5: score += 10
            elif vol_ratio >= 1.0: score += 5
            if 45 <= rsi <= 65:    score += 10
            elif 35 <= rsi <= 70:  score += 5
            if direction == "BUY"  and lc > e21: score += 10
            if direction == "SELL" and lc < e21: score += 10

            # SL / Target
            sl     = round(lc - 1.5 * atr_val, 2) if direction == "BUY"  else (round(lc + 1.5 * atr_val, 2) if direction == "SELL" else None)
            target = round(lc + 3.0 * atr_val, 2) if direction == "BUY"  else (round(lc - 3.0 * atr_val, 2) if direction == "SELL" else None)

            # Liquidity rating
            if avg_vol >= 100_000:   liquidity = "🟢 High"
            elif avg_vol >= 20_000:  liquidity = "🟡 Medium"
            else:                    liquidity = "🔴 Low"

            # Metadata from universe (optional enrichment)
            s_meta = sym_to_meta.get(sym, {})
            name   = s_meta.get("name", sym)

            rows.append({
                "Symbol":     sym,
                "Name":       name,
                "Signal":     direction,
                "Score":      score,
                "Price (₹)":  round(lc, 2),
                "Day Chg %":  round(chg, 2),
                "ADX":        round(adx, 1),
                "RSI":        round(rsi, 1),
                "Vol Ratio":  vol_ratio,
                "Avg Vol":    int(avg_vol),
                "Liquidity":  liquidity,
                "Stop (₹)":   sl,
                "Target (₹)": target,
                "ATR":        round(atr_val, 2),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out.sort_values(["Score", "ADX"], ascending=[False, False], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_sentiment() -> dict:
    """
    Fetch key Indian market indices, India VIX, and sector performance.
    Returns a dict with keys: indices, sectors, vix, breadth.
    Cached 5 min.
    """
    # ── Key indices ───────────────────────────────────────────────────────────
    INDEX_TICKERS = {
        "Nifty 50":    "^NSEI",
        "Sensex":      "^BSESN",
        "Bank Nifty":  "^NSEBANK",
        "Nifty IT":    "^CNXIT",
        "Nifty Midcap 100": "^CNXMIDCAP",
        "Nifty Smallcap": "^CNXSMALLCAP",
        "India VIX":   "^INDIAVIX",
    }

    # ── Sector indices ────────────────────────────────────────────────────────
    SECTOR_TICKERS = {
        "IT":       "^CNXIT",
        "Bank":     "^NSEBANK",
        "Auto":     "^CNXAUTO",
        "Pharma":   "^CNXPHARMA",
        "FMCG":     "^CNXFMCG",
        "Metal":    "^CNXMETAL",
        "Energy":   "^CNXENERGY",
        "Realty":   "^CNXREALTY",
        "Infra":    "^CNXINFRA",
        "Media":    "^CNXMEDIA",
        "Finance":  "^CNXFINANCE",
        "PSU Bank": "^CNXPSUBANK",
    }

    def _fetch_quote(ticker: str) -> dict | None:
        try:
            df = yf.download(ticker, period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 2:
                return None
            last  = float(df["Close"].iloc[-1])
            prev  = float(df["Close"].iloc[-2])
            chg   = last - prev
            chg_p = chg / prev * 100
            week_ago = float(df["Close"].iloc[0])
            week_chg = (last - week_ago) / week_ago * 100
            return {"last": last, "chg": chg, "chg_pct": chg_p, "week_chg": week_chg}
        except Exception:
            return None

    indices = {}
    for name, tkr in INDEX_TICKERS.items():
        q = _fetch_quote(tkr)
        if q:
            indices[name] = q

    sectors = {}
    for name, tkr in SECTOR_TICKERS.items():
        q = _fetch_quote(tkr)
        if q:
            sectors[name] = q

    # ── FII / DII flows from NSE (best-effort scrape) ─────────────────────────
    fii_dii = []
    try:
        url  = "https://www.nseindia.com/api/fiidiiTradeReact"
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        import requests as _req
        # First hit homepage to get cookies
        s = _req.Session()
        s.get("https://www.nseindia.com", headers=hdrs, timeout=8)
        r = s.get(url, headers=hdrs, timeout=8)
        if r.status_code == 200:
            data = r.json()
            for row in data[:5]:   # last 5 trading days
                fii_dii.append({
                    "Date":       row.get("date", ""),
                    "FII Net (₹Cr)": row.get("fiidiiData", [{}])[0].get("netVal", 0) if row.get("fiidiiData") else 0,
                    "DII Net (₹Cr)": row.get("fiidiiData", [{}])[1].get("netVal", 0) if len(row.get("fiidiiData", [])) > 1 else 0,
                })
    except Exception:
        pass

    # ── Overall sentiment score ───────────────────────────────────────────────
    score = 50  # neutral baseline
    nifty = indices.get("Nifty 50", {})
    vix   = indices.get("India VIX", {})

    if nifty:
        if nifty["chg_pct"] > 1:    score += 15
        elif nifty["chg_pct"] > 0:  score += 7
        elif nifty["chg_pct"] < -1: score -= 15
        else:                        score -= 7

        if nifty["week_chg"] > 2:   score += 10
        elif nifty["week_chg"] < -2: score -= 10

    if vix:
        if vix["last"] < 15:    score += 10
        elif vix["last"] < 20:  score += 5
        elif vix["last"] > 25:  score -= 10
        elif vix["last"] > 20:  score -= 5

    # Sector breadth — how many sectors are green today
    green_sectors = sum(1 for v in sectors.values() if v["chg_pct"] > 0)
    total_sectors = len(sectors) or 1
    breadth_ratio = green_sectors / total_sectors
    if breadth_ratio >= 0.7:   score += 10
    elif breadth_ratio <= 0.3: score -= 10

    score = max(0, min(100, score))

    if score >= 65:   sentiment = "BULLISH"
    elif score >= 45: sentiment = "NEUTRAL"
    else:             sentiment = "BEARISH"

    return {
        "indices":   indices,
        "sectors":   sectors,
        "fii_dii":   fii_dii,
        "score":     score,
        "sentiment": sentiment,
        "green_sectors": green_sectors,
        "total_sectors": total_sectors,
    }


@st.cache_data(ttl=86400, show_spinner="Loading stock universe…")
def load_universe() -> dict:
    """Mainboard stocks only — SME excluded (insufficient yfinance history for charting)."""
    stocks = get_universe()
    result = {}
    for s in stocks:
        if s.cap == "sme":
            continue   # SME stocks live in the SME tab only; yfinance coverage is too thin to chart
        label = f"{s.symbol}  —  {s.name}  ({s.exchange})"
        result[label] = {
            "symbol": s.symbol, "name": s.name,
            "exchange": s.exchange, "cap": s.cap,
            "yf_ticker": s.yf_ticker,
        }
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def load_sme_universe() -> dict:
    """SME-only universe for the SME tab."""
    stocks = get_universe()
    result = {}
    for s in stocks:
        if s.cap != "sme":
            continue
        label = f"{s.symbol}  —  {s.name}  (NSE Emerge)"
        result[label] = {
            "symbol": s.symbol, "name": s.name,
            "exchange": s.exchange, "cap": s.cap,
            "yf_ticker": s.yf_ticker,
        }
    return result


_OHLCV = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}

def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns that yfinance 0.2.50+ returns for single-ticker
    downloads. Detects which level holds the OHLCV names and keeps that one.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Find which level contains OHLCV names
        lvl = 0
        if len(set(df.columns.get_level_values(0)) & _OHLCV) < \
           len(set(df.columns.get_level_values(1)) & _OHLCV):
            lvl = 1
        df.columns = df.columns.get_level_values(lvl)
        df = df.loc[:, ~df.columns.duplicated()]
    # Ensure every column is a plain 1-D Series
    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    return df


_CACHE_VERSION = "v5"   # bump this to invalidate all cached OHLCV on next deploy

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(yf_ticker: str, interval: str, period: str,
                _v: str = _CACHE_VERSION) -> pd.DataFrame | None:
    # Build list of tickers to try. For NSE stocks that failed, also try the
    # SME suffix (-SM.NS) in case the universe cache has a stale plain ticker.
    candidates = [yf_ticker]
    if yf_ticker.endswith(".NS") and "-SM" not in yf_ticker:
        base = yf_ticker[:-3]  # strip ".NS"
        candidates.append(f"{base}-SM.NS")

    for ticker in candidates:
        for attempt in range(3):
            try:
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True)
                df = _flatten_yf(df)
                df = df.dropna()
                if not df.empty and len(df) >= 20:
                    return df
            except Exception:
                pass
            time.sleep(1)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(yf_ticker: str) -> dict:
    try:
        return yf.Ticker(yf_ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_day_range(yf_ticker: str) -> tuple[float, float] | None:
    """Return (day_high, day_low) using fast_info — live, unadjusted."""
    try:
        fi = yf.Ticker(yf_ticker).fast_info
        return float(fi.day_high), float(fi.day_low)
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_spot_price(yf_ticker: str) -> float | None:
    """Quick last-price fetch for ATM strike computation."""
    try:
        df = yf.download(yf_ticker, period="1d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].dropna().iloc[-1]) if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_signal(yf_ticker: str) -> dict:
    """
    Fetch OHLCV, compute indicators, return bias + score + key values.
    Used by the portfolio tab to show live signal per holding.
    """
    try:
        df = yf.download(yf_ticker, period="2y", interval="1d",
                         progress=False, auto_adjust=True)
        df = _flatten_yf(df).dropna()
        if len(df) < 30:
            return {"bias": "N/A", "score": 0, "rsi": None, "adx": None,
                    "atr": None, "last_price": None, "error": "Not enough data"}
        df_e = compute_all(df)
        sig  = compute_signals(df_e)
        last = df_e.iloc[-1]
        fi   = yf.Ticker(yf_ticker).fast_info
        lp   = float(fi.last_price) if fi and fi.last_price else float(last["Close"])
        return {
            "bias":        sig["current_bias"],
            "score":       round(sig["strength_score"], 1),
            "regime":      sig.get("regime", "MIXED"),
            "rsi":         round(float(last["rsi"]),  1) if "rsi"  in df_e.columns else None,
            "adx":         round(float(last["adx"]),  1) if "adx"  in df_e.columns else None,
            "atr":         round(float(last["atr"]),  2) if "atr"  in df_e.columns else None,
            "ema21":       round(float(last["ema_21"]), 2) if "ema_21" in df_e.columns else None,
            "ema50":       round(float(last["ema_50"]), 2) if "ema_50" in df_e.columns else None,
            "last_price":  lp,
            "error":       None,
        }
    except Exception as e:
        return {"bias": "N/A", "score": 0, "rsi": None, "adx": None,
                "atr": None, "last_price": None, "error": str(e)}


@st.cache_data(ttl=120, show_spinner=False)
def fetch_nse_option_chain(symbol: str) -> pd.DataFrame | None:
    """
    Fetch live NSE option chain for NIFTY / BANKNIFTY / FINNIFTY / MIDCPNIFTY.
    Returns a flattened DataFrame with CE + PE rows per strike.
    """
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
    })
    try:
        session.get("https://www.nseindia.com/", timeout=8)
        url  = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        resp = session.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        data    = resp.json()
        records = data["records"]["data"]
        rows = []
        for rec in records:
            strike = rec["strikePrice"]
            for side in ("CE", "PE"):
                if side not in rec:
                    continue
                d = rec[side]
                rows.append({
                    "Strike":    strike,
                    "Type":      side,
                    "LTP":       d.get("lastPrice", 0),
                    "OI":        d.get("openInterest", 0),
                    "Chg OI":    d.get("changeinOpenInterest", 0),
                    "Volume":    d.get("totalTradedVolume", 0),
                    "IV %":      round(d.get("impliedVolatility", 0), 2),
                    "Bid":       d.get("bidprice", 0),
                    "Ask":       d.get("askPrice", 0),
                    "Δ Chg":     d.get("change", 0),
                })
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_chain(yf_ticker: str) -> dict | None:
    try:
        tk  = yf.Ticker(yf_ticker)
        exp = tk.options
        if not exp:
            return None
        chain = tk.option_chain(exp[0])
        return {"expirations": list(exp), "calls": chain.calls, "puts": chain.puts}
    except Exception:
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_trade_setups(signals: list, df: pd.DataFrame, bias: str = "NEUTRAL") -> list:
    """
    Build actionable trade setups anchored to TODAY's price.

    Key rules:
    - Entry is always the CURRENT close (last bar) — never a stale historical price
    - SL and Target are calculated from current ATR
    - Signals from the last 10 bars vote on direction and conviction
    - Strength threshold ≥ 2 (looser) so there's always something to show
    - If no signals qualify but bias is clear, generate a "Bias Setup"
    """
    from collections import defaultdict

    # ── Current price anchor (always today's close) ───────────────────────
    current_close = float(df["Close"].iloc[-1])
    current_atr   = float(df["atr"].iloc[-1]) if "atr" in df.columns and pd.notna(df["atr"].iloc[-1]) else current_close * 0.012
    current_date  = df.index[-1]

    # ── Keep only recent signals (last 10 bars) ───────────────────────────
    if len(df) >= 10:
        cutoff  = df.index[-10]
        signals = [s for s in signals if s["date"] >= cutoff]

    # ── Align with trend bias ─────────────────────────────────────────────
    if bias == "BULLISH":
        signals = [s for s in signals if s["type"] == "BUY"]
    elif bias == "BEARISH":
        signals = [s for s in signals if s["type"] == "SELL"]

    # ── Group all qualifying signals into one directional vote ────────────
    by_date: dict = defaultdict(list)
    for s in signals:
        if s["type"] in ("BUY", "SELL"):
            by_date[s["date"]].append(s)

    # Merge all recent signals into a single pool for overall conviction
    all_recent = [s for sigs in by_date.values() for s in sigs]
    buy_str    = sum(s["strength"] for s in all_recent if s["type"] == "BUY")
    sell_str   = sum(s["strength"] for s in all_recent if s["type"] == "SELL")
    total_str  = buy_str + sell_str

    setups = []

    if total_str >= 2:
        direction  = "BUY" if buy_str >= sell_str else "SELL"
        indicators = list(dict.fromkeys(s["indicator"] for s in all_recent))
        n_signals  = len(all_recent)
        conviction = "HIGH" if total_str >= 6 else "MEDIUM" if total_str >= 3 else "LOW"

        sl     = current_close - 1.5 * current_atr if direction == "BUY" else current_close + 1.5 * current_atr
        target = current_close + 3.0 * current_atr if direction == "BUY" else current_close - 3.0 * current_atr
        risk   = abs(current_close - sl)
        reward = abs(target - current_close)

        setups.append({
            "date":       current_date,
            "type":       direction,
            "entry":      current_close,
            "sl":         sl,
            "target":     target,
            "sl_diff":    sl - current_close,
            "tgt_diff":   target - current_close,
            "sl_pct":     (sl - current_close) / current_close * 100,
            "tgt_pct":    (target - current_close) / current_close * 100,
            "rr":         round(reward / risk, 1) if risk > 0 else 0,
            "strength":   total_str,
            "indicators": indicators,
            "n_signals":  n_signals,
            "conviction": conviction,
        })

    # ── Fallback: bias-driven setup when no indicator signals ─────────────
    elif bias in ("BULLISH", "BEARISH"):
        direction = "BUY" if bias == "BULLISH" else "SELL"
        sl     = current_close - 1.5 * current_atr if direction == "BUY" else current_close + 1.5 * current_atr
        target = current_close + 3.0 * current_atr if direction == "BUY" else current_close - 3.0 * current_atr
        risk   = abs(current_close - sl)
        reward = abs(target - current_close)

        setups.append({
            "date":       current_date,
            "type":       direction,
            "entry":      current_close,
            "sl":         sl,
            "target":     target,
            "sl_diff":    sl - current_close,
            "tgt_diff":   target - current_close,
            "sl_pct":     (sl - current_close) / current_close * 100,
            "tgt_pct":    (target - current_close) / current_close * 100,
            "rr":         round(reward / risk, 1) if risk > 0 else 0,
            "strength":   1,
            "indicators": ["Bias"],
            "n_signals":  0,
            "conviction": "LOW",
        })

    return setups


def _run_backtest(signals: list, df: pd.DataFrame, initial_capital: float) -> tuple[list, float]:
    """
    Upgraded backtest with 5 guard layers:

    1. No overlapping trades    — one position at a time
    2. Macro trend filter       — skip BUY below EMA50+EMA200, skip SELL above them
    3. Volatility filter        — skip when ATR/Price < 0.3% (choppy market)
    4. Time-of-day filter       — intraday: skip first 10 min and midday (12:00–13:30)
    5. Pullback entry           — wait up to 5 bars for retracement to EMA9/EMA21
                                  before entering; skip if price runs > 1 ATR away

    Exit logic (Trailing Stop):
    - At 2R profit → partial exit 50% of position, trail remaining with EMA21
    - Stop loss     → exit 100% (before partial) or remaining (after partial)
    - Time exit     → max 40 bars hold; exit remaining at close
    """
    from collections import defaultdict

    def _fmt_dt(d):
        return d.strftime("%d %b %Y") if hasattr(d, "strftime") else str(d)

    by_date: dict = defaultdict(list)
    for s in signals:
        if s["type"] in ("BUY", "SELL"):
            by_date[s["date"]].append(s)

    trades:         list  = []
    capital:        float = float(initial_capital)
    open_until_loc: int   = -1

    is_intraday = (
        isinstance(df.index, pd.DatetimeIndex)
        and df.index.resolution in ("minute", "hour")
    )

    for dt in sorted(by_date.keys()):
        sigs     = by_date[dt]
        buy_str  = sum(s["strength"] for s in sigs if s["type"] == "BUY")
        sell_str = sum(s["strength"] for s in sigs if s["type"] == "SELL")
        if max(buy_str, sell_str) < 4:
            continue

        direction = "BUY" if buy_str >= sell_str else "SELL"

        try:
            sig_loc = df.index.get_indexer([dt], method="nearest")[0]
            sig_close = float(df["Close"].iloc[sig_loc])
            atr_val   = float(df["atr"].iloc[sig_loc]) if "atr" in df.columns else sig_close * 0.012
            if pd.isna(atr_val):
                atr_val = sig_close * 0.012
        except Exception:
            continue

        # ── Guard 1: no overlapping trades ────────────────────────────────
        if sig_loc <= open_until_loc:
            continue

        # ── Guard 2: macro trend filter ───────────────────────────────────
        ema50  = df["ema_50"].iloc[sig_loc]  if "ema_50"  in df.columns else np.nan
        ema200 = df["ema_200"].iloc[sig_loc] if "ema_200" in df.columns else np.nan
        if not (pd.isna(ema50) or pd.isna(ema200)):
            if direction == "BUY"  and sig_close < float(ema50) and sig_close < float(ema200):
                continue
            if direction == "SELL" and sig_close > float(ema50) and sig_close > float(ema200):
                continue

        # ── Guard 3: volatility filter ────────────────────────────────────
        if atr_val / sig_close < 0.003:        # ATR < 0.3% of price → skip chop
            continue

        # ── Guard 4: time-of-day filter (intraday only) ───────────────────
        if is_intraday and hasattr(dt, "hour"):
            t_mins = dt.hour * 60 + dt.minute
            # Skip first 10 min after NSE open (9:15–9:25) and lunch (12:00–13:30)
            if t_mins < 565 or (720 <= t_mins <= 810):
                continue

        # ── Guard 5: pullback entry — wait up to 5 bars for EMA9/21 touch ─
        entry_loc   = sig_loc
        entry_price = sig_close
        ema9_col    = "ema_9"  if "ema_9"  in df.columns else None
        ema21_col   = "ema_21" if "ema_21" in df.columns else None

        for pb_j in range(sig_loc + 1, min(sig_loc + 6, len(df))):
            pb_low   = float(df["Low"].iloc[pb_j])
            pb_high  = float(df["High"].iloc[pb_j])
            pb_close = float(df["Close"].iloc[pb_j])

            # If price runs away > 1 ATR without pulling back → skip the trade
            gap = (pb_close - sig_close) if direction == "BUY" else (sig_close - pb_close)
            if gap > atr_val:
                entry_loc = -1   # signal "no pullback found, skip"
                break

            for ecol in filter(None, [ema9_col, ema21_col]):
                ema_v = float(df[ecol].iloc[pb_j])
                if pd.isna(ema_v):
                    continue
                if direction == "BUY"  and pb_low <= ema_v <= pb_high:
                    entry_loc   = pb_j
                    entry_price = ema_v
                    break
                if direction == "SELL" and pb_low <= ema_v <= pb_high:
                    entry_loc   = pb_j
                    entry_price = ema_v
                    break
            if entry_loc != sig_loc:
                break

        if entry_loc == -1:
            continue   # price ran away — skip

        # ── Position sizing ───────────────────────────────────────────────
        shares = int(capital * 0.95 / entry_price)
        if shares <= 0:
            continue

        initial_risk   = 1.5 * atr_val
        sl             = entry_price - initial_risk if direction == "BUY" else entry_price + initial_risk
        partial_target = entry_price + 2 * initial_risk if direction == "BUY" else entry_price - 2 * initial_risk

        # ── Forward simulation with trailing stop ─────────────────────────
        half         = max(1, shares // 2)
        rest         = shares - half
        partial_done = False
        trail_sl     = sl
        pnl          = 0.0
        exit_price   = entry_price
        exit_reason  = "Time Exit"
        exit_loc     = min(entry_loc + 40, len(df) - 1)

        for j in range(entry_loc + 1, min(entry_loc + 41, len(df))):
            h  = float(df["High"].iloc[j])
            lo = float(df["Low"].iloc[j])

            if direction == "BUY":
                if lo <= trail_sl:
                    rem = rest if partial_done else shares
                    pnl += rem * (trail_sl - entry_price)
                    exit_price  = trail_sl
                    exit_reason = "Trail Stop" if partial_done else "Stop Loss"
                    exit_loc    = j
                    break
                if not partial_done and h >= partial_target:
                    pnl          += half * (partial_target - entry_price)
                    partial_done  = True
                    trail_sl      = entry_price          # move stop to breakeven
                if partial_done and ema21_col:
                    ev = float(df[ema21_col].iloc[j])
                    if not np.isnan(ev) and ev > trail_sl:
                        trail_sl = ev
            else:  # SELL
                if h >= trail_sl:
                    rem = rest if partial_done else shares
                    pnl += rem * (entry_price - trail_sl)
                    exit_price  = trail_sl
                    exit_reason = "Trail Stop" if partial_done else "Stop Loss"
                    exit_loc    = j
                    break
                if not partial_done and lo <= partial_target:
                    pnl          += half * (entry_price - partial_target)
                    partial_done  = True
                    trail_sl      = entry_price
                if partial_done and ema21_col:
                    ev = float(df[ema21_col].iloc[j])
                    if not np.isnan(ev) and ev < trail_sl:
                        trail_sl = ev
        else:
            # Loop completed without break → time exit on remaining
            close_at = float(df["Close"].iloc[exit_loc])
            rem      = rest if partial_done else shares
            pnl     += rem * (close_at - entry_price) if direction == "BUY" \
                        else rem * (entry_price - close_at)
            exit_price = close_at

        open_until_loc = exit_loc
        capital       += pnl

        amount_invested = round(shares * entry_price, 2)
        current_amount  = round(
            (half * partial_target + rest * exit_price) if partial_done
            else shares * exit_price, 2
        )

        trades.append({
            "Entry Date":      _fmt_dt(dt),
            "Exit Date":       _fmt_dt(df.index[exit_loc]),
            "Type":            direction,
            "Entry Price":     round(entry_price, 2),
            "Exit Price":      round(exit_price, 2),
            "SL":              round(sl, 2),
            "Partial @ 2R":    round(partial_target, 2),
            "Shares":          shares,
            "Amount Invested": amount_invested,
            "Current Amount":  current_amount,
            "P&L":             round(pnl, 2),
            "P&L %":           round(pnl / amount_invested * 100, 2) if amount_invested else 0,
            "Result":          exit_reason,
            "Capital After":   round(capital, 2),
        })

    return trades, round(capital, 2)


def _options_strategy(bias: str, regime: str, atr_pct: float) -> dict:
    """
    Map market conditions to the most appropriate options strategy.

    Parameters
    ----------
    bias     : "BULLISH" | "BEARISH" | "NEUTRAL"
    regime   : "TRENDING" | "RANGING" | "MIXED"
    atr_pct  : ATR as % of price (volatility proxy)

    Returns a dict with 'strategy', 'legs', 'rationale', 'risk'
    """
    low_vol  = atr_pct < 0.008   # ATR < 0.8% of price → low volatility
    high_vol = atr_pct > 0.020   # ATR > 2% of price → high volatility

    if regime == "TRENDING":
        if bias == "BULLISH":
            return {
                "strategy": "Buy ATM Call",
                "legs":     "BUY 1 ATM CE",
                "rationale": "Strong uptrend — buy directional call to ride the move with limited risk.",
                "risk":     "Premium paid · Theta decay · Needs quick move",
                "emoji":    "📈",
            }
        elif bias == "BEARISH":
            return {
                "strategy": "Buy ATM Put",
                "legs":     "BUY 1 ATM PE",
                "rationale": "Strong downtrend — buy directional put to profit from the fall.",
                "risk":     "Premium paid · Theta decay · Needs quick move",
                "emoji":    "📉",
            }
        else:  # NEUTRAL in trend
            return {
                "strategy": "Bull/Bear Spread (wait for bias)",
                "legs":     "Wait for BULLISH or BEARISH bias before entering",
                "rationale": "Trend is strong but direction unclear. Wait for bias to confirm.",
                "risk":     "Missing move if you act too early",
                "emoji":    "⏳",
            }
    elif regime == "RANGING":
        if low_vol:
            return {
                "strategy": "Short Iron Condor",
                "legs":     "SELL OTM CE + BUY further OTM CE  |  SELL OTM PE + BUY further OTM PE",
                "rationale": "Market is ranging with low volatility — collect premium from both sides.",
                "risk":     "Large move in either direction · Max loss = spread width − premium",
                "emoji":    "🪤",
            }
        elif bias == "BULLISH":
            return {
                "strategy": "Bull Call Spread",
                "legs":     "BUY ATM CE + SELL OTM CE (1 strike up)",
                "rationale": "Ranging market with bullish lean — spread reduces cost vs naked call.",
                "risk":     "Capped upside · Max profit = strike diff − net premium",
                "emoji":    "📊",
            }
        elif bias == "BEARISH":
            return {
                "strategy": "Bear Put Spread",
                "legs":     "BUY ATM PE + SELL OTM PE (1 strike down)",
                "rationale": "Ranging market with bearish lean — spread reduces cost vs naked put.",
                "risk":     "Capped downside capture · Max profit = strike diff − net premium",
                "emoji":    "📊",
            }
        else:
            return {
                "strategy": "Short Strangle",
                "legs":     "SELL OTM CE + SELL OTM PE (both 1–2 strikes out)",
                "rationale": "Market going sideways — sell both sides and collect time decay.",
                "risk":     "Unlimited risk if market breaks out hard",
                "emoji":    "🎯",
            }
    else:  # MIXED
        if low_vol:
            return {
                "strategy": "Long Straddle (pre-breakout)",
                "legs":     "BUY 1 ATM CE + BUY 1 ATM PE",
                "rationale": "Low volatility in a mixed market — cheap straddle before a potential big move.",
                "risk":     "Double theta decay · Needs large move to profit",
                "emoji":    "💥",
            }
        elif high_vol and bias == "NEUTRAL":
            return {
                "strategy": "Short Straddle / Strangle",
                "legs":     "SELL 1 ATM CE + SELL 1 ATM PE",
                "rationale": "High volatility but no clear direction — sell premium and profit from IV crush.",
                "risk":     "Unlimited risk · Needs market to stay in range",
                "emoji":    "💰",
            }
        elif bias == "BULLISH":
            return {
                "strategy": "Buy Call / Bull Call Spread",
                "legs":     "BUY ATM CE  (or) BUY ATM CE + SELL OTM CE",
                "rationale": "Mixed market with bullish lean — naked call for aggressive play, spread for safer.",
                "risk":     "Premium at risk · Spread caps profit",
                "emoji":    "📈",
            }
        elif bias == "BEARISH":
            return {
                "strategy": "Buy Put / Bear Put Spread",
                "legs":     "BUY ATM PE  (or) BUY ATM PE + SELL OTM PE",
                "rationale": "Mixed market with bearish lean — naked put for aggressive, spread for safer.",
                "risk":     "Premium at risk · Spread caps profit",
                "emoji":    "📉",
            }
        else:
            return {
                "strategy": "Wait — No Clear Edge",
                "legs":     "Hold cash until regime and bias align",
                "rationale": "Mixed market with no bias. Best edge comes from clarity — wait.",
                "risk":     "Opportunity cost only",
                "emoji":    "🕐",
            }


def _chart_signals_deduped(signals: list) -> list:
    """One chart marker per date — strongest BUY or SELL wins."""
    best: dict = {}
    for s in signals:
        if s["type"] not in ("BUY", "SELL"):
            continue
        dt = s["date"]
        if dt not in best or s["strength"] > best[dt]["strength"]:
            best[dt] = s
    return list(best.values())


def _fmt_vol(v: float) -> str:
    if v >= 1e7: return f"{v/1e7:.2f} Cr"
    if v >= 1e5: return f"{v/1e5:.2f} L"
    return f"{v:,.0f}"

def _fmt_price(v) -> str:
    return f"₹{float(v):,.2f}" if v and not (isinstance(v, float) and np.isnan(v)) else "—"

def _strength_dots(n: int) -> str:
    return "●" * n + "○" * (3 - n)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📈 Live Trading")

    category = st.radio(
        "Category",
        ["Stocks", "Indices", "Currencies", "Custom"],
        horizontal=True,
        label_visibility="collapsed",
    )

    stock = None

    if category == "Stocks":
        universe = load_universe()
        lbl   = st.selectbox("Stock", list(universe.keys()),
                             placeholder="🔍 Search symbol or name…",
                             label_visibility="collapsed")
        stock = universe[lbl]

    elif category == "Indices":
        idx_type = st.radio(
            "Type", ["Index", "Futures", "Options"],
            horizontal=True, key="idx_type",
            label_visibility="collapsed",
        )

        if idx_type == "Index":
            key   = st.selectbox("Index", list(INDICES.keys()), label_visibility="collapsed")
            stock = INDICES[key]

        elif idx_type == "Futures":
            underlying = st.selectbox(
                "Underlying", list(_FNO_UNDERLYINGS.keys()),
                label_visibility="collapsed", key="fut_ul",
            )
            expiries   = _nse_monthly_expiries(3)
            exp_choice = st.selectbox(
                "Expiry", range(len(expiries)),
                format_func=lambda i: expiries[i][0],
                label_visibility="collapsed", key="fut_exp",
            )
            label, code = expiries[exp_choice]
            ticker = f"{underlying}{code}FUT.NS"
            stock  = {
                "symbol":     f"{underlying} FUT",
                "name":       f"{underlying} Futures — {label}",
                "yf_ticker":  ticker,
                "exchange":   "NSE",
                "cap":        "futures",
                "underlying": underlying,
                "exp_label":  label,
            }
            st.caption(f"Chart uses {underlying} spot (NSE F&O not on Yahoo Finance)")

        else:  # Options
            underlying = st.selectbox(
                "Underlying", list(_FNO_UNDERLYINGS.keys()),
                label_visibility="collapsed", key="opt_ul",
            )
            ul_meta = _FNO_UNDERLYINGS[underlying]

            expiries   = _nse_monthly_expiries(3)
            exp_choice = st.selectbox(
                "Expiry", range(len(expiries)),
                format_func=lambda i: expiries[i][0],
                label_visibility="collapsed", key="opt_exp_sel",
            )
            label, code = expiries[exp_choice]

            # Fetch spot to centre strikes on ATM
            spot = fetch_spot_price(ul_meta["spot_ticker"]) or 23000.0
            step = ul_meta["step"]
            strikes = _atm_strikes(spot, step, 12)
            atm     = round(spot / step) * step
            atm_idx = strikes.index(atm) if atm in strikes else len(strikes) // 2

            strike = st.selectbox(
                "Strike", strikes,
                index=atm_idx,
                format_func=lambda s: f"{s}  {'← ATM' if s == atm else ('↑ OTM' if s > spot else '↓ ITM')}",
                label_visibility="collapsed", key="opt_strike",
            )
            opt_type = st.radio(
                "CE / PE", ["CE", "PE"],
                horizontal=True, key="opt_type",
                label_visibility="collapsed",
            )

            ticker = f"{underlying}{code}{strike}{opt_type}.NS"
            stock  = {
                "symbol":     f"{underlying} {strike}{opt_type}",
                "name":       f"{underlying} {strike} {opt_type} — {label}",
                "yf_ticker":  ticker,
                "exchange":   "NSE",
                "cap":        "options",
                "underlying": underlying,
                "strike":     strike,
                "opt_type":   opt_type,
                "exp_label":  label,
                "exp_code":   code,
                "spot":       spot,
                "atm":        atm,
            }
            st.caption(f"Spot ≈ {spot:,.0f}  ·  ATM {atm}")

    elif category == "Currencies":
        key   = st.selectbox("Pair", list(CURRENCIES.keys()), label_visibility="collapsed")
        stock = CURRENCIES[key]

    else:  # Custom
        raw = st.text_input(
            "Symbol",
            placeholder="e.g. RELIANCE25MARFUT.NS  or  NIFTY25APR23000CE.NS",
            label_visibility="collapsed",
        )
        st.caption("Enter any yfinance ticker — NSE futures, options, global stocks, crypto.")
        if raw.strip():
            sym   = raw.strip().split(".")[0].upper()
            exch  = "NSE" if raw.upper().endswith(".NS") else "BSE" if raw.upper().endswith(".BO") else "—"
            stock = {"symbol": sym, "name": raw.strip(), "exchange": exch,
                     "cap": "custom", "yf_ticker": raw.strip()}

    st.html("<hr class='thin-divider'>")

    st.markdown("**Timeframe**")
    timeframe   = st.select_slider("Timeframe", options=list(TIMEFRAMES.keys()), value="1D",
                                    label_visibility="collapsed")
    tf          = TIMEFRAMES[timeframe]
    period_bars = st.slider("Bars", 50, tf["max_bars"], min(200, tf["max_bars"]),
                             step=10, label_visibility="collapsed")
    st.caption(f"{period_bars} bars  ·  {timeframe} chart")

    st.html("<hr class='thin-divider'>")

    st.markdown("**Overlays**")
    ind_ema        = st.checkbox("EMA  9 / 21 / 50 / 200", value=True)
    ind_bb         = st.checkbox("Bollinger Bands",          value=True)
    ind_vwap       = st.checkbox("VWAP  (intraday only)",    value=True)
    ind_supertrend = st.checkbox("Supertrend  (10, 3)",      value=False)
    ind_donchian   = st.checkbox("Donchian Channels  (20)",  value=False)

    st.markdown("**Sub-charts**")
    ind_rsi   = st.checkbox("RSI (14)",               value=True)
    ind_macd  = st.checkbox("MACD",                   value=True)
    ind_stoch = st.checkbox("Stochastic",             value=False)

    st.markdown("**Signal Detection**")
    ind_candle = st.checkbox("Candlestick Patterns",  value=True)
    ind_volume = st.checkbox("Volume Signals",         value=True)

    st.html("<hr class='thin-divider'>")

    st.markdown("**Support & Resistance**")
    sr_pivot = st.checkbox("Pivot Points",       value=True)
    sr_swing = st.checkbox("Swing Highs / Lows", value=True)
    sr_fib   = st.checkbox("Fibonacci",          value=timeframe in ("1D", "1W"))

    st.html("<hr class='thin-divider'>")
    if st.button("🔄  Refresh live data", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# ─── Top-level tab layout ─────────────────────────────────────────────────────
tab_chart, tab_scan, tab_sentiment, tab_portfolio, tab_pullback, tab_sme = st.tabs([
    "📊  Live Chart", "🎯  Trade Opportunities", "🌡️  Market Sentiment",
    "📋  My Portfolio", "🔁  Pullback Scanner", "🏭  SME Stocks"
])


with tab_portfolio:
    st.markdown("### 📋 My Portfolio")
    st.caption("Track your Zerodha trades — live P&L, current signal, and risk status for each position.")

    # ── Load / initialise portfolio from disk ─────────────────────────────────
    if "portfolio_trades" not in st.session_state:
        st.session_state["portfolio_trades"] = _load_portfolio()

    trades: list[dict] = st.session_state["portfolio_trades"]

    # ── Capital input ─────────────────────────────────────────────────────────
    with st.expander("💼  Capital Settings", expanded=False):
        cap_col1, cap_col2 = st.columns([1, 3])
        with cap_col1:
            pf_capital = st.number_input(
                "Total Capital in Market (₹)",
                min_value=0.0, step=1000.0, format="%.0f",
                value=float(st.session_state.get("pf_capital", 0.0)),
                key="pf_capital",
                help="Total cash you have put into the market (across all open positions).",
            )
        with cap_col2:
            st.caption(
                "Enter the total amount you have deployed. "
                "This is used to show return % on capital and how much of your money is working."
            )

    # ── Add new trade form ────────────────────────────────────────────────────
    with st.expander("➕  Add New Trade", expanded=len(trades) == 0):
        universe_for_portfolio = load_universe()
        _pf_labels = sorted([
            k for k, v in universe_for_portfolio.items()
            if v.get("exchange") in ("NSE", "BSE")
        ])

        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            pf_label = st.selectbox(
                "Stock / Index",
                options=[""] + _pf_labels,
                format_func=lambda x: "Search…" if x == "" else x,
                key="pf_add_label",
            )
        with pf2:
            pf_direction = st.selectbox("Direction", ["BUY", "SELL"], key="pf_add_dir")
        with pf3:
            pf_entry = st.number_input("Entry Price (₹)", min_value=0.01, value=100.0,
                                       step=0.05, format="%.2f", key="pf_add_entry")

        pf4, pf5, pf6 = st.columns(3)
        with pf4:
            pf_qty = st.number_input("Quantity (shares)", min_value=1, value=10,
                                     step=1, key="pf_add_qty")
        with pf5:
            pf_sl = st.number_input("Stop Loss (₹)  [optional, 0 = ATR-based]",
                                    min_value=0.0, value=0.0, step=0.05, format="%.2f",
                                    key="pf_add_sl")
        with pf6:
            pf_target = st.number_input("Target (₹)  [optional, 0 = ATR-based]",
                                        min_value=0.0, value=0.0, step=0.05, format="%.2f",
                                        key="pf_add_target")

        pf7, pf8 = st.columns([1, 2])
        with pf7:
            pf_exit = st.number_input(
                "Exit Price (₹)  [0 = still open]",
                min_value=0.0, value=0.0, step=0.05, format="%.2f",
                key="pf_add_exit",
                help="If you already closed this trade, enter the price you exited at. Leave 0 if still open.",
            )
        with pf8:
            pf_notes = st.text_input("Notes (optional)", placeholder="e.g. Breakout above resistance",
                                     key="pf_add_notes")

        if st.button("Add to Portfolio", type="primary", key="pf_add_btn"):
            if pf_label and pf_entry > 0 and pf_qty > 0:
                stock_meta = universe_for_portfolio.get(pf_label, {})
                new_trade = {
                    "id":          str(uuid.uuid4())[:8],
                    "symbol":      stock_meta.get("symbol", pf_label.split("  —  ")[0]),
                    "name":        stock_meta.get("name", ""),
                    "yf_ticker":   stock_meta.get("yf_ticker", ""),
                    "exchange":    stock_meta.get("exchange", "NSE"),
                    "direction":   pf_direction,
                    "entry_price": round(pf_entry, 2),
                    "qty":         int(pf_qty),
                    "stop_loss":   round(pf_sl, 2) if pf_sl > 0 else None,
                    "target":      round(pf_target, 2) if pf_target > 0 else None,
                    "exit_price":  round(pf_exit, 2) if pf_exit > 0 else None,
                    "entry_date":  str(date.today()),
                    "notes":       pf_notes.strip(),
                }
                trades.append(new_trade)
                st.session_state["portfolio_trades"] = trades
                _save_portfolio(trades)
                st.success(f"Added {new_trade['symbol']} {pf_direction} @ ₹{pf_entry:,.2f} × {pf_qty}")
                st.rerun()
            else:
                st.warning("Select a stock and enter a valid entry price and quantity.")

    if not trades:
        st.info("No trades yet. Click **Add New Trade** above to log your first position.")

    if trades:
        col_ref, col_del = st.columns([5, 1])
        with col_ref:
            if st.button("🔄  Refresh signals", key="pf_refresh"):
                st.cache_data.clear()
        with col_del:
            _pf_clear = st.button("🗑 Clear All", key="pf_clear_all",
                                  help="Remove all trades from portfolio")
        if _pf_clear:
            st.session_state["portfolio_trades"] = []
            _save_portfolio([])
            st.rerun()

        _signal_cache: dict = {}
        with st.spinner("Fetching live prices and signals…"):
            for t in trades:
                tk = t.get("yf_ticker", "")
                if tk and tk not in _signal_cache:
                    _signal_cache[tk] = fetch_stock_signal(tk)

        # ── Build portfolio rows ──────────────────────────────────────────────────
        pf_rows = []
        for t in trades:
            sig = _signal_cache.get(t.get("yf_ticker", ""), {})
            lp  = sig.get("last_price") or t["entry_price"]
            atr = sig.get("atr") or 0

            entry      = t["entry_price"]
            qty        = t["qty"]
            is_buy     = t["direction"] == "BUY"
            exit_price = t.get("exit_price")          # None = still open
            is_closed  = exit_price is not None and exit_price > 0

            # Price used for P&L: exit price if closed, else live price
            price_for_pnl = exit_price if is_closed else lp

            # P&L
            pnl_per = (price_for_pnl - entry) if is_buy else (entry - price_for_pnl)
            pnl_inr = round(pnl_per * qty, 2)
            pnl_pct = round(pnl_per / entry * 100, 2) if entry else 0

            # Stop loss: user-defined first, fallback to 1.5×ATR (only meaningful for open trades)
            sl = t.get("stop_loss")
            if not sl and atr and not is_closed:
                sl = round(entry - 1.5 * atr, 2) if is_buy else round(entry + 1.5 * atr, 2)

            # Target: user-defined first, fallback to 3×ATR
            tgt = t.get("target")
            if not tgt and atr and not is_closed:
                tgt = round(entry + 3.0 * atr, 2) if is_buy else round(entry - 3.0 * atr, 2)

            # Risk assessment (only relevant for open trades)
            risk_per_share = abs(entry - sl) if (sl and not is_closed) else None
            risk_inr       = round(risk_per_share * qty, 2) if risk_per_share else None
            reward         = abs(tgt - entry) if (tgt and not is_closed) else None
            rr             = round(reward / risk_per_share, 1) if (reward and risk_per_share) else None

            # Status
            if is_closed:
                if pnl_inr > 0:  status = f"✅ Closed +{pnl_pct:.1f}%"
                elif pnl_inr < 0: status = f"❌ Closed {pnl_pct:.1f}%"
                else:             status = "✅ Closed Break-even"
            elif sl and tgt:
                if is_buy:
                    if lp <= sl:              status = "🔴 STOP HIT"
                    elif lp >= tgt:           status = "🎯 TARGET HIT"
                    elif lp > entry * 1.005:  status = "🟢 In Profit"
                    elif lp < entry * 0.995:  status = "🟡 Below Entry"
                    else:                      status = "⚪ Near Entry"
                else:
                    if lp >= sl:              status = "🔴 STOP HIT"
                    elif lp <= tgt:           status = "🎯 TARGET HIT"
                    elif lp < entry * 0.995:  status = "🟢 In Profit"
                    elif lp > entry * 1.005:  status = "🟡 Below Entry"
                    else:                      status = "⚪ Near Entry"
            else:
                status = "🟢 Profit" if pnl_inr > 0 else ("🔴 Loss" if pnl_inr < 0 else "⚪ Flat")

            # Signal alignment (suppressed for closed trades)
            bias = sig.get("bias", "N/A")
            if is_closed:
                signal_align = "✅ Closed"
            elif bias == "N/A":
                signal_align = "—"
            elif (is_buy and bias == "BULLISH") or (not is_buy and bias == "BEARISH"):
                signal_align = "✅ Aligned"
            elif (is_buy and bias == "BEARISH"):
                # BUY trade but signal turned bearish — always a risk
                signal_align = "⚠️ Against"
            elif (not is_buy and bias == "BULLISH"):
                # SELL (short) trade but signal is bullish.
                # If the trade is essentially flat (±0.5% of entry) it's likely an exit
                # record rather than an intentional short — don't fire a warning.
                if abs(pnl_pct) <= 0.5:
                    signal_align = "🟡 Neutral"
                elif pnl_inr > 0:
                    # Short is in profit but trend is against — advise caution
                    signal_align = "🟡 Risky (trend vs you)"
                else:
                    # Short is losing AND trend is bullish — real risk
                    signal_align = "⚠️ Against"
            else:
                signal_align = "🟡 Neutral"

            pf_rows.append({
                "Symbol":        t["symbol"],
                "Direction":     t["direction"],
                "Date":          t["entry_date"],
                "Entry (₹)":     entry,
                "Exit (₹)":      exit_price if is_closed else None,
                "Qty":           qty,
                "Current (₹)":   round(exit_price, 2) if is_closed else round(lp, 2),
                "P&L (₹)":       pnl_inr,
                "P&L %":         pnl_pct,
                "Stop (₹)":      sl if not is_closed else None,
                "Target (₹)":    tgt if not is_closed else None,
                "Risk (₹)":      risk_inr,
                "R:R":           rr,
                "Signal":        bias,
                "Alignment":     signal_align,
                "RSI":           sig.get("rsi"),
                "ADX":           sig.get("adx"),
                "Status":        status,
                "Notes":         t.get("notes", ""),
                "_id":           t["id"],
                "_closed":       is_closed,
            })

        df_pf = pd.DataFrame(pf_rows)

        # ── Portfolio summary header ──────────────────────────────────────────────
        open_rows   = [r for r in pf_rows if not r["_closed"]]
        closed_rows = [r for r in pf_rows if r["_closed"]]

        realized_pnl   = sum(r["P&L (₹)"] for r in closed_rows)
        unrealized_pnl = sum(r["P&L (₹)"] for r in open_rows)
        total_pnl      = realized_pnl + unrealized_pnl

        open_invested  = sum(r["Entry (₹)"] * r["Qty"] for r in open_rows)
        open_current   = sum(r["Current (₹)"] * r["Qty"] for r in open_rows)
        total_risk     = sum(r["Risk (₹)"] for r in open_rows if r["Risk (₹)"])

        n_profit       = sum(1 for r in pf_rows if r["P&L (₹)"] > 0)
        n_loss         = sum(1 for r in pf_rows if r["P&L (₹)"] < 0)
        n_stop_hit     = sum(1 for r in open_rows if "STOP HIT" in r["Status"])
        n_target_hit   = sum(1 for r in open_rows if "TARGET HIT" in r["Status"])

        capital        = st.session_state.get("pf_capital", 0.0)
        cap_return_pct = total_pnl / capital * 100 if capital > 0 else None

        pnl_color  = "#1a7a3c" if total_pnl >= 0 else "#c0392b"
        pnl_arrow  = "▲" if total_pnl >= 0 else "▼"
        r_color    = "#1a7a3c" if realized_pnl >= 0 else "#c0392b"
        u_color    = "#1a7a3c" if unrealized_pnl >= 0 else "#c0392b"

        cap_html = ""
        if capital > 0:
            deployed_pct = open_invested / capital * 100
            cap_html = f"""
          <div style="border-left:1px solid #dee2e6; padding-left:24px;">
            <div style="font-size:0.75rem; color:#888;">Capital</div>
            <div style="font-size:1.1rem; font-weight:600;">₹{capital:,.0f}</div>
            <div style="font-size:0.75rem; color:#888;">{deployed_pct:.0f}% deployed</div>
          </div>
          <div style="border-left:1px solid #dee2e6; padding-left:24px;">
            <div style="font-size:0.75rem; color:#888;">Return on Capital</div>
            <div style="font-size:1.3rem; font-weight:700; color:{pnl_color};">{cap_return_pct:+.2f}%</div>
          </div>"""

        st.html(f"""
        <div style="background:#f8f9fa; border-radius:10px; padding:16px 20px; margin-bottom:8px;
                    display:flex; gap:28px; flex-wrap:wrap; align-items:flex-start;">
          <div>
            <div style="font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:.05em;">Total P&L</div>
            <div style="font-size:1.8rem; font-weight:700; color:{pnl_color};">
              {pnl_arrow} ₹{abs(total_pnl):,.2f}
            </div>
          </div>
          <div style="border-left:1px solid #dee2e6; padding-left:24px;">
            <div style="font-size:0.75rem; color:#888;">Realized (closed)</div>
            <div style="font-size:1.2rem; font-weight:700; color:{r_color};">{'+' if realized_pnl >= 0 else ''}₹{realized_pnl:,.2f}</div>
            <div style="font-size:0.72rem; color:#aaa;">{len(closed_rows)} trades closed</div>
          </div>
          <div style="border-left:1px solid #dee2e6; padding-left:24px;">
            <div style="font-size:0.75rem; color:#888;">Unrealized (open)</div>
            <div style="font-size:1.2rem; font-weight:700; color:{u_color};">{'+' if unrealized_pnl >= 0 else ''}₹{unrealized_pnl:,.2f}</div>
            <div style="font-size:0.72rem; color:#aaa;">₹{open_invested:,.0f} → ₹{open_current:,.0f}</div>
          </div>
          <div style="border-left:1px solid #dee2e6; padding-left:24px;">
            <div style="font-size:0.75rem; color:#888;">Open Risk</div>
            <div style="font-size:1.1rem; font-weight:600; color:#c0392b;">₹{total_risk:,.0f}</div>
          </div>
          <div style="border-left:1px solid #dee2e6; padding-left:24px;">
            <div style="font-size:0.75rem; color:#888;">Positions</div>
            <div style="font-size:0.9rem; line-height:1.6;">
              <span style="color:#1a7a3c;">🟢 {n_profit} winning</span> &nbsp;
              <span style="color:#c0392b;">🔴 {n_loss} losing</span>
              {'<br><span style="color:#c0392b; font-weight:700">⚠️ ' + str(n_stop_hit) + ' stop hit</span>' if n_stop_hit else ''}
              {'<br><span style="color:#1a7a3c; font-weight:700">🎯 ' + str(n_target_hit) + ' target hit</span>' if n_target_hit else ''}
            </div>
          </div>
          {cap_html}
        </div>
        """)

        # ── Per-position table ────────────────────────────────────────────────────
        def _pf_style(df):
            def row_style(row):
                s = str(row.get("Status", ""))
                if "STOP HIT"   in s: return ["background-color:#fdecea"] * len(row)
                if "TARGET HIT" in s: return ["background-color:#e8f5ec"] * len(row)
                if "Profit"     in s or "In Profit" in s: return ["background-color:#f0faf4"] * len(row)
                return [""] * len(row)

            def pnl_style(val):
                if not isinstance(val, (int, float)): return ""
                return "color:#1a7a3c; font-weight:700" if val > 0 else ("color:#c0392b; font-weight:700" if val < 0 else "")

            def dir_style(val):
                return "color:#1a7a3c; font-weight:700" if val == "BUY" else "color:#c0392b; font-weight:700"

            def sig_style(val):
                if val == "BULLISH": return "color:#1a7a3c; font-weight:600"
                if val == "BEARISH": return "color:#c0392b; font-weight:600"
                return "color:#b8860b"

            def align_style(val):
                if "Aligned" in str(val): return "color:#1a7a3c; font-weight:600"
                if "Against" in str(val): return "color:#c0392b; font-weight:700"
                return "color:#b8860b"

            def rr_style(val):
                if not isinstance(val, (int, float)): return ""
                if val >= 3.0: return "color:#1a7a3c; font-weight:700"
                if val >= 2.0: return "color:#b8860b"
                return "color:#c0392b"

            cols = df.columns.tolist()
            style = (df.style
                       .apply(row_style, axis=1)
                       .applymap(dir_style,   subset=["Direction"])
                       .applymap(pnl_style,   subset=["P&L (₹)", "P&L %"])
                       .applymap(sig_style,   subset=["Signal"]))
            if "Alignment" in cols: style = style.applymap(align_style, subset=["Alignment"])
            if "R:R"       in cols: style = style.applymap(rr_style,    subset=["R:R"])
            return style.format({
                "Entry (₹)":    "₹{:,.2f}",
                "Exit (₹)":     lambda v: f"₹{v:,.2f}" if v else "—",
                "Current (₹)":  "₹{:,.2f}",
                "P&L (₹)":      lambda v: f"₹{v:+,.2f}",
                "P&L %":        "{:+.2f}%",
                "Stop (₹)":     lambda v: f"₹{v:,.2f}" if v else "—",
                "Target (₹)":   lambda v: f"₹{v:,.2f}" if v else "—",
                "Risk (₹)":     lambda v: f"₹{v:,.0f}" if v else "—",
                "R:R":          lambda v: f"1:{v:.1f}" if v else "—",
                "RSI":          lambda v: f"{v:.1f}" if v else "—",
                "ADX":          lambda v: f"{v:.1f}" if v else "—",
            })

        _display_pf_cols = [
            "Symbol", "Direction", "Date", "Entry (₹)", "Exit (₹)", "Qty", "Current (₹)",
            "P&L (₹)", "P&L %", "Stop (₹)", "Target (₹)", "Risk (₹)", "R:R",
            "Signal", "Alignment", "RSI", "ADX", "Status", "Notes",
        ]
        _dc_pf = [c for c in _display_pf_cols if c in df_pf.columns]
        df_pf_display = df_pf[_dc_pf].copy()

        st.dataframe(
            _pf_style(df_pf_display),
            hide_index=True,
            width="stretch",
            height=min(800, 65 + len(df_pf_display) * 42),
        )

        # ── Close / Remove trades ─────────────────────────────────────────────────
        st.html("<hr class='thin-divider'>")
        _open_rows   = [r for r in pf_rows if not r["_closed"]]
        _closed_rows = [r for r in pf_rows if r["_closed"]]

        cl1, cl2 = st.columns(2)

        with cl1:
            st.markdown("**Close a position (record exit)**")
            _close_options = {
                f"{r['Symbol']} {r['Direction']} @ ₹{r['Entry (₹)']} ({r['Date']})  [{r['_id']}]": r["_id"]
                for r in _open_rows
            }
            if _close_options:
                _close_label = st.selectbox("Select open trade to close",
                                             [""] + list(_close_options.keys()),
                                             key="pf_close_sel")
                _close_exit = st.number_input("Exit Price (₹)", min_value=0.01, value=100.0,
                                              step=0.05, format="%.2f", key="pf_close_price")
                if st.button("✅ Mark as Closed", type="primary", key="pf_close_btn") and _close_label:
                    _cid = _close_options[_close_label]
                    for t in trades:
                        if t["id"] == _cid:
                            t["exit_price"] = round(_close_exit, 2)
                    st.session_state["portfolio_trades"] = trades
                    _save_portfolio(trades)
                    st.rerun()
            else:
                st.caption("No open positions to close.")

        with cl2:
            st.markdown("**Remove a position**")
            _remove_options = {
                f"{r['Symbol']} {r['Direction']} @ ₹{r['Entry (₹)']} ({r['Date']})  [{r['_id']}]": r["_id"]
                for r in pf_rows
            }
            _remove_label = st.selectbox("Select trade to remove", [""] + list(_remove_options.keys()),
                                          key="pf_remove_sel")
            if st.button("🗑 Remove selected trade", key="pf_remove_btn") and _remove_label:
                _rid = _remove_options[_remove_label]
                trades = [t for t in trades if t["id"] != _rid]
                st.session_state["portfolio_trades"] = trades
                _save_portfolio(trades)
                st.rerun()

        # ── Risk alerts (open positions only) ────────────────────────────────────
        _alerts = [r for r in pf_rows if not r["_closed"] and
                   ("STOP HIT" in r["Status"] or "Against" in r["Alignment"] or "Risky" in r["Alignment"])]
        if _alerts:
            st.html("<hr class='thin-divider'>")
            st.markdown("#### ⚠️ Risk Alerts")
            for r in _alerts:
                if "STOP HIT" in r["Status"]:
                    st.error(
                        f"**{r['Symbol']}** {r['Direction']} — Stop loss hit! "
                        f"Current ₹{r['Current (₹)']:,.2f} vs Stop ₹{r['Stop (₹)']:,.2f} · "
                        f"P&L: ₹{r['P&L (₹)']:+,.2f} ({r['P&L %']:+.2f}%)"
                    )
                elif "Against" in r["Alignment"]:
                    st.warning(
                        f"**{r['Symbol']}** {r['Direction']} — Signal turned **{r['Signal']}** "
                        f"(against your {r['Direction']} position). "
                        f"P&L: ₹{r['P&L (₹)']:+,.2f} ({r['P&L %']:+.2f}%) · "
                        f"RSI: {r['RSI']}  ADX: {r['ADX']}"
                    )
                elif "Risky" in r["Alignment"]:
                    st.info(
                        f"**{r['Symbol']}** SELL (short) — market signal is **BULLISH** but trade is in profit. "
                        f"P&L: ₹{r['P&L (₹)']:+,.2f} ({r['P&L %']:+.2f}%) · "
                        f"Consider taking profit — trend is working against you."
                    )

with tab_chart:

    if stock is None:
        st.info("👈  Select an instrument from the sidebar to get started.")

if stock is not None:
  with tab_chart:
    itype = _detect_itype(stock["yf_ticker"])

    with st.spinner(f"Loading {stock['symbol']}…"):
        df_raw  = fetch_ohlcv(stock["yf_ticker"], tf["interval"], tf["period"])
        info    = fetch_info(stock["yf_ticker"])

    # ── F&O fallback: NSE F&O OHLCV is not on Yahoo Finance → use spot index ─
    _fno_fallback_used = False
    if df_raw is None and itype in ("futures", "options"):
        ul = stock.get("underlying", "")
        if ul in _FNO_UNDERLYINGS:
            spot_ticker = _FNO_UNDERLYINGS[ul]["spot_ticker"]
            df_raw = fetch_ohlcv(spot_ticker, tf["interval"], tf["period"])
            if df_raw is not None:
                _fno_fallback_used = True

    if df_raw is None:
        st.error(f"No data for **{stock['symbol']}** ({timeframe}). "
                 "Try a different timeframe or check the symbol.")
        st.stop()

    df_raw = df_raw.dropna(subset=["Close", "High", "Low", "Open"]).copy()

    # Compute indicators on the FULL fetched history so EMA(200), ADX, Supertrend etc.
    # have enough bars to converge accurately.  Trim to period_bars for display AFTER.
    df_enriched_full = _cached_compute_all(df_raw,
                                           use_supertrend=ind_supertrend,
                                           use_donchian=ind_donchian)
    df_enriched = df_enriched_full.tail(period_bars).copy()

    signal_result = _cached_compute_signals(
        df_enriched,
        lookback=period_bars,
        use_candlestick=ind_candle,
        use_volume=ind_volume,
    )
    bias          = signal_result["current_bias"]
    score         = signal_result["strength_score"]
    signals       = signal_result["signals"]
    regime        = signal_result.get("regime", "MIXED")

    sr_levels = _cached_sr_levels(df_enriched, sr_pivot, sr_swing, sr_fib)

    # Always use fast_info for live price display — never rely on cached OHLCV for this
    # For BSE stocks, try NSE ticker first (more liquid, more reliable live feed)
    def _get_fast_info(yf_ticker: str):
        candidates = [yf_ticker]
        if yf_ticker.endswith(".BO"):
            candidates.insert(0, yf_ticker.replace(".BO", ".NS"))
        for t in candidates:
            try:
                fi = yf.Ticker(t).fast_info
                if fi and fi.last_price:
                    return fi
            except Exception:
                pass
        return None

    _fi_live = _get_fast_info(stock["yf_ticker"])

    if _fi_live and _fi_live.last_price:
        last_close = float(_fi_live.last_price)
        prev_close = float(_fi_live.regular_market_previous_close) if _fi_live.regular_market_previous_close else last_close
        day_high   = float(_fi_live.day_high) if _fi_live.day_high else last_close
        day_low    = float(_fi_live.day_low)  if _fi_live.day_low  else last_close
    else:
        last_close = float(df_raw["Close"].iloc[-1])
        prev_close = float(df_raw["Close"].iloc[-2]) if len(df_raw) > 1 else last_close
        day_high   = float(df_raw["High"].max())
        day_low    = float(df_raw["Low"].min())

    change_abs = last_close - prev_close
    change_pct = change_abs / prev_close * 100 if prev_close else 0.0
    volume     = int(df_raw["Volume"].iloc[-1])

    is_up = change_pct >= 0


    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — STOCK HEADER
    # ════════════════════════════════════════════════════════════════════════════

    chg_cls   = "header-up" if is_up else "header-down"
    chg_arrow = "▲" if is_up else "▼"
    pfx       = "" if itype in ("index", "currency") else "₹"

    st.html(f"""
    <div class="header-card">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
        <div>
          <p class="header-symbol">{stock['symbol']}&nbsp;&nbsp;{_itype_badge(itype)}</p>
          <p class="header-name">{stock['name']}  ·  {stock['exchange']}  ·  {stock['cap'].upper()}</p>
        </div>
        <div style="text-align:right;">
          <p class="header-price">{pfx}{last_close:,.2f}</p>
          <p class="{chg_cls}">{chg_arrow} {abs(change_pct):.2f}%&nbsp;&nbsp;{change_abs:+.2f}</p>
          <p style="font-size:0.7rem; color:#aaa; margin:0;">Last traded price · NSE official close may differ slightly</p>
        </div>
      </div>
    </div>
    """)


    # ── Key stats pills ───────────────────────────────────────────────────────────

    def _pill(label: str, value: str) -> str:
        return (f'<div class="stat-pill">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'</div>')

    hi52  = info.get("fiftyTwoWeekHigh")
    lo52  = info.get("fiftyTwoWeekLow")
    mc    = info.get("marketCap")
    pe    = info.get("trailingPE")
    beta  = info.get("beta")
    avgv  = info.get("averageVolume")

    pills = [
        _pill("Day High",  _fmt_price(day_high)),
        _pill("Day Low",   _fmt_price(day_low)),
        _pill("Volume",    _fmt_vol(volume)),
        _pill("52W High",  _fmt_price(hi52)),
        _pill("52W Low",   _fmt_price(lo52)),
        _pill("Mkt Cap",   f"₹{mc/1e7:.0f} Cr" if mc else "—"),
        _pill("P/E",       f"{pe:.1f}" if pe and pe < 5000 else "—"),
        _pill("Beta",      f"{beta:.2f}" if beta else "—"),
        _pill("Avg Vol",   _fmt_vol(int(avgv)) if avgv else "—"),
    ]

    st.html(f'<div class="stats-row">{"".join(pills)}</div>')

    # ── F&O info banner ───────────────────────────────────────────────────────────
    if _fno_fallback_used:
        ul = stock.get("underlying", "")
        if itype == "futures":
            st.info(
                f"**{ul} Futures — {stock.get('exp_label', '')}**  ·  "
                "NSE F&O historical data is not available via Yahoo Finance. "
                f"Showing **{ul} spot index** chart (futures price tracks spot within ~0.1–0.3%)."
            )
        elif itype == "options":
            strike   = stock.get("strike", "")
            opt_type = stock.get("opt_type", "")
            intrinsic = max(0, (last_close - strike) if opt_type == "CE" else (strike - last_close))
            st.info(
                f"**{ul} {strike} {opt_type} — {stock.get('exp_label', '')}**  ·  "
                "Individual option price data is not available via Yahoo Finance. "
                f"Showing **{ul} spot** chart.  "
                f"Intrinsic value ≈ **{intrinsic:,.0f}** pts  ·  "
                f"Spot {last_close:,.0f}  ·  ATM {stock.get('atm', '')}  ·  "
                f"{'ITM' if intrinsic > 0 else 'OTM'}"
            )

    # ── Live NSE option chain (options mode) ──────────────────────────────────────
    if itype == "options":
        ul = stock.get("underlying", "")
        with st.expander("📊  Live NSE Option Chain", expanded=True):
            with st.spinner("Fetching NSE option chain…"):
                oc_df = fetch_nse_option_chain(ul)

            if oc_df is None:
                st.caption("Could not fetch NSE option chain. NSE may be closed or blocking the request.")
            else:
                spot_val   = stock.get("spot", last_close)
                sel_strike = stock.get("strike", 0)
                sel_type   = stock.get("opt_type", "CE")
                step       = _FNO_UNDERLYINGS.get(ul, {}).get("step", 50)
                atm_strike = round(spot_val / step) * step

                # Show strikes ±10 from ATM
                nearby = oc_df[abs(oc_df["Strike"] - atm_strike) <= step * 10]

                # Split CE and PE side by side
                ce_df = nearby[nearby["Type"] == "CE"].drop(columns="Type").set_index("Strike")
                pe_df = nearby[nearby["Type"] == "PE"].drop(columns="Type").set_index("Strike")

                def _oc_style(df, side):
                    def _row(row):
                        strike = row.name
                        styles = [""] * len(row)
                        if strike == atm_strike:
                            styles = ["background-color:#fff8dc; font-weight:600"] * len(row)
                        elif (side == "CE" and strike == sel_strike and sel_type == "CE") or \
                             (side == "PE" and strike == sel_strike and sel_type == "PE"):
                            styles = ["background-color:#e8f0fd; font-weight:600"] * len(row)
                        return styles
                    return df.style.apply(_row, axis=1)

                st.caption(f"Spot: **{spot_val:,.0f}**  ·  ATM: **{atm_strike}**  ·  Selected: **{sel_strike} {sel_type}**  ·  Yellow = ATM  ·  Blue = selected strike")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Calls (CE)**")
                    st.dataframe(_oc_style(ce_df, "CE"), width="stretch", height=380)
                with c2:
                    st.markdown("**Puts (PE)**")
                    st.dataframe(_oc_style(pe_df, "PE"), width="stretch", height=380)


    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — BIAS SUMMARY  (above chart so it's the first thing you see)
    # ════════════════════════════════════════════════════════════════════════════

    BIAS_META = {
        "BULLISH": ("bias-bullish", "bias-bull-txt", "📈", "Most indicators bullish right now"),
        "BEARISH": ("bias-bearish", "bias-bear-txt", "📉", "Most indicators bearish right now"),
        "NEUTRAL": ("bias-neutral", "bias-neut-txt", "➡️", "Mixed signals — no clear direction"),
    }
    panel_cls, txt_cls, icon, desc = BIAS_META[bias]

    # Count signals by type
    buy_count   = sum(1 for s in signals if s["type"] == "BUY")
    sell_count  = sum(1 for s in signals if s["type"] == "SELL")
    watch_count = sum(1 for s in signals if s["type"] == "WATCH")

    badges = ""
    if buy_count:   badges += f'<span class="signal-badge badge-buy">▲ {buy_count} BUY</span>'
    if sell_count:  badges += f'<span class="signal-badge badge-sell">▼ {sell_count} SELL</span>'
    if watch_count: badges += f'<span class="signal-badge badge-watch">◆ {watch_count} WATCH</span>'
    if not badges:  badges  = '<span style="color:#555; font-size:0.82rem;">No active signals</span>'

    adx_val = float(df_enriched["adx"].iloc[-1]) if "adx" in df_enriched.columns and pd.notna(df_enriched["adx"].iloc[-1]) else None
    _regime_meta = {
        "TRENDING": ("🔥 TRENDING", "#1a5aad", "#e8f0fd"),
        "RANGING":  ("〰 RANGING",  "#b8860b", "#fff8e1"),
        "MIXED":    ("⚡ MIXED",    "#555",    "#f5f5f5"),
    }
    regime_lbl, regime_color, regime_bg = _regime_meta.get(regime, _regime_meta["MIXED"])
    adx_note = f"ADX {adx_val:.1f}" if adx_val is not None else "ADX —"
    score_display = f"{score:+.1f}"

    st.html(f"""
    <div class="bias-panel {panel_cls}">
      <div style="font-size:2rem; line-height:1;">{icon}</div>
      <div style="flex:1;">
        <div class="bias-label {txt_cls}">{bias}</div>
        <div class="bias-score">{desc}  ·  Score: {score_display}</div>
        <div style="margin-top:6px;">
          <span style="display:inline-block; background:{regime_bg}; color:{regime_color};
                       border:1px solid {regime_color}; border-radius:4px;
                       padding:2px 8px; font-size:0.75rem; font-weight:600;
                       margin-right:8px;">{regime_lbl} &nbsp;·&nbsp; {adx_note}</span>
        </div>
        <div style="margin-top:8px;">{badges}</div>
      </div>
    </div>
    """)


    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — CHART
    # ════════════════════════════════════════════════════════════════════════════

    indicators_config = {
        "ema":        ind_ema,
        "bb":         ind_bb,
        "vwap":       ind_vwap and timeframe in ("5m", "15m", "1h"),
        "rsi":        ind_rsi,
        "macd":       ind_macd,
        "stoch":      ind_stoch,
        "supertrend": ind_supertrend,
        "donchian":   ind_donchian,
        "atr":        False,
    }

    # Only show the 3 closest S/R levels on the chart to avoid clutter
    chart_sr = sorted(sr_levels, key=lambda x: abs(x["level"] - last_close))[:3]
    fig = _cached_build_chart(df_enriched, indicators_config, chart_sr,
                              _chart_signals_deduped(signals),
                              timeframe=timeframe)
    st.html('<div class="chart-wrapper">')
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,           # mouse-wheel zoom on the chart
            "displayModeBar": True,
            "modeBarButtonsToRemove": [   # keep toolbar clean
                "select2d", "lasso2d", "autoScale2d",
            ],
            "toImageButtonOptions": {
                "filename": stock["symbol"],
                "scale": 2,
            },
        },
    )
    st.html('</div>')


    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3b — OPTIONS STRATEGY + CHAIN
    # ════════════════════════════════════════════════════════════════════════════

    if itype == "options":
        atr_pct_now = (float(df_enriched["atr"].iloc[-1]) / last_close
                       if "atr" in df_enriched.columns and pd.notna(df_enriched["atr"].iloc[-1])
                       else 0.01)
        opt_strat = _options_strategy(bias, regime, atr_pct_now)
        _strat_bg  = {"📈": "#e8f5ec", "📉": "#fdecea", "💥": "#f0e8ff",
                      "💰": "#fff8e1", "🪤": "#e8f0fd", "📊": "#e8f0fd",
                      "🎯": "#fff8e1", "⏳": "#f5f5f5", "🕐": "#f5f5f5"}
        _strat_bd  = {"📈": "#1a7a3c", "📉": "#c0392b", "💥": "#7c3aed",
                      "💰": "#b8860b", "🪤": "#1a5aad", "📊": "#1a5aad",
                      "🎯": "#b8860b", "⏳": "#888",    "🕐": "#888"}
        sb  = _strat_bg.get(opt_strat["emoji"], "#f5f5f5")
        sbd = _strat_bd.get(opt_strat["emoji"], "#aaa")
        adx_disp = f"{float(df_enriched['adx'].iloc[-1]):.1f}" if "adx" in df_enriched.columns else "—"

        st.html(f"""
    <div style="background:{sb}; border:1.5px solid {sbd}; border-radius:10px;
                padding:16px 20px; margin:12px 0;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <span style="font-size:1.6rem;">{opt_strat['emoji']}</span>
        <span style="font-size:1.1rem; font-weight:700; color:{sbd};">
          {opt_strat['strategy']}
        </span>
        <span style="margin-left:auto; font-size:0.75rem; color:#555;">
          Regime: <b>{regime}</b> (ADX {adx_disp}) · Bias: <b>{bias}</b>
        </span>
      </div>
      <div style="font-size:0.85rem; font-family:monospace; background:rgba(0,0,0,0.04);
                  padding:6px 10px; border-radius:4px; margin-bottom:8px;">
        {opt_strat['legs']}
      </div>
      <div style="font-size:0.82rem; color:#333; margin-bottom:4px;">
        <b>Why:</b> {opt_strat['rationale']}
      </div>
      <div style="font-size:0.78rem; color:#888;">
        <b>Risk:</b> {opt_strat['risk']}
      </div>
    </div>
    """)

    if itype in ("stock", "index", "options"):
        with st.expander("🎯  Options Chain", expanded=(itype == "options")):
            _lookup = stock["yf_ticker"]
            opt_data = fetch_options_chain(_lookup)

            if opt_data is None:
                st.caption(
                    "Options data not available for this instrument via yfinance. "
                    "Indian NSE options data requires a direct NSE API session. "
                    "Try entering a US-listed ticker (e.g. AAPL) to see a sample chain."
                )
            else:
                COLS = ["strike", "lastPrice", "bid", "ask", "volume",
                        "openInterest", "impliedVolatility", "inTheMoney"]

                exp_choice = st.selectbox(
                    "Expiry", opt_data["expirations"],
                    label_visibility="collapsed", key="opt_exp",
                )
                if exp_choice != opt_data["expirations"][0]:
                    try:
                        ch = yf.Ticker(_lookup).option_chain(exp_choice)
                        calls, puts = ch.calls, ch.puts
                    except Exception:
                        calls, puts = opt_data["calls"], opt_data["puts"]
                else:
                    calls, puts = opt_data["calls"], opt_data["puts"]

                c_cols = [x for x in COLS if x in calls.columns]
                p_cols = [x for x in COLS if x in puts.columns]

                oc1, oc2 = st.columns(2)
                with oc1:
                    st.markdown("**Calls**")
                    def _call_style(v):
                        return "background-color:#e8f5ec" if v is True else ""
                    style_calls = calls[c_cols].style
                    if "inTheMoney" in c_cols:
                        style_calls = style_calls.applymap(_call_style, subset=["inTheMoney"])
                    st.dataframe(style_calls, hide_index=True, width="stretch", height=280)
                with oc2:
                    st.markdown("**Puts**")
                    def _put_style(v):
                        return "background-color:#fdecea" if v is True else ""
                    style_puts = puts[p_cols].style
                    if "inTheMoney" in p_cols:
                        style_puts = style_puts.applymap(_put_style, subset=["inTheMoney"])
                    st.dataframe(style_puts, hide_index=True, width="stretch", height=280)


    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — INSIGHTS  (signals + S/R side by side)
    # ════════════════════════════════════════════════════════════════════════════

    col_sig, col_sr = st.columns([1, 1], gap="large")

    # ── Trade Setups ──────────────────────────────────────────────────────────────
    with col_sig:
        st.html('<p class="section-header">Trade Setups</p>')

        setups = _build_trade_setups(signals, df_enriched, bias)
        pfx_s  = "" if itype in ("index", "currency") else "₹"

        if not setups:
            no_msg = {
                "BULLISH": "No BUY setups in the last 20 bars.",
                "BEARISH": "No SELL setups in the last 20 bars.",
                "NEUTRAL": "No trade setups in the last 20 bars.",
            }.get(bias, "No trade setups found.")
            st.html(f'<p style="color:#555; font-size:0.85rem;">{no_msg}</p>')
        else:
            for setup in setups:
                is_buy   = setup["type"] == "BUY"
                dir_cls  = "setup-buy" if is_buy else "setup-sell"
                dir_lbl  = "▲ BUY" if is_buy else "▼ SELL"
                dir_tcls = "setup-dir-buy" if is_buy else "setup-dir-sell"
                date_str = (setup["date"].strftime("%d %b %Y")
                            if hasattr(setup["date"], "strftime") else str(setup["date"]))

                sl_color  = "setup-red"  if is_buy else "setup-green"
                tgt_color = "setup-green" if is_buy else "setup-red"

                tags_html = "".join(f'<span class="setup-tag">{ind}</span>'
                                    for ind in setup["indicators"])
                n         = setup["n_signals"]
                conviction = setup.get("conviction", "MEDIUM")
                conv_color = {"HIGH": "#1a7a3c", "MEDIUM": "#b8860b", "LOW": "#888"}.get(conviction, "#888")
                conv_icon  = {"HIGH": "●●●", "MEDIUM": "●●○", "LOW": "●○○"}.get(conviction, "●○○")
                sig_note  = f"{n} signal{'s' if n>1 else ''}" if n else "Bias only"

                st.html(f"""
    <div class="setup-card {dir_cls}">
      <div class="setup-header">
        <span class="{dir_tcls}">{dir_lbl}</span>
        <span class="setup-date">{date_str} &nbsp;·&nbsp; {sig_note}</span>
        <span style="margin-left:auto; font-size:0.72rem; font-weight:600; color:{conv_color};">{conv_icon} {conviction}</span>
      </div>
      <div class="setup-grid">
        <span class="setup-lbl">Entry</span>
        <span class="setup-val">{pfx_s}{setup['entry']:,.2f}</span>
        <span></span>

        <span class="setup-lbl">Stop</span>
        <span class="setup-val {sl_color}">{pfx_s}{setup['sl']:,.2f}</span>
        <span class="setup-chg {sl_color}">{setup['sl_diff']:+.2f} ({setup['sl_pct']:+.1f}%)</span>

        <span class="setup-lbl">Target</span>
        <span class="setup-val {tgt_color}">{pfx_s}{setup['target']:,.2f}</span>
        <span class="setup-chg {tgt_color}">{setup['tgt_diff']:+.2f} ({setup['tgt_pct']:+.1f}%)</span>
      </div>
      <span class="setup-rr">R:R &nbsp; 1 : {setup['rr']}</span>
      <div class="setup-tags">{tags_html}</div>
    </div>""")

            # ── Per-setup how-to guide ─────────────────────────────────────────
            is_buy_g = setup["type"] == "BUY"
            action   = "BUY (go long)" if is_buy_g else "SELL SHORT"
            num_cls  = "" if is_buy_g else "sell-num"
            e_p      = f"{pfx_s}{setup['entry']:,.2f}"
            sl_p     = f"{pfx_s}{setup['sl']:,.2f}"
            tg_p     = f"{pfx_s}{setup['target']:,.2f}"
            sl_risk  = abs(setup['sl_diff'])
            budget   = 100_000
            est_shares = max(1, int(budget * 0.95 / setup["entry"]))
            est_risk   = round(est_shares * sl_risk, 2)
            est_reward = round(est_shares * abs(setup["tgt_diff"]), 2)

            if is_buy_g:
                step3_txt = (f"Set a <span class='guide-red'>stop-loss order at {sl_p}</span> "
                             f"the moment you enter. This limits your loss to ≈ {pfx_s}{est_risk:,.0f}.")
                step4_txt = (f"Set a <span class='guide-green'>limit sell order at {tg_p}</span> "
                             f"to lock in ≈ {pfx_s}{est_reward:,.0f} profit automatically.")
                step5_txt = ("If price moves up strongly but hasn't hit the target yet, "
                             "consider moving your stop-loss up to your entry price to make it a risk-free trade.")
            else:
                step3_txt = (f"Set a <span class='guide-red'>buy-to-cover stop at {sl_p}</span> "
                             f"to cap your loss at ≈ {pfx_s}{est_risk:,.0f}.")
                step4_txt = (f"Set a <span class='guide-green'>limit buy-back order at {tg_p}</span> "
                             f"to bank ≈ {pfx_s}{est_reward:,.0f} profit.")
                step5_txt = ("If price falls fast toward the target, trail your stop down to entry "
                             "so you can't lose money on the trade.")

            st.html(f"""
    <div class="guide-box">
      <div class="guide-step">
        <div class="guide-num {num_cls}">1</div>
        <div class="guide-text">
          <b>Confirm the bias</b> — the panel above says <b>{bias}</b>.
          Only take <b>{setup["type"]}</b> trades when the bias agrees.
          If the bias changes, skip this setup.
        </div>
      </div>
      <div class="guide-step">
        <div class="guide-num {num_cls}">2</div>
        <div class="guide-text">
          <b>Enter the trade</b> — place a <b>{action}</b> order at
          <span class="guide-price">{e_p}</span>.
          With ₹1,00,000 capital you can buy ≈ <b>{est_shares} shares</b> (95% deployed).
        </div>
      </div>
      <div class="guide-step">
        <div class="guide-num {num_cls}">3</div>
        <div class="guide-text">{step3_txt}</div>
      </div>
      <div class="guide-step">
        <div class="guide-num {num_cls}">4</div>
        <div class="guide-text">{step4_txt}</div>
      </div>
      <div class="guide-step">
        <div class="guide-num {num_cls}">5</div>
        <div class="guide-text">{step5_txt}</div>
      </div>
      <div class="guide-rule">
        ⚠️ <b>Golden rule:</b> Never risk more than 2% of your total capital on a single trade.
        At ₹1,00,000 that is ₹2,000 max risk per trade.
        If the stop-loss distance implies more than that, reduce your position size accordingly.
      </div>
    </div>""")

    # ── Support & Resistance ──────────────────────────────────────────────────────
    with col_sr:
        st.html('<p class="section-header">Support &amp; Resistance</p>')

        if not sr_levels:
            st.html('<p style="color:#555; font-size:0.85rem;">Enable S/R options in the sidebar.</p>')
        else:
            TYPE_ROW = {"resistance": "sr-res", "support": "sr-sup", "pivot": "sr-piv"}
            TYPE_COL = {"resistance": "#f85149", "support": "#3fb950", "pivot": "#bc8cff"}

            # Sort: resistances above current price (desc), then supports below (desc)
            above = sorted([l for l in sr_levels if l["level"] > last_close],
                           key=lambda x: x["level"])
            below = sorted([l for l in sr_levels if l["level"] <= last_close],
                           key=lambda x: x["level"], reverse=True)

            for lv in (above + below)[:12]:
                dist     = (lv["level"] - last_close) / last_close * 100
                row_cls  = TYPE_ROW.get(lv["type"], "")
                col      = TYPE_COL.get(lv["type"], "#8b949e")
                dist_col = "#3fb950" if dist > 0 else "#f85149"
                st.html(f"""
    <div class="sr-row {row_cls}">
      <div>
        <div class="sr-label">{lv['label']}</div>
        <div class="sr-price">{pfx}{lv['level']:,.2f}</div>
      </div>
      <div style="text-align:right;">
        <div class="sr-dist" style="color:{dist_col};">{dist:+.2f}%</div>
        <div class="sr-label" style="color:{col}; text-transform:capitalize;">{lv['type']}</div>
      </div>
    </div>""")


    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 5 — DETAILS  (advanced / expandable)
    # ════════════════════════════════════════════════════════════════════════════

    st.html('<hr class="thin-divider">')

    with st.expander("📊  Indicator Values  —  last bar"):
        last     = df_enriched.iloc[-1]
        ind_cols = [
            ("RSI",         "rsi"),
            ("MACD Line",   "macd_line"),
            ("MACD Signal", "macd_signal"),
            ("MACD Hist",   "macd_hist"),
            ("EMA 9",       "ema_9"),
            ("EMA 21",      "ema_21"),
            ("EMA 50",      "ema_50"),
            ("EMA 200",     "ema_200"),
            ("BB Upper",    "bb_upper"),
            ("BB Mid",      "bb_mid"),
            ("BB Lower",    "bb_lower"),
            ("VWAP",        "vwap"),
            ("ATR",         "atr"),
            ("Stoch K",      "stoch_k"),
            ("Stoch D",      "stoch_d"),
            ("Supertrend",   "supertrend"),
            ("ST Direction", "st_direction"),
            ("DC Upper",     "dc_upper"),
            ("DC Mid",       "dc_mid"),
            ("DC Lower",     "dc_lower"),
        ]
        rows = []
        for label, col in ind_cols:
            if col in df_enriched.columns:
                v = last.get(col, np.nan)
                rows.append({"Indicator": label,
                             "Value": f"{v:.4f}" if pd.notna(v) else "—"})

        c1, c2 = st.columns(2)
        half = len(rows) // 2
        with c1:
            st.dataframe(pd.DataFrame(rows[:half]),  hide_index=True, width="stretch")
        with c2:
            st.dataframe(pd.DataFrame(rows[half:]), hide_index=True, width="stretch")

    with st.expander("📋  All Raw Signals  —  every indicator that fired"):
        if signals:
            sig_rows = []
            for s in signals:
                date_str = (s["date"].strftime("%d %b %Y  %H:%M")
                            if hasattr(s["date"], "strftime") else str(s["date"]))
                sig_rows.append({
                    "Type":        s["type"],
                    "Indicator":   s["indicator"],
                    "Description": s["description"],
                    "Strength":    _strength_dots(s["strength"]),
                    "Date":        date_str,
                })
            sig_df = pd.DataFrame(sig_rows)
            def _sig_bg(val):
                if val == "BUY":   return "background-color:#e8f5ec; color:#1a7a3c; font-weight:bold"
                if val == "SELL":  return "background-color:#fdecea; color:#c0392b; font-weight:bold"
                return "background-color:#fff8e1; color:#b8860b; font-weight:bold"
            st.dataframe(sig_df.style.applymap(_sig_bg, subset=["Type"]),
                         hide_index=True, width="stretch")
        else:
            st.info("No signals in the current window.")

    with st.expander("💰  Signal Backtest  —  simulate trades with real capital"):
        st.caption(
            "Simulates every signal where 3+ indicators agree. "
            "Entry = signal-bar close · Stop = 1.5× ATR · Target = 3× ATR · "
            "Max hold = 20 bars · Position = 95% of available capital."
        )

        bt_capital = st.number_input(
            "Starting capital (₹)", value=100_000, step=10_000, min_value=1_000,
            label_visibility="collapsed",
            key="bt_cap",
        )

        if st.button("▶  Run Backtest", type="primary", key="bt_run"):
            bt_trades, bt_final = _run_backtest(signals, df_enriched, bt_capital)

            if not bt_trades:
                st.info("No qualifying signals found in this window to backtest.")
            else:
                total_pnl  = bt_final - bt_capital
                wins       = sum(1 for t in bt_trades if t["P&L"] > 0)
                losses     = len(bt_trades) - wins
                win_rate   = wins / len(bt_trades) * 100
                avg_win    = np.mean([t["P&L"] for t in bt_trades if t["P&L"] > 0] or [0])
                avg_loss   = np.mean([t["P&L"] for t in bt_trades if t["P&L"] <= 0] or [0])
                pnl_color  = "normal" if total_pnl >= 0 else "inverse"

                # ── Summary metrics ────────────────────────────────────────────
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Final Capital",  f"{pfx}{bt_final:,.0f}",
                          f"{total_pnl:+,.0f} ({total_pnl/bt_capital*100:+.1f}%)")
                m2.metric("Total Trades",   len(bt_trades))
                m3.metric("Win Rate",       f"{win_rate:.0f}%",
                          f"{wins}W  {losses}L")
                m4.metric("Avg Win",        f"{pfx}{avg_win:+,.0f}")
                m5.metric("Avg Loss",       f"{pfx}{avg_loss:+,.0f}")

                # ── Equity curve ───────────────────────────────────────────────
                equity = [bt_capital] + [t["Capital After"] for t in bt_trades]
                eq_df  = pd.DataFrame({
                    "Trade": range(len(equity)),
                    "Capital": equity,
                })
                eq_fig = go.Figure(go.Scatter(
                    x=eq_df["Trade"], y=eq_df["Capital"],
                    mode="lines+markers",
                    line=dict(color="#1a5aad", width=2),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(26,90,173,0.07)",
                ))
                eq_fig.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=30),
                    paper_bgcolor="#fff", plot_bgcolor="#fafafa",
                    xaxis=dict(title="Trade #", showgrid=True, gridcolor="#e8e8e8"),
                    yaxis=dict(title="Capital ₹", showgrid=True, gridcolor="#e8e8e8"),
                    showlegend=False,
                )
                st.plotly_chart(eq_fig, width="stretch",
                                config={"displayModeBar": False})

                # ── Trade log table ────────────────────────────────────────────
                bt_df = pd.DataFrame(bt_trades)

                def _bt_row_style(row):
                    color = "#e8f5ec" if row["P&L"] > 0 else "#fdecea"
                    return [f"background-color:{color}"] * len(row)

                def _result_style(val):
                    if val in ("Target Hit", "Trail Stop"): return "color:#1a7a3c; font-weight:600"
                    if val == "Stop Loss":                  return "color:#c0392b; font-weight:600"
                    return "color:#888"

                styled = (
                    bt_df.style
                    .apply(_bt_row_style, axis=1)
                    .applymap(_result_style, subset=["Result"])
                )
                st.dataframe(styled, hide_index=True, width="stretch")

                # ── Download button ────────────────────────────────────────────
                ticker_label = stock.get("label", stock.get("ticker", "backtest"))
                csv_buf = bt_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇  Download Backtest as CSV",
                    data=csv_buf,
                    file_name=f"backtest_{ticker_label}_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download all trades with invested amount, current amount and P&L",
                )


    with st.expander("📐  Full S/R Level Table"):
        if sr_levels:
            sr_rows = []
            for lv in sorted(sr_levels, key=lambda x: x["level"], reverse=True):
                dist = (lv["level"] - last_close) / last_close * 100
                sr_rows.append({
                    "Price ₹":   f"₹{lv['level']:,.2f}",
                    "Type":       lv["type"].capitalize(),
                    "Label":      lv["label"],
                    "Distance":   f"{dist:+.2f}%",
                })
            sr_df = pd.DataFrame(sr_rows)
            def _sr_col(val):
                if val == "Resistance": return "color:#f85149"
                if val == "Support":    return "color:#3fb950"
                return "color:#bc8cff"
            st.dataframe(sr_df.style.applymap(_sr_col, subset=["Type"]),
                         hide_index=True, width="stretch")
        else:
            st.info("No S/R levels computed. Enable options in the sidebar.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADE OPPORTUNITIES SCANNER
# ════════════════════════════════════════════════════════════════════════════

with tab_scan:
    st.markdown("### 🎯 Trade Opportunities")

    # ════════════════════════════════════════════════════════════════════════
    # RISK MODEL — settings
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("⚙️  Risk Model Settings", expanded=True):
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            rm_capital = st.number_input(
                "Account Capital (₹)",
                min_value=10_000, max_value=100_000_000,
                value=int(st.session_state.get("rm_capital", 500_000)),
                step=10_000, format="%d", key="rm_capital",
                help="Total trading capital. Position sizes are calculated from this.",
            )
        with rc2:
            rm_risk_pct = st.number_input(
                "Risk per Trade (%)",
                min_value=0.1, max_value=5.0,
                value=float(st.session_state.get("rm_risk_pct", 1.0)),
                step=0.1, format="%.1f", key="rm_risk_pct",
                help="Max % of capital you are willing to lose on a single trade. "
                     "1% is standard; 2% is aggressive.",
            )
        with rc3:
            rm_max_open = st.number_input(
                "Max Open Positions",
                min_value=1, max_value=30,
                value=int(st.session_state.get("rm_max_open", 8)),
                step=1, key="rm_max_open",
                help="Never hold more than this many positions simultaneously. "
                     "Limits total portfolio risk exposure.",
            )
        with rc4:
            rm_monthly_loss = st.number_input(
                "Monthly Loss Limit (%)",
                min_value=1.0, max_value=20.0,
                value=float(st.session_state.get("rm_monthly_loss", 5.0)),
                step=0.5, format="%.1f", key="rm_monthly_loss",
                help="If you lose this % of capital in the current calendar month, "
                     "stop taking new trades until next month.",
            )

        rc5, rc6, rc7, rc8 = st.columns(4)
        with rc5:
            rm_min_rr = st.number_input(
                "Min R:R Ratio",
                min_value=1.0, max_value=5.0,
                value=float(st.session_state.get("rm_min_rr", 2.0)),
                step=0.5, format="%.1f", key="rm_min_rr",
                help="Only show trades where Target / Stop distance ≥ this ratio. "
                     "2.0 means you make 2× what you risk.",
            )
        with rc6:
            rm_max_sector_pct = st.number_input(
                "Max Sector Concentration (%)",
                min_value=10, max_value=100,
                value=int(st.session_state.get("rm_max_sector_pct", 30)),
                step=5, key="rm_max_sector_pct",
                help="Flag if a single sector makes up more than this % of "
                     "available slots. Prevents over-concentration in one sector.",
            )
        with rc7:
            rm_monthly_loss_taken = st.number_input(
                "Losses Taken This Month (₹)",
                min_value=0, max_value=int(rm_capital),
                value=int(st.session_state.get("rm_monthly_loss_taken", 0)),
                step=1_000, format="%d", key="rm_monthly_loss_taken",
                help="Enter how much you have already lost this month. "
                     "The model will warn or block if you approach the monthly limit.",
            )
        with rc8:
            rm_open_positions = st.number_input(
                "Current Open Positions",
                min_value=0, max_value=30,
                value=int(st.session_state.get("rm_open_positions", 0)),
                step=1, key="rm_open_positions",
                help="How many positions you currently hold. Affects how many new "
                     "trades the model allows.",
            )

    # ── Derived risk limits ───────────────────────────────────────────────────
    _risk_per_trade_inr    = rm_capital * rm_risk_pct / 100
    _monthly_limit_inr     = rm_capital * rm_monthly_loss / 100
    _monthly_remaining     = max(0.0, _monthly_limit_inr - rm_monthly_loss_taken)
    _slots_available       = max(0, rm_max_open - rm_open_positions)
    _portfolio_risk_used   = rm_open_positions * rm_risk_pct
    _portfolio_risk_max    = rm_max_open * rm_risk_pct
    _monthly_loss_pct_used = rm_monthly_loss_taken / rm_capital * 100 if rm_capital else 0

    # ── Portfolio health banner ───────────────────────────────────────────────
    if _monthly_remaining <= 0:
        _health_bg = "#fdecea"; _health_border = "#c0392b"
        _health_msg = "🛑 MONTHLY LOSS LIMIT REACHED — No new trades this month."
        _health_text = "#c0392b"
        _trading_allowed = False
    elif _monthly_loss_pct_used >= rm_monthly_loss * 0.75:
        _health_bg = "#fff8e1"; _health_border = "#b8860b"
        _health_msg = f"⚠️  Approaching monthly limit — ₹{_monthly_remaining:,.0f} remaining before stop."
        _health_text = "#b8860b"
        _trading_allowed = True
    elif _slots_available == 0:
        _health_bg = "#fff8e1"; _health_border = "#b8860b"
        _health_msg = f"⚠️  Max positions reached ({rm_max_open}) — close a position before adding new ones."
        _health_text = "#b8860b"
        _trading_allowed = False
    else:
        _health_bg = "#e8f5ec"; _health_border = "#1a7a3c"
        _health_msg = f"✅  Account healthy — {_slots_available} slot{'s' if _slots_available != 1 else ''} available for new trades."
        _health_text = "#1a7a3c"
        _trading_allowed = True

    st.html(f"""
    <div style="background:{_health_bg}; border-left:5px solid {_health_border};
                border-radius:8px; padding:14px 18px; margin:8px 0 12px 0;">
      <div style="font-size:1rem; font-weight:700; color:{_health_text}; margin-bottom:10px;">
        {_health_msg}
      </div>
      <div style="display:flex; gap:32px; flex-wrap:wrap; font-size:0.82rem; color:#444;">
        <div><b>Risk/Trade</b><br>₹{_risk_per_trade_inr:,.0f} ({rm_risk_pct}%)</div>
        <div><b>Monthly Budget</b><br>₹{_monthly_remaining:,.0f} left of ₹{_monthly_limit_inr:,.0f}
          &nbsp;·&nbsp; {_monthly_loss_pct_used:.1f}% used</div>
        <div><b>Portfolio Risk</b><br>{_portfolio_risk_used:.0f}% in market · max {_portfolio_risk_max:.0f}%</div>
        <div><b>Slots</b><br>{rm_open_positions} open · {_slots_available} available of {rm_max_open}</div>
        <div><b>Min R:R</b><br>1 : {rm_min_rr:.1f} filter active</div>
      </div>
    </div>
    """)

    # ════════════════════════════════════════════════════════════════════════
    # SCAN FILTERS
    # ════════════════════════════════════════════════════════════════════════
    st.caption("Stocks scored on trend strength, direction clarity, volume, and momentum. Sorted by score — highest first.")
    st.html("""<div style="display:flex; gap:16px; margin-bottom:8px; font-size:0.78rem;">
      <span style="background:#e8f5ec; border-radius:4px; padding:2px 8px;">🟢 BUY — bullish setup</span>
      <span style="background:#fdecea; border-radius:4px; padding:2px 8px;">🔴 SELL — bearish setup</span>
      <span style="background:#fff8e1; border-radius:4px; padding:2px 8px;">🟡 WATCH — mixed, monitor</span>
      <span style="background:#f0f4fc; border-radius:4px; padding:2px 8px;">Score: 70+ strong · 50–70 good · 30–50 moderate</span>
    </div>""")

    sf1, sf2, sf3, sf4, sf5 = st.columns([1, 1, 1, 1, 1])
    with sf1:
        scan_signal  = st.multiselect("Signal", ["BUY", "SELL", "WATCH"],
                                      default=["BUY", "SELL", "WATCH"],
                                      key="scan_sig")
    with sf2:
        scan_min_score = st.slider("Min Score", 0, 100, 30, 5,
                                   help="Higher = better setup quality. 30+ = tradeable, 60+ = strong",
                                   key="scan_score")
    with sf3:
        scan_cap = st.multiselect("Cap", ["MIDCAP", "SMALLCAP", "MICROCAP"],
                                  default=["MIDCAP", "SMALLCAP"],
                                  key="scan_cap")
    with sf4:
        scan_min_adx = st.slider("Min ADX", 0, 40, 15, 5,
                                 help="ADX > 25 = trending market, > 20 = mild trend",
                                 key="scan_adx")
    with sf5:
        scan_microcap = st.checkbox("Include Microcap",
                                    value=False,
                                    help="Scans ~400 stocks by default (Midcap+Smallcap NSE). Enable to scan all ~2900 — much slower.",
                                    key="scan_micro")

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔍 Scan Now", type="primary", key="scan_btn"):
            st.cache_data.clear()
    with c2:
        st.caption("Default: ~400 NSE Midcap + Smallcap stocks · ~30 sec. Enable Microcap for full universe (~5 min).")

    with st.spinner("Scanning for trade opportunities…"):
        df_scan = scan_tradeable_stocks(include_microcap=scan_microcap)

    if df_scan.empty:
        st.warning("Scan returned no results. Click **🔍 Scan Now** to retry.")
        st.caption("If this persists, try restarting the app to clear the cache.")
    else:
        # ── Apply scan filters ────────────────────────────────────────────────
        filtered_scan = df_scan[
            (df_scan["Signal"].isin(scan_signal)) &
            (df_scan["Score"]  >= scan_min_score) &
            (df_scan["ADX"]    >= scan_min_adx)
        ].copy()
        if scan_cap:
            filtered_scan = filtered_scan[filtered_scan["Cap"].isin(scan_cap)]

        # ── Apply risk model to each row ─────────────────────────────────────
        def _apply_risk(row):
            price  = row["Price (₹)"]
            sl     = row["Stop (₹)"]
            target = row["Target (₹)"]

            if sl is None or target is None or price <= 0 or sl <= 0 or target <= 0:
                row["Risk (₹/share)"] = None
                row["Qty"]            = None
                row["Trade Value (₹)"]= None
                row["R:R"]            = None
                row["Risk Flag"]      = "—"
                return row

            risk_per_share  = abs(price - sl)
            reward_per_share = abs(target - price)
            rr = round(reward_per_share / risk_per_share, 1) if risk_per_share > 0 else 0

            qty = int(_risk_per_trade_inr / risk_per_share) if risk_per_share > 0 else 0
            # Cap qty so trade value doesn't exceed 20% of capital (concentration limit)
            max_qty_by_concentration = int(rm_capital * 0.20 / price) if price > 0 else qty
            qty = min(qty, max_qty_by_concentration)

            trade_value = round(qty * price, 0)

            # Risk flag
            flags = []
            if rr < rm_min_rr:
                flags.append(f"R:R {rr:.1f} < {rm_min_rr:.1f}")
            if trade_value > rm_capital * 0.20:
                flags.append(">20% capital")
            flag_str = " · ".join(flags) if flags else "✓ OK"

            row["Risk (₹/share)"]  = round(risk_per_share, 2)
            row["Qty"]             = qty if qty > 0 else None
            row["Trade Value (₹)"] = trade_value if qty > 0 else None
            row["R:R"]             = rr
            row["Risk Flag"]       = flag_str
            return row

        filtered_scan = filtered_scan.apply(_apply_risk, axis=1)

        # Remove trades that fail R:R unless user explicitly wants to see them
        rr_failed = filtered_scan[filtered_scan["R:R"].notna() & (filtered_scan["R:R"] < rm_min_rr)]
        filtered_scan = filtered_scan[
            filtered_scan["R:R"].isna() | (filtered_scan["R:R"] >= rm_min_rr)
        ]

        # ── Sector concentration warning ─────────────────────────────────────
        if not filtered_scan.empty and "Cap" in filtered_scan.columns and _slots_available > 0:
            # Use Cap as proxy for sector grouping when sector column is absent
            # (the scan output doesn't include a Sector column, use the name prefix heuristic)
            top_cap = filtered_scan["Cap"].value_counts()
            for cap_name, cap_count in top_cap.items():
                concentration = cap_count / max(_slots_available, 1) * 100
                if concentration > rm_max_sector_pct:
                    st.warning(
                        f"⚠️  **Concentration alert**: {cap_count} of {_slots_available} available "
                        f"slots ({concentration:.0f}%) are **{cap_name}** stocks — above your "
                        f"{rm_max_sector_pct}% limit. Consider spreading across cap segments."
                    )

        # ── Summary metrics ───────────────────────────────────────────────────
        buys   = len(df_scan[df_scan["Signal"] == "BUY"])
        sells  = len(df_scan[df_scan["Signal"] == "SELL"])
        high_q = len(df_scan[df_scan["Score"] >= 70])
        strong = len(df_scan[df_scan["ADX"] >= 30])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("BUY Setups",      buys,   help="Bullish trend + signal")
        m2.metric("SELL Setups",     sells,  help="Bearish trend + signal")
        m3.metric("High Quality (≥70)", high_q, help="Score ≥ 70 — strongest setups")
        m4.metric("Strong Trend (ADX≥30)", strong)
        m5.metric("Failed R:R Filter", len(rr_failed),
                  help=f"Trades removed because R:R < {rm_min_rr:.1f}")

        _filter_parts = [f"**{len(filtered_scan)} pass all filters**", f"{len(df_scan)} scanned"]
        if len(rr_failed): _filter_parts.append(f"{len(rr_failed)} low R:R removed")
        st.markdown(" · ".join(_filter_parts))

        if not _trading_allowed:
            st.error("Trading blocked by risk model. See status above.")
        elif filtered_scan.empty:
            st.info("No stocks match. Try lowering Min Score, Min ADX, or Min R:R ratio.")
        else:
            def _style_scan(df):
                _rr_col  = "R:R"
                _flag_col = "Risk Flag"

                def row_style(row):
                    sig = row["Signal"]
                    if sig == "BUY":   return ["background-color:#e8f5ec"] * len(row)
                    if sig == "SELL":  return ["background-color:#fdecea"] * len(row)
                    return ["background-color:#fff8e1"] * len(row)

                def score_style(val):
                    if not isinstance(val, (int, float)): return ""
                    if val >= 70: return "color:#1a7a3c; font-weight:700"
                    if val >= 50: return "color:#b8860b; font-weight:600"
                    return "color:#888"

                def sig_style(val):
                    if val == "BUY":   return "color:#1a7a3c; font-weight:700"
                    if val == "SELL":  return "color:#c0392b; font-weight:700"
                    return "color:#b8860b; font-weight:600"

                def adx_style(val):
                    if not isinstance(val, float): return ""
                    if val >= 30: return "color:#1a5aad; font-weight:600"
                    return ""

                def chg_style(val):
                    if not isinstance(val, float): return ""
                    return "color:#1a7a3c; font-weight:600" if val > 0 else "color:#c0392b; font-weight:600"

                def rr_style(val):
                    if not isinstance(val, (int, float)): return ""
                    if val >= 3.0: return "color:#1a7a3c; font-weight:700"
                    if val >= 2.0: return "color:#b8860b; font-weight:600"
                    return "color:#c0392b"

                def flag_style(val):
                    if str(val).startswith("✓"): return "color:#1a7a3c"
                    return "color:#c0392b; font-weight:600"

                def type_style(val):
                    return {
                        "Swing Trading":        "color:#1a7a3c; font-weight:700",
                        "Positional Trading":   "color:#1a5aad; font-weight:700",
                        "Momentum Trading":     "color:#7b2d8b; font-weight:700",
                        "Breakout Trading":     "color:#e65100; font-weight:700",
                        "Reversal Trading":     "color:#c0392b; font-weight:700",
                        "Long-term Investing":  "color:#1a7a3c",
                        "Intraday Trading":     "color:#b8860b; font-weight:600",
                        "Scalping":             "color:#888",
                        "Algorithmic Trading":  "color:#aaa",
                    }.get(str(val), "")

                fmt = {
                    "Price (₹)":      "₹{:,.2f}",
                    "Stop (₹)":       lambda v: f"₹{v:,.2f}" if v else "—",
                    "Target (₹)":     lambda v: f"₹{v:,.2f}" if v else "—",
                    "Day Chg %":      "{:+.2f}%",
                    "Vol Ratio":      "{:.1f}×",
                    "ADX":            "{:.1f}",
                    "RSI":            "{:.1f}",
                    "R:R":            lambda v: f"1:{v:.1f}" if v else "—",
                    "Risk (₹/share)": lambda v: f"₹{v:,.2f}" if v else "—",
                    "Qty":            lambda v: f"{int(v):,}" if v else "—",
                    "Trade Value (₹)":lambda v: f"₹{v:,.0f}" if v else "—",
                }
                cols_present = [c for c in df.columns]
                style = (df.style
                           .apply(row_style, axis=1)
                           .applymap(score_style, subset=["Score"])
                           .applymap(sig_style,   subset=["Signal"])
                           .applymap(adx_style,   subset=["ADX"])
                           .applymap(chg_style,   subset=["Day Chg %"]))
                if _rr_col   in cols_present: style = style.applymap(rr_style,   subset=[_rr_col])
                if _flag_col in cols_present: style = style.applymap(flag_style, subset=[_flag_col])
                if "Type"    in cols_present: style = style.applymap(type_style, subset=["Type"])
                return style.format(fmt)

            # Column order
            _col_order = [
                "Symbol", "Name", "Cap", "Type", "Signal", "Score",
                "Price (₹)", "Day Chg %",
                "ADX", "RSI", "Vol Ratio",
                "R:R", "Stop (₹)", "Target (₹)",
                "Risk (₹/share)", "Qty", "Trade Value (₹)", "Risk Flag", "ATR",
            ]
            _display_cols = [c for c in _col_order if c in filtered_scan.columns]
            df_display = filtered_scan[_display_cols].reset_index(drop=True)

            st.dataframe(
                _style_scan(df_display),
                hide_index=True,
                width="stretch",
                height=min(700, 65 + len(df_display) * 38),
            )

            # ── Risk summary for selected trades ─────────────────────────────
            total_trade_val = filtered_scan["Trade Value (₹)"].dropna().sum()
            total_risk_inr  = filtered_scan.apply(
                lambda r: (r["Risk (₹/share)"] or 0) * (r["Qty"] or 0), axis=1
            ).sum()
            n_valid = (filtered_scan["Qty"].notna() & (filtered_scan["Qty"] > 0)).sum()

            if n_valid > 0:
                st.html(f"""
                <div style="background:#f0f4fc; border-radius:8px; padding:12px 18px;
                            margin:8px 0; font-size:0.83rem; color:#333;">
                  <b>If you took ALL {n_valid} filtered trades:</b> &nbsp;
                  Total deployed = <b>₹{total_trade_val:,.0f}</b>
                  ({total_trade_val/rm_capital*100:.1f}% of capital) &nbsp;·&nbsp;
                  Total risk = <b>₹{total_risk_inr:,.0f}</b>
                  ({total_risk_inr/rm_capital*100:.1f}% of capital) &nbsp;·&nbsp;
                  {'<span style="color:#c0392b; font-weight:700">⚠ Exceeds monthly loss budget!</span>'
                    if total_risk_inr > _monthly_remaining else
                   '<span style="color:#1a7a3c; font-weight:700">✓ Within monthly budget</span>'}
                </div>
                """)

            csv_scan = filtered_scan[_display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download as CSV",
                data=csv_scan,
                file_name=f"trade_opportunities_{date.today()}.csv",
                mime="text/csv",
                key="dl_scan",
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET SENTIMENT
# ════════════════════════════════════════════════════════════════════════════

with tab_sentiment:
    st.markdown("### 🌡️ Market Sentiment")
    st.caption("Live overview of Indian market conditions — indices, VIX, sector breadth, and FII/DII flows.")

    if st.button("🔄 Refresh", key="sentiment_refresh"):
        st.cache_data.clear()

    with st.spinner("Fetching market data…"):
        sdata = fetch_market_sentiment()

    sentiment    = sdata["sentiment"]
    score        = sdata["score"]
    indices      = sdata["indices"]
    sectors      = sdata["sectors"]
    fii_dii      = sdata["fii_dii"]
    green_sectors = sdata["green_sectors"]
    total_sectors = sdata["total_sectors"]

    # ── Sentiment gauge ───────────────────────────────────────────────────────
    if sentiment == "BULLISH":
        gauge_color = "#1a7a3c"
        gauge_bg    = "#e8f5ec"
        gauge_icon  = "🟢"
    elif sentiment == "BEARISH":
        gauge_color = "#c0392b"
        gauge_bg    = "#fdecea"
        gauge_icon  = "🔴"
    else:
        gauge_color = "#b8860b"
        gauge_bg    = "#fff8e1"
        gauge_icon  = "🟡"

    st.html(f"""
    <div style="background:{gauge_bg}; border-left:5px solid {gauge_color};
                border-radius:8px; padding:16px 20px; margin-bottom:16px;
                display:flex; align-items:center; gap:20px;">
      <div style="font-size:2.5rem;">{gauge_icon}</div>
      <div>
        <div style="font-size:1.6rem; font-weight:700; color:{gauge_color};">
          {sentiment} MARKET
        </div>
        <div style="font-size:0.9rem; color:#555;">
          Sentiment score: <strong>{score}/100</strong> ·
          {green_sectors}/{total_sectors} sectors green today
        </div>
      </div>
    </div>
    """)

    # ── Key Indices ───────────────────────────────────────────────────────────
    st.markdown("#### 📈 Key Indices")

    INDEX_ORDER = ["Nifty 50", "Sensex", "Bank Nifty", "Nifty IT",
                   "Nifty Midcap 100", "Nifty Smallcap", "India VIX"]

    idx_cols = st.columns(len(INDEX_ORDER))
    for col, name in zip(idx_cols, INDEX_ORDER):
        q = indices.get(name)
        if not q:
            col.metric(name, "—")
            continue
        last  = q["last"]
        chg_p = q["chg_pct"]
        # Format large numbers
        if last > 10000:
            label = f"{last:,.0f}"
        else:
            label = f"{last:,.2f}"
        delta_str = f"{chg_p:+.2f}%"
        col.metric(name, label, delta_str)

    st.html("<hr style='margin:16px 0; border:none; border-top:1px solid #e0e0e0;'>")

    # ── Sector Heatmap ────────────────────────────────────────────────────────
    st.markdown("#### 🗂️ Sector Performance")

    if sectors:
        # Sort by day change
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]["chg_pct"], reverse=True)

        # Build heatmap-style grid (4 per row)
        cols_per_row = 4
        rows = [sorted_sectors[i:i+cols_per_row] for i in range(0, len(sorted_sectors), cols_per_row)]

        for row in rows:
            rcols = st.columns(cols_per_row)
            for col, (sec_name, q) in zip(rcols, row):
                chg = q["chg_pct"]
                wk  = q["week_chg"]
                if chg >= 1.5:
                    bg, fg = "#1a7a3c", "white"
                elif chg >= 0.5:
                    bg, fg = "#4caf50", "white"
                elif chg >= 0:
                    bg, fg = "#c8e6c9", "#1b5e20"
                elif chg >= -0.5:
                    bg, fg = "#ffcdd2", "#b71c1c"
                elif chg >= -1.5:
                    bg, fg = "#e53935", "white"
                else:
                    bg, fg = "#b71c1c", "white"

                col.html(f"""
                <div style="background:{bg}; color:{fg}; border-radius:8px;
                            padding:10px 12px; text-align:center; margin:2px;">
                  <div style="font-weight:700; font-size:0.95rem;">{sec_name}</div>
                  <div style="font-size:1.1rem; font-weight:700;">{chg:+.2f}%</div>
                  <div style="font-size:0.75rem; opacity:0.85;">Week: {wk:+.1f}%</div>
                </div>
                """)
    else:
        st.info("Sector data unavailable.")

    st.html("<hr style='margin:16px 0; border:none; border-top:1px solid #e0e0e0;'>")

    # ── India VIX interpretation ──────────────────────────────────────────────
    vix_data = indices.get("India VIX")
    if vix_data:
        vix_val = vix_data["last"]
        vix_chg = vix_data["chg_pct"]
        if vix_val < 15:
            vix_label, vix_color, vix_msg = "LOW", "#1a7a3c", "Market is calm — good for trend trading"
        elif vix_val < 20:
            vix_label, vix_color, vix_msg = "MODERATE", "#b8860b", "Normal volatility — trade with standard stops"
        elif vix_val < 25:
            vix_label, vix_color, vix_msg = "ELEVATED", "#e65100", "Higher volatility — use wider stops, smaller size"
        else:
            vix_label, vix_color, vix_msg = "HIGH", "#c0392b", "Fear in market — avoid aggressive entries"

        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("India VIX", f"{vix_val:.2f}", f"{vix_chg:+.2f}%")
        with c2:
            st.html(f"""
            <div style="background:#f8f9fa; border-left:4px solid {vix_color};
                        border-radius:6px; padding:12px 16px; margin-top:4px;">
              <span style="color:{vix_color}; font-weight:700;">VIX {vix_label}</span>
              <span style="color:#555; margin-left:8px;">{vix_msg}</span>
            </div>
            """)

    # ── FII / DII Flows ───────────────────────────────────────────────────────
    if fii_dii:
        st.markdown("#### 💰 FII / DII Activity (Last 5 Days)")
        df_fii = pd.DataFrame(fii_dii)

        def _fii_style(df):
            def fii_col(val):
                if not isinstance(val, (int, float)): return ""
                return "color:#1a7a3c; font-weight:600" if val > 0 else "color:#c0392b; font-weight:600"
            return (df.style
                    .applymap(fii_col, subset=["FII Net (₹Cr)", "DII Net (₹Cr)"])
                    .format({"FII Net (₹Cr)": "{:+,.0f}", "DII Net (₹Cr)": "{:+,.0f}"}))

        st.dataframe(_fii_style(df_fii), hide_index=True, width="stretch")
    else:
        st.caption("FII/DII flow data unavailable (NSE API may be rate-limited). Try refreshing.")

    # ── Nifty 50 mini chart ───────────────────────────────────────────────────
    st.html("<hr style='margin:16px 0; border:none; border-top:1px solid #e0e0e0;'>")
    st.markdown("#### 📉 Nifty 50 — 3 Month Chart")

    try:
        nifty_df = yf.download("^NSEI", period="3mo", interval="1d",
                                progress=False, auto_adjust=True)
        if nifty_df is not None and len(nifty_df) > 5:
            nifty_df.index = pd.to_datetime(nifty_df.index)
            fig_n = go.Figure()
            colors = ["#c0392b" if float(nifty_df["Close"].iloc[i]) < float(nifty_df["Open"].iloc[i])
                      else "#1a7a3c" for i in range(len(nifty_df))]
            fig_n.add_trace(go.Candlestick(
                x=nifty_df.index,
                open=nifty_df["Open"].squeeze(),
                high=nifty_df["High"].squeeze(),
                low=nifty_df["Low"].squeeze(),
                close=nifty_df["Close"].squeeze(),
                name="Nifty 50",
                increasing_line_color="#1a7a3c",
                decreasing_line_color="#c0392b",
            ))
            # EMA 20
            ema20 = nifty_df["Close"].squeeze().ewm(span=20, adjust=False).mean()
            fig_n.add_trace(go.Scatter(
                x=nifty_df.index, y=ema20,
                line=dict(color="#1a5aad", width=1.5),
                name="EMA 20",
            ))
            fig_n.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="white",
                plot_bgcolor="#fafafa",
                xaxis=dict(type="date", showgrid=False, rangeslider_visible=False,
                           rangebreaks=[dict(bounds=["sat", "mon"])]),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                legend=dict(orientation="h", y=1.02, x=0),
                showlegend=True,
            )
            st.plotly_chart(fig_n, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.info("Nifty chart unavailable.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PULLBACK SCANNER
# ════════════════════════════════════════════════════════════════════════════

with tab_pullback:
    st.markdown("### 🔁 Pullback Scanner")
    st.caption(
        "Finds NSE/BSE stocks in confirmed uptrends that have pulled back into a buy zone. "
        "Entry rule: EMA9 > EMA21 > EMA50 · Supertrend bullish · Price near EMA9 or EMA21 · RSI 35–62 · Pullback 3–25%."
    )

    st.html("""<div style="display:flex; gap:16px; margin-bottom:10px; font-size:0.78rem; flex-wrap:wrap;">
      <span style="background:#e8f5ec; border-radius:4px; padding:2px 10px;">🟢 Near EMA9 — tightest entry, trend still hot</span>
      <span style="background:#fff8e1; border-radius:4px; padding:2px 10px;">🟡 Near EMA21 — standard pullback entry</span>
      <span style="background:#f0f4fc; border-radius:4px; padding:2px 10px;">🔵 Deep — larger pullback, more confirmation needed</span>
      <span style="background:#f8f9fa; border-radius:4px; padding:2px 10px;">PB Score 70+ = high quality setup</span>
    </div>""")

    # ── Filters ───────────────────────────────────────────────────────────────
    pb_f1, pb_f2, pb_f3, pb_f4 = st.columns(4)
    with pb_f1:
        pb_stage_filter = st.multiselect(
            "Stage",
            ["🟢 Near EMA9", "🟡 Near EMA21", "🔵 Deep"],
            default=["🟢 Near EMA9", "🟡 Near EMA21", "🔵 Deep"],
            key="pb_stage",
        )
    with pb_f2:
        pb_min_score = st.slider("Min PB Score", 0, 100, 40, 5, key="pb_min_score")
    with pb_f3:
        pb_cap_filter = st.multiselect(
            "Cap",
            ["LARGECAP", "MIDCAP", "SMALLCAP"],
            default=[],
            key="pb_cap",
            placeholder="All caps",
        )
    with pb_f4:
        pb_microcap = st.checkbox("Include Microcap", value=False, key="pb_micro",
                                  help="Adds ~2,500 more stocks — much slower scan")

    pb_f5, pb_f6, pb_f7 = st.columns(3)
    with pb_f5:
        pb_min_pullback = st.slider("Min Pullback %", 0, 20, 3, 1, key="pb_min_pb")
    with pb_f6:
        pb_max_pullback = st.slider("Max Pullback %", 5, 25, 20, 1, key="pb_max_pb")
    with pb_f7:
        pb_min_adx = st.slider("Min ADX", 0, 35, 18, 1, key="pb_min_adx")

    pc1, pc2 = st.columns([1, 4])
    with pc1:
        pb_do_scan = st.button("🔍 Scan Now", type="primary", key="pb_scan_btn")
    with pc2:
        st.caption("Scans ~400 NSE Midcap + Smallcap stocks · ~30–60 sec. Enable Microcap for full universe (~5 min).")

    if pb_do_scan:
        scan_pullback_stocks.clear()
        st.session_state["pb_scan_done"] = True

    if not st.session_state.get("pb_scan_done"):
        st.info("👆 Click **Scan Now** to find pullback setups.")
    else:
        with st.spinner("Scanning for pullback setups…"):
            df_pb = scan_pullback_stocks(include_microcap=pb_microcap)

        if df_pb.empty:
            st.warning("No pullback setups found. Try relaxing filters or click Scan Now to refresh.")
        else:
            # Apply filters
            filt_pb = df_pb[
                (df_pb["Stage"].isin(pb_stage_filter)) &
                (df_pb["PB Score"] >= pb_min_score) &
                (df_pb["Pullback %"] >= pb_min_pullback) &
                (df_pb["Pullback %"] <= pb_max_pullback) &
                (df_pb["ADX"] >= pb_min_adx)
            ].copy()
            if pb_cap_filter:
                filt_pb = filt_pb[filt_pb["Cap"].isin(pb_cap_filter)]

            # Drop internal column
            filt_pb = filt_pb.drop(columns=["_yf_ticker"], errors="ignore")

            # ── Summary metrics ───────────────────────────────────────────────
            n_ema9  = len(df_pb[df_pb["Stage"] == "🟢 Near EMA9"])
            n_ema21 = len(df_pb[df_pb["Stage"] == "🟡 Near EMA21"])
            n_deep  = len(df_pb[df_pb["Stage"] == "🔵 Deep"])
            n_hq    = len(df_pb[df_pb["PB Score"] >= 70])

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Setups", len(df_pb))
            m2.metric("Near EMA9",   n_ema9,  help="Tightest entry zone")
            m3.metric("Near EMA21",  n_ema21, help="Standard pullback entry")
            m4.metric("Deep",        n_deep,  help="Larger pullback, needs care")
            m5.metric("High Quality (≥70)", n_hq)

            st.markdown(f"**{len(filt_pb)} setups match filters** · {len(df_pb)} total pullbacks found")

            if filt_pb.empty:
                st.info("No setups match current filters. Try lowering Min PB Score or widening the Pullback % range.")
            else:
                def _pb_style(df):
                    def row_style(row):
                        s = str(row.get("Stage", ""))
                        if "EMA9"  in s: return ["background-color:#e8f5ec"] * len(row)
                        if "EMA21" in s: return ["background-color:#fff8e1"] * len(row)
                        return ["background-color:#f0f4fc"] * len(row)

                    def score_style(val):
                        if not isinstance(val, (int, float)): return ""
                        if val >= 70: return "color:#1a7a3c; font-weight:700"
                        if val >= 50: return "color:#b8860b; font-weight:600"
                        return "color:#888"

                    def pb_pct_style(val):
                        if not isinstance(val, (int, float)): return ""
                        if 5.0 <= val <= 12.0: return "color:#1a7a3c; font-weight:700"   # sweet spot
                        if val <= 5.0:          return "color:#b8860b"                   # shallow
                        if val > 18.0:          return "color:#c0392b"                   # deep/risky
                        return "color:#555"

                    def chg_style(val):
                        if not isinstance(val, float): return ""
                        return "color:#1a7a3c; font-weight:600" if val > 0 else "color:#c0392b; font-weight:600"

                    def adx_style(val):
                        if not isinstance(val, float): return ""
                        return "color:#1a5aad; font-weight:600" if val >= 25 else ""

                    def rsi_style(val):
                        if not isinstance(val, (int, float)): return ""
                        if 40 <= val <= 55: return "color:#1a7a3c; font-weight:600"
                        if val < 38:        return "color:#c0392b"
                        return ""

                    cols = df.columns.tolist()
                    style = (df.style
                               .apply(row_style, axis=1)
                               .applymap(score_style, subset=["PB Score"])
                               .applymap(pb_pct_style, subset=["Pullback %"])
                               .applymap(chg_style,    subset=["Day Chg %"])
                               .applymap(adx_style,    subset=["ADX"]))
                    if "RSI" in cols: style = style.applymap(rsi_style, subset=["RSI"])
                    return style.format({
                        "Price (₹)":  "₹{:,.2f}",
                        "EMA9":       "₹{:,.2f}",
                        "EMA21":      "₹{:,.2f}",
                        "EMA50":      "₹{:,.2f}",
                        "Stop (₹)":   lambda v: f"₹{v:,.2f}" if v else "—",
                        "Target (₹)": lambda v: f"₹{v:,.2f}" if v else "—",
                        "Day Chg %":  "{:+.2f}%",
                        "Pullback %": "{:.1f}%",
                        "Vol Ratio":  "{:.1f}×",
                        "ADX":        "{:.1f}",
                        "RSI":        "{:.1f}",
                    })

                _pb_col_order = [
                    "Symbol", "Name", "Cap", "Stage", "PB Score",
                    "Price (₹)", "Day Chg %", "Pullback %",
                    "EMA9", "EMA21", "EMA50",
                    "RSI", "ADX", "Vol Ratio",
                    "Stop (₹)", "Target (₹)", "ATR",
                ]
                _pb_dc = [c for c in _pb_col_order if c in filt_pb.columns]
                df_pb_disp = filt_pb[_pb_dc].reset_index(drop=True)

                st.dataframe(
                    _pb_style(df_pb_disp),
                    hide_index=True,
                    width="stretch",
                    height=min(700, 65 + len(df_pb_disp) * 38),
                )

                # ── What to look for explanation ─────────────────────────────
                with st.expander("📖 How to read this table", expanded=False):
                    st.markdown("""
**Pullback %** — how far price has dropped from the 20-day high. Sweet spot is **5–12%** (green).
Too shallow (<3%) may just be noise. Too deep (>20%) risks a trend break.

**Stage tells you the entry precision:**
- 🟢 **Near EMA9** — price is just above EMA9 within 1 ATR. Most aggressive entry, trend is very fresh.
- 🟡 **Near EMA21** — price has pulled to the 21-day EMA. Standard textbook pullback entry.
- 🔵 **Deep** — price touched EMA21 in the last 5 bars but has since moved up slightly. Slightly later entry.

**Entry rule:** Buy when today's candle holds above EMA21. Stop goes 1 ATR below EMA21. Target is 3×ATR from entry.

**PB Score** weights: full EMA stack (9>21>50>200), ADX strength, RSI in 40–55 zone, entry tightness, pullback depth.
                    """)

                st.download_button(
                    "⬇ Download CSV",
                    data=filt_pb[_pb_dc].to_csv(index=False).encode("utf-8"),
                    file_name=f"pullback_setups_{date.today()}.csv",
                    mime="text/csv",
                    key="pb_dl",
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — SME STOCKS (NSE Emerge)
# ════════════════════════════════════════════════════════════════════════════

with tab_sme:
    st.markdown("### 🏭 SME Stocks — NSE Emerge")
    st.caption(
        "Scans NSE Emerge (SME) listed stocks for trade setups. "
        "SME stocks are smaller companies — always check liquidity before trading."
    )

    st.html("""
    <div style="background:#fff8e1; border-left:4px solid #f59e0b; border-radius:6px;
                padding:10px 16px; margin-bottom:12px; font-size:0.82rem; color:#555;">
      ⚠️  <b>SME stocks carry higher risk</b> — lower liquidity, wider spreads, and
      limited analyst coverage. Only trade stocks with <b>Liquidity: 🟡 Medium or 🟢 High</b>.
      Always use a stop loss and size positions smaller than mainboard trades.
    </div>
    """)

    # ── Filters ───────────────────────────────────────────────────────────────
    sme_f1, sme_f2, sme_f3, sme_f4 = st.columns(4)
    with sme_f1:
        sme_signal = st.multiselect(
            "Signal", ["BUY", "SELL", "WATCH"],
            default=["BUY", "SELL"],
            key="sme_signal",
        )
    with sme_f2:
        sme_min_score = st.slider("Min Score", 0, 100, 40, 5, key="sme_min_score")
    with sme_f3:
        sme_min_adx = st.slider("Min ADX", 0, 40, 18, 1, key="sme_min_adx")
    with sme_f4:
        sme_liquidity = st.multiselect(
            "Liquidity",
            ["🟢 High", "🟡 Medium", "🔴 Low"],
            default=["🟢 High", "🟡 Medium"],
            key="sme_liq",
            help="Filter by average daily volume. High ≥ 1L shares · Medium ≥ 20K",
        )

    sc1, sc2 = st.columns([1, 4])
    with sc1:
        sme_do_scan = st.button("🔍 Scan Now", type="primary", key="sme_scan_btn")
    with sc2:
        st.caption("Scans NSE Emerge SME universe · ~1–2 min")

    if sme_do_scan:
        scan_sme_stocks.clear()
        _build_sme_ohlcv.clear()
        _fetch_nse_bhavcopy_day.clear()
        st.session_state["sme_scan_done"] = True

    if not st.session_state.get("sme_scan_done"):
        st.info("👆 Click **Scan Now** to scan NSE Emerge SME stocks.")
    else:
        with st.spinner("Scanning SME universe…"):
            df_sme = scan_sme_stocks()

        if df_sme.empty:
            st.warning(
                "No SME stocks found. The NSE bhavcopy data may not be available yet for today — "
                "markets are open Mon–Fri; try again after 4 PM IST, or click **Scan Now** to retry."
            )
        else:
            # Apply filters
            filt_sme = df_sme[
                (df_sme["Signal"].isin(sme_signal)) &
                (df_sme["Score"]  >= sme_min_score) &
                (df_sme["ADX"]    >= sme_min_adx)
            ].copy()
            if sme_liquidity:
                filt_sme = filt_sme[filt_sme["Liquidity"].isin(sme_liquidity)]

            # Summary
            n_buy   = len(df_sme[df_sme["Signal"] == "BUY"])
            n_sell  = len(df_sme[df_sme["Signal"] == "SELL"])
            n_high  = len(df_sme[df_sme["Liquidity"] == "🟢 High"])
            n_med   = len(df_sme[df_sme["Liquidity"] == "🟡 Medium"])
            n_hq    = len(df_sme[df_sme["Score"] >= 70])

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total SME Setups", len(df_sme))
            m2.metric("BUY",  n_buy)
            m3.metric("SELL", n_sell)
            m4.metric("High Liquidity",   n_high)
            m5.metric("High Quality (≥70)", n_hq)

            st.markdown(f"**{len(filt_sme)} match filters** · {len(df_sme)} total SME setups found")

            if filt_sme.empty:
                st.info("No SME stocks match current filters. Try lowering Min Score or including Low liquidity.")
            else:
                def _sme_style(df):
                    def row_style(row):
                        sig = str(row.get("Signal", ""))
                        if sig == "BUY":  return ["background-color:#e8f5ec"] * len(row)
                        if sig == "SELL": return ["background-color:#fdecea"] * len(row)
                        return ["background-color:#fff8e1"] * len(row)

                    def sig_style(val):
                        if val == "BUY":  return "color:#1a7a3c; font-weight:700"
                        if val == "SELL": return "color:#c0392b; font-weight:700"
                        return "color:#b8860b"

                    def score_style(val):
                        if not isinstance(val, (int, float)): return ""
                        if val >= 70: return "color:#1a7a3c; font-weight:700"
                        if val >= 50: return "color:#b8860b; font-weight:600"
                        return "color:#888"

                    def liq_style(val):
                        v = str(val)
                        if "High"   in v: return "color:#1a7a3c; font-weight:700"
                        if "Medium" in v: return "color:#b8860b; font-weight:600"
                        return "color:#c0392b; font-weight:600"

                    def chg_style(val):
                        if not isinstance(val, float): return ""
                        return "color:#1a7a3c; font-weight:600" if val > 0 else "color:#c0392b; font-weight:600"

                    cols = df.columns.tolist()
                    style = (df.style
                               .apply(row_style, axis=1)
                               .applymap(sig_style,   subset=["Signal"])
                               .applymap(score_style, subset=["Score"])
                               .applymap(chg_style,   subset=["Day Chg %"]))
                    if "Liquidity" in cols: style = style.applymap(liq_style, subset=["Liquidity"])
                    return style.format({
                        "Price (₹)":  "₹{:,.2f}",
                        "Stop (₹)":   lambda v: f"₹{v:,.2f}" if v else "—",
                        "Target (₹)": lambda v: f"₹{v:,.2f}" if v else "—",
                        "Day Chg %":  "{:+.2f}%",
                        "Vol Ratio":  "{:.1f}×",
                        "Avg Vol":    lambda v: f"{int(v):,}",
                        "ADX":        "{:.1f}",
                        "RSI":        "{:.1f}",
                    })

                _sme_cols = [
                    "Symbol", "Name", "Signal", "Score",
                    "Price (₹)", "Day Chg %",
                    "Liquidity", "Avg Vol", "Vol Ratio",
                    "ADX", "RSI",
                    "Stop (₹)", "Target (₹)", "ATR",
                ]
                _sme_dc = [c for c in _sme_cols if c in filt_sme.columns]
                df_sme_disp = filt_sme[_sme_dc].reset_index(drop=True)

                st.dataframe(
                    _sme_style(df_sme_disp),
                    hide_index=True,
                    width="stretch",
                    height=min(700, 65 + len(df_sme_disp) * 38),
                )

                with st.expander("📖 SME trading tips", expanded=False):
                    st.markdown("""
**Liquidity matters most in SME stocks:**
- 🟢 **High** (≥ 1 lakh shares/day) — tradeable with normal lot sizes
- 🟡 **Medium** (≥ 20K shares/day) — trade with smaller quantities, expect wider spreads
- 🔴 **Low** (< 20K shares/day) — avoid or use very small sizes; getting out can be difficult

**Key differences from mainboard stocks:**
- Circuit limits are 5% instead of 20% — hits upper/lower circuit more frequently
- No F&O — only cash market, so you need full capital upfront for buying
- Market Maker obligation ends after 3 years of listing — older SME stocks may be illiquid
- Quarterly results may be less audited — always check promoter background

**Recommended position size:** 50% of your normal mainboard trade size for the same setup.
                    """)

                st.download_button(
                    "⬇ Download CSV",
                    data=filt_sme[_sme_dc].to_csv(index=False).encode("utf-8"),
                    file_name=f"sme_setups_{date.today()}.csv",
                    mime="text/csv",
                    key="sme_dl",
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MY PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════


