"""
US Stock Live Trading UI
------------------------
Streamlit app — port 8503
Mirror of the Indian trading UI but for US markets (NYSE / NASDAQ).
"""

import time
import sys
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ── Allow importing from sibling trading_ui folder ────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trading_ui.indicators        import compute_all
from trading_ui.signals           import compute_signals
from trading_ui.charts            import build_chart
from trading_ui.support_resistance import pivot_points, swing_levels, fibonacci_levels
from trading_ui_us.universe        import get_us_universe

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Trading Suite",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  hr.thin-divider { border: none; border-top: 1px solid #e0e0e0; margin: 8px 0; }
  .pill {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600; margin: 2px;
  }
</style>
""", unsafe_allow_html=True)

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES = {
    "5m":  {"interval": "5m",  "period": "5d",  "max_bars": 390},
    "15m": {"interval": "15m", "period": "60d", "max_bars": 390},
    "1h":  {"interval": "60m", "period": "60d", "max_bars": 500},
    "1D":  {"interval": "1d",  "period": "2y",  "max_bars": 504},
    "1W":  {"interval": "1wk", "period": "10y", "max_bars": 520},
}

# ── US Market Indices (always shown in sidebar) ───────────────────────────────
_INDICES = {
    "S&P 500":    {"symbol": "SPX",    "name": "S&P 500 Index",      "yf_ticker": "^GSPC",  "sector": "Index"},
    "NASDAQ 100": {"symbol": "NDX",    "name": "NASDAQ 100 Index",   "yf_ticker": "^NDX",   "sector": "Index"},
    "Dow Jones":  {"symbol": "DJIA",   "name": "Dow Jones Industrial","yf_ticker": "^DJI",   "sector": "Index"},
    "Russell 2000":{"symbol":"RUT",    "name": "Russell 2000",       "yf_ticker": "^RUT",   "sector": "Index"},
    "VIX":        {"symbol": "VIX",    "name": "CBOE Volatility",    "yf_ticker": "^VIX",   "sector": "Index"},
    "10Y Bond":   {"symbol": "TNX",    "name": "10-Year Treasury",   "yf_ticker": "^TNX",   "sector": "Bonds"},
    "Gold":       {"symbol": "GOLD",   "name": "Gold Futures",       "yf_ticker": "GC=F",   "sector": "Commodity"},
    "Crude Oil":  {"symbol": "OIL",    "name": "Crude Oil Futures",  "yf_ticker": "CL=F",   "sector": "Commodity"},
    "USD Index":  {"symbol": "DXY",    "name": "US Dollar Index",    "yf_ticker": "DX-Y.NYB","sector": "Forex"},
}

# ── Cached wrappers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _cached_compute_all(df, use_supertrend=True, use_donchian=True):
    return compute_all(df, use_supertrend=use_supertrend, use_donchian=use_donchian)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_compute_signals(df, lookback=50, use_candlestick=True, use_volume=True):
    return compute_signals(df, lookback=lookback,
                           use_candlestick=use_candlestick, use_volume=use_volume)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_sr_levels(df, do_pivot, do_swing, do_fib):
    levels = []
    if do_pivot: levels += pivot_points(df)
    if do_swing: levels += swing_levels(df)
    if do_fib:   levels += fibonacci_levels(df)
    return levels

@st.cache_data(ttl=300, show_spinner=False)
def _cached_build_chart(df, indicators_config, sr_levels, signals, timeframe="1D"):
    return build_chart(df, indicators_config, sr_levels, signals, timeframe=timeframe)

_OHLCV = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}

def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl = 0
        if len(set(df.columns.get_level_values(0)) & _OHLCV) < \
           len(set(df.columns.get_level_values(1)) & _OHLCV):
            lvl = 1
        df.columns = df.columns.get_level_values(lvl)
        df = df.loc[:, ~df.columns.duplicated()]
    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    return df

_CACHE_VER = "us_v1"

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str,
                _v: str = _CACHE_VER) -> pd.DataFrame | None:
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
def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_options_chain(ticker: str) -> dict | None:
    try:
        tk   = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None
        exp  = exps[0]
        chain = tk.option_chain(exp)
        return {"expiry": exp, "calls": chain.calls, "puts": chain.puts}
    except Exception:
        return None

@st.cache_data(ttl=86400, show_spinner="Loading US stock universe…")
def load_universe() -> dict:
    stocks = get_us_universe()
    result = {}
    for s in stocks:
        label = f"{s.symbol}  —  {s.name}"
        result[label] = {
            "symbol": s.symbol, "name": s.name,
            "sector": s.sector, "index": s.index,
            "yf_ticker": s.symbol,   # US tickers need no suffix
        }
    # Add indices
    for name, meta in _INDICES.items():
        label = f"{meta['symbol']}  —  {meta['name']}"
        result[label] = {
            "symbol": meta["symbol"], "name": meta["name"],
            "sector": meta["sector"], "index": "INDEX",
            "yf_ticker": meta["yf_ticker"],
        }
    return result

@st.cache_data(ttl=1800, show_spinner=False)
def scan_tradeable_stocks() -> pd.DataFrame:
    universe = load_universe()
    stocks   = {k: v for k, v in universe.items() if v["index"] != "INDEX"}
    tickers  = list({v["yf_ticker"] for v in stocks.values()})
    meta     = {v["yf_ticker"]: v for v in stocks.values()}

    BATCH = 100
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            chunk = yf.download(batch, period="6mo", interval="1d",
                                progress=False, auto_adjust=True,
                                group_by="ticker", threads=True)
            frames.append((batch, chunk))
        except Exception:
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
                if df is None: continue
                df.dropna(how="all", inplace=True)
                if len(df) < 30: continue

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

                lc   = float(close.iloc[-1]); pc = float(close.iloc[-2]) if n >= 2 else lc
                rsi  = float(rsi_s.iloc[-1]); adx = float(adx_s.iloc[-1])
                pdi  = float(plus_di.iloc[-1]); mdi = float(minus_di.iloc[-1])
                st   = int(st_dir[-1])
                e9   = float(ema9.iloc[-1]); e21 = float(ema21.iloc[-1])
                e50  = float(ema50.iloc[-1]); e200 = float(ema200.iloc[-1])
                avgv = float(df["Volume"].iloc[-20:].mean())
                volr = round(float(df["Volume"].iloc[-1]) / avgv, 1) if avgv > 0 else 0
                chg  = (lc - pc) / pc * 100

                if any(np.isnan(v) for v in (rsi, adx, e21)): continue

                if e9 > e21 > e50 and lc > e21 and st == 1 and pdi > mdi:
                    direction = "BUY"
                elif e9 < e21 < e50 and lc < e21 and st == -1 and mdi > pdi:
                    direction = "SELL"
                else:
                    direction = "WATCH"

                score = 0
                if adx > 35:   score += 25
                elif adx > 25: score += 18
                elif adx > 20: score += 8
                if direction == "BUY"  and st ==  1: score += 20
                if direction == "SELL" and st == -1: score += 20
                if direction == "WATCH": score += 5
                if e9 > e21 > e50 > e200: score += 20
                elif e9 < e21 < e50 < e200: score += 20
                elif e9 > e21 > e50: score += 12
                elif e9 < e21 < e50: score += 12
                if volr >= 2.0:   score += 15
                elif volr >= 1.5: score += 10
                elif volr >= 1.0: score += 5
                if 45 <= rsi <= 65:   score += 10
                elif 35 <= rsi <= 70: score += 5
                if direction == "BUY"  and lc > e21: score += 10
                if direction == "SELL" and lc < e21: score += 10

                # ── Stock type classification ────────────────────────────────
                _CYCLICAL_SECTORS = {"Energy", "Materials", "Industrials", "Financials", "Consumer Disc"}
                _ticker_sector = meta[ticker]["sector"]
                _full_bull = e9 > e21 > e50 > e200
                _early_bull = e9 > e21 and not (e9 > e21 > e50)
                if volr >= 2.5 or (rsi > 70 and direction == "BUY"):
                    stock_type = "Narrative"
                elif _full_bull and adx > 20 and 35 <= rsi <= 65:
                    stock_type = "Compounder"
                elif _ticker_sector in _CYCLICAL_SECTORS:
                    stock_type = "Cyclical"
                elif _early_bull and direction == "BUY" and 35 <= rsi <= 55:
                    stock_type = "Turnaround"
                else:
                    stock_type = "Cyclical"

                atr_val = float(atr_s.iloc[-1])
                sl      = round(lc - 1.5*atr_val, 2) if direction == "BUY" else (round(lc + 1.5*atr_val, 2) if direction == "SELL" else None)
                target  = round(lc + 3.0*atr_val, 2) if direction == "BUY" else (round(lc - 3.0*atr_val, 2) if direction == "SELL" else None)

                s = meta[ticker]
                rows.append({
                    "Symbol":    s["symbol"],
                    "Name":      s["name"],
                    "Sector":    s["sector"],
                    "Type":      stock_type,
                    "Signal":    direction,
                    "Score":     score,
                    "Price ($)": round(lc, 2),
                    "Day Chg %": round(chg, 2),
                    "ADX":       round(adx, 1),
                    "RSI":       round(rsi, 1),
                    "Vol Ratio": volr,
                    "Stop ($)":  sl,
                    "Target ($)":target,
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_sentiment() -> dict:
    INDEX_TICKERS = {
        "S&P 500":    "^GSPC",
        "NASDAQ 100": "^NDX",
        "Dow Jones":  "^DJI",
        "Russell 2000":"^RUT",
        "VIX":        "^VIX",
    }
    SECTOR_ETFS = {
        "Technology":   "XLK",
        "Financials":   "XLF",
        "Healthcare":   "XLV",
        "Energy":       "XLE",
        "Consumer Disc":"XLY",
        "Consumer Staples":"XLP",
        "Industrials":  "XLI",
        "Materials":    "XLB",
        "Utilities":    "XLU",
        "Real Estate":  "XLRE",
        "Communication":"XLC",
    }

    def _q(ticker):
        try:
            df = yf.download(ticker, period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < 2: return None
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg_p = (last - prev) / prev * 100
            week_chg = (last - float(df["Close"].iloc[0])) / float(df["Close"].iloc[0]) * 100
            return {"last": last, "chg_pct": chg_p, "week_chg": week_chg}
        except Exception:
            return None

    indices = {n: q for n, t in INDEX_TICKERS.items() if (q := _q(t))}
    sectors = {n: q for n, t in SECTOR_ETFS.items()  if (q := _q(t))}

    score = 50
    sp = indices.get("S&P 500", {})
    vix = indices.get("VIX", {})
    if sp:
        if sp["chg_pct"] > 1:    score += 15
        elif sp["chg_pct"] > 0:  score += 7
        elif sp["chg_pct"] < -1: score -= 15
        else:                     score -= 7
        if sp["week_chg"] > 2:   score += 10
        elif sp["week_chg"] < -2: score -= 10
    if vix:
        if vix["last"] < 15:    score += 10
        elif vix["last"] < 20:  score += 5
        elif vix["last"] > 30:  score -= 15
        elif vix["last"] > 20:  score -= 5

    green = sum(1 for v in sectors.values() if v["chg_pct"] > 0)
    total = len(sectors) or 1
    if green / total >= 0.7:    score += 10
    elif green / total <= 0.3:  score -= 10

    score = max(0, min(100, score))
    sentiment = "BULLISH" if score >= 65 else ("BEARISH" if score < 45 else "NEUTRAL")

    # Fear & Greed proxy: VIX + breadth
    fear_greed = max(0, min(100, 100 - (vix["last"] * 2.5 if vix else 50) + (green/total * 30)))

    return {
        "indices": indices, "sectors": sectors,
        "score": score, "sentiment": sentiment,
        "green_sectors": green, "total_sectors": total,
        "fear_greed": round(fear_greed),
        "vix": vix,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🇺🇸 US Trading Suite")
    st.caption("NYSE · NASDAQ · Live Charts · Signals")
    st.html("<hr class='thin-divider'>")

    universe = load_universe()
    labels   = [""] + sorted(universe.keys())

    selected_label = st.selectbox(
        "Search stock / index",
        options=labels,
        format_func=lambda x: "Type to search…" if x == "" else x,
        key="us_stock_select",
    )
    stock = universe.get(selected_label)

    st.html("<hr class='thin-divider'>")

    st.markdown("**Timeframe**")
    timeframe = st.select_slider("Timeframe", options=list(TIMEFRAMES.keys()),
                                 value="1D", label_visibility="collapsed")
    tf = TIMEFRAMES[timeframe]
    period_bars = st.slider("Bars", 50, tf["max_bars"], min(200, tf["max_bars"]), 10)
    st.caption(f"{period_bars} bars · {timeframe} chart")

    st.html("<hr class='thin-divider'>")
    st.markdown("**Indicators**")
    ind_supertrend = st.checkbox("Supertrend",     value=True)
    ind_donchian   = st.checkbox("Donchian",        value=False)
    ind_candle     = st.checkbox("Candle Patterns", value=True)
    ind_volume     = st.checkbox("Volume Signals",  value=True)

    st.html("<hr class='thin-divider'>")
    st.markdown("**Support / Resistance**")
    sr_pivot = st.checkbox("Pivot Points", value=True)
    sr_swing = st.checkbox("Swing Levels", value=True)
    sr_fib   = st.checkbox("Fibonacci",    value=timeframe in ("1D", "1W"))

    st.html("<hr class='thin-divider'>")
    if st.button("🔄  Refresh live data", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# ── Guard ─────────────────────────────────────────────────────────────────────
if stock is None:
    st.info("👈  Select a stock or index from the sidebar to get started.")
    st.markdown("### 📊 Market Overview")

    if st.button("Load Market Overview", key="us_load_mkt"):
        st.session_state["us_mkt_loaded"] = True

    if st.session_state.get("us_mkt_loaded"):
        with st.spinner("Loading market data…"):
            sdata = fetch_market_sentiment()
        if sdata["indices"]:
            cols = st.columns(len(sdata["indices"]))
            for col, (name, q) in zip(cols, sdata["indices"].items()):
                val = f"{q['last']:,.0f}" if q["last"] > 1000 else f"{q['last']:.2f}"
                col.metric(name, val, f"{q['chg_pct']:+.2f}%")
    st.stop()


# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {stock['symbol']}…"):
    df_raw = fetch_ohlcv(stock["yf_ticker"], tf["interval"], tf["period"])
    info   = fetch_info(stock["yf_ticker"])

if df_raw is None:
    st.error(f"No data for **{stock['symbol']}** ({timeframe}). Try a different timeframe.")
    st.stop()

df_raw = df_raw.dropna(subset=["Close","High","Low","Open"]).copy()

# Compute on full history so EMA(200), ADX, Supertrend have enough bars to converge
df_enriched_full = _cached_compute_all(df_raw, use_supertrend=ind_supertrend, use_donchian=ind_donchian)
df_enriched      = df_enriched_full.tail(period_bars).copy()

signal_result = _cached_compute_signals(df_enriched, lookback=period_bars,
                                        use_candlestick=ind_candle, use_volume=ind_volume)
bias    = signal_result["current_bias"]
score   = signal_result["strength_score"]
signals = signal_result["signals"]
regime  = signal_result.get("regime", "MIXED")

sr_levels = _cached_sr_levels(df_enriched, sr_pivot, sr_swing, sr_fib)

# Live price from fast_info
try:
    fi = yf.Ticker(stock["yf_ticker"]).fast_info
    last_close = float(fi.last_price)
    prev_close = float(fi.regular_market_previous_close) if fi.regular_market_previous_close else last_close
    day_high   = float(fi.day_high) if fi.day_high else last_close
    day_low    = float(fi.day_low)  if fi.day_low  else last_close
except Exception:
    last_close = float(df_raw["Close"].iloc[-1])
    prev_close = float(df_raw["Close"].iloc[-2]) if len(df_raw) > 1 else last_close
    day_high   = float(df_raw["High"].max())
    day_low    = float(df_raw["Low"].min())

change_abs = last_close - prev_close
change_pct = change_abs / prev_close * 100 if prev_close else 0.0
volume     = int(df_raw["Volume"].iloc[-1])
is_up      = change_pct >= 0


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_chart, tab_scan, tab_sentiment = st.tabs(
    ["📊  Live Chart", "🎯  Trade Opportunities", "🌡️  Market Sentiment"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE CHART
# ════════════════════════════════════════════════════════════════════════════

with tab_chart:

    # ── Header ────────────────────────────────────────────────────────────────
    price_color = "#1a7a3c" if is_up else "#c0392b"
    arrow       = "▲" if is_up else "▼"
    chg_sign    = "+" if is_up else ""

    st.html(f"""
    <div style="display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:4px;">
      <span style="font-size:1.6rem; font-weight:700;">{stock['symbol']}</span>
      <span style="font-size:0.9rem; color:#666;">{stock['name']} · {stock.get('sector','')}</span>
    </div>
    <div style="display:flex; align-items:baseline; gap:10px; margin-bottom:8px; flex-wrap:wrap;">
      <span style="font-size:2.2rem; font-weight:700; color:{price_color};">${last_close:,.2f}</span>
      <span style="font-size:1.1rem; color:{price_color};">{arrow} {chg_sign}{change_pct:.2f}%  {chg_sign}${change_abs:,.2f}</span>
    </div>
    """)

    # ── Bias badge ────────────────────────────────────────────────────────────
    BIAS_STYLE = {
        "BULLISH": ("background:#e8f5ec; color:#1a7a3c; border:1px solid #1a7a3c;", "▲ BULLISH"),
        "BEARISH": ("background:#fdecea; color:#c0392b; border:1px solid #c0392b;", "▼ BEARISH"),
        "NEUTRAL": ("background:#fff8e1; color:#b8860b; border:1px solid #b8860b;", "◆ NEUTRAL"),
    }
    bstyle, blabel = BIAS_STYLE.get(bias, BIAS_STYLE["NEUTRAL"])
    STRENGTH_DOTS  = {1: "●○○ LOW", 2: "●●○ MEDIUM", 3: "●●● HIGH"}
    sdots = STRENGTH_DOTS.get(min(3, max(1, round(score / 33.4))), "●○○")

    st.html(f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px;">
      <span style="padding:4px 14px; border-radius:16px; font-weight:700; font-size:0.9rem; {bstyle}">{blabel}</span>
      <span style="padding:4px 14px; border-radius:16px; background:#f0f4fc; font-size:0.85rem;">{sdots}  Score: {score:.0f}/100</span>
      <span style="padding:4px 14px; border-radius:16px; background:#f5f5f5; font-size:0.85rem;">Regime: {regime}</span>
    </div>
    """)

    # ── Key stats pills ───────────────────────────────────────────────────────
    hi52  = info.get("fiftyTwoWeekHigh")
    lo52  = info.get("fiftyTwoWeekLow")
    mc    = info.get("marketCap")
    pe    = info.get("trailingPE")
    beta  = info.get("beta")
    avgv  = info.get("averageVolume")
    div_y = info.get("dividendYield")

    def _pill(label, val):
        if val is None: return ""
        return f'<span class="pill" style="background:#f0f4fc; color:#333;"><b>{label}</b> {val}</span>'

    def _fmt(v, decimals=2):
        if v is None: return "—"
        return f"${v:,.{decimals}f}"

    def _fmt_mc(v):
        if v is None: return "—"
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9:  return f"${v/1e9:.1f}B"
        if v >= 1e6:  return f"${v/1e6:.1f}M"
        return f"${v:,.0f}"

    pills_html = "".join([
        _pill("Day High",  _fmt(day_high)),
        _pill("Day Low",   _fmt(day_low)),
        _pill("52W High",  _fmt(hi52)),
        _pill("52W Low",   _fmt(lo52)),
        _pill("Mkt Cap",   _fmt_mc(mc)),
        _pill("P/E",       f"{pe:.1f}x" if pe else None),
        _pill("Beta",      f"{beta:.2f}" if beta else None),
        _pill("Div Yield", f"{div_y*100:.2f}%" if div_y else None),
        _pill("Avg Vol",   f"{avgv:,.0f}" if avgv else None),
    ])
    st.html(f'<div style="margin-bottom:12px; line-height:2;">{pills_html}</div>')

    # ── Chart ─────────────────────────────────────────────────────────────────
    ind_cfg = {
        "show_ema":        True,
        "show_supertrend": ind_supertrend,
        "show_donchian":   ind_donchian,
        "show_volume":     True,
        "show_macd":       True,
        "show_rsi":        True,
        "show_bb":         True,
        "show_stoch":      False,
        "show_adx":        True,
    }
    fig = _cached_build_chart(df_enriched, ind_cfg, sr_levels, signals, timeframe=timeframe)
    st.plotly_chart(fig, config={"displayModeBar": True,
                                 "modeBarButtonsToRemove": ["select2d","lasso2d"],
                                 "scrollZoom": True}, width="stretch")

    # ── Recent signals ────────────────────────────────────────────────────────
    if signals:
        st.markdown("#### 🔔 Recent Signals")
        sig_cols = st.columns(min(4, len(signals)))
        for col, sig in zip(sig_cols, signals[:4]):
            d = sig.get("date", "")
            t = sig.get("type", "")
            p = sig.get("pattern", sig.get("reason", ""))
            bg = "#e8f5ec" if "BUY" in t.upper() else "#fdecea" if "SELL" in t.upper() else "#fff8e1"
            col.html(f"""<div style="background:{bg}; border-radius:8px; padding:8px 12px; font-size:0.82rem;">
              <b>{t}</b><br><span style="color:#666;">{p}</span><br>
              <span style="color:#999; font-size:0.75rem;">{d}</span></div>""")

    # ── Trade setup ───────────────────────────────────────────────────────────
    st.html("<hr class='thin-divider'>")
    st.markdown("#### 📋 Trade Setup")

    _atr = float(df_enriched["atr"].iloc[-1]) if "atr" in df_enriched.columns else last_close * 0.02
    if bias == "BULLISH":
        _entry  = last_close
        _sl     = round(last_close - 1.5 * _atr, 2)
        _target = round(last_close + 3.0 * _atr, 2)
        _rr     = round((_target - _entry) / (_entry - _sl), 1) if _entry != _sl else 0
        _sl_pct = round((_sl - _entry) / _entry * 100, 1)
        _tgt_pct = round((_target - _entry) / _entry * 100, 1)
        _dir    = "LONG"
        _bg     = "#e8f5ec"; _cl = "#1a7a3c"
    elif bias == "BEARISH":
        _entry  = last_close
        _sl     = round(last_close + 1.5 * _atr, 2)
        _target = round(last_close - 3.0 * _atr, 2)
        _rr     = round((_entry - _target) / (_sl - _entry), 1) if _entry != _sl else 0
        _sl_pct = round((_sl - _entry) / _entry * 100, 1)
        _tgt_pct = round((_target - _entry) / _entry * 100, 1)
        _dir    = "SHORT"
        _bg     = "#fdecea"; _cl = "#c0392b"
    else:
        _dir = None

    if _dir:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Direction",   _dir)
        c2.metric("Entry",       f"${_entry:,.2f}")
        c3.metric("Stop Loss",   f"${_sl:,.2f}", f"{_sl_pct:+.1f}%")
        c4.metric("Target",      f"${_target:,.2f}", f"{_tgt_pct:+.1f}%")
        st.caption(f"R:R  1 : {_rr}  ·  ATR = ${_atr:.2f}  ·  Based on current price + 1.5/3× ATR")
    else:
        st.info("No clear setup — bias is NEUTRAL. Wait for a directional signal.")

    # ── Options chain ─────────────────────────────────────────────────────────
    st.html("<hr class='thin-divider'>")
    with st.expander("🔗 Options Chain", expanded=False):
        opt = fetch_options_chain(stock["yf_ticker"])
        if opt:
            st.caption(f"Expiry: **{opt['expiry']}** · Nearest strikes shown")
            calls = opt["calls"]
            puts  = opt["puts"]

            # Filter near ATM
            atm = last_close
            calls_atm = calls[(calls["strike"] >= atm * 0.95) & (calls["strike"] <= atm * 1.05)].head(10)
            puts_atm  = puts [(puts["strike"]  >= atm * 0.95) & (puts["strike"]  <= atm * 1.05)].head(10)

            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown("**Calls**")
                if not calls_atm.empty:
                    show_cols = [c for c in ["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"] if c in calls_atm.columns]
                    st.dataframe(calls_atm[show_cols].reset_index(drop=True),
                                 hide_index=True, width="stretch", height=280)
            with oc2:
                st.markdown("**Puts**")
                if not puts_atm.empty:
                    show_cols = [c for c in ["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"] if c in puts_atm.columns]
                    st.dataframe(puts_atm[show_cols].reset_index(drop=True),
                                 hide_index=True, width="stretch", height=280)
        else:
            st.info("Options data not available for this symbol.")

    # ── S/R levels table ──────────────────────────────────────────────────────
    if sr_levels:
        st.html("<hr class='thin-divider'>")
        with st.expander("📏 Support & Resistance Levels", expanded=False):
            sr_df = pd.DataFrame(sr_levels)
            if "price" in sr_df.columns:
                sr_df = sr_df.sort_values("price", ascending=False)
                sr_df["price"] = sr_df["price"].apply(lambda v: f"${v:,.2f}")
            def _sr_col(val):
                v = str(val).upper()
                if "RESIST" in v: return "color:#c0392b; font-weight:600"
                if "SUPPORT" in v: return "color:#1a7a3c; font-weight:600"
                return "color:#bc8cff"
            st.dataframe(sr_df.style.applymap(_sr_col, subset=["Type"]) if "Type" in sr_df.columns else sr_df,
                         hide_index=True, width="stretch")


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
            us_rm_capital = st.number_input(
                "Account Capital ($)",
                min_value=1_000, max_value=100_000_000,
                value=int(st.session_state.get("us_rm_capital", 50_000)),
                step=1_000, format="%d", key="us_rm_capital",
            )
        with rc2:
            us_rm_risk_pct = st.number_input(
                "Risk per Trade (%)",
                min_value=0.1, max_value=5.0,
                value=float(st.session_state.get("us_rm_risk_pct", 1.0)),
                step=0.1, format="%.1f", key="us_rm_risk_pct",
                help="Max % of capital to risk on a single trade.",
            )
        with rc3:
            us_rm_max_open = st.number_input(
                "Max Open Positions",
                min_value=1, max_value=30,
                value=int(st.session_state.get("us_rm_max_open", 8)),
                step=1, key="us_rm_max_open",
            )
        with rc4:
            us_rm_monthly_loss = st.number_input(
                "Monthly Loss Limit (%)",
                min_value=1.0, max_value=20.0,
                value=float(st.session_state.get("us_rm_monthly_loss", 5.0)),
                step=0.5, format="%.1f", key="us_rm_monthly_loss",
            )

        rc5, rc6, rc7, rc8 = st.columns(4)
        with rc5:
            us_rm_min_rr = st.number_input(
                "Min R:R Ratio",
                min_value=1.0, max_value=5.0,
                value=float(st.session_state.get("us_rm_min_rr", 2.0)),
                step=0.5, format="%.1f", key="us_rm_min_rr",
            )
        with rc6:
            us_rm_max_sector_pct = st.number_input(
                "Max Sector Concentration (%)",
                min_value=10, max_value=100,
                value=int(st.session_state.get("us_rm_max_sector_pct", 30)),
                step=5, key="us_rm_max_sector_pct",
            )
        with rc7:
            us_rm_monthly_loss_taken = st.number_input(
                "Losses Taken This Month ($)",
                min_value=0, max_value=int(us_rm_capital),
                value=int(st.session_state.get("us_rm_monthly_loss_taken", 0)),
                step=100, format="%d", key="us_rm_monthly_loss_taken",
            )
        with rc8:
            us_rm_open_positions = st.number_input(
                "Current Open Positions",
                min_value=0, max_value=30,
                value=int(st.session_state.get("us_rm_open_positions", 0)),
                step=1, key="us_rm_open_positions",
            )

    # ── Derived limits ────────────────────────────────────────────────────────
    _us_risk_per_trade   = us_rm_capital * us_rm_risk_pct / 100
    _us_monthly_limit    = us_rm_capital * us_rm_monthly_loss / 100
    _us_monthly_rem      = max(0.0, _us_monthly_limit - us_rm_monthly_loss_taken)
    _us_slots            = max(0, us_rm_max_open - us_rm_open_positions)
    _us_port_risk_used   = us_rm_open_positions * us_rm_risk_pct
    _us_port_risk_max    = us_rm_max_open * us_rm_risk_pct
    _us_monthly_used_pct = us_rm_monthly_loss_taken / us_rm_capital * 100 if us_rm_capital else 0

    if _us_monthly_rem <= 0:
        _us_hbg, _us_hborder, _us_htxt = "#fdecea", "#c0392b", "#c0392b"
        _us_hmsg = "🛑 MONTHLY LOSS LIMIT REACHED — No new trades this month."
        _us_trading_ok = False
    elif _us_monthly_used_pct >= us_rm_monthly_loss * 0.75:
        _us_hbg, _us_hborder, _us_htxt = "#fff8e1", "#b8860b", "#b8860b"
        _us_hmsg = f"⚠️  Approaching monthly limit — ${_us_monthly_rem:,.0f} remaining."
        _us_trading_ok = True
    elif _us_slots == 0:
        _us_hbg, _us_hborder, _us_htxt = "#fff8e1", "#b8860b", "#b8860b"
        _us_hmsg = f"⚠️  Max positions reached ({us_rm_max_open}) — close a position first."
        _us_trading_ok = False
    else:
        _us_hbg, _us_hborder, _us_htxt = "#e8f5ec", "#1a7a3c", "#1a7a3c"
        _us_hmsg = f"✅  Account healthy — {_us_slots} slot{'s' if _us_slots != 1 else ''} available."
        _us_trading_ok = True

    st.html(f"""
    <div style="background:{_us_hbg}; border-left:5px solid {_us_hborder};
                border-radius:8px; padding:14px 18px; margin:8px 0 12px 0;">
      <div style="font-size:1rem; font-weight:700; color:{_us_htxt}; margin-bottom:10px;">
        {_us_hmsg}
      </div>
      <div style="display:flex; gap:32px; flex-wrap:wrap; font-size:0.82rem; color:#444;">
        <div><b>Risk/Trade</b><br>${_us_risk_per_trade:,.0f} ({us_rm_risk_pct}%)</div>
        <div><b>Monthly Budget</b><br>${_us_monthly_rem:,.0f} left of ${_us_monthly_limit:,.0f}
          &nbsp;·&nbsp; {_us_monthly_used_pct:.1f}% used</div>
        <div><b>Portfolio Risk</b><br>{_us_port_risk_used:.0f}% in market · max {_us_port_risk_max:.0f}%</div>
        <div><b>Slots</b><br>{us_rm_open_positions} open · {_us_slots} available of {us_rm_max_open}</div>
        <div><b>Min R:R</b><br>1 : {us_rm_min_rr:.1f} filter active</div>
      </div>
    </div>
    """)

    # ════════════════════════════════════════════════════════════════════════
    # SCAN FILTERS
    # ════════════════════════════════════════════════════════════════════════
    st.caption("S&P 500 + NASDAQ 100 stocks scored on trend strength, volume, and momentum.")
    st.html("""<div style="display:flex; gap:16px; margin-bottom:8px; font-size:0.78rem;">
      <span style="background:#e8f5ec; border-radius:4px; padding:2px 8px;">🟢 BUY — bullish setup</span>
      <span style="background:#fdecea; border-radius:4px; padding:2px 8px;">🔴 SELL — bearish setup</span>
      <span style="background:#fff8e1; border-radius:4px; padding:2px 8px;">🟡 WATCH — mixed</span>
      <span style="background:#f0f4fc; border-radius:4px; padding:2px 8px;">Score 70+ = strong</span>
    </div>""")

    sf1, sf2, sf3, sf4 = st.columns([1,1,1,1])
    with sf1:
        scan_signal = st.multiselect("Signal", ["BUY","SELL","WATCH"],
                                     default=["BUY","SELL","WATCH"], key="us_scan_sig")
    with sf2:
        scan_min_score = st.slider("Min Score", 0, 100, 30, 5, key="us_scan_score")
    with sf3:
        scan_sector = st.multiselect("Sector",
            ["Technology","Financials","Healthcare","Energy","Consumer Disc",
             "Consumer Staples","Industrials","Materials","Utilities","Real Estate","Communication"],
            default=[], key="us_scan_sector",
            placeholder="All sectors")
    with sf4:
        scan_min_adx = st.slider("Min ADX", 0, 40, 15, 5, key="us_scan_adx")

    c1, c2 = st.columns([1,4])
    with c1:
        do_scan = st.button("🔍 Scan Now", type="primary", key="us_scan_btn")
    with c2:
        st.caption("Scans ~500 S&P 500 + NASDAQ 100 stocks · ~1-2 min")

    if do_scan:
        scan_tradeable_stocks.clear()
        st.session_state["us_scan_done"] = True

    if not st.session_state.get("us_scan_done"):
        st.info("👆 Click **Scan Now** to find trade setups across S&P 500 + NASDAQ 100 stocks.")
        st.stop()

    with st.spinner("Scanning US market…"):
        df_scan = scan_tradeable_stocks()

    if df_scan.empty:
        st.warning("Scan returned no results. Click **🔍 Scan Now** to retry.")
    else:
        filtered = df_scan[
            (df_scan["Signal"].isin(scan_signal)) &
            (df_scan["Score"]  >= scan_min_score) &
            (df_scan["ADX"]    >= scan_min_adx)
        ].copy()
        if scan_sector:
            filtered = filtered[filtered["Sector"].isin(scan_sector)]

        # ── Apply risk model ──────────────────────────────────────────────────
        def _us_apply_risk(row):
            price  = row["Price ($)"]
            sl     = row["Stop ($)"]
            target = row["Target ($)"]
            if sl is None or target is None or price <= 0 or sl <= 0 or target <= 0:
                row["R:R"] = None; row["Qty"] = None
                row["Trade Value ($)"] = None; row["Risk Flag"] = "—"
                return row
            rps  = abs(price - sl)
            rwps = abs(target - price)
            rr   = round(rwps / rps, 1) if rps > 0 else 0
            qty  = int(_us_risk_per_trade / rps) if rps > 0 else 0
            qty  = min(qty, int(us_rm_capital * 0.20 / price) if price > 0 else qty)
            tv   = round(qty * price, 0)
            flags = []
            if rr < us_rm_min_rr:   flags.append(f"R:R {rr:.1f} < {us_rm_min_rr:.1f}")
            if tv > us_rm_capital * 0.20: flags.append(">20% capital")
            row["R:R"]            = rr
            row["Qty"]            = qty if qty > 0 else None
            row["Trade Value ($)"]= tv if qty > 0 else None
            row["Risk Flag"]      = " · ".join(flags) if flags else "✓ OK"
            return row

        filtered = filtered.apply(_us_apply_risk, axis=1)
        rr_failed = filtered[filtered["R:R"].notna() & (filtered["R:R"] < us_rm_min_rr)]
        filtered  = filtered[filtered["R:R"].isna() | (filtered["R:R"] >= us_rm_min_rr)]

        # Sector concentration warning
        if not filtered.empty and "Sector" in filtered.columns and _us_slots > 0:
            top_sec = filtered["Sector"].value_counts()
            for sec_name, sec_count in top_sec.items():
                if sec_count / max(_us_slots, 1) * 100 > us_rm_max_sector_pct:
                    st.warning(
                        f"⚠️  **Sector concentration**: {sec_count}/{_us_slots} available slots are "
                        f"**{sec_name}** stocks ({sec_count/max(_us_slots,1)*100:.0f}%) — above "
                        f"your {us_rm_max_sector_pct}% limit."
                    )

        buys   = len(df_scan[df_scan["Signal"] == "BUY"])
        sells  = len(df_scan[df_scan["Signal"] == "SELL"])
        high_q = len(df_scan[df_scan["Score"] >= 70])
        strong = len(df_scan[df_scan["ADX"] >= 30])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("BUY Setups",   buys)
        m2.metric("SELL Setups",  sells)
        m3.metric("High Quality (≥70)", high_q)
        m4.metric("Strong Trend (ADX≥30)", strong)
        m5.metric("Failed R:R Filter", len(rr_failed))

        st.markdown(f"**{len(filtered)} trades pass risk filter** · {len(rr_failed)} removed (R:R < {us_rm_min_rr:.1f}) · {len(df_scan)} scanned")

        if not _us_trading_ok:
            st.error("Trading blocked by risk model. See status above.")
        elif filtered.empty:
            st.info("No stocks match. Try lowering Min Score, Min ADX, or Min R:R.")
        else:
            def _style_scan(df):
                def row_style(row):
                    if row["Signal"] == "BUY":   return ["background-color:#e8f5ec"] * len(row)
                    if row["Signal"] == "SELL":  return ["background-color:#fdecea"] * len(row)
                    return ["background-color:#fff8e1"] * len(row)
                def score_style(val):
                    if not isinstance(val, (int,float)): return ""
                    if val >= 70: return "color:#1a7a3c; font-weight:700"
                    if val >= 50: return "color:#b8860b; font-weight:600"
                    return "color:#888"
                def sig_style(val):
                    if val == "BUY":  return "color:#1a7a3c; font-weight:700"
                    if val == "SELL": return "color:#c0392b; font-weight:700"
                    return "color:#b8860b"
                def chg_style(val):
                    if not isinstance(val, float): return ""
                    return "color:#1a7a3c; font-weight:600" if val > 0 else "color:#c0392b; font-weight:600"
                def rr_style(val):
                    if not isinstance(val, (int,float)): return ""
                    if val >= 3.0: return "color:#1a7a3c; font-weight:700"
                    if val >= 2.0: return "color:#b8860b; font-weight:600"
                    return "color:#c0392b"
                def flag_style(val):
                    return "color:#1a7a3c" if str(val).startswith("✓") else "color:#c0392b; font-weight:600"
                def type_style(val):
                    return {
                        "Compounder": "color:#1a7a3c; font-weight:700",
                        "Cyclical":   "color:#1a5aad; font-weight:600",
                        "Turnaround": "color:#b8860b; font-weight:600",
                        "Narrative":  "color:#7b2d8b; font-weight:700",
                    }.get(str(val), "")
                cols = df.columns.tolist()
                style = (df.style
                         .apply(row_style, axis=1)
                         .applymap(score_style, subset=["Score"])
                         .applymap(sig_style,   subset=["Signal"])
                         .applymap(chg_style,   subset=["Day Chg %"]))
                if "R:R" in cols:       style = style.applymap(rr_style,   subset=["R:R"])
                if "Risk Flag" in cols: style = style.applymap(flag_style, subset=["Risk Flag"])
                if "Type" in cols:      style = style.applymap(type_style, subset=["Type"])
                fmt = {
                    "Price ($)":      "${:,.2f}",
                    "Stop ($)":       lambda v: f"${v:,.2f}" if v else "—",
                    "Target ($)":     lambda v: f"${v:,.2f}" if v else "—",
                    "Day Chg %":      "{:+.2f}%",
                    "Vol Ratio":      "{:.1f}×",
                    "ADX":            "{:.1f}", "RSI": "{:.1f}",
                    "R:R":            lambda v: f"1:{v:.1f}" if v else "—",
                    "Qty":            lambda v: f"{int(v):,}" if v else "—",
                    "Trade Value ($)":lambda v: f"${v:,.0f}" if v else "—",
                }
                return style.format(fmt)

            _col_order = ["Symbol","Name","Sector","Type","Signal","Score",
                          "Price ($)","Day Chg %",
                          "ADX","RSI","Vol Ratio",
                          "R:R","Stop ($)","Target ($)","Qty","Trade Value ($)","Risk Flag","ATR"]
            _dc = [c for c in _col_order if c in filtered.columns]
            df_disp = filtered[_dc].reset_index(drop=True)

            st.dataframe(_style_scan(df_disp), hide_index=True, width="stretch",
                         height=min(700, 65 + len(df_disp) * 38))

            # Risk summary
            _tv_sum = filtered["Trade Value ($)"].dropna().sum()
            _risk_sum = filtered.apply(
                lambda r: abs((r["Price ($)"] or 0) - (r["Stop ($)"] or 0)) * (r["Qty"] or 0), axis=1
            ).sum()
            n_v = (filtered["Qty"].notna() & (filtered["Qty"] > 0)).sum()
            if n_v > 0:
                st.html(f"""
                <div style="background:#f0f4fc; border-radius:8px; padding:12px 18px;
                            margin:8px 0; font-size:0.83rem; color:#333;">
                  <b>If you took ALL {n_v} filtered trades:</b> &nbsp;
                  Total deployed = <b>${_tv_sum:,.0f}</b> ({_tv_sum/us_rm_capital*100:.1f}% of capital) &nbsp;·&nbsp;
                  Total risk = <b>${_risk_sum:,.0f}</b> ({_risk_sum/us_rm_capital*100:.1f}% of capital) &nbsp;·&nbsp;
                  {'<span style="color:#c0392b; font-weight:700">⚠ Exceeds monthly budget!</span>'
                    if _risk_sum > _us_monthly_rem else
                   '<span style="color:#1a7a3c; font-weight:700">✓ Within monthly budget</span>'}
                </div>
                """)

            st.download_button("⬇ Download CSV",
                data=filtered[_dc].to_csv(index=False).encode("utf-8"),
                file_name=f"us_trade_opportunities_{date.today()}.csv",
                mime="text/csv", key="us_dl_scan")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET SENTIMENT
# ════════════════════════════════════════════════════════════════════════════

with tab_sentiment:
    st.markdown("### 🌡️ US Market Sentiment")
    st.caption("Live overview — indices, VIX, sector ETF performance, Fear & Greed.")

    col_r1, col_r2 = st.columns([1, 4])
    with col_r1:
        if st.button("🔄 Load / Refresh", type="primary", key="us_sentiment_refresh"):
            fetch_market_sentiment.clear()
            st.session_state["us_sent_loaded"] = True
    with col_r2:
        st.caption("Fetches indices, VIX, and 11 sector ETFs")

    if not st.session_state.get("us_sent_loaded"):
        st.info("👆 Click **Load / Refresh** to fetch live US market sentiment.")
        st.stop()

    with st.spinner("Fetching market data…"):
        sdata = fetch_market_sentiment()

    sentiment    = sdata["sentiment"]
    score_s      = sdata["score"]
    indices      = sdata["indices"]
    sectors      = sdata["sectors"]
    fg           = sdata["fear_greed"]
    green_s      = sdata["green_sectors"]
    total_s      = sdata["total_sectors"]
    vix_data     = sdata["vix"]

    # ── Sentiment gauge ───────────────────────────────────────────────────────
    if sentiment == "BULLISH":
        gc, gb, gi = "#1a7a3c", "#e8f5ec", "🟢"
    elif sentiment == "BEARISH":
        gc, gb, gi = "#c0392b", "#fdecea", "🔴"
    else:
        gc, gb, gi = "#b8860b", "#fff8e1", "🟡"

    st.html(f"""
    <div style="background:{gb}; border-left:5px solid {gc}; border-radius:8px;
                padding:16px 20px; margin-bottom:16px; display:flex; align-items:center; gap:20px;">
      <div style="font-size:2.5rem;">{gi}</div>
      <div>
        <div style="font-size:1.6rem; font-weight:700; color:{gc};">{sentiment} MARKET</div>
        <div style="font-size:0.9rem; color:#555;">
          Sentiment score: <strong>{score_s}/100</strong> ·
          {green_s}/{total_s} sectors green ·
          Fear &amp; Greed proxy: <strong>{fg}/100</strong>
        </div>
      </div>
    </div>
    """)

    # ── Key indices ───────────────────────────────────────────────────────────
    st.markdown("#### 📈 Key Indices")
    if indices:
        idx_cols = st.columns(len(indices))
        for col, (name, q) in zip(idx_cols, indices.items()):
            val = f"{q['last']:,.0f}" if q["last"] > 1000 else f"{q['last']:.2f}"
            col.metric(name, val, f"{q['chg_pct']:+.2f}%")

    st.html("<hr style='margin:16px 0; border:none; border-top:1px solid #e0e0e0;'>")

    # ── VIX interpretation ────────────────────────────────────────────────────
    if vix_data:
        vix_val = vix_data["last"]
        vix_chg = vix_data["chg_pct"]
        if vix_val < 15:
            vl, vc, vm = "LOW",      "#1a7a3c", "Calm market — good for trend trades"
        elif vix_val < 20:
            vl, vc, vm = "MODERATE", "#b8860b", "Normal vol — trade with standard stops"
        elif vix_val < 30:
            vl, vc, vm = "ELEVATED", "#e65100", "Higher vol — use wider stops, smaller size"
        else:
            vl, vc, vm = "HIGH",     "#c0392b", "Fear in market — avoid aggressive entries"

        c1, c2 = st.columns([1,3])
        with c1:
            st.metric("VIX", f"{vix_val:.2f}", f"{vix_chg:+.2f}%")
        with c2:
            st.html(f"""<div style="background:#f8f9fa; border-left:4px solid {vc};
                         border-radius:6px; padding:12px 16px; margin-top:4px;">
              <span style="color:{vc}; font-weight:700;">VIX {vl}</span>
              <span style="color:#555; margin-left:8px;">{vm}</span></div>""")

    st.html("<hr style='margin:16px 0; border:none; border-top:1px solid #e0e0e0;'>")

    # ── Sector ETF heatmap ────────────────────────────────────────────────────
    st.markdown("#### 🗂️ Sector ETF Performance")
    if sectors:
        sorted_s = sorted(sectors.items(), key=lambda x: x[1]["chg_pct"], reverse=True)
        rows = [sorted_s[i:i+4] for i in range(0, len(sorted_s), 4)]
        for row in rows:
            rcols = st.columns(4)
            for col, (sec, q) in zip(rcols, row):
                chg = q["chg_pct"]
                wk  = q["week_chg"]
                if chg >= 1.5:    bg, fg2 = "#1a7a3c", "white"
                elif chg >= 0.5:  bg, fg2 = "#4caf50", "white"
                elif chg >= 0:    bg, fg2 = "#c8e6c9", "#1b5e20"
                elif chg >= -0.5: bg, fg2 = "#ffcdd2", "#b71c1c"
                elif chg >= -1.5: bg, fg2 = "#e53935", "white"
                else:             bg, fg2 = "#b71c1c", "white"
                col.html(f"""<div style="background:{bg}; color:{fg2}; border-radius:8px;
                              padding:10px 12px; text-align:center; margin:2px;">
                  <div style="font-weight:700; font-size:0.95rem;">{sec}</div>
                  <div style="font-size:1.1rem; font-weight:700;">{chg:+.2f}%</div>
                  <div style="font-size:0.75rem; opacity:0.85;">Week: {wk:+.1f}%</div>
                </div>""")

    st.html("<hr style='margin:16px 0; border:none; border-top:1px solid #e0e0e0;'>")

    # ── S&P 500 mini chart ────────────────────────────────────────────────────
    st.markdown("#### 📉 S&P 500 — 3 Month Chart")
    try:
        sp_df = yf.download("^GSPC", period="3mo", interval="1d",
                             progress=False, auto_adjust=True)
        if isinstance(sp_df.columns, pd.MultiIndex):
            sp_df.columns = sp_df.columns.get_level_values(0)
        sp_df = sp_df.dropna()
        if len(sp_df) > 5:
            ema20 = sp_df["Close"].squeeze().ewm(span=20, adjust=False).mean()
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Candlestick(
                x=sp_df.index,
                open=sp_df["Open"].squeeze(), high=sp_df["High"].squeeze(),
                low=sp_df["Low"].squeeze(),   close=sp_df["Close"].squeeze(),
                increasing_line_color="#1a7a3c", decreasing_line_color="#c0392b",
                name="S&P 500",
            ))
            fig_sp.add_trace(go.Scatter(x=sp_df.index, y=ema20,
                line=dict(color="#1a5aad", width=1.5), name="EMA 20"))
            fig_sp.update_layout(
                height=320, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="white", plot_bgcolor="#fafafa",
                xaxis=dict(type="date", showgrid=False, rangeslider_visible=False,
                           rangebreaks=[dict(bounds=["sat","mon"])]),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                legend=dict(orientation="h", y=1.02, x=0),
            )
            st.plotly_chart(fig_sp, width="stretch",
                            config={"displayModeBar": False})
    except Exception:
        st.info("S&P 500 chart unavailable.")
