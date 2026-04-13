"""
Technical indicator calculations for the live trading UI.

All functions accept a DataFrame with Open, High, Low, Close, Volume columns.
compute_all(df) returns an enriched copy with every indicator as a new column.
"""

import numpy as np
import pandas as pd


# ─── Individual indicators ─────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower). Uses population std (ddof=0) to match TradingView."""
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std(ddof=0)
    return mid + std_dev * sigma, mid, mid - std_dev * sigma


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothing RSI — com = period-1 matches standard charting platforms."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative VWAP. For intraday data the index is a DatetimeIndex — VWAP resets
    each calendar day. For daily/weekly data it's computed over the whole window.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].replace(0, np.nan)

    if isinstance(df.index, pd.DatetimeIndex) and df.index.resolution in ("minute", "hour"):
        # Intraday: reset per day
        dates    = df.index.normalize()
        cumvol   = vol.groupby(dates).cumsum()
        cumtpvol = (typical * vol).groupby(dates).cumsum()
    else:
        cumvol   = vol.cumsum()
        cumtpvol = (typical * vol).cumsum()

    return cumtpvol / cumvol


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's smoothing ATR."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Returns (stoch_k, stoch_d)."""
    lowest_low   = df["Low"].rolling(k_period).min()
    highest_high = df["High"].rolling(k_period).max()
    denom        = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k      = 100 * (df["Close"] - lowest_low) / denom
    stoch_d      = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d


def supertrend(
    df: pd.DataFrame, period: int = 10, mult: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.
    Returns (line, direction) where direction +1 = bullish, -1 = bearish.

    Algorithm:
      1. Compute ATR-based basic upper/lower bands.
      2. Band adjustment pass (iterative recurrence): upper band only moves down,
         lower band only moves up; each resets when price closes through it.
      3. Direction pass: flip direction when close crosses the active band.
    """
    atr_vals    = atr(df, period).values
    hl2         = ((df["High"] + df["Low"]) / 2).values
    close       = df["Close"].values
    n           = len(df)

    basic_upper = hl2 + mult * atr_vals
    basic_lower = hl2 - mult * atr_vals

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    # ── Pass 1: adjust bands ───────────────────────────────────────────────
    for i in range(1, n):
        if np.isnan(atr_vals[i]):
            continue
        # If previous band is NaN (first valid ATR bar), initialise from scratch
        if np.isnan(final_upper[i - 1]):
            final_upper[i] = basic_upper[i]
        elif basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if np.isnan(final_lower[i - 1]):
            final_lower[i] = basic_lower[i]
        elif basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

    # ── Pass 2: determine direction and supertrend line ────────────────────
    st        = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(final_upper[i]) or np.isnan(final_lower[i]):
            continue
        prev_dir = direction[i - 1]
        if prev_dir == 0:
            # Initialise: price above midpoint → bullish
            direction[i] = 1 if close[i] >= (final_upper[i] + final_lower[i]) / 2 else -1
        elif prev_dir == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:  # prev_dir == -1
            direction[i] =  1 if close[i] > final_upper[i] else -1

        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return (
        pd.Series(st,        index=df.index, name="supertrend"),
        pd.Series(direction, index=df.index, name="st_direction"),
    )


def donchian(
    df: pd.DataFrame, period: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channels. Returns (upper, mid, lower)."""
    upper = df["High"].rolling(period).max()
    lower = df["Low"].rolling(period).min()
    mid   = (upper + lower) / 2
    return upper, mid, lower


def adx(
    df: pd.DataFrame, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (ADX) with +DI and -DI.
    Returns (adx, plus_di, minus_di).

    ADX > 25 → market is TRENDING  (use EMA / Supertrend / MACD)
    ADX < 20 → market is RANGING   (use RSI / BB / Stochastic)
    20–25    → MIXED / transitional
    """
    high       = df["High"]
    low        = df["Low"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # Wilder's smoothing (same as ATR)
    atr_s      = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di_s  = (100 * plus_dm.ewm(com=period - 1, min_periods=period).mean()
                  / atr_s.replace(0, np.nan))
    minus_di_s = (100 * minus_dm.ewm(com=period - 1, min_periods=period).mean()
                  / atr_s.replace(0, np.nan))

    dx    = (100 * (plus_di_s - minus_di_s).abs()
             / (plus_di_s + minus_di_s).replace(0, np.nan))
    adx_s = dx.ewm(com=period - 1, min_periods=period).mean()

    return adx_s, plus_di_s, minus_di_s


# ─── Master function ───────────────────────────────────────────────────────────

def compute_all(
    df: pd.DataFrame,
    use_supertrend: bool = True,
    use_donchian:   bool = True,
) -> pd.DataFrame:
    """Enrich a raw OHLCV DataFrame with all technical indicators."""
    d = df.copy()
    # Flatten MultiIndex columns that newer yfinance versions sometimes return
    if isinstance(d.columns, pd.MultiIndex):
        d = d.droplevel(level=1, axis=1)
        d = d.loc[:, ~d.columns.duplicated()]
    # Ensure every OHLCV column is a 1-D Series, not a single-column DataFrame
    for _col in list(d.columns):
        if isinstance(d[_col], pd.DataFrame):
            d[_col] = d[_col].iloc[:, 0]
    c = d["Close"]

    d["ema_9"]   = ema(c, 9)
    d["ema_21"]  = ema(c, 21)
    d["ema_50"]  = ema(c, 50)
    d["ema_200"] = ema(c, 200)
    d["sma_20"]  = sma(c, 20)

    d["bb_upper"], d["bb_mid"], d["bb_lower"] = bollinger_bands(c)

    d["rsi"] = rsi(c)

    d["macd_line"], d["macd_signal"], d["macd_hist"] = macd(c)

    if "Volume" in d.columns:
        d["vwap"] = vwap(d)

    d["atr"] = atr(d)

    d["stoch_k"], d["stoch_d"] = stochastic(d)

    if use_supertrend:
        d["supertrend"], d["st_direction"] = supertrend(d)

    if use_donchian:
        d["dc_upper"], d["dc_mid"], d["dc_lower"] = donchian(d)

    d["adx"], d["plus_di"], d["minus_di"] = adx(d)

    return d
