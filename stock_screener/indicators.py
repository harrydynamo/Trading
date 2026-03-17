"""
Technical indicator calculations for the stock screener.
All functions operate on a pandas DataFrame with OHLCV columns.
Returns scalar values (latest bar) used by the scorer.
"""

import numpy as np
import pandas as pd


# ─── Moving averages ──────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ─── RSI ─────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── MACD ────────────────────────────────────────────────────────────────────

def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line= ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


# ─── ATR ─────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─── Bollinger Bands ─────────────────────────────────────────────────────────

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    mid   = sma(series, period)
    std   = series.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


# ─── OBV (On-Balance Volume) ─────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff())
    return (direction * df["Volume"]).fillna(0).cumsum()


# ─── Weekly MA helper ─────────────────────────────────────────────────────────

def weekly_sma(daily_df: pd.DataFrame, period: int) -> pd.Series:
    """
    Resample to weekly, compute SMA, then forward-fill back to daily.
    Uses shift(1) so each day sees the previous completed week's value.
    """
    weekly = daily_df["Close"].resample("W-FRI").last()
    w_sma  = weekly.rolling(period).mean().shift(1)
    return w_sma.reindex(daily_df.index, method="ffill")


# ─── Rate of change ──────────────────────────────────────────────────────────

def roc(series: pd.Series, period: int) -> pd.Series:
    """Percentage rate of change over `period` bars."""
    return (series - series.shift(period)) / series.shift(period) * 100


# ─── 52-week high/low ────────────────────────────────────────────────────────

def high_52w(df: pd.DataFrame) -> float:
    return float(df["High"].rolling(252).max().iloc[-1])

def low_52w(df: pd.DataFrame) -> float:
    return float(df["Low"].rolling(252).min().iloc[-1])


# ─── Compute all indicators, return flat dict of latest values ────────────────

def compute_all(df: pd.DataFrame) -> dict:
    """
    Compute every indicator needed by the scorer and return
    a flat dict of scalar values (all from the latest bar).
    """
    if len(df) < 60:
        return {}

    close  = df["Close"]
    volume = df["Volume"]

    # SMAs
    sma20  = sma(close, 20)
    sma50  = sma(close, 50)
    sma200 = sma(close, 200)

    # Weekly 200 & 20 SMAs (need enough history)
    w_sma200 = weekly_sma(df, 200) if len(df) >= 1000 else pd.Series(np.nan, index=df.index)
    w_sma20  = weekly_sma(df, 20)

    # RSI
    rsi14 = rsi(close, 14)

    # MACD
    macd_line, signal_line, histogram = macd(close)

    # ATR
    atr14  = atr(df, 14)
    atr_pct = atr14 / close * 100          # ATR as % of price (volatility %)

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20)
    bb_width  = (bb_upper - bb_lower) / bb_mid * 100   # band width %
    bb_pos    = (close - bb_lower) / (bb_upper - bb_lower) * 100  # 0=at lower, 100=at upper

    # OBV
    obv_series  = obv(df)
    obv_sma20   = sma(obv_series, 20)

    # Volume averages
    vol_sma5    = sma(volume, 5)
    vol_sma20   = sma(volume, 20)
    vol_ratio   = volume / vol_sma20       # today's vol vs 20-day avg

    # 52-week levels
    h52 = close.rolling(252).max()
    l52 = close.rolling(252).min()
    pct_from_52h = (close - h52) / h52 * 100   # negative = below 52w high
    pct_from_52l = (close - l52) / l52 * 100   # positive = above 52w low

    # Rate of change
    roc1m  = roc(close, 21)    # ~1 month
    roc3m  = roc(close, 63)    # ~3 months
    roc6m  = roc(close, 126)   # ~6 months

    # Golden / Death cross (daily 50 vs 200)
    golden_cross = (sma50 > sma200).astype(int)

    def _last(s: pd.Series) -> float:
        v = s.iloc[-1]
        return float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan

    return {
        "price":         _last(close),
        "sma20":         _last(sma20),
        "sma50":         _last(sma50),
        "sma200":        _last(sma200),
        "w_sma200":      _last(w_sma200),
        "w_sma20":       _last(w_sma20),
        "rsi":           _last(rsi14),
        "macd":          _last(macd_line),
        "macd_signal":   _last(signal_line),
        "macd_hist":     _last(histogram),
        "atr":           _last(atr14),
        "atr_pct":       _last(atr_pct),
        "bb_upper":      _last(bb_upper),
        "bb_lower":      _last(bb_lower),
        "bb_pos":        _last(bb_pos),          # 0–100, higher = stronger
        "bb_width":      _last(bb_width),
        "obv":           _last(obv_series),
        "obv_above_sma": int(_last(obv_series) > _last(obv_sma20)),
        "vol_ratio":     _last(vol_ratio),       # today's vol / 20d avg
        "vol_5d_vs_20d": int(_last(vol_sma5) > _last(vol_sma20)),
        "pct_from_52h":  _last(pct_from_52h),
        "pct_from_52l":  _last(pct_from_52l),
        "roc1m":         _last(roc1m),
        "roc3m":         _last(roc3m),
        "roc6m":         _last(roc6m),
        "golden_cross":  _last(golden_cross),
        "above_sma20":   int(_last(close) > _last(sma20)),
        "above_sma50":   int(_last(close) > _last(sma50)),
        "above_sma200":  int(_last(close) > _last(sma200)),
        "above_w200":    int(_last(close) > _last(w_sma200)) if not np.isnan(_last(w_sma200)) else 0,
    }
