"""
Technical indicator calculations used by the strategy.
All functions accept a pandas DataFrame with OHLCV columns.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def is_hammer(df: pd.DataFrame, idx: int = -1) -> bool:
    """
    Detects a hammer candlestick at a given row index.

    Hammer rules:
    - Real body is small (< 30% of total candle range)
    - Lower shadow is at least 2x the body size
    - Upper shadow is at most 30% of the body size
    """
    row = df.iloc[idx]
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]

    candle_range = h - l
    if candle_range == 0:
        return False

    body = abs(c - o)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l

    body_pct = body / candle_range
    if body_pct > 0.30:          # body must be small
        return False
    if body == 0:
        return False
    if lower_shadow < 2 * body:  # long lower shadow
        return False
    if upper_shadow > 0.3 * body:  # tiny upper shadow
        return False
    return True


def volume_rank_percentile(df: pd.DataFrame, lookback_days: int) -> pd.Series:
    """
    Returns a rolling percentile rank (0–100) of today's volume
    relative to the past `lookback_days` trading days.
    """
    def _pct_rank(x):
        if len(x) < 2:
            return np.nan
        return (x[:-1] < x[-1]).mean() * 100  # exclude today from the sample

    return df["Volume"].rolling(window=lookback_days + 1).apply(_pct_rank, raw=True)


def volume_above_percentile(df: pd.DataFrame, lookback_days: int,
                             threshold_pct: float = 80) -> pd.Series:
    """Returns a boolean series: True when volume rank >= threshold_pct."""
    return volume_rank_percentile(df, lookback_days) >= threshold_pct


def volume_consecutive_days(df: pd.DataFrame, lookback_days: int,
                             threshold_pct: float = 80, consecutive: int = 3) -> pd.Series:
    """
    Returns True on a row if volume has been in the top (100-threshold_pct)%
    for `consecutive` days in a row including today.
    """
    flag = volume_above_percentile(df, lookback_days, threshold_pct).astype(int)
    return flag.rolling(window=consecutive).sum() == consecutive


def volume_below_peak(df: pd.DataFrame, peak_days: int = 10) -> pd.Series:
    """
    Returns True when today's volume is NOT the highest in the last `peak_days`.
    This filters out climax/exhaustion volume spikes.
    """
    rolling_max = df["Volume"].rolling(window=peak_days).max()
    return df["Volume"] < rolling_max


def find_resistance(df: pd.DataFrame, lookback: int = 60) -> float:
    """
    Simple resistance estimation: highest high over the lookback window
    (excluding the most recent candle).
    """
    return df["High"].iloc[-lookback:-1].max()


def near_weekly_ma(weekly_close: float, weekly_ma20: float,
                   threshold: float = 0.05) -> bool:
    """Price is within `threshold` (5%) of the weekly 20 MA."""
    if weekly_ma20 == 0:
        return False
    return abs(weekly_close - weekly_ma20) / weekly_ma20 <= threshold
