"""
Core strategy logic — scans a stock and returns entry/exit signals.

Uses:
  - Weekly data: 200 MA trend filter, 20 MA proximity filter
  - Daily data:  Setup A / Setup B entry, ATR stop, volume conditions, exit rules
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

import config
import indicators as ind

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol: str
    setup: str                      # "A", "B", or "EXIT"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    atr: float = 0.0
    reward_risk: float = 0.0
    notes: str = ""


@dataclass
class Position:
    symbol: str
    setup: str
    entry_price: float
    stop_loss: float
    target_price: float
    atr: float
    quantity: int
    capital_deployed: float
    trail_stop: float = 0.0        # updated dynamically


def fetch_data(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Download OHLCV data from Yahoo Finance (NSE: symbol.NS)."""
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            logger.warning(f"{symbol}: insufficient data for {interval}")
            return None
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"{symbol}: data fetch error — {e}")
        return None


def passes_weekly_filter(symbol: str) -> tuple[bool, float]:
    """
    Weekly timeframe checks:
    1. Close > 200 MA (trend direction)
    2. Close is near the 20 MA (mean-reversion proximity)

    Returns (passes: bool, weekly_close: float)
    """
    df = fetch_data(symbol, interval="1wk", period="5y")
    if df is None or len(df) < config.WEEKLY_TREND_MA + 5:
        return False, 0.0

    df["ma200"] = ind.sma(df["Close"], config.WEEKLY_TREND_MA)
    df["ma20"] = ind.sma(df["Close"], config.WEEKLY_PROXIMITY_MA)
    df.dropna(inplace=True)

    latest = df.iloc[-1]
    close = float(latest["Close"])
    ma200 = float(latest["ma200"])
    ma20 = float(latest["ma20"])

    above_200 = close > ma200
    near_20 = ind.near_weekly_ma(close, ma20, config.PROXIMITY_THRESHOLD)

    logger.debug(f"{symbol} weekly | close={close:.2f} ma200={ma200:.2f} "
                 f"ma20={ma20:.2f} above_200={above_200} near_20={near_20}")

    return (above_200 and near_20), close


def _volume_conditions_met(df: pd.DataFrame) -> bool:
    """
    Volume must:
    1. Be in top 20% relative to last 12 weeks (84 trading days) for 3 consecutive days
    2. Not be the highest in the last 10 days (avoid climax volume)
    """
    lookback = config.VOLUME_LOOKBACK_WEEKS * 5  # ~12 weeks in trading days

    consec = ind.volume_consecutive_days(
        df, lookback,
        threshold_pct=config.VOLUME_PERCENTILE,
        consecutive=config.VOLUME_CONSECUTIVE_DAYS,
    )
    not_peak = ind.volume_below_peak(df, config.VOLUME_PEAK_DAYS)

    return bool(consec.iloc[-1]) and bool(not_peak.iloc[-1])


def scan_setup_a(symbol: str, df_daily: pd.DataFrame, atr_val: float) -> Optional[Signal]:
    """
    Setup A — 5-Day Low Pullback
    - Price is at a 5-day low today
    - Price is above weekly 200 MA (already confirmed by caller)
    - Volume conditions met
    """
    if not _volume_conditions_met(df_daily):
        return None

    recent_lows = df_daily["Low"].iloc[-config.SETUP_A_LOW_DAYS:]
    today_low = float(df_daily["Low"].iloc[-1])
    today_close = float(df_daily["Close"].iloc[-1])

    if today_close > float(recent_lows.min()):
        return None  # not at 5-day low

    return Signal(
        symbol=symbol,
        setup="A",
        entry_price=today_close,
        stop_loss=today_close - config.ATR_STOP_MULTIPLIER * atr_val,
        atr=atr_val,
        notes="5-day low pullback with elevated volume",
    )


def scan_setup_b(symbol: str, df_daily: pd.DataFrame, atr_val: float) -> Optional[Signal]:
    """
    Setup B — 10% Drop + Hammer Confirmation
    - Price has dropped >= 10% over last 3 days
    - Today's candle is a hammer
    - Volume conditions met
    """
    if not _volume_conditions_met(df_daily):
        return None

    if not ind.is_hammer(df_daily, idx=-1):
        return None

    close_now = float(df_daily["Close"].iloc[-1])
    close_3d_ago = float(df_daily["Close"].iloc[-config.SETUP_B_DROP_DAYS - 1])

    if close_3d_ago == 0:
        return None

    drop_pct = (close_3d_ago - close_now) / close_3d_ago
    if drop_pct < config.SETUP_B_DROP_THRESHOLD:
        return None

    return Signal(
        symbol=symbol,
        setup="B",
        entry_price=close_now,
        stop_loss=close_now - config.ATR_STOP_MULTIPLIER * atr_val,
        atr=atr_val,
        notes=f"10%+ drop ({drop_pct:.1%}) + hammer confirmation",
    )


def calculate_target(df_daily: pd.DataFrame, entry: float,
                     stop: float) -> tuple[float, float]:
    """
    Find the nearest resistance level and check for 2:1 R:R.
    Returns (target_price, reward_risk_ratio).
    """
    resistance = ind.find_resistance(df_daily, lookback=60)
    risk = entry - stop
    if risk <= 0:
        return 0.0, 0.0
    reward = resistance - entry
    rr = reward / risk
    return resistance, rr


def scan_symbol(symbol: str) -> Optional[Signal]:
    """
    Full scan for a single symbol.
    Returns a Signal if an entry is triggered, else None.
    """
    # Step 1: Weekly filter
    passes, weekly_close = passes_weekly_filter(symbol)
    if not passes:
        logger.debug(f"{symbol}: failed weekly filter")
        return None

    # Step 2: Fetch daily data
    df = fetch_data(symbol, interval="1d", period="1y")
    if df is None or len(df) < 100:
        return None

    df["atr"] = ind.atr(df, config.ATR_PERIOD)
    df.dropna(inplace=True)

    atr_val = float(df["atr"].iloc[-1])
    if atr_val <= 0:
        return None

    # Step 3: Try Setup A then Setup B
    signal = scan_setup_a(symbol, df, atr_val) or scan_setup_b(symbol, df, atr_val)
    if signal is None:
        return None

    # Step 4: R:R check
    target, rr = calculate_target(df, signal.entry_price, signal.stop_loss)
    if rr < config.MIN_REWARD_RISK:
        logger.debug(f"{symbol}: R:R {rr:.2f} below minimum {config.MIN_REWARD_RISK}")
        return None

    signal.target_price = target
    signal.reward_risk = rr
    return signal


# ─── EXIT CHECKS ─────────────────────────────────────────────────────────────

def check_exit(position: Position, df_daily: pd.DataFrame) -> tuple[bool, str]:
    """
    Returns (should_exit: bool, reason: str) for an open position.

    Exit conditions (any one triggers exit):
    1. Hard stop: current price <= 3 ATR stop
    2. Trailing stop: current price <= 2 ATR trail stop
    3. Target hit: current price >= target
    4. Daily close below 20 MA
    """
    if df_daily is None or df_daily.empty:
        return False, ""

    df_daily = df_daily.copy()
    df_daily["ma20"] = ind.sma(df_daily["Close"], config.DAILY_EXIT_MA)
    df_daily["atr"] = ind.atr(df_daily, config.ATR_PERIOD)
    df_daily.dropna(inplace=True)

    current_price = float(df_daily["Close"].iloc[-1])
    current_atr = float(df_daily["atr"].iloc[-1])
    ma20 = float(df_daily["ma20"].iloc[-1])

    # Update trailing stop
    new_trail = current_price - config.ATR_TRAIL_MULTIPLIER * current_atr
    position.trail_stop = max(position.trail_stop, new_trail)

    if current_price <= position.stop_loss:
        return True, f"Hard stop hit @ {current_price:.2f} (stop={position.stop_loss:.2f})"

    if current_price <= position.trail_stop:
        return True, f"Trailing stop hit @ {current_price:.2f} (trail={position.trail_stop:.2f})"

    if current_price >= position.target_price:
        return True, f"Target hit @ {current_price:.2f} (target={position.target_price:.2f})"

    if current_price < ma20:
        return True, f"Daily close below 20 MA @ {current_price:.2f} (MA20={ma20:.2f})"

    return False, ""
