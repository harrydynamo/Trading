"""
Fetches the latest NSE data and applies the strategy to produce
real-time BUY / WATCH / EXIT signals.

Signal types:
  BUY   — all entry conditions met right now (Setup A or B)
  WATCH — weekly filter passes + volume building, entry not triggered yet
  EXIT  — a tracked position has hit a stop, target, or MA exit
"""

import sys
import os
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from simulation.backtest import precompute

logger = logging.getLogger(__name__)


# ─── Signal data classes ──────────────────────────────────────────────────────

@dataclass
class BuySignal:
    symbol:      str
    setup:       str          # "A" or "B"
    price:       float        # current close
    stop_loss:   float        # 3 ATR below entry
    target:      float        # nearest resistance
    atr:         float
    reward_risk: float
    vol_rank:    float        # volume percentile today
    drop_3d:     float        # % drop over last 3 days (Setup B only)
    notes:       str

    @property
    def risk_per_share(self) -> float:
        return self.price - self.stop_loss

    def shares_for_capital(self, capital: float) -> int:
        return int(capital // self.price)


@dataclass
class WatchSignal:
    symbol:    str
    price:     float
    w_ma200:   float
    w_ma20:    float
    vol_rank:  float          # current volume rank (may be below 80 yet)
    vol_days:  int            # consecutive days of elevated volume so far
    reason:    str            # what is still missing for a full entry


@dataclass
class ExitSignal:
    symbol:    str
    reason:    str
    price:     float
    entry_price: float
    pnl_pct:   float


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_and_precompute(symbol: str) -> Optional[pd.DataFrame]:
    """Download 5 years of daily data and precompute all indicators."""
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, period="5y", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        if len(df) < 250:
            return None
        return precompute(df)
    except Exception as e:
        logger.error(f"{symbol}: {e}")
        return None


# ─── Signal checks ────────────────────────────────────────────────────────────

def check_buy_signal(symbol: str, df: pd.DataFrame) -> Optional[BuySignal]:
    """Return a BuySignal if the latest bar meets all entry conditions."""
    row = df.iloc[-1]

    if not bool(row.get("entry_signal", False)):
        return None

    price  = float(row["Close"])
    stop   = float(row["stop"])
    target = float(row["resistance"])
    atr    = float(row["atr"])
    rr     = float(row["rr"])
    setup  = str(row["entry_setup"])
    vrank  = float(row["vol_rank"]) if not np.isnan(row["vol_rank"]) else 0.0
    drop   = float(row["drop_3d"]) if not np.isnan(row["drop_3d"]) else 0.0

    notes = (
        f"{config.SETUP_A_LOW_DAYS}-day closing low pullback" if setup == "A"
        else f"3-day drop {drop:.1%} + hammer candle"
    )

    return BuySignal(
        symbol=symbol, setup=setup, price=price,
        stop_loss=stop, target=target, atr=atr,
        reward_risk=rr, vol_rank=vrank, drop_3d=drop,
        notes=notes,
    )


def check_watch_signal(symbol: str, df: pd.DataFrame) -> Optional[WatchSignal]:
    """
    Return a WatchSignal if the weekly filter passes but the daily
    entry has not triggered yet — useful to monitor 'almost ready' stocks.
    """
    row = df.iloc[-1]

    # Already a full BUY — don't double-count
    if bool(row.get("entry_signal", False)):
        return None

    # Must at least pass weekly trend filter
    if not bool(row.get("weekly_ok", False)):
        return None

    price   = float(row["Close"])
    w_ma200 = float(row["w_ma200"]) if not np.isnan(row.get("w_ma200", np.nan)) else 0.0
    w_ma20  = float(row["w_ma20"])  if not np.isnan(row.get("w_ma20",  np.nan)) else 0.0
    vrank   = float(row["vol_rank"]) if not np.isnan(row.get("vol_rank", np.nan)) else 0.0

    # Count how many of the last 3 days had elevated volume
    vol_days = int(df["vol_rank"].tail(3).ge(config.VOLUME_PERCENTILE).sum())

    missing = []
    if not bool(row.get("vol_ok", False)):
        missing.append(f"volume (rank {vrank:.0f}%, need {vol_days}/3 elevated days)")
    if not bool(row.get("setup_a_price", False)) and not (
        bool(row.get("drop_3d", 0) >= config.SETUP_B_DROP_THRESHOLD) and bool(row.get("is_hammer", False))
    ):
        missing.append(f"price trigger ({config.SETUP_A_LOW_DAYS}d low or {config.SETUP_B_DROP_THRESHOLD:.0%} drop+hammer)")

    if not missing:
        return None  # nothing blocking — should have been a BUY

    return WatchSignal(
        symbol=symbol, price=price, w_ma200=w_ma200, w_ma20=w_ma20,
        vol_rank=vrank, vol_days=vol_days,
        reason="Waiting for: " + " | ".join(missing),
    )


def check_exit_signal(position: dict, df: pd.DataFrame) -> Optional[ExitSignal]:
    """
    Check whether a tracked position should be exited right now.
    `position` is a dict loaded from the portfolio JSON.
    """
    MIN_HOLD_DAYS = config.MIN_HOLD_DAYS

    row          = df.iloc[-1]
    price        = float(row["Close"])
    current_atr  = float(row["atr"]) if not np.isnan(row["atr"]) else position["atr"]
    ma20         = float(row["ma20_daily"]) if not np.isnan(row.get("ma20_daily", np.nan)) else price
    entry_price  = float(position["entry_price"])
    stop_loss    = float(position["stop_loss"])
    target       = float(position["target_price"])
    trail_stop   = float(position.get("trail_stop", entry_price - config.ATR_TRAIL_MULTIPLIER * current_atr))
    entry_date   = pd.Timestamp(position["entry_date"])
    days_held    = (pd.Timestamp.today() - entry_date).days

    # Update trailing stop (ratchet up only)
    new_trail = price - config.ATR_TRAIL_MULTIPLIER * current_atr
    trail_stop = max(trail_stop, new_trail)

    pnl_pct = (price - entry_price) / entry_price

    reason = None
    if price <= stop_loss:
        reason = f"Hard stop hit  (stop ₹{stop_loss:.2f})"
    elif price <= trail_stop and days_held >= MIN_HOLD_DAYS:
        reason = f"Trailing stop  (trail ₹{trail_stop:.2f})"
    elif price >= target:
        reason = f"Target reached (target ₹{target:.2f})"
    elif price < ma20 and days_held >= MIN_HOLD_DAYS:
        reason = f"Close below daily 20 MA  (MA ₹{ma20:.2f})"

    if not reason:
        return None

    return ExitSignal(
        symbol=position["symbol"], reason=reason,
        price=price, entry_price=entry_price, pnl_pct=pnl_pct,
    )


# ─── Full scan ────────────────────────────────────────────────────────────────

def run_scan(symbols: list[str], positions: list[dict]) -> dict:
    """
    Scan all symbols and return:
      buys    — list[BuySignal]
      watches — list[WatchSignal]
      exits   — list[ExitSignal]
    """
    buys:    list[BuySignal]   = []
    watches: list[WatchSignal] = []
    exits:   list[ExitSignal]  = []

    held_symbols = {p["symbol"] for p in positions}

    for symbol in symbols:
        df = fetch_and_precompute(symbol)
        if df is None:
            continue

        # Exit check first (for positions we hold)
        if symbol in held_symbols:
            pos = next(p for p in positions if p["symbol"] == symbol)
            exit_sig = check_exit_signal(pos, df)
            if exit_sig:
                exits.append(exit_sig)
        else:
            # Entry scan (only for stocks we don't already hold)
            buy = check_buy_signal(symbol, df)
            if buy:
                buys.append(buy)
            else:
                watch = check_watch_signal(symbol, df)
                if watch:
                    watches.append(watch)

    # Sort: BUY by R:R desc, WATCH by vol_rank desc
    buys.sort(key=lambda s: s.reward_risk, reverse=True)
    watches.sort(key=lambda s: s.vol_rank, reverse=True)

    return {"buys": buys, "watches": watches, "exits": exits}
