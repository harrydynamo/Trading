"""
Scores each stock out of 100 across five categories.

Score breakdown (100 pts total):
─────────────────────────────────────────────────────────
  Category              Max pts   What it measures
─────────────────────────────────────────────────────────
  1. Trend              30        Is the stock in an uptrend?
  2. Momentum           25        Is price acceleration positive?
  3. Volume             20        Is smart money flowing in?
  4. Relative Strength  15        How strong vs its own history?
  5. Volatility/Setup   10        Is it stable and ready to move?
─────────────────────────────────────────────────────────
"""

import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScoreBreakdown:
    symbol:     str
    name:       str
    exchange:   str
    cap:        str
    price:      float

    trend:      float = 0.0   # /30
    momentum:   float = 0.0   # /25
    volume:     float = 0.0   # /20
    strength:   float = 0.0   # /15
    setup:      float = 0.0   # /10

    indicators: dict = field(default_factory=dict)
    error:      str  = ""

    @property
    def total(self) -> float:
        return self.trend + self.momentum + self.volume + self.strength + self.setup

    @property
    def grade(self) -> str:
        t = self.total
        if t >= 80: return "A+"
        if t >= 70: return "A"
        if t >= 60: return "B+"
        if t >= 50: return "B"
        if t >= 40: return "C"
        return "D"

    @property
    def signal(self) -> str:
        t = self.total
        if t >= 75: return "STRONG BUY"
        if t >= 60: return "BUY"
        if t >= 45: return "WATCH"
        if t >= 30: return "NEUTRAL"
        return "AVOID"


def _safe(v, fallback=0.0):
    """Return fallback if value is NaN or None."""
    if v is None:
        return fallback
    try:
        return fallback if math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return fallback


def score_trend(ind: dict) -> float:
    """
    30 points — Is the stock in a healthy uptrend?

    Points  Condition
    ──────  ─────────────────────────────────────────────────
     10     Price above weekly 200 SMA (major trend direction)
      8     Price above daily 50 SMA   (intermediate trend)
      7     Price above daily 20 SMA   (short-term trend)
      5     Daily 50 SMA > 200 SMA (golden cross)
    """
    pts = 0.0

    if _safe(ind.get("above_w200")):     pts += 10
    if _safe(ind.get("above_sma50")):    pts += 8
    if _safe(ind.get("above_sma20")):    pts += 7
    if _safe(ind.get("golden_cross")):   pts += 5

    return pts


def score_momentum(ind: dict) -> float:
    """
    25 points — Is momentum positive and building?

    Points  Condition
    ──────  ──────────────────────────────────────────────────────────────
     12     RSI: 50–65 → 12pts  |  65–75 → 8pts  |  40–50 → 5pts  |  else 0
      8     MACD line above signal line (bullish crossover)
      5     1-month price change > 0%
    """
    pts  = 0.0
    rsi  = _safe(ind.get("rsi"), 50)
    roc1 = _safe(ind.get("roc1m"), 0)

    if   50 <= rsi < 65:  pts += 12
    elif 65 <= rsi < 75:  pts += 8
    elif 40 <= rsi < 50:  pts += 5
    # RSI < 40 or > 75 → 0 (oversold/overbought)

    macd_line = _safe(ind.get("macd"), 0)
    macd_sig  = _safe(ind.get("macd_signal"), 0)
    if macd_line > macd_sig:  pts += 8

    if roc1 > 0:  pts += 5

    return pts


def score_volume(ind: dict) -> float:
    """
    20 points — Is buying volume healthy?

    Points  Condition
    ──────  ─────────────────────────────────────────────────────────────
      8     5-day avg volume > 20-day avg (rising participation)
      7     Today's volume > 1.5× the 20-day average (strong session)
      5     OBV above its 20-day SMA (accumulation phase)
    """
    pts = 0.0

    if _safe(ind.get("vol_5d_vs_20d")):   pts += 8

    vol_ratio = _safe(ind.get("vol_ratio"), 0)
    if vol_ratio >= 1.5:    pts += 7
    elif vol_ratio >= 1.1:  pts += 4

    if _safe(ind.get("obv_above_sma")):   pts += 5

    return pts


def score_strength(ind: dict) -> float:
    """
    15 points — How strong is the stock relative to its own history?

    Points  Condition
    ──────  ──────────────────────────────────────────────────────────────
      5     Within 10% of 52-week high  (near highs = strength)
      5     3-month return > 5%         (positive medium-term momentum)
      5     6-month return > 10%        (strong longer trend)
    """
    pts    = 0.0
    h52pct = _safe(ind.get("pct_from_52h"), -100)   # 0 = at 52w high, negative = below
    roc3   = _safe(ind.get("roc3m"), 0)
    roc6   = _safe(ind.get("roc6m"), 0)

    if h52pct >= -10:   pts += 5    # within 10% of 52w high
    elif h52pct >= -20: pts += 2    # within 20%

    if roc3 > 5:   pts += 5
    elif roc3 > 0: pts += 2

    if roc6 > 10:   pts += 5
    elif roc6 > 0:  pts += 2

    return pts


def score_setup(ind: dict) -> float:
    """
    10 points — Is the stock in a stable, low-risk setup?

    Points  Condition
    ──────  ──────────────────────────────────────────────────────────────
      5     ATR% < 2.5% (price is stable, not wildly volatile)
      5     Bollinger Band position 40–70% (healthy mid-to-upper zone,
             not overextended or at the bottom)
    """
    pts     = 0.0
    atr_pct = _safe(ind.get("atr_pct"), 999)
    bb_pos  = _safe(ind.get("bb_pos"), 50)

    if atr_pct < 2.0:   pts += 5
    elif atr_pct < 3.0: pts += 3
    elif atr_pct < 4.5: pts += 1

    if 40 <= bb_pos <= 70:   pts += 5
    elif 30 <= bb_pos < 40:  pts += 3
    elif 70 < bb_pos <= 85:  pts += 3

    return pts


def score_stock(symbol: str, name: str, exchange: str, cap: str,
                ind: dict) -> ScoreBreakdown:
    """Compute the full score breakdown for a single stock."""
    result = ScoreBreakdown(
        symbol=symbol, name=name, exchange=exchange, cap=cap,
        price=_safe(ind.get("price"), 0),
        indicators=ind,
    )

    if not ind:
        result.error = "No indicator data"
        return result

    result.trend    = round(score_trend(ind),    2)
    result.momentum = round(score_momentum(ind), 2)
    result.volume   = round(score_volume(ind),   2)
    result.strength = round(score_strength(ind), 2)
    result.setup    = round(score_setup(ind),    2)

    return result
