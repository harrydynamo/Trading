"""
Support and resistance level detection.

All functions return a list of dicts:
    {"level": float, "type": "support"|"resistance"|"pivot", "label": str}
"""

import numpy as np
import pandas as pd


def pivot_points(df: pd.DataFrame) -> list[dict]:
    """Classic Floor Trader pivot points from the last completed bar."""
    bar       = df.iloc[-1]
    H, L, C   = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    P         = (H + L + C) / 3

    levels = [
        {"level": P,                "type": "pivot",      "label": "PP"},
        {"level": 2*P - L,          "type": "resistance", "label": "R1"},
        {"level": P + (H - L),      "type": "resistance", "label": "R2"},
        {"level": H + 2*(P - L),    "type": "resistance", "label": "R3"},
        {"level": 2*P - H,          "type": "support",    "label": "S1"},
        {"level": P - (H - L),      "type": "support",    "label": "S2"},
        {"level": L - 2*(H - P),    "type": "support",    "label": "S3"},
    ]
    return [lv for lv in levels if not np.isnan(lv["level"])]


def swing_levels(df: pd.DataFrame, window: int = 10) -> list[dict]:
    """
    Detect swing highs (resistance) and lows (support) using a rolling window,
    then cluster nearby levels to reduce noise.
    """

    def _cluster(values: np.ndarray, pct: float = 0.005) -> list[float]:
        """Merge levels within pct% of each other, return cluster means."""
        if len(values) == 0:
            return []
        values = np.sort(values)
        clusters, current = [], [values[0]]
        for v in values[1:]:
            if (v - current[-1]) / max(current[-1], 1e-9) <= pct:
                current.append(v)
            else:
                clusters.append(float(np.mean(current)))
                current = [v]
        clusters.append(float(np.mean(current)))
        return clusters

    highs = df["High"]
    lows  = df["Low"]

    # Swing high: bar where High equals the rolling max (local peak)
    roll_max = highs.rolling(window).max()
    roll_min = lows.rolling(window).min()
    swing_highs = highs[highs == roll_max].dropna().values
    swing_lows  = lows[lows  == roll_min].dropna().values

    current_price = float(df["Close"].iloc[-1])
    max_distance  = current_price * 0.20   # show only levels within 20% of current price

    res_raw = _cluster(swing_highs)
    sup_raw = _cluster(swing_lows)

    results = []
    for lvl in res_raw:
        if abs(lvl - current_price) <= max_distance:
            results.append({
                "level": round(lvl, 2),
                "type":  "resistance",
                "label": f"Swing R ₹{lvl:,.2f}",
            })
    for lvl in sup_raw:
        if abs(lvl - current_price) <= max_distance:
            results.append({
                "level": round(lvl, 2),
                "type":  "support",
                "label": f"Swing S ₹{lvl:,.2f}",
            })

    # Sort by proximity, keep top 20 to avoid chart clutter
    results.sort(key=lambda x: abs(x["level"] - current_price))
    return results[:20]


def fibonacci_levels(df: pd.DataFrame) -> list[dict]:
    """
    Fibonacci retracement levels from the 100-bar swing high and low.
    Only meaningful on daily/weekly timeframes.
    """
    lookback   = df.tail(100)
    swing_high = float(lookback["High"].max())
    swing_low  = float(lookback["Low"].min())
    diff       = swing_high - swing_low
    if diff < 1e-9:
        return []

    current_price = float(df["Close"].iloc[-1])

    FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    FIB_LABELS = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]

    results = []
    for ratio, label in zip(FIB_RATIOS, FIB_LABELS):
        level = swing_high - ratio * diff
        ltype = "support" if level <= current_price else "resistance"
        results.append({
            "level": round(level, 2),
            "type":  ltype,
            "label": f"Fib {label}  ₹{level:,.2f}",
        })
    return results
