"""
Buy/sell signal detection from technical indicators.

compute_signals(df) expects the enriched DataFrame from indicators.compute_all().
Returns a dict with:
  - signals:       list of signal dicts
  - current_bias:  "BULLISH" | "BEARISH" | "NEUTRAL"
  - strength_score: float (weighted)
  - regime:        "TRENDING" | "RANGING" | "MIXED"
"""

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """Returns +1 where a crosses above b, -1 where a crosses below b, 0 otherwise."""
    above_now  = a > b
    above_prev = a.shift(1) > b.shift(1)
    result = pd.Series(0, index=a.index, dtype=int)
    result[above_now  & ~above_prev] =  1
    result[~above_now &  above_prev] = -1
    return result


def _market_regime(df: pd.DataFrame) -> str:
    """
    Classify the current market regime using ADX.

    TRENDING : ADX > 25  — trend-following indicators work best
    RANGING  : ADX < 20  — oscillators work best
    MIXED    : 20–25     — no strong conviction either way
    """
    if "adx" not in df.columns:
        return "MIXED"
    val = df["adx"].iloc[-1]
    if pd.isna(val):
        return "MIXED"
    val = float(val)
    if val > 25:
        return "TRENDING"
    if val < 20:
        return "RANGING"
    return "MIXED"


# ─── Signal detectors ─────────────────────────────────────────────────────────

def _rsi_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    if "rsi" not in df.columns:
        return signals

    for i in range(-lookback, 0):
        curr = df["rsi"].iloc[i]
        prev = df["rsi"].iloc[i - 1]
        idx  = df.index[i]
        if pd.isna(curr) or pd.isna(prev):
            continue

        if prev < 30 and curr >= 30:
            signals.append({
                "type": "BUY", "indicator": "RSI",
                "description": f"RSI crossed back above 30 ({curr:.1f}) — oversold reversal",
                "strength": 2, "date": idx,
            })
        elif prev > 70 and curr <= 70:
            signals.append({
                "type": "SELL", "indicator": "RSI",
                "description": f"RSI crossed back below 70 ({curr:.1f}) — overbought reversal",
                "strength": 2, "date": idx,
            })
        elif curr < 30:
            signals.append({
                "type": "WATCH", "indicator": "RSI",
                "description": f"RSI in oversold zone ({curr:.1f})",
                "strength": 1, "date": idx,
            })
        elif curr > 70:
            signals.append({
                "type": "WATCH", "indicator": "RSI",
                "description": f"RSI in overbought zone ({curr:.1f})",
                "strength": 1, "date": idx,
            })

    return signals


def _macd_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    if "macd_line" not in df.columns or "macd_signal" not in df.columns:
        return signals

    cross = _crossover(df["macd_line"], df["macd_signal"])

    for i in range(-lookback, 0):
        idx = df.index[i]
        if cross.iloc[i] == 1:
            above_zero = df["macd_line"].iloc[i] > 0
            strength   = 3 if above_zero else 2
            signals.append({
                "type": "BUY", "indicator": "MACD",
                "description": "MACD bullish crossover" + (" (above zero line)" if above_zero else ""),
                "strength": strength, "date": idx,
            })
        elif cross.iloc[i] == -1:
            below_zero = df["macd_line"].iloc[i] < 0
            strength   = 3 if below_zero else 2
            signals.append({
                "type": "SELL", "indicator": "MACD",
                "description": "MACD bearish crossover" + (" (below zero line)" if below_zero else ""),
                "strength": strength, "date": idx,
            })

    return signals


def _ema_cross_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []

    if "ema_9" in df.columns and "ema_21" in df.columns:
        cross = _crossover(df["ema_9"], df["ema_21"])
        for i in range(-lookback, 0):
            idx = df.index[i]
            if cross.iloc[i] == 1:
                signals.append({
                    "type": "BUY", "indicator": "EMA 9/21",
                    "description": "EMA 9 crossed above EMA 21 — short-term bullish",
                    "strength": 2, "date": idx,
                })
            elif cross.iloc[i] == -1:
                signals.append({
                    "type": "SELL", "indicator": "EMA 9/21",
                    "description": "EMA 9 crossed below EMA 21 — short-term bearish",
                    "strength": 2, "date": idx,
                })

    if "ema_50" in df.columns and "ema_200" in df.columns:
        cross = _crossover(df["ema_50"], df["ema_200"])
        for i in range(-lookback, 0):
            idx = df.index[i]
            if cross.iloc[i] == 1:
                signals.append({
                    "type": "BUY", "indicator": "EMA 50/200",
                    "description": "Golden Cross: EMA 50 crossed above EMA 200",
                    "strength": 3, "date": idx,
                })
            elif cross.iloc[i] == -1:
                signals.append({
                    "type": "SELL", "indicator": "EMA 50/200",
                    "description": "Death Cross: EMA 50 crossed below EMA 200",
                    "strength": 3, "date": idx,
                })

    return signals


def _bb_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
        return signals

    for i in range(-lookback, 0):
        close = df["Close"].iloc[i]
        lower = df["bb_lower"].iloc[i]
        upper = df["bb_upper"].iloc[i]
        idx   = df.index[i]
        if pd.isna(lower) or pd.isna(upper):
            continue

        tol = (upper - lower) * 0.01
        if close <= lower + tol:
            signals.append({
                "type": "WATCH", "indicator": "BB",
                "description": f"Price at lower Bollinger Band (₹{close:,.2f}) — potential support",
                "strength": 1, "date": idx,
            })
        elif close >= upper - tol:
            signals.append({
                "type": "WATCH", "indicator": "BB",
                "description": f"Price at upper Bollinger Band (₹{close:,.2f}) — potential resistance",
                "strength": 1, "date": idx,
            })

    return signals


def _stoch_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
        return signals

    cross = _crossover(df["stoch_k"], df["stoch_d"])

    for i in range(-lookback, 0):
        idx = df.index[i]
        k   = df["stoch_k"].iloc[i]
        if pd.isna(k):
            continue

        if cross.iloc[i] == 1 and k < 30:
            signals.append({
                "type": "BUY", "indicator": "Stochastic",
                "description": f"Stochastic bullish crossover in oversold zone (K={k:.1f})",
                "strength": 2, "date": idx,
            })
        elif cross.iloc[i] == -1 and k > 70:
            signals.append({
                "type": "SELL", "indicator": "Stochastic",
                "description": f"Stochastic bearish crossover in overbought zone (K={k:.1f})",
                "strength": 2, "date": idx,
            })

    return signals


def _supertrend_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    if "st_direction" not in df.columns or "supertrend" not in df.columns:
        return signals

    for i in range(-lookback, 0):
        curr_dir = int(df["st_direction"].iloc[i])
        prev_dir = int(df["st_direction"].iloc[i - 1])
        idx      = df.index[i]
        st_val   = df["supertrend"].iloc[i]
        if pd.isna(st_val) or curr_dir == 0 or prev_dir == 0:
            continue

        if prev_dir == -1 and curr_dir == 1:
            signals.append({
                "type": "BUY", "indicator": "Supertrend",
                "description": f"Supertrend flipped BULLISH — line at {st_val:,.2f}",
                "strength": 3, "date": idx,
            })
        elif prev_dir == 1 and curr_dir == -1:
            signals.append({
                "type": "SELL", "indicator": "Supertrend",
                "description": f"Supertrend flipped BEARISH — line at {st_val:,.2f}",
                "strength": 3, "date": idx,
            })

    last_dir = int(df["st_direction"].iloc[-1]) if not pd.isna(df["st_direction"].iloc[-1]) else 0
    if last_dir == 1:
        signals.append({
            "type": "WATCH", "indicator": "Supertrend",
            "description": "Supertrend is currently BULLISH — price above the line",
            "strength": 1, "date": df.index[-1],
        })
    elif last_dir == -1:
        signals.append({
            "type": "WATCH", "indicator": "Supertrend",
            "description": "Supertrend is currently BEARISH — price below the line",
            "strength": 1, "date": df.index[-1],
        })
    return signals


def _donchian_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    if "dc_upper" not in df.columns or "dc_lower" not in df.columns:
        return signals

    tol = 0.002
    for i in range(-lookback, 0):
        close = float(df["Close"].iloc[i])
        upper = float(df["dc_upper"].iloc[i])
        lower = float(df["dc_lower"].iloc[i])
        idx   = df.index[i]
        if pd.isna(upper) or pd.isna(lower):
            continue

        if close >= upper * (1 - tol):
            signals.append({
                "type": "BUY", "indicator": "Donchian",
                "description": f"Price at 20-period Donchian upper ({upper:,.2f}) — channel breakout",
                "strength": 2, "date": idx,
            })
        elif close <= lower * (1 + tol):
            signals.append({
                "type": "SELL", "indicator": "Donchian",
                "description": f"Price at 20-period Donchian lower ({lower:,.2f}) — channel breakdown",
                "strength": 2, "date": idx,
            })
    return signals


def _candlestick_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    for i in range(-lookback, 0):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        idx        = df.index[i]
        body       = abs(c - o)
        candle_rng = h - l
        if candle_rng < 1e-9:
            continue

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if body <= candle_rng * 0.1:
            signals.append({
                "type": "WATCH", "indicator": "Candlestick",
                "description": f"Doji at {c:,.2f} — indecision, watch for direction break",
                "strength": 1, "date": idx,
            })
        elif lower_wick >= 2 * body and upper_wick <= body * 0.5 and body > 0:
            signals.append({
                "type": "BUY", "indicator": "Candlestick",
                "description": f"Hammer at {c:,.2f} — potential bullish reversal",
                "strength": 2, "date": idx,
            })
        elif upper_wick >= 2 * body and lower_wick <= body * 0.5 and body > 0:
            signals.append({
                "type": "SELL", "indicator": "Candlestick",
                "description": f"Shooting Star at {c:,.2f} — potential bearish reversal",
                "strength": 2, "date": idx,
            })
        else:
            po = float(df["Open"].iloc[i - 1])
            pc = float(df["Close"].iloc[i - 1])
            if c > o and pc < po and o < pc and c > po:
                signals.append({
                    "type": "BUY", "indicator": "Candlestick",
                    "description": f"Bullish Engulfing at {c:,.2f} — strong reversal signal",
                    "strength": 3, "date": idx,
                })
            elif c < o and pc > po and o > pc and c < po:
                signals.append({
                    "type": "SELL", "indicator": "Candlestick",
                    "description": f"Bearish Engulfing at {c:,.2f} — strong reversal signal",
                    "strength": 3, "date": idx,
                })
    return signals


def _volume_signals(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
    signals = []
    if "Volume" not in df.columns:
        return signals

    avg_vol = df["Volume"].rolling(20).mean()
    for i in range(-lookback, 0):
        vol   = float(df["Volume"].iloc[i])
        avg   = float(avg_vol.iloc[i])
        close = float(df["Close"].iloc[i])
        prev  = float(df["Close"].iloc[i - 1])
        idx   = df.index[i]
        if pd.isna(avg) or avg < 1:
            continue

        ratio = vol / avg
        if ratio >= 2.0:
            stype = "BUY" if close >= prev else "SELL"
            mood  = "bullish buying" if stype == "BUY" else "bearish selling"
            signals.append({
                "type": stype, "indicator": "Volume",
                "description": f"Volume surge {ratio:.1f}× average — strong {mood}",
                "strength": 2, "date": idx,
            })
    return signals


# ─── Weighted current-state scorer ────────────────────────────────────────────

# Weights per indicator family
_W_TREND    = 2.0   # EMA position, Supertrend
_W_MOMENTUM = 1.5   # MACD
_W_OSCILLATOR = 1.0 # RSI, Stochastic
_W_NOISE    = 0.5   # BB midline

def _current_state_score(df: pd.DataFrame, regime: str) -> float:
    """
    Weighted bias score for the current bar.

    Fixes vs v1:
    - EMA alignment bonus/penalty: stacked bull (9>21>50>200) +3, stacked bear -3
    - +DI vs -DI from ADX: directional pressure of current move (weight 2.0)
    - EMA slope: penalise if EMA21 is declining (last 5 bars)
    - Price vs 20-bar high/low: penalise if price near recent lows, reward near highs
    - Removed raw "price > EMA200" score — slow EMAs masked fast reversals
    """
    score = 0.0
    last  = df.iloc[-1]
    close = float(last["Close"])

    use_trend      = regime in ("TRENDING", "MIXED")
    use_oscillator = regime in ("RANGING",  "MIXED")

    if use_trend:
        # ── 1. EMA alignment (structure, not just price level) ─────────────
        ema_vals = {}
        for col in ("ema_9", "ema_21", "ema_50", "ema_200"):
            if col in df.columns and pd.notna(last[col]):
                ema_vals[col] = float(last[col])

        if len(ema_vals) >= 2:
            # Price vs short-term EMAs only (fast-reacting)
            for col in ("ema_9", "ema_21"):
                if col in ema_vals:
                    score += _W_TREND if close > ema_vals[col] else -_W_TREND

            # Bonus if EMAs are fully stacked (trend is clean and aligned)
            if len(ema_vals) == 4:
                e9, e21, e50, e200 = (ema_vals.get("ema_9"), ema_vals.get("ema_21"),
                                      ema_vals.get("ema_50"), ema_vals.get("ema_200"))
                if e9 and e21 and e50 and e200:
                    if e9 > e21 > e50 > e200:    # fully bullish stack
                        score += 3.0
                    elif e9 < e21 < e50 < e200:  # fully bearish stack (death alignment)
                        score -= 3.0
                    elif e21 > e50 > e200:        # partial bull stack
                        score += 1.5
                    elif e21 < e50 < e200:        # partial bear stack
                        score -= 1.5

        # ── 2. EMA21 slope (is it rising or falling?) ──────────────────────
        if "ema_21" in df.columns and len(df) >= 5:
            ema21_now  = float(df["ema_21"].iloc[-1])
            ema21_5ago = float(df["ema_21"].iloc[-5])
            if not (np.isnan(ema21_now) or np.isnan(ema21_5ago)):
                slope_pct = (ema21_now - ema21_5ago) / ema21_5ago * 100
                if   slope_pct >  0.3: score += 1.5   # clearly rising
                elif slope_pct < -0.3: score -= 1.5   # clearly falling

        # ── 3. Supertrend direction ────────────────────────────────────────
        if "st_direction" in df.columns and pd.notna(last["st_direction"]):
            d = int(last["st_direction"])
            if   d ==  1: score += _W_TREND * 2
            elif d == -1: score -= _W_TREND * 2

        # ── 4. ADX directional: +DI vs -DI ────────────────────────────────
        if "plus_di" in df.columns and "minus_di" in df.columns:
            pdi = last.get("plus_di"); mdi = last.get("minus_di")
            if pd.notna(pdi) and pd.notna(mdi):
                di_diff = float(pdi) - float(mdi)
                if   di_diff >  5: score += 2.0   # bulls clearly winning
                elif di_diff < -5: score -= 2.0   # bears clearly winning

        # ── 5. MACD ────────────────────────────────────────────────────────
        if "macd_hist" in df.columns and pd.notna(last["macd_hist"]):
            score += _W_MOMENTUM if float(last["macd_hist"]) > 0 else -_W_MOMENTUM
        if "macd_line" in df.columns and "macd_signal" in df.columns:
            ml = last.get("macd_line"); ms = last.get("macd_signal")
            if pd.notna(ml) and pd.notna(ms):
                score += _W_MOMENTUM if float(ml) > float(ms) else -_W_MOMENTUM

        # ── 6. Price vs 20-bar range (is it near the top or bottom?) ──────
        if len(df) >= 20:
            hi20  = float(df["High"].iloc[-20:].max())
            lo20  = float(df["Low"].iloc[-20:].min())
            rng20 = hi20 - lo20
            if rng20 > 0:
                pos = (close - lo20) / rng20   # 0 = at bottom, 1 = at top
                if   pos > 0.75: score += 1.0   # near recent high → bullish
                elif pos < 0.25: score -= 1.0   # near recent low  → bearish

    # ── Oscillators: RSI + Stochastic ─────────────────────────────────────
    if use_oscillator:
        if "rsi" in df.columns and pd.notna(last["rsi"]):
            rsi_v = float(last["rsi"])
            if   rsi_v > 55: score += _W_OSCILLATOR
            elif rsi_v < 45: score -= _W_OSCILLATOR
            if   rsi_v > 70: score -= _W_OSCILLATOR   # overbought penalty
            elif rsi_v < 30: score += _W_OSCILLATOR   # oversold bonus
        if "stoch_k" in df.columns and pd.notna(last["stoch_k"]):
            score += _W_OSCILLATOR if float(last["stoch_k"]) > 50 else -_W_OSCILLATOR

    # ── BB midline (always active, low weight) ─────────────────────────────
    if "bb_mid" in df.columns and pd.notna(last["bb_mid"]):
        score += _W_NOISE if close > float(last["bb_mid"]) else -_W_NOISE

    return score


# ─── Master function ───────────────────────────────────────────────────────────

def compute_signals(
    df: pd.DataFrame,
    lookback: int = 50,
    use_candlestick: bool = True,
    use_volume: bool = True,
) -> dict:
    """
    Detect all signals from the enriched DataFrame.

    Market Regime Filter:
      TRENDING → run EMA, Supertrend, MACD, Donchian, Volume
                 skip RSI and Stochastic (oscillators mislead in trends)
      RANGING  → run RSI, BB, Stochastic, Candlestick
                 skip EMA cross, Supertrend, Donchian (trend-followers whipsaw)
      MIXED    → run everything

    Returns dict with 'signals', 'current_bias', 'strength_score', 'regime'.
    """
    lb     = min(lookback, len(df) - 1)
    regime = _market_regime(df)

    # Gate detectors by regime
    if regime == "TRENDING":
        all_signals = (
            _macd_signals(df, lb)
            + _ema_cross_signals(df, lb)
            + _supertrend_signals(df, lb)
            + _donchian_signals(df, lb)
            + (_volume_signals(df, lb) if use_volume else [])
        )
    elif regime == "RANGING":
        all_signals = (
            _rsi_signals(df, lb)
            + _bb_signals(df, lb)
            + _stoch_signals(df, lb)
            + (_candlestick_signals(df, lb) if use_candlestick else [])
        )
    else:  # MIXED — run everything
        all_signals = (
            _rsi_signals(df, lb)
            + _macd_signals(df, lb)
            + _ema_cross_signals(df, lb)
            + _bb_signals(df, lb)
            + _stoch_signals(df, lb)
            + _supertrend_signals(df, lb)
            + _donchian_signals(df, lb)
            + (_candlestick_signals(df, lb) if use_candlestick else [])
            + (_volume_signals(df, lb)      if use_volume      else [])
        )

    all_signals.sort(key=lambda s: s["date"], reverse=True)

    # Deduplicate: same indicator + type + same date
    seen, deduped = set(), []
    for s in all_signals:
        key = (s["indicator"], s["type"], s["date"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    all_signals = deduped

    # Weighted bias score
    score = _current_state_score(df, regime)

    # ── Hard overrides: EMA death/golden alignment beats any score ─────────
    # If EMAs are in full bearish stack regardless of score → force BEARISH
    # This catches the case where score is marginally positive but structure is bearish
    last = df.iloc[-1]
    _hard_bias = None
    if all(c in df.columns for c in ("ema_9", "ema_21", "ema_50", "ema_200")):
        e9  = float(last["ema_9"])
        e21 = float(last["ema_21"])
        e50 = float(last["ema_50"])
        e200= float(last["ema_200"])
        close_now = float(last["Close"])
        if not any(np.isnan(v) for v in (e9, e21, e50, e200)):
            if e9 < e21 < e50 < e200 and close_now < e21:
                _hard_bias = "BEARISH"   # death alignment + price below EMA21
            elif e9 > e21 > e50 > e200 and close_now > e21:
                _hard_bias = "BULLISH"   # golden alignment + price above EMA21

    # Supertrend hard override: if Supertrend is bearish AND price is falling, don't show BULLISH
    if _hard_bias != "BEARISH" and "st_direction" in df.columns:
        st_dir = int(last["st_direction"]) if pd.notna(last["st_direction"]) else 0
        if st_dir == -1 and len(df) >= 3:
            recent_close_drop = float(df["Close"].iloc[-1]) < float(df["Close"].iloc[-3])
            if recent_close_drop and score < 6.0:
                _hard_bias = "BEARISH"

    if _hard_bias:
        bias = _hard_bias
    else:
        # Score thresholds: scaled per regime
        if regime == "TRENDING":
            bull_thr, bear_thr = 7.0, -7.0
        elif regime == "RANGING":
            bull_thr, bear_thr = 3.0, -3.0
        else:
            bull_thr, bear_thr = 5.0, -5.0

        if   score >= bull_thr: bias = "BULLISH"
        elif score <= bear_thr: bias = "BEARISH"
        else:                   bias = "NEUTRAL"

    return {
        "signals":        all_signals,
        "current_bias":   bias,
        "strength_score": score,
        "regime":         regime,
    }
