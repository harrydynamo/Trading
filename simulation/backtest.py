"""
Walk-forward backtesting engine — no lookahead bias.

How it works:
  1. Downloads full price history for every symbol in the watchlist.
  2. Pre-computes all indicators on the complete history using strictly
     backward-looking rolling windows (safe against lookahead).
  3. Steps through every trading day chronologically:
       a. Mark open positions to market.
       b. Check exit rules — hard stop, trailing stop, target, MA cross.
       c. Scan for new entry signals if capacity allows.
       d. Record end-of-day portfolio value in the equity curve.

Entry/exit prices use the day's closing price (slightly optimistic vs.
entering at next open, which is noted in results).
"""

import logging
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Allow imports from the parent Trading/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SimTrade:
    symbol: str
    setup: str
    entry_date: pd.Timestamp
    entry_price: float
    stop_loss: float
    target_price: float
    atr_at_entry: float
    quantity: int
    capital_deployed: float
    exit_date: pd.Timestamp = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trail_stop: float = 0.0


@dataclass
class BacktestResult:
    initial_capital: float
    trades: list = field(default_factory=list)          # completed SimTrades
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ─── Vectorised indicator helpers ────────────────────────────────────────────

def _hammer_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised hammer detection — returns a boolean Series."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    candle_range = h - l
    body = (c - o).abs()
    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
    return (
        (candle_range > 0)
        & (body > 0)
        & ((body / candle_range) <= 0.30)
        & (lower_shadow >= 2 * body)
        & (upper_shadow <= 0.30 * body)
    )


def _vol_rank_series(vol: pd.Series, lookback: int) -> pd.Series:
    """Rolling volume percentile rank vs past `lookback` bars (excludes today)."""
    def _pct(x):
        return (x[:-1] < x[-1]).mean() * 100 if len(x) > 1 else np.nan
    return vol.rolling(window=lookback + 1).apply(_pct, raw=True)


def precompute(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all strategy indicator columns to a daily OHLCV DataFrame.
    All computations are strictly backward-looking (no lookahead bias).
    """
    df = daily_df.copy()

    # ── ATR (14-period) ───────────────────────────────────────────────────────
    prev_c = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_c).abs(),
                    (df["Low"]  - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ── Daily 20 MA (exit rule) ───────────────────────────────────────────────
    df["ma20_daily"] = df["Close"].rolling(20).mean()

    # ── Weekly filter: resample → compute MAs → shift 1 week → ffill to daily ─
    # shift(1) ensures each day sees the PREVIOUS week's completed candle.
    weekly = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}
    )
    weekly["w_ma200"] = weekly["Close"].rolling(200).mean()
    weekly["w_ma20"]  = weekly["Close"].rolling(20).mean()
    weekly = weekly[["w_ma200", "w_ma20"]].shift(1)           # no lookahead
    df = df.join(weekly, how="left")
    df[["w_ma200", "w_ma20"]] = df[["w_ma200", "w_ma20"]].ffill()

    df["weekly_trend_ok"] = df["Close"] > df["w_ma200"]
    df["weekly_near_20"]  = (
        (df["Close"] - df["w_ma20"]).abs() / df["w_ma20"]
    ) <= config.PROXIMITY_THRESHOLD
    df["weekly_ok"] = df["weekly_trend_ok"] & df["weekly_near_20"]

    # ── Volume conditions ─────────────────────────────────────────────────────
    lookback_days = config.VOLUME_LOOKBACK_WEEKS * 5   # ~60 trading days
    df["vol_rank"]     = _vol_rank_series(df["Volume"], lookback_days)
    vol_high           = (df["vol_rank"] >= config.VOLUME_PERCENTILE).astype(int)
    df["vol_consec_ok"] = vol_high.rolling(config.VOLUME_CONSECUTIVE_DAYS).sum() \
                          == config.VOLUME_CONSECUTIVE_DAYS
    # Avoid climax volume — today must NOT be the 10-day peak
    df["vol_not_peak"] = df["Volume"] < df["Volume"].rolling(config.VOLUME_PEAK_DAYS).max()
    df["vol_ok"]       = df["vol_consec_ok"] & df["vol_not_peak"]

    # ── Setup A — N-day low pullback ──────────────────────────────────────────
    # "Price pulls back to a N-day low" = today's close is the lowest close
    # of the past N trading days (i.e. a new N-day closing low).
    df["5d_close_min"] = df["Close"].rolling(config.SETUP_A_LOW_DAYS).min()
    df["setup_a_price"]= df["Close"] <= df["5d_close_min"]
    df["setup_a"]      = df["weekly_ok"] & df["vol_ok"] & df["setup_a_price"]

    # ── Setup B — 10% drop + hammer ───────────────────────────────────────────
    close_n_ago        = df["Close"].shift(config.SETUP_B_DROP_DAYS)
    df["drop_3d"]      = (close_n_ago - df["Close"]) / close_n_ago
    df["is_hammer"]    = _hammer_series(df)
    df["setup_b"]      = (
        df["weekly_ok"]
        & df["vol_ok"]
        & (df["drop_3d"] >= config.SETUP_B_DROP_THRESHOLD)
        & df["is_hammer"]
    )

    # ── Resistance & R:R ─────────────────────────────────────────────────────
    # Use shift(1) so today's bar is excluded from resistance look-back.
    df["resistance"] = df["High"].shift(1).rolling(60).max()
    df["stop"]       = df["Close"] - config.ATR_STOP_MULTIPLIER * df["atr"]
    risk             = df["Close"] - df["stop"]
    reward           = df["resistance"] - df["Close"]
    df["rr"]         = reward / risk.replace(0, np.nan)

    # ── Final entry signal ────────────────────────────────────────────────────
    df["entry_signal"] = (
        (df["setup_a"] | df["setup_b"])
        & (df["rr"] >= config.MIN_REWARD_RISK)
        & df["stop"].notna()
        & df["resistance"].notna()
    )
    df["entry_setup"] = np.where(df["setup_a"], "A",
                        np.where(df["setup_b"], "B", ""))

    return df


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_all_data(symbols: list[str], start: str, end: str,
                  cache_dir: str = None) -> dict[str, pd.DataFrame]:
    """
    Download and pre-compute indicators for every symbol.
    Optionally cache raw CSVs in `cache_dir` to speed up re-runs.
    """
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ticker = f"{sym}.NS"
        cache_path = os.path.join(cache_dir, f"{sym}.csv") if cache_dir else None

        try:
            if cache_path and os.path.exists(cache_path):
                raw = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                logger.info(f"{sym}: loaded from cache")
            else:
                raw = yf.download(ticker, start=start, end=end,
                                  progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw.dropna(inplace=True)
                if cache_path:
                    raw.to_csv(cache_path)
                logger.info(f"{sym}: downloaded {len(raw)} rows")

            if len(raw) < 250:
                logger.warning(f"{sym}: too little history, skipping")
                continue

            data[sym] = precompute(raw)

        except Exception as e:
            logger.error(f"{sym}: failed — {e}")

    logger.info(f"Loaded {len(data)}/{len(symbols)} symbols.")
    return data


# ─── Walk-forward engine ──────────────────────────────────────────────────────

def run_backtest(data: dict[str, pd.DataFrame],
                 initial_capital: float = None,
                 active_from: str = None) -> BacktestResult:
    """
    Step through every trading day and apply the strategy.

    active_from: ISO date string (e.g. "2023-01-01"). Only days on or after
                 this date are used for trading. Earlier days are purely for
                 indicator warmup (weekly 200 MA needs ~200 weeks of history).
    """
    capital = initial_capital or config.TOTAL_CAPITAL
    result  = BacktestResult(initial_capital=capital)

    cash: float                    = capital
    open_positions: list[SimTrade] = []
    equity_records: dict           = {}

    # Build a unified calendar of all trading days across all symbols
    all_dates = sorted(set(
        date for df in data.values() for date in df.index
    ))

    # Restrict to active trading window
    if active_from:
        cutoff = pd.Timestamp(active_from)
        active_dates = [d for d in all_dates if d >= cutoff]
    else:
        # Fallback: skip first 220 bars as a minimal warmup
        active_dates = all_dates[220:] if len(all_dates) > 220 else all_dates

    if not active_dates:
        raise ValueError("No active trading dates found. Check active_from or data range.")

    logger.info(f"Running backtest: {active_dates[0].date()} → {active_dates[-1].date()}")
    logger.info(f"Initial capital: ₹{capital:,.0f}")

    for today in active_dates:

        # ── 1. Mark open positions to market ─────────────────────────────────
        mtm_value = cash
        for pos in open_positions:
            sym_data = data.get(pos.symbol)
            if sym_data is None or today not in sym_data.index:
                mtm_value += pos.capital_deployed   # use cost if no price
                continue
            today_price = float(sym_data.loc[today, "Close"])
            mtm_value  += pos.quantity * today_price

        equity_records[today] = mtm_value

        # ── 2. Check exits ────────────────────────────────────────────────────
        # The "close below daily 20 MA" exit is meant for unwinding profitable
        # trends, not for stopping out immediately at entry. We enforce a
        # MIN_HOLD_DAYS grace period before that rule activates.
        MIN_HOLD_DAYS = config.MIN_HOLD_DAYS

        still_open = []
        for pos in open_positions:
            sym_data = data.get(pos.symbol)
            if sym_data is None or today not in sym_data.index:
                still_open.append(pos)
                continue

            row         = sym_data.loc[today]
            price       = float(row["Close"])
            current_atr = float(row["atr"]) if not np.isnan(row["atr"]) else pos.atr_at_entry
            ma20        = float(row["ma20_daily"]) if not np.isnan(row["ma20_daily"]) else price

            # Ratchet up the trailing stop
            new_trail = price - config.ATR_TRAIL_MULTIPLIER * current_atr
            pos.trail_stop = max(pos.trail_stop, new_trail)

            days_held = (today - pos.entry_date).days

            exit_reason = ""
            if price <= pos.stop_loss:
                exit_reason = "Hard stop (3 ATR)"
            elif price <= pos.trail_stop and days_held >= MIN_HOLD_DAYS:
                exit_reason = "Trailing stop (2 ATR)"
            elif price >= pos.target_price:
                exit_reason = "Target hit"
            elif price < ma20 and days_held >= MIN_HOLD_DAYS:
                exit_reason = "Close below daily 20 MA"

            if exit_reason:
                pnl     = (price - pos.entry_price) * pos.quantity
                pnl_pct = (price - pos.entry_price) / pos.entry_price

                pos.exit_date   = today
                pos.exit_price  = price
                pos.exit_reason = exit_reason
                pos.pnl         = pnl
                pos.pnl_pct     = pnl_pct

                cash += pos.quantity * price
                result.trades.append(pos)

                logger.debug(f"EXIT {pos.symbol} | {exit_reason} | "
                             f"P&L ₹{pnl:+,.0f} ({pnl_pct:+.1%})")
            else:
                still_open.append(pos)

        open_positions = still_open

        # ── 3. Scan for entries ───────────────────────────────────────────────
        if len(open_positions) >= config.MAX_POSITIONS:
            continue

        deployed_pct = sum(p.capital_deployed for p in open_positions) / capital
        if deployed_pct >= config.MAX_CAPITAL_DEPLOYED:
            continue

        open_symbols = {p.symbol for p in open_positions}

        # Collect all signals firing today, rank by R:R
        signals_today = []
        for sym, df in data.items():
            if sym in open_symbols:
                continue
            if today not in df.index:
                continue
            row = df.loc[today]
            if not bool(row.get("entry_signal", False)):
                continue
            if np.isnan(row["atr"]) or np.isnan(row["stop"]) or np.isnan(row["resistance"]):
                continue
            signals_today.append((sym, row))

        # Sort best R:R first
        signals_today.sort(key=lambda x: float(x[1]["rr"]), reverse=True)

        for sym, row in signals_today:
            if len(open_positions) >= config.MAX_POSITIONS:
                break

            deployed_pct = sum(p.capital_deployed for p in open_positions) / capital
            if deployed_pct >= config.MAX_CAPITAL_DEPLOYED:
                break

            entry   = float(row["Close"])
            stop    = float(row["stop"])
            target  = float(row["resistance"])
            atr_val = float(row["atr"])
            setup   = str(row["entry_setup"])

            max_cap  = min(config.MAX_POSITION_PCT * capital, cash)
            quantity = int(max_cap // entry)
            if quantity == 0:
                continue

            cap_used = quantity * entry
            cash    -= cap_used

            trail_init = entry - config.ATR_TRAIL_MULTIPLIER * atr_val

            pos = SimTrade(
                symbol=sym, setup=setup,
                entry_date=today, entry_price=entry,
                stop_loss=stop, target_price=target,
                atr_at_entry=atr_val,
                quantity=quantity, capital_deployed=cap_used,
                trail_stop=trail_init,
            )
            open_positions.append(pos)

            logger.debug(f"ENTER {sym} Setup {setup} | qty={quantity} @ ₹{entry:.2f} | "
                         f"stop=₹{stop:.2f} target=₹{target:.2f} R:R={row['rr']:.2f}")

    # ── Force-close any still-open positions at last available price ──────────
    last_date = active_dates[-1]
    for pos in open_positions:
        sym_data = data.get(pos.symbol)
        price    = pos.entry_price  # fallback
        if sym_data is not None:
            available = sym_data.index[sym_data.index <= last_date]
            if len(available):
                price = float(sym_data.loc[available[-1], "Close"])

        pnl     = (price - pos.entry_price) * pos.quantity
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        pos.exit_date   = last_date
        pos.exit_price  = price
        pos.exit_reason = "End of backtest"
        pos.pnl         = pnl
        pos.pnl_pct     = pnl_pct
        cash += pos.quantity * price
        result.trades.append(pos)

    # ── Build equity curve & daily returns ────────────────────────────────────
    result.equity_curve  = pd.Series(equity_records).sort_index()
    result.daily_returns = result.equity_curve.pct_change().dropna()

    return result
