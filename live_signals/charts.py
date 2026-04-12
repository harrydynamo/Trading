"""
Candlestick chart generator for live signal stocks.

Each chart has three panels:
  1. Price  — candlesticks + Bollinger Bands + MA20 + MA50 + Weekly MA200
              + resistance line + stop-loss line + N-day low zone + entry markers
  2. Volume — bars coloured by candle direction + 20-bar volume MA
  3. RSI    — RSI(14) line with overbought/oversold bands

Usage:
  python live_signals/run.py --chart LUPIN
  python live_signals/run.py --chart LUPIN ASTRAL BSE
  python live_signals/run.py --chart-signals      # chart every current BUY signal
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from live_signals.scanner import fetch_and_precompute

logger = logging.getLogger(__name__)

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")

# ── Dark theme colours ────────────────────────────────────────────────────────
BG_COLOR       = "#0d1117"
AXES_COLOR     = "#161b22"
GRID_COLOR     = "#21262d"
TEXT_COLOR     = "#c9d1d9"
DIM_COLOR      = "#8b949e"

UP_COLOR       = "#26a641"
DOWN_COLOR     = "#e05252"

MA20_COLOR     = "#58a6ff"
MA50_COLOR     = "#d29922"
W_MA200_COLOR  = "#bc8cff"
BB_FILL_COLOR  = "#21262d"
BB_LINE_COLOR  = "#6e7681"
RESIST_COLOR   = "#f0883e"
STOP_COLOR     = "#e05252"
NDAY_ZONE_COLOR= "#2d333b"
ENTRY_COLOR    = "#26a641"

VOL_UP_COLOR   = "#1f6b34"
VOL_DOWN_COLOR = "#7c2626"
VOL_MA_COLOR   = "#8b949e"

RSI_COLOR      = "#58a6ff"
RSI_OB_COLOR   = "#e05252"
RSI_OS_COLOR   = "#26a641"


# ─── Indicator helpers (chart-only, not part of strategy precompute) ─────────

def _compute_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA50, Bollinger Bands and RSI — used only for display."""
    d = df.copy()

    d["ma50_daily"] = d["Close"].rolling(50).mean()

    rolling_std   = d["Close"].rolling(20).std()
    d["bb_upper"] = d["ma20_daily"] + 2 * rolling_std
    d["bb_lower"] = d["ma20_daily"] - 2 * rolling_std

    delta    = d["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    return d


# ─── Low-level drawing helpers ────────────────────────────────────────────────

def _draw_candlesticks(ax, df_slice: pd.DataFrame, x: np.ndarray):
    """Draw OHLCV candlesticks using integer x-positions (no weekend gaps)."""
    o = df_slice["Open"].values
    h = df_slice["High"].values
    l = df_slice["Low"].values
    c = df_slice["Close"].values

    for i in range(len(df_slice)):
        color = UP_COLOR if c[i] >= o[i] else DOWN_COLOR

        # Wick
        ax.plot([x[i], x[i]], [l[i], h[i]],
                color=color, linewidth=0.8, zorder=2)

        # Body
        body_bottom = min(o[i], c[i])
        body_height = abs(c[i] - o[i]) or (h[i] - l[i]) * 0.01
        rect = mpatches.FancyBboxPatch(
            (x[i] - 0.3, body_bottom), 0.6, body_height,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor=color,
            linewidth=0, zorder=3,
        )
        ax.add_patch(rect)


def _style_axes(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(AXES_COLOR)
        ax.tick_params(colors=DIM_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.8)
        ax.yaxis.label.set_color(DIM_COLOR)


def _set_x_ticks(ax, df_slice: pd.DataFrame, x: np.ndarray, n_ticks: int = 12):
    step   = max(1, len(x) // n_ticks)
    ticks  = x[::step]
    labels = [df_slice.index[i].strftime("%d %b '%y") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       fontsize=7, color=DIM_COLOR)


# ─── Main chart builder ───────────────────────────────────────────────────────

def draw_chart(symbol: str, df: pd.DataFrame,
               lookback_bars: int = 120,
               output_dir: str = CHARTS_DIR) -> str:
    """
    Render a 3-panel dark candlestick chart and save as PNG.
    Returns the absolute path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = _compute_chart_indicators(df)
    df_slice = df.iloc[-lookback_bars:].copy()
    # After slicing keep DatetimeIndex; reset to get integer positions
    df_slice = df_slice.reset_index(drop=False)
    date_col = df_slice.columns[0]
    df_slice = df_slice.set_index(date_col)

    x = np.arange(len(df_slice))
    n = len(df_slice)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_price, ax_vol, ax_rsi) = plt.subplots(
        3, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [5, 1.5, 1.5]},
    )
    _style_axes(fig, [ax_price, ax_vol, ax_rsi])

    last_close = float(df_slice["Close"].iloc[-1])
    last_rsi   = df_slice["rsi"].iloc[-1]
    last_rr    = df_slice["rr"].iloc[-1]
    has_signal = bool(df_slice["entry_signal"].iloc[-1])
    scan_date  = df_slice.index[-1].strftime("%d %b %Y")

    sig_label  = "[ENTRY SIGNAL]" if has_signal else ""
    sig_color  = "#26a641" if has_signal else TEXT_COLOR

    fig.suptitle(f"  {symbol}   {sig_label}",
                 fontsize=14, fontweight="bold",
                 color=sig_color, x=0.01, ha="left")
    stats = (f"Close ₹{last_close:,.2f}   "
             f"RSI {last_rsi:.1f}   "
             f"R:R {last_rr:.2f}   "
             f"{scan_date}")
    fig.text(0.01, 0.955, stats, fontsize=8, color=DIM_COLOR)

    # ── Panel 1 — Price ───────────────────────────────────────────────────────

    # Bollinger Bands (draw first — candles on top)
    bb_u = df_slice["bb_upper"].values
    bb_l = df_slice["bb_lower"].values
    valid_bb = ~(np.isnan(bb_u) | np.isnan(bb_l))
    ax_price.fill_between(x[valid_bb], bb_l[valid_bb], bb_u[valid_bb],
                          color=BB_FILL_COLOR, alpha=0.6, zorder=1)
    ax_price.plot(x[valid_bb], bb_u[valid_bb],
                  color=BB_LINE_COLOR, linewidth=0.7, linestyle="--", zorder=2)
    ax_price.plot(x[valid_bb], bb_l[valid_bb],
                  color=BB_LINE_COLOR, linewidth=0.7, linestyle="--", zorder=2,
                  label="Bollinger Bands (20,2)")

    # N-day closing low zone (where Setup A triggers)
    nday = df_slice["5d_close_min"].values
    valid_nd = ~np.isnan(nday)
    ax_price.fill_between(x[valid_nd], nday[valid_nd] * 0.985, nday[valid_nd],
                          color=NDAY_ZONE_COLOR, alpha=0.9, zorder=1,
                          label=f"{config.SETUP_A_LOW_DAYS}-day low zone")

    # Candlesticks
    _draw_candlesticks(ax_price, df_slice, x)

    # MA20 daily
    ma20 = df_slice["ma20_daily"].values
    valid = ~np.isnan(ma20)
    ax_price.plot(x[valid], ma20[valid],
                  color=MA20_COLOR, linewidth=1.1, zorder=4, label="MA20 (daily)")

    # MA50 daily
    ma50 = df_slice["ma50_daily"].values
    valid = ~np.isnan(ma50)
    ax_price.plot(x[valid], ma50[valid],
                  color=MA50_COLOR, linewidth=1.1, zorder=4, label="MA50 (daily)")

    # Weekly 200 MA
    wma200 = df_slice["w_ma200"].values
    valid  = ~np.isnan(wma200)
    ax_price.plot(x[valid], wma200[valid],
                  color=W_MA200_COLOR, linewidth=1.2, linestyle="-.", zorder=4,
                  label="MA200 (weekly)")

    # Resistance horizontal
    last_resist = df_slice["resistance"].iloc[-1]
    if pd.notna(last_resist):
        ax_price.axhline(last_resist, color=RESIST_COLOR, linewidth=1.0,
                         linestyle="--", zorder=5)
        ax_price.text(n + 0.5, last_resist,
                      f" R  ₹{last_resist:,.0f}",
                      color=RESIST_COLOR, fontsize=7, va="center")

    # Stop-loss horizontal
    last_stop = df_slice["stop"].iloc[-1]
    if pd.notna(last_stop):
        ax_price.axhline(last_stop, color=STOP_COLOR, linewidth=1.0,
                         linestyle=":", zorder=5)
        ax_price.text(n + 0.5, last_stop,
                      f" SL ₹{last_stop:,.0f}",
                      color=STOP_COLOR, fontsize=7, va="center")

    # Entry signal markers (green triangles above wick)
    sig_mask = df_slice["entry_signal"].fillna(False).astype(bool).values
    if sig_mask.any():
        entry_y = df_slice["High"].values[sig_mask] * 1.006
        ax_price.scatter(x[sig_mask], entry_y,
                         marker="^", color=ENTRY_COLOR, s=70, zorder=6,
                         label="Entry signal")

    ax_price.set_xlim(-1, n + 8)
    ax_price.yaxis.tick_right()
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax_price.set_ylabel("Price (₹)", color=DIM_COLOR)
    ax_price.legend(loc="upper left", fontsize=7, ncol=4,
                    framealpha=0.4, labelcolor=TEXT_COLOR,
                    facecolor=AXES_COLOR, edgecolor=GRID_COLOR)
    ax_price.set_xticks([])

    # ── Panel 2 — Volume ──────────────────────────────────────────────────────
    close_a = df_slice["Close"].values
    open_a  = df_slice["Open"].values
    vol_a   = df_slice["Volume"].values
    vcols   = [VOL_UP_COLOR if close_a[i] >= open_a[i] else VOL_DOWN_COLOR
               for i in range(n)]
    ax_vol.bar(x, vol_a, color=vcols, width=0.7, zorder=2)

    vol_ma = pd.Series(vol_a).rolling(20).mean().values
    valid  = ~np.isnan(vol_ma)
    ax_vol.plot(x[valid], vol_ma[valid],
                color=VOL_MA_COLOR, linewidth=0.9, zorder=3, label="Vol MA20")

    ax_vol.set_xlim(-1, n + 8)
    ax_vol.yaxis.tick_right()
    ax_vol.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    ax_vol.set_ylabel("Volume", color=DIM_COLOR)
    ax_vol.legend(loc="upper left", fontsize=7, framealpha=0.4,
                  labelcolor=TEXT_COLOR, facecolor=AXES_COLOR, edgecolor=GRID_COLOR)
    ax_vol.set_xticks([])

    # ── Panel 3 — RSI ─────────────────────────────────────────────────────────
    rsi_v = df_slice["rsi"].values
    valid = ~np.isnan(rsi_v)
    ax_rsi.plot(x[valid], rsi_v[valid], color=RSI_COLOR, linewidth=1.0, zorder=3)

    ax_rsi.axhline(70, color=RSI_OB_COLOR, linewidth=0.7, linestyle="--", alpha=0.7)
    ax_rsi.axhline(50, color=DIM_COLOR,    linewidth=0.4, linestyle="-",  alpha=0.4)
    ax_rsi.axhline(30, color=RSI_OS_COLOR, linewidth=0.7, linestyle="--", alpha=0.7)

    ax_rsi.fill_between(x[valid], rsi_v[valid], 70,
                        where=(rsi_v[valid] >= 70),
                        color=RSI_OB_COLOR, alpha=0.15, zorder=1)
    ax_rsi.fill_between(x[valid], rsi_v[valid], 30,
                        where=(rsi_v[valid] <= 30),
                        color=RSI_OS_COLOR, alpha=0.15, zorder=1)

    ax_rsi.set_ylim(5, 95)
    ax_rsi.set_yticks([30, 50, 70])
    ax_rsi.yaxis.tick_right()
    ax_rsi.set_ylabel("RSI (14)", color=DIM_COLOR)
    ax_rsi.set_xlim(-1, n + 8)
    ax_rsi.text(n + 0.5, 70, " OB", color=RSI_OB_COLOR, fontsize=6, va="center")
    ax_rsi.text(n + 0.5, 30, " OS", color=RSI_OS_COLOR, fontsize=6, va="center")

    _set_x_ticks(ax_rsi, df_slice, x)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(output_dir, f"{symbol}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return os.path.abspath(path)


# ─── Public entry point ───────────────────────────────────────────────────────

def chart_symbol(symbol: str, lookback_bars: int = 120) -> str:
    """
    Fetch data for `symbol`, compute indicators, draw chart, save PNG.
    Returns the saved file path (empty string on failure).
    """
    from rich.console import Console
    console = Console()
    console.print(f"  [dim]Fetching data for {symbol}…[/dim]")
    df = fetch_and_precompute(symbol)
    if df is None:
        console.print(f"  [red]✗ No data for {symbol} — skipped.[/red]")
        return ""
    path = draw_chart(symbol, df, lookback_bars=lookback_bars)
    console.print(f"  [green]✔ Chart saved →[/green] {path}")
    return path
