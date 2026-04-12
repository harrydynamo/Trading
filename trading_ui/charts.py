"""
Plotly chart builder for the live trading UI.

build_chart(df, indicators_config, sr_levels, signals) → plotly Figure

Layout (4 rows):
  Row 1 (60%): Candlestick + EMA/BB/VWAP overlays + S/R lines + signal markers
  Row 2 (12%): Volume
  Row 3 (14%): RSI
  Row 4 (14%): MACD
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_BG       = "#ffffff"
_GRID     = "#e8e8e8"
_TEXT     = "#111111"
_GREEN    = "#1a7a3c"
_RED      = "#c0392b"

_ST_BULL    = "#1a7a3c"
_ST_BEAR    = "#c0392b"
_DONCHIAN   = "rgba(255, 140, 0, 0.60)"

_EMA_STYLES = {
    "ema_9":   {"color": "#e67e00", "name": "EMA 9",   "width": 1.2},
    "ema_21":  {"color": "#c0392b", "name": "EMA 21",  "width": 1.2},
    "ema_50":  {"color": "#1a5aad", "name": "EMA 50",  "width": 1.5},
    "ema_200": {"color": "#6a0dad", "name": "EMA 200", "width": 1.5},
}

_SR_COLORS = {
    "support":    "rgba(38,  166,  65, 0.7)",
    "resistance": "rgba(224,  82,  82, 0.7)",
    "pivot":      "rgba(147,  51, 234, 0.7)",
}


def _rangebreaks_for(df: pd.DataFrame) -> list[dict]:
    """
    Return Plotly rangebreaks that hide non-trading periods.
    With type='date' x-axis, rangebreaks keep the chart gap-free while
    still allowing time-based rangeselector buttons (1D / 1W / 1M …) to work.
    """
    breaks = [dict(bounds=["sat", "mon"])]   # always hide weekends

    # For intraday data also hide outside market hours (09:15–15:30 IST = 03:45–10:00 UTC)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.resolution in ("minute", "hour"):
        breaks.append(dict(bounds=[10, 3.75], pattern="hour"))   # 10:00–03:45 UTC gap

    return breaks


def _default_xrange(df: pd.DataFrame, timeframe: str) -> tuple[str, str] | None:
    """
    Return (x_start, x_end) strings so lower timeframes open at a readable zoom level.
    The rangeselector buttons let the user zoom out further.

    Default visible window per timeframe:
      5m  → last 1 trading day   (~78 bars)
      15m → last 3 trading days  (~75 bars)
      1h  → last 10 trading days (~65 bars)
      1D  → last 6 months
      1W  → all data (no restriction)
    """
    if df.index.empty:
        return None

    end = df.index[-1]

    windows = {
        "5m":  pd.Timedelta(days=1),
        "15m": pd.Timedelta(days=3),
        "1h":  pd.Timedelta(days=10),
        "1D":  pd.Timedelta(days=180),
    }
    delta = windows.get(timeframe)
    if delta is None:
        return None

    start = max(df.index[0], end - delta)
    return str(start), str(end)


def build_chart(
    df: pd.DataFrame,
    indicators_config: dict,
    sr_levels: list[dict],
    signals: list[dict],
    timeframe: str = "1D",
) -> go.Figure:
    """
    Build and return a complete multi-panel trading chart as a Plotly Figure.

    Parameters
    ----------
    df                : enriched OHLCV DataFrame (output of compute_all)
    indicators_config : dict of bool toggles, e.g. {"ema": True, "bb": True, ...}
    sr_levels         : list of {"level", "type", "label"} dicts
    signals           : list of signal dicts from compute_signals
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.60, 0.12, 0.14, 0.14],
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
        increasing_fillcolor=_GREEN,
        decreasing_fillcolor=_RED,
        showlegend=False,
        line=dict(width=1),
    ), row=1, col=1)

    # ── EMA lines ─────────────────────────────────────────────────────────────
    if indicators_config.get("ema"):
        for col, style in _EMA_STYLES.items():
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode="lines", name=style["name"],
                    line=dict(color=style["color"], width=style["width"]),
                    opacity=0.85,
                ), row=1, col=1)

    # ── Bollinger Bands (shaded) ───────────────────────────────────────────────
    if indicators_config.get("bb") and "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"],
            mode="lines", name="BB Upper",
            line=dict(color="rgba(80,80,200,0.45)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"],
            mode="lines", name="BB Lower",
            fill="tonexty",
            fillcolor="rgba(80,80,200,0.05)",
            line=dict(color="rgba(80,80,200,0.45)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_mid"],
            mode="lines", name="BB Mid",
            line=dict(color="rgba(80,80,200,0.35)", width=0.8),
            showlegend=False,
        ), row=1, col=1)

    # ── Supertrend ────────────────────────────────────────────────────────────
    if indicators_config.get("supertrend") and "supertrend" in df.columns:
        bull_st = np.where(df["st_direction"] == 1,  df["supertrend"].values, np.nan)
        bear_st = np.where(df["st_direction"] == -1, df["supertrend"].values, np.nan)
        fig.add_trace(go.Scatter(
            x=df.index, y=bull_st,
            mode="lines", name="ST Bull",
            line=dict(color=_ST_BULL, width=2.2),
            connectgaps=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bear_st,
            mode="lines", name="ST Bear",
            line=dict(color=_ST_BEAR, width=2.2),
            connectgaps=False,
        ), row=1, col=1)

    # ── Donchian Channels ─────────────────────────────────────────────────────
    if indicators_config.get("donchian") and "dc_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["dc_upper"],
            mode="lines", name="DC Upper",
            line=dict(color=_DONCHIAN, width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["dc_lower"],
            mode="lines", name="DC Lower",
            fill="tonexty",
            fillcolor="rgba(255,140,0,0.04)",
            line=dict(color=_DONCHIAN, width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["dc_mid"],
            mode="lines", name="DC Mid",
            line=dict(color=_DONCHIAN, width=0.7),
            showlegend=False,
        ), row=1, col=1)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    if indicators_config.get("vwap") and "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vwap"],
            mode="lines", name="VWAP",
            line=dict(color="#d63384", width=1.3, dash="dot"),
        ), row=1, col=1)

    # ── Support & Resistance lines ────────────────────────────────────────────
    for lv in sr_levels:
        color = _SR_COLORS.get(lv["type"], "rgba(150,150,150,0.5)")
        fig.add_hline(
            y=lv["level"],
            line=dict(color=color, width=1, dash="dash"),
            annotation_text=lv["label"],
            annotation_position="right",
            annotation_font=dict(color=color, size=9),
            row=1, col=1,
        )

    # ── Buy / Sell signal markers ─────────────────────────────────────────────
    buy_signals  = [s for s in signals if s["type"] == "BUY"]
    sell_signals = [s for s in signals if s["type"] == "SELL"]

    def _safe_price(date, col, offset):
        try:
            return float(df.loc[date, col]) * offset
        except KeyError:
            # Find nearest index entry
            loc = df.index.get_indexer([date], method="nearest")[0]
            if loc >= 0:
                return float(df[col].iloc[loc]) * offset
            return np.nan

    if buy_signals:
        bdates  = [s["date"] for s in buy_signals]
        bprices = [_safe_price(d, "Low", 0.994) for d in bdates]
        btexts  = [s["description"] for s in buy_signals]
        fig.add_trace(go.Scatter(
            x=bdates, y=bprices,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=13, color=_GREEN,
                        line=dict(color="white", width=1)),
            text=["▲"] * len(bdates),
            textposition="bottom center",
            textfont=dict(color=_GREEN, size=9),
            name="BUY Signal",
            hovertext=btexts,
            hoverinfo="x+text",
        ), row=1, col=1)

    if sell_signals:
        sdates  = [s["date"] for s in sell_signals]
        sprices = [_safe_price(d, "High", 1.006) for d in sdates]
        stexts  = [s["description"] for s in sell_signals]
        fig.add_trace(go.Scatter(
            x=sdates, y=sprices,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=13, color=_RED,
                        line=dict(color="white", width=1)),
            text=["▼"] * len(sdates),
            textposition="top center",
            textfont=dict(color=_RED, size=9),
            name="SELL Signal",
            hovertext=stexts,
            hoverinfo="x+text",
        ), row=1, col=1)

    # ── Row 2: Volume ─────────────────────────────────────────────────────────
    vol_colors = [
        _GREEN if float(c) >= float(o) else _RED
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors,
        name="Volume",
        showlegend=False,
        opacity=0.8,
    ), row=2, col=1)

    # ── Row 3: RSI ────────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            mode="lines", name="RSI (14)",
            line=dict(color="#1a5aad", width=1.3),
            showlegend=False,
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, line_width=0,
                      fillcolor="rgba(224,82,82,0.10)", row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  line_width=0,
                      fillcolor="rgba(38,166,65,0.10)", row=3, col=1)
        fig.add_hline(y=70, line=dict(color=_RED,   width=0.8, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color=_GREEN, width=0.8, dash="dash"), row=3, col=1)
        fig.add_hline(y=50, line=dict(color="#555",  width=0.5), row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    # ── Row 4: MACD ───────────────────────────────────────────────────────────
    if "macd_line" in df.columns:
        hist_colors = [
            _GREEN if (v >= 0) else _RED
            for v in df["macd_hist"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"],
            marker_color=hist_colors,
            name="MACD Hist",
            showlegend=False,
            opacity=0.75,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_line"],
            mode="lines", name="MACD",
            line=dict(color="#1a5aad", width=1.3),
            showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"],
            mode="lines", name="Signal",
            line=dict(color="#FF8C00", width=1.1, dash="dot"),
            showlegend=False,
        ), row=4, col=1)
        fig.add_hline(y=0, line=dict(color="#555", width=0.6), row=4, col=1)

    # ── Row labels (y-axis titles) ────────────────────────────────────────────
    fig.update_yaxes(title_text="Price",  title_font=dict(size=10), row=1, col=1)
    fig.update_yaxes(title_text="Vol",    title_font=dict(size=10), row=2, col=1)
    fig.update_yaxes(title_text="RSI",    title_font=dict(size=10), row=3, col=1)
    fig.update_yaxes(title_text="MACD",   title_font=dict(size=10), row=4, col=1)

    # ── Timeframe range-selector buttons (all panels sync via shared_xaxes) ──
    _rangeselector = dict(
        buttons=[
            dict(count=1,  label="1D",  step="day",   stepmode="backward"),
            dict(count=5,  label="1W",  step="day",   stepmode="backward"),
            dict(count=1,  label="1M",  step="month", stepmode="backward"),
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(count=6,  label="6M",  step="month", stepmode="backward"),
            dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        activecolor="#1a5aad",
        bgcolor="#f0f4fc",
        bordercolor="#c5d4ef",
        borderwidth=1,
        font=dict(size=11, color="#111"),
        x=0,
        y=1.015,
        xanchor="left",
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=860,
        paper_bgcolor=_BG,
        plot_bgcolor="#fafafa",
        font=dict(color=_TEXT, size=11),
        margin=dict(l=10, r=100, t=48, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.055,
            xanchor="right",  x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e0e0e0",
            borderwidth=1,
            font=dict(size=10, color="#111"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#ccc",
            font=dict(color="#111", size=11),
        ),
        hoverdistance=50,
        spikedistance=-1,
        # Range selector on top x-axis (row 1); all panels follow via shared_xaxes
        xaxis=dict(
            rangeselector=_rangeselector,
            rangeslider=dict(visible=False),
            **( {"range": list(_default_xrange(df, timeframe))}
                if _default_xrange(df, timeframe) else {} ),
        ),
    )

    # ── Axes styling + crosshair spike lines ─────────────────────────────────
    _spike = dict(
        showspikes=True,
        spikemode="across+toaxis",
        spikesnap="cursor",
        spikecolor="#aaa",
        spikethickness=1,
        spikedash="dot",
    )
    _ygrid = dict(
        showgrid=True, gridcolor=_GRID, gridwidth=0.5,
        zerolinecolor=_GRID, zerolinewidth=1,
        side="right",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#aaa",
        spikethickness=1,
        spikedash="dot",
        tickformat=",.2f",
        tickfont=dict(color="#555"),
    )

    rb = _rangebreaks_for(df)

    for row_i in range(1, 5):
        fig.update_xaxes(
            **_spike,
            type="date",            # ← force date mode so rangeselector works
            rangebreaks=rb,         # ← hide weekends / off-hours (keeps chart clean)
            showgrid=True,
            gridcolor=_GRID,
            gridwidth=0.5,
            zerolinecolor=_GRID,
            row=row_i, col=1,
        )
        fig.update_yaxes(**_ygrid, row=row_i, col=1)

    # Constrain RSI y-axis
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Hide x-tick labels on all panels except the bottom one
    for row_i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=row_i, col=1)

    return fig
