"""
Plotly chart builder — TradingView-style dark theme.

Layout:
  Row 1 (68%): Candlestick + EMAs + BB + overlays + S/R + signals
               Volume bars overlaid at bottom of this panel (semi-transparent)
  Row 2 (16%): RSI
  Row 3 (16%): MACD

Rangeselector buttons (1D / 1W / 1M / 3M / 6M / 1Y / All) work correctly
because rangebreaks are only applied to intraday charts, not daily/weekly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── TradingView-inspired palette ──────────────────────────────────────────────
_BG         = "#131722"
_PANEL_BG   = "#131722"
_GRID       = "#1e2535"
_TEXT       = "#d1d4dc"
_BORDER     = "#2a2e39"

_GREEN      = "#26a69a"   # TradingView teal-green
_RED        = "#ef5350"   # TradingView red
_VOL_GREEN  = "rgba(38, 166, 154, 0.35)"
_VOL_RED    = "rgba(239, 83, 80, 0.35)"

_EMA_STYLES = {
    "ema_9":   {"color": "#f7c948", "name": "EMA 9",   "width": 1.0},
    "ema_21":  {"color": "#ff6b35", "name": "EMA 21",  "width": 1.0},
    "ema_50":  {"color": "#4fc3f7", "name": "EMA 50",  "width": 1.2},
    "ema_200": {"color": "#ce93d8", "name": "EMA 200", "width": 1.2},
}

_SR_COLORS = {
    "support":    "rgba(38, 166, 154, 0.8)",
    "resistance": "rgba(239, 83,  80, 0.8)",
    "pivot":      "rgba(180, 130, 255, 0.8)",
}


def _is_intraday(df: pd.DataFrame) -> bool:
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return False
    return df.index.resolution in ("minute", "hour")


def _rangebreaks_for(df: pd.DataFrame) -> list[dict]:
    """Only apply rangebreaks for intraday charts — avoids breaking rangeselector on daily+."""
    if not _is_intraday(df):
        return []
    return [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[10, 3.75], pattern="hour"),   # hide 22:00–09:45 UTC (outside IST market)
    ]


def build_chart(
    df: pd.DataFrame,
    indicators_config: dict,
    sr_levels: list[dict],
    signals: list[dict],
    timeframe: str = "1D",
) -> go.Figure:
    """
    Build a TradingView-style multi-panel trading chart.

    Parameters
    ----------
    df                : enriched OHLCV DataFrame (output of compute_all)
    indicators_config : dict of bool toggles, e.g. {"ema": True, "bb": True, ...}
    sr_levels         : list of {"level", "type", "label"} dicts
    signals           : list of signal dicts from compute_signals
    timeframe         : string key ("5m", "15m", "1h", "1D", "1W", …)
    """

    show_rsi  = indicators_config.get("rsi",  True) and "rsi"       in df.columns
    show_macd = indicators_config.get("macd", True) and "macd_line" in df.columns

    n_rows = 1 + int(show_rsi) + int(show_macd)
    row_heights = {1: [1.0], 2: [0.68, 0.32], 3: [0.60, 0.20, 0.20]}[n_rows]
    subplot_titles = [""] * n_rows   # no panel titles — cleaner look

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    rsi_row  = 2 if show_rsi else None
    macd_row = (3 if show_rsi else 2) if show_macd else None

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="",
        increasing=dict(line=dict(color=_GREEN, width=1), fillcolor=_GREEN),
        decreasing=dict(line=dict(color=_RED,   width=1), fillcolor=_RED),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1e2535"),
    ), row=1, col=1)

    # ── Volume (overlaid at bottom of price panel, semi-transparent) ─────────
    vol_colors = [
        _VOL_GREEN if float(c) >= float(o) else _VOL_RED
        for c, o in zip(df["Close"], df["Open"])
    ]
    # Scale volume to ~15% of price range so it sits cleanly at the bottom
    price_range = float(df["High"].max() - df["Low"].min())
    price_min   = float(df["Low"].min())
    max_vol     = float(df["Volume"].max()) if float(df["Volume"].max()) > 0 else 1
    vol_scaled  = df["Volume"] / max_vol * price_range * 0.15 + price_min

    fig.add_trace(go.Bar(
        x=df.index,
        y=(df["Volume"] / max_vol * price_range * 0.15),
        base=price_min,
        marker_color=vol_colors,
        name="Vol",
        showlegend=False,
        hovertemplate="Vol: %{customdata:,.0f}<extra></extra>",
        customdata=df["Volume"],
    ), row=1, col=1)

    # ── EMA lines ─────────────────────────────────────────────────────────────
    if indicators_config.get("ema"):
        for col, style in _EMA_STYLES.items():
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode="lines", name=style["name"],
                    line=dict(color=style["color"], width=style["width"]),
                    opacity=0.90,
                    hovertemplate=f"{style['name']}: %{{y:,.2f}}<extra></extra>",
                ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if indicators_config.get("bb") and "bb_upper" in df.columns:
        bb_color = "rgba(120, 160, 255, 0.55)"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"],
            mode="lines", name="BB",
            line=dict(color=bb_color, width=0.8),
            showlegend=True,
            hovertemplate="BB Upper: %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"],
            fill="tonexty",
            fillcolor="rgba(120, 160, 255, 0.06)",
            mode="lines", name="",
            line=dict(color=bb_color, width=0.8),
            showlegend=False,
            hovertemplate="BB Lower: %{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Supertrend ────────────────────────────────────────────────────────────
    if indicators_config.get("supertrend") and "supertrend" in df.columns:
        bull = np.where(df["st_direction"] == 1,  df["supertrend"].values, np.nan)
        bear = np.where(df["st_direction"] == -1, df["supertrend"].values, np.nan)
        fig.add_trace(go.Scatter(x=df.index, y=bull, mode="lines", name="ST ↑",
            line=dict(color=_GREEN, width=1.8), connectgaps=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bear, mode="lines", name="ST ↓",
            line=dict(color=_RED, width=1.8), connectgaps=False), row=1, col=1)

    # ── Donchian Channels ─────────────────────────────────────────────────────
    if indicators_config.get("donchian") and "dc_upper" in df.columns:
        dc_color = "rgba(255, 165, 0, 0.55)"
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_upper"], mode="lines",
            name="DC", line=dict(color=dc_color, width=0.8, dash="dot"), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_lower"], fill="tonexty",
            fillcolor="rgba(255,165,0,0.05)", mode="lines", name="",
            line=dict(color=dc_color, width=0.8, dash="dot"), showlegend=False), row=1, col=1)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    if indicators_config.get("vwap") and "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vwap"],
            mode="lines", name="VWAP",
            line=dict(color="#ff80ab", width=1.1, dash="dot"),
        ), row=1, col=1)

    # ── Support & Resistance lines ────────────────────────────────────────────
    for lv in sr_levels:
        color = _SR_COLORS.get(lv["type"], "rgba(200,200,200,0.5)")
        fig.add_hline(
            y=lv["level"],
            line=dict(color=color, width=0.8, dash="dash"),
            annotation_text=f"  {lv['label']}",
            annotation_position="right",
            annotation_font=dict(color=color, size=9),
            row=1, col=1,
        )

    # ── BUY / SELL signal markers ─────────────────────────────────────────────
    def _safe_price(date, col, offset):
        try:
            return float(df.loc[date, col]) * offset
        except KeyError:
            loc = df.index.get_indexer([date], method="nearest")[0]
            return float(df[col].iloc[loc]) * offset if loc >= 0 else np.nan

    buy_sigs  = [s for s in signals if s["type"] == "BUY"]
    sell_sigs = [s for s in signals if s["type"] == "SELL"]

    if buy_sigs:
        fig.add_trace(go.Scatter(
            x=[s["date"] for s in buy_sigs],
            y=[_safe_price(s["date"], "Low", 0.992) for s in buy_sigs],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color=_GREEN,
                        line=dict(color=_GREEN, width=1)),
            name="BUY",
            hovertext=[s["description"] for s in buy_sigs],
            hoverinfo="x+text",
        ), row=1, col=1)

    if sell_sigs:
        fig.add_trace(go.Scatter(
            x=[s["date"] for s in sell_sigs],
            y=[_safe_price(s["date"], "High", 1.008) for s in sell_sigs],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color=_RED,
                        line=dict(color=_RED, width=1)),
            name="SELL",
            hovertext=[s["description"] for s in sell_sigs],
            hoverinfo="x+text",
        ), row=1, col=1)

    # ── RSI panel ─────────────────────────────────────────────────────────────
    if show_rsi:
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)",
                      line_width=0, row=rsi_row, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.08)",
                      line_width=0, row=rsi_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            mode="lines", name="RSI",
            line=dict(color="#7986cb", width=1.2),
            showlegend=False,
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=rsi_row, col=1)
        fig.add_hline(y=70, line=dict(color=_RED,   width=0.6, dash="dash"), row=rsi_row, col=1)
        fig.add_hline(y=30, line=dict(color=_GREEN, width=0.6, dash="dash"), row=rsi_row, col=1)
        fig.add_hline(y=50, line=dict(color="#555",  width=0.4), row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70],
                         tickfont=dict(size=9), row=rsi_row, col=1)

    # ── MACD panel ────────────────────────────────────────────────────────────
    if show_macd:
        hist_colors = [_GREEN if v >= 0 else _RED for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"],
            marker_color=hist_colors,
            name="", showlegend=False, opacity=0.7,
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_line"],
            mode="lines", name="MACD",
            line=dict(color="#4fc3f7", width=1.1),
            showlegend=False,
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"],
            mode="lines", name="Signal",
            line=dict(color="#ff8f00", width=1.0, dash="dot"),
            showlegend=False,
        ), row=macd_row, col=1)
        fig.add_hline(y=0, line=dict(color="#444", width=0.6), row=macd_row, col=1)

    # ── Rangeselector buttons (only on top x-axis, syncs all panels) ─────────
    rangeselector = dict(
        buttons=[
            dict(count=1,  label="1D", step="day",   stepmode="backward"),
            dict(count=5,  label="1W", step="day",   stepmode="backward"),
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        activecolor="#2563eb",
        bgcolor="#1e2535",
        bordercolor="#2a2e39",
        borderwidth=1,
        font=dict(size=11, color=_TEXT),
        x=0, y=1.02, xanchor="left",
    )

    # ── Rangebreaks (intraday only) ───────────────────────────────────────────
    rb = _rangebreaks_for(df)

    # ── Default visible window ────────────────────────────────────────────────
    _windows = {
        "5m":  pd.Timedelta(days=1),
        "15m": pd.Timedelta(days=3),
        "1h":  pd.Timedelta(days=10),
    }
    _xrange = {}
    if timeframe in _windows and not df.index.empty:
        end   = df.index[-1]
        start = max(df.index[0], end - _windows[timeframe])
        _xrange = {"range": [str(start), str(end)]}

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        height=820,
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT, size=11, family="Inter, system-ui, sans-serif"),
        margin=dict(l=0, r=80, t=44, b=24),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right",  x=1,
            bgcolor="rgba(19,23,34,0.85)",
            bordercolor=_BORDER,
            borderwidth=1,
            font=dict(size=10, color=_TEXT),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1e2535",
            bordercolor=_BORDER,
            font=dict(color=_TEXT, size=11),
        ),
        hoverdistance=60,
        spikedistance=-1,
        # Rangeselector on the top x-axis only
        xaxis=dict(
            rangeselector=rangeselector,
            rangeslider=dict(visible=False),
            type="date",
            rangebreaks=rb,
            showgrid=True, gridcolor=_GRID, gridwidth=0.5,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#555", spikethickness=1, spikedash="dot",
            showticklabels=False,
            **_xrange,
        ),
    )

    # ── Style all axes consistently ───────────────────────────────────────────
    _yaxis_common = dict(
        showgrid=True, gridcolor=_GRID, gridwidth=0.5,
        zerolinecolor=_GRID,
        side="right",
        tickfont=dict(color="#888", size=9),
        tickformat=",.2f",
        showspikes=True, spikemode="across", spikecolor="#555",
        spikethickness=1, spikedash="dot",
    )
    _xaxis_common = dict(
        type="date",
        rangebreaks=rb,
        showgrid=True, gridcolor=_GRID, gridwidth=0.5,
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#555", spikethickness=1, spikedash="dot",
    )

    # Apply to all rows
    for row_i in range(1, n_rows + 1):
        fig.update_yaxes(**_yaxis_common, row=row_i, col=1)
        fig.update_xaxes(**_xaxis_common, row=row_i, col=1)
        if row_i < n_rows:
            fig.update_xaxes(showticklabels=False, row=row_i, col=1)

    # Bottom row x-axis shows tick labels
    fig.update_xaxes(showticklabels=True, tickfont=dict(color="#888", size=9),
                     row=n_rows, col=1)

    # Restore RSI range after bulk update
    if show_rsi:
        fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70],
                         tickformat=",.0f", row=rsi_row, col=1)

    # Sub-panel y-axis labels (small, right-side)
    if show_rsi:
        fig.update_yaxes(title_text="RSI",  title_font=dict(size=9, color="#666"),
                         title_standoff=4, row=rsi_row, col=1)
    if show_macd:
        fig.update_yaxes(title_text="MACD", title_font=dict(size=9, color="#666"),
                         title_standoff=4, tickformat=".3f", row=macd_row, col=1)

    return fig
