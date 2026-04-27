"""
Plotly chart builder for the live trading UI.

Layout:
  Row 1 (65%): Candlestick + EMAs + BB/overlays + S/R lines + signals
               Volume bars overlaid (semi-transparent) at bottom of price panel
  Row 2 (17%): RSI
  Row 3 (18%): MACD

Rangeselector (1D / 1W / 1M / 3M / 6M / 1Y / All) is applied AFTER all
axis updates so it is never accidentally overwritten.
rangebreaks are applied only for intraday charts to avoid conflicting with
Plotly's calendar-step rangeselector on daily/weekly/monthly data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_BG      = "#ffffff"
_PANEL   = "#fafafa"
_GRID    = "#e8e8e8"
_TEXT    = "#111111"
_GREEN   = "#1a7a3c"
_RED     = "#c0392b"
_VOL_G   = "rgba(26, 122, 60, 0.30)"
_VOL_R   = "rgba(192, 57, 43, 0.30)"

_EMA_STYLES = {
    "ema_9":   {"color": "#e67e00", "name": "EMA 9",   "width": 1.1},
    "ema_21":  {"color": "#c0392b", "name": "EMA 21",  "width": 1.1},
    "ema_50":  {"color": "#1a5aad", "name": "EMA 50",  "width": 1.3},
    "ema_200": {"color": "#6a0dad", "name": "EMA 200", "width": 1.3},
}

_SR_COLORS = {
    "support":    "rgba(26,  122,  60, 0.75)",
    "resistance": "rgba(192,  57,  43, 0.75)",
    "pivot":      "rgba(106,  13, 173, 0.75)",
}


def _is_intraday(df: pd.DataFrame) -> bool:
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return False
    return df.index.resolution in ("minute", "hour")


def build_chart(
    df: pd.DataFrame,
    indicators_config: dict,
    sr_levels: list[dict],
    signals: list[dict],
    timeframe: str = "1D",
) -> go.Figure:

    show_rsi  = indicators_config.get("rsi",  True) and "rsi"       in df.columns
    show_macd = indicators_config.get("macd", True) and "macd_line" in df.columns

    n_rows = 1 + int(show_rsi) + int(show_macd)
    row_heights = {1: [1.0], 2: [0.65, 0.35], 3: [0.58, 0.20, 0.22]}[n_rows]

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
    ), row=1, col=1)

    # ── Volume overlaid at bottom of price panel ──────────────────────────────
    vol_colors = [
        _VOL_G if float(c) >= float(o) else _VOL_R
        for c, o in zip(df["Close"], df["Open"])
    ]
    price_lo  = float(df["Low"].min())
    price_rng = float(df["High"].max() - df["Low"].min()) or 1.0
    max_vol   = float(df["Volume"].max()) or 1.0
    vol_h     = df["Volume"] / max_vol * price_rng * 0.14

    fig.add_trace(go.Bar(
        x=df.index,
        y=vol_h,
        base=price_lo,
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
                ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if indicators_config.get("bb") and "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], mode="lines", name="BB Upper",
            line=dict(color="rgba(80,80,200,0.4)", width=0.9),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], fill="tonexty",
            fillcolor="rgba(80,80,200,0.05)",
            mode="lines", name="BB Lower",
            line=dict(color="rgba(80,80,200,0.4)", width=0.9),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_mid"], mode="lines", name="BB Mid",
            line=dict(color="rgba(80,80,200,0.3)", width=0.7),
            showlegend=False,
        ), row=1, col=1)

    # ── Supertrend ────────────────────────────────────────────────────────────
    if indicators_config.get("supertrend") and "supertrend" in df.columns:
        bull = np.where(df["st_direction"] == 1,  df["supertrend"].values, np.nan)
        bear = np.where(df["st_direction"] == -1, df["supertrend"].values, np.nan)
        fig.add_trace(go.Scatter(x=df.index, y=bull, mode="lines", name="ST ↑",
            line=dict(color=_GREEN, width=2), connectgaps=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bear, mode="lines", name="ST ↓",
            line=dict(color=_RED, width=2), connectgaps=False), row=1, col=1)

    # ── Donchian Channels ─────────────────────────────────────────────────────
    if indicators_config.get("donchian") and "dc_upper" in df.columns:
        dc = "rgba(255,140,0,0.55)"
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_upper"], mode="lines", name="DC",
            line=dict(color=dc, width=0.9, dash="dot"), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_lower"], fill="tonexty",
            fillcolor="rgba(255,140,0,0.05)", mode="lines", name="",
            line=dict(color=dc, width=0.9, dash="dot"), showlegend=False), row=1, col=1)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    if indicators_config.get("vwap") and "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vwap"],
            mode="lines", name="VWAP",
            line=dict(color="#d63384", width=1.2, dash="dot"),
        ), row=1, col=1)

    # ── S/R horizontal lines ──────────────────────────────────────────────────
    for lv in sr_levels:
        color = _SR_COLORS.get(lv["type"], "rgba(150,150,150,0.5)")
        fig.add_hline(
            y=lv["level"],
            line=dict(color=color, width=0.9, dash="dash"),
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
                        line=dict(color="white", width=1)),
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
                        line=dict(color="white", width=1)),
            name="SELL",
            hovertext=[s["description"] for s in sell_sigs],
            hoverinfo="x+text",
        ), row=1, col=1)

    # ── RSI panel ─────────────────────────────────────────────────────────────
    if show_rsi:
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(192,57,43,0.07)",
                      line_width=0, row=rsi_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(26,122,60,0.07)",
                      line_width=0, row=rsi_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            mode="lines", name="RSI",
            line=dict(color="#1a5aad", width=1.2),
            showlegend=False,
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=rsi_row, col=1)
        fig.add_hline(y=70, line=dict(color=_RED,   width=0.7, dash="dash"), row=rsi_row, col=1)
        fig.add_hline(y=30, line=dict(color=_GREEN, width=0.7, dash="dash"), row=rsi_row, col=1)
        fig.add_hline(y=50, line=dict(color="#aaa",  width=0.5), row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70],
                         tickfont=dict(size=9), row=rsi_row, col=1)

    # ── MACD panel ────────────────────────────────────────────────────────────
    if show_macd:
        hist_colors = [_GREEN if v >= 0 else _RED for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
            marker_color=hist_colors, name="", showlegend=False, opacity=0.75),
            row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_line"],
            mode="lines", name="MACD",
            line=dict(color="#1a5aad", width=1.2), showlegend=False,
            hovertemplate="MACD: %{y:.3f}<extra></extra>"),
            row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
            mode="lines", name="Signal",
            line=dict(color="#e67e00", width=1.0, dash="dot"), showlegend=False),
            row=macd_row, col=1)
        fig.add_hline(y=0, line=dict(color="#bbb", width=0.6), row=macd_row, col=1)

    # ── Common axis settings ──────────────────────────────────────────────────
    # Only add rangebreaks for intraday — daily+ use plain date axis so
    # the rangeselector calendar steps (1D/1W/1M/3M/6M) work correctly.
    intraday = _is_intraday(df)
    rb = [dict(bounds=["sat", "mon"]),
          dict(bounds=[10, 3.75], pattern="hour")] if intraday else []

    _y_common = dict(
        showgrid=True, gridcolor=_GRID, gridwidth=0.5,
        zerolinecolor=_GRID,
        side="right",
        tickfont=dict(color="#555", size=9),
        tickformat=",.2f",
        showspikes=True, spikemode="across", spikecolor="#aaa",
        spikethickness=1, spikedash="dot",
    )
    _x_common = dict(
        type="date",
        showgrid=True, gridcolor=_GRID, gridwidth=0.5,
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#aaa", spikethickness=1, spikedash="dot",
    )
    # Only add rangebreaks if we have them (avoids empty-list side-effects)
    if rb:
        _x_common["rangebreaks"] = rb

    for row_i in range(1, n_rows + 1):
        fig.update_yaxes(**_y_common, row=row_i, col=1)
        fig.update_xaxes(**_x_common,
                         showticklabels=(row_i == n_rows),
                         row=row_i, col=1)

    # Restore RSI y-axis range (update_xaxes loop resets it)
    if show_rsi:
        fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70],
                         tickformat=",.0f", row=rsi_row, col=1)
    if show_macd:
        fig.update_yaxes(tickformat=".3f", row=macd_row, col=1)

    # Sub-panel labels
    if show_rsi:
        fig.update_yaxes(title_text="RSI",  title_font=dict(size=9, color="#888"),
                         title_standoff=2, row=rsi_row, col=1)
    if show_macd:
        fig.update_yaxes(title_text="MACD", title_font=dict(size=9, color="#888"),
                         title_standoff=2, row=macd_row, col=1)

    # Default visible window for intraday (daily+ shows all data)
    _xrange = {}
    _windows = {"5m": pd.Timedelta(days=1), "15m": pd.Timedelta(days=3),
                "1h": pd.Timedelta(days=10)}
    if timeframe in _windows and not df.index.empty:
        end = df.index[-1]
        start = max(df.index[0], end - _windows[timeframe])
        _xrange = {"range": [str(start), str(end)]}

    # ── Rangeselector — set LAST so it is never overwritten ──────────────────
    _rangeselector = dict(
        buttons=[
            dict(count=1,  label="1D", step="day",   stepmode="backward"),
            dict(count=5,  label="1W", step="day",   stepmode="backward"),
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        activecolor="#1a5aad",
        bgcolor="#f0f4fc",
        bordercolor="#c5d4ef",
        borderwidth=1,
        font=dict(size=11, color="#111"),
        x=0, y=1.015, xanchor="left",
    )

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        height=820,
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(color=_TEXT, size=11),
        margin=dict(l=0, r=90, t=46, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="right",  x=1,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#e0e0e0", borderwidth=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#ccc",
                        font=dict(color="#111", size=11)),
        hoverdistance=60,
        spikedistance=-1,
        # Candlestick rangeslider off
        xaxis_rangeslider_visible=False,
    )

    # Apply rangeselector to the top x-axis AFTER update_layout
    # (update_xaxes in the loop above must not touch xaxis after this point)
    rs_patch = dict(rangeselector=_rangeselector, rangeslider=dict(visible=False))
    if _xrange:
        rs_patch.update(_xrange)
    fig.update_xaxes(rs_patch, row=1, col=1)

    return fig
