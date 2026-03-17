"""
Formats and exports screener results.

Outputs:
  - Rich terminal table (ranked by score)
  - results/scores.csv       — full data for every stock
  - results/scores.xlsx      — Excel with conditional formatting
  - results/top20_chart.png  — bar chart of top 20 stocks
  - results/heatmap.png      — score category heatmap
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_screener.scorer import ScoreBreakdown

console = Console()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter
    EXCEL = True
except ImportError:
    EXCEL = False


# ─── Terminal display ─────────────────────────────────────────────────────────

def _signal_style(signal: str) -> str:
    return {
        "STRONG BUY": "bold green",
        "BUY":        "green",
        "WATCH":      "yellow",
        "NEUTRAL":    "dim",
        "AVOID":      "red",
    }.get(signal, "white")


def _score_color(score: float, max_score: float) -> str:
    pct = score / max_score
    if pct >= 0.80: return "bold green"
    if pct >= 0.60: return "green"
    if pct >= 0.40: return "yellow"
    return "red"


def print_results(scores: list[ScoreBreakdown], top_n: int = 30):
    valid = [s for s in scores if not s.error]
    valid.sort(key=lambda s: s.total, reverse=True)
    show  = valid[:top_n]

    timestamp = datetime.now().strftime("%d %b %Y  %H:%M")
    console.print()
    console.print(Panel(
        f"  [bold white]NSE / BSE STOCK SCREENER[/bold white]   "
        f"[dim]{len(valid)} stocks scored  |  {timestamp}[/dim]",
        border_style="blue", box=box.DOUBLE_EDGE,
    ))
    console.print()

    t = Table(
        title=f"Top {len(show)} Stocks by Score",
        box=box.SIMPLE_HEAVY, border_style="blue",
        title_style="bold white", show_lines=False,
    )

    t.add_column("Rank",     justify="right",  min_width=5)
    t.add_column("Symbol",   style="bold cyan", min_width=12)
    t.add_column("Name",                        min_width=26)
    t.add_column("Exch/Cap", justify="center",  min_width=10)
    t.add_column("Price ₹",  justify="right",   min_width=10)
    t.add_column("Score",    justify="center",  min_width=8)
    t.add_column("Grade",    justify="center",  min_width=6)
    t.add_column("Signal",   justify="center",  min_width=12)
    t.add_column("Trend",    justify="center",  min_width=7)
    t.add_column("Momntm",   justify="center",  min_width=7)
    t.add_column("Volume",   justify="center",  min_width=7)
    t.add_column("Strength", justify="center",  min_width=8)
    t.add_column("Setup",    justify="center",  min_width=6)

    for rank, s in enumerate(show, 1):
        cap_label = f"{s.exchange}/{s.cap[:3].upper()}"
        sig_style = _signal_style(s.signal)
        tot_color = _score_color(s.total, 100)

        t.add_row(
            str(rank),
            s.symbol,
            s.name[:25],
            f"[dim]{cap_label}[/dim]",
            f"₹{s.price:,.2f}",
            f"[{tot_color}]{s.total:.0f}/100[/{tot_color}]",
            f"[{tot_color}]{s.grade}[/{tot_color}]",
            f"[{sig_style}]{s.signal}[/{sig_style}]",
            f"[{_score_color(s.trend,    30)}]{s.trend:.0f}/30[/{_score_color(s.trend,    30)}]",
            f"[{_score_color(s.momentum, 25)}]{s.momentum:.0f}/25[/{_score_color(s.momentum, 25)}]",
            f"[{_score_color(s.volume,   20)}]{s.volume:.0f}/20[/{_score_color(s.volume,   20)}]",
            f"[{_score_color(s.strength, 15)}]{s.strength:.0f}/15[/{_score_color(s.strength, 15)}]",
            f"[{_score_color(s.setup,    10)}]{s.setup:.0f}/10[/{_score_color(s.setup,    10)}]",
        )

    console.print(t)

    # Print scoring legend
    console.print(
        "  Score guide: "
        "[bold green]80–100 = A+ Strong Buy[/bold green]  "
        "[green]70–79 = A Buy[/green]  "
        "[yellow]50–69 = B/C Watch[/yellow]  "
        "[red]< 50 = Avoid[/red]"
    )
    console.print()


# ─── CSV export ───────────────────────────────────────────────────────────────

def save_csv(scores: list[ScoreBreakdown], output_dir: str):
    rows = []
    for s in sorted(scores, key=lambda x: x.total, reverse=True):
        ind = s.indicators
        rows.append({
            "Symbol":        s.symbol,
            "Name":          s.name,
            "Exchange":      s.exchange,
            "Cap":           s.cap,
            "Price (₹)":     round(s.price, 2),
            "Total Score":   round(s.total, 1),
            "Grade":         s.grade,
            "Signal":        s.signal,
            "Trend /30":     round(s.trend, 1),
            "Momentum /25":  round(s.momentum, 1),
            "Volume /20":    round(s.volume, 1),
            "Strength /15":  round(s.strength, 1),
            "Setup /10":     round(s.setup, 1),
            "RSI":           round(ind.get("rsi", np.nan), 1),
            "MACD Hist":     round(ind.get("macd_hist", np.nan), 3),
            "ATR%":          round(ind.get("atr_pct", np.nan), 2),
            "Vol Ratio":     round(ind.get("vol_ratio", np.nan), 2),
            "BB Position%":  round(ind.get("bb_pos", np.nan), 1),
            "ROC 1M%":       round(ind.get("roc1m", np.nan), 2),
            "ROC 3M%":       round(ind.get("roc3m", np.nan), 2),
            "ROC 6M%":       round(ind.get("roc6m", np.nan), 2),
            "% from 52W High": round(ind.get("pct_from_52h", np.nan), 2),
            "Above W200 MA": int(ind.get("above_w200", 0)),
            "Golden Cross":  int(ind.get("golden_cross", 0)),
            "Error":         s.error,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "scores.csv")
    df.to_csv(path, index=False)
    console.print(f"  [green]CSV saved →[/green] {path}")
    return df


# ─── Excel export ─────────────────────────────────────────────────────────────

def save_excel(df: pd.DataFrame, output_dir: str):
    if not EXCEL:
        return
    path = os.path.join(output_dir, "scores.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Scores")
        ws = writer.sheets["Scores"]

        # Header style
        header_fill = PatternFill("solid", fgColor="1F3864")
        header_font = Font(color="FFFFFF", bold=True)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        # Conditional colour for Total Score column (col F = index 6)
        score_col = 6
        for row in ws.iter_rows(min_row=2, min_col=score_col, max_col=score_col):
            for cell in row:
                try:
                    v = float(cell.value or 0)
                    if v >= 80:   color = "00B050"   # dark green
                    elif v >= 70: color = "92D050"   # light green
                    elif v >= 55: color = "FFEB9C"   # yellow
                    elif v >= 40: color = "FFCC00"   # amber
                    else:         color = "FF0000"   # red
                    cell.fill = PatternFill("solid", fgColor=color)
                except (TypeError, ValueError):
                    pass

        # Auto-fit column widths
        for col in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col) + 2
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len, 30)

    console.print(f"  [green]Excel saved →[/green] {path}")


# ─── Charts ───────────────────────────────────────────────────────────────────

def _signal_hex(signal: str) -> str:
    return {
        "STRONG BUY": "#27ae60",
        "BUY":        "#2ecc71",
        "WATCH":      "#f39c12",
        "NEUTRAL":    "#95a5a6",
        "AVOID":      "#e74c3c",
    }.get(signal, "#7f8c8d")


def save_top_chart(scores: list[ScoreBreakdown], output_dir: str, top_n: int = 20):
    if not MATPLOTLIB:
        return
    valid = sorted([s for s in scores if not s.error], key=lambda x: x.total, reverse=True)
    show  = valid[:top_n]

    labels  = [s.symbol for s in show]
    totals  = [s.total  for s in show]
    colors  = [_signal_hex(s.signal) for s in show]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(labels[::-1], totals[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)

    for bar, score in zip(bars, totals[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (out of 100)", fontsize=11)
    ax.set_title(f"Top {top_n} Stocks — NSE/BSE Screener Score", fontsize=13, fontweight="bold")
    ax.axvline(75, color="#27ae60", linestyle="--", alpha=0.5, label="Strong Buy (75)")
    ax.axvline(60, color="#f39c12", linestyle="--", alpha=0.5, label="Buy (60)")

    legend = [
        mpatches.Patch(color="#27ae60", label="Strong Buy (≥75)"),
        mpatches.Patch(color="#2ecc71", label="Buy (60–74)"),
        mpatches.Patch(color="#f39c12", label="Watch (45–59)"),
        mpatches.Patch(color="#95a5a6", label="Neutral (30–44)"),
        mpatches.Patch(color="#e74c3c", label="Avoid (<30)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "top20_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]Top-20 chart →[/green] {path}")


def save_heatmap(scores: list[ScoreBreakdown], output_dir: str, top_n: int = 40):
    if not MATPLOTLIB:
        return
    valid = sorted([s for s in scores if not s.error], key=lambda x: x.total, reverse=True)
    show  = valid[:top_n]

    data = np.array([
        [s.trend / 30, s.momentum / 25, s.volume / 20, s.strength / 15, s.setup / 10]
        for s in show
    ])
    labels_y = [s.symbol for s in show]
    labels_x = ["Trend\n/30", "Momentum\n/25", "Volume\n/20", "Strength\n/15", "Setup\n/10"]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.35)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=8)

    for i in range(len(show)):
        for j, (val, max_v) in enumerate(zip(
            [show[i].trend, show[i].momentum, show[i].volume, show[i].strength, show[i].setup],
            [30, 25, 20, 15, 10]
        )):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.3 < data[i,j] < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Score fraction (1.0 = max)")
    ax.set_title(f"Score Heatmap — Top {top_n} Stocks", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]Heatmap →[/green] {path}")


# ─── Watchlist export (used by simulation + live_signals) ────────────────────

# Shared file path — both modules import this constant to locate the file
WATCHLIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),   # Trading/
    "stock_screener", "results", "top_watchlist.json"
)

def save_top_watchlist(scores: list[ScoreBreakdown], top_n: int = 50):
    """
    Save the top N scored stocks to a shared JSON file that the simulation
    and live_signals modules read as their watchlist.
    """
    import json
    valid = [s for s in scores if not s.error and s.total > 0]
    valid.sort(key=lambda s: s.total, reverse=True)
    top = valid[:top_n]

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_n":        top_n,
        "stocks": [
            {
                "symbol":   s.symbol,
                "name":     s.name,
                "exchange": s.exchange,
                "cap":      s.cap,
                "score":    round(s.total, 1),
                "grade":    s.grade,
                "signal":   s.signal,
            }
            for s in top
        ],
    }

    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(data, f, indent=2)

    console.print(
        f"  [bold green]Watchlist saved →[/bold green] {WATCHLIST_PATH}  "
        f"[dim]({top_n} stocks — simulation & live_signals will use these)[/dim]"
    )


# ─── Master export function ───────────────────────────────────────────────────

def generate_report(scores: list[ScoreBreakdown], output_dir: str, top_n: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    print_results(scores, top_n=top_n)
    df = save_csv(scores, output_dir)
    save_excel(df, output_dir)
    save_top_chart(scores, output_dir)
    save_heatmap(scores, output_dir)
    save_top_watchlist(scores, top_n=50)   # always save top 50 for other modules
    console.print()
