"""
Performance metrics and chart generation for backtest results.

Generates:
  - Console summary table
  - trades.csv — full trade log
  - equity_curve.png — portfolio value over time with drawdown overlay
  - monthly_returns.png — heatmap of monthly P&L
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from backtest import BacktestResult, SimTrade

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no display required
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
    logger.warning("matplotlib not installed — charts will be skipped. "
                   "Install with: pip install matplotlib")


# ─── Metric calculations ──────────────────────────────────────────────────────

def _format_period_label(total_years: float) -> str:
    """Convert a decimal year count into a human-readable 'Xy Xm Xw Xd' string."""
    total_days = int(round(total_years * 365.25))
    y, rem  = divmod(total_days, 365)
    mo, rem = divmod(rem, 30)
    w, d    = divmod(rem, 7)
    parts = []
    if y:  parts.append(f"{y}y")
    if mo: parts.append(f"{mo}m")
    if w:  parts.append(f"{w}w")
    if d:  parts.append(f"{d}d")
    return " ".join(parts) if parts else "< 1d"


def compute_metrics(result: BacktestResult) -> dict:
    trades = [t for t in result.trades if t.exit_reason != "End of backtest"]
    all_trades = result.trades

    if not all_trades:
        return {"error": "No trades to analyse."}

    pnls       = [t.pnl for t in all_trades]
    winners    = [t for t in all_trades if t.pnl > 0]
    losers     = [t for t in all_trades if t.pnl <= 0]

    final_equity = float(result.equity_curve.iloc[-1]) if not result.equity_curve.empty \
                   else result.initial_capital + sum(pnls)

    total_return_pct = (final_equity - result.initial_capital) / result.initial_capital * 100

    # Annualised return (CAGR)
    if not result.equity_curve.empty:
        start = result.equity_curve.index[0]
        end   = result.equity_curve.index[-1]
        years = max((end - start).days / 365.25, 0.01)
    else:
        years = 1.0
    cagr = ((final_equity / result.initial_capital) ** (1 / years) - 1) * 100

    # Max drawdown
    eq = result.equity_curve
    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max
    max_dd      = float(drawdown.min()) * 100

    # Sharpe ratio — only meaningful when equity is moving; use annualised returns
    ann_return = cagr / 100
    rf_annual  = 0.065          # ~6.5% India risk-free
    vol        = result.daily_returns.std() * np.sqrt(252)
    sharpe     = (ann_return - rf_annual) / vol if vol > 0 else 0

    # Win rate
    win_rate = len(winners) / len(all_trades) * 100 if all_trades else 0

    avg_win  = np.mean([t.pnl for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl for t in losers])  if losers  else 0

    profit_factor = (sum(t.pnl for t in winners) /
                     abs(sum(t.pnl for t in losers))) if losers else float("inf")

    avg_rr = np.mean([t.pnl / abs((t.entry_price - t.stop_loss) * t.quantity)
                      for t in all_trades
                      if t.entry_price != t.stop_loss]) if all_trades else 0

    hold_days = [(t.exit_date - t.entry_date).days for t in all_trades
                 if t.exit_date and t.entry_date]
    avg_hold = np.mean(hold_days) if hold_days else 0

    # Return on deployed capital (ignores idle cash)
    total_deployed = sum(t.capital_deployed for t in all_trades) or 1
    total_pnl = sum(pnls)
    rodc = total_pnl / total_deployed * 100

    # Expectancy per trade (₹ expected per rupee risked)
    expectancy = (
        (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss
    )

    # Exit reason breakdown
    exit_counts: dict[str, int] = {}
    for t in all_trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # Setup breakdown
    setup_a = [t for t in all_trades if t.setup == "A"]
    setup_b = [t for t in all_trades if t.setup == "B"]

    return {
        "initial_capital":   result.initial_capital,
        "final_equity":      final_equity,
        "total_return_pct":  total_return_pct,
        "cagr_pct":          cagr,
        "max_drawdown_pct":  max_dd,
        "sharpe_ratio":      sharpe,
        "total_trades":      len(all_trades),
        "win_rate_pct":      win_rate,
        "avg_win_inr":       avg_win,
        "avg_loss_inr":      avg_loss,
        "profit_factor":     profit_factor,
        "avg_rr_achieved":   avg_rr,
        "avg_hold_days":     avg_hold,
        "return_on_deployed_capital_pct": rodc,
        "expectancy_per_trade_inr": expectancy,
        "exit_breakdown":    exit_counts,
        "setup_a_trades":    len(setup_a),
        "setup_a_win_rate":  (len([t for t in setup_a if t.pnl > 0]) / len(setup_a) * 100) if setup_a else 0,
        "setup_b_trades":    len(setup_b),
        "setup_b_win_rate":  (len([t for t in setup_b if t.pnl > 0]) / len(setup_b) * 100) if setup_b else 0,
        "backtest_years":    round(years, 2),
        "backtest_period_label": _format_period_label(years),
    }


def print_summary(metrics: dict):
    if "error" in metrics:
        print(f"\n  {metrics['error']}\n")
        return

    sep = "─" * 52
    print(f"\n{'═' * 52}")
    print(f"   BACKTEST RESULTS — Indian Stock Strategy")
    print(f"{'═' * 52}")
    print(f"  Period          : {metrics['backtest_period_label']}")
    print(f"  Initial Capital : ₹{metrics['initial_capital']:>12,.0f}")
    print(f"  Final Equity    : ₹{metrics['final_equity']:>12,.0f}")
    print(sep)
    print(f"  Total Return    : {metrics['total_return_pct']:>+10.2f}%")
    print(f"  CAGR            : {metrics['cagr_pct']:>+10.2f}%")
    print(f"  Max Drawdown    : {metrics['max_drawdown_pct']:>+10.2f}%")
    print(f"  Sharpe Ratio    :  {metrics['sharpe_ratio']:>9.2f}")
    print(sep)
    print(f"  Total Trades    : {metrics['total_trades']:>10}")
    print(f"  Win Rate        : {metrics['win_rate_pct']:>10.1f}%")
    print(f"  Avg Win         : ₹{metrics['avg_win_inr']:>11,.0f}")
    print(f"  Avg Loss        : ₹{metrics['avg_loss_inr']:>11,.0f}")
    print(f"  Profit Factor   : {metrics['profit_factor']:>10.2f}")
    print(f"  Avg R:R achieved: {metrics['avg_rr_achieved']:>10.2f}")
    print(f"  Avg Hold (days) : {metrics['avg_hold_days']:>10.1f}")
    print(f"  Expectancy/trade: ₹{metrics['expectancy_per_trade_inr']:>10,.0f}")
    print(sep)
    print(f"  Return on deployed capital: {metrics['return_on_deployed_capital_pct']:>+.2f}%")
    print(f"  (Total return low because strategy is mostly in cash)")
    print(sep)
    print(f"  Setup A trades  : {metrics['setup_a_trades']:>4}  (win rate {metrics['setup_a_win_rate']:.1f}%)")
    print(f"  Setup B trades  : {metrics['setup_b_trades']:>4}  (win rate {metrics['setup_b_win_rate']:.1f}%)")
    print(sep)
    print("  Exit reasons:")
    for reason, count in sorted(metrics["exit_breakdown"].items(),
                                 key=lambda x: -x[1]):
        print(f"    {reason:<28}: {count}")
    print(f"{'═' * 52}\n")


# ─── CSV export ───────────────────────────────────────────────────────────────

def save_trades_csv(result: BacktestResult, output_dir: str):
    if not result.trades:
        logger.info("No trades to save.")
        return
    rows = []
    for t in result.trades:
        rows.append({
            "Symbol":         t.symbol,
            "Setup":          t.setup,
            "Entry Date":     t.entry_date.date() if t.entry_date else "",
            "Entry Price":    round(t.entry_price, 2),
            "Exit Date":      t.exit_date.date() if t.exit_date else "",
            "Exit Price":     round(t.exit_price, 2),
            "Stop Loss":      round(t.stop_loss, 2),
            "Target":         round(t.target_price, 2),
            "Quantity":       t.quantity,
            "Capital (₹)":    round(t.capital_deployed, 2),
            "P&L (₹)":        round(t.pnl, 2),
            "P&L (%)":        round(t.pnl_pct * 100, 2),
            "ATR at Entry":   round(t.atr_at_entry, 2),
            "Exit Reason":    t.exit_reason,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "trades.csv")
    df.to_csv(path, index=False)
    print(f"  Trade log saved → {path}")


# ─── Charts ───────────────────────────────────────────────────────────────────

def _plot_equity_curve(result: BacktestResult, output_dir: str):
    eq = result.equity_curve
    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.suptitle("Strategy Equity Curve — NSE India", fontsize=14, fontweight="bold")

    # Equity
    ax1.plot(eq.index, eq.values / 1e5, color="#1f77b4", linewidth=1.5, label="Portfolio Value")
    ax1.axhline(result.initial_capital / 1e5, color="gray", linestyle="--",
                linewidth=0.8, label="Initial Capital")
    ax1.fill_between(eq.index, result.initial_capital / 1e5, eq.values / 1e5,
                     where=eq.values >= result.initial_capital,
                     alpha=0.15, color="green")
    ax1.fill_between(eq.index, result.initial_capital / 1e5, eq.values / 1e5,
                     where=eq.values < result.initial_capital,
                     alpha=0.15, color="red")
    ax1.set_ylabel("Portfolio Value (₹ Lakh)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}L"))
    ax1.grid(alpha=0.3)

    # Drawdown
    ax2.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.4)
    ax2.plot(drawdown.index, drawdown.values, color="darkred", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity curve saved → {path}")


def _plot_monthly_returns(result: BacktestResult, output_dir: str):
    if result.equity_curve.empty:
        return

    monthly = result.equity_curve.resample("M").last().pct_change().dropna() * 100
    if monthly.empty:
        return

    df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Ret":   monthly.values,
    })
    pivot = df.pivot_table(index="Year", columns="Month", values="Ret", aggfunc="sum")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6 + 1)))
    fig.suptitle("Monthly Returns (%)", fontsize=13, fontweight="bold")

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    im   = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                     vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Return (%)")
    plt.tight_layout()
    path = os.path.join(output_dir, "monthly_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Monthly returns saved → {path}")


def _plot_trade_distribution(result: BacktestResult, output_dir: str):
    if not result.trades:
        return

    pnl_pcts = [t.pnl_pct * 100 for t in result.trades]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trade Distribution", fontsize=13, fontweight="bold")

    # P&L histogram
    colors = ["#e74c3c" if p < 0 else "#2ecc71" for p in pnl_pcts]
    axes[0].bar(range(len(pnl_pcts)), sorted(pnl_pcts), color=sorted(colors))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Individual Trade Returns")
    axes[0].set_xlabel("Trade #")
    axes[0].set_ylabel("Return (%)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Exit reason pie
    exit_counts = {}
    for t in result.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    axes[1].pie(exit_counts.values(), labels=exit_counts.keys(),
                autopct="%1.0f%%", startangle=140,
                colors=["#e74c3c","#f39c12","#2ecc71","#3498db","#9b59b6"])
    axes[1].set_title("Exit Reasons")

    plt.tight_layout()
    path = os.path.join(output_dir, "trade_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Trade distribution saved → {path}")


def generate_report(result: BacktestResult, output_dir: str):
    """Main entry point — print metrics, save CSV, render all charts."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = compute_metrics(result)
    print_summary(metrics)
    save_trades_csv(result, output_dir)

    if MATPLOTLIB:
        _plot_equity_curve(result, output_dir)
        _plot_monthly_returns(result, output_dir)
        _plot_trade_distribution(result, output_dir)
    else:
        print("  (Install matplotlib to generate charts: pip install matplotlib)")

    # Save metrics to a text file too
    metrics_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"  Metrics summary saved → {metrics_path}\n")
