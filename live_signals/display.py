"""
Terminal display for live signals using the Rich library.
"""

import os
import sys
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

console = Console()


def _market_status() -> str:
    now = datetime.now()
    from datetime import time as dtime
    open_t  = dtime(config.MARKET_OPEN_HOUR,  config.MARKET_OPEN_MINUTE)
    close_t = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    if open_t <= now.time() <= close_t and now.weekday() < 5:
        return "[bold green]● MARKET OPEN[/bold green]"
    return "[bold red]● MARKET CLOSED[/bold red]"


def print_header(scan_time: str):
    title = Text("  NSE LIVE SIGNAL SCANNER  ", style="bold white on dark_blue")
    subtitle = (
        f"  Scanned at {scan_time}   {_market_status()}   "
        f"Capital ₹{config.TOTAL_CAPITAL:,.0f}  "
    )
    console.print()
    console.print(Panel(title, subtitle=subtitle, box=box.DOUBLE_EDGE,
                        border_style="blue"))


def print_buy_signals(buys: list):
    console.print()
    if not buys:
        console.print(Panel(
            "[dim]No BUY signals right now — conditions not yet met.[/dim]",
            title="[bold green] 🟢 BUY SIGNALS [/bold green]",
            border_style="green",
        ))
        return

    t = Table(
        title=f"🟢  BUY SIGNALS  ({len(buys)} found)",
        box=box.SIMPLE_HEAVY, border_style="green",
        title_style="bold green", show_lines=True,
    )
    t.add_column("Symbol",    style="bold cyan",   min_width=12)
    t.add_column("Setup",     justify="center",    min_width=7)
    t.add_column("Price ₹",   justify="right",     min_width=10)
    t.add_column("Stop ₹",    justify="right",     min_width=10)
    t.add_column("Target ₹",  justify="right",     min_width=10)
    t.add_column("R : R",     justify="center",    min_width=7)
    t.add_column("ATR",       justify="right",     min_width=8)
    t.add_column("Vol Rank",  justify="center",    min_width=9)
    t.add_column("Shares @15%",justify="right",    min_width=11)
    t.add_column("Notes",     min_width=30)

    capital_per_trade = config.TOTAL_CAPITAL * config.MAX_POSITION_PCT

    for s in buys:
        shares     = s.shares_for_capital(capital_per_trade)
        rr_color   = "green" if s.reward_risk >= 3 else "yellow" if s.reward_risk >= 2 else "red"
        vrank_color= "green" if s.vol_rank >= 85 else "yellow"

        t.add_row(
            s.symbol,
            f"[bold]{'A' if s.setup == 'A' else 'B'}[/bold]",
            f"[bold]₹{s.price:,.2f}[/bold]",
            f"[red]₹{s.stop_loss:,.2f}[/red]",
            f"[green]₹{s.target:,.2f}[/green]",
            f"[{rr_color}]{s.reward_risk:.1f}x[/{rr_color}]",
            f"{s.atr:.2f}",
            f"[{vrank_color}]{s.vol_rank:.0f}%[/{vrank_color}]",
            str(shares),
            f"[dim]{s.notes}[/dim]",
        )

    console.print(t)
    console.print(
        f"  [dim]Risk per trade = price − stop × shares  |  "
        f"Max position = ₹{capital_per_trade:,.0f}  (15% of capital)[/dim]"
    )


def print_exit_signals(exits: list):
    console.print()
    if not exits:
        console.print(Panel(
            "[dim]No exit signals — all tracked positions look fine.[/dim]",
            title="[bold red] 🔴 EXIT SIGNALS [/bold red]",
            border_style="red",
        ))
        return

    t = Table(
        title=f"🔴  EXIT SIGNALS  ({len(exits)} position(s) to close)",
        box=box.SIMPLE_HEAVY, border_style="red",
        title_style="bold red", show_lines=True,
    )
    t.add_column("Symbol",       style="bold cyan", min_width=12)
    t.add_column("Current ₹",   justify="right",   min_width=10)
    t.add_column("Entry ₹",     justify="right",   min_width=10)
    t.add_column("P&L",         justify="right",   min_width=10)
    t.add_column("Exit Reason",                    min_width=40)

    for e in exits:
        pnl_pct   = e.pnl_pct * 100
        pnl_color = "green" if pnl_pct >= 0 else "red"
        pnl_str   = f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]"

        t.add_row(
            e.symbol,
            f"₹{e.price:,.2f}",
            f"₹{e.entry_price:,.2f}",
            pnl_str,
            f"[bold red]{e.reason}[/bold red]",
        )

    console.print(t)


def print_watch_list(watches: list):
    console.print()
    if not watches:
        return

    # Show top 10 only to keep the display clean
    top = watches[:10]
    t = Table(
        title=f"🟡  ON RADAR  (top {len(top)} — weekly filter passing, waiting for entry trigger)",
        box=box.SIMPLE_HEAVY, border_style="yellow",
        title_style="bold yellow", show_lines=False,
    )
    t.add_column("Symbol",    style="bold cyan", min_width=12)
    t.add_column("Price ₹",  justify="right",   min_width=10)
    t.add_column("W-MA200",  justify="right",   min_width=10)
    t.add_column("W-MA20",   justify="right",   min_width=10)
    t.add_column("Vol Rank", justify="center",  min_width=9)
    t.add_column("Vol Days", justify="center",  min_width=9)
    t.add_column("Still Waiting For", min_width=50)

    for w in top:
        vrank_color = "green" if w.vol_rank >= 80 else "yellow" if w.vol_rank >= 60 else "dim"
        t.add_row(
            w.symbol,
            f"₹{w.price:,.2f}",
            f"₹{w.w_ma200:,.2f}",
            f"₹{w.w_ma20:,.2f}",
            f"[{vrank_color}]{w.vol_rank:.0f}%[/{vrank_color}]",
            f"{w.vol_days}/3",
            f"[dim]{w.reason}[/dim]",
        )

    console.print(t)


def print_portfolio(positions: list, scan_results: dict):
    console.print()
    if not positions:
        console.print(Panel(
            "[dim]No positions tracked yet.\n"
            "Use  python live_signals/run.py --add SYMBOL QTY ENTRY STOP TARGET ATR\n"
            "to log a trade you've taken.[/dim]",
            title="[bold blue] 📋 MY POSITIONS [/bold blue]",
            border_style="blue",
        ))
        return

    # Build a quick price lookup from exit signals (have current price)
    price_map = {e.symbol: e.price for e in scan_results.get("exits", [])}

    t = Table(
        title=f"📋  MY POSITIONS  ({len(positions)} open)",
        box=box.SIMPLE_HEAVY, border_style="blue",
        title_style="bold blue", show_lines=True,
    )
    t.add_column("Symbol",     style="bold cyan", min_width=12)
    t.add_column("Setup",      justify="center",  min_width=7)
    t.add_column("Entry ₹",   justify="right",   min_width=10)
    t.add_column("Qty",        justify="right",   min_width=6)
    t.add_column("Stop ₹",    justify="right",   min_width=10)
    t.add_column("Target ₹",  justify="right",   min_width=10)
    t.add_column("Trail ₹",   justify="right",   min_width=10)
    t.add_column("Entry Date", justify="center",  min_width=12)
    t.add_column("Status",                        min_width=20)

    exit_symbols = {e.symbol for e in scan_results.get("exits", [])}

    for p in positions:
        sym    = p["symbol"]
        status = "[bold red]⚠ EXIT SIGNAL[/bold red]" if sym in exit_symbols else "[green]Holding[/green]"
        t.add_row(
            sym,
            p.get("setup", "?"),
            f"₹{p['entry_price']:,.2f}",
            str(p["quantity"]),
            f"[red]₹{p['stop_loss']:,.2f}[/red]",
            f"[green]₹{p['target_price']:,.2f}[/green]",
            f"₹{p.get('trail_stop', 0):,.2f}",
            str(p.get("entry_date", "?")),
            status,
        )

    console.print(t)


def print_summary_line(buys: int, exits: int, watches: int):
    console.print()
    console.print(
        f"  [bold green]{buys} BUY[/bold green]  |  "
        f"[bold red]{exits} EXIT[/bold red]  |  "
        f"[bold yellow]{watches} ON RADAR[/bold yellow]"
        f"  — next refresh in {config.SCAN_INTERVAL_MINUTES} min"
    )
    console.print()
