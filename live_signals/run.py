"""
Real-time strategy signal scanner for NSE Indian stocks.

── COMMANDS ────────────────────────────────────────────────────────────────────

  Scan once and show signals:
    python live_signals/run.py

  Continuous live mode (auto-refreshes every 15 min during market hours):
    python live_signals/run.py --watch

  Log a trade you've taken (so the scanner tracks exits for you):
    python live_signals/run.py --add SYMBOL QTY ENTRY STOP TARGET ATR
    Example:
      python live_signals/run.py --add ICICIBANK 38 1300 1241 1450 19.8

  Remove a closed position:
    python live_signals/run.py --remove ICICIBANK

  Show only your open positions:
    python live_signals/run.py --positions

  Scan a specific subset of stocks:
    python live_signals/run.py --symbols RELIANCE TCS INFY

────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import live_signals.portfolio as portfolio
from watchlist import get_watchlist
from live_signals.scanner  import run_scan
from live_signals.display  import (
    console, print_header, print_buy_signals, print_exit_signals,
    print_watch_list, print_portfolio, print_summary_line,
)

logging.basicConfig(
    level=logging.WARNING,       # suppress noisy yfinance INFO logs in live mode
    format="%(levelname)s  %(message)s",
)


def _is_market_hours() -> bool:
    from datetime import time as dtime
    now    = datetime.now()
    open_t = dtime(config.MARKET_OPEN_HOUR,  config.MARKET_OPEN_MINUTE)
    close_t= dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    return open_t <= now.time() <= close_t and now.weekday() < 5   # Mon–Fri


def do_scan(symbols: list[str], wl_source: str = ""):
    """Run one full scan cycle and print all signals."""
    positions = portfolio.load()
    scan_time = datetime.now().strftime("%d %b %Y  %H:%M:%S IST")

    console.print(f"\n  [dim]Watchlist: {wl_source}[/dim]")
    console.print(f"  [dim]Scanning {len(symbols)} symbols…[/dim]")
    results = run_scan(symbols, positions)

    buys    = results["buys"]
    exits   = results["exits"]
    watches = results["watches"]

    print_header(scan_time)
    print_buy_signals(buys)
    print_exit_signals(exits)
    print_watch_list(watches)
    print_portfolio(positions, results)
    print_summary_line(len(buys), len(exits), len(watches))

    # Auto-update trailing stops for held positions based on latest prices
    for exit_sig in exits:
        # position hit exit — remind user to close and remove
        pass
    # (Trail stop update happens inside scanner.check_exit_signal;
    #  user should call --remove after closing.)


def main():
    parser = argparse.ArgumentParser(
        description="Live NSE signal scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--watch",   action="store_true",
                        help="Continuous mode — refresh every 15 min during market hours")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Scan specific symbols instead of the full watchlist")
    parser.add_argument("--positions", action="store_true",
                        help="Show only your tracked positions and exit them")

    # Portfolio management
    parser.add_argument("--add",    nargs="+", metavar=("SYMBOL","..."),
                        help="Log a trade: --add SYMBOL QTY ENTRY STOP TARGET ATR")
    parser.add_argument("--remove", metavar="SYMBOL",
                        help="Remove a closed position from tracking")

    args = parser.parse_args()

    if args.symbols:
        symbols, wl_source = args.symbols, f"--symbols flag ({len(args.symbols)} stocks)"
    else:
        symbols, wl_source = get_watchlist(top_n=50)

    # ── Portfolio management commands ─────────────────────────────────────────
    if args.add:
        usage = "--add SYMBOL QTY ENTRY_PRICE STOP_LOSS TARGET ATR"
        if len(args.add) < 6:
            console.print(f"[red]Need 6 arguments: {usage}[/red]")
            console.print(f"  Example: --add ICICIBANK 38 1300.00 1241.00 1450.00 19.80")
            sys.exit(1)
        sym, qty, entry, stop, target, atr = args.add[:6]
        pos = portfolio.add(
            symbol      = sym.upper(),
            quantity    = int(qty),
            entry_price = float(entry),
            stop_loss   = float(stop),
            target_price= float(target),
            atr         = float(atr),
        )
        console.print(f"\n[green]✔ Added position:[/green]  "
                      f"[bold cyan]{pos['symbol']}[/bold cyan]  "
                      f"qty={pos['quantity']}  entry=₹{pos['entry_price']:.2f}  "
                      f"stop=₹{pos['stop_loss']:.2f}  target=₹{pos['target_price']:.2f}\n")
        return

    if args.remove:
        sym = args.remove.upper()
        if portfolio.remove(sym):
            console.print(f"\n[green]✔ Removed[/green] [bold cyan]{sym}[/bold cyan] from tracking.\n")
        else:
            console.print(f"\n[yellow]⚠  {sym} not found in positions.[/yellow]\n")
        return

    # ── Positions-only view ───────────────────────────────────────────────────
    if args.positions:
        positions = portfolio.load()
        if not positions:
            console.print("\n[dim]No positions tracked. Use --add to log a trade.[/dim]\n")
            return
        # Do a quick exit check for held symbols only
        held_symbols = [p["symbol"] for p in positions]
        console.print(f"\n  [dim]Checking {len(held_symbols)} held position(s)…[/dim]")
        results = run_scan(held_symbols, positions)
        print_header(datetime.now().strftime("%d %b %Y  %H:%M:%S IST"))
        print_exit_signals(results["exits"])
        print_portfolio(positions, results)
        return

    # ── Scan mode ─────────────────────────────────────────────────────────────
    if args.watch:
        console.print("\n[bold blue]Live watch mode started.[/bold blue]  "
                      "Press Ctrl+C to stop.\n")
        try:
            while True:
                do_scan(symbols, wl_source)
                interval = config.SCAN_INTERVAL_MINUTES * 60
                console.print(f"  [dim]Sleeping {config.SCAN_INTERVAL_MINUTES} min…[/dim]")
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Watch mode stopped.[/dim]\n")
    else:
        do_scan(symbols, wl_source)


if __name__ == "__main__":
    main()
