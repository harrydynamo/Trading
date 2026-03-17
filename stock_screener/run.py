"""
NSE / BSE Small, Mid & Micro-Cap Stock Screener

Fetches the LIVE official stock lists from NSE (and BSE where available),
downloads price history in parallel, scores every stock out of 100, and ranks.

── COMMANDS ────────────────────────────────────────────────────────────────────

  Score ALL stocks (midcap + smallcap + microcap):
    python stock_screener/run.py

  Filter by exchange:
    python stock_screener/run.py --exchange NSE
    python stock_screener/run.py --exchange BSE

  Filter by cap category:
    python stock_screener/run.py --cap midcap
    python stock_screener/run.py --cap smallcap
    python stock_screener/run.py --cap microcap

  Show top N results:
    python stock_screener/run.py --top 50

  Force re-fetch the stock list (ignores 24h cache):
    python stock_screener/run.py --refresh-universe

  Force re-download price data (ignores cached CSVs):
    python stock_screener/run.py --no-cache

  Tune parallel workers (default 20):
    python stock_screener/run.py --workers 30

  Combine filters:
    python stock_screener/run.py --exchange NSE --cap smallcap --top 40

────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import sys
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)

from stock_screener.universe   import get_universe, summary as universe_summary, Stock
from stock_screener.indicators import compute_all
from stock_screener.scorer     import score_stock, ScoreBreakdown
from stock_screener.report     import generate_report

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "data_cache")


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_ohlcv(yf_ticker: str, cache_dir: str | None) -> pd.DataFrame | None:
    """
    Download 5 years of daily OHLCV. Uses local cache if available.
    Cache files are CSV stored in data_cache/.
    """
    safe_name  = yf_ticker.replace(".", "_").replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{safe_name}.csv") if cache_dir else None

    try:
        if cache_path and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(yf_ticker, period="5y", interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            if cache_path and not df.empty:
                df.to_csv(cache_path)

        return df if len(df) >= 60 else None

    except Exception:
        return None


def fetch_and_score(stock: Stock, cache_dir: str | None) -> ScoreBreakdown:
    """Download data + compute indicators + score — runs inside a thread."""
    df = fetch_ohlcv(stock.yf_ticker, cache_dir)

    if df is None or df.empty:
        return ScoreBreakdown(
            symbol=stock.symbol, name=stock.name,
            exchange=stock.exchange, cap=stock.cap,
            price=0, error="No data / delisted",
        )

    ind    = compute_all(df)
    result = score_stock(
        symbol=stock.symbol, name=stock.name,
        exchange=stock.exchange, cap=stock.cap,
        ind=ind,
    )
    return result


# ─── Parallel screener ────────────────────────────────────────────────────────

def run_screener(stocks: list[Stock], cache_dir: str | None,
                 max_workers: int = 20) -> list[ScoreBreakdown]:
    """
    Score all stocks in parallel using a thread pool.
    Progress bar updates as each stock completes.
    """
    results: list[ScoreBreakdown] = []
    total = len(stocks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Scoring {total} stocks ({max_workers} parallel workers)…",
            total=total,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(fetch_and_score, stock, cache_dir): stock
                for stock in stocks
            }

            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = ScoreBreakdown(
                        symbol=stock.symbol, name=stock.name,
                        exchange=stock.exchange, cap=stock.cap,
                        price=0, error=str(e),
                    )
                results.append(result)
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]{stock.symbol:<14}[/cyan] scored",
                )

    return results


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NSE/BSE Small & Mid-Cap Stock Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exchange", choices=["NSE", "BSE"], default=None,
                        help="Filter by exchange (default: both NSE and BSE)")
    parser.add_argument("--cap",
                        choices=["midcap", "smallcap", "microcap"], default=None,
                        help="Filter by cap category (default: all)")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top stocks to display (default: 30)")
    parser.add_argument("--workers", type=int, default=20,
                        help="Parallel download workers (default: 20)")
    parser.add_argument("--refresh-universe", action="store_true",
                        help="Re-fetch stock lists from NSE/BSE (ignore 24h cache)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-download all price data (ignore cached CSVs)")
    args = parser.parse_args()

    cache_dir = None if args.no_cache else CACHE_DIR
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Build / load universe ─────────────────────────────────────────
    console.print()
    console.print("  [bold blue]NSE / BSE STOCK SCREENER[/bold blue]")
    console.print()
    console.print("  [dim]Step 1/3 — Loading stock universe…[/dim]")

    t0 = time.time()
    stocks = get_universe(
        exchange=args.exchange,
        cap=args.cap,
        force_refresh=args.refresh_universe,
    )

    all_stocks = get_universe()   # for summary counts
    console.print("  ", end="")
    universe_summary(all_stocks)

    if args.exchange or args.cap:
        console.print(
            f"  [dim]Filter: exchange={args.exchange or 'both'}  "
            f"cap={args.cap or 'all'}  →  {len(stocks)} stocks selected[/dim]"
        )

    console.print(f"  [dim]Cache: {'off (re-downloading)' if args.no_cache else CACHE_DIR}[/dim]")
    console.print()

    if not stocks:
        console.print("[red]No stocks match the selected filters.[/red]")
        sys.exit(1)

    # ── Step 2: Download + score in parallel ───────────────────────────────────
    console.print(f"  [dim]Step 2/3 — Downloading & scoring {len(stocks)} stocks "
                  f"({args.workers} parallel workers)…[/dim]")
    console.print()

    scores = run_screener(stocks, cache_dir, max_workers=args.workers)

    scored  = [s for s in scores if not s.error]
    errored = [s for s in scores if s.error]
    elapsed = time.time() - t0

    console.print(
        f"  [green]✔ Scored {len(scored)} stocks[/green]  "
        f"[dim]({len(errored)} skipped — no data or delisted)  "
        f"in {elapsed:.0f}s[/dim]"
    )
    console.print()

    # ── Step 3: Generate report ────────────────────────────────────────────────
    console.print("  [dim]Step 3/3 — Generating report…[/dim]")
    console.print()
    generate_report(scores, output_dir=RESULTS_DIR, top_n=args.top)


if __name__ == "__main__":
    main()
