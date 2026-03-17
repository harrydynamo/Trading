"""
Run the backtest simulation.

Usage:
    python simulation/run.py                                    # default: 3 years
    python simulation/run.py --years 2 --months 6              # 2 years 6 months
    python simulation/run.py --months 9                         # 9 months only
    python simulation/run.py --weeks 10                         # 10 weeks only
    python simulation/run.py --days 45                          # 45 days only
    python simulation/run.py --years 1 --months 3 --weeks 2    # combined
    python simulation/run.py --capital 1000000                  # ₹10L capital
    python simulation/run.py --symbols RELIANCE TCS             # specific stocks
    python simulation/run.py --no-cache                         # force re-download

Results are saved to simulation/results/
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

# Allow imports from the parent Trading/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from simulation.backtest import load_all_data, run_backtest
from simulation.report import generate_report
from watchlist import get_watchlist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "data_cache")


def parse_period(years: int, months: int, weeks: int, days: int) -> int:
    """Convert years/months/weeks/days into a total number of calendar days."""
    return (years * 365) + (months * 30) + (weeks * 7) + days


def format_period(years: int, months: int, weeks: int, days: int) -> str:
    """Human-readable period string, e.g. '2y 6m 3w 5d'."""
    parts = []
    if years:  parts.append(f"{years}y")
    if months: parts.append(f"{months}m")
    if weeks:  parts.append(f"{weeks}w")
    if days:   parts.append(f"{days}d")
    return " ".join(parts) if parts else "0d"


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the Indian stock strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Period examples:\n"
            "  --years 3                    → 3 years\n"
            "  --years 1 --months 6         → 1 year 6 months\n"
            "  --months 9                   → 9 months\n"
            "  --weeks 10                   → 10 weeks\n"
            "  --days 90                    → 90 days\n"
            "  --years 2 --months 3 --days 15 → 2 years 3 months 15 days"
        ),
    )
    parser.add_argument("--years",    type=int,   default=0,
                        help="Number of years in the backtest window")
    parser.add_argument("--months",   type=int,   default=0,
                        help="Number of months in the backtest window")
    parser.add_argument("--weeks",    type=int,   default=0,
                        help="Number of weeks in the backtest window")
    parser.add_argument("--days",     type=int,   default=0,
                        help="Number of days in the backtest window")
    parser.add_argument("--capital",  type=float, default=config.TOTAL_CAPITAL,
                        help=f"Starting capital in INR (default: {config.TOTAL_CAPITAL:,.0f})")
    parser.add_argument("--symbols",  nargs="+",  default=None,
                        help="Override watchlist with specific NSE symbols")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download all data (ignore cache)")
    args = parser.parse_args()

    # Default to 3 years if no period given
    if not any([args.years, args.months, args.weeks, args.days]):
        args.years = 3

    total_days   = parse_period(args.years, args.months, args.weeks, args.days)
    period_label = format_period(args.years, args.months, args.weeks, args.days)

    if total_days < 30:
        print("Error: backtest period must be at least 30 days.")
        sys.exit(1)

    if args.symbols:
        symbols, wl_source = args.symbols, f"--symbols flag ({len(args.symbols)} stocks)"
    else:
        symbols, wl_source = get_watchlist(top_n=50)
    end_date = datetime.today().strftime("%Y-%m-%d")

    # The weekly 200 MA needs ~200 weeks (~4 years) of history to warm up.
    # Always download (period + 4.5 years) of data; only trade the active window.
    WARMUP_DAYS = int(4.5 * 365)
    data_start   = (datetime.today() - timedelta(days=total_days + WARMUP_DAYS)).strftime("%Y-%m-%d")
    active_start = (datetime.today() - timedelta(days=total_days)).strftime("%Y-%m-%d")

    cache_dir = None if args.no_cache else CACHE_DIR
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'═' * 52}")
    print(f"  INDIAN STOCK STRATEGY — SIMULATION")
    print(f"{'═' * 52}")
    from rich.console import Console as _C; _rc = _C()
    print(f"  Period        : {period_label}  ({active_start} → {end_date})")
    print(f"  Data download : {data_start} → {end_date} (warmup included)")
    print(f"  Symbols       : {len(symbols)}")
    print(f"  Capital       : ₹{args.capital:,.0f}")
    print(f"  Cache         : {'disabled' if args.no_cache else CACHE_DIR}")
    _rc.print(f"  Watchlist     : {wl_source}")
    print(f"{'═' * 52}\n")

    # ── Step 1: Load and precompute data ──────────────────────────────────────
    logger.info("Downloading / loading historical data…")
    data = load_all_data(symbols, start=data_start, end=end_date, cache_dir=cache_dir)

    if not data:
        print("No data loaded. Check your internet connection or symbol names.")
        sys.exit(1)

    # ── Step 2: Run walk-forward backtest ─────────────────────────────────────
    logger.info("Running walk-forward backtest…")
    result = run_backtest(data, initial_capital=args.capital, active_from=active_start)

    # ── Step 3: Generate report ───────────────────────────────────────────────
    logger.info("Generating report…")
    generate_report(result, output_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()
