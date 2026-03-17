"""
Main entry point — orchestrates scanning, entry, and exit.

Usage:
    python main.py --login         # Step 1: generate Kite login URL (do once per day)
    python main.py --token <tok>   # Step 2: save access token after redirect
    python main.py                 # Step 3: run the live strategy loop
    python main.py --scan-once     # Single scan and exit (useful for testing)
"""

import argparse
import logging
import time
from datetime import datetime, time as dtime

import config
from broker import BrokerClient
from strategy import Position, Signal, check_exit, fetch_data, scan_symbol

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


# ── Risk management helpers ───────────────────────────────────────────────────

def calculate_position_size(signal: Signal, available_cash: float,
                             total_capital: float) -> tuple[int, float]:
    """
    Returns (quantity, capital_to_deploy) based on:
    - Max 15% of total capital per position
    - Available cash after existing positions
    """
    max_capital = min(
        config.MAX_POSITION_PCT * total_capital,
        available_cash * config.MAX_CAPITAL_DEPLOYED,
    )
    if signal.entry_price <= 0 or max_capital <= 0:
        return 0, 0.0
    quantity = int(max_capital // signal.entry_price)
    capital = quantity * signal.entry_price
    return quantity, capital


def capital_deployed_pct(positions: list[Position]) -> float:
    total_deployed = sum(p.capital_deployed for p in positions)
    return total_deployed / config.TOTAL_CAPITAL


def is_market_open() -> bool:
    now = datetime.now().time()
    open_t = dtime(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    close_t = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    return open_t <= now <= close_t


# ── Core scan loop ────────────────────────────────────────────────────────────

def run_scan(broker: BrokerClient, open_positions: list[Position]) -> list[Position]:
    """
    1. Check exits on all open positions.
    2. Scan watchlist for new entry signals if capacity allows.
    Returns the updated list of open positions.
    """
    logger.info("=" * 60)
    logger.info(f"Scan started | open positions: {len(open_positions)}")

    # ── Step 1: Exit checks ───────────────────────────────────────────────────
    positions_to_close = []
    for pos in open_positions:
        df = fetch_data(pos.symbol, interval="1d", period="6mo")
        if df is None:
            continue
        should_exit, reason = check_exit(pos, df)
        if should_exit:
            positions_to_close.append((pos, reason))

    for pos, reason in positions_to_close:
        if broker.place_sell_order(pos, reason):
            open_positions = [p for p in open_positions if p.symbol != pos.symbol]

    # ── Step 2: Entry scan ────────────────────────────────────────────────────
    open_symbols = {p.symbol for p in open_positions}
    deployed_pct = capital_deployed_pct(open_positions)

    if len(open_positions) >= config.MAX_POSITIONS:
        logger.info("Max positions reached — skipping entry scan.")
        return open_positions

    if deployed_pct >= config.MAX_CAPITAL_DEPLOYED:
        logger.info(f"Capital {deployed_pct:.0%} deployed — skipping entry scan.")
        return open_positions

    available_cash = broker.get_available_cash()
    logger.info(f"Available cash: ₹{available_cash:,.0f} | deployed: {deployed_pct:.0%}")

    signals: list[Signal] = []
    for symbol in config.WATCHLIST:
        if symbol in open_symbols:
            continue
        try:
            signal = scan_symbol(symbol)
            if signal:
                logger.info(
                    f"SIGNAL [{signal.setup}] {symbol} | "
                    f"entry={signal.entry_price:.2f} stop={signal.stop_loss:.2f} "
                    f"target={signal.target_price:.2f} R:R={signal.reward_risk:.2f}"
                )
                signals.append(signal)
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

    # Sort signals by best R:R first
    signals.sort(key=lambda s: s.reward_risk, reverse=True)

    for signal in signals:
        if len(open_positions) >= config.MAX_POSITIONS:
            break
        if capital_deployed_pct(open_positions) >= config.MAX_CAPITAL_DEPLOYED:
            break

        quantity, capital = calculate_position_size(
            signal, available_cash, config.TOTAL_CAPITAL
        )
        if quantity == 0:
            logger.info(f"{signal.symbol}: quantity 0 — insufficient capital, skipping.")
            continue

        success = broker.place_buy_order(signal, quantity, capital)
        if success:
            open_positions.append(Position(
                symbol=signal.symbol,
                setup=signal.setup,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target_price=signal.target_price,
                atr=signal.atr,
                quantity=quantity,
                capital_deployed=capital,
                trail_stop=signal.entry_price - config.ATR_TRAIL_MULTIPLIER * signal.atr,
            ))
            available_cash -= capital

    logger.info(f"Scan complete | open positions: {len(open_positions)}")
    return open_positions


# ── Entry points ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Indian Stock Market Strategy Bot")
    parser.add_argument("--login", action="store_true",
                        help="Generate Kite Connect login URL")
    parser.add_argument("--token", type=str, default=None,
                        help="Complete login with request_token from redirect URL")
    parser.add_argument("--scan-once", action="store_true",
                        help="Run a single scan then exit")
    args = parser.parse_args()

    broker = BrokerClient()

    if args.login:
        url = broker.generate_login_url()
        print(f"\nOpen this URL in your browser:\n{url}\n")
        print("After login, copy the 'request_token' from the redirect URL and run:")
        print("  python main.py --token <request_token>\n")
        return

    if args.token:
        broker.complete_login(args.token)
        print("Login complete. You can now run: python main.py")
        return

    # ── Main strategy loop ────────────────────────────────────────────────────
    logger.info("Strategy bot started.")
    logger.info(f"Mode: {'PAPER TRADE' if config.PAPER_TRADE else 'LIVE TRADE'}")
    logger.info(f"Watchlist: {len(config.WATCHLIST)} stocks")
    logger.info(f"Capital: ₹{config.TOTAL_CAPITAL:,.0f}")

    open_positions: list[Position] = []

    if args.scan_once:
        open_positions = run_scan(broker, open_positions)
        return

    while True:
        try:
            if is_market_open():
                open_positions = run_scan(broker, open_positions)
            else:
                now = datetime.now().strftime("%H:%M")
                logger.info(f"Market closed ({now} IST) — waiting...")

            interval = config.SCAN_INTERVAL_MINUTES * 60
            logger.info(f"Next scan in {config.SCAN_INTERVAL_MINUTES} minutes.")
            time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(60)  # brief pause before retry


if __name__ == "__main__":
    main()
