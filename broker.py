"""
Broker interface — Zerodha Kite Connect.

Paper trade mode (config.PAPER_TRADE = True) logs all orders without
sending them to the exchange. Set PAPER_TRADE = False for live execution.

Daily login flow:
    1. python main.py --login        → prints login URL
    2. Paste the redirected URL      → saves access token to file
    3. python main.py                → runs the strategy with saved token
"""

import logging
import os
from datetime import date

import config
from strategy import Position, Signal

logger = logging.getLogger(__name__)

# ── Optional Kite Connect import ──────────────────────────────────────────────
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logger.warning("kiteconnect not installed — paper trade mode only. "
                   "Install with: pip install kiteconnect")


class BrokerClient:
    """Thin wrapper around Kite Connect with paper-trade fallback."""

    def __init__(self):
        self.kite = None
        self.paper = config.PAPER_TRADE or not KITE_AVAILABLE
        self._paper_positions: list[Position] = []  # in-memory for paper trades

        if not self.paper:
            self._init_kite()

    # ── Authentication ────────────────────────────────────────────────────────

    def _init_kite(self):
        self.kite = KiteConnect(api_key=config.KITE_API_KEY)
        token = self._load_token()
        if token:
            self.kite.set_access_token(token)
            logger.info("Kite Connect initialised with saved token.")
        else:
            logger.error("No access token found. Run: python main.py --login")

    def generate_login_url(self) -> str:
        if not KITE_AVAILABLE:
            return "kiteconnect not installed"
        kite = KiteConnect(api_key=config.KITE_API_KEY)
        return kite.login_url()

    def complete_login(self, request_token: str):
        """Exchange request_token for access_token and save it."""
        kite = KiteConnect(api_key=config.KITE_API_KEY)
        data = kite.generate_session(request_token, api_secret=config.KITE_API_SECRET)
        access_token = data["access_token"]
        self._save_token(access_token)
        logger.info("Login successful. Access token saved.")
        return access_token

    def _save_token(self, token: str):
        with open(config.ACCESS_TOKEN_FILE, "w") as f:
            f.write(f"{date.today().isoformat()}:{token}")

    def _load_token(self) -> str | None:
        if not os.path.exists(config.ACCESS_TOKEN_FILE):
            return None
        with open(config.ACCESS_TOKEN_FILE) as f:
            content = f.read().strip()
        if not content:
            return None
        parts = content.split(":", 1)
        if len(parts) == 2 and parts[0] == date.today().isoformat():
            return parts[1]
        logger.warning("Access token is from a previous day — re-login required.")
        return None

    # ── Account info ──────────────────────────────────────────────────────────

    def get_available_cash(self) -> float:
        if self.paper:
            deployed = sum(p.capital_deployed for p in self._paper_positions)
            return config.TOTAL_CAPITAL - deployed
        try:
            margins = self.kite.margins()
            return float(margins["equity"]["available"]["live_balance"])
        except Exception as e:
            logger.error(f"Could not fetch margins: {e}")
            return 0.0

    def get_open_positions(self) -> list[Position]:
        if self.paper:
            return list(self._paper_positions)
        # In live mode, positions are managed by the portfolio tracker (main.py)
        return []

    # ── Order placement ───────────────────────────────────────────────────────

    def place_buy_order(self, signal: Signal, quantity: int,
                        capital: float) -> bool:
        """Place a market buy order for the given signal."""
        if quantity <= 0:
            logger.warning(f"{signal.symbol}: calculated quantity is 0, skipping.")
            return False

        if self.paper:
            pos = Position(
                symbol=signal.symbol,
                setup=signal.setup,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target_price=signal.target_price,
                atr=signal.atr,
                quantity=quantity,
                capital_deployed=capital,
                trail_stop=signal.entry_price - config.ATR_TRAIL_MULTIPLIER * signal.atr,
            )
            self._paper_positions.append(pos)
            logger.info(
                f"[PAPER BUY] {signal.symbol} | Setup {signal.setup} | "
                f"qty={quantity} @ {signal.entry_price:.2f} | "
                f"stop={signal.stop_loss:.2f} | target={signal.target_price:.2f} | "
                f"R:R={signal.reward_risk:.2f} | {signal.notes}"
            )
            return True

        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=signal.symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                order_type=self.kite.ORDER_TYPE_MARKET,
                product=self.kite.PRODUCT_CNC,  # CNC = delivery / equity
            )
            logger.info(
                f"[LIVE BUY] {signal.symbol} | order_id={order_id} | "
                f"qty={quantity} @ market | stop={signal.stop_loss:.2f} | "
                f"target={signal.target_price:.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"Buy order failed for {signal.symbol}: {e}")
            return False

    def place_sell_order(self, position: Position, reason: str) -> bool:
        """Close an existing position at market price."""
        if self.paper:
            self._paper_positions = [
                p for p in self._paper_positions if p.symbol != position.symbol
            ]
            logger.info(
                f"[PAPER SELL] {position.symbol} | qty={position.quantity} | "
                f"reason: {reason}"
            )
            return True

        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=position.symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                quantity=position.quantity,
                order_type=self.kite.ORDER_TYPE_MARKET,
                product=self.kite.PRODUCT_CNC,
            )
            logger.info(
                f"[LIVE SELL] {position.symbol} | order_id={order_id} | "
                f"qty={position.quantity} | reason: {reason}"
            )
            return True
        except Exception as e:
            logger.error(f"Sell order failed for {position.symbol}: {e}")
            return False
