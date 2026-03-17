"""
Tracks open positions in a local JSON file (live_signals/my_positions.json).

The user tells the system which trades they've taken; the scanner then
monitors those positions for exit signals.
"""

import json
import os
from datetime import date
from typing import Optional

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "my_positions.json")


def load() -> list[dict]:
    if not os.path.exists(POSITIONS_FILE):
        return []
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def save(positions: list[dict]):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def add(symbol: str, quantity: int, entry_price: float,
        stop_loss: float, target_price: float, atr: float,
        setup: str = "?") -> dict:
    """Add a new position. Overwrites if the symbol already exists."""
    positions = load()
    positions = [p for p in positions if p["symbol"] != symbol]  # remove duplicate

    trail_stop = entry_price - 2 * atr  # initial trail = 2 ATR below entry

    pos = {
        "symbol":       symbol,
        "setup":        setup,
        "entry_date":   date.today().isoformat(),
        "entry_price":  entry_price,
        "stop_loss":    stop_loss,
        "target_price": target_price,
        "atr":          atr,
        "quantity":     quantity,
        "trail_stop":   trail_stop,
    }
    positions.append(pos)
    save(positions)
    return pos


def remove(symbol: str) -> bool:
    positions = load()
    before = len(positions)
    positions = [p for p in positions if p["symbol"] != symbol]
    save(positions)
    return len(positions) < before


def update_trail_stop(symbol: str, new_trail: float):
    """Ratchet up the trailing stop for a position (never lower it)."""
    positions = load()
    for p in positions:
        if p["symbol"] == symbol:
            p["trail_stop"] = max(float(p.get("trail_stop", 0)), new_trail)
    save(positions)


def get(symbol: str) -> Optional[dict]:
    return next((p for p in load() if p["symbol"] == symbol), None)
