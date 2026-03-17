"""
Dynamic NSE + BSE stock universe fetcher.

Sources (all official, public, no login required):
  NSE EQUITY_L.csv          — every EQ-series stock listed on NSE (~2000 stocks)
  NSE Midcap 150 CSV        — 150 stocks classified as midcap
  NSE Smallcap 250 CSV      — 250 stocks classified as smallcap
  NSE Smallcap 100 CSV      — 100 stocks (subset of 250, used for double-check)
  BSE bhavcopy              — latest daily BSE equity list (BSE-only stocks)

Universe is cached locally for 24 hours so repeated runs don't re-download.
Force refresh with: get_universe(force_refresh=True)
"""

import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─── NSE URLs ─────────────────────────────────────────────────────────────────
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}

_EQUITY_LIST_URL   = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_MIDCAP_150_URL    = "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
_SMALLCAP_250_URL  = "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
_SMALLCAP_100_URL  = "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap100list.csv"

# ─── BSE bhav copy (latest trading day equity list) ──────────────────────────
# BSE publishes a daily bhavcopy with all traded stocks.
# We use it to capture BSE-only stocks (those not on NSE).
# BSE bhavcopy — BSE changed their URL scheme; we try multiple patterns.
# If all fail, BSE-only stocks are simply skipped (most trade on NSE anyway).
_BSE_BHAVCOPY_URLS = [
    "https://www.bseindia.com/download/BhavCopy/Equity/EQ{date}_CSV.ZIP",
    "https://www.bseindia.com/bsedownloads/BhavCopy/Equity/EQ{date}_CSV.ZIP",
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",  # fallback: skip BSE
]

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "data_cache")
CACHE_FILE  = os.path.join(CACHE_DIR, "universe_cache.csv")
CACHE_TTL_H = 24   # hours before refreshing universe list


# ─── Stock dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stock:
    symbol:   str
    name:     str
    cap:      str       # "midcap" | "smallcap" | "microcap" | "other"
    exchange: str       # "NSE" | "BSE"
    industry: str = ""

    @property
    def yf_ticker(self) -> str:
        suffix = ".NS" if self.exchange == "NSE" else ".BO"
        return self.symbol + suffix


# ─── Fetch helpers ────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 15) -> bytes | None:
    try:
        r = requests.get(url, headers=_NSE_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.warning(f"Could not fetch {url}: {e}")
        return None


def _fetch_index_symbols(url: str) -> dict[str, tuple[str, str]]:
    """
    Download a NIFTY index CSV and return {SYMBOL: (company_name, industry)}.

    CSV format:
      Company Name, Industry, Symbol, Series, ISIN Code
    """
    raw = _get(url)
    if raw is None:
        return {}
    try:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = df.columns.str.strip()
        result = {}
        for _, row in df.iterrows():
            sym = str(row.get("Symbol", "")).strip()
            name = str(row.get("Company Name", "")).strip()
            ind  = str(row.get("Industry", "")).strip()
            if sym and sym != "nan":
                result[sym] = (name, ind)
        return result
    except Exception as e:
        logger.warning(f"Could not parse {url}: {e}")
        return {}


def _fetch_all_nse_equities() -> dict[str, str]:
    """
    Download EQUITY_L.csv and return {SYMBOL: company_name} for EQ-series only.

    CSV format:
      SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, ...
    """
    raw = _get(_EQUITY_LIST_URL)
    if raw is None:
        return {}
    try:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = df.columns.str.strip()   # remove leading/trailing spaces
        # Keep only regular equity (EQ) — exclude SM, BE, BL, etc.
        series_col = "SERIES"
        eq = df[df[series_col].str.strip() == "EQ"]
        return {
            str(row["SYMBOL"]).strip(): str(row["NAME OF COMPANY"]).strip()
            for _, row in eq.iterrows()
            if str(row["SYMBOL"]).strip()
        }
    except Exception as e:
        logger.warning(f"Could not parse EQUITY_L: {e}")
        return {}


def _fetch_bse_equities() -> dict[str, str]:
    """
    Fetch today's (or yesterday's) BSE bhavcopy ZIP for BSE-only stocks.
    Falls back gracefully if unavailable.
    Returns {BSE_CODE (as string): company_name}
    """
    import zipfile

    for days_back in range(1, 7):      # start from yesterday, go back up to 6 days
        dt = datetime.today() - timedelta(days=days_back)
        if dt.weekday() >= 5:          # skip weekends
            continue
        date_str = dt.strftime("%d%m%y")
        raw = None
        for url_template in _BSE_BHAVCOPY_URLS:
            url = url_template.format(date=date_str)
            raw = _get(url, timeout=15)
            if raw:
                break
        if raw is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                csv_name = [n for n in z.namelist() if n.endswith(".CSV")][0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            # BSE bhavcopy columns: SC_CODE, SC_NAME, SC_GROUP, SC_TYPE, ...
            # Keep A/B group equities (mid/small liquids)
            if "SC_GROUP" in df.columns:
                df = df[df["SC_GROUP"].isin(["A", "B", "T"])]
            return {
                str(row["SC_CODE"]).strip(): str(row["SC_NAME"]).strip()
                for _, row in df.iterrows()
                if str(row.get("SC_CODE", "")).strip()
            }
        except Exception as e:
            logger.debug(f"BSE bhavcopy parse error ({date_str}): {e}")

    logger.warning("Could not fetch BSE bhavcopy — BSE stocks skipped.")
    return {}


# ─── Universe builder ────────────────────────────────────────────────────────

def _build_universe() -> list[Stock]:
    """
    Fetch and classify ALL NSE EQ-series stocks + BSE Group A/B stocks.

    Classification priority:
      1. In NIFTY Midcap 150            → midcap
      2. In NIFTY Smallcap 250 or 100   → smallcap
      3. In EQUITY_L.csv (EQ series)    → microcap  (smaller unlisted-in-index stocks)
      4. BSE only (not on NSE)           → bse_only
    """
    from rich.console import Console
    console = Console()

    console.print("  [dim]Fetching NSE Midcap 150…[/dim]")
    midcap_map  = _fetch_index_symbols(_MIDCAP_150_URL)    # 150 stocks

    console.print("  [dim]Fetching NSE Smallcap 250…[/dim]")
    smallcap_map = _fetch_index_symbols(_SMALLCAP_250_URL)  # 250 stocks

    console.print("  [dim]Fetching NSE Smallcap 100 (supplement)…[/dim]")
    smallcap_100 = _fetch_index_symbols(_SMALLCAP_100_URL)  # 100 stocks (overlap OK)
    smallcap_map.update(smallcap_100)

    console.print("  [dim]Fetching ALL NSE EQ-series equities…[/dim]")
    all_nse = _fetch_all_nse_equities()                     # ~2000 stocks

    console.print("  [dim]Fetching BSE daily bhavcopy…[/dim]")
    all_bse = _fetch_bse_equities()

    stocks: list[Stock] = []
    seen_isins: set[str] = set()

    # ── NSE stocks ────────────────────────────────────────────────────────────
    for symbol, name in all_nse.items():
        if symbol in midcap_map:
            cap = "midcap"
            _, industry = midcap_map[symbol]
        elif symbol in smallcap_map:
            cap = "smallcap"
            _, industry = smallcap_map[symbol]
        else:
            cap = "microcap"    # real NSE-listed stock but outside the named indices
            industry = ""

        stocks.append(Stock(symbol=symbol, name=name, cap=cap,
                            exchange="NSE", industry=industry))
        seen_isins.add(symbol)

    # ── BSE-only stocks (not traded on NSE) ───────────────────────────────────
    # BSE codes are numeric (e.g. "500325"). We check if the company doesn't
    # already appear in NSE by cross-referencing names (approximate).
    nse_names_lower = {s.name.lower()[:20] for s in stocks}
    bse_added = 0
    for bse_code, bse_name in all_bse.items():
        if bse_name.lower()[:20] in nse_names_lower:
            continue   # already have this company via NSE
        stocks.append(Stock(symbol=bse_code, name=bse_name,
                            cap="smallcap", exchange="BSE", industry=""))
        bse_added += 1

    logger.info(f"Universe: {len(stocks)} total  "
                f"({len(midcap_map)} midcap, {len(smallcap_map)} smallcap, "
                f"{bse_added} BSE-only)")
    return stocks


# ─── Cache logic ──────────────────────────────────────────────────────────────

def _save_cache(stocks: list[Stock]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = [{"symbol": s.symbol, "name": s.name, "cap": s.cap,
             "exchange": s.exchange, "industry": s.industry}
            for s in stocks]
    pd.DataFrame(rows).to_csv(CACHE_FILE, index=False)


def _load_cache() -> list[Stock]:
    df = pd.read_csv(CACHE_FILE)
    return [
        Stock(symbol=row["symbol"], name=row["name"], cap=row["cap"],
              exchange=row["exchange"],
              industry=row.get("industry", "") if pd.notna(row.get("industry")) else "")
        for _, row in df.iterrows()
    ]


def _cache_is_fresh() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return age < timedelta(hours=CACHE_TTL_H)


# ─── Public API ──────────────────────────────────────────────────────────────

def get_universe(exchange: str = None, cap: str = None,
                 force_refresh: bool = False) -> list[Stock]:
    """
    Return all stocks, optionally filtered by exchange or cap category.

    exchange : "NSE" | "BSE" | None (both)
    cap      : "midcap" | "smallcap" | "microcap" | "other" | None (all)
    """
    if force_refresh or not _cache_is_fresh():
        stocks = _build_universe()
        _save_cache(stocks)
    else:
        stocks = _load_cache()

    if exchange:
        stocks = [s for s in stocks if s.exchange.upper() == exchange.upper()]
    if cap:
        stocks = [s for s in stocks if s.cap.lower() == cap.lower()]

    return stocks


def summary(stocks: list[Stock] = None):
    if stocks is None:
        stocks = get_universe()
    caps = {}
    for s in stocks:
        key = f"{s.exchange}/{s.cap}"
        caps[key] = caps.get(key, 0) + 1
    parts = [f"{k}: {v}" for k, v in sorted(caps.items())]
    print(f"Universe: {len(stocks)} stocks  |  " + "  |  ".join(parts))
