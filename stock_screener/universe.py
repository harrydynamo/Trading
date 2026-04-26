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
_SME_EMERGE_URL    = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"

# ─── BSE bhav copy (latest trading day equity list) ──────────────────────────
# BSE publishes a daily bhavcopy with all traded stocks.
# We use it to capture BSE-only stocks (those not on NSE).
# Old EQ{DDMMYY}_CSV.ZIP format was discontinued by BSE on 8 Jul 2024.
# New formats (date tokens filled in per-request, not stored here):
#   ZIP  : BSE_EQ_BHAVCOPY_{DDMMYYYY}_T0.ZIP  (contains a .CSV inside)
#   CSV  : BhavCopy_BSE_CM_0_0_0_{YYYYMMDD}_F_0000.CSV  (direct CSV, no ZIP)
#   ISIN : EQ_ISINCODE_{DDMMYY}_T0.CSV  (direct CSV, ISIN-based)
_BSE_BASE = "https://www.bseindia.com/download/BhavCopy/Equity/"

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
        if self.exchange == "NSE":
            # NSE Emerge (SME) stocks use the -SM suffix on Yahoo Finance
            return f"{self.symbol}-SM.NS" if self.cap == "sme" else f"{self.symbol}.NS"
        return f"{self.symbol}.BO"


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
        _SKIP = ("-RE", "-PP", "-N", "-W", "-SPE")
        eq = df[df[series_col].str.strip() == "EQ"]
        return {
            str(row["SYMBOL"]).strip(): str(row["NAME OF COMPANY"]).strip()
            for _, row in eq.iterrows()
            if str(row["SYMBOL"]).strip()
            and not any(str(row["SYMBOL"]).strip().upper().endswith(s) for s in _SKIP)
        }
    except Exception as e:
        logger.warning(f"Could not parse EQUITY_L: {e}")
        return {}


def _fetch_nse_sme() -> dict[str, str]:
    """
    Download NSE Emerge (SME) equity list.
    Returns {SYMBOL: company_name} for SME-listed stocks.
    CSV format: SYMBOL, NAME OF COMPANY, SERIES, ...
    Falls back to parsing EQUITY_L.csv series='SM' if the dedicated URL fails.

    Filters out non-tradeable instruments:
      -RE  rights entitlements (temporary, no OHLCV on yfinance)
      -PP  partly-paid shares
      -N   new / odd-lot series
    """
    # Suffixes that indicate non-tradeable temporary instruments
    _SKIP_SUFFIXES = ("-RE", "-PP", "-N", "-W", "-SPE")

    def _clean(result: dict) -> dict:
        return {
            sym: name for sym, name in result.items()
            if not any(sym.upper().endswith(s) for s in _SKIP_SUFFIXES)
        }

    # Try dedicated NSE Emerge URL first
    raw = _get(_SME_EMERGE_URL)
    if raw:
        try:
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = df.columns.str.strip()
            sym_col  = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
            name_col = next((c for c in df.columns if "NAME" in c.upper()), None)
            if sym_col and name_col:
                return _clean({
                    str(row[sym_col]).strip(): str(row[name_col]).strip()
                    for _, row in df.iterrows()
                    if str(row[sym_col]).strip()
                })
        except Exception as e:
            logger.debug(f"SME_EQUITY_L parse error: {e}")

    # Fallback: parse EQUITY_L.csv keeping series "SM" (NSE Emerge)
    raw = _get(_EQUITY_LIST_URL)
    if raw is None:
        return {}
    try:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = df.columns.str.strip()
        sme = df[df["SERIES"].str.strip() == "SM"]
        return _clean({
            str(row["SYMBOL"]).strip(): str(row["NAME OF COMPANY"]).strip()
            for _, row in sme.iterrows()
            if str(row["SYMBOL"]).strip()
        })
    except Exception as e:
        logger.warning(f"Could not parse SME from EQUITY_L: {e}")
        return {}


def _fetch_bse_equities() -> dict[str, str]:
    """
    Fetch the latest BSE bhavcopy for BSE-only stocks.
    Tries the three current URL formats (post-July 2024) silently,
    falls back gracefully if all fail.
    Returns {BSE_CODE (as string): company_name}
    """
    import zipfile
    import requests as _req

    def _try(url: str) -> bytes | None:
        """Silent fetch — no WARNING logged on 404."""
        try:
            r = _req.get(url, headers=_NSE_HEADERS, timeout=15)
            r.raise_for_status()
            return r.content
        except Exception:
            return None

    def _parse_zip(raw: bytes) -> dict[str, str]:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            csv_name = next(n for n in z.namelist()
                            if n.upper().endswith(".CSV"))
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        # New ZIP format: TckrSymb / FinInstrmId = BSE code, SctySrs = EQ
        if "TckrSymb" in df.columns:
            if "SctySrs" in df.columns:
                df = df[df["SctySrs"].str.strip() == "EQ"]
            name_col = "FinInstrmNm" if "FinInstrmNm" in df.columns else "TckrSymb"
            return {str(r["TckrSymb"]).strip(): str(r[name_col]).strip()
                    for _, r in df.iterrows() if str(r["TckrSymb"]).strip()}
        # Old ZIP format: SC_CODE, SC_NAME, SC_GROUP
        if "SC_CODE" in df.columns:
            if "SC_GROUP" in df.columns:
                df = df[df["SC_GROUP"].isin(["A", "B", "T"])]
            return {str(r["SC_CODE"]).strip(): str(r["SC_NAME"]).strip()
                    for _, r in df.iterrows() if str(r.get("SC_CODE", "")).strip()}
        return {}

    def _parse_csv(raw: bytes) -> dict[str, str]:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = df.columns.str.strip()
        # UDiFF CSV format: TckrSymb, FinInstrmNm, SctySrs (BSE group: A/B/T/Z…)
        # FinInstrmTp == "STK" keeps equities only (excludes derivatives)
        if "TckrSymb" in df.columns:
            if "FinInstrmTp" in df.columns:
                df = df[df["FinInstrmTp"].str.strip() == "STK"]
            # Keep liquid groups A, B, T; drop Z (suspended), others
            if "SctySrs" in df.columns:
                df = df[df["SctySrs"].str.strip().isin(["A", "B", "T"])]
            name_col = next((c for c in ("FinInstrmNm", "ShrtNm", "TckrSymb")
                             if c in df.columns), "TckrSymb")
            return {str(r["TckrSymb"]).strip(): str(r[name_col]).strip()
                    for _, r in df.iterrows() if str(r["TckrSymb"]).strip()}
        return {}

    for days_back in range(1, 7):          # try yesterday back to 6 days ago
        dt = datetime.today() - timedelta(days=days_back)
        if dt.weekday() >= 5:              # skip weekends
            continue

        ddmmyyyy = dt.strftime("%d%m%Y")   # e.g. 20032026
        yyyymmdd = dt.strftime("%Y%m%d")   # e.g. 20260320
        ddmmyy   = dt.strftime("%d%m%y")   # e.g. 200326

        attempts = [
            # Format 1: new ZIP (post-Jul 2024)
            (_try(f"{_BSE_BASE}BSE_EQ_BHAVCOPY_{ddmmyyyy}_T0.ZIP"), "zip"),
            # Format 2: UDiFF direct CSV
            (_try(f"{_BSE_BASE}BhavCopy_BSE_CM_0_0_0_{yyyymmdd}_F_0000.CSV"), "csv"),
            # Format 3: ISIN-based CSV
            (_try(f"{_BSE_BASE}EQ_ISINCODE_{ddmmyy}_T0.CSV"), "csv"),
        ]

        for raw, fmt in attempts:
            if not raw:
                continue
            try:
                result = _parse_zip(raw) if fmt == "zip" else _parse_csv(raw)
                if result:
                    logger.debug(f"BSE bhavcopy loaded for {dt.date()} ({len(result)} stocks)")
                    return result
            except Exception as e:
                logger.debug(f"BSE parse error ({dt.date()}): {e}")

    logger.warning("BSE bhavcopy unavailable — BSE-only stocks skipped (NSE stocks unaffected).")
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

    console.print("  [dim]Fetching NSE Emerge (SME) equities…[/dim]")
    all_sme = _fetch_nse_sme()                              # ~700 SME stocks

    console.print("  [dim]Fetching BSE daily bhavcopy…[/dim]")
    all_bse = _fetch_bse_equities()

    stocks: list[Stock] = []
    seen_isins: set[str] = set()

    # ── NSE SME stocks (NSE Emerge) ───────────────────────────────────────────
    for symbol, name in all_sme.items():
        stocks.append(Stock(symbol=symbol, name=name, cap="sme",
                            exchange="NSE", industry=""))
        seen_isins.add(symbol)

    # ── NSE main-board stocks ─────────────────────────────────────────────────
    for symbol, name in all_nse.items():
        if symbol in seen_isins:
            continue   # already added as SME
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
                f"{len(all_sme)} SME, {bse_added} BSE-only)")
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
