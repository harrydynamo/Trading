"""
NSE Insider Trading — Promoter Buy Transactions
================================================
Fetches recent promoter buy disclosures from the NSE corporate insider
trading endpoint (SEBI PIT regulation filings).

Promoters must disclose trades within 2 trading days, so this data is
much more granular than the quarterly shareholding pattern.

Usage:
    from stock_screener.insider import fetch_promoter_buys
    df = fetch_promoter_buys(days=30)
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# ─── NSE session ──────────────────────────────────────────────────────────────

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
    "Origin":          "https://www.nseindia.com",
}

_session: requests.Session | None = None
_session_ts: float = 0.0
_SESSION_TTL = 300   # refresh session cookie every 5 min


def _get_session() -> requests.Session:
    global _session, _session_ts
    now = time.time()
    if _session is None or (now - _session_ts) > _SESSION_TTL:
        s = requests.Session()
        try:
            # Hit the main page to seed session cookies
            s.get(
                "https://www.nseindia.com",
                headers=_NSE_HEADERS,
                timeout=12,
            )
            time.sleep(0.5)   # brief pause — NSE rate-limits aggressive bots
        except Exception as e:
            log.warning("NSE session init failed: %s", e)
        _session = s
        _session_ts = now
    return _session


# ─── Person-category filter ───────────────────────────────────────────────────

_PROMOTER_KEYWORDS = {"promoter", "promoter group", "promoters"}


def _is_promoter(category: str) -> bool:
    return any(kw in category.lower() for kw in _PROMOTER_KEYWORDS)


# ─── Main fetch ───────────────────────────────────────────────────────────────

_INSIDER_URL = (
    "https://www.nseindia.com/api/corporate-insider-trading"
    "?index=equities"
)


def fetch_promoter_buys(days: int = 30) -> pd.DataFrame:
    """
    Fetch promoter BUY transactions filed with NSE in the last `days` days.

    Returns a DataFrame with columns:
        symbol, company, promoter_name, category,
        transaction_date, from_date, to_date,
        shares_bought, transaction_value_lakh,
        holding_before_pct, holding_after_pct, change_pct

    Returns an empty DataFrame (with those columns) on failure.
    """
    empty = pd.DataFrame(columns=[
        "symbol", "company", "promoter_name", "category",
        "transaction_date", "from_date", "to_date",
        "shares_bought", "transaction_value_lakh",
        "holding_before_pct", "holding_after_pct", "change_pct",
    ])

    cutoff = datetime.now() - timedelta(days=days)

    try:
        session = _get_session()
        resp = session.get(
            _INSIDER_URL,
            headers=_NSE_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        log.warning("NSE insider trading fetch failed: %s", e)
        return empty

    # NSE returns either {"data": [...]} or a bare list
    if isinstance(raw, dict):
        records = raw.get("data", [])
    elif isinstance(raw, list):
        records = raw
    else:
        return empty

    rows = []
    for r in records:
        # ── Filter: promoter category ──────────────────────────────────
        category = str(r.get("personCategory") or r.get("acqCategory") or "")
        if not _is_promoter(category):
            continue

        # ── Filter: buy transactions only ─────────────────────────────
        txn_type = str(r.get("tdpTransactionType") or r.get("transactionType") or "")
        if "buy" not in txn_type.lower() and "acqui" not in txn_type.lower():
            continue

        # ── Parse date ────────────────────────────────────────────────
        raw_date = (
            r.get("date") or r.get("acqtoDt") or
            r.get("tdpDt") or r.get("acqfromDt") or ""
        )
        try:
            txn_date = pd.to_datetime(raw_date, dayfirst=True)
        except Exception:
            txn_date = pd.NaT

        if pd.isna(txn_date) or txn_date < cutoff:
            continue

        # ── Parse numeric fields ───────────────────────────────────────
        def _f(key, *alt_keys):
            for k in (key, *alt_keys):
                v = r.get(k)
                if v not in (None, "", "-", "NA"):
                    try:
                        return float(str(v).replace(",", ""))
                    except ValueError:
                        pass
            return np.nan

        shares      = _f("secAcq", "noOfShareAcq", "sharesAcq")
        value_lakh  = _f("tdpVal", "acqVal")          # NSE reports in ₹ (not lakhs)
        bef_pct     = _f("befAcqSharesPer", "beforeAcqPer")
        aft_pct     = _f("afterAcqSharesPer", "afterAcqPer")

        # Convert value to lakhs if it looks like raw rupees (> 1e5)
        if not np.isnan(value_lakh) and value_lakh > 1e5:
            value_lakh = round(value_lakh / 1e5, 2)

        change_pct = round(aft_pct - bef_pct, 4) if not (np.isnan(aft_pct) or np.isnan(bef_pct)) else np.nan

        rows.append({
            "symbol":               str(r.get("symbol") or ""),
            "company":              str(r.get("company") or r.get("companyName") or ""),
            "promoter_name":        str(r.get("acqName") or r.get("personName") or ""),
            "category":             category,
            "transaction_date":     txn_date,
            "from_date":            r.get("acqfromDt") or "",
            "to_date":              r.get("acqtoDt") or "",
            "shares_bought":        shares,
            "transaction_value_lakh": value_lakh,
            "holding_before_pct":   bef_pct,
            "holding_after_pct":    aft_pct,
            "change_pct":           change_pct,
        })

    if not rows:
        return empty

    df = pd.DataFrame(rows)
    df = df.sort_values("transaction_date", ascending=False).reset_index(drop=True)
    return df
