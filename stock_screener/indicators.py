"""
Fundamental indicator calculations for the stock screener.

All ratios use TTM (Trailing Twelve Months) — sum of last 4 quarterly filings
for P&L and cash flow items; latest quarter for balance sheet items.
This matches screener.in's methodology.

Formulas aligned to screener.in:
  P/E           = Market Cap / TTM Net Profit
  P/S           = Market Cap / TTM Revenue
  ROCE          = EBIT(TTM) / (Total Assets − Current Liabilities) × 100
  Operating Margin = EBITDA(TTM) / Revenue(TTM) × 100   [EBIT + Depreciation]
  Net Profit Margin = Net Income(TTM) / Revenue(TTM) × 100
  ROE           = Net Income(TTM) / Shareholders' Equity × 100
  FCF Margin    = (Operating CF − Capex)(TTM) / Revenue(TTM) × 100
  Sales Growth  = (TTM Revenue − Prior Year Revenue) / |Prior Year| × 100
  Receivable Days = Receivables / Revenue(TTM) × 365
  Inventory Days  = Inventory / COGS(TTM) × 365
  Payable Days    = Payables / COGS(TTM) × 365
  CCC           = Receivable Days + Inventory Days − Payable Days
  Capex/Sales   = |Capex(TTM)| / Revenue(TTM) × 100
  Receivable/Sales = Receivables / Revenue(TTM) × 100

Not available via any free public API:
  Promoter Holding — heldPercentInsiders is used as a proxy but is unreliable
  Change in Promoter Holding — requires BSE quarterly shareholding pattern scraping
  Promoter buying, Order book, Segmental revenue, Sales breakup
"""

import logging
import re
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ─── Shared requests session for screener.in ──────────────────────────────────
_SCREENER_SESSION: requests.Session | None = None
_SCREENER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _screener_session() -> requests.Session:
    global _SCREENER_SESSION
    if _SCREENER_SESSION is None:
        _SCREENER_SESSION = requests.Session()
        _SCREENER_SESSION.headers.update(_SCREENER_HEADERS)
    return _SCREENER_SESSION


_HOLDING_CATEGORY_KEYWORDS = re.compile(
    r"^(FII|DII|Public|Institutions|Others|Mutual Fund|Insurance|Government|Foreign)",
    re.IGNORECASE,
)


def fetch_promoter_holding_screener(
    yf_ticker: str,
) -> tuple[float, float, str, float]:
    """
    Scrape promoter holding from screener.in.
    Tries consolidated page first, then standalone.

    Returns
    -------
    (total_pct, change_vs_prev_quarter_pct, top_promoter_name, top_promoter_pct)
    NaN / "" where unavailable.
    """
    clean = re.sub(r"\.(NS|BO)$", "", yf_ticker.upper().strip())
    if not clean:
        return np.nan, np.nan, "", np.nan

    session = _screener_session()

    for suffix in ("/consolidated/", "/"):
        url = f"https://www.screener.in/company/{clean}{suffix}"
        try:
            resp = session.get(url, timeout=8)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.find_all("tr")

            total_pct   = np.nan
            change_pct  = np.nan
            found_total = False

            # Individual promoter sub-rows collected after the Promoters total row
            individuals: list[tuple[str, float]] = []

            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True)

                if not found_total:
                    # Look for the main "Promoters" total row
                    if not re.match(r"^Promoters$", label, re.IGNORECASE):
                        continue
                    values: list[float] = []
                    for cell in cells[1:]:
                        m = re.search(r"([\d.]+)", cell.get_text(strip=True))
                        if m:
                            values.append(float(m.group(1)))
                    if not values:
                        break
                    total_pct  = values[0]
                    change_pct = round(total_pct - values[1], 2) if len(values) >= 2 else np.nan
                    found_total = True

                else:
                    # Rows after the Promoters total — stop at next main category
                    if _HOLDING_CATEGORY_KEYWORDS.match(label):
                        break
                    # Skip rows that look like totals again
                    if re.match(r"^Promoters$", label, re.IGNORECASE):
                        break
                    # Collect individual promoter rows (non-empty label, has a number)
                    if label:
                        m = re.search(r"([\d.]+)", cells[1].get_text(strip=True))
                        if m:
                            individuals.append((label, float(m.group(1))))

            if not np.isnan(total_pct):
                # Pick the individual with the highest holding
                if individuals:
                    top_name, top_pct = max(individuals, key=lambda x: x[1])
                else:
                    top_name, top_pct = "", np.nan
                return total_pct, change_pct, top_name, top_pct

        except Exception:
            continue

    return np.nan, np.nan, "", np.nan


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _val(df: pd.DataFrame, *keys, col: int = 0) -> float:
    """Extract a scalar from an annual financial statement DataFrame."""
    if df is None or df.empty:
        return np.nan
    for key in keys:
        try:
            if key in df.index:
                v = df.loc[key].iloc[col]
                if pd.notna(v):
                    return float(v)
        except Exception:
            continue
    return np.nan


def _ttm(qdf: pd.DataFrame, adf: pd.DataFrame, *keys) -> float:
    """
    Return TTM (sum of last 4 quarters) for P&L / cash flow items.
    Falls back to most recent annual figure if quarterly data is sparse.
    If fewer than 4 quarters exist, annualises proportionally (×4/n).
    """
    if qdf is not None and not qdf.empty:
        for key in keys:
            if key in qdf.index:
                series = qdf.loc[key]
                vals   = series.iloc[:4]          # newest → oldest
                n_valid = vals.notna().sum()
                if n_valid >= 2:
                    total = float(vals.fillna(0).sum())
                    if n_valid < 4:
                        total = total * 4 / n_valid   # annualise partial year
                    return total
    # Fallback: annual col 0
    return _val(adf, *keys, col=0)


def _latest(qdf: pd.DataFrame, adf: pd.DataFrame, *keys) -> float:
    """Return the most recent value from quarterly balance sheet, fallback annual."""
    if qdf is not None and not qdf.empty:
        for key in keys:
            if key in qdf.index:
                v = qdf.loc[key].iloc[0]
                if pd.notna(v):
                    return float(v)
    return _val(adf, *keys, col=0)


# ─── Main computation ─────────────────────────────────────────────────────────

def compute_fundamentals(ticker_obj: yf.Ticker, info: dict,
                         yf_ticker: str = "") -> dict:
    """
    Compute all fundamental ratios for a single stock using TTM methodology.
    Returns a flat dict of scalar values (NaN where data is unavailable).
    """
    # ── Fetch all statement DataFrames ────────────────────────────────────────
    try:
        fin_a = ticker_obj.financials               # annual income statement
    except Exception:
        fin_a = pd.DataFrame()
    try:
        fin_q = ticker_obj.quarterly_financials     # quarterly income statement
    except Exception:
        fin_q = pd.DataFrame()

    try:
        bs_a = ticker_obj.balance_sheet             # annual balance sheet
    except Exception:
        bs_a = pd.DataFrame()
    try:
        bs_q = ticker_obj.quarterly_balance_sheet   # quarterly balance sheet
    except Exception:
        bs_q = pd.DataFrame()

    try:
        cf_a = ticker_obj.cashflow                  # annual cash flow
    except Exception:
        cf_a = pd.DataFrame()
    try:
        cf_q = ticker_obj.quarterly_cashflow        # quarterly cash flow
    except Exception:
        cf_q = pd.DataFrame()

    # ── Market data from info dict ────────────────────────────────────────────
    price      = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    market_cap = float(info.get("marketCap") or np.nan)
    shares     = float(info.get("sharesOutstanding") or np.nan)
    sector     = str(info.get("sector")   or "Unknown")
    industry   = str(info.get("industry") or "Unknown")

    # Detect currency mismatch: price/mktcap are in INR but some companies
    # (e.g. INFY, WIPRO) report financials in USD via yfinance.
    # When there's a mismatch all ratio calculations that divide mktcap by a
    # financial statement figure will be wrong — fall back to yfinance pre-computed.
    price_currency   = str(info.get("currency") or "INR").upper()
    fin_currency     = str(info.get("financialCurrency") or price_currency).upper()
    currency_mismatch = (price_currency != fin_currency)

    # ── TTM Revenue ───────────────────────────────────────────────────────────
    revenue_ttm = _ttm(fin_q, fin_a, "Total Revenue", "Revenue")

    # Prior-year annual revenue for sales growth (use annual col 1)
    revenue_prior = _val(fin_a, "Total Revenue", "Revenue", col=1)
    # Fallback: 5 quarters ago vs 1 quarter ago from quarterly data
    if np.isnan(revenue_prior) and fin_q is not None and not fin_q.empty:
        for key in ("Total Revenue", "Revenue"):
            if key in fin_q.index:
                series = fin_q.loc[key]
                if len(series) >= 8:
                    # sum quarters 4–7 (one year ago TTM)
                    old = float(series.iloc[4:8].fillna(0).sum())
                    if old > 0:
                        revenue_prior = old
                        break

    # ── TTM Income Statement items ────────────────────────────────────────────
    ebit_ttm = _ttm(fin_q, fin_a,
                    "EBIT", "Operating Income", "Ebit",
                    "Normalized EBITDA")        # last resort

    net_income_ttm = _ttm(fin_q, fin_a,
                          "Net Income",
                          "Net Income From Continuing Operations",
                          "Net Income Common Stockholders")

    cogs_ttm = _ttm(fin_q, fin_a,
                    "Cost Of Revenue",
                    "Cost of Goods and Services Sold",
                    "Reconciled Cost Of Revenue",
                    "Cost Of Goods Sold")

    # ── TTM Cash Flow items ───────────────────────────────────────────────────
    depreciation_ttm = _ttm(cf_q, cf_a,
                             "Depreciation And Amortization",
                             "Depreciation Amortization Depletion",
                             "Depreciation",
                             "Amortization Of Intangibles Assets")

    op_cf_ttm = _ttm(cf_q, cf_a,
                     "Operating Cash Flow",
                     "Total Cash From Operating Activities",
                     "Cash Flows From Operating Activities")

    capex_ttm = _ttm(cf_q, cf_a,
                     "Capital Expenditure",
                     "Capital Expenditures",
                     "Purchase Of Property Plant And Equipment",
                     "Purchases Of Property Plant And Equipment")

    # ── Balance Sheet (latest quarter or annual) ──────────────────────────────
    total_assets   = _latest(bs_q, bs_a, "Total Assets")
    current_liab   = _latest(bs_q, bs_a,
                             "Current Liabilities",
                             "Total Current Liabilities Net Minority Interest")
    total_equity   = _latest(bs_q, bs_a,
                             "Stockholders Equity",
                             "Total Stockholder Equity",
                             "Total Equity Gross Minority Interest")
    receivables    = _latest(bs_q, bs_a,
                             "Net Receivables", "Receivables", "Accounts Receivable")
    inventory      = _latest(bs_q, bs_a, "Inventory", "Inventories")
    payables       = _latest(bs_q, bs_a,
                             "Accounts Payable", "Payables And Accrued Expenses")

    # ── P/E — Market Cap / TTM Net Profit ─────────────────────────────────────
    raw_pe = float(info.get("trailingPE") or np.nan)
    if currency_mismatch:
        # financials in foreign currency — use yfinance pre-computed (currency-aware)
        pe_ratio = raw_pe if (not np.isnan(raw_pe) and 0 < raw_pe < 5000) else np.nan
    elif (not np.isnan(market_cap) and not np.isnan(net_income_ttm)
            and net_income_ttm > 0):
        pe_ratio = market_cap / net_income_ttm
    else:
        pe_ratio = raw_pe if (not np.isnan(raw_pe) and 0 < raw_pe < 5000) else np.nan

    # ── P/S — Market Cap / TTM Revenue ────────────────────────────────────────
    if (not currency_mismatch and not np.isnan(market_cap)
            and not np.isnan(revenue_ttm) and revenue_ttm > 0):
        ps_ratio = market_cap / revenue_ttm
    else:
        ps_ratio = float(info.get("priceToSalesTrailing12Months") or np.nan)

    # ── Sales Growth (YoY) ────────────────────────────────────────────────────
    if (not np.isnan(revenue_ttm) and not np.isnan(revenue_prior)
            and revenue_prior != 0):
        sales_growth = (revenue_ttm - revenue_prior) / abs(revenue_prior) * 100
    else:
        sales_growth = np.nan

    # ── ROCE = EBIT(TTM) / Capital Employed  (CE = Total Assets − Curr Liab) ─
    # ROCE is not meaningful for banks/NBFCs/insurance — their "current liabilities"
    # include customer deposits which inflates CE and distorts the ratio.
    _is_financial = "financial" in sector.lower()
    capital_employed = np.nan
    if not _is_financial:
        if not np.isnan(total_assets) and not np.isnan(current_liab):
            capital_employed = total_assets - current_liab

    if (not _is_financial and not np.isnan(ebit_ttm)
            and not np.isnan(capital_employed) and capital_employed > 0):
        roce = ebit_ttm / capital_employed * 100
    else:
        roce = np.nan

    # ── Operating Margin = EBITDA(TTM) / Revenue(TTM)  [EBIT + Depreciation] ─
    if (not np.isnan(ebit_ttm) and not np.isnan(revenue_ttm)
            and revenue_ttm > 0):
        if not np.isnan(depreciation_ttm):
            ebitda = ebit_ttm + abs(depreciation_ttm)   # D&A is negative in CF
        else:
            ebitda = ebit_ttm                           # fallback: EBIT margin
        operating_margin = ebitda / revenue_ttm * 100
    else:
        raw = float(info.get("operatingMargins") or np.nan)
        operating_margin = raw * 100 if not np.isnan(raw) else np.nan

    # ── Net Profit Margin = Net Income(TTM) / Revenue(TTM) ───────────────────
    if (not np.isnan(net_income_ttm) and not np.isnan(revenue_ttm)
            and revenue_ttm > 0):
        net_profit_margin = net_income_ttm / revenue_ttm * 100
    else:
        raw = float(info.get("profitMargins") or np.nan)
        net_profit_margin = raw * 100 if not np.isnan(raw) else np.nan

    # ── ROE = Net Income(TTM) / Shareholders' Equity ──────────────────────────
    if (not currency_mismatch and not np.isnan(net_income_ttm)
            and not np.isnan(total_equity) and total_equity > 0):
        roe = net_income_ttm / total_equity * 100
    else:
        raw = float(info.get("returnOnEquity") or np.nan)
        roe = raw * 100 if not np.isnan(raw) else np.nan

    # ── Net EPS = Net Income(TTM) / Shares Outstanding ────────────────────────
    if (not currency_mismatch and not np.isnan(net_income_ttm)
            and not np.isnan(shares) and shares > 0):
        net_eps = net_income_ttm / shares
    else:
        net_eps = float(info.get("trailingEps") or np.nan)

    # ── FCF and FCF Margin ────────────────────────────────────────────────────
    if not np.isnan(op_cf_ttm) and not np.isnan(capex_ttm):
        fcf = op_cf_ttm + capex_ttm          # capex stored as negative in yfinance CF
    else:
        fcf = _ttm(cf_q, cf_a, "Free Cash Flow")

    fcf_margin = (fcf / revenue_ttm * 100
                  if not np.isnan(fcf) and not np.isnan(revenue_ttm) and revenue_ttm > 0
                  else np.nan)

    # ── Capex / Sales ─────────────────────────────────────────────────────────
    capex_sales = (abs(capex_ttm) / revenue_ttm * 100
                   if not np.isnan(capex_ttm) and not np.isnan(revenue_ttm) and revenue_ttm > 0
                   else np.nan)

    # ── Receivable Days & Receivable/Sales  (denominator = TTM Revenue) ───────
    if not np.isnan(receivables) and not np.isnan(revenue_ttm) and revenue_ttm > 0:
        receivable_days  = receivables / revenue_ttm * 365
        receivable_sales = receivables / revenue_ttm * 100
    else:
        receivable_days = receivable_sales = np.nan

    # ── CCC: Inventory & Payable days use COGS (matching screener.in) ─────────
    # If COGS unavailable fall back to revenue as cost base
    cost_base = cogs_ttm if (not np.isnan(cogs_ttm) and cogs_ttm > 0) else revenue_ttm

    inv_days = (inventory / cost_base * 365
                if not np.isnan(inventory) and not np.isnan(cost_base) and cost_base > 0
                else np.nan)

    pay_days = (payables / cost_base * 365
                if not np.isnan(payables) and not np.isnan(cost_base) and cost_base > 0
                else np.nan)

    if not np.isnan(receivable_days) and not np.isnan(inv_days) and not np.isnan(pay_days):
        ccc = receivable_days + inv_days - pay_days
    elif not np.isnan(receivable_days):
        ccc = receivable_days          # partial CCC when inventory/payables missing
    else:
        ccc = np.nan

    # ── Promoter Holding & Change — from screener.in (direct BSE filing data) ─
    # Use the yf_ticker passed by the caller (e.g. "SINTERCOM.NS").
    # Deriving it from ticker_obj.ticker is unreliable after unpickling.
    sym_for_screener = yf_ticker or getattr(ticker_obj, "ticker", "") or ""
    (promoter_holding,
     change_promoter_holding,
     top_promoter_name,
     top_promoter_pct) = fetch_promoter_holding_screener(sym_for_screener)

    # Fallback for promoter_holding only: heldPercentInsiders
    if np.isnan(promoter_holding):
        ph_raw = float(info.get("heldPercentInsiders") or np.nan)
        promoter_holding = ph_raw * 100 if not np.isnan(ph_raw) else np.nan
    # change_promoter_holding / top_promoter_* stay NaN if screener.in was unavailable

    return {
        "price":                   price,
        "market_cap":              market_cap,           # INR — for UI cap-band tabs
        "sector":                  sector,
        "industry":                industry,
        "pe_ratio":                pe_ratio,             # lower = cheaper
        "ps_ratio":                ps_ratio,             # lower = cheaper
        "roce":                    roce,                 # % — higher = better
        "fcf":                     fcf,                  # absolute INR
        "fcf_margin":              fcf_margin,           # % of revenue
        "sales_growth":            sales_growth,         # % YoY
        "capex_sales":             capex_sales,          # % — moderate is best
        "receivable_sales":        receivable_sales,     # % — lower = better
        "receivable_days":         receivable_days,      # days — lower = better
        "ccc":                     ccc,                  # days — lower = better
        "roe":                     roe,                  # % — higher = better
        "operating_margin":        operating_margin,     # EBITDA% — higher = better
        "net_profit_margin":       net_profit_margin,    # % — higher = better
        "net_eps":                 net_eps,              # INR — higher = better
        "promoter_holding":        promoter_holding,          # total promoter group % from screener.in
        "change_promoter_holding": change_promoter_holding,   # QoQ change in pp
        "top_promoter_name":       top_promoter_name,         # name of largest individual promoter
        "top_promoter_pct":        top_promoter_pct,          # their individual holding %
        "promoter_buying":         np.nan,
        "order_book":              np.nan,
    }
