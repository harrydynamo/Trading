"""
Scores each stock out of 100 using fundamental ratios.

Score breakdown (100 pts total):
──────────────────────────────────────────────────────────────────────
  Category           Max pts   Metrics
──────────────────────────────────────────────────────────────────────
  1. Valuation          20     P/E ratio, Market Cap / Sales
  2. Profitability      25     ROCE, Operating Margin, FCF Margin
  3. Growth             20     Sales growth (YoY)
  4. Efficiency         20     Cash Conversion Cycle,
                               Receivable Days, Capex/Sales
  5. Quality            15     Net Profit Margin, ROE,
                               Receivable/Sales ratio
──────────────────────────────────────────────────────────────────────

Ratios not scored but stored and shown in reports / UI:
  Net EPS, Promoter Holding % (proxy via heldPercentInsiders)

Ratios NOT available via free APIs:
  Change in Promoter Holding, Promoter Buying, Order Book,
  Segmental Revenue, Sales Breakup
"""

import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScoreBreakdown:
    symbol:       str
    name:         str
    exchange:     str
    cap:          str
    price:        float

    valuation:    float = 0.0   # /20  — PE + PS ratio
    profitability:float = 0.0   # /25  — ROCE + FCF margin
    growth:       float = 0.0   # /20  — Sales growth YoY
    efficiency:   float = 0.0   # /20  — CCC + Receivable Days + Capex/Sales
    quality:      float = 0.0   # /15  — Receivable/Sales + ROE

    indicators:   dict  = field(default_factory=dict)
    error:        str   = ""

    @property
    def total(self) -> float:
        return (self.valuation + self.profitability + self.growth
                + self.efficiency + self.quality)

    @property
    def grade(self) -> str:
        t = self.total
        if t >= 80: return "A+"
        if t >= 70: return "A"
        if t >= 60: return "B+"
        if t >= 50: return "B"
        if t >= 40: return "C"
        return "D"

    @property
    def signal(self) -> str:
        t = self.total
        if t >= 75: return "STRONG BUY"
        if t >= 60: return "BUY"
        if t >= 45: return "WATCH"
        if t >= 30: return "NEUTRAL"
        return "AVOID"


def _safe(v, fallback=0.0):
    if v is None:
        return fallback
    try:
        return fallback if math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return fallback


def _has(v) -> bool:
    """True if value is a usable non-NaN number."""
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ─── Category 1: Valuation (20 pts) ──────────────────────────────────────────

def score_valuation(ind: dict) -> float:
    """
    20 points — Is the stock reasonably priced?

    P/E Ratio (10 pts):
      < 15        → 10   (value zone for Indian mid-caps)
      15–25       →  8   (fairly valued)
      25–40       →  5   (growth premium)
      40–60       →  2   (expensive)
      > 60 or ≤ 0 →  0   (overbought or losing money)

    Market Cap / Sales — P/S ratio (10 pts):
      < 1         → 10   (very cheap relative to revenue)
      1–2         →  8
      2–4         →  6
      4–6         →  3
      6–10        →  1
      > 10        →  0
    """
    pts = 0.0

    pe = _safe(ind.get("pe_ratio"), np.nan)
    if _has(pe):
        if   0 < pe < 15:  pts += 10
        elif pe < 25:      pts += 8
        elif pe < 40:      pts += 5
        elif pe < 60:      pts += 2

    ps = _safe(ind.get("ps_ratio"), np.nan)
    if _has(ps):
        if   ps < 1:   pts += 10
        elif ps < 2:   pts += 8
        elif ps < 4:   pts += 6
        elif ps < 6:   pts += 3
        elif ps < 10:  pts += 1

    return pts


# ─── Category 2: Profitability (25 pts) ──────────────────────────────────────

def score_profitability(ind: dict) -> float:
    """
    25 points — Is the business generating strong returns?

    ROCE (10 pts):
      > 25%  → 10   (exceptional capital efficiency)
      > 20%  →  8
      > 15%  →  6
      > 10%  →  3
      > 5%   →  1
      ≤ 5%   →  0

    Operating Profit Margin % (8 pts):
      > 25%  →  8
      > 20%  →  6
      > 15%  →  4
      > 10%  →  2
      > 5%   →  1
      ≤ 5%   →  0

    Free Cash Flow margin — FCF / Revenue % (7 pts):
      > 15%  →  7   (strong cash generation)
      > 10%  →  5
      >  5%  →  3
      >  0%  →  2   (positive FCF — business is self-funding)
      ≤  0%  →  0   (cash burn)
    """
    pts = 0.0

    roce = _safe(ind.get("roce"), np.nan)
    if _has(roce):
        if   roce > 25: pts += 10
        elif roce > 20: pts +=  8
        elif roce > 15: pts +=  6
        elif roce > 10: pts +=  3
        elif roce >  5: pts +=  1

    opm = _safe(ind.get("operating_margin"), np.nan)
    if _has(opm):
        if   opm > 25: pts += 8
        elif opm > 20: pts += 6
        elif opm > 15: pts += 4
        elif opm > 10: pts += 2
        elif opm >  5: pts += 1

    fcf_m = _safe(ind.get("fcf_margin"), np.nan)
    if _has(fcf_m):
        if   fcf_m > 15: pts += 7
        elif fcf_m > 10: pts += 5
        elif fcf_m >  5: pts += 3
        elif fcf_m >  0: pts += 2

    return pts


# ─── Category 3: Growth (20 pts) ─────────────────────────────────────────────

def score_growth(ind: dict) -> float:
    """
    20 points — Is the business growing revenue consistently?

    Sales growth YoY %:
      > 30%  → 20   (hyper-growth)
      > 20%  → 16
      > 15%  → 12
      > 10%  →  8
      >  5%  →  4
      0–5%   →  2   (slow but positive)
      < 0%   →  0   (revenue declining)
    """
    pts = 0.0

    sg = _safe(ind.get("sales_growth"), np.nan)
    if _has(sg):
        if   sg > 30: pts += 20
        elif sg > 20: pts += 16
        elif sg > 15: pts += 12
        elif sg > 10: pts +=  8
        elif sg >  5: pts +=  4
        elif sg >  0: pts +=  2

    return pts


# ─── Category 4: Efficiency (20 pts) ─────────────────────────────────────────

def score_efficiency(ind: dict) -> float:
    """
    20 points — How efficiently does the business manage working capital?

    Cash Conversion Cycle — CCC in days (7 pts):
      < 0    →  7   (collect before you pay — negative CCC is exceptional)
      < 30   →  6
      < 60   →  4
      < 90   →  2
      < 120  →  1
      ≥ 120  →  0

    Receivable Days (6 pts):
      < 30   →  6   (fast collections)
      < 60   →  4
      < 90   →  2
      ≥ 90   →  0

    Capex / Sales % (7 pts):
      2–8%   →  7   (healthy reinvestment for growth)
      8–15%  →  5   (capital-intensive but investing)
      0–2%   →  4   (asset-light but low reinvestment)
      15–25% →  2   (very capital-heavy)
      > 25%  →  0
    """
    pts = 0.0

    ccc = _safe(ind.get("ccc"), np.nan)
    if _has(ccc):
        if   ccc < 0:   pts += 7
        elif ccc < 30:  pts += 6
        elif ccc < 60:  pts += 4
        elif ccc < 90:  pts += 2
        elif ccc < 120: pts += 1

    rd = _safe(ind.get("receivable_days"), np.nan)
    if _has(rd):
        if   rd < 30: pts += 6
        elif rd < 60: pts += 4
        elif rd < 90: pts += 2

    cs = _safe(ind.get("capex_sales"), np.nan)
    if _has(cs):
        if   2  <= cs <  8:  pts += 7
        elif 8  <= cs < 15:  pts += 5
        elif 0  <= cs <  2:  pts += 4
        elif 15 <= cs < 25:  pts += 2

    return pts


# ─── Category 5: Quality (15 pts) ────────────────────────────────────────────

def score_quality(ind: dict) -> float:
    """
    15 points — Is the business high quality with good shareholder returns?

    Net Profit Margin % (5 pts):
      > 20%  →  5
      > 15%  →  4
      > 10%  →  3
      >  5%  →  1
      ≤  5%  →  0

    Return on Equity — ROE % (6 pts):
      > 25%  →  6
      > 20%  →  5
      > 15%  →  3
      > 10%  →  1
      ≤ 10%  →  0

    Receivable / Sales % (4 pts):
      < 10%  →  4   (very tight receivables — little credit risk)
      < 20%  →  3
      < 30%  →  2
      < 40%  →  1
      ≥ 40%  →  0
    """
    pts = 0.0

    npm = _safe(ind.get("net_profit_margin"), np.nan)
    if _has(npm):
        if   npm > 20: pts += 5
        elif npm > 15: pts += 4
        elif npm > 10: pts += 3
        elif npm >  5: pts += 1

    roe = _safe(ind.get("roe"), np.nan)
    if _has(roe):
        if   roe > 25: pts += 6
        elif roe > 20: pts += 5
        elif roe > 15: pts += 3
        elif roe > 10: pts += 1

    rs = _safe(ind.get("receivable_sales"), np.nan)
    if _has(rs):
        if   rs < 10: pts += 4
        elif rs < 20: pts += 3
        elif rs < 30: pts += 2
        elif rs < 40: pts += 1

    return pts


# ─── Master scorer ────────────────────────────────────────────────────────────

def score_stock(symbol: str, name: str, exchange: str, cap: str,
                ind: dict) -> ScoreBreakdown:
    """Compute the full fundamental score breakdown for a single stock."""
    result = ScoreBreakdown(
        symbol=symbol, name=name, exchange=exchange, cap=cap,
        price=_safe(ind.get("price"), 0),
        indicators=ind,
    )

    if not ind:
        result.error = "No indicator data"
        return result

    result.valuation    = round(score_valuation(ind),    2)
    result.profitability= round(score_profitability(ind),2)
    result.growth       = round(score_growth(ind),       2)
    result.efficiency   = round(score_efficiency(ind),   2)
    result.quality      = round(score_quality(ind),      2)

    return result
