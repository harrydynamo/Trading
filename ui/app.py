"""
Streamlit UI for NSE/BSE Fundamental Stock Screener.

Run with:
    streamlit run ui/app.py

The screener fetches and stores raw fundamental ratios for every stock.
Your sidebar weight inputs (%) define how important each ratio is —
stocks are ranked by a weighted percentile composite score computed live
from your inputs, with no pre-baked fixed scoring.
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import yfinance as yf
from stock_screener.universe import get_universe
from stock_screener.indicators import compute_fundamentals
from stock_screener.insider import fetch_promoter_buys


# ─── Constants ────────────────────────────────────────────────────────────────

RATIOS = [
    "pe_ratio",
    "ps_ratio",
    "roce",
    "operating_margin",
    "net_profit_margin",
    "fcf_margin",
    "sales_growth",
    "net_eps",
    "capex_sales",
    "receivable_sales",
    "receivable_days",
    "ccc",
    "roe",
    "promoter_holding",
    "change_promoter_holding",
]

RATIO_LABELS = {
    "pe_ratio":                "P/E Ratio",
    "ps_ratio":                "Market Cap / Sales (P/S)",
    "roce":                    "ROCE %",
    "operating_margin":        "Operating Profit Margin %",
    "net_profit_margin":       "Net Profit Margin %",
    "fcf_margin":              "FCF Margin %",
    "sales_growth":            "Sales Growth % (YoY)",
    "net_eps":                 "Net EPS (₹)",
    "capex_sales":             "Capex / Sales %",
    "receivable_sales":        "Receivable / Sales %",
    "receivable_days":         "Receivable Days",
    "ccc":                     "Cash Conversion Cycle (days)",
    "roe":                     "Return on Equity (ROE) %",
    "promoter_holding":        "Promoter Holding % (proxy)",
    "change_promoter_holding": "Change in Promoter Holding",
}

# +1 = higher is better, -1 = lower is better
RATIO_DIRECTION = {
    "pe_ratio":                -1,
    "ps_ratio":                -1,
    "roce":                    +1,
    "operating_margin":        +1,
    "net_profit_margin":       +1,
    "fcf_margin":              +1,
    "sales_growth":            +1,
    "net_eps":                 +1,
    "capex_sales":             -1,
    "receivable_sales":        -1,
    "receivable_days":         -1,
    "ccc":                     -1,
    "roe":                     +1,
    "promoter_holding":        +1,
    "change_promoter_holding": +1,   # increasing promoter holding = bullish signal
}

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "stock_screener", "data_cache"
)
CACHE_TTL_HOURS = 24


# ─── Cache helpers (reuse stock_screener's pickle cache) ──────────────────────

def _load_cache(symbol: str):
    path = os.path.join(CACHE_DIR, f"{symbol}_fundamentals.pkl")
    if not os.path.exists(path):
        return None
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    if age > timedelta(hours=CACHE_TTL_HOURS):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(symbol: str, data: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_fundamentals.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def _fetch_one(stock) -> dict | None:
    """Fetch raw fundamental ratios for one stock. Returns None on failure."""
    cached = _load_cache(stock.symbol)
    if cached:
        info, ticker_obj = cached["info"], cached["ticker_obj"]
    else:
        try:
            ticker_obj = yf.Ticker(stock.yf_ticker)
            info = ticker_obj.info or {}
            if not info or "marketCap" not in info:
                return None
            _save_cache(stock.symbol, {"info": info, "ticker_obj": ticker_obj})
        except Exception:
            return None

    try:
        ind = compute_fundamentals(ticker_obj, info, yf_ticker=stock.yf_ticker)
    except Exception:
        return None

    return {
        "symbol":   stock.symbol,
        "name":     stock.name,
        "exchange": stock.exchange,
        "cap":      stock.cap,
        **ind,
    }


# ─── Scoring ──────────────────────────────────────────────────────────────────

def compute_weighted_scores(df_raw: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Rank each ratio as a 0–1 percentile (direction-adjusted),
    then compute a weighted composite score (0–100).
    Missing values are filled with 0.5 (neutral percentile).
    """
    df = df_raw.copy()
    total_weight = sum(weights.values()) or 1

    weighted_cols = []
    for ratio, w in weights.items():
        if w <= 0 or ratio not in df.columns:
            continue
        col_name = f"_rank_{ratio}"
        ascending = RATIO_DIRECTION.get(ratio, 1) == -1  # lower→rank ascending → lowest gets high pct
        df[col_name] = (
            df[ratio]
            .rank(pct=True, ascending=ascending, na_option="keep")
            .fillna(0.5)
        )
        weighted_cols.append((col_name, w / total_weight))

    if weighted_cols:
        df["composite_score"] = sum(df[col] * w for col, w in weighted_cols) * 100
    else:
        df["composite_score"] = 0.0

    return df


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE/BSE Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state
for key in ("raw_df", "fetch_ts"):
    if key not in st.session_state:
        st.session_state[key] = None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Universe")
    exchange    = st.selectbox("Exchange",     ["Both", "NSE", "BSE"])
    cap         = st.selectbox("Market Cap",   ["All", "midcap", "smallcap", "microcap"])
    top_n       = st.slider("Show top N",       10, 100, 30)
    workers     = st.slider("Parallel workers",  5,  40, 20)
    force_ref   = st.checkbox("Force-refresh universe (ignore 24h cache)")

    st.divider()
    st.subheader("Ratios for Scoring")
    st.caption("Check to enable a ratio. Set its weight % (0 = disabled).")

    weights: dict[str, int] = {}
    for ratio in RATIOS:
        direction_hint = "↑" if RATIO_DIRECTION[ratio] == 1 else "↓"
        col_cb, col_num = st.columns([3, 2])
        with col_cb:
            checked = st.checkbox(
                f"{direction_hint}  {RATIO_LABELS[ratio]}",
                value=True,
                key=f"cb_{ratio}",
            )
        with col_num:
            pct = st.number_input(
                "Weight %",
                min_value=0,
                max_value=100,
                value=10 if checked else 0,
                step=5,
                key=f"w_{ratio}",
                label_visibility="collapsed",
                disabled=not checked,
            )
        weights[ratio] = int(pct) if checked else 0

    st.divider()

    with st.expander("📖 How scoring works"):
        st.markdown("""
**Composite Score (0 – 100)**

Each checked ratio is ranked as a **percentile** across all stocks in the
current universe (0 = worst, 1 = best). Direction is applied automatically:

| Direction | Meaning |
|-----------|---------|
| ↑ Higher is better | ROCE, OPM, NPM, FCF Margin, Sales Growth, EPS, ROE, Promoter Holding, Promo Change |
| ↓ Lower is better | P/E, P/S, Capex/Sales, Recv/Sales, Recv Days, CCC |

The per-ratio percentile scores are then **weighted and summed**:

```
Score = Σ (percentile_i × weight_i) / Σ(weight_i) × 100
```

Stocks with missing data for a ratio are assigned a **neutral 0.5 percentile**
(neither rewarded nor penalised).

---

**CLI Screener — Fixed-Threshold Score (also 0 – 100)**

The `run.py` CLI uses a separate rubric with fixed industry thresholds:

| Category | Max pts | Key metrics |
|----------|---------|-------------|
| Valuation | 20 | P/E < 15 → 10 pts; P/S < 1 → 10 pts |
| Profitability | 25 | ROCE > 25% → 10; OPM > 25% → 8; FCF > 15% → 7 |
| Growth | 20 | Sales growth > 30% → 20; > 20% → 16; … |
| Efficiency | 20 | CCC < 0 → 7; Recv Days < 30 → 6; Capex 2–8% → 7 |
| Quality | 15 | NPM > 20% → 5; ROE > 25% → 6; Recv/Sales < 10% → 4 |

Grades: **A+** ≥ 80 · **A** ≥ 70 · **B+** ≥ 60 · **B** ≥ 50 · **C** ≥ 40 · **D** < 40

Signals: **STRONG BUY** ≥ 75 · **BUY** ≥ 60 · **WATCH** ≥ 45 · **NEUTRAL** ≥ 30 · **AVOID** < 30
""")

    run_btn = st.button("🔍  Run Screener", type="primary", use_container_width=True)


# ─── Main area ────────────────────────────────────────────────────────────────

st.title("📊  NSE / BSE Fundamental Stock Screener")
st.caption(
    "Raw fundamental ratios are fetched and stored. "
    "Your sidebar weights define what matters — scoring is computed live from your inputs."
)

# ── Auto-load last CLI run if no session data yet ─────────────────────────────
_SCORES_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "stock_screener", "results", "scores.csv"
)
_CLI_COL_MAP = {
    "Symbol":                   "symbol",
    "Name":                     "name",
    "Exchange":                 "exchange",
    "Cap":                      "cap",
    "Price (₹)":                "price",
    "P/E Ratio":                "pe_ratio",
    "Market Cap/Sales":         "ps_ratio",
    "ROCE %":                   "roce",
    "Operating Margin %":       "operating_margin",
    "Net Profit Margin %":      "net_profit_margin",
    "FCF Margin %":             "fcf_margin",
    "Sales Growth % YoY":       "sales_growth",
    "Net EPS (₹)":              "net_eps",
    "Capex/Sales %":            "capex_sales",
    "Receivable/Sales %":       "receivable_sales",
    "Receivable Days":          "receivable_days",
    "Cash Conv. Cycle":         "ccc",
    "ROE %":                    "roe",
    "Promoter Holding % (proxy)": "promoter_holding",
    "Change in Promoter Holding": "change_promoter_holding",
}

if st.session_state.raw_df is None and os.path.exists(_SCORES_CSV):
    try:
        _csv_df = pd.read_csv(_SCORES_CSV)
        _csv_df = _csv_df.rename(columns=_CLI_COL_MAP)
        # Lowercase any remaining columns that match expected names
        _csv_df.columns = [c.lower() if c == c.upper() else c for c in _csv_df.columns]
        _mtime  = datetime.fromtimestamp(os.path.getmtime(_SCORES_CSV))
        st.session_state.raw_df  = _csv_df
        st.session_state.fetch_ts = _mtime.strftime("%d %b %Y  %H:%M") + " (loaded from last CLI run)"
        st.info(
            f"📂  Loaded **{len(_csv_df):,} stocks** from last screener run "
            f"({_mtime.strftime('%d %b %Y %H:%M')}).  "
            "Click **Run Screener** to refresh with live data.",
            icon="ℹ️",
        )
    except Exception:
        pass

# ── Fetch phase ───────────────────────────────────────────────────────────────
if run_btn:
    exchange_arg = None if exchange == "Both" else exchange
    cap_arg      = None if cap      == "All"  else cap

    with st.spinner("Loading stock universe…"):
        stocks = get_universe(
            exchange=exchange_arg,
            cap=cap_arg,
            force_refresh=force_ref,
        )

    progress_bar = st.progress(0.0, text=f"Fetching fundamentals for {len(stocks)} stocks…")
    raw_rows: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, s): s for s in stocks}
        for future in as_completed(futures):
            done += 1
            progress_bar.progress(done / len(stocks), text=f"Fetched {done}/{len(stocks)}…")
            result = future.result()
            if result:
                raw_rows.append(result)

    progress_bar.empty()

    if raw_rows:
        st.session_state.raw_df  = pd.DataFrame(raw_rows)
        st.session_state.fetch_ts = datetime.now().strftime("%d %b %Y  %H:%M")
        skipped = len(stocks) - len(raw_rows)
        st.success(
            f"✅  {len(raw_rows)} stocks loaded"
            + (f"  ({skipped} skipped — no data / delisted)" if skipped else "")
        )
    else:
        st.error("No data returned. Check filters or network connection.")


# ── Results phase ─────────────────────────────────────────────────────────────
if st.session_state.raw_df is not None:
    raw_df = st.session_state.raw_df

    # Filter: only show profitable companies
    # A stock is kept only when NEITHER metric signals a loss,
    # AND at least one confirms positive profit.
    eps = raw_df["net_eps"].fillna(np.nan)          if "net_eps"          in raw_df.columns else pd.Series(np.nan, index=raw_df.index)
    npm = raw_df["net_profit_margin"].fillna(np.nan) if "net_profit_margin" in raw_df.columns else pd.Series(np.nan, index=raw_df.index)

    profitable_mask = (
        (eps.isna() | (eps > 0)) &      # EPS not negative
        (npm.isna() | (npm > 0)) &      # Net profit margin not negative
        ((eps > 0) | (npm > 0))         # At least one confirms profit
    )
    filtered_df = raw_df[profitable_mask]
    excluded    = len(raw_df) - len(filtered_df)

    # Recompute scores whenever weights change (no re-fetch needed)
    scored_df = compute_weighted_scores(filtered_df, weights)
    scored_df = scored_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    scored_df.index += 1

    excl_note = f"  |  {excluded} excluded (negative/no EPS)" if excluded else ""
    st.caption(f"Data fetched: {st.session_state.fetch_ts}  |  {len(scored_df)} stocks{excl_note}")

    # ── Top-5 metric cards ────────────────────────────────────────────────────
    top5   = scored_df.head(5)
    metric_cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        with metric_cols[i]:
            st.metric(
                label=row["symbol"],
                value=f"{row['composite_score']:.1f}",
                delta=str(row["name"])[:22] if pd.notna(row.get("name")) else "",
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    CR = 1e7   # 1 Crore = 10,000,000 INR

    def mcap_band(mcap):
        if pd.isna(mcap) or mcap <= 0:
            return "Unknown"
        cr = mcap / CR
        if cr <= 250:        return "0–250 Cr"
        elif cr <= 500:      return "250–500 Cr"
        elif cr <= 2500:     return "500–2,500 Cr"
        else:                return "Above 2,500 Cr"

    def mcap_cr_fmt(mcap):
        if pd.isna(mcap) or mcap <= 0:
            return "—"
        cr = mcap / CR
        if cr >= 1000:
            return f"₹{cr/1000:.1f}K Cr"
        return f"₹{cr:.0f} Cr"

    _mcap_col = "market_cap" if "market_cap" in scored_df.columns else None
    scored_df["mcap_band"] = scored_df[_mcap_col].apply(mcap_band) if _mcap_col else "—"
    scored_df["mcap_cr"]   = scored_df[_mcap_col].apply(mcap_cr_fmt) if _mcap_col else "—"

    ratio_cols_present = [r for r in RATIOS if r in scored_df.columns]

    def _make_display(df):
        extra = []
        if "mcap_cr" in df.columns:    extra.append("mcap_cr")
        if "sector"  in df.columns:    extra.append("sector")
        cols = ["symbol", "name", "exchange", "cap"] + extra + ["price", "composite_score"] + ratio_cols_present
        cols = [c for c in cols if c in df.columns]
        out = df[cols].head(top_n).copy()
        rename = {
            "symbol":          "Symbol",
            "name":            "Name",
            "exchange":        "Exchange",
            "cap":             "Cap",
            "mcap_cr":         "Mkt Cap",
            "sector":          "Sector",
            "price":           "Price ₹",
            "composite_score": "Score (0–100)",
        }
        rename.update({r: RATIO_LABELS[r] for r in ratio_cols_present})
        return out.rename(columns=rename)

    def _score_color(val):
        if not isinstance(val, (int, float)):
            return ""
        if val >= 75: return "background-color:#1a7a3c; color:white"
        if val >= 60: return "background-color:#2ecc71; color:black"
        if val >= 45: return "background-color:#f39c12; color:black"
        return "background-color:#c0392b; color:white"

    def _render_table(df, key_suffix=""):
        show = _make_display(df)
        styled = show.style.applymap(_score_color, subset=["Score (0–100)"])
        st.dataframe(styled, use_container_width=True, height=600)
        st.download_button(
            "⬇️  Download CSV",
            data=show.to_csv().encode(),
            file_name=f"screener_{key_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key=f"dl_{key_suffix}",
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_micro, tab_small, tab_mid, tab_others, tab_promo, tab_sector, tab_charts, tab_detail = st.tabs([
        "🔵 0–250 Cr",
        "🟡 250–500 Cr",
        "🟢 500–2,500 Cr",
        "🔶 Others (>2,500 Cr)",
        "🛒 Promoter Buying",
        "🏭 By Sector",
        "📈 Charts",
        "🔍 Stock Detail",
    ])

    with tab_micro:
        band_df = scored_df[scored_df["mcap_band"] == "0–250 Cr"]
        st.subheader(f"0–250 Cr — {len(band_df)} stocks")
        if band_df.empty:
            st.info("No stocks in this market cap range.")
        else:
            _render_table(band_df, "0_250cr")

    with tab_small:
        band_df = scored_df[scored_df["mcap_band"] == "250–500 Cr"]
        st.subheader(f"250–500 Cr — {len(band_df)} stocks")
        if band_df.empty:
            st.info("No stocks in this market cap range.")
        else:
            _render_table(band_df, "250_500cr")

    with tab_mid:
        band_df = scored_df[scored_df["mcap_band"] == "500–2,500 Cr"]
        st.subheader(f"500–2,500 Cr — {len(band_df)} stocks")
        if band_df.empty:
            st.info("No stocks in this market cap range.")
        else:
            _render_table(band_df, "500_2500cr")

    with tab_others:
        band_df = scored_df[scored_df["mcap_band"] == "Above 2,500 Cr"]
        st.subheader(f"Above 2,500 Cr — {len(band_df)} stocks")
        if band_df.empty:
            st.info("No stocks in this market cap range.")
        else:
            _render_table(band_df, "above_2500cr")

    with tab_promo:

        # ── Section 1: NSE live insider filings ───────────────────────────────
        st.subheader("NSE Insider Filings — Promoter Buys (Live)")
        st.caption(
            "Pulls promoter buy disclosures directly from NSE (SEBI PIT regulation). "
            "Promoters must file within **2 trading days** of any transaction. "
            "Not limited to your screened universe — covers all NSE-listed stocks."
        )

        nse_days = st.slider("Look-back window (days)", 7, 90, 30, key="nse_days")
        fetch_nse_btn = st.button("🔄  Fetch NSE Insider Trades", key="fetch_nse")

        if "nse_insider_df" not in st.session_state:
            st.session_state.nse_insider_df = None

        if fetch_nse_btn:
            with st.spinner("Fetching from NSE…"):
                st.session_state.nse_insider_df = fetch_promoter_buys(days=nse_days)

        nse_df = st.session_state.nse_insider_df

        if nse_df is not None:
            if nse_df.empty:
                st.warning(
                    "No promoter buy filings returned. "
                    "NSE may have blocked the request — try again in a few seconds, "
                    "or check your internet connection."
                )
            else:
                ni1, ni2, ni3, ni4 = st.columns(4)
                ni1.metric("Filings", len(nse_df))
                ni2.metric("Unique stocks", nse_df["symbol"].nunique())
                total_val = nse_df["transaction_value_lakh"].sum()
                ni3.metric(
                    "Total value",
                    f"₹{total_val/100:.1f} Cr" if not np.isnan(total_val) else "—",
                )
                ni4.metric(
                    "Latest filing",
                    nse_df["transaction_date"].max().strftime("%d %b %Y")
                    if not nse_df["transaction_date"].isna().all() else "—",
                )

                st.divider()

                # Aggregate by symbol so each row = one company
                agg = (
                    nse_df.groupby("symbol", as_index=False)
                    .agg(
                        company        =("company",               "first"),
                        filings        =("promoter_name",         "count"),
                        promoters      =("promoter_name",         lambda x: ", ".join(sorted(set(x)))),
                        total_shares   =("shares_bought",         "sum"),
                        total_value_lakh=("transaction_value_lakh","sum"),
                        holding_before =("holding_before_pct",    "first"),
                        holding_after  =("holding_after_pct",     "last"),
                        change_pct     =("change_pct",            "sum"),
                        latest_date    =("transaction_date",      "max"),
                    )
                    .sort_values("total_value_lakh", ascending=False)
                    .reset_index(drop=True)
                )
                agg.index += 1

                def _fmt_val(v):
                    if np.isnan(v): return "—"
                    if v >= 100:    return f"₹{v/100:.1f} Cr"
                    return f"₹{v:.1f} L"

                agg_show = agg.copy()
                agg_show["total_value_lakh"] = agg_show["total_value_lakh"].apply(_fmt_val)
                agg_show["total_shares"]     = agg_show["total_shares"].apply(
                    lambda x: f"{int(x):,}" if not np.isnan(x) else "—"
                )
                agg_show["latest_date"] = agg_show["latest_date"].dt.strftime("%d %b %Y")
                agg_show = agg_show.rename(columns={
                    "symbol":            "Symbol",
                    "company":           "Company",
                    "filings":           "Filings",
                    "promoters":         "Promoter(s)",
                    "total_shares":      "Shares Bought",
                    "total_value_lakh":  "Value",
                    "holding_before":    "Holding Before %",
                    "holding_after":     "Holding After %",
                    "change_pct":        "Change (% pts)",
                    "latest_date":       "Latest Filing",
                })

                def _nse_change_color(val):
                    if not isinstance(val, (int, float)): return ""
                    if val > 2:   return "background-color:#1a7a3c; color:white"
                    if val > 0:   return "background-color:#2ecc71; color:black"
                    return ""

                st.dataframe(
                    agg_show.style.applymap(_nse_change_color, subset=["Change (% pts)"]),
                    use_container_width=True,
                    height=500,
                )
                st.download_button(
                    "⬇️  Download CSV",
                    data=agg_show.to_csv().encode(),
                    file_name=f"nse_insider_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="dl_nse_insider",
                )

                # Raw filings expander
                with st.expander("View individual filings"):
                    raw_show = nse_df.copy()
                    raw_show["transaction_date"] = raw_show["transaction_date"].dt.strftime("%d %b %Y")
                    raw_show["transaction_value_lakh"] = raw_show["transaction_value_lakh"].apply(_fmt_val)
                    raw_show["shares_bought"] = raw_show["shares_bought"].apply(
                        lambda x: f"{int(x):,}" if not np.isnan(x) else "—"
                    )
                    st.dataframe(raw_show, use_container_width=True, height=400)

        st.divider()

        # ── Section 2: Quarterly snapshot from screener.in (screened universe) ─
        st.subheader("Quarterly Snapshot — Screened Universe")
        st.caption(
            "Stocks from your screened universe where promoters increased their "
            "stake in the most recent quarter (sourced from screener.in)."
        )

        if "change_promoter_holding" not in scored_df.columns or "promoter_holding" not in scored_df.columns:
            st.info("Promoter holding data not available. Re-run the screener.")
        else:
            promo_df = (
                scored_df[scored_df["change_promoter_holding"] > 0]
                .copy()
                .sort_values("change_promoter_holding", ascending=False)
            )

            if promo_df.empty:
                st.info("No stocks with promoter buying found in this universe.")
            else:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Stocks with promoter buying", len(promo_df))
                mc2.metric("Avg increase", f"{promo_df['change_promoter_holding'].mean():.2f}%")
                mc3.metric(
                    "Largest single increase",
                    f"{promo_df['change_promoter_holding'].max():.2f}%  "
                    f"({promo_df.loc[promo_df['change_promoter_holding'].idxmax(), 'symbol']})",
                )

                st.divider()

                disp_cols = ["symbol", "name", "exchange", "cap", "mcap_cr"]
                if "sector" in promo_df.columns:
                    disp_cols.append("sector")
                disp_cols += [
                    "price",
                    "promoter_holding", "top_promoter_pct", "top_promoter_name",
                    "change_promoter_holding", "composite_score",
                ]
                disp_cols = [c for c in disp_cols if c in promo_df.columns]

                promo_show = promo_df[disp_cols].copy().rename(columns={
                    "symbol":                  "Symbol",
                    "name":                    "Name",
                    "exchange":                "Exchange",
                    "cap":                     "Cap",
                    "mcap_cr":                 "Mkt Cap",
                    "sector":                  "Sector",
                    "price":                   "Price ₹",
                    "promoter_holding":        "Total Promoter %",
                    "top_promoter_pct":        "Highest Promoter %",
                    "top_promoter_name":       "Largest Promoter",
                    "change_promoter_holding": "Change (% pts)",
                    "composite_score":         "Score (0–100)",
                })

                def _change_color(val):
                    if not isinstance(val, (int, float)): return ""
                    if val > 2:   return "background-color:#1a7a3c; color:white"
                    if val > 0.5: return "background-color:#2ecc71; color:black"
                    return "background-color:#f39c12; color:black"

                st.dataframe(
                    promo_show.style
                    .applymap(_change_color, subset=["Change (% pts)"])
                    .applymap(_score_color,  subset=["Score (0–100)"]),
                    use_container_width=True,
                    height=500,
                )
                st.download_button(
                    "⬇️  Download CSV",
                    data=promo_show.to_csv().encode(),
                    file_name=f"promoter_buying_quarterly_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="dl_promo",
                )

                st.divider()
                st.subheader("Change in Promoter Holding — Top 20")
                chart_df = promo_df.head(20)
                fig_promo = px.bar(
                    chart_df,
                    x="change_promoter_holding",
                    y="symbol",
                    orientation="h",
                    color="change_promoter_holding",
                    color_continuous_scale="Greens",
                    hover_data={"promoter_holding": ":.2f", "composite_score": ":.1f"},
                    labels={
                        "change_promoter_holding": "Change (% pts)",
                        "symbol":                  "Stock",
                        "promoter_holding":        "Current Holding %",
                        "composite_score":         "Score",
                    },
                    title="Promoter Stake Increase — Latest Quarter",
                    height=max(400, len(chart_df) * 30),
                )
                fig_promo.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_promo, use_container_width=True)

    with tab_sector:
        st.subheader("Stocks by Sector")
        if "sector" not in scored_df.columns:
            st.info("Sector data not available.")
        else:
            sectors = sorted(scored_df["sector"].dropna().unique().tolist())
            sectors = [s for s in sectors if s and s != "Unknown"]

            # Sector overview bar chart
            sector_counts = (
                scored_df[scored_df["sector"].isin(sectors)]
                .groupby("sector")
                .agg(count=("symbol", "count"), avg_score=("composite_score", "mean"))
                .reset_index()
                .sort_values("avg_score", ascending=False)
            )
            fig_sec = px.bar(
                sector_counts, x="avg_score", y="sector",
                orientation="h",
                color="avg_score",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                hover_data={"count": True},
                labels={"avg_score": "Avg Score", "sector": "", "count": "Stocks"},
                title="Average Weighted Score by Sector",
                height=max(300, len(sectors) * 30),
            )
            fig_sec.update_layout(
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_sec, use_container_width=True)

            st.divider()
            selected_sector = st.selectbox("Drill into sector", ["All"] + sectors, key="sector_sel")
            if selected_sector == "All":
                sec_df = scored_df
            else:
                sec_df = scored_df[scored_df["sector"] == selected_sector]
            st.write(f"**{len(sec_df)} stocks** in *{selected_sector}*")
            _render_table(sec_df, f"sector_{selected_sector[:10]}")

    with tab_charts:
        st.subheader("Composite Score — Top 30")
        bar_df = scored_df.head(min(30, top_n))
        fig_bar = px.bar(
            bar_df, x="composite_score", y="symbol",
            orientation="h",
            color="composite_score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            labels={"composite_score": "Weighted Score", "symbol": "Stock"},
            title="Top Stocks by Your Weighted Composite Score",
            height=620,
        )
        fig_bar.update_layout(
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        st.subheader("Ratio Scatter Plot")
        col1, col2 = st.columns(2)
        with col1:
            default_x = ratio_cols_present.index("pe_ratio") if "pe_ratio" in ratio_cols_present else 0
            x_ratio = st.selectbox("X axis", ratio_cols_present, index=default_x, key="scatter_x")
        with col2:
            default_y = ratio_cols_present.index("roce") if "roce" in ratio_cols_present else min(1, len(ratio_cols_present)-1)
            y_ratio = st.selectbox("Y axis", ratio_cols_present, index=default_y, key="scatter_y")

        scatter_df = (
            scored_df[["symbol", "name", "composite_score", x_ratio, y_ratio]]
            .dropna(subset=[x_ratio, y_ratio])
            .head(top_n)
        )
        fig_scat = px.scatter(
            scatter_df,
            x=x_ratio, y=y_ratio,
            color="composite_score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            hover_name="symbol",
            hover_data={"name": True, "composite_score": ":.1f"},
            labels={
                x_ratio: RATIO_LABELS.get(x_ratio, x_ratio),
                y_ratio: RATIO_LABELS.get(y_ratio, y_ratio),
                "composite_score": "Score",
            },
            title=f"{RATIO_LABELS.get(x_ratio, x_ratio)}  vs  {RATIO_LABELS.get(y_ratio, y_ratio)}",
            height=500,
        )
        fig_scat.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1a2e",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with tab_detail:
        st.subheader("Individual Stock Deep Dive")
        selected_sym = st.selectbox("Select stock", scored_df["symbol"].tolist(), key="detail_sym")
        row = scored_df[scored_df["symbol"] == selected_sym].iloc[0]

        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.metric("Composite Score", f"{row['composite_score']:.1f} / 100")
            price_val = row.get("price", np.nan)
            st.metric(
                "Price",
                f"₹{float(price_val):,.2f}" if pd.notna(price_val) and price_val else "—"
            )
            st.write(f"**Exchange:** {row['exchange']}  |  **Cap:** {row['cap']}")
            st.write(f"**Mkt Cap:** {mcap_cr_fmt(row.get('market_cap', np.nan))}  |  **Band:** {row.get('mcap_band', '—')}")
            st.write(f"**Sector:** {row.get('sector', '—')}  |  **Industry:** {row.get('industry', '—')}")
            st.write(f"**Name:** {row['name']}")

        with col_r:
            pct_rows = []
            for ratio in ratio_cols_present:
                rank_col = f"_rank_{ratio}"
                raw_val  = row.get(ratio, np.nan)
                pct_val  = float(row.get(rank_col, 0.5)) * 100
                pct_rows.append({
                    "Ratio":       RATIO_LABELS.get(ratio, ratio),
                    "percentile":  round(pct_val, 1),
                    "Raw Value":   f"{raw_val:.2f}" if pd.notna(raw_val) else "N/A",
                    "Weight %":    weights.get(ratio, 0),
                })
            pct_df = pd.DataFrame(pct_rows)
            fig_pct = px.bar(
                pct_df,
                x="percentile", y="Ratio",
                orientation="h",
                color="percentile",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                hover_data={"Raw Value": True, "Weight %": True},
                labels={"percentile": "Percentile vs peers (0=worst, 100=best)"},
                title=f"{selected_sym} — Percentile Rank per Ratio",
                height=420,
            )
            fig_pct.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1a2e",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        st.subheader("Raw Fundamental Ratios")
        ratio_table = pd.DataFrame({
            "Ratio":           [RATIO_LABELS.get(r, r) for r in ratio_cols_present],
            "Raw Value":       [
                f"{row[r]:.2f}" if pd.notna(row.get(r)) else "N/A"
                for r in ratio_cols_present
            ],
            "Direction":       [
                "↑ Higher is better" if RATIO_DIRECTION[r] == 1 else "↓ Lower is better"
                for r in ratio_cols_present
            ],
            "Weight %":        [weights.get(r, 0) for r in ratio_cols_present],
        })
        st.dataframe(ratio_table, use_container_width=True, hide_index=True)

        st.caption(
            "**Not available via free APIs** (require BSE filings / annual reports): "
            "Promoter Holding %, Promoter Buying, Order Book, Segmental Revenue, Sales Breakup."
        )

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.info("👈  Configure settings in the sidebar and click **Run Screener** to begin.")
    st.markdown("""
### How it works
1. **Select ratios** — check which fundamentals to include in scoring (sidebar checkboxes)
2. **Run the screener** — live data is fetched from NSE/BSE via yfinance and raw ratios are stored
3. **Review rankings** — stocks are ranked by a composite percentile score across selected ratios
4. **Adjust on the fly** — toggle ratio checkboxes without re-fetching; rankings update instantly
5. **Drill into details** — see each stock's raw ratios and percentile position vs all peers

### Available Ratios
| Ratio | Direction | Description |
|-------|-----------|-------------|
| P/E Ratio | ↓ Lower = better | Price vs earnings — lower means cheaper |
| Market Cap / Sales (P/S) | ↓ Lower = better | Price vs revenue — lower means cheaper |
| ROCE % | ↑ Higher = better | Return on Capital Employed — efficiency of capital |
| Operating Profit Margin % | ↑ Higher = better | EBIT / Revenue — operating efficiency |
| Net Profit Margin % | ↑ Higher = better | Net income as % of revenue |
| FCF Margin % | ↑ Higher = better | Free cash flow as % of revenue |
| Sales Growth % (YoY) | ↑ Higher = better | Revenue growth year-on-year |
| Net EPS (₹) | ↑ Higher = better | Trailing 12-month earnings per share |
| Capex / Sales % | ↓ Lower = better | Capital investment as % of revenue |
| Receivable / Sales % | ↓ Lower = better | Revenue tied up in receivables — lower = less credit risk |
| Receivable Days | ↓ Lower = better | How quickly customers pay |
| Cash Conversion Cycle | ↓ Lower = better | Working capital efficiency in days |
| ROE % | ↑ Higher = better | Return on shareholders equity |
| Promoter Holding % (proxy) | ↑ Higher = better | Insider/promoter ownership via heldPercentInsiders |
| Change in Promoter Holding | ↑ Higher = better | QoQ change in promoter stake — requires BSE filings (shows N/A) |

> **Note:** Change in Promoter Holding requires BSE quarterly shareholding filings and is
> not available via any free API. The weight slider is present for when data becomes available.
> Other unavailable fields: Promoter Buying, Order Book, Segmental Revenue, Sales Breakup.
""")
