"""Live market data from Yahoo Finance (via yfinance), cached on disk.

Prices, index levels, quote-derived ratios and company profiles are fetched on
demand and cached to `data/live_cache` with a TTL, so a conversation that asks
five questions about one company costs one fetch, not five. Filed financials come
from SEC EDGAR instead (see `sec`), which is the authoritative source for them.

Nothing here needs a login or an API key.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import DATA

CACHE = DATA / "live_cache"
CACHE.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept-Language": "en-US,en;q=0.9"}

# seconds
TTL_PRICES = 15 * 60
TTL_INFO = 6 * 3600

# name -> Yahoo ticker, so `benchmark=` and index_data accept plain names.
INDEX_TICKERS = {
    "sp500": "^GSPC", "nasdaq": "^IXIC", "dowjones": "^DJI",
}


# ------------------------------------------------------------------ caching --
def _cache_path(kind: str, key: str, ext: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    d = CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.{ext}"


def _fresh(path: Path, ttl: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl


# ------------------------------------------------------------------ symbols --
def to_ticker(symbol: str) -> str:
    """Map a symbol to its Yahoo ticker.

    Share classes are written with a dot in filings and most quotes (BRK.B) but
    with a hyphen on Yahoo (BRK-B). Index carets (^GSPC) and futures/FX suffixes
    (GC=F) pass through unchanged.
    """
    s = symbol.strip().upper()
    if s.startswith("^") or "=" in s:
        return s
    return s.replace(".", "-")


# ------------------------------------------------------------------- prices --
def prices(symbol: str, period: str = "max") -> pd.DataFrame | None:
    """Daily OHLCV with both `Close` and `Adj Close`.

    Callers rely on the split: nominal Close for quoted levels and 52-week ranges,
    Adj Close for anything performance-related (unadjusted series invent -50%
    drawdowns on split dates).
    """
    import yfinance as yf

    tk = to_ticker(symbol)
    p = _cache_path("prices", tk, "parquet")
    if _fresh(p, TTL_PRICES):
        return pd.read_parquet(p)

    def _stale():
        """Serve the cached copy when the network refuses.

        Yahoo rate-limits sustained use and then rejects EVERY endpoint for a
        while. Expiring a TTL must not mean losing data we already hold: a price
        series that is a few hours or days old still answers almost every question,
        and callers surface the `as_of` date anyway. Returning None here would take
        the whole tool suite down for the length of a cooldown.
        """
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                return None
        return None

    try:
        raw = yf.Ticker(tk).history(period=period, auto_adjust=False)
    except Exception:
        return _stale()
    if raw is None or raw.empty:
        return _stale()
    df = raw.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if c in df.columns]
    df = df[keep].sort_values("Date")
    # Yahoo emits a partial bar for the session in progress, with Close = NaN until
    # it settles. Callers all read .iloc[-1] for "current price", so leaving it in
    # turns every latest-value metric into NaN. Drop unpriced rows.
    df = df[df["Close"].notna()].reset_index(drop=True)
    if df.empty:
        return _stale()

    # Never let a fetch shrink what we hold. A throttled or partial response can
    # come back with a truncated window, and blindly writing it would destroy years
    # of history to save a few hundred KB. Union on Date, preferring the fresh rows
    # where they overlap (splits restate Adj Close).
    prev = _stale()
    if prev is not None and not prev.empty:
        combined = pd.concat([prev, df], ignore_index=True)
        combined = (combined.sort_values("Date")
                            .drop_duplicates(subset="Date", keep="last")
                            .reset_index(drop=True))
        if len(combined) > len(df):
            df = combined
    df.to_parquet(p, index=False)
    return df


def index_prices(name: str) -> pd.DataFrame | None:
    tk = INDEX_TICKERS.get(str(name).lower().replace(" ", ""), None)
    if tk is None:
        tk = name if str(name).startswith("^") else None
    if tk is None:
        return None
    return prices(tk)


def index_list() -> list[str]:
    return sorted(INDEX_TICKERS)


# ------------------------------------------------------------------ profile --
def yf_info(symbol: str) -> dict | None:
    """Raw yfinance `.info`, cached. ~170 fields for a large cap."""
    import yfinance as yf

    tk = to_ticker(symbol)
    p = _cache_path("info", tk, "json")

    def _cached():
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    if _fresh(p, TTL_INFO):
        got = _cached()
        if got:
            return got
    try:
        i = yf.Ticker(tk).info
    except Exception:
        return _cached()          # rate-limited: stale ratios beat none
    if not i:
        return _cached()
    p.write_text(json.dumps(i, default=str), encoding="utf-8")
    return i


def _usd_m(v) -> float | None:
    return round(v / 1e6, 1) if isinstance(v, (int, float)) and v else None


def profile(symbol: str) -> dict | None:
    """Business description and identity fields from the live quote."""
    i = yf_info(symbol) or {}
    if not i:
        return None
    return {
        "symbol": symbol.upper(),
        "company_name": i.get("longName") or i.get("shortName"),
        "about": i.get("longBusinessSummary"),
        "sector": i.get("sector"),
        "industry": i.get("industry"),
        "market_cap_usd_m": _usd_m(i.get("marketCap")),
        "currency": i.get("currency"),
        "exchange": i.get("exchange"),
        "website": i.get("website"),
        "employees": i.get("fullTimeEmployees"),
        "source": "yfinance (live)",
    }


def valuation(symbol: str) -> dict | None:
    """Valuation multiples, margins and consensus figures from live quotes."""
    i = yf_info(symbol)
    if not i:
        return None
    def pct(k):
        v = i.get(k)
        return round(v * 100, 2) if isinstance(v, (int, float)) else None
    return {
        "symbol": symbol.upper(),
        "as_of": time.strftime("%Y-%m-%d"),
        "current_price": i.get("currentPrice") or i.get("regularMarketPrice"),
        "currency": i.get("currency"),
        "market_cap_usd_m": _usd_m(i.get("marketCap")),
        "enterprise_value_usd_m": _usd_m(i.get("enterpriseValue")),
        "trailing_pe": i.get("trailingPE"), "forward_pe": i.get("forwardPE"),
        "price_to_book": i.get("priceToBook"),
        "price_to_sales": i.get("priceToSalesTrailing12Months"),
        "ev_to_ebitda": i.get("enterpriseToEbitda"),
        "ev_to_revenue": i.get("enterpriseToRevenue"),
        "peg_ratio": i.get("pegRatio") or i.get("trailingPegRatio"),
        "roe_pct": pct("returnOnEquity"), "roa_pct": pct("returnOnAssets"),
        "profit_margin_pct": pct("profitMargins"),
        "operating_margin_pct": pct("operatingMargins"),
        "debt_to_equity": i.get("debtToEquity"),
        "current_ratio": i.get("currentRatio"),
        "dividend_yield": i.get("dividendYield"),
        "book_value": i.get("bookValue"),
        "beta": i.get("beta"),
        "fifty_two_week_high": i.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": i.get("fiftyTwoWeekLow"),
        "analyst_target_mean": i.get("targetMeanPrice"),
        "analyst_recommendation": i.get("recommendationKey"),
        "analyst_count": i.get("numberOfAnalystOpinions"),
        "source": "yfinance (live) — analyst targets are third-party consensus, "
                  "reported as data, not as this agent's own price target",
    }


# ---------------------------------------------------------------- documents --
# Some hosts answer with a placeholder page instead of the document. That must
# surface as a failed fetch, not as a document: a model handed the placeholder as
# a filing will summarise it.
_PLACEHOLDER_MARKERS = (
    "page you are looking for has been moved",
    "page you are looking for does not exist",
    "access denied",
)
_DOC_HEADERS = {**UA, "Accept": "application/pdf,application/octet-stream,*/*;q=0.8"}


def _looks_like_document(text: str) -> bool:
    body = text.strip()
    low = body.lower()
    if any(m in low for m in _PLACEHOLDER_MARKERS):
        return False
    letters = sum(ch.isalpha() for ch in body)
    return letters >= 400        # a real document has pages of prose; a stub does not


def document_text(url: str, max_pages: int = 40) -> str | None:
    """Download a PDF and extract its text.

    Returns None for anything that is not a real document (fetch error, non-PDF
    body, or a placeholder page), and never caches those.
    """
    import pymupdf

    p = _cache_path("docs", url[-80:], "txt")
    if p.exists():
        cached = p.read_text(encoding="utf-8")
        if _looks_like_document(cached):
            return cached
        p.unlink(missing_ok=True)
    try:
        r = requests.get(url, headers=_DOC_HEADERS, timeout=90, allow_redirects=True)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            return None
        doc = pymupdf.open(stream=r.content, filetype="pdf")
    except Exception:
        return None
    text = "\n".join(page.get_text() for page in list(doc)[:max_pages])
    if not _looks_like_document(text):
        return None
    if len(doc) > max_pages:
        text += f"\n\n[truncated — {len(doc) - max_pages} further pages not shown]"
    p.write_text(text, encoding="utf-8")
    return text


# -------------------------------------------------------------------- macro --
# Only series with a genuine free live source are offered; anything else returns
# None so callers report 'unavailable' instead of quietly substituting a proxy.
MACRO_TICKERS = {
    "treasury_10y": "^TNX",      # 10-year Treasury yield, in percent
    "crude": "BZ=F",             # Brent crude futures, USD/bbl
    "gold": "GC=F",              # COMEX gold futures, USD/oz
    "silver": "SI=F",            # COMEX silver futures, USD/oz
}


def macro_series(name: str) -> pd.DataFrame | None:
    tk = MACRO_TICKERS.get(str(name).lower())
    if not tk:
        return None
    df = prices(tk, period="5y")
    if df is None or df.empty:
        return None
    return df[["Date", "Close"]].rename(columns={"Date": "date", "Close": name})


def macro_list() -> list[str]:
    return sorted(MACRO_TICKERS)
