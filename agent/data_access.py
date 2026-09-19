"""Loaders the tools read through: SEC statements, live prices and quote data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import live

# Annual statements, plus the income statement by quarter.
STATEMENTS = ("income_statement", "balance_sheet", "cash_flow", "quarterly_results")


# ------------------------------------------------------------ financials --
def financial_statement(symbol: str, statement: str) -> pd.DataFrame | None:
    """A statement from the company's SEC XBRL filings, or None if it has none."""
    if statement not in STATEMENTS:
        raise ValueError(f"unknown statement '{statement}'; one of {list(STATEMENTS)}")
    from . import sec
    if not sec.is_filer(symbol):
        return None
    quarterly = statement == "quarterly_results"
    df = sec.statement(symbol, "income_statement" if quarterly else statement,
                       annual=not quarterly)
    if df is None or df.empty:
        return None
    return df


# ---------------------------------------------------------------- profile --
def profile(symbol: str) -> dict | None:
    """Quote-side profile (description, sector, size) plus SEC registration details."""
    from . import sec
    quote = live.profile(symbol) or {}
    filer = sec.profile(symbol) or {}
    if not quote and not filer:
        return None
    out = {"symbol": symbol.upper()}
    out.update({k: v for k, v in quote.items() if v is not None and k not in ("symbol", "source")})
    if filer:
        out.setdefault("company_name", filer.get("company_name"))
        out["sec_registration"] = {k: filer.get(k) for k in
                                   ("cik", "company_name", "sic_description", "exchange",
                                    "state", "fiscal_year_end")}
    out["sources"] = [s for s in (quote.get("source"), filer.get("source")) if s]
    return out


def valuation(symbol: str) -> dict | None:
    return live.valuation(symbol)


# ------------------------------------------------------------------- prices --
def prices(symbol: str) -> pd.DataFrame | None:
    return live.prices(symbol)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def technicals(symbol: str) -> dict | None:
    """Moving averages, RSI, MACD and trailing returns from the daily price history."""
    df = prices(symbol)
    if df is None or df.empty or "Close" not in df:
        return None
    c = df["Close"].astype(float).dropna()
    if len(c) < 20:
        return None
    last_date = df.loc[c.index[-1], "Date"]
    v = df["Volume"].astype(float) if "Volume" in df else pd.Series(dtype=float)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    rsi = _rsi(c)
    last252 = c.tail(252)

    def g(series, i=-1):
        try:
            x = series.iloc[i]
            return round(float(x), 2) if pd.notna(x) else None
        except (IndexError, ValueError):
            return None

    def ret(n):
        if len(c) <= n:
            return None
        a, b = c.iloc[-1], c.iloc[-1 - n]
        return round((a / b - 1) * 100, 2) if b else None

    m = {
        "as_of": str(pd.Timestamp(last_date).date()),
        "price": g(c),
        "sma50": g(sma50), "sma200": g(sma200),
        "rsi14": g(rsi), "macd": g(macd), "macd_signal": g(signal),
        "volume": g(v),
        "all_time_high": round(float(c.max()), 2),
        "all_time_low": round(float(c.min()), 2),
        "return_1d_pct": ret(1), "return_1w_pct": ret(5),
        "return_1m_pct": ret(21), "return_3m_pct": ret(63),
        "return_6m_pct": ret(126), "return_1y_pct": ret(252),
        "return_3y_pct": ret(756), "return_5y_pct": ret(1260),
    }
    if len(last252):
        hi = float(last252.max())
        if hi:
            m["pct_below_52w_high"] = round((float(c.iloc[-1]) / hi - 1) * 100, 2)
    if m.get("sma50") and m.get("price"):
        m["above_sma50"] = m["price"] > m["sma50"]
    if m.get("sma200") and m.get("price"):
        m["above_sma200"] = m["price"] > m["sma200"]
    return {k: val for k, val in m.items() if val is not None}


def price_analytics(symbol: str, benchmark: str = "sp500") -> dict | None:
    """Trader-oriented price stats beyond basic momentum: 52w range & position,
    max drawdown, annualized volatility, volume trend, moving-average crossover
    (golden/death cross) and relative strength vs a benchmark index."""
    df = prices(symbol)
    if df is None or df.empty or "Close" not in df:
        return None
    c = df["Close"].astype(float).dropna()          # nominal price levels
    if len(c) < 30:
        return None
    # Adjusted series (splits/dividends) for all PERFORMANCE metrics - raw Close
    # produces phantom -50% "drawdowns" on split dates. 52w range, current price
    # and moving averages stay on nominal Close (what the user sees quoted).
    ca = df["Adj Close"].astype(float).dropna() if "Adj Close" in df else c
    dates = df.loc[c.index, "Date"]
    cur = float(c.iloc[-1])
    last_date = pd.Timestamp(dates.iloc[-1])
    out: dict = {"symbol": symbol.upper(), "as_of": str(last_date.date()),
                 "current_price": round(cur, 2),
                 "performance_basis": "Adj Close (split/dividend-adjusted)"}

    last252 = c.tail(252)
    hi52, lo52 = float(last252.max()), float(last252.min())
    out["high_52w"], out["low_52w"] = round(hi52, 2), round(lo52, 2)
    out["pct_below_52w_high"] = round((cur / hi52 - 1) * 100, 1) if hi52 else None
    out["pct_above_52w_low"] = round((cur / lo52 - 1) * 100, 1) if lo52 else None

    # all-time high & drawdown on ADJUSTED series (correct peak-to-trough)
    athc = float(ca.max())
    out["pct_below_all_time_high"] = round((float(ca.iloc[-1]) / athc - 1) * 100, 1) if athc else None
    roll_max = ca.cummax()
    dd = (ca / roll_max - 1) * 100
    out["max_drawdown_pct"] = round(float(dd.min()), 1)
    out["current_drawdown_pct"] = round(float(dd.iloc[-1]), 1)

    # annualized volatility from last-1y adjusted daily returns
    rets = ca.pct_change().dropna()
    if len(rets) >= 30:
        out["annualized_volatility_pct"] = round(float(rets.tail(252).std() * (252 ** 0.5) * 100), 1)

    # volume trend: recent 20d avg vs 200d avg
    if "Volume" in df:
        v = df["Volume"].astype(float).dropna()
        if len(v) >= 200:
            v20, v200 = float(v.tail(20).mean()), float(v.tail(200).mean())
            out["avg_volume_20d"] = int(v20)
            out["volume_vs_200d_avg_pct"] = round((v20 / v200 - 1) * 100, 1) if v200 else None

    # moving-average crossover state
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    if pd.notna(sma50.iloc[-1]) and pd.notna(sma200.iloc[-1]):
        s50, s200 = float(sma50.iloc[-1]), float(sma200.iloc[-1])
        out["sma50"], out["sma200"] = round(s50, 2), round(s200, 2)
        out["ma_structure"] = ("golden cross (50-day above 200-day average - bullish structure)"
                               if s50 > s200 else
                               "death cross (50-day below 200-day average - bearish structure)")

    # relative strength vs benchmark over 3m / 1y (adjusted series)
    def ret_over(series, days):
        if len(series) <= days:
            return None
        b = series.iloc[-1 - days]
        return (series.iloc[-1] / b - 1) * 100 if b else None
    bench = index_prices(benchmark)
    rs = {}
    for label, days in [("3m", 63), ("1y", 252)]:
        sr = ret_over(ca, days)
        if sr is None:
            continue
        br = None
        if bench is not None and "Close" in bench:
            bc = bench["Close"].astype(float).dropna()
            br = ret_over(bc, days)
        if br is not None:
            rs[label] = {"stock_return_pct": round(float(sr), 1),
                         f"{benchmark}_return_pct": round(float(br), 1),
                         "relative_strength_pct": round(float(sr - br), 1)}
        else:
            rs[label] = {"stock_return_pct": round(float(sr), 1)}
    if rs:
        out["relative_strength"] = rs
    return out


# -------------------------------------------------------------- reference ---
def index_list() -> list[str]:
    return live.index_list()


def index_prices(name: str) -> pd.DataFrame | None:
    return live.index_prices(name)


def macro_list() -> list[str]:
    return live.macro_list()


def macro_series(name: str) -> pd.DataFrame | None:
    return live.macro_series(name)


# ---------------------------------------------------------------- helpers ---
def df_to_md(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "(no data)"
    shown = df.head(max_rows)
    try:
        txt = shown.to_markdown(index=False)
    except ImportError:
        txt = shown.to_string(index=False)
    if len(df) > max_rows:
        txt += f"\n... ({len(df) - max_rows} more rows)"
    return txt
