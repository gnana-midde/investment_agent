"""US fundamentals and filings from SEC EDGAR — the authoritative source.

Why this rather than yfinance for fundamentals: EDGAR is what companies actually
file, it reaches ~17 years deep (yfinance gives 4-5), it needs no key, and the
documented limit is 10 requests/second with a full bulk archive available — so it
has none of the throttling that makes bulk yfinance use unworkable.

Filings are HTML, not PDF, so a 10-K extracts to text with no OCR and no PDF
parsing.
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache

import pandas as pd
import requests

from .config import DATA

CACHE = DATA / "live_cache" / "sec"
CACHE.mkdir(parents=True, exist_ok=True)

# SEC blocks requests without a descriptive User-Agent carrying contact details.
# www.sec.gov/files/company_tickers.json returns 403 for a bare product token and
# 200 once a contact address is appended. Without a configured contact every SEC
# tool silently degrades to "not an SEC filer", because an empty ticker map is
# indistinguishable from an unlisted symbol.
SEC_UA_HINT = ("the SEC rejects requests whose User-Agent carries no contact "
               "address — set the SEC_USER_AGENT environment variable to "
               "'YourProject/1.0 (you@example.com)' and restart the server")
_UA_FALLBACK = "InvestmentAgent/1.0"
UA = {"User-Agent": os.getenv("SEC_USER_AGENT", "").strip() or _UA_FALLBACK,
      "Accept-Encoding": "gzip, deflate"}


def ua_configured() -> bool:
    """Does the User-Agent carry a contact address? sec.gov rejects it otherwise.
    Checked on the string itself, not on whether the env var was set, so a
    deployment whose fallback already carries a contact is not told to set one."""
    return "@" in UA["User-Agent"]

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

TTL_TICKERS = 30 * 24 * 3600
TTL_FACTS = 24 * 3600            # only changes when a new filing lands
TTL_SUBMISSIONS = 12 * 3600

# XBRL tag varies by filer and by year, so each line is a list of candidates tried in
# order. Filers that report none of them simply omit the row — better an absent line
# than a wrong one silently mapped from a similar-sounding concept.
INCOME_STATEMENT = [
    ("Revenue", ["RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax",
                 "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"]),
    ("Cost of revenue", ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfSales"]),
    ("Gross profit", ["GrossProfit"]),
    ("R&D", ["ResearchAndDevelopmentExpense"]),
    ("SG&A", ["SellingGeneralAndAdministrativeExpense",
              "GeneralAndAdministrativeExpense"]),
    ("Operating income", ["OperatingIncomeLoss"]),
    ("Pretax income", ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]),
    ("Tax", ["IncomeTaxExpenseBenefit"]),
    ("Net income", ["NetIncomeLoss", "ProfitLoss"]),
    ("EPS (diluted)", ["EarningsPerShareDiluted"]),
]

BALANCE_SHEET = [
    ("Total assets", ["Assets"]),
    ("Current assets", ["AssetsCurrent"]),
    ("Cash & equivalents", ["CashAndCashEquivalentsAtCarryingValue"]),
    ("Inventory", ["InventoryNet"]),
    ("Total liabilities", ["Liabilities"]),
    ("Current liabilities", ["LiabilitiesCurrent"]),
    ("Long-term debt", ["LongTermDebtNoncurrent", "LongTermDebt"]),
    ("Equity", ["StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
]

CASH_FLOW = [
    ("Operating cash flow", ["NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]),
    ("Investing cash flow", ["NetCashProvidedByUsedInInvestingActivities"]),
    ("Financing cash flow", ["NetCashProvidedByUsedInFinancingActivities"]),
    ("Capex", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    ("Dividends paid", ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"]),
    ("Buybacks", ["PaymentsForRepurchaseOfCommonStock"]),
]

STATEMENT_MAP = {"income_statement": INCOME_STATEMENT, "balance_sheet": BALANCE_SHEET,
                 "cash_flow": CASH_FLOW}


def _get(url: str, cache_name: str, ttl: int):
    p = CACHE / cache_name
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        r = requests.get(url, headers=UA, timeout=90)
        r.raise_for_status()
        data = r.json()
    except Exception:
        if p.exists():                       # stale beats nothing
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None
    p.write_text(json.dumps(data), encoding="utf-8")
    return data


@lru_cache(maxsize=1)
def ticker_map() -> dict[str, int]:
    """TICKER -> CIK for every SEC filer (~10,400). One request."""
    d = _get(TICKERS_URL, "company_tickers.json", TTL_TICKERS)
    if not d:
        return {}
    return {v["ticker"].upper(): int(v["cik_str"]) for v in d.values()}


def _norm_symbol(symbol: str) -> str:
    """The SEC writes share classes with a hyphen (BRK-B); quotes often use a dot."""
    return symbol.strip().upper().replace(".", "-")


def is_filer(symbol: str) -> bool:
    return _norm_symbol(symbol) in ticker_map()


def cik(symbol: str) -> int | None:
    return ticker_map().get(_norm_symbol(symbol))


def search(query: str, limit: int = 8) -> list[dict]:
    """Filers whose ticker or registered name matches `query`.

    An exact ticker comes first, then names that start with the query, then names
    that contain it anywhere. Within each group the SEC's own listing order is kept.
    """
    d = _get(TICKERS_URL, "company_tickers.json", TTL_TICKERS)
    q = query.strip()
    if not d or not q:
        return []
    qs, ql = _norm_symbol(q), q.lower()
    exact, prefix, contains = [], [], []
    for v in d.values():
        tick = str(v.get("ticker", "")).upper()
        name = str(v.get("title", ""))
        rec = {"symbol": tick, "company_name": name, "cik": int(v["cik_str"])}
        low = name.lower()
        if tick == qs:
            exact.append(rec)
        elif low.startswith(ql):
            prefix.append(rec)
        elif ql in low:
            contains.append(rec)
    return (exact + prefix + contains)[:limit]


def unavailable_reason() -> str | None:
    """Why an SEC lookup could not be answered, or None if the ticker map loaded.

    Callers use this to separate "this symbol is not an SEC filer" from "we could
    not reach the SEC at all" — reporting the second as the first is how a real
    filer ends up described as a non-filer."""
    if ticker_map():
        return None
    if not ua_configured():
        return f"the SEC filer list could not be loaded: {SEC_UA_HINT}"
    return ("the SEC filer list could not be loaded (sec.gov unreachable or "
            "rate-limited); try again shortly")


def company_facts(symbol: str) -> dict | None:
    c = cik(symbol)
    if c is None:
        return None
    return _get(FACTS_URL.format(cik=c), f"facts_{c:010d}.json", TTL_FACTS)


def _series(facts: dict, tags: list[str], annual: bool):
    """Values per fiscal period for a line item, plus the unit they're reported in.

    Candidate tags are MERGED rather than first-match-wins: filers switch XBRL tags
    between years (Apple reported revenue under SalesRevenueNet before moving to
    RevenueFromContractWithCustomer...), so taking only the first tag that has any
    data leaves holes in the early years. Earlier tags in the list win where both
    cover a period, so the preferred concept still takes precedence.
    """
    import datetime as _dt

    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: dict[str, tuple] = {}
    unit_seen = "USD"

    def span_days(p):
        """Length of the period a fact covers, or None for instant (balance-sheet) facts."""
        s, e = p.get("start"), p.get("end")
        if not s or not e:
            return None
        try:
            return (_dt.date.fromisoformat(e) - _dt.date.fromisoformat(s)).days
        except ValueError:
            return None

    for tag in reversed(tags):            # reversed so earlier tags overwrite later
        node = gaap.get(tag)
        if not node:
            continue
        for unit in ("USD", "USD/shares"):
            pts = node.get("units", {}).get(unit)
            if not pts:
                continue
            unit_seen = unit
            for p in pts:
                end, val = p.get("end"), p.get("val")
                if not end or val is None:
                    continue
                d = span_days(p)
                # The period a number covers is defined by start/end, NOT by fy/fp --
                # those describe the FILING's fiscal year, and a 10-K restates two or
                # three prior years, so keying on fy silently files comparatives under
                # the wrong year. Select on duration instead: ~1 year for annual,
                # ~1 quarter for quarterly, and duration-less (instant) facts are
                # balance-sheet items, which are dated by `end` alone.
                if d is None:
                    if not annual:
                        continue
                elif annual and not (330 <= d <= 400):
                    continue
                elif not annual and not (60 <= d <= 120):
                    continue
                try:
                    ed = _dt.date.fromisoformat(end)
                except ValueError:
                    continue
                # Label a fiscal year by the CALENDAR YEAR IT ENDS IN. That is the
                # convention almost every US filer uses for its own FY (Apple's FY2025
                # ends Sep-2025, NVIDIA's ends Jan-2025, Microsoft's ends Jun-2025), so
                # it lines up with how the numbers are quoted in the filings and by the
                # companies themselves. Shifting Jan-Jun year-ends back a year would
                # rename NVIDIA's FY2025 to FY2024.
                # A minority of retailers label the other way; period_end_dates on the
                # frame records the exact end date so the mapping is never ambiguous.
                label = f"FY{ed.year}" if annual else f"{end}"
                prev = merged.get(label)
                # prefer the most recently FILED figure (restatements supersede)
                filed = str(p.get("filed", ""))
                if prev is None or filed >= prev[0]:
                    merged[label] = (filed, val, end)
    if not merged:
        return None
    ends = {k: v[2] for k, v in merged.items()}
    return {k: v[1] for k, v in merged.items()}, unit_seen, ends


def statement(symbol: str, which: str = "income_statement",
              annual: bool = True, periods: int = 12) -> pd.DataFrame | None:
    """One statement as an (item, <period columns>) frame, newest period last."""
    import datetime as _dt

    spec = STATEMENT_MAP.get(which)
    if spec is None:
        return None
    facts = company_facts(symbol)
    if not facts:
        return None
    rows, cols, period_ends = [], set(), {}
    for label, tags in spec:
        got = _series(facts, tags, annual)
        if not got:
            continue
        s, unit, ends = got
        rows.append((label, s, unit))
        cols |= set(s)
        period_ends.update(ends)
    if not rows:
        return None

    def key(c: str):
        """Chronological sort for both column styles.

        Annual columns are 'FY2024'; quarterly columns are the ISO period-end date.
        A key that only understood 'FY####' left quarterly columns in hash order, so
        the newest-N slice below returned an arbitrary set of quarters.
        """
        s = str(c)
        if s.startswith("FY"):
            digits = s[2:]
            return (int(digits), 12, 31) if digits.isdigit() else (0, 0, 0)
        try:
            d = _dt.date.fromisoformat(s)
            return (d.year, d.month, d.day)
        except ValueError:
            return (0, 0, 0)

    ordered = sorted(cols, key=key)[-periods:]
    data = []
    for label, s, unit in rows:
        # Per-share figures must NOT be scaled to millions — dividing EPS by 1e6
        # rounds every year to 0.00 and reads as "the company earns nothing".
        per_share = unit == "USD/shares"
        rec = {"item": f"{label} (USD/sh)" if per_share else label}
        for c in ordered:
            v = s.get(c)
            if not isinstance(v, (int, float)):
                rec[c] = ""
            elif per_share:
                rec[c] = round(v, 2)
            else:
                rec[c] = round(v / 1e6, 1)     # raw USD makes a 12-column table unreadable
        data.append(rec)
    df = pd.DataFrame(data, columns=["item", *ordered])
    df.attrs["units"] = "USD millions unless the row says USD/sh"
    # exact period end per column, so 'FY2025' is never ambiguous across filers that
    # label their fiscal years differently
    df.attrs["period_end_dates"] = {c: period_ends.get(c) for c in ordered}
    return df


def filings(symbol: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
            limit: int = 20) -> list[dict]:
    """Recent filings with direct document URLs. These are HTML — no PDF parsing."""
    c = cik(symbol)
    if c is None:
        return []
    sub = _get(SUBMISSIONS_URL.format(cik=c), f"sub_{c:010d}.json", TTL_SUBMISSIONS)
    if not sub:
        return []
    r = sub.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(r.get("form", [])):
        if form not in forms:
            continue
        acc = r["accessionNumber"][i].replace("-", "")
        doc = r["primaryDocument"][i]
        out.append({
            "form": form,
            "filed": r["filingDate"][i],
            "period": r.get("reportDate", [None] * (i + 1))[i],
            "description": r.get("primaryDocDescription", [""] * (i + 1))[i],
            "url": f"https://www.sec.gov/Archives/edgar/data/{c}/{acc}/{doc}",
        })
        if len(out) >= limit:
            break
    return out


def filing_text(url: str, max_chars: int = 400_000) -> str | None:
    """Extract readable text from an EDGAR HTML filing."""
    from bs4 import BeautifulSoup
    try:
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    txt = soup.get_text("\n", strip=True)
    if len(txt) > max_chars:
        txt = txt[:max_chars] + f"\n\n[truncated at {max_chars:,} chars]"
    return txt


def profile(symbol: str) -> dict | None:
    c = cik(symbol)
    if c is None:
        return None
    sub = _get(SUBMISSIONS_URL.format(cik=c), f"sub_{c:010d}.json", TTL_SUBMISSIONS)
    if not sub:
        return None
    return {"symbol": symbol.upper(), "cik": c,
            "company_name": sub.get("name"),
            "sic_description": sub.get("sicDescription"),
            "exchange": (sub.get("exchanges") or [None])[0],
            "state": sub.get("stateOfIncorporation"),
            "fiscal_year_end": sub.get("fiscalYearEnd"),
            "source": "SEC EDGAR (authoritative filings)"}
