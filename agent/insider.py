"""US insider dealing and executive pay, from SEC primary sources.

Insider transactions come from the SEC's own quarterly Forms 3/4/5 data sets — a
~14 MB zip per quarter covering every filer, so this needs no per-company scraping
and cannot be rate limited. Officers and directors must report trades within two
business days, which makes this close to real time.

Executive compensation lives in the DEF 14A proxy statement's Summary Compensation
Table. That is narrative HTML, not XBRL, so the proxy is located and returned as
text for reading rather than parsed into a table — a compensation table's shape
varies enough between filers that a scraper would quietly mis-attribute figures.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from .config import DATA
from .sec import UA, cik, filings

CACHE = DATA / "sec" / "insider"
CACHE.mkdir(parents=True, exist_ok=True)

FORM345_URL = ("https://www.sec.gov/files/structureddata/data/"
               "insider-transactions-data-sets/{yr}q{q}_form345.zip")

# Codes that matter for reading intent. Open-market buys and sells are the signal;
# option exercises and grants are compensation mechanics and are labelled as such
# so they are never counted as conviction buying.
TXN_CODES = {
    "P": "open-market purchase", "S": "open-market sale",
    "A": "grant/award", "M": "option exercise", "F": "tax withholding",
    "G": "gift", "C": "conversion", "D": "disposition to issuer",
}


def quarter_zip(year: int, quarter: int) -> zipfile.ZipFile | None:
    """Download (and cache) one quarterly Forms 3/4/5 data set."""
    p = CACHE / f"{year}q{quarter}_form345.zip"
    if not p.exists():
        try:
            r = requests.get(FORM345_URL.format(yr=year, q=quarter), headers=UA, timeout=180)
            r.raise_for_status()
            p.write_bytes(r.content)
        except Exception:
            return None
    try:
        return zipfile.ZipFile(p)
    except zipfile.BadZipFile:
        p.unlink(missing_ok=True)
        return None


def _iso(v) -> str:
    """SEC ships transaction dates as DD-MMM-YYYY. Normalise to ISO.

    Slicing the raw string to 10 characters loses the last digit of the year
    ('21-JAN-2026' -> '21-JAN-202') and, worse, makes a lexical sort meaningless
    because the day leads.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    ts = pd.to_datetime(str(v), errors="coerce", dayfirst=True)
    return "" if pd.isna(ts) else ts.date().isoformat()


def _read(z: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    match = next((n for n in z.namelist() if n.lower().endswith(name)), None)
    if not match:
        return None
    with z.open(match) as f:
        return pd.read_csv(io.TextIOWrapper(f, "utf-8", errors="replace"),
                           sep="\t", low_memory=False)


def transactions(symbol: str, year: int, quarter: int, limit: int = 40) -> dict | None:
    """Insider transactions for one company in one quarter.

    Joins SUBMISSION (who filed), REPORTINGOWNER (the insider and their role) and
    NONDERIV_TRANS (the trade itself) on the accession number.
    """
    c = cik(symbol)
    if c is None:
        return None
    z = quarter_zip(year, quarter)
    if z is None:
        return None
    sub = _read(z, "submission.tsv")
    own = _read(z, "reportingowner.tsv")
    trn = _read(z, "nonderiv_trans.tsv")
    if sub is None or trn is None:
        return None

    sub.columns = [c_.upper() for c_ in sub.columns]
    trn.columns = [c_.upper() for c_ in trn.columns]
    mine = sub[pd.to_numeric(sub.get("ISSUERCIK"), errors="coerce") == c]
    if mine.empty:
        return {"symbol": symbol.upper(), "period": f"{year}Q{quarter}",
                "transactions": [], "note": "no Forms 3/4/5 filed this quarter"}

    accs = set(mine["ACCESSION_NUMBER"])
    t = trn[trn["ACCESSION_NUMBER"].isin(accs)].copy()
    if own is not None:
        own.columns = [c_.upper() for c_ in own.columns]
        names = (own[own["ACCESSION_NUMBER"].isin(accs)]
                 .set_index("ACCESSION_NUMBER")[["RPTOWNERNAME", "RPTOWNER_TITLE"]]
                 .to_dict("index"))
    else:
        names = {}

    rows = []
    for _, r in t.iterrows():
        acc = r["ACCESSION_NUMBER"]
        meta = names.get(acc, {})
        code = str(r.get("TRANS_CODE", ""))
        shares = pd.to_numeric(r.get("TRANS_SHARES"), errors="coerce")
        price = pd.to_numeric(r.get("TRANS_PRICEPERSHARE"), errors="coerce")
        rows.append({
            "date": _iso(r.get("TRANS_DATE")),
            "insider": meta.get("RPTOWNERNAME"),
            "title": meta.get("RPTOWNER_TITLE"),
            "code": code,
            "action": TXN_CODES.get(code, code),
            "shares": None if pd.isna(shares) else float(shares),
            "price": None if pd.isna(price) else round(float(price), 2),
            "value_usd": (None if (pd.isna(shares) or pd.isna(price))
                          else round(float(shares) * float(price))),
            "shares_owned_after": pd.to_numeric(r.get("SHRS_OWND_FOLWNG_TRANS"),
                                                errors="coerce"),
            "direct_or_indirect": r.get("DIRECT_INDIRECT_OWNERSHIP"),
        })
    rows.sort(key=lambda x: x["date"], reverse=True)

    buys = [r for r in rows if r["code"] == "P" and r["value_usd"]]
    sells = [r for r in rows if r["code"] == "S" and r["value_usd"]]
    return {
        "symbol": symbol.upper(), "period": f"{year}Q{quarter}",
        "summary": {
            "open_market_buys": len(buys),
            "open_market_sells": len(sells),
            "buy_value_usd": sum(r["value_usd"] for r in buys) or 0,
            "sell_value_usd": sum(r["value_usd"] for r in sells) or 0,
            "note": "Only codes P and S are open-market decisions. A/M/F are "
                    "grants, option exercises and tax withholding — compensation "
                    "mechanics, not conviction trades — and are excluded from these "
                    "totals but listed below.",
        },
        "transactions": rows[:limit],
        "source": f"SEC Forms 3/4/5 data set {year}Q{quarter}",
    }


def proxy_statements(symbol: str, limit: int = 5) -> list[dict]:
    """DEF 14A proxies — where the Summary Compensation Table lives."""
    return filings(symbol, forms=("DEF 14A", "DEFA14A"), limit=limit)


def compensation_context(symbol: str) -> dict | None:
    """Pointers to executive-pay disclosure for a company.

    Returns the proxy filings rather than a parsed pay table: the Summary
    Compensation Table's layout differs enough between filers that automated
    extraction misattributes figures between named officers, which is worse than
    reading the table.
    """
    c = cik(symbol)
    if c is None:
        return None
    proxies = proxy_statements(symbol)
    out = {
        "symbol": symbol.upper(), "cik": c,
        "proxy_filings": proxies,
        "what_to_read": [
            "Summary Compensation Table — salary, bonus, stock and option awards, "
            "total pay per named executive officer, three years",
            "Pay Ratio disclosure — CEO pay vs median employee",
            "Pay Versus Performance table — compensation actually paid vs TSR",
            "Beneficial Ownership table — insider and >5% holder stakes",
        ],
        "note": "Fetch a proxy URL with read_document. Figures are not parsed out "
                "automatically because compensation tables vary in shape between "
                "filers and mis-attributing pay between officers is worse than "
                "reading the table directly.",
    }
    if not proxies:
        # A foreign private issuer has a CIK but files 20-F and never a DEF 14A,
        # so an empty list here is a filing-form fact, not a missing-data one.
        out["why_empty"] = (
            f"{symbol.upper()} has an SEC CIK but no DEF 14A proxy statements. "
            "Foreign private issuers file Form 20-F instead, which carries "
            "compensation in its own section — list it with "
            "company_documents(kind='annual').")
    return out
