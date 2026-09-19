"""MCP tool definitions: US equity research over SEC filings and live market data."""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

import pandas as pd

from claude_agent_sdk import tool

from . import data_access as da


def _text(s: str) -> dict:
    return {"content": [{"type": "text", "text": s}]}


def _err(s: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {s}"}], "is_error": True}


def _sec_unavailable_reason() -> str | None:
    """Why an SEC lookup came back empty when the SEC itself could not be reached,
    so that is never reported as "not an SEC filer"."""
    try:
        from . import sec
        return sec.unavailable_reason()
    except Exception:
        return None


def _js(obj, limit: int = 12000) -> str:
    s = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "\n... (truncated)"


# ------------------------------------------------------------------ tools ---
@tool(
    "resolve_company",
    "Resolve a company name or partial name to its ticker and SEC CIK. Always use this "
    "first when the user mentions a company by name rather than ticker.",
    {"type": "object",
     "properties": {"query": {"type": "string", "description": "company name or ticker"}},
     "required": ["query"]},
)
async def resolve_company(args):
    from . import sec
    hits = await asyncio.to_thread(sec.search, args["query"])
    if not hits:
        blocked = await asyncio.to_thread(_sec_unavailable_reason)
        if blocked:
            return _err(blocked)
        return _text(f"No company matched '{args['query']}'.")
    return _text(_js(hits))


@tool(
    "company_overview",
    "Company profile card: business description, sector and industry, market cap (USD "
    "millions), exchange, website and headcount from the live quote, plus the SEC "
    "registration details (CIK, SIC industry, state of incorporation, fiscal year end).",
    {"type": "object",
     "properties": {"symbol": {"type": "string"}},
     "required": ["symbol"]},
)
async def company_overview(args):
    p = await asyncio.to_thread(da.profile, args["symbol"])
    if not p:
        return _err(f"no profile for {args['symbol']} (check the ticker with resolve_company)")
    return _text(_js(p))


@tool(
    "financial_statements",
    "Financial statements straight from the company's SEC XBRL filings, in USD millions "
    "(per-share rows are marked USD/sh). statement: income_statement | balance_sheet | "
    "cash_flow (annual, up to 12 fiscal years) or quarterly_results (the income statement "
    "by quarter, up to 12 quarters). Fiscal years are labelled by the calendar year they "
    "end in; the exact period-end dates are listed under the table.",
    {"type": "object",
     "properties": {
         "symbol": {"type": "string"},
         "statement": {"type": "string", "enum": list(da.STATEMENTS)}},
     "required": ["symbol", "statement"]},
)
async def financial_statements(args):
    statement = args["statement"]
    if statement not in da.STATEMENTS:
        return _err(f"unknown statement '{statement}'; one of {list(da.STATEMENTS)}")
    df = await asyncio.to_thread(da.financial_statement, args["symbol"], statement)
    if df is None:
        blocked = await asyncio.to_thread(_sec_unavailable_reason)
        if blocked:
            return _err(blocked)
        return _err(f"no {statement} data for {args['symbol']} — check the ticker is an "
                    f"SEC filer with resolve_company")
    notes = [f"Units: {df.attrs.get('units', 'USD millions')}."]
    ends = {k: v for k, v in (df.attrs.get("period_end_dates") or {}).items() if v}
    if ends:
        notes.append("Period ends: " + ", ".join(f"{k} = {v}" for k, v in ends.items()) + ".")
    return _text(da.df_to_md(df) + "\n\n" + " ".join(notes))


@tool(
    "valuation_summary",
    "Current valuation snapshot from the live quote: price, market cap and enterprise "
    "value (USD millions), trailing/forward P/E, P/B, P/S, EV/EBITDA, EV/revenue, PEG, "
    "dividend yield, ROE/ROA, margins, debt/equity, beta, the 52-week range and the "
    "analyst consensus (mean target, recommendation, analyst count — third-party figures, "
    "reported as data, not as a price target of our own).",
    {"type": "object",
     "properties": {"symbol": {"type": "string"}},
     "required": ["symbol"]},
)
async def valuation_summary(args):
    v = await asyncio.to_thread(da.valuation, args["symbol"])
    if not v:
        return _err(f"no valuation data for {args['symbol']}")
    return _text(_js(v))


@tool(
    "technicals_momentum",
    "Technical indicators & momentum computed from the daily price history: price vs "
    "50/200-day simple moving averages, RSI(14), MACD, returns over "
    "1d/1w/1m/3m/6m/1y/3y/5y, and distance from the 52-week high.",
    {"type": "object",
     "properties": {"symbol": {"type": "string"}},
     "required": ["symbol"]},
)
async def technicals_momentum(args):
    sym = args["symbol"]
    t = await asyncio.to_thread(da.technicals, sym)
    if not t:
        return _err(f"no price history for {sym}")
    return _text(_js({"symbol": sym.upper(), **t}))


# daily needs no resampling; the other two map to a pandas offset alias.
_PRICE_INTERVALS = {"daily": None, "weekly": "W", "monthly": "ME"}


@tool(
    "price_history",
    "OHLCV price history between dates (YYYY-MM-DD). interval: daily|weekly|monthly. "
    "Use for 'what happened to the stock around <date>' style questions.",
    {"type": "object",
     "properties": {"symbol": {"type": "string"},
                    "start": {"type": "string"}, "end": {"type": "string"},
                    "interval": {"type": "string", "enum": sorted(_PRICE_INTERVALS),
                                 "default": "monthly"}},
     "required": ["symbol"]},
)
async def price_history(args):
    interval = args.get("interval") or "monthly"
    # Validated here rather than letting the resample lookup raise: a bare
    # KeyError tells the model nothing about which value was wrong.
    if interval not in _PRICE_INTERVALS:
        return _err(f"unknown interval '{interval}'; one of "
                    f"{sorted(_PRICE_INTERVALS)}")

    def work():
        df = da.prices(args["symbol"])
        if df is None:
            return None
        if args.get("start"):
            df = df[df["Date"] >= pd.Timestamp(args["start"])]
        if args.get("end"):
            df = df[df["Date"] <= pd.Timestamp(args["end"])]
        if df.empty:
            return df
        iv = interval
        if iv != "daily":
            rule = _PRICE_INTERVALS[iv]
            df = (df.set_index("Date")
                    .resample(rule)
                    .agg({"Open": "first", "High": "max", "Low": "min",
                          "Close": "last", "Volume": "sum"})
                    .dropna(subset=["Close"]).reset_index())
        df["Date"] = df["Date"].dt.date
        return df.round(2)
    df = await asyncio.to_thread(work)
    if df is None:
        return _err(f"no price data for {args['symbol']}")
    return _text(da.df_to_md(df, max_rows=80))


@tool(
    "screen_stocks",
    "Objective stock screen over US SEC filers, using the table refresh_screening_data "
    "builds from the SEC's bulk Company Facts archive (each filer's latest annual "
    "figures). Numeric minimums/maximums on:\n"
    "SIZE (USD millions): public_float_usd_m, revenue_usd_m, net_income_usd_m, "
    "assets_usd_m, fcf_usd_m.\n"
    "VALUATION: pe_on_float (public float / net income).\n"
    "PROFITABILITY: roe_pct, roa_pct, gross_margin_pct, operating_margin_pct, "
    "net_margin_pct, fcf_margin_pct.\n"
    "LEVERAGE/LIQUIDITY: debt_equity (long-term debt / equity), current_ratio.\n"
    "The table is built from filings and has NO price feed: size is PUBLIC FLOAT (the "
    "market value of non-affiliate shares reported on the 10-K cover), not market cap, "
    "and pe_on_float therefore understates the true P/E of insider-controlled companies. "
    "Returns a table + match count + a FUNNEL (companies remaining after each criterion, "
    "in order — so it's visible what each criterion actually cost) + NEAR MISSES "
    "(companies that failed exactly one criterion by <= near_miss_tolerance_pct of its "
    "threshold — i.e. the good companies you left out by being strict; set to 0 to "
    "disable). A company MISSING data for a filtered column is excluded rather than "
    "passed, and the funnel says how many were dropped that way. exclude_categories "
    "removes companies BEFORE ranking whose names match a category (e.g. ['tobacco']) — "
    "only apply a category if the user has explicitly asked for it; the table carries "
    "no industry codes, so this only catches companies whose names say what they do. "
    "Screening only — never advice.",
    {"type": "object",
     "properties": {
         "min": {"type": "object", "description": "column -> minimum value",
                 "additionalProperties": {"type": "number"}},
         "max": {"type": "object", "description": "column -> maximum value",
                 "additionalProperties": {"type": "number"}},
         "exclude_categories": {"type": "array", "items": {"type": "string"},
                                "description": "e.g. ['tobacco'] — only categories the "
                                "user explicitly confirmed"},
         "sort_by": {"type": "string", "default": "public_float_usd_m"},
         "ascending": {"type": "boolean", "default": False},
         "limit": {"type": "integer", "default": 25},
         "near_miss_tolerance_pct": {"type": "number", "default": 15.0,
                                    "description": "0 disables near-miss reporting"}},
     "required": []},
)
async def screen_stocks(args):
    from .screener import DEFAULT_SORT, screen

    def work():
        return screen(
            min_filters=args.get("min"), max_filters=args.get("max"),
            exclude_categories=args.get("exclude_categories"),
            sort_by=args.get("sort_by") or DEFAULT_SORT,
            ascending=bool(args.get("ascending", False)),
            limit=int(args.get("limit", 25)),
            near_miss_tolerance_pct=float(args.get("near_miss_tolerance_pct", 15.0)),
        )
    try:
        df, matched, excl = await asyncio.to_thread(work)
    except (FileNotFoundError, ValueError) as e:
        return _err(str(e))
    parts = [f"{matched} companies matched; showing {len(df)} "
             f"(sorted by {excl['sorted_by']})."]

    # Lead with this when a filter had little data behind it. Results contain only
    # verified passes, but sparse source coverage still limits representativeness.
    cov = excl.get("coverage") or {}
    weak = excl.get("unusable_filters") or []
    if weak:
        detail = "; ".join(f"{c}: only {cov[c]['with_data']}/{cov[c]['of_candidates']} "
                           f"candidate rows ({cov[c]['pct']}%) have this value"
                           for c in weak)
        parts.insert(0, (
            f"DATA COVERAGE WARNING — results below are verified passes, but {detail}. "
            f"Rows missing a filtered value were excluded. Treat this as a partial-coverage "
            f"screen rather than a complete ranking of the universe."))
    elif cov:
        parts.append("Coverage of filtered columns: "
                     + "; ".join(f"{c} {v['pct']}%" for c, v in cov.items()) + ".")
    if excl["applied"]:
        by_cat = ", ".join(f"{c}: {n}" for c, n in excl["by_category"].items()) or "none matched"
        parts.append(f"Exclusions applied ({', '.join(excl['applied'])}): "
                     f"{excl['excluded_count']} companies removed before ranking ({by_cat}).")
    if excl["unknown"]:
        parts.append(f"NOTE: unknown exclusion categories ignored: {excl['unknown']} "
                     f"— not yet defined in agent/ethics.py.")
    if excl["funnel"]:
        def _step(f):
            s = f"{f['stage']}: {f['remaining']}"
            miss = f.get("dropped_missing_data")
            if miss:
                s += f" ({miss} rows with missing data excluded)"
            return s
        parts.append("Funnel (companies you left at each step): " +
                     " -> ".join(_step(f) for f in excl["funnel"]))
    if excl["near_misses"]:
        nm_lines = "; ".join(f"{n['symbol']} (missed {n['missed_filter']} by {n['missed_by_pct']}%)"
                             for n in excl["near_misses"][:8])
        parts.append(f"Near misses — failed exactly one criterion by a small margin: {nm_lines}")
    head = "\n".join(parts) + "\n\n"
    return _text(head + da.df_to_md(df, max_rows=int(args.get("limit", 25))))


@tool(
    "price_analytics",
    "Trader-oriented price statistics beyond basic momentum: 52-week range & position, "
    "distance from all-time high, max drawdown & current drawdown, annualised volatility, "
    "volume trend vs 200-day average, 50/200-day moving-average crossover (golden/death "
    "cross), and relative strength vs a benchmark index over 3m/1y. Performance metrics "
    "use split- and dividend-adjusted prices.",
    {"type": "object",
     "properties": {"symbol": {"type": "string"},
                    "benchmark": {"type": "string", "default": "sp500",
                                  "description": "index for relative strength: sp500, "
                                                 "nasdaq or dowjones"}},
     "required": ["symbol"]},
)
async def price_analytics(args):
    r = await asyncio.to_thread(da.price_analytics, args["symbol"], args.get("benchmark", "sp500"))
    if r is None:
        return _err(f"no price history for {args['symbol']}")
    return _text(_js(r))


@tool(
    "portfolio_risk",
    "Portfolio-level risk from ACTUAL historical daily returns — not single-stock "
    "volatility/beta treated as a portfolio proxy. Takes named holdings + weights (any "
    "positive numbers; renormalized to sum to 1, flagged if so), pulls each symbol's real "
    "daily price history, and computes: the correlation matrix between holdings, portfolio "
    "annualized volatility (Markowitz w'*cov*w, cross-checked two ways), and — via the "
    "`empyrical` risk-stats library (empyrical-reloaded, the maintained fork of Quantopian's "
    "widely-used open-source package) applied to the actual weighted portfolio-return series "
    "— CAGR/max drawdown/Calmar ratio, Sharpe and Sortino ratios (risk-free rate from the "
    "10-year Treasury yield unless supplied), HISTORICAL (empirical-percentile, non-"
    "parametric) VaR and CVaR at 95%/99% — not a normal-distribution z-score approximation, "
    "and portfolio beta vs a benchmark index cross-checked against the weighted average of "
    "individual betas. Also reports a Herfindahl-index-based 'effective number of positions' "
    "(how concentrated the weights actually are, distinct from the raw holding count) — not "
    "an empyrical metric, computed directly. Use whenever the user gives actual holdings/"
    "weights and asks 'how risky is my portfolio' — this is the real computation, not "
    "individual-stock statistics treated as a stand-in for portfolio risk.",
    {"type": "object",
     "properties": {
         "holdings": {"type": "array",
                      "items": {"type": "object",
                                "properties": {"symbol": {"type": "string"},
                                               "weight": {"type": "number"}},
                                "required": ["symbol", "weight"]},
                      "description": "e.g. [{'symbol':'AAPL','weight':0.3}, "
                                     "{'symbol':'JPM','weight':0.7}] — weights need not "
                                     "sum to 1, they'll be renormalized"},
         "benchmark": {"type": "string", "default": "sp500"},
         "years": {"type": "number", "default": 3.0,
                   "description": "lookback window in years if start/end not given"},
         "start": {"type": "string", "description": "YYYY-MM-DD"},
         "end": {"type": "string", "description": "YYYY-MM-DD"},
         "risk_free_pct": {"type": "number",
                           "description": "annual %, overrides the 10-year Treasury default"},
     },
     "required": ["holdings"]},
)
async def portfolio_risk(args):
    from . import portfolio_risk as pr
    def work():
        return pr.compute(args["holdings"], benchmark=args.get("benchmark", "sp500"),
                          years=float(args.get("years", 3.0)),
                          start=args.get("start"), end=args.get("end"),
                          risk_free_pct=args.get("risk_free_pct"))
    try:
        r = await asyncio.to_thread(work)
    except ValueError as e:
        return _err(str(e))
    return _text(_js(r))


@tool(
    "index_data",
    "US index levels. kind=prices: monthly OHLC for the last 3 years. kind=list: "
    "available indices (sp500, nasdaq, dowjones).",
    {"type": "object",
     "properties": {"index": {"type": "string"},
                    "kind": {"type": "string", "enum": ["prices", "list"],
                             "default": "prices"}},
     "required": []},
)
async def index_data(args):
    def work():
        kind = args.get("kind", "prices")
        if kind == "list" or not args.get("index"):
            return "Available indices: " + ", ".join(da.index_list())
        name = args["index"].lower().replace(" ", "")
        df = da.index_prices(name)
        if df is None:
            return None
        m = (df.set_index("Date").resample("ME")
               .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
               .dropna().reset_index())
        m["Date"] = m["Date"].dt.date
        return da.df_to_md(m.round(1).tail(36))
    r = await asyncio.to_thread(work)
    if r is None:
        return _err(f"unknown index '{args.get('index')}'. Available: {', '.join(da.index_list())}")
    return _text(r)


@tool(
    "macro_data",
    "Rates and commodity series (daily closes): treasury_10y (10-year Treasury yield, %), "
    "crude (Brent, USD/bbl), gold and silver (COMEX, USD/oz). Omit 'series' to list "
    "them. Returns the most recent 24 observations.",
    {"type": "object",
     "properties": {"series": {"type": "string"}},
     "required": []},
)
async def macro_data(args):
    def work():
        if not args.get("series"):
            return "Available series: " + ", ".join(da.macro_list())
        df = da.macro_series(args["series"])
        if df is None:
            return None
        return da.df_to_md(df.tail(24))
    r = await asyncio.to_thread(work)
    if r is None:
        return _err(f"unknown series '{args.get('series')}'. {', '.join(da.macro_list())}")
    return _text(r)


# SEC form types behind each document kind. Foreign issuers file 20-F/40-F in place of
# a 10-K and 6-K in place of an 8-K, so those sit in the same buckets.
_FORM_KINDS = {
    "annual": ("10-K", "10-K/A", "20-F", "40-F"),
    "quarterly": ("10-Q", "10-Q/A"),
    "current": ("8-K", "6-K"),
    "proxy": ("DEF 14A", "DEFA14A"),
}


@tool(
    "company_documents",
    "List a company's recent SEC filings with dates and direct URLs: annual reports "
    "(10-K, or 20-F/40-F for foreign issuers), quarterly reports (10-Q), current reports "
    "on material events (8-K/6-K) and proxy statements (DEF 14A). Then call "
    "read_document on the ones you need — use this when you want the company's own "
    "words: strategy, risk factors, MD&A, guidance.",
    {"type": "object",
     "properties": {"symbol": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["all", *sorted(_FORM_KINDS)],
                             "default": "all"},
                    "limit": {"type": "integer", "default": 15}},
     "required": ["symbol"]},
)
async def company_documents(args):
    from . import sec
    sym = args["symbol"].upper()
    kind = args.get("kind") or "all"
    if kind != "all" and kind not in _FORM_KINDS:
        return _err(f"unknown kind '{kind}'; one of {['all', *sorted(_FORM_KINDS)]}")
    forms = (tuple(f for fs in _FORM_KINDS.values() for f in fs)
             if kind == "all" else _FORM_KINDS[kind])
    docs = await asyncio.to_thread(sec.filings, sym, forms, int(args.get("limit", 15)))
    if not docs:
        blocked = await asyncio.to_thread(_sec_unavailable_reason)
        if blocked:
            return _err(blocked)
        return _err(f"no {kind} filings listed for {sym} — check the ticker with "
                    f"resolve_company")
    return _text(_js({"symbol": sym, "kind": kind, "filings": docs,
                      "_note": "Pass a url to read_document to get the filing's text."}))


@tool(
    "read_document",
    "Return the text of a filing listed by company_documents or executive_pay (any "
    "sec.gov filing URL), or of a PDF at an http(s) URL. SEC filings are HTML and come "
    "back as plain text, truncated at max_chars; a 10-K runs to a few hundred thousand "
    "characters, so raise max_chars only when the section you need is past the cut. "
    "Quote it with the form type and period.",
    {"type": "object",
     "properties": {"url": {"type": "string",
                            "description": "URL from company_documents or executive_pay"},
                    "max_chars": {"type": "integer", "default": 150000,
                                  "description": "SEC filings: truncate the text here"},
                    "max_pages": {"type": "integer", "default": 40,
                                  "description": "PDFs: read at most this many pages"}},
     "required": ["url"]},
)
async def read_document(args):
    url = str(args["url"]).strip()
    if not url.lower().startswith(("http://", "https://")):
        return _err("url must be an http(s) link, e.g. one returned by company_documents")
    host = (urlparse(url).hostname or "").lower()
    is_sec = host == "sec.gov" or host.endswith(".sec.gov")

    def work():
        if is_sec:
            from . import sec
            return sec.filing_text(url, max_chars=int(args.get("max_chars", 150_000)))
        from . import live
        return live.document_text(url, max_pages=int(args.get("max_pages", 40)))

    txt = await asyncio.to_thread(work)
    if not txt:
        # _err, not _text: a fetch that failed must not reach the model shaped like
        # a document that happened to be empty.
        if is_sec:
            blocked = await asyncio.to_thread(_sec_unavailable_reason)
            return _err(blocked or f"could not fetch {url} from sec.gov; check the URL "
                                   f"came from company_documents, or try again shortly")
        return _err(f"could not read a document from {url}. Outside sec.gov only PDFs "
                    f"are supported, and this one returned no PDF with a text layer — it "
                    f"may be an HTML page, a scanned image, a placeholder page, or "
                    f"temporarily unavailable.")
    return _text(txt)


@tool(
    "refresh_screening_data",
    "Rebuild the screening table (used by screen_stocks) from the SEC's bulk Company "
    "Facts archive, which the SEC regenerates nightly. action='status' (default) says "
    "when the table was built, which SEC archive it came from, whether that archive has "
    "changed since, and the progress of any refresh under way. action='start' begins a "
    "refresh in the background and returns at once: the download is ~1.4 GB and parsing "
    "~20,000 filers takes several minutes, so call action='status' again later rather "
    "than waiting. If the SEC archive is unchanged since the last build nothing is "
    "downloaded (force=true overrides that). Refreshes are limited to one per ~20 hours. "
    "The refreshed table is used immediately; whether it also survives a server restart "
    "depends on a Hugging Face dataset being configured with a write token, and the "
    "status says which. When the server is protected, action='start' needs the "
    "operator's passphrase -- ask the user for it rather than guessing, and never pass "
    "a Hugging Face token. Use when the user asks to update, refresh or bring current "
    "the screening data.",
    {"type": "object",
     "properties": {
         "action": {"type": "string", "enum": ["status", "start"], "default": "status"},
         "force": {"type": "boolean", "default": False,
                   "description": "rebuild even if the SEC archive has not changed"},
         "passphrase": {"type": "string",
                        "description": "the operator's refresh passphrase (the server's "
                                       "REFRESH_TOKEN setting). Only needed when the server "
                                       "is protected. Never a Hugging Face hf_ token."}},
     "required": []},
)
async def refresh_screening_data(args):
    from . import sec_refresh
    action = args.get("action") or "status"
    if action not in ("status", "start"):
        return _err(f"unknown action '{action}'; one of ['status', 'start']")
    if action == "start":
        st = await asyncio.to_thread(sec_refresh.start, bool(args.get("force")),
                                     args.get("passphrase"))
        lead = ("refresh started; call refresh_screening_data(action='status') in a few "
                "minutes" if st.get("accepted") else f"refresh not started: {st.get('reason')}")
        return _text(lead + "\n\n" + _js(st))
    st = await asyncio.to_thread(sec_refresh.status)
    if st.get("error"):
        return _err(f"last refresh failed: {st['error']}\n\n" + _js(st))
    return _text(_js(st))


@tool(
    "insider_trades",
    "Insider dealing from SEC Forms 3/4/5 for one company and quarter. Officers "
    "and directors must report within two business days, so this is close to real "
    "time. Separates open-market BUYS and SELLS (codes P/S — actual decisions) from "
    "grants, option exercises and tax withholding (A/M/F — compensation mechanics), "
    "which must not be read as conviction trades. Covers every SEC filer.",
    {"type": "object",
     "properties": {"symbol": {"type": "string"},
                    "year": {"type": "integer", "description": "calendar year, e.g. 2026"},
                    "quarter": {"type": "integer", "enum": [1, 2, 3, 4]},
                    "limit": {"type": "integer", "default": 40}},
     "required": ["symbol", "year", "quarter"]},
)
async def insider_trades(args):
    def work():
        from . import insider
        return insider.transactions(args["symbol"], int(args["year"]),
                                    int(args["quarter"]), int(args.get("limit", 40)))
    res = await asyncio.to_thread(work)
    if res is None:
        blocked = await asyncio.to_thread(_sec_unavailable_reason)
        if blocked:
            return _err(blocked)
        return _err(f"no insider data for {args['symbol']} "
                    f"{args['year']}Q{args['quarter']} — check the symbol is an SEC "
                    f"filer, or that the quarter's data set has been published yet")
    return _text(_js(res))


@tool(
    "executive_pay",
    "Locate executive-compensation disclosure for a company: the DEF 14A proxy "
    "statements containing the Summary Compensation Table (salary, bonus, stock and "
    "option awards per named officer), CEO pay ratio, pay-versus-performance, and "
    "the beneficial-ownership table. Returns filing URLs to read with "
    "read_document — the figures are deliberately not auto-parsed, because "
    "compensation tables vary in shape between filers and mis-attributing pay "
    "between officers is worse than reading the table.",
    {"type": "object",
     "properties": {"symbol": {"type": "string"}},
     "required": ["symbol"]},
)
async def executive_pay(args):
    def work():
        from . import insider
        return insider.compensation_context(args["symbol"])
    res = await asyncio.to_thread(work)
    if res is None:
        blocked = await asyncio.to_thread(_sec_unavailable_reason)
        if blocked:
            return _err(blocked)
        return _err(f"{args['symbol']} is not an SEC filer (no CIK) — check the ticker "
                    f"with resolve_company")
    return _text(_js(res))


# ALL_TOOLS is assembled from the module's globals rather than written out by hand:
# a hand-kept list silently omits any tool defined after it, and a tool missing from
# the list is invisible to the model with no error raised anywhere.
def _collect_tools():
    """Every @tool-decorated object defined in this module, name-sorted.

    Discovery rather than a hand-kept list, so adding a tool is one edit and cannot
    silently fail to register. Sorted so the MCP tools/list order is deterministic
    (MCP 2026-07-28 asks for this, and it keeps the model's prompt cache warm).
    """
    seen = {}
    for obj in globals().values():
        name = getattr(obj, "name", None)
        if isinstance(name, str) and hasattr(obj, "handler") and hasattr(obj, "input_schema"):
            seen[name] = obj
    return [seen[k] for k in sorted(seen)]


ALL_TOOLS = _collect_tools()
