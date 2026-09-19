"""Stock screening over the SEC-derived table of US filers.

`sec_refresh` builds the table from the SEC's bulk Company Facts archive: one row
per filer with its latest annual figures. Screening = objective filtering on user
criteria. No recommendations.
"""
from __future__ import annotations

import pandas as pd

from .sec_refresh import OUT as UNIVERSE

FILTERABLE = ["public_float_usd_m", "revenue_usd_m", "net_income_usd_m",
              "assets_usd_m", "fcf_usd_m", "pe_on_float", "roe_pct", "roa_pct",
              "gross_margin_pct", "operating_margin_pct", "net_margin_pct",
              "fcf_margin_pct", "debt_equity", "current_ratio"]

SHOW_COLS = ["symbol", "company_name", "public_float_usd_m", "revenue_usd_m",
             "net_income_usd_m", "pe_on_float", "roe_pct", "roa_pct",
             "gross_margin_pct", "operating_margin_pct", "net_margin_pct",
             "fcf_margin_pct", "debt_equity", "current_ratio", "as_of"]

DEFAULT_SORT = "public_float_usd_m"


def build_metrics() -> pd.DataFrame:
    """The screening table, with size columns in USD millions.

    Metrics a filing cannot supply are left ABSENT rather than faked: there is no
    price feed behind this table, so it has no price, P/E, P/B or momentum columns,
    and nothing is ranked on an invented number.
    """
    if not UNIVERSE.exists():
        # Reaches the model verbatim through the MCP error path, so it names the
        # tool that fixes it.
        raise FileNotFoundError(
            "the screening table has not been built on this server yet — run "
            "refresh_screening_data(action='start') (a ~1.4 GB SEC download plus a "
            "few minutes of parsing), then retry")
    df = pd.read_parquet(UNIVERSE)
    df = df[df["symbol"].notna()].copy()

    # Size is PUBLIC FLOAT: the market value of non-affiliate shares from the 10-K
    # cover, the only market-value figure a filing carries.
    df["public_float_usd_m"] = (pd.to_numeric(df["public_float_usd"], errors="coerce")
                                / 1e6).round(1)
    df["revenue_usd_m"] = df.get("revenue_musd")
    df["net_income_usd_m"] = df.get("net_income_musd")
    df["assets_usd_m"] = df.get("assets_musd")
    df["fcf_usd_m"] = df.get("fcf_musd")

    # An earnings multiple on FLOAT, not market cap: float excludes insider-held
    # stock, so this understates the true multiple for founder-controlled companies.
    # Named for exactly what it is so it is never read as a quoted P/E.
    ni = pd.to_numeric(df["net_income"], errors="coerce")
    fl = pd.to_numeric(df["public_float_usd"], errors="coerce")
    df["pe_on_float"] = (fl / ni.where(ni > 0)).round(1)
    return df.reset_index(drop=True)


def _find_near_misses(pre_numeric: pd.DataFrame, numeric_filters: list[tuple[str, str, float]],
                      tolerance_pct: float) -> list[dict]:
    """Companies that failed EXACTLY ONE of the numeric filters, by no more than
    tolerance_pct of that filter's threshold, while passing every other one —
    the 'left out by a small margin' set, so a strict screen's cost is visible."""
    pass_mask = {}
    miss_margin = {}   # % by which the filter was missed, only where it WAS missed
    for col, kind, v in numeric_filters:
        vals = pd.to_numeric(pre_numeric[col], errors="coerce")
        # A numeric screen is a verified-pass set: unknown values are not matches.
        # Missing values can never be reported as a near miss because there is no
        # observed distance from the threshold.
        if kind == "min":
            pass_mask[col] = vals.notna() & (vals >= float(v))
            denom = abs(v) if v else 1.0
            miss_margin[col] = ((float(v) - vals) / denom * 100).clip(lower=0)
        else:
            pass_mask[col] = vals.notna() & (vals <= float(v))
            denom = abs(v) if v else 1.0
            miss_margin[col] = ((vals - float(v)) / denom * 100).clip(lower=0)

    pass_df = pd.DataFrame(pass_mask)
    n_failed = (~pass_df).sum(axis=1)
    exactly_one = n_failed == 1
    if not exactly_one.any():
        return []

    out = []
    idx = pre_numeric.index[exactly_one]
    for i in idx:
        failed_cols = [c for c in pass_mask if not pass_mask[c].loc[i]]
        if len(failed_cols) != 1:
            continue
        col = failed_cols[0]
        margin = float(miss_margin[col].loc[i])
        # NaN margin means the value itself is missing, not "close but short" —
        # that's not a near miss, it's unknown data; exclude it rather than
        # report a meaningless "missed by nan%".
        if pd.isna(margin) or margin > tolerance_pct:
            continue
        kind = next(k for c, k, v in numeric_filters if c == col)
        thresh = next(v for c, k, v in numeric_filters if c == col)
        out.append({
            "symbol": pre_numeric.loc[i, "symbol"],
            "company_name": pre_numeric.loc[i].get("company_name"),
            "missed_filter": f"{col} {'>=' if kind == 'min' else '<='} {thresh}",
            "actual_value": round(float(pd.to_numeric(pre_numeric.loc[i, col], errors="coerce")), 2),
            "missed_by_pct": round(margin, 1),
        })
    out.sort(key=lambda r: r["missed_by_pct"])
    return out[:15]


def screen(
    min_filters: dict | None = None,     # {"roe_pct": 15, "revenue_usd_m": 500, ...}
    max_filters: dict | None = None,     # {"pe_on_float": 25, "debt_equity": 0.5, ...}
    exclude_categories: list[str] | None = None,
    sort_by: str = DEFAULT_SORT,
    ascending: bool = False,
    limit: int = 25,
    near_miss_tolerance_pct: float = 15.0,   # 0 disables near-miss reporting
) -> tuple[pd.DataFrame, int, dict]:
    """Return (result_df, matched_count, summary).

    summary: {"applied": [...], "unknown": [...], "excluded_count": N,
    "by_category": {cat: count}, "funnel": [...], "near_misses": [...],
    "coverage": {...}, "unusable_filters": [...], "sorted_by": col}.
    "funnel" shows the remaining count after each filter is applied, in order —
    "how many did each criterion drop." "near_misses" lists companies that
    failed EXACTLY ONE numeric filter, by no more than near_miss_tolerance_pct
    of that filter's threshold, and would otherwise have matched — i.e. the
    ones you left out by a small margin, so a screen's strictness is visible,
    not just its final count. A company MISSING data for a filtered column is
    excluded from the verified result set: absence is not evidence it fails the
    bar, but it is also not evidence it passes. Coverage and missing-data drops
    are reported so callers can describe how representative the screen is.
    """
    unknown_cols = sorted(c for c in {*(min_filters or {}), *(max_filters or {})}
                          if c not in FILTERABLE)
    if unknown_cols:
        raise ValueError(f"unknown filter column(s) {unknown_cols}; filterable: {FILTERABLE}")

    df = build_metrics()
    summary = {"applied": [], "unknown": [], "excluded_count": 0,
               "by_category": {}, "funnel": [], "near_misses": []}

    if exclude_categories:
        from . import ethics
        unknown = ethics.unknown_categories(exclude_categories)
        known = [c for c in exclude_categories if c not in unknown]
        if known:
            df, excluded_df = ethics.apply_exclusions(df, known)
            summary["applied"] = known
            summary["excluded_count"] = len(excluded_df)
            if not excluded_df.empty:
                for cat in known:
                    n = excluded_df["exclusion_reason"].str.startswith(f"[{cat}]").sum()
                    if n:
                        summary["by_category"][cat] = int(n)
        summary["unknown"] = unknown
    summary["funnel"].append({"stage": "universe (after exclusions)", "remaining": int(len(df))})

    # candidates before numeric filters — the base for near-miss comparison
    pre_numeric = df

    numeric_filters: list[tuple[str, str, float]] = (  # (col, "min"|"max", value)
        [(c, "min", v) for c, v in (min_filters or {}).items()] +
        [(c, "max", v) for c, v in (max_filters or {}).items()]
    )
    # Measure source coverage before applying any numeric filter. Final matches
    # necessarily have 100% coverage, which would hide whether the underlying
    # universe was only sparsely measured.
    coverage = {}
    for c in sorted({*((min_filters or {}).keys()), *((max_filters or {}).keys())}):
        have = int(pd.to_numeric(pre_numeric[c], errors="coerce").notna().sum())
        coverage[c] = {
            "with_data": have,
            "of_candidates": int(len(pre_numeric)),
            "pct": round(100 * have / len(pre_numeric), 1) if len(pre_numeric) else 0.0,
        }
    summary["coverage"] = coverage
    summary["unusable_filters"] = [c for c, item in coverage.items() if item["pct"] < 50]

    for col, kind, v in numeric_filters:
        vals = pd.to_numeric(df[col], errors="coerce")
        cleared = (vals >= float(v)) if kind == "min" else (vals <= float(v))
        dropped_missing = int(vals.isna().sum())
        df = df[cleared]
        summary["funnel"].append({
            "stage": f"{col} {'>=' if kind == 'min' else '<='} {v}",
            "remaining": int(len(df)),
            "dropped_missing_data": dropped_missing,
        })

    if numeric_filters and near_miss_tolerance_pct > 0 and not pre_numeric.empty:
        summary["near_misses"] = _find_near_misses(
            pre_numeric, numeric_filters, near_miss_tolerance_pct)

    matched = len(df)
    if sort_by not in df.columns:
        sort_by = DEFAULT_SORT
    summary["sorted_by"] = sort_by
    df = df.sort_values(sort_by, ascending=ascending, na_position="last")

    show_cols = [c for c in SHOW_COLS if c in df.columns]
    extra = [c for c in {sort_by, *((min_filters or {}).keys()), *((max_filters or {}).keys())}
             if c in df.columns and c not in show_cols]
    return (df[show_cols + extra].head(limit).reset_index(drop=True), matched, summary)
