"""Exclusion screening by category.

Each category is a values judgment the USER makes, not the agent — this module
only mechanizes categories the user has explicitly confirmed. Categories are
matched against whatever descriptive text the table carries (the company name,
plus description/sector/industry columns when present), as a case-insensitive
substring.

To add a category: add an entry to EXCLUSION_CATEGORIES with its keywords and a
short reason, which is reported next to every company it excludes.
"""
from __future__ import annotations

import pandas as pd

EXCLUSION_CATEGORIES: dict[str, dict] = {
    "tobacco": {
        "keywords": ["tobacco", "cigarette", "cigar"],
        "reason": "excluded at the user's request: tobacco products",
    },
    "gambling": {
        "keywords": ["gambling", "casino", "lottery", "betting", "wagering"],
        "reason": "excluded at the user's request: gambling operations",
    },
}


def _match_columns(df: pd.DataFrame) -> list[str]:
    # A business description, when a table has one, matters MORE than sector tags:
    # conglomerates are often tagged by their largest segment even when an excluded
    # product line is a large business of theirs, so tag-only matching misses them.
    return [c for c in ("about", "sector", "industry", "company_name") if c in df.columns]


def apply_exclusions(df: pd.DataFrame, categories: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (kept, excluded). excluded gets an 'exclusion_reason' column.

    Unknown category names are ignored (not silently excluded-nothing — caller
    should surface a warning if a requested category isn't in EXCLUSION_CATEGORIES).
    """
    cols = _match_columns(df)
    excluded_mask = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=object)

    for cat in categories:
        spec = EXCLUSION_CATEGORIES.get(cat)
        if not spec:
            continue
        cat_mask = pd.Series(False, index=df.index)
        for col in cols:
            text = df[col].astype(str).str.lower()
            for kw in spec["keywords"]:
                cat_mask |= text.str.contains(kw, na=False, regex=False)
        newly = cat_mask & ~excluded_mask
        reasons[newly] = f"[{cat}] {spec['reason']}"
        excluded_mask |= cat_mask

    kept = df[~excluded_mask].copy()
    excluded = df[excluded_mask].copy()
    excluded["exclusion_reason"] = reasons[excluded_mask]
    return kept, excluded


def known_categories() -> list[str]:
    return sorted(EXCLUSION_CATEGORIES)


def unknown_categories(requested: list[str]) -> list[str]:
    return [c for c in requested if c not in EXCLUSION_CATEGORIES]
