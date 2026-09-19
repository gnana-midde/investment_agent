"""Portfolio-level risk: real historical correlation/covariance, portfolio variance,
and per-series risk stats — computed from actual daily return history, not a
single-asset proxy or a normal-distribution assumption.

Per-series stats (Sharpe, Sortino, historical VaR/CVaR, max drawdown, Calmar, beta) are
computed by calling `empyrical` (empyrical-reloaded on PyPI, the maintained fork of
Quantopian's open-source risk-stats library used throughout the buy-side/quant industry
via pyfolio/zipline) on the weighted portfolio-return series, rather than
reimplementing those formulas by hand. Built here, because empyrical does not do it:
the weighted multi-asset portfolio return series itself, the correlation matrix between
holdings, the Markowitz w'*cov*w cross-check, the risk-free rate from the 10-year
Treasury yield, and the Herfindahl-index concentration read.
"""
from __future__ import annotations

import empyrical as em
import numpy as np
import pandas as pd

from . import data_access as da

TRADING_DAYS = 252


def _adj_close(symbol: str) -> pd.Series | None:
    df = da.prices(symbol)
    if df is None or df.empty or "Close" not in df:
        return None
    col = "Adj Close" if "Adj Close" in df else "Close"
    s = df.set_index(pd.to_datetime(df["Date"]).dt.tz_localize(None))[col].astype(float).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _returns_frame(symbols: list[str], years: float, start: str | None, end: str | None):
    series = {}
    missing = []
    for sym in symbols:
        s = _adj_close(sym)
        if s is None or len(s) < 30:
            missing.append(sym)
            continue
        series[sym] = s.pct_change().dropna()
    if missing:
        raise ValueError(f"no usable price history for: {', '.join(missing)}")
    rets = pd.concat(series, axis=1, join="inner").sort_index()
    if start:
        rets = rets[rets.index >= pd.Timestamp(start)]
    if end:
        rets = rets[rets.index <= pd.Timestamp(end)]
    elif not start:
        cutoff = rets.index.max() - pd.Timedelta(days=int(years * 365.25))
        rets = rets[rets.index >= cutoff]
    return rets


def _risk_free_annual(risk_free_pct: float | None) -> tuple[float, str]:
    if risk_free_pct is not None:
        return risk_free_pct / 100, "user-supplied"
    g = da.macro_series("treasury_10y")
    if g is not None and not g.empty:
        row = g.dropna().iloc[-1]
        return (float(row["treasury_10y"]) / 100,
                f"10-year Treasury yield as of {row.get('date', '?')} (macro_data)")
    return 0.0, "no risk-free proxy available — assumed 0%"


def compute(holdings: list[dict], benchmark: str = "sp500", years: float = 3.0,
            start: str | None = None, end: str | None = None,
            risk_free_pct: float | None = None) -> dict:
    symbols = [h["symbol"].upper() for h in holdings]
    raw_w = np.array([float(h["weight"]) for h in holdings], dtype=float)
    if raw_w.sum() <= 0:
        raise ValueError("weights must sum to a positive number")
    weights_renormalized = abs(round(float(raw_w.sum()), 4) - 1.0) > 0.01
    w = raw_w / raw_w.sum()

    rets = _returns_frame(symbols, years, start, end)
    if len(rets) < 60:
        raise ValueError(f"only {len(rets)} overlapping trading days across {symbols} "
                          "— need at least ~60 for a meaningful covariance estimate")
    rets = rets[symbols]  # enforce order matching w

    bench = da.index_prices(benchmark)
    bench_rets = None
    if bench is not None and "Close" in bench:
        bs = bench.set_index(pd.to_datetime(bench["Date"]).dt.tz_localize(None))["Close"] \
                  .astype(float).pct_change().dropna()
        bench_rets = bs.reindex(rets.index).dropna()

    cov_annual = rets.cov() * TRADING_DAYS
    corr = rets.corr()
    indiv_vol_annual = (rets.std() * np.sqrt(TRADING_DAYS) * 100).round(2)

    port_daily = rets.dot(w)  # daily-rebalanced-to-target-weight portfolio return series
    port_vol_matrix = float(np.sqrt(w @ cov_annual.values @ w))
    port_vol_empyrical = float(em.annual_volatility(port_daily))

    rf_annual, rf_source = _risk_free_annual(risk_free_pct)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1

    cagr = float(em.annual_return(port_daily))
    mdd = float(em.max_drawdown(port_daily))
    calmar = float(em.calmar_ratio(port_daily)) if mdd else None
    sharpe = float(em.sharpe_ratio(port_daily, risk_free=rf_daily))
    sortino = float(em.sortino_ratio(port_daily, required_return=rf_daily))
    var95 = float(em.value_at_risk(port_daily, cutoff=0.05))
    cvar95 = float(em.conditional_value_at_risk(port_daily, cutoff=0.05))
    var99 = float(em.value_at_risk(port_daily, cutoff=0.01))
    cvar99 = float(em.conditional_value_at_risk(port_daily, cutoff=0.01))

    beta_port = None
    indiv_betas = {}
    if bench_rets is not None and len(bench_rets) >= 60:
        joined = pd.concat([port_daily, bench_rets], axis=1, join="inner").dropna()
        if len(joined) >= 60:
            beta_port = float(em.beta(joined.iloc[:, 0], joined.iloc[:, 1]))
        for sym in symbols:
            j2 = pd.concat([rets[sym], bench_rets], axis=1, join="inner").dropna()
            if len(j2) >= 60:
                indiv_betas[sym] = round(float(em.beta(j2.iloc[:, 0], j2.iloc[:, 1])), 2)
    beta_weighted_check = (round(float(sum(w[i] * indiv_betas[s] for i, s in enumerate(symbols)
                                           if s in indiv_betas)), 2)
                          if indiv_betas else None)

    hhi = float(np.sum(w ** 2))
    effective_n = round(1 / hhi, 1) if hhi else None

    return {
        "holdings": [{"symbol": s, "weight_used": round(float(w[i]), 4)}
                     for i, s in enumerate(symbols)],
        "weights_were_renormalized_to_sum_1": weights_renormalized,
        "window": {"start": str(rets.index.min().date()), "end": str(rets.index.max().date()),
                   "trading_days": len(port_daily)},
        "individual_annualized_volatility_pct": indiv_vol_annual.to_dict(),
        "correlation_matrix": corr.round(2).to_dict(),
        "portfolio_annualized_volatility_pct": {
            "matrix_form_w_cov_w": round(port_vol_matrix * 100, 2),
            "empyrical_direct_from_return_series": round(port_vol_empyrical * 100, 2),
            "note": "two independent computations of the same quantity shown for "
                    "cross-check; small differences are floating-point only",
        },
        "diversification": {
            "herfindahl_index": round(hhi, 4),
            "effective_number_of_positions": effective_n,
            "note": "1/HHI — the count of EQUAL-weighted positions that would give the "
                    "same concentration as these actual weights; lower than the raw "
                    "holding count whenever weights are uneven",
        },
        "portfolio_cagr_pct": round(cagr * 100, 2),
        "portfolio_max_drawdown_pct": round(mdd * 100, 2),
        "calmar_ratio": round(calmar, 2) if calmar is not None else None,
        "risk_free_rate_annual_pct": round(rf_annual * 100, 2),
        "risk_free_source": rf_source,
        "sharpe_ratio_annualized": round(sharpe, 2),
        "sortino_ratio_annualized": round(sortino, 2),
        "historical_var_cvar_daily_pct": {
            "var_95": round(var95 * 100, 2), "cvar_95": round(cvar95 * 100, 2),
            "var_99": round(var99 * 100, 2), "cvar_99": round(cvar99 * 100, 2),
            "method": "empirical percentile of the actual daily portfolio-return "
                      "distribution over the window above (empyrical.value_at_risk/"
                      "conditional_value_at_risk) — NOT a parametric volatility*z-score "
                      "estimate, so no normality assumption",
        },
        "beta_vs_benchmark": {
            "portfolio_beta": round(beta_port, 2) if beta_port is not None else None,
            "weighted_avg_of_individual_betas_crosscheck": beta_weighted_check,
            "individual_betas": indiv_betas,
            "benchmark": benchmark,
            "note": "should match the crosscheck almost exactly (covariance is linear "
                    "in weights) when both are computed over the same aligned window; "
                    "a large gap usually means one name's history is much shorter and "
                    "is dragging the common window",
        } if beta_port is not None else {"note": f"no usable benchmark series for '{benchmark}'"},
        "stats_library": "empyrical-reloaded (github.com/stefan-jansen/pyfolio-reloaded, "
                          "maintained fork of Quantopian's empyrical)",
    }
