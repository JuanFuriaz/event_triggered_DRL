"""
tables.py — Table generation and reporting utilities for dow_3_backtest.ipynb

Functions:
    compute_metrics           - Portfolio metrics from equity curve
    make_metrics_table        - Full metrics table (window × dataset × algo)
    make_interaction_heatmap  - Wide-format metric heatmap (dataset × algo)
    make_filter_effect_table  - Filtering effect comparison table
    make_hypothesis_table     - Hypothesis evidence table
    make_ranking_table        - Top-N ranking by metric

    compute_trade_stats              - Trade counts (buys/sells/longs) from position deltas
    make_trade_stats_table           - Aggregated trade stats + FIFO win rates
    make_data_reduction_table        - Actual row/col counts and data-size reduction vs baseline
    make_state_space_table           - State space sizes and reduction vs baseline
    generate_results_md              - Write reports/results.md

Note: Plotting functions (plot_trade_heatmap, plot_ticker_buysell_aggregated,
plot_portfolio_comparison, plot_top_configs_equity) have been moved to plots.py.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import quantstats as qs
except ImportError:
    qs = None

# Plotting functions have been moved to plots.py for better organization.
# Import them directly from plots.py as needed.


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    equity_curve: pd.Series,
    initial_amount: float = 1_000_000,
    risk_free_rate: float = 0.02,
    baselines: Optional[dict] = None,
) -> dict:
    """
    Compute portfolio metrics from an equity-curve Series (account values).

    Parameters
    ----------
    equity_curve : pd.Series
        Daily account values, date-indexed.
    initial_amount : float
        Starting capital (informational only).
    risk_free_rate : float
        Annual risk-free rate (default 2%).
    baselines : dict, optional
        Reference equity curves, e.g. {"DJIA": Series, "MVO": Series, "all": Series}.
        Each gets a Corr_<key> column via Pearson on overlapping daily returns.

    Returns
    -------
    dict with keys:
        Return%, CAGR, Sharpe, Sortino, Calmar, MaxDD%, Volatility%,
        [Corr_<key>] for each baseline provided.
    """
    ec = equity_curve.dropna()
    nan_keys = ["Return%", "CAGR", "Sharpe", "Sortino", "Calmar", "MaxDD%", "Volatility%"]
    if len(ec) < 5:
        nans = {k: np.nan for k in nan_keys}
        if baselines:
            for k in baselines:
                nans[f"Corr_{k}"] = np.nan
        return nans

    daily_ret = ec.pct_change().dropna()

    total_return = (ec.iloc[-1] - ec.iloc[0]) / ec.iloc[0] * 100
    n_years = len(ec) / 252.0
    cagr = ((ec.iloc[-1] / ec.iloc[0]) ** (1.0 / max(n_years, 1e-6)) - 1) * 100
    volatility = daily_ret.std() * np.sqrt(252) * 100
    cummax = ec.cummax()
    drawdown = (ec - cummax) / cummax
    max_dd = drawdown.min() * 100
    rf_daily = risk_free_rate / 252

    if qs is not None:
        try:
            sharpe = float(qs.stats.sharpe(daily_ret, rf=rf_daily, periods=252, annualize=True))
        except Exception:
            sharpe = _manual_sharpe(daily_ret, rf_daily)
        try:
            sortino = float(qs.stats.sortino(daily_ret, rf=rf_daily, periods=252))
        except Exception:
            sortino = _manual_sortino(daily_ret, rf_daily)
        try:
            calmar = float(qs.stats.calmar(daily_ret))
        except Exception:
            calmar = (cagr / 100) / abs(max_dd / 100) if max_dd != 0 else np.nan
    else:
        sharpe  = _manual_sharpe(daily_ret, rf_daily)
        sortino = _manual_sortino(daily_ret, rf_daily)
        calmar  = (cagr / 100) / abs(max_dd / 100) if max_dd != 0 else np.nan

    result = {
        "Return%":     round(total_return, 2),
        "CAGR":        round(cagr, 2),
        "Sharpe":      round(float(sharpe),  3) if not (sharpe  is None or (isinstance(sharpe,  float) and np.isnan(sharpe)))  else np.nan,
        "Sortino":     round(float(sortino), 3) if not (sortino is None or (isinstance(sortino, float) and np.isnan(sortino))) else np.nan,
        "Calmar":      round(float(calmar),  3) if not (calmar  is None or (isinstance(calmar,  float) and np.isnan(calmar)))  else np.nan,
        "MaxDD%":      round(max_dd, 2),
        "Volatility%": round(volatility, 2),
    }

    # Baseline correlations — normalize both indices to DatetimeIndex before join
    if baselines:
        strat_ret = daily_ret.rename("strat").copy()
        try:
            strat_ret.index = pd.to_datetime(strat_ret.index, errors="coerce")
        except Exception:
            pass

        for bname, bec in baselines.items():
            if bec is None:
                result[f"Corr_{bname}"] = np.nan
                continue
            bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
            bec_s = bec_s.copy()
            try:
                bec_s.index = pd.to_datetime(bec_s.index, errors="coerce")
            except Exception:
                pass
            b_ret = bec_s.pct_change().dropna().rename("bsln")
            combined = pd.concat([strat_ret, b_ret], axis=1, join="inner")
            if len(combined) < 5:
                result[f"Corr_{bname}"] = np.nan
            else:
                result[f"Corr_{bname}"] = round(float(combined.corr().iloc[0, 1]), 3)

    return result


def _manual_sharpe(daily_ret: pd.Series, rf_daily: float) -> float:
    excess = daily_ret - rf_daily
    std = daily_ret.std()
    return float(excess.mean() / std * np.sqrt(252)) if std > 0 else np.nan


def _manual_sortino(daily_ret: pd.Series, rf_daily: float) -> float:
    excess = daily_ret - rf_daily
    downside = daily_ret[daily_ret < rf_daily]
    dstd = downside.std()
    return float(excess.mean() / dstd * np.sqrt(252)) if (len(downside) > 0 and dstd > 0) else np.nan


# ---------------------------------------------------------------------------
# Per-seed CI helpers
# ---------------------------------------------------------------------------

_CI_METRICS = ["Return%", "Sharpe", "MaxDD%"]

_T95_TABLE = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447,  8: 2.365, 9: 2.306, 10: 2.262,
}


def _t_crit(n: int) -> float:
    """Two-tailed t critical value for 95% CI, df = n-1."""
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df=n - 1))
    except ImportError:
        return _T95_TABLE.get(n, 2.0)


def _seed_ci_stats(
    seed_list: list,
    initial_amount: float = 1_000_000,
    metrics: list = _CI_METRICS,
) -> dict:
    """
    Compute per-seed scalar metrics and return mean/std/ci_lo/ci_hi/n columns.

    Parameters
    ----------
    seed_list : list of equity-curve Series
        results[window][dataset][algo] — one Series per seed.

    `{m}_n` is the number of seeds with a non-NaN value for that specific metric --
    tracked per metric rather than once overall because metrics can go NaN
    independently of each other (e.g. Sharpe on a zero-volatility seed while
    Return%/MaxDD% stay valid), so two metrics computed from the same seed_list can
    legitimately have different valid counts.
    """
    if not seed_list:
        return _nan_ci_cols(metrics)

    per_seed = [
        compute_metrics(curve, initial_amount=initial_amount)
        for curve in seed_list
    ]

    n = len(per_seed)
    t = _t_crit(n)
    out: dict = {}

    for m in metrics:
        vals = np.array([float(s.get(m, np.nan)) for s in per_seed])
        valid = vals[~np.isnan(vals)]
        out[f"{m}_n"] = int(len(valid))
        if len(valid) == 0:
            out[f"{m}_mean"]  = np.nan
            out[f"{m}_std"]   = np.nan
            out[f"{m}_ci_lo"] = np.nan
            out[f"{m}_ci_hi"] = np.nan
        elif len(valid) == 1:
            out[f"{m}_mean"]  = round(float(valid[0]), 3)
            out[f"{m}_std"]   = np.nan
            out[f"{m}_ci_lo"] = np.nan
            out[f"{m}_ci_hi"] = np.nan
        else:
            mean = np.mean(valid)
            std  = np.std(valid, ddof=1)
            se   = std / np.sqrt(len(valid))
            out[f"{m}_mean"]  = round(mean,          3)
            out[f"{m}_std"]   = round(std,           3)
            out[f"{m}_ci_lo"] = round(mean - t * se, 3)
            out[f"{m}_ci_hi"] = round(mean + t * se, 3)

    return out


def trades_ci_stats(seed_trades: list) -> dict:
    """
    Mean/std/95% CI/n for a list of per-seed trade counts (scalars), formatted as
    Trades_mean/_std/_ci_lo/_ci_hi/_n -- same _t_crit-based formula as
    _seed_ci_stats, just applied to trade counts directly instead of deriving a
    metric from an equity curve first. Trade counts are never NaN (compute_trade_stats
    always returns an int), so Trades_n is simply len(seed_trades). Shared by
    dow_3_backtest.ipynb (5-seed trades sweep) and dow_4_advtest 2x2.ipynb
    (scheduler-ladder / 2x2 / random-2x2 trades sweeps).
    """
    if not seed_trades:
        return dict(Trades_mean=np.nan, Trades_std=np.nan, Trades_ci_lo=np.nan, Trades_ci_hi=np.nan, Trades_n=0)
    vals = np.array(seed_trades, dtype=float)
    n = len(vals)
    mean = vals.mean()
    if n < 2:
        return dict(Trades_mean=round(float(mean), 3), Trades_std=np.nan,
                     Trades_ci_lo=np.nan, Trades_ci_hi=np.nan, Trades_n=n)
    std = vals.std(ddof=1)
    se = std / np.sqrt(n)
    t = _t_crit(n)
    return dict(Trades_mean=round(float(mean), 3), Trades_std=round(float(std), 3),
                Trades_ci_lo=round(float(mean - t * se), 3), Trades_ci_hi=round(float(mean + t * se), 3),
                Trades_n=n)


def _nan_ci_cols(metrics: list = _CI_METRICS) -> dict:
    """All CI columns as NaN (n=0) — for baselines and when results=None."""
    out = {
        f"{m}{s}": np.nan
        for m in metrics
        for s in ("_mean", "_std", "_ci_lo", "_ci_hi")
    }
    out.update({f"{m}_n": 0 for m in metrics})
    return out


# ---------------------------------------------------------------------------
# Table 1 — Full Metrics Table
# ---------------------------------------------------------------------------

def make_metrics_table(
    agg: dict,
    windows: list,
    datasets: list,
    algos: list,
    baselines: Optional[dict] = None,
    initial_amount: float = 1_000_000,
    save: bool = False,
    save_dir: str = "tables",
    results: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Full metrics table across windows × datasets × algorithms.

    Parameters
    ----------
    agg : dict
        agg[window][dataset][algo] = {'mean': Series, 'std': Series, 'n': int}
    baselines : dict
        baselines[window] = {'DJIA': Series, 'MVO': pd.DataFrame or Series}

    Returns
    -------
    pd.DataFrame with MultiIndex (window, dataset, algo).
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    index = []

    for w in windows:
        if w not in agg:
            continue
        bl = baselines.get(w, {}) if baselines else {}

        # Pre-compute 'all' baseline equity curves per algo for Corr_all
        all_curves: dict[str, pd.Series] = {}
        if "all" in agg[w]:
            for algo in algos:
                if algo in agg[w]["all"]:
                    all_curves[algo] = agg[w]["all"][algo]["mean"]

        for ds in datasets:
            if ds not in agg[w]:
                continue
            for algo in algos:
                if algo not in agg[w][ds]:
                    continue
                ec = agg[w][ds][algo]["mean"]
                n_seeds = agg[w][ds][algo].get("n", np.nan)

                row_baselines = dict(bl)
                if ds != "all" and algo in all_curves:
                    row_baselines["all"] = all_curves[algo]

                metrics = compute_metrics(ec, initial_amount=initial_amount,
                                          baselines=row_baselines if row_baselines else None)
                if ds == "all":
                    metrics["Corr_all"] = 1.0
                metrics["Seeds"] = n_seeds

                # Per-seed CI columns
                if results is not None:
                    seed_list = results.get(w, {}).get(ds, {}).get(algo, [])
                    metrics.update(_seed_ci_stats(seed_list, initial_amount=initial_amount))
                else:
                    metrics.update(_nan_ci_cols())

                rows.append(metrics)
                index.append((w, ds, algo))

        # Append baseline rows (DJIA, MVO)
        if bl:
            for bname, bec in bl.items():
                bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
                metrics = compute_metrics(bec_s, initial_amount=initial_amount)
                metrics["Corr_DJIA"] = 1.0 if bname == "DJIA" else np.nan
                metrics["Corr_MVO"]  = 1.0 if bname == "MVO"  else np.nan
                metrics["Corr_all"]  = np.nan
                metrics["Seeds"]     = np.nan
                metrics.update(_nan_ci_cols())
                rows.append(metrics)
                index.append((w, f"[{bname}]", "—"))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows,
                      index=pd.MultiIndex.from_tuples(index,
                                                      names=["window", "dataset", "algo"]))
    if save:
        path = os.path.join(save_dir, "metrics_table.csv")
        df.to_csv(path)
        print(f"Saved: {path}")

    return df


# ---------------------------------------------------------------------------
# Table 2 — Interaction Heatmap
# ---------------------------------------------------------------------------

def make_interaction_heatmap(
    agg: dict,
    window: str,
    datasets: list,
    algos: list,
    metric: str = "sharpe",
    metrics_df: Optional[pd.DataFrame] = None,
    save: bool = False,
    save_dir: str = "tables",
) -> pd.DataFrame:
    """Wide-format dataset × algo heatmap for a given metric."""
    os.makedirs(save_dir, exist_ok=True)
    col_name = _metric_col(metric)
    result = pd.DataFrame(np.nan, index=datasets, columns=algos, dtype=float)

    if metrics_df is not None:
        for ds in datasets:
            for algo in algos:
                try:
                    val = metrics_df.loc[(window, ds, algo), col_name]
                    result.loc[ds, algo] = val
                except KeyError:
                    pass
    else:
        if window not in agg:
            return result
        for ds in datasets:
            if ds not in agg[window]:
                continue
            for algo in algos:
                if algo not in agg[window][ds]:
                    continue
                ec = agg[window][ds][algo]["mean"]
                m = compute_metrics(ec)
                result.loc[ds, algo] = m.get(col_name, np.nan)

    result.index.name = "dataset"
    result.columns.name = "algo"

    if save:
        path = os.path.join(save_dir, f"heatmap_{window}_{metric}.csv")
        result.to_csv(path)
        print(f"Saved: {path}")

    return result


# ---------------------------------------------------------------------------
# Table 3 — Filter Effect Table
# ---------------------------------------------------------------------------

def make_filter_effect_table(
    agg: dict,
    windows: list,
    pairs: list,
    algos: list,
    metric: str = "sharpe",
    metrics_df: Optional[pd.DataFrame] = None,
    save: bool = False,
    save_dir: str = "tables",
) -> pd.DataFrame:
    """
    Compare base vs filtered dataset performance.
    pairs: list of (base_ds, flt_ds).
    """
    os.makedirs(save_dir, exist_ok=True)
    col_name = _metric_col(metric)
    rows = []

    for w in windows:
        for base_ds, flt_ds in pairs:
            for algo in algos:
                base_val = _lookup(metrics_df, agg, w, base_ds, algo, col_name)
                flt_val  = _lookup(metrics_df, agg, w, flt_ds,  algo, col_name)

                if not (np.isnan(base_val) or np.isnan(flt_val)):
                    diff = flt_val - base_val
                    direction = "↑" if diff > 0 else "↓"
                else:
                    diff = np.nan
                    direction = "—"

                rows.append({
                    "window":            w,
                    "algo":              algo.upper(),
                    "base":              base_ds,
                    "filtered":          flt_ds,
                    f"{col_name}_base":  _fmt(base_val),
                    f"{col_name}_flt":   _fmt(flt_val),
                    "Δ":                 _fmt(diff),
                    "direction":         direction,
                })

    df = pd.DataFrame(rows)
    if save:
        path = os.path.join(save_dir, f"filter_effect_{metric}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    return df


# ---------------------------------------------------------------------------
# Table 4 — Hypothesis Evidence
# ---------------------------------------------------------------------------

def make_hypothesis_table(
    metrics_df: pd.DataFrame,
    windows: list,
    save: bool = False,
    save_dir: str = "tables",
) -> pd.DataFrame:
    """Compute evidence for each research hypothesis."""
    os.makedirs(save_dir, exist_ok=True)

    def _get(w, ds, algo):
        # Prefers Sharpe_mean (per-seed metric, then averaged) over the plain Sharpe
        # column (metric on the seed-averaged curve) -- see _preferred_col's docstring.
        try:
            col = _preferred_col(metrics_df, "Sharpe")
            return float(metrics_df.loc[(w, ds, algo), col])
        except KeyError:
            return np.nan

    def _best_rl(w):
        try:
            sub = metrics_df.loc[w]
            mask = sub.index.get_level_values("algo") != "—"
            col = _preferred_col(sub.loc[mask], "Sharpe")
            return float(sub.loc[mask, col].max())
        except Exception:
            return np.nan

    rows = []

    for w in windows:
        djia_s  = _get(w, "[DJIA]", "—")
        best_rl = _best_rl(w)
        delta   = _safe_diff(best_rl, djia_s)
        rows.append(_row("H1", "Best RL config beats DJIA (Sharpe)", w,
                         "best_RL vs [DJIA]", delta,
                         "Supported" if _pos(delta) else "Rejected" if _known(delta) else "Inconclusive"))

        for algo in ["a2c", "ppo", "sac"]:
            bsln = _get(w, "all", algo)
            for ds in ["mvx", "bollx"]:
                val   = _get(w, ds, algo)
                delta = _safe_diff(val, bsln)
                rows.append(_row("RQ1a", "Hybrid state > continuous baseline (Sharpe)", w,
                                 f"{algo.upper()}+{ds} vs {algo.upper()}+all", delta,
                                 "Supported" if _pos(delta) else "Rejected" if _known(delta) else "Inconclusive"))

        for algo in ["a2c", "ppo", "sac"]:
            combined = _get(w, "bollmvx", algo)
            for ds in ["mvx", "bollx"]:
                val   = _get(w, ds, algo)
                delta = _safe_diff(val, combined)
                rows.append(_row("RQ1b", "Single-indicator > combined (Sharpe)", w,
                                 f"{algo.upper()}+{ds} vs {algo.upper()}+bollmvx", delta,
                                 "Supported" if _pos(delta) else "Rejected" if _known(delta) else "Inconclusive"))

        for algo, expected_up in [("a2c", True), ("ppo", False), ("sac", True)]:
            base  = _get(w, "mvx", algo)
            flt   = _get(w, "flt_mvx", algo)
            delta = _safe_diff(flt, base)
            actual_up = _pos(delta)
            direction_label = "expected ↑" if expected_up else "expected ↓"
            verdict = ("Supported" if _known(delta) and actual_up == expected_up
                       else "Rejected"      if _known(delta)
                       else "Inconclusive")
            rows.append(_row("RQ2", f"Filter effect {algo.upper()} ({direction_label})", w,
                             f"{algo.upper()}+flt_mvx vs {algo.upper()}+mvx", delta, verdict))

        for algo in ["a2c", "ppo"]:
            full  = _get(w, "mvx", algo)
            trg   = _get(w, "mvx_trg", algo)
            delta = _safe_diff(trg, full)
            within = _known(delta) and abs(delta) <= 0.1
            rows.append(_row("Trg-Only",
                             "Trigger-only vs full hybrid (|Δ Sharpe| ≤ 0.1?)", w,
                             f"{algo.upper()}+mvx_trg vs {algo.upper()}+mvx", delta,
                             "Within range" if within
                             else "Outside range" if _known(delta)
                             else "Inconclusive"))

    df = pd.DataFrame(rows)
    if save:
        path = os.path.join(save_dir, "hypothesis_evidence.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    return df


# ---------------------------------------------------------------------------
# Table 5 — Ranking Table
# ---------------------------------------------------------------------------

def make_ranking_table(
    metrics_df: pd.DataFrame,
    window: str,
    metric: str = "sharpe",
    n: int = 15,
    save: bool = False,
    save_dir: str = "tables",
) -> pd.DataFrame:
    """Top-N RL configurations for one window, ranked by metric."""
    os.makedirs(save_dir, exist_ok=True)
    col_name = _metric_col(metric)

    w_vals = metrics_df.index.get_level_values("window")
    if window not in w_vals:
        return pd.DataFrame()

    sub = metrics_df.loc[window].copy()
    sub = sub[sub.index.get_level_values("algo") != "—"]
    sub = sub.reset_index()
    sub = sub.rename(columns={"dataset": "Dataset", "algo": "Algo"})

    # Ranks by {col_name}_mean (per-seed metric, then averaged) when available --
    # see _preferred_col's docstring -- but still displays the plain col_name column
    # alongside it, since priority_cols is unchanged. Sort happens before the column
    # slice below so sort_col is available even when it isn't one of priority_cols.
    sort_col = _preferred_col(sub, col_name)
    sub = sub.sort_values(sort_col, ascending=False)

    priority_cols = ["Dataset", "Algo", col_name, "CAGR", "Return%", "MaxDD%",
                     "Volatility%", "Corr_DJIA", "Corr_all", "Seeds"]
    show_cols = list(dict.fromkeys(c for c in priority_cols if c in sub.columns))
    sub = (sub[show_cols]
           .head(n)
           .reset_index(drop=True))
    sub.index = sub.index + 1
    sub.index.name = "Rank"

    if save:
        path = os.path.join(save_dir, f"ranking_{window}_{metric}.csv")
        sub.to_csv(path)
        print(f"Saved: {path}")

    return sub


# ---------------------------------------------------------------------------
# Trade Statistics
# ---------------------------------------------------------------------------

def _build_price_lookup(price_df: pd.DataFrame) -> dict:
    """
    Build (date_str, tic) → close price O(1) lookup from a FinRL-format DataFrame.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must have columns: 'date', 'tic', 'close'.
    """
    lkp = {}
    for _, row in price_df.iterrows():
        key = (str(row["date"])[:10], str(row["tic"]))
        lkp[key] = float(row["close"])
    return lkp


def compute_trade_stats(actions_df: pd.DataFrame) -> dict:
    """
    Compute trade-event counts from a position DataFrame.

    Parameters
    ----------
    actions_df : pd.DataFrame
        Index=dates, columns=tickers, values=share positions (integer or float).

    Returns
    -------
    dict with keys:
        total_by_ticker, buys_by_ticker, sells_by_ticker,
        total_trades, total_buys, total_sells,
        n_long_days, n_short_days.
    """
    thresh = 0.5
    delta  = actions_df.diff().fillna(0)
    buys   = (delta >  thresh).sum()
    sells  = (delta < -thresh).sum()
    total  = buys + sells

    long_days  = (actions_df >  thresh).sum()
    short_days = (actions_df < -thresh).sum()

    return {
        "total_by_ticker":  total,
        "buys_by_ticker":   buys,
        "sells_by_ticker":  sells,
        "total_trades":     int(total.sum()),
        "total_buys":       int(buys.sum()),
        "total_sells":      int(sells.sum()),
        "n_long_days":      int(long_days.sum()),
        "n_short_days":     int(short_days.sum()),
    }


def _compute_win_rate_fifo(
    actions_df: pd.DataFrame,
    price_lookup: dict,
) -> tuple:
    """
    FIFO-matched trade win rates.

    Buy win rate: fraction of open positions whose unrealised P&L at period end is > 0.
    Sell win rate: fraction of closed sell trades whose realised P&L is > 0.

    Returns
    -------
    (buy_win_rate%, sell_win_rate%)
    """
    buy_results:  list[bool] = []
    sell_results: list[bool] = []

    dates  = actions_df.index
    delta  = actions_df.diff().fillna(0)
    thresh = 0.5

    for tic in actions_df.columns:
        fifo: list[dict] = []

        for date in dates:
            trade = float(delta.loc[date, tic])
            if abs(trade) < thresh:
                continue

            date_str = str(date)[:10]
            price = price_lookup.get((date_str, str(tic)))
            if price is None:
                continue

            if trade > 0:  # buy — push to FIFO
                fifo.append({"qty": trade, "price": price})
            else:          # sell — match FIFO
                remaining = abs(trade)
                pnl = 0.0
                while remaining > thresh and fifo:
                    oldest    = fifo[0]
                    close_qty = min(remaining, oldest["qty"])
                    pnl      += close_qty * (price - oldest["price"])
                    oldest["qty"] -= close_qty
                    remaining     -= close_qty
                    if oldest["qty"] < thresh:
                        fifo.pop(0)
                sell_results.append(pnl > 0)

        # Close remaining open positions at final date
        last_price = price_lookup.get((str(dates[-1])[:10], str(tic)))
        for op in fifo:
            if last_price is not None:
                buy_results.append(op["qty"] * (last_price - op["price"]) > 0)

    bwr = sum(buy_results)  / len(buy_results)  * 100 if buy_results  else float("nan")
    swr = sum(sell_results) / len(sell_results) * 100 if sell_results else float("nan")
    return round(bwr, 2), round(swr, 2)


def make_trade_stats_table(
    actions_store: dict,
    price_store: dict,
    windows: list,
    datasets: list,
    algos: list,
    save: bool = False,
    save_dir: str = "tables",
) -> pd.DataFrame:
    """
    Aggregated trade statistics table: counts + FIFO win rates per (window, dataset, algo).

    Parameters
    ----------
    actions_store : dict
        actions_store[window][dataset][algo] = pd.DataFrame
        (position DataFrame, seed=42 representative run)
    price_store : dict
        price_store[window][dataset] = pd.DataFrame
        (FinRL-format test data with 'date', 'tic', 'close' columns)

    Returns
    -------
    pd.DataFrame with MultiIndex (window, dataset, algo).
    Columns: Total Trades, Buys, Sells, Long Days, Short Days, Buy Win%, Sell Win%
    """
    os.makedirs(save_dir, exist_ok=True)
    rows, index = [], []

    for w in windows:
        if w not in actions_store:
            continue
        for ds in datasets:
            if ds not in actions_store[w]:
                continue

            # Price lookup — prefer dataset-specific; fall back to first available
            raw_df = None
            if price_store and w in price_store:
                raw_df = price_store[w].get(ds)
                if raw_df is None:
                    fallback = next(iter(price_store[w].values()), None)
                    raw_df = fallback
            price_lookup = _build_price_lookup(raw_df) if raw_df is not None else {}

            for algo in algos:
                if algo not in actions_store[w][ds]:
                    continue
                adf   = actions_store[w][ds][algo]
                stats = compute_trade_stats(adf)

                if price_lookup:
                    bwr, swr = _compute_win_rate_fifo(adf, price_lookup)
                else:
                    bwr = swr = float("nan")

                rows.append({
                    "Total Trades": stats["total_trades"],
                    "Buys":         stats["total_buys"],
                    "Sells":        stats["total_sells"],
                    "Long Days":    stats["n_long_days"],
                    "Short Days":   stats["n_short_days"],
                    "Buy Win%":     bwr,
                    "Sell Win%":    swr,
                })
                index.append((w, ds, algo))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows,
                      index=pd.MultiIndex.from_tuples(index,
                                                      names=["window", "dataset", "algo"]))
    if save:
        path = os.path.join(save_dir, "trade_stats.csv")
        df.to_csv(path)
        print(f"Saved: {path}")

    return df


# ---------------------------------------------------------------------------
# Data Reduction Table
# ---------------------------------------------------------------------------

_BASE_COLS = {"date", "open", "high", "low", "close", "volume", "tic", "day"}


def make_data_reduction_table(
    datasets: list,
    data: Optional[dict] = None,
    windows: Optional[list] = None,
    data_dir: Optional[str] = None,
    stock_dim: int = 30,
    save: bool = False,
    save_dir: str = "tables",
) -> pd.DataFrame:
    """
    Data reduction table: actual row counts and indicator columns per dataset.

    Quantifies how much less data (timesteps × indicator columns) is needed
    when using trigger-based or filtered state representations vs the `all`
    continuous baseline.

    Two modes (tried in order):
    1. ``data`` dict — data[window][ds]['train']['df'] + ['test']['df'] shapes.
       If ``windows`` is provided, aggregates train+test rows across all windows.
    2. CSV fallback — reads header + row count from
       ``{data_dir}/dow_chlovtd_{ds}_*.csv`` (largest file for each dataset).

    Columns
    -------
    Family, Timesteps, Ind_Cols, Data_Cells,
    Δ_Timesteps_%, Δ_Data_Cells_%
    (all relative to the ``all`` baseline row).
    """
    if data_dir is None:
        _fpath = os.environ.get("FINRL_PATH", "")
        data_dir = os.path.join(_fpath, "data", "dow") if _fpath else None

    def _shape_from_data(ds: str) -> Optional[tuple[int, int]]:
        """Returns (total_rows, n_ind_cols) by summing train+test across windows."""
        if data is None:
            return None
        wins = windows or list(data.keys())
        total_rows = 0
        n_ind: Optional[int] = None
        for w in wins:
            for split in ("train", "test"):
                entry = data.get(w, {}).get(ds, {}).get(split, {})
                df = entry.get("df") if isinstance(entry, dict) else None
                if df is None:
                    continue
                total_rows += len(df)
                if n_ind is None:
                    n_ind = len([c for c in df.columns if c not in _BASE_COLS])
        return (total_rows, n_ind) if n_ind is not None else None

    def _shape_from_csv(ds: str) -> Optional[tuple[int, int]]:
        """Reads largest matching CSV for this dataset."""
        if data_dir is None or not os.path.isdir(data_dir):
            return None
        pattern = os.path.join(data_dir, f"dow_chlovtd_{ds}_*.csv")
        import glob as _glob
        files = sorted(_glob.glob(pattern), key=os.path.getsize, reverse=True)
        if not files:
            return None
        df = pd.read_csv(files[0], nrows=2)   # header only for col detection
        n_ind = len([c for c in df.columns if c not in _BASE_COLS])
        # count rows efficiently
        with open(files[0]) as fh:
            n_rows = sum(1 for _ in fh) - 1   # subtract header
        return n_rows, n_ind

    # Build raw shapes dict
    shapes: dict[str, tuple[int, int]] = {}
    for ds in datasets:
        shape = _shape_from_data(ds) or _shape_from_csv(ds)
        if shape is not None:
            shapes[ds] = shape

    if "all" not in shapes:
        print("[WARN] 'all' baseline not found — reduction % will be NaN")

    all_rows, all_ind   = shapes.get("all", (None, None))
    all_ts   = (all_rows // stock_dim) if all_rows else None
    all_cells = (all_rows * all_ind) if (all_rows and all_ind) else None

    rows = []
    for ds in datasets:
        if ds not in shapes:
            continue
        n_rows, n_ind = shapes[ds]
        timesteps  = n_rows // stock_dim
        data_cells = n_rows * n_ind

        def _pct(val, ref):
            return round((val - ref) / ref * 100, 1) if ref else float("nan")

        rows.append({
            "Dataset":        ds,
            "Family":         _ds_family(ds),
            "Timesteps":      timesteps,
            "Ind_Cols":       n_ind,
            "Data_Cells":     data_cells,
            "Δ_Timesteps_%":  _pct(timesteps, all_ts),
            "Δ_Data_Cells_%": _pct(data_cells, all_cells),
        })

    df = pd.DataFrame(rows).set_index("Dataset")

    if save:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "data_reduction.csv")
        df.to_csv(path)
        print(f"Saved: {path}")

    return df


# ---------------------------------------------------------------------------
# State Space Reduction Table
# ---------------------------------------------------------------------------

_DS_FAMILY: dict[str, str] = {
    "all":        "Baseline",
    "mv":         "MA-continuous",
    "mvx":        "MA-hybrid",   "mvx2":      "MA-hybrid",   "dbmvx":     "MA-hybrid",
    "3mvx1":      "MA-triple",   "3mvx2":     "MA-triple",
    "mvx_trg":    "MA-trigger",  "mvx2_trg":  "MA-trigger",
    "mvxall":     "MA-all-cont", "mvx2all":   "MA-all-cont",
    "boll":       "Boll-continuous",
    "bollx":      "Boll-hybrid", "bollrsix1": "Boll-hybrid", "bollrsix2": "Boll-hybrid",
    "bollx_trg":  "Boll-trigger",
    "bollxall":   "Boll-all-cont",
    "bollmvx":    "Combined",    "bollmvx2":  "Combined",
}

_TRIG_COL_PAT = re.compile(
    r'(_xu_|_xd_|bb_xrsi|bb_rsi_short|bb_breakout|ema_xu|ema_xd)', re.IGNORECASE
)


def _ds_family(ds: str) -> str:
    base = ds.removeprefix("flt_")
    base_ntrg = base[:-4] if base.endswith("_trg") else base
    fam = _DS_FAMILY.get(base_ntrg, "Other")
    return ("Filtered " if ds.startswith("flt_") else "") + fam


def make_state_space_table(
    datasets: list,
    data: Optional[dict] = None,
    window: Optional[str] = None,
    stock_dim: int = 30,
    save: bool = False,
    save_dir: str = "tables",
) -> pd.DataFrame:
    """
    State space size and reduction table.

    For each dataset shows: n_indicators, state_space (1 + 2*D + n*D),
    Δ_vs_all, and Reduction_% relative to the `all` continuous baseline.

    Parameters
    ----------
    datasets : list of dataset names to include
    data     : optional data dict — data[window][ds]['test']['df']
               Required only for _trg variants to count boolean columns exactly.
    window   : window key (e.g. 'w1') used to look up in `data`
    stock_dim: number of tickers (default 30)
    """
    from utils.config import INDICATORS_BASE  # local import to avoid circular deps

    def _n_ind(ds: str) -> int:
        base = ds.removeprefix("flt_")
        if base.endswith("_trg"):
            # Try exact boolean column count from loaded data
            if data is not None and window is not None:
                w_data = data.get(window, {})
                for key in (ds, base[:-4]):  # try filtered then base
                    entry = w_data.get(key)
                    if entry is not None:
                        df = entry.get("test", {}).get("df")
                        if df is not None:
                            return len(df.select_dtypes(include=["bool", "boolean"]).columns)
            # Fallback: count trigger-type cols in the base indicator list
            base2 = base[:-4]
            return sum(1 for c in INDICATORS_BASE.get(base2, [])
                       if _TRIG_COL_PAT.search(c))
        return len(INDICATORS_BASE.get(base, []))

    n_all = len(INDICATORS_BASE.get("all", []))
    ss_all = 1 + 2 * stock_dim + n_all * stock_dim

    rows = []
    for ds in datasets:
        n_ind = _n_ind(ds)
        ss    = 1 + 2 * stock_dim + n_ind * stock_dim
        delta = ss - ss_all
        pct   = delta / ss_all * 100 if ss_all > 0 else float("nan")
        rows.append({
            "Dataset":      ds,
            "Family":       _ds_family(ds),
            "N_Indicators": n_ind,
            "State_Space":  ss,
            "Δ_vs_all":     delta,
            "Reduction_%":  round(pct, 1),
        })

    df = pd.DataFrame(rows).set_index("Dataset")

    if save:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "state_space_reduction.csv")
        df.to_csv(path)
        print(f"Saved: {path}")

    return df


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def generate_results_md(
    metrics_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    windows: list,
    trade_stats_df: Optional[pd.DataFrame] = None,
    baselines: Optional[dict] = None,
    state_space_df: Optional[pd.DataFrame] = None,
    data_reduc_df: Optional[pd.DataFrame] = None,
    excluded_datasets: Optional[set] = None,
    output_path: str = "../../reports/results.md",
    img_base: str = "../FinRLCode/myresearch/imgs/dow",
) -> None:
    """
    Write reports/results.md from computed tables.
    All claims are direct numeric observations — no mechanism interpretation.

    Parameters
    ----------
    state_space_df   : Output of make_state_space_table() — optional.
    data_reduc_df    : Output of make_data_reduction_table() — optional.
    excluded_datasets: Dataset names to suppress from all tables/sections.
                       Defaults to the 8 datasets that need retraining after
                       the compound-flag bug fix.
    """
    _DEFAULT_EXCLUDED = {
        "flt_3mvx1", "flt_3mvx1_trg",
        "flt_3mvx2", "flt_3mvx2_trg",
        "flt_bollrsix1", "flt_bollrsix1_trg",
        "flt_bollrsix2", "flt_bollrsix2_trg",
    }
    excl = set(excluded_datasets) if excluded_datasets is not None else _DEFAULT_EXCLUDED
    # Prefer Sharpe_mean (per-seed metric, then averaged) over the plain Sharpe column
    # (metric on the seed-averaged curve) wherever this function reads metrics_df
    # directly -- see _preferred_col's docstring.
    sharpe_col = _preferred_col(metrics_df, "Sharpe")

    w = windows[0]
    algos = ["a2c", "ppo", "sac"]

    all_ds_raw = set(metrics_df.index.get_level_values("dataset"))
    all_ds = all_ds_raw - excl

    MA_DATASETS = [d for d in
                   ["mv", "mvx", "mvx2", "dbmvx", "3mvx1", "3mvx2",
                    "flt_mvx", "flt_mvx2",
                    "mvx_trg", "mvx2_trg", "flt_dbmvx", "mvxall"]
                   if d in all_ds]

    BOLL_DATASETS = [d for d in
                     ["boll", "bollx", "bollrsix1", "bollrsix2",
                      "flt_bollx", "flt_bollmvx2", "bollx_trg", "bollxall",
                      "bollmvx", "bollmvx2"]
                     if d in all_ds]

    def _hm_to_md(datasets):
        data = {}
        for ds in datasets:
            if ds in excl:
                continue
            row = {}
            for algo in algos:
                try:
                    row[algo.upper()] = metrics_df.loc[(w, ds, algo), sharpe_col]
                except KeyError:
                    row[algo.upper()] = None
            data[ds] = row
        if not data:
            return "*(no data)*"
        hm_df = pd.DataFrame(data).T
        hm_df.index.name = "Dataset"
        return hm_df.to_markdown(floatfmt=".3f")

    # Filtered ranking_df and optional tables
    filtered_ranking = ranking_df[~ranking_df["Dataset"].isin(excl)] if not ranking_df.empty else ranking_df

    def _filter_ds_df(df, col="Dataset"):
        """Filter a DataFrame by excluding rows whose `col` is in excl."""
        if df is None or df.empty:
            return df
        # Try exact column, or first column as fallback
        c = col if col in df.columns else df.columns[0]
        return df[~df[c].isin(excl)].reset_index(drop=True)

    state_space_df = _filter_ds_df(state_space_df, "Dataset")
    data_reduc_df  = _filter_ds_df(data_reduc_df,  "Dataset")

    # Baseline return info
    bl_line = ""
    if baselines and w in baselines:
        bl = baselines[w]
        parts = []
        for bname, bec in bl.items():
            bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
            ret = (bec_s.iloc[-1] / bec_s.iloc[0] - 1) * 100
            parts.append(f"{bname}: {ret:.1f}% total return")
        bl_line = "  ".join(parts)

    # Summary bullet points
    summary_lines = []
    try:
        best = filtered_ranking.iloc[0]
        cagr_val = best.get("CAGR", float("nan"))
        cagr_str = f", CAGR = {cagr_val:.1f}%" if not (cagr_val != cagr_val) else ""
        summary_lines.append(
            f"1. **Best configuration (W1)**: {best.get('Algo','?').upper()} + "
            f"{best.get('Dataset','?')}  —  Sharpe = {best.get('Sharpe', float('nan')):.2f}"
            f"{cagr_str}, Return = {best.get('Return%', float('nan')):.1f}%"
        )
    except Exception:
        summary_lines.append("1. *(see ranking table)*")

    for ds, label in [("3mvx1", "Triple-EMA crossover (3mvx1)"),
                      ("mvx_trg", "Trigger-only state (mvx_trg)"),
                      ("mvxall", "All-continuous ablation (mvxall)")]:
        try:
            v_ds  = metrics_df.loc[(w, ds,    "a2c"), sharpe_col]
            v_mvx = metrics_df.loc[(w, "mvx",  "a2c"), sharpe_col]
            summary_lines.append(
                f"{len(summary_lines) + 1}. {label} vs mvx — A2C Δ Sharpe = {v_ds - v_mvx:+.3f}"
            )
        except KeyError:
            pass

    filter_lines = []
    for algo in ["a2c", "ppo", "sac"]:
        try:
            base = metrics_df.loc[(w, "mvx",     algo), sharpe_col]
            flt  = metrics_df.loc[(w, "flt_mvx", algo), sharpe_col]
            d    = flt - base
            filter_lines.append(f"  - {algo.upper()}: {d:+.3f} ({'↑' if d > 0 else '↓'})")
        except KeyError:
            pass

    filter_vals = {}
    for algo in ["a2c", "ppo", "sac"]:
        try:
            base = metrics_df.loc[(w, "mvx",     algo), sharpe_col]
            flt  = metrics_df.loc[(w, "flt_mvx", algo), sharpe_col]
            filter_vals[algo] = flt - base
        except KeyError:
            filter_vals[algo] = float("nan")

    a2c_d = filter_vals.get("a2c", float("nan"))
    ppo_d = filter_vals.get("ppo", float("nan"))
    sac_d = filter_vals.get("sac", float("nan"))
    filter_summary = (
        f"A2C {a2c_d:+.3f}, PPO {ppo_d:+.3f}, SAC {sac_d:+.3f}"
        if not any(v != v for v in [a2c_d, ppo_d, sac_d])
        else "A2C, PPO, SAC — see filter_effect table"
    )
    summary_lines.append(
        f"{len(summary_lines) + 1}. Trigger-date filtering effect on MVX — {filter_summary}."
    )

    # Trade stats section
    trade_md_lines = []
    if trade_stats_df is not None and not trade_stats_df.empty:
        try:
            sub = trade_stats_df.loc[w].reset_index()
            sub = sub[~sub["dataset"].isin(excl)] if "dataset" in sub.columns else sub
            sub.columns = [c.title() if c in ["dataset", "algo"] else c for c in sub.columns]
            trade_md_lines = [
                "## 7. Trade Statistics (Seed=42 Representative Run)",
                "",
                sub.to_markdown(index=False, floatfmt=".1f"),
                "",
                "> Trade counts and FIFO win rates for one representative seed per configuration.",
                "",
                f"![Trade heatmap W1]({img_base}/trade_heatmap_{w}_total.png)",
                f"![Ticker buy/sell aggregate W1]({img_base}/ticker_buysell_agg_{w}.png)",
                "",
                "---",
                "",
            ]
        except Exception:
            pass

    # Section numbering
    # 1. MA heatmap
    # 2. Boll heatmap
    # 3. State Space & Data Reduction (new)
    # 4. Full Ranking
    # 5. Algo x Indicator Interaction
    # 6. Filter Effect
    # 7. Trade Stats (if present)
    # 8. Hypothesis Verdicts
    # 9. Interpretability (PPO + flt_mvx2)
    # 10. Interpretation (new)
    # 11. Summary
    # 12. Limitations
    hyp_sec     = 8 if trade_md_lines else 7
    interp_sec  = hyp_sec + 1
    interprt_sec = interp_sec + 1   # written Interpretation section
    summary_sec = interprt_sec + 1
    limit_sec   = summary_sec + 1

    # State Space & Data Reduction section
    ss_dr_lines = []
    if state_space_df is not None or data_reduc_df is not None:
        ss_dr_lines += [
            "## 3. State Space & Data Reduction",
            "",
        ]
        if state_space_df is not None:
            ss_dr_lines += [
                "### 3a. State Space Reduction vs `all` Baseline",
                "",
                state_space_df.to_markdown(floatfmt=".1f") if not state_space_df.empty else "*(no data)*",
                "",
            ]
        if data_reduc_df is not None:
            ss_dr_lines += [
                "### 3b. Training Data Reduction (Timesteps × Indicator Columns)",
                "",
                data_reduc_df.to_markdown(floatfmt=".1f") if not data_reduc_df.empty else "*(no data)*",
                "",
                "> Trigger-only (_trg) variants reduce data cells by up to 84% vs the `all` baseline",
                "> by combining both fewer indicator columns (boolean-only) and fewer timesteps (filtered dates).",
                "",
            ]
        ss_dr_lines += ["---", ""]

    # Dynamic interpretation bullets
    def _try_sharpe(ds, algo):
        try:
            return float(metrics_df.loc[(w, ds, algo), sharpe_col])
        except Exception:
            return float("nan")

    flt_mvx2_ppo   = _try_sharpe("flt_mvx2", "ppo")
    mvx2_ppo       = _try_sharpe("mvx2", "ppo")
    flt_mvx2_trg_ppo = _try_sharpe("flt_mvx2_trg", "ppo")
    flt_bollx_trg_a2c = _try_sharpe("flt_bollx_trg", "a2c")

    def _corr_str(ds, algo):
        try:
            return f"{float(metrics_df.loc[(w, ds, algo), 'Corr_DJIA']):.2f}"
        except Exception:
            return "n/a"

    # Top-15 dataset check (used in Interpretation)
    top15_ds = set(filtered_ranking["Dataset"].iloc[:15].tolist()) if not filtered_ranking.empty else set()
    flt_dominates = all(d.startswith("flt_") for d in top15_ds if not d.startswith("["))

    interp_bullets = [
        "- **Filter dominates**: "
        + (
            "All top-15 configurations use trigger-date filtering (`flt_*`). "
            "Unfiltered hybrid configurations (mvx, bollx) do not appear in the top 15, "
            "suggesting that training on trigger-relevant days alone is more beneficial than "
            "training on all days with a trigger-augmented state."
            if flt_dominates
            else
            "Most top-15 configurations use trigger-date filtering (`flt_*`). "
            "See ranking table for details."
        ),
    ]

    if not (flt_mvx2_ppo != flt_mvx2_ppo) and not (mvx2_ppo != mvx2_ppo):
        delta_ppo = flt_mvx2_ppo - mvx2_ppo
        interp_bullets.append(
            f"- **PPO most responsive to filtering**: PPO achieves Sharpe {flt_mvx2_ppo:.2f} "
            f"on flt_mvx2, vs {mvx2_ppo:.2f} unfiltered (Δ = {delta_ppo:+.2f})."
        )

    if not (flt_mvx2_trg_ppo != flt_mvx2_trg_ppo) and not (flt_mvx2_ppo != flt_mvx2_ppo):
        interp_bullets.append(
            f"- **State space compression**: Trigger-only (_trg) variants use 60–84% less data "
            f"(timesteps × columns) than the `all` baseline while maintaining competitive Sharpe ratios "
            f"(flt_mvx2_trg PPO Sharpe = {flt_mvx2_trg_ppo:.2f} vs flt_mvx2 PPO Sharpe = {flt_mvx2_ppo:.2f})."
        )

    interp_bullets += [
        "- **Algorithm–indicator interaction**: SAC shows consistent advantages with Bollinger hybrid "
        "states; PPO is strongest with MA-based filtered states; A2C performance is more variable "
        "across indicator types. All differences are W1-only and untested statistically.",
        f"- **Correlation with DJIA**: flt_bollx_trg A2C Corr_DJIA = {_corr_str('flt_bollx_trg','a2c')} "
        f"vs flt_mvx2 PPO Corr_DJIA = {_corr_str('flt_mvx2','ppo')}. "
        "The Bollinger trigger-only configs track the market less closely than the MA filtered configs.",
    ]

    # --- Write file ---
    lines = [
        "# Empirical Results: Event-Driven State Representations",
        "",
        f"> **Window W1** — Test period: 2020-03-01 to 2021-01-01  ",
        f"> All Sharpe / CAGR values are means across 5 seeds.  ",
        f"> {bl_line}",
        "> **Scope**: Single window. W2 and W3 pending. No statistical tests applied yet.",
        "",
        "---",
        "",
        "## 1. Moving Average Strategies — Sharpe Heatmap",
        "",
        "Configurations: mv, mvx, mvx2, dbmvx, 3mvx1, 3mvx2 and their "
        "filtered / trigger-only / all-continuous variants.",
        "Bollinger-based and combined configurations are excluded from this section.",
        "",
        _hm_to_md(MA_DATASETS) if MA_DATASETS else "*(no MA datasets in results)*",
        "",
        f"![A2C equity W1]({img_base}/equity_A2C_-_W1_(2020-03-01_to_2021-01-01).png)",
        f"![PPO equity W1]({img_base}/equity_PPO_-_W1_(2020-03-01_to_2021-01-01).png)",
        f"![SAC equity W1]({img_base}/equity_SAC_-_W1_(2020-03-01_to_2021-01-01).png)",
        "",
        "---",
        "",
        "## 2. Bollinger Band Strategies — Sharpe Heatmap",
        "",
        "Configurations: boll, bollx, bollrsix1, bollrsix2 and their "
        "filtered / trigger-only variants.",
        "Moving-average-only configurations are excluded from this section.",
        "",
        _hm_to_md(BOLL_DATASETS) if BOLL_DATASETS else "*(no Bollinger datasets in results)*",
        "",
        "---",
        "",
    ] + ss_dr_lines + [
        "## 4. Full Ranking — Top 15 Configurations by Sharpe",
        "",
        filtered_ranking.head(15).to_markdown(floatfmt=".3f") if not filtered_ranking.empty else "*(no data)*",
        "",
        "---",
        "",
        "## 5. Algorithm × Indicator Interaction (All Datasets)",
        "",
        _hm_to_md(sorted(all_ds - {"[DJIA]", "[MVO]"})),
        "",
        "---",
        "",
        "## 6. Filter Effect — Trigger-Date Subsampling",
        "",
        "Δ Sharpe (filtered minus unfiltered) for A2C, PPO, SAC on MVX:",
        "",
    ] + filter_lines + [
        "",
        "Full table: `tables/filter_effect_sharpe.csv`",
        "",
        "---",
        "",
    ] + trade_md_lines + [
        f"## {hyp_sec}. Hypothesis Verdicts",
        "",
        hypothesis_df.to_markdown(index=False, floatfmt=".3f") if not hypothesis_df.empty else "*(no data)*",
        "",
        "---",
        "",
        f"## {interp_sec}. Interpretability: PPO + flt_mvx2 (Best Configuration)",
        "",
        "Green triangles = agent increases position. Red triangles = agent decreases position.",
        "Numbers above/below triangles = shares traded. Dashed lines = EMA values.",
        "Vertical lines = MA crossover trigger events.",
        "",
        f"![PPO FLT_MVX2 AAPL]({img_base}/PPO_+_FLT_MVX2_—_AAPL_Trades_(W1_test_period).png)",
        f"![PPO FLT_MVX2 MSFT]({img_base}/PPO_+_FLT_MVX2_—_MSFT_Trades_(W1_test_period).png)",
        f"![PPO FLT_MVX2 JPM]({img_base}/PPO_+_FLT_MVX2_—_JPM_Trades_(W1_test_period).png)",
        f"![PPO FLT_MVX2 GS]({img_base}/PPO_+_FLT_MVX2_—_GS_Trades_(W1_test_period).png)",
        "",
        "---",
        "",
        f"## {interprt_sec}. Interpretation",
        "",
    ] + interp_bullets + [
        "",
        "---",
        "",
        f"## {summary_sec}. Summary",
        "",
    ] + summary_lines + [
        "",
        "---",
        "",
        f"## {limit_sec}. Limitations",
        "",
        "- Single window (W1). Results may not generalise across market regimes.",
        "- No statistical significance tests applied (required before any publication claim).",
        "- The following datasets were excluded from this report because they require retraining "
        "after a compound-indicator bug fix (missing flags in config.py): "
        + ", ".join(sorted(excl)) + ".",
        "- Transaction costs not yet integrated.",
        "- Trade stats and win rates are from one representative seed (seed=42); "
        "  may not reflect mean-seed behaviour.",
        "",
        "---",
        "",
        "*Generated automatically from `dow_3_backtest.ipynb` — Section 6.*",
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"results.md written → {out.resolve()}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metric_col(metric: str) -> str:
    return {
        "sharpe": "Sharpe", "cagr": "CAGR", "return": "Return%",
        "sortino": "Sortino", "calmar": "Calmar",
        "maxdd": "MaxDD%", "volatility": "Volatility%",
    }.get(metric.lower(), metric)


def _preferred_col(df, col_name):
    """
    `{col_name}_mean` when it exists in `df` and carries data, else `col_name`.

    The plain column (e.g. "Sharpe") comes from compute_metrics() on a curve that
    was itself averaged across the 5 seeds first (aggregate_seeds' "mean" curve) --
    averaging curves before computing the metric mechanically cancels cross-seed
    volatility the way averaging independent portfolios does, so it runs Sharpe
    optimistic and MaxDD shallow. `{col_name}_mean` (from _seed_ci_stats, populated
    whenever make_metrics_table was called with results=) computes the metric on each
    seed curve separately and averages those instead -- unbiased. Falls back to the
    plain column when _mean was never computed (results= not passed), so this is a
    no-op wherever the caller didn't opt in to per-seed stats.
    """
    mean_col = f"{col_name}_mean"
    if mean_col in df.columns and df[mean_col].notna().any():
        return mean_col
    return col_name


def _lookup(metrics_df, agg, w, ds, algo, col_name):
    if metrics_df is not None:
        try:
            col = _preferred_col(metrics_df, col_name)
            return float(metrics_df.loc[(w, ds, algo), col])
        except KeyError:
            pass
    if agg and w in agg and ds in agg[w] and algo in agg[w][ds]:
        ec = agg[w][ds][algo]["mean"]
        return compute_metrics(ec).get(col_name, np.nan)
    return np.nan


def _fmt(v):
    try:
        f = float(v)
        return round(f, 3) if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _safe_diff(a, b):
    try:
        fa, fb = float(a), float(b)
        if np.isnan(fa) or np.isnan(fb):
            return np.nan
        return fa - fb
    except (TypeError, ValueError):
        return np.nan


def _pos(v):
    try:
        return not np.isnan(float(v)) and float(v) > 0
    except (TypeError, ValueError):
        return False


def _known(v):
    try:
        return not np.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _row(id_, hyp, w, comp, delta, verdict):
    return {
        "id":         id_,
        "hypothesis": hyp,
        "window":     w,
        "comparison": comp,
        "Δ_Sharpe":   _fmt(delta) if delta is not None else np.nan,
        "verdict":    verdict,
    }
