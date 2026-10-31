import os
from utils.common import filter_date_range
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import re
from typing import Optional


def plot_bollrsix(df, tic='AAPL', start='2025-01-01', end='2026-01-01',
                  long_col='bb_xrsi_long', short_col='bb_rsi_short',
                  axes=None, save=False, save_dir='imgs'):
    d = filter_date_range(df, start_date=start, end_date=end, tic=tic, stock_df=False).copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    long_days  = d[d[long_col]]
    short_days = d[d[short_col]]

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'{tic}  |  {long_col} / {short_col}  ({start} – {end})', fontsize=13)

    ax1, ax2 = axes
    ax1.plot(d.index, d['close'],   color='steelblue', lw=1.2, label='close')
    ax1.plot(d.index, d['boll_ub'], color='grey',      lw=0.8, ls='--', label='boll_ub')
    ax1.plot(d.index, d['boll_lb'], color='grey',      lw=0.8, ls='--', label='boll_lb')
    ax1.fill_between(d.index, d['boll_lb'], d['boll_ub'], alpha=0.08, color='grey')
    ax1.scatter(long_days.index,  long_days['close'],  marker='^', color='green', s=80, zorder=5, label=f'{long_col} ↑')
    ax1.scatter(short_days.index, short_days['close'], marker='v', color='red',   s=80, zorder=5, label=f'{short_col} ↓')
    ax1.set_ylabel('Price'); ax1.legend(loc='upper left', fontsize=8)

    ax2.plot(d.index, d['rsi_14'], color='purple', lw=1.0, label='RSI-14')
    ax2.axhline(65, color='red',   lw=0.8, ls='--', label='65 (overbought)')
    ax2.axhline(35, color='green', lw=0.8, ls='--', label='35 (oversold)')
    ax2.fill_between(d.index, 35, d['rsi_14'].clip(upper=35), alpha=0.15, color='green')
    ax2.fill_between(d.index, 65, d['rsi_14'].clip(lower=65), alpha=0.15, color='red')
    ax2.scatter(long_days.index,  long_days['rsi_14'],  marker='^', color='green', s=60, zorder=5)
    ax2.scatter(short_days.index, short_days['rsi_14'], marker='v', color='red',   s=60, zorder=5)
    ax2.set_ylabel('RSI'); ax2.set_ylim(0, 100); ax2.legend(loc='upper left', fontsize=8)

    if standalone:
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"dow_{tic}_{long_col}_{start}_to_{end}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', dpi=150)
            print(f"Saved → {os.path.join(save_dir, fname)}")
        plt.show()


def plot_3mvx(df, tic='AAPL', start='2025-01-01', end='2026-01-01',
              up_col='ema_xu_5_10_50', down_col='ema_xd_5_10_50',
              ax=None, save=False, save_dir='imgs'):
    d = filter_date_range(df, start_date=start, end_date=end, tic=tic, stock_df=False).copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    up_days   = d[d[up_col]]
    down_days = d[d[down_col]]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f'{tic}  |  {up_col} / {down_col}  ({start} – {end})', fontsize=12)

    ax.plot(d.index, d['close'], color='steelblue', lw=1.2, label='close', zorder=1)
    ax.plot(d.index, d['close_5_ema'],  color='orange', lw=0.9, alpha=0.7, label='EMA-5')
    ax.plot(d.index, d['close_10_ema'], color='green',  lw=0.9, alpha=0.7, label='EMA-10')
    ax.plot(d.index, d['close_50_ema'], color='red',    lw=0.9, alpha=0.7, label='EMA-50')
    ax.scatter(up_days.index,   up_days['close'],   marker='^', color='green', s=100, zorder=5, label=f'{up_col} ↑')
    ax.scatter(down_days.index, down_days['close'], marker='v', color='red',   s=100, zorder=5, label=f'{down_col} ↓')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"dow_{tic}_{up_col}_{start}_to_{end}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', dpi=150)
            print(f"Saved → {os.path.join(save_dir, fname)}")
        plt.show()


def plot_mvx(df, tic='AAPL', start='2025-01-01', end='2026-01-01',
             up_col='close_5_ema_xu_close_20_sma',
             down_col='close_5_ema_xd_close_20_sma',
             ax=None, save=False, save_dir='imgs'):
    d = filter_date_range(df, start_date=start, end_date=end, tic=tic, stock_df=False).copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    up_days   = d[d[up_col]]
    down_days = d[d[down_col]]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f'{tic}  |  {up_col} / {down_col}  ({start} – {end})', fontsize=12)

    ax.plot(d.index, d['close'],       color='steelblue', lw=1.2, label='close', zorder=1)
    ax.plot(d.index, d['close_5_ema'], color='orange', lw=0.9, alpha=0.7, label='EMA-5')
    ax.plot(d.index, d['close_20_sma'], color='green',  lw=0.9, alpha=0.7, label='SMA-20')
    ax.scatter(up_days.index,   up_days['close'],   marker='^', color='green', s=100, zorder=5, label=f'{up_col} ↑')
    ax.scatter(down_days.index, down_days['close'], marker='v', color='red',   s=100, zorder=5, label=f'{down_col} ↓')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"dow_{tic}_{up_col}_{start}_to_{end}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', dpi=150)
            print(f"Saved → {os.path.join(save_dir, fname)}")
        plt.show()


def plot_bollx(df, tic='AAPL', start='2025-01-01', end='2026-01-01',
               long_col='close_xu_boll_lb', short_col='close_xu_boll_ub',
               ax=None, save=False, save_dir='imgs'):
    d = filter_date_range(df, start_date=start, end_date=end, tic=tic, stock_df=False).copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    long_days  = d[d[long_col]]
    short_days = d[d[short_col]]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f'{tic}  |  {long_col} / {short_col}  ({start} – {end})', fontsize=12)

    ax.plot(d.index, d['close'],   color='steelblue', lw=1.2, label='close', zorder=1)
    ax.plot(d.index, d['boll_ub'], color='grey', lw=0.8, ls='--', label='boll_ub')
    ax.plot(d.index, d['boll_lb'], color='grey', lw=0.8, ls='--', label='boll_lb')
    ax.fill_between(d.index, d['boll_lb'], d['boll_ub'], alpha=0.08, color='grey')
    ax.scatter(long_days.index,  long_days['close'],  marker='^', color='green', s=100, zorder=5, label=f'{long_col} ↑')
    ax.scatter(short_days.index, short_days['close'], marker='v', color='red',   s=100, zorder=5, label=f'{short_col} ↓')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"dow_{tic}_{long_col}_{start}_to_{end}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', dpi=150)
            print(f"Saved → {os.path.join(save_dir, fname)}")
        plt.show()


def plot_equity_curves_with_std(agg_window, datasets, algos, title="",
                                baselines=None, date_map=None, save=False, save_dir='imgs'):
    """
    Plot equity curves with ±1σ bands.

    date_map: dict[ds] → pd.DatetimeIndex with actual dates for each dataset's equity curve.
              If provided, uses these dates instead of baseline dates for proper alignment.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = cm.tab20.colors

    # Extract x-axis dates from first baseline (all baselines cover same period)
    x_dates = None
    if baselines:
        for bname, bec in baselines.items():
            bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
            x_dates = pd.to_datetime(bec_s.index)
            break

    for i, (ds, algo) in enumerate([(d, a) for d in datasets for a in algos]):
        if ds not in agg_window or algo not in agg_window[ds]:
            continue
        mean_ec = agg_window[ds][algo]["mean"]
        std_ec = agg_window[ds][algo].get("std")
        color = palette[i % len(palette)]

        # Each dataset's own equity-curve index already carries its real dates -- sparse
        # (trigger dates only, spread across the whole window) for flt_* datasets, daily
        # otherwise -- so always prefer parsing THAT index directly, exactly like the
        # pd.to_datetime(...) already used for baselines below. date_map / the baseline's
        # date range are only a fallback for the rare case where mean_ec's own index
        # can't be parsed as dates at all. Truncating a DIFFERENT (denser, daily) series'
        # date range to len(mean_ec) is wrong for sparse datasets: it silently reassigns
        # every point to the wrong date, compressing the curve into the first N calendar
        # days of the window instead of spreading it across the dates the account value
        # was actually computed on.
        try:
            x_plot = pd.to_datetime(mean_ec.index)
        except (ValueError, TypeError):
            if date_map and ds in date_map:
                x_plot = date_map[ds][:len(mean_ec)]
            elif x_dates is not None:
                x_plot = x_dates[:len(mean_ec)]
            else:
                x_plot = mean_ec.index

        ax.plot(x_plot, mean_ec.values, color=color, linewidth=1.2,
                label=f"{ds}/{algo.upper()}", zorder=3)
        if std_ec is not None:
            std_aligned = std_ec.reindex(mean_ec.index)
            ax.fill_between(x_plot, (mean_ec - std_aligned).values, (mean_ec + std_aligned).values,
                           color=color, alpha=0.12, linewidth=0, zorder=2)

    if baselines:
        for bname, bec in baselines.items():
            bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
            x_plot = pd.to_datetime(bec_s.index)
            ax.plot(x_plot, bec_s.values, linestyle="--", linewidth=1.5,
                   label=bname, zorder=4)

    ax.set_ylabel("Account Value ($)")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()
    plt.close(fig)


def plot_bollmvx(df_mvx, df_bollx, tic='AAPL', start='2025-01-01', end='2026-01-01',
                 save=False, save_dir='imgs'):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'{tic}  |  MVX + BOLLX  ({start} – {end})', fontsize=12)

    # Panel 1: MVX
    plot_mvx(df_mvx, tic, start, end, ax=axes[0], save=False)
    axes[0].set_title(f'{tic}  |  MVX')

    # Panel 2: BOLLX
    plot_bollx(df_bollx, tic, start, end, ax=axes[1], save=False)
    axes[1].set_title(f'{tic}  |  BOLLX')

    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"dow_{tic}_bollmvx_{start}_to_{end}.png"
        plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', dpi=150)
        print(f"Saved → {os.path.join(save_dir, fname)}")
    plt.show()


def _detect_ma_cols(df_tic):
    return [c for c in df_tic.columns if any(x in c for x in ['_ema', '_sma'])]


def _detect_trigger_cols(df_tic):
    return [c for c in df_tic.columns if any(x in c for x in ['_xu_', '_xd_'])]


def plot_rl_trades_on_price(
    actions_df,
    price_df,
    tic,
    signal_df=None,
    signal_up_col=None,
    signal_dn_col=None,
    ma_cols=None,
    auto_detect_signals=True,
    auto_detect_ma=True,
    show_ma=True,
    cross_ticker_price_df=None,
    algo=None,
    dataset=None,
    title="",
    save=False,
    save_dir="imgs/dow",
    only_ma_cross=False,
    show_cross_ticker_signals=False,
    show_position_subplot=True,
):
    """
    Plot RL agent trades overlaid on price chart with optional MA lines.

    Parameters
    ----------
    only_ma_cross : bool, default False
        If True, show buy/sell markers ONLY on dates where the selected stock
        has an MA crossover signal (_xu_ or _xd_).
    show_cross_ticker_signals : bool, default False
        If True, draw faint vertical lines on price chart at dates when other
        tickers (from cross_ticker_price_df) have trigger signals.
    show_position_subplot : bool, default True
        If True, show a second subplot displaying cumulative shares held.
        Works independently of other options.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Handle both WIDE format (tickers as columns) and LONG format (single 'action' column)
    if "action" not in actions_df.columns and tic in actions_df.columns:
        # WIDE format detected: extract ticker column and convert to LONG
        actions_df = actions_df[[tic]].reset_index()
        actions_df.columns = ['date', 'action']

    if price_df is not None:
        # Infer date range from price_df to avoid hardcoded defaults
        price_df_copy = price_df.copy()
        if "date" in price_df_copy.columns:
            price_df_copy["date"] = pd.to_datetime(price_df_copy["date"])
            date_col = price_df_copy["date"]
        else:
            date_col = pd.to_datetime(price_df_copy.index)
        date_min = date_col.min()
        date_max = date_col.max()
        tic_data = filter_date_range(price_df, tic=tic, start_date=str(date_min.date()),
                                      end_date=str(date_max.date()), stock_df=False).copy()
        if tic_data is not None:
            tic_data["date"] = pd.to_datetime(tic_data["date"])
            tic_data = tic_data.set_index("date").sort_index()
    else:
        tic_data = None

    if tic_data is None or tic_data.empty:
        print(f"[WARN] No price data for {tic}")
        return

    # Create figure with 1 or 2 subplots depending on show_position_subplot flag
    if show_position_subplot:
        fig, (ax_price, ax_pos) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                               gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax_price = plt.subplots(1, 1, figsize=(14, 6))
        ax_pos = None
    if title:
        fig.suptitle(title, fontsize=12)

    ax_price.plot(tic_data.index, tic_data["close"], color="steelblue", linewidth=1.2, label="close", zorder=1)

    # Detect MA cross dates early (needed for both only_ma_cross and agent trade filtering)
    ma_cross_dates = pd.DatetimeIndex([])
    if only_ma_cross:
        trig_cols = _detect_trigger_cols(tic_data)
        xu_cols = [c for c in trig_cols if '_xu_' in c]
        xd_cols = [c for c in trig_cols if '_xd_' in c]

        if xu_cols or xd_cols:
            # Explicit trigger columns exist
            cross_idx = []
            if xu_cols:
                cross_idx.extend(tic_data[tic_data[xu_cols[0]].astype(bool)].index.tolist())
            if xd_cols:
                cross_idx.extend(tic_data[tic_data[xd_cols[0]].astype(bool)].index.tolist())
            ma_cross_dates = pd.DatetimeIndex(cross_idx).unique()
        else:
            # No explicit trigger columns found.
            # If price_df is pre-filtered (flt_* datasets), all rows ARE trigger days
            print(f"[INFO] No trigger columns found. Assuming all {len(tic_data)} rows are trigger days (pre-filtered dataset)")
            ma_cross_dates = tic_data.index

    # Plot agent trades (buy/sell markers from actions_df)
    # When only_ma_cross=True, filter to trades on MA cross dates only
    if not actions_df.empty and "action" in actions_df.columns and "date" in actions_df.columns:
        actions_df_copy = actions_df.copy()
        actions_df_copy["date"] = pd.to_datetime(actions_df_copy["date"])
        actions_df_copy = actions_df_copy.set_index("date")

        if only_ma_cross:
            # Filter to only dates that have MA cross signals
            actions_df_copy = actions_df_copy[actions_df_copy.index.isin(ma_cross_dates)]

        # Find buy actions (action > 0)
        buys = actions_df_copy[actions_df_copy["action"] > 0]
        for date_idx, row in buys.iterrows():
            if date_idx in tic_data.index:
                ax_price.scatter(date_idx, tic_data.loc[date_idx, "close"],
                               marker="^", color="lime", s=120, zorder=6, alpha=0.8, edgecolors="darkgreen", linewidth=1)

        # Find sell actions (action < 0)
        sells = actions_df_copy[actions_df_copy["action"] < 0]
        for date_idx, row in sells.iterrows():
            if date_idx in tic_data.index:
                ax_price.scatter(date_idx, tic_data.loc[date_idx, "close"],
                               marker="v", color="red", s=120, zorder=6, alpha=0.8, edgecolors="darkred", linewidth=1)

        # Add to legend if we have any trades
        if not buys.empty:
            ax_price.scatter([], [], marker="^", color="lime", s=120, edgecolors="darkgreen", linewidth=1,
                           label=f"Agent Buy ({len(buys)})")
        if not sells.empty:
            ax_price.scatter([], [], marker="v", color="red", s=120, edgecolors="darkred", linewidth=1,
                           label=f"Agent Sell ({len(sells)})")

    if show_ma and auto_detect_ma:
        ma_cols = _detect_ma_cols(tic_data)
        for col in ma_cols[:3]:
            ax_price.plot(tic_data.index, tic_data[col], linewidth=0.8, alpha=0.6)

    # Feature 1: Show MA cross signals on chart (ONLY show if not filtering by trades)
    # When only_ma_cross=True, we're filtering agent trades to MA cross dates,
    # so don't show the MA signal markers separately (agent trades are the markers)
    if False:  # Disabled: MA cross markers are redundant with filtered agent trades
        trig_cols = _detect_trigger_cols(tic_data)
        xu_cols = [c for c in trig_cols if '_xu_' in c]
        xd_cols = [c for c in trig_cols if '_xd_' in c]
        if xu_cols:
            up_days = tic_data[tic_data[xu_cols[0]].astype(bool)]
            if not up_days.empty:
                ax_price.scatter(up_days.index, up_days["close"], marker="^", color="darkgreen", s=80, zorder=4, alpha=0.5, label="MA ↑ signal")
        if xd_cols:
            dn_days = tic_data[tic_data[xd_cols[0]].astype(bool)]
            if not dn_days.empty:
                ax_price.scatter(dn_days.index, dn_days["close"], marker="v", color="darkred", s=80, zorder=4, alpha=0.5, label="MA ↓ signal")

    if signal_df is not None and not signal_df.empty:
        sig_data = filter_date_range(signal_df, tic=tic, stock_df=False).copy()
        if sig_data is not None and not sig_data.empty:
            sig_data["date"] = pd.to_datetime(sig_data["date"])
            sig_data = sig_data.set_index("date").sort_index()

            if auto_detect_signals:
                trigger_cols = _detect_trigger_cols(sig_data)
                if len(trigger_cols) >= 2:
                    signal_up_col, signal_dn_col = trigger_cols[0], trigger_cols[1]

            if signal_up_col and signal_up_col in sig_data.columns:
                up_days = sig_data[sig_data[signal_up_col]]
                ax_price.scatter(up_days.index, tic_data.loc[up_days.index, "close"],
                               marker="^", color="green", s=100, zorder=5, label="Buy signal")

            if signal_dn_col and signal_dn_col in sig_data.columns:
                dn_days = sig_data[sig_data[signal_dn_col]]
                ax_price.scatter(dn_days.index, tic_data.loc[dn_days.index, "close"],
                               marker="v", color="red", s=100, zorder=5, label="Sell signal")

    # Feature 2: Show faint vertical lines for other tickers' trigger signals with ticker names
    if show_cross_ticker_signals and cross_ticker_price_df:
        # Get price range for label positioning
        price_range = tic_data["close"].max() - tic_data["close"].min()
        label_y = tic_data["close"].max() - 0.05 * price_range

        # Counter to show every 3rd ticker name (to avoid crowding)
        ticker_counter = 0

        for other_tic, other_df in cross_ticker_price_df.items():
            if other_df is None or other_df.empty:
                continue
            other_data = other_df.copy()
            if "tic" in other_data.columns:
                other_data = other_data[other_data["tic"] == other_tic]
            if other_data.empty:
                continue
            if "date" in other_data.columns:
                other_data["date"] = pd.to_datetime(other_data["date"])
                other_data = other_data.set_index("date").sort_index()
            trig_cols = _detect_trigger_cols(other_data)
            for col in trig_cols:
                signal_dates = other_data[other_data[col].astype(bool)].index
                for d in signal_dates:
                    ax_price.axvline(d, color="blue", linewidth=0.5, alpha=0.3, linestyle="--", zorder=0)
                    # Show ticker name only every 3rd signal (and only for every 3rd ticker)
                    if ticker_counter % 3 == 0:
                        ax_price.text(d, label_y, other_tic, fontsize=4.5, rotation=90,
                                    va="bottom", ha="right", color="red", alpha=0.5)
            ticker_counter += 1

    ax_price.set_ylabel("Price ($)")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(alpha=0.3)

    # Only show position subplot if not in only_ma_cross mode
    if ax_pos is not None:
        if not actions_df.empty and "action" in actions_df.columns:
            # Use actual action values (+1 buy, -1 sell, 0 no action) to track net position
            pos_change = actions_df["action"].astype(float)
            if "date" in actions_df.columns:
                pos_change.index = pd.to_datetime(actions_df["date"])
            pos = pos_change.cumsum().reindex(tic_data.index).ffill().fillna(0)
        else:
            pos = pd.Series(0, index=tic_data.index)
        ax_pos.plot(pos.index, pos.values, color="steelblue", linewidth=0.9)
        ax_pos.axhline(0, color="black", linewidth=0.5)
        ax_pos.set_ylabel("Shares held")
        ax_pos.set_xlabel("Date")
        ax_pos.grid(alpha=0.3)
    else:
        ax_price.set_xlabel("Date")

    plt.tight_layout()

    if save:
        safe = (title or f"rl_trades_{tic}").replace(" ", "_").replace("/", "-").replace(":", "")
        fpath = os.path.join(save_dir, f"{safe[:80]}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()
    plt.close(fig)


# ── Trade heatmap (moved from tables.py) ────────────────────────────────────────
def plot_trade_heatmap(
    actions_store: dict,
    window: str,
    datasets: list,
    algos: list,
    metric: str = "total",
    figsize: Optional[tuple] = None,
    save: bool = False,
    save_dir: str = "imgs/dow",
) -> None:
    """Heatmap: rows = (dataset/algo), columns = ticker. Values = trade event counts."""
    from utils.tables import compute_trade_stats

    os.makedirs(save_dir, exist_ok=True)

    if window not in actions_store:
        print(f"[WARN] Window {window} not in actions_store")
        return

    key_map = {"total": "total_by_ticker", "buys": "buys_by_ticker", "sells": "sells_by_ticker"}
    stat_key = key_map.get(metric, "total_by_ticker")

    row_labels: list[str] = []
    vectors: list[np.ndarray] = []
    all_tickers: Optional[list] = None

    for ds in datasets:
        if ds not in actions_store[window]:
            continue
        for algo in algos:
            if algo not in actions_store[window][ds]:
                continue
            adf = actions_store[window][ds][algo]
            stats = compute_trade_stats(adf)
            vec = stats[stat_key]

            if all_tickers is None:
                all_tickers = list(vec.index)

            row_labels.append(f"{ds}/{algo.upper()}")
            vectors.append(vec.reindex(all_tickers).fillna(0).values)

    if not vectors:
        print("[WARN] No data for trade heatmap")
        return

    mat = np.array(vectors, dtype=float)
    n_rows, n_cols = mat.shape

    if figsize is None:
        figsize = (max(14, n_cols * 0.55), max(4, n_rows * 0.38))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(all_tickers, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=7)

    vmax = mat.max()
    for i in range(n_rows):
        for j in range(n_cols):
            val = int(mat[i, j])
            if val > 0:
                color = "white" if mat[i, j] > vmax * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label=metric.capitalize())
    label_str = {"total": "Total Trades", "buys": "Buy Trades", "sells": "Sell Trades"}.get(metric, metric)
    ax.set_title(f"{label_str} by Ticker — {window.upper()}")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Dataset / Algorithm")
    plt.tight_layout()

    if save:
        fpath = os.path.join(save_dir, f"trade_heatmap_{window}_{metric}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()
    plt.close(fig)


# ── Ticker buy/sell aggregated (moved from tables.py) ────────────────────────────────────────
def plot_ticker_buysell_aggregated(
    actions_store: dict,
    window: str,
    datasets: list,
    algos: list,
    figsize: Optional[tuple] = None,
    save: bool = False,
    save_dir: str = "imgs/dow",
) -> None:
    """Aggregate buy/sell events across datasets for each algorithm."""
    from utils.tables import compute_trade_stats

    os.makedirs(save_dir, exist_ok=True)

    if window not in actions_store:
        print(f"[WARN] Window {window} not in actions_store")
        return

    n_algos = len(algos)
    if figsize is None:
        figsize = (16, 4 * n_algos)

    fig, axes = plt.subplots(n_algos, 1, figsize=figsize, sharex=False)
    if n_algos == 1:
        axes = [axes]

    all_tickers: Optional[list] = None

    for ax, algo in zip(axes, algos):
        agg_buys = None
        agg_sells = None

        for ds in datasets:
            if ds not in actions_store.get(window, {}):
                continue
            if algo not in actions_store[window][ds]:
                continue
            adf = actions_store[window][ds][algo]
            stats = compute_trade_stats(adf)

            b = stats["buys_by_ticker"]
            s = stats["sells_by_ticker"]

            if all_tickers is None:
                all_tickers = list(b.index)

            b = b.reindex(all_tickers).fillna(0)
            s = s.reindex(all_tickers).fillna(0)
            agg_buys = b if agg_buys is None else agg_buys + b
            agg_sells = s if agg_sells is None else agg_sells + s

        if agg_buys is None:
            ax.set_title(f"{algo.upper()} — no data")
            continue

        tickers = all_tickers or list(agg_buys.index)
        x = np.arange(len(tickers))
        width = 0.4

        ax.bar(x - width / 2, agg_buys.values, width, label="Buys", color="#2ecc71", alpha=0.85)
        ax.bar(x + width / 2, agg_sells.values, width, label="Sells", color="#e74c3c", alpha=0.85)
        ax.set_title(f"{algo.upper()} — Buy/Sell Distribution by Ticker (aggregated, {window.upper()})")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Trade events")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        fpath = os.path.join(save_dir, f"ticker_buysell_agg_{window}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()
    plt.close(fig)


# ── Portfolio comparison (moved from tables.py with linewidth=0 fix) ────────────────────────────────────────
def plot_portfolio_comparison(
    agg: dict,
    window: str,
    datasets: list,
    algos: list,
    baselines: Optional[dict] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    top_n: Optional[int] = None,
    figsize_curves: tuple = (15, 6),
    figsize_bar: tuple = (14, 5),
    save: bool = False,
    save_dir: str = "imgs/dow",
) -> None:
    """Two-chart portfolio comparison: equity curves + sorted bar chart."""
    os.makedirs(save_dir, exist_ok=True)

    if window not in agg:
        print(f"[WARN] Window {window} not in agg")
        return

    # Equity curves
    curves: dict[str, pd.Series] = {}
    for ds in datasets:
        if ds not in agg[window]:
            continue
        for algo in algos:
            if algo not in agg[window][ds]:
                continue
            curves[f"{ds}/{algo.upper()}"] = agg[window][ds][algo]["mean"]

    if not curves:
        print("[WARN] No equity curves found")
        return

    fig1, ax1 = plt.subplots(figsize=figsize_curves)
    cmap_colors = cm.tab20.colors
    for i, (label, ec) in enumerate(curves.items()):
        ax1.plot(ec.index, ec.values, linewidth=0.9, alpha=0.7,
                color=cmap_colors[i % len(cmap_colors)], label=label)

    if baselines and window in baselines:
        bl_styles = {"DJIA": ("black", 2.0), "MVO": ("dimgray", 1.8)}
        for bname, bec in baselines[window].items():
            bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
            col, lw = bl_styles.get(bname, ("gray", 1.5))
            ax1.plot(bec_s.index, bec_s.values, linewidth=lw, linestyle="--", color=col, label=bname, zorder=10)

    ax1.set_title(f"Portfolio Equity Curves — {window.upper()}")
    ax1.set_ylabel("Account Value ($)")
    ax1.set_xlabel("Date")
    ax1.legend(loc="upper left", fontsize=6, ncol=3)
    ax1.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        fpath = os.path.join(save_dir, f"equity_comparison_{window}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")
    plt.show()
    plt.close(fig1)

    # Return bar chart
    returns: dict[str, float] = {}
    for label, ec in curves.items():
        ec_clean = ec.dropna()
        if len(ec_clean) >= 2:
            ret = (ec_clean.iloc[-1] / ec_clean.iloc[0] - 1) * 100
            returns[label] = round(ret, 2)

    if baselines and window in baselines:
        for bname, bec in baselines[window].items():
            bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
            bec_s = bec_s.dropna()
            if len(bec_s) >= 2:
                returns[bname] = round((bec_s.iloc[-1] / bec_s.iloc[0] - 1) * 100, 2)

    if not returns:
        return

    ret_s = pd.Series(returns).sort_values(ascending=False)
    if top_n:
        ret_s = ret_s.head(top_n)

    fig2, ax2 = plt.subplots(figsize=figsize_bar)
    colors_bar = ["steelblue" if "/" in lbl else "black" for lbl in ret_s.index]
    bars = ax2.bar(range(len(ret_s)), ret_s.values, color=colors_bar, alpha=0.8)

    ax2.set_xticks(range(len(ret_s)))
    ax2.set_xticklabels(ret_s.index, rotation=60, ha="right", fontsize=7)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_ylabel("Total Return (%)")
    title_suffix = f"Top {top_n}" if top_n else "All"
    ax2.set_title(f"Final Portfolio Returns — {window.upper()} ({title_suffix})")
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, ret_s.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()

    if save:
        fpath = os.path.join(save_dir, f"return_bar_{window}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")
    plt.show()
    plt.close(fig2)


# ── Top configs equity (moved from tables.py with linewidth=0 and baseline fix) ────────────────────────────────────────
def plot_top_configs_equity(
    agg: dict,
    window: str,
    datasets: list,
    algos: list,
    top_n: int = 10,
    metric: str = "Sharpe",
    metrics_df: Optional[pd.DataFrame] = None,
    baselines: Optional[dict] = None,
    exclude_datasets: Optional[list] = None,
    initial_amount: float = 1_000_000,
    figsize: tuple = (14, 6),
    save: bool = False,
    save_dir: str = "imgs/dow",
) -> None:
    """
    Plot equity curves for top-N configurations ranked by metric.
    with ±1σ shading and optional baseline overlays.
    """
    os.makedirs(save_dir, exist_ok=True)
    exclude_set = set(exclude_datasets or [])

    _col_map = {"sharpe": "Sharpe", "return": "Return%", "return%": "Return%",
                "cagr": "CAGR", "sortino": "Sortino", "calmar": "Calmar"}
    metric_col = _col_map.get(metric.lower(), metric)

    # Build candidate pairs
    candidates = []
    for ds in datasets:
        if ds in exclude_set:
            continue
        if window not in agg or ds not in agg[window]:
            continue
        for algo in algos:
            if algo in agg[window][ds] and "mean" in agg[window][ds][algo]:
                candidates.append((ds, algo))

    if not candidates:
        print(f"[plot_top_configs_equity] No candidates found for window={window}.")
        return

    # Rank by metric
    if metrics_df is not None:
        ranked = []
        # Detect whether metrics_df is full (3-level: window/dataset/algo) or pre-sliced (2-level: dataset/algo)
        nlevels = getattr(metrics_df.index, 'nlevels', 1)

        for ds, algo in candidates:
            try:
                # Use 3-level key if full index, 2-level key if pre-sliced
                key = (window, ds, algo) if nlevels >= 3 else (ds, algo)
                val = float(metrics_df.loc[key, metric_col])
            except (KeyError, TypeError):
                val = float("nan")
            ranked.append((val, ds, algo))
        ranked.sort(key=lambda x: (float("nan") if x[0] != x[0] else -x[0]))
        selected = [(ds, algo) for _, ds, algo in ranked[:top_n]]
        metric_map = {(ds, algo): v for v, ds, algo in ranked}
    else:
        selected = candidates[:top_n]
        metric_map = {}

    metric_label = {"Return%": "R", "CAGR": "CAGR", "Sharpe": "S",
                   "Sortino": "So", "Calmar": "Ca"}.get(metric_col, metric_col)

    fig, ax = plt.subplots(figsize=figsize)
    palette = plt.cm.tab20.colors
    x_dates_range = None  # Track overall date range for baseline clipping

    for i, (ds, algo) in enumerate(selected):
        mean_ec = agg[window][ds][algo]["mean"]
        std_ec = agg[window][ds][algo].get("std", None)
        color = palette[i % len(palette)]

        # Normalize to 1.0 at start
        norm = mean_ec / initial_amount
        x_dates = pd.to_datetime(norm.index)  # Use each config's own index

        # Track overall date range (first config sets the range for baseline clipping)
        if x_dates_range is None:
            x_dates_range = (x_dates[0], x_dates[-1])

        label = f"{ds}/{algo.upper()}"
        val = metric_map.get((ds, algo), float("nan"))
        if not (val != val):  # not nan
            label += f" {metric_label}={val:.2f}"

        ax.plot(x_dates, norm.values, color=color, linewidth=1.4, label=label, zorder=3)

        if std_ec is not None:
            norm_std = std_ec / initial_amount
            ax.fill_between(x_dates, (norm - norm_std).values, (norm + norm_std).values,
                           color=color, alpha=0.12, linewidth=0, zorder=2)

    # Baseline overlays
    if baselines and window in baselines and x_dates_range is not None:
        bl_palette = ["black", "dimgray", "silver"]
        for j, (bname, bec) in enumerate(baselines[window].items()):
            bec_s = bec.iloc[:, 0] if isinstance(bec, pd.DataFrame) else bec
            norm_b = bec_s / bec_s.iloc[0]

            # Convert baseline index to DatetimeIndex and clip to agent date range
            norm_b_idx = pd.to_datetime(norm_b.index)
            mask = (norm_b_idx >= x_dates_range[0]) & (norm_b_idx <= x_dates_range[1])

            ax.plot(norm_b_idx[mask], norm_b.values[mask],
                   color=bl_palette[j % len(bl_palette)],
                   linewidth=1.2, linestyle="--", alpha=0.7,
                   label=bname, zorder=4)

    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("Portfolio Value (normalised to 1.0)")
    ax.set_xlabel("Date")
    ax.set_title(f"Top {len(selected)} Configurations by {metric_col} — Equity Curves ({window.upper()})")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        metric_slug = metric_col.lower().replace("%", "pct")
        fpath = os.path.join(save_dir, f"top{top_n}_configs_{window}_{metric_slug}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()
    plt.close(fig)
