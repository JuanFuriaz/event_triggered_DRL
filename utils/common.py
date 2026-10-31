
import os
from pathlib import Path

import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from utils.config import INDICATORS_BASE


def get_indicators(dataset_name: str, df=None) -> list:
    """
    Resolve the technical indicator list for a dataset name.

    Parameters
    ----------
    dataset_name : str
        e.g. 'mvx', 'flt_bollx', 'mvx_trg'
    df : pd.DataFrame, optional
        Required for _trg datasets (reads boolean columns from the DataFrame).

    Returns
    -------
    list[str]
    """
    base = dataset_name.removeprefix("flt_")
    if base.endswith("_trg"):
        if df is None:
            raise ValueError(f"df required for _trg dataset '{dataset_name}'")
        return list(df.select_dtypes(include=["bool", "boolean"]).columns)
    if base not in INDICATORS_BASE:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Known: {list(INDICATORS_BASE)}")
    return INDICATORS_BASE[base]


def process_df_for_mvo(df):
    """
    Convert a FinRL-format DataFrame (date, tic, close, ...) into
    a (date × ticker) price matrix suitable for MVO optimisation.
    """
    df = df.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]
    stock_dimension = len(df['tic'].unique())
    tickers = df.iloc[:stock_dimension]['tic'].tolist()

    mvo = pd.DataFrame(columns=tickers, dtype=float)
    for i in range(df.shape[0] // stock_dimension):
        chunk = df.iloc[i * stock_dimension:(i + 1) * stock_dimension]
        mvo.loc[chunk['date'].iloc[0]] = chunk['close'].tolist()
    return mvo


def stock_returns_computing(stock_price, rows, columns):
    """Compute period-over-period percentage returns from a price matrix."""
    stock_return = np.zeros([rows - 1, columns])
    for j in range(columns):
        for i in range(rows - 1):
            stock_return[i, j] = (
                (stock_price[i + 1, j] - stock_price[i, j]) / stock_price[i, j]
            ) * 100
    return stock_return


def run_prediction_vecnorm(model, plain_env, vecnorm_path):
    """
    Run backtest with saved VecNormalize statistics.

    DRLAgent.DRL_prediction expects a plain StockTradingEnv and calls get_sb_env()
    internally, so it cannot accept a VecNormalize wrapper directly. This function
    replicates the same loop but wraps the internal VecEnv with the trained
    normalization statistics before running inference.

    Parameters
    ----------
    model : stable_baselines3 policy (A2C / PPO / SAC)
    plain_env : StockTradingEnv  (NOT a VecEnv)
    vecnorm_path : str | Path
        Path to the saved VecNormalize .pkl file.

    Returns
    -------
    df_account_value : pd.DataFrame  (columns: date, account_value)
    df_actions       : pd.DataFrame
    """
    import torch
    import pandas as pd
    from stable_baselines3.common.vec_env import VecNormalize

    test_venv, _ = plain_env.get_sb_env()
    test_venv = VecNormalize.load(str(vecnorm_path), test_venv)
    test_venv.training    = False
    test_venv.norm_reward = False

    n_steps = len(plain_env.df.index.unique())
    obs = test_venv.reset()
    account_memory = actions_memory = None

    with torch.no_grad():
        for i in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = test_venv.step(action)
            if i == n_steps - 2:
                account_memory = test_venv.env_method("save_asset_memory")
                actions_memory = test_venv.env_method("save_action_memory")

    return pd.DataFrame(account_memory[0]), pd.DataFrame(actions_memory[0])


def build_full_price_wide(df):
    """
    Pivot a full daily FinRL-format dataframe (date, tic, close, ...) into a
    (date x ticker) close-price matrix, DatetimeIndex-sorted -- the calendar/price
    source reconstruct_daily_curve() marks sparse curves to market against.
    """
    tickers = sorted(df["tic"].unique())
    wide = df.pivot(index="date", columns="tic", values="close")[tickers]
    wide.index = pd.to_datetime(wide.index)
    return wide, wide.index


def reconstruct_daily_curve(date_memory, state_memory, stock_dim, initial_amount,
                             full_price_wide, full_dates):
    """
    Rebuild a full daily-calendar equity curve from a sparse (filtered/trigger-only)
    rollout. Between two consecutive decision dates the agent's position is flat (no
    trades happen on the skipped days -- the filtered environment never presents them),
    so the true mark-to-market value on those skipped days is exactly reconstructable:
    carry the holdings forward from state_memory and reprice against the real daily
    close. This is exact, not an approximation -- it fixes a real bug where computing
    metrics directly off the sparse curve treats every row as one consecutive trading
    day, inflating annualized Sharpe/CAGR/Volatility by roughly sqrt(gap) and truncating
    the measured period wherever the sparse schedule ends early.

    Reconstructs across the ENTIRE full_dates calendar, not just from the schedule's own
    first date -- a sparse schedule commonly starts a few days after the true calendar
    start, and curves being compared must span the identical calendar to be comparable.
    Cash-only before the first selected decision date; holdings held between selected
    dates; holdings marked to market every trading day; holdings carried to the common
    final daily date.
    """
    idx = full_dates.sort_values()
    values = pd.Series(index=idx, dtype=float)

    # Cash-only from the TRUE calendar start (idx[0]) through the schedule's own
    # first decision date -- not from date_memory[0], which can start later.
    seg_start = idx[0]
    seg_end = pd.Timestamp(date_memory[1]) if len(date_memory) > 1 else idx[-1] + pd.Timedelta(days=1)
    values[(idx >= seg_start) & (idx < seg_end)] = initial_amount

    for k in range(len(state_memory)):
        s = np.asarray(state_memory[k], dtype=float)
        cash = s[0]
        holdings = s[1 + stock_dim: 1 + 2 * stock_dim]
        seg_start = pd.Timestamp(date_memory[k + 1])
        seg_end = pd.Timestamp(date_memory[k + 2]) if (k + 2) < len(date_memory) else idx[-1] + pd.Timedelta(days=1)
        mask = (idx >= seg_start) & (idx < seg_end)
        seg_dates = idx[mask]
        if len(seg_dates) == 0:
            continue
        prices = full_price_wide.loc[seg_dates].values
        values[mask] = cash + prices.dot(holdings)

    return values


def run_prediction_vecnorm_reconstructed(model_path, vecnorm_path, algo_cls, sparse_df, indicators,
                                          full_price_wide, full_dates, device="cpu"):
    """
    Like run_prediction_vecnorm, but for a FILTERED/sparse eval_df: evaluates the model
    on its own sparse schedule only (never the full daily calendar as the action
    environment), then reconstructs a full daily equity curve via reconstruct_daily_curve
    for a fair, gap-aware metric comparison against full-daily curves. Also returns a
    (date x ticker) holdings-levels frame for trade-stats -- NOT the same shape as
    run_prediction_vecnorm's df_actions (per-day trade deltas): compute_trade_stats()
    expects holdings levels and does its own .diff() to recover trade events.
    """
    import torch
    from stable_baselines3.common.vec_env import VecNormalize
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    env_kwargs = make_env_kwargs(sparse_df, indicators)
    df_eval = sparse_df.copy().sort_values(["date", "tic"], ignore_index=True)
    df_eval["day"] = df_eval["date"].factorize()[0]
    df_indexed = df_eval.set_index("day")

    plain_env = StockTradingEnv(df=df_indexed, **env_kwargs)
    test_venv, _ = plain_env.get_sb_env()
    test_venv = VecNormalize.load(str(vecnorm_path), test_venv)
    test_venv.training = False
    test_venv.norm_reward = False

    model = algo_cls.load(str(model_path), device=device)

    n_steps = len(plain_env.df.index.unique())
    obs = test_venv.reset()
    with torch.no_grad():
        for i in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = test_venv.step(action)
            if i == n_steps - 2:
                date_mem = test_venv.get_attr("date_memory")[0]
                state_mem = test_venv.get_attr("state_memory")[0]

    stock_dim = env_kwargs["stock_dim"]
    initial_amount = env_kwargs["initial_amount"]
    curve = reconstruct_daily_curve(date_mem, state_mem, stock_dim, initial_amount,
                                     full_price_wide, full_dates)

    # Curves being compared must span the identical calendar.
    assert curve.index.min() == full_dates.min(), (curve.index.min(), full_dates.min())
    assert curve.index.max() == full_dates.max(), (curve.index.max(), full_dates.max())

    tickers = sorted(sparse_df["tic"].unique())
    holdings_rows = [np.asarray(s, dtype=float)[1 + stock_dim:1 + 2 * stock_dim] for s in state_mem]
    # Prepend the (all-zero) opening position so compute_trade_stats' .diff() captures
    # the FIRST trade too -- state_memory starts only after the first action already
    # executed, so without this row the opening trade is silently uncounted.
    opening = pd.DataFrame([np.zeros(stock_dim)], index=[pd.Timestamp(date_mem[0])], columns=tickers)
    positions_df = pd.concat([
        opening,
        pd.DataFrame(holdings_rows, index=pd.to_datetime(date_mem[1:]), columns=tickers),
    ])

    # Every holdings/action date must lie within the sparse schedule actually given to the env.
    schedule_dates = set(pd.to_datetime(sparse_df["date"].unique()))
    assert set(positions_df.index).issubset(schedule_dates), (
        "Reconstructed action dates outside the sparse schedule -- indexing bug."
    )

    return curve, positions_df


def actions_to_positions(df_actions):
    """
    Convert FinRL's per-step action-delta frame (shares traded that day, +buy/-sell,
    from save_action_memory()) into a cumulative holdings-levels frame -- the shape
    compute_trade_stats() actually expects (it internally .diff()s its input to recover
    trade events; feeding it action-deltas directly double-differences the data and
    produces nonsensical counts -- e.g. a fake "Short Days" count whenever a sell action
    alone exceeds the threshold, even though this env can never hold a net-short
    position, since _sell_stock() always clamps a sell to current holdings).

    Prepends an all-zero "opening position" row one calendar day before the first trade
    so the first day's trade is captured by that diff() too, not silently dropped.
    """
    positions = df_actions.cumsum()
    opening_date = pd.to_datetime(positions.index[0]) - pd.Timedelta(days=1)
    opening = pd.DataFrame(0, index=[opening_date], columns=positions.columns)
    return pd.concat([opening, positions])


def read_csv(path):
    """Load CSV and set index correctly"""
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df.index.names = ['']
    return df

def make_env_kwargs(df, indicators):
    """Create environment kwargs from df and indicator list"""
    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
    return {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": indicators,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    

def filter_x(df):
    df_cp = df.copy()
    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
    df_cp[bool_cols] = df[bool_cols].astype(int)  # transforming not transformed df booleans 
    filtered_df = df_cp[df_cp[bool_cols].any(axis=1)]
    return filtered_df

def filter_x_keep_all_companies(df):
    """
    Filter to keep dates where ANY company has a trigger,
    but keep ALL companies for those dates.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date, tic (ticker), OHLCV data, and boolean trigger columns

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing ALL companies on dates where any trigger occurred
        Boolean columns preserve their original True/False values per company
    """
    # Identify boolean columns (trigger indicators)
    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns

    # For each date, check if ANY company has ANY trigger
    # Group by date and check if any boolean column is True for any company on that date
    dates_with_triggers = df.groupby('date')[bool_cols].apply(
        lambda x: x.any().any()  # First .any() checks across columns, second across rows
    )

    # Get the dates where at least one trigger occurred
    trigger_dates = dates_with_triggers[dates_with_triggers].index.tolist()

    # Filter to keep ALL companies on those dates
    filtered_df = df[df['date'].isin(trigger_dates)].copy()

    return filtered_df


def load_dataset(name: str, cvtd: bool = False,
                 data_dir: Path | None = None,
                 full_date: str = "2009-01-01_to_2026-01-01") -> pd.DataFrame:
    """
    Load a pre-generated DOW-30 dataset CSV.

    Parameters
    ----------
    name : str
        Dataset variant, e.g. 'mvx', 'bollx', 'flt_bollx', '3mvx1'.
    cvtd : bool
        If True load compact variant (no open/high/low). Default False → chlovtd.
    data_dir : Path, optional
        Override data directory. Default: $FINRL_PATH/data/dow
    full_date : str
        Date range suffix used in CSV filenames.
    """
    if data_dir is None:
        data_dir = Path(os.environ["FINRL_PATH"]) / "data" / "dow"
    prefix = "cvtd" if cvtd else "chlovtd"
    path = data_dir / f"dow_{prefix}_{name}_{full_date}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run dow_1_data.ipynb to generate it."
        )
    return pd.read_csv(path)


def make_window_data(
    dataset_names: list,
    windows: list | None = None,
    splits: list | None = None,
    cvtd: bool = False,
    data_dir: Path | None = None,
    full_date: str = "2009-01-01_to_2026-01-01",
    test: bool = False,
) -> dict:
    """
    Slice pre-generated datasets into walk-forward train/test windows.

    Imports WINDOWS from dow_1_train (single source of truth for window dates).

    Parameters
    ----------
    dataset_names : list[str]
        Names of datasets to load, e.g. ['mvx', 'bollx', 'flt_bollx'].
    windows : list[str], optional
        Which windows to produce. Default: all windows defined in dow_1_train.WINDOWS.
    splits : list[str], optional
        Which splits to produce: 'train', 'test', or both. Default: ['train', 'test'].
    cvtd : bool
        Load compact (cvtd) variant instead of full OHLCV (chlovtd).
    data_dir : Path, optional
        Override data directory.
    full_date : str
        Date range suffix used in CSV filenames.

    Returns
    -------
    dict
        result[window][dataset_name][split] = {"df": pd.DataFrame, "tag": str}
        tag is the canonical agent name suffix:
            e.g. "dow_chlovtd_mvx_2009-01-01_to_2018-12-01"
    """
    from utils.config import WINDOWS as _WINDOWS  # single source of truth
    from utils.config import TEST_WINDOWS
    
    if test:
        _WINDOWS = TEST_WINDOWS

    if windows is None:
        windows = list(_WINDOWS.keys())
    if splits is None:
        splits = ["train", "test"]

    unknown_windows = set(windows) - set(_WINDOWS)
    if unknown_windows:
        raise ValueError(f"Unknown windows: {unknown_windows}. Valid: {list(_WINDOWS)}")
    unknown_splits = set(splits) - {"train", "test"}
    if unknown_splits:
        raise ValueError(f"Unknown splits: {unknown_splits}. Valid: ['train', 'test']")

    prefix = "cvtd" if cvtd else "chlovtd"
    result: dict = {w: {} for w in windows}

    for name in dataset_names:
        df_full = load_dataset(name, cvtd=cvtd, data_dir=data_dir, full_date=full_date)

        for w in windows:
            if name not in result[w]:
                result[w][name] = {}
            for split in splits:
                start, end = _WINDOWS[w][split]
                df_slice = filter_date_range(
                    df_full, start_date=start, end_date=end, tic="all", stock_df=False
                )
                tag = f"dow_{prefix}_{name}_{start}_to_{end}"
                result[w][name][split] = {"df": df_slice, "tag": tag}

    return result


def filter_date_range(df, start_date='2020-01-01', end_date='2021-02-01', tic='all', stock_df=True):
    """
    Filter DataFrame by date range and optionally by ticker symbol.
    Handles both 'date' as index (stockstats) and 'date' as column.
    """
    # Normalise: work with the index if date is there, column otherwise
    if df.index.name == 'date':
        date_index = df.index
    elif 'date' in df.columns:
        date_index = pd.to_datetime(df['date'])
    else:
        raise ValueError("DataFrame must have 'date' as index or column")

    if 'tic' not in df.columns:
        raise ValueError("DataFrame must contain a 'tic' column")

    mask = (date_index >= start_date) & (date_index <= end_date)
    df_filtered = df.loc[mask]

    if tic != 'all':
        if tic not in df['tic'].values:
            raise ValueError(f"Ticker '{tic}' not found. Available: {df['tic'].unique()}")
        df_filtered = df_filtered[df_filtered['tic'] == tic]

    if stock_df:
        df_filtered = StockDataFrame.retype(df_filtered)

    return df_filtered


# ============================================================================
# Baseline Benchmark Functions (for down_3_backtest.ipynb Section 8)
# ============================================================================

def get_etf_baseline(ticker, test_start, test_end, initial_amount=1_000_000):
    """
    Download ETF or index ticker (e.g., 'SPY', 'DIA', '^GSPC') and normalise
    to initial capital.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'SPY', 'DIA')
    test_start : str
        Start date (YYYY-MM-DD)
    test_end : str
        End date (YYYY-MM-DD)
    initial_amount : float
        Initial capital in dollars

    Returns
    -------
    pd.Series
        Date-indexed equity curve starting at initial_amount
    """
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    df = YahooDownloader(start_date=test_start, end_date=test_end,
                         ticker_list=[ticker]).fetch_data()
    df = df.sort_values("date")
    series = df["close"] / df["close"].iloc[0] * initial_amount
    series.index = pd.to_datetime(df["date"])
    series.name = ticker
    return series


def compute_bh_dj30(test_df, initial_amount=1_000_000, freq="Q", cost_bps=0.001):
    """
    Equal-weight DJ30 portfolio, rebalanced back to equal weight on a fixed calendar
    schedule. freq='Q' (quarterly) matches the S&P Equal Weight Index's own real-world
    rebalancing convention; freq='M' (monthly) rebalances more often. cost_bps is the
    per-transaction cost fraction, matching the RL environment's own buy_cost_pct/
    sell_cost_pct (0.001 = 10bps, see make_env_kwargs below) -- pass 0 to ignore costs.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test period data (date, tic, close, ...)
    initial_amount : float
        Initial capital
    freq : str
        Pandas period alias for the rebalancing calendar ('Q' or 'M').
    cost_bps : float
        Transaction cost as fraction (default 0.001 = 10bps, matching the RL agents)

    Returns
    -------
    equity : pd.Series
        Date-indexed equity curve
    n_trades : int
        From compute_trade_stats on the reconstructed share-position frame -- the same
        trade-counting logic already used for every RL row, so it's directly comparable.
    """
    from utils.tables import compute_trade_stats

    prices = test_df.pivot(index="date", columns="tic", values="close")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    n = prices.shape[1]

    shares = (initial_amount / n) / prices.iloc[0]
    cost_paid = initial_amount * cost_bps
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    equity = pd.Series(index=prices.index, dtype=float)
    current_period = None

    for date, row in prices.iterrows():
        period = date.to_period(freq)
        if period != current_period:
            if current_period is not None:
                cur_val = (shares * row).sum()
                cost_paid += cur_val * cost_bps
                shares = (cur_val / n) / row
            current_period = period
        positions.loc[date] = shares
        equity[date] = (shares * row).sum()

    n_trades = compute_trade_stats(positions)["total_trades"]
    return (equity - cost_paid).rename(f"EW_DJ30_{freq}"), n_trades


def compute_ew_bh(test_df, initial_amount=1_000_000):
    """
    Simple equal-weight buy-and-hold over the test period. No rebalancing.
    Buys equal dollar amounts of all stocks on the first available trading day
    and holds until the last available trading day.

    Return formula:
        R_EW = mean_i( close_i(T_end) / close_i(T_start) - 1 ) * 100

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-period data with columns: date, tic, close (adjusted close)
    initial_amount : float
        Starting capital (default $1,000,000)

    Returns
    -------
    equity : pd.Series
        Date-indexed equity curve starting at initial_amount
    ret_pct : float
        Simple close-to-close percentage return over the test window
    missing : list
        Tickers that had any NaN values (forward-filled before use)
    """
    prices = test_df.pivot(index="date", columns="tic", values="close")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    missing = prices.columns[prices.isnull().any()].tolist()
    if missing:
        print(f"  [EW_BH] NaN in {missing} — forward-filling")
        prices = prices.ffill().bfill()

    n = prices.shape[1]
    shares = (initial_amount / n) / prices.iloc[0]   # equal $ at T_start
    equity = (prices * shares).sum(axis=1)
    equity = equity / equity.iloc[0] * initial_amount  # normalise

    # Scalar return = mean of per-stock simple returns (equal-weight)
    per_stock = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    ret_pct = float(per_stock.mean())

    return equity.rename("EW_DJ30_BH"), ret_pct, missing


def compute_solo_mvx(test_df, initial_amount=1_000_000, cost_bps=0.001):
    """
    Rule-based (non-RL) strategy: Buy on EMA-5 > SMA-20 golden cross,
    sell on death cross. Equal-weight among 'long' stocks.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test period with mvx trigger signals (close_5_ema_xu_close_20_sma, close_5_ema_xd_close_20_sma)
    initial_amount : float
        Initial capital
    cost_bps : float
        Transaction cost (default 0.001 = 10bps)

    Returns
    -------
    pd.Series
        Date-indexed equity curve
    """
    prices   = test_df.pivot(index="date", columns="tic", values="close")
    buy_sig  = test_df.pivot(index="date", columns="tic",
                              values="close_5_ema_xu_close_20_sma").fillna(False)
    sell_sig = test_df.pivot(index="date", columns="tic",
                              values="close_5_ema_xd_close_20_sma").fillna(False)

    # Cumulative position: 1=long, 0=out
    state = pd.Series(0, index=prices.columns)
    positions = []
    for date in prices.index:
        state[buy_sig.loc[date].astype(bool)]  = 1
        state[sell_sig.loc[date].astype(bool)] = 0
        positions.append(state.copy())

    position = pd.DataFrame(positions, index=prices.index)
    active   = position.sum(axis=1).replace(0, np.nan)
    weights  = position.div(active, axis=0).fillna(0)
    daily_ret = prices.pct_change().fillna(0)
    turnover  = position.diff().abs().sum(axis=1) / position.shape[1]
    port_ret  = (weights * daily_ret).sum(axis=1) - turnover * cost_bps

    return ((1 + port_ret).cumprod() * initial_amount).rename("Solo_MVX")


def compute_solo_bollx(test_df, initial_amount=1_000_000, cost_bps=0.001):
    """
    Rule-based (non-RL) strategy: Buy on price break above lower Bollinger Band,
    sell on price break above upper Band. Equal-weight among 'long' stocks.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test period with bollx trigger signals (close_xu_boll_lb, close_xu_boll_ub)
    initial_amount : float
        Initial capital
    cost_bps : float
        Transaction cost (default 0.001 = 10bps)

    Returns
    -------
    pd.Series
        Date-indexed equity curve
    """
    prices   = test_df.pivot(index="date", columns="tic", values="close")
    buy_sig  = test_df.pivot(index="date", columns="tic",
                              values="close_xu_boll_lb").fillna(False)
    sell_sig = test_df.pivot(index="date", columns="tic",
                              values="close_xu_boll_ub").fillna(False)

    state = pd.Series(0, index=prices.columns)
    positions = []
    for date in prices.index:
        state[buy_sig.loc[date].astype(bool)]  = 1
        state[sell_sig.loc[date].astype(bool)] = 0
        positions.append(state.copy())

    position = pd.DataFrame(positions, index=prices.index)
    active   = position.sum(axis=1).replace(0, np.nan)
    weights  = position.div(active, axis=0).fillna(0)
    daily_ret = prices.pct_change().fillna(0)
    turnover  = position.diff().abs().sum(axis=1) / position.shape[1]
    port_ret  = (weights * daily_ret).sum(axis=1) - turnover * cost_bps

    return ((1 + port_ret).cumprod() * initial_amount).rename("Solo_BOLLX")


def compute_momentum_baseline(train_df, test_df, initial_amount=1_000_000, cost_bps=0.001):
    """
    12-month lookback momentum strategy: monthly rebalance, long top 10 stocks
    (top tercile) by 12-month return (skip last month).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (for lookback history)
    test_df : pd.DataFrame
        Test period data
    initial_amount : float
        Initial capital
    cost_bps : float
        Transaction cost (default 0.001 = 10bps)

    Returns
    -------
    pd.Series
        Date-indexed equity curve
    """
    # Use last 14 months of train + all test for history
    cutoff = pd.Timestamp(test_df["date"].min()) - pd.DateOffset(months=14)
    hist = train_df[train_df["date"] >= cutoff.strftime("%Y-%m-%d")]
    full_df = pd.concat([hist, test_df]).drop_duplicates(subset=["date", "tic"])

    prices = full_df.pivot(index="date", columns="tic", values="close").sort_index()
    test_dates = pd.DatetimeIndex(sorted(test_df["date"].unique()))

    portfolio_val = pd.Series(dtype=float, index=test_dates)
    shares = None
    current_month = None

    for date in test_dates:
        if date.month != current_month:
            # 12-1 momentum: return from 13 months ago to 1 month ago
            t_minus_1  = date - pd.DateOffset(months=1)
            t_minus_13 = date - pd.DateOffset(months=13)
            hist_slice = prices.loc[:t_minus_1]

            if len(hist_slice) < 20:
                current_month = date.month
                continue

            p_start = hist_slice.loc[:t_minus_13].iloc[-1]
            p_end   = hist_slice.iloc[-1]
            mom     = (p_end - p_start) / p_start
            top10   = mom.nlargest(10).index

            if shares is not None:
                cur_val = (shares * prices.loc[date]).sum()
            else:
                cur_val = initial_amount

            cost = cur_val * cost_bps
            cur_val -= cost
            shares = pd.Series(0.0, index=prices.columns)
            shares[top10] = (cur_val / 10) / prices.loc[date, top10]
            current_month = date.month

        if shares is not None:
            portfolio_val[date] = (shares * prices.loc[date]).sum()

    return portfolio_val.rename("Momentum_12_1")


def bootstrap_ci(equity_curve, n_boot=1000, ci=0.95, seed=42):
    """
    Compute block bootstrap confidence intervals for Return%, Sharpe, MaxDD%.

    Parameters
    ----------
    equity_curve : pd.Series
        Date-indexed daily account values
    n_boot : int
        Number of bootstrap resamples (default 1000)
    ci : float
        Confidence level (default 0.95 = 95% CI)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Keys: Return%_mean, Return%_std, Return%_lo95, Return%_hi95, etc.
    """
    from utils.tables import compute_metrics

    rng = np.random.default_rng(seed)
    rets = equity_curve.pct_change().dropna().values
    n = len(rets)
    block = max(1, int(np.sqrt(n)))

    boot_metrics = []
    for _ in range(n_boot):
        indices = np.concatenate([
            np.arange(s, min(s + block, n))
            for s in rng.integers(0, n - block + 1, size=(n // block) + 1)
        ])[:n]
        boot_ec = (1 + rets[indices]).cumprod() * equity_curve.iloc[0]
        boot_ec = pd.Series(boot_ec, index=equity_curve.index[:n])
        m = compute_metrics(boot_ec)
        boot_metrics.append(m)

    lo = (1 - ci) / 2
    hi = 1 - lo
    results = {}

    for key in ["Return%", "Sharpe", "MaxDD%"]:
        vals = [m[key] for m in boot_metrics if m[key] is not None and not np.isnan(m[key])]
        if len(vals) == 0:
            continue
        results[f"{key}_mean"] = np.mean(vals)
        results[f"{key}_std"]  = np.std(vals)
        results[f"{key}_lo95"] = np.quantile(vals, lo)
        results[f"{key}_hi95"] = np.quantile(vals, hi)

    return results




def compare_placebo(
    base_curves: list,
    null_curves: list,
    algo_name: str,
    test_label: str,
    compute_metrics_fn,
    alpha: float = 0.05,
) -> dict:
    """
    Compare a baseline distribution (N seeds) against a null distribution (N shuffles).

    Uses Mann-Whitney U (one-sided, greater) — no normality assumption.
    Reports z-score relative to null std, % null samples beaten, and p-value.

    Parameters
    ----------
    base_curves : list of pd.Series
        Per-seed equity curves for the real model (typically N=5).
    null_curves : list of pd.Series
        Shuffle/permutation equity curves (typically N=20).
    algo_name : str
        Used for printing.
    test_label : str
        Description of what null_curves represent, e.g. 'random dates' or 'shuffled triggers'.
    compute_metrics_fn : callable
        Function that takes an equity curve Series and returns a dict with 'Return%' and 'Sharpe'.
    alpha : float
        Significance threshold. Default 0.05.

    Returns
    -------
    dict with keys: b_ret_mu, b_ret_sd, r_ret_mu, r_ret_sd, z_ret, pct_ret, p_ret,
                    b_shr_mu, b_shr_sd, r_shr_mu, r_shr_sd, z_shr, pct_shr, p_shr,
                    sig_ret, sig_shr, verdict
    """
    from scipy import stats as scipy_stats

    base_ret = [compute_metrics_fn(c).get('Return%', 0) for c in base_curves]
    base_shr = [compute_metrics_fn(c).get('Sharpe',   0) for c in base_curves]
    null_ret = [compute_metrics_fn(c).get('Return%', 0) for c in null_curves]
    null_shr = [compute_metrics_fn(c).get('Sharpe',   0) for c in null_curves]

    b_ret_mu, b_ret_sd = np.mean(base_ret), np.std(base_ret, ddof=1)
    r_ret_mu, r_ret_sd = np.mean(null_ret), np.std(null_ret,  ddof=1)
    z_ret  = (b_ret_mu - r_ret_mu) / (r_ret_sd + 1e-9)
    pct_ret = np.mean([b_ret_mu > r for r in null_ret]) * 100
    _, p_ret = scipy_stats.mannwhitneyu(base_ret, null_ret, alternative='greater')

    b_shr_mu, b_shr_sd = np.mean(base_shr), np.std(base_shr, ddof=1)
    r_shr_mu, r_shr_sd = np.mean(null_shr), np.std(null_shr,  ddof=1)
    z_shr  = (b_shr_mu - r_shr_mu) / (r_shr_sd + 1e-9)
    pct_shr = np.mean([b_shr_mu > r for r in null_shr]) * 100
    _, p_shr = scipy_stats.mannwhitneyu(base_shr, null_shr, alternative='greater')

    sig_ret = p_ret < alpha
    sig_shr = p_shr < alpha
    if sig_ret and sig_shr:
        verdict = "SIGNAL  (both Return & Sharpe significant)"
    elif sig_ret or sig_shr:
        verdict = "PARTIAL (one metric significant)"
    else:
        verdict = "NOISE   (not better than null)"

    print(f'\n    {algo_name.upper()}  '
          f'(baseline N={len(base_ret)} seeds | {test_label} N={len(null_ret)})')
    print(f'      Return%  baseline: {b_ret_mu:+6.2f}% ± {b_ret_sd:.2f}')
    print(f'               null:     {r_ret_mu:+6.2f}% ± {r_ret_sd:.2f}'
          f'   beats {pct_ret:.0f}%   z={z_ret:+.2f}  p={p_ret:.3f}')
    print(f'      Sharpe   baseline: {b_shr_mu:+6.2f}  ± {b_shr_sd:.2f}')
    print(f'               null:     {r_shr_mu:+6.2f}  ± {r_shr_sd:.2f}'
          f'   beats {pct_shr:.0f}%   z={z_shr:+.2f}  p={p_shr:.3f}')
    symbol = '✓' if sig_ret and sig_shr else ('~' if sig_ret or sig_shr else '✗')
    print(f'      → {symbol} {verdict}')

    return dict(
        b_ret_mu=b_ret_mu, b_ret_sd=b_ret_sd,
        r_ret_mu=r_ret_mu, r_ret_sd=r_ret_sd,
        z_ret=z_ret, pct_ret=pct_ret, p_ret=p_ret,
        b_shr_mu=b_shr_mu, b_shr_sd=b_shr_sd,
        r_shr_mu=r_shr_mu, r_shr_sd=r_shr_sd,
        z_shr=z_shr, pct_shr=pct_shr, p_shr=p_shr,
        sig_ret=sig_ret, sig_shr=sig_shr, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Frozen random/circular-shift schedule manifests (dow_5_smrsc's control-group
# panels) -- shared so dow_4_advtest 2x2.ipynb can evaluate its own daily-trained
# models on the identical frozen dates, without re-deriving dow_5's schedule format.
# ---------------------------------------------------------------------------

def manifest_row_lookup(manifest, family, window, split, seed):
    """Single row from a frozen schedule manifest for (family, window, split, seed)."""
    selected = manifest[
        (manifest["family"] == family)
        & (manifest["window"] == window)
        & (manifest["split"] == split)
        & (manifest["model_seed"] == seed)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one manifest row for {(family, window, split, seed)}, "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def load_frozen_target_dates(prefix, path, manifest_row):
    """Read authoritative target dates without regenerating a schedule."""
    if prefix == "placebo":
        dates = {
            date for date in str(manifest_row["selected_random_dates"]).split(";")
            if date
        }
    elif prefix == "circshift":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen circular schedule: {path}")
        dates = set(pd.read_csv(path, usecols=["date"])["date"].astype(str))
    else:
        raise ValueError(f"Unsupported schedule prefix: {prefix}")

    expected_count = int(manifest_row["retained_date_count"])
    if len(dates) != expected_count:
        raise ValueError(
            f"{path} should contain {expected_count} frozen target dates; "
            f"found {len(dates)}"
        )
    return dates


def control_model_run_name(treatment_tag, dataset_name, tag_infix, family, algorithm, seed):
    """
    Run name for a control model (e.g. dow_5_smrsc's random-date-trained placebo
    model), derived from its matching treatment run's tag by swapping in the control
    label. Pure string logic -- callers resolve treatment_tag/dataset_name/MODELS_DIR
    themselves, this only builds the name.
    """
    control_tag = treatment_tag.replace(dataset_name, f"{tag_infix}_{family}")
    return f"{control_tag}_{algorithm}_s{seed}"



