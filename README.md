# Event-Triggered DRL for Multi-Asset Trading

Experiment code for studying event-triggered trading schedules in deep
reinforcement learning. Instead of letting a trading agent act on every
daily bar, a trigger scheduler restricts *which* dates the agent may train
and trade on — either dates where a moving-average or Bollinger crossover
fires, or a random-date control of the same count — while daily prices keep
accruing in the background between those dates. The code trains and
backtests A2C/PPO/SAC agents on the Dow Jones 30 under several input
representations and trading-date schedules, runs the controls needed to
separate scheduling from the agent's learned behavior, and regenerates the
descriptive and inferential figures built from the results.

## Installation

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JuanFuriaz/event_triggered_DRL.git
cd event_triggered_DRL
uv sync
```

`uv sync` downloads the pinned Python 3.12.12 interpreter if it isn't
already on your machine and installs every dependency from `uv.lock` into a
local `.venv/`, including `finrl` pinned to the exact git commit this
project was built against (it predates FinRL's own PyPI releases).

### Set `FINRL_PATH`

The notebooks read and write all *generated* data, trained models, and
training logs under `$FINRL_PATH`, not inside this repository — only the
small raw OHLCV panel in `data/` is committed here. Point it at wherever you
want that (potentially large) storage to live:

```bash
export FINRL_PATH=/path/to/your/storage/
```

Add this to your shell profile so it's set for every session. Running
`dow_1_data.ipynb` creates `$FINRL_PATH/data/dow/`; running
`dow_2_train.ipynb` creates `$FINRL_PATH/models/dow/` and
`$FINRL_PATH/logs/dow/`.

### Run Jupyter

```bash
uv run jupyter lab
```

This launches Jupyter from inside the project's own virtual environment, so
the notebook kernel already has every dependency available — no separate
kernel registration needed.

### GPU (optional, but the training notebooks expect one)

`dow_2_train.ipynb` and `dow_5_controls.ipynb` default every algorithm to
`device="cuda"` and split a fixed GPU-memory fraction across parallel
training workers (`N_WORKERS`, `GPU_FRAC`). `uv sync` installs PyTorch's
standard build, which includes CUDA support and will use a GPU automatically
if compatible drivers are present:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Without a GPU, lower `N_WORKERS` and change `device` to `"cpu"` in the
training cells. Backtesting and figure generation (`dow_3`, `dow_6`) don't
need a GPU.

## Repository layout

A fresh clone contains:

- `dow_1_data.ipynb` … `dow_6_figures.ipynb` — the numbered pipeline, described below
- `utils/` — the shared pipeline code the notebooks import: config, data/env helpers, training workers, figure and statistics generation
- `data/` — the raw OHLCV panel; every derived dataset is generated locally under `$FINRL_PATH`, not committed
- `supplementary/` — result tables referenced from the paper's supplementary material

Running the notebooks creates further local, untracked directories:
`imgs/` and `tables_mv2boll/` for generated figures and result tables. A
`legacy/` folder (superseded notebooks and scripts, kept only for
provenance) and a `tests/` folder also exist in development but are not
part of the published repository.

## Notebooks

### `dow_1_data.ipynb` — Dataset Creation
Downloads Dow Jones 30 daily OHLCV data (`YahooDownloader`), computes
technical indicators and compound crossover triggers (`FeatureEngineer`),
and exports every representation and trading-date variant used downstream:
the full indicator set, the crossover-only columns (code suffix `_trg`),
and date-filtered (`flt_`) versions restricted to crossover trading dates.

### `dow_2_train.ipynb` — Training
Loads the walk-forward window data, builds the Stable-Baselines3
A2C/PPO/SAC training jobs for every dataset × window × seed combination,
and dispatches them across parallel GPU workers with resume-on-restart
(jobs whose model file already exists are skipped).

### `dow_3_backtest.ipynb` — Walk-Forward Backtest
Loads the trained models and their saved `VecNormalize` statistics, runs
inference across all seeds, aggregates to mean ± std equity curves, and
builds the financial benchmarks (DJIA, mean-variance optimization, S&P 500,
equal-weight Dow 30 buy-and-hold). Produces the metrics tables, comparison
plots, and a per-trade interpretability view of where an agent actually
trades.

### `dow_4_ablations.ipynb` — Schedule-Component Ablations and Daily-Baseline Schedule Tests
Separates the two things a trigger schedule changes — which dates are used
for *training* versus which dates the agent may *trade* on — into a 2×2
grid (daily/crossover training × daily/crossover trading), repeats the same
grid against the random-date control, and runs a daily-trained-vs-schedule-trained
ladder. Also includes a cross-distribution generalization check: evaluating
each model on the trading dates it was *not* trained for.

### `dow_5_controls.ipynb` — Date Controls
Builds and evaluates the two placebo controls for each real crossover
schedule: a random-date trigger-panel placebo (assigns the schedule to the
same number of randomly selected real market dates) and a circular-shift
placebo (translates the whole schedule by a random offset, preserving its
inter-event gaps). Both retrain a fresh policy rather than re-evaluating an
existing one.

### `dow_6_figures.ipynb` — Figures
Regenerates the descriptive and inferential figures from the exported
result CSVs: the input-retention and performance-vs-benchmark descriptive
figures, the random-vs-crossover and random-vs-daily seed-paired
comparisons, and the schedule-component heatmap. `utils/descriptive_figures.py`
is the shared plotting implementation.
