"""Seed-paired, shared-schedule block summaries for schedule comparisons.

All metrics are computed upstream for each seed. This module only constructs
paired condition differences and averages outcomes that share a schedule before
calculating a Student-t interval across the resulting blocks.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ['Return%', 'Sharpe', 'MaxDD%', 'Trades']
KEYS = ['family', 'window', 'algo', 'seed']


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def stars(p, symbol='*'):
    return symbol * sum(p < threshold for threshold in (.05, .01, .001))


def annotate(frame, scope):
    frame = frame.copy()
    frame['p_holm'] = frame.groupby(scope, sort=False).p.transform(holm)
    frame['stars'] = frame.p.map(stars)
    frame['daggers'] = frame.p_holm.map(lambda p: stars(p, '†'))
    return frame


def load_schedule_seed_data(data_dir, schedule):
    name = ('2x2/2x2_train_eval_sequence_seed_mean_metrics.csv'
            if schedule == 'Crossover'
            else '2x2/random2x2_train_eval_sequence_seed_mean_metrics.csv')
    raw = pd.read_csv(Path(data_dir) / name)
    if raw.duplicated(KEYS + ['cell']).any() or len(raw) != 720:
        raise ValueError(f'Invalid seed table: {name}')
    if raw.groupby(KEYS).cell.nunique().ne(4).any():
        raise ValueError(f'Incomplete cells in {name}')
    return raw


def paired_differences(raw, treatment, reference):
    left = raw.loc[raw.cell.eq(treatment)].set_index(KEYS)[METRICS].sort_index()
    right = raw.loc[raw.cell.eq(reference)].set_index(KEYS)[METRICS].sort_index()
    if not left.index.equals(right.index) or len(left) != 180:
        raise ValueError(f'Unmatched seed pairs: {treatment} versus {reference}')
    out = (left - right).reset_index()
    out['Trades'] *= -1
    out['representation'] = np.where(out.family.str.endswith('_trg'), 'Crossovers', 'Indicators + crossovers')
    out['indicator'] = np.where(out.family.str.startswith('mv'), 'Moving average', 'Bollinger')
    out['schedule_family'] = out.family.str.removesuffix('_trg')
    return out


def blocked_summary(delta, selector, block_columns=('schedule_family', 'window', 'seed')):
    selected = delta.loc[selector].copy()
    if selected.empty:
        raise ValueError('Empty aggregation group')
    records = []
    for metric in METRICS:
        values = selected.groupby(list(block_columns), sort=True)[metric].mean().to_numpy()
        test = stats.ttest_1samp(values, 0)
        ci = test.confidence_interval(.95)
        records.append(dict(metric=metric, n_blocks=len(values), n_seed_differences=len(selected),
                            estimate=values.mean(), ci_low=ci.low, ci_high=ci.high, p=test.pvalue))
    return records


def selector_for_layout(delta, block, row):
    if block == 'Overall':
        return np.ones(len(delta), dtype=bool)
    if block == 'Representation':
        return delta.representation.eq(row)
    if block == 'Indicator family':
        return delta.indicator.eq(row)
    if block == 'Algorithm':
        return delta.algo.eq(row.lower())
    if block == 'Evaluation window':
        return delta.window.eq({'2020': 'w1', '2022': 'w4', '2025': 'w3'}[row])
    raise ValueError(block)
