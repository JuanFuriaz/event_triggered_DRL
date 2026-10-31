"""Seed-paired random-schedule intervals and method-comparison figure."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from utils.descriptive_figures import BLUE, INK, performance_data
from utils.seed_paired_stats import (METRICS, annotate, blocked_summary, load_schedule_seed_data,
                               paired_differences, selector_for_layout)


def configuration_reference(data_dir, layout):
    raw = pd.read_csv(Path(data_dir) / '2x2/random2x2_train_eval_sequence_mean_metrics.csv')
    keys = ['family', 'window', 'algo']
    dd = raw.loc[raw.cell.eq('DD')].set_index(keys)[METRICS].sort_index()
    rr = raw.loc[raw.cell.eq('RR')].set_index(keys)[METRICS].sort_index()
    delta = (rr - dd).reset_index()
    delta['Trades'] *= -1
    delta['representation'] = np.where(delta.family.str.endswith('_trg'), 'Crossovers', 'Indicators + crossovers')
    delta['indicator'] = np.where(delta.family.str.startswith('mv'), 'Moving average', 'Bollinger')
    records = []
    for row in layout.itertuples():
        group = delta.loc[selector_for_layout(delta, row.block, row.row)]
        for metric in METRICS:
            values = group[metric].to_numpy()
            test = stats.ttest_1samp(values, 0)
            ci = test.confidence_interval(.95)
            records.append(dict(block=row.block, row=row.row, y=row.y, metric=metric,
                                n_configurations=len(values), estimate=values.mean(),
                                ci_low=ci.low, ci_high=ci.high, p=test.pvalue,
                                method='Configuration means'))
    return pd.DataFrame(records)


def seed_paired_reference(delta, layout):
    records = []
    for row in layout.itertuples():
        group = delta.loc[selector_for_layout(delta, row.block, row.row)]
        for metric in METRICS:
            values = group[metric].to_numpy()
            test = stats.ttest_1samp(values, 0)
            ci = test.confidence_interval(.95)
            records.append(dict(block=row.block, row=row.row, y=row.y, metric=metric,
                                n_seed_differences=len(values), estimate=values.mean(),
                                ci_low=ci.low, ci_high=ci.high, p=test.pvalue,
                                method='Seed paired, unblocked'))
    return pd.DataFrame(records)


def calculate(data_dir):
    layout, _ = performance_data(data_dir)
    layout = layout.loc[layout.schedule.eq('Matched daily')]
    raw = load_schedule_seed_data(data_dir, 'Random')
    delta = paired_differences(raw, 'RR', 'DD')
    records = []
    for row in layout.itertuples():
        for record in blocked_summary(delta, selector_for_layout(delta, row.block, row.row)):
            records.append(dict(block=row.block, row=row.row, y=row.y,
                                contrast='DD - RR' if record['metric'] == 'Trades' else 'RR - DD',
                                method='Seed paired, blocked', **record))
    result = annotate(pd.DataFrame(records), 'block')
    result['p_holm_all_44'] = annotate(result, lambda _: True).p_holm
    return result, configuration_reference(data_dir, layout), seed_paired_reference(delta, layout)


def plot_blocked(result):
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 8.5), sharey=True)
    fig.subplots_adjust(left=.15, right=.90, top=.85, bottom=.085, wspace=.32)
    fig.suptitle('Random scheduling: seed-paired, blocked 95% confidence intervals',
                 x=.035, y=.97, ha='left', fontsize=19, fontweight='bold', color=INK)
    titles = ['Return change (pp)', 'Sharpe change', 'Drawdown change (pp)', 'Trades avoided']
    for ax, metric, title in zip(axes, METRICS, titles):
        group = result.loc[result.metric.eq(metric)]
        ax.axhspan(-.45, .45, color='#EAF3F8', zorder=0)
        ax.axvline(0, color='#76838C', lw=1)
        for record in group.itertuples():
            color = BLUE if record.estimate >= 0 else '#C43B3B'
            ax.errorbar(record.estimate, record.y,
                        xerr=[[record.estimate-record.ci_low], [record.ci_high-record.estimate]],
                        fmt='o', color=color, ecolor=color, markersize=5, capsize=3, elinewidth=1.5, zorder=3)
            annotation = f'{record.stars}\n{record.daggers}'
            if annotation.strip():
                ax.annotate(annotation, (1.02, record.y), xycoords=(ax.transAxes, ax.transData),
                            va='center', ha='left', fontsize=8.5, color='#202B33', linespacing=.72)
        lo, hi = min(0, group.ci_low.min()), max(0, group.ci_high.max())
        span = hi-lo
        ax.set_xlim(lo-.07*span, hi+.07*span)
        ax.set_ylim(13.85, -.7)
        labels = group.row.replace({'Indicators + crossovers': 'Indicators & crossovers'})
        ax.set_yticks(group.y, labels)
        ax.tick_params(axis='y', length=0, pad=12, labelsize=11)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='x', color='#E5E9EC', lw=.7)
        ax.set_axisbelow(True)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B9C1C7')
    for y, label in [(1, 'REPRESENTATION'), (3.8, 'INDICATOR FAMILY'), (6.6, 'ALGORITHM'), (10.4, 'EVALUATION WINDOW')]:
        axes[0].text(-.51, y, label, transform=axes[0].get_yaxis_transform(), fontsize=9,
                     color='#566670', fontweight='bold', clip_on=False)
    return fig


def plot_comparison(config, seed, blocked):
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 8.5), sharey=True)
    fig.subplots_adjust(left=.15, right=.98, top=.85, bottom=.085, wspace=.32)
    fig.suptitle('Random scheduling: how uncertainty changes with seed pairing and blocking',
                 x=.035, y=.97, ha='left', fontsize=19, fontweight='bold', color=INK)
    legend = [('#78848D', 'Configuration means'), ('#D48624', 'Seed paired, unblocked'), (BLUE, 'Seed paired, blocked')]
    fig.legend(handles=[Line2D([0], [0], color=color, marker='o', label=label) for color, label in legend],
               loc='upper left', bbox_to_anchor=(.15, .935), ncol=3, frameon=False, fontsize=10)
    titles = ['Return change (pp)', 'Sharpe change', 'Drawdown change (pp)', 'Trades avoided']
    for ax, metric, title in zip(axes, METRICS, titles):
        groups = [frame.loc[frame.metric.eq(metric)] for frame in (config, seed, blocked)]
        ax.axhspan(-.45, .45, color='#EAF3F8', zorder=0)
        ax.axvline(0, color='#76838C', lw=1)
        for group, (color, _), offset in zip(groups, legend, (-.13, 0, .13)):
            for record in group.itertuples():
                ax.errorbar(record.estimate, record.y+offset,
                            xerr=[[record.estimate-record.ci_low], [record.ci_high-record.estimate]],
                            fmt='o', color=color, ecolor=color, markersize=3.5, capsize=2, elinewidth=1.2, zorder=3)
        lo = min(0, *(group.ci_low.min() for group in groups))
        hi = max(0, *(group.ci_high.max() for group in groups))
        span = hi-lo
        ax.set_xlim(lo-.07*span, hi+.07*span)
        ax.set_ylim(13.85, -.7)
        labels = groups[-1].row.replace({'Indicators + crossovers': 'Indicators & crossovers'})
        ax.set_yticks(groups[-1].y, labels)
        ax.tick_params(axis='y', length=0, pad=12, labelsize=11)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='x', color='#E5E9EC', lw=.7)
        ax.set_axisbelow(True)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B9C1C7')
    for y, label in [(1, 'REPRESENTATION'), (3.8, 'INDICATOR FAMILY'), (6.6, 'ALGORITHM'), (10.4, 'EVALUATION WINDOW')]:
        axes[0].text(-.51, y, label, transform=axes[0].get_yaxis_transform(), fontsize=9,
                     color='#566670', fontweight='bold', clip_on=False)
    return fig


def plot_seed_comparison(seed, blocked):
    """Compare seed pairing alone with seed pairing plus shared-schedule blocks."""
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 8.5), sharey=True)
    fig.subplots_adjust(left=.15, right=.98, top=.85, bottom=.085, wspace=.32)
    fig.suptitle('Random scheduling: seed pairing with and without shared-schedule blocking',
                 x=.035, y=.97, ha='left', fontsize=16, fontweight='bold', color=INK)
    legend = [('#D48624', 'Seed paired, unblocked'), (BLUE, 'Seed paired, blocked')]
    fig.legend(handles=[Line2D([0], [0], color=color, marker='o', label=label) for color, label in legend],
               loc='upper left', bbox_to_anchor=(.15, .935), ncol=2, frameon=False, fontsize=10)
    titles = ['Return change (pp)', 'Sharpe change', 'Drawdown change (pp)', 'Trades avoided']
    for ax, metric, title in zip(axes, METRICS, titles):
        groups = [frame.loc[frame.metric.eq(metric)] for frame in (seed, blocked)]
        ax.axhspan(-.45, .45, color='#EAF3F8', zorder=0)
        ax.axvline(0, color='#76838C', lw=1)
        for group, (color, _), offset in zip(groups, legend, (-.08, .08)):
            for record in group.itertuples():
                ax.errorbar(record.estimate, record.y+offset,
                            xerr=[[record.estimate-record.ci_low], [record.ci_high-record.estimate]],
                            fmt='o', color=color, ecolor=color, markersize=4, capsize=2.5, elinewidth=1.3, zorder=3)
        lo = min(0, *(group.ci_low.min() for group in groups))
        hi = max(0, *(group.ci_high.max() for group in groups))
        span = hi-lo
        ax.set_xlim(lo-.07*span, hi+.07*span)
        ax.set_ylim(13.85, -.7)
        labels = groups[-1].row.replace({'Indicators + crossovers': 'Indicators & crossovers'})
        ax.set_yticks(groups[-1].y, labels)
        ax.tick_params(axis='y', length=0, pad=12, labelsize=11)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='x', color='#E5E9EC', lw=.7)
        ax.set_axisbelow(True)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B9C1C7')
    for y, label in [(1, 'REPRESENTATION'), (3.8, 'INDICATOR FAMILY'), (6.6, 'ALGORITHM'), (10.4, 'EVALUATION WINDOW')]:
        axes[0].text(-.51, y, label, transform=axes[0].get_yaxis_transform(), fontsize=9,
                     color='#566670', fontweight='bold', clip_on=False)
    return fig


def generate(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result, config, seed = calculate(data_dir)
    result.to_csv(output_dir/'random_schedule_intervals.csv', index=False)
    comparison = pd.concat([config, seed, result], ignore_index=True, sort=False)
    comparison.to_csv(output_dir/'random_schedule_interval_method_comparison.csv', index=False)
    for stem, figure in [('random_schedule_intervals', plot_blocked(result)),
                         ('random_schedule_interval_method_comparison', plot_comparison(config, seed, result)),
                         ('random_schedule_interval_seed_block_comparison', plot_seed_comparison(seed, result))]:
        for ext in ['png', 'pdf', 'svg']:
            figure.savefig(output_dir/f'{stem}.{ext}', dpi=180, facecolor='white')
        plt.close(figure)
    notes = ('Pointwise 95% Student-t intervals use seed-paired differences. Within each family–window–seed '
             'block, outcomes sharing a random schedule are averaged before inference; all configurations retain '
             'equal weight. RR-DD is used for performance and DD-RR for trades avoided. Stars show unadjusted '
             'thresholds (.05/.01/.001); daggers below show Holm-adjusted thresholds within the named aggregation '
             'family. The intervals are not multiplicity-adjusted or simultaneous. Shared market windows remain fixed.')
    (output_dir/'random_schedule_caption_notes.txt').write_text(notes+'\n')
    seed_file = data_dir/'2x2/random2x2_train_eval_sequence_seed_mean_metrics.csv'
    provenance = {'source': str(seed_file), 'source_sha256': hashlib.sha256(seed_file.read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'scipy': scipy.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
                  'checks': ['720 seed rows', '180 paired seed differences', 'shared-schedule block averages', '44 result rows'],
                  'interpretation': notes}
    (output_dir/'random_schedule_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent/'tables_mv2boll')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent.parent/'imgs/mv2boll/descriptive')
    args = parser.parse_args()
    print(generate(args.data_dir, args.output_dir).round(5).to_string(index=False))
