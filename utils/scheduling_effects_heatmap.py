"""Seed-paired, blocked 2x2 and fixed-policy scheduling contrasts."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
import numpy as np
import pandas as pd

from utils.seed_paired_stats import (METRICS, annotate, blocked_summary, load_schedule_seed_data,
                               paired_differences)

CONCEPTS = [
    ('Do daily-trained agents also benefit\nfrom scheduled trading?', 'DE', 'DD', 'DR', 'DD'),
    ('Does scheduled training help\nwhen trading remains daily?', 'ED', 'DD', 'RD', 'DD'),
    ('Do schedule-trained agents perform better \nwith scheduled trading than with daily trading?', 'EE', 'ED', 'RR', 'RD'),
    ('Do schedule-trained agents perform better \nthan daily-trained ones with scheduled trading?', 'EE', 'DE', 'RR', 'DR'),
    ('What is the complete scheduling effect\nof scheduled training and trading?', 'EE', 'DD', 'RR', 'DD'),
    ('Does restricting trading help the\ndaily-trained indicator baseline?', None, None, None, None),
    ('Does restricting trading help the\ndaily benchmark?', None, None, None, None),
]
# Rows before this index are paired 2x2 contrasts (four components and the total);
# the remaining rows are the fixed-policy daily-baseline tests.
N_2X2_ROWS = 5
N_ROWS = len(CONCEPTS)
PROFILES = [('Crossover', 'Indicators + crossovers'), ('Random', 'Indicators + crossovers'),
            ('Crossover', 'Crossovers'), ('Random', 'Crossovers')]


class RaisedVPacker(VPacker):
    """Keep the compact significance column just above the estimate."""

    def __init__(self, *args, rise_px=3, **kwargs):
        self.rise_px = rise_px
        super().__init__(*args, **kwargs)

    def set_offset(self, xy):
        if callable(xy):
            return super().set_offset(xy)
        x, y = xy
        return super().set_offset((x, y + self.rise_px))


def cell_annotation(effect, stars, daggers, textprops):
    """Pack the raw and Holm markers beside an effect, without detached rows."""
    base_size = textprops.get('fontsize', 9)
    effect_props = {**textprops, 'fontsize': base_size * 1.25}
    marker_props = {**textprops, 'fontsize': base_size * .80}
    effect_text = TextArea(effect, textprops=effect_props)
    if not stars:
        return effect_text
    marker_children = [TextArea(stars, textprops=marker_props)]
    if daggers:
        marker_children.append(TextArea(daggers, textprops=marker_props))
    marker_stack = RaisedVPacker(children=marker_children, align='right', sep=-2, pad=0, rise_px=3)
    return HPacker(children=[effect_text, marker_stack], align='top', sep=4, pad=0)


def control_differences(data_dir, policy, schedule):
    raw = pd.read_csv(Path(data_dir) / 'ladder/scheduler_ladder_seed_controls_placebo.csv')
    if policy == 'Indicator baseline':
        reference = raw.loc[raw.stage.eq('daily baseline')]
        target_stage = 'baseline restricted to selected dates' if schedule == 'Crossover' else 'baseline restricted to random dates'
        left = raw.loc[raw.stage.eq(target_stage)]
        keys = ['family', 'window', 'algo', 'seed']
        right = reference
    else:
        reference = raw.loc[raw.stage.eq('daily benchmark')]
        target_stage = ('daily benchmark restricted to selected dates' if schedule == 'Crossover'
                        else 'daily benchmark restricted to random dates')
        left = raw.loc[raw.stage.eq(target_stage)]
        keys = ['window', 'algo', 'seed']
        right = reference
    left = left.copy()
    left_indexed = left.set_index(keys).sort_index()
    a = left_indexed[METRICS]
    b = right.set_index(keys)[METRICS].sort_index()
    if policy == 'Daily RL benchmark':
        b = b.reindex(a.index.droplevel('family') if 'family' in a.index.names else a.index)
        b.index = a.index
    if not a.index.equals(b.index):
        raise ValueError(f'Unmatched ladder pairs for {policy}/{schedule}')
    out = (a-b).reset_index()
    if policy == 'Daily RL benchmark':
        # The benchmark itself has family='all'; the restricted evaluation
        # inherits the moving-average/Bollinger schedule family from `left`.
        out['family'] = left_indexed['family'].to_numpy()
    out['Trades'] *= -1
    out['schedule_family'] = out.family
    return out


def calculate(data_dir):
    sources = {schedule: load_schedule_seed_data(data_dir, schedule) for schedule in ['Crossover', 'Random']}
    rows = []
    for row_index, (concept, ea, eb, ra, rb) in enumerate(CONCEPTS):
        for profile_index, (schedule, representation) in enumerate(PROFILES):
            if row_index < N_2X2_ROWS:
                treatment, reference = (ea, eb) if schedule == 'Crossover' else (ra, rb)
                delta = paired_differences(sources[schedule], treatment, reference)
                delta = delta.loc[delta.representation.eq(representation)]
                records = blocked_summary(delta, np.ones(len(delta), dtype=bool))
                # `paired_differences` returns treatment minus reference, with
                # Trades reoriented so positive means fewer trades.  Keep the
                # exported label in the same direction as the shown estimate.
                contrast = f'{treatment} - {reference}'
            elif representation == 'Crossovers':
                rows.append(dict(row_index=row_index, profile_index=profile_index, concept=concept.replace('\n', ' '),
                                 schedule=schedule, representation=representation, contrast='Not applicable', metric=np.nan,
                                 n_blocks=np.nan, n_seed_differences=np.nan, estimate=np.nan, ci_low=np.nan, ci_high=np.nan,
                                 p=np.nan, status='Not applicable: fixed-policy control has no Crossovers-only counterpart'))
                continue
            else:
                policy = 'Indicator baseline' if row_index == N_2X2_ROWS else 'Daily RL benchmark'
                delta = control_differences(data_dir, policy, schedule)
                # The daily RL benchmark has no indicator family: the same daily reference
                # outcome is subtracted in both the moving-average and the Bollinger block.
                # Average the two families first, so the inference blocks are the 15 independent
                # window-seed combinations. The Indicator baseline has its own daily policy per
                # family and keeps the 30 family-window-seed blocks.
                blocks = (('window', 'seed') if policy == 'Daily RL benchmark'
                          else ('schedule_family', 'window', 'seed'))
                records = blocked_summary(delta, np.ones(len(delta), dtype=bool), block_columns=blocks)
                contrast = f'{policy}: restricted trading - daily trading'
            for record in records:
                metric = record['metric']
                shown_contrast = contrast
                rows.append(dict(row_index=row_index, profile_index=profile_index, concept=concept.replace('\n', ' '),
                                 schedule=schedule, representation=representation, contrast=shown_contrast,
                                 status='computed', **record))
    out = pd.DataFrame(rows)
    valid = out.p.notna()
    out.loc[valid, 'p_holm_concept'] = out.loc[valid].groupby(['row_index', 'schedule']).p.transform(
        lambda p: annotate(pd.DataFrame({'p': p}), lambda _: True).p_holm.to_numpy())
    out['stars'] = out.p.map(lambda p: '' if pd.isna(p) else '*' * sum(p < threshold for threshold in (.05, .01, .001)))
    out['daggers'] = out.p_holm_concept.map(lambda p: '' if pd.isna(p) else '†' * sum(p < threshold for threshold in (.05, .01, .001)))
    out.loc[valid, 'p_holm_all_available'] = annotate(out.loc[valid, ['p']].copy(), lambda _: True).p_holm.to_numpy()
    return out


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5))
    fig.subplots_adjust(left=.25, right=.945, top=.95, bottom=.07, wspace=.40, hspace=.46)
    cmap = LinearSegmentedColormap.from_list('effects', ['#B63138', '#FFFFFF', '#0072B2'])
    titles = ['Return change (pp)', 'Sharpe change', 'Drawdown change (pp)', 'Trades avoided']
    for index, (axis, metric, title) in enumerate(zip(axes.flat, METRICS, titles)):
        group = results.loc[results.metric.eq(metric)]
        matrix = group.pivot(index='row_index', columns='profile_index', values='estimate').reindex(index=range(N_ROWS), columns=range(4)).to_numpy()
        limit = np.nanmax(np.abs(matrix))
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
        axis.set_facecolor('#E9ECEF')
        for record in group.itertuples():
            effect = f'{record.estimate:+.3f}' if metric == 'Sharpe' else f'{record.estimate:+.1f}'
            color = 'white' if abs(record.estimate) > .60*limit else '#202B33'
            annotation = cell_annotation(effect, record.stars, record.daggers, {
                'color': color, 'fontsize': 9, 'fontweight': 'bold' if record.daggers else 'normal',
            })
            axis.add_artist(AnnotationBbox(annotation, (record.profile_index, record.row_index),
                                            xycoords='data', box_alignment=(.5, .5), frameon=False, pad=0))
        for row in range(N_2X2_ROWS, N_ROWS):
            for column in [2, 3]:
                axis.text(column, row, '–', ha='center', va='center', fontsize=12, color='#69757D')
        axis.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=12)
        axis.set_xticks(range(4), ['Crossover', 'Random', 'Crossover', 'Random'], fontsize=10)
        axis.set_yticks(range(N_ROWS), [item[0] for item in CONCEPTS] if index % 2 == 0 else ['']*N_ROWS, fontsize=9)
        axis.tick_params(length=0, pad=8)
        axis.set_xticks(np.arange(-.5, 4, 1), minor=True)
        axis.set_yticks(np.arange(-.5, N_ROWS, 1), minor=True)
        axis.grid(which='minor', color='white', lw=1)
        axis.tick_params(which='minor', length=0)
        axis.axvline(1.5, color='#7C8991', lw=1.6)
        axis.axhline(N_2X2_ROWS - 1.5, color='#7C8991', lw=1.2)   # components | total
        axis.axhline(N_2X2_ROWS - .5, color='#7C8991', lw=2)      # 2x2 contrasts | fixed-policy tests
        axis.axhline(N_2X2_ROWS + .5, color='#7C8991', lw=1.2)    # Indicator baseline | daily benchmark
        axis.text(.25, -.12, 'Indicators & crossovers', transform=axis.transAxes, ha='center', fontsize=10, fontweight='bold')
        axis.text(.75, -.12, 'Crossovers', transform=axis.transAxes, ha='center', fontsize=10, fontweight='bold')
        for spine in axis.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=axis, fraction=.035, pad=.035)
    return fig


def generate(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir)
    results.to_csv(output_dir/'scheduling_effects_heatmap_values.csv', index=False)
    figure = plot(results)
    for extension in ['png', 'pdf', 'svg']:
        figure.savefig(output_dir/f'scheduling_effects_heatmap.{extension}', dpi=180, facecolor='white')
    plt.close(figure)
    notes = ('Each comparison first forms seed-paired differences. A2C, PPO and SAC outcomes sharing one schedule '
             'are averaged within a family–window–seed block before inference. Pointwise 95% Student-t intervals and '
             'p-values are across these blocks. Stars are unadjusted thresholds (.05/.01/.001); daggers below are '
             'Holm-adjusted thresholds within each conceptual comparison and schedule. No interval is multiplicity-adjusted. '
             'Rows 1-4 separate training and trading; row 5 is their total effect (EE-DD, RR-DD). '
             'The final two rows hold a daily-trained policy fixed while restricting its trading dates; they do not establish '
             'that fewer trades alone cause gains. The Indicator baseline row uses 30 family-window-seed blocks. The daily '
             'benchmark row averages the two indicator families first, because both share one daily reference outcome, and '
             'uses 15 window-seed blocks.')
    (output_dir/'scheduling_effects_heatmap_caption.txt').write_text(notes+'\n')
    sources = [data_dir/'2x2/2x2_train_eval_sequence_seed_mean_metrics.csv',
               data_dir/'2x2/random2x2_train_eval_sequence_seed_mean_metrics.csv',
               data_dir/'ladder/scheduler_ladder_seed_controls_placebo.csv']
    (output_dir/'scheduling_effects_heatmap_provenance.json').write_text(json.dumps({
        'inputs': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'computed_tests': int(results.p.notna().sum()), 'not_applicable': 16, 'notes': notes}, indent=2)+'\n')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent/'tables_mv2boll')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent.parent/'imgs/mv2boll/descriptive')
    args = parser.parse_args()
    print(generate(args.data_dir, args.output_dir).round(5).to_string(index=False))
