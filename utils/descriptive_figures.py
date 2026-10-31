"""Descriptive scheduling figures. Requires numpy, pandas and matplotlib.

Run independently of legacy notebook cells and model caches:
    python descriptive_figures.py --data-dir tables_mv2boll --output-dir imgs/mv2boll/descriptive
Use --retention-only to preserve all existing performance outputs.

All performance values are equal-weight means of exported per-seed metric
means. No inference or continuous-portfolio metric is computed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

METRICS = ['Return%_mean', 'Sharpe_mean', 'MaxDD%_mean', 'Trades_mean']
WINDOWS = {'w1': '2020', 'w4': '2022', 'w3': '2025'}
ALGOS = ['a2c', 'ppo', 'sac']
DATASETS = {'mvx2': ('Moving average', 'Indicators + crossovers'),
            'mvx2_trg': ('Moving average', 'Crossovers'),
            'bollx': ('Bollinger', 'Indicators + crossovers'),
            'bollx_trg': ('Bollinger', 'Crossovers')}
BLUE, GREY, INK = '#0072B2', '#8A9299', '#202B33'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def performance_data(data_dir):
    raw = pd.read_csv(data_dir / 'metrics_table.csv')
    parts = []
    for dataset, (family, representation) in DATASETS.items():
        for prefix, schedule in [('', 'Matched daily'), ('flt_', 'Crossover scheduling')]:
            part = raw.loc[raw.dataset.eq(prefix + dataset)].copy()
            part['family'], part['representation'], part['schedule'] = family, representation, schedule
            parts.append(part)
    selected = pd.concat(parts, ignore_index=True)
    bench = raw.loc[raw.dataset.eq('all')].copy()
    expected = {(w, a) for w in WINDOWS for a in ALGOS}
    for dataset, group in selected.groupby('dataset'):
        require(len(group) == 9 and set(zip(group.window, group.algo)) == expected,
                f'Incomplete or duplicate evaluation grid: {dataset}')
    require(len(bench) == 9 and set(zip(bench.window, bench.algo)) == expected,
            'Daily benchmark must contain exactly the same 3 windows x 3 algorithms')
    for frame in [selected, bench]:
        require(np.isfinite(frame[METRICS].to_numpy()).all(), 'Missing/nonfinite per-seed mean metrics')
        require(frame.Seeds.eq(5).all(), 'Expected five seeds per evaluation')
    keys = ['family', 'representation', 'window', 'algo']
    paired = selected.query("schedule == 'Matched daily'").merge(
        selected.query("schedule == 'Crossover scheduling'"), on=keys,
        suffixes=('_daily', '_scheduled'), validate='one_to_one')
    require(len(paired) == 36, 'Expected 36 matched configuration pairs')
    specs = [('Overall', 'Overall', None, None, 0),
             ('Indicator family', 'Moving average', 'family', 'Moving average', 1.8),
             ('Indicator family', 'Bollinger', 'family', 'Bollinger', 2.8),
             ('Algorithm', 'A2C', 'algo', 'a2c', 4.6),
             ('Algorithm', 'PPO', 'algo', 'ppo', 5.6),
             ('Algorithm', 'SAC', 'algo', 'sac', 6.6),
             ('Evaluation window', '2020', 'window', 'w1', 8.4),
             ('Evaluation window', '2022', 'window', 'w4', 9.4),
             ('Evaluation window', '2025', 'window', 'w3', 10.4)]
    specs = [specs[0],
             ('Representation', 'Indicators + crossovers', 'representation', 'Indicators + crossovers', 1.8),
             ('Representation', 'Crossovers', 'representation', 'Crossovers', 2.8)] + [
             (*row[:4], row[4]+2.8) for row in specs[1:]]
    records = []
    for block, label, column, value, y in specs:
        subset = selected if column is None else selected.loc[selected[column].eq(value)]
        b = bench.loc[bench[column].eq(value)] if column in ['algo', 'window'] else bench
        for schedule in ['Matched daily', 'Crossover scheduling', 'Daily DRL benchmark']:
            group = b if schedule == 'Daily DRL benchmark' else subset.loc[subset.schedule.eq(schedule)]
            records.append(dict(block=block, row=label, y=y, schedule=schedule,
                                n_configurations=len(group), **group[METRICS].mean().to_dict()))
    summary = pd.DataFrame(records)
    # Balanced-grid identity: mean pairwise changes equals difference of means.
    overall = summary.loc[summary.row.eq('Overall')].set_index('schedule')
    for metric in METRICS:
        require(np.isclose((paired[metric + '_scheduled'] - paired[metric + '_daily']).mean(),
                           overall.loc['Crossover scheduling', metric] - overall.loc['Matched daily', metric]),
                'Pairing/weighting identity failed')
    # Independently established rounded descriptive checkpoints.
    require(np.allclose(overall.loc['Matched daily', METRICS[:3]].astype(float),
                        [11.320, .5835, -19.2752], atol=.0001), 'Daily checkpoint mismatch')
    require(np.allclose(overall.loc['Crossover scheduling', METRICS].astype(float),
                        [12.7712, .715944, -16.727944, 606.3], atol=.0001), 'Crossover checkpoint mismatch')
    return summary, selected


def retention_data(data_dir):
    # These exports are full-dataset counts, NOT independent window estimates.
    tables = {w: pd.read_csv(data_dir / w / 'data_reduction.csv').set_index('Dataset') for w in WINDOWS}
    states = {w: pd.read_csv(data_dir / w / 'state_space_reduction.csv').set_index('Dataset') for w in WINDOWS}
    for w in WINDOWS:
        require(tables[w].equals(tables['w1']) and states[w].equals(states['w1']),
                'Input-count exports changed across windows: reassess scope before plotting')
    dates, state = tables['w1'], states['w1']
    baseline_dates = int(dates.loc['all', 'Timesteps'])
    baseline_dim = 361  # Author-confirmed ten-indicator benchmark; supersedes export's 301.
    require(baseline_dates == 4275, 'Baseline date count changed; reassess provenance')
    records = []
    for dataset, (family, representation) in DATASETS.items():
        scheduled = 'flt_' + dataset
        date_source = scheduled if scheduled in dates.index else 'flt_bollx'
        state_source = scheduled if scheduled in state.index else dataset
        n_dates = int(dates.loc[date_source, 'Timesteps'])
        dim = int(state.loc[state_source, 'State_Space'])
        require(dim == 1 + 30 * (2 + (4 if representation == 'Indicators + crossovers' else 2)),
                'Unexpected state dimension')
        retained = n_dates * dim / (baseline_dates * baseline_dim)
        require(0 < retained <= 1, 'Invalid input-volume ratio')
        records.append(dict(dataset=scheduled, family=family, representation=representation,
                            dates=n_dates, state_dimension=dim, baseline_dates=baseline_dates,
                            baseline_state_dimension=baseline_dim,
                            retained_pct=100 * retained,
                            date_retention_pct=100 * n_dates / baseline_dates,
                            state_retention_pct=100 * dim / baseline_dim,
                            dates_source=date_source, state_source=state_source,
                            inferred=scheduled not in dates.index or scheduled not in state.index,
                            scope='full-dataset export estimate'))
    out = pd.DataFrame(records)
    require(np.isclose(out.retained_pct.mean(), 21.6381720691), 'Retention checkpoint mismatch')
    require(np.allclose(out.date_retention_pct * out.state_retention_pct / 100,
                        out.retained_pct), 'Retention factorization failed')
    return out


def performance_changes(summary):
    daily = summary.loc[summary.schedule.eq('Matched daily')].set_index('row')
    scheduled = summary.loc[summary.schedule.eq('Crossover scheduling')].set_index('row')
    changes = daily[['block', 'y']].copy()
    changes[METRICS] = scheduled[METRICS] - daily[METRICS]
    changes['Trades_avoided'] = -changes['Trades_mean']
    return changes.reset_index()


def plot_performance(summary):
    changes = performance_changes(summary)
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 8.5), sharey=True)
    fig.subplots_adjust(left=.145, right=.985, top=.95, bottom=.085, wspace=.30)
    titles = ['Return change (pp)', 'Sharpe change', 'Drawdown change (pp)', 'Trades avoided']
    plot_metrics = METRICS[:3] + ['Trades_avoided']
    for ax, metric, title in zip(axes, plot_metrics, titles):
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14)
        ax.axhspan(-.45, .45, color='#EAF3F8', zorder=0)
        values = changes[metric]
        low, high = min(0, values.min()), max(0, values.max())
        span = high-low
        ax.set_xlim(low-.24*span, high+.38*span)
        ax.axvline(0, color='#7D8991', lw=1.2)
        for _, row in changes.iterrows():
            label, y, value = row['row'], row['y'], row[metric]
            ax.barh(y, value, height=.22, color=BLUE if value>=0 else '#C43B3B', zorder=3)
            text = f'{value:+.3f}' if metric == 'Sharpe_mean' else f'{value:+.2f}'
            ax.annotate(text, (value,y), xytext=(6 if value>=0 else -6,0), textcoords='offset points',
                        ha='left' if value>=0 else 'right', va='center', fontsize=10,
                        color=INK, fontweight='bold' if label=='Overall' else 'normal')
        ax.set_ylim(13.85,-.7)
        ax.set_yticks(changes.y,changes.row.replace({'Indicators + crossovers':'Indicators & crossovers'}))
        ax.tick_params(axis='y',length=0,labelsize=11,pad=12)
        ax.tick_params(axis='x',labelsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='x',color='#E4E8EB',lw=.7); ax.set_axisbelow(True)
        for side in ['top','right','left']: ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B9C1C7')
    for y, label in [(1.0,'REPRESENTATION'),(3.8,'INDICATOR FAMILY'),(6.6,'ALGORITHM'),(10.4,'EVALUATION WINDOW')]:
        axes[0].text(-.51,y,label,transform=axes[0].get_yaxis_transform(),fontsize=9,
                     color='#566670',fontweight='bold',clip_on=False)
    return fig


def plot_retention(retention):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.8), sharey=True,
                             gridspec_kw={'width_ratios': [3, 1.05, 1.25]})
    fig.subplots_adjust(left=.32, right=.99, top=.78, bottom=.16, wspace=.12)
    bar, dates, state = axes
    y = np.array([0,1,2.6,3.6])
    bar.barh(y, retention.retained_pct, color=BLUE, height=.56)
    for pos, row in zip(y, retention.itertuples()):
        bar.text(row.retained_pct + 1, pos, f'{row.retained_pct:.2f}%',
                 ha='left', va='center', color=INK, fontsize=12, fontweight='bold')
        dates.text(.5, pos, f'{row.date_retention_pct:.2f}',
                   transform=dates.get_yaxis_transform(), ha='center', va='center', fontsize=13)
        state.text(.5, pos, f'{row.state_retention_pct:.2f}',
                   transform=state.get_yaxis_transform(), ha='center', va='center', fontsize=13)
    bar.set_yticks(y, ['Indicators & crossovers', 'Crossovers', 'Indicators & crossovers', 'Crossovers'])
    bar.tick_params(axis='y', length=0, pad=9, labelsize=13)
    bar.set_xlim(0, 44); bar.set_ylim(4.2,-.9); bar.set_xticks([0,20,40])
    bar.tick_params(axis='x', labelsize=12)
    bar.set_xlabel('Input retention (%)', fontsize=13, labelpad=8)
    dates.set_title('Dates\nretained (%)', fontsize=12, pad=12)
    state.set_title('State width\nretained (%)', fontsize=12, pad=12)
    dates.set_axis_off(); state.set_axis_off()
    bar.text(0,-.60,'MOVING AVERAGE', fontsize=10, fontweight='bold', color='#566670')
    bar.text(0,1.98,'BOLLINGER', fontsize=10, fontweight='bold', color='#566670')
    for side in ['top','right','left']: bar.spines[side].set_visible(False)
    bar.spines['bottom'].set_color('#B9C1C7')
    return fig


def generate(data_dir, output_dir, retention_only=False):
    started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42, 'ps.fonttype':42})
    performance = None
    retention = retention_data(data_dir)
    csv_path = output_dir/'descriptive_input_retention_values.csv'
    retention.to_csv(csv_path, index=False)
    generated_paths = [csv_path]
    figures = [('descriptive_input_retention', plot_retention(retention))]
    if not retention_only:
        performance, _ = performance_data(data_dir)
        performance.to_csv(output_dir/'descriptive_performance_values.csv', index=False)
        performance_changes(performance).to_csv(output_dir/'descriptive_performance_changes.csv', index=False)
        generated_paths.extend(output_dir/name for name in
                               ['descriptive_performance_values.csv', 'descriptive_performance_changes.csv'])
        figures.append(('descriptive_performance', plot_performance(performance)))
    for name, fig in figures:
        for ext in ['png','pdf','svg']:
            path = output_dir/f'{name}.{ext}'
            fig.savefig(path, dpi=180, facecolor='white')
            generated_paths.append(path)
        plt.close(fig)
    inputs = [data_dir/w/f for w in WINDOWS for f in ['data_reduction.csv','state_space_reduction.csv']]
    if not retention_only:
        inputs.append(data_dir/'metrics_table.csv')
    manifest = dict(schema_version=1, claim_id='input-retention' if retention_only else 'descriptive-scheduling-figures',
        repository={'commit':'source data outside paper Git repository; see input hashes', 'dirty':True},
        command=sys.executable+' '+str(Path(__file__).resolve())+' --data-dir '+str(data_dir)+' --output-dir '+str(output_dir)+(' --retention-only' if retention_only else ''),
        environment={'software':[f'Python {platform.python_version()}',f'numpy {np.__version__}',f'pandas {pd.__version__}',f'matplotlib {matplotlib.__version__}'], 'hardware':platform.platform()},
        mathematics={'assertion_tested':'Four full-dataset scalar-entry retention ratios and their date/state-width factors'+('' if retention_only else '; balanced means over 36 matched configurations'),
            'coefficient_domain':'IEEE float64', 'conventions':'Full-dataset input counts; percentages; author-confirmed daily state dimension 361'+('' if retention_only else '; per-seed metric means; signed drawdown'),
            'inputs':[{'path':str(p),'sha256':sha(p)} for p in inputs],
            'bounds':({'configurations':4,'source_windows':list(WINDOWS)} if retention_only else {'pairs':36,'seeds_per_configuration':5,'windows':list(WINDOWS),'algorithms':ALGOS}),
            'non_claims':['No significance or population generality','No continuous-portfolio metrics','No training-window counts or compute savings']},
        randomness={'used':False,'generator':'','seed':None},
        run={'started_at':started,'runtime_seconds':time.perf_counter()-tick,'exit_status':0},
        outputs=[{'path':str(p),'sha256':sha(p)} for p in sorted(generated_paths)],
        checks=['State-dimension formula','Retention checkpoint','Multiplicative date/state-width identity']+([] if retention_only else ['Complete unique matched grid','Finite per-seed means; five seeds per configuration','Balanced pairing identity','Independent rounded checkpoints']),
        result='Retention checkpoints and bounded structural checks passed',
        residual_risks=['Full-dataset input counts, not per-window training counts','Bollinger Crossovers count/dimension inferred from shared schedule and representation','Export benchmark dimension 301 superseded by author-confirmed 361'])
    manifest['script_sha256'] = sha(__file__)
    if not retention_only:
        (output_dir/'descriptive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Every path that refreshes retention artifacts refreshes their dedicated
    # manifest too, including the full generation used by the notebook.
    retention_manifest = dict(manifest)
    retention_manifest.update(
        claim_id='input-retention',
        mathematics={**manifest['mathematics'],
                     'assertion_tested':'Four full-dataset scalar-entry retention ratios and their date/state-width factors',
                     'conventions':'Full-dataset input counts; percentages; author-confirmed daily state dimension 361',
                     'inputs':[entry for entry in manifest['mathematics']['inputs']
                               if Path(entry['path']).name != 'metrics_table.csv'],
                     'bounds':{'configurations':4,'source_windows':list(WINDOWS)}},
        outputs=[entry for entry in manifest['outputs']
                 if Path(entry['path']).name.startswith('descriptive_input_retention')],
        checks=['State-dimension formula','Retention checkpoint','Multiplicative date/state-width identity'])
    (output_dir/'descriptive_input_retention_manifest.json').write_text(json.dumps(retention_manifest,indent=2)+'\n')
    return performance, retention


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent/'tables_mv2boll')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent.parent/'imgs/mv2boll/descriptive')
    parser.add_argument('--retention-only', action='store_true', help='Generate only input-retention artifacts; preserve existing performance outputs')
    args = parser.parse_args()
    p,r = generate(args.data_dir,args.output_dir,retention_only=args.retention_only)
    if p is not None:
        print(p.loc[p.row.eq('Overall'), ['schedule']+METRICS].to_string(index=False))
    print(r[['family','representation','retained_pct']].to_string(index=False))
    print('Figures and audit exports:',args.output_dir)
