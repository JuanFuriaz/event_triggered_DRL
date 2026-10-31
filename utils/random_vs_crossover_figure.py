"""Direct random-versus-crossover, seed-paired blocked interval figures.

The random and crossover 2x2 exports have matching family/window/algorithm/
seed keys. This module constructs RR-EE at the matched seed level, averages
outcomes sharing a random schedule within family-window-seed blocks, and uses
the same pointwise Student-t intervals as the random-versus-daily figure.
Trades are reoriented as EE-RR so a positive value means fewer trades under
random scheduling.
"""

from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scipy

from utils.descriptive_figures import BLUE, INK, performance_data
from utils.random_schedule_figure import calculate as calculate_random_daily
from utils.seed_paired_stats import (KEYS, METRICS, annotate, blocked_summary,
                               load_schedule_seed_data, selector_for_layout)


RED = "#C43B3B"
ORANGE = "#D48624"
DIRECT_SERIES = "Random relative to crossover"
DAILY_SERIES = "Random relative to daily"


def direct_paired_differences(data_dir):
    """Return matched RR-EE seed differences, with trades shown as EE-RR."""
    crossover = load_schedule_seed_data(data_dir, "Crossover")
    random = load_schedule_seed_data(data_dir, "Random")

    crossover_daily = crossover.loc[crossover.cell.eq("DD")].set_index(KEYS)[METRICS].sort_index()
    random_daily = random.loc[random.cell.eq("DD")].set_index(KEYS)[METRICS].sort_index()
    if not crossover_daily.index.equals(random_daily.index):
        raise ValueError("Crossover and random DD keys do not match")
    if not np.allclose(crossover_daily.to_numpy(), random_daily.to_numpy()):
        raise ValueError("Crossover and random DD outcomes differ")

    rr = random.loc[random.cell.eq("RR")].set_index(KEYS)[METRICS].sort_index()
    ee = crossover.loc[crossover.cell.eq("EE")].set_index(KEYS)[METRICS].sort_index()
    if not rr.index.equals(ee.index) or len(rr) != 180:
        raise ValueError("Unmatched RR and EE seed pairs")

    delta = (rr - ee).reset_index()
    delta["Trades"] *= -1
    delta["representation"] = np.where(
        delta.family.str.endswith("_trg"), "Crossovers", "Indicators + crossovers"
    )
    delta["indicator"] = np.where(
        delta.family.str.startswith("mv"), "Moving average", "Bollinger"
    )
    delta["schedule_family"] = delta.family.str.removesuffix("_trg")
    return delta


def _layout(data_dir):
    layout, _ = performance_data(data_dir)
    return layout.loc[layout.schedule.eq("Matched daily")].copy()


def _blocked_results(delta, layout):
    """Calculate the 44 block summaries used in the interval-figure layout."""
    records = []
    for row in layout.itertuples():
        selected = selector_for_layout(delta, row.block, row.row)
        for record in blocked_summary(delta, selected):
            contrast = "EE - RR" if record["metric"] == "Trades" else "RR - EE"
            records.append(dict(
                block=row.block,
                row=row.row,
                y=row.y,
                contrast=contrast,
                method="Seed paired, blocked",
                **record,
            ))
    result = annotate(pd.DataFrame(records), "block")
    result["p_holm_all_44"] = annotate(result, lambda _: True).p_holm
    if len(result) != 44:
        raise ValueError(f"Expected 44 direct comparison results, received {len(result)}")
    return result


def calculate(data_dir):
    """Calculate the direct RR-EE, seed-paired blocked results."""
    data_dir = Path(data_dir)
    return _blocked_results(direct_paired_differences(data_dir), _layout(data_dir))


def calculate_comparison(data_dir, direct=None):
    """Stack direct RR-EE and existing RR-DD results for the no-symbol plot."""
    if direct is None:
        direct = calculate(data_dir)
    daily, _, _ = calculate_random_daily(data_dir)

    direct = direct.copy()
    daily = daily.copy()
    direct["series"] = DIRECT_SERIES
    daily["series"] = DAILY_SERIES

    shared = ["block", "row", "y", "metric", "n_blocks", "n_seed_differences"]
    direct_layout = direct[shared].sort_values(shared).reset_index(drop=True)
    daily_layout = daily[shared].sort_values(shared).reset_index(drop=True)
    if not direct_layout.equals(daily_layout):
        raise ValueError("Direct and random-daily interval layouts do not match")
    return pd.concat([direct, daily], ignore_index=True, sort=False)


def _style_axis(axis, group, title):
    lo = min(0, group.ci_low.min())
    hi = max(0, group.ci_high.max())
    span = hi - lo
    if span == 0:
        span = 1
    axis.set_xlim(lo - .07 * span, hi + .07 * span)
    axis.set_ylim(13.85, -.7)
    labels = group.drop_duplicates("row").sort_values("y").row.replace(
        {"Indicators + crossovers": "Indicators & crossovers"}
    )
    y = group.drop_duplicates("row").sort_values("y").y
    axis.set_yticks(y, labels)
    axis.tick_params(axis="y", length=0, pad=12, labelsize=11)
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=14)
    axis.axhspan(-.45, .45, color="#EAF3F8", zorder=0)
    axis.axvline(0, color="#76838C", lw=1)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.grid(axis="x", color="#E5E9EC", lw=.7)
    axis.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#B9C1C7")


def _add_group_labels(axis):
    for y, label in [
        (1, "REPRESENTATION"),
        (3.8, "INDICATOR FAMILY"),
        (6.6, "ALGORITHM"),
        (10.4, "EVALUATION WINDOW"),
    ]:
        axis.text(-.51, y, label, transform=axis.get_yaxis_transform(), fontsize=9,
                  color="#566670", fontweight="bold", clip_on=False)


def plot_direct(result):
    """Plot direct RR-EE intervals with raw and Holm significance markers."""
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 8.5), sharey=True)
    fig.subplots_adjust(left=.15, right=.90, top=.94, bottom=.085, wspace=.32)
    titles = ["Return change (pp)", "Sharpe change", "Drawdown improvement (pp)", "Trades avoided"]
    for axis, metric, title in zip(axes, METRICS, titles):
        group = result.loc[result.metric.eq(metric)]
        _style_axis(axis, group, title)
        for record in group.itertuples():
            color = BLUE if record.estimate >= 0 else RED
            axis.errorbar(record.estimate, record.y,
                          xerr=[[record.estimate - record.ci_low],
                                [record.ci_high - record.estimate]],
                          fmt="o", color=color, ecolor=color, markersize=5,
                          capsize=3, elinewidth=1.5, zorder=3)
            annotation = f"{record.stars}\n{record.daggers}"
            if annotation.strip():
                axis.annotate(annotation, (1.02, record.y),
                              xycoords=(axis.transAxes, axis.transData),
                              va="center", ha="left", fontsize=8.5,
                              color="#202B33", linespacing=.72)
    _add_group_labels(axes[0])
    return fig


def plot_comparison(result):
    """Plot RR-EE and RR-DD intervals together without significance markers."""
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 8.5), sharey=True)
    fig.subplots_adjust(left=.15, right=.98, top=.86, bottom=.085, wspace=.32)
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, marker="o", label="Random relative to crossover"),
            Line2D([0], [0], color=ORANGE, marker="s", label="Random relative to daily"),
        ],
        loc="upper left", bbox_to_anchor=(.15, .995), ncol=2,
        frameon=False, fontsize=10,
    )
    styles = {
        DIRECT_SERIES: (BLUE, "o", -.12),
        DAILY_SERIES: (ORANGE, "s", .12),
    }
    titles = ["Return change (pp)", "Sharpe change", "Drawdown improvement (pp)", "Trades avoided"]
    for axis, metric, title in zip(axes, METRICS, titles):
        group = result.loc[result.metric.eq(metric)]
        _style_axis(axis, group, title)
        for series, (color, marker, offset) in styles.items():
            current = group.loc[group.series.eq(series)]
            for record in current.itertuples():
                axis.errorbar(record.estimate, record.y + offset,
                              xerr=[[record.estimate - record.ci_low],
                                    [record.ci_high - record.estimate]],
                              fmt=marker, color=color, ecolor=color, markersize=4.5,
                              capsize=2.5, elinewidth=1.3, zorder=3)
    _add_group_labels(axes[0])
    return fig


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(data_dir, output_dir):
    """Write direct-analysis CSVs, figures, caption notes and provenance."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direct = calculate(data_dir)
    comparison = calculate_comparison(data_dir, direct=direct)
    direct.to_csv(output_dir / "random_vs_crossover_intervals.csv", index=False)
    comparison.to_csv(output_dir / "random_vs_crossover_vs_daily_intervals.csv", index=False)
    for stem, figure in [
        ("random_vs_crossover_intervals", plot_direct(direct)),
        ("random_vs_crossover_vs_daily_intervals", plot_comparison(comparison)),
    ]:
        for extension in ["png", "pdf", "svg"]:
            figure.savefig(output_dir / f"{stem}.{extension}", dpi=180, facecolor="white")
        plt.close(figure)

    direct_notes = (
        "Pointwise 95% Student-t intervals use direct seed-paired RR-EE differences. "
        "Within each family-window-seed block, the three algorithms and two "
        "representations sharing the random schedule are averaged before inference. "
        "Return, Sharpe and drawdown show RR-EE; trades show EE-RR, so positive "
        "values consistently favor random scheduling. Stars show unadjusted thresholds "
        "(.05/.01/.001); daggers show Holm-adjusted thresholds within the named "
        "aggregation family. Intervals are pointwise, not multiplicity-adjusted or simultaneous."
    )
    comparison_notes = (
        "Each pointwise 95% Student-t interval uses the same seed-paired blocked "
        "calculation as its source figure. Blue circles compare random with crossover "
        "scheduling; orange squares compare random with daily scheduling. Return, Sharpe "
        "and drawdown use RR-EE and RR-DD; trades use EE-RR and DD-RR. Positive values "
        "therefore favor random scheduling in every panel. This comparison figure has no "
        "significance symbols; intervals are pointwise and not multiplicity-adjusted."
    )
    (output_dir / "random_vs_crossover_caption_notes.txt").write_text(direct_notes + "\n")
    (output_dir / "random_vs_crossover_vs_daily_caption_notes.txt").write_text(
        comparison_notes + "\n"
    )

    crossover_file = data_dir / "2x2/2x2_train_eval_sequence_seed_mean_metrics.csv"
    random_file = data_dir / "2x2/random2x2_train_eval_sequence_seed_mean_metrics.csv"
    provenance = {
        "sources": [
            {"path": str(crossover_file), "sha256": _sha256(crossover_file)},
            {"path": str(random_file), "sha256": _sha256(random_file)},
        ],
        "script_sha256": _sha256(__file__),
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "checks": [
            "720 seed rows in each source table",
            "180 matched RR-EE seed pairs",
            "identical DD source outcomes",
            "shared-schedule block averages",
            "44 direct results",
            "44 direct plus 44 random-daily comparison rows",
        ],
        "outputs": [
            "random_vs_crossover_intervals.csv",
            "random_vs_crossover_intervals.png",
            "random_vs_crossover_intervals.pdf",
            "random_vs_crossover_intervals.svg",
            "random_vs_crossover_vs_daily_intervals.csv",
            "random_vs_crossover_vs_daily_intervals.png",
            "random_vs_crossover_vs_daily_intervals.pdf",
            "random_vs_crossover_vs_daily_intervals.svg",
            "random_vs_crossover_caption_notes.txt",
            "random_vs_crossover_vs_daily_caption_notes.txt",
        ],
        "interpretation": direct_notes,
    }
    (output_dir / "random_vs_crossover_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return direct, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "tables_mv2boll")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "imgs/mv2boll/descriptive")
    args = parser.parse_args()
    direct, comparison = generate(args.data_dir, args.output_dir)
    print(direct.round(5).to_string(index=False))
    print(f"Generated {len(direct)} direct and {len(comparison)} comparison rows.")
