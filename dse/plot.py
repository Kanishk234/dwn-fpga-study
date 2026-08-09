"""Pareto plots for Study 1.

Two figures, each answering a question the tables state but do not make obvious:

  frontier.png   accuracy vs area, with the Pareto frontier drawn through it. The deliverable
                 brief §10 asks for.
  area_split.png core vs encoder LUTs across the size ladder, stacked. This is the one that
                 matters most: the encoder costs 14x the core at `sm`, and every "% of Basys 3"
                 figure in the literature derived from core-only numbers understates the truth.
                 A single "total LUTs" bar would hide exactly that.

Colors come from the validated categorical palette (dataviz skill, references/palette.md).
Two constraints from it are load-bearing rather than cosmetic:

  - **Scatter uses the all-pairs rule, which caps at three categorical slots.** The grid has six
    groups; they fold to three (ladder / one-factor / group-B), which is also the distinction
    that means something. More slots would fail CVD separation.
  - **Aqua is below 3:1 on the light surface**, so the relief rule applies: frontier points
    carry visible direct labels, and dse/report.py provides the table view.

Usage:
    python dse/plot.py                  # writes into build/dse/
    python dse/plot.py --outdir docs/figures
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use('Agg')                      # no display on this machine; write files only
import matplotlib.pyplot as plt            # noqa: E402

from area_model import DEVICE_LUTS         # noqa: E402
from report import AREA, ACC, derive, load, pareto  # noqa: E402

# Validated categorical slots 1-3 (light mode). Do not extend past three for scatter.
SERIES = {'ladder': '#2a78d6', 'one-factor': '#eb6834', 'group-b': '#1baf7a'}
CORE_C, ENC_C = '#2a78d6', '#eb6834'

SURFACE = '#fcfcfb'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#dcdcd8'


def group_of(r):
    """Six grid groups folded to the three the scatter can carry (all-pairs CVD caps at 3).

    Reads the recorded grid group. It used to GUESS from the label -- "one-factor if the label
    contains a space" -- which silently mis-classified every multi-layer config, because `2x100`
    and `3x65` are ofat-L points whose labels have no space. They were coloured as ladder points
    and pulled into the ladder-only area chart.
    """
    g = r.get('group')
    if g:
        return {'ladder': 'ladder', 'group-b': 'group-b'}.get(g, 'one-factor')
    # Older records predate the stored group; fall back rather than crash.
    if r.get('pipe') != '1111' or r.get('clock_ns') != 10.0:
        return 'group-b'
    return 'one-factor' if ' ' in (r.get('label') or '') else 'ladder'


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc='left', pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    # Recessive grid and axes: the data is the figure, the frame is not.
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)


def plot_frontier(rows, path):
    ok = [r for r in rows if r['status'] == 'ok' and r.get(AREA) and r.get(ACC)]
    if not ok:
        print('  frontier: skipped, no config has both area and accuracy yet')
        return False

    fig, ax = plt.subplots(figsize=(11, 6.8), facecolor=SURFACE)
    style(ax, 'dwn_top LUTs (core + encoder)', 'accuracy (%)',
          'Accuracy vs area — JSC on XC7A35T')

    drawn = 0
    for name, color in SERIES.items():
        pts = [r for r in ok if group_of(r) == name]
        if pts:
            # >=8px markers, 2px surface ring so overlapping points stay separable.
            ax.scatter([r[AREA] for r in pts], [r[ACC] for r in pts], s=90, c=color,
                       label=name, zorder=3, edgecolors=SURFACE, linewidths=2)
            drawn += 1

    # The frontier is computed over GROUP A only. Group B varies register placement on an
    # already-trained model, so its points sit at the same accuracy as their rung and differ in
    # area by a handful of LUTs -- enough to technically dominate (3-stage `1x1600` is 15 LUTs
    # cheaper than the 4-stage one) while saying nothing about the accuracy/area tradeoff this
    # figure is about. They stay plotted, so the reader sees them; they just do not define the
    # curve or claim label space. Their real axis is latency, which report.py's 3-objective
    # frontier and the Group B table cover.
    front = pareto([r for r in ok if group_of(r) != 'group-b'])
    if len(front) > 1:
        # The frontier is an annotation, not a series, so it wears ink rather than a hue.
        # Recessive: the frontier is an annotation over the data, not a series competing with
        # it. Thin and pale, so the staircase reads as a boundary rather than as the subject.
        ax.step([r[AREA] for r in front], [r[ACC] for r in front], where='post',
                color=INK_2, linewidth=1.4, alpha=0.45, zorder=2, label='Pareto frontier')
    # EVERY frontier point is labelled, and only frontier points are.
    #
    # A previous version labelled only the ladder rungs, to cut crowding. That was the wrong
    # cut: it silently dropped `1x360 z=50`, `1x200 n=2` and three others -- precisely the
    # points that carry the study's arguments (z is nearly free; n=2 is not dominated). A
    # frontier point with no label is unreadable, because nothing else in the figure identifies
    # which config a dot is.
    #
    # The real cause of the crowding was canvas size, fixed by going to 11x6.8. Offsets
    # alternate above/below because a Pareto frontier is monotonic, so a fixed offset would
    # stack every label along one diagonal.
    for i, r in enumerate(front):
        above = i % 2 == 0
        ax.annotate(r['label'], (r[AREA], r[ACC]), textcoords='offset points',
                    xytext=(10, 6 if above else -13), fontsize=9, color=INK,
                    va='bottom' if above else 'top')

    ax.axvline(DEVICE_LUTS, color=INK_2, linewidth=1.5, linestyle=(0, (4, 3)), alpha=0.6)
    # Ceiling label at the BOTTOM of the line. At full grid size the top-right corner is where
    # the largest frontier point and its direct label land, and the two overlapped.
    ax.annotate(f'XC7A35T = {DEVICE_LUTS:,} LUTs', (DEVICE_LUTS, ax.get_ylim()[0]),
                textcoords='offset points', xytext=(-6, 8), fontsize=8,
                color=INK_2, ha='right', va='bottom')
    # Headroom so a direct label on the topmost point is not clipped by the axis.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.06 * (hi - lo))

    # Legend OUTSIDE the axes. Any in-axes corner is data at full grid size -- lower-left hid
    # the smallest frontier point once the sweep filled in, which is the corner a
    # minimize-area/maximize-accuracy frontier always reaches into.
    if drawn >= 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2,
                  loc='upper center', bbox_to_anchor=(0.5, -0.13),
                  ncol=4, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, REPO)}')
    return True


def plot_area_split(rows, path):
    # LADDER ONLY, deliberately. This figure's job is how the core/encoder split evolves with
    # MODEL SIZE. Plotting all 35 configs buried that: the one-factor variants sit at two fixed
    # widths and the five Group B configs have *identical* area to their base rung, so the chart
    # became 35 crammed labels with "12.6x" printed five times on top of itself. The ladder is
    # eight bars, readable, and is the comparison the figure is for.
    ok = [r for r in rows if r['status'] == 'ok' and r.get('dwn_core_luts') is not None
          and group_of(r) == 'ladder']
    ok.sort(key=lambda r: r.get('nodes') or 0)
    if not ok:
        print('  area_split: skipped, no synthesized ladder config yet')
        return False

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    style(ax, '', 'LUTs', 'Where the area goes — encoder vs core, across the size ladder')

    x = range(len(ok))
    core = [r['dwn_core_luts'] for r in ok]
    enc = [r.get('thermometer_encoder_luts') or 0 for r in ok]

    # Fixed bar width: with one or two configs matplotlib stretches bars across the whole axis,
    # which reads as a filled panel rather than a bar.
    ax.bar(x, core, width=0.6, color=CORE_C, label='core (LUT nodes + reduction)', zorder=3)
    # 2px surface gap between stacked segments, per the mark spec.
    ax.bar(x, enc, width=0.6, bottom=core, color=ENC_C, label='thermometer encoder', zorder=3,
           edgecolor=SURFACE, linewidth=2)
    ax.set_xlim(-0.7, len(ok) - 0.3)

    ax.set_xticks(list(x))
    rot = 30 if len(ok) > 6 else 0
    ax.set_xticklabels([r['label'] for r in ok], fontsize=8, color=INK_2,
                       rotation=rot, ha='right' if rot else 'center')
    for i, r in zip(x, ok):
        ratio = (r.get('thermometer_encoder_luts') or 0) / r['dwn_core_luts']
        ax.annotate(f'{ratio:.1f}x', (i, core[i] + enc[i]), textcoords='offset points',
                    xytext=(0, 5), ha='center', fontsize=8, color=INK)

    # The device ceiling must NOT set the y-scale. At `sm` the design is 1,619 LUTs against a
    # 20,800 ceiling, so anchoring the axis to the ceiling squashes the data into a sliver and
    # the 108-LUT core becomes invisible -- destroying the one thing this figure exists to show.
    # Scale to the data; draw the ceiling only when it is actually near, else say so in words.
    top = max(c + e for c, e in zip(core, enc))
    ax.set_ylim(0, top * 1.30)
    if DEVICE_LUTS <= top * 1.30:
        ax.axhline(DEVICE_LUTS, color=INK_2, linewidth=1.5, linestyle=(0, (4, 3)), alpha=0.6)
        # Label on the LEFT. The bars ascend, so the right end is where the tallest bar and its
        # ratio label sit -- and that bar is the one that breaks the ceiling, i.e. the whole
        # point of the figure. Never obscure it.
        ax.annotate(f'XC7A35T = {DEVICE_LUTS:,}', (-0.5, DEVICE_LUTS),
                    textcoords='offset points', xytext=(4, 5), ha='left',
                    fontsize=8, color=INK_2)
    else:
        ax.annotate(f'XC7A35T ceiling {DEVICE_LUTS:,} LUTs — above this scale '
                    f'(largest here is {top/DEVICE_LUTS:.0%} of it)',
                    (0.99, 0.99), xycoords='axes fraction', ha='right', va='top',
                    fontsize=8, color=INK_2)

    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, REPO)}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description='Pareto plots for the Phase 2 sweep.')
    ap.add_argument('--outdir', default=os.path.join(REPO, 'build', 'dse'))
    ap.add_argument('--results', help='read an alternate results.json')
    ap.add_argument('--snapshot', action='store_true',
                    help='write the figures into docs/results/ as committed evidence')
    args = ap.parse_args()
    outdir = os.path.join(REPO, 'docs', 'results') if args.snapshot else args.outdir
    os.makedirs(outdir, exist_ok=True)

    rows = derive(load(args.results))
    print(f'{len(rows)} result(s)')
    plot_frontier(rows, os.path.join(outdir, 'frontier.png'))
    plot_area_split(rows, os.path.join(outdir, 'area_split.png'))
    print('\nA one-point frontier is a dot, not a frontier. These fill in as 2c training and')
    print('2e synthesis land -- the figures are wired up now so the schema is settled first.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
