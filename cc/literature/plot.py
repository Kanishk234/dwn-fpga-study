"""Phase 3 comparison plots: accuracy vs LUTs, ONE FIGURE PER JSC DATASET.

Never one figure. JSC is two datasets ~1.05 pp apart (docs/jsc/phase3-ledger.md, 2026-08-10), so a
combined plot would show a gap that is partly the data and partly the design. This script refuses
to mix them, which is the whole point of it existing.

Marker fill encodes the accounting convention, because that is the other way these numbers lie:
    filled   the LUT count INCLUDES the input encoder (ours, DWN-PEN+FT)
    hollow   core only, encoder excluded (the DWN paper's own numbers)
    square   no separate encoder stage exists in that architecture

    .venv\\Scripts\\python.exe cc\\literature\\plot.py
    .venv\\Scripts\\python.exe cc\\literature\\plot.py --snapshot   # -> docs/jsc/results-cc/
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from table import (BENCHMARKS, load_conifer, load_hls4ml,        # noqa: E402
                   load_literature, load_ours, use_benchmark)

# Per-benchmark output dirs. JSC keeps docs/jsc/results-cc/ -- docs/jsc/report.md and the jsc-complete tag
# reference it. The two studies are NEVER plotted together: their accuracy scales differ by ~20
# points and their encoder economics run opposite (docs/mnist/phase2-ledger.md).
OUTDIRS = {'jsc':   (os.path.join(REPO, 'build', 'cc', 'literature'),
                     os.path.join(REPO, 'docs', 'jsc', 'results-cc')),
           'mnist': (os.path.join(REPO, 'build', 'cc', 'literature-mnist'),
                     os.path.join(REPO, 'docs', 'mnist', 'results-cc'))}
OUT = OUTDIRS['jsc'][0]
SNAP = OUTDIRS['jsc'][1]

# Which dataset variants get a figure, in order, and the filename prefix. JSC deliberately
# EXCLUDES 'unknown': those rows have no established dataset, so plotting them would invite
# exactly the comparison the study says cannot be made.
FIGURES = {'jsc': ('jsc', ('openml', 'cernbox')),
           'mnist': ('mnist', ('mnist',))}
DEVICE_LUTS = 20800

# Okabe-Ito plus four extensions, assigned in a FIXED order so a method keeps its colour
# across both figures. Colour is a secondary cue only -- every point is also directly
# labelled, so identity never rests on hue alone.
C = {
    # our silicon
    'this project': '#0072B2',
    'conifer (GBDT)': '#D55E00',
    'hls4ml (measured)': '#9467BD',
    # the DWN family
    'DWN': '#009E73',
    # published, OpenML
    'TreeLUT': '#CC79A7',
    'NeuraLUT-Assemble': '#E69F00',
    'FPGN': '#56B4E9',
    'hls4ml (Fahim et al.)': '#8B4513',
    'hls4ml (Duarte et al.)': '#7F7F7F',
    # published, CERNBox -- these were all falling through to grey, which made a nine-method
    # figure unreadable
    'PolyLUT': '#B22222',
    'PolyLUT-Add': '#FF7F0E',
    'NeuraLUT': '#1B9E77',
    'LogicNets': '#6A3D9A',
    'AmigoLUT-NeuraLUT-S': '#17BECF',
    'AmigoLUT-NeuraLUT-XS': '#005F73',
    'ReducedLUT': '#A6761D',
    'SparseLUT': '#E7298A',
    'LLNN': '#444444',
}
DEFAULT = '#666666'
_LBL = [0]        # rotates label offsets across series


def style(row):
    """(marker, filled) from the encoder convention."""
    e = row.get('encoder_included')
    if e is True:
        return 'o', True
    if e is False:
        return 'o', False
    if e == 'n/a':
        return 's', True
    return 'D', False


def pareto(rows):
    """Non-dominated by (low LUT, high accuracy)."""
    out, best = [], -1e9
    for r in sorted(rows, key=lambda x: x['lut']):
        if r['accuracy_pct'] > best:
            out.append(r)
            best = r['accuracy_pct']
    return out


def draw(ax, rows, dataset, ours_present, label=None):
    # Series are keyed by (method, convention), never method alone: a Pareto curve drawn
    # across both would connect core-only points to encoder-included ones and show a frontier
    # that no single accounting produces. That is the error this whole figure exists to avoid.
    groups = {}
    for r in rows:
        e = r.get('encoder_included')
        tag = {True: ' (+encoder)', False: ' (core only)'}.get(e, '')
        groups.setdefault(r['method'] + tag, []).append(r)

    for method, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        col = C.get(method.split(' (+')[0].split(' (core')[0], DEFAULT)
        curve = len(pts) > 2
        if curve:
            front = pareto(pts)
            ax.plot([p['lut'] for p in front], [p['accuracy_pct'] for p in front],
                    '-', color=col, lw=2, alpha=.85, zorder=2)
        for p in pts:
            m, filled = style(p)
            ax.plot(p['lut'], p['accuracy_pct'], m, ms=8, color=col,
                    mfc=col if filled else 'none', mew=2, zorder=3,
                    label=method if p is pts[0] else None)

        # Label only the extremes of a swept curve; every point on a 41-point sweep is noise.
        to_label = pts if not curve else (
            [max(pts, key=lambda x: x['accuracy_pct']), min(pts, key=lambda x: x['lut'])])
        for i, p in enumerate(to_label):
            text = p['model'] if curve else method
            if p.get('accuracy_is_upper_bound'):
                text += ' *'      # accuracy is the float model's, not the built design's
            # Alternate the offset: several published points sit within 0.3 pp of each other
            # at ~76%, and a fixed offset stacks their labels on top of one another.
            dx, dy = ((8, 6), (8, -12), (-8, 8), (-8, -12))[(_LBL[0] + i) % 4]
            ax.annotate(text, (p['lut'], p['accuracy_pct']),
                        textcoords='offset points', xytext=(dx, dy), fontsize=7.5,
                        ha='left' if dx > 0 else 'right', color=col, zorder=4)
        _LBL[0] += len(to_label)

    if ours_present:
        ax.axvline(DEVICE_LUTS, color='#B00020', ls='--', lw=1.4, zorder=1)
        lo, hi = ax.get_ylim()
        ax.text(DEVICE_LUTS, lo + (hi - lo) * .42, ' XC7A35T limit — 20,800 LUTs ',
                rotation=90, ha='right', va='center', fontsize=8, color='#B00020',
                fontweight='bold', bbox=dict(fc='white', ec='none', alpha=.75, pad=1.5))

    ax.set_xscale('log')
    ax.set_xlabel('LUTs (log scale)  —  lower is better')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, which='both', alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    # 'JSC-OPENML' names a variant; for a single-variant benchmark the variant name alone is
    # right and 'JSC-MNIST' would be nonsense.
    ax.set_title(label or f'JSC-{dataset.upper()}', fontsize=13, fontweight='bold', loc='left')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', choices=sorted(BENCHMARKS), default='jsc',
                    help='which study to plot (default %(default)s)')
    ap.add_argument('--snapshot', action='store_true',
                    help="also write into the benchmark's docs/jsc/results-cc dir for committing")
    args = ap.parse_args(argv)

    global OUT, SNAP
    use_benchmark(args.benchmark)
    OUT, SNAP = OUTDIRS[args.benchmark]

    lit = load_literature()
    rows = list(lit['results']) + load_ours() + load_conifer() + load_hls4ml()
    os.makedirs(OUT, exist_ok=True)

    written = []
    prefix, variants = FIGURES[args.benchmark]
    for dataset in variants:
        sel = [r for r in rows if r['dataset'] == dataset and r.get('lut')
               and r.get('accuracy_pct')]
        if not sel:
            continue
        ours = any(r['part'] == 'xc7a35t-1' for r in sel)
        fig, ax = plt.subplots(figsize=(11, 6.8))
        draw(ax, sel, dataset, ours,
             label=dataset.upper() if prefix == dataset else None)

        # The published parts differ by benchmark: JSC's comparators are all xcvu9p, but MNIST's
        # weightless rows (BTHOWeN) are xc7z020 -- same 7-series family and speed grade as ours,
        # which is the closest silicon match in the study and worth naming.
        published = {'jsc': 'xcvu9p', 'mnist': 'xcvu9p + xc7z020-1'}[args.benchmark]
        sub = (f'our silicon (xc7a35t-1) vs published ({published}) — LUTs transfer, ns does not'
               if ours else
               'published work only — this project has NO results here, by design')
        fig.suptitle(f'{args.benchmark.upper()} accuracy vs area — {sub}', fontsize=9.5, y=.965,
                     x=.125, ha='left', color='#444')
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, loc='lower center', fontsize=8, frameon=False, ncol=3,
                  bbox_to_anchor=(.5, -.005), columnspacing=1.4, handletextpad=.4)
        if not ours:
            # Make headroom rather than hunt for a gap: with 9 methods and a direct label on
            # every point there is no reliable empty patch inside the data, and a caption that
            # overlaps the points it is explaining defeats its own purpose.
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + (hi - lo) * .24)
            ax.text(.5, .985,
                    'This project does not appear on this figure — by design. '
                    'We train on JSC-OpenML;\n'
                    'every method here is on JSC-CERNBox, which runs ~1.05 pp harder, so '
                    'plotting ours\n'
                    'alongside them would overstate us by about a percentage point.',
                    transform=ax.transAxes, ha='center', va='top', fontsize=9,
                    color='#B00020', linespacing=1.5,
                    bbox=dict(fc='#FFF4F4', ec='#B00020', lw=1.2, alpha=.97, pad=7))
        note = ('filled = LUT count includes input encoder   ·   hollow = core only, encoder '
                'excluded   ·   square = no separate encoder stage')
        if any(r.get('accuracy_is_upper_bound') for r in rows):
            note += ('\n*  accuracy is the full-precision model, an upper bound on what the '
                     'built design scores, not a measurement of it')
        fig.text(.125, .012, note, fontsize=7.5, color='#666')
        fig.tight_layout(rect=[0, .03, 1, .94])

        for d in ([OUT, SNAP] if args.snapshot else [OUT]):
            os.makedirs(d, exist_ok=True)
            # `mnist-mnist.png` would be silly: when the benchmark has one variant and it shares
            # the benchmark's name, the prefix carries no information.
            stem = dataset if prefix == dataset else f'{prefix}-{dataset}'
            path = os.path.join(d, f'{stem}.png')
            fig.savefig(path, dpi=200)
            written.append(os.path.relpath(path, REPO))
        plt.close(fig)

    for w in written:
        print('wrote', w)
    if args.benchmark == 'jsc':
        print('\nTwo figures, never one: the datasets are ~1.05 pp apart and cannot share an axis.')
    else:
        # One figure, and the REASON matters: it is the absence of JSC's two-datasets defect,
        # not an oversight. See docs/mnist/phase3-ledger.md 2.3.
        print('\nOne figure: MNIST has a single canonical split, so no variant axis exists.')
        print('NOTE gaps below 0.24 pp are inside our measured noise floor and are not rankings')
        print('     (docs/mnist/reduction-ledger.md). Several literature rows are printed to whole')
        print('     percent, i.e. +/-0.5 pp -- the table marks which.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
