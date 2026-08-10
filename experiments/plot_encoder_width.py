"""Figure: encoder area and accuracy against input word width, on `1x2400 z=50`.

Two stacked panels sharing the x-axis rather than one plot with two y-scales. A dual-axis chart
would let the reader's eye invent a crossing point between two quantities in different units; the
whole point of this figure is that the area cliff and the accuracy limit sit one bit apart, which
is a comparison of positions along x, not of heights.

Numbers are measured, not modelled: area from out-of-context synthesis of the encoder alone
(experiment_encoder_area.py), accuracy from all 166,000 test samples (experiment_encoder_width.py).

    .venv\\Scripts\\python.exe experiments\\plot_encoder_width.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'results', 'encoder-width.png')

# input word (bits) -> encoder LUTs, measured
AREA = {16: 5753, 13: 4916, 12: 4157, 11: 992, 10: 891, 9: 794, 8: 655}
# input word -> accuracy on the full test set, measured
ACC = {16: 76.1771, 13: 76.1361, 12: 76.0663, 11: 76.0349,
       10: 75.9584, 9: 75.7747, 8: 75.0849}
NOISE_PP = 0.15          # measured run-to-run floor
BASE = ACC[16]

BLUE, RED, GREY = '#0072B2', '#B00020', '#666666'


def main():
    ws = sorted(AREA)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [1.35, 1], 'hspace': .12})

    ax1.plot(ws, [AREA[w] for w in ws], 'o-', color=BLUE, lw=2.2, ms=8, zorder=3)
    for w in ws:
        ax1.annotate(f'{AREA[w]:,}', (w, AREA[w]), textcoords='offset points',
                     xytext=(0, 11), ha='center', fontsize=8.5, color=BLUE)
    ax1.set_ylabel('Encoder area (lookup tables)')
    ax1.set_ylim(0, 6900)
    ax1.set_title('Encoder area and accuracy against input word width  —  1×2400, 50 thresholds',
                  fontsize=12.5, fontweight='bold', loc='left', pad=26)

    deltas = [ACC[w] - BASE for w in ws]
    ax2.axhspan(-NOISE_PP, NOISE_PP, color=GREY, alpha=.16, zorder=1)
    # x is inverted below, so 16 bits sits at the LEFT edge: anchor the label rightwards from it
    ax2.annotate('measurement noise floor (±0.15 pp)', (16, -NOISE_PP),
                 textcoords='offset points', xytext=(6, 6), ha='left',
                 fontsize=8, color=GREY)
    ax2.plot(ws, deltas, 'o-', color=RED, lw=2.2, ms=8, zorder=3)
    ax2.axhline(0, color=GREY, lw=.8, zorder=2)
    ax2.set_ylabel('Accuracy change (pp)')
    ax2.set_xlabel('Input word width (bits)')

    # the two limits, and the one bit between them
    for ax in (ax1, ax2):
        ax.axvspan(10.5, 11.5, color='#009E73', alpha=.13, zorder=0)
        ax.grid(True, alpha=.25, lw=.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    ax1.annotate('area collapses\nbetween 12 and 11 bits', (11.5, 4157),
                 textcoords='offset points', xytext=(26, 6), fontsize=9,
                 color=BLUE, ha='left',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.3))
    ax2.annotate('accuracy leaves the noise\nfloor below 11 bits', (10, ACC[10] - BASE),
                 textcoords='offset points', xytext=(-14, -34), fontsize=9,
                 color=RED, ha='right',
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.3))

    ax1.set_xticks(ws)
    ax1.invert_xaxis()      # narrower to the right: the direction of the optimisation
    fig.text(.125, .015,
             'Shaded band marks 11 bits — the narrowest word that preserves accuracy, and the '
             'cheap side of the area cliff, by one bit.',
             fontsize=8.5, color=GREY)
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(OUT, dpi=200)
    print('wrote', os.path.relpath(OUT, REPO))
    return 0


if __name__ == '__main__':
    sys.exit(main())
