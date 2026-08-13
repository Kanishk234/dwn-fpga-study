"""Is there any structure left in the encoder to exploit? Pure analysis -- no RTL, no Vivado.

Phase 1 measured the encoder at 1519 LUTs for 202 comparators (7.5 each, about W/2 -- the
carry-chain cost of a 16-bit signed compare-against-constant) and concluded that Vivado is
already near-optimal *inside* a comparator. The open question the ledger leaves is whether
anything can be shared *between* comparators of the same feature, since they all compare the
same value against sorted constants. Mecik & Kumm's FloPoCo encoder presumably did exactly that.

The obvious idea -- binary search for the bucket index, then decode -- DOES NOT WORK, and it is
worth writing down why so nobody re-derives it:

    Each level of a binary search must select which threshold to compare against next. That is
    a multiplexer over 16-bit constants whose input count doubles per level. For k=46 the
    level-5 mux is 32-to-1 over 16 bits, on the order of 176 LUTs by itself -- more than the
    345 LUTs the 46 direct comparators cost in total. Instantiating the full tree instead needs
    1+2+4+...+32 = 63 comparators, which is worse than the 46 you started with.

So sharing can only pay if the thresholds CLUSTER -- if many of them agree in their high bits,
a coarse comparison on those high bits can be computed once and reused, with a cheap low-bit
tiebreak only where it is actually needed. Whether that holds is a property of the trained
model, not of the RTL, so it is measurable without generating anything.

This script measures it and prints an optimistic bound. Read the bound as a ceiling on what any
sharing scheme could buy -- if it is close to the 1519 baseline, the encoder is at its floor and
`z` (which sets how many thresholds exist at all) is the only real lever left.

Usage:
    python exporter/analyze_encoder_sharing.py training/artifacts/<run>_checkpoint.pt
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
from extract import (load_checkpoint, layer_indices,  # noqa: E402
                     extract_wiring)

sys.path.insert(0, REPO)
import datasets  # noqa: E402

# These are JSC ANALYSES, not part of the shipped flow -- `experiments/` is deliberately
# outside the dataset-agnostic contract in datasets/__init__.py (which covers exporter/,
# rtlgen/, rtl/, tb/, scripts/ and harness/). Several bake in JSC measurements outright,
# e.g. LUT_PER_BIT = 1519 / 202. So the JSC binding is stated HERE, explicitly, rather
# than inherited from a module constant that pretended to be universal.
FRAC_BITS = datasets.JSC.frac_bits
WORD_BITS = datasets.JSC.word_bits

# What one comparator of a given width costs, measured: 1519 LUTs / 202 comparators at 16 bits.
LUT_PER_BIT = 1519 / 202 / WORD_BITS


def cost_direct(k, width=WORD_BITS):
    """k independent compare-against-constant, the shipped scheme."""
    return k * width * LUT_PER_BIT


def cost_shared(hi_bits, groups, k):
    """Optimistic cost of a shared high-bit decomposition.

    For a split at `hi_bits`, each distinct high pattern needs one `>` and one `==` on the high
    slice; each threshold then needs a low-slice compare plus a 1-LUT combine
    (hi_gt | (hi_eq & lo_gt)). This IGNORES the routing and the combine's fan-in, so it is a
    lower bound on the shared scheme -- deliberately generous to it.
    """
    lo_bits = WORD_BITS - hi_bits
    return (groups * 2 * hi_bits * LUT_PER_BIT       # high compares, shared
            + k * lo_bits * LUT_PER_BIT              # low compares, one per threshold
            + k)                                     # the combine, ~1 LUT each


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    args = ap.parse_args()

    ck = load_checkpoint(args.checkpoint)
    cfg = ck['config']
    n, z = cfg['n'], cfg['thermometer_bits']
    thresholds = ck['thermometer']['thresholds'].numpy()

    wiring, _ = extract_wiring(ck['state_dict'], layer_indices(ck['state_dict'])[0], n)
    used = np.unique(wiring)

    print(f'checkpoint : {os.path.basename(args.checkpoint)}')
    print(f'comparators: {used.size} of {thresholds.size} thermometer bits '
          f'({100*used.size/thresholds.size:.1f}%)')
    print(f'cost model : {LUT_PER_BIT:.4f} LUT/bit, calibrated on the measured 1519 / 202')
    print()

    # Quantized thresholds, per feature, sorted -- the same integers the emitter folds in.
    print(f'{"feature":>7} {"k":>4} {"direct":>8} | '
          f'{"best split":>10} {"groups":>7} {"shared":>8} {"save":>7}')
    print('-' * 62)

    tot_direct = tot_best = 0.0
    for f in range(thresholds.shape[0]):
        sel = used[(used // z) == f]
        k = sel.size
        if k == 0:
            continue
        tq = np.sort(np.floor(thresholds[f, sel % z] * 2**FRAC_BITS).astype(np.int64))

        d = cost_direct(k)
        best, best_hi, best_groups = d, None, None
        # Try every high/low split point. Fewer distinct high patterns => more sharing.
        for hi in range(2, WORD_BITS):
            groups = len(np.unique(tq >> (WORD_BITS - hi)))
            c = cost_shared(hi, groups, k)
            if c < best:
                best, best_hi, best_groups = c, hi, groups

        tot_direct += d
        tot_best += best
        split = f'{best_hi}/{WORD_BITS-best_hi}' if best_hi else '--'
        grp = str(best_groups) if best_groups else '--'
        save = f'{100*(best-d)/d:+.0f}%' if best_hi else '--'
        print(f'{f:>7} {k:>4} {d:>8.0f} | {split:>10} {grp:>7} {best:>8.0f} {save:>7}')

    print('-' * 62)
    print(f'{"TOTAL":>7} {used.size:>4} {tot_direct:>8.0f} | '
          f'{"":>10} {"":>7} {tot_best:>8.0f} '
          f'{100*(tot_best-tot_direct)/tot_direct:>+6.0f}%')
    print()
    print(f'shipped encoder, measured : 1519 LUTs')
    print(f'optimistic shared bound   : {tot_best:.0f} LUTs')
    print()
    if tot_best > 0.85 * 1519:
        print('VERDICT: thresholds do not cluster enough for sharing to pay. The encoder is at')
        print('its floor for this scheme, and the ledger\'s "most of the 14x is real" stands.')
        print('The remaining lever is z, which decides how many thresholds exist at all --')
        print('a config change, not an RTL one.')
    else:
        print('VERDICT: there IS exploitable clustering. Worth emitting a shared-prefix encoder')
        print('and measuring it out-of-context -- but remember this bound ignores routing and')
        print('combine fan-in, so treat it as a ceiling, not a prediction.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
