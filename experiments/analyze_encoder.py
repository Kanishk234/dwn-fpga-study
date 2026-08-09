"""Where does the thermometer encoder's area actually go, and how much of it is avoidable?

Measured: the encoder is 1519 LUTs against a 108-LUT core -- 14x, where brief §12 risk #3
anticipated at most 3.2x (Mecik & Kumm). 202 comparators at ~7.5 LUTs each is exactly what
independent 16-bit compares cost, so Vivado is sharing nothing. The question this answers is
how much of that 14x is DWN's encoder and how much is our naive construction.

Two structural facts the current emitter ignores:

1. EVERY COMPARATOR IS 16 BITS WIDE, whether it needs to be or not. Q3.12 was chosen globally,
   from the worst-case feature. A feature whose selected thresholds are far apart can be
   compared at much lower precision without changing a single bit of output.

2. THRESHOLDS OF ONE FEATURE ARE SORTED, so its output bits are a thermometer code -- strictly
   monotonic. 46 comparisons of the same value against 46 sorted constants carry far less
   information than 46 independent comparisons, and cost the same today.

This script quantifies (1) exactly and estimates (2), so the optimization is chosen from
evidence rather than from which idea sounds cleverest.

Caveat carried from analyze_precision.py: only 1000 samples exist locally, so "preserves every
comparison" means on those 1000. Narrowing is a spec change, and the golden model would have to
narrow identically for Gate 1 to stay bit-exact.

Usage:
    python exporter/analyze_encoder.py training/artifacts/<run>_checkpoint.pt
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
from extract import (FRAC_BITS, WORD_BITS, load_checkpoint, layer_indices,  # noqa: E402
                     extract_wiring)

# Measured cost of one 16-bit signed compare-against-constant on this part: 1519 LUTs / 202
# comparators. Used only to estimate savings; the real number comes from synthesis.
LUTS_PER_16BIT_CMP = 1519 / 202


def min_frac_bits(x_col, thresholds, max_frac=FRAC_BITS):
    """Fewest fractional bits that reproduce every comparison this feature makes.

    Reference is the Q3.12 encoding the design already uses, not float -- Q3.12 is the spec
    (docs/phase1-ledger.md), so anything matching it is bit-exact by definition.
    """
    ref = np.array([(np.floor(x_col * 2**FRAC_BITS) > np.floor(t * 2**FRAC_BITS))
                    for t in thresholds])
    for f in range(0, max_frac + 1):
        got = np.array([(np.floor(x_col * 2**f) > np.floor(t * 2**f)) for t in thresholds])
        if np.array_equal(got, ref):
            return f
    return max_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--vectors')
    args = ap.parse_args()

    ck = load_checkpoint(args.checkpoint)
    cfg = ck['config']
    n, z = cfg['n'], cfg['thermometer_bits']
    thresholds = ck['thermometer']['thresholds'].numpy()
    names = ck['feature_names']

    wiring, _ = extract_wiring(ck['state_dict'], layer_indices(ck['state_dict'])[0], n)
    used = np.unique(wiring)

    vec_path = args.vectors or args.checkpoint.replace('_checkpoint.pt', '_testvectors.npz')
    x_raw = np.load(vec_path)['x_raw'].astype(np.float64)

    print(f'comparators : {used.size}')
    print(f'measured    : 1519 LUTs total, {LUTS_PER_16BIT_CMP:.1f} LUTs per '
          f'{WORD_BITS}-bit comparator')
    print()
    print(f'{"feature":18s} {"cmps":>5} {"int":>4} {"frac":>5} {"width":>6} {"LUTs now":>9} '
          f'{"LUTs min":>9}')
    print('-' * 68)

    now_total = 0.0
    min_total = 0.0
    rows = []

    for f in range(thresholds.shape[0]):
        sel = used[(used // z) == f]
        if sel.size == 0:
            continue
        thr = thresholds[f, sel % z].astype(np.float64)
        col = x_raw[:, f]

        frac = min_frac_bits(col, thr)
        peak = max(abs(float(col.min())), abs(float(col.max())),
                   abs(float(thr.min())), abs(float(thr.max())))
        int_bits = max(0, int(np.ceil(np.log2(peak + 1))))
        width = int_bits + frac + 1

        now = sel.size * LUTS_PER_16BIT_CMP
        # Comparator cost is roughly linear in width on a carry chain.
        mini = sel.size * LUTS_PER_16BIT_CMP * width / WORD_BITS

        now_total += now
        min_total += mini
        rows.append((names[f], sel.size, int_bits, frac, width, now, mini))
        print(f'{names[f]:18s} {sel.size:5d} {int_bits:4d} {frac:5d} {width:6d} '
              f'{now:9.0f} {mini:9.0f}')

    print('-' * 68)
    print(f'{"TOTAL":18s} {used.size:5d} {"":4s} {"":5s} {"":6s} '
          f'{now_total:9.0f} {min_total:9.0f}')
    print()
    saving = 1 - min_total / now_total
    print(f'(1) NARROWER COMPARATORS: {now_total:.0f} -> {min_total:.0f} LUTs, '
          f'{100*saving:.0f}% smaller')
    print('    Bit-exact against the Q3.12 spec on the available samples: each feature keeps')
    print('    exactly the precision its own thresholds need. Requires the golden model to')
    print('    narrow identically, so Gate 1 must be re-run against the new spec.')
    print()

    # (2) monotonic decomposition. Thresholds of one feature are sorted, so its k output bits
    # are a thermometer code with only k+1 reachable values. In principle the rank can be
    # found with ceil(log2(k+1)) comparisons and decoded, instead of k independent ones.
    tree_total = 0.0
    for name, k, int_bits, frac, width, now, mini in rows:
        levels = int(np.ceil(np.log2(k + 1)))
        cmp_cost = levels * LUTS_PER_16BIT_CMP * width / WORD_BITS
        decode_cost = k * 0.5      # each output bit is a small compare on the rank
        tree_total += cmp_cost + decode_cost

    print(f'(2) MONOTONIC DECOMPOSITION (estimate): {now_total:.0f} -> {tree_total:.0f} LUTs')
    print('    Sorted thresholds mean the outputs are a thermometer code, so the rank could')
    print('    be found by binary search and decoded. But the search is SEQUENTIAL -- depth')
    print('    grows with log(k) comparator delays -- and the encoder currently has 7.084 ns')
    print('    of slack, the largest in the design. This trades the one thing we have spare')
    print('    for area, and it is far more code to generate and verify than (1).')
    print()
    print('MEASURED (exporter/experiment_narrow_encoder.py): narrowing gives 1519 -> 1259')
    print('LUTs, -17.1%, better than the -12.4% this linear model predicts, with timing')
    print('untouched. Not adopted at this model size: 260 LUTs on a design occupying 7.78% of')
    print('the part does not justify a spec change that the golden model, the vector')
    print('generator and the host all have to match, plus a Gate 1 re-run.')
    print()
    print('It becomes the deciding factor at larger configs. Comparator count grows with node')
    print('count and saturates at features x z, so a big model pays for nearly the whole')
    print('thermometer -- see the projection in docs/phase1-ledger.md.')

    biggest = max(rows, key=lambda r: r[5] - r[6])
    print()
    print(f'Biggest single win: {biggest[0]} at {biggest[1]} comparators, '
          f'{WORD_BITS} -> {biggest[4]} bits '
          f'({biggest[5]:.0f} -> {biggest[6]:.0f} LUTs)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
