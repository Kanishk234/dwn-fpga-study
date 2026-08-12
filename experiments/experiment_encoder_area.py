"""What does the encoder's input word width actually cost in LUTs?

The area half of experiment_encoder_width.py, which established that accuracy survives down to
10 bits (same scaling) or 8 bits (per-feature renormalization, no retraining). This measures
what those widths buy.

Only the encoder is synthesized: it is a standalone out-of-context target with no primitives,
and the core is unaffected by the input word, so a full build would add ~2 minutes per point to
re-measure a constant.

Constants are emitted with the in-place scheme (3 integer bits kept, fractional bits dropped).
Area depends on comparator WIDTH, not on which constants are compared against, so the renorm
scheme at the same width costs the same -- it just tolerates a narrower one.

    .venv\\Scripts\\python.exe experiments\\experiment_encoder_area.py <checkpoint>
    .venv\\Scripts\\python.exe experiments\\experiment_encoder_area.py <checkpoint> --widths 16,10,8
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ('exporter', 'rtlgen', 'scripts'):
    sys.path.insert(0, os.path.join(REPO, sub))

from extract import (extract_wiring, layer_indices,  # noqa: E402
                     load_checkpoint)

sys.path.insert(0, REPO)
import datasets  # noqa: E402

# These are JSC ANALYSES, not part of the shipped flow -- `experiments/` is deliberately
# outside the dataset-agnostic contract in datasets/__init__.py (which covers exporter/,
# rtlgen/, rtl/, tb/, scripts/ and harness/). Several bake in JSC measurements outright,
# e.g. LUT_PER_BIT = 1519 / 202. So the JSC binding is stated HERE, explicitly, rather
# than inherited from a module constant that pretended to be universal.
FRAC_BITS = datasets.JSC.frac_bits
WORD_BITS = datasets.JSC.word_bits
from emit_encoder import emit_encoder                                       # noqa: E402
from run_gate1 import find_vivado_bin                                       # noqa: E402
from run_synth import DEFAULT_PART, parse_utilization, run_one              # noqa: E402

OUT = os.path.join(REPO, 'build', 'experiments', 'encwidth')
INT_BITS = WORD_BITS - 1 - FRAC_BITS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--widths', default='16,12,10,9,8,7,6')
    ap.add_argument('--part', default=DEFAULT_PART)
    ap.add_argument('--vivado-bin', default=None)
    ap.add_argument('--impl', action='store_true',
                    help='place and route too (post-synthesis area is close, timing is not)')
    args = ap.parse_args(argv)

    ck = load_checkpoint(args.checkpoint)
    thr = np.asarray(ck['thermometer']['thresholds'].numpy(), dtype=np.float64)
    n = ck['config']['n']
    wiring, _ = extract_wiring(ck['state_dict'], layer_indices(ck['state_dict'])[0], n)
    used = np.unique(wiring)
    widths = [int(w) for w in args.widths.split(',')]
    vivado_bin = find_vivado_bin(args.vivado_bin)

    print(f'checkpoint : {os.path.basename(args.checkpoint)}')
    print(f'encoder    : {used.size} wired comparators of {thr.size} thermometer bits')
    print(f'part       : {args.part}   widths: {widths}')
    print()

    rows = []
    for word in widths:
        frac = word - 1 - INT_BITS
        rtl_dir = os.path.join(OUT, f'w{word}', 'rtl')
        os.makedirs(rtl_dir, exist_ok=True)
        enc_v = os.path.join(rtl_dir, 'thermometer_encoder.v')
        thr_q = emit_encoder(ck, used, enc_v, frac_bits=frac, word=word)

        # How many of the wired comparators still compare against DISTINCT constants. Below the
        # bit-exactness floor some collapse, and synthesis then dedupes them -- a real saving,
        # but one that comes from losing a distinction rather than from a cheaper comparator.
        flat = np.asarray(thr_q).reshape(-1)[used]
        feat = used // thr.shape[1]
        distinct = len({(int(f), int(t)) for f, t in zip(feat, flat)})

        rel = os.path.relpath(enc_v, REPO).replace('\\', '/')
        ok, out_dir = run_one(vivado_bin, 'thermometer_encoder', [rel], args.part, OUT,
                              impl=args.impl, name=f'w{word}')
        if not ok:
            print(f'  w={word}: SYNTHESIS FAILED')
            rows.append((word, frac, distinct, None))
            continue
        rpt = 'utilization_routed.rpt' if args.impl else 'utilization.rpt'
        util = parse_utilization(os.path.join(out_dir, rpt))
        rows.append((word, frac, distinct, util.get('luts')))
        print(f'  w={word:2d} (Q{INT_BITS}.{frac}): {util.get("luts")} LUTs, '
              f'{distinct}/{used.size} distinct constants')

    base = next((l for w, _, _, l in rows if w == WORD_BITS and l), None)
    print()
    hdr = (f"{'word':>5} {'format':>9} {'distinct':>9} {'LUTs':>7} {'per cmp':>8} "
           f"{'vs 16-bit':>10}")
    print(hdr)
    print('-' * len(hdr))
    for word, frac, distinct, luts in rows:
        if luts is None:
            print(f'{word:>5} {"Q%d.%d" % (INT_BITS, frac):>9} {distinct:>9} {"FAIL":>7}')
            continue
        ratio = f'{base / luts:.2f}x' if base and luts else '-'
        print(f'{word:>5} {"Q%d.%d" % (INT_BITS, frac):>9} {distinct:>9} {luts:>7} '
              f'{luts / used.size:>8.2f} {ratio:>10}')
    print()
    print(f'Output under {os.path.relpath(OUT, REPO)} (gitignored). '
          f'Nothing in the shipped flow was touched.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
