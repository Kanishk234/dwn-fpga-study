"""How narrow can the encoder's input word be before the encoding changes?

Phase 3, prompted by Mecik & Kumm (arXiv:2512.15251), whose encoder costs 201 LUTs where ours
costs 1,519 on the same 50-node model at the same z=200. They normalize features to [-1,1) and
quantize to 6-9 bits; we carry a global Q3.12 16-bit word whose 3 integer bits exist only because
our features are standard-scaled to roughly [-4.6, +4.3].

Two schemes, both scored by ACCURACY on the FULL 166k test set - not by bit differences against
Q3.12 (which is itself an approximation, so "differs" does not mean "worse"), and not on the
1000-sample subset, which is how the Phase 1 narrowing result went wrong (docs/jsc/phase2-report.md
5.6). The test vectors' `x_raw` is already StandardScaler-applied - it is what the board is fed.

  in-place    keep the current scaling, just drop fractional bits. Cheap to adopt, bounded win.
  renorm      per-feature affine map into [-1,1), then W bits. `x > t` is invariant under a
              monotonic affine map applied to BOTH sides, so this needs NO retraining - the
              host already applies a per-feature scaler, and this only changes its constants.

Bit-exactness is judged on the comparators that are actually WIRED (`used`); the rest are tied
low and synthesis removes them, so they cannot affect the design.

    .venv\\Scripts\\python.exe experiments\\experiment_encoder_width.py <checkpoint>
    .venv\\Scripts\\python.exe experiments\\experiment_encoder_width.py <checkpoint> --vectors <npz>
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))

from extract import (extract_tables, extract_wiring,  # noqa: E402
                     forward, layer_indices, load_checkpoint)

sys.path.insert(0, REPO)
import datasets  # noqa: E402

# These are JSC ANALYSES, not part of the shipped flow -- `experiments/` is deliberately
# outside the dataset-agnostic contract in datasets/__init__.py (which covers exporter/,
# rtlgen/, rtl/, tb/, scripts/ and harness/). Several bake in JSC measurements outright,
# e.g. LUT_PER_BIT = 1519 / 202. So the JSC binding is stated HERE, explicitly, rather
# than inherited from a module constant that pretended to be universal.
FRAC_BITS = datasets.JSC.frac_bits
WORD_BITS = datasets.JSC.word_bits


def reference_bits(x, thr, used):
    """What the hardware does today: global Q3.12, 16-bit, saturating."""
    lo, hi = -(2 ** (WORD_BITS - 1)), 2 ** (WORD_BITS - 1) - 1
    xq = np.clip(np.floor(x * (2 ** FRAC_BITS)), lo, hi).astype(np.int64)
    tq = np.floor(thr * (2 ** FRAC_BITS)).astype(np.int64)
    return (xq[:, :, None] > tq[None, :, :]).reshape(x.shape[0], -1)[:, used]


INT_BITS = WORD_BITS - 1 - FRAC_BITS     # 3, the integer bits Q3.12 spends on range
NOISE_PP = 0.15                          # measured run-to-run floor, Phase 2
CHUNK = 8192                             # samples per forward pass; see classify()


def bits_in_place(x, thr, used, frac, int_bits=INT_BITS):
    """Same scaling and the same 3 integer bits, fewer fractional bits.

    Saturating, exactly as the hardware does. Saturation is lossless while every threshold stays
    strictly inside the range (extract.saturation_is_lossless) -- a feature clamped to the rail
    is still on the same side of every threshold, which is why the 8.08 outliers cost nothing.
    """
    word = 1 + int_bits + frac
    lo, hi = -(2 ** (word - 1)), 2 ** (word - 1) - 1
    xq = np.clip(np.floor(x * (2 ** frac)), lo, hi).astype(np.int64)
    tq = np.clip(np.floor(thr * (2 ** frac)), lo, hi).astype(np.int64)
    return (xq[:, :, None] > tq[None, :, :]).reshape(x.shape[0], -1)[:, used], word


def bits_renorm(x, thr, used, word):
    """Per-feature affine map of the THRESHOLD span into [-1,1), then `word` bits, saturating.

    The span comes from the thresholds, not the data: anything outside a feature's threshold
    range is above all of them or below all of them, so saturating there changes no bit. Sizing
    the map from the data instead spends resolution on outliers -- the wired thresholds span
    [-4.55, 4.34] while the features reach 8.08, and per feature the mismatch is wider still.

    The map is monotonic and applied to features and thresholds alike, so it cannot change a
    comparison by itself. Everything lost is lost to the narrower word, which is the point.
    Because it is exact, it needs NO retraining: it only changes the host's scaler constants.
    """
    frac = word - 1
    lo_f, hi_f = thr.min(axis=1), thr.max(axis=1)
    mid = (hi_f + lo_f) / 2.0
    half = np.maximum((hi_f - lo_f) / 2.0, 1e-12) * 1.02      # a little headroom past the rails
    xs = (x - mid[None, :]) / half[None, :]
    ts = (thr - mid[:, None]) / half[:, None]
    lo, hi = -(2 ** (word - 1)), 2 ** (word - 1) - 1
    xq = np.clip(np.floor(xs * (2 ** frac)), lo, hi).astype(np.int64)
    tq = np.clip(np.floor(ts * (2 ** frac)), lo, hi).astype(np.int64)
    return (xq[:, :, None] > tq[None, :, :]).reshape(x.shape[0], -1)[:, used]


def classify(bits_used, layers, num_classes):
    """Run the golden model on encoder bits restricted to the wired comparators.

    Delegates to extract.forward rather than re-deriving the address order: slot l is bit l of
    the address, LSB first, and getting that backwards yields a model that is wrong on most
    inputs but structurally plausible (docs/reference/checkpoint-format.md 2). `layers` must already carry
    wiring remapped onto the `used` subset -- see remap_layers.
    """
    # Chunked over samples: extract.lut_forward materializes an (N, nodes, n) int64 array, which
    # at 2400 nodes and 166k samples is ~19 GB. Chunking bounds it to a few hundred MB and does
    # not change a single result -- samples are independent.
    out = np.empty(bits_used.shape[0], dtype=np.int64)
    for lo in range(0, bits_used.shape[0], CHUNK):
        idx, _ = forward(bits_used[lo:lo + CHUNK], layers, num_classes)
        out[lo:lo + CHUNK] = idx
    return out


def remap_layers(sd, n, used):
    """Layers with the first layer's wiring reindexed from the full thermometer space onto
    `used`, so the forward pass never materializes 3,200 columns for 166k samples."""
    pos = np.full(int(used.max()) + 1, -1, dtype=np.int64)
    pos[used] = np.arange(used.size)
    layers = []
    for li in layer_indices(sd):
        wiring, kind = extract_wiring(sd, li, n)
        if li == layer_indices(sd)[0]:
            wiring = pos[wiring]
            assert wiring.min() >= 0, 'a wired input is missing from `used`'
        layers.append((extract_tables(sd, li), wiring, kind))
    return layers


def separation_floor(thr, used, z):
    """Bits needed just to keep the WIRED thresholds of a feature distinct, per feature.

    A lower bound for a BIT-EXACT encoding: if two used thresholds of one feature collapse onto
    the same integer, that distinction is gone for good. It is NOT a bound on accuracy -- losing
    a distinction between two near-identical thresholds usually costs nothing measurable, which
    is why both schemes below hold accuracy well past this width.
    """
    out, dupes = [], 0
    for f in range(thr.shape[0]):
        mine = [u - f * z for u in used if f * z <= u < (f + 1) * z]
        if len(mine) < 2:
            out.append((f, len(mine), 0))
            continue
        t = np.sort(thr[f, mine])
        gaps = np.diff(t)
        dupes += int((gaps == 0).sum())
        gaps = gaps[gaps > 0]
        if gaps.size == 0:
            out.append((f, len(mine), 0))
            continue
        # resolution must be finer than the smallest gap, across the feature's own span
        bits = int(np.ceil(np.log2((t.max() - t.min()) / gaps.min()))) + 1
        out.append((f, len(mine), bits))
    return out, dupes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--vectors', help='.npz of test features (default: the *_testset_full.npz '
                                      'beside the checkpoint, else *_testvectors.npz)')
    args = ap.parse_args(argv)

    ck = load_checkpoint(args.checkpoint)
    thr = np.asarray(ck['thermometer']['thresholds'].numpy(), dtype=np.float64)
    z = thr.shape[1]
    n = ck['config']['n']
    wiring, _ = extract_wiring(ck['state_dict'], layer_indices(ck['state_dict'])[0], n)
    used = np.unique(wiring)

    stem = args.vectors
    if not stem:
        base = args.checkpoint.replace('_checkpoint.pt', '')
        for suffix in ('_testset_full.npz', '_testvectors.npz'):
            if os.path.exists(base + suffix):
                stem = base + suffix
                break
    if not stem:
        raise SystemExit('no test vectors found; pass --vectors')
    npz = np.load(stem)
    key = next(k for k in ('x_raw', 'x', 'X', 'features', 'x_test') if k in npz)
    # `x_raw` means "raw as the board receives it" -- the StandardScaler is ALREADY applied.
    # Do not apply it again: its range is [-6.16, 8.08], which is the 8.08 outlier that
    # extract.quantize's docstring describes. Re-scaling pushes it to 713 and accuracy to 34%.
    x = np.asarray(npz[key], dtype=np.float64)

    print(f'checkpoint : {os.path.basename(args.checkpoint)}')
    print(f'vectors    : {os.path.basename(stem)}  ({x.shape[0]:,} samples)')
    print(f'encoder    : {thr.shape[0]} features x z={z} = {thr.size} bits, '
          f'{used.size} wired')
    print(f'today      : Q{WORD_BITS-1-FRAC_BITS}.{FRAC_BITS} signed, {WORD_BITS}-bit, '
          f'threshold range [{thr.min():.3f}, {thr.max():.3f}]')
    print()

    ref = reference_bits(x, thr, used)

    floor, dupes = separation_floor(thr, used, z)
    worst_f, _, worst = max(floor, key=lambda r: r[2])
    print(f'Bit-exactness floor: {worst} bits, set by feature {worst_f} '
          f'(finest wired threshold gap relative to its own span).')
    print('  A bound on exactness, NOT on accuracy -- see below.')
    if dupes:
        print(f'  NOTE {dupes} wired threshold(s) are exact duplicates of another wired '
              f'threshold on the same feature -- redundant comparators, removable today.')
    print()

    # Accuracy is the metric that decides this. Bit differences against Q3.12 are not errors --
    # Q3.12 is itself an approximation of the float model, so a narrower encoding that differs
    # may be closer to it. Evaluate on all 166k against the labels.
    sd = ck['state_dict']
    layers = remap_layers(sd, n, used)
    num_classes = ck['config']['num_classes']
    y = np.asarray(npz['y']).astype(np.int64)

    exact = (x[:, :, None] > thr[None, :, :]).reshape(x.shape[0], -1)[:, used]
    acc_float = float((classify(exact, layers, num_classes) == y).mean())
    acc_ref = float((classify(ref, layers, num_classes) == y).mean())
    print(f'float encoder (no quantization) : {acc_float * 100:.4f}%')
    print(f'Q3.12 16-bit, what ships today  : {acc_ref * 100:.4f}%')
    # The checkpoint's OWN recorded accuracy, not the npz's `pred` column: a full test set can be
    # shared between checkpoints (the scaler and split are identical across the sweep), but its
    # `pred` belongs to whichever model dumped it. Agreement here validates that the vectors,
    # the wiring and the thresholds all belong together.
    recorded = ck.get('results', {}).get('final_acc')
    if recorded is not None:
        delta = (acc_float - recorded) * 100
        flag = 'OK' if abs(delta) < 0.01 else '<-- MISMATCH, wrong vectors for this checkpoint?'
        print(f'checkpoint records              : {recorded * 100:.4f}%  ({delta:+.4f} pp) {flag}')
    print()

    hdr = (f"{'scheme':<9} {'word':>5} {'accuracy':>10} {'vs today':>10} "
           f"{'bits differ':>12}  verdict")
    print(hdr)
    print('-' * len(hdr))
    results = {}

    def row(scheme, word, got):
        diff = int((got != ref).sum())
        acc = float((classify(got, layers, num_classes) == y).mean())
        d_pp = (acc - acc_ref) * 100
        results[(scheme, word)] = (acc, diff)
        if diff == 0:
            verdict = 'BIT-EXACT'
        elif abs(d_pp) <= NOISE_PP:
            verdict = 'same within noise'
        else:
            verdict = 'ACCURACY MOVES'
        print(f'{scheme:<9} {word:>5} {acc * 100:>9.4f}% {d_pp:>+9.3f} {diff:>12,}  {verdict}')

    for frac in range(FRAC_BITS, 2, -1):
        got, word = bits_in_place(x, thr, used, frac)
        row('in-place', word, got)
    print()
    for word in range(WORD_BITS, 3, -1):
        row('renorm', word, bits_renorm(x, thr, used, word))

    print()
    ok = [(s, w) for (s, w), (a, _) in results.items()
          if abs(a - acc_ref) * 100 <= NOISE_PP]
    for scheme in ('in-place', 'renorm'):
        widths = [w for s, w in ok if s == scheme]
        if widths:
            print(f'{scheme:<9} holds accuracy down to {min(widths)} bits '
                  f'({WORD_BITS - min(widths)} narrower than today)')
    print()
    print(f'"Same within noise" uses the measured {NOISE_PP} pp run-to-run floor from Phase 2.')
    print('Area is NOT measured here. Emit at these widths and synthesize to get that.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
