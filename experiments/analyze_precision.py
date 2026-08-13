"""Find the smallest fixed-point input format that preserves the thermometer encoding.

The software encoder compares float32 features against float32 thresholds. Hardware cannot, so
the feature word has to be quantized -- and quantization can flip an encoder bit, which can
flip a LUT address, which can change the predicted class. This script measures where that
starts happening instead of guessing a width.

Quantization scheme (chosen to be cheap in hardware and to fail in only one direction):

    q_x = floor(x * 2**F)        truncation, not rounding -- free in hardware
    T   = floor(t * 2**F)        threshold folded into a constant at export time
    bit = q_x > T

With T = floor(t * 2**F), `q_x > T` implies `x > t` exactly, because q_x > T means
x >= (T+1)/2**F > t. So this scheme never produces a FALSE POSITIVE. The only error is a false
negative, when x and t land in the same quantization bucket. More fractional bits shrink those
buckets, so the error rate falls monotonically in F -- which is what makes a sweep meaningful.

Two error metrics, because they are not the same thing and only the second one matters:
  bit errors    encoder bits that differ from the software encoding
  class changes predictions that differ. A flipped bit often changes no address that any
                node reads, or changes an address whose table entry is the same, or shifts a
                popcount without changing the argmax.

Caveat, stated because it bounds the conclusion: only 1000 test samples exist locally (the
.npz). The full 166,000-sample test set lives on Kaggle. A format that is clean on 1000 samples
is evidence, not proof -- Gate 1b will need the full set anyway.

Usage:
    python exporter/analyze_precision.py training/artifacts/<run>_checkpoint.pt
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
from extract import (load_checkpoint, layer_indices, extract_tables,  # noqa: E402
                     extract_wiring, forward)

FRAC_BITS = range(2, 21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--vectors')
    args = ap.parse_args()

    ck = load_checkpoint(args.checkpoint)
    cfg = ck['config']
    n, num_classes, z = cfg['n'], cfg['num_classes'], cfg['thermometer_bits']
    thresholds = ck['thermometer']['thresholds'].numpy()      # (16, z), scaled space

    layers = []
    for i in layer_indices(ck['state_dict']):
        layers.append((extract_tables(ck['state_dict'], i),
                       *extract_wiring(ck['state_dict'], i, n)))

    vec_path = args.vectors or args.checkpoint.replace('_checkpoint.pt', '_testvectors.npz')
    v = np.load(vec_path)
    x_raw = v['x_raw']                       # (N, 16) float32, AFTER StandardScaler
    x_bin = v['x_binarized'].astype(bool)    # (N, 3200)
    pred = v['pred']
    n_samples = x_raw.shape[0]

    # Only bits some node actually reads need a comparator at all.
    used = np.unique(layers[0][1])
    feat_of = used // z
    thr_of = used % z
    thr_vals = thresholds[feat_of, thr_of]

    print(f'samples          : {n_samples}')
    print(f'features         : {x_raw.shape[1]}, range '
          f'[{x_raw.min():+.4f}, {x_raw.max():+.4f}]')
    print(f'comparators      : {used.size} (of {thresholds.size} thermometer bits)')
    print(f'threshold range  : [{thr_vals.min():+.4f}, {thr_vals.max():+.4f}]')

    # Sanity: reconstruct the software encoding from x_raw and confirm it matches what the
    # notebook saved. If this fails, the feature/threshold indexing below is wrong.
    ref = x_raw[:, feat_of] > thr_vals
    assert np.array_equal(ref, x_bin[:, used]), \
        'float recomputation disagrees with saved x_binarized -- indexing is wrong'
    print('float recompute  : matches saved x_binarized on the used bits  OK')

    # Integer bits needed to hold the widest value, before the sign bit.
    peak = max(abs(float(x_raw.min())), abs(float(x_raw.max())),
               abs(float(thr_vals.min())), abs(float(thr_vals.max())))
    int_bits = int(np.ceil(np.log2(peak + 1)))
    print(f'integer bits     : {int_bits} (peak magnitude {peak:.4f}) + 1 sign')
    print()

    print(f'{"F":>3}  {"width":>5}  {"bit errors":>12}  {"bit err rate":>12}  '
          f'{"class changes":>13}  {"accuracy":>9}')
    print('-' * 70)

    base_correct = None
    best = None
    for F in FRAC_BITS:
        scale = float(2 ** F)
        qx = np.floor(x_raw.astype(np.float64) * scale)
        T = np.floor(thr_vals.astype(np.float64) * scale)
        got = qx[:, feat_of] > T

        bit_err = int((got != ref).sum())
        total_bits = ref.size

        # Rebuild the full binarized vector with quantized bits, then run the golden model.
        x_q = x_bin.copy()
        x_q[:, used] = got
        pred_q, _ = forward(x_q, layers, num_classes)
        changed = int((pred_q != pred).sum())
        acc = float((pred_q == v['y']).mean())
        if base_correct is None:
            base_correct = float((pred == v['y']).mean())

        width = int_bits + F + 1
        print(f'{F:>3}  {width:>5}  {bit_err:>12}  {bit_err/total_bits:>11.2e}  '
              f'{changed:>13}  {100*acc:>8.2f}%')

        if bit_err == 0 and best is None:
            best = (F, width)

    print('-' * 70)
    print(f'float32 reference accuracy on these {n_samples} samples: {100*base_correct:.2f}%')
    print()
    if best:
        F, width = best
        print(f'RECOMMENDATION: Q{int_bits}.{F} signed, {width} bits total')
        print(f'  First format with ZERO bit errors on all {n_samples} samples.')
        print(f'  Each comparator is a {width}-bit signed compare against a constant.')
    else:
        print(f'No format in F={FRAC_BITS.start}..{FRAC_BITS.stop-1} reached zero bit errors.')
    print()
    print('Input precision is a real area knob, not just a correctness threshold -- comparator')
    print('cost scales with width, and 202 of them is ~4x the node count. Formats above that')
    print('cost fewer LUTs for a handful of class changes are a Phase 2 sweep axis, not a')
    print('mistake (docs/reference/paper-configs.md: nobody has measured encoder cost where it binds).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
