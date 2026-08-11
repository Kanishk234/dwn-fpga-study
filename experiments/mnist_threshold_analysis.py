"""How many thermometer thresholds per pixel can MNIST afford on a Basys 3?

784 features against JSC's 16 is the whole problem: `features x z` is the pool the first layer
draws its inputs from, and every bit some node actually reads costs a comparator.

Method, and where each number comes from:

  comparators   dse/area_model.predict_comparators -- an occupancy model with a fitted
                correction. Validated here against 37 JSC checkpoints: 3.9% mean error.
  LUTs each     MEASURED, not modelled. experiment_encoder_area.py on 1x2400 z=50. The area
                model scales this linearly with word width and that is WRONG -- there is a
                cliff between 12 and 11 bits where a carry-chain comparator collapses into
                plain LUT logic, a 4.7x drop for one bit. Linear scaling would predict 5.2
                LUTs at 11 bits; the measured value is 1.33.
  core          one LUT per node (measured exactly on JSC) plus the reduction.

    .venv\\Scripts\\python.exe experiments\\mnist_threshold_analysis.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'dse'))

import area_model as AM        # noqa: E402

MNIST_FEATURES = 784
MNIST_CLASSES = 10
DEVICE_LUTS = 20800
HARNESS = AM.HARNESS_LUTS       # UART, loader, vector store, FSM -- does not scale with the model

# Measured on 1x2400 z=50, encoder synthesized alone, out of context.
LUT_PER_COMPARATOR = {16: 7.71, 13: 6.59, 12: 5.57, 11: 1.33, 10: 1.19, 9: 1.06, 8: 0.88}

# The ratio slots/available where the selection model was actually fitted (JSC, 37 configs).
CALIBRATED_RATIO = (0.09, 9.4)


def core_luts(layers, num_classes):
    """Nodes are one LUT each; the reduction is popcount plus argmax."""
    nodes = sum(layers)
    final = layers[-1]
    group = final // num_classes
    score_w = max(1, int(__import__('math').ceil(__import__('math').log2(group + 1))))
    reduction = final * AM.popcount_lut_per_bit(group) + AM.argmax_luts(num_classes, score_w)
    return nodes + reduction


def analyse(layers, z, word, n=6, features=MNIST_FEATURES, classes=MNIST_CLASSES):
    comps = AM.predict_comparators(layers, n, z, features=features)
    enc = comps * LUT_PER_COMPARATOR[word]
    core = core_luts(layers, classes)
    total = enc + core + HARNESS
    slots, avail = n * layers[0], features * z
    ratio = slots / avail
    return {
        'comparators': comps, 'encoder': enc, 'core': core, 'total': total,
        'device_pct': total / DEVICE_LUTS * 100,
        'ratio': ratio,
        'extrapolated': not (CALIBRATED_RATIO[0] <= ratio <= CALIBRATED_RATIO[1]),
    }


def main():
    print(__doc__.split('\n\n')[0])
    print(f'device {DEVICE_LUTS:,} LUTs, harness {HARNESS} reserved\n')

    print("=" * 78)
    print("The paper's MNIST configuration: two layers, 1000 + 500")
    print("=" * 78)
    print(f"{'z':>5}{'word':>6}{'comparators':>13}{'encoder':>10}{'core':>8}{'total':>9}"
          f"{'device':>9}  verdict")
    print('-' * 78)
    for z in (8, 25, 50, 100, 200):
        for word in (16, 11, 8):
            r = analyse([1000, 500], z, word)
            flag = ' ~' if r['extrapolated'] else ''
            verdict = 'FITS' if r['device_pct'] <= 90 else ('tight' if r['device_pct'] <= 100
                                                            else 'OVER')
            print(f"{z:>5}{word:>6}{r['comparators']:>13,.0f}{r['encoder']:>10,.0f}"
                  f"{r['core']:>8,.0f}{r['total']:>9,.0f}{r['device_pct']:>8.1f}%  "
                  f"{verdict}{flag}")

    print()
    print("=" * 78)
    print('Smaller first layers, at the accuracy-safe 11-bit word')
    print("=" * 78)
    print(f"{'layers':>14}{'z':>5}{'comparators':>13}{'encoder':>10}{'core':>8}{'total':>9}"
          f"{'device':>9}  verdict")
    print('-' * 78)
    for layers in ([1000, 500], [500, 250], [300, 100], [200], [100]):
        for z in (8, 25, 50):
            r = analyse(layers, z, 11)
            flag = ' ~' if r['extrapolated'] else ''
            verdict = 'FITS' if r['device_pct'] <= 90 else ('tight' if r['device_pct'] <= 100
                                                            else 'OVER')
            print(f"{str(layers):>14}{z:>5}{r['comparators']:>13,.0f}{r['encoder']:>10,.0f}"
                  f"{r['core']:>8,.0f}{r['total']:>9,.0f}{r['device_pct']:>8.1f}%  "
                  f"{verdict}{flag}")

    print()
    print('~ = the slots/available ratio is outside the range the selection model was fitted on')
    print(f'  (JSC covered {CALIBRATED_RATIO[0]}-{CALIBRATED_RATIO[1]}); treat those rows as weaker.')
    print()
    print('EVERY row is a projection. All 37 calibration configs have 16 input features; MNIST')
    print('has 784, a 49x extrapolation on the one axis never swept. Nothing here replaces')
    print('synthesizing one real MNIST config, which is step M1e.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
