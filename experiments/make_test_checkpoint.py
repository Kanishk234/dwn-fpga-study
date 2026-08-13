"""Build a synthetic checkpoint so Gate 1 can run on a shape nothing has been trained at.

`n` is a Phase 2 sweep axis (dse-plan §3: 6, then 4, then 2), but training happens on Kaggle and
every checkpoint that exists is n=6. So the emitters, the golden model and the RTL had never been
exercised at any other `n` -- and that is where the 2026-08-08 table-packing bug lived, where
`np.packbits` silently shifted any table shorter than 8 entries.

**Gate 1 does not need a TRAINED model.** It asks whether the emitted RTL matches the golden
software model, and both are derived from the same checkpoint. Random tables exercise that
machinery exactly as well as learned ones -- better, in fact, since random tables hit address
patterns a trained model might never produce. What a synthetic checkpoint cannot check is
numpy-vs-PyTorch agreement, but that is not where `n` dependence lives.

So: fabricate a checkpoint at the requested `n`, compute reference predictions with the golden
model itself, write the `_testvectors.npz` Gate 1 consumes, and hand both to the normal flow.

    python scripts/make_test_checkpoint.py --n 4
    python scripts/run_gate1.py --checkpoint build/testckpt/n4_synthetic_checkpoint.pt \\
        --rtl-dir build/testckpt/n4/rtl --work build/testckpt/n4/gate1
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
sys.path.insert(0, REPO)
from extract import (encode, extract_tables, extract_wiring,  # noqa: E402
                     forward, layer_indices, quantize, quantize_thresholds)

sys.path.insert(0, REPO)
import datasets  # noqa: E402

# These are JSC ANALYSES, not part of the shipped flow -- `experiments/` is deliberately
# outside the dataset-agnostic contract in datasets/__init__.py (which covers exporter/,
# rtlgen/, rtl/, tb/, scripts/ and harness/). Several bake in JSC measurements outright,
# e.g. LUT_PER_BIT = 1519 / 202. So the JSC binding is stated HERE, explicitly, rather
# than inherited from a module constant that pretended to be universal.
FRAC_BITS = datasets.JSC.frac_bits
WORD_BITS = datasets.JSC.word_bits

# Real feature vectors, so quantization and the thermometer see realistic ranges. Random
# features would still exercise the logic, but thresholds fitted to real data would then sit
# outside the sample range and most comparators would be constant -- which is exactly the case
# synthesis optimizes away, hiding whatever we were trying to test.
DEFAULT_VECTORS = os.path.join(
    REPO, 'training', 'artifacts', 'dwn_jsc_t200_distributive_50_l_b100_testvectors.npz')


def main():
    ap = argparse.ArgumentParser(description='Synthetic checkpoint for Gate 1 at any n.')
    ap.add_argument('--n', type=int, required=True, help='LUT inputs per node')
    ap.add_argument('--z', type=int, default=8, help='thermometer bits per feature')
    ap.add_argument('--width', type=int, default=20, help='nodes in the (single) layer')
    ap.add_argument('--layers', type=int, nargs='+', default=None,
                    help='nodes per layer, e.g. --layers 100 50. Overrides --width.')
    ap.add_argument('--classes', type=int, default=None)
    ap.add_argument('--dataset', default=None,
                    help='shape the input space from a datasets/ descriptor (jsc, mnist, ...) '
                         'instead of borrowing JSC vectors. Required for any feature count '
                         'other than 16.')
    ap.add_argument('--samples', type=int, default=1000,
                    help='synthetic vectors to generate, when --dataset is given')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--outdir', default=os.path.join(REPO, 'build', 'testckpt'))
    args = ap.parse_args()

    layers_w = args.layers if args.layers else [args.width]
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.dataset:
        # Synthesize the input space from the descriptor. Thresholds are fitted to the SAME
        # samples they will be tested on, deliberately: that keeps every comparator inside the
        # data range, so none is constant. A constant comparator is optimized away by synthesis,
        # which would silently shrink the very thing under test (the note below on real vectors
        # is the same concern, reached from the other direction).
        from datasets import get as get_dataset
        ds = get_dataset(args.dataset)
        n_features = ds.features
        classes = args.classes if args.classes is not None else ds.classes
        # Match the descriptor's scaling so the thermometer sees a plausible distribution:
        # standard-scaled data is roughly normal, min-max data is bounded in [0, 1).
        if ds.scaling == 'minmax':
            x_raw = rng.random((args.samples, n_features), dtype=np.float32)
        else:
            x_raw = rng.normal(size=(args.samples, n_features)).astype(np.float32)
        y = rng.integers(0, classes, size=args.samples)
    else:
        src = np.load(DEFAULT_VECTORS)
        x_raw = src['x_raw'].astype(np.float32)      # (N, 16) scaled feature space
        y = src['y']
        n_features = x_raw.shape[1]
        classes = args.classes if args.classes is not None else 5

    assert layers_w[-1] % classes == 0, (
        f'final layer {layers_w[-1]} must divide by num_classes {classes} -- GroupSum '
        f'zero-pads silently otherwise and hardware disagrees with software about group bounds')
    input_bits = n_features * args.z

    # Thermometer thresholds at per-feature quantiles -- the same idea as
    # DistributiveThermometer, computed here so no GPU is needed.
    qs = (np.arange(1, args.z + 1) / (args.z + 1))
    thresholds = np.stack([np.quantile(x_raw[:, f], qs) for f in range(n_features)])

    # Random LUT tables and a random learnable mapping per layer. Only the SIGN of a table
    # entry matters (checkpoint-format §1) and only the argmax of a mapping column does (§3a),
    # so normal noise exercises the same machinery a trained model would -- and reaches address
    # patterns a trained model might never produce.
    state = {}
    prev_bits = input_bits
    for li, w in enumerate(layers_w):
        state[f'{li}.luts'] = torch.from_numpy(
            rng.normal(size=(w, 2 ** args.n)).astype(np.float32))
        state[f'{li}.mapping.weights'] = torch.from_numpy(
            rng.normal(size=(prev_bits, w * args.n)).astype(np.float32))
        prev_bits = w

    ck = {
        'run_name': f'{args.dataset or "jsc"}_n{args.n}_synthetic',
        'config': {
            'thermometer': 'distributive', 'thermometer_bits': args.z, 'n': args.n,
            'layers': list(layers_w), 'mapping': ['learnable'] * len(layers_w),
            'num_classes': classes,
            'tau': 1.0, 'batch_size': 100, 'epochs': 0, 'lr': 0.0,
            'lr_step': 1, 'lr_gamma': 1.0, 'seed': args.seed,
        },
        'pinned_commit': 'SYNTHETIC -- not a trained model',
        'state_dict': state,
        'thermometer': {'kind': 'distributive', 'num_bits': args.z,
                        'thresholds': torch.from_numpy(thresholds.astype(np.float32))},
        'scaler': {'mean': torch.zeros(n_features), 'scale': torch.ones(n_features)},
        'classes': [str(i) for i in range(classes)],
        'feature_names': [f'f{i}' for i in range(n_features)],
        'results': {'final_acc': 0.0, 'best_acc': 0.0, 'history': [], 'epoch_losses': []},
        'torch_version': torch.__version__,
        'synthetic': True,
    }

    shape = 'x'.join(str(w) for w in layers_w)
    stem = os.path.join(
        args.outdir,
        f'{args.dataset or "jsc"}_f{n_features}_c{classes}_n{args.n}_z{args.z}_w{shape}_synthetic')
    torch.save(ck, stem + '_checkpoint.pt')

    # Reference predictions from the GOLDEN MODEL, quantized exactly as the hardware will be.
    # This is what Gate 1 compares the RTL against, so it must go through the same quantize ->
    # encode path the design implements, not a float shortcut.
    layers = [(extract_tables(ck['state_dict'], i),
               *extract_wiring(ck['state_dict'], i, args.n))
              for i in layer_indices(ck['state_dict'])]
    thr_q = quantize_thresholds(thresholds, FRAC_BITS)
    xq = quantize(x_raw, FRAC_BITS, WORD_BITS)
    bits = encode(xq, thr_q)
    pred, _ = forward(bits, layers, classes)

    np.savez_compressed(stem + '_testvectors.npz',
                        x_binarized=bits.astype(np.uint8), x_raw=x_raw,
                        y=y, pred=pred.astype(np.int64))

    used = np.unique(layers[0][1])
    print(f'wrote {os.path.relpath(stem, REPO)}_checkpoint.pt  (+ _testvectors.npz)')
    print(f'  dataset={args.dataset or "jsc (borrowed vectors)"}  features={n_features}  '
          f'classes={classes}')
    print(f'  n={args.n}  z={args.z}  layers={list(layers_w)}  samples={x_raw.shape[0]}')
    print(f'  input bits {input_bits}, table entries {2**args.n}, '
          f'{used.size} distinct bits selected of {input_bits}')
    print(f'  predictions span classes {sorted(set(pred.tolist()))}')
    # A degenerate model (every sample one class) would pass Gate 1 trivially and prove nothing.
    if len(set(pred.tolist())) < 2:
        print('  WARNING: predictions are constant -- Gate 1 would be vacuous. Change --seed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
