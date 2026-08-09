"""Build a synthetic checkpoint so Gate 1 can run at an `n` nothing has been trained at.

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
from extract import (FRAC_BITS, encode, extract_tables, extract_wiring,  # noqa: E402
                     forward, layer_indices, quantize, quantize_thresholds)

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
    ap.add_argument('--classes', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--outdir', default=os.path.join(REPO, 'build', 'testckpt'))
    args = ap.parse_args()

    assert args.width % args.classes == 0, 'width must divide by num_classes (GroupSum)'
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    src = np.load(DEFAULT_VECTORS)
    x_raw = src['x_raw'].astype(np.float32)          # (N, 16) scaled feature space
    y = src['y']
    n_features = x_raw.shape[1]
    input_bits = n_features * args.z

    # Thermometer thresholds at per-feature quantiles -- the same idea as
    # DistributiveThermometer, computed here so no GPU is needed.
    qs = (np.arange(1, args.z + 1) / (args.z + 1))
    thresholds = np.stack([np.quantile(x_raw[:, f], qs) for f in range(n_features)])

    # Random LUT tables and a random learnable mapping. Only signs and argmax matter.
    luts = rng.normal(size=(args.width, 2 ** args.n)).astype(np.float32)
    mapping = rng.normal(size=(input_bits, args.width * args.n)).astype(np.float32)

    ck = {
        'run_name': f'n{args.n}_synthetic',
        'config': {
            'thermometer': 'distributive', 'thermometer_bits': args.z, 'n': args.n,
            'layers': [args.width], 'mapping': ['learnable'], 'num_classes': args.classes,
            'tau': 1.0, 'batch_size': 100, 'epochs': 0, 'lr': 0.0,
            'lr_step': 1, 'lr_gamma': 1.0, 'seed': args.seed,
        },
        'pinned_commit': 'SYNTHETIC -- not a trained model',
        'state_dict': {'0.luts': torch.from_numpy(luts),
                       '0.mapping.weights': torch.from_numpy(mapping)},
        'thermometer': {'kind': 'distributive', 'num_bits': args.z,
                        'thresholds': torch.from_numpy(thresholds.astype(np.float32))},
        'scaler': {'mean': torch.zeros(n_features), 'scale': torch.ones(n_features)},
        'classes': [str(i) for i in range(args.classes)],
        'feature_names': [f'f{i}' for i in range(n_features)],
        'results': {'final_acc': 0.0, 'best_acc': 0.0, 'history': [], 'epoch_losses': []},
        'torch_version': torch.__version__,
        'synthetic': True,
    }

    stem = os.path.join(args.outdir, f'n{args.n}_synthetic')
    torch.save(ck, stem + '_checkpoint.pt')

    # Reference predictions from the GOLDEN MODEL, quantized exactly as the hardware will be.
    # This is what Gate 1 compares the RTL against, so it must go through the same quantize ->
    # encode path the design implements, not a float shortcut.
    layers = [(extract_tables(ck['state_dict'], i),
               *extract_wiring(ck['state_dict'], i, args.n))
              for i in layer_indices(ck['state_dict'])]
    thr_q = quantize_thresholds(thresholds, FRAC_BITS)
    xq = quantize(x_raw, FRAC_BITS)
    bits = encode(xq, thr_q)
    pred, _ = forward(bits, layers, args.classes)

    np.savez_compressed(stem + '_testvectors.npz',
                        x_binarized=bits.astype(np.uint8), x_raw=x_raw,
                        y=y, pred=pred.astype(np.int64))

    used = np.unique(layers[0][1])
    print(f'wrote {os.path.relpath(stem, REPO)}_checkpoint.pt  (+ _testvectors.npz)')
    print(f'  n={args.n}  z={args.z}  layers=[{args.width}]  classes={args.classes}')
    print(f'  input bits {input_bits}, table entries {2**args.n}, '
          f'{used.size} distinct bits selected of {input_bits}')
    print(f'  predictions span classes {sorted(set(pred.tolist()))}')
    # A degenerate model (every sample one class) would pass Gate 1 trivially and prove nothing.
    if len(set(pred.tolist())) < 2:
        print('  WARNING: predictions are constant -- Gate 1 would be vacuous. Change --seed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
