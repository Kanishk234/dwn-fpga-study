"""Build the full-test-set .npz that Gate 1b needs, locally.

Gate 1b's claim is about the WHOLE test set. The 1,000-sample `_testvectors.npz` that ships
beside every checkpoint is a testbench file: enough to prove Gate 1 in simulation, not enough to
support an accuracy number. `scripts/host.py` says so and refuses to call a subset run Gate 1b.

JSC's full set was dumped from a Kaggle notebook because its split was made there with a seed.
Nothing about this needs a GPU, though: the float32 model and the golden fixed-point model differ
ONLY in the thermometer comparison, and everything after it -- LUT lookup, GroupSum, argmax -- is
exact integer work either way. So both reference predictions are plain numpy, and any dataset
whose descriptor declares a reproducible `test_split` can be dumped on this machine.

    .venv\\Scripts\\python.exe scripts\\dump_testset.py training\\artifacts\\<run>_checkpoint.pt

WHY THIS VALIDATES ITSELF FIRST. A test set that is subtly wrong -- wrong split, wrong scaling,
wrong row order -- produces a Gate 1b run that looks entirely normal and means nothing. So before
writing anything, this regenerates the rows that already exist in the committed
`_testvectors.npz` and requires them to match EXACTLY: same features, same predictions. If the
overlap does not reproduce, the fetch or the scaling is wrong and the dump is refused.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
sys.path.insert(0, REPO)
import datasets                                                             # noqa: E402
from extract import (load_checkpoint, layer_indices, extract_tables,        # noqa: E402
                     extract_wiring, encode, forward)


def golden_float(x_scaled, thresholds, layers, num_classes, chunk=20000):
    """The float32 model's predictions, in numpy.

    Identical to the quantized golden model except that the thermometer comparison happens in
    float rather than against floor(t * 2**frac). Chunked because encoding the whole set at once
    is (N x features x z) booleans, which is gigabytes at MNIST's shape.
    """
    out = []
    for i in range(0, len(x_scaled), chunk):
        bits = encode(x_scaled[i:i + chunk], thresholds)
        out.append(forward(bits, layers, num_classes)[0])
    return np.concatenate(out)


def fetch(ds):
    """Raw features and labels, in the dataset's published order."""
    from sklearn.datasets import fetch_openml
    print(f'fetching {ds.openml_name} from OpenML (this is the slow part)...')
    raw = fetch_openml(ds.openml_name, version=1, as_frame=False)
    return raw.data.astype(np.float32), raw.target


def take_split(ds, X, y):
    kind, _, arg = ds.test_split.partition(':')
    if kind != 'tail':
        raise SystemExit(
            f'{ds.name}: test_split {ds.test_split!r} is not reproducible from the raw data '
            f'(only "tail:N" is). Its split was made elsewhere -- dump it there, or the rows '
            f'will differ from every published number for this dataset.')
    n = int(arg)
    print(f'split      : last {n} rows of {len(X)} (the canonical test split)')
    return X[-n:], y[-n:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--out', default=None, help='default: <checkpoint>_testset_full.npz')
    args = ap.parse_args(argv)

    ck = load_checkpoint(args.checkpoint)
    ds = datasets.identify(ck)
    ds.check_checkpoint(ck)
    if not ds.test_split:
        raise SystemExit(
            f'{ds.name} has no reproducible test_split in its descriptor, so this script cannot '
            f'build its test set without inventing a different one. See datasets/__init__.py.')
    print(f'dataset    : {ds.name}  ({ds.features} features, {ds.classes} classes, '
          f'{ds.fixed_point})')

    sd = ck['state_dict']
    cfg = ck['config']
    layers = [(extract_tables(sd, i), *extract_wiring(sd, i, cfg['n']))
              for i in layer_indices(sd)]
    thresholds = ck['thermometer']['thresholds'].numpy()

    X_raw, y_str = fetch(ds)
    X_test, y_test = take_split(ds, X_raw, y_str)

    # Scaling comes from the CHECKPOINT, not from this script: it is whatever the training run
    # actually applied, and a mismatch here silently shifts every feature.
    mean = ck['scaler']['mean'].numpy()
    scale = ck['scaler']['scale'].numpy()
    x_scaled = ((X_test - mean) / scale).astype(np.float32)
    print(f'scaling    : from the checkpoint ({ds.scaling}), '
          f'range [{x_scaled.min():.4f}, {x_scaled.max():.4f}]')

    classes = list(ck.get('classes') or [])
    if classes and not np.issubdtype(y_test.dtype, np.number):
        lookup = {str(c): i for i, c in enumerate(classes)}
        y = np.array([lookup[str(v)] for v in y_test], dtype=np.int64)
    else:
        y = y_test.astype(np.int64)

    print('running the float32 golden model...')
    pred = golden_float(x_scaled, thresholds, layers, cfg['num_classes'])

    # ---- validate against the committed testbench file BEFORE writing ----
    tb_path = args.checkpoint.replace('_checkpoint.pt', '_testvectors.npz')
    if not os.path.exists(tb_path):
        raise SystemExit(f'no {os.path.basename(tb_path)} to validate against; refusing to write '
                         f'an unchecked test set')
    tb = np.load(tb_path)
    n_tb = len(tb['x_raw'])
    if not np.array_equal(x_scaled[:n_tb], tb['x_raw']):
        bad = int((x_scaled[:n_tb] != tb['x_raw']).sum())
        raise SystemExit(
            f'ABORT: the first {n_tb} regenerated rows do not match {os.path.basename(tb_path)} '
            f'({bad} differing values). The fetch, the split or the scaling disagrees with what '
            f'training used -- a test set built on that would make Gate 1b meaningless.')
    if not np.array_equal(pred[:n_tb], tb['pred']):
        bad = int((pred[:n_tb] != tb['pred']).sum())
        raise SystemExit(
            f'ABORT: {bad}/{n_tb} regenerated predictions differ from the committed testbench '
            f'file. The golden model here does not reproduce the one the checkpoint shipped.')
    print(f'validated  : first {n_tb} rows reproduce {os.path.basename(tb_path)} EXACTLY '
          f'(features and predictions)')

    out = args.out or args.checkpoint.replace('_checkpoint.pt', '_testset_full.npz')
    np.savez_compressed(out, x_raw=x_scaled, y=y, pred=pred)
    print(f'accuracy   : float32 model {100 * (pred == y).mean():.4f}% on {len(y)} samples')
    print(f'wrote      : {out}  ({os.path.getsize(out) / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
