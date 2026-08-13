"""Generate Gate 1 test vectors for both testbenches.

Gate 1 (CLAUDE.md): the RTL is not complete until it bit-exact matches the golden software
model on EVERY test vector, INCLUDING EDGE CASES, not just the happy path.

Two levels are emitted, because a failure in one should not be able to hide in the other:

  core   pre-binarized 3200-bit vectors -> dwn_core          (x_binarized.hex)
  top    quantized 16x16-bit features   -> dwn_top           (x_quant.hex)

Splitting them is why the training notebook saves both x_raw and x_binarized. If the top-level
fails while the core passes, the encoder is at fault and nothing else needs re-examining.

The golden model for the top level QUANTIZES exactly as the hardware does. Quantization is part
of the specification, not an error -- so Gate 1 stays bit-exact, and the float-vs-fixed
difference is reported below as a characterization number rather than buried as a tolerance.

Usage:
    python tb/gen_vectors.py training/artifacts/<run>_checkpoint.pt
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
sys.path.insert(0, REPO)
import datasets                                                             # noqa: E402
from extract import (load_checkpoint, layer_indices,  # noqa: E402
                     extract_tables, extract_wiring, forward, quantize,
                     quantize_thresholds, encode, fits_in_word)

N_RANDOM = 500
SEED = 20260802


def bits_to_hex(row, width):
    """bool[width] -> hex where bit i of the value is row[i].

    Mirrors how $readmemh loads a reg vector: rightmost hex digit is bits [3:0], so the value
    must be sum(row[i] << i) -- the same LSB-first convention the LUT address uses
    (docs/checkpoint-format.md §2).
    """
    assert row.size == width
    return np.packbits(row[::-1].astype(np.uint8), bitorder='big').tobytes().hex()


def words_to_hex(words, word_bits):
    """int[F] -> hex for a packed feature vector, feature f at [f*word_bits +: word_bits]."""
    value = 0
    mask = (1 << word_bits) - 1
    for f, w in enumerate(words):
        value |= (int(w) & mask) << (f * word_bits)      # two's complement
    return f'{value:0{len(words)*word_bits//4}x}'


def write_lines(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--vectors')
    ap.add_argument('--outdir', default=os.path.join(REPO, 'build', 'gate1'))
    # Must match what emit_encoder.py was given, or the golden model and the RTL are quantising
    # differently and Gate 1 compares two different designs. run_gate1.py passes both together.
    # Default from the dataset descriptor, not a constant. These must match what
    # emit_encoder.py used; both now resolve the same way from the same checkpoint, so the
    # default case cannot disagree.
    ap.add_argument('--word-bits', type=int, default=None)
    ap.add_argument('--frac-bits', type=int, default=None)
    args = ap.parse_args()

    ck = load_checkpoint(args.checkpoint)
    _ds = datasets.identify(ck)
    _ds.check_checkpoint(ck)
    if args.word_bits is None:
        args.word_bits = _ds.word_bits
    if args.frac_bits is None:
        args.frac_bits = _ds.frac_bits
    cfg = ck['config']
    n, num_classes, z = cfg['n'], cfg['num_classes'], cfg['thermometer_bits']
    thresholds = ck['thermometer']['thresholds'].numpy()
    n_features = thresholds.shape[0]
    width = n_features * z

    layers = []
    for i in layer_indices(ck['state_dict']):
        layers.append((extract_tables(ck['state_dict'], i),
                       *extract_wiring(ck['state_dict'], i, n)))

    vec_path = args.vectors or args.checkpoint.replace('_checkpoint.pt', '_testvectors.npz')
    v = np.load(vec_path)
    real = v['x_binarized'].astype(bool)
    real_expected = v['pred'].astype(np.int64)
    x_raw = v['x_raw']
    assert real.shape[1] == width, f'vector width {real.shape[1]} != expected {width}'

    # Guard: the golden model must reproduce PyTorch before it is trusted to label anything.
    golden_on_real, _ = forward(real, layers, num_classes)
    if not np.array_equal(golden_on_real, real_expected):
        bad = int((golden_on_real != real_expected).sum())
        print(f'ABORT: numpy golden model disagrees with PyTorch on {bad}/{real.shape[0]} '
              'real vectors. Not emitting test vectors -- fix the extractor first.')
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------- core level ----------------
    edge = np.stack([
        np.zeros(width, dtype=bool),
        np.ones(width, dtype=bool),
        np.arange(width) % 2 == 0,
        np.arange(width) % 2 == 1,
    ])
    rand = rng.integers(0, 2, size=(N_RANDOM, width), dtype=np.int8).astype(bool)
    synth = np.concatenate([edge, rand])
    synth_expected, _ = forward(synth, layers, num_classes)

    x_core = np.concatenate([real, synth])
    y_core = np.concatenate([real_expected, synth_expected])

    write_lines(os.path.join(args.outdir, 'x_binarized.hex'),
                [bits_to_hex(r, width) for r in x_core])
    write_lines(os.path.join(args.outdir, 'expected.hex'),
                [f'{int(y):X}' for y in y_core])
    write_lines(os.path.join(args.outdir, 'vec_params.vh'), [
        '// GENERATED by tb/gen_vectors.py -- do not edit.',
        f'`define N_VEC {x_core.shape[0]}',
        f'`define VEC_W {width}',
        # Width of class_idx. The testbenches hardcoded 3 (JSC's five classes), which left
        # the upper bits undriven below five classes and TRUNCATED the comparison above
        # eight -- a 10-class design was being checked on 3 of its 4 index bits.
        f'`define IDX_W {max(1, int(np.ceil(np.log2(num_classes))))}',
    ])

    # ---------------- top level ----------------
    word, frac = args.word_bits, args.frac_bits
    thr_q = quantize_thresholds(thresholds, frac)
    xq_real = quantize(x_raw, frac, word)

    used = np.unique(layers[0][1])
    used_thr = thr_q[used // z, used % z]

    # Edge cases chosen for the ENCODER specifically, not the core:
    #   zeros / min / max          rail the comparators in both directions
    #   exactly-on-threshold       q_x == T must give 0, since the comparison is strict `>`.
    #                              This is the encoder's equivalent of the argmax tie case:
    #                              a `>=` would pass everything else and fail only here.
    lo, hi = int(xq_real.min()), int(xq_real.max())
    edge_q = [np.zeros(n_features, dtype=np.int64),
              np.full(n_features, lo, dtype=np.int64),
              np.full(n_features, hi, dtype=np.int64)]
    edge_names = ['all-zeros', 'all-min', 'all-max']
    for f in range(n_features):
        # put feature f exactly on one of its own thresholds, others at zero
        f_thr = used_thr[(used // z) == f]
        if f_thr.size:
            row = np.zeros(n_features, dtype=np.int64)
            row[f] = int(np.median(f_thr))
            edge_q.append(row)
            edge_names.append(f'feature{f}-on-threshold')
    edge_q = np.stack(edge_q)

    rand_q = rng.integers(lo, hi + 1, size=(N_RANDOM, n_features), dtype=np.int64)

    xq_all = np.concatenate([xq_real, edge_q, rand_q])
    if not fits_in_word(xq_all, word):
        print(f'ABORT: quantized features do not fit {word}-bit signed: '
              f'range [{xq_all.min()}, {xq_all.max()}]')
        return 1

    bits_all = encode(xq_all, thr_q)
    y_top, _ = forward(bits_all, layers, num_classes)

    write_lines(os.path.join(args.outdir, 'x_quant.hex'),
                [words_to_hex(r, word) for r in xq_all])
    write_lines(os.path.join(args.outdir, 'expected_top.hex'),
                [f'{int(y):X}' for y in y_top])
    write_lines(os.path.join(args.outdir, 'top_params.vh'), [
        '// GENERATED by tb/gen_vectors.py -- do not edit.',
        f'`define N_TOP {xq_all.shape[0]}',
        f'`define X_W {n_features * word}',
        f'`define IDX_W {max(1, int(np.ceil(np.log2(num_classes))))}',
    ])

    # Characterization: what the fixed-point front end costs against float32.
    quant_vs_float = int((y_top[:real.shape[0]] != real_expected).sum())
    bit_diff = int((bits_all[:real.shape[0]][:, used] != real[:, used]).sum())

    print(f'core level: {x_core.shape[0]} vectors ({width} bits) -> '
          f'x_binarized.hex / expected.hex')
    print(f'  {real.shape[0]:5d} real   {edge.shape[0]:3d} edge   {rand.shape[0]:5d} random')
    print(f'top level : {xq_all.shape[0]} vectors ({n_features}x{word} bits) -> '
          f'x_quant.hex / expected_top.hex')
    print(f'  {xq_real.shape[0]:5d} real   {edge_q.shape[0]:3d} edge   '
          f'{rand_q.shape[0]:5d} random')
    print(f'  edge cases: {", ".join(edge_names[:3])}, '
          f'{len(edge_names)-3}x feature-exactly-on-threshold')
    print()
    print(f'golden model vs PyTorch on the real vectors: '
          f'{real.shape[0]}/{real.shape[0]} -- OK to proceed')
    print(f'Q{word-1-frac}.{frac} vs float32 on those same vectors: '
          f'{bit_diff} encoder bit differences, {quant_vs_float} class changes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
