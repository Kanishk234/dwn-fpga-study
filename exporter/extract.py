"""Extract LUT tables, wiring, and thresholds from a trained DWN checkpoint.

This is the front half of the exporter (repo layout: `exporter/`). It does two jobs:

1. Turn a checkpoint into the three things hardware needs -- per-node truth tables, per-node
   input wiring, and the thermometer thresholds those inputs come from.
2. **Prove the extraction is right**, by running a pure-numpy forward pass over the extracted
   artifacts and checking it reproduces the `pred` array PyTorch wrote into the test-vector
   file, on every vector.

Job 2 is not optional. Every structural fact this file relies on -- LSB-first address order,
`argmax(dim=0)` for learnable wiring, `> 0` table thresholding, contiguous GroupSum groups --
was read out of `third_party/DWN` and written up in `docs/checkpoint-format.md`, but read
correctly is not the same as implemented correctly. If the numpy model disagrees with PyTorch
here, the Verilog written from these tables would be wrong in exactly the same way, and Gate 1
would find it much later and much more expensively.

Usage:
    python exporter/extract.py training/artifacts/<run>_checkpoint.pt
"""

import argparse
import re
import sys

import numpy as np
import torch


def load_checkpoint(path):
    # weights_only=False: the checkpoint holds the config dict, class-name list, and results
    # alongside the tensors. Trusted input -- we produced it.
    return torch.load(path, map_location='cpu', weights_only=False)


def layer_indices(state_dict):
    """Layer indices, in order, from the `<i>.luts` keys."""
    found = {int(m.group(1)) for k in state_dict
             if (m := re.fullmatch(r'(\d+)\.luts', k))}
    return sorted(found)


def extract_tables(state_dict, i):
    """(output_size, 2**n) float -> bool truth tables.

    docs/checkpoint-format.md §1: the hardware bit is `luts[j][addr] > 0`. Strictly greater --
    an exact 0.0 emits 0. Unlikely in a trained model, but Gate 1 covers edge cases.
    """
    luts = state_dict[f'{i}.luts'].numpy()
    return luts > 0.0


def extract_wiring(state_dict, i, n):
    """(output_size, n) int -> which input bit feeds each node slot.

    Two completely different representations, per docs/checkpoint-format.md §3.
    """
    learnable_key = f'{i}.mapping.weights'
    if learnable_key in state_dict:
        # §3a. weights is (input_size, output_size*n); argmax over the INPUT-BIT axis gives
        # one input index per node slot. Node j slot k -> index [j*n + k].
        weights = state_dict[learnable_key].numpy()
        flat = weights.argmax(axis=0)
        assert flat.size % n == 0
        return flat.reshape(-1, n).astype(np.int64), 'learnable'

    # §3b. Already the final wiring, no transformation.
    #
    # NB: `<i>._LUTLayer__dummy_mapping` is deliberately NOT consulted. It has the same shape
    # and dtype as a real mapping but is only arange() reshaped, so keying off shape instead
    # of off the presence of `.mapping.weights` yields a valid-looking, totally wrong export.
    return state_dict[f'{i}.mapping'].numpy().astype(np.int64), 'fixed'


def lut_forward(x_bits, tables, wiring):
    """One LUT layer. x_bits: (N, input_size) bool. Returns (N, output_size) bool."""
    # docs/checkpoint-format.md §2: slot l contributes bit l of the address -- LSB FIRST.
    # Reversing this gives a model that is wrong on most inputs but structurally plausible.
    gathered = x_bits[:, wiring]                       # (N, output_size, n)
    n = wiring.shape[1]
    weights = (1 << np.arange(n, dtype=np.int64))      # [1, 2, 4, 8, 16, 32]
    addr = (gathered.astype(np.int64) * weights).sum(axis=2)   # (N, output_size)
    return tables[np.arange(tables.shape[0]), addr]


def group_sum_argmax(x_bits, num_classes):
    """GroupSum + argmax. docs/checkpoint-format.md §4.

    Groups are contiguous and in order. `tau` is a uniform divisor across classes, so it
    cannot change the argmax and the hardware never needs it -- popcount and compare only.
    """
    width = x_bits.shape[1]
    assert width % num_classes == 0, (
        f'final layer width {width} not divisible by num_classes {num_classes}; '
        'GroupSum would zero-pad silently')
    scores = x_bits.reshape(x_bits.shape[0], num_classes, width // num_classes).sum(axis=2)
    return scores.argmax(axis=1), scores


def forward(x_bits, layers, num_classes):
    for tables, wiring, _ in layers:
        x_bits = lut_forward(x_bits, tables, wiring)
    return group_sum_argmax(x_bits, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--vectors', help='test-vector .npz (default: alongside the checkpoint)')
    args = ap.parse_args()

    ck = load_checkpoint(args.checkpoint)
    cfg = ck['config']
    n = cfg['n']
    num_classes = cfg['num_classes']
    sd = ck['state_dict']

    print(f"run          : {ck.get('run_name', '(unnamed)')}")
    print(f"config       : n={n}, layers={cfg['layers']}, z={cfg['thermometer_bits']}, "
          f"classes={num_classes}")
    print(f"accuracy     : final {ck['results']['final_acc']:.4f}  "
          f"best {ck['results']['best_acc']:.4f}")
    print()

    layers = []
    for i in layer_indices(sd):
        tables = extract_tables(sd, i)
        wiring, kind = extract_wiring(sd, i, n)
        assert tables.shape[0] == wiring.shape[0], 'table/wiring node count mismatch'
        assert tables.shape[1] == 2 ** n, f'expected {2**n} entries per table'
        layers.append((tables, wiring, kind))
        print(f'layer {i}: {tables.shape[0]} nodes, {kind} mapping, '
              f'tables {tables.shape}, wiring {wiring.shape}, '
              f'reads input bits [{wiring.min()}..{wiring.max()}]')

    thresholds = ck['thermometer']['thresholds'].numpy()
    print(f'thresholds   : {thresholds.shape} (features x bits)')

    # How much of the thermometer survives into hardware. Only bits the first layer's wiring
    # actually references need a comparator; the rest feed no node. This is the claim in
    # docs/paper-configs.md that makes z=200 plausible on a small part.
    used = np.unique(layers[0][1])
    total_bits = thresholds.size
    print(f'encoder      : {used.size} of {total_bits} thermometer bits are wired to a node '
          f'({100*used.size/total_bits:.1f}%)')
    print(f'               -> at most {used.size} comparators, not {total_bits}')
    print()

    # ---- node 0, concretely: this is what gets hand-written as Verilog in Phase 1c ----
    tables0, wiring0, _ = layers[0]
    word = int(np.packbits(tables0[0][::-1], bitorder='big').tobytes().hex(), 16)
    print('--- layer 0, node 0 (the 1c hand-write target) ---')
    print(f'  reads input bits {wiring0[0].tolist()}   (slot 0 = address LSB)')
    for slot, bit in enumerate(wiring0[0].tolist()):
        feat, thr_idx = divmod(bit, thresholds.shape[1])
        print(f'    slot {slot} -> bit {bit:4d} = feature {feat:2d} > {thresholds[feat, thr_idx]:+.6f}')
    print(f'  TABLE = 64\'h{word:016X}')
    print(f'  as bits[0..15] = {tables0[0][:16].astype(int).tolist()}')
    print()

    # ---- verification ----
    vec_path = args.vectors or args.checkpoint.replace('_checkpoint.pt', '_testvectors.npz')
    v = np.load(vec_path)
    x_bits = v['x_binarized'].astype(bool)
    expected = v['pred']

    pred, scores = forward(x_bits, layers, num_classes)
    matches = int((pred == expected).sum())
    total = expected.size

    # Ties matter for Gate 1 and for the RTL argmax: when two classes share the top score the
    # winner is decided by tie-break order, not by the data. numpy and torch both take the
    # lowest index, so the RTL must too.
    top = scores.max(axis=1, keepdims=True)
    ties = int(((scores == top).sum(axis=1) > 1).sum())

    print('--- verification against PyTorch ---')
    print(f'  numpy forward vs checkpoint pred: {matches}/{total}')
    print(f'  vectors with a tied top score   : {ties}'
          + ('  <- RTL argmax must break ties toward the LOWEST class index' if ties else ''))

    if matches != total:
        bad = np.flatnonzero(pred != expected)[:5]
        print(f'  FAIL. first mismatches at {bad.tolist()}')
        print(f'    expected {expected[bad].tolist()}, got {pred[bad].tolist()}')
        print('  Do not write RTL against this extraction. Re-check '
              'docs/checkpoint-format.md §2 (address bit order) and §3 (wiring).')
        return 1

    print('  PASS -- the extraction reproduces PyTorch exactly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
