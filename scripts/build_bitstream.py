"""Build the Basys 3 bitstream for dwn_basys3_top.

Everything measured so far has been OUT-OF-CONTEXT: no pins, no I/O buffers, no clock network.
Those numbers answer "does the logic fit and close timing". This answers "does the actual
design, wired to actual pins, build" -- and it is the last thing standing between simulation
and Gate 1b.

Expect the numbers to differ from the out-of-context reports: I/O buffers add LUT-adjacent
resources, the clock reaches the whole die through a BUFG, and pad delays eat margin. The
out-of-context figures stay the comparable-to-the-paper ones (the paper is out-of-context too);
these are the honest "what is really on the board" figures. Report both, never one as the other.

Usage:
    .venv\\Scripts\\python.exe scripts/build_bitstream.py
    .venv\\Scripts\\python.exe scripts/build_bitstream.py --open-hw    # print programming steps
"""

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exporter'))
from run_gate1 import DEFAULT_CHECKPOINT, REPO, find_vivado_bin  # noqa: E402
from extract import load_checkpoint  # noqa: E402
sys.path.insert(0, REPO)
import datasets  # noqa: E402
from run_synth import (BOARD_PERIOD_NS, DEFAULT_PART, DEVICE_LUTS,  # noqa: E402
                       parse_utilization, parse_wns, run_one)

TOP = 'dwn_basys3_top'


def check_rtl_matches(rtl_dir, ck, n_features, word_bits, n_classes, label_w):
    """The generated RTL must be THIS checkpoint's, not whatever was emitted last.

    WHY THIS EXISTS. This script derives the harness -- record size, store width, label width --
    from the checkpoint, and prints a confident `dataset: <name>` banner from it. It does NOT
    emit the network; that comes from `rtl_dir`, whatever happens to be sitting there. So a
    build can pair one dataset's harness with another dataset's core and report the first one.

    That is not hypothetical. On 2026-08-12 a `verify_phase1.py --with-board` run regenerated
    JSC's RTL into build/rtl; the MNIST bitstream built minutes later wrapped an MNIST-shaped
    1,569-byte loader around JSC's 256-bit core. Verilog TRUNCATES a too-wide connection rather
    than erroring, so the build succeeded, met timing, programmed, ran, and agreed with the
    golden model on 1,035 of 10,000 samples -- chance for ten classes. Nothing failed loudly.

    Three checks, because no one of them is sufficient:
      - x_flat width catches a different DATASET (the failure above)
      - class_idx width catches a different CLASS COUNT, which truncates the answer rather
        than the input, and can still look plausible
      - the lut_node count catches a different MODEL at the SAME dimensions -- two MNIST
        checkpoints have identical port widths, so widths alone would build either one happily
    """
    top_v = os.path.join(rtl_dir, 'dwn_top.v')
    core_v = os.path.join(rtl_dir, 'dwn_core.v')
    if not (os.path.exists(top_v) and os.path.exists(core_v)):
        return                                  # the `missing` check above reports this better

    src = open(top_v).read()
    problems = []

    # x_flat is features x word_bits -- NOT the store's DATA_W, which pads each feature to a
    # whole byte. At 9 bits those differ (7,056 against 12,544) and conflating them would make
    # this check fail on every correct MNIST build.
    m = re.search(r'input\s+wire\s*\[(\d+):0\]\s*x_flat', src)
    want = n_features * word_bits
    if m and int(m.group(1)) + 1 != want:
        problems.append(f'x_flat is {int(m.group(1))+1} bits, checkpoint implies {want} '
                        f'({n_features} features x {word_bits}-bit)')

    m = re.search(r'output\s+wire\s*\[(\d+):0\]\s*class_idx', src)
    if m and int(m.group(1)) + 1 != label_w:
        problems.append(f'class_idx is {int(m.group(1))+1} bits, checkpoint implies {label_w} '
                        f'({n_classes} classes)')

    want_nodes = sum(ck['config']['layers'])
    got_nodes = len(re.findall(r'lut_node #', open(core_v).read()))
    if got_nodes != want_nodes:
        problems.append(f'dwn_core has {got_nodes} lut_node instances, checkpoint has '
                        f'{want_nodes} ({ck["config"]["layers"]})')

    if problems:
        raise SystemExit(
            'GENERATED RTL DOES NOT MATCH THE CHECKPOINT\n\n  ' + '\n  '.join(problems) +
            f'\n\n{os.path.relpath(rtl_dir, REPO)} holds a different model. Regenerate it from '
            'the checkpoint you meant:\n\n'
            '  .venv\\Scripts\\python.exe scripts\\run_gate1.py --checkpoint <checkpoint>\n\n'
            'run_gate1.py re-emits AND proves the result bit-exact, so it is the right way to '
            'refill the directory. Building anyway would pair this harness with that model and '
            'report these dimensions -- silently, because Verilog truncates.')

# XC7A35T: 50 x 36 Kbit block RAMs. The vector store is the only thing that uses them, so this
# is the whole budget it competes for.
DEVICE_BRAM_BITS = 50 * 36 * 1024
XDC = 'constraints/basys3.xdc'

_PRIM = ['rtl/lut_node.v', 'rtl/popcount.v', 'rtl/argmax.v', 'rtl/pipe_reg.v']
_HARNESS = [
    'harness/uart_tx.v', 'harness/uart_rx.v', 'harness/uart_loader.v',
    'harness/vector_store.v', 'harness/benchmark_fsm.v', 'harness/seg7.v',
    'harness/dwn_basys3_top.v',
]


def sources(rtl_dir=None):
    """Board sources, with generated RTL taken from `rtl_dir` (default: build/rtl)."""
    g = (os.path.relpath(rtl_dir, REPO).replace('\\', '/') if rtl_dir else 'build/rtl')
    return (_PRIM + [f'{g}/dwn_core.v', f'{g}/thermometer_encoder.v', f'{g}/dwn_top.v'] +
            _HARNESS)


SOURCES = sources()


def main():
    ap = argparse.ArgumentParser(description='Build the Basys 3 bitstream.')
    ap.add_argument('--part', default=DEFAULT_PART)
    ap.add_argument('--vivado-bin', default=None)
    ap.add_argument('--outdir', default=os.path.join(REPO, 'build', 'bitstream'))
    ap.add_argument('--rtl-dir', default=None,
                    help='where to read generated RTL from (default: build/rtl)')
    # Overrides the BAUD parameter without editing RTL. Sweeping baud IS the I/O-wall
    # characterization (brief §14), so it has to be a command-line knob, not a source edit.
    # 100 MHz divides exactly at 1M(100), 2M(50), 4M(25), 5M(20), 10M(10).
    ap.add_argument('--baud', type=int, default=None,
                    help='override the BAUD parameter (default: whatever the RTL says)')
    # Harness dimensions are DERIVED from the checkpoint, not typed in. They were defaults in
    # dwn_basys3_top.v sized for JSC (DATA_W=256, LABEL_W=3, DEPTH=1024), which would have been
    # silently wrong for any other dataset -- the loader would still have accepted 33-byte
    # records and written garbage.
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                    help='sets DATA_W / LABEL_W from the model (default: the Phase 1 config)')
    ap.add_argument('--word-bits', type=int, default=None,
                    help='input word width; must match what emit_encoder was given '
                         '(default: the dataset descriptor)')
    ap.add_argument('--bram-budget', type=float, default=0.15,
                    help='fraction of block RAM the vector store may use (default %(default)s, '
                         'which is what JSC used at DEPTH=1024)')
    args = ap.parse_args()

    vivado_bin = find_vivado_bin(args.vivado_bin)
    src = sources(args.rtl_dir)

    # dwn_basys3_top includes dwn_top_params.vh for the pipeline latency, so the generated RTL
    # must already exist. Fail here with a useful message rather than inside Vivado.
    missing = [s for s in src + [XDC] if not os.path.exists(os.path.join(REPO, s))]
    if missing:
        raise SystemExit('missing:\n  ' + '\n  '.join(missing) +
                         '\n\nGenerated RTL absent? Run scripts/run_gate1.py first -- it '
                         'regenerates it from the checkpoint AND proves it correct.')

    print(f'top    : {TOP}')
    print(f'part   : {args.part}')
    print(f'xdc    : {XDC}')
    print(f'clock  : {BOARD_PERIOD_NS:.1f} ns ({1000/BOARD_PERIOD_NS:.0f} MHz)')
    print()

    # ---- harness dimensions, derived from the model ----
    ck = load_checkpoint(args.checkpoint)
    ds = datasets.identify(ck)
    ds.check_checkpoint(ck)
    if args.word_bits is None:
        args.word_bits = ds.word_bits
    n_features = ck['thermometer']['thresholds'].numpy().shape[0]
    n_classes = ck['config']['num_classes']
    per_feat = -(-args.word_bits // 8)          # ceiling: a 9-bit word travels as 2 bytes
    print(f'dataset: {ds.name}  ({ds.fixed_point} by descriptor)')
    data_w = n_features * per_feat * 8
    label_w = max(1, math.ceil(math.log2(n_classes)))
    bits_per_vec = data_w + label_w
    depth = 1 << int(math.floor(math.log2(max(
        1, DEVICE_BRAM_BITS * args.bram_budget / bits_per_vec))))
    addr_w = max(1, int(math.log2(depth)))

    # Before anything is printed as fact about the design, prove the generated RTL is this
    # checkpoint's. Everything below describes the harness; the network comes from rtl_dir.
    check_rtl_matches(args.rtl_dir or os.path.join(REPO, 'build', 'rtl'),
                      ck, n_features, args.word_bits, n_classes, label_w)

    print(f'model  : {n_features} features x {args.word_bits}-bit, {n_classes} classes')
    print(f'record : {n_features * per_feat + 1} bytes ({per_feat} B/feature + 1 label)')
    print(f'store  : DATA_W={data_w} LABEL_W={label_w} DEPTH={depth} ADDR_W={addr_w}  '
          f'({depth * bits_per_vec / 1024:.0f} Kbit, '
          f'{depth * bits_per_vec / DEVICE_BRAM_BITS:.0%} of block RAM)')
    if depth < 256:
        print(f'         NOTE only {depth} vectors fit on-chip, so a full test set needs many '
              f'load/run batches. That is throughput, not correctness.')
    print()

    generics = [f'FEATURES={n_features}', f'WORD_BITS={args.word_bits}',
                f'DATA_W={data_w}', f'LABEL_W={label_w}',
                f'DEPTH={depth}', f'ADDR_W={addr_w}']
    if args.baud:
        generics.append(f'BAUD={args.baud}')
    if args.baud:
        div = 100_000_000 / args.baud
        print(f'baud   : {args.baud:,} (override) -- {div:g} clocks/bit'
              + ('' if div == int(div) else '  <- NOT an integer divisor, expect errors'))
        print()

    ok, out_dir = run_one(vivado_bin, TOP, src, args.part, args.outdir,
                          impl=True, xdc=XDC, name='basys3', generics=generics)
    if not ok:
        return 1

    bit = os.path.join(out_dir, f'{TOP}.bit')
    util = parse_utilization(os.path.join(out_dir, 'utilization_routed.rpt'))
    wns = parse_wns(os.path.join(out_dir, 'timing_summary_routed.rpt'))

    print()
    print('=' * 64)
    print('BITSTREAM (post-route, with pins and I/O buffers)')
    print('-' * 64)
    luts = util.get('luts')
    if luts is not None:
        print(f'  LUTs  {luts:6d}  ({100*luts/DEVICE_LUTS:.2f}% of {DEVICE_LUTS})')
    print(f'  FF    {util.get("ff", "?"):>6}')
    print(f'  BRAM  {util.get("bram", "?"):>6}')
    print(f'  DSP   {util.get("dsp", "?"):>6}')
    if wns is not None:
        verdict = 'MEETS' if wns >= 0 else 'FAILS'
        print(f'  WNS   {wns:+6.3f} ns  -> {verdict} timing at '
              f'{1000/BOARD_PERIOD_NS:.0f} MHz')
        if wns < 0:
            print()
            print('  Timing FAILED. Do not program this: a design that misses setup will')
            print('  produce wrong answers intermittently, which looks like a logic bug and')
            print('  is far harder to diagnose than a clean failure. Out-of-context had')
            print('  +3.200 ns, so the I/O buffers and clock network ate more than expected.')
    print('=' * 64)

    if os.path.exists(bit):
        print(f'\nbitstream: {os.path.relpath(bit, REPO)} '
              f'({os.path.getsize(bit)/1e6:.2f} MB)')
        print()
        print('To program (board plugged in, switch on):')
        print(f'  1. {os.path.join(vivado_bin, "vivado")}   -> Open Hardware Manager')
        print('  2. Open Target -> Auto Connect')
        print(f'  3. Program Device -> {os.path.relpath(bit, REPO)}')
        print()
        print('Then check the link before anything else:')
        print('  .venv\\Scripts\\python.exe scripts\\host.py --port COM? --ping')
        print('  (Device Manager -> Ports, the Basys 3 shows as a USB Serial Port)')
        return 0

    print('\nNo bitstream produced -- check the Vivado log in the output directory.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
