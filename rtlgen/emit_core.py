"""Emit the model-specific DWN core Verilog from a checkpoint, then read it back and check it.

Scope note: written in Phase 1c as a one-shot emitter for ONE trained model -- deliberately,
since you cannot write a generator before you know what it generates. Phase 2 step 2a moved it
here and made its output directory and pipeline depth arguments, so it now serves any config in
the sweep. The read-back check below is what makes that safe to trust per config.

What is hand-written vs emitted:
  hand-written   rtl/lut_node.v, rtl/popcount.v, rtl/argmax.v  -- the structure and the
                 address/tie-break conventions, i.e. everything with a decision in it
  emitted        build/rtl/dwn_core.v                            -- 50 tables and 300 wire
                 indices, i.e. everything that is pure transcription

Transcribing 50 64-bit constants by hand teaches nothing and risks a silent one-character
error, so it is emitted. To make sure the emission itself is trustworthy, this script parses
its own output back and asserts every table and every wire index against the checkpoint,
including concatenation order.

Usage:
    python rtlgen/emit_core.py training/artifacts/<run>_checkpoint.pt
"""

import argparse
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# extract.py stays in exporter/ -- brief §11 splits the *exporter* (checkpoint -> tables,
# wiring, thresholds) from *rtlgen* (that export -> Verilog). This file is the second half.
sys.path.insert(0, os.path.join(REPO, 'exporter'))
from extract import (load_checkpoint, layer_indices, extract_tables,  # noqa: E402
                     extract_wiring)

# DEFAULT pipeline configuration -- the Phase 1 shipped depth. These are now argparse/function
# defaults rather than the value itself: `emit()` takes the depth as arguments so a Phase 2
# sweep point can vary it without editing this file (docs/dse-plan.md §1 -- a sweep point is a
# config, not a code edit). rtlgen/config.py asserts its own HardwareConfig defaults against
# these three, so the two cannot drift apart.
PIPE_LUT = 1
PIPE_POP = 1
PIPE_OUT = 1


def table_to_hex(row):
    """bool[2**n] -> integer where bit `addr` is row[addr].

    Written as an explicit loop, not np.packbits. packbits pads a partial byte on the LOW side
    with bitorder='big', so a table shorter than 8 entries came out shifted left: at n=2 the
    4-entry table [1,1,1,0] emitted as 0x70 instead of 0x07 and every entry sat at the wrong
    address. n>=3 was unaffected (2**n is then a whole number of bytes), which is why n=6 never
    showed it.

    That mattered more than a normal off-by-one: dse-plan §3 PREDICTS n=2 fails (routing
    congestion) and says such a failure is a data point marking the frontier's edge. A Gate 1
    failure caused by this bug would have looked exactly like that expected finding.

    n <= 6, so at most 64 iterations per node -- clarity beats vectorization here.
    """
    v = 0
    for addr, bit in enumerate(row):
        if bit:
            v |= 1 << addr
    return v


MAX_N = 6      # one LUT6 holds 2**6 entries; see the assertion in emit()


def emit(ck, out_path, pipe_lut=PIPE_LUT, pipe_pop=PIPE_POP, pipe_out=PIPE_OUT):
    cfg = ck['config']
    n, num_classes = cfg['n'], cfg['num_classes']

    # n > 6 would emit a 2**n-bit table into a 64-bit parameter and Verilog would TRUNCATE it
    # silently -- the design elaborates, synthesizes, and computes garbage for every address
    # above 63. Same shape as the n=2 packing bug: valid-looking, totally wrong.
    #
    # Supporting n=8 is not just a width change. Three things move together (this format string,
    # `parameter [63:0] TABLE` in rtl/lut_node.v, and verify_emitted's `64'h([0-9A-F]{16})`
    # regex) -- but more importantly one node would no longer BE one LUT6, so the architectural
    # premise brief §4 rests on, and every area estimate built on it, stops holding. That is a
    # different study, not a wider parameter.
    assert n <= MAX_N, (
        f'n={n} exceeds MAX_N={MAX_N}. A {2**n}-entry table does not fit the 64-bit TABLE '
        f'parameter and would be silently truncated. Supporting it requires changing '
        f'rtl/lut_node.v, this emitter and verify_emitted together -- and reworking the area '
        f'model, since one node would no longer map to one LUT6 (brief §4).')
    sd = ck['state_dict']

    layers = []
    for i in layer_indices(sd):
        layers.append((extract_tables(sd, i), *extract_wiring(sd, i, n)))

    # The encoder's output width, read from the thresholds themselves -- (features, z), so
    # .size is features x z. This used to be `16 * cfg['thermometer_bits']`, with JSC's feature
    # count written in. That was wrong for any other dataset and would not have failed loudly:
    # it emits a core whose input port is the wrong width, which elaborates and then disagrees
    # with the encoder. rtlgen/emit_encoder.py already derived it this way; this matches it.
    input_bits = ck['thermometer']['thresholds'].numpy().size
    final_width = layers[-1][0].shape[0]
    group = final_width // num_classes
    score_w = int(np.ceil(np.log2(group + 1)))
    idx_w = max(1, int(np.ceil(np.log2(num_classes))))

    L = []
    L.append('// GENERATED by rtlgen/emit_core.py -- do not edit by hand.')
    L.append(f"// run        : {ck.get('run_name', '(unnamed)')}")
    L.append(f"// config     : n={n}, layers={cfg['layers']}, z={cfg['thermometer_bits']}, "
             f'classes={num_classes}')
    L.append(f"// accuracy   : final {ck['results']['final_acc']:.4f} "
             f"(software, {ck['results']['best_acc']:.4f} best epoch)")
    L.append('//')
    L.append('// Takes the FULL binarized vector so it can be driven straight from the Gate 1')
    L.append('// test vectors (x_binarized). Most of those bits feed nothing -- see the report')
    L.append('// from exporter/extract.py -- and synthesis is expected to drop them. The real')
    L.append('// board design wires only the selected bits, from the thermometer encoder.')
    L.append('')
    # Required: xsim rejects a design mixing modules with and without a timescale, and the
    # testbench declares one.
    L.append('`timescale 1ns / 1ps')
    L.append('`default_nettype none')
    L.append('')
    L.append('// Pipeline stages are parameters, not hand-placement: depth is a Phase 2 sweep')
    L.append('// axis (brief §10). Set any to 0 to compile that register out. LATENCY is the')
    L.append('// sum of the enabled stages and is emitted into dwn_core_params.vh for the')
    L.append('// testbench, so the two can never disagree about it.')
    L.append('//')
    L.append('// Placement follows measurement, not intuition: out-of-context synthesis put')
    L.append('// 10.194 ns in this module against 2.962 ns in the encoder, so the depth is in')
    L.append('// the popcount and argmax trees. Hence a stage after each of them.')
    L.append('module dwn_core #(')
    L.append(f'    parameter integer PIPE_LUT = {pipe_lut},   // after each LUT layer')
    L.append(f'    parameter integer PIPE_POP = {pipe_pop},   // after the popcounts')
    L.append(f'    parameter integer PIPE_OUT = {pipe_out}    // after the argmax')
    L.append(')(')
    L.append('    input  wire clk,')
    L.append(f'    input  wire [{input_bits-1}:0] x,')
    L.append(f'    output wire [{idx_w-1}:0]    class_idx')
    L.append(');')
    L.append('')

    prev = 'x'
    for li, (tables, wiring, kind) in enumerate(layers):
        count = tables.shape[0]
        L.append(f'    // ---- layer {li}: {count} nodes, {kind} mapping ----')
        L.append(f'    wire [{count-1}:0] layer{li};')
        for j in range(count):
            # Concatenation is MSB-first, and slot 0 must land on addr[0], so the slots are
            # listed in REVERSE order here. This is the single most reversible-looking
            # decision in the whole design (docs/checkpoint-format.md §2).
            slots = ', '.join(f'{prev}[{b}]' for b in reversed(wiring[j].tolist()))
            L.append(f'    lut_node #(.N({n}), .TABLE(64\'h{table_to_hex(tables[j]):016X}))')
            L.append(f'        u_l{li}_n{j} (.addr({{{slots}}}), .out(layer{li}[{j}]));')
        L.append('')
        L.append(f'    wire [{count-1}:0] layer{li}_q;')
        L.append(f'    pipe_reg #(.WIDTH({count}), .ENABLE(PIPE_LUT)) u_pipe_l{li} '
                 f'(.clk(clk), .d(layer{li}), .q(layer{li}_q));')
        L.append('')
        prev = f'layer{li}_q'

    L.append('    // ---- output stage: contiguous per-class popcounts, then argmax ----')
    L.append(f'    wire [{num_classes*score_w-1}:0] scores;')
    for c in range(num_classes):
        lo, hi = c * group, (c + 1) * group - 1
        L.append(f'    popcount #(.WIDTH({group})) u_pc{c} '
                 f'(.bits({prev}[{hi}:{lo}]), .count(scores[{c*score_w+score_w-1}:{c*score_w}]));')
    L.append('')
    L.append(f'    wire [{num_classes*score_w-1}:0] scores_q;')
    L.append(f'    pipe_reg #(.WIDTH({num_classes*score_w}), .ENABLE(PIPE_POP)) u_pipe_pop '
             f'(.clk(clk), .d(scores), .q(scores_q));')
    L.append('')
    L.append(f'    wire [{idx_w-1}:0] idx;')
    L.append(f'    argmax #(.K({num_classes}), .W({score_w})) u_argmax '
             f'(.scores_flat(scores_q), .index(idx));')
    L.append('')
    L.append(f'    pipe_reg #(.WIDTH({idx_w}), .ENABLE(PIPE_OUT)) u_pipe_out '
             f'(.clk(clk), .d(idx), .q(class_idx));')
    L.append('')
    L.append('endmodule')
    L.append('')
    L.append('`default_nettype wire')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(L) + '\n')

    # Latency emitted alongside the RTL so the testbench reads it from the same source that
    # generated the pipeline. Hardcoding it in the testbench is how a depth change silently
    # turns into an off-by-one comparison against the wrong vector.
    core_latency = pipe_lut * len(layers) + pipe_pop + pipe_out
    # The individual stage enables go here too, not just their sum. dwn_top instantiates
    # dwn_core with .PIPE_LUT(PIPE_LUT) etc., so dwn_top's OWN defaults override the core's --
    # emitting a 0 here and leaving dwn_top at 1 would silently put the stage back. Publishing
    # them makes emit_encoder read the depth rather than be told it a second time.
    with open(os.path.join(os.path.dirname(out_path), 'dwn_core_params.vh'), 'w') as f:
        f.write('// GENERATED by rtlgen/emit_core.py -- do not edit.\n')
        f.write(f'`define DWN_CORE_LATENCY {core_latency}\n')
        f.write(f'`define DWN_CORE_PIPE_LUT {pipe_lut}\n')
        f.write(f'`define DWN_CORE_PIPE_POP {pipe_pop}\n')
        f.write(f'`define DWN_CORE_PIPE_OUT {pipe_out}\n')

    return layers, input_bits, group, score_w, core_latency


def verify_emitted(path, layers, n):
    """Parse the emitted Verilog back and check it against the checkpoint.

    Without a simulator this is the only real check available, and it is a meaningful one:
    it catches a wrong table, a wrong wire index, and -- most importantly -- a reversed
    address concatenation, which is the failure mode that would otherwise survive all the
    way to Gate 1.
    """
    src = open(path).read()
    pat = re.compile(
        r"lut_node #\(\.N\(\d+\), \.TABLE\(64'h([0-9A-F]{16})\)\)\s*"
        r"u_l(\d+)_n(\d+) \(\.addr\(\{([^}]*)\}\)",
        re.MULTILINE)

    seen = 0
    for m in pat.finditer(src):
        tbl_hex, li, j = m.group(1), int(m.group(2)), int(m.group(3))
        tables, wiring, _ = layers[li]

        expect_hex = f'{table_to_hex(tables[j]):016X}'
        assert tbl_hex == expect_hex, (
            f'layer {li} node {j}: table {tbl_hex} != expected {expect_hex}')

        idx = [int(x) for x in re.findall(r'\[(\d+)\]', m.group(4))]
        # emitted MSB-first, so reversing recovers slot order (slot 0 first)
        assert idx[::-1] == wiring[j].tolist(), (
            f'layer {li} node {j}: wiring {idx[::-1]} != expected {wiring[j].tolist()} '
            '-- address bit order is wrong')
        seen += 1

    total = sum(t.shape[0] for t, _, _ in layers)
    assert seen == total, f'parsed {seen} nodes, expected {total}'
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    # A DIRECTORY, matching emit_encoder.py -- both emitters write several files into the same
    # place (dwn_core.v, dwn_core_params.vh, thermometer_encoder.v, dwn_top.v) and Phase 2 hands
    # them one `Config.rtl_dir`. This was `--out <file>` in Phase 1, which made the two emitters
    # take different kinds of argument for the same idea.
    ap.add_argument('--outdir', default=os.path.join(REPO, 'build', 'rtl'),
                    help='where to write dwn_core.v and dwn_core_params.vh '
                         '(default: the Phase 1 location)')
    # Pipeline depth is a Group B sweep axis (docs/dse-plan.md §3): it changes timing and FF
    # count at zero GPU cost, since accuracy is invariant. 0 compiles a stage out entirely.
    ap.add_argument('--pipe-lut', type=int, default=PIPE_LUT,
                    help='register after each LUT layer (default %(default)s)')
    ap.add_argument('--pipe-pop', type=int, default=PIPE_POP,
                    help='register after the popcounts (default %(default)s)')
    ap.add_argument('--pipe-out', type=int, default=PIPE_OUT,
                    help='register after the argmax (default %(default)s)')
    args = ap.parse_args()

    out_path = os.path.join(args.outdir, 'dwn_core.v')
    ck = load_checkpoint(args.checkpoint)
    layers, input_bits, group, score_w, core_latency = emit(
        ck, out_path, args.pipe_lut, args.pipe_pop, args.pipe_out)

    rel = os.path.relpath(out_path, REPO)
    total = sum(t.shape[0] for t, _, _ in layers)
    print(f'wrote {rel}')
    print(f'  {total} lut_node instances, {input_bits}-bit input, '
          f'{len(layers)} layer(s)')
    print(f'  output stage: {ck["config"]["num_classes"]} x popcount({group}) '
          f'-> {score_w}-bit scores -> argmax')
    print(f'  pipeline: LUT={args.pipe_lut} POP={args.pipe_pop} OUT={args.pipe_out} '
          f'-> core latency {core_latency} cycles, II=1')
    print()

    seen = verify_emitted(out_path, layers, ck['config']['n'])
    print(f'read-back check: {seen}/{total} nodes match the checkpoint '
          '(tables + wire indices + address bit order)')
    print('  PASS')
    print()
    print('This check proves the file SAYS what the checkpoint says. It proves nothing about')
    print('how the Verilog BEHAVES -- only Gate 1 does that. Run: scripts/run_gate1.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
