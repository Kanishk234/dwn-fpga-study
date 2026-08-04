"""EXPERIMENT: how much pipeline does dwn_top actually need?

The shipped design has 4 stages and closes 100 MHz with +3.572 ns to spare. That slack is not
free -- every stage costs flip-flops and a cycle of latency -- so the question is whether a
shallower pipeline still closes.

This is the pipeline-depth axis from brief §10, collected early. Depth is already a `pipe_reg`
ENABLE parameter, so a config is a synthesis generic, not a code edit; that is the whole reason
the stages were built that way.

What matters here is NOT just "does it fit". Latency is a headline number for a trigger
application (brief §7: the LHC's Level-1 trigger has a hard deadline), and II stays 1 in every
variant, so a shallower pipe is strictly better ON LATENCY as long as timing closes. The
trade is flip-flops and margin against cycles.

Reported in cycles as well as ns, per brief §6 -- the paper's clock speeds are speed-graded
parts and do not transfer to a -1 Artix-7.

Usage:
    python scripts/experiment_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gate1 import REPO, find_vivado_bin  # noqa: E402
from run_synth import (BOARD_PERIOD_NS, DEFAULT_PART, DEVICE_LUTS,  # noqa: E402
                       parse_utilization, parse_wns, run_one)

SOURCES = ['rtl/lut_node.v', 'rtl/popcount.v', 'rtl/argmax.v', 'rtl/pipe_reg.v',
           'rtl/gen/dwn_core.v', 'rtl/gen/thermometer_encoder.v', 'rtl/gen/dwn_top.v']

# Each variant drops a different stage. ENC is never dropped: the encoder is 202 comparators
# feeding 3200 wires, and merging it into the LUT layer is the one combination most likely to
# blow the critical path.
# (label, tag, generics, latency). Tags are explicit because two variants are both "3-stage"
# and would otherwise share an output directory and silently overwrite each other's reports.
VARIANTS = [
    ('4-stage (shipped)',       'p4',        [],                                        4),
    ('3-stage: no POP reg',     'p3_nopop',  ['PIPE_POP=0'],                            3),
    ('3-stage: no OUT reg',     'p3_noout',  ['PIPE_OUT=0'],                            3),
    ('2-stage: no POP, no OUT', 'p2',        ['PIPE_POP=0', 'PIPE_OUT=0'],              2),
    ('1-stage: encoder only',   'p1',        ['PIPE_LUT=0', 'PIPE_POP=0', 'PIPE_OUT=0'], 1),
]


def main():
    vivado_bin = find_vivado_bin(None)
    out_root = os.path.join(REPO, 'build', 'experiments', 'pipeline')

    print(f'part       : {DEFAULT_PART}')
    print(f'constraint : {BOARD_PERIOD_NS:.1f} ns ({1000/BOARD_PERIOD_NS:.0f} MHz board clock)')
    print()

    rows = []
    for label, tag, generics, latency in VARIANTS:
        print(f'=== {label} ===')
        ok, out_dir = run_one(vivado_bin, 'dwn_top', SOURCES, DEFAULT_PART, out_root,
                              generics=generics, name=tag)
        if not ok:
            return 1
        util = parse_utilization(os.path.join(out_dir, 'utilization.rpt'))
        wns = parse_wns(os.path.join(out_dir, 'timing_summary.rpt'))
        rows.append((label, latency, util.get('luts'), util.get('ff'), wns))
        print(f'  LUTs {util.get("luts")}  FF {util.get("ff")}  WNS {wns:+.3f}'
              if wns is not None else '  (no timing)')
        print()

    print('=' * 88)
    print(f'{"variant":28s} {"cycles":>7} {"LUTs":>7} {"FF":>6} {"WNS ns":>9} '
          f'{"Fmax MHz":>9} {"@100MHz":>9}')
    print('-' * 88)
    for label, latency, luts, ff, wns in rows:
        if wns is None:
            print(f'{label:28s} {latency:>7} {luts:>7} {ff:>6} {"?":>9} {"?":>9} {"?":>9}')
            continue
        fmax = 1000.0 / (BOARD_PERIOD_NS - wns)
        # Latency in ns at the board clock, which is what a trigger deadline cares about.
        ns_at_100 = latency * BOARD_PERIOD_NS
        print(f'{label:28s} {latency:>7} {luts:>7} {ff:>6} {wns:>+9.3f} {fmax:>9.1f} '
              f'{ns_at_100:>8.0f}n')
    print('=' * 88)
    print(f'device: {DEVICE_LUTS} LUTs. II=1 in every variant, so cycles == latency and')
    print('throughput is one classification per clock regardless of depth.')
    print()

    ok_rows = [r for r in rows if r[4] is not None and r[4] >= 0]
    if ok_rows:
        best = min(ok_rows, key=lambda r: r[1])
        base = rows[0]
        print(f'Shallowest variant that still meets 100 MHz: {best[0]} '
              f'({best[1]} cycles, {best[3]} FF)')
        if best[1] < base[1]:
            print(f'  vs shipped: {base[1]-best[1]} fewer cycles '
                  f'({(base[1]-best[1])*BOARD_PERIOD_NS:.0f} ns less latency), '
                  f'{base[3]-best[3]} fewer FF, '
                  f'{best[4]:+.3f} ns slack remaining')
            print('  Not adopted automatically -- changing depth changes DWN_TOP_LATENCY, so')
            print('  benchmark_fsm realigns and Gate 1 must be re-run. This is a data point.')
    failing = [r for r in rows if r[4] is not None and r[4] < 0]
    for label, latency, luts, ff, wns in failing:
        print(f'FAILS at 100 MHz: {label} by {-wns:.3f} ns '
              f'-> {1000.0/(BOARD_PERIOD_NS - wns):.1f} MHz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
