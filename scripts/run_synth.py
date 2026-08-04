"""Synthesize the design out-of-context and report area, encoder and core separately.

This produces the number brief §6 says to measure rather than inherit, and the number the DWN
paper never reported: what the thermometer encoder actually costs. Their resource totals
exclude the binarization front end entirely, and Mecik & Kumm measured that omission at up to
3.2x. On an xcvu9p it did not matter. This model is 50 LUT6 nodes and 202 comparators on a
20,800-LUT part, so it may well dominate.

Three separate runs rather than one, because separate runs give unambiguous numbers:
    dwn_core             the LUT network alone
    thermometer_encoder  the 202 comparators alone
    dwn_top              the two together (which is NOT necessarily the sum -- synthesis can
                         share logic across the boundary, and that difference is itself worth
                         knowing)

Phase 2's sweep reuses run_one(); this is the same flow, called in a loop over configs.

Usage:
    .venv\\Scripts\\python.exe scripts/run_synth.py
    .venv\\Scripts\\python.exe scripts/run_synth.py --part xc7a35tcpg236-1
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gate1 import REPO, find_vivado_bin, run, tool  # noqa: E402

# Basys 3 = Artix-7 XC7A35T-1CPG236C. The -1 speed grade is the slow one (brief §6): the
# paper's 827-3030 MHz figures are speed-graded parts and will not transfer.
DEFAULT_PART = 'xc7a35tcpg236-1'

TARGETS = [
    ('dwn_core', ['rtl/lut_node.v', 'rtl/popcount.v', 'rtl/argmax.v', 'rtl/pipe_reg.v',
                  'rtl/gen/dwn_core.v']),
    ('thermometer_encoder', ['rtl/gen/thermometer_encoder.v']),
    ('dwn_top', ['rtl/lut_node.v', 'rtl/popcount.v', 'rtl/argmax.v', 'rtl/pipe_reg.v',
                 'rtl/gen/dwn_core.v', 'rtl/gen/thermometer_encoder.v',
                 'rtl/gen/dwn_top.v']),
]

# Total LUTs on the part, for the "% of device" column that decides what fits.
DEVICE_LUTS = 20800

# The Basys 3's on-board oscillator is 100 MHz. Constraining at exactly that makes the slack
# answer the question that matters: does this run on the board as-is?
BOARD_PERIOD_NS = 10.0


def parse_utilization(path):
    """Pull the headline rows out of report_utilization."""
    if not os.path.exists(path):
        return {}
    text = open(path, errors='replace').read()
    out = {}
    for key, label in [('luts', r'Slice LUTs\*?'), ('ff', r'Slice Registers'),
                       ('bram', r'Block RAM Tile'), ('dsp', r'DSPs')]:
        m = re.search(r'\|\s*' + label + r'\s*\|\s*(\d+)\s*\|', text)
        if m:
            out[key] = int(m.group(1))
    return out


def parse_timing(path):
    """Longest path delay, in ns."""
    if not os.path.exists(path):
        return None
    text = open(path, errors='replace').read()
    m = re.search(r'Data Path Delay:\s*([\d.]+)ns', text)
    return float(m.group(1)) if m else None


def parse_wns(path):
    """Worst negative slack from report_timing_summary, in ns. Negative means failing."""
    if not os.path.exists(path):
        return None
    text = open(path, errors='replace').read()
    m = re.search(r'WNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)', text, re.S)
    return float(m.group(1)) if m else None


def parse_logic_levels(path):
    if not os.path.exists(path):
        return None
    text = open(path, errors='replace').read()
    levels = [int(x) for x in re.findall(r'^\s*\|\s*(\d+)\s*\|', text, re.MULTILINE)]
    return max(levels) if levels else None


def run_one(vivado_bin, top, sources, part, out_root, period=BOARD_PERIOD_NS, impl=False,
            generics=None, name=None, xdc=None):
    """Synthesize one top. Returns (ok, absolute out_dir).

    Everything is passed as a path RELATIVE to the repo root, and Vivado runs with the repo as
    its working directory. This is not a style choice: Vivado splits `-tclargs` on whitespace,
    and the repo lives under "Coding Projects", so absolute paths arrive in the Tcl script
    chopped in half at the space. Relative paths here contain no spaces.
    """
    out_rel = os.path.relpath(os.path.join(out_root, name or top), REPO).replace('\\', '/')
    out_abs = os.path.join(REPO, out_rel)
    os.makedirs(out_abs, exist_ok=True)

    missing = [s for s in sources if not os.path.exists(os.path.join(REPO, s))]
    if missing:
        raise SystemExit('missing sources:\n  ' + '\n  '.join(missing))

    env = dict(os.environ)
    env['PATH'] = vivado_bin + os.pathsep + env.get('PATH', '')
    cmd = [tool(vivado_bin, 'vivado'), '-mode', 'batch', '-notrace',
           '-log', f'{out_rel}/vivado.log', '-journal', f'{out_rel}/vivado.jou',
           '-source', 'scripts/build.tcl',
           '-tclargs', top, part, out_rel, str(period), '1' if impl else '0',
           # NAME:VALUE+NAME:VALUE -- see build.tcl for why not '=' or ','.
           '+'.join(g.replace('=', ':') for g in generics) if generics else '-',
           xdc or '-'] + list(sources)

    r = run(cmd, cwd=REPO, env=env, capture=True)
    log = (r.stdout or '') + (r.stderr or '')
    ok = 'BUILD_TCL_DONE' in log
    if not ok:
        tail = '\n'.join(log.strip().splitlines()[-40:])
        print(f'  FAILED. last lines of Vivado output:\n{tail}')
    return ok, out_abs


def main():
    ap = argparse.ArgumentParser(description='Out-of-context synthesis + area report.')
    ap.add_argument('--part', default=DEFAULT_PART)
    ap.add_argument('--vivado-bin', default=None)
    ap.add_argument('--outdir', default=os.path.join(REPO, 'build', 'synth'))
    ap.add_argument('--impl', action='store_true',
                    help='place and route too. Slower, but post-synthesis timing uses '
                         'estimated routing and is systematically optimistic.')
    args = ap.parse_args()

    vivado_bin = find_vivado_bin(args.vivado_bin)
    stage = 'synth + implementation' if args.impl else 'synthesis only'
    print(f'part    : {args.part}  ({DEVICE_LUTS} LUTs)')
    print(f'vivado  : {vivado_bin}')
    print(f'outdir  : {os.path.relpath(args.outdir, REPO)}')
    print(f'stage   : {stage}')
    print()

    results = {}
    for top, sources in TARGETS:
        print(f'=== {top} (out-of-context, {stage}) ===')
        ok, out_dir = run_one(vivado_bin, top, sources, args.part, args.outdir,
                              impl=args.impl)
        if not ok:
            return 1
        # Prefer post-route reports when they exist: they are the numbers that can be quoted.
        util_rpt = 'utilization_routed.rpt' if args.impl else 'utilization.rpt'
        tsum_rpt = 'timing_summary_routed.rpt' if args.impl else 'timing_summary.rpt'
        time_rpt = 'timing_routed.rpt' if args.impl else 'timing.rpt'
        util = parse_utilization(os.path.join(out_dir, util_rpt))
        results[top] = {
            **util,
            'delay': parse_timing(os.path.join(out_dir, time_rpt)),
            'wns': parse_wns(os.path.join(out_dir, tsum_rpt)),
            'levels': parse_logic_levels(os.path.join(out_dir, 'logic_levels.rpt')),
        }
        print(f'  LUTs {util.get("luts", "?")}   FF {util.get("ff", "?")}   '
              f'BRAM {util.get("bram", "?")}   DSP {util.get("dsp", "?")}')
        print()

    print('=' * 82)
    print(f'{"module":22s} {"LUTs":>7} {"% dev":>7} {"FF":>6} {"BRAM":>5} {"DSP":>5} '
          f'{"WNS ns":>8} {"Fmax MHz":>9}')
    print('-' * 82)
    for top, _ in TARGETS:
        r = results[top]
        luts = r.get('luts')
        pct = f'{100*luts/DEVICE_LUTS:.2f}' if luts is not None else '?'
        wns = r.get('wns')
        wns_s = f'{wns:+.3f}' if wns is not None else '?'
        # Fmax = 1 / (constrained period - slack). Positive slack means headroom.
        fmax = f'{1000.0/(BOARD_PERIOD_NS - wns):.1f}' if wns is not None else '?'
        print(f'{top:22s} {luts if luts is not None else "?":>7} {pct:>7} '
              f'{r.get("ff", "?"):>6} {r.get("bram", "?"):>5} {r.get("dsp", "?"):>5} '
              f'{wns_s:>8} {fmax:>9}')
    print('=' * 82)
    print(f'constrained at {BOARD_PERIOD_NS:.1f} ns '
          f'({1000/BOARD_PERIOD_NS:.0f} MHz, the Basys 3 board clock). '
          'WNS >= 0 means it meets timing.')

    core = results.get('dwn_core', {}).get('luts')
    enc = results.get('thermometer_encoder', {}).get('luts')
    top_l = results.get('dwn_top', {}).get('luts')
    if None not in (core, enc, top_l):
        print()
        print(f'encoder / core ratio : {enc/core:.2f}x'
              if core else 'encoder / core ratio : n/a')
        print(f'core + encoder       : {core + enc}   dwn_top: {top_l}   '
              f'(difference {top_l - (core + enc):+d}, from cross-boundary optimization)')
        print()
        print('The paper reports this model at 110 LUTs (Table 2) and EXCLUDES the encoder')
        print('from its totals entirely -- brief §6. Report core and encoder separately in')
        print('every table, which is what makes our numbers comparable to both the paper')
        print('(core-only) and Mecik & Kumm (encoder-inclusive).')

    wns = results.get('dwn_top', {}).get('wns')
    if wns is not None:
        print()
        if wns >= 0:
            print(f'dwn_top MEETS timing at 100 MHz with {wns:.3f} ns to spare '
                  f'-> {1000.0/(BOARD_PERIOD_NS - wns):.1f} MHz achievable.')
        else:
            print(f'dwn_top FAILS timing at 100 MHz by {-wns:.3f} ns '
                  f'-> only {1000.0/(BOARD_PERIOD_NS - wns):.1f} MHz. '
                  'More pipeline stages needed.')
        print('Report latency in CYCLES as well as ns (brief §6): the paper\'s clock speeds')
        print('are speed-graded parts and do not transfer to a -1 Artix-7.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
