"""Run a standalone RTL unit testbench under xsim.

Gate 1 (scripts/run_gate1.py) is the golden-model check on the DWN datapath and regenerates
everything it tests. This is the simpler sibling for harness modules -- UART, BRAM store,
FSMs -- which have no checkpoint behind them and just need compiling and running.

Every testbench is expected to print `RESULT : PASS` or `RESULT : FAIL`; $finish always exits
0, so the verdict is read from stdout rather than the exit code.

Usage:
    .venv\\Scripts\\python.exe scripts/run_tb.py uart
    .venv\\Scripts\\python.exe scripts/run_tb.py --list
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gate1 import REPO, find_vivado_bin, python_exe, run, run_xsim, xvlog  # noqa: E402

# Default checkpoint, for suites that need real golden vectors.
DEFAULT_CHECKPOINT = os.path.join(
    'training', 'artifacts', 'dwn_jsc_t200_distributive_50_l_b100_checkpoint.pt')

# name -> {top, sources, [include], [needs_vectors]}
# needs_vectors: regenerate the golden test vectors into the suite's work dir first, so the
# testbench can never run against a stale set.
SUITES = {
    'uart': {'top': 'uart_tb',
             'sources': ['harness/uart_tx.v', 'harness/uart_rx.v', 'tb/uart_tb.v']},
    'benchmark': {'top': 'benchmark_tb',
                  'sources': ['harness/vector_store.v', 'harness/benchmark_fsm.v',
                              'tb/benchmark_tb.v']},
    'loader': {'top': 'loader_tb',
               'sources': ['harness/uart_tx.v', 'harness/uart_rx.v',
                           'harness/uart_loader.v', 'harness/vector_store.v',
                           'tb/loader_tb.v']},
    'top': {'top': 'top_tb',
            'sources': ['rtl/lut_node.v', 'rtl/popcount.v', 'rtl/argmax.v', 'rtl/pipe_reg.v',
                        'build/rtl/dwn_core.v', 'build/rtl/thermometer_encoder.v',
                        'build/rtl/dwn_top.v',
                        'harness/uart_tx.v', 'harness/uart_rx.v', 'harness/uart_loader.v',
                        'harness/vector_store.v', 'harness/benchmark_fsm.v',
                        'harness/seg7.v', 'harness/dwn_basys3_top.v',
                        'tb/top_tb.v'],
            'include': ['build/rtl'],
            'needs_vectors': True},
}


def main():
    ap = argparse.ArgumentParser(description='Run an RTL unit testbench.')
    ap.add_argument('suite', nargs='?', help='which suite to run (default: all)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--vivado-bin', default=None)
    args = ap.parse_args()

    if args.list:
        for name, spec in SUITES.items():
            print(f'{name:12s} -> {spec["top"]}')
        return 0

    names = [args.suite] if args.suite else list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise SystemExit(f'unknown suite(s): {", ".join(unknown)}. '
                         f'Known: {", ".join(SUITES)}')

    vivado_bin = find_vivado_bin(args.vivado_bin)
    failures = []

    for name in names:
        spec = SUITES[name]
        top, sources = spec['top'], spec['sources']
        work = os.path.join(REPO, 'build', 'tb', name)
        srcs = [os.path.join(REPO, s) for s in sources]
        missing = [s for s in srcs if not os.path.exists(s)]
        if missing:
            raise SystemExit('missing sources:\n  ' + '\n  '.join(missing) +
                             '\n(generated RTL? run rtlgen/emit_core.py and '
                             'rtlgen/emit_encoder.py, or scripts/run_gate1.py)')

        print(f'=== {name} ({top}) ===')

        if spec.get('needs_vectors'):
            os.makedirs(work, exist_ok=True)
            ckpt = os.path.join(REPO, DEFAULT_CHECKPOINT)
            r = run([python_exe(), os.path.join(REPO, 'tb', 'gen_vectors.py'), ckpt,
                     '--outdir', work])
            if r.returncode != 0:
                raise SystemExit('gen_vectors.py failed')

        includes = [os.path.join(REPO, p) for p in spec.get('include', [])]
        ok, output = xvlog(vivado_bin, work, srcs, include=includes)
        if not ok:
            print(output.strip())
            failures.append(name)
            continue

        ok, output = run_xsim(vivado_bin, work, top)
        for line in output.splitlines():
            if re.search(r'====|RESULT|MISMATCH|ERROR|bytes|framing|mismatches|test|batch'
                         r'|vectors|correct|cycles|path|BOARD|loaded', line):
                print(line)

        verdict = re.search(r'RESULT\s+:\s+(PASS|FAIL)', output)
        if not ok or not verdict or verdict.group(1) != 'PASS':
            failures.append(name)
        print()

    if failures:
        print(f'FAILED: {", ".join(failures)}')
        return 1
    print(f'ALL PASSED ({", ".join(names)})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
