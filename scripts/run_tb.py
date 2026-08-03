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
from run_gate1 import REPO, find_vivado_bin, run_xsim, xvlog  # noqa: E402

# name -> (testbench top, sources)
SUITES = {
    'uart': ('uart_tb', ['harness/uart_tx.v', 'harness/uart_rx.v', 'tb/uart_tb.v']),
    'benchmark': ('benchmark_tb', ['harness/vector_store.v', 'harness/benchmark_fsm.v',
                                   'tb/benchmark_tb.v']),
    'loader': ('loader_tb', ['harness/uart_tx.v', 'harness/uart_rx.v',
                             'harness/uart_loader.v', 'harness/vector_store.v',
                             'tb/loader_tb.v']),
}


def main():
    ap = argparse.ArgumentParser(description='Run an RTL unit testbench.')
    ap.add_argument('suite', nargs='?', help='which suite to run (default: all)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--vivado-bin', default=None)
    args = ap.parse_args()

    if args.list:
        for name, (top, _) in SUITES.items():
            print(f'{name:12s} -> {top}')
        return 0

    names = [args.suite] if args.suite else list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise SystemExit(f'unknown suite(s): {", ".join(unknown)}. '
                         f'Known: {", ".join(SUITES)}')

    vivado_bin = find_vivado_bin(args.vivado_bin)
    failures = []

    for name in names:
        top, sources = SUITES[name]
        work = os.path.join(REPO, 'build', 'tb', name)
        srcs = [os.path.join(REPO, s) for s in sources]
        missing = [s for s in srcs if not os.path.exists(s)]
        if missing:
            raise SystemExit('missing sources:\n  ' + '\n  '.join(missing))

        print(f'=== {name} ({top}) ===')
        ok, output = xvlog(vivado_bin, work, srcs)
        if not ok:
            print(output.strip())
            failures.append(name)
            continue

        ok, output = run_xsim(vivado_bin, work, top)
        for line in output.splitlines():
            if re.search(r'====|RESULT|MISMATCH|ERROR|bytes|framing|mismatches|test|batch',
                         line):
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
