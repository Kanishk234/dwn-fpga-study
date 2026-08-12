"""Acceptance test: does Phase 1 reproduce on THIS machine?

Run this on a new machine before doing any Phase 2 work. It re-runs the Phase 1 flow and checks
the results against the values recorded in docs/phase1-report.md, so "it seems to work" becomes
a pass/fail.

The reason this matters is not sentiment. **A Pareto frontier assembled from two Vivado versions
is two half-frontiers with an unknown offset between them.** Vivado's optimizer changes between
releases, so identical RTL can land on different LUT counts. If this machine does not reproduce
Phase 1's areas exactly, its sweep points are not comparable to Phase 1's numbers and the two
cannot appear in the same table.

    .venv\\Scripts\\python.exe scripts/verify_phase1.py               # sim + synthesis
    .venv\\Scripts\\python.exe scripts/verify_phase1.py --with-board  # + bitstream, Gate 1b

What must match EXACTLY (these are deterministic):
    Gate 1 vector counts, all LUT/FF/BRAM/DSP counts, Gate 1b agreement, core cycles, accuracy.

What is ALLOWED to vary:
    timing slack (placement is stochastic -- tens of picoseconds), wall-clock times, link rate.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gate1 import DEFAULT_RTL_DIR, REPO, python_exe, run  # noqa: E402

EXPECTED = {
    'gate1_core_vectors': 1504,
    'gate1_top_vectors': 1518,
    'nodes': 50,
    'comparators': 202,
    # ⚠️ core_luts and top_luts CHANGED on 2026-08-11, from 108 and 1619, when rtl/argmax.v
    # became a balanced tree unconditionally (it was a sequential scan). The tree costs +2 LUTs
    # at five classes and is what lets ten classes meet the board clock at all.
    #
    # The pre-change values are not lost: they are reproducible at the `jsc-complete` tag, which
    # is what REPORT.md's JSC figures are measured at and now say so. Pinning a published result
    # to a tag is the right mechanism; freezing the RTL to protect a printed number is not.
    #
    # docs/results/sweep-results.json still holds chain-era areas. Anything synthesized after
    # this change reads ~2 LUTs higher -- 0.12% at 1x50, 0.02% at the headline config, so no
    # frontier point or conclusion moves. Do not mix the two in one table without saying so.
    'core_luts': 110, 'core_ff': 73,
    'enc_luts': 1519, 'enc_ff': 0,
    'top_luts': 1621, 'top_ff': 269,
    # The board design moved TWICE, in opposite directions, and only the second is large:
    #
    #   2058  jsc-complete tag, chain argmax + indexed-write loader
    #   2060  argmax tree (+2). Confirmed on hardware -- Gate 1b 166,000/166,000, cycles and
    #         both accuracies unchanged, so the +2 was area only.
    #   1893  uart_loader byte shift register (-167). The variable part-select write it
    #         replaced cost 439 LUTs in the loader; a shift register costs 77 and lets Vivado
    #         map part of it into SRLs. WNS also improved +1.753 -> +2.014 ns.
    #
    # ALL THREE ARE POST-ROUTE, from utilization_routed.rpt, which is what build_bitstream.py
    # parses and therefore the only source this dict may be fed from. utilization.rpt in the
    # same directory is POST-SYNTH and reads 1896/861 for the same design -- close enough to
    # look like the right number and wrong enough to fail this check. Do not mix them.
    #
    # Only dwn_core/dwn_top feed the DSE sweep, and neither moved here -- the loader is harness,
    # outside dwn_top. So no Phase 2 frontier point is affected by this line.
    'board_luts': 1893, 'board_ff': 864, 'board_bram': 8, 'board_dsp': 0,
    'gate1b_agree': 166000, 'gate1b_total': 166000,
    'core_cycles': 166815,
    'acc_float': '73.8361', 'acc_fixed': '73.8349',
}

results = []      # (label, got, want, ok)


def check(label, got, want):
    ok = str(got) == str(want)
    results.append((label, got, want, ok))
    print(f'  {"OK  " if ok else "FAIL"}  {label:34s} {str(got):>10}  (expected {want})')
    return ok


def capture(cmd, label):
    print(f'\n=== {label} ===')
    sys.stdout.flush()
    r = run(cmd, cwd=REPO, capture=True)
    out = (r.stdout or '') + (r.stderr or '')
    if r.returncode != 0:
        print(out.strip()[-1500:])
        raise SystemExit(f'{label} failed (exit {r.returncode})')
    return out


def main():
    ap = argparse.ArgumentParser(description='Verify Phase 1 reproduces on this machine.')
    ap.add_argument('--with-board', action='store_true',
                    help='also build the bitstream, program, and run Gate 1b')
    args = ap.parse_args()
    py = python_exe()

    # ---- 1. simulation ----
    out = capture([py, 'scripts/run_gate1.py'], 'Gate 1 (simulation)')
    vec = [int(m) for m in re.findall(r'vectors tested\s*:\s*(\d+)', out)]
    print()
    # Imported from run_gate1 rather than spelled out, so this cannot drift from where the
    # emitters actually write -- which is exactly how this check broke during the 2a restructure.
    check('emitted LUT nodes', len(re.findall(r'lut_node #', open(
        os.path.join(DEFAULT_RTL_DIR, 'dwn_core.v')).read())), EXPECTED['nodes'])
    m = re.search(r'read-back check: (\d+)/\d+ comparators', out)
    check('encoder comparators', m.group(1) if m else '?', EXPECTED['comparators'])
    check('Gate 1 core vectors', vec[0] if len(vec) > 0 else '?',
          EXPECTED['gate1_core_vectors'])
    check('Gate 1 top vectors', vec[1] if len(vec) > 1 else '?',
          EXPECTED['gate1_top_vectors'])
    check('Gate 1 verdict', 'PASS' if 'GATE 1 PASSED' in out else 'FAIL', 'PASS')

    # ---- 2. harness unit tests ----
    out = capture([py, 'scripts/run_tb.py'], 'harness unit tests')
    check('unit tests', 'PASS' if 'ALL PASSED' in out else 'FAIL', 'PASS')

    # ---- 3. synthesis ----
    out = capture([py, 'scripts/run_synth.py', '--impl'], 'synthesis (out-of-context, routed)')
    print()
    for mod, key in [('dwn_core', 'core'), ('thermometer_encoder', 'enc'), ('dwn_top', 'top')]:
        m = re.search(rf'^{mod}\s+(\d+)\s+[\d.]+\s+(\d+)', out, re.MULTILINE)
        check(f'{mod} LUTs', m.group(1) if m else '?', EXPECTED[f'{key}_luts'])
        check(f'{mod} FF', m.group(2) if m else '?', EXPECTED[f'{key}_ff'])

    # ---- 4. board ----
    if args.with_board:
        out = capture([py, 'scripts/build_bitstream.py'], 'bitstream')
        print()
        for field, key in [('LUTs', 'board_luts'), ('FF', 'board_ff'),
                           ('BRAM', 'board_bram'), ('DSP', 'board_dsp')]:
            m = re.search(rf'^\s+{field}\s+(\d+)', out, re.MULTILINE)
            check(f'board {field}', m.group(1) if m else '?', EXPECTED[key])
        m = re.search(r'WNS\s+([+-][\d.]+)', out)
        wns = float(m.group(1)) if m else None
        # Slack is allowed to vary -- it just has to be positive.
        check('board meets 100 MHz', 'yes' if (wns is not None and wns >= 0) else 'no', 'yes')
        print(f'        (slack {wns:+.3f} ns -- allowed to vary, placement is stochastic)')

        capture([py, 'scripts/program.py'], 'program the board')

        full = os.path.join(REPO, 'training', 'artifacts',
                            'dwn_jsc_t200_distributive_50_l_b100_testset_full.npz')
        if not os.path.exists(full):
            print('\n  SKIPPED Gate 1b: the full 166k test set is not present.')
            print('  It is gitignored. Regenerate with training/dump_testset_kaggle.ipynb')
            print('  (inference only -- do NOT re-run the training notebook), or copy the')
            print('  .npz across from the machine that has it.')
        else:
            out = capture([py, 'scripts/host.py', '--gate1b'], 'Gate 1b (hardware)')
            print()
            m = re.search(r'hardware == software\s*:\s*(\d+)/(\d+)', out)
            check('Gate 1b agreement', m.group(1) if m else '?', EXPECTED['gate1b_agree'])
            check('Gate 1b samples', m.group(2) if m else '?', EXPECTED['gate1b_total'])
            m = re.search(r'core cycles\s*:\s*(\d+)', out)
            check('core cycles (II=1)', m.group(1) if m else '?', EXPECTED['core_cycles'])
            m = re.search(r'float32 model\s*:\s*([\d.]+)%', out)
            check('accuracy float32', m.group(1) if m else '?', EXPECTED['acc_float'])
            m = re.search(r'on hardware\)\s*:\s*([\d.]+)%', out)
            check('accuracy Q3.12', m.group(1) if m else '?', EXPECTED['acc_fixed'])

    # ---- verdict ----
    failed = [r for r in results if not r[3]]
    print('\n' + '=' * 66)
    print(f'{len(results) - len(failed)}/{len(results)} checks passed')
    if not failed:
        print('PHASE 1 REPRODUCES ON THIS MACHINE -- safe to start Phase 2 here.')
        if not args.with_board:
            print('(simulation + synthesis only; re-run with --with-board for Gate 1b)')
        print('=' * 66)
        return 0

    print('MISMATCHES:')
    for label, got, want, _ in failed:
        print(f'  {label:34s} got {got}, expected {want}')
    print()
    print('If the LUT or FF counts differ, this machine has a different Vivado version or')
    print('settings. That is not a bug -- but its sweep numbers will NOT be comparable to')
    print("Phase 1's, so either match the toolchain or re-measure Phase 1 here and say so in")
    print('the writeup. Do not mix them in one table.')
    print('=' * 66)
    return 1


if __name__ == '__main__':
    sys.exit(main())
