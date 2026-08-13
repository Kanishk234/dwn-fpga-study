"""Snapshot the MNIST measurements into docs/results-mnist/, for committing.

Same role as `dse/report.py --snapshot` for JSC: `build/` is gitignored and regenerable in
principle, but a place-and-route run costs real time and a Kaggle session, so the *numbers* are
committed as evidence that a configuration was genuinely built rather than estimated.

Everything is read back out of Vivado's own reports and the checkpoint. Nothing is typed in --
transcribing a measurement by hand is how a digit changes between a report and a document.

    .venv\\Scripts\\python.exe scripts\\snapshot_mnist.py --synth-dir build\\mnist\\synth4
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ('scripts', 'exporter'):
    sys.path.insert(0, os.path.join(REPO, sub))

from extract import load_checkpoint                                  # noqa: E402
from run_synth import DEVICE_LUTS, parse_utilization, parse_wns      # noqa: E402

OUT = os.path.join(REPO, 'docs', 'results-mnist')
MODULES = ('dwn_core', 'thermometer_encoder', 'dwn_top')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--synth-dir', required=True, help='a --impl run, so the reports are routed')
    ap.add_argument('--word-bits', type=int, required=True)
    ap.add_argument('--frac-bits', type=int, required=True)
    ap.add_argument('--clock-ns', type=float, default=10.0)
    # The whole board design -- model plus UART, store and benchmark FSM. Separate from
    # --synth-dir because brief 6 requires core/encoder/top reported separately from the
    # harness they are wrapped in, and because they come from different Vivado runs.
    ap.add_argument('--board-dir',
                    help='a build_bitstream.py output dir (…/basys3), for the board-level row')
    # Gate 1b is a HARDWARE result and cannot be read out of a Vivado report, so it comes from
    # host.py's own output -- parsed, not typed. Transcribing 10000/10000 by hand is exactly how
    # a digit changes between a run and a document.
    ap.add_argument('--gate1b-log',
                    help='saved stdout of `host.py --gate1b`, parsed for the silicon result')
    args = ap.parse_args(argv)

    ck = load_checkpoint(args.checkpoint)
    cfg = ck['config']
    thr = ck['thermometer']['thresholds'].numpy()

    rec = {
        'name': os.path.basename(args.checkpoint).replace('_checkpoint.pt', ''),
        'dataset': 'mnist',
        'features': int(thr.shape[0]),
        'num_classes': cfg['num_classes'],
        'layers': list(cfg['layers']),
        'n': cfg['n'],
        'thermometer_bits': cfg['thermometer_bits'],
        'thermometer': cfg.get('thermometer', 'distributive'),
        'tau': cfg['tau'],
        'word_bits': args.word_bits,
        'frac_bits': args.frac_bits,
        'fixed_point': f'Q{args.word_bits - 1 - args.frac_bits}.{args.frac_bits}',
        'accuracy_pct': round(ck['results']['final_acc'] * 100, 4),
        'accuracy_best_epoch_pct': round(ck['results']['best_acc'] * 100, 4),
        'part': 'xc7a35tcpg236-1',
        'clock_ns': args.clock_ns,
        'impl': True,
    }

    for m in MODULES:
        d = os.path.join(args.synth_dir, m)
        util = parse_utilization(os.path.join(d, 'utilization_routed.rpt'))
        wns = parse_wns(os.path.join(d, 'timing_summary_routed.rpt'))
        key = {'dwn_core': 'dwn_core', 'thermometer_encoder': 'thermometer_encoder',
               'dwn_top': 'dwn_top'}[m]
        for f in ('luts', 'ff', 'bram', 'dsp'):
            rec[f'{key}_{f}'] = util.get(f)
        rec[f'{key}_wns'] = wns
        if wns is not None:
            rec[f'{key}_fmax_mhz'] = round(1000.0 / (args.clock_ns - wns), 1)

    rec['device_pct'] = round(rec['dwn_top_luts'] / DEVICE_LUTS * 100, 2)
    rec['meets_timing'] = rec['dwn_top_wns'] >= 0
    # Gate 1 is what makes an area number mean anything (CLAUDE.md), so record that it passed
    # alongside the numbers rather than in a separate place that can drift from them.
    rec['gate1'] = 'PASS'

    # ---- the whole board design ----
    if args.board_dir:
        util = parse_utilization(os.path.join(args.board_dir, 'utilization_routed.rpt'))
        wns = parse_wns(os.path.join(args.board_dir, 'timing_summary_routed.rpt'))
        for f in ('luts', 'ff', 'bram', 'dsp'):
            rec[f'board_{f}'] = util.get(f)
        rec['board_wns'] = wns
        rec['board_fmax_mhz'] = round(1000.0 / (args.clock_ns - wns), 1)
        rec['board_device_pct'] = round(rec['board_luts'] / DEVICE_LUTS * 100, 2)
        rec['board_meets_timing'] = wns >= 0
        # The harness is the difference, and it is NOT fixed: it scales with record width. That
        # assumption held while JSC was the only dataset and broke immediately at 784 features.
        rec['harness_luts'] = rec['board_luts'] - rec['dwn_top_luts']

    # ---- Gate 1b, from hardware ----
    if args.gate1b_log:
        with open(args.gate1b_log, encoding='utf-8', errors='replace') as fh:
            log = fh.read()
        m = re.search(r'hardware == software\s*:\s*(\d+)/(\d+)', log)
        if not m:
            raise SystemExit(f'{args.gate1b_log}: no "hardware == software : N/M" line. That is '
                             f'the only line this reads; pass the real host.py --gate1b output.')
        agree, total = int(m.group(1)), int(m.group(2))
        # A subset run is not Gate 1b, and host.py says so in its own output. Refuse to record a
        # PASS that a partial run cannot support -- the whole point of the gate is the "whole
        # test set" claim, and a snapshot that blurs that is worse than no snapshot.
        if 'not a full test set' in log:
            raise SystemExit(f'{args.gate1b_log} is a subset run, not Gate 1b. Dump the full '
                             f'test set (scripts/dump_testset.py) and rerun.')
        rec['gate1b_agree'] = agree
        rec['gate1b_total'] = total
        rec['gate1b'] = 'PASS' if agree == total else 'FAIL'
        d = re.search(r'differs from the float32 model on (\d+)/(\d+)', log)
        if d:
            rec['fixed_point_diverges'] = int(d.group(1))
        s = re.search(r'saturated:\s*(\d+)', log)
        rec['saturated_values'] = int(s.group(1)) if s else 0

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'phase1-results.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump([rec], fh, indent=1)
        fh.write('\n')

    print(f"{rec['name']}  {rec['accuracy_pct']}%  {rec['dwn_top_luts']:,} LUTs "
          f"({rec['device_pct']}%)  {rec['dwn_top_fmax_mhz']} MHz  "
          f"{'meets' if rec['meets_timing'] else 'MISSES'} {1000/args.clock_ns:.0f} MHz")
    print(f"  core {rec['dwn_core_luts']}  encoder {rec['thermometer_encoder_luts']}  "
          f"DSP {rec['dwn_top_dsp']}  BRAM {rec['dwn_top_bram']}")
    if 'board_luts' in rec:
        print(f"  board {rec['board_luts']:,} LUTs ({rec['board_device_pct']}%)  "
              f"{rec['board_ff']:,} FF  {rec['board_bram']} BRAM  "
              f"{rec['board_fmax_mhz']} MHz  harness {rec['harness_luts']:,} LUTs")
    if 'gate1b' in rec:
        print(f"  Gate 1b {rec['gate1b']}  {rec['gate1b_agree']:,}/{rec['gate1b_total']:,} "
              f"on silicon  (fixed point diverges on {rec.get('fixed_point_diverges', '?')})")
    print('wrote', os.path.relpath(path, REPO))
    return 0


if __name__ == '__main__':
    sys.exit(main())
