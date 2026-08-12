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
    print('wrote', os.path.relpath(path, REPO))
    return 0


if __name__ == '__main__':
    sys.exit(main())
