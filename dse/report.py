"""Turn sweep results into the two things Study 1 owes: a table and a Pareto frontier.

`docs/dse-plan.md` §6 and brief §10 ask for the accuracy/area/latency frontier plus ONE headline
number -- *the largest DWN that fits an XC7A35T, and what it scores*. This computes both from
`build/dse/results.json`.

Three things it deliberately does NOT do:

  - it does not hide failures. A config that failed to fit or failed Gate 1 stays in the table.
    Failure to route locates the congestion wall (brief §12 risk #2), which is a measurement.
  - it does not report one "total LUTs" column. Core and encoder are separate, always (brief §6)
    -- a combined number is what made the encoder's 14x cost invisible in the literature.
  - it does not compare across toolchains. Every row must come from one machine and one Vivado
    version or the frontier is two half-frontiers with an unknown offset between them.

Usage:
    python dse/report.py                 # table + frontier + headline
    python dse/report.py --csv out.csv   # dump for plotting elsewhere
"""

import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from area_model import DEVICE_LUTS  # noqa: E402

RESULTS = os.path.join(REPO, 'build', 'dse', 'results.json')

# Columns worth dumping. Core and encoder stay separate on purpose.
CSV_FIELDS = [
    'name', 'label', 'status', 'nodes', 'n', 'z', 'encoding', 'layers', 'pipe', 'clock_ns',
    'accuracy_pct', 'dwn_core_luts', 'thermometer_encoder_luts', 'dwn_top_luts', 'dwn_top_ff',
    'device_pct', 'dwn_top_fmax_mhz', 'latency', 'predicted_board_luts',
    'predicted_extrapolated', 'impl', 'seconds',
]


def load():
    if not os.path.exists(RESULTS):
        raise SystemExit(f'no results yet: {os.path.relpath(RESULTS, REPO)}\n'
                         'Run dse/run.py first.')
    with open(RESULTS) as f:
        return list(json.load(f).values())


def pareto(rows, x='dwn_top_luts', y='accuracy_pct'):
    """Configs not beaten on BOTH axes: minimize x (area), maximize y (accuracy).

    Ties matter here. Two configs with identical area and accuracy are both on the frontier --
    dropping one would silently hide, say, a cheaper pipeline variant that costs nothing.
    """
    usable = [r for r in rows if r.get(x) is not None and r.get(y) is not None]
    front = []
    for r in usable:
        dominated = any(o is not r and o[x] <= r[x] and o[y] >= r[y] and
                        (o[x] < r[x] or o[y] > r[y]) for o in usable)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r[x])


def fmt(v, spec=''):
    return format(v, spec) if isinstance(v, (int, float)) else '-'


def main() -> int:
    ap = argparse.ArgumentParser(description='Sweep results, frontier, headline number.')
    ap.add_argument('--csv', help='write the full table to a CSV file')
    args = ap.parse_args()
    rows = load()

    ok = [r for r in rows if r['status'] == 'ok']
    failed = [r for r in rows if r['status'] != 'ok']

    print(f'{"config":24s} {"acc%":>6} {"core":>6} {"enc":>7} {"top":>7} {"%dev":>6} '
          f'{"Fmax":>7} {"lat":>4}  status')
    print('-' * 82)
    for r in sorted(rows, key=lambda r: (r['status'] != 'ok', r.get('nodes') or 0)):
        print(f'{r["label"][:24]:24s} '
              f'{fmt(r.get("accuracy_pct"), ".2f"):>6} '
              f'{fmt(r.get("dwn_core_luts")):>6} '
              f'{fmt(r.get("thermometer_encoder_luts")):>7} '
              f'{fmt(r.get("dwn_top_luts")):>7} '
              f'{fmt(r.get("device_pct"), ".2f"):>6} '
              f'{fmt(r.get("dwn_top_fmax_mhz"), ".1f"):>7} '
              f'{fmt(r.get("latency")):>4}  {r["status"]}')
    print('-' * 82)
    print(f'{len(ok)} ok, {len(failed)} failed/incomplete')

    # ---- failures are results ----
    if failed:
        print()
        print('Not synthesized or failed -- kept, because a config that cannot fit or cannot')
        print('route is where the frontier ENDS, and that is a measurement (brief risk #2):')
        for r in failed:
            print(f'  {r["label"]:24s} {r["status"]:14s} {r.get("error", "")}')

    # ---- frontier ----
    front = pareto(ok)
    if not front:
        print()
        print('No frontier yet: it needs at least one config with BOTH accuracy and area.')
        print('Accuracy comes from the checkpoint, so this fills in as 2c training lands.')
        return 0

    print()
    print('Pareto frontier (minimize dwn_top LUTs, maximize accuracy):')
    print(f'{"config":24s} {"acc%":>6} {"top LUTs":>9} {"%dev":>6} {"Fmax":>7}')
    print('-' * 58)
    for r in front:
        print(f'{r["label"][:24]:24s} {fmt(r.get("accuracy_pct"), ".2f"):>6} '
              f'{r["dwn_top_luts"]:>9} {fmt(r.get("device_pct"), ".2f"):>6} '
              f'{fmt(r.get("dwn_top_fmax_mhz"), ".1f"):>7}')
    print('-' * 58)

    # ---- the headline number brief §10 asks for ----
    fits = [r for r in ok if r.get('dwn_top_luts') and r['dwn_top_luts'] <= DEVICE_LUTS]
    if fits:
        big = max(fits, key=lambda r: r['nodes'])
        print()
        print('HEADLINE (brief sec.10): largest DWN measured to fit an XC7A35T')
        print(f'  config   : {big["label"]}  ({big["nodes"]} nodes, n={big["n"]}, z={big["z"]})')
        print(f'  accuracy : {fmt(big.get("accuracy_pct"), ".2f")}%')
        print(f'  area     : {big["dwn_top_luts"]} LUTs '
              f'({fmt(big.get("device_pct"), ".2f")}% of device) = '
              f'core {big.get("dwn_core_luts")} + encoder '
              f'{big.get("thermometer_encoder_luts")}')
        print(f'  Fmax     : {fmt(big.get("dwn_top_fmax_mhz"), ".1f")} MHz, '
              f'latency {big.get("latency")} cycles, II=1')
        if not any(r['status'] == 'synth-failed' for r in rows):
            print('  CAVEAT: no config in this sweep has actually FAILED to fit, so this is the')
            print('          largest TRIED, not the largest that fits. Extend the ladder upward')
            print('          until one fails -- otherwise the frontier has no measured edge.')
        if not big.get('impl'):
            print('  CAVEAT: post-synthesis, not post-route. Phase 1 measured 161.0 -> 147.1 MHz')
            print('          across that boundary. Re-run with --impl before quoting Fmax.')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\nwrote {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
