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
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from area_model import DEVICE_LUTS  # noqa: E402

sys.path.insert(0, REPO)
import datasets  # noqa: E402

# Default dataset and its paths; main() rebinds from --dataset. JSC's are historical and must not
# move -- see datasets/__init__.py and dse/run.py:use_dataset().
DS = datasets.JSC
RESULTS = os.path.join(REPO, 'build', 'dse', 'results.json')

# Columns worth dumping. Core and encoder stay separate on purpose.
CSV_FIELDS = [
    'name', 'label', 'status', 'nodes', 'n', 'z', 'encoding', 'layers', 'pipe', 'clock_ns',
    'accuracy_pct', 'dwn_core_luts', 'thermometer_encoder_luts', 'dwn_top_luts', 'dwn_top_ff',
    # BRAM and DSP are ZERO for every DWN config measured. That zero is the point -- it is the
    # claim against hls4ml (whose MLPs spend DSPs on MACs) and conifer, so it belongs in the
    # published table rather than in a footnote.
    'dwn_top_bram', 'dwn_top_dsp',
    'device_pct', 'dwn_top_fmax_mhz', 'latency', 'predicted_board_luts',
    'predicted_extrapolated', 'impl', 'seconds',
]


def load(path=None):
    """Rows from a results file.

    `path` allows inspecting an alternate run -- and lets the report and the plots be exercised
    on synthetic data, which is how their layout gets checked BEFORE a real sweep rather than
    after one. These are the deliverable; a crash or an unreadable figure discovered at the end
    of seven hours of Vivado is discovered too late.
    """
    path = path or RESULTS
    if not os.path.exists(path):
        raise SystemExit(f'no results yet: {os.path.relpath(path, REPO)}\n'
                         'Run dse/run.py first.')
    with open(path) as f:
        return list(json.load(f).values())


AREA, ACC, LAT = 'dwn_top_luts', 'accuracy_pct', 'latency_ns'


def derive(rows):
    """Add latency in NANOSECONDS -- the quantity a trigger application actually cares about.

    Cycles alone cannot rank pipeline variants, because dropping a stage removes a cycle but
    also lowers Fmax. Phase 1 measured both ends of that trade: 4 stages at 161.0 MHz is
    24.8 ns, 3 stages at 122.9 MHz is 24.4 ns -- nearly identical real latency despite a whole
    cycle of difference. Ranking on cycles would have called that a clear win; ranking on
    nanoseconds shows it is a wash. Cycles are still reported (brief §6 requires it), but the
    frontier is computed on time.
    """
    for r in rows:
        cycles, fmax = r.get('latency'), r.get('dwn_top_fmax_mhz')
        r[LAT] = round(1000.0 * cycles / fmax, 2) if cycles and fmax else None
        # Does it close the clock it was constrained at? A config missing timing is not a
        # frontier point at that clock, whatever its area says.
        wns = r.get('dwn_top_wns')
        r['meets_timing'] = None if wns is None else wns >= 0
    return rows


def pareto(rows, objectives=((AREA, 'min'), (ACC, 'max'))):
    """Configs not dominated on ALL objectives simultaneously.

    Generic over the objective list because Study 1 wants two views of the same data: the
    classic accuracy-vs-area frontier, and an accuracy/area/latency one. Group B configs share
    an accuracy and an area and differ only in timing, so under the 2-objective view they all
    tie and land on the frontier as indistinguishable points -- which is exactly the thing
    dse-plan's Group B exists to discriminate.

    Ties are kept deliberately: two configs equal on every objective are both on the frontier,
    and dropping one would silently hide, say, a pipeline variant that costs nothing.
    """
    keys = [k for k, _ in objectives]
    usable = [r for r in rows if all(r.get(k) is not None for k in keys)]

    def dominates(o, r):
        better = False
        for k, d in objectives:
            if d == 'min':
                if o[k] > r[k]:
                    return False
                if o[k] < r[k]:
                    better = True
            else:
                if o[k] < r[k]:
                    return False
                if o[k] > r[k]:
                    better = True
        return better

    front = [r for r in usable if not any(dominates(o, r) for o in usable if o is not r)]
    return sorted(front, key=lambda r: r[keys[0]])


def fmt(v, spec=''):
    return format(v, spec) if isinstance(v, (int, float)) else '-'


def main() -> int:
    ap = argparse.ArgumentParser(description='Sweep results, frontier, headline number.')
    ap.add_argument('--csv', help='write the full table to a CSV file')
    ap.add_argument('--dataset', default=datasets.JSC.name, choices=datasets.names(),
                    help='which dataset to report on (default %(default)s)')
    ap.add_argument('--results', help='read an alternate results.json')
    ap.add_argument('--snapshot', action='store_true',
                    help='copy the results into docs/ as committed evidence')
    args = ap.parse_args()

    global DS, RESULTS
    DS = datasets.get(args.dataset)
    RESULTS = os.path.join(REPO, 'build', 'dse', DS.build_subdir, 'results.json')
    rows = derive(load(args.results))

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

    def show(title, rows_):
        print()
        print(title)
        print(f'{"config":24s} {"acc%":>6} {"top LUTs":>9} {"%dev":>6} {"Fmax":>7} '
              f'{"cyc":>4} {"lat ns":>7}')
        print('-' * 70)
        for r in rows_:
            print(f'{r["label"][:24]:24s} {fmt(r.get("accuracy_pct"), ".2f"):>6} '
                  f'{r["dwn_top_luts"]:>9} {fmt(r.get("device_pct"), ".2f"):>6} '
                  f'{fmt(r.get("dwn_top_fmax_mhz"), ".1f"):>7} '
                  f'{fmt(r.get("latency")):>4} {fmt(r.get(LAT), ".2f"):>7}')
        print('-' * 70)

    show('Pareto frontier -- accuracy vs area:', front)

    # The three-objective view. Group B points share an accuracy and an area, so they only
    # separate once latency is an objective -- that is the whole reason Group B is swept.
    front3 = pareto(ok, objectives=((AREA, 'min'), (ACC, 'max'), (LAT, 'min')))
    if len(front3) != len(front):
        show('Pareto frontier -- accuracy vs area vs latency (ns):', front3)
        print(f'{len(front3) - len(front)} extra point(s) appear once latency is an objective '
              f'-- these are\nconfigs that trade timing at equal accuracy and area, i.e. Group B.')

    missed = [r for r in ok if r.get('meets_timing') is False]
    if missed:
        print()
        print('Constrained clock NOT met -- not frontier points at that clock:')
        for r in missed:
            print(f'  {r["label"]:24s} WNS {fmt(r.get("dwn_top_wns"), "+.3f")} ns '
                  f'at {fmt(r.get("clock_ns"), ".1f")} ns')

    # ---- the headline number brief §10 asks for ----
    # "Fits" means BOTH: inside the LUT budget AND meeting the clock it was constrained at.
    # Area alone is not enough -- `1x3000 z=50` uses 67% of the device and would win on nodes,
    # but runs at 96.2 MHz against the board's 100 MHz. A design that cannot be clocked is not
    # a design that fits, and reporting it as the headline would be the same class of error as
    # trusting Vivado's exit status over the measured routed area.
    fits = [r for r in ok
            if r.get('dwn_top_luts') and r['dwn_top_luts'] <= DEVICE_LUTS
            and r.get('meets_timing') is not False]
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
        # A config can complete Vivado's flow and still not fit: placement starts from the
        # post-synthesis netlist, and physical optimization can push the routed design past the
        # device while chasing timing. `1x2000` did exactly that -- 96.8% post-synthesis,
        # 102.8% post-route. So "did it fit" must be judged on the MEASURED routed area and on
        # timing, never on whether the tool returned success.
        over = [r for r in rows if (r.get('dwn_top_luts') or 0) > DEVICE_LUTS]
        missed = [r for r in rows if r.get('meets_timing') is False
                  and (r.get('clock_ns') == 10.0)]
        if not over and not any(r['status'] == 'synth-failed' for r in rows):
            print('  CAVEAT: no config in this sweep has actually FAILED to fit, so this is the')
            print('          largest TRIED, not the largest that fits. Extend the ladder upward')
            print('          until one fails -- otherwise the frontier has no measured edge.')
        else:
            print()
            print('  THE EDGE IS MEASURED -- the ladder was extended until configs broke:')
            for r in over:
                print(f'    {r["label"]:22s} {r["dwn_top_luts"]:>6} LUTs '
                      f'({r["device_pct"]:.1f}% of device) -- EXCEEDS the part')
            for r in missed:
                if r in over:
                    continue
                print(f'    {r["label"]:22s} WNS {r["dwn_top_wns"]:+.3f} ns '
                      f'({r["dwn_top_fmax_mhz"]:.1f} MHz) -- MISSES the 100 MHz board clock')
        if not big.get('impl'):
            print('  CAVEAT: post-synthesis, not post-route. Phase 1 measured 161.0 -> 147.1 MHz')
            print('          across that boundary. Re-run with --impl before quoting Fmax.')

    def write_csv(path):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)

    if args.csv:
        write_csv(args.csv)
        print(f'\nwrote {args.csv}')

    if args.snapshot:
        # `build/dse/results.json` is gitignored with the rest of build/, but it is NOT a
        # regenerable build product in any useful sense: reproducing it costs a Kaggle GPU
        # session plus hours of Vivado. It is the RECORD OF WHAT WAS MEASURED, and it is the
        # evidence that a config was actually built and tested rather than merely planned.
        # A few hundred KB of JSON is cheap; the 933 MB of weights it describes is not, and is
        # not needed to make the claim.
        out = os.path.join(REPO, 'docs', DS.results_dir)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'sweep-results.json'), 'w') as f:
            json.dump({r['name']: r for r in rows}, f, indent=2, sort_keys=True)
        write_csv(os.path.join(out, 'sweep-results.csv'))
        # The grid AS TRAINED, not as grid.py would regenerate it today. It has already changed
        # once -- the ladder went from 8 rungs to 10 -- so this records what the notebook
        # actually embedded, which is the thing the checkpoints correspond to.
        src = os.path.join(REPO, 'build', 'dse', 'train_grid.json')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, 'train_grid.json'))
        # The path, not a literal: this printed 'docs/results/' while writing to
        # docs/results-mnist/, which reads exactly like the JSC snapshot has just
        # been overwritten -- alarming, and wrong.
        print(f'\nsnapshot -> docs/{DS.results_dir}/  ({len(rows)} configs) -- commit these')
        print('  sweep-results.json/.csv + train_grid.json'
              '  (figures: dse/plot.py --snapshot)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
