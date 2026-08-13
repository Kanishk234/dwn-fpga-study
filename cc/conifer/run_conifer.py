"""conifer (GBDT) through OUR synthesis flow -- Phase 3 §2.1.

Trains an xgboost GBDT on JSC, converts it with conifer, synthesizes the generated Verilog
through `scripts/build.tcl` at `xc7a35tcpg236-1` / 10 ns, and records the same columns the DWN
sweep records. One config per invocation, or `--sweep` for the depth x n_estimators curve the
plan asks for (iso-accuracy and iso-area both need a curve, not one point).

WHY THIS DOES NOT CALL `conifer.model.build()`. Two reasons, and the second is the important one:

  1. It cannot work here. conifer detects the HLS tool with `os.system('type X > /dev/null')`
     -- a POSIX builtin -- and then invokes a command named `vitis_hls`, which Vivado/Vitis
     2025.2 does not ship. HLS moved to `vitis-run --mode hls --tcl`.
  2. `docs/jsc/phase3-handoff.md` §2.1 requires every competitor design to go through OUR synthesis
     path, not the vendor's default project flow. conifer's `build()` IS the vendor default
     flow. Driving HLS ourselves is the method, not a workaround.

THREE THINGS MEASURED THE HARD WAY, all recorded so nobody re-derives them:

  * `vitis-run` accepts NO trailing arguments, but conifer's `build_hls.tcl` reads its flags
    from `$argv`. They are injected via a generated wrapper Tcl that sets `argv`/`argc` and
    then sources conifer's script.
  * `csim=0` is MANDATORY. conifer defaults to csim=1, its C++ testbench fails to link, and the
    run dies before synthesis ever starts.
  * conifer's VHDL backend is Linux-only (compiles a helper with
    `g++ ... $(python3 -m pybind11 --includes) ... -o X.so`), so the HLS backend is the only
    route -- which is fine, because it emits Verilog and `build.tcl` only reads Verilog.

THE ACCURACY NUMBER IS GATED, deliberately. conifer prints
*"Some prediction disagreements are observed for xgboost versions >= 2.0.0"* on import. So
before any accuracy is recorded, conifer's own emitted ensemble JSON is evaluated independently
in numpy and compared against xgboost's predictions on the full test set. A mismatch means the
converted model is not the trained model, and the row is refused. This is the conifer-side
equivalent of Gate 1 -- and it needs no C++ compiler, unlike `conifer.model.compile()`.

Data prep matches `training/dwn_jsc_kaggle.ipynb` EXACTLY -- same OpenML fetch, same
`train_test_split(test_size=0.2, random_state=20260802, stratify=y)`, same `StandardScaler` fit
on train only. A different split makes every accuracy comparison meaningless.

Usage:
    .venv\\Scripts\\python.exe cc\\conifer\\run_conifer.py --depth 4 --trees 20
    .venv\\Scripts\\python.exe cc\\conifer\\run_conifer.py --sweep
    .venv\\Scripts\\python.exe cc\\conifer\\run_conifer.py --depth 4 --trees 20 --no-synth
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from run_gate1 import find_vivado_bin  # noqa: E402
from run_synth import DEFAULT_PART, DEVICE_LUTS, parse_utilization, parse_wns, run_one  # noqa: E402

sys.path.insert(0, REPO)
import datasets  # noqa: E402

# ---------------------------------------------------------------------------------------------
# THE DATASET. Module-level DS is the default (JSC), so a bare `python run_conifer.py` behaves
# exactly as before; main() rebinds it and every derived path from --dataset. Nothing below may
# name a dataset's dimensions -- adding a third means adding a descriptor, not editing this file.
# ---------------------------------------------------------------------------------------------
DS = datasets.JSC

# Matches training/dwn_jsc_kaggle.ipynb and dse/grid.py TRAINING. Do not change independently.
# These describe the SPLIT, not the dataset: a dataset whose descriptor declares a canonical
# `test_split` (MNIST's 'tail:10000') uses that instead and never reaches these.
SEED = 20260802
TEST_SIZE = 0.2

BOARD_PERIOD_NS = 10.0
PRECISION = 'ap_fixed<18,8>'          # conifer's default; a sweep axis in its own right

CACHE = os.path.join(REPO, 'build', 'cc', f'{DS.name}_data.npz')
OUT_ROOT = os.path.join(REPO, 'build', 'cc', 'conifer')
RESULTS = os.path.join(OUT_ROOT, 'results.json')
SNAPSHOT_DIR = os.path.join(REPO, 'docs', DS.cc_results_dir)


def use_dataset(name):
    """Point every derived path and the sweep grid at `name`. Call before anything reads them.

    JSC's paths are unchanged by construction -- `build/cc/conifer` and `docs/jsc/results-cc` are
    what the 14 measured rows already live in, and `docs/jsc/phase3-handoff.md` names the second.
    Other datasets get a suffixed sibling rather than a subdirectory, so nothing existing moves.
    """
    global DS, CACHE, OUT_ROOT, RESULTS, SNAPSHOT_DIR, SWEEP
    DS = datasets.get(name)
    suffix = '' if DS is datasets.JSC else f'-{DS.name}'
    CACHE = os.path.join(REPO, 'build', 'cc', f'{DS.name}_data.npz')
    OUT_ROOT = os.path.join(REPO, 'build', 'cc', f'conifer{suffix}')
    RESULTS = os.path.join(OUT_ROOT, 'results.json')
    SNAPSHOT_DIR = os.path.join(REPO, 'docs', DS.cc_results_dir)
    SWEEP = _sweep_for(DS)

VITIS_RUN_CANDIDATES = [
    os.environ.get('VITIS_RUN', ''),
    r'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat',
    r'C:\Xilinx\Vitis\2025.2\bin\vitis-run.bat',
]

# The plan wants a CURVE -- iso-accuracy and iso-area both need one from each side.
#
# Bounded by the first measurement rather than guessed: `gbdt_d4_n10` (50 trees) came to 8,005
# LUTs, i.e. ~160 LUTs per depth-4 tree, on a 20,800-LUT part. Leaves roughly double per level,
# so the ceiling is around 26 rounds at depth 4, ~14 at depth 5 and ~7 at depth 6. A rectangular
# grid up to 80 rounds would put most of its points several times over the device and spend
# place-and-route reaching a foregone conclusion.
#
# So the grid brackets the useful region and deliberately steps just past the edge at each depth:
# a config that does not fit is a data point (brief §12 risk #2), it just should not be most of
# them. Ordered cheapest-first by total tree count, so an interrupted run banks the most points.
# The grid itself is data -- `datasets.<DS>.cc_gbdt_grid`, with the reasoning for its bounds
# recorded there. Only the ORDERING lives here, because it is a property of how this script runs
# rather than of the dataset: cheapest-first by total tree count, so an interrupted sweep banks
# the most points.
def _sweep_for(ds):
    return sorted(ds.cc_gbdt_grid, key=lambda dn: dn[1] * (2 ** dn[0]))


SWEEP = _sweep_for(DS)


def find_vitis_run():
    for p in VITIS_RUN_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise SystemExit('cannot find vitis-run.bat; set VITIS_RUN to its full path')


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------

def load_data():
    """The dataset, prepared exactly as the DWN training notebooks prepare it.

    EVERY choice here comes from the descriptor, because a comparison is only controlled if the
    competitor sees the same data the DWN saw. Getting the scaler or the split wrong does not
    error -- it silently compares two models trained on different problems, which is the failure
    mode this whole phase exists to avoid.

    Two split conventions, and which one applies is a property of the dataset:

      `test_split='tail:N'`  a CANONICAL split the published literature uses. MNIST's last
                             10,000 rows. Reproducible from the raw data alone, so it is the one
                             every MNIST number in the world is measured on.
      (empty)                no canonical split exists, so the project chose one: a stratified
                             sklearn split at TEST_SIZE/SEED, matching the DWN training
                             notebook. JSC is this case.

    Cached per dataset after first pull. The cache filename carries the dataset name -- a single
    shared `data.npz` would silently serve JSC's features to an MNIST run.
    """
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d['X_train'], d['X_test'], d['y_train'], d['y_test']

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

    print(f'fetching {DS.openml_name} from OpenML (first run only)...')
    data = fetch_openml(DS.openml_name, version=1, as_frame=True)
    X = data.data.to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(data.target.to_numpy())
    if X.shape[1] != DS.features or len(np.unique(y)) != DS.classes:
        raise SystemExit(
            f'{DS.name}: OpenML gave {X.shape[1]} features x {len(np.unique(y))} classes, '
            f'descriptor says {DS.features} x {DS.classes}. Refusing rather than comparing '
            'against a different problem than the DWN was trained on.')

    if DS.test_split.startswith('tail:'):
        n_test = int(DS.test_split.split(':', 1)[1])
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]
    elif DS.test_split:
        raise SystemExit(f'{DS.name}: unsupported test_split {DS.test_split!r}')
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    # MinMax for a dataset whose features are natively bounded (MNIST's 8-bit pixels), standard
    # otherwise. Trees are invariant to a monotonic per-feature rescale, so this does not change
    # the GBDT's accuracy -- but it changes the THRESHOLD VALUES conifer emits, and therefore
    # the comparator widths and the area. Matching the DWN's scaler keeps that comparable.
    scaler = (MinMaxScaler() if DS.scaling == 'minmax' else StandardScaler()).fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------------------------------
# the independent check -- conifer's ensemble, evaluated in numpy
# ------------------------------------------------------------------------------------------

def eval_ensemble(ens, X):
    """Evaluate conifer's OWN emitted ensemble JSON. No compiler, no conifer runtime.

    This is the golden model for the conifer side. It reads what conifer wrote out and walks
    the trees directly, so it cannot inherit a bug from the conversion path it is checking --
    the same reason Gate 1 uses an independent numpy model rather than the emitter's read-back.
    """
    n_classes = ens['n_classes']
    n_out = 1 if n_classes == 2 else n_classes
    scores = np.tile(np.asarray(ens['init_predict'], dtype=np.float64), (len(X), 1))
    le = ens.get('splitting_convention', '<=') == '<='

    for group in ens['trees']:                       # one boosting round
        for c, tree in enumerate(group):             # one tree per class
            feat = np.asarray(tree['feature'])
            thr = np.asarray(tree['threshold'], dtype=np.float64)
            left = np.asarray(tree['children_left'])
            right = np.asarray(tree['children_right'])
            val = np.asarray(tree['value'], dtype=np.float64)

            node = np.zeros(len(X), dtype=np.int64)
            active = left[node] != -1
            while active.any():
                idx = np.where(active)[0]
                f, t = feat[node[idx]], thr[node[idx]]
                go_left = (X[idx, f] <= t) if le else (X[idx, f] < t)
                node[idx] = np.where(go_left, left[node[idx]], right[node[idx]])
                active = left[node] != -1
            scores[:, c % n_out] += val[node]

    return scores / ens.get('norm', 1)


# The project's measured run-to-run training noise. Any accuracy difference below this is not a
# difference (docs/jsc/phase3-handoff.md §2.4), so it is the right bar for "did the conversion
# preserve the model" -- what goes in the comparison table is accuracy, not prediction identity.
NOISE_FLOOR_PP = 0.15


def validate(ens, model, X_test, y_test):
    """Does conifer's converted model still score what the trained model scores?

    Gates on ACCURACY, not on prediction identity, and the distinction is deliberate. Exact
    agreement is the wrong bar: conifer's trees are traversed in a different order and with
    different rounding than xgboost's, so samples sitting on a split boundary flip. Measured at
    d4/n10: 0.61% of predictions differ, but their median top-2 margin is 0.082 against 1.773
    for the agreeing samples -- 21x smaller -- i.e. the flips are where the model is nearly
    indifferent, and they move accuracy by 0.029 pp.

    Returns the CONIFER accuracy as the one to report, because that is the model the HDL
    implements. xgboost's is kept alongside as the reference it was converted from.
    """
    ours = eval_ensemble(ens, X_test)
    pred_ours = (ours[:, 0] > 0).astype(int) if ours.shape[1] == 1 else ours.argmax(1)
    pred_theirs = model.predict(X_test)
    acc_ours = float((pred_ours == y_test).mean())
    acc_theirs = float((pred_theirs == y_test).mean())
    mism = int((pred_ours != pred_theirs).sum())
    delta_pp = 100 * abs(acc_ours - acc_theirs)
    print(f'  conifer-vs-xgboost : accuracy {100*acc_ours:.4f}% vs {100*acc_theirs:.4f}% '
          f'(delta {delta_pp:.4f} pp)')
    print(f'                       {mism}/{len(X_test)} predictions differ '
          f'({100*mism/len(X_test):.4f}%), concentrated at decision boundaries')
    return acc_ours, acc_theirs, mism, delta_pp <= NOISE_FLOOR_PP


# ------------------------------------------------------------------------------------------
# flow
# ------------------------------------------------------------------------------------------

def run_hls(prj_dir):
    """Drive conifer's build_hls.tcl through vitis-run, with argv injected and csim OFF."""
    wrapper = os.path.join(prj_dir, '_synth_only.tcl')
    with open(wrapper, 'w') as fh:
        fh.write('# GENERATED by cc/conifer/run_conifer.py\n'
                 '# vitis-run takes no trailing args; conifer reads its flags from $argv.\n'
                 '# csim=0 is mandatory -- conifer\'s C++ testbench does not link, and csim=1\n'
                 '# kills the run before synthesis.\n'
                 'set argv [list {reset=1 csim=0 synth=1 cosim=0 export=0}]\n'
                 'set argc [llength $argv]\n'
                 'source [file join [file dirname [info script]] build_hls.tcl]\n')
    r = subprocess.run([find_vitis_run(), '--mode', 'hls', '--tcl', wrapper],
                       cwd=prj_dir, capture_output=True, text=True)
    log = (r.stdout or '') + (r.stderr or '')
    ok = 'Generating Verilog RTL' in log
    if not ok:
        print('\n'.join(log.strip().splitlines()[-25:]))
    return ok


def parse_hls_latency(prj_dir, top):
    """Latency in CYCLES and initiation interval, from HLS's own synthesis report.

    Brief §6 requires latency in cycles as well as nanoseconds, because the paper's clock
    speeds do not transfer to a -1 Artix-7 -- and cycles are what survives the part difference.

    This does NOT come from the Vivado reports, which is why it was missed at first. Vivado
    reports Fmax; the pipeline depth HLS chose is only in `<top>_csynth.rpt`:

        |  Latency (cycles) |   Latency (absolute)  |  Interval | Pipeline|
        |   min   |   max   |    min    |    max    | min | max |   Type  |
        |        3|        3|  30.000 ns|  30.000 ns|    1|    1|      yes|

    It varies with the model (measured 3 to 8 cycles across the sweep) and is always II=1, so
    it is directly comparable to DWN's 4 cycles at II=1.
    """
    rpt = os.path.join(prj_dir, top, 'solution1', 'syn', 'report', f'{top}_csynth.rpt')
    if not os.path.exists(rpt):
        return None
    text = open(rpt, errors='replace').read()
    # The FIRST such row after "+ Latency:" is the top-level summary; the per-instance detail
    # tables below it have the same shape, so anchoring on the section header matters.
    head = text.split('+ Latency:', 1)[-1]
    m = re.search(r'\|\s*(\d+)\|\s*(\d+)\|\s*([\d.]+) ns\|\s*[\d.]+ ns\|\s*(\d+)\|\s*(\d+)\|\s*(\w+)\|',
                  head)
    if not m:
        return None
    return {'latency_cycles': int(m.group(2)), 'latency_hls_ns': float(m.group(3)),
            'ii': int(m.group(5)), 'pipelined': m.group(6).strip().lower() == 'yes'}


def stage_rtl(prj_dir, name, top):
    """Concatenate HLS's Verilog into ONE file, and return its repo-relative path.

    Not tidiness -- a hard Windows limit. HLS emits one module per tree with names like
    `conifer_gbdt_d4_n10_decision_function_ap_fixed_18_8_5_3_0_ap_fixed_18_8_5_3_0_49.v`, and
    even the SMALLEST sweep config (depth 4, 10 rounds = 50 trees) produces 100 files totalling
    14,341 characters of command line. `run_one` passes every source as a `-tclargs` argument,
    and cmd.exe caps a command line at ~8,191 characters -- Vivado fails with "The command line
    is too long." At 80 rounds it would be ~8x worse.

    Concatenation rather than changing `scripts/build.tcl`, deliberately: that file is the
    shipped flow every Phase 1 and Phase 2 number came from, and the whole point of Phase 3 is
    that competitor designs go through it UNCHANGED. Verified safe for this output -- no
    `include` directives, no duplicate module names, and per-file `timescale` directives stay
    valid when concatenated.
    """
    syn = os.path.join(prj_dir, top, 'solution1', 'syn', 'verilog')
    parts = sorted(glob.glob(os.path.join(syn, '*.v')))
    if not parts:
        raise SystemExit(f'no Verilog emitted under {syn}')
    rtl_dir = os.path.join(OUT_ROOT, name, 'rtl')
    os.makedirs(rtl_dir, exist_ok=True)
    merged = os.path.join(rtl_dir, f'{top}.v')
    with open(merged, 'w') as out:
        out.write(f'// CONCATENATED by cc/conifer/run_conifer.py from {len(parts)} HLS files.\n'
                  '// See stage_rtl() -- this exists because the Windows command line cannot\n'
                  '// carry that many source paths, not because the RTL was modified.\n\n')
        for p in parts:
            out.write(f'// ---- {os.path.basename(p)} ----\n')
            out.write(open(p, encoding='utf-8', errors='replace').read())
            out.write('\n')
    print(f'  staged RTL         : {len(parts)} files -> {os.path.relpath(merged, REPO)}')
    return os.path.relpath(merged, REPO).replace('\\', '/')


def synthesize(prj_dir, name, top):
    srcs = [stage_rtl(prj_dir, name, top)]
    ok, out = run_one(find_vivado_bin(None), top, srcs, DEFAULT_PART, OUT_ROOT,
                      impl=True, name=name)
    if not ok:
        return None
    util = parse_utilization(os.path.join(out, 'utilization_routed.rpt'))
    wns = parse_wns(os.path.join(out, 'timing_summary_routed.rpt'))
    return {**util, 'wns': wns,
            'fmax_mhz': None if wns is None else round(1000.0 / (BOARD_PERIOD_NS - wns), 1)}


def run_config(depth, trees, X_train, y_train, X_test, y_test, do_synth=True):
    import conifer
    import xgboost as xgb

    name = f'gbdt_d{depth}_n{trees}'
    top = f'conifer_{name}'
    prj = os.path.join(OUT_ROOT, name, 'hls_prj')
    print(f'=== {name} ===')

    t0 = time.time()
    # base_score=0.5 IS NOT COSMETIC -- it is what makes the conversion well-defined.
    #
    # xgboost >= 2.0 auto-fits a per-class base score when base_score is left at None, and
    # conifer 1.9's converter cannot read it: the emitted ensemble comes back with
    #     init_predict = [-4.965, NaN, -4.965, -5.742, -4.862]
    # a NaN for one class, which makes that class's score NaN for every sample and sends the
    # argmax arbitrary. Measured: 127,034 of 166,000 predictions disagreed with xgboost.
    #
    # Setting it explicitly to 0.5 (xgboost's own pre-2.0 default) yields init_predict all
    # zeros and a clean conversion. This is exactly the failure conifer warns about on import
    # ("prediction disagreements ... for xgboost versions >= 2.0.0") and exactly why the
    # validate() step below gates the accuracy number instead of trusting the converter.
    clf = xgb.XGBClassifier(n_estimators=trees, max_depth=depth, tree_method='hist',
                            base_score=0.5, random_state=SEED, n_jobs=8, verbosity=0)
    clf.fit(X_train, y_train)
    print(f'  xgboost            : trained in {time.time()-t0:.0f}s')

    cfg = conifer.backends.xilinxhls.auto_config()
    cfg.update(OutputDir=prj, ProjectName=top, XilinxPart=DEFAULT_PART,
               ClockPeriod=str(BOARD_PERIOD_NS), Precision=PRECISION)
    model = conifer.converters.convert_from_xgboost(clf, cfg)
    model.write()

    ens = json.load(open(os.path.join(prj, f'{top}.json')))
    acc, acc_xgb, mism, agreed = validate(ens, clf, X_test, y_test)

    row = {'name': name, 'depth': depth, 'trees': trees, 'precision': PRECISION,
           'accuracy_pct': round(100 * acc, 4), 'accuracy_xgboost_pct': round(100 * acc_xgb, 4),
           'convert_mismatches': mism, 'convert_ok': bool(agreed),
           'part': DEFAULT_PART, 'clock_ns': BOARD_PERIOD_NS}

    if not agreed:
        # Refused rather than recorded: an accuracy that describes a different model than the
        # hardware implements is exactly the failure this project keeps catching.
        row['status'] = 'convert-mismatch'
        print(f'  REFUSED: conversion moved accuracy by more than {NOISE_FLOOR_PP} pp, so the '
              'row would describe a different model than the HDL implements.')
        return row

    if not do_synth:
        row['status'] = 'not-synthesized'
        return row

    if not run_hls(prj):
        row['status'] = 'hls-failed'
        return row
    res = synthesize(prj, name, top)
    if res is None:
        row['status'] = 'synth-failed'
        return row

    lat = parse_hls_latency(prj, top) or {}
    row.update(status='ok', **{k: res.get(k) for k in ('luts', 'ff', 'bram', 'dsp')},
               wns=res['wns'], fmax_mhz=res['fmax_mhz'], **lat,
               device_pct=round(100 * res['luts'] / DEVICE_LUTS, 2) if res.get('luts') else None)
    print(f"  post-route         : {res.get('luts')} LUT  {res.get('ff')} FF  "
          f"{res.get('bram')} BRAM  {res.get('dsp')} DSP  "
          f"WNS {res['wns']:+.3f}  {res['fmax_mhz']} MHz")
    return row


def save(rows):
    os.makedirs(OUT_ROOT, exist_ok=True)
    existing = {}
    if os.path.exists(RESULTS):
        existing = {r['name']: r for r in json.load(open(RESULTS))}
    existing.update({r['name']: r for r in rows})
    tmp = RESULTS + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(sorted(existing.values(), key=lambda r: r['name']), fh, indent=2)
    os.replace(tmp, RESULTS)
    print(f'\nresults -> {os.path.relpath(RESULTS, REPO)}  ({len(existing)} rows)')


# SNAPSHOT_DIR is defined with the other paths near the top and rebound by use_dataset(); it is
# NOT redefined here. It was, and a second assignment below the first silently reverted every
# --dataset run to JSC's docs/jsc/results-cc/.
SNAPSHOT_COLS = ['name', 'depth', 'trees', 'status', 'accuracy_pct', 'accuracy_xgboost_pct',
                 'convert_mismatches', 'luts', 'ff', 'bram', 'dsp', 'device_pct',
                 'latency_cycles', 'ii', 'wns', 'fmax_mhz', 'latency_ns',
                 'precision', 'part', 'clock_ns']


def snapshot():
    """Commit the measurements, the way `dse/report.py --snapshot` does for the DWN sweep.

    `build/cc/conifer/results.json` is gitignored with the rest of `build/`, on the rule that
    everything there is regenerable. These are not regenerable in any useful sense -- 14 configs
    of HLS plus place-and-route is most of a day of wall clock -- and they are the evidence that
    a competitor design was actually built and measured on our part, not cited from a paper.
    `docs/jsc/phase3-handoff.md` §2.6 names this destination.
    """
    import csv as _csv
    rows = json.load(open(RESULTS))
    rows.sort(key=lambda r: (r['depth'], r['trees']))
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    with open(os.path.join(SNAPSHOT_DIR, 'conifer-results.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(SNAPSHOT_DIR, 'conifer-results.csv'), 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=SNAPSHOT_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    # `status == 'ok'` means SYNTHESIS SUCCEEDED, not that the design fits -- a config several
    # times over the part still routes far enough to report a LUT count, and that count is the
    # measurement. This summary used to print non-ok as "over-device", which lumped
    # `not-synthesized`, `hls-failed` and `convert-mismatch` under a label none of them mean.
    # Fitting is a property of device_pct; being measured at all is a property of status.
    ok = [r for r in rows if r.get('status') == 'ok']
    fit = sum(1 for r in ok if (r.get('device_pct') or 0) <= 100)
    over = len(ok) - fit
    other = {}
    for r in rows:
        if r.get('status') != 'ok':
            other[r['status']] = other.get(r['status'], 0) + 1
    tail = ''.join(f', {n} {s}' for s, n in sorted(other.items()))
    print(f'snapshot -> {os.path.relpath(SNAPSHOT_DIR, REPO)}  '
          f'({len(rows)} rows, {fit} fit, {over} over-device{tail})')
    return 0


def main():
    ap = argparse.ArgumentParser(description='conifer GBDT through our synthesis flow.')
    ap.add_argument('--dataset', default=datasets.JSC.name, choices=datasets.names(),
                    help='which dataset to compare on (default %(default)s)')
    ap.add_argument('--snapshot', action='store_true',
                    help="copy results into the dataset's docs/jsc/results-cc*/ for committing")
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--trees', type=int, default=20)
    ap.add_argument('--sweep', action='store_true', help='the full depth x n_estimators curve')
    ap.add_argument('--no-synth', action='store_true', help='train + convert + validate only')
    ap.add_argument('--force', action='store_true', help='re-run configs already in results.json')
    args = ap.parse_args()

    # Before anything reads a path or the grid. --snapshot included: it writes to a
    # dataset-specific directory and would otherwise overwrite JSC's committed rows.
    use_dataset(args.dataset)

    if args.snapshot:
        return snapshot()

    X_train, X_test, y_train, y_test = load_data()
    split = (f'canonical {DS.test_split}' if DS.test_split
             else f'seed {SEED}, test_size {TEST_SIZE}')
    print(f'{DS.name}: train {X_train.shape}  test {X_test.shape}  '
          f'({DS.scaling} scaling, {split} -- matches the DWN split)')
    print()

    configs = SWEEP if args.sweep else [(args.depth, args.trees)]

    # Resumable: a full sweep is hours of serial Vivado, and an interruption at point 10 must
    # not cost points 1-9. Same rule as dse/run.py.
    done = set()
    if os.path.exists(RESULTS) and not args.force:
        done = {r['name'] for r in json.load(open(RESULTS)) if r.get('status') == 'ok'}
    todo = [(d, n) for d, n in configs if f'gbdt_d{d}_n{n}' not in done]
    if done:
        print(f'{len(done)} already measured, {len(todo)} to go (--force to redo)')
        print()

    rows = []
    for i, (d, n) in enumerate(todo, 1):
        print(f'[{i}/{len(todo)}]', end=' ')
        rows.append(run_config(d, n, X_train, y_train, X_test, y_test,
                               do_synth=not args.no_synth))
        save(rows)          # write after EVERY config, not at the end
        rows = []
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
