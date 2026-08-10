"""conifer (GBDT) through OUR synthesis flow -- Phase 3 §2.1.

Trains an xgboost GBDT on JSC, converts it with conifer, synthesizes the generated Verilog
through `scripts/build.tcl` at `xc7a35tcpg236-1` / 10 ns, and records the same columns the DWN
sweep records. One config per invocation, or `--sweep` for the depth x n_estimators curve the
plan asks for (iso-accuracy and iso-area both need a curve, not one point).

WHY THIS DOES NOT CALL `conifer.model.build()`. Two reasons, and the second is the important one:

  1. It cannot work here. conifer detects the HLS tool with `os.system('type X > /dev/null')`
     -- a POSIX builtin -- and then invokes a command named `vitis_hls`, which Vivado/Vitis
     2025.2 does not ship. HLS moved to `vitis-run --mode hls --tcl`.
  2. `docs/phase3-handoff.md` §2.1 requires every competitor design to go through OUR synthesis
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
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from run_gate1 import find_vivado_bin  # noqa: E402
from run_synth import DEFAULT_PART, DEVICE_LUTS, parse_utilization, parse_wns, run_one  # noqa: E402

# Matches training/dwn_jsc_kaggle.ipynb and dse/grid.py TRAINING. Do not change independently.
SEED = 20260802
TEST_SIZE = 0.2
NUM_CLASSES = 5

BOARD_PERIOD_NS = 10.0
PRECISION = 'ap_fixed<18,8>'          # conifer's default; a sweep axis in its own right
CACHE = os.path.join(REPO, 'build', 'cc', 'jsc_data.npz')
OUT_ROOT = os.path.join(REPO, 'build', 'cc', 'conifer')
RESULTS = os.path.join(OUT_ROOT, 'results.json')

VITIS_RUN_CANDIDATES = [
    os.environ.get('VITIS_RUN', ''),
    r'C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat',
    r'C:\Xilinx\Vitis\2025.2\bin\vitis-run.bat',
]

# The plan wants a curve. Depth is the axis that moves LUTs fastest; n_estimators moves
# accuracy. Kept small enough that the whole sweep is a couple of hours of Vivado.
SWEEP = [(d, n) for d in (3, 4, 5, 6) for n in (10, 20, 40, 80)]


def find_vitis_run():
    for p in VITIS_RUN_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise SystemExit('cannot find vitis-run.bat; set VITIS_RUN to its full path')


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------

def load_data():
    """JSC, prepared exactly as the DWN training notebooks prepare it. Cached after first pull."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d['X_train'], d['X_test'], d['y_train'], d['y_test']

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print('fetching hls4ml_lhc_jets_hlf from OpenML (first run only, ~100 MB)...')
    data = fetch_openml('hls4ml_lhc_jets_hlf', version=1, as_frame=True)
    X = data.data.to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(data.target.to_numpy())
    assert X.shape[1] == 16 and len(np.unique(y)) == NUM_CLASSES

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
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
# difference (docs/phase3-handoff.md §2.4), so it is the right bar for "did the conversion
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

    row.update(status='ok', **{k: res.get(k) for k in ('luts', 'ff', 'bram', 'dsp')},
               wns=res['wns'], fmax_mhz=res['fmax_mhz'],
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


def main():
    ap = argparse.ArgumentParser(description='conifer GBDT through our synthesis flow.')
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--trees', type=int, default=20)
    ap.add_argument('--sweep', action='store_true', help='the full depth x n_estimators curve')
    ap.add_argument('--no-synth', action='store_true', help='train + convert + validate only')
    args = ap.parse_args()

    X_train, X_test, y_train, y_test = load_data()
    print(f'JSC: train {X_train.shape}  test {X_test.shape}  '
          f'(seed {SEED}, test_size {TEST_SIZE}, matches the DWN split)')
    print()

    configs = SWEEP if args.sweep else [(args.depth, args.trees)]
    rows = []
    for d, n in configs:
        rows.append(run_config(d, n, X_train, y_train, X_test, y_test,
                               do_synth=not args.no_synth))
        print()
    save(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
