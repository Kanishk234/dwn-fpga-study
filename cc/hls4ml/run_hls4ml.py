"""hls4ml (quantized MLP) through OUR synthesis flow -- Phase 3 §2.2.

Trains the published JSC MLP in PyTorch, converts it with hls4ml, and synthesizes the generated
Verilog through `scripts/build.tcl` at `xc7a35tcpg236-1` / 10 ns -- the same flow all 54 DWN rows
and all 14 conifer rows went through.

THE EXPERIMENT IS THE SHRINK, NOT THE BASELINE. The published design is **63,251 LUTs on an
xcvu9p** against our 20,800-LUT part -- 3x over, which is arithmetic and needs no measuring. What
nobody has published is what accuracy survives when it is forced to fit. So the baseline is one
row and the shrink sequence is the result.

WINDOWS / 2025.2 NOTES, since the ledger previously recorded this as a likely blocker:

  * **No WSL, no second Vivado install, no `vitis_hls` shim.** hls4ml 1.3.0's Vitis backend
    already issues `vitis-run --tcl build_prj.tcl --mode hls`, which is the 2025.x entry point,
    and it guards its POSIX tool-detection with `if 'linux' in sys.platform`, so that never runs
    here. It also passes build options through a generated `build_opt.tcl` FILE rather than as
    trailing arguments, so the `$argv` problem that conifer hit does not exist.
  * **`vitis-run` only has to be on PATH** -- this script prepends it.
  * **No TensorFlow.** `convert_from_pytorch_model` uses the torch already pinned in
    `requirements.txt`.
  * `build_prj.tcl` runs `catch {config_array_partition -maximum_size $maximum_size}`, and 2025.2
    rejects that option. It is inside a `catch`, so it is cosmetic and the build proceeds -- but
    array partitioning falls back to HLS defaults, which may move resource numbers.

⚠️ **THE ACCURACY COLUMN IS THE FLOAT MODEL'S, NOT THE QUANTIZED DESIGN'S.** Measuring what
hls4ml's fixed-point implementation scores needs `hls_model.predict()`, which compiles the
generated C++ -- and there is no C++ compiler on this machine (the same wall conifer's
`compile()` hit). The conifer flow avoided this by evaluating conifer's own emitted ensemble JSON
in numpy; hls4ml has no equivalent artifact to evaluate, because its model is a C++ dataflow
program rather than a declarative tree list. So the accuracy here is an UPPER BOUND on the
design's, and every row says so.

Data prep matches `training/dwn_jsc_kaggle.ipynb` and `cc/conifer/run_conifer.py` exactly --
same OpenML fetch, same split, same seed, same scaler.

Usage:
    .venv\\Scripts\\python.exe cc\\hls4ml\\run_hls4ml.py --published
    .venv\\Scripts\\python.exe cc\\hls4ml\\run_hls4ml.py --layers 32 16 --reuse 4
    .venv\\Scripts\\python.exe cc\\hls4ml\\run_hls4ml.py --shrink
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'cc', 'conifer'))

from run_gate1 import find_vivado_bin  # noqa: E402
from run_synth import DEFAULT_PART, DEVICE_LUTS, parse_utilization, parse_wns, run_one  # noqa: E402

SEED = 20260802
NUM_CLASSES = 5
N_FEATURES = 16
BOARD_PERIOD_NS = 10.0

# Duarte et al. / Fahim et al., the design every JSC hls4ml number refers to.
PUBLISHED_LAYERS = [64, 32, 32]

OUT_ROOT = os.path.join(REPO, 'build', 'cc', 'hls4ml')
RESULTS = os.path.join(OUT_ROOT, 'results.json')
VITIS_BIN = r'C:\AMDDesignTools\2025.2\Vitis\bin'

# The shrink sequence. Three knobs move area, and they are not interchangeable:
#   layers    -- fewer/narrower neurons: fewer multiplies, and it costs accuracy directly
#   reuse     -- time-multiplex the multipliers: big area win, costs LATENCY not accuracy
#   precision -- narrower fixed point: costs accuracy, and is the axis §7 of the report showed
#                dominates encoder cost on our side too
# Ordered cheapest-to-run first so an interrupted run banks the informative points.
SHRINK = [
    (PUBLISHED_LAYERS, 1, 'ap_fixed<16,6>'),     # the published design, as published
    (PUBLISHED_LAYERS, 4, 'ap_fixed<16,6>'),     # same model, time-multiplexed
    (PUBLISHED_LAYERS, 16, 'ap_fixed<16,6>'),
    ([32, 16, 16], 4, 'ap_fixed<16,6>'),         # half width
    ([32, 16, 16], 4, 'ap_fixed<12,6>'),         # + narrower datapath
    ([16, 8, 8], 4, 'ap_fixed<12,6>'),           # quarter width
]


def load_data():
    """Shared with the conifer flow so the split cannot drift between the two comparisons."""
    from run_conifer import load_data as _ld
    return _ld()


def train_mlp(layers, X_train, y_train, X_test, y_test, epochs=20, bs=1024):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mods, prev = [], N_FEATURES
    for w in layers:
        mods += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    mods.append(nn.Linear(prev, NUM_CLASSES))
    net = nn.Sequential(*mods)

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).long()
    Xte = torch.from_numpy(X_test).float()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc = float((net(Xte).argmax(1).numpy() == y_test).mean())
    return net, acc


def convert_and_build(net, layers, reuse, precision, name):
    import hls4ml
    os.environ['PATH'] = VITIS_BIN + os.pathsep + os.environ.get('PATH', '')
    prj = os.path.join(OUT_ROOT, name, 'prj')

    cfg = hls4ml.utils.config_from_pytorch_model(
        net, input_shape=(None, N_FEATURES), granularity='name',
        default_precision=precision, default_reuse_factor=reuse)
    cfg['Model']['Strategy'] = 'Resource' if reuse > 1 else 'Latency'

    model = hls4ml.converters.convert_from_pytorch_model(
        net, hls_config=cfg, output_dir=prj, project_name=name,
        backend='Vitis', part=DEFAULT_PART, clock_period=BOARD_PERIOD_NS)
    model.write()
    try:
        model.build(reset=True, csim=False, synth=True, cosim=False,
                    validation=False, export=False, vsynth=False)
    except Exception as e:
        print(f'  HLS build failed: {e}')
        return None, prj
    return model, prj


def hls_solution(prj, name):
    """hls4ml names the HLS project `<project_name>_prj`, NOT `<project_name>`.

    conifer does not do this, so the path that worked there silently found nothing here and the
    row came back `no-rtl` even though HLS had succeeded and written 24 Verilog files. Globbed
    rather than hardcoded so a future hls4ml naming change fails loudly instead of quietly.
    """
    for cand in ([os.path.join(prj, f'{name}_prj')] + sorted(glob.glob(os.path.join(prj, '*_prj')))):
        sol = os.path.join(cand, 'solution1')
        if os.path.isdir(sol):
            return sol
    return None


def stage_rtl(prj, name):
    """Concatenate HLS's Verilog into one file, carrying the weight ROMs with it.

    ⚠️ THE `.dat` FILES ARE THE WHOLE PROBLEM HERE, and getting this wrong produces a
    spectacular false result rather than an error.

    At `reuse=1` hls4ml uses the Latency strategy and bakes weights in as inline constants --
    self-contained, and concatenation is safe (that is why the conifer flow, which also uses
    inline constants, needed none of this). At `reuse>1` it switches to the Resource strategy and
    puts the weights in ROMs loaded at elaboration:

        $readmemh("./<module>_wN_ROM_NP_xxxx.dat", rom0);

    That path is RELATIVE, and Vivado runs with the repo root as its working directory. Merging
    the Verilog elsewhere leaves every ROM unresolved, so the weights read as zero, the
    synthesizer folds the dead arithmetic away, and a 64/32/32 MLP reports **235 LUTs and 0 DSPs**
    -- measured, plausible-looking, and completely wrong.

    So: copy the `.dat` files next to the merged Verilog and rewrite each `$readmemh` to an
    absolute path. `impl/verilog` is used rather than `syn/verilog` because it is the
    export-ready set and is where the `.dat` files live.
    """
    sol = hls_solution(prj, name)
    if sol is None:
        return None
    src_dir = os.path.join(sol, 'impl', 'verilog')
    if not os.path.isdir(src_dir):
        src_dir = os.path.join(sol, 'syn', 'verilog')
    parts = sorted(glob.glob(os.path.join(src_dir, '*.v')))
    if not parts:
        return None

    rtl_dir = os.path.join(OUT_ROOT, name, 'rtl')
    os.makedirs(rtl_dir, exist_ok=True)

    dats = glob.glob(os.path.join(src_dir, '*.dat'))
    for d in dats:
        shutil.copy(d, os.path.join(rtl_dir, os.path.basename(d)))

    import re as _re
    merged = os.path.join(rtl_dir, f'{name}.v')
    with open(merged, 'w') as out:
        out.write(f'// CONCATENATED by cc/hls4ml/run_hls4ml.py from {len(parts)} HLS files.\n'
                  f'// {len(dats)} weight ROM .dat files copied alongside; $readmemh paths\n'
                  f'// rewritten to absolute. See stage_rtl() -- relative paths here silently\n'
                  f'// zero the weights and the design collapses to a fraction of its real size.\n\n')
        for p in parts:
            text = open(p, encoding='utf-8', errors='replace').read()
            text = _re.sub(
                r'\$readmemh\("\./([^"]+)"',
                lambda m: '$readmemh("' + os.path.join(rtl_dir, m.group(1)).replace('\\', '/') + '"',
                text)
            out.write(f'// ---- {os.path.basename(p)} ----\n{text}\n')

    verify_staged(merged, name, len(dats))
    print(f'  staged RTL         : {len(parts)} files + {len(dats)} ROMs '
          f'-> {os.path.relpath(merged, REPO)}')
    return os.path.relpath(merged, REPO).replace('\\', '/')


def verify_staged(merged, name, n_dats):
    """Refuse to synthesize RTL that cannot be complete. Loud beats plausible.

    Two checks, both of which the 235-LUT result would have failed:
      * every instantiated module is defined in the file (note HLS writes attribute-prefixed
        declarations like `(* ... *) module foo`, so anchoring on `^module` misses them)
      * every `$readmemh` target actually exists on disk
    """
    import re as _re
    text = open(merged, errors='replace').read()
    defined = set(_re.findall(r'(?:^|\)\s*)module\s+(\w+)', text, _re.M))
    inst = set(_re.findall(r'^\s*([A-Za-z_]\w*)\s+(?:#\s*\([^;]*?\)\s*)?\w+\s*\(', text, _re.M))
    kw = {'assign', 'wire', 'reg', 'always', 'if', 'case', 'input', 'output', 'parameter',
          'localparam', 'generate', 'for', 'begin', 'end', 'initial', 'function', 'task',
          'endmodule', 'module', 'integer', 'genvar', 'else', 'real', 'defparam', 'and', 'or',
          'not', 'nand', 'nor', 'xor', 'buf', 'return', 'while', 'repeat', 'forever'}
    missing = sorted(i for i in inst if i not in defined and i not in kw)
    if missing:
        raise SystemExit(f'staged RTL for {name} instantiates undefined modules: {missing[:6]} '
                         f'-- synthesizing it would report a fraction of the real area')
    for target in _re.findall(r'\$readmemh\("([^"]+)"', text):
        if not os.path.exists(target):
            raise SystemExit(f'staged RTL for {name} reads a missing ROM: {target} '
                             f'-- the weights would be zero and the design would collapse')


def parse_drc_overuse(out_dir):
    """How badly an over-device design missed, from Vivado's DRC errors.

    A config that does not fit is a data point (brief §12 risk #2), and "synth-failed" alone
    throws away the interesting part. When DRC rejects a design the placer never runs, so there
    is no utilization report -- but the DRC text carries exact required-vs-available counts:

        Resource utilization: LUT as Logic over-utilized ... requires 259492 of such cell types
        but only 20800 compatible sites are available

    Those numbers are what make "the standard flow does not fit" quantitative instead of a claim.
    """
    log = os.path.join(out_dir, 'vivado.log')
    if not os.path.exists(log):
        return {}
    import re
    text = open(log, errors='replace').read()
    out = {}
    for res, req, avail in re.findall(
            r'Resource utilization: ([\w ]+?) over-utilized.*?requires (\d+) of such cell types '
            r'but only (\d+) compatible sites', text, re.S):
        out[res.strip()] = {'required': int(req), 'available': int(avail),
                            'over_by': round(int(req) / int(avail), 1)}
    m = re.search(r'type DSP have been overutilized\. Used = (\d+), Available = (\d+)', text)
    if m:
        out['DSP'] = {'required': int(m.group(1)), 'available': int(m.group(2)),
                      'over_by': round(int(m.group(1)) / int(m.group(2)), 1)}
    return out


UNIT_NS = {'ns': 1.0, 'us': 1e3, 'ms': 1e6, 'sec': 1e9, 's': 1e9}


def parse_hls_latency(prj, name):
    """Latency in CYCLES and the initiation interval, from HLS's own `* Summary:` table.

    ⚠️ THE UNIT IS NOT ALWAYS ns, and assuming it was produced a silently wrong number rather
    than an error. Latency-strategy designs report

        |        4|        4|  40.000 ns|  40.000 ns|    1|    1|      yes|

    but Resource-strategy (reuse>1) designs are slower and HLS switches to microseconds:

        |       30|       34|   0.300 us|   0.340 us|    4|    4| dataflow|

    A regex requiring ' ns|' fails to match the second form, and every reuse>1 row came back
    reporting **0 cycles** -- which hid the single most important latency result in this
    comparison: the fitting hls4ml design runs at 30-34 cycles with **II=4**, against DWN's
    4 cycles at II=1.

    Also note the Pipeline Type is `dataflow`, not `yes`, for these -- so a boolean test against
    'yes' silently reports them as un-pipelined.
    """
    sol = hls_solution(prj, name)
    if sol is None:
        return {}
    rpt = os.path.join(sol, 'syn', 'report', f'{name}_csynth.rpt')
    if not os.path.exists(rpt):
        return {}
    import re
    text = open(rpt, errors='replace').read()
    # Anchor on the top-level Summary: the per-instance Detail tables below share this shape.
    seg = text.split('+ Latency:', 1)[-1].split('+ Detail:', 1)[0]
    m = re.search(r'\|\s*(\d+)\|\s*(\d+)\|\s*([\d.]+)\s*(\w+)\|\s*([\d.]+)\s*(\w+)\|'
                  r'\s*(\d+)\|\s*(\d+)\|\s*([\w-]+)\s*\|', seg)
    if not m:
        return {}
    scale = UNIT_NS.get(m.group(6).lower(), 1.0)
    ptype = m.group(9).strip().lower()
    return {'latency_cycles': int(m.group(2)), 'latency_cycles_min': int(m.group(1)),
            'latency_hls_ns': round(float(m.group(5)) * scale, 3),
            'ii': int(m.group(8)), 'pipeline_type': ptype,
            'pipelined': ptype in ('yes', 'dataflow')}


def config_name(layers, reuse, precision):
    """One place, so the resume check and the runner cannot disagree about a config's identity."""
    tag = 'x'.join(map(str, layers))
    prec = precision.replace('ap_fixed<', '').replace('>', '').replace(',', '.')
    return f'mlp_{tag}_r{reuse}_{prec}'.replace('.', '_')


def run_config(layers, reuse, precision, X_train, y_train, X_test, y_test):
    name = config_name(layers, reuse, precision)
    print(f'=== {name} ===')

    t0 = time.time()
    net, acc = train_mlp(layers, X_train, y_train, X_test, y_test)
    print(f'  pytorch float      : {100*acc:.4f}% test accuracy  ({time.time()-t0:.0f}s)')

    row = {'name': name, 'layers': layers, 'reuse': reuse, 'precision': precision,
           'accuracy_float_pct': round(100 * acc, 4), 'accuracy_is_float_upper_bound': True,
           'part': DEFAULT_PART, 'clock_ns': BOARD_PERIOD_NS}

    model, prj = convert_and_build(net, layers, reuse, precision, name)
    if model is None:
        row['status'] = 'hls-failed'
        return row
    row.update(parse_hls_latency(prj, name))

    src = stage_rtl(prj, name)
    if src is None:
        row['status'] = 'no-rtl'
        return row

    ok, out = run_one(find_vivado_bin(None), name, [src], DEFAULT_PART, OUT_ROOT,
                      impl=True, name=name)
    if not ok:
        row['status'] = 'synth-failed'
        row['drc_overuse'] = parse_drc_overuse(os.path.join(OUT_ROOT, name))
        worst = max(row['drc_overuse'].items(), key=lambda kv: kv[1]['over_by'], default=None)
        if worst:
            print(f"  post-route         : DOES NOT FIT -- worst {worst[0]}: "
                  f"{worst[1]['required']:,} needed vs {worst[1]['available']:,} "
                  f"({worst[1]['over_by']}x over)")
        return row
    util = parse_utilization(os.path.join(out, 'utilization_routed.rpt'))
    wns = parse_wns(os.path.join(out, 'timing_summary_routed.rpt'))
    row.update(status='ok', **{k: util.get(k) for k in ('luts', 'ff', 'bram', 'dsp')}, wns=wns,
               fmax_mhz=None if wns is None else round(1000.0 / (BOARD_PERIOD_NS - wns), 1),
               device_pct=round(100 * util['luts'] / DEVICE_LUTS, 2) if util.get('luts') else None)
    print(f"  post-route         : {util.get('luts')} LUT  {util.get('ff')} FF  "
          f"{util.get('bram')} BRAM  {util.get('dsp')} DSP  "
          f"{row.get('device_pct')}% dev  {row.get('fmax_mhz')} MHz")
    return row


def save(rows):
    os.makedirs(OUT_ROOT, exist_ok=True)
    have = {}
    if os.path.exists(RESULTS):
        have = {r['name']: r for r in json.load(open(RESULTS))}
    have.update({r['name']: r for r in rows})
    tmp = RESULTS + '.tmp'
    json.dump(sorted(have.values(), key=lambda r: r['name']), open(tmp, 'w'), indent=2)
    os.replace(tmp, RESULTS)
    print(f'\nresults -> {os.path.relpath(RESULTS, REPO)}  ({len(have)} rows)')


SNAPSHOT_DIR = os.path.join(REPO, 'docs', 'jsc', 'results-cc')
SNAPSHOT_COLS = ['name', 'layers', 'reuse', 'precision', 'status', 'accuracy_float_pct',
                 'accuracy_is_float_upper_bound', 'luts', 'ff', 'bram', 'dsp', 'device_pct',
                 'latency_cycles', 'ii', 'pipeline_type', 'wns', 'fmax_mhz',
                 'lut_required', 'lut_over_by', 'dsp_required', 'dsp_over_by',
                 'part', 'clock_ns']


def snapshot():
    """Commit the measurements, as `dse/report.py --snapshot` does for the DWN sweep.

    Over-device rows carry their DRC required-vs-available counts flattened into columns, since
    "it does not fit" is only useful with the margin attached.
    """
    import csv as _csv
    rows = json.load(open(RESULTS))
    rows.sort(key=lambda r: -(((r.get('drc_overuse') or {}).get('LUT as Logic') or {}).get(
        'required') or r.get('luts') or 0))
    flat = []
    for r in rows:
        d = r.get('drc_overuse') or {}
        lut, dsp = d.get('LUT as Logic') or {}, d.get('DSP') or {}
        flat.append({**r,
                     'lut_required': lut.get('required'), 'lut_over_by': lut.get('over_by'),
                     'dsp_required': dsp.get('required'), 'dsp_over_by': dsp.get('over_by'),
                     'layers': 'x'.join(map(str, r.get('layers', [])))})
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    json.dump(rows, open(os.path.join(SNAPSHOT_DIR, 'hls4ml-results.json'), 'w'), indent=2)
    with open(os.path.join(SNAPSHOT_DIR, 'hls4ml-results.csv'), 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=SNAPSHOT_COLS, extrasaction='ignore')
        w.writeheader(); w.writerows(flat)
    ok = sum(1 for r in rows if r.get('status') == 'ok')
    print(f'snapshot -> {os.path.relpath(SNAPSHOT_DIR, REPO)}  '
          f'({len(rows)} rows, {ok} fit, {len(rows)-ok} over-device)')
    return 0


def main():
    ap = argparse.ArgumentParser(description='hls4ml MLP through our synthesis flow.')
    ap.add_argument('--published', action='store_true', help='the published 64/32/32 design')
    ap.add_argument('--layers', type=int, nargs='+', default=None)
    ap.add_argument('--reuse', type=int, default=1)
    ap.add_argument('--precision', default='ap_fixed<16,6>')
    ap.add_argument('--shrink', action='store_true', help='the full shrink sequence')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--snapshot', action='store_true',
                    help='copy results into docs/jsc/results-cc/ for committing')
    args = ap.parse_args()

    if args.snapshot:
        return snapshot()

    X_train, X_test, y_train, y_test = load_data()
    print(f'JSC: train {X_train.shape}  test {X_test.shape}  (seed {SEED}, matches DWN split)\n')

    if args.shrink:
        configs = SHRINK
    elif args.published:
        configs = [(PUBLISHED_LAYERS, args.reuse, args.precision)]
    else:
        configs = [(args.layers or PUBLISHED_LAYERS, args.reuse, args.precision)]

    # Both 'ok' and 'synth-failed' are RESULTS -- a config that does not fit is a data point
    # (brief §12 risk #2), and re-running one costs ~25 min of Vivado to learn nothing. Only
    # 'hls-failed' / 'no-rtl' are errors worth retrying.
    done = set()
    if os.path.exists(RESULTS) and not args.force:
        done = {r['name'] for r in json.load(open(RESULTS))
                if r.get('status') in ('ok', 'synth-failed')}

    todo = [(l, r, p) for (l, r, p) in configs if config_name(l, r, p) not in done]
    if done:
        print(f'{len(done)} already measured, {len(todo)} to go (--force to redo)\n')

    for i, (layers, reuse, prec) in enumerate(todo, 1):
        print(f'[{i}/{len(todo)}]', end=' ')
        save([run_config(layers, reuse, prec, X_train, y_train, X_test, y_test)])
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
