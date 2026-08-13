"""Render the JSC literature comparison table (Phase 3, docs/phase3-plan.md sec 3).

Citation only - nothing here is resynthesized. The one rule this script enforces is that
rows from different JSC datasets are never printed in the same table, because JSC is two
datasets with a ~1 pp systematic offset. See docs/phase3-ledger.md, 2026-08-10.

    .venv\\Scripts\\python.exe cc\\literature\\table.py
    .venv\\Scripts\\python.exe cc\\literature\\table.py --dataset cernbox
    .venv\\Scripts\\python.exe cc\\literature\\table.py --markdown > out.md
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# Per-benchmark inputs. JSC's paths are its historical ones and must not move -- docs/jsc-report.md,
# README.md and the `jsc-complete` tag reference docs/results/ and docs/results-cc/.
BENCHMARKS = {
    'jsc': {
        'lit': 'jsc_literature.json',
        'ours': os.path.join('docs', 'results', 'sweep-results.json'),
        'conifer': os.path.join('docs', 'results-cc', 'conifer-results.json'),
        'hls4ml': os.path.join('docs', 'results-cc', 'hls4ml-results.json'),
        'default_dataset': 'openml',      # JSC is two datasets; this is the one we trained on
        'noise_floor_pp': 0.15,
    },
    'mnist': {
        'lit': 'mnist_literature.json',
        'ours': os.path.join('docs', 'results-mnist', 'sweep-results.json'),
        'conifer': os.path.join('docs', 'results-cc-mnist', 'conifer-results.json'),
        'hls4ml': os.path.join('docs', 'results-cc-mnist', 'hls4ml-results.json'),
        'default_dataset': 'mnist',       # one canonical split, so no variant to choose
        'noise_floor_pp': 0.24,
    },
}

# Module-level defaults keep every existing call site working unchanged; main() rebinds them.
LIT = os.path.join(HERE, BENCHMARKS['jsc']['lit'])
OURS = os.path.join(REPO, BENCHMARKS['jsc']['ours'])
CONIFER = os.path.join(REPO, BENCHMARKS['jsc']['conifer'])
HLS4ML = os.path.join(REPO, BENCHMARKS['jsc']['hls4ml'])
OUR_DATASET = BENCHMARKS['jsc']['default_dataset']
NOISE_FLOOR_PP = BENCHMARKS['jsc']['noise_floor_pp']

OUR_PART = 'xc7a35t-1'
BOARD_MHZ = 100.0        # the Basys 3 clock every 'fits' claim is judged against


def use_benchmark(name):
    """Point every input at `name`. Call before loading anything."""
    global LIT, OURS, CONIFER, HLS4ML, OUR_DATASET, NOISE_FLOOR_PP
    b = BENCHMARKS[name]
    LIT = os.path.join(HERE, b['lit'])
    OURS = os.path.join(REPO, b['ours'])
    CONIFER = os.path.join(REPO, b['conifer'])
    HLS4ML = os.path.join(REPO, b['hls4ml'])
    OUR_DATASET = b['default_dataset']
    NOISE_FLOOR_PP = b['noise_floor_pp']
    return b


def ours_source():
    return os.path.relpath(OURS, REPO).replace(os.sep, '/')


def load_literature(path=None):
    path = path or LIT
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def load_ours(path=None, fitting_only=True):
    """Our measured, placed-and-routed configs, in the same record shape as the literature.

    fitting_only drops configs that overflow the Basys 3 or miss 100 MHz. They are real
    measurements and they locate the frontier's edge (brief sec 12 risk #2), but a config
    that does not fit is not a design anyone can compare against, so it is off by default.
    """
    path = path or OURS
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        raw = json.load(fh)
    out = []
    for cfg in raw.values():
        if not cfg.get('impl') or cfg.get('dwn_top_luts') is None:
            continue
        # Fits = fits the device AND runs at the board clock. `meets_timing` alone is not
        # enough: a Group B config constrained at 12 ns "meets timing" at 87 MHz, which does not
        # run on a 100 MHz board. JSC never exposed this because its 12 ns variant still reached
        # 102 MHz; MNIST's does not.
        fmax = cfg.get('dwn_top_fmax_mhz')
        meets_board = fmax is None or fmax >= BOARD_MHZ
        fits = ((cfg.get('device_pct') or 0) <= 100.0
                and cfg.get('meets_timing') is not False and meets_board)
        if fitting_only and not fits:
            continue
        out.append({
            'fits': fits,
            'device_pct': cfg.get('device_pct'),
            'method': 'this project',
            'model': cfg.get('label') or cfg['name'],
            'variant': None,
            'dataset': OUR_DATASET,
            'accuracy_pct': cfg['accuracy_pct'],
            'lut': cfg['dwn_top_luts'],
            'ff': cfg.get('dwn_top_ff'),
            'fmax_mhz': cfg.get('dwn_top_fmax_mhz'),
            'latency_ns': cfg.get('latency_ns'),
            'latency_cycles': cfg.get('latency'),
            'part': OUR_PART,
            'encoder_included': True,
            'source': ours_source(),
            'confidence': 'measured',
            'lut_core': cfg.get('dwn_core_luts'),
            'lut_encoder': cfg.get('thermometer_encoder_luts'),
        })
    out.sort(key=lambda r: -r['accuracy_pct'])
    return out


def load_conifer(path=None, fitting_only=True):
    """The conifer GBDT sweep, in the same record shape. Same silicon, same build.tcl, same
    10 ns clock as every DWN row -- these are measurements, not citations."""
    path = path or CONIFER
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        raw = json.load(fh)
    out = []
    for c in raw:
        fits = c.get('status') == 'ok' and c.get('luts')
        if fitting_only and not fits:
            continue
        out.append({
            'method': 'conifer (GBDT)', 'model': c['name'], 'variant': None,
            'dataset': OUR_DATASET, 'accuracy_pct': c['accuracy_pct'],
            'lut': c.get('luts'), 'ff': c.get('ff'), 'dsp': c.get('dsp'), 'bram': c.get('bram'),
            'fmax_mhz': c.get('fmax_mhz'), 'latency_ns': c.get('latency_ns'),
            'latency_cycles': c.get('latency_cycles'), 'part': OUR_PART,
            'encoder_included': 'n/a', 'source': os.path.relpath(CONIFER, REPO).replace(os.sep, '/'),
            'confidence': 'measured', 'fits': bool(fits), 'device_pct': c.get('device_pct'),
        })
    out.sort(key=lambda r: -r['accuracy_pct'])
    return out


def load_hls4ml(path=None, fitting_only=True):
    """The hls4ml quantised MLP sweep, same silicon and same build.tcl as everything else.

    NOTE the accuracy is the FLOAT model's, an upper bound on what the quantised design scores --
    hls4ml's own accuracy needs its generated C++ compiled, which the measuring machine could not
    do. Carried through as a note so the row cannot be quoted as a measured accuracy.
    """
    path = path or HLS4ML
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        raw = json.load(fh)
    out = []
    for c in raw:
        fits = c.get('status') == 'ok' and c.get('luts')
        if fitting_only and not fits:
            continue
        cyc, fmax = c.get('latency_cycles'), c.get('fmax_mhz')
        out.append({
            'method': 'hls4ml (measured)', 'model': c['name'], 'variant': None,
            'dataset': OUR_DATASET, 'accuracy_pct': c.get('accuracy_float_pct'),
            'lut': c.get('luts'), 'ff': c.get('ff'), 'dsp': c.get('dsp'), 'bram': c.get('bram'),
            'fmax_mhz': fmax,
            'latency_ns': (cyc / fmax * 1000) if (cyc and fmax) else None,
            'latency_cycles': cyc, 'part': OUR_PART, 'encoder_included': 'n/a',
            'source': os.path.relpath(HLS4ML, REPO).replace(os.sep, '/'), 'confidence': 'measured',
            'fits': bool(fits), 'device_pct': c.get('device_pct'),
            'accuracy_is_upper_bound': bool(c.get('accuracy_is_float_upper_bound')),
        })
    out.sort(key=lambda r: -(r['accuracy_pct'] or 0))
    return out


def _conv(row):
    e = row.get('encoder_included')
    return {True: 'incl', False: 'core only', 'n/a': 'n/a', 'unknown': '?'}.get(e, '?')


def _fmt(v, spec=''):
    return '-' if v is None else format(v, spec)


def render(rows, markdown=False):
    hdr = ['Method', 'Model', 'Acc %', 'LUT', 'FF', 'Fmax', 'Lat ns', 'Encoder', 'Part']
    body = [[
        r['method'] + ('' if r.get('fits', True) else ' !'),
        r['model'] or '',
        _fmt(r['accuracy_pct'], '.2f') + ('*' if r.get('accuracy_is_upper_bound') else ''),
        _fmt(r['lut'], ','),
        _fmt(r['ff'], ','), _fmt(r['fmax_mhz'], '.0f'), _fmt(r['latency_ns'], '.1f'),
        _conv(r), r['part'] or '?',
    ] for r in rows]
    if markdown:
        lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
        lines += ['| ' + ' | '.join(r) + ' |' for r in body]
        return '\n'.join(lines)
    w = [max(len(hdr[i]), *(len(r[i]) for r in body)) for i in range(len(hdr))] if body else \
        [len(h) for h in hdr]
    lines = ['  '.join(h.ljust(w[i]) for i, h in enumerate(hdr))]
    lines.append('  '.join('-' * x for x in w))
    lines += ['  '.join(c.ljust(w[i]) for i, c in enumerate(r)) for r in body]
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', choices=sorted(BENCHMARKS), default='jsc',
                    help='which study to tabulate (default %(default)s)')
    # NOTE this is the JSC-variant filter, not the benchmark. JSC is two datasets ~1.05 pp apart
    # (docs/phase3-ledger.md); MNIST has one canonical split, so for --benchmark mnist the only
    # value is 'mnist' and the filter is a no-op. Default None = the benchmark's own default.
    ap.add_argument('--dataset', default=None,
                    help="dataset variant within the benchmark, or 'all' (JSC only: "
                         "openml/cernbox/unknown)")
    ap.add_argument('--markdown', action='store_true')
    ap.add_argument('--no-ours', action='store_true', help='literature only')
    ap.add_argument('--include-unfittable', action='store_true',
                    help="also show our configs that overflow the board or miss timing, marked '!'")
    args = ap.parse_args(argv)

    use_benchmark(args.benchmark)
    lit = load_literature()
    rows = list(lit['results'])
    if not args.no_ours:
        rows += load_ours(fitting_only=not args.include_unfittable)
        rows += load_conifer(fitting_only=not args.include_unfittable)
        rows += load_hls4ml(fitting_only=not args.include_unfittable)

    ds = lit['datasets']
    sel_ds = args.dataset or OUR_DATASET
    real = {r['dataset'] for r in rows}          # 'offset_pp' is a note, not a dataset
    wanted = sorted(k for k in ds if k in real) if sel_ds == 'all' else [sel_ds]

    for name in wanted:
        sel = sorted((r for r in rows if r['dataset'] == name),
                     key=lambda r: -(r['accuracy_pct'] or 0))
        if not sel:
            continue
        meta = ds.get(name)
        title = name.upper() if args.benchmark == 'mnist' else f'JSC-{name.upper()}'
        # JSC's descriptors are dicts with name/instances; MNIST's is a plain description string,
        # because there is only one MNIST and nothing to disambiguate.
        if isinstance(meta, dict):
            title += f"  -  {meta['name']}, {meta['instances']:,} instances"
        print()
        print(title)
        print('=' * len(title))
        if name == 'unknown':
            print('!! Dataset not established. Do NOT compare these against anything.')
        print(render(sel, args.markdown))

    # JSC-only: the two-datasets offset. MNIST has one canonical split and no such note.
    off = ds.get('offset_pp') if isinstance(ds, dict) else None
    if off:
        print()
        print(f"NOTE  OpenML scores ~{off['value']} pp higher than CERNBox for the same method.")
        print(f"      {off['how']}")
        print(f"      Our dataset: {lit['_about']['our_dataset'].split(' - ')[0]}")

    unver = [r for r in lit['results'] if r['confidence'] != 'verified']
    if unver:
        print()
        print('UNVERIFIED rows (do not publish without tracing to a primary source):')
        for r in unver:
            print(f"  - {r['method']} {r['model']}: {r['confidence']} ({r['source']})")
    print()
    print(f"PENDING: {', '.join(p['method'] for p in lit['pending'])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
