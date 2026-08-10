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
LIT = os.path.join(HERE, 'jsc_literature.json')
OURS = os.path.join(REPO, 'docs', 'results', 'sweep-results.json')

OUR_PART = 'xc7a35t-1'


def load_literature(path=LIT):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def load_ours(path=OURS, fitting_only=True):
    """Our measured, placed-and-routed configs, in the same record shape as the literature.

    fitting_only drops configs that overflow the Basys 3 or miss 100 MHz. They are real
    measurements and they locate the frontier's edge (brief sec 12 risk #2), but a config
    that does not fit is not a design anyone can compare against, so it is off by default.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        raw = json.load(fh)
    out = []
    for cfg in raw.values():
        if not cfg.get('impl') or cfg.get('dwn_top_luts') is None:
            continue
        fits = (cfg.get('device_pct') or 0) <= 100.0 and cfg.get('meets_timing') is not False
        if fitting_only and not fits:
            continue
        out.append({
            'fits': fits,
            'device_pct': cfg.get('device_pct'),
            'method': 'this project',
            'model': cfg.get('label') or cfg['name'],
            'variant': None,
            'dataset': 'openml',
            'accuracy_pct': cfg['accuracy_pct'],
            'lut': cfg['dwn_top_luts'],
            'ff': cfg.get('dwn_top_ff'),
            'fmax_mhz': cfg.get('dwn_top_fmax_mhz'),
            'latency_ns': cfg.get('latency_ns'),
            'latency_cycles': cfg.get('latency'),
            'part': OUR_PART,
            'encoder_included': True,
            'source': 'docs/results/sweep-results.json',
            'confidence': 'measured',
            'lut_core': cfg.get('dwn_core_luts'),
            'lut_encoder': cfg.get('thermometer_encoder_luts'),
        })
    out.sort(key=lambda r: -r['accuracy_pct'])
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
        r['model'] or '', _fmt(r['accuracy_pct'], '.2f'), _fmt(r['lut'], ','),
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
    ap.add_argument('--dataset', choices=['openml', 'cernbox', 'unknown', 'all'],
                    default='openml', help='which JSC dataset (default: openml, the one we use)')
    ap.add_argument('--markdown', action='store_true')
    ap.add_argument('--no-ours', action='store_true', help='literature only')
    ap.add_argument('--include-unfittable', action='store_true',
                    help="also show our configs that overflow the board or miss timing, marked '!'")
    args = ap.parse_args(argv)

    lit = load_literature()
    rows = list(lit['results'])
    if not args.no_ours:
        rows += load_ours(fitting_only=not args.include_unfittable)

    wanted = ['openml', 'cernbox', 'unknown'] if args.dataset == 'all' else [args.dataset]
    ds = lit['datasets']

    for name in wanted:
        sel = sorted((r for r in rows if r['dataset'] == name),
                     key=lambda r: -(r['accuracy_pct'] or 0))
        if not sel:
            continue
        meta = ds.get(name)
        title = f'JSC-{name.upper()}'
        if meta:
            title += f"  -  {meta['name']}, {meta['instances']:,} instances"
        print()
        print(title)
        print('=' * len(title))
        if name == 'unknown':
            print('!! Dataset not established. Do NOT compare these against anything.')
        print(render(sel, args.markdown))

    off = ds['offset_pp']
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
