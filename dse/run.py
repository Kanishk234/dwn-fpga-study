"""The sweep runner: one config in, one measured result out.

Per config: emit RTL -> Gate 1 -> synthesize -> parse reports -> append a result record.

Three rules this encodes, each of which exists because breaking it would quietly corrupt the
frontier rather than fail:

1. **Gate 1 gates synthesis.** An area number for RTL that has not been proven bit-exact against
   the golden model describes nothing (CLAUDE.md, dse-plan §1). A config whose Gate 1 fails is
   recorded as a failure and NOT synthesized -- it does not silently contribute a data point.

2. **A config that does not fit is a RESULT, not an error.** Failure to route locates the
   congestion wall (brief §12 risk #2), which is a thing Study 1 is trying to measure. Those
   rows are kept, flagged, and plotted.

3. **Core and encoder areas are recorded separately, always.** Brief §6. The encoder costs 14x
   the core at `sm`, and any table that reports only a total understates what the model costs.

Resumability matters more than it looks: 34 synthesis points is several sittings on one machine
(CLAUDE.md -- one machine, not two), and a run interrupted at point 20 must not restart at 1.
Results are appended to a JSON file keyed by config name, and completed configs are skipped
unless --force.

Usage:
    python dse/run.py --list                       # what would run, and what is already done
    python dse/run.py --config n6_z200_... --checkpoint <ckpt.pt>
    python dse/run.py --all --checkpoint <ckpt.pt>   # every config with a checkpoint
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'rtlgen'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gate1 import find_vivado_bin, gate1  # noqa: E402
from run_synth import (DEVICE_LUTS, parse_utilization, parse_wns,  # noqa: E402
                       run_one, targets)
import grid as grid_mod  # noqa: E402
from area_model import is_extrapolated, predict  # noqa: E402

RESULTS = os.path.join(REPO, 'build', 'dse', 'results.json')


def load_results():
    if not os.path.exists(RESULTS):
        return {}
    with open(RESULTS) as f:
        return json.load(f)


def save_result(rec):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    all_r = load_results()
    all_r[rec['name']] = rec
    # Write via a temp file: a sweep is hours long and an interrupted write would lose every
    # earlier point, not just the current one.
    tmp = RESULTS + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(all_r, f, indent=2, sort_keys=True)
    os.replace(tmp, RESULTS)


def accuracy_of(checkpoint):
    """Software accuracy, read from the checkpoint that produced the RTL.

    Without this a result row has area and timing but no accuracy, and a Pareto frontier over
    (accuracy, area) cannot be built at all -- the entire point of Study 1. It is read from the
    checkpoint rather than passed in, so it cannot be attached to the wrong config.

    `final_acc` is the primary number: the saved weights are the final epoch, and there is no
    best-checkpoint tracking, so `best_acc` describes weights that were never saved
    (docs/phase1-ledger.md). Quote final, keep best for context.
    """
    import torch
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    r = ck.get('results', {})
    return {'accuracy': r.get('final_acc'), 'accuracy_best_epoch': r.get('best_acc')}


def run_config(cfg, checkpoint, vivado_bin, label='', impl=False, quiet=True):
    """Emit, Gate 1, synthesize, parse. Returns a result record (never raises on a bad config)."""
    t0 = time.time()
    est = predict(list(cfg.model.layers), cfg.model.n, cfg.model.thermometer_bits,
                  cfg.model.num_classes, word_bits=cfg.hw.word_bits)
    rec = {
        'name': cfg.name,
        'label': label,
        'checkpoint': os.path.basename(checkpoint),
        'nodes': cfg.model.nodes,
        'n': cfg.model.n,
        'z': cfg.model.thermometer_bits,
        'encoding': cfg.model.thermometer,
        'layers': list(cfg.model.layers),
        'pipe': cfg.hw.pipe_slug,
        'clock_ns': cfg.hw.clock_ns,
        'predicted_board_luts': round(est.board_luts),
        'predicted_extrapolated': is_extrapolated(cfg.model.n, cfg.model.thermometer_bits),
        'status': 'pending',
    }
    rec.update(accuracy_of(checkpoint))

    print(f'--- {cfg.name} ---')
    print(f'    predicted {est.board_luts:.0f} LUTs '
          f'({est.device_pct:.1f}% of device)'
          f'{"  [extrapolated]" if rec["predicted_extrapolated"] else ""}')

    ok, info = gate1(checkpoint, vivado_bin, rtl_dir=cfg.rtl_dir,
                     work=os.path.join(cfg.build_dir, 'gate1'),
                     pipe={'lut': cfg.hw.pipe_lut, 'pop': cfg.hw.pipe_pop,
                           'out': cfg.hw.pipe_out, 'enc': cfg.hw.pipe_enc},
                     quiet=quiet)
    rec['latency'] = info.get('latency')
    rec['gate1_core_vectors'] = info.get('dwn_core_tb_vectors')
    rec['gate1_top_vectors'] = info.get('dwn_top_tb_vectors')

    if not ok:
        # Rule 1: no Gate 1, no synthesis. This row exists so the failure is visible in the
        # results table rather than showing up as a missing config nobody notices.
        rec['status'] = 'gate1-failed'
        rec['error'] = info.get('error')
        rec['seconds'] = round(time.time() - t0, 1)
        print(f'    GATE 1 FAILED: {rec["error"]} -- not synthesizing')
        save_result(rec)
        return rec

    print(f'    Gate 1 PASSED (core {rec["gate1_core_vectors"]}, '
          f'top {rec["gate1_top_vectors"]}, latency {rec["latency"]})')

    synth_root = os.path.join(cfg.build_dir, 'synth')
    for top, sources in targets(cfg.rtl_dir):
        ok, out_dir = run_one(vivado_bin, top, sources, cfg.hw.part, synth_root,
                              period=cfg.hw.clock_ns, impl=impl)
        if not ok:
            # Rule 2: this is a data point. A big config failing to place or route is exactly
            # where the frontier's edge is.
            rec['status'] = 'synth-failed'
            rec['error'] = f'{top} failed to build'
            rec['seconds'] = round(time.time() - t0, 1)
            print(f'    SYNTH FAILED on {top} -- recorded as the frontier edge')
            save_result(rec)
            return rec

        u_rpt = 'utilization_routed.rpt' if impl else 'utilization.rpt'
        t_rpt = 'timing_summary_routed.rpt' if impl else 'timing_summary.rpt'
        util = parse_utilization(os.path.join(out_dir, u_rpt))
        wns = parse_wns(os.path.join(out_dir, t_rpt))
        # Rule 3: per-module, never collapsed into one total.
        rec[f'{top}_luts'] = util.get('luts')
        rec[f'{top}_ff'] = util.get('ff')
        if wns is not None:
            rec[f'{top}_wns'] = round(wns, 3)
            rec[f'{top}_fmax_mhz'] = round(1000.0 / (cfg.hw.clock_ns - wns), 1)

    rec['status'] = 'ok'
    rec['impl'] = impl
    rec['device_pct'] = (round(100.0 * rec['dwn_top_luts'] / DEVICE_LUTS, 2)
                         if rec.get('dwn_top_luts') else None)
    rec['seconds'] = round(time.time() - t0, 1)
    print(f'    core {rec.get("dwn_core_luts")} LUT | '
          f'encoder {rec.get("thermometer_encoder_luts")} LUT | '
          f'top {rec.get("dwn_top_luts")} LUT ({rec.get("device_pct")}% dev) | '
          f'Fmax {rec.get("dwn_top_fmax_mhz")} MHz | {rec["seconds"]:.0f}s')
    save_result(rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description='Run sweep configs through Gate 1 + synthesis.')
    ap.add_argument('--checkpoint', help='trained checkpoint for the config(s) being run')
    ap.add_argument('--config', help='run one config by name (or its grid label)')
    ap.add_argument('--all', action='store_true', help='run every config in the grid')
    ap.add_argument('--impl', action='store_true', help='place and route, not just synthesize')
    ap.add_argument('--force', action='store_true', help='re-run configs already in results')
    ap.add_argument('--list', action='store_true', help='show grid vs results and exit')
    ap.add_argument('--vivado-bin', default=None)
    ap.add_argument('--verbose', action='store_true', help='stream Gate 1 output')
    args = ap.parse_args()

    entries = grid_mod.build()
    done = load_results()

    if args.list:
        print(f'{"config":26s} {"group":10s} {"status":14s} {"top LUTs":>9}')
        print('-' * 63)
        for group, label, cfg, _ in entries:
            r = done.get(cfg.name)
            print(f'{label:26s} {group:10s} '
                  f'{(r["status"] if r else "not run"):14s} '
                  f'{(r.get("dwn_top_luts") if r else "") or "":>9}')
        print('-' * 63)
        print(f'{len(done)} of {len(entries)} configs have results')
        return 0

    if not args.checkpoint:
        raise SystemExit('--checkpoint is required to run anything.\n'
                         'Most grid configs have no trained checkpoint yet -- that is step 2c '
                         '(Kaggle).\nUse --list to see what is done.')
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(REPO, args.checkpoint)
    if not os.path.exists(ckpt):
        raise SystemExit(f'checkpoint not found: {ckpt}')

    if args.config:
        sel = [e for e in entries if args.config in (e[2].name, e[1])]
        if not sel:
            raise SystemExit(f'no config matching {args.config!r}. Try --list.')
    elif args.all:
        sel = entries
    else:
        raise SystemExit('pass --config <name> or --all (or --list).')

    vivado_bin = find_vivado_bin(args.vivado_bin)
    ran = skipped = 0
    for group, label, cfg, _ in sel:
        if cfg.name in done and not args.force:
            skipped += 1
            continue
        run_config(cfg, ckpt, vivado_bin, label=label, impl=args.impl,
                   quiet=not args.verbose)
        ran += 1

    print()
    print(f'ran {ran}, skipped {skipped} already-done '
          f'(--force to redo). results: {os.path.relpath(RESULTS, REPO)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
