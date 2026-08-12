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
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'rtlgen'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract import load_checkpoint, required_int_bits  # noqa: E402
from run_gate1 import find_vivado_bin, gate1, python_exe, run  # noqa: E402
from run_synth import (DEVICE_LUTS, parse_utilization, parse_wns,  # noqa: E402
                       run_one, targets)
import grid as grid_mod  # noqa: E402
from area_model import is_extrapolated, predict  # noqa: E402

sys.path.insert(0, REPO)
import datasets  # noqa: E402

# The dataset being swept. Rebound by main() from --dataset; grid_mod.DS is kept in step with it,
# because build() reads that global. Everything below is DERIVED from DS -- see use_dataset().
DS = datasets.JSC

RESULTS = os.path.join(REPO, 'build', 'dse', 'results.json')
ARTIFACTS = os.path.join(REPO, 'training', 'artifacts')
# Sweep checkpoints live in a subfolder: 37 configs x 2 files would bury the handful of Phase 1
# artifacts, and those must NOT move -- scripts/verify_phase1.py, host.py, run_tb.py and
# run_gate1.py all locate them by a hardcoded path. Worst of all, verify_phase1.py finds the
# 166k test set that way, and if it moved, Gate 1b would silently SKIP rather than fail.
SWEEPS = os.path.join(ARTIFACTS, DS.sweeps_dir)

# Phase 1's checkpoint predates the slug convention. Rather than rename a file every recorded
# number refers to, map it. Step 2c's training notebook writes `<slug>_checkpoint.pt` directly,
# so this table should gain no further entries.
ALIASES = dict(DS.checkpoint_aliases)


def resolve_checkpoint(cfg):
    """The checkpoint that trained THIS config's model, or None.

    Why this exists: `run_config()` emits RTL from the checkpoint, and uses `cfg` only for
    naming and area prediction. So passing one checkpoint to `--all` would emit the SAME design
    37 times under 37 different config names -- every row wrong, and nothing failing to make it
    obvious. Resolution is per config, by model slug, or it does not run.

    ALIASES WIN over a same-slug sweep checkpoint, deliberately. `1x50` is both the grid's first
    ladder rung AND Phase 1's reference config, so 2c retrains it and two files can exist for
    one slug. The Phase 1 one is the checkpoint Gate 1b verified on silicon and the one every
    recorded number (108/1519/1619, 73.84%) refers to, so it stays authoritative -- otherwise
    the baseline row would silently become a different model than the rest of the project's
    numbers describe. Delete the alias if you ever want the retrained one instead.
    """
    slug = cfg.model.slug
    alias = ALIASES.get(slug)
    if alias:
        p = os.path.join(ARTIFACTS, alias)
        if os.path.exists(p):
            return p
    # Three spellings, most specific first.
    #
    # 1. TAU-SUFFIXED, e.g. `mnist_n6_z3_distributive_w300_tau1p678_checkpoint.pt`. When a
    #    schedule changes, a config is retrained at the new tau and the suffix keeps the new
    #    file beside the old one instead of overwriting a checkpoint other numbers refer to.
    #    The suffix is CONSTRUCTED from the tau this grid wants, not globbed -- so it can only
    #    ever match the right one, and a directory holding both tau=3.3333 and tau=1.678 for
    #    the same architecture is unambiguous. This is why it is tried first: a bare-slug file
    #    may well exist and be the superseded model.
    # 2. PREFIXED, `mnist_<slug>_...` -- MNIST's notebooks write the dataset name; JSC's do not.
    # 3. BARE, `<slug>_...` -- JSC's convention.
    #
    # None of this normalises by renaming: renaming a trained checkpoint breaks every recorded
    # number that refers to it, which is the same reason ALIASES exists above.
    tau_tag = f'{cfg.model.tau:.3f}'.replace('.', 'p')
    names = [f'{DS.slug_prefix}{slug}_tau{tau_tag}_checkpoint.pt',
             f'{DS.slug_prefix}{slug}_checkpoint.pt']
    if DS.slug_prefix:
        names.append(f'{slug}_tau{tau_tag}_checkpoint.pt')
        names.append(f'{slug}_checkpoint.pt')
    for root in (SWEEPS, ARTIFACTS):
        for nm in names:
            p = os.path.join(root, nm)
            if os.path.exists(p):
                return p
    return None


SNAPSHOT = os.path.join(REPO, 'docs', DS.results_dir, 'sweep-results.json')


def use_dataset(name):
    """Point every sweep path at `name`'s artifacts. Call before anything reads them.

    JSC's paths are its historical ones and must stay put: docs/results/ and
    training/artifacts/sweeps/ are referenced by REPORT.md, README.md and the `jsc-complete` tag,
    and build/dse/results.json holds 54 measured configs that a moved path would silently re-run
    from scratch. The layout is recorded per dataset in `datasets/` rather than special-cased
    here.
    """
    global DS, RESULTS, SWEEPS, ALIASES, SNAPSHOT
    DS = datasets.get(name)
    grid_mod.DS = DS                      # build() reads the grid module's global
    RESULTS = os.path.join(REPO, 'build', 'dse', DS.build_subdir, 'results.json')
    SWEEPS = os.path.join(ARTIFACTS, DS.sweeps_dir)
    ALIASES = dict(DS.checkpoint_aliases)
    SNAPSHOT = os.path.join(REPO, 'docs', DS.results_dir, 'sweep-results.json')
    return DS


def load_results():
    """Measured results, preferring the working copy and falling back to the committed one.

    `build/dse/results.json` is the one thing under `build/` that CLAUDE.md's rule does not
    actually cover: everything there is supposed to be "regenerable by re-running the flow that
    made it", and this is not -- regenerating it costs a Kaggle GPU session plus hours of
    Vivado. That made `build/` unsafe to delete, which is the opposite of what the rule intends.

    So the committed snapshot (`docs/results/sweep-results.json`, written by
    `dse/report.py --snapshot`) doubles as the recovery source. Wiping `build/` now costs
    nothing but disk: the next run reloads what was already measured and only builds what is
    genuinely missing.
    """
    for path in (RESULTS, SNAPSHOT):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if path is SNAPSHOT:
                print(f'(resuming from the committed snapshot: {len(data)} configs -- '
                      f'build/dse/results.json is absent)')
            return data
    return {}


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


def area_of_cfg(cfg):
    return predict(list(cfg.model.layers), cfg.model.n, cfg.model.thermometer_bits,
                   cfg.model.num_classes, word_bits=cfg.hw.word_bits)


def accuracy_of(checkpoint):
    """Software accuracy, read from the checkpoint that produced the RTL.

    Without this a result row has area and timing but no accuracy, and a Pareto frontier over
    (accuracy, area) cannot be built at all -- the entire point of Study 1. It is read from the
    checkpoint rather than passed in, so it cannot be attached to the wrong config.

    `final_acc` is the primary number: the saved weights are the final epoch, and there is no
    best-checkpoint tracking, so `best_acc` describes weights that were never saved
    (docs/phase1-ledger.md). Quote final, keep best for context.

    UNITS: the checkpoint stores a FRACTION (0.7383614...), while every ledger, report and table
    in this project quotes PERCENT (73.84%). Converted here, once, and the field is named
    `_pct` so a consumer cannot be unsure. The assertion is deliberate -- if a future checkpoint
    ever stores percent, this must fail loudly rather than silently report 7384%.
    """
    import torch
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    r = ck.get('results', {})

    def pct(v):
        if v is None:
            return None
        assert 0.0 <= v <= 1.0, (
            f'accuracy {v} is not a fraction -- the checkpoint format changed, and converting '
            f'it here would be wrong. Fix this function, do not relax the assertion.')
        return round(100.0 * v, 4)

    return {'accuracy_pct': pct(r.get('final_acc')),
            'accuracy_best_epoch_pct': pct(r.get('best_acc'))}


def checkpoint_matches(cfg, checkpoint):
    """Does this checkpoint actually describe the config we think it does? (ok, reason).

    Filenames are slug-based, so a checkpoint from an ABANDONED run has exactly the same name
    as its replacement. That is not hypothetical: the 2026-08-08 tau fix invalidated every
    interpolated-width model, and a pre-fix `n6_z8_distributive_w200` survived in the artifacts
    folder afterwards, silently contributing a row measured against a model that no longer
    exists.

    `tau` is the sharp one. It never reaches the hardware -- it is a uniform divisor and cannot
    change an argmax -- so a wrong-vintage checkpoint produces perfectly valid RTL and perfectly
    plausible area numbers, with only its accuracy quietly belonging to a different model. The
    slug cannot encode it either, since the slug is the training identity and tau is derived
    from width. So it has to be checked.
    """
    import torch
    c = torch.load(checkpoint, map_location='cpu', weights_only=False).get('config', {})
    m = cfg.model
    for field, want, got in (('n', m.n, c.get('n')),
                             ('thermometer_bits', m.thermometer_bits, c.get('thermometer_bits')),
                             ('thermometer', m.thermometer, c.get('thermometer')),
                             ('layers', list(m.layers), list(c.get('layers') or []))):
        if got != want:
            return False, f'{field}: checkpoint has {got!r}, grid expects {want!r}'
    if c.get('tau') is not None and abs(c['tau'] - m.tau) > 1e-6:
        return False, (f'tau: checkpoint has {c["tau"]:.4f}, grid expects {m.tau:.4f} '
                       f'-- this checkpoint predates a schedule change and is a DIFFERENT model')
    return True, 'matches'


def measure_only(cfg, checkpoint, vivado_bin):
    """Emit + synthesize a config WITHOUT Gate 1 and without place-and-route.

    Used only for configs the filter has already rejected, to replace a predicted area with a
    measured one. Two deliberate omissions:

    - **No place-and-route.** It is what fails on an over-budget design, and it is the expensive
      step. Synthesis alone reports utilization past 100%, which is the number wanted here.
    - **No Gate 1.** This config is not entering the frontier as a working design -- it is
      entering it as the point where the part runs out. Area is the claim; correctness is not,
      and running a simulation on something that will never be built would be time spent to
      support a claim nobody is making.

    That second point is the reason this is a separate function rather than a flag on
    `run_config`: Gate 1 gates synthesis THERE, and it must keep doing so. A config that is
    going to be reported as `ok` has to be verified. This one is reported as too big.
    """
    py = python_exe()
    out = {}
    for script, extra in (('emit_core.py', []), ('emit_encoder.py', [])):
        r = run([py, os.path.join(REPO, 'rtlgen', script), checkpoint,
                 '--outdir', cfg.rtl_dir] + extra, capture=True)
        if r.returncode != 0:
            return {'measure_error': f'{script} failed'}

    for top, sources in targets(cfg.rtl_dir):
        ok, out_dir = run_one(vivado_bin, top, sources, cfg.hw.part,
                              os.path.join(cfg.build_dir, 'synth'),
                              period=cfg.hw.clock_ns, impl=False)
        if not ok:
            out['measure_error'] = f'{top} failed to synthesize'
            return out
        util = parse_utilization(os.path.join(out_dir, 'utilization.rpt'))
        out[f'{top}_luts'] = util.get('luts')
        out[f'{top}_ff'] = util.get('ff')
    out['measured_synth_only'] = True
    return out


def widen_for_checkpoint(cfg, checkpoint):
    """Widen the word if this checkpoint's thresholds do not fit the dataset's default.

    The descriptor's `word_bits` is a DEFAULT, not an invariant. MNIST is Q0.8 for z<=8 and
    needs Q1.8 at z=25, where one quantile threshold lands on exactly 1.0 and Q0.8 represents
    [-1, 1). Same dataset, same features, different config -- so a per-dataset width cannot be
    right for every point in a sweep that varies z.

    `required_int_bits()` derives the floor exactly, so this widens rather than guessing, and
    only ever widens: a config whose thresholds fit keeps the descriptor's width, which is what
    makes JSC untouched. The result is a genuinely different hardware config and its name says
    so (`q10.8` vs `q9.8`), which is correct -- it is not the same design.

    Fractional bits are NOT touched. They are not derivable from the checkpoint (see
    required_int_bits' docstring), and narrowing them to keep the word width would trade an
    exact representation for a smaller one silently.

    ⚠️ THAT CHOICE HAS A COST, and it is not always the right trade. Widening changes the word
    width, so a widened config is no longer area-comparable with the rest of a sweep -- which
    matters most on a one-factor-at-a-time axis, where the whole point is that only one thing
    varies. JSC's two `linear` configs are the case in point: Q4.11 would represent them at
    IDENTICAL 16-bit area, while this function would take them to 17-bit Q4.12 and confound the
    encoding axis with word width.

    So: correct when the fractional bits are load-bearing (MNIST needs all 8 to represent 8-bit
    pixels exactly, so Q1.8 at 10 bits is the only option), and the wrong lever when an equally
    exact narrower-fraction format exists at the same width. This function optimises for
    exactness, which is the safe default, not the free one -- check `word_bits_widened` in a
    result before reading a widened config against unwidened ones.
    """
    ck = load_checkpoint(checkpoint)
    thr = ck['thermometer']['thresholds'].numpy()
    need = required_int_bits(thr)
    have = cfg.hw.word_bits - 1 - cfg.hw.frac_bits
    if need <= have:
        return cfg, None
    word = 1 + need + cfg.hw.frac_bits
    note = (f'widened Q{have}.{cfg.hw.frac_bits} -> Q{need}.{cfg.hw.frac_bits} '
            f'({cfg.hw.word_bits} -> {word} bits): thresholds span '
            f'+/-{float(np.abs(thr).max()):.6g} and need {need} integer bits')
    return replace(cfg, hw=replace(cfg.hw, word_bits=word)), note


def run_config(cfg, checkpoint, vivado_bin, label='', group='', impl=False, quiet=True):
    """Emit, Gate 1, synthesize, parse. Returns a result record (never raises on a bad config)."""
    t0 = time.time()
    # BEFORE the record is built, so cfg.name carries the real precision.
    cfg, widen_note = widen_for_checkpoint(cfg, checkpoint)
    est = predict(list(cfg.model.layers), cfg.model.n, cfg.model.thermometer_bits,
                  cfg.model.num_classes, word_bits=cfg.hw.word_bits)
    rec = {
        'name': cfg.name,
        'label': label,
        'group': group,
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
    if widen_note:
        rec['word_bits_widened'] = widen_note

    print(f'--- {cfg.name} ---')
    if widen_note:
        print(f'    {widen_note}')
    print(f'    predicted {est.board_luts:.0f} LUTs '
          f'({est.device_pct:.1f}% of device)'
          f'{"  [extrapolated]" if rec["predicted_extrapolated"] else ""}')

    ok, why = checkpoint_matches(cfg, checkpoint)
    if not ok:
        rec['status'] = 'checkpoint-mismatch'
        rec['error'] = why
        rec['seconds'] = round(time.time() - t0, 1)
        print(f'    CHECKPOINT MISMATCH: {why}')
        print('    Not building. Re-download this config, or delete the stale file.')
        save_result(rec)
        return rec

    ok, info = gate1(checkpoint, vivado_bin, rtl_dir=cfg.rtl_dir,
                     work=os.path.join(cfg.build_dir, 'gate1'),
                     pipe={'lut': cfg.hw.pipe_lut, 'pop': cfg.hw.pipe_pop,
                           'out': cfg.hw.pipe_out, 'enc': cfg.hw.pipe_enc},
                     # Explicit, not left to the emitter's own descriptor lookup: the config
                     # name claims a precision and the RTL must actually be built at it. They
                     # agreed only by coincidence while nothing overrode the default.
                     word_bits=cfg.hw.word_bits, frac_bits=cfg.hw.frac_bits,
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
        #
        # BRAM and DSP are recorded even though every DWN design so far uses ZERO of both. That
        # zero IS the result: it is the central claim against hls4ml, whose quantized MLPs spend
        # DSPs on multiply-accumulate, and against conifer. A comparison table cannot assert
        # "no DSPs, no BRAM" from data that never captured the columns -- and `parse_utilization`
        # was already returning them, so they were being parsed and thrown away.
        for key in ('luts', 'ff', 'bram', 'dsp'):
            rec[f'{top}_{key}'] = util.get(key)
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
    ap.add_argument('--dataset', default=datasets.JSC.name, choices=datasets.names(),
                    help='which dataset to sweep (default %(default)s)')
    ap.add_argument('--checkpoint', help='trained checkpoint for the config(s) being run')
    ap.add_argument('--config', help='run one config by name (or its grid label)')
    ap.add_argument('--all', action='store_true', help='run every config in the grid')
    ap.add_argument('--impl', action='store_true', help='place and route, not just synthesize')
    ap.add_argument('--force', action='store_true', help='re-run configs already in results')
    ap.add_argument('--no-filter', action='store_true',
                    help='synthesize even configs confidently predicted not to fit')
    ap.add_argument('--measure-filtered', action='store_true',
                    help='synthesize (not implement) too-big configs so their area is '
                         'MEASURED rather than predicted -- the frontier edge')
    ap.add_argument('--list', action='store_true', help='show grid vs results and exit')
    ap.add_argument('--vivado-bin', default=None)
    ap.add_argument('--verbose', action='store_true', help='stream Gate 1 output')
    args = ap.parse_args()

    # Before anything reads a path. grid_mod.build() below reads grid's DS, which this sets.
    use_dataset(args.dataset)
    print(f'dataset: {DS.name}  (results {os.path.relpath(RESULTS, REPO)}, '
          f'checkpoints {os.path.relpath(SWEEPS, REPO)})')

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

    if args.config:
        sel = [e for e in entries if args.config in (e[2].name, e[1])]
        if not sel:
            raise SystemExit(f'no config matching {args.config!r}. Try --list.')
    elif args.all:
        sel = entries
    else:
        raise SystemExit('pass --config <name> or --all (or --list).')

    # An explicit --checkpoint applies to ONE config only. Allowing it with --all is the bug
    # this guard exists for: it would emit one model's RTL under every config's name.
    if args.checkpoint and len(sel) > 1:
        raise SystemExit(
            '--checkpoint applies to a single --config. With --all, each config resolves its '
            'own\ncheckpoint by model slug -- otherwise one model would be built under every '
            "config's\nname and every row would be wrong without anything failing.")

    vivado_bin = find_vivado_bin(args.vivado_bin)
    ran = skipped = missing = filtered = 0
    for group, label, cfg, _ in sel:
        # A recorded result counts as done ONLY if it is a measurement. `checkpoint-mismatch`
        # and `untrained` describe the state of the artifacts tree at the time, not the config
        # -- and both become stale the moment the right checkpoint arrives. Treating them as
        # done meant that downloading the corrected `_tau*` set changed nothing: the sweep
        # skipped all five rungs it had just been unblocked for, and reported "skipped 24
        # already-done" as though there were nothing to do.
        #
        # `gate1-failed` and `synth-failed` DO count as done: those are results about the
        # design, and re-running them without changing anything would just fail again.
        TRANSIENT = ('checkpoint-mismatch', 'untrained')
        prior = done.get(cfg.name)
        if prior is not None and prior.get('status') in TRANSIENT:
            prior = None                      # retry: the blocker may have been resolved
        if prior is not None and not args.force:
            skipped += 1
            continue

        # Step 2d: do not spend serial Vivado time on a config confidently too big. The
        # filter deliberately still runs configs just past the threshold -- see
        # grid.should_synthesize; skipping every overshoot would mean nothing ever fails to
        # fit, and the frontier's edge would be predicted rather than measured.
        run_it, why = grid_mod.should_synthesize(cfg)
        if not run_it and not args.no_filter:
            filtered += 1
            rec = {
                'name': cfg.name, 'label': label, 'group': group,
                'status': 'filtered-too-big',
                'nodes': cfg.model.nodes, 'n': cfg.model.n,
                'z': cfg.model.thermometer_bits, 'encoding': cfg.model.thermometer,
                'layers': list(cfg.model.layers), 'pipe': cfg.hw.pipe_slug,
                'clock_ns': cfg.hw.clock_ns,
                'predicted_board_luts': round(area_of_cfg(cfg).board_luts),
                'predicted_extrapolated': is_extrapolated(
                    cfg.model.n, cfg.model.thermometer_bits),
                'error': why,
            }
            # Record ACCURACY even though this config is never synthesized. It was trained --
            # the checkpoint is sitting right there -- and "76.20% is achievable but needs 128%
            # of the device" is exactly the frontier-edge datapoint Study 1 owes (brief §12
            # risk #2). Bailing out before reading it would throw away a measurement we already
            # paid GPU time for, and leave the accuracy-vs-width curve stopping short of the
            # wall it is supposed to locate.
            ck = resolve_checkpoint(cfg)
            if ck:
                rec.update(accuracy_of(ck))
                rec['checkpoint'] = os.path.basename(ck)
            print(f'--- {label} ---')
            acc = rec.get('accuracy_pct')
            print(f'    FILTERED: {why} -- not implemented (--no-filter to force)'
                  + (f'\n    accuracy {acc:.2f}% (trained, kept as a frontier-edge point)'
                     if acc is not None else
                     '\n    no checkpoint, so no accuracy -- area prediction only'))

            # --measure-filtered: SYNTHESIZE (never implement) a too-big config, so its area is
            # measured rather than predicted.
            #
            # Vivado's synthesis reports utilization even past 100% of the part -- verified at
            # 139.28% on a 2400-node design. Only place-and-route actually fails on an
            # over-budget design, and that is the step this skips. So the cost is one synthesis,
            # not a full implementation, and the payoff is that the frontier's edge stops being
            # a prediction: "measured 28,970 LUTs at 76.20%, does not fit" is the claim brief
            # §12 risk #2 asks for, where "predicted 128% of device" is not.
            #
            # Off by default because a normal sweep should not spend Vivado time on configs it
            # has already decided against.
            if args.measure_filtered and ck:
                ok, why_ck = checkpoint_matches(cfg, ck)
                if not ok:
                    print(f'    skipping measurement: {why_ck}')
                else:
                    print('    --measure-filtered: synthesizing for MEASURED area '
                          '(no place-and-route)')
                    m = measure_only(cfg, ck, vivado_bin)
                    rec.update(m)
                    if m.get('dwn_top_luts'):
                        rec['device_pct'] = round(100.0 * m['dwn_top_luts'] / DEVICE_LUTS, 2)
                        print(f'    measured core {m.get("dwn_core_luts")} | '
                              f'encoder {m.get("thermometer_encoder_luts")} | '
                              f'top {m["dwn_top_luts"]} ({rec["device_pct"]}% of device)')
            save_result(rec)
            continue
        if args.checkpoint:
            ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
                    else os.path.join(REPO, args.checkpoint))
            if not os.path.exists(ckpt):
                raise SystemExit(f'checkpoint not found: {ckpt}')
        else:
            ckpt = resolve_checkpoint(cfg)
        if not ckpt:
            missing += 1
            print(f'--- {label} ---')
            print(f'    SKIP: no checkpoint for model {cfg.model.slug} '
                  f'(expected {cfg.model.slug}_checkpoint.pt) -- train it in step 2c')
            continue
        run_config(cfg, ckpt, vivado_bin, label=label, group=group, impl=args.impl,
                   quiet=not args.verbose)
        ran += 1

    print()
    print(f'ran {ran}, skipped {skipped} already-done, {filtered} filtered too-big, '
          f'{missing} untrained (--force to redo). '
          f'results: {os.path.relpath(RESULTS, REPO)}')
    if missing:
        print(f'{missing} configs have no checkpoint. That is step 2c (Kaggle training).')
    if filtered:
        print(f'{filtered} filtered on predicted area. They are RECORDED as '
              f'`filtered-too-big`, not\nhidden -- the frontier ends somewhere and that is a '
              f'result. --no-filter overrides.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
