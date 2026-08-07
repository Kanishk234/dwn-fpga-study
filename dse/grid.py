"""The sweep grid: which configs Phase 2 actually runs, and why not the others.

A full factorial over the axes in `docs/dse-plan.md` §3 is ~1,024 configs, which at 10-20 min of
serial Vivado is 200-340 hours. This file is the **slice** that maps the frontier's shape at a
cost that can actually be paid -- dse-plan §6, adjusted for the fact that this project runs on
ONE machine (CLAUDE.md), not the two the brief's estimate assumed.

Structure, following dse-plan §6:

  1. size ladder      the spine of the frontier, and where the part runs out
  2. one-factor       on two mid-ladder rungs, vary z / encoding / n / L one at a time.
                      This is what reveals which axis buys the most accuracy per LUT --
                      the actual scientific content of Study 1.
  3. group B          pipeline/clock on survivors only. No retraining, so nearly free.

Two things this file does NOT do:
  - it does not train anything (Group A configs each need a Kaggle GPU run -- step 2c)
  - it does not decide what fits; it asks `dse/area_model.py`, and a predicted overshoot is a
    reason to skip Vivado, never a reason to hide the config. Configs that fail to fit are
    plotted as the frontier's edge (brief §12 risk #2).

Usage:
    python dse/grid.py                # summary + budget
    python dse/grid.py --list         # every config, with predicted area
    python dse/grid.py --group-a      # only the configs needing a training run
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'rtlgen'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, HardwareConfig, ModelConfig  # noqa: E402
from area_model import DEVICE_LUTS, is_extrapolated, predict  # noqa: E402

NUM_CLASSES = 5
BASE_N = 6
BASE_Z = 200
BASE_ENC = 'distributive'

# Minutes of serial Vivado per synthesis point, from Phase 1: OOC synth+impl on this machine
# ran ~4 min per target, and a sweep point synthesizes dwn_top once.
MINUTES_PER_SYNTH = 12

# ---------------------------------------------------------------------------------------------
# tau tracks layer width -- it is NOT a constant to copy from `sm`.
# The paper's JSC values, by total node count (docs/paper-configs.md).
# ---------------------------------------------------------------------------------------------
TAU_ANCHORS = [(10, 1 / 0.7), (50, 1 / 0.3), (360, 1 / 0.1), (2400, 1 / 0.03)]


def tau_for(nodes):
    """Interpolate the paper's tau schedule in log-width.

    Getting this wrong does not fail loudly -- it just trains a worse model, and the sweep
    point then reports an accuracy that says more about tau than about the architecture.
    """
    if nodes <= TAU_ANCHORS[0][0]:
        return TAU_ANCHORS[0][1]
    if nodes >= TAU_ANCHORS[-1][0]:
        return TAU_ANCHORS[-1][1]
    import math
    for (w0, t0), (w1, t1) in zip(TAU_ANCHORS, TAU_ANCHORS[1:]):
        if w0 <= nodes <= w1:
            f = (math.log(nodes) - math.log(w0)) / (math.log(w1) - math.log(w0))
            return t0 + f * (t1 - t0)
    return TAU_ANCHORS[-1][1]


# ---------------------------------------------------------------------------------------------
# 1. Size ladder. Widths are multiples of NUM_CLASSES -- GroupSum zero-pads silently otherwise
# and hardware and software then disagree about group boundaries.
#
# Chosen to BRACKET the wall rather than stop short of it. area_model puts the encoder
# saturation point near W~800 and the fit boundary near W~550-600, so the top rungs are
# expected to fail. That is the point: a config that does not fit locates the frontier's edge.
# ---------------------------------------------------------------------------------------------
SIZE_LADDER = [50, 100, 200, 360, 500, 600, 800, 1200]

# Rungs used for one-factor-at-a-time. Mid-ladder on purpose: at 50 nodes everything fits and
# nothing discriminates; at 1200 nothing fits and nothing discriminates.
OFAT_RUNGS = [200, 360]

# The axes varied one at a time. `z` leads because it is the one the paper never sweeps, and
# it sets the saturation ceiling on encoder area -- which dominates (phase2-ledger).
OFAT = {
    # z spans roughly log-uniformly from far below the binding point to far above it, because
    # the two regimes answer different questions and the model cannot predict either:
    #
    #   z small  (16z < slots x ratio)  encoder area is set by z -- fewer bits exist than the
    #                                   mapping has slots, so z directly buys or costs LUTs.
    #   z large  (16z > slots x ratio)  the model says area is FLAT, because comparators are
    #                                   slot-limited. If that holds, accuracy above z~50 is
    #                                   free -- which would be the sweep's headline result.
    #                                   If it does not, the 67% selection ratio is rising
    #                                   toward 1.0 as collisions get rarer, and the area model
    #                                   needs a z-dependent ratio. EITHER OUTCOME IS A RESULT.
    #
    # 200 is the paper's value and is already the ladder rung, so it is not repeated here.
    'z': [8, 25, 50, 100, 400, 800],
    'encoding': ['gaussian', 'linear'],
    'n': [4, 2],
    'layers': ['two', 'three'],
}

# 3. Group B -- no retraining, synthesis only. (label, HardwareConfig kwargs)
GROUP_B = [
    ('3-stage: no OUT reg', dict(pipe_out=0)),
    ('3-stage: no POP reg', dict(pipe_pop=0)),
    ('2-stage', dict(pipe_pop=0, pipe_out=0)),
    ('clock 8ns (125 MHz)', dict(clock_ns=8.0)),
    ('clock 12ns (83 MHz)', dict(clock_ns=12.0)),
]


# Training hyperparameters. NOT swept -- they do not change the hardware, and the paper's JSC
# schedule is what Phase 1 reproduced to 73.84%. Exported so the notebook cannot drift from it.
# Paper: BS=100, LR 1e-2(14) / 1e-3(14) / 1e-4(4) = 32 epochs, i.e. StepLR(14, 0.1).
TRAINING = {
    'batch_size': 100, 'epochs': 32, 'lr': 1e-2,
    'lr_step': 14, 'lr_gamma': 0.1, 'seed': 20260802,
}


def _model(layers, n=BASE_N, z=BASE_Z, enc=BASE_ENC):
    return ModelConfig(n=n, thermometer_bits=z, thermometer=enc, layers=tuple(layers),
                       num_classes=NUM_CLASSES, tau=tau_for(sum(layers)))


def _split(width, parts):
    """A `parts`-layer stack of roughly `width` total nodes, every layer divisible by classes.

    The FINAL layer is what GroupSum reduces, so it is the one that must divide exactly; the
    others are kept aligned anyway so a config reads consistently.
    """
    per = max(NUM_CLASSES, round(width / parts / NUM_CLASSES) * NUM_CLASSES)
    return [per] * parts


def build():
    """Every config in the sweep, tagged by which part of the strategy produced it."""
    out = []  # (group, label, Config, needs_training)

    for w in SIZE_LADDER:
        out.append(('ladder', f'1x{w}', Config(model=_model([w])), True))

    for rung in OFAT_RUNGS:
        for z in OFAT['z']:
            out.append(('ofat-z', f'1x{rung} z={z}', Config(model=_model([rung], z=z)), True))
        for enc in OFAT['encoding']:
            out.append(('ofat-enc', f'1x{rung} {enc}',
                        Config(model=_model([rung], enc=enc)), True))
        for n in OFAT['n']:
            out.append(('ofat-n', f'1x{rung} n={n}', Config(model=_model([rung], n=n)), True))
        for depth, parts in (('two', 2), ('three', 3)):
            layers = _split(rung, parts)
            out.append(('ofat-L', f'{parts}x{layers[0]}', Config(model=_model(layers)), True))

    # Group B rides on the baseline rung's already-trained model: same ModelConfig, different
    # HardwareConfig. That is the whole reason the two are separate objects.
    base = _model([OFAT_RUNGS[-1]])
    for label, hw_kw in GROUP_B:
        out.append(('group-b', f'1x{OFAT_RUNGS[-1]} {label}',
                    Config(model=base, hw=HardwareConfig(**hw_kw)), False))

    return out


def area_of(cfg):
    return predict(list(cfg.model.layers), cfg.model.n, cfg.model.thermometer_bits,
                   cfg.model.num_classes, word_bits=cfg.hw.word_bits)


def main() -> int:
    ap = argparse.ArgumentParser(description='The Phase 2 sweep grid.')
    ap.add_argument('--list', action='store_true', help='every config with predicted area')
    ap.add_argument('--group-a', action='store_true', help='only configs needing training')
    ap.add_argument('--json', metavar='PATH', nargs='?', const='-',
                    help='export the Group A training set as JSON (for the 2c notebook)')
    args = ap.parse_args()

    grid = build()

    if args.json:
        # One source of truth for the sweep. The training notebook consumes this rather than
        # restating the grid, so the two cannot drift -- and a config trained under different
        # parameters than the one synthesized here would be a silently wrong data point.
        #
        # Deduplicated by model slug: Group B shares a trained model with its ladder rung, and
        # nothing else in the grid should ask for the same training run twice.
        seen, out = set(), []
        for group, label, cfg, needs in grid:
            if not needs or cfg.model.slug in seen:
                continue
            seen.add(cfg.model.slug)
            m = cfg.model
            out.append({
                'slug': m.slug, 'label': label, 'group': group,
                'n': m.n, 'thermometer_bits': m.thermometer_bits,
                'thermometer': m.thermometer, 'layers': list(m.layers),
                'mapping': ['learnable'] * len(m.layers),
                'num_classes': m.num_classes, 'tau': m.tau,
                **TRAINING,
            })
        payload = {'training_set': out, 'count': len(out)}
        text = json.dumps(payload, indent=2)
        if args.json == '-':
            print(text)
        else:
            with open(args.json, 'w') as f:
                f.write(text)
            print(f'wrote {args.json}: {len(out)} training runs')
        return 0

    if args.group_a:
        grid = [g for g in grid if g[3]]

    if args.list or args.group_a:
        print(f'{"group":10s} {"config":24s} {"comps":>6} {"core":>6} {"enc":>7} '
              f'{"board":>7} {"%dev":>6}  fit')
        print('-' * 78)
        for group, label, cfg, _ in grid:
            e = area_of(cfg)
            ex = '~' if is_extrapolated(cfg.model.n, cfg.model.thermometer_bits) else ' '
            print(f'{group:10s} {label:24s} {e.comparators:>5}{ex} {e.core_luts:>6.0f} '
                  f'{e.encoder_luts:>7.0f} {e.board_luts:>7.0f} {e.device_pct:>5.1f}%  '
                  f'{"yes" if e.fits() else "NO"}')
        print('-' * 78)
        print('~ = predicted away from the one config the selection ratio was measured at')
        print('    (n=6, z=200). These estimates are extrapolations -- do NOT filter on them.')

    # A config is skipped ONLY if it is predicted to overshoot *and* the prediction is at the
    # calibrated point. An extrapolated overshoot is not evidence -- it would silently drop the
    # z and n points, which are the ones the sweep exists to measure.
    def skip(cfg):
        e = area_of(cfg)
        return (not e.fits()) and not is_extrapolated(cfg.model.n, cfg.model.thermometer_bits)

    runs = [g for g in grid if not skip(g[2])]
    extrap = [g for g in grid if is_extrapolated(g[2].model.n, g[2].model.thermometer_bits)]
    synth_pts = len(runs)

    print()
    print(f'configs total          : {len(grid)}')
    print(f'  will synthesize      : {synth_pts}  <- these get Vivado time')
    print(f'  skipped (overshoot)  : {len(grid) - synth_pts}  <- REPORTED as the '
          f"frontier's edge, not hidden")
    print(f'  extrapolated area    : {len(extrap)}  <- never skipped on prediction alone')
    print(f'training runs needed   : {len([g for g in grid if g[3]])}  (Kaggle GPU, step 2c)')
    print()
    est_h = synth_pts * MINUTES_PER_SYNTH / 60
    print(f'serial Vivado estimate : {synth_pts} x {MINUTES_PER_SYNTH} min = {est_h:.1f} h')
    print(f'dse-plan sec.6 budget  : 40-70 runs, 15-25 h on ONE machine')
    if synth_pts > 70:
        print('  OVER BUDGET -- drop rungs or OFAT points before starting 2e.')
    elif est_h > 25:
        print('  over the hour budget even though the run count fits; expect >1 sitting.')
    else:
        print('  within budget.')
    print()
    print(f'device: {DEVICE_LUTS} LUTs. "fit" uses a 90% margin -- LUT count is necessary,')
    print('not sufficient, and routing can still fail (brief risk #2).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
