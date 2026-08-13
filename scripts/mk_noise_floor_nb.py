"""Generate training/mnist_noise_floor_kaggle.ipynb from the corrected-tau notebook.

The noise-floor run is the SAME pipeline as every other MNIST grid -- environment check, upstream
clone, binarization, training loop, resume logic. Only the title and the grid differ, so this
derives the notebook rather than forking 11 cells that would then drift apart.

    .venv\\Scripts\\python.exe scripts\\mk_noise_floor_nb.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'training', 'mnist_reduction_tau_kaggle.ipynb')
DST = os.path.join(REPO, 'training', 'mnist_noise_floor_kaggle.ipynb')

# The power law the reduction study fitted and confirmed on MNIST: tau proportional to the FINAL
# layer width to the 0.57, anchored on upstream's own value. Reproduces the `tau1p678` slug at
# width 300 exactly, which is how it was checked.
TAU_ANCHOR_W, TAU_ANCHOR = 1000, 1 / 0.3
TAU_EXP = 0.57


def tau_for(final_width):
    return TAU_ANCHOR * (final_width / TAU_ANCHOR_W) ** TAU_EXP


# Four seeds each. The first is the seed every existing MNIST checkpoint used, so config #1 of
# each row is a REPRODUCTION of an already-measured number -- if it does not come back within a
# rounding error, the floor is not the only thing that moved and the run needs investigating
# before its spread is trusted.
SEEDS = [20260811, 20260812, 20260813, 20260814]

# Four widths, not one. JSC measured its 0.15 pp floor from a single configuration and then
# applied it everywhere, which is the same "generalised from one point" error this project has
# had to retract four times. Two of these are the sides of the contested +0.06 pp comparison;
# the other two establish whether the floor varies with width at all.
CONFIGS = [
    ('1x300',          [300],        ['learnable'],             'on the board; bring-up config'),
    ('1x1000',         [1000],       ['learnable'],             'the tau anchor, mid-ladder'),
    ('1x2000',         [2000],       ['learnable'],             'top of the ladder; 98.26%'),
    ('2x[2000,1000]',  [2000, 1000], ['learnable', 'random'],   'best in study, 98.32%'),
]

COMMON = dict(thermometer='distributive', num_classes=10, batch_size=100,
              epochs=30, lr=1e-2, lr_step=14, lr_gamma=0.1,
              thermometer_bits=3, n=6)


def build_grid():
    grid = []
    for label, layers, mapping, note in CONFIGS:
        tau = tau_for(layers[-1])
        w = 'x'.join(str(x) for x in layers)
        for seed in SEEDS:
            grid.append(dict(
                slug=f'mnist_n6_z3_distributive_w{w}_nf_s{seed}',
                label=f'{label} seed={seed}',
                group=f'nf-{label}',
                layers=list(layers), mapping=list(mapping), tau=tau, seed=seed,
                groupsum_group=layers[-1] // COMMON['num_classes'],
                note=note, **COMMON))
    return grid


TITLE = """# MNIST noise floor — the same configurations, four seeds each

**This measures how much a DWN accuracy number moves when nothing changes but the seed.** It is
the cheapest outstanding item in the MNIST study and it gates several conclusions that are
currently unresolved rather than small.

## Why it is needed

`docs/mnist/reduction-ledger.md` reports differences that cannot presently be interpreted:

| claim | margin |
|---|---|
| `2x[2000,1000]` beats `1x2000` | **+0.06 pp** |
| tapers lose to a plain narrow layer at width 500 | −0.13 and −0.27 pp |
| `1x2000` gained from the `tau` correction | +0.09 pp |

JSC measured a **0.15 pp** floor by training one configuration twice. If MNIST's floor is
comparable, every margin in that table is noise and the conclusions drawn from them have to be
withdrawn. **Nothing here assumes it will be** — that is the measurement.

## What it does, and why four widths rather than one

16 runs: **four configurations × four seeds**. JSC took its floor from a single configuration and
then applied it across the whole study, which is the same "generalised from one point" mistake
this project has had to retract four times. Two of the four are the sides of the contested +0.06 pp
comparison; the other two establish whether the floor varies with width at all. It plausibly does
— a 300-node model has far fewer parameters to average over than a 3,000-node one.

**Seed 20260811 is the seed every existing MNIST checkpoint used**, so the first run of each
configuration reproduces an already-measured number. If it does not come back within a rounding
error, something other than the seed has changed and the spread should not be trusted until that
is understood.

`tau` follows the power law the reduction study confirmed — `tau ∝ final_width**0.57`, anchored on
upstream's `1/0.3` at width 1000 — so **no configuration here is `tau`-confounded**. It reproduces
the `tau1p678` value at width 300 exactly, which is how the formula was checked.

## Before you run anything

1. **Accelerator → GPU** (P100 or T4 x2)
2. **Internet → On** — *off by default*. The git clone and the OpenML fetch both need it.

Roughly 2 hours. To continue an unfinished run, add the previous run's **Output** as an input
dataset; the carry-forward cell picks those checkpoints up.

## Reading the result

For each configuration, the **spread across seeds (max − min)** is the floor for a comparison at
that size. Report the spread, not a standard deviation from four points. Then:

- **If the spread exceeds a margin in the table above, that finding is withdrawn**, not weakened.
- If the spread varies with width, say so and use the *relevant* width's floor per comparison
  rather than one global number — which is precisely what JSC did not do.
- Record the result in `docs/mnist/reduction-ledger.md`, correcting the affected rows in place.
"""


def main():
    nb = json.loads(open(SRC, encoding='utf-8').read())
    cells = nb['cells']

    cells[0] = {'cell_type': 'markdown', 'metadata': {},
                'source': TITLE.splitlines(keepends=True)}

    grid = build_grid()
    body = (
        '# ---- the grid: four configurations, four seeds each ----\n'
        '# Generated by scripts/mk_noise_floor_nb.py -- edit that, not this.\n'
        '#\n'
        '# tau follows the power law the reduction study confirmed (tau ~ final_width**0.57,\n'
        '# anchored at 1/0.3 for width 1000), so nothing here is tau-confounded. Seed 20260811 is\n'
        '# the seed the existing checkpoints used, so run #1 of each row reproduces a known number.\n'
        'TRAINING_SET = ' + json.dumps(grid, indent=1) + '\n\n'
        'ONLY_SLUGS = None      # restrict to specific slugs; applied BEFORE by_group is built\n'
        'ONLY_N = None          # cap configs per session; None runs until the grid is done\n'
        'PRECOMPUTE_LIMIT_GB = 6.0\n'
        "print(f'{len(TRAINING_SET)} runs = "
        "{len({c[\"group\"] for c in TRAINING_SET})} configs x "
        "{len({c[\"seed\"] for c in TRAINING_SET})} seeds')\n"
        "for g in sorted({c['group'] for c in TRAINING_SET}):\n"
        "    row = [c for c in TRAINING_SET if c['group'] == g]\n"
        "    print(f\"  {g:16s} layers={row[0]['layers']}  group={row[0]['groupsum_group']:<4} \"\n"
        "          f\"tau={row[0]['tau']:.4f}  {row[0]['note']}\")\n"
    )
    cells[4] = {'cell_type': 'code', 'execution_count': None, 'metadata': {},
                'outputs': [], 'source': body.splitlines(keepends=True)}

    cells[-1] = {'cell_type': 'markdown', 'metadata': {}, 'source': (
        '## After this runs\n\n'
        'Download every `*_checkpoint.pt` into `training/artifacts/`, then compute the spread per '
        'configuration and write it into `docs/mnist/reduction-ledger.md`.\n\n'
        '**The floor is the spread, and it is a per-width number until measured otherwise.** Any '
        'margin in the reduction study smaller than the floor at its width is withdrawn, not '
        'weakened.\n'
    ).splitlines(keepends=True)}

    with open(DST, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(nb, indent=1) + '\n')

    print(f'wrote {os.path.relpath(DST, REPO)}')
    print(f'{len(grid)} runs, {len(CONFIGS)} configs x {len(SEEDS)} seeds')
    for label, layers, _, note in CONFIGS:
        print(f'  {label:16s} group={layers[-1] // 10:<4} tau={tau_for(layers[-1]):.4f}  {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
