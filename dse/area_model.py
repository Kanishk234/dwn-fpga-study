"""Predict a config's LUT area without launching Vivado.

This is step 2b. It exists because step 2d filters configs on predicted area BEFORE spending
serial Vivado time on them, and `docs/dse-plan.md` §5's original formula assumed the encoder
costs "up to 3.2x the core" (brief §12 risk #3, from Mecik & Kumm). Phase 1 measured **14.06x**.
Filtering with the old number would pass configs that overshoot the part by a factor of four,
and the sweep would find that out one 15-minute synthesis at a time.

WHAT IS MEASURED AND WHAT IS INFERRED -- the distinction matters more than the numbers:

  measured   encoder 1519 LUTs / 202 comparators at 16 bits  -> 0.4700 LUT/bit
  measured   dwn_top 1619 LUTs, dwn_core 108, board 2058     -> harness = 2058 - 1619 = 439
  measured   202 of 300 wiring slots were distinct           -> 67% selection ratio, ONE POINT
  inferred   reduction ~58 LUTs = 108 core - 50 nodes        -> by subtraction, NOT measured

The reduction number is the weak one. Vivado inlined `lut_node`, `popcount` and `argmax` into
the top level, so the hierarchical report attributes all 108 LUTs to `dwn_core` and the split
between "the network" and "the adder trees" is arithmetic, not observation. The ledger already
carries an open question to synthesize the reduction standalone; until that happens, treat the
core estimate as the softer half of this model even though it is the smaller half.

The encoder half is the one that decides what fits, and it is the better-grounded one.

Usage:
    python dse/area_model.py                # self-test against the Phase 1 measurements
    python dse/area_model.py --ladder       # project the paper's sm/md/lg configs
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'rtlgen'))

DEVICE_LUTS = 20800          # XC7A35T
JSC_FEATURES = 16

# ---------------------------------------------------------------------------------------------
# Calibration constants. Every one of these traces to a number in docs/phase1-ledger.md.
# ---------------------------------------------------------------------------------------------

# 1519 LUTs / (202 comparators x 16 bits). A compare-against-constant costs about W/2 on a
# carry chain, and 0.47 x 16 = 7.5 LUTs is exactly that -- Vivado is already near-optimal per
# comparator, and per-comparator sharing was measured and ruled out at -5%.
LUT_PER_COMPARATOR_BIT = 1519 / (202 * 16)

# One DWN node is one LUT6 (brief §4) -- the architectural premise of the whole project.
# MEASURED 2026-08-07, not assumed: `nodes_only` (50 lut_node, real tables and wiring)
# synthesizes to exactly 50 LUTs out-of-context. See scripts/experiment_reduction.py.
LUT_PER_NODE = 1.0

# Reduction = per-class popcounts + an argmax tree.
#
# RECALIBRATED 2026-08-07 on three measured configs, after a flat 1.0 LUT/bit (fitted on `sm`
# alone) underestimated the core by 14-19% at wider layers:
#
#   config       group  reduction  LUT per final bit
#   1x50            10         58               1.00
#   1x200 z=8       40        266               1.27
#   1x360 z=8       72        505               1.36
#
# Cost per bit RISES with group width, and it has to: a popcount is an adder tree, and wider
# groups mean more tree levels carrying wider adders. A constant was always going to be wrong
# away from the width it was fitted at -- it just could not be seen from one data point.
LUT_PER_FINAL_BIT_BASE = 1.0          # at the reference group width below
REDUCTION_GROUP_REF = 10              # `sm`'s group size, where the base was measured
LUT_PER_FINAL_BIT_SLOPE = 0.13        # per doubling of group width


def popcount_lut_per_bit(group):
    """LUTs per final-layer bit for a `group`-wide popcount."""
    if group <= 0:
        return LUT_PER_FINAL_BIT_BASE
    return (LUT_PER_FINAL_BIT_BASE
            + LUT_PER_FINAL_BIT_SLOPE * math.log2(group / REDUCTION_GROUP_REF))


def argmax_luts(num_classes, score_w):
    """A K-way argmax is K-1 comparisons of score_w bits, at about W/2 LUTs each."""
    return (num_classes - 1) * score_w / 2.0

# Everything outside dwn_top on the board: UART, loader, vector store, benchmark FSM, seg7,
# I/O buffers. Measured as 2058 - 1619. Roughly constant -- it does not scale with the model.
HARNESS_LUTS = 2058 - 1619

# Fraction of wiring slots that resolve to DISTINCT thermometer bits. At `sm`, the learnable
# mapping selected 202 distinct bits out of 50 nodes x 6 slots = 300.
# *** ONE DATA POINT. *** A wider model may reuse thresholds more (pushing this down) or spread
# out more (pushing it up). Only training md/lg settles it.
SELECTION_RATIO = 202 / 300

# The config the ratio above was measured at. Away from this point the estimate is an
# EXTRAPOLATION, and `is_extrapolated()` says so.
#
# Why this matters more than it looks: 202/300 is not a collision statistic. If the mapping
# picked slots uniformly from the 3200 available bits, occupancy would predict ~286 distinct,
# not 202. The gap is the LEARNED CONCENTRATION -- four features carry 153 of 202 comparators.
# How hard the mapping concentrates depends on how many thresholds each feature has, i.e. on z,
# and on how many slots each node has, i.e. on n. One data point cannot separate those.
#
# Consequence for step 2d: configs off this point must NOT be filtered out on predicted area
# alone. z is the axis the sweep most wants to characterize and the one this model is least
# able to predict -- so those points get trained and measured, not estimated away.
CALIBRATED_N = 6
CALIBRATED_Z = 200


def is_extrapolated(n, z):
    """True when the selection ratio is being used away from where it was measured."""
    return n != CALIBRATED_N or z != CALIBRATED_Z


@dataclass
class AreaEstimate:
    nodes: int
    comparators: int
    core_luts: float
    encoder_luts: float

    @property
    def top_luts(self) -> float:
        return self.core_luts + self.encoder_luts

    @property
    def board_luts(self) -> float:
        return self.top_luts + HARNESS_LUTS

    @property
    def device_pct(self) -> float:
        return 100.0 * self.board_luts / DEVICE_LUTS

    @property
    def encoder_ratio(self) -> float:
        return self.encoder_luts / self.core_luts if self.core_luts else float('inf')

    def fits(self, margin=0.90) -> bool:
        """Does this plausibly fit? Margin is deliberate: LUT count is necessary, not
        sufficient -- routing can fail on a design that fits by count (brief §12 risk #2)."""
        return self.board_luts <= margin * DEVICE_LUTS


def predict_comparators(layers, n, z, features=JSC_FEATURES, ratio=SELECTION_RATIO):
    """How many distinct thermometer bits the first layer reads.

    Two regimes, and the ceiling is the important one: the first layer has `W1 x n` wiring
    slots, but there are only `features x z` thermometer bits in existence. Past that point,
    adding nodes stops adding comparators -- the encoder SATURATES. That is why encoder cost
    does not scale with the model the way the core does, and why `z` (not node count) sets the
    ceiling on encoder area.
    """
    slots = layers[0] * n
    return int(min(features * z, round(slots * ratio)))


def predict(layers, n, z, num_classes, word_bits=16, features=JSC_FEATURES):
    """Predicted area for one config. Pure arithmetic -- no Vivado, no checkpoint."""
    nodes = sum(layers)
    comparators = predict_comparators(layers, n, z, features)

    encoder = comparators * word_bits * LUT_PER_COMPARATOR_BIT

    # The reduction depends on GROUP width, not just on how many final bits there are: the same
    # 360 bits cost more as 5 groups of 72 than as 36 groups of 10.
    group = layers[-1] // num_classes
    score_w = max(1, math.ceil(math.log2(group + 1)))
    reduction = layers[-1] * popcount_lut_per_bit(group) + argmax_luts(num_classes, score_w)
    core = nodes * LUT_PER_NODE + reduction

    return AreaEstimate(nodes=nodes, comparators=comparators,
                        core_luts=core, encoder_luts=encoder)


# ---------------------------------------------------------------------------------------------
# Self-test: reproduce the Phase 1 measurements. A model that cannot recover the one config we
# actually synthesized has no business filtering the other forty.
# ---------------------------------------------------------------------------------------------

# Every config measured out-of-context so far. A model fitted on one point cannot be checked
# against that same point in any meaningful way -- these are what stop it drifting.
MEASURED = [
    # (label, layers, n, z, core, encoder, top)
    ('1x50',      [50],  6, 200, 108, 1519, 1619),
    ('1x200 z=8', [200], 6,   8, 466,  879, 1345),
    ('1x360 z=8', [360], 6,   8, 865,  970, 1835),
]
MEASURED_BOARD = 2058     # 1x50 with the harness and I/O buffers


def _selftest() -> int:
    print(f'{"config":12s} {"quantity":10s} {"predicted":>10} {"measured":>10} {"error":>9}')
    print('-' * 55)
    worst, worst_extrap = 0.0, 0.0
    for label, layers, n, z, core, enc, top in MEASURED:
        est = predict(layers=layers, n=n, z=z, num_classes=5)
        ex = is_extrapolated(n, z)
        for q, got, want in (('core', est.core_luts, core),
                             ('encoder', est.encoder_luts, enc),
                             ('dwn_top', est.top_luts, top)):
            err = 100.0 * (got - want) / want
            # Only calibrated-point error is a FAILURE. Holding an extrapolation to the same
            # bar would either force a dishonest fit or a tolerance loose enough to hide real
            # regressions at the point the model actually claims to be accurate.
            if ex:
                worst_extrap = max(worst_extrap, abs(err))
            else:
                worst = max(worst, abs(err))
            print(f'{label:12s} {q:10s} {got:>10.0f} {want:>10} {err:>+8.1f}%'
                  f'{"  ~" if ex else ""}')
    est50 = predict(layers=[50], n=6, z=200, num_classes=5)
    err = 100.0 * (est50.board_luts - MEASURED_BOARD) / MEASURED_BOARD
    worst = max(worst, abs(err))
    print(f'{"1x50":12s} {"board":10s} {est50.board_luts:>10.0f} '
          f'{MEASURED_BOARD:>10} {err:>+8.1f}%')
    print('-' * 55)
    print(f'worst error at the calibrated point (n=6, z=200): {worst:.1f}%')
    if worst_extrap:
        print(f'worst error on ~ extrapolated configs           : {worst_extrap:.1f}%')
    print()
    # 5% is not a hard theoretical bound -- it is the tolerance at which this model is useful
    # for FILTERING, which only has to separate "fits" from "does not fit by 4x".
    if worst > 5.0:
        print('FAIL: the model does not reproduce the config we actually measured.')
        return 1
    print('OK: reproduces the calibrated point within tolerance.')
    if worst_extrap > 5.0:
        print()
        print(f'~ Extrapolated configs are off by up to {worst_extrap:.1f}%, which is EXPECTED and')
        print('  is why dse/grid.py never filters on an extrapolated estimate. The known cause:')
        print('  encoder area saturates at `used_features x z`, not `features x z` -- the')
        print('  learnable mapping ignores some features entirely (Phase 1: d2_b2_mmdt was')
        print('  never read), so fewer bits exist to select than the ceiling assumes.')
    print()
    ratio = MEASURED[0][6 - 1] / MEASURED[0][4]     # encoder / core at 1x50
    print(f'encoder/core ratio: predicted {predict([50], 6, 200, 5).encoder_ratio:.2f}x, '
          f'measured {ratio:.2f}x')
    # No non-ASCII in printed output: the Windows console is cp1252 and mangles it.
    print(f'the superseded dse-plan sec.5 assumption was 3.2x -- filtering with it would have '
          f'underestimated\nthe encoder by {1519/(3.2*108):.1f}x at this config alone.')
    return 0


PAPER_LADDER = [('sm', [50]), ('md', [360]), ('lg', [2400])]


def _ladder() -> int:
    print(f'{"config":6s} {"nodes":>6} {"comps":>7} {"core":>7} {"encoder":>9} '
          f'{"board":>8} {"% dev":>7}  fits?')
    print('-' * 62)
    for name, layers in PAPER_LADDER:
        e = predict(layers, n=6, z=200, num_classes=5)
        sat = '*' if e.comparators >= JSC_FEATURES * 200 else ' '
        print(f'{name:6s} {e.nodes:>6} {e.comparators:>6}{sat} {e.core_luts:>7.0f} '
              f'{e.encoder_luts:>9.0f} {e.board_luts:>8.0f} {e.device_pct:>6.1f}%  '
              f'{"yes" if e.fits() else "NO"}')
    print('-' * 62)
    print('* encoder saturated at features x z = 3200 comparators')
    print()
    print('Caveats before quoting any of this:')
    print('  - comparator counts for md/lg use a 67% selection ratio measured at ONE config')
    print('  - LUT count is necessary, not sufficient: routing can still fail (risk #2)')
    print('  - the reduction term is inferred by subtraction, never synthesized standalone')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='Predict config area without synthesizing.')
    ap.add_argument('--ladder', action='store_true',
                    help="project the paper's sm/md/lg JSC configs")
    args = ap.parse_args()
    if args.ladder:
        return _ladder()
    return _selftest()


if __name__ == '__main__':
    sys.exit(main())
