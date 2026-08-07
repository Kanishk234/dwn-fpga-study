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

# One DWN node is one LUT6 (brief §4). This is the architectural premise of the whole project,
# and the 50-node core is consistent with it.
LUT_PER_NODE = 1.0

# Reduction, split to match the single measured point (58 LUTs at final_width=50, classes=5).
# A W-bit popcount is an adder tree costing ~1 LUT per input bit; the argmax is a small
# comparison tree over num_classes scores. 50 x 1.0 + 5 x 1.6 = 58.
LUT_PER_FINAL_BIT = 1.0
LUT_PER_CLASS_ARGMAX = 1.6

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
    reduction = layers[-1] * LUT_PER_FINAL_BIT + num_classes * LUT_PER_CLASS_ARGMAX
    core = nodes * LUT_PER_NODE + reduction

    return AreaEstimate(nodes=nodes, comparators=comparators,
                        core_luts=core, encoder_luts=encoder)


# ---------------------------------------------------------------------------------------------
# Self-test: reproduce the Phase 1 measurements. A model that cannot recover the one config we
# actually synthesized has no business filtering the other forty.
# ---------------------------------------------------------------------------------------------

MEASURED = {'comparators': 202, 'core': 108, 'encoder': 1519, 'top': 1619, 'board': 2058}


def _selftest() -> int:
    est = predict(layers=[50], n=6, z=200, num_classes=5)
    rows = [
        ('comparators', est.comparators, MEASURED['comparators']),
        ('core LUTs', est.core_luts, MEASURED['core']),
        ('encoder LUTs', est.encoder_luts, MEASURED['encoder']),
        ('dwn_top LUTs', est.top_luts, MEASURED['top']),
        ('board LUTs', est.board_luts, MEASURED['board']),
    ]
    print(f'{"quantity":16s} {"predicted":>10} {"measured":>10} {"error":>9}')
    print('-' * 49)
    worst = 0.0
    for label, got, want in rows:
        err = 100.0 * (got - want) / want
        worst = max(worst, abs(err))
        print(f'{label:16s} {got:>10.0f} {want:>10} {err:>+8.1f}%')
    print('-' * 49)
    print(f'worst error: {worst:.1f}%')
    print()
    # 5% is not a hard theoretical bound -- it is the tolerance at which this model is useful
    # for FILTERING, which only has to separate "fits" from "does not fit by 4x".
    if worst > 5.0:
        print('FAIL: the model does not reproduce the config we actually measured.')
        return 1
    print('OK: reproduces Phase 1 within tolerance.')
    print()
    print(f'encoder/core ratio: predicted {predict([50], 6, 200, 5).encoder_ratio:.2f}x, '
          f'measured {MEASURED["encoder"]/MEASURED["core"]:.2f}x')
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
