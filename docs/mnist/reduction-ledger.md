# MNIST learnable reduction — running ledger

Running log for the learnable-reduction study on MNIST. Plan and ground rules:
`docs/mnist/plan.md`. Bring-up log: `docs/mnist/phase1-ledger.md`.

**Correct entries rather than appending to them.** When a measurement overturns an earlier
conclusion, retract it in place and say what was withdrawn and why. This ledger already contains
one retraction of its own first result; leaving the wrong turn visible is the point.

**Status (2026-08-11): 21 configs trained, and the reduction question is still open.** The first
reduction grid produced a clean-looking answer — tapers lose up to 7 pp — and that answer is
**retracted**. It measured `tau`, not the taper. A corrected grid is written and not yet run.

---

## Status

| # | What | Status |
|---|---|---|
| R1 | Baseline grid — 14 configs, four axes | ✅ done 2026-08-11 |
| R2 | Reduction grid — 7 tapers appended to trained baselines | ✅ trained 2026-08-11 |
| R3 | Read the result | ⚠️ **retracted** — confounded by `tau`, see below |
| R4 | Corrected grid at a `tau` that is not 10× hot | ⬜ **written, not run** — `training/mnist_reduction_tau_kaggle.ipynb`, 14 configs, ~70 min |
| R5 | Gate 1 + synthesis on whichever taper survives | ⬜ blocked on R4 |
| R6 | Measure an MNIST run-to-run noise floor | ⬜ **nothing here is safely readable without it** |

---

## Log

### 2026-08-11 — ⚠️ RETRACTED: "tapers lose 7 pp" measured `tau`, not the taper

`training/mnist_reduction_kaggle.ipynb` trained all 7 tapers. Read against `1x2000` (98.17%),
every config ending at 100 nodes collapses to ~91%. That looked like a verdict on learnable
reduction. It is not one.

`GroupSum.forward` is `x.sum(dim=-1) / self.tau`. With `tau` held constant the **logit range is
set by the group size** — and the group size is exactly the axis the grid sweeps. Tapering 2000 →
100 does not only shrink the popcount; it shrinks the logit range from 60 to 3 and puts a hard
floor under cross-entropy that no amount of learning can pass.

| final width | group | logit range at τ=3.3333 | CE floor | observed final loss |
|---|---|---|---|---|
| 100 | 10 | 3.0 | **0.3702** | 0.5716 – 0.6306 |
| 200 | 20 | 6.0 | 0.0221 | 0.1219 |
| 500 | 50 | 15.0 | ~0 | 0.0242 |
| 2000 | 200 | 60.0 | ~0 | 0.0008 |

The floor is not cosmetic. At group 10 a *correctly* classified example still returns a gradient
of magnitude ≥ 0.31 for all 30 epochs, so easy samples compete for the update budget with hard
ones and the optimizer never concentrates.

**The proof that this is group size and not tapering: `1x100` shows the identical signature** —
group 10, loss stuck at 0.7945, 88.37%, and no taper anywhere in it. The whole low end of the
baseline ladder is confounded the same way, which means the *baselines* are not usable as
baselines either.

#### What survives the retraction

Three things, because they compare configs at the same group size:

1. **Taper depth is irrelevant.** One, two and three steps to the same endpoint give 91.46 /
   91.28 / 91.83 — a 0.55 pp spread. Closed: if a taper works, use the shallowest one.
2. **A taper beats a plain narrow layer**, and this is the encouraging result that got buried:
   **+3.09 pp** at width 100, **+2.43 pp** at 200, **+0.13 pp** at 500. The taper preserves
   information a single narrow layer discards — the mechanism the paper claims, showing up.
3. **`3x[2000, 300, 500]` at 97.56% is trustworthy**, because at group 50 it was never
   floor-limited. That is **−0.61 pp against `1x2000` with a 4× smaller adder tree**, and it is
   the one row worth putting through synthesis today.

The sharpest single comparison in the set is capacity-matched: `4x[2000,500,200,100]` and
`3x[2000,300,500]` are **both 2800 nodes**, 91.83% against 97.56%. Same nodes, 5.7 pp apart, and
the only difference is where the reduction floor sits.

#### The methodological error, stated plainly

The grid froze `tau` **specifically** to avoid confounding — the JSC study had already lost a run
to a `tau` error that read as an architectural finding, and the reduction notebook says so in its
own header. Freezing it was the wrong correction:

> **When a hyperparameter's correct value is a function of the swept axis, holding it constant is
> not a control. It is a systematic bias against one end of the sweep.**

The control is to scale it by the known relationship, which is what R4 does.

### 2026-08-11 — The corrected grid, and why it measures the schedule instead of assuming one

`training/mnist_reduction_tau_kaggle.ipynb`. 14 configs, ~70 min on a Kaggle GPU. Every config
already exists as a trained checkpoint at τ=3.3333, so **every row is a paired comparison against
a number already measured**; only `tau` differs. Slugs carry the `tau` value, so nothing is
overwritten.

**MNIST gives exactly one anchor** — upstream's `examples/mnist.py` uses `tau = 1/0.3` on a
1000-wide final layer, group 100, range 30 — and one anchor cannot fix an exponent. JSC's schedule
does not transfer: `dse/grid.py: tau_for` indexed at 1000 nodes gives **19.12**, against
upstream's **3.33** for MNIST at the same width. So both defensible readings get measured:

| schedule | rule | at group 10 | logit range |
|---|---|---|---|
| **power law** | `tau ∝ width**0.57`, JSC's measured log-log slope | 0.897 | 11.15 |
| **flat range** | `tau ∝ width`, holding the range at the anchor's 30 | 0.333 | 30.00 |

Both reproduce 3.3333 at width 1000 exactly, by construction. Both kill the CE floor: 0.0001 and
0.0000 against 0.3702.

| group | configs | what it re-runs |
|---|---|---|
| **A-reduction** | 7 | the whole reduction grid at power-law `tau` |
| **B-ladder** | 5 | `1x100/200/300/500/2000` at power-law `tau` — **the baselines were confounded too.** `1x1000` is the anchor and is deliberately absent |
| **C-schedule** | 2 | `1x100` and `2x[2000,100]` at flat-range `tau`, giving **three tau points on one architecture** |

Nothing else moves: same seed, epochs, LR schedule, binarization, `train_one`.

⚠️ **B-ladder is the part that is easy to skip and must not be.** A taper at group 10 has to be
read against a single layer at group 10 *at the same tau*. Without B, A is still being compared
against numbers trained under the bug.

### 2026-08-11 — Baseline grid: 14 configs, four axes

`training/mnist_grid_kaggle.ipynb`. One-factor-at-a-time star centred on `1x1000` n=6 z=3.
Mapping is not an independent axis — single-layer configs are `learnable`, the two multilayer ones
are `['learnable', 'random']`, matching upstream's `examples/mnist.py` and the paper so the
numbers stay comparable.

**The ladder knees at 500.**

| width | 100 | 200 | 300 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| acc | 88.37 | 93.91 | 96.14 | 97.43 | 97.97 | 98.17 |
| Δ | — | +5.54 | +2.23 | +1.29 | +0.54 | +0.20 |

500 → 2000 is 4× the core for +0.74 pp. ⚠️ The low rungs are `tau`-confounded (above), so the
knee's *position* is provisional until R4 lands.

**z barely matters, and this is the largest practical result in the set.**

| z | 1 | 2 | 3 | 8 | 25 |
|---|---|---|---|---|---|
| acc | 97.91 | 98.05 | 97.97 | 98.13 | 98.23 |

**0.32 pp across a 25× encoder.** For a design whose encoder is the only arithmetic in the
datapath — and which was 94% of JSC's `1x50` — z=1 is a very large area saving for a third of a
point. This one is not `tau`-confounded: every z config is `1x1000`, group 100, the anchor.
**Worth synthesizing independently of the reduction work.**

**n is monotone; n=4 is the interesting point.** 96.30 / 97.57 / 97.97 for n=2/4/6 — n=4 reaches
99.6% of n=6's accuracy on a quarter-size table. Whether that is cheaper on an Artix-7 depends on
how Vivado packs sub-LUT6 functions, which is a synthesis question the probe already knows how to
ask.

**Depth does not pay.** `2x[1000,500]` 97.93 against `1x1000` 97.97 — 50% more nodes for nothing.
`2x[2000,1000]` 98.32 against `1x2000` 98.17, +0.15 pp, at JSC's noise floor.

---

## Numbers worth quoting

| | |
|---|---|
| **best trustworthy taper** | `3x[2000, 300, 500]` — **97.56%**, −0.61 pp vs `1x2000`, **4× smaller adder tree** |
| **capacity-matched pair** | 2800 nodes each: `4x[...100]` 91.83% vs `3x[...500]` 97.56% — **5.7 pp on the floor alone** |
| **taper vs plain narrow layer** | +3.09 pp @ 100, +2.43 pp @ 200, +0.13 pp @ 500 (all at the old `tau`) |
| **taper depth** | 91.46 / 91.28 / 91.83 for 1 / 2 / 3 steps to width 100 — **no effect** |
| **the confound** | group 10 at τ=3.3333 ⇒ logit range 3.0 ⇒ **CE floor 0.3702**; observed loss 0.57–0.63 |
| **z is nearly free** | z=1 97.91% vs z=25 98.23% — **0.32 pp across a 25× encoder** |
| **ladder knee** | 500 nodes; 500 → 2000 is 4× the core for +0.74 pp |
| **depth does not pay** | `2x[1000,500]` 97.93 vs `1x1000` 97.97 — 50% more nodes, nothing |
| MNIST tau anchor | `tau = 1/0.3` at final width 1000, group 100, range 30 (upstream `examples/mnist.py`) |
| JSC's schedule does not transfer | `tau_for(1000)` = 19.12 against MNIST's 3.33 at the same width |

---

## Open questions

| | |
|---|---|
| **Does a trained taper hold accuracy at a correct `tau`?** ⚠️ | **The question everything waits on.** R4 answers it. If the group-10 tapers move up toward the ladder, the first grid measured the schedule. If they don't, a 10-bit floor genuinely cannot carry MNIST — also a clean result, and stronger for having been tested rather than assumed. |
| **No MNIST noise floor exists** ⚠️ | JSC measured 0.15 pp by training one config twice. **Nothing here has an equivalent**, so every gap under ~0.3 pp in this ledger is unresolved rather than small — including the +0.13 pp at width 500 and the +0.15 pp for `2x[2000,1000]`. One extra config under a second seed buys it. |
| **Which `tau` schedule does MNIST follow?** | Power law and flat range differ 2.7× at group 10. C-schedule measures it. If the JSC exponent transfers across datasets that is worth stating out loud — nobody has checked. |
| **Is the low end of the ladder wrong as published?** ⚠️ | `1x100` at 88.37% and `1x200` at 93.91% were trained 10× and 5× hot. `docs/mnist/phase1-ledger.md` quotes **`1x300` at 96.14%** as the bring-up model, and that rung ran ~2× hot too. The Gate 1 and area results are unaffected — `tau` never reaches hardware — but the **accuracy number attached to the bring-up design may rise.** B-ladder settles it. |
| **Area is unmeasured for every taper** | The entire reason to want one is that the popcount was 35% of the JSC core and on the critical path. Nothing in this ledger is a LUT count. R5. |
| **Is a `LUTLayer` taper "Learnable Reduction as the paper means it"?** | Functionally it replaces the popcount. That is not the same as being the same construction, and the claim needs the paper re-read before it goes in a writeup. |
| **`z=1` is untested in hardware** | 0.32 pp for a 25× encoder reduction is the biggest area lever in the grid and it has nothing to do with the reduction study. It should not wait behind it. |

---

## Pointers

- `training/mnist_grid_kaggle.ipynb` — the 14 baselines
- `training/mnist_reduction_kaggle.ipynb` — the 7 tapers, ⚠️ trained at the confounded `tau`
- `training/mnist_reduction_tau_kaggle.ipynb` — the corrected grid (R4)
- `third_party/DWN/src/torch_dwn/utils.py` — `GroupSum`, where `/ tau` lives
- `third_party/DWN/src/torch_dwn/mapping.py` — `learnable` vs `random`, and the other `tau`
- `dse/grid.py: tau_for` — JSC's four anchors and the 0.57 exponent
- `docs/datapath.md` — what the reduction stage does, and why it is the critical path
- `docs/mnist/phase1-ledger.md` — the bring-up log, including the `1x300` accuracy this may move
