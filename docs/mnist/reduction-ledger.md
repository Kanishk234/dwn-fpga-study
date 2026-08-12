# MNIST learnable reduction — running ledger

Running log for the learnable-reduction study on MNIST. Plan and ground rules:
`docs/mnist/plan.md`. Bring-up log: `docs/mnist/phase1-ledger.md`.

**Correct entries rather than appending to them.** When a measurement overturns an earlier
conclusion, retract it in place and say what was withdrawn and why. This ledger already contains
one retraction of its own first result; leaving the wrong turn visible is the point.

**Status (2026-08-11): 35 configs trained, and the question is answered.** A learned taper works —
it beats a plain narrow layer by up to **+3.69 pp** at the same group size — and it is **still not
worth building on MNIST**, because every taper measured sits strictly inside the single-layer
frontier. `1x500` reaches **97.70%** with 500 nodes and the identical 500-bit popcount that
`3x[2000,300,500]` spends **2,800** nodes to reach 97.43% with.

The one exception is a *mild* taper: **`2x[2000,1000]` at 98.32% is the best model in the entire
study**, with a 2× smaller adder tree than `1x2000`. Everything at 4:1 compression or worse is
over-compression.

---

## Status

| # | What | Status |
|---|---|---|
| R1 | Baseline grid — 14 configs, four axes | ✅ done 2026-08-11 |
| R2 | Reduction grid — 7 tapers appended to trained baselines | ✅ trained 2026-08-11 |
| R3 | Read the result | ⚠️ **retracted** — confounded by `tau`, see below |
| R4 | Corrected grid at a `tau` that is not 10× hot | ✅ done 2026-08-11 — **confirmed: it was `tau`.** +3.6 to +5.3 pp on every floor-limited config |
| R5 | Gate 1 + synthesis on whichever taper survives | ⬜ **de-prioritised** — no taper is on the projected frontier, so there is nothing whose area is worth measuring first. `2x[2000,1000]` is the only candidate |
| R6 | Measure an MNIST run-to-run noise floor | ⬜ **still nothing here is safely readable without it** |
| R7 | `2x[2000,500]` — the missing monotone taper | ⬜ `3x[2000,300,500]` is an hourglass, not a taper (below) |

---

## Log

### 2026-08-11 — ✅ THE ANSWER: the taper works, and it still does not pay

R4 ran, all 14 configs, ~68 min. Three findings, in the order they were asked.

#### 1. It was `tau`, and the correction scales with how hot it was

| config | group | at τ=3.3333 | at power-law τ | Δ |
|---|---|---|---|---|
| `3x[2000,300,100]` | 10 | 91.28 | **96.62** | **+5.34** |
| `2x[1000,100]` | 10 | 91.02 | **96.20** | **+5.18** |
| `2x[2000,100]` | 10 | 91.46 | **96.32** | **+4.86** |
| `4x[2000,500,200,100]` | 10 | 91.83 | **95.95** | **+4.12** |
| `3x[1000,500,100]` | 10 | 93.05 | **96.67** | **+3.62** |
| `3x[2000,300,200]` | 20 | 96.34 | 96.97 | +0.63 |
| `3x[2000,300,500]` | 50 | 97.56 | 97.43 | **−0.13** |

And B-ladder, independently: `1x100` **+4.61**, `1x200` +2.02, `1x300` +0.63, `1x500` +0.27,
`1x2000` +0.09 (`1x1000` is the anchor). **Two independent families, and in both the gain decays
monotonically to zero exactly as the CE floor does.** The group-50 config, which was never
floor-limited, moved −0.13 pp — i.e. not at all. The 7 pp collapse is fully retracted.

#### 2. JSC's `tau` exponent transfers to MNIST — and killing the CE floor was NOT sufficient

Three points on one architecture, and it is an **interior optimum**, not a monotone:

| | τ=3.333 (range 3.0) | **τ=0.897 (range 11.2)** | τ=0.333 (range 30.0) |
|---|---|---|---|
| `1x100` | 88.37 | **92.98** | 91.36 |
| `2x[2000,100]` | 91.46 | **96.32** | 95.56 |

Power law wins both, by **1.62 and 0.76 pp**. Flat-range has a CE floor of *zero* and still loses,
so the floor was never the whole story: too cold over-confidences the softmax and starves the
gradient from the other side. `width**0.57` — an exponent measured on JSC, a different dataset
with a different class count — lands on the optimum. **That transfer is worth stating; nobody has
checked it.**

#### 3. A taper genuinely works — the mechanism is real

Same group, same `tau`, the comparison the first grid could not make honestly:

| final width | best taper | plain single layer | Δ |
|---|---|---|---|
| 100 | `3x[1000,500,100]` 96.67 | `1x100` 92.98 | **+3.69** |
| 200 | `3x[2000,300,200]` 96.97 | `1x200` 95.93 | **+1.04** |
| 500 | `3x[2000,300,500]` 97.43 | `1x500` 97.70 | **−0.27** |

Large when the floor is tight, gone by 500. Same shape the confounded reading gave, now measured
under correct conditions. Taper **depth** stays irrelevant — 96.32 / 96.62 / 95.95 for 1 / 2 / 3
steps to width 100, and if anything the deepest is worst.

#### ⚠️ 4. And it is dominated anyway

Against the corrected `1x2000` at **98.26%**, the best taper costs **−0.83 pp for 800 extra
nodes**. But the damaging comparison is one no config in the grid was built to make:

> **`1x500` scores 97.70% — better than the best taper — with 500 nodes instead of 2,800 and the
> identical 500-bit popcount.** `1x300` scores 96.77% against `2x[2000,100]`'s 96.32% with 300
> nodes instead of 2,100.

Projecting core area as nodes + adder tree (**~1.3 LUTs/bit**, from the JSC fragment sweep —
approximate, 5-class fragments read onto a 10-class model, and **not yet synthesized**):

| | proj. LUTs | acc | |
|---|---|---|---|
| `1x300` | ~700 | 96.77 | dominates `2x[2000,100]` — ~2,200 / 96.32 |
| `1x500` | ~1,120 | 97.70 | dominates `2x[1000,100]` — ~1,210 / 96.20 |
| `1x1000` | ~2,300 | 97.97 | dominates `3x[2000,300,500]` — ~3,420 / 97.43 |

**The ranking is robust to the encoder**: z=3 over 784 features is identical in every row, so it
shifts all of them by the same constant.

**Conclusion: on MNIST, learnable reduction is dominated — not because the taper fails, but
because MNIST saturates so early on width that the same accuracy is always reachable more cheaply
with a narrower single layer.** The same reason it did not move JSC's headline.

#### The one taper that is NOT dominated, and it was never labelled as one

**`2x[2000,1000]` — 98.32%, the best model in all 35 configs.** It is a 2:1 taper to group 100,
its `tau` of 3.3333 is *exactly* the anchor value, so it was never confounded and its number
stands as trained. Against `1x2000`: **+0.06 pp** (noise) on a **2× smaller adder tree**, ~4,300
projected LUTs against ~4,620. It sat in the `multilayer` group of the baseline grid and was never
read as a reduction result.

**So the viable regime is a mild taper.** 2:1 wins; 4:1 costs 0.83 pp; 20:1 costs 1.94 pp.

#### ⚠️ A grid design error, recorded because it bounds the result

`3x[2000,300,500]` is **2000 → 300 → 500**: it *widens* at the end. That is an hourglass, not a
taper, and every bit passes a 300-wide bottleneck. It is the only non-monotone config in the grid
and still the best of the seven, so its 97.43% is a **lower bound** on what a 500-wide floor can
do. `2x[2000,500]` is the missing point (R7) and would plausibly beat it — which would narrow, but
on these margins not close, the gap to `1x1000`.

### 2026-08-11 — ⚠️ RETRACTED: "tapers lose 7 pp" measured `tau`, not the taper

**Confirmed by R4 above**, which is the entry to read; this one is kept for the reasoning.

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

⚠️ **Superseded 2026-08-11** — every rung except `1x1000` was `tau`-confounded. Corrected: 92.98 /
95.93 / 96.77 / 97.70 / 97.97 / 98.26. **The knee stays at 500** (300→500 is +0.93 pp, 500→1000 is
+0.27), but the corrected curve is much flatter at the bottom, so the *cost* of the knee changed:
500 → 2000 is now 4× the core for **+0.56 pp**.

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

All accuracies at the corrected `tau` unless marked. **The corrected ladder** — the frontier every
taper is judged against:

| width | 100 | 200 | 300 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| acc | 92.98 | 95.93 | 96.77 | **97.70** | 97.97 | **98.26** |
| (was) | 88.37 | 93.91 | 96.14 | 97.43 | anchor | 98.17 |

| | |
|---|---|
| **best model in the study** | **`2x[2000,1000]` — 98.32%**, a 2:1 taper, group 100, **2× smaller adder tree** than `1x2000` |
| **best explicit taper** | `3x[2000,300,500]` — 97.43%, **−0.83 pp vs `1x2000`** for +800 nodes |
| **the domination** | `1x500` **97.70% / 500 nodes** beats it, at the **same 500-bit popcount** |
| **taper vs plain narrow layer** | **+3.69 pp @ 100**, +1.04 @ 200, −0.27 @ 500 — the mechanism is real, and it fades |
| **taper depth** | 96.32 / 96.62 / 95.95 for 1 / 2 / 3 steps to width 100 — **no effect**, deepest is worst |
| **the `tau` correction** | **+3.6 to +5.3 pp** on every group-10 config; +0.09 pp at group 200 |
| **the schedule is an interior optimum** | `1x100`: 88.37 (τ=3.333) → **92.98** (τ=0.897) → 91.36 (τ=0.333) |
| **JSC's exponent transfers** | `tau ∝ width**0.57` beats flat-range by 1.62 and 0.76 pp on MNIST |
| **the confound** | group 10 at τ=3.3333 ⇒ logit range 3.0 ⇒ **CE floor 0.3702**; observed loss 0.57–0.63 |
| **z is nearly free** | z=1 97.91% vs z=25 98.23% — **0.32 pp across a 25× encoder** |
| **depth does not pay** | `2x[1000,500]` 97.93 vs `1x1000` 97.97 — 50% more nodes, nothing |
| MNIST tau anchor | `tau = 1/0.3` at final width 1000, group 100, range 30 (upstream `examples/mnist.py`) |
| JSC's *anchors* do not transfer, its *exponent* does | `tau_for(1000)` = 19.12 against MNIST's 3.33 at the same width |

---

## Open questions

| | |
|---|---|
| ~~Does a trained taper hold accuracy at a correct `tau`?~~ | ✅ **Closed 2026-08-11: yes, and it still does not pay.** Group-10 tapers gained +3.6 to +5.3 pp, confirming the confound. They then lost to a *narrower single layer* at the same popcount width. See the R4 entry. |
| ~~Which `tau` schedule does MNIST follow?~~ | ✅ **Closed 2026-08-11: the power law**, by 1.62 and 0.76 pp on two architectures. And it is an interior optimum — flat-range has **zero** CE floor and still loses, so removing the floor was never sufficient. |
| ~~Is the low end of the ladder wrong as published?~~ | ✅ **Closed 2026-08-11: yes, and now corrected.** `1x100` +4.61, `1x200` +2.02, `1x300` **+0.63 → 96.77%**. Gate 1, area and timing are all unaffected — `tau` never reaches hardware — but every accuracy in the baseline grid below width 1000 is superseded. |
| **No MNIST noise floor exists** ⚠️ | JSC measured 0.15 pp by training one config twice. **Nothing here has an equivalent**, and it is now the binding limit: `2x[2000,1000]`'s +0.06 pp over `1x2000`, the −0.13 and −0.27 pp taper deltas, and `1x2000`'s +0.09 pp `tau` gain are all **unresolved, not small**. One config under a second seed buys it, and it is the cheapest outstanding item in the study. |
| **The area projection is a projection** ⚠️ | ~1.3 LUTs/bit comes from JSC's **5-class** fragment sweep read onto a **10-class** model, and nothing here has been synthesized. The domination margins (3× on `1x300` vs `2x[2000,100]`) are far too wide for that to flip, but the exact frontier is not measured. |
| **Is `2x[2000,1000]` worth building?** | It is the only taper on the projected frontier — best accuracy in the study, 2× smaller adder tree. But it is 3,000 nodes against `1x300`'s 300 for +1.55 pp, so whether it fits the board at all is the first question, not whether it is Pareto-optimal. R5. |
| **`2x[2000,500]` is missing** | `3x[2000,300,500]` widens at the end — an hourglass through a 300-wide bottleneck — so its 97.43% is a **lower bound** on a 500-wide floor. R7. Cheap: one config, ~8 min. |
| **Is a `LUTLayer` taper "Learnable Reduction as the paper means it"?** | Functionally it replaces the popcount. That is not the same as being the same construction, and the claim needs the paper re-read before it goes in a writeup. |
| **`z=1` is untested in hardware** | 0.32 pp for a 25× encoder reduction is the biggest area lever in the grid and it has nothing to do with the reduction study. It should not wait behind it. |

---

## Pointers

- `training/mnist_grid_kaggle.ipynb` — the 14 baselines
- `training/mnist_reduction_kaggle.ipynb` — the 7 tapers, ⚠️ trained at the confounded `tau`
- `training/mnist_reduction_tau_kaggle.ipynb` — the corrected grid (R4), ✅ trained; these are the
  checkpoints to export, and the `_tau*` slug suffix keeps them beside the confounded originals
- `third_party/DWN/src/torch_dwn/utils.py` — `GroupSum`, where `/ tau` lives
- `third_party/DWN/src/torch_dwn/mapping.py` — `learnable` vs `random`, and the other `tau`
- `dse/grid.py: tau_for` — JSC's four anchors and the 0.57 exponent
- `docs/datapath.md` — what the reduction stage does, and why it is the critical path
- `docs/mnist/phase1-ledger.md` — the bring-up log, including the `1x300` accuracy this may move
