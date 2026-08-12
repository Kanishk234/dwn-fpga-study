# MNIST Phase 2 ledger — design-space exploration on the second dataset

Running log for the MNIST DSE. Phase 1 is complete and written up in
`docs/mnist/phase1-report.md`; its running log is `docs/mnist/phase1-ledger.md`.

**Branch:** `mnist`. **Written to be picked up on a different machine and in a different session
than the one that finished Phase 1** — §1 is the handoff, and none of it should need reconstructing
from chat history.

**Correct entries rather than appending to them.** When a measurement overturns an earlier
conclusion, strike the old one through and say what was retracted and why. Phase 1 withdrew five
conclusions this way and every one was caught by measuring at a second point.

---

## Status

| Step | What | Status |
|---|---|---|
| M2a | Machine parity — prove this box reproduces Phase 1 | ⬜ **do this first, before anything else** |
| M2·0 | **Noise floor** — run `mnist_noise_floor_kaggle.ipynb` | 🟡 notebook built, needs a Kaggle GPU session. Not strictly blocking, but it decides whether several reduction-study findings survive |
| M2b | Get the checkpoints onto the machine — **the `_tau*` set, see §1.3** | ⬜ trained on Kaggle, **not present locally** |
| M2c | Make `dse/` dataset-aware (it is JSC-shaped today) | ⬜ blocking; see §3 |
| M2d | Fix the two known-wrong constants in `dse/area_model.py` | ⬜ blocking for any *prediction*, not for measurement |
| M2e | Gate 1 across the grid | ⬜ |
| M2f | Synthesis + place-and-route sweep | ⬜ the long pole |
| M2g | Report, frontier, snapshot | ⬜ |

---

## 1. Handoff — what a fresh machine needs

### 1.1 Prove parity before trusting a single sweep point

```
.venv\Scripts\python.exe scripts\verify_phase1.py            # 12/12, no board
.venv\Scripts\python.exe scripts\verify_phase1.py --with-board   # 22/22
```

⚠️ **The expected values changed during Phase 1 and older documents quote the old ones.** The
current, correct set is:

| | value | note |
|---|---|---|
| `dwn_core` | **110** LUTs, 73 FF | was 108 before the argmax tree |
| `thermometer_encoder` | **1,519** LUTs, 0 FF | unchanged |
| `dwn_top` | **1,621** LUTs, 269 FF | was 1,619 |
| whole board design | **1,893** LUTs, 864 FF, 8 BRAM | was 2,058, then 2,060 |
| Gate 1b | **166,000 / 166,000** | |

`REPORT.md` and `README.md` quote the **`jsc-complete` tag's** figures (108 / 1,619 / 2,058) and
say so. Both are right; they describe different commits. Do not mix them in one table.

**If parity fails, stop.** Past that point an MNIST result cannot be attributed — a difference
could be the sweep, the toolchain, or the machine, and there is no way to tell which.

### 1.2 The checkpoints are not in the repo

Everything is trained — 14 baselines plus the 35-config reduction/`tau` study — but **only
`mnist_n6_z3_distributive_w300` is on the machine that finished Phase 1.** Checkpoints are large
and deliberately not in git (one JSC file exceeds GitHub's 100 MB limit); `docs/results-mnist/`
describes them in ~10 KB instead.

Download `_checkpoint.pt` + `_testvectors.npz` pairs into `training/artifacts/` from **three**
Kaggle Outputs, and mind which:

| notebook | what | take it? |
|---|---|---|
| `mnist_reduction_tau_kaggle.ipynb` | the corrected-`tau` grid | ✅ **yes — these are the ones to sweep** |
| `mnist_grid_kaggle.ipynb` | the 14 baselines | ⚠️ only `1x1000` (the anchor) and the z/n sweeps, which are all `1x1000` and unaffected |
| `mnist_reduction_kaggle.ipynb` | the 7 tapers | ❌ trained at the confounded `tau`; superseded |

`dse/run.py --list` will then say which configs have a checkpoint and which do not, and refuses to
synthesize one that is missing.

### 1.3 The grid — and ⚠️ **do not sweep the baseline checkpoints below width 1000**

| group | configs |
|---|---|
| ladder | `1x100` `1x200` `1x300` `1x500` `1x1000` `1x2000`, all n=6 z=3 |
| z-sweep | `1x1000` at z = 1, 2, 3, 8, 25 |
| multilayer | `2x[2000,1000]` (upstream's own), `2x[1000,500]` (the paper's) |
| n-sweep | `1x1000` at n = 2 and n = 4 |

Source of truth: `training/mnist_grid_kaggle.ipynb`, cell 4. The multilayer configs use `random`
mapping on the second layer because that is `LUTLayer`'s default and therefore what upstream
trains; it also exercises the fixed-wiring export path, which the `300-100` JSC checkpoint is the
only other coverage for.

⚠️ **Every ladder rung except `1x1000` was trained at a confounded `tau` and its accuracy is
superseded.** The grid froze `tau = 1/0.3` for all 14 configs, but `GroupSum` divides by `tau`, so
with `tau` fixed the **logit range is set by group size** — which is the axis the ladder sweeps. At
`1x100` (group 10) the range is 3.0, giving a cross-entropy floor of 0.3702 that the model cannot
train below. `1x1000` is the anchor (group 100, range 30) and was never affected.

| width | 100 | 200 | 300 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| **corrected** | **92.98** | **95.93** | **96.77** | **97.70** | 97.97 | **98.26** |
| ~~as first published~~ | 88.37 | 93.91 | 96.14 | 97.43 | anchor | 98.17 |

**The checkpoints to export are `training/mnist_reduction_tau_kaggle.ipynb`'s**, whose `_tau*` slug
suffix keeps them beside the confounded originals. Full analysis: `docs/mnist/reduction-ledger.md`.

**Area, timing and Gate 1 are entirely unaffected** — `tau` is a training-time constant that never
reaches hardware. Only the accuracy column moves. But it moves the *shape* of the ladder, not just
its offset: the corrected curve is far flatter at the bottom, so **500 → 2000 is now 4× the core
for +0.56 pp**. The knee stays at 500.

### 1.4 Board runs need `--depth`

```
.venv\Scripts\python.exe scripts\host.py --gate1b --depth 16 --checkpoint <ckpt>
```

`host.py` defaults to JSC's 1024. MNIST's vector store holds **16** records at the default
`--bram-budget 0.15`, because a 12,544-bit-wide store cannot use block RAM at all (a BRAM36 is at
most 72 bits wide; one cycle of that width would need ~175 of the 50 on the device). Loading more
than the store holds wraps silently and every result after the wrap is wrong. `build_bitstream.py`
prints the `DEPTH` it derived — read it from there rather than assuming 16 carries across configs,
since it falls further as `z` and layer width grow.

---

## 2. Budget, and why it is bigger than JSC's

The project brief's 2–3 week Phase 2 estimate assumed two machines running Vivado in parallel.
**This project runs on one machine, both people present** (`CLAUDE.md`), so roughly double that,
and the sweep should be sized accordingly.

MNIST is additionally slower per point than JSC was:

- **The designs are larger.** `1x300` is 1,548 LUTs against JSC `1x50`'s 1,621 — comparable — but
  the ladder goes to `1x2000` and the multilayer configs to 3,000 nodes.
- **Place-and-route dominates.** Use `--impl` for anything quotable: Phase 1 measured 161.0 MHz
  post-synthesis against 147.1 post-route on the same JSC design, and post-synthesis timing is
  systematically optimistic.
- **A full board pass is 31.9 s**, versus 11.2 s for JSC, and needs 625 batches at `DEPTH=16`.
  Only configs that get a Gate 1b run pay this.

`dse/run.py` is resumable and records configs that fail to build rather than dropping them. **A
config that cannot route is a data point marking the frontier's edge** (brief §12 risk #2), not a
mistake to prevent.

---

## 3. `dse/` is JSC-shaped and must be generalised first

This is the largest known work item and it is not optional — `dse/` does not import `datasets` at
all, so every dataset fact in it is a private copy of JSC's.

| file | what is JSC-specific |
|---|---|
| `grid.py` | `NUM_CLASSES`, the size ladder, the `tau` power law fitted to four JSC anchors, slug construction |
| `area_model.py` | `JSC_FEATURES = 16`, `LUT_PER_COMPARATOR_BIT = 1519/(202*16)`, `num_classes=5` defaults |
| `run.py`, `report.py`, `plot.py` | inherit the above through `Config` |

**Note that `dse/` is NOT in the dataset-agnostic list in `CLAUDE.md`** (`exporter/`, `rtlgen/`,
`rtl/`, `tb/`, `scripts/`, `harness/`). That was reasonable when it was JSC-only sweep automation.
It is no longer, and the honest options are:

1. **Bring `dse/` under the contract** — it reads `datasets.identify(ck)` like everything else, and
   the §1.5 promise ("a third dataset means a descriptor and a docs directory, nothing else")
   becomes true of the sweep too. More work now.
2. **Keep `dse/` JSC-bound and add a parallel MNIST grid** — faster, and repeats exactly the
   mistake Phase 1 spent its whole budget undoing.

⚠️ **Decide this deliberately and record the decision here.** Option 2 is how `datasets/` came to
exist and be imported by nothing.

### 3.1 Two constants in `area_model.py` are known wrong

Both were measured on JSC and both were falsified by MNIST. **Neither affects a measurement — only
a prediction** — so the sweep can proceed before they are fixed, but no projected area should be
quoted until they are.

- **`harness = 2058 - 1619 = 439`, treated as fixed.** It is 3,038 LUTs for MNIST and scales with
  record width. Any "model + harness" projection using 439 is wrong for any dataset but JSC.
- **`argmax_luts() = (num_classes - 1) * score_w / 2`** is described as "K−1 comparisons", which
  was written for the *linear chain* that `rtl/argmax.v` no longer is. ⚠️ **This one is milder
  than it looks and the first draft of this section got it backwards.** A balanced tree over K
  leaves has K−1 internal merge nodes too, so the *count* is unchanged; what the tree changes is
  **depth** (`ceil(log2 K)` instead of K−1), which is a timing property this model does not
  predict at all. The measured area delta was **+2 LUTs at K=5**, i.e. the formula very slightly
  *under*-estimates. Fix the comment, and do not expect a large area correction from it.

A third is suspect rather than wrong: `LUT_PER_COMPARATOR_BIT = 1519/(202*16)` is a JSC fit, and
Phase 1 measured 1.27 LUTs/comparator on real MNIST against 1.50 on synthetic MNIST data. The
mechanism is unexplained — an amortisation hypothesis was proposed and falsified by a third point.
The current, **untested** hypothesis is that threshold *values* matter: real MNIST pixels are
mostly zero, so quantile thresholds cluster near zero and a comparison against a near-zero constant
collapses to a test on a couple of bits. Cheap to test — emit the same model at several `z` and
compare LUTs-per-comparator against the threshold distribution. **Nothing should rest on either
explanation until someone does.**

---

## 4. Sweep priorities — the training study already ranked them

`docs/mnist/reduction-ledger.md` measured accuracy across 35 configurations. **Every area number
in it is a projection and none of it has been synthesized** — that is exactly what Phase 2 is for.
Ordered by how much a synthesis result would settle:

1. **`z=1` — the biggest area lever in the study, and it has nothing to do with anything else.**
   z=1 scores 97.91% against z=25's 98.23%: **0.32 pp across a 25× encoder.** For a design whose
   encoder is the only arithmetic in the datapath — and which was 94% of JSC's `1x50` — this is a
   very large saving for a third of a point. All z configs are `1x1000`, group 100, the anchor, so
   **none is `tau`-confounded**. Not blocked by the `dse/` refactor; synthesize it early.
2. **`2x[2000,1000]`, the best model at 98.32%** — does 3,000 nodes fit at all? See §5.
3. **The corrected ladder, `1x100` … `1x2000`** — the frontier every other result is judged
   against, and the only way to turn the projected Pareto curve into a measured one.
4. **`n=4`** — 97.57% against n=6's 97.97%, i.e. **99.6% of the accuracy on a quarter-size table.**
   Whether that is cheaper on an Artix-7 depends on how Vivado packs sub-LUT6 functions, which is a
   pure synthesis question and one `scripts/probe/` already knows how to ask.
5. **Tapers** — measured, real, and they lose: `1x500` at 97.70% beats the best explicit taper at
   the same popcount width. Worth confirming in area, low priority for accuracy.

⚠️ **The domination claims rest on ~1.3 LUTs/bit**, read from JSC's **5-class** fragment sweep onto
a **10-class** model. The margins are wide (3× on `1x300` vs `2x[2000,100]`) so they are unlikely
to flip, but "unlikely to flip" is what Phase 2 replaces with a measurement.

## 5. What Phase 1 says to expect

Predictions, recorded now so they can be scored later rather than reconstructed favourably.

- **Timing will bind before area.** `1x300` uses 7.44% of the device and closes 100 MHz with
  0.742 ns, but the whole board design has only **0.292 ns — 2.9% margin**. The ladder goes to
  2,000 nodes. Expect the large configs to miss the clock while still fitting comfortably, which
  is the opposite of the JSC frontier's shape.
- **Pipeline depth is the lever**, and it needs no retraining — the harness already parameterises
  it. Sweeping it is a Group B axis, cheap relative to Group A.
- **`z` is cheap here and word width is not.** MNIST is slot-limited: 784 × z far exceeds the input
  slots of any layer that fits, so z=8 → z=200 is only 2.3× the comparators. At 16 bits the
  paper's configuration is over the device at every `z`; at 11 bits it fits comfortably. **This
  inverts the JSC conclusion**, where `z` dominated area.
- **Q0.8 is exactly lossless at `1x300`** — 0 divergence from float32 on all 10,000 samples,
  because 8-bit pixels have only 256 distinct values. Expect this to hold across the grid, but
  **check it per config**: on JSC the safe word width moved between two configurations of the same
  dataset, so it cannot be assumed to transfer even within MNIST.

---

## 6. Open questions

| Question | Status |
|---|---|
| ~~Does the `tau` correction change the frontier?~~ | ✅ **Closed 2026-08-11, before Phase 2 started** — and yes, it changes the ladder's *shape*, not just its offset. Corrected grid trained (35 configs), analysis in `docs/mnist/reduction-ledger.md`. **Consequence for Phase 2: sweep the `_tau*` checkpoints, not the baselines.** §1.3 |
| Does `dse/` come under the dataset-agnostic contract? | ⬜ **Open, and blocking M2c.** See §3 |
| **No MNIST noise floor exists** ⚠️ | 🟡 **Notebook built, not yet run** — `training/mnist_noise_floor_kaggle.ipynb`, 16 runs (4 configs × 4 seeds), ~2 h on a Kaggle GPU. Until it lands, `2x[2000,1000]`'s +0.06 pp over `1x2000` and the −0.13 / −0.27 pp taper deltas are **unresolved, not small**. Deliberately four *widths*, since JSC took its 0.15 pp from one config and applied it everywhere |
| Does the paper's `2x[1000,500]` fit at any precision? | ⬜ Open. Phase 1 projected yes at 11-bit (38.0% of device), never built. ⚠️ Also **97.93% against `1x1000`'s 97.97** — 50% more nodes for nothing, so it is a fit question, not a frontier one |
| Does `2x[2000,1000]` fit? | ⬜ Open, and **the most interesting single build in Phase 2.** Best model in the study at **98.32%**, the only taper on the projected frontier, with a 2× smaller adder tree than `1x2000` — but 3,000 nodes against `1x300`'s 300, projected ~4,620 LUTs. Whether it fits at all is the first question |
| Why is LUTs-per-comparator lower on real data than synthetic? | ⬜ Open, hypothesis untested. §3.1 |
| Is 5 Mbaud a ceiling for a 1,569-byte record? | ⬜ Open, deliberately not measured. Phase 1 §6 — the I/O wall is 243,271×, so a faster link changes no reported number. Revisit only if a config makes a full pass painful |
| Does a corrected-`tau` model reproduce on silicon? | ⬜ Open. The board currently holds the **confounded** `1x300` (96.14%). Gate 1b's bit-exactness is unaffected by `tau`, but the hardware demo is not the best available model |

---

## Log

*(empty — Phase 2 has not started. Add a dated entry per work session, newest first.)*
