# MNIST Phase 2 ledger — design-space exploration on the second dataset

**🏁 CLOSED 2026-08-12.** 25 configurations built and place-and-routed, 0 failures, snapshot in
`docs/results-mnist/`. The frontier is **bounded by what was trained, not by the device** — a
deliberate stopping point, see the closing log entry. ⚠️ `device_pct` throughout both studies is
computed against a **marketing LUT cap, not the fabric**; the part has 32,600 LUT sites, not
20,800. Two Phase 1 predictions
were retracted along the way. Open: MNIST figures (`dse/plot.py` is not dataset-aware) and the
area model, which should not be quoted.

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
| M2a | Machine parity — prove this box reproduces Phase 1 | ✅ **done 2026-08-12 — 12/12**, areas exact at 110 / 1,519 / 1,621. ⚠️ no-board half only; run `--with-board` before M2f |
| M2·0 | **Noise floor** | ✅ **done 2026-08-12 — 0.24 pp.** Withdrew 3 reduction-study claims incl. the headline; `1x2000` replaces `2x[2000,1000]` as best model. See `reduction-ledger.md` |
| M2b | Get the checkpoints onto the machine | ✅ **done 2026-08-12 — all 14 local**, including the `_tau*` ladder and the `2x[1000,500]` retrain. 4 grid entries remain untrained (gaussian, linear, `2x500`, `3x330`) — no notebook produces them |
| M2c | Make `dse/` dataset-aware | ✅ **done 2026-08-12 — option 1, JSC byte-identical.** See the log |
| M2d | Fix `dse/area_model.py` | ⚠️ **bigger than stated.** `predict_comparators` is off by +108% on MNIST at z=3 and needs a z-dependent model, not a constant. Still not blocking measurement |
| M2e | Gate 1 across the grid | ✅ **done 2026-08-12 — 25/25 pass**, every built config bit-exact |
| M2f | Synthesis + place-and-route sweep | ✅ **done 2026-08-12 — 25 configs, 0 failures.** ⚠️ nothing failed to fit, so the frontier has no measured edge |
| M2g | Report, frontier, snapshot | ✅ **snapshotted to `docs/results-mnist/`.** ⚠️ frontier has **no measured edge** — bounded by training, deliberately. Plots pending: `dse/plot.py` has no `--dataset` |

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

✅ **As of 2026-08-12 all 14 sweepable checkpoints are local** in
`training/artifacts/sweeps-mnist/`, and the sweep is complete. This section is kept for a machine
starting from a fresh clone: checkpoints are large and deliberately not in git (one JSC file
exceeds GitHub's 100 MB limit), so `docs/results-mnist/` describes them in ~10 KB instead and a new
box has to re-download them.

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

⚠️ **`2x[1000,500]` is confounded too, and no corrected checkpoint exists — 13 of 14, not 14.**
The rule is *the final layer's width sets the group*, so the paper's config has group **50** and
wanted `tau = 2.246`, not 3.333 — the same ~48% error `1x500` had. It was missed because the
correction was framed as a *ladder* fix and this config is in the multilayer group. Everything else
is genuinely fine: the z-sweep, the n-sweep, `1x1000` and `2x[2000,1000]` all end at a 1000-wide
layer, group 100, which **is** the anchor.

| | count | source |
|---|---|---|
| ends at width 1000 → `tau` correct as trained | 8 | `mnist_grid_kaggle.ipynb` |
| ladder rungs 100/200/300/500/2000 | 5 | `mnist_reduction_tau_kaggle.ipynb` (`_tau*`) |
| **`2x[1000,500]`** | **1** | ⬜ **needs one retrain at τ=2.246, ~4 min** |

~~Expect roughly +0.27 pp from it, by analogy with `1x500`.~~ ⚠️ **Retrained 2026-08-12 and the
prediction was wrong**: 97.93 → **97.76**, slightly *down* and well inside the 0.24 pp floor. The
analogy to `1x500` did not hold. It was still worth the 4 minutes — every point in the grid is now
trained the same way, and the config turned out to matter for a different reason entirely (it is
the only ~3,500-LUT model that meets 100 MHz; see the sweep entry).

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

## 3. ~~`dse/` is JSC-shaped and must be generalised first~~ ✅ done 2026-08-12

**Kept for the record; the work is done.** `dse/` now imports `datasets` and JSC is verified
byte-identical. What follows is the survey that scoped it, and §3.1 is still live — the area
model is worse than this section estimated. See the log.

Originally: `dse/` did not import `datasets` at all, so every dataset fact in it was a private
copy of JSC's.

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
| ~~Does `dse/` come under the dataset-agnostic contract?~~ | ✅ **Closed 2026-08-12: yes, option 1.** JSC verified byte-identical |
| **No MNIST noise floor exists** ⚠️ | 🟡 **Notebook built, not yet run** — `training/mnist_noise_floor_kaggle.ipynb`, 16 runs (4 configs × 4 seeds), ~2 h on a Kaggle GPU. Until it lands, `2x[2000,1000]`'s +0.06 pp over `1x2000` and the −0.13 / −0.27 pp taper deltas are **unresolved, not small**. Deliberately four *widths*, since JSC took its 0.15 pp from one config and applied it everywhere |
| Does the paper's `2x[1000,500]` fit at any precision? | ⬜ Open. Phase 1 projected yes at 11-bit (38.0% of device), never built. ⚠️ Also **97.93% against `1x1000`'s 97.97** — 50% more nodes for nothing, so it is a fit question, not a frontier one |
| Does `2x[2000,1000]` fit? | ⬜ Open, and **the most interesting single build in Phase 2.** Best model in the study at **98.32%**, the only taper on the projected frontier, with a 2× smaller adder tree than `1x2000` — but 3,000 nodes against `1x300`'s 300, projected ~4,620 LUTs. Whether it fits at all is the first question |
| ~~Why is LUTs-per-comparator lower on real data than synthetic?~~ | ✅ **Closed 2026-08-12: two mechanisms, previously conflated.** Within a dataset it is logic sharing (falls monotonically with comparators-per-feature, 3 points); between datasets it is threshold values. See the log |
| Should `word_bits` be stored or derived per config? | ⬜ **Open, and newly urgent.** `z=25` needs Q1.8 where every other MNIST config needs Q0.8, so the descriptor's word width is a default, not an invariant. `required_int_bits()` derives the floor exactly. Decide before sweeping many z |
| Is 5 Mbaud a ceiling for a 1,569-byte record? | ⬜ Open, deliberately not measured. Phase 1 §6 — the I/O wall is 243,271×, so a faster link changes no reported number. Revisit only if a config makes a full pass painful |
| Does a corrected-`tau` model reproduce on silicon? | ⬜ Open. The board currently holds the **confounded** `1x300` (96.14%). Gate 1b's bit-exactness is unaffected by `tau`, but the hardware demo is not the best available model |

---

## Log

### 2026-08-12 — 🏁 Phase 2 closed. The MNIST frontier is bounded by training, not by the device.

**Decision, taken deliberately: the ladder stops at 2,000 nodes and the frontier's edge is not
measured.** Recording it as a stated limit rather than leaving it as an unexamined gap.

The largest MNIST design built is **6,877 LUTs**. Nothing came close to failing. ⚠️ That is
**33.06% of Vivado's quoted 20,800-LUT cap but only ~21% of the 32,600 physical LUT sites** — see
the denominator finding below; the wall is further out than the reported percentage suggests, and
the node estimate that used to sit here was too low by ~1.57x and has been removed. Reaching it costs ~1 h of Kaggle GPU, **1.2 GB** of checkpoints to download, and ~1.5 h of
place-and-route that scales worse than linearly near high occupancy.

**Why it is not worth it now.** The frontier edge answers *"how large can a DWN get on a Basys 3
before routing fails"* — a device-characterisation question. It is not an MNIST question, and it is
not a prerequisite for the tool, which is a generator: it turns a checkpoint into Verilog and never
needs to know where the device runs out. The paper's MNIST configuration is `2x[1000,500]`, which
fits at **16.65%** and meets timing, so the largest model anyone would plausibly build is already
measured with room to spare.

⚠️ **This is a real limitation and should be stated as one, not buried.** Any sentence of the form
"the largest MNIST model that fits is…" is unsupported by this study. The supported sentence is
*"the largest tried was 33.06% of the device and it fit comfortably."*

#### The two frontiers must stay separate, and not only for tidiness

They are not comparable, and a combined Pareto plot would mislead:

| | JSC | MNIST |
|---|---|---|
| features / classes | 16 / 5 | 784 / 10 |
| word format | Q3.12 (16-bit) | Q0.8 (9-bit) |
| accuracy scale | ~73–76% | ~93–98% |
| widest single layer built | `1x3000 z=50`, 13,972 LUTs (**67.2%**\*) | `1x2000`, 6,294 LUTs (**30.3%**\*) |
| cost per node there | **4.66 LUT/node** | **3.15 LUT/node** |
| encoder share there | **42%** | **21%** |

**The mechanism is the encoder, and it runs opposite in the two datasets.** JSC has 16 wide
features and a 16-bit word, so comparators are expensive (7.52 LUTs each) and the encoder stays a
large fraction of the design at every width. MNIST has 784 narrow features and a 9-bit word, the
mapping saturates early, and the encoder flattens at ~1,315 LUTs — 21% of the design at 2,000
nodes and falling. **So JSC reaches high occupancy at a fraction of MNIST's node count**, which is
exactly why JSC is the cheaper vehicle for probing the routing limit and MNIST is not.

\* Percentages are of Vivado's quoted 20,800-LUT cap, which is **not the fabric size** — see the
denominator finding below. Against the 32,600 physical LUT sites these are ~42.9% and ~19.3%. The
comparison between the two columns is unaffected, since both use the same (wrong) denominator.

Accuracy scales differ by 20 points, so even a shared axis would be meaningless. Keep
`docs/results/` and `docs/results-mnist/` separate, and plot them separately.

#### ⬜ Future work, explicitly deferred rather than forgotten

- **Extend the MNIST ladder to ~4,000 and ~6,000 nodes** to bracket the wall. Two rungs, ~30 min
  GPU, ~565 MB. Not needed for the tool; do it if the study ever wants a measured MNIST edge.
- **Do not merge the frontiers.** If a combined figure is ever wanted, it needs a normalised axis
  (LUTs per node, or accuracy relative to each dataset's ceiling) and the reasoning above stated
  alongside it.

#### ⚠️⚠️ `device_pct` is measured against the WRONG DENOMINATOR, in both studies

Chased down after JSC's `1x2000` appeared at **102.80% of device, routed, 0 errors**. That looked
impossible. It is not — the percentage is wrong, not the build.

| | Vivado's "available" for `xc7a35tcpg236-1` |
|---|---|
| Slice LUTs | 20,800 |
| **Slices** | **8,150** → 8,150 x 4 = **32,600 LUT sites** |

**The observation is solid; the mechanism is a hypothesis. Keep them apart.**

*Observed, and not in doubt:* the build is real — Placer Task ran, `route_design` completed,
`post_route.dcp` was written, 0 errors, clean DRC — and it used **5,532 of 8,150 slices = 67.88%**.
So **20,800 LUTs is not a hard ceiling on this part**, whatever the reason.

*Hypothesised:* 8,150 slices is the **XC7A50T's** count, and the 35T is a capacity-binned 50T
sharing the same die, so the physical grid is larger than the licensed LUT figure.

⚠️ **That hypothesis does not fully fit the report, and the gap is worth stating.** If Vivado were
simply modelling 50T fabric, flip-flops would read 65,200 (8,150 x 8). They read **41,600**, which
is the 35T's number:

| | LUT | FF | Slices |
|---|---|---|---|
| XC7A35T datasheet | 20,800 | 41,600 | 5,200 |
| XC7A50T datasheet | 32,600 | 65,200 | 8,150 |
| **this report** | **20,800** | **41,600** | **8,150** |

So the report mixes 35T LUT/FF availables with a 50T slice count. The consistent reading is that
the LUT and FF "available" columns are licensed caps while the Slice row reports the physical grid
— but that is inference from one report, not something verified against the part database. **Do not
state the 50T-die explanation as established.** What is established is the observation above.

**Consequences, and they touch every area number in both studies:**

- `DEVICE_LUTS = 20800` in `run_synth.py` / `area_model.py` makes `device_pct` a **fraction of a
  marketing number**, not of the fabric. Above roughly 64% it reads over 100% while the part is
  half empty.
- **Slice occupancy is the real placement constraint**, and it is not proportional to LUT count:
  MNIST's board design is 22.05% by LUTs but **28.87% by slices** (sparse packing at low
  utilisation), while JSC's `1x2000` is 102.80% by LUTs and **67.88% by slices** (dense packing
  when it has to). LUT % is neither an upper nor a lower bound.
- **JSC's frontier edge is not measured either.** Its largest build reached 67.88% slice occupancy
  without failing, so "the largest JSC model that fits" is also unsupported. Both studies stop
  short of the wall; JSC merely stops closer to it.
- The MNIST wall estimate below (~6,800-7,600 nodes) was derived against 20,800 and is therefore
  **too low by roughly 1.57x**. Do not quote it.

⚠️ **Not fixed here.** Changing `DEVICE_LUTS` would move `device_pct` in every committed JSC and
MNIST result, including the published snapshot. That is a deliberate, separate change for whoever
owns both studies — and it needs a decision on what the denominator should be (32,600 LUT sites, or
8,150 slices with LUT% reported alongside). **Recorded, not actioned.**

### 2026-08-12 — ✅ M2e/M2f: the MNIST sweep is complete. 25 configs, 0 failures.

`dse/run.py --dataset mnist --all --impl`, snapshotted to `docs/results-mnist/`. Every config that
has a checkpoint is built and place-and-routed. Four grid entries remain untrained (`gaussian`,
`linear`, `2x500`, `3x330`) — no notebook ever produced them.

#### The ladder

| rung | acc% | core | encoder | top | %dev | Fmax |
|---|---|---|---|---|---|---|
| `1x100` | 92.98 | 234 | 611 | **845** | 4.06 | **123.8** |
| `1x200` | 95.93 | 420 | 839 | 1,264 | 6.08 | **111.3** |
| `1x300` | 96.77 | 630 | 971 | 1,597 | 7.68 | **107.5** |
| `1x500` | 97.70 | 1,168 | 1,079 | 2,246 | 10.80 | **108.4** |
| `1x1000` | 97.97 | 2,272 | 1,220 | 3,490 | 16.78 | 93.1 |
| `1x2000` | 98.26 | 4,821 | 1,315 | 6,294 | 30.26 | 94.0 |

#### ⚠️ RETRACTED: "timing binds before area on MNIST"

Recorded in §5 as a Phase 1 prediction and reported as confirmed after the z-sweep. **It is wrong.**
The whole bottom half of the ladder clears 100 MHz comfortably — `1x100` reaches 123.8 MHz — and
the 90–95 MHz figures were a property of `1x1000` and wider, not of the dataset. The z-sweep looked
like confirmation only because every config in it was `1x1000`.

**Corrected:** timing binds above roughly 500 nodes. Below that the design is comfortable on both
axes. The prediction was generalised from one width, which is the same error this project has now
retracted five times.

#### ⚠️ RETRACTED: "pipeline depth is the lever, and it needs no retraining"

Also from Phase 1, also wrong — and wrong in the unhelpful direction. Fewer stages is worse on
**both** axes, at every width tested:

| config | stages | core | top | Fmax |
|---|---|---|---|---|
| `1x2000` | 4 | 4,821 | 6,294 | **94.0** |
| `1x2000` no OUT reg | 3 | 4,815 | 6,433 | 82.6 |
| `1x2000` no POP reg | 3 | 5,560 | 6,871 | **53.2** |
| `1x2000` 2-stage | 2 | 5,560 | 6,877 | 48.5 |

Removing the popcount register costs **41 MHz and 739 core LUTs** — a 200-wide group sum is a deep
adder tree and that register is load-bearing. **4 stages is both the architectural maximum for a
single-layer model and the optimum.** There is no pipeline lever to pull.

**What does move timing is the clock constraint itself.** Asking Vivado for 8 ns gives `1x1000`
**97.4 MHz**; asking for 10 ns gives 93.1. Over-constraining buys 4.3 MHz for 128 LUTs.

#### ✅ NEW: depth pays — for timing, not accuracy

The five configs that actually meet the 100 MHz board clock:

| config | acc% | LUTs | Fmax | latency |
|---|---|---|---|---|
| **`2x[1000,500]`** (the paper's) | 97.76 | **3,464** | **103.8** | 5 |
| `1x500` | 97.70 | 2,246 | 108.4 | 4 |
| `1x300` | 96.77 | 1,597 | 107.5 | 4 |
| `1x200` | 95.93 | 1,264 | 111.3 | 4 |
| `1x100` | 92.98 | 845 | 123.8 | 4 |

`2x[1000,500]` and `1x1000` are the **same size** (3,464 vs 3,490 LUTs) and statistically the same
accuracy (97.76 vs 97.97 — inside the 0.24 pp floor), but the two-layer model gets a **fifth
pipeline stage for free** and hits 103.8 MHz where the single-layer manages 93.1.

**At the board's real clock, the deep one is usable and the flat one is not.** This does not
contradict the reduction study's "depth does not pay" — that was an accuracy claim and it stands.
It adds an axis the accuracy study could not see.

#### The encoder saturates; `n` is not a lever

Encoder share of `dwn_top`: **72.3%** at `1x100`, 48.0% at `1x500`, **20.9%** at `1x2000`. Across a
20× node range the encoder grows only 611 → 1,315 LUTs, because MNIST is slot-limited — widening
the model buys new comparators only until the mapping saturates. The core/encoder ratio therefore
inverts across the ladder.

`n` buys almost nothing: `n=2` saves 7% of area for **−1.67 pp**, `n=4` saves 2% for −0.40 pp. Core
is flat (2,257 / 2,272 / 2,272) because one node is one LUT6 regardless of `n`; only the encoder
moves. On JSC `n` mattered. Here it does not.

#### M2d: the area model, now measurable

| rung | predicted | measured | error |
|---|---|---|---|
| `1x100` | 852 | 845 | +0.8% |
| `1x200` | 1,091 | 1,264 | **−13.7%** |
| `1x300` | 1,326 | 1,597 | **−17.0%** |
| `1x500` | 1,820 | 2,246 | **−19.0%** |
| `1x1000` | 3,105 | 3,490 | −11.0% |
| `1x2000` | 5,802 | 6,294 | −7.8% |

It **under**-predicts by 8–19% across the ladder, worst mid-range, and is accurate only at the
smallest rung. Combined with the +108% comparator error at z=3, the model is not usable for MNIST
projections. It never blocked the sweep — nothing was filtered — but **no projected MNIST area
should be quoted.**

#### ⚠️ The frontier has no measured edge

`dse/report.py` says it and it is worth repeating: **no config in this sweep failed to fit.** The
largest is 30.26% of the device, so that is *the largest tried*, not the frontier's edge. MNIST's
ladder stops at 2,000 nodes; JSC's went to 3,000 specifically to find the wall. Extending upward
until something fails is outstanding work.

#### Corrections to my own predictions in this ledger

- **`2x[1000,500]` at corrected `tau` was predicted at "roughly +0.27 pp".** Measured: 97.93 →
  **97.76**, i.e. slightly *down*, and well inside the noise floor. The analogy to `1x500` did not
  hold.
- Three process errors, none of which corrupted data: (1) I diagnosed two concurrent sweeps
  clobbering `results.json` and killed a healthy run — the two PIDs were a parent/child pair from
  one launch, and counting `python.exe` by command-line substring was never a valid test; (2)
  `dse/run.py` counted `checkpoint-mismatch` as "already done", so downloading the corrected
  checkpoints changed nothing until transient statuses were made retryable; (3) the snapshot
  printed `-> docs/results/` while writing to `docs/results-mnist/`, which reads exactly like the
  JSC snapshot being overwritten. All three are fixed. **`save_result()` still does an unguarded
  read-modify-write on shared JSON** — a real hazard if two sweeps ever do run at once.

### 2026-08-12 — word width is now derived per config, and it found two dead JSC sweep points

`z=25` could not be swept: the grid emits at the descriptor's Q0.8 and that config needs Q1.8.
`dse/run.py` now calls `widen_for_checkpoint()` before building — it reads the checkpoint, derives
the floor with `required_int_bits()`, and widens the word if the descriptor's default is too
narrow. It **only ever widens**, never narrows, so a config that fits keeps the descriptor's width.
Fractional bits are untouched, because they are not derivable from a checkpoint and silently
trading an exact representation for a smaller one is the failure mode `required_int_bits` warns
about.

The config name carries the result (`q10.8` vs `q9.8`), which is right — it is a different design.

Also fixed while in there: `run_config` was calling `gate1()` **without** passing
`cfg.hw.word_bits`, so the emitter re-resolved precision from the descriptor on its own. The name
and the RTL agreed only by coincidence, and would have diverged the moment anything overrode the
default — which is exactly what widening does.

#### ~~This revealed that the JSC study has no `linear` encoding data at all~~ — ⚠️ overstated, withdrawn

Running the check across JSC's grid, two configs would widen: `1x200 linear` and `1x360 linear`,
thresholds spanning ±8.906 against Q3.12's 3 integer bits. Both are recorded as `gate1-failed`, and
they are the only 2 failures in all 54 JSC configs.

**This was written up as a discovery. It is not one.** `docs/phase2-ledger.md` already records the
cause and the decision: Q4.11 would represent them at *identical* area (still 16-bit), but the
encoding axis spread is **0.12 pp against a 0.15 pp noise floor**, so the precision plumbing was
judged not worth it. Documented failure with a stated reason, not a silent gap.

**And rebuilding them with `widen_for_checkpoint()` would be actively wrong.** It widens the WORD
(16 → 17 bits, Q4.12) because it refuses to touch fractional bits. The comparable fix is **Q4.11 —
16 bits, identical area**. At 17 bits the linear points stop being comparable to the gaussian and
distributive points at 16, so the encoding axis becomes confounded with word width. That is the
same defect the MNIST `z=1` vs `z=25` comparison had to be caveated for, and here it would be
self-inflicted on an axis already known to be inside the noise.

**Decision: not rebuilding.** No committed JSC number changes.

⚠️ **The general lesson for `widen_for_checkpoint()`:** widening is right when the fractional bits
are load-bearing (MNIST needs all 8 to represent 8-bit pixels exactly, so Q1.8 at 10 bits is the
only option). It is the wrong lever when an equally exact narrower-fraction format exists at the
same width, because it silently trades comparability for representability. The function does not
know the difference — it optimises for exactness, which is the safe default, not the free one.

### 2026-08-12 — `run.py` and `report.py` too, and the sweep runner caught the `tau` confound itself

All four of `grid.py`, `area_model.py`, `run.py`, `report.py` now take `--dataset`. Sweep paths are
descriptor fields rather than special cases in code:

| | JSC (unchanged) | MNIST |
|---|---|---|
| results | `build/dse/results.json` | `build/dse/mnist/results.json` |
| snapshot | `docs/results/` | `docs/results-mnist/` |
| checkpoints | `training/artifacts/sweeps/` | `.../sweeps-mnist/` |

JSC's are recorded as data with a comment saying they are historical: `docs/results/` is
referenced by `REPORT.md`, `README.md` and the `jsc-complete` tag, and `build/dse/results.json`
holds 54 measured configs that a moved path would silently re-run from scratch. Verified
unchanged — `--list` byte-identical, 54/54 results found, 40/40 checkpoints resolve.

#### ✅ `dse/run.py` detects the `tau` confound without being told

Running `1x100`:

```
CHECKPOINT MISMATCH: tau: checkpoint has 3.3333, grid expects 0.8972
-- this checkpoint predates a schedule change and is a DIFFERENT model
```

The grid computes `tau` from the descriptor's power law, so every confounded ladder checkpoint is
**rejected automatically**. That is strictly better than §1.3's warning, which relies on someone
reading it. The config is recorded as `checkpoint-mismatch` rather than dropped.

#### Two defects found, one of them mine

- **Slug convention mismatch.** JSC's notebook writes `n6_z200_..._checkpoint.pt`, MNIST's writes
  `mnist_n6_z3_...`. Added `slug_prefix` to the descriptor and made resolution try both, rather
  than renaming trained checkpoints — renaming breaks every recorded number that refers to them.
  14 of 18 MNIST training configs now resolve; the 4 missing (gaussian, linear, and the equal-split
  multilayers) were never trained.
- ⚠️ **MNIST was being built at JSC's Q3.12.** `Config`'s default `HardwareConfig` carries
  `word_bits=16, frac_bits=12`, and the grid was not overriding it — so every MNIST config would
  have emitted a silently oversized 16-bit encoder. Visible **only** as `q16.12` in the config
  name that `dse/run.py` prints. Fixed with a `_hw()` helper that fills precision from the
  descriptor. JSC is unaffected because its descriptor values equal the old defaults, which is
  exactly why nothing caught this earlier.

#### Cross-check: the sweep path and the hand path agree

`1x1000 z=1` through `dse/run.py --impl` gives 2,272 / 846 / 3,118 LUTs and WNS −0.487 ns —
**identical** to the same config run by hand through `scripts/run_synth.py`. The hand measurements
taken while the refactor was in progress can therefore be discarded rather than reconciled.

`dse/plot.py` is **not** converted. One config is not a frontier, so there is nothing to plot yet.

### 2026-08-12 — M2c done: `dse/` is dataset-aware. M2d is worse than it looked.

**`dse/grid.py` and `dse/area_model.py` now read `datasets/`.** Option 1 from §3 — `dse/` is under
the dataset-agnostic contract. `python dse/grid.py --dataset mnist` builds a 29-config grid; the
default is still JSC.

Moved from module constants into the descriptor: `classes`, the size ladder, z values, base `n`
and encoding, the `tau` schedule, the training recipe, OFAT rungs, Group B rungs, corners, and the
encoder calibration.

**JSC is byte-identical.** `--json` training set unchanged across all 40 runs, `--list` unchanged
across all 54 configs, `rtlgen/config.py` self-test still passes. Two things nearly broke it and
are worth naming:

- `tau_for` moved to `Dataset.tau_for` and had to reproduce four-anchor geometric interpolation
  exactly — verified over every width 5–3000, zero mismatches. The descriptor also had to express
  MNIST's *one* anchor plus a borrowed exponent, which four-anchor interpolation cannot.
- The Group B clock-variant gate reads `ofat_rungs[-1]` (360), **not** `group_b_rungs[-1]` (1600).
  Substituting the wrong one moved every clock variant to a different width. Caught by the diff,
  which is the only reason the baseline was captured first.

`tau_basis` is a new descriptor field because JSC feeds `tau_for(sum(layers))` while the
physically motivated width is the **final** layer — GroupSum's logit range is
`(final / classes) / tau`. JSC keeps `'total'` for reproducibility with a comment saying it is
preserved, not endorsed; MNIST uses `'final'`. They differ only for multi-layer configs.

#### ⚠️ The MNIST area model is not usable, and the cause is not the constant I set out to fix

`predict_comparators` is calibrated on **one** JSC configuration and does not transfer:

| config | predicted | measured | error |
|---|---|---|---|
| JSC `1x50 z=200` (the calibration point) | 224 | 202 | **+10.9%** |
| MNIST `1x1000 z=1` | 666 | 431 | **+54.5%** |
| MNIST `1x1000 z=3` | 1,496 | 720 | **+107.8%** |
| MNIST `1x1000 z=25` | 3,364 | 3,400 | −1.1% |

It is accurate only where the encoder saturates (z=25: 19,600 bits available against 6,000 slots).
At the low `z` that matters most for MNIST — z=1–3, the cheap end the reduction study says is
nearly free — it is wrong by a factor of two. `selection_fraction` was fitted deep in a regime
MNIST does not occupy.

**The `LUT_PER_COMPARATOR_BIT` constant is now per-dataset and is measured, but it cannot rescue
this**, because the error is in the comparator count it multiplies.

| | comparators | comps/feature | LUT/bit | LUT/comparator |
|---|---|---|---|---|
| JSC `1x50 z=200` | 202 | 12.62 | 0.4700 | 7.52 |
| MNIST `1x1000 z=1` | 431 | 0.55 | 0.2181 | 1.96 |
| MNIST `1x1000 z=3` | 720 | 0.92 | 0.1417 | 1.27 |
| MNIST `1x1000 z=25` | 3,400 | 4.34 | 0.0863 | 0.86 |

**This resolves the open question about cost-per-comparator — as two mechanisms, not one.** Within
MNIST, cost per comparator falls monotonically as comparators-per-feature rises: logic sharing
between comparisons against different constants on the *same* input word. That is the
"amortisation" idea, and it survives on three points. It does **not** explain the gap *between*
datasets — JSC has by far the most comparators per feature (12.62) and is still the most expensive
per bit — which leaves threshold *values* as the between-dataset mechanism, as hypothesised.
Previously these two were conflated into a single claim, which is why one measurement appeared to
falsify it.

⚠️ **A z-dependent comparator model is the real fix and is not done.** Until it is, no MNIST area
*projection* should be quoted. This does **not** block the sweep: `should_synthesize` only skips
configs predicted far too large, every MNIST config is predicted to fit, so nothing is filtered and
no measurement depends on the prediction.

#### The JSC self-test was already failing, before any of this

`python dse/area_model.py` reports **FAIL** at 10.9% on the `1x50` encoder, and does so identically
at the commit before the descriptor refactor — verified by running the old file. It also compares
against a stale board number (2,058; now 1,893). Pre-existing, not caused here, and still open.

### 2026-08-12 — the first two MNIST sweep points, and z=1 is as cheap as the study hoped

`1x1000` at both ends of the z axis, Gate 1 bit-exact, place-and-routed:

| | z=1 | z=25 | Δ |
|---|---|---|---|
| accuracy | 97.91% | 98.23% | +0.32 pp |
| `dwn_core` | 2,272 | 2,272 | — |
| `thermometer_encoder` | **846** | **2,935** | **3.47×** |
| `dwn_top` | **3,118 (15.0%)** | **5,195 (25.0%)** | **+2,077 LUTs, +66.6%** |
| Fmax | 95.4 MHz | 92.8 MHz | both **MISS** 100 |

**0.32 pp of accuracy costs 2,077 LUTs — two thirds of the design.** The cores are identical, as
they must be: same 1,000 nodes, same group size, so the entire difference is encoder.

⚠️ **Not a clean single-variable comparison.** z=25 must be built at **Q1.8 (10-bit)** and z=1 at
Q0.8 (9-bit), so a 1.11× word-width factor is folded into that 3.47×. Correcting for it crudely
leaves ~3.1×, so the conclusion is unchanged in direction and slightly overstated in size. There is
no 9-bit z=25 build to compare against, because one cannot exist — see below.

#### `z=25` cannot be built at Q0.8, and the emitter caught it

Its maximum threshold is **exactly 1.000000** — at 25 quantiles one lands on the top pixel value —
and Q0.8 represents [−1, 1). The emitter refused rather than emitting comparisons against a wrapped
constant. Every other config in the grid tops out at 0.992157 and is fine at 9 bits.

**So `word_bits` in the descriptor is a default, not a dataset invariant.** It is correct for z≤8
and wrong for z=25 *on the same dataset* — the same shape as the JSC finding that the safe width
moved between two configurations. `required_int_bits()` derives the floor exactly, so the open
question is whether the descriptor should store a width at all or compute it per config. **Decide
before sweeping many z.**

#### Both miss 100 MHz, which confirms a Phase 1 prediction

`1x1000 z=1` uses 15.0% of the device and still misses the clock at 95.4 MHz. Phase 1 predicted
timing would bind before area for MNIST, against the JSC frontier's shape. **First evidence in, and
it holds.** Pipeline depth is the lever and needs no retraining — Group B rungs for MNIST are
already 300 / 1000 / 2000 in the descriptor.

### 2026-08-12 — M2a: this machine reproduces Phase 1, 12/12

```
dwn_core              110 LUTs,  73 FF      (expected 110 / 73)
thermometer_encoder  1519 LUTs,   0 FF      (expected 1519 / 0)
dwn_top              1621 LUTs, 269 FF      (expected 1621 / 269)
Gate 1               1504 core / 1518 top vectors, PASS
harness unit tests   PASS
```

Exact on every area, and on the **post-argmax-tree** values rather than the `jsc-complete` tag's
108 / 1,619 — so §1.1's corrected table is the right one to check against, confirmed by
measurement rather than by reading.

**What this establishes is narrower than "the code works":** this machine's Vivado produces
bit-identical areas to the machine that recorded Phase 1. That is the precondition for an MNIST
sweep point being comparable to a JSC one. A drift of even one LUT would have made every
cross-dataset comparison in the eventual report unattributable — which is the failure this check
exists to prevent, and it is cheap precisely because it is run before anything depends on it.

⚠️ **Simulation and synthesis only.** The board half — `--with-board`, 22/22, Gate 1b
166,000/166,000 — has not been run in this session and needs a Basys 3 attached. Nothing in M2b–M2g
requires it, but **M2f's board builds do**, and it should be run before the first one rather than
after a failure.

#### A gap found while reading the handoff, not by running anything

`2x[1000,500]` is `tau`-confounded and has no corrected checkpoint — 13 of 14, not 14. Its final
layer is 500, so group 50, so it wanted τ=2.246 like `1x500` did. It was missed because the
correction was framed as a *ladder* fix and this config sits in the multilayer group. Recorded in
§1.3 with the full per-config table; costs one ~4-minute Kaggle run.
