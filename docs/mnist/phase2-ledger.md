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
| M2b | Get the 13 remaining checkpoints onto the machine | ⬜ trained on Kaggle, **not present locally** |
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

All 14 configurations are trained, but **only `mnist_n6_z3_distributive_w300` is on the machine
that finished Phase 1.** Checkpoints are ~933 MB total and deliberately not in git (one file
exceeds GitHub's 100 MB limit); `docs/results-mnist/` describes them in ~10 KB instead.

Download all 14 `_checkpoint.pt` and `_testvectors.npz` pairs from the Kaggle run's Output into
`training/artifacts/`. `dse/run.py --list` will then say which configs have a checkpoint and which
do not, and refuses to synthesize one that is missing.

### 1.3 The grid — 14 configs, already trained

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

## 4. What Phase 1 says to expect

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

## 5. Open questions

| Question | Status |
|---|---|
| Does `dse/` come under the dataset-agnostic contract? | ⬜ **Open, and blocking M2c.** See §3 |
| Does the paper's `2x[1000,500]` fit at any precision? | ⬜ Open. Phase 1 projected yes at 11-bit (38.0% of device), never built. A negative answer is a result, not a failure |
| Why is LUTs-per-comparator lower on real data than synthetic? | ⬜ Open, hypothesis untested. §3.1 |
| Does the `tau` correction change the frontier? | ⬜ **Open and cheap to get wrong.** `1x300` gained +0.63 pp when `tau` was corrected to 1.678, and the narrow rungs moved ~5× as much. If the grid was trained at `tau=1/0.3` throughout, the ladder's *shape* is distorted, not just its offset. See `docs/mnist/reduction-ledger.md` |
| Is 5 Mbaud a ceiling for a 1,569-byte record? | ⬜ Open, deliberately not measured. Phase 1 §6 — the I/O wall is 243,271×, so a faster link changes no reported number. Revisit only if a config makes a full pass painful |
| Does the `tau1p678` model reproduce on silicon? | ⬜ Open. Never exported, synthesized or run — the checkpoint is not on the Phase 1 machine |

---

## Log

*(empty — Phase 2 has not started. Add a dated entry per work session, newest first.)*
