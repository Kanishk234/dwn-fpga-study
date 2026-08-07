# Phase 2 — DSE: running ledger

**Live document. Update it as work lands, not afterwards.** Status table first, then the
chronological log, then the numbers worth quoting, then what is still open.

**Status: not started.** Phase 1 is complete (`docs/phase1-report.md`); Gate 1b passed
166,000/166,000 on hardware.

Phase 2's question (brief §10, Study 1): *what is the accuracy/area/latency Pareto frontier for
DWN on a fixed small FPGA?* Its deliverables are Pareto plots plus one headline number — **the
largest DWN that fits an XC7A35T, and what it scores** — with **core and encoder LUTs reported
separately, always** (brief §6).

Read first: `docs/dse-plan.md` (what gets swept and why), `docs/phase2-handoff.md` (machine
setup and the restructure decision), `docs/phase1-report.md` (what already works).

---

## Status

| Step | What | Status |
|---|---|---|
| **2a** | **Make the flow config-driven** — see below. Mostly code, not file moves | ✅ baseline reproduces 108/1519/1619 through `dse/run.py` |
| **2b** | Recalibrate the area model against measured encoder cost | ✅ `dse/area_model.py`, 0.5% on the measured config |
| **2c** | Train the Group A grid (GPU-bound, Kaggle, batched) | 🟡 `training/train_grid_kaggle.ipynb` written; needs a Kaggle run |
| **2d** | Filter on predicted area before spending any Vivado time | ❌ |
| **2e** | Synthesize the survivors (serial Vivado, the expensive part) | ❌ |
| **2f** | Group B sweeps (pipeline depth, clock, strategy) on survivors only | ❌ |
| **2g** | Merge into one Pareto frontier + the headline number | 🟡 tooling done (`dse/report.py`, `dse/plot.py`); needs data |
| — | *Optional:* n=2 congestion characterization, reported separately | ❌ |

### Prerequisites before any of the above

| Item | Status |
|---|---|
| Machine passes `scripts\verify_phase1.py --with-board` (22/22) | ✅ 2026-08-04, this machine |
| **All sweep synthesis from ONE machine and ONE Vivado version** | decide before 2e |
| Full 166k test set present (gitignored, does not travel) | ✅ here |
| `rtl/gen/` retired in favour of `build/<config>/rtl/` | part of 2a |

---

## Step 2a in detail — what "config-driven" actually means

Roughly 20% file moves, 80% code. `docs/phase2-handoff.md` Part 2 has the directory decision;
this is the work behind it.

**The mechanical part** (minutes): `exporter/emit_core.py` and `emit_encoder.py` move to
`rtlgen/` — brief §11 splits *exporter* (checkpoint → tables/wiring/thresholds) from *rtlgen*
(that export → Verilog), and Phase 2 needs them separable because the filtering step estimates
area from a checkpoint **without** emitting RTL. `rtl/gen/` is deleted; output moves to
`build/<config>/rtl/`. `rtl/` (the hand-written primitives) does not change.

**The actual work** — four things are hardcoded today that a sweep must vary:

| Hardcoded now | Needs to become |
|---|---|
| `emit_core.py` writes to `rtl/gen/dwn_core.v` | an output directory argument — otherwise sweep point #7 overwrites #8 |
| `PIPE_LUT/POP/OUT = 1` as module-level constants | config fields, since Group B sweeps exactly these |
| `run_gate1.py`'s `RTL_SOURCES`, `run_synth.py`'s `TARGETS` | derived from the config's build directory |
| "config" = the `CONFIG` dict inside a checkpoint | a first-class object spanning **training params + hardware params** (pipeline depth, clock target, strategy) — the checkpoint knows nothing about the latter |

Then `dse/` is written on top: the grid, the loop over configs, report parsing into a results
table, Pareto computation, plots. It should **import** `run_xsim()`/`find_vivado_bin()` from
`scripts/run_gate1.py` and `run_one()` from `scripts/run_synth.py` rather than shelling out and
parsing stdout — those were exposed in Phase 1 for this.

**Why this is load-bearing, not tidying:** `dse-plan.md` §1 requires that a sweep point be a
config, not a code edit. If changing configuration means editing a `.py` or `.v`, every sweep
point synthesizes code that has not passed Gate 1 — forty points would mean forty
re-verifications.

**What is already done:** `extract.py` and both emitters handle arbitrary layer counts, widths,
`n` and class counts — Phase 1 proved it by running the same code over a `[300,100]` two-layer
model and a `[50]` single-layer one. What is missing is plumbing: where they write, and what
tells them what to build.

## What Phase 1 already answered

`docs/dse-plan.md` §7 listed four open questions. Three are now measured, and two of the answers
change the plan.

### 1. Encoder cost — far worse than budgeted ⚠️

| | |
|---|---|
| anticipated (brief §12 risk #3, Mecik & Kumm) | up to **3.2×** the core |
| **measured** | **14.06×** — 1519 LUTs of encoder against 108 of core |

**The §5 area formula must be rebuilt around this before it is used to filter anything.**
Filtering with a 3.2× assumption would pass configs that cannot fit by a factor of four.

Worse, encoder cost does **not** scale with the core. It scales with the number of *distinct
thresholds the mapping selects*, which grows with node count and **saturates at
`features × z` = 3200 comparators**:

| Config | Nodes | Comparators | Encoder LUTs | Core LUTs | vs 20,800 |
|---|---|---|---|---|---|
| `sm` 1×50 | 50 | **202 (measured)** | **1,519** | 108 | **7.8%** ✅ |
| `md` 1×360 | 360 | ≤ 2,160 | ≲ 16,000 | ~720 | ~80% — tight |
| `lg` 1×2400 | 2400 | → ~3,200 | ~24,000 | ~4,972 | **>100%** ❌ |

**`lg` almost certainly does not fit on a Basys 3, and the network is not the reason.** Its 4,972
core LUTs are 24% of the part, exactly as brief §6 predicts. Plan the size ladder against the
encoder, not the core.

*(`md`/`lg` comparator counts are bounds. At `sm` the mapping selected 202 distinct bits from 300
slots — 67% — so real numbers may land lower. Only training those configs settles it.)*

### 2. Pipeline depth does move Fmax — Group B does not collapse

dse-plan §7 asked whether extra stages buy anything. Measured on `dwn_top`:

| Stages | Cycles | LUTs | FF | Fmax |
|---|---|---|---|---|
| 1 | 1 | 1619 | 196 | 84.2 MHz ❌ |
| 2 | 2 | 1619 | 246 | 94.6 MHz ❌ |
| 3 | 3 | 1619 | 249–266 | 115.7–122.9 MHz ✅ |
| **4** | **4** | **1619** | **269** | **161.0 MHz** ✅ |

Nearly 2× across the range, and **LUT count never changes** — pipelining costs flip-flops, not
logic. So Group B is a real axis, it is cheap (no retraining), and 3 stages is the floor for the
board's 100 MHz.

### 3. Reduction cost — the deferred decision now has a number 🟡

dse-plan set the bar explicitly: *"if it's 40% of area, building the pyramid is obviously worth
it; if it's 3%, this was never an interesting axis."*

`dwn_core` is **108 LUTs** for 50 nodes plus 5 × popcount(10) and a 5-way argmax. At one LUT6
per node, the reduction is **~58 LUTs — roughly 54% of the core, and slightly larger than the
network itself.** That matches the paper's warning that in small models "the popcount circuit can
be as large as the network."

⚠️ **This is inference by subtraction, not a measurement.** Vivado inlined `lut_node`, `popcount`
and `argmax` into the top level, so the hierarchical report attributes everything to `dwn_core`.
**Confirm it by synthesizing the reduction standalone before committing to build Learnable
Reduction** — but on this evidence it clears dse-plan's bar comfortably.

### 4. Real widths for the size ladder

`sm` = 1×50 is known-good end to end at **73.83%** (paper: 74.0%). The paper's other JSC points
are `md` = 1×360 (75.6%) and `lg` = 1×2400 (76.3%), all at z=200, single layer, tau tracking
width (1/0.7, 1/0.3, 1/0.1, 1/0.03 for 10/50/360/2400 nodes).

---

## Constraints carried out of Phase 1

Things that will silently produce wrong sweep points if forgotten:

- **Final layer width must be divisible by `num_classes`.** `GroupSum` zero-pads silently
  otherwise, and hardware and software then disagree about group boundaries
  (`docs/checkpoint-format.md` §4). The emitter asserts this; the sweep grid must respect it.
- **`tau` tracks layer width.** It is not a constant to copy from `sm` — see the paper's values
  above.
- **Vector store depth limits batch size**, and bigger models mean wider vectors. `DEPTH=1024` ×
  259 bits ≈ 265 Kbit today, ~15% of block RAM.
- **The Q3.12 reference, not the float model**, is what hardware is scored against. Quantization
  is spec, not error (Phase 1 lost a debugging cycle to this).
- **One machine, one Vivado version**, or the frontier is two half-frontiers.

---

## Log

### 2026-08-04 — Phase 1 reproduced on this machine (the Phase 2 entry gate)

`scripts/verify_phase1.py` exists to answer one question before any sweep starts: *are this
machine's numbers comparable to Phase 1's?* A Pareto frontier assembled from two toolchains is
two half-frontiers with an unknown offset. Run here from a clean tree, no cached `build/`.

**Everything deterministic matched exactly.** Simulation and out-of-context synthesis:
12/12 checks. Then bitstream, program, and Gate 1b across the full test set:

```
Gate 1               : core 1504/1504 · top 1518/1518 · 50 nodes · 202 comparators
harness unit tests   : PASS
dwn_core             : 108 LUT   73 FF
thermometer_encoder  : 1519 LUT   0 FF
dwn_top              : 1619 LUT 269 FF
board design         : 2058 LUT 865 FF 8 BRAM 0 DSP · WNS +1.753 ns
Gate 1b              : 166000/166000 · 166815 core cycles
accuracy             : float32 73.8361% · Q3.12 73.8349%
float-vs-fixed       : 30/166000 (0.0181%) · 1 saturation event
```

`166815 = 166000 + 162 × (LATENCY+1)`, so **II=1 held on silicon again**. Only the stochastic
quantities moved, all within the tolerance the script allows: wall clock 11.4 s (was 11.2),
link rate 14,537/s (was 14,823), I/O wall 6,846× (was 6,713×).

**Python 3.14, not the documented 3.12 — and it was not a confound.** The `.venv` here was
empty and this machine no longer has 3.12 registered (`py --list` shows only 3.14 and 3.11).
Installed the *exact* pins (`numpy==2.3.4`, `torch==2.13.0`, `pyserial==3.5`), which have cp314
wheels, and every bit-exact result held. Reasoning: LUT/FF counts are pure Vivado 2025.2 with no
Python involved, and the golden model's numerics come from numpy's compiled kernels, which do
not change across interpreter minor versions. **Do not read this as permission to drop the 3.12
pin** — `requirements.txt` justifies it on Kaggle parity and on Phase 3's hls4ml/conifer lagging
new Python releases, and neither claim was tested here. It means only that Phase 1's *results*
are interpreter-independent.

**Two corrections came out of this run:**

- `docs/phase1-ledger.md`'s board table read **2054 LUTs / +1.662 ns**; the measured value is
  **2058 / +1.753 ns**, matching the report, handoff, manifest and `verify_phase1.py`. Ledger
  corrected in place with the retraction kept visible.
- The 166k test set is gitignored and **does not travel with the repo**. It was absent here and
  cannot be regenerated locally — upstream `torch_dwn` has no CPU path, so even inference needs
  a GPU. Copied from the Phase 1 machine and verified before use: 166,000 samples, software
  accuracy 73.8361%, first 1000 predictions 1000/1000 against the committed `testvectors.npz`,
  and `x_raw` max diff 4.768e-07 — the recorded 1-ULP `scaler.transform()` fingerprint, which
  is what proves it is the same dump rather than a re-derived one.

**Verdict: this machine is safe to run Phase 2 sweeps on**, and its points are comparable to
Phase 1's. Re-run afterwards as the single scripted command: **22/22 passed.**

### 2026-08-07 — 2a step 1: the config object exists

`rtlgen/config.py`. First piece of 2a, and the first thing to live in `rtlgen/` — which until
now did not exist at all, despite being in the repo layout. **Nothing was moved and no existing
file was touched**, so the flow provably cannot have changed and 22/22 still stands.

`Config = ModelConfig (from the checkpoint) + HardwareConfig (chosen per run)`. The split is the
point: the checkpoint knows only about *training*, while pipeline depth, clock, part and
strategy are Group B axes swept on an already-trained model. Putting them in the checkpoint
would be wrong — one checkpoint feeds several hardware configs. `with_hw()` produces a Group B
variant without retraining, which is exactly what 2f needs.

**Every default reproduces Phase 1 exactly**, so the object can be threaded through the flow
without changing an emitted bit. `python rtlgen/config.py` asserts that against the constants
still owned by other modules — `emit_core.PIPE_LUT/POP/OUT`, `extract.WORD_BITS/FRAC_BITS`,
`run_synth.DEFAULT_PART`, and latency == 4. If someone edits one and not the other it fails
loudly instead of desynchronizing silently. That guard is temporary by design: it deletes itself
when those modules start reading the config instead of owning constants.

Two details worth recording:

- **`name` carries every swept axis** (`n6_z200_distributive_w50_q16.12_p1111_c10`). Two configs
  differing in any axis must get different directories, or one silently overwrites the other's
  build — the exact failure the object exists to prevent.
- **`latency` is derived** (`pipe_enc+lut+pop+out`), never hand-copied. `benchmark_fsm` aligns
  labels with it and a drifted value scores every sample against the wrong answer — a bug that
  has already happened once in this project.
- `ModelConfig.__post_init__` rejects `layers[-1] % num_classes != 0` up front, so a bad sweep
  point dies before Vivado is launched rather than after GroupSum silently zero-pads.

**Next**, in an order that keeps `verify_phase1.py` green after every step: output dir as a
parameter → pipeline depth from config → path derivation in `run_gate1`/`run_synth`/
`build_bitstream` → *then* the moves (`rtlgen/`, `build/<config>/rtl/`, delete `rtl/gen/`).
The moves come last because by then everything already takes paths as inputs.

### 2026-08-07 — 2a step 2: both emitters take an output directory

Smaller than planned, because **`emit_encoder.py` already had `--outdir`**. The real defect was
an asymmetry: `emit_core.py` took `--out <file>` while `emit_encoder.py` took `--outdir <dir>`,
so the two emitters wanted different kinds of argument for the same idea — awkward for a caller
that hands both a single `Config.rtl_dir`. `emit_core.py` now takes `--outdir` too and derives
`dwn_core.v` / `dwn_core_params.vh` from it. Defaults are unchanged (`rtl/gen`).

`--out` is **gone, not deprecated**. Nothing called it — `run_gate1.py` passes no output flag at
all and relies on the default — so keeping both spellings would have been dead weight.

**Verified three ways, not one:**

- emitted into a throwaway `build/configs/` directory: all five files land there, read-back
  checks 50/50 nodes and 202/202 comparators
- regenerated into the default location: **`git diff rtl/gen/` is empty** — byte-identical to
  the committed Phase 1 output, which is the strongest available regression check while
  `rtl/gen/` is still committed
- custom directory vs default: all five files `cmp`-identical
- **Gate 1 re-run: 1504/1504 core, 1518/1518 top, PASS.** Byte-identical output means it could
  not have failed, but the emitter changed and the rule is that confidence is not verification.

`rtlgen/config.py`'s self-test still passes, so the pipeline constants have not drifted.

### 2026-08-07 — `--impl` validated through the runner

The last untested path in the sweep pipeline. Everything so far had gone through
synthesis-only, but `--impl` is what every real sweep point will use.

```
core 108 | encoder 1519 | top 1619 | Fmax 147.1 MHz | 379s
```

**147.1 MHz is Phase 1's post-route figure exactly** (against 161.0 post-synthesis), so the flag
genuinely reaches Vivado through `run_one()` rather than being silently dropped. Area identical.
The checkpoint resolved **by slug with no `--checkpoint` argument**, exercising the resolution
path added in 6d. `report.py` correctly dropped its post-route caveat once `impl` was true, and
latency was restated as **27.19 ns** (4 cycles at 147.1 MHz) instead of the optimistic 24.84 at
post-synthesis Fmax — which is the point of computing latency in nanoseconds.

**Cost: 379 s = 6.3 min per config**, against the 12 min budgeted. `MINUTES_PER_SYNTH` is
deliberately **not** lowered: 6.3 min is the *smallest* config in the grid, `1x1200` has 24× the
nodes, and place-and-route scales worse than linearly with occupancy. Planning the ladder from
its cheapest rung would underestimate exactly the end that runs a session out of time.

### 2026-08-07 — The reduction, measured: 50 + 58 = 108 exactly

`scripts/experiment_reduction.py` synthesizes the two halves of `dwn_core` separately, closing
an open question Phase 1 left and `dse/area_model.py` depended on.

| Part | LUTs | % of core |
|---|---|---|
| `nodes_only` — 50 `lut_node`, real tables and wiring | **50** | 46.3% |
| `reduction_only` — 5 × popcount(10) + argmax | **58** | 53.7% |
| **sum** | **108** | **100.0%** |
| `dwn_core` measured in Phase 1 | 108 | |

**Two results, and the first is the more valuable.**

**1. One DWN node is exactly one LUT6 — measured, not assumed.** 50 nodes, 50 LUTs. Brief §4's
architectural premise, and the thing the whole area model rests on, had never been observed
directly because Vivado inlines `lut_node` into the top level. Now it has.

**2. The inferred reduction was exactly right, and the sum is exact.** 108 = 50 + 58 with *zero*
cross-boundary optimization — the fragments do not share logic with each other in the real core.
So `area_model.py`'s reduction term (1.0 LUT per final-layer bit + 1.6 per class) is
measurement-backed rather than a fitted guess.

⚠️ **But do NOT read "54% of the core" as "worth building Learnable Reduction".** The script
prints that verdict because `dse-plan` §3 set the bar at *"40% of area"* — and that bar was
written before anyone knew the encoder costs 14× the core. Judged against the whole design:

| | `sm` 1×50 | `md` 1×360 (projected) |
|---|---|---|
| reduction | 58 | ~368 |
| dwn_top | 1,619 | ~12,101 |
| **reduction as % of design** | **3.6%** | **~3%** |

**Eliminating the reduction entirely would save ~3.6%** of a design the encoder dominates.
dse-plan's own framing — *"if it's 3%, this was never an interesting axis"* — is the one that
applies once the denominator is the real design rather than the core. The 40% figure is not
wrong, it is measuring the wrong ratio.

**Recommendation: leave Learnable Reduction deferred.** It is unimplemented upstream, so it is
custom training code plus new RTL plus a Gate 1 re-run, for ~3% of area on the axis that is not
binding. `z` and per-feature narrowing both move far more. Revisit only if a config lands
marginal and the encoder levers are exhausted.

*(Caveat carried from the script: fragments synthesized alone cannot share logic with
neighbours, so both figures are upper bounds on in-context cost. That the sum lands exactly on
108 says there was nothing to share here.)*

### 2026-08-07 — 2c: the grid training notebook

`training/train_grid_kaggle.ipynb`, plus `dse/grid.py --json` so the notebook consumes the grid
rather than restating it. `dwn_jsc_kaggle.ipynb` is untouched — it reproduces the paper's `sm`
and is the Gate 1 reference; it should not grow a loop.

**Binarization is grouped, and the saving is large.** 32 configs need only **9** distinct
`(encoding, z)` binarizations, because 16 of them — the entire width ladder, the `n` sweep and
the layer-count sweep — share `(distributive, 200)` and differ only in model shape. Binarizing
per config would repeat the most expensive setup step 16 times for byte-identical output.

**z=800 does not fit, so there are two data paths.** Measured sizes for the binarized set
(830k samples × 16 features × z, uint8):

| z | 8 | 25 | 50 | 100 | 200 | 400 | **800** |
|---|---|---|---|---|---|---|---|
| GB | 0.11 | 0.33 | 0.66 | 1.33 | 2.66 | 5.31 | **10.62** |

Above 6 GB the notebook binarizes **on the GPU per batch** instead of precomputing. The risk is
obvious — two code paths producing different bits would make a large-z config incomparable to a
small-z one, and the difference would look like a *result*. So the precompute path **asserts the
GPU path reproduces it exactly** on 256 samples before training anything. It is a
speed/memory tradeoff, never a correctness one.

**Resumability is mandatory, not a nicety.** 32 runs will not fit one Kaggle session (9 h cap,
30 h/week quota). The notebook skips any config whose checkpoint already exists, orders configs
cheapest-first within a group so a session that dies late still banks the most models, and has
`ONLY_N` to bound a session deliberately rather than by timeout.

**Schema verified against Phase 1 rather than assumed.** The checkpoint `config` dict carries
exactly the same 13 keys as the Phase 1 checkpoint — none missing, none extra — and includes all
four the exporter actually reads (`layers`, `n`, `num_classes`, `thermometer_bits`). The
`_testvectors.npz` carries `x_binarized` / `x_raw` / `pred`, which is what `gen_vectors.py`
consumes, under the `<slug>_testvectors.npz` name it derives from the checkpoint path.

**Filenames are the contract**: `<slug>_checkpoint.pt`, from `ModelConfig.slug`. `dse/run.py`
resolves by exactly that, so renaming the downloads makes the sweep silently find nothing.

**Deliberately NOT produced: a 166k test-set dump per config.** Gate 1b is Phase 1's exit
condition and was done once for the reference config; Phase 2 takes area and timing from Vivado
reports and accuracy from the checkpoint. Per-config dumps would be ~300 MB that nothing reads.
If a specific sweep config is ever taken to hardware, `dump_testset_kaggle.ipynb` handles it.

⚠️ **Not yet run.** Every cell parses and the schema is verified, but nothing here has executed —
`torch_dwn` has no CPU path, so the notebook cannot be tested off a GPU. Expect the first Kaggle
run to be the real test, and expect training time to be the open question: 32 configs × 32 epochs,
with the larger rungs slower than `sm`.

### 2026-08-07 — 6d: results, frontier, plots — and three bugs it caught first

`dse/report.py` + `dse/plot.py`. Building the reporting layer **before** 2c was the right order:
it found three defects that would each have been baked into 32 training runs' worth of results.

**1. No accuracy in the result schema.** The record had area and timing but nothing to put on
the other axis — a Pareto frontier over (accuracy, area) was literally impossible. Now read from
the checkpoint, so it cannot be attached to the wrong config.

**2. Accuracy units were wrong by 100×.** The checkpoint stores `final_acc` as a *fraction*
(`0.7383614`), while every ledger, table and paper comparison here quotes *percent*. The first
report printed **0.74%**. Converted once at the source, fields renamed `accuracy_pct` /
`accuracy_best_epoch_pct` so no consumer has to guess, and an **assertion** rejects anything
outside [0,1] — if a checkpoint ever stores percent it must fail loudly, not report 7384%.
(`final_acc` is the primary number: the saved weights are the final epoch and `best_acc`
describes weights that were never saved.)

**3. ⚠️ A silent-corruption footgun in `dse/run.py`.** `run_config()` emits RTL from the
checkpoint and uses `cfg` only for naming and area prediction. So `--all --checkpoint <one>`
would have emitted **one model's RTL under all 37 config names** — every row wrong, nothing
failing, and the frontier would have looked plausible. Fixed three ways:

- `ModelConfig.slug` (`n6_z200_distributive_w50`) is the identity of the *trained model*:
  n, z, encoding, widths, and deliberately **no hardware params**, so all five Group B configs
  share one checkpoint instead of demanding five identical training runs.
- `resolve_checkpoint()` looks up `<slug>_checkpoint.pt` per config; no match means skip with
  the expected filename printed, never a wrong build.
- `--checkpoint` with more than one config is now a **hard error**.

This also fixes the naming contract for 2c: the notebook must write `<slug>_checkpoint.pt`.
Phase 1's checkpoint predates the convention and is mapped by an alias rather than renaming a
file every recorded number refers to.

**Latency is a real axis now.** `pareto()` is generic over objectives, and latency is computed
in **nanoseconds** (`cycles / Fmax`), not cycles. Cycles alone cannot rank pipeline variants:
Phase 1 measured 4 stages at 161.0 MHz = **24.84 ns** against 3 stages at 122.9 MHz = 24.4 ns —
a whole cycle apart and effectively the same real latency. Ranking on cycles would call that a
clear win; ranking on time shows it is a wash. Cycles are still reported (brief §6). Under the
2-objective view all five Group B configs tie; the 3-objective view is what separates them, so
the report prints both and says which points only the second view reveals.

**Plots** (`matplotlib==3.10.7` added to requirements — nothing else imports it, so a missing
matplotlib degrades plotting only). Two figures into `build/dse/`: `frontier.png` (accuracy vs
area, Pareto line, device ceiling) and `area_split.png` (core vs encoder stacked, per rung).

Two constraints worth recording because they are not aesthetic:

- **Scatter caps at three categorical colors** (all-pairs CVD separation). The grid's six groups
  fold to ladder / one-factor / group-B — which is also the distinction that carries meaning.
- **The device ceiling must not set the y-scale on `area_split`.** Anchoring the axis to 20,800
  squashed a 1,619-LUT design into a sliver and made the 108-LUT core *invisible* — destroying
  the one thing the figure exists to show. It now scales to the data and states the ceiling in
  words when it is off-scale. Caught only by rendering the PNG and looking at it; the palette
  validator checks color, never layout.

### 2026-08-07 — 2a step 6c: the sweep runner, and 2a is done

`dse/run.py` plus a refactor of `scripts/run_gate1.py` that extracts Gate 1 from `main()` into an
importable `gate1(ckpt, vivado_bin, rtl_dir, work, pipe) -> (ok, info)`. `main()` is now a thin
CLI wrapper over it. `dse/` imports it rather than shelling out and scraping stdout — the reason
`run_xsim()` was exposed back in Phase 1.

**The baseline config went through the whole pipeline and reproduced Phase 1 exactly:**

| Module | LUTs | FF | WNS | Fmax | Phase 1 |
|---|---|---|---|---|---|
| `dwn_core` | 108 | 73 | +3.790 | 161.0 MHz | identical |
| `thermometer_encoder` | 1519 | 0 | +7.013 | 334.8 MHz | identical |
| `dwn_top` | 1619 | 269 | +3.790 | 161.0 MHz | identical |

This is the acceptance test for 2a as a whole. The numbers were reached by a **different path**
than Phase 1 took — RTL emitted into `build/configs/<name>/rtl`, include dirs derived from the
sources by the rewritten `build.tcl`, clock from `cfg.hw.clock_ns`, Gate 1 called as a function
— and landed on the same values. The config-driven flow is equivalent to the flow every Phase 1
number came from, which is what makes sweep points comparable to them.

**Three rules are enforced in code, not left to discipline**, because breaking any of them
corrupts the frontier quietly rather than failing:

1. **Gate 1 gates synthesis.** A config failing Gate 1 is recorded `gate1-failed` and is *not*
   synthesized. Area for unverified RTL describes nothing (dse-plan §1), and the row stays
   visible rather than becoming a config nobody notices is missing.
2. **Failure to build is a RESULT.** Recorded `synth-failed` and kept — that is where the
   congestion wall is (brief risk #2), and it is a thing Study 1 measures.
3. **Core / encoder / top recorded separately, always** (brief §6). A total-only table would
   have hidden the 14× encoder finding entirely.

**Resumability is deliberate.** 34 points is several sittings on one machine (CLAUDE.md), so
results are keyed by config name in `build/dse/results.json`, done configs skip unless
`--force`, and writes go through a temp file — an interruption at point 20 must not cost points
1–19.

**Measured cost: 288 s per config** for three targets, synthesis only. `MINUTES_PER_SYNTH = 12`
in `grid.py` is the *implementation* figure (Phase 1's `--impl` runs), so the 6.8 h budget
estimate holds for post-route and the synthesis-only pass is more like 2.7 h. The sweep should
use `--impl` for anything quotable: post-synthesis timing uses estimated routing and is
systematically optimistic, and Phase 1 already found post-route to be 147.1 vs 161.0 MHz.

### 2026-08-07 — 2b: the area model, recalibrated and honest about what it cannot predict

`dse/area_model.py`. dse-plan §5 assumed the encoder costs "up to 3.2× the core"; measured is
**14.06×**. Filtering 2d with the old number would have underestimated encoder area by **4.4×**
at `sm` alone.

Reproduces every Phase 1 measurement — comparators, core, encoder exact; `dwn_top` and board
+0.5%, which is real rather than rounding (measured 1619 < core+encoder 1627, so Vivado
optimizes slightly across the module boundary).

**Projected size ladder**, and it revises an earlier ledger entry:

| Config | Nodes | Comparators | Core | Encoder | Board | % dev | Fits |
|---|---|---|---|---|---|---|---|
| `sm` 1×50 | 50 | 202 | 108 | 1,519 | 2,066 | 9.9% | ✅ |
| `md` 1×360 | 360 | 1,454 | 728 | 10,934 | 12,101 | **58.2%** | ✅ |
| `lg` 1×2400 | 2400 | 3,200 (sat.) | 4,808 | 24,063 | 29,310 | 140.9% | ❌ |

⚠️ **Correction: `md` was previously recorded as "≲16,000 encoder LUTs, ~80% — tight".** That
used the *upper bound* of 2,160 comparators (every wiring slot distinct). Applying the measured
67% selection ratio gives 1,454 comparators and **58.2%** — `md` is a comfortable rung, not a
marginal one, which makes it the natural centre of the ladder rather than its edge.

`lg` confirms as not fitting, and starkly: its **encoder alone (24,063) exceeds the whole part**,
while its core (4,808) would fit easily. Note also that `lg`'s encoder/core ratio is **5.0×**,
not 14× — further confirmation that 14× is an artifact of `sm`'s unusually tiny core, not a
constant to extrapolate.

**What this model CANNOT do, discovered while building the grid.** The 67% selection ratio is
**not a collision statistic**. If the mapping picked slots uniformly from the 3,200 available
bits, occupancy would predict ~286 distinct comparators, not 202. The 84-comparator gap is
**learned concentration** — four features carry 153 of 202. How hard the mapping concentrates
depends on how many thresholds each feature has (**z**) and how many slots each node has (**n**),
so a ratio measured at one `(n=6, z=200)` point cannot be transported to other values of either.

Consequence, and it is load-bearing for 2d: **`z` is simultaneously the axis the sweep most wants
to characterize and the one this model is least able to predict.** `is_extrapolated(n, z)` marks
those estimates, and `dse/grid.py` **refuses to skip a config on an extrapolated overshoot** — it
skips only when the prediction is at the calibrated point. Otherwise the filter would silently
discard exactly the configs Study 1 exists to measure.

The softer half is the **reduction term** (~58 LUTs, inferred by subtracting 50 nodes from the
108-LUT core). Vivado inlined `popcount`/`argmax`, so that split is arithmetic, not observation.
The standalone-synthesis open question stands.

### 2026-08-07 — 2a step 6b: the sweep grid

`dse/grid.py` — 37 configs: 8 ladder rungs, 24 one-factor points on two mid-ladder rungs
(z, encoding, n, layer count), 5 Group B variants. Group B rides on the baseline rung's trained
model — same `ModelConfig`, different `HardwareConfig`, no retraining. That is the payoff for
splitting the two objects in step 1.

**`z` gets six values (8, 25, 50, 100, 400, 800), spanning both regimes on purpose.** They
answer different questions, and the area model predicts neither:

| | z=8 | z=25 | z=50 | z=100 | z=400 | z=800 |
|---|---|---|---|---|---|---|
| `1×200` | 8.7% | 18.5% | 33.0% | 33.3% | 33.3% | 33.3% |
| `1×360` | 10.2% | 20.1% | 34.5% | 58.2% | 58.2% | 58.2% |

Below the binding point (`16z < slots × ratio`) encoder area is set by `z` directly. Above it
the model says area is **flat**, because comparators become slot-limited. **Either outcome is a
result:** if flat holds, accuracy above z≈50 is free, which would be the sweep's headline; if it
does not, the 67% selection ratio is climbing toward 1.0 as collisions get rarer, and the area
model needs a z-dependent ratio. The two rungs cross the transition at different z, which is
what makes the flat region testable rather than assumed.

- **The ladder brackets the wall rather than stopping short of it.** `1×500` fits at 80.0%,
  `1×600` fails at 95.6%, `1×800`/`1×1200` fail with the encoder saturated at 3,200. A config
  that does not fit is a data point marking the frontier's edge (brief risk #2), so it is
  reported, never hidden.
- **`tau` interpolates the paper's schedule in log-width**, not copied from `sm`. Getting it
  wrong fails silently — it just trains a worse model, and the point then reports an accuracy
  that says more about tau than about the architecture.
- **Budget: 34 synthesis points ≈ 6.8 h**, against dse-plan §6's 40–70 runs / 15–25 h on one
  machine, plus 32 training runs (Kaggle, step 2c). Still ~2× headroom. Remaining places to
  spend it, in rough order of value: **finer ladder spacing near the wall** (500→600 is a single
  100-node jump and the fit boundary is inside it), a **third OFAT rung**, and `n`×`z`
  interaction points — since those are the two axes the area model cannot extrapolate over,
  and it currently assumes they are independent.

### 2026-08-07 — 2a step 5: the restructure

The move the handoff specified, now done. `emit_core.py` and `emit_encoder.py` are in `rtlgen/`
(via `git mv`, so history follows); `extract.py` stays in `exporter/` per brief §11's split, as
do the `analyze_*`/`experiment_*` scripts. **`rtl/gen/` is deleted, not gitignored** — output
goes to `build/rtl` (default) or `build/configs/<name>/rtl` (swept), both under the
already-ignored `build/`. Exactly as the handoff put it: "it doesn't get gitignored, it stops
existing."

**`scripts/build.tcl` was the only non-mechanical part.** It hardcoded
`-include_dirs rtl/gen`. Include paths are now **derived from the source files actually read**:

```tcl
foreach f $sources { ... lappend inc_dirs [file dirname $f] }
```

A fixed path there would have compiled a swept config against some *other* config's
`DWN_TOP_LATENCY`, silently, with a wrong number as the only symptom. Deriving it means the
headers cannot come from anywhere but the RTL that was read. A second `--include-dir` argument
would have been the obvious alternative and the wrong one — two ways to specify one value.

⚠️ **One real failure, worth recording rather than just fixing.** `scripts/verify_phase1.py`
hardcoded `os.path.join(REPO, 'rtl', 'gen', 'dwn_core.v')` and crashed. Two lessons:

1. **The regression harness was the one file that broke.** Everything it guards survived the
   path change; the guard itself did not.
2. **A path grep is not sufficient.** The components were separate strings, so searching for
   `rtl/gen` could not match `os.path.join(REPO, 'rtl', 'gen', ...)`. Found only by running it.

Fixed by importing `DEFAULT_RTL_DIR` from `run_gate1` instead of respelling the path, so it now
fails loudly on the next move rather than silently reading somewhere else.

**Verified: `verify_phase1.py` 12/12** after the restructure — Gate 1 1504/1518, harness unit
tests, and all six area numbers unchanged.

### 2026-08-07 — 2a step 3: pipeline depth is an argument, not a constant

`emit_core.py` takes `--pipe-lut/--pipe-pop/--pipe-out`, `emit_encoder.py` takes `--pipe-enc`.
`PIPE_LUT/POP/OUT` survive as the *defaults*, which is what keeps `rtlgen/config.py`'s drift
guard working. Group B can now sweep depth without a code edit (dse-plan §1).

**A real trap found while doing it.** `dwn_top` instantiates the core as
`dwn_core #(.PIPE_LUT(PIPE_LUT), ...)`, so **dwn_top's own parameter defaults override the
core's**. Emitting `PIPE_OUT=0` from `emit_core.py` while `emit_encoder.py` still defaulted
`dwn_top`'s to 1 would have silently put the register back — the design would have looked
3-stage in one file and 4-stage in another, and only a timing number would have hinted at it.

Fixed by making the depth flow through **one** source rather than two: `emit_core.py` now writes
`DWN_CORE_PIPE_LUT/POP/OUT` into `dwn_core_params.vh` alongside the latency, and
`emit_encoder.py` reads them. The core's stages are deliberately **not** flags on
`emit_encoder.py` — two ways to specify one value is how they end up contradicting. This extends
the mechanism that file already existed for ("so the two emitters cannot disagree about pipeline
depth").

**Verified:**

- defaults: only `dwn_core_params.vh` changes, gaining exactly the 3 new defines at 1/1/1.
  `dwn_core.v`, `dwn_top.v`, `thermometer_encoder.v`, `dwn_top_params.vh` all byte-identical
  to committed Phase 1.
- non-default (`--pipe-out 0`): propagates to `dwn_core.v` **and** `dwn_top.v`, both reading
  `PIPE_OUT = 0`, with `DWN_CORE_LATENCY 2` / `DWN_TOP_LATENCY 3`. That 3 matches the Phase 1
  pipeline sweep's recorded 3-stage / no-OUT-reg point.
- **Gate 1: 1504/1504 core, 1518/1518 top, PASS.** Config self-test still green.

⚠️ **The non-default depth is verified textually, not behaviourally.** `run_gate1.py` still
hardcodes `rtl/gen`, so a 3-stage variant cannot be simulated yet — the golden model would need
to be told the new latency. **That is step 4**, and until it lands, no non-default pipeline
config should be trusted past "the text looks right". Do not synthesize a swept depth and quote
its Fmax before Gate 1 can run on it.

### 2026-08-04 — Encoder sharing: measured, and it is a dead end ❌

`exporter/analyze_encoder_sharing.py`. Phase 1 left one encoder question genuinely open: Vivado
is near-optimal *inside* a comparator (7.5 LUTs ≈ W/2), but nothing has ever been shared
*between* the comparators of one feature, which all compare the same value against sorted
constants. Mecik & Kumm's FloPoCo encoder presumably did exactly that, and at `md`/`lg` sizes
the encoder is what decides whether a config fits — so this is worth more than curiosity.

**Two ideas, both dead, and the reasoning is recorded so nobody re-derives them.**

**1. Binary search for the bucket index — arithmetically worse, not better.** The appeal is
obvious: 46 sorted thresholds is one question ("which of 47 buckets?"), seemingly worth
`log2 k ≈ 6` comparisons rather than 46. It does not work, because each level must *select*
which threshold to compare against next, and that is a multiplexer over 16-bit constants whose
input count doubles per level. For k=46 the level-5 mux alone is 32-to-1 over 16 bits, on the
order of 176 LUTs — more than the 345 the 46 direct comparators cost in total. Instantiating
the whole tree instead needs `1+2+4+...+32 = 63` comparators, worse than the 46 you started
with. **The mux cost is fundamental to any combinational bucket decode.**

**2. Shared high-bit prefixes — only pays if thresholds cluster, and they do not.** Measured
across every split point, with a cost model calibrated on the measured 1519/202:

| Feature | k | direct | best split | groups | shared bound |
|---|---|---|---|---|---|
| 14 (`mass_mmdt`) | 46 | 346 | 5/11 | 8 | 321 (−7%) |
| 15 (`multiplicity`) | 39 | 293 | 4/12 | 5 | 278 (−5%) |
| 0 (`zlogz`) | 37 | 278 | 4/12 | 5 | 264 (−5%) |
| 3 (`c1_b2_mmdt`) | 31 | 233 | 5/11 | 4 | 210 (−10%) |
| **total** | **202** | **1519** | | | **1441 (−5%)** |

**−5%, and that is an optimistic ceiling** — the model ignores routing and the combine's
fan-in, and was deliberately written to flatter the shared scheme. The cause is the encoding
itself: `DistributiveThermometer` places thresholds at **quantiles**, so they spread across the
data distribution and their high bits rarely agree. 46 thresholds land in 8 distinct high-bit
groups; 37 land in 5. There is nothing to share.

**Conclusion: the encoder is at its floor for this scheme.** Phase 1's *"most of the 14× is
real, not naive construction"* is now measured rather than asserted. The three levers that
remain, in order of value:

1. **`z`** — sets how many thresholds exist at all, and encoder cost saturates at
   `features × z`. A **config** change, no RTL. Still the most valuable unswept axis.
2. **Per-feature narrowing** — already measured at −17.1%, still not worth a spec change at
   `sm`, still possibly decisive at `md`/`lg`.
3. **A different encoding scheme** — `Thermometer` (evenly spaced) would cluster in high bits
   far better than `DistributiveThermometer` does, so sharing might pay there. But encoding
   scheme is a Group A axis that changes accuracy, so it is a sweep point, not an optimization.

⚠️ **Retraction, kept visible:** an earlier estimate in conversation put bucket-decode at ~2.3×
(1519 → ~670 LUTs). That was wrong — it counted `log2 k` comparators and ignored the
multiplexer cost entirely. The measured ceiling is −5%.

---

## Numbers worth quoting

*(empty — Phase 1's are in `docs/phase1-ledger.md`)*

---

## Open questions

| | |
|---|---|
| ~~Is the reduction really ~54% of the core?~~ | ✅ **Closed 2026-08-07. Yes, exactly: 50 + 58 = 108.** And one node is exactly one LUT6, measured. **But the ratio that matters is 3.6% of the whole design**, not 54% of the core — dse-plan §3's 40% bar predates the 14× encoder finding. Learnable Reduction stays deferred; see the log entry. |
| **Does `md` actually fit?** | The bound says ~80% of the part, which is tight enough that routing (not LUT count) may decide it. A failure to route is a data point, not a mistake (brief §12 risk #2). |
| **How many thresholds does a bigger model really select?** | `sm` chose 202 of 300 slots (67% unique). If that ratio holds, `md`/`lg` encoder estimates drop. Only training says. |
| **Per-feature comparator narrowing** | Measured −17.1% at `sm` and not adopted — 260 LUTs did not justify a spec change. **At `md`/`lg` it may decide whether a config fits at all.** Revisit when a config is marginal. Now the *largest* remaining RTL-side lever, since sharing is dead (below). |
| ~~Can comparators share logic across thresholds of a feature?~~ | ❌ **Closed 2026-08-04, negative.** Binary-search decode is arithmetically worse (mux cost doubles per level); shared high-bit prefixes bound at **−5%** because quantile-spaced thresholds do not cluster. See the log entry. `exporter/analyze_encoder_sharing.py` re-runs it for any checkpoint — worth repeating if a config ever uses evenly-spaced `Thermometer`, where clustering should be much stronger. |
| **`z` is the axis nobody has swept** | The paper fixes z=200 for every JSC config and never reports its cost. `z` sets the saturation ceiling on encoder area, which dominates. Accuracy vs area vs `z`, on a part where it binds, is unmeasured by anyone — probably the single most publishable axis here. |
| **3-stage pipeline** | Closes 100 MHz post-synthesis but was never re-verified post-route. One cheap Group B point. |
| **FTDI D2XX for >5 Mbaud** | Optional; the VCP driver, not the design, is the wall. Two more I/O-wall points if Phase 3 leaves time. |
| **MNIST port** | Stretch, after Phase 3. Scoped in `docs/phase1-ledger.md` — three harness breakages, ~1–2 days, and a *higher* accuracy number that means less, not more. |

---

## Pointers

- `docs/dse-plan.md` — what gets swept, the knob groups, the strategy
- `docs/phase2-handoff.md` — machine acceptance test + the `rtlgen/` restructure
- `docs/phase1-report.md` — what already works, and how to reproduce it
- `docs/phase1-ledger.md` — Phase 1's raw log and its open questions
- `docs/project-brief.md` §10 — the DSE study as originally specified
